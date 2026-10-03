"""
PRISE — Module M1: Ingestion, Normalization, Artifact Gate, Censoring & QC
=========================================================================

M1 takes a *raw* canonical `AggregationSeries` (from the CPAD ETL or a user
upload) and returns it augmented with a triage verdict, so every downstream
module sees a uniform, honestly-flagged object. M1 is FAITHFUL — it inspects,
normalises, and tags. It does NOT fit models, infer mechanism, or score; those
belong to M2+.

Implements PRISE_DESIGN.md §3 Module 1:
  * artifact gate     -> fittability_class (fittable | suspect | unfittable) + reasons
  * censoring          -> censoring_class (none | left | right | left_right | interval)
  * normalization      -> baseline subtraction; min-max ONLY when a plateau is
                          validated (never plateau-manufacturing on right-censored
                          curves); records signal_basis + normalization_mode
  * recommended_handling -> monotonic_fit | nonmonotonic_fit | descriptive_only | reject
  * plate QC           -> N/A for digitized literature curves (no co-located controls)

All heuristics are scale-robust (CPAD intensities are digitized a.u. with unknown
normalisation): decisions use the curve's own dynamic range and point-to-point
noise, never absolute thresholds.

Usage:
    python engine/m1_ingest.py [--input data/processed/curves.jsonl]
                               [--output data/processed/curves_triaged.jsonl]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Optional

# Module identity — consumed by M7's engine_build_id (the comparability key, §7).
# Purely a provenance tag; bumping it makes downstream builds non-comparable.
M1_VERSION = "m1-triage-1.0"

# Service-C-CALIBRATED threshold source (the governed apply artifact, §7/§8).
# LAG_SLOPE_FRAC is now sourced from `calibrated_thresholds` (thresholds-1.0) —
# it is NO LONGER a first-pass hand value. The previous 0.30 is retained there as
# `first_pass`; the applied 0.45 is Service C's held-out recommendation (out-of-
# sample AUC 1.0). The constant NAME is unchanged (consumers/tests reference it);
# only its VALUE is now governed. See engine/calibrated_thresholds.py.
from calibrated_thresholds import calibrated_value as _cal, THRESHOLDS_VERSION

# --------------------------- tunable thresholds ---------------------------- #
MIN_POINTS_ANY = 4        # below this: cannot fit anything
MIN_POINTS_FIT = 6        # below this (but >=ANY): sparse -> suspect
SNR_FLAT = 3.0            # rise must exceed SNR_FLAT * noise to be "real signal"
# Service-C-calibrated (thresholds-1.0): start slope this fraction of peak slope
# -> no lag (left). first_pass was 0.30; calibrated (applied) is 0.45.
LAG_SLOPE_FRAC = _cal("M1_LAG_SLOPE_FRAC", default=0.30)
END_SLOPE_FRAC = 0.15         # end slope this fraction of peak slope -> right-censored
POSTPEAK_DROP_FRAC = 0.20     # fall from peak this fraction of range -> non-monotonic
PEAK_NEAR_END_FRAC = 0.88     # peak beyond this position -> "still rising" candidate
PLATEAU_TAIL_FRAC = 0.20      # fraction of points used as the plateau/tail window
IRREGULAR_CV = 1.0            # CV of time spacing above this -> irregular sampling


# ------------------------------ small math --------------------------------- #
def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def _mad(xs: list[float]) -> float:
    """Median absolute deviation of consecutive differences (noise proxy)."""
    if len(xs) < 2:
        return 0.0
    diffs = [abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
    return _median(diffs)


def _slope(xs: list[float], ys: list[float]) -> float:
    """Ordinary least-squares slope; 0 if degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return sxy / sxx


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


# ------------------------------ feature pass ------------------------------- #
def curve_features(x: list[float], y: list[float]) -> dict:
    """Scale-robust descriptors used by the artifact/censoring logic."""
    n = len(x)
    k = max(1, min(3, n // 5 or 1))
    ymin, ymax = min(y), max(y)
    i_max = max(range(n), key=lambda i: y[i])
    rng = ymax - ymin
    noise = _mad(y)
    y_start = _median(y[:k])
    y_end = _median(y[-k:])

    # local slopes between consecutive points; robust max via 90th percentile
    local = []
    for i in range(n - 1):
        dt = x[i + 1] - x[i]
        if dt > 0:
            local.append((y[i + 1] - y[i]) / dt)
    max_slope = _percentile([s for s in local if s > 0], 0.9) if local else 0.0

    # start/end slopes over the leading/trailing windows
    tail = max(2, int(round(PLATEAU_TAIL_FRAC * n)))
    start_slope = _slope(x[:tail], y[:tail])
    end_slope = _slope(x[-tail:], y[-tail:])

    # time-spacing regularity
    dts = [x[i + 1] - x[i] for i in range(n - 1) if x[i + 1] > x[i]]
    if dts and _median(dts) > 0:
        cv_dt = (_mad(dts)) / _median(dts)
    else:
        cv_dt = 0.0

    return {
        "n_points": n, "ymin": ymin, "ymax": ymax, "i_max": i_max,
        "i_max_frac": i_max / (n - 1) if n > 1 else 0.0,
        "range": rng, "noise": noise, "y_start": y_start, "y_end": y_end,
        "net_rise": y_end - y_start, "post_peak_drop": ymax - y_end,
        "max_slope": max_slope, "start_slope": start_slope,
        "end_slope": end_slope, "cv_dt": cv_dt,
    }


# ------------------------------ triage core -------------------------------- #
def triage_features(f: dict) -> dict:
    """Map features -> verdict (no model fitting)."""
    n = f["n_points"]
    rng = f["range"]
    noise = f["noise"]
    reasons: list[str] = []

    # 1) hard reject: too few points
    if n < MIN_POINTS_ANY:
        return {
            "fittability_class": "unfittable", "censoring_class": "none",
            "recommended_handling": "reject", "plateau_reached": False,
            "reasons": ["too_few_points"], "flat_no_signal": False,
            "non_monotonic": False,
        }

    # 2) flat / no detectable signal (a valid DESCRIPTIVE outcome, not a reject)
    flat = rng <= SNR_FLAT * noise or f["max_slope"] <= 0
    if flat:
        return {
            "fittability_class": "fittable", "censoring_class": "none",
            "recommended_handling": "descriptive_only", "plateau_reached": True,
            "reasons": ["flat_no_detectable_aggregation"], "flat_no_signal": True,
            "non_monotonic": False,
        }

    # 3) shape / censoring on a curve with real signal
    non_monotonic = (
        f["post_peak_drop"] > POSTPEAK_DROP_FRAC * rng
        and f["i_max_frac"] < PEAK_NEAR_END_FRAC
    )
    if non_monotonic:
        reasons.append("non_monotonic_postpeak_decline")

    # Left-censoring proxy: NO lag phase — the curve is already in steep rise
    # at t0 (start slope a large fraction of peak slope). NOTE this can also be
    # genuine downhill/no-lag kinetics; M5 disambiguates. Flagged, not asserted.
    left = f["max_slope"] > 0 and f["start_slope"] > LAG_SLOPE_FRAC * f["max_slope"]
    if left:
        reasons.append("no_lag_phase(possible_left_censoring_or_downhill)")

    still_rising = (
        f["i_max_frac"] >= PEAK_NEAR_END_FRAC
        and f["max_slope"] > 0
        and f["end_slope"] > END_SLOPE_FRAC * f["max_slope"]
    )
    right = still_rising and not non_monotonic
    if right:
        reasons.append("right_censored_no_plateau")

    irregular = f["cv_dt"] > IRREGULAR_CV
    if irregular:
        reasons.append("irregular_sampling")

    if left and right:
        censoring = "left_right"
    elif left:
        censoring = "left"
    elif right:
        censoring = "right"
    elif irregular:
        censoring = "interval"
    else:
        censoring = "none"

    plateau_reached = not right and not non_monotonic

    # 4) fittability + handling
    if non_monotonic:
        cls, handling = "suspect", "nonmonotonic_fit"
    elif n < MIN_POINTS_FIT:
        cls, handling = "suspect", "monotonic_fit"
        reasons.append("sparse_lt6_points")
    else:
        cls, handling = "fittable", "monotonic_fit"

    return {
        "fittability_class": cls, "censoring_class": censoring,
        "recommended_handling": handling, "plateau_reached": plateau_reached,
        "reasons": reasons or ["clean"], "flat_no_signal": False,
        "non_monotonic": non_monotonic,
    }


def normalize(x: list[float], y: list[float], verdict: dict, f: dict) -> dict:
    """Baseline-subtract; min-max ONLY when a plateau is validated.

    Never min-max a right-censored / non-plateaued curve (that manufactures a
    plateau and corrupts t50/γ — PRISE_DESIGN.md §3-M1)."""
    baseline = f["ymin"]
    y_bs = [v - baseline for v in y]
    if verdict["plateau_reached"] and f["range"] > 0 and not verdict["flat_no_signal"]:
        span = f["ymax"] - baseline
        y_norm = [v / span for v in y_bs] if span > 0 else y_bs
        return {
            "signal_basis": "normalized",
            "normalization_mode": "min_max(plateau_validated)",
            "baseline": baseline, "y_processed": y_norm,
            "min_max_suppressed": False,
        }
    return {
        "signal_basis": "baseline_subtracted",
        "normalization_mode": "baseline_only",
        "baseline": baseline, "y_processed": y_bs,
        "min_max_suppressed": True,  # plateau-manufacturing avoided
    }


def triage_series(series: dict) -> dict:
    """Augment one AggregationSeries dict with an `m1` block + updated fields."""
    x = series.get("x_hours") or []
    y = series.get("y_intensity") or []
    if len(x) != len(y) or len(x) < 2:
        series["m1"] = {
            "fittability_class": "unfittable", "censoring_class": "none",
            "recommended_handling": "reject", "reasons": ["malformed_or_empty"],
            "plate_qc": "not_applicable",
        }
        series["fittability_class"] = "unfittable"
        series["censoring_class"] = "none"
        return series

    f = curve_features(x, y)
    v = triage_features(f)
    norm = normalize(x, y, v, f)

    m1 = {
        **v,
        "features": {k: (round(val, 6) if isinstance(val, float) else val)
                     for k, val in f.items()},
        "normalization_mode": norm["normalization_mode"],
        "signal_basis": norm["signal_basis"],
        "baseline": norm["baseline"],
        "min_max_suppressed": norm["min_max_suppressed"],
        "y_processed": [round(v_, 6) for v_ in norm["y_processed"]],
        "digitization_uncertainty": bool(series.get("digitization_uncertainty", True)),
        # CPAD curves are digitized from single literature figures with no
        # co-located buffer-only / positive controls -> plate QC inapplicable.
        "plate_qc": "not_applicable: digitized literature curve (no co-located controls)",
    }
    series["m1"] = m1
    # surface the top-level fields the rest of the engine reads
    series["fittability_class"] = m1["fittability_class"]
    series["censoring_class"] = m1["censoring_class"]
    series["signal_basis"] = m1["signal_basis"]
    series["normalization_mode"] = m1["normalization_mode"]
    return series


# --------------------------------- I/O ------------------------------------- #
def load_curves(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def process_file(input_path: Path, output_path: Path) -> dict:
    fit_counts: Counter = Counter()
    cens_counts: Counter = Counter()
    handling_counts: Counter = Counter()
    n = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for s in load_curves(input_path):
            s = triage_series(s)
            n += 1
            fit_counts[s["m1"]["fittability_class"]] += 1
            cens_counts[s["m1"]["censoring_class"]] += 1
            handling_counts[s["m1"]["recommended_handling"]] += 1
            out.write(json.dumps(s) + "\n")
    return {
        "total": n,
        "fittability_class": dict(fit_counts),
        "censoring_class": dict(cens_counts),
        "recommended_handling": dict(handling_counts),
    }


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="PRISE M1 ingestion/triage")
    ap.add_argument("--input", type=Path,
                    default=root / "data" / "processed" / "curves.jsonl")
    ap.add_argument("--output", type=Path,
                    default=root / "data" / "processed" / "curves_triaged.jsonl")
    args = ap.parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input} (run the ETL first)")
    print(f"[M1] triaging {args.input.name} ...")
    summary = process_file(args.input, args.output)
    print(f"[M1] wrote {args.output}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
