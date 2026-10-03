"""
Tests for M1 ingestion/triage.

Synthetic curves with known shapes exercise each verdict; a smoke test runs M1
over the real ETL output if present.

    python engine/test_m1.py        # standalone PASS/FAIL
    pytest engine/test_m1.py        # if pytest installed
"""
from __future__ import annotations

import math
from pathlib import Path

from m1_ingest import process_file, triage_series

ROOT = Path(__file__).resolve().parent.parent
CURVES = ROOT / "data" / "processed" / "curves.jsonl"


# ------------------------------ generators --------------------------------- #
def _series(x, y, sid="syn"):
    return {"series_id": sid, "x_hours": list(map(float, x)),
            "y_intensity": list(map(float, y)), "digitization_uncertainty": True}


def logistic(t, t0, k):
    return 1.0 / (1.0 + math.exp(-k * (t - t0)))


def clean_sigmoid():           # lag -> rise -> plateau, regular sampling
    x = [i for i in range(41)]
    return _series(x, [logistic(t, 20, 0.4) for t in x], "clean")


def right_censored():          # still rising at the end, no plateau
    x = [i * 0.5 for i in range(31)]            # 0..15 h
    return _series(x, [logistic(t, 14, 0.5) for t in x], "right")


def no_lag_saturating():       # starts at max slope (no lag) -> plateau
    x = [i for i in range(41)]
    return _series(x, [1 - math.exp(-0.15 * t) for t in x], "nolag")


def biphasic():                # rise to a peak then decline (settling)
    x = [i for i in range(41)]
    return _series(x, [math.exp(-((t - 15) / 6.0) ** 2) for t in x], "biphasic")


def flat_noise():              # no real signal
    x = [i for i in range(41)]
    return _series(x, [0.5 + 0.002 * (1 if i % 2 else -1) for i in x], "flat")


def too_few():
    return _series([0, 1, 2], [0.0, 0.5, 1.0], "few")


# -------------------------------- tests ------------------------------------ #
def test_clean_sigmoid():
    m = triage_series(clean_sigmoid())["m1"]
    assert m["fittability_class"] == "fittable", m
    assert m["censoring_class"] == "none", m
    assert m["plateau_reached"] is True
    assert m["recommended_handling"] == "monotonic_fit"
    assert m["signal_basis"] == "normalized"
    assert m["min_max_suppressed"] is False
    assert 0.0 <= min(m["y_processed"]) and max(m["y_processed"]) <= 1.0001


def test_right_censored_blocks_minmax():
    m = triage_series(right_censored())["m1"]
    assert "right" in m["censoring_class"], m
    assert m["plateau_reached"] is False
    assert m["min_max_suppressed"] is True            # no plateau-manufacturing
    assert m["signal_basis"] == "baseline_subtracted"


def test_no_lag_left_flag():
    m = triage_series(no_lag_saturating())["m1"]
    assert "left" in m["censoring_class"], m
    assert any("no_lag_phase" in r for r in m["reasons"])


def test_biphasic_is_suspect_nonmonotonic():
    m = triage_series(biphasic())["m1"]
    assert m["non_monotonic"] is True
    assert m["fittability_class"] == "suspect"
    assert m["recommended_handling"] == "nonmonotonic_fit"


def test_flat_is_descriptive_only():
    m = triage_series(flat_noise())["m1"]
    assert m["flat_no_signal"] is True
    assert m["recommended_handling"] == "descriptive_only"
    assert m["fittability_class"] == "fittable"        # flat is a valid outcome


def test_too_few_points_rejected():
    m = triage_series(too_few())["m1"]
    assert m["fittability_class"] == "unfittable"
    assert m["recommended_handling"] == "reject"


def test_top_level_fields_surfaced():
    s = triage_series(clean_sigmoid())
    for key in ("fittability_class", "censoring_class", "signal_basis",
                "normalization_mode"):
        assert s[key] == s["m1"][key]


def test_real_data_smoke():
    if not CURVES.exists():
        raise FileNotFoundError(CURVES)
    out = ROOT / "data" / "processed" / "curves_triaged.jsonl"
    summary = process_file(CURVES, out)
    assert summary["total"] == 1654
    # classes partition the corpus
    assert sum(summary["fittability_class"].values()) == 1654
    # the bulk of curves should be analysable (fittable or suspect, not rejected)
    rejected = summary["fittability_class"].get("unfittable", 0)
    assert rejected < 0.5 * summary["total"], summary["fittability_class"]


# ------------------------------- runner ------------------------------------ #
def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except FileNotFoundError as e:
            print(f"SKIP  {t.__name__}: {e} (run ETL first)")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
