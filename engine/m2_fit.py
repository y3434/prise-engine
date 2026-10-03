"""
PRISE — Module M2: Model Fitting Bank
=====================================

Fits the licensed candidate models to each M1-triaged curve and reports per-fit
diagnostics (SSE, R², AIC/AICc/BIC, parameter SEs, convergence). M2 fits and
reports only — it does NOT select a model (M3) or claim mechanism (M5).

Three fitting entry points (one per data axis):
  * fit_curve(series)           — KINETIC: descriptive + biphasic + closed-form
                                  mechanistic (Finke–Watzky) on a single curve
  * fit_dose_response(d, r)     — DOSE: LNT / threshold / Hill / Brain–Cousens /
                                  scaling-law on response-vs-concentration
  * mechanistic.fit_mechanistic — TIER B: Knowles/Cohen ODE family (separate
                                  module; applied to concentration-series / on demand)

Model definitions live in `models.py`; the ODE mechanistic family in
`mechanistic.py`. All outputs are JSON-serialisable (web-API ready).

Usage:
    python engine/m2_fit.py [--input data/processed/curves_triaged.jsonl]
                            [--output data/processed/fits.jsonl] [--limit N]
    python engine/m2_fit.py --dose   # fit dose-response to R-rows by protein
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

import fit_provenance as fp

# The GOVERNED start policy (bootstrap_policy.py). READ, not hard-coded, so the
# previous name-keyed-only behaviour stays one value away and cannot drift from
# the record describing it.
try:                                                   # pragma: no cover - import glue
    from bootstrap_policy import applied_policy as _start_policy
except Exception:                                      # pragma: no cover - fallback
    def _start_policy(name, default=None):
        return {"M2_START_TOPUP": "spread_within_bounds"}.get(name, default)
from models import (
    DOSE_RESPONSE,
    KINETIC_BIPHASIC,
    KINETIC_DESCRIPTIVE,
    KINETIC_MECHANISTIC,
    REGISTRY,
    _time_at_fraction,
)

warnings.simplefilter("ignore", OptimizeWarning)
warnings.simplefilter("ignore", RuntimeWarning)

# Module identity — consumed by M7's engine_build_id (the comparability key, §7).
# Purely a provenance tag; bumping it makes downstream builds non-comparable.
#
# NOT BUMPED for the §2.3 FitProvenance record, deliberately, and here is the
# reasoning for whoever later wonders why this axis did not move when the M2
# output schema gained keys. §7's operative sentence is "Comparability is gated
# on equal `engine_build_id`" — comparability of RESULTS, explicitly "not on
# eyeballing independent version integers". The provenance record adds no
# optimizer call, no RNG and no reordering; it only writes down values the
# fitting loop already computed, so every fitted number is byte-identical before
# and after (proved, not asserted: see `test_artifact_invariance.py`). Bumping
# would therefore declare a non-comparability that does not exist and force a
# rebuild of every descendant for nothing. A change to any emitted VALUE does
# bump this. `harness_version` likewise does not move: recording what the
# inference recipe already does is not a change to the recipe.
M2_VERSION = "m2-fitbank-1.0"

# The §2.3 recipe this module fits under (the constant half of FitProvenance,
# emitted once into data/processed/fit_recipes.json and referenced by id).
M2_RECIPE = "m2/trf-bounded/det-starts-1.0"

CANDIDATES = {
    "monotonic_fit": KINETIC_DESCRIPTIVE + KINETIC_MECHANISTIC,
    "nonmonotonic_fit": KINETIC_BIPHASIC,
    "descriptive_only": ["lnt"],
    "reject": [],
}


# ------------------------------ diagnostics -------------------------------- #
def _metrics(y, yhat, p):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    n = len(y)
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    sigma2 = max(sse / n, 1e-12)
    loglik = -0.5 * n * (math.log(2 * math.pi) + math.log(sigma2) + 1.0)
    aic = 2 * p - 2 * loglik
    aicc = aic + (2 * p * (p + 1) / (n - p - 1)) if (n - p - 1) > 0 else float("inf")
    return {"sse": sse, "r2": 1 - sse / sst, "loglik": loglik,
            "aic": aic, "aicc": aicc, "bic": p * math.log(n) - 2 * loglik}


# ------------------------------- fitting ----------------------------------- #

def _spread_starts(base, lo, hi, n):
    """`n` deterministic, in-bounds additional starts for a model whose parameter
    names the name-keyed rules do not recognise.

    Each start places every parameter at a fixed FRACTION of its own bounded
    range, so the probes are spread across the feasible box rather than clustered
    around one guess -- which is the whole point of a multi-start: to find out
    whether the reported optimum is the basin the data actually prefer, or the one
    the initial guess happened to fall into.

    No RNG: the fractions are a frozen sequence, so the ledger is reproducible and
    §7's gate reads the same spread on every run. Where a bound is non-finite or
    degenerate the parameter falls back to a multiplicative perturbation of its
    own base value, which keeps the start finite without inventing a scale.
    """
    fractions = (0.25, 0.75, 0.5, 0.1, 0.9)
    out = []
    for f in fractions[:max(0, int(n))]:
        g = []
        for i, b in enumerate(base):
            l, h = lo[i], hi[i]
            if math.isfinite(l) and math.isfinite(h) and h > l:
                g.append(l + f * (h - l))
            else:                       # unbounded/degenerate: scale, never guess
                g.append(b * (0.3 if f < 0.5 else 3.0) if b else f)
        out.append(g)
    return out


def fit_one(x, y, model_name: str, n_starts: int = 4) -> dict:
    """Bounded multi-start least-squares fit of one registry model. Never raises.

    Emits a §2.3 `FitProvenance` block under `provenance`: a reference to the
    frozen recipe plus the per-start BASIN LEDGER that §7's basin-stability gate
    needs ("anchors freezable only if basin-stable; knife-edge fits are
    refused"). Before this, the loop kept only the best-SSE start and `continue`d
    past failures without recording them, so the spread that gate needs was
    destroyed at fit time and the gate was *unevaluable* rather than merely
    unenforced (§10.12, commit a6c9f88).

    VALUE-PRESERVING BY CONSTRUCTION. The ledger is assembled from quantities
    this loop already computed — `curve_fit` still runs exactly once per start
    and `_metrics` exactly once per converged start — and the winner is still
    chosen by the same strict `<` scan in construction order, so ties still keep
    the earlier start. Nothing here can move a fitted number.

    MULTI-START, now genuine for every model (fit-start-policy-1.0). The extra
    starts used to be keyed on PARAMETER NAMES (`t0`, `k`), so `logistic` /
    `gompertz` / `richards` got 4, `exponential` got 2, and the other 9 registry
    models — including `finke_watzky` (rates named k1/k2) and every dose-response
    model — got exactly ONE. A single-start fit cannot be shown to be anything but
    a local optimum, and §7's basin-stability gate ("anchors freezable only if
    basin-stable; knife-edge fits refused") was therefore `not_evaluable` for most
    of the registry: the engine was publishing fitted parameters whose basin
    status was unknown, which is a validity gap rather than a missing feature.

    `_spread_starts` now tops every model up to at least `n_starts` genuine,
    DETERMINISTIC, in-bounds starts placed across each parameter's own bounded
    range. The name-keyed starts are still constructed FIRST and in the same
    order, so a model that already had `n_starts` of them is bit-for-bit
    unchanged; only the previously single-start models gain probes. Read from the
    governed policy, so the previous behaviour remains one value away.
    """
    spec = REGISTRY[model_name]
    func, pnames = spec["func"], spec["params"]
    p = len(pnames)
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    if len(xa) <= p:                       # not enough points to identify p params
        return {"model": model_name, "converged": False, "n_params": p,
                "params": None, "reason": "too_few_points",
                "provenance": fp.fit_provenance(M2_RECIPE, {
                    **fp.basin_ledger([]),
                    "reason_code": "too_few_points"})}
    lo, hi = spec["bounds"](list(x), list(y))
    base = spec["guess"](list(x), list(y))

    starts = [base]
    if "t0" in pnames:
        j = pnames.index("t0")
        for tf in (0.3, 0.7):
            g = list(base); g[j] = _time_at_fraction(list(x), list(y), tf)
            starts.append(g)
    if "k" in pnames:
        j = pnames.index("k")
        g = list(base); g[j] *= 3.0; starts.append(g)
    # Top up to n_starts for the models the name-keyed rules never reached. Added
    # AFTER the name-keyed starts and in fixed order, so any model that already had
    # n_starts is untouched and ties still keep the earliest start.
    if (_start_policy("M2_START_TOPUP", "spread_within_bounds")
            == "spread_within_bounds" and len(starts) < max(1, n_starts)):
        starts.extend(_spread_starts(base, lo, hi,
                                     max(1, n_starts) - len(starts)))
    starts = starts[:max(1, n_starts)]

    # `outcomes` retains EVERY start — including the ones the old code silently
    # `continue`d past. This is the §7 evidence that used to be destroyed here.
    outcomes = []
    best = None
    for si, g0 in enumerate(starts):
        g0 = [min(max(v, lo[i]), hi[i]) for i, v in enumerate(g0)]
        try:
            popt, pcov = curve_fit(func, xa, ya, p0=g0, bounds=(lo, hi), maxfev=5000)
        except (RuntimeError, ValueError) as exc:
            outcomes.append({"index": si, "converged": False, "sse": None,
                             "theta": None, "failure": type(exc).__name__})
            continue
        yhat = func(xa, *popt)
        if not np.all(np.isfinite(yhat)):
            outcomes.append({"index": si, "converged": False, "sse": None,
                             "theta": None, "failure": "non_finite_prediction"})
            continue
        met = _metrics(ya, yhat, p)
        outcomes.append({"index": si, "converged": True, "sse": met["sse"],
                         "theta": [float(v) for v in popt], "failure": None})
        if best is None or met["sse"] < best["_sse"]:
            with np.errstate(invalid="ignore"):
                se = np.sqrt(np.diag(pcov)) if pcov is not None else None
            best = {"model": model_name, "converged": True, "n_params": p,
                    "params": {pnames[i]: float(popt[i]) for i in range(p)},
                    "param_se": {pnames[i]: (float(se[i]) if se is not None
                                             and np.isfinite(se[i]) else None)
                                 for i in range(p)},
                    "_sse": met["sse"],
                    **{k: (float(v) if math.isfinite(v) else None)
                       for k, v in met.items()}}

    # SST is passed so the inter-basin SSE gap is measured against the DATA's
    # scale; a near-perfect fit (SSE -> 0) otherwise gets compared against an
    # arbitrary epsilon. See fit_provenance._sse_denom.
    sst = float(np.sum((ya - ya.mean()) ** 2))
    prov = fp.fit_provenance(M2_RECIPE,
                             fp.basin_ledger(outcomes, lo, hi, sst=sst))
    if best is None:
        return {"model": model_name, "converged": False, "n_params": p,
                "params": None, "reason": "no_convergence", "provenance": prov}
    best.pop("_sse", None)
    best["provenance"] = prov
    return best


def _fit_bank(x, y, candidates, axis_label, n_starts=4) -> dict:
    out = {"candidates": candidates, "fits": {}, "best_by_aicc": None,
           "axis": axis_label,
           "note": "AICc here is preliminary; principled selection is M3."}
    best_name, best_aicc = None, float("inf")
    for name in candidates:
        r = fit_one(x, y, name, n_starts=n_starts)
        out["fits"][name] = r
        if r.get("converged") and r.get("aicc") is not None and r["aicc"] < best_aicc:
            best_aicc, best_name = r["aicc"], name
    out["best_by_aicc"] = best_name
    return out


def fit_curve(series: dict, n_starts: int = 4) -> dict:
    """KINETIC entry point: fit licensed models for one M1-triaged series."""
    m1 = series.get("m1", {})
    handling = m1.get("recommended_handling", "monotonic_fit")
    x = series.get("x_hours") or []
    y = m1.get("y_processed") or series.get("y_intensity") or []
    res = {"series_id": series.get("series_id"), "handling": handling,
           "signal_used": "m1.y_processed" if m1.get("y_processed") else "y_intensity"}
    cands = CANDIDATES.get(handling, [])
    if len(x) != len(y) or len(x) < 3 or not cands:
        res.update({"candidates": cands, "fits": {}, "best_by_aicc": None,
                    "note": "not fitted (rejected / malformed / no candidates)"})
        return res
    res.update(_fit_bank(x, y, cands, "time", n_starts))
    return res


def fit_dose_response(doses, responses, n_starts: int = 4) -> dict:
    """DOSE entry point: fit response-vs-concentration models (LNT, threshold,
    Hill, Brain–Cousens, scaling-law). Models needing more params than data
    points are skipped automatically by fit_one."""
    return _fit_bank(list(doses), list(responses), DOSE_RESPONSE, "dose", n_starts)


# --------------------------------- I/O ------------------------------------- #
def _run_kinetic(args):
    best_counter: Counter = Counter(); conv_fail = n = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[M2] kinetic fitting {args.input.name} ...")
    with open(args.input, encoding="utf-8") as fh, \
            open(args.output, "w", encoding="utf-8") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            res = fit_curve(json.loads(line))
            n += 1
            best_counter[res.get("best_by_aicc") or "none"] += 1
            conv_fail += sum(1 for r in res.get("fits", {}).values()
                             if not r.get("converged"))
            out.write(json.dumps(res) + "\n")
            if args.limit and n >= args.limit:
                break
    print(f"[M2] wrote {args.output}  ({n} curves)")
    print(json.dumps({"n_curves": n,
                      "best_model_by_aicc(preliminary)": dict(best_counter),
                      "non_converged_fits": conv_fail}, indent=2))


def _run_dose(args):
    """Fit dose-response to the R-rows (k_agg vs concentration) grouped by protein."""
    root = args.input.parent
    rc = root / "rate_concentration.jsonl"
    if not rc.exists():
        raise SystemExit(f"{rc} not found (run the ETL first)")
    groups = defaultdict(list)
    for line in open(rc, encoding="utf-8"):
        d = json.loads(line)
        c = d.get("concentration", {}).get("value_uM")
        if c is not None and d.get("k_agg") is not None:
            groups[d["protein"]].append((c, d["k_agg"]))
    out_path = root / "dose_response_fits.jsonl"
    n = 0
    best_counter: Counter = Counter()
    with open(out_path, "w", encoding="utf-8") as out:
        for protein, pts in groups.items():
            pts = sorted(set(pts))
            if len(pts) < 3:
                continue
            d, r = zip(*pts)
            res = fit_dose_response(d, r)
            res["protein"] = protein
            res["n_points"] = len(pts)
            best_counter[res.get("best_by_aicc") or "none"] += 1
            out.write(json.dumps(res) + "\n")
            n += 1
    print(f"[M2] dose-response: fitted {n} proteins -> {out_path}")
    print(json.dumps({"best_dose_model_by_aicc(preliminary)": dict(best_counter)},
                     indent=2))


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="PRISE M2 fitting bank")
    ap.add_argument("--input", type=Path,
                    default=root / "data" / "processed" / "curves_triaged.jsonl")
    ap.add_argument("--output", type=Path,
                    default=root / "data" / "processed" / "fits.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dose", action="store_true",
                    help="fit dose-response models to R-rows by protein")
    args = ap.parse_args(argv)
    # The constant half of §2.3 FitProvenance, written once. Every per-fit
    # `provenance.recipe` in the artifacts below is a reference INTO this file,
    # so it must exist for those references to resolve.
    fp.write_recipe_table(args.output.parent / "fit_recipes.json")
    if args.dose:
        _run_dose(args)
        return 0
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input} (run M1 first)")
    _run_kinetic(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
