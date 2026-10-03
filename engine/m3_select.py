"""
PRISE — Module M3: Comparison, Selection & Identifiability (bootstrap-enveloped)
================================================================================

The central statistical move (PRISE_DESIGN.md §3-M3, §4): **model selection runs
INSIDE the bootstrap**. Selecting a model and then bootstrapping conditional on
the winner under-states uncertainty (post-selection inference). So each bootstrap
resample re-fits the candidates and re-selects; we report:

  * selection-stability frequencies   (how often each model wins)
  * a cross-model predictive interval for t50 (a model-transferable functional),
    with a multimodality flag when selection is unstable
  * per-model *conditional* parameter CIs (model-specific params are reported only
    for the resamples that selected that model — never pooled into one number)
  * an identifiability verdict for the point-selected model (sensitivity-
    collinearity condition number — the column-normalised Jacobian's correlation
    matrix, NOT the Fisher Information Matrix; the true observed FIM for the ODE
    family is in global_fit.py): condition number, the sloppy direction,
    practical (non-)identifiability flag

Resampling: **wild bootstrap** (Mammen weights) — preserves the heteroscedastic
(multiplicative) ThT noise. NOTE serial correlation along time is not yet handled
(a block/sieve bootstrap is the documented refinement); BCa intervals and a
cross-validated descriptive-vs-mechanistic criterion are also deferred.

Tiers: descriptive vs mechanistic are reported separately (best_descriptive /
best_mechanistic) so a sloppy mechanistic fit can't masquerade as a clean win.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import fit_provenance as fp
from m2_fit import fit_curve, fit_one

# The GOVERNED bootstrap RNG policy (bootstrap_policy.py). READ, not hard-coded, so
# the corrected per-series stream is one value away from being in force and cannot
# drift from the record that describes it. Its `applied` value is still
# "constant_zero" -- see that module for the measurement (98.4% of curves share a
# resample stream) and for why applying it needs a ~5.2 h M3 rebuild first.
try:                                                   # pragma: no cover - import glue
    from bootstrap_policy import applied_policy as _bootstrap_policy
    from bootstrap_policy import series_seed as _series_seed
except Exception:                                      # pragma: no cover - fallback
    def _bootstrap_policy(name, default=None):
        return {"M3_SEED_STREAM": "per_series_derived",
                "M3_BOOTSTRAP_BLOCKING": "blockwise_n_cbrt"}.get(name, default)

    def _series_seed(series_id, base=0):
        import hashlib
        d = hashlib.blake2b(f"{base}:{series_id}".encode("utf-8"),
                            digest_size=8).digest()
        return int.from_bytes(d, "big") >> 1


def resolve_bootstrap_seed(series_id, seed=None) -> int:
    """The seed this curve's bootstrap draws from.

    An EXPLICIT `seed` always wins (tests and the --series path rely on it).
    Otherwise the governed policy decides: "constant_zero" reproduces the shipped
    corpus exactly, "per_series_derived" gives each curve a reproducible stream of
    its own so equal-length curves stop sharing resample indices."""
    if seed is not None:
        return int(seed)
    if _bootstrap_policy("M3_SEED_STREAM", "constant_zero") == "per_series_derived":
        return _series_seed(series_id)
    return 0
from models import REGISTRY

# Module identity — consumed by M7's engine_build_id (the comparability key, §7).
# Purely a provenance tag; bumping it makes downstream builds non-comparable.
M3_VERSION = "m3-select-1.1"
M3_RECIPE = "m3/blockwise-wild-bootstrap-mammen-2.0"      # §2.3 recipe id (fit_recipes.json)

MECHANISTIC = {"finke_watzky"}          # closed-form mechanistic models in the kinetic bank
MIN_RELIABLE_POINTS = 8                 # below this, bootstrap CIs flagged unreliable
# Collinearity condition number (column-normalised sensitivity matrix) above which
# a parameter combination is practically unconstrained. NOW Service-C-CALIBRATED
# (thresholds-1.0): the first-pass round guess (1000.0) is superseded by the
# held-out recommendation 107.0 (out-of-sample separation AUC 1.0) — clean fits
# separate from sloppy/collinear ones at ~10² not 10³. The previous 1000.0 is
# retained in engine/calibrated_thresholds.py as `first_pass`. Name unchanged.
from calibrated_thresholds import calibrated_value as _cal, THRESHOLDS_VERSION
COND_NUMBER_NONIDENT = _cal("M3_COND_NUMBER_NONIDENT", default=1000.0)
_S5 = math.sqrt(5.0)


# ------------------------------ functionals -------------------------------- #
def predict(name: str, params: dict, x) -> np.ndarray:
    spec = REGISTRY[name]
    vals = [params[p] for p in spec["params"]]
    return np.asarray(spec["func"](np.asarray(x, float), *vals), float)


def t50_of(name: str, params: dict, x) -> float | None:
    """Time to half-maximum of the fitted curve — a model-transferable functional."""
    grid = np.linspace(float(min(x)), float(max(x)), 400)
    y = predict(name, params, grid)
    if not np.all(np.isfinite(y)):
        return None
    ymin, ymax = float(y.min()), float(y.max())
    if ymax - ymin < 1e-9:
        return None
    half = ymin + 0.5 * (ymax - ymin)
    for i in range(len(grid) - 1):
        if (y[i] - half) * (y[i + 1] - half) <= 0 and y[i + 1] != y[i]:
            r = (half - y[i]) / (y[i + 1] - y[i])
            return float(grid[i] + r * (grid[i + 1] - grid[i]))
    return None


def akaike_weights(converged: dict) -> dict:
    amin = min(r["aicc"] for r in converged.values())
    dw = {n: math.exp(-0.5 * (r["aicc"] - amin)) for n, r in converged.items()}
    z = sum(dw.values()) or 1.0
    return {n: v / z for n, v in dw.items()}


# ------------------------------ identifiability ---------------------------- #
def fim_identifiability(name: str, params: dict, x, y) -> dict:
    """Identifiability via the condition number of the *column-normalised*
    sensitivity matrix (collinearity of parameter sensitivities, independent of
    parameter scale). cond≈1 → orthogonal/well-identified; cond→∞ → a parameter
    combination is unconstrained (practically non-identifiable). The eigenvector
    of the smallest eigenvalue is the sloppiest (least-constrained) direction.

    HONESTY (naming): despite the legacy function name, this statistic is the
    *sensitivity-collinearity condition number* — the condition number of the
    column-normalised Jacobian's correlation matrix — NOT the Fisher Information
    Matrix (the parameter-covariance curvature J^Tσ^-2 J). It detects PRACTICAL
    non-identifiability (sloppy / collinear parameter combinations) for the
    closed-form kinetic models. The true observed FIM for the Tier-B ODE family
    lives in `global_fit.shared_block_identifiability` (cross-reference). The name
    and output keys are kept for backward compatibility; the `statistic`/`note`
    fields below carry the honest label."""
    _clarifier = {
        "statistic": "sensitivity_collinearity_condition_number",
        "note": ("collinearity of parameter sensitivities (column-normalized "
                 "Jacobian), NOT the Fisher-information parameter covariance; "
                 "detects PRACTICAL non-identifiability. The true observed FIM "
                 "(J^Tσ^-2 J) for the ODE family is in "
                 "global_fit.shared_block_identifiability."),
    }
    order = REGISTRY[name]["params"]
    func = REGISTRY[name]["func"]
    theta = np.array([params[p] for p in order], float)
    xa = np.asarray(x, float)
    n, p = len(xa), len(theta)
    J = np.zeros((n, p))
    for j in range(p):
        h = 1e-5 * max(abs(theta[j]), 1e-3)
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h; tm[j] -= h
        J[:, j] = (func(xa, *tp) - func(xa, *tm)) / (2 * h)

    norms = np.sqrt(np.sum(J ** 2, axis=0))
    dead = [order[j] for j in range(p) if norms[j] <= 1e-12]   # no effect at all
    live = norms > 1e-12
    if dead or not np.all(live):
        # a parameter has ~zero sensitivity -> structurally non-identifiable here
        return {"flag": "structurally_non_identifiable",
                "condition_number": None, "log10_collinearity": None,
                "dead_parameters": dead,
                "sloppiest_direction": {order[j]: (0.0 if live[j] else 1.0)
                                        for j in range(p)},
                **_clarifier}

    Jn = J[:, :] / norms                      # unit-norm columns
    C = Jn.T @ Jn                              # unit-diagonal sensitivity correlation
    try:
        eig, vecs = np.linalg.eigh(C)
    except np.linalg.LinAlgError:
        return {"flag": "fim_failed", "condition_number": None, **_clarifier}
    eig = np.clip(eig, 0.0, None)
    emax = float(eig.max())
    emin = float(eig[eig > 0].min()) if np.any(eig > 0) else 0.0
    cond = (emax / emin) if emin > 0 else float("inf")
    sloppy = {order[i]: round(float(vecs[i, 0]), 3) for i in range(p)}
    flag = ("practically_non_identifiable"
            if (not math.isfinite(cond) or cond > COND_NUMBER_NONIDENT)
            else "identifiable")
    return {"flag": flag,
            "condition_number": (float(cond) if math.isfinite(cond) else None),
            "log10_collinearity": (round(math.log10(cond), 2)
                                   if math.isfinite(cond) and cond > 0 else None),
            "sloppiest_direction": sloppy,
            **_clarifier}


# ------------------------------ bootstrap ---------------------------------- #
def _mammen(n: int, rng) -> np.ndarray:
    """Wild-bootstrap weights (Mammen two-point): E=0, Var=1, preserves heterosked."""
    p = (_S5 + 1) / (2 * _S5)
    u = rng.random(n)
    return np.where(u < p, -(_S5 - 1) / 2.0, (_S5 + 1) / 2.0)


def block_length(n: int) -> int:
    """Frozen block length for the blockwise wild bootstrap: ceil(n**(1/3)).

    The standard rate for block bootstraps, and deliberately a FUNCTION OF n
    ALONE -- not of the estimated autocorrelation. A data-chosen block length
    would make the interval depend on a tuning parameter selected from the same
    residuals it is being applied to, which is the post-selection trap this
    module already fights elsewhere. A frozen rate is conservative, reproducible
    and versioned."""
    return max(1, int(math.ceil(max(1, int(n)) ** (1.0 / 3.0))))


def _block_mammen(n: int, rng, block: int) -> np.ndarray:
    """Mammen weights held CONSTANT within blocks of consecutive residuals.

    WHY THIS EXISTS. The pointwise wild bootstrap multiplies every residual by an
    INDEPENDENT draw, which is valid for heteroskedastic but SERIALLY INDEPENDENT
    errors. Kinetic residuals are not serially independent: a model that slightly
    misfits a sigmoid's shape leaves consecutive residuals sharing a sign.
    MEASURED over the corpus -- median lag-1 autocorrelation +0.182, 48.2% of
    curves above 0.2 and 24.9% above 0.5, median Durbin-Watson 1.483 with 30.3%
    below 1.0. Resampling those as independent destroys the dependence the data
    actually has and yields intervals that are TOO NARROW, worst exactly on the
    curves whose fits are least trustworthy.

    Holding the weight constant across a block preserves within-block dependence
    (Shao's blockwise wild bootstrap). Where residuals really are independent the
    estimator remains valid, merely slightly less efficient -- so applying it
    corpus-wide is the conservative choice rather than a per-curve decision."""
    if block <= 1:
        return _mammen(n, rng)
    n_blocks = int(math.ceil(n / float(block)))
    w = _mammen(n_blocks, rng)
    return np.repeat(w, block)[:n]


def bootstrap_select(x, y, converged: dict, best: str, B: int, rng,
                     delta: float = 10.0, n_starts: int = 1) -> dict:
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    yhat = predict(best, converged[best]["params"], xa)
    resid = ya - yhat
    amin = converged[best]["aicc"]
    subset = [n for n in converged if converged[n]["aicc"] - amin < delta]
    if best not in subset:
        subset.append(best)

    sel: Counter = Counter()
    t50s: list[float] = []
    pby: dict[str, list[dict]] = defaultdict(list)
    blk = (block_length(len(resid))
           if _bootstrap_policy("M3_BOOTSTRAP_BLOCKING", "blockwise_n_cbrt")
           == "blockwise_n_cbrt" else 1)
    out_block_length = blk
    for _ in range(B):
        ystar = yhat + resid * _block_mammen(len(resid), rng, blk)
        local = {}
        for nm in subset:
            r = fit_one(list(xa), list(ystar), nm, n_starts=n_starts)
            if r.get("converged") and r.get("aicc") is not None:
                local[nm] = r
        if not local:
            continue
        s = min(local, key=lambda n: local[n]["aicc"])
        sel[s] += 1
        t = t50_of(s, local[s]["params"], xa)
        if t is not None:
            t50s.append(t)
        pby[s].append(local[s]["params"])

    total = sum(sel.values()) or 1
    freqs = {n: round(sel[n] / total, 3) for n in sel}
    t50_ci = None
    if len(t50s) >= 10:
        a = np.array(t50s)
        t50_ci = [float(np.percentile(a, 2.5)), float(np.median(a)),
                  float(np.percentile(a, 97.5))]
    maxf = max(freqs.values()) if freqs else 0.0
    cond_ci = {}
    for nm, plist in pby.items():
        if len(plist) >= 10:
            keys = list(plist[0].keys())
            cond_ci[nm] = {k: [float(np.percentile([pp[k] for pp in plist], 2.5)),
                               float(np.percentile([pp[k] for pp in plist], 97.5))]
                           for k in keys}
    return {"selection_frequencies": freqs,
            "selection_stability": round(maxf, 3),
            "t50_predictive_interval": t50_ci,
            "t50_multimodal": bool(maxf < 0.6),
            "per_model_conditional_CIs": cond_ci,
            "n_bootstrap": total, "resampled_models": subset,
            # consumed by analyze_curve into bootstrap_provenance, then popped;
            # it is provenance about HOW the interval was built, not a result
            "_block_length": out_block_length}


# ------------------------------ orchestration ------------------------------ #
def analyze_curve(series: dict, B: int = 200, seed: int | None = None) -> dict:
    fitres = fit_curve(series)
    fits = fitres.get("fits", {})
    converged = {n: r for n, r in fits.items()
                 if r.get("converged") and r.get("aicc") is not None}
    out = {"series_id": series.get("series_id"),
           "handling": fitres.get("handling")}
    if not converged:
        out["status"] = "no_converged_fit"
        return out

    weights = akaike_weights(converged)
    best = min(converged, key=lambda n: converged[n]["aicc"])
    best_desc = min((n for n in converged if n not in MECHANISTIC),
                    key=lambda n: converged[n]["aicc"], default=None)
    best_mech = min((n for n in converged if n in MECHANISTIC),
                    key=lambda n: converged[n]["aicc"], default=None)

    x = series.get("x_hours") or []
    y = series.get("m1", {}).get("y_processed") or series.get("y_intensity") or []
    seed = resolve_bootstrap_seed(series.get("series_id"), seed)
    rng = np.random.default_rng(seed)
    boot = bootstrap_select(x, y, converged, best, B, rng)
    # §2.3/§10.12: this bootstrap DOES draw, so unlike M2 it carries a real seed
    # rather than a not-applicable marker. `requested_B` vs `accepted_B` matters
    # — resamples where no candidate converged are dropped, so the accepted count
    # is what the CI was actually built from and was previously the only clue
    # anyone had about the batch size.
    # M3 CAN report accepted_B honestly: bootstrap_select returns the number of
    # resamples that actually produced a selection (`n_bootstrap`), which is what
    # the interval was built from and is <= B when a resample failed to converge.
    boot["bootstrap_provenance"] = fp.bootstrap_provenance(
        M3_RECIPE, seed, requested_B=B, version=M3_VERSION,
        accepted_B=int(boot.get("n_bootstrap") or 0),
        # the block length the interval was actually built with. Recorded per
        # record because it is a function of THIS curve's n, so unlike the
        # resampler description it genuinely varies and cannot be resolved
        # through the recipe alone.
        block_length=int(boot.pop("_block_length", 1)))
    ident = fim_identifiability(best, converged[best]["params"], x, y)

    out.update({
        "status": "ok",
        "best_point": best,
        "best_descriptive": best_desc,
        "best_mechanistic": best_mech,
        "akaike_weights": {n: round(w, 3) for n, w in weights.items()},
        "point_aicc": {n: round(converged[n]["aicc"], 2) for n in converged},
        "selection": boot,
        "identifiability_of_best": ident,
        "n_points": len(x),
        "ci_reliable": len(x) >= MIN_RELIABLE_POINTS,
    })
    return out


# --------------------------------- I/O ------------------------------------- #
def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="PRISE M3 bootstrap selection")
    ap.add_argument("--input", type=Path,
                    default=root / "data" / "processed" / "curves_triaged.jsonl")
    ap.add_argument("--output", type=Path,
                    default=root / "data" / "processed" / "m3_sample.jsonl")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N curves OF THIS SHARD (resume support)")
    ap.add_argument("--shards", type=int, default=1,
                    help="split the fittable curves into N disjoint shards")
    ap.add_argument("--shard", type=int, default=0,
                    help="which shard (0..shards-1) this process handles")
    ap.add_argument("--limit", type=int, default=60,
                    help="number of fittable curves to analyse (0=all; expensive)")
    ap.add_argument("--bootstrap", type=int, default=200)
    args = ap.parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input} (run M1 first)")

    stab = []
    multimodal = nonident = unreliable = n = 0
    fittable_idx = -1
    shard_seen = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[M3] bootstrap-selecting (B={args.bootstrap}) ...")
    with open(args.input, encoding="utf-8") as fh, \
            open(args.output, "w", encoding="utf-8") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("m1", {}).get("fittability_class") != "fittable":
                continue
            # SHARDING. Curves are fitted independently and the bootstrap seed is
            # now per-curve, so there is no cross-curve state to serialise: a shard
            # produces exactly the records a single process would have produced for
            # those indices. The index counts FITTABLE curves only, so the partition
            # does not shift with unrelated triage changes.
            fittable_idx += 1
            if args.shards > 1 and (fittable_idx % args.shards) != args.shard:
                continue
            # --skip lets a shard be completed across several bounded runs without
            # redoing work: it counts curves belonging to THIS shard, so
            # (--skip 0 --limit K) then (--skip K --limit K) partitions the shard
            # exactly. Purely a scheduling control -- it cannot change any record,
            # because every curve is fitted independently from a per-curve seed.
            shard_seen += 1
            if args.skip and shard_seen <= args.skip:
                continue
            res = analyze_curve(s, B=args.bootstrap)
            if res.get("status") != "ok":
                continue
            n += 1
            stab.append(res["selection"]["selection_stability"])
            multimodal += res["selection"]["t50_multimodal"]
            nonident += (res["identifiability_of_best"]["flag"]
                         == "practically_non_identifiable")
            unreliable += not res["ci_reliable"]
            out.write(json.dumps(res) + "\n")
            if args.limit and n >= args.limit:
                break
    print(f"[M3] wrote {args.output}  ({n} curves)")
    print(json.dumps({
        "n_curves": n,
        "mean_selection_stability": round(float(np.mean(stab)), 3) if stab else None,
        "curves_with_multimodal_t50": multimodal,
        "curves_practically_non_identifiable": nonident,
        "curves_with_unreliable_CIs": unreliable,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
