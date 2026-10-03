"""
PRISE — Fit provenance & the §7 basin-stability gate
=====================================================

`PRISE_DESIGN.md` §2.3 specifies a `FitProvenance` record on every emitted
parameter, and §10.12 makes it an invariant. §7 goes further: an anchor is
freezable **only if basin-stable** ("winning multi-start basin robust to start
perturbation"; "knife-edge fits are refused"). Until this module, none of it was
emitted — and the §7 gate was not merely unenforced but **unevaluable**, because
`m2_fit.fit_one` kept only the best-SSE start and `continue`d past failures
without recording them, destroying the spread the gate needs at fit time.

This module is the one place that record is CONSTRUCTED. Fitting modules
(`m2_fit`, `global_fit`, `m3_select`, `m4_features`) hand it what they already
computed; it returns the JSON-able provenance block. It runs no optimizer, draws
no random number and changes no fitted value — by construction, since it only
ever sees outcomes that already exist.

=== TWO HALVES: A REFERENCED RECIPE + A PER-FIT LEDGER =====================

The §2.3 fields split cleanly into a part that is CONSTANT for a given fitting
policy (optimizer, tolerances, RNG applicability, version, env manifest) and a
part that VARIES per fit (how many starts ran, which won, how far apart they
landed). Inlining the constant half on all 9,633 model-fits in `fits.jsonl`
would add ~4.3 MB to say the same sentence 9,633 times. So the constant half is
emitted ONCE into `data/processed/fit_recipes.json` and referenced by `recipe`
id — which is exactly the shape §2.3 already uses for `env_manifest_ref`, a REF
rather than an inlined manifest.

The same logic applies twice more inside the per-fit block, and both were found
by MEASURING the rebuilt artifact rather than by estimating — the estimate was
wrong, which is the point of measuring. A first pass grew `fits.jsonl` by
**76%**, not the ~26% predicted. Two fixes: the explanatory prose became a
`reason_code` resolved by a table in the recipe file (860 KB inlined — 10.2% of
the whole artifact — to say one of seven things repeatedly), and
`winning_basin` / `basin_spread` are omitted in the single-basin case where the
first is derivable from `n_converged` and the second has no rival to measure
against. M3's `bootstrap_provenance` got the same treatment: its ~530-character
`stream_policy` resolves through the recipe instead of being restated on all
1,240 records.

MEASURED FINAL COST: `fits.jsonl` 4.78 MB -> 6.71 MB, **+40%** (~200 bytes per
model-fit over 9,633 fits). Higher than the original estimate and recorded here
as measured rather than as predicted.

A referenced record is only honest if the reference RESOLVES. `test_fit_
provenance.py` asserts every emitted `recipe` id exists in the recipe table.

=== WHAT IS HONESTLY NOT APPLICABLE =======================================

M2, `mechanistic` and `global_fit` use NO random number generator. Their
multi-start guesses are constructed deterministically from the data (the
registry `guess()`, `_time_at_fraction(x, y, {0.3, 0.7})`, and a x3 multiplier on
k), and TRF is a deterministic descent. For those fits `rng_seed` is not 0 and
not null — it is **not applicable**, recorded as such WITH THE REASON. Writing
`rng_seed: 0` into a deterministic fit would be a fabricated provenance record:
a lie that PASSES, strictly worse than the honest absence it replaced. See
`rng_not_applicable` and the negative control that guards it.

M3's wild bootstrap and M4's gamma bootstraps DO draw. Those get a real
`bootstrap_provenance`: the seed and batch sizes inline, and the algorithm plus
a `stream_policy` resolved through the recipe — where the policy states what it
actually is, including its known defect.

=== THE BASIN VERDICT (the §7 gate) =======================================

The load-bearing measurement, and the one that is easy to get wrong: **SSE alone
cannot decide basin stability.** Measured over 343 real model-fits, the
runner-up start's SSE is typically within 1e-11 RELATIVE of the winner's — not
because the fit is knife-edge but because both starts converged to the SAME
optimum. That is the definition of stable. A knife-edge is the opposite
situation: near-equal SSE at a MATERIALLY DIFFERENT parameter vector, where
which vector "wins" is a numerical accident of start ordering.

So converged starts are clustered into BASINS by parameter-vector distance
first, and SSE is compared only BETWEEN basins. Four verdicts, no fifth:

  stable         >=2 converged starts AND (they all agree on one basin OR every
                 rival basin is worse by more than `sse_rel_tol`)
  knife_edge     >=2 distinct basins whose SSE differs by <= `sse_rel_tol`
                 -> §7 refuses to freeze an anchor on this fit
  not_evaluable  <2 converged starts. Carries a `reason_code`. The honest
                 verdict for 9 of the 13 registry models, which construct only
                 ONE start (see below) and therefore have nothing to be stable
                 BETWEEN. It must never be silently upgraded to `stable`.
  no_fit         nothing converged

WHY SO MANY MODELS ARE `not_evaluable`: `m2_fit.fit_one` builds its extra starts
by testing PARAMETER NAMES (`if "t0" in pnames`, `if "k" in pnames`), not by any
general perturbation. So `logistic`/`gompertz`/`richards` get 4 starts and
`exponential` gets 2, while `lnt`, `scaling_law`, `brain_cousens`,
`finke_watzky` (its rates are named k1/k2, not k) and ALL FIVE dose-response
models get exactly ONE. Giving every model >=2 genuine starts would CHANGE
FITTED VALUES, so it is deliberately deferred to its own governed change; this
module reports the limitation instead of papering over it.

=== TOLERANCES: FROZEN, AND DELIBERATELY *NOT* SERVICE-C-CALIBRATED =======

`BASIN_SSE_REL_TOL = 1e-3` is empirically supported, not invented. Measured over
343 real model-fits: same-basin SSE gaps cluster at ~1e-11, while the genuine
start-to-start disagreements begin at 6.4% (and run to 4.2e8%). 1e-3 sits in a
five-order-of-magnitude empty gap between those populations.

`BASIN_PARAM_REL_TOL = 1e-3` was raised from an initial 1e-4 during detector
verification, and the direction matters: raising it makes the detector fire
LESS, so this is not a threshold fitted to manufacture a passing test. Two
independent reasons, the first of which does not depend on the resulting rate:

  (1) SCIENTIFIC. Two parameter vectors within 0.1% of each other are the same
      answer as far as anything downstream is concerned — t50, lag time, regime
      classification and gamma are all insensitive at that level. Calling them
      different basins would be a distinction PRISE never acts on. It also
      matches BASIN_SSE_REL_TOL in magnitude, which is coherent: the same
      "indistinguishable" standard applied to both axes.

  (2) EMPIRICAL, from a tolerance sweep over 1,116 real model-fits:

          param_rel_tol   stable   knife_edge   knife %
                  1e-05      292          283     25.4%
                  1e-04      506           69      6.2%
                  1e-03      551           24      2.2%
                  1e-02      554           21      1.9%

      At 1e-4, 71% of the flagged cases have an inter-basin distance below 1e-3,
      i.e. they pile up immediately above the threshold — the signature of a cut
      placed inside a noise population, not between two real ones. TRF stops
      when its STEP is small, so independent starts settle at points differing
      by ~1e-4 relative while sitting in the same optimum. Between 1e-3 and 1e-2
      the counts barely move (24 -> 21 knife-edge, 551 -> 554 stable): that
      plateau is where the genuine population lives, and a threshold chosen on a
      plateau is robust where one chosen on a slope is fitted. The distance
      distribution is visibly bimodal, with an empty gap between ~3e-3 and ~0.13.

Both constants live HERE and not in `calibrated_thresholds.py` on purpose. That
module is the signed, dated, `thresholds_version`-bound registry of Service-C
held-out-CALIBRATED SCIENTIFIC cutoffs. These are NUMERICAL tolerances that were
never calibrated against held-out data; filing them there would falsely imply
Service-C provenance and would drag `thresholds_version` (and therefore
`engine_build_id`) along for no reason.

"""
from __future__ import annotations

import math
import sys

# Identity of the provenance record's own schema. Distinct from the fitting
# modules' version tags: this versions the RECORD, not the fit.
FIT_PROVENANCE_VERSION = "fitprov-1.0"

# --- basin tolerances (frozen; see the module docstring for the measurement) --
BASIN_SSE_REL_TOL = 1e-3      # rival basins closer than this in SSE => knife-edge
BASIN_PARAM_REL_TOL = 1e-3    # parameter vectors closer than this => same basin

# Absolute floors for the relative parameter distance, so a parameter that is
# legitimately ~0 (e.g. a baseline fitted to 0.0) does not divide by nothing.
# When a bound is finite we scale the floor to the bound WIDTH, which is the only
# scale information available for that parameter; several registry models are
# genuinely unbounded (`_b_free` returns +/-inf), so a constant fallback is
# required rather than optional.
_ATOL_FRAC = 1e-9
_ATOL_ABS = 1e-12

# Floor for the inter-basin SSE denominator, as a fraction of the data's total
# sum of squares. Binds only when the winning fit is near-perfect; see
# `_sse_denom` and the `basin_ledger` docstring.
_SSE_DENOM_SST_FRAC = 1e-9

# Per-fit records carry a SHORT CODE; the prose lives here and is emitted once
# into fit_recipes.json. The same pattern m17_explain.py uses for its reason
# codes, and for the same reason: at 9,633 model-fits the explanatory sentence
# costs ~860 KB if inlined, to say one of six things over and over. A code plus
# a resolvable table is the same information at a fraction of the bytes.
BASIN_REASON_CODES = {
    "all_starts_agree":
        "every converged start landed in one basin",
    "rivals_decisively_worse":
        ("more than one basin was found, but every rival is worse than the "
         "winner by more than the SSE tolerance, so the winner is not a "
         "coin-flip"),
    "single_start_constructed":
        ("only one start was constructed for this model: m2_fit adds perturbed "
         "starts only when a parameter is literally named 't0' or 'k', which is "
         "false for 9 of the 13 registry models. There is nothing to be stable "
         "BETWEEN, so stability is unevaluated rather than verified — and §7 "
         "treats unevaluated as not-freezable"),
    "single_start_converged":
        ("more than one start was constructed but only one converged, so no "
         "comparison between basins is possible"),
    "basins_within_sse_tol":
        ("two or more materially different parameter vectors fit "
         "indistinguishably well, so which one wins is a numerical accident of "
         "start ordering. PRISE_DESIGN.md §7 refuses to freeze an anchor here"),
    "no_start_converged":
        "no start converged to a finite objective",
    "too_few_points":
        ("fewer data points than free parameters, so no start was constructed "
         "and no fit was attempted"),
}


def env_manifest_ref() -> str:
    """The environment tag this fit ran under.

    DELIBERATELY DUPLICATED from `m7_assemble._default_env_manifest()` rather
    than imported: `m7_assemble` imports `m2_fit`, so importing it back here
    would close an import cycle. `test_fit_provenance.py` asserts the two agree,
    which is a stronger guarantee than the import would have been (it fails
    loudly if either drifts) -- and it is what keeps this duplicate honest now
    that the tag is a real fingerprint rather than a constant.

    It carries python micro, numpy, scipy, BLAS vendor + version, OS and machine:
    everything in the software stack that can move the last digits of a bounded
    least-squares fit. It is still NOT §7's pinned-container guarantee -- the
    live thread-count environment can reorder BLAS reductions and is pinned by
    nothing here, because folding it in would make the build id depend on how the
    process was launched. That residual is documented rather than papered over.
    """
    try:
        import platform as _platform
        import sys as _sys
        py = "%d.%d.%d" % (_sys.version_info.major, _sys.version_info.minor,
                           _sys.version_info.micro)
        npv = spv = blas = blasv = None
        try:
            import numpy as _np
            npv = _np.__version__
            _b = (_np.__config__.show(mode="dicts").get(
                "Build Dependencies") or {}).get("blas") or {}
            blas, blasv = _b.get("name"), _b.get("version")
        except Exception:
            pass
        try:
            import scipy as _sp
            spv = _sp.__version__
        except Exception:
            pass
        return "|".join(("py" + py, "np" + str(npv), "sp" + str(spv),
                         "blas:" + str(blas) + "-" + str(blasv),
                         str(_platform.system()) + "/" + str(_platform.machine())))
    except Exception:                                   # pragma: no cover
        return "unknown"

def rng_not_applicable(reason: str) -> dict:
    """Record that NO random draw occurred, and why.

    Note what this deliberately does NOT contain: any `rng_seed` key, under any
    value. An absent stream has no seed. Emitting `rng_seed: 0` here would state
    that a stream was opened at seed 0, which is false, and would satisfy a
    naive §10.12 check while making the record less truthful than the absence it
    replaced. `test_m2_never_fabricates_an_rng_seed` exists to keep it that way.
    """
    if not reason:
        raise ValueError("an inapplicable RNG must carry a reason")
    return {"applicable": False, "reason": reason}


def rng_policy(algorithm: str, stream_policy: str) -> dict:
    """The CONSTANT half of an applicable RNG: which generator, and how its
    draws are laid out. Carries no seed — the seed is a property of the RUN and
    belongs in the per-record block via `bootstrap_provenance`. `stream_policy`
    is §2.3's field and is where a policy's KNOWN DEFECTS belong — it exists to
    expose the stream layout, not to flatter it. Splitting the policy from the
    seed is what lets the recipe be emitted once and referenced."""
    if not algorithm or not stream_policy:
        raise ValueError("an applicable RNG must carry an algorithm and a policy")
    return {"applicable": True, "rng_algorithm": algorithm,
            "stream_policy": stream_policy}


# =========================================================================== #
# THE RECIPE TABLE — the constant half of §2.3, emitted once
# =========================================================================== #
_M2_START_POLICY = (
    "deterministic-from-data: REGISTRY[model].guess(x, y); plus t0 set to "
    "_time_at_fraction(x, y, 0.3) and (x, y, 0.7) IFF a parameter is literally "
    "named 't0'; plus k multiplied by 3.0 IFF a parameter is literally named "
    "'k'; each start clipped into bounds; list truncated to n_starts. Because "
    "the extra starts are keyed on PARAMETER NAMES, 9 of the 13 registry models "
    "construct exactly ONE start and their basin stability is not evaluable."
)

_M2_RNG_REASON = (
    "no random draw occurs in this fit: the initial guesses are constructed "
    "deterministically from the data (registry guess(), _time_at_fraction at "
    "fractions 0.3/0.7, and a x3 multiplier on k) and trust-region-reflective "
    "least squares is a deterministic descent. rng_seed is therefore NOT "
    "APPLICABLE rather than 0 or null."
)

RECIPES = {
    "m2/trf-bounded/det-starts-1.0": {
        "optimizer": ("scipy.optimize.curve_fit -> "
                      "least_squares(method='trf', bounded)"),
        "tolerance": {"ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8, "max_nfev": 5000},
        "start_policy": _M2_START_POLICY,
        "rng": rng_not_applicable(_M2_RNG_REASON),
        "version": "m2-fitbank-1.0",
        "basin_tolerance": {"sse_rel": BASIN_SSE_REL_TOL,
                            "param_rel": BASIN_PARAM_REL_TOL,
                            "note": ("numerical tolerances, NOT Service-C "
                                     "calibrated; see fit_provenance.py")},
    },
    "mechanistic/trf-bounded-lsoda/single-start-1.0": {
        "optimizer": ("scipy.optimize.curve_fit -> least_squares(method='trf', "
                      "bounded) over scipy.integrate.solve_ivp(LSODA)"),
        "tolerance": {"ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8, "max_nfev": 4000,
                      "ode_rtol": 1e-6, "ode_atol": 1e-9},
        "start_policy": ("single deterministic start from the frozen log10 "
                         "guess table (kn -3, kp -1, k2 -2, kminus -2, KM -0.5)"),
        "rng": rng_not_applicable(_M2_RNG_REASON),
        "version": "mechanistic-tierb",
    },
    "global_fit/trf-bounded-lsoda/det-starts-1.0": {
        "optimizer": ("scipy.optimize.least_squares(method='trf', bounded, "
                      "x_scale='jac') over scipy.integrate.solve_ivp(LSODA)"),
        "tolerance": {"ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8, "max_nfev": 300,
                      "ode_rtol": 1e-6, "ode_atol": 1e-9},
        "start_policy": ("3 deterministic starts: the VARIANT_RATES decade "
                         "guesses; the same shifted -1 decade on kn and +1 on "
                         "kp; and the reaction orders nudged to n = 2*gamma_hint "
                         "- 1 from the phenomenological collapse gamma"),
        "rng": rng_not_applicable(
            "no random draw occurs: all three starts are constructed "
            "deterministically and TRF is a deterministic descent. NOTE the "
            "`seed` argument threaded through fit_global_series/_fit_mechanism "
            "is DECORATIVE — _fit_mechanism never references it, and the "
            "np.random.seed() call seeds the legacy global RandomState which "
            "least_squares does not consult."),
        "version": "global-fit-1.0",
        "basin_tolerance": {"sse_rel": BASIN_SSE_REL_TOL,
                            "param_rel": BASIN_PARAM_REL_TOL},
    },
    "m3/wild-bootstrap-mammen-1.0": {
        "optimizer": ("per-resample m2_fit.fit_one(n_starts=1) -> "
                      "curve_fit/least_squares(trf, bounded)"),
        "tolerance": {"ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8, "max_nfev": 5000},
        "start_policy": ("ONE start per resample (n_starts=1), so bootstrap "
                         "refits are not multi-started and their basin "
                         "stability is deliberately not assessed"),
        "rng": rng_policy(
            "numpy.random.Generator(PCG64)",
            "ONE Generator per curve, built as default_rng(seed) in "
            "analyze_curve with seed defaulting to 0 and NO --seed CLI knob. "
            "Each replicate draws n Mammen weights sequentially from that "
            "stream. KNOWN LIMITATION, recorded because stream_policy exists to "
            "expose exactly this: since every curve restarts from the same "
            "constant seed, two curves of equal length receive IDENTICAL weight "
            "sequences rather than independent ones. Deterministic, but the "
            "bootstrap ensemble is correlated ACROSS curves."),
        "resampler": "wild bootstrap, Mammen two-point weights (E=0, Var=1)",
        "version": "m3-select-1.0",
        "superseded_by": "m3/blockwise-wild-bootstrap-mammen-2.0",
        "superseded_note": ("SUPERSEDED and retained only so records written "
                            "under it still RESOLVE. Two defects it documents "
                            "or embodies are fixed in 2.0: the shared constant "
                            "seed (estimation-policy-1.0) and the pointwise "
                            "resampling of serially correlated residuals "
                            "(estimation-policy-1.1)."),
    },
    "m3/blockwise-wild-bootstrap-mammen-2.0": {
        "optimizer": ("per-resample m2_fit.fit_one(n_starts=1) -> "
                      "curve_fit/least_squares(trf, bounded)"),
        "tolerance": {"ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8, "max_nfev": 5000},
        "start_policy": ("ONE start per resample (n_starts=1), so bootstrap "
                         "refits are not multi-started and their basin "
                         "stability is deliberately not assessed"),
        "rng": rng_policy(
            "numpy.random.Generator(PCG64)",
            "ONE Generator per curve, built as default_rng(series_seed(id)) in "
            "analyze_curve -- a BLAKE2b digest of the series id, so each curve "
            "has a reproducible stream OF ITS OWN. This replaces the constant "
            "seed 0 under which two curves of equal length received identical "
            "weight sequences; measured, 98.4% of the corpus shared a length "
            "with another curve, so the bootstrap ensemble was correlated "
            "across curves (estimation-policy-1.0)."),
        "resampler": ("BLOCKWISE wild bootstrap, Mammen two-point weights "
                      "(E=0, Var=1) held CONSTANT within blocks of "
                      "ceil(n**(1/3)) consecutive residuals. The pointwise form "
                      "assumes serially INDEPENDENT errors; measured over the "
                      "corpus the residuals are not (median lag-1 "
                      "autocorrelation +0.182, 48.2% above 0.2, median "
                      "Durbin-Watson 1.483), so pointwise resampling destroyed "
                      "the dependence and produced intervals that were too "
                      "narrow. Block length is a function of n ALONE, never of "
                      "the estimated autocorrelation, so the interval does not "
                      "depend on a tuning parameter chosen from the same "
                      "residuals (estimation-policy-1.1)."),
        "version": "m3-select-1.1",
    },
    "m4/gamma-bootstrap-1.0": {
        "optimizer": ("censored log-log ML via scipy.optimize.minimize"
                      "(Nelder-Mead) for gamma_regression; least_squares(trf, "
                      "bounded) for the gamma_global master-curve collapse"),
        "tolerance": {"nelder_mead_xatol": 1e-5, "nelder_mead_fatol": 1e-5,
                      "nelder_mead_maxiter": 1500,
                      "least_squares_max_nfev": 4000},
        "start_policy": ("single deterministic start: OLS on the uncensored "
                         "points for the censored ML, and the OLS log-log slope "
                         "for the collapse. No multi-start, so basin stability "
                         "is not assessed for gamma."),
        "rng": rng_policy(
            "numpy.random.Generator(PCG64)",
            "TWO derived streams per concentration series from one base seed "
            "(default 0, no --seed CLI knob): default_rng(seed) for the "
            "2000-permutation informative-censoring test AND for the "
            "gamma_global wild bootstrap; default_rng(seed + 1) for the "
            "gamma_regression case-resampling bootstrap."),
        "resampler": ("case resampling (gamma_regression); wild bootstrap "
                      "(gamma_global); permutation (censoring test)"),
        "version": "m4-defs-1.0",
        "load_bearing_note": (
            "THIS STREAM GATES A PUBLISHED COUNT. m7_assemble.gamma_significant() "
            "requires gamma_ci to EXCLUDE ZERO before a curve earns the "
            "`scaling` tier of the information_yield ladder, and gamma_ci comes "
            "out of the bootstrap above. An unrecorded seed here was never "
            "bookkeeping: it was an unrecorded input to a headline number."),
    },
}


def recipe(recipe_id: str) -> dict:
    """Resolve a recipe id, raising loudly on an unknown one (a dangling
    reference is the one failure mode a referenced record can have)."""
    if recipe_id not in RECIPES:
        raise KeyError(f"unknown fit recipe: {recipe_id!r}")
    return RECIPES[recipe_id]


def recipe_table() -> dict:
    """The full, JSON-able recipe table written to data/processed/fit_recipes.json.

    `env_manifest_ref` is resolved at emission time (it is a property of the RUN,
    not of the recipe), which is why it is attached here rather than baked into
    the RECIPES literal."""
    env = env_manifest_ref()
    return {
        "schema": f"fit-recipes-{FIT_PROVENANCE_VERSION.split('-')[-1]}",
        "fit_provenance_version": FIT_PROVENANCE_VERSION,
        "env_manifest_ref": env,
        "note": ("The constant half of PRISE_DESIGN.md §2.3 FitProvenance, "
                 "emitted once and referenced by id from each fit's compact "
                 "provenance block. See engine/fit_provenance.py."),
        "recipes": {k: {**v, "env_manifest_ref": env}
                    for k, v in sorted(RECIPES.items())},
        # Resolves each per-fit `reason_code`. Kept out of the per-fit record for
        # the same reason the recipe is: 9,633 copies of a sentence is 860 KB.
        "basin_reason_codes": dict(sorted(BASIN_REASON_CODES.items())),
        "basin_verdicts": {
            "stable": "the winning basin is robust to start perturbation (§7)",
            "knife_edge": "§7 refuses to freeze an anchor on this fit",
            "not_evaluable": ("fewer than 2 converged starts, so stability is "
                              "UNVERIFIED, not verified-absent. §7 is "
                              "conservative: unevaluable is not freezable"),
            "no_fit": "nothing converged",
        },
    }


def write_recipe_table(path) -> dict:
    """Write the recipe table to `path` (deterministic bytes: sorted keys)."""
    import json
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    table = recipe_table()
    p.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
    return table


# =========================================================================== #
# THE BASIN LEDGER — the per-fit half, and the §7 gate
# =========================================================================== #
def _atol(lo, hi) -> float:
    """Absolute floor for one parameter's relative distance. Uses the bound WIDTH
    where the bounds are finite (the only scale information available), and a
    constant otherwise — several registry models are genuinely unbounded, and an
    infinite width would collapse every distance to zero and silently declare
    every fit single-basin."""
    if lo is None or hi is None:
        return _ATOL_ABS
    try:
        w = float(hi) - float(lo)
    except (TypeError, ValueError):
        return _ATOL_ABS
    if math.isfinite(w) and w > 0:
        return max(w * _ATOL_FRAC, _ATOL_ABS)
    return _ATOL_ABS


def param_distance(theta_a, theta_b, lo=None, hi=None) -> float:
    """Scale-free distance between two parameter vectors: the max over
    coordinates of |a-b| / max(|a|, |b|, atol). Relative, so a t0 of ~100 h and a
    k of ~0.02 /h are compared on equal terms rather than the larger coordinate
    dominating."""
    a = list(theta_a)
    b = list(theta_b)
    if len(a) != len(b):
        raise ValueError("parameter vectors of unequal length")
    worst = 0.0
    for j in range(len(a)):
        lo_j = lo[j] if lo is not None and j < len(lo) else None
        hi_j = hi[j] if hi is not None and j < len(hi) else None
        av, bv = float(a[j]), float(b[j])
        if not (math.isfinite(av) and math.isfinite(bv)):
            return float("inf")
        denom = max(abs(av), abs(bv), _atol(lo_j, hi_j))
        worst = max(worst, abs(av - bv) / denom)
    return float(worst)


def _sse_denom(sse_win: float, sst: float | None) -> float:
    """Denominator for the inter-basin SSE gap. Normally the winner's own SSE;
    floored at a billionth of the data's total variance so a near-perfect fit
    (SSE -> 0) is still compared on a meaningful scale rather than against an
    arbitrary absolute epsilon. See `basin_ledger`'s docstring for the real curve
    that forced this."""
    floor = _ATOL_ABS
    if sst is not None:
        try:
            s = float(sst)
            if math.isfinite(s) and s > 0:
                floor = max(s * _SSE_DENOM_SST_FRAC, _ATOL_ABS)
        except (TypeError, ValueError):
            pass
    return max(abs(float(sse_win)), floor)


def _cluster(converged, lo, hi, param_rel_tol):
    """Single-linkage cluster of converged starts into basins. n <= 4 in practice,
    so an explicit O(n^2) union-find is clearer than anything cleverer."""
    n = len(converged)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            d = param_distance(converged[i]["theta"], converged[j]["theta"], lo, hi)
            if d <= param_rel_tol:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    # order basins by their best SSE so basin 0 is always the winning basin
    return sorted(groups.values(),
                  key=lambda g: min(converged[i]["sse"] for i in g))


def basin_ledger(outcomes: list[dict], lo=None, hi=None, sst: float | None = None,
                 sse_rel_tol: float = BASIN_SSE_REL_TOL,
                 param_rel_tol: float = BASIN_PARAM_REL_TOL) -> dict:
    """Build the §7 basin record from per-start outcomes the caller ALREADY has.

    `outcomes` is one dict per CONSTRUCTED start, in construction order:
        {"index": int, "converged": bool,
         "sse": float | None, "theta": list[float] | None,
         "failure": str | None}

    Runs no optimizer and draws no random number: everything here is arithmetic
    on values the fitting loop already computed, which is what makes adding this
    record provably value-preserving.

    The winner is picked with a STRICT `<` scan in construction order, exactly
    reproducing `m2_fit.fit_one`'s existing rule (ties keep the EARLIER start).
    That tie-breaking is order-dependent, which is precisely why `win` is
    reported alongside the basin verdict and never on its own.

    `sst` (total sum of squares of the data) is what the inter-basin SSE gap is
    measured RELATIVE TO when the winning fit is near-perfect, and passing it is
    strongly recommended. WHY THIS EXISTS, since it was not obvious until the
    detector was verified against real data: a pure `(sse_rival - sse_win) /
    sse_win` ratio blows up when `sse_win` is ~0. Real curve CPAD-TK-1102 fits to
    `sse = 7e-22` (a near-exact digitised step), so the ratio was being taken
    against a bare absolute floor of 1e-12 — which produced the RIGHT verdict
    (`knife_edge`) by an arbitrary mechanism rather than a principled one.
    Flooring the denominator at `sst * 1e-9` instead makes the comparison
    dimensionally meaningful in every regime: "the rival basin differs by less
    than a billionth of the data's total variance" is a statement about
    identifiability, whereas "less than 1e-12 absolute" is a statement about
    nothing. For an ordinary fit (`sse_win >> sst * 1e-9`) the floor never binds
    and the measure is the plain relative gap, unchanged.
    """
    n_constructed = len(outcomes)
    conv = [o for o in outcomes
            if o.get("converged") and o.get("sse") is not None
            and o.get("theta") is not None and math.isfinite(o["sse"])]
    failures = [{"start": o.get("index"), "reason": o.get("failure") or "unknown"}
                for o in outcomes if not o.get("converged")]

    base = {"n_starts": n_constructed,
            "n_converged": len(conv),
            "n_failed": n_constructed - len(conv)}
    if failures:
        base["failed_starts"] = failures

    if not conv:
        base.update({"basin": "no_fit", "win": None, "n_basins": 0,
                     "reason_code": "no_start_converged"})
        return base

    # winner: strict `<` in construction order (ties keep the earlier start)
    win = conv[0]
    for o in conv[1:]:
        if o["sse"] < win["sse"]:
            win = o
    base["win"] = win.get("index")

    if len(conv) < 2:
        base.update({
            "basin": "not_evaluable", "n_basins": 1,
            "reason_code": ("single_start_constructed" if n_constructed < 2
                            else "single_start_converged"),
        })
        return base

    basins = _cluster(conv, lo, hi, param_rel_tol)
    win_basin = next(g for g in basins if any(conv[i] is win for i in g))
    base["n_basins"] = len(basins)
    if len(basins) == 1:
        # winning_basin would be {index: 0, n_starts: n_converged, fraction: 1.0}
        # — fully derivable from n_converged, so it is omitted rather than
        # written 4,710 times. Same for basin_spread, which has no rival to
        # measure against.
        base.update({"basin": "stable", "reason_code": "all_starts_agree"})
        return base

    base["winning_basin"] = {
        "index": basins.index(win_basin),
        "n_starts": len(win_basin),
        "fraction": round(len(win_basin) / len(conv), 4),
    }

    denom = _sse_denom(win["sse"], sst)
    rivals = [g for g in basins if g is not win_basin]
    gap = min((min(conv[i]["sse"] for i in g) - win["sse"]) / denom
              for g in rivals)
    dist = min(min(param_distance(conv[i]["theta"], win["theta"], lo, hi)
                   for i in g) for g in rivals)
    base["basin_spread"] = {"sse_rel_gap": float(gap),
                            "param_rel_dist": float(dist)}
    if gap > sse_rel_tol:
        base.update({"basin": "stable",
                     "reason_code": "rivals_decisively_worse"})
    else:
        base.update({"basin": "knife_edge",
                     "reason_code": "basins_within_sse_tol"})
    return base


def bootstrap_provenance(recipe_id: str, seed: int, requested_B: int,
                         version: str, accepted_B: int | None = None,
                         **extra) -> dict:
    """The per-record block for a module whose RNG genuinely runs (M3, M4).

    Deliberately COMPACT, for the same reason the fit ledger is: the algorithm
    and `stream_policy` are constant for the recipe, so they are resolved
    THROUGH `recipe` rather than restated on every record — M3's stream_policy
    alone is ~530 characters and would add ~660 KB to `m3_sample.jsonl` to
    repeat one sentence 1,240 times. What varies per record is the seed and the
    batch sizes, and those are inline.

    `requested_B` vs `accepted_B` are both kept where both are known, because
    they differ: resamples where no candidate converged are dropped, so the
    accepted count is what the interval was actually built from.

    `accepted_B` is OPTIONAL and defaults to None, which is emitted as
    `accepted_B_tracked: false` rather than being back-filled with
    `requested_B`. M4 is the case that forced this: its gamma bootstrap rejects
    divergent draws and separately skips degenerate resamples, and it does not
    return the surviving count — so writing `accepted_B = requested_B` there
    would assert that every draw survived, which is false. An untracked count is
    reported as untracked. This is the same rule as `rng_seed`: a number nobody
    measured does not get invented to fill the field."""
    rec = recipe(recipe_id)
    if not rec["rng"].get("applicable"):
        raise ValueError(f"{recipe_id} declares no RNG; use fit_provenance()")
    if seed is None:
        raise ValueError("an applicable RNG must carry a seed")
    out = {"recipe": recipe_id, "rng_seed": int(seed),
           "requested_B": int(requested_B),
           "env_manifest_ref": env_manifest_ref(), "version": version}
    if accepted_B is None:
        out["accepted_B_tracked"] = False
    else:
        out["accepted_B"] = int(accepted_B)
    out.update(extra)
    return out


def fit_provenance(recipe_id: str, ledger: dict) -> dict:
    """The compact per-fit block written into an artifact: a recipe REFERENCE
    plus the basin ledger. Validates the reference resolves before emitting it."""
    recipe(recipe_id)
    return {"recipe": recipe_id, **ledger}


# =========================================================================== #
# §7 consumption: is a fit freezable?
# =========================================================================== #
def is_basin_stable(prov: dict | None) -> bool:
    """§7's gate, as a single callable: an anchor is freezable ONLY if the fit
    behind it is basin-stable. `not_evaluable` returns False — an unevaluable fit
    is not a stable one, and §7's posture ("knife-edge fits are refused") is
    conservative. Being unable to check is not permission."""
    if not isinstance(prov, dict):
        return False
    return prov.get("basin") == "stable"
