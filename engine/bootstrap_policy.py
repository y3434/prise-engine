"""
PRISE — Estimation Policy (the GOVERNED apply artifact for RNG + fit starts)
============================================================================

The third governed apply artifact, in the same form as `calibrated_thresholds.py`
and `join_policy.py`. It records two **measured estimation defects** — one in M3's
bootstrap seeding, one in M2's optimisation starts — and their corrections, both
now APPLIED.

DEFECT 1 — M3's shared bootstrap stream. `analyze_curve(series, B, seed=0)` took a
constant seed and the driver never overrode it, so EVERY curve drew its
wild-bootstrap resamples from `np.random.default_rng(0)`. Two curves with the same
number of points therefore received **byte-identical resample index matrices** —
not merely similar draws, the same ones. MEASURED: 1,654 curves carry an x grid and
**1,627 of them — 98.4% — share their length with at least one other curve**
(n=5: 88 curves, n=11: 83, n=7: 81, n=9: 64).

Why "it is deterministic" is not a defence: determinism is required (§7) and the
constant seed does deliver it — but it delivers it by making Monte-Carlo error
COMMON across curves rather than independent. `pooling.py`'s empirical-Bayes
shrinkage and `m14_meta.py`'s DerSimonian-Laird meta-regression both treat each
curve's interval as independently estimated. It is not: within a length class the
resampling noise is perfectly correlated, so pooling does not average it away at
the rate independence implies, and an unrepresentative resample pattern biases
every curve of that length in the same direction at once.

The fix is not "use a random seed" — that destroys reproducibility. It is a
per-series derivation: `series_seed()` uses BLAKE2b over the id, stable across
processes, platforms and Python versions, unlike the salted builtin `hash()`.

DEFECT 2 — M2's single-start fits. The extra optimisation starts were keyed on
PARAMETER NAMES (`t0`, `k`), so `logistic`/`gompertz`/`richards` got 4 starts,
`exponential` got 2, and **the other 9 registry models got exactly ONE** —
including `finke_watzky` and every dose-response model. A single-start fit cannot
be distinguished from a local optimum, and §7's basin-stability gate ("anchors
freezable only if basin-stable; knife-edge fits refused") was therefore
`not_evaluable` for most of the registry: the engine published fitted parameters
whose basin status was unknown. `_spread_starts` now tops every model up to
`n_starts` deterministic in-bounds probes, appended AFTER the name-keyed starts so
a model that already had enough is bit-for-bit unchanged.

BOTH ARE APPLIED, and `BOOTSTRAP_POLICY_VERSION` is folded into `engine_build_id`
via m7_assemble.build_components, so the prior corpus is correctly marked
non-comparable (§7).

Nothing here raises: a lookup miss returns the supplied default.
"""
from __future__ import annotations

import hashlib

# --------------------------------------------------------------------------- #
# Version tag — folded into engine_build_id (m7_assemble.build_components), so
# changing ANY applied value below invalidates descendants (section 7).
# --------------------------------------------------------------------------- #
BOOTSTRAP_POLICY_VERSION = "estimation-policy-1.1"

SOURCE_ARTIFACT = "M2/M3 estimation audit (2026-08-17)"

BOOTSTRAP_POLICY = {
    "M2_START_TOPUP": {
        "module": "m2_fit.fit_one",
        "previous": "name_keyed_only",
        "applied": "spread_within_bounds",
        "status": "applied",
        "metric": ("how many genuine optimisation starts each registry model "
                   "receives, and therefore whether the basin-stability gate can "
                   "be evaluated for it at all"),
        "evidence": {
            "definition": ("starts constructed per model in m2_fit.fit_one before "
                           "truncation to n_starts, counted by parameter name over "
                           "the 13 registry models"),
            "starts_before": {"logistic": 4, "gompertz": 4, "richards": 4,
                              "exponential": 2, "each_of_the_other_9": 1},
            "n_models_single_start_before": 9,
            "consequence_before": ("a single-start fit cannot be distinguished "
                                   "from a local optimum, so basin stability was "
                                   "`not_evaluable` for those 9 and the section 7 "
                                   "gate could not be applied to them at all"),
            "after": "every registry model receives n_starts genuine starts",
        },
        "rationale": ("the top-up is DETERMINISTIC -- a frozen fraction sequence "
                      "across each parameter's own bounded range, no RNG -- so the "
                      "basin ledger is reproducible, and it is appended AFTER the "
                      "name-keyed starts so any model that already had n_starts is "
                      "bit-for-bit unchanged. Only the previously single-start "
                      "models gain probes: the minimum change that makes the gate "
                      "evaluable."),
        "effect": ("basin stability becomes evaluable across the registry; fitted "
                   "values move wherever a probe finds a better optimum than the "
                   "initial guess did, which IS the defect being corrected"),
    },
    "M3_BOOTSTRAP_BLOCKING": {
        "module": "m3_select.bootstrap_select",
        "previous": "pointwise",
        "applied": "blockwise_n_cbrt",
        "status": "applied",
        "metric": ("whether the wild-bootstrap weight is drawn independently per "
                   "residual or held constant across blocks of consecutive ones"),
        "evidence": {
            "definition": ("residuals of each curve's AICc-best converged fit "
                           "against its own x/y grid; lag-1 autocorrelation is "
                           "sum(e[:-1]*e[1:])/sum(e*e) after centring, and "
                           "Durbin-Watson is sum(diff(e)**2)/sum(e*e), over the "
                           "1,504 curves where a best fit and a usable grid both "
                           "exist"),
            "n_curves_measured": 1504,
            "median_lag1_autocorrelation": 0.182,
            "fraction_rho_gt_0_2": 0.482,
            "fraction_rho_gt_0_5": 0.249,
            "median_durbin_watson": 1.483,
            "fraction_dw_lt_1_5": 0.503,
            "fraction_dw_lt_1_0": 0.303,
            "consequence": ("the pointwise wild bootstrap is valid for "
                            "heteroskedastic but SERIALLY INDEPENDENT errors. "
                            "Half this corpus is not serially independent, so "
                            "resampling pointwise destroyed the dependence the "
                            "data has and produced intervals that are TOO "
                            "NARROW -- and worst on the most autocorrelated "
                            "curves, i.e. exactly the fits least worth trusting"),
        },
        "rationale": ("Shao's blockwise wild bootstrap: one Mammen draw per block "
                      "of ceil(n**(1/3)) consecutive residuals preserves "
                      "within-block dependence while keeping E=0, Var=1 (verified: "
                      "within-block correlation 1.0, across-block ~0). The block "
                      "length is a function of n ALONE and never of the estimated "
                      "autocorrelation -- a data-chosen block would make the "
                      "interval depend on a tuning parameter selected from the "
                      "same residuals, which is the post-selection trap this "
                      "module fights elsewhere. Where residuals really are "
                      "independent the estimator stays valid and merely loses a "
                      "little efficiency, so applying it corpus-wide is the "
                      "conservative choice rather than a per-curve decision."),
        "effect": ("every published M3 interval widens on autocorrelated curves; "
                   "selection frequencies and t50 predictive intervals move"),
    },
    "M3_SEED_STREAM": {
        "module": "m3_select.analyze_curve",
        "previous": "constant_zero",
        "applied": "per_series_derived",
        "status": "applied",
        "metric": ("the RNG stream each curve's wild bootstrap draws its resample "
                   "indices from"),
        "evidence": {
            "definition": ("curves in curves_triaged.jsonl carrying a non-empty "
                           "x_hours grid, grouped by len(x_hours); a curve is "
                           "COUPLED when at least one other curve shares its "
                           "length, since np.random.default_rng(0) then yields it "
                           "the identical resample index matrix"),
            "n_curves_with_a_grid": 1654,
            "n_coupled": 1627,
            "fraction_coupled": 0.984,
            "largest_length_classes": {"n=5": 88, "n=11": 83, "n=7": 81,
                                       "n=9": 64, "n=8": 60, "n=16": 51},
            "consequence": ("per-curve interval Monte-Carlo error is COMMON, not "
                            "independent, within a length class; pooling.py and "
                            "m14_meta.py both aggregate those intervals as though "
                            "it were independent"),
        },
        "cost_of_applying": {
            "reason": ("it moves every published M3 confidence interval, and M3 "
                       "feeds the spine's model_selection block, pooling and the "
                       "meta-analysis"),
            "rebuild_chain": ["m2_fit", "m3_select", "m4_features", "m5_classify",
                              "m6_propensity", "m7_assemble", "m8_reachability",
                              "m9_recommender", "pooling", "m14_meta",
                              "m17_explain"],
            "how_it_was_made_tractable": (
                "single-threaded M3 measured ~3.9 h at B=150. m3_select gained "
                "`--shard i --shards n`, sound because curves are fitted "
                "independently and the seed is now per-curve, so there is no "
                "cross-curve state to serialise. The merge step restores the exact "
                "single-process record order."),
        },
        "effect_when_applied": ("each curve keeps a reproducible stream of its own, "
                                "so equal-length curves no longer share resample "
                                "indices and downstream aggregation over per-curve "
                                "intervals stops treating common noise as "
                                "independent"),
    },
}

APPLY_RECORD = {
    "recorded_at": "2026-08-18",
    "by": "governed apply step",
    "rationale": ("two estimation defects that made published numbers "
                  "unattributable rather than merely imprecise: M3's constant seed "
                  "made bootstrap Monte-Carlo error common across 98.4% of curves "
                  "while pooling and the meta-analysis aggregate those intervals "
                  "as though it were independent; and 9 of 13 registry models "
                  "received a single optimisation start, so their fits could not "
                  "be distinguished from local optima and the basin gate was "
                  "`not_evaluable` for them."),
    "source_artifact": SOURCE_ARTIFACT,
    "bootstrap_policy_version": BOOTSTRAP_POLICY_VERSION,
    "applied": True,
    "reversible": True,
    "note": ("m2_fit and m3_select READ this policy rather than hard-coding "
             "either behaviour, so each correction stays one value from its "
             "predecessor and cannot drift from the record describing it."),
}


def series_seed(series_id, base: int = 0) -> int:
    """A deterministic, platform-stable per-series bootstrap seed.

    BLAKE2b over the id rather than the builtin `hash()`, which is salted per
    process (PYTHONHASHSEED) and would make the "deterministic" guarantee false
    across runs — the exact failure this project's byte-identity checks exist to
    catch. Truncated to 63 bits so it is a valid numpy seed on every platform.

    `base` lets a caller offset the whole corpus's streams reproducibly (e.g. to
    run a genuinely independent replicate) without reintroducing a shared stream.
    """
    digest = hashlib.blake2b(f"{base}:{series_id}".encode("utf-8"),
                             digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1


def applied_policy(name: str, default=None):
    """The behaviour currently IN FORCE (not the proposed one)."""
    rec = BOOTSTRAP_POLICY.get(name)
    if not rec:
        return default
    val = rec.get("applied")
    return default if val is None else val


def previous_policy(name: str, default=None):
    """The retained PREVIOUS behaviour, so a revert is a value change and never a
    rewrite. Both branches of both decisions are exercised by the tests."""
    rec = BOOTSTRAP_POLICY.get(name)
    if not rec:
        return default
    val = rec.get("previous")
    return default if val is None else val


def policy_record(name: str) -> dict:
    return dict(BOOTSTRAP_POLICY.get(name, {}))


def manifest() -> dict:
    return {
        "bootstrap_policy_version": BOOTSTRAP_POLICY_VERSION,
        "source_artifact": SOURCE_ARTIFACT,
        "bootstrap_policy": {k: dict(v) for k, v in BOOTSTRAP_POLICY.items()},
        "apply_record": dict(APPLY_RECORD),
    }


if __name__ == "__main__":   # pragma: no cover - human-readable dump
    import json
    print(json.dumps(manifest(), indent=2))
