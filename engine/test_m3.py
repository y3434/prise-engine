"""
Tests for M3 — bootstrap-enveloped selection + identifiability.

    python engine/test_m3.py
    pytest engine/test_m3.py
"""
from __future__ import annotations

import numpy as np

from m2_fit import fit_one
from m3_select import (
    _mammen,
    akaike_weights,
    analyze_curve,
    fim_identifiability,
    resolve_bootstrap_seed,
    t50_of,
)
from models import m_logistic

T = np.arange(0, 40, 1.0)


def _series(y, handling="monotonic_fit", sid="syn"):
    return {"series_id": sid, "x_hours": list(map(float, T)),
            "m1": {"recommended_handling": handling, "y_processed": list(y)}}


def _noisy(yc, sd=0.01, seed=0):
    # local RNG -> tests are deterministic and order-independent
    rng = np.random.default_rng(seed)
    return list(np.asarray(yc) + rng.normal(0, sd, size=len(yc)))


# ------------------------------ functionals -------------------------------- #
def test_t50_of_logistic_equals_t0():
    params = {"base": 0.0, "amp": 1.0, "k": 0.5, "t0": 18.0}
    assert abs(t50_of("logistic", params, T) - 18.0) < 0.3


def test_mammen_weights_moments():
    w = _mammen(200000, np.random.default_rng(0))
    assert abs(w.mean()) < 0.02 and abs(w.var() - 1.0) < 0.05


def test_akaike_weights_sum_to_one():
    converged = {"a": {"aicc": 100.0}, "b": {"aicc": 102.0}, "c": {"aicc": 110.0}}
    w = akaike_weights(converged)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["a"] > w["b"] > w["c"]               # lower AICc -> higher weight


# ------------------------------ identifiability ---------------------------- #
def test_extra_parameter_is_sloppier():
    """Richards (5 params) on logistic data has a loose nu -> worse conditioning
    than the logistic (4 params) fit of the same data."""
    y = _noisy(m_logistic(T, 0, 1, 0.4, 20), sd=0.01, seed=42)
    rl = fit_one(list(T), y, "logistic")
    rr = fit_one(list(T), y, "richards")
    il = fim_identifiability("logistic", rl["params"], T, y)
    ir = fim_identifiability("richards", rr["params"], T, y)
    assert il["condition_number"] is not None and ir["condition_number"] is not None
    assert ir["condition_number"] > il["condition_number"]   # extra param -> more collinear
    assert il["flag"] == "identifiable"                       # a clean logistic IS identifiable


def test_identifiability_reports_structure():
    y = _noisy(m_logistic(T, 0, 1, 0.4, 20), seed=7)
    r = fit_one(list(T), y, "logistic")
    ident = fim_identifiability("logistic", r["params"], T, y)
    assert ident["flag"] in ("identifiable", "practically_non_identifiable")
    assert set(ident["sloppiest_direction"]) == {"base", "amp", "k", "t0"}


def test_identifiability_carries_honest_statistic_clarifier():
    """HONESTY (non-breaking): the verdict carries a `statistic` clarifier saying it
    is the sensitivity-collinearity condition number, NOT the Fisher Information
    Matrix — while keeping every legacy key (condition_number, flag, ...) intact."""
    y = _noisy(m_logistic(T, 0, 1, 0.4, 20), seed=7)
    r = fit_one(list(T), y, "logistic")
    ident = fim_identifiability("logistic", r["params"], T, y)
    # the new clarifier fields
    assert ident["statistic"] == "sensitivity_collinearity_condition_number"
    assert "NOT the Fisher-information" in ident["note"]
    # non-breaking: the legacy keys consumers depend on are still present
    for k in ("flag", "condition_number", "sloppiest_direction"):
        assert k in ident


# --------------------------- bootstrap envelope ---------------------------- #
def test_analyze_clean_logistic():
    y = _noisy(m_logistic(T, 0, 1, 0.5, 20), sd=0.01, seed=11)
    res = analyze_curve(_series(y), B=80, seed=1)
    assert res["status"] == "ok"
    # the winner is a sigmoid family member
    assert res["best_point"] in ("logistic", "gompertz", "richards")
    sel = res["selection"]
    assert abs(sum(sel["selection_frequencies"].values()) - 1.0) < 0.02  # rounded freqs
    # t50 predictive interval is tight around the true t50 (=20)
    lo, med, hi = sel["t50_predictive_interval"]
    assert lo - 0.5 <= 20.0 <= hi + 0.5 and abs(med - 20.0) < 1.5


def test_analyze_outputs_tiers_and_weights():
    y = _noisy(m_logistic(T, 0, 1, 0.5, 20), seed=13)
    res = analyze_curve(_series(y), B=60, seed=2)
    assert res["best_descriptive"] is not None
    assert abs(sum(res["akaike_weights"].values()) - 1.0) < 0.02  # rounded weights
    assert "selection_stability" in res["selection"]
    assert isinstance(res["selection"]["t50_multimodal"], bool)
    assert res["ci_reliable"] is True             # 40 points


def test_no_converged_fit_handled():
    res = analyze_curve(_series([0, 1], handling="reject"), B=10)
    # reject -> no candidates -> no converged fit
    assert res["status"] in ("no_converged_fit",)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


# ===== the GOVERNED bootstrap RNG policy (bootstrap_policy.py) ============== #
# THE DEFECT, measured: analyze_curve took seed=0 and the driver never overrode
# it, so every curve drew its wild-bootstrap resamples from default_rng(0). 1,627
# of the 1,654 curves carrying an x grid -- 98.4% -- share their length with
# another curve, and equal length + equal seed means byte-IDENTICAL resample index
# matrices. Determinism was preserved; independence was not, and pooling.py and
# m14_meta.py both aggregate per-curve intervals as though it were.
#
# The correction is implemented and READ from the policy, but is NOT in force:
# flipping it moves every published M3 interval and needs a ~5.2 h rebuild first.
# Both branches are tested, so the unapplied path is not untested code.

def test_bootstrap_seed_policy_in_force_is_per_series():
    import bootstrap_policy as bp
    assert bp.applied_policy("M3_SEED_STREAM") == "per_series_derived"
    assert bp.previous_policy("M3_SEED_STREAM") == "constant_zero"
    # distinct, stable, and never the old shared stream
    seeds = {sid: resolve_bootstrap_seed(sid)
             for sid in ("CPAD-TK-1041", "CPAD-TK-2530", "CPAD-TK-9999")}
    assert len(set(seeds.values())) == 3, seeds
    assert all(v != 0 for v in seeds.values()), seeds
    assert resolve_bootstrap_seed("CPAD-TK-1041") == seeds["CPAD-TK-1041"]
    # an EXPLICIT seed still wins, which the --series path and fixtures rely on
    assert resolve_bootstrap_seed("CPAD-TK-1041", 7) == 7


def test_the_shipped_corpus_actually_carries_the_per_series_seeds():
    """The policy is only real if it reached the artifact. Every M3 record must
    carry the seed its own id derives, and no two may share one."""
    import json
    from pathlib import Path
    import bootstrap_policy as bp
    art = Path(__file__).resolve().parent.parent / "data/processed/m3_sample.jsonl"
    if not art.exists():
        return
    seeds = {}
    for line in art.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        prov = (r.get("selection") or {}).get("bootstrap_provenance") or {}
        seeds[r["series_id"]] = prov.get("rng_seed")
    assert seeds, "no M3 records"
    assert all(v is not None for v in seeds.values()), "a record carries no seed"
    assert len(set(seeds.values())) == len(seeds), "two curves share a seed"
    assert all(v == bp.series_seed(sid) for sid, v in seeds.items())


def test_series_seed_is_deterministic_and_not_the_salted_builtin_hash():
    """The correction is only a correction if the stream is reproducible ACROSS
    PROCESSES. Python's builtin hash() is salted by PYTHONHASHSEED, so using it
    would have traded a shared-stream defect for a non-determinism defect --
    exactly what this project's byte-identity checks exist to catch. Pinned to a
    literal so a change of digest cannot silently move every published interval."""
    import bootstrap_policy as bp
    assert bp.series_seed("CPAD-TK-1041") == 4344273415487201371
    assert bp.series_seed("CPAD-TK-1041") == bp.series_seed("CPAD-TK-1041")
    assert bp.series_seed("CPAD-TK-1041") != bp.series_seed("CPAD-TK-1042")
    assert 0 <= bp.series_seed("x") < 2 ** 63       # valid numpy seed everywhere
    assert bp.series_seed("a", base=1) != bp.series_seed("a", base=0)


def test_equal_length_curves_no_longer_share_resample_indices():
    """THE DEFECT AND THE FIX, on the thing that actually matters: the resample
    INDICES two equal-length curves receive.

    Before, every curve drew from default_rng(0), so equal length meant a
    byte-identical index matrix -- and 98.4% of the corpus shares its length with
    another curve. Both branches are exercised so the retained `previous` is not
    untested code."""
    import numpy as np
    import bootstrap_policy as bp
    import m3_select as m3

    def indices(sid, n=11, B=150):
        rng = np.random.default_rng(m3.resolve_bootstrap_seed(sid))
        return rng.integers(0, n, size=(B, n))

    # IN FORCE: decoupled, but each curve still perfectly reproducible
    a, b = indices("CPAD-TK-1041"), indices("CPAD-TK-1042")
    assert not np.array_equal(a, b)
    assert np.array_equal(a, indices("CPAD-TK-1041"))

    # the retained previous behaviour reproduces the defect exactly
    real = m3._bootstrap_policy
    try:
        m3._bootstrap_policy = lambda name, default=None: bp.previous_policy(
            name, default)
        assert np.array_equal(indices("CPAD-TK-1041"), indices("CPAD-TK-1042"))
    finally:
        m3._bootstrap_policy = real
    assert not np.array_equal(indices("CPAD-TK-1041"), indices("CPAD-TK-1042"))


def test_block_length_is_a_frozen_function_of_n_alone():
    """The block length must NOT be chosen from the data. A block picked by
    looking at the estimated autocorrelation would make the published interval
    depend on a tuning parameter selected from the very residuals it is applied
    to -- the post-selection trap this module fights everywhere else. It is
    ceil(n**(1/3)), the standard block-bootstrap rate, and nothing else."""
    from m3_select import block_length
    assert block_length(8) == 2
    assert block_length(27) == 3
    assert block_length(64) == 4
    assert block_length(1) == 1 and block_length(0) == 1     # never zero
    # monotone non-decreasing in n, and a pure function of n
    prev = 0
    for n in range(1, 200):
        b = block_length(n)
        assert b >= prev and b >= 1
        prev = b


def test_block_mammen_preserves_mammen_moments_and_within_block_dependence():
    """The blocked weights must still be a valid wild bootstrap (E=0, Var=1) --
    blocking changes the DEPENDENCE, not the marginal. And the whole point is
    that the weight is constant WITHIN a block and independent ACROSS blocks."""
    import numpy as np
    from m3_select import _block_mammen

    w = _block_mammen(12, np.random.default_rng(7), 3)
    assert len(w) == 12
    for s in range(0, 12, 3):
        assert len(set(np.round(w[s:s + 3], 12))) == 1, w   # constant in-block

    rng = np.random.default_rng(11)
    vals = np.concatenate([_block_mammen(9, rng, 3) for _ in range(20000)])
    assert abs(float(vals.mean())) < 0.02, vals.mean()      # E = 0
    assert abs(float(vals.var()) - 1.0) < 0.02, vals.var()  # Var = 1

    rng = np.random.default_rng(3)
    W = np.array([_block_mammen(9, rng, 3) for _ in range(5000)])
    assert float(np.corrcoef(W[:, 0], W[:, 1])[0, 1]) > 0.99   # same block
    assert abs(float(np.corrcoef(W[:, 0], W[:, 3])[0, 1])) < 0.1   # different

    # block=1 must be EXACTLY the pointwise sampler, so the retained `previous`
    # behaviour is reproduced bit-for-bit and not merely approximated
    from m3_select import _mammen
    a = _block_mammen(20, np.random.default_rng(5), 1)
    b = _mammen(20, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_blocking_policy_records_the_autocorrelation_it_acted_on():
    """The defect this corrects is invisible unless measured, so the governed
    record has to carry the measurement rather than assert the conclusion."""
    import bootstrap_policy as bp
    rec = bp.policy_record("M3_BOOTSTRAP_BLOCKING")
    assert rec["previous"] == "pointwise"
    assert rec["applied"] == "blockwise_n_cbrt"
    ev = rec["evidence"]
    assert ev["n_curves_measured"] == 1504
    assert ev["median_lag1_autocorrelation"] == 0.182
    assert ev["fraction_dw_lt_1_0"] == 0.303
    assert "TOO" in ev["consequence"] and "NARROW" in ev["consequence"]


def test_the_shipped_intervals_were_built_blockwise_and_say_so():
    """Provenance must record the block length PER RECORD, because unlike the
    resampler description it is a function of this curve's own n and therefore
    genuinely varies -- it cannot be resolved through the recipe alone."""
    import json
    from pathlib import Path
    from m3_select import block_length
    art = Path(__file__).resolve().parent.parent / "data/processed/m3_sample.jsonl"
    if not art.exists():
        return
    n = 0
    for line in art.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        prov = (r.get("selection") or {}).get("bootstrap_provenance") or {}
        assert prov.get("recipe") == "m3/blockwise-wild-bootstrap-mammen-2.0", prov
        bl = prov.get("block_length")
        assert isinstance(bl, int) and bl >= 1, prov
        n += 1
    assert n > 0


def test_bootstrap_policy_record_states_the_measurement_it_acted_on():
    """A governed change is only auditable if it records what it measured. This
    pins the coupling figure that justified it and the retained previous value,
    so the record cannot decay into an assertion."""
    import bootstrap_policy as bp
    rec = bp.policy_record("M3_SEED_STREAM")
    assert rec["applied"] == "per_series_derived"
    assert rec["previous"] == "constant_zero"
    assert rec["status"] == "applied"
    ev = rec["evidence"]
    assert ev["n_coupled"] == 1627 and ev["n_curves_with_a_grid"] == 1654
    assert ev["fraction_coupled"] == 0.984
    assert bp.APPLY_RECORD["applied"] is True
    assert bp.APPLY_RECORD["reversible"] is True
    # the M2 half of the same governed change
    m2 = bp.policy_record("M2_START_TOPUP")
    assert m2["previous"] == "name_keyed_only"
    assert m2["applied"] == "spread_within_bounds"
    assert m2["evidence"]["n_models_single_start_before"] == 9
    # and the version axis must be folded into the build id, or a change to how
    # the corpus is FITTED would ship under an unchanged engine_build_id
    import m7_assemble
    comps = m7_assemble.build_components("k", "s", "cv", "h", "env")
    assert comps.get("estimation_policy_version") == bp.BOOTSTRAP_POLICY_VERSION


if __name__ == "__main__":
    raise SystemExit(_run())
