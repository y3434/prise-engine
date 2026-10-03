"""
Tests for the PRISE deterministic empirical-Bayes partial-pooling layer
(engine/pooling.py). Fast (<~1s; no fitting, no I/O of real artifacts) and fully
deterministic — every assertion is a closed-form property of the shrinkage math
or the DerSimonian–Laird estimator.

Proves the §4 claims the layer makes:
  * the down-weight IS w = τ²/(τ²+s²) — the frozen, versioned, monotone statistic;
  * a noisy estimate in a TIGHT stratum shrinks toward the mean (|pooled−μ| < |raw−μ|);
  * a PRECISE estimate barely moves (w≈1);
  * DerSimonian–Laird τ² recovers a PLANTED between-group variance;
  * replicate random-effects pooled SE < individual SE (when scatter is sampling-led);
  * large planted replicate t50 scatter sets the stochastic-nucleation flag;
  * a stratum with n<3 is NOT pooled (w=1, flagged);
  * byte-identical determinism.

    python engine/test_pooling.py
    pytest engine/test_pooling.py
"""
from __future__ import annotations

import json
import math

import pooling as P


# --------------------------- helpers --------------------------------------- #
def _curve(series_id, value, sd, source="m4_bootstrap_ci",
           uniprot="P1", construct="Wild Type", assay="ThT", pH=7.4, temp=37.0,
           conc=10.0):
    """A minimal protein_analysis-shaped record carrying one POINT t50 + a known
    sampling SD encoded as a symmetric log10 predictive interval so the loader picks
    it up deterministically (no fitting)."""
    lo = 10 ** (math.log10(value) - 1.959963984540054 * sd)
    hi = 10 ** (math.log10(value) + 1.959963984540054 * sd)
    return {
        "series_id": series_id, "protein_id": "Prot", "uniprot_id": uniprot,
        "curve_features": {"status": "ok", "t50_status": "point",
                           "features": {"t50": value}},
        "model_selection": {"t50_predictive_interval": [lo, value, hi]},
        "condition_vector": {"assay_type": assay, "pH": pH, "temperature_C": temp,
                             "construct_id": construct,
                             "concentration": {"value_uM": conc}},
    }


# --------------------------- 0. determinism -------------------------------- #
def test_determinism_byte_identical():
    recs = [_curve(f"S{i}", v, 0.1)
            for i, v in enumerate([5.0, 6.0, 7.0, 50.0, 5.5])]
    a = json.dumps(P.build_pooling(recs), sort_keys=True)
    b = json.dumps(P.build_pooling(recs), sort_keys=True)
    assert a == b


# --------------------------- 1. the frozen weight -------------------------- #
def test_weight_is_tau2_over_tau2_plus_s2():
    """THE §4 invariant: shrinkage weight = τ²/(τ²+s²), exactly."""
    for tau2, s2 in [(0.04, 0.01), (0.01, 0.04), (0.1, 0.1), (1e-6, 0.5)]:
        sh = P.shrink(theta_hat=2.0, s2=s2, mu=1.0, tau2=tau2)
        assert abs(sh["weight"] - tau2 / (tau2 + s2)) < 1e-15


def test_weight_monotone_decreasing_in_s2():
    """w is monotone DECREASING in the named statistic s² (a noisier estimate is
    pulled harder toward the mean) — the monotonicity §4 requires."""
    tau2 = 0.05
    ws = [P.shrink(2.0, s2, 1.0, tau2)["weight"]
          for s2 in [0.001, 0.01, 0.05, 0.2, 1.0]]
    assert all(ws[i] > ws[i + 1] for i in range(len(ws) - 1))


def test_weight_in_unit_interval():
    for s2 in [1e-9, 0.01, 1.0, 100.0]:
        w = P.shrink(2.0, s2, 1.0, 0.03)["weight"]
        assert 0.0 <= w <= 1.0


# --------------------------- 2. shrinkage behaviour ------------------------ #
def test_noisy_estimate_in_tight_stratum_shrinks_toward_mean():
    """A noisy outlier in an otherwise tight stratum is pulled toward μ:
    |pooled − μ| < |raw − μ|."""
    # 4 precise curves near t50≈5h + 1 noisy outlier at 50h
    recs = ([_curve(f"T{i}", 5.0 + 0.2 * i, 0.05) for i in range(4)]
            + [_curve("OUT", 50.0, 0.5)])   # big s² outlier
    out = P.build_pooling(recs)
    rec = next(r for r in out["curve_records"] if r["series_id"] == "OUT")
    mu = rec["mu_stratum"]
    assert abs(rec["theta_pooled"] - mu) < abs(rec["theta_raw"] - mu)
    assert rec["moved"] > 0.0
    assert rec["shrinkage_weight"] < 0.9          # genuinely down-weighted


def test_precise_estimate_barely_moves():
    """A precise estimate (tiny s²) keeps its own value: w≈1, moved≈0. The stratum
    must carry REAL between-curve spread (τ²>0) — values 2h..40h with modest sampling
    noise — so the shrinkage is governed by s²/τ², and the precise SHARP curve (whose
    value sits away from μ) keeps its own value via w≈1."""
    recs = ([_curve(f"P{i}", v, 0.1)
             for i, v in enumerate([2.0, 4.0, 8.0, 20.0, 40.0])]      # wide τ²>0 stratum
            + [_curve("SHARP", 6.0, 0.0005)])                         # very precise
    out = P.build_pooling(recs)
    rec = next(r for r in out["curve_records"] if r["series_id"] == "SHARP")
    assert rec["tau2"] > 0.0                          # genuine between-curve spread
    assert rec["shrinkage_weight"] > 0.99
    assert rec["moved"] < 0.01


def test_shrinkage_toward_not_past_mean():
    """A shrunken estimate lands BETWEEN the raw value and μ (convex combination)."""
    recs = [_curve(f"C{i}", v, 0.1)
            for i, v in enumerate([4.0, 5.0, 6.0, 30.0])]
    out = P.build_pooling(recs)
    for r in out["curve_records"]:
        lo, hi = sorted([r["theta_raw"], r["mu_stratum"]])
        assert lo - 1e-9 <= r["theta_pooled"] <= hi + 1e-9


# --------------------------- 3. DerSimonian–Laird τ² ----------------------- #
def test_dl_recovers_planted_between_group_variance():
    """DL τ² recovers a PLANTED between-group variance. We plant true means with a
    known spread τ²_true and tiny sampling variance; DL τ̂² should land near τ²_true."""
    import random
    rng = random.Random(0)
    tau_true = 0.5
    s = 0.02
    thetas, s2 = [], []
    for _ in range(200):
        true_mean = rng.gauss(0.0, tau_true)        # planted between-group spread
        thetas.append(true_mean + rng.gauss(0.0, s))
        s2.append(s * s)
    dl = P.dersimonian_laird(thetas, s2)
    # τ̂² should recover τ²_true = 0.25 within moment-estimator slack
    assert abs(dl["tau2"] - tau_true ** 2) < 0.08
    assert abs(dl["mu"]) < 0.1                       # μ ≈ 0 (the planted grand mean)


def test_dl_zero_tau2_when_all_sampling_noise():
    """When dispersion is fully explained by sampling variance, τ̂²→0 (Q≈k−1)."""
    import random
    rng = random.Random(1)
    s = 0.3
    thetas = [3.0 + rng.gauss(0.0, s) for _ in range(150)]   # one true mean
    s2 = [s * s] * len(thetas)
    dl = P.dersimonian_laird(thetas, s2)
    assert dl["tau2"] < 0.02                         # ≈ 0
    assert abs(dl["mu"] - 3.0) < 0.1


def test_dl_clamps_tau2_nonnegative():
    dl = P.dersimonian_laird([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert dl["tau2"] >= 0.0


# --------------------------- 4. replicate meta-analysis -------------------- #
def test_replicate_pooled_se_lt_individual_when_sampling_led():
    """Random-effects pooled SE < the mean individual sampling SD when the scatter is
    sampling-led (small τ²) — combining replicates sharpens the estimate (§3-M1)."""
    # 6 tight replicates (identical condition) → low between-replicate variance
    recs = [_curve(f"R{i}", 10.0 + 0.05 * i, 0.2, conc=10.0) for i in range(6)]
    out = P.build_pooling(recs)
    assert len(out["replicate_meta"]) == 1
    m = out["replicate_meta"][0]
    assert m["n_replicates"] == 6
    assert m["pooled_se_lt_individual"] is True
    assert m["se"] < m["mean_individual_sampling_sd"]


def test_large_replicate_scatter_sets_stochastic_flag():
    """Large planted between-replicate t50 SCATTER sets the stochastic-nucleation
    flag (§3-M1: primary/stochastic-nucleation-dominated)."""
    # replicate t50s span 4h..60h at one identical condition → CV well above 0.5
    vals = [4.0, 6.0, 10.0, 25.0, 60.0]
    recs = [_curve(f"N{i}", v, 0.1, conc=10.0) for i, v in enumerate(vals)]
    out = P.build_pooling(recs)
    m = out["replicate_meta"][0]
    assert m["stochastic_nucleation_flag"] is True
    cv = m["stochastic_nucleation_signal"]["between_replicate_t50_cv"]
    assert cv > P.STOCHASTIC_NUCLEATION_CV


def test_tight_replicates_no_stochastic_flag():
    recs = [_curve(f"Q{i}", 10.0 + 0.1 * i, 0.1, conc=10.0) for i in range(5)]
    out = P.build_pooling(recs)
    m = out["replicate_meta"][0]
    assert m["stochastic_nucleation_flag"] is False


def test_replicates_retained_not_averaged():
    """Replicate scatter is RETAINED as a signal, not collapsed: the member ids and
    the per-replicate CV are reported."""
    recs = [_curve(f"M{i}", v, 0.1, conc=10.0) for i, v in enumerate([5.0, 8.0, 40.0])]
    out = P.build_pooling(recs)
    m = out["replicate_meta"][0]
    assert len(m["member_series_ids"]) == 3
    assert m["stochastic_nucleation_signal"]["between_replicate_t50_cv"] is not None


# --------------------------- 5. small-stratum gate ------------------------- #
def test_stratum_too_small_no_pooling():
    """A stratum with n<MIN_STRATUM_N is NOT pooled: w=1, raw==pooled, flagged."""
    recs = [_curve("A", 5.0, 0.1, conc=10.0),
            _curve("B", 50.0, 0.1, conc=20.0)]    # 2 distinct concentrations, n=2
    out = P.build_pooling(recs)
    assert all(r["shrinkage_weight"] == 1.0 for r in out["curve_records"])
    assert all("stratum_too_small" in r["flags"] for r in out["curve_records"])
    assert all(abs(r["theta_pooled"] - r["theta_raw"]) < 1e-12
               for r in out["curve_records"])


def test_no_cross_condition_pooling():
    """Two curves with the SAME protein but DIFFERENT pH land in DIFFERENT strata —
    no cross-condition pooling (Service A discipline). With one curve each, neither
    is pooled."""
    recs = ([_curve(f"X{i}", 5.0 + i, 0.1, pH=7.4, conc=10.0 + i) for i in range(3)]
            + [_curve("Y0", 99.0, 0.1, pH=5.0, conc=10.0)])
    out = P.build_pooling(recs)
    y = next(r for r in out["curve_records"] if r["series_id"] == "Y0")
    # Y0 is alone in its pH=5.0 stratum → not pooled, untouched by the pH=7.4 mean
    assert "stratum_too_small" in y["flags"]
    assert abs(y["theta_pooled"] - y["theta_raw"]) < 1e-12


# --------------------------- 6. provenance + honesty ----------------------- #
def test_default_variance_flagged():
    """A curve with no bootstrap interval gets the documented default s², FLAGGED."""
    rec = {"series_id": "D", "protein_id": "Prot", "uniprot_id": "P1",
           "curve_features": {"status": "ok", "t50_status": "point",
                              "features": {"t50": 5.0}},
           "condition_vector": {"assay_type": "ThT", "pH": 7.4,
                                "temperature_C": 37.0, "construct_id": "Wild Type",
                                "concentration": {"value_uM": 10.0}}}
    recs = [rec] + [_curve(f"E{i}", 5.0 + i, 0.1, conc=20.0 + i) for i in range(3)]
    out = P.build_pooling(recs)
    d = next(r for r in out["curve_records"] if r["series_id"] == "D")
    assert d["s2_source"] == "default"
    assert "sampling_variance_defaulted" in d["flags"]
    assert abs(d["s2"] - P.DEFAULT_LOG10_SD ** 2) < 1e-12


def test_censored_t50_excluded():
    """A right-censored (lower-bound) t50 is a ≥-inequality, not a point — excluded
    from the pool (consistent with M6)."""
    rec = _curve("CEN", 5.0, 0.1)
    rec["curve_features"]["t50_status"] = "lower_bound"
    recs = [rec] + [_curve(f"OK{i}", 5.0 + i, 0.1, conc=20.0 + i) for i in range(3)]
    out = P.build_pooling(recs)
    assert all(r["series_id"] != "CEN" for r in out["curve_records"])
    assert out["summary"]["n_censored_excluded"] >= 1


def test_version_stamped():
    out = P.build_pooling([_curve("Z", 5.0, 0.1)])
    assert out["pooling_version"] == P.POOLING_VERSION
    for r in out["curve_records"]:
        assert r["pooling_version"] == P.POOLING_VERSION


def test_log10_back_transform():
    """pooled_value is the back-transform of theta_pooled (10**θ on the log scale)."""
    recs = [_curve(f"L{i}", v, 0.1) for i, v in enumerate([4.0, 5.0, 6.0, 40.0])]
    out = P.build_pooling(recs)
    for r in out["curve_records"]:
        assert abs(r["pooled_value"] - 10 ** r["theta_pooled"]) < 1e-9


# --------------------------- 7. assembler shape ---------------------------- #
def test_build_pooling_shape():
    recs = [_curve(f"W{i}", v, 0.1) for i, v in enumerate([4.0, 5.0, 6.0, 7.0, 40.0])]
    out = P.build_pooling(recs)
    assert out["status"] == "ok"
    for key in ("curve_records", "stratum_summaries", "replicate_meta", "summary"):
        assert key in out
    s = out["summary"]
    assert s["n_poolable_point"] == 5
    assert s["median_shrinkage_weight"] is not None
    assert s["biggest_shrinkage"]["series_id"] == "W4"   # the 40h outlier moves most


def test_unknown_target_errors_gracefully():
    out = P.build_pooling([_curve("Z", 5.0, 0.1)], target="bogus")
    assert out["status"] == "error" and "bogus" in out["reason"]


# ------------------------------- runner ---------------------------------- #
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
        except Exception as e:                       # noqa
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
