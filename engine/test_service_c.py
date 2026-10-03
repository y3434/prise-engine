"""
Tests for Service C — synthetic calibration & identifiability engine.

Deterministic + fast (<~30s total): small n_trials, all seeded.

    python engine/test_service_c.py
    pytest engine/test_service_c.py
"""
from __future__ import annotations

import json

import numpy as np

import service_c as sc


# --------------------------- 1. generator -------------------------------- #
def test_generator_determinism_byte_identical():
    """Same seed -> byte-identical curve (pre-registration + pinned RNG, §7)."""
    a = sc.generate_curve({"generator": "closed_form", "model": "logistic"}, seed=123)
    b = sc.generate_curve({"generator": "closed_form", "model": "logistic"}, seed=123)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_generator_different_seeds_differ():
    a = sc.generate_curve({"generator": "closed_form", "model": "logistic"}, seed=1)
    b = sc.generate_curve({"generator": "closed_form", "model": "logistic"}, seed=2)
    assert a["y_obs"] != b["y_obs"]      # noise differs across seeds


def test_clean_logistic_recovers_its_t50():
    """Ground-truth correctness: a clean logistic's reported t50_true matches its t0."""
    c = sc.generate_curve(
        {"generator": "closed_form", "model": "logistic",
         "params": {"base": 0.0, "amp": 1.0, "k": 0.4, "t0": 20.0},
         "noise": False, "drift": False, "digitize": False}, seed=0)
    assert abs(c["ground_truth"]["t50_true"] - 20.0) < 0.5


def test_make_series_left_censor_yields_left_class():
    """A left-censored spec (cut the lead, t0 mid-window) -> M1 censoring_class 'left'."""
    c = sc.generate_curve(
        {"generator": "closed_form", "model": "logistic",
         "params": {"base": 0.0, "amp": 1.0, "k": 0.4, "t0": 15.0},
         "censor": "left"}, seed=7)
    s = sc.make_series(c)
    assert s["m1"]["censoring_class"] in ("left", "left_right")


def test_make_series_right_censor_yields_right_class():
    """A right-censored spec (late t0, cut the tail) -> M1 censoring_class 'right'."""
    c = sc.generate_curve(
        {"generator": "closed_form", "model": "logistic",
         "params": {"base": 0.0, "amp": 1.0, "k": 0.4, "t0": 28.0},
         "censor": "right"}, seed=7)
    s = sc.make_series(c)
    assert s["m1"]["censoring_class"] in ("right", "left_right")


def test_make_series_shape_consumable_by_engine():
    """make_series produces an m1 block with the keys M2/M4/M5 read."""
    c = sc.generate_curve({"generator": "closed_form", "model": "logistic"}, seed=3)
    s = sc.make_series(c)
    assert "m1" in s and "y_processed" in s["m1"]
    assert s["m1"]["fittability_class"] in ("fittable", "suspect", "unfittable")
    assert len(s["x_hours"]) == len(s["y_intensity"])


def test_mechanistic_generator_monotone_rising():
    c = sc.generate_curve({"generator": "mechanistic",
                           "variant": "secondary_nucleation",
                           "noise": False}, seed=5)
    y = c["y_true"]
    assert y[-1] > y[0] and 0.0 <= y[0] <= 1.0


# --------------------------- 2. recovery --------------------------------- #
def test_recovery_bias_small_on_clean_logistic():
    """Recovery bias is small for a well-behaved model under the noise law."""
    r = sc.parameter_recovery("logistic", n_trials=15, seed=0)
    assert r["n_converged"] >= 12
    t0 = r["per_parameter"]["t0"]
    assert t0["bias"] is not None and abs(t0["bias"]) < 0.5      # << window length
    assert r["t50_recovery"]["rmse"] is not None and r["t50_recovery"]["rmse"] < 1.0


def test_recovery_coverage_near_nominal():
    r = sc.parameter_recovery("logistic", n_trials=15, seed=1)
    covs = [v["coverage_95"] for v in r["per_parameter"].values()
            if v["coverage_95"] is not None]
    assert covs and np.mean(covs) >= 0.6          # Wald CIs roughly cover (nominal 0.95)


# --------------------------- 3. confusion -------------------------------- #
def test_single_curve_confusion_row_stochastic():
    cm = sc.confusion_matrix("single_curve", n_trials=10, seed=0)
    for g, row in cm["matrix"].items():
        total = sum(row.values())
        assert abs(total - 1.0) < 1e-6           # each row sums to 1 (stochastic)


def test_single_curve_easy_regime_diagonal_dominant():
    """Distinct descriptive shapes (gompertz/exponential) are mostly recovered ->
    a diagonal-dominant matrix for an EASY regime."""
    cm = sc.confusion_matrix("single_curve", n_trials=10, seed=0)
    assert cm["mean_diagonal"] > 0.5


def test_known_degenerate_pair_grouped():
    """logistic and richards are a KNOWN-degenerate pair (Richards@nu=1 IS logistic)
    -> they land in the same equivalence class (off-diagonal-heavy)."""
    cm = sc.confusion_matrix("single_curve", n_trials=10, seed=0)
    classes = [set(c) for c in cm["equivalence_classes"]]
    assert any({"logistic", "richards"} <= c for c in classes)


def test_mechanistic_single_curve_is_degenerate():
    """Single-curve (single-concentration) Tier-B mechanisms are broadly confusable
    (K2) -> one broad class and a low mean diagonal. This is the BASELINE the
    concentration series must improve on."""
    cm = sc.confusion_matrix("single_concentration_mech",
                             generators=list(sc.MECH_SERIES_RATES.keys()),
                             n_trials=4, seed=0)
    assert cm["mean_diagonal"] < 0.7
    biggest = max(len(c) for c in cm["equivalence_classes"])
    assert biggest >= 3                          # mechanisms collapse together


# ------------------ 3b. concentration series (γ) — MUST-FIX 2 ------------- #
def test_concentration_series_generates_multiconcentration_family():
    """The concentration_series discriminator must actually build a MULTI-concentration
    family (not one single-concentration curve) and recover a γ per mechanism."""
    members = sc._generate_mech_family("secondary_nucleation", trial_seed=1)
    # one member per pre-registered concentration, each carrying its own concentration
    assert len(members) == len(sc.MECH_SERIES_CONCS_UM)
    concs = sorted(m["condition_vector"]["concentration"]["value_uM"] for m in members)
    assert concs == sorted(sc.MECH_SERIES_CONCS_UM)      # a real decade of concentrations
    assert len({m["concentration_series_id"] for m in members}) == 1   # one family id


def test_concentration_series_recovers_gamma_per_mechanism():
    """The γ-scaling discriminator recovers a per-mechanism γ over the family and reports
    equivalence classes + a pairwise separation AUC table."""
    cs = sc.concentration_series_gamma_confusion(n_trials=4, seed=0)
    assert cs["selector"] == "gamma_global_scaling_separability"
    gm = cs["gamma_by_mechanism"]
    # secondary_nucleation has the steepest γ-scaling and must be recovered with n>=1
    assert gm["secondary_nucleation"]["mean"] is not None
    assert gm["secondary_nucleation"]["n"] >= 1
    assert "pairwise_separation_auc" in cs and cs["equivalence_classes"]


def test_concentration_series_narrows_degeneracy_vs_single():
    """K2 headline: a clean concentration series NARROWS the degeneracy vs a single
    concentration. With this design γ separates the steep- vs shallow-scaling mechanisms,
    so the series partition has MORE classes than the single-concentration partition."""
    single = sc.confusion_matrix("single_concentration_mech",
                                 generators=list(sc.MECH_SERIES_RATES.keys()),
                                 n_trials=4, seed=0)
    series = sc.concentration_series_gamma_confusion(n_trials=5, seed=0)
    nar = sc._series_narrowing(single, series)
    # the series resolves at least one pair the single concentration pooled
    assert nar["series_narrows_degeneracy"] is True
    assert len(nar["pairs_newly_resolved_by_series"]) >= 1
    assert (nar["concentration_series_partition"]["n_classes"]
            > nar["single_concentration_partition"]["n_classes"])


def test_concentration_series_carries_separability_caveat():
    """HONESTY (non-breaking): the γ-separability result is partly a CONSTRUCTION
    artifact (MECH_SERIES_RATES were chosen to be separable), so the confusion
    output, the narrowing verdict, and the report root all carry an explicit
    upper-bound caveat — it must not be silently buried."""
    cs = sc.concentration_series_gamma_confusion(n_trials=4, seed=0)
    assert "UPPER BOUND" in cs["separability_caveat"]
    single = sc.confusion_matrix("single_concentration_mech",
                                 generators=list(sc.MECH_SERIES_RATES.keys()),
                                 n_trials=4, seed=0)
    nar = sc._series_narrowing(single, cs)
    assert "separability_caveat" in nar
    # when the series DOES narrow, the verdict text itself carries the caveat
    if nar["series_narrows_degeneracy"]:
        assert "CAVEAT" in nar["verdict"]


def test_report_surfaces_separability_caveat_at_root():
    """The validity-ceiling block at the report ROOT carries the separability caveat
    (so it is visible without digging into the confusion sub-tree)."""
    rep = sc.run_battery(n_trials=8, seed=0, mech_trials=4)
    vc = rep["validity_ceiling"]
    assert vc["gamma_separability_is_upper_bound"] is True
    assert "UPPER BOUND" in vc["separability_caveat"]
    # and the per-regime confusion output still carries it too
    assert "separability_caveat" in rep["confusion_matrices"]["concentration_series"]


# --------------------------- 4. identifiability -------------------------- #
def test_identifiability_clean_logistic_low_false_positive():
    iv = sc.identifiability_validation(n_trials=10, seed=0)
    # clean logistic should rarely be flagged non-identifiable
    assert iv["false_positive_rate"] is not None and iv["false_positive_rate"] <= 0.2


# --------------------------- 5. calibration ------------------------------ #
def test_fdr_null_realized_fdp_under_alpha():
    """FDR-null: realized false-discovery proportion <= nominal alpha + tolerance."""
    fn = sc.fdr_null_check(30, seed=1)
    assert fn["n_tested"] >= 10
    assert fn["realized_fdp"] <= fn["nominal_alpha"] + fn["tolerance"]
    assert fn["fdr_controlled"] is True


def test_fdr_null_has_power_arm_meaningful():
    """MUST-FIX 3: the FDR check reports a POWER arm — the runs test rejects on a
    bank-UNrepresentable curve (double-sigmoid) with power > 0 — so 'controlled' is
    MEANINGFUL (not just no-power). Both FDP<=alpha+tol AND power>floor must hold."""
    fn = sc.fdr_null_check(30, seed=1)
    # power arm exists and actually has power
    assert fn["n_alt_tested"] >= 10
    assert fn["power_at_alpha"] is not None and fn["power_at_alpha"] > 0.0
    assert fn["power_at_alpha"] > fn["power_floor"]      # bank-unrepresentable -> rejects
    # control is meaningful: FDP controlled AND power above the floor
    assert fn["realized_fdp"] <= fn["nominal_alpha"] + fn["tolerance"]
    assert fn["meaningful"] is True


def test_bank_unrepresentable_curve_triggers_runs_test():
    """The alternative-arm generator produces a curve the closed-form bank cannot fit,
    so residual_anomaly's runs test rejects it (low p), unlike a clean bank curve."""
    from m5_classify import residual_anomaly, ANOMALY_FDR_ALPHA
    curve = sc._bank_unrepresentable_curve(seed=3)
    s = sc.make_series(curve, series_id="alt-probe")
    a = residual_anomaly(s)
    assert a.get("testable") and a.get("p_too_few_runs") is not None
    assert a["p_too_few_runs"] < ANOMALY_FDR_ALPHA       # structured lack-of-fit detected


def test_ece_brier_bounded():
    dc = sc._descriptive_calibration(20, seed=0)
    assert 0.0 <= dc["ece"] <= 1.0 and 0.0 <= dc["brier"] <= 1.0


def test_threshold_recommendations_current_recommended_and_separation():
    """One sweep checks (a) current+recommended present, (b) current = the live
    module constant (Service C never edits it), (c) the lag-slope sweep separates
    lag vs no-lag classes well (held-out AUC high)."""
    tr = sc.threshold_recommendations(16, seed=0)
    for key in ("M1_lag_slope_frac", "M3_cond_number_nonident",
                "M5_threshold_sharpness"):
        assert "current" in tr[key] and "recommended" in tr[key]
    import m1_ingest
    assert tr["M1_lag_slope_frac"]["current"] == m1_ingest.LAG_SLOPE_FRAC
    auc = tr["M1_lag_slope_frac"]["separation_auc"]
    assert auc is None or auc >= 0.7


def test_threshold_recommendations_report_train_and_heldout_auc():
    """MUST-FIX 1: every separating-threshold rec reports BOTH a TRAIN and a HELD-OUT
    AUC (the headline separation_auc IS the held-out one), with disjoint train/test
    counts — so the in-sample optimism is visible and not reported as the headline."""
    tr = sc.threshold_recommendations(24, seed=0)
    for key in ("M1_lag_slope_frac", "M3_cond_number_nonident",
                "M5_threshold_sharpness"):
        rec = tr[key]
        assert "separation_auc_train" in rec and "separation_auc_heldout" in rec
        # headline separation_auc is the held-out one
        assert rec["separation_auc"] == rec["separation_auc_heldout"]
        # train and test are disjoint and both populated
        assert rec["n_train"] >= 2 and rec["n_test"] >= 2
        assert rec["n_train"] + rec["n_test"] <= rec["n"]


def test_heldout_split_exposes_leakage_on_pure_noise():
    """Leaky-by-construction check: when the feature carries NO real signal (pure noise
    vs random labels), the Youden-J threshold OVERFITS the train split -> train AUC is
    optimistic while the held-out AUC collapses toward chance. The honest split must
    therefore report heldout_auc < train_auc (the leakage the old in-sample AUC hid)."""
    rng = np.random.default_rng(0)
    n = 200
    values = rng.normal(0.0, 1.0, size=n).tolist()       # signal-free feature
    labels = rng.integers(0, 2, size=n).tolist()         # labels independent of values
    r = sc._best_separating_threshold(values, labels, positive_is_high=True,
                                       split_seed=1)
    assert r["auc_train"] is not None and r["auc"] is not None
    # in-sample AUC is inflated above chance; held-out collapses below it
    assert r["auc_train"] > 0.5
    assert r["auc"] < r["auc_train"]      # held-out <= train (leakage made visible)


# --------------------------- 6. de-circularization ----------------------- #
def test_mismatched_generator_returns_non_r2_degradation_metric():
    """MUST-FIX 4: the degradation metric is mechanism-identification accuracy, NOT R².
    The mechanism-id accuracy drops under the wrong fitter family while R² stays ~flat —
    so the headline degradation is positive and the R² delta is near zero."""
    mm = sc.mismatched_generator_check(n_trials=8, seed=0)
    assert mm["primary_metric"].startswith("mechanism-identification accuracy")
    assert mm["matched_mechanism_id_accuracy"] is not None
    assert mm["mismatched_mechanism_id_accuracy"] == 0.0      # closed-form names no mechanism
    # the non-R² degradation is a real positive number (mechanism information lost)
    assert mm["misidentification_degradation"] is not None
    assert mm["misidentification_degradation"] > 0.0
    # and R² (the OLD metric) barely moves — exactly why it was a bad metric
    r2d = mm["r2_secondary_diagnostic"]["degradation_r2"]
    assert r2d is not None and abs(r2d) < 0.1


# --------------------------- helper purity ------------------------------- #
def test_best_separating_threshold_picks_obvious_split():
    # perfectly separable -> both the train-chosen threshold and the HELD-OUT AUC are clean
    vals = [0.1, 0.12, 0.2, 0.15, 0.18, 0.9, 0.92, 1.0, 0.95, 0.97]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    r = sc._best_separating_threshold(vals, labels, positive_is_high=True, split_seed=0)
    assert r["recommended"] is not None and 0.2 < r["recommended"] < 0.9
    assert r["auc"] == 1.0          # held-out AUC (separable both sides of the split)
    assert r["auc_train"] == 1.0
    assert r["n_train"] >= 2 and r["n_test"] >= 2


def test_ece_perfect_calibration_zero():
    # predicted prob == empirical accuracy in every bin -> ECE 0
    probs = [0.9] * 10
    correct = [True] * 9 + [False]      # 90% correct at confidence 0.9
    e = sc._ece_brier(probs, correct, n_bins=10)
    assert e["ece"] < 0.05


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
