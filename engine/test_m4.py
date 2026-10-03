"""
Tests for M4 — feature extraction (definition contract), censoring-aware tags,
and dual-γ (censored regression + global master-curve collapse).

    python engine/test_m4.py
    pytest engine/test_m4.py
"""
from __future__ import annotations

import numpy as np

from m4_features import (
    DEFINITION_CONTRACT,
    _disagreement,
    dual_gamma,
    extract_features,
    feature_geometry,
    gamma_global,
    gamma_regression,
    series_t50_table,
)
from models import m_logistic

T = np.arange(0, 40, 1.0)


def _logistic_series(t0=20.0, k=0.5, cens="none", handling="monotonic_fit",
                     sid="syn", n=None, seed=0):
    x = T if n is None else np.linspace(0, 40, n)
    rng = np.random.default_rng(seed)
    y = m_logistic(x, 0.0, 1.0, k, t0) + rng.normal(0, 0.008, size=len(x))
    return {"series_id": sid, "x_hours": list(map(float, x)),
            "signal_basis": "baseline_subtracted",
            "condition_vector": {"concentration": {"value_uM": None}},
            "m1": {"recommended_handling": handling, "censoring_class": cens,
                   "plateau_reached": cens in ("none", "left"),
                   "signal_basis": "baseline_subtracted",
                   "y_processed": list(map(float, y))}}


# ------------------------------ definition contract ------------------------ #
def test_geometry_logistic_t50_equals_t0():
    p = {"base": 0.0, "amp": 1.0, "k": 0.5, "t0": 18.0}
    g = feature_geometry("logistic", p, T)
    assert abs(g["t50"] - 18.0) < 0.2
    # logistic inflection coincides with t50
    assert abs(g["inflection_time"] - 18.0) < 0.3


def test_lag_tangent_intercept_definition():
    """For a logistic, the tangent-intercept lag is t0 - 2/k (max slope = amp*k/4
    at t0, baseline crossing at t0 - (amp/2)/(amp*k/4) = t0 - 2/k)."""
    k, t0 = 0.5, 20.0
    g = feature_geometry("logistic", {"base": 0.0, "amp": 1.0, "k": k, "t0": t0}, T)
    assert abs(g["lag_time"] - (t0 - 2.0 / k)) < 0.4
    # lag-to-t50 ratio is the dimensionless shape invariant
    assert abs(g["lag_to_t50_ratio"] - (t0 - 2.0 / k) / t0) < 0.05


def test_extract_features_ok_and_contract_attached():
    res = extract_features(_logistic_series(t0=20.0))
    assert res["status"] == "ok"
    assert res["definition_contract"] == DEFINITION_CONTRACT["version"]
    assert res["t50_status"] == "point"
    assert res["model"] in ("logistic", "gompertz", "richards")
    assert abs(res["features"]["t50"] - 20.0) < 1.5


def test_bootstrap_cis_present_when_requested():
    res = extract_features(_logistic_series(t0=20.0, seed=3), B=40, seed=1)
    ci = res.get("bootstrap_ci", {})
    assert "t50" in ci and ci["t50"][0] <= res["features"]["t50"] <= ci["t50"][1]
    assert res["ci_reliable"] is True


# ------------------------------ censoring tags ----------------------------- #
def test_right_censoring_marks_t50_lower_bound():
    res = extract_features(_logistic_series(t0=20.0, cens="right"))
    assert res["t50_status"] == "lower_bound"
    assert "plateau_not_reached_extrapolated" in res["validity_flags"]


def test_left_censoring_makes_lag_undefined():
    res = extract_features(_logistic_series(t0=20.0, cens="left"))
    assert res["lag_status"] == "undefined_left_censored"
    assert res["features"]["lag_time"] is None
    assert res["features"]["lag_to_t50_ratio"] is None


# ------------------------------ dual-γ ------------------------------------- #
def _conc_member(m, gamma=0.5, A=80.0, k=0.5, cens="none", seed=0, n=24):
    """A logistic curve whose t50 obeys t50 = A·m^(-gamma)."""
    t50 = A * m ** (-gamma)
    x = np.linspace(0, max(4 * t50, 10), n)
    rng = np.random.default_rng(seed)
    y = m_logistic(x, 0.0, 1.0, k, t50) + rng.normal(0, 0.01, size=len(x))
    return {"series_id": f"m{m}", "x_hours": list(map(float, x)),
            "condition_vector": {"concentration": {"value_uM": float(m)}},
            "m1": {"recommended_handling": "monotonic_fit", "censoring_class": cens,
                   "plateau_reached": True, "signal_basis": "baseline_subtracted",
                   "y_processed": list(map(float, y))}}


def test_gamma_regression_recovers_known_exponent():
    true_g = 0.5
    members = [_conc_member(m, gamma=true_g, seed=i)
               for i, m in enumerate([5, 10, 20, 40, 80])]
    rows = series_t50_table(members)
    assert len(rows) == 5
    res = gamma_regression(rows, B=100, seed=0)
    assert res["status"] == "ok"
    assert abs(res["gamma"] - true_g) < 0.15
    assert res["gamma_ci"] is not None


def test_gamma_global_collapse_recovers_known_exponent():
    true_g = 0.7
    members = [_conc_member(m, gamma=true_g, seed=i)
               for i, m in enumerate([5, 10, 20, 40, 80])]
    res = gamma_global(members, B=0)
    assert res["status"] == "ok"
    assert abs(res["gamma"] - true_g) < 0.2
    assert res["collapse_r2"] > 0.9


def test_right_censored_point_does_not_break_regression():
    true_g = 0.5
    members = [_conc_member(m, gamma=true_g, seed=i,
                            cens=("right" if m == 80 else "none"))
               for i, m in enumerate([5, 10, 20, 40, 80])]
    rows = series_t50_table(members)
    res = gamma_regression(rows, B=80, seed=1)
    assert res["status"] == "ok"
    assert res["n_right_censored"] >= 1
    assert abs(res["gamma"] - true_g) < 0.25         # censoring handled, not wild


def test_dual_gamma_assembles_and_disagreement_is_boolean():
    members = [_conc_member(m, gamma=0.5, seed=i)
               for i, m in enumerate([5, 10, 20, 40, 80])]
    meta = {"concentration_series_id": "CS-T", "protein": "Synthetic",
            "pH": 7.4, "temperature_C": 37.0, "assay": "ThT", "mutation": "WT"}
    res = dual_gamma(meta, members, B_reg=80, B_glob=40)
    assert res["gamma_regression"]["status"] == "ok"
    assert res["gamma_global"]["status"] == "ok"
    d = res["disagreement"]
    assert isinstance(d["testable"], bool)
    if d["testable"]:
        assert isinstance(d["disagree"], bool)
    # two consistent estimators on clean data should broadly agree
    assert abs(d["gamma_regression"] - d["gamma_global"]) < 0.3


def test_insufficient_concentrations_reported():
    members = [_conc_member(m, seed=i) for i, m in enumerate([5, 10])]
    rows = series_t50_table(members)
    assert gamma_regression(rows)["status"] == "insufficient_data"
    assert gamma_global(members)["status"] == "insufficient_data"


# --------------------- Part-B honesty fixes (negative-γ gate, disagreement) ---- #
def _conc_member_rising_t50(m, A=20.0, gamma_pos=0.5, k=0.5, seed=0, n=24):
    """A curve family whose t50 RISES with concentration (t50 = A·m^(+gamma_pos)),
    i.e. the censored regression sees γ<0 — the non-physical case the gate catches."""
    t50 = A * m ** (gamma_pos)
    x = np.linspace(0, max(4 * t50, 10), n)
    rng = np.random.default_rng(seed)
    y = m_logistic(x, 0.0, 1.0, k, t50) + rng.normal(0, 0.01, size=len(x))
    return {"series_id": f"m{m}", "x_hours": list(map(float, x)),
            "condition_vector": {"concentration": {"value_uM": float(m)}},
            "m1": {"recommended_handling": "monotonic_fit", "censoring_class": "none",
                   "plateau_reached": True, "signal_basis": "baseline_subtracted",
                   "y_processed": list(map(float, y))}}


def test_negative_gamma_gate_fires_and_sets_physical_false():
    # t50 increases with m -> γ < 0 -> must be gated as non-physical
    members = [_conc_member_rising_t50(m, seed=i)
               for i, m in enumerate([5, 10, 20, 40, 80])]
    rows = series_t50_table(members)
    res = gamma_regression(rows, B=80, seed=0)
    assert res["status"] == "ok"
    assert res["gamma_physical"] is False
    assert res["gamma_reliable"] is False              # gated out of reliability
    assert res["gamma_raw"] is not None and res["gamma_raw"] < 0   # raw kept for transparency
    assert any("non-physical" in r for r in res["reliability_reasons"])


def test_positive_gamma_stays_physical():
    members = [_conc_member(m, gamma=0.5, seed=i)
               for i, m in enumerate([5, 10, 20, 40, 80])]
    rows = series_t50_table(members)
    res = gamma_regression(rows, B=80, seed=0)
    assert res["gamma_physical"] is True
    assert res["gamma"] is not None and res["gamma"] > 0


def test_gamma_global_relabelled_without_breaking_key():
    members = [_conc_member(m, gamma=0.5, seed=i)
               for i, m in enumerate([5, 10, 20, 40, 80])]
    res = gamma_global(members, B=0)
    # the consumer-facing key/payload still has `gamma`, plus the honesty relabel
    assert res["status"] == "ok" and "gamma" in res
    assert res["method"] == "master_curve_collapse"
    assert res["is_shared_rate_ode_fit"] is False


def test_disagreement_reports_pointwise_even_without_ci():
    """Plant a disagreeing pair: regression γ and collapse γ differ by > margin, and
    NO bootstrap CIs are supplied. The fixed test must still report the pointwise
    difference + disagree_pointwise (previously this was suppressed)."""
    reg = {"status": "ok", "gamma": 1.2, "gamma_ci": None,
           "gamma_reliable": True, "gamma_physical": True}
    glob = {"status": "ok", "gamma": 0.2, "gamma_ci": None, "collapse_r2": 0.95}
    d = _disagreement(reg, glob)
    assert d["testable"] is True
    assert abs(d["difference"] - 1.0) < 1e-9
    assert d["disagree_pointwise"] is True
    assert d["disagree"] is True


def test_disagreement_not_suppressed_when_regression_unreliable():
    # even a flagged-unreliable regression γ now yields a reported difference
    reg = {"status": "ok", "gamma": 1.5, "gamma_ci": None,
           "gamma_reliable": False, "gamma_physical": True}
    glob = {"status": "ok", "gamma": 0.3, "gamma_ci": None, "collapse_r2": 0.9}
    d = _disagreement(reg, glob)
    assert d["difference"] is not None
    assert d["disagree_pointwise"] is True
    assert d["testable"] is True                       # NOT dropped


def test_poor_collapse_flags_shape_not_concentration_invariant():
    reg = {"status": "ok", "gamma": 0.5, "gamma_ci": None,
           "gamma_reliable": True, "gamma_physical": True}
    glob = {"status": "ok", "gamma": 0.55, "gamma_ci": None,
            "collapse_r2": 0.5, "collapse_poor": True}   # bad collapse
    d = _disagreement(reg, glob)
    assert d["shape_not_concentration_invariant"] is True
    assert d["disagree"] is True                        # poor collapse is itself signal


def test_disagreement_consistent_when_close_and_good_collapse():
    reg = {"status": "ok", "gamma": 0.50, "gamma_ci": [0.4, 0.6],
           "gamma_reliable": True, "gamma_physical": True}
    glob = {"status": "ok", "gamma": 0.52, "gamma_ci": [0.45, 0.6],
            "collapse_r2": 0.97, "collapse_poor": False}
    d = _disagreement(reg, glob)
    assert d["disagree_pointwise"] is False
    assert d["disagree"] is False
    assert d["ci_overlap"] is True


def test_dual_gamma_joins_global_ode_fit_when_available():
    members = [_conc_member(m, gamma=0.5, seed=i)
               for i, m in enumerate([5, 10, 20, 40, 80])]
    meta = {"concentration_series_id": "CS-J", "protein": "Synthetic"}
    fake_global = {"CS-J": {
        "status": "ok", "version": "global-fit-1.0",
        "best_mechanism": "secondary_nucleation",
        "reaction_orders": {"n_c": 2.0, "n_2": 1.9, "n_2_free": True},
        "gamma_mechanistic": 1.45,
        "gamma_mechanistic_formula": "(n_2+1)/2",
        "mechanism_aicc_ranking": [],
        "sloppy": True,
        "identifiability": {"fim_condition_number": None,
                            "interpretation": "SLOPPY",
                            "identifiable_combinations": [],
                            "individual_rate_status": {}},
        "honesty_note": "does NOT license a single mechanism"}}
    res = dual_gamma(meta, members, B_reg=60, B_glob=30,
                     global_fit_by_csid=fake_global)
    assert "global_ode_fit" in res
    assert res["global_ode_fit"]["gamma_mechanistic"] == 1.45
    assert res["global_ode_fit"]["best_mechanism"] == "secondary_nucleation"
    # no global fit -> no key
    res2 = dual_gamma(meta, members, B_reg=60, B_glob=30)
    assert "global_ode_fit" not in res2


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}"); failed += 1
        except Exception as e:                       # noqa
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
