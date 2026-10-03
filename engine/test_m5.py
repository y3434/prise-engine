"""
Tests for M5 — classification, regime registry, mechanistic degeneracy & anomaly.

    python engine/test_m5.py
    pytest engine/test_m5.py
"""
from __future__ import annotations

import copy

import numpy as np

from m5_classify import (
    REGIME_REGISTRY,
    benjamini_hochberg,
    classify_curve,
    classify_descriptive,
    mechanistic_assessment,
    propose_registry_mutation,
    residual_anomaly,
    _runs_test,
)
from models import m_logistic

T = np.arange(0, 40, 1.0)


def _series(y, handling="monotonic_fit", cens="none", flat=False, non_mono=False,
            assay_mass=True, agit=None, seeded=None, pdb=None, sid="syn",
            plateau=True, csid=None):
    cv = {"assay_reports_mass": assay_mass,
          "assay_type": "ThT" if assay_mass else "Turbidity",
          "agitation": agit, "seeded": seeded, "pdb_id": pdb, "construct_id": "WT",
          "field_provenance": {"agitation": "known" if agit is not None else "unknown",
                               "seeded": "known" if seeded is not None else "unknown"}}
    return {"series_id": sid, "x_hours": list(map(float, T)), "data_mode": "kinetic",
            "concentration_series_id": csid, "condition_vector": cv,
            "y_intensity": list(map(float, y)),
            "m1": {"recommended_handling": handling, "censoring_class": cens,
                   "fittability_class": "fittable", "plateau_reached": plateau,
                   "flat_no_signal": flat, "non_monotonic": non_mono,
                   "signal_basis": "baseline_subtracted",
                   "y_processed": list(map(float, y))}}


def _feat(coop, sharp=None, ratio=None, status="ok"):
    return {"status": status, "r2": 0.99, "n_points": 40,
            "features": {"has_interior_inflection": coop,
                         "transition_sharpness": sharp, "lag_to_t50_ratio": ratio}}


# --------------------------- descriptive registry -------------------------- #
def test_descriptive_cooperative_sigmoid():
    d = classify_descriptive({"recommended_handling": "monotonic_fit"},
                             _feat(True, sharp=1.5, ratio=0.6))
    assert d["regime"] == "cooperative_sigmoidal"


def test_descriptive_threshold_driven():
    # Inputs updated for the GOVERNED Service-C calibration (thresholds-1.0): the
    # threshold-driven gate is now sharpness > 5.7 AND lag/t50 > 0.97 (was 3.0 /
    # 0.80). The old (sharp=10.0, ratio=0.9) case now legitimately falls to
    # cooperative_sigmoidal because ratio=0.9 < the calibrated 0.97; we keep the
    # BEHAVIORAL coverage (a genuinely abrupt, long-lag curve → threshold_driven) by
    # feeding a case that clears BOTH calibrated cutoffs.
    d = classify_descriptive({"recommended_handling": "monotonic_fit"},
                             _feat(True, sharp=10.0, ratio=0.99))
    assert d["regime"] == "threshold_driven"


def test_descriptive_gradual_non_cooperative():
    d = classify_descriptive({"recommended_handling": "monotonic_fit"},
                             _feat(False))
    assert d["regime"] == "gradual_non_cooperative"


def test_descriptive_non_monotonic():
    d = classify_descriptive({"recommended_handling": "nonmonotonic_fit"},
                             _feat(True, 1.5, 0.6))
    assert d["regime"] == "non_monotonic_settling"


def test_descriptive_no_detectable():
    d = classify_descriptive({"recommended_handling": "descriptive_only"},
                             _feat(False, status="flat_no_transition"))
    assert d["regime"] == "no_detectable_aggregation"
    assert "assay" in d["definition"] or "assay" in REGIME_REGISTRY["regimes"][
        "no_detectable_aggregation"]["qualifier"]


def test_descriptive_no_fit_is_anomalous():
    """anomalous_unclassified is reserved for genuinely unusable curves (no fit),
    NOT mild misfit — the anomaly overlay is a separate, parallel channel."""
    d = classify_descriptive({"recommended_handling": "reject"},
                             {"status": "no_fit", "features": {}})
    assert d["regime"] == "anomalous_unclassified"


# ---------------------- mechanistic degeneracy & gates --------------------- #
def test_single_mechanism_always_refused():
    """The headline honesty contract: M5 never names a single mechanism."""
    s = _series(m_logistic(T, 0, 1, 0.5, 20), assay_mass=True, agit=True, seeded=True)
    m = mechanistic_assessment(s, {"regime": "cooperative_sigmoidal"},
                               _feat(True, 1.5, 0.6), "kinetic")
    assert m["single_mechanism_call"] is None
    assert len(m["equivalence_class"]) >= 2


def test_mechanism_licensed_when_all_gates_pass():
    s = _series(m_logistic(T, 0, 1, 0.5, 20), assay_mass=True, agit=True, seeded=True)
    m = mechanistic_assessment(s, {"regime": "cooperative_sigmoidal"},
                               _feat(True, 1.5, 0.6), "kinetic")
    assert m["mechanistic_inference_licensed"] is True and m["gates_failed"] == []


def test_assay_gate_blocks_non_mass_assay():
    s = _series(m_logistic(T, 0, 1, 0.5, 20), assay_mass=False, agit=True, seeded=True)
    m = mechanistic_assessment(s, {"regime": "cooperative_sigmoidal"},
                               _feat(True, 1.5, 0.6), "kinetic")
    assert "assay_not_mass_proportional" in m["gates_failed"]
    assert m["mechanistic_inference_licensed"] is False


def test_agitation_seeding_unknown_branch_undetermined():
    s = _series(m_logistic(T, 0, 1, 0.5, 20), assay_mass=True)   # agit/seeded unknown
    m = mechanistic_assessment(s, {"regime": "cooperative_sigmoidal"},
                               _feat(True, 1.5, 0.6), "kinetic")
    assert "agitation_or_seeding_unknown_branch_undetermined" in m["gates_failed"]


def test_shape_gate_blocks_non_nucleated():
    s = _series(m_logistic(T, 0, 1, 0.5, 20), assay_mass=True, agit=True, seeded=True)
    m = mechanistic_assessment(s, {"regime": "gradual_non_cooperative"},
                               _feat(False), "kinetic")
    assert "no_nucleated_transition_in_shape" in m["gates_failed"]


def test_assay_unknown_distinct_from_not_mass():
    s = _series(m_logistic(T, 0, 1, 0.5, 20), assay_mass=None, agit=True, seeded=True)
    m = mechanistic_assessment(s, {"regime": "cooperative_sigmoidal"},
                               _feat(True, 1.5, 0.6), "kinetic")
    assert "assay_mass_proportionality_unknown" in m["gates_failed"]
    assert "assay_not_mass_proportional" not in m["gates_failed"]


# ----------------------- degeneracy reflects the analysis ------------------ #
def _gamma(reliable, curvature, status="ok"):
    return {"gamma_regression": {"status": status, "gamma": 0.5,
                                 "gamma_reliable": reliable,
                                 "curvature_test": {"significant": curvature}}}


def test_degeneracy_single_curve_is_broad():
    s = _series(m_logistic(T, 0, 1, 0.5, 20))
    m = mechanistic_assessment(s, {"regime": "cooperative_sigmoidal"},
                               _feat(True, 1.5, 0.6), "kinetic", gamma=None)
    assert m["degeneracy"] == "broad_single_curve"


def test_degeneracy_reduced_by_reliable_gamma():
    s = _series(m_logistic(T, 0, 1, 0.5, 20), agit=True, seeded=True)
    m = mechanistic_assessment(s, {"regime": "cooperative_sigmoidal"},
                               _feat(True, 1.5, 0.6), "concentration_series",
                               gamma=_gamma(reliable=True, curvature=False))
    assert m["degeneracy"] == "evidence_reduced_by_reliable_gamma"


def test_degeneracy_rebroadened_by_curvature():
    s = _series(m_logistic(T, 0, 1, 0.5, 20), agit=True, seeded=True)
    m = mechanistic_assessment(s, {"regime": "cooperative_sigmoidal"},
                               _feat(True, 1.5, 0.6), "concentration_series",
                               gamma=_gamma(reliable=True, curvature=True))
    assert m["degeneracy"] == "re_broadened_saturating_or_mcrit"


def test_degeneracy_broad_when_gamma_unavailable():
    s = _series(m_logistic(T, 0, 1, 0.5, 20), agit=True, seeded=True)
    m = mechanistic_assessment(s, {"regime": "cooperative_sigmoidal"},
                               _feat(True, 1.5, 0.6), "concentration_series",
                               gamma=_gamma(False, False, status="insufficient_data"))
    assert m["degeneracy"] == "broad_gamma_unavailable"


def test_equivalence_class_never_narrowed():
    """Even with a reliable γ the actual class is not narrowed (needs Service C)."""
    s = _series(m_logistic(T, 0, 1, 0.5, 20), agit=True, seeded=True)
    m = mechanistic_assessment(s, {"regime": "cooperative_sigmoidal"},
                               _feat(True, 1.5, 0.6), "concentration_series",
                               gamma=_gamma(reliable=True, curvature=False))
    assert m["equivalence_class_narrowed"] is False


# ------------------------------ anomaly detector --------------------------- #
def test_runs_test_flags_structured_residuals():
    # one sign-switch over 12 points -> 2 runs (far below expected) -> tiny p
    resid = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1], float)
    r = _runs_test(resid)
    assert r["testable"] and r["runs"] == 2 and r["p_too_few_runs"] < 0.05


def test_runs_test_passes_random_residuals():
    rng = np.random.default_rng(0)
    r = _runs_test(rng.normal(size=40))
    assert r["testable"] and r["p_too_few_runs"] > 0.05


def test_residual_anomaly_clean_logistic_not_flagged():
    rng = np.random.default_rng(1)
    y = m_logistic(T, 0, 1, 0.5, 20) + rng.normal(0, 0.01, size=len(T))
    a = residual_anomaly(_series(y))
    assert a["testable"] and a["p_too_few_runs"] > 0.05


# ------------------------------ BH-FDR control ----------------------------- #
def test_benjamini_hochberg_rejects_only_small_p():
    bh = benjamini_hochberg([0.001, 0.2, 0.5, 0.8, 0.9], alpha=0.05)
    assert bh["n_rejected"] == 1 and bh["reject"][0] is True
    assert all(bh["reject"][i] is False for i in (1, 2, 3, 4))


def test_benjamini_hochberg_handles_no_tests():
    bh = benjamini_hochberg([None, None], alpha=0.05)
    assert bh["n_tested"] == 0 and bh["n_rejected"] == 0


# ------------------------ governance: proposal not mutation ---------------- #
def test_registry_proposal_does_not_mutate_registry():
    before = copy.deepcopy(REGIME_REGISTRY)
    p = propose_registry_mutation("syn", {"best_model": "logistic", "runs": 2,
                                          "p_too_few_runs": 0.001, "r2": 0.8})
    assert p["status"] == "PROPOSED_PENDING_HUMAN_APPROVAL"
    assert p["taxonomy_version"] == REGIME_REGISTRY["version"]
    assert "proposed_at" in p
    assert REGIME_REGISTRY == before          # registry is untouched (human-governed)


# ------------------------------ integration -------------------------------- #
def test_classify_curve_end_to_end():
    rng = np.random.default_rng(2)
    y = m_logistic(T, 0, 1, 0.5, 20) + rng.normal(0, 0.01, size=len(T))
    res = classify_curve(_series(y, agit=True, seeded=True, pdb="2BEG"))
    assert res["descriptive_regime"]["regime"] == "cooperative_sigmoidal"
    assert res["analysis_scope"] == "single_curve"
    assert res["mechanistic"]["single_mechanism_call"] is None
    assert res["mechanistic"]["degeneracy"] == "broad_single_curve"
    assert res["structure_linkage"]["pdb_id"] == "2BEG"
    assert res["structure_linkage"]["relationship"] == "associative_only"
    assert res["confidence"]["level"] == "high"        # clean fit -> high
    assert "t_max_hours" in res["window_of_validity"]


def test_no_fit_confidence_is_low():
    """A curve with no converged fit is anomalous_unclassified AND low-confidence
    (was a bug: it returned 'medium')."""
    s = _series([1.0, 1.0, 1.0], handling="reject")
    s["x_hours"] = [0.0, 1.0, 2.0]
    res = classify_curve(s)
    assert res["descriptive_regime"]["regime"] == "anomalous_unclassified"
    assert res["confidence"]["level"] == "low"


def test_material_misfit_gates_proposals():
    """A near-perfect fit with (possibly) structured residuals is NOT material_misfit;
    a curve the monotonic bank fits badly IS."""
    rng = np.random.default_rng(5)
    clean = m_logistic(T, 0, 1, 0.5, 20) + rng.normal(0, 0.005, size=len(T))
    rc = classify_curve(_series(clean))
    assert rc["anomaly"]["material_misfit"] is False         # R^2 ~ 1

    oscillating = 0.5 + 0.4 * np.sin(T / 2.0)                # bank cannot represent
    ro = classify_curve(_series(oscillating))
    assert ro["anomaly"]["material_misfit"] is True          # poor R^2


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
