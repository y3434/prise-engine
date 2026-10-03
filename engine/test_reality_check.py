"""
Tests for the blind external reality check (engine/reality_check.py).

Deterministic, fast — driven by SYNTHETIC/crafted artifact indices (no disk I/O in
the logic tests; a tiny tmp-dir round-trip covers the loaders). Covers:

  * a protein whose PRISE regime = cooperative_sigmoidal + γ in band -> CONSISTENT
  * a crafted "gradual_non_cooperative on a strong-consensus nucleated amyloid"
    -> CONTRADICTION (the falsifying event)
  * a strong-consensus γ landing OUT of band -> CONTRADICTION
  * a missing protein -> INSUFFICIENT_DATA
  * the pre-registered consensus table loads with citations + strengths
  * the leakage / reserved-proteins check (data leak fails; prose mention is OK)
  * the whole thing never raises on empty artifacts

    python engine/test_reality_check.py
    pytest engine/test_reality_check.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import reality_check as rc
from reality_check import (
    CONSENSUS,
    check_gamma,
    check_mechanism_class,
    check_regime,
    evaluate_protein,
    extract_prise,
    leakage_check,
    load_artifacts,
    run_reality_check,
)

_BROAD_CLASS = ["primary_nucleation_dominated", "secondary_nucleation_dominated",
                "fragmentation_dominated", "saturating_secondary_nucleation"]


def _artifacts(per_curve_regimes=None, gamma_by_protein=None,
               global_fit_by_protein=None, series_modal=None, service_c=None):
    """Build a crafted artifacts dict shaped exactly like load_artifacts() output."""
    return {
        "per_curve_regimes": per_curve_regimes or {},
        "series_modal": series_modal or {},
        "gamma_by_protein": gamma_by_protein or {},
        "global_fit_by_protein": global_fit_by_protein or {},
        "service_c": service_c or {},
    }


def _gamma_series(csid, gamma, reliable=True, physical=True, ci=None,
                  collapse_gamma=None, collapse_r2=0.98):
    reg = {"status": "ok", "gamma": gamma, "gamma_reliable": reliable,
           "gamma_physical": physical, "gamma_ci": ci or [gamma - 0.1, gamma + 0.1]}
    glob = {"status": "insufficient_data"}
    if collapse_gamma is not None:
        glob = {"status": "ok", "gamma": collapse_gamma,
                "collapse_r2": collapse_r2, "collapse_poor": collapse_r2 < 0.8}
    return {"concentration_series_id": csid, "gamma_regression": reg,
            "gamma_global": glob}


def _series_modal(protein, modal, equiv=None, gates=None):
    return {"protein": protein, "descriptive_regime_modal": modal,
            "mechanistic": {"equivalence_class": equiv or list(_BROAD_CLASS),
                            "gates_failed": gates or
                            ["no_nucleated_transition_in_shape",
                             "agitation_or_seeding_unknown_branch_undetermined"]}}


# --------------------------------------------------------------------------- #
# consensus table
# --------------------------------------------------------------------------- #
def test_consensus_table_loads_with_citations():
    assert "Amyloid Beta peptide-ABeta42" in CONSENSUS
    for name, e in CONSENSUS.items():
        assert e["citation"] and isinstance(e["citation"], str)
        assert e["consensus_strength"] in ("strong", "moderate")
        assert e["gamma_consensus_strength"] in ("strong", "moderate")
        assert e["aliases"] and name in e["aliases"]
    # Aβ42 carries the strong γ≈1.1 fixed point
    ab42 = CONSENSUS["Amyloid Beta peptide-ABeta42"]
    assert ab42["gamma_consensus_strength"] == "strong"
    assert ab42["gamma_band"][0] <= 1.1 <= ab42["gamma_band"][1]
    assert ab42["consensus_mechanism"] == "secondary_nucleation_dominated"


# --------------------------------------------------------------------------- #
# CONSISTENT: cooperative sigmoid + γ in band
# --------------------------------------------------------------------------- #
def test_cooperative_sigmoidal_gamma_in_band_is_consistent():
    entry = CONSENSUS["Amyloid Beta peptide-ABeta42"]
    art = _artifacts(
        per_curve_regimes={"Amyloid Beta peptide-ABeta42":
                           {"cooperative_sigmoidal": 100, "gradual_non_cooperative": 10}},
        gamma_by_protein={"Amyloid Beta peptide-ABeta42":
                          [_gamma_series("CS1", 1.07, ci=[0.94, 1.14])]},
        series_modal={"Amyloid Beta peptide-ABeta42":
                      [_series_modal("Amyloid Beta peptide-ABeta42", "cooperative_sigmoidal")]},
    )
    res = evaluate_protein("Amyloid Beta peptide-ABeta42", entry, art)
    assert res["verdict"] == "CONSISTENT"
    assert res["checks"]["regime"]["status"] == "pass"
    assert res["checks"]["gamma"]["status"] == "in_band"
    assert res["checks"]["mechanism_class"]["status"] == "consistent"
    # mechanism is honestly NOT licensed (expected)
    assert res["prise"]["mechanistic_inference_licensed"] is False


# --------------------------------------------------------------------------- #
# CONTRADICTION: gradual_non_cooperative on a strong-consensus nucleated amyloid
# --------------------------------------------------------------------------- #
def test_gradual_non_cooperative_on_resolved_amyloid_is_contradiction():
    entry = CONSENSUS["Amyloid Beta peptide-ABeta42"]
    art = _artifacts(
        per_curve_regimes={"Amyloid Beta peptide-ABeta42":
                           {"gradual_non_cooperative": 90, "cooperative_sigmoidal": 5}},
        gamma_by_protein={"Amyloid Beta peptide-ABeta42":
                          [_gamma_series("CS1", 1.05, ci=[0.9, 1.2])]},
    )
    res = evaluate_protein("Amyloid Beta peptide-ABeta42", entry, art)
    assert res["verdict"] == "CONTRADICTION"
    assert res["checks"]["regime"]["status"] == "contradict"
    assert "regime" in res["notes"]


def test_no_detectable_on_resolved_amyloid_is_contradiction():
    entry = CONSENSUS["insulin"]
    art = _artifacts(
        per_curve_regimes={"insulin": {"no_detectable_aggregation": 12}},
    )
    res = evaluate_protein("insulin", entry, art)
    assert res["verdict"] == "CONTRADICTION"
    assert res["checks"]["regime"]["status"] == "contradict"


# --------------------------------------------------------------------------- #
# CONTRADICTION: strong-consensus γ out of band
# --------------------------------------------------------------------------- #
def test_strong_consensus_gamma_out_of_band_is_contradiction():
    entry = CONSENSUS["Amyloid Beta peptide-ABeta42"]     # strong γ band [0.8, 1.4]
    art = _artifacts(
        per_curve_regimes={"Amyloid Beta peptide-ABeta42": {"cooperative_sigmoidal": 50}},
        gamma_by_protein={"Amyloid Beta peptide-ABeta42":
                          [_gamma_series("CS1", 0.2, ci=[0.1, 0.3])]},   # way below band
    )
    res = evaluate_protein("Amyloid Beta peptide-ABeta42", entry, art)
    assert res["checks"]["gamma"]["status"] == "out_of_band"
    assert res["verdict"] == "CONTRADICTION"
    assert "gamma_strong_consensus" in res["notes"]


def test_moderate_consensus_gamma_out_of_band_is_flag_not_contradiction():
    # Aβ40 has a MODERATE γ consensus -> out-of-band is a flag, not falsifying.
    entry = CONSENSUS["Amyloid Beta peptide-ABeta40"]
    art = _artifacts(
        per_curve_regimes={"Amyloid Beta peptide-ABeta40": {"cooperative_sigmoidal": 50}},
        gamma_by_protein={"Amyloid Beta peptide-ABeta40":
                          [_gamma_series("CS1", 0.05, ci=[0.0, 0.1])]},  # below moderate band
    )
    res = evaluate_protein("Amyloid Beta peptide-ABeta40", entry, art)
    assert res["checks"]["gamma"]["status"] == "out_of_band"
    assert res["verdict"] == "CONSISTENT"     # moderate γ out-of-band does NOT falsify


# --------------------------------------------------------------------------- #
# mechanism-class check: excluded -> contradiction
# --------------------------------------------------------------------------- #
def test_mechanism_class_excluded_is_contradiction():
    entry = CONSENSUS["Amyloid Beta peptide-ABeta42"]   # consensus secondary_nucleation
    # craft an equivalence class that POSITIVELY EXCLUDES secondary nucleation
    excl_class = ["primary_nucleation_dominated", "fragmentation_dominated"]
    art = _artifacts(
        per_curve_regimes={"Amyloid Beta peptide-ABeta42": {"cooperative_sigmoidal": 40}},
        series_modal={"Amyloid Beta peptide-ABeta42":
                      [_series_modal("Amyloid Beta peptide-ABeta42", "cooperative_sigmoidal",
                                     equiv=excl_class)]},
    )
    res = evaluate_protein("Amyloid Beta peptide-ABeta42", entry, art)
    assert res["checks"]["mechanism_class"]["status"] == "excluded"
    assert res["verdict"] == "CONTRADICTION"


def test_broad_class_includes_consensus_is_consistent():
    entry = CONSENSUS["Amyloid Beta peptide-ABeta42"]
    art = _artifacts(
        per_curve_regimes={"Amyloid Beta peptide-ABeta42": {"cooperative_sigmoidal": 40}},
        series_modal={"Amyloid Beta peptide-ABeta42":
                      [_series_modal("Amyloid Beta peptide-ABeta42", "cooperative_sigmoidal")]},
    )
    res = evaluate_protein("Amyloid Beta peptide-ABeta42", entry, art)
    assert res["checks"]["mechanism_class"]["status"] == "consistent"


# --------------------------------------------------------------------------- #
# INSUFFICIENT_DATA: missing protein
# --------------------------------------------------------------------------- #
def test_missing_protein_is_insufficient_data():
    entry = CONSENSUS["Beta-2 microglobulin (Beta-2m)"]
    res = evaluate_protein("Beta-2 microglobulin (Beta-2m)", entry, _artifacts())
    assert res["verdict"] == "INSUFFICIENT_DATA"
    assert res["checks"]["regime"]["status"] == "na"


# --------------------------------------------------------------------------- #
# alias folding: ABeta42 (synthetic) folds into Aβ42
# --------------------------------------------------------------------------- #
def test_alias_folding_across_synthetic_variant():
    entry = CONSENSUS["Amyloid Beta peptide-ABeta42"]
    art = _artifacts(
        per_curve_regimes={
            "Amyloid Beta peptide-ABeta42": {"cooperative_sigmoidal": 20},
            "Amyloid Beta peptide-ABeta42 (synthetic)": {"cooperative_sigmoidal": 30},
        },
    )
    prise = extract_prise(entry, art)
    assert prise["n_curves"] == 50            # both variants folded
    assert prise["modal_regime"] == "cooperative_sigmoidal"


# --------------------------------------------------------------------------- #
# gamma source preference: reliable regression > collapse
# --------------------------------------------------------------------------- #
def test_gamma_prefers_reliable_regression_then_collapse():
    entry = CONSENSUS["Amyloid Beta peptide-ABeta42"]
    # unreliable regression but a good collapse γ in band -> collapse used
    art = _artifacts(
        per_curve_regimes={"Amyloid Beta peptide-ABeta42": {"cooperative_sigmoidal": 10}},
        gamma_by_protein={"Amyloid Beta peptide-ABeta42":
                          [_gamma_series("CS1", 5.0, reliable=False,
                                         collapse_gamma=1.1, collapse_r2=0.97)]},
    )
    chk = check_gamma(entry, extract_prise(entry, art))
    assert chk["prise_gamma_source"] == "gamma_global_collapse"
    assert chk["status"] == "in_band"


# --------------------------------------------------------------------------- #
# leakage / reserved discipline
# --------------------------------------------------------------------------- #
def test_leakage_prose_mention_is_not_a_leak():
    # a note field mentioning insulin documents the §6C role — NOT a data leak
    sc = {"deferred": ["reality check on Aβ42/insulin/β2m vs consensus (reserved)"],
          "note": "insulin is reserved solely for §6C"}
    lk = leakage_check(_artifacts(service_c=sc))
    assert lk["no_leakage"] is True
    assert "insulin" in [n.lower() for n in lk["reserved_names_in_prose_only"]]
    assert lk["reserved_names_in_calibration_data"] == []


def test_leakage_data_field_mention_is_a_leak():
    # a reserved protein used as an ACTUAL calibration anchor (data-bearing key/value)
    sc = {"anchors": {"insulin": {"t50": 3.0}},   # data-bearing: a leak
          "calibration_proteins": ["insulin"]}
    lk = leakage_check(_artifacts(service_c=sc))
    assert lk["no_leakage"] is False
    assert any(n.lower() == "insulin" for n in lk["reserved_names_in_calibration_data"])


def test_leakage_empty_calibration_is_clean():
    lk = leakage_check(_artifacts())
    assert lk["no_leakage"] is True
    assert len(lk["reserved_proteins"]) >= 5


# --------------------------------------------------------------------------- #
# full run: never raises; overall structure + honest framing present
# --------------------------------------------------------------------------- #
def test_run_reality_check_on_empty_dir_is_all_insufficient():
    with tempfile.TemporaryDirectory() as d:
        report = run_reality_check(Path(d))
    r = report["reality_check_result"]
    assert r["n_contradiction"] == 0
    assert r["n_consistent"] == 0
    assert r["n_insufficient"] == len(CONSENSUS)
    assert r["n_fixed_points_evaluated"] == 0
    assert report["version"] == "reality-check-1.0"
    assert "NOT proof of correctness" in report["honesty_note"]
    assert "FALSIF" in report["falsification_note"].upper()
    assert report["reserved_proteins_note"]["no_leakage"] is True


def test_run_reality_check_crafted_dir_round_trip():
    # write minimal artifacts to a tmp dir and confirm the loaders + verdicts wire up
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        (dd / "curves_triaged.jsonl").write_text(
            json.dumps({"series_id": "s1", "protein_id": "insulin"}) + "\n",
            encoding="utf-8")
        (dd / "regimes.jsonl").write_text(
            json.dumps({"series_id": "s1",
                        "descriptive_regime": {"regime": "cooperative_sigmoidal"}}) + "\n",
            encoding="utf-8")
        report = run_reality_check(dd)
    ins = next(p for p in report["fixed_points"] if p["protein"] == "insulin")
    assert ins["checks"]["regime"]["status"] == "pass"
    assert ins["verdict"] == "CONSISTENT"


def test_run_reality_check_is_deterministic():
    with tempfile.TemporaryDirectory() as d:
        a = run_reality_check(Path(d))
        b = run_reality_check(Path(d))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_loaders_never_raise_on_corrupt_lines():
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        (dd / "regimes.jsonl").write_text("{not json\n", encoding="utf-8")
        (dd / "gamma.jsonl").write_text("garbage\n", encoding="utf-8")
        art = load_artifacts(dd)          # must not raise
    assert art["per_curve_regimes"] == {}
    assert art["gamma_by_protein"] == {}


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
