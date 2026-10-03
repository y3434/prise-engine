"""
Tests for M9 — Experimental-Design Recommender.

Deterministic + fast. Operates on SYNTHETIC small state dicts only (does NOT read
the real data files), so it is hermetic and pins the §5-M9 contract:

  - each candidate experiment FIRES only when its blocker is present;
  - ranking is by reduction/cost (a CHEAP unblock-mechanism move outranks the
    MEDIUM concentration series with the same measured reduction);
  - a STRUCTURAL-degeneracy state is REFUSED (no experiment recommended) with the
    structural flag — both via the M3 flag and via the Service-C residual pair;
  - the concentration-series recommendation carries Service C's MEASURED 1->4
    narrowing + the per-pair γ-separation AUC;
  - the validity-ceiling + method-statement blocks are present, and the method
    commits to the rule table while explicitly DEFERRING EIG;
  - the corpus rollup ranks "record agitation/seeding" as the top action and
    counts the ~33 it unblocks (tied to the M8 GAP);
  - missing inputs degrade (do not crash).

    python engine/test_m9.py
    pytest engine/test_m9.py
"""
from __future__ import annotations

import json

import m9_recommender as m9


# --------------------------------------------------------------------------- #
# synthetic state + artifact builders
# --------------------------------------------------------------------------- #
def _state(**over):
    """A flat synthetic state with NO blocker present by default; the test sets
    only the blocker(s) it wants to fire."""
    base = {
        "unit_id": "U1",
        "protein_id": "PROT",
        "agitation_seeding_unknown": False,
        "has_concentration_series": True,   # so 'add series' does NOT fire by default
        "t50_censored": False,
        "assay_mass_unknown": False,
        "degeneracy_class": None,
        "structural_non_identifiable": False,
        "mechanistic_licensed": False,
        "residual_pair": None,
        "is_replicated": True,              # so 'add replicates' does NOT fire by default
    }
    base.update(over)
    return base


def _experiments(out):
    return {r["experiment"] for r in out["recommendations"]}


def _by_exp(out, exp):
    for r in out["recommendations"]:
        if r["experiment"] == exp:
            return r
    return None


# Minimal Service-C artifact carrying the measured narrowing (mirrors the real
# service_c_calibration.json fields M9 reads).
_SERVICE_C = {
    "service_c_version": "service-c-1.0",
    "concentration_series_narrowing": {
        "single_concentration_partition": {"n_classes": 1},
        "concentration_series_partition": {"n_classes": 4},
    },
    "confusion_matrices": {
        "concentration_series": {
            "pairwise_separation_auc": {
                "fragmentation|saturating_secondary": 0.88,
                "nucleation_elongation|secondary_nucleation": 1.0,
            }
        }
    },
    "mismatched_generator": {
        "mechanism_misidentification_degradation": 0.5333333333333333},
}


# --------------------------------------------------------------------------- #
# firing: each candidate fires ONLY when its blocker is present
# --------------------------------------------------------------------------- #
def test_no_blockers_only_default_does_not_fire_unblockers():
    # a clean unit (series present, replicated, no censoring, mass assay, agitation
    # known) -> NONE of the blocker-gated experiments fire.
    out = m9.recommend(_state(), _SERVICE_C)
    exps = _experiments(out)
    assert "record_agitation_and_seeding" not in exps
    assert "add_concentration_series" not in exps
    assert "extend_observation_window" not in exps
    assert "use_mass_proportional_assay" not in exps


def test_agitation_fires_only_when_unknown():
    assert "record_agitation_and_seeding" not in _experiments(
        m9.recommend(_state(agitation_seeding_unknown=False), _SERVICE_C))
    assert "record_agitation_and_seeding" in _experiments(
        m9.recommend(_state(agitation_seeding_unknown=True), _SERVICE_C))


def test_concentration_series_fires_only_when_absent():
    assert "add_concentration_series" not in _experiments(
        m9.recommend(_state(has_concentration_series=True), _SERVICE_C))
    assert "add_concentration_series" in _experiments(
        m9.recommend(_state(has_concentration_series=False), _SERVICE_C))


def test_extend_window_fires_only_when_censored():
    assert "extend_observation_window" not in _experiments(
        m9.recommend(_state(t50_censored=False), _SERVICE_C))
    assert "extend_observation_window" in _experiments(
        m9.recommend(_state(t50_censored=True), _SERVICE_C))


def test_replicates_fires_only_when_not_replicated():
    assert "add_replicates" not in _experiments(
        m9.recommend(_state(is_replicated=True), _SERVICE_C))
    assert "add_replicates" in _experiments(
        m9.recommend(_state(is_replicated=False), _SERVICE_C))


def test_mass_assay_fires_only_when_unknown():
    assert "use_mass_proportional_assay" not in _experiments(
        m9.recommend(_state(assay_mass_unknown=False), _SERVICE_C))
    assert "use_mass_proportional_assay" in _experiments(
        m9.recommend(_state(assay_mass_unknown=True), _SERVICE_C))


# --------------------------------------------------------------------------- #
# ranking: by reduction/cost — cheap unblock outranks the expensive series
# --------------------------------------------------------------------------- #
def test_cheap_unblock_outranks_expensive_series_WHEN_series_present():
    # WITH a concentration series already present, recording agitation/seeding is
    # SUFFICIENT (the last missing piece) and carries the SAME measured 4x
    # reduction as a series WOULD -> the cheaper metadata fix must rank #1
    # (higher priority = reduction/cost). (When NO series is present the crediting
    # is dependency-aware — see test_no_series_agitation_*.)
    out = m9.recommend(
        _state(agitation_seeding_unknown=True, has_concentration_series=True),
        _SERVICE_C)
    assert out["top_recommendation"] == "record_agitation_and_seeding"
    agit = _by_exp(out, "record_agitation_and_seeding")
    assert agit["rank"] == 1
    # the series does NOT fire (already present) — agitation is the sole mechanism
    # move, at full measured reduction.
    assert _by_exp(out, "add_concentration_series") is None
    assert agit["expected_degeneracy_reduction"] == 4.0
    assert agit["priority_score"] == 4.0
    # sufficient here -> NOT flagged necessary-but-not-sufficient.
    assert agit.get("necessary_but_not_sufficient") is not True


# --------------------------------------------------------------------------- #
# DEPENDENCY-AWARE crediting of agitation/seeding (the reviewed defect fix):
# on a SINGLE curve (no concentration series) recording agitation/seeding is
# NECESSARY-but-NOT-SUFFICIENT for mechanism resolution — it must NOT out-rank
# add_concentration_series (the binding prerequisite) for the mechanism goal.
# --------------------------------------------------------------------------- #
def test_no_series_agitation_is_necessary_but_not_sufficient():
    out = m9.recommend(
        _state(agitation_seeding_unknown=True, has_concentration_series=False),
        _SERVICE_C)
    agit = _by_exp(out, "record_agitation_and_seeding")
    series = _by_exp(out, "add_concentration_series")
    assert agit is not None and series is not None
    # explicitly flagged + pointed at the binding prerequisite.
    assert agit["necessary_but_not_sufficient"] is True
    assert agit["also_requires"] == ["add_concentration_series"]
    # the note states the dependency honestly.
    note = agit["expected_reduction_note"].lower()
    assert "not" in note and "sufficient" in note
    assert "concentration series" in note
    # standalone reduction is DISCOUNTED to future-gate removal only — NOT the ~4x
    # mechanism-resolution gain (which needs a series too).
    assert agit["expected_degeneracy_reduction"] < 4.0
    assert agit["reduction_kind"] == "removes_future_mechanistic_gate"


def test_no_series_series_is_NOT_outranked_by_agitation_for_mechanism():
    # the binding constraint (add_concentration_series) must rank ABOVE the
    # over-creditable agitation rec on a single-curve unit.
    out = m9.recommend(
        _state(agitation_seeding_unknown=True, has_concentration_series=False),
        _SERVICE_C)
    assert out["top_recommendation"] == "add_concentration_series"
    agit = _by_exp(out, "record_agitation_and_seeding")
    series = _by_exp(out, "add_concentration_series")
    assert series["rank"] < agit["rank"]
    assert series["priority_score"] > agit["priority_score"]
    # series keeps its full measured 4x; agitation is discounted below it.
    assert series["expected_degeneracy_reduction"] == 4.0


def test_with_series_agitation_keeps_full_reduction_and_ranks_top():
    # the complement: WITH a series, agitation is sufficient, full 4x, ranks #1.
    out = m9.recommend(
        _state(agitation_seeding_unknown=True, has_concentration_series=True),
        _SERVICE_C)
    agit = _by_exp(out, "record_agitation_and_seeding")
    assert agit["rank"] == 1
    assert agit["expected_degeneracy_reduction"] == 4.0
    assert agit.get("necessary_but_not_sufficient") is not True
    assert "also_requires" not in agit


def test_ranks_are_dense_and_sorted_by_priority():
    out = m9.recommend(
        _state(agitation_seeding_unknown=True, has_concentration_series=False,
               t50_censored=True, is_replicated=False), _SERVICE_C)
    recs = out["recommendations"]
    assert [r["rank"] for r in recs] == list(range(1, len(recs) + 1))
    prios = [r["priority_score"] for r in recs]
    assert prios == sorted(prios, reverse=True)


# --------------------------------------------------------------------------- #
# practical-vs-structural gate: REFUSE structural, recommend nothing for it
# --------------------------------------------------------------------------- #
def test_structural_flag_refuses_and_recommends_no_mechanism_experiment():
    out = m9.recommend(
        _state(structural_non_identifiable=True,
               agitation_seeding_unknown=True,    # would normally fire
               has_concentration_series=False,    # would normally fire
               assay_mass_unknown=True),          # would normally fire
        _SERVICE_C)
    assert out["refusals"], "structural state must produce a refusal"
    ref = out["refusals"][0]
    assert ref["refused_experiment"] is None
    assert "structural degeneracy" in ref["reason"]
    # NO mechanism-resolving experiment is recommended for a structural unit.
    exps = _experiments(out)
    assert "record_agitation_and_seeding" not in exps
    assert "add_concentration_series" not in exps
    assert "use_mass_proportional_assay" not in exps


def test_residual_saturating_secondary_pair_is_refused_as_structural():
    # the Service-C irreducible pair {fragmentation, saturating_secondary} a clean
    # series still cannot break -> treated as structural, refused.
    out = m9.recommend(
        _state(residual_pair=("saturating_secondary", "fragmentation"),
               has_concentration_series=False,
               agitation_seeding_unknown=True),
        _SERVICE_C)
    assert out["refusals"]
    assert out["refusals"][0]["addresses"] == "structural_non_identifiability"
    assert "add_concentration_series" not in _experiments(out)


def test_practical_non_identifiability_is_NOT_refused():
    # a practically-degenerate unit (broad single-curve class, no structural flag)
    # is addressable -> it gets recommendations, NOT a refusal.
    out = m9.recommend(
        _state(degeneracy_class="broad_single_curve",
               has_concentration_series=False,
               agitation_seeding_unknown=True), _SERVICE_C)
    assert out["refusals"] == []
    assert "add_concentration_series" in _experiments(out)


# --------------------------------------------------------------------------- #
# the concentration-series recommendation carries the MEASURED 1->4 narrowing
# --------------------------------------------------------------------------- #
def test_concentration_series_carries_service_c_measured_narrowing():
    out = m9.recommend(_state(has_concentration_series=False), _SERVICE_C)
    series = _by_exp(out, "add_concentration_series")
    m = series["service_c_measured"]
    assert m["single_concentration_classes"] == 1
    assert m["concentration_series_classes"] == 4
    assert m["resolution_gain_x"] == 4.0
    # the per-pair γ-separation AUC is carried (the quantified gain).
    assert m["pairwise_separation_auc"]["fragmentation|saturating_secondary"] == 0.88
    assert series["expected_degeneracy_reduction"] == 4.0
    assert series["expected_gain_is_upper_bound"] is True


def test_measured_narrowing_used_in_method_statement():
    out = m9.recommend(_state(), _SERVICE_C)
    ms = out["method_statement"]
    assert ms["method"] == "service_c_distilled_rule_table"
    assert "4x" in ms["committed"] or "4 singletons" in ms["committed"]


# --------------------------------------------------------------------------- #
# method statement + validity ceiling present; EIG explicitly deferred
# --------------------------------------------------------------------------- #
def test_method_statement_commits_rule_table_and_defers_eig():
    out = m9.recommend(_state(), _SERVICE_C)
    ms = out["method_statement"]
    assert ms["method"] == "service_c_distilled_rule_table"
    assert "DEFERRED" in ms["deferred"] and "EIG" in ms["deferred"]
    assert ms["cost_tiers_pre_registered"] == m9.COST_TIERS


def test_validity_ceiling_present_and_cites_mismatch():
    out = m9.recommend(_state(), _SERVICE_C)
    vc = out["validity_ceiling"]
    assert "UPPER bound" in vc["clause"]
    deg = vc["evidence_a_different_forward_model_degrades_resolution"][
        "mechanism_misidentification_degradation"]
    assert abs(deg - 0.5333333333333333) < 1e-9
    assert "PRACTICAL" in vc["practical_vs_structural"]


def test_every_recommendation_is_upper_bound():
    out = m9.recommend(
        _state(agitation_seeding_unknown=True, has_concentration_series=False,
               t50_censored=True, is_replicated=False, assay_mass_unknown=True),
        _SERVICE_C)
    assert out["recommendations"]
    assert all(r["expected_gain_is_upper_bound"] is True
               for r in out["recommendations"])


# --------------------------------------------------------------------------- #
# corpus rollup: top action = agitation/seeding; counts the ~33 it unblocks
# --------------------------------------------------------------------------- #
def _m8(gap_proteins=33, gap_series=48, gap=33):
    return {
        "mechanistically_resolvable_boundary": {
            "structural_reachable_proteins_upper_bound": gap_proteins,
            "structural_reachable_series_upper_bound": gap_series,
            "gap_structural_minus_licensed": gap,
        }
    }


def test_rollup_ranks_agitation_seeding_top_and_counts_gap():
    # a corpus with SERIES-BEARING units that need the metadata fix: there
    # recording agitation/seeding IS sufficient (last missing piece) -> it earns
    # full value and is the top corpus action that closes the GAP.
    units = [
        m9.recommend(_state(unit_id=f"U{i}", protein_id=f"P{i % 3}",
                            agitation_seeding_unknown=True,
                            has_concentration_series=True,  # series present
                            t50_censored=(i % 2 == 0), is_replicated=False),
                     _SERVICE_C)
        for i in range(6)
    ]
    roll = m9.build_rollup(units, _m8(), _SERVICE_C, engine_build_id="eb1")
    assert roll["top_corpus_action"] == "record_agitation_and_seeding"
    top = roll["corpus_actions_ranked"][0]
    assert top["experiment"] == "record_agitation_and_seeding"
    assert top["corpus_rank"] == 1
    # it advances every (series-bearing) unit at cheap cost, at FULL value.
    assert top["n_units_advanced"] == 6
    assert top["cost_tier"] == "cheap"
    assert top["priority_score"] == 4.0
    # dependency-aware split: all 6 are sufficient-with-series, 0 single-curve.
    assert top["n_units_sufficient_with_series"] == 6
    assert top["n_units_necessary_but_not_sufficient_single_curve"] == 0
    # the M8 GAP (~33) is carried + attributed to this action.
    assert roll["m8_gap"]["structural_reachable_proteins_upper_bound"] == 33
    assert roll["m8_gap"]["gap_structural_minus_licensed"] == 33
    assert roll["m8_gap"]["closed_by"] == "record_agitation_and_seeding"
    assert "33 proteins" in roll["headline"]
    assert "agitation" in roll["headline"].lower()


def test_rollup_headline_scopes_honestly_and_does_not_claim_all_units_advanced():
    # MIXED corpus: 2 series-bearing units (agitation sufficient) + 3 single-curve
    # units (agitation necessary-but-not-sufficient). The headline must scope the
    # mechanism unblock to the series-bearing units and NOT claim all 5 are
    # advanced toward mechanism.
    series_units = [
        m9.recommend(_state(unit_id=f"S{i}", protein_id=f"PS{i}",
                            agitation_seeding_unknown=True,
                            has_concentration_series=True), _SERVICE_C)
        for i in range(2)
    ]
    single_units = [
        m9.recommend(_state(unit_id=f"C{i}", protein_id=f"PC{i}",
                            agitation_seeding_unknown=True,
                            has_concentration_series=False), _SERVICE_C)
        for i in range(3)
    ]
    roll = m9.build_rollup(series_units + single_units, _m8(), _SERVICE_C)
    agit = next(a for a in roll["corpus_actions_ranked"]
                if a["experiment"] == "record_agitation_and_seeding")
    # the dependency split is reported honestly.
    assert agit["n_units_advanced"] == 5
    assert agit["n_units_sufficient_with_series"] == 2
    assert agit["n_units_necessary_but_not_sufficient_single_curve"] == 3
    # the headline must NOT claim it advances all 5 units toward mechanism.
    head = roll["headline"]
    assert "advances 5 analysis units" not in head
    # it DOES scope the unblock to the series-bearing units + names the single-
    # curve prerequisite.
    assert "series-bearing" in head
    assert "future" in head.lower()
    assert "add_concentration_series" in head or "concentration series" in head


def test_rollup_counts_units_with_structural_refusal():
    units = [
        m9.recommend(_state(unit_id="ok", agitation_seeding_unknown=True,
                            has_concentration_series=False), _SERVICE_C),
        m9.recommend(_state(unit_id="struct", structural_non_identifiable=True),
                     _SERVICE_C),
    ]
    roll = m9.build_rollup(units, _m8(), _SERVICE_C)
    assert roll["n_units_with_structural_refusal"] == 1


# --------------------------------------------------------------------------- #
# graceful degradation on missing inputs
# --------------------------------------------------------------------------- #
def test_missing_service_c_degrades_not_crash():
    out = m9.recommend(_state(has_concentration_series=False), service_c=None)
    series = _by_exp(out, "add_concentration_series")
    # falls back to the pre-registered measured numbers, stamped as such.
    assert series["service_c_measured"]["resolution_gain_x"] == 4.0
    assert "fallback" in out["method_statement"]["service_c_source"]


def test_missing_m8_degrades_rollup_not_crash():
    units = [m9.recommend(_state(agitation_seeding_unknown=True,
                                 has_concentration_series=False), _SERVICE_C)]
    roll = m9.build_rollup(units, m8=None, service_c=None)
    assert roll["is_degraded"] is True
    # still carries the known ~33 GAP from the Service-C/M8 fallback.
    assert roll["m8_gap"]["structural_reachable_proteins_upper_bound"] == 33


def test_empty_corpus_no_crash():
    roll = m9.build_rollup([], m8=None, service_c=None)
    assert roll["is_degraded"] is True
    assert roll["n_units"] == 0
    assert roll["top_corpus_action"] is None


def test_recommend_none_state_no_crash():
    out = m9.recommend(None, _SERVICE_C)
    assert out["n_recommendations"] >= 0
    assert "method_statement" in out


def test_output_is_json_serialisable():
    out = m9.recommend(
        _state(agitation_seeding_unknown=True, has_concentration_series=False),
        _SERVICE_C)
    roll = m9.build_rollup([out], _m8(), _SERVICE_C)
    json.dumps(out)   # must not raise
    json.dumps(roll)


def test_extract_state_from_full_m7_record():
    # a full M7-shaped record (not the flat synthetic) routes through the M7 path.
    rec = {
        "series_id": "CPAD-X",
        "protein_id": "ABeta42",
        "data_mode": "kinetic",
        "censoring_class": "left",
        "condition_vector": {
            "assay_reports_mass": True,
            "agitation": None, "seeded": None,
            "field_provenance": {"agitation": "unknown", "seeded": "unknown",
                                 "assay_type": "known"},
        },
        "mechanistic": {"degeneracy": "broad_single_curve",
                        "mechanistic_inference_licensed": False,
                        "verdict_or_equivalence_class": ["A", "B", "C", "D"]},
        "model_selection": {"identifiability": {"flag": "identifiable"}},
        "curve_features": {"t50_status": "biased_early"},
    }
    st = m9.extract_state(rec)
    assert st["unit_id"] == "CPAD-X"
    assert st["agitation_seeding_unknown"] is True
    assert st["t50_censored"] is True
    assert st["has_concentration_series"] is False
    assert st["structural_non_identifiable"] is False
    out = m9.recommend(rec, _SERVICE_C)
    # NO concentration series present -> add_concentration_series is the binding
    # prerequisite and ranks #1; recording agitation/seeding is necessary-but-not-
    # sufficient here (dependency-aware crediting).
    assert out["top_recommendation"] == "add_concentration_series"
    agit = _by_exp(out, "record_agitation_and_seeding")
    assert agit["necessary_but_not_sufficient"] is True


# --------------------------------------------------------------------------- #
def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
    print(f"test_m9: {passed} passed")
    return passed


if __name__ == "__main__":
    _run_all()
