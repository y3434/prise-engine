"""
Tests for M8 — Inferential Reachability Map.

Deterministic + fast. Operates on SYNTHETIC small artifact dicts only (does NOT
read the real data files), so it is hermetic and pins the contract:

  - ladder counts are monotone-non-increasing up the rungs and reconcile with N;
  - grid cells PARTITION the curves (sum == N);
  - the pre-registered inferential-status rule assigns
    recoverable/degenerate/unconstrained on crafted cells;
  - the blocker census ranks + counts correctly;
  - the resolvable-boundary GAP == structural - licensed;
  - the validity-ceiling block is present + non-empty + cites the mismatch evidence;
  - a missing artifact (e.g. no service_c) degrades, does not crash.

    python engine/test_m8.py
    pytest engine/test_m8.py
"""
from __future__ import annotations

import json

import m8_reachability as m8


# --------------------------------------------------------------------------- #
# synthetic artifact builders
# --------------------------------------------------------------------------- #
def _curve(series_id, *, fit="fittable", censor="none", flat=False,
           cs_id=None, mass=True, protein="PROT", uniprot="U1",
           known=("concentration", "pH", "temperature", "assay_type")):
    fp = {k: "known" for k in
          ["concentration", "pH", "temperature", "assay_type", "agitation", "seeded"]}
    for k in fp:
        if k not in known:
            fp[k] = "unknown"
    return {
        "series_id": series_id,
        "protein_id": protein,
        "uniprot_id": uniprot,
        "concentration_series_id": cs_id,
        "condition_vector": {
            "assay_reports_mass": mass,
            "field_provenance": fp,
        },
        "m1": {"fittability_class": fit, "censoring_class": censor,
               "flat_no_signal": flat},
    }


def _synthetic_artifacts():
    """A small corpus: 10 curves; a 3-curve mass concentration series; mix of
    fittability/censoring/completeness; one no-signal curve."""
    curves = []
    # a 3-concentration series, mass assay, full 4-field completeness -> recoverable
    for i in range(3):
        curves.append(_curve(f"S{i}", cs_id="CS1", mass=True))
    # 4 single curves, fittable, high completeness -> degenerate
    for i in range(4):
        curves.append(_curve(f"C{i}", fit="fittable", censor="left"))
    # 1 single curve, medium completeness (3 known)
    curves.append(_curve("C4", known=("concentration", "pH", "temperature")))
    # 1 single curve, low completeness (2 known) -> unconstrained
    curves.append(_curve("C5", known=("concentration", "pH")))
    # 1 no-signal curve (flat)
    curves.append(_curve("C6", fit="unfittable", flat=True))

    n = len(curves)  # 10
    rollup = {
        "engine_build_id": "test-build",
        "engine_build_components": {"corpus_version": "vtest"},
        "n_results": n,
        "by_information_yield": {
            "mechanistic": 0, "scaling": 2, "descriptive": 5,
            "signal_only": 2, "uninformative": 1,
        },
        "n_mechanistic_licensed": 0,
    }
    manifest = {"counts": {
        "curves": n, "distinct_proteins": 1,
        "proteins_with_concentration_series": 1,
        "concentration_series_groups": 1,
    }}
    service_c = {
        "service_c_version": "service-c-test",
        "mismatched_generator": {
            "misidentification_degradation": 0.5333333333333333,
            "matched_mechanism_id_accuracy": 0.5333333333333333,
        },
        "concentration_series_narrowing": {
            "single_concentration_partition": {"n_classes": 1},
            "concentration_series_partition": {"n_classes": 4},
            "series_narrows_degeneracy": True,
            "pairs_newly_resolved_by_series": [["a", "b"]],
        },
    }
    return {"curves_triaged": curves, "m7_rollup": rollup,
            "etl_manifest": manifest, "service_c": service_c}


def _synthetic_m11():
    """A minimal M11 corpus rollup (metadata_quality.json shape), enough to drive
    the metadata_explanation: the {agitation, seeded} universal blocker + a ranked
    corpus recommendation list whose #1 is record_agitation_and_seeding."""
    return {"corpus": {
        "version": "m11-metadata-1.0",
        "ontology_version": "m11-ontology-1.0",
        "n_curves": 10,
        "n_mechanism_licensed": 0,
        "metadata_score": {"distribution": {"median": 0.685185, "mean": 0.6887}},
        "mechanistic_completeness": {"distribution": {"median": 0.4, "mean": 0.356}},
        "universal_blocker": {
            "n_curves_agitation_or_seeding_blocked": 10,
            "fraction": 1.0,
            "why_mechanism_licensed_is_zero":
                "MECHANISM is licensed for 0 / 10 curves. The universal blocker is "
                "{agitation, seeded} = 'unknown' corpus-wide ...",
        },
        "corpus_recommendations_ranked": [
            {"rank": 1, "action": "record_agitation_and_seeding", "impact": "high",
             "effort": "cheap", "total_gain_over_effort": 14886.0, "curves_affected": 20},
            {"rank": 2, "action": "add_concentration_series", "impact": "high",
             "effort": "medium", "total_gain_over_effort": 1507.5, "curves_affected": 7},
            {"rank": 3, "action": "record_ionic_strength", "impact": "moderate",
             "effort": "cheap", "total_gain_over_effort": 1026.0, "curves_affected": 6},
        ],
    }}


# --------------------------------------------------------------------------- #
# 1. ladder
# --------------------------------------------------------------------------- #
def test_ladder_monotone_non_increasing():
    rmap = m8.build_reachability(_synthetic_artifacts())
    ladder = rmap["inference_ladder"]
    counts = [r["curves"] for r in ladder["rungs"]]
    assert counts == sorted(counts, reverse=True), counts
    assert ladder["monotone_non_increasing"] is True


def test_ladder_reconciles_with_totals():
    """scaling rung = mechanistic+scaling; descriptive = those + descriptive."""
    arts = _synthetic_artifacts()
    rmap = m8.build_reachability(arts)
    rungs = {r["rung"]: r["curves"] for r in rmap["inference_ladder"]["rungs"]}
    by = arts["m7_rollup"]["by_information_yield"]
    assert rungs["mechanistic_licensed"] == by["mechanistic"]
    assert rungs["scaling"] == by["mechanistic"] + by["scaling"]
    assert rungs["descriptive_regime"] == by["mechanistic"] + by["scaling"] + by["descriptive"]
    # signal_present = curves with usable signal (one is flat -> 9 of 10)
    assert rungs["signal_present"] == 9
    # N comes from the manifest census
    assert rmap["inference_ladder"]["n_curves"] == 10


def test_ladder_statements_carry_conditioning_clause():
    rmap = m8.build_reachability(_synthetic_artifacts())
    for r in rmap["inference_ladder"]["rungs"]:
        assert "PRISE forward model" in r["curve_statement"]


# --------------------------------------------------------------------------- #
# 2. grid partition + status rule
# --------------------------------------------------------------------------- #
def test_grid_cells_partition_curves():
    arts = _synthetic_artifacts()
    rmap = m8.build_reachability(arts)
    grid = rmap["reachability_grid"]
    total = sum(c["count"] for c in grid["cells"].values())
    assert total == len(arts["curves_triaged"])
    assert grid["partition_ok"] is True
    assert sum(grid["n_curves_by_status"].values()) == len(arts["curves_triaged"])


def test_status_rule_assigns_recoverable_degenerate_unconstrained():
    """Crafted cells hit each status per the pre-registered STATUS_RULE."""
    # concentration_series + high -> recoverable
    assert m8.STATUS_RULE[("concentration_series", "high")][0] == "recoverable"
    # single_curve + high -> degenerate (descriptive ok, mechanism degenerate)
    assert m8.STATUS_RULE[("single_curve", "high")][0] == "degenerate"
    # single_curve + low -> unconstrained
    assert m8.STATUS_RULE[("single_curve", "low")][0] == "unconstrained"
    # and the built grid carries those statuses on the populated cells
    rmap = m8.build_reachability(_synthetic_artifacts())
    cells = rmap["reachability_grid"]["cells"]
    assert cells["concentration_series|high"]["inferential_status"] == "recoverable"
    assert cells["single_curve|high"]["inferential_status"] == "degenerate"
    assert cells["single_curve|low"]["inferential_status"] == "unconstrained"


def test_completeness_bins_thresholds():
    # 4 known -> high, 3 -> medium, 2 -> low
    assert m8._completeness_bin({k: "known" for k in
                                 ["concentration", "pH", "temperature", "assay_type"]})[1] == "high"
    assert m8._completeness_bin({k: "known" for k in
                                 ["concentration", "pH", "temperature"]})[1] == "medium"
    assert m8._completeness_bin({k: "known" for k in
                                 ["concentration", "pH"]})[1] == "low"


# --------------------------------------------------------------------------- #
# 3. blocker census
# --------------------------------------------------------------------------- #
def test_blocker_census_ranks_and_counts():
    arts = _synthetic_artifacts()
    rmap = m8.build_reachability(arts)
    blockers = rmap["blocker_census"]["blockers_impact_ranked"]
    # agitation/seeding unknown for ALL 10 -> top blocker, count == N
    assert blockers[0]["blocker"] == "agitation_seeding_unknown"
    assert blockers[0]["count"] == 10
    # impact ranks are strictly increasing and counts are sorted descending
    ranks = [b["impact_rank"] for b in blockers]
    assert ranks == list(range(1, len(blockers) + 1))
    counts = [b["count"] for b in blockers]
    assert counts == sorted(counts, reverse=True)
    # no_concentration_series = 10 - 3 (the series) = 7
    nos = next(b for b in blockers if b["blocker"] == "no_concentration_series")
    assert nos["count"] == 7


def test_blocker_censoring_counts_only_censored():
    arts = _synthetic_artifacts()
    rmap = m8.build_reachability(arts)
    cens = next(b for b in rmap["blocker_census"]["blockers_impact_ranked"]
                if b["blocker"] == "censoring_biases_t50")
    # 4 'left'-censored single curves in the synthetic set
    assert cens["count"] == 4


# --------------------------------------------------------------------------- #
# 4. mechanistically-resolvable boundary GAP
# --------------------------------------------------------------------------- #
def test_resolvable_boundary_gap_equals_structural_minus_licensed():
    arts = _synthetic_artifacts()
    rmap = m8.build_reachability(arts)
    b = rmap["mechanistically_resolvable_boundary"]
    assert b["gap_structural_minus_licensed"] == (
        b["structural_reachable_proteins_upper_bound"]
        - b["actual_mechanistically_licensed"])
    # 1 protein with a mass series; 0 licensed -> gap 1
    assert b["structural_reachable_proteins_upper_bound"] == 1
    assert b["actual_mechanistically_licensed"] == 0
    assert b["gap_structural_minus_licensed"] == 1
    assert b["is_upper_bound"] is True


def test_resolvable_boundary_excludes_non_mass_series():
    """A non-mass concentration series is NOT structurally mechanism-reachable."""
    arts = _synthetic_artifacts()
    # make the series non-mass
    for cu in arts["curves_triaged"]:
        if cu["concentration_series_id"] == "CS1":
            cu["condition_vector"]["assay_reports_mass"] = False
    rmap = m8.build_reachability(arts)
    b = rmap["mechanistically_resolvable_boundary"]
    assert b["observed_structural_proteins_mass_series"] == 0


# --------------------------------------------------------------------------- #
# 5. validity ceiling
# --------------------------------------------------------------------------- #
def test_validity_ceiling_present_and_nonempty():
    rmap = m8.build_reachability(_synthetic_artifacts())
    vc = rmap["validity_ceiling"]
    assert vc["clause"] and "UPPER bound" in vc["clause"]
    ev = vc["evidence_a_different_forward_model_degrades_resolution"]
    assert ev["mechanism_misidentification_degradation"] == 0.5333333333333333
    assert vc["deferred_external_validation"]["status"] == "DEFERRED"
    assert vc["deferred_external_validation"]["items"]


def test_conditioning_clause_top_level():
    rmap = m8.build_reachability(_synthetic_artifacts())
    assert "PRISE FORWARD MODEL" in rmap["conditioning_clause"]
    assert rmap["validity_ceiling"]  # prominent block exists


# --------------------------------------------------------------------------- #
# 6. graceful degradation + query + serialisability
# --------------------------------------------------------------------------- #
def test_missing_service_c_degrades_no_crash():
    arts = _synthetic_artifacts()
    arts["service_c"] = None
    rmap = m8.build_reachability(arts)  # must not raise
    assert rmap["is_degraded"] is True
    assert any("service_c" in d for d in rmap["degraded"])
    # validity ceiling still present, evidence value just None
    vc = rmap["validity_ceiling"]
    assert vc["clause"]
    assert vc["evidence_a_different_forward_model_degrades_resolution"][
        "mechanism_misidentification_degradation"] is None


def test_all_empty_artifacts_no_crash():
    rmap = m8.build_reachability({})  # everything missing
    assert rmap["is_degraded"] is True
    assert rmap["validity_ceiling"]["clause"]
    # ladder still emits the rungs (all zero / fallback), no crash
    assert len(rmap["inference_ladder"]["rungs"]) == len(m8.LADDER)


def test_output_is_json_serialisable():
    rmap = m8.build_reachability(_synthetic_artifacts())
    s = json.dumps(rmap)  # must not raise
    assert json.loads(s)["m8_version"] == "m8-reachability-1.0"


def test_reachable_at_query():
    rmap = m8.build_reachability(_synthetic_artifacts())
    q = m8.reachable_at(rmap, "scaling")
    assert q["level"] == "scaling"
    assert q["curves"] == 2  # mechanistic(0) + scaling(2)
    assert "PRISE forward model" in q["statement"]
    bad = m8.reachable_at(rmap, "nonsense")
    assert "error" in bad


# --------------------------------------------------------------------------- #
# 7. M11 metadata_explanation (additive, graceful)
# --------------------------------------------------------------------------- #
def test_metadata_explanation_present_when_m11_exists():
    """When the M11 artifact is supplied, M8 attaches a CAUSAL metadata_explanation
    that cites the {agitation, seeded} universal blocker + the ranked why_blocked
    on the boundary (with the Service-C-measured #1 action)."""
    arts = _synthetic_artifacts()
    arts["m11_metadata"] = _synthetic_m11()
    rmap = m8.build_reachability(arts)

    me = rmap["metadata_explanation"]
    assert me["m11_available"] is True
    # the universal blocker is the {agitation, seeded} pair, unknown corpus-wide
    assert me["universal_blocker"]["fields"] == ["agitation", "seeded"]
    assert me["universal_blocker"]["fraction"] == 1.0
    assert me["universal_blocker"]["n_mechanism_licensed"] == 0
    # why_blocked ranks record_agitation_and_seeding #1, Service-C-measured
    wb = me["why_blocked"]
    assert wb and wb[0]["m11_action"] == "record_agitation_and_seeding"
    assert wb[0]["gain_basis"] == "service_c_measured"
    assert "agitation" in wb[0]["m11_fields"] and "seeding" in wb[0]["m11_fields"]

    # top-level completeness summary pulls the M11 medians
    mcs = rmap["metadata_completeness_summary"]
    assert mcs["m11_available"] is True
    assert mcs["median_metadata_score"] == 0.685185
    assert mcs["median_mechanistic_completeness"] == 0.4

    # the boundary carries WHY the gap is unresolved, citing {agitation, seeded}
    wg = rmap["mechanistically_resolvable_boundary"]["why_gap_unresolved"]
    assert wg["m11_available"] is True
    assert wg["universal_blocker_fields"] == ["agitation", "seeded"]
    assert "agitation" in wg["cause"] and "seeded" in wg["cause"]
    assert wg["why_blocked"][0]["m11_action"] == "record_agitation_and_seeding"
    # honesty: completeness ≠ correctness + information-gain upper-bound carried
    assert "completeness_neq_correctness" in wg
    assert "information_gain_upper_bound" in wg

    # each mappable blocker is cross-referenced to its M11 field(s) + gain-to-unblock
    census = rmap["blocker_census"]["blockers_impact_ranked"]
    top = next(b for b in census if b["blocker"] == "agitation_seeding_unknown")
    mx = top["m11_explanation"]
    assert mx["m11_fields_responsible"] == ["agitation", "seeding"]
    assert mx["m11_action_to_unblock"] == "record_agitation_and_seeding"
    assert mx["gain_basis"] == "service_c_measured"
    assert mx["m11_information_gain_over_effort"] == 14886.0

    # serialisable end-to-end
    json.dumps(rmap)


def test_metadata_explanation_degrades_when_m11_absent():
    """With NO M11 artifact, M8 still works: metadata_explanation degrades to
    {m11_available: False} and EVERY existing M8 key + the validity ceiling is intact."""
    arts = _synthetic_artifacts()           # no m11_metadata key at all
    rmap = m8.build_reachability(arts)       # must not raise

    me = rmap["metadata_explanation"]
    assert me["m11_available"] is False
    assert "note" in me and "m11_metadata.json" in me["note"] or me["note"]
    # completeness summary also degrades gracefully
    assert rmap["metadata_completeness_summary"]["m11_available"] is False
    # the boundary WHY is degraded but the boundary + GAP themselves are untouched
    b = rmap["mechanistically_resolvable_boundary"]
    assert b["why_gap_unresolved"]["m11_available"] is False
    assert b["gap_structural_minus_licensed"] == 1
    assert b["structural_reachable_proteins_upper_bound"] == 1
    # blockers keep their existing shape (no m11_explanation attached)
    top = next(x for x in rmap["blocker_census"]["blockers_impact_ranked"]
               if x["blocker"] == "agitation_seeding_unknown")
    assert "m11_explanation" not in top
    # validity-ceiling honesty is fully intact (unchanged)
    assert "UPPER bound" in rmap["validity_ceiling"]["clause"]
    assert rmap["conditioning_clause"]
    # M8 flags the M11 degradation in the degraded list
    assert any("m11_metadata" in d for d in rmap["degraded"])
    json.dumps(rmap)


def test_metadata_explanation_none_m11_no_crash():
    """An explicitly-None m11_metadata (missing file) also degrades, never crashes."""
    arts = _synthetic_artifacts()
    arts["m11_metadata"] = None
    rmap = m8.build_reachability(arts)
    assert rmap["metadata_explanation"]["m11_available"] is False


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
    print(f"test_m8: {passed} passed")
    return passed


if __name__ == "__main__":
    _run_all()
