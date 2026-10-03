"""
Tests for M7 — output assembler (yield ladder, compact/full contract,
engine_build_id determinism + comparability, M9-hook rule table, caveat
impact-ranking + cap, and graceful handling of missing inputs).

Synthetic input dicts only — NO dependence on the real data artifacts.

    python engine/test_m7.py
    pytest engine/test_m7.py
"""
from __future__ import annotations

from m7_assemble import (
    YIELD_LADDER,
    YIELD_RANK,
    assemble_curve,
    build_components,
    build_rollup,
    collect_caveats,
    comparable,
    engine_build_id,
    gamma_significant,
    information_yield,
    m9_hook,
)

BUILD = "deadbeef0000"


# --------------------------- synthetic builders ---------------------------- #
def _m1(fittability="fittable", censoring="point", **kw):
    d = {"fittability_class": fittability, "censoring_class": censoring,
         "signal_basis": "normalized", "reasons": [], "normalization_mode": "min_max"}
    d.update(kw)
    return d


def _feat(status="ok", regime_shape=True, **kw):
    d = {"status": status, "signal_basis": "normalized", "censoring_class": "point",
         "definition_contract": "m4-defs-1.0", "t50_status": "point",
         "lag_status": "ok", "validity_flags": [],
         "features": {"lag_to_t50_ratio": 0.3, "t50": 10.0}}
    d.update(kw)
    return d


def _regime(regime="cooperative_sigmoidal", licensed=False, **kw):
    d = {"descriptive_regime": {"regime": regime, "definition": "x", "evidence": "y"},
         "mechanistic": {"mechanistic_inference_licensed": licensed,
                         "equivalence_class": ["primary_nucleation_dominated"],
                         "single_mechanism_call": None, "gates_failed": ["g1", "g2"],
                         "degeneracy": "broad_single_curve"},
         "anomaly": {"testable": True, "fdr_significant": False,
                     "material_misfit": False, "proposal_actionable": False},
         "confidence": {"level": "medium"},
         "window_of_validity": {"t_min_hours": 0.0, "t_max_hours": 100.0}}
    d.update(kw)
    return d


def _gamma(reliable=True, ci=None):
    """Synthetic γ record. `ci` is the gamma_regression 95% CI: pass a CI that
    EXCLUDES 0 (e.g. [0.6, 1.6]) for a SIGNIFICANT γ (→ scaling tier), or one that
    INCLUDES 0 (e.g. [-0.4, 1.6], the default) for a reliable-but-not-significant γ
    (→ descriptive tier + the 'CI includes zero' caveat)."""
    if ci is None:
        ci = [-0.4, 1.6]   # straddles 0: reliable fit, but no scaling resolved
    return {"concentration_series_id": "CS1",
            "gamma_regression": {"gamma": 1.1, "gamma_reliable": reliable,
                                 "gamma_ci": ci,
                                 "reliability_reasons": ["too few uncensored anchors"]},
            "gamma_global": {"gamma": 1.0}, "disagreement": {}, "n_member_curves": 5}


def _gamma_significant():
    """A γ that lands a curve on the `scaling` tier: reliable AND CI excludes 0."""
    return _gamma(reliable=True, ci=[0.6, 1.6])


def _triage(series_id="S1", cs_id=None, **kw):
    d = {"series_id": series_id, "protein_id": "Prot-A", "uniprot_id": "P1",
         "source": "bundled:CPAD2.0", "data_mode": "kinetic",
         "concentration_series_id": cs_id,
         "fittability_class": "fittable", "censoring_class": "point",
         "normalization_mode": "min_max", "digitization_uncertainty": False,
         "quality_flags": {}, "m1": _m1(),
         "condition_vector": {"assay_type": "ThT", "construct_id": "Wild Type",
                              "assay_reports_mass": True,
                              "agitation": "shaken", "seeded": "de_novo",
                              "field_provenance": {"agitation": "known",
                                                   "seeded": "known"}}}
    d.update(kw)
    return d


# ============================ yield ladder ================================== #
def test_yield_mechanistic():
    y = information_yield(_m1(), _feat(), _regime(licensed=True), _gamma(True))
    assert y == "mechanistic"


def test_yield_scaling():
    # not mechanistic, but a SIGNIFICANT γ (reliable AND CI excludes 0) exists.
    y = information_yield(_m1(), _feat(), _regime(licensed=False), _gamma_significant())
    assert y == "scaling"


def test_yield_reliable_but_ci_includes_zero_is_descriptive():
    # the honesty contract: a γ that is RELIABLE but whose 95% CI INCLUDES 0 does
    # NOT resolve a scaling exponent -> the curve falls to `descriptive`, not
    # `scaling`, and earns the "CI includes zero" caveat (no scaling resolved).
    g = _gamma(reliable=True, ci=[-0.4, 1.6])   # straddles 0
    y = information_yield(_m1(), _feat(status="ok"),
                          _regime("cooperative_sigmoidal", licensed=False), g)
    assert y == "descriptive"
    caveats = collect_caveats(_m1(), _feat(status="ok"),
                              _regime("cooperative_sigmoidal", licensed=False), g,
                              {"field_provenance": {}}, has_concentration_series=True,
                              cap=3)
    assert any("CI includes zero" in c for c in caveats)


def test_yield_descriptive():
    # not mechanistic, no reliable γ (no series) but real shape + ok features.
    y = information_yield(_m1(), _feat(status="ok"),
                          _regime("gradual_non_cooperative", licensed=False), None)
    assert y == "descriptive"


def test_yield_scaling_beats_descriptive():
    # an unreliable γ does NOT promote to scaling -> falls to descriptive.
    y = information_yield(_m1(), _feat(status="ok"),
                          _regime("cooperative_sigmoidal"), _gamma(reliable=False))
    assert y == "descriptive"


def test_yield_signal_only_flat():
    y = information_yield(_m1(), _feat(status="flat_no_transition"),
                          _regime("no_detectable_aggregation"), None)
    assert y == "signal_only"


def test_yield_uninformative_unfittable():
    y = information_yield(_m1(fittability="unfittable"), None, None, None)
    assert y == "uninformative"


def test_yield_uninformative_no_fit():
    y = information_yield(_m1(), _feat(status="no_fit"),
                          _regime("anomalous_unclassified"), None)
    assert y == "uninformative"


def test_yield_ladder_is_sortable():
    # the const exposes a total order, highest -> lowest.
    assert YIELD_LADDER[0] == "mechanistic"
    assert YIELD_LADDER[-1] == "uninformative"
    assert YIELD_RANK["mechanistic"] < YIELD_RANK["uninformative"]
    assert sorted(YIELD_LADDER, key=lambda t: YIELD_RANK[t]) == YIELD_LADDER


# ===================== compact vs full contract ============================ #
_HEAVY_FIELDS = {"model_selection", "curve_features", "gamma_regression",
                 "quality_report", "sequence_axis", "propensity", "mechanistic"}


def test_full_result_has_heavy_fields():
    # a SIGNIFICANT γ -> scaling tier -> full schema with every heavy field.
    r = assemble_curve(_triage(cs_id="CS1"), _feat(),
                       _regime("cooperative_sigmoidal"), _gamma_significant(),
                       None, {"best_by_aicc": "logistic"}, None, BUILD)
    assert r["schema"] == "full"
    assert r["information_yield"] == "scaling"
    for f in _HEAVY_FIELDS:
        assert f in r, f"full result missing {f}"
    assert r["engine_build_id"] == BUILD


def test_compact_result_omits_heavy_fields():
    # flat curve -> signal_only -> compact.
    r = assemble_curve(_triage(), _feat(status="flat_no_transition"),
                       _regime("no_detectable_aggregation"), None,
                       None, None, None, BUILD)
    assert r["schema"] == "compact"
    assert r["information_yield"] == "signal_only"
    for f in _HEAVY_FIELDS:
        assert f not in r, f"compact result should NOT carry {f}"
    # compact carries exactly the contract fields
    assert set(r) >= {"protein_id", "uniprot_id", "series_id", "condition_vector",
                      "descriptive_regime", "information_yield", "m9_hook",
                      "caveats", "engine_build_id"}
    assert len(r["caveats"]) <= 3
    assert "suggestion" in r["m9_hook"]


def test_compact_for_uninformative():
    r = assemble_curve(_triage(m1=_m1(fittability="unfittable")), None, None, None,
                       None, None, None, BUILD)
    assert r["schema"] == "compact"
    assert r["information_yield"] == "uninformative"


# ===================== engine_build_id determinism ========================= #
def test_build_id_deterministic():
    c = build_components(corpus_version="abc")
    assert engine_build_id(c) == engine_build_id(dict(reversed(list(c.items()))))


def test_build_id_changes_with_version():
    a = build_components(corpus_version="abc")
    b = build_components(corpus_version="abc")
    b2 = dict(b)
    b2["m5_regime_registry"] = "m5-regimes-2.0"   # a taxonomy bump
    assert engine_build_id(a) != engine_build_id(b2)


def test_build_id_changes_with_corpus():
    assert (engine_build_id(build_components(corpus_version="abc"))
            != engine_build_id(build_components(corpus_version="def")))


def test_comparable_gate():
    a = engine_build_id(build_components(corpus_version="abc"))
    b = engine_build_id(build_components(corpus_version="abc"))
    c = engine_build_id(build_components(corpus_version="zzz"))
    assert comparable(a, b) is True
    assert comparable(a, c) is False
    assert comparable("", "") is False        # empty id never comparable


def test_build_id_short_and_stable():
    bid = engine_build_id(build_components(corpus_version="abc"))
    assert isinstance(bid, str) and len(bid) == 12


# ======================== M9-hook rule table =============================== #
def test_m9_no_concentration_series():
    h = m9_hook(_m1(), _feat(), _regime(), None, {}, has_concentration_series=False)
    assert h["rule_fired"] == "no_concentration_series"
    assert "concentration decade" in h["suggestion"]
    assert "stub" in h


def test_m9_agitation_unknown():
    cv = {"agitation": None, "seeded": None,
          "field_provenance": {"agitation": "unknown", "seeded": "unknown"}}
    h = m9_hook(_m1(), _feat(), _regime(), _gamma(), cv, has_concentration_series=True)
    assert h["rule_fired"] == "agitation_or_seeding_unknown"


def test_m9_t50_censored():
    cv = {"agitation": "shaken", "seeded": "de_novo",
          "field_provenance": {"agitation": "known", "seeded": "known"}}
    h = m9_hook(_m1(censoring="left"), _feat(t50_status="biased_early"),
               _regime(), _gamma(), cv, has_concentration_series=True)
    assert h["rule_fired"] == "t50_censored"
    assert "de-censor" in h["suggestion"]


def test_m9_material_misfit():
    cv = {"agitation": "shaken", "seeded": "de_novo",
          "field_provenance": {"agitation": "known", "seeded": "known"}}
    reg = _regime()
    reg["anomaly"]["material_misfit"] = True
    h = m9_hook(_m1(), _feat(t50_status="point"), reg, _gamma(), cv,
               has_concentration_series=True)
    assert h["rule_fired"] == "anomaly_material_misfit"


def test_m9_default_replicates():
    cv = {"agitation": "shaken", "seeded": "de_novo",
          "field_provenance": {"agitation": "known", "seeded": "known"}}
    h = m9_hook(_m1(), _feat(t50_status="point"), _regime(), _gamma(), cv,
               has_concentration_series=True)
    assert h["rule_fired"] == "default_add_replicates"
    assert "replicates" in h["suggestion"]


# ===================== caveats: impact rank + cap ========================== #
def test_caveats_capped_at_three():
    # pile on many true caveats; only the top-3 by impact survive.
    cv = {"assay_reports_mass": False, "digitization_uncertainty": True,
          "field_provenance": {}}
    caveats = collect_caveats(
        _m1(censoring="right", digitization_uncertainty=True),
        _feat(t50_status="biased_early", signal_basis="raw_au"),
        _regime(licensed=False),
        _gamma(reliable=False),
        cv, has_concentration_series=True, cap=3)
    assert len(caveats) == 3
    # impact-1 (mechanism not licensed) must be present; impact-6 (digitization) must not.
    assert any("mechanistic inference not licensed" in c for c in caveats)
    assert not any("digitization" in c for c in caveats)


def test_caveats_deduplicated():
    # no concentration series + unreliable γ should not both fire (mutually exclusive
    # branch); ensure no duplicate text and impact order preserved.
    caveats = collect_caveats(_m1(), _feat(), _regime(licensed=False), None,
                              {"field_provenance": {}},
                              has_concentration_series=False, cap=3)
    assert len(caveats) == len(set(caveats))
    assert any("no concentration series" in c for c in caveats)


def test_caveats_clean_curve_short():
    # a clean, licensed, point-t50, γ-reliable curve has few caveats (no avalanche).
    caveats = collect_caveats(_m1(), _feat(t50_status="point"),
                              _regime(licensed=True), _gamma(True),
                              {"assay_reports_mass": True, "field_provenance": {}},
                              has_concentration_series=True, cap=3)
    assert len(caveats) <= 3


# ===================== graceful missing-input handling ===================== #
def test_missing_m3_falls_back_to_m2():
    # full result, no M3 record -> model_selection from M2 with the no-bootstrap flag.
    r = assemble_curve(_triage(cs_id="CS1"), _feat(),
                       _regime("cooperative_sigmoidal"), _gamma(True),
                       None, {"best_by_aicc": "gompertz"}, None, BUILD)
    ms = r["model_selection"]
    assert ms["source"] == "m2_point_aicc"
    assert ms["best_point"] == "gompertz"
    assert "point_selection_no_bootstrap" in ms["flag"]
    assert ms["selection_frequencies"] is None


def test_present_m3_used():
    m3 = {"best_point": "lnt", "best_descriptive": "lnt",
          "best_mechanistic": "finke_watzky", "ci_reliable": True,
          "identifiability_of_best": {"flag": "identifiable"},
          "selection": {"selection_frequencies": {"lnt": 0.9},
                        "selection_stability": 0.9,
                        "t50_predictive_interval": [9, 10, 11],
                        "t50_multimodal": False}}
    r = assemble_curve(_triage(cs_id="CS1"), _feat(),
                       _regime("cooperative_sigmoidal"), _gamma(True),
                       m3, {"best_by_aicc": "x"}, None, BUILD)
    ms = r["model_selection"]
    assert ms["source"] == "m3_bootstrap"
    assert ms["selection_stability"] == 0.9
    assert ms["identifiability"]["flag"] == "identifiable"


def test_missing_propensity_absent_flagged():
    r = assemble_curve(_triage(cs_id="CS1"), _feat(),
                       _regime("cooperative_sigmoidal"), _gamma(True),
                       None, {"best_by_aicc": "x"}, None, BUILD)
    assert r["propensity"]["available"] is False
    assert r["propensity"]["uniprot_id"] == "P1"
    assert r["sequence_axis"]["available"] is False


def _prop(construct="Wild Type", assay="ThT", uniprot="P1"):
    """Synthetic M6 propensity record. `construct`/`assay` populate the record's
    own condition_vector so construct-matching (item 4) can be exercised."""
    return {"uniprot_id": uniprot, "assay": assay, "scored_feature": "t50",
            "feature_status": "point", "condition_match": "assay_only",
            "condition_vector": {"construct_id": construct},
            "propensity": {"surface": {"available": True},
                           "intrinsic": {"value": 10, "anchor_kind": "corpus"},
                           "cohort": {"percentile": 50, "N": 20}},
            "sequence_axis": {"available": True, "sub_axes": {}}}


def test_present_propensity_joined():
    # construct-matched record (triage construct == prop construct == "Wild Type").
    prop = _prop(construct="Wild Type")
    r = assemble_curve(_triage(cs_id="CS1"), _feat(),
                       _regime("cooperative_sigmoidal"), _gamma_significant(),
                       None, {"best_by_aicc": "x"}, prop, BUILD)
    assert r["propensity"]["available"] is True
    assert r["propensity"]["construct_match"] is True
    assert r["propensity"]["prop_construct_id"] == "Wild Type"
    assert r["propensity"]["intrinsic"]["anchor_kind"] == "corpus"
    assert r["sequence_axis"]["available"] is True


def test_unfittable_curve_is_compact_uninformative():
    t = _triage(m1=_m1(fittability="unfittable"), fittability_class="unfittable")
    r = assemble_curve(t, None, None, None, None, None, None, BUILD)
    assert r["schema"] == "compact"
    assert r["information_yield"] == "uninformative"
    assert r["series_id"] == "S1"
    assert "m9_hook" in r and "caveats" in r


def test_assemble_never_raises_on_empty_inputs():
    # the contract: never raise on bad/empty input.
    r = assemble_curve({"series_id": "S0"}, None, None, None, None, None, None, BUILD)
    assert r["series_id"] == "S0"
    assert r["information_yield"] in YIELD_LADDER


# ============= NEW: item 2 — rollup reports BOTH γ counts ================== #
def _full_result(yield_tier_gamma, **kw):
    """Assemble one FULL result for rollup tests; yield_tier_gamma is the γ record."""
    return assemble_curve(_triage(cs_id="CS1"), _feat(),
                          _regime("cooperative_sigmoidal"), yield_tier_gamma,
                          None, {"best_by_aicc": "x"}, None, BUILD)


def _rollup_of(results):
    comps = build_components(corpus_version="abc")
    return build_rollup(results, BUILD, comps, regimes_by_id={})


def test_rollup_reports_both_gamma_counts():
    # one SIGNIFICANT γ (CI excludes 0 -> scaling) and one reliable-but-CI-spans-0 γ
    # (reliable fit, NOT significant -> descriptive). Both are reliable fits; only
    # one is significant -> the two rollup counts must differ accordingly.
    r_sig = _full_result(_gamma_significant())                 # scaling
    r_rel = _full_result(_gamma(reliable=True, ci=[-0.4, 1.6]))  # descriptive
    roll = _rollup_of([r_sig, r_rel])
    assert "n_with_reliable_gamma_fit" in roll
    assert "n_with_significant_gamma" in roll
    assert "n_with_reliable_gamma" not in roll          # the overstated single count is gone
    assert roll["n_with_reliable_gamma_fit"] == 2       # both fits are reliable
    assert roll["n_with_significant_gamma"] == 1         # only one CI excludes 0
    assert roll["by_information_yield"]["scaling"] == 1


def test_rollup_tier_counts_sum_to_n_results():
    results = [
        _full_result(_gamma_significant()),                       # scaling
        _full_result(_gamma(reliable=True, ci=[-0.4, 1.6])),      # descriptive
        assemble_curve(_triage(), _feat(status="flat_no_transition"),
                       _regime("no_detectable_aggregation"), None,
                       None, None, None, BUILD),                  # signal_only
        assemble_curve(_triage(m1=_m1(fittability="unfittable")), None, None, None,
                       None, None, None, BUILD),                  # uninformative
    ]
    roll = _rollup_of(results)
    assert roll["n_results"] == len(results)
    assert sum(roll["by_information_yield"].values()) == roll["n_results"]
    # schema counts also partition the results
    assert sum(roll["schema_counts"].values()) == roll["n_results"]


def test_rollup_null_regime_bucket_labeled_and_sorted():
    # a result with no resolved regime maps to the readable "(no resolved regime)"
    # key, and the dict is sorted (item 7).
    no_regime = assemble_curve(_triage(), _feat(status="flat_no_transition"),
                               _regime(regime=None), None, None, None, None, BUILD)
    roll = _rollup_of([no_regime, _full_result(_gamma_significant())])
    assert "(no resolved regime)" in roll["by_descriptive_regime"]
    assert None not in roll["by_descriptive_regime"]
    keys = list(roll["by_descriptive_regime"].keys())
    assert keys == sorted(keys, key=lambda k: (k == "(no resolved regime)", k))


def test_rollup_has_corpus_caveats():
    roll = _rollup_of([_full_result(_gamma_significant())])
    assert isinstance(roll["corpus_caveats"], list) and len(roll["corpus_caveats"]) >= 2
    assert any("digitization" in c for c in roll["corpus_caveats"])
    assert any("agitation" in c for c in roll["corpus_caveats"])


# ============= NEW: item 3 — mechanistic.confidence None when unlicensed ==== #
def test_mechanistic_confidence_none_when_unlicensed():
    # unlicensed mechanism (the corpus-wide case): mechanistic.confidence MUST be
    # None (no borrowing the M5 shape confidence), but the shape confidence is
    # preserved verbatim in shape_confidence + descriptive_regime.confidence.
    r = assemble_curve(_triage(cs_id="CS1"), _feat(),
                       _regime("cooperative_sigmoidal", licensed=False),
                       _gamma_significant(), None, {"best_by_aicc": "x"}, None, BUILD)
    assert r["mechanistic"]["mechanistic_inference_licensed"] is False
    assert r["mechanistic"]["confidence"] is None
    # the M5 descriptive-shape confidence is NOT lost — it's preserved, just relabeled.
    assert r["shape_confidence"]["level"] == "medium"
    assert r["descriptive_regime"]["confidence"]["level"] == "medium"


def test_mechanistic_confidence_present_when_licensed():
    # when a mechanism IS licensed, the confidence level is carried (not nulled).
    r = assemble_curve(_triage(cs_id="CS1"), _feat(),
                       _regime("cooperative_sigmoidal", licensed=True),
                       _gamma_significant(), None, {"best_by_aicc": "x"}, None, BUILD)
    assert r["mechanistic"]["mechanistic_inference_licensed"] is True
    assert r["mechanistic"]["confidence"] == "medium"


# ============= NEW: item 4 — propensity construct mismatch never WT ========= #
def test_propensity_construct_mismatch_not_attached():
    # triage is a mutant construct "A30P"; the only M6 record is Wild Type ->
    # available=False, construct_match=False, WT propensity NEVER attached.
    t = _triage(cs_id="CS1")
    t["condition_vector"]["construct_id"] = "A30P"        # mutant curve
    wt_prop = _prop(construct="Wild Type")
    r = assemble_curve(t, _feat(), _regime("cooperative_sigmoidal"),
                       _gamma_significant(), None, {"best_by_aicc": "x"}, wt_prop, BUILD)
    assert r["propensity"]["available"] is False
    assert r["propensity"]["construct_match"] is False
    assert r["propensity"]["prop_construct_id"] == "Wild Type"
    assert r["propensity"]["prop_assay"] == "ThT"
    # the WT propensity values must NOT have leaked into the block.
    assert "intrinsic" not in r["propensity"]
    # but the protein-level sequence_axis may still surface (construct-independent).
    assert r["sequence_axis"]["available"] is True


def test_propensity_construct_match_surfaces_audit_fields():
    # a matched record always surfaces the audit fields too.
    prop = _prop(construct="Wild Type")
    r = assemble_curve(_triage(cs_id="CS1"), _feat(),
                       _regime("cooperative_sigmoidal"), _gamma_significant(),
                       None, {"best_by_aicc": "x"}, prop, BUILD)
    assert r["propensity"]["construct_match"] is True
    assert r["propensity"]["prop_construct_id"] == "Wild Type"
    assert r["propensity"]["prop_assay"] == "ThT"


def test_rollup_counts_construct_mismatch_suppressions():
    t = _triage(cs_id="CS1")
    t["condition_vector"]["construct_id"] = "A30P"
    mismatch = assemble_curve(t, _feat(), _regime("cooperative_sigmoidal"),
                              _gamma_significant(), None, {"best_by_aicc": "x"},
                              _prop(construct="Wild Type"), BUILD)
    matched = assemble_curve(_triage(cs_id="CS1"), _feat(),
                             _regime("cooperative_sigmoidal"), _gamma_significant(),
                             None, {"best_by_aicc": "x"}, _prop(construct="Wild Type"),
                             BUILD)
    roll = _rollup_of([mismatch, matched])
    js = roll["propensity_join_summary"]
    assert js["n_construct_mismatch_suppressed"] == 1
    assert js["n_propensity_attached"] == 1


# ============= join-policy-1.0 — the ASSAY gate ============================= #
# Before this gate, a uniprot-only fallback could attach a record computed for a
# DIFFERENT assay while `_propensity_block` checked only the construct — and the
# function's own comment claimed the triple key guaranteed assay equality, which
# was false on that path. Measured on the live corpus at the time: 9 series
# published `available: true, construct_match: true` while wearing another assay's
# propensity, and 7 carried a cross-assay sequence axis.

def test_propensity_assay_mismatch_not_attached():
    # the curve is a turbidity measurement; the only M6 record is ThT.
    t = _triage(cs_id="CS1")
    t["condition_vector"]["assay_type"] = "turbidity"
    r = assemble_curve(t, _feat(), _regime("cooperative_sigmoidal"),
                       _gamma_significant(), None, {"best_by_aicc": "x"},
                       _prop(assay="ThT"), BUILD)
    p = r["propensity"]
    assert p["available"] is False
    assert p["assay_match"] is False
    assert p["prop_assay"] == "ThT"
    # the ThT values must NOT have leaked into the block
    assert "intrinsic" not in p and "cohort" not in p and "surface" not in p
    # intrinsic/cohort are anchored in an assay-matched stratum, and the refusal
    # must say so rather than being a bare flag
    assert "assay" in p["reason"] and "turbidity" in p["reason"]


def test_assay_is_gated_before_construct_when_both_mismatch():
    # a record failing BOTH axes is reported as the ASSAY failure: assay-matching
    # is the single comparability guarantee an M6 record makes about itself, so it
    # is the stronger claim and the more informative refusal.
    t = _triage(cs_id="CS1")
    t["condition_vector"]["assay_type"] = "turbidity"
    t["condition_vector"]["construct_id"] = "A30P"
    r = assemble_curve(t, _feat(), _regime("cooperative_sigmoidal"),
                       _gamma_significant(), None, {"best_by_aicc": "x"},
                       _prop(construct="Wild Type", assay="ThT"), BUILD)
    p = r["propensity"]
    assert p["available"] is False
    assert p["assay_match"] is False and p["construct_match"] is False
    assert "assay" in p["reason"], p["reason"]
    # the join stays auditable on BOTH axes even when refused
    assert p["prop_assay"] == "ThT" and p["prop_construct_id"] == "Wild Type"


def test_assay_match_is_surfaced_on_every_path():
    # matched, construct-mismatched, assay-mismatched, and no-record-at-all.
    matched = assemble_curve(_triage(cs_id="CS1"), _feat(),
                             _regime("cooperative_sigmoidal"), _gamma_significant(),
                             None, {"best_by_aicc": "x"}, _prop(), BUILD)
    t_c = _triage(cs_id="CS1"); t_c["condition_vector"]["construct_id"] = "A30P"
    con = assemble_curve(t_c, _feat(), _regime("cooperative_sigmoidal"),
                         _gamma_significant(), None, {"best_by_aicc": "x"},
                         _prop(construct="Wild Type"), BUILD)
    t_a = _triage(cs_id="CS1"); t_a["condition_vector"]["assay_type"] = "turbidity"
    asy = assemble_curve(t_a, _feat(), _regime("cooperative_sigmoidal"),
                         _gamma_significant(), None, {"best_by_aicc": "x"},
                         _prop(assay="ThT"), BUILD)
    none = assemble_curve(_triage(cs_id="CS1"), _feat(),
                          _regime("cooperative_sigmoidal"), _gamma_significant(),
                          None, {"best_by_aicc": "x"}, None, BUILD)
    for r in (matched, con, asy, none):
        assert "assay_match" in r["propensity"], r["propensity"]
    assert matched["propensity"]["assay_match"] is True
    assert asy["propensity"]["assay_match"] is False


def test_sequence_axis_is_refused_across_the_assay_but_not_the_construct():
    # CONSTRUCT mismatch: the sequence axis is protein-level and still surfaces —
    # its predictor scores are computed over the protein's deposited peptides.
    t_c = _triage(cs_id="CS1"); t_c["condition_vector"]["construct_id"] = "A30P"
    con = assemble_curve(t_c, _feat(), _regime("cooperative_sigmoidal"),
                         _gamma_significant(), None, {"best_by_aicc": "x"},
                         _prop(construct="Wild Type"), BUILD)
    assert con["sequence_axis"]["available"] is True

    # ASSAY mismatch: refused. The block's licensing fields
    # (assay_endpoint_class, endpoint_match_to_assay, comparison_licensed) are pure
    # functions of the assay, so surfacing it would publish licensing computed
    # against the wrong endpoint.
    t_a = _triage(cs_id="CS1"); t_a["condition_vector"]["assay_type"] = "turbidity"
    asy = assemble_curve(t_a, _feat(), _regime("cooperative_sigmoidal"),
                         _gamma_significant(), None, {"best_by_aicc": "x"},
                         _prop(assay="ThT"), BUILD)
    sa = asy["sequence_axis"]
    assert sa["available"] is False
    assert sa["assay_match"] is False and sa["prop_assay"] == "ThT"
    assert "licens" in sa["reason"].lower(), sa["reason"]


def test_rollup_counts_assay_mismatch_in_its_own_class():
    t_a = _triage(cs_id="CS1"); t_a["condition_vector"]["assay_type"] = "turbidity"
    asy = assemble_curve(t_a, _feat(), _regime("cooperative_sigmoidal"),
                         _gamma_significant(), None, {"best_by_aicc": "x"},
                         _prop(assay="ThT"), BUILD)
    t_c = _triage(cs_id="CS1"); t_c["condition_vector"]["construct_id"] = "A30P"
    con = assemble_curve(t_c, _feat(), _regime("cooperative_sigmoidal"),
                         _gamma_significant(), None, {"best_by_aicc": "x"},
                         _prop(construct="Wild Type"), BUILD)
    matched = assemble_curve(_triage(cs_id="CS1"), _feat(),
                             _regime("cooperative_sigmoidal"), _gamma_significant(),
                             None, {"best_by_aicc": "x"}, _prop(), BUILD)
    js = _rollup_of([asy, con, matched])["propensity_join_summary"]
    # DISJOINT classes: an assay mismatch must not be miscounted as a construct one
    assert js["n_assay_mismatch_suppressed"] == 1
    assert js["n_construct_mismatch_suppressed"] == 1
    assert js["n_propensity_attached"] == 1


def test_join_policy_is_read_not_hardcoded_so_the_revert_path_is_real():
    """The governed record retains `previous` for every decision. That is only a
    revert path if the consumers READ the policy — if the new behaviour were
    inlined, `previous` would be a comment. Proven by flipping the policy and
    observing the pre-fix behaviour return."""
    import join_policy
    import m7_assemble as m7

    assert join_policy.applied_policy("PROPENSITY_ASSAY_GATE") == "construct_and_assay"
    assert join_policy.previous_policy("PROPENSITY_ASSAY_GATE") == "construct_only"

    t = _triage(cs_id="CS1")
    t["condition_vector"]["assay_type"] = "turbidity"
    args = (t, _feat(), _regime("cooperative_sigmoidal"), _gamma_significant(),
            None, {"best_by_aicc": "x"}, _prop(assay="ThT"), BUILD)
    assert assemble_curve(*args)["propensity"]["available"] is False

    real = m7._applied_policy
    try:
        m7._applied_policy = lambda name, default=None: (
            join_policy.previous_policy(name, default))
        reverted = assemble_curve(*args)
        # the pre-fix bleed returns EXACTLY: a cross-assay record attached and
        # presented as a clean construct match
        assert reverted["propensity"]["available"] is True
        assert reverted["propensity"]["construct_match"] is True
        assert reverted["propensity"]["prop_assay"] == "ThT"
        assert reverted["sequence_axis"]["available"] is True
    finally:
        m7._applied_policy = real
    # and the gate is back
    assert assemble_curve(*args)["propensity"]["available"] is False


# ============= NEW: item 5 — build-id sensitivity (dynamic, parametrized) === #
def test_build_id_changes_with_each_component():
    # parametrize over EVERY component axis; flipping any one must change the id.
    # Expected ids are computed DYNAMICALLY (no hardcoded literal): we mutate one
    # key of a baseline component dict and assert the hash differs.
    base = build_components(corpus_version="corpus-v1")
    base_id = engine_build_id(base)
    for key in base:
        mutated = dict(base)
        mutated[key] = str(base[key]) + "-CHANGED"
        assert engine_build_id(mutated) != base_id, f"id unchanged when {key} bumped"


def _corpus_version_from_digests(digests):
    """Mirror the driver's corpus_version: sorted-join the source shas, sha256[:12].
    Kept local so the test computes the expected value dynamically rather than
    hardcoding a literal build-id."""
    import hashlib
    src = "|".join(sorted(d for d in digests if d))
    return hashlib.sha256(src.encode()).hexdigest()[:12] if src else "unknown-corpus"


def test_build_id_changes_when_aprs_or_structures_digest_changes():
    # item 5: the corpus digest now folds aprs + structures (not just peptides +
    # kinetics). Changing the aprs sha OR the structures sha must flip corpus_version
    # and therefore the build-id — computed dynamically here, not asserted as a literal.
    kin, pep, aprs, struct = "KIN", "PEP", "APRS", "STRUCT"
    base_corpus = _corpus_version_from_digests([kin, pep, aprs, struct])
    aprs_changed = _corpus_version_from_digests([kin, pep, "APRS2", struct])
    struct_changed = _corpus_version_from_digests([kin, pep, aprs, "STRUCT2"])
    assert base_corpus != aprs_changed
    assert base_corpus != struct_changed
    base_id = engine_build_id(build_components(corpus_version=base_corpus))
    assert base_id != engine_build_id(build_components(corpus_version=aprs_changed))
    assert base_id != engine_build_id(build_components(corpus_version=struct_changed))


def test_corpus_digest_is_order_independent():
    # the driver SORTS the digests before hashing, so manifest source order can't
    # change the corpus_version (item 5).
    a = _corpus_version_from_digests(["KIN", "PEP", "APRS", "STRUCT"])
    b = _corpus_version_from_digests(["STRUCT", "APRS", "PEP", "KIN"])
    assert a == b


def test_build_id_changes_with_harness_version():
    a = engine_build_id(build_components(corpus_version="abc", harness_version="harness-1.0"))
    b = engine_build_id(build_components(corpus_version="abc", harness_version="harness-2.0"))
    assert a != b


# ============= NEW: comparable(None, x) + gamma_significant edge cases ====== #
def test_comparable_none_is_false():
    bid = engine_build_id(build_components(corpus_version="abc"))
    assert comparable(None, bid) is False
    assert comparable(bid, None) is False
    assert comparable(None, None) is False


def test_gamma_significant_edge_cases():
    assert gamma_significant(None) is False
    # reliable but no CI -> not significant
    assert gamma_significant({"gamma_regression": {"gamma_reliable": True}}) is False
    # reliable, CI straddles 0 -> not significant
    assert gamma_significant(
        {"gamma_regression": {"gamma_reliable": True, "gamma_ci": [-0.1, 1.2]}}) is False
    # reliable, CI excludes 0 (both positive) -> significant
    assert gamma_significant(
        {"gamma_regression": {"gamma_reliable": True, "gamma_ci": [0.3, 1.2]}}) is True
    # reliable, CI excludes 0 (both negative) -> significant
    assert gamma_significant(
        {"gamma_regression": {"gamma_reliable": True, "gamma_ci": [-1.2, -0.3]}}) is True
    # NOT reliable, even with a 0-excluding CI -> not significant
    assert gamma_significant(
        {"gamma_regression": {"gamma_reliable": False, "gamma_ci": [0.3, 1.2]}}) is False


# ============= NEW: item 11 — m9 ranked hooks =============================== #
def test_m9_hooks_ranked_present_and_capped():
    cv = {"agitation": None, "seeded": None,
          "field_provenance": {"agitation": "unknown", "seeded": "unknown"}}
    h = m9_hook(_m1(censoring="left"), _feat(t50_status="biased_early"),
                _regime(), None, cv, has_concentration_series=False)
    assert "m9_hooks_ranked" in h
    assert 1 <= len(h["m9_hooks_ranked"]) <= 3
    # the primary hook is the top of the ranked list (impact order preserved).
    assert h["m9_hooks_ranked"][0]["rule_fired"] == h["rule_fired"]
    assert h["rule_fired"] == "no_concentration_series"   # highest impact wins
    assert "not multi-objective" in h["stub"] or "NOT multi-objective" in h["stub"]


# --------------------------------- runner ---------------------------------- #
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
