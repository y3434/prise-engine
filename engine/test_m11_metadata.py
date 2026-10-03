"""
Tests for M11 — Metadata Quality & Information Completeness Engine.

Deterministic + fast. Operates on SYNTHETIC series (does NOT read the real data
files), so it is hermetic and pins the contract:

  - full metadata -> high metadata_score;
  - missing agitation -> MECHANISM blocked, Δ-power HIGH, mechanistic_completeness
    drops, consequence chain correct;
  - missing buffer / reducing_agent -> negligible impact, score barely moves;
  - configurable weights are honored (an override changes the score);
  - the dependency graph blocks the right target per missing prerequisite;
  - not_captured_by_source is EXCLUDED from the denominator;
  - never raises on empty / malformed input.

    python engine/test_m11_metadata.py
    pytest engine/test_m11_metadata.py
"""
from __future__ import annotations

import m11_metadata as m11


# --------------------------------------------------------------------------- #
# synthetic series builders (mirror the REAL condition_vector schema)
# --------------------------------------------------------------------------- #
def _full_series(*, agitation="known", seeded="known", assay_mass=True,
                 cs_id="CS1", cs_size=3, buffer="phosphate 50mM",
                 additives="NaN3 0.02%", regime="cooperative_sigmoidal"):
    """A maximally-complete series. Provenance 'known' everywhere it matters."""
    fp = {"concentration": "known", "pH": "known", "temperature": "known",
          "ionic_strength": "known", "assay_type": "known",
          "agitation": agitation, "seeded": seeded, "tht_concentration": "known"}
    s = {
        "series_id": "SYN-1",
        "protein_id": "Amyloid Beta-ABeta42",
        "uniprot_id": "P05067",
        "concentration_series_id": cs_id,
        "digitization_uncertainty": True,
        "source_study": {"pmid": "12345678", "reference": "J Test 2026"},
        "condition_vector": {
            "concentration": {"value_uM": 20.0, "unit": "microM"},
            "temperature_C": 37.0, "pH": 7.4,
            "ion": "NaCl", "ion_concentration": "100 mM",
            "ionic_strength_note": "NaCl 100 mM",
            "buffer": buffer, "additives": additives,
            "agitation": ("shaken" if agitation == "known" else None),
            "seeded": (False if seeded == "known" else None),
            "assay_type": "ThT", "assay_reports_mass": assay_mass,
            "tht_concentration": 10.0, "construct_id": "Wild Type",
            "pdb_id": "1IYT",
            "field_provenance": fp,
        },
        "_regime": regime,
    }
    # attach a corpus index so concentration_series/replicate_count score
    s["_m11_index"] = {"series_sizes": {cs_id: cs_size} if cs_id else {},
                       "replicate_counts": {"SYN-1": 3}}
    return s


def _regime_of(s):
    return s.get("_regime")


# --------------------------------------------------------------------------- #
# 1. full metadata -> high score
# --------------------------------------------------------------------------- #
def test_full_metadata_high_score():
    s = _full_series()
    comp = m11.score_completeness(s)
    assert comp["metadata_score"] >= 0.9, comp["metadata_score"]
    # denominator excludes not_captured_by_source fields
    assert comp["not_captured_by_source"] == [] or all(
        f in m11.METADATA_ONTOLOGY["fields"] for f in comp["not_captured_by_source"])
    # every mechanism prereq satisfied -> mechanistic_completeness == 1.0
    tgt = m11.evaluate_targets(s, features_regime=_regime_of(s))
    assert tgt["mechanistic_completeness"] == 1.0, tgt["mechanistic_completeness"]
    assert tgt["dependency_evaluation"]["MECHANISM"]["satisfied"] is True


# --------------------------------------------------------------------------- #
# 2. missing agitation -> MECHANISM blocked, Δ HIGH, mech_completeness drops
# --------------------------------------------------------------------------- #
def test_missing_agitation_blocks_mechanism_high():
    full = _full_series()
    miss = _full_series(agitation="unknown")

    tgt_full = m11.evaluate_targets(full, features_regime=_regime_of(full))
    tgt_miss = m11.evaluate_targets(miss, features_regime=_regime_of(miss))

    # mechanism now blocked, and specifically ON agitation
    assert tgt_miss["dependency_evaluation"]["MECHANISM"]["satisfied"] is False
    assert "agitation" in tgt_miss["mechanistic_blockers"]
    # mechanistic_completeness strictly drops
    assert tgt_miss["mechanistic_completeness"] < tgt_full["mechanistic_completeness"]

    # Δ-power flags agitation as HIGH with the correct consequence chain
    dp = m11.delta_inferential_power(miss, features_regime=_regime_of(miss))
    agit = [it for it in dp["delta_inferential_power"] if it["field"] == "agitation"]
    assert agit, "agitation not surfaced in delta power"
    assert agit[0]["impact"] == "high"
    assert "mechanism" in agit[0]["blocks"]
    assert "quiescent-vs-shaken" in agit[0]["consequence_chain"]
    assert "HIGH" in agit[0]["consequence_chain"]


# --------------------------------------------------------------------------- #
# 3. missing buffer / reducing_agent -> negligible, score barely moves
# --------------------------------------------------------------------------- #
def test_missing_buffer_negligible():
    full = _full_series()
    no_buffer = _full_series(buffer=None, additives=None)  # no reducing token either

    s_full = m11.score_completeness(full)["metadata_score"]
    s_nob = m11.score_completeness(no_buffer)["metadata_score"]
    # buffer weight is LOW -> score barely moves (well under 5 points)
    assert abs(s_full - s_nob) < 0.05, (s_full, s_nob)

    dp = m11.delta_inferential_power(no_buffer, features_regime=_regime_of(no_buffer))
    buf = [it for it in dp["delta_inferential_power"] if it["field"] == "buffer"]
    assert buf and buf[0]["impact"] == "negligible"
    # mechanism must NOT be affected by a missing buffer
    tgt = m11.evaluate_targets(no_buffer, features_regime=_regime_of(no_buffer))
    assert tgt["mechanistic_completeness"] == 1.0


def test_agitation_impact_beats_buffer_impact():
    """The core honesty claim: missing agitation is HIGH, missing buffer NEGLIGIBLE."""
    miss_ag = _full_series(agitation="unknown")
    dp = m11.delta_inferential_power(miss_ag, features_regime=_regime_of(miss_ag))
    ranked = dp["delta_inferential_power"]
    # agitation ranks above any negligible field
    agit_i = next(i for i, it in enumerate(ranked) if it["field"] == "agitation")
    for i, it in enumerate(ranked):
        if it["impact"] == "negligible":
            assert agit_i < i, "a negligible field outranked agitation"


# --------------------------------------------------------------------------- #
# 4. configurable weights honored
# --------------------------------------------------------------------------- #
def test_weights_configurable_honored():
    s = _full_series(agitation="unknown")  # agitation missing -> its weight matters
    base = m11.score_completeness(s)["metadata_score"]
    # crank agitation's completeness weight WAY up -> missing it hurts more -> lower
    heavy = m11.score_completeness(s, weights={"agitation": 100.0})["metadata_score"]
    assert heavy < base, (heavy, base)
    # zero it out -> missing it costs nothing -> higher
    light = m11.score_completeness(s, weights={"agitation": 0.0})["metadata_score"]
    assert light > base, (light, base)
    assert m11.score_completeness(s, weights={"agitation": 0.0})["weights_source"] == "overridden"


# --------------------------------------------------------------------------- #
# 5. dependency graph blocks the right target per missing prereq
# --------------------------------------------------------------------------- #
def test_dependency_graph_targeted_blocking():
    # missing pH -> COMPARABILITY blocked, MECHANISM still fine on its own prereqs
    miss_ph = _full_series()
    miss_ph["condition_vector"]["pH"] = None
    miss_ph["condition_vector"]["field_provenance"]["pH"] = "unknown"
    tgt = m11.evaluate_targets(miss_ph, features_regime=_regime_of(miss_ph))
    assert tgt["dependency_evaluation"]["COMPARABILITY"]["satisfied"] is False
    assert "pH" in tgt["dependency_evaluation"]["COMPARABILITY"]["blocking_prerequisites"]

    # non-mass assay -> SCALING and MECHANISM both blocked on assay
    non_mass = _full_series(assay_mass=False)
    tgt2 = m11.evaluate_targets(non_mass, features_regime=_regime_of(non_mass))
    assert tgt2["dependency_evaluation"]["SCALING"]["satisfied"] is False
    assert "assay" in tgt2["dependency_evaluation"]["SCALING"]["blocking_prerequisites"]
    assert "assay" in tgt2["dependency_evaluation"]["MECHANISM"]["blocking_prerequisites"]

    # no concentration series -> SCALING blocked on the series prereq
    no_series = _full_series(cs_id=None, cs_size=0)
    tgt3 = m11.evaluate_targets(no_series, features_regime=_regime_of(no_series))
    assert "concentration_series" in \
        tgt3["dependency_evaluation"]["SCALING"]["blocking_prerequisites"]


def test_shape_prereq_needs_curve_when_no_regime():
    """Without an M5 regime, cooperative_shape is unknown_needs_curve, NOT faked."""
    s = _full_series()
    tgt = m11.evaluate_targets(s, features_regime=None)
    shp = tgt["dependency_evaluation"]["MECHANISM"]["prerequisite_status"]["cooperative_shape"]
    assert shp["satisfied"] is False
    assert "unknown_needs_curve" in shp["note"]


# --------------------------------------------------------------------------- #
# 6. not_captured_by_source excluded from the denominator
# --------------------------------------------------------------------------- #
def test_not_captured_excluded_from_denominator():
    # a series with NO reducing agent and NO uniprot -> reducing_agent + organism
    # are not_captured_by_source; they must NOT appear in the denominator.
    s = _full_series(additives="NaN3 0.02%")  # no DTT/TCEP token
    s["uniprot_id"] = None
    comp = m11.score_completeness(s)
    assert "reducing_agent" in comp["not_captured_by_source"]
    assert "organism" in comp["not_captured_by_source"]
    # those fields report present=None and are flagged, but do not lower the score
    assert comp["per_field"]["reducing_agent"]["present"] is None
    # denominator == sum of weights over CAPTURABLE fields only
    cap_weight = sum(f["weight"] for f in comp["per_field"].values()
                     if f["present"] is not None)
    assert abs(comp["denominator"] - cap_weight) < 1e-9


def test_service_c_measured_gain_labelled():
    """The concentration-series gain must be labelled service_c_measured, with the
    1->4 Tier-B split, when a mass assay is present."""
    s = _full_series(cs_id=None, cs_size=0, assay_mass=True)  # series missing
    dp = m11.delta_inferential_power(s, features_regime=_regime_of(s))
    cs = [it for it in dp["delta_inferential_power"]
          if it["field"] == "concentration_series"]
    assert cs, "concentration_series missing not surfaced"
    assert cs[0]["gain_basis"] == "service_c_measured"
    ig = cs[0]["information_gain"]
    assert ig["tier_b_classes_before"] == 1 and ig["tier_b_classes_after"] == 4


# --------------------------------------------------------------------------- #
# 7. uncertainty is separate + honest
# --------------------------------------------------------------------------- #
def test_uncertainty_separate_from_completeness():
    s = _full_series()
    u = m11.metadata_uncertainty(s)
    assert 0.0 <= u["metadata_uncertainty"] <= 1.0
    # even a complete record carries digitization baseline > 0
    assert u["components"]["digitization_baseline"] > 0.0
    assert "completeness ≠ correctness" in u["caveat"]


# --------------------------------------------------------------------------- #
# 8. recommendations rank by gain/effort in M9 language
# --------------------------------------------------------------------------- #
def test_recommendations_ranked_action_language():
    s = _full_series(agitation="unknown", seeded="unknown")
    recs = m11.recommended_missing_metadata(s, features_regime=_regime_of(s))
    rlist = recs["recommended_missing_metadata"]
    assert rlist, "no recommendations produced"
    # the cheap high-impact agitation/seeding action should top the ranking
    assert rlist[0]["action"] in ("record_agitation_and_seeding",)
    # gain/effort is monotone non-increasing by rank
    gains = [r["gain_over_effort"] for r in rlist]
    assert gains == sorted(gains, reverse=True)


# --------------------------------------------------------------------------- #
# 9. NEVER raises on empty / malformed
# --------------------------------------------------------------------------- #
def test_never_raises_on_malformed():
    for bad in [None, {}, [], "nonsense", 42, {"condition_vector": None},
                {"condition_vector": {"field_provenance": None}}]:
        r = m11.assess_metadata(bad)
        assert isinstance(r, dict)
        assert "version" in r
    # score/uncertainty/targets on junk all return dicts, no raise
    assert isinstance(m11.score_completeness(None), dict)
    assert isinstance(m11.metadata_uncertainty("x"), dict)
    assert isinstance(m11.evaluate_targets(None), dict)
    assert isinstance(m11.delta_inferential_power(123), dict)


def test_assess_full_record_shape():
    s = _full_series()
    r = m11.assess_metadata(s, features_regime=_regime_of(s))
    for key in ("metadata_completeness", "metadata_uncertainty",
                "mechanistic_completeness", "dependency_evaluation",
                "delta_inferential_power", "recommended_missing_metadata",
                "not_captured_by_source", "honesty"):
        assert key in r, f"missing {key}"
    assert r["version"] == m11.M11_VERSION


def test_ontology_field_count_and_categories():
    fields = m11.METADATA_ONTOLOGY["fields"]
    assert len(fields) == 19, len(fields)
    cats = {f["category"] for f in fields.values()}
    assert cats <= {"identity", "condition", "assay", "statistical",
                    "provenance", "bibliographic"}
    # mechanism-gating fields are weighted HIGH vs buffer/reducing_agent LOW
    for hi in ("assay", "agitation", "seeding", "concentration_series"):
        assert fields[hi]["completeness_weight"] >= 4.0
    for lo in ("buffer", "reducing_agent", "organism", "publication"):
        assert fields[lo]["completeness_weight"] <= 0.5


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
    print(f"test_m11_metadata: {passed} passed")
    return passed


if __name__ == "__main__":
    _run_all()
