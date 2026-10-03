"""
Tests for M6 — propensity (surface / intrinsic / cohort), sequence axis & structure.

    python engine/test_m6.py
    pytest engine/test_m6.py
"""
from __future__ import annotations

import json
import os
import tempfile

from m6_propensity import (
    MIN_ANCHOR_EFFECTIVE_N,
    MIN_N_FOR_95_INTERVAL,
    REFERENCE_SCALES,
    _assay_endpoint_class,
    _percentile_rank,
    cohort_rank,
    intrinsic_anchor,
    score_protein,
    sequence_axis,
    structure_linkage,
    surface,
)


def _sub(value, available=True):
    return {"value": value, "granularity": "residue", "endpoint": "x",
            "reduction": "max", "version": "v", "summary": None, "available": available}


# Waltz is COVERAGE: value=None, availability = "has a tested hexapeptide".
def _waltz(available=True):
    return {"value": None, "granularity": "hexapeptide", "endpoint": "amyloid (hexapeptide)",
            "reduction": "coverage", "version": "v", "summary": None,
            "n_waltz_amyloid_peptides": 2 if available else 0, "available": available}


SEQ = {
    "by_uniprot": {
        "P1": {
            "protein_name": "Test protein", "n_peptides": 5, "n_amyloid_peptides": 3,
            "sequence_predictors": {
                "tango": _sub(100.0), "aggrescan": _sub(50.0),
                "pasta": _sub(-10.0), "waltz": _waltz(True)},
            "aprs": [{"position": "1-5", "length": 5}],
            "structures": [{"pdb_id": "1XXX", "amyloid": "Amyloid",
                            "method": "X-ray", "resolution": 2.0},
                           {"pdb_id": "2YYY", "amyloid": "Non-amyloid",
                            "method": "NMR", "resolution": None}],
        }},
    "absent_predictors": {"zyggregator": "absent", "camsol": "absent (solubility)"},
}


# ------------------------------ intrinsic ---------------------------------- #
def test_intrinsic_corpus_anchor_when_enough():
    stratum = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]   # n=10 >= 8
    r = intrinsic_anchor("t50", 55.0, stratum)
    assert r["anchor_kind"] == "corpus"
    assert r["anchor_effective_n"] == 10
    assert 40 <= r["percentile_in_stratum"] <= 60          # ~middle


def test_intrinsic_reference_fallback_when_thin():
    r = intrinsic_anchor("t50", 100.0, [10, 20, 30])        # n=3 < 8
    assert r["anchor_kind"] == "reference_scale"
    assert 0.0 <= r["reference_position_0_1"] <= 1.0
    assert r["anchor_version"] == REFERENCE_SCALES["version"]


def test_intrinsic_reference_position_log_clamped():
    # t50 ref scale is log10 over [1, 1000]; a value above range clamps to 1.0
    r = intrinsic_anchor("gamma", 1.0, [0.1, 0.2])           # linear [0,2] -> 0.5
    assert abs(r["reference_position_0_1"] - 0.5) < 1e-9
    assert r["reference_position_clamped"] is False
    assert r["reference_position_out_of_range"] is None


def test_intrinsic_reference_position_out_of_range_flagged():
    # gamma value 3.0 is above the linear [0,2] scale -> clamps to 1.0, flagged "high"
    r = intrinsic_anchor("gamma", 3.0, [0.1, 0.2])
    assert r["reference_position_0_1"] == 1.0
    assert r["reference_position_clamped"] is True
    assert r["reference_position_out_of_range"] == "high"


def test_intrinsic_95_interval_null_when_n_too_small():
    # 10 <= n < 20: corpus-anchored, but the 95% interval is null + flagged
    stratum = list(range(1, 16))                             # n=15 (>=8, <20)
    r = intrinsic_anchor("t50", 8.0, stratum)
    assert r["anchor_kind"] == "corpus"
    assert r["stratum_reference_interval_95"] is None
    assert r["stratum_reference_interval_95_flag"] == "n_too_small_for_95_interval"


def test_intrinsic_95_interval_interpolated_when_n_large():
    # n >= 20: numpy.percentile interpolation -> not just the min/max
    stratum = list(range(1, 101))                            # n=100 (1..100)
    r = intrinsic_anchor("t50", 50.0, stratum)
    assert r["stratum_reference_interval_95_flag"] is None
    lo, hi = r["stratum_reference_interval_95"]
    # numpy.percentile([1..100],[2.5,97.5]) = [3.475, 97.525], NOT [1, 100]
    assert abs(lo - 3.475) < 1e-6 and abs(hi - 97.525) < 1e-6


def test_intrinsic_corpus_propensity_percentile_inverts_for_t50():
    # low t50 = HIGH propensity: the corpus-anchored intrinsic view reports a
    # direction + a sign-corrected propensity_percentile_in_stratum that is the
    # exact complement of the raw percentile_in_stratum (review: direction parity).
    stratum = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]      # n=10 >= 8 -> corpus
    r = intrinsic_anchor("t50", 25.0, stratum)
    assert r["anchor_kind"] == "corpus"
    assert r["direction"] == "low_is_high_propensity"
    assert (r["propensity_percentile_in_stratum"]
            == 100.0 - r["percentile_in_stratum"])


def test_intrinsic_corpus_propensity_percentile_passthrough_for_high_is_high():
    # gamma is high-is-high: propensity_percentile_in_stratum == percentile_in_stratum
    stratum = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]  # n=9 -> corpus
    r = intrinsic_anchor("gamma", 1.0, stratum)
    assert r["anchor_kind"] == "corpus"
    assert r["direction"] == "high_is_high_propensity"
    assert (r["propensity_percentile_in_stratum"]
            == r["percentile_in_stratum"])


def test_intrinsic_reference_propensity_position_inverts_for_t50():
    # reference-scale branch, low-is-high feature: propensity position = 1 - position
    r = intrinsic_anchor("t50", 100.0, [10, 20, 30])          # n=3 < 8 -> ref scale
    assert r["anchor_kind"] == "reference_scale"
    assert r["direction"] == "low_is_high_propensity"
    assert (abs(r["propensity_reference_position_0_1"]
                - (1.0 - r["reference_position_0_1"])) < 1e-9)


def test_intrinsic_reference_propensity_position_passthrough_for_high_is_high():
    # reference-scale branch, high-is-high feature: propensity position == position
    r = intrinsic_anchor("gamma", 1.0, [0.1, 0.2])            # n=2 < 8 -> ref scale
    assert r["anchor_kind"] == "reference_scale"
    assert r["direction"] == "high_is_high_propensity"
    assert (r["propensity_reference_position_0_1"]
            == r["reference_position_0_1"])


def test_intrinsic_cohort_share_one_direction():
    # consistency by construction: intrinsic and cohort report the SAME direction
    # for the SAME feature (they route through one shared direction helper).
    big = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for feat in ("t50", "lag_time", "lag_to_t50_ratio", "gamma",
                 "transition_sharpness"):
        intr = intrinsic_anchor(feat, 55.0, big)
        coh = cohort_rank(feat, 55.0, big)
        assert intr["direction"] == coh["direction"], feat


# ------------------------------- cohort ------------------------------------ #
def test_cohort_rank_and_percentile_excludes_self():
    # value 30 is IN the pool: self-exclusion -> others = [10,20,40,50], N=4.
    # Hazen percentile of 30 vs others = 100*(2 below + 0)/4 = 50.0; rank = 2+1 = 3.
    r = cohort_rank("t50", 30.0, [10, 20, 30, 40, 50])
    assert r["suppressed"] is False and r["N"] == 4 and r["N_excludes_self"] is True
    assert r["rank"] == 3 and abs(r["percentile"] - 50.0) < 1e-9


def test_cohort_suppressed_for_singleton():
    # singleton cohort -> after self-exclusion N=0 < MIN_COHORT_N
    r = cohort_rank("t50", 30.0, [30.0])
    assert r["suppressed"] is True and r["reason"] == "N too small" and r["N"] == 0


def test_cohort_bars_amplitude_feature():
    r = cohort_rank("plateau", 1.0, [0.5, 1.0, 1.5, 2.0])    # amplitude -> barred
    assert r["suppressed"] is True and "amplitude" in r["reason"]


def test_cohort_rank_percentile_reconcile_with_ties():
    # ties case: pool has repeated values; Hazen mid-rank splits ties evenly so
    # rank and percentile stay self-consistent. value 20 (one removed as self) ->
    # others = [10, 20, 20, 30, 40]; below=1, equal=2 -> pct = 100*(1+1)/5 = 40.0;
    # rank = #(<20)+1 = 1+1 = 2.
    r = cohort_rank("t50", 20.0, [10, 20, 20, 20, 30, 40])
    assert r["N"] == 5 and r["rank"] == 2 and abs(r["percentile"] - 40.0) < 1e-9


def test_cohort_propensity_percentile_inverts_for_t50():
    # low t50 = HIGH propensity: a fast (low-t50) protein gets a LOW feature
    # percentile but a HIGH propensity_percentile (sign-corrected).
    r = cohort_rank("t50", 5.0, [5.0, 10, 20, 30, 40, 50])   # self-excluded -> 5 others
    assert r["direction"] == "low_is_high_propensity"
    # 5 is below all 5 others -> feature percentile 0.0 -> propensity percentile 100.0
    assert abs(r["percentile"] - 0.0) < 1e-9
    assert abs(r["propensity_percentile"] - 100.0) < 1e-9


def test_cohort_propensity_percentile_passthrough_for_high_is_high():
    # gamma is NOT a low-is-high feature: propensity_percentile == feature percentile
    r = cohort_rank("gamma", 1.5, [0.5, 1.0, 1.5, 2.0, 2.5])
    assert r["direction"] == "high_is_high_propensity"
    assert r["propensity_percentile"] == r["percentile"]


def test_percentile_rank_hazen_midrank_symmetry():
    # a value equal to the whole pool sits at the mid-rank 50.0 (ties split evenly)
    assert abs(_percentile_rank(5.0, [5.0, 5.0, 5.0, 5.0]) - 50.0) < 1e-9
    # intrinsic and cohort share this convention, so they reconcile by construction
    assert abs(_percentile_rank(30.0, [10, 20, 40, 50]) - 50.0) < 1e-9


# --------------------------- sequence axis --------------------------------- #
def test_sequence_axis_endpoint_match_tht():
    """ThT is amyloid-specific: only the AMYLOID-SPECIFIC predictors (PASTA, Waltz)
    are licensed. TANGO is generic β-aggregation and AGGRESCAN is generic
    aggregation — both endpoint-mismatched vs an amyloid dye (review item A3)."""
    sa = sequence_axis("P1", SEQ, "ThT")
    subs = sa["sub_axes"]
    assert sa["assay_endpoint_class"] == "amyloid"
    assert subs["pasta"]["comparison_licensed"] is True
    assert subs["waltz"]["comparison_licensed"] is True
    assert subs["tango"]["comparison_licensed"] is False        # generic vs amyloid (A3)
    assert subs["aggrescan"]["comparison_licensed"] is False    # generic vs amyloid


def test_sequence_axis_congo_red_is_amyloid():
    """Congo Red is amyloid-specific -> PASTA & Waltz licensed (review item A1)."""
    for assay in ("Congo Red", "CongoRed", "congo red staining"):
        sa = sequence_axis("P1", SEQ, assay)
        subs = sa["sub_axes"]
        assert sa["assay_endpoint_class"] == "amyloid", assay
        assert subs["pasta"]["comparison_licensed"] is True
        assert subs["waltz"]["comparison_licensed"] is True
        assert subs["tango"]["comparison_licensed"] is False


def test_sequence_axis_cytofluor_and_ths_are_amyloid():
    for assay in ("Cytofluor", "ThS", "Thioflavin S"):
        sa = sequence_axis("P1", SEQ, assay)
        assert sa["assay_endpoint_class"] == "amyloid", assay
        assert sa["sub_axes"]["pasta"]["comparison_licensed"] is True


def test_sequence_axis_tango_generic_barred_vs_tht():
    """TANGO demoted to generic: explicitly barred vs a ThT (amyloid) assay."""
    sa = sequence_axis("P1", SEQ, "ThT")
    assert sa["sub_axes"]["tango"]["endpoint_class"] == "generic"
    assert sa["sub_axes"]["tango"]["endpoint_match_to_assay"] is False
    assert sa["sub_axes"]["tango"]["comparison_licensed"] is False


def test_sequence_axis_endpoint_match_turbidity():
    sa = sequence_axis("P1", SEQ, "Turbidity")
    subs = sa["sub_axes"]
    assert sa["assay_endpoint_class"] == "generic"
    # generic assay licenses the generic predictors (TANGO, AGGRESCAN)
    assert subs["aggrescan"]["comparison_licensed"] is True
    assert subs["tango"]["comparison_licensed"] is True
    assert subs["pasta"]["comparison_licensed"] is False        # amyloid vs generic
    assert subs["waltz"]["comparison_licensed"] is False


def test_sequence_axis_unknown_assay_licenses_nothing():
    """Unknown/None/empty assay -> 'unknown' class that licenses NOTHING (item A2)."""
    for assay in (None, "", "   ", "some_unrecognised_readout"):
        sa = sequence_axis("P1", SEQ, assay)
        assert sa["assay_endpoint_class"] == "unknown", repr(assay)
        assert sa["no_comparison_licensed_reason"] is not None
        assert all(s["comparison_licensed"] is False for s in sa["sub_axes"].values())


def test_assay_endpoint_class_unknown_not_generic():
    # the regression we are guarding: unknown must NOT silently become "generic"
    assert _assay_endpoint_class(None) == "unknown"
    assert _assay_endpoint_class("") == "unknown"
    assert _assay_endpoint_class("turbidity") == "generic"
    assert _assay_endpoint_class("fluorescence") == "generic"   # generic fluorescence
    assert _assay_endpoint_class("ThT") == "amyloid"
    # ...and an unrecognised NON-EMPTY assay still refuses rather than guessing
    assert _assay_endpoint_class("nonsense-assay") == "unknown"


def test_every_assay_in_the_corpus_vocabulary_is_classified():
    """The whole point of `unknown` is that it is a REFUSAL, not a resting place.
    An assay the corpus actually contains that classifies `unknown` is a gap in the
    vocabulary, not a conservative answer — it silently bars every sequence
    predictor. This pins the full corpus vocabulary so a new assay cannot arrive
    and quietly license nothing."""
    corpus_vocabulary = {
        "ThT": "amyloid",        # thioflavin T   — cross-β dye
        "ThS": "amyloid",        # thioflavin S   — cross-β dye
        "CongoRed": "amyloid",   # Congo red      — cross-β dye
        "k114": "amyloid",       # Congo-red/X-34 derived fluorophore, see below
        "turbidity": "generic",  # light scattering — any aggregate, not cross-β
    }
    for assay, expected in corpus_vocabulary.items():
        assert _assay_endpoint_class(assay) == expected, assay
        assert _assay_endpoint_class(assay.lower()) == expected, assay


def test_k114_is_an_amyloid_endpoint_and_the_classification_is_governed():
    """K114 (Crystal et al., J Neurochem 2003;86:1359-1368) is a Congo-red/X-34
    derived fluorophore introduced to QUANTIFY AMYLOID FIBRILLOGENESIS — it binds
    the cross-β fibril and fluoresces on binding, the same detection principle and
    target as ThT/ThS/Congo Red.

    join-policy-1.0 deliberately left it `unknown` (licensing nothing) because the
    call needed a citation it did not yet have. That was honest but WRONG on the
    science: it barred PASTA and Waltz — the cross-β-specific predictors — from a
    cross-β-specific assay, which is exactly the endpoint match the licensing rule
    exists to permit. join-policy-1.1 corrects it, and the value is READ from the
    governed record so the classification is reversible and carries its citation."""
    import join_policy
    import m6_propensity

    assert _assay_endpoint_class("k114") == "amyloid"
    assert join_policy.applied_policy("ASSAY_ENDPOINT_AMYLOID_ADDITIONS") == ["k114"]
    assert join_policy.previous_policy("ASSAY_ENDPOINT_AMYLOID_ADDITIONS") == []
    rec = join_policy.policy_record("ASSAY_ENDPOINT_AMYLOID_ADDITIONS")
    # a vocabulary claim about chemistry must ship its source, not just its verdict
    assert "Crystal" in rec["evidence"]["citation"]
    assert "J Neurochem" in rec["evidence"]["citation"]

    # licensing follows the class: cross-β predictors in, generic ones out
    sa = sequence_axis("P1", SEQ, "k114")
    assert sa["assay_endpoint_class"] == "amyloid"
    licensed = {n for n, ax in sa["sub_axes"].items() if ax["comparison_licensed"]}
    assert licensed == {"pasta", "waltz"}, licensed

    # and it is the SAME licensing an already-amyloid dye receives
    tht = sequence_axis("P1", SEQ, "ThT")
    assert {n: ax["comparison_licensed"] for n, ax in sa["sub_axes"].items()} == \
           {n: ax["comparison_licensed"] for n, ax in tht["sub_axes"].items()}


def test_sequence_axis_declares_absent_predictors():
    sa = sequence_axis("P1", SEQ, "ThT")
    assert set(sa["absent_predictors"]) == {"zyggregator", "camsol"}


def test_sequence_axis_unavailable_for_unknown_uniprot():
    sa = sequence_axis("NOPE", SEQ, "ThT")
    assert sa["available"] is False


# -------------------------- structure linkage ------------------------------ #
def test_structure_linkage_counts_and_caveats():
    sl = structure_linkage("P1", SEQ)
    assert sl["available"] is True
    assert sl["n_structures"] == 2 and sl["n_amyloid_structures"] == 1
    assert sl["n_aprs"] == 1
    assert sl["relationship"] == "associative_only"
    assert any("polymorphism" in c for c in sl["caveats"])


# ------------------------------- surface ----------------------------------- #
def test_surface_unavailable_without_gamma():
    assert surface(None)["available"] is False


def test_surface_from_gamma_record():
    g = {"gamma_regression": {"gamma": 1.0, "gamma_ci": [0.9, 1.1], "gamma_reliable": True},
         "gamma_global": {"gamma": 1.05, "gamma_ci": [0.95, 1.15]},
         "disagreement": {"disagree": False}}
    s = surface(g)
    assert s["available"] is True
    assert s["gamma_regression"]["value"] == 1.0 and s["gamma_regression"]["reliable"] is True
    assert s["gamma_disagreement"] is False
    assert "window_of_validity" not in s                       # not present upstream


def test_surface_passes_through_window_of_validity():
    # when the γ record carries window_of_validity, surface() surfaces it (item 12)
    g = {"gamma_regression": {"gamma": 1.0}, "gamma_global": {"gamma": 1.0},
         "disagreement": {"disagree": False},
         "window_of_validity": {"conc_uM": [5.0, 50.0]}}
    s = surface(g)
    assert s["window_of_validity"] == {"conc_uM": [5.0, 50.0]}


# ------------------------------ assembler ---------------------------------- #
def test_score_protein_assembles_all_views():
    stratum_excl = [10, 20, 30, 40, 50, 60, 70, 80, 90]      # n=9 -> corpus
    rec = score_protein("P1", "Test protein", "ThT", 45.0, stratum_excl,
                        stratum_excl + [45.0], None, SEQ, feature="t50")
    p = rec["propensity"]
    assert p["intrinsic"]["anchor_kind"] == "corpus"
    assert p["cohort"]["suppressed"] is False
    assert p["surface"]["available"] is False                 # no gamma record
    assert rec["sequence_axis"]["available"] is True
    assert rec["structure_linkage"]["available"] is True
    assert rec["feature_portable"] is True
    assert rec["condition_match"] == "assay_only"             # assay-only anchor (item 10)
    assert rec["feature_status"] == "point"


def test_score_protein_censored_lower_bound_suppresses_pools():
    """A right-censored (lower_bound) t50 is a ≥ inequality, not a rankable point:
    the record is still emitted but intrinsic AND cohort are suppressed (item C9)."""
    big_pool = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    rec = score_protein("P1", "Test protein", "ThT", None, big_pool, big_pool,
                        None, SEQ, feature="t50", feature_status="lower_bound")
    p = rec["propensity"]
    assert rec["feature_status"] == "lower_bound"
    assert p["intrinsic"]["suppressed"] is True
    assert "right-censored" in p["intrinsic"]["reason"]
    assert p["cohort"]["suppressed"] is True
    # ... but the non-rankable views are still present
    assert rec["sequence_axis"]["available"] is True


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


# ============= join-policy-1.0 — the ASSAY axis in the grouping key ========= #
# The key was (uniprot, construct), so a protein x construct measured by two
# assays was pooled into ONE median and labelled first-wins. Measured on the live
# corpus: 4 of 133 groups pooled across assays, the worst being P01160 Wild Type,
# published as assay "CongoRed" carrying 44.851 h -- the ThT median -- while its
# CongoRed data sat at 586.870 h, 13x away. A wrong NUMBER in a wrong stratum
# under a wrong label.

def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _run_m6_on(tmp, curves):
    """Run the real M6 CLI over a synthetic corpus.

    `curves` is a list of (series_id, assay, t50) or, for the leakage tests,
    (series_id, assay, t50, construct, pmid)."""
    import m6_propensity
    tri, feats = [], []
    for row in curves:
        if len(row) == 3:
            sid, assay, t50 = row
            construct, pmid = "Wild Type", None
        else:
            sid, assay, t50, construct, pmid = row
        rec = {"series_id": sid, "uniprot_id": "P1", "protein_id": "Prot1",
               "condition_vector": {"assay_type": assay,
                                    "construct_id": construct}}
        if pmid:
            rec["source_study"] = {"pmid": pmid}
        tri.append(rec)
        feats.append({"series_id": sid, "status": "ok", "t50_status": "point",
                      "features": {"t50": t50}})
    t = os.path.join(tmp, "curves_triaged.jsonl")
    f = os.path.join(tmp, "features.jsonl")
    g = os.path.join(tmp, "gamma.jsonl")
    s = os.path.join(tmp, "sequence.json")
    o = os.path.join(tmp, "propensity.jsonl")
    _write_jsonl(t, tri)
    _write_jsonl(f, feats)
    _write_jsonl(g, [])
    with open(s, "w", encoding="utf-8") as fh:
        json.dump({"by_uniprot": {}, "absent_predictors": {}}, fh)
    m6_propensity.main(["--triaged", t, "--features", f, "--gamma", g,
                        "--sequence", s, "--output", o])
    out = []
    with open(o, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def test_group_key_splits_on_assay_and_never_pools_a_median_across_them():
    with tempfile.TemporaryDirectory() as tmp:
        recs = _run_m6_on(tmp, [
            ("s1", "ThT", 10.0), ("s2", "ThT", 20.0),      # ThT median 15
            ("s3", "CongoRed", 500.0), ("s4", "CongoRed", 600.0),   # CR median 550
        ])
    by_assay = {r["assay"]: r for r in recs}
    # TWO records, not one pooled record wearing one arbitrary label
    assert len(recs) == 2, [r["assay"] for r in recs]
    assert set(by_assay) == {"ThT", "CongoRed"}
    assert by_assay["ThT"]["propensity"]["intrinsic"]["value"] == 15.0
    assert by_assay["CongoRed"]["propensity"]["intrinsic"]["value"] == 550.0
    # the pre-fix behaviour would have emitted ONE record with the pooled median
    # of all four (260.0) under whichever assay was seen first
    assert all(r["propensity"]["intrinsic"]["value"] != 260.0 for r in recs)


def test_group_key_is_read_from_the_policy_so_the_revert_is_real():
    """Flipping the governed key back must restore the pooled-median defect
    exactly -- otherwise the retained `previous` is decoration, not a revert."""
    import join_policy
    import m6_propensity

    assert join_policy.applied_policy("M6_GROUP_KEY") == [
        "uniprot", "construct", "assay"]
    assert join_policy.previous_policy("M6_GROUP_KEY") == ["uniprot", "construct"]

    curves = [("s1", "ThT", 10.0), ("s2", "ThT", 20.0),
              ("s3", "CongoRed", 500.0), ("s4", "CongoRed", 600.0)]
    real = m6_propensity._applied_policy
    try:
        m6_propensity._applied_policy = lambda name, default=None: (
            join_policy.previous_policy(name, default))
        with tempfile.TemporaryDirectory() as tmp:
            recs = _run_m6_on(tmp, curves)
        # the defect returns EXACTLY: one record, median pooled across assays,
        # labelled with the first assay seen
        assert len(recs) == 1, [r["assay"] for r in recs]
        assert recs[0]["propensity"]["intrinsic"]["value"] == 260.0
    finally:
        m6_propensity._applied_policy = real
    with tempfile.TemporaryDirectory() as tmp:
        assert len(_run_m6_on(tmp, curves)) == 2


def test_anchor_pool_is_leave_one_STUDY_out_not_merely_leave_one_group_out():
    """A corpus anchor is only leakage-free if the pool is independent of the
    value being ranked against it. Excluding just the group itself left the pool
    full of OTHER groups from the SAME study, sharing lab, protocol and batch --
    measured on the live corpus at 151 of 164 groups (92.1%).

    Constructed so the two policies give visibly different answers: 10 groups from
    one study plus 2 from another. Under leave-one-group-out the study-A groups see
    a pool of 11 and corpus-anchor; under leave-one-study-out they see only the 2
    study-B groups, fall below MIN_ANCHOR_EFFECTIVE_N and must say so."""
    import join_policy
    import m6_propensity

    curves = []
    for i in range(10):                       # study A: 10 distinct constructs
        curves.append((f"a{i}", "ThT", 10.0 + i, f"CA{i}", "PMID:AAA"))
    for i in range(2):                        # study B: 2 distinct constructs
        curves.append((f"b{i}", "ThT", 50.0 + i, f"CB{i}", "PMID:BBB"))

    with tempfile.TemporaryDirectory() as tmp:
        recs = _run_m6_on(tmp, curves)
    kinds = {}
    for r in recs:
        cid = (r.get("condition_vector") or {}).get("construct_id")
        kinds[cid] = ((r.get("propensity") or {}).get("intrinsic") or {}).get(
            "anchor_kind")
    assert len(recs) == 12, len(recs)
    # every study-A group is left with only the 2 study-B comparators -> too thin
    for i in range(10):
        assert kinds[f"CA{i}"] == "reference_scale", (i, kinds)

    # and the retained previous behaviour reproduces the leak exactly
    real = m6_propensity._applied_policy
    try:
        m6_propensity._applied_policy = lambda name, default=None: (
            join_policy.previous_policy(name, default)
            if name == "M6_ANCHOR_LEAKAGE" else real(name, default))
        with tempfile.TemporaryDirectory() as tmp:
            leaky = _run_m6_on(tmp, curves)
    finally:
        m6_propensity._applied_policy = real
    leaky_kinds = {(r.get("condition_vector") or {}).get("construct_id"):
                   ((r.get("propensity") or {}).get("intrinsic") or {}).get("anchor_kind")
                   for r in leaky}
    assert leaky_kinds["CA0"] == "corpus", leaky_kinds
    assert join_policy.applied_policy("M6_ANCHOR_LEAKAGE") == "leave_one_study_out"


def test_the_shipped_corpus_shows_the_leakage_correction():
    """The ThS stratum is essentially one study, so under leave-one-study-out none
    of its groups can be corpus-anchored -- they must fall back rather than report
    a percentile against their own study."""
    import json
    from pathlib import Path
    art = Path(__file__).resolve().parent.parent / "data/processed/propensity.jsonl"
    if not art.exists():
        return
    kinds = {}
    for line in art.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        intr = (p.get("propensity") or {}).get("intrinsic") or {}
        if intr.get("value") is None:
            continue
        kinds.setdefault(p.get("assay"), []).append(intr.get("anchor_kind"))
    assert kinds.get("ThS"), "no ThS groups scored"
    assert set(kinds["ThS"]) == {"reference_scale"}, kinds["ThS"]
    # ThT is genuinely multi-study and must KEEP its corpus anchor -- otherwise the
    # rule is just destroying information rather than removing leakage
    assert "corpus" in set(kinds.get("ThT", [])), kinds.get("ThT")


if __name__ == "__main__":
    raise SystemExit(_run())
