"""
PRISE — Join Policy (the GOVERNED apply artifact for the M6/M7 assay axis)
==========================================================================

This module is the **versioned, signed, dated** record of the M6 grouping key and
the M7 propensity/sequence-axis join being corrected onto the ASSAY axis — the
same governance form `calibrated_thresholds.py` uses, and for the same reason:
the change moves published numbers, so it must invalidate descendants.

It carries, for EACH consequential join decision:
  * `previous`  — the behaviour that shipped before this record,
  * `applied`   — the behaviour now in force,
  * `evidence`  — the measurement that justified the change,
so nothing is lost: the prior behaviour is retained and auditable, and the change
is reversible (restore `applied` → `previous` and re-bump the version).

`JOIN_POLICY_VERSION` ("join-policy-1.0") is folded into `engine_build_id`
(m7_assemble.build_components). This is **load-bearing, not decorative**: every
one of the 14 pre-existing build components is a version string or an ETL byte
digest, so NONE of them is a function of this code. Without this axis the fix
would ship the SAME `engine_build_id` with DIFFERENT numbers, and `comparable()`
(which gates on id equality alone) would declare two materially different corpora
comparable. That is precisely the silent comparability violation §7 forbids.

CONSUMERS MUST READ THE POLICY, NOT HARD-CODE THE NEW BEHAVIOUR. `m6_propensity`
builds its group key from `applied_policy("M6_GROUP_KEY")` and `m7_assemble`
branches on `applied_policy("PROPENSITY_ASSAY_GATE")` /
`applied_policy("SEQUENCE_AXIS_ASSAY_GATE")`. If the new behaviour were inlined,
`previous` would be a comment rather than a revert path — the one way this
precedent can be copied badly. Nothing here raises: a lookup miss returns the
supplied default.

WHAT WAS WRONG. `m6_propensity` grouped curves on `(uniprot, construct)` with the
record's `assay` label assigned first-wins, so a protein×construct measured by two
assays was **pooled into one median and labelled with one arbitrarily-chosen
assay**. Measured on the live corpus, 4 of 133 groups pooled across assays; the
sharpest is P01160 Wild Type, published as `assay: "CongoRed"` carrying t50
44.851 h — which is the **ThT** median, while that protein's CongoRed data sits at
586.870 h, 13x away. That is a wrong NUMBER in a wrong stratum under a wrong
label, not merely a wrong attachment. It is also why `k114` appears in no M6
record at all: those curves were absorbed into ThT-labelled groups.

Downstream, `m7_assemble`'s uniprot-only fallback then attached such records to
curves of a different assay, gated on construct alone, while asserting in comment
that "assay equality is enforced upstream by the triple key" — false on the
fallback path, and `want_assay` was accepted as a parameter and never read.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Version tag — folded into engine_build_id (§7). Changing ANY `applied` value
# below MUST bump this so descendants are invalidated (old results non-comparable).
# --------------------------------------------------------------------------- #
JOIN_POLICY_VERSION = "join-policy-1.2"

# The analysis this record was distilled from.
SOURCE_ARTIFACT = "M6/M7 assay-axis design checkpoint (2026-08-10)"

# --------------------------------------------------------------------------- #
# The governed join table. `applied` is in force; `previous` is retained so the
# change is reversible. `evidence` states the measurement AND the definition that
# produced it — a bare integer with no definition is not evidence.
# --------------------------------------------------------------------------- #
JOIN_POLICY = {
    "M6_GROUP_KEY": {
        "module": "m6_propensity.score_corpus",
        "previous": ["uniprot", "construct"],
        "applied": ["uniprot", "construct", "assay"],
        "metric": ("the key under which curves are grouped into ONE propensity "
                   "record, whose t50 median and anchoring stratum follow from it"),
        "evidence": {
            "definition": ("groups keyed (uniprot, construct) over curves with a "
                           "POINT t50, counted as pooling-across-assay when the "
                           "group spans >1 distinct condition_vector.assay_type"),
            "groups_total": 133,
            "groups_pooling_across_assay": 4,
            "worst_case": {
                "group": "P01160 / Wild Type",
                "published_label": "CongoRed",
                "published_t50_hours": 44.851,
                "actual_congored_median_hours": 586.870,
                "actual_tht_median_hours": 44.851,
                "note": ("the record is labelled CongoRed, placed in the CongoRed "
                         "stratum, and carries the ThT median"),
            },
            "other_pooled_groups": [
                {"group": "P05067 / Wild Type", "published": 2.030,
                 "tht": 2.001, "congored": 24.000},
                {"group": "P37840 / G 51 D", "published": 29.560,
                 "tht": 20.934, "k114": 48.000},
                {"group": "P37840 / Wild Type", "published": 14.151,
                 "tht": 14.062, "k114": 46.565},
            ],
        },
        "effect": ("SPLITS pooled records onto the assay axis, so a median is no "
                   "longer taken across assays and the anchoring stratum is the "
                   "curve's own assay. Also makes `k114` reachable as an assay in "
                   "its own right rather than absorbed into ThT-labelled groups."),
    },
    "M6_ANCHOR_LEAKAGE": {
        "module": "m6_propensity.score_corpus",
        "previous": "leave_one_group_out",
        "applied": "leave_one_study_out",
        "metric": ("which other groups may appear in the corpus-anchor pool a "
                   "group's intrinsic percentile is computed against"),
        "evidence": {
            "definition": ("for each M6 group, the set of source_study.pmid values "
                           "backing its curves; a group LEAKS when at least one "
                           "other group in its own assay stratum shares a pmid "
                           "with it"),
            "n_groups_with_a_resolvable_study": 164,
            "n_sharing_a_study_with_another_group_in_stratum": 151,
            "fraction_leaking": 0.921,
            "by_assay": {"ThT": 134, "ThS": 11, "CongoRed": 2, "turbidity": 2,
                         "k114": 2},
            "why_the_previous_justification_was_wrong": (
                "the README argued the residual leakage was small because "
                "'proteins rarely co-occur across studies'. That is true -- only "
                "36 of 164 groups draw on more than one study -- but it is not "
                "the mechanism. The leak runs the other way: ONE study "
                "contributing SEVERAL groups to one stratum, so a group is ranked "
                "against measurements sharing its own lab, protocol and batch "
                "effects. Measured that way it is 92.1%, not small."),
            "measured_cost": ("10 of 129 scored groups (7.8%) lose their corpus "
                              "anchor and fall back to the versioned reference "
                              "scale -- all 10 in the ThS stratum, whose median "
                              "pool drops 9 -> 4 because that stratum is "
                              "essentially one study's measurements. ThT, the "
                              "bulk, drops only 110 -> 107. So the cost lands "
                              "exactly where the anchor was never independent "
                              "evidence, which is the point rather than a "
                              "regrettable side effect."),
        },
        "effect": ("the intrinsic corpus anchor becomes leakage-free at the STUDY "
                   "level as section 8 requires, and a group whose stratum cannot "
                   "supply enough independent comparators says so via "
                   "anchor_kind='reference_scale' instead of reporting a "
                   "percentile against its own study"),
    },
    "M6_GAMMA_JOIN": {
        "module": "m6_propensity.score_corpus",
        "previous": "protein_name_last_wins",
        "applied": "protein_name_and_assay",
        "metric": ("the key under which an M4 dual-γ record is attached to a "
                   "propensity group to build its `surface` block"),
        "evidence": {
            "definition": ("γ records in gamma.jsonl resolved to an assay through "
                           "their concentration_series_id via curves_triaged"),
            "gamma_records": 52,
            "resolvable_to_exactly_one_assay": 52,
            "spanning_more_than_one_assay": 0,
            "protein_names_with_gamma": 34,
            "protein_names_whose_gamma_spans_assays": ["Alpha-Synuclein"],
            "alpha_synuclein_assays": ["ThT", "k114"],
            "rationale": ("gamma.jsonl carries a protein NAME, not a uniprot id, so "
                          "M6's surface join was name-based with last-record-wins — "
                          "already a documented limitation. It was harmless only "
                          "because one protein name mapped to one group. Once "
                          "M6_GROUP_KEY splits groups by assay, a single name maps "
                          "to SEVERAL groups and the assay-blind lookup would hand "
                          "the same γ to all of them. γ is the scaling exponent of a "
                          "concentration series measured by a particular assay, so "
                          "that is the same cross-assay contamination this record "
                          "exists to remove — surfacing here as a second-order "
                          "effect of the re-key rather than as a new defect."),
            "scope": ("exactly one protein is affected today (Alpha-Synuclein). The "
                      "fix is nonetheless made at the join rather than special-cased, "
                      "because the hazard is structural and the corpus will grow."),
        },
        "effect": ("ATTACHES a γ surface only from a concentration series measured "
                   "by the group's own assay. A group with no assay-matched γ gets "
                   "no surface, rather than a foreign one."),
    },
    "ASSAY_ENDPOINT_AMYLOID_ADDITIONS": {
        "module": "m6_propensity._AMYLOID_ASSAYS",
        "previous": [],
        "applied": ["k114"],
        "metric": ("which assay-name substrings classify as an AMYLOID endpoint, "
                   "and therefore license the amyloid-specific sequence predictors "
                   "(PASTA, Waltz) rather than nothing"),
        "evidence": {
            "definition": ("assay vocabulary enumerated over all 1,654 curves in "
                           "curves_triaged.jsonl, each passed through "
                           "_assay_endpoint_class"),
            "vocabulary": {"ThT": 1563, "turbidity": 37, "ThS": 33,
                           "CongoRed": 12, "k114": 9},
            "classified_before": {"ThT": "amyloid", "ThS": "amyloid",
                                  "CongoRed": "amyloid", "turbidity": "generic",
                                  "k114": "unknown"},
            "the_only_misclassification": "k114",
            "citation": ("Crystal AS, Giasson BI, Crowe A, Kung M-P, Zhuang Z-P, "
                         "Trojanowski JQ, Lee VM-Y. 'A comparison of amyloid "
                         "fibrillogenesis using the novel fluorescent compound "
                         "K114.' J Neurochem. 2003;86(6):1359-1368."),
            "rationale": ("K114 is (trans,trans)-1-bromo-2,5-bis(3-hydroxycarbonyl-"
                          "4-hydroxy)styrylbenzene, a Congo-red/X-34-derived "
                          "fluorophore introduced specifically to QUANTIFY AMYLOID "
                          "FIBRILLOGENESIS: it binds the cross-beta fibril and "
                          "fluoresces on binding, the same detection principle and "
                          "the same molecular target as Thioflavin T/S and Congo "
                          "Red, all three of which this table already classes as "
                          "amyloid. Leaving it 'unknown' was not a conservative "
                          "choice, it was a WRONG one: it barred PASTA and Waltz — "
                          "the cross-beta-specific predictors — from comparison "
                          "against a cross-beta-specific assay, which is precisely "
                          "the endpoint MATCH the licensing rule exists to permit."),
            "why_it_was_not_bundled_with_join_policy_1_0": (
                "it changes published `comparison_licensed` values and is a claim "
                "about the chemistry rather than about the join, so it needed its "
                "own citation and its own version bump rather than riding along "
                "inside a join fix."),
        },
        "effect": ("k114 series classify as `amyloid`, so PASTA and Waltz become "
                   "licensed against them and TANGO/AGGRESCAN stay barred — the "
                   "same licensing ThT, ThS and CongoRed already receive."),
    },
    "PROPENSITY_ASSAY_GATE": {
        "module": "m7_assemble._propensity_block",
        "previous": "construct_only",
        "applied": "construct_and_assay",
        "metric": ("which mismatches between the curve and the joined M6 record "
                   "refuse the propensity attachment"),
        "evidence": {
            "definition": ("full-schema spine rows whose (uniprot, assay_type, "
                           "construct_id) triple misses the M6 index but whose "
                           "uniprot alone hits, i.e. resolved via the fallback"),
            "n_fallback_full_schema": 96,
            "n_fallback_different_assay": 14,
            "n_presented_as_clean_join": 9,
            "series_presented_as_clean_join": [
                "CPAD-TK-1061", "CPAD-TK-1434", "CPAD-TK-1435", "CPAD-TK-1436",
                "CPAD-TK-1438", "CPAD-TK-1439", "CPAD-TK-1440", "CPAD-TK-1441",
                "CPAD-TK-2530",
            ],
            "rationale": ("m6_propensity attaches a caveat to every record stating "
                          "that intrinsic/cohort are matched on ASSAY ENDPOINT ONLY "
                          "— concentration, pH, temperature and agitation are "
                          "explicitly NOT matched. Assay-matching is therefore the "
                          "single comparability guarantee the record makes about "
                          "itself, and a cross-assay join voids it, leaving "
                          "condition_match='assay_only' a false statement."),
        },
        "effect": ("REFUSES a cross-assay propensity with an explicit reason, "
                   "symmetrically with the existing construct refusal. Assay is "
                   "tested FIRST because it is the stronger claim."),
    },
    "SEQUENCE_AXIS_ASSAY_GATE": {
        "module": "m7_assemble.assemble_curve",
        "previous": "any_uniprot_record",
        "applied": "assay_matched_only",
        "metric": ("whether a protein-level sequence axis may be surfaced from an "
                   "M6 record whose assay differs from the curve's"),
        "evidence": {
            "definition": ("of the 14 different-assay fallbacks, those with "
                           "sequence_axis.available true and sequence_axis.assay "
                           "!= the curve's condition_vector.assay_type"),
            "n_contaminated_blocks": 7,
            "series": ["CPAD-TK-1061", "CPAD-TK-1062", "CPAD-TK-1064",
                       "CPAD-TK-1065", "CPAD-TK-1066", "CPAD-TK-2530",
                       "CPAD-TK-2531"],
            "licensing_decisions_total": 28,
            "licensing_wrong_true": 14,
            "licensing_wrong_false": 10,
            "licensing_correct": 4,
            "rationale": ("the block splits into assay-INVARIANT fields (protein "
                          "name, peptide counts, length range, predictor payloads, "
                          "each predictor's own endpoint_class) and assay-DEPENDENT "
                          "ones (assay, assay_endpoint_class, and per-sub-axis "
                          "endpoint_match_to_assay / comparison_licensed / "
                          "no_comparison_licensed_reason), which are pure functions "
                          "of the assay. Licensing therefore FLIPS rather than "
                          "merely over-licensing: turbidity curves published "
                          "PASTA+Waltz where a generic endpoint licenses "
                          "TANGO+AGGRESCAN. The prior justification — that the "
                          "block is 'identical across the protein's constructs' — "
                          "is true on the construct axis and silent on the assay "
                          "axis, which is the axis that determines licensing."),
            "why_not_suppress_the_dependent_fields": (
                "suppression would be honest about the 14 wrong-True decisions but "
                "would also discard the 10 wrong-False ones, which are genuinely "
                "available comparisons. It is a second information loss, not the "
                "neutral safe choice."),
            "why_not_recompute_in_m7": (
                "recomputing licensing for the curve's own assay inside M7 would "
                "preserve those blocks, but M7's contract is that it does no new "
                "inference — it joins — and it would create a second authoring "
                "site for a rule m6_propensity owns. After M6_GROUP_KEY lands, the "
                "residual cross-assay cases are exactly those with no assay-matched "
                "kinetics at all, where no honest per-curve licensing exists."),
        },
        "effect": ("REFUSES a cross-assay sequence axis with a reason naming the "
                   "licensing dependency. Costs ~5 blocks and keeps the licensing "
                   "rule single-authored in M6."),
    },
}

# --------------------------------------------------------------------------- #
# The signed / dated / rationale-bearing APPLY RECORD (§7 governance).
# --------------------------------------------------------------------------- #
APPLY_RECORD = {
    "applied_at": "2026-08-17",
    "by": "governed apply step",
    "rationale": ("M6 pooled t50 across assays under a first-wins assay label, and "
                  "M7's uniprot-only fallback then attached such records across the "
                  "assay axis while gating on construct alone. Both corrected in "
                  "1.0. 1.1 additionally classifies K114 as an amyloid endpoint, "
                  "with a citation, so the cross-beta-specific predictors are "
                  "licensed against a cross-beta-specific dye."),
    "source_artifact": SOURCE_ARTIFACT,
    "join_policy_version": JOIN_POLICY_VERSION,
    "governance": ("PRISE_DESIGN.md §7/§8: a change that moves published values is "
                   "a signed, dated, rationale-bearing, version-bound artifact; it "
                   "bumps engine_build_id and invalidates descendants (old results "
                   "non-comparable). The previous behaviour is retained for every "
                   "decision, so the change is reversible."),
    "reversible": True,
    "note": ("m6_propensity and m7_assemble CONSUME these values rather than "
             "hard-coding the new behaviour; reverting is a matter of restoring "
             "each `applied` to its `previous` and re-bumping JOIN_POLICY_VERSION."),
    "also_bumped": ("m6_propensity.ANCHOR_VERSION m6-anchor-1.0 -> m6-anchor-1.1 — "
                    "the re-key changes the COMPOSITION of the anchoring stratum, "
                    "which is exactly what that axis exists to describe; leaving it "
                    "unchanged would be a second false version statement."),
    "superseded_note": ("join-policy-1.0 deliberately left "
                        "_assay_endpoint_class('k114') returning 'unknown', so those "
                        "series licensed nothing — honest, but not correct. "
                        "join-policy-1.1 closes it with the citation that call "
                        "required: see ASSAY_ENDPOINT_AMYLOID_ADDITIONS."),
}


# --------------------------------------------------------------------------- #
# Accessors (pure; never raise — a lookup miss returns the supplied default).
# --------------------------------------------------------------------------- #
def applied_policy(name: str, default=None):
    """The APPLIED behaviour for a join decision (e.g. 'PROPENSITY_ASSAY_GATE').
    Falls back to `default` on any miss so a consumer never crashes on a typo."""
    rec = JOIN_POLICY.get(name)
    if not rec:
        return default
    val = rec.get("applied")
    return default if val is None else val


def previous_policy(name: str, default=None):
    """The retained PREVIOUS behaviour for a join decision."""
    rec = JOIN_POLICY.get(name)
    if not rec:
        return default
    val = rec.get("previous")
    return default if val is None else val


def policy_record(name: str) -> dict:
    """The full governed record (previous + applied + evidence + effect) for one
    decision, or {} if unknown. JSON-serialisable."""
    return dict(JOIN_POLICY.get(name, {}))


def manifest() -> dict:
    """The complete governed artifact as a JSON-serialisable dict: version, source,
    the full policy table (both behaviours retained), and the signed/dated apply
    record. Used by anything that persists or displays the governance record."""
    return {
        "join_policy_version": JOIN_POLICY_VERSION,
        "source_artifact": SOURCE_ARTIFACT,
        "join_policy": {k: dict(v) for k, v in JOIN_POLICY.items()},
        "apply_record": dict(APPLY_RECORD),
    }


if __name__ == "__main__":   # pragma: no cover - human-readable dump
    import json
    print(json.dumps(manifest(), indent=2))
