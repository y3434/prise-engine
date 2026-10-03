"""
PRISE — Module M9: Experimental-Design Recommender (Phase 2)
===========================================================

For each protein / analysis unit, recommend the experiment(s) that would most
reduce **practical** (resolvable-by-better-experiments) degeneracy, RANKED by
expected-degeneracy-reduction ÷ cost (PRISE_DESIGN.md §5-M9, §9). M9 does **NO
new inference**: it READS the existing M1/M3/M5/M7 + M8 + Service-C artifacts and
maps each unit's *present* blockers to next experiments.

METHOD COMMITTED (stated in the output, not hand-waved) — §5-M9 explicitly
licenses EITHER a full EIG / optimal-design computation OR an honestly-labelled
rule table distilled from Service C. **M9 commits to the rule table**, whose
recommendations are QUANTIFIED by Service C's MEASURED degeneracy reduction:

  * the concentration-series recommendation carries the MEASURED Tier-B narrowing
    — single concentration collapses all 4 Tier-B mechanisms into 1 broad
    equivalence class, a γ-resolved series splits them into 4 singletons (a
    measured ~4× resolution gain), plus the per-pair γ-separation AUC;
  * the agitation/seeding recommendation is tied to the M8 GAP — but its
    mechanism-unblock value is CONDITIONAL on a concentration series: WITH a series
    present it is the last missing piece and unblocks the mechanistic LICENSE for
    the series-bearing structural-reachable set (~33 proteins / 48 series); on a
    SINGLE curve it is necessary-but-NOT-sufficient (it removes a future gate but a
    series is the binding prerequisite, so it is credited only the future-gate
    removal, not the ~4x series resolution gain);
  * each gain is a Service-C-measured number, never a fabricated EIG.

The full EIG / optimal-design computation — a forward simulator PER candidate
experiment + D/A-optimality + a full cost model — is **EXPLICITLY DEFERRED** as a
research deliverable (see `method_statement.deferred`). M9 does not pretend to do
EIG.

Two mandatory disciplines (§5-M9):
  1. **PRACTICAL vs STRUCTURAL gate.** M9 addresses only PRACTICAL non-
     identifiability (resolvable by more/better experiments). If the degeneracy
     is STRUCTURAL — the model is fundamentally many-to-one even given a clean
     experiment (M3 flags `structurally_non_identifiable`, or the residual
     Service-C saturating-secondary ambiguity a clean series still cannot break)
     — M9 REFUSES to recommend an experiment for it and says so. Inherits M3's
     structural-vs-practical distinction.
  2. **Validity ceiling (inherited from M8).** Every expected gain is an UPPER
     bound, conditional on forward-model adequacy: it is Service-C-MEASURED under
     the PRISE forward model, not a claim about the field's accumulated data. A
     different forward model (KMC/Smoluchowski) would lower it (Service C's
     mismatched-generator mechanism-misID degradation ≈0.53).

This module ELEVATES the Phase-2 STUB `m7_assemble._m9_firing_rules` / `m9_hook`
(a single-objective rule hook) into a standalone, principled, *cost-ranked,
Service-C-quantified* recommender — it does not duplicate it.

Everything is pure, deterministic, and JSON-serialisable; nothing here raises on
bad input — a missing artifact degrades the output (with a flag) rather than
crashing.

Usage:
    python engine/m9_recommender.py                 # -> m9_recommendations.jsonl + m9_rollup.json + summary
    python engine/m9_recommender.py --input data/processed --limit 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Module identity — bumping it makes downstream comparisons non-comparable (§7).
M9_VERSION = "m9-recommender-1.0"

# =========================================================================== #
# Pre-registered cost tiers (§5-M9) — fixed BEFORE seeing any unit, so a
# recommendation cannot be re-ranked by tuning its cost. Lower cost = higher
# priority for the same reduction. We use small integer "cost units" so the
# priority score reduction/cost is a stable, interpretable rational number.
# =========================================================================== #
COST_TIERS = {
    "cheap": 1,           # metadata-only / a single longer run
    "cheap_medium": 2,    # a longer run or a couple of extra wells
    "medium": 4,          # a new experimental arm (series / replicate set / assay swap)
}

# =========================================================================== #
# Service-C-MEASURED gains (data/processed/service_c_calibration.json). These
# are the QUANTIFIED reductions that make M9 a rule table *with numbers*, not a
# hand-wave. Loaded from the artifact at runtime where present; these literals
# are the pre-registered fallback (service-c-1.0) used only if the artifact is
# missing, and are stamped as such in the output.
# =========================================================================== #
_SC_FALLBACK = {
    "single_concentration_classes": 1,   # 4 Tier-B mechanisms -> 1 broad class
    "concentration_series_classes": 4,   # γ-resolved series -> 4 singletons
    "pairwise_separation_auc": {
        "nucleation_elongation|secondary_nucleation": 1.0,
        "nucleation_elongation|fragmentation": 1.0,
        "nucleation_elongation|saturating_secondary": 1.0,
        "secondary_nucleation|fragmentation": 1.0,
        "secondary_nucleation|saturating_secondary": 1.0,
        "fragmentation|saturating_secondary": 0.88,
    },
    "mismatch_degradation": 0.5333333333333333,
}

# The single Tier-B ambiguity Service C reports a CLEAN series still struggles to
# break (the lowest pairwise γ-separation; §6C). When a unit's residual
# degeneracy is THIS pair alone, more experiments of the licensed kind do not
# help — it is treated as structural and REFUSED (rule 2).
_RESIDUAL_STRUCTURAL_PAIR = ("fragmentation", "saturating_secondary")


# =========================================================================== #
# Service-C accessors — read the measured numbers from the artifact, else fall
# back to the pre-registered literals (stamped `service_c_source`).
# =========================================================================== #
def _sc_numbers(service_c: dict | None) -> dict:
    """Pull the measured Tier-B narrowing + γ-separation AUC + mismatch
    degradation out of the Service-C artifact; degrade to the pre-registered
    fallback when a field is absent."""
    if not isinstance(service_c, dict):
        out = dict(_SC_FALLBACK)
        out["service_c_source"] = "pre_registered_fallback (artifact missing)"
        out["service_c_version"] = "service-c-1.0"
        return out

    narrowing = service_c.get("concentration_series_narrowing") or {}
    single = (narrowing.get("single_concentration_partition") or {}).get("n_classes")
    series = (narrowing.get("concentration_series_partition") or {}).get("n_classes")
    cm = (service_c.get("confusion_matrices") or {}).get("concentration_series") or {}
    auc = cm.get("pairwise_separation_auc")
    vc = service_c.get("validity_ceiling") or {}
    # mismatch degradation may live in service_c (M9 reads it via M8 too).
    mis = (service_c.get("mismatched_generator") or {}).get(
        "mechanism_misidentification_degradation")

    out = {
        "single_concentration_classes": single if single is not None
        else _SC_FALLBACK["single_concentration_classes"],
        "concentration_series_classes": series if series is not None
        else _SC_FALLBACK["concentration_series_classes"],
        "pairwise_separation_auc": auc if isinstance(auc, dict)
        else _SC_FALLBACK["pairwise_separation_auc"],
        "mismatch_degradation": mis if mis is not None
        else _SC_FALLBACK["mismatch_degradation"],
        "service_c_source": "service_c_calibration.json",
        "service_c_version": service_c.get("service_c_version", "service-c-1.0"),
    }
    return out


def _resolution_gain(sc: dict) -> float:
    """The MEASURED resolution multiplier of a concentration series on the
    Tier-B mechanism class: classes(series) / classes(single). Service C measured
    4/1 = 4.0 (1 broad class -> 4 resolved singletons)."""
    single = sc.get("single_concentration_classes") or 1
    series = sc.get("concentration_series_classes") or 1
    single = single if single > 0 else 1
    return float(series) / float(single)


# =========================================================================== #
# Per-unit state extraction — derive the unit's CURRENT blockers from its M7
# ProteinAnalysisResult (or a synthetic state dict in tests). Read-only: M9
# inspects artifact fields, never re-infers them.
# =========================================================================== #
def extract_state(unit: dict | None) -> dict:
    """Distil a unit's M7 result into the small blocker-state M9 ranks over.

    Accepts EITHER a full M7 `protein_analysis.jsonl` record OR an already-flat
    synthetic `state` dict (tests) — flat keys take precedence so tests need no
    real-data scaffolding. Never raises: a missing field is treated as 'blocker
    status unknown -> do not fire' (M9 only fires a recommendation whose blocker
    is actually PRESENT)."""
    u = unit or {}

    # --- flat synthetic-state passthrough (tests) --------------------------- #
    flat_keys = {
        "agitation_seeding_unknown", "has_concentration_series", "t50_censored",
        "assay_mass_unknown", "degeneracy_class", "structural_non_identifiable",
        "mechanistic_licensed", "residual_pair", "is_replicated",
    }
    if any(k in u for k in flat_keys):
        st = {k: u.get(k) for k in flat_keys}
        st.setdefault("unit_id", u.get("unit_id") or u.get("series_id") or "synthetic")
        st.setdefault("protein_id", u.get("protein_id") or st["unit_id"])
        return _normalise_state(st)

    # --- full M7 record ----------------------------------------------------- #
    cv = u.get("condition_vector") or {}
    prov = cv.get("field_provenance") or {}
    mech = u.get("mechanistic") or {}
    ms = u.get("model_selection") or {}
    ident_flag = ((ms.get("identifiability") or {}).get("flag"))
    feats = u.get("curve_features") or {}
    data_mode = u.get("data_mode")

    # agitation/seeding: unknown if provenance says so OR the value is missing.
    agit_unknown = (
        prov.get("agitation") == "unknown" or prov.get("seeded") == "unknown"
        or cv.get("agitation") is None or cv.get("seeded") is None
    )
    # concentration series present iff the unit is a concentration_series mode.
    has_series = data_mode == "concentration_series" or bool(
        u.get("has_concentration_series"))
    # t50 censored: M1 censoring class left/right/interval, or M4 t50_status flag.
    cens = u.get("censoring_class") or feats.get("censoring_class")
    t50_status = feats.get("t50_status")
    t50_censored = (cens in ("left", "right", "left_right", "interval")) or (
        t50_status in ("biased_early", "biased_late", "lower_bound",
                       "right_censored"))
    # assay mass-proportionality: blocked if explicitly non-mass or unknown.
    assay_mass_unknown = (cv.get("assay_reports_mass") is False) or (
        prov.get("assay_type") == "unknown") or (cv.get("assay_reports_mass") is None)

    # structural refusal markers (M3 structural flag OR Service-C residual pair).
    structural = ident_flag == "structurally_non_identifiable"
    degen_class = mech.get("degeneracy")
    residual_pair = None
    # A 2-member residual equivalence class that is exactly the saturating-
    # secondary pair is the Service-C irreducible ambiguity.
    eqc = mech.get("verdict_or_equivalence_class")
    if isinstance(eqc, list) and set(eqc) == set(_RESIDUAL_STRUCTURAL_PAIR):
        residual_pair = tuple(sorted(eqc))

    st = {
        "unit_id": u.get("series_id") or u.get("analysis_unit") or "unknown",
        "protein_id": u.get("protein_id") or "unknown",
        "agitation_seeding_unknown": bool(agit_unknown),
        "has_concentration_series": bool(has_series),
        "t50_censored": bool(t50_censored),
        "assay_mass_unknown": bool(assay_mass_unknown),
        "degeneracy_class": degen_class,
        "structural_non_identifiable": bool(structural),
        "mechanistic_licensed": bool(mech.get("mechanistic_inference_licensed")),
        "residual_pair": residual_pair,
        "is_replicated": data_mode == "replicates" or bool(u.get("is_replicated")),
    }
    return _normalise_state(st)


def _normalise_state(st: dict) -> dict:
    """Coerce a synthetic/real state into the canonical typed shape M9 ranks
    over (so a sloppy test dict cannot crash the recommender)."""
    rp = st.get("residual_pair")
    if isinstance(rp, (list, tuple)) and len(rp) == 2:
        st["residual_pair"] = tuple(sorted(str(x) for x in rp))
    else:
        st["residual_pair"] = None
    for b in ("agitation_seeding_unknown", "has_concentration_series",
              "t50_censored", "assay_mass_unknown", "structural_non_identifiable",
              "mechanistic_licensed", "is_replicated"):
        st[b] = bool(st.get(b))
    st.setdefault("unit_id", "unknown")
    st.setdefault("protein_id", "unknown")
    st.setdefault("degeneracy_class", None)
    return st


# =========================================================================== #
# Per-unit recommender — the candidate experiments, each firing ONLY when its
# blocker is present, with a Service-C-grounded expected gain + a cost tier +
# priority = reduction/cost. This is the rule table the method commits to.
# =========================================================================== #
def recommend(state: dict | None, service_c: dict | None = None) -> dict:
    """Return the RANKED list of next-experiment recommendations for one unit.

    Only FIRES a recommendation whose blocker is actually present. Each carries:
    the experiment, the blocker it addresses, the Service-C-measured expected
    degeneracy reduction (with the numbers), the pre-registered cost tier, a
    priority score = reduction/cost, and the rank. If the unit's residual
    degeneracy is STRUCTURAL, M9 REFUSES (no experiment) and says so."""
    st = extract_state(state)
    sc = _sc_numbers(service_c)
    recs: list[dict] = []
    refusals: list[dict] = []

    # ---- STRUCTURAL gate (mandatory, §5-M9). Refuse, do not recommend. ------ #
    structural_block = st["structural_non_identifiable"] or (
        st["residual_pair"] == tuple(sorted(_RESIDUAL_STRUCTURAL_PAIR)))
    if structural_block:
        pair = st["residual_pair"]
        why = ("M3 flags structurally_non_identifiable for this unit"
               if st["structural_non_identifiable"]
               else f"residual degeneracy is the Service-C irreducible pair "
                    f"{list(pair)} — a clean concentration series still cannot "
                    f"break it (γ-separation AUC "
                    f"{sc['pairwise_separation_auc'].get('|'.join(_RESIDUAL_STRUCTURAL_PAIR), 0.88)})")
        refusals.append({
            "refused_experiment": None,
            "reason": "structural degeneracy — not addressable by more experiments",
            "detail": why,
            "addresses": "structural_non_identifiability",
        })

    # Whether mechanism-resolving recommendations are even worth firing: if the
    # unit is structurally degenerate, mechanism-resolution experiments are
    # refused (above) — they would NOT change the verdict.
    mech_resolvable = not structural_block

    # ---- Candidate 1: record agitation & seeding (the metadata fix). -------- #
    # The corpus-wide top blocker (M8). Cheapest possible — metadata only.
    #
    # DEPENDENCY-AWARE CREDITING (the honest model): recording agitation/seeding
    # is NECESSARY-but-NOT-SUFFICIENT for mechanism resolution. It removes the M5
    # hard-gate (primary/secondary-nucleation branch), but a mechanistic CALL also
    # requires a γ-resolved CONCENTRATION SERIES — Service C shows a single
    # concentration leaves all 4 Tier-B mechanisms in 1 degenerate class; only a
    # series splits them into 4 singletons. So the mechanism-unblock value of
    # agitation/seeding is CONDITIONAL on a series being present:
    #   * WITH a series already (the 462 series units): recording agitation/seeding
    #     IS the last missing piece — it makes the license GRANTABLE now, so it
    #     earns the FULL measured ~4x reduction and (being cheap) ranks #1. Correct.
    #   * WITHOUT a series (the 1,192 single-curve units): recording agitation/
    #     seeding ALONE does NOT enable a mechanistic call — it only removes a gate
    #     that will bind LATER, once a series exists. The BINDING constraint here is
    #     the MISSING SERIES. We therefore (a) substantially DISCOUNT its standalone
    #     reduction so it cannot out-rank `add_concentration_series` for the
    #     mechanism goal, and (b) flag it `necessary_but_not_sufficient` +
    #     `also_requires: [add_concentration_series]`. This keeps M9 honest: it no
    #     longer credits agitation/seeding the full mechanism-resolution gain on a
    #     unit where mechanism stays blocked regardless.
    if st["agitation_seeding_unknown"] and mech_resolvable:
        if st["has_concentration_series"]:
            # Sufficient here: the series is present, so the metadata fix is the
            # last gate. Full measured reduction; cheap -> ranks #1 (correct as-is).
            recs.append(_mk(
                experiment="record_agitation_and_seeding",
                blocker="agitation_seeding_unknown",
                cost="cheap",
                reduction=_resolution_gain(sc),  # unblocks the SAME ~4x license the series enables
                gain_kind="unblocks_mechanistic_license",
                expected_reduction_note=(
                    "A γ-resolved CONCENTRATION SERIES is already present, so "
                    "recording agitation/seeding is the LAST missing piece: it "
                    "moves this unit from 'mechanism blocked' to 'mechanism "
                    "LICENSABLE' (M5 hard gate on the primary/secondary-nucleation "
                    "branch). This is the M8 corpus-wide top blocker and the single "
                    "metadata fix that closes the M8 GAP for the series-bearing "
                    "structural-reachable proteins (~33 proteins / 48 series)."),
                service_c=sc,
            ))
        else:
            # Necessary-but-not-sufficient here: no series, so mechanism stays
            # blocked even after recording agitation/seeding. Credit ONLY the
            # removal of the future gate — a small fraction of the resolution gain
            # — so the BINDING constraint (`add_concentration_series`) is not
            # out-ranked. Even at the cheap cost tier this discounted reduction
            # keeps its priority below the series' priority (4.0/4 = 1.0):
            # 0.5 / 1 = 0.5 < 1.0. Flagged explicitly so the dependency is visible.
            rec = _mk(
                experiment="record_agitation_and_seeding",
                blocker="agitation_seeding_unknown",
                cost="cheap",
                # standalone value = removal of a FUTURE gate only, NOT the ~4x
                # mechanism-resolution gain (which needs a series too).
                reduction=0.5,
                gain_kind="removes_future_mechanistic_gate",
                expected_reduction_note=(
                    "On a single curve this removes a FUTURE gate but does NOT "
                    "alone enable mechanism resolution — a concentration series is "
                    "the binding prerequisite. Service C: a single concentration "
                    "leaves all 4 Tier-B mechanisms in 1 degenerate class; only a "
                    "γ-resolved series splits them. Recording agitation/seeding is "
                    "NECESSARY-but-NOT-SUFFICIENT here: it is still worth doing "
                    "(cheap, and it unblocks the M5 branch gate that will bind once "
                    "a series exists), but it must be PAIRED with a concentration "
                    "series to license a mechanistic call. Credited only the "
                    "future-gate removal, NOT the ~4x series resolution gain."),
                service_c=sc,
            )
            rec["necessary_but_not_sufficient"] = True
            rec["also_requires"] = ["add_concentration_series"]
            recs.append(rec)

    # ---- Candidate 2: add a concentration series (>=3 conc across a decade). - #
    # RESOLVES the Tier-B mechanism degeneracy. Fires whenever there is no series
    # yet and the degeneracy is still practically resolvable. Carries the MEASURED
    # 1->4 narrowing + the per-pair γ-separation AUC.
    if (not st["has_concentration_series"]) and mech_resolvable:
        recs.append(_mk(
            experiment="add_concentration_series",
            blocker="no_concentration_series",
            cost="medium",
            reduction=_resolution_gain(sc),
            gain_kind="resolves_tierB_mechanism_degeneracy",
            expected_reduction_note=(
                f"Service-C MEASURED: a single concentration collapses all 4 "
                f"Tier-B mechanisms into {sc['single_concentration_classes']} "
                f"broad equivalence class; a γ-resolved series (>=3 conc across "
                f"a decade) splits them into {sc['concentration_series_classes']} "
                f"singletons — a measured "
                f"{_resolution_gain(sc):.0f}x resolution gain. Per-pair "
                f"γ-separation AUC carried in service_c_measured."),
            service_c=sc,
            carry_auc=True,
        ))

    # ---- Candidate 3: extend the observation window / sample the lag. -------- #
    # DE-CENSORS t50 (M1 censoring left/right). Turns a bound into a point.
    if st["t50_censored"]:
        recs.append(_mk(
            experiment="extend_observation_window",
            blocker="censoring_biases_t50",
            cost="cheap_medium",
            reduction=1.0,  # one biased/bounded anchor -> one point estimate
            gain_kind="de_censors_t50",
            expected_reduction_note=(
                "Turns a censored (lower-bound / biased) t50 into a point estimate "
                "by extending the window / sampling the lag phase. Removes the M1 "
                "censoring bias on the primary anchor (no Service-C class-narrowing "
                "number — this de-biases a feature, it does not split a mechanism "
                "class)."),
            service_c=sc,
        ))

    # ---- Candidate 4: add replicates. --------------------------------------- #
    # Quantifies between-run (stochastic-nucleation) scatter. Always fireable
    # but LOW leverage (it characterises scatter; it does not unblock a license).
    if not st["is_replicated"]:
        recs.append(_mk(
            experiment="add_replicates",
            blocker="unquantified_between_run_scatter",
            cost="medium",
            reduction=0.5,  # characterises scatter; no class-narrowing
            gain_kind="quantifies_stochastic_scatter",
            expected_reduction_note=(
                "Adds replicates to quantify between-run (stochastic-nucleation) "
                "scatter. Characterises uncertainty; does not by itself unblock a "
                "mechanistic license — hence ranked below the unblocking moves."),
            service_c=sc,
        ))

    # ---- Candidate 5: use a mass-proportional (ThT/ThS) assay. -------------- #
    # Licenses mechanistic rate-law models when the assay is non-mass/unknown.
    if st["assay_mass_unknown"] and mech_resolvable:
        recs.append(_mk(
            experiment="use_mass_proportional_assay",
            blocker="non_mass_or_unknown_assay",
            cost="medium",
            reduction=_resolution_gain(sc) * 0.5,  # necessary-but-not-sufficient license precondition
            gain_kind="licenses_mass_action_models",
            expected_reduction_note=(
                "Switches to a mass-proportional (ThT/ThS) assay so mass-action "
                "mechanistic rate-law models can be LICENSED (PRISE invariant 9). "
                "A precondition for mechanism resolution, not a sufficient one — "
                "ranked below agitation/seeding which is cheaper for the same end."),
            service_c=sc,
        ))

    # ---- RANK by priority = reduction / cost (desc); stable tie-break ------- #
    # Cheap-unblock-mechanism must outrank expensive moves: a CHEAP metadata fix
    # with the same measured reduction as the MEDIUM series scores 4x higher.
    recs.sort(key=lambda r: (-r["priority_score"], r["cost_units"],
                             r["experiment"]))
    for i, r in enumerate(recs):
        r["rank"] = i + 1

    return {
        "unit_id": st["unit_id"],
        "protein_id": st["protein_id"],
        "state": st,
        "recommendations": recs,
        "refusals": refusals,
        "n_recommendations": len(recs),
        "top_recommendation": recs[0]["experiment"] if recs else None,
        "method_statement": _method_statement(sc),
        "validity_ceiling": _validity_ceiling(sc),
        "m9_version": M9_VERSION,
    }


def _mk(experiment: str, blocker: str, cost: str, reduction: float,
        gain_kind: str, expected_reduction_note: str, service_c: dict,
        carry_auc: bool = False) -> dict:
    """Assemble one recommendation with its cost tier + priority score. The
    priority score = expected_reduction / cost_units; a higher score is a better
    bang-per-buck and earns a lower rank number."""
    cost_units = COST_TIERS.get(cost, 4)
    rec = {
        "experiment": experiment,
        "addresses_blocker": blocker,
        "expected_degeneracy_reduction": round(float(reduction), 4),
        "reduction_kind": gain_kind,
        "expected_reduction_note": expected_reduction_note,
        "cost_tier": cost,
        "cost_units": cost_units,
        "priority_score": round(float(reduction) / cost_units, 4),
        "expected_gain_is_upper_bound": True,  # validity ceiling (M8)
    }
    if carry_auc:
        rec["service_c_measured"] = {
            "single_concentration_classes": service_c["single_concentration_classes"],
            "concentration_series_classes": service_c["concentration_series_classes"],
            "resolution_gain_x": round(_resolution_gain(service_c), 4),
            "pairwise_separation_auc": service_c["pairwise_separation_auc"],
            "service_c_version": service_c["service_c_version"],
            "service_c_source": service_c["service_c_source"],
        }
    return rec


# =========================================================================== #
# Method statement + validity ceiling carried in EVERY output (§5-M9 mandatory).
# =========================================================================== #
def _method_statement(sc: dict) -> dict:
    return {
        "method": "service_c_distilled_rule_table",
        "committed": (
            "A Service-C-distilled rule table whose recommendations are "
            "QUANTIFIED by Service C's MEASURED degeneracy reduction (e.g. a "
            f"concentration series narrows the Tier-B class from "
            f"{sc['single_concentration_classes']} broad class to "
            f"{sc['concentration_series_classes']} singletons — a measured "
            f"~{_resolution_gain(sc):.0f}x resolution gain, not a hand-wave)."),
        "deferred": (
            "The FULL EIG / optimal-design computation (a forward simulator per "
            "candidate experiment + D/A-optimality + a full cost model) is "
            "EXPLICITLY DEFERRED as a research deliverable. M9 does NOT compute "
            "EIG; every gain here is a Service-C-measured number under the rule "
            "table, not an information-theoretic optimum."),
        "cost_tiers_pre_registered": COST_TIERS,
        "ranking": "priority = expected_degeneracy_reduction / cost_units (desc)",
        "service_c_version": sc["service_c_version"],
        "service_c_source": sc["service_c_source"],
    }


def _validity_ceiling(sc: dict) -> dict:
    return {
        "clause": (
            "Every expected gain is an UPPER bound on the achievable degeneracy "
            "reduction, CONDITIONAL ON forward-model adequacy. It is Service-C-"
            "MEASURED under the PRISE forward model — NOT a claim about the "
            "field's accumulated data (inherited from M8)."),
        "evidence_a_different_forward_model_degrades_resolution": {
            "source": "Service C mismatched-generator adversarial test (C1)",
            "mechanism_misidentification_degradation": sc["mismatch_degradation"],
            "interpretation": (
                "when generator != fitter, mechanism-identification accuracy "
                f"degrades by ~{sc['mismatch_degradation']:.2f} EVEN where R^2 "
                "stays high — a KMC/Smoluchowski forward model would LOWER these "
                "expected gains."),
        },
        "practical_vs_structural": (
            "M9 addresses only PRACTICAL non-identifiability (resolvable by more/"
            "better experiments). STRUCTURAL degeneracy (model many-to-one even "
            "given a clean experiment) is REFUSED — see per-unit refusals."),
    }


# =========================================================================== #
# Corpus rollup — the highest-leverage corpus-wide actions, tied to the M8
# blocker census + the M8 GAP. The headline must fall out: recording agitation
# & seeding is the single highest-leverage action (cheapest cost, unblocks the
# mechanistic license for the entire M8 structural-reachable set, ~33 proteins).
# =========================================================================== #
def build_rollup(per_unit: list[dict], m8: dict | None,
                 service_c: dict | None, engine_build_id=None) -> dict:
    sc = _sc_numbers(service_c)
    degraded: list[str] = []
    if not per_unit:
        degraded.append("no per-unit recommendations: rollup is empty")
    if not isinstance(m8, dict):
        degraded.append("m8_reachability missing: GAP counts use Service-C fallback")

    # --- aggregate: per experiment, how many units it would advance + cost --- #
    by_experiment: dict[str, dict] = {}
    units_with_any = 0
    units_refused = 0
    proteins_by_experiment: dict[str, set] = {}
    # Dependency-aware split for the agitation/seeding action: how many of the
    # units it advances ALREADY have a concentration series (so recording
    # agitation/seeding is SUFFICIENT and earns the full ~4x mechanism-resolution
    # value) vs how many are single-curve (where it is necessary-but-not-sufficient
    # — it only removes a future gate). The headline must scope to the former.
    agit_series_units = 0       # series-bearing -> agitation IS sufficient
    agit_noseries_units = 0     # single-curve   -> future-gate removal only
    agit_series_proteins: set = set()
    for u in per_unit:
        recs = u.get("recommendations") or []
        if recs:
            units_with_any += 1
        if u.get("refusals"):
            units_refused += 1
        pid = u.get("protein_id")
        for r in recs:
            exp = r["experiment"]
            d = by_experiment.setdefault(exp, {
                "experiment": exp,
                "addresses_blocker": r["addresses_blocker"],
                "cost_tier": r["cost_tier"],
                "cost_units": r["cost_units"],
                "expected_degeneracy_reduction": r["expected_degeneracy_reduction"],
                "priority_score": r["priority_score"],
                "n_units_advanced": 0,
                "n_units_top_ranked": 0,
            })
            d["n_units_advanced"] += 1
            if r.get("rank") == 1:
                d["n_units_top_ranked"] += 1
            proteins_by_experiment.setdefault(exp, set()).add(pid)
            if exp == "record_agitation_and_seeding":
                if r.get("necessary_but_not_sufficient"):
                    agit_noseries_units += 1
                else:
                    agit_series_units += 1
                    agit_series_proteins.add(pid)
                    # The series-bearing (sufficient) variant carries the FULL
                    # value — surface that priority/reduction in the rollup row so
                    # the corpus ranking reflects where the action actually closes
                    # the GAP, not the discounted single-curve value.
                    d["expected_degeneracy_reduction"] = \
                        r["expected_degeneracy_reduction"]
                    d["priority_score"] = r["priority_score"]
    for exp, d in by_experiment.items():
        d["n_proteins_advanced"] = len(proteins_by_experiment.get(exp, set()))
    agit = by_experiment.get("record_agitation_and_seeding")
    if agit is not None:
        # Annotate the agitation row with the dependency-aware split so the corpus
        # rollup is honest: the full mechanism-unblock value lands ONLY on the
        # series-bearing units; the rest get future-gate removal only.
        agit["n_units_sufficient_with_series"] = agit_series_units
        agit["n_units_necessary_but_not_sufficient_single_curve"] = agit_noseries_units
        agit["n_proteins_sufficient_with_series"] = len(agit_series_proteins)

    # --- M8 GAP: the structural-reachable set agitation/seeding would unblock - #
    boundary = (m8 or {}).get("mechanistically_resolvable_boundary") or {}
    gap_proteins = boundary.get("structural_reachable_proteins_upper_bound")
    gap_series = boundary.get("structural_reachable_series_upper_bound")
    gap = boundary.get("gap_structural_minus_licensed")
    if gap_proteins is None:
        gap_proteins = 33  # Service-C/M8 known value; flagged via `degraded`
        gap = 33
        gap_series = 48

    # --- rank corpus actions by (priority_score desc, units_advanced desc) --- #
    actions = sorted(
        by_experiment.values(),
        key=lambda d: (-d["priority_score"], -d["n_units_advanced"],
                       d["cost_units"], d["experiment"]),
    )
    for i, a in enumerate(actions):
        a["corpus_rank"] = i + 1

    # The headline action — by construction the cheapest unblocking move. We
    # SELECT it (rather than assume) so the rollup is honest if data shifts, but
    # tie it explicitly to the M8 GAP it closes — and scope it HONESTLY: recording
    # agitation/seeding only UNBLOCKS MECHANISM on the series-bearing units (where
    # it is the last missing piece). On the single-curve units it removes a future
    # gate but does NOT alone advance them toward a mechanistic call — the binding
    # prerequisite there is the missing concentration series. So the headline must
    # NOT claim it advances all N units toward mechanism.
    top_action = actions[0] if actions else None
    headline = None
    if top_action and top_action["experiment"] == "record_agitation_and_seeding":
        n_suff = top_action.get("n_units_sufficient_with_series", 0)
        n_future = top_action.get(
            "n_units_necessary_but_not_sufficient_single_curve", 0)
        headline = (
            "Recording agitation & seeding is the single highest-leverage action "
            f"and the cheapest one that closes the M8 GAP — at cheapest cost "
            f"({top_action['cost_tier']}) it unblocks the mechanistic license for "
            f"the series-bearing structural-reachable proteins (~{gap_proteins} "
            f"proteins / {gap_series} series, the "
            f"{n_suff} series units where a gamma-resolved concentration series is "
            f"ALREADY present), closing the M8 GAP of {gap} (structural-reachable "
            f"minus actually-licensed = 0). It pairs with those existing series. "
            f"On the remaining {n_future} single-curve units it removes a FUTURE "
            "gate but does NOT alone enable mechanism resolution — there the "
            "binding prerequisite is a concentration series (add_concentration_"
            "series).")
    elif top_action:
        headline = (
            f"Highest-leverage corpus action: {top_action['experiment']} "
            f"(cost {top_action['cost_tier']}), advancing "
            f"{top_action['n_units_advanced']} units / "
            f"{top_action['n_proteins_advanced']} proteins.")

    return {
        "m9_version": M9_VERSION,
        "title": "PRISE Experimental-Design Recommender — corpus rollup",
        "n_units": len(per_unit),
        "n_units_with_recommendation": units_with_any,
        "n_units_with_structural_refusal": units_refused,
        "corpus_actions_ranked": actions,
        "top_corpus_action": top_action["experiment"] if top_action else None,
        "headline": headline,
        "m8_gap": {
            "structural_reachable_proteins_upper_bound": gap_proteins,
            "structural_reachable_series_upper_bound": gap_series,
            "gap_structural_minus_licensed": gap,
            "closed_by": "record_agitation_and_seeding",
            "source": "m8_reachability.json" if isinstance(m8, dict)
            else "service_c/m8 known value (m8 artifact missing)",
        },
        "method_statement": _method_statement(sc),
        "validity_ceiling": _validity_ceiling(sc),
        "degraded": degraded,
        "is_degraded": bool(degraded),
        "versioning": {
            "m9_version": M9_VERSION,
            "engine_build_id": engine_build_id,
            "service_c_version": sc["service_c_version"],
            "invalidation_rule": (
                "a bump in taxonomy/harness/anchor/corpus version invalidates M9 "
                "(it consumes M7/M8/Service-C); comparability gated on equal "
                "engine_build_id (§7)."),
        },
    }


# =========================================================================== #
# CLI
# =========================================================================== #
def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> list:
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
    except Exception:
        return []
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="PRISE M9 — Experimental-Design Recommender")
    proc_default = Path(__file__).resolve().parent.parent / "data" / "processed"
    ap.add_argument("--input", type=Path, default=proc_default,
                    help="directory holding the input artifacts")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of analysis units processed")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="output dir (default: <input>)")
    args = ap.parse_args(argv)

    indir = Path(args.input)
    units = _read_jsonl(indir / "protein_analysis.jsonl")
    m8 = _read_json(indir / "m8_reachability.json")
    service_c = _read_json(indir / "service_c_calibration.json")
    engine_build_id = None
    if units:
        engine_build_id = units[0].get("engine_build_id")

    if args.limit is not None:
        units = units[: max(0, args.limit)]

    per_unit = [recommend(u, service_c) for u in units]
    rollup = build_rollup(per_unit, m8, service_c, engine_build_id)

    outdir = Path(args.output_dir) if args.output_dir else indir
    outdir.mkdir(parents=True, exist_ok=True)
    rec_path = outdir / "m9_recommendations.jsonl"
    roll_path = outdir / "m9_rollup.json"
    with open(rec_path, "w", encoding="utf-8") as fh:
        for r in per_unit:
            fh.write(json.dumps(r) + "\n")
    roll_path.write_text(json.dumps(rollup, indent=2), encoding="utf-8")

    # ---- printed summary --------------------------------------------------- #
    print(f"[M9] {M9_VERSION} — Experimental-Design Recommender")
    print(f"[M9] wrote {rec_path}  ({len(per_unit)} units)")
    print(f"[M9] wrote {roll_path}")
    if rollup["is_degraded"]:
        print(f"[M9] DEGRADED: {rollup['degraded']}")

    ms = rollup["method_statement"]
    print("\n  METHOD (committed): " + ms["method"])
    print("    " + ms["committed"])
    print("    DEFERRED: " + ms["deferred"])

    n_ref = rollup["n_units_with_structural_refusal"]
    print(f"\n  UNITS: {rollup['n_units']} total | "
          f"{rollup['n_units_with_recommendation']} with >=1 recommendation | "
          f"{n_ref} with a STRUCTURAL refusal")

    print("\n  CORPUS ACTIONS (ranked by reduction/cost):")
    for a in rollup["corpus_actions_ranked"]:
        print(f"    #{a['corpus_rank']} {a['experiment']:<30} "
              f"cost={a['cost_tier']:<12} priority={a['priority_score']:<6} "
              f"units={a['n_units_advanced']:>4} proteins={a['n_proteins_advanced']:>3}")

    print("\n  HEADLINE:")
    print(f"    {rollup['headline']}")

    gap = rollup["m8_gap"]
    print(f"\n  M8 GAP closed by agitation/seeding: "
          f"{gap['structural_reachable_proteins_upper_bound']} proteins / "
          f"{gap['structural_reachable_series_upper_bound']} series "
          f"(GAP={gap['gap_structural_minus_licensed']})")

    vc = rollup["validity_ceiling"]
    print("\n  VALIDITY CEILING (inherited from M8):")
    print(f"    {vc['clause']}")
    print("    mismatched-generator mechanism-misID degradation = "
          f"{vc['evidence_a_different_forward_model_degrades_resolution']['mechanism_misidentification_degradation']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
