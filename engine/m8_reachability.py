"""
PRISE — Module M8: Inferential Reachability Map (Phase 2)
========================================================

A database-scale map of **what is inferable from real bulk aggregation kinetics,
and what is not** — re-scoped honestly as reachability **under the PRISE forward
model** (PRISE_DESIGN.md §5-M8, decision rows R4 + C1). M8 does **NO new
inference**: it CONSOLIDATES the existing M1–M7 + Service-C artifacts into a
falsifiable reachability map. Every headline carries the conditioning clause
"reachability UNDER the PRISE forward model" and the mandatory validity ceiling
(§5-M8): M8 reassurances are *upper bounds* on reachability and *lower bounds* on
degeneracy, conditional on forward-model adequacy. This is NOT a claim about "the
field's accumulated data" — the external KMC/Smoluchowski + blind-literature
validation that would license that framing is DEFERRED (§6C).

The map has six parts (per the §5-M8 construction spec):

  1. **Inference ladder** — of N curves (and N proteins), how many REACH each
     rung: signal_present -> fittable -> descriptive -> scaling(γ-significant) ->
     mechanistic_licensed. Each rung is a falsifiable statement with count +
     fraction + the conditioning clause. Counts come from the M7 rollup
     `by_information_yield`, the ETL census, and the M1 fittability classes — so
     the ladder is monotone-non-increasing and reconciles with the corpus total.

  2. **Reachability grid** (the "map") — a 2D census over
     axis A = data_richness {single_curve, replicates, concentration_series}
     × axis B = condition_completeness {low, medium, high}. Each cell carries a
     count and a modal **inferential-status metric** ∈ {recoverable, degenerate,
     unconstrained}, assigned by a PRE-REGISTERED rule table (`STATUS_RULE`).

  3. **Blocker census** — IMPACT-RANKED, with counts, of *why* the corpus does
     not reach higher rungs (agitation/seeding unknown; no concentration series;
     censoring; non-mass / unknown assay; low metadata completeness).

  4. **Mechanistically-resolvable boundary** (Service-C-backed) — the honest
     headline. STRUCTURAL reachability (proteins/series whose data STRUCTURE
     could resolve mechanism via γ under the forward model — an UPPER bound, per
     Service C's clean-series K2 split) vs ACTUAL licensed (0 corpus-wide). The
     GAP is the finding: the structure supports mechanism resolution for ~N
     proteins, but a metadata gap (agitation/seeding unrecorded) blocks ALL.

  5. **Validity ceiling** (MANDATORY, prominent top-level block) — the upper/
     lower-bound clause, the Service-C mismatched-generator mechanism-misID
     degradation (~0.53) as evidence a different forward model degrades
     resolution, and the DEFERRED-validation flag.

  6. **Versioning + query** — carries the M7 `engine_build_id`, the Service-C
     version, and this module's version; states the invalidation rule (§7); and
     exposes `reachable_at(level)` over the assembled map.

  7. **Metadata explanation** (M11 dependency, ADDITIVE + GRACEFUL) — M8 CONSUMES
     the M11 corpus artifact `data/processed/metadata_quality.json` to make the
     reachability CAUSAL: a `metadata_explanation` block (WHY the boundary GAP is
     unresolved — the {agitation, seeded}='unknown' universal blocker M11
     quantifies), a `why_gap_unresolved` on the resolvable boundary (the ranked
     `why_blocked` metadata fields + their M11 information_gain), a per-blocker
     M11 cross-reference on the blocker census (field(s) responsible +
     information-gain-to-unblock), and a top-level `metadata_completeness_summary`
     (median metadata_score + mechanistic_completeness). When the M11 artifact is
     ABSENT this degrades to `{m11_available: false}` and every other M8 key +
     the validity-ceiling honesty is preserved UNCHANGED. HONESTY carried through:
     completeness ≠ correctness, and M11 information_gain numbers are STRUCTURAL
     UPPER BOUNDS (inheriting this map's validity ceiling) except the
     Service-C-measured ones.

Everything is pure, deterministic, and JSON-serialisable; nothing here raises on
bad input — a missing artifact degrades the map (with a `degraded` flag + note)
rather than crashing.

Usage:
    python engine/m8_reachability.py                 # -> data/processed/m8_reachability.json + summary
    python engine/m8_reachability.py --input data/processed
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

M8_VERSION = "m8-reachability-1.0"

# --------------------------------------------------------------------------- #
# §5-M8 axis vocabularies + the inference ladder (ordered low -> high). The
# ladder mirrors M7's information_yield but is expressed as cumulative REACH:
# reaching a rung implies reaching every rung below it (monotone).
# --------------------------------------------------------------------------- #
LADDER = [
    "signal_present",        # a curve exists with usable signal (not pure noise)
    "fittable",              # M1 deemed it fittable (a model can be fit)
    "descriptive_regime",    # M7 licensed a descriptive (or higher) regime call
    "scaling",               # γ scaling exponent resolved (95% CI excludes 0)
    "mechanistic_licensed",  # a single Tier-B mechanism licensed (Service-C gated)
]

# comparability-critical condition fields (§6A). agitation+seeded are the
# mechanism-branch keys; the first four are the comparability keys.
CRITICAL_FIELDS = ["concentration", "pH", "temperature", "assay_type",
                   "agitation", "seeded"]
MECH_BRANCH_FIELDS = ["agitation", "seeded"]

DATA_RICHNESS = ["single_curve", "replicates", "concentration_series"]
COMPLETENESS_BINS = ["low", "medium", "high"]

# --------------------------------------------------------------------------- #
# PRE-REGISTERED inferential-status rule (§5-M8 per-cell metric). Keyed by
# (data_richness, completeness_bin); the value is the modal status a cell of that
# kind earns UNDER THE FORWARD MODEL, with the qualifier of WHAT it is
# (un)reachable FOR. The rule is a small constant table so it is auditable and
# falsifiable, not buried in branches.
#
#   recoverable  — the data structure + metadata could resolve the target
#                  inference under the forward model (an UPPER bound).
#   degenerate   — the structure is informative but the target is degenerate
#                  given what is known (e.g. mechanism without agitation/seeding).
#   unconstrained— a prerequisite is missing, so the target is not even posed
#                  (e.g. scaling without a concentration axis; comparability with
#                  low metadata).
#
# Reading: status is for the HIGHEST target the cell could in principle reach.
# Because agitation/seeded are unknown corpus-wide, NO real cell reaches the
# mechanistic target — the best any cell earns is `recoverable` for SCALING.
# --------------------------------------------------------------------------- #
STATUS_RULE = {
    ("concentration_series", "high"):   ("recoverable",
        "γ-scaling resolvable; mechanism degenerate (agitation/seeding unknown)"),
    ("concentration_series", "medium"): ("recoverable",
        "γ-scaling resolvable; comparability widened by missing condition fields"),
    ("concentration_series", "low"):    ("degenerate",
        "concentration axis present but low metadata blocks comparability"),
    ("replicates", "high"):             ("degenerate",
        "descriptive resolvable; no concentration axis -> scaling degenerate"),
    ("replicates", "medium"):           ("degenerate",
        "descriptive resolvable; scaling/mechanism degenerate"),
    ("replicates", "low"):              ("unconstrained",
        "no concentration axis and low metadata"),
    ("single_curve", "high"):           ("degenerate",
        "descriptive resolvable; single curve leaves Tier-B degenerate (K2)"),
    ("single_curve", "medium"):         ("degenerate",
        "descriptive resolvable; scaling/mechanism degenerate"),
    ("single_curve", "low"):            ("unconstrained",
        "single curve + low metadata: comparability and scaling unconstrained"),
}
# the worst applicable status (for summarising a whole map)
STATUS_ORDER = {"unconstrained": 0, "degenerate": 1, "recoverable": 2}


# ======================================================================== #
# small pure helpers
# ======================================================================== #
def _completeness_bin(field_provenance: dict) -> tuple[int, str]:
    """#critical fields KNOWN -> a low/medium/high bin.

    WHY thirds of the 6-field count: agitation+seeded are unknown corpus-wide, so
    the achievable max is 4/6; we still bin on the full 6 so the metric does not
    silently 'reward' a corpus for fields it can never have — a curve missing only
    agitation+seeding lands in `high` (4/6), one also missing pH/T lands lower.
    """
    fp = field_provenance or {}
    known = sum(1 for k in CRITICAL_FIELDS if fp.get(k) == "known")
    if known <= 2:
        return known, "low"
    if known <= 3:
        return known, "medium"
    return known, "high"


def _data_richness(curve: dict, series_sizes: dict) -> str:
    """concentration_series (>=3 distinct-concentration group) / replicates / single."""
    cs = curve.get("concentration_series_id")
    if cs and series_sizes.get(cs, 0) >= 3:
        return "concentration_series"
    if cs and series_sizes.get(cs, 0) >= 2:
        return "replicates"
    return "single_curve"


def _frac(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


# ======================================================================== #
# 1. INFERENCE LADDER
# ======================================================================== #
def build_ladder(rollup: dict, n_curves: int, n_proteins: int,
                 fittability: Counter, n_signal_curves: int,
                 proteins_by_rung: dict | None = None) -> dict:
    """Cumulative reach per rung, falsifiable, with the conditioning clause.

    Curve counts come from the M7 rollup `by_information_yield` (mechanistic /
    scaling / descriptive / signal_only / uninformative) made CUMULATIVE, cross-
    checked against the M1 fittability census. Monotone-non-increasing by
    construction (each rung sums the tiers at/above it).
    """
    by = (rollup or {}).get("by_information_yield", {}) or {}
    mech = int(by.get("mechanistic", 0))
    scaling = int(by.get("scaling", 0))
    descriptive = int(by.get("descriptive", 0))
    signal_only = int(by.get("signal_only", 0))
    # fittable from M1 (independent census) — used to cross-check the descriptive rung
    n_fittable = int(fittability.get("fittable", 0)) or (mech + scaling + descriptive)

    # cumulative reach: a curve "reaches" rung R if its yield is R or higher.
    reach = {
        "signal_present":       n_signal_curves,
        "fittable":             n_fittable,
        "descriptive_regime":   mech + scaling + descriptive,
        "scaling":              mech + scaling,
        "mechanistic_licensed": mech,
    }

    statements = {
        "signal_present":
            "%d / %d curves (%s) carry usable signal under the PRISE forward model",
        "fittable":
            "%d / %d curves (%s) are fittable (a model can be fit) under the PRISE forward model",
        "descriptive_regime":
            "%d / %d curves (%s) reach a licensed descriptive regime under the PRISE forward model",
        "scaling":
            "%d / %d curves (%s) reach a resolved quantitative scaling exponent (γ 95%% CI excludes 0) under the PRISE forward model",
        "mechanistic_licensed":
            "%d / %d curves (%s) reach a licensed single Tier-B mechanism under the PRISE forward model",
    }

    rungs = []
    prev = None
    monotone = True
    for name in LADDER:
        c = reach[name]
        if prev is not None and c > prev:
            monotone = False
        prev = c
        rungs.append({
            "rung": name,
            "curves": c,
            "curve_fraction": _frac(c, n_curves),
            "curve_statement": statements[name] % (c, n_curves, _pct(c, n_curves)),
        })

    # proteins per rung (a protein REACHES a rung if any of its curves does)
    if proteins_by_rung:
        for r in rungs:
            p = int(proteins_by_rung.get(r["rung"], 0))
            r["proteins"] = p
            r["protein_fraction"] = _frac(p, n_proteins)

    return {
        "n_curves": n_curves,
        "n_proteins": n_proteins,
        "rungs": rungs,
        "monotone_non_increasing": monotone,
        "conditioning_clause": "all counts are reachability UNDER THE PRISE FORWARD MODEL (see validity_ceiling)",
    }


# ======================================================================== #
# 2. REACHABILITY GRID
# ======================================================================== #
def build_grid(curves: list, series_sizes: dict) -> dict:
    """2D census (data_richness × condition_completeness) with per-cell status.

    Cells PARTITION the curves (every curve lands in exactly one cell), so the
    cell counts sum to N. Status is assigned by the pre-registered STATUS_RULE,
    refined per-cell by whether the assay is mass-proportional (a mechanism-
    license prerequisite) — recorded but it cannot upgrade a cell past
    `recoverable` because agitation/seeded are unknown corpus-wide.
    """
    cells = {}
    for rich in DATA_RICHNESS:
        for comp in COMPLETENESS_BINS:
            status, reason = STATUS_RULE[(rich, comp)]
            cells[f"{rich}|{comp}"] = {
                "data_richness": rich,
                "completeness": comp,
                "count": 0,
                "n_mass_assay": 0,
                "inferential_status": status,
                "status_reason": reason,
            }

    for cu in curves:
        cv = cu.get("condition_vector", {}) or {}
        rich = _data_richness(cu, series_sizes)
        _, comp = _completeness_bin(cv.get("field_provenance", {}) or {})
        cell = cells[f"{rich}|{comp}"]
        cell["count"] += 1
        if cv.get("assay_reports_mass") is True:
            cell["n_mass_assay"] += 1

    total = sum(c["count"] for c in cells.values())
    by_status = Counter(c["inferential_status"] for c in cells.values()
                        for _ in range(c["count"]))
    return {
        "axis_A_data_richness": DATA_RICHNESS,
        "axis_B_condition_completeness": COMPLETENESS_BINS,
        "completeness_definition":
            "bin on #known of 6 comparability-critical fields "
            "(concentration,pH,temperature,assay_type,agitation,seeded): "
            "low<=2, medium=3, high>=4. agitation+seeded are unknown corpus-wide "
            "so the achievable ceiling is 'high' at 4/6.",
        "status_rule": {f"{k[0]}|{k[1]}": {"status": v[0], "reason": v[1]}
                        for k, v in STATUS_RULE.items()},
        "cells": cells,
        "curves_partitioned": total,
        "n_curves_by_status": dict(by_status),
        "partition_ok": total,  # caller asserts == n_curves
    }


# ======================================================================== #
# 3. BLOCKER CENSUS
# ======================================================================== #
def build_blocker_census(curves: list, series_sizes: dict,
                         censoring: Counter) -> dict:
    """Impact-ranked count of what blocks higher rungs.

    Each blocker is a count of curves it AFFECTS, plus the rung it blocks. Ranked
    by count descending (impact). Corpus-wide blockers (agitation/seeding) are
    flagged as such — they block the SAME branch for every curve.
    """
    n = len(curves)
    n_agit_unknown = 0
    n_seed_unknown = 0
    n_no_series = 0
    n_non_mass = 0
    n_low_completeness = 0
    for cu in curves:
        cv = cu.get("condition_vector", {}) or {}
        fp = cv.get("field_provenance", {}) or {}
        if fp.get("agitation") != "known":
            n_agit_unknown += 1
        if fp.get("seeded") != "known":
            n_seed_unknown += 1
        if _data_richness(cu, series_sizes) != "concentration_series":
            n_no_series += 1
        if cv.get("assay_reports_mass") is not True:
            n_non_mass += 1
        _, comp = _completeness_bin(fp)
        if comp == "low":
            n_low_completeness += 1

    # censoring biases t50 — count curves with ANY censoring (not 'none')
    n_censored = sum(v for k, v in censoring.items() if k not in (None, "none"))

    blockers = [
        {"blocker": "agitation_seeding_unknown",
         "count": max(n_agit_unknown, n_seed_unknown),
         "fraction": _frac(max(n_agit_unknown, n_seed_unknown), n),
         "blocks_rung": "mechanistic_licensed",
         "scope": "corpus_wide",
         "detail": f"agitation unknown on {n_agit_unknown}, seeding unknown on "
                   f"{n_seed_unknown} curves — undetermines the primary/secondary "
                   "nucleation branch corpus-wide (same branch for every curve)"},
        {"blocker": "no_concentration_series",
         "count": n_no_series,
         "fraction": _frac(n_no_series, n),
         "blocks_rung": "scaling",
         "scope": "per_curve",
         "detail": f"{n_no_series} curves are not in a >=3-concentration family, "
                   "so a γ scaling exponent cannot be resolved"},
        {"blocker": "censoring_biases_t50",
         "count": n_censored,
         "fraction": _frac(n_censored, n),
         "blocks_rung": "descriptive_regime",
         "scope": "per_curve",
         "detail": f"{n_censored} curves carry left/right/interval censoring "
                   "that biases t50 and lag estimates",
         "by_censoring_class": {k: v for k, v in sorted(censoring.items(),
                                                        key=lambda kv: -kv[1])
                                if k not in (None, "none")}},
        {"blocker": "non_mass_or_unknown_assay",
         "count": n_non_mass,
         "fraction": _frac(n_non_mass, n),
         "blocks_rung": "mechanistic_licensed",
         "scope": "per_curve",
         "detail": f"{n_non_mass} curves use a non-mass-proportional or unknown "
                   "assay, so a mass-action mechanism cannot be licensed"},
        {"blocker": "low_metadata_completeness",
         "count": n_low_completeness,
         "fraction": _frac(n_low_completeness, n),
         "blocks_rung": "comparability",
         "scope": "per_curve",
         "detail": f"{n_low_completeness} curves know <=2 of 6 comparability-"
                   "critical fields, blocking cross-condition comparability"},
    ]
    blockers.sort(key=lambda b: b["count"], reverse=True)
    for i, b in enumerate(blockers, 1):
        b["impact_rank"] = i
    return {
        "n_curves": n,
        "blockers_impact_ranked": blockers,
        "headline": "what blocks inference (impact-ranked); the top corpus-wide "
                    "blocker undetermines the mechanistic branch for the WHOLE corpus",
    }


# ======================================================================== #
# 4. MECHANISTICALLY-RESOLVABLE BOUNDARY (Service-C-backed)
# ======================================================================== #
def build_resolvable_boundary(curves: list, series_sizes: dict,
                              manifest: dict, rollup: dict,
                              service_c: dict | None) -> dict:
    """STRUCTURAL-reachable (upper bound) vs ACTUAL-licensed; the GAP is the headline.

    STRUCTURAL: proteins / series that have a >=3-concentration family with a
    mass-proportional assay — Service C shows a CLEAN such series splits the
    Tier-B degeneracy via γ (the K2 result), so these COULD resolve mechanism
    UNDER THE FORWARD MODEL. This is an UPPER bound.
    ACTUAL: n_mechanistic_licensed from the M7 rollup (0 corpus-wide).
    GAP = structural - actual: the data STRUCTURE supports mechanism resolution
    for ~N proteins, but a metadata gap (agitation/seeding) blocks ALL of them.
    """
    # observed structural reachability from the triaged curves (join-level truth)
    prot_series_mass: dict[str, bool] = {}
    series_mass: dict[str, bool] = {}
    for cu in curves:
        cs = cu.get("concentration_series_id")
        if not cs or series_sizes.get(cs, 0) < 3:
            continue
        cv = cu.get("condition_vector", {}) or {}
        is_mass = cv.get("assay_reports_mass") is True
        pid = cu.get("protein_id")
        if pid is not None:
            prot_series_mass[pid] = prot_series_mass.get(pid, False) or is_mass
        series_mass[cs] = series_mass.get(cs, False) or is_mass

    obs_structural_proteins = sum(1 for v in prot_series_mass.values() if v)
    obs_structural_series = sum(1 for v in series_mass.values() if v)

    # manifest census (the design-cited structural counts: 34 proteins / 52 series)
    counts = (manifest or {}).get("counts", {}) or {}
    manifest_proteins_with_series = int(counts.get("proteins_with_concentration_series", 0))
    manifest_series_groups = int(counts.get("concentration_series_groups", 0))

    actual_licensed = int((rollup or {}).get("n_mechanistic_licensed", 0))

    # Service-C evidence that a clean series SPLITS Tier-B (the upper-bound basis)
    sc_split = None
    sc_version = None
    if service_c:
        sc_version = service_c.get("service_c_version")
        narrowing = service_c.get("concentration_series_narrowing", {}) or {}
        sc_split = {
            "single_concentration_classes":
                (narrowing.get("single_concentration_partition", {}) or {}).get("n_classes"),
            "concentration_series_classes":
                (narrowing.get("concentration_series_partition", {}) or {}).get("n_classes"),
            "series_narrows_degeneracy": narrowing.get("series_narrows_degeneracy"),
            "pairs_newly_resolved_by_series":
                narrowing.get("pairs_newly_resolved_by_series"),
        }

    structural_upper = obs_structural_proteins or manifest_proteins_with_series
    gap = structural_upper - actual_licensed
    return {
        "definition": "STRUCTURAL reachability = proteins/series with a "
                      ">=3-concentration family AND a mass-proportional assay; "
                      "Service C shows such a series γ-splits the Tier-B degeneracy "
                      "(K2), so they COULD resolve mechanism UNDER the forward model "
                      "(an UPPER bound). ACTUAL = mechanism licenses granted.",
        "structural_reachable_proteins_upper_bound": structural_upper,
        "structural_reachable_series_upper_bound": obs_structural_series or manifest_series_groups,
        "manifest_proteins_with_concentration_series": manifest_proteins_with_series,
        "manifest_concentration_series_groups": manifest_series_groups,
        "observed_structural_proteins_mass_series": obs_structural_proteins,
        "observed_structural_series_mass": obs_structural_series,
        "actual_mechanistically_licensed": actual_licensed,
        "gap_structural_minus_licensed": gap,
        "headline": (
            f"The data STRUCTURE supports mechanism resolution for ~{structural_upper} "
            f"proteins ({obs_structural_series or manifest_series_groups} series) UNDER "
            f"the PRISE forward model, but {actual_licensed} are actually licensed "
            f"corpus-wide — a GAP of {gap}. The whole gap is closed by ONE metadata "
            "fix: recording agitation/seeding (the corpus-wide mechanistic blocker)."),
        "service_c_evidence": sc_split,
        "service_c_version": sc_version,
        "is_upper_bound": True,
    }


# ======================================================================== #
# 5. VALIDITY CEILING (MANDATORY)
# ======================================================================== #
def build_validity_ceiling(service_c: dict | None) -> dict:
    """The mandatory §5-M8 clause. Prominent, non-empty, citing Service-C evidence."""
    mech_misid_degradation = None
    matched_acc = None
    sc_version = None
    if service_c:
        sc_version = service_c.get("service_c_version")
        mm = service_c.get("mismatched_generator", {}) or {}
        mech_misid_degradation = mm.get("misidentification_degradation")
        matched_acc = mm.get("matched_mechanism_id_accuracy")
    return {
        "clause": "M8 reachability counts are UPPER bounds on reachability and "
                  "LOWER bounds on degeneracy, CONDITIONAL ON forward-model "
                  "adequacy. This is reachability UNDER THE PRISE FORWARD MODEL — "
                  "NOT a claim about the field's accumulated data.",
        "evidence_a_different_forward_model_degrades_resolution": {
            "source": "Service C mismatched-generator adversarial test (C1)",
            "mechanism_misidentification_degradation": mech_misid_degradation,
            "matched_mechanism_id_accuracy": matched_acc,
            "interpretation": "when generator != fitter, mechanism-identification "
                              "accuracy degrades by ~"
                              f"{mech_misid_degradation if mech_misid_degradation is not None else 'N/A'}"
                              " EVEN where R^2 stays high — so a different (e.g. KMC/"
                              "Smoluchowski) forward model would lower these reachability "
                              "counts. The reassurances are conditional, not absolute.",
        },
        "deferred_external_validation": {
            "status": "DEFERRED",
            "items": [
                "full external KMC/Smoluchowski mismatched generator",
                "blind real-literature reality check (Aβ42/α-syn/insulin/β2m vs field consensus)",
                "artifact-realism + synthetic->real transfer measurement",
            ],
            "consequence": "until these hold, the 'field's accumulated data' framing "
                           "is NOT licensed; this map is 'reachability under the PRISE "
                           "forward model' only (§5-M8, §6C).",
        },
        "without_this_clause": "the map would be an unfalsifiable over-claim and is not shipped (§5-M8).",
        "service_c_version": sc_version,
    }


# ======================================================================== #
# M11 metadata-quality EXPLANATION (why the reachability gap is unresolved)
# ======================================================================== #
# M8 answers WHAT is (un)reachable; M11 answers WHY. M8 CONSUMES the M11 corpus
# artifact (data/processed/metadata_quality.json) — read-only, graceful — and
# attaches a `metadata_explanation` block that makes the reachability CAUSAL:
# the structural boundary GAP is not a mystery, it is the {agitation, seeded}
# metadata blocker M11 quantifies. M11 mirrors the SAME M5 gates M8 already
# reports, so this is a re-statement in the metadata layer, never a new claim.
#
# HONESTY (carried through into the block): completeness ≠ correctness (a
# metadata_score is PRESENCE, not value-correctness), and the M11 information_gain
# numbers are STRUCTURAL UPPER BOUNDS under the DAG (inheriting M8's own validity
# ceiling) EXCEPT the Service-C-measured ones (gain_basis='service_c_measured').

# Map each M8 blocker to the M11 ontology field(s) responsible for it + the M11
# corpus action whose information_gain unblocks it. Keyed by the M8 blocker name.
_M8_BLOCKER_TO_M11 = {
    "agitation_seeding_unknown": {
        "m11_fields": ["agitation", "seeding"],
        "m11_action": "record_agitation_and_seeding",
        "gain_basis": "service_c_measured",
    },
    "no_concentration_series": {
        "m11_fields": ["concentration_series"],
        "m11_action": "add_concentration_series",
        "gain_basis": "service_c_measured",
    },
    "non_mass_or_unknown_assay": {
        "m11_fields": ["assay", "dye"],
        "m11_action": "record_assay_mass_proportionality",
        "gain_basis": "structural_estimate",
    },
    "low_metadata_completeness": {
        "m11_fields": ["pH", "temperature", "concentration", "construct"],
        "m11_action": "record_pH",
        "gain_basis": "structural_estimate",
    },
    "censoring_biases_t50": {
        "m11_fields": [],   # censoring is a DATA-window limit, not a metadata field
        "m11_action": None,
        "gain_basis": "not_a_metadata_field",
    },
}


def _m11_action_index(corpus: dict) -> dict:
    """Index the M11 corpus recommendations by action name (rank/impact/effort/
    gain-over-effort), so an M8 blocker can cite the M11 information-gain-to-unblock."""
    out = {}
    for r in (corpus or {}).get("corpus_recommendations_ranked", []) or []:
        act = r.get("action")
        if act:
            out[act] = r
    return out


def build_metadata_explanation(m11: dict | None) -> dict:
    """The `metadata_explanation` block: M11's causal account of WHY M8's
    reachability gap is unresolved. Degrades gracefully to {m11_available: False}
    when the M11 artifact is absent — M8 keeps ALL its other keys unchanged.

    `m11` is the parsed data/processed/metadata_quality.json ({corpus, by_protein})."""
    if not m11 or not isinstance(m11, dict) or "corpus" not in m11:
        return {
            "m11_available": False,
            "note": ("data/processed/metadata_quality.json not found — run "
                     "`python engine/m11_metadata.py` to make the reachability "
                     "gap CAUSAL. M8 reachability is unchanged (degraded)."),
        }
    corpus = m11.get("corpus", {}) or {}
    ub = corpus.get("universal_blocker", {}) or {}
    action_idx = _m11_action_index(corpus)

    # ranked metadata blockers with the M11 information_gain to unblock. The
    # {agitation, seeded} pair is the corpus-wide #1 (M11/M9 both rank it first).
    why_blocked = []
    for rec in (corpus.get("corpus_recommendations_ranked", []) or [])[:5]:
        why_blocked.append({
            "rank": rec.get("rank"),
            "m11_action": rec.get("action"),
            "m11_fields": _M8_BLOCKER_TO_M11.get(  # reverse-lookup best-effort
                next((b for b, m in _M8_BLOCKER_TO_M11.items()
                      if m.get("m11_action") == rec.get("action")), ""), {}
            ).get("m11_fields", []),
            "impact": rec.get("impact"),
            "effort": rec.get("effort"),
            "information_gain_over_effort": rec.get("total_gain_over_effort"),
            "curves_affected": rec.get("curves_affected"),
            "gain_basis": ("service_c_measured"
                           if rec.get("action") in ("record_agitation_and_seeding",
                                                    "add_concentration_series")
                           else "structural_estimate"),
        })

    return {
        "m11_available": True,
        "m11_version": corpus.get("version"),
        "m11_ontology_version": corpus.get("ontology_version"),
        "n_curves": corpus.get("n_curves"),
        "universal_blocker": {
            "fields": MECH_BRANCH_FIELDS,   # ['agitation', 'seeded']
            "n_curves_blocked": ub.get("n_curves_agitation_or_seeding_blocked"),
            "fraction": ub.get("fraction"),
            "n_mechanism_licensed": corpus.get("n_mechanism_licensed"),
            "why_mechanism_licensed_is_zero": ub.get("why_mechanism_licensed_is_zero"),
        },
        # the ranked metadata blockers -> the M11 information_gain to unblock each
        "why_blocked": why_blocked,
        "action_index": action_idx,   # action -> full M11 corpus rec (for cross-ref)
        "honesty": {
            "completeness_neq_correctness": (
                "metadata_score is how much metadata is PRESENT — it does NOT verify "
                "the recorded VALUES are correct (a present pH may still be wrong)."),
            "information_gain_upper_bound": (
                "M11 information_gain numbers are STRUCTURAL UPPER BOUNDS (tier lifts "
                "under the DAG, inheriting M8's validity ceiling) EXCEPT the Service-C-"
                "measured ones (gain_basis='service_c_measured' — the γ 1->4 split)."),
        },
        "source": "data/processed/metadata_quality.json (M11 corpus rollup)",
    }


def _attach_metadata_explanation(rmap: dict, m11: dict | None) -> dict:
    """Cross-reference M11 onto the EXISTING M8 blocks in-place (additive only):
      - top-level `metadata_explanation` block;
      - `metadata_completeness_summary` (median metadata_score + mechanistic_completeness);
      - `why_blocked` + universal-blocker fields on the resolvable BOUNDARY;
      - per-blocker M11 field(s) + information-gain-to-unblock on the blocker_census.
    Never raises; if M11 is absent, only the {m11_available: False} stub is added."""
    explanation = build_metadata_explanation(m11)
    rmap["metadata_explanation"] = explanation

    corpus = (m11 or {}).get("corpus", {}) if isinstance(m11, dict) else {}
    # ---- top-level completeness summary (median score + mechanistic_completeness) --
    if explanation.get("m11_available"):
        md = (corpus.get("metadata_score", {}) or {}).get("distribution", {}) or {}
        mc = (corpus.get("mechanistic_completeness", {}) or {}).get("distribution", {}) or {}
        rmap["metadata_completeness_summary"] = {
            "m11_available": True,
            "median_metadata_score": md.get("median"),
            "mean_metadata_score": md.get("mean"),
            "median_mechanistic_completeness": mc.get("median"),
            "mean_mechanistic_completeness": mc.get("mean"),
            "note": ("median metadata_score is high but median mechanistic_completeness "
                     "is low — the corpus records the CONDITIONS but not the mechanism-"
                     "gating {agitation, seeded}, which is exactly why the M8 boundary "
                     "GAP is unresolved. completeness ≠ correctness."),
            "source": "M11 corpus rollup (metadata_quality.json)",
        }
    else:
        rmap["metadata_completeness_summary"] = {
            "m11_available": False,
            "note": explanation.get("note"),
        }

    # ---- attach WHY onto the mechanistically-resolvable BOUNDARY -------------- #
    boundary = rmap.get("mechanistically_resolvable_boundary")
    if isinstance(boundary, dict):
        if explanation.get("m11_available"):
            gap = boundary.get("gap_structural_minus_licensed")
            boundary["why_gap_unresolved"] = {
                "m11_available": True,
                "cause": ("the structural GAP is closed by recording the 2 mechanism-"
                          "gating fields {agitation, seeded}: they are 'unknown' on "
                          f"~{pct_str(explanation['universal_blocker'].get('fraction'))} "
                          "of curves corpus-wide, so the M5 quiescent-vs-shaken / "
                          "primary-vs-secondary branch is undetermined for EVERY curve — "
                          "the mechanism gate refuses regardless of how complete the "
                          "REST of the metadata is (median metadata_score is high). "
                          f"Recording them closes the {gap}-protein gap; M11/M9 rank "
                          "this #1."),
                "universal_blocker_fields": MECH_BRANCH_FIELDS,
                "why_blocked": explanation.get("why_blocked", []),
                "completeness_neq_correctness": explanation["honesty"][
                    "completeness_neq_correctness"],
                "information_gain_upper_bound": explanation["honesty"][
                    "information_gain_upper_bound"],
            }
        else:
            boundary["why_gap_unresolved"] = {
                "m11_available": False,
                "note": explanation.get("note"),
            }

    # ---- cross-reference each blocker to the M11 field(s) + gain-to-unblock --- #
    census = rmap.get("blocker_census")
    if isinstance(census, dict) and explanation.get("m11_available"):
        action_idx = explanation.get("action_index", {})
        for b in census.get("blockers_impact_ranked", []) or []:
            mapping = _M8_BLOCKER_TO_M11.get(b.get("blocker"))
            if not mapping:
                continue
            act = mapping.get("m11_action")
            m11_rec = action_idx.get(act) if act else None
            b["m11_explanation"] = {
                "m11_fields_responsible": mapping.get("m11_fields", []),
                "m11_action_to_unblock": act,
                "m11_information_gain_over_effort": (
                    (m11_rec or {}).get("total_gain_over_effort")),
                "m11_impact": (m11_rec or {}).get("impact"),
                "m11_effort": (m11_rec or {}).get("effort"),
                "gain_basis": mapping.get("gain_basis"),
                "note": ("this blocker is EXPLAINED by the M11 field(s) above; "
                         "recording them supplies the information_gain to unblock "
                         "(gain_basis carries the honesty: service_c_measured is a "
                         "measured γ-split, structural_estimate is a DAG upper bound)."
                         if mapping.get("m11_fields") else
                         "censoring is a DATA-window limit, not a recordable metadata "
                         "field — no M11 metadata action unblocks it."),
            }
    return rmap


def pct_str(frac) -> str:
    """'100%' style formatting for a fraction; '?' if None (never raises)."""
    try:
        return f"{100.0 * float(frac):.0f}%"
    except Exception:
        return "?"


# ======================================================================== #
# top-level builder
# ======================================================================== #
def build_reachability(artifacts: dict) -> dict:
    """Consolidate M1–M7 + Service-C artifacts into the reachability map.

    `artifacts` keys (all optional; missing -> graceful degradation, no raise):
        curves_triaged : list[dict]   per-curve M1 triage + condition_vector
        m7_rollup      : dict         the corpus rollup (by_information_yield, ...)
        etl_manifest   : dict         the ETL census (counts)
        service_c      : dict         service_c_calibration.json
        m11_metadata   : dict         metadata_quality.json (M11 corpus rollup) — OPTIONAL;
                                      when present, M8 attaches a `metadata_explanation`
                                      block that makes the reachability CAUSAL (WHY the
                                      GAP is unresolved). Absent -> {m11_available: False}
                                      and today's behavior is preserved unchanged.
    """
    curves = artifacts.get("curves_triaged") or []
    rollup = artifacts.get("m7_rollup") or {}
    manifest = artifacts.get("etl_manifest") or {}
    service_c = artifacts.get("service_c")  # may be None
    m11 = artifacts.get("m11_metadata")     # may be None (graceful)

    degraded = []
    if not curves:
        degraded.append("curves_triaged missing/empty: grid + blockers degraded")
    if not rollup:
        degraded.append("m7_rollup missing: ladder + boundary use ETL/M1 fallback only")
    if not manifest:
        degraded.append("etl_manifest missing: census falls back to observed counts")
    if not service_c:
        degraded.append("service_c missing: validity-ceiling + boundary evidence degraded")
    if not m11 or "corpus" not in (m11 or {}):
        degraded.append("m11_metadata missing: metadata_explanation degraded "
                        "({m11_available: false}) — reachability unchanged")

    # ---- derive per-curve censuses (single pass) --------------------------- #
    series_sizes: Counter = Counter()
    for cu in curves:
        cs = cu.get("concentration_series_id")
        if cs:
            series_sizes[cs] += 1

    fittability: Counter = Counter()
    censoring: Counter = Counter()
    n_signal_curves = 0
    proteins_all: set = set()
    proteins_by_rung: dict = {r: set() for r in LADDER}
    for cu in curves:
        m1 = cu.get("m1", {}) or {}
        fc = m1.get("fittability_class")
        fittability[fc] += 1
        censoring[m1.get("censoring_class")] += 1
        # signal present = not flat_no_signal (a curve with usable signal)
        if not m1.get("flat_no_signal", False):
            n_signal_curves += 1
        pid = cu.get("uniprot_id") or cu.get("protein_id")
        if pid is not None:
            proteins_all.add(pid)

    # totals: prefer the ETL manifest census (authoritative), fall back to observed
    counts = (manifest or {}).get("counts", {}) or {}
    n_curves = int(counts.get("curves", 0)) or len(curves) or int(rollup.get("n_results", 0))
    n_proteins = int(counts.get("distinct_proteins", 0)) or len(proteins_all)

    # protein-level reach: a protein reaches a rung if ANY of its curves' yield does.
    # We approximate per-curve yield from the rollup-consistent ladder using the
    # M7 per-curve artifact when available; otherwise leave proteins off the rung.
    per_curve_yield = artifacts.get("per_curve_yield")  # optional {series_id: yield}
    proteins_by_rung_counts = None
    if per_curve_yield:
        rung_index = {y: i for i, y in enumerate(
            ["uninformative", "signal_only", "descriptive", "scaling", "mechanistic"])}
        # map M7 yield names onto ladder rungs
        yield_to_rung = {
            "uninformative": "signal_present",
            "signal_only": "signal_present",
            "descriptive": "descriptive_regime",
            "scaling": "scaling",
            "mechanistic": "mechanistic_licensed",
        }
        prot_best: dict = {}
        for cu in curves:
            sid = cu.get("series_id")
            y = per_curve_yield.get(sid)
            if y is None:
                continue
            pid = cu.get("uniprot_id") or cu.get("protein_id")
            prot_best[pid] = max(prot_best.get(pid, -1), rung_index.get(y, 0))
        # cumulative: a protein at level L reaches all rungs <= its mapped rung
        ladder_rank = {r: i for i, r in enumerate(LADDER)}
        counts_by_rung = {r: 0 for r in LADDER}
        for pid, best in prot_best.items():
            best_yield = [k for k, v in rung_index.items() if v == best]
            best_rung = yield_to_rung.get(best_yield[0], "signal_present") if best_yield else "signal_present"
            if best == 0:  # uninformative -> reaches signal_present only if signal
                best_rung = "signal_present"
            for r in LADDER:
                if ladder_rank[r] <= ladder_rank[best_rung]:
                    counts_by_rung[r] += 1
        proteins_by_rung_counts = counts_by_rung

    ladder = build_ladder(rollup, n_curves, n_proteins, fittability,
                          n_signal_curves, proteins_by_rung_counts)
    grid = build_grid(curves, series_sizes)
    grid["partition_ok"] = (grid["curves_partitioned"] == len(curves))
    blockers = build_blocker_census(curves, series_sizes, censoring)
    boundary = build_resolvable_boundary(curves, series_sizes, manifest, rollup, service_c)
    ceiling = build_validity_ceiling(service_c)

    # ---- versioning (§7) --------------------------------------------------- #
    engine_build_id = (rollup or {}).get("engine_build_id")
    components = (rollup or {}).get("engine_build_components", {}) or {}
    versioning = {
        "m8_version": M8_VERSION,
        "engine_build_id": engine_build_id,
        "engine_build_components": components,
        "service_c_version": (service_c or {}).get("service_c_version"),
        "invalidation_rule": "this map is INVALIDATED and rebuilt when any of "
                             "{generator/harness, taxonomy, anchor, corpus} version "
                             "changes (§7 versioning DAG). Comparability is gated on "
                             "equal engine_build_id.",
    }

    rmap = {
        "m8_version": M8_VERSION,
        "title": "PRISE Inferential Reachability Map — reachability UNDER the PRISE forward model",
        "validity_ceiling": ceiling,          # MANDATORY, kept prominent (top)
        "inference_ladder": ladder,
        "reachability_grid": grid,
        "blocker_census": blockers,
        "mechanistically_resolvable_boundary": boundary,
        "versioning": versioning,
        "degraded": degraded,
        "is_degraded": bool(degraded),
        "conditioning_clause": "EVERY headline in this map is reachability UNDER "
                               "THE PRISE FORWARD MODEL (see validity_ceiling).",
    }
    # ADDITIVE: consume the M11 metadata-quality artifact to make the reachability
    # CAUSAL (attaches `metadata_explanation`, `metadata_completeness_summary`, the
    # `why_gap_unresolved` on the boundary, and the per-blocker M11 cross-reference).
    # Degrades gracefully to {m11_available: False} when M11 is absent; every
    # existing M8 key + the validity-ceiling honesty is preserved untouched.
    rmap = _attach_metadata_explanation(rmap, m11)
    return rmap


# ======================================================================== #
# query helper (§5-M8 query interface)
# ======================================================================== #
def reachable_at(reachability_map: dict, level: str) -> dict:
    """Query the assembled map: count + fraction of curves reaching `level`."""
    ladder = (reachability_map or {}).get("inference_ladder", {}) or {}
    for r in ladder.get("rungs", []):
        if r.get("rung") == level:
            out = {"level": level, "curves": r.get("curves"),
                   "curve_fraction": r.get("curve_fraction"),
                   "statement": r.get("curve_statement")}
            if "proteins" in r:
                out["proteins"] = r["proteins"]
                out["protein_fraction"] = r.get("protein_fraction")
            return out
    return {"level": level, "error": "unknown level",
            "valid_levels": LADDER}


# ======================================================================== #
# CLI
# ======================================================================== #
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


def _load_per_curve_yield(path: Path) -> dict:
    """Optional: per-curve information_yield from the M7 protein_analysis artifact."""
    out = {}
    for r in _read_jsonl(path):
        sid = r.get("series_id") or r.get("analysis_unit")
        y = r.get("information_yield")
        if sid and y:
            out[sid] = y
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PRISE M8 — Inferential Reachability Map")
    # repo-root-anchored default (Path(__file__)…), matching M1–M7 + Service C, so
    # the build is invariant to the cwd it is launched from (a bare relative path
    # crashes when run from engine/).
    proc_default = Path(__file__).resolve().parent.parent / "data" / "processed"
    ap.add_argument("--input", type=Path, default=proc_default,
                    help="directory holding the input artifacts")
    ap.add_argument("--output", type=Path, default=None,
                    help="output JSON path (default: <input>/m8_reachability.json)")
    args = ap.parse_args(argv)

    indir = Path(args.input)
    artifacts = {
        "curves_triaged": _read_jsonl(indir / "curves_triaged.jsonl"),
        "m7_rollup": _read_json(indir / "m7_rollup.json") or {},
        "etl_manifest": _read_json(indir / "etl_manifest.json") or {},
        "service_c": _read_json(indir / "service_c_calibration.json"),
        "per_curve_yield": _load_per_curve_yield(indir / "protein_analysis.jsonl"),
        # M11 metadata-quality corpus rollup — OPTIONAL; makes the reachability
        # CAUSAL (metadata_explanation). Absent -> graceful {m11_available: false}.
        "m11_metadata": _read_json(indir / "metadata_quality.json"),
    }
    rmap = build_reachability(artifacts)

    out_path = Path(args.output) if args.output else (indir / "m8_reachability.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rmap, indent=2), encoding="utf-8")

    # ---- printed summary --------------------------------------------------- #
    ladder = rmap["inference_ladder"]
    boundary = rmap["mechanistically_resolvable_boundary"]
    print(f"[M8] wrote {out_path}")
    print(f"[M8] {rmap['title']}")
    if rmap["is_degraded"]:
        print(f"[M8] DEGRADED: {rmap['degraded']}")
    print("\n  INFERENCE LADDER (reachability UNDER the PRISE forward model):")
    for r in ladder["rungs"]:
        ptxt = (f"  | proteins {r['proteins']}/{ladder['n_proteins']}"
                if "proteins" in r else "")
        print(f"    {r['rung']:<22} {r['curves']:>5} / {ladder['n_curves']} "
              f"({_pct(r['curves'], ladder['n_curves'])}){ptxt}")
    print(f"    monotone_non_increasing = {ladder['monotone_non_increasing']}")

    grid = rmap["reachability_grid"]
    print(f"\n  REACHABILITY GRID ({len(DATA_RICHNESS)}x{len(COMPLETENESS_BINS)}; "
          f"partition_ok={grid['partition_ok']}):")
    for key, cell in grid["cells"].items():
        if cell["count"]:
            print(f"    {key:<34} n={cell['count']:>4}  "
                  f"status={cell['inferential_status']}")
    print(f"    curves by status: {grid['n_curves_by_status']}")

    print("\n  BLOCKER CENSUS (impact-ranked):")
    for b in rmap["blocker_census"]["blockers_impact_ranked"]:
        print(f"    #{b['impact_rank']} {b['blocker']:<28} "
              f"n={b['count']:>5} ({_pct(b['count'], ladder['n_curves'])}) "
              f"-> blocks {b['blocks_rung']} [{b['scope']}]")

    print("\n  MECHANISTICALLY-RESOLVABLE BOUNDARY:")
    print(f"    structural-reachable (upper bound): "
          f"{boundary['structural_reachable_proteins_upper_bound']} proteins / "
          f"{boundary['structural_reachable_series_upper_bound']} series")
    print(f"    actually licensed:                  "
          f"{boundary['actual_mechanistically_licensed']}")
    print(f"    GAP (structural - licensed):        "
          f"{boundary['gap_structural_minus_licensed']}")
    print(f"    {boundary['headline']}")

    me = rmap.get("metadata_explanation", {})
    print("\n  METADATA EXPLANATION (M11 — WHY the gap is unresolved):")
    if me.get("m11_available"):
        mcs = rmap.get("metadata_completeness_summary", {})
        print(f"    m11={me.get('m11_version')}  median metadata_score="
              f"{mcs.get('median_metadata_score')}  median mechanistic_completeness="
              f"{mcs.get('median_mechanistic_completeness')}")
        ub = me.get("universal_blocker", {})
        print(f"    universal blocker {ub.get('fields')} unknown on "
              f"{pct_str(ub.get('fraction'))} of curves; MECHANISM licensed "
              f"{ub.get('n_mechanism_licensed')} / {me.get('n_curves')}")
        wg = (rmap.get("mechanistically_resolvable_boundary", {})
              .get("why_gap_unresolved", {}))
        print(f"    WHY GAP: {(wg.get('cause') or '')[:220]}...")
        for w in me.get("why_blocked", [])[:3]:
            print(f"      #{w.get('rank')} {w.get('m11_action'):<32} "
                  f"impact={w.get('impact'):<9} gain/effort="
                  f"{w.get('information_gain_over_effort')} [{w.get('gain_basis')}]")
    else:
        print(f"    {me.get('note')}")

    print("\n  VALIDITY CEILING (mandatory):")
    vc = rmap["validity_ceiling"]
    print(f"    {vc['clause']}")
    deg = vc["evidence_a_different_forward_model_degrades_resolution"][
        "mechanism_misidentification_degradation"]
    print(f"    evidence: mismatched-generator mechanism-misID degradation = {deg}")
    print(f"    deferred external validation: {vc['deferred_external_validation']['status']}")

    print(f"\n  VERSIONING: m8={rmap['versioning']['m8_version']} "
          f"engine_build_id={rmap['versioning']['engine_build_id']} "
          f"service_c={rmap['versioning']['service_c_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
