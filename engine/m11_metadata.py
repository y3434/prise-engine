"""
PRISE — Module M11: Metadata Quality & Information Completeness Engine
=====================================================================

Implements PRISE_DESIGN.md §3-M11. M11 FORMALIZES and QUANTIFIES the metadata
layer the PRISE engine ALREADY licenses inference against — it does NOT invent a
parallel licensing logic. Every gate M11 reasons about is a MIRROR of the REAL
mechanistic gates in M5 (`m5_classify.mechanistic_assessment`) and the
reachability structure in M8, read from the SAME ground-truth `condition_vector`
+ `field_provenance` that M1 builds. M11 is standalone: it READS artifacts and
never mutates M1/M5/M8.

What M11 answers, per triaged series:
  1. **METADATA_ONTOLOGY** (`m11-ontology-1.0`) — each metadata field is a node
     with a category, an extractor, the inference gates it licenses, a
     (CONFIGURABLE, VERSIONED) completeness_weight + inferential_weight, and a
     required/recommended/optional tier. Mechanism-gating fields (assay/dye,
     agitation, seeding, concentration_series) are weighted HIGH; buffer /
     reducing_agent / organism / publication LOW.
  2. **score_completeness** — metadata_score = Σ w·present(f) / Σ w over the
     CAPTURABLE fields (fields NOT_CAPTURED_BY_THE_SOURCE_SCHEMA are excluded
     from the denominator but reported). present(f) ∈ {1.0, 0.5 (inferred/
     uncertain provenance), 0.0}. Weights are overridable and MUST be honored.
  3. **metadata_uncertainty** — a SEPARATE [0,1] trust scalar (0 = trustworthy):
     completeness = how much is there; uncertainty = how much to trust it.
  4. **delta_inferential_power** — per missing/unknown field: which inference
     tiers it blocks, its impact, a plain-English consequence chain, and the
     tier-lift it would enable (labelled `service_c_measured` where anchored to
     the Service-C 1→4 Tier-B split, else `structural_estimate`).
  5. **dependency_graph / evaluate_targets** — a DAG DERIVED FROM M5's gates,
     with mechanistic_completeness ∈ [0,1] = fraction of MECHANISM prerequisites
     satisfied + the explicit blocker list.
  6. **recommended_missing_metadata** — missing fields ranked by
     information_gain ÷ effort, in M9's action language.
  7. **assess_metadata** — the full per-dataset record. NEVER raises.

=== HONESTY (mandatory, PRISE ethos) ===
  * COMPLETENESS ≠ CORRECTNESS. metadata_score measures how much metadata is
    PRESENT; it does NOT verify the VALUES. A present pH may still be wrong —
    metadata_uncertainty flags trust but CANNOT verify a recorded value.
  * information-gain / Δ-power numbers are STRUCTURAL UPPER BOUNDS (tier lifts
    under the DAG), inheriting M8's validity ceiling — EXCEPT the Service-C
    measured ones (`gain_basis='service_c_measured'`), the γ-resolved 1→4 split.
  * the default weights are a VERSIONED MODELING CHOICE (`m11-ontology-1.0`),
    CONFIGURABLE, never "objective truth".
  * `not_captured_by_source` is a DATABASE-SCHEMA limitation (the field is not a
    first-class CPAD field), NOT the dataset's fault — kept distinct throughout.

All outputs are pure, deterministic, JSON-serialisable. Nothing here raises on
bad input: a malformed/empty series yields a flagged minimal result.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

M11_VERSION = "m11-metadata-1.0"
ONTOLOGY_VERSION = "m11-ontology-1.0"

# The EXACT M5 gate strings M11 mirrors (kept as literals so a divergence from
# m5_classify is a visible, greppable inconsistency — M11 must stay consistent
# with the real gates, not re-derive them).
M5_GATE_ASSAY_NOT_MASS = "assay_not_mass_proportional"
M5_GATE_ASSAY_UNKNOWN = "assay_mass_proportionality_unknown"
M5_GATE_AGIT_SEED = "agitation_or_seeding_unknown_branch_undetermined"
M5_GATE_NO_SHAPE = "no_nucleated_transition_in_shape"

# Tokens that mark a reducing agent inside the free-text `additives` field. WHY a
# scan: reducing_agent is NOT a first-class CPAD field — it is only recoverable
# by string-matching additives, so it is CAPTURABLE (not not_captured) but weak.
_REDUCING_AGENT_RE = re.compile(r"\b(DTT|TCEP|BME|2-?ME|beta-?mercaptoethanol|"
                                r"2-mercaptoethanol|dithiothreitol|"
                                r"tris\(2-carboxyethyl\)phosphine)\b", re.I)

# ---------------------------------------------------------------------------- #
# Inference tiers a metadata field can license. These name the DAG targets, not
# a mechanism — M11 quantifies the METADATA layer, the engine does the inference.
#   identity      — which protein/construct this even is
#   comparability — can this curve be compared across conditions (M8 axis B)
#   scaling       — γ scaling exponent resolvable (needs a concentration series)
#   mechanism     — a single Tier-B mechanism licensable (the M5 hard gate)
#   uncertainty   — supports the trust/uncertainty accounting only
#   none          — bookkeeping field, licenses no inference tier
# ---------------------------------------------------------------------------- #
TIERS = ("identity", "comparability", "scaling", "mechanism", "uncertainty", "none")


# ============================ METADATA ONTOLOGY ============================= #
# Each node: category, extractor (plain-English how it is read from the series),
# gates (tiers it licenses), completeness_weight (CONFIGURABLE default),
# inferential_weight, tier (required|recommended|optional). Mechanism-gating
# fields are weighted HIGH; buffer/reducing_agent/organism/publication LOW.
#
# `not_first_class` marks fields that are NOT native CPAD schema fields; they are
# mapped from what IS captured (see extractor). If that mapping yields nothing,
# the field is reported `not_captured_by_source` and EXCLUDED from the score
# denominator — a schema limitation, not the dataset's fault.
METADATA_ONTOLOGY = {
    "version": ONTOLOGY_VERSION,
    "weights_are": ("a VERSIONED modeling choice — CONFIGURABLE via the `weights` "
                    "argument; NOT objective truth"),
    "completeness_neq_correctness": (
        "these weights score how much metadata is PRESENT, never whether a "
        "recorded value is CORRECT"),
    "fields": {
        # ----- identity -------------------------------------------------- #
        "protein": {
            "category": "identity", "tier": "required",
            "extractor": "series.protein_id",
            "gates": ["identity"],
            "completeness_weight": 3.0, "inferential_weight": 3.0},
        "uniprot": {
            "category": "identity", "tier": "required",
            "extractor": "series.uniprot_id",
            "gates": ["identity", "comparability"],
            "completeness_weight": 2.0, "inferential_weight": 2.0},
        "mutation": {
            "category": "identity", "tier": "recommended",
            "extractor": "condition_vector.construct_id (WT vs mutant)",
            "gates": ["identity", "comparability"],
            "completeness_weight": 2.0, "inferential_weight": 2.0,
            "not_first_class": "mapped <- construct_id"},
        "construct": {
            "category": "identity", "tier": "recommended",
            "extractor": "condition_vector.construct_id",
            "gates": ["identity", "comparability"],
            "completeness_weight": 2.0, "inferential_weight": 2.0},
        # ----- assay (mechanism-gating: HIGH) ---------------------------- #
        "assay": {
            "category": "assay", "tier": "required",
            "extractor": "condition_vector.assay_type + assay_reports_mass",
            "gates": ["comparability", "scaling", "mechanism"],
            "completeness_weight": 4.0, "inferential_weight": 4.0},
        "dye": {
            "category": "assay", "tier": "recommended",
            "extractor": "condition_vector.assay_type (+ tht_concentration)",
            "gates": ["comparability", "mechanism"],
            "completeness_weight": 3.0, "inferential_weight": 3.0,
            "not_first_class": "mapped <- assay_type + tht_concentration"},
        # ----- condition (comparability) --------------------------------- #
        "temperature": {
            "category": "condition", "tier": "required",
            "extractor": "condition_vector.temperature_C",
            "gates": ["comparability"],
            "completeness_weight": 2.5, "inferential_weight": 2.5},
        "pH": {
            "category": "condition", "tier": "required",
            "extractor": "condition_vector.pH",
            "gates": ["comparability"],
            "completeness_weight": 2.5, "inferential_weight": 2.5},
        "ionic_strength": {
            "category": "condition", "tier": "recommended",
            "extractor": "condition_vector.ion_concentration / ionic_strength_note",
            "gates": ["comparability"],
            "completeness_weight": 1.5, "inferential_weight": 1.5},
        # ----- condition (mechanism-gating: HIGH) ------------------------ #
        "agitation": {
            "category": "condition", "tier": "required",
            "extractor": "condition_vector.agitation (provenance-gated)",
            "gates": ["mechanism"],
            "completeness_weight": 4.0, "inferential_weight": 4.0},
        "seeding": {
            "category": "condition", "tier": "required",
            "extractor": "condition_vector.seeded (provenance-gated)",
            "gates": ["mechanism"],
            "completeness_weight": 4.0, "inferential_weight": 4.0},
        "concentration": {
            "category": "condition", "tier": "required",
            "extractor": "condition_vector.concentration.value_uM",
            "gates": ["comparability", "scaling"],
            "completeness_weight": 2.5, "inferential_weight": 2.5},
        # ----- statistical (scaling/mechanism-gating: HIGH) -------------- #
        "concentration_series": {
            "category": "statistical", "tier": "required",
            "extractor": "series.concentration_series_id (>=3 members)",
            "gates": ["scaling", "mechanism"],
            "completeness_weight": 4.0, "inferential_weight": 4.0,
            "needs_series_index": True},
        "replicate_count": {
            "category": "statistical", "tier": "recommended",
            "extractor": "count of same-condition curves for this protein",
            "gates": ["uncertainty"],
            "completeness_weight": 1.0, "inferential_weight": 1.0,
            "not_first_class": "derived <- count of same-condition curves",
            "needs_series_index": True},
        # ----- provenance ------------------------------------------------ #
        "digitization_confidence": {
            "category": "provenance", "tier": "optional",
            "extractor": "series.digitization_uncertainty / m1 provenance "
                         "(ALL CPAD is digitized)",
            "gates": ["uncertainty"],
            "completeness_weight": 1.0, "inferential_weight": 1.0},
        # ----- bibliographic / low-weight -------------------------------- #
        "publication": {
            "category": "bibliographic", "tier": "optional",
            "extractor": "series.source_study (pmid/reference) if present",
            "gates": ["none"],
            "completeness_weight": 0.5, "inferential_weight": 0.5,
            "not_first_class": "source ref if present else not_captured"},
        "organism": {
            "category": "identity", "tier": "optional",
            "extractor": "derivable from uniprot; not a first-class CPAD field",
            "gates": ["identity"],
            "completeness_weight": 0.5, "inferential_weight": 0.5,
            "not_first_class": "derivable from uniprot else not_captured"},
        "buffer": {
            "category": "condition", "tier": "optional",
            "extractor": "condition_vector.buffer",
            "gates": ["comparability"],
            "completeness_weight": 0.5, "inferential_weight": 0.5},
        "reducing_agent": {
            "category": "condition", "tier": "optional",
            "extractor": "scan condition_vector.additives for DTT/TCEP/BME/2-ME",
            "gates": ["comparability"],
            "completeness_weight": 0.5, "inferential_weight": 0.5,
            "not_first_class": "scanned <- additives (no first-class field)"},
    },
}

# The mechanism prerequisites, mirroring the M5 gate set exactly (§3-M5). These
# are the fields whose satisfaction M5 requires before it licenses a mechanism.
MECHANISM_PREREQS = ["assay", "agitation", "seeding", "concentration_series",
                     "cooperative_shape"]


# ============================ present() logic ============================== #
def _series_index(series: dict) -> dict:
    """Optional per-corpus context for series-membership + replicate counting.

    The driver passes {series_sizes: {cs_id: n}, replicate_counts: {series_id: n}}
    so `concentration_series`/`replicate_count` can be scored. When absent (unit
    tests / a lone series) we fall back to the series' own fields and SAY SO."""
    return series.get("_m11_index", {}) or {}


def _prov(cv: dict, key: str) -> str | None:
    return (cv.get("field_provenance", {}) or {}).get(key)


def _extract_field(field: str, series: dict, cv: dict, idx: dict) -> dict:
    """Return {status, value, provenance} for ONE ontology field.

    status ∈ {"present","inferred","missing","not_captured_by_source"}.
      present               -> present(f)=1.0
      inferred              -> present(f)=0.5 (present but provenance uncertain)
      missing               -> present(f)=0.0 (CAPTURABLE but absent/unknown)
      not_captured_by_source-> EXCLUDED from the denominator (schema limitation)
    """
    fp_prov = None

    if field == "protein":
        v = series.get("protein_id")
        return _pv(v is not None and str(v).strip() != "", v, "source")
    if field == "uniprot":
        v = series.get("uniprot_id")
        return _pv(bool(v), v, "source")
    if field in ("mutation", "construct"):
        v = cv.get("construct_id")
        return _pv(v is not None and str(v).strip() != "", v, "source")
    if field == "assay":
        at = cv.get("assay_type")
        amp = cv.get("assay_reports_mass")
        prov = _prov(cv, "assay_type")
        if at is None:
            return _pv(False, None, prov)
        # present, but if mass-proportionality is UNKNOWN the assay is only
        # half-informative for the mechanism gate -> provenance 'inferred'.
        if amp is None:
            return {"status": "inferred", "value": {"assay_type": at,
                    "assay_reports_mass": None}, "provenance": "mass_unknown"}
        return _pv(True, {"assay_type": at, "assay_reports_mass": amp}, prov)
    if field == "dye":
        at = cv.get("assay_type")
        return _pv(at is not None, {"assay_type": at,
                   "tht_concentration": cv.get("tht_concentration")},
                   _prov(cv, "assay_type"))
    if field == "temperature":
        return _pv(cv.get("temperature_C") is not None, cv.get("temperature_C"),
                   _prov(cv, "temperature"))
    if field == "pH":
        return _pv(cv.get("pH") is not None, cv.get("pH"), _prov(cv, "pH"))
    if field == "ionic_strength":
        v = cv.get("ion_concentration") or cv.get("ionic_strength_note")
        return _pv(v is not None, v, _prov(cv, "ionic_strength"))
    if field == "concentration":
        c = cv.get("concentration", {}) or {}
        v = c.get("value_uM") if isinstance(c, dict) else None
        return _pv(v is not None, v, _prov(cv, "concentration"))
    if field in ("agitation", "seeding"):
        # provenance-gated EXACTLY like the M5 gate: known-and-present, else missing.
        cvkey = "agitation" if field == "agitation" else "seeded"
        provkey = "agitation" if field == "agitation" else "seeded"
        prov = _prov(cv, provkey)
        val = cv.get(cvkey)
        present = prov == "known" and val is not None
        return _pv(present, val, prov)
    if field == "concentration_series":
        cs = series.get("concentration_series_id")
        size = (idx.get("series_sizes", {}) or {}).get(cs)
        if size is None and cs:
            # no corpus index -> we cannot confirm >=3; report what we know.
            return {"status": "inferred", "value": {"concentration_series_id": cs,
                    "size": "unknown_no_index"}, "provenance": "no_series_index"}
        present = bool(cs) and (size or 0) >= 3
        return _pv(present, {"concentration_series_id": cs, "size": size}, "source")
    if field == "replicate_count":
        rc = (idx.get("replicate_counts", {}) or {}).get(series.get("series_id"))
        if rc is None:
            return {"status": "inferred", "value": None,
                    "provenance": "no_replicate_index"}
        return _pv(rc >= 2, rc, "derived")
    if field == "digitization_confidence":
        # ALL CPAD is digitized -> the confidence is ALWAYS available (as a
        # provenance flag), but it is inherently 'inferred' (digitization noise).
        du = series.get("digitization_uncertainty", True)
        return {"status": "inferred", "value": {"digitization_uncertainty": du},
                "provenance": "all_cpad_digitized"}
    if field == "publication":
        ss = series.get("source_study") or {}
        v = ss.get("pmid") or ss.get("reference") or series.get("source")
        if v:
            return _pv(True, v, "source")
        return {"status": "not_captured_by_source", "value": None,
                "provenance": "not_captured"}
    if field == "organism":
        # NOT a first-class CPAD field. We can only mark it not_captured unless a
        # uniprot is present (from which organism is derivable in principle).
        up = series.get("uniprot_id")
        if up:
            return {"status": "inferred", "value": {"derivable_from": up},
                    "provenance": "derivable_from_uniprot"}
        return {"status": "not_captured_by_source", "value": None,
                "provenance": "not_captured"}
    if field == "buffer":
        return _pv(cv.get("buffer") is not None, cv.get("buffer"), "source")
    if field == "reducing_agent":
        add = cv.get("additives")
        if add and _REDUCING_AGENT_RE.search(str(add)):
            m = _REDUCING_AGENT_RE.search(str(add))
            return _pv(True, m.group(0), "scanned_from_additives")
        # NOT a first-class field: absence of a token is NOT the dataset's fault.
        return {"status": "not_captured_by_source", "value": None,
                "provenance": "not_captured_no_first_class_field"}
    # unknown field name -> treat as missing (defensive; never raises)
    return _pv(False, None, fp_prov)


def _pv(present: bool, value, provenance) -> dict:
    """Map a boolean-present + value + provenance into the extractor result.

    Provenance strings that mean 'present but uncertain' downgrade to 'inferred'
    (present(f)=0.5)."""
    if not present:
        return {"status": "missing", "value": None, "provenance": provenance}
    if provenance in ("inferred", "uncertain", "mass_unknown", "no_series_index"):
        return {"status": "inferred", "value": value, "provenance": provenance}
    return {"status": "present", "value": value, "provenance": provenance}


_PRESENT_SCORE = {"present": 1.0, "inferred": 0.5, "missing": 0.0}


# ============================ 2. COMPLETENESS ============================== #
def score_completeness(series: dict, weights: dict | None = None) -> dict:
    """metadata_score = Σ w·present(f) / Σ w over CAPTURABLE fields ∈ [0,1].

    `not_captured_by_source` fields are EXCLUDED from the denominator (a schema
    limitation, reported separately). `weights` overrides the ontology defaults
    per-field and MUST be honored. Returns the score + per-field detail. Never
    raises."""
    if not isinstance(series, dict):
        series = {}
    cv = series.get("condition_vector", {}) or {}
    idx = _series_index(series)
    w_over = weights or {}

    per_field = {}
    num = 0.0
    den = 0.0
    not_captured = []
    for field, node in METADATA_ONTOLOGY["fields"].items():
        try:
            res = _extract_field(field, series, cv, idx)
        except Exception:  # defensive: a malformed field never crashes the score
            res = {"status": "missing", "value": None, "provenance": "extractor_error"}
        w = float(w_over.get(field, node["completeness_weight"]))
        status = res["status"]
        if status == "not_captured_by_source":
            not_captured.append(field)
            per_field[field] = {
                "present": None, "status": status, "value_seen": res.get("value"),
                "provenance": res.get("provenance"), "weight": w,
                "note": "not_captured_by_source: schema limitation, NOT counted "
                        "in the denominator (not the dataset's fault)"}
            continue
        p = _PRESENT_SCORE[status]
        num += w * p
        den += w
        per_field[field] = {
            "present": p, "status": status, "value_seen": res.get("value"),
            "provenance": res.get("provenance"), "weight": w,
            "category": node["category"], "tier": node["tier"]}

    score = round(num / den, 6) if den > 0 else 0.0
    return {
        "metadata_score": score,
        "numerator": round(num, 6),
        "denominator": round(den, 6),
        "n_capturable_fields": sum(1 for f in per_field.values()
                                   if f["present"] is not None),
        "not_captured_by_source": not_captured,
        "per_field": per_field,
        "weights_source": ("overridden" if weights else "ontology_default"),
        "ontology_version": ONTOLOGY_VERSION,
        "completeness_neq_correctness": (
            "this is how much metadata is PRESENT, not whether values are correct"),
    }


# ============================ 3. UNCERTAINTY =============================== #
def metadata_uncertainty(series: dict) -> dict:
    """A SEPARATE [0,1] trust scalar (0 = trustworthy). Reported apart from
    completeness: completeness = how much is there; uncertainty = how much to
    trust it. Combines (a) a digitization baseline (ALL CPAD is digitized), (b)
    the fraction of PRESENT fields whose provenance is uncertain/inferred, and
    (c) the unknown-fraction over capturable fields. Never verifies a value —
    completeness ≠ correctness, and uncertainty flags trust but cannot confirm."""
    if not isinstance(series, dict):
        series = {}
    comp = score_completeness(series)
    per = comp["per_field"]

    captured = [f for f in per.values() if f["present"] is not None]
    n = len(captured)
    present_fields = [f for f in captured if f["status"] in ("present", "inferred")]
    n_present = len(present_fields)

    # (a) digitization baseline — CPAD is figure-digitized, so even a fully
    # populated record carries irreducible read-off uncertainty.
    du = bool(series.get("digitization_uncertainty", True))
    digitization_baseline = 0.15 if du else 0.05

    # (b) fraction of PRESENT fields carrying uncertain/inferred provenance
    inferred_present = sum(1 for f in present_fields if f["status"] == "inferred")
    frac_inferred = (inferred_present / n_present) if n_present else 0.0

    # (c) unknown fraction over capturable fields (missing == provenance unknown)
    n_missing = sum(1 for f in captured if f["status"] == "missing")
    frac_unknown = (n_missing / n) if n else 1.0

    # weighted blend, clamped to [0,1]. Weights are a versioned modeling choice.
    raw = 0.40 * digitization_baseline + 0.30 * frac_inferred + 0.30 * frac_unknown
    uncertainty = round(max(0.0, min(1.0, raw)), 6)

    return {
        "metadata_uncertainty": uncertainty,
        "interpretation": "0 = trustworthy, 1 = untrustworthy; SEPARATE from "
                          "completeness (present ≠ correct)",
        "components": {
            "digitization_baseline": digitization_baseline,
            "frac_present_fields_inferred": round(frac_inferred, 6),
            "frac_capturable_unknown": round(frac_unknown, 6),
        },
        "caveat": ("completeness ≠ correctness: a PRESENT pH may still be WRONG. "
                   "This scalar flags TRUST, it cannot VERIFY any recorded value."),
    }


# ===================== 5. DEPENDENCY GRAPH (from M5) ======================= #
def dependency_graph() -> dict:
    """The inference-target DAG DERIVED FROM M5's gates (§3-M5). Each target lists
    the metadata prerequisites that license it; the MECHANISM node mirrors the M5
    hard gate set (assay-mass ∧ agitation-known ∧ seeding-known ∧ series ∧ shape)."""
    return {
        "version": M11_VERSION,
        "derived_from": "m5_classify.mechanistic_assessment gate set (§3-M5)",
        "targets": {
            "IDENTITY": {
                "prerequisites": ["protein"],
                "logic": "protein"},
            "COMPARABILITY": {
                "prerequisites": ["construct", "pH", "temperature"],
                "logic": "construct ∧ pH ∧ temperature"},
            "SCALING": {
                "prerequisites": ["concentration_series", "assay"],
                "logic": "concentration_series(≥3) ∧ assay(mass)",
                "note": "assay must be mass-proportional (M5 gate)"},
            "MECHANISM": {
                "prerequisites": MECHANISM_PREREQS,
                "logic": ("assay(mass) ∧ agitation(known) ∧ seeding(known) ∧ "
                          "concentration_series(≥3) ∧ cooperative_shape"),
                "mirrors_m5_gates": [M5_GATE_ASSAY_NOT_MASS, M5_GATE_ASSAY_UNKNOWN,
                                     M5_GATE_AGIT_SEED, M5_GATE_NO_SHAPE]},
        },
    }


def _prereq_satisfied(prereq: str, comp_per_field: dict, series: dict,
                      features_regime: str | None) -> tuple[bool, str]:
    """Is ONE dependency prerequisite satisfied? Returns (ok, note). Mirrors the
    M5 gates: assay must be mass-proportional (not merely present); agitation/
    seeding must be provenance='known'; the shape prereq needs M5 output."""
    if prereq == "cooperative_shape":
        # M11 does NOT re-run M5; if the driver supplies the descriptive regime we
        # honor it, else we say the shape is undetermined (never faked).
        if features_regime in ("cooperative_sigmoidal", "threshold_driven"):
            return True, f"M5 regime '{features_regime}' has a nucleated transition"
        if features_regime is None:
            return False, ("cooperative_shape unknown_needs_curve: no M5 output "
                           "supplied — NOT faked (M5 mirrors gate "
                           f"'{M5_GATE_NO_SHAPE}')")
        return False, (f"M5 regime '{features_regime}' has no nucleated transition "
                       f"(M5 gate '{M5_GATE_NO_SHAPE}')")

    fld = comp_per_field.get(prereq, {})
    status = fld.get("status")
    if prereq == "assay":
        cv = series.get("condition_vector", {}) or {}
        amp = cv.get("assay_reports_mass")
        if amp is True:
            return True, "assay is mass-proportional (M5 gate satisfied)"
        if amp is None:
            return False, f"assay mass-proportionality UNKNOWN (M5 gate '{M5_GATE_ASSAY_UNKNOWN}')"
        return False, f"assay NOT mass-proportional (M5 gate '{M5_GATE_ASSAY_NOT_MASS}')"
    # for the rest: present or inferred counts as satisfied for the prereq
    ok = status in ("present", "inferred")
    if prereq in ("agitation", "seeding"):
        # mirror M5 EXACTLY: only provenance='known' satisfies the branch
        ok = status == "present"
        if not ok:
            return False, (f"{prereq} not provenance='known' (M5 gate "
                           f"'{M5_GATE_AGIT_SEED}')")
    return ok, (f"{prereq}={status}")


def evaluate_targets(series: dict, features_regime: str | None = None) -> dict:
    """Evaluate every DAG target: satisfied/blocked + the blocking prerequisites.

    `features_regime` is the M5 descriptive regime (optional); without it the
    cooperative_shape prereq is `unknown_needs_curve` (never faked). Also returns
    **mechanistic_completeness ∈ [0,1]** = fraction of MECHANISM prereqs satisfied,
    with the explicit blocker list. Never raises."""
    if not isinstance(series, dict):
        series = {}
    comp = score_completeness(series)
    per = comp["per_field"]
    graph = dependency_graph()

    target_eval = {}
    for tname, tdef in graph["targets"].items():
        blockers = []
        prereq_status = {}
        for pr in tdef["prerequisites"]:
            ok, note = _prereq_satisfied(pr, per, series, features_regime)
            prereq_status[pr] = {"satisfied": ok, "note": note}
            if not ok:
                blockers.append(pr)
        target_eval[tname] = {
            "satisfied": len(blockers) == 0,
            "logic": tdef["logic"],
            "blocking_prerequisites": blockers,
            "prerequisite_status": prereq_status,
        }

    # mechanistic_completeness — fraction of MECHANISM prereqs satisfied
    mech = target_eval["MECHANISM"]
    n_prereq = len(MECHANISM_PREREQS)
    n_sat = sum(1 for pr in MECHANISM_PREREQS
                if mech["prerequisite_status"][pr]["satisfied"])
    mech_completeness = round(n_sat / n_prereq, 6) if n_prereq else 0.0

    return {
        "dependency_evaluation": target_eval,
        "mechanistic_completeness": mech_completeness,
        "mechanism_prerequisites_satisfied": n_sat,
        "mechanism_prerequisites_total": n_prereq,
        "mechanistic_blockers": mech["blocking_prerequisites"],
        "shape_prereq_note": ("cooperative_shape requires M5 output; when absent it "
                              "is 'unknown_needs_curve' and NOT assumed"),
        "derived_from": "M5 mechanistic gate set (§3-M5)",
    }


# ===================== 4. Δ-INFERENTIAL POWER ============================= #
# Service-C MEASURED anchor: single-concentration collapses 4 Tier-B mechanisms
# into 1 class; a γ-resolved concentration series splits them into 4 singletons
# (the 1->4 split, 6 pairs newly resolved). Read live from service_c if the
# driver supplies it; else fall back to the measured constants.
_SC_MEASURED_SPLIT = {
    "single_concentration_classes": 1,
    "concentration_series_classes": 4,
    "pairs_newly_resolved": 6,
    "gain_basis": "service_c_measured",
    "source": "service_c_calibration.concentration_series_narrowing (K2)",
    "caveat": "an UPPER BOUND on discriminability — real mechanisms can share γ",
}

# Which tiers each field blocks when MISSING, its impact, the consequence chain,
# and the basis of its information gain. Mechanism/scaling-gating fields anchor to
# the Service-C measured split; everything else is a STRUCTURAL upper bound.
_FIELD_IMPACT = {
    "agitation": ("high", ["mechanism"],
        "missing agitation -> quiescent-vs-shaken branch undetermined -> "
        "mechanism refused (M5 gate agitation_or_seeding_unknown) -> HIGH"),
    "seeding": ("high", ["mechanism"],
        "missing seeding -> primary-vs-secondary nucleation branch undetermined "
        "-> mechanism refused (M5 gate agitation_or_seeding_unknown) -> HIGH"),
    "assay": ("high", ["scaling", "mechanism"],
        "missing/unknown-mass assay -> no mass-action rate law -> scaling & "
        "mechanism refused (M5 gate assay_*_mass_proportional) -> HIGH"),
    "concentration_series": ("high", ["scaling", "mechanism"],
        "missing concentration series -> γ scaling exponent unresolvable -> "
        "Tier-B stays 1 broad class (not split to 4) -> scaling & mechanism -> HIGH"),
    "concentration": ("moderate", ["comparability", "scaling"],
        "missing concentration -> cannot place the curve on the γ axis -> "
        "scaling weakened -> MODERATE"),
    "dye": ("moderate", ["comparability", "mechanism"],
        "missing dye/assay detail -> mass-proportionality ambiguous -> "
        "mechanism confidence reduced -> MODERATE"),
    "pH": ("moderate", ["comparability"],
        "missing pH -> cross-condition comparability widened -> MODERATE"),
    "temperature": ("moderate", ["comparability"],
        "missing temperature -> cross-condition comparability widened -> MODERATE"),
    "construct": ("moderate", ["identity", "comparability"],
        "missing construct -> WT/mutant ambiguous -> identity & comparability -> MODERATE"),
    "mutation": ("moderate", ["identity", "comparability"],
        "missing mutation -> WT/mutant ambiguous -> identity & comparability -> MODERATE"),
    "uniprot": ("moderate", ["identity", "comparability"],
        "missing uniprot -> weaker protein identity/cross-refs -> MODERATE"),
    "protein": ("high", ["identity"],
        "missing protein -> no identity -> NOTHING is licensable -> HIGH"),
    "ionic_strength": ("moderate", ["comparability"],
        "missing ionic strength -> ionic-condition comparability widened -> MODERATE"),
    "replicate_count": ("moderate", ["uncertainty"],
        "missing replicates -> stochastic-nucleation variance unquantified -> "
        "uncertainty inflated -> MODERATE"),
    "digitization_confidence": ("negligible", ["uncertainty"],
        "digitization confidence is always inferred (CPAD digitized) -> NEGLIGIBLE"),
    "buffer": ("negligible", ["comparability"],
        "missing buffer -> minor comparability detail -> NEGLIGIBLE"),
    "reducing_agent": ("negligible", ["comparability"],
        "missing reducing agent -> minor comparability detail, not a first-class "
        "field -> NEGLIGIBLE"),
    "organism": ("negligible", ["identity"],
        "missing organism -> minor identity detail (derivable from uniprot) -> NEGLIGIBLE"),
    "publication": ("negligible", ["none"],
        "missing publication -> provenance/audit only, no inference -> NEGLIGIBLE"),
}

# fields whose information_gain is anchored to the Service-C MEASURED split
_SC_ANCHORED = {"concentration_series", "assay", "agitation", "seeding"}


def delta_inferential_power(series: dict, features_regime: str | None = None,
                            service_c: dict | None = None) -> dict:
    """For EACH missing/unknown field: the tiers it blocks, its impact, the
    plain-English consequence_chain, the tier-lift it would enable CONDITIONAL on
    what else is missing, and gain_basis. Mechanism/scaling-gating fields anchor
    to the Service-C MEASURED 1->4 split; the rest are STRUCTURAL upper bounds.

    HONESTY: every non-service_c number is a STRUCTURAL UPPER BOUND under the DAG,
    inheriting M8's validity ceiling. Never raises."""
    if not isinstance(series, dict):
        series = {}
    comp = score_completeness(series)
    per = comp["per_field"]
    tgt = evaluate_targets(series, features_regime)

    # what ELSE is missing conditions the gain (a series lift is worthless if the
    # assay is non-mass; agitation is the last piece only WITH a series present).
    def _missing(f):
        return per.get(f, {}).get("status") in ("missing",) or (
            per.get(f, {}).get("status") == "inferred" and f in _SC_ANCHORED)

    has_series = not _missing("concentration_series")
    has_mass_assay = (series.get("condition_vector", {}) or {}).get(
        "assay_reports_mass") is True

    sc_split = _sc_split(service_c)

    items = []
    for field, fdef in per.items():
        status = fdef["status"]
        if status not in ("missing", "inferred"):
            continue
        impact, blocks, chain = _FIELD_IMPACT.get(
            field, ("negligible", ["none"], f"missing {field} -> minor -> NEGLIGIBLE"))

        # information_gain, CONDITIONAL on co-missing fields
        gain_basis = "structural_estimate"
        info_gain = None
        if field in _SC_ANCHORED:
            gain_basis = sc_split["gain_basis"]
            if field == "concentration_series":
                if has_mass_assay:
                    info_gain = {
                        "tier_lift": "scaling+mechanism",
                        "tier_b_classes_before": sc_split["single_concentration_classes"],
                        "tier_b_classes_after": sc_split["concentration_series_classes"],
                        "pairs_newly_resolved": sc_split["pairs_newly_resolved"],
                        "conditional_on": "mass-proportional assay present",
                        "note": "the MEASURED γ 1->4 Tier-B split (K2)"}
                else:
                    info_gain = {
                        "tier_lift": "none_until_assay_is_mass",
                        "conditional_on": "BLOCKED: assay is not mass-proportional",
                        "note": "a series γ-lift is worthless without a mass assay"}
            elif field == "assay":
                info_gain = {
                    "tier_lift": "unblocks scaling+mechanism prerequisite",
                    "conditional_on": "concentration series also required for the full lift",
                    "note": "mass-proportional assay is a gate for both scaling & mechanism"}
            elif field in ("agitation", "seeding"):
                if has_series and has_mass_assay:
                    info_gain = {
                        "tier_lift": "mechanism_license (LAST missing piece)",
                        "conditional_on": "series + mass assay already present",
                        "note": "with a series present, recording this UNBLOCKS the "
                                "mechanism license (the corpus-wide M8 GAP closer)"}
                else:
                    info_gain = {
                        "tier_lift": "removes_a_future_gate_only",
                        "conditional_on": "NOT sufficient alone: a concentration "
                                          "series (+mass assay) is the binding prereq",
                        "note": "necessary-but-not-sufficient on a single curve"}
        else:
            # STRUCTURAL upper bound: the tiers this field participates in
            node = METADATA_ONTOLOGY["fields"].get(field, {})
            info_gain = {
                "tier_lift": "+".join(node.get("gates", ["none"])),
                "conditional_on": "structural upper bound under the DAG (M8 ceiling)",
                "note": "STRUCTURAL estimate — a tier-lift ceiling, not a measured gain"}

        items.append({
            "field": field,
            "current_status": status,
            "blocks": blocks,
            "impact": impact,
            "consequence_chain": chain,
            "information_gain": info_gain,
            "gain_basis": gain_basis,
        })

    # rank: high>moderate>negligible, then service_c-anchored first
    _imp = {"high": 0, "moderate": 1, "negligible": 2}
    items.sort(key=lambda it: (_imp.get(it["impact"], 3),
                               0 if it["gain_basis"] == "service_c_measured" else 1,
                               it["field"]))
    return {
        "delta_inferential_power": items,
        "n_missing_or_uncertain": len(items),
        "mechanistic_completeness": tgt["mechanistic_completeness"],
        "honesty": ("Δ-power numbers are STRUCTURAL UPPER BOUNDS (tier lifts under "
                    "the DAG) inheriting M8's validity ceiling — EXCEPT gain_basis="
                    "'service_c_measured' (the γ-resolved 1->4 Tier-B split)."),
    }


def _sc_split(service_c: dict | None) -> dict:
    """Read the MEASURED 1->4 split live from service_c if available, else the
    pinned measured constants."""
    if service_c:
        n = service_c.get("concentration_series_narrowing", {}) or {}
        s1 = (n.get("single_concentration_partition", {}) or {}).get("n_classes")
        s4 = (n.get("concentration_series_partition", {}) or {}).get("n_classes")
        pairs = n.get("pairs_newly_resolved_by_series")
        if s1 and s4:
            return {
                "single_concentration_classes": s1,
                "concentration_series_classes": s4,
                "pairs_newly_resolved": len(pairs) if pairs else None,
                "gain_basis": "service_c_measured",
                "source": "service_c_calibration.concentration_series_narrowing (K2)",
            }
    return dict(_SC_MEASURED_SPLIT)


# ===================== 6. RECOMMENDED MISSING METADATA ==================== #
# Effort model (M9 action language): recording agitation/seeding is CHEAP; adding
# a concentration series is MEDIUM; everything else is cheap-to-medium.
_EFFORT = {
    "agitation": (1.0, "cheap", "record_agitation_and_seeding"),
    "seeding": (1.0, "cheap", "record_agitation_and_seeding"),
    "assay": (1.5, "cheap", "record_assay_mass_proportionality"),
    "dye": (1.0, "cheap", "record_dye_and_concentration"),
    "concentration_series": (3.0, "medium", "add_concentration_series"),
    "concentration": (1.0, "cheap", "record_concentration"),
    "pH": (1.0, "cheap", "record_pH"),
    "temperature": (1.0, "cheap", "record_temperature"),
    "ionic_strength": (1.0, "cheap", "record_ionic_strength"),
    "construct": (1.0, "cheap", "record_construct"),
    "mutation": (1.0, "cheap", "record_mutation"),
    "uniprot": (1.0, "cheap", "record_uniprot_id"),
    "replicate_count": (2.5, "medium", "add_replicates"),
    "buffer": (1.0, "cheap", "record_buffer"),
    "reducing_agent": (1.0, "cheap", "record_reducing_agent"),
}
_IMPACT_GAIN = {"high": 3.0, "moderate": 1.5, "negligible": 0.3}


def recommended_missing_metadata(series: dict, features_regime: str | None = None,
                                 service_c: dict | None = None) -> dict:
    """Missing fields ranked by information_gain ÷ effort, in M9's action language.
    Never raises."""
    if not isinstance(series, dict):
        series = {}
    dp = delta_inferential_power(series, features_regime, service_c)
    recs = []
    for it in dp["delta_inferential_power"]:
        field = it["field"]
        effort_val, effort_label, action = _EFFORT.get(
            field, (2.0, "medium", f"record_{field}"))
        # service_c-measured mechanism unblockers get a leverage bump
        gain = _IMPACT_GAIN.get(it["impact"], 0.3)
        if it["gain_basis"] == "service_c_measured" and it["impact"] == "high":
            gain *= 1.5
        score = round(gain / effort_val, 4)
        recs.append({
            "field": field,
            "action": action,
            "impact": it["impact"],
            "effort": effort_label,
            "gain_over_effort": score,
            "gain_basis": it["gain_basis"],
            "blocks": it["blocks"],
            "rationale": it["consequence_chain"],
        })
    recs.sort(key=lambda r: r["gain_over_effort"], reverse=True)
    for i, r in enumerate(recs, 1):
        r["rank"] = i
    return {
        "recommended_missing_metadata": recs,
        "method": "information_gain ÷ effort; ties broken by impact then field",
        "action_language": "aligned with M9 experimental-design recommender",
    }


# ============================ 7. ASSESS (full) ============================= #
def assess_metadata(series: dict, features_regime: str | None = None,
                    weights: dict | None = None,
                    service_c: dict | None = None) -> dict:
    """The full per-dataset M11 record. NEVER raises — bad input yields a flagged
    minimal result."""
    if not isinstance(series, dict):
        return {
            "series_id": None, "version": M11_VERSION, "malformed": True,
            "note": "input was not a dict — flagged minimal result (never raises)",
            "metadata_completeness": score_completeness({}),
            "metadata_uncertainty": metadata_uncertainty({}),
        }
    try:
        completeness = score_completeness(series, weights)
        uncertainty = metadata_uncertainty(series)
        targets = evaluate_targets(series, features_regime)
        dpow = delta_inferential_power(series, features_regime, service_c)
        recs = recommended_missing_metadata(series, features_regime, service_c)
    except Exception as exc:  # defensive envelope — the module NEVER raises
        return {
            "series_id": series.get("series_id"), "version": M11_VERSION,
            "malformed": True, "error": str(exc),
            "metadata_completeness": score_completeness({}),
        }

    return {
        "series_id": series.get("series_id"),
        "protein_id": series.get("protein_id"),
        "uniprot_id": series.get("uniprot_id"),
        "concentration_series_id": series.get("concentration_series_id"),
        "version": M11_VERSION,
        "ontology_version": ONTOLOGY_VERSION,
        "metadata_completeness": completeness,
        "metadata_uncertainty": uncertainty,
        "mechanistic_completeness": targets["mechanistic_completeness"],
        "mechanistic_blockers": targets["mechanistic_blockers"],
        "dependency_evaluation": targets["dependency_evaluation"],
        "delta_inferential_power": dpow["delta_inferential_power"],
        "recommended_missing_metadata": recs["recommended_missing_metadata"],
        "not_captured_by_source": completeness["not_captured_by_source"],
        "features_regime_used": features_regime,
        "honesty": {
            "completeness_neq_correctness": (
                "metadata_score = how much is PRESENT; it does NOT verify values"),
            "delta_power_upper_bound": (
                "Δ-power is a STRUCTURAL UPPER BOUND (tier lifts under the DAG, "
                "M8 ceiling) except gain_basis='service_c_measured'"),
            "weights_are_a_modeling_choice": (
                f"defaults are the versioned {ONTOLOGY_VERSION} choice; configurable"),
            "not_captured_is_schema_not_dataset": (
                "not_captured_by_source is a DATABASE-SCHEMA limitation, NOT the "
                "dataset's fault; excluded from the score denominator"),
        },
    }


# ================================== I/O ==================================== #
def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _regime_index(regimes_path: Path) -> dict:
    """Optional: map series_id -> M5 descriptive regime (for the shape prereq)."""
    out = {}
    for r in _read_jsonl(regimes_path) if regimes_path.exists() else []:
        sid = r.get("series_id")
        reg = (r.get("descriptive_regime", {}) or {}).get("regime")
        if sid and reg:
            out[sid] = reg
    return out


def _build_series_index(curves: list) -> dict:
    """series_sizes {cs_id: n} + replicate_counts {series_id: n_same_condition}."""
    series_sizes: Counter = Counter()
    for cu in curves:
        cs = cu.get("concentration_series_id")
        if cs:
            series_sizes[cs] += 1
    # replicate_count = #curves for the same protein at the SAME (conc,pH,T,assay)
    cond_groups: dict = {}
    for cu in curves:
        cv = cu.get("condition_vector", {}) or {}
        c = cv.get("concentration", {}) or {}
        key = (cu.get("protein_id"), (c.get("value_uM") if isinstance(c, dict) else None),
               cv.get("pH"), cv.get("temperature_C"), cv.get("assay_type"))
        cond_groups.setdefault(key, []).append(cu.get("series_id"))
    replicate_counts = {}
    for _, sids in cond_groups.items():
        for sid in sids:
            replicate_counts[sid] = len(sids)
    return {"series_sizes": dict(series_sizes), "replicate_counts": replicate_counts}


def _percentiles(vals: list[float]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    def q(p):
        k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
        return round(s[k], 6)
    return {"min": round(s[0], 6), "p25": q(0.25), "median": q(0.5),
            "p75": q(0.75), "max": round(s[-1], 6),
            "mean": round(sum(s) / len(s), 6)}


def _histogram(vals: list[float], edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.01)) -> dict:
    h = {}
    labels = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
    for lab in labels:
        h[lab] = 0
    for v in vals:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                h[labels[i]] += 1
                break
    return h


def run_corpus(indir: Path, outdir: Path, limit: int = 0) -> dict:
    """Run M11 over all triaged curves; roll up to protein + corpus; write the
    JSON + JSONL artifacts. Deterministic."""
    curves = list(_read_jsonl(indir / "curves_triaged.jsonl"))
    if limit:
        curves = curves[:limit]
    idx = _build_series_index(curves)
    regime_idx = _regime_index(indir / "regimes.jsonl")
    service_c = _read_json(indir / "service_c_calibration.json")

    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / "metadata_quality.jsonl"

    scores = []
    mech_completeness = []
    uncerts = []
    n_agit_seed_blocked = 0
    n_mechanism_licensed = 0
    per_protein: dict = {}
    rec_action_counter: Counter = Counter()
    rec_gain_accum: dict = {}

    with open(jsonl_path, "w", encoding="utf-8") as out:
        for cu in curves:
            cu["_m11_index"] = idx
            regime = regime_idx.get(cu.get("series_id"))
            res = assess_metadata(cu, features_regime=regime, service_c=service_c)
            cu.pop("_m11_index", None)
            res.pop("_m11_index", None)

            score = res["metadata_completeness"]["metadata_score"]
            mc = res["mechanistic_completeness"]
            scores.append(score)
            mech_completeness.append(mc)
            uncerts.append(res["metadata_uncertainty"]["metadata_uncertainty"])

            blk = res["mechanistic_blockers"]
            if "agitation" in blk or "seeding" in blk:
                n_agit_seed_blocked += 1
            if res["dependency_evaluation"]["MECHANISM"]["satisfied"]:
                n_mechanism_licensed += 1

            # corpus recommendation rollup — sum gain/effort per action
            for r in res["recommended_missing_metadata"]:
                rec_action_counter[r["action"]] += 1
                rec_gain_accum.setdefault(r["action"], {"gain": 0.0, "impact": r["impact"],
                                                        "effort": r["effort"]})
                rec_gain_accum[r["action"]]["gain"] += r["gain_over_effort"]

            pid = res.get("protein_id")
            per_protein.setdefault(pid, {"scores": [], "mech": [], "n": 0})
            per_protein[pid]["scores"].append(score)
            per_protein[pid]["mech"].append(mc)
            per_protein[pid]["n"] += 1

            out.write(json.dumps(res) + "\n")

    # protein rollup
    protein_rollup = {}
    for pid, d in per_protein.items():
        protein_rollup[pid] = {
            "n_curves": d["n"],
            "median_metadata_score": _percentiles(d["scores"]).get("median"),
            "median_mechanistic_completeness": _percentiles(d["mech"]).get("median"),
        }

    # corpus recommendation ranking (by total gain/effort across curves)
    corpus_recs = sorted(
        ({"action": a, "curves_affected": rec_action_counter[a],
          "total_gain_over_effort": round(v["gain"], 4),
          "impact": v["impact"], "effort": v["effort"]}
         for a, v in rec_gain_accum.items()),
        key=lambda r: r["total_gain_over_effort"], reverse=True)
    for i, r in enumerate(corpus_recs, 1):
        r["rank"] = i

    n = len(curves)
    corpus = {
        "version": M11_VERSION,
        "ontology_version": ONTOLOGY_VERSION,
        "n_curves": n,
        "n_proteins": len([p for p in per_protein if p is not None]),
        "metadata_score": {
            "distribution": _percentiles(scores),
            "histogram": _histogram(scores),
        },
        "mechanistic_completeness": {
            "distribution": _percentiles(mech_completeness),
            "histogram": _histogram(mech_completeness),
        },
        "metadata_uncertainty": {"distribution": _percentiles(uncerts)},
        "universal_blocker": {
            "n_curves_agitation_or_seeding_blocked": n_agit_seed_blocked,
            "fraction": round(n_agit_seed_blocked / n, 6) if n else 0.0,
            "why_mechanism_licensed_is_zero": (
                f"MECHANISM is licensed for {n_mechanism_licensed} / {n} curves. "
                "The universal blocker is {agitation, seeded} = 'unknown' corpus-"
                "wide: the M5 quiescent-vs-shaken / primary-vs-secondary hard "
                "branch is undetermined for EVERY curve, so the mechanism gate "
                "refuses regardless of how complete the rest of the metadata is. "
                "This is the SAME gate M5 applies and the SAME GAP M8 reports — "
                "M11 quantifies WHY, it does not invent a parallel logic."),
        },
        "n_mechanism_licensed": n_mechanism_licensed,
        "corpus_recommendations_ranked": corpus_recs,
        "honesty": {
            "completeness_neq_correctness": (
                "metadata_score measures PRESENCE, not correctness of values"),
            "delta_power_upper_bound": (
                "Δ-power is a STRUCTURAL UPPER BOUND except service_c_measured"),
            "weights_are_a_modeling_choice": (
                f"{ONTOLOGY_VERSION} default weights; configurable"),
            "not_captured_is_schema_not_dataset": (
                "not_captured_by_source excluded from the denominator (schema, "
                "not the dataset's fault)"),
        },
    }
    (outdir / "metadata_quality.json").write_text(
        json.dumps({"corpus": corpus, "by_protein": protein_rollup}, indent=2),
        encoding="utf-8")

    return {"corpus": corpus, "jsonl_path": str(jsonl_path)}


def main(argv=None) -> int:
    # WHY: the printed summary carries a few non-ASCII glyphs (≠, ≈, ÷); a Windows
    # cp1252 console would UnicodeEncodeError on them. Reconfigure stdout/stderr to
    # UTF-8 with a replace fallback so the CLI NEVER crashes on the print (the JSON
    # artifacts are already written UTF-8 and are unaffected). Best-effort.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    root = Path(__file__).resolve().parent.parent
    proc_default = root / "data" / "processed"
    ap = argparse.ArgumentParser(
        description="PRISE M11 — Metadata Quality & Information Completeness Engine")
    ap.add_argument("--input", type=Path, default=proc_default,
                    help="directory holding curves_triaged.jsonl + artifacts")
    ap.add_argument("--output", type=Path, default=None,
                    help="output directory (default: same as --input)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    indir = Path(args.input)
    outdir = Path(args.output) if args.output else indir
    triaged = indir / "curves_triaged.jsonl"
    if not triaged.exists():
        raise SystemExit(f"input not found: {triaged} (run M1 first)")

    print(f"[M11] scoring metadata quality over {triaged.name} ...")
    result = run_corpus(indir, outdir, limit=args.limit)
    c = result["corpus"]

    print(f"[M11] wrote {outdir / 'metadata_quality.json'}")
    print(f"[M11] wrote {result['jsonl_path']}")
    print("\n  METADATA_SCORE distribution (completeness ≠ correctness):")
    d = c["metadata_score"]["distribution"]
    print(f"    median={d.get('median')}  mean={d.get('mean')}  "
          f"min={d.get('min')}  max={d.get('max')}")
    print(f"    histogram: {c['metadata_score']['histogram']}")
    print("\n  MECHANISTIC_COMPLETENESS distribution (fraction of M5 mechanism "
          "prereqs satisfied):")
    md = c["mechanistic_completeness"]["distribution"]
    print(f"    median={md.get('median')}  mean={md.get('mean')}  max={md.get('max')}")
    print(f"    histogram: {c['mechanistic_completeness']['histogram']}")
    ub = c["universal_blocker"]
    print(f"\n  UNIVERSAL BLOCKER: {ub['n_curves_agitation_or_seeding_blocked']} / "
          f"{c['n_curves']} curves have the {{agitation, seeded}} blocker "
          f"({100 * ub['fraction']:.1f}%)")
    print(f"  MECHANISM licensed for {c['n_mechanism_licensed']} / {c['n_curves']} curves")
    print(f"  WHY ≈ 0: {ub['why_mechanism_licensed_is_zero'][:200]}...")
    print("\n  TOP CORPUS RECOMMENDATIONS (information gain ÷ effort):")
    for r in c["corpus_recommendations_ranked"][:5]:
        print(f"    #{r['rank']} {r['action']:<34} impact={r['impact']:<10} "
              f"effort={r['effort']:<7} gain/effort={r['total_gain_over_effort']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
