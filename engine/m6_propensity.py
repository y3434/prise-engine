"""
PRISE — Module M6: Propensity Scorer & Sequence axis
====================================================

Turns the per-curve/per-series analysis (M4 features + dual-γ, M5 regimes) and the
CPAD sequence/structure layer into a **propensity** statement, plus a de-conflated
**sequence axis** and an associative **structure-linkage** panel
(PRISE_DESIGN.md §3-M6).

The honesty rules that shape this module:

  * **Three propensity views, never collapsed into one number** (§3-M6):
      - `surface`   — feature vs driving-force + γ (PRIMARY where a concentration
                      series exists); from M4.
      - `intrinsic` — a position on a *reference-anchored* scale: a leakage-free
                      **corpus** anchor where a condition/assay-matched stratum has
                      adequate effective n, else a **versioned reference scale**.
                      `anchor_kind` records which; valid only at its condition vector.
      - `cohort`    — rank/percentile within the cohort under analysis; suppressed
                      ("N too small") for singletons.
  * **Portability rule:** only *time-domain* features (t50, γ, lag, sharpness) are
    portable. Amplitude / a.u. features are instrument-local and **barred from
    cohort comparison** even within a matched stratum.
  * **Sequence axis is de-conflated:** TANGO / AGGRESCAN / PASTA / Waltz are
    **separate named sub-axes** (granularity, endpoint, version, explicit reduction);
    Zyggregator & CamSol are **declared absent** (not silently dropped). A
    sequence-vs-kinetics comparison is **licensed only on endpoint match** (an
    amyloid predictor ↔ a ThT assay, not ↔ a turbidity/amorphous curve).
  * **Structure linkage is associative only** — never a causal claim, polymorphism
    a first-class caveat.

The GROUPING UNIT is one record per **protein × construct × assay** (the governed
`M6_GROUP_KEY`, join-policy-1.0). The assay axis was added after it was measured
that 4 of 133 groups pooled t50 across assays and then labelled the pooled median
with whichever assay was seen first — P01160 Wild Type shipped as `assay:
"CongoRed"` carrying 44.851 h, which is the ThT median, while its CongoRed data
sits at 586.870 h. Splitting on assay also makes each anchoring stratum contain
only values measured the same way, and makes `k114` reachable as an assay rather
than absorbed into a ThT-labelled group.

Inputs are precomputed artifacts (no fitting here). All outputs JSON-serialisable.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# The GOVERNED join policy (join_policy.py). The grouping key is READ from here
# rather than hard-coded: if the new key were inlined, the retained `previous`
# would be a comment instead of a revert path. The fallback mirrors the shipped
# `applied` value so an isolated import cannot silently restore the pre-fix key.
try:                                                   # pragma: no cover - import glue
    from join_policy import applied_policy as _applied_policy
except Exception:                                      # pragma: no cover - fallback path
    def _applied_policy(name, default=None):
        return {"M6_GROUP_KEY": ["uniprot", "construct", "assay"],
                "M6_GAMMA_JOIN": "protein_name_and_assay",
                "M6_ANCHOR_LEAKAGE": "leave_one_study_out",
                "ASSAY_ENDPOINT_AMYLOID_ADDITIONS": ["k114"]}.get(name, default)

# ---- portability (design: time-domain portable; amplitude/a.u. barred) ------ #
PORTABLE_FEATURES = {"t50", "lag_time", "lag_to_t50_ratio",
                     "transition_sharpness", "gamma"}
AMPLITUDE_FEATURES = {"plateau", "dynamic_range", "max_rate"}   # instrument-local

MIN_ANCHOR_EFFECTIVE_N = 8     # stratum needs >= this many OTHER values to corpus-anchor
MIN_COHORT_N = 3               # below this the cohort percentile is suppressed
# Bumped whenever the COMPOSITION of an anchoring stratum changes -- precisely what
# this axis exists to describe, so leaving it fixed while the pool moved would be a
# false version statement. 1.0 -> 1.1: the grouping key gained the ASSAY axis
# (join-policy-1.0). 1.1 -> 1.2: the anchor pool became leave-one-STUDY-out
# (join-policy-1.2), which removes same-study groups from every comparison.
ANCHOR_VERSION = "m6-anchor-1.2"

# Versioned physical/reference scales used when the corpus stratum is too thin.
# "deterministic given anchor_version" — NOT "absolute across conditions".
REFERENCE_SCALES = {
    "version": "m6-refscale-1.0",
    "scales": {
        "t50": {"transform": "log10", "ref_low": 1.0, "ref_high": 1000.0,
                "rationale": "amyloid half-times span ~hours to weeks"},
        "gamma": {"transform": "linear", "ref_low": 0.0, "ref_high": 2.0,
                  "rationale": "physical half-time scaling exponents"},
        "lag_to_t50_ratio": {"transform": "linear", "ref_low": 0.0, "ref_high": 1.0,
                             "rationale": "dimensionless shape invariant in [0,1)"},
        "transition_sharpness": {"transform": "log10", "ref_low": 0.3,
                                 "ref_high": 30.0, "rationale": "t50/width sharpness"},
    },
}

# Endpoint classes for the sequence-vs-kinetics match gate.
# Case-insensitive substring keys covering the amyloid/fibril-SPECIFIC dyes from the
# canonical assay vocabulary: ThT, ThS, Congo Red, Cytofluor (review item A1).
# turbidity / light_scattering / DLS / SLS / generic "fluorescence" are NOT here —
# they stay "generic" (they report mass/scatter, not cross-β amyloid structure).
# Amyloid-SPECIFIC dyes: they report the cross-β fibril itself, so a cross-β
# sequence predictor (PASTA, Waltz) is endpoint-matched against them.
#
# `k114` arrives from the GOVERNED vocabulary (join-policy-1.1) rather than being
# hard-coded, so the classification is reversible and carries its citation. K114 is
# a Congo-red/X-34-derived fluorophore introduced to quantify amyloid
# fibrillogenesis (Crystal et al., J Neurochem 2003;86:1359-1368) — same detection
# principle and same molecular target as ThT/ThS/Congo Red. Classing it "unknown"
# was not conservative but wrong: it barred the cross-β-specific predictors from a
# cross-β-specific assay, which is the endpoint match the rule exists to permit.
_AMYLOID_ASSAYS = {"tht", "ths", "thioflavin", "congo", "cytoflu"} | {
    str(a).strip().lower()
    for a in (_applied_policy("ASSAY_ENDPOINT_AMYLOID_ADDITIONS", ["k114"]) or ())
    if str(a).strip()}
# Known GENERIC (non-amyloid-specific) assays from the canonical vocabulary:
# turbidity / light-scattering (DLS/SLS) / generic fluorescence. A non-empty assay
# that matches NONE of the amyloid or generic sets is genuinely unrecognised and
# is treated as "unknown" (licenses nothing), NOT silently as "generic".
_GENERIC_ASSAYS = {"turbid", "scatter", "dls", "sls", "fluor", "absorb", "abs350",
                   "od", "light_scattering"}
_PREDICTOR_ENDPOINT_CLASS = {                            # amyloid vs generic vs solubility
    # TANGO is GENERIC β-aggregation (design glossary); only PASTA & Waltz are
    # amyloid-specific cross-β predictors. So vs a ThT assay, only PASTA & Waltz
    # are licensed; TANGO & AGGRESCAN are endpoint-mismatched (review item A3).
    "tango": "generic", "pasta": "amyloid", "waltz": "amyloid",
    "aggrescan": "generic", "zyggregator": "generic", "camsol": "solubility"}


# ------------------------------ small stats -------------------------------- #
def _percentile_rank(value: float, vals: list[float]) -> float:
    """Hazen mid-rank percentile of `value` against `vals`, in percent (0–100).

    ONE convention shared by intrinsic and cohort (review item 8):
        percentile = 100 * (#below + 0.5*#equal) / N
    Mid-rank splits ties evenly, so the percentile is self-consistent with the
    ordinal rank and symmetric for ties (a value equal to the whole pool -> 50)."""
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    n = len(vals)
    if n == 0:
        return float("nan")
    below = sum(1 for v in vals if v < value)
    equal = sum(1 for v in vals if v == value)
    return 100.0 * (below + 0.5 * equal) / n


def _ref_position(value: float, scale: dict) -> dict:
    """Map a value onto [0,1] on a versioned reference scale.

    Returns {position, clamped, out_of_range} (review item 11): `clamped` is True
    when the raw position fell outside [0,1] (so the caller knows the protein is at
    or beyond a scale edge), `out_of_range` is "low"/"high"/None. position is None
    when the value is unmappable (non-finite, or non-positive on a log scale)."""
    null = {"position": None, "clamped": False, "out_of_range": None}
    if value is None or not math.isfinite(value):
        return null
    lo, hi = scale["ref_low"], scale["ref_high"]
    v, a, b = value, lo, hi
    if scale["transform"] == "log10":
        if value <= 0 or lo <= 0 or hi <= 0:
            return null
        v, a, b = math.log10(value), math.log10(lo), math.log10(hi)
    if b == a:
        return null
    raw = (v - a) / (b - a)
    pos = max(0.0, min(1.0, raw))
    out = "low" if raw < 0.0 else ("high" if raw > 1.0 else None)
    return {"position": pos, "clamped": out is not None, "out_of_range": out}


# --------------------------- propensity direction -------------------------- #
# Features where a LOW value means a HIGH aggregation propensity (fast/early
# events): t50, lag_time, and lag_to_t50_ratio are all "sooner = more prone".
# For these we expose a `direction` + a derived (sign-corrected) propensity view
# so the sign of the comparison is explicit rather than left to the reader to
# invert. intrinsic AND cohort BOTH route through these helpers, so the two can
# never disagree on direction or on the inversion (consistency by construction).
_LOW_IS_HIGH_PROPENSITY = {"t50", "lag_time", "lag_to_t50_ratio"}


def _propensity_direction(feature: str) -> str:
    """'low_is_high_propensity' (t50/lag-type) vs 'high_is_high_propensity'.

    THE single source of truth for direction — shared by intrinsic & cohort."""
    return ("low_is_high_propensity" if feature in _LOW_IS_HIGH_PROPENSITY
            else "high_is_high_propensity")


def _to_propensity_percentile(feature: str, percentile: float | None):
    """Sign-correct a raw feature percentile (0–100) into a propensity percentile.

    For a low-is-high feature HIGH propensity = LOW feature value, so the
    propensity percentile is `100 - percentile`; pass-through otherwise. None in
    -> None out. THE single inversion rule — shared by intrinsic & cohort so the
    two can never disagree on the sign correction (consistency by construction)."""
    if percentile is None:
        return None
    if _propensity_direction(feature) == "low_is_high_propensity":
        return round(100.0 - percentile, 1)
    return percentile


def _to_propensity_position(feature: str, position: float | None):
    """Sign-correct a [0,1] reference position into a propensity position.

    Low-is-high feature -> `1 - position` (a low value is a high propensity);
    pass-through otherwise. None in -> None out (e.g. an unmappable value)."""
    if position is None:
        return None
    if _propensity_direction(feature) == "low_is_high_propensity":
        return 1.0 - position
    return position


# ----------------------------- intrinsic ----------------------------------- #
# Minimum stratum n for an interpolated 95% reference interval to be meaningful.
MIN_N_FOR_95_INTERVAL = 20

def intrinsic_anchor(feature: str, value: float, stratum_values: list[float]) -> dict:
    """Reference-anchored intrinsic position. `stratum_values` MUST already exclude
    the query (leakage-free, §8). Corpus anchor when n is adequate, else the
    versioned reference scale; `anchor_kind` records which."""
    vals = [v for v in (stratum_values or []) if v is not None and math.isfinite(v)]
    if len(vals) >= MIN_ANCHOR_EFFECTIVE_N:
        pct = _percentile_rank(value, vals)
        # 95% reference interval: numpy.percentile interpolation (review item 7),
        # but ONLY emit it when n >= 20 — with fewer points the 2.5/97.5 tails are
        # essentially the min/max and the "interval" is not meaningful. Below that
        # we null it and flag, rather than report a misleadingly precise spread.
        if len(vals) >= MIN_N_FOR_95_INTERVAL:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            ref_interval = [float(lo), float(hi)]
            interval_flag = None
        else:
            ref_interval = None
            interval_flag = "n_too_small_for_95_interval"
        pct_round = round(pct, 1)
        return {"value": value, "anchor_kind": "corpus",
                "percentile_in_stratum": pct_round,
                # sign-corrected propensity view — routed through the SAME helper as
                # cohort, so intrinsic & cohort can never disagree on direction/sign.
                "direction": _propensity_direction(feature),
                "propensity_percentile_in_stratum":
                    _to_propensity_percentile(feature, pct_round),
                "anchor_effective_n": len(vals),
                # the stratum's 95% reference spread the protein is placed within —
                # NOT a sampling CI of the percentile (named to avoid implying that)
                "stratum_reference_interval_95": ref_interval,
                "stratum_reference_interval_95_flag": interval_flag,
                "anchor_version": ANCHOR_VERSION}
    scale = REFERENCE_SCALES["scales"].get(feature)
    if scale is None:
        return {"value": value, "anchor_kind": "none",
                "reason": f"no reference scale for '{feature}'",
                "anchor_version": REFERENCE_SCALES["version"]}
    rp = _ref_position(value, scale)
    return {"value": value, "anchor_kind": "reference_scale",
            "reference_position_0_1": rp["position"],
            "reference_position_clamped": rp["clamped"],
            "reference_position_out_of_range": rp["out_of_range"],
            # sign-corrected propensity view of the reference position: for a
            # low-is-high feature 1 - position; pass-through otherwise; null when
            # position is null. Same direction helper as cohort (consistency).
            "direction": _propensity_direction(feature),
            "propensity_reference_position_0_1":
                _to_propensity_position(feature, rp["position"]),
            "anchor_ref": {k: scale[k] for k in ("transform", "ref_low", "ref_high")},
            "anchor_effective_n": len(vals),
            "fallback_reason": f"stratum effective n={len(vals)} < {MIN_ANCHOR_EFFECTIVE_N}",
            "anchor_version": REFERENCE_SCALES["version"]}


# ------------------------------- cohort ------------------------------------ #
def cohort_rank(feature: str, value: float, cohort_values: list[float],
                exclude_self: bool = True) -> dict:
    """Rank/percentile of this protein within the cohort. Amplitude features are
    barred (instrument-local); cohorts with < MIN_COHORT_N OTHER proteins are
    suppressed ('N too small').

    N excludes self (review item 8): we remove ONE occurrence of `value` from the
    pool so the percentile is a position among the OTHER proteins, not inflated by
    the protein ranking against itself. Percentile uses the same Hazen mid-rank
    convention as intrinsic (ties split evenly), so rank and percentile reconcile.

    Because low t50/lag means HIGH propensity, we also emit `direction` and a
    derived `propensity_percentile = 100 - feature_percentile` for those features,
    so callers never have to guess the sign."""
    if feature in AMPLITUDE_FEATURES:
        return {"suppressed": True,
                "reason": "amplitude/a.u. feature barred from cohort comparison "
                          "(instrument-local, not portable)"}
    vals = [v for v in (cohort_values or []) if v is not None and math.isfinite(v)]
    if exclude_self and value in vals:
        # drop exactly one occurrence of self -> N = number of OTHER proteins
        others = list(vals)
        others.remove(value)
    else:
        others = vals
    n = len(others)
    if n < MIN_COHORT_N:
        return {"suppressed": True, "reason": "N too small", "N": n,
                "N_excludes_self": True}
    # ordinal rank among OTHERS by raw feature value (1 = smallest feature value)
    rank = sum(1 for v in others if v < value) + 1
    feat_pct = round(_percentile_rank(value, others), 1)
    direction = _propensity_direction(feature)
    out = {"suppressed": False, "value": value, "rank": rank,
           "percentile": feat_pct, "N": n, "N_excludes_self": True,
           "direction": direction,
           "direction_note": ("feature percentile is on the raw feature scale; for "
                              "t50/lag-type features LOW value = HIGH propensity, so "
                              "use propensity_percentile for propensity ranking")}
    # derived propensity percentile (sign-corrected) for low-is-high features —
    # via the SAME shared helper intrinsic uses, so the two can never disagree.
    out["propensity_percentile"] = _to_propensity_percentile(feature, feat_pct)
    return out


# ------------------------------- surface ----------------------------------- #
def surface(gamma_record: dict | None) -> dict:
    """γ surface (PRIMARY where a concentration series exists). Reads the M4 dual-γ
    payload for this protein/series; otherwise marks the surface unavailable."""
    if not gamma_record:
        return {"available": False,
                "reason": "no concentration series — surface needs ≥3 concentrations"}
    reg = gamma_record.get("gamma_regression", {}) or {}
    glob = gamma_record.get("gamma_global", {}) or {}
    out = {
        "available": True,
        "gamma_regression": {"value": reg.get("gamma"), "ci": reg.get("gamma_ci"),
                             "reliable": reg.get("gamma_reliable")},
        "gamma_global": {"value": glob.get("gamma"), "ci": glob.get("gamma_ci")},
        "gamma_disagreement": gamma_record.get("disagreement", {}).get("disagree"),
        "feature_vs_driving_force": "t50 ∝ m^(−γ) across the concentration series",
    }
    # Pass through the window_of_validity from the γ record when present (the M6
    # design's surface block carries it; review item 12). Defensive: the current
    # gamma artifact does not emit it, so we only surface it if upstream adds it.
    if "window_of_validity" in gamma_record:
        out["window_of_validity"] = gamma_record.get("window_of_validity")
    return out


# --------------------------- sequence axis --------------------------------- #
def _assay_endpoint_class(assay: str | None) -> str:
    """Endpoint class of a kinetic assay for the sequence-match gate.

    Unknown/None/empty assay -> "unknown": it licenses NOTHING (we must never
    silently fall through to "generic", which would wrongly license AGGRESCAN /
    TANGO; review item A2). A known generic assay (turbidity/DLS/SLS/scatter/
    generic fluorescence) -> "generic". An amyloid-specific dye -> "amyloid"."""
    a = (assay or "").strip().lower()
    if not a:
        return "unknown"
    if any(k in a for k in _AMYLOID_ASSAYS):
        return "amyloid"
    if any(k in a for k in _GENERIC_ASSAYS):
        return "generic"        # turbidity / DLS / SLS / generic fluorescence etc.
    return "unknown"            # non-empty but unrecognised -> licenses nothing (A2)


def sequence_axis(uniprot: str | None, seq_lookup: dict, assay: str | None) -> dict:
    """De-conflated named sub-axes for this protein + the endpoint-match licensing of
    a sequence-vs-kinetics comparison (only an endpoint-matched predictor may be
    compared to this assay). The null disagreement rate is a documented deferral."""
    rec = (seq_lookup.get("by_uniprot", {}) or {}).get(uniprot) if uniprot else None
    absent = seq_lookup.get("absent_predictors", {})
    if not rec or not rec.get("sequence_predictors"):
        return {"available": False, "uniprot_id": uniprot,
                "reason": "no CPAD sequence-layer entry for this UniProt",
                "absent_predictors": absent}
    assay_class = _assay_endpoint_class(assay)
    subs = {}
    for name, sa in rec["sequence_predictors"].items():
        pcls = _PREDICTOR_ENDPOINT_CLASS.get(name, "generic")
        # An "unknown" assay class matches NOTHING (no predictor is class "unknown"),
        # so comparison_licensed is False for every predictor — by construction.
        endpoint_match = (assay_class != "unknown" and pcls == assay_class)
        subs[name] = {**sa, "endpoint_class": pcls,
                      "endpoint_match_to_assay": endpoint_match,
                      "comparison_licensed": bool(sa.get("available") and endpoint_match)}
    return {
        "available": True, "uniprot_id": uniprot, "protein_name": rec.get("protein_name"),
        "assay": assay, "assay_endpoint_class": assay_class,
        "n_peptides": rec.get("n_peptides"),
        "n_peptides_wt": rec.get("n_peptides_wt"),
        "n_peptides_total": rec.get("n_peptides_total"),
        "n_amyloid_peptides": rec.get("n_amyloid_peptides"),
        "peptide_length_range": rec.get("peptide_length_range"),
        "excluded_available_predictors": rec.get("excluded_available_predictors", []),
        "sub_axes": subs,
        "absent_predictors": absent,
        # An unknown assay licenses no comparison at all (auditable, not silent).
        "no_comparison_licensed_reason": (
            "assay endpoint class unknown — no sequence↔kinetics comparison licensed"
            if assay_class == "unknown" else None),
        "null_disagreement_rate": None,
        "null_disagreement_note": ("interpreting sequence↔kinetics disagreement against a "
                                   "null (predictor FP rate + assay detection limit) is a "
                                   "documented deferral — needs Service-C calibration"),
    }


# -------------------------- structure linkage ------------------------------ #
def structure_linkage(uniprot: str | None, seq_lookup: dict, max_list: int = 8) -> dict:
    """Associative CPAD/PDB structure-linkage panel. Recorded, never causal;
    fibril polymorphism is a first-class caveat."""
    rec = (seq_lookup.get("by_uniprot", {}) or {}).get(uniprot) if uniprot else None
    structures = (rec or {}).get("structures", []) if rec else []
    aprs = (rec or {}).get("aprs", []) if rec else []
    n_amyloid = sum(1 for s in structures if (s.get("amyloid") or "").lower() == "amyloid")
    return {
        "available": bool(structures or aprs),
        "uniprot_id": uniprot,
        "n_structures": len(structures),
        "n_amyloid_structures": n_amyloid,
        "n_aprs": len(aprs),
        "structures": structures[:max_list],
        "relationship": "associative_only",
        "caveats": ["fibril polymorphism: 'the structure of protein P' can be ill-posed",
                    "not a causal claim from a single structure"],
    }


# Caveat attached to every record: the t50 anchor/cohort is ASSAY-matched only.
_CONDITION_MATCH_CAVEAT = (
    "intrinsic/cohort are matched on ASSAY ENDPOINT ONLY; concentration / pH / "
    "temperature / agitation are NOT matched. Fully-licensed cross-protein ranking "
    "requires condition+assay matching (Service A) — this is an assay-only anchor.")


# ------------------------------ assembler ---------------------------------- #
def score_protein(uniprot, protein_name, assay, feature_value, stratum_values,
                  cohort_values, gamma_record, seq_lookup,
                  feature="t50", condition_vector=None,
                  feature_status="point") -> dict:
    """Assemble the full §3-M6 Propensity payload for one protein, on one portable
    feature (default t50). `stratum_values`/`cohort_values` must be POINT values only
    and `stratum_values` must EXCLUDE this protein (leakage-free).

    `feature_status` is the t50 censoring status. A right-censored "lower_bound" is a
    ≥ inequality, NOT a rankable point: such proteins still emit a record but with
    intrinsic AND cohort suppressed (review item C9)."""
    portable = feature in PORTABLE_FEATURES
    censored = (feature_status == "lower_bound")
    if censored:
        # right-censored lower bound is not a point — cannot be anchored/ranked
        suppressed = {"suppressed": True,
                      "reason": "t50 right-censored (lower bound) — not rankable as a point"}
        intrinsic, cohort = dict(suppressed), dict(suppressed)
    else:
        intrinsic = (intrinsic_anchor(feature, feature_value, stratum_values)
                     if (portable and feature_value is not None) else
                     {"suppressed": True, "reason": "non-portable or missing feature"})
        cohort = (cohort_rank(feature, feature_value, cohort_values)
                  if feature_value is not None else
                  {"suppressed": True, "reason": "missing feature"})
    return {
        "uniprot_id": uniprot,
        "protein": protein_name,
        "assay": assay,
        "scored_feature": feature,
        "feature_portable": portable,
        "feature_value": feature_value,
        "feature_status": feature_status,
        "feature_direction": _propensity_direction(feature),
        # t50 anchor is assay-matched only (condition vector NOT matched) — item 10
        "condition_match": "assay_only",
        "condition_match_caveat": _CONDITION_MATCH_CAVEAT,
        "propensity": {
            "surface": surface(gamma_record),
            "intrinsic": intrinsic,
            "cohort": cohort,
        },
        "sequence_axis": sequence_axis(uniprot, seq_lookup, assay),
        "structure_linkage": structure_linkage(uniprot, seq_lookup),
        "condition_vector": condition_vector,
        "portability_note": ("intrinsic/cohort computed on a time-domain feature; "
                             "amplitude/a.u. features are barred (instrument-local)"),
    }


# --------------------------------- I/O ------------------------------------- #
def _load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _protein_t50(feat_rec: dict):
    """Return (t50_value, t50_status) usable for propensity, or (None, None).

    Accepts POINT and right-censored LOWER_BOUND t50 (left-biased excluded as
    unreliable for ranking). The CALLER decides pooling: only POINT t50 may enter
    the anchor/cohort pools, because a right-censored "lower_bound" is a ≥ inequality
    (we know only t50 >= v) and must not be pooled as if it were a point (item C9)."""
    if feat_rec.get("status") != "ok":
        return None, None
    status = feat_rec.get("t50_status")
    if status not in ("point", "lower_bound"):
        return None, None
    return (feat_rec.get("features") or {}).get("t50"), status


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="PRISE M6 propensity + sequence axis")
    proc = root / "data" / "processed"
    ap.add_argument("--triaged", type=Path, default=proc / "curves_triaged.jsonl")
    ap.add_argument("--features", type=Path, default=proc / "features.jsonl")
    ap.add_argument("--gamma", type=Path, default=proc / "gamma.jsonl")
    ap.add_argument("--sequence", type=Path, default=proc / "sequence_structure.json")
    ap.add_argument("--output", type=Path, default=proc / "propensity.jsonl")
    args = ap.parse_args(argv)
    if not args.features.exists():
        raise SystemExit(f"{args.features} not found (run M4 features first)")

    seq_lookup = (json.loads(args.sequence.read_text(encoding="utf-8"))
                  if args.sequence.exists() else {"by_uniprot": {}, "absent_predictors": {}})
    # series_id -> (uniprot, protein, assay, construct, condition_vector)
    meta = {}
    # concentration_series_id -> assay. gamma.jsonl is keyed by concentration series
    # and carries no assay of its own, so this is how a γ record is resolved to the
    # assay it was measured by (see the M6_GAMMA_JOIN policy). Measured: all 52 γ
    # records resolve to exactly one assay; none spans two.
    conc_assay = {}
    for s in _load_jsonl(args.triaged):
        cv = s.get("condition_vector", {}) or {}
        csid = s.get("concentration_series_id")
        if csid:
            conc_assay.setdefault(csid, cv.get("assay_type"))
        meta[s.get("series_id")] = {
            "uniprot": s.get("uniprot_id"), "protein": s.get("protein_id") or s.get("protein"),
            "assay": cv.get("assay_type"),
            # construct_id distinguishes a mutant construct from WT — added to the
            # grouping key so engineered mutants are NOT merged into the WT protein
            # (review item 10). Missing construct -> "Wild Type" (the corpus default).
            "construct": cv.get("construct_id") or "Wild Type",
            "cv": cv,
            "pmid": ((s.get("source_study") or {}).get("pmid")),
        }
    feats = {f.get("series_id"): f for f in _load_jsonl(args.features)}
    # The GOVERNED grouping key (join-policy-1.0): (uniprot, construct, assay).
    #
    # It previously omitted ASSAY, so a protein×construct measured by two assays was
    # pooled into ONE median and then labelled with whichever assay happened to be
    # seen first. Measured on the live corpus, 4 of 133 groups pooled across assays;
    # the sharpest was P01160 Wild Type, published as assay "CongoRed" carrying t50
    # 44.851 h — the ThT median — while that protein's CongoRed data sits at 586.870 h,
    # 13x away. A wrong NUMBER in a wrong stratum under a wrong label, not merely a
    # wrong attachment. It is also why `k114` reached no M6 record at all: those
    # curves were absorbed into ThT-labelled groups.
    #
    # Read from the policy, not inlined, so the retained `previous` is a real revert.
    group_key_fields = tuple(_applied_policy("M6_GROUP_KEY",
                                             ["uniprot", "construct", "assay"]))
    by_prot = defaultdict(lambda: {"t50_point": [], "n_lower_bound": 0,
                                   "uniprot": None, "assay": None, "protein": None,
                                   "construct": None, "cv": None,
                                   # the studies backing this group -- needed for
                                   # LEAVE-ONE-STUDY-OUT anchoring, below
                                   "studies": set()})
    for sid, f in feats.items():
        m = meta.get(sid)
        if not m or not m.get("uniprot"):
            continue
        t50, status = _protein_t50(f)
        if t50 is None:
            continue
        key = tuple(m.get(field) for field in group_key_fields)
        d = by_prot[key]
        # Only POINT t50 enters the pool. A lower_bound is a ≥ inequality and must
        # not be pooled as a point (review item C9); we just count its presence.
        if status == "point":
            d["t50_point"].append(t50)
        else:                                  # status == "lower_bound"
            d["n_lower_bound"] += 1
        d["uniprot"] = d["uniprot"] or m["uniprot"]
        # With `assay` in the group key every member of a group shares it, so this
        # first-wins assignment now resolves to the group's OWN assay. It is kept in
        # this form deliberately: if the policy is reverted to the 2-field key it
        # restores the previous (arbitrary-label) behaviour exactly, which is what a
        # revert path has to do.
        d["assay"] = d["assay"] or m["assay"]
        d["protein"] = d["protein"] or m["protein"]
        d["construct"] = d["construct"] or m["construct"]
        d["cv"] = d["cv"] or m["cv"]
        if m.get("pmid"):
            d["studies"].add(m["pmid"])

    # group-level summary value = median POINT t50 (portable)
    def _median(v):
        v = sorted(v)
        n = len(v)
        return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])

    prot_value = {key: _median(d["t50_point"])
                  for key, d in by_prot.items() if d["t50_point"]}
    # γ by protein name (from the M4 dual-γ artifact, if present). NOTE: gamma.jsonl
    # carries a protein *name* not a UniProt id, so the surface join is name-based
    # (a documented limitation — a UniProt key on the γ artifact would be more robust).
    # ...and, since join-policy-1.0, ALSO by the assay that concentration series was
    # measured by. The name-based join was harmless only while one protein name mapped
    # to one group; once M6_GROUP_KEY splits groups by assay a single name maps to
    # several, and an assay-blind lookup would hand the same γ to all of them. γ is
    # the scaling exponent of a concentration series measured by a particular assay,
    # so that would be the very cross-assay contamination this change removes.
    gamma_join = _applied_policy("M6_GAMMA_JOIN", "protein_name_and_assay")
    gamma_by_protein = {}
    for g in _load_jsonl(args.gamma):
        name = g.get("protein")
        if not name:
            continue
        if gamma_join == "protein_name_and_assay":
            gamma_by_protein[(name, conc_assay.get(g.get("concentration_series_id")))] = g
        else:
            gamma_by_protein[name] = g          # retained previous behaviour

    def _gamma_for(protein, assay):
        """The γ record for this group, assay-matched when the policy says so.
        Returns None rather than a foreign-assay record when there is no match."""
        if gamma_join == "protein_name_and_assay":
            return gamma_by_protein.get((protein, assay))
        return gamma_by_protein.get(protein)

    # assay strata of per-group median POINT t50 (leakage-free corpus anchoring).
    # Keyed by the grouping key, so the corpus-anchor pool contains one entry per
    # construct (a mutant and its WT are distinct pool members) and — since
    # join-policy-1.0 put ASSAY in that key — `d["assay"]` is now the group's own
    # assay rather than a first-wins label, so a group can no longer be filed in
    # the stratum of an assay it was not measured by.
    strata = defaultdict(dict)            # assay -> {group_key: value}
    for key, d in by_prot.items():
        if key in prot_value:
            strata[d["assay"]][key] = prot_value[key]

    # IMPLEMENTED (join-policy-1.2): the corpus anchor is leave-one-STUDY-out.
    # It used to be leave-one-GROUP-out, and this comment used to argue the
    # residual leakage was small "because proteins rarely co-occur across
    # studies". That premise is true -- only 36 of 164 groups draw on more than
    # one study -- but it is NOT the mechanism. The leak runs the other way: one
    # study contributing SEVERAL groups to one stratum, so a group was ranked
    # against measurements sharing its own lab, protocol and batch. Measured that
    # way, 151 of 164 groups (92.1%) leaked. See the exclusion in the writer loop
    # below and the M6_ANCHOR_LEAKAGE policy record.

    n = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for key, d in by_prot.items():
            assay = d["assay"]
            grec = _gamma_for(d["protein"], assay)
            if key not in prot_value:
                # CENSORED-ONLY group: every t50 is a right-censored lower bound.
                # Still emit a record, but intrinsic/cohort suppressed (item C9).
                lb = d["t50_point"][-1] if d["t50_point"] else None  # always None here
                rec = score_protein(d["uniprot"], d["protein"], assay, lb, [], [],
                                    grec, seq_lookup, feature="t50",
                                    condition_vector=d["cv"], feature_status="lower_bound")
                out.write(json.dumps(rec) + "\n")
                n += 1
                continue
            value = prot_value[key]
            stratum = strata[assay]
            # LEAVE-ONE-STUDY-OUT (join-policy... governed as M6_ANCHOR_LEAKAGE).
            # Excluding only THIS group left the anchor pool full of other groups
            # from the SAME study, which share lab, protocol and batch effects, so
            # the "corpus" reference was not independent of the value being ranked
            # against it. MEASURED: 151 of 164 groups (92.1%) shared >=1 study with
            # another group in their own stratum. The README previously argued the
            # leakage was small because "proteins rarely co-occur across studies" --
            # true (only 36 groups draw on >1 study) but not the mechanism: the
            # leak is one STUDY contributing several groups to one stratum.
            mine = d.get("studies") or set()
            if _applied_policy("M6_ANCHOR_LEAKAGE",
                               "leave_one_study_out") == "leave_one_study_out":
                stratum_excl = [v for k, v in stratum.items()
                                if k != key and not (
                                    (by_prot[k].get("studies") or set()) & mine)]
            else:
                stratum_excl = [v for k, v in stratum.items() if k != key]
            # the COHORT is a different object: an explicit "rank within the cohort
            # under analysis", not a leakage-free reference, and cohort_rank
            # self-excludes internally. It is deliberately left alone.
            cohort_vals = list(stratum.values())
            rec = score_protein(d["uniprot"], d["protein"], assay, value, stratum_excl,
                                cohort_vals, grec, seq_lookup,
                                feature="t50", condition_vector=d["cv"],
                                feature_status="point")
            out.write(json.dumps(rec) + "\n")
            n += 1
    # summary
    with_seq = with_struct = corpus_anchored = surface_n = censored_only = 0
    for rec in _load_jsonl(args.output):
        with_seq += bool(rec["sequence_axis"].get("available"))
        with_struct += bool(rec["structure_linkage"].get("available"))
        corpus_anchored += (rec["propensity"]["intrinsic"].get("anchor_kind") == "corpus")
        surface_n += bool(rec["propensity"]["surface"].get("available"))
        censored_only += (rec.get("feature_status") == "lower_bound")
    print(f"[M6] wrote {args.output}  ({n} proteins)")
    print(json.dumps({
        "n_proteins": n,
        "with_sequence_axis": with_seq,
        "with_structure_linkage": with_struct,
        "intrinsic_corpus_anchored": corpus_anchored,
        "intrinsic_reference_scale": n - corpus_anchored - censored_only,
        "censored_only_suppressed": censored_only,
        "with_gamma_surface": surface_n,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
