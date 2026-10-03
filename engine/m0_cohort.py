"""
PRISE — Module M0: Dataset Selection & Cohort Assembly
======================================================

The plan's ENTRY POINT (PRISE_DESIGN.md §3-M0), previously UNBUILT: the product
only ever analysed ONE curve at a time. M0 lets the user browse a protein's real
CPAD datasets and select **one OR several** to analyse together, then ROUTES the
assembled set to the scientifically-licensed analysis — after a Service-A
comparability gate that runs BEFORE any fitting.

WHY M0 exists (§3-M0). CPAD 2.0 holds many datasets per protein across different
conditions (concentration, pH, temperature, assay, construct). What is *licensed*
depends entirely on what the user assembled and on what VARIES across it:

  | assembled selection (comparable)              | data_mode            | unlocks
  |-----------------------------------------------|----------------------|---------
  | 1 dataset                                     | single_curve         | per-curve descriptive + features (M4/M5)
  | ≥2 with IDENTICAL condition_vector            | replicates           | random-effects meta (M1) + stochastic-nucleation scatter
  | concentration VARIES, else matched (≥3 conc)  | concentration_series | dual-γ (M4) + shared-rate ODE (global_fit)
  | pH or temp VARIES, concentration fixed        | condition_series     | feature-vs-condition SURFACE (no γ)
  | >1 condition axis co-varies                   | confounded           | REFUSED → descriptive-only + confound flag (Invariant 19)

VALIDITY GUARDS M0 ENFORCES BEFORE ANYTHING IS FIT (§3-M0, §6 Service A):
  * COMPARABILITY GATE — the assembled set must share protein/uniprot, construct_id
    and assay_type. A mismatch is a HARD confound flag: mechanism/scaling REFUSED,
    only per-curve descriptive allowed, and the mismatching field(s) named. MISSING
    (unknown) comparability-critical fields degrade confidence + widen — never a
    silent merge.
  * CONFOUND DETECTION — if more than one condition axis varies across the selection,
    γ/mechanism is confounded (a γ from a series where pH also drifts is not a clean
    reaction-order signal); the set is routed to descriptive/surface and the
    co-varying axes are reported (Invariant 19).
  * RANGE-BOUNDED INFERENCE — `window_of_validity` reports the MEASURED condition
    ranges; inference is not licensed beyond them (no extrapolation, no fabricated
    curve for an unmeasured condition).

WHAT M0 IS (honest framing). M0 ORCHESTRATES; it does not re-implement inference.
It imports and calls the existing, tested engine:
    m1_ingest.triage_series   (triage, if a raw selection sneaks in)
    m2_fit.fit_curve          (per-curve fit, single_curve route)
    m4_features.extract_features / dual_gamma   (features; concentration-series γ)
    m5_classify.classify_curve (descriptive regime, per curve)
    global_fit.fit_global_series (shared-rate Knowles/Cohen ODE, concentration route)
    pooling.replicate_meta     (replicate random-effects + stochastic-nucleation)
It NEVER modifies those modules. Everything here is pure, deterministic,
JSON-serialisable, and NEVER raises — a bad/empty selection degrades to a flagged
minimal `cohort_result`.

Usage:
    from m0_cohort import assemble_cohort
    result = assemble_cohort(selected_triaged_series)     # list[dict] -> cohort_result

    python engine/m0_cohort.py    # demo over a REAL concentration series
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

M0_VERSION = "m0-cohort-1.0"

# Comparability-critical fields (§3-M0 / §6 Service A). A DIFFERENCE across the
# selection in any of these is a HARD confound (mechanism/scaling refused). A
# MISSING value is a degrade (widen + flag), never a silent merge.
_COMPARABILITY_FIELDS = ("protein", "construct_id", "assay_type")

# Condition axes M0 watches for co-variation (the confound test). concentration is
# the mechanism axis (its variation UNLOCKS γ when it is the ONLY axis moving);
# pH/temperature are the surface axes; agitation/seeding are hard mechanism branches
# (§3-M5) and are treated as comparability-critical for confounding too.
_CONDITION_AXES = ("concentration_uM", "pH", "temperature_C", "agitation", "seeded")

# Rounding granularity so trivial float noise / digitisation does not fake a
# condition axis as "varying" (Service-A widening, mirrors pooling.py).
_CONC_ROUND = 4
_PH_ROUND = 1
_TEMP_ROUND = 0

# A concentration_series needs ≥3 DISTINCT concentrations to support a γ (§3-M4).
MIN_CONC_FOR_SERIES = 3


# --------------------------------------------------------------------------- #
#  Condition extraction (pure, tolerant of missing fields)
# --------------------------------------------------------------------------- #
def _cv(series: dict) -> dict:
    return series.get("condition_vector") or {}


def _protein(series: dict):
    """Protein identity: prefer uniprot (stable) then the CPAD protein name. A
    None here means the comparability field is UNKNOWN (degrade), not mismatched."""
    return series.get("uniprot_id") or series.get("protein_id") or _cv(series).get("protein")


def _concentration(series: dict):
    c = (_cv(series).get("concentration") or {}).get("value_uM")
    return float(c) if isinstance(c, (int, float)) else None


def _round_or_none(v, ndigits):
    return round(float(v), ndigits) if isinstance(v, (int, float)) else None


def _axis_values(series: dict) -> dict:
    """The condition-axis coordinates M0 watches, rounded so float noise does not
    fake variation. None = the field is UNKNOWN for this curve."""
    cv = _cv(series)
    return {
        "concentration_uM": _round_or_none(_concentration(series), _CONC_ROUND),
        "pH": _round_or_none(cv.get("pH"), _PH_ROUND),
        "temperature_C": _round_or_none(cv.get("temperature_C"), _TEMP_ROUND),
        # agitation / seeded are categorical (None means unknown, not "off")
        "agitation": cv.get("agitation"),
        "seeded": cv.get("seeded"),
    }


def _comparability_values(series: dict) -> dict:
    cv = _cv(series)
    return {
        "protein": _protein(series),
        "construct_id": cv.get("construct_id"),
        "assay_type": cv.get("assay_type"),
    }


def _member_record(series: dict) -> dict:
    """Compact, JSON-serialisable member descriptor for the cohort_result."""
    cv = _cv(series)
    return {
        "series_id": series.get("series_id"),
        "protein_id": series.get("protein_id"),
        "uniprot_id": series.get("uniprot_id"),
        "conditions": {
            "concentration_uM": _concentration(series),
            "pH": cv.get("pH"),
            "temperature_C": cv.get("temperature_C"),
            "assay_type": cv.get("assay_type"),
            "assay_reports_mass": cv.get("assay_reports_mass"),
            "construct_id": cv.get("construct_id"),
            "agitation": cv.get("agitation"),
            "seeded": cv.get("seeded"),
        },
        "censoring_class": (series.get("m1") or {}).get("censoring_class")
        or series.get("censoring_class"),
        "fittability_class": (series.get("m1") or {}).get("fittability_class")
        or series.get("fittability_class"),
    }


# --------------------------------------------------------------------------- #
#  Comparability gate (Service A discipline — runs BEFORE any analysis)
# --------------------------------------------------------------------------- #
def check_comparability(members: list[dict]) -> dict:
    """Service-A comparability gate (§6). Require the SAME protein/uniprot,
    construct_id and assay_type across the selection.

    Returns {ok, mismatches[], degraded_fields[], confound_flag, ...}. A genuine
    DIFFERENCE in a comparability field -> a hard mismatch (mechanism/scaling
    refused). A MISSING value (None) with the rest agreeing -> a degraded field
    (widen + flag), never a silent merge across an unknown."""
    mismatches = []
    degraded = []
    for field in _COMPARABILITY_FIELDS:
        seen = defaultdict(list)          # value -> [series_id, ...]
        n_missing = 0
        for m in members:
            v = _comparability_values(m).get(field)
            if v is None:
                n_missing += 1
            else:
                seen[v].append(m.get("series_id"))
        distinct = sorted(seen.keys(), key=lambda x: str(x))
        if len(distinct) > 1:
            # a genuine mismatch on a comparability-critical field -> HARD confound
            mismatches.append({
                "field": field,
                "distinct_values": distinct,
                "value_to_series": {str(k): v for k, v in seen.items()},
                "severity": "hard",
                "note": (f"selection mixes {len(distinct)} distinct {field} values — "
                         "Service-A comparability violated; mechanism/scaling refused, "
                         "per-curve descriptive only"),
            })
        elif n_missing > 0:
            # some/all unknown but no contradiction among the known -> DEGRADE, widen
            degraded.append({
                "field": field,
                "n_missing": n_missing,
                "n_total": len(members),
                "known_value": (distinct[0] if distinct else None),
                "note": (f"{field} unknown for {n_missing}/{len(members)} members — "
                         "confidence degraded and window widened; never silently merged"),
            })
    ok = not mismatches
    return {
        "ok": bool(ok),
        "mismatches": mismatches,
        "mismatched_fields": [m["field"] for m in mismatches],
        "degraded_fields": degraded,
        "confound_flag": (not ok),
        "gate": "service_a_comparability",
        "note": ("comparability OK — same protein/construct/assay across the selection"
                 if ok else
                 "HARD comparability confound — mechanism/scaling REFUSED, only "
                 "per-curve descriptive licensed; mismatching field(s): "
                 + ", ".join(m["field"] for m in mismatches)),
    }


# --------------------------------------------------------------------------- #
#  Data-mode detection (from what VARIES across the comparable selection)
# --------------------------------------------------------------------------- #
def _varying_axes(members: list[dict]) -> dict:
    """Which condition axes VARY across the selection, and each axis's distinct
    values. Unknown (None) values do NOT count as a distinct level (they degrade,
    handled by the comparability gate), so an all-unknown axis is 'not varying'."""
    axis_levels = {ax: set() for ax in _CONDITION_AXES}
    for m in members:
        av = _axis_values(m)
        for ax in _CONDITION_AXES:
            v = av.get(ax)
            if v is not None:
                axis_levels[ax].add(v)
    varying = {ax: sorted(levels, key=lambda x: (str(type(x)), str(x)))
               for ax, levels in axis_levels.items() if len(levels) > 1}
    return varying


def detect_data_mode(members: list[dict], comparability: dict) -> dict:
    """Route the (already comparability-checked) selection to a data_mode from what
    VARIES across it. See the module table. Returns {data_mode, reason,
    varying_axes, n_distinct_concentrations, confound_axes}."""
    n = len(members)
    if n == 0:
        return {"data_mode": "empty", "reason": "no datasets selected",
                "varying_axes": {}, "confound_axes": []}

    # A HARD comparability mismatch dominates: the set is confounded by construct/assay
    # (a different KIND of confound than co-varying conditions, but the same refusal).
    if not comparability.get("ok", True):
        return {
            "data_mode": "confounded",
            "reason": ("comparability confound: selection mixes "
                       + ", ".join(comparability.get("mismatched_fields", []))
                       + " — mechanism/scaling refused"),
            "varying_axes": _varying_axes(members),
            "confound_axes": comparability.get("mismatched_fields", []),
            "confound_kind": "comparability",
        }

    if n == 1:
        return {"data_mode": "single_curve",
                "reason": "one dataset selected",
                "varying_axes": {}, "confound_axes": []}

    varying = _varying_axes(members)
    conc_levels = sorted({_axis_values(m)["concentration_uM"] for m in members
                          if _axis_values(m)["concentration_uM"] is not None})
    n_distinct_conc = len(conc_levels)

    # ≥2 datasets, NOTHING varies (identical condition_vector incl. concentration)
    # -> genuine REPLICATES.
    if not varying:
        return {"data_mode": "replicates",
                "reason": (f"{n} datasets with identical condition_vector "
                           "(same concentration/pH/temp/construct/assay) — replicates"),
                "varying_axes": {}, "n_distinct_concentrations": n_distinct_conc,
                "confound_axes": []}

    varying_axes = set(varying.keys())
    conc_varies = "concentration_uM" in varying_axes
    non_conc_varying = varying_axes - {"concentration_uM"}

    # MORE THAN ONE axis co-varies -> confounded (Invariant 19). This includes
    # concentration AND pH, pH AND temperature, concentration AND agitation, etc.
    if len(varying_axes) > 1:
        return {
            "data_mode": "confounded",
            "reason": ("more than one condition axis co-varies ("
                       + ", ".join(sorted(varying_axes))
                       + ") — γ/mechanism confounded; descriptive/surface only "
                         "(Invariant 19)"),
            "varying_axes": varying,
            "n_distinct_concentrations": n_distinct_conc,
            "confound_axes": sorted(varying_axes),
            "confound_kind": "co_varying_conditions",
        }

    # EXACTLY ONE axis varies:
    if conc_varies:
        # concentration VARIES, everything else matched.
        if n_distinct_conc >= MIN_CONC_FOR_SERIES:
            return {"data_mode": "concentration_series",
                    "reason": (f"concentration varies across {n_distinct_conc} distinct "
                               "levels, all other conditions matched — γ / mechanism "
                               "constraint licensed"),
                    "varying_axes": varying,
                    "n_distinct_concentrations": n_distinct_conc,
                    "confound_axes": []}
        # concentration varies but <3 levels -> not enough for γ; treat as a small
        # condition set (descriptive/surface), flagged.
        return {"data_mode": "condition_series",
                "reason": (f"concentration varies but only {n_distinct_conc} distinct "
                           f"level(s) (<{MIN_CONC_FOR_SERIES}) — too few for a γ; "
                           "feature-vs-concentration surface only"),
                "varying_axes": varying,
                "n_distinct_concentrations": n_distinct_conc,
                "confound_axes": [],
                "surface_axis": "concentration_uM"}

    # pH or temperature varies, concentration fixed, otherwise matched -> surface.
    surf_axis = sorted(non_conc_varying)[0]
    return {"data_mode": "condition_series",
            "reason": (f"{surf_axis} varies (concentration fixed, else matched) — "
                       "propensity/feature SURFACE over the measured range; no γ"),
            "varying_axes": varying,
            "n_distinct_concentrations": n_distinct_conc,
            "confound_axes": [],
            "surface_axis": surf_axis}


# --------------------------------------------------------------------------- #
#  Window of validity (measured ranges — inference not licensed beyond)
# --------------------------------------------------------------------------- #
def window_of_validity(members: list[dict]) -> dict:
    """The MEASURED condition ranges across the selection (§3-M0 range-bounded
    inference). Inference is not licensed beyond these; a missing axis is reported
    as unknown, never extrapolated."""
    def _range(vals):
        vv = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
        if not vv:
            return None
        return {"min": float(min(vv)), "max": float(max(vv)), "n_distinct": len(set(vv))}

    concs = [_concentration(m) for m in members]
    phs = [_cv(m).get("pH") for m in members]
    temps = [_cv(m).get("temperature_C") for m in members]
    assays = sorted({_cv(m).get("assay_type") for m in members if _cv(m).get("assay_type")})
    constructs = sorted({_cv(m).get("construct_id") for m in members
                         if _cv(m).get("construct_id")})
    return {
        "concentration_uM": _range(concs),
        "pH": _range(phs),
        "temperature_C": _range(temps),
        "assays": assays,
        "constructs": constructs,
        "note": ("inference is licensed only within these MEASURED ranges; PRISE does "
                 "not extrapolate or fabricate a curve for an unmeasured condition (§3-M0)"),
    }


# --------------------------------------------------------------------------- #
#  Routed analyses (each imports the existing engine; none re-implemented here)
# --------------------------------------------------------------------------- #
def _route_single_curve(members: list[dict]) -> dict:
    """single_curve -> the existing per-curve result (fit + features + descriptive
    regime). Points to M4/M5. Never raises."""
    s = members[0]
    out = {"route": "single_curve",
           "points_to": "M4 (features) / M5 (regime) — single-curve descriptive",
           "series_id": s.get("series_id")}
    try:
        from m2_fit import fit_curve
        from m4_features import extract_features
        from m5_classify import classify_curve
        fit = fit_curve(s)
        feat = extract_features(s, fit_result=fit)
        cls = classify_curve(s, fit_result=fit)
        out.update({
            "status": "ok",
            "best_model": fit.get("best_by_aicc"),
            "features": feat.get("features"),
            "t50_status": feat.get("t50_status"),
            "descriptive_regime": (cls.get("descriptive_regime") or {}).get("regime"),
            "mechanistic_licensed": (cls.get("mechanistic") or {}).get(
                "mechanistic_inference_licensed"),
            "information_note": ("single curve: a γ scaling exponent needs a "
                                 "≥3-concentration series; mechanism not resolvable here"),
        })
    except Exception as exc:
        out.update({"status": "degraded", "reason": f"{type(exc).__name__}: {exc}"})
    return out


def _route_replicates(members: list[dict]) -> dict:
    """replicates -> the §3-M1 REPLICATE random-effects meta-analysis
    (pooling.replicate_meta): pooled t50 + tau2_between + the retained
    stochastic-nucleation t50-scatter signal. This is the M1 deliverable, now
    reachable through M0. Never raises."""
    out = {"route": "replicates",
           "points_to": "M1 replicate random-effects meta-analysis (pooling.py)"}
    try:
        from m2_fit import fit_curve
        from m4_features import extract_features
        from pooling import replicate_meta, _to_scale, _curve_sampling_sd

        # Build the poolable per-curve t50 units the replicate meta expects. We fit
        # each replicate individually (§3-M1: replicates fit individually, combined
        # by random effects — never curve-averaged) and read the point t50.
        units = []
        excluded = []
        for s in members:
            fit = fit_curve(s)
            feat = extract_features(s, fit_result=fit)
            t50 = (feat.get("features") or {}).get("t50")
            status = feat.get("t50_status")
            if feat.get("status") != "ok" or t50 is None or t50 <= 0 or status != "point":
                excluded.append({"series_id": s.get("series_id"),
                                 "reason": f"t50 not a usable point (status={status})"})
                continue
            theta = _to_scale(t50, "log10")
            sd, src = _curve_sampling_sd(feat, "t50", "log10")
            units.append({
                "series_id": s.get("series_id"),
                "theta_hat": theta, "raw_value": float(t50),
                "s2": float(sd) ** 2, "s2_source": src,
                "rep_str": None, "rep_dict": None,
                "concentration_uM": _concentration(s),
            })
        out["n_usable_replicates"] = len(units)
        out["excluded"] = excluded
        if len(units) < 2:
            out.update({"status": "insufficient_data",
                        "reason": f"only {len(units)} replicate(s) with a point t50 "
                                  "(need >=2 for a between-replicate variance)"})
            return out
        meta = replicate_meta(units, "t50", "log10")
        out.update({
            "status": "ok",
            "pooled_t50": meta.get("pooled_value"),
            "pooled_se_log10": meta.get("se"),
            "tau2_between": meta.get("tau2_between"),
            "pooled_se_lt_individual": meta.get("pooled_se_lt_individual"),
            "mean_individual_sampling_sd": meta.get("mean_individual_sampling_sd"),
            "stochastic_nucleation_signal": meta.get("stochastic_nucleation_signal"),
            "stochastic_nucleation_flag": meta.get("stochastic_nucleation_flag"),
            "replicate_meta": meta,
        })
    except Exception as exc:
        out.update({"status": "degraded", "reason": f"{type(exc).__name__}: {exc}"})
    return out


def _route_concentration_series(members: list[dict], series_meta: dict,
                                run_global_ode: bool = True,
                                b_reg: int = 200, b_glob: int = 60) -> dict:
    """concentration_series -> the mechanism-constraint path: dual-γ over the members
    (gamma_regression + gamma_global + disagreement) AND, when computable, the
    shared-rate Knowles/Cohen ODE global fit (reaction orders + γ_mechanistic +
    identifiability). Never raises.

    `run_global_ode` gates the (slow, stiff-ODE) global fit — default on for the
    scientifically-complete result; a caller under a latency budget can disable it.
    `b_reg`/`b_glob` are the dual-γ bootstrap budgets (a web caller lowers them for
    latency; the point γ estimates are unchanged, only the CI resolution)."""
    out = {"route": "concentration_series",
           "points_to": "M4 dual-γ + global_fit shared-rate ODE (mechanism constraint)"}
    try:
        from m4_features import dual_gamma
        dg = dual_gamma(series_meta, members, B_reg=b_reg, B_glob=b_glob)
        out["status"] = "ok"
        out["dual_gamma"] = dg
        out["gamma_regression"] = dg.get("gamma_regression")
        out["gamma_global"] = dg.get("gamma_global")
        out["disagreement"] = dg.get("disagreement")
    except Exception as exc:
        out.update({"status": "degraded", "reason": f"{type(exc).__name__}: {exc}"})

    if run_global_ode:
        try:
            from global_fit import fit_global_series
            gamma_hint = ((out.get("gamma_global") or {}).get("gamma"))
            gf = fit_global_series(members, meta=series_meta,
                                   gamma_hint=gamma_hint if isinstance(gamma_hint, (int, float))
                                   else None)
            out["global_ode_fit"] = {
                "status": gf.get("status"),
                "best_mechanism": gf.get("best_mechanism"),
                "reaction_orders": gf.get("reaction_orders"),
                "gamma_mechanistic": gf.get("gamma_mechanistic"),
                "gamma_mechanistic_formula": gf.get("gamma_mechanistic_formula"),
                "mechanism_aicc_ranking": gf.get("mechanism_aicc_ranking"),
                "sloppy": gf.get("sloppy"),
                "identifiability": gf.get("identifiability"),
                "honesty_note": gf.get("honesty_note"),
            }
        except Exception as exc:
            out["global_ode_fit"] = {"status": "degraded",
                                     "reason": f"{type(exc).__name__}: {exc}"}
    else:
        out["global_ode_fit"] = {"status": "skipped",
                                 "reason": "run_global_ode=False (latency budget)"}
    out["mechanism_note"] = ("γ constrains a combination of reaction orders "
                             "(many-to-one onto mechanism); the single-mechanism call "
                             "is M5's, gated on curve shape + agitation/seeding")
    return out


def _route_condition_series(members: list[dict], surface_axis: str) -> dict:
    """condition_series -> a propensity/feature SURFACE over the measured pH/temp
    (or small-N concentration) range. Reports the feature vs the varying condition;
    NO γ (a single condition axis over a non-concentration variable is not a scaling
    exponent). Never raises."""
    out = {"route": "condition_series",
           "surface_axis": surface_axis,
           "points_to": "M6 propensity surface (feature vs condition; NO γ)"}
    try:
        from m2_fit import fit_curve
        from m4_features import extract_features
        surf = []
        for s in members:
            av = _axis_values(s)
            cond_value = av.get(surface_axis)
            fit = fit_curve(s)
            feat = extract_features(s, fit_result=fit)
            feats = feat.get("features") or {}
            surf.append({
                "series_id": s.get("series_id"),
                "condition_value": cond_value,
                "condition_axis": surface_axis,
                "t50": feats.get("t50"),
                "t50_status": feat.get("t50_status"),
                "lag_time": feats.get("lag_time"),
                "max_rate": feats.get("max_rate"),
                "transition_sharpness": feats.get("transition_sharpness"),
                "descriptive_status": feat.get("status"),
            })
        surf.sort(key=lambda r: (r["condition_value"] is None, r["condition_value"]))
        out.update({
            "status": "ok",
            "surface": surf,
            "n_points": len(surf),
            "surface_note": (f"feature values vs {surface_axis} over the MEASURED range; "
                             "no scaling exponent is licensed for a non-concentration axis "
                             "(and the range bounds validity — §3-M0)"),
        })
    except Exception as exc:
        out.update({"status": "degraded", "reason": f"{type(exc).__name__}: {exc}"})
    return out


def _route_confounded(members: list[dict], mode: dict, comparability: dict) -> dict:
    """confounded -> REFUSE mechanism/scaling; emit descriptive-only per curve + a
    prominent confound flag naming the co-varying axes (Invariant 19). Never raises."""
    out = {
        "route": "confounded",
        "mechanism_refused": True,
        "confound_axes": mode.get("confound_axes", []),
        "confound_kind": mode.get("confound_kind"),
        "refusal_reason": mode.get("reason"),
        "points_to": "descriptive-only per curve (Invariant 19)",
    }
    # descriptive-only per curve (NO γ, NO mechanism, NO cross-curve scaling)
    per_curve = []
    try:
        from m2_fit import fit_curve
        from m5_classify import classify_curve
        for s in members:
            try:
                fit = fit_curve(s)
                cls = classify_curve(s, fit_result=fit)
                per_curve.append({
                    "series_id": s.get("series_id"),
                    "conditions": _member_record(s)["conditions"],
                    "descriptive_regime": (cls.get("descriptive_regime") or {}).get("regime"),
                    "best_model": fit.get("best_by_aicc"),
                })
            except Exception as exc:
                per_curve.append({"series_id": s.get("series_id"),
                                  "status": "degraded",
                                  "reason": f"{type(exc).__name__}: {exc}"})
        out["status"] = "ok"
    except Exception as exc:
        out["status"] = "degraded"
        out["reason"] = f"{type(exc).__name__}: {exc}"
    out["per_curve_descriptive"] = per_curve
    out["confound_flag"] = {
        "flagged": True,
        "invariant": "Invariant 19 (co-varying conditions confound γ/mechanism)",
        "co_varying_axes": mode.get("confound_axes", []),
        "message": ("REFUSED: mechanism and cross-curve scaling (γ) are not licensed — "
                    + (mode.get("reason") or "multiple conditions co-vary")
                    + ". Only per-curve descriptive regime is reported."),
    }
    if not comparability.get("ok", True):
        out["comparability_confound"] = comparability
    return out


# --------------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------------- #
def assemble_cohort(selected_series: list[dict],
                    series_meta: dict | None = None,
                    run_global_ode: bool = True,
                    b_reg: int = 200, b_glob: int = 60) -> dict:
    """Assemble ≥1 already-triaged AggregationSeries for ONE protein into a routed,
    comparability-checked cohort_result (§3-M0). NEVER raises.

    Steps: (1) Service-A comparability gate; (2) data-mode detection from what varies;
    (3) route to the licensed analysis; (4) assemble the cohort_result with the
    window of validity. A bad/empty selection -> a flagged minimal result.

    `series_meta` (optional) supplies the concentration_series identity fields
    (protein/pH/temperature_C/assay/mutation) for the dual-γ + global-fit payloads;
    when absent it is DERIVED from the members. `run_global_ode` gates the slow
    shared-rate ODE fit on the concentration_series route (default on). `b_reg`/
    `b_glob` are the dual-γ bootstrap budgets (a latency-bound caller lowers them —
    the point γ estimates are unchanged, only the CI resolution)."""
    base = {
        "version": M0_VERSION,
        "module": "M0 — Dataset Selection & Cohort Assembly",
    }

    # ---- guard: empty / malformed selection -> flagged minimal result ----------
    if not isinstance(selected_series, list) or len(selected_series) == 0:
        base.update({
            "data_mode": "empty",
            "comparability": {"ok": False, "mismatches": [], "degraded_fields": [],
                              "confound_flag": False,
                              "note": "no datasets selected"},
            "members": [],
            "assembled_analysis": {"route": "none",
                                   "status": "empty",
                                   "reason": "empty selection — nothing to assemble"},
            "window_of_validity": {},
            "n_selected": 0,
            "note": "empty/invalid selection; select >=1 dataset for the same protein",
        })
        return base

    # keep only dict-shaped members (a malformed entry is dropped + flagged, never raises)
    members = [s for s in selected_series if isinstance(s, dict)]
    n_dropped = len(selected_series) - len(members)

    if not members:
        base.update({
            "data_mode": "empty",
            "comparability": {"ok": False, "mismatches": [], "degraded_fields": [],
                              "confound_flag": False, "note": "no valid dataset objects"},
            "members": [], "assembled_analysis": {"route": "none", "status": "empty"},
            "window_of_validity": {}, "n_selected": len(selected_series),
            "note": "selection contained no valid dataset objects",
        })
        return base

    # ---- (1) comparability gate (BEFORE any analysis) --------------------------
    comparability = check_comparability(members)

    # ---- (2) data-mode detection ----------------------------------------------
    mode = detect_data_mode(members, comparability)
    data_mode = mode["data_mode"]

    # ---- window of validity (measured ranges) ---------------------------------
    wov = window_of_validity(members)

    # ---- (3) route -------------------------------------------------------------
    meta = dict(series_meta or {})
    # derive concentration_series identity fields from the members if not supplied
    if not meta:
        first_cv = _cv(members[0])
        meta = {
            "concentration_series_id": (series_meta or {}).get("concentration_series_id")
            or f"cohort:{members[0].get('series_id')}",
            "protein": members[0].get("protein_id") or first_cv.get("protein"),
            "pH": first_cv.get("pH"),
            "temperature_C": first_cv.get("temperature_C"),
            "assay": first_cv.get("assay_type"),
            "mutation": first_cv.get("construct_id"),
        }

    if data_mode == "single_curve":
        assembled = _route_single_curve(members)
    elif data_mode == "replicates":
        assembled = _route_replicates(members)
    elif data_mode == "concentration_series":
        assembled = _route_concentration_series(members, meta,
                                                run_global_ode=run_global_ode,
                                                b_reg=b_reg, b_glob=b_glob)
    elif data_mode == "condition_series":
        assembled = _route_condition_series(members, mode.get("surface_axis") or "pH")
    elif data_mode == "confounded":
        assembled = _route_confounded(members, mode, comparability)
    else:                                                # empty (shouldn't reach)
        assembled = {"route": "none", "status": data_mode}

    # ---- (4) assemble the cohort_result ---------------------------------------
    result = {
        **base,
        "data_mode": data_mode,
        "data_mode_reason": mode.get("reason"),
        "n_selected": len(members),
        "n_dropped_malformed": n_dropped,
        "comparability": {
            "ok": comparability["ok"],
            "mismatches": comparability["mismatches"],
            "mismatched_fields": comparability["mismatched_fields"],
            "confound_flag": comparability["confound_flag"],
            "degraded_fields": comparability["degraded_fields"],
            "note": comparability["note"],
        },
        "varying_axes": mode.get("varying_axes", {}),
        "confound_axes": mode.get("confound_axes", []),
        "members": [_member_record(m) for m in members],
        "assembled_analysis": assembled,
        "window_of_validity": wov,
        "honesty": (
            "M0 orchestrates the existing engine (M1/M2/M4/M5/global_fit/pooling); it "
            "gates comparability (Service A) and confounds BEFORE fitting, routes by "
            "what VARIES across the selection, and bounds inference to the measured "
            "condition ranges. Mechanism/scaling is REFUSED on a confound (Invariant 19)."),
    }
    return result


# --------------------------------------------------------------------------- #
#  Demo (real concentration series from the ETL artifacts)
# --------------------------------------------------------------------------- #
def _load_triaged(path: Path) -> dict:
    by_id = {}
    if not path.exists():
        return by_id
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except Exception:
                continue
            by_id[s.get("series_id")] = s
    return by_id


def _demo():                                             # pragma: no cover
    root = Path(__file__).resolve().parent.parent / "data" / "processed"
    triaged = _load_triaged(root / "curves_triaged.jsonl")
    cs_path = root / "concentration_series.json"
    if not triaged or not cs_path.exists():
        print("[m0] demo needs data/processed/curves_triaged.jsonl + "
              "concentration_series.json (run the ETL + M1 first)")
        return 0
    series = json.loads(cs_path.read_text(encoding="utf-8"))
    # pick a modest real concentration series (>=3 curves) for a quick demo
    meta = None
    for m in series:
        members = [triaged[s] for s in m.get("member_series_ids", []) if s in triaged]
        if len(members) >= 3:
            meta = m
            break
    if meta is None:
        print("[m0] no concentration series with >=3 members found")
        return 0
    members = [triaged[s] for s in meta["member_series_ids"] if s in triaged]
    print(f"[m0] REAL concentration series {meta['concentration_series_id']} "
          f"[{meta['protein']}] — {len(members)} member curves")
    # keep the demo fast: skip the slow ODE fit here (the dual-γ alone demonstrates
    # the routing; the CLI global fit is exercised separately)
    res = assemble_cohort(members, series_meta=meta, run_global_ode=False)
    print(json.dumps({
        "data_mode": res["data_mode"],
        "data_mode_reason": res["data_mode_reason"],
        "comparability_ok": res["comparability"]["ok"],
        "n_selected": res["n_selected"],
        "route": res["assembled_analysis"].get("route"),
        "gamma_regression": (res["assembled_analysis"].get("gamma_regression") or {}).get("gamma"),
        "gamma_global": (res["assembled_analysis"].get("gamma_global") or {}).get("gamma"),
        "window_of_validity_concentration": res["window_of_validity"].get("concentration_uM"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
