"""
PRISE — Module M5: Classification & Regime Registry (degeneracy- & shape-aware)
==============================================================================

Consumes M1 triage + M2/M3 fits + M4 features and emits three things
(PRISE_DESIGN.md §3-M5):

  1. **Descriptive regime** — ALWAYS available, from a declarative, hierarchical,
     versioned **Regime Registry**. Driven by the M4 shape features, never by a
     mechanistic assumption.
  2. **Mechanistic verdict OR a data-regime-conditional equivalence class** — the
     honest, hard part. We DO NOT name a single mechanism: the call is
       * **assay-licensed** (mass-proportional ThT/ThS only),
       * **shape-gated** (a nucleated transition must actually be present; γ is a
         *consistency check only*, never sole evidence — K1),
       * **hard-branched on agitation & seeding** (quiescent vs shaken are
         different mechanisms; unknown ⇒ branch undetermined — §3-M5),
       * **data-regime-conditional in its degeneracy** (single curve ⇒ broad
         equivalence class; clean concentration series + reliable γ ⇒ narrower;
         significant log–log curvature ⇒ saturating-secondary / m_crit re-broadens
         it — K2/K3).
     The *specific* confusion-matrix equivalence class and the FDR-controlled
     single-mechanism degeneracy threshold are **Service C deliverables** (synthetic
     calibration), not yet built — so M5 returns the equivalence-class STRUCTURE and
     refuses single-mechanism calls, flagging the Service-C dependency. This mirrors
     the first-pass-threshold honesty of M1/M3.
  3. **Confidence + rationale + condition vector + window of validity.**

Plus: a **model-agnostic anomaly detector** (Wald–Wolfowitz runs test on the
best-fit residuals — structure the model bank cannot represent), **FDR-controlled
across the corpus** (Benjamini–Hochberg), feeding a **human-governed registry
proposer**: M5 emits a signed/dated/rationale-bearing PROPOSAL artifact bound to a
`taxonomy_version` but **never mutates the registry** (§7).

Plus: a **structure-linkage** associative panel (CPAD/PDB cross-refs) — surfaced as
*associative* insight with a polymorphism caveat, never a causal claim.

All outputs JSON-serialisable. Nothing here raises on bad input.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from m2_fit import fit_curve
from m3_select import predict
from m4_features import dual_gamma, extract_features

# ---------------------------------------------------------------------------- #
# Declarative, hierarchical, VERSIONED regime registry. Open-world: the
# `anomalous_unclassified` leaf absorbs everything the model bank cannot represent.
# Governance is human-only — see `propose_registry_mutation`; M5 never edits this.
# ---------------------------------------------------------------------------- #
REGIME_REGISTRY = {
    "version": "m5-regimes-1.0",
    "open_world": True,
    "governance": ("human-governed; only manual approval mutates; every mutation is "
                   "a signed, dated, rationale-bearing, version-bound artifact (§7)"),
    "regimes": {
        "no_detectable_aggregation": {
            "parent": None,
            "definition": "no transition above assay noise over the observed window",
            "qualifier": "assay-sensitivity-limited — 'no aggregation' is a statement "
                         "about THIS assay's detection limit, not the protein"},
        "gradual_non_cooperative": {
            "parent": None,
            "definition": "monotone rise with NO interior inflection (downhill / "
                          "decelerating / power-law) — no nucleation lag"},
        "threshold_driven": {
            "parent": None,
            "definition": "near-flat then abrupt onset: interior inflection with a "
                          "high relative lag and a very sharp transition"},
        "cooperative_sigmoidal": {
            "parent": None,
            "definition": "smooth S-curve: lag → accelerating growth (interior "
                          "inflection) → plateau"},
        "non_monotonic_settling": {
            "parent": None,
            "definition": "rise-then-fall / biphasic — sedimentation, ThT bleaching, "
                          "or a secondary process confounds a simple growth model"},
        "anomalous_unclassified": {
            "parent": None,
            "definition": "open-world: best-fit leaves structured residuals the model "
                          "bank cannot represent, or selection is unusable"},
    },
}

# --- Service-C-CALIBRATED shape thresholds (thresholds-1.0; §7 governed apply) - #
# These are NO LONGER first-pass. THRESHOLD_SHARPNESS / THRESHOLD_LAG_RATIO now
# source their VALUE from the governed calibrated-thresholds artifact (held-out
# out-of-sample AUC 1.0 for sharpness); the previous 3.0 / 0.80 are retained there
# as `first_pass`. ANOMALY_FDR_ALPHA stays 0.05 (Service C confirmed FDR-null
# control holds). Constant NAMES are unchanged (consumers/tests reference them).
from calibrated_thresholds import calibrated_value as _cal, THRESHOLDS_VERSION
THRESHOLD_SHARPNESS = _cal("M5_THRESHOLD_SHARPNESS", default=3.0)   # first_pass 3.0 -> 5.7
THRESHOLD_LAG_RATIO = _cal("M5_THRESHOLD_LAG_RATIO", default=0.80)  # first_pass 0.80 -> 0.97
LOW_R2 = 0.80                  # below this the descriptive call is low-confidence
ANOMALY_FDR_ALPHA = _cal("M5_ANOMALY_FDR_ALPHA", default=0.05)      # unchanged (0.05)
MIN_POINTS_RUNS = 8            # runs-test normal approximation needs enough points
# A registry PROPOSAL needs structure AND magnitude: a runs-significant but
# near-perfect fit is not a candidate new regime. Only structured residuals on a
# materially-misfit curve (best-fit R² below this) become human-triage proposals.
ANOMALY_MIN_MISFIT_R2 = 0.90

# Nucleation mechanisms that a SINGLE curve cannot tell apart (the broad class).
# This is a placeholder breadth; the data-regime-specific class is a Service C
# confusion-matrix output (deferred) — never presented as a calibrated set.
_BROAD_MECHANISM_CLASS = [
    "primary_nucleation_dominated",
    "secondary_nucleation_dominated",
    "fragmentation_dominated",
    "saturating_secondary_nucleation",
]


# ============================ descriptive regime ============================= #
def classify_descriptive(m1: dict, feat: dict) -> dict:
    """Assign a top-level descriptive regime from triage + shape features. Pure
    SHAPE — no mechanism assumed, and never erased by the anomaly overlay (that is a
    separate, parallel, FDR-controlled channel). `anomalous_unclassified` is reserved
    for genuinely unusable curves (no converged fit), not mild misfit."""
    handling = m1.get("recommended_handling")
    flat = bool(m1.get("flat_no_signal"))
    non_mono = bool(m1.get("non_monotonic")) or handling == "nonmonotonic_fit"

    if feat.get("status") == "no_fit":
        regime = "anomalous_unclassified"
        why = "no converged model fit — curve not representable by the bank"
    elif flat or feat.get("status") == "flat_no_transition":
        regime = "no_detectable_aggregation"
        why = "no transition resolved above assay noise"
    elif non_mono:
        regime = "non_monotonic_settling"
        why = "non-monotonic / biphasic trace (settling or secondary process)"
    else:
        f = feat.get("features", {})
        coop = bool(f.get("has_interior_inflection"))
        sharp = f.get("transition_sharpness")
        ratio = f.get("lag_to_t50_ratio")
        if not coop:
            regime = "gradual_non_cooperative"
            why = "monotone rise with no interior inflection (no nucleation lag)"
        elif (sharp is not None and sharp > THRESHOLD_SHARPNESS
              and ratio is not None and ratio > THRESHOLD_LAG_RATIO):
            regime = "threshold_driven"
            why = (f"abrupt onset after a long flat phase "
                   f"(sharpness {sharp:.1f} > {THRESHOLD_SHARPNESS}, "
                   f"lag/t50 {ratio:.2f} > {THRESHOLD_LAG_RATIO})")
        else:
            regime = "cooperative_sigmoidal"
            why = "smooth sigmoid: lag, interior inflection, plateau"
    return {"regime": regime,
            "definition": REGIME_REGISTRY["regimes"][regime]["definition"],
            "evidence": why,
            "registry_version": REGIME_REGISTRY["version"]}


# ====================== mechanistic equivalence class ======================= #
def mechanistic_assessment(series: dict, descriptive: dict, feat: dict,
                           data_regime: str, gamma: dict | None = None) -> dict:
    """Shape-gated, data-regime-conditional mechanistic statement. Refuses to name a
    single mechanism (returns an equivalence class) and lists every gate it failed."""
    cv = series.get("condition_vector", {}) or {}
    regime = descriptive["regime"]
    gates_failed = []

    # Gate A — assay must report mass for a mechanistic rate law to mean anything
    #          (unknown mass-proportionality is distinct from known-not-mass)
    amp = cv.get("assay_reports_mass")
    if amp is None:
        gates_failed.append("assay_mass_proportionality_unknown")
    elif not amp:
        gates_failed.append("assay_not_mass_proportional")
    # Gate B — there must actually be a nucleated transition to discuss nucleation
    if regime not in ("cooperative_sigmoidal", "threshold_driven"):
        gates_failed.append("no_nucleated_transition_in_shape")
    # Gate C — agitation & seeding are HARD mechanism branches (quiescent vs shaken)
    prov = cv.get("field_provenance", {}) or {}
    agit_known = prov.get("agitation") == "known" and cv.get("agitation") is not None
    seed_known = prov.get("seeded") == "known" and cv.get("seeded") is not None
    if not (agit_known and seed_known):
        gates_failed.append("agitation_or_seeding_unknown_branch_undetermined")

    licensed = not gates_failed

    # Degeneracy reflects what THIS analysis actually constrains — i.e. whether a
    # reliable series-level γ is in hand — NOT the bare data mode (K2/K3). A single
    # curve never narrows; a reliable, curvature-free γ narrows the *evidence*; log–log
    # curvature re-broadens (saturating secondary / m_crit / transition). The specific
    # confusion-matrix class itself is a Service C deliverable — so the equivalence
    # class is NEVER actually narrowed here (`equivalence_class_narrowed = False`).
    reg = (gamma or {}).get("gamma_regression") or {}
    reg_ok = reg.get("status") == "ok"
    curv_sig = bool(reg.get("curvature_test", {}).get("significant")) if reg_ok else False
    if gamma is None:
        degeneracy, deg_note = ("broad_single_curve",
                                "single-curve analysis under-determines the mechanism")
    elif not reg_ok:
        degeneracy, deg_note = ("broad_gamma_unavailable",
                                "no usable series γ (too few anchors) — broad degeneracy")
    elif curv_sig:
        degeneracy, deg_note = ("re_broadened_saturating_or_mcrit",
                                "significant log–log curvature ⇒ concentration-dependent "
                                "reaction order: saturating secondary nucleation, approach "
                                "to m_crit, OR a mechanism transition")
    elif reg.get("gamma_reliable"):
        degeneracy, deg_note = ("evidence_reduced_by_reliable_gamma",
                                "a reliable γ constrains a combination of reaction orders "
                                "(still many-to-one onto mechanism) — consistency check only")
    else:
        degeneracy, deg_note = ("broad_gamma_unreliable",
                                "γ not reliably estimated for this series — broad degeneracy")

    return {
        "single_mechanism_call": None,                       # refused, by design
        "single_mechanism_refused_reason": (
            "single-mechanism calls require a Service-C confusion matrix + an "
            "FDR-controlled degeneracy threshold (§4); not yet built"),
        "equivalence_class": list(_BROAD_MECHANISM_CLASS),
        "equivalence_class_narrowed": False,                 # narrowing needs Service C
        "equivalence_class_provenance": (
            "PLACEHOLDER breadth — even when γ reduces the EVIDENCE, the actual "
            "data-regime-specific class is a Service C deliverable; not narrowed here"),
        "degeneracy": degeneracy,
        "degeneracy_note": deg_note,
        "mechanistic_inference_licensed": bool(licensed),
        "gates_failed": gates_failed,
        "data_regime_context": data_regime,
        "gamma_consistency_check": (
            None if not (gamma and gamma.get("gamma_regression")) else {
                "gamma_regression": gamma["gamma_regression"].get("gamma"),
                "gamma_reliable": gamma["gamma_regression"].get("gamma_reliable"),
                "role": "consistency check only — never sole evidence (K1)"}),
    }


# ============================ anomaly detector ============================== #
def _runs_test(resid: np.ndarray) -> dict:
    """Wald–Wolfowitz runs test on residual signs. Too FEW runs ⇒ long same-sign
    stretches ⇒ systematic lack of fit (structure the model missed). One-sided
    p for too-few-runs via the normal approximation."""
    s = np.sign(resid)
    s = s[s != 0]
    n = len(s)
    if n < MIN_POINTS_RUNS:
        return {"testable": False, "reason": "too few non-zero residuals"}
    n1 = int(np.sum(s > 0)); n2 = int(np.sum(s < 0))
    if n1 == 0 or n2 == 0:
        # all residuals one sign -> maximal structure
        return {"testable": True, "runs": 1, "z": float("-inf"),
                "p_too_few_runs": 0.0}
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    mu = 2.0 * n1 * n2 / (n1 + n2) + 1.0
    var = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2)
           / ((n1 + n2) ** 2 * (n1 + n2 - 1.0)))
    if var <= 0:
        return {"testable": False, "reason": "degenerate variance"}
    z = (runs - mu) / math.sqrt(var)
    p_low = 0.5 * math.erfc(-z / math.sqrt(2.0))     # P(R <= observed) ~ Phi(z)
    return {"testable": True, "runs": runs, "expected_runs": float(mu),
            "z": float(z), "p_too_few_runs": float(p_low)}


def residual_anomaly(series: dict, fit_result: dict | None = None) -> dict:
    """Best-fit residual structure test (pre-FDR). Uses the AICc-best converged
    model from M2 unless a fit is supplied."""
    fr = fit_result or fit_curve(series)
    conv = {n: r for n, r in fr.get("fits", {}).items()
            if r.get("converged") and r.get("aicc") is not None}
    x = series.get("x_hours") or []
    y = series.get("m1", {}).get("y_processed") or series.get("y_intensity") or []
    if not conv or len(x) != len(y) or len(x) < MIN_POINTS_RUNS:
        return {"testable": False, "reason": "no fit / too few points",
                "best_model": (min(conv, key=lambda n: conv[n]["aicc"]) if conv else None)}
    best = min(conv, key=lambda n: conv[n]["aicc"])
    yhat = predict(best, conv[best]["params"], x)
    resid = np.asarray(y, float) - yhat
    rt = _runs_test(resid)
    rt["best_model"] = best
    rt["r2"] = conv[best].get("r2")
    return rt


def benjamini_hochberg(pvals: list[float], alpha: float = ANOMALY_FDR_ALPHA) -> dict:
    """BH step-up FDR. Returns the rejection threshold and a boolean mask aligned to
    the input order. (Independence/PRDS assumed; Benjamini–Yekutieli is the
    conservative alternative if dependence is a concern.)"""
    idx = [i for i, p in enumerate(pvals) if p is not None and math.isfinite(p)]
    m = len(idx)
    if m == 0:
        return {"alpha": alpha, "n_tested": 0, "threshold": None,
                "reject": [False] * len(pvals), "n_rejected": 0}
    order = sorted(idx, key=lambda i: pvals[i])
    thresh = 0.0
    kmax = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= rank / m * alpha:
            kmax = rank
            thresh = rank / m * alpha
    reject = [False] * len(pvals)
    for rank, i in enumerate(order, start=1):
        if rank <= kmax:
            reject[i] = True
    return {"alpha": alpha, "n_tested": m, "threshold": (thresh if kmax else None),
            "reject": reject, "n_rejected": int(kmax)}


def propose_registry_mutation(series_id: str, anomaly: dict) -> dict:
    """Emit a PROPOSAL artifact for the human-governed proposer. This NEVER mutates
    REGIME_REGISTRY — it records the triggering evidence for manual review (§7)."""
    return {
        "kind": "registry_mutation_proposal",
        "status": "PROPOSED_PENDING_HUMAN_APPROVAL",
        "proposed_at": _dt.date.today().isoformat(),
        "taxonomy_version": REGIME_REGISTRY["version"],
        "trigger": "fdr_significant_structured_residuals",
        "evidence": {"series_id": series_id,
                     "best_model": anomaly.get("best_model"),
                     "runs": anomaly.get("runs"),
                     "expected_runs": anomaly.get("expected_runs"),
                     "p_too_few_runs": anomaly.get("p_too_few_runs"),
                     "r2": anomaly.get("r2")},
        "note": "only manual approval mutates the registry; this is a proposal only",
    }


# ============================ structure linkage ============================= #
def structure_panel(series: dict) -> dict:
    """Associative structure-linkage (CPAD/PDB). Recorded, NOT used to stratify;
    polymorphism is a first-class caveat — never a causal claim from one structure."""
    cv = series.get("condition_vector", {}) or {}
    pdb = cv.get("pdb_id")
    return {
        "pdb_id": pdb,
        "construct": cv.get("construct_id"),
        "structural_statement": series.get("structural_statement", "unknown"),
        "relationship": "associative_only",
        "caveats": ["fibril polymorphism: 'the structure of protein P' can be "
                    "ill-posed", "not a causal claim from a single structure"],
        "available": bool(pdb),
    }


# ============================== confidence ================================== #
def _confidence(feat: dict, anomalous: bool, m3: dict | None) -> dict:
    r2 = feat.get("r2")
    n = feat.get("n_points")
    if m3 and m3.get("selection") and m3["selection"].get("selection_stability") is not None:
        stab = m3["selection"]["selection_stability"]
    else:
        stab = None
    if r2 is None or anomalous or r2 < LOW_R2:
        level = "low"                     # no usable fit / anomalous / poor fit
    elif r2 >= 0.95 and (stab is None or stab >= 0.6):
        level = "high"
    else:
        level = "medium"
    return {"level": level, "r2": r2, "n_points": n,
            "selection_stability": stab,
            "basis": "R² + anomaly + (M3 selection-stability when supplied)"}


# ============================ orchestration ================================= #
def classify_curve(series: dict, m3: dict | None = None,
                   fit_result: dict | None = None) -> dict:
    """Full per-curve M5 record (descriptive + single-curve mechanistic + anomaly +
    structure panel + confidence/validity). `anomaly_significant` here is the raw
    per-curve flag; corpus FDR is applied by the driver (`benjamini_hochberg`)."""
    fr = fit_result or fit_curve(series)
    feat = extract_features(series, fit_result=fr)
    anomaly = residual_anomaly(series, fr)
    # raw (pre-FDR) anomaly flag — a PARALLEL overlay, not a regime eraser. Corpus
    # FDR control is applied by the driver; this raw flag only feeds confidence.
    raw_anom = bool(anomaly.get("testable")
                    and anomaly.get("p_too_few_runs") is not None
                    and anomaly["p_too_few_runs"] < ANOMALY_FDR_ALPHA)

    m1 = series.get("m1", {})
    descriptive = classify_descriptive(m1, feat)
    # `data_regime` = the data mode AVAILABLE for this curve; the per-curve mechanistic
    # analysis itself is always single-curve scope (no global fit here -> gamma=None),
    # so its degeneracy is broad_single_curve regardless of series membership.
    data_regime = ("concentration_series" if series.get("concentration_series_id")
                   else (series.get("data_mode") or "kinetic"))
    mech = mechanistic_assessment(series, descriptive, feat, data_regime, gamma=None)

    # material-misfit flag (effect size) — pairs with the FDR runs test so the driver
    # only proposes new regimes for curves the bank genuinely cannot represent
    r2 = anomaly.get("r2")
    material_misfit = bool(r2 is not None and r2 < ANOMALY_MIN_MISFIT_R2)

    x = series.get("x_hours") or []
    return {
        "series_id": series.get("series_id"),
        "registry_version": REGIME_REGISTRY["version"],
        "analysis_scope": "single_curve",
        "descriptive_regime": descriptive,
        "mechanistic": mech,
        "anomaly": {**anomaly, "raw_flag": raw_anom,
                    "material_misfit": material_misfit,
                    "note": "corpus FDR control applied at the driver level; a proposal "
                            "additionally requires material_misfit (effect size)"},
        "structure_linkage": structure_panel(series),
        "confidence": _confidence(feat, raw_anom, m3),
        "window_of_validity": {
            "t_min_hours": (float(min(x)) if x else None),
            "t_max_hours": (float(max(x)) if x else None),
            "censoring_class": m1.get("censoring_class"),
            "note": "inference is not licensed beyond the observed window / basis"},
        "condition_vector": series.get("condition_vector"),
        "data_regime": data_regime,
    }


def classify_series(series_meta: dict, members: list[dict],
                    B_reg: int = 150, B_glob: int = 60) -> dict:
    """Concentration-series M5: a modal descriptive regime across members + the
    γ-informed mechanistic equivalence class (shape-gated, γ as consistency check)."""
    per = [classify_curve(s) for s in members]
    regimes = Counter(p["descriptive_regime"]["regime"] for p in per)
    modal, modal_n = (regimes.most_common(1)[0] if regimes else ("unknown", 0))
    gamma = dual_gamma(series_meta, members, B_reg=B_reg, B_glob=B_glob)

    # representative member for the assay/agitation gates (conditions are shared)
    rep = members[0] if members else {}
    rep_feat = extract_features(rep) if rep else {"features": {}}
    rep_desc = {"regime": modal}
    mech = mechanistic_assessment(rep, rep_desc, rep_feat,
                                  "concentration_series", gamma=gamma)
    return {
        "concentration_series_id": series_meta.get("concentration_series_id"),
        "protein": series_meta.get("protein"),
        "registry_version": REGIME_REGISTRY["version"],
        "n_members": len(members),
        "descriptive_regime_modal": modal,
        "descriptive_regime_agreement": (modal_n / len(per)) if per else None,
        "descriptive_regime_distribution": dict(regimes),
        "mechanistic": mech,
        "dual_gamma": gamma,
        "conditions": {k: series_meta.get(k) for k in
                       ("pH", "temperature_C", "assay", "mutation")},
    }


# ================================== I/O ==================================== #
def _run_curves(args):
    out_path = args.output or (args.input.parent / "regimes.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records, pvals = [], []
    regime_counts: Counter = Counter()
    licensed = 0
    print("[M5] classifying curves ...")
    with open(args.input, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("m1", {}).get("fittability_class") != "fittable":
                continue
            res = classify_curve(s)
            records.append(res)
            regime_counts[res["descriptive_regime"]["regime"]] += 1
            licensed += bool(res["mechanistic"]["mechanistic_inference_licensed"])
            a = res["anomaly"]
            pvals.append(a.get("p_too_few_runs") if a.get("testable") else None)
            if args.limit and len(records) >= args.limit:
                break

    # corpus-level FDR on the anomaly p-values; a PROPOSAL needs FDR-significant
    # structure AND material misfit (effect size) — structure alone over-flags
    # sparse digitized curves.
    bh = benjamini_hochberg(pvals)
    fdr_sig = proposals = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for res, rej in zip(records, bh["reject"]):
            res["anomaly"]["fdr_significant"] = bool(rej)
            fdr_sig += bool(rej)
            actionable = bool(rej and res["anomaly"].get("material_misfit"))
            res["anomaly"]["proposal_actionable"] = actionable
            if actionable:
                res["registry_proposal"] = propose_registry_mutation(
                    res["series_id"], res["anomaly"])
                proposals += 1
            out.write(json.dumps(res) + "\n")
    print(f"[M5] wrote {out_path}  ({len(records)} curves)")
    print(json.dumps({
        "n_curves": len(records),
        "descriptive_regimes": dict(regime_counts),
        "mechanistic_inference_licensed": licensed,
        "mechanistic_note": ("0 expected on this corpus: agitation/seeding are "
                             "'unknown' everywhere, so the quiescent-vs-shaken hard "
                             "branch is undetermined and the mechanistic gate refuses"),
        "anomaly_fdr": {"n_tested": bh["n_tested"], "n_rejected_structure": bh["n_rejected"],
                        "alpha": bh["alpha"],
                        "min_misfit_r2_for_proposal": ANOMALY_MIN_MISFIT_R2},
        "anomaly_note": ("FDR-significant = structured residuals (runs test); a curve "
                         "becomes a PROPOSAL only if ALSO materially misfit (R²<"
                         f"{ANOMALY_MIN_MISFIT_R2}). The runs-test threshold is first-pass, "
                         "pending Service C calibration; proposals are for human triage, "
                         "never asserted new regimes"),
        "registry_proposals_emitted": proposals,
    }, indent=2))


def _run_series(args):
    root = args.input.parent
    cs_path = root / "concentration_series.json"
    if not cs_path.exists():
        raise SystemExit(f"{cs_path} not found (run the ETL first)")
    series = json.loads(cs_path.read_text(encoding="utf-8"))
    by_id = {}
    with open(args.input, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                s = json.loads(line)
                by_id[s.get("series_id")] = s
    out_path = args.output or (root / "regimes_series.jsonl")
    n = 0
    modal_counts: Counter = Counter()
    print(f"[M5] classifying {len(series)} concentration series ...")
    with open(out_path, "w", encoding="utf-8") as out:
        for meta in series:
            members = [by_id[sid] for sid in meta.get("member_series_ids", [])
                       if sid in by_id]
            if len(members) < 3:
                continue
            res = classify_series(meta, members, B_reg=args.bootstrap,
                                  B_glob=min(args.bootstrap, 60))
            modal_counts[res["descriptive_regime_modal"]] += 1
            out.write(json.dumps(res) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"[M5] wrote {out_path}  ({n} series)")
    print(json.dumps({"n_series": n, "modal_regimes": dict(modal_counts)}, indent=2))


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="PRISE M5 classification & regime registry")
    ap.add_argument("--input", type=Path,
                    default=root / "data" / "processed" / "curves_triaged.jsonl")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=150)
    ap.add_argument("--series", action="store_true",
                    help="classify concentration series (mechanistic equivalence class)")
    args = ap.parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input} (run M1 first)")
    if args.series:
        _run_series(args)
    else:
        _run_curves(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
