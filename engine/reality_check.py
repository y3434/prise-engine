"""
PRISE — Blind External Reality Check (the de-circularizing falsification test)
=============================================================================

This is the falsification test PRISE_DESIGN.md §6C (Service C part (2), decision
row C1) names as the thing that de-circularizes the whole engine:

    "run blind, once, on the handful of proteins the field considers
     mechanistically RESOLVED ... compare verdicts to consensus; disagreement on
     'resolved' cases falsifies the map ... a falsification test on a few decisive
     cases (like a thermometer against fixed points), NOT a statistical
     calibration ... consumes real proteins reserved SOLELY for this role, never
     used for anchoring."

WHAT THIS IS (and is NOT).
  * It is a NON-CONTRADICTION test on a HANDFUL of fixed points — a thermometer
    against known freezing/boiling points, not a regression calibration.
  * It is NOT proof of correctness, NOT statistical calibration, and NOT a "pass =
    the engine is validated" stamp. n is deliberately tiny (a few decisive cases).
  * PRISE HONESTLY REFUSES to name a single mechanism corpus-wide, because
    agitation & seeding are unrecorded (the quiescent-vs-shaken hard branch is
    undetermined — see m5_classify Gate C). That refusal is EXPECTED and is stated
    in every payload. The test is therefore on PRISE's HONEST outputs — the
    descriptive REGIME, the scaling exponent γ, and the mechanistic EQUIVALENCE
    CLASS — asking whether they CONTRADICT the field consensus, never whether
    PRISE named the "right" mechanism.

THE THREE CONSISTENCY CHECKS (per fixed point).
  1. REGIME — does PRISE call the protein cooperative_sigmoidal (or threshold_
     driven), consistent with a "nucleation-dependent sigmoid"? A
     gradual_non_cooperative / no_detectable modal call on a field-resolved
     nucleated amyloid is a CONTRADICTION (fail).
  2. GAMMA — where a STRONG consensus γ exists (e.g. Aβ42 ~1.1), is PRISE's γ
     (regression and/or mechanistic) inside a tolerance band? Reported with the
     value + in/out flag. Out-of-band on a strong-consensus γ is a flag.
  3. MECHANISM-CLASS — is the consensus mechanism (e.g. secondary_nucleation for
     Aβ42) a MEMBER of PRISE's equivalence class / not excluded by the global-fit
     reaction-order constraint? PRISE passes unless its evidence POSITIVELY
     EXCLUDES the consensus mechanism (it should sit inside the broad class).

RESERVED-PROTEINS DISCIPLINE (leakage check).
  These proteins are reserved SOLELY for this §6C falsification role — never used
  for anchoring / calibration / demonstrators (§8 budget; §8.1 leakage discipline).
  `leakage_check()` verifies they are absent from any Service-C / anchor calibration
  set, and the result carries the reserved-proteins note.

This module only READS artifacts (protein_analysis / gamma / regimes_series /
global_fit); it performs NO inference and does NOT modify any engine module.
Everything is deterministic, JSON-serialisable, and NEVER raises (a missing
protein -> INSUFFICIENT_DATA, a missing artifact -> empty index).

Usage:
    python engine/reality_check.py        # writes data/processed/reality_check.json + prints summary
    python engine/reality_check.py --data <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REALITY_CHECK_VERSION = "reality-check-1.0"

# ============================================================================ #
# PRE-REGISTERED CONSENSUS — FROZEN ground truth for the fixed points.
# Supplied by the reviewer; treated as pre-registered (§8 external reality check,
# blind & once). Each entry cites the field consensus and marks consensus_strength.
# Where a strong consensus γ exists it is given with a tolerance band; where the
# consensus mechanism is condition-sensitive that is stated. These values are NOT
# tuned to PRISE output — they are the fixed points the thermometer is read against.
# ============================================================================ #
CONSENSUS = {
    "Amyloid Beta peptide-ABeta42": {
        "display_name": "Amyloid-β 42 (Aβ42)",
        "aliases": ["Amyloid Beta peptide-ABeta42",
                    "Amyloid Beta peptide-ABeta42 (synthetic)"],
        "expected_regimes": ["cooperative_sigmoidal", "threshold_driven"],
        "consensus_shape": "cooperative sigmoidal with a lag phase (nucleation-dependent)",
        "consensus_mechanism": "secondary_nucleation_dominated",
        "gamma_consensus": 1.1,
        "gamma_band": [0.8, 1.4],           # steep monomer-dependence; strong γ
        "gamma_consensus_strength": "strong",
        "consensus_strength": "strong",
        "citation": ("Cohen, Linse, Vendruscolo, Dobson, Knowles et al., "
                     "PNAS 110:9758 (2013): Aβ42 aggregation is dominated by "
                     "secondary nucleation; cooperative sigmoid with lag; "
                     "half-time scaling γ ≈ 1.1 (steep monomer dependence)."),
    },
    "Amyloid Beta peptide-ABeta40": {
        "display_name": "Amyloid-β 40 (Aβ40)",
        "aliases": ["Amyloid Beta peptide-ABeta40"],
        "expected_regimes": ["cooperative_sigmoidal", "threshold_driven"],
        "consensus_shape": "nucleation-dependent cooperative sigmoid; slower than Aβ42",
        "consensus_mechanism": "secondary_nucleation_dominated",
        "gamma_consensus": 0.45,
        "gamma_band": [0.2, 0.7],           # weaker monomer dependence -> lower γ
        "gamma_consensus_strength": "moderate",
        "consensus_strength": "strong",     # that it is a nucleated sigmoid is strong
        "citation": ("Meisl, Knowles et al., PNAS 111:9384 (2014) & Nat. Protoc. "
                     "11:252 (2016): Aβ40 is nucleation-dependent and cooperative, "
                     "slower than Aβ42; secondary processes present but weaker "
                     "monomer-dependence -> lower γ (~0.3–0.6, moderate)."),
    },
    "Alpha-Synuclein": {
        "display_name": "α-Synuclein",
        "aliases": ["Alpha-Synuclein", "Alpha-synuclein"],
        "expected_regimes": ["cooperative_sigmoidal", "threshold_driven"],
        "consensus_shape": "cooperative sigmoid, nucleation-dependent; condition-sensitive "
                           "(quiescent primary nucleation very slow; often agitation/seed-driven)",
        "consensus_mechanism": "primary_nucleation_dominated",
        "gamma_consensus": None,            # no single strong γ (condition-sensitive)
        "gamma_band": None,
        "gamma_consensus_strength": "moderate",
        "consensus_strength": "moderate",   # mechanism moderate; nucleated-amyloid strong
        "citation": ("Buell, Galvagnion, Dobson, Knowles et al., PNAS 111:7671 "
                     "(2014): α-synuclein is a nucleation-dependent amyloid; under "
                     "QUIESCENT conditions primary nucleation is very slow and "
                     "aggregation is agitation/seed-dependent — mechanism is "
                     "condition-sensitive (moderate consensus on mechanism, strong "
                     "that it IS a nucleation-dependent amyloid)."),
    },
    "insulin": {
        "display_name": "Insulin",
        "aliases": ["insulin", "Insulin"],
        "expected_regimes": ["cooperative_sigmoidal", "threshold_driven"],
        "consensus_shape": "classic nucleation-dependent amyloid; cooperative sigmoid with lag",
        "consensus_mechanism": "secondary_nucleation_dominated",
        "gamma_consensus": None,
        "gamma_band": None,
        "gamma_consensus_strength": "moderate",
        "consensus_strength": "strong",     # nucleated sigmoid strong; mechanism moderate
        "citation": ("Librizzi & Rischel, Protein Sci. 14:3129 (2005); Foderà et "
                     "al., J. Phys. Chem. B 112:3853 (2008): insulin is a classic "
                     "nucleation-dependent amyloid — cooperative sigmoid with lag "
                     "(strong); dominant microscopic mechanism moderate."),
    },
    "Beta-2 microglobulin (Beta-2m)": {
        "display_name": "β2-microglobulin (β2m)",
        "aliases": ["Beta-2 microglobulin (Beta-2m)",
                    "Beta-2-microglobulin (Beta-2m)"],
        "expected_regimes": ["cooperative_sigmoidal", "threshold_driven"],
        "consensus_shape": "nucleation-dependent amyloid; cooperative sigmoid; "
                           "fragmentation / secondary processes contribute",
        "consensus_mechanism": "fragmentation_dominated",
        "gamma_consensus": None,
        "gamma_band": None,
        "gamma_consensus_strength": "moderate",
        "consensus_strength": "strong",     # nucleated sigmoid strong
        "citation": ("Xue, Homans & Radford, PNAS 105:8926 (2008): β2-microglobulin "
                     "is a nucleation-dependent amyloid forming a cooperative sigmoid; "
                     "fibril fragmentation / secondary processes contribute (strong "
                     "that it is a nucleated sigmoid)."),
    },
}

# Descriptive regimes that ARE consistent with "nucleation-dependent sigmoid".
_NUCLEATED_REGIMES = {"cooperative_sigmoidal", "threshold_driven"}
# Descriptive regimes that CONTRADICT a field-resolved nucleated amyloid.
_CONTRADICTORY_REGIMES = {"gradual_non_cooperative", "no_detectable_aggregation"}


# ============================================================================ #
# Artifact loading (read-only; never raises)
# ============================================================================ #
def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue        # skip a corrupt line, never abort


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_artifacts(data_dir: Path) -> dict:
    """Read the (read-only) artifacts the check consumes. Any missing file yields an
    empty index — the check then reports INSUFFICIENT_DATA for the affected protein.

      * per-curve descriptive regimes, grouped by protein via curves_triaged
      * concentration-series modal regimes (regimes_series.jsonl)
      * dual-γ per series (gamma.jsonl)
      * shared-rate ODE global fits (global_fit.json)
    """
    data_dir = Path(data_dir)

    # series_id -> protein (from triage), to group per-curve regimes by protein
    sid2protein = {}
    for r in _iter_jsonl(data_dir / "curves_triaged.jsonl"):
        sid = r.get("series_id")
        prot = r.get("protein_id") or (r.get("condition_vector", {}) or {}).get("protein")
        if sid is not None:
            sid2protein[sid] = prot

    per_curve_regimes: dict = {}     # protein -> {regime: count}
    for r in _iter_jsonl(data_dir / "regimes.jsonl"):
        prot = sid2protein.get(r.get("series_id"))
        if not prot:
            continue
        regime = ((r.get("descriptive_regime", {}) or {}).get("regime")) or "unknown"
        per_curve_regimes.setdefault(prot, {})
        per_curve_regimes[prot][regime] = per_curve_regimes[prot].get(regime, 0) + 1

    series_modal: dict = {}          # protein -> list of concentration-series modal records
    for r in _iter_jsonl(data_dir / "regimes_series.jsonl"):
        series_modal.setdefault(r.get("protein"), []).append(r)

    gamma_by_protein: dict = {}      # protein -> list of dual-γ series records
    for r in _iter_jsonl(data_dir / "gamma.jsonl"):
        gamma_by_protein.setdefault(r.get("protein"), []).append(r)

    gf = _load_json(data_dir / "global_fit.json")
    global_fit_by_protein: dict = {}
    for s in (gf.get("series", []) or []):
        if s.get("status") == "ok":
            global_fit_by_protein.setdefault(s.get("protein"), []).append(s)

    return {
        "per_curve_regimes": per_curve_regimes,
        "series_modal": series_modal,
        "gamma_by_protein": gamma_by_protein,
        "global_fit_by_protein": global_fit_by_protein,
        # calibration / anchor sets, for the leakage check
        "service_c": _load_json(data_dir / "service_c_calibration.json"),
    }


# ============================================================================ #
# Per-protein PRISE-side extraction (BLIND — reads whatever the engine emitted)
# ============================================================================ #
def _match_names(aliases: list[str], index: dict) -> list[str]:
    """All keys in `index` whose lowercased value matches any alias (exact, then
    substring) — so 'ABeta42' and 'ABeta42 (synthetic)' both fold into Aβ42."""
    want = {a.lower() for a in aliases}
    keys = [k for k in index if k is not None]
    exact = [k for k in keys if k.lower() in want]
    if exact:
        return exact
    return [k for k in keys if any(a in k.lower() for a in want)]


def _modal_regime(counts: dict) -> tuple[str | None, int]:
    if not counts:
        return None, 0
    regime, n = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return regime, n


def extract_prise(entry: dict, artifacts: dict) -> dict:
    """Blind PRISE-side summary for one pre-registered protein: the per-curve modal
    regime + distribution, the best available γ (regression preferred when reliable,
    else global collapse), the global-fit mechanistic γ + reaction-order class, and
    the equivalence class + why mechanism is not licensed. Pure lookup — no inference."""
    aliases = entry["aliases"]

    # --- descriptive regime (pooled across every curve of this protein) --------
    reg_index = artifacts["per_curve_regimes"]
    reg_counts: dict = {}
    for k in _match_names(aliases, reg_index):
        for regime, c in reg_index[k].items():
            reg_counts[regime] = reg_counts.get(regime, 0) + c
    modal_regime, modal_n = _modal_regime(reg_counts)
    n_curves = sum(reg_counts.values())

    # --- γ: prefer a reliable regression γ; else a global-collapse γ -----------
    gamma_index = artifacts["gamma_by_protein"]
    gamma_reg_best = None       # reliable regression γ (value, ci, csid)
    gamma_reg_any = None        # any regression γ even if unreliable
    gamma_collapse_best = None  # a good-collapse global γ
    for k in _match_names(aliases, gamma_index):
        for rec in gamma_index[k]:
            reg = rec.get("gamma_regression", {}) or {}
            glob = rec.get("gamma_global", {}) or {}
            csid = rec.get("concentration_series_id")
            if reg.get("status") == "ok" and reg.get("gamma") is not None:
                cand = {"gamma": reg.get("gamma"), "ci": reg.get("gamma_ci"),
                        "reliable": bool(reg.get("gamma_reliable")),
                        "physical": bool(reg.get("gamma_physical", True)),
                        "csid": csid}
                if gamma_reg_any is None:
                    gamma_reg_any = cand
                if cand["reliable"] and cand["physical"] and (gamma_reg_best is None):
                    gamma_reg_best = cand
            if (glob.get("status") == "ok" and glob.get("gamma") is not None
                    and not glob.get("collapse_poor")):
                cand = {"gamma": glob.get("gamma"), "collapse_r2": glob.get("collapse_r2"),
                        "csid": csid}
                if gamma_collapse_best is None:
                    gamma_collapse_best = cand

    # --- global-fit mechanistic γ + reaction-order class -----------------------
    gf_index = artifacts["global_fit_by_protein"]
    global_ode = None
    for k in _match_names(aliases, gf_index):
        recs = gf_index[k]
        if recs:
            s = recs[0]
            global_ode = {
                "csid": s.get("concentration_series_id"),
                "best_mechanism": s.get("best_mechanism"),
                "gamma_mechanistic": s.get("gamma_mechanistic"),
                "reaction_orders": s.get("reaction_orders"),
                "sloppy": s.get("sloppy"),
            }
            break

    # --- equivalence class + licensing (from the concentration-series M5, if any) --
    equiv_class = None
    gates_failed = None
    for k in _match_names(aliases, artifacts["series_modal"]):
        for rec in artifacts["series_modal"][k]:
            mech = rec.get("mechanistic", {}) or {}
            equiv_class = mech.get("equivalence_class") or equiv_class
            gates_failed = mech.get("gates_failed") or gates_failed
            if equiv_class:
                break
        if equiv_class:
            break

    return {
        "modal_regime": modal_regime,
        "modal_regime_count": modal_n,
        "n_curves": n_curves,
        "regime_distribution": dict(sorted(reg_counts.items())),
        "gamma_regression": gamma_reg_best or gamma_reg_any,
        "gamma_regression_reliable_available": gamma_reg_best is not None,
        "gamma_collapse": gamma_collapse_best,
        "global_ode_fit": global_ode,
        "equivalence_class": equiv_class,
        "gates_failed": gates_failed,
        # PRISE refuses to LICENSE a single mechanism corpus-wide — expected & honest.
        "mechanistic_inference_licensed": False,
        "licensed_refused_reason": (
            "EXPECTED: agitation & seeding are unrecorded corpus-wide, so the "
            "quiescent-vs-shaken hard branch is undetermined and M5 refuses to "
            "license a single mechanism (m5_classify Gate C). The reality check is on "
            "regime + γ + equivalence-class CONSISTENCY, not on a named mechanism."),
    }


# ============================================================================ #
# The three consistency checks
# ============================================================================ #
def check_regime(entry: dict, prise: dict) -> dict:
    """REGIME check: cooperative_sigmoidal / threshold_driven -> pass (consistent
    with nucleation-dependent sigmoid); gradual_non_cooperative / no_detectable ->
    contradict on a field-resolved nucleated amyloid."""
    modal = prise.get("modal_regime")
    if modal is None:
        return {"status": "na", "modal_regime": None,
                "note": "no descriptive regime available for this protein"}
    if modal in _NUCLEATED_REGIMES:
        return {"status": "pass", "modal_regime": modal,
                "note": f"modal regime '{modal}' is consistent with a "
                        "nucleation-dependent sigmoid"}
    if modal in _CONTRADICTORY_REGIMES:
        return {"status": "contradict", "modal_regime": modal,
                "note": f"modal regime '{modal}' CONTRADICTS the field consensus of a "
                        "nucleation-dependent cooperative sigmoid for this resolved "
                        "amyloid"}
    return {"status": "na", "modal_regime": modal,
            "note": f"modal regime '{modal}' is neither clearly consistent nor "
                    "contradictory (non-monotonic / anomalous)"}


def check_gamma(entry: dict, prise: dict) -> dict:
    """GAMMA check: only decisive where a STRONG consensus γ band exists (e.g. Aβ42).
    Reports PRISE's γ (reliable regression preferred, else global collapse, else
    mechanistic) and whether it falls inside the pre-registered band."""
    band = entry.get("gamma_band")
    strength = entry.get("gamma_consensus_strength")
    consensus = entry.get("gamma_consensus")
    if band is None or consensus is None:
        return {"status": "na", "consensus_gamma": consensus,
                "note": "no strong consensus γ for this protein — γ is not a decisive "
                        "fixed point here (checked on regime + equivalence class instead)"}

    # pick PRISE's most-authoritative γ: reliable regression > global collapse > mechanistic
    reg = prise.get("gamma_regression")
    coll = prise.get("gamma_collapse")
    ode = prise.get("global_ode_fit") or {}
    picked = None
    if reg and reg.get("reliable") and reg.get("physical") and reg.get("gamma") is not None:
        picked = ("gamma_regression_reliable", reg.get("gamma"))
    elif coll and coll.get("gamma") is not None:
        picked = ("gamma_global_collapse", coll.get("gamma"))
    elif reg and reg.get("gamma") is not None:
        picked = ("gamma_regression_unreliable", reg.get("gamma"))
    elif ode.get("gamma_mechanistic") is not None:
        picked = ("gamma_mechanistic_ode", ode.get("gamma_mechanistic"))

    if picked is None:
        return {"status": "na", "consensus_gamma": consensus, "band": band,
                "prise_gamma": None,
                "note": "no usable γ estimate for this protein (too few anchors / "
                        "no concentration series) — γ fixed point not evaluable"}
    source, value = picked
    lo, hi = band
    in_band = bool(lo <= value <= hi)
    return {
        "status": ("in_band" if in_band else "out_of_band"),
        "consensus_gamma": consensus, "band": band,
        "prise_gamma": float(value), "prise_gamma_source": source,
        "consensus_strength": strength,
        "note": (f"PRISE γ={value:.2f} ({source}) "
                 f"{'falls INSIDE' if in_band else 'falls OUTSIDE'} the consensus band "
                 f"[{lo}, {hi}] (consensus γ≈{consensus}, {strength})"),
    }


def check_mechanism_class(entry: dict, prise: dict) -> dict:
    """MECHANISM-CLASS check: PRISE passes iff the consensus mechanism is NOT EXCLUDED
    — i.e. it is a MEMBER of PRISE's broad equivalence class (and the global-fit
    reaction-order constraint does not positively rule it out). It fails ONLY if
    PRISE's evidence positively excludes the consensus mechanism."""
    consensus_mech = entry.get("consensus_mechanism")
    equiv = prise.get("equivalence_class")
    if not consensus_mech:
        return {"status": "na", "note": "no single consensus mechanism to test"}
    if not equiv:
        return {"status": "na", "consensus_mechanism": consensus_mech,
                "note": "PRISE emitted no equivalence class for this protein (no "
                        "concentration-series mechanistic record) — mechanism class "
                        "not evaluable"}
    member = consensus_mech in equiv
    return {
        "status": ("consistent" if member else "excluded"),
        "consensus_mechanism": consensus_mech,
        "prise_equivalence_class": list(equiv),
        "note": (f"consensus mechanism '{consensus_mech}' IS a member of PRISE's broad "
                 "equivalence class — NOT excluded (PRISE refuses to narrow, which is "
                 "honest, but does not contradict consensus)" if member else
                 f"consensus mechanism '{consensus_mech}' is NOT in PRISE's equivalence "
                 "class — PRISE's evidence would positively exclude it (a contradiction)"),
    }


# ============================================================================ #
# Per-protein verdict
# ============================================================================ #
def evaluate_protein(name: str, entry: dict, artifacts: dict) -> dict:
    """Full per-fixed-point record: consensus, PRISE blind output, the three checks,
    and the CONSISTENT / CONTRADICTION / INSUFFICIENT_DATA verdict.

    Verdict logic (non-contradiction):
      * INSUFFICIENT_DATA — PRISE has no descriptive regime AND no γ for this protein
        (the protein is absent / has no usable curves in the corpus).
      * CONTRADICTION     — the REGIME check contradicts (a field-resolved nucleated
        amyloid called non-cooperative / no-detectable), OR the mechanism class
        positively excludes the consensus mechanism, OR (only for a STRONG-consensus
        γ) γ is out of band. These are the falsifying events.
      * CONSISTENT        — no check contradicts and at least the regime is evaluable
        and passes.
    """
    prise = extract_prise(entry, artifacts)
    c_reg = check_regime(entry, prise)
    c_gam = check_gamma(entry, prise)
    c_mech = check_mechanism_class(entry, prise)

    has_regime = c_reg["status"] != "na"
    has_gamma = c_gam["status"] not in ("na",)
    # a strong-consensus γ that lands out of band is a flag; it only *falsifies*
    # (drives CONTRADICTION) when the γ consensus itself is STRONG.
    gamma_strong = entry.get("gamma_consensus_strength") == "strong"

    if not has_regime and not has_gamma and c_mech["status"] == "na":
        verdict = "INSUFFICIENT_DATA"
        why = ("protein absent / no usable curves in the corpus for a blind PRISE "
               "verdict — reported honestly rather than scored")
    else:
        contradictions = []
        if c_reg["status"] == "contradict":
            contradictions.append("regime")
        if c_mech["status"] == "excluded":
            contradictions.append("mechanism_class")
        if c_gam["status"] == "out_of_band" and gamma_strong:
            contradictions.append("gamma_strong_consensus")
        if contradictions:
            verdict = "CONTRADICTION"
            why = ("PRISE CONTRADICTS the strong field consensus on: "
                   + ", ".join(contradictions)
                   + " — this FALSIFIES the forward model on a resolved fixed point")
        elif not has_regime:
            verdict = "INSUFFICIENT_DATA"
            why = "no descriptive regime available (γ/mechanism alone insufficient)"
        else:
            verdict = "CONSISTENT"
            flags = []
            if c_gam["status"] == "out_of_band":
                flags.append("γ out of a MODERATE-consensus band (flag, not falsifying)")
            why = ("PRISE's blind output does NOT contradict the field consensus "
                   "(regime consistent; consensus mechanism not excluded; "
                   + ("γ in band" if c_gam["status"] == "in_band"
                      else "γ not a decisive fixed point here" if c_gam["status"] == "na"
                      else "; ".join(flags) or "γ flagged") + ")")

    return {
        "protein": name,
        "display_name": entry["display_name"],
        "consensus": {
            "shape": entry["consensus_shape"],
            "mechanism": entry["consensus_mechanism"],
            "gamma_consensus": entry.get("gamma_consensus"),
            "gamma_band": entry.get("gamma_band"),
            "gamma_consensus_strength": entry.get("gamma_consensus_strength"),
            "consensus_strength": entry["consensus_strength"],
            "citation": entry["citation"],
        },
        "prise": {
            "modal_regime": prise["modal_regime"],
            "regime_distribution": prise["regime_distribution"],
            "n_curves": prise["n_curves"],
            "gamma_regression": prise["gamma_regression"],
            "gamma_collapse": prise["gamma_collapse"],
            "global_ode_fit": prise["global_ode_fit"],
            "equivalence_class": prise["equivalence_class"],
            "mechanistic_inference_licensed": prise["mechanistic_inference_licensed"],
            "licensed_refused_reason": prise["licensed_refused_reason"],
        },
        "checks": {"regime": c_reg, "gamma": c_gam, "mechanism_class": c_mech},
        "verdict": verdict,
        "notes": why,
    }


# ============================================================================ #
# Leakage / reserved-proteins discipline
# ============================================================================ #
# Fields that carry human PROSE (not calibration data): a reserved protein NAME
# appearing here is documentation (e.g. a note describing the reality-check role),
# NOT a leak of the protein into the calibration set. The leakage check ignores them
# so a doc-string mention does not masquerade as anchoring leakage.
_PROSE_FIELD_KEYS = {
    "note", "notes", "description", "desc", "citation", "citations",
    "interpretation", "honesty_note", "reason", "role", "purpose", "summary",
    "provenance", "definition", "explainer", "comment", "caveat", "caveats",
    "plain_english", "verdict_plain_english", "headline",
    # roadmap / disclosure lists — narrative, not calibration data. `deferred`
    # holds "what we have NOT built" notes (which legitimately reference the §6C
    # reserved-protein reality check by name); it is documentation, never anchoring.
    "deferred", "limitations", "todo", "future_work", "disclosures", "scope",
}


def _data_bearing_values(obj, _in_prose_field: bool = False):
    """Yield every STRING that sits in a data-bearing position of the calibration
    artifact — i.e. NOT inside a prose/note/citation field. Reserved protein names
    are legitimately mentioned in prose (documenting the §6C role); only an
    appearance as actual anchoring/calibration DATA is a leak."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            is_prose = _in_prose_field or (isinstance(k, str)
                                           and k.lower() in _PROSE_FIELD_KEYS)
            # keys are themselves data-bearing (e.g. a per-protein anchor map key)
            if isinstance(k, str) and not _in_prose_field:
                yield k
            yield from _data_bearing_values(v, is_prose)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _data_bearing_values(v, _in_prose_field)
    elif isinstance(obj, str) and not _in_prose_field:
        yield obj


def leakage_check(artifacts: dict) -> dict:
    """Verify the reserved fixed-point proteins are NOT used for anchoring /
    calibration (§8 budget; §8.1 leakage discipline).

    The Service-C calibration is SYNTHETIC (known-truth generated curves — no real
    protein anchors it). We confirm no reserved protein name appears in any
    DATA-BEARING position of the calibration artifact. A name that appears only in a
    prose/note/citation field (documenting the §6C role) is explicitly NOT a leak —
    the earlier naive whole-blob substring scan produced a false positive on exactly
    such a doc-string mention (a note reading '…reality check (Aβ42/α-syn/insulin/β2m
    vs field consensus)'), so prose fields are excluded here."""
    reserved = sorted({a for e in CONSENSUS.values() for a in e["aliases"]})
    sc = artifacts.get("service_c", {}) or {}
    data_blob = "\n".join(_data_bearing_values(sc)).lower()
    leaked = sorted({name for name in reserved if name.lower() in data_blob})
    # also record prose-only mentions transparently (informational, not a leak)
    full_blob = json.dumps(sc).lower()
    prose_only = sorted({name for name in reserved
                         if name.lower() in full_blob and name not in leaked})
    return {
        "reserved_proteins": reserved,
        "service_c_is_synthetic": True,
        "reserved_names_in_calibration_data": leaked,
        "reserved_names_in_prose_only": prose_only,
        "no_leakage": (len(leaked) == 0),
        "note": ("these proteins are reserved SOLELY for the §6C blind reality check "
                 "and are NOT used for anchoring / calibration / demonstrators (§8 "
                 "budget). Service-C calibration is synthetic (known-truth), so no real "
                 "protein anchors it; this check confirms none of the reserved names "
                 "appears in a DATA-BEARING position of the calibration artifact "
                 "(prose/citation mentions of the reality-check role are excluded — "
                 "they document the discipline, they do not violate it)."),
    }


# ============================================================================ #
# Orchestration
# ============================================================================ #
def run_reality_check(data_dir: Path) -> dict:
    """Evaluate every pre-registered fixed point that EXISTS in the corpus, tally the
    overall non-contradiction result, and attach the leakage/reserved discipline +
    the honest framing. Deterministic; never raises."""
    artifacts = load_artifacts(Path(data_dir))
    per_protein = [evaluate_protein(name, entry, artifacts)
                   for name, entry in CONSENSUS.items()]

    n_consistent = sum(1 for p in per_protein if p["verdict"] == "CONSISTENT")
    n_contra = sum(1 for p in per_protein if p["verdict"] == "CONTRADICTION")
    n_insuf = sum(1 for p in per_protein if p["verdict"] == "INSUFFICIENT_DATA")
    n_eval = n_consistent + n_contra          # fixed points actually evaluated

    if n_contra > 0:
        plain = (f"FALSIFIED on {n_contra} of {n_eval} evaluated fixed point(s): PRISE "
                 "contradicts the field consensus on a mechanistically-RESOLVED "
                 "protein. This is a real finding about the forward model, not a bug to "
                 "hide — investigate before trusting corpus-wide verdicts.")
    elif n_eval == 0:
        plain = ("No fixed point was evaluable on the current artifacts (all "
                 "INSUFFICIENT_DATA) — the test could not be run; regenerate the "
                 "corpus artifacts and re-run.")
    else:
        plain = (f"PASSED (non-contradiction) on all {n_eval} evaluated fixed point(s): "
                 "PRISE's blind, honest output (descriptive regime + γ + mechanistic "
                 "equivalence class) does NOT contradict the field consensus on any "
                 "mechanistically-resolved protein it can see. This is a thermometer "
                 "reading against a few fixed points — NOT proof of correctness and NOT "
                 "statistical calibration.")

    return {
        "version": REALITY_CHECK_VERSION,
        "reality_check_result": {
            "n_fixed_points_registered": len(CONSENSUS),
            "n_fixed_points_evaluated": n_eval,
            "n_consistent": n_consistent,
            "n_contradiction": n_contra,
            "n_insufficient": n_insuf,
            "verdict_plain_english": plain,
        },
        "fixed_points": per_protein,
        "reserved_proteins_note": leakage_check(artifacts),
        "falsification_note": (
            "A few decisive fixed points, not a statistical calibration — a "
            "CONTRADICTION on a strong-consensus case FALSIFIES the forward model "
            "(§6C, decision row C1). This is a thermometer against fixed points."),
        "honesty_note": (
            "NON-CONTRADICTION test on a HANDFUL of fixed points (small n). NOT proof "
            "of correctness and NOT statistical calibration. PRISE's refusal to LICENSE "
            "a single mechanism corpus-wide is EXPECTED (agitation/seeding unknown) — "
            "the test is on regime + γ + equivalence-class CONSISTENCY, never on a "
            "named mechanism. A 'pass' does NOT validate the whole engine."),
    }


# ============================================================================ #
# CLI
# ============================================================================ #
def _fmt_g(v):
    return "—" if v is None else (f"{v:.2f}" if isinstance(v, (int, float)) else str(v))


def _print_summary(report: dict) -> None:
    res = report["reality_check_result"]
    print("\n" + "=" * 74)
    print(f"PRISE BLIND EXTERNAL REALITY CHECK  ({report['version']})")
    print("=" * 74)
    for p in report["fixed_points"]:
        pr = p["prise"]
        reg = pr.get("gamma_regression") or {}
        g = reg.get("gamma") if reg else None
        gband = p["consensus"]["gamma_band"]
        cstr = p["consensus"]["consensus_strength"]
        print(f"\n  {p['display_name']}  [{p['verdict']}]  (consensus: {cstr})")
        print(f"    consensus : {p['consensus']['shape']}")
        print(f"    PRISE regime : {pr.get('modal_regime')}  "
              f"(dist {pr.get('regime_distribution')})")
        print(f"    γ : PRISE={_fmt_g(g)}  consensus={_fmt_g(p['consensus']['gamma_consensus'])}"
              f"  band={gband}  -> {p['checks']['gamma']['status']}")
        print(f"    checks : regime={p['checks']['regime']['status']} · "
              f"gamma={p['checks']['gamma']['status']} · "
              f"mechanism_class={p['checks']['mechanism_class']['status']}")
        print(f"    -> {p['notes']}")
    print("\n" + "-" * 74)
    print(f"  evaluated={res['n_fixed_points_evaluated']}  "
          f"CONSISTENT={res['n_consistent']}  "
          f"CONTRADICTION={res['n_contradiction']}  "
          f"INSUFFICIENT={res['n_insufficient']}")
    lk = report["reserved_proteins_note"]
    print(f"  reserved/leakage: no_leakage={lk['no_leakage']} "
          f"(reserved {len(lk['reserved_proteins'])} proteins, solely for §6C)")
    print(f"\n  {res['verdict_plain_english']}")
    print("=" * 74)


def main(argv=None) -> int:
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")     # Windows cp1252 safety
    except Exception:                                # pragma: no cover
        pass
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        description="PRISE blind external reality check (§6C falsification test)")
    ap.add_argument("--data", type=Path, default=root / "data" / "processed",
                    help="directory holding the processed artifacts")
    ap.add_argument("--output", type=Path, default=None,
                    help="output path (default <data>/reality_check.json)")
    args = ap.parse_args(argv)

    report = run_reality_check(args.data)
    out_path = args.output or (Path(args.data) / "reality_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[reality-check] wrote {out_path}")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
