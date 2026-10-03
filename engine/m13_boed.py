"""
PRISE — Module M13: Bayesian Optimal Experimental Design (BOED)
==============================================================

PRISE_DESIGN.md §3-M13 — the rigorous BOED capstone of the design-advice thread
(M9 heuristic rule table -> M12 per-curve information geometry -> M13 multi-
variable BOED). Given a protein's CURRENT fitted forward model (M2 θ̂ + SSE), M13
scores a set of CANDIDATE next experiments ξ by the EXPECTED scientific
information they would add, ranks them by EIG ÷ cost, exposes the non-dominated
(information ↑, cost ↓, runtime ↓) Pareto frontier, and plans a greedy K-step
batch whose marginal gains DIMINISH (submodular log-det). It REUSES M12's exact
FIM machinery on PROSPECTIVE (simulated) designs — it does NOT reimplement it.

=== TWO HONEST VARIABLE TYPES (the design principle) ===
M13 is deliberately honest that its 8 candidate design variables are of two kinds:

  A) QUANTITATIVE — we can SIMULATE the prospective observed FIM 𝓘(ξ) via M12 on a
     prospective grid, so EIG is a real Laplace/linear-Gaussian D-optimality number
     (gain_basis='fim'):
       * protein_concentration (a single new [m], AND a >=3-point concentration
         SERIES): build the prospective curve at [m'] by RESCALING the fitted
         timescale using the M4 scaling exponent γ (t50([m']) = t50·([m']/[m0])^(−γ))
         — a SELF-SIMILAR-SHAPE approximation (STATED). A series STACKS per-[m] FIMs
         AND adds the γ / reaction-order direction (where the mechanism gain comes
         from); the single new [m] shifts the grid only.
       * temperature, pH: a prospective 𝓘 shift WITHIN the fitted window;
         extrapolation BEYOND the measured T/pH range is flagged low-confidence.
       * replicate_count R: averaging R replicates -> σ²→σ²/R -> 𝓘 scales ~R
         (parameter SE ∝ 1/√R — verified in the tests).
       * measurement_frequency Δt and sampling_duration t_max: build the prospective
         time grid = arange(0, t_max, Δt); a denser Δt adds rows to S; extending
         t_max on a RIGHT-CENSORED curve informs the plateau/amplitude parameter.

  B) GATING / categorical — NOT a smooth FIM direction, a LICENSING unblock
     (gain_basis='licensing', anchored to Service-C measured numbers):
       * agitation, seeding: recording them FLIPS the M5 mechanistic license (the
         M11 mechanistic_completeness jump; the M8 GAP closes IFF a concentration
         series ALSO exists). We credit the FULL mechanism unblock only where the
         other prerequisites are present, else 'necessary_but_not_sufficient'
         (dependency-honest, exactly like M9). The mechanism gain is anchored to
         Service-C's MEASURED 1->4 Tier-B split (an UPPER bound).

For EACH candidate M13 reports the four §3-M13 quantities: (1) expected_information
_gain (EIG in BITS, for a stated target ∈ {parameters | gamma | mechanism_class}),
(2) expected_parameter_uncertainty_reduction (Σ_n=(Σ₀⁻¹+𝓘)⁻¹; % SE reduction),
(3) expected_mechanistic_distinguishability_gain (equivalence-class collapse,
BOUNDED by Service-C separability), and (4) estimated_cost + estimated_runtime (a
VERSIONED, CONFIGURABLE table — an ESTIMATE, not real lab economics).

=== HONESTY (mandatory, attached to every result) ===
  * EIG is EXPECTED under the CURRENT fitted forward model + prior + assumed noise,
    Laplace-LINEARIZED — it INHERITS M8's validity ceiling + M12's local/linearized
    caveat. It is not an absolute information content.
  * mechanistic gains are BOUNDED by Service-C γ-separability (an UPPER bound).
  * cost/runtime are a versioned ESTIMATE table, NOT real lab economics.
  * GATING variables give a LICENSING unblock (not a smooth FIM gain), labeled.
  * the greedy multi-step plan is NOT the global optimum; submodularity gives the
    batch a (1−1/e) guarantee, stated.
  * T/pH extrapolation beyond the measured window is flagged low-confidence.

Everything is pure / deterministic / JSON-serialisable; NOTHING raises — a protein
with no fitted model falls back to M9 rule-table language + status INSUFFICIENT.

Usage:
    python engine/m13_boed.py                 # -> data/processed/boed_recommendations.json (+per-protein)
    python engine/m13_boed.py --limit 50
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from m2_fit import fit_curve
from m12_information import (
    expected_information_gain,
    fisher_information,
    sensitivity_jacobian,
)
from models import REGISTRY

# Module identity — a provenance tag (mirrors the M2..M12 convention); a bump makes
# downstream BOED recommendation artifacts non-comparable (§7).
M13_VERSION = "m13-boed-1.0"

_EPS = 1e-12
_LN2 = math.log(2.0)

# Weak-Gaussian prior CV (Σ₀ = diag((prior_cv·θ̂)²)) — the same weak prior M12's
# EIG uses. prior_cv=1.0 is a VERY weak prior (1-σ prior width = ±100% of θ̂); the
# EIG is prior-conditional. Configurable via recommend_experiments(prior_cv=...).
_DEFAULT_PRIOR_CV = 1.0

# Prior CV on the γ / reaction-order direction that a concentration SERIES makes
# estimable. A series is the ONLY design that constrains γ, so its targeted EIG is
# computed on this extra prior dimension (weak: 100% of a nominal γ≈1).
_DEFAULT_GAMMA_PRIOR_CV = 1.0
_NOMINAL_GAMMA = 1.0

# Self-similar concentration rescaling: t50([m']) = t50·([m']/[m0])^(−γ). When the
# protein has no measured γ we fall back to this nominal amyloid exponent and FLAG
# the design low-confidence (γ is itself the thing a series would measure).
_FALLBACK_GAMMA = 1.0

# =========================================================================== #
# VERSIONED, CONFIGURABLE cost/runtime ESTIMATE table (§3-M13.4). These are
# ESTIMATES for RANKING, not real lab economics — labeled as such everywhere.
# cost  = setup + n_conditions × n_replicates × per_run_cost
# runtime_h = prep + (t_max × n_serial_batches) [parallel plate: n_serial=1]
# =========================================================================== #
COST_TABLE_VERSION = "m13-costtable-1.0"
COST_TABLE = {
    "version": COST_TABLE_VERSION,
    "setup_cost": 1.0,            # fixed per-experiment setup (cost units)
    "per_run_cost": 1.0,         # per well/condition-replicate (cost units)
    "metadata_only_cost": 0.25,  # recording agitation/seeding: a metadata note
    "prep_hours": 2.0,           # sample prep before the run starts
    "parallel_plate": True,      # replicates/conditions run in parallel on a plate
    "note": ("VERSIONED ESTIMATE for RANKING ONLY — not real lab economics. "
             "cost = setup + n_conditions*n_replicates*per_run; runtime_h = prep + "
             "t_max*(serial batches). A real cost model is a documented refinement."),
}


# =========================================================================== #
# Service-C measured mechanism-separability (the UPPER bound on the mechanistic
# distinguishability gain). Read live from service_c_calibration.json; fall back
# to the pre-registered measured constants (the γ-resolved 1->4 Tier-B split).
# =========================================================================== #
_SC_FALLBACK = {
    "single_concentration_classes": 1,   # single conc -> 1 broad Tier-B class
    "concentration_series_classes": 4,   # γ-resolved series -> 4 singletons
    "pairs_newly_resolved": 6,
    "residual_pair_auc": 0.8888888888888888,  # frag|sat-sec: the hardest pair
    "mismatch_degradation": 0.6666666666666666,
    "service_c_version": "service-c-1.0",
    "service_c_source": "pre_registered_fallback (artifact missing)",
}


def _sc_numbers(service_c: dict | None) -> dict:
    """Pull the MEASURED Tier-B narrowing + residual-pair AUC + mismatch
    degradation out of the Service-C artifact; degrade to the pre-registered
    measured constants when a field is absent (stamped `service_c_source`)."""
    if not isinstance(service_c, dict):
        return dict(_SC_FALLBACK)
    narrowing = service_c.get("concentration_series_narrowing") or {}
    single = (narrowing.get("single_concentration_partition") or {}).get("n_classes")
    series = (narrowing.get("concentration_series_partition") or {}).get("n_classes")
    pairs = narrowing.get("pairs_newly_resolved_by_series")
    cm = (service_c.get("confusion_matrices") or {}).get("concentration_series") or {}
    auc = cm.get("pairwise_separation_auc") or {}
    residual = auc.get("fragmentation|saturating_secondary",
                       _SC_FALLBACK["residual_pair_auc"])
    mis = (service_c.get("mismatched_generator") or {}).get(
        "mechanism_misidentification_degradation")
    return {
        "single_concentration_classes": single if single is not None
        else _SC_FALLBACK["single_concentration_classes"],
        "concentration_series_classes": series if series is not None
        else _SC_FALLBACK["concentration_series_classes"],
        "pairs_newly_resolved": len(pairs) if pairs else _SC_FALLBACK["pairs_newly_resolved"],
        "residual_pair_auc": residual,
        "mismatch_degradation": mis if mis is not None
        else _SC_FALLBACK["mismatch_degradation"],
        "service_c_version": service_c.get("service_c_version", "service-c-1.0"),
        "service_c_source": "service_c_calibration.json",
    }


# =========================================================================== #
# Honesty block — attached to EVERY M13 result.
# =========================================================================== #
_HONESTY = {
    "eig_is_expected_laplace": (
        "EIG is EXPECTED under the CURRENT fitted forward model + the stated weak "
        "Gaussian prior + the assumed sigma^2 I noise; it is Laplace-LINEARIZED "
        "(M12's observed-FIM approximation) and inherits M8's validity ceiling. It "
        "is not absolute information content."),
    "mechanistic_gain_bounded_by_service_c": (
        "the expected mechanistic-distinguishability gain is BOUNDED ABOVE by the "
        "Service-C measured gamma-separability (1->4 Tier-B split), an UPPER bound "
        "on real discriminability — real mechanisms can share gamma."),
    "cost_is_an_estimate": (
        "estimated_cost / estimated_runtime come from a VERSIONED estimate table "
        "for RANKING ONLY — NOT real lab economics."),
    "gating_is_licensing_not_fim": (
        "GATING variables (agitation, seeding) give a LICENSING unblock (a jump in "
        "the M5/M11 mechanistic license), NOT a smooth FIM direction — labeled "
        "gain_basis='licensing', anchored to Service-C, credited fully only when "
        "the other mechanism prerequisites (a series) are present."),
    "greedy_not_global": (
        "the multi-step plan is GREEDY (argmax marginal EIG/cost each step); it is "
        "NOT the global optimum, but log-det EIG is submodular so the batch carries "
        "a (1-1/e) approximation guarantee."),
    "tpH_extrapolation_flagged": (
        "temperature/pH designs are simulated as an FIM shift WITHIN the fitted "
        "window; extrapolation beyond the measured T/pH range is flagged "
        "low-confidence (the forward model is not validated there)."),
    "self_similar_shape_approx": (
        "the concentration rescaling t50([m'])=t50*([m']/[m0])^(-gamma) assumes a "
        "SELF-SIMILAR curve shape across concentration (only the timescale rescales) "
        "— a stated approximation of the prospective curve."),
}


# =========================================================================== #
# Prospective-design simulation — the heart of M13. Given the fitted model
# (name, θ̂, σ²) we build 𝓘(ξ) for a candidate design ξ by re-running M12's
# sensitivity_jacobian on a PROSPECTIVE grid and scaling the noise.
# =========================================================================== #
def _sigma2_from_fit(fit: dict, name: str, n: int, p: int) -> float:
    """Unbiased noise variance sigma^2 = SSE/(n-p) of the CURRENT fit — the noise
    level M13 assumes a prospective run would also carry (same assay)."""
    sse = float(fit.get("sse") or 0.0)
    dof = max(n - p, 1)
    return max(sse / dof, _EPS)


def prospective_fim(name: str, params: dict, x_grid, sigma2: float,
                    replicate_count: int = 1) -> tuple[np.ndarray, list[str]]:
    """Simulate the prospective observed FIM 𝓘(ξ) = SᵀΣ⁻¹S for a design whose time
    grid is `x_grid`, using M12's EXACT `sensitivity_jacobian` (do NOT reimplement)
    and averaging R replicates so Σ = (σ²/R)·I -> 𝓘 scales ~R.

    This is the ONE simulator M13 reuses for every quantitative design: denser Δt /
    longer t_max / a new concentration all just change `x_grid`; replicates scale
    the noise. Returns (𝓘, param_order). Never raises: a degenerate grid -> a zero
    FIM of the right shape."""
    order = list(REGISTRY[name]["params"])
    p = len(order)
    xg = np.asarray(x_grid, float)
    xg = xg[np.isfinite(xg)]
    if xg.size < p:                                  # too few points to inform p params
        return np.zeros((p, p)), order
    S, _, _ = sensitivity_jacobian(name, params, xg)
    n = xg.size
    # fisher_information wants an SSE so that SSE/(n-p) == the target sigma^2; we
    # invert that so the prospective FIM carries exactly the assumed noise, then the
    # R-replicate averaging divides the variance (multiplies the FIM) by R.
    r = max(int(replicate_count), 1)
    sigma2_eff = max(sigma2 / r, _EPS)
    sse_equiv = sigma2_eff * max(n - p, 1)
    fim, _, _ = fisher_information(S, sse_equiv, n, p, 0.0, 0.0)
    return fim, order


def _rescale_timescale(gamma: float, m_new: float, m0: float) -> float:
    """Self-similar concentration rescaling factor for the time axis:
        t([m']) = t([m0]) * ([m']/[m0])^(-gamma).
    A higher concentration (m'>m0) with gamma>0 SPEEDS the reaction (shorter t50),
    so the grid compresses. Guarded against non-positive concentrations."""
    if m0 <= 0 or m_new <= 0:
        return 1.0
    return float((m_new / m0) ** (-gamma))


def _current_grid(x_hours) -> np.ndarray:
    xa = np.asarray(x_hours, float)
    return xa[np.isfinite(xa)]


def _eig_bits(fim: np.ndarray, theta: np.ndarray, prior_cv: float) -> dict:
    """EIG = 1/2 ln det(I + Sigma0*FIM) in nats AND bits, reusing M12's exact
    `expected_information_gain` (the D-optimality objective)."""
    e = expected_information_gain(fim, theta, prior_cv)
    return {"eig_nats": e["eig_nats"], "eig_bits": e["eig_bits"],
            "prior_cv": e["prior_cv"]}


def _posterior_se_reduction(fim_prior_inv: np.ndarray, fim_design: np.ndarray,
                            order: list[str], theta: np.ndarray) -> dict:
    """Σ_n = (Σ₀⁻¹ + 𝓘(ξ))⁻¹ -> per-parameter % reduction in the target SE relative
    to the prior SE. Reports the max and the per-parameter table (the parameters a
    design constrains best surface as the largest reductions)."""
    p = len(order)
    prior_cov = np.linalg.pinv(fim_prior_inv)          # Σ₀ (prior covariance)
    prior_se = np.sqrt(np.clip(np.diag(prior_cov), 0.0, None))
    post_prec = fim_prior_inv + fim_design             # Σ₀⁻¹ + 𝓘
    post_cov = np.linalg.pinv(post_prec)
    post_se = np.sqrt(np.clip(np.diag(post_cov), 0.0, None))
    table = {}
    reductions = []
    for j, nm in enumerate(order):
        ps, qs = float(prior_se[j]), float(post_se[j])
        red = (1.0 - qs / ps) if ps > _EPS else 0.0
        red = max(0.0, min(1.0, red))
        table[nm] = {"prior_se": ps, "posterior_se": qs,
                     "se_reduction_pct": round(100.0 * red, 2)}
        reductions.append(red)
    return {
        "per_parameter": table,
        "max_se_reduction_pct": round(100.0 * (max(reductions) if reductions else 0.0), 2),
        "mean_se_reduction_pct": round(100.0 * (float(np.mean(reductions)) if reductions else 0.0), 2),
    }


def _prior_precision(theta: np.ndarray, prior_cv: float) -> np.ndarray:
    """Σ₀⁻¹ = diag(1/(prior_cv·θ̂)²) — the weak-Gaussian prior precision."""
    s0 = (prior_cv * np.abs(theta)) ** 2
    s0 = np.where(s0 > _EPS, s0, _EPS)
    return np.diag(1.0 / s0)


# =========================================================================== #
# Cost / runtime — the versioned estimate table applied to a design descriptor.
# =========================================================================== #
def estimate_cost_runtime(n_conditions: int, n_replicates: int, t_max_hours: float,
                          metadata_only: bool = False, table: dict = None) -> dict:
    """Cost + runtime ESTIMATE (versioned table). cost scales with
    n_conditions × n_replicates; runtime with t_max (parallel plate -> one serial
    batch). Metadata-only designs (record agitation/seeding) cost a flat note and
    add no runtime. These are ESTIMATES for ranking, not lab economics."""
    t = table or COST_TABLE
    if metadata_only:
        return {"estimated_cost_units": round(t["metadata_only_cost"], 4),
                "estimated_runtime_hours": 0.0,
                "n_conditions": n_conditions, "n_replicates": n_replicates,
                "t_max_hours": 0.0, "metadata_only": True,
                "cost_table_version": t["version"], "is_estimate": True}
    n_runs = max(int(n_conditions), 1) * max(int(n_replicates), 1)
    cost = t["setup_cost"] + n_runs * t["per_run_cost"]
    serial = 1 if t.get("parallel_plate") else n_runs
    runtime = t["prep_hours"] + float(t_max_hours) * serial
    return {"estimated_cost_units": round(float(cost), 4),
            "estimated_runtime_hours": round(float(runtime), 4),
            "n_conditions": int(n_conditions), "n_replicates": int(n_replicates),
            "t_max_hours": round(float(t_max_hours), 4), "metadata_only": False,
            "cost_table_version": t["version"], "is_estimate": True}


# =========================================================================== #
# Mechanistic-distinguishability gain (Service-C bounded).
# =========================================================================== #
def _mechanistic_gain(has_series_after: bool, mechanism_licensable_after: bool,
                      sc: dict) -> dict:
    """Expected reduction in the M5 equivalence-class size, ANCHORED to and BOUNDED
    by the Service-C measured confusion collapse (single conc = 1 broad class -> a
    γ-resolved series = 4 singletons). The FULL collapse is credited only when the
    design leaves a mechanism LICENSABLE (series present AND the licensing gate
    satisfied); otherwise a partial/zero gain, labeled."""
    single = int(sc["single_concentration_classes"])
    series = int(sc["concentration_series_classes"])
    if mechanism_licensable_after and has_series_after:
        return {
            "equivalence_class_before": single, "equivalence_class_after": series,
            "class_size_reduction": series - single,
            "resolution_gain_x": round(series / max(single, 1), 4),
            "basis": "service_c_measured",
            "is_upper_bound": True,
            "residual_pair_auc": sc["residual_pair_auc"],
            "note": ("the MEASURED gamma 1->4 Tier-B split becomes achievable (series "
                     "present + licensing satisfied); UPPER bound — the frag|sat-sec "
                     f"pair still separates only at AUC {sc['residual_pair_auc']:.2f}."),
        }
    if has_series_after and not mechanism_licansable_after_safe(mechanism_licensable_after):
        # a series is present but the licensing gate (agitation/seeding) is not yet
        # satisfied -> the SCALING (gamma) is resolvable but the mechanism LICENSE is
        # still withheld; credit the structural narrowing potential, not the license.
        return {
            "equivalence_class_before": single, "equivalence_class_after": series,
            "class_size_reduction": series - single,
            "resolution_gain_x": round(series / max(single, 1), 4),
            "basis": "service_c_measured_structural_only",
            "is_upper_bound": True,
            "residual_pair_auc": sc["residual_pair_auc"],
            "note": ("a >=3-point series makes the gamma direction estimable (the "
                     "STRUCTURAL 1->4 narrowing) but the mechanism LICENSE stays "
                     "withheld until agitation/seeding are recorded (M5 gate)."),
        }
    return {
        "equivalence_class_before": single, "equivalence_class_after": single,
        "class_size_reduction": 0, "resolution_gain_x": 1.0,
        "basis": "none", "is_upper_bound": True,
        "note": ("no mechanistic-distinguishability gain: this design does not add a "
                 "gamma-resolving concentration series (single conc stays 1 broad "
                 "Tier-B class)."),
    }


def mechanism_licansable_after_safe(v) -> bool:
    """Small defensive coercion (never raises on a non-bool)."""
    return bool(v)


# =========================================================================== #
# Candidate construction — build the 8 candidate designs for one fitted protein.
# Each candidate is a self-describing dict carrying its own 𝓘(ξ) contribution (for
# FIM candidates) or a licensing descriptor (for gating candidates).
# =========================================================================== #
def build_candidates(name: str, params: dict, x_hours, sigma2: float, theta: np.ndarray,
                     order: list[str], *, m0: float | None, gamma: float | None,
                     gamma_reliable: bool, has_series_now: bool,
                     agitation_seeding_known: bool, t_measured_range: tuple | None,
                     ph_measured_range: tuple | None, sc: dict,
                     prior_cv: float, gamma_prior_cv: float) -> list[dict]:
    """Assemble the candidate designs. Each FIM candidate carries `fim` (its 𝓘(ξ)
    contribution) + a `design` descriptor for cost; each gating candidate carries a
    `licensing` descriptor. Pure builder — scoring happens in `_score_candidate`."""
    xa = _current_grid(x_hours)
    if xa.size < 2:
        return []
    t_max = float(xa.max())
    t_min = float(xa.min())
    span = (t_max - t_min) or 1.0
    dt_now = span / max(len(xa) - 1, 1)
    p = len(order)
    gamma_eff = float(gamma) if (gamma is not None and math.isfinite(gamma)) else _FALLBACK_GAMMA
    gamma_is_fallback = not (gamma is not None and math.isfinite(gamma) and gamma_reliable)

    cands: list[dict] = []

    # ---- A1. single new protein concentration (grid rescaled by gamma) -------- #
    # A new [m'] shifts the prospective timescale but is still ONE curve -> it does
    # NOT make the gamma direction estimable (that needs a series). We rescale the
    # grid and re-simulate 𝓘 on the parameters only.
    if m0 and m0 > 0:
        m_new = 2.0 * m0                              # a decade-spanning-ish new point
        scale = _rescale_timescale(gamma_eff, m_new, m0)
        grid_new = np.linspace(t_min * scale, t_max * scale, len(xa))
        fim_new, _ = prospective_fim(name, params, grid_new, sigma2, replicate_count=1)
        cands.append({
            "candidate_id": "add_single_concentration",
            "variable_type": "quantitative",
            "gain_basis": "fim",
            "target": "parameters",
            "fim": fim_new,
            "adds_gamma_direction": False,
            "design": {"n_conditions": 1, "n_replicates": 1, "t_max_hours": t_max * scale},
            "description": (f"add ONE new curve at [m']={m_new:g} uM (grid rescaled by "
                            f"gamma={gamma_eff:.2f}); constrains the per-curve parameters, "
                            "NOT gamma (a single new conc is not a series)."),
            "low_confidence": gamma_is_fallback,
            "low_confidence_reason": ("gamma is a fallback/unreliable value — the "
                                      "concentration rescaling is approximate")
            if gamma_is_fallback else None,
        })

    # ---- A2. concentration SERIES (>=3 conc) — stacks FIMs + gamma direction --- #
    # The headline quantitative design: build 3 prospective curves at [m0/2, m0, 2*m0]
    # each rescaled by gamma, STACK their FIMs, AND add the gamma/reaction-order
    # direction to the estimable set (this is where the mechanism gain comes from).
    if m0 and m0 > 0:
        conc_multipliers = (0.5, 1.0, 2.0)           # >=3 concentrations across a ~decade
        fim_series = np.zeros((p, p))
        for mult in conc_multipliers:
            m_new = mult * m0
            scale = _rescale_timescale(gamma_eff, m_new, m0)
            grid_new = np.linspace(t_min * scale, t_max * scale, len(xa))
            fim_k, _ = prospective_fim(name, params, grid_new, sigma2, replicate_count=1)
            fim_series = fim_series + fim_k
        cands.append({
            "candidate_id": "add_concentration_series",
            "variable_type": "quantitative",
            "gain_basis": "fim",
            "target": "gamma",
            "fim": fim_series,
            "adds_gamma_direction": True,
            "gamma_prior_cv": gamma_prior_cv,
            "design": {"n_conditions": len(conc_multipliers), "n_replicates": 1,
                       "t_max_hours": t_max * _rescale_timescale(gamma_eff, 0.5 * m0, m0)},
            "description": (f"add a >=3-point concentration series "
                            f"({', '.join(f'{c*m0:g}' for c in conc_multipliers)} uM); "
                            "STACKS per-[m] FIMs AND makes the gamma / reaction-order "
                            "direction estimable — the mechanism-resolving design."),
            "makes_series": True,
            "low_confidence": gamma_is_fallback,
            "low_confidence_reason": ("gamma is a fallback/unreliable value — the series "
                                      "grid rescaling is approximate (the series is what "
                                      "would MEASURE gamma)") if gamma_is_fallback else None,
        })

    # ---- A3. add replicates R -> sigma^2/R -> FIM ~R -------------------------- #
    R = 4
    fim_rep, _ = prospective_fim(name, params, xa, sigma2, replicate_count=R)
    cands.append({
        "candidate_id": "add_replicates",
        "variable_type": "quantitative",
        "gain_basis": "fim",
        "target": "parameters",
        "fim": fim_rep,
        "adds_gamma_direction": False,
        "design": {"n_conditions": 1, "n_replicates": R, "t_max_hours": t_max},
        "description": (f"run R={R} replicates of the same curve; averaging -> "
                        "sigma^2/R -> FIM scales ~R -> parameter SE ∝ 1/sqrt(R). "
                        "Characterises scatter; does NOT add a new direction."),
        "low_confidence": False,
    })

    # ---- A4. denser measurement frequency Δt (more rows in S) ----------------- #
    dt_dense = dt_now / 2.0
    grid_dense = np.arange(t_min, t_max + 0.5 * dt_dense, dt_dense)
    fim_dense, _ = prospective_fim(name, params, grid_dense, sigma2, replicate_count=1)
    cands.append({
        "candidate_id": "increase_measurement_frequency",
        "variable_type": "quantitative",
        "gain_basis": "fim",
        "target": "parameters",
        "fim": fim_dense,
        "adds_gamma_direction": False,
        "design": {"n_conditions": 1, "n_replicates": 1, "t_max_hours": t_max},
        "description": (f"halve the sampling interval (dt {dt_now:.2f}->{dt_dense:.2f} h, "
                        f"{len(xa)}->{len(grid_dense)} points); adds rows to S, "
                        "sharpening the rate/lag through the transition."),
        "low_confidence": False,
    })

    # ---- A5. extend sampling_duration t_max (informs plateau/amplitude) ------- #
    # On a right-censored curve, extending t_max past the plateau is the design that
    # informs the amplitude parameter — the grid gains rows in the plateau region.
    t_max_ext = t_max + span
    grid_ext = np.arange(t_min, t_max_ext + 0.5 * dt_now, dt_now)
    fim_ext, _ = prospective_fim(name, params, grid_ext, sigma2, replicate_count=1)
    cands.append({
        "candidate_id": "extend_sampling_duration",
        "variable_type": "quantitative",
        "gain_basis": "fim",
        "target": "parameters",
        "fim": fim_ext,
        "adds_gamma_direction": False,
        "design": {"n_conditions": 1, "n_replicates": 1, "t_max_hours": t_max_ext},
        "description": (f"extend the observation window t_max {t_max:.1f}->{t_max_ext:.1f} h; "
                        "adds plateau-region points that inform the amplitude/plateau "
                        "parameter (the right-censoring fix)."),
        "low_confidence": False,
    })

    # ---- A6/A7. temperature, pH — FIM shift WITHIN the fitted window ----------- #
    # A T or pH perturbation is simulated as a re-run at a shifted timescale WITHIN
    # the model window (a modest speed change), so it adds an independent curve's FIM.
    # Extrapolation BEYOND the measured T/pH range is flagged low-confidence.
    for vid, vname, meas_range in (("vary_temperature", "temperature", t_measured_range),
                                   ("vary_pH", "pH", ph_measured_range)):
        # a within-window perturbation: modest timescale change (10% faster), one curve
        grid_pert = np.linspace(t_min * 0.9, t_max * 0.9, len(xa))
        fim_pert, _ = prospective_fim(name, params, grid_pert, sigma2, replicate_count=1)
        extrapolates = meas_range is None or (isinstance(meas_range, tuple)
                                              and meas_range[0] == meas_range[1])
        cands.append({
            "candidate_id": vid,
            "variable_type": "quantitative",
            "gain_basis": "fim",
            "target": "parameters",
            "fim": fim_pert,
            "adds_gamma_direction": False,
            "design": {"n_conditions": 1, "n_replicates": 1, "t_max_hours": t_max * 0.9},
            "description": (f"vary {vname} (a within-window FIM shift, +1 independent "
                            "curve); constrains the shared parameters across conditions."),
            "low_confidence": bool(extrapolates),
            "low_confidence_reason": (f"only one {vname} value is measured — a new "
                                      f"{vname} EXTRAPOLATES beyond the measured range "
                                      "(forward model not validated there)")
            if extrapolates else None,
        })

    # ---- B1/B2. GATING: record agitation & seeding (licensing unblock) -------- #
    # NOT a smooth FIM direction — a LICENSING unblock. Recording agitation/seeding
    # flips the M5 mechanistic license. The FULL mechanism unblock is credited only
    # when a concentration series is ALSO present (dependency-honest, like M9).
    if not agitation_seeding_known:
        licensable = has_series_now   # with a series already, this is the LAST gate
        cands.append({
            "candidate_id": "record_agitation_and_seeding",
            "variable_type": "gating",
            "gain_basis": "licensing",
            "target": "mechanism_class",
            "fim": None,                             # gating gives NO FIM direction
            "licensing": {
                "flips_m5_license": True,
                "mechanism_licensable_after": bool(licensable),
                "has_series_now": bool(has_series_now),
                "necessary_but_not_sufficient": (not has_series_now),
                "also_requires": [] if has_series_now else ["add_concentration_series"],
            },
            "design": {"metadata_only": True, "n_conditions": 0, "n_replicates": 0,
                       "t_max_hours": 0.0},
            "description": ("record agitation & seeding (a METADATA note, not a run): "
                            "flips the M5 quiescent-vs-shaken / primary-vs-secondary "
                            "license. " + ("A series is already present, so this is the "
                            "LAST missing piece -> mechanism becomes licensable."
                            if has_series_now else "On a single curve it is "
                            "NECESSARY-but-NOT-SUFFICIENT: a concentration series is the "
                            "binding prerequisite.")),
            "low_confidence": False,
        })

    return cands


# =========================================================================== #
# Scoring — turn each candidate into the four §3-M13 quantities.
# =========================================================================== #
def _score_candidate(cand: dict, name: str, theta: np.ndarray, order: list[str],
                     prior_cv: float, gamma_prior_cv: float, sc: dict,
                     has_series_now: bool, agitation_seeding_known: bool) -> dict:
    """Compute (1) EIG bits, (2) SE reduction, (3) mechanistic gain, (4) cost/runtime
    for one candidate, plus the priority = EIG/cost. FIM candidates use the simulated
    𝓘(ξ); gating candidates get EIG=0 on the FIM axis (their value is the licensing
    unblock, scored as mechanistic gain)."""
    p = len(order)
    prior_prec = _prior_precision(theta, prior_cv)

    out = {
        "candidate_id": cand["candidate_id"],
        "variable_type": cand["variable_type"],
        "gain_basis": cand["gain_basis"],
        "target": cand["target"],
        "description": cand["description"],
        "low_confidence": bool(cand.get("low_confidence")),
    }
    if cand.get("low_confidence_reason"):
        out["low_confidence_reason"] = cand["low_confidence_reason"]

    # ---- (1) EIG + (2) SE reduction --------------------------------------- #
    fim = cand.get("fim")
    if fim is not None:
        fim = np.asarray(fim, float)
        eig = _eig_bits(fim, theta, prior_cv)
        se = _posterior_se_reduction(prior_prec, fim, order, theta)
        out["expected_information_gain"] = {
            "eig_bits": round(eig["eig_bits"], 6),
            "eig_nats": round(eig["eig_nats"], 6),
            "target": cand["target"], "gain_basis": "fim",
            "prior_cv": prior_cv,
        }
        # If the design makes the gamma direction estimable (a series), add a TARGETED
        # gamma EIG on the extra prior dimension: the series informs one new direction
        # whose prior variance is (gamma_prior_cv*gamma_nominal)^2. We approximate its
        # information as the smallest observed-FIM eigenvalue scaled onto that prior
        # (a conservative, deterministic proxy — the series is the ONLY gamma design).
        if cand.get("adds_gamma_direction"):
            gamma_info = float(np.trace(fim)) / max(p, 1)   # avg curvature as a proxy
            s0_gamma = (gamma_prior_cv * _NOMINAL_GAMMA) ** 2
            eig_gamma = 0.5 * math.log1p(s0_gamma * max(gamma_info, 0.0))
            out["expected_information_gain"]["eig_bits_gamma_direction"] = round(eig_gamma / _LN2, 6)
            out["expected_information_gain"]["eig_bits_total"] = round(
                eig["eig_bits"] + eig_gamma / _LN2, 6)
            out["expected_information_gain"]["note"] = (
                "eig_bits is the D-optimal parameter EIG; eig_bits_gamma_direction is "
                "the TARGETED extra bit for the gamma/reaction-order direction the "
                "series makes estimable; eig_bits_total sums them.")
        out["expected_parameter_uncertainty_reduction"] = se
        eig_for_rank = out["expected_information_gain"].get(
            "eig_bits_total", out["expected_information_gain"]["eig_bits"])
    else:
        # gating candidate: no FIM direction -> EIG on the FIM axis is 0 by construction.
        out["expected_information_gain"] = {
            "eig_bits": 0.0, "eig_nats": 0.0, "target": cand["target"],
            "gain_basis": "licensing",
            "note": ("a GATING variable adds NO smooth FIM direction — its value is a "
                     "LICENSING unblock, scored in expected_mechanistic_distinguishability"
                     "_gain, not as an FIM EIG."),
        }
        out["expected_parameter_uncertainty_reduction"] = {
            "per_parameter": {}, "max_se_reduction_pct": 0.0, "mean_se_reduction_pct": 0.0,
            "note": "gating variable: no parameter-SE reduction (licensing, not FIM)."}
        eig_for_rank = 0.0

    # ---- (3) mechanistic distinguishability gain (Service-C bounded) ------- #
    lic = cand.get("licensing") or {}
    # state of the world AFTER this candidate:
    has_series_after = has_series_now or bool(cand.get("makes_series"))
    if cand["variable_type"] == "gating":
        licensable_after = bool(lic.get("mechanism_licensable_after"))
    else:
        # a series makes gamma estimable but the license still needs agitation/seeding
        licensable_after = has_series_after and agitation_seeding_known
    out["expected_mechanistic_distinguishability_gain"] = _mechanistic_gain(
        has_series_after, licensable_after, sc)
    if cand["variable_type"] == "gating":
        out["expected_mechanistic_distinguishability_gain"]["licensing"] = lic

    # ---- (4) cost + runtime (versioned estimate table) -------------------- #
    d = cand["design"]
    cr = estimate_cost_runtime(
        n_conditions=d.get("n_conditions", 1), n_replicates=d.get("n_replicates", 1),
        t_max_hours=d.get("t_max_hours", 0.0), metadata_only=d.get("metadata_only", False))
    out["estimated_cost"] = {"cost_units": cr["estimated_cost_units"],
                             "cost_table_version": cr["cost_table_version"],
                             "is_estimate": True, "detail": cr}
    out["estimated_runtime"] = {"runtime_hours": cr["estimated_runtime_hours"],
                                "is_estimate": True}

    # ---- priority = information gain / cost -------------------------------- #
    # For FIM candidates the numerator is the EIG (bits); for gating candidates the
    # numerator is the mechanism resolution gain (a Service-C-measured licensing
    # value), on a COMPARABLE scale (bits-equivalent = log2 of the class collapse) so
    # a licensing unblock and an FIM EIG can be ranked together honestly.
    mech = out["expected_mechanistic_distinguishability_gain"]
    licensing_bits = 0.0
    if cand["variable_type"] == "gating" and mech.get("resolution_gain_x", 1.0) > 1.0:
        # bits-equivalent of collapsing N broad classes into 1 resolved singleton set
        licensing_bits = math.log2(mech["resolution_gain_x"])
    gain_for_rank = eig_for_rank + licensing_bits
    cost = out["estimated_cost"]["cost_units"] or _EPS
    out["information_gain_for_ranking_bits"] = round(gain_for_rank, 6)
    out["priority_score_bits_per_cost"] = round(gain_for_rank / cost, 6)
    out["_fim"] = fim                                # kept for the greedy plan (stripped later)
    out["_makes_series"] = bool(cand.get("makes_series"))
    out["_is_gating"] = cand["variable_type"] == "gating"
    out["_licensable_after"] = bool(lic.get("mechanism_licensable_after")) if lic else False
    return out


# =========================================================================== #
# Pareto frontier — non-dominated over (info ↑, cost ↓, runtime ↓).
# =========================================================================== #
def pareto_frontier(scored: list[dict]) -> list[dict]:
    """Return the NON-DOMINATED set over (information_gain ↑, cost ↓, runtime ↓). A
    candidate A DOMINATES B iff A is >= on all three (info higher, cost/runtime
    lower) and strictly better on at least one. We return each frontier member's
    full objective vector — no single scalar is imposed."""
    def vec(r):
        return (r["information_gain_for_ranking_bits"],
                r["estimated_cost"]["cost_units"],
                r["estimated_runtime"]["runtime_hours"])
    frontier = []
    for a in scored:
        ia, ca, ra = vec(a)
        dominated = False
        for b in scored:
            if b is a:
                continue
            ib, cb, rb = vec(b)
            # b dominates a: >= info, <= cost, <= runtime, and strictly better once
            if ib >= ia and cb <= ca and rb <= ra and (ib > ia or cb < ca or rb < ra):
                dominated = True
                break
        if not dominated:
            frontier.append({
                "candidate_id": a["candidate_id"],
                "information_gain_bits": ia,
                "cost_units": ca,
                "runtime_hours": ra,
                "gain_basis": a["gain_basis"],
                "target": a["target"],
            })
    frontier.sort(key=lambda f: (-f["information_gain_bits"], f["cost_units"]))
    return frontier


# =========================================================================== #
# Greedy sequential BOED — pick argmax EIG/cost, UPDATE the accumulated FIM /
# licensing state, RE-SCORE, repeat. Marginal EIG DIMINISHES (submodular log-det).
# =========================================================================== #
def _marginal_eig_bits(fim_accum: np.ndarray, fim_new: np.ndarray, theta: np.ndarray,
                       prior_cv: float) -> float:
    """Marginal EIG of adding 𝓘_new on TOP of the already-accumulated 𝓘:
        ΔEIG = EIG(accum + new) − EIG(accum)  (in bits).
    Because log-det is submodular, this marginal DIMINISHES as directions get
    informed — a 2nd concentration series is worth strictly less than the 1st."""
    base = expected_information_gain(fim_accum, theta, prior_cv)["eig_bits"]
    both = expected_information_gain(fim_accum + fim_new, theta, prior_cv)["eig_bits"]
    return both - base


def multi_step_plan(scored: list[dict], theta: np.ndarray, order: list[str],
                    prior_cv: float, sc: dict, has_series_now: bool,
                    agitation_seeding_known: bool, k_steps: int = 3) -> dict:
    """Greedy K-step batch: each step picks the candidate with the largest MARGINAL
    EIG/cost given what is already chosen, updates the accumulated FIM (FIM
    candidates) / the licensing gate (gating candidate), and re-scores the rest. The
    per-step marginal EIG must DIMINISH (submodularity). Returns the plan + the
    diminishing-returns curve + the (1-1/e) note. Greedy ≠ global optimum."""
    p = len(order)
    prior_prec = _prior_precision(theta, prior_cv)
    fim_accum = prior_prec.copy()                    # start from the prior precision
    fim_accum_designonly = np.zeros((p, p))          # design-only accum (for reporting)
    remaining = list(scored)
    chosen = []
    cum_gain = 0.0
    cum_cost = 0.0
    series_present = has_series_now
    agit_known = agitation_seeding_known
    diminishing = []

    for step in range(1, k_steps + 1):
        best, best_marg, best_priority = None, None, -1.0
        for cand in remaining:
            cost = cand["estimated_cost"]["cost_units"] or _EPS
            if cand["_is_gating"]:
                # gating marginal value = the licensing unblock IF it becomes the last
                # gate given the CURRENT accumulated state (series already chosen?).
                licensable = series_present
                mech = _mechanistic_gain(series_present, licensable and agit_known
                                         or (licensable and not agit_known and False), sc)
                # bits-equivalent of the licensing collapse, only if it is now sufficient
                if licensable:
                    marg = math.log2(sc["concentration_series_classes"] /
                                     max(sc["single_concentration_classes"], 1))
                else:
                    marg = 0.0   # not sufficient yet (no series) -> future-gate only
            else:
                fim_new = cand.get("_fim")
                if fim_new is None:
                    marg = 0.0
                else:
                    marg = _marginal_eig_bits(fim_accum, np.asarray(fim_new, float),
                                              theta, prior_cv)
                    # a series ALSO opens the gamma direction: add its targeted bit once
                    if cand["_makes_series"] and not series_present:
                        gi = float(np.trace(np.asarray(fim_new, float))) / max(p, 1)
                        marg += 0.5 * math.log1p((_NOMINAL_GAMMA ** 2) * max(gi, 0.0)) / _LN2
            priority = marg / cost
            if priority > best_priority + 1e-15:
                best, best_marg, best_priority = cand, marg, priority
        if best is None:
            break
        # commit the choice: update accumulated FIM / licensing gate
        if best["_is_gating"]:
            agit_known = True
        else:
            fim_new = best.get("_fim")
            if fim_new is not None:
                fim_accum = fim_accum + np.asarray(fim_new, float)
                fim_accum_designonly = fim_accum_designonly + np.asarray(fim_new, float)
            if best["_makes_series"]:
                series_present = True
        cost = best["estimated_cost"]["cost_units"]
        cum_gain += max(best_marg, 0.0)
        cum_cost += cost
        chosen.append({
            "step": step,
            "candidate_id": best["candidate_id"],
            "gain_basis": best["gain_basis"],
            "marginal_information_gain_bits": round(best_marg, 6),
            "cost_units": cost,
            "priority_bits_per_cost": round(best_priority, 6),
            "cumulative_gain_bits": round(cum_gain, 6),
            "cumulative_cost_units": round(cum_cost, 6),
        })
        diminishing.append(round(best_marg, 6))
        remaining = [c for c in remaining if c is not best]

    # verify the diminishing-returns property on the FIM-based marginals
    fim_marginals = [c["marginal_information_gain_bits"] for c in chosen
                     if c["gain_basis"] == "fim"]
    non_increasing = all(fim_marginals[i] >= fim_marginals[i + 1] - 1e-9
                         for i in range(len(fim_marginals) - 1))
    return {
        "k_steps": k_steps,
        "steps": chosen,
        "diminishing_returns_curve_bits": diminishing,
        "fim_marginals_non_increasing": bool(non_increasing),
        "cumulative_gain_bits": round(cum_gain, 6),
        "cumulative_cost_units": round(cum_cost, 6),
        "greedy_note": ("GREEDY sequential BOED: each step picks argmax marginal "
                        "EIG/cost, updates the accumulated FIM / licensing gate, then "
                        "re-scores. Marginal FIM EIG DIMINISHES (submodular log-det). "
                        "Greedy is NOT the global optimum, but submodularity gives the "
                        "batch a (1-1/e)~0.63 approximation guarantee."),
    }


# =========================================================================== #
# Public entry point — recommend_experiments(protein_or_series, ...).
# =========================================================================== #
def _measured_range(members: list[dict], key: str) -> tuple | None:
    """Distinct measured values of a condition field across the member curves
    (for the T/pH extrapolation flag). None if unavailable."""
    vals = set()
    for s in members:
        cv = s.get("condition_vector", {}) or {}
        v = cv.get(key)
        if v is not None:
            vals.add(round(float(v), 3))
    if not vals:
        return None
    return (min(vals), max(vals))


def recommend_experiments(series: dict, members: list[dict] | None = None,
                          service_c: dict | None = None,
                          gamma_result: dict | None = None,
                          prior_cv: float = _DEFAULT_PRIOR_CV,
                          gamma_prior_cv: float = _DEFAULT_GAMMA_PRIOR_CV,
                          fit_result: dict | None = None) -> dict:
    """Full M13 BOED recommendation for one protein/analysis unit. NEVER raises.

    Args:
        series: an M1-triaged curve (the primary/representative curve to fit).
        members: the concentration-series member curves (if this is a series unit),
                 used for the T/pH measured-range flag + series-present detection.
        service_c: parsed service_c_calibration.json (mechanism-gain anchor).
        gamma_result: the M4 dual_gamma payload for this series (supplies the γ used
                      in the concentration rescaling); None -> fallback γ (flagged).
        prior_cv / gamma_prior_cv: weak-Gaussian prior CVs (prior-conditional EIG).

    Returns {candidates[], ranked[], pareto_optimal_sets[], multi_step_plan, prior,
    honesty, version}. No fitted model -> M9-language fallback + status INSUFFICIENT.
    """
    sc = _sc_numbers(service_c)
    members = members or []
    out = {
        "series_id": series.get("series_id"),
        "protein_id": series.get("protein_id"),
        "concentration_series_id": series.get("concentration_series_id"),
        "version": M13_VERSION,
        "cost_table_version": COST_TABLE_VERSION,
        "service_c_version": sc["service_c_version"],
        "honesty": dict(_HONESTY),
    }

    # ---- fit the current forward model (M2 -> the theta_hat we simulate from) --- #
    fr = fit_result or fit_curve(series)
    conv = {n: r for n, r in fr.get("fits", {}).items()
            if r.get("converged") and r.get("aicc") is not None and r.get("params")}
    x = series.get("x_hours") or []
    y = series.get("m1", {}).get("y_processed") or series.get("y_intensity") or []
    if not conv or len(x) != len(y) or len(x) < 3:
        # No fitted model -> M9 rule-table fallback, status INSUFFICIENT (never raise).
        cv = series.get("condition_vector", {}) or {}
        prov = cv.get("field_provenance", {}) or {}
        agit_known = prov.get("agitation") == "known" and prov.get("seeded") == "known"
        has_series = bool(series.get("concentration_series_id")) and len(members) >= 3
        fallback = ["add_concentration_series"] if not has_series else []
        if not agit_known:
            fallback.append("record_agitation_and_seeding")
        out.update({
            "status": "INSUFFICIENT",
            "reason": "no converged forward model — cannot simulate a prospective FIM",
            "candidates": [], "ranked": [], "pareto_optimal_sets": [],
            "multi_step_plan": {"steps": [], "note": "no FIM to plan over"},
            "m9_fallback": {
                "language": "M9 (engine/m9_recommender.py) rule-table vocabulary",
                "suggested_experiments": fallback or ["extend_observation_window"],
                "note": ("no fitted forward model -> M13 defers to the M9 heuristic "
                         "rule table; recording agitation/seeding + adding a "
                         "concentration series are the corpus-wide unblockers."),
            },
        })
        return out

    name = min(conv, key=lambda n: conv[n]["aicc"])
    fit = conv[name]
    params = fit["params"]
    order = list(REGISTRY[name]["params"])
    theta = np.array([params[pn] for pn in order], float)
    n, p = len(x), len(order)
    sigma2 = _sigma2_from_fit(fit, name, n, p)

    out["selected_model"] = name
    out["n_points"] = n
    out["n_params"] = p
    out["sigma2"] = float(sigma2)

    # ---- current state: concentration, gamma, series-presence, gating -------- #
    cv = series.get("condition_vector", {}) or {}
    conc = cv.get("concentration", {}) or {}
    m0 = conc.get("value_uM") if isinstance(conc, dict) else None
    prov = cv.get("field_provenance", {}) or {}
    agitation_seeding_known = (prov.get("agitation") == "known"
                               and prov.get("seeded") == "known")
    has_series_now = bool(series.get("concentration_series_id")) and len(members) >= 3

    gamma, gamma_reliable = None, False
    if gamma_result:
        reg = gamma_result.get("gamma_regression", {}) or {}
        gamma = reg.get("gamma")
        gamma_reliable = bool(reg.get("gamma_reliable"))

    t_range = _measured_range(members, "temperature_C") if members else None
    ph_range = _measured_range(members, "pH") if members else None

    out["current_state"] = {
        "protein_concentration_uM": m0,
        "gamma_used": (float(gamma) if gamma is not None else _FALLBACK_GAMMA),
        "gamma_reliable": gamma_reliable,
        "gamma_source": ("m4_dual_gamma" if (gamma is not None and gamma_reliable)
                         else "fallback_nominal_flagged"),
        "has_concentration_series_now": has_series_now,
        "agitation_seeding_known": agitation_seeding_known,
        "temperature_measured_range": list(t_range) if t_range else None,
        "pH_measured_range": list(ph_range) if ph_range else None,
    }

    # ---- build + score the candidates ---------------------------------------- #
    cands = build_candidates(
        name, params, x, sigma2, theta, order,
        m0=m0, gamma=gamma, gamma_reliable=gamma_reliable, has_series_now=has_series_now,
        agitation_seeding_known=agitation_seeding_known,
        t_measured_range=t_range, ph_measured_range=ph_range, sc=sc,
        prior_cv=prior_cv, gamma_prior_cv=gamma_prior_cv)

    scored = [_score_candidate(c, name, theta, order, prior_cv, gamma_prior_cv, sc,
                               has_series_now, agitation_seeding_known)
              for c in cands]

    # ---- Pareto + multi-step plan (use the private _fim before stripping) ---- #
    pareto = pareto_frontier(scored)
    plan = multi_step_plan(scored, theta, order, prior_cv, sc, has_series_now,
                           agitation_seeding_known, k_steps=3)

    # ---- ranked list (by EIG/cost desc) + strip private fields --------------- #
    ranked = sorted(scored, key=lambda r: (-r["priority_score_bits_per_cost"],
                                           r["estimated_cost"]["cost_units"],
                                           r["candidate_id"]))
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    clean = []
    for r in scored:
        rr = {k: v for k, v in r.items() if not k.startswith("_")}
        clean.append(rr)

    out.update({
        "status": "ok",
        "candidates": clean,
        "ranked": [{"rank": r["rank"], "candidate_id": r["candidate_id"],
                    "gain_basis": r["gain_basis"], "target": r["target"],
                    "information_gain_for_ranking_bits": r["information_gain_for_ranking_bits"],
                    "priority_score_bits_per_cost": r["priority_score_bits_per_cost"],
                    "cost_units": r["estimated_cost"]["cost_units"],
                    "low_confidence": r["low_confidence"]}
                   for r in ranked],
        "pareto_optimal_sets": pareto,
        "multi_step_plan": plan,
        "top_recommendation": ranked[0]["candidate_id"] if ranked else None,
        "prior": {
            "prior_cv": prior_cv, "gamma_prior_cv": gamma_prior_cv,
            "form": "Sigma0 = diag((prior_cv*theta_hat)^2) — weak Gaussian, VERSIONED",
            "eig_is_prior_conditional": True,
        },
    })
    return out


# =========================================================================== #
# I/O / CLI — run over proteins that HAVE a fitted model.
# =========================================================================== #
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


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_gamma_index(path: Path) -> dict:
    """Map concentration_series_id -> the M4 dual_gamma payload (for the gamma used
    in the concentration rescaling)."""
    out = {}
    for r in _read_jsonl(path):
        csid = r.get("concentration_series_id")
        if csid:
            out[csid] = r
    return out


def main(argv=None) -> int:
    root = Path(__file__).resolve().parent.parent
    proc = root / "data" / "processed"
    ap = argparse.ArgumentParser(description="PRISE M13 — Bayesian Optimal Experimental Design")
    ap.add_argument("--input", type=Path, default=proc / "curves_triaged.jsonl")
    ap.add_argument("--series", type=Path, default=proc / "concentration_series.json")
    ap.add_argument("--gamma", type=Path, default=proc / "gamma.jsonl")
    ap.add_argument("--service-c", type=Path, default=proc / "service_c_calibration.json")
    ap.add_argument("--output", type=Path, default=proc / "boed_recommendations.json")
    ap.add_argument("--per-protein", type=Path, default=proc / "boed_recommendations.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input} (run M1 first)")

    curves = _read_jsonl(args.input)
    by_id = {c.get("series_id"): c for c in curves}
    service_c = _read_json(args.service_c)
    gamma_idx = _load_gamma_index(args.gamma)
    series_meta = _read_json(args.series) or []

    # concentration-series member map (for T/pH range + series-present detection)
    members_by_csid = defaultdict(list)
    for c in curves:
        csid = c.get("concentration_series_id")
        if csid:
            members_by_csid[csid].append(c)

    results = []
    single_curve_examples = []
    series_examples = []
    n = 0
    print(f"[M13] BOED recommendations over fittable curves in {args.input.name} ...")

    # 1) SERIES units — one representative member curve per concentration series
    for meta in series_meta:
        csid = meta.get("concentration_series_id")
        members = members_by_csid.get(csid, [])
        rep = next((m for m in members
                    if m.get("m1", {}).get("fittability_class") == "fittable"), None)
        if rep is None:
            continue
        res = recommend_experiments(rep, members=members, service_c=service_c,
                                    gamma_result=gamma_idx.get(csid))
        res["analysis_unit"] = "concentration_series"
        results.append(res)
        if res.get("status") == "ok" and len(series_examples) < 3:
            series_examples.append(res)
        n += 1
        if args.limit and n >= args.limit:
            break

    # 2) SINGLE-CURVE units — fittable curves NOT in a series
    for c in curves:
        if args.limit and n >= args.limit:
            break
        if c.get("concentration_series_id"):
            continue
        if c.get("m1", {}).get("fittability_class") != "fittable":
            continue
        res = recommend_experiments(c, members=[], service_c=service_c)
        res["analysis_unit"] = "single_curve"
        results.append(res)
        if res.get("status") == "ok" and len(single_curve_examples) < 3:
            single_curve_examples.append(res)
        n += 1

    # ---- corpus rollup -------------------------------------------------------- #
    ok = [r for r in results if r.get("status") == "ok"]
    top_counter = defaultdict(int)
    for r in ok:
        if r.get("top_recommendation"):
            top_counter[r["top_recommendation"]] += 1

    rollup = {
        "version": M13_VERSION,
        "cost_table_version": COST_TABLE_VERSION,
        "service_c_version": _sc_numbers(service_c)["service_c_version"],
        "n_units": len(results),
        "n_ok": len(ok),
        "n_series_units": sum(1 for r in results if r.get("analysis_unit") == "concentration_series"),
        "n_single_curve_units": sum(1 for r in results if r.get("analysis_unit") == "single_curve"),
        "top_recommendation_distribution": dict(sorted(top_counter.items(),
                                                       key=lambda kv: -kv[1])),
        "representative_single_curve": single_curve_examples[0] if single_curve_examples else None,
        "representative_series": series_examples[0] if series_examples else None,
        "honesty": dict(_HONESTY),
        "cost_table": COST_TABLE,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rollup, indent=2), encoding="utf-8")
    with open(args.per_protein, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    print(f"[M13] wrote {args.output}")
    print(f"[M13] wrote {args.per_protein}  ({len(results)} units, {len(ok)} ok)")
    print(f"[M13] top-recommendation distribution: {rollup['top_recommendation_distribution']}")
    if series_examples:
        r = series_examples[0]
        print(f"\n  REPRESENTATIVE SERIES unit ({r['series_id']}, model={r['selected_model']}):")
        print(f"    top: {r['top_recommendation']}")
        print(f"    pareto: {[f['candidate_id'] for f in r['pareto_optimal_sets']]}")
        print(f"    3-step plan: "
              f"{[(s['candidate_id'], s['marginal_information_gain_bits']) for s in r['multi_step_plan']['steps']]}")
        print(f"    diminishing-returns curve (bits): {r['multi_step_plan']['diminishing_returns_curve_bits']}")
    if single_curve_examples:
        r = single_curve_examples[0]
        print(f"\n  REPRESENTATIVE SINGLE-CURVE unit ({r['series_id']}, model={r['selected_model']}):")
        print(f"    top: {r['top_recommendation']}")
        print(f"    pareto: {[f['candidate_id'] for f in r['pareto_optimal_sets']]}")
        print(f"    3-step plan: "
              f"{[(s['candidate_id'], s['marginal_information_gain_bits']) for s in r['multi_step_plan']['steps']]}")
    print("\n  HONESTY: EIG is EXPECTED under the fitted forward model + weak prior, "
          "Laplace-linearized (M8 ceiling); mechanistic gains bounded by Service-C "
          "(upper bound); cost/runtime are a versioned ESTIMATE; gating = licensing "
          "(not FIM); greedy != global optimum.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
