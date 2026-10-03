"""
PRISE — Module M12: Information Content & Experimental-Design Geometry
=====================================================================

PRISE_DESIGN.md §3-M12 (the per-experiment information analysis). M12 UNIFIES the
information geometry the pipeline already touches into ONE coherent per-curve
treatment — it is *consistent with*, not a third opinion of, the existing modules:

  * the Jacobian / sensitivity object is the SAME finite-difference sensitivity M3
    (`m3_select.fim_identifiability`) and global_fit build; M12's condition number
    on the column-normalised (correlation) FIM MATCHES M3's collinearity condition
    number on the same curve (cross-checked in the tests).
  * the observed FIM 𝓘 = SᵀΣ⁻¹S with Σ=σ²I mirrors `global_fit._identifiability`
    (same J^Tσ^-2 J observed-FIM convention, eigen-decomposition, stiff/sloppy
    labelling of identifiable combinations).
  * the phase segmentation (lag / growth / plateau) reads M4's feature boundaries
    (`m4_features.feature_geometry`: `lag_time`, `t50`, `has_interior_inflection`).
  * the model + θ̂ + SSE come from the M2-SELECTED (best-AICc) fit
    (`m2_fit.fit_curve`), exactly as M3/M4 pick the analysed model.

Given a fitted curve, M12 computes: the sensitivity Jacobian S; the observed
Fisher Information Matrix 𝓘; its eigen-spectrum (effective rank — hard + entropy
erank, condition number, spectral entropy); the estimator differential entropy;
per-parameter observability (CRLB SE / CV / observable flag) and stiff-vs-sloppy
identifiable combinations; the per-point information-density profile D(t) and its
fraction by lag/growth/plateau phase; measurement redundancy (n_eff vs n); the
expected-variance-reduction optimal-design scan (Sherman–Morrison) for the best
next time-point; a prior-conditional Bayesian information gain (EIG); a bounded
information-richness score; and an M9-language recommended-measurement action.

HONESTY (attached to every result, non-negotiable): the FIM here is LOCAL,
LINEARIZED (a Laplace/Gaussian curvature approximation of the likelihood at θ̂),
CONDITIONAL on the M2-selected model AND the σ²I noise model, and POST-SELECTION
(model-averaging over M3's selection posterior is a documented deferred
extension). The `information_richness_score` is a bounded COLLAPSE of the
spectrum — it is always returned WITH the spectrum it summarises, never instead
of it. EIG is prior-conditional (a stated weak Gaussian prior). A SLOPPY model
has erank ≪ p and M12 SAYS SO.

Everything is pure / deterministic / seeded / JSON-serialisable; nothing raises —
a rank-deficient / degenerate / flat / no-fit curve returns a FLAGGED minimal
result (Moore–Penrose pseudo-inverse + pseudo-logdet, erank≈0/1, richness≈0),
never an exception.

Usage:
    python engine/m12_information.py                       # -> information_content.jsonl + .json
    python engine/m12_information.py --limit 200
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from m2_fit import fit_curve
from m4_features import feature_geometry
from models import REGISTRY

# Module identity — provenance tag (mirrors the M2..M9 convention); a bump makes
# downstream information-content artifacts non-comparable.
M12_VERSION = "m12-information-1.0"

_EPS = 1e-12
# Effective-rank tolerance: an eigenvalue below tol·λ_max is round-off, not signal
# (matches global_fit's ~1e-3 stiff cut being looser; here we use a strict 1e-10
# floor for the HARD rank and report the entropy erank alongside).
_RANK_TOL = 1e-10
# Observability: a parameter with coefficient-of-variation (CRLB SE / |θ̂|) below
# this is "observable" (constrained to better than its own magnitude). 1.0 is the
# stated, conservative default (CV=1 ⇒ the 1-σ interval spans ±100% of θ̂).
_OBSERVABLE_CV = 1.0
# Digitization-variance floor: CPAD curves are graph-digitized, so the noise floor
# is at least the pixel/quantisation scale. σ² = max(SSE/(n−p), (frac·range)²).
# Default small-but-present so it only bites when the fit is near-perfect.
_DEFAULT_DIGITIZATION_FRAC = 0.0
# Weak Gaussian prior for the EIG: Σ₀ = diag((prior_cv·θ̂)²). prior_cv=1.0 is a
# VERY weak prior (1-σ prior width = ±100% of θ̂) — the EIG is prior-conditional.
_DEFAULT_PRIOR_CV = 1.0


# --------------------------------------------------------------------------- #
#  Sensitivity Jacobian (central finite differences — the SAME object M3 uses)
# --------------------------------------------------------------------------- #
def sensitivity_jacobian(name: str, params: dict, x) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """S[i,j] = ∂f(t_i;θ)/∂θ_j by CENTRAL finite difference, plus the relative
    sensitivities (θ_j/f)·∂f/∂θ_j and the parameter order.

    Relative step ~1e-6·|θ_j| with an absolute floor (1e-6) so θ_j=0 is handled
    (this is the same central-difference sensitivity M3/global_fit build; M12 just
    uses the raw S, not the column-normalised correlation, for the observed FIM)."""
    order = REGISTRY[name]["params"]
    func = REGISTRY[name]["func"]
    theta = np.array([params[p] for p in order], float)
    xa = np.asarray(x, float)
    n, p = len(xa), len(theta)
    f0 = np.asarray(func(xa, *theta), float)

    S = np.zeros((n, p))
    for j in range(p):
        h = 1e-6 * max(abs(theta[j]), 1.0)          # relative step, absolute floor
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        col = (np.asarray(func(xa, *tp), float) - np.asarray(func(xa, *tm), float)) / (2.0 * h)
        S[:, j] = np.where(np.isfinite(col), col, 0.0)

    # relative sensitivity profile (θ_j/f)·∂f/∂θ_j — dimensionless per-parameter
    # time profile; f→0 is guarded so a baseline-crossing curve does not blow up.
    fsafe = np.where(np.abs(f0) > _EPS, f0, np.sign(f0) * _EPS + _EPS)
    S_rel = S * (theta[None, :] / fsafe[:, None])
    S_rel = np.where(np.isfinite(S_rel), S_rel, 0.0)
    return S, S_rel, list(order)


# --------------------------------------------------------------------------- #
#  Observed Fisher Information Matrix  𝓘 = SᵀΣ⁻¹S,  Σ = σ²I
# --------------------------------------------------------------------------- #
def fisher_information(S: np.ndarray, sse: float, n: int, p: int,
                       digitization_frac: float, y_range: float) -> tuple[np.ndarray, float, dict]:
    """Observed, model-conditional, linearized (Laplace) FIM 𝓘 = SᵀΣ⁻¹S with
    Σ=σ²I. σ² = SSE/(n−p) (unbiased) with an OPTIONAL digitization-variance floor
    σ² = max(SSE/(n−p), (digitization_frac·range)²). Mirrors global_fit's observed
    J^Tσ^-2 J. Returns (𝓘, σ², noise_meta)."""
    dof = max(n - p, 1)
    sigma2_raw = max(float(sse) / dof, _EPS)
    sigma2_floor = (float(digitization_frac) * float(y_range)) ** 2 if digitization_frac > 0 else 0.0
    sigma2 = max(sigma2_raw, sigma2_floor)
    floored = sigma2_floor > sigma2_raw
    fim = (S.T @ S) / sigma2                         # SᵀΣ⁻¹S with Σ=σ²I
    fim = 0.5 * (fim + fim.T)                        # symmetrise (round-off)
    meta = {
        "sigma2": float(sigma2),
        "sigma2_unbiased_sse": float(sigma2_raw),
        "sigma2_digitization_floor": float(sigma2_floor),
        "digitization_frac": float(digitization_frac),
        "noise_dof": int(dof),
        "digitization_floor_active": bool(floored),
        "noise_model": "Gaussian iid Sigma=sigma^2 I (homoscedastic Laplace/observed FIM)",
    }
    return fim, sigma2, meta


# --------------------------------------------------------------------------- #
#  Eigen-spectrum, effective rank, entropies
# --------------------------------------------------------------------------- #
def _pseudo_logdet(evals: np.ndarray, tol: float) -> tuple[float, int]:
    """log-det over the OBSERVABLE subspace (product of eigenvalues above tol).
    Returns (pseudo_logdet, n_observable). Used when the FIM is rank-deficient so
    the differential entropy is finite (flagged)."""
    keep = evals[evals > tol]
    if keep.size == 0:
        return float("-inf"), 0
    return float(np.sum(np.log(keep))), int(keep.size)


def eigen_spectrum(fim: np.ndarray, order: list[str]) -> dict:
    """Symmetric eigen-decomposition (numpy.linalg.eigh) of the observed FIM.
    Reports λ₁≥…≥λ_p (tiny negatives clipped to 0), condition number, hard +
    entropy effective rank, spectral entropy, and stiff/sloppy eigenvectors with
    dominant parameter loadings (the identifiable-combination labelling that
    mirrors global_fit._identifiability)."""
    p = fim.shape[0]
    try:
        evals, evecs = np.linalg.eigh(fim)
    except np.linalg.LinAlgError:
        return {"computable": False, "reason": "eigh_failed", "n_params": p}
    # eigh returns ascending; clip round-off negatives, then sort DESCENDING
    evals = np.clip(evals, 0.0, None)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    lam_max = float(evals[0]) if evals.size else 0.0
    if lam_max <= 0:
        return {"computable": False, "reason": "degenerate_fim_zero_curvature",
                "n_params": p, "eigenvalues": [0.0] * p}

    lam_min = float(evals[-1])
    cond = (lam_max / lam_min) if lam_min > _EPS else float("inf")

    # hard effective rank: count eigenvalues above tol·λ_max
    hard_rank = int(np.sum(evals > _RANK_TOL * lam_max))
    # entropy-based effective rank: erank = exp(-Σ p_k ln p_k), p_k = λ_k/Σλ
    ssum = float(np.sum(evals))
    pk = evals / ssum if ssum > _EPS else np.ones(p) / p
    pk_pos = pk[pk > _EPS]
    spectral_entropy = float(-np.sum(pk_pos * np.log(pk_pos)))
    erank = float(math.exp(spectral_entropy))

    # stiff vs sloppy identifiable combinations (dominant ± loadings, as global_fit)
    combos = []
    for k in range(p):
        ev = float(evals[k])
        vec = evecs[:, k]
        terms = sorted(zip(order, vec), key=lambda t: -abs(t[1]))
        desc = " + ".join(f"{w:+.2f}*{nm}" for nm, w in terms if abs(w) > 0.15)
        dominant = terms[0][0]
        combos.append({
            "eigenvalue": ev,
            "relative_eigenvalue": ev / lam_max,
            "stiff": bool(ev > _RANK_TOL * lam_max and ev / lam_max > 1e-3),
            "dominant_parameter": dominant,
            "loadings": {nm: round(float(w), 3) for nm, w in zip(order, vec)},
            "combination": desc or "(diffuse)",
        })

    return {
        "computable": True,
        "n_params": p,
        "eigenvalues": [float(e) for e in evals],
        "lambda_max": lam_max,
        "lambda_min": lam_min,
        "condition_number": (float(cond) if math.isfinite(cond) else None),
        "log10_condition_number": (round(math.log10(cond), 3)
                                   if math.isfinite(cond) and cond > 0 else None),
        "effective_rank_hard": hard_rank,
        "effective_rank_entropy": erank,
        "spectral_entropy": spectral_entropy,
        "identifiable_combinations": combos,
        "rank_deficient": bool(hard_rank < p),
        "stiff_directions": int(np.sum(evals / lam_max > 1e-3)),
        "sloppy_directions": int(np.sum(evals / lam_max <= 1e-3)),
    }


def differential_entropy(spectrum: dict, p: int) -> dict:
    """Estimator/posterior differential entropy of the Laplace-Gaussian at θ̂:
        H = (p/2) ln(2πe) − ½ ln det 𝓘.
    Uses a PSEUDO-logdet over the observable subspace when rank-deficient (flagged
    — the entropy is then over the constrained directions only, not the full p)."""
    evals = np.asarray(spectrum.get("eigenvalues", []), float)
    lam_max = spectrum.get("lambda_max", 0.0) or 0.0
    tol = _RANK_TOL * lam_max
    logdet_full = float(np.sum(np.log(evals[evals > 0]))) if np.any(evals > 0) else float("-inf")
    pseudo_logdet, n_obs = _pseudo_logdet(evals, tol)
    rank_deficient = n_obs < p
    # full H uses full logdet; pseudo H uses the observable subspace dimension
    const = 0.5 * math.log(2.0 * math.pi * math.e)
    H_full = (0.5 * p * math.log(2.0 * math.pi * math.e) - 0.5 * logdet_full
              if math.isfinite(logdet_full) else None)
    H_obs = (n_obs * const - 0.5 * pseudo_logdet
             if math.isfinite(pseudo_logdet) else None)
    return {
        "differential_entropy_full": (float(H_full) if H_full is not None else None),
        "differential_entropy_observable_subspace": (float(H_obs) if H_obs is not None else None),
        "n_observable_directions": int(n_obs),
        "half_logdet_fim": (0.5 * logdet_full if math.isfinite(logdet_full) else None),
        "half_pseudo_logdet_fim": (0.5 * pseudo_logdet if math.isfinite(pseudo_logdet) else None),
        "uses_pseudo_logdet": bool(rank_deficient),
        "note": ("H = (p/2)ln(2*pi*e) - 0.5*ln det I; when rank-deficient the "
                 "observable-subspace form uses a pseudo-logdet over eigenvalues "
                 ">tol (the unconstrained directions carry infinite differential "
                 "entropy and are excluded, flagged)"),
    }


# --------------------------------------------------------------------------- #
#  Parameter observability — CRLB via Moore–Penrose pinv (never blows up)
# --------------------------------------------------------------------------- #
def parameter_observability(fim: np.ndarray, theta: np.ndarray, order: list[str],
                            spectrum: dict) -> dict:
    """C = pinv(𝓘) (Moore–Penrose so it never blows up); per-parameter CRLB
    SE_j = sqrt(C_jj), CV_j = SE_j/|θ̂_j|, observable flag (CV < threshold). Also
    labels each eigenvector stiff/sloppy with its dominant parameter loadings."""
    C = np.linalg.pinv(fim)
    C = 0.5 * (C + C.T)
    diag = np.clip(np.diag(C), 0.0, None)
    se = np.sqrt(diag)
    table = {}
    for j, nm in enumerate(order):
        se_j = float(se[j])
        th = float(theta[j])
        cv = (se_j / abs(th)) if abs(th) > _EPS else None
        observable = bool(cv is not None and math.isfinite(cv) and cv < _OBSERVABLE_CV)
        table[nm] = {
            "theta_hat": th,
            "crlb_se": se_j,
            "cv": (float(cv) if cv is not None and math.isfinite(cv) else None),
            "observable": observable,
        }
    # stiff/sloppy summary from the spectrum's identifiable combinations
    stiff = [c for c in spectrum.get("identifiable_combinations", []) if c["stiff"]]
    sloppy = [c for c in spectrum.get("identifiable_combinations", []) if not c["stiff"]]
    return {
        "covariance_pinv_diag": [float(d) for d in diag],
        "per_parameter": table,
        "n_observable": int(sum(1 for v in table.values() if v["observable"])),
        "observable_cv_threshold": _OBSERVABLE_CV,
        "stiff_combinations": [{"dominant_parameter": c["dominant_parameter"],
                                "eigenvalue": c["eigenvalue"],
                                "combination": c["combination"]} for c in stiff],
        "sloppy_combinations": [{"dominant_parameter": c["dominant_parameter"],
                                 "eigenvalue": c["eigenvalue"],
                                 "combination": c["combination"]} for c in sloppy],
    }, C


# --------------------------------------------------------------------------- #
#  Information-density profile D(t) = s(t)ᵀ pinv(𝓘) s(t)  +  phase segmentation
# --------------------------------------------------------------------------- #
def _phase_boundaries(name: str, params: dict, x) -> dict:
    """Segment lag / growth / plateau from M4's feature geometry:
        lag    = [t0, lag_time]
        growth = [lag_time, 2*t50 - lag_time]   (symmetric growth window about t50)
        plateau= [growth_end, t_end]
    If the curve has no interior inflection (non-cooperative), report a single
    region — the phase segmentation degrades honestly rather than faking phases."""
    xa = np.asarray(x, float)
    t0, tend = float(xa.min()), float(xa.max())
    g = feature_geometry(name, params, list(xa))
    if g is None or not g.get("has_interior_inflection"):
        return {"cooperative": False, "t0": t0, "t_end": tend}
    lag_time = g.get("lag_time")
    t50 = g.get("t50")
    if lag_time is None or t50 is None or not (math.isfinite(lag_time) and math.isfinite(t50)):
        return {"cooperative": False, "t0": t0, "t_end": tend}
    lag_time = max(t0, min(float(lag_time), tend))
    growth_end = min(tend, 2.0 * float(t50) - lag_time)     # symmetric about t50
    growth_end = max(lag_time, growth_end)
    return {"cooperative": True, "t0": t0, "lag_end": lag_time,
            "growth_end": growth_end, "t_end": tend, "t50": float(t50)}


def information_density(S: np.ndarray, C: np.ndarray, x, name: str, params: dict) -> dict:
    """D(t_i) = s(t_i)ᵀ C s(t_i) with C = pinv(𝓘): each point's leverage /
    contribution to constraining the fit, aligned to the time grid. Segments the
    profile into lag/growth/plateau (M4 boundaries) and reports the FRACTION of
    total information in each phase (expected: growth/inflection >> plateau)."""
    xa = np.asarray(x, float)
    # per-point leverage; guard non-finite to 0 (never raises)
    D = np.einsum("ij,jk,ik->i", S, C, S)
    D = np.where(np.isfinite(D), np.clip(D, 0.0, None), 0.0)
    total = float(np.sum(D))

    ph = _phase_boundaries(name, params, xa)
    out = {
        "density": [float(d) for d in D],
        "total_information": total,
        "cooperative_phases": bool(ph.get("cooperative")),
    }
    if not ph.get("cooperative") or total <= _EPS:
        # non-cooperative or no signal -> single region (honest degradation)
        out["phase_fractions"] = {"single_region": 1.0 if total > _EPS else 0.0}
        out["phase_note"] = ("no interior inflection (non-cooperative) or zero "
                             "information — phase segmentation degrades to a single "
                             "region rather than faking lag/growth/plateau")
        out["phase_boundaries"] = {k: v for k, v in ph.items() if k != "cooperative"}
        return out

    lag_end, growth_end = ph["lag_end"], ph["growth_end"]
    lag_mask = xa <= lag_end
    growth_mask = (xa > lag_end) & (xa <= growth_end)
    plateau_mask = xa > growth_end
    frac = {
        "lag": float(np.sum(D[lag_mask]) / total),
        "growth": float(np.sum(D[growth_mask]) / total),
        "plateau": float(np.sum(D[plateau_mask]) / total),
    }
    out["phase_fractions"] = frac
    out["phase_boundaries"] = {"t0": ph["t0"], "lag_end": lag_end,
                               "growth_end": growth_end, "t_end": ph["t_end"],
                               "t50": ph["t50"]}
    out["phase_note"] = ("fraction of Sigma D(t) in each phase; growth/inflection "
                         "is expected to dominate over the plateau (a flat plateau "
                         "adds little Fisher information)")
    return out


# --------------------------------------------------------------------------- #
#  Redundancy — effective independent measurements
# --------------------------------------------------------------------------- #
def redundancy(D: list[float]) -> dict:
    """Effective independent measurements n_eff = participation ratio of the
    per-point leverage (Σ D)² / Σ D². redundancy = 1 − n_eff/n. Also reports the
    #points needed to reach 90% of Σ D (a long flat plateau ⇒ high redundancy)."""
    d = np.asarray(D, float)
    n = int(d.size)
    s1 = float(np.sum(d))
    s2 = float(np.sum(d ** 2))
    n_eff = (s1 * s1 / s2) if s2 > _EPS else 0.0
    n_eff = min(max(n_eff, 0.0), float(n))
    # #points (largest-leverage first) to reach 90% of the total information
    n90 = None
    if s1 > _EPS:
        srt = np.sort(d)[::-1]
        c = np.cumsum(srt)
        hit = np.where(c >= 0.9 * s1)[0]
        n90 = int(hit[0] + 1) if hit.size else n
    return {
        "n_points": n,
        "n_eff_participation_ratio": float(n_eff),
        "redundancy": float(1.0 - n_eff / n) if n > 0 else 0.0,
        "n_points_for_90pct_information": n90,
        "note": ("n_eff = (Sigma D)^2/Sigma D^2 (participation ratio of per-point "
                 "leverage); a long flat plateau of near-identical low-leverage "
                 "points raises redundancy"),
    }


# --------------------------------------------------------------------------- #
#  Expected variance reduction (optimal design) — rank-1 Sherman–Morrison
# --------------------------------------------------------------------------- #
def _sensitivity_at(name: str, params: dict, order: list[str], tstar: float) -> np.ndarray:
    """s(t*) — the sensitivity row at a single candidate new time t* (central FD)."""
    func = REGISTRY[name]["func"]
    theta = np.array([params[p] for p in order], float)
    p = len(theta)
    s = np.zeros(p)
    for j in range(p):
        h = 1e-6 * max(abs(theta[j]), 1.0)
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        vp = float(np.asarray(func(np.array([tstar], float), *tp), float)[0])
        vm = float(np.asarray(func(np.array([tstar], float), *tm), float)[0])
        val = (vp - vm) / (2.0 * h)
        s[j] = val if math.isfinite(val) else 0.0
    return s


def expected_variance_reduction(name: str, params: dict, order: list[str], C: np.ndarray,
                                sigma2: float, x, phase_bounds: dict,
                                n_grid: int = 60) -> dict:
    """For candidate new times t* on a grid spanning the observed window (extended
    slightly to expose 'extend the window' recommendations), the rank-1
    Sherman–Morrison variance reduction of a target functional g:
        ΔVar(t*) = (Cᵀ s(t*))²_target / (σ² + s(t*)ᵀ C s(t*)).
    We evaluate it for the amplitude/rate parameters and for a t50-proxy functional
    (the t0/location parameter when present). Returns the argmax t* per target +
    which phase it lands in — the rigorous basis for 'sample denser at the knee /
    extend the window'."""
    xa = np.asarray(x, float)
    span = float(xa.max() - xa.min()) or 1.0
    # extend 25% past the last observed time so a right-censored plateau surfaces as
    # 'the best next point is BEYOND the current window' (extend-window signal)
    grid = np.linspace(float(xa.min()), float(xa.max()) + 0.25 * span, n_grid)

    # target functionals: unit vectors picking a parameter of interest. amplitude /
    # rate / location (t50-proxy). We reduce Var(g^T θ) for g = e_target.
    targets = {}
    lower = {p.lower(): p for p in order}
    if "amp" in lower:
        targets["amplitude"] = lower["amp"]
    if "k" in lower:
        targets["rate_k"] = lower["k"]
    if "t0" in lower:
        targets["t50_proxy_t0"] = lower["t0"]
    if not targets:                                  # fallback: the largest-|θ| param
        theta = np.array([params[p] for p in order], float)
        targets["dominant_param"] = order[int(np.argmax(np.abs(theta)))]

    def _phase_of(t: float) -> str:
        if not phase_bounds.get("cooperative", phase_bounds.get("t50") is not None):
            pass
        le = phase_bounds.get("lag_end")
        ge = phase_bounds.get("growth_end")
        tend = phase_bounds.get("t_end", float(xa.max()))
        if le is None or ge is None:
            return "beyond_window" if t > float(xa.max()) else "single_region"
        if t > tend:
            return "beyond_window"
        if t <= le:
            return "lag"
        if t <= ge:
            return "growth"
        return "plateau"

    results = {}
    for tname, pname in targets.items():
        g = np.zeros(len(order))
        g[order.index(pname)] = 1.0
        Cg = C @ g                                   # column of C for the target
        best_t, best_dv = None, -1.0
        curve = []
        for t in grid:
            s = _sensitivity_at(name, params, order, float(t))
            denom = sigma2 + float(s @ C @ s)
            num = float(s @ Cg) ** 2
            dv = num / denom if denom > _EPS else 0.0
            curve.append(dv)
            if dv > best_dv:
                best_dv, best_t = dv, float(t)
        results[tname] = {
            "target_parameter": pname,
            "best_time": best_t,
            "best_delta_var": float(best_dv),
            "best_time_phase": _phase_of(best_t) if best_t is not None else None,
            "beyond_current_window": bool(best_t is not None and best_t > float(xa.max()) + _EPS),
        }
    # overall best next point = the largest single ΔVar across targets
    overall = max(results.values(), key=lambda r: r["best_delta_var"]) if results else None
    return {
        "grid_min": float(grid.min()), "grid_max": float(grid.max()),
        "observed_t_max": float(xa.max()),
        "per_target": results,
        "recommended_time": (overall["best_time"] if overall else None),
        "recommended_time_phase": (overall["best_time_phase"] if overall else None),
        "recommended_beyond_window": bool(overall["beyond_current_window"] if overall else False),
        "note": ("Sherman-Morrison rank-1 variance reduction for adding one point at "
                 "t*; the argmax is the most informative next measurement for the "
                 "target functional (conditional on the linearized model)"),
    }


# --------------------------------------------------------------------------- #
#  Bayesian information gain (EIG) — prior-conditional, flagged
# --------------------------------------------------------------------------- #
def expected_information_gain(fim: np.ndarray, theta: np.ndarray, prior_cv: float) -> dict:
    """EIG = ½·ln det(I + Σ₀𝓘) for a weak Gaussian prior Σ₀ = diag((prior_cv·θ̂)²).
    Prior-conditional (state prior_cv). This is the expected reduction in posterior
    entropy relative to the stated prior — NOT an absolute information content."""
    p = fim.shape[0]
    s0 = (prior_cv * np.abs(theta)) ** 2
    s0 = np.where(s0 > _EPS, s0, _EPS)               # guard θ_j=0 -> tiny prior var
    Sigma0 = np.diag(s0)
    M = np.eye(p) + Sigma0 @ fim
    sign, logabsdet = np.linalg.slogdet(M)
    if sign <= 0 or not math.isfinite(logabsdet):
        # fall back to eigenvalue product of the symmetric part (never raises)
        ev = np.clip(np.linalg.eigvalsh(0.5 * (M + M.T)), _EPS, None)
        logabsdet = float(np.sum(np.log(ev)))
    eig = 0.5 * float(logabsdet)
    return {
        "eig_nats": eig,
        "eig_bits": eig / math.log(2.0),
        "prior_cv": float(prior_cv),
        "prior": "Sigma0 = diag((prior_cv*theta_hat)^2) — weak Gaussian, VERSIONED",
        "note": ("EIG = 0.5*ln det(I + Sigma0*FIM); PRIOR-CONDITIONAL (a weak "
                 "Gaussian prior). It is the expected posterior-entropy reduction "
                 "relative to that prior, not absolute information content"),
    }


# --------------------------------------------------------------------------- #
#  Information richness score — a bounded collapse of the spectrum (shown WITH it)
# --------------------------------------------------------------------------- #
def information_richness(spectrum: dict, entropy: dict, p: int, snr: float = None) -> dict:
    """A bounded, normalized transform of ½·ln det 𝓘 (total curvature) and
    erank/p (dimensional richness). ALWAYS returned WITH the spectrum it collapses
    — never instead of it. Three orthogonal components:
        volume_term = logistic(½·ln det 𝓘)   in (0,1)  — how much total information
        rank_term   = erank / p               in [0,1]  — how many directions
        signal_term = min(SNR, 1)             in [0,1]  — is there a transition at all
    richness = (volume_term · rank_term · signal_term)^(1/3) (geometric mean).

    The signal_term is the HONEST flat-curve gate: a curve whose fitted model has a
    dynamic range at or below the noise (SNR = fitted_range/sigma ≲ 1) carries no
    information about a TRANSITION — its parameters may be well-determined (a flat
    line has a crisp intercept) but it is information-poor about aggregation, so
    richness -> ~0. Without SNR the term is 1 (pure spectral richness)."""
    half_logdet = entropy.get("half_logdet_fim")
    if half_logdet is None:                          # rank-deficient -> observable form
        half_logdet = entropy.get("half_pseudo_logdet_fim")
    erank = spectrum.get("effective_rank_entropy", 0.0)
    rank_term = float(min(max(erank / p, 0.0), 1.0)) if p > 0 else 0.0
    # signal_term saturates to 1 for a transition well above the noise (SNR>~3) and
    # collapses to ~0 for a flat curve (SNR<~1): 1 - exp(-SNR) is a smooth hard gate.
    signal_term = (float(1.0 - math.exp(-max(snr, 0.0))) if snr is not None else 1.0)
    if half_logdet is None or not math.isfinite(half_logdet):
        return {"information_richness_score": 0.0, "volume_term": 0.0,
                "rank_term": rank_term, "signal_term": signal_term,
                "half_logdet_fim": None,
                "note": "degenerate FIM -> richness 0 (collapse of an empty spectrum)"}
    # squash the unbounded ½·ln det into (0,1) with a gentle logistic (scale 5 nats)
    volume_term = 1.0 / (1.0 + math.exp(-float(half_logdet) / 5.0))
    # spectral richness (volume x rank) is a geometric mean; the signal_term is a
    # MULTIPLICATIVE envelope (a flat curve is information-poor no matter how crisp
    # its intercept), so richness -> 0 as SNR -> 0.
    score = math.sqrt(max(volume_term, 0.0) * max(rank_term, 0.0)) * max(signal_term, 0.0)
    return {
        "information_richness_score": float(score),
        "volume_term": float(volume_term),
        "rank_term": float(rank_term),
        "signal_term": float(signal_term),
        "half_logdet_fim": float(half_logdet),
        "note": ("BOUNDED collapse of the spectrum: geometric mean of a "
                 "logistic-squashed 0.5*ln det I (total information), erank/p "
                 "(dimensional richness), and min(SNR,1) (is there a transition). "
                 "Shown WITH the spectrum, never instead of it; a sloppy model "
                 "(erank<<p) is penalised via rank_term, a flat curve via signal_term"),
    }


# --------------------------------------------------------------------------- #
#  Recommended measurement — M9 action language + censoring tie-in
# --------------------------------------------------------------------------- #
def recommended_measurement(evr: dict, obsv: dict, phase_frac: dict,
                            censoring_class: str, plateau_reached: bool) -> dict:
    """Turn the optimal-design scan into an M9-language experiment action, tied to
    censoring: a right-censored curve whose amplitude/plateau parameter is
    unobservable AND whose best next point is beyond the window ⇒ EXTEND the
    observation window (M9's `extend_observation_window`)."""
    rec_phase = evr.get("recommended_time_phase")
    beyond = evr.get("recommended_beyond_window")
    right_censored = censoring_class in ("right", "left_right")
    amp_unobservable = any(
        (not v.get("observable")) and k.lower() in ("amp", "base")
        for k, v in obsv.get("per_parameter", {}).items()
    )
    # decision
    if beyond or (right_censored and not plateau_reached) or (right_censored and amp_unobservable):
        experiment = "extend_observation_window"
        rationale = ("the most informative next measurement lies beyond the current "
                     "window (right-censored / plateau unobservable) — extend the "
                     "observation window to constrain the amplitude/plateau")
    elif rec_phase in ("growth", "lag") or (isinstance(phase_frac, dict)
                                            and phase_frac.get("growth", 0.0) >= phase_frac.get("plateau", 1.0)):
        experiment = "sample_denser_at_transition"
        rationale = ("information concentrates in the lag/growth (knee) region — "
                     "sample denser through the transition to sharpen the rate/lag")
    else:
        experiment = "sample_denser_at_recommended_time"
        rationale = ("add a measurement at the optimal-design argmax to reduce the "
                     "target-functional variance")
    return {
        "experiment": experiment,
        "recommended_time": evr.get("recommended_time"),
        "recommended_time_phase": rec_phase,
        "beyond_current_window": bool(beyond),
        "tied_to_censoring": right_censored,
        "rationale": rationale,
        "action_language": "M9 (engine/m9_recommender.py) experiment vocabulary",
    }


# --------------------------------------------------------------------------- #
#  Honesty block — attached to EVERY result
# --------------------------------------------------------------------------- #
_HONESTY = {
    "fim_is_local": "the FIM is the curvature of the log-likelihood at theta_hat (a LOCAL/point property)",
    "fim_is_linearized": "linearized (Laplace/Gaussian) approximation of the likelihood; a sensitivity-based observed FIM, not the exact information",
    "model_conditional": "conditional on the M2-SELECTED (best-AICc) model AND the Sigma=sigma^2 I noise model",
    "post_selection": "the FIM, spectrum and per-parameter CRLB are CONDITIONAL on the AICc-winning model -- a per-parameter CRLB cannot be model-averaged, because a parameter of one model is not a parameter of another. The collapsed richness VERDICT is averaged over M3's selection posterior in the `post_selection` block, which also reports the spread across candidates and the weight of the model conditioned on",
    "richness_is_a_collapse": "information_richness_score is a bounded COLLAPSE of the spectrum — always returned WITH the spectrum, never instead of it",
    "eig_prior_conditional": "EIG is PRIOR-CONDITIONAL (a stated weak Gaussian prior), not absolute information content",
    "sloppy_is_reported": "a sloppy model has erank << n_params and M12 reports it (effective_rank_hard / _entropy vs n_params)",
}


# --------------------------------------------------------------------------- #
#  Public entry point — the full per-curve information analysis
# --------------------------------------------------------------------------- #
def _select_model(series: dict, fit_result: dict | None, force_model=None):
    """Pick the M2-SELECTED (best-AICc) converged fit — the same model M3/M4
    analyse. Returns (name, fit, handling) or (None, None, handling).

    `force_model` conditions on a NAMED model instead of the AICc winner. It
    exists for the post-selection pass: to ask what the information geometry
    would have been had the bootstrap's runner-up won, you have to be able to
    condition on the runner-up."""
    fr = fit_result or fit_curve(series)
    conv = {n: r for n, r in fr.get("fits", {}).items()
            if r.get("converged") and r.get("aicc") is not None and r.get("params")}
    if not conv:
        return None, None, fr.get("handling")
    if force_model is not None:
        if force_model not in conv:
            return None, None, fr.get("handling")
        return force_model, conv[force_model], fr.get("handling")
    best = min(conv, key=lambda n: conv[n]["aicc"])
    return best, conv[best], fr.get("handling")


def assess_information(series: dict, digitization_frac: float = _DEFAULT_DIGITIZATION_FRAC,
                       prior_cv: float = _DEFAULT_PRIOR_CV,
                       fit_result: dict | None = None,
                       force_model=None) -> dict:
    """Full per-curve information analysis (all of §3-M12). NEVER raises:
    degenerate / flat / no-fit -> a flagged minimal result (erank≈0/1, richness≈0).

    Args:
        series: an M1-triaged curve (needs x_hours + m1.y_processed / y_intensity).
        digitization_frac: optional digitization-variance floor fraction (of range).
        prior_cv: weak-Gaussian-prior CV for the EIG (1.0 = very weak).
    """
    m1 = series.get("m1", {})
    x = series.get("x_hours") or []
    y = m1.get("y_processed") or series.get("y_intensity") or []
    cens = m1.get("censoring_class", "none")
    plateau_reached = bool(m1.get("plateau_reached"))
    y_range = (float(np.max(y)) - float(np.min(y))) if len(y) else 0.0

    out = {
        "series_id": series.get("series_id"),
        "version": M12_VERSION,
        "honesty": dict(_HONESTY),
        "censoring_class": cens,
    }

    name, fit, handling = _select_model(series, fit_result, force_model)
    out["handling"] = handling
    out["selected_model"] = name
    if name is None or len(x) != len(y) or len(x) < 3:
        out.update({"status": "no_fit",
                    "information_richness_score": 0.0,
                    "effective_rank_hard": 0, "effective_rank_entropy": 0.0,
                    "note": "no converged model / malformed grid — minimal flagged result"})
        return out

    params = fit["params"]
    order = REGISTRY[name]["params"]
    theta = np.array([params[p] for p in order], float)
    n, p = len(x), len(order)
    sse = float(fit.get("sse") or 0.0)

    out["model_selection"] = {
        "selected_by": "AICc (M2 best_by_aicc / M3-M4 convention)",
        "n_params": p, "n_points": n, "sse": sse, "r2": fit.get("r2"),
        "params": {k: float(v) for k, v in params.items()},
        "m2_param_se": fit.get("param_se"),
    }

    # model-predicted dynamic range (for the SNR flat-gate); f(t;theta) over the grid
    func = REGISTRY[name]["func"]
    yhat = np.asarray(func(np.asarray(x, float), *theta), float)
    model_range = float(np.max(yhat) - np.min(yhat)) if np.all(np.isfinite(yhat)) else 0.0

    # 1. sensitivity Jacobian
    S, S_rel, order = sensitivity_jacobian(name, params, x)
    # per-parameter peak relative-sensitivity time (WHERE each param is most informed)
    rel_peak_time = {}
    xa = np.asarray(x, float)
    for j, nm in enumerate(order):
        col = np.abs(S_rel[:, j])
        rel_peak_time[nm] = float(xa[int(np.argmax(col))]) if np.any(col > 0) else None
    out["sensitivity"] = {
        "n_points": n, "n_params": p,
        "column_norms": [float(v) for v in np.sqrt(np.sum(S ** 2, axis=0))],
        "relative_sensitivity_peak_time": rel_peak_time,
        "note": "S[i,j]=df/dtheta_j (central FD); relative sensitivity (theta_j/f)*df/dtheta_j",
    }

    # 2. observed FIM
    fim, sigma2, noise_meta = fisher_information(S, sse, n, p, digitization_frac, y_range)
    out["noise_model"] = noise_meta

    # 3. eigen-spectrum
    spectrum = eigen_spectrum(fim, order)
    if not spectrum.get("computable"):
        # degenerate FIM (zero curvature) -> flagged minimal result, no crash
        out.update({
            "status": "degenerate_fim",
            "fisher_information": {"computable": False, "reason": spectrum.get("reason")},
            "information_richness_score": 0.0,
            "effective_rank_hard": 0, "effective_rank_entropy": 0.0,
            "note": "degenerate/flat FIM (zero curvature) — minimal flagged result",
        })
        return out
    out["fisher_information"] = {
        "matrix": [[float(v) for v in row] for row in fim],
        "sigma2": sigma2,
        "trace": float(np.trace(fim)),
    }
    out["spectrum"] = spectrum
    out["effective_rank_hard"] = spectrum["effective_rank_hard"]
    out["effective_rank_entropy"] = spectrum["effective_rank_entropy"]
    out["condition_number"] = spectrum["condition_number"]

    # 3b. condition number on the COLUMN-NORMALISED (correlation) FIM — this is the
    # object M3.fim_identifiability reports, computed here for the consistency
    # cross-check (M12 correlation-cond ≈ M3 collinearity-cond on the same curve).
    norms = np.sqrt(np.sum(S ** 2, axis=0))
    live = norms > 1e-12
    if np.all(live):
        Jn = S / norms
        Cn = Jn.T @ Jn
        en = np.clip(np.linalg.eigvalsh(0.5 * (Cn + Cn.T)), 0.0, None)
        emax, emin = float(en.max()), float(en[en > 0].min()) if np.any(en > 0) else 0.0
        corr_cond = (emax / emin) if emin > 0 else None
    else:
        corr_cond = None
    out["condition_number_correlation"] = (float(corr_cond) if corr_cond is not None
                                           and math.isfinite(corr_cond) else None)
    out["condition_number_note"] = ("condition_number is on the RAW observed FIM "
                                    "(S^T S/sigma^2, scale-carrying); "
                                    "condition_number_correlation is on the "
                                    "column-normalised FIM = M3.fim_identifiability's "
                                    "collinearity condition number (cross-checked)")

    # 4. entropies
    out["entropy"] = differential_entropy(spectrum, p)
    out["entropy"]["spectral_entropy"] = spectrum["spectral_entropy"]

    # 5. parameter observability + identifiable combinations
    obsv, C = parameter_observability(fim, theta, order, spectrum)
    out["parameter_observability"] = obsv
    # cross-check vs m2's param_se (should be the same order of magnitude)
    m2se = fit.get("param_se") or {}
    se_cross = {}
    for nm in order:
        crlb = obsv["per_parameter"][nm]["crlb_se"]
        m2 = m2se.get(nm)
        se_cross[nm] = {"crlb_se_m12": crlb, "param_se_m2": m2,
                        "same_order": (bool(m2 is not None and crlb > 0 and m2 > 0
                                            and abs(math.log10(crlb / m2)) < 1.0)
                                       if m2 else None)}
    out["se_cross_check_vs_m2"] = se_cross

    # 6. information-density profile + phase segmentation
    dens = information_density(S, C, x, name, params)
    out["information_density"] = dens

    # 7. redundancy
    out["redundancy"] = redundancy(dens["density"])

    # 8. expected variance reduction (optimal design)
    phase_bounds = dict(dens.get("phase_boundaries", {}))
    phase_bounds["cooperative"] = dens.get("cooperative_phases", False)
    evr = expected_variance_reduction(name, params, order, C, sigma2, x, phase_bounds)
    out["expected_variance_reduction"] = evr

    # 9. Bayesian information gain (prior-conditional)
    out["expected_information_gain"] = expected_information_gain(fim, theta, prior_cv)

    # 10. reports. SNR = fitted model dynamic range / noise sigma — the flat-curve
    # gate: a transition an order of magnitude above the noise is information-rich;
    # a fitted range at/below the noise (flat) is information-POOR about a transition.
    snr = model_range / math.sqrt(sigma2) if sigma2 > _EPS else 0.0
    out["signal_to_noise"] = {"model_dynamic_range": model_range,
                              "sigma": math.sqrt(sigma2), "snr": float(snr)}
    out["information_richness"] = information_richness(spectrum, out["entropy"], p, snr=snr)
    out["information_richness_score"] = out["information_richness"]["information_richness_score"]
    out["recommended_additional_measurements"] = recommended_measurement(
        evr, obsv, dens.get("phase_fractions", {}), cens, plateau_reached)

    out["status"] = "ok"
    return out


# --------------------------------------------------------------------------- #
#  POST-SELECTION: the information verdict, averaged over M3's selection posterior
# --------------------------------------------------------------------------- #
MIN_POSTERIOR_WEIGHT = 0.05      # ignore models the bootstrap essentially never picks


def post_selection_richness(series: dict, weights: dict,
                            fit_result: dict | None = None,
                            digitization_frac: float = _DEFAULT_DIGITIZATION_FRAC,
                            prior_cv: float = _DEFAULT_PRIOR_CV) -> dict:
    """The richness verdict re-asked of every model M3 actually selected.

    WHY ONLY THE SCALAR. Per-parameter CRLB cannot be model-averaged and this
    function does not pretend otherwise: `k` in a logistic and `b` in an `lnt`
    are different quantities, so averaging their standard errors would be
    meaningless arithmetic over incommensurable things. What IS common to every
    candidate is the collapsed verdict -- how informative this curve is -- and
    that is what gets averaged.

    WHY IT MATTERS HERE. Measured on the corpus, the AICc winner takes a median
    of only 0.770 of bootstrap resamples; 25.1% of curves are flagged
    multimodal, and the median runner-up holds 0.247. Reporting the information
    geometry of the winner alone, with no indication of how much it depended on
    that choice, states a conditional result as though it were unconditional.

    Returns the weighted mean, the SPREAD across candidates, and the weight of
    the model actually conditioned on -- the spread is the point: a small spread
    means the verdict is robust to the selection, a large one means it is not,
    and only reporting the mean would hide which."""
    out = {"available": False, "reason": None,
           "min_weight": MIN_POSTERIOR_WEIGHT,
           "selected_model_weight": None,
           "n_models_averaged": 0,
           "per_model": {},
           "weighted_richness": None,
           "richness_spread": None,
           "verdict_robust_to_selection": None}
    weights = {k: float(v) for k, v in (weights or {}).items()
               if isinstance(v, (int, float))}
    if not weights:
        out["reason"] = "no_selection_posterior"
        return out
    fr = fit_result or fit_curve(series)
    considered = {k: w for k, w in weights.items() if w >= MIN_POSTERIOR_WEIGHT}
    if not considered:
        out["reason"] = "no_model_above_min_weight"
        return out
    scores = {}
    for model, w in sorted(considered.items()):
        try:
            r = assess_information(series, digitization_frac=digitization_frac,
                                   prior_cv=prior_cv, fit_result=fr,
                                   force_model=model)
        except Exception:                               # never raise on bad input
            continue
        sc = r.get("information_richness_score")
        if sc is None or not math.isfinite(float(sc)):
            continue
        scores[model] = {"weight": w, "information_richness_score": float(sc)}
    if not scores:
        out["reason"] = "no_candidate_produced_a_score"
        return out
    tot = sum(v["weight"] for v in scores.values())
    vals = [v["information_richness_score"] for v in scores.values()]
    best, _fit, _h = _select_model(series, fr)
    out.update({
        "available": True,
        "per_model": scores,
        "n_models_averaged": len(scores),
        "selected_model_weight": (float(weights.get(best))
                                  if best in weights else None),
        "weighted_richness": float(
            sum(v["weight"] * v["information_richness_score"]
                for v in scores.values()) / tot) if tot > 0 else None,
        "richness_spread": float(max(vals) - min(vals)),
    })
    # "robust" is deliberately a CONJUNCTION: a confident winner is not enough if
    # the runner-up would have given a very different verdict, and a tight spread
    # is not enough if the winner is nearly arbitrary.
    w_sel = out["selected_model_weight"]
    out["verdict_robust_to_selection"] = bool(
        out["richness_spread"] is not None and out["richness_spread"] < 0.10
        and (w_sel is None or w_sel >= 0.60))
    return out


# --------------------------------------------------------------------------- #
#  Corpus rollup
# --------------------------------------------------------------------------- #
def _rollup(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("status") == "ok"]
    def _pct(a, q):
        return float(np.percentile(a, q)) if len(a) else None
    erank_hard = [r["effective_rank_hard"] for r in ok]
    erank_ent = [r["effective_rank_entropy"] for r in ok]
    nparams = [r["model_selection"]["n_params"] for r in ok]
    richness = [r["information_richness_score"] for r in ok]
    redun = [r["redundancy"]["redundancy"] for r in ok]
    cond = [r["condition_number"] for r in ok if r.get("condition_number") is not None]
    # sloppy = erank_hard < n_params (a direction is unconstrained)
    sloppy = sum(1 for r in ok if r["effective_rank_hard"] < r["model_selection"]["n_params"])
    # info fraction by phase (only cooperative curves have real phases)
    coop = [r for r in ok if r["information_density"].get("cooperative_phases")]
    lag_f = [r["information_density"]["phase_fractions"].get("lag", 0.0) for r in coop]
    growth_f = [r["information_density"]["phase_fractions"].get("growth", 0.0) for r in coop]
    plateau_f = [r["information_density"]["phase_fractions"].get("plateau", 0.0) for r in coop]
    # modal recommended-measurement experiment + phase
    rec_exp = Counter(r["recommended_additional_measurements"]["experiment"] for r in ok)
    rec_phase = Counter(r["recommended_additional_measurements"].get("recommended_time_phase")
                        for r in ok)
    erank_hist = Counter(r["effective_rank_hard"] for r in ok)
    return {
        "version": M12_VERSION,
        "n_analysed": len(results),
        "n_ok": len(ok),
        "n_cooperative": len(coop),
        "effective_rank_hard_distribution": {str(k): v for k, v in sorted(erank_hist.items())},
        "effective_rank_hard_median": _pct(erank_hard, 50),
        "effective_rank_entropy_median": _pct(erank_ent, 50),
        "n_params_median": _pct(nparams, 50),
        "n_sloppy_erank_lt_nparams": sloppy,
        "frac_sloppy": (sloppy / len(ok)) if ok else None,
        "information_richness_median": _pct(richness, 50),
        "information_richness_iqr": [_pct(richness, 25), _pct(richness, 75)],
        "redundancy_median": _pct(redun, 50),
        "redundancy_iqr": [_pct(redun, 25), _pct(redun, 75)],
        "condition_number_median": _pct(cond, 50),
        "info_fraction_by_phase_median": {
            "lag": _pct(lag_f, 50), "growth": _pct(growth_f, 50),
            "plateau": _pct(plateau_f, 50),
        },
        "growth_gt_plateau_fraction": (
            float(np.mean([g > p for g, p in zip(growth_f, plateau_f)])) if coop else None),
        "recommended_measurement_modal": (rec_exp.most_common(1)[0][0] if rec_exp else None),
        "recommended_measurement_distribution": dict(rec_exp),
        "recommended_phase_modal": (rec_phase.most_common(1)[0][0] if rec_phase else None),
        "recommended_phase_distribution": {str(k): v for k, v in rec_phase.items()},
        "honesty": ("all metrics are LOCAL / LINEARIZED (Laplace) / model-conditional "
                    "(M2-selected) / POST-SELECTION; richness is a bounded collapse of "
                    "the spectrum; EIG is prior-conditional; sloppy models (erank<n_params) "
                    "are counted and reported, not hidden"),
    }


# --------------------------------------------------------------------------- #
#  I/O / CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="PRISE M12 information content & design geometry")
    ap.add_argument("--input", type=Path,
                    default=root / "data" / "processed" / "curves_triaged.jsonl")
    ap.add_argument("--output", type=Path,
                    default=root / "data" / "processed" / "information_content.jsonl")
    ap.add_argument("--rollup", type=Path,
                    default=root / "data" / "processed" / "information_content.json")
    ap.add_argument("--m3", type=Path,
                    default=root / "data" / "processed" / "m3_sample.jsonl",
                    help="M3 selection posterior, for the post-selection pass")
    ap.add_argument("--no-post-selection", action="store_true",
                    help="skip the model-averaged richness (conditional only)")
    ap.add_argument("--shards", type=int, default=1,
                    help="split the fittable curves into N disjoint shards")
    ap.add_argument("--shard", type=int, default=0,
                    help="which shard (0..shards-1) this process handles")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--digitization-frac", type=float, default=_DEFAULT_DIGITIZATION_FRAC)
    ap.add_argument("--prior-cv", type=float, default=_DEFAULT_PRIOR_CV)
    args = ap.parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input} (run M1 first)")

    results = []
    n = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # M3's selection posterior, for the post-selection pass. Optional: absent or
    # unreadable, M12 still emits its conditional result and SAYS the averaging
    # was unavailable rather than silently reporting the winner as unconditional.
    weights_by_series = {}
    if not args.no_post_selection and args.m3 and args.m3.exists():
        try:
            with open(args.m3, encoding="utf-8") as mh:
                for ln in mh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    _r = json.loads(ln)
                    _fq = (_r.get("selection") or {}).get("selection_frequencies")
                    if _r.get("series_id") and _fq:
                        weights_by_series[_r["series_id"]] = _fq
        except Exception:
            weights_by_series = {}

    print(f"[M12] information content over fittable curves in {args.input.name} ...")
    fittable_idx = -1
    with open(args.input, encoding="utf-8") as fh, \
            open(args.output, "w", encoding="utf-8") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("m1", {}).get("fittability_class") != "fittable":
                continue
            # sharding: curves are analysed independently and M12 draws no
            # random numbers, so a shard emits exactly the records a single
            # process would have produced for those indices
            fittable_idx += 1
            if args.shards > 1 and (fittable_idx % args.shards) != args.shard:
                continue
            res = assess_information(s, digitization_frac=args.digitization_frac,
                                    prior_cv=args.prior_cv)
            if args.no_post_selection:
                res["post_selection"] = {"available": False,
                                         "reason": "disabled_by_flag"}
            else:
                res["post_selection"] = post_selection_richness(
                    s, weights_by_series.get(res.get("series_id")) or {},
                    digitization_frac=args.digitization_frac,
                    prior_cv=args.prior_cv)
            results.append(res)
            out.write(json.dumps(res) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    roll = _rollup(results)
    args.rollup.write_text(json.dumps(roll, indent=2), encoding="utf-8")
    print(f"[M12] wrote {args.output}  ({n} curves)")
    print(f"[M12] wrote {args.rollup}")
    # readable summary (ASCII-only for Windows cp1252 consoles)
    print(json.dumps({
        "n_ok": roll["n_ok"],
        "effective_rank_hard_median": roll["effective_rank_hard_median"],
        "effective_rank_distribution": roll["effective_rank_hard_distribution"],
        "n_sloppy_erank_lt_nparams": roll["n_sloppy_erank_lt_nparams"],
        "frac_sloppy": roll["frac_sloppy"],
        "information_richness_median": roll["information_richness_median"],
        "redundancy_median": roll["redundancy_median"],
        "info_fraction_by_phase_median": roll["info_fraction_by_phase_median"],
        "growth_gt_plateau_fraction": roll["growth_gt_plateau_fraction"],
        "recommended_measurement_modal": roll["recommended_measurement_modal"],
        "recommended_phase_modal": roll["recommended_phase_modal"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
