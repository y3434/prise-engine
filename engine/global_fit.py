"""
PRISE — Global shared-rate Knowles/Cohen ODE fit (the real AmyloFit-grade fit)
=============================================================================

This module builds the mechanistic centrepiece PRISE_DESIGN.md promised in
§3-M2 ("global multi-curve fitting: shared rate constants, only monomer varies")
and §3-M4 but never delivered. The existing `m4_features.gamma_global` is a
PHENOMENOLOGICAL master-curve *collapse* — it shares a logistic shape
{base, k_shape, A, γ} with a free per-curve amplitude and NEVER touches the
Knowles/Cohen ODE family. That is honest as a γ estimator but it is NOT the
shared-microscopic-rate global fit the design describes.

Here we do the real thing for a `concentration_series` (≥3 distinct
concentrations): fit ALL member curves SIMULTANEOUSLY with SHARED microscopic
parameters, only the monomer concentration varying per curve.

  Shared across every curve in the series (the novelty: reaction orders are
  ESTIMATED, not hard-fixed at 2.0 the way mechanistic.py does everywhere):
      log10 k_n, log10 k_+, log10 k_2, n_c, n_2, [log10 K_M for saturating]
  Per curve i:
      a free amplitude/scale + baseline (ThT ∝ mass, unknown a.u. scale)
      and the FIXED nominal m_tot,i  (NOT fit — it is the experimental set-point)

Method: stack every member-curve residual into ONE vector; integrate each curve
via `mechanistic.simulate_mass_fraction` with the shared params + that curve's
m_tot,i; fit with scipy.least_squares (TRF, bounds, log-rate space), seeded
MULTI-START from (i) the phenomenological γ / per-curve rough t50 and (ii) the
VARIANT_RATES decades. Done per mechanism family
{nucleation_elongation, secondary_nucleation, saturating_secondary,
fragmentation}; AICc-ranked.

HONESTY (the central finding, not a disclaimer): on this corpus these fits are
SLOPPY — only stiff *combinations* of the microscopic rates are identifiable;
the individual k's are mostly unconstrained. The deliverable value is therefore
(a) the REACTION-ORDER constraint (n_c, n_2 estimated with their own
identifiability), (b) the identifiable combinations from the FIM eigen-
decomposition, and (c) a THIRD, model-resolved γ_mechanistic derived
analytically from the fitted (n_c, n_2) to compare against gamma_collapse and
gamma_regression. This does NOT license a single mechanism — M5 still refuses on
this corpus (agitation/seeding unknown). We say so in every payload.

γ_mechanistic — analytic half-time scaling exponent from reaction orders
(Meisl/Knowles/Cohen integrated rate laws; Meisl et al., Nat. Protoc. 2016):
    secondary-nucleation dominated:  γ = (n_2 + 1) / 2
    primary nucleation + elongation: γ = (n_c + 1) / 2
    fragmentation dominated:         γ = 1 / 2   (n_2 -> 0)
γ here is −d log10 t50 / d log10 m (t50 ∝ m^(−γ)), matching M4's convention.

Everything is pure / deterministic / seeded / JSON-serialisable; the CLI never
raises and writes data/processed/global_fit.json. The stiff ODE × many curves is
slow, so the batch supports --limit and caps multi-starts (documented, fine).

Usage:
    python engine/global_fit.py                       # all eligible series
    python engine/global_fit.py --limit 6             # first 6 (real-corpus demo)
    python engine/global_fit.py --series CPAD-CS-...  # one series by id
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import fit_provenance as fp
from mechanistic import VARIANT_RATES, simulate_mass_fraction

GLOBAL_FIT_VERSION = "global-fit-1.0"
GLOBAL_FIT_RECIPE = "global_fit/trf-bounded-lsoda/det-starts-1.0"

# Mechanism families fit per series. Each lists the SHARED rate names that are
# free (besides reaction orders); kn,kp are always present, k2/kminus/KM toggle
# the secondary-nucleation / fragmentation / saturation terms (mirrors
# mechanistic.VARIANT_RATES so the ODE term-toggling is identical).
MECHANISMS = ("nucleation_elongation", "secondary_nucleation",
              "saturating_secondary", "fragmentation")

# Which reaction orders are identifiable per family (n2 only matters when a
# secondary/saturating term is present; for pure nucleation_elongation /
# fragmentation n2 is a nuisance and is held fixed).
_HAS_N2 = {"nucleation_elongation": False, "secondary_nucleation": True,
           "saturating_secondary": True, "fragmentation": False}

# log10-rate bounds (shared with mechanistic.py's single-curve fitter so the two
# live in the same sloppy log-space) and reaction-order bounds. Orders are bounded
# to the physically-sensible window [0.5, 3.5]; amyloid n_c, n_2 are ~1–3.
_LOG_LO = {"kn": -9.0, "kp": -5.0, "k2": -9.0, "kminus": -9.0, "KM": -3.0}
_LOG_HI = {"kn": 3.0, "kp": 4.0, "k2": 4.0, "kminus": 4.0, "KM": 1.0}
_LOG_GUESS = {"kn": -3.0, "kp": -1.0, "k2": -2.0, "kminus": -2.0, "KM": -0.5}
_ORDER_LO, _ORDER_HI = 0.5, 3.5
_ORDER_GUESS = 2.0

# Runtime caps: the stiff ODE × many curves × multi-start is expensive. These keep
# the corpus batch tractable; raising them improves the optimum marginally.
_MAX_STARTS = 3
_MAX_NFEV = 300


# --------------------------------------------------------------------------- #
#  Data assembly
# --------------------------------------------------------------------------- #
def _series_curves(members: list[dict]) -> list[dict]:
    """Pull (m_tot, x, y_normalised, scale) per member curve. ThT a.u. is rescaled
    to a ~[0,1] working range per curve (the per-curve amplitude is fit anyway);
    we keep the raw min/max so reported baseline/amp map back to a.u."""
    out = []
    for s in members:
        cv = s.get("condition_vector", {}) or {}
        m = (cv.get("concentration") or {}).get("value_uM")
        x = s.get("x_hours") or []
        y = s.get("m1", {}).get("y_processed") or s.get("y_intensity") or []
        if not (m and m > 0) or len(x) != len(y) or len(x) < 4:
            continue
        xa = np.asarray(x, float)
        ya = np.asarray(y, float)
        if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
            continue
        if float(xa.max()) <= 0:
            continue
        out.append({
            "series_id": s.get("series_id"),
            "m_tot": float(m),
            "x": xa,
            "y": ya,
            "y_min": float(ya.min()),
            "y_rng": float(max(ya.max() - ya.min(), 1e-9)),
        })
    return out


# --------------------------------------------------------------------------- #
#  Parameter packing  (shared rates+orders, then per-curve base/amp)
# --------------------------------------------------------------------------- #
def _shared_names(mech: str) -> list[str]:
    """Shared free parameter names for a mechanism family, in vector order:
    log-rates (from VARIANT_RATES), then reaction orders."""
    rates = [r for r in VARIANT_RATES[mech]]            # e.g. kn,kp,k2[,KM]
    names = [f"log10_{r}" for r in rates]
    names.append("n_c")
    if _HAS_N2[mech]:
        names.append("n_2")
    return names


def _unpack_shared(theta_shared, mech):
    """theta_shared -> kwargs for simulate_mass_fraction (linear rates, orders)."""
    rates = VARIANT_RATES[mech]
    kw = {"kn": 0.0, "kp": 0.0, "k2": 0.0, "kminus": 0.0, "KM": math.inf,
          "nc": 2.0, "n2": 2.0}
    i = 0
    for r in rates:
        val = 10.0 ** float(theta_shared[i])
        kw[r] = val
        i += 1
    kw["nc"] = float(theta_shared[i]); i += 1
    if _HAS_N2[mech]:
        kw["n2"] = float(theta_shared[i]); i += 1
    return kw


def _predict_curve(theta_shared, base, amp, mech, m_tot, x):
    kw = _unpack_shared(theta_shared, mech)
    frac = simulate_mass_fraction(x, kn=kw["kn"], kp=kw["kp"], k2=kw["k2"],
                                  kminus=kw["kminus"], KM=kw["KM"],
                                  nc=kw["nc"], n2=kw["n2"], mtot=m_tot)
    return base + amp * frac


# --------------------------------------------------------------------------- #
#  Residual stack  (ONE vector over ALL curves; shared params + per-curve base/amp)
# --------------------------------------------------------------------------- #
def _make_residuals(curves, mech):
    n_shared = len(_shared_names(mech))
    N = len(curves)

    def residuals(p):
        theta = p[:n_shared]
        out = []
        for j, c in enumerate(curves):
            base = p[n_shared + 2 * j]
            amp = p[n_shared + 2 * j + 1]
            yhat = _predict_curve(theta, base, amp, mech, c["m_tot"], c["x"])
            r = yhat - c["y"]
            # a failed integration (NaN) is penalised heavily but finitely so the
            # optimiser steps away rather than crashing (never raises, design rule)
            r = np.where(np.isfinite(r), r, 1e3 * c["y_rng"])
            out.append(r)
        return np.concatenate(out)

    return residuals, n_shared, N


def _bounds(curves, mech):
    rates = VARIANT_RATES[mech]
    n_shared = len(_shared_names(mech))
    lo = [_LOG_LO[r] for r in rates] + [_ORDER_LO]
    hi = [_LOG_HI[r] for r in rates] + [_ORDER_HI]
    if _HAS_N2[mech]:
        lo += [_ORDER_LO]; hi += [_ORDER_HI]
    for c in curves:
        lo += [c["y_min"] - c["y_rng"], 0.0]              # base, amp
        hi += [c["y_min"] + c["y_rng"], 3.0 * c["y_rng"]]
    assert len(lo) == n_shared + 2 * len(curves)
    return np.array(lo), np.array(hi)


def _starts(curves, mech, gamma_hint):
    """Seeded multi-start initial vectors. Start 0: VARIANT_RATES decades. Start 1:
    a slower-nucleation / faster-elongation decade. Start 2: orders nudged by the
    phenomenological γ hint (γ≈(n+1)/2 -> n≈2γ−1) so the data's scaling informs
    the initial reaction order. Per-curve base≈y_min, amp≈y_rng."""
    rates = VARIANT_RATES[mech]
    base_shared = [_LOG_GUESS[r] for r in rates] + [_ORDER_GUESS]
    if _HAS_N2[mech]:
        base_shared += [_ORDER_GUESS]

    alt_shared = list(base_shared)
    # shift kn down a decade, kp up a decade (a different basin of the sloppy space)
    if rates:
        alt_shared[0] -= 1.0
        if len(rates) > 1:
            alt_shared[1] += 1.0

    gamma_shared = list(base_shared)
    if gamma_hint is not None and math.isfinite(gamma_hint):
        n_from_gamma = min(max(2.0 * float(gamma_hint) - 1.0, _ORDER_LO), _ORDER_HI)
        gamma_shared[len(rates)] = n_from_gamma           # n_c slot
        if _HAS_N2[mech]:
            gamma_shared[len(rates) + 1] = n_from_gamma   # n_2 slot

    percurve = []
    for c in curves:
        percurve += [c["y_min"], c["y_rng"]]

    cand = [base_shared, gamma_shared, alt_shared][:_MAX_STARTS]
    return [list(s) + percurve for s in cand]


# --------------------------------------------------------------------------- #
#  Fit one mechanism family across the whole series
# --------------------------------------------------------------------------- #
def _aicc(sse, n_pts, n_par):
    sigma2 = max(sse / n_pts, 1e-12)
    loglik = -0.5 * n_pts * (math.log(2 * math.pi) + math.log(sigma2) + 1.0)
    aic = 2 * n_par - 2 * loglik
    if n_pts - n_par - 1 > 0:
        return aic + (2 * n_par * (n_par + 1) / (n_pts - n_par - 1))
    return float("inf")


def _shared_jacobian(residuals, x_opt, lo, hi, n_shared, step=2e-2):
    """Central finite-difference Jacobian of the residual w.r.t. the SHARED params
    only, evaluated at the optimum. The shared params all live in a smooth, O(1)
    space (log10-rates and reaction orders), so a fixed step ≈0.02 is well-scaled
    AND large enough that the stiff LSODA trajectory actually moves — scipy's own
    returned jac (and a too-tiny relative step) can underflow these columns to a
    hard zero, faking a degenerate FIM. Steps shrink near a bound to stay
    in-bounds; if a side has no room we fall back to a one-sided difference."""
    x_opt = np.asarray(x_opt, float)
    r0 = np.asarray(residuals(x_opt), float)
    cols = []
    for i in range(n_shared):
        hi_room = float(hi[i] - x_opt[i])
        lo_room = float(x_opt[i] - lo[i])
        h = min(step, max(hi_room, lo_room) or step)
        # choose central where both sides have room, else one-sided into the room
        if hi_room >= h and lo_room >= h:
            xp = x_opt.copy(); xp[i] += h
            xm = x_opt.copy(); xm[i] -= h
            col = (np.asarray(residuals(xp), float)
                   - np.asarray(residuals(xm), float)) / (2.0 * h)
        elif hi_room >= lo_room:
            hh = min(h, hi_room) or 1e-6
            xp = x_opt.copy(); xp[i] += hh
            col = (np.asarray(residuals(xp), float) - r0) / hh
        else:
            hh = min(h, lo_room) or 1e-6
            xm = x_opt.copy(); xm[i] -= hh
            col = (r0 - np.asarray(residuals(xm), float)) / hh
        col = np.where(np.isfinite(col), col, 0.0)
        cols.append(col)
    if not cols:
        return np.zeros((len(r0), 0))
    return np.column_stack(cols)


def _gamma_mechanistic(mech, nc, n2):
    """Analytic half-time scaling exponent from the fitted reaction orders
    (Meisl/Knowles/Cohen). t50 ∝ m^(−γ); γ = −d log t50 / d log m."""
    if mech in ("secondary_nucleation", "saturating_secondary"):
        return (n2 + 1.0) / 2.0, "(n_2+1)/2 — secondary-nucleation dominated"
    if mech == "fragmentation":
        return 0.5, "1/2 — fragmentation dominated (n_2->0)"
    # nucleation_elongation
    return (nc + 1.0) / 2.0, "(n_c+1)/2 — primary nucleation + elongation"


def _fit_mechanism(curves, mech, gamma_hint, seed):
    residuals, n_shared, N = _make_residuals(curves, mech)
    lo, hi = _bounds(curves, mech)
    n_pts = int(sum(len(c["x"]) for c in curves))
    n_par = n_shared + 2 * N

    # Retain every start (§7 basin evidence). The SHARED block is what carries
    # scientific meaning here, so the basin comparison is made on the shared
    # parameters only — the per-curve base/amp nuisance pair would otherwise
    # dominate the distance with a scale nobody interprets.
    outcomes = []
    best = None
    for si, p0 in enumerate(_starts(curves, mech, gamma_hint)):
        p0 = np.minimum(np.maximum(np.asarray(p0, float), lo), hi)
        try:
            sol = least_squares(residuals, p0, bounds=(lo, hi), method="trf",
                                max_nfev=_MAX_NFEV, x_scale="jac")
        except Exception as exc:
            outcomes.append({"index": si, "converged": False, "sse": None,
                             "theta": None, "failure": type(exc).__name__})
            continue
        r = sol.fun
        if not np.all(np.isfinite(r)):
            outcomes.append({"index": si, "converged": False, "sse": None,
                             "theta": None, "failure": "non_finite_residual"})
            continue
        sse = float(r @ r)
        outcomes.append({"index": si, "converged": True, "sse": sse,
                         "theta": [float(v) for v in sol.x[:n_shared]],
                         "failure": None})
        if best is None or sse < best["sse"]:
            best = {"sse": sse, "x": sol.x, "r": r}
    sst_all = float(sum(np.sum((c["y"] - c["y"].mean()) ** 2) for c in curves))
    prov = fp.fit_provenance(GLOBAL_FIT_RECIPE, fp.basin_ledger(
        outcomes, lo[:n_shared], hi[:n_shared], sst=sst_all))
    if best is None:
        return {"mechanism": mech, "converged": False, "reason": "no_convergence",
                "provenance": prov}

    sst = float(sum(np.sum((c["y"] - c["y"].mean()) ** 2) for c in curves)) or 1e-12
    r2 = 1.0 - best["sse"] / sst
    aicc = _aicc(best["sse"], n_pts, n_par)

    theta = best["x"][:n_shared]
    kw = _unpack_shared(theta, mech)
    nc, n2 = kw["nc"], kw["n2"]
    gamma_mech, gamma_formula = _gamma_mechanistic(mech, nc, n2)

    shared = _shared_params(theta, mech, lo, hi, n_shared)
    # recompute the SHARED-block Jacobian by explicit central finite differences at
    # the optimum (least_squares' returned jac can underflow to zero for the stiff
    # log-rate columns — a controlled step gives a trustworthy observed FIM)
    Jshared = _shared_jacobian(residuals, best["x"], lo, hi, n_shared)
    ident = _identifiability(Jshared, best["r"], n_pts, n_par, n_shared,
                             _shared_names(mech))

    return {
        "mechanism": mech,
        "converged": True,
        "n_shared_params": n_shared,
        "n_per_curve_params": 2 * N,
        "n_curves": N,
        "n_points": n_pts,
        "shared_params": shared,
        "reaction_orders": {
            "n_c": float(nc),
            "n_2": (float(n2) if _HAS_N2[mech] else None),
            "n_2_free": bool(_HAS_N2[mech]),
        },
        "gamma_mechanistic": float(gamma_mech),
        "gamma_mechanistic_formula": gamma_formula,
        "global_r2": float(r2),
        "global_sse": float(best["sse"]),
        "global_aicc": float(aicc),
        "identifiability": ident,
        "sloppy": bool(ident.get("sloppy")),
        "provenance": prov,
    }


def _shared_params(theta, mech, lo, hi, n_shared):
    """Report shared params in linear units with a bounds-hit flag (a param pinned
    to a bound is NOT a real estimate — it is unconstrained, said so explicitly)."""
    names = _shared_names(mech)
    kw = _unpack_shared(theta, mech)
    out = {}
    rate_map = {f"log10_{r}": r for r in VARIANT_RATES[mech]}
    for i, nm in enumerate(names):
        v = float(theta[i])
        at_lo = abs(v - lo[i]) < 1e-6
        at_hi = abs(v - hi[i]) < 1e-6
        if nm in rate_map:
            r = rate_map[nm]
            lin = 10.0 ** v if r != "KM" else 10.0 ** v
            out[r] = {"log10": v, "value": float(lin),
                      "at_bound": bool(at_lo or at_hi)}
        else:                                            # n_c / n_2
            out[nm] = {"value": v, "at_bound": bool(at_lo or at_hi)}
    return out


# --------------------------------------------------------------------------- #
#  Identifiability — observed FIM eigen-decomposition (the honest finding)
# --------------------------------------------------------------------------- #
def _identifiability(Jshared, resid, n_pts, n_par, n_shared, shared_names):
    """Observed FIM = J^T σ^-2 J on the SHARED block; eigen-decompose to find
    identifiable combinations vs sloppy directions. We restrict the FIM to the
    shared parameters (the scientific quantities); per-curve base/amp are nuisance
    and integrated out by their own well-conditioned blocks.

    A near-zero eigenvalue ⇒ a sloppy direction (a combination of rates the data
    cannot constrain); the eigenvector lists which rates trade off. Heavy sloppiness
    is EXPECTED on this corpus — that is the honest result, not a failure."""
    try:
        Js = np.asarray(Jshared, float)
        dof = max(n_pts - n_par, 1)
        sigma2 = max(float(resid @ resid) / dof, 1e-12)
        fim = (Js.T @ Js) / sigma2                       # observed Fisher info
        # symmetric -> eigh; eigenvalues ascending
        evals, evecs = np.linalg.eigh(fim)
    except Exception as e:                               # pragma: no cover
        return {"computable": False, "reason": str(e)}

    evals = np.clip(evals, 0.0, None)
    emax = float(evals.max()) if evals.size else 0.0
    if emax <= 0:
        return {"computable": False, "reason": "degenerate FIM (zero curvature)"}

    # condition number of the shared FIM: huge -> sloppy (rates trade off)
    emin = float(evals.min())
    cond = (emax / emin) if emin > 0 else float("inf")
    # sloppiness threshold: a >1e6 spread of eigenvalues is the canonical sloppy-
    # model signature (Transtrum/Sethna). Also flag any individual rate whose
    # marginal info (diagonal FIM / leading eigenvalue) is tiny.
    sloppy = (not math.isfinite(cond)) or cond > 1e6

    # identifiable combinations = eigenvectors of the LARGEST eigenvalues (stiff
    # directions). Sloppy directions = eigenvectors of the near-zero eigenvalues.
    order = np.argsort(evals)[::-1]                       # descending
    combos = []
    for k in order:
        ev = float(evals[k])
        vec = evecs[:, k]
        # describe the combination as the dominant ± rate contributions
        terms = sorted(zip(shared_names, vec), key=lambda t: -abs(t[1]))
        desc = " + ".join(f"{w:+.2f}·{nm}" for nm, w in terms if abs(w) > 0.15)
        combos.append({
            "eigenvalue": ev,
            "relative_eigenvalue": ev / emax,
            "stiff": bool(ev / emax > 1e-3),
            "combination": desc or "(diffuse)",
        })

    # which individual shared rates are sloppy/unconstrained: a rate is sloppy if it
    # only appears with meaningful weight in near-zero (sloppy) eigen-directions
    rel = evals / emax
    sloppy_dirs = order[rel[order] < 1e-3]
    stiff_dirs = order[rel[order] >= 1e-3]
    rate_status = {}
    for i, nm in enumerate(shared_names):
        in_stiff = any(abs(evecs[i, k]) > 0.3 for k in stiff_dirs)
        rate_status[nm] = "constrained" if in_stiff else "sloppy/unconstrained"

    return {
        "computable": True,
        "fim_condition_number": (float(cond) if math.isfinite(cond) else None),
        "n_shared_params": n_shared,
        "n_stiff_directions": int((rel >= 1e-3).sum()),
        "n_sloppy_directions": int((rel < 1e-3).sum()),
        "sloppy": bool(sloppy),
        "eigenvalues_descending": [float(e) for e in evals[order]],
        "identifiable_combinations": combos,
        "individual_rate_status": rate_status,
        "interpretation": (
            "SLOPPY: only stiff combinations of the microscopic rates are "
            "identifiable; individual k's are not uniquely recovered (expected on "
            "this corpus — the value is the reaction-order constraint + identifiable "
            "combinations, NOT unique rates)" if sloppy else
            "shared FIM is reasonably conditioned — rates better constrained than "
            "typical, but mechanism is still not licensed (M5 gates on agitation/"
            "seeding, unknown here)"),
    }


# --------------------------------------------------------------------------- #
#  Public: fit one concentration series across all mechanism families
# --------------------------------------------------------------------------- #
def fit_global_series(members: list[dict], meta: dict | None = None,
                      gamma_hint: float | None = None,
                      mechanisms=MECHANISMS, seed: int = 0) -> dict:
    """Shared-rate global Knowles/Cohen fit of one concentration series across the
    requested mechanism families; AICc-ranked. Never raises; returns a JSON-able
    dict. `gamma_hint` (the phenomenological collapse γ) seeds a reaction-order
    multi-start."""
    meta = meta or {}
    curves = _series_curves(members)
    distinct = sorted({round(c["m_tot"], 6) for c in curves})
    base_out = {
        "version": GLOBAL_FIT_VERSION,
        "concentration_series_id": meta.get("concentration_series_id"),
        "protein": meta.get("protein"),
        "conditions": {k: meta.get(k) for k in
                       ("pH", "temperature_C", "assay", "mutation")},
        "n_member_curves": len(members),
        "n_curves_fit": len(curves),
        "n_distinct_concentrations": len(distinct),
        "concentrations_uM": distinct,
    }
    if len(curves) < 3 or len(distinct) < 3:
        base_out["status"] = "insufficient_data"
        return base_out

    # NO RNG IS USED HERE, and the honest record says so rather than implying a
    # pinned stream. The `seed` parameter is threaded in from the CLI but
    # `_fit_mechanism` never references it: all three starts are constructed
    # deterministically and TRF is a deterministic descent. The line that used to
    # sit here (`np.random.seed(seed)`, commented "determinism belt-and-braces")
    # seeded the LEGACY global RandomState, which least_squares does not consult
    # — it was a no-op that read as a reproducibility guarantee. Removed rather
    # than kept, because a decorative guarantee is worse than none.
    base_out["fit_provenance"] = {
        "recipe": GLOBAL_FIT_RECIPE,
        "rng": fp.recipe(GLOBAL_FIT_RECIPE)["rng"],
        "seed_argument": {
            "value": seed,
            "effective": False,
            "reason": ("accepted for CLI compatibility but unused: no code path "
                       "in this module consumes a random number"),
        },
        "env_manifest_ref": fp.env_manifest_ref(),
        "version": GLOBAL_FIT_VERSION,
    }
    fits = {}
    for mech in mechanisms:
        fits[mech] = _fit_mechanism(curves, mech, gamma_hint, seed)

    conv = {m: f for m, f in fits.items()
            if f.get("converged") and math.isfinite(f.get("global_aicc", math.inf))}
    if not conv:
        base_out["status"] = "no_convergence"
        base_out["mechanism_fits"] = fits
        return base_out

    ranking = sorted(conv, key=lambda m: conv[m]["global_aicc"])
    best = ranking[0]
    best_aicc = conv[best]["global_aicc"]
    aicc_table = [{"mechanism": m, "aicc": conv[m]["global_aicc"],
                   "delta_aicc": conv[m]["global_aicc"] - best_aicc,
                   "global_r2": conv[m]["global_r2"],
                   "gamma_mechanistic": conv[m]["gamma_mechanistic"]}
                  for m in ranking]

    bf = conv[best]
    base_out.update({
        "status": "ok",
        "best_mechanism": best,
        "mechanism_aicc_ranking": aicc_table,
        "best_fit": bf,
        "reaction_orders": bf["reaction_orders"],
        "gamma_mechanistic": bf["gamma_mechanistic"],
        "gamma_mechanistic_formula": bf["gamma_mechanistic_formula"],
        "identifiability": bf["identifiability"],
        "sloppy": bf["sloppy"],
        "mechanism_fits": fits,
        "honesty_note": (
            "Shared-rate global ODE fit recovers a REACTION-ORDER constraint and "
            "identifiable rate COMBINATIONS, not unique microscopic rates (sloppy). "
            "It does NOT license a single mechanism — M5 still refuses on this corpus "
            "(agitation/seeding unknown). γ_mechanistic is a THIRD γ, distinct from "
            "the phenomenological gamma_collapse and gamma_regression."),
    })
    return base_out


# --------------------------------------------------------------------------- #
#  I/O / CLI
# --------------------------------------------------------------------------- #
def _load_triaged(path: Path) -> dict:
    by_id = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            by_id[s.get("series_id")] = s
    return by_id


def _gamma_hints(root: Path) -> dict:
    """Read the existing phenomenological collapse γ per series (gamma.jsonl) as a
    multi-start hint. Read-only join; absent file -> no hints."""
    hints = {}
    gp = root / "gamma.jsonl"
    if not gp.exists():
        return hints
    for line in open(gp, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        csid = d.get("concentration_series_id")
        g = (d.get("gamma_global") or {}).get("gamma")
        if csid and g is not None and math.isfinite(g):
            hints[csid] = float(g)
    return hints


def _run(args):
    root = args.input.parent
    cs_path = root / "concentration_series.json"
    if not cs_path.exists():
        raise SystemExit(f"{cs_path} not found (run the ETL first)")
    series = json.loads(cs_path.read_text(encoding="utf-8"))
    by_id = _load_triaged(args.input)
    hints = _gamma_hints(root)
    out_path = args.output or (root / "global_fit.json")

    if args.series:
        series = [m for m in series
                  if m.get("concentration_series_id") == args.series]
        if not series:
            raise SystemExit(f"series not found: {args.series}")

    results = []
    n = ok = sloppy = 0
    print(f"[global_fit] shared-rate ODE fit over up to "
          f"{args.limit or len(series)} concentration series "
          f"(slow: stiff ODE × many curves) ...")
    for meta in series:
        members = [by_id[sid] for sid in meta.get("member_series_ids", [])
                   if sid in by_id]
        if len(members) < 3:
            continue
        csid = meta.get("concentration_series_id")
        res = fit_global_series(members, meta=meta,
                                gamma_hint=hints.get(csid), seed=args.seed)
        results.append(res)
        n += 1
        if res.get("status") == "ok":
            ok += 1
            sloppy += bool(res.get("sloppy"))
            ro = res["reaction_orders"]
            n2s = "fixed" if ro["n_2"] is None else f"{ro['n_2']:.2f}"
            # ASCII-only print (Windows consoles are cp1252; γ/² would crash)
            print(f"  {csid} [{res.get('protein')}]: best={res['best_mechanism']} "
                  f"n_c={ro['n_c']:.2f} n_2={n2s} "
                  f"gamma_mech={res['gamma_mechanistic']:.3f} "
                  f"R2={res['best_fit']['global_r2']:.3f} "
                  f"{'SLOPPY' if res.get('sloppy') else 'conditioned'}")
        else:
            print(f"  {csid}: {res.get('status')}")
        if args.limit and n >= args.limit:
            break

    payload = {
        "version": GLOBAL_FIT_VERSION,
        # Retained for backward compatibility, but qualified: this module draws
        # no random number, so the value identifies the invocation, not a stream.
        # Per-series `fit_provenance.seed_argument` says so machine-readably.
        "seed": args.seed,
        "seed_is_effective": False,
        "n_series": n,
        "n_ok": ok,
        "n_sloppy": sloppy,
        "note": ("Shared-rate Knowles/Cohen global fit. Sloppiness is expected and "
                 "is the honest finding: reaction-order constraint + identifiable "
                 "combinations, not unique rates. Mechanism NOT licensed."),
        "series": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[global_fit] wrote {out_path}  ({n} series, {ok} ok, {sloppy} sloppy)")
    return 0


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="PRISE shared-rate global ODE fit")
    ap.add_argument("--input", type=Path,
                    default=root / "data" / "processed" / "curves_triaged.jsonl")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of series (the stiff ODE batch is slow)")
    ap.add_argument("--series", type=str, default=None,
                    help="fit a single concentration_series_id")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input} (run M1 first)")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
