"""
PRISE — Module M4: Feature Extractor, Definition Contract & Dual-γ
=================================================================

Turns fitted curves (M2) and selected models (M3) into the *interpretable*
quantities the rest of the pipeline reasons about — under an explicit, versioned
**definition contract**, because cross-protein comparison is only valid between
features computed the same way (a Gompertz t50 and a tangent-intercept lag from
two different definitions are NOT comparable). See PRISE_DESIGN.md §3-M4, §4.

Two products:

1. **Single-curve feature vector** (`extract_features`) — computed from the
   *fitted model* on a dense grid (so logistic/Gompertz/Richards/... all yield
   the same definitionally-homogeneous features):
     * `t50` (primary anchor)         — half-max crossing
     * `lag_time`                     — tangent-intercept to baseline (def attached)
     * `lag_to_t50_ratio`             — dimensionless shape invariant (a rare
                                        single-curve mechanistic signal)
     * `inflection_time`, `max_rate`, `max_rate_normalized`
     * `t10`, `t90`, `transition_width`, `transition_sharpness`
     * `plateau`, `dynamic_range`
   Each carries a **censoring-aware status** (point vs lower-bound vs biased)
   and **wild-bootstrap CIs**.

2. **Dual-γ** (`dual_gamma`) — the headline mechanistic constraint for a
   `concentration_series`. γ is the **half-time scaling exponent**: t50 ∝ m^(−γ).
   Two estimators, **separately provenanced and never merged**:
     * `gamma_regression` — censored regression of log t50 on log driving-force
       (right-censored curves contribute an inequality, not a point); with a
       log–log **curvature test** (concentration-dependent reaction order →
       saturating secondary nucleation / approach to m_crit / mechanism change),
       an **informative-censoring test** (blocks γ if censoring correlates with
       concentration), and an **m_crit identifiability diagnostic** (nominal-m is
       the default; a supersaturation correction is only applied when m_crit is
       actually constrained — otherwise a *flagged* downgrade, never silent).
     * `gamma_global` — γ from a **phenomenological master-curve COLLAPSE** that
       couples all member curves through one shared logistic SHAPE + one γ (free
       per-curve amplitude). It uses the whole curve shape (robust to per-curve
       censoring) and is distinct machinery from the t50 regression — but it is
       NOT a shared-microscopic-rate fit: it never touches the Knowles/Cohen ODE
       family (`method="master_curve_collapse"`, `is_shared_rate_ode_fit=False`).
       **The real shared-rate Knowles/Cohen ODE global fit (and its analytic THIRD
       γ_mechanistic from estimated reaction orders) lives in engine/global_fit.py;**
       when its output is available, dual_gamma joins it in as `global_ode_fit`.
   Their **disagreement** is reported as signal (it means the shape is not
   concentration-invariant): the fixed test ALWAYS reports the raw pointwise
   |γ_reg − γ_collapse| + a flag beyond a pre-registered margin (never suppressed
   when a CI is missing), gates on collapse quality (collapse_r2<0.8 ⇒ shape is
   concentration-dependent), and keeps the CI-overlap test when CIs exist.

γ is a **constraint on a combination of reaction orders, many-to-one onto
mechanism** (M5 gates mechanistic calls on curve shape, with γ as a consistency
check only). A NEGATIVE γ (t50 rising with concentration) is gated as non-physical
(`gamma_physical=false`), not emitted as a clean estimate. Everything here is
JSON-serialisable (web-API ready).

Usage:
    python engine/m4_features.py                      # per-curve features -> features.jsonl
    python engine/m4_features.py --limit 200 --bootstrap 150
    python engine/m4_features.py --gamma              # dual-γ per concentration series -> gamma.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize

import fit_provenance as fp
from m2_fit import fit_curve, fit_one
from m3_select import _mammen, predict, t50_of

# §2.3 recipe id for the two gamma bootstraps (see data/processed/fit_recipes.json)
M4_GAMMA_RECIPE = "m4/gamma-bootstrap-1.0"

# ----------------------------------------------------------------------------- #
# Versioned definition contract — attached to every feature payload so cross-
# protein comparisons can verify definition-homogeneity before pooling.
# ----------------------------------------------------------------------------- #
DEFINITION_CONTRACT = {
    "version": "m4-defs-1.0",
    "basis": "all features are read off the SELECTED model evaluated on a dense "
             "grid over the observed time window (model-smoothed, not raw points)",
    "t50": "time at which the fitted signal first reaches ymin + 0.5*(ymax-ymin)",
    "t10": "time at which the fitted signal first reaches ymin + 0.10*(ymax-ymin)",
    "t90": "time at which the fitted signal first reaches ymin + 0.90*(ymax-ymin)",
    "inflection_time": "time of maximum fitted slope over the observed window",
    "max_rate": "fitted slope d(signal)/d(time) at the inflection time",
    "max_rate_normalized": "max_rate * window_span / dynamic_range (dimensionless)",
    "lag_time": "tangent-intercept: inflection_time - (y_inflection - ymin)/max_rate "
                "(intersection of the maximum-slope tangent with the baseline ymin)",
    "lag_to_t50_ratio": "lag_time / t50 (dimensionless shape invariant)",
    "transition_width": "t90 - t10",
    "transition_sharpness": "t50 / (t90 - t10) (dimensionless; higher = sharper)",
    "plateau": "max fitted signal over the observed window",
    "dynamic_range": "ymax - ymin of the fitted signal over the observed window",
    "gamma": "half-time scaling exponent: t50 ∝ (driving force)^(-gamma)",
    "driving_force_default": "nominal monomer concentration (m); supersaturation "
                             "(m - m_crit) only when m_crit is identifiable",
}

_GRID_N = 600
_EPS = 1e-9
# γ reliability gates. Amyloid half-time scaling exponents are physically ~0–2
# (occasionally a little more); |γ| beyond this means the censored likelihood has
# diverged on a poorly-anchored series — reported as NON-identifiable, not faked.
GAMMA_PLAUSIBLE = 5.0
MIN_UNCENSORED_FOR_GAMMA = 3      # need >=3 real (non-inequality) t50 anchors


# --------------------------------------------------------------------------- #
#  Single-curve geometry off the fitted model
# --------------------------------------------------------------------------- #
def _first_crossing(grid, y, target):
    for i in range(len(grid) - 1):
        a, b = y[i] - target, y[i + 1] - target
        if a == 0.0:
            return float(grid[i])
        if a * b < 0 and y[i + 1] != y[i]:
            r = (target - y[i]) / (y[i + 1] - y[i])
            return float(grid[i] + r * (grid[i + 1] - grid[i]))
    return None


def feature_geometry(name: str, params: dict, x) -> dict | None:
    """Compute the raw geometric features from a fitted model. Returns None when
    the curve is flat / non-finite (no transition to characterise)."""
    grid = np.linspace(float(min(x)), float(max(x)), _GRID_N)
    y = predict(name, params, grid)
    if not np.all(np.isfinite(y)):
        return None
    ymin, ymax = float(y.min()), float(y.max())
    dyn = ymax - ymin
    if dyn < _EPS:
        return None
    t10 = _first_crossing(grid, y, ymin + 0.10 * dyn)
    t50 = _first_crossing(grid, y, ymin + 0.50 * dyn)
    t90 = _first_crossing(grid, y, ymin + 0.90 * dyn)
    dy = np.gradient(y, grid)
    n = len(grid)
    i = int(np.argmax(dy))                        # steepest rising point
    max_rate = float(dy[i])
    infl_t, y_infl = float(grid[i]), float(y[i])
    # A genuine sigmoidal (cooperative) transition accelerates then decelerates, so
    # its steepest point is INTERIOR and clearly steeper than the endpoints. A line
    # (lnt), decelerating saturation (exponential) or power growth has no interior
    # inflection — the tangent-intercept lag/inflection are then NOT cooperative-onset
    # features and must be flagged, not silently reported (definition-homogeneity).
    edge = max(dy[0], dy[-1], _EPS)
    has_inflection = (0 < i < n - 1) and (max_rate > 1.15 * edge)
    lag = (infl_t - (y_infl - ymin) / max_rate) if max_rate > _EPS else None
    span = float(max(x) - min(x)) or 1.0
    width = (t90 - t10) if (t10 is not None and t90 is not None) else None
    return {
        "t10": t10, "t50": t50, "t90": t90,
        "inflection_time": (infl_t if has_inflection else None),
        "max_rate": max_rate,
        "max_rate_normalized": max_rate * span / dyn,
        "lag_time": (lag if has_inflection else None),
        "lag_to_t50_ratio": (lag / t50 if (has_inflection and lag is not None
                                           and t50 and t50 > _EPS) else None),
        "transition_width": width,
        "transition_sharpness": (t50 / width if (width and width > _EPS and t50)
                                 else None),
        "plateau": ymax,
        "dynamic_range": dyn,
        "has_interior_inflection": bool(has_inflection),
    }


def _best_fit(series: dict, fit_result: dict | None = None):
    """Fit a curve's licensed bank (M2) and return (name, fit) of the AICc winner.
    A precomputed `fit_result` (from `m2_fit.fit_curve`) is reused to avoid
    re-fitting when the caller already has it (e.g. M5)."""
    fr = fit_result or fit_curve(series)
    conv = {n: r for n, r in fr.get("fits", {}).items()
            if r.get("converged") and r.get("aicc") is not None}
    if not conv:
        return None, None, fr.get("handling")
    best = min(conv, key=lambda n: conv[n]["aicc"])
    return best, conv[best], fr.get("handling")


def _bootstrap_feature_cis(name, params, x, y, B, seed):
    """Wild-bootstrap (Mammen) CIs for the features, refitting the SAME model each
    resample. These are *conditional on the selected model*; cross-model selection
    uncertainty for t50 is reported separately by M3 — kept distinct on purpose."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    yhat = predict(name, params, xa)
    resid = ya - yhat
    rng = np.random.default_rng(seed)
    keys = ("t50", "lag_time", "lag_to_t50_ratio", "inflection_time",
            "max_rate", "transition_width", "transition_sharpness")
    acc: dict[str, list] = defaultdict(list)
    for _ in range(B):
        ystar = yhat + resid * _mammen(len(resid), rng)
        r = fit_one(list(xa), list(ystar), name, n_starts=1)
        if not r.get("converged"):
            continue
        g = feature_geometry(name, r["params"], xa)
        if g is None:
            continue
        for k in keys:
            if g.get(k) is not None and math.isfinite(g[k]):
                acc[k].append(g[k])
    out = {}
    for k, vals in acc.items():
        if len(vals) >= 10:
            a = np.asarray(vals, float)
            out[k] = [float(np.percentile(a, 2.5)),
                      float(np.percentile(a, 97.5))]
    return out


def extract_features(series: dict, B: int = 0, seed: int = 0,
                     fit_result: dict | None = None) -> dict:
    """Single-curve feature vector under the definition contract, with censoring-
    aware status tags and (optional) wild-bootstrap CIs. `fit_result` (a precomputed
    `m2_fit.fit_curve` output) is reused to avoid re-fitting."""
    m1 = series.get("m1", {})
    cens = m1.get("censoring_class", "none")
    signal_basis = m1.get("signal_basis") or series.get("signal_basis")
    plateau_reached = bool(m1.get("plateau_reached"))
    x = series.get("x_hours") or []
    y = m1.get("y_processed") or series.get("y_intensity") or []

    out = {
        "series_id": series.get("series_id"),
        "definition_contract": DEFINITION_CONTRACT["version"],
        "signal_basis": signal_basis,
        "censoring_class": cens,
    }
    name, fit, handling = _best_fit(series, fit_result)
    out["handling"] = handling
    if name is None or len(x) != len(y) or len(x) < 3:
        out["status"] = "no_fit"
        return out

    g = feature_geometry(name, fit["params"], x)
    if g is None:
        out["status"] = "flat_no_transition"
        out["model"] = name
        return out

    # ---- censoring-aware interpretation of the headline features -------------
    # right / left_right (no plateau): the fitted ymax is an extrapolation, so the
    #   half-max it defines is reached too early -> t50 is a LOWER BOUND.
    # left / left_right (curve underway at t0): no lag phase observed -> lag undefined,
    #   t50 biased early.
    right_cens = cens in ("right", "left_right")
    left_cens = cens in ("left", "left_right")
    if right_cens:
        t50_status = "lower_bound"          # true t50 >= reported value
    elif left_cens:
        t50_status = "biased_early"
    elif cens == "interval":
        t50_status = "irregular"
    else:
        t50_status = "point"

    flags = []
    if not plateau_reached:
        flags.append("plateau_not_reached_extrapolated")
    if g["lag_time"] is not None and g["lag_time"] < 0:
        flags.append("lag_nonpositive_no_lag_phase")
    cooperative = g.get("has_interior_inflection", False)
    if not cooperative:
        flags.append("non_cooperative_no_interior_inflection")
    if left_cens:
        g = dict(g)
        g["lag_time"] = None                # lag is undefined under left-censoring
        g["lag_to_t50_ratio"] = None
        g["inflection_time"] = None
        lag_status = "undefined_left_censored"
    elif not cooperative:
        lag_status = "undefined_non_cooperative"   # line/downhill/power: no lag phase
    else:
        lag_status = "point"

    out.update({
        "status": "ok",
        "model": name,
        "r2": fit.get("r2"),
        "features": g,
        "t50_status": t50_status,
        "lag_status": lag_status,
        "plateau_reached": plateau_reached,
        "validity_flags": flags,
        "n_points": len(x),
    })
    if B > 0:
        out["bootstrap_ci"] = _bootstrap_feature_cis(name, fit["params"], x, y, B, seed)
        out["ci_reliable"] = len(x) >= 8
    return out


# --------------------------------------------------------------------------- #
#  Concentration-series t50 table (input to the dual-γ)
# --------------------------------------------------------------------------- #
def series_t50_table(members: list[dict]) -> list[dict]:
    """For each member curve: fit, extract t50, attach concentration + censoring.
    Right-censored t50 is recorded as a lower bound (handled by the censored
    regression); left-censored t50 is flagged biased and excluded from γ."""
    rows = []
    for s in members:
        cv = s.get("condition_vector", {}) or {}
        m = (cv.get("concentration") or {}).get("value_uM")
        if m is None or m <= 0:
            continue
        name, fit, _ = _best_fit(s)
        if name is None:
            continue
        t50 = t50_of(name, fit["params"], s.get("x_hours") or [])
        if t50 is None or t50 <= 0:
            continue
        cens = s.get("m1", {}).get("censoring_class", "none")
        rows.append({
            "series_id": s.get("series_id"),
            "concentration_uM": float(m),
            "t50": float(t50),
            "censoring_class": cens,
            "right_censored": cens in ("right", "left_right"),
            "left_censored": cens in ("left",),       # left-only -> excluded from γ
            "model": name,
            "x": [float(v) for v in (s.get("x_hours") or [])],
            "y": [float(v) for v in (s.get("m1", {}).get("y_processed")
                                     or s.get("y_intensity") or [])],
        })
    return rows


# --------------------------------------------------------------------------- #
#  γ estimator 1 — censored log–log regression of t50 on driving force
# --------------------------------------------------------------------------- #
def _censored_loglog_fit(logm, logt, right_cens):
    """Maximum-likelihood line  logt = a + b·logm  with right-censored responses
    (true logt >= observed bound) contributing P(Y >= bound). γ = −b.

    With nominal (set-point) literature concentrations the x-error is negligible,
    so the errors-in-variables term reduces to this censored regression; an x-error
    term is a documented refinement when concentration uncertainty is supplied."""
    logm = np.asarray(logm, float)
    logt = np.asarray(logt, float)
    rc = np.asarray(right_cens, bool)
    obs = ~rc
    # OLS init on the uncensored points (fall back to all if none uncensored)
    base = logm[obs] if obs.sum() >= 2 else logm
    baset = logt[obs] if obs.sum() >= 2 else logt
    A = np.vstack([np.ones_like(base), base]).T
    (a0, b0), *_ = np.linalg.lstsq(A, baset, rcond=None)
    resid0 = baset - (a0 + b0 * base)
    s0 = max(float(np.std(resid0)), 1e-3)

    # No censoring -> the censored likelihood reduces to OLS (closed form; fast path)
    if not rc.any():
        return {"a": float(a0), "b": float(b0), "sigma": s0,
                "gamma": float(-b0), "ols_gamma": float(-b0)}

    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def nll(theta):
        a, b, ls = theta
        ls = min(max(ls, -20.0), 20.0)               # keep sigma in (e^-20, e^20)
        s = math.exp(ls)
        mu = a + b * logm
        z = (logt - mu) / s
        # uncensored: log normal pdf (note log(s) == ls, used directly to avoid
        #   math.log of an underflowed sigma); right-censored: log survival P(Y>=bound)
        ll = 0.0
        if obs.any():
            no = int(obs.sum())
            ll += (-0.5 * float(np.sum(z[obs] ** 2)) - no * ls
                   - 0.5 * no * math.log(2 * math.pi))
        if rc.any():
            # log( 1 - Phi(z) ) via erfc for numerical stability
            sf = 0.5 * np.array([math.erfc(zz * inv_sqrt2) for zz in z[rc]])
            sf = np.clip(sf, 1e-12, 1.0)
            ll += float(np.sum(np.log(sf)))
        return -ll if math.isfinite(ll) else 1e18

    res = minimize(nll, [a0, b0, math.log(s0)], method="Nelder-Mead",
                   options={"maxiter": 1500, "xatol": 1e-5, "fatol": 1e-5})
    a, b, ls = res.x
    ls = min(max(float(ls), -20.0), 20.0)            # clamp before exp (no overflow)
    return {"a": float(a), "b": float(b), "sigma": float(math.exp(ls)),
            "gamma": float(-b), "ols_gamma": float(-b0)}


def _curvature_test(logm, logt, obs_mask):
    """Test for significant log–log curvature on the uncensored points (a quadratic
    coefficient ≠ 0). Significant curvature ⇒ concentration-dependent reaction order
    (saturating secondary nucleation / approach to m_crit / mechanism change)."""
    lm, lt = np.asarray(logm)[obs_mask], np.asarray(logt)[obs_mask]
    nq = len(lm)
    if len(set(np.round(lm, 6))) < 4 or nq < 4:
        return {"testable": False, "reason": "needs >=4 distinct uncensored concentrations"}
    X = np.vstack([np.ones(nq), lm, lm ** 2]).T
    beta, *_ = np.linalg.lstsq(X, lt, rcond=None)
    resid = lt - X @ beta
    dof = nq - 3
    if dof <= 0:
        return {"testable": False, "reason": "insufficient degrees of freedom"}
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se_c = math.sqrt(max(cov[2, 2], 1e-30))
    tstat = beta[2] / se_c if se_c > 0 else 0.0
    # two-sided p; t-dist approximated by a normal for these small tables (flagged)
    p = math.erfc(abs(tstat) / math.sqrt(2.0))
    return {"testable": True, "curvature_coef": float(beta[2]),
            "t_stat": float(tstat), "p_value": float(min(p, 1.0)),
            "significant": bool(p < 0.05), "dof": dof,
            "interpretation": ("concentration-dependent reaction order — route to "
                               "global fit (sat. sec. nucleation / m_crit / transition)"
                               if p < 0.05 else
                               "no curvature detectable at this design's power")}


def _informative_censoring_test(logm, right_cens, n_perm=2000, seed=0):
    """Is right-censoring correlated with concentration? If so, censoring is
    informative and the censored-regression γ is biased — block the γ call (§S3)."""
    lm = np.asarray(logm, float)
    rc = np.asarray(right_cens, float)
    if rc.sum() < 2 or (len(rc) - rc.sum()) < 2:
        return {"testable": False, "reason": "need >=2 censored and >=2 uncensored",
                "informative": False}
    if np.std(lm) < 1e-12 or np.std(rc) < 1e-12:
        return {"testable": False, "reason": "no variation", "informative": False}
    obs_corr = float(np.corrcoef(lm, rc)[0, 1])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        if abs(float(np.corrcoef(lm, rng.permutation(rc))[0, 1])) >= abs(obs_corr):
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    return {"testable": True, "correlation": obs_corr, "p_value": float(p),
            "informative": bool(p < 0.05),
            "interpretation": ("censoring correlates with concentration — γ call "
                               "blocked (informative censoring)" if p < 0.05 else
                               "censoring not detectably concentration-dependent")}


def _mcrit_diagnostic(m, logt, right_cens):
    """Driving-force / m_crit handling (§K4). Profile m_crit over (0, min(m)); is the
    supersaturation fit meaningfully better AND m_crit constrained away from 0? If
    not, the correction is cosmetic → keep nominal m, and SAY SO (flagged downgrade,
    never silent)."""
    m = np.asarray(m, float)
    logt = np.asarray(logt, float)
    rc = np.asarray(right_cens, bool)
    obs = ~rc
    mmin = float(m.min())
    if obs.sum() < 4 or mmin <= 0:
        return {"driving_force_used": "nominal_concentration",
                "m_crit_constrained": False,
                "reason": "too few uncensored concentrations to identify m_crit"}

    # Identifiability profile: OLS of log t50 on log(m - m_crit) over the UNCENSORED
    # points (the diagnostic objective is the uncensored SSE — a fast closed form).
    mo, lto = m[obs], logt[obs]

    def ols_sse(x):
        X = np.vstack([np.ones_like(x), x]).T
        beta, *_ = np.linalg.lstsq(X, lto, rcond=None)
        r = lto - X @ beta
        return float(r @ r), float(-beta[1])         # sse, gamma=-slope

    sse_nom, _ = ols_sse(np.log10(mo))
    best = None
    for mc in np.linspace(0.0, 0.95 * mmin, 12):
        df = mo - mc
        if np.any(df <= 0):
            continue
        sse, gamma_mc = ols_sse(np.log10(df))
        if best is None or sse < best[0]:
            best = (sse, float(mc), gamma_mc)
    if best is None:
        return {"driving_force_used": "nominal_concentration",
                "m_crit_constrained": False,
                "reason": "no admissible m_crit (range does not approach solubility)"}
    sse_mc, mc_hat, gamma_mc = best
    improved = sse_mc < 0.9 * sse_nom            # needs a real SSE improvement
    constrained = improved and mc_hat > 0.05 * mmin
    return {
        "driving_force_used": ("supersaturation(m - m_crit)" if constrained
                               else "nominal_concentration"),
        "m_crit_constrained": bool(constrained),
        "m_crit_estimate_uM": (float(mc_hat) if constrained else None),
        "gamma_supersaturation": (float(gamma_mc) if constrained else None),
        "note": ("m_crit identifiable from the measured range" if constrained else
                 "range does not approach m_crit — supersaturation correction is "
                 "cosmetic; nominal m used (flagged validity downgrade, not silent)"),
    }


def gamma_regression(rows: list[dict], B: int = 300, seed: int = 0) -> dict:
    """γ from the censored log–log regression of t50 on concentration, with the
    curvature, informative-censoring and m_crit diagnostics."""
    usable = [r for r in rows if not r["left_censored"]]   # left-only excluded (biased)
    n_excluded = len(rows) - len(usable)
    distinct = sorted({round(r["concentration_uM"], 6) for r in usable})
    if len(usable) < 3 or len(distinct) < 3:
        return {"status": "insufficient_data", "n_usable": len(usable),
                "n_distinct_concentrations": len(distinct),
                "n_excluded_left_censored": n_excluded}

    m = np.array([r["concentration_uM"] for r in usable])
    logm = np.log10(m)
    logt = np.log10(np.array([r["t50"] for r in usable]))
    rc = np.array([r["right_censored"] for r in usable])

    fit = _censored_loglog_fit(logm, logt, rc)
    info_cens = _informative_censoring_test(logm, rc, seed=seed)
    curv = _curvature_test(logm, logt, ~rc)
    mcrit = _mcrit_diagnostic(m, logt, rc)
    n_uncens = int((~rc).sum())

    # Point estimate: if the censored ML diverged (unphysical |γ|), fall back to the
    # uncensored OLS slope where it is itself sane; otherwise report γ as None (a
    # diverged number is never emitted as if it were an estimate) and flag.
    gamma = fit["gamma"]
    gamma_raw = gamma if math.isfinite(gamma) else None     # transparency: raw slope
    gamma_diverged = not math.isfinite(gamma) or abs(gamma) > GAMMA_PLAUSIBLE
    if gamma_diverged:
        og = fit["ols_gamma"]
        gamma = og if (n_uncens >= 2 and math.isfinite(og)
                       and abs(og) <= GAMMA_PLAUSIBLE) else None
        if gamma is not None:
            gamma_raw = gamma

    # NEGATIVE-γ DRIVING-FORCE GATE (§K4/§S3). t50 ∝ m^(−γ): a NEGATIVE γ means t50
    # RISES with concentration — non-physical for nucleation-driven aggregation
    # (more monomer should not slow the half-time). It is the signature of the wrong
    # driving force (uncorrected m_crit / supersaturation), informative censoring, or
    # an unconstrained series — NOT a clean reaction-order estimate. We keep the raw
    # number under `gamma_raw` for transparency but do NOT emit it as gamma, and mark
    # the series non-physical + unreliable rather than reporting a tidy negative γ.
    gamma_physical = True
    nonphysical_reason = None
    if gamma is not None and gamma < 0:
        gamma_physical = False
        nonphysical_reason = ("non-physical (γ<0): t50 rises with concentration — "
                              "wrong driving force / informative censoring / m_crit "
                              "unconstrained; not a clean reaction-order estimate")

    # case-resampling bootstrap CI, keeping only physically-plausible draws (a
    # diverged resample is uninformative about the interval, not a fat tail)
    gammas, rejected = [], 0
    rng = np.random.default_rng(seed + 1)
    idx = np.arange(len(usable))
    for _ in range(B):
        bi = rng.choice(idx, size=len(idx), replace=True)
        if len(set(np.round(logm[bi], 6))) < 2:
            continue
        try:
            g = _censored_loglog_fit(logm[bi], logt[bi], rc[bi])["gamma"]
        except Exception:
            continue
        if math.isfinite(g) and abs(g) <= GAMMA_PLAUSIBLE:
            gammas.append(g)
        else:
            rejected += 1
    ci = None
    if len(gammas) >= 20:
        a = np.asarray(gammas, float)
        ci = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    # reliability gate (reported, not enforced by deletion)
    reasons = []
    if n_uncens < MIN_UNCENSORED_FOR_GAMMA:
        reasons.append(f"only {n_uncens} uncensored t50 anchors "
                       f"(<{MIN_UNCENSORED_FOR_GAMMA}); γ dominated by inequalities")
    if gamma_diverged:
        reasons.append("censored ML diverged (unphysical |γ|) — used uncensored OLS slope")
    if not gamma_physical:
        # a γ<0 series must not pass the reliability gate (it is not a clean estimate)
        reasons.append(nonphysical_reason)
    if ci is None:
        reasons.append("too few admissible bootstrap draws for a CI")
    elif (ci[1] - ci[0]) > 4.0:
        reasons.append("bootstrap CI width > 4 — γ poorly constrained")
    reliable = not reasons

    blocked = info_cens.get("informative", False)
    return {
        "status": "ok",
        "estimator": "censored_loglog_regression",
        "gamma": (float(gamma) if gamma is not None else None),
        "gamma_raw": (float(gamma_raw) if gamma_raw is not None else None),
        "gamma_physical": bool(gamma_physical),
        "gamma_physical_reason": nonphysical_reason,
        "gamma_ci": ci,
        "gamma_reliable": bool(reliable),
        "reliability_reasons": reasons,
        "ols_gamma_uncensored": fit["ols_gamma"],
        "intercept": fit["a"], "residual_sigma": fit["sigma"],
        "n_usable": len(usable), "n_uncensored": n_uncens,
        "n_distinct_concentrations": len(distinct),
        "n_right_censored": int(rc.sum()),
        "n_excluded_left_censored": n_excluded,
        "n_bootstrap_rejected_divergent": rejected,
        "driving_force": mcrit,
        "curvature_test": curv,
        "informative_censoring_test": info_cens,
        "blocked_for_mechanism": blocked,
        "blocked_reason": ("informative censoring" if blocked else None),
    }


# --------------------------------------------------------------------------- #
#  γ estimator 2 — global master-curve collapse
# --------------------------------------------------------------------------- #
def _rough_t50(x, y):
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if ymax - ymin < _EPS:
        return None
    target = ymin + 0.5 * (ymax - ymin)
    c = _first_crossing(np.asarray(x, float), np.asarray(y, float), target)
    return c if (c is not None and c > 0) else float(np.median(x))


def gamma_global(members: list[dict], B: int = 0, seed: int = 0) -> dict:
    """γ from a global fit that collapses every member curve onto one master
    profile in rescaled time τ = t / t50(m), with t50(m) = A·m^(−γ):

        ŷ_ij = base + amp_i / (1 + exp(−k·(t_ij / t50_i − 1)))

    Shared across curves: {base, k (shape), A, γ}; per-curve: amp_i. Because it
    uses the whole curve shape (and a free per-curve amplitude), it tolerates
    per-curve right-censoring that a t50 point-estimate cannot — and it is
    genuinely different machinery from the t50 regression, so disagreement is
    informative rather than redundant."""
    data = []
    for s in members:
        cv = s.get("condition_vector", {}) or {}
        m = (cv.get("concentration") or {}).get("value_uM")
        x = s.get("x_hours") or []
        y = s.get("m1", {}).get("y_processed") or s.get("y_intensity") or []
        if m and m > 0 and len(x) == len(y) and len(x) >= 4:
            data.append((float(m), np.asarray(x, float), np.asarray(y, float)))
    distinct = sorted({round(m, 6) for m, _, _ in data})
    if len(data) < 3 or len(distinct) < 3:
        return {"status": "insufficient_data", "n_curves": len(data),
                "n_distinct_concentrations": len(distinct)}

    N = len(data)
    base0 = min(float(y.min()) for _, _, y in data)
    amp0 = [max(float(y.max() - y.min()), 1e-6) for _, _, y in data]
    t50r = [(_rough_t50(x, y) or float(np.median(x))) for _, x, y in data]
    # initial γ from a quick OLS of log t50_rough vs log m
    lm = np.log10([m for m, _, _ in data])
    lt = np.log10(np.clip(t50r, 1e-6, None))
    A_ = np.vstack([np.ones_like(lm), lm]).T
    (a_i, b_i), *_ = np.linalg.lstsq(A_, lt, rcond=None)
    g0 = float(-b_i)
    logA0 = float(a_i)

    def t50_of_m(m, logA, g):
        return max((10.0 ** logA) * m ** (-g), 1e-6)

    def residuals(p):
        base, k, logA, g = p[0], p[1], p[2], p[3]
        amps = p[4:]
        out = []
        for j, (m, x, y) in enumerate(data):
            t50j = t50_of_m(m, logA, g)
            z = np.clip(k * (x / t50j - 1.0), -50.0, 50.0)
            yhat = base + amps[j] / (1.0 + np.exp(-z))
            out.append(yhat - y)
        return np.concatenate(out)

    p0 = [base0, 5.0, logA0, g0] + amp0
    lo = [-np.inf, 1e-3, -4.0, -5.0] + [0.0] * N
    hi = [np.inf, 1e3, 9.0, 5.0] + [3.0 * a for a in amp0]
    p0 = [min(max(v, lo[i]), hi[i]) for i, v in enumerate(p0)]
    try:
        sol = least_squares(residuals, p0, bounds=(lo, hi), max_nfev=4000)
    except Exception as e:                         # pragma: no cover
        return {"status": "fit_failed", "reason": str(e)}

    gamma = float(sol.x[3])
    r = sol.fun
    n_pts = len(r)
    sse = float(r @ r)
    sst = float(np.sum([np.sum((y - y.mean()) ** 2) for _, _, y in data])) or 1e-12

    ci = None
    # each global refit costs O(4+N params); scale the bootstrap budget down for
    # large series so the corpus batch stays tractable (CI still uses >=20 draws)
    B_eff = B if N <= 20 else (max(20, B // 2) if N <= 40 else max(20, B // 4))
    if B > 0:
        # wild-bootstrap the global residuals, refit, collect γ
        rng = np.random.default_rng(seed)
        yhat_all = [base_amp_predict(sol.x, data, j) for j in range(N)]
        gammas = []
        offs = np.cumsum([0] + [len(x) for _, x, _ in data])
        for _ in range(B_eff):
            w = _mammen(n_pts, rng)
            pert = []
            for j, (m, x, y) in enumerate(data):
                seg = r[offs[j]:offs[j + 1]]
                pert.append((m, x, yhat_all[j] + seg * w[offs[j]:offs[j + 1]]))

            def resid_b(p, _d=pert):
                base, k, logA, g = p[0], p[1], p[2], p[3]
                amps = p[4:]
                o = []
                for j, (m, x, y) in enumerate(_d):
                    t50j = t50_of_m(m, logA, g)
                    z = np.clip(k * (x / t50j - 1.0), -50.0, 50.0)
                    o.append(base + amps[j] / (1.0 + np.exp(-z)) - y)
                return np.concatenate(o)
            try:
                sb = least_squares(resid_b, sol.x, bounds=(lo, hi), max_nfev=2000)
                gammas.append(float(sb.x[3]))
            except Exception:
                continue
        if len(gammas) >= 20:
            a = np.asarray(gammas, float)
            ci = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    collapse_r2 = 1.0 - sse / sst
    return {
        "status": "ok",
        "estimator": "global_master_curve_collapse",
        # HONESTY RELABEL (consumers m6/m7/web read the key `gamma_global`, so the
        # key is preserved): this is a PHENOMENOLOGICAL master-curve COLLAPSE that
        # shares only a logistic SHAPE + one γ across curves — it does NOT share
        # microscopic Knowles/Cohen rate constants and never touches the ODE family.
        # The real shared-rate ODE global fit lives in engine/global_fit.py.
        "method": "master_curve_collapse",
        "is_shared_rate_ode_fit": False,
        "gamma": gamma,
        "gamma_ci": ci,
        "shared_k_shape": float(sol.x[1]),
        "intercept_log10A": float(sol.x[2]),
        "n_curves": N, "n_distinct_concentrations": len(distinct),
        "collapse_r2": collapse_r2,
        # a poor collapse (<~0.8) means the SHAPE is concentration-dependent — itself
        # a signal, surfaced to the disagreement test
        "collapse_poor": bool(collapse_r2 < 0.8),
    }


def base_amp_predict(p, data, j):
    base, k, logA, g = p[0], p[1], p[2], p[3]
    amp = p[4 + j]
    m, x, _ = data[j]
    t50j = max((10.0 ** logA) * m ** (-g), 1e-6)
    z = np.clip(k * (x / t50j - 1.0), -50.0, 50.0)
    return base + amp / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------- #
#  Dual-γ assembler + disagreement test
# --------------------------------------------------------------------------- #
# pre-registered pointwise disagreement margin (|γ_reg − γ_collapse| beyond this is
# flagged even when bootstrap CIs are absent — the loud disagreements were being
# DROPPED when a CI was missing or γ was unreliable, suppressing the real signal)
DISAGREE_POINTWISE_MARGIN = 0.3


def _disagreement(reg: dict, glob: dict, margin: float = 0.0) -> dict:
    """Operational disagreement test — fixed so the LOUDEST disagreements are no
    longer suppressed:

      * ALWAYS report the raw point-difference |γ_reg − γ_collapse| and a
        `disagree_pointwise` flag (|diff| > DISAGREE_POINTWISE_MARGIN), even when a
        bootstrap CI is absent or the regression γ is flagged unreliable. (Previously
        the test returned testable:False in exactly those cases, dropping the signal.)
      * COLLAPSE-QUALITY gate: a master-curve collapse with collapse_r2 < 0.8 means
        the shape does NOT collapse → it is concentration-dependent → that IS the
        signal: flag `shape_not_concentration_invariant: true`.
      * Keep the CI-overlap test when BOTH estimators have CIs (the strongest form).

    Non-overlap / large pointwise gap / poor collapse all ⇒ the curve shape is not
    concentration-invariant — reported as signal, never silently merged."""
    gr, gg = reg.get("gamma"), glob.get("gamma")
    cr, cg = reg.get("gamma_ci"), glob.get("gamma_ci")
    diff = abs(gr - gg) if (gr is not None and gg is not None) else None
    collapse_r2 = glob.get("collapse_r2")
    collapse_poor = bool(glob.get("collapse_poor")) or (
        collapse_r2 is not None and collapse_r2 < 0.8)

    # pointwise disagreement — ALWAYS computed when both point estimates exist
    disagree_pointwise = bool(diff is not None and diff > DISAGREE_POINTWISE_MARGIN)

    base = {
        "gamma_regression": gr, "gamma_global": gg,
        "difference": diff,
        "pointwise_margin": DISAGREE_POINTWISE_MARGIN,
        "disagree_pointwise": disagree_pointwise,
        "regression_gamma_reliable": bool(reg.get("gamma_reliable", True)),
        "regression_gamma_physical": bool(reg.get("gamma_physical", True)),
        "collapse_r2": collapse_r2,
        "shape_not_concentration_invariant": bool(collapse_poor),
    }

    # CI-overlap test — only when BOTH CIs exist (the strongest evidence)
    ci_overlap = None
    if cr and cg:
        ci_overlap = bool((cr[0] - margin) <= cg[1] and (cg[0] - margin) <= cr[1])
    base["ci_overlap"] = ci_overlap
    base["ci_test_available"] = ci_overlap is not None

    # overall `disagree`: the CI test where available, else the pointwise gap; the
    # collapse-quality flag is an independent shape signal and also raises disagree
    if ci_overlap is not None:
        disagree = (not ci_overlap) or collapse_poor
    elif diff is not None:
        disagree = disagree_pointwise or collapse_poor
    else:
        disagree = collapse_poor

    # `testable` now means "we could form ANY comparison" (point or CI or collapse) —
    # it is True far more often than before, by design (no more silent suppression)
    testable = (diff is not None) or (ci_overlap is not None) or (collapse_r2 is not None)
    if not testable:
        base.update({"testable": False, "disagree": False,
                     "reason": "neither γ available and no collapse R² — nothing to compare"})
        return base

    reasons = []
    if ci_overlap is False:
        reasons.append("bootstrap CIs do not overlap")
    if disagree_pointwise:
        reasons.append(f"|γ_reg − γ_collapse| = {diff:.2f} > {DISAGREE_POINTWISE_MARGIN}")
    if collapse_poor:
        reasons.append(f"master curve does not collapse (R²={collapse_r2:.2f}<0.8) — "
                       "shape is concentration-dependent")
    if not reg.get("gamma_reliable", True):
        reasons.append("regression γ flagged unreliable (difference still reported, "
                       "not suppressed)")

    base.update({
        "testable": True,
        "disagree": bool(disagree),
        "disagreement_reasons": reasons,
        "interpretation": ("γ estimators / curve shape disagree — shape is NOT "
                           "concentration-invariant (possible mechanism transition / "
                           "saturation / wrong driving force)" if disagree else
                           "γ estimators consistent and master curve collapses"),
    })
    return base


def _global_ode_summary(global_fit_by_csid: dict | None, csid) -> dict | None:
    """Read-only join: if global_fit.json carries a shared-rate ODE fit for this
    series, summarise its third γ_mechanistic + identifiability for the dual-γ
    payload. NEVER recomputes the (slow) ODE fit in the hot path — pure lookup."""
    if not global_fit_by_csid or csid is None:
        return None
    gf = global_fit_by_csid.get(csid)
    if not gf or gf.get("status") != "ok":
        return None
    ident = gf.get("identifiability", {}) or {}
    return {
        "source": "engine/global_fit.py (shared-rate Knowles/Cohen ODE fit)",
        "version": gf.get("version"),
        "best_mechanism": gf.get("best_mechanism"),
        "reaction_orders": gf.get("reaction_orders"),
        "gamma_mechanistic": gf.get("gamma_mechanistic"),
        "gamma_mechanistic_formula": gf.get("gamma_mechanistic_formula"),
        "mechanism_aicc_ranking": gf.get("mechanism_aicc_ranking"),
        "sloppy": gf.get("sloppy"),
        "fim_condition_number": ident.get("fim_condition_number"),
        "identifiable_combinations": ident.get("identifiable_combinations"),
        "individual_rate_status": ident.get("individual_rate_status"),
        "identifiability_note": ident.get("interpretation"),
        "honesty_note": gf.get("honesty_note"),
    }


def dual_gamma(series_meta: dict, members: list[dict],
               B_reg: int = 300, B_glob: int = 80, seed: int = 0,
               global_fit_by_csid: dict | None = None) -> dict:
    """Full dual-γ payload for one concentration series.

    `global_fit_by_csid` (optional) is a read-only map {csid -> global_fit series}
    from data/processed/global_fit.json; when present, the shared-rate ODE fit's
    third γ_mechanistic + identifiability is attached as `global_ode_fit`. The slow
    ODE fit is NEVER run here — this is a pure join."""
    rows = series_t50_table(members)
    reg = gamma_regression(rows, B=B_reg, seed=seed)
    glob = gamma_global(members, B=B_glob, seed=seed)
    csid = series_meta.get("concentration_series_id")
    out = {
        "concentration_series_id": csid,
        "protein": series_meta.get("protein"),
        "conditions": {k: series_meta.get(k) for k in
                       ("pH", "temperature_C", "assay", "mutation")},
        "definition_contract": DEFINITION_CONTRACT["version"],
        "gamma_provenance": "gamma_regression (censored t50 log–log) and gamma_global "
                            "(master-curve COLLAPSE — phenomenological, not a shared-"
                            "rate ODE fit) are reported separately and never merged "
                            "(§3-M4); the shared-rate Knowles/Cohen ODE global fit "
                            "(third γ_mechanistic) lives in engine/global_fit.py",
        "n_member_curves": len(members),
        "n_curves_with_t50": len(rows),
        "gamma_regression": reg,
        "gamma_global": glob,
        # ---- §2.3 FitProvenance for the two gamma bootstraps (§10.12) --------
        # THIS BLOCK IS LOAD-BEARING, not bookkeeping. `gamma_ci` below is
        # produced by a random resampling, and m7_assemble.gamma_significant()
        # requires that CI to EXCLUDE ZERO before a curve earns the `scaling`
        # tier of the information_yield ladder. So the seed and batch sizes
        # recorded here are inputs to a PUBLISHED COUNT in the headline
        # inference ladder. Before this record existed they lived only as
        # hard-coded defaults in source plus a --bootstrap value in the README,
        # which meant a shipped gamma_ci could not be reproduced from the
        # artifact alone.
        # accepted_B is deliberately NOT supplied: the gamma bootstraps reject
        # divergent draws and skip degenerate resamples without returning the
        # surviving count, so any number here would be invented. The rejected
        # count that IS measured lives in gamma_regression.
        "bootstrap_provenance": fp.bootstrap_provenance(
            M4_GAMMA_RECIPE, seed,
            requested_B=B_reg,
            version=DEFINITION_CONTRACT["version"],
            requested_B_gamma_global=int(B_glob),
            censoring_permutations=2000,
            accepted_B_note=("not tracked by gamma_regression/gamma_global; the "
                             "measured rejection count is "
                             "gamma_regression.n_bootstrap_rejected_divergent"),
            gates_published_count=("m7_assemble.gamma_significant() -> "
                                   "information_yield `scaling` tier")),
    }
    out["disagreement"] = _disagreement(reg, glob)
    # optional enrichment: the shared-rate ODE fit's third γ + identifiability
    ode = _global_ode_summary(global_fit_by_csid, csid)
    if ode is not None:
        out["global_ode_fit"] = ode
    # γ is a constraint, not a mechanism verdict (M5 decides, shape-gated)
    out["mechanism_note"] = ("γ constrains a combination of reaction orders "
                             "(many-to-one onto mechanism); the mechanistic call is "
                             "M5, gated on curve shape with γ as a consistency check")
    return out


# --------------------------------------------------------------------------- #
#  I/O
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


def _load_global_fit(root: Path) -> dict | None:
    """Read data/processed/global_fit.json (shared-rate ODE fits) into a
    {csid -> series} map for the read-only dual-γ enrichment join. Absent/invalid
    -> None (no enrichment). Never raises."""
    gp = root / "global_fit.json"
    if not gp.exists():
        return None
    try:
        payload = json.loads(gp.read_text(encoding="utf-8"))
    except Exception:
        return None
    out = {}
    for s in payload.get("series", []):
        csid = s.get("concentration_series_id")
        if csid:
            out[csid] = s
    return out or None


def _run_features(args):
    out_path = args.output or (args.input.parent / "features.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = ok = 0
    statuses: dict[str, int] = defaultdict(int)
    print(f"[M4] extracting features (bootstrap={args.bootstrap}) ...")
    with open(args.input, encoding="utf-8") as fh, open(out_path, "w", encoding="utf-8") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("m1", {}).get("fittability_class") != "fittable":
                continue
            res = extract_features(s, B=args.bootstrap)
            statuses[res.get("status", "?")] += 1
            n += 1
            ok += (res.get("status") == "ok")
            out.write(json.dumps(res) + "\n")
            if args.limit and n >= args.limit:
                break
    print(f"[M4] wrote {out_path}  ({n} curves, {ok} with full feature vectors)")
    print(json.dumps({"n": n, "status_breakdown": dict(statuses)}, indent=2))


def _run_gamma(args):
    root = args.input.parent
    cs_path = root / "concentration_series.json"
    if not cs_path.exists():
        raise SystemExit(f"{cs_path} not found (run the ETL first)")
    series = json.loads(cs_path.read_text(encoding="utf-8"))
    by_id = _load_triaged(args.input)
    out_path = args.output or (root / "gamma.jsonl")
    # optional read-only join: the shared-rate ODE fit (global_fit.json) — present
    # only if engine/global_fit.py was run; absent -> no enrichment, no recompute
    global_fit_by_csid = _load_global_fit(root)
    n = ok = reliable = disagree = curv = blocked = nonphysical = 0
    print(f"[M4] dual-gamma over {len(series)} concentration series ...")
    with open(out_path, "w", encoding="utf-8") as out:
        for meta in series:
            members = [by_id[sid] for sid in meta.get("member_series_ids", [])
                       if sid in by_id]
            if len(members) < 3:
                continue
            res = dual_gamma(meta, members, B_reg=args.bootstrap,
                             B_glob=min(args.bootstrap, 80),
                             global_fit_by_csid=global_fit_by_csid)
            n += 1
            reg = res["gamma_regression"]
            if reg.get("status") == "ok":
                ok += 1
                reliable += bool(reg.get("gamma_reliable"))
                blocked += bool(reg.get("blocked_for_mechanism"))
                curv += bool(reg.get("curvature_test", {}).get("significant"))
                nonphysical += bool(not reg.get("gamma_physical", True))
            disagree += bool(res["disagreement"].get("disagree"))
            out.write(json.dumps(res) + "\n")
            if args.limit and n >= args.limit:
                break
    print(f"[M4] wrote {out_path}  ({n} series)")
    print(json.dumps({"n_series": n, "with_regression_gamma": ok,
                      "reliable_regression_gamma": reliable,
                      "gamma_estimators_disagree": disagree,
                      "gamma_nonphysical_negative_gated": nonphysical,
                      "curvature_significant": curv,
                      "blocked_informative_censoring": blocked}, indent=2))


def _run_t50_points(args):
    """Precompute the per-concentration-series (concentration, t50) table.

    WHY THIS EXISTS: /api/series-gamma built this table live on every request by
    calling series_t50_table(), which refits every member curve. Measured on the
    serving box that is 1-30 s per protein (AL-12: 30.5 s), and the deployment
    runs on 0.1 CPU, so the browser waited minutes. The rows are deterministic
    given frozen curves, so they are computed once here instead.

    We store the RAW series_t50_table() rows, not a reshaped view: the web layer
    keeps its own shaping, so the published artifact stays a faithful record of
    what the engine computed. Note the per-curve M4 t50 in protein_analysis.jsonl
    is NOT a substitute -- it is None for curves this series-level table still
    estimates (e.g. CPAD-TK-1489), so reusing it would silently drop points from
    the log-log scatter.
    """
    root = Path(__file__).resolve().parent.parent / "data" / "processed"
    cs_path = root / "concentration_series.json"
    if not cs_path.exists():
        raise SystemExit(f"{cs_path} not found (run the ETL first)")
    series = json.loads(cs_path.read_text(encoding="utf-8"))
    by_id = _load_triaged(args.input)
    out_path = args.output or (root / "series_t50_points.jsonl")

    n = n_rows = skipped = 0
    print(f"[M4] t50 points over {len(series)} concentration series ...")
    with open(out_path, "w", encoding="utf-8") as out:
        for meta in series:
            csid = meta.get("concentration_series_id")
            members = [by_id[sid] for sid in meta.get("member_series_ids", [])
                       if sid in by_id]
            if not csid or not members:
                skipped += 1
                continue
            try:
                rows = series_t50_table(members)
            except Exception as exc:
                skipped += 1
                print(f"[M4]   WARN {csid}: {type(exc).__name__}: {exc}")
                continue
            out.write(json.dumps({
                "concentration_series_id": csid,
                "n_members": len(members),
                "rows": rows,
            }) + "\n")
            n += 1
            n_rows += len(rows)
    print(f"[M4] wrote {out_path}  ({n} series, {n_rows} rows, {skipped} skipped)")


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="PRISE M4 features + dual-γ")
    ap.add_argument("--input", type=Path,
                    default=root / "data" / "processed" / "curves_triaged.jsonl")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="bootstrap reps (features: per-feature CIs; γ: γ CIs)")
    ap.add_argument("--gamma", action="store_true",
                    help="run dual-γ per concentration series instead of per-curve features")
    ap.add_argument("--t50-points", dest="t50_points", action="store_true",
                    help="precompute the per-series (concentration, t50) table "
                         "that /api/series-gamma would otherwise refit per request")
    args = ap.parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input} (run M1 first)")
    if args.t50_points:
        _run_t50_points(args)
    elif args.gamma:
        if not args.bootstrap:
            args.bootstrap = 300
        _run_gamma(args)
    else:
        _run_features(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
