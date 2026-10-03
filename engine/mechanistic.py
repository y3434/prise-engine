"""
PRISE — Tier B: Knowles/Cohen mechanistic kinetics (ODE framework)
==================================================================

The amyloid master equations reduce, for the principal moments, to a 2-ODE
system in the fibril number P(t) and fibril mass M(t):

    dP/dt = k_n m^{n_c}                      (primary nucleation)
          + k_2 m^{n_2} / (1 + (m^{n_2}/K_M)) · M   (secondary nucleation, saturating)
          + k_-  · M                          (fragmentation)
    dM/dt = 2 k_+ m P                         (elongation)
    m(t)  = m_tot − M(t)                       (mass conservation)

One framework, several mechanisms by toggling terms:
    nucleation_elongation   k_2 = 0, k_- = 0
    secondary_nucleation    k_2 > 0  (K_M = ∞)
    saturating_secondary    k_2 > 0, finite K_M
    fragmentation           k_- > 0, k_2 = 0

Integrating numerically (scipy LSODA) is *exact* and avoids transcribing the
notoriously error-prone closed-form solutions. The trade-off is speed, so
mechanistic fitting is applied to the concentration-series subset (and on
demand), not to every curve.

Reaction orders n_c, n_2 are fixed by default (2.0) — they are weakly identifiable
from a single curve; M3/M4 (global concentration-series fits + the γ scaling
exponent) is where reaction orders are actually constrained.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

VARIANT_RATES = {
    "nucleation_elongation": ["kn", "kp"],
    "secondary_nucleation": ["kn", "kp", "k2"],
    "saturating_secondary": ["kn", "kp", "k2", "KM"],
    "fragmentation": ["kn", "kp", "kminus"],
}


def simulate_mass_fraction(t, kn, kp, k2=0.0, kminus=0.0, KM=math.inf,
                           nc=2.0, n2=2.0, mtot=1.0):
    """Return M(t)/m_tot ∈ [0,1] for the mechanistic ODE."""
    t = np.asarray(t, float)
    tmax = float(t.max()) if t.size else 0.0
    if tmax <= 0:
        return np.zeros_like(t)

    def rhs(_t, s):
        P, M = s
        m = mtot - M
        if m < 0.0:
            m = 0.0
        prim = kn * (m ** nc) if m > 0 else 0.0
        sec = 0.0
        if k2 > 0 and m > 0:
            mn2 = m ** n2
            sat = 1.0 + mn2 / KM if math.isfinite(KM) else 1.0
            sec = k2 * mn2 / sat * M
        frag = kminus * M
        return [prim + sec + frag, 2.0 * kp * m * P]

    sol = solve_ivp(rhs, (0.0, tmax), [0.0, 0.0], t_eval=t, method="LSODA",
                    rtol=1e-6, atol=1e-9, max_step=max(tmax / 25.0, 1e-6))
    if not sol.success:
        return np.full_like(t, np.nan)
    return np.clip(sol.y[1], 0.0, mtot) / mtot


def _metrics(y, yhat, p):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    n = len(y)
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    sigma2 = max(sse / n, 1e-12)
    loglik = -0.5 * n * (math.log(2 * math.pi) + math.log(sigma2) + 1.0)
    aic = 2 * p - 2 * loglik
    aicc = aic + (2 * p * (p + 1) / (n - p - 1)) if (n - p - 1) > 0 else float("inf")
    return {"sse": sse, "r2": 1 - sse / sst, "aic": aic, "aicc": aicc,
            "bic": p * math.log(n) - 2 * loglik}


def _model_factory(variant, nc, n2):
    rates = VARIANT_RATES[variant]

    def f(t, base, amp, *log_rates):
        kw = {name: 10.0 ** lv for name, lv in zip(rates, log_rates)}
        KM = kw.pop("KM", math.inf)   # popped value is already 10**log10(KM)
        frac = simulate_mass_fraction(
            t, kn=kw.get("kn", 0.0), kp=kw.get("kp", 0.0),
            k2=kw.get("k2", 0.0), kminus=kw.get("kminus", 0.0),
            KM=KM, nc=nc, n2=n2, mtot=1.0)
        return base + amp * frac
    return f, rates


def fit_mechanistic(x, y, variant="secondary_nucleation", nc=2.0, n2=2.0) -> dict:
    """Fit one mechanistic variant to a single (time, signal) curve.

    Rates are fit in log10 space (the models are sloppy). Returns a JSON-able
    dict; never raises."""
    if variant not in VARIANT_RATES:
        raise ValueError(variant)
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    f, rates = _model_factory(variant, nc, n2)
    p = 2 + len(rates)
    if len(xa) <= p + 1:
        return {"variant": variant, "converged": False, "reason": "too_few_points"}

    rng = float((ya.max() - ya.min()) or 1.0)
    # guesses (log10): kn small, kp moderate, k2 small, kminus small, KM ~ 0.3
    log_guess = {"kn": -3.0, "kp": -1.0, "k2": -2.0, "kminus": -2.0, "KM": -0.5}
    log_lo = {"kn": -9, "kp": -5, "k2": -9, "kminus": -9, "KM": -3}
    log_hi = {"kn": 3, "kp": 4, "k2": 4, "kminus": 4, "KM": 1}
    p0 = [float(ya.min()), rng] + [log_guess[r] for r in rates]
    lo = [ya.min() - rng, 0.0] + [log_lo[r] for r in rates]
    hi = [ya.max() + rng, 3 * rng] + [log_hi[r] for r in rates]

    try:
        popt, pcov = curve_fit(f, xa, ya, p0=p0, bounds=(lo, hi), maxfev=4000)
    except (RuntimeError, ValueError):
        return {"variant": variant, "converged": False, "reason": "no_convergence"}
    yhat = f(xa, *popt)
    if not np.all(np.isfinite(yhat)):
        return {"variant": variant, "converged": False, "reason": "non_finite"}

    met = _metrics(ya, yhat, p)
    params = {"base": float(popt[0]), "amp": float(popt[1])}
    for name, lv in zip(rates, popt[2:]):
        params[name] = float(10.0 ** lv)  # back to linear units
    return {"variant": variant, "converged": True, "n_params": p,
            "nc": nc, "n2": n2, "params": params,
            **{k: (float(v) if math.isfinite(v) else None) for k, v in met.items()}}
