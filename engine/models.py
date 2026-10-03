"""
PRISE — model registry
======================

All closed-form models used by M2, organised by the AXIS they live on:

  KINETIC      — signal vs TIME (a single aggregation curve)
  DOSE         — response vs CONCENTRATION (across a concentration series)
  MECHANISTIC  — signal vs TIME, mechanistic closed forms (Tier B; the ODE-based
                 Knowles/Cohen family lives in mechanistic.py)

Scientific note on Brain–Cousens & LNT: these are *dose–response* models
(response as a function of dose). They are placed primarily in the DOSE bank,
and an LNT/Brain–Cousens form is also exposed on the KINETIC axis (a line / a
hormesis bump in time) per project direction.

Each registry entry is:
    name -> {func, params, guess(x,y)->[...], bounds(x,y)->(lo,hi), axis, note}
`func` takes (x, *params) and is numpy-vectorised and overflow-guarded.
"""
from __future__ import annotations

import math
import numpy as np

_CLIP = 50.0
_EPS = 1e-9


def _e(z):
    return np.exp(np.clip(z, -_CLIP, _CLIP))


# --------------------------- shape helpers --------------------------------- #
def _rng(y):
    return (max(y) - min(y)) or 1.0


def _span(x):
    return (max(x) - min(x)) or 1.0


def _max_slope(x, y):
    s = [(y[i + 1] - y[i]) / (x[i + 1] - x[i])
         for i in range(len(x) - 1) if x[i + 1] > x[i]]
    pos = sorted(v for v in s if v > 0)
    return pos[int(0.9 * (len(pos) - 1))] if pos else 0.0


def _time_at_fraction(x, y, frac):
    ymin, ymax = min(y), max(y)
    target = ymin + frac * (ymax - ymin)
    for i in range(len(y) - 1):
        if (y[i] - target) * (y[i + 1] - target) <= 0 and y[i + 1] != y[i]:
            r = (target - y[i]) / (y[i + 1] - y[i])
            return float(x[i] + r * (x[i + 1] - x[i]))
    return float(x[len(x) // 2])


# ============================ KINETIC (vs time) ============================= #
def m_lnt(t, a, b):
    """Linear, no threshold: a + b·t (a line through baseline, no threshold)."""
    return a + b * np.asarray(t, float)


def m_logistic(t, base, amp, k, t0):
    return base + amp / (1.0 + _e(-k * (np.asarray(t, float) - t0)))


def m_gompertz(t, base, amp, k, t0):
    return base + amp * _e(-_e(-k * (np.asarray(t, float) - t0)))


def m_richards(t, base, amp, k, t0, nu):
    """Generalised-logistic (Richards) sigmoid; nu→1 ≈ logistic, nu→0 ≈ Gompertz."""
    z = _e(-k * (np.asarray(t, float) - t0))
    return base + amp / np.power(1.0 + nu * z, 1.0 / max(nu, 1e-3))


def m_exponential(t, base, amp, k):
    return base + amp * (1.0 - _e(-k * np.asarray(t, float)))


def m_scaling(t, a, n, c):
    """Power-law / scaling growth: c + a·t^n (the exponent connects to scaling)."""
    tt = np.clip(np.asarray(t, float), _EPS, None)
    return c + a * np.power(tt, n)


def m_brain_cousens(t, c, d, f, b, e):
    """Brain–Cousens hormesis (biphasic): rise (the +f·x term) then logistic
    decline. Native dose model; used here as the kinetic biphasic form."""
    x = np.clip(np.asarray(t, float), _EPS, None)
    return c + (d - c + f * x) / (1.0 + _e(b * (np.log(x) - math.log(max(e, _EPS)))))


# ===================== MECHANISTIC closed form (vs time) ==================== #
def m_finke_watzky(t, base, amp, k1, k2):
    """Finke–Watzky 2-step (slow nucleation A→B, autocatalysis A+B→2B), exact
    closed form for fraction converted (derived for [A]0=1):
        f(t) = k1(e^{Kt}-1) / (k1 e^{Kt} + k2),  K = k1 + k2
    NOTE k1,k2 are FW lumped constants — NOT amyloid nucleation/elongation rates."""
    K = k1 + k2
    eKt = _e(K * np.asarray(t, float))
    f = k1 * (eKt - 1.0) / (k1 * eKt + k2)
    return base + amp * f


# ========================= DOSE (vs concentration) ========================= #
def m_dr_lnt(d, bg, slope):
    """Linear no-threshold dose–response."""
    return bg + slope * np.asarray(d, float)


def m_dr_threshold(d, bg, tau, slope):
    """Hockey-stick: flat baseline until threshold dose τ, then linear."""
    dd = np.asarray(d, float)
    return bg + slope * np.clip(dd - tau, 0.0, None)


def m_dr_hill(d, c, top, ec50, h):
    """4-parameter Hill / log-logistic sigmoidal dose–response."""
    dd = np.clip(np.asarray(d, float), _EPS, None)
    return c + (top - c) / (1.0 + np.power(ec50 / dd, h))


def m_dr_brain_cousens(d, c, top, f, b, e):
    return m_brain_cousens(d, c, top, f, b, e)


def m_dr_scaling(d, a, n, c):
    """Power-law dose scaling: c + a·dose^n  (n is the scaling exponent)."""
    dd = np.clip(np.asarray(d, float), _EPS, None)
    return c + a * np.power(dd, n)


# ------------------------------- guesses ----------------------------------- #
def _g_lnt(x, y):
    return [float(min(y)), _max_slope(x, y) or (float(y[-1] - y[0]) / _span(x))]


def _g_sigmoid(x, y, gomp=False):
    amp = _rng(y)
    k = (math.e if gomp else 4.0) * (_max_slope(x, y) or 0.1) / amp
    return [float(min(y)), float(amp), max(k, 1e-3), _time_at_fraction(x, y, 0.5)]


def _g_richards(x, y):
    base, amp, k, t0 = _g_sigmoid(x, y)
    return [base, amp, k, t0, 1.0]


def _g_exp(x, y):
    return [float(min(y)), float(_rng(y)), 0.3]


def _g_scaling(x, y):
    return [float(_rng(y)) / (_span(x) ** 1.0), 1.0, float(min(y))]


def _g_brain(x, y):
    return [float(min(y)), float(max(y)), 0.0, 2.0, _time_at_fraction(x, y, 0.5)]


def _g_fw(x, y):
    return [float(min(y)), float(_rng(y)), 0.05, 1.0]


def _g_dr_lnt(x, y):
    return [float(min(y)), float(y[-1] - y[0]) / _span(x)]


def _g_dr_threshold(x, y):
    return [float(min(y)), float(np.median(x)), float(y[-1] - y[0]) / _span(x)]


def _g_dr_hill(x, y):
    return [float(min(y)), float(max(y)), float(np.median(x)), 1.0]


def _g_dr_scaling(x, y):
    return [float(_rng(y)) / (np.median(x) ** 1.0 or 1.0), 1.0, float(min(y))]


# ------------------------------- bounds ------------------------------------ #
def _b_free(n):
    return lambda x, y: ([-np.inf] * n, [np.inf] * n)


def _b_sigmoid(x, y):
    r, sp = _rng(y), _span(x)
    return ([min(y) - r, 0.0, 1e-4, min(x) - sp],
            [max(y) + r, 3 * r, 1e3, max(x) + sp])


def _b_richards(x, y):
    lo, hi = _b_sigmoid(x, y)
    return (lo + [0.05], hi + [20.0])


def _b_exp(x, y):
    r = _rng(y)
    return ([min(y) - r, 0.0, 1e-4], [max(y) + r, 3 * r, 1e3])


def _b_scaling(x, y):
    r = _rng(y)
    return ([-100 * r, -5.0, min(y) - r], [100 * r, 5.0, max(y) + r])


def _b_brain(x, y):
    r, sp = _rng(y), _span(x)
    return ([min(y) - r, min(y) - r, -100 * r, -50.0, _EPS],
            [max(y) + r, max(y) + 2 * r, 100 * r, 50.0, max(x) + sp])


def _b_fw(x, y):
    r = _rng(y)
    return ([min(y) - r, 0.0, 1e-5, 1e-5], [max(y) + r, 3 * r, 1e2, 1e3])


def _b_dr_hill(x, y):
    r = _rng(y)
    return ([min(y) - r, min(y) - r, _EPS, 0.1],
            [max(y) + r, max(y) + r, max(x) * 10, 20.0])


def _b_dr_threshold(x, y):
    r = _rng(y)
    return ([min(y) - r, min(x), -100 * r], [max(y) + r, max(x), 100 * r])


# ------------------------------ registry ----------------------------------- #
REGISTRY = {
    # kinetic (vs time)
    "lnt":         dict(func=m_lnt, params=["a", "b"], guess=_g_lnt, bounds=_b_free(2), axis="time", note="linear no-threshold"),
    "logistic":    dict(func=m_logistic, params=["base", "amp", "k", "t0"], guess=lambda x, y: _g_sigmoid(x, y), bounds=_b_sigmoid, axis="time", note="symmetric sigmoid"),
    "gompertz":    dict(func=m_gompertz, params=["base", "amp", "k", "t0"], guess=lambda x, y: _g_sigmoid(x, y, True), bounds=_b_sigmoid, axis="time", note="asymmetric sigmoid"),
    "richards":    dict(func=m_richards, params=["base", "amp", "k", "t0", "nu"], guess=_g_richards, bounds=_b_richards, axis="time", note="generalised-logistic sigmoid"),
    "exponential": dict(func=m_exponential, params=["base", "amp", "k"], guess=_g_exp, bounds=_b_exp, axis="time", note="downhill / no-lag saturation"),
    "scaling_law": dict(func=m_scaling, params=["a", "n", "c"], guess=_g_scaling, bounds=_b_scaling, axis="time", note="power-law growth"),
    "brain_cousens": dict(func=m_brain_cousens, params=["c", "d", "f", "b", "e"], guess=_g_brain, bounds=_b_brain, axis="time", note="biphasic / hormesis (Brain–Cousens)"),
    "finke_watzky": dict(func=m_finke_watzky, params=["base", "amp", "k1", "k2"], guess=_g_fw, bounds=_b_fw, axis="time", note="Finke–Watzky 2-step (lumped; not amyloid rates)"),
    # dose (vs concentration)
    "dr_lnt":      dict(func=m_dr_lnt, params=["bg", "slope"], guess=_g_dr_lnt, bounds=_b_free(2), axis="dose", note="linear no-threshold"),
    "dr_threshold": dict(func=m_dr_threshold, params=["bg", "tau", "slope"], guess=_g_dr_threshold, bounds=_b_dr_threshold, axis="dose", note="hockey-stick threshold"),
    "dr_hill":     dict(func=m_dr_hill, params=["c", "top", "ec50", "h"], guess=_g_dr_hill, bounds=_b_dr_hill, axis="dose", note="Hill / 4PL sigmoid"),
    "dr_brain_cousens": dict(func=m_dr_brain_cousens, params=["c", "top", "f", "b", "e"], guess=_g_brain, bounds=_b_brain, axis="dose", note="Brain–Cousens hormesis"),
    "dr_scaling":  dict(func=m_dr_scaling, params=["a", "n", "c"], guess=_g_dr_scaling, bounds=_b_scaling, axis="dose", note="power-law dose scaling"),
}

KINETIC_DESCRIPTIVE = ["lnt", "logistic", "gompertz", "richards", "exponential", "scaling_law"]
KINETIC_BIPHASIC = ["lnt", "brain_cousens"]
KINETIC_MECHANISTIC = ["finke_watzky"]          # + ODE family (mechanistic.py)
DOSE_RESPONSE = ["dr_lnt", "dr_threshold", "dr_hill", "dr_brain_cousens", "dr_scaling"]
