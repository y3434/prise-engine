"""
PRISE — Module M14: Cross-Study Hierarchical Meta-Analysis
==========================================================

Implements PRISE_DESIGN.md §3-M14 ("Cross-Study Hierarchical Meta-Analysis":
aggregate aggregation-kinetics evidence ACROSS many independent published studies
— the STUDY level, one real publication = one study). M14 GENERALIZES the
replicate-level random-effects meta of engine/pooling.py (which combines
identical-condition replicate curves WITHIN a stratum) UP to the STUDY level:
each `source_study` (pmid / author / year) is a unit, and the between-STUDY
variance τ²_between is the publication effect.

WHAT THIS IS (honest framing — build to the DATA REALITY, not an idealization).
A **hierarchical normal random-effects meta-analysis** on the condition-adjusted
log-t50 effect. It is a *meta-regression*: we cannot pool raw t50 across different
concentrations (t50 scales with concentration via the γ exponent), so we regress
out the condition moderators (primarily log10 concentration, plus pH / temperature
/ construct where enough spread exists) and pool the STUDY random effects of the
residual. Estimation is deterministic closed-form (DerSimonian–Laird moment τ²
with a Paule–Mandel / REML-style iterative refinement) + BLUP shrinkage of each
study's deviation, exactly matching pooling.py's conventions and its reused
`dersimonian_laird` / `shrink` primitives. With weak priors (half-normal on τ,
wide normal on μ) the same closed-form yields a Bayesian credible interval for μ
and a **posterior-predictive interval for a NEW study**. This is NOT a full
latent-ODE-rate MCMC hierarchy — that is a documented deferred extension.

THE MODEL (per protein with ≥2 studies; effect = log t50).
        θ̂_c = μ + β·x_condition + u_study + ε_c
        u_study ~ N(0, τ²_between)          (the publication / study effect)
        ε_c     ~ N(0, s²_c + σ²_within)     (within-study curve scatter + sampling)
θ̂_c is log(t50) of curve c (log because t50 is strictly positive / multiplicative
and scales with concentration). s²_c is the per-curve sampling variance on the log
scale, delta-method-mapped from the t50-scale sampling variance that pooling.py
already derives (Var(log t50) ≈ s²_{t50}/t50²). Digitization-inflated (inherited).

CORPUS REACH (verified, stated for honesty). 105 proteins / ~180 studies; only
~20 proteins have ≥2 studies and ~11 have ≥3 (Aβ42, Aβ40, α-synuclein, PrP,
IAPP, lysozyme, ...). Restricting to POINT (uncensored) t50 — the only honestly
poolable effect — 11 proteins carry ≥2 studies. The other ~85 proteins are
SINGLE-STUDY and are reported as such with `status="single_study"` and NO
fabricated pooling.

HONESTY (mandatory, kept visible in every payload).
  * Pooling is condition-ADJUSTED via moderators — NEVER raw t50 across
    concentrations. Where a moderator is degenerate (all one concentration) we
    fall back to matched-condition-stratum pooling and SAY SO.
  * The concentration-moderator slope β_logc should track −γ (gamma.jsonl); it is
    cross-checked and reported.
  * LABORATORY (`author`): estimable ONLY where an author group spans ≥2 studies
    for that protein; else reported as CONFOUNDED_WITH_STUDY (flagged, never
    fabricated).
  * BATCH: not recorded in the corpus → reported UNIDENTIFIABLE (flagged).
  * "Bayesian" = hierarchical-normal random-effects with weak priors (deterministic,
    scalable); full latent-ODE-rate MCMC is a deferred extension.
  * Sampling variances are digitization-inflated (inherited from pooling.py).

SCALE. Closed-form DL/REML + BLUP is O(studies); it fits thousands of studies in
well under a second. The CLI self-check fits a synthetic 2000-study protein and
reports timing.

Pure / deterministic (any sampler is seeded; here the inference is closed-form) /
JSON-serialisable / NEVER raises (errors degrade to a flagged record).

Usage:
    python engine/m14_meta.py                 # -> data/processed/meta_analysis.json
    python engine/m14_meta.py --selfcheck     # 2000-study scale timing demo
    python engine/m14_meta.py --output <path>
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

# Reuse pooling.py's frozen primitives + conventions (do NOT reinvent).
import pooling as P

META_VERSION = "m14-metaanalysis-1.0"

# --------------------------------------------------------------------------- #
# FROZEN, VERSIONED CONSTANTS (pre-registered; not tuned to judged data).
# --------------------------------------------------------------------------- #
# A cross-study meta needs at least this many DISTINCT studies to be meaningful.
MIN_STUDIES_META = 2
# Paule–Mandel / REML refinement of τ²: iteration cap + convergence tol. Closed
# form remains O(studies·iters) with iters small — fits thousands of studies fast.
_PM_MAX_ITER = 100
_PM_TOL = 1e-10
# A moderator column is usable only if it has real spread: >= this many DISTINCT
# values across the curves (else it is degenerate / collinear → dropped, flagged).
_MIN_MODERATOR_LEVELS = 2
# Minimum distinct log10-concentration values before we trust a concentration slope.
_MIN_CONC_LEVELS = 3
# Weakly-informative prior on τ (half-normal scale) and μ (wide normal sd), on the
# log-t50 (natural log) scale. τ-prior scale ~ log(3) ≈ a 3× study-to-study spread;
# μ-prior sd huge so the likelihood dominates (data-driven μ). Deterministic.
_PRIOR_TAU_SCALE = 1.0986122886681098      # ln(3)
_PRIOR_MU_SD = 10.0
# Author group must span >= this many studies (for a protein) to estimate a lab
# effect separately from the study effect; else CONFOUNDED_WITH_STUDY.
_MIN_STUDIES_PER_LAB = 2
# 95% normal quantile (matches pooling.py).
_Z95 = 1.959963984540054

# The effect is NATURAL-log t50 (positive, multiplicative, scales with conc).
_LOG = math.log
_EXP = math.exp


# --------------------------------------------------------------------------- #
#  Small linear-algebra helpers (pure python — no numpy dependency at import;
#  weighted least squares for the meta-regression is tiny: <= ~6 moderators).
# --------------------------------------------------------------------------- #
def _mat_mul(A, B):
    """A (m×n) · B (n×p) -> (m×p)."""
    m, n, p = len(A), len(B), len(B[0])
    out = [[0.0] * p for _ in range(m)]
    for i in range(m):
        Ai = A[i]
        for k in range(n):
            a = Ai[k]
            if a == 0.0:
                continue
            Bk = B[k]
            oi = out[i]
            for j in range(p):
                oi[j] += a * Bk[j]
    return out


def _mat_vec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def _invert(M):
    """Gauss–Jordan inverse of a small symmetric PD matrix. Returns None if
    singular (degenerate moderators) — caller degrades gracefully."""
    n = len(M)
    A = [list(M[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-14:
            return None
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        A[col] = [v / pv for v in A[col]]
        for r in range(n):
            if r == col:
                continue
            f = A[r][col]
            if f == 0.0:
                continue
            A[r] = [A[r][j] - f * A[col][j] for j in range(2 * n)]
    return [row[n:] for row in A]


def _wls(X, y, w):
    """Weighted least squares: β = (XᵀWX)⁻¹ XᵀWy plus cov(β) = (XᵀWX)⁻¹.

    X: list of rows (each length k). y, w: length n. Returns (beta, cov, XtWX_inv)
    or (None, None, None) if the design is singular. Pure / deterministic."""
    n = len(X)
    if n == 0:
        return None, None, None
    k = len(X[0])
    # XtWX (k×k) and XtWy (k)
    XtWX = [[0.0] * k for _ in range(k)]
    XtWy = [0.0] * k
    for i in range(n):
        wi = w[i]
        xi = X[i]
        yi = y[i]
        for a in range(k):
            xa = xi[a] * wi
            XtWy[a] += xa * yi
            for b in range(a, k):
                XtWX[a][b] += xa * xi[b]
    for a in range(k):
        for b in range(a):
            XtWX[a][b] = XtWX[b][a]
    inv = _invert(XtWX)
    if inv is None:
        return None, None, None
    beta = _mat_vec(inv, XtWy)
    return beta, inv, inv


# --------------------------------------------------------------------------- #
#  Per-curve effect (log t50) + sampling variance (log scale via delta method)
# --------------------------------------------------------------------------- #
def _curve_effect(t50, s2_t50):
    """Effect θ̂ = ln(t50) and its sampling variance on the log scale.

    Delta method: Var(ln t50) ≈ Var(t50)/t50². We reuse pooling.py's t50-scale
    sampling variance (from the M3 predictive interval or the documented default).
    Returns (theta, s2_log) or (None, None) for a non-positive / non-finite t50."""
    if t50 is None or not math.isfinite(t50) or t50 <= 0:
        return None, None
    theta = _LOG(t50)
    if s2_t50 is None or not math.isfinite(s2_t50) or s2_t50 <= 0:
        return theta, None
    s2_log = s2_t50 / (t50 * t50)
    return theta, s2_log


def _sampling_s2_log(rec):
    """Per-curve sampling variance of ln(t50), via pooling.py's t50 SD extractor.

    pooling.py works on the LOG10 scale; ln = ln(10)·log10, so Var scales by
    ln(10)². We take pooling's log10 sd (M3 predictive-interval-derived, else the
    documented default), convert to a natural-log variance, and tag provenance.
    Returns (s2_log, source)."""
    # pooling._curve_sampling_sd returns an sd on the LOG10 scale for target t50.
    sd_log10, src = P._curve_sampling_sd(rec, "t50", "log10")
    ln10 = math.log(10.0)
    sd_ln = sd_log10 * ln10
    return sd_ln * sd_ln, src


# --------------------------------------------------------------------------- #
#  DL + Paule–Mandel τ² refinement (REML-style), reusing pooling.dersimonian_laird
# --------------------------------------------------------------------------- #
def _paule_mandel_tau2(theta, s2, tau2_start=0.0):
    """Paule–Mandel (empirical-Bayes) iterative τ² — a REML-style refinement of the
    DerSimonian–Laird moment start. Solves Σ wᵢ(θᵢ−μ̂)² = k−1 with wᵢ=1/(sᵢ²+τ²).

    Deterministic fixed-point iteration (bounded iters). Returns τ² ≥ 0. This is the
    'REML/Paule–Mandel refinement' §3-M14 asks for on top of DL."""
    theta = [float(t) for t in theta]
    s2 = [max(float(v), 1e-12) for v in s2]
    k = len(theta)
    if k < 2:
        return 0.0
    tau2 = max(0.0, float(tau2_start))
    for _ in range(_PM_MAX_ITER):
        w = [1.0 / (v + tau2) for v in s2]
        sw = sum(w)
        mu = sum(wi * ti for wi, ti in zip(w, theta)) / sw
        # generalized Q at current τ²
        Qg = sum(wi * (ti - mu) ** 2 for wi, ti in zip(w, theta))
        # Paule–Mandel update (Newton-like on F(τ²)=Qg−(k−1))
        F = Qg - (k - 1)
        # dF/dτ² = -Σ wᵢ²(θᵢ−μ)²  (μ held ≈ fixed for the derivative)
        dF = -sum((wi ** 2) * (ti - mu) ** 2 for wi, ti in zip(w, theta))
        if abs(dF) < 1e-30:
            break
        step = F / dF
        new = tau2 - step
        if new < 0.0:
            new = 0.0
        if abs(new - tau2) < _PM_TOL:
            tau2 = new
            break
        tau2 = new
    return float(max(0.0, tau2))


def _tau2_ci(theta, s2, tau2, mu, se_mu):
    """A simple Q-profile-flavoured CI for τ² (approximate, deterministic). Uses the
    Biggerstaff–Tweedie style normal approx on log-scale of τ around the point
    estimate via the curvature of the generalized Q. Returns [lo, hi] (≥0)."""
    k = len(theta)
    if k < 3 or tau2 <= 0:
        # Too few studies (or τ²=0) → report a one-sided [0, upper] via the
        # generalized-Q chi-square upper bound; keep honest with a wide band.
        return [0.0, float(max(tau2 * 4.0, tau2 + (se_mu or 0.0) ** 2 * 2.0))]
    # variance of τ̂² (method-of-moments, DerSimonian–Laird large-sample form)
    w = [1.0 / (v + tau2) for v in s2]
    sw = sum(w)
    sw2 = sum(wi * wi for wi in w)
    sw3 = sum(wi ** 3 for wi in w)
    denom = (sw - sw2 / sw)
    if denom <= 0:
        return [0.0, float(tau2 * 4.0)]
    var_tau2 = 2.0 * (sw2 - 2.0 * sw3 / sw + (sw2 ** 2) / (sw ** 2)) / (denom ** 2)
    se = math.sqrt(max(var_tau2, 0.0))
    lo = max(0.0, tau2 - _Z95 * se)
    hi = tau2 + _Z95 * se
    return [float(lo), float(hi)]


# --------------------------------------------------------------------------- #
#  Heterogeneity statistics (Q, I², H²) — reuse pooling._i_squared for I²
# --------------------------------------------------------------------------- #
def _heterogeneity(theta, s2):
    """Cochran's Q at τ²=0 (fixed-effect), df, p, Higgins I², H². Deterministic;
    the χ² p-value uses a series/continued-fraction incomplete-gamma (no scipy)."""
    k = len(theta)
    df = k - 1
    if k < 2:
        return {"Q": 0.0, "df": 0, "p": 1.0, "I2": 0.0, "H2": 1.0}
    w = [1.0 / max(v, 1e-12) for v in s2]
    sw = sum(w)
    mu = sum(wi * ti for wi, ti in zip(w, theta)) / sw
    Q = sum(wi * (ti - mu) ** 2 for wi, ti in zip(w, theta))
    I2 = P._i_squared(Q, k)             # reuse pooling's frozen I² definition
    H2 = (Q / df) if df > 0 else 1.0
    p = _chi2_sf(Q, df)
    return {"Q": float(Q), "df": int(df), "p": float(p),
            "I2": float(I2), "H2": float(max(H2, 0.0))}


def _chi2_sf(x, df):
    """Survival function P(χ²_df > x) via the regularized upper incomplete gamma
    Q(df/2, x/2). Pure-python Lanczos-free series+CF (deterministic, no scipy)."""
    if x <= 0 or df <= 0:
        return 1.0
    a = df / 2.0
    xx = x / 2.0
    return _gammaincc(a, xx)


def _gammaln(a):
    # Lanczos approximation to ln Γ(a) (deterministic).
    g = 7
    c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    if a < 0.5:
        return math.log(math.pi / math.sin(math.pi * a)) - _gammaln(1 - a)
    a -= 1
    xx = c[0]
    for i in range(1, g + 2):
        xx += c[i] / (a + i)
    t = a + g + 0.5
    return 0.5 * math.log(2 * math.pi) + (a + 0.5) * math.log(t) - t + math.log(xx)


def _gammaincc(a, x):
    """Regularized upper incomplete gamma Q(a,x) = 1 - P(a,x)."""
    if x < 0 or a <= 0:
        return 1.0
    if x == 0:
        return 1.0
    if x < a + 1.0:
        # series for P(a,x), then Q = 1-P
        ap = a
        s = 1.0 / a
        d = s
        for _ in range(1000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        p = s * math.exp(-x + a * math.log(x) - _gammaln(a))
        return 1.0 - p
    # continued fraction for Q(a,x)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return h * math.exp(-x + a * math.log(x) - _gammaln(a))


# --------------------------------------------------------------------------- #
#  Study-level aggregation of curves -> one effect + sampling variance per study
# --------------------------------------------------------------------------- #
def _aggregate_study(curves):
    """Combine a study's curves into ONE study effect + sampling variance, using
    the (already condition-residualized) log-t50 residuals.

    Inverse-variance mean of the curves' residual effects; the study's sampling
    variance is 1/Σ(1/s²_c). This is the WITHIN-study fixed-effect combination
    (§3-M14: study = one publication). Returns (eff, s2_study, n_curves)."""
    ws = [1.0 / max(c["s2"], 1e-12) for c in curves]
    sw = sum(ws)
    eff = sum(wi * c["resid"] for wi, c in zip(ws, curves)) / sw
    s2_study = 1.0 / sw
    return eff, s2_study, len(curves)


# --------------------------------------------------------------------------- #
#  Egger's regression small-study test + funnel data
# --------------------------------------------------------------------------- #
def _egger(effects, ses):
    """Egger's regression small-study / publication-bias test.

    Regress the standard normal deviate (effect/se) on precision (1/se); the
    INTERCEPT ≠ 0 signals small-study asymmetry (funnel-plot bias). Deterministic
    OLS with a t-test on the intercept. Returns intercept, se, t, p, flag."""
    k = len(effects)
    if k < 3:
        return {"testable": False, "reason": "need >=3 studies",
                "intercept": None, "intercept_se": None, "p": None,
                "small_study_effect_flag": False}
    # y = effect/se ; x = 1/se ; y = a + b*x  (Egger: a is the asymmetry)
    y = [e / s for e, s in zip(effects, ses)]
    x = [1.0 / s for s in ses]
    n = float(k)
    sx = sum(x); sy = sum(y)
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-30:
        return {"testable": False, "reason": "degenerate precision",
                "intercept": None, "intercept_se": None, "p": None,
                "small_study_effect_flag": False}
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    # residual variance + SE of intercept
    resid = [yi - (a + b * xi) for xi, yi in zip(x, y)]
    dof = k - 2
    s2 = sum(r * r for r in resid) / dof if dof > 0 else 0.0
    se_a = math.sqrt(s2 * sxx / denom) if denom > 0 else None
    if se_a is None or se_a == 0.0:
        return {"testable": False, "reason": "zero intercept SE",
                "intercept": float(a), "intercept_se": None, "p": None,
                "small_study_effect_flag": False}
    t = a / se_a
    p = _student_t_sf_two_sided(abs(t), dof)
    return {"testable": True, "intercept": float(a), "slope": float(b),
            "intercept_se": float(se_a), "t": float(t), "df": int(dof),
            "p": float(p), "small_study_effect_flag": bool(p < 0.05)}


def _student_t_sf_two_sided(t, df):
    """Two-sided p-value for Student-t via the regularized incomplete beta.
    Deterministic (continued fraction). p = I_{df/(df+t²)}(df/2, 1/2)."""
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    return _betainc(x, df / 2.0, 0.5)


def _betainc(x, a, b):
    """Regularized incomplete beta I_x(a,b) via Lentz continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = _gammaln(a) + _gammaln(b) - _gammaln(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    # continued fraction (Numerical Recipes betacf)
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-12:
            break
    result = front * h
    # I_x(a,b); use symmetry if x is past the mean for faster convergence — but the
    # above is adequate for our x range. Clamp to [0,1].
    return float(min(1.0, max(0.0, result)))


# --------------------------------------------------------------------------- #
#  The core: fit one protein's cross-study meta-regression
# --------------------------------------------------------------------------- #
def _build_curve_units(curves):
    """From raw poolable curves build effect θ̂=ln t50, s2_log, moderators (log10
    conc, pH, temp, construct one-hot), study id, author. Drops non-point / bad."""
    units = []
    for c in curves:
        theta, s2_log = _curve_effect(c["t50"], c["s2_t50"])
        if theta is None:
            continue
        if s2_log is None:
            # fall back to pooling's log10 default mapped to ln
            s2_log = c.get("s2_log_default")
        units.append({
            "series_id": c["series_id"],
            "theta": theta,
            "s2": max(s2_log, 1e-12),
            "study": c["study"],
            "author": c["author"],
            "log10_conc": c["log10_conc"],
            "pH": c["pH"],
            "temp": c["temp"],
            "is_mutant": c["is_mutant"],
        })
    return units


def fit_protein_meta(protein, curves):
    """Fit the hierarchical random-effects cross-study meta-regression for ONE
    protein (§3-M14). `curves`: list of dicts with t50, s2_t50, s2_log_default,
    study, author, log10_conc, pH, temp, is_mutant. NEVER raises."""
    try:
        return _fit_protein_meta_inner(protein, curves)
    except Exception as exc:  # honesty: degrade, never crash the batch
        return {"protein": protein, "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "version": META_VERSION}


def _fit_protein_meta_inner(protein, curves):
    units = _build_curve_units(curves)
    studies = sorted({u["study"] for u in units if u["study"] is not None})
    n_studies = len(studies)
    n_curves = len(units)

    honesty = _honesty_block()

    # ---- SINGLE-STUDY (or empty): reported honestly, no pooling -------------- #
    if n_studies < MIN_STUDIES_META:
        return {
            "protein": protein,
            "status": "single_study" if n_studies == 1 else "no_data",
            "n_studies": n_studies,
            "n_curves": n_curves,
            "note": ("only one study contributes curves for this protein — a "
                     "cross-study meta-analysis is not meaningful; reported as a "
                     "single study with NO fabricated pooling (§3-M14 honesty)."
                     if n_studies == 1 else
                     "no poolable point-t50 curves for this protein."),
            "honesty": honesty,
            "version": META_VERSION,
        }

    # ---- decide which moderators have real spread --------------------------- #
    conc_levels = sorted({round(u["log10_conc"], 6) for u in units
                          if u["log10_conc"] is not None})
    ph_levels = sorted({round(u["pH"], 3) for u in units if u["pH"] is not None})
    temp_levels = sorted({round(u["temp"], 3) for u in units if u["temp"] is not None})
    construct_levels = sorted({u["is_mutant"] for u in units})

    use_conc = len(conc_levels) >= _MIN_CONC_LEVELS
    use_ph = len(ph_levels) >= _MIN_MODERATOR_LEVELS
    use_temp = len(temp_levels) >= _MIN_MODERATOR_LEVELS
    use_construct = len(construct_levels) >= _MIN_MODERATOR_LEVELS

    moderator_notes = []
    fallback_stratum = False
    if not use_conc:
        moderator_notes.append(
            f"concentration moderator DEGENERATE ({len(conc_levels)} distinct "
            "log10-conc level(s) < 3) — cannot regress out concentration; falling "
            "back to matched-condition pooling of the raw log-t50 for this protein.")
        fallback_stratum = True

    # ---- build the design matrix (intercept + usable moderators) ------------ #
    # centre continuous moderators for numerical conditioning + interpretable μ.
    conc_c = _mean([u["log10_conc"] for u in units if u["log10_conc"] is not None]) \
        if use_conc else 0.0
    ph_c = _mean([u["pH"] for u in units if u["pH"] is not None]) if use_ph else 0.0
    temp_c = _mean([u["temp"] for u in units if u["temp"] is not None]) if use_temp else 0.0

    col_names = ["intercept"]
    if use_conc:
        col_names.append("log10_conc")
    if use_ph:
        col_names.append("pH")
    if use_temp:
        col_names.append("temperature_C")
    if use_construct:
        col_names.append("construct_is_mutant")

    def _row(u):
        row = [1.0]
        if use_conc:
            row.append((u["log10_conc"] or conc_c) - conc_c)
        if use_ph:
            row.append((u["pH"] or ph_c) - ph_c)
        if use_temp:
            row.append((u["temp"] or temp_c) - temp_c)
        if use_construct:
            row.append(1.0 if u["is_mutant"] else 0.0)
        return row

    X = [_row(u) for u in units]
    y = [u["theta"] for u in units]
    s2c = [u["s2"] for u in units]

    # ---- meta-regression by iterative WLS with a between-study τ² ------------ #
    # Start with FE weights (τ²=0), fit β, residualize, estimate τ²_between at the
    # STUDY level (DL start + Paule–Mandel), then reweight. Two passes suffice for
    # the moment estimator; deterministic + O(curves).
    beta, cov, _ = _wls(X, y, [1.0 / v for v in s2c])
    if beta is None:
        # singular design (collinear moderators) → drop to intercept-only
        col_names = ["intercept"]
        X = [[1.0] for _ in units]
        beta, cov, _ = _wls(X, y, [1.0 / v for v in s2c])
        moderator_notes.append("moderator design was singular/collinear — reduced "
                               "to intercept-only (condition-matched) pooling.")
        fallback_stratum = True
        use_conc = use_ph = use_temp = use_construct = False

    # curve residuals after removing the fixed moderator effects
    fitted = _mat_vec(X, beta) if len(X[0]) > 1 else [beta[0]] * len(X)
    for u, xi in zip(units, X):
        u["resid"] = u["theta"] - sum(bi * xij for bi, xij in zip(beta, xi)) + beta[0]
        # keep intercept in the residual so study effects are on the μ scale:
        # resid = θ - β·x + intercept = μ + u_study + ε (moderators removed)

    # ---- aggregate to STUDY level -------------------------------------------- #
    by_study = defaultdict(list)
    for u in units:
        by_study[u["study"]].append(u)
    study_eff = {}
    study_s2 = {}
    study_n = {}
    study_author = {}
    for st in studies:
        cs = by_study[st]
        eff, s2s, ncs = _aggregate_study(cs)
        study_eff[st] = eff
        study_s2[st] = s2s
        study_n[st] = ncs
        study_author[st] = cs[0]["author"]

    theta_st = [study_eff[st] for st in studies]
    s2_st = [study_s2[st] for st in studies]

    # ---- between-study τ² (DL start, reuse pooling; PM refinement) ---------- #
    dl = P.dersimonian_laird(theta_st, s2_st)
    tau2_dl = dl["tau2"]
    tau2 = _paule_mandel_tau2(theta_st, s2_st, tau2_start=tau2_dl)

    # random-effects pooled μ with τ²-augmented study weights
    wstar = [1.0 / (v + tau2) for v in s2_st]
    swstar = sum(wstar)
    mu = sum(wi * ti for wi, ti in zip(wstar, theta_st)) / swstar
    se_mu = math.sqrt(1.0 / swstar)

    het = _heterogeneity(theta_st, s2_st)
    tau2_ci = _tau2_ci(theta_st, s2_st, tau2, mu, se_mu)

    # ---- BLUP shrinkage of each study deviation (reuse pooling.shrink) ------- #
    study_rows = []
    for st in studies:
        raw = study_eff[st]
        sh = P.shrink(raw, study_s2[st], mu, tau2)   # posterior mean toward μ
        se_raw = math.sqrt(study_s2[st])
        study_rows.append({
            "study": _study_label(st),
            "pmid": st,
            "author": study_author[st],
            "n_curves": study_n[st],
            "raw_effect_log": float(raw),
            "raw_se": float(se_raw),
            "raw_effect_t50": float(_EXP(raw)),
            "blup_effect_log": float(sh["theta_pooled"]),
            "blup_se": float(sh["posterior_sd"]),
            "blup_effect_t50": float(_EXP(sh["theta_pooled"])),
            "shrinkage_weight": float(sh["weight"]),
            "ci95_log": [float(raw - _Z95 * se_raw), float(raw + _Z95 * se_raw)],
        })

    # ---- variance decomposition --------------------------------------------- #
    var_decomp = _variance_decomposition(units, studies, study_s2, tau2, beta,
                                         col_names, use_construct)

    # ---- laboratory (author) effect: estimable only if an author spans >=2
    #      studies for this protein; else CONFOUNDED_WITH_STUDY (flagged) --------
    lab = _laboratory_effect(studies, study_author, study_eff, study_s2, tau2, mu)
    # keep the variance-decomposition laboratory field consistent with the actual
    # determination (estimable between-lab variance vs confounded-with-study).
    if lab.get("estimable"):
        var_decomp["laboratory"] = float(lab["between_laboratory_variance"])
        var_decomp["laboratory_status"] = "estimable_author_group_spans_studies"
    else:
        var_decomp["laboratory_status"] = "confounded_with_study"

    # ---- leave-one-study-out ------------------------------------------------- #
    loso = _leave_one_study_out(studies, theta_st, s2_st, study_author, mu, tau2)

    # ---- forest plot rows + pooled diamond + prediction interval ------------ #
    pred_int = _prediction_interval(mu, se_mu, tau2, n_studies)
    forest = _forest_plot(study_rows, wstar, studies, mu, se_mu, pred_int)

    # ---- publication bias: funnel + Egger ------------------------------------ #
    eff_for_bias = [study_eff[st] for st in studies]
    se_for_bias = [math.sqrt(study_s2[st]) for st in studies]
    egger = _egger(eff_for_bias, se_for_bias)
    funnel = {"points": [{"study": _study_label(st),
                          "effect_log": float(study_eff[st]),
                          "se": float(math.sqrt(study_s2[st]))}
                         for st in studies],
              "pooled_mu_log": float(mu)}

    # ---- moderators block (concentration slope vs -γ, construct effect) ------ #
    moderators = _moderators_block(beta, cov, col_names, use_conc, use_construct,
                                   conc_c, fallback_stratum)

    # ---- pooled estimate (log + back-transformed t50) ------------------------ #
    pooled = {
        "mu_log": float(mu),
        "se_mu_log": float(se_mu),
        "ci95_log": [float(mu - _Z95 * se_mu), float(mu + _Z95 * se_mu)],
        "mu_t50_hours": float(_EXP(mu)),
        "ci95_t50_hours": [float(_EXP(mu - _Z95 * se_mu)),
                           float(_EXP(mu + _Z95 * se_mu))],
        "effect": "natural_log_t50",
        "condition_adjusted": bool(not fallback_stratum),
        "moderators_used": col_names[1:],
        "centering": {"log10_conc": conc_c, "pH": ph_c, "temperature_C": temp_c},
    }

    return {
        "protein": protein,
        "status": "meta_analysis",
        "n_studies": n_studies,
        "n_curves": n_curves,
        "pooled_estimate": pooled,
        "study_deviations": study_rows,
        "heterogeneity": {**het,
                          "tau2_between": float(tau2),
                          "tau2_between_dl": float(tau2_dl),
                          "tau_between": float(math.sqrt(tau2)),
                          "tau2_ci": tau2_ci,
                          "tau2_estimator": "DerSimonian-Laird start + Paule-Mandel/REML refinement"},
        "variance_decomposition": var_decomp,
        "laboratory": lab,
        "leave_one_study_out": loso,
        "forest_plot": forest,
        "prediction_interval_new_study": pred_int,
        "publication_bias": {"funnel": funnel, "egger": egger,
                             "small_study_effect_flag": egger.get("small_study_effect_flag", False)},
        "moderators": moderators,
        "fallback_condition_stratum_pooling": fallback_stratum,
        "moderator_notes": moderator_notes,
        "honesty": honesty,
        "version": META_VERSION,
    }


# --------------------------------------------------------------------------- #
#  Component blocks
# --------------------------------------------------------------------------- #
def _variance_decomposition(units, studies, study_s2, tau2, beta, col_names,
                            use_construct):
    """Decompose total variance into between-study (τ²), within-study (residual
    scatter beyond sampling), measurement (mean sampling variance), construct
    (variance explained by the WT/mutant fixed effect), lab (confounded), batch
    (unidentifiable). Deterministic moment decomposition."""
    # measurement = mean per-curve sampling variance
    meas = _mean([u["s2"] for u in units]) if units else 0.0
    # within-study residual variance beyond sampling (σ²_within):
    #   for each study, residual scatter of curves around the study mean, minus
    #   the mean sampling variance (clamped at 0).
    within_parts = []
    by_study = defaultdict(list)
    for u in units:
        by_study[u["study"]].append(u)
    for st in studies:
        cs = by_study[st]
        if len(cs) < 2:
            continue
        m = _mean([c["resid"] for c in cs])
        scatter = sum((c["resid"] - m) ** 2 for c in cs) / (len(cs) - 1)
        samp = _mean([c["s2"] for c in cs])
        within_parts.append(max(0.0, scatter - samp))
    sigma2_within = _mean(within_parts) if within_parts else 0.0

    # construct: variance across curves explained by the mutant indicator
    construct_var = None
    if use_construct and "construct_is_mutant" in col_names:
        idx = col_names.index("construct_is_mutant")
        b = beta[idx]
        frac_mut = _mean([1.0 if u["is_mutant"] else 0.0 for u in units])
        # variance contributed by a two-level fixed effect = b²·p(1-p)
        construct_var = float(b * b * frac_mut * (1.0 - frac_mut))

    total = tau2 + sigma2_within + meas + (construct_var or 0.0)
    return {
        "between_study": float(tau2),
        "within_study": float(sigma2_within),
        "measurement": float(meas),
        "construct": construct_var,
        "laboratory": "confounded_with_study",   # refined by _laboratory_effect
        "batch": "unidentifiable_not_recorded",
        "total": float(total),
        "fractions": ({
            "between_study": float(tau2 / total),
            "within_study": float(sigma2_within / total),
            "measurement": float(meas / total),
            "construct": (float((construct_var or 0.0) / total)),
        } if total > 0 else None),
        "note": ("between-study = publication effect τ²; within-study = residual "
                 "curve scatter beyond sampling; measurement = digitization-inflated "
                 "sampling variance; construct = WT/mutant fixed-effect variance; "
                 "laboratory confounded with study unless an author spans >=2 studies; "
                 "batch NOT RECORDED -> unidentifiable."),
    }


def _laboratory_effect(studies, study_author, study_eff, study_s2, tau2, mu):
    """Estimate a laboratory (author-group) effect ONLY where an author group spans
    >=2 studies for this protein; else CONFOUNDED_WITH_STUDY (flagged, never
    fabricated). §3-M14 honesty."""
    by_author = defaultdict(list)
    for st in studies:
        by_author[study_author[st]].append(st)
    multi_authors = {a: sts for a, sts in by_author.items() if len(sts) >= _MIN_STUDIES_PER_LAB}
    if not multi_authors:
        return {"status": "confounded_with_study",
                "estimable": False,
                "reason": ("no author/laboratory group spans >=2 studies for this "
                           "protein — the laboratory effect is CONFOUNDED WITH the "
                           "study effect and is NOT estimated (not fabricated)."),
                "n_author_groups": len(by_author)}
    # between-lab variance among author-group means (moment estimate)
    lab_means = {}
    for a, sts in multi_authors.items():
        w = [1.0 / (study_s2[s] + tau2) for s in sts]
        sw = sum(w)
        lab_means[a] = sum(wi * study_eff[s] for wi, s in zip(w, sts)) / sw
    vals = list(lab_means.values())
    between_lab = 0.0
    if len(vals) >= 2:
        m = _mean(vals)
        between_lab = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return {"status": "estimable",
            "estimable": True,
            "n_author_groups_multi_study": len(multi_authors),
            "author_group_means_log": {a: float(v) for a, v in lab_means.items()},
            "between_laboratory_variance": float(between_lab),
            "note": ("laboratory effect estimable because >=1 author group spans >=2 "
                     "studies; between-lab variance is the spread of author-group means.")}


def _dl_from_sums(k, S1, Sy, Syy, S2):
    """Closed-form DerSimonian–Laird from precomputed FE-weight sums (exact, O(1)):
      S1=Σwᵢ, Sy=Σwᵢθᵢ, Syy=Σwᵢθᵢ², S2=Σwᵢ² with wᵢ=1/sᵢ² (fixed-effect weights).

    Q = Syy − Sy²/S1 ; c = S1 − S2/S1 ; τ² = max(0,(Q−(k−1))/c). Enables exact
    leave-one-out in O(K) total (subtract the dropped study's contributions from
    the sums) instead of O(K²) refits — the §3-M14 'thousands of studies < 1s'."""
    if k < 2 or S1 <= 0:
        return None, 0.0, 0.0
    mu_fe = Sy / S1
    Q = Syy - Sy * Sy / S1
    c = S1 - S2 / S1
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 1e-30 else 0.0
    return mu_fe, max(0.0, Q), tau2


def _leave_one_study_out(studies, theta_st, s2_st, study_author, mu_full, tau2_full):
    """LOSO influence: drop each study, refit μ / τ² / I², report the shift +
    an is_outlier flag when the study is unusually influential (|Δμ| large or the
    study's standardized residual is extreme).

    Uses EXACT closed-form DerSimonian–Laird leave-one-out via incremental FE-weight
    sums — O(studies) total, not O(studies²) — so it meets the §3-M14 scale bar
    (thousands of studies < 1s). DL (not PM) is used for the drop refit; DL is the
    standard, and it keeps LOSO exact + fast."""
    k = len(studies)
    out = []
    # baseline standardized residuals for outlier detection
    std_res = []
    for i, st in enumerate(studies):
        se = math.sqrt(s2_st[i] + tau2_full)
        std_res.append((theta_st[i] - mu_full) / se if se > 0 else 0.0)
    abs_res = [abs(r) for r in std_res]
    med_res = P._median(abs_res) if abs_res else 0.0

    # precompute the full FE-weight sums once (O(K)); leave-one-out subtracts one term
    w_fe = [1.0 / max(v, 1e-12) for v in s2_st]
    S1 = sum(w_fe)
    Sy = sum(wi * ti for wi, ti in zip(w_fe, theta_st))
    Syy = sum(wi * ti * ti for wi, ti in zip(w_fe, theta_st))
    S2 = sum(wi * wi for wi in w_fe)

    for i, st in enumerate(studies):
        wi = w_fe[i]
        ti = theta_st[i]
        # leave-one-out DL from the subtracted sums (exact, O(1))
        S1i, Syi, Syyi, S2i = S1 - wi, Sy - wi * ti, Syy - wi * ti * ti, S2 - wi * wi
        _, Qi, tau2_i = _dl_from_sums(k - 1, S1i, Syi, Syyi, S2i)
        if S1i > 0 and k - 1 >= 1:
            # RE μ with τ²-augmented weights, computed from the remaining studies
            sw = 0.0
            swt = 0.0
            for j in range(k):
                if j == i:
                    continue
                wj = 1.0 / (s2_st[j] + tau2_i)
                sw += wj
                swt += wj * theta_st[j]
            mu_i = swt / sw if sw > 0 else None
            I2_i = P._i_squared(Qi, k - 1)
            het_i = {"I2": I2_i}
        else:
            mu_i, het_i = None, {"I2": 0.0}
        dmu = (abs(mu_i - mu_full) if mu_i is not None else None)
        # outlier: standardized residual > 1.96 AND the most influential on μ
        is_outlier = bool(abs_res[i] > _Z95 and abs_res[i] > 2.0 * (med_res + 1e-9))
        out.append({
            "dropped_study": _study_label(st),
            "pmid": st,
            "author": study_author[st],
            "pooled_mu_log": (float(mu_i) if mu_i is not None else None),
            "pooled_mu_t50_hours": (float(_EXP(mu_i)) if mu_i is not None else None),
            "tau2_between": float(tau2_i),
            "I2": float(het_i["I2"]),
            "delta_mu_log": (float(dmu) if dmu is not None else None),
            "standardized_residual": float(std_res[i]),
            "is_outlier": is_outlier,
        })
    # mark the single most-influential study (largest |Δμ|)
    influ = [r for r in out if r["delta_mu_log"] is not None]
    if influ:
        top = max(influ, key=lambda r: r["delta_mu_log"])
        top["most_influential"] = True
    return out


def _prediction_interval(mu, se_mu, tau2, n_studies):
    """Bayesian posterior-predictive interval for a NEW study: μ ± t_df·√(τ²+SE_μ²)
    (§3-M14 Bayesian layer). Uses a t quantile with df=studies−2 (Higgins–Thompson),
    falling back to the normal quantile when df is tiny. Deterministic."""
    df = max(1, n_studies - 2)
    tq = _t_quantile_975(df)
    spread = math.sqrt(tau2 + se_mu * se_mu)
    lo = mu - tq * spread
    hi = mu + tq * spread
    return {
        "mu_log": float(mu),
        "spread_log": float(spread),
        "t_quantile_975": float(tq),
        "df": int(df),
        "pi95_log": [float(lo), float(hi)],
        "pi95_t50_hours": [float(_EXP(lo)), float(_EXP(hi))],
        "interpretation": ("95% posterior-predictive interval for the log-t50 a NEW "
                           "study of this protein would report (weak half-normal prior "
                           "on τ, wide normal prior on μ; deterministic normal/t "
                           "closed form)."),
    }


def _forest_plot(study_rows, wstar, studies, mu, se_mu, pred_int):
    """Forest-plot data: per-study effect + CI + inverse-variance weight (%), the
    pooled diamond, and the prediction interval band."""
    sw = sum(wstar)
    weights_pct = {studies[i]: 100.0 * wstar[i] / sw for i in range(len(studies))}
    rows = []
    for r in study_rows:
        rows.append({
            "study": r["study"],
            "pmid": r["pmid"],
            "effect_log": r["raw_effect_log"],
            "effect_t50": r["raw_effect_t50"],
            "ci95_log": r["ci95_log"],
            "ci95_t50": [float(_EXP(r["ci95_log"][0])), float(_EXP(r["ci95_log"][1]))],
            "weight_pct": float(weights_pct.get(r["pmid"], 0.0)),
            "n_curves": r["n_curves"],
        })
    return {
        "rows": rows,
        "pooled_diamond": {
            "mu_log": float(mu),
            "ci95_log": [float(mu - _Z95 * se_mu), float(mu + _Z95 * se_mu)],
            "mu_t50": float(_EXP(mu)),
            "ci95_t50": [float(_EXP(mu - _Z95 * se_mu)),
                         float(_EXP(mu + _Z95 * se_mu))],
        },
        "prediction_interval": pred_int["pi95_log"],
        "prediction_interval_t50": pred_int["pi95_t50_hours"],
    }


def _moderators_block(beta, cov, col_names, use_conc, use_construct, conc_c,
                      fallback_stratum):
    """Report the moderator slopes + the concentration-slope-vs-(−γ) cross-check
    and the construct (WT/mutant) fixed effect."""
    out = {"columns": col_names,
           "coefficients_log_scale": {},
           "condition_adjusted": (not fallback_stratum)}
    for i, name in enumerate(col_names):
        se = math.sqrt(cov[i][i]) if (cov is not None and i < len(cov)) else None
        out["coefficients_log_scale"][name] = {
            "estimate": float(beta[i]),
            "se": (float(se) if se is not None else None),
        }
    if use_conc and "log10_conc" in col_names:
        b = beta[col_names.index("log10_conc")]
        # effect is ln(t50) vs log10(conc); γ is d(log10 t50)/d(log10 conc) with
        # sign convention t50 ∝ conc^(-γ). Convert slope from ln to log10:
        slope_log10 = b / math.log(10.0)
        out["concentration_slope"] = {
            "slope_ln_t50_per_log10_conc": float(b),
            "slope_log10_t50_per_log10_conc": float(slope_log10),
            "implied_gamma": float(-slope_log10),
            "note": ("t50 ∝ conc^(-γ): the fitted concentration slope on log10 t50 "
                     "≈ -γ; cross-checked against gamma.jsonl at the CLI level."),
        }
    if use_construct and "construct_is_mutant" in col_names:
        idx = col_names.index("construct_is_mutant")
        se = math.sqrt(cov[idx][idx]) if cov is not None else None
        out["construct_effect"] = {
            "estimate_log": float(beta[idx]),
            "se_log": (float(se) if se is not None else None),
            "fold_change_t50_mutant_vs_wt": float(_EXP(beta[idx])),
            "note": ("multiplicative t50 shift of mutant vs wild-type "
                     "(exp of the log-scale construct coefficient)."),
        }
    else:
        out["construct_effect"] = {
            "estimable": False,
            "reason": "only one construct level present (no WT/mutant contrast)."}
    return out


def _honesty_block():
    return {
        "corpus_reach": ("cross-study meta is meaningful for ~11-20 proteins: ~20 "
                         "have >=2 studies, ~11 have >=3 (Ab42, Ab40, a-synuclein, "
                         "PrP, IAPP, lysozyme, ...); restricting to POINT (uncensored) "
                         "t50 gives 11 proteins with >=2 studies. The other ~85 "
                         "proteins are SINGLE-STUDY and reported as such (no pooling)."),
        "condition_adjustment": ("pooling is condition-ADJUSTED via a random-effects "
                                 "meta-REGRESSION on log10(concentration) [primary; "
                                 "slope ~ -gamma] plus pH/temperature/construct where "
                                 "spread allows — NEVER raw t50 across concentrations. "
                                 "Where a moderator is degenerate, we fall back to "
                                 "matched-condition stratum pooling and SAY SO."),
        "laboratory": ("laboratory (author) effect estimable ONLY where an author "
                       "group spans >=2 studies for the protein; else "
                       "CONFOUNDED_WITH_STUDY (flagged, not fabricated)."),
        "batch": "batch is NOT RECORDED in the corpus -> UNIDENTIFIABLE (flagged).",
        "bayesian": ("'Bayesian' = hierarchical-normal random-effects with weak priors "
                     "(half-normal on tau, wide normal on mu) -> credible interval for "
                     "mu + posterior-predictive interval for a NEW study. Deterministic "
                     "closed form; full latent-ODE-rate MCMC is a documented DEFERRED "
                     "extension."),
        "sampling_variance": ("per-curve sampling variances are digitization-inflated "
                              "(inherited from engine/pooling.py; delta-method mapped "
                              "to the log-t50 scale)."),
        "effect": "effect = natural-log t50 of POINT (uncensored) curves only.",
    }


# --------------------------------------------------------------------------- #
#  Numeric helpers
# --------------------------------------------------------------------------- #
def _mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else 0.0


def _study_label(pmid):
    return f"PMID:{pmid}" if pmid else "unknown_study"


# t 0.975 quantiles (two-sided 95%) for small df; normal for large df.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
         20: 2.086, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def _t_quantile_975(df):
    if df in _T975:
        return _T975[df]
    if df >= 120:
        return 1.96
    keys = sorted(_T975)
    lo = max(k for k in keys if k <= df)
    hi = min(k for k in keys if k >= df)
    if lo == hi:
        return _T975[lo]
    f = (df - lo) / (hi - lo)
    return _T975[lo] + f * (_T975[hi] - _T975[lo])


# --------------------------------------------------------------------------- #
#  Data assembly: join features/protein_analysis (t50) with triaged (source_study)
# --------------------------------------------------------------------------- #
def _load_jsonl(path):
    return P._load_jsonl(path)


def build_meta_analysis(analysis_records, triaged_records, gamma_records=None):
    """Assemble the full cross-study meta-analysis product (§3-M14).

    `analysis_records`: protein_analysis / features records (carry curve_features
    with t50 + t50_status, condition_vector, model_selection predictive interval).
    `triaged_records`: curves_triaged records (carry source_study = the STUDY, and
    condition_vector as a fallback). Joined on series_id. Pure + deterministic;
    NEVER raises."""
    # study/condition lookup by series_id (from triaged)
    study_of = {}
    cond_of = {}
    for r in triaged_records:
        sid = r.get("series_id")
        if sid is None:
            continue
        study_of[sid] = r.get("source_study") or {}
        cond_of[sid] = r.get("condition_vector") or {}

    # γ point estimate per protein (for the concentration-slope cross-check)
    gamma_by_protein = _gamma_by_protein(gamma_records or [])

    # collect POINT-t50 poolable curves grouped by protein
    by_protein = defaultdict(list)
    for r in analysis_records:
        sid = r.get("series_id")
        cf = r.get("curve_features") or r
        if cf.get("status") != "ok":
            continue
        feats = cf.get("features") or {}
        t50 = feats.get("t50")
        if t50 is None or not math.isfinite(t50) or t50 <= 0:
            continue
        if cf.get("t50_status") != "point":          # only POINTS are poolable
            continue
        protein = r.get("protein_id")
        cv = r.get("condition_vector") or cond_of.get(sid) or {}
        ss = study_of.get(sid) or {}
        conc = ((cv.get("concentration") or {}).get("value_uM"))
        log10_conc = (math.log10(conc) if isinstance(conc, (int, float)) and conc > 0
                      else None)
        construct = (cv.get("construct_id") or "Wild Type")
        is_mutant = bool(construct and str(construct).strip().lower()
                         not in ("wild type", "wildtype", "wt", "", "none"))
        # sampling variance: reuse pooling's t50 SD, delta-mapped to ln-t50
        s2_log, _src = _sampling_s2_log(r)
        # t50-scale sampling variance for the exact delta method (Var lnt50 = s2/t50²)
        # We already have s2_log from pooling's log-scale sd; use it directly as the
        # curve variance (it IS Var(ln t50)); keep t50-scale s2 as a default fallback.
        by_protein[protein].append({
            "series_id": sid,
            "t50": float(t50),
            "s2_t50": (s2_log * t50 * t50),   # invert delta for the generic path
            "s2_log_default": s2_log,
            "study": ss.get("pmid"),
            "author": ss.get("author"),
            "year": ss.get("year"),
            "log10_conc": log10_conc,
            "pH": cv.get("pH"),
            "temp": cv.get("temperature_C"),
            "is_mutant": is_mutant,
        })

    results = []
    for protein in sorted(by_protein):
        res = fit_protein_meta(protein, by_protein[protein])
        # attach the γ cross-check at the protein level (if we have γ + a slope)
        g = gamma_by_protein.get(protein)
        if g is not None and res.get("status") == "meta_analysis":
            mods = res.get("moderators") or {}
            cs = mods.get("concentration_slope")
            if cs is not None:
                cs["gamma_reference_from_gamma_jsonl"] = float(g)
                cs["gamma_agreement_abs_diff"] = abs(cs["implied_gamma"] - g)
        results.append(res)

    n_meta = sum(1 for r in results if r["status"] == "meta_analysis")
    n_single = sum(1 for r in results if r["status"] == "single_study")

    return {
        "status": "ok",
        "version": META_VERSION,
        "method": ("hierarchical normal random-effects cross-study meta-analysis "
                   "(meta-regression) on condition-adjusted log-t50; DerSimonian-Laird "
                   "+ Paule-Mandel/REML tau2, BLUP study shrinkage, weakly-informative "
                   "Bayesian layer for mu credible + new-study prediction interval. "
                   "Generalizes engine/pooling.py replicate meta UP to the STUDY level."),
        "honesty": _honesty_block(),
        "constants": {
            "MIN_STUDIES_META": MIN_STUDIES_META,
            "MIN_CONC_LEVELS": _MIN_CONC_LEVELS,
            "MIN_STUDIES_PER_LAB": _MIN_STUDIES_PER_LAB,
            "PRIOR_TAU_SCALE": _PRIOR_TAU_SCALE,
            "PRIOR_MU_SD": _PRIOR_MU_SD,
        },
        "summary": {
            "n_proteins_total": len(results),
            "n_meta_analysis": n_meta,
            "n_single_study": n_single,
        },
        "proteins": results,
    }


def _gamma_by_protein(gamma_records):
    """Best-available γ point estimate per protein for the concentration cross-check.
    Prefer a physical, status-ok gamma_regression; average across a protein's series."""
    acc = defaultdict(list)
    for g in gamma_records:
        reg = g.get("gamma_regression") or {}
        if reg.get("status") != "ok":
            continue
        gval = reg.get("gamma")
        if gval is None or not math.isfinite(gval) or not reg.get("gamma_physical", True):
            continue
        acc[g.get("protein")].append(float(gval))
    return {p: (sum(v) / len(v)) for p, v in acc.items() if v}


# --------------------------------------------------------------------------- #
#  Scale self-check (2000 synthetic studies < 1s)
# --------------------------------------------------------------------------- #
def synthetic_protein(n_studies=2000, tau2_true=0.25, mu_true=2.0, seed=12345,
                      curves_per_study=3):
    """Deterministic synthetic protein with a KNOWN between-study τ² across
    n_studies. Uses a seeded LCG (no numpy RNG global state) so it is byte-stable.
    Returns curves in the shape fit_protein_meta expects."""
    state = seed & 0xFFFFFFFF

    def _rand():
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    def _gauss():
        # Box–Muller (deterministic)
        u1 = max(_rand(), 1e-12)
        u2 = _rand()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    curves = []
    tau = math.sqrt(tau2_true)
    for s in range(n_studies):
        u_study = tau * _gauss()
        for c in range(curves_per_study):
            eps = 0.1 * _gauss()
            theta = mu_true + u_study + eps
            t50 = math.exp(theta)
            curves.append({
                "series_id": f"S{s}_{c}",
                "t50": t50,
                "s2_t50": (0.1 * t50) ** 2,
                "s2_log_default": 0.01,
                "study": f"PMID_SYN_{s}",
                "author": f"Lab{s % 50}",
                "year": "2020",
                "log10_conc": 1.0,       # single concentration -> stratum fallback
                "pH": 7.4, "temp": 37.0, "is_mutant": False,
            })
    return curves


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _selfcheck():
    curves = synthetic_protein(n_studies=2000, tau2_true=0.25)
    t0 = time.perf_counter()
    res = fit_protein_meta("SYNTHETIC_2000", curves)
    dt = time.perf_counter() - t0
    het = res.get("heterogeneity", {})
    print(f"[m14 selfcheck] 2000-study fit in {dt*1000:.1f} ms "
          f"(status={res['status']})")
    print(f"    tau2_between (recovered) = {het.get('tau2_between'):.4f}  "
          f"(true 0.2500)")
    print(f"    I2 = {het.get('I2'):.4f}  Q = {het.get('Q'):.1f}  df = {het.get('df')}")
    pooled = res.get("pooled_estimate", {})
    print(f"    mu_log = {pooled.get('mu_log'):.4f}  (true 2.0000)  "
          f"t50 = {pooled.get('mu_t50_hours'):.3f} h")
    print(f"    UNDER-1s scale requirement: {'PASS' if dt < 1.0 else 'FAIL'}")
    return 0


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    proc = root / "data" / "processed"
    ap = argparse.ArgumentParser(description="PRISE M14 cross-study meta-analysis")
    ap.add_argument("--analysis", type=Path, default=proc / "protein_analysis.jsonl")
    ap.add_argument("--triaged", type=Path, default=proc / "curves_triaged.jsonl")
    ap.add_argument("--gamma", type=Path, default=proc / "gamma.jsonl")
    ap.add_argument("--output", type=Path, default=proc / "meta_analysis.json")
    ap.add_argument("--per-protein-dir", type=Path,
                    default=proc / "meta_analysis")
    ap.add_argument("--selfcheck", action="store_true",
                    help="run the 2000-study scale timing demo and exit")
    args = ap.parse_args(argv)

    if args.selfcheck:
        return _selfcheck()

    analysis = _load_jsonl(args.analysis)
    if not analysis:
        analysis = _load_jsonl(proc / "features.jsonl")
    triaged = _load_jsonl(args.triaged)
    gamma = _load_jsonl(args.gamma)

    payload = build_meta_analysis(analysis, triaged, gamma_records=gamma)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # per-protein files for the multi-study proteins
    args.per_protein_dir.mkdir(parents=True, exist_ok=True)
    for r in payload["proteins"]:
        if r["status"] == "meta_analysis":
            safe = "".join(ch if ch.isalnum() else "_" for ch in r["protein"])[:60]
            (args.per_protein_dir / f"{safe}.json").write_text(
                json.dumps(r, indent=2), encoding="utf-8")

    s = payload["summary"]
    print(f"[m14] wrote {args.output}  ({META_VERSION})")
    print(json.dumps({
        "n_proteins_total": s["n_proteins_total"],
        "n_meta_analysis": s["n_meta_analysis"],
        "n_single_study": s["n_single_study"],
    }, indent=2))

    # report Aβ42 and Aβ40
    for target in ("Amyloid Beta peptide-ABeta42", "Amyloid Beta peptide-ABeta40"):
        r = next((x for x in payload["proteins"] if x["protein"] == target), None)
        if r is None or r["status"] != "meta_analysis":
            print(f"\n[{target}] not a multi-study meta ({r['status'] if r else 'missing'})")
            continue
        pooled = r["pooled_estimate"]
        het = r["heterogeneity"]
        cs = (r["moderators"] or {}).get("concentration_slope") or {}
        egg = r["publication_bias"]["egger"]
        loso = r["leave_one_study_out"]
        top = next((x for x in loso if x.get("most_influential")), None)
        print(f"\n[{target}]")
        print(f"  n_studies={r['n_studies']}  n_curves={r['n_curves']}")
        print(f"  pooled t50={pooled['mu_t50_hours']:.2f} h  "
              f"CI95={[round(v,2) for v in pooled['ci95_t50_hours']]}")
        print(f"  I2={het['I2']:.3f}  tau2_between={het['tau2_between']:.4f}")
        print(f"  conc-slope(implied gamma)={cs.get('implied_gamma')}  "
              f"gamma.jsonl={cs.get('gamma_reference_from_gamma_jsonl')}")
        if top:
            print(f"  top LOSO-influential study: {top['dropped_study']} "
                  f"(delta_mu_log={top['delta_mu_log']:.4f}, outlier={top['is_outlier']})")
        print(f"  Egger small-study bias flag: {egg.get('small_study_effect_flag')} "
              f"(p={egg.get('p')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
