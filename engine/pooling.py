"""
PRISE — Deterministic empirical-Bayes PARTIAL-POOLING layer
===========================================================

Builds the partial-pooling / shrinkage spine that PRISE_DESIGN.md §4 specifies
("Partial-pooling hierarchical estimation spine: replicates → protein → stratum;
partial pooling shrinks unstable per-curve estimates toward stratum means — the
right response to sparse per-protein data") but which was previously UNBUILT —
every M4 estimate is per-curve frequentist. §4 also pins the method: *deterministic
approximate inference* (Laplace / fixed-seed variational), **NOT free-running
MCMC**, and an explicit invariant that any "down-weight is an explicit, frozen,
versioned monotone function of a named statistic." This module honours both.

WHAT THIS IS (honest framing). Deterministic empirical-Bayes (James–Stein)
shrinkage at the FEATURE / γ level — a two-level Gaussian model fit by CLOSED-FORM
moment estimators. It is NOT a full Bayesian latent-rate ODE hierarchy and makes
no MCMC draws. It is the right, defensible first layer of the §4 spine: it shrinks
a noisy single-curve t50 (or γ, or log max-rate) toward its CONDITION+ASSAY-MATCHED
stratum mean, by an amount that is a frozen, versioned, monotone function of the
estimate's own sampling variance. Cross-condition pooling is forbidden (Service A
discipline): strata are keyed on protein/uniprot + construct + assay + rounded
pH/temp, never across conditions.

THE MODEL (two-level Gaussian; §4 "replicates → protein → stratum").
For a condition+assay-matched stratum, per-curve target estimates θ̂ᵢ with sampling
variances sᵢ²:
        θᵢ ~ N(μ_stratum, τ²)            (between-curve / true spread)
        θ̂ᵢ ~ N(θᵢ, sᵢ²)                 (within-curve sampling error)
μ_stratum and τ² are estimated by EMPIRICAL BAYES — DerSimonian–Laird closed-form
for τ² (a deterministic method-of-moments estimator; no iteration, no MCMC). The
shrunken (posterior-mean) estimate is

        θ_pooled,i = wᵢ · θ̂ᵢ + (1 − wᵢ) · μ_stratum,   wᵢ = τ² / (τ² + sᵢ²)

wᵢ ∈ [0,1] IS the frozen, versioned, MONOTONE function of the named statistic
sᵢ²/τ² that §4 requires: a precise curve (small sᵢ²) keeps its own value (wᵢ→1);
a noisy curve in a tight stratum (large sᵢ²/τ²) is pulled toward μ (wᵢ→0). The
posterior sd is sqrt(wᵢ·sᵢ²) (= sqrt(1/(1/sᵢ² + 1/τ²))). Strata with n<3 cannot
estimate τ² → no pooling (w=1), flagged `stratum_too_small`.

REPLICATES (§3-M1). Where a stratum contains true REPLICATES (same protein +
construct + IDENTICAL condition_vector incl. concentration, ≥2 curves), the per-
feature estimates are combined by DerSimonian–Laird random-effects meta-analysis
(within-fit + between-replicate variance), giving {pooled, se, tau2_between}. The
replicate t50 SCATTER is RETAINED, not averaged away: a high between-replicate t50
CV is a *stochastic-nucleation signal* (large scatter ⇒ primary / stochastic-
nucleation-dominated kinetics — §3-M1) and sets `stochastic_nucleation_flag`.

s² PROVENANCE (sampling variance per curve). In priority order:
  1. M3 bootstrap-selection t50 predictive interval (`t50_predictive_interval`,
     a [lo, med, hi] in data/processed/protein_analysis.jsonl) → on the log10
     scale a half-width / 1.96 gives s. This is the §4 "wild-bootstrap CI
     half-width" path, the preferred source.
  2. M4 per-feature bootstrap CI (`bootstrap_ci`) where present.
  3. A DOCUMENTED DEFAULT when neither exists: s = DEFAULT_LOG10_SD (a fixed log10
     scatter, frozen below). The default is flagged per curve (`s2_source`) so a
     shrinkage driven by a defaulted variance is never silent.

Everything is pure / deterministic / seed-free (closed form) / JSON-serialisable /
CLI-driven and NEVER raises (errors degrade to a flagged record).

Usage:
    python engine/pooling.py                      # -> data/processed/pooling.json + summary
    python engine/pooling.py --target t50         # log10 t50 (default)
    python engine/pooling.py --output <path>
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

POOLING_VERSION = "pooling-1.0"

# --------------------------------------------------------------------------- #
# FROZEN, VERSIONED CONSTANTS (pre-registered; not tuned to judged data).
# --------------------------------------------------------------------------- #
# Documented default sampling SD on the working (log10) scale, used ONLY when no
# bootstrap-derived sᵢ is available for a curve. ~0.15 dex ≈ a ±40% multiplicative
# t50 1-sigma — a deliberately conservative single-curve digitized-trace error.
# It is FLAGGED per curve (s2_source="default") so a defaulted-variance shrinkage
# is auditable and never silent.
DEFAULT_LOG10_SD = 0.15
DEFAULT_LOG10_SD_RATE = 0.20      # log10 max_rate: noisier off the fitted slope
DEFAULT_SD_GAMMA = 0.30           # γ is on a linear scale already (not logged)

# A stratum needs at least this many curves to estimate τ² (a between-curve
# variance from <3 points is meaningless) → below it, NO pooling (w=1), flagged.
MIN_STRATUM_N = 3
# A replicate group needs ≥2 curves for a between-replicate variance.
MIN_REPLICATE_N = 2
# Between-replicate t50 coefficient-of-variation above which we raise the
# stochastic-nucleation flag (§3-M1: large t50 scatter ⇒ primary/stochastic-
# nucleation-dominated). Frozen, named statistic = CV of the linear t50 values.
STOCHASTIC_NUCLEATION_CV = 0.5

# Rounding granularity for the condition+assay stratum key (Service-A widening:
# pH to 0.1, temperature to 1 °C — matched conditions, never cross-condition).
_PH_ROUND = 1
_TEMP_ROUND = 0

# Target registry: how to read the raw estimate + which scale to pool on. log10
# for the strictly-positive, multiplicatively-distributed kinetic times/rates;
# γ is a dimensionless exponent already, pooled on its linear scale.
_TARGETS = {
    "t50":      {"scale": "log10", "default_sd": DEFAULT_LOG10_SD},
    "max_rate": {"scale": "log10", "default_sd": DEFAULT_LOG10_SD_RATE},
    "gamma":    {"scale": "linear", "default_sd": DEFAULT_SD_GAMMA},
}


# --------------------------------------------------------------------------- #
#  Closed-form empirical-Bayes core (DerSimonian–Laird)
# --------------------------------------------------------------------------- #
def dersimonian_laird(theta, s2):
    """DerSimonian–Laird random-effects moment estimator (deterministic, closed
    form — the §4 'deterministic approximate inference', no MCMC).

    Inputs: per-unit estimates θ̂ᵢ and their sampling variances sᵢ². Returns the
    pooled mean μ, the between-unit variance τ² (clamped at 0), the inverse-variance
    weights, and Cochran's Q. μ is the τ²-augmented inverse-variance mean — exactly
    the empirical-Bayes stratum mean toward which curves shrink."""
    theta = [float(t) for t in theta]
    s2 = [max(float(v), 1e-12) for v in s2]          # guard zero variance
    k = len(theta)
    if k == 0:
        return {"mu": None, "tau2": 0.0, "Q": None, "k": 0}
    if k == 1:
        return {"mu": theta[0], "tau2": 0.0, "Q": 0.0, "k": 1}

    # fixed-effect (inverse-variance) weights + mean
    w = [1.0 / v for v in s2]
    sw = sum(w)
    mu_fe = sum(wi * ti for wi, ti in zip(w, theta)) / sw

    # Cochran's Q and the DL τ² moment estimator
    Q = sum(wi * (ti - mu_fe) ** 2 for wi, ti in zip(w, theta))
    sw2 = sum(wi * wi for wi in w)
    c = sw - sw2 / sw                                # > 0 for k>=2 with distinct w
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 1e-30 else 0.0

    # random-effects weights + μ using τ²-augmented variances
    wstar = [1.0 / (v + tau2) for v in s2]
    swstar = sum(wstar)
    mu = sum(wi * ti for wi, ti in zip(wstar, theta)) / swstar
    se_mu = math.sqrt(1.0 / swstar) if swstar > 0 else None
    return {"mu": float(mu), "tau2": float(tau2), "Q": float(Q), "k": k,
            "se_mu": (float(se_mu) if se_mu is not None else None)}


def shrink(theta_hat, s2, mu, tau2):
    """Empirical-Bayes / James–Stein shrinkage of ONE estimate toward μ.

    weight w = τ²/(τ²+s²) — THE frozen, versioned, monotone function of the named
    statistic s²/τ² (§4 invariant): monotone decreasing in s² (a noisier estimate is
    pulled harder toward the mean) and increasing in τ² (a more dispersed stratum
    lets each curve keep more of its own value). Returns the posterior mean + sd."""
    s2 = max(float(s2), 1e-12)
    tau2 = max(float(tau2), 0.0)
    w = tau2 / (tau2 + s2)                            # in [0, 1)
    pooled = w * theta_hat + (1.0 - w) * mu
    post_var = w * s2                                 # == 1/(1/s2 + 1/tau2)
    return {"theta_pooled": float(pooled), "weight": float(w),
            "posterior_sd": float(math.sqrt(max(post_var, 0.0)))}


# --------------------------------------------------------------------------- #
#  Scale helpers + per-curve sampling-variance extraction
# --------------------------------------------------------------------------- #
def _to_scale(value, scale):
    """Map a raw positive feature value onto the pooling scale. None if non-finite
    or non-positive on a log scale (log of a non-positive is undefined)."""
    if value is None or not math.isfinite(value):
        return None
    if scale == "log10":
        return math.log10(value) if value > 0 else None
    return float(value)


def _from_scale(theta, scale):
    if theta is None or not math.isfinite(theta):
        return None
    return (10.0 ** theta) if scale == "log10" else float(theta)


def _s_from_interval(lo, med, hi, scale):
    """Sampling SD on the pooling scale from a [lo, med, hi] predictive interval
    (M3 bootstrap-selection envelope). On a log10 scale we transform the bounds
    first; the half-width / 1.96 is the ~1σ. Returns None if degenerate."""
    if lo is None or hi is None or hi <= lo:
        return None
    if scale == "log10":
        if lo <= 0 or hi <= 0:
            return None
        lo, hi = math.log10(lo), math.log10(hi)
    half = 0.5 * (hi - lo)
    s = half / 1.959963984540054                     # 95% → 1σ
    return s if s > 0 else None


def _curve_sampling_sd(rec, target, scale):
    """Per-curve sampling SD on the pooling scale, with provenance.

    Priority: (1) M3 t50 predictive interval, (2) M4 per-feature bootstrap CI,
    (3) the documented DEFAULT (flagged). Returns (sd, source_tag)."""
    default_sd = _TARGETS[target]["default_sd"]

    # (1) M3 bootstrap-selection predictive interval (t50 only; the field the M3
    #     sampler emits). Preferred — it is the §4 'wild-bootstrap CI half-width'.
    if target == "t50":
        pi = ((rec.get("model_selection") or {}).get("t50_predictive_interval"))
        if isinstance(pi, (list, tuple)) and len(pi) == 3:
            s = _s_from_interval(pi[0], pi[1], pi[2], scale)
            if s is not None:
                return s, "m3_predictive_interval"

    # (2) M4 per-feature wild-bootstrap CI [lo, hi] (present only if M4 ran with
    #     --bootstrap; on the current artifacts it is absent → falls through).
    cf = rec.get("curve_features") or rec            # accept both join shapes
    bci = (cf.get("bootstrap_ci") or {}).get(target)
    if isinstance(bci, (list, tuple)) and len(bci) == 2:
        s = _s_from_interval(bci[0], None, bci[1], scale)
        if s is not None:
            return s, "m4_bootstrap_ci"

    # (3) documented default — FLAGGED so a defaulted-variance shrinkage is auditable
    return default_sd, "default"


# --------------------------------------------------------------------------- #
#  Record → (target value, status) extraction
# --------------------------------------------------------------------------- #
def _feature_value(rec, target):
    """Pull the raw target value + its censoring status from a protein_analysis /
    features record. Only POINT t50 / max_rate are poolable (a censored t50 is a
    ≥-inequality, not a point — consistent with M6). γ comes from the joined γ
    artifact (handled separately)."""
    cf = rec.get("curve_features") or rec
    if cf.get("status") != "ok":
        return None, None
    feats = cf.get("features") or {}
    if target == "t50":
        return feats.get("t50"), cf.get("t50_status")
    if target == "max_rate":
        # max_rate has no per-curve censoring tag; treat as point when finite > 0
        v = feats.get("max_rate")
        return v, ("point" if (v is not None and v > 0) else None)
    return None, None


def _stratum_key(rec):
    """Condition+assay-matched stratum key (NO cross-condition pooling — Service A).
    (uniprot | construct | assay | round(pH,1) | round(temp,0)). Missing fields are
    kept as None (they widen, never silently merge across an unknown)."""
    cv = rec.get("condition_vector") or {}
    ph = cv.get("pH")
    temp = cv.get("temperature_C")
    return (
        rec.get("uniprot_id") or rec.get("protein_id"),
        cv.get("construct_id") or "Wild Type",
        cv.get("assay_type"),
        round(ph, _PH_ROUND) if isinstance(ph, (int, float)) else None,
        round(temp, _TEMP_ROUND) if isinstance(temp, (int, float)) else None,
    )


def _replicate_key(rec):
    """True-replicate key: identical condition_vector INCLUDING concentration (§3-M1
    'same protein + same conditions'). Adds concentration + ionic detail to the
    stratum key so only genuinely-replicated runs are combined by random effects."""
    cv = rec.get("condition_vector") or {}
    conc = (cv.get("concentration") or {}).get("value_uM")
    sk = _stratum_key(rec)
    return sk + (
        round(conc, 4) if isinstance(conc, (int, float)) else None,
        cv.get("ion"), cv.get("ion_concentration"),
    )


def _key_to_dict(key):
    """JSON-serialisable stratum key (tuples are not JSON keys)."""
    return {"uniprot": key[0], "construct": key[1], "assay": key[2],
            "pH": key[3], "temperature_C": key[4]}


def _key_str(key):
    return "|".join("None" if k is None else str(k) for k in key)


# --------------------------------------------------------------------------- #
#  Per-stratum partial pooling (the core product)
# --------------------------------------------------------------------------- #
def pool_stratum(curves, target, scale):
    """Empirical-Bayes shrinkage over one stratum's POINT estimates.

    `curves`: list of {series_id, theta_hat, s2, s2_source, raw_value}. Returns the
    per-curve pooled records + the stratum summary {mu, tau2, n, ...}. With n<MIN_
    STRATUM_N τ² is unestimable → no pooling (w=1), every record flagged."""
    n = len(curves)
    mu_scale = _from_scale  # alias

    if n < MIN_STRATUM_N:
        # No pooling possible — emit raw==pooled, w=1, flagged. (μ still reported as
        # the simple mean for context, but it is NOT used to move anything.)
        thetas = [c["theta_hat"] for c in curves]
        mu = sum(thetas) / n if n else None
        out = []
        for c in curves:
            out.append(_curve_record(c, target, scale, theta_pooled=c["theta_hat"],
                                     weight=1.0, posterior_sd=math.sqrt(c["s2"]),
                                     mu=mu, tau2=0.0, n=n,
                                     stratum_too_small=True))
        summary = {"mu_scale": mu, "mu_raw": _from_scale(mu, scale), "tau2": 0.0,
                   "n": n, "pooled": False, "reason": "stratum_too_small"}
        return out, summary

    dl = dersimonian_laird([c["theta_hat"] for c in curves],
                           [c["s2"] for c in curves])
    mu, tau2 = dl["mu"], dl["tau2"]
    out = []
    for c in curves:
        sh = shrink(c["theta_hat"], c["s2"], mu, tau2)
        out.append(_curve_record(c, target, scale,
                                 theta_pooled=sh["theta_pooled"],
                                 weight=sh["weight"],
                                 posterior_sd=sh["posterior_sd"],
                                 mu=mu, tau2=tau2, n=n,
                                 stratum_too_small=False))
    weights = [r["shrinkage_weight"] for r in out]
    summary = {
        "mu_scale": float(mu), "mu_raw": _from_scale(mu, scale),
        "tau2": float(tau2), "tau": float(math.sqrt(tau2)),
        "n": n, "pooled": True,
        "Q": dl.get("Q"), "se_mu": dl.get("se_mu"),
        "mean_shrinkage_weight": float(sum(weights) / len(weights)),
        "I2": _i_squared(dl.get("Q"), n),
    }
    return out, summary


def _i_squared(Q, k):
    """Higgins' I² = max(0, (Q-(k-1))/Q) — fraction of total variance that is real
    between-curve (vs sampling) heterogeneity. A descriptive companion to τ²."""
    if Q is None or Q <= 0 or k < 2:
        return 0.0
    return float(max(0.0, (Q - (k - 1)) / Q))


def _curve_record(c, target, scale, *, theta_pooled, weight, posterior_sd,
                  mu, tau2, n, stratum_too_small):
    """Assemble the per-curve pooled output record (the §4 deliverable shape)."""
    theta_raw = c["theta_hat"]
    moved = abs(theta_pooled - theta_raw)
    rec = {
        "series_id": c["series_id"],
        "protein_id": c.get("protein_id"),
        "uniprot_id": c.get("uniprot_id"),
        "target": target,
        "scale": scale,
        # on the pooling scale
        "theta_raw": float(theta_raw),
        "theta_pooled": float(theta_pooled),
        "mu_stratum": (float(mu) if mu is not None else None),
        "tau2": float(tau2),
        "shrinkage_weight": float(weight),
        "posterior_sd": float(posterior_sd),
        "moved": float(moved),
        # back-transformed to the natural unit (hours / rate / γ) for display
        "raw_value": c.get("raw_value"),
        "pooled_value": _from_scale(theta_pooled, scale),
        "mu_stratum_value": (_from_scale(mu, scale) if mu is not None else None),
        # provenance + gates
        "s2": float(c["s2"]),
        "sampling_sd": float(math.sqrt(c["s2"])),
        "s2_source": c.get("s2_source"),
        "n_stratum": n,
        "stratum_key": c.get("stratum_str"),
        "stratum": c.get("stratum_dict"),
        "pooling_version": POOLING_VERSION,
    }
    flags = []
    if stratum_too_small:
        flags.append("stratum_too_small")
    if c.get("s2_source") == "default":
        flags.append("sampling_variance_defaulted")
    rec["flags"] = flags
    return rec


# --------------------------------------------------------------------------- #
#  Replicate random-effects meta-analysis (§3-M1)
# --------------------------------------------------------------------------- #
def replicate_meta(curves, target, scale):
    """DerSimonian–Laird random-effects combination of true REPLICATES (§3-M1) plus
    the retained t50-scatter stochastic-nucleation signal.

    `curves`: same shape as pool_stratum. Returns {pooled, se, tau2_between,
    n_replicates, individual_se_*, stochastic_nucleation_*}. The replicate t50
    SCATTER is reported (not averaged away): high between-replicate CV ⇒ primary /
    stochastic-nucleation-dominated kinetics → stochastic_nucleation_flag."""
    n = len(curves)
    thetas = [c["theta_hat"] for c in curves]
    s2s = [c["s2"] for c in curves]
    dl = dersimonian_laird(thetas, s2s)
    pooled, tau2b, se = dl["mu"], dl["tau2"], dl.get("se_mu")

    # individual SEs (mean sampling SD) for the "pooled SE < individual SE" check
    indiv_sd = [math.sqrt(v) for v in s2s]
    mean_indiv_sd = sum(indiv_sd) / n if n else None

    # retained t50 SCATTER on the NATURAL (linear) scale: the named stochastic-
    # nucleation statistic is the coefficient of variation of the raw t50 values.
    raw_vals = [c["raw_value"] for c in curves
                if c.get("raw_value") is not None and math.isfinite(c["raw_value"])]
    cv_t50 = None
    if len(raw_vals) >= 2:
        m = sum(raw_vals) / len(raw_vals)
        if m > 0:
            var = sum((v - m) ** 2 for v in raw_vals) / len(raw_vals)  # population sd
            cv_t50 = math.sqrt(var) / m
    flag = bool(cv_t50 is not None and cv_t50 > STOCHASTIC_NUCLEATION_CV
                and target == "t50")

    return {
        "stratum_key": curves[0].get("rep_str"),
        "stratum": curves[0].get("rep_dict"),
        "concentration_uM": curves[0].get("concentration_uM"),
        "target": target,
        "scale": scale,
        "n_replicates": n,
        "member_series_ids": [c["series_id"] for c in curves],
        "pooled_scale": (float(pooled) if pooled is not None else None),
        "pooled_value": _from_scale(pooled, scale),
        "se": (float(se) if se is not None else None),
        "tau2_between": float(tau2b),
        "Q": dl.get("Q"),
        "mean_individual_sampling_sd": (float(mean_indiv_sd)
                                        if mean_indiv_sd is not None else None),
        "pooled_se_lt_individual": bool(
            se is not None and mean_indiv_sd is not None and se < mean_indiv_sd),
        # the retained stochastic-nucleation scatter signal (§3-M1)
        "stochastic_nucleation_signal": {
            "between_replicate_t50_cv": (float(cv_t50) if cv_t50 is not None else None),
            "threshold": STOCHASTIC_NUCLEATION_CV,
            "named_statistic": "coefficient_of_variation_of_raw_t50",
            "interpretation": (
                "high between-replicate t50 scatter — kinetics are primary / "
                "stochastic-nucleation-dominated (§3-M1); replicates NOT averaged "
                "away" if flag else
                "between-replicate t50 scatter within the deterministic-nucleation "
                "range"),
        },
        "stochastic_nucleation_flag": flag,
        "pooling_version": POOLING_VERSION,
    }


# --------------------------------------------------------------------------- #
#  γ-level pooling (across strata that share protein + conditions, by concentration
#  series). γ̂ per series with its bootstrap-CI sampling variance.
# --------------------------------------------------------------------------- #
def _gamma_curves_from_records(gamma_records):
    """Turn dual-γ payloads into poolable γ̂ units. Only PHYSICAL, status-ok γ with a
    finite point estimate enter; the bootstrap CI gives sᵢ (default otherwise).
    Stratum = protein + assay + pH + temperature (γ is itself a per-series summary,
    so the γ-stratum groups SERIES of the same protein/condition family)."""
    units = []
    for g in gamma_records:
        reg = g.get("gamma_regression") or {}
        if reg.get("status") != "ok":
            continue
        gamma = reg.get("gamma")
        if gamma is None or not math.isfinite(gamma) or not reg.get("gamma_physical", True):
            continue
        ci = reg.get("gamma_ci")
        if isinstance(ci, (list, tuple)) and len(ci) == 2 and ci[1] > ci[0]:
            s = 0.5 * (ci[1] - ci[0]) / 1.959963984540054
            src = "gamma_bootstrap_ci"
        else:
            s = DEFAULT_SD_GAMMA
            src = "default"
        cond = g.get("conditions") or {}
        sk = (g.get("protein"), cond.get("assay"), cond.get("pH"),
              cond.get("temperature_C"))
        units.append({
            "series_id": g.get("concentration_series_id"),
            "protein_id": g.get("protein"),
            "uniprot_id": None,
            "theta_hat": float(gamma), "raw_value": float(gamma),
            "s2": float(s) ** 2, "s2_source": src,
            "gamma_stratum": sk,
        })
    return units


# --------------------------------------------------------------------------- #
#  I/O helpers
# --------------------------------------------------------------------------- #
def _load_jsonl(path):
    out = []
    if not Path(path).exists():
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


# --------------------------------------------------------------------------- #
#  Assembler
# --------------------------------------------------------------------------- #
def build_pooling(records, gamma_records=None, target="t50"):
    """Assemble the full deterministic partial-pooling product (§4 spine):

      * per-curve pooled records (shrunk toward the condition+assay-matched mean),
      * per-stratum summaries (μ, τ², n, mean weight),
      * replicate random-effects meta-analyses + stochastic-nucleation flags,
      * (target=t50 only) a γ-level empirical-Bayes pool across concentration series.

    `records`: protein_analysis / features records (must carry curve_features +
    condition_vector). Pure + deterministic — same input → byte-identical output."""
    spec = _TARGETS.get(target)
    if spec is None:
        return {"status": "error", "reason": f"unknown target '{target}'",
                "pooling_version": POOLING_VERSION}
    scale = spec["scale"]

    # 1) collect poolable POINT units, grouped by stratum
    by_stratum = defaultdict(list)
    by_replicate = defaultdict(list)
    n_seen = n_poolable = n_censored = 0
    for rec in records:
        value, status = _feature_value(rec, target)
        n_seen += 1
        if value is None or not math.isfinite(value) or value <= 0:
            continue
        if status != "point":                        # censored/biased → not a point
            n_censored += 1
            continue
        theta = _to_scale(value, scale)
        if theta is None:
            continue
        sd, src = _curve_sampling_sd(rec, target, scale)
        sk = _stratum_key(rec)
        rk = _replicate_key(rec)
        cv = rec.get("condition_vector") or {}
        unit = {
            "series_id": rec.get("series_id"),
            "protein_id": rec.get("protein_id"),
            "uniprot_id": rec.get("uniprot_id"),
            "theta_hat": theta, "raw_value": float(value),
            "s2": float(sd) ** 2, "s2_source": src,
            "stratum_str": _key_str(sk), "stratum_dict": _key_to_dict(sk),
            "rep_str": _key_str(rk), "rep_dict": _key_to_dict(sk),
            "concentration_uM": (cv.get("concentration") or {}).get("value_uM"),
        }
        by_stratum[sk].append(unit)
        by_replicate[rk].append(unit)
        n_poolable += 1

    # 2) pool each stratum (deterministic order: sorted by key string)
    curve_records = []
    stratum_summaries = []
    for sk in sorted(by_stratum, key=_key_str):
        curves = by_stratum[sk]
        out, summary = pool_stratum(curves, target, scale)
        curve_records.extend(out)
        summary["stratum_key"] = _key_str(sk)
        summary["stratum"] = _key_to_dict(sk)
        stratum_summaries.append(summary)

    # 3) replicate random-effects meta (≥2 identical-condition curves)
    replicate_metas = []
    for rk in sorted(by_replicate, key=_key_str):
        curves = by_replicate[rk]
        if len(curves) < MIN_REPLICATE_N:
            continue
        replicate_metas.append(replicate_meta(curves, target, scale))

    # 4) γ-level pool (only meaningful for t50 runs; γ is its own target family)
    gamma_pool = None
    if target == "t50" and gamma_records:
        gunits = _gamma_curves_from_records(gamma_records)
        gstrat = defaultdict(list)
        for u in gunits:
            gstrat[u["gamma_stratum"]].append(u)
        g_curve_records, g_summaries = [], []
        for gk in sorted(gstrat, key=lambda k: tuple("None" if x is None else str(x)
                                                     for x in k)):
            cs = gstrat[gk]
            for u in cs:
                u["stratum_str"] = "γ:" + "|".join(
                    "None" if x is None else str(x) for x in gk)
                u["stratum_dict"] = {"protein": gk[0], "assay": gk[1],
                                     "pH": gk[2], "temperature_C": gk[3]}
            out, summary = pool_stratum(cs, "gamma", "linear")
            g_curve_records.extend(out)
            summary["gamma_stratum"] = list(gk)
            g_summaries.append(summary)
        gamma_pool = {"n_gamma_units": len(gunits),
                      "n_gamma_strata": len(gstrat),
                      "curve_records": g_curve_records,
                      "stratum_summaries": g_summaries}

    # 5) summary counters
    n_shrunk = sum(1 for r in curve_records
                   if not r["flags"] or "stratum_too_small" not in r["flags"])
    n_moved = sum(1 for r in curve_records if r["moved"] > 1e-9
                  and "stratum_too_small" not in r["flags"])
    n_too_small = sum(1 for r in curve_records
                      if "stratum_too_small" in r["flags"])
    weights = [r["shrinkage_weight"] for r in curve_records
               if "stratum_too_small" not in r["flags"]]
    moved_vals = [r["moved"] for r in curve_records
                  if "stratum_too_small" not in r["flags"]]
    median_w = _median(weights) if weights else None
    n_stoch = sum(1 for m in replicate_metas if m["stochastic_nucleation_flag"])

    # biggest-shrinkage example (largest |pooled − raw| among genuinely-pooled)
    biggest = None
    pooled_only = [r for r in curve_records
                   if "stratum_too_small" not in r["flags"]]
    if pooled_only:
        b = max(pooled_only, key=lambda r: r["moved"])
        biggest = {
            "series_id": b["series_id"], "protein_id": b["protein_id"],
            "stratum_key": b["stratum_key"],
            "raw_value": b["raw_value"], "pooled_value": b["pooled_value"],
            "mu_stratum_value": b["mu_stratum_value"],
            "shrinkage_weight": b["shrinkage_weight"],
            "moved_log10": b["moved"], "n_stratum": b["n_stratum"],
        }

    return {
        "status": "ok",
        "pooling_version": POOLING_VERSION,
        "target": target, "scale": scale,
        "method": ("deterministic empirical-Bayes (James–Stein) shrinkage; "
                   "DerSimonian–Laird closed-form τ² (NO MCMC). weight = "
                   "τ²/(τ²+s²) — frozen, versioned, monotone in s²/τ² (§4 invariant). "
                   "Condition+assay-matched strata only (no cross-condition pooling)."),
        "honesty": ("deterministic empirical-Bayes shrinkage at the FEATURE/γ level "
                    "(closed-form, no MCMC) — NOT a full Bayesian latent-rate ODE "
                    "hierarchy; the shrinkage weight is a frozen versioned function "
                    "of the curve's own sampling variance"),
        "constants": {
            "MIN_STRATUM_N": MIN_STRATUM_N,
            "DEFAULT_LOG10_SD": DEFAULT_LOG10_SD,
            "DEFAULT_LOG10_SD_RATE": DEFAULT_LOG10_SD_RATE,
            "DEFAULT_SD_GAMMA": DEFAULT_SD_GAMMA,
            "STOCHASTIC_NUCLEATION_CV": STOCHASTIC_NUCLEATION_CV,
            "pH_round": _PH_ROUND, "temp_round": _TEMP_ROUND,
        },
        "summary": {
            "n_curves_seen": n_seen,
            "n_poolable_point": n_poolable,
            "n_censored_excluded": n_censored,
            "n_strata": len(stratum_summaries),
            "n_strata_pooled": sum(1 for s in stratum_summaries if s["pooled"]),
            "n_shrunk": len(weights),
            "n_moved": n_moved,
            "n_in_too_small_strata": n_too_small,
            "median_shrinkage_weight": median_w,
            "mean_shrinkage_weight": (sum(weights) / len(weights)
                                      if weights else None),
            "median_moved_log10": _median(moved_vals) if moved_vals else None,
            "n_replicate_groups": len(replicate_metas),
            "n_stochastic_nucleation_flagged": n_stoch,
            "biggest_shrinkage": biggest,
        },
        "curve_records": curve_records,
        "stratum_summaries": stratum_summaries,
        "replicate_meta": replicate_metas,
        "gamma_pool": gamma_pool,
    }


def _median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return None
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    proc = root / "data" / "processed"
    ap = argparse.ArgumentParser(description="PRISE deterministic EB partial pooling")
    ap.add_argument("--input", type=Path, default=proc / "protein_analysis.jsonl",
                    help="per-curve records with curve_features + condition_vector")
    ap.add_argument("--gamma", type=Path, default=proc / "gamma.jsonl")
    ap.add_argument("--output", type=Path, default=proc / "pooling.json")
    ap.add_argument("--target", default="t50", choices=sorted(_TARGETS))
    args = ap.parse_args(argv)

    records = _load_jsonl(args.input)
    if not records:
        # M7 artifact absent → fall back to raw M4 features (also carries
        # curve_features-equivalent fields under the record root)
        records = _load_jsonl(proc / "features.jsonl")
    gamma_records = _load_jsonl(args.gamma)

    payload = build_pooling(records, gamma_records=gamma_records, target=args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    s = payload.get("summary", {})
    print(f"[pooling] wrote {args.output}  (target={args.target}, "
          f"{POOLING_VERSION})")
    print(json.dumps({
        "target": args.target,
        "n_poolable_point": s.get("n_poolable_point"),
        "n_strata": s.get("n_strata"),
        "n_strata_pooled": s.get("n_strata_pooled"),
        "n_shrunk": s.get("n_shrunk"),
        "n_moved": s.get("n_moved"),
        "median_shrinkage_weight": s.get("median_shrinkage_weight"),
        "n_replicate_groups": s.get("n_replicate_groups"),
        "n_stochastic_nucleation_flagged": s.get("n_stochastic_nucleation_flagged"),
        "biggest_shrinkage": s.get("biggest_shrinkage"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
