"""
PRISE — Conformal Prediction layer (split / Mondrian)
=====================================================

Closes the design's own admission (PRISE_DESIGN.md §4 "Statistical inference
spine"; §6C Service C) that the engine emits point estimates + wild-bootstrap
percentile CIs but **no finite-sample-valid, distribution-free coverage**. On
this small / noisy / autocorrelated / right-censored corpus the bootstrap
percentile intervals under-cover. Split-conformal prediction fixes that: given a
calibration set with KNOWN truth it yields intervals (continuous targets) and
prediction sets (the regime classifier) with a **distribution-free, finite-sample
coverage guarantee** of >= 1 - alpha — assuming only exchangeability between the
calibration and test points. This is genuinely novel for amyloid kinetics.

WHY Service C is the calibration source: split conformal needs labelled examples
where the truth is known. Real CPAD curves are digitized literature traces with
NO ground-truth t50. Service C's synthetic generator (`generate_curve` +
`make_series`) emits curves from a *known* forward model under the SAME
pre-registered noise / artifact / censoring law (§6C, decision C2) the real
corpus suffers — so a residual measured against synthetic truth is the engine's
genuine recovery error, leakage-free and reproducible. We run the REAL engine on
each synthetic curve (M1 triage via `make_series`, M2 `fit_curve`, M4
`extract_features` -> t50, M5 `classify_descriptive` -> regime) so the
nonconformity scores are exactly what the deployed pipeline produces.

What this module implements (§4):
  1. CONTINUOUS conformal for t50 — split-conformal absolute-residual score
     s = |t50_hat - t50_true|. The (1-alpha) empirical quantile of calibration
     scores gives a marginal interval [t50_hat - q, t50_hat + q] with guaranteed
     >= 1-alpha coverage. MONDRIAN (group-conditional) stratifies that quantile by
     (censoring_class, assay_endpoint_class, regime tier) so coverage holds WITHIN
     strata; thin strata fall back to the marginal quantile, flagged.
     RIGHT-CENSORED t50 is a LOWER BOUND -> ONE-SIDED conformal: only the upper
     tail is corrected, interval [t50_hat - q, +inf). Left-censored excluded with
     a flag (consistent with M4, which excludes left-only t50 from γ).
  2. CLASSIFICATION conformal for the descriptive regime — nonconformity =
     1 - normalized_score(true_class) from the engine's R²/AICc Akaike weights;
     the calibration quantile yields PREDICTION SETS (>= 1-alpha marginal coverage)
     that contain >1 regime only when the call is genuinely ambiguous.
  3. VALIDATION — a deterministic, frozen proper-calibration / validation split;
     reports REALIZED coverage (should be ~>= target) + mean interval width / set
     size, overall and per Mondrian stratum. This is the proof the layer works.

Pure / deterministic / seeded / JSON-serialisable / never raises. Same seed +
frozen split -> byte-identical re-runs (§7).

Usage:
    python engine/conformal.py                       # default battery -> JSON + summary
    python engine/conformal.py --n 400 --seed 0 --alpha 0.1
    python engine/conformal.py --output data/processed/service_c_conformal.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Conformal sits ON TOP of the engine + Service C and only IMPORTS them — it does
# NOT modify any inference module (M1–M6 / Service C). The calibration source is
# Service C's generator; the scores come from the real engine fit/feature/classify.
import service_c as sc
from m2_fit import fit_curve
from m3_select import akaike_weights, t50_of
from m4_features import extract_features
from m5_classify import classify_descriptive

CONFORMAL_VERSION = "conformal-1.0"

# ============================================================================ #
#  PRE-REGISTERED CONSTANTS (frozen; not tuned to any judged data)
# ============================================================================ #
# Default miscoverage level. alpha=0.1 -> 90% target coverage (the headline).
DEFAULT_ALPHA = 0.10
# A Mondrian stratum needs at least this many proper-calibration scores before we
# trust its own quantile; below it we FALL BACK to the marginal (pooled) quantile
# and flag the stratum as "thin". 20 keeps the finite-sample quantile inflation
# (the (n+1) correction) modest while still letting most strata stand alone.
MIN_STRATUM_CALIB = 20
# Fraction of the synthetic set held out for VALIDATION (the rest is proper
# calibration). 0.5 = an even split; seeded + frozen so re-runs are identical.
VALIDATION_FRAC = 0.50
# The descriptive regimes the conformal classifier ranges over (the M5 leaves a
# single curve can land in; non_monotonic / anomalous are routed away from the
# conformal classifier — they are detector outputs, not a graded score).
REGIME_LABELS = (
    "cooperative_sigmoidal",
    "gradual_non_cooperative",
    "threshold_driven",
    "no_detectable_aggregation",
)
# Pre-registered calibration grid (generators × conditions × censoring). Mirrors
# Service C's own banks so the calibration distribution matches what the engine
# meets downstream. Mechanistic variants are included so right-censoring and the
# non-cooperative tail are represented. Closed-form models carry their known
# regime; mechanistic curves derive it from shape (Service C's _infer_regime).
_CALIB_CLOSED_FORM = ("logistic", "gompertz", "richards", "exponential", "scaling_law")
_CALIB_MECH = ("nucleation_elongation", "secondary_nucleation",
               "fragmentation", "saturating_secondary")
_CALIB_CENSOR = ("none", "right", "left")


# ============================================================================ #
#  1. CALIBRATION-SET CONSTRUCTION  (run the REAL engine on synthetic truth)
# ============================================================================ #
def _assay_endpoint_class(assay_mass: bool) -> str:
    """The Mondrian assay axis = the SAME mass-proportionality tag M5/M6 gate on:
    a fibril-specific mass-reporting assay (ThT/ThS) is 'amyloid'; everything else
    is 'generic'. Conformal coverage is conditioned on it because t50 recovery
    differs by assay semantics."""
    return "amyloid" if assay_mass else "generic"


def _stratum_key(censoring_class: str, assay_endpoint: str, regime: str) -> str:
    """Mondrian stratum = (censoring_class, assay_endpoint_class, regime tier),
    all engine-produced tags. A flat string key keeps the frozen quantile table
    JSON-serialisable and re-run stable."""
    return f"{censoring_class}|{assay_endpoint}|{regime}"


def _draw_calib_spec(i: int, rng: np.random.Generator) -> dict:
    """Deterministically draw one calibration curve spec from the pre-registered
    grid. The index + a per-draw seed make the whole set reproducible; we vary the
    generator family, the censoring, and (for closed-form) the noise scale so the
    calibration distribution spans clean -> noisy and uncensored -> censored."""
    # interleave closed-form and mechanistic generators across the grid
    use_mech = (i % 3 == 2)                      # ~1/3 mechanistic, 2/3 closed-form
    censor = _CALIB_CENSOR[i % len(_CALIB_CENSOR)]
    # vary the assay axis: most synthetic ThT curves are mass-reporting (amyloid);
    # a deterministic minority are tagged non-mass (generic) so that stratum exists.
    assay_mass = not (i % 4 == 3)
    if use_mech:
        variant = _CALIB_MECH[i % len(_CALIB_MECH)]
        conc = float(sc.MECH_SERIES_CONCS_UM[i % len(sc.MECH_SERIES_CONCS_UM)])
        spec = {"generator": "mechanistic", "variant": variant,
                "concentration_uM": conc, "n_points": sc.N_POINTS_DEFAULT,
                "t_max": 40.0, "censor": censor}
    else:
        model = _CALIB_CLOSED_FORM[i % len(_CALIB_CLOSED_FORM)]
        spec = {"generator": "closed_form", "model": model,
                "n_points": sc.N_POINTS_DEFAULT, "t_max": 40.0, "censor": censor}
    return {"spec": spec, "assay_mass": assay_mass}


def _regime_scores(series: dict, fit_result: dict) -> dict:
    """Per-regime normalized score for the classification conformal layer. Built
    from the engine's own AICc Akaike weights (M3 `akaike_weights`) mapped onto the
    descriptive regime each converged model implies, so the score is the engine's
    actual evidence for each regime — NOT an external probability. Returns a dict
    {regime: score in [0,1]} that sums to ~1 over the regimes with support."""
    fits = fit_result.get("fits", {}) or {}
    converged = {n: r for n, r in fits.items()
                 if r.get("converged") and r.get("aicc") is not None}
    # the regime a converged closed-form model implies (mirrors Service C's map +
    # M5's shape logic at the coarse level)
    model_regime = {
        "logistic": "cooperative_sigmoidal", "gompertz": "cooperative_sigmoidal",
        "richards": "cooperative_sigmoidal", "finke_watzky": "cooperative_sigmoidal",
        "exponential": "gradual_non_cooperative", "scaling_law": "gradual_non_cooperative",
        "lnt": "gradual_non_cooperative",
        "brain_cousens": "non_monotonic_settling",
    }
    scores = {r: 0.0 for r in REGIME_LABELS}
    if not converged:
        return scores
    weights = akaike_weights(converged)
    for name, w in weights.items():
        reg = model_regime.get(name)
        if reg in scores:
            scores[reg] += float(w)
    # refine cooperative -> threshold_driven using the SAME M5 descriptive call on
    # the actual features (so the threshold_driven stratum is populated honestly)
    feat = extract_features(series, fit_result=fit_result)
    desc = classify_descriptive(series.get("m1", {}), feat)
    dreg = desc.get("regime")
    if dreg == "threshold_driven" and scores.get("cooperative_sigmoidal", 0.0) > 0:
        # move the cooperative mass onto threshold_driven (the sharp sub-shape)
        scores["threshold_driven"] += scores["cooperative_sigmoidal"]
        scores["cooperative_sigmoidal"] = 0.0
    elif dreg == "no_detectable_aggregation":
        scores = {r: 0.0 for r in REGIME_LABELS}
        scores["no_detectable_aggregation"] = 1.0
    z = sum(scores.values()) or 1.0
    return {r: v / z for r, v in scores.items()}


def _build_calibration_points(n: int, seed: int) -> list[dict]:
    """Generate `n` synthetic curves across the pre-registered grid, run the REAL
    engine on each, and emit one calibration point per curve carrying the
    nonconformity ingredients: t50_hat vs t50_true, the per-feature SD (for optional
    studentization), the Mondrian tags, and the per-regime classification scores +
    true regime. Never raises; a bad/un-fittable draw is skipped (recorded as a
    miss count), so the calibration set is exactly the curves the engine can score."""
    points = []
    n_skipped = 0
    for i in range(n):
        s = seed * 100003 + i
        rng = sc._rng(s)
        draw = _draw_calib_spec(i, rng)
        curve = sc.generate_curve(draw["spec"], seed=s + 1)
        gt = curve.get("ground_truth", {})
        t50_true = gt.get("t50_true")
        regime_true = gt.get("regime_true")
        if t50_true is None:
            n_skipped += 1
            continue
        series = sc.make_series(curve, series_id=f"calib-{i}",
                                assay_mass=draw["assay_mass"])
        m1 = series.get("m1", {})
        cens = m1.get("censoring_class", "none")
        # left-only t50 is biased early and EXCLUDED from continuous conformal
        # (consistent with M4 / §S3). It can still feed the classifier, but we keep
        # the calibration point's t50 channel inactive via a flag.
        left_only = cens in ("left",)
        fr = fit_curve(series)
        feat = extract_features(series, fit_result=fr, B=20, seed=s + 2)
        if feat.get("status") != "ok":
            n_skipped += 1
            continue
        f = feat.get("features", {}) or {}
        t50_hat = f.get("t50")
        # per-fit SD for studentization: half the bootstrap t50 CI width / 1.96,
        # the engine's own conditional uncertainty (M4). None -> unstudentized.
        ci = (feat.get("bootstrap_ci") or {}).get("t50")
        t50_sd = None
        if ci and len(ci) == 2 and all(math.isfinite(v) for v in ci):
            t50_sd = max((ci[1] - ci[0]) / (2.0 * 1.96), 1e-6)
        assay_endpoint = _assay_endpoint_class(draw["assay_mass"])
        # regime tag for the stratum: the engine's descriptive call (what we will
        # actually condition on at test time, since true regime is unknown then)
        regime_tag = classify_descriptive(m1, feat).get("regime", "unknown")
        # a non-monotonic / settling-confounded curve has NO well-defined sigmoidal
        # t50 (the half-max of a biphasic fit is meaningless), so it is excluded from
        # the t50 channel — consistent with M5 routing it to `non_monotonic_settling`
        # and M2 fitting it with a biphasic model. It still feeds the regime channel.
        t50_excluded = left_only or regime_tag == "non_monotonic_settling"
        scores = _regime_scores(series, fr)
        points.append({
            "idx": i,
            "t50_true": float(t50_true),
            "t50_hat": (float(t50_hat) if t50_hat is not None else None),
            "t50_sd": (float(t50_sd) if t50_sd is not None else None),
            "censoring_class": cens,
            "right_censored": cens in ("right", "left_right"),
            "left_only": bool(left_only),
            "t50_excluded": bool(t50_excluded),
            "t50_excluded_reason": ("left_only_biased_early" if left_only else
                                    ("non_monotonic_no_sigmoidal_t50"
                                     if regime_tag == "non_monotonic_settling" else None)),
            "assay_endpoint": assay_endpoint,
            "regime_tag": regime_tag,
            "regime_true": regime_true,
            "stratum": _stratum_key(cens, assay_endpoint, regime_tag),
            "regime_scores": scores,
        })
    if points:
        points[0]["_n_skipped"] = n_skipped     # carried for the report, harmless
    return points


# ============================================================================ #
#  2. SPLIT / MONDRIAN QUANTILES
# ============================================================================ #
def _conformal_quantile(scores: list[float], alpha: float) -> float | None:
    """The finite-sample split-conformal quantile: the ceil((n+1)(1-alpha))/n
    empirical quantile of the calibration nonconformity scores. The (n+1)
    correction is what buys the EXACT finite-sample >= 1-alpha guarantee (vs the
    plain (1-alpha) sample quantile, which only holds asymptotically). Returns the
    max score when the rank exceeds n (small-n / high-coverage regime), or None
    when there are no scores."""
    s = sorted(v for v in scores if v is not None and math.isfinite(v))
    n = len(s)
    if n == 0:
        return None
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank >= n:                                # quantile beyond the sample -> +inf
        return float(s[-1])                      # report the max (widest finite q)
    return float(s[rank - 1])                     # 1-indexed rank


def _t50_nonconformity(p: dict, studentize: bool) -> float | None:
    """Absolute-residual nonconformity for one calibration point. For RIGHT-censored
    points the true t50 is a LOWER BOUND, so a fit that under-estimates is the only
    error that matters -> the score is the ONE-SIDED positive part (true - hat);
    an over-estimate beyond a lower bound is not penalised. Uncensored points use
    the symmetric |hat - true|. Optionally studentized by the per-fit SD so the
    quantile is a multiple of each curve's own uncertainty (tighter where the fit
    is well-determined)."""
    hat, true = p.get("t50_hat"), p.get("t50_true")
    if hat is None or true is None:
        return None
    if p.get("right_censored"):
        raw = max(true - hat, 0.0)               # only under-estimation is an error
    else:
        raw = abs(hat - true)
    if studentize:
        sd = p.get("t50_sd")
        if sd is None or sd <= 0:
            return None                          # cannot studentize -> drop from that pool
        return raw / sd
    return raw


def _regime_nonconformity(p: dict) -> float | None:
    """Classification nonconformity = 1 - normalized_score(TRUE class) (the standard
    split-conformal score for sets). A high score = the engine gave the true regime
    little evidence -> that calibration point pushes the quantile up -> the test-time
    set must be larger to stay covered. None when the true regime is outside the
    conformal label space (e.g. a non_monotonic generator)."""
    reg_true = p.get("regime_true")
    scores = p.get("regime_scores") or {}
    if reg_true not in scores:
        return None
    return float(1.0 - scores.get(reg_true, 0.0))


def build_conformal_calibration(n: int = 400, seed: int = 0,
                                alpha: float = DEFAULT_ALPHA,
                                studentize: bool = False,
                                calibration_points: list[dict] | None = None
                                ) -> dict:
    """Freeze the per-stratum conformal quantiles + metadata from `n` Service C
    synthetic curves (the calibration source). Returns the JSON-serialisable
    quantile table the deployed engine applies at test time:

      {
        version, alpha, target_coverage, studentize,
        t50: { marginal_quantile, marginal_n,
               strata: { stratum_key: {quantile, n, thin, one_sided} } },
        regime: { quantile, n, labels },
        ...
      }

    `calibration_points` may be supplied to reuse a pre-built set (the validator
    builds once and splits); otherwise it is generated here. Deterministic given
    (n, seed): the same inputs -> byte-identical quantiles."""
    pts = calibration_points if calibration_points is not None \
        else _build_calibration_points(n, seed)

    # ---- continuous t50: marginal + Mondrian per-stratum quantiles ------------
    # left-only (biased-early) AND non-monotonic (no sigmoidal t50) points are
    # excluded from the t50 channel (see `t50_excluded`); both still feed the regime
    # channel. Pre-`t50_excluded` artifacts fall back to the left_only flag.
    def _t50_ok(p):
        excl = p.get("t50_excluded")
        if excl is None:
            excl = p.get("left_only", False)
        return (not excl) and p.get("t50_hat") is not None
    t50_pts = [p for p in pts if _t50_ok(p)]
    marginal_scores = [s for s in (_t50_nonconformity(p, studentize) for p in t50_pts)
                       if s is not None]
    marginal_q = _conformal_quantile(marginal_scores, alpha)

    by_stratum: dict[str, list[float]] = defaultdict(list)
    stratum_one_sided: dict[str, bool] = {}
    for p in t50_pts:
        s = _t50_nonconformity(p, studentize)
        if s is None:
            continue
        by_stratum[p["stratum"]].append(s)
        # a stratum is one-sided iff it is a right-censored stratum (its quantile
        # corrects only the upper tail of [t50_hat - q, +inf))
        stratum_one_sided[p["stratum"]] = bool(p.get("right_censored"))

    strata_q = {}
    for k, sc_list in by_stratum.items():
        q = _conformal_quantile(sc_list, alpha)
        thin = len(sc_list) < MIN_STRATUM_CALIB
        strata_q[k] = {
            "quantile": (q if not thin else marginal_q),
            "own_quantile": q,
            "n": len(sc_list),
            "thin": bool(thin),
            "fell_back_to_marginal": bool(thin),
            "one_sided": bool(stratum_one_sided.get(k, False)),
        }

    # ---- classification regime: marginal + Mondrian per-(censoring|assay) -------
    # The regime is the TARGET, so the Mondrian axes here are the NON-regime axes
    # (censoring|assay). Conditioning the set-quantile on them lifts coverage in
    # strata where the engine's regime evidence is structurally weaker (e.g. heavily
    # censored curves), instead of one pooled quantile under-covering those cells.
    regime_scores = [s for s in (_regime_nonconformity(p) for p in pts)
                     if s is not None]
    regime_q = _conformal_quantile(regime_scores, alpha)

    reg_by_cax: dict[str, list[float]] = defaultdict(list)
    for p in pts:
        s = _regime_nonconformity(p)
        if s is None:
            continue
        cax = "|".join(p["stratum"].split("|")[:2])    # censoring|assay
        reg_by_cax[cax].append(s)
    regime_strata = {}
    for k, sl in reg_by_cax.items():
        q = _conformal_quantile(sl, alpha)
        thin = len(sl) < MIN_STRATUM_CALIB
        regime_strata[k] = {
            "quantile": (q if not thin else regime_q),
            "own_quantile": q,
            "n": len(sl),
            "thin": bool(thin),
            "fell_back_to_marginal": bool(thin),
        }

    n_skipped = next((p["_n_skipped"] for p in pts if "_n_skipped" in p), 0)
    return {
        "version": CONFORMAL_VERSION,
        "alpha": alpha,
        "target_coverage": round(1.0 - alpha, 4),
        "studentize": bool(studentize),
        "min_stratum_calib": MIN_STRATUM_CALIB,
        "calibration_source": "service_c synthetic ground truth (generate_curve + "
                              "make_series; real engine M1/M2/M4/M5)",
        "n_calibration_curves_requested": n,
        "n_calibration_points_used": len(pts),
        "n_calibration_skipped": int(n_skipped),
        "t50": {
            "score": ("studentized_absolute_residual" if studentize
                      else "absolute_residual"),
            "marginal_quantile": marginal_q,
            "marginal_n": len(marginal_scores),
            "strata": strata_q,
            "note": "right-censored strata are ONE-SIDED: interval [t50_hat - q, +inf) "
                    "because a right-censored t50 is a lower bound; left-only t50 is "
                    "excluded from the t50 channel (biased early, per M4/§S3).",
        },
        "regime": {
            "score": "1 - normalized_akaike_weight(true_regime)",
            "quantile": regime_q,
            "n": len(regime_scores),
            "labels": list(REGIME_LABELS),
            "strata": regime_strata,
            "strata_axes": "censoring_class|assay_endpoint_class",
            "note": "prediction set = {regime : 1 - score(regime) <= quantile}; "
                    "set size > 1 only when the engine's evidence is genuinely split. "
                    "Mondrian-conditional on (censoring|assay); thin cells fall back "
                    "to the marginal quantile.",
        },
        "mondrian_axes": ["censoring_class", "assay_endpoint_class(amyloid|generic)",
                          "descriptive_regime_tier"],
    }


# ============================================================================ #
#  3. TEST-TIME APPLICATION  (pure functions the deployed engine calls)
# ============================================================================ #
def conformal_t50_interval(t50_hat: float, stratum: str, censoring: str,
                           calibration_quantiles: dict) -> dict:
    """Apply a frozen calibration to one test t50. Returns
        {lo, hi, q, target_coverage, stratum_used, one_sided, marginal_fallback}.

    Picks the Mondrian stratum's quantile when it exists and is not thin, else the
    marginal quantile (flagged). Right-censoring -> ONE-SIDED interval
    [t50_hat - q, +inf) (the reported t50 is a lower bound, so the upper end is
    unbounded). Left-only censoring -> the t50 interval is suppressed (returns
    `suppressed: True`), mirroring M4's exclusion of biased-early t50. Never raises;
    a missing/ malformed calibration -> a degraded record, not an exception."""
    out = {"t50_hat": (float(t50_hat) if t50_hat is not None else None),
           "stratum_requested": stratum, "censoring": censoring,
           "target_coverage": (calibration_quantiles or {}).get("target_coverage")}
    if t50_hat is None or not math.isfinite(float(t50_hat)):
        out.update({"available": False, "reason": "no point t50 to anchor an interval"})
        return out
    if censoring in ("left",):
        out.update({"available": False,
                    "reason": "left-only t50 is biased early — interval suppressed "
                              "(consistent with M4)"})
        return out
    t50_block = (calibration_quantiles or {}).get("t50") or {}
    strata = t50_block.get("strata") or {}
    marginal_q = t50_block.get("marginal_quantile")
    right_cens = censoring in ("right", "left_right")

    entry = strata.get(stratum)
    if entry is not None and not entry.get("thin", False) \
            and entry.get("quantile") is not None:
        q = entry["quantile"]
        stratum_used = stratum
        fallback = False
        one_sided = bool(entry.get("one_sided", right_cens))
    else:
        q = marginal_q
        stratum_used = "marginal"
        fallback = True
        one_sided = right_cens
    if q is None:
        out.update({"available": False, "reason": "no calibration quantile available"})
        return out

    hat = float(t50_hat)
    one_sided = bool(one_sided or right_cens)
    if one_sided:
        lo, hi = hat - q, math.inf                 # lower-bound-aware one-sided interval
    else:
        lo, hi = hat - q, hat + q
    out.update({
        "available": True,
        "lo": float(lo),
        "hi": (None if math.isinf(hi) else float(hi)),   # JSON: +inf -> null (open upper)
        "hi_is_infinite": bool(math.isinf(hi)),
        "q": float(q),
        "stratum_used": stratum_used,
        "marginal_fallback": bool(fallback),
        "one_sided": bool(one_sided),
        "width": (None if math.isinf(hi) else float(hi - lo)),
    })
    return out


def conformal_regime_set(scores_by_regime: dict, calibration_quantiles: dict,
                         regime_stratum: str | None = None) -> dict:
    """Apply a frozen regime calibration to one test curve's per-regime scores.
    The PREDICTION SET = every regime whose nonconformity (1 - score) is <= the
    calibration quantile: {r : 1 - score(r) <= q}. Marginal coverage >= 1-alpha.
    Returns {set, set_size, target_coverage, scores, stratum_used}. Set size > 1
    only when the engine's evidence is split across regimes (genuine ambiguity); a
    confident curve yields a singleton. An empty raw set is widened to the argmax so
    we never emit the empty set.

    `regime_stratum` = the (censoring|assay) key; when supplied and non-thin its own
    quantile is used (Mondrian-conditional), else the marginal regime quantile."""
    reg_block = (calibration_quantiles or {}).get("regime") or {}
    marginal_q = reg_block.get("quantile")
    labels = reg_block.get("labels") or list(REGIME_LABELS)
    scores = {r: float((scores_by_regime or {}).get(r, 0.0)) for r in labels}
    out = {"target_coverage": (calibration_quantiles or {}).get("target_coverage"),
           "scores": scores}
    # pick the Mondrian (censoring|assay) quantile when it exists and is not thin
    q, stratum_used, fallback = marginal_q, "marginal", True
    strata = reg_block.get("strata") or {}
    if regime_stratum is not None:
        cax = "|".join(regime_stratum.split("|")[:2])   # tolerate a full 3-axis key
        entry = strata.get(cax)
        if entry is not None and not entry.get("thin", False) \
                and entry.get("quantile") is not None:
            q, stratum_used, fallback = entry["quantile"], cax, False
    if q is None:
        out.update({"set": [], "set_size": 0, "available": False,
                    "reason": "no calibration quantile"})
        return out
    pred = [r for r in labels if (1.0 - scores.get(r, 0.0)) <= q + 1e-12]
    if not pred:                                  # never emit the empty set
        pred = [max(labels, key=lambda r: scores.get(r, 0.0))]
    # a set spanning ALL labels is a vacuous (but still valid >= 1-alpha) prediction:
    # in this stratum the engine's evidence cannot resolve the regime at all. Flag it
    # honestly rather than presenting an all-inclusive set as a confident answer.
    saturated = len(pred) >= len(labels)
    out.update({"set": sorted(pred), "set_size": len(pred), "available": True,
                "quantile": float(q), "stratum_used": stratum_used,
                "marginal_fallback": bool(fallback),
                "regime_unresolvable_in_stratum": bool(saturated)})
    return out


# ============================================================================ #
#  4. VALIDATION  (frozen split -> realized coverage table)
# ============================================================================ #
def _deterministic_split(pts: list[dict], seed: int,
                         val_frac: float = VALIDATION_FRAC):
    """Frozen, seeded proper-calibration / validation split. Deterministic given
    (pts order, seed): a fixed permutation sends `val_frac` to validation. Same
    inputs -> identical split -> byte-identical realized-coverage table (§7)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pts))
    n_val = int(round(val_frac * len(pts)))
    val_idx = set(int(i) for i in idx[:n_val])
    cal = [p for j, p in enumerate(pts) if j not in val_idx]
    val = [p for j, p in enumerate(pts) if j in val_idx]
    return cal, val


def validate_coverage(n: int = 400, seed: int = 0, alpha: float = DEFAULT_ALPHA,
                      studentize: bool = False,
                      calibration_points: list[dict] | None = None) -> dict:
    """Build the synthetic set, split it deterministically into proper-calibration
    / validation, FREEZE the quantiles on calibration only, then measure REALIZED
    coverage + mean width / set size on the held-out validation curves — overall and
    per Mondrian stratum. Realized coverage should be ~>= target (within Monte-Carlo
    slack); that is the proof the layer is valid. Pure / deterministic."""
    pts = calibration_points if calibration_points is not None \
        else _build_calibration_points(n, seed)
    cal, val = _deterministic_split(pts, seed=seed + 7)
    # freeze quantiles on the PROPER-CALIBRATION half only (no validation leakage)
    quant = build_conformal_calibration(n=n, seed=seed, alpha=alpha,
                                        studentize=studentize,
                                        calibration_points=cal)

    # ---- continuous t50 realized coverage -------------------------------------
    def _t50_ok(p):
        excl = p.get("t50_excluded")
        if excl is None:
            excl = p.get("left_only", False)
        return (not excl) and p.get("t50_hat") is not None
    t50_val = [p for p in val if _t50_ok(p)]
    overall = {"covered": 0, "n": 0, "width_sum": 0.0, "width_n": 0,
               "one_sided_n": 0}
    per_stratum: dict[str, dict] = defaultdict(
        lambda: {"covered": 0, "n": 0, "width_sum": 0.0, "width_n": 0,
                 "fallback": 0, "one_sided": False})
    for p in t50_val:
        iv = conformal_t50_interval(p["t50_hat"], p["stratum"], p["censoring_class"],
                                    quant)
        if not iv.get("available"):
            continue
        lo, hi = iv["lo"], iv["hi"]
        hi_eff = math.inf if iv.get("hi_is_infinite") else hi
        covered = (lo <= p["t50_true"] <= hi_eff)
        ps = per_stratum[p["stratum"]]
        overall["n"] += 1
        ps["n"] += 1
        overall["covered"] += int(covered)
        ps["covered"] += int(covered)
        ps["fallback"] += int(iv.get("marginal_fallback", False))
        if iv.get("one_sided"):
            overall["one_sided_n"] += 1
            ps["one_sided"] = True
        w = iv.get("width")
        if w is not None and math.isfinite(w):
            overall["width_sum"] += w
            overall["width_n"] += 1
            ps["width_sum"] += w
            ps["width_n"] += 1

    def _cov(d):
        return (d["covered"] / d["n"]) if d["n"] else None

    def _wid(d):
        return (d["width_sum"] / d["width_n"]) if d["width_n"] else None

    t50_strata_table = {}
    for k, d in sorted(per_stratum.items()):
        t50_strata_table[k] = {
            "realized_coverage": _cov(d),
            "n_validation": d["n"],
            "mean_width": _wid(d),
            "one_sided": bool(d["one_sided"]),
            "used_marginal_fallback": (d["fallback"] > 0),
            "n_fallback": d["fallback"],
        }

    # ---- classification regime realized coverage + mean set size --------------
    reg_overall = {"covered": 0, "n": 0, "size_sum": 0, "ambiguous": 0}
    reg_per_stratum: dict[str, dict] = defaultdict(
        lambda: {"covered": 0, "n": 0, "size_sum": 0})
    for p in val:
        if p.get("regime_true") not in REGIME_LABELS:
            continue
        rs = conformal_regime_set(p.get("regime_scores") or {}, quant,
                                  regime_stratum=p["stratum"])
        if not rs.get("available"):
            continue
        covered = p["regime_true"] in rs["set"]
        reg_overall["n"] += 1
        reg_overall["covered"] += int(covered)
        reg_overall["size_sum"] += rs["set_size"]
        reg_overall["ambiguous"] += int(rs["set_size"] > 1)
        # condition the regime table on the assay+censoring half of the stratum
        # (regime itself is the target, so we group by the non-regime axes)
        cax = "|".join(p["stratum"].split("|")[:2])
        d = reg_per_stratum[cax]
        d["n"] += 1
        d["covered"] += int(covered)
        d["size_sum"] += rs["set_size"]

    reg_strata_table = {}
    for k, d in sorted(reg_per_stratum.items()):
        reg_strata_table[k] = {
            "realized_coverage": (d["covered"] / d["n"]) if d["n"] else None,
            "n_validation": d["n"],
            "mean_set_size": (d["size_sum"] / d["n"]) if d["n"] else None,
        }

    return {
        "version": CONFORMAL_VERSION,
        "alpha": alpha,
        "target_coverage": round(1.0 - alpha, 4),
        "studentize": bool(studentize),
        "n_total_points": len(pts),
        "n_proper_calibration": len(cal),
        "n_validation": len(val),
        "split_frac_validation": VALIDATION_FRAC,
        "t50": {
            "marginal": {
                "realized_coverage": _cov(overall),
                "n_validation": overall["n"],
                "mean_width": _wid(overall),
                "n_one_sided": overall["one_sided_n"],
            },
            "mondrian_strata": t50_strata_table,
        },
        "regime": {
            "marginal": {
                "realized_coverage": ((reg_overall["covered"] / reg_overall["n"])
                                      if reg_overall["n"] else None),
                "n_validation": reg_overall["n"],
                "mean_set_size": ((reg_overall["size_sum"] / reg_overall["n"])
                                  if reg_overall["n"] else None),
                "n_ambiguous_sets": reg_overall["ambiguous"],
                "fraction_ambiguous": ((reg_overall["ambiguous"] / reg_overall["n"])
                                       if reg_overall["n"] else None),
            },
            "mondrian_strata": reg_strata_table,
        },
        "note": "realized coverage measured on a held-out validation split frozen "
                "from the SAME synthetic ground truth; should be ~>= target within "
                "Monte-Carlo slack. Mondrian strata report within-stratum coverage; "
                "thin strata fell back to the marginal quantile at fit time.",
    }


# ============================================================================ #
#  5. FROZEN ARTIFACT + CLI
# ============================================================================ #
def build_report(n: int = 400, seed: int = 0, alpha: float = DEFAULT_ALPHA,
                 studentize: bool = False) -> dict:
    """Assemble the frozen artifact: build the calibration points ONCE, freeze the
    full-set quantiles, and validate on the deterministic held-out split (which
    re-freezes on its own calibration half). One synthetic build -> consistent,
    reproducible numbers."""
    pts = _build_calibration_points(n, seed)         # ONE engine pass; reused below
    calibration = build_conformal_calibration(n=n, seed=seed, alpha=alpha,
                                              studentize=studentize,
                                              calibration_points=pts)
    validation = validate_coverage(n=n, seed=seed, alpha=alpha,
                                   studentize=studentize, calibration_points=pts)
    # NOTE both consumers receive the SAME pre-built points, so the whole artifact
    # costs exactly one synthetic+engine pass (not three).
    return {
        "version": CONFORMAL_VERSION,
        "seed": seed,
        "alpha": alpha,
        "target_coverage": round(1.0 - alpha, 4),
        "studentize": bool(studentize),
        "calibration_quantiles": calibration,
        "validation": validation,
        "guarantee": "split/Mondrian conformal prediction: distribution-free, "
                     "finite-sample marginal coverage >= 1-alpha under exchangeability "
                     "of calibration and test points; Mondrian conditions coverage on "
                     "(censoring, assay endpoint, regime) using engine-produced tags.",
        "honesty": "calibrated on Service C SYNTHETIC ground truth (the only source "
                   "with known t50). Coverage on real CPAD curves holds insofar as the "
                   "pre-registered synthetic noise/censoring law matches the real "
                   "digitized traces — a stated, testable assumption (§6C), not a claim "
                   "of validity on arbitrary real data.",
    }


def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    try:
        return f"{float(v):.{nd}g}"
    except (TypeError, ValueError):
        return str(v)


def _print_summary(report: dict) -> None:
    val = report["validation"]
    tgt = report["target_coverage"]
    print("\n" + "=" * 72)
    print(f"  PRISE Conformal Prediction — {report['version']}  "
          f"(alpha={report['alpha']}, target>={tgt}, studentize={report['studentize']})")
    print(f"  calibration source: Service C synthetic ground truth  seed={report['seed']}")
    print(f"  points: {val['n_total_points']} total -> "
          f"{val['n_proper_calibration']} calib / {val['n_validation']} validation")
    print("=" * 72)

    t = val["t50"]["marginal"]
    print("\n[t50] CONTINUOUS conformal — realized coverage vs target:")
    print(f"  MARGINAL   realized={_fmt(t['realized_coverage'])} (target>={tgt})  "
          f"mean_width={_fmt(t['mean_width'])}  n={t['n_validation']}  "
          f"one_sided={t['n_one_sided']}")
    print("  MONDRIAN per stratum (censoring|assay|regime):")
    for k, d in val["t50"]["mondrian_strata"].items():
        flag = " [FALLBACK->marginal]" if d.get("used_marginal_fallback") else ""
        os_ = " [1-sided]" if d.get("one_sided") else ""
        print(f"    {k:<54} cov={_fmt(d['realized_coverage'])} "
              f"width={_fmt(d['mean_width'])} n={d['n_validation']}{os_}{flag}")

    r = val["regime"]["marginal"]
    print("\n[regime] CLASSIFICATION conformal — prediction sets:")
    print(f"  MARGINAL   realized={_fmt(r['realized_coverage'])} (target>={tgt})  "
          f"mean_set_size={_fmt(r['mean_set_size'])}  "
          f"frac_ambiguous={_fmt(r['fraction_ambiguous'])}  n={r['n_validation']}")
    for k, d in val["regime"]["mondrian_strata"].items():
        print(f"    {k:<40} cov={_fmt(d['realized_coverage'])} "
              f"set_size={_fmt(d['mean_set_size'])} n={d['n_validation']}")
    print("=" * 72 + "\n")


def main(argv=None):
    # Windows consoles default to cp1252; force UTF-8 so any unicode never crashes.
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                 # pragma: no cover
        pass
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="PRISE conformal prediction layer")
    ap.add_argument("--n", type=int, default=400,
                    help="synthetic calibration curves (split into calib/validation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="miscoverage level (default 0.1 -> 90%% target coverage)")
    ap.add_argument("--studentize", action="store_true",
                    help="studentize the t50 residual by the per-fit bootstrap SD")
    ap.add_argument("--output", type=Path,
                    default=root / "data" / "processed" / "service_c_conformal.json")
    args = ap.parse_args(argv)

    print(f"[conformal] building calibration (n={args.n}, seed={args.seed}, "
          f"alpha={args.alpha}) — running the real engine on synthetic truth ...")
    report = build_report(n=args.n, seed=args.seed, alpha=args.alpha,
                          studentize=args.studentize)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[conformal] wrote {args.output}")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
