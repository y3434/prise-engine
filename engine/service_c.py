"""
PRISE — Service C: Validation & Identifiability Engine (de-circularized)
=======================================================================

The synthetic-calibration backbone (PRISE_DESIGN.md §6C "Service C", §4 inference
spine, decision rows C1/C2). Service C is **self-referential** — it judges the
same engine (M1–M7) it is built beside — so it MUST break the loop. It does that
by generating curves from a *known* ground truth, running the real engine on them,
and measuring how well the engine recovers what it cannot have peeked at. Where it
can, it generates from a DIFFERENT forward model than the fitter (C1).

KEY PRINCIPLE (pre-registration, C2): every noise / artifact / censoring constant
and every candidate threshold below is FIXED here as a module constant from
physics / instrument specs — it is NEVER tuned to the data it later judges. Service C
does NOT auto-edit M1–M7 thresholds; it emits calibrated-value RECOMMENDATIONS plus
a report for a separate human-governed step. Everything is seeded → byte-identical
re-runs under a pinned environment (§7).

Build order (so partial completion still delivers value):
  1. synthetic generator           generate_curve / make_series (mechanistic generator is
                                    concentration-dependent: t50 scales with concentration)
  2. parameter-recovery battery     parameter_recovery
  3. data-regime confusion matrices confusion_matrix (single-curve) +
                                    concentration_series_gamma_confusion (γ-scaling over a
                                    multi-concentration FAMILY) + equivalence classes
  4. identifiability validation     identifiability_validation (broadened exemplar set)
  5. calibration report            calibration_report (ECE/Brier + FDR-null WITH a power arm
                                    + threshold recs on an HONEST held-out split, no leakage)
  6. basic de-circularization      mismatched_generator_check (mechanism-id degradation, not R²)

DEFERRED to a later Service-C slice (explicitly flagged, not silently skipped):
  * the FULL external forward model (coarse-grained KMC / Smoluchowski) as the
    mismatched generator — here the mismatch is the Tier-B ODE vs the closed-form
    bank (a real family mismatch, but both live inside PRISE);
  * the BLIND external real-literature reality check (Aβ42/α-syn/insulin/β2m vs
    field consensus) — consumes reserved real proteins, governed separately (§8);
  * the artifact-realism distributional check against real flagged traces;
  * the real held-out / leave-one-study-out synthetic→real transfer measurement.

Usage:
    python engine/service_c.py                       # default battery -> JSON + summary
    python engine/service_c.py --n-trials 200 --seed 0
    python engine/service_c.py --output data/processed/service_c_calibration.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Service C sits ON TOP of the engine and only IMPORTS it (never edits it).
from models import REGISTRY, KINETIC_DESCRIPTIVE
from mechanistic import VARIANT_RATES, simulate_mass_fraction
from m2_fit import fit_one, fit_curve, _metrics
from m3_select import fim_identifiability, t50_of, predict
from m4_features import extract_features, gamma_global
from m5_classify import (
    residual_anomaly,
    benjamini_hochberg,
    classify_descriptive,
    ANOMALY_FDR_ALPHA,
    THRESHOLD_SHARPNESS,
    THRESHOLD_LAG_RATIO,
)
import m1_ingest
import m3_select as _m3

SERVICE_C_VERSION = "service-c-1.0"

# ============================================================================ #
#  PRE-REGISTERED CONSTANTS (C2) — fixed from physics / instrument specs.
#  These are the generator's noise/artifact/censoring law and Service C's own
#  confusability / calibration knobs. They are NOT fitted to any data later judged.
# ============================================================================ #

# --- noise model (ThT amyloid kinetics on a plate reader / digitized figure) --- #
# Multiplicative (heteroscedastic) ThT noise: fluorescence shot + dye/probe
# variation scale with signal, so y_obs ≈ y_true*(1+eps), eps~N(0, SIGMA_MULT).
# ~5% relative is a standard plate-reader ThT well-to-well CV (Meisl et al. 2016
# protocols report few-% replicate scatter on clean runs).
SIGMA_MULT = 0.05
# Additive baseline noise as a fraction of the curve's dynamic range: detector /
# read noise + baseline ThT fluorescence floor, signal-independent. ~2% of range.
SIGMA_ADD_FRAC = 0.02
# Optional slow baseline drift (evaporation / lamp drift / dye bleaching), as a
# fraction of dynamic range across the FULL window. Small and linear.
BASELINE_DRIFT_FRAC = 0.03
# Digitization rounding step (graph-digitization from a literature figure): points
# read off a plot land on a quantization grid. ~1/256 of range (8-bit-ish pixel read).
DIGITIZE_STEP_FRAC = 1.0 / 256.0

# --- sampling / window --- #
N_POINTS_DEFAULT = 30          # typical literature kinetic trace length
# left-censor: drop the leading fraction of the window (assay started mid-lag).
LEFT_CENSOR_FRAC = 0.30
# right-censor: truncate the trailing fraction (stopped before plateau).
RIGHT_CENSOR_FRAC = 0.35

# --- Service C confusability / calibration knobs (pre-registered) --- #
# Two mechanisms are pooled into one equivalence class for a data regime when the
# SELECTOR confuses them at least this often (symmetric confusion rate). 0.30 = a
# selection that is wrong ≥30% of the time cannot be claimed to resolve the pair.
CONFUSABILITY_THRESHOLD = 0.30
# Cap on trials for the single-curve MECHANISTIC AICc confusion sweeps (each curve costs
# 4 stiff ODE fits). The result is a coarse degeneracy map, not a precise rate, so a
# modest count suffices and keeps the whole battery under a few minutes.
MECH_CONFUSION_MAX_TRIALS = 12
# Reliability-diagram bin count for ECE (10 equal-width bins on [0,1]).
N_CALIB_BINS = 10
# Tolerance allowed on the realized FDR vs nominal alpha (finite-sample slack).
FDR_NULL_TOLERANCE = 0.05
# Held-out fraction for the HONEST threshold-recommendation split (C2): a threshold
# is CHOSEN on TRAIN and its separation AUC is REPORTED on a disjoint TEST set, so
# the in-sample optimism (the M1 AUC=1.0 tell) is no longer reported as the headline.
# 0.50 = an even train/test split; the split is seeded -> deterministic.
THRESHOLD_HELDOUT_FRAC = 0.50
# A control claim (FDR-null) is only MEANINGFUL if the same runs test actually has
# power to reject when the model bank genuinely cannot represent the curve. Below
# this realized power (TPR on a bank-unrepresentable alternative) "FDR controlled"
# is no-power, not control — so we flag it. 0.30 = a modest but non-trivial floor.
FDR_POWER_FLOOR = 0.30

# --- mechanistic concentration-series design (pre-registered) --- #
# Map the nominal concentration (µM) to the ODE's monomer units. The Tier-B ODE here
# runs in non-dimensional monomer units; this scale puts a ~few-µM reference family
# in the regime where t50 sits mid-window across a decade (so the t50∝m^(−γ) scaling
# — the mechanistic signal a series exploits — is actually exercised, MUST-FIX 2).
MECH_MTOT_PER_UM = 1.0
# Concentrations (µM) in a synthetic mechanistic FAMILY: 5 points across ~one decade,
# geometric, centred so the middle concentration's t50 lands inside the window.
MECH_SERIES_CONCS_UM = (0.5, 1.0, 2.0, 4.0, 8.0)
# Per-mechanism base rate constants tuned (NOT to judged data — to physics: they set
# the reference half-time mid-window so the decade of concentrations spans the curve)
# so the family DEVELOPS across the window and the γ scaling is visible. Reaction
# orders (nc, n2) stay at the ODE defaults (2.0); the DIFFERENCE in realized γ between
# mechanisms is what a series can exploit (steep nucleation-elongation/sec-nuc scaling
# vs the shallower fragmentation / saturating-secondary scaling).
MECH_SERIES_RATES = {
    "nucleation_elongation": {"kn": 2e-2, "kp": 2e-1},
    "secondary_nucleation":  {"kn": 2e-3, "kp": 2e-1, "k2": 2e-2},
    "fragmentation":         {"kn": 3e-3, "kp": 2e-1, "kminus": 3e-2},
    "saturating_secondary":  {"kn": 2e-3, "kp": 2e-1, "k2": 2e-2, "KM": 0.3},
}
# A γ-based separability statistic pools two mechanisms into one class for the
# concentration-series regime when their per-trial recovered-γ distributions OVERLAP
# this much (Mann–Whitney separation AUC within [0.5±margin] of chance). 0.70 mirrors
# the spirit of CONFUSABILITY_THRESHOLD: γ that separates a pair < this AUC cannot be
# claimed to resolve it.
GAMMA_SEPARABILITY_AUC = 0.70

# HONESTY caveat (surfaced, never buried): the per-mechanism rate constants in
# MECH_SERIES_RATES were hand-picked to set reference half-times mid-window AND to
# be γ-separable. So a measured "the concentration series resolves all mechanisms
# into singletons" is partly a CONSTRUCTION artifact, not a law of nature: the
# γ-separability we report is an UPPER BOUND on real-world discriminability —
# real mechanisms can share γ. Attached to the concentration-series confusion
# output, the narrowing verdict, and the report's validity-ceiling text.
SEPARABILITY_CAVEAT = (
    "the per-mechanism rate constants in MECH_SERIES_RATES were chosen to be "
    "distinguishable; the measured gamma-separability is an UPPER BOUND on "
    "real-world discriminability — real mechanisms can share gamma."
)

# Ordinal confidence -> numeric probability map for ECE/Brier on M5's
# descriptive-regime confidence (pre-registered, not fitted): low/medium/high.
CONFIDENCE_NUMERIC = {"low": 0.30, "medium": 0.65, "high": 0.90}


# ============================================================================ #
#  1. SYNTHETIC GENERATOR  (the core)
# ============================================================================ #
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _draw_params(model_name: str, rng: np.random.Generator) -> dict:
    """Draw a physically-reasonable parameter set for a closed-form model, INSIDE
    the registry bounds but well away from the edges (edge draws are unfittable and
    would unfairly punish recovery). Deterministic given the rng."""
    spec = REGISTRY[model_name]
    pnames = spec["params"]
    # canonical clean curve on a 0..T window, dynamic range ~1, lag in the interior
    x0 = np.linspace(0.0, 40.0, N_POINTS_DEFAULT)
    presets = {
        "logistic":    {"base": 0.0, "amp": 1.0, "k": float(rng.uniform(0.2, 0.6)),
                        "t0": float(rng.uniform(12.0, 28.0))},
        "gompertz":    {"base": 0.0, "amp": 1.0, "k": float(rng.uniform(0.1, 0.3)),
                        "t0": float(rng.uniform(12.0, 26.0))},
        "richards":    {"base": 0.0, "amp": 1.0, "k": float(rng.uniform(0.2, 0.5)),
                        "t0": float(rng.uniform(12.0, 26.0)), "nu": float(rng.uniform(0.5, 3.0))},
        "exponential": {"base": 0.0, "amp": 1.0, "k": float(rng.uniform(0.05, 0.25))},
        "scaling_law": {"a": float(rng.uniform(0.002, 0.02)), "n": float(rng.uniform(1.2, 2.2)),
                        "c": 0.0},
        "lnt":         {"a": 0.0, "b": float(rng.uniform(0.01, 0.03))},
    }
    if model_name in presets:
        return {k: presets[model_name][k] for k in pnames}
    # fallback: use the registry guess on a canonical clean logistic-ish shape
    y0 = REGISTRY["logistic"]["func"](x0, 0.0, 1.0, 0.4, 20.0)
    g = spec["guess"](list(x0), list(y0))
    return {pnames[i]: float(g[i]) for i in range(len(pnames))}


def _draw_mechanistic_params(variant: str, rng: np.random.Generator) -> dict:
    """Draw Tier-B ODE rate constants (linear units) for a mechanism variant,
    centred on the literature-plausible decades used by `fit_mechanistic`."""
    rates = VARIANT_RATES[variant]
    # log10 draws centred on the fitter's own guesses, within its bounds
    centres = {"kn": -3.0, "kp": -1.0, "k2": -2.0, "kminus": -2.0, "KM": -0.5}
    out = {}
    for r in rates:
        out[r] = float(10.0 ** (centres[r] + rng.uniform(-0.5, 0.5)))
    return out


def _t50_from_grid(x: np.ndarray, y: np.ndarray):
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if ymax - ymin < 1e-9:
        return None
    half = ymin + 0.5 * (ymax - ymin)
    for i in range(len(x) - 1):
        if (y[i] - half) * (y[i + 1] - half) <= 0 and y[i + 1] != y[i]:
            r = (half - y[i]) / (y[i + 1] - y[i])
            return float(x[i] + r * (x[i + 1] - x[i]))
    return None


def _infer_regime_from_shape(x: np.ndarray, y: np.ndarray):
    """Derive a shape-based regime label from the GENERATED noise-free curve, mirroring
    M4's interior-inflection test (SHOULD-FIX 7 — do not hard-code 'cooperative_sigmoidal'
    for mechanistic curves that may have negligible lag at high concentration).

    A genuinely cooperative (sigmoidal) transition accelerates then decelerates, so its
    steepest point is INTERIOR and clearly steeper than the endpoints. A purely
    decelerating / saturating rise has its steepest slope at the start -> non-cooperative.
    Returns 'cooperative_sigmoidal', 'gradual_non_cooperative', or None (flat)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if y.size < 5 or (float(np.max(y)) - float(np.min(y))) < 1e-9:
        return None
    dy = np.gradient(y, x)
    i = int(np.argmax(dy))
    edge = max(dy[0], dy[-1], 1e-9)
    has_inflection = (0 < i < len(y) - 1) and (dy[i] > 1.15 * edge)
    return "cooperative_sigmoidal" if has_inflection else "gradual_non_cooperative"


def generate_curve(spec: dict, seed: int) -> dict:
    """Generate one synthetic curve with PRE-REGISTERED noise/artifacts/censoring.

    spec keys (all optional except the generator):
      generator        : "closed_form" | "mechanistic"
      model            : registry model name (closed_form)
      variant          : VARIANT_RATES key (mechanistic)
      params           : explicit params (else drawn deterministically from seed)
      concentration_uM : monomer concentration (mechanistic / metadata); default 20
      n_points         : sample count (default N_POINTS_DEFAULT)
      t_max            : window length (default 40 h)
      censor           : "none" | "left" | "right" | "left_right"
      noise            : bool (default True) — apply the pre-registered noise law
      drift, digitize  : bool toggles for the optional artifacts (default True)

    Returns {x, y_true, y_obs, ground_truth}. Never raises; bad spec -> a flagged
    minimal record so the battery does not crash on one bad draw."""
    rng = _rng(seed)
    gen = spec.get("generator", "closed_form")
    n = int(spec.get("n_points", N_POINTS_DEFAULT))
    t_max = float(spec.get("t_max", 40.0))
    censor = spec.get("censor", "none")
    do_noise = spec.get("noise", True)
    do_drift = spec.get("drift", True)
    do_digit = spec.get("digitize", True)
    conc = float(spec.get("concentration_uM", 20.0))

    x = np.linspace(0.0, t_max, n)

    # ---- ground-truth signal (no noise) -------------------------------------
    if gen == "mechanistic":
        variant = spec.get("variant", "secondary_nucleation")
        params = spec.get("params") or _draw_mechanistic_params(variant, rng)
        kw = {"kn": params.get("kn", 0.0), "kp": params.get("kp", 0.0),
              "k2": params.get("k2", 0.0), "kminus": params.get("kminus", 0.0),
              "KM": params.get("KM", math.inf)}
        # CONCENTRATION-DEPENDENT initial monomer (MUST-FIX 2). The half-time scales
        # with concentration (t50 ∝ m^(−γ)) and THAT scaling is the mechanistic signal
        # a concentration series exploits; holding mtot=1.0 fixed (old behaviour) erased
        # it, so a series was no harder/easier than a single curve. mtot is the nominal
        # concentration in the generator's monomer units (caller passes concentration_uM).
        mtot = conc * MECH_MTOT_PER_UM
        frac = simulate_mass_fraction(x, nc=2.0, n2=2.0, mtot=mtot, **kw)
        y_true = np.asarray(frac, float)
        model_or_mech = variant
        # derive the shape label from the curve itself (SHOULD-FIX 7) rather than
        # asserting sigmoidal — at high concentration the lag can vanish.
        regime_true = _infer_regime_from_shape(x, y_true)
    else:
        model = spec.get("model", "logistic")
        params = spec.get("params") or _draw_params(model, rng)
        order = REGISTRY[model]["params"]
        vals = [params[p] for p in order]
        y_true = np.asarray(REGISTRY[model]["func"](x, *vals), float)
        model_or_mech = model
        # crude shape label used only for descriptive-calibration ground truth
        if model in ("logistic", "gompertz", "richards"):
            regime_true = "cooperative_sigmoidal"
        elif model in ("exponential", "scaling_law"):
            regime_true = "gradual_non_cooperative"
        else:
            regime_true = "gradual_non_cooperative"

    if not np.all(np.isfinite(y_true)):
        return {"x": x.tolist(), "y_true": [], "y_obs": [],
                "ground_truth": {"generator": gen, "model_or_mechanism": model_or_mech,
                                 "status": "non_finite_truth"}}

    rng_lo, rng_hi = float(np.min(y_true)), float(np.max(y_true))
    dyn = (rng_hi - rng_lo) or 1.0
    t50_true = _t50_from_grid(x, y_true)
    has_lag = bool(regime_true == "cooperative_sigmoidal")

    # ---- pre-registered artifacts + noise -----------------------------------
    y_obs = y_true.copy()
    if do_drift:
        # linear baseline drift across the window (sign random, magnitude fixed)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        y_obs = y_obs + sign * BASELINE_DRIFT_FRAC * dyn * (x / (t_max or 1.0))
    if do_noise:
        eps_mult = rng.normal(0.0, SIGMA_MULT, size=n)      # heteroscedastic ThT
        eps_add = rng.normal(0.0, SIGMA_ADD_FRAC * dyn, size=n)  # additive baseline
        y_obs = y_obs * (1.0 + eps_mult) + eps_add
    if do_digit:
        step = DIGITIZE_STEP_FRAC * dyn
        if step > 0:
            y_obs = np.round(y_obs / step) * step           # graph-digitization grid

    # ---- censoring (truncate the observed window) ---------------------------
    # matches M1's censoring_class taxonomy: left = started after lag; right =
    # stopped before plateau; left_right = both.
    lo_i, hi_i = 0, n
    if censor in ("left", "left_right"):
        lo_i = int(round(LEFT_CENSOR_FRAC * n))
    if censor in ("right", "left_right"):
        hi_i = n - int(round(RIGHT_CENSOR_FRAC * n))
    lo_i = max(0, min(lo_i, n - 4))
    hi_i = max(lo_i + 4, min(hi_i, n))
    xs = x[lo_i:hi_i]
    yt = y_true[lo_i:hi_i]
    yo = y_obs[lo_i:hi_i]

    ground_truth = {
        "generator": gen,
        "model_or_mechanism": model_or_mech,
        "params": {k: float(v) for k, v in params.items() if math.isfinite(float(v))},
        "t50_true": t50_true,
        "has_lag": has_lag,
        "regime_true": regime_true,
        "concentration_uM": conc,
        "censor_requested": censor,
        "n_points": int(len(xs)),
        "dynamic_range_true": float(dyn),
    }
    return {"x": xs.tolist(), "y_true": yt.tolist(), "y_obs": yo.tolist(),
            "ground_truth": ground_truth}


def make_series(curve: dict, series_id: str = "syn",
                concentration_uM: float | None = None,
                assay_mass: bool = True, agit=None, seeded=None,
                pdb=None, concentration_series_id=None) -> dict:
    """Wrap a generated curve into the AggregationSeries shape M1/M2/M4/M5 consume,
    then run M1 triage so the `m1` block (y_processed, censoring_class, handling) is
    EXACTLY what the real engine produces. Mirrors a real curves_triaged.jsonl record."""
    gt = curve.get("ground_truth", {})
    conc = concentration_uM if concentration_uM is not None else gt.get("concentration_uM", 20.0)
    series = {
        "series_id": series_id,
        "data_mode": "kinetic",
        "concentration_series_id": concentration_series_id,
        "x_hours": [float(v) for v in curve["x"]],
        "y_intensity": [float(v) for v in curve["y_obs"]],
        "condition_vector": {
            "assay_reports_mass": assay_mass,
            "assay_type": "ThT" if assay_mass else "Turbidity",
            "concentration": {"value_uM": float(conc), "unit": "microM"},
            "agitation": agit, "seeded": seeded,
            "construct_id": "synthetic", "pdb_id": pdb,
            "field_provenance": {
                "agitation": "known" if agit is not None else "unknown",
                "seeded": "known" if seeded is not None else "unknown",
            },
        },
        "digitization_uncertainty": True,
        "_ground_truth": gt,   # carried for Service C scoring; ignored by the engine
    }
    # run the REAL triage so downstream modules see an authentic m1 block
    return m1_ingest.triage_series(series)


# ============================================================================ #
#  2. PARAMETER-RECOVERY BATTERY
# ============================================================================ #
def parameter_recovery(model_name: str, n_trials: int = 100, seed: int = 0,
                       t_max: float = 40.0, n_points: int = N_POINTS_DEFAULT) -> dict:
    """Draw known params, generate noisy curves, fit BACK with m2_fit.fit_one, and
    report per-parameter bias / RMSE / 95%-CI coverage (using param_se) plus t50
    recovery. Coverage uses a normal Wald interval theta_hat ± 1.96*se."""
    spec = REGISTRY[model_name]
    pnames = spec["params"]
    err = {p: [] for p in pnames}
    cover = {p: [] for p in pnames}
    t50_err = []
    n_ok = 0
    for i in range(n_trials):
        s = seed * 100003 + i
        rng = _rng(s)
        true_params = _draw_params(model_name, rng)
        curve = generate_curve(
            {"generator": "closed_form", "model": model_name,
             "params": true_params, "n_points": n_points, "t_max": t_max,
             "censor": "none"}, seed=s + 1)
        x, y = curve["x"], curve["y_obs"]
        if len(x) < len(pnames) + 1:
            continue
        fit = fit_one(x, y, model_name)
        if not fit.get("converged"):
            continue
        n_ok += 1
        rp = fit["params"]
        se = fit.get("param_se") or {}
        for p in pnames:
            e = rp[p] - true_params[p]
            err[p].append(e)
            s_e = se.get(p)
            if s_e is not None and math.isfinite(s_e) and s_e > 0:
                cover[p].append(abs(e) <= 1.96 * s_e)
        # t50 recovery: compare fitted-model t50 to the true-model t50 on the window
        t50_hat = t50_of(model_name, rp, x)
        t50_tru = curve["ground_truth"].get("t50_true")
        if t50_hat is not None and t50_tru is not None:
            t50_err.append(t50_hat - t50_tru)

    def _summ(es, cov):
        if not es:
            return {"bias": None, "rmse": None, "coverage_95": None, "n": 0}
        a = np.asarray(es, float)
        return {"bias": float(np.mean(a)), "rmse": float(np.sqrt(np.mean(a ** 2))),
                "coverage_95": (float(np.mean(cov)) if cov else None), "n": len(es)}

    per = {p: _summ(err[p], cover[p]) for p in pnames}
    t50_rec = _summ(t50_err, [])
    t50_rec.pop("coverage_95", None)
    return {"model": model_name, "n_trials": n_trials, "n_converged": n_ok,
            "per_parameter": per, "t50_recovery": t50_rec,
            "note": "coverage_95 = fraction of Wald 95% CIs (theta_hat±1.96·se) "
                    "covering truth; nominal 0.95",
            "caveat": "Wald 95% CIs from a BOUNDED curve_fit are APPROXIMATE, especially "
                      "near a bound where the sampling distribution is non-normal and the "
                      "symmetric ±1.96·se interval can under-cover; draws here are kept "
                      "interior (SHOULD-FIX 5) to limit this, but coverage at/near bounds "
                      "should be read as approximate, not exact."}


# ============================================================================ #
#  3. DATA-REGIME-CONDITIONAL CONFUSION / DEGENERACY MATRICES  (K2)
# ============================================================================ #
# Generators per data regime. single_curve & saturating_secondary use a small
# bank that mixes closed-form descriptive shapes; the mechanistic Tier-B variants
# are added under the regimes where a series can constrain them.
_CLOSED_FORM_GENERATORS = ["logistic", "gompertz", "richards", "exponential"]
_MECH_GENERATORS = list(VARIANT_RATES.keys())


def _select_closed_form(x, y) -> str | None:
    """The engine's single-curve descriptive selection: AICc-best over the licensed
    closed-form bank (mirrors fit_curve/M3 point selection, restricted to descriptive
    models so a sloppy mechanistic closed form cannot masquerade — §3-M3 tiers)."""
    best, best_aicc = None, float("inf")
    for name in KINETIC_DESCRIPTIVE:
        r = fit_one(x, y, name)
        if r.get("converged") and r.get("aicc") is not None and r["aicc"] < best_aicc:
            best_aicc, best = r["aicc"], name
    return best


def _select_mechanistic_variant(x, y) -> str | None:
    """Tier-B selection: AICc-best ODE variant. ODE fits are stiff/slow, so this is
    used only for the mechanistic regimes and with a modest trial budget."""
    from mechanistic import fit_mechanistic
    best, best_aicc = None, float("inf")
    for variant in _MECH_GENERATORS:
        r = fit_mechanistic(x, y, variant=variant)
        if r.get("converged") and r.get("aicc") is not None and r["aicc"] < best_aicc:
            best_aicc, best = r["aicc"], variant
    return best


def _equivalence_classes_from_adjacency(labels: list[str], adj: dict) -> list:
    """Connected-components grouping on a 'confusable' adjacency graph.

    SHOULD-FIX 8 (documented choice): connected components can OVER-MERGE via transitive
    chaining — if A~B and B~C but A is well-resolved from C, all three still land in one
    class. We keep connected components on purpose: an equivalence class is meant to be
    the set a single-mechanism call cannot safely pick WITHIN, and a chain A~B~C means no
    single boundary cleanly separates the ends either, so the conservative (merge) choice
    is the right default for a degeneracy map. The alternative — grouping only mutually-
    confused PAIRS — is available by reading the pairwise `confusable_pairs` we also emit."""
    seen, classes = set(), []
    for g in labels:
        if g in seen:
            continue
        stack, comp = [g], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            stack.extend(adj[u] - seen)
        classes.append(sorted(comp))
    return classes


def confusion_matrix(data_regime: str, generators: list[str] | None = None,
                     n_trials: int = 50, seed: int = 0) -> dict:
    """Per TRUE generating mechanism/model, generate curves at `data_regime`, run the
    engine's selection, and tally which model/mechanism is SELECTED -> a row-stochastic
    confusion matrix. Then derive equivalence classes by pooling mechanisms whose
    pairwise (symmetric) confusion exceeds CONFUSABILITY_THRESHOLD.

    Regimes:
      single_curve              — one closed-form descriptive curve, closed-form selection
      single_concentration_mech — Tier-B variants, ONE single-concentration curve each,
                                  mechanistic AICc selection (the broad single-curve
                                  collapse; the BASELINE the concentration series improves on)
      saturating_secondary      — Tier-B variants incl. finite-K_M, mechanistic selection
                                  (saturating secondary re-introduces degeneracy, K2)

    NOTE: the `concentration_series` regime is handled by
    `concentration_series_gamma_confusion` (a γ-scaling separability discriminator over a
    multi-concentration family), NOT by this single-curve AICc tally.
    """
    if generators is None:
        if data_regime == "single_curve":
            generators = _CLOSED_FORM_GENERATORS
            mech_regime = False
        else:
            generators = _MECH_GENERATORS
            mech_regime = True
    else:
        mech_regime = data_regime in ("concentration_series", "saturating_secondary",
                                      "single_concentration_mech")

    counts = {g: Counter() for g in generators}
    for gi, gtrue in enumerate(generators):
        for i in range(n_trials):
            s = seed * 7919 + gi * 1009 + i
            if mech_regime:
                # saturating regime forces a finite K_M on every secondary variant
                params = None
                if data_regime == "saturating_secondary" and gtrue in (
                        "secondary_nucleation", "saturating_secondary"):
                    rng = _rng(s)
                    base = _draw_mechanistic_params("saturating_secondary", rng)
                    base["KM"] = float(10.0 ** rng.uniform(-1.0, 0.0))  # finite, saturating
                    params = base
                cspec = {"generator": "mechanistic", "variant": gtrue,
                         "params": params, "n_points": N_POINTS_DEFAULT,
                         "t_max": 40.0, "censor": "none"}
            else:
                cspec = {"generator": "closed_form", "model": gtrue,
                         "n_points": N_POINTS_DEFAULT, "t_max": 40.0, "censor": "none"}
            curve = generate_curve(cspec, seed=s + 1)
            x, y = curve["x"], curve["y_obs"]
            if len(x) < 6:
                continue
            sel = (_select_mechanistic_variant(x, y) if mech_regime
                   else _select_closed_form(x, y))
            counts[gtrue][sel or "none"] += 1

    # row-stochastic matrix
    labels = list(generators)
    matrix = {}
    for gtrue in labels:
        tot = sum(counts[gtrue].values()) or 1
        matrix[gtrue] = {sel: counts[gtrue].get(sel, 0) / tot for sel in labels}
        matrix[gtrue]["_other"] = sum(
            v for k, v in counts[gtrue].items() if k not in labels) / tot

    # equivalence classes: pool i,j when symmetric confusion exceeds the threshold.
    # symmetric confusion = max(P(select j | true i), P(select i | true j)).
    adj = {g: set() for g in labels}
    confusable_pairs = []
    for ai, a in enumerate(labels):
        for b in labels[ai + 1:]:
            cab = matrix[a].get(b, 0.0)
            cba = matrix[b].get(a, 0.0)
            if max(cab, cba) >= CONFUSABILITY_THRESHOLD:
                adj[a].add(b)
                adj[b].add(a)
                confusable_pairs.append(sorted([a, b]))
    classes = _equivalence_classes_from_adjacency(labels, adj)

    diag = [matrix[g].get(g, 0.0) for g in labels]
    return {
        "data_regime": data_regime,
        "generators": labels,
        "selector": "mechanistic_aicc" if mech_regime else "closed_form_descriptive_aicc",
        "n_trials_per_generator": n_trials,
        "matrix": matrix,
        "mean_diagonal": float(np.mean(diag)) if diag else None,
        "min_diagonal": float(np.min(diag)) if diag else None,
        "equivalence_classes": classes,
        "confusable_pairs": confusable_pairs,   # mutually-confused PAIRS (pre-merge; SHOULD-FIX 8)
        "confusability_threshold": CONFUSABILITY_THRESHOLD,
        "note": "row-stochastic: row=TRUE generator, col=SELECTED. Equivalence classes "
                "pool generators with symmetric confusion >= threshold (the K2 "
                "data-regime-conditional degeneracy map M5 needs, replacing hard-coded pairs).",
    }


def _generate_mech_family(variant: str, trial_seed: int) -> list[dict]:
    """Generate ONE multi-concentration FAMILY (MECH_SERIES_CONCS_UM) for a true
    mechanism and wrap each member through make_series so it carries the concentration
    and the authentic m1 block the dual-γ machinery consumes. The per-mechanism rate
    constants are fixed (MECH_SERIES_RATES); concentration enters via mtot in the
    generator, so the FAMILY exhibits the mechanism's t50∝m^(−γ) scaling."""
    rates = MECH_SERIES_RATES[variant]
    members = []
    for j, conc in enumerate(MECH_SERIES_CONCS_UM):
        curve = generate_curve(
            {"generator": "mechanistic", "variant": variant, "params": dict(rates),
             "concentration_uM": conc, "n_points": N_POINTS_DEFAULT,
             "t_max": 40.0, "censor": "none"}, seed=trial_seed * 100 + j)
        members.append(make_series(
            curve, series_id=f"{variant}-{trial_seed}-{j}", concentration_uM=conc,
            concentration_series_id=f"{variant}-{trial_seed}"))
    return members


def concentration_series_gamma_confusion(generators: list[str] | None = None,
                                         n_trials: int = 12, seed: int = 0) -> dict:
    """MUST-FIX 2: the concentration_series regime actually USES a concentration series.

    For each TRUE mechanism, generate a multi-concentration FAMILY per trial, recover γ
    with the engine's GLOBAL master-curve collapse (m4_features.gamma_global — the
    dual-γ machinery), and test whether γ SEPARATES the mechanisms. Two mechanisms are
    pooled into one equivalence class when their per-trial γ distributions are NOT
    separable: the rank-based (Mann–Whitney) separation AUC sits within the chance band
    [1 - GAMMA_SEPARABILITY_AUC, GAMMA_SEPARABILITY_AUC]. This is the γ-scaling
    separability discriminator (clearly labelled — we use gamma_global over the family,
    NOT a full per-curve mechanistic global fit, to keep the battery runtime sane).

    The point is to TEST the K2 claim that a clean concentration series can break
    sec-nuc/frag via γ while saturating secondary re-introduces degeneracy — not assert
    it. Whether γ NARROWS the degeneracy is read off by comparing these classes to the
    single-curve / single-concentration mechanistic classes (done in run_battery)."""
    if generators is None:
        generators = list(MECH_SERIES_RATES.keys())
    gammas = {g: [] for g in generators}      # per-mechanism recovered-γ samples
    for gi, gtrue in enumerate(generators):
        for i in range(n_trials):
            ts = seed * 7919 + gi * 1009 + i + 1
            members = _generate_mech_family(gtrue, ts)
            res = gamma_global(members, B=0)   # no bootstrap -> fast; point γ only
            if res.get("status") == "ok" and res.get("gamma") is not None \
                    and math.isfinite(res["gamma"]):
                gammas[gtrue].append(float(res["gamma"]))

    labels = list(generators)
    gamma_summary = {
        g: ({"mean": float(np.mean(gammas[g])), "std": float(np.std(gammas[g])),
             "n": len(gammas[g])} if gammas[g] else {"mean": None, "std": None, "n": 0})
        for g in labels}

    # pairwise γ separability (rank AUC) -> adjacency of NON-separable (still-degenerate) pairs
    pair_auc = {}
    adj = {g: set() for g in labels}
    confusable_pairs = []
    for ai, a in enumerate(labels):
        for b in labels[ai + 1:]:
            ga, gb = np.asarray(gammas[a], float), np.asarray(gammas[b], float)
            if len(ga) < 2 or len(gb) < 2:
                continue
            # AUC = P(γ_a > γ_b); fold to [0.5,1] so it measures SEPARATION either way
            auc = _rank_auc(ga, gb, positive_is_high=True)
            sep = max(auc, 1.0 - auc)
            pair_auc[f"{a}|{b}"] = float(sep)
            if sep < GAMMA_SEPARABILITY_AUC:        # γ does NOT resolve this pair
                adj[a].add(b)
                adj[b].add(a)
                confusable_pairs.append(sorted([a, b]))
    classes = _equivalence_classes_from_adjacency(labels, adj)

    return {
        "data_regime": "concentration_series",
        "generators": labels,
        "selector": "gamma_global_scaling_separability",
        "discriminator": "m4_features.gamma_global over a multi-concentration family "
                         f"({len(MECH_SERIES_CONCS_UM)} concentrations across a decade); "
                         "γ-scaling separability statistic (rank AUC), NOT a full "
                         "per-curve mechanistic global fit (kept out for runtime)",
        "n_trials_per_generator": n_trials,
        "concentrations_uM": list(MECH_SERIES_CONCS_UM),
        "gamma_by_mechanism": gamma_summary,
        "pairwise_separation_auc": pair_auc,
        "gamma_separability_auc_threshold": GAMMA_SEPARABILITY_AUC,
        "equivalence_classes": classes,
        "confusable_pairs": confusable_pairs,
        "separability_caveat": SEPARABILITY_CAVEAT,
        "note": "per-mechanism recovered-γ over a real concentration FAMILY. Pairs whose "
                "γ distributions separate (rank AUC >= threshold) are RESOLVED by the "
                "series; pairs that overlap stay in one equivalence class. Compare these "
                "classes to single_curve to see whether the series NARROWS the degeneracy.",
    }


def _class_sizes(classes: list) -> dict:
    return {"n_classes": len(classes),
            "largest_class": (max((len(c) for c in classes), default=0)),
            "classes": [sorted(c) for c in classes]}


def _series_narrowing(single_conc: dict, series: dict) -> dict:
    """Honest K2 test (MUST-FIX 2): does the concentration SERIES narrow the degeneracy
    relative to a single concentration? Compares the two equivalence-class partitions of
    the same mechanism set. The series 'narrows' iff it produces MORE classes (or a
    smaller largest class) — i.e. γ split at least one pair the single curve could not.
    Reports both partitions and which pairs the series newly resolved, with no spin: if
    γ does not help at this design, it says so."""
    sc_classes = single_conc.get("equivalence_classes", [])
    se_classes = series.get("equivalence_classes", [])

    def _pair_set(classes):
        # set of unordered pairs that are POOLED (in the same class)
        pairs = set()
        for c in classes:
            cs = sorted(c)
            for a in range(len(cs)):
                for b in range(a + 1, len(cs)):
                    pairs.add((cs[a], cs[b]))
        return pairs

    sc_pairs, se_pairs = _pair_set(sc_classes), _pair_set(se_classes)
    newly_resolved = sorted([list(p) for p in (sc_pairs - se_pairs)])   # pooled→split by γ
    newly_merged = sorted([list(p) for p in (se_pairs - sc_pairs)])     # (should be empty)
    sc_sz, se_sz = _class_sizes(sc_classes), _class_sizes(se_classes)
    narrows = (se_sz["n_classes"] > sc_sz["n_classes"]
               or se_sz["largest_class"] < sc_sz["largest_class"])
    if narrows:
        verdict = ("YES — the concentration series narrows the degeneracy: γ-scaling "
                   "resolved pair(s) the single concentration could not "
                   f"({newly_resolved}). Consistent with K2 (clean series breaks some "
                   "mechanism pairs). CAVEAT: " + SEPARABILITY_CAVEAT)
    else:
        verdict = ("NO — at this design the concentration series does NOT narrow the "
                   "degeneracy beyond the single concentration (γ did not separate any "
                   "additional pair). Reported honestly, not asserted.")
    return {
        "single_concentration_partition": sc_sz,
        "concentration_series_partition": se_sz,
        "pairs_newly_resolved_by_series": newly_resolved,
        "pairs_newly_merged_by_series": newly_merged,
        "series_narrows_degeneracy": bool(narrows),
        "verdict": verdict,
        "separability_caveat": SEPARABILITY_CAVEAT,
        "method": "compare equivalence-class partitions of the SAME Tier-B mechanism set: "
                  "single-concentration mechanistic-AICc confusion vs γ-scaling "
                  "separability over a multi-concentration family.",
    }


# ============================================================================ #
#  4. IDENTIFIABILITY VALIDATION
# ============================================================================ #
def _identifiability_exemplars(s: int):
    """Build the per-trial exemplar specs for identifiability validation (SHOULD-FIX 6 —
    broadened beyond the original 2 so TPR/FPR are estimated on MORE than 2 points). Each
    is (exemplar_name, truth_label, model, generate_curve spec). truth_label='identifiable'
    means fim_identifiability SHOULD NOT flag; 'non_identifiable' means it SHOULD flag."""
    rng = _rng(s)
    out = []
    # --- IDENTIFIABLE exemplars (well-conditioned; should NOT be flagged) ---
    # clean logistic, full window, low noise
    out.append(("clean_logistic", "identifiable", "logistic",
                {"generator": "closed_form", "model": "logistic",
                 "params": _draw_params("logistic", _rng(s + 1)),
                 "n_points": N_POINTS_DEFAULT, "t_max": 40.0, "censor": "none"}))
    # clean gompertz, full window (a second genuinely-identifiable shape — SHOULD-FIX 6)
    out.append(("clean_gompertz", "identifiable", "gompertz",
                {"generator": "closed_form", "model": "gompertz",
                 "params": _draw_params("gompertz", _rng(s + 2)),
                 "n_points": N_POINTS_DEFAULT, "t_max": 40.0, "censor": "none"}))
    # --- NON-IDENTIFIABLE / sloppy exemplars (should BE flagged) ---
    # Richards near the logistic limit (nu≈1) on a short, right-censored window -> the
    # (k, nu) shape combination is unconstrained (the original sloppy combo)
    pr = _draw_params("richards", _rng(s + 3))
    pr["nu"] = 1.0 + float(rng.uniform(-0.05, 0.05))
    out.append(("richards_nu1_short", "non_identifiable", "richards",
                {"generator": "closed_form", "model": "richards", "params": pr,
                 "n_points": 12, "t_max": 40.0, "censor": "right"}))
    # near-degenerate (almost flat) exponential: tiny k over a short window -> base/amp/k
    # are barely distinguishable (a near-degenerate exponential — SHOULD-FIX 6)
    out.append(("near_flat_exponential", "non_identifiable", "exponential",
                {"generator": "closed_form", "model": "exponential",
                 "params": {"base": 0.0, "amp": 1.0,
                            "k": float(_rng(s + 4).uniform(0.002, 0.01))},
                 "n_points": 10, "t_max": 40.0, "censor": "right"}))
    return out


def identifiability_validation(n_trials: int = 40, seed: int = 0) -> dict:
    """Cross-check m3_select.fim_identifiability's practical-(non)identifiability flag
    against ground truth, over a BROADENED exemplar set (SHOULD-FIX 6):
      IDENTIFIABLE truth   — clean logistic; clean gompertz (well-conditioned).
      NON-IDENTIFIABLE truth — Richards@nu≈1 on a short window (sloppy (k,nu) combo);
                               a near-flat exponential (near-degenerate base/amp/k).
    Reports overall TPR/FPR AND a per-exemplar flag rate, so the rates rest on >2 points."""
    rows = []           # (truth_label, flagged)
    per = defaultdict(lambda: {"truth": None, "flagged": 0, "n": 0})
    for i in range(n_trials):
        s = seed * 131 + i
        for name, truth, model, spec in _identifiability_exemplars(s):
            c = generate_curve(spec, seed=s + 7)
            fit = fit_one(c["x"], c["y_obs"], model)
            if not fit.get("converged"):
                continue
            ident = fim_identifiability(model, fit["params"], c["x"], c["y_obs"])
            flagged = ident["flag"] in ("practically_non_identifiable",
                                        "structurally_non_identifiable")
            rows.append((truth, flagged))
            per[name]["truth"] = truth
            per[name]["flagged"] += int(flagged)
            per[name]["n"] += 1

    conf = {"identifiable": {"flagged": 0, "not_flagged": 0},
            "non_identifiable": {"flagged": 0, "not_flagged": 0}}
    for truth, flagged in rows:
        conf[truth]["flagged" if flagged else "not_flagged"] += 1
    n_id = conf["identifiable"]["flagged"] + conf["identifiable"]["not_flagged"]
    n_ni = conf["non_identifiable"]["flagged"] + conf["non_identifiable"]["not_flagged"]
    per_exemplar = {k: {"truth": v["truth"], "n": v["n"],
                        "flag_rate": (v["flagged"] / v["n"]) if v["n"] else None}
                    for k, v in per.items()}
    return {
        "n_trials": n_trials,
        "confusion": conf,
        "per_exemplar": per_exemplar,
        "false_positive_rate": (conf["identifiable"]["flagged"] / n_id) if n_id else None,
        "true_positive_rate": (conf["non_identifiable"]["flagged"] / n_ni) if n_ni else None,
        "note": "false-positive = a well-conditioned identifiable exemplar (clean "
                "logistic/gompertz) wrongly flagged; true-positive = a sloppy/near-"
                "degenerate exemplar (Richards@nu≈1, near-flat exponential) correctly "
                "flagged. Broadened beyond 2 exemplars (SHOULD-FIX 6); cond cutoff = "
                f"m3.COND_NUMBER_NONIDENT ({_m3.COND_NUMBER_NONIDENT}).",
    }


# ============================================================================ #
#  5. CALIBRATION REPORT  (ECE/Brier + FDR-null + threshold recommendations)
# ============================================================================ #
def _ece_brier(pred_prob: list[float], correct: list[bool], n_bins: int = N_CALIB_BINS):
    """Expected Calibration Error + Brier score on (predicted-prob, correctness) pairs.
    ECE = sum_b (|bin|/N) * |acc_b - conf_b| over equal-width bins."""
    if not pred_prob:
        return {"ece": None, "brier": None, "n": 0, "bins": []}
    p = np.asarray(pred_prob, float)
    c = np.asarray(correct, float)
    brier = float(np.mean((p - c) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    N = len(p)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & (p < hi) if b < n_bins - 1 else (p >= lo) & (p <= hi)
        nb = int(mask.sum())
        if nb == 0:
            continue
        acc_b = float(c[mask].mean())
        conf_b = float(p[mask].mean())
        ece += (nb / N) * abs(acc_b - conf_b)
        bins.append({"lo": float(lo), "hi": float(hi), "n": nb,
                     "accuracy": acc_b, "confidence": conf_b})
    return {"ece": float(ece), "brier": brier, "n": N, "bins": bins}


def _descriptive_calibration(n_trials: int, seed: int) -> dict:
    """ECE/Brier for M5's descriptive-regime confidence. Generate curves with a known
    regime, run classify_descriptive on the engine's own fit/feature output, map the
    confidence LEVEL to a numeric probability, and bin predicted-confidence vs whether
    the descriptive regime was correct."""
    # generators with a clean known regime label
    bank = [("logistic", "cooperative_sigmoidal"),
            ("gompertz", "cooperative_sigmoidal"),
            ("richards", "cooperative_sigmoidal"),
            ("exponential", "gradual_non_cooperative"),
            ("scaling_law", "gradual_non_cooperative")]
    probs, correct = [], []
    for i in range(n_trials):
        s = seed * 311 + i
        rng = _rng(s)
        model, regime_true = bank[i % len(bank)]
        curve = generate_curve({"generator": "closed_form", "model": model,
                                "n_points": N_POINTS_DEFAULT, "t_max": 40.0,
                                "censor": "none"}, seed=s + 1)
        series = make_series(curve, series_id=f"calib-{i}")
        fr = fit_curve(series)
        feat = extract_features(series, fit_result=fr)
        desc = classify_descriptive(series.get("m1", {}), feat)
        # confidence proxy: M5 _confidence uses R²+anomaly; here we reuse the same
        # signal — high R² clean fit -> high. Use the feature R² to assign the level.
        r2 = feat.get("r2")
        if r2 is None or feat.get("status") != "ok":
            level = "low"
        elif r2 >= 0.95:
            level = "high"
        elif r2 >= 0.80:
            level = "medium"
        else:
            level = "low"
        probs.append(CONFIDENCE_NUMERIC[level])
        correct.append(bool(desc["regime"] == regime_true))
    out = _ece_brier(probs, correct)
    out["target"] = ("M5 descriptive-regime confidence vs empirical correctness on "
                     "synthetic ground truth")
    out["confidence_numeric_map"] = CONFIDENCE_NUMERIC
    out["empirical_accuracy"] = (float(np.mean(correct)) if correct else None)
    return out


def _bank_unrepresentable_curve(seed: int, n_points: int = N_POINTS_DEFAULT,
                                t_max: float = 40.0) -> dict:
    """ALTERNATIVE-arm generator for the FDR power check: a DOUBLE-SIGMOID (two
    well-separated transitions). The closed-form bank (M2) can fit only a SINGLE
    monotone sigmoid, so it must leave a long, systematic, same-sign residual stretch
    over the second transition — exactly the structured lack-of-fit the Wald–Wolfowitz
    runs test ('too few runs') is meant to catch. This is a known mechanism-misfit the
    bank cannot represent, so a CORRECT anomaly detector SHOULD reject here (giving the
    null check a power arm). Carries the SAME pre-registered noise/artifact law as the
    null curves so the only difference is the unmodelled structure. Deterministic."""
    rng = _rng(seed)
    x = np.linspace(0.0, t_max, n_points)
    # two transitions at ~1/4 and ~3/4 of the window, each carrying half the range
    b1 = 0.5 / (1.0 + np.exp(-0.8 * (x - 0.25 * t_max)))
    b2 = 0.5 / (1.0 + np.exp(-0.8 * (x - 0.75 * t_max)))
    y_true = b1 + b2
    dyn = float(np.max(y_true) - np.min(y_true)) or 1.0
    # reuse the pre-registered noise law (mult + additive), no artifact toggles needed
    eps_mult = rng.normal(0.0, SIGMA_MULT, size=n_points)
    eps_add = rng.normal(0.0, SIGMA_ADD_FRAC * dyn, size=n_points)
    y_obs = y_true * (1.0 + eps_mult) + eps_add
    gt = {"generator": "bank_unrepresentable", "model_or_mechanism": "double_sigmoid",
          "regime_true": None, "concentration_uM": 20.0}
    return {"x": x.tolist(), "y_true": y_true.tolist(), "y_obs": y_obs.tolist(),
            "ground_truth": gt}


def _runs_test_pvalues(curves_specs, label_prefix: str) -> list:
    """Run M1 triage + residual_anomaly on each generated curve, returning the
    too-few-runs p-value (or None where untestable). Shared by both FDR arms so the
    null and the alternative go through the SAME detector path."""
    pvals = []
    for i, curve in enumerate(curves_specs):
        series = make_series(curve, series_id=f"{label_prefix}-{i}")
        a = residual_anomaly(series)
        if a.get("testable") and a.get("p_too_few_runs") is not None:
            pvals.append(a["p_too_few_runs"])
        else:
            pvals.append(None)
    return pvals


def fdr_null_check(n_trials: int, seed: int) -> dict:
    """FDR check with BOTH a NULL arm and a POWER (alternative) arm.

    * NULL arm — curves the bank CAN represent (residuals are pure pre-registered
      noise). Under the null EVERY BH rejection is false, so realized FDP =
      n_rejected/n_tested; control requires FDP <= alpha + tolerance.
    * POWER arm — curves the bank CANNOT represent (double-sigmoid mechanism-misfit),
      where the runs test SHOULD reject. Power (TPR) = fraction of alternative curves
      flagged at the RAW nominal alpha (per-curve sensitivity of the detector).

    A control claim is only MEANINGFUL if power > FDR_POWER_FLOOR: 'FDR controlled'
    with zero power is NO POWER, not control. Both numbers are reported and a
    `meaningful` flag gates the claim accordingly."""
    # ---- NULL arm: bank-representable curves -> realized FDP under BH -----------
    null_curves = []
    for i in range(n_trials):
        s = seed * 521 + i
        model = _CLOSED_FORM_GENERATORS[i % len(_CLOSED_FORM_GENERATORS)]
        null_curves.append(generate_curve(
            {"generator": "closed_form", "model": model,
             "n_points": N_POINTS_DEFAULT, "t_max": 40.0, "censor": "none"}, seed=s + 1))
    null_p = _runs_test_pvalues(null_curves, "null")

    bh = benjamini_hochberg(null_p, alpha=ANOMALY_FDR_ALPHA)
    n_tested = bh["n_tested"]
    realized_fdp = (bh["n_rejected"] / n_tested) if n_tested else 0.0
    fdr_controlled = bool(realized_fdp <= ANOMALY_FDR_ALPHA + FDR_NULL_TOLERANCE)

    # uniformity hint on the null p-values (partial v6 dependency/uniformity check)
    valid_p = [p for p in null_p if p is not None]
    mean_p = float(np.mean(valid_p)) if valid_p else None

    # ---- POWER arm: bank-UNrepresentable curves -> TPR of the runs test ---------
    # half the null budget (modest) is enough to estimate power on a strong alternative.
    n_alt = max(20, n_trials // 2)
    alt_curves = [_bank_unrepresentable_curve(seed * 911 + 1 + i) for i in range(n_alt)]
    alt_p = _runs_test_pvalues(alt_curves, "alt")
    alt_tested = [p for p in alt_p if p is not None]
    n_alt_tested = len(alt_tested)
    # per-curve power at the raw nominal alpha (detector sensitivity to real structure)
    n_alt_rejected = int(np.sum(np.asarray(alt_tested) < ANOMALY_FDR_ALPHA)) if alt_tested else 0
    power_at_alpha = (n_alt_rejected / n_alt_tested) if n_alt_tested else None

    meaningful = bool(power_at_alpha is not None and power_at_alpha > FDR_POWER_FLOOR)
    return {
        "nominal_alpha": ANOMALY_FDR_ALPHA,
        "n_curves": n_trials,
        "n_tested": n_tested,
        "n_rejected": bh["n_rejected"],
        "realized_fdp": float(realized_fdp),
        "fdr_controlled": fdr_controlled,
        "tolerance": FDR_NULL_TOLERANCE,
        "null_pvalue_mean": mean_p,
        "null_pvalue_uniform_hint": ("~0.5 expected under a uniform null"
                                     if mean_p is not None else None),
        # power arm
        "alternative": "double_sigmoid (bank-unrepresentable mechanism-misfit)",
        "n_alt_curves": n_alt,
        "n_alt_tested": n_alt_tested,
        "n_alt_rejected": n_alt_rejected,
        "power_at_alpha": (float(power_at_alpha) if power_at_alpha is not None else None),
        "power_floor": FDR_POWER_FLOOR,
        "meaningful": meaningful,
        "note": "NULL arm: under the null every BH rejection is false, FDP = "
                "n_rejected/n_tested; control if FDP <= alpha + tolerance. POWER arm: "
                "TPR of the runs test on a bank-unrepresentable double-sigmoid at raw "
                "alpha. 'controlled' is only MEANINGFUL when power > power_floor "
                "(otherwise it is no-power, not control).",
    }


def threshold_recommendations(n_trials: int, seed: int) -> dict:
    """Using labeled synthetic data, sweep and RECOMMEND calibrated values for the
    flagged first-pass thresholds. RECOMMEND ONLY — Service C never edits M1/M3/M5/M6.

    Calibrated targets:
      * M1 LAG_SLOPE_FRAC  — separate true-no-lag (downhill/left-censored) from
        true-lag (sigmoid) by the start/peak slope-ratio; recommend the ratio that
        best separates the two labeled classes (Youden-J on the ROC).
      * M3 COND_NUMBER_NONIDENT — recommend the condition-number cutoff that best
        separates identifiable (clean logistic) from sloppy (Richards@nu≈1) fits.
      * M5 ANOMALY_FDR_ALPHA — recommend keeping the nominal alpha iff the FDR-null
        check shows control holds (do not loosen a level that already controls FDR).
      * M5 THRESHOLD_SHARPNESS / THRESHOLD_LAG_RATIO — recommend values separating
        threshold-driven from smooth-sigmoid synthetic shapes.
      * M6 null_disagreement — DEFERRED (the M6 rate is itself None/unimplemented).
    """
    rng = _rng(seed)

    # ---- M1 lag-slope fraction ------------------------------------------------
    # label: lag (sigmoid with a real lag) vs no_lag (exponential/scaling downhill).
    lag_ratios, lag_labels = [], []
    for i in range(n_trials):
        s = seed * 911 + i
        is_lag = (i % 2 == 0)
        model = ("logistic" if is_lag else "exponential")
        curve = generate_curve({"generator": "closed_form", "model": model,
                                "n_points": N_POINTS_DEFAULT, "t_max": 40.0,
                                "censor": "none"}, seed=s + 1)
        series = make_series(curve, series_id=f"lag-{i}")
        f = series.get("m1", {}).get("features", {})
        ms, ss = f.get("max_slope"), f.get("start_slope")
        if ms and ms > 0 and ss is not None:
            lag_ratios.append(max(ss, 0.0) / ms)   # start/peak slope ratio
            lag_labels.append(0 if is_lag else 1)  # 1 = no-lag (left/downhill)
    m1_rec = _best_separating_threshold(lag_ratios, lag_labels,
                                        positive_is_high=True, split_seed=seed + 11)

    # ---- M3 condition-number cutoff ------------------------------------------
    conds, cond_labels = [], []
    for i in range(n_trials):
        s = seed * 733 + i
        is_ident = (i % 2 == 0)
        if is_ident:
            p = _draw_params("logistic", _rng(s))
            c = generate_curve({"generator": "closed_form", "model": "logistic",
                                "params": p, "n_points": N_POINTS_DEFAULT,
                                "t_max": 40.0, "censor": "none"}, seed=s + 1)
            fit = fit_one(c["x"], c["y_obs"], "logistic")
            name = "logistic"
        else:
            pr = _draw_params("richards", _rng(s))
            pr["nu"] = 1.0 + float(_rng(s + 5).uniform(-0.05, 0.05))
            c = generate_curve({"generator": "closed_form", "model": "richards",
                                "params": pr, "n_points": 12, "t_max": 40.0,
                                "censor": "right"}, seed=s + 1)
            fit = fit_one(c["x"], c["y_obs"], "richards")
            name = "richards"
        if not fit.get("converged"):
            continue
        ident = fim_identifiability(name, fit["params"], c["x"], c["y_obs"])
        cn = ident.get("condition_number")
        if cn is not None and math.isfinite(cn) and cn > 0:
            conds.append(math.log10(cn))
            cond_labels.append(0 if is_ident else 1)  # 1 = non-identifiable
    m3_rec = _best_separating_threshold(conds, cond_labels, positive_is_high=True,
                                        split_seed=seed + 13)
    # report back in linear condition-number units
    m3_reco_linear = (10.0 ** m3_rec["recommended"]
                      if m3_rec["recommended"] is not None else None)

    # ---- M5 sharpness / lag-ratio (threshold-driven vs smooth sigmoid) -------
    sharp_vals, sharp_labels = [], []
    ratio_vals = []
    for i in range(n_trials):
        s = seed * 277 + i
        # smooth sigmoid (logistic) vs sharp threshold (very steep logistic, late t0)
        threshold_like = (i % 2 == 1)
        if threshold_like:
            params = {"base": 0.0, "amp": 1.0, "k": float(_rng(s).uniform(1.5, 3.0)),
                      "t0": float(_rng(s + 1).uniform(28.0, 34.0))}
        else:
            params = {"base": 0.0, "amp": 1.0, "k": float(_rng(s).uniform(0.2, 0.5)),
                      "t0": float(_rng(s + 1).uniform(14.0, 22.0))}
        curve = generate_curve({"generator": "closed_form", "model": "logistic",
                                "params": params, "n_points": N_POINTS_DEFAULT,
                                "t_max": 40.0, "censor": "none"}, seed=s + 2)
        series = make_series(curve, series_id=f"sharp-{i}")
        feat = extract_features(series)
        fdict = feat.get("features", {}) if feat.get("status") == "ok" else {}
        sh = fdict.get("transition_sharpness")
        rt = fdict.get("lag_to_t50_ratio")
        if sh is not None and math.isfinite(sh):
            sharp_vals.append(sh)
            sharp_labels.append(1 if threshold_like else 0)
            ratio_vals.append(rt if (rt is not None and math.isfinite(rt)) else 0.0)
    m5_sharp_rec = _best_separating_threshold(sharp_vals, sharp_labels,
                                              positive_is_high=True, split_seed=seed + 17)
    # lag-ratio recommendation: median ratio among the threshold-driven class (the
    # value above which a sharp curve must also sit to be called threshold-driven)
    thr_ratios = [r for r, l in zip(ratio_vals, sharp_labels) if l == 1]
    m5_ratio_reco = float(np.median(thr_ratios)) if thr_ratios else None

    # ---- M5 anomaly alpha — keep nominal iff FDR control holds ----------------
    fdr = fdr_null_check(max(40, n_trials // 2), seed + 1)

    return {
        "method": "labeled-synthetic sweeps; threshold CHOSEN on a seeded TRAIN split "
                  "(Youden-J), separation AUC REPORTED out-of-sample on a disjoint TEST "
                  "split (C2 — no leakage). RECOMMEND ONLY (Service C never edits M1–M7). "
                  "separation_auc is the HELD-OUT headline; separation_auc_train exposes "
                  "the in-sample optimism gap.",
        "M1_lag_slope_frac": {
            "module": "m1_ingest.LAG_SLOPE_FRAC",
            "current": m1_ingest.LAG_SLOPE_FRAC,
            "recommended": m1_rec["recommended"],
            "metric": "start/peak slope ratio separating lag vs no-lag (Youden-J on TRAIN)",
            "separation_auc": m1_rec["auc"],            # held-out (honest headline)
            "separation_auc_heldout": m1_rec["auc"],
            "separation_auc_train": m1_rec["auc_train"],
            "accuracy_heldout": m1_rec["accuracy_heldout"],
            "n": m1_rec["n"], "n_train": m1_rec["n_train"], "n_test": m1_rec["n_test"]},
        "M3_cond_number_nonident": {
            "module": "m3_select.COND_NUMBER_NONIDENT",
            "current": _m3.COND_NUMBER_NONIDENT,
            "recommended": m3_reco_linear,
            "recommended_log10": m3_rec["recommended"],
            "metric": "log10 condition number separating identifiable vs sloppy "
                      "(Youden-J on TRAIN)",
            "separation_auc": m3_rec["auc"],            # held-out
            "separation_auc_heldout": m3_rec["auc"],
            "separation_auc_train": m3_rec["auc_train"],
            "accuracy_heldout": m3_rec["accuracy_heldout"],
            "n": m3_rec["n"], "n_train": m3_rec["n_train"], "n_test": m3_rec["n_test"]},
        "M5_anomaly_fdr_alpha": {
            "module": "m5_classify.ANOMALY_FDR_ALPHA",
            "current": ANOMALY_FDR_ALPHA,
            "recommended": (ANOMALY_FDR_ALPHA if fdr["fdr_controlled"] else None),
            "metric": "keep nominal alpha iff FDR-null control holds",
            "fdr_controlled": fdr["fdr_controlled"],
            "realized_fdp": fdr["realized_fdp"]},
        "M5_threshold_sharpness": {
            "module": "m5_classify.THRESHOLD_SHARPNESS",
            "current": THRESHOLD_SHARPNESS,
            "recommended": m5_sharp_rec["recommended"],
            "metric": "transition_sharpness separating threshold-driven vs smooth "
                      "(Youden-J on TRAIN)",
            "separation_auc": m5_sharp_rec["auc"],      # held-out
            "separation_auc_heldout": m5_sharp_rec["auc"],
            "separation_auc_train": m5_sharp_rec["auc_train"],
            "accuracy_heldout": m5_sharp_rec["accuracy_heldout"],
            "n": m5_sharp_rec["n"], "n_train": m5_sharp_rec["n_train"],
            "n_test": m5_sharp_rec["n_test"]},
        "M5_threshold_lag_ratio": {
            "module": "m5_classify.THRESHOLD_LAG_RATIO",
            "current": THRESHOLD_LAG_RATIO,
            "recommended": m5_ratio_reco,
            "metric": "median lag/t50 ratio in the threshold-driven synthetic class"},
        "M6_null_disagreement": {
            "module": "m6_propensity.sequence_axis.null_disagreement_rate",
            "current": None,
            "recommended": None,
            "metric": "DEFERRED — the M6 null-disagreement rate is itself unimplemented "
                      "(returns None); calibrating it needs a sequence-predictor FP model "
                      "and assay detection-limit null, a later Service-C slice"},
    }


def _rank_auc(pos: np.ndarray, neg: np.ndarray, positive_is_high: bool) -> float | None:
    """Rank-based separation AUC (Mann-Whitney): P(pos value > neg value), with the
    0.5 tie correction. None when either class is empty (AUC undefined)."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = 0.0
    for pv in pos:
        wins += np.sum(pv > neg) + 0.5 * np.sum(pv == neg)
    auc = wins / (len(pos) * len(neg))
    return float(auc if positive_is_high else 1.0 - auc)


def _youden_threshold(v: np.ndarray, y: np.ndarray, positive_is_high: bool):
    """Threshold maximising Youden's J (TPR - FPR) on the given (values, labels).
    Candidate cut-points are midpoints between sorted unique values."""
    uniq = np.unique(v)
    cands = (uniq[:-1] + uniq[1:]) / 2.0 if len(uniq) > 1 else uniq
    best_j, best_t = -1.0, None
    for t in cands:
        pred = (v >= t).astype(int) if positive_is_high else (v <= t).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        tn = int(np.sum((pred == 0) & (y == 0)))
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t, float(best_j)


def _accuracy_at(v: np.ndarray, y: np.ndarray, t: float, positive_is_high: bool):
    """Classification accuracy applying threshold `t` to (values, labels)."""
    if t is None or len(v) == 0:
        return None
    pred = (v >= t).astype(int) if positive_is_high else (v <= t).astype(int)
    return float(np.mean(pred == y))


def _best_separating_threshold(values: list[float], labels: list[int],
                               positive_is_high: bool = True,
                               heldout_frac: float = THRESHOLD_HELDOUT_FRAC,
                               split_seed: int = 0) -> dict:
    """HONEST (C2-compliant) separating-threshold recommendation. NEVER report the
    cutoff's AUC on the SAME draws it was chosen on — that in-sample AUC is optimistic
    (the M1 AUC=1.0 tell of calibration LEAKAGE). Instead:
      * split the labeled synthetic set into disjoint TRAIN / TEST (seeded, stratified
        by label so both classes appear in each side);
      * CHOOSE the Youden-J threshold on TRAIN only;
      * REPORT the recommended threshold = the TRAIN-chosen value, the TRAIN AUC, and
        the OUT-OF-SAMPLE (held-out) TEST AUC + accuracy.
    The headline `auc` is the HELD-OUT one; `separation_auc_train` exposes the
    optimism gap. Pure/deterministic; returns Nones when a side is too small.

    `positive_is_high` = positive class (label 1) tends to have higher values."""
    if len(values) < 4 or len(set(labels)) < 2:
        return {"recommended": None, "auc": None, "auc_train": None,
                "accuracy_heldout": None, "n": len(values),
                "n_train": 0, "n_test": 0}
    v = np.asarray(values, float)
    y = np.asarray(labels, int)
    rng = _rng(split_seed)
    # stratified split: shuffle WITHIN each class, send `heldout_frac` of each to TEST,
    # so neither side loses a class (an unstratified split could leave TRAIN one-class).
    tr_idx, te_idx = [], []
    for cls in (0, 1):
        ci = np.where(y == cls)[0]
        ci = ci[rng.permutation(len(ci))]
        n_test = int(round(heldout_frac * len(ci)))
        n_test = max(1, min(n_test, len(ci) - 1)) if len(ci) >= 2 else 0
        te_idx.extend(ci[:n_test].tolist())
        tr_idx.extend(ci[n_test:].tolist())
    tr_idx, te_idx = np.asarray(tr_idx, int), np.asarray(te_idx, int)
    if len(tr_idx) < 2 or len(te_idx) < 2:
        return {"recommended": None, "auc": None, "auc_train": None,
                "accuracy_heldout": None, "n": len(values),
                "n_train": int(len(tr_idx)), "n_test": int(len(te_idx))}

    vtr, ytr = v[tr_idx], y[tr_idx]
    vte, yte = v[te_idx], y[te_idx]
    # choose the cutoff on TRAIN only (no peeking at TEST)
    best_t, best_j = _youden_threshold(vtr, ytr, positive_is_high)
    auc_train = _rank_auc(vtr[ytr == 1], vtr[ytr == 0], positive_is_high)
    auc_held = _rank_auc(vte[yte == 1], vte[yte == 0], positive_is_high)
    acc_held = _accuracy_at(vte, yte, best_t, positive_is_high)
    return {
        "recommended": best_t,             # train-chosen threshold (the recommendation)
        "auc": auc_held,                   # HEADLINE = honest held-out separation AUC
        "auc_train": auc_train,            # in-sample AUC (exposes the optimism gap)
        "accuracy_heldout": acc_held,
        "youden_j": best_j,
        "n": len(values), "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
    }


def calibration_report(n_trials: int = 100, seed: int = 0) -> dict:
    """Assemble the calibration report: descriptive-confidence ECE/Brier, the FDR-null
    realized-vs-nominal result, and the threshold-calibration RECOMMENDATIONS."""
    return {
        "descriptive_confidence": _descriptive_calibration(n_trials, seed),
        "fdr_null": fdr_null_check(n_trials, seed + 1),
        "threshold_recommendations": threshold_recommendations(
            max(40, n_trials), seed + 2),
    }


# ============================================================================ #
#  6. BASIC DE-CIRCULARIZATION  (C1 — mismatched generator)
# ============================================================================ #
def mismatched_generator_check(n_trials: int = 40, seed: int = 0) -> dict:
    """Generate with one family and fit/select with the OTHER, and report the
    DEGRADATION when generator != fitter. First step of breaking self-reference (C1);
    the FULL external forward model (coarse-grained KMC / Smoluchowski) and the blind
    real-literature reality check stay DEFERRED.

    MUST-FIX 4: R² is the WRONG degradation metric here — both families fit sigmoids, so
    R² barely moves under mismatch (a high-R² closed-form fit to ODE data hides that the
    MECHANISM is gone). We therefore report a metric that actually degrades under
    generator≠fitter: MECHANISM-IDENTIFICATION ACCURACY. R² is kept only as a secondary
    diagnostic to SHOW it does NOT move (the very reason it was a bad metric).

    Direction A (ODE truth):
      * matched fitter = mechanistic Tier-B bank -> mechanism-id accuracy = P(AICc variant
        == true variant). A genuinely MEASURED number (single curves are degenerate, so it
        is well below 1 — that itself is the K2 story).
      * mismatched fitter = closed-form descriptive bank -> NO closed-form model names a
        Tier-B mechanism, so mechanism-id accuracy is structurally 0. We additionally
        report the MEASURED coarse-regime accuracy of the descriptive call (cooperative vs
        gradual) to show that even the coarse regime carries less mechanism information.
    Headline degradation = matched mechanism-id accuracy − mismatched mechanism-id accuracy.
    """
    from mechanistic import fit_mechanistic

    # closed-form model -> coarse regime it implies (for the measured regime-accuracy arm)
    _cf_regime = {"logistic": "cooperative_sigmoidal", "gompertz": "cooperative_sigmoidal",
                  "richards": "cooperative_sigmoidal", "exponential": "gradual_non_cooperative",
                  "scaling_law": "gradual_non_cooperative", "lnt": "gradual_non_cooperative"}

    mech_id_matched = []         # matched mechanistic fitter: AICc variant == true variant?
    regime_ok_mismatch = []      # mismatched closed-form fitter: coarse regime correct?
    r2_matched_A, r2_mismatch_A = [], []
    for i in range(n_trials):
        s = seed * 401 + i
        variant = _MECH_GENERATORS[i % len(_MECH_GENERATORS)]
        curve = generate_curve({"generator": "mechanistic", "variant": variant,
                                "n_points": N_POINTS_DEFAULT, "t_max": 40.0,
                                "censor": "none", "concentration_uM": 4.0}, seed=s + 1)
        x, y = curve["x"], curve["y_obs"]
        regime_true = curve["ground_truth"].get("regime_true")
        if len(x) < 8:
            continue
        # MATCHED fitter family: mechanistic bank -> did AICc recover the TRUE variant?
        sel_variant = _select_mechanistic_variant(x, y)
        if sel_variant is not None:
            mech_id_matched.append(int(sel_variant == variant))   # 1 = correct mechanism
            fm = fit_mechanistic(x, y, variant=sel_variant)
            if fm.get("converged") and fm.get("r2") is not None:
                r2_matched_A.append(fm["r2"])
        # MISMATCHED fitter family: closed-form bank -> mechanism is structurally lost;
        # record the MEASURED coarse-regime accuracy of the descriptive winner.
        best_cf = _select_closed_form(x, y)
        if best_cf is not None:
            implied = _cf_regime.get(best_cf)
            if regime_true is not None and implied is not None:
                regime_ok_mismatch.append(int(implied == regime_true))
            rf = fit_one(x, y, best_cf)
            if rf.get("converged") and rf.get("r2") is not None:
                r2_mismatch_A.append(rf["r2"])

    def _mean(v):
        return float(np.mean(v)) if v else None

    matched_mech_id_acc = _mean(mech_id_matched)
    # closed-form names no Tier-B mechanism -> mechanism-id accuracy is structurally 0
    mismatched_mech_id_acc = 0.0 if regime_ok_mismatch else None
    mismatch_regime_acc = _mean(regime_ok_mismatch)
    # headline non-R² degradation: mechanism information LOST by using the wrong family
    misident_degradation = (
        (matched_mech_id_acc - mismatched_mech_id_acc)
        if (matched_mech_id_acc is not None and mismatched_mech_id_acc is not None) else None)
    mA_match, mA_mis = _mean(r2_matched_A), _mean(r2_mismatch_A)
    r2_degradation = (mA_match - mA_mis) if (mA_match is not None and mA_mis is not None) else None

    return {
        "n_trials": n_trials,
        # ---- HEADLINE: non-R² mechanism-identification-accuracy degradation ----
        "primary_metric": "mechanism-identification accuracy under generator≠fitter",
        "matched_mechanism_id_accuracy": matched_mech_id_acc,
        "mismatched_mechanism_id_accuracy": mismatched_mech_id_acc,
        "mismatched_coarse_regime_accuracy": mismatch_regime_acc,
        "misidentification_degradation": (float(misident_degradation)
                                          if misident_degradation is not None else None),
        # ---- secondary R² diagnostic (kept to SHOW R² is ~flat, the MUST-FIX-4 point) ----
        "r2_secondary_diagnostic": {
            "r2_matched_mechanistic": mA_match,
            "r2_mismatched_closed_form": mA_mis,
            "degradation_r2": (float(r2_degradation) if r2_degradation is not None else None),
            "comment": "R² barely moves under mismatch (both families fit sigmoids) — this "
                       "is exactly why R² is NOT used as the degradation metric (MUST-FIX 4)."},
        "deferred": [
            "FULL external forward model (coarse-grained KMC / Smoluchowski) as the "
            "mismatched generator — here the mismatch is Tier-B ODE vs the closed-form "
            "bank, both internal to PRISE",
            "BLIND external real-literature reality check (Aβ42/α-syn/insulin/β2m vs "
            "field consensus) — governed separately (§6C/§8), consumes reserved real proteins",
            "artifact-realism distributional check vs real flagged traces",
        ],
        "note": "misidentification_degradation > 0 means the wrong fitter family loses "
                "MECHANISM-identification accuracy even where R² stays high — the genuine "
                "self-reference-breaking signal (C1) that an R² delta misses. The matched "
                "accuracy is itself well below 1 (single curves are degenerate, K2), so the "
                "degradation is the information a wrong family throws away ON TOP of that.",
    }


# ============================================================================ #
#  DEFAULT BATTERY + CLI
# ============================================================================ #
def run_battery(n_trials: int = 100, seed: int = 0,
                mech_trials: int | None = None) -> dict:
    """Run the full default Service C battery. ODE trials are kept modest (stiff
    fits) so total runtime stays under a few minutes at the default n_trials."""
    if mech_trials is None:
        mech_trials = max(15, n_trials // 4)   # ODE is slow -> fewer trials

    # 2. parameter recovery on the well-behaved descriptive bank
    recovery = {m: parameter_recovery(m, n_trials=n_trials, seed=seed)
                for m in ("logistic", "gompertz", "richards", "exponential")}

    # 3. data-regime confusion matrices + equivalence classes
    # single_concentration_mech = the BASELINE: each Tier-B mechanism gets ONE single-
    #   concentration curve and the engine's single-curve mechanistic AICc selection
    #   (the broad collapse a single curve cannot resolve).
    # concentration_series = the SAME mechanisms but now given a real multi-concentration
    #   FAMILY, discriminated by the γ-scaling the series exposes (MUST-FIX 2).
    # The single-curve mechanistic AICc sweep does 4 stiff ODE fits per curve, so its
    # trial budget is capped tighter than mech_trials to keep the battery under a few min.
    mech_conf_trials = min(mech_trials, MECH_CONFUSION_MAX_TRIALS)
    single_conc_mech = confusion_matrix("single_concentration_mech",
                                        generators=list(MECH_SERIES_RATES.keys()),
                                        n_trials=mech_conf_trials, seed=seed)
    series_gamma = concentration_series_gamma_confusion(
        n_trials=max(8, mech_trials), seed=seed)   # γ-collapse is fast -> full budget
    confusion = {
        "single_curve": confusion_matrix("single_curve", n_trials=n_trials, seed=seed),
        "single_concentration_mech": single_conc_mech,
        "concentration_series": series_gamma,
        "saturating_secondary": confusion_matrix(
            "saturating_secondary", n_trials=mech_conf_trials, seed=seed),
    }
    equivalence_classes = {k: v["equivalence_classes"] for k, v in confusion.items()}

    # Does the concentration series NARROW the degeneracy vs a single concentration?
    # Compare the equivalence-class partitions of the SAME mechanism set (honest test
    # of the K2 claim — report it whether or not γ helps).
    narrowing = _series_narrowing(single_conc_mech, series_gamma)

    # 4. identifiability validation
    identifiability = identifiability_validation(n_trials=max(30, n_trials // 3),
                                                 seed=seed)

    # 5. calibration report
    calibration = calibration_report(n_trials=n_trials, seed=seed)

    # 6. mismatched-generator de-circularization
    mismatched = mismatched_generator_check(n_trials=mech_trials, seed=seed)

    return {
        "service_c_version": SERVICE_C_VERSION,
        "seed": seed,
        "n_trials": n_trials,
        "mechanistic_trials": mech_trials,
        "pre_registered_constants": {
            "SIGMA_MULT": SIGMA_MULT, "SIGMA_ADD_FRAC": SIGMA_ADD_FRAC,
            "BASELINE_DRIFT_FRAC": BASELINE_DRIFT_FRAC,
            "DIGITIZE_STEP_FRAC": DIGITIZE_STEP_FRAC,
            "LEFT_CENSOR_FRAC": LEFT_CENSOR_FRAC,
            "RIGHT_CENSOR_FRAC": RIGHT_CENSOR_FRAC,
            "CONFUSABILITY_THRESHOLD": CONFUSABILITY_THRESHOLD,
            "FDR_NULL_TOLERANCE": FDR_NULL_TOLERANCE,
            "THRESHOLD_HELDOUT_FRAC": THRESHOLD_HELDOUT_FRAC,
            "FDR_POWER_FLOOR": FDR_POWER_FLOOR,
            "MECH_MTOT_PER_UM": MECH_MTOT_PER_UM,
            "MECH_SERIES_CONCS_UM": list(MECH_SERIES_CONCS_UM),
            "GAMMA_SEPARABILITY_AUC": GAMMA_SEPARABILITY_AUC,
            "basis": "fixed from physics/instrument specs; NOT tuned to judged data (C2)",
        },
        "parameter_recovery": recovery,
        "confusion_matrices": confusion,
        "equivalence_classes": equivalence_classes,
        "concentration_series_narrowing": narrowing,
        # HONESTY / validity ceiling: the γ-separability headline is partly a
        # construction artifact (rate constants chosen to be separable) — surfaced
        # at the report root so it is not buried inside the confusion sub-tree.
        "validity_ceiling": {
            "gamma_separability_is_upper_bound": True,
            "separability_caveat": SEPARABILITY_CAVEAT,
            "note": ("Service C measures γ-separability on synthetic families whose "
                     "per-mechanism rates (MECH_SERIES_RATES) were hand-picked to be "
                     "distinguishable. The reported per-regime equivalence classes are "
                     "therefore an OPTIMISTIC (upper-bound) view of discriminability; "
                     "real mechanisms can share γ and would collapse classes further."),
        },
        "identifiability_validation": identifiability,
        "calibration_report": calibration,
        "mismatched_generator": mismatched,
        "deferred_slices": [
            "external forward model (coarse-grained KMC / Smoluchowski)",
            "blind external real-literature reality check",
            "real held-out / leave-one-study-out synthetic→real transfer measurement",
            "artifact-realism distributional check vs real flagged traces",
            "M6 null-disagreement-rate calibration",
        ],
    }


def _print_summary(report: dict) -> None:
    print("\n" + "=" * 70)
    print(f"  PRISE Service C — calibration battery  ({report['service_c_version']})")
    print(f"  n_trials={report['n_trials']}  mech_trials={report['mechanistic_trials']}"
          f"  seed={report['seed']}")
    print("=" * 70)

    print("\n[2] PARAMETER RECOVERY (bias / rmse / t50-rmse / 95%-coverage):")
    for m, r in report["parameter_recovery"].items():
        t50 = r["t50_recovery"]
        covs = [v["coverage_95"] for v in r["per_parameter"].values()
                if v["coverage_95"] is not None]
        mean_cov = round(float(np.mean(covs)), 2) if covs else None
        print(f"  {m:<12} conv={r['n_converged']:>3}/{r['n_trials']:<3} "
              f"t50_rmse={_fmt(t50.get('rmse'))}  mean_cov95={mean_cov}")

    print("\n[3] CONFUSION / EQUIVALENCE CLASSES per data_regime:")
    for regime, cm in report["confusion_matrices"].items():
        if "mean_diagonal" in cm:
            print(f"  {regime:<26} mean_diag={_fmt(cm['mean_diagonal'])} "
                  f"min_diag={_fmt(cm['min_diagonal'])}")
        else:   # γ-scaling discriminator (concentration_series) — no AICc diagonal
            gm = {g: _fmt(v["mean"]) for g, v in cm.get("gamma_by_mechanism", {}).items()}
            print(f"  {regime:<26} selector={cm.get('selector')}")
            print(f"      gamma_by_mechanism = {gm}")
        print(f"      equivalence_classes = {cm['equivalence_classes']}")
    nar = report.get("concentration_series_narrowing", {})
    print(f"  >> series narrows degeneracy vs single concentration? "
          f"{nar.get('series_narrows_degeneracy')}  "
          f"(single={nar.get('single_concentration_partition', {}).get('classes')} -> "
          f"series={nar.get('concentration_series_partition', {}).get('classes')}; "
          f"newly resolved by γ: {nar.get('pairs_newly_resolved_by_series')})")
    print(f"     CAVEAT (separability is an UPPER BOUND): {SEPARABILITY_CAVEAT}")

    iv = report["identifiability_validation"]
    print("\n[4] IDENTIFIABILITY VALIDATION:")
    print(f"  TPR(sloppy flagged)={_fmt(iv['true_positive_rate'])}  "
          f"FPR(clean flagged)={_fmt(iv['false_positive_rate'])}")

    cal = report["calibration_report"]
    dc = cal["descriptive_confidence"]
    fn = cal["fdr_null"]
    print("\n[5] CALIBRATION:")
    print(f"  descriptive confidence: ECE={_fmt(dc['ece'])} Brier={_fmt(dc['brier'])} "
          f"emp_acc={_fmt(dc['empirical_accuracy'])} (n={dc['n']})")
    print(f"  FDR-null: nominal={fn['nominal_alpha']} realized_FDP={_fmt(fn['realized_fdp'])} "
          f"controlled={fn['fdr_controlled']}  power={_fmt(fn['power_at_alpha'])} "
          f"meaningful={fn['meaningful']} (n_tested={fn['n_tested']}, n_alt={fn['n_alt_tested']})")
    print("  threshold recommendations (current -> recommended | AUC train/heldout):")
    for key, rec in cal["threshold_recommendations"].items():
        if not isinstance(rec, dict) or "current" not in rec:
            continue
        at = rec.get("separation_auc_train")
        ah = rec.get("separation_auc_heldout")
        auc_str = (f"  AUC {_fmt(at)}/{_fmt(ah)}" if at is not None or ah is not None else "")
        print(f"    {key:<26} {_fmt(rec['current'])} -> {_fmt(rec['recommended'])}{auc_str}")

    mm = report["mismatched_generator"]
    print("\n[6] MISMATCHED-GENERATOR DEGRADATION (C1):")
    print(f"  mechanism-id accuracy: matched={_fmt(mm['matched_mechanism_id_accuracy'])}"
          f"  mismatched={_fmt(mm['mismatched_mechanism_id_accuracy'])}"
          f"  (mismatch coarse-regime acc={_fmt(mm['mismatched_coarse_regime_accuracy'])})")
    print(f"  >> misidentification degradation = {_fmt(mm['misidentification_degradation'])}"
          f"   (R² secondary dR2={_fmt(mm['r2_secondary_diagnostic']['degradation_r2'])} — "
          f"~flat, why R² is not the metric)")
    print("=" * 70 + "\n")


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int,)):
        return str(v)
    try:
        return f"{float(v):.3g}"
    except (TypeError, ValueError):
        return str(v)


def main(argv=None):
    # Windows consoles default to cp1252; force UTF-8 so any unicode in the summary
    # never crashes the run (the JSON report is written UTF-8 regardless).
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                 # pragma: no cover
        pass
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="PRISE Service C — synthetic calibration battery")
    ap.add_argument("--n-trials", type=int, default=120,
                    help="trials per closed-form sweep (modest for runtime)")
    ap.add_argument("--mech-trials", type=int, default=None,
                    help="trials for the stiff ODE sweeps (default n_trials//4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path,
                    default=root / "data" / "processed" / "service_c_calibration.json")
    args = ap.parse_args(argv)

    print(f"[Service C] running battery (n_trials={args.n_trials}) ...")
    report = run_battery(n_trials=args.n_trials, seed=args.seed,
                         mech_trials=args.mech_trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Service C] wrote {args.output}")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
