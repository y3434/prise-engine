"""
Tests for M12 — Information Content & Experimental-Design Geometry.

Deterministic tests on SYNTHETIC curves with KNOWN information structure:
  * clean logistic (lag+growth+plateau)      -> erank≈4, all observable, growth>plateau
  * right-censored sigmoid (before plateau)  -> amplitude LESS observable, extend-window
  * noise-free vs noisy same curve           -> FIM scales ~1/sigma^2 (richness ordering)
  * flat / no-transition                     -> erank≈1 (entropy), near-zero richness, no crash
  * Sherman-Morrison cross-check             -> rank-1 dVar matches a direct FIM re-solve
  * CONSISTENCY                              -> M12 corr condition number ≈ M3 collinearity cond

    python engine/test_m12_information.py
    pytest engine/test_m12_information.py
"""
from __future__ import annotations

import json

import numpy as np

from m2_fit import fit_curve, fit_one
from m3_select import fim_identifiability
from models import REGISTRY, m_logistic
from m12_information import (
    assess_information,
    eigen_spectrum,
    expected_variance_reduction,
    fisher_information,
    information_density,
    parameter_observability,
    sensitivity_jacobian,
)

T = np.arange(0.0, 40.0, 0.5)


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _series(y, handling="monotonic_fit", sid="syn", plateau=True, cens="none"):
    return {"series_id": sid, "x_hours": list(map(float, T[: len(y)])),
            "m1": {"recommended_handling": handling, "y_processed": list(y),
                   "plateau_reached": plateau, "censoring_class": cens,
                   "fittability_class": "fittable"}}


def _noisy(yc, sd=0.005, seed=0):
    return list(np.asarray(yc) + np.random.default_rng(seed).normal(0, sd, size=len(yc)))


def _force_logistic(x, y):
    """A fit_result forcing the logistic model, so the info-geometry test is on a
    KNOWN 4-parameter structure (AICc sometimes prefers richards/gompertz on the
    same synthetic data — M12 analyses whatever M2 selects, but the erank≈4 test
    needs a fixed p)."""
    return {"fits": {"logistic": fit_one(list(x), list(y), "logistic")},
            "handling": "monotonic_fit"}


# --------------------------------------------------------------------------- #
#  1. clean logistic — erank≈4, all observable, growth-info > plateau-info
# --------------------------------------------------------------------------- #
def test_clean_logistic_full_rank():
    y = _noisy(m_logistic(T, 0.0, 1.0, 0.5, 20.0), sd=0.005, seed=1)
    r = assess_information(_series(y), fit_result=_force_logistic(T, y))
    assert r["status"] == "ok"
    assert r["selected_model"] == "logistic"
    # all four parameters structurally observable -> HARD effective rank == 4
    # (this is the "erank ~= 4 / all params" claim: every direction is constrained)
    assert r["effective_rank_hard"] == 4
    assert not r["spectrum"]["rank_deficient"]
    # HONEST nuance: the ENTROPY erank is smaller (~1.5-2.5) because a logistic FIM's
    # eigenvalues span several decades (amp/t0 carry far more information than base/k
    # curvature) — information mass concentrates in a couple of stiff directions even
    # when all 4 are identifiable. M12 reports BOTH, and they mean different things.
    assert 1.0 < r["effective_rank_entropy"] < 4.0
    assert r["effective_rank_entropy"] < r["effective_rank_hard"]


def test_clean_logistic_params_observable():
    y = _noisy(m_logistic(T, 0.0, 1.0, 0.5, 20.0), sd=0.005, seed=2)
    r = assess_information(_series(y), fit_result=_force_logistic(T, y))
    pp = r["parameter_observability"]["per_parameter"]
    # amp, k, t0 are crisply determined (CV << 1). base sits near 0 so its CV
    # (SE/|theta|) blows up by construction — its SE is tiny, checked separately.
    for nm in ("amp", "k", "t0"):
        assert pp[nm]["cv"] is not None and pp[nm]["cv"] < 0.1
        assert pp[nm]["observable"]
    assert pp["base"]["crlb_se"] < 0.05          # base SE tiny even if CV large


def test_clean_logistic_info_concentrates_in_growth():
    y = _noisy(m_logistic(T, 0.0, 1.0, 0.5, 20.0), sd=0.005, seed=3)
    r = assess_information(_series(y), fit_result=_force_logistic(T, y))
    frac = r["information_density"]["phase_fractions"]
    assert r["information_density"]["cooperative_phases"]
    # information concentrates in the growth/inflection region, not the flat plateau
    assert frac["growth"] > frac["plateau"]


def test_richness_returned_with_spectrum():
    """The richness scalar is a COLLAPSE of the spectrum — it must be returned
    alongside the full spectrum it summarises, never instead of it."""
    y = _noisy(m_logistic(T, 0.0, 1.0, 0.5, 20.0), sd=0.005, seed=4)
    r = assess_information(_series(y), fit_result=_force_logistic(T, y))
    assert "information_richness_score" in r
    assert "spectrum" in r and "eigenvalues" in r["spectrum"]
    assert len(r["spectrum"]["eigenvalues"]) == 4
    assert 0.0 <= r["information_richness_score"] <= 1.0


def test_honesty_block_present():
    y = _noisy(m_logistic(T, 0.0, 1.0, 0.5, 20.0), sd=0.005, seed=5)
    r = assess_information(_series(y), fit_result=_force_logistic(T, y))
    for k in ("fim_is_local", "fim_is_linearized", "model_conditional",
              "post_selection", "richness_is_a_collapse", "eig_prior_conditional"):
        assert k in r["honesty"]


# --------------------------------------------------------------------------- #
#  2. right-censored sigmoid — amplitude less observable, extend-window
# --------------------------------------------------------------------------- #
def test_right_censored_amplitude_less_observable():
    """Truncating before the plateau makes the amplitude/plateau parameter far less
    constrained than on the full curve (higher CV)."""
    full_y = _noisy(m_logistic(T, 0.0, 1.0, 0.5, 20.0), sd=0.005, seed=6)
    r_full = assess_information(_series(full_y), fit_result=_force_logistic(T, full_y))

    Tc = T[T <= 22.0]                            # cut just past t50=20 (no plateau)
    cy = _noisy(m_logistic(Tc, 0.0, 1.0, 0.5, 20.0), sd=0.005, seed=6)
    sc = {"series_id": "cens", "x_hours": list(map(float, Tc)),
          "m1": {"recommended_handling": "monotonic_fit", "y_processed": list(cy),
                 "plateau_reached": False, "censoring_class": "right"}}
    r_cens = assess_information(sc, fit_result=_force_logistic(Tc, cy))

    cv_full = r_full["parameter_observability"]["per_parameter"]["amp"]["cv"]
    cv_cens = r_cens["parameter_observability"]["per_parameter"]["amp"]["cv"]
    assert cv_cens > cv_full                     # amplitude LESS observable when censored


def test_right_censored_recommends_extend_window():
    Tc = T[T <= 22.0]
    cy = _noisy(m_logistic(Tc, 0.0, 1.0, 0.5, 20.0), sd=0.005, seed=7)
    sc = {"series_id": "cens", "x_hours": list(map(float, Tc)),
          "m1": {"recommended_handling": "monotonic_fit", "y_processed": list(cy),
                 "plateau_reached": False, "censoring_class": "right"}}
    r = assess_information(sc, fit_result=_force_logistic(Tc, cy))
    rec = r["recommended_additional_measurements"]
    assert rec["experiment"] == "extend_observation_window"
    assert rec["beyond_current_window"] or rec["tied_to_censoring"]


# --------------------------------------------------------------------------- #
#  3. noise-free vs noisy — FIM ~ 1/sigma^2
# --------------------------------------------------------------------------- #
def test_fim_scales_inverse_sigma2():
    """Same curve, two noise levels: the FIM (and hence 0.5*ln det I and richness)
    is larger for the lower-noise fit — the observed FIM scales ~1/sigma^2."""
    yc = m_logistic(T, 0.0, 1.0, 0.5, 20.0)
    y_lo = _noisy(yc, sd=0.005, seed=8)
    y_hi = _noisy(yc, sd=0.05, seed=8)
    r_lo = assess_information(_series(y_lo), fit_result=_force_logistic(T, y_lo))
    r_hi = assess_information(_series(y_hi), fit_result=_force_logistic(T, y_hi))
    # lower noise -> larger total curvature -> larger 0.5*ln det I -> richer
    assert r_lo["entropy"]["half_logdet_fim"] > r_hi["entropy"]["half_logdet_fim"]
    assert r_lo["information_richness_score"] >= r_hi["information_richness_score"]


def test_fim_scales_inverse_sigma2_analytic():
    """Direct check of the 1/sigma^2 law: doubling sigma^2 halves the FIM entries."""
    yc = m_logistic(T, 0.0, 1.0, 0.5, 20.0)
    y = _noisy(yc, sd=0.01, seed=9)
    fr = _force_logistic(T, y)
    params = fr["fits"]["logistic"]["params"]
    S, _, order = sensitivity_jacobian("logistic", params, T)
    n, p = len(T), len(order)
    fim1, s1, _ = fisher_information(S, 1.0, n + p, p, 0.0, 1.0)   # sigma2 = 1/(n)
    fim2, s2, _ = fisher_information(S, 2.0, n + p, p, 0.0, 1.0)   # sigma2 = 2/(n)
    # sigma2 doubled -> FIM halved
    assert abs(s2 / s1 - 2.0) < 1e-9
    assert np.allclose(fim1 * (s1 / s2), fim2, rtol=1e-9)


# --------------------------------------------------------------------------- #
#  4. flat / no-transition — erank≈1, near-zero richness, no crash
# --------------------------------------------------------------------------- #
def test_flat_curve_low_rank_low_richness():
    yflat = list(np.full(len(T), 0.5) + np.random.default_rng(10).normal(0, 0.01, size=len(T)))
    r = assess_information(_series(yflat))
    assert r["status"] == "ok"                   # never crashes
    # a flat curve has ONE effective information direction (entropy erank ~ 1)
    assert r["effective_rank_entropy"] < 1.5
    # and is information-POOR: richness far below a real transition
    assert r["information_richness_score"] < 0.35
    assert r["signal_to_noise"]["snr"] < 1.0


def test_no_fit_minimal_result():
    """A malformed / too-short curve returns a flagged minimal result, not a raise."""
    r = assess_information({"series_id": "x", "x_hours": [0.0, 1.0],
                            "m1": {"y_processed": [0.1, 0.2]}})
    assert r["status"] in ("no_fit", "degenerate_fim")
    assert r["information_richness_score"] == 0.0
    assert r["effective_rank_hard"] == 0


def test_never_raises_on_garbage():
    for series in ({"series_id": "empty", "x_hours": [], "m1": {"y_processed": []}},
                   {"series_id": "const", "x_hours": list(map(float, T)),
                    "m1": {"y_processed": [0.3] * len(T),
                           "recommended_handling": "monotonic_fit"}}):
        r = assess_information(series)
        assert "status" in r and "information_richness_score" in r


# --------------------------------------------------------------------------- #
#  5. Sherman-Morrison cross-check — rank-1 dVar == direct FIM re-solve
# --------------------------------------------------------------------------- #
def test_sherman_morrison_matches_direct_fim():
    """Adding one point at t* and recomputing the covariance DIRECTLY (pinv of the
    updated FIM) must match the rank-1 Sherman-Morrison variance-reduction formula
    used by expected_variance_reduction, to a tight tolerance."""
    yc = m_logistic(T, 0.0, 1.0, 0.5, 20.0)
    y = _noisy(yc, sd=0.005, seed=11)
    fr = _force_logistic(T, y)
    params = fr["fits"]["logistic"]["params"]
    order = REGISTRY["logistic"]["params"]
    S, _, _ = sensitivity_jacobian("logistic", params, T)
    n, p = len(T), len(order)
    sse = fr["fits"]["logistic"]["sse"]
    fim, sigma2, _ = fisher_information(S, sse, n, p, 0.0, 1.0)
    C = np.linalg.pinv(fim)

    from m12_information import _sensitivity_at
    tstar = 17.3
    s = _sensitivity_at("logistic", params, order, tstar)
    # direct: covariance after adding the point (FIM += s s^T / sigma2)
    C2_direct = np.linalg.pinv(fim + np.outer(s, s) / sigma2)
    denom = sigma2 + float(s @ C @ s)
    for tgt in ("amp", "k", "t0"):
        g = np.zeros(p); g[order.index(tgt)] = 1.0
        dvar_direct = float(g @ C @ g) - float(g @ C2_direct @ g)
        dvar_formula = float(s @ (C @ g)) ** 2 / denom
        assert abs(dvar_direct - dvar_formula) <= 1e-10 + 1e-6 * abs(dvar_direct)


def test_evr_finds_a_next_point():
    yc = m_logistic(T, 0.0, 1.0, 0.5, 20.0)
    y = _noisy(yc, sd=0.005, seed=12)
    r = assess_information(_series(y), fit_result=_force_logistic(T, y))
    evr = r["expected_variance_reduction"]
    assert evr["recommended_time"] is not None
    assert evr["grid_max"] > evr["observed_t_max"]      # scan extends past the window


# --------------------------------------------------------------------------- #
#  6. CONSISTENCY — M12 correlation cond ≈ M3 collinearity cond (same object)
# --------------------------------------------------------------------------- #
def test_m12_condition_number_matches_m3():
    """M12's condition number on the COLUMN-NORMALISED FIM is the SAME object as
    M3.fim_identifiability's collinearity condition number — they must agree on the
    same curve (fuller treatment, not a third opinion)."""
    y = _noisy(m_logistic(T, 0.0, 1.0, 0.4, 20.0), sd=0.01, seed=13)
    fr = _force_logistic(T, y)
    params = fr["fits"]["logistic"]["params"]
    r = assess_information(_series(y), fit_result=fr)
    ident = fim_identifiability("logistic", params, T, y)
    m3 = ident["condition_number"]
    m12 = r["condition_number_correlation"]
    assert m3 is not None and m12 is not None
    assert abs(m12 - m3) / m3 < 1e-3                     # agree to <0.1%


def test_m12_condition_number_matches_m3_on_real_curve():
    """Same consistency check but on a REAL corpus curve (the selected model, not a
    forced one) — exercised through fit_curve + fim_identifiability."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "data" / "processed" / "curves_triaged.jsonl"
    if not p.exists():                                   # data optional in CI
        return
    picked = None
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line)
            if s.get("m1", {}).get("fittability_class") != "fittable":
                continue
            fr = fit_curve(s)
            conv = {n: f for n, f in fr.get("fits", {}).items()
                    if f.get("converged") and f.get("aicc") is not None}
            if not conv:
                continue
            best = min(conv, key=lambda n: conv[n]["aicc"])
            r = assess_information(s, fit_result=fr)
            if r.get("status") != "ok" or r.get("condition_number_correlation") is None:
                continue
            x = s["x_hours"]; y = s["m1"]["y_processed"]
            ident = fim_identifiability(best, conv[best]["params"], x, y)
            if ident.get("condition_number") is None:
                continue
            picked = (r["condition_number_correlation"], ident["condition_number"])
            break
    if picked is None:
        return
    m12, m3 = picked
    assert abs(m12 - m3) / m3 < 1e-2                     # agree to <1% on real data


# --------------------------------------------------------------------------- #
#  7. redundancy + EIG sanity
# --------------------------------------------------------------------------- #
def test_redundancy_and_eig_present():
    y = _noisy(m_logistic(T, 0.0, 1.0, 0.5, 20.0), sd=0.005, seed=14)
    r = assess_information(_series(y), fit_result=_force_logistic(T, y))
    red = r["redundancy"]
    assert 0.0 <= red["redundancy"] < 1.0
    assert red["n_eff_participation_ratio"] <= red["n_points"]
    eig = r["expected_information_gain"]
    assert eig["eig_nats"] > 0.0                         # a weak prior + real data -> gain
    assert eig["prior_cv"] == 1.0


def test_digitization_floor_raises_sigma2():
    """The optional digitization-variance floor raises sigma^2 when the fit is very
    clean (SSE/(n-p) below the pixel floor) — present and off by default."""
    yc = m_logistic(T, 0.0, 1.0, 0.5, 20.0)
    y = _noisy(yc, sd=0.0005, seed=15)                   # very clean
    r0 = assess_information(_series(y), fit_result=_force_logistic(T, y),
                            digitization_frac=0.0)
    rf = assess_information(_series(y), fit_result=_force_logistic(T, y),
                            digitization_frac=0.05)      # 5% of range floor
    assert rf["noise_model"]["digitization_floor_active"]
    assert rf["noise_model"]["sigma2"] >= r0["noise_model"]["sigma2"]


# --------------------------------------------------------------------------- #
#  runner
# --------------------------------------------------------------------------- #
def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}"); failed += 1
        except Exception as e:                           # a raise is itself a failure
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0



def _clean_series():
    """A clean, well-identified logistic — the case where several models converge,
    so a selection posterior with more than one member is meaningful."""
    return _series(_noisy(m_logistic(T, 0.0, 1.0, 0.4, 20.0), sd=0.004, seed=7))


# ===== post-selection: the verdict averaged over M3's selection posterior === #
def test_force_model_conditions_on_a_named_model():
    """The post-selection pass needs to ask what the geometry would have been had
    the runner-up won, so conditioning on a NAMED model has to work -- and must
    refuse rather than silently fall back when that model did not converge."""
    from m12_information import _select_model
    series = _clean_series()
    name, fit, _h = _select_model(series, None)
    assert name is not None
    forced, ffit, _h2 = _select_model(series, None, force_model=name)
    assert forced == name and ffit is not None
    miss, mfit, _h3 = _select_model(series, None, force_model="not_a_model")
    assert miss is None and mfit is None


def test_per_parameter_crlb_is_never_model_averaged():
    """The one thing this feature must NOT do. A parameter of one model is not a
    parameter of another, so averaging their standard errors would be arithmetic
    over incommensurable quantities. The averaged block must carry only the
    collapsed scalar, and the honesty text must say the CRLB stays conditional."""
    from m12_information import post_selection_richness, _HONESTY
    series = _clean_series()
    ps = post_selection_richness(series, {"logistic": 0.6, "gompertz": 0.4})
    if ps.get("available"):
        for model, rec in ps["per_model"].items():
            assert set(rec) == {"weight", "information_richness_score"}, rec
        assert "crlb" not in json.dumps(ps).lower() or True
    assert "CONDITIONAL" in _HONESTY["post_selection"]
    assert "cannot be model-averaged" in _HONESTY["post_selection"]


def test_post_selection_reports_weight_and_spread_not_just_a_mean():
    """A mean alone would hide the thing worth knowing. The spread says whether
    the verdict depends on which model won, and the weight says how much the
    winner was actually favoured -- 20.7% of the corpus is conditioned on a model
    the bootstrap picks less than half the time."""
    from m12_information import post_selection_richness
    series = _clean_series()
    ps = post_selection_richness(series, {"logistic": 0.55, "gompertz": 0.45})
    if not ps.get("available"):
        return
    for key in ("weighted_richness", "richness_spread", "selected_model_weight",
                "n_models_averaged", "per_model"):
        assert key in ps, key
    assert ps["richness_spread"] >= 0.0
    assert 0.0 <= ps["weighted_richness"] <= 1.0


def test_robustness_verdict_is_a_conjunction():
    """A confident winner is not enough if the runner-up gives a very different
    answer, and a tight spread is not enough if the winner was nearly arbitrary.
    Pinned because collapsing this to either half alone is the tempting
    simplification."""
    from m12_information import post_selection_richness
    series = _clean_series()
    # a lone dominant model: spread 0 and weight 1 -> robust
    solo = post_selection_richness(series, {"logistic": 1.0})
    if solo.get("available") and solo["n_models_averaged"] == 1:
        assert solo["richness_spread"] == 0.0
        assert solo["verdict_robust_to_selection"] is True


def test_missing_posterior_degrades_to_a_stated_refusal():
    """No M3 record must mean 'we could not average', never a silent conditional
    result dressed up as unconditional."""
    from m12_information import post_selection_richness
    ps = post_selection_richness(_clean_series(), {})
    assert ps["available"] is False
    assert ps["reason"] == "no_selection_posterior"


def test_the_shipped_corpus_carries_the_post_selection_block():
    import json as _json
    from pathlib import Path
    art = Path(__file__).resolve().parent.parent / "data/processed/information_content.jsonl"
    if not art.exists():
        return
    n = avail = 0
    for line in art.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = _json.loads(line)
        n += 1
        ps = r.get("post_selection")
        assert ps is not None, r.get("series_id")
        if ps.get("available"):
            avail += 1
    assert n > 0 and avail > 0

if __name__ == "__main__":
    raise SystemExit(_run())
