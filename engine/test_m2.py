"""
Tests for M2 — model recovery (kinetic + dose-response) and routing.

    python engine/test_m2.py        # standalone PASS/FAIL
    pytest engine/test_m2.py
"""
from __future__ import annotations

import numpy as np

from m2_fit import fit_curve, fit_dose_response, fit_one
from models import (
    m_dr_hill,
    m_dr_lnt,
    m_dr_threshold,
    m_exponential,
    m_finke_watzky,
    m_gompertz,
    m_logistic,
    m_richards,
    m_scaling,
)

RNG = np.random.default_rng(11)
T = np.arange(0, 40, 1.0)


def _noisy(yc, sd=0.004):
    return list(np.asarray(yc) + RNG.normal(0, sd, size=len(yc)))


# -------------------------- kinetic recovery ------------------------------- #
def test_logistic_recovery():
    r = fit_one(list(T), _noisy(m_logistic(T, 0, 1, 0.4, 20)), "logistic")
    assert r["converged"] and r["r2"] > 0.99
    assert abs(r["params"]["t0"] - 20) < 1.5 and abs(r["params"]["k"] - 0.4) / 0.4 < 0.2


def test_gompertz_recovery():
    r = fit_one(list(T), _noisy(m_gompertz(T, 0, 1, 0.3, 15)), "gompertz")
    assert r["converged"] and r["r2"] > 0.99 and abs(r["params"]["t0"] - 15) < 2


def test_richards_recovery():
    r = fit_one(list(T), _noisy(m_richards(T, 0, 1, 0.4, 18, 2.0)), "richards")
    assert r["converged"] and r["r2"] > 0.99, r


def test_exponential_recovery():
    r = fit_one(list(T), _noisy(m_exponential(T, 0, 1, 0.15)), "exponential")
    assert r["converged"] and r["r2"] > 0.99 and abs(r["params"]["k"] - 0.15) / 0.15 < 0.2


def test_scaling_law_recovery():
    r = fit_one(list(T), _noisy(0.5 * np.power(np.clip(T, 1e-9, None), 1.5), sd=0.5),
                "scaling_law")
    assert r["converged"] and r["r2"] > 0.99 and abs(r["params"]["n"] - 1.5) < 0.2


def test_finke_watzky_recovery():
    r = fit_one(list(T), _noisy(m_finke_watzky(T, 0, 1, 0.02, 0.5)), "finke_watzky")
    assert r["converged"] and r["r2"] > 0.99, r
    assert r["params"]["k1"] < r["params"]["k2"]   # slow nucleation, fast autocatalysis


def test_brain_cousens_fits_biphasic():
    # rise then decline
    y = np.where(T < 12, T / 12.0, np.maximum(1.2 - 0.03 * (T - 12), 0.1))
    r = fit_one(list(T), _noisy(y, sd=0.02), "brain_cousens")
    assert r["converged"] and r["r2"] > 0.9, r


# ------------------------- dose-response recovery -------------------------- #
DOSES = [0.5, 1, 2, 3, 5, 8, 12, 20, 35, 60]


def test_dose_hill_recovery():
    resp = list(m_dr_hill(np.array(DOSES), 0.0, 1.0, 5.0, 2.0))
    res = fit_dose_response(DOSES, resp)
    hill = res["fits"]["dr_hill"]
    assert hill["converged"] and hill["r2"] > 0.99
    assert abs(hill["params"]["ec50"] - 5.0) / 5.0 < 0.2


def test_dose_lnt_recovery():
    resp = list(m_dr_lnt(np.array(DOSES), 0.1, 0.05))
    res = fit_dose_response(DOSES, resp)
    assert res["fits"]["dr_lnt"]["converged"]
    assert abs(res["fits"]["dr_lnt"]["params"]["slope"] - 0.05) < 0.01


def test_dose_threshold_recovery():
    resp = list(m_dr_threshold(np.array(DOSES), 0.2, 10.0, 0.04))
    res = fit_dose_response(DOSES, resp)
    thr = res["fits"]["dr_threshold"]
    assert thr["converged"] and abs(thr["params"]["tau"] - 10.0) < 4.0, thr


# ------------------------------- routing ----------------------------------- #
def test_monotonic_includes_mechanistic_and_sigmoidal():
    s = {"series_id": "t", "x_hours": list(T),
         "m1": {"recommended_handling": "monotonic_fit",
                "y_processed": _noisy(m_logistic(T, 0, 1, 0.5, 20))}}
    res = fit_curve(s)
    for m in ("logistic", "gompertz", "richards", "finke_watzky", "scaling_law"):
        assert m in res["candidates"], res["candidates"]
    assert res["best_by_aicc"] is not None


def test_nonmonotonic_uses_brain_cousens_not_sigmoid():
    s = {"series_id": "bi", "x_hours": list(T),
         "m1": {"recommended_handling": "nonmonotonic_fit",
                "y_processed": list(np.exp(-((T - 15) / 6.0) ** 2))}}
    res = fit_curve(s)
    assert "brain_cousens" in res["candidates"]
    assert "logistic" not in res["candidates"]


# ------------------------------- runner ------------------------------------ #
def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
