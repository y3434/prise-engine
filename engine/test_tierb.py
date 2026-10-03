"""
Tests for Tier B — Knowles/Cohen mechanistic ODE family (mechanistic.py).

Validates the simulator's qualitative behaviour and that fits achieve high R²
on synthetic mechanistic curves. NOTE: we deliberately do NOT assert exact rate
recovery — these models are *sloppy* (only certain rate combinations are
identifiable from a single curve), which is precisely why M3 adds an
identifiability gate. The tests document that "fits well" != "rates recovered".

    python engine/test_tierb.py
    pytest engine/test_tierb.py
"""
from __future__ import annotations

import numpy as np

from mechanistic import VARIANT_RATES, fit_mechanistic, simulate_mass_fraction

RNG = np.random.default_rng(3)
T = np.arange(0, 50, 1.0)


def _noisy(y, sd=0.005):
    return list(np.asarray(y) + RNG.normal(0, sd, size=len(y)))


def test_simulator_is_lag_rise_plateau():
    f = simulate_mass_fraction(T, kn=1e-4, kp=2.0, k2=5e-2)
    assert f[0] < 0.05 and f[-1] > 0.9              # lag at start, plateau at end
    assert np.all(np.diff(f) >= -1e-9)               # monotonically non-decreasing
    assert np.all((f >= -1e-9) & (f <= 1 + 1e-9))    # mass fraction in [0,1]


def test_secondary_nucleation_fit_high_r2():
    y = _noisy(simulate_mass_fraction(T, kn=1e-4, kp=2.0, k2=5e-2))
    r = fit_mechanistic(T, y, "secondary_nucleation")
    assert r["converged"] and r["r2"] > 0.97, r
    assert set(r["params"]) >= {"base", "amp", "kn", "kp", "k2"}


def test_nucleation_elongation_fit_high_r2():
    y = _noisy(simulate_mass_fraction(T, kn=1e-3, kp=2.0, k2=0.0))
    r = fit_mechanistic(T, y, "nucleation_elongation")
    assert r["converged"] and r["r2"] > 0.97, r


def test_secondary_sharper_than_pure_elongation():
    # secondary nucleation autocatalyses -> steeper transition than pure NE
    f_sec = simulate_mass_fraction(T, kn=1e-4, kp=2.0, k2=1e-1)
    f_ne = simulate_mass_fraction(T, kn=1e-4, kp=2.0, k2=0.0)
    assert np.max(np.diff(f_sec)) > np.max(np.diff(f_ne))


def test_all_variants_fit_without_error():
    y = _noisy(simulate_mass_fraction(T, kn=1e-4, kp=2.0, k2=5e-2))
    for variant in VARIANT_RATES:
        r = fit_mechanistic(T, y, variant)
        assert "converged" in r, (variant, r)


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
