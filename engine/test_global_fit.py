"""
Tests for global_fit — the real shared-rate Knowles/Cohen GLOBAL fit (the
mechanistic centrepiece PRISE_DESIGN.md §3-M2/§3-M4 promised).

These are deterministic tests on SYNTHETIC data: we generate a multi-
concentration family from mechanistic.simulate_mass_fraction with KNOWN shared
rates + KNOWN reaction orders (n_c, n_2), fit globally with SHARED params, and
assert:
  * the shared fit recovers the reaction-order / γ constraint (the novelty —
    orders are ESTIMATED, not fixed at 2.0 like mechanistic.py),
  * γ_mechanistic matches the analytic (n_2+1)/2 value,
  * identifiability is reported and honestly flags sloppiness on individual rates
    (this corpus mostly CANNOT pin unique rates — the correct finding),
  * the shared fit constrains reaction order strictly better than a per-curve
    fit can (per-curve mechanistic.py hard-fixes the order),
  * determinism (same seed -> identical numbers).

We restrict mechanism families per test (the stiff ODE × all families × many
curves is slow) and keep families small — runtime stays a few seconds.

    python engine/test_global_fit.py
    pytest engine/test_global_fit.py
"""
from __future__ import annotations

import numpy as np

from global_fit import (
    GLOBAL_FIT_VERSION,
    _gamma_mechanistic,
    fit_global_series,
)
from mechanistic import simulate_mass_fraction


# --------------------------------------------------------------------------- #
#  synthetic multi-concentration family with KNOWN shared rates + orders
# --------------------------------------------------------------------------- #
def _family(concs, kn=1e-4, kp=2.0, k2=5e-2, nc=2.0, n2=2.0,
            tmax=80.0, n=20, noise=0.5, base=2.0, amp=100.0, seed0=0):
    members = []
    for i, m in enumerate(concs):
        rng = np.random.default_rng(seed0 + i)
        x = np.linspace(0.0, tmax, n)
        frac = simulate_mass_fraction(x, kn=kn, kp=kp, k2=k2, nc=nc, n2=n2, mtot=m)
        y = base + amp * frac + rng.normal(0, noise, size=len(x))
        members.append({
            "series_id": f"syn-m{m}",
            "x_hours": list(map(float, x)),
            "condition_vector": {"concentration": {"value_uM": float(m)}},
            "m1": {"censoring_class": "none",
                   "y_processed": list(map(float, y))},
        })
    return members


# --------------------------------------------------------------------------- #
#  analytic γ_mechanistic
# --------------------------------------------------------------------------- #
def test_gamma_mechanistic_formula_matches_meisl():
    # secondary nucleation: γ = (n_2+1)/2
    g, _ = _gamma_mechanistic("secondary_nucleation", nc=2.0, n2=2.0)
    assert abs(g - 1.5) < 1e-12
    g, _ = _gamma_mechanistic("secondary_nucleation", nc=2.0, n2=1.0)
    assert abs(g - 1.0) < 1e-12
    # primary nucleation+elongation: γ = (n_c+1)/2
    g, _ = _gamma_mechanistic("nucleation_elongation", nc=3.0, n2=2.0)
    assert abs(g - 2.0) < 1e-12
    # fragmentation: γ = 1/2
    g, _ = _gamma_mechanistic("fragmentation", nc=2.0, n2=2.0)
    assert abs(g - 0.5) < 1e-12


# --------------------------------------------------------------------------- #
#  shared fit recovers the reaction-order constraint + analytic γ_mechanistic
# --------------------------------------------------------------------------- #
def test_shared_fit_recovers_reaction_order_and_gamma():
    members = _family([5, 10, 20, 40], nc=2.0, n2=2.0)
    res = fit_global_series(members, meta={"concentration_series_id": "SYN"},
                            gamma_hint=1.5,
                            mechanisms=("secondary_nucleation",), seed=0)
    assert res["status"] == "ok"
    ro = res["reaction_orders"]
    # n_2 is ESTIMATED (not fixed) and recovered near the true 2.0
    assert ro["n_2_free"] is True
    assert abs(ro["n_2"] - 2.0) < 0.6
    # γ_mechanistic = (n_2+1)/2 matches the analytic value AND the formula applied
    assert abs(res["gamma_mechanistic"] - (ro["n_2"] + 1.0) / 2.0) < 1e-9
    assert abs(res["gamma_mechanistic"] - 1.5) < 0.4
    # the global fit is essentially exact on clean synthetic data
    assert res["best_fit"]["global_r2"] > 0.97


def test_shared_fit_recovers_lower_reaction_order():
    """A family with n_2 = 1 must drive γ_mechanistic toward 1.0 — i.e. the order is
    genuinely estimated from the data, not anchored to a default."""
    members = _family([5, 10, 20, 40], n2=1.0, seed0=100)
    res = fit_global_series(members, meta={"concentration_series_id": "SYN1"},
                            gamma_hint=1.0,
                            mechanisms=("secondary_nucleation",), seed=0)
    assert res["status"] == "ok"
    # estimated n_2 clearly below the n_2=2 family (order is data-driven)
    assert res["reaction_orders"]["n_2"] < 1.8
    assert res["gamma_mechanistic"] < 1.5


# --------------------------------------------------------------------------- #
#  honest sloppiness: identifiability is reported; individual rates are sloppy
# --------------------------------------------------------------------------- #
def test_identifiability_reports_sloppiness():
    members = _family([5, 10, 20, 40], seed0=7)
    res = fit_global_series(members, meta={"concentration_series_id": "SYN"},
                            gamma_hint=1.5,
                            mechanisms=("secondary_nucleation",), seed=0)
    ident = res["identifiability"]
    assert ident["computable"] is True
    # FIM is ill-conditioned -> sloppy is the correct, expected finding
    assert ident["fim_condition_number"] is None or ident["fim_condition_number"] > 1e3
    assert res["sloppy"] is True
    # identifiable COMBINATIONS are reported (stiff directions), and at least one
    # sloppy direction exists (the honest deliverable)
    assert len(ident["identifiable_combinations"]) >= 1
    assert ident["n_stiff_directions"] >= 1
    # at least one direction is genuinely sloppy on this design
    assert ident["n_sloppy_directions"] >= 1 or ident["fim_condition_number"] > 1e6
    assert "individual_rate_status" in ident


def test_shared_fit_beats_percurve_on_reaction_order():
    """The shared fit ESTIMATES the reaction order; per-curve mechanistic.py
    HARD-FIXES it at 2.0 and can never report a different one. So the shared fit's
    constraint on the order is strictly more informative — demonstrated by feeding
    a family whose true n_2 = 1.3 and showing the shared estimate moves toward it
    while the per-curve order is stuck at the 2.0 default."""
    from mechanistic import fit_mechanistic
    members = _family([5, 10, 20, 40], n2=1.3, seed0=42)
    res = fit_global_series(members, meta={"concentration_series_id": "SYN13"},
                            gamma_hint=1.15,
                            mechanisms=("secondary_nucleation",), seed=0)
    shared_n2 = res["reaction_orders"]["n_2"]
    # per-curve fit (single curve) cannot estimate n_2 at all — it is fixed at 2.0
    one = members[0]
    pc = fit_mechanistic(one["x_hours"],
                         one["m1"]["y_processed"], variant="secondary_nucleation")
    assert pc.get("n2") == 2.0                      # per-curve order is hard-fixed
    # the shared estimate is closer to the truth (1.3) than the fixed 2.0 default
    assert abs(shared_n2 - 1.3) < abs(2.0 - 1.3)


# --------------------------------------------------------------------------- #
#  determinism + plumbing
# --------------------------------------------------------------------------- #
def test_determinism_same_seed_identical():
    members = _family([5, 10, 20, 40], seed0=3)
    kw = dict(meta={"concentration_series_id": "SYN"}, gamma_hint=1.5,
              mechanisms=("secondary_nucleation",), seed=0)
    a = fit_global_series(members, **kw)
    b = fit_global_series(members, **kw)
    assert a["reaction_orders"]["n_2"] == b["reaction_orders"]["n_2"]
    assert a["gamma_mechanistic"] == b["gamma_mechanistic"]
    assert a["best_fit"]["global_sse"] == b["best_fit"]["global_sse"]


def test_mechanism_aicc_ranking_present_when_multiple_families():
    members = _family([5, 10, 20], seed0=11)
    res = fit_global_series(
        members, meta={"concentration_series_id": "SYN"}, gamma_hint=1.5,
        mechanisms=("nucleation_elongation", "secondary_nucleation"), seed=0)
    assert res["status"] == "ok"
    rank = res["mechanism_aicc_ranking"]
    assert len(rank) == 2
    # ranked ascending by AICc; best has delta 0
    assert rank[0]["delta_aicc"] == 0.0
    assert rank[1]["delta_aicc"] >= 0.0
    assert res["best_mechanism"] == rank[0]["mechanism"]


def test_insufficient_concentrations_reported():
    members = _family([5, 10], seed0=0)
    res = fit_global_series(members, meta={"concentration_series_id": "SYN"})
    assert res["status"] == "insufficient_data"
    assert res["version"] == GLOBAL_FIT_VERSION


def test_payload_is_json_serialisable_and_honest():
    import json
    members = _family([5, 10, 20], seed0=5)
    res = fit_global_series(members, meta={"concentration_series_id": "SYN",
                                           "protein": "syn"}, gamma_hint=1.5,
                            mechanisms=("secondary_nucleation",), seed=0)
    s = json.dumps(res)                              # must not raise
    assert "honesty_note" in res
    assert "does NOT license a single mechanism" in res["honesty_note"]
    assert isinstance(json.loads(s), dict)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}"); failed += 1
        except Exception as e:                       # noqa
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
