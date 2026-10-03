"""
Tests for the external forward model — the de-circularisation generator.

    python engine/test_external_forward_model.py
    pytest engine/test_external_forward_model.py

WHAT THESE GUARD

The generator exists to break a circularity: Service C's "mismatched generator"
was `mechanistic.simulate_mass_fraction`, the SAME function the engine fits, so
mechanism-recovery results were the engine recovering its own model. A
replacement generator is only worth anything if two things hold, and both are
easy to get wrong in opposite directions:

  * it must be the SAME PHYSICS -- otherwise a recovery failure just means the
    data was nonsense, not that the engine is misspecified. Pinned by mass
    conservation and by agreement with the closure in the limit where the
    closure is provably valid.
  * it must be GENUINELY DIFFERENT where the closure's assumptions fail --
    otherwise generating from it is theatre. Pinned by requiring a material
    discrepancy in the secondary-nucleation and depletion regimes.
"""
from __future__ import annotations

import numpy as np

from external_forward_model import (
    EXTERNAL_MODEL_VERSION,
    closure_discrepancy,
    external_mechanism_identification,
    manifest,
    paired_generation,
    replicate_cases,
    simulate_size_resolved,
)

T = np.linspace(0.0, 60.0, 80)


# ===================== it is the same physics ============================= #
def test_mass_is_conserved_to_machine_precision():
    """The master equation must not create or destroy protein. This is the check
    that the fragmentation kernel's loss and gain terms actually balance --
    getting that wrong is the classic way to write a plausible-looking but
    non-physical coagulation model."""
    for rates in ({"kn": 1e-4, "kp": 1.0},
                  {"kn": 1e-5, "kp": 1.0, "k2": 1e-2},
                  {"kn": 1e-4, "kp": 1.0, "kminus": 1e-3}):
        r = simulate_size_resolved(T, **rates)
        assert r["ok"], (rates, r["reason"])
        assert r["mass_conservation_error"] < 1e-10, (rates, r)


def test_mass_fraction_is_a_physical_trajectory():
    r = simulate_size_resolved(T, kn=1e-4, kp=1.0)
    mf = np.asarray(r["mass_fraction"])
    assert len(mf) == len(T)
    assert mf[0] <= 1e-9                      # starts unaggregated
    assert np.all(mf >= -1e-12) and np.all(mf <= 1.0 + 1e-12)
    assert np.all(np.diff(mf) >= -1e-9)       # monomer only converts one way here
    assert r["final_mean_length"] and r["final_mean_length"] > 1.0


def test_size_axis_truncation_does_not_bind():
    """If aggregates pile up in the top bin the size axis was too short and the
    trajectory is an artefact of the truncation, not of the physics."""
    for rates in ({"kn": 1e-4, "kp": 1.0}, {"kn": 1e-5, "kp": 1.0, "k2": 1e-2}):
        r = simulate_size_resolved(T, **rates)
        assert r["truncation_occupancy"] < 1e-6, (rates, r["truncation_occupancy"])


def test_it_agrees_with_the_closure_where_the_closure_is_valid():
    """The two-moment closure is derived from this master equation and is a good
    approximation for primary nucleation plus elongation. If they DISAGREED here,
    the generator would be modelling different chemistry and no recovery result
    from it would be interpretable."""
    d = closure_discrepancy(T, kn=1e-4, kp=1.0, k2=0.0, kminus=0.0)
    assert d["ok"], d
    assert d["rmse"] < 0.02, d
    assert d["max_abs"] < 0.05, d


# ===================== ...and genuinely different ========================= #
def test_it_diverges_from_the_closure_where_the_closure_fails():
    """The whole point. If the closure reproduced the master equation everywhere,
    generating from the master equation would tell us nothing the internal
    generator could not. Secondary nucleation is the regime that matters most --
    it is what PRISE's mechanistic layer exists to detect."""
    sec = closure_discrepancy(T, kn=1e-5, kp=1.0, k2=1e-2, kminus=0.0)
    assert sec["ok"], sec
    assert sec["max_abs"] > 0.05, sec        # measured ~0.11 absolute mass fraction

    dep = closure_discrepancy(T, kn=1e-3, kp=0.05, k2=0.0, kminus=0.0)
    assert dep["ok"], dep
    assert dep["max_abs"] > 0.03, dep        # strong monomer depletion


def test_the_generator_is_not_the_function_the_engine_fits():
    """Structural, not stylistic: the engine's model tracks two moments and has no
    size axis at all, so it cannot report a mean aggregate length. If this module
    ever became a wrapper around it, that distinction would vanish."""
    import mechanistic
    r = simulate_size_resolved(T, kn=1e-4, kp=1.0)
    assert r["final_mean_length"] is not None
    assert "simulate_size_resolved" not in dir(mechanistic)
    man = manifest()
    assert "no moment closure" in man["formulation"].lower().replace("_", " ")
    # and it declares what it does NOT cover, rather than implying completeness.
    # (It used to say "no Gillespie arm"; that gap is now closed, so the claim
    # moved on to what is genuinely still missing rather than being deleted.)
    assert "artifact-realism" in man["not_covered"]


# ===================== the measurement it licenses ======================== #
def test_paired_generation_changes_only_the_forward_model():
    """Pairing is what makes the comparison mean anything: if the two arms used
    different parameters, misspecification would be confounded with difficulty."""
    pair = paired_generation(t=T)
    for name, rec in pair["cases"].items():
        assert rec["external_ok"], (name, rec)
        assert rec["internal"] is not None, name
        assert len(rec["internal"]) == len(rec["external"]) == len(T), name


def test_mechanism_identification_reports_both_arms_and_a_ceiling():
    """The internal arm is the CEILING -- data from the very model being selected.
    Reporting only the external number would make an identifiability limit look
    like a misspecification penalty, which is the misreading this whole module
    exists to prevent."""
    r = external_mechanism_identification(t=T, cases=replicate_cases(2))
    assert r["ok"], r
    assert r["internal_n"] == r["external_n"] > 0
    for arm in ("internal_accuracy", "external_accuracy"):
        assert 0.0 <= r[arm] <= 1.0, r
    # ground truth must be resolved for every replicate, suffix and all
    for name, row in r["per_case"].items():
        assert row["ground_truth"] is not None, name


def test_replicates_are_deterministic_and_preserve_the_class():
    a = replicate_cases(4)
    b = replicate_cases(4)
    assert a == b                                  # frozen ladder, no RNG
    assert len(a) == 4 * 4
    for name in a:
        assert "__r" in name
        base = name.split("__")[0]
        # only the nucleation channels are scaled; a zero rate stays zero, so the
        # mechanism CLASS cannot drift between replicates
        for k in ("k2", "kminus"):
            if not _base_rate(base, k):
                assert not a[name].get(k)


def _base_rate(base, key):
    from external_forward_model import _MECHANISM_CASES
    return _MECHANISM_CASES[base].get(key)


# ===================== the stochastic arm ================================= #
def test_gillespie_is_reproducible_given_a_seed():
    """A stochastic generator still has to be reproducible, or a validation run
    cannot be repeated -- the same rule the rest of the engine follows."""
    from external_forward_model import simulate_gillespie
    a = simulate_gillespie(T, kn=1e-4, kp=1.0, n_molecules=800, seed=3)
    b = simulate_gillespie(T, kn=1e-4, kp=1.0, n_molecules=800, seed=3)
    c = simulate_gillespie(T, kn=1e-4, kp=1.0, n_molecules=800, seed=4)
    assert a["ok"] and b["ok"] and c["ok"]
    assert a["mass_fraction"] == b["mass_fraction"]
    assert a["mass_fraction"] != c["mass_fraction"]      # ...and genuinely random


def test_gillespie_mean_converges_to_the_deterministic_model():
    """THE CORRECTNESS CHECK. The SSA mean must approach the master equation as
    the system grows; if it did not, the stochastic arm would be simulating
    different chemistry and its spread would mean nothing."""
    from external_forward_model import gillespie_ensemble
    det = np.asarray(simulate_size_resolved(T, kn=1e-4, kp=1.0)["mass_fraction"])

    def rmse(N):
        e = gillespie_ensemble(T, n_reps=8, base_seed=0, kn=1e-4, kp=1.0,
                               n_molecules=N)
        assert e["ok"], N
        return float(np.sqrt(np.mean((np.asarray(e["mean"]) - det) ** 2)))

    small, large = rmse(500), rmse(20000)
    assert large < small, (small, large)
    assert large < 0.02, large


def test_intrinsic_noise_shrinks_with_system_size():
    """The well-to-well CV is the finding, so its scaling has to be right: a
    rare-event process fluctuates more in a small system. A CV that did NOT fall
    with N would mean the spread was an artefact of the sampler."""
    from external_forward_model import gillespie_ensemble
    cvs = []
    for N in (500, 20000):
        e = gillespie_ensemble(T, n_reps=10, base_seed=0, kn=1e-4, kp=1.0,
                               n_molecules=N)
        cvs.append(e["well_to_well_cv_t50"])
    assert all(c is not None for c in cvs), cvs
    assert cvs[0] > cvs[1], cvs
    assert cvs[0] > 0.05, cvs          # and it is MATERIAL at small N, not a rounding


def test_stochastic_report_states_both_the_check_and_the_finding():
    from external_forward_model import stochastic_noise_report
    r = stochastic_noise_report(t=T, sizes=(500, 8000), n_reps=6)
    assert r["ok"], r
    assert r["converges_to_deterministic"] is True, r
    assert len(r["by_system_size"]) == 2
    for row in r["by_system_size"]:
        assert row["rmse_mean_vs_deterministic"] >= 0.0
    # the interpretation must name the misattribution risk, not just the numbers
    assert "measurement noise" in r["interpretation"]


def test_manifest_no_longer_claims_stochastic_noise_is_untested():
    from external_forward_model import manifest
    man = manifest()
    assert "stochastic_arm" in man
    assert "gillespie" in man["stochastic_arm"].lower()
    # ...and still declares what genuinely is not covered
    assert "artifact-realism" in man["not_covered"]


def _run():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}"); failed += 1
    print("")
    print(str(passed) + " passed, " + str(failed) + " failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
