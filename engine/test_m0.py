"""
PRISE — Module M0 tests (deterministic, synthetic selections)
=============================================================

Exercises the M0 cohort-assembly contract (§3-M0) on SYNTHETIC selections so the
routing, comparability gate and confound refusal are checked in isolation, fast:

  * 1 curve                                   -> single_curve
  * 2 identical-condition curves              -> replicates (+ replicate meta,
                                                 pooled SE < mean individual SE)
  * >=3 matched varying-concentration curves  -> concentration_series (dual-γ)
  * varying pH, concentration fixed           -> condition_series (surface, no γ)
  * concentration AND pH both varying         -> confounded (mechanism REFUSED)
  * different construct / assay               -> comparability mismatch (hard confound)
  * empty selection                           -> flagged minimal result (no raise)

Run:  python engine/test_m0.py       (exit 0 = all pass)
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m0_cohort import assemble_cohort, check_comparability, detect_data_mode  # noqa: E402


# --------------------------------------------------------------------------- #
#  Synthetic curve builder (a real triaged-series shape M0 + the engine accept)
# --------------------------------------------------------------------------- #
def _logistic(t, t50, k, amp=1.0, base=0.0):
    return base + amp / (1.0 + math.exp(-k * (t - t50)))


def make_curve(series_id, *, conc=10.0, pH=7.4, temp=37.0, assay="ThT",
               construct="Wild Type", agitation=None, seeded=None,
               protein="TestProt", uniprot="P00001", t50=40.0, k=0.15,
               n=30, tmax=100.0, noise_seed=0):
    """A synthetic monotonic logistic curve wrapped in the triaged-series schema
    (condition_vector + a minimal m1 block) that M0/M1..M5 accept."""
    xs = [i * (tmax / (n - 1)) for i in range(n)]
    # tiny deterministic jitter so the fit is non-degenerate but reproducible
    rng = (noise_seed * 2654435761) & 0xFFFFFFFF
    ys = []
    for t in xs:
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        jit = ((rng / 0x7FFFFFFF) - 0.5) * 0.01
        ys.append(_logistic(t, t50, k) + jit)
    return {
        "series_id": series_id,
        "protein_id": protein,
        "uniprot_id": uniprot,
        "data_mode": "kinetic",
        "x_hours": xs,
        "y_intensity": ys,
        "condition_vector": {
            "concentration": {"value_uM": conc, "unit": "microM"},
            "pH": pH,
            "temperature_C": temp,
            "assay_type": assay,
            "assay_reports_mass": assay in ("ThT", "ThS"),
            "construct_id": construct,
            "agitation": agitation,
            "seeded": seeded,
        },
        "m1": {
            "fittability_class": "fittable",
            "censoring_class": "none",
            "recommended_handling": "monotonic_fit",
            "plateau_reached": True,
            "signal_basis": "normalized",
            "y_processed": ys,
        },
        "fittability_class": "fittable",
        "censoring_class": "none",
    }


def _require(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #
def test_single_curve():
    res = assemble_cohort([make_curve("S1")])
    _require(res["data_mode"] == "single_curve",
             f"1 curve -> single_curve, got {res['data_mode']}")
    _require(res["comparability"]["ok"] is True, "single curve must be comparable")
    a = res["assembled_analysis"]
    _require(a["route"] == "single_curve", "route must be single_curve")
    _require(a.get("status") == "ok", f"single-curve route status {a.get('status')}")
    _require(a.get("best_model") is not None, "single-curve route produced no fit")
    # window of validity is present and bounds the (single) concentration
    _require(res["window_of_validity"]["concentration_uM"] is not None,
             "single curve must report a concentration window")
    print("  [ok] single_curve")


def test_replicates():
    # two IDENTICAL-condition curves (same concentration/pH/temp/construct/assay)
    reps = [make_curve("R1", t50=40.0, noise_seed=1),
            make_curve("R2", t50=44.0, noise_seed=2),
            make_curve("R3", t50=42.0, noise_seed=3)]
    res = assemble_cohort(reps)
    _require(res["data_mode"] == "replicates",
             f"identical-condition curves -> replicates, got {res['data_mode']}")
    _require(res["comparability"]["ok"] is True, "replicates must be comparable")
    a = res["assembled_analysis"]
    _require(a["route"] == "replicates", "route must be replicates")
    _require(a.get("status") == "ok",
             f"replicate route status {a.get('status')}: {a.get('reason')}")
    # calls the replicate random-effects meta-analysis
    meta = a.get("replicate_meta")
    _require(meta is not None, "replicate route must call the replicate meta-analysis")
    _require("tau2_between" in a, "replicate route must report tau2_between")
    _require("stochastic_nucleation_signal" in a,
             "replicate route must retain the stochastic-nucleation scatter signal")
    # pooled SE < mean individual sampling SD (the point of pooling)
    _require(a.get("pooled_se_lt_individual") is True,
             "pooled SE must be < mean individual sampling SD")
    print("  [ok] replicates (pooled SE < individual; tau2 + scatter present)")


def test_concentration_series():
    # >=3 distinct concentrations, everything else matched -> concentration_series
    curves = []
    for i, c in enumerate([5.0, 10.0, 20.0, 40.0, 80.0]):
        # t50 falls with concentration (nucleation-like): t50 ~ c^-gamma
        t50 = 60.0 * (c / 10.0) ** (-0.5)
        curves.append(make_curve(f"C{i}", conc=c, t50=t50, k=0.2, noise_seed=10 + i))
    res = assemble_cohort(curves, run_global_ode=False)
    _require(res["data_mode"] == "concentration_series",
             f">=3 conc matched -> concentration_series, got {res['data_mode']}")
    _require(res["comparability"]["ok"] is True, "conc series must be comparable")
    a = res["assembled_analysis"]
    _require(a["route"] == "concentration_series", "route must be concentration_series")
    # routes to dual_gamma (M4)
    _require("dual_gamma" in a, "concentration_series must route to dual_gamma")
    _require("gamma_regression" in a and "gamma_global" in a,
             "concentration_series must report both γ estimators")
    reg = a["gamma_regression"] or {}
    _require(reg.get("status") == "ok",
             f"gamma_regression status {reg.get('status')}")
    # window of validity spans the measured concentration decade
    wov = res["window_of_validity"]["concentration_uM"]
    _require(wov["min"] == 5.0 and wov["max"] == 80.0,
             "concentration window must span the measured range")
    print(f"  [ok] concentration_series (dual-gamma: gamma_reg={reg.get('gamma')})")


def test_condition_series():
    # pH varies, concentration fixed, otherwise matched -> condition_series (no γ)
    curves = []
    for i, ph in enumerate([6.0, 6.5, 7.0, 7.5, 8.0]):
        curves.append(make_curve(f"P{i}", conc=10.0, pH=ph, t50=40.0 + i * 4,
                                  noise_seed=20 + i))
    res = assemble_cohort(curves)
    _require(res["data_mode"] == "condition_series",
             f"varying pH conc-fixed -> condition_series, got {res['data_mode']}")
    a = res["assembled_analysis"]
    _require(a["route"] == "condition_series", "route must be condition_series")
    _require(a.get("surface_axis") == "pH", "surface axis must be pH")
    _require("gamma" not in a and "dual_gamma" not in a,
             "condition_series must NOT emit a γ")
    _require(a.get("status") == "ok" and a.get("surface"),
             "condition_series must produce a feature-vs-condition surface")
    _require(res["window_of_validity"]["pH"]["min"] == 6.0,
             "pH window must bound validity")
    print("  [ok] condition_series (feature-vs-pH surface, no gamma)")


def test_confounded():
    # concentration AND pH both varying -> confounded (mechanism refused)
    curves = []
    for i, (c, ph) in enumerate([(5.0, 6.0), (10.0, 6.5), (20.0, 7.0), (40.0, 7.5)]):
        curves.append(make_curve(f"X{i}", conc=c, pH=ph, t50=50.0 - i * 5,
                                  noise_seed=30 + i))
    res = assemble_cohort(curves)
    _require(res["data_mode"] == "confounded",
             f"conc AND pH varying -> confounded, got {res['data_mode']}")
    a = res["assembled_analysis"]
    _require(a["route"] == "confounded", "route must be confounded")
    _require(a.get("mechanism_refused") is True, "confounded must REFUSE mechanism")
    axes = set(a.get("confound_axes", []))
    _require({"concentration_uM", "pH"} <= axes,
             f"confound_axes must name the co-varying axes, got {axes}")
    _require(a.get("confound_flag", {}).get("flagged") is True,
             "confounded must carry a prominent confound flag")
    _require("Invariant 19" in a["confound_flag"]["invariant"],
             "confound flag must cite Invariant 19")
    # no γ, no dual_gamma anywhere in the assembled analysis
    _require("dual_gamma" not in a and "gamma_regression" not in a,
             "confounded must emit NO γ")
    # descriptive-only per curve is still provided
    _require(a.get("per_curve_descriptive"), "confounded must emit per-curve descriptive")
    print("  [ok] confounded (mechanism REFUSED + confound flag naming conc+pH)")


def test_comparability_mismatch_construct():
    # different construct_id across the selection -> HARD comparability confound
    curves = [make_curve("M1", construct="Wild Type", noise_seed=40),
              make_curve("M2", construct="A53T", noise_seed=41),
              make_curve("M3", construct="Wild Type", noise_seed=42)]
    res = assemble_cohort(curves)
    comp = res["comparability"]
    _require(comp["ok"] is False, "mixed construct must fail comparability")
    _require("construct_id" in comp["mismatched_fields"],
             "construct_id must be flagged as mismatched")
    _require(comp["confound_flag"] is True, "mixed construct must raise confound_flag")
    _require(res["data_mode"] == "confounded",
             "hard comparability mismatch -> confounded routing")
    a = res["assembled_analysis"]
    _require(a.get("mechanism_refused") is True,
             "comparability confound must refuse mechanism")
    print("  [ok] comparability mismatch (construct) -> hard confound, mechanism refused")


def test_comparability_mismatch_assay():
    # different assay_type -> HARD comparability confound (even if conc series otherwise)
    curves = [make_curve("A1", conc=5.0, assay="ThT", noise_seed=50),
              make_curve("A2", conc=10.0, assay="ThT", noise_seed=51),
              make_curve("A3", conc=20.0, assay="turbidity", noise_seed=52)]
    res = assemble_cohort(curves, run_global_ode=False)
    comp = res["comparability"]
    _require(comp["ok"] is False, "mixed assay must fail comparability")
    _require("assay_type" in comp["mismatched_fields"], "assay_type must be flagged")
    _require(res["data_mode"] == "confounded",
             "mixed assay -> confounded (not concentration_series)")
    print("  [ok] comparability mismatch (assay) -> hard confound, not conc-series")


def test_degraded_missing_field():
    # construct_id UNKNOWN (None) for some members, no contradiction -> degrade, widen
    c1 = make_curve("D1", noise_seed=60)
    c2 = make_curve("D2", noise_seed=61)
    c1["condition_vector"]["construct_id"] = None      # unknown, not mismatched
    res = assemble_cohort([c1, c2])
    comp = res["comparability"]
    _require(comp["ok"] is True, "an UNKNOWN field must not be a hard mismatch")
    degraded = [d["field"] for d in comp["degraded_fields"]]
    _require("construct_id" in degraded,
             "missing construct_id must be a DEGRADED field (widen + flag)")
    print("  [ok] missing comparability field -> degrade (widen), not silent merge")


def test_empty_selection():
    res = assemble_cohort([])
    _require(res["data_mode"] == "empty", "empty selection -> data_mode empty")
    _require(res["assembled_analysis"]["status"] == "empty",
             "empty selection -> flagged minimal result")
    # never raised, always JSON-serialisable
    import json
    json.dumps(res)
    # a malformed member is dropped, not raised
    res2 = assemble_cohort(["not-a-dict", 42, None])
    _require(res2["data_mode"] == "empty", "all-malformed selection -> empty")
    print("  [ok] empty / malformed selection -> flagged minimal result (no raise)")


def test_never_raises_and_json():
    # a member missing condition fields entirely must not raise
    import json
    bad = {"series_id": "B1", "x_hours": [0, 1, 2], "y_intensity": [0, 0.5, 1]}
    res = assemble_cohort([bad])
    json.dumps(res)                     # must be JSON-serialisable
    _require("data_mode" in res, "result must always carry a data_mode")
    print("  [ok] never raises + always JSON-serialisable")


def test_units_used_for_comparability():
    # uniprot mismatch is a protein mismatch (identity comparability)
    c1 = make_curve("U1", uniprot="P00001", noise_seed=70)
    c2 = make_curve("U2", uniprot="P99999", noise_seed=71)
    comp = check_comparability([c1, c2])
    _require(comp["ok"] is False, "different uniprot -> protein mismatch")
    _require("protein" in comp["mismatched_fields"], "protein field must be flagged")
    print("  [ok] protein/uniprot identity is comparability-gated")


def main():
    tests = [
        test_single_curve,
        test_replicates,
        test_concentration_series,
        test_condition_series,
        test_confounded,
        test_comparability_mismatch_construct,
        test_comparability_mismatch_assay,
        test_degraded_missing_field,
        test_empty_selection,
        test_never_raises_and_json,
        test_units_used_for_comparability,
    ]
    print(f"[test_m0] running {len(tests)} deterministic M0 tests ...")
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failures.append(f"{t.__name__}: {exc}")
        except Exception as exc:  # a route that raised is itself a failure
            failures.append(f"{t.__name__}: UNEXPECTED {type(exc).__name__}: {exc}")
    if failures:
        for f in failures:
            sys.stderr.write("FAIL: %s\n" % f)
        print("test_m0: FAILED (%d)" % len(failures))
        return 1
    print("test_m0: OK — all %d M0 tests passed" % len(tests))
    return 0


if __name__ == "__main__":
    sys.exit(main())
