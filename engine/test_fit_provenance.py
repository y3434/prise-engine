"""
Tests for §2.3 FitProvenance and the §7 basin-stability gate — the invariant
§10.12 that carried a `[test]` tag it had never earned.

    python engine/test_fit_provenance.py     # standalone PASS/FAIL
    pytest engine/test_fit_provenance.py

WHY THIS FILE EXISTS, AND WHAT IT IS GUARDING AGAINST

Invariant §10.12 ("Every parameter carries fit provenance; anchors are
tolerance-reproducible ... and basin-stable") was tagged `[test]` while having
neither an implementation nor a test — discovered by M17 and recorded in commit
`a6c9f88`. This file is what makes the tag true.

Four positive tests assert the record is emitted and complete. Four NEGATIVE
CONTROLS assert the harder half — that the record is *honest*:

  * a detector that never fires passes every positive test while detecting
    nothing, so `test_knife_edge_is_actually_detected` proves it fires;
  * a detector that fires everywhere is equally useless, so
    `test_clean_fit_is_stable_not_knife_edge` proves it discriminates;
  * being UNABLE to assess a fit must never be silently upgraded to `stable`, so
    `test_a_single_start_fit_is_still_never_reported_stable` proves that case is
    reported. (Until `estimation-policy-1.0` this was the STRUCTURAL state of 9 of
    the 13 registry models, which built only one start; the companion
    `test_every_registry_model_now_gets_a_genuine_multi_start` pins that it no
    longer is, and the negative control above pins that the honest verdict still
    survives where it is genuinely earned);
  * and the sharpest one, `test_m2_never_fabricates_an_rng_seed`, guards against
    the specific bad fix this invariant invites.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import sys

import numpy as np

import fit_provenance as fp
from m2_fit import fit_curve, fit_dose_response, fit_one
from models import m_dr_hill, m_logistic

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

# The frozen deterministic fixtures. A module-local generator on purpose:
# test_m2.py shares one module-level RNG across its tests in name order, so a
# shared stream makes one test's data depend on which other tests exist.
_RNG = np.random.default_rng(20260804)
T = np.arange(0, 40, 1.0)

_REQUIRED_LEDGER_KEYS = ("n_starts", "n_converged", "n_failed", "basin")
_VALID_VERDICTS = ("stable", "knife_edge", "not_evaluable", "no_fit")


def _corpus_available() -> bool:
    return PROC.joinpath("fits.jsonl").exists()


def _clean_logistic():
    y = m_logistic(T, 0.0, 1.0, 0.4, 20.0) + _RNG.normal(0, 0.004, len(T))
    return list(T), list(y)


def _noiseless_step(tstep=20.0):
    """The verified knife-edge fixture. A noiseless sharp step is fitted to
    machine precision by many different large-k sigmoids, so the shape parameters
    are unidentifiable above a threshold and independent starts settle on
    materially different vectors at indistinguishable SSE. That is the textbook
    knife-edge, and it is exactly why real curve CPAD-TK-1102 bifurcates.

    VERIFIED, not assumed: with `richards` this produces 4 distinct basins at an
    inter-basin parameter distance of 0.26 — over two orders of magnitude clear
    of BASIN_PARAM_REL_TOL, so the fixture does not depend on the tolerance
    happening to sit at its current value. A symmetric two-step staircase was
    tried first and REJECTED: all four starts converge to one broad optimum
    spanning both steps, so it does not bifurcate at all."""
    tt = np.arange(0.0, 40.0, 0.5)
    return list(tt), list((tt >= tstep).astype(float))


# ======================================================================== #
# POSITIVE — the record exists and is complete
# ======================================================================== #
def test_every_registry_model_emits_a_resolvable_provenance_record():
    """Every fit, converged or not, carries provenance, and its recipe id
    RESOLVES. A referenced record is only honest if the reference resolves —
    that is the one failure mode the recipe-ref design can have."""
    x, y = _clean_logistic()
    for name in ("lnt", "logistic", "gompertz", "richards", "exponential",
                 "scaling_law", "finke_watzky"):
        r = fit_one(x, y, name)
        prov = r.get("provenance")
        assert prov is not None, f"{name} emitted no provenance"
        assert prov["recipe"] in fp.RECIPES, f"{name}: dangling recipe ref"
        for k in _REQUIRED_LEDGER_KEYS:
            assert k in prov, f"{name} missing {k}"
        assert prov["basin"] in _VALID_VERDICTS, prov["basin"]
        # every verdict must explain itself, and the code must RESOLVE
        assert prov.get("reason_code") in fp.BASIN_REASON_CODES, (
            f"{name}: dangling reason_code {prov.get('reason_code')!r}")


def test_recipe_table_carries_the_full_2_3_field_set():
    """§2.3 names optimizer, version, tolerance and env_manifest_ref. The recipe
    half of the record must actually carry them, not merely have a slot."""
    table = fp.recipe_table()
    assert table["recipes"], "empty recipe table"
    for rid, rec in table["recipes"].items():
        for key in ("optimizer", "tolerance", "version", "env_manifest_ref",
                    "start_policy", "rng"):
            assert rec.get(key), f"{rid} missing {key}"
        assert isinstance(rec["rng"]["applicable"], bool)
        if not rec["rng"]["applicable"]:
            assert rec["rng"].get("reason"), f"{rid}: N/A rng without a reason"


def test_env_manifest_ref_agrees_with_the_m7_build_id_component():
    """`fit_provenance.env_manifest_ref` deliberately duplicates
    `m7_assemble._default_env_manifest` (importing it back would close a cycle:
    m7_assemble imports m2_fit). This asserts the duplicate has not drifted —
    a stronger guarantee than the import would have been."""
    import m7_assemble
    assert fp.env_manifest_ref() == m7_assemble._default_env_manifest()


def test_dose_response_fits_also_carry_provenance():
    """§10.12 says EVERY parameter. The dose axis is a separate entry point and
    is easy to forget — all five of its models are single-start."""
    doses = [0.5, 1, 2, 3, 5, 8, 12, 20, 35, 60]
    resp = list(m_dr_hill(np.array(doses), 0.0, 1.0, 5.0, 2.0))
    res = fit_dose_response(doses, resp)
    assert res["fits"], "no dose fits produced"
    for name, r in res["fits"].items():
        assert r.get("provenance"), f"dose model {name} emitted no provenance"
        assert r["provenance"]["recipe"] in fp.RECIPES


# ======================================================================== #
# NEGATIVE CONTROLS — the record is HONEST
# ======================================================================== #
def test_knife_edge_is_actually_detected():
    """NEGATIVE CONTROL: a detector that never fires passes every other test in
    this file while detecting nothing, and would stamp `stable` on precisely the
    fits §7 says must be refused. This proves it fires on a case that genuinely
    bifurcates (see `_noiseless_step` for why this fixture and not another)."""
    x, y = _noiseless_step()
    prov = fit_one(x, y, "richards")["provenance"]
    assert prov["basin"] == "knife_edge", prov
    assert prov["n_basins"] >= 2, prov
    spread = prov["basin_spread"]
    assert spread["sse_rel_gap"] <= fp.BASIN_SSE_REL_TOL, spread
    assert spread["param_rel_dist"] > fp.BASIN_PARAM_REL_TOL, spread
    # robust to the tolerance choice: this fixture is >100x clear of the cut,
    # so it does not depend on the constant sitting at its current value
    assert spread["param_rel_dist"] > 100 * fp.BASIN_PARAM_REL_TOL, spread
    assert not fp.is_basin_stable(prov), "§7 must refuse to freeze a knife-edge"


def test_clean_fit_is_stable_not_knife_edge():
    """NEGATIVE CONTROL in the other direction: a detector that fires everywhere
    is as useless as one that never fires. A clean, well-identified logistic
    must come back `stable` and be freezable under §7."""
    x, y = _clean_logistic()
    prov = fit_one(x, y, "logistic")["provenance"]
    assert prov["basin"] == "stable", prov
    assert prov["n_converged"] >= 2, prov
    assert fp.is_basin_stable(prov)


def test_every_registry_model_now_gets_a_genuine_multi_start():
    """The defect this replaces: `m2_fit` built its extra starts by testing
    PARAMETER NAMES (`if "t0" in pnames` / `if "k" in pnames`), so 9 of the 13
    registry models — including the mechanistic `finke_watzky`, whose rates are
    named k1/k2, and every dose-response model — constructed exactly ONE start
    and had nothing to be stable BETWEEN. Their fits could not be distinguished
    from local optima, and the §7 gate was `not_evaluable` for them: the engine
    published fitted parameters whose basin status was unknown.

    `estimation-policy-1.0` tops every model up to n_starts deterministic
    in-bounds probes, so the gate is now answerable across the registry."""
    x, y = _clean_logistic()
    for name in ("finke_watzky", "lnt", "scaling_law"):
        prov = fit_one(x, y, name)["provenance"]
        assert prov["n_starts"] > 1, (name, prov)
        assert prov["basin"] != "not_evaluable", (name, prov)
        assert prov["reason_code"] != "single_start_constructed", (name, prov)


def test_a_single_start_fit_is_still_never_reported_stable():
    """THE NEGATIVE CONTROL, preserved — it is the part that must not rot.

    Multi-start makes the gate ANSWERABLE; it does not make every answer `yes`.
    A fit that still ends up with one usable start — because the other probes
    failed to converge, which happens — must say `not_evaluable` and say why.
    The tempting bug is to let it default to `stable`, putting a §7
    freeze-approval on a fit whose stability was never tested. §7 is
    conservative: being unable to check is not permission.

    Exercised directly on the ledger so it holds regardless of how many starts
    the policy currently constructs."""
    one = fp.basin_ledger([{"index": 0, "converged": True, "sse": 1.0,
                            "theta": [1.0], "failure": None}])
    assert one["basin"] == "not_evaluable", one
    assert one["reason_code"] in ("single_start_converged",
                                 "single_start_constructed"), one
    assert not fp.is_basin_stable(one), (
        "an unevaluable fit must not pass the §7 gate")

    # and the corpus proves it is not merely theoretical: the rebuilt fits still
    # contain such cases, they are just no longer the default state of 9 models
    import json
    from pathlib import Path
    fits = Path(__file__).resolve().parent.parent / "data/processed/fits.jsonl"
    if not fits.exists():
        return
    seen = {}
    for line in fits.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        for model, f in (json.loads(line).get("fits") or {}).items():
            b = (f.get("provenance") or {}).get("basin")
            if b:
                seen[b] = seen.get(b, 0) + 1
    assert seen.get("stable", 0) > 0, seen
    # not_evaluable must now be RARE rather than structural
    total = sum(seen.values())
    assert seen.get("not_evaluable", 0) / total < 0.01, seen


def test_m2_never_fabricates_an_rng_seed():
    """NEGATIVE CONTROL, and the most important test in this file.

    M2 uses NO random number generator: its starts are built deterministically
    from the data and TRF is a deterministic descent. The obvious wrong way to
    "satisfy §10.12" is to write `rng_seed: 0` into the record and move on. That
    would be strictly WORSE than the honest absence it replaced — a fabricated
    provenance record is a lie that PASSES, and it would state that a stream was
    opened at seed 0 when none was opened at all. Reproducibility tooling
    downstream would then trust a seed that controls nothing.

    So: the M2 recipe must declare the RNG inapplicable WITH a reason, and must
    carry no seed under any value. This test fails the moment someone invents
    one."""
    for rid in ("m2/trf-bounded/det-starts-1.0",
                "mechanistic/trf-bounded-lsoda/single-start-1.0",
                "global_fit/trf-bounded-lsoda/det-starts-1.0"):
        rng = fp.recipe(rid)["rng"]
        assert rng["applicable"] is False, rid
        assert rng.get("reason"), rid
        assert "rng_seed" not in rng, (
            f"{rid} fabricated an rng_seed for a fit that draws no random number")
        assert "rng_algorithm" not in rng, rid
    # and the constructor refuses to build a reasonless N/A record
    try:
        fp.rng_not_applicable("")
        raise AssertionError("rng_not_applicable accepted an empty reason")
    except ValueError:
        pass


def test_untracked_bootstrap_count_is_never_backfilled():
    """NEGATIVE CONTROL, same principle as the rng_seed one applied to a second
    field. M4's gamma bootstraps reject divergent draws without returning the
    surviving count, so `accepted_B` is genuinely unknown there. The tempting
    shortcut is to write `accepted_B = requested_B`, which asserts that every
    draw survived — false, and invisible once written. An untracked count must
    surface as untracked."""
    bp = fp.bootstrap_provenance("m4/gamma-bootstrap-1.0", 0, 150, "m4-defs-1.0")
    assert bp["accepted_B_tracked"] is False, bp
    assert "accepted_B" not in bp, (
        "an unmeasured accepted_B was back-filled with a number")
    # and where it IS known it is reported as a real value
    bp2 = fp.bootstrap_provenance("m3/wild-bootstrap-mammen-1.0", 0, 150,
                                  "m3-select-1.0", accepted_B=147)
    assert bp2["accepted_B"] == 147 and "accepted_B_tracked" not in bp2, bp2
    # a recipe that declares no RNG must not accept a bootstrap record at all
    try:
        fp.bootstrap_provenance("m2/trf-bounded/det-starts-1.0", 0, 10, "v")
        raise AssertionError("built a bootstrap record for a non-RNG recipe")
    except ValueError:
        pass


# ======================================================================== #
# The live corpus
# ======================================================================== #
def test_live_fits_artifact_carries_provenance_on_every_model_fit():
    """§10.12 says EVERY parameter — so this checks the shipped artifact, not
    just freshly computed fits."""
    if not _corpus_available():
        return
    n_fits = n_missing = 0
    verdicts = set()
    with PROC.joinpath("fits.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            for name, r in (rec.get("fits") or {}).items():
                n_fits += 1
                prov = r.get("provenance")
                if not prov or prov.get("recipe") not in fp.RECIPES:
                    n_missing += 1
                    continue
                verdicts.add(prov.get("basin"))
    assert n_fits > 0, "no model-fits found in fits.jsonl"
    assert n_missing == 0, f"{n_missing}/{n_fits} model-fits lack provenance"
    assert verdicts <= set(_VALID_VERDICTS), verdicts


def test_recipe_table_artifact_matches_the_module():
    """The emitted recipe table must be the one the code would emit now —
    otherwise every `recipe` reference in fits.jsonl resolves to stale text."""
    p = PROC / "fit_recipes.json"
    if not p.exists():
        return
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["recipes"] == fp.recipe_table()["recipes"]


def test_bootstrap_modules_declare_a_real_rng_stream():
    """The mirror of `test_m2_never_fabricates_an_rng_seed`: where an RNG IS
    used, the record must be concrete. M3's wild bootstrap and M4's gamma
    bootstrap both draw, so both must carry a numeric seed, a named algorithm
    and a stream policy — the invariant is enforced in BOTH directions."""
    from m3_select import analyze_curve
    x, y = _clean_logistic()
    series = {"series_id": "prov-test", "x_hours": x,
              "m1": {"recommended_handling": "monotonic_fit", "y_processed": y}}
    res = analyze_curve(series, B=25)
    if res.get("status") != "ok":
        return
    bp = res["selection"]["bootstrap_provenance"]
    assert isinstance(bp["rng_seed"], int), bp
    assert bp["requested_B"] == 25 and bp["accepted_B"] <= 25, bp
    # the seed is inline; the algorithm and stream policy resolve THROUGH the
    # recipe, so the reference must actually land on a declared live stream
    rec = fp.recipe(bp["recipe"])
    assert rec["rng"]["applicable"] is True, rec["rng"]
    assert rec["rng"]["rng_algorithm"], rec["rng"]
    assert rec["rng"]["stream_policy"], rec["rng"]


# ======================================================================== #
# Unit-level: the ledger itself
# ======================================================================== #
def test_ledger_records_failed_starts_rather_than_swallowing_them():
    """`m2_fit.fit_one` `continue`s past a failed start, which is how the §7
    evidence was destroyed at fit time. A failure must now survive into the
    record with its index and cause."""
    outcomes = [
        {"index": 0, "converged": True, "sse": 1.0, "theta": [1.0, 2.0],
         "failure": None},
        {"index": 1, "converged": False, "sse": None, "theta": None,
         "failure": "RuntimeError"},
        {"index": 2, "converged": True, "sse": 1.0 + 1e-12, "theta": [1.0, 2.0],
         "failure": None},
    ]
    led = fp.basin_ledger(outcomes, lo=[0.0, 0.0], hi=[10.0, 10.0], sst=100.0)
    assert led["n_starts"] == 3 and led["n_converged"] == 2
    assert led["n_failed"] == 1
    assert led["failed_starts"] == [{"start": 1, "reason": "RuntimeError"}]
    assert led["basin"] == "stable"          # the two survivors agree
    assert led["win"] == 0                   # strict `<` keeps the EARLIER start


def test_ledger_reports_no_fit_when_nothing_converged():
    led = fp.basin_ledger(
        [{"index": 0, "converged": False, "sse": None, "theta": None,
          "failure": "RuntimeError"}], sst=1.0)
    assert led["basin"] == "no_fit" and led["win"] is None
    assert led["reason_code"] == "no_start_converged"
    assert not fp.is_basin_stable(led)


def test_near_perfect_fits_are_compared_on_the_data_scale():
    """Regression guard for a flaw found during detector verification: with a
    near-exact fit (SSE ~1e-22, as on real curve CPAD-TK-1102) a plain
    `gap / sse_win` ratio is taken against an arbitrary absolute epsilon and
    means nothing. The denominator is floored at a billionth of the data's total
    variance so the comparison stays dimensionally meaningful."""
    outcomes = [
        {"index": 0, "converged": True, "sse": 7e-22, "theta": [1.0, 2.0],
         "failure": None},
        {"index": 1, "converged": True, "sse": 5e-16, "theta": [1.0, 9.0],
         "failure": None},
    ]
    led = fp.basin_ledger(outcomes, lo=[0.0, 0.0], hi=[10.0, 10.0], sst=1.0e5)
    assert led["n_basins"] == 2, led
    # both fit to ~machine zero on a variance scale of 1e5 -> indistinguishable
    assert led["basin"] == "knife_edge", led
    assert led["basin_spread"]["sse_rel_gap"] < 1e-9, led


# ------------------------------- runner ------------------------------------ #
def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}"); failed += 1
        except Exception as e:                       # a crash is a failure too
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0



def test_env_manifest_fingerprints_the_stack_not_just_the_interpreter():
    """§7's environment axis used to be the string `python3.13` and nothing else.

    That is the same defect class this codebase has corrected repeatedly: a
    comparability key that fails to capture something which moves numbers. Two
    builds on the same Python minor but a different BLAS can diverge in the last
    digits of a bounded least-squares fit, and `comparable()` -- which tests
    build-id equality alone -- would have declared them comparable.

    Pinned here by REQUIRING the axis to move when the stack moves, rather than by
    asserting the current literal (which would break on every dependency bump and
    teach the next reader to delete the test)."""
    import m7_assemble
    tag = m7_assemble._default_env_manifest()
    fp = m7_assemble.environment_fingerprint()
    # every stack component that can move a float must be IN the tag
    for key in ("python", "numpy", "scipy", "blas", "blas_version", "machine"):
        val = fp.get(key)
        if val:
            assert str(val) in tag, (key, val, tag)
    # ...and it must be more than the interpreter minor
    assert tag != "python%d.%d" % (sys.version_info.major, sys.version_info.minor)
    assert len(tag) > 20, tag


def test_thread_env_is_deliberately_excluded_from_the_build_id():
    """Thread count genuinely can reorder BLAS reductions and change the last
    digits -- so leaving it out is a real, named limitation, not an oversight.

    It is excluded because folding it in would make `engine_build_id` depend on
    how the shell launched the process: the same corpus rebuilt with a different
    degree of parallelism would be declared non-comparable WITH ITSELF. This
    project's own sharded M3 rebuild sets OMP_NUM_THREADS=1, so that failure mode
    is not hypothetical. Pinned so the trade-off cannot be silently reversed."""
    import os
    import m7_assemble
    before = m7_assemble._default_env_manifest()
    old = os.environ.get("OMP_NUM_THREADS")
    try:
        os.environ["OMP_NUM_THREADS"] = "7"
        assert m7_assemble._default_env_manifest() == before
    finally:
        if old is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old
    # and the residual sensitivity must stay documented in the docstring
    assert "thread" in (m7_assemble._default_env_manifest.__doc__ or "").lower()


def test_the_shipped_artifacts_carry_the_widened_tag():
    """A widened tag that never reached the corpus would be a claim, not a fact."""
    import json
    if not _corpus_available():
        return
    import m7_assemble
    expect = m7_assemble._default_env_manifest()
    seen = set()
    with PROC.joinpath("m3_sample.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            prov = (json.loads(line).get("selection") or {}).get(
                "bootstrap_provenance") or {}
            if prov.get("env_manifest_ref"):
                seen.add(prov["env_manifest_ref"])
    assert seen == {expect}, seen

if __name__ == "__main__":
    raise SystemExit(_run())
