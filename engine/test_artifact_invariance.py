"""
Tests for the artifact value-invariance guarantee.

    python engine/test_artifact_invariance.py     # standalone PASS/FAIL
    pytest engine/test_artifact_invariance.py

WHAT IS BEING PROVED, AND WHY IT IS A TEST RATHER THAN A ONE-OFF SCRIPT

The §2.3 FitProvenance change claims to be ADDITIVE and VALUE-PRESERVING — it
writes down quantities the fitting loops already computed, adds no optimizer
call, no RNG and no reordering. That claim is the whole argument for NOT bumping
`engine_build_id`: §7 gates comparability on RESULTS, and results that did not
move are still comparable. A version axis that moves for a schema change nobody's
numbers depend on would assert a non-comparability that does not exist and force
a rebuild of every descendant for nothing.

But "I reasoned carefully about the control flow" is exactly the kind of claim
M17 just taught this codebase not to accept. So it is checked mechanically
against a snapshot taken before the rebuild, and the check ships as a test so it
cannot rot into a claim nobody re-runs.

It is also the real guard on the gamma-bootstrap hazard: `gamma.jsonl` was built
with a `--bootstrap` size recorded only in `engine/README.md`, never in the
artifact, and `m7_assemble.gamma_significant()` gates the published `scaling`
tier count on whether `gamma_ci` excludes zero. Pinning B from the README was a
precaution; this is the proof the precaution held.

If the snapshot is absent (a fresh clone, or after the snapshot is cleaned up)
the corpus-dependent tests skip rather than fail — the same posture the rest of
the suite takes toward `data/processed`.
"""
from __future__ import annotations

import json
from pathlib import Path

import artifact_invariance as ai

ROOT = Path(__file__).resolve().parent.parent
SNAP = ai._snapshot_dir(ROOT)


def _snapshot_available() -> bool:
    return SNAP.exists() and any(SNAP.glob("*.json*"))


def _baseline_is_this_build() -> bool:
    """Byte-identity is only the right assertion WITHIN one engine_build_id.

    The original baseline (`pre_fitprovenance`) was taken to prove the §2.3
    FitProvenance change was value-preserving, and it did. `estimation-policy-1.0`
    then moved fitted values ON PURPOSE -- M2 now gives every registry model
    genuine multi-starts and M3 seeds each curve's bootstrap separately -- so
    comparing today's artifacts against a pre-FitProvenance baseline would assert
    an invariant this project has deliberately, and governedly, broken.

    So: same build id => nothing may move (accidental drift still fails loudly);
    different build id => the move is expected, and `test_a_moved_value_is_always
    _explained_by_a_governed_policy` takes over."""
    bid = ai.snapshot_build_id(ROOT)
    cur = ai._current_build_id(ROOT / "data" / "processed")
    return bool(bid) and bid == cur


# ======================================================================== #
# The guarantee
# ======================================================================== #
def test_no_pre_existing_value_moved_in_any_rebuilt_artifact():
    """THE LOAD-BEARING TEST. Every key that existed before the FitProvenance
    change must still exist with a byte-identical value; the only permitted
    difference is an added provenance key."""
    if not (_snapshot_available() and _baseline_is_this_build()):
        return
    report = ai.check(ROOT)
    assert report["n_checked"] > 0, "snapshot present but nothing compared"
    bad = [r for r in report["reports"] if r["status"] == "violations"]
    assert not bad, json.dumps(
        [{"artifact": r["artifact"], "violations": r["violations"][:6]}
         for r in bad], indent=2)[:4000]


def test_gamma_ci_specifically_did_not_move():
    """Called out separately because it is the one value whose drift would be
    invisible AND consequential: gamma_ci feeds gamma_significant(), which gates
    the published `scaling` tier. A generic pass could in principle mask a
    gamma-shaped failure, so this asserts it by name."""
    if not (_snapshot_available() and _baseline_is_this_build()):
        return
    old_p, new_p = SNAP / "gamma.jsonl", ROOT / "data/processed/gamma.jsonl"
    if not (old_p.exists() and new_p.exists()):
        return
    old, new = ai._load(old_p), ai._load(new_p)
    checked = 0
    for csid, rec in old.items():
        assert csid in new, f"series {csid} disappeared from gamma.jsonl"
        for block in ("gamma_regression", "gamma_global"):
            o = (rec.get(block) or {}).get("gamma_ci")
            n = (new[csid].get(block) or {}).get("gamma_ci")
            assert o == n, f"{csid}.{block}.gamma_ci moved: {o} -> {n}"
            og = (rec.get(block) or {}).get("gamma")
            ng = (new[csid].get(block) or {}).get("gamma")
            assert og == ng, f"{csid}.{block}.gamma moved: {og} -> {ng}"
            checked += 1
    assert checked > 0, "no gamma blocks compared"


def test_fitted_parameters_are_byte_identical():
    """The narrowest, sharpest statement of the claim: the actual fitted numbers
    in fits.jsonl are unchanged."""
    if not (_snapshot_available() and _baseline_is_this_build()):
        return
    old_p, new_p = SNAP / "fits.jsonl", ROOT / "data/processed/fits.jsonl"
    if not (old_p.exists() and new_p.exists()):
        return
    old, new = ai._load(old_p), ai._load(new_p)
    n_params = 0
    for sid, rec in old.items():
        assert sid in new, f"series {sid} disappeared from fits.jsonl"
        for model, fit in (rec.get("fits") or {}).items():
            nf = (new[sid].get("fits") or {}).get(model)
            assert nf is not None, f"{sid}/{model} disappeared"
            assert fit.get("params") == nf.get("params"), f"{sid}/{model} params moved"
            assert fit.get("param_se") == nf.get("param_se"), f"{sid}/{model} SE moved"
            for k in ("sse", "r2", "aic", "aicc", "bic", "loglik", "converged"):
                assert fit.get(k) == nf.get(k), f"{sid}/{model}.{k} moved"
            n_params += 1
    assert n_params > 0, "no model-fits compared"




def test_a_moved_value_is_always_explained_by_a_governed_policy():
    """The other half of the guarantee, and the one that keeps the guard honest
    once the baseline and the corpus disagree.

    Values are allowed to move ONLY across an engine_build_id change, and an
    engine_build_id only moves when a governed version axis moves. So when the
    baseline predates the current build, this requires that at least one governed
    policy actually records a change -- `applied` differing from its retained
    `previous`. Without this, "the build id changed" would become an unfalsifiable
    excuse for any drift at all."""
    if not _snapshot_available():
        return
    if _baseline_is_this_build():
        return                      # nothing moved; the tests above proved it
    import bootstrap_policy as bp
    import join_policy as jp
    changed = []
    for mod in (bp.BOOTSTRAP_POLICY, jp.JOIN_POLICY):
        for name, rec in mod.items():
            if rec.get("applied") != rec.get("previous"):
                changed.append(name)
    assert changed, (
        "artifacts differ from the baseline but no governed policy records a "
        "change -- that is unexplained drift, not a governed change")
    # and every such record must retain the previous value, or the revert path
    # it claims to offer does not exist
    for mod in (bp.BOOTSTRAP_POLICY, jp.JOIN_POLICY):
        for name, rec in mod.items():
            assert "previous" in rec, name


# ======================================================================== #
# NEGATIVE CONTROL for the checker itself
# ======================================================================== #
def test_checker_detects_a_changed_value():
    """A checker that never reports a violation passes every test above while
    checking nothing. This feeds it a known-moved value and requires it to
    complain."""
    old = {"a": {"sse": 1.0, "params": {"k": 0.5}}}
    new = {"a": {"sse": 1.0, "params": {"k": 0.5000001}}}
    out: list = []
    ai.diff_value(old, new, "", out)
    assert any(v["issue"] == "value_changed" for v in out), out


def test_checker_detects_a_removed_key_and_an_unexpected_new_key():
    out: list = []
    ai.diff_value({"a": 1, "b": 2}, {"a": 1}, "", out)
    assert any(v["issue"] == "key_removed" for v in out), out

    out = []
    ai.diff_value({"a": 1}, {"a": 1, "surprise": 2}, "", out)
    assert any(v["issue"] == "unexpected_new_key" for v in out), out


def test_checker_permits_exactly_the_provenance_keys_and_no_others():
    """The allowlist is the point: additive means additive IN ONE PLACE. If this
    ever silently widens, the invariance guarantee stops meaning anything."""
    for allowed in ("provenance", "fit_provenance", "bootstrap_provenance"):
        out: list = []
        ai.diff_value({"a": 1}, {"a": 1, allowed: {"x": 1}}, "", out)
        assert not out, (allowed, out)
    assert "params" not in ai.ALLOWED_NEW_KEYS
    assert "sse" not in ai.ALLOWED_NEW_KEYS


def test_nan_is_not_reported_as_drift():
    """A NaN that stayed NaN did not move. Without this the checker would report
    every non-finite diagnostic as a violation on every run."""
    out: list = []
    ai.diff_value({"x": float("nan")}, {"x": float("nan")}, "", out)
    assert not out, out
    out = []
    ai.diff_value({"x": float("nan")}, {"x": 1.0}, "", out)
    assert any(v["issue"] == "value_changed" for v in out), out


# ------------------------------- runner ------------------------------------ #
def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}"); failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
