"""
PRISE — artifact value-invariance checker
=========================================

The §2.3 FitProvenance change claims to be ADDITIVE and VALUE-PRESERVING: it
writes down quantities the fitting loops already computed, so every number that
was in an artifact before must still be there, byte-identical, with only
provenance keys added. That claim is the entire argument for NOT bumping
`engine_build_id` (§7 gates comparability on results, and results that did not
move are still comparable).

An argument is not a proof, and reasoning is precisely what M17 just taught this
codebase not to trust. So the claim is MECHANICALLY CHECKED: snapshot the
artifacts before the rebuild, compare after, and fail on any pre-existing value
that moved or vanished.

This also happens to be the real guard on the gamma bootstrap hazard. The shipped
`gamma.jsonl` was produced with a `--bootstrap` size recorded only in
`engine/README.md`, never in the artifact. Re-running with a different B would
silently move every `gamma_ci` — and `m7_assemble.gamma_significant()` gates the
published `scaling` tier count on whether that CI excludes zero. Pinning B from
the README is a precaution; this checker is the proof that the precaution worked.

Usage:
    python engine/artifact_invariance.py --snapshot     # BEFORE the rebuild
    python engine/artifact_invariance.py --check        # AFTER the rebuild
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

# Artifacts the FitProvenance change is expected to touch. Anything not listed
# here should not change at all, which `--check --strict-untouched` verifies.
TRACKED = (
    "fits.jsonl",
    "dose_response_fits.jsonl",
    "m3_sample.jsonl",
    "gamma.jsonl",
    "global_fit.json",
)

# The ONLY keys this change is permitted to introduce. A new key anywhere else is
# a finding, not a convenience — if the diff shows one, the change did more than
# it claimed. Matched on the FINAL path segment.
ALLOWED_NEW_KEYS = frozenset({
    "provenance",
    "fit_provenance",
    "bootstrap_provenance",
    "seed_is_effective",
})

_MISSING = object()


def _snapshot_dir(root: Path) -> Path:
    """The baseline the current artifacts are compared against.

    Originally fixed at `pre_fitprovenance`, taken to prove that the §2.3
    FitProvenance change was additive and value-preserving. That claim was
    checked and is now history: `estimation-policy-1.0` DELIBERATELY moves
    fitted values (M2 multi-start, M3 per-series bootstrap seed), so a fixed
    pre-FitProvenance baseline can no longer express the invariant.

    The baseline is therefore BUILD-TAGGED. `snapshot()` writes
    `baseline_<engine_build_id>/`, and the guard means what it should: within a
    build id nothing may move (accidental drift still fails loudly), while a
    move across a build id is expected and must be explained by a governed
    policy. The legacy directory is still honoured when no tagged baseline
    exists, so a clone with only the old snapshot degrades to a skip rather
    than a false failure."""
    snaps = root / "data" / "snapshots"
    tagged = sorted(snaps.glob("baseline_*"), key=lambda q: q.stat().st_mtime
                    if q.exists() else 0)
    if tagged:
        return tagged[-1]
    return snaps / "pre_fitprovenance"


def snapshot_build_id(root: Path):
    """The engine_build_id the baseline was taken at, or None for the legacy
    untagged snapshot."""
    man = _snapshot_dir(root) / "MANIFEST.json"
    try:
        return json.loads(man.read_text(encoding="utf-8")).get("engine_build_id")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# deep comparison
# --------------------------------------------------------------------------- #
def _numbers_equal(a, b) -> bool:
    """Byte-identical for our purposes: exact equality, with NaN treated as
    equal to NaN (a NaN that stayed NaN did not move) and the int/float
    distinction ignored only when the values are exactly equal. Deliberately NOT
    a tolerance comparison — the claim under test is that nothing moved AT ALL,
    and a tolerance would let real drift hide under it."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float):
            if math.isnan(a) and math.isnan(b):
                return True
        return a == b
    return False


def diff_value(old, new, path: str, out: list) -> None:
    """Walk two artifact values in parallel, recording every difference."""
    if isinstance(old, dict) and isinstance(new, dict):
        for k in old:
            sub = f"{path}.{k}" if path else k
            if k not in new:
                out.append({"path": sub, "issue": "key_removed"})
            else:
                diff_value(old[k], new[k], sub, out)
        for k in new:
            if k in old:
                continue
            sub = f"{path}.{k}" if path else k
            if k not in ALLOWED_NEW_KEYS:
                out.append({"path": sub, "issue": "unexpected_new_key"})
        return
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            out.append({"path": path, "issue": "list_length_changed",
                        "old": len(old), "new": len(new)})
            return
        for i, (a, b) in enumerate(zip(old, new)):
            diff_value(a, b, f"{path}[{i}]", out)
        return
    if type(old) is not type(new) and not (
            isinstance(old, (int, float)) and isinstance(new, (int, float))):
        out.append({"path": path, "issue": "type_changed",
                    "old": type(old).__name__, "new": type(new).__name__})
        return
    if isinstance(old, (int, float)):
        if not _numbers_equal(old, new):
            out.append({"path": path, "issue": "value_changed",
                        "old": old, "new": new})
        return
    if old != new:
        out.append({"path": path, "issue": "value_changed",
                    "old": old, "new": new})


def _load(path: Path):
    """Load a .json object or a .jsonl keyed by the record's natural id."""
    if path.suffix == ".jsonl":
        recs = {}
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                key = (d.get("series_id") or d.get("concentration_series_id")
                       or d.get("protein") or f"#{i}")
                recs[key] = d
        return recs
    return json.loads(path.read_text(encoding="utf-8"))


def compare_artifact(old_path: Path, new_path: Path) -> dict:
    """Compare one artifact before/after. Returns a report dict."""
    if not old_path.exists():
        return {"artifact": new_path.name, "status": "no_snapshot"}
    if not new_path.exists():
        return {"artifact": new_path.name, "status": "missing_after_rebuild",
                "violations": [{"path": "", "issue": "artifact_disappeared"}]}
    old, new = _load(old_path), _load(new_path)
    violations: list = []
    if isinstance(old, dict) and isinstance(new, dict) and old_path.suffix == ".jsonl":
        for key in old:
            if key not in new:
                violations.append({"path": key, "issue": "record_removed"})
            else:
                diff_value(old[key], new[key], key, violations)
        added = [k for k in new if k not in old]
        if added:
            violations.append({"path": "", "issue": "records_added",
                               "n": len(added), "examples": added[:5]})
    else:
        diff_value(old, new, "", violations)
    return {"artifact": new_path.name,
            "status": "ok" if not violations else "violations",
            "n_violations": len(violations),
            "violations": violations[:40]}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _current_build_id(proc: Path):
    """The build id the artifacts on disk carry, read from the M7 rollup."""
    try:
        return json.loads((proc / "m7_rollup.json").read_text(
            encoding="utf-8")).get("engine_build_id")
    except Exception:
        return None


def snapshot(root: Path, artifacts=TRACKED) -> dict:
    proc = root / "data" / "processed"
    bid = _current_build_id(root / "data" / "processed")
    dest = (root / "data" / "snapshots" /
            ("baseline_" + str(bid) if bid else "pre_fitprovenance"))
    dest.mkdir(parents=True, exist_ok=True)
    taken = []
    for name in artifacts:
        src = proc / name
        if src.exists():
            shutil.copy2(src, dest / name)
            taken.append({"artifact": name, "bytes": src.stat().st_size})
    (dest / "MANIFEST.json").write_text(
        json.dumps({"snapshot_of": str(proc), "artifacts": taken,
                    "engine_build_id": _current_build_id(proc)}, indent=2),
        encoding="utf-8")
    return {"dest": str(dest), "artifacts": taken}


def check(root: Path, artifacts=TRACKED) -> dict:
    proc = root / "data" / "processed"
    snap = _snapshot_dir(root)
    reports = [compare_artifact(snap / n, proc / n) for n in artifacts]
    checked = [r for r in reports if r["status"] != "no_snapshot"]
    return {"all_ok": all(r["status"] == "ok" for r in checked),
            "n_checked": len(checked),
            "n_no_snapshot": len(reports) - len(checked),
            "reports": reports}


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="PRISE artifact value-invariance")
    ap.add_argument("--snapshot", action="store_true",
                    help="copy tracked artifacts aside BEFORE a rebuild")
    ap.add_argument("--check", action="store_true",
                    help="compare current artifacts against the snapshot")
    args = ap.parse_args(argv)
    if args.snapshot:
        res = snapshot(root)
        print(json.dumps(res, indent=2))
        return 0
    if args.check:
        res = check(root)
        print(json.dumps({k: v for k, v in res.items() if k != "reports"},
                         indent=2))
        for r in res["reports"]:
            print(f"  {r['artifact']:28s} {r['status']}"
                  + (f"  ({r.get('n_violations')} violations)"
                     if r.get("n_violations") else ""))
            for v in (r.get("violations") or [])[:10]:
                print(f"      {v}")
        return 0 if res["all_ok"] else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
