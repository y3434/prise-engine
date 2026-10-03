"""
PRISE — run every test suite and REPORT THE COUNTS FROM THE RUN.

    python engine/run_all_tests.py             # run everything, summarise
    python engine/run_all_tests.py --roster    # + emit the README roster lines

WHY THIS EXISTS. `engine/README.md` carries a per-file roster of test counts that
was maintained by hand, and it drifted twice: `test_m10` sat at 25 after a commit
raised it to 30, and `test_fit_provenance` sat at 14 after a control was added.
A self-reported number that quietly diverges from reality is the same class of
defect as the `[test]` tag §10.12 carried for months while having no test behind
it — the artifact asserts something nobody re-checked. So the counts are now
DERIVED from an actual run, and the README says which command regenerates them.

The suites are plain scripts (no pytest), each printing its own tally in one of a
few historical formats; `_parse_count` handles them and, importantly, reports
`None` rather than guessing when it cannot parse. An unparseable suite is a
failure of this runner, not a pass — silently scoring it 0 would reintroduce
exactly the drift this file exists to stop.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "engine"

# Suites that are runners for other files rather than test files themselves.
_NOT_A_SUITE = {"run_all_tests.py"}

# Suites that assert by EXIT CODE and print no tally. `web/test_web.py` boots a
# live server and drives it end to end; it has no `def test_*` functions, so "no
# count" is its CORRECT state, not a parse failure. Its exit code is the verdict.
# Listing it here keeps the honest distinction between "this suite has nothing to
# count" and "this runner could not read a count that exists" — collapsing the two
# is how a real parse failure gets waved through.
_EXIT_CODE_ONLY = {"test_web.py"}

_PATTERNS = (
    re.compile(r"^\s*(\d+)\s+passed,\s+(\d+)\s+failed", re.M),   # "N passed, M failed"
    # "17/17 passed". MUST precede the bare "N passed" form below. That pattern's
    # greedy \S* backtracks INTO the number — on "17/17 passed" it consumed "17/1"
    # and captured 7, under-reporting test_m14_meta by 10 and the total by the
    # same. Found by cross-checking the run against a static `def test_` count;
    # the two must agree, which is why both numbers are reported.
    re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s+passed\s*$", re.M),     # "17/17 passed"
    re.compile(r"all\s+(\d+)\s+\w+\s+tests?\s+passed", re.I),    # "all 11 M0 tests passed"
    re.compile(r"OK\s*[^\d]*?(\d+)\s+tests?\s+passed", re.I),    # "OK - 30 tests passed"
    # the colon is REQUIRED (was `:?`), so this can no longer match "17/17 passed"
    re.compile(r"^\s*\S+:\s*(\d+)\s+passed\s*$", re.M),          # "test_m8: 19 passed"
)


def _parse_count(out: str):
    """(passed, failed) from a suite's stdout, or (None, None) if unparseable."""
    m = _PATTERNS[0].search(out)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _PATTERNS[1].search(out)          # "P/T passed" -> failed = T - P
    if m:
        p, t = int(m.group(1)), int(m.group(2))
        return p, max(0, t - p)
    for pat in _PATTERNS[2:]:
        m = pat.search(out)
        if m:
            return int(m.group(1)), 0
    return None, None


def discover() -> list[Path]:
    files = sorted(p for p in ENGINE.glob("test_*.py")
                   if p.name not in _NOT_A_SUITE)
    web = ROOT / "web" / "test_web.py"
    if web.exists():
        files.append(web)
    return files


def run_one(path: Path) -> dict:
    proc = subprocess.run([sys.executable, str(path)], cwd=str(path.parent),
                          capture_output=True, text=True, errors="replace")
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    exit_only = path.name in _EXIT_CODE_ONLY
    passed, failed = (None, None) if exit_only else _parse_count(out)
    return {"file": path.name,
            "where": "web" if path.parent.name == "web" else "engine",
            "exit": proc.returncode, "passed": passed, "failed": failed,
            "exit_only": exit_only,
            "parsed": exit_only or passed is not None,
            "tail": out.strip().splitlines()[-1:] or [""]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="run every PRISE test suite")
    ap.add_argument("--roster", action="store_true",
                    help="emit README roster lines with the measured counts")
    args = ap.parse_args(argv)

    results = [run_one(p) for p in discover()]
    eng = [r for r in results if r["where"] == "engine"]
    web = [r for r in results if r["where"] == "web"]

    print(f"{'suite':38s} {'exit':>4} {'passed':>7} {'failed':>7}")
    for r in results:
        if r["exit_only"]:
            p = f = "n/a"
        else:
            p = r["passed"] if r["parsed"] else "?"
            f = r["failed"] if r["parsed"] else "?"
        print(f"{r['file']:38s} {r['exit']:>4} {str(p):>7} {str(f):>7}")

    unparsed = [r for r in results if not r["parsed"] and not r["exit_only"]]
    bad_exit = [r for r in results if r["exit"] != 0]
    # `r["failed"]` is None for exit-code-only suites and 0 for a clean run, both
    # falsy; a suite with 0 passed and N failed must still be caught, so this
    # keys on `failed` alone rather than gating on `passed` being truthy.
    failing = [r for r in results if r["failed"]]

    eng_total = sum(r["passed"] for r in eng if r["passed"] is not None)
    exit_only = [r for r in results if r["exit_only"]]
    print(f"\nengine suites : {len(eng)} files, {eng_total} test functions")
    for r in exit_only:
        print(f"exit-code only: {r['file']} -> "
              f"{'PASS' if r['exit'] == 0 else 'FAIL'} (asserts by exit code; "
              f"no test tally to count)")
    print(f"TOTAL         : {eng_total} counted test functions "
          f"+ {len(exit_only)} exit-code suite(s)")

    # Cross-check the RUN against a static `def test_` count. These must agree:
    # a run-count that silently drifts from the source is the exact defect this
    # file exists to stop, and it is how the "17/17 -> 7" regex bug was caught.
    static = 0
    for p in discover():
        if p.name in _EXIT_CODE_ONLY:
            continue
        static += sum(1 for ln in p.read_text(encoding="utf-8", errors="replace")
                      .splitlines() if ln.startswith("def test_"))
    flag = "OK" if static == eng_total else "MISMATCH"
    print(f"static `def test_` count: {static}  vs run: {eng_total}  -> {flag}")
    if static != eng_total:
        print("  a run/source disagreement means one of them is lying; "
              "fix before trusting either number")
    if unparsed:
        print("\nUNPARSEABLE (counted as a runner failure, never as zero):")
        for r in unparsed:
            print(f"  {r['file']}  exit={r['exit']}  last line: {r['tail'][0][:90]}")
    if failing:
        print("\nFAILING:")
        for r in failing:
            print(f"  {r['file']}: {r['failed']} failed")
    if bad_exit:
        print("\nNON-ZERO EXIT:")
        for r in bad_exit:
            print(f"  {r['file']}: exit {r['exit']}")

    if args.roster:
        print("\n--- README roster (measured; paste over the block) ---")
        print(f"All {len(eng)} engine test files, with the counts each one "
              f"printed on the run that produced this README:")
        print("```")
        for r in eng:
            print(f"python engine/{r['file']:<32} # ({r['passed']})")
        print("```")

    ok = (not unparsed and not failing and not bad_exit
          and static == eng_total)
    print("\nRESULT:", "ALL GREEN" if ok else "PROBLEMS ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
