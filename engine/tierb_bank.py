"""Precompute the per-curve Tier-B (Knowles/Cohen) mechanistic fit bank.

WHY THIS EXISTS
---------------
`GET /api/models/{series_id}` reads its descriptive/biphasic/autocatalytic rows
straight out of fits.jsonl (see 9cd3e46), but the four Tier-B mechanistic
variants had no artifact to read, so the endpoint fitted four stiff ODE models
LIVE on every request. Measured over a random sample of 14 curves:

    median 0.29 s   mean 8.52 s   max 114.17 s  (CPAD-TK-2185, only 22 points)

The median is fine; the tail is not. The serving container runs on 0.1 CPU, so
a tail curve simply never finishes in a browser. server.py memoised the result
per series, which is why a protein felt fast on a second visit and slow on a
first -- and why it went slow again whenever the bounded cache cleared.

These fits are a deterministic function of (x, y) over frozen curves, so they
belong in an artifact like every other published fit.

OUTPUT
------
data/processed/mechanistic_fits.jsonl -- one record per series:

    {"series_id": ..., "signal_used": ..., "n_points": ...,
     "seconds": <wall time to fit all variants>,
     "fits": {variant: <fit_mechanistic() dict>, ...}}

The RAW fit_mechanistic() dicts are stored, unshaped: web/server.py keeps its
own presentation logic, so the artifact stays a faithful record of what the
engine computed rather than a view of it.

USAGE
-----
    python engine/tierb_bank.py                 # build (resumes by default)
    python engine/tierb_bank.py --no-resume     # rebuild from scratch
    python engine/tierb_bank.py --limit 50      # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mechanistic as _mech  # noqa: E402
import m2_fit as _m2  # noqa: E402


def _load_curves(path):
    """series_id -> curve record, from curves_triaged.jsonl."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            sid = r.get("series_id")
            if sid:
                out[sid] = r
    return out


def _xy(cur):
    """The exact (x, y) server.py fits: m1.y_processed when present, else raw."""
    m1 = cur.get("m1") or {}
    x = [float(v) for v in (cur.get("x_hours") or [])]
    y_src = m1.get("y_processed") or cur.get("y_intensity") or []
    y = [float(v) for v in y_src]
    signal = "m1.y_processed" if m1.get("y_processed") else "y_intensity"
    if len(x) != len(y) or len(x) < 3:
        return None, None, signal
    return x, y, signal


def _already_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                sid = json.loads(line).get("series_id")
            except Exception:
                continue
            if sid:
                done.add(sid)
    return done


# The registry models /api/models compares. Kept in step with
# web/server.py:_COMPARISON_REGISTRY_MODELS.
COMPARISON_REGISTRY_MODELS = [
    "lnt", "logistic", "gompertz", "richards", "exponential",
    "scaling_law", "finke_watzky", "brain_cousens",
]


def _load_published_fits(path):
    """series_id -> {model: fit} already published by M2."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            sid = r.get("series_id")
            if sid:
                out[sid] = r.get("fits") or {}
    return out


def build_registry_gapfill(args):
    """Publish the registry fits /api/models needs but M2 did not select.

    M2 picks a CANDIDATE SET per curve from its shape: a `monotonic_fit` curve
    gets the seven growth models and no brain_cousens; a `nonmonotonic_fit`
    curve gets only lnt + brain_cousens. That is a deliberate engine decision,
    not a gap.

    /api/models deliberately shows the FULL bank anyway, so the reader can see
    the excluded models ranked alongside the selected ones. That is a real
    web-layer choice, so the answer is to publish those extra fits -- not to
    drop them and quietly narrow what the page compares. Every curve needs at
    least one (brain_cousens on 1331 of them; the sigmoidal set on 377).
    """
    root = ROOT / "data" / "processed"
    curves = _load_curves(args.input)
    published = _load_published_fits(root / "fits.jsonl")
    out_path = root / "registry_gapfill_fits.jsonl"

    n = n_fits = 0
    t0 = time.time()
    print(f"[Registry] gap-filling over {len(curves)} curves ...", flush=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for i, (sid, cur) in enumerate(curves.items(), 1):
            have = published.get(sid) or {}
            miss = [m for m in COMPARISON_REGISTRY_MODELS if not have.get(m)]
            if not miss:
                continue
            x, y, signal = _xy(cur)
            if x is None:
                continue
            fits = {}
            for m in miss:
                try:
                    fits[m] = _m2.fit_one(x, y, m)
                except Exception as exc:
                    fits[m] = {"converged": False,
                               "reason": f"{type(exc).__name__}: {exc}"}
            out.write(json.dumps({
                "series_id": sid, "signal_used": signal, "fits": fits,
            }) + "\n")
            n += 1
            n_fits += len(fits)
            if i % 400 == 0:
                print(f"[Registry] {i}/{len(curves)} ...", flush=True)
    print(f"[Registry] wrote {out_path}  ({n} series, {n_fits} fits, "
          f"{(time.time()-t0)/60:.1f} min)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="PRISE Tier-B mechanistic fit bank")
    ap.add_argument("--input", type=Path,
                    default=ROOT / "data" / "processed" / "curves_triaged.jsonl")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "data" / "processed" / "mechanistic_fits.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--registry", action="store_true",
                    help="build the registry gap-fill bank instead of the Tier-B bank")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="rebuild from scratch instead of skipping finished series")
    args = ap.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input} (run M1 first)")

    if args.registry:
        build_registry_gapfill(args)
        return 0

    curves = _load_curves(args.input)
    done = _already_done(args.output) if args.resume else set()
    if not args.resume and os.path.exists(args.output):
        os.remove(args.output)

    todo = [s for s in curves if s not in done]
    if args.limit:
        todo = todo[:args.limit]

    variants = list(_mech.VARIANT_RATES)
    print(f"[TierB] {len(curves)} curves, {len(done)} already done, "
          f"{len(todo)} to fit x {len(variants)} variants", flush=True)

    t_start = time.time()
    n = skipped = 0
    slowest = []
    with open(args.output, "a", encoding="utf-8") as out:
        for i, sid in enumerate(todo, 1):
            x, y, signal = _xy(curves[sid])
            if x is None:
                skipped += 1
                continue
            t0 = time.time()
            fits = {}
            for v in variants:
                try:
                    fits[v] = _mech.fit_mechanistic(x, y, variant=v)
                except Exception as exc:
                    fits[v] = {"variant": v, "converged": False,
                               "reason": f"{type(exc).__name__}: {exc}"}
            dt = time.time() - t0
            out.write(json.dumps({
                "series_id": sid,
                "signal_used": signal,
                "n_points": len(x),
                "seconds": round(dt, 3),
                "fits": fits,
            }) + "\n")
            out.flush()
            n += 1
            slowest.append((dt, sid))
            if dt > 20 or i % 100 == 0:
                el = time.time() - t_start
                rate = el / max(i, 1)
                print(f"[TierB] {i}/{len(todo)}  {sid}  {dt:.1f}s   "
                      f"elapsed {el/60:.1f}m  eta {(len(todo)-i)*rate/60:.1f}m",
                      flush=True)

    el = time.time() - t_start
    slowest.sort(reverse=True)
    print(f"[TierB] wrote {args.output}  ({n} series, {skipped} unfittable, "
          f"{el/60:.1f} min)", flush=True)
    print("[TierB] slowest: " + ", ".join(f"{s} {d:.1f}s" for d, s in slowest[:5]),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
