# PRISE

**Protein Aggregation Regime Inference & Scoring Engine**

An honesty-first analysis engine over protein-aggregation kinetics. It reads the
CPAD 2.0 corpus, fits and ranks a bank of kinetic models per curve, and — the
part that matters — reports explicitly what the data does **not** license.

**Live app: [prise-ay5v.onrender.com](https://prise-ay5v.onrender.com)**

> It is hosted on a free tier that sleeps after 15 minutes idle, so the first
> request may take ~50 seconds to wake. Everything after that is instant.

---

## The headline result is a negative one

PRISE grades every curve by how much inference it actually supports, then
reports the corpus-wide tally:

| Information-yield tier | Curves |
|---|---:|
| Mechanistic | **0** |
| Scaling | 59 |
| Descriptive | 1,135 |
| Signal only | 454 |
| Uninformative | 6 |

**Zero of 1,654 curves meet the bar for a mechanistic claim.** That is the
finding, not a bug and not a missing feature. The engine is built to surface it
rather than let a good-looking fit imply a mechanism it cannot support.

The same principle runs throughout:

- A high Tier-B (Knowles/Cohen) R² is **not** a licensed mechanism call. M5
  refuses mechanism on a single curve, and says so in the response.
- An identifiability test flags when a winning model is not actually
  distinguishable from its rivals, so "best fit" never silently becomes "right
  model".
- The two γ estimators (regression and global collapse) are reported
  **separately and never merged**; disagreement is surfaced as a finding, since
  it means curve shape is not concentration-invariant.
- A 1-D sequence proxy is always stamped `is_3d_derived=false`. A real 3D value
  appears only where a PDB was actually fetched and parsed. No PDB means no
  `real_3d` key — a proxy is never dressed up as a measurement.
- A missing artifact answers `available: false` with a note naming the module to
  run. It never produces a blank page, a 500, or an invented number.

## What's here

```
engine/    M0–M16 analysis modules — triage, fitting, selection, features,
           classification, propensity, assembly, reachability, recommendation,
           cross-modal, metadata quality, information content, BOED,
           meta-analysis, literature ingestion, structure bridge
web/       stdlib-only HTTP backend + vanilla-JS single-page frontend
etl/       CPAD 2.0 ingestion
docs/      implementation notes
reports/   generated product reports
```

`PRISE_DESIGN.md` is the full system design — the authoritative specification,
including the review history that reshaped it across seven revisions.

## Scale

| | |
|---|---:|
| Kinetic curves analysed | 1,654 |
| Proteins | 105 |
| Source corpus | ~83,000 CPAD 2.0 data points |
| Analysis modules | 17 (M0–M16) |
| Models in the comparison bank | 13 |
| Proteins with a concentration series | 34 |

## Running it

The web app is deliberately **dependency-light**: the backend is
`http.server.ThreadingHTTPServer` plus a small path router — no FastAPI, no
Flask, no uvicorn — and the frontend is vanilla JS with no build step. Only the
engine needs third-party packages (numpy, scipy).

```bash
pip install -r requirements.txt
python web/server.py     # http://127.0.0.1:8000
python web/test_web.py   # stdlib-only suite; exit 0 = pass
```

The test suite is **hermetic** — it reads frozen artifacts and talks only to its
own loopback server. Verified with all non-loopback sockets and DNS blocked: 0
outbound attempts.

### You will need the artifacts

**This repository contains code, not data.** `data/` is gitignored.

CPAD 2.0 is attribution-gated — bulk reuse requires contacting the authors — so
neither the raw corpus nor the derived artifacts are redistributed here. The
deployed app serves only derived results and single curves, one at a time, with
attribution on every response and deliberately no bulk-dump endpoint. Publishing
125 MB of derived artifacts to a public repo would route around that, so it
isn't done.

To run against real data you need CPAD 2.0 from the authors, then the ETL in
`etl/` followed by the engine modules in order. To simply see the engine's
output, use the live app linked above.

## Performance notes

Three endpoints originally recomputed results on every request. They now read
published artifacts, with a live-fit fallback so a missing artifact degrades to
*slow* rather than *wrong*:

| | Before | After |
|---|---:|---:|
| `/api/models` worst curve (`CPAD-TK-2513`) | 2,159 s | 0.058 s |
| `/api/series-gamma` across all 34 proteins | 198.8 s | 0.48 s |
| `/api/models` registry gap (`CPAD-TK-2185`) | 5.15 s | 0.003 s |

Each change was verified by diffing full endpoint payloads against the previous
implementation — 34/34, 14/14 and 13/13 byte-identical respectively. The
speedups move computation, never results.

## Attribution

Analysis is built on **CPAD 2.0** (Curated Protein Aggregation Database).
Attribution — source dataset, citation, and engine build id — is carried on
every API response and shown in the app footer.

## Author

Yassin Ali — independent research project.
