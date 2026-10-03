# PRISE as a web app — architecture notes & insights

> **HISTORICAL — pre-implementation design notes.** Written *before* any web code
> existed, to keep the engine evolving in a web-friendly direction. **The web app
> now ships** (`web/server.py`, `web/index.html`, `web/static/app.js`,
> `web/test_web.py`), so this file is kept as a record of the reasoning, **not as a
> description of what was built**. For the shipped app see `../engine/README.md`
> and `web/README.md`.
>
> **Where the implementation deliberately diverged from the stack recommended below:**
> - **§1 backend** — recommended FastAPI + Pydantic; shipped **stdlib only**
>   (`http.server.ThreadingHTTPServer`). PRISE has no third-party web dependency;
>   numpy/scipy remain the only non-stdlib libraries in the whole project.
> - **§3 async compute** — recommended arq/RQ + Redis; shipped **no job queue at
>   all**. The app took this section's own *other* option — precompute the frozen
>   corpus at build time into `data/processed/` and serve it read-only — which
>   removed the need for a queue rather than deferring it.
> - **§1 frontend** — recommended React + TypeScript + Plotly; shipped a single
>   dependency-free `static/app.js` with no build step.
>
> The load-bearing constraints in **§2 (data residency)** and the honesty-UI intent
> in **§5** carried through to the implementation; the stack recommendations did not.
> Treat §§1 and 3 as superseded, and §§2, 5, 6 as still-relevant reasoning.

## 1. The core is already web-shaped — keep it that way
Every module is a **pure function: data in → JSON-serialisable data out**
(`triage_series(series) → series`, `fit_curve(series) → fits`, …). That is
exactly the shape a web API wants. Two cheap disciplines to maintain as we build:
1. keep each module callable as a pure function (not only a CLI);
2. converge on one stable **result schema** (`ProteinAnalysisResult`, design §3-M7)
   that the API serialises unchanged.

Recommended stack (chosen to fit what we already have):
- **Backend:** FastAPI + Pydantic. Pydantic mirrors our dataclasses almost 1:1,
  so request/response models are nearly free.
- **Async compute:** arq or RQ + Redis (see §3).
- **Frontend:** React + TypeScript; **Plotly** for curves/fits; **Mol\*** or
  **NGL** for the PDB structure-linkage view.
- **Deploy:** one container with the **frozen corpus baked in server-side**
  (see §2); reproducibility via `engine_build_id`.

## 2. Data residency / licensing is a hard constraint (design-shaping)
CPAD bulk data is **attribution-gated and not redistributable** (we gitignore it
for exactly this reason). Consequences for the web app:
- The frozen CPAD/PDB corpus lives **server-side only** — never shipped to the
  browser, never exposed as a bulk download endpoint.
- The browser asks for **analysis**; the server runs the engine and returns
  **derived results** (regimes, fits, scores) + **attribution/citation**. Derived
  analytical results are fine to serve; raw bulk records are not.
- PDB is CC0 → structure data/coordinates *can* be served or fetched client-side
  (e.g. Mol\* loading from RCSB by `PDB-id`).
- This actually **reinforces the "pure analytical core, no runtime DB"** design:
  the server holds a frozen snapshot, not a live CPAD connection.

## 3. Compute model — synchronous vs job queue
Cost ladder (measured): a single descriptive `fit_curve` ≈ 50 ms; the full
kinetic bank over 1,654 curves ≈ 2.4 min; one mechanistic ODE fit ≈ 0.1 s;
**M3 bootstrap-enveloped selection will be seconds–minutes per curve**.
- **Synchronous** (HTTP request → response): single-curve descriptive fit, M0
  browsing, plotting a stored curve.
- **Async job** (queue + poll / WebSocket progress): mechanistic fitting, global
  concentration-series fits, and especially **M3 bootstrap**. Return a `job_id`,
  stream progress, deliver the result when ready.
- **Cache aggressively:** the engine is deterministic given `engine_build_id`, so
  cache results keyed by `(series_id | upload_hash, engine_build_id, params)`.
  Bundled-corpus results can even be **precomputed at build time** and served
  instantly — the browser mostly reads cached analyses, only uploads trigger
  fresh compute.

## 4. API surface (sketch)
```
GET  /proteins                         # M0: browse (name, #datasets, behaviour tag)
GET  /proteins/{id}/datasets           # available curves + conditions for selection
GET  /curves/{series_id}               # raw points (+ M1 triage) for plotting
POST /analyze                          # select dataset(s) OR upload -> M0..M7 result
POST /uploads                          # user CSV -> validated AggregationSeries
GET  /jobs/{job_id}                    # async result (bootstrap / mechanistic / global)
GET  /structures/{pdb_id}              # metadata for the Mol* viewer (or client fetch)
GET  /meta/build                       # engine_build_id, corpus_version, citations
```
Selection/assembly (M0) maps directly: the UI lets the user pick one dataset,
replicates, a concentration series, or a condition set — and the server tags the
`data_mode` and routes.

## 5. UX — make the honesty visible (the product's whole point)
The design's epistemic discipline should be the UI's differentiator, not a
footnote:
- **Curve view:** raw points + fitted-model overlay; shade the **lag** and
  **plateau** regions; badge **right/left censoring** (the design forbids
  plateau-manufacturing — show *why* a t50 is or isn't trusted).
- **Result panel:** descriptive regime *and* the mechanistic verdict — or, when
  the data can't separate mechanisms, a first-class **"indistinguishable:
  {secondary nucleation, fragmentation}"** card rather than a fake winner.
- **Confidence with its caveat:** show calibrated confidence *and* the
  validity-ceiling note ("calibrated on synthetic data; real-data transfer
  inferred"). Never a bare number.
- **`information_yield` badge:** mechanistic / scaling / descriptive / signal-only
  / uninformative — so users instantly see how much the data supports.
- **Degenerate case (the common one):** a compact "here's what's knowable + the
  one experiment that would resolve it" (M9), not a wall of caveats.
- **Structure-linkage:** embed a Mol\*/NGL 3D viewer keyed on the record's
  `PDB-id`, linking behaviour ↔ structure (associative, clearly not causal).

## 6. Safety / robustness
- Validate + size-limit uploads; parse in a sandbox; reject malformed series at
  the M1 gate (already does).
- Rate-limit / queue expensive compute (bootstrap) to prevent abuse.
- Stamp every result with `engine_build_id` so the UI can flag stale/incomparable
  results (the design already blocks cross-build comparison).

## 7. Suggested build order (when we get there)
1. Wrap the existing engine in a minimal FastAPI app (`/curves`, `/analyze` sync).
2. Precompute + serve the bundled-corpus results (read-only, fast, no queue).
3. Add uploads + the async job queue once M3 bootstrap lands.
4. Frontend: M0 browse → curve+fit plot → result panel → structure viewer.
5. Polish the honesty UI (censoring shading, degeneracy cards, M9 hints).

**Bottom line:** nothing about the web app changes how we build the engine — it
only rewards the choices already made (pure functions, JSON schemas, frozen
deterministic corpus, no runtime DB). The main new concern is operational
(server-side data residency + an async queue for bootstrap), not architectural.
