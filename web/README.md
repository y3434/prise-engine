# PRISE web app

A zero-install local web UI over the completed PRISE protein-aggregation analysis
engine. It is a **stdlib-only** backend (`http.server.ThreadingHTTPServer` + a tiny
path router — no FastAPI / Flask / uvicorn) plus a **vanilla-JS single-page app**
(no framework, no build step). It loads the precomputed **M0..M16** artifacts from
`../data/processed` at startup, caches them in memory, and serves derived
analytical results + single curves.

Everything past the core (M10 cross-modal, M11 metadata quality, M12 information
content, M13 BOED, M14 meta-analysis, M15 literature ingestion, M16
sequence↔structure↔kinetics) is loaded **optionally**: if an artifact is missing the
corresponding endpoint answers honestly (`available: false` + a note saying which
engine module to run) and the rest of the app is unaffected. A missing artifact never
produces a blank page, a 500, or an invented number.

## Launch

```bash
python web/server.py            # serves http://127.0.0.1:8000
python web/server.py 8123       # override port (positional arg)
PRISE_PORT=8123 python web/server.py
```

Open <http://127.0.0.1:8000>. Python 3.10+ (3.13 tested). No pip installs are
required for Browse / Corpus overview / single curves. The live **Analyze your own
curve** path and the fitted-model overlay additionally use the engine in
`../engine`; if that import fails (e.g. SciPy missing) the app degrades gracefully —
Browse and the overview still work and the page is never blank.

## The flow (Browse-first)

The app **lands on Browse & analyze datasets**. The intended path is:

1. Pick a protein in the left list (search by name / UniProt).
2. The protein's **CPAD 2.0 datasets** appear as a scannable table — one row per
   curve, showing concentration (µM), pH, temperature, assay (mass / non-mass),
   number of points, censoring class, information-yield tier and descriptive regime.
3. The **most informative dataset is auto-selected**, so you immediately see a
   concrete analysis. Click any other row to switch.
4. The analysis leads with the dataset's **identity + conditions** and the
   **curve & fit plot**, followed by a **model comparison / behaviour
   classification** card (the full candidate bank ranked by AICc, with
   multi-model overlay toggles on the plot, an identifiability line, bootstrap
   selection stability and a family legend), features, propensity (three views),
   the sequence axis, the M9 recommended next experiment, and an honesty section
   (validity ceiling, failed mechanistic gates, censoring tags, caveats).
5. When the protein has a **concentration series**, a protein-level
   **Concentration-series scaling (dual-γ)** card shows γ_global vs γ_regression
   (reported separately, never merged), the reliability flag + reasons, the
   disagreement verdict, the log–log curvature test, and a log–log scatter of
   t50 vs [monomer] with the fitted −γ slope line (censored points marked
   distinctly). Tier-B fits and γ are honesty-flagged: a good fit is **not** a
   licensed mechanism call — M5 still refuses mechanism on a single curve.

You do **not** upload anything to analyze CPAD curves — every CPAD 2.0 dataset is
directly analyzable from Browse. The **Analyze your own curve** tab is only for
curves you bring yourself (paste two columns or upload a CSV); it runs the live
engine (M1 triage → M2 fit → M4 features → M5 classify).

The **Corpus overview** tab is a corpus-wide aggregate of those same per-dataset
results — the inference ladder, information-yield tiers, blocker census, the
structural-vs-licensed gap and the validity ceiling across all CPAD curves. It is
*not* the analysis of any one dataset and *not* invented numbers; each count is
computed by the engine from the individual curve results.

## Endpoints

| Method & path                | Returns |
|------------------------------|---------|
| `GET /`                      | the SPA (`index.html`) |
| `GET /static/<file>`         | JS / CSS assets (path-traversal guarded) |
| `GET /api/health`            | engine status + artifact counts |
| `GET /api/overview`          | corpus-wide rollup (M7/M8/M9 aggregates) |
| `GET /api/proteins[?q=]`     | protein list (grouped), optional name/UniProt filter |
| `GET /api/protein/{id}`      | one protein: all `units` (per-curve `ProteinAnalysisResult`), M9 recs, sequence axis, structure linkage |
| `GET /api/curve/{series_id}` | **single** curve: x/y, processed y, M1 triage, fitted-model overlay |
| `GET /api/models/{series_id}` | **model comparison / behaviour classification**: the full candidate bank (descriptive + biphasic + autocatalytic + Tier-B Knowles/Cohen mechanistic) fit explicitly to the stored curve, ranked by AICc with ΔAICc + Akaike weights, a dense fitted grid for the top ~6 (for overlay), a sensitivity-collinearity identifiability verdict on the winner (the column-normalised-Jacobian condition number — **not** the Fisher Information Matrix; the true observed FIM for the ODE family lives in `global_fit.py`), and bootstrap selection stability when available. Honesty-flagged: a high Tier-B R² is **not** a licensed mechanism. |
| `GET /api/series-gamma/{protein_id}` | **concentration-series scaling (dual-γ)** for a protein with a series (list; a protein may have >1). Each entry: `gamma_regression` (with `gamma_physical` — the negative-γ driving-force gate — and `gamma_raw`) + `gamma_global` (the **phenomenological master-curve collapse**, `is_shared_rate_ode_fit=false`; separate, never merged), the fixed `disagreement` verdict (always-reported `disagree_pointwise` + `difference` even without CIs; `shape_not_concentration_invariant` from the collapse-quality gate), `curvature_test`, the per-member `(concentration_uM, t50, censoring_class)` points (with `log10_*`) for the log–log scatter + fitted −γ slope line, and — when `engine/global_fit.py` has been run for that series — `global_ode_fit`: the **shared-rate Knowles/Cohen ODE fit**'s third `gamma_mechanistic` (from ESTIMATED reaction orders n_c/n_2), `best_mechanism`, `mechanism_aicc_ranking`, and the identifiability/sloppiness report (`sloppy`, `fim_condition_number`, identifiable combinations, per-rate status). Honesty-first: reaction-order CONSTRAINT + identifiable combos, **not** a unique mechanism (M5 still refuses). |
| `GET /api/series-list`       | proteins that have a concentration series (dual-γ available) |
| `GET /api/dose-response/{protein_id}` | dose–response (k_agg vs concentration) for a protein with CPAD R-rows; `404` when absent |
| `GET /api/crossmodal`        | **M10** cross-modal Sequence × Kinetics × Structure payload |
| `GET /api/conformal`         | the frozen conformal-prediction calibration (finite-sample-valid t50 intervals + regime prediction sets) |
| `GET /api/reality-check`     | the blind external reality check (§6C falsification run) |
| `GET /api/metadata-overview` | **M11** corpus metadata-quality summary |
| `GET /api/metadata/{series_id}` | **M11** per-dataset metadata assessment; `404` when absent |
| `GET /api/information/{series_id}` | **M12** per-curve information-content record; `404` when absent |
| `GET /api/boed/{series_id}`  | **M13** Bayesian optimal experimental design recommendation; `404` when absent |
| `GET /api/pooling/{protein_id}` | deterministic empirical-Bayes partial-pooling result; `404` when absent |
| `GET /api/meta/{protein_id}` | **M14** cross-study hierarchical meta-analysis. Always `200`: a single-study or absent protein is an honest answer (`status: single_study`), not a 404, and a single study never gets a fabricated forest plot |
| `GET /api/structure/{protein_id}` | **M16** sequence↔structure↔kinetics bridge: APR/sequence proxy features, the 8 structural features, the GNN-ready graph summary, structure sources and the corpus structure→kinetics association table. Always `200` (an absent protein is an honest `available: false` that still surfaces the corpus table). Each of the 6 genuinely-3D features is a nested block `{proxy, has_real_3d[, real_3d]}`: the 1-D sequence proxy is **always** present and stamped `is_3d_derived=false`, and a **real** 3D value (`derivation=pdb_3d`, `is_3d_derived=true`, per-feature `fidelity`) is added **only** where a PDB was actually fetched and parsed. No PDB ⇒ no `real_3d` key — a proxy is never dressed up as a measurement |
| `GET /api/literature`        | **M15** literature-ingestion & claim-extraction layer (append-only claim ledger + validation). Its honesty block carries `frontend_capabilities`: the **detected** state of the four pluggable front-end seams (`pdf_to_document`, `ocr_page`, `ner_extract`, `digitize_figure_curve`). A seam is unavailable when its dependency is not installed on this interpreter — a dependency-availability fact, not a claim about network access |
| `POST /api/analyze`          | live single-curve analysis of a user-supplied `{x, y, concentration_uM?, assay?}` |
| `POST /api/cohort`           | **M0** multi-dataset assembly + routing over `{series_ids: [...]}` |

## Data residency

CPAD bulk is attribution-gated. The server serves **only derived results + single
curves** (one series at a time). There is deliberately **no** bulk-dump /
"download all curves" endpoint. Attribution (source dataset, citation, engine build
id) is carried on every response and shown in the page footer.

## Status

What ships **is** this stdlib `http.server` backend + vanilla-JS SPA — that is the
app, not a placeholder for one. The design docs additionally sketch a FastAPI +
React deployment as a *possible future* production path; that is an aspiration, and
none of it is built or depended on here. Keeping the served app stdlib-only is a
deliberate constraint (install-free, dependency-light, reproducible), so treat any
proposal to add a runtime dependency to `web/` as a change to that constraint rather
than routine work.

## Tests

```bash
python web/test_web.py
```

Stdlib-only and deterministic (no pytest, no third-party runner; exit 0 = pass). It
boots the server on an ephemeral port in a background thread and exercises `GET /`,
`GET /static/app.js` (parsed for syntax when `esprima` happens to be installed, so a
frontend that silently fails to load can't pass), `/api/health`, `/api/overview`,
`/api/proteins`, `/api/protein/{first}`, `/api/curve/{first series of that protein}`,
`/api/models/{a fittable series}` (asserts a ranked models list spanning >1 family
incl. a Tier-B mechanistic family, AICc-ascending order, a flagged winner and an
identifiability block), `/api/series-gamma/{a protein with a series}` (asserts
`gamma_regression` + `gamma_global` + the per-concentration points), the M11–M16
endpoints, `POST /api/analyze` with a synthetic sigmoid and `POST /api/cohort`.

The M16 `/api/structure` checks assert the **full nested schema on both branches** —
one protein that has a fetched PDB and one that does not — including that a proxy is
always stamped `is_3d_derived=false`, that a real measurement carries `pdb_3d` +
`fidelity`, and that a protein with no PDB grows **no** `real_3d` key and no contact
map. Fabricating a 3D number, or flipping a proxy's honesty stamp, fails the suite.

**No network required.** The suite is hermetic: it reads frozen artifacts from
`../data/processed` and talks only to its own loopback server. Verified by running it
with all non-loopback sockets and DNS hard-blocked — 0 outbound attempts, exit 0. The
PDB fetching lives in `engine/m16_structure.py` (artifact *build* time, cached under
`data/pdb_cache/`); `web/server.py` never imports it.

Runtime is roughly **80 s**, and it is compute-bound, not I/O-bound: a single
`GET /api/series-gamma/...` accounts for ~63 s and the three `POST /api/cohort` calls
for ~11 s, because those endpoints do real curve fitting on request. The remaining
~40 endpoint checks together cost well under a second.
