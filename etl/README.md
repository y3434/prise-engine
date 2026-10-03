# PRISE ETL — CPAD 2.0 kinetics + sequence/structure

Build-time ETL that turns CPAD 2.0's bulk workbooks into the engine's canonical
inputs:
- **`cpad_kinetics_etl.py`** → `AggregationSeries` objects + concentration-series
  index (from `aggregation kinetics.xlsx`); the verified schema in
  `../PRISE_DESIGN.md` §1.4a.
- **`cpad_sequence_etl.py`** → per-UniProt sequence/structure lookups for M6 (from
  `aggregating peptides.xlsx`, `APR information.xlsx`, `amyloid structure.xlsx`).

## Prerequisites
```
python -m pip install openpyxl
```
The raw CPAD data must be present (it is **git-ignored** — attribution-gated,
not redistributable). Default location:
```
data/cpad_raw/aggregation kinetics.xlsx
```
Obtain it via CPAD's bulk download (https://web.iitm.ac.in/bioinfo2/cpad2/downloadall/),
then unzip `full_dataset_17_11_19.zip` into `data/cpad_raw/`.

## Run
```
python etl/cpad_kinetics_etl.py                 # uses default in/out paths
python etl/cpad_kinetics_etl.py --input PATH --outdir DIR
python etl/cpad_sequence_etl.py                 # sequence/structure layer (M6)
```

## Outputs (`data/processed/`, git-ignored)
| file | contents |
|---|---|
| `curves.jsonl` | one `AggregationSeries` per line (T-rows → time-series curves) |
| `concentration_series.json` | matched-condition groups with ≥3 concentrations (γ-eligible) |
| `endpoints.jsonl` | O-rows (per-condition endpoints) |
| `rate_concentration.jsonl` | R-rows (k_agg vs concentration) |
| `etl_manifest.json` | counts, source SHA-256, schema id, build timestamp |

## Verified output (full_dataset_17_11_19)
**Kinetics** (`cpad_kinetics_etl.py`):
- 1,654 curves · 1,526 fittable (≥6 pts) · 81,484 usable (time,intensity) pairs
- 105 distinct proteins
- 52 concentration-series groups across 34 proteins
- 688 endpoint rows · 350 rate-vs-concentration rows

**Sequence/structure** (`cpad_sequence_etl.py` `cpad-sequence-1.1`,
`sequence_structure.json` + `sequence_manifest.json`): 382 UniProts · 231 with
**WT-only** predictor reductions (TANGO, AGGRESCAN, PASTA 2.0 with the `+10000`
no-pairing sentinel stripped, Waltz-DB amyloid-hexapeptide **coverage** counts) ·
267 with APRs · 74 with amyloid structures. Predictor scores are tagged
`as-bundled-in-CPAD-2.0` (upstream tool versions not recorded by CPAD); AGGRESCAN/
TANGO reductions are fragment-level maxima over deposited WT peptides (not whole-
sequence native runs); **Zyggregator & CamSol are absent** from the bundle and
**NuAPRpred is intentionally excluded** — all declared so for M6.

## Schema recap
- Long format: 1 row per (curve × time-point).
- Curve id = `Time Kinetics`; time parsed from `Parameter` ("…at X {sec|min|hours|days}")
  → normalised to **hours**; intensity = `Aggregation Rate` column (misnamed).
- The ETL is faithful: it loads/structures/tags only. Fitting, normalisation,
  artifact + censoring decisions belong to **M1** downstream.

## Tests
```
python etl/test_cpad_etl.py        # standalone PASS/FAIL
pytest etl/test_cpad_etl.py        # if pytest installed
```
Pure-function tests run without data; artifact tests require the ETL outputs.
