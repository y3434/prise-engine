"""
PRISE — CPAD 2.0 kinetics ETL
=============================

Build-time ETL that parses CPAD 2.0's `aggregation kinetics.xlsx` (bulk download)
into the engine's canonical `AggregationSeries` objects, plus a concentration-series
index, and a frozen/versioned manifest.

This implements the verified schema documented in PRISE_DESIGN.md §1.4a:

  * The kinetics sheet is LONG format: one row per (curve x time-point).
  * `Entry` prefix selects the row class:
        T- = time-kinetics points (the curves)      -> AggregationSeries
        O- = endpoint/summary rows                   -> endpoints (aux)
        R- = rate-vs-concentration (k_agg) rows      -> rate-conc (aux)
  * A curve = rows sharing the same `Time Kinetics` integer id.
  * time      = parsed from `Parameter`  ("Fluorescence Intensity at X Hours")
  * intensity = the `Aggregation Rate` column (misnamed; it is the intensity)
  * conditions = the per-row condition columns (constant within a curve)
  * a concentration_series = curves grouped by matched conditions with >=3 concs.

The ETL is faithful: it loads, structures, and tags the data. It does NOT fit,
normalize, or judge curves — that is M1's job downstream. It only precomputes a
preliminary point-count (the final fittability call is M1's).

Usage:
    python etl/cpad_kinetics_etl.py \
        [--input data/cpad_raw/"aggregation kinetics.xlsx"] \
        [--outdir data/processed]

Outputs (under --outdir):
    curves.jsonl               one AggregationSeries per line (T-rows)
    concentration_series.json  index of >=3-concentration matched groups
    endpoints.jsonl            O-rows (per-condition endpoints)
    rate_concentration.jsonl   R-rows (k_agg vs concentration)
    etl_manifest.json          version, counts, source hash, schema id
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import openpyxl
except ImportError:  # pragma: no cover
    sys.exit("openpyxl is required:  python -m pip install openpyxl")

SCHEMA_VERSION = "cpad-kinetics-1.0"
SOURCE_DATASET = "CPAD 2.0 (full_dataset_17_11_19)"
SOURCE_CITATION = "Rawat P, Prabakaran R, et al. CPAD 2.0. Amyloid. 2020;27(2)."

# Sheet column names (verified against aggregation kinetics.xlsx)
C_ENTRY = "Entry"
C_PROT = "Protein Name"
C_UNIPROT = "Uniprot ID"
C_PDB = "PDB ID"
C_MUT = "Mutation"
C_TEMP = "Temperature"
C_PH = "pH"
C_BUFFER = "Buffer Name"
C_BUFCONC = "Buffer Concentration"
C_ION = "Ion"
C_IONCONC = "Ion Concentration"
C_ADDITIVES = "Additives"
C_PROTCONC = "Protein Concentration"
C_MEASURE = "Measure"
C_METHOD = "Method"
C_RATE = "Aggregation Rate"          # NOTE: for T-rows this is the per-point INTENSITY
C_PARAM = "Parameter"               # holds "... at X Hours" for T-rows
C_PMID = "PMID"
C_REF = "Reference"
C_AUTHOR = "Author"
C_YEAR = "Year"
C_REMARKS = "Remarks"
C_TYPE = "Type of Kinetics"
C_TK = "Time Kinetics"              # curve id (integer) for T-rows

# Preliminary point-count threshold (final fittability is M1's call)
MIN_FIT_POINTS = 6
# Distinct concentrations needed to call a matched group a concentration series
MIN_SERIES_CONCS = 3

# Assay -> (canonical name, assay_reports_mass [assumed, unverified])
# Fibril-specific fluorescent / dye assays are treated as mass-proportional
# (UNVERIFIED — flagged), light-scattering/turbidity are not.
_ASSAY_TABLE = [
    (("tht", "thioflavin t"), ("ThT", True)),
    (("ths", "thioflavin s"), ("ThS", True)),
    (("congo",), ("CongoRed", True)),
    (("cytoflu",), ("Cytofluor", True)),
    (("turbid",), ("turbidity", False)),
    (("light scatter", "scatter", "dls", "sls"), ("light_scattering", False)),
    (("fluor",), ("fluorescence", True)),
]


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _clean(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s in {"-", "NA", "N/A", "None", "nan"}:
        return None
    return s


# Time units appearing in the `Parameter` column, normalised to HOURS.
# CPAD records time as Seconds / Minutes / Hours / Days (e.g.
# "Fluorescence Intensity at 24 Hours", "Turbidity Measured at 5 Minutes",
# "Fluorescence Intensity at 2 Days"). First letter is unambiguous: s/m/h/d.
_TIME_FIRST_LETTER_TO_HR = {"s": 1.0 / 3600, "m": 1.0 / 60, "h": 1.0, "d": 24.0}


def parse_hours(parameter_text: Any) -> Optional[float]:
    """Extract the time-point in HOURS from a `Parameter` string, handling
    seconds / minutes / hours / days. Returns None if no time token is found."""
    s = _clean(parameter_text)
    if not s:
        return None
    m = re.search(r"([\d.]+)\s*(seconds?|sec|minutes?|min|hours?|hrs?|days?)\b",
                  s, flags=re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1)) * _TIME_FIRST_LETTER_TO_HR[m.group(2)[0].lower()]


def parse_float(v: Any) -> Optional[float]:
    s = _clean(v)
    if s is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


_UNIT_TO_UM = {"m": 1e6, "mm": 1e3, "um": 1.0, "µm": 1.0, "micro m": 1.0,
               "microm": 1.0, "nm": 1e-3, "pm": 1e-6}


def parse_concentration(v: Any) -> dict:
    """Return {raw, value_uM, unit} normalised to micromolar where possible."""
    raw = _clean(v)
    out = {"raw": raw, "value_uM": None, "unit": None}
    if not raw:
        return out
    m = re.search(r"([\d.]+)\s*([a-zµ]+\s*[mM]?)", raw)
    if not m:
        return out
    val = float(m.group(1))
    unit = m.group(2).lower().replace(" ", "")
    unit = {"microm": "um", "micromolar": "um", "µm": "um"}.get(unit, unit)
    out["unit"] = m.group(2).strip()
    if unit in _UNIT_TO_UM:
        out["value_uM"] = val * _UNIT_TO_UM[unit]
    return out


def classify_assay(method: Any) -> tuple[Optional[str], Optional[bool]]:
    s = _clean(method)
    if not s:
        return None, None
    low = s.lower()
    for keys, (name, mass) in _ASSAY_TABLE:
        if any(k in low for k in keys):
            return name, mass
    return s, None  # unknown assay -> mass-proportionality unknown


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Canonical data objects (subset of PRISE_DESIGN.md §2)
# --------------------------------------------------------------------------- #
@dataclass
class ConditionVector:
    concentration: dict                       # {raw, value_uM, unit}
    temperature_C: Optional[float]
    pH: Optional[float]
    ion: Optional[str]
    ion_concentration: Optional[str]
    ionic_strength_note: Optional[str]
    buffer: Optional[str]
    additives: Optional[str]
    agitation: Optional[str]                  # not a CPAD column -> usually None
    seeded: Optional[bool]                    # not a CPAD column -> usually None
    assay_type: Optional[str]
    assay_reports_mass: Optional[bool]        # assumed/unverified for dye assays
    tht_concentration: Optional[str]          # parsed from additives if present
    construct_id: Optional[str]               # mutation = a distinct construct
    pdb_id: Optional[str]
    field_provenance: dict = field(default_factory=dict)


@dataclass
class AggregationSeries:
    series_id: str
    source: str
    source_study: dict                        # {pmid, reference, author, year}
    protein_id: str
    uniprot_id: Optional[str]
    data_mode: str                            # kinetic | concentration_series member
    concentration_series_id: Optional[str]
    x_hours: list[float]
    y_intensity: list[float]
    units: dict                               # {x:"hours", y:"a.u. (digitized)"}
    n_points: int
    prelim_fittable: bool                     # >=MIN_FIT_POINTS; M1 makes final call
    signal_basis: str                         # "as_reported" (unknown norm.)
    digitization_uncertainty: bool            # always True for CPAD
    condition_vector: ConditionVector
    quality_flags: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# ETL core
# --------------------------------------------------------------------------- #
def _row_getter(header: list[str]):
    idx = {h: i for i, h in enumerate(header)}

    def get(row, name):
        i = idx.get(name)
        return row[i] if (i is not None and i < len(row)) else None
    return get


def build_condition_vector(get, row) -> ConditionVector:
    additives = _clean(get(row, C_ADDITIVES))
    tht = None
    if additives:
        m = re.search(r"([\d.]+\s*[µu]?M)\s*tht", additives, flags=re.IGNORECASE)
        if m:
            tht = m.group(1)
    assay, mass = classify_assay(get(row, C_METHOD))
    ion = _clean(get(row, C_ION))
    ionconc = _clean(get(row, C_IONCONC))
    cv = ConditionVector(
        concentration=parse_concentration(get(row, C_PROTCONC)),
        temperature_C=parse_float(get(row, C_TEMP)),
        pH=parse_float(get(row, C_PH)),
        ion=ion,
        ion_concentration=ionconc,
        ionic_strength_note=(f"{ion} {ionconc}" if ion else None),
        buffer=" ".join(x for x in [_clean(get(row, C_BUFFER)),
                                    _clean(get(row, C_BUFCONC))] if x) or None,
        additives=additives,
        agitation=None,
        seeded=None,
        assay_type=assay,
        assay_reports_mass=mass,
        tht_concentration=tht,
        construct_id=_clean(get(row, C_MUT)),
        pdb_id=_clean(get(row, C_PDB)),
    )
    # provenance: known vs unknown per comparability-critical field
    cv.field_provenance = {
        "concentration": "known" if cv.concentration["raw"] else "unknown",
        "pH": "known" if cv.pH is not None else "unknown",
        "temperature": "known" if cv.temperature_C is not None else "unknown",
        "ionic_strength": "known" if cv.ion else "unknown",
        "assay_type": "known" if cv.assay_type else "unknown",
        "agitation": "unknown",        # never a CPAD column
        "seeded": "unknown",           # never a CPAD column
        "tht_concentration": "known" if cv.tht_concentration else "unknown",
    }
    return cv


def series_signature(cv: ConditionVector, protein: str) -> tuple:
    """Matched-condition key for concentration-series grouping (concentration
    deliberately excluded — that is the axis that varies)."""
    return (protein, cv.pH, cv.temperature_C, cv.assay_type,
            cv.construct_id, cv.ion, cv.ion_concentration)


def run_etl(input_path: Path, outdir: Path) -> dict:
    wb = openpyxl.load_workbook(input_path, read_only=True, data_only=True)
    ws = wb["Sheet1"]
    it = ws.iter_rows(values_only=True)
    header = list(next(it))
    get = _row_getter(header)

    curve_rows: dict[Any, list] = defaultdict(list)   # T-rows by Time Kinetics id
    endpoints: list[dict] = []                         # O-rows
    rate_conc: list[dict] = []                         # R-rows

    for row in it:
        entry = _clean(get(row, C_ENTRY)) or ""
        typ = _clean(get(row, C_TYPE))
        if entry.startswith("T") and typ == "Time":
            curve_rows[get(row, C_TK)].append(row)
        elif entry.startswith("O"):
            endpoints.append({
                "entry": entry, "protein": _clean(get(row, C_PROT)),
                "concentration": parse_concentration(get(row, C_PROTCONC)),
                "pH": parse_float(get(row, C_PH)),
                "temperature_C": parse_float(get(row, C_TEMP)),
                "assay": classify_assay(get(row, C_METHOD))[0],
                "parameter": _clean(get(row, C_PARAM)),
                "value": parse_float(get(row, "Change in Aggregation Rate")),
                "pmid": _clean(get(row, C_PMID)),
            })
        elif entry.startswith("R"):
            rate_conc.append({
                "entry": entry, "protein": _clean(get(row, C_PROT)),
                "concentration": parse_concentration(get(row, C_PROTCONC)),
                "k_agg": parse_float(get(row, C_RATE)),
                "parameter": _clean(get(row, C_PARAM)),
                "pmid": _clean(get(row, C_PMID)),
            })
    wb.close()

    # ---- build AggregationSeries from each curve group -------------------- #
    series_list: list[AggregationSeries] = []
    for tk, rows in curve_rows.items():
        pts = []
        for r in rows:
            t = parse_hours(get(r, C_PARAM))
            y = parse_float(get(r, C_RATE))
            if t is not None and y is not None:
                pts.append((t, y))
        if not pts:
            continue
        # sort by time, de-duplicate identical timestamps (keep first)
        pts.sort(key=lambda p: p[0])
        xs, ys, seen = [], [], set()
        for t, y in pts:
            if t in seen:
                continue
            seen.add(t)
            xs.append(t)
            ys.append(y)
        ref = rows[0]
        cv = build_condition_vector(get, ref)
        protein = _clean(get(ref, C_PROT)) or "unknown"
        n = len(xs)
        s = AggregationSeries(
            series_id=f"CPAD-TK-{tk}",
            source="bundled:CPAD2.0",
            source_study={
                "pmid": _clean(get(ref, C_PMID)),
                "reference": _clean(get(ref, C_REF)),
                "author": _clean(get(ref, C_AUTHOR)),
                "year": _clean(get(ref, C_YEAR)),
            },
            protein_id=protein,
            uniprot_id=_clean(get(ref, C_UNIPROT)),
            data_mode="kinetic",
            concentration_series_id=None,
            x_hours=xs,
            y_intensity=ys,
            units={"x": "hours", "y": "a.u. (digitized, normalisation unknown)"},
            n_points=n,
            prelim_fittable=n >= MIN_FIT_POINTS,
            signal_basis="as_reported",
            digitization_uncertainty=True,
            condition_vector=cv,
            quality_flags={
                "raw_rows": len(rows),
                "dropped_pts": len(rows) - n,
                "graph_digitized": True,
            },
        )
        series_list.append(s)

    # ---- concentration-series grouping ----------------------------------- #
    groups: dict[tuple, list[AggregationSeries]] = defaultdict(list)
    for s in series_list:
        groups[series_signature(s.condition_vector, s.protein_id)].append(s)

    conc_series_index = []
    for sig, members in groups.items():
        concs = {m.condition_vector.concentration["raw"] for m in members
                 if m.condition_vector.concentration["raw"]}
        if len(concs) >= MIN_SERIES_CONCS:
            cs_id = f"CPAD-CS-{abs(hash(sig)) % (10**8):08d}"
            for m in members:
                m.concentration_series_id = cs_id
                m.data_mode = "concentration_series"
            conc_series_index.append({
                "concentration_series_id": cs_id,
                "protein": sig[0], "pH": sig[1], "temperature_C": sig[2],
                "assay": sig[3], "mutation": sig[4],
                "ion": sig[5], "ion_concentration": sig[6],
                "n_curves": len(members),
                "n_distinct_concentrations": len(concs),
                "member_series_ids": [m.series_id for m in members],
            })

    # ---- write outputs --------------------------------------------------- #
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "curves.jsonl", "w", encoding="utf-8") as fh:
        for s in series_list:
            fh.write(json.dumps(dataclasses.asdict(s)) + "\n")
    with open(outdir / "concentration_series.json", "w", encoding="utf-8") as fh:
        json.dump(conc_series_index, fh, indent=2)
    with open(outdir / "endpoints.jsonl", "w", encoding="utf-8") as fh:
        for e in endpoints:
            fh.write(json.dumps(e) + "\n")
    with open(outdir / "rate_concentration.jsonl", "w", encoding="utf-8") as fh:
        for r in rate_conc:
            fh.write(json.dumps(r) + "\n")

    proteins_with_series = {c["protein"] for c in conc_series_index}
    fittable = sum(1 for s in series_list if s.prelim_fittable)
    usable_pairs = sum(s.n_points for s in series_list)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": SOURCE_DATASET,
        "source_citation": SOURCE_CITATION,
        "source_file": input_path.name,
        "source_sha256": sha256_of(input_path),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "curves": len(series_list),
            "curves_prelim_fittable_ge6pts": fittable,
            "usable_time_intensity_pairs": usable_pairs,
            "distinct_proteins": len({s.protein_id for s in series_list}),
            "concentration_series_groups": len(conc_series_index),
            "proteins_with_concentration_series": len(proteins_with_series),
            "endpoint_rows": len(endpoints),
            "rate_concentration_rows": len(rate_conc),
        },
        "notes": [
            "Long-format source; curve = Time Kinetics id; time from Parameter; "
            "intensity from 'Aggregation Rate' column.",
            "All kinetic values are CPAD graph-digitizations (marginal error).",
            "Final fittability / artifact / censoring decisions belong to M1.",
        ],
    }
    with open(outdir / "etl_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="CPAD 2.0 kinetics ETL for PRISE")
    ap.add_argument("--input", type=Path,
                    default=root / "data" / "cpad_raw" / "aggregation kinetics.xlsx")
    ap.add_argument("--outdir", type=Path, default=root / "data" / "processed")
    args = ap.parse_args(argv)
    if not args.input.exists():
        sys.exit(f"input not found: {args.input}")
    print(f"[ETL] reading {args.input.name} ...")
    manifest = run_etl(args.input, args.outdir)
    print(f"[ETL] wrote outputs to {args.outdir}")
    print(json.dumps(manifest["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
