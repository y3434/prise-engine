"""
Tests for the CPAD 2.0 kinetics ETL.

Two groups:
  * pure-function tests (no data needed) — time/concentration/assay parsing;
  * artifact tests — validate the structure & counts of the produced outputs
    under data/processed/ (run the ETL first).

Runnable two ways:
    python etl/test_cpad_etl.py        # standalone, prints PASS/FAIL
    pytest etl/test_cpad_etl.py        # if pytest is installed
"""
from __future__ import annotations

import json
from pathlib import Path

from cpad_kinetics_etl import (
    MIN_FIT_POINTS,
    classify_assay,
    parse_concentration,
    parse_hours,
)
from cpad_sequence_etl import (
    _drop_pasta_sentinel,
    _is_wt,
    _num,
    _reduce,
    _summary,
)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

# Numbers verified against full_dataset_17_11_19 (see PRISE_DESIGN.md §1.4a).
EXPECTED = {
    "curves": 1654,
    "curves_prelim_fittable_ge6pts": 1526,
    "concentration_series_groups": 52,
    "proteins_with_concentration_series": 34,
    "endpoint_rows": 688,
    "rate_concentration_rows": 350,
}


# --------------------------- pure-function tests --------------------------- #
def test_parse_hours_units():
    assert parse_hours("Fluorescence Intensity at 24 Hours") == 24.0
    assert parse_hours("Turbidity Measured at 30 Minutes") == 0.5
    assert parse_hours("Fluorescence Intensity at 2 Days") == 48.0
    assert parse_hours("Fluorescence Intensity at 3600 Seconds") == 1.0
    assert parse_hours("Fluorescence Intensity at 0 Hours") == 0.0
    assert parse_hours("no time here") is None
    assert parse_hours(None) is None


def test_parse_concentration():
    assert parse_concentration("20 microM")["value_uM"] == 20.0
    assert parse_concentration("1 mM")["value_uM"] == 1000.0
    assert parse_concentration("500 nM")["value_uM"] == 0.5
    assert parse_concentration("0.1 microM")["value_uM"] == 0.1
    assert parse_concentration(None)["value_uM"] is None


def test_classify_assay():
    assert classify_assay("ThT") == ("ThT", True)
    assert classify_assay("Thioflavin T") == ("ThT", True)
    assert classify_assay("Turbidity (350 nm)") == ("turbidity", False)
    name, mass = classify_assay("Some Unknown Assay")
    assert name == "Some Unknown Assay" and mass is None


# --------------------- sequence-ETL pure-function tests -------------------- #
def test_num_handles_scientific_notation():
    assert _num("1.2e-3") == 0.0012
    assert _num("4E+05") == 400000.0
    assert _num("-3.5e2") == -350.0
    assert _num("-4.53937") == -4.53937
    assert _num("42") == 42.0
    assert _num(None) is None


def test_pasta_sentinel_dropped_before_reduction_and_summary():
    # PASTA's +10000 "no cross-β pairing" sentinel must be MISSING for BOTH the
    # min-reduction and the summary (otherwise the mean is poisoned, e.g. Aβ +103).
    vals = [10000.0, -5.0, -4.0, -3.0]
    clean = _drop_pasta_sentinel(vals)
    assert 10000.0 not in clean and len(clean) == 3
    assert _reduce(clean, "min") == -5.0                # most stable, not the sentinel
    s = _summary(clean)
    assert s["max"] == -3.0 and s["mean"] < 0           # mean is negative, not +2498


def test_is_wt_excludes_engineered_mutants():
    for wt in (None, "", "No", "no", "WT", "Wild Type", "wildtype"):
        assert _is_wt(wt) is True, wt
    for mut in ("A30G", "Q686K, A692P", "E22Q", "N27I, I32K"):
        assert _is_wt(mut) is False, mut


# ------------------------------ artifact tests ----------------------------- #
def _load_manifest():
    return json.loads((PROC / "etl_manifest.json").read_text(encoding="utf-8"))


def _iter_curves():
    with open(PROC / "curves.jsonl", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def test_manifest_counts_match_expected():
    counts = _load_manifest()["counts"]
    for k, v in EXPECTED.items():
        assert counts[k] == v, f"{k}: got {counts[k]} expected {v}"


def test_curve_structure_invariants():
    n = 0
    for s in _iter_curves():
        n += 1
        x, y = s["x_hours"], s["y_intensity"]
        assert len(x) == len(y) == s["n_points"], s["series_id"]
        assert all(v is not None for v in x) and all(v is not None for v in y)
        assert x == sorted(x) and len(set(x)) == len(x), \
            f"{s['series_id']} times not strictly increasing"
        assert s["prelim_fittable"] == (s["n_points"] >= MIN_FIT_POINTS)
        assert s["digitization_uncertainty"] is True
    assert n == EXPECTED["curves"]


def test_concentration_series_consistency():
    cs = json.loads((PROC / "concentration_series.json").read_text(encoding="utf-8"))
    assert len(cs) == EXPECTED["concentration_series_groups"]
    member_ids = {sid for g in cs for sid in g["member_series_ids"]}
    for g in cs:
        assert g["n_distinct_concentrations"] >= 3
        assert g["n_curves"] == len(g["member_series_ids"])
    # every curve tagged as a series member must carry the matching mode/id
    for s in _iter_curves():
        if s["series_id"] in member_ids:
            assert s["data_mode"] == "concentration_series"
            assert s["concentration_series_id"] is not None


def test_sequence_structure_abeta_pasta_and_waltz():
    """Aβ/P05067 in the regenerated sequence artifact: PASTA summary mean must be
    NEGATIVE (sentinel dropped, WT-only) and Waltz must be COVERAGE (value=None)."""
    d = json.loads((PROC / "sequence_structure.json").read_text(encoding="utf-8"))
    rec = d["by_uniprot"]["P05067"]
    # WT-only reduction bookkeeping
    assert rec["n_peptides_wt"] < rec["n_peptides_total"]      # mutants excluded
    assert rec["peptide_length_range"] is not None
    assert "NuAPRpred" in rec["excluded_available_predictors"]
    sp = rec["sequence_predictors"]
    pasta = sp["pasta"]
    assert pasta["summary"]["max"] < 10000                    # sentinel gone
    assert pasta["summary"]["mean"] < 0                        # was a fake +103
    assert pasta["summary"]["min"] == pasta["value"]          # min-reduction
    waltz = sp["waltz"]
    assert waltz["value"] is None                             # COVERAGE, not magnitude
    assert "n_waltz_amyloid_peptides" in waltz
    assert "UNTESTED" in waltz["note"]


def test_sequence_structure_zero_wt_protein_emits_unavailable():
    """A protein with only engineered-mutant peptides still emits a record, but
    its predictors are unavailable with the documented reason."""
    d = json.loads((PROC / "sequence_structure.json").read_text(encoding="utf-8"))
    zero = [v for v in d["by_uniprot"].values()
            if v.get("n_peptides_total", 0) > 0 and v.get("n_peptides_wt", 0) == 0]
    assert zero, "expected at least one mutant-only protein"
    v = zero[0]
    assert v["sequence_predictors"]["tango"]["available"] is False
    assert v["sequence_predictors"]["tango"]["reason"] == "only mutant peptides deposited"


def test_known_abeta40_curve():
    target = next((s for s in _iter_curves() if s["series_id"] == "CPAD-TK-1039"), None)
    assert target is not None, "expected curve CPAD-TK-1039 (Abeta40) missing"
    assert target["protein_id"].startswith("Amyloid Beta")
    assert target["condition_vector"]["pdb_id"] == "1IYT"
    assert target["condition_vector"]["concentration"]["value_uM"] == 20.0
    first3 = [list(p) for p in zip(target["x_hours"], target["y_intensity"])][:3]
    assert first3 == [[0.0, 0.00319], [1.0, 0.03987], [4.0, 0.01542]], first3


# ------------------------------- runner ------------------------------------ #
def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except FileNotFoundError as e:
            print(f"SKIP  {t.__name__}: outputs missing ({e}); run the ETL first")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
