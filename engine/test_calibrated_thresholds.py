"""
Tests for the GOVERNED calibrated-thresholds apply artifact
===========================================================

Verifies that the versioned artifact loads, is stamped `thresholds-1.0`, carries
BOTH the previous (`first_pass`) and applied (`calibrated`) value for every
threshold plus the Service-C source, carries a signed/dated apply record, and that
M1/M3/M5 now READ the calibrated value (not the first-pass one). Also verifies the
engine_build_id folds the thresholds axis (so applying invalidates old results).

    python engine/test_calibrated_thresholds.py
    pytest engine/test_calibrated_thresholds.py
"""
from __future__ import annotations

import calibrated_thresholds as ct


# ------------------------------ artifact loads ----------------------------- #
def test_version_stamped():
    assert ct.THRESHOLDS_VERSION == "thresholds-1.0"
    m = ct.manifest()
    assert m["thresholds_version"] == "thresholds-1.0"
    assert m["source_artifact"] == "service_c_calibration.json"
    assert m["source_version"] == "service-c-1.0"


def test_every_threshold_carries_both_values_and_source():
    """(b) BOTH first_pass and calibrated retained, with the Service-C source."""
    for name, rec in ct.THRESHOLDS.items():
        assert "first_pass" in rec, name
        assert "calibrated" in rec, name
        assert rec["first_pass"] is not None, name
        assert rec["calibrated"] is not None, name
        assert "source" in rec and rec["source"].get("engine") == "service-c-1.0", name


def test_apply_record_signed_dated_versioned():
    """(c) the apply record is signed/dated/version-bound + rationale-bearing."""
    ar = ct.APPLY_RECORD
    assert ar["applied_at"] == "2026-07-01"
    assert ar["by"] == "governed apply step"
    assert ar["rationale"] == "Service C held-out calibration (service-c-1.0)"
    assert ar["source_artifact"] == "service_c_calibration.json"
    assert ar["thresholds_version"] == "thresholds-1.0"
    assert ar["reversible"] is True


def test_calibrated_values_are_the_sensibly_rounded_recommendations():
    assert ct.calibrated_value("M1_LAG_SLOPE_FRAC") == 0.45
    assert ct.calibrated_value("M3_COND_NUMBER_NONIDENT") == 107.0
    assert ct.calibrated_value("M5_THRESHOLD_SHARPNESS") == 5.7
    assert ct.calibrated_value("M5_THRESHOLD_LAG_RATIO") == 0.97
    assert ct.calibrated_value("M5_ANOMALY_FDR_ALPHA") == 0.05    # unchanged


def test_first_pass_values_are_the_previous_hand_values():
    assert ct.first_pass_value("M1_LAG_SLOPE_FRAC") == 0.30
    assert ct.first_pass_value("M3_COND_NUMBER_NONIDENT") == 1000.0
    assert ct.first_pass_value("M5_THRESHOLD_SHARPNESS") == 3.0
    assert ct.first_pass_value("M5_THRESHOLD_LAG_RATIO") == 0.80
    assert ct.first_pass_value("M5_ANOMALY_FDR_ALPHA") == 0.05


def test_calibrated_differs_from_first_pass_where_a_change_was_applied():
    """The three consequential thresholds actually MOVED (not a no-op apply)."""
    for name in ("M1_LAG_SLOPE_FRAC", "M3_COND_NUMBER_NONIDENT",
                 "M5_THRESHOLD_SHARPNESS", "M5_THRESHOLD_LAG_RATIO"):
        assert ct.calibrated_value(name) != ct.first_pass_value(name), name
    # anomaly alpha is the deliberate no-op (Service C confirmed control holds)
    assert ct.calibrated_value("M5_ANOMALY_FDR_ALPHA") == ct.first_pass_value(
        "M5_ANOMALY_FDR_ALPHA")


def test_accessors_never_raise_on_miss():
    assert ct.calibrated_value("NOPE", default=1.23) == 1.23
    assert ct.first_pass_value("NOPE", default=4.56) == 4.56
    assert ct.threshold_record("NOPE") == {}


# --------------------- M1/M3/M5 consume the calibrated value ---------------- #
def test_m1_reads_calibrated_lag_slope():
    import m1_ingest
    assert m1_ingest.LAG_SLOPE_FRAC == ct.calibrated_value("M1_LAG_SLOPE_FRAC")
    assert m1_ingest.LAG_SLOPE_FRAC == 0.45                      # not the first-pass 0.30


def test_m3_reads_calibrated_cond_number():
    import m3_select
    assert m3_select.COND_NUMBER_NONIDENT == ct.calibrated_value("M3_COND_NUMBER_NONIDENT")
    assert m3_select.COND_NUMBER_NONIDENT == 107.0              # not the first-pass 1000.0


def test_m5_reads_calibrated_shape_thresholds():
    import m5_classify
    assert m5_classify.THRESHOLD_SHARPNESS == ct.calibrated_value("M5_THRESHOLD_SHARPNESS")
    assert m5_classify.THRESHOLD_LAG_RATIO == ct.calibrated_value("M5_THRESHOLD_LAG_RATIO")
    assert m5_classify.ANOMALY_FDR_ALPHA == ct.calibrated_value("M5_ANOMALY_FDR_ALPHA")
    assert m5_classify.THRESHOLD_SHARPNESS == 5.7               # not the first-pass 3.0


# --------------------- engine_build_id folds the thresholds axis ------------ #
def test_engine_build_id_includes_thresholds_version():
    import m7_assemble as m7
    comps = m7.build_components(corpus_version="abc")
    assert comps.get("thresholds_version") == "thresholds-1.0"
    # flipping the thresholds axis MUST change the build id (invalidation, §7)
    base_id = m7.engine_build_id(comps)
    mutated = dict(comps)
    mutated["thresholds_version"] = "thresholds-0.0-first-pass"
    assert m7.engine_build_id(mutated) != base_id


# -------------------------------- runner ----------------------------------- #
def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}"); failed += 1
        except Exception as e:                       # noqa
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
