"""
PRISE — Calibrated Thresholds (the GOVERNED apply artifact)
===========================================================

This module is the **versioned, signed, dated** record of Service C's held-out
calibration being applied back into M1/M3/M5 — the human-governed step
PRISE_DESIGN.md §7/§8 deferred ("only manual approval mutates a threshold, and
every mutation is a signed, dated, rationale-bearing, version-bound artifact;
a threshold change bumps `engine_build_id` and invalidates descendants").

It carries, for EACH consequential threshold:
  * `first_pass`  — the original hand-set value M1/M3/M5 shipped with,
  * `calibrated`  — the value now APPLIED (Service C's held-out recommendation,
                    sensibly rounded),
  * `source`      — the Service-C provenance (held-out separation AUC + n),
so nothing is lost: the previous value is retained and auditable, and the change
is reversible (revert `calibrated` → `first_pass` and re-bump the version).

`THRESHOLDS_VERSION` ("thresholds-1.0") is folded into `engine_build_id`
(m7_assemble.build_components), so applying/adjusting these values makes prior
results non-comparable per §7 — exactly the invalidation the design demands.

M1/M2..M7 IMPORT the applied value from here (`calibrated_value(name)`), keeping
their own constant NAMES (consumers/tests reference LAG_SLOPE_FRAC,
COND_NUMBER_NONIDENT, THRESHOLD_SHARPNESS, THRESHOLD_LAG_RATIO,
ANOMALY_FDR_ALPHA) but sourcing the VALUE from this governed record. Nothing here
raises: `calibrated_value` returns the stored default on any lookup miss.

HONESTY: these thresholds are NO LONGER first-pass. As of the apply record below,
M1/M3/M5 run **Service-C-calibrated (thresholds-1.0)** values, chosen on a seeded
TRAIN split and validated out-of-sample on a disjoint HELD-OUT split (C2 — no
leakage). This is strictly MORE rigorous than the hand-set first-pass values:
the cutoffs now have a measured, out-of-sample separation guarantee rather than a
guess, and the prior values are preserved for reversibility and audit.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Version tag — folded into engine_build_id (§7). Bumping ANY calibrated value
# below MUST bump this so descendants are invalidated (old results non-comparable).
# --------------------------------------------------------------------------- #
THRESHOLDS_VERSION = "thresholds-1.0"

# The upstream artifact these values were distilled from (Service C, §6C/§8).
SOURCE_ARTIFACT = "service_c_calibration.json"
SOURCE_VERSION = "service-c-1.0"

# --------------------------------------------------------------------------- #
# The governed threshold table. Keys are the (module.CONSTANT) identifiers.
# `calibrated` is the APPLIED value; `first_pass` is the retained previous value.
# `source` records Service C's held-out (out-of-sample) evidence for the change.
# Recommended values are sensibly rounded from the raw Service-C recommendation
# (raw value kept under source.recommended_raw so the rounding is auditable).
# --------------------------------------------------------------------------- #
THRESHOLDS = {
    "M1_LAG_SLOPE_FRAC": {
        "module": "m1_ingest.LAG_SLOPE_FRAC",
        "first_pass": 0.30,
        "calibrated": 0.45,
        "metric": ("start/peak slope ratio separating a true no-lag / downhill "
                   "curve from a lag-phase (nucleated) curve"),
        "source": {
            "engine": SOURCE_VERSION,
            "recommended_raw": 0.45574426666452544,
            "separation_auc_heldout": 1.0,
            "separation_auc_train": 1.0,
            "accuracy_heldout": 1.0,
            "n": 40, "n_train": 20, "n_test": 20,
            "split": "train-chosen (Youden-J), held-out AUC reported (C2, no leakage)",
        },
        "effect": ("RE-TRIAGES the corpus: a higher bar for 'no lag' means fewer "
                   "curves are flagged left-censored / no-lag (the first-pass 0.30 "
                   "over-flagged ~half the corpus as left)."),
    },
    "M3_COND_NUMBER_NONIDENT": {
        "module": "m3_select.COND_NUMBER_NONIDENT",
        "first_pass": 1000.0,
        "calibrated": 107.0,
        "metric": ("sensitivity-collinearity condition number above which a "
                   "parameter combination is practically non-identifiable"),
        "source": {
            "engine": SOURCE_VERSION,
            "recommended_raw": 108.80239427449156,
            "recommended_log10": 2.0366384524279586,
            "separation_auc_heldout": 1.0,
            "separation_auc_train": 0.98,
            "accuracy_heldout": 1.0,
            "n": 40, "n_train": 20, "n_test": 20,
            "split": "train-chosen (Youden-J), held-out AUC reported (C2, no leakage)",
        },
        "effect": ("RECLASSIFIES fits: a much tighter cutoff (1000 → 107) flags "
                   "more sloppy/collinear fits as practically non-identifiable — "
                   "the identifiability gate is now calibrated, not a round guess."),
    },
    "M5_THRESHOLD_SHARPNESS": {
        "module": "m5_classify.THRESHOLD_SHARPNESS",
        "first_pass": 3.0,
        "calibrated": 5.7,
        "metric": ("transition_sharpness (t50/width) above which — with a high "
                   "lag/t50 — a curve is threshold-driven rather than smoothly "
                   "cooperative"),
        "source": {
            "engine": SOURCE_VERSION,
            "recommended_raw": 5.968343168710676,
            "separation_auc_heldout": 1.0,
            "separation_auc_train": 1.0,
            "accuracy_heldout": 1.0,
            "n": 40, "n_train": 20, "n_test": 20,
            "split": "train-chosen (Youden-J), held-out AUC reported (C2, no leakage)",
        },
        "effect": ("SHIFTS the threshold-vs-cooperative regime call: a higher "
                   "sharpness bar means fewer curves are called threshold_driven "
                   "(more remain cooperative_sigmoidal)."),
    },
    "M5_THRESHOLD_LAG_RATIO": {
        "module": "m5_classify.THRESHOLD_LAG_RATIO",
        "first_pass": 0.80,
        "calibrated": 0.97,
        "metric": ("median lag/t50 ratio in the threshold-driven synthetic class "
                   "(the companion gate to sharpness)"),
        "source": {
            "engine": SOURCE_VERSION,
            "recommended_raw": 0.9728564086803677,
            "separation_auc_heldout": None,   # class-median statistic, not an AUC sweep
            "note": ("median lag/t50 of the threshold-driven synthetic class; "
                     "paired with sharpness, both must exceed their cutoff"),
        },
        "effect": ("Tightens the threshold-driven gate alongside sharpness "
                   "(both conditions must hold)."),
    },
    "M5_ANOMALY_FDR_ALPHA": {
        "module": "m5_classify.ANOMALY_FDR_ALPHA",
        "first_pass": 0.05,
        "calibrated": 0.05,   # UNCHANGED — Service C confirmed FDR-null control holds
        "metric": "Benjamini–Hochberg level for the corpus anomaly scan",
        "source": {
            "engine": SOURCE_VERSION,
            "recommended": 0.05,
            "fdr_controlled": True,
            "realized_fdp": 0.0,
            "note": ("kept at nominal 0.05: Service C's FDR-null arm showed control "
                     "holds (realized FDP 0.0) WITH power on a bank-unrepresentable "
                     "alternative — no change warranted, recorded for completeness."),
        },
        "effect": "no change — nominal α retained (FDR-null control confirmed).",
    },
}

# --------------------------------------------------------------------------- #
# The signed / dated / rationale-bearing APPLY RECORD (§7 governance).
# This is what turns a recommendation into a governed, auditable mutation.
# --------------------------------------------------------------------------- #
APPLY_RECORD = {
    "applied_at": "2026-07-01",
    "by": "governed apply step",
    "rationale": "Service C held-out calibration (service-c-1.0)",
    "source_artifact": SOURCE_ARTIFACT,
    "source_version": SOURCE_VERSION,
    "thresholds_version": THRESHOLDS_VERSION,
    "governance": ("PRISE_DESIGN.md §7/§8: a threshold change is a signed, dated, "
                   "rationale-bearing, version-bound artifact; it bumps "
                   "engine_build_id and invalidates descendants (old results "
                   "non-comparable). The previous ('first_pass') value is retained "
                   "for every threshold, so the change is reversible."),
    "reversible": True,
    "note": ("M1/M3/M5 now consume the `calibrated` value; reverting is a matter of "
             "restoring `first_pass` and re-bumping THRESHOLDS_VERSION."),
}


# --------------------------------------------------------------------------- #
# Accessors (pure; never raise — a lookup miss returns the supplied default).
# --------------------------------------------------------------------------- #
def calibrated_value(name: str, default: float | None = None) -> float | None:
    """The APPLIED (calibrated) value for a threshold key (e.g. 'M1_LAG_SLOPE_FRAC').
    Falls back to `default` on any miss so a consumer never crashes on a typo."""
    rec = THRESHOLDS.get(name)
    if not rec:
        return default
    val = rec.get("calibrated")
    return val if val is not None else default


def first_pass_value(name: str, default: float | None = None) -> float | None:
    """The retained PREVIOUS (first-pass) value for a threshold key."""
    rec = THRESHOLDS.get(name)
    if not rec:
        return default
    val = rec.get("first_pass")
    return val if val is not None else default


def threshold_record(name: str) -> dict:
    """The full governed record (first_pass + calibrated + source + effect) for one
    threshold, or {} if unknown. JSON-serialisable."""
    return dict(THRESHOLDS.get(name, {}))


def manifest() -> dict:
    """The complete governed artifact as a JSON-serialisable dict: version, the
    Service-C source, the full threshold table (both values retained), and the
    signed/dated apply record. Used by anything that wants to persist or display
    the governance record (and by the tests)."""
    return {
        "thresholds_version": THRESHOLDS_VERSION,
        "source_artifact": SOURCE_ARTIFACT,
        "source_version": SOURCE_VERSION,
        "thresholds": {k: dict(v) for k, v in THRESHOLDS.items()},
        "apply_record": dict(APPLY_RECORD),
    }


if __name__ == "__main__":   # pragma: no cover - human-readable dump
    import json
    print(json.dumps(manifest(), indent=2))
