"""
PRISE — Module M7: Output Assembler (graceful degradation built in)
===================================================================

The last Phase-1 module. It does NO new inference: it JOINS the per-curve /
per-series / per-protein artifacts from M1–M6 into one auditable, sortable
`ProteinAnalysisResult` per **analysis unit** (= one curve / `series_id`,
carrying protein_id+uniprot_id so results group by protein) — plus a separate,
FDR-aware, definition-homogeneous **corpus rollup** (PRISE_DESIGN.md §3-M7, §7).

Three honesty features shape this module:

  1. **A sortable `information_yield` ladder** (mechanistic > scaling > descriptive
     > signal_only > uninformative), derived per curve from what the upstream
     modules actually licensed — so a reader can rank curves by how much was
     genuinely learnable, not by how confident a number looks.

  2. **The degenerate-case contract** (the common case for bundled data): for the
     two lowest tiers we emit a COMPACT result — one ranked M9 hook + ≤3
     impact-ranked caveats — NOT the full refusing payload, and NOT a caveat
     avalanche. For the three informative tiers we emit the full schema.

  3. **The comparability key** — every result carries `engine_build_id`, a short
     deterministic hash over a SORTED dict of every version axis (the scoring
     schemas, taxonomy/regime registry, anchors, reference scale, the two ETL
     corpus digests, this assembler's own version, and an env/harness tag).
     Comparability is gated on EQUAL `engine_build_id` (`comparable()`), never on
     eyeballing independent version integers (§7, R2).

`m9_hook` is a clearly-flagged Phase-2 STUB: a rule table, not an EIG / optimal-
design computation (§5-M9 explicitly licenses the shippable rule-table variant).

Everything is pure and JSON-serialisable; nothing here raises on bad input.

Usage:
    python engine/m7_assemble.py                 # full corpus -> protein_analysis.jsonl + m7_rollup.json
    python engine/m7_assemble.py --limit 200
    python engine/m7_assemble.py --input data/processed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Version axes pulled by IMPORTING from the source-of-truth modules where we can
# (so a version bump in any module automatically changes engine_build_id), with
# constant fallbacks if the import fails (e.g. unit tests run without the modules
# on the path) — flagged so the provenance of every component is auditable.
# --------------------------------------------------------------------------- #
M7_VERSION = "m7-assembler-1.0"

# §7 names six version axes (scoring schema, taxonomy, harness, anchor, corpus,
# env). To honour "scoring schema = every per-module output contract", we pull a
# version tag from EACH of M1–M6 (not just M4–M6): M1/M2/M3 carry purely-additive
# module-level VERSION constants so a bump in any upstream module flows into
# engine_build_id. Constant fallbacks keep the assembler importable in isolation
# (e.g. unit tests run without the full module graph on the path).
try:                                                   # pragma: no cover - import glue
    from m1_ingest import M1_VERSION as _M1_VERSION_IMP
    from m2_fit import M2_VERSION as _M2_VERSION_IMP
    from m3_select import M3_VERSION as _M3_VERSION_IMP
    from m4_features import DEFINITION_CONTRACT as _M4_DEFS
    from m5_classify import REGIME_REGISTRY as _M5_REG
    from m6_propensity import (
        ANCHOR_VERSION as _M6_ANCHOR,
        REFERENCE_SCALES as _M6_REFSCALE,
    )
    from calibrated_thresholds import THRESHOLDS_VERSION as _THRESHOLDS_VERSION_IMP
    from join_policy import JOIN_POLICY_VERSION as _JOIN_POLICY_VERSION_IMP
    from bootstrap_policy import BOOTSTRAP_POLICY_VERSION as _EST_VERSION_IMP
    _M1_VERSION = _M1_VERSION_IMP
    _M2_VERSION = _M2_VERSION_IMP
    _M3_VERSION = _M3_VERSION_IMP
    _M4_DEFS_VERSION = _M4_DEFS["version"]
    _M5_REG_VERSION = _M5_REG["version"]
    _M6_ANCHOR_VERSION = _M6_ANCHOR
    _M6_REFSCALE_VERSION = _M6_REFSCALE["version"]
    _THRESHOLDS_VERSION = _THRESHOLDS_VERSION_IMP
    _JOIN_POLICY_VERSION = _JOIN_POLICY_VERSION_IMP
    _ESTIMATION_POLICY_VERSION = _EST_VERSION_IMP
    _VERSIONS_IMPORTED = True
except Exception:                                      # pragma: no cover - fallback path
    _M1_VERSION = "m1-triage-1.0"
    _M2_VERSION = "m2-fitbank-1.0"
    _M3_VERSION = "m3-select-1.0"
    _M4_DEFS_VERSION = "m4-defs-1.0"
    _M5_REG_VERSION = "m5-regimes-1.0"
    _M6_ANCHOR_VERSION = "m6-anchor-1.0"
    _M6_REFSCALE_VERSION = "m6-refscale-1.0"
    _THRESHOLDS_VERSION = "thresholds-1.0"
    _JOIN_POLICY_VERSION = "join-policy-1.1"
    _ESTIMATION_POLICY_VERSION = "estimation-policy-1.0"
    _VERSIONS_IMPORTED = False

# The GOVERNED join policy accessor (join_policy.py). Imported separately from the
# version glue above because the assembler READS these values at join time rather
# than only hashing them: hard-coding the new behaviour would make `previous` a
# comment instead of a revert path. The fallback mirrors the shipped `applied`
# values so an isolated import cannot silently restore the pre-fix join.
try:                                                   # pragma: no cover - import glue
    from join_policy import applied_policy as _applied_policy
except Exception:                                      # pragma: no cover - fallback path
    def _applied_policy(name, default=None):
        return {"PROPENSITY_ASSAY_GATE": "construct_and_assay",
                "SEQUENCE_AXIS_ASSAY_GATE": "assay_matched_only",
                "M6_GROUP_KEY": ["uniprot", "construct", "assay"]}.get(name, default)

# The deterministic-inference harness identity (§7's distinct "harness" axis): the
# seeded-bootstrap policy (M3/M4 wild bootstrap, fixed RNG seed) + the TRF/least-
# squares optimisation policy. This is the reproducibility *recipe*, separate from
# `env_manifest` (the interpreter/OS tag) — a harness change can alter results
# even on an identical interpreter, so it MUST be its own comparability component.
HARNESS_VERSION = "harness-1.0"  # seeded-bootstrap + TRF least-squares policy

# ETL schema versions (the two corpus layers). Constants here mirror the manifests;
# the driver overrides them with the real manifest values + corpus digests at load.
_KINETICS_SCHEMA_DEFAULT = "cpad-kinetics-1.0"
_SEQUENCE_SCHEMA_DEFAULT = "cpad-sequence-1.1"

# --------------------------------------------------------------------------- #
# information_yield ladder — ORDERED highest -> lowest so results are sortable on
# a single ordinal key. The two lowest tiers trigger the compact contract.
# --------------------------------------------------------------------------- #
YIELD_LADDER = [
    "mechanistic",     # M5 licensed a mechanistic inference
    "scaling",         # a SIGNIFICANT γ (reliable AND CI excludes 0) for the series
    "descriptive",     # a resolved shape regime + ok features
    "signal_only",     # fittable but no resolved transition
    "uninformative",   # not fittable / no converged fit / reject
]
YIELD_RANK = {tier: i for i, tier in enumerate(YIELD_LADDER)}   # 0 = highest yield
_COMPACT_TIERS = {"signal_only", "uninformative"}

# Descriptive regimes that count as a "real shape" for the `descriptive` tier
# (everything except the no-transition / unusable buckets).
_REAL_SHAPE_REGIMES = {
    "cooperative_sigmoidal", "gradual_non_cooperative",
    "threshold_driven", "non_monotonic_settling",
}


# =========================================================================== #
# engine_build_id — the comparability key
# =========================================================================== #
def build_components(
    kinetics_schema_version: str = _KINETICS_SCHEMA_DEFAULT,
    sequence_schema_version: str = _SEQUENCE_SCHEMA_DEFAULT,
    corpus_version: str | None = None,
    env_manifest: str | None = None,
    harness_version: str = HARNESS_VERSION,
) -> dict:
    """The full, auditable component dict that backs `engine_build_id`.

    These are the SIX version axes §7 names — scoring schema, taxonomy, harness,
    anchor, corpus, env — expressed against PRISE's concrete module versions.
    Every per-module output contract (M1–M7) is folded in so a bump anywhere
    upstream changes the id. `harness_version` is a DISTINCT axis from
    `env_manifest`: the former is the seeded-inference recipe, the latter the
    interpreter/OS tag — a recipe change can alter results on an identical
    interpreter. `corpus_version` is first-class (R2): a digest tying the build
    to the exact frozen corpus bytes (all source shas from BOTH ETL manifests,
    see the driver). The dict is emitted ALONGSIDE the hash so a reader can see
    WHY two builds are (in)comparable instead of trusting an opaque id."""
    return {
        # scoring-schema axis (the per-module output contracts, M1 → M7)
        "m1_triage": _M1_VERSION,
        "m2_fitbank": _M2_VERSION,
        "m3_select": _M3_VERSION,
        "m4_definition_contract": _M4_DEFS_VERSION,
        "m5_regime_registry": _M5_REG_VERSION,           # taxonomy_version
        "m6_anchor_version": _M6_ANCHOR_VERSION,         # anchor_version
        "m6_reference_scale": _M6_REFSCALE_VERSION,
        "m7_assembler": M7_VERSION,
        # governed threshold-calibration axis (§7): M1/M3/M5 now run Service-C-
        # calibrated (thresholds-1.0) cutoffs, not first-pass. A threshold change
        # bumps THIS component, so old (first-pass) results become non-comparable.
        "thresholds_version": _THRESHOLDS_VERSION,
        # governed JOIN-POLICY axis (§7): the M6 grouping key and the M7
        # propensity/sequence-axis assay gates. This axis exists because EVERY other
        # component here is a version string or an ETL byte digest — none is a
        # function of the join code — so without it a join change would ship the SAME
        # engine_build_id with DIFFERENT numbers, and comparable() (which tests id
        # equality alone) would call two materially different corpora comparable.
        "join_policy_version": _JOIN_POLICY_VERSION,
        # governed ESTIMATION axis: M2 optimisation starts + M3 bootstrap
        # stream. Neither is a function of any other component here, so
        # without this axis a change to how the corpus is FITTED would ship
        # the same build id with different numbers.
        "estimation_policy_version": _ESTIMATION_POLICY_VERSION,
        # corpus axis (both ETL layers + the all-source byte digest)
        "etl_kinetics_schema": kinetics_schema_version,
        "etl_sequence_schema": sequence_schema_version,
        "corpus_version": corpus_version or "unknown-corpus",
        # harness axis (the deterministic-inference recipe — §7, distinct from env)
        "harness_version": harness_version,
        # env axis (the reproducibility environment tag, §7)
        "env_manifest": env_manifest or _default_env_manifest(),
    }


def environment_fingerprint() -> dict:
    """The software stack this build ran on, as structured facts.

    Kept separate from the STRING the build id folds (`_default_env_manifest`)
    so a reader can see what the tag is made of instead of parsing it."""
    out = {"python": "unknown", "numpy": None, "scipy": None,
           "blas": None, "blas_version": None,
           "platform": None, "machine": None}
    try:
        out["python"] = "%d.%d.%d" % (sys.version_info.major,
                                      sys.version_info.minor,
                                      sys.version_info.micro)
    except Exception:                                   # pragma: no cover
        pass
    try:
        import numpy as _np
        out["numpy"] = _np.__version__
        cfg = _np.__config__.show(mode="dicts")
        blas = (cfg.get("Build Dependencies") or {}).get("blas") or {}
        out["blas"] = blas.get("name")
        out["blas_version"] = blas.get("version")
    except Exception:                                   # pragma: no cover
        pass
    try:
        import scipy as _sp
        out["scipy"] = _sp.__version__
    except Exception:                                   # pragma: no cover
        pass
    try:
        out["platform"] = platform.system()
        out["machine"] = platform.machine()
    except Exception:                                   # pragma: no cover
        pass
    return out


def _default_env_manifest() -> str:
    """The §7 environment axis: a fingerprint of the SOFTWARE STACK that can
    change a floating-point result, not merely the interpreter minor version.

    WHY IT WIDENED. It used to be `python3.13` and nothing else, with the gap
    flagged as deferred. That is the same defect class this codebase has now
    corrected three times over: a comparability key that fails to capture
    something which moves numbers. Two builds on the same Python minor but a
    different BLAS can and do diverge in the last digits of a least-squares fit,
    and `comparable()` -- which tests build-id equality alone -- would have
    called them comparable. Folding the stack in makes that non-comparability
    detectable instead of silent.

    WHAT IS IN IT: python micro, numpy, scipy, BLAS vendor + version, OS and
    machine. All are properties of the INSTALLATION, so the id is stable across
    runs on one machine and differs across materially different ones.

    WHAT IS DELIBERATELY NOT IN IT: the live thread-count environment
    (OMP_NUM_THREADS and friends). Thread count genuinely can change BLAS
    reduction order and therefore the last digits -- but folding it in would make
    `engine_build_id` depend on how the shell happened to launch the process, so
    the same corpus rebuilt with a different `-j` would be declared
    non-comparable with itself. It is reported by `environment_fingerprint()` as
    an observed fact and is pinned by nothing; that residual sensitivity is the
    honest remainder of §7's pinned-container requirement, which still needs a
    container rather than a fingerprint."""
    fp = environment_fingerprint()
    return "|".join((
        "py" + str(fp.get("python")),
        "np" + str(fp.get("numpy")),
        "sp" + str(fp.get("scipy")),
        "blas:" + str(fp.get("blas")) + "-" + str(fp.get("blas_version")),
        str(fp.get("platform")) + "/" + str(fp.get("machine")),
    ))


def engine_build_id(components: dict) -> str:
    """Short deterministic hash over a SORTED component dict. Sorting the keys
    makes the id independent of dict insertion order; sha256[:12] is collision-
    safe at our handful-of-builds scale and short enough to eyeball."""
    canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def comparable(build_a: str, build_b: str) -> bool:
    """Hard comparability gate (§7, R2): two results may be compared iff their
    engine_build_id is EQUAL. No compatibility table is declared in Phase 1, so
    equality is the whole rule — deliberately conservative."""
    return bool(build_a) and build_a == build_b


# =========================================================================== #
# γ significance — the scaling-tier gate
# =========================================================================== #
def gamma_significant(gamma_rec: dict | None) -> bool:
    """True iff a γ scaling exponent is actually RESOLVED for this series.

    The honesty bug this fixes: M4's `gamma_reliable` only certifies that the
    *fit/CI machinery* ran cleanly (enough anchors, a CI was formed, no
    divergence) — it does NOT certify that γ differs from zero. ~77% of
    reliable-γ curves have a 95% CI that spans 0, i.e. no scaling was resolved.
    A curve earns the `scaling` tier ONLY when (a) gamma_reliable is True AND
    (b) a CI exists AND (c) the CI excludes 0 (both bounds same sign / strictly
    away from 0). γ is a sign-meaningful exponent, so a CI straddling 0 is "no
    detectable concentration dependence", not a scaling claim."""
    if not gamma_rec:
        return False
    reg = (gamma_rec.get("gamma_regression", {}) or {})
    if reg.get("gamma_reliable") is not True:
        return False
    ci = reg.get("gamma_ci")
    if not (isinstance(ci, (list, tuple)) and len(ci) == 2):
        return False
    lo, hi = ci[0], ci[1]
    if lo is None or hi is None:
        return False
    # CI excludes 0 iff both endpoints are strictly the same side of 0.
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)


# =========================================================================== #
# information_yield ladder
# =========================================================================== #
def information_yield(
    m1: dict | None,
    feat: dict | None,
    regime_rec: dict | None,
    gamma_rec: dict | None,
) -> str:
    """Derive the sortable yield tier for one curve from the upstream licences.

    The ladder is evaluated top-down; the FIRST satisfied rung wins (so a curve
    that is both γ-significant and mechanistically licensed lands on the higher
    `mechanistic` rung). Missing inputs degrade gracefully — a curve with no
    fittable triage falls straight to `uninformative` rather than erroring."""
    fittability = (m1 or {}).get("fittability_class")
    feat_status = (feat or {}).get("status")
    mech = (regime_rec or {}).get("mechanistic", {}) or {}
    regime = ((regime_rec or {}).get("descriptive_regime", {}) or {}).get("regime")

    # uninformative: not representable at all — no fit / reject / rejected triage.
    if fittability in (None, "reject", "unfittable") or feat_status == "no_fit":
        return "uninformative"

    # mechanistic: M5 actually licensed a mechanistic inference (expected: 0).
    if mech.get("mechanistic_inference_licensed") is True:
        return "mechanistic"

    # scaling: a SIGNIFICANT γ (reliable AND CI excludes 0) for this series. A
    # reliable-but-CI-includes-0 γ does NOT promote — it falls to `descriptive`
    # and earns the "CI includes zero" caveat (no scaling exponent resolved).
    if gamma_significant(gamma_rec):
        return "scaling"

    # descriptive: a real shape regime AND M4 features status == "ok".
    if regime in _REAL_SHAPE_REGIMES and feat_status == "ok":
        return "descriptive"

    # signal_only: fittable but no resolved transition.
    if regime == "no_detectable_aggregation" or feat_status == "flat_no_transition":
        return "signal_only"

    # Fittable, has a fit, but didn't clear any informative rung (e.g. suspect
    # triage with an unusable/anomalous regime) — honestly signal_only, not a
    # silent promotion to descriptive.
    return "signal_only"


# =========================================================================== #
# m9_hook — Phase-2 recommender STUB (rule table, NOT an EIG computation)
# =========================================================================== #
_M9_STUB_FLAG = ("phase2_recommender_stub: rule-based, not an EIG/optimal-design "
                 "computation (PRISE_DESIGN.md §5-M9 licenses the shippable rule "
                 "table; the optimization is the research deliverable). Returns ONE "
                 "suggestion (the single highest-impact firing rule) and is NOT "
                 "multi-objective — it does not trade off cost/time/information or "
                 "rank a frontier; see m9_hooks_ranked for the (still rule-based) "
                 "top-few view")


def _m9_firing_rules(
    m1: dict | None,
    feat: dict | None,
    regime_rec: dict | None,
    condition_vector: dict | None,
    has_concentration_series: bool,
) -> list[tuple[str, str]]:
    """All next-experiment rules that FIRE for this curve, in IMPACT order
    (highest-leverage first). Returns (rule_name, suggestion) pairs. The default
    'add replicates' rule always fires last so the list is never empty. Splitting
    this out lets us expose both the single primary hook AND a ranked top-few view
    without re-deriving the rules (item 11)."""
    cv = condition_vector or {}
    prov = cv.get("field_provenance", {}) or {}
    anomaly = (regime_rec or {}).get("anomaly", {}) or {}
    t50_status = (feat or {}).get("t50_status")
    fired: list[tuple[str, str]] = []

    # Rule 1 (highest impact): no concentration series -> can't even attempt γ/mechanism.
    if not has_concentration_series:
        fired.append(("no_concentration_series",
                      "add a concentration decade (≥3 concentrations spanning ~1 "
                      "decade) to enable γ-scaling and a mechanistic call"))
    # Rule 2: agitation or seeding unknown -> the mechanistic branch is undetermined.
    if (prov.get("agitation") == "unknown" or prov.get("seeded") == "unknown"
            or cv.get("agitation") is None or cv.get("seeded") is None):
        fired.append(("agitation_or_seeding_unknown",
                      "record agitation & seeding (quiescent vs shaken, seeded vs "
                      "de novo) to license a mechanistic-branch call"))
    # Rule 3: t50 censored -> the primary anchor is a bound, not a point.
    if t50_status in ("biased_early", "lower_bound", "right_censored",
                      "biased_late") or (m1 or {}).get("censoring_class") in (
            "left", "right"):
        fired.append(("t50_censored",
                      "extend the observation window / sample the lag phase to "
                      "de-censor t50 (turn the bound into a point estimate)"))
    # Rule 4: a material misfit anomaly -> the model bank cannot represent this curve.
    if anomaly.get("material_misfit") is True:
        fired.append(("anomaly_material_misfit",
                      "model bank cannot represent this curve — flag for human "
                      "regime review (candidate new descriptive regime)"))
    # Rule 5 (fallback, always fires last): quantify between-run scatter.
    fired.append(("default_add_replicates",
                  "add replicates to quantify between-run scatter "
                  "(the stochastic-nucleation signal)"))
    return fired


def m9_hook(
    m1: dict | None,
    feat: dict | None,
    regime_rec: dict | None,
    gamma_rec: dict | None,
    condition_vector: dict | None,
    has_concentration_series: bool,
) -> dict:
    """ONE highest-impact next-experiment suggestion from an honest rule table.

    The rules are checked in IMPACT order — the highest-leverage missing piece
    wins. Returns the primary suggestion + the rule that fired, plus a ranked
    top-≤3 view (`m9_hooks_ranked`, item 11) so a reader sees the runner-up
    suggestions too. Always flagged as a stub: it is single-objective, not an
    EIG/optimal-design computation, so no one mistakes it for optimal design."""
    fired = _m9_firing_rules(m1, feat, regime_rec, condition_vector,
                             has_concentration_series)
    primary_rule, primary_suggestion = fired[0]
    ranked = [{"rule_fired": r, "suggestion": s} for r, s in fired[:3]]
    return {
        "suggestion": primary_suggestion,
        "rule_fired": primary_rule,
        "m9_hooks_ranked": ranked,   # ≤3, impact-ordered (still rule-based, not EIG)
        "stub": _M9_STUB_FLAG,
    }


# =========================================================================== #
# caveats — impact-ranked, de-duplicated, capped (anti-"caveat avalanche")
# =========================================================================== #
# Each caveat carries an integer IMPACT (lower = more important) so the cap keeps
# the most decisive refusals, not the first ones encountered.
def _amplitude_features_present(feat: dict | None) -> bool:
    """True iff the emitted feature vector actually carries amplitude-type features
    (plateau / dynamic_range / max_rate) — the ones that are instrument-local when
    the signal basis is not normalized. Used to make the portability caveat
    cap-exempt only when it is genuinely load-bearing for THIS result."""
    feats = (feat or {}).get("features", {}) or {}
    return any(feats.get(k) is not None
               for k in ("plateau", "dynamic_range", "max_rate",
                         "max_rate_normalized", "amplitude"))


def collect_caveats(
    m1: dict | None,
    feat: dict | None,
    regime_rec: dict | None,
    gamma_rec: dict | None,
    condition_vector: dict | None,
    has_concentration_series: bool,
    cap: int = 3,
) -> list[str]:
    """Gather the salient refusals/flags, impact-rank, de-duplicate, and cap.

    This is the anti-"caveat avalanche" requirement: a refusing curve has dozens
    of true caveats; emitting all of them is noise. We keep only the few most
    decisive, in impact order. A small set of caveats are EXEMPT from the cap
    because dropping them would mislead (see `_forced` below): a non-portable
    amplitude basis when amplitude features are actually emitted, and a γ that is
    reliable but non-significant on a curve being read as a scaling candidate."""
    cv = condition_vector or {}
    prov = cv.get("field_provenance", {}) or {}
    mech = (regime_rec or {}).get("mechanistic", {}) or {}
    t50_status = (feat or {}).get("t50_status")
    candidates: list[tuple[int, str]] = []   # (impact, text)
    forced: list[str] = []                   # cap-exempt; always emitted, in order

    # (impact 1) mechanism not licensed + top reason — the headline refusal.
    if mech.get("mechanistic_inference_licensed") is False:
        reasons = mech.get("gates_failed") or []
        top = reasons[0] if reasons else "single-curve under-determination"
        candidates.append((1, f"mechanistic inference not licensed ({top})"))

    # (impact 2) t50 censoring status — biases the primary anchor.
    if t50_status and t50_status != "point":
        candidates.append((2, f"t50 is censored/biased ({t50_status}); the primary "
                              f"anchor is a bound, not a point"))
    elif (m1 or {}).get("censoring_class") in ("left", "right"):
        candidates.append((2, f"curve is {(m1 or {}).get('censoring_class')}-censored; "
                              f"t50/lag may be biased"))

    # (impact 3) γ unreliable / absent / reliable-but-not-significant — no scaling.
    if not has_concentration_series:
        candidates.append((3, "no concentration series — γ scaling exponent is "
                              "unconstrained (no mechanistic scaling check)"))
    elif gamma_rec:
        reg = (gamma_rec.get("gamma_regression", {}) or {})
        if reg.get("gamma_reliable") is False:
            why = (reg.get("reliability_reasons") or ["too few uncensored anchors"])[0]
            candidates.append((3, f"γ regression unreliable ({why})"))
        elif not gamma_significant(gamma_rec):
            # reliable-but-non-significant: the headline honesty caveat for the
            # ~77% of reliable γ whose CI spans 0 — they are demoted off `scaling`.
            candidates.append((3, "γ estimated but 95% CI includes zero — "
                                  "no scaling exponent resolved"))

    # (impact 4) non-portable amplitude basis — signal not comparable across assays.
    # If amplitude features are actually emitted on this result, the warning is
    # load-bearing and must NOT be evicted by the cap, so it is FORCED.
    signal_basis = (feat or {}).get("signal_basis") or (m1 or {}).get("signal_basis")
    if signal_basis and signal_basis != "normalized":
        text = (f"signal basis '{signal_basis}' is instrument-local; "
                f"amplitude features are non-portable")
        if _amplitude_features_present(feat):
            forced.append(text)
        else:
            candidates.append((4, text))

    # (impact 5) assay mass-proportionality unknown — ThT need not track mass.
    if cv.get("assay_reports_mass") is False:
        candidates.append((5, "assay is not known to report aggregate mass "
                              "proportionally (signal↔mass mapping unverified)"))

    # (impact 6) high digitization uncertainty — graph-digitized literature curve.
    if (feat or {}).get("digitization_uncertainty") or (m1 or {}).get(
            "digitization_uncertainty") or cv.get("digitization_uncertainty"):
        candidates.append((6, "graph-digitized literature curve: elevated "
                              "digitization uncertainty on every point"))

    # (impact 7) poor fit guard — a descriptive shape read off a low-R² fit is
    # shaky; we caveat (not demote) so the reader discounts it accordingly.
    r2 = ((regime_rec or {}).get("confidence", {}) or {}).get("r2")
    if r2 is None:
        r2 = ((regime_rec or {}).get("anomaly", {}) or {}).get("r2")
    if isinstance(r2, (int, float)) and r2 < 0.8:
        candidates.append((7, f"poor_fit (R²={r2:.2f}); descriptive shape read off "
                              f"a low-quality fit"))

    # de-duplicate by text (keep the lowest-impact number seen), then sort + cap.
    best: dict[str, int] = {}
    for impact, text in candidates:
        if text not in best or impact < best[text]:
            best[text] = impact
    ranked = sorted(best.items(), key=lambda kv: (kv[1], kv[0]))
    capped = [text for text, _ in ranked[:cap]]
    # forced caveats are prepended and de-duplicated against the capped list so a
    # load-bearing portability/γ warning is never silently dropped by the cap.
    out: list[str] = []
    for text in forced + capped:
        if text not in out:
            out.append(text)
    return out


# =========================================================================== #
# per-curve assembly
# =========================================================================== #
def _model_selection(m3_rec: dict | None, fit_rec: dict | None) -> dict:
    """Model selection block: M3 (bootstrap selection frequencies + stability +
    identifiability) when this curve was in the expensive M3 sample, ELSE the M2
    point selection (best_by_aicc) with an explicit no-bootstrap flag — never
    silently presenting a point pick as if it carried selection uncertainty."""
    if m3_rec:
        sel = m3_rec.get("selection", {}) or {}
        return {
            "source": "m3_bootstrap",
            "best_point": m3_rec.get("best_point"),
            "best_descriptive": m3_rec.get("best_descriptive"),
            "best_mechanistic": m3_rec.get("best_mechanistic"),
            "selection_frequencies": sel.get("selection_frequencies"),
            "selection_stability": sel.get("selection_stability"),
            "t50_predictive_interval": sel.get("t50_predictive_interval"),
            "t50_multimodal": sel.get("t50_multimodal"),
            "identifiability": m3_rec.get("identifiability_of_best"),
            "ci_reliable": m3_rec.get("ci_reliable"),
        }
    if fit_rec:
        return {
            "source": "m2_point_aicc",
            "best_point": fit_rec.get("best_by_aicc"),
            "flag": "point_selection_no_bootstrap (M3 not run for this curve)",
            "selection_frequencies": None,
            "selection_stability": None,
            "identifiability": None,
        }
    return {"source": "absent",
            "flag": "no M2 fit and no M3 record for this curve"}


def _surface_from_gamma(gamma_rec: dict | None) -> dict:
    """γ surface (gamma_global / gamma_regression / disagreement) for this curve's
    concentration series, or an absent-flag when no series exists."""
    if not gamma_rec:
        return {"available": False,
                "reason": "no concentration series for this curve "
                          "(γ needs ≥3 concentrations)"}
    return {
        "available": True,
        "concentration_series_id": gamma_rec.get("concentration_series_id"),
        "gamma_regression": gamma_rec.get("gamma_regression"),
        "gamma_global": gamma_rec.get("gamma_global"),
        "gamma_disagreement": gamma_rec.get("disagreement"),
        "n_member_curves": gamma_rec.get("n_member_curves"),
    }


def _propensity_block(
    prop_rec: dict | None,
    uniprot: str | None,
    want_construct: str | None = None,
    want_assay: str | None = None,
) -> dict:
    """Propensity (surface / intrinsic / cohort) + sequence_axis joined by UniProt
    on the FULL (uniprot, assay, construct) triple, absent-flagged when there is no
    construct-matched M6 record for this protein (most curves DO have one; mutants
    without a portable t50 do not).

    HONESTY FIX (item 4): the previous driver fell back to a uniprot-only record on
    a triple miss, silently attaching a DIFFERENT construct's (usually Wild-Type's)
    propensity to ~164 mutant curves. We no longer attach a non-construct-matched
    record as if it were this curve's: a mismatched record yields available=False
    with an explicit reason. Whatever the join, we ALWAYS surface `construct_match`,
    `assay_match`, `prop_construct_id`, and `prop_assay` so the join is auditable
    from the result.

    ASSAY GATE (join-policy-1.0). The construct fix above left the ASSAY axis open:
    the same uniprot-only fallback could attach a record computed for a different
    assay, and this function gated on construct alone while asserting in comment
    that the triple key guaranteed assay equality — false on the fallback path,
    with `want_assay` accepted and never read. Measured before the fix: 96 of the
    1,194 full-schema series resolved via the fallback, 14 of those to a different
    assay, and 9 of those were published as `available: true, construct_match:
    true`. m6_propensity attaches a caveat to every record stating that
    intrinsic/cohort are matched on ASSAY ENDPOINT ONLY — so assay-matching is the
    single comparability guarantee the record makes about itself, and a cross-assay
    join voids it, leaving `condition_match: "assay_only"` a false statement in the
    payload. ASSAY IS TESTED FIRST because it is the stronger claim: a record that
    fails both should report the assay failure."""
    gate = _applied_policy("PROPENSITY_ASSAY_GATE", "construct_and_assay")
    if not prop_rec:
        return {
            "available": False,
            "uniprot_id": uniprot,
            "construct_match": False,
            "assay_match": False,
            "prop_construct_id": None,
            "prop_assay": None,
            "reason": "no construct-matched M6 record for this protein×construct×assay "
                      "(e.g. censored-only t50, or no sequence/kinetics join)",
        }
    prop_cv = prop_rec.get("condition_vector", {}) or {}
    prop_construct = prop_cv.get("construct_id")
    prop_assay = prop_rec.get("assay")
    # Both axes the uniprot-only fallback can violate. `want_* is None` means the
    # curve declares no value on that axis, which cannot be a mismatch.
    construct_match = (want_construct is None) or (prop_construct == want_construct)
    assay_match = (want_assay is None) or (prop_assay == want_assay)
    if gate == "construct_and_assay" and not assay_match:
        # A real M6 record EXISTS for this uniprot, but it was computed for another
        # assay. Refuse it: intrinsic/cohort are anchored in an assay-matched
        # stratum, so attaching it would rank this curve against a pool containing
        # no member measured by its own assay.
        return {
            "available": False,
            "uniprot_id": prop_rec.get("uniprot_id"),
            "construct_match": construct_match,
            "assay_match": False,
            "prop_construct_id": prop_construct,
            "prop_assay": prop_assay,
            "reason": "no assay-matched M6 record — the only M6 record for this "
                      f"protein×construct is assay '{prop_assay}', not "
                      f"'{want_assay}' (refusing to attach a different assay's "
                      "propensity: intrinsic/cohort are anchored in an "
                      "assay-matched stratum, so a cross-assay attachment voids "
                      "the record's own condition_match='assay_only' claim)",
        }
    if not construct_match:
        # A real M6 record EXISTS for this uniprot, but for a different construct.
        # Refuse to attach it; report the mismatch explicitly instead of WT-bleed.
        return {
            "available": False,
            "uniprot_id": prop_rec.get("uniprot_id"),
            "construct_match": False,
            "assay_match": assay_match,
            "prop_construct_id": prop_construct,
            "prop_assay": prop_assay,
            "reason": "no construct-matched M6 record — the only M6 record for this "
                      f"protein is construct '{prop_construct}', not '{want_construct}' "
                      "(refusing to attach a different construct's propensity)",
        }
    p = prop_rec.get("propensity", {}) or {}
    return {
        "available": True,
        "uniprot_id": prop_rec.get("uniprot_id"),
        "construct_match": True,
        "assay_match": assay_match,
        "prop_construct_id": prop_construct,
        "prop_assay": prop_assay,
        "scored_feature": prop_rec.get("scored_feature"),
        "feature_status": prop_rec.get("feature_status"),
        "condition_match": prop_rec.get("condition_match"),
        "surface": p.get("surface"),
        "intrinsic": p.get("intrinsic"),
        "cohort": p.get("cohort"),
    }


def _sequence_axis_block(prop_rec: dict | None, want_assay: str | None = None) -> dict:
    """The protein-level sequence axis, refused across the ASSAY axis (join-policy-1.0).

    The block is deliberately surfaced from ANY record for this uniprot regardless of
    CONSTRUCT: the predictor scores are computed over the protein's deposited
    peptides and are genuinely construct-invariant, so refusing them on a construct
    mismatch would discard real protein-level information. That reasoning does NOT
    extend to the assay, and the previous comment here — "identical across the
    protein's constructs" — was silent on exactly the axis that matters.

    The block splits in two. ASSAY-INVARIANT: `uniprot_id`, `protein_name`,
    `n_peptides*`, `peptide_length_range`, `excluded_available_predictors`,
    `absent_predictors`, and each sub-axis's predictor payload including its own
    `endpoint_class`. ASSAY-DEPENDENT, each a pure function of the assay: `assay`,
    `assay_endpoint_class`, and per sub-axis `endpoint_match_to_assay`,
    `comparison_licensed`, plus `no_comparison_licensed_reason`.

    So a cross-assay block does not merely over-license — the licensing FLIPS.
    Measured over the 7 contaminated blocks before the fix: of 28 (series ×
    predictor) licensing decisions, 14 were wrong-True, 10 wrong-False and only 4
    correct. Turbidity curves published PASTA+Waltz as licensed when a `generic`
    endpoint licenses TANGO+AGGRESCAN instead.

    Suppressing just the assay-dependent fields was rejected: it would be honest
    about the 14 wrong-True decisions while silently discarding the 10 wrong-False
    ones, which are genuinely available comparisons — a second information loss, not
    the neutral choice. Recomputing licensing for the curve's own assay was also
    rejected: M7 does no new inference, it joins, and it must not become a second
    authoring site for a rule m6_propensity owns. After the M6 re-key onto
    (uniprot, construct, assay), the residual cross-assay cases are exactly those
    with no assay-matched kinetics at all, where no honest per-curve licensing
    exists to surface."""
    if not prop_rec:
        return {"available": False,
                "reason": "no M6 record joined for this protein"}
    sa = prop_rec.get("sequence_axis")
    gate = _applied_policy("SEQUENCE_AXIS_ASSAY_GATE", "assay_matched_only")
    if gate != "assay_matched_only":
        return sa
    rec_assay = prop_rec.get("assay")
    if want_assay is None or rec_assay == want_assay:
        return sa
    return {
        "available": False,
        "prop_assay": rec_assay,
        "assay_match": False,
        "reason": ("no assay-matched M6 record — the only sequence axis for this "
                   f"protein was computed for assay '{rec_assay}', not "
                   f"'{want_assay}'. The predictor scores are protein-level and "
                   "assay-invariant, but their comparison LICENSING "
                   "(assay_endpoint_class, endpoint_match_to_assay, "
                   "comparison_licensed) is a pure function of the assay, so "
                   "surfacing this block would publish licensing computed against "
                   "the wrong endpoint"),
    }


def assemble_curve(
    triage: dict,
    feat: dict | None,
    regime_rec: dict | None,
    gamma_rec: dict | None,
    m3_rec: dict | None,
    fit_rec: dict | None,
    prop_rec: dict | None,
    build_id: str,
) -> dict:
    """Assemble ONE ProteinAnalysisResult for one curve (series_id).

    Returns the FULL schema for the three informative yield tiers and the COMPACT
    schema for {signal_only, uninformative} (the degenerate-case contract). The
    yield tier itself is always computed from the joined upstream licences."""
    m1 = triage.get("m1", {}) or {}
    cv = triage.get("condition_vector", {}) or {}
    has_cs = triage.get("concentration_series_id") is not None
    yield_tier = information_yield(m1, feat, regime_rec, gamma_rec)
    regime_block = (regime_rec or {}).get("descriptive_regime", {}) or {}

    # ---- COMPACT result (signal_only / uninformative) ---------------------- #
    # NOT the full refusing payload, NOT a caveat avalanche: just identity, the
    # regime, the tier, ONE M9 hook, and ≤3 impact-ranked caveats.
    if yield_tier in _COMPACT_TIERS:
        return {
            "schema": "compact",
            "protein_id": triage.get("protein_id"),
            "uniprot_id": triage.get("uniprot_id"),
            "series_id": triage.get("series_id"),
            "condition_vector": cv,
            "descriptive_regime": regime_block,
            "information_yield": yield_tier,
            "information_yield_rank": YIELD_RANK[yield_tier],
            "m9_hook": m9_hook(m1, feat, regime_rec, gamma_rec, cv, has_cs),
            "caveats": collect_caveats(m1, feat, regime_rec, gamma_rec, cv, has_cs, cap=3),
            "engine_build_id": build_id,
        }

    # ---- FULL result (mechanistic / scaling / descriptive) ----------------- #
    mech = (regime_rec or {}).get("mechanistic", {}) or {}
    anomaly = (regime_rec or {}).get("anomaly", {}) or {}
    f = feat or {}
    feats = f.get("features", {}) or {}
    return {
        "schema": "full",
        "protein_id": triage.get("protein_id"),
        "uniprot_id": triage.get("uniprot_id"),
        "series_id": triage.get("series_id"),
        "source": triage.get("source"),
        "data_mode": triage.get("data_mode"),
        "condition_vector": cv,
        "fittability_class": triage.get("fittability_class") or m1.get("fittability_class"),
        "censoring_class": triage.get("censoring_class") or m1.get("censoring_class"),
        "information_yield": yield_tier,
        "information_yield_rank": YIELD_RANK[yield_tier],

        # The M5 confidence block ("R²+anomaly+M3 stability") is a DESCRIPTIVE-SHAPE
        # confidence, not a mechanistic one. It belongs to the descriptive regime
        # (kept here verbatim) and is ALSO surfaced top-level as `shape_confidence`
        # so a reader never has to dig it out of the mechanistic block to find it.
        "descriptive_regime": {
            **regime_block,
            "confidence": (regime_rec or {}).get("confidence"),
        },
        "shape_confidence": (regime_rec or {}).get("confidence"),
        "mechanistic": {
            "verdict_or_equivalence_class": (
                mech.get("single_mechanism_call") or mech.get("equivalence_class")),
            "mechanistic_inference_licensed": mech.get("mechanistic_inference_licensed"),
            # HONESTY FIX (item 3): when a mechanistic inference is NOT licensed
            # (the corpus-wide case), `confidence` MUST be None — borrowing the M5
            # descriptive-shape confidence here presents shape certainty as
            # mechanistic certainty. The shape confidence is preserved unchanged in
            # `descriptive_regime.confidence` / top-level `shape_confidence`; it is
            # only the mechanistic confidence that is nulled when unlicensed.
            "confidence": (
                ((regime_rec or {}).get("confidence", {}) or {}).get("level")
                if mech.get("mechanistic_inference_licensed") is True
                and isinstance((regime_rec or {}).get("confidence"), dict)
                else None),
            "confidence_note": (
                None if mech.get("mechanistic_inference_licensed") is True
                else "mechanistic_inference_licensed is False — mechanistic "
                     "confidence is undefined; see shape_confidence for the M5 "
                     "descriptive-shape confidence (do NOT read it as mechanistic)"),
            "gates_failed": mech.get("gates_failed"),
            "degeneracy": mech.get("degeneracy"),
        },
        "model_selection": _model_selection(m3_rec, fit_rec),

        "curve_features": {
            "features": feats,
            "definition_contract": f.get("definition_contract"),
            "signal_basis": f.get("signal_basis") or m1.get("signal_basis"),
            "censoring_class": f.get("censoring_class") or m1.get("censoring_class"),
            "t50_status": f.get("t50_status"),
            "lag_status": f.get("lag_status"),
            "validity_flags": f.get("validity_flags"),
            "status": f.get("status"),
        },
        "lag_to_t50_ratio": feats.get("lag_to_t50_ratio"),

        "gamma_global": (gamma_rec or {}).get("gamma_global") if gamma_rec else None,
        "gamma_regression": (gamma_rec or {}).get("gamma_regression") if gamma_rec else None,
        "gamma_disagreement": (gamma_rec or {}).get("disagreement") if gamma_rec else None,
        "window_of_validity": (regime_rec or {}).get("window_of_validity"),

        # propensity is construct-gated (item 4): a non-construct-matched record is
        # reported available=False, never attached as if it were this curve's.
        "propensity": _propensity_block(
            prop_rec, triage.get("uniprot_id"),
            want_construct=cv.get("construct_id"),
            want_assay=cv.get("assay_type")),
        # sequence_axis is a PROTEIN-level (uniprot) view, so it is surfaced from any
        # record for this uniprot even when the CONSTRUCT differs — the predictor
        # scores are construct-invariant and it is not a per-curve propensity claim.
        # It is NOT surfaced across the ASSAY, because the block's licensing fields
        # are pure functions of the assay; see _sequence_axis_block.
        "sequence_axis": _sequence_axis_block(prop_rec,
                                              want_assay=cv.get("assay_type")),

        "quality_report": {
            "quality_flags": triage.get("quality_flags"),
            "artifact_flags": m1.get("reasons"),
            "normalization_mode": triage.get("normalization_mode") or m1.get("normalization_mode"),
            "signal_basis": m1.get("signal_basis"),
            "plate_qc": m1.get("plate_qc"),
            "digitization_uncertainty": triage.get("digitization_uncertainty"),
            "anomaly": {
                "fdr_significant": anomaly.get("fdr_significant"),
                "material_misfit": anomaly.get("material_misfit"),
                "proposal_actionable": anomaly.get("proposal_actionable"),
            },
            "missingness": _missingness(cv),
        },
        # Full results carry impact-ranked caveats too (capped) — honesty without
        # the avalanche, same machinery as the compact path.
        "caveats": collect_caveats(m1, feat, regime_rec, gamma_rec, cv, has_cs, cap=3),
        "engine_build_id": build_id,
    }


def _missingness(condition_vector: dict | None) -> dict:
    """Compact summary of comparability-critical condition fields that are
    UNKNOWN (the missingness-vs-mismatch policy, Service A): we report HOW MANY
    and WHICH, not a per-field avalanche."""
    cv = condition_vector or {}
    prov = cv.get("field_provenance", {}) or {}
    unknown = sorted(k for k, v in prov.items() if v == "unknown")
    return {"n_unknown_fields": len(unknown), "unknown_fields": unknown}


# =========================================================================== #
# corpus rollup — FDR-aware, definition-homogeneous
# =========================================================================== #
# Corpus-UNIVERSAL limitations (item 8): true of EVERY curve in this corpus, so
# they are factored out of the per-curve ≤3 caveat cap (where they would crowd out
# curve-specific refusals) and stated ONCE here. These are evicted from the per-
# curve list by design — naming them at the corpus level keeps them honest.
_CORPUS_CAVEATS = [
    "every kinetic value is a CPAD graph-digitization — elevated digitization "
    "uncertainty on every point of every curve (a corpus-wide systematic, not a "
    "per-curve flag)",
    "agitation and seeding are unrecorded corpus-wide for most curves — the "
    "mechanistic primary/secondary-nucleation branch is undetermined corpus-wide, "
    "independent of any single curve",
]

# Null-regime display key for by_descriptive_regime (item 7): a curve with no
# resolved shape regime maps to None; we surface it under a readable label rather
# than a bare JSON null key.
_NO_REGIME_KEY = "(no resolved regime)"


def build_rollup(results: list[dict], build_id: str, components: dict,
                 regimes_by_id: dict,
                 duplicate_key_counts: dict | None = None,
                 dose_response_summary: dict | None = None) -> dict:
    """A compact, honest corpus rollup: tier counts, regime counts, BOTH γ counts
    (reliable-fit vs significant), n mechanistically licensed (expect 0 — noted),
    the anomaly FDR summary, a definition-homogeneity note, corpus-universal
    caveats, propensity construct-mismatch counts, and duplicate-key collision
    counts. Only definition-homogeneous results are pooled."""
    n = len(results)
    by_tier = Counter(r["information_yield"] for r in results)

    # by_descriptive_regime: map the None-regime key to a readable label and SORT
    # the dict for parity with by_information_yield (item 7).
    raw_by_regime = Counter(
        (r.get("descriptive_regime", {}) or {}).get("regime") for r in results)
    by_regime = {}
    for regime in sorted(raw_by_regime, key=lambda k: (k is None, k or "")):
        label = _NO_REGIME_KEY if regime is None else regime
        by_regime[label] = raw_by_regime[regime]

    # γ COUNT SPLIT (item 2): report BOTH counts instead of a single overstated one,
    # mirroring the anomaly n_fdr_significant/n_actionable pattern. A curve lands on
    # the `scaling` tier iff its γ is SIGNIFICANT (reliable AND CI excludes 0), so
    # the tier count == the significant count; the reliable-fit count is larger (it
    # includes the ~77% whose CI spans 0 — a clean fit but no resolved exponent).
    def _gamma_flags(r: dict) -> tuple[bool, bool]:
        # Read the γ regression off whichever schema the result carries (full
        # results expose gamma_regression top-level; compact results don't carry γ).
        reg = (r.get("gamma_regression") or {})
        reliable = reg.get("gamma_reliable") is True
        significant = r.get("information_yield") == "scaling"
        return reliable, significant

    n_with_reliable_gamma_fit = sum(1 for r in results if _gamma_flags(r)[0])
    n_with_significant_gamma = sum(1 for r in results if _gamma_flags(r)[1])
    n_mechanistic = by_tier.get("mechanistic", 0)

    # PROPENSITY construct-mismatch tally (item 4): count results where an M6 record
    # exists for the protein but for a different construct (construct_match False)
    # vs results that attached a genuinely construct-matched propensity.
    # Since join-policy-1.0 the ASSAY mismatch is tallied separately: a consumer
    # could not previously tell the two refusal classes apart, and they have
    # different causes and different fixes.
    n_prop_attached = n_prop_construct_mismatch_suppressed = 0
    n_prop_assay_mismatch_suppressed = 0
    for r in results:
        prop = r.get("propensity") or {}
        if "construct_match" not in prop:
            continue                                  # compact result, no propensity
        if prop.get("construct_match") is True and prop.get("available") is True:
            n_prop_attached += 1
        elif (prop.get("assay_match") is False
              and prop.get("prop_assay") is not None):
            # available=False BECAUSE a real record existed for another ASSAY. Counted
            # first because the assay gate is tested first: a record failing both axes
            # is reported as the assay failure, which is the stronger claim.
            n_prop_assay_mismatch_suppressed += 1
        elif (prop.get("construct_match") is False
              and prop.get("prop_construct_id") is not None):
            # available=False BECAUSE a real record existed for another construct
            # (distinct from "no M6 record at all", where prop_construct_id is None).
            n_prop_construct_mismatch_suppressed += 1

    # anomaly FDR summary — read straight from the M5 regimes artifact. The FDR
    # control is applied at the M5 driver; we only TALLY here, no re-test. NOTE this
    # is CORPUS-WIDE (the full regimes file), so it ignores --limit (item 6) — named
    # accordingly so a --limit run's smaller result denominator isn't confused with it.
    n_fdr_sig = n_actionable = n_anomaly_testable = 0
    for rec in regimes_by_id.values():
        an = rec.get("anomaly", {}) or {}
        n_anomaly_testable += bool(an.get("testable"))
        n_fdr_sig += bool(an.get("fdr_significant"))
        n_actionable += bool(an.get("proposal_actionable"))

    # definition-homogeneity: pooling is only valid across results that share the
    # same definition contract + regime registry (we only emit ONE build, so all
    # pooled here are homogeneous by construction — stated, not assumed).
    contracts = {components["m4_definition_contract"]}
    registries = {components["m5_regime_registry"]}

    return {
        "engine_build_id": build_id,
        "engine_build_components": components,
        "n_results": n,
        "denominator_note": (
            "denominator = every curve in curves_triaged (fittable + suspect + "
            "unfittable); unfittable/no-transition curves are included as "
            "signal_only/uninformative so the rollup is complete"),
        "by_information_yield": {t: by_tier.get(t, 0) for t in YIELD_LADDER},
        "by_descriptive_regime": by_regime,
        # BOTH γ counts (item 2) — never a single overstated n_with_reliable_gamma.
        "n_with_reliable_gamma_fit": n_with_reliable_gamma_fit,
        "n_with_significant_gamma": n_with_significant_gamma,
        "gamma_count_note": (
            "n_with_reliable_gamma_fit = γ fit/CI machinery ran cleanly; "
            "n_with_significant_gamma = of those, the 95% CI also EXCLUDES 0 (a "
            "resolved scaling exponent) — only the latter earn the `scaling` tier. "
            "The gap is the ~77% reliable-but-CI-spans-0 curves (no scaling resolved)."),
        "propensity_join_summary": {
            "n_propensity_attached": n_prop_attached,
            "n_construct_mismatch_suppressed": n_prop_construct_mismatch_suppressed,
            "n_assay_mismatch_suppressed": n_prop_assay_mismatch_suppressed,
            "note": "construct-mismatch suppressions are curves whose protein HAS an "
                    "M6 record but only for a different construct (usually Wild-Type); "
                    "we refuse to attach it (item 4) and report available=False with "
                    "construct_match=False rather than bleed WT propensity onto a mutant",
            "assay_note": "assay-mismatch suppressions are curves whose protein HAS an "
                          "M6 record for the right construct but a DIFFERENT assay "
                          "(join-policy-1.0). Refused because intrinsic/cohort are "
                          "anchored in an assay-matched stratum, so attaching one "
                          "would rank the curve against a pool containing no member "
                          "measured by its own assay. Counted before the construct "
                          "class, so the two are disjoint",
        },
        "n_mechanistic_licensed": n_mechanistic,
        "n_mechanistic_licensed_note": (
            "expected 0: single-mechanism calls require a Service-C confusion "
            "matrix + FDR-controlled degeneracy threshold (§4), not yet built — "
            "so no curve in this corpus is mechanistically licensed"),
        # renamed (item 6): this tally is CORPUS-WIDE (full regimes file), NOT the
        # assembled `results`, so under --limit it has a different (larger) denominator.
        "anomaly_fdr_summary_corpus_wide": {
            "n_anomaly_testable": n_anomaly_testable,
            "n_fdr_significant": n_fdr_sig,
            "n_actionable": n_actionable,
            "fdr_note": "Benjamini–Hochberg control applied at the M5 driver; a "
                        "proposal additionally requires material_misfit (effect size)",
            "scope_note": "tallied over the FULL M5 regimes artifact (every curve), "
                          "so this IGNORES --limit — its denominator is the corpus, "
                          "not the assembled results count above",
        },
        "definition_homogeneity": {
            "definition_contracts_pooled": sorted(contracts),
            "regime_registries_pooled": sorted(registries),
            "note": "all results share one engine_build_id, hence one definition "
                    "contract + regime registry — pooling is definition-homogeneous "
                    "by construction; cross-build pooling is BLOCKED (§7)",
        },
        # corpus-universal caveats (item 8): factored out of the per-curve cap.
        "corpus_caveats": _CORPUS_CAVEATS,
        # duplicate-key collisions surfaced instead of silent last-wins (item 12).
        "index_duplicate_key_counts": duplicate_key_counts or {},
        "cohort_suppression_note": (
            "cohort percentiles are suppressed for singleton strata (N<3) upstream "
            "in M6; this rollup reports tiers, not cohort ranks, so no singleton "
            "stratum is surfaced here"),
        "schema_counts": dict(Counter(r["schema"] for r in results)),
        # DOSE-RESPONSE (k_agg vs concentration) corpus summary — the complementary,
        # endpoint-style R-row scaling view (m2_fit --dose); empty if not built.
        "dose_response_summary": dose_response_summary or {
            "n_proteins": 0,
            "note": "dose_response_fits.jsonl not built — run `python engine/m2_fit.py --dose`",
        },
    }


# =========================================================================== #
# driver
# =========================================================================== #
def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL artifact; missing file -> [] (never raises — an absent
    optional artifact must degrade gracefully, not crash the assembler)."""
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue       # skip a corrupt line rather than abort the build
    return out


def _index(records: list[dict], key: str) -> dict:
    """Index records by a single key (last-wins on duplicate keys)."""
    return {r[key]: r for r in records if r.get(key) is not None}


def _index_duplicate_count(records: list[dict], key: str) -> int:
    """Count how many records were dropped to duplicate-key collisions when
    indexing by `key` (item 12). last-wins is silent in `_index`; here we surface
    the collision count so the rollup can report it instead of hiding lost records.
    n_dropped = n_records_with_key − n_distinct_keys."""
    keys = [r[key] for r in records if r.get(key) is not None]
    return len(keys) - len(set(keys))


def _read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def main(argv: list[str] | None = None) -> int:
    # Default input dir is repo-root-anchored (Path(__file__)…), matching the M1–M6
    # drivers, so `python engine/m7_assemble.py` works from any cwd.
    proc_default = Path(__file__).resolve().parent.parent / "data" / "processed"
    ap = argparse.ArgumentParser(description="M7 — assemble ProteinAnalysisResults.")
    ap.add_argument("--input", type=Path, default=proc_default,
                    help="directory holding the M1–M6 artifacts (default data/processed)")
    ap.add_argument("--limit", type=int, default=None,
                    help="assemble at most N curves (debug)")
    args = ap.parse_args(argv)

    indir = Path(args.input)

    # ---- load all artifacts ------------------------------------------------ #
    triaged = _load_jsonl(indir / "curves_triaged.jsonl")
    features_raw = _load_jsonl(indir / "features.jsonl")
    regimes_raw = _load_jsonl(indir / "regimes.jsonl")
    m3_raw = _load_jsonl(indir / "m3_sample.jsonl")
    fits_raw = _load_jsonl(indir / "fits.jsonl")
    gamma_raw = _load_jsonl(indir / "gamma.jsonl")
    features = _index(features_raw, "series_id")
    regimes = _index(regimes_raw, "series_id")
    m3 = _index(m3_raw, "series_id")
    fits = _index(fits_raw, "series_id")
    gamma = _index(gamma_raw, "concentration_series_id")

    # Surface duplicate series_id / concentration_series_id collisions (item 12):
    # `_index` is last-wins, which silently drops earlier records. We count the
    # drops per artifact so the rollup can report them instead of hiding the loss.
    duplicate_key_counts = {
        "features.series_id": _index_duplicate_count(features_raw, "series_id"),
        "regimes.series_id": _index_duplicate_count(regimes_raw, "series_id"),
        "m3_sample.series_id": _index_duplicate_count(m3_raw, "series_id"),
        "fits.series_id": _index_duplicate_count(fits_raw, "series_id"),
        "gamma.concentration_series_id": _index_duplicate_count(
            gamma_raw, "concentration_series_id"),
    }

    # propensity is keyed by (uniprot_id, assay, construct_id). We keep a uniprot-
    # only index too, but ONLY so a protein-level sequence_axis can still surface on
    # a triple miss — the per-curve `propensity` block is construct-gated downstream
    # (_propensity_block), so a uniprot-fallback record never bleeds its (usually
    # Wild-Type) propensity onto a mutant curve. The mismatch is counted in the
    # rollup from each result's `propensity.construct_match` flag.
    propensity = _load_jsonl(indir / "propensity.jsonl")
    prop_by_triple: dict[tuple, dict] = {}
    prop_by_uniprot: dict[str, dict] = {}
    for p in propensity:
        up = p.get("uniprot_id")
        cv = p.get("condition_vector", {}) or {}
        triple = (up, p.get("assay"), cv.get("construct_id"))
        prop_by_triple[triple] = p
        prop_by_uniprot.setdefault(up, p)   # first record = uniprot fallback (seq_axis only)

    # ---- engine_build_id (corpus digest from the manifests) ---------------- #
    # corpus_version folds EVERY source byte-digest from BOTH ETL manifests, not
    # just the two it used to (item 5): the kinetics source_sha256 AND all three
    # sequence sources (peptides + aprs + structures). The digests are SORTED before
    # hashing so the corpus id is order-independent (a manifest re-emit that reorders
    # keys must not change the build), and any one source byte-change flips the id.
    kin_man = _read_manifest(indir / "etl_manifest.json")
    seq_man = _read_manifest(indir / "sequence_manifest.json")
    seq_sources = seq_man.get("sources", {}) or {}
    corpus_source_digests = [d for d in [
        kin_man.get("source_sha256"),
        (seq_sources.get("peptides", {}) or {}).get("sha256"),
        (seq_sources.get("aprs", {}) or {}).get("sha256"),
        (seq_sources.get("structures", {}) or {}).get("sha256"),
    ] if d]
    corpus_digest_src = "|".join(sorted(corpus_source_digests)) or None
    corpus_version = (
        hashlib.sha256(corpus_digest_src.encode()).hexdigest()[:12]
        if corpus_digest_src else "unknown-corpus")
    components = build_components(
        kinetics_schema_version=kin_man.get("schema_version", _KINETICS_SCHEMA_DEFAULT),
        sequence_schema_version=seq_man.get("schema_version", _SEQUENCE_SCHEMA_DEFAULT),
        corpus_version=corpus_version,
    )
    build_id = engine_build_id(components)

    # ---- assemble one result per curve ------------------------------------- #
    rows = triaged if args.limit is None else triaged[: args.limit]
    results: list[dict] = []
    for triage in rows:
        sid = triage.get("series_id")
        up = triage.get("uniprot_id")
        cv = triage.get("condition_vector", {}) or {}
        cs_id = triage.get("concentration_series_id")
        prop = (prop_by_triple.get((up, cv.get("assay_type"), cv.get("construct_id")))
                or prop_by_uniprot.get(up))
        result = assemble_curve(
            triage=triage,
            feat=features.get(sid),
            regime_rec=regimes.get(sid),
            gamma_rec=gamma.get(cs_id) if cs_id else None,
            m3_rec=m3.get(sid),
            fit_rec=fits.get(sid),
            prop_rec=prop,
            build_id=build_id,
        )
        results.append(result)

    # ---- write per-curve results + rollup ---------------------------------- #
    out_results = indir / "protein_analysis.jsonl"
    with open(out_results, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    # DOSE-RESPONSE corpus summary (read-only; the per-protein fits + the web card
    # are the authoritative surface — here we just tally best-model counts).
    dr_rows = _load_jsonl(indir / "dose_response_fits.jsonl")
    dose_summary = None
    if dr_rows:
        best_counts: Counter = Counter()
        for d in dr_rows:
            best_counts[d.get("best_by_aicc") or "none"] += 1
        dose_summary = {
            "n_proteins": len(dr_rows),
            "best_dose_model_by_aicc_preliminary": dict(best_counts),
            "source": "CPAD rate-vs-concentration R-rows (precomputed k_agg); a "
                      "complementary endpoint-style scaling view, distinct from the "
                      "engine's fitted t50-scaling γ. Selection here is preliminary (M3).",
        }

    rollup = build_rollup(results, build_id, components, regimes,
                          duplicate_key_counts=duplicate_key_counts,
                          dose_response_summary=dose_summary)
    out_rollup = indir / "m7_rollup.json"
    out_rollup.write_text(json.dumps(rollup, indent=2), encoding="utf-8")

    # ---- summary ----------------------------------------------------------- #
    by_tier = Counter(r["information_yield"] for r in results)
    by_schema = Counter(r["schema"] for r in results)
    print(f"[M7] wrote {out_results}  ({len(results)} results)")
    print(f"[M7] wrote {out_rollup}")
    print(json.dumps({
        "n_results": len(results),
        "full": by_schema.get("full", 0),
        "compact": by_schema.get("compact", 0),
        "by_information_yield": {t: by_tier.get(t, 0) for t in YIELD_LADDER},
        "engine_build_id": build_id,
        "versions_imported_from_modules": _VERSIONS_IMPORTED,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
