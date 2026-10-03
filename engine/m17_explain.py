"""
PRISE — Module M17: Explanation, Provenance & Evidence-Graph Layer (Stage A)
============================================================================

M17 turns every value the engine emits into a node of a directed ACYCLIC
evidence graph whose edges are facts the engine ALREADY computed. It is the
answer to "why does PRISE say this?" — assembled only from artifacts on disk,
never from a fresh calculation.

=== THE LOAD-BEARING INVARIANT: M17 DERIVES NOTHING ========================

M17 performs no fitting, no statistics, no thresholding, and emits no number of
its own. It is a HARVESTER + GRAPH BUILDER + DETERMINISTIC RENDERER. The reason
is not stylistic: if M17 could compute, its explanations could disagree with the
engine, and an explanation that disagrees with the thing it explains is worse
than no explanation at all.

This is enforced MECHANICALLY, in four independent ways, so that a violation
FAILS rather than merely being discouraged:

  (a) VALUE PROVENANCE. Every node whose `value` is not None carries a
      `source_ref` = (artifact, record_key, key_path). `assert_no_derivation()`
      re-resolves that ref against the artifact RE-READ FROM DISK and requires
      the canonical JSON encodings to be byte-identical. A number M17 computed
      cannot exist at any key path in any artifact, so it cannot pass.
  (b) ARITHMETIC BAN. `_frac()` is the ONLY function in this module permitted to
      contain a division/multiplication/subtraction/power, and it operates only
      on M17's OWN node counts (never on an engine value). The test suite parses
      this file with `ast` and asserts every such operator lies inside `_frac`.
  (c) TEMPLATE-ONLY RENDERING. `render_node()` is a single dict lookup followed
      by a single `str.format_map`. It contains no string concatenation, no
      f-string and no `join`, so a sentence CANNOT be composed at runtime. Same
      graph in => byte-identical text out.
  (d) NO NUMPY. This module imports stdlib only. Every float in the graph
      arrives via `json.loads` and is therefore a plain Python float, never a
      `numpy.float64` whose repr could drift between builds.

=== STAGE A SCOPE =========================================================

Stage A changes NOTHING in m0..m16. M17 owns one small read-only ADAPTER per
source module (all living in this file), each of which:

  * DECLARES the artifact key paths it reads (`declared_keys`). A contract test
    asserts those paths actually exist in the live artifacts, so when a module
    later changes shape the M17 suite FAILS LOUDLY instead of silently degrading
    to `explanation_available: false`.
  * DECLARES which emitted M7-spine fields its nodes explain (`explains`), which
    is what makes coverage MEASURED rather than asserted.

=== ONE GRAPH, FOUR PROJECTIONS ===========================================

Assumption trees, evidence chains, uncertainty chains and human-readable
explanations are NOT four data structures — they are four VIEWS of the one DAG
(filter by edge kind, then topologically order). They are implemented as
projections over shared node ids, so they cannot contradict one another. A test
asserts the union of the projections' edge sets is the WHOLE edge vocabulary:
no edge kind may be invisible in every view.

=== CONFIDENCE IS A TAGGED LIST, NEVER A SCALAR ===========================

The engine carries at least four INCOMMENSURABLE confidences:
  * bootstrap CI            (M4 γ; asymptotic, model-conditional)
  * conformal quantile      (conformal.py; distribution-free, finite-sample)
  * calibrated claim conf.  (M15; ECE/Brier measured — on FIXTURES, see below)
  * metadata completeness   (M11; PRESENCE, explicitly NOT correctness)
plus M5's ordinal shape confidence. Averaging them would fabricate a number and
violate invariant §10.5 (γ is never a single merged scalar). So `confidence` is
ALWAYS a list of `{kind, value, basis, valid_at}` objects and there is no code
path anywhere in this module that reduces that list to one number.

=== WHAT M17 ACTIVELY REFUSES =============================================

  * NO PROSE beyond deterministic templates over typed nodes.
  * NO merged confidence scalar.
  * NO CAUSAL VERB where the source node is statistical-only. Every node carries
    a `causal_license`; the renderer inherits it. Note the licence value
    "measured_causal" is DECLARED BUT UNUSED — nothing in this engine licenses a
    causal claim about nature, and a test asserts zero nodes carry it.
  * NO EXPLANATION IT CANNOT TRACE -> `explanation_available: false` + a reason.
    An untraceable field is NEVER narrated around.
  * NO QUANTITATIVE COUNTERFACTUAL unless the M11/M13 edge carries
    `gain_basis: "service_c_measured"`. A `structural_estimate` edge renders as
    an UPPER BOUND and inherits M8's validity ceiling verbatim.
  * NO CONFORMAL INTERVAL. Building `t50_hat ± q` is arithmetic on engine
    values, and selecting the Mondrian stratum would require an assay-endpoint
    classification (amyloid vs generic) that M17 is not licensed to make. M17
    therefore attaches the MARGINAL quantile verbatim and says so. This is a
    named narrowing of Stage A, recorded in `honesty_block()` and in the
    coverage artifact's `stage_b_spec` — discoverable from the artifact, not
    only from this source file.

=== HONEST LIMITS OF STAGE A (measured, see explanation_coverage.json) =====

  * `regimes.jsonl` and `features.jsonl` hold 1,240 records while the M7 spine
    holds 1,654. For the 414 unfittable/suspect series there is NO refusal, no
    regime and no feature vector to explain. That is the engine producing
    nothing, NOT M17 lacking an adapter — the coverage artifact reports the two
    populations SEPARATELY and never blends them into one flattering number.
  * §2.3 `FitProvenance` IS emitted by the modules that run an optimizer
    (`engine/fit_provenance.py`). `node["method"]["fit_provenance"]` carries
    the record's ADDRESS — artifact, key path, recipe table — and never its
    values, because M17 derives nothing. Methods that fit nothing (triage,
    classify, join) still report `available: false` with a reason, which is a
    true statement about them rather than a gap.
  * `structure_features.json` (17 MB) is DELIBERATELY NOT READ.

=== WHAT THE DEFAULT RUN WRITES (and what it deliberately does NOT) =======

The graph for one series is DERIVED — deterministically, in ~10 ms, from
artifacts already on disk. So the full corpus dump is a stored view of data
that is already stored, which is the same category error as serialising the
projections. It is therefore NOT written by default:

  * `explanation_coverage.json` (~54 KB) — ALWAYS. This is the deliverable.
  * `explanations_sample.jsonl` (<1 MB) — ALWAYS. A small, deterministically
    selected, stratified sample (see `select_sample_series`) so tests and
    consumers have real graphs to read without a 90 MB file.
  * `explanations.jsonl` (~90 MB) — ONLY under `--dump-graphs`. A build-time
    audit affordance, not a runtime input.

WHY THIS MATTERS CONCRETELY: `web/server.py` indexes the large per-series
JSONLs into memory AT STARTUP. A 90 MB file of ~175k fully-traced nodes becomes
several hundred MB of live Python objects at boot, in an app whose whole point
is being a zero-install local demo. Per-series trees are built ON DEMAND
instead — M17 derives nothing, so building one costs a dict lookup and a walk,
against the ~63 s that /api/series-gamma already spends doing real curve fits.

Usage:
    python engine/m17_explain.py                  # -> explanation_coverage.json
                                                  #  + explanations_sample.jsonl
    python engine/m17_explain.py --dump-graphs    # + explanations.jsonl (~90 MB)
    python engine/m17_explain.py --limit 50       # quick subset
    python engine/m17_explain.py --series CPAD-TK-1061   # one series, printed
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

M17_VERSION = "m17-explain-1.0"

# The directory every `source_ref` is expressed relative to. Refs are always
# POSIX-relative strings (never a Path, never absolute) so a graph built on
# Windows is byte-identical to one built on Linux.
ARTIFACT_DIR = "data/processed"

# A sentinel distinct from None: an artifact may legitimately STORE null, which
# is different from "this key path does not resolve".
_MISSING = object()

# The single documented reason `method.fit_provenance` is never populated. Kept
# as a module constant so the (long) explanation lives in exactly one place and
# every node carries only the short code.
FIT_PROVENANCE_ABSENT = {
    "available": False,
    "reason_code": "module_emits_no_per_fit_provenance",
    "reason": (
        "PRISE_DESIGN.md §2.3 defines FitProvenance {rng_seed, rng_algorithm, "
        "stream_policy, n_starts, winning_basin, basin_spread, optimizer, "
        "version, tolerance, convergence_flag, env_manifest_ref}. It IS now "
        "emitted by the modules that actually fit parameters — see "
        "FIT_PROVENANCE_AVAILABLE and engine/fit_provenance.py — but this "
        "particular method does not fit anything (it triages, classifies or "
        "joins), so there is no optimizer run, no start set and no basin for it "
        "to report. What it carries instead is version-tag provenance "
        "(definition contracts, registry versions, engine_build_id). That is a "
        "true statement about this method, not a gap: a method with no fit has "
        "no fit provenance."),
}


def fit_provenance_available(artifact: str, key_path: str,
                             recipe_table: str = "fit_recipes.json") -> dict:
    """Point at a §2.3 FitProvenance record that EXISTS, without reading or
    deriving it. M17 derives nothing, so it publishes the address of the record
    (artifact + key path + the recipe table its `recipe` id resolves into) and
    lets the consumer read it from the source of truth."""
    return {
        "available": True,
        "artifact": artifact,
        "key_path": key_path,
        "recipe_table": recipe_table,
        "note": ("the per-fit half (basin ledger: n_starts, n_converged, "
                 "winning_basin, basin_spread, the §7 basin verdict) lives at "
                 "`key_path`; the constant half (optimizer, tolerance, RNG "
                 "applicability, version, env_manifest_ref) lives in "
                 "`recipe_table` under the record's `recipe` id. M17 reports "
                 "the address and derives nothing."),
    }


# M2 is a fitting module and now emits the record §2.3 specifies.
FIT_PROVENANCE_M2 = fit_provenance_available("fits.jsonl", "fits.<model>.provenance")


# ========================================================================== #
# CLOSED VOCABULARIES
#
# All three are ORDERED TUPLES, never sets: iteration order is part of the
# output, and a set's iteration order is not a stable contract. Membership tests
# use the frozenset views. The test suite pins each tuple exactly, so widening a
# vocabulary is a deliberate, reviewed act rather than a drive-by import.
# ========================================================================== #
NODE_KINDS = (
    "result",          # an emitted value / verdict
    "datum",           # an observed input
    "metadata_fact",   # one `field_provenance` entry (or its M11 mirror)
    "method",          # a procedure + its version tag
    "assumption",      # taken as true, NOT measured
    "gate",            # a licensing pass/fail
    "refusal",         # a deliberate non-answer
    "uncertainty",     # an interval + its kind
    "deferral",        # a named not-yet-built dependency
    "ceiling",         # a validity bound on the claim
)
NODE_KIND_SET = frozenset(NODE_KINDS)

EDGE_KINDS = (
    "computed_from",
    "assumes",
    "gated_by",
    "blocked_by",
    "uncertainty_from",
    "bounded_by",
    "unblocked_by",    # points at the M11 action / M13 experiment that resolves it
)
EDGE_KIND_SET = frozenset(EDGE_KINDS)
_EDGE_RANK = {k: i for i, k in enumerate(EDGE_KINDS)}

# The incommensurable confidences. Each is a DIFFERENT kind of guarantee about a
# DIFFERENT thing; there is deliberately no ordering and no combining rule.
CONFIDENCE_KINDS = (
    "bootstrap_ci",              # M4 γ — asymptotic, conditional on the model
    "conformal_quantile",        # conformal.py — distribution-free finite-sample
    "claim_confidence_calibrated",  # M15 — ECE/Brier measured (on fixtures)
    "completeness_presence",     # M11 — PRESENCE, explicitly not correctness
    "shape_confidence_ordinal",  # M5 — an ordinal level, not a probability
)
CONFIDENCE_KIND_SET = frozenset(CONFIDENCE_KINDS)

CAUSAL_LICENSES = (
    "none",                        # no causal reading of any sort
    "deterministic_engine_logic",  # a fact about the ENGINE's control flow: a
                                   # failed gate genuinely causes the refusal.
                                   # This is a claim about software, not nature.
    "statistical_only",            # M10 / M16 associations — associative verbs only
    "measured_causal",             # DECLARED BUT UNUSED: nothing in this engine
                                   # licenses a causal claim about nature.
)
CAUSAL_LICENSE_SET = frozenset(CAUSAL_LICENSES)

# The four projections. Each is a filter over EDGE kinds; nodes are then
# topologically ordered. A test asserts the union is exactly EDGE_KIND_SET, so
# no edge kind can be invisible in every view.
PROJECTIONS = OrderedDict((
    ("assumption_tree", ("assumes", "gated_by")),
    ("evidence_chain", ("computed_from", "blocked_by", "unblocked_by")),
    ("uncertainty_chain", ("uncertainty_from", "bounded_by")),
    ("explanation", EDGE_KINDS),
))

# Indentation prefixes as a literal tuple. WHY not `"  " * depth`: multiplication
# is banned outside `_frac` (guard (b) in the module docstring), and a fixed
# tuple also caps render depth deterministically.
_INDENT = ("", "  ", "    ", "      ", "        ", "          ", "            ",
           "              ", "                ", "                  ",
           "                    ", "                      ",
           "                        ", "                          ",
           "                            ", "                              ")


def honesty_block() -> dict:
    """The machine-readable honesty stamp attached to every M17 output."""
    return {
        "module": M17_VERSION,
        "derives_nothing": (
            "M17 performs NO fitting, NO statistics, NO thresholding and emits "
            "NO number of its own. Every non-null node value is a VERBATIM read "
            "from a named artifact at a named key path, and assert_no_derivation() "
            "re-resolves each one against the artifact re-read from disk."),
        "confidence_never_merged": (
            "`confidence` is ALWAYS a list of {kind, value, basis, valid_at}. The "
            "engine's confidences are incommensurable (asymptotic bootstrap CI vs "
            "finite-sample conformal vs ECE/Brier-calibrated claim confidence vs "
            "M11 PRESENCE-completeness); averaging them would fabricate a number "
            "and violate invariant §10.5. No code path collapses the list."),
        "deterministic_templates_only": (
            "text is rendered by dict-lookup + str.format_map over typed nodes. "
            "Same graph => BYTE-IDENTICAL text. No free-form generation exists."),
        "causal_license_is_inherited": (
            "every node carries a causal_license and the renderer inherits it; a "
            "statistical-only source (M10/M16 associations) can never render a "
            "causal verb. 'measured_causal' is declared but UNUSED — nothing in "
            "this engine licenses a causal claim about nature."),
        "untraceable_is_refused_not_narrated": (
            "a field with no adapter returns explanation_available: false plus a "
            "reason. It is never narrated around and never partially guessed."),
        "counterfactuals_are_gated_on_gain_basis": (
            "a quantitative counterfactual is rendered ONLY for an unblocked_by "
            "edge with gain_basis='service_c_measured'. A 'structural_estimate' "
            "edge renders as an UPPER BOUND and inherits M8's validity ceiling."),
        "conformal_interval_is_refused": (
            "M17 attaches the MARGINAL conformal quantile verbatim and does NOT "
            "build t50_hat ± q: that is arithmetic on engine values, and choosing "
            "the Mondrian stratum would require an assay-endpoint classification "
            "M17 is not licensed to make. A named Stage-A narrowing."),
        "fit_provenance_is_addressed_not_derived": (
            "§2.3 FitProvenance IS emitted by the modules that run an optimizer "
            "(engine/fit_provenance.py). M17 publishes the ADDRESS of each "
            "record — artifact, key path, and the recipe table its `recipe` id "
            "resolves into — and never copies or recomputes its values. A method "
            "that fits nothing reports available: false with a reason, which is "
            "a true statement about that method rather than a gap."),
        "coverage_is_measured_not_asserted": (
            "explanation_coverage.json enumerates the DENOMINATOR explicitly (every "
            "M7-spine field path) and reports two separate coverage numbers — over "
            "explainable series and over all series — because 'M17 has no adapter' "
            "and 'the engine produced nothing to explain' are different failures "
            "with different fixes. They are never blended."),
        "m17_counts_are_m17s_own": (
            "the integers in explanation_coverage.json count M17's OWN nodes and "
            "fields. They are properties of the explanation layer, not engine "
            "results, and are the only numbers in this module not read from an "
            "upstream artifact."),
        "stage_a_edits_nothing": (
            "Stage A changes no file in m0..m16 and no file in web/. M17 is "
            "read-only over data/processed."),
        "structure_features_not_read": (
            "data/processed/structure_features.json (17 MB) is deliberately not "
            "read; M16 enters only via the small associations artifact, and in "
            "Stage A not at all."),
    }


# ========================================================================== #
# CANONICAL ENCODING + THE ONE ARITHMETIC FUNCTION
# ========================================================================== #
def _canon(obj) -> str:
    """The single canonical JSON encoding, used for hashing, byte-comparison and
    file writes. `allow_nan=False` means a NaN/Inf in an artifact RAISES here
    (and is caught by the caller, which marks the node absent) rather than
    silently emitting invalid JSON that no strict parser will read back."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False)


def _canon_safe(obj):
    """`_canon` but returning None instead of raising on NaN/Inf/unserialisable."""
    try:
        return _canon(obj)
    except (ValueError, TypeError):
        return None


def _frac(n, d):
    """THE ONLY function in this module containing arithmetic.

    It divides one M17 node/field COUNT by another to express coverage. It never
    touches an engine value. The test suite parses this file with `ast` and
    asserts every Div/Mult/Sub/Pow operator in the module lies inside this
    function — that is the mechanical form of "M17 derives nothing"."""
    if not d:
        return None
    return round(n / d, 6)


def _text(v) -> str:
    """Render a value for a template slot. Strings pass through unquoted (so a
    verbatim artifact sentence reads as a sentence); everything else uses the
    canonical encoding, so a float never picks up a locale- or platform-specific
    repr."""
    if isinstance(v, str):
        return v
    c = _canon_safe(v)
    return c if c is not None else "<unserialisable>"


# ========================================================================== #
# KEY-PATH GRAMMAR
#
#   a.b.c                       dict traversal
#   a[0].b                      list index
#   a[field=agitation].b        FIRST list element whose `field` == "agitation"
#
# The `[key=value]` selector exists because several artifacts store ranked lists
# (M11's delta_inferential_power, M8's blockers_impact_ranked). Addressing those
# by position would make a source_ref silently point at a DIFFERENT fact the
# moment upstream re-ranks; addressing by content keeps the ref stable and makes
# the contract test meaningful.
# ========================================================================== #
def parse_key_path(path: str) -> tuple:
    """Parse a key path into steps. Malformed input yields () — never raises."""
    if not isinstance(path, str) or not path:
        return ()
    steps = []
    buf = ""
    i = 0
    n = len(path)
    while i < n:
        ch = path[i]
        if ch == ".":
            if buf:
                steps.append(("key", buf))
                buf = ""
            i += 1
        elif ch == "[":
            if buf:
                steps.append(("key", buf))
                buf = ""
            j = path.find("]", i)
            if j < 0:
                return ()
            inner = path[i + 1:j]
            if "=" in inner:
                k, _, v = inner.partition("=")
                steps.append(("select", k, v))
            else:
                steps.append(("index", inner))
            i = j + 1
        else:
            buf += ch
            i += 1
    if buf:
        steps.append(("key", buf))
    return tuple(steps)


def resolve_key_path(root, path: str):
    """Resolve `path` against `root`. Returns `_MISSING` when it does not
    resolve, which is deliberately distinct from a stored null."""
    steps = parse_key_path(path)
    if not steps:
        return _MISSING
    cur = root
    for st in steps:
        if st[0] == "key":
            if not isinstance(cur, dict) or st[1] not in cur:
                return _MISSING
            cur = cur[st[1]]
        elif st[0] == "index":
            try:
                k = int(st[1])
            except (TypeError, ValueError):
                return _MISSING
            if not isinstance(cur, list) or k < 0 or k >= len(cur):
                return _MISSING
            cur = cur[k]
        else:
            if not isinstance(cur, list):
                return _MISSING
            hit = _MISSING
            for el in cur:
                if isinstance(el, dict) and _text(el.get(st[1])) == st[2]:
                    hit = el
                    break
            if hit is _MISSING:
                return _MISSING
            cur = hit
    return cur


def leaf_paths(obj, prefix: str = ""):
    """Enumerate dotted key paths to the leaves of a JSON object.

    DEFINITION (published in the coverage artifact so the denominator is
    auditable): a LIST is a leaf — its elements are not enumerated. An EMPTY
    dict, and a null, are leaves at their own path. So a field that is null on a
    compact record contributes its own parent path, which is why e.g.
    `descriptive_regime` and `descriptive_regime.regime` both appear in the
    universe: they are emitted by different record schemas."""
    if isinstance(obj, dict) and obj:
        for k, v in obj.items():
            child = k if not prefix else ".".join((prefix, k))
            for p in leaf_paths(v, child):
                yield p
    else:
        yield prefix


# ========================================================================== #
# ARTIFACT REGISTRY + LOADING
#
# JSONL artifacts are loaded into a PRUNED index: only the key paths the
# adapters DECLARE are retained. This keeps the whole corpus in memory (the raw
# JSONL set is ~45 MB of text and would be several hundred MB as live objects)
# and, usefully, makes the declared-keys contract load-bearing at runtime rather
# than merely documented. The test suite independently re-reads RAW records
# straight from disk and re-resolves every source_ref against them, so pruning
# can never hide a bad ref.
# ========================================================================== #
#
# A `record_key` is EITHER a scalar field name (one value identifies a record)
# OR a TUPLE of key paths (a COMPOSITE key). See `_record_index_key`.
# ========================================================================== #
# M6's propensity records are identified by the TRIPLE
# (uniprot_id, assay, condition_vector.construct_id) — m7_assemble.py:1002-1008.
# Indexing them on uniprot_id alone would collapse 28 P05067 records onto one
# and attach a Wild-Type propensity to a mutant curve, which is exactly the bug
# m7_assemble.py:548-552 exists to prevent.
PROPENSITY_RECORD_KEY = ("uniprot_id", "assay", "condition_vector.construct_id")

# ...and M7 FALLS BACK to a uniprot-only record when that triple misses
# (m7_assemble.py:1045-1046, `prop_by_triple.get(...) or prop_by_uniprot.get(up)`).
# A `record_key_fallback` must be a PREFIX of `record_key`, and it resolves to
# the FULL key of the FIRST record carrying that prefix in file order — which is
# exactly `prop_by_uniprot`'s `setdefault`. M17 MIRRORS this rather than
# implementing the join M7 should have written: see the
# `m6_assay_bleed_fixed_m17_mirrors_the_corrected_join` named narrowing. Reporting a
# value the product does not show would be a worse failure than the bug.
PROPENSITY_RECORD_KEY_FALLBACK = ("uniprot_id",)

ARTIFACTS = OrderedDict((
    ("curves_triaged.jsonl", {"kind": "jsonl", "record_key": "series_id"}),
    ("fits.jsonl", {"kind": "jsonl", "record_key": "series_id"}),
    ("features.jsonl", {"kind": "jsonl", "record_key": "series_id"}),
    ("regimes.jsonl", {"kind": "jsonl", "record_key": "series_id"}),
    ("metadata_quality.jsonl", {"kind": "jsonl", "record_key": "series_id"}),
    ("gamma.jsonl", {"kind": "jsonl", "record_key": "concentration_series_id"}),
    ("propensity.jsonl", {"kind": "jsonl", "record_key": PROPENSITY_RECORD_KEY,
                          "record_key_fallback": PROPENSITY_RECORD_KEY_FALLBACK}),
    ("m8_reachability.json", {"kind": "json", "record_key": None}),
    ("service_c_conformal.json", {"kind": "json", "record_key": None}),
    ("lit_validation.json", {"kind": "json", "record_key": None}),
    ("m7_rollup.json", {"kind": "json", "record_key": None}),
))

# ---- WHERE A JOIN KEY'S VALUES COME FROM --------------------------------- #
# ARCHITECTURAL RULE: a join key is sourced from an UPSTREAM artifact, NEVER
# from the M7 spine. The spine (protein_analysis.jsonl) is the coverage
# DENOMINATOR; keying an adapter off the very record whose fields it claims to
# explain would make the explanation circular. No adapter reads the spine.
#
#   LEFT  — the key path INSIDE the target artifact's record. This is also the
#           member name used in `Ctx.join_keys`, so the two sides cannot drift.
#   RIGHT — the key path inside curves_triaged.jsonl (M1) supplying the value.
#
# NOTE THE ASSAY ASYMMETRY: M6 stores the assay at top level as `assay`, the
# curve stores it at `condition_vector.assay_type`. m7_assemble.py builds its
# index from the M6 record's `assay` (line 1007) and looks it up with the
# curve's `condition_vector.assay_type` (line 1045). MEASURED on the live
# corpus, these are the same vocabulary — `assay == condition_vector.assay_type`
# in all 160 M6 records — so the triple is a real join, not a permanent miss.
#
# LATENT HAZARD (recorded, not yet closed): the LEFT column is a FLAT namespace.
# `join_keys_for_series` folds every artifact's members into ONE dict, so two
# artifacts declaring the same member name under different origins would silently
# overwrite each other and the second join would key off the first's value. Today
# the four member names are distinct, so nothing collides; the collision would be
# invisible if it ever happened. Stage B: key `join_keys` by (artifact, member).
JOIN_KEY_SOURCES = OrderedDict((
    ("gamma.jsonl", (("concentration_series_id", "concentration_series_id"),)),
    ("propensity.jsonl", (("uniprot_id", "uniprot_id"),
                          ("assay", "condition_vector.assay_type"),
                          ("condition_vector.construct_id",
                           "condition_vector.construct_id"))),
))


def join_keys_for_series(triage_record) -> dict:
    """Every join key M17 needs for one series, read from M1's triage record.

    Built in ONE place so the CLI, the tests and any consumer agree. A join key
    invented at a call site is how two code paths quietly start joining
    differently — and a mis-join is indistinguishable from a wrong answer."""
    rec = triage_record or {}
    keys: dict = {}
    for members in JOIN_KEY_SOURCES.values():
        for member, origin in members:
            v = resolve_key_path(rec, origin)
            keys[member] = None if v is _MISSING else v
    return keys


# The coverage DENOMINATOR source. Read by streaming: never held whole.
SPINE_ARTIFACT = "protein_analysis.jsonl"
SPINE_RECORD_KEY = "series_id"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_jsonl(path: Path):
    """Stream a JSONL file, skipping unparseable lines. Never raises."""
    try:
        fh = open(path, encoding="utf-8")
    except OSError:
        return
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def _assign_path(dst: dict, path: str, value):
    """Write `value` into `dst` at `path`, creating containers as needed, so a
    pruned record stays resolvable by the SAME path grammar as the raw one. A
    `[key=value]` selector materialises as a list element carrying that key, and
    an `[i]` index pads the list, so both resolve identically afterwards.

    The container type is chosen by LOOKING AHEAD at the next step: a step that
    is an index or a selector needs a LIST, everything else needs a dict.
    Getting this wrong silently prunes selector-addressed facts out of the
    bundle (they then resolve as missing and their nodes never appear), which is
    exactly the kind of quiet degradation M17 exists to prevent."""
    steps = parse_key_path(path)
    if not steps:
        return
    cur = dst
    for idx, st in enumerate(steps):
        rest = steps[idx + 1:]
        last = not rest
        if st[0] == "key":
            if not isinstance(cur, dict):
                return
            if last:
                cur[st[1]] = value
                return
            want_list = rest[0][0] in ("index", "select")
            nxt = cur.get(st[1])
            if want_list and not isinstance(nxt, list):
                nxt = []
                cur[st[1]] = nxt
            elif not want_list and not isinstance(nxt, dict):
                nxt = {}
                cur[st[1]] = nxt
            cur = nxt
        elif st[0] == "select":
            if not isinstance(cur, list):
                return
            hit = None
            for el in cur:
                if isinstance(el, dict) and _text(el.get(st[1])) == st[2]:
                    hit = el
                    break
            if hit is None:
                hit = {st[1]: st[2]}
                cur.append(hit)
            if last:
                return          # a selector addresses an element, not a value
            cur = hit
        else:
            try:
                k = int(st[1])
            except (TypeError, ValueError):
                return
            if not isinstance(cur, list) or k < 0:
                return
            while len(cur) <= k:
                cur.append(None)
            if last:
                cur[k] = value
                return
            if not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]


def _prune(record: dict, paths) -> dict:
    """Copy only the declared key paths out of `record`. Values are copied by
    reference and never transformed — pruning must be value-preserving or the
    byte-comparison guarantee is worthless."""
    out: dict = {}
    for p in paths:
        v = resolve_key_path(record, p)
        if v is _MISSING:
            continue
        _assign_path(out, p, v)
    return out


def _record_index_key(record, record_key):
    """Build the index key for ONE JSONL record.

    A SCALAR `record_key` is a single top-level field name and yields its value
    unchanged — byte-for-byte the behaviour every existing artifact relies on.

    A TUPLE `record_key` is a COMPOSITE key: each member is a KEY PATH resolved
    against the record, and the index key is the TUPLE of the resolved values.
    A member that does not resolve, or resolves to null, makes the WHOLE key
    None so the record is SKIPPED — a partially-keyed record indexed under a
    truncated key is a lookup waiting to hit the wrong record, which is the
    entire failure class the composite key exists to prevent.

    A DELIBERATE, MEASURED DIVERGENCE FROM UPSTREAM: m7_assemble.py:1007-1008
    indexes such a record anyway, under a tuple CONTAINING None, so a curve whose
    own key member is also null could match it. M17 skips it instead. Measured on
    the live corpus: 0 of the 160 propensity records has a null triple member, so
    the two indexes are identical today and no mirrored value can differ. If that
    ever changes, M17 will resolve NO record where M7 resolved one, which shows up
    as an absent explanation — the safe direction, and visible, not silent."""
    if not isinstance(record, dict):
        return None
    if isinstance(record_key, tuple):
        parts = []
        for member in record_key:
            v = resolve_key_path(record, member)
            if v is _MISSING or v is None:
                return None
            parts.append(v)
        return tuple(parts)
    return record.get(record_key)


def _fallback_index_key(index_key, fallback_key, record_key):
    """The leading PREFIX of a composite index key selected by `fallback_key`.

    `fallback_key` must be a LEADING PREFIX of `record_key` — same members, same
    order, starting at position 0. Anything else (a member not in `record_key`, a
    reordering, or a subset that skips a member) is a DECLARATION ERROR and yields
    None, i.e. no fallback, rather than a guess.

    The rule is prefix and not merely subset because the fallback index is built
    once per artifact and shared by every lookup: `("assay",)` against the triple
    would index 160 records under ~4 assay values and hand back an arbitrary
    protein's record. Only a leading prefix is a coarsening of the same key, which
    is what a fallback is allowed to be. Pinned by
    `test_fallback_key_must_be_a_leading_prefix_of_the_record_key`."""
    if not (isinstance(index_key, tuple) and isinstance(record_key, tuple)
            and isinstance(fallback_key, tuple)):
        return None
    if fallback_key != record_key[:len(fallback_key)]:
        return None
    if len(index_key) != len(record_key):
        return None
    return tuple(index_key[:len(fallback_key)])


def _prefix_paths(record: dict, prefixes) -> list:
    """Expand each declared PREFIX into the concrete leaf paths present in this
    record (used for blocks like `condition_vector.*` whose members vary)."""
    out = []
    for pre in prefixes:
        sub = resolve_key_path(record, pre)
        if sub is _MISSING or not isinstance(sub, dict):
            continue
        for leaf in leaf_paths(sub, pre):
            out.append(leaf)
    return out


def load_bundle(indir: Path) -> dict:
    """Load every registered artifact, pruning JSONL records to the declared key
    paths (plus the declared prefixes expanded per record)."""
    declared: dict = {}
    prefixes: dict = {}
    for ad in ADAPTERS:
        art = ad["artifact"]
        declared.setdefault(art, set()).update(ad["declared_keys"])
        prefixes.setdefault(art, set()).update(ad.get("declared_prefixes", ()))

    artifacts: dict = {}
    missing = []
    for name, spec in ARTIFACTS.items():
        path = indir.joinpath(name)
        if not path.exists():
            missing.append(name)
            artifacts[name] = {"kind": spec["kind"], "record_key": spec["record_key"],
                               "data": None, "available": False}
            continue
        if spec["kind"] == "json":
            artifacts[name] = {"kind": "json", "record_key": None,
                               "data": _read_json(path), "available": True}
            continue
        keys = sorted(declared.get(name, ()))
        pres = sorted(prefixes.get(name, ()))
        rk = spec["record_key"]
        fbk = spec.get("record_key_fallback")
        index: dict = {}
        fallback_index: dict = {}
        for rec in _iter_jsonl(path):
            key = _record_index_key(rec, rk)
            if key is None:
                continue
            wanted = list(keys)
            if pres:
                wanted.extend(_prefix_paths(rec, pres))
            index[key] = _prune(rec, wanted)
            if fbk:
                # FIRST record wins, in file order — mirroring the upstream
                # `setdefault` this reproduces (m7_assemble.py:1009).
                fk = _fallback_index_key(key, fbk, rk)
                if fk is not None:
                    fallback_index.setdefault(fk, key)
        artifacts[name] = {"kind": "jsonl", "record_key": rk,
                           "data": index, "fallback_index": fallback_index,
                           "available": True}

    rollup = artifacts.get("m7_rollup.json", {}).get("data") or {}
    return {
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "engine_build_id": rollup.get("engine_build_id"),
        "indir": str(indir),
    }


def bundle_resolve(bundle: dict, artifact: str, record_key, key_path: str):
    """Resolve (artifact, record_key, key_path) against the loaded bundle."""
    entry = (bundle.get("artifacts") or {}).get(artifact)
    if not entry or entry.get("data") is None:
        return _MISSING
    data = entry["data"]
    if entry["kind"] == "jsonl":
        if record_key is None:
            return _MISSING
        rec = data.get(record_key)
        if rec is None:
            return _MISSING
        return resolve_key_path(rec, key_path)
    return resolve_key_path(data, key_path)


# ========================================================================== #
# NODE / EDGE CONSTRUCTION
#
# Nodes are emitted COMPACT: empty containers are omitted (absent == empty on
# read-back). At ~150k nodes over the corpus that is the difference between a
# usable artifact and an unusable one, and it loses no information.
#
# Note the `source` block carries `study_ref` (the CPAD pmid) rather than the
# whole study record: the full source_study sits ONCE per record in the
# `provenance` header, and duplicating author/reference strings across every
# node would inflate the artifact for no traceability gain. The reference still
# resolves — which is what provenance requires.
# ========================================================================== #
def _slug(text: str) -> str:
    out = []
    for ch in str(text).lower():
        out.append(ch if (ch.isalnum() or ch in "._-") else "_")
    return "".join(out)


def make_node(kind: str, adapter: str, slug: str, label: str, template: str,
              value=None, source_ref=None, series_id=None, study_ref=None,
              method=None, assumptions=(), confidence=(), explains=(),
              causal_license="none", detail=None, value_absent_reason=None) -> dict:
    """Build one typed node. `value` MUST be a verbatim artifact read; anything
    else will fail `assert_no_derivation`."""
    node = {
        "node_id": ":".join((kind, adapter, _slug(slug))),
        "kind": kind,
        "label": label,
        "template": template,
    }
    if causal_license != "none":          # "none" is the default; omit it
        node["causal_license"] = causal_license
    if value is _MISSING or value_absent_reason is not None:
        node["value"] = None
        node["value_kind"] = "absent"
        node["value_absent_reason"] = value_absent_reason or "key_path_did_not_resolve"
    else:
        node["value"] = value
        # `value_kind` is omitted when it is the default "artifact_verbatim":
        # absent == default, declared in the artifact header.
        if value is None:
            node["value_kind"] = "absent"
            node["value_absent_reason"] = "artifact_stores_null"
    if source_ref:
        # A node's SOURCE is: this resolvable pointer, plus the record's
        # `series_id` and `provenance.source_study` (the CPAD study). The series
        # and study are identical for every node in a record and a node is not
        # meaningful outside its record, so storing them per-node was pure
        # duplication — the same "derived, not stored" rule applied to the
        # projections. `node_source(graph, node)` reassembles the full block.
        node["source_ref"] = source_ref
    node["method"] = method or {"module": None, "function": None, "version": None,
                                "fit_provenance": FIT_PROVENANCE_ABSENT}
    if node["label"] == _slug(slug):      # label adds nothing over the node_id
        node.pop("label")
    if detail is not None:
        node["detail"] = detail
    if assumptions:
        node["assumptions"] = list(assumptions)
    if confidence:
        node["confidence"] = list(confidence)
    if explains:
        node["explains"] = sorted(set(explains))
    return node


def make_edge(src: str, dst: str, kind: str, adapter: str, source_ref=None,
              gain_basis=None, causal_license="none", template=None) -> dict:
    edge = {
        "from": src,
        "to": dst,
        "kind": kind,
        "adapter": adapter,
    }
    if template is not None and template != kind:   # default: template == kind
        edge["template"] = template
    if causal_license != "none":          # "none" is the default; omit it
        edge["causal_license"] = causal_license
    if source_ref:
        edge["source_ref"] = source_ref
    if gain_basis:
        edge["gain_basis"] = gain_basis
    return edge


def make_confidence(kind: str, value, basis: str, valid_at: str) -> dict:
    """One tagged confidence. Always used inside a LIST — there is no code path
    in this module that reduces a list of these to a scalar."""
    return {"kind": kind, "value": value, "basis": basis, "valid_at": valid_at}


def _ref(artifact: str, record_key, key_path: str) -> dict:
    """Build a source_ref.

    `artifact` is the BARE filename, never a path: the directory is constant
    (`ARTIFACT_DIR`, recorded once in the artifact header), and storing a path
    on every one of ~175k nodes would both bloat the artifact and reintroduce
    the platform-separator hazard the canonical encoding exists to avoid."""
    return {"artifact": artifact, "record_key": record_key, "key_path": key_path}


def _artifact_name(ref: dict) -> str:
    """The artifact filename from a source_ref."""
    return (ref or {}).get("artifact") or ""


# ========================================================================== #
# CANONICAL CROSS-ADAPTER NODE IDS
#
# Adapters emit edges that point at nodes OWNED BY ANOTHER ADAPTER (the M5
# refusal is blocked by an M1 metadata fact and unblocked by an M11 action).
# Node ids are deterministic, so this works — but the ids must be agreed in ONE
# place rather than re-spelled at each site. The graph builder DROPS any edge
# whose endpoint is absent and records it in `graph["dropped_edges"]`; a test
# asserts that list is empty over the live corpus, so a typo here fails loudly
# instead of quietly thinning the graph.
# ========================================================================== #
NID_REFUSAL_MECHANISM = "refusal:m5_regimes:single_mechanism_call"
NID_GATE_MECH_LICENSE = "gate:m5_regimes:mechanistic_inference_licensed"
NID_DEFERRAL_SERVICE_C = "deferral:m5_regimes:service_c_confusion_matrix"
NID_META_AGITATION = "metadata_fact:m1_triage:condition_vector.field_provenance.agitation"
NID_META_SEEDED = "metadata_fact:m1_triage:condition_vector.field_provenance.seeded"
NID_DATUM_ASSAY_MASS = "datum:m1_triage:condition_vector.assay_reports_mass"
NID_ASSUMPTION_DYE_MASS = "assumption:m1_triage:dye_signal_proportional_to_mass"
NID_RESULT_T50 = "result:m4_features:features.t50"
NID_METHOD_M2_MODEL = "method:m2_fit:best_by_aicc"
NID_CEILING_M8 = "ceiling:m8_reachability:validity_ceiling"
NID_BLOCKER_AGIT_SEED = "datum:m8_reachability:blocker.agitation_seeding_unknown"
NID_BLOCKER_NO_SERIES = "datum:m8_reachability:blocker.no_concentration_series"
NID_BLOCKER_NON_MASS = "datum:m8_reachability:blocker.non_mass_or_unknown_assay"
NID_RESULT_REGIME = "result:m5_regimes:descriptive_regime.regime"

# M5 gate name -> node id(s) of the artifact fact that gate reads. This lets the
# M5 adapter attach `blocked_by` edges down to the real leaf without re-deriving
# WHY the gate failed: the mapping is a restatement of m5_classify's own gate
# construction, not a new inference.
_M5_GATE_TO_EVIDENCE = {
    "agitation_or_seeding_unknown_branch_undetermined": (NID_BLOCKER_AGIT_SEED,),
    "assay_not_mass_proportional": (NID_BLOCKER_NON_MASS,),
    "assay_mass_proportionality_unknown": (NID_BLOCKER_NON_MASS,),
    # `no_nucleated_transition_in_shape` is a SHAPE fact, not a metadata fact:
    # its evidence is the descriptive regime, which M5 itself emits.
    "no_nucleated_transition_in_shape": (NID_RESULT_REGIME,),
}

# M8 blocker -> the leaf nodes it stands over. WHICH leaves is what M8's own
# blocker `detail` states in prose; this is that prose made machine-readable.
_M8_BLOCKER_EVIDENCE = {
    "agitation_seeding_unknown": (NID_META_AGITATION, NID_META_SEEDED),
    "non_mass_or_unknown_assay": (NID_DATUM_ASSAY_MASS,),
    "no_concentration_series": (),   # the evidence is an ABSENT series id
}


# ========================================================================== #
# ADAPTER CONTEXT
# ========================================================================== #
class Ctx:
    """Per-series adapter context: resolves declared key paths against the
    loaded bundle and stamps every read with a source_ref."""

    def __init__(self, series_id, bundle, join_keys=None):
        self.series_id = series_id
        self.bundle = bundle
        self.join_keys = join_keys or {}
        self.study_ref = None

    def _composite_key(self, members):
        """Assemble a composite record key from `join_keys`. `join_keys` is
        keyed by the TARGET-SIDE key paths (the members of `record_key`), so the
        index side and the lookup side cannot drift apart. Any missing or null
        member => None: two thirds of a triple is NOT a key, and resolving on a
        partial key is precisely the uniprot-only join that bleeds one
        construct's propensity onto another curve."""
        parts = []
        for member in members:
            v = self.join_keys.get(member)
            if v is None:
                return None
            parts.append(v)
        return tuple(parts)

    def record_key_for(self, artifact):
        spec = ARTIFACTS.get(artifact) or {}
        if spec.get("kind") != "jsonl":
            return None
        rk = spec.get("record_key")
        if not isinstance(rk, tuple):                 # SCALAR — unchanged
            if rk == "series_id":
                return self.series_id
            return self.join_keys.get(rk)
        key = self._composite_key(rk)
        if key is None:
            return None
        entry = (self.bundle.get("artifacts") or {}).get(artifact) or {}
        data = entry.get("data")
        if isinstance(data, dict) and key in data:
            return key
        fbk = spec.get("record_key_fallback")
        if not fbk:
            return key            # a miss is a miss; the ref simply won't resolve
        fk = self._composite_key(fbk)
        if fk is None:
            return key
        # MIRRORING m7_assemble.py:1045-1046. The returned key is the REAL key of
        # the record the engine used, so the source_ref addresses that record
        # and nothing is invented — including when the engine's own join was
        # wrong (see the m6_assay_bleed_fixed_m17_mirrors_the_corrected_join narrowing).
        return (entry.get("fallback_index") or {}).get(fk) or key

    def get(self, artifact, key_path):
        """Return (value, source_ref). Value is `_MISSING` when unresolved."""
        rk = self.record_key_for(artifact)
        val = bundle_resolve(self.bundle, artifact, rk, key_path)
        return val, _ref(artifact, rk, key_path)

    def record(self, artifact):
        entry = (self.bundle.get("artifacts") or {}).get(artifact)
        if not entry or entry.get("data") is None:
            return None
        if entry["kind"] == "json":
            return entry["data"]
        rk = self.record_key_for(artifact)
        if rk is None:
            return None
        return entry["data"].get(rk)


def _emit_leaves(ctx, nodes, adapter, artifact, origin_prefix, spine_prefix,
                 kind, template, method, causal_license="none",
                 kind_for=None, template_for=None):
    """Emit one typed node per LEAF under `origin_prefix`, each declaring the
    matching spine path under `spine_prefix` as the field it explains.

    This is how a whole emitted block (condition_vector, the M4 feature vector,
    the gamma payload) becomes nodes without hand-listing every member — while
    still keeping ONE NODE PER EMITTED FIELD, as the design requires."""
    rec = ctx.record(artifact)
    if rec is None:
        return
    sub = resolve_key_path(rec, origin_prefix)
    if sub is _MISSING or not isinstance(sub, dict):
        return
    for path in leaf_paths(sub, origin_prefix):
        val, ref = ctx.get(artifact, path)
        if val is _MISSING:
            continue
        tail = path[len(origin_prefix):] if path.startswith(origin_prefix) else path
        spine = "".join((spine_prefix, tail)) if spine_prefix is not None else None
        k = kind_for(path, val) if kind_for else kind
        t = template_for(path, val) if template_for else template
        nodes.append(make_node(
            k, adapter, path, path, t, value=val, source_ref=ref,
            series_id=ctx.series_id, study_ref=ctx.study_ref, method=method,
            explains=(spine,) if spine else (), causal_license=causal_license))


# ========================================================================== #
# ADAPTER 1 — M1 triage (curves_triaged.jsonl)
#
# The LEAF LAYER. Every `blocked_by META agitation = "unknown"` edge in the
# corpus bottoms out here, at condition_vector.field_provenance.agitation. Also
# supplies the CPAD study provenance (pmid) that every node's `source` cites.
# ========================================================================== #
_M1_METHOD = {"module": "m1_ingest", "function": "triage_series",
              "version": "m1-triage", "fit_provenance": FIT_PROVENANCE_ABSENT}

_M1_STATIC = (
    "series_id", "protein_id", "uniprot_id", "source", "data_mode",
    "concentration_series_id", "signal_basis", "digitization_uncertainty",
    "n_points", "source_study.pmid",
    "m1.fittability_class", "m1.censoring_class", "m1.recommended_handling",
    "m1.normalization_mode", "m1.signal_basis", "m1.reasons", "m1.flat_no_signal",
)

# spine path <- origin key path. Every pair here was verified VALUE-IDENTICAL
# across the whole live corpus before being declared (0 mismatches over 1,194
# joinable series); the test suite re-verifies it, so an upstream change that
# breaks the identity fails loudly rather than silently mis-attributing.
_M1_DIRECT = (
    ("fittability_class", "m1.fittability_class"),
    ("censoring_class", "m1.censoring_class"),
    ("data_mode", "data_mode"),
    ("source", "source"),
    ("protein_id", "protein_id"),
    ("uniprot_id", "uniprot_id"),
    ("series_id", "series_id"),
    ("quality_report.artifact_flags", "m1.reasons"),
    ("quality_report.signal_basis", "signal_basis"),
    ("quality_report.normalization_mode", "m1.normalization_mode"),
    ("quality_report.digitization_uncertainty", "digitization_uncertainty"),
)


def _adapter_m1(ctx):
    nodes, edges = [], []
    rec = ctx.record("curves_triaged.jsonl")
    if rec is None:
        return nodes, edges
    pmid, _pref = ctx.get("curves_triaged.jsonl", "source_study.pmid")
    ctx.study_ref = None if pmid is _MISSING else pmid

    def kind_for(path, val):
        return "metadata_fact" if ".field_provenance." in path else "datum"

    def template_for(path, val):
        if ".field_provenance." in path:
            return "metadata_fact.unknown" if val != "known" else "metadata_fact.known"
        return "datum"

    _emit_leaves(ctx, nodes, "m1_triage", "curves_triaged.jsonl",
                 "condition_vector", "condition_vector", None, None, _M1_METHOD,
                 kind_for=kind_for, template_for=template_for)
    _emit_leaves(ctx, nodes, "m1_triage", "curves_triaged.jsonl",
                 "quality_flags", "quality_report.quality_flags", "datum",
                 "datum", _M1_METHOD)

    for spine, origin in _M1_DIRECT:
        val, ref = ctx.get("curves_triaged.jsonl", origin)
        if val is _MISSING:
            continue
        nodes.append(make_node("datum", "m1_triage", origin, origin, "datum",
                               value=val, source_ref=ref, series_id=ctx.series_id,
                               study_ref=ctx.study_ref, method=_M1_METHOD,
                               explains=(spine,)))

    # ---- invariant §10.9: the dye-signal-proportional-to-mass ASSUMPTION ----
    # Mechanistic models are licensed only on mass-proportional assays, and a
    # ThT/ThS signal is NOT guaranteed proportional to fibril mass. Whether the
    # assay gate passes or fails, that proportionality is TAKEN AS TRUE and
    # never measured — so it is an `assumption` node, hanging off the assay
    # datum. This is why the assumption appears even on series whose assay gate
    # PASSES: a passed gate is not a verified assumption.
    assay, assay_ref = ctx.get("curves_triaged.jsonl", "condition_vector.assay_type")
    if assay is not _MISSING:
        nodes.append(make_node(
            "assumption", "m1_triage", "dye_signal_proportional_to_mass",
            "assay signal is proportional to fibril mass",
            "assumption.unverified", value=None, source_ref=assay_ref,
            series_id=ctx.series_id, study_ref=ctx.study_ref, method=_M1_METHOD,
            value_absent_reason="an assumption has no measured value, by definition",
            assumptions=("PRISE_DESIGN.md §10.9",),
            detail=("mechanistic models are licensed only on fibril-reporting AND "
                    "mass-proportional assays; a dye signal is NOT guaranteed "
                    "proportional to fibril mass (polymorph-, surface- and "
                    "concentration-dependent). Unverified for this assay.")))
        if any(n["node_id"] == NID_DATUM_ASSAY_MASS for n in nodes):
            edges.append(make_edge(NID_DATUM_ASSAY_MASS, NID_ASSUMPTION_DYE_MASS,
                                   "assumes", "m1_triage", source_ref=assay_ref))
    return nodes, edges


# ========================================================================== #
# ADAPTER 2 — M2 fit bank (fits.jsonl)
#
# Supplies the `method` node that every M4 feature is `computed_from`. It claims
# NO spine field of its own (M3 owns model_selection in the M7 spine), and that
# is deliberate: its job is to give the evidence chain a real procedure to point
# at, not to inflate the coverage numerator.
#
# The per-model key paths (fits.<model>.r2) are DYNAMIC — the model name is
# itself read from the artifact (`best_by_aicc`). They are therefore validated
# by source-ref RESOLUTION rather than by the static declared-keys contract,
# which is why `fits` is declared as a prefix and not as a fixed path.
# ========================================================================== #
_M2_METHOD = {"module": "m2_fit", "function": "fit_curve",
              "version": "m2-fitbank", "fit_provenance": FIT_PROVENANCE_M2}

# The known Stage-B blind spot: m2_fit.py:152 collapses THREE distinct causes
# (mismatched/malformed arrays, fewer than 3 points, an empty candidate bank)
# into ONE string. No adapter can disambiguate it from the artifact, so M17
# reports the ambiguity instead of guessing which cause applied.
M2_COLLAPSED_REASON = "not fitted (rejected / malformed / no candidates)"


def _adapter_m2(ctx):
    nodes, edges = [], []
    best, best_ref = ctx.get("fits.jsonl", "best_by_aicc")
    if best is _MISSING:
        return nodes, edges
    handling, h_ref = ctx.get("fits.jsonl", "handling")
    note, note_ref = ctx.get("fits.jsonl", "note")

    nodes.append(make_node(
        "method", "m2_fit", "best_by_aicc", "AICc-best fitted model",
        "method.model", value=best, source_ref=best_ref, series_id=ctx.series_id,
        study_ref=ctx.study_ref, method=_M2_METHOD,
        detail="AICc here is preliminary; principled selection is M3."))
    if handling is not _MISSING:
        nodes.append(make_node(
            "datum", "m2_fit", "handling",
            "M1 recommended handling (routes the candidate bank)", "datum",
            value=handling, source_ref=h_ref, series_id=ctx.series_id,
            study_ref=ctx.study_ref, method=_M2_METHOD))
        edges.append(make_edge(NID_METHOD_M2_MODEL, "datum:m2_fit:handling",
                               "computed_from", "m2_fit", source_ref=h_ref))
    if isinstance(best, str):
        for metric in ("r2", "aicc", "converged"):
            path = ".".join(("fits", best, metric))
            val, ref = ctx.get("fits.jsonl", path)
            if val is _MISSING:
                continue
            nodes.append(make_node(
                "datum", "m2_fit", path, path, "datum", value=val, source_ref=ref,
                series_id=ctx.series_id, study_ref=ctx.study_ref, method=_M2_METHOD))
            edges.append(make_edge(NID_METHOD_M2_MODEL,
                                   ":".join(("datum", "m2_fit", _slug(path))),
                                   "computed_from", "m2_fit", source_ref=ref))
    # ---- the known Stage-B ambiguity, REPORTED rather than resolved ---------
    if note is not _MISSING and note == M2_COLLAPSED_REASON:
        nodes.append(make_node(
            "deferral", "m2_fit", "collapsed_failure_reason",
            "M2 failure cause is not recoverable from the artifact",
            "deferral.ambiguous_cause", value=note, source_ref=note_ref,
            series_id=ctx.series_id, study_ref=ctx.study_ref, method=_M2_METHOD,
            detail=("m2_fit.py:152 emits ONE string for THREE distinct causes "
                    "(mismatched/malformed arrays; fewer than 3 points; an empty "
                    "candidate bank). The cause exists only in that function local "
                    "scope and never reaches an artifact, so no adapter can "
                    "disambiguate it. Stage B must widen the upstream record.")))
    return nodes, edges


# ========================================================================== #
# ADAPTER 3 — M4 feature extractor (features.jsonl)
#
# The NUMBER layer: without real floats, "every number byte-compares to its
# artifact" would be a slogan rather than a test.
# ========================================================================== #
_M4_METHOD = {"module": "m4_features", "function": "extract_features",
              "version": "m4-defs-1.0", "fit_provenance": FIT_PROVENANCE_ABSENT}

_M4_DIRECT = (
    ("curve_features.t50_status", "t50_status"),
    ("curve_features.lag_status", "lag_status"),
    ("curve_features.validity_flags", "validity_flags"),
    ("curve_features.status", "status"),
    ("curve_features.signal_basis", "signal_basis"),
    ("curve_features.censoring_class", "censoring_class"),
    ("curve_features.definition_contract", "definition_contract"),
)


def _adapter_m4_features(ctx):
    nodes, edges = [], []
    if ctx.record("features.jsonl") is None:
        return nodes, edges

    _emit_leaves(ctx, nodes, "m4_features", "features.jsonl", "features",
                 "curve_features.features", "result", "result", _M4_METHOD)
    # `lag_to_t50_ratio` is ALSO emitted at the spine top level (verified
    # identical corpus-wide), so the SAME node explains both paths.
    for n in nodes:
        if n["node_id"] == "result:m4_features:features.lag_to_t50_ratio":
            n["explains"] = sorted(set(list(n.get("explains", ())) + ["lag_to_t50_ratio"]))

    for spine, origin in _M4_DIRECT:
        val, ref = ctx.get("features.jsonl", origin)
        if val is _MISSING:
            continue
        kind = ("uncertainty" if origin in ("t50_status", "lag_status",
                                            "validity_flags") else "datum")
        tpl = "uncertainty.censoring_bound" if kind == "uncertainty" else "datum"
        nodes.append(make_node(kind, "m4_features", origin, origin, tpl,
                               value=val, source_ref=ref, series_id=ctx.series_id,
                               study_ref=ctx.study_ref, method=_M4_METHOD,
                               explains=(spine,)))

    have = set()
    for n in nodes:
        have.add(n["node_id"])
    # every feature is computed from the M2-selected model
    results = []
    for n in nodes:
        if n["kind"] == "result":
            results.append(n["node_id"])
    for nid in results:
        edges.append(make_edge(nid, NID_METHOD_M2_MODEL, "computed_from",
                               "m4_features"))
    # t50 carries a CENSORING bound — a censored t50 is a bound, never a clean
    # value (invariant §10.6)
    if NID_RESULT_T50 in have and "uncertainty:m4_features:t50_status" in have:
        _v, r = ctx.get("features.jsonl", "t50_status")
        edges.append(make_edge(NID_RESULT_T50, "uncertainty:m4_features:t50_status",
                               "bounded_by", "m4_features", source_ref=r))
    if "uncertainty:m4_features:validity_flags" in have:
        for nid in results:
            edges.append(make_edge(nid, "uncertainty:m4_features:validity_flags",
                                   "uncertainty_from", "m4_features"))
    return nodes, edges


# ========================================================================== #
# ADAPTER 4 — M4 dual-gamma (gamma.jsonl, joined on concentration_series_id)
#
# The only source of the `bootstrap_ci` confidence kind — the most canonical
# confidence in the engine. gamma exists for 52 concentration series only, so
# this adapter is deliberately PARTIAL and the coverage artifact says so.
#
# Note gamma_regression and gamma_global are NEVER merged here, mirroring
# invariant §10.5: two separately-provenanced estimators, two nodes.
# ========================================================================== #
_GAMMA_METHOD = {"module": "m4_features", "function": "dual_gamma",
                 "version": "m4-defs-1.0", "fit_provenance": FIT_PROVENANCE_ABSENT}


def _adapter_m4_gamma(ctx):
    nodes, edges = [], []
    if ctx.record("gamma.jsonl") is None:
        return nodes, edges
    for origin_prefix, spine_prefix in (("gamma_regression", "gamma_regression"),
                                        ("gamma_global", "gamma_global"),
                                        ("disagreement", "gamma_disagreement")):
        _emit_leaves(ctx, nodes, "m4_gamma", "gamma.jsonl", origin_prefix,
                     spine_prefix, "result", "result", _GAMMA_METHOD)

    have = set()
    for n in nodes:
        have.add(n["node_id"])

    # ---- the BOOTSTRAP CI, as a tagged confidence on the gamma result -------
    for est, ci_path, reliable_path in (
            ("gamma_regression", "gamma_regression.gamma_ci",
             "gamma_regression.gamma_reliable"),
            ("gamma_global", "gamma_global.gamma_ci", None)):
        gid = ":".join(("result", "m4_gamma", _slug(".".join((est, "gamma")))))
        if gid not in have:
            continue
        ci, ci_ref = ctx.get("gamma.jsonl", ci_path)
        if ci is _MISSING:
            continue
        est_name, est_ref = ctx.get("gamma.jsonl", ".".join((est, "estimator")))
        for n in nodes:
            if n["node_id"] != gid:
                continue
            n["confidence"] = [make_confidence(
                "bootstrap_ci", ci,
                "wild-bootstrap percentile interval, CONDITIONAL on the selected "
                "model; asymptotic, not finite-sample valid",
                "the measured concentration window of this series only")]
        cid = ":".join(("result", "m4_gamma", _slug(ci_path)))
        if cid in have:
            edges.append(make_edge(gid, cid, "uncertainty_from", "m4_gamma",
                                   source_ref=ci_ref))
        if reliable_path:
            rid = ":".join(("result", "m4_gamma", _slug(reliable_path)))
            if rid in have:
                edges.append(make_edge(gid, rid, "gated_by", "m4_gamma"))
    return nodes, edges


# ========================================================================== #
# ADAPTER 5 — M5 classification & regime registry (regimes.jsonl)
#
# The REFUSAL, the GATE, the DEFERRAL and the EQUIVALENCE CLASS. This is the
# spine of the worked example. Note `regimes.jsonl` holds 1,240 records against
# the M7 spine's 1,654: for the other 414 series there is no refusal to explain
# because the engine produced none — a different thing from M17 lacking an
# adapter, and reported separately in explanation_coverage.json.
# ========================================================================== #
_M5_METHOD = {"module": "m5_classify", "function": "classify_series",
              "version": "m5-regimes-1.0", "fit_provenance": FIT_PROVENANCE_ABSENT}

_M5_STATIC = (
    "series_id", "registry_version", "analysis_scope",
    "descriptive_regime.regime", "descriptive_regime.definition",
    "descriptive_regime.evidence", "descriptive_regime.registry_version",
    "mechanistic.single_mechanism_call", "mechanistic.single_mechanism_refused_reason",
    "mechanistic.equivalence_class", "mechanistic.equivalence_class_narrowed",
    "mechanistic.equivalence_class_provenance", "mechanistic.degeneracy",
    "mechanistic.degeneracy_note", "mechanistic.mechanistic_inference_licensed",
    "mechanistic.gates_failed", "mechanistic.data_regime_context",
    "confidence.level", "confidence.r2", "confidence.n_points", "confidence.basis",
    "window_of_validity.t_min_hours", "window_of_validity.t_max_hours",
    "window_of_validity.censoring_class", "window_of_validity.note",
)


def _adapter_m5(ctx):
    nodes, edges = [], []
    if ctx.record("regimes.jsonl") is None:
        return nodes, edges
    art = "regimes.jsonl"

    # ---- descriptive regime (always available; never erased by anomaly) -----
    _emit_leaves(ctx, nodes, "m5_regimes", art, "descriptive_regime",
                 "descriptive_regime", "result", "result", _M5_METHOD)
    # M5's ordinal shape confidence explains BOTH spine locations (verified
    # value-identical corpus-wide): descriptive_regime.confidence.* AND
    # shape_confidence.* — M7 carries the same object twice, deliberately, so a
    # consumer cannot mistake it for a mechanistic confidence.
    _emit_leaves(ctx, nodes, "m5_regimes", art, "confidence",
                 "descriptive_regime.confidence", "result", "result", _M5_METHOD)
    for n in nodes:
        nid = n["node_id"]
        if nid.startswith("result:m5_regimes:confidence."):
            tail = nid.rsplit(".", 1)[-1]
            n["explains"] = sorted(set(list(n.get("explains", ())) +
                                       [".".join(("shape_confidence", tail))]))

    level, level_ref = ctx.get(art, "confidence.level")
    basis, _b = ctx.get(art, "confidence.basis")
    for n in nodes:
        if n["node_id"] == NID_RESULT_REGIME and level is not _MISSING:
            n["confidence"] = [make_confidence(
                "shape_confidence_ordinal", level,
                _text(basis) if basis is not _MISSING else "M5 shape confidence",
                "SHAPE only — this is NOT a mechanistic confidence; mechanistic "
                "confidence is null while the mechanism verdict is unlicensed")]

    # ---- the window of validity: a CEILING on the claim ---------------------
    _emit_leaves(ctx, nodes, "m5_regimes", art, "window_of_validity",
                 "window_of_validity", "ceiling", "ceiling.window", _M5_METHOD)

    # ---- the mechanistic block: refusal / gate / class / deferral -----------
    call, call_ref = ctx.get(art, "mechanistic.single_mechanism_call")
    reason, reason_ref = ctx.get(art, "mechanistic.single_mechanism_refused_reason")
    licensed, lic_ref = ctx.get(art, "mechanistic.mechanistic_inference_licensed")
    eq, eq_ref = ctx.get(art, "mechanistic.equivalence_class")
    narrowed, nar_ref = ctx.get(art, "mechanistic.equivalence_class_narrowed")
    degen, deg_ref = ctx.get(art, "mechanistic.degeneracy")
    gates, gates_ref = ctx.get(art, "mechanistic.gates_failed")

    if call is not _MISSING:
        nodes.append(make_node(
            "refusal", "m5_regimes", "single_mechanism_call",
            "single_mechanism_call", "refusal.mechanism", value=call,
            source_ref=call_ref, series_id=ctx.series_id, study_ref=ctx.study_ref,
            method=_M5_METHOD,
            detail=_text(reason) if reason is not _MISSING else "",
            causal_license="deterministic_engine_logic"))
    if licensed is not _MISSING:
        nodes.append(make_node(
            "gate", "m5_regimes", "mechanistic_inference_licensed",
            "mechanistic_inference_licensed",
            "gate.passed" if licensed else "gate.failed", value=licensed,
            source_ref=lic_ref, series_id=ctx.series_id, study_ref=ctx.study_ref,
            method=_M5_METHOD, explains=("mechanistic.mechanistic_inference_licensed",),
            causal_license="deterministic_engine_logic"))
        edges.append(make_edge(NID_REFUSAL_MECHANISM, NID_GATE_MECH_LICENSE,
                               "gated_by", "m5_regimes", source_ref=lic_ref,
                               causal_license="deterministic_engine_logic"))
    if eq is not _MISSING:
        nodes.append(make_node(
            "result", "m5_regimes", "mechanistic.equivalence_class",
            "mechanistic equivalence class", "result.equivalence_class", value=eq,
            source_ref=eq_ref, series_id=ctx.series_id, study_ref=ctx.study_ref,
            method=_M5_METHOD, explains=("mechanistic.verdict_or_equivalence_class",),
            detail=_text(narrowed) if narrowed is not _MISSING else "<unavailable>"))
        edges.append(make_edge(NID_REFUSAL_MECHANISM,
                               "result:m5_regimes:mechanistic.equivalence_class",
                               "computed_from", "m5_regimes", source_ref=eq_ref))
    if narrowed is not _MISSING:
        nodes.append(make_node(
            "datum", "m5_regimes", "mechanistic.equivalence_class_narrowed",
            "equivalence_class_narrowed", "datum", value=narrowed,
            source_ref=nar_ref, series_id=ctx.series_id, study_ref=ctx.study_ref,
            method=_M5_METHOD))
    if degen is not _MISSING:
        nodes.append(make_node(
            "datum", "m5_regimes", "mechanistic.degeneracy", "degeneracy",
            "datum", value=degen, source_ref=deg_ref, series_id=ctx.series_id,
            study_ref=ctx.study_ref, method=_M5_METHOD,
            explains=("mechanistic.degeneracy",)))
    if reason is not _MISSING:
        # The refusal names a NOT-YET-BUILT dependency: the Service-C confusion
        # matrix + the FDR single-mechanism degeneracy threshold (§4).
        nodes.append(make_node(
            "deferral", "m5_regimes", "service_c_confusion_matrix",
            "Service-C confusion matrix + FDR single-mechanism degeneracy "
            "threshold (§4)", "deferral.named", value=reason,
            source_ref=reason_ref, series_id=ctx.series_id,
            study_ref=ctx.study_ref, method=_M5_METHOD,
            detail=_text(reason)))
        edges.append(make_edge(NID_REFUSAL_MECHANISM, NID_DEFERRAL_SERVICE_C,
                               "blocked_by", "m5_regimes", source_ref=reason_ref,
                               causal_license="deterministic_engine_logic"))
    if gates is not _MISSING and isinstance(gates, list):
        nodes.append(make_node(
            "datum", "m5_regimes", "mechanistic.gates_failed", "gates_failed",
            "datum", value=gates, source_ref=gates_ref, series_id=ctx.series_id,
            study_ref=ctx.study_ref, method=_M5_METHOD,
            explains=("mechanistic.gates_failed",)))
        for i, gname in enumerate(gates):
            path = "".join(("mechanistic.gates_failed[", str(i), "]"))
            gval, gref = ctx.get(art, path)
            if gval is _MISSING:
                continue
            gid = ":".join(("gate", "m5_regimes", _slug(gval)))
            nodes.append(make_node(
                "gate", "m5_regimes", gval, gval, "gate.failed", value=gval,
                source_ref=gref, series_id=ctx.series_id, study_ref=ctx.study_ref,
                method=_M5_METHOD, causal_license="deterministic_engine_logic"))
            edges.append(make_edge(NID_GATE_MECH_LICENSE, gid, "blocked_by",
                                   "m5_regimes", source_ref=gref,
                                   causal_license="deterministic_engine_logic"))
            for ev in _M5_GATE_TO_EVIDENCE.get(gval, ()):
                edges.append(make_edge(gid, ev, "blocked_by", "m5_regimes",
                                       causal_license="deterministic_engine_logic"))
    return nodes, edges


# ========================================================================== #
# ADAPTER 6 — M8 inferential reachability (m8_reachability.json)
#
# CORPUS-WIDE facts spliced into every series graph: the mandatory validity
# ceiling, and the impact-ranked blocker census. M8's artifact already carries
# the materialised M11 cross-reference (`blocker.m11_explanation`), so the
# blocker -> field -> action chain is READ, not reconstructed.
# ========================================================================== #
_M8_METHOD = {"module": "m8_reachability", "function": "build_reachability",
              "version": "m8-reachability-1.0", "fit_provenance": FIT_PROVENANCE_ABSENT}

_M8_BLOCKERS = ("agitation_seeding_unknown", "no_concentration_series",
                "non_mass_or_unknown_assay")


def _m8_blocker_path(blocker, field):
    return "".join(("blocker_census.blockers_impact_ranked[blocker=", blocker,
                    "].", field))


_M8_STATIC = tuple(
    ["m8_version", "validity_ceiling.clause",
     "validity_ceiling.evidence_a_different_forward_model_degrades_resolution."
     "mechanism_misidentification_degradation",
     "mechanistically_resolvable_boundary.gap_structural_minus_licensed"]
    + [_m8_blocker_path(b, f) for b in _M8_BLOCKERS
       for f in ("count", "fraction", "blocks_rung", "scope", "detail",
                 "impact_rank", "m11_explanation.gain_basis",
                 "m11_explanation.m11_action_to_unblock")])


def _adapter_m8(ctx):
    nodes, edges = [], []
    art = "m8_reachability.json"
    clause, clause_ref = ctx.get(art, "validity_ceiling.clause")
    if clause is _MISSING:
        return nodes, edges
    nodes.append(make_node(
        "ceiling", "m8_reachability", "validity_ceiling",
        "M8 forward-model validity ceiling", "ceiling.validity", value=clause,
        source_ref=clause_ref, series_id=ctx.series_id, method=_M8_METHOD,
        detail=_text(clause)))

    for blocker in _M8_BLOCKERS:
        cnt, cnt_ref = ctx.get(art, _m8_blocker_path(blocker, "count"))
        if cnt is _MISSING:
            continue
        scope, _s = ctx.get(art, _m8_blocker_path(blocker, "scope"))
        rung, _r = ctx.get(art, _m8_blocker_path(blocker, "blocks_rung"))
        detail, _d = ctx.get(art, _m8_blocker_path(blocker, "detail"))
        bid = ":".join(("datum", "m8_reachability", _slug(".".join(("blocker", blocker)))))
        nodes.append(make_node(
            "datum", "m8_reachability", ".".join(("blocker", blocker)), blocker,
            "datum.blocker", value=cnt, source_ref=cnt_ref,
            series_id=ctx.series_id, method=_M8_METHOD,
            causal_license="deterministic_engine_logic",
            detail=_text({"scope": scope if scope is not _MISSING else None,
                          "blocks_rung": rung if rung is not _MISSING else None,
                          "m8_detail": detail if detail is not _MISSING else None})))
        for ev in _M8_BLOCKER_EVIDENCE.get(blocker, ()):
            edges.append(make_edge(bid, ev, "blocked_by", "m8_reachability",
                                   causal_license="deterministic_engine_logic"))
        # every blocker inherits the mandatory validity ceiling
        edges.append(make_edge(bid, NID_CEILING_M8, "bounded_by", "m8_reachability",
                               source_ref=clause_ref))
    return nodes, edges


# ========================================================================== #
# ADAPTER 7 — M11 metadata quality (metadata_quality.jsonl)
#
# Supplies (a) the PER-SERIES blocking prerequisites for the MECHANISM tier —
# richer than M8's corpus-wide census — and (b) the ACTION nodes the refusal is
# `unblocked_by`, each carrying the `gain_basis` that decides whether a
# QUANTITATIVE counterfactual may be rendered at all.
#
# VOCABULARY NOTE: the closed node vocabulary has no `action` kind, and Stage A
# does not widen it. An action is modelled as a `deferral` — a NAMED not-yet-
# available dependency which, if supplied, changes what is licensed. The
# renderer prints it as "ACTION"; the type stays closed.
# ========================================================================== #
_M11_METHOD = {"module": "m11_metadata", "function": "score_series",
               "version": "m11-metadata-1.0", "fit_provenance": FIT_PROVENANCE_ABSENT}

_M11_ACTION_FIELDS = ("agitation", "seeding", "concentration_series")


def _m11_action_path(field, attr):
    return "".join(("recommended_missing_metadata[field=", field, "].", attr))


_M11_STATIC = tuple(
    ["series_id", "version", "ontology_version",
     "metadata_completeness.metadata_score", "mechanistic_completeness",
     "mechanistic_blockers", "metadata_uncertainty.metadata_uncertainty",
     "metadata_uncertainty.caveat", "honesty.completeness_neq_correctness",
     "dependency_evaluation.MECHANISM.satisfied",
     "dependency_evaluation.MECHANISM.logic",
     "dependency_evaluation.MECHANISM.blocking_prerequisites",
     "dependency_evaluation.SCALING.satisfied",
     "dependency_evaluation.SCALING.blocking_prerequisites"]
    + [_m11_action_path(f, a) for f in _M11_ACTION_FIELDS
       for a in ("action", "gain_basis", "impact", "effort", "gain_over_effort",
                 "rationale", "blocks")])


def _adapter_m11(ctx):
    nodes, edges = [], []
    art = "metadata_quality.jsonl"
    if ctx.record(art) is None:
        return nodes, edges

    comp, comp_ref = ctx.get(art, "mechanistic_completeness")
    caveat, _c = ctx.get(art, "honesty.completeness_neq_correctness")
    if comp is not _MISSING:
        nodes.append(make_node(
            "metadata_fact", "m11_metadata", "mechanistic_completeness",
            "mechanistic_completeness", "metadata_fact.known", value=comp,
            source_ref=comp_ref, series_id=ctx.series_id, method=_M11_METHOD,
            confidence=[make_confidence(
                "completeness_presence", comp,
                _text(caveat) if caveat is not _MISSING else
                "completeness measures PRESENCE, not correctness",
                "this series' recorded metadata only; it cannot verify a value")]))

    # ---- per-series MECHANISM blocking prerequisites -----------------------
    prereqs, pre_ref = ctx.get(art, "dependency_evaluation.MECHANISM.blocking_prerequisites")
    if isinstance(prereqs, list):
        for i, name in enumerate(prereqs):
            path = "".join(("dependency_evaluation.MECHANISM.blocking_prerequisites[",
                            str(i), "]"))
            val, ref = ctx.get(art, path)
            if val is _MISSING:
                continue
            pid = ":".join(("metadata_fact", "m11_metadata",
                            _slug(".".join(("blocking_prerequisite", val)))))
            nodes.append(make_node(
                "metadata_fact", "m11_metadata",
                ".".join(("blocking_prerequisite", val)), val,
                "metadata_fact.blocking", value=val, source_ref=ref,
                series_id=ctx.series_id, method=_M11_METHOD,
                causal_license="deterministic_engine_logic",
                detail="blocks the MECHANISM tier for this series"))
            edges.append(make_edge(NID_REFUSAL_MECHANISM, pid, "blocked_by",
                                   "m11_metadata", source_ref=ref,
                                   causal_license="deterministic_engine_logic"))

    # ---- the ACTIONS that would unblock, each carrying its gain_basis ------
    # Collected action-first: several M11 fields can recommend the SAME action
    # (agitation and seeding both recommend record_agitation_and_seeding), and
    # that is one action, hence one node.
    by_action = OrderedDict()
    for field in _M11_ACTION_FIELDS:
        action, a_ref = ctx.get(art, _m11_action_path(field, "action"))
        if action is _MISSING:
            continue
        entry = by_action.get(action)
        if entry is None:
            gb, _g = ctx.get(art, _m11_action_path(field, "gain_basis"))
            impact, _i = ctx.get(art, _m11_action_path(field, "impact"))
            effort, _e = ctx.get(art, _m11_action_path(field, "effort"))
            by_action[action] = {
                "ref": a_ref, "fields": [field],
                "gain_basis": gb if gb is not _MISSING else None,
                "impact": impact if impact is not _MISSING else None,
                "effort": effort if effort is not _MISSING else None}
        else:
            entry["fields"].append(field)

    for action, info in by_action.items():
        gain_basis = info["gain_basis"]
        aid = ":".join(("deferral", "m11_metadata", _slug(".".join(("action", action)))))
        nodes.append(make_node(
            "deferral", "m11_metadata", ".".join(("action", action)), action,
            "deferral.action", value=action, source_ref=info["ref"],
            series_id=ctx.series_id, method=_M11_METHOD,
            detail=_text({"impact": info["impact"], "effort": info["effort"],
                          "gain_basis": gain_basis,
                          "recommended_for_fields": sorted(info["fields"])})))
        tpl = "unblocked_by"
        if gain_basis in ("service_c_measured", "structural_estimate"):
            tpl = ".".join(("unblocked_by", gain_basis))
        edges.append(make_edge(NID_REFUSAL_MECHANISM, aid, "unblocked_by",
                               "m11_metadata", source_ref=info["ref"],
                               gain_basis=gain_basis, template=tpl,
                               causal_license="deterministic_engine_logic"))
        # a structural_estimate gain is an UPPER BOUND and inherits M8's ceiling
        if gain_basis != "service_c_measured":
            edges.append(make_edge(aid, NID_CEILING_M8, "bounded_by",
                                   "m11_metadata"))
    return nodes, edges


# ========================================================================== #
# ADAPTER 8 — conformal prediction (service_c_conformal.json)
#
# The `conformal_quantile` confidence kind: distribution-free, finite-sample
# valid — a DIFFERENT guarantee from M4's asymptotic bootstrap CI, which is
# exactly why the two may never be averaged.
#
# NAMED STAGE-A NARROWING: M17 attaches the MARGINAL quantile verbatim and does
# NOT construct t50_hat ± q. Building the interval is arithmetic on engine
# values, and selecting the Mondrian stratum would require classifying the assay
# endpoint as amyloid-vs-generic — a judgement M17 is not licensed to make. The
# refusal is recorded in honesty_block() and in the coverage artifact so a user
# meets it in the data, not only in this source file.
# ========================================================================== #
_CONFORMAL_METHOD = {"module": "conformal", "function": "calibrate",
                     "version": "conformal-1.0",
                     "fit_provenance": FIT_PROVENANCE_ABSENT}

_CONFORMAL_STATIC = (
    "version", "alpha", "target_coverage", "guarantee",
    "calibration_quantiles.t50.marginal_quantile",
    "calibration_quantiles.t50.marginal_n",
    "calibration_quantiles.t50.score",
    "calibration_quantiles.calibration_source",
)


def _adapter_conformal(ctx):
    nodes, edges = [], []
    art = "service_c_conformal.json"
    q, q_ref = ctx.get(art, "calibration_quantiles.t50.marginal_quantile")
    if q is _MISSING:
        return nodes, edges
    target, _t = ctx.get(art, "target_coverage")
    src, _s = ctx.get(art, "calibration_quantiles.calibration_source")
    nodes.append(make_node(
        "uncertainty", "conformal", "t50_marginal_quantile",
        "conformal marginal half-width for t50", "uncertainty.conformal",
        value=q, source_ref=q_ref, series_id=ctx.series_id,
        method=_CONFORMAL_METHOD,
        confidence=[make_confidence(
            "conformal_quantile", q,
            "split-conformal absolute-residual quantile: distribution-free, "
            "finite-sample valid under exchangeability — NOT an asymptotic CI",
            _text({"target_coverage": target if target is not _MISSING else None,
                   "calibrated_on": src if src is not _MISSING else None,
                   "stratum": "MARGINAL — the Mondrian stratum is deliberately "
                              "NOT selected; see honesty.conformal_interval_is_refused"}))],
        detail="MARGINAL quantile, read verbatim. M17 does not build t50 +/- q: "
               "that is arithmetic on engine values, and choosing the Mondrian "
               "stratum needs an assay-endpoint classification M17 may not make."))
    edges.append(make_edge(NID_RESULT_T50, "uncertainty:conformal:t50_marginal_quantile",
                           "bounded_by", "conformal", source_ref=q_ref))
    return nodes, edges


# ========================================================================== #
# ADAPTER 9 — M15 literature validation (lit_validation.json) — CORPUS ONLY
#
# The `claim_confidence_calibrated` kind (ECE/Brier-measured). It is attached to
# a CORPUS node and deliberately NOT joined to any series: M15's ledger holds 59
# claims over three synthetic FIXTURE documents with no series_id, and inventing
# a join to the kinetics corpus would be the single most dishonest thing this
# module could do.
# ========================================================================== #
_M15_METHOD = {"module": "m15_litingest", "function": "validate_against_cpad",
               "version": "m15-litingest-1.0",
               "fit_provenance": FIT_PROVENANCE_ABSENT}

_M15_STATIC = ("version", "validation.pmid", "validation.precision",
               "validation.recall", "validation.accuracy",
               "validation.fields_evaluated")


def _adapter_m15(ctx):
    nodes, edges = [], []
    art = "lit_validation.json"
    acc, acc_ref = ctx.get(art, "validation.accuracy")
    if acc is _MISSING:
        return nodes, edges
    pmid, _p = ctx.get(art, "validation.pmid")
    prec, _pr = ctx.get(art, "validation.precision")
    nodes.append(make_node(
        "result", "m15_litingest", "validation.accuracy",
        "M15 field-level extraction accuracy", "result.claim_validation",
        value=acc, source_ref=acc_ref, method=_M15_METHOD,
        confidence=[make_confidence(
            "claim_confidence_calibrated", acc,
            "field-level precision/recall MEASURED against CPAD's curated "
            "condition_vector, with an ECE/Brier calibration hook",
            _text({"pmid": pmid if pmid is not _MISSING else None,
                   "scope": "M15 FIXTURE documents ONLY — this is NOT evidence of "
                            "extraction accuracy on real PDFs and does NOT join to "
                            "the CPAD kinetics corpus",
                   "precision": prec if prec is not _MISSING else None}))],
        detail="corpus-scoped; deliberately NOT joined to any kinetics series"))
    return nodes, edges


# ========================================================================== #
# ADAPTERS 10 & 11 — M6 propensity + sequence axis (propensity.jsonl)
#
# Two adapters over ONE module, mirroring the _adapter_m4_features /
# _adapter_m4_gamma split: the two blocks answer different questions, join on
# different terms and fail independently, so one adapter would hide which half
# is missing.
#
# THE JOIN IS A TRIPLE, AND M17 MIRRORS M7's VERSION OF IT — BUG INCLUDED.
# M6 records are keyed (uniprot_id, assay, condition_vector.construct_id).
# m7_assemble.py:1045-1046 looks up that triple and, ON A MISS, FALLS BACK to
# the first record for the uniprot alone. M17 reproduces both steps (see
# ARTIFACTS / `Ctx.record_key_for`) because its contract is to explain the value
# the PRODUCT SHOWS. Implementing the join M7 *should* have written — dropping
# the fallback, or gating it on assay — would make M17 emit NO M6 node on the
# series where the fallback is what supplied the record: MEASURED 161 of 1,654
# series corpus-wide, 96 of them full-schema. The spine would still be showing
# M6-authored propensity/sequence_axis values there, so M17 would be silently
# refusing to explain fields the product displays. A provenance layer that
# quietly disagrees is worse than one that faithfully exposes a defect: the
# source_ref names the exact record used, so the mis-join is visible in the
# graph. Named in `m6_assay_bleed_fixed_m17_mirrors_the_corrected_join`.
#   DEFINITION of the 161/96: the (uniprot_id, condition_vector.assay_type,
#   condition_vector.construct_id) triple read from curves_triaged.jsonl is not a
#   key of the propensity triple index, and uniprot_id alone is — i.e. the `or`
#   at m7_assemble.py:1045-1046 supplied the record.
#
# WHAT THESE ADAPTERS DELIBERATELY DO NOT CLAIM: `propensity.available`,
# `propensity.construct_match` and `propensity.reason` are SYNTHESIZED by M7's
# join gate (m7_assemble.py:554-583) and exist in no artifact — the reason
# string is composed with an f-string in local scope and never reaches disk.
# No node is emitted for them and FIELD_OWNER_PREFIXES re-attributes them to M7.
#
# KNOWN DEFECT, PUBLISHED NOT FIXED: these adapters fire on whether the M6 RECORD
# joined, never on whether the SPINE emits the block. 460 results are
# `schema: compact` and carry no `propensity`/`sequence_axis` key at all, yet 382
# of them join an M6 record and receive 26,547 nodes explaining fields their own
# result does not contain. Coverage is unaffected (observe() walks the spine
# record's own leaves), and the same class predates these adapters in m4_gamma /
# m1_triage / m5_regimes / m4_features. The fix is M17-WIDE and needs an upstream
# fact M7 does not currently write — see the
# `explanations_reach_fields_the_compact_schema_never_emits` narrowing for the
# measurement, the investigation, and the Stage B plan.
#
# NO CROSS-MODULE EDGE TO m4_gamma. M6's `surface` block is a RESHAPED
# passthrough of a γ record (m6_propensity.py:279-300 — {value, ci, reliable}
# from {gamma, gamma_ci, gamma_reliable}), which invites a `computed_from` edge
# to the M4-γ node. It is refused because the two γs are usually NOT the same
# number: M6 keys γ by PROTEIN NAME and, since join-policy-1.0, by the ASSAY that
# concentration series was measured by — but still per PROTEIN, while M4-γ is keyed
# per CONCENTRATION SERIES, so one γ record still stands for a whole protein×assay.
# MEASURED — definition below — 304 series carry both, but on 100 of them NEITHER
# estimator produced a value, so the honest claim is about the 204 where at least
# one did: there the point estimates DIFFER on 174 and agree on 30. The edge would
# assert a provenance link that does not hold.
#   DEFINITION of the 304/204/174: population = spine series whose assembled
#   `propensity.surface.available` is true AND whose curves_triaged
#   `concentration_series_id` resolves to a record in gamma.jsonl; differs =
#   `propensity.surface.gamma_regression.value` is not byte-equal to that γ
#   record's `gamma_regression.gamma`, where an ABSENT key and a stored null are
#   both "no value" and therefore EQUAL to one another. The whole
#   `gamma_regression` blocks are NOT comparable — M6 renames the fields — so the
#   comparison is scalar-to-scalar and the field pair is named.
#   MIND THE ABSENT-VS-NULL TRAP, which produced a wrong published figure once:
#   31 of the 52 γ records carry `gamma_regression.status: insufficient_data` and
#   OMIT the `gamma` key, while M6's surface() writes `reg.get("gamma")` as an
#   explicit null — so the two sides encode the same fact differently. Comparing
#   them with `_canon_safe(resolve_key_path(...))` reads absent as the _MISSING
#   sentinel (whose _canon_safe is Python None) and null as the STRING "null",
#   which reports DIFFER on 92 such series. Compare the scalars directly.
# ========================================================================== #
_M6_PROP_METHOD = {"module": "m6_propensity", "function": "score_protein",
                   "version": "m6-anchor-1.0",
                   "fit_provenance": FIT_PROVENANCE_ABSENT}
_M6_SEQ_METHOD = {"module": "m6_propensity", "function": "sequence_axis",
                  "version": "m6-sequence-axis",
                  "fit_provenance": FIT_PROVENANCE_ABSENT}

# The JOIN AUDIT trail. M7 surfaces these three whatever the join outcome
# (m7_assemble.py:552-553) precisely so the join is auditable from the result,
# so they are emitted for every joined record — matched construct or not.
_M6_PROP_IDENTITY = (
    ("propensity.uniprot_id", "uniprot_id"),
    ("propensity.prop_construct_id", "condition_vector.construct_id"),
    ("propensity.prop_assay", "assay"),
)

# Emitted ONLY when the joined record's construct matches this curve's — the
# same gate m7_assemble.py:570 applies. On a mismatch M7 emits none of these,
# so claiming them would be explaining fields the product does not show.
_M6_PROP_SCORED = (
    ("propensity.scored_feature", "scored_feature"),
    ("propensity.feature_status", "feature_status"),
    ("propensity.condition_match", "condition_match"),
)

_M6_PROP_BLOCKS = ("propensity.surface", "propensity.intrinsic",
                   "propensity.cohort")

_M6_PROP_STATIC = tuple(
    ["uniprot_id", "assay", "condition_vector.construct_id", "scored_feature",
     "feature_status", "condition_match",
     "propensity.surface.available", "propensity.surface.reason",
     "propensity.surface.gamma_regression.value",
     "propensity.surface.gamma_regression.ci",
     "propensity.surface.gamma_regression.reliable",
     "propensity.surface.gamma_global.value", "propensity.surface.gamma_global.ci",
     "propensity.surface.gamma_disagreement",
     "propensity.surface.feature_vs_driving_force",
     "propensity.intrinsic.value", "propensity.intrinsic.anchor_kind",
     "propensity.intrinsic.anchor_version", "propensity.intrinsic.direction",
     "propensity.intrinsic.anchor_effective_n",
     "propensity.intrinsic.percentile_in_stratum",
     "propensity.intrinsic.propensity_percentile_in_stratum",
     "propensity.intrinsic.stratum_reference_interval_95",
     "propensity.intrinsic.stratum_reference_interval_95_flag",
     "propensity.intrinsic.suppressed", "propensity.intrinsic.reason",
     "propensity.cohort.value", "propensity.cohort.rank",
     "propensity.cohort.percentile", "propensity.cohort.propensity_percentile",
     "propensity.cohort.direction", "propensity.cohort.direction_note",
     "propensity.cohort.N", "propensity.cohort.N_excludes_self",
     "propensity.cohort.suppressed", "propensity.cohort.reason"])

_M6_SEQ_STATIC = tuple(
    ["uniprot_id", "assay", "condition_vector.construct_id",
     "sequence_axis.available", "sequence_axis.reason",
     "sequence_axis.uniprot_id", "sequence_axis.protein_name",
     "sequence_axis.assay", "sequence_axis.assay_endpoint_class",
     "sequence_axis.n_peptides", "sequence_axis.n_peptides_wt",
     "sequence_axis.n_peptides_total", "sequence_axis.n_amyloid_peptides",
     "sequence_axis.peptide_length_range",
     "sequence_axis.no_comparison_licensed_reason",
     "sequence_axis.excluded_available_predictors",
     "sequence_axis.null_disagreement_rate",
     "sequence_axis.null_disagreement_note",
     "sequence_axis.absent_predictors.zyggregator",
     "sequence_axis.absent_predictors.camsol"]
    + [".".join(("sequence_axis.sub_axes", p, f))
       for p in ("tango", "aggrescan", "pasta", "waltz")
       for f in ("available", "value", "version", "granularity", "endpoint",
                 "endpoint_class", "endpoint_match_to_assay",
                 "comparison_licensed", "reduction", "condition_inputs")])


def _m6_construct_matches(ctx) -> bool:
    """Mirror of m7_assemble._propensity_block's construct gate (line 570).

    Comparing two artifact strings to decide whether a node EXISTS is control
    flow, not derivation — every adapter here already branches on artifact
    values. No value is computed: whatever is emitted is still a verbatim read."""
    got, _r = ctx.get("propensity.jsonl", "condition_vector.construct_id")
    if got is _MISSING:
        return False
    return got == ctx.join_keys.get("condition_vector.construct_id")


def _m6_assay_matches(ctx) -> bool:
    """Mirror of the ASSAY gate m7_assemble gained in join-policy-1.0
    (`_propensity_block` and `_sequence_axis_block`).

    M17 must mirror the gate as well as the JOIN. Mirroring only the join is what
    produced 5 value-identity mismatches on the first rebuild after the gate
    landed: M7 refuses a cross-assay `sequence_axis` and publishes
    `available: false`, while the M6 record this adapter reads still says `true`,
    so a passthrough node contradicted the product it explains — the one thing
    this module may never do. Same control-flow-not-derivation reasoning as the
    construct mirror above."""
    got, _r = ctx.get("propensity.jsonl", "assay")
    if got is _MISSING:
        return False
    return got == ctx.join_keys.get("assay")


def _adapter_m6_propensity(ctx):
    nodes, edges = [], []
    art = "propensity.jsonl"
    if ctx.record(art) is None:
        return nodes, edges

    for spine, origin in _M6_PROP_IDENTITY:
        val, ref = ctx.get(art, origin)
        if val is _MISSING:
            continue
        nodes.append(make_node(
            "datum", "m6_propensity", origin, origin, "datum.m6_join", value=val,
            source_ref=ref, series_id=ctx.series_id, study_ref=ctx.study_ref,
            method=_M6_PROP_METHOD, explains=(spine,)))

    # Mirror M7's gates IN ITS ORDER: assay first, then construct. A record failing
    # both is refused by M7 as an ASSAY failure, and M17 must reach the same verdict
    # for the same reason or its explanation contradicts the product.
    if not _m6_assay_matches(ctx) or not _m6_construct_matches(ctx):
        # A real M6 record exists for this protein but for a different ASSAY or a
        # different CONSTRUCT. M7 refuses to attach its propensity (no cross-assay
        # attachment, no Wild-Type bleed onto a mutant), so there is nothing further
        # to explain — the join audit above is the whole of what the product shows.
        return nodes, edges

    for spine, origin in _M6_PROP_SCORED:
        val, ref = ctx.get(art, origin)
        if val is _MISSING:
            continue
        nodes.append(make_node(
            "datum", "m6_propensity", origin, origin, "datum.m6_joined", value=val,
            source_ref=ref, series_id=ctx.series_id, study_ref=ctx.study_ref,
            method=_M6_PROP_METHOD, explains=(spine,)))

    # The three VIEWS, never collapsed into one number (the M6 design's own
    # rule, and the same reason M4's two γ estimators stay two nodes).
    for block in _M6_PROP_BLOCKS:
        _emit_leaves(ctx, nodes, "m6_propensity", art, block, block,
                     "result", "result.m6_joined", _M6_PROP_METHOD)
    return nodes, edges


def _adapter_m6_sequence_axis(ctx):
    """A verbatim passthrough of the joined record's `sequence_axis` block, emitted
    only where M7 actually surfaces it (`_sequence_axis_block`).

    M17 invents no gate, but it MUST mirror M7's. On the 148 series with no joined
    record this adapter emits NOTHING, because there `sequence_axis.available` /
    `reason` are synthesized by M7 and a node claiming them would carry a FALSE
    source_ref. Since join-policy-1.0 the same is true across the ASSAY: M7 refuses
    a cross-assay sequence axis (its licensing fields are pure functions of the
    assay), so on those series the block the product publishes is M7's refusal, not
    M6's record. Passing M6's `available: true` through there produced 5 verified
    value-identity mismatches against the spine — kept as a regression test."""
    nodes, edges = [], []
    art = "propensity.jsonl"
    if ctx.record(art) is None:
        return nodes, edges
    if not _m6_assay_matches(ctx):
        return nodes, edges
    _emit_leaves(ctx, nodes, "m6_sequence_axis", art, "sequence_axis",
                 "sequence_axis", "result", "result.m6_joined", _M6_SEQ_METHOD)
    return nodes, edges


# ========================================================================== #
# THE ADAPTER REGISTRY
#
# `declared_keys` is the CONTRACT: a test asserts each path resolves in the live
# artifacts, so an upstream shape change fails the M17 suite loudly instead of
# silently degrading every explanation to `explanation_available: false`.
# ========================================================================== #
ADAPTERS = [
    {"name": "m1_triage", "source_module": "m1_ingest",
     "artifact": "curves_triaged.jsonl", "scope": "series",
     "declared_keys": _M1_STATIC,
     "declared_prefixes": ("condition_vector", "quality_flags"),
     "explains_declared": ("condition_vector.*", "quality_report.quality_flags.*",
                           "fittability_class", "censoring_class", "data_mode",
                           "source", "protein_id", "uniprot_id", "series_id",
                           "quality_report.artifact_flags",
                           "quality_report.signal_basis",
                           "quality_report.normalization_mode",
                           "quality_report.digitization_uncertainty"),
     "build": _adapter_m1},
    {"name": "m2_fit", "source_module": "m2_fit", "artifact": "fits.jsonl",
     "scope": "series",
     "declared_keys": ("series_id", "handling", "best_by_aicc", "candidates",
                       "note", "signal_used"),
     "declared_prefixes": ("fits",),
     "explains_declared": (),
     "build": _adapter_m2},
    {"name": "m4_features", "source_module": "m4_features",
     "artifact": "features.jsonl", "scope": "series",
     "declared_keys": ("series_id", "model", "r2", "status", "t50_status",
                       "lag_status", "validity_flags", "signal_basis",
                       "censoring_class", "definition_contract", "n_points",
                       "handling", "plateau_reached"),
     "declared_prefixes": ("features",),
     "explains_declared": ("curve_features.features.*", "curve_features.t50_status",
                           "curve_features.lag_status", "curve_features.validity_flags",
                           "curve_features.status", "curve_features.signal_basis",
                           "curve_features.censoring_class",
                           "curve_features.definition_contract", "lag_to_t50_ratio"),
     "build": _adapter_m4_features},
    {"name": "m4_gamma", "source_module": "m4_features", "artifact": "gamma.jsonl",
     "scope": "series",
     "declared_keys": ("concentration_series_id", "definition_contract",
                       "gamma_provenance", "gamma_regression.gamma",
                       "gamma_regression.gamma_ci", "gamma_regression.gamma_reliable",
                       "gamma_regression.status", "gamma_regression.estimator",
                       "gamma_global.gamma", "gamma_global.gamma_ci",
                       "gamma_global.estimator", "disagreement.difference",
                       "disagreement.disagree"),
     "declared_prefixes": ("gamma_regression", "gamma_global", "disagreement"),
     "explains_declared": ("gamma_regression.*", "gamma_global.*",
                           "gamma_disagreement.*"),
     "build": _adapter_m4_gamma},
    {"name": "m5_regimes", "source_module": "m5_classify",
     "artifact": "regimes.jsonl", "scope": "series",
     "declared_keys": _M5_STATIC,
     "declared_prefixes": ("descriptive_regime", "confidence", "window_of_validity"),
     "explains_declared": ("descriptive_regime.*", "shape_confidence.*",
                           "mechanistic.mechanistic_inference_licensed",
                           "mechanistic.verdict_or_equivalence_class",
                           "mechanistic.gates_failed", "mechanistic.degeneracy",
                           "window_of_validity.*"),
     "build": _adapter_m5},
    {"name": "m8_reachability", "source_module": "m8_reachability",
     "artifact": "m8_reachability.json", "scope": "series",
     "declared_keys": _M8_STATIC, "declared_prefixes": (),
     "explains_declared": (),
     "build": _adapter_m8},
    {"name": "m11_metadata", "source_module": "m11_metadata",
     "artifact": "metadata_quality.jsonl", "scope": "series",
     "declared_keys": _M11_STATIC, "declared_prefixes": (),
     "explains_declared": (),
     "build": _adapter_m11},
    {"name": "conformal", "source_module": "conformal",
     "artifact": "service_c_conformal.json", "scope": "series",
     "declared_keys": _CONFORMAL_STATIC, "declared_prefixes": (),
     "explains_declared": (),
     "build": _adapter_conformal},
    {"name": "m6_propensity", "source_module": "m6_propensity",
     "artifact": "propensity.jsonl", "scope": "series",
     "declared_keys": _M6_PROP_STATIC,
     "declared_prefixes": _M6_PROP_BLOCKS,
     "explains_declared": ("propensity.surface.*", "propensity.intrinsic.*",
                           "propensity.cohort.*", "propensity.scored_feature",
                           "propensity.feature_status", "propensity.condition_match",
                           "propensity.uniprot_id", "propensity.prop_construct_id",
                           "propensity.prop_assay"),
     "build": _adapter_m6_propensity},
    {"name": "m6_sequence_axis", "source_module": "m6_propensity",
     "artifact": "propensity.jsonl", "scope": "series",
     "declared_keys": _M6_SEQ_STATIC,
     "declared_prefixes": ("sequence_axis",),
     "explains_declared": ("sequence_axis.*",),
     "build": _adapter_m6_sequence_axis},
    {"name": "m15_litingest", "source_module": "m15_litingest",
     "artifact": "lit_validation.json", "scope": "corpus",
     "declared_keys": _M15_STATIC, "declared_prefixes": (),
     "explains_declared": (),
     "build": _adapter_m15},
]

ADAPTER_NAMES = tuple(a["name"] for a in ADAPTERS)

# Which module OWNS each M7-spine field. This table is M17's OWN declaration
# (longest-prefix match), used only for the by-module breakdown in the coverage
# artifact — it is documentation of the pipeline's shape, not a derived result.
FIELD_OWNER_PREFIXES = (
    ("condition_vector", "m1"), ("quality_report", "m1"),
    # M7 assembles these UNDER quality_report, but they are not M1's: the
    # anomaly panel is M5's runs-test/FDR output and the missingness summary is
    # computed by M7 itself. Longest-prefix match routes them correctly, which
    # matters because the Stage-B spec is grouped by owner.
    ("quality_report.anomaly", "m5"), ("quality_report.missingness", "m7"),
    ("fittability_class", "m1"), ("censoring_class", "m1"), ("data_mode", "m1"),
    ("source", "m1"), ("protein_id", "m1"), ("uniprot_id", "m1"),
    ("series_id", "m1"),
    ("model_selection", "m3"),
    ("curve_features", "m4"), ("lag_to_t50_ratio", "m4"),
    ("gamma_regression", "m4"), ("gamma_global", "m4"),
    ("gamma_disagreement", "m4"),
    ("descriptive_regime", "m5"), ("shape_confidence", "m5"),
    ("mechanistic", "m5"), ("window_of_validity", "m5"),
    ("propensity", "m6"), ("sequence_axis", "m6"),
    # M7 assembles these UNDER `propensity`, but they are NOT M6's: all three are
    # synthesized by M7's join GATE (m7_assemble.py:554-583) and exist at no key
    # path in propensity.jsonl — the `reason` string is built with an f-string in
    # local scope and never reaches an artifact. Attributing them to M6 charged
    # M6's denominator for three fields no M6 adapter could ever explain.
    #
    # THIS RE-ATTRIBUTION FLATTERS M6 (119 -> 116 paths, and 116/116 rather than
    # 116/119), which is the opposite direction from the m5 correction in the
    # Stage-A commit. It is made for the same reason regardless of direction:
    # the Stage-B spec is grouped by OWNER, and a field filed under the module
    # that cannot fix it is a mis-specification. Both denominators are stated in
    # the `m6_propensity_join_outcome_fields_are_m7s` narrowing so the change is
    # visible in the data rather than only in this table.
    ("propensity.available", "m7"), ("propensity.construct_match", "m7"),
    ("propensity.reason", "m7"),
    # join-policy-1.0 added the ASSAY gate, and its outputs are M7's for exactly
    # the same reason: `assay_match` is computed by the gate, and on the refusal
    # path `sequence_axis` is replaced wholesale by an M7-authored block whose
    # `assay_match`/`prop_assay`/`reason` exist at no key path in propensity.jsonl.
    # Filed here rather than left in M6's column so the Stage-B spec still names
    # the module that can actually close each gap.
    ("propensity.assay_match", "m7"),
    ("sequence_axis.assay_match", "m7"), ("sequence_axis.prop_assay", "m7"),
    ("information_yield", "m7"), ("information_yield_rank", "m7"),
    ("caveats", "m7"), ("schema", "m7"), ("engine_build_id", "m7"),
    ("m9_hook", "m9"),
)


def field_owner(path: str) -> str:
    best, owner = "", "unassigned"
    for pre, mod in FIELD_OWNER_PREFIXES:
        if (path == pre or path.startswith(".".join((pre, "")))) and len(pre) > len(best):
            best, owner = pre, mod
    return owner


# ========================================================================== #
# GRAPH ASSEMBLY
#
# EDGE DIRECTION CONVENTION (fixed, and relied on by every projection):
#   an edge points FROM the dependent node TO the thing it depends on.
#   `refusal --gated_by--> gate`, `gate --blocked_by--> metadata_fact`.
# So a topological order that emits every `to` before its `from` is a
# dependencies-first order, which is what the renderer walks.
# ========================================================================== #
def _topological_order(node_ids, edges):
    """Kahn with a lexicographically-sorted ready queue, so the order is UNIQUE
    rather than merely valid — two runs cannot pick different valid orders."""
    ids = sorted(node_ids)
    deps = {}
    rdeps = {}
    for i in ids:
        deps[i] = set()
        rdeps[i] = set()
    for e in edges:
        u, v = e["from"], e["to"]
        if u in deps and v in deps and u != v:
            deps[u].add(v)
            rdeps[v].add(u)
    remaining = {}
    for i in ids:
        remaining[i] = set(deps[i])
    ready = sorted([i for i in ids if not remaining[i]])
    order = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        newly = []
        for m in sorted(rdeps[n]):
            remaining[m].discard(n)
            if not remaining[m]:
                newly.append(m)
        for m in newly:
            ready.append(m)
        ready.sort()
    return order, len(order) == len(ids)


def build_graph(series_id, bundle, join_keys=None, scope="series",
                spine_record=None) -> dict:
    """Assemble the evidence DAG for one series (or the corpus record).

    `spine_record` enforces THE SPINE-EMISSION INVARIANT: *no node may explain a
    field path the product does not publish for this series.* It is applied here,
    in the builder, and NOT inside any adapter — the architectural rule that no
    adapter may read the spine is intact, because an adapter still cannot see it.
    Nothing is derived either: this only decides whether a node SURVIVES, exactly
    the control-flow-not-derivation reasoning the construct/assay mirrors use.

    WHY IT EXISTS. `propensity` and `sequence_axis` (and gamma, and several M1/M5
    blocks) live only in M7's FULL schema, while 460 results are `schema: compact`
    and carry those keys not at all. Adapters gate on whether an UPSTREAM record
    joined, never on whether the spine emits the block, so 382 compact series were
    receiving 25,660 M6 nodes — 42,113 M17-wide across six adapters — explaining
    fields their own result does not contain. Coverage was never affected (the
    accumulator walks the spine record's own leaves), so this was a wrong-GRAPH
    defect, not a wrong-number one: `m17_explain --series CPAD-TK-1041` would print
    `propensity.cohort.N = 109` for a series the engine publishes no propensity for.
    Explaining a value the product refuses to publish is the mirror image of the
    contradiction this module exists to prevent.

    Passing `spine_record=None` (the corpus scope, and unit tests that build a graph
    with no spine) disables the gate rather than dropping everything."""
    ctx = Ctx(series_id, bundle, join_keys)
    nodes_by_id = OrderedDict()
    raw_edges = []
    adapter_errors = []
    node_conflicts = []
    adapters_fired = []

    for ad in ADAPTERS:
        if ad["scope"] != scope:
            continue
        try:
            ns, es = ad["build"](ctx)
        except Exception as exc:                      # never raise on bad input
            adapter_errors.append({"adapter": ad["name"],
                                   "error_type": type(exc).__name__,
                                   "error": str(exc)})
            continue
        if ns or es:
            adapters_fired.append(ad["name"])
        for n in ns:
            nid = n["node_id"]
            prev = nodes_by_id.get(nid)
            if prev is not None:
                if _canon_safe(prev) != _canon_safe(n):
                    node_conflicts.append({
                        "node_id": nid,
                        "reason": "duplicate_node_id_with_different_content",
                        "adapters": sorted({node_adapter(prev), node_adapter(n)})})
                continue
            nodes_by_id[nid] = n
        raw_edges.extend(es)

    # ---- THE SPINE-EMISSION GATE -------------------------------------------
    # A node with NO `explains` is internal (method, assumption, corpus fact) and
    # makes no claim about a published field, so it is never gated. A node that
    # DOES declare paths survives if at least one of them resolves in this series'
    # own spine record; if none does, the product publishes nothing it could be
    # explaining and the node is removed. Counted, never silent.
    gated_nodes = []
    if spine_record is not None:
        for nid in list(nodes_by_id):
            exp = [p for p in (nodes_by_id[nid].get("explains") or ()) if p]
            if not exp:
                continue
            if any(resolve_key_path(spine_record, p) is not _MISSING for p in exp):
                continue
            gated_nodes.append({"node_id": nid,
                                "adapter": node_adapter(nodes_by_id[nid]),
                                "explains": exp,
                                "reason": "spine_does_not_emit_these_paths"})
            del nodes_by_id[nid]

    # ---- drop dangling edges, but RECORD them (a test asserts this is empty) -
    _gated_ids = {g["node_id"] for g in gated_nodes}
    edges = []
    dropped = []
    seen_edges = set()
    for e in raw_edges:
        if e["from"] not in nodes_by_id or e["to"] not in nodes_by_id:
            # an edge whose endpoint the GATE removed is a consequence of the gate,
            # not a dangling reference — keeping the two apart matters because the
            # dangling list is a defect signal and this one is expected behaviour
            reason = ("endpoint_node_gated_not_in_spine"
                      if (e["from"] in _gated_ids or e["to"] in _gated_ids)
                      else "endpoint_node_absent")
            dropped.append({"from": e["from"], "to": e["to"], "kind": e["kind"],
                            "adapter": e["adapter"], "reason": reason})
            continue
        key = (e["from"], e["to"], e["kind"])
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append(e)

    # `record_key` is omitted when it equals this record's series_id (the
    # overwhelmingly common case); `node_source()` restores it.
    for n in nodes_by_id.values():
        ref = n.get("source_ref")
        if ref and ref.get("record_key") == series_id:
            ref.pop("record_key")
    edges.sort(key=lambda e: (_EDGE_RANK.get(e["kind"], 99), e["from"], e["to"]))
    nodes = sorted(nodes_by_id.values(), key=lambda n: n["node_id"])

    # A node's DEPENDENCIES are its outgoing edges, already present in this
    # record's `edges` list; `node_dependencies(graph, node_id)` reads them.
    # Duplicating them onto each node stored the same facts twice and could
    # drift — the same rule the projections follow.

    _order, acyclic = _topological_order(list(nodes_by_id.keys()), edges)

    # ---- INTERN the method blocks ------------------------------------------
    # Every node carries module + function + version IN ITS method_ref string,
    # which resolves to the full method object (including the fit_provenance
    # absence record) in this same record's `methods` table. Inlining the full
    # block instead cost ~900 bytes on each of ~175k nodes — the §2.3-absence
    # prose alone, repeated — for zero additional traceability.
    methods = {}
    for n in nodes:
        m = n.pop("method", None) or {}
        mid = "@".join((".".join((str(m.get("module")), str(m.get("function")))),
                        str(m.get("version"))))
        if mid not in methods:
            methods[mid] = m
        n["method_ref"] = mid

    graph = {
        "methods": methods,
        "m17_version": M17_VERSION,
        "series_id": series_id,
        "scope": scope,
        "engine_build_id": bundle.get("engine_build_id"),
        "nodes": nodes,
        "edges": edges,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        # The canonical topological order is DERIVED on read by
        # m17_explain._topological_order(...); storing the id list again cost
        # ~4.6 KB per record to restate what the edges already determine. That
        # the order EXISTS (i.e. the graph is acyclic) is a real fact, so the
        # boolean is stored and asserted.
        "acyclic": acyclic,
        "adapters_fired": sorted(adapters_fired),
        "adapter_errors": adapter_errors,
        "node_conflicts": node_conflicts,
        "dropped_edges": dropped,
        "spine_gated_nodes": gated_nodes,
    }
    if scope == "series":
        rec = ctx.record("curves_triaged.jsonl")
        study = resolve_key_path(rec, "source_study") if rec else _MISSING
        graph["provenance"] = {
            "source_study": None if study is _MISSING else study,
            "source_ref": _ref("curves_triaged.jsonl", series_id, "source_study"),
            "note": ("the full CPAD study record sits here ONCE; each node's "
                     "source.study_ref carries the pmid that resolves to it, "
                     "rather than duplicating author/reference on every node"),
        }
    # PROJECTIONS ARE NOT STORED. They are VIEWS over this same DAG, produced
    # by `project(graph, name)` on read. Serialising them would (a) duplicate
    # every edge one-to-two more times and (b) quietly contradict the design
    # claim that the four views cannot diverge — a stored copy CAN drift from
    # the graph it came from; a function call cannot. The available projections
    # and their edge-kind filters are declared in the artifact header.
    graph["projection_names"] = sorted(PROJECTIONS)
    return graph


def project(graph: dict, name: str) -> dict:
    """One of the four VIEWS of the single DAG: filter by edge kind, then
    topologically order. Projections share node ids with the graph and copy
    nothing, so they cannot drift apart or contradict one another."""
    kinds = PROJECTIONS.get(name)
    if kinds is None:
        return {"projection": name, "available": False,
                "reason": "unknown projection", "valid": sorted(PROJECTIONS)}
    sel = [e for e in graph.get("edges", []) if e["kind"] in kinds]
    ids = set()
    for e in sel:
        ids.add(e["from"])
        ids.add(e["to"])
    order, acyclic = _topological_order(ids, sel)
    return {
        "projection": name,
        "edge_kinds": list(kinds),
        "node_order": order,
        "edges": [[e["from"], e["kind"], e["to"]] for e in sel],
        "n_nodes": len(order),
        "n_edges": len(sel),
        "acyclic": acyclic,
    }


# ========================================================================== #
# THE DETERMINISTIC RENDERER
#
# Templates are the ONLY way text is produced. `render_node` is a dict lookup
# plus one `str.format_map` — it contains no concatenation, no f-string and no
# `join`, so a sentence CANNOT be composed at runtime. Same graph => byte-
# identical text.
# ========================================================================== #
TEMPLATES = {
    "datum": "DATUM {label} = {value}   [{artifact}:{key_path}]",
    "datum.blocker": ("BLOCKER {label} affects {value} curves   {detail}   "
                      "[{artifact}:{key_path}]"),
    # The M6 templates print {record_key} — the (uniprot_id, assay, construct_id)
    # triple of the record actually joined. That is what makes M7's uniprot-only
    # fallback VISIBLE: on a fallback series the printed assay differs from the
    # curve's own, so the mis-join is readable in the tree rather than hidden.
    "datum.m6_join": ("DATUM {label} = {value} — join audit: this curve resolved "
                      "to M6 record {record_key}   [{artifact}:{key_path}]"),
    "datum.m6_joined": ("DATUM {label} = {value}   [{artifact}:{key_path} "
                        "@ record {record_key}]"),
    "metadata_fact.known": ("META {label} = {value} (RECORDED)   "
                            "[{artifact}:{key_path}]"),
    "metadata_fact.unknown": ("META {label} = {value} — NOT RECORDED by the source "
                              "study   [{artifact}:{key_path}]"),
    "metadata_fact.blocking": ("META {label} — {detail}   [{artifact}:{key_path}]"),
    "result": "RESULT {label} = {value}   [{artifact}:{key_path}]",
    "result.m6_joined": ("RESULT {label} = {value}   [{artifact}:{key_path} "
                         "@ record {record_key}]"),
    # An ASSOCIATIVE template. No causal verb appears here, and the renderer
    # selects it from the node's causal_license, so a statistical-only source
    # cannot reach a causal sentence.
    "result.statistical_only": ("RESULT {label} = {value} — ASSOCIATIVE ONLY; no "
                                "causal reading is licensed   [{artifact}:{key_path}]"),
    "result.equivalence_class": ("EQUIVALENCE CLASS {label}: the evidence CANNOT "
                                 "DISTINGUISH {value}. narrowed = {detail}   "
                                 "[{artifact}:{key_path}]"),
    "result.claim_validation": ("RESULT {label} = {value} — measured on M15 FIXTURE "
                                "documents, NOT on the CPAD kinetics corpus   "
                                "[{artifact}:{key_path}]"),
    "method.model": ("METHOD selected model = {value} ({detail})   "
                     "[{artifact}:{key_path}]"),
    "assumption.unverified": ("ASSUMPTION (taken as true, NOT measured) {label}: "
                              "{detail}"),
    "gate.passed": "GATE {label} = {value} (passed)   [{artifact}:{key_path}]",
    "gate.failed": "GATE {label} = {value} (FAILED)   [{artifact}:{key_path}]",
    "refusal.mechanism": ("REFUSED: PRISE does not name a single mechanism for this "
                          "series. Verbatim reason [{artifact}:{key_path}]: {detail}"),
    "deferral.named": "DEFERRAL (named, not yet built): {label}",
    "deferral.action": ("ACTION {label} — supplying this changes what is licensed   "
                        "{detail}   [{artifact}:{key_path}]"),
    "deferral.ambiguous_cause": ("DEFERRAL (Stage B) {label}. Artifact records only: "
                                 "{value}. {detail}"),
    "uncertainty.censoring_bound": ("UNCERTAINTY {label} = {value} — a censored "
                                    "estimate is a BOUND, never a clean value   "
                                    "[{artifact}:{key_path}]"),
    "uncertainty.conformal": ("UNCERTAINTY conformal MARGINAL half-width = {value}. "
                              "{detail}   [{artifact}:{key_path}]"),
    "ceiling.validity": "CEILING {label}: {detail}   [{artifact}:{key_path}]",
    "ceiling.window": ("CEILING window_of_validity {label} = {value} — inference is "
                       "not licensed beyond it   [{artifact}:{key_path}]"),
}

EDGE_TEMPLATES = {
    "computed_from": "computed_from -> {to_label}",
    "assumes": "assumes -> {to_label}",
    "gated_by": "gated_by -> {to_label}",
    "blocked_by": "blocked_by -> {to_label}",
    "uncertainty_from": "uncertainty_from -> {to_label}",
    "bounded_by": "bounded_by -> {to_label}",
    # A QUANTITATIVE counterfactual is licensed only on a MEASURED gain basis.
    "unblocked_by.service_c_measured": (
        "unblocked_by -> {to_label}   [gain_basis=service_c_measured: the gain is "
        "MEASURED (Service-C gamma 1->4 Tier-B split), so a quantitative "
        "counterfactual is licensed]"),
    "unblocked_by.structural_estimate": (
        "unblocked_by -> {to_label}   [gain_basis=structural_estimate: an UPPER "
        "BOUND only, not measured; inherits the M8 validity ceiling]"),
    "unblocked_by": (
        "unblocked_by -> {to_label}   [gain_basis absent: NO quantitative "
        "counterfactual is licensed]"),
}


class _Fields(dict):
    """A format mapping that never raises on a missing slot."""

    def __missing__(self, key):
        return "<unavailable>"


def node_source(graph: dict, node: dict) -> dict:
    """The node's FULL source block: its resolvable pointer plus the series and
    CPAD study provenance carried once at record level."""
    ref = node.get("source_ref") or {}
    study = ((graph.get("provenance") or {}).get("source_study") or {})
    return {
        "series_id": graph.get("series_id"),
        "artifact": "/".join((ARTIFACT_DIR, ref.get("artifact", ""))),
        "record_key": ref.get("record_key", graph.get("series_id")),
        "key_path": ref.get("key_path"),
        "source_study": study or None,
    }


def node_dependencies(graph: dict, node_id: str) -> list:
    """The node's dependency edges (the outgoing edges of the DAG). Derived from
    the record's own `edges` rather than duplicated onto every node."""
    out = []
    for e in graph.get("edges", []):
        if e.get("from") == node_id:
            out.append({"edge": e["kind"], "to": e["to"]})
    out.sort(key=lambda x: (x["edge"], x["to"]))
    return out


def node_adapter(node: dict) -> str:
    """The adapter that built the node — the middle segment of its node_id."""
    parts = (node.get("node_id") or "").split(":", 2)
    return parts[1] if len(parts) > 2 else ""


def node_label(node: dict) -> str:
    """The node's label. When omitted it is identical to the slug segment of the
    node_id (that is WHY it was omitted), so it is reconstructed from there."""
    lab = node.get("label")
    if lab is not None:
        return lab
    return (node.get("node_id") or "").split(":", 2)[-1]


def node_causal_license(node: dict) -> str:
    """The node's causal licence, defaulting to "none" when omitted."""
    return node.get("causal_license", "none")


def _template_fields(node: dict) -> dict:
    """The substitution table for `render_node`'s single format_map.

    THIS IS THE OTHER HALF OF THE RENDER PATH, and it is under the SAME rule:
    no f-string, no `+` on strings, no str.join. Every value here must be a
    verbatim node/ref field or a `_text()` canonicalisation of one — the moment a
    field is ASSEMBLED here, `render_node` stays a pure lookup while the sentence
    is composed one frame down, which is the guard's blind spot rather than its
    satisfaction. The AST test walks `render_node` AND this function together
    (`test_render_node_cannot_compose_a_sentence_at_runtime`)."""
    ref = node.get("source_ref") or {}
    return {
        "label": node_label(node),
        "value": _text(node.get("value")),
        "detail": node.get("detail") if node.get("detail") is not None else "",
        "artifact": ref.get("artifact", ""),
        "key_path": ref.get("key_path", ""),
        # A COMPOSITE record_key is a tuple; `_text` gives it the canonical JSON
        # encoding so a rendered line cannot pick up a Python repr.
        "record_key": _text(ref.get("record_key", "")),
        "kind": node.get("kind"),
        "node_id": node.get("node_id"),
    }


def _template_key(node: dict) -> str:
    """Pick the template. A `statistical_only` node is forced onto the
    associative template regardless of what the adapter asked for, so a causal
    verb cannot be reached from a statistical source even by mistake."""
    if node_causal_license(node) == "statistical_only":
        return "result.statistical_only"
    return node.get("template")


def render_node(node: dict) -> dict:
    """ONE dict lookup + ONE str.format_map. No concatenation, no f-string, no
    join — a sentence cannot be composed at runtime. An unknown template is a
    REFUSAL to narrate, never an improvised sentence."""
    tpl = TEMPLATES.get(_template_key(node))
    if tpl is None:
        return {"explanation_available": False,
                "reason_code": "no_template_for_node",
                "template_key": node.get("template"),
                "node_id": node.get("node_id")}
    return {"explanation_available": True,
            "text": tpl.format_map(_Fields(_template_fields(node)))}


def render_edge(edge: dict, nodes_by_id: dict) -> dict:
    tpl = EDGE_TEMPLATES.get(edge.get("template", edge["kind"]))
    if tpl is None:
        return {"explanation_available": False,
                "reason_code": "no_template_for_edge",
                "template_key": edge.get("template")}
    target = nodes_by_id.get(edge["to"])
    label = node_label(target) if target else edge["to"]
    return {"explanation_available": True,
            "text": tpl.format_map(_Fields({"to_label": label}))}


def _indent(depth: int) -> str:
    if depth < 0:
        return _INDENT[0]
    if depth >= len(_INDENT):
        return _INDENT[-1]
    return _INDENT[depth]


def render_tree(graph: dict, root_id: str) -> str:
    """The rooted ASCII explanation. Deterministic: nodes and edges are already
    in a canonical order, and a revisited node prints its line but not its
    subtree (so the walk terminates and the output is stable)."""
    nodes_by_id = {n["node_id"]: n for n in graph.get("nodes", [])}
    out = []
    seen = set()
    out_edges = {}
    for e in graph.get("edges", []):
        out_edges.setdefault(e["from"], []).append(e)

    def walk(nid, depth):
        node = nodes_by_id.get(nid)
        if node is None:
            return
        r = render_node(node)
        line = r["text"] if r.get("explanation_available") else _canon(r)
        out.append("".join((_indent(depth), line)))
        if nid in seen:
            return
        seen.add(nid)
        for e in out_edges.get(nid, []):
            er = render_edge(e, nodes_by_id)
            etext = er["text"] if er.get("explanation_available") else _canon(er)
            out.append("".join((_indent(depth + 1), etext)))
            walk(e["to"], depth + 2)

    walk(root_id, 0)
    return "\n".join(out)


def primary_root(graph: dict):
    """The node an explanation is rooted at: the mechanism refusal when the
    engine produced one, else the highest-ranked available result. Returns None
    when nothing explainable exists — which is then reported as such."""
    ids = {n["node_id"] for n in graph.get("nodes", [])}
    for candidate in (NID_REFUSAL_MECHANISM, NID_RESULT_T50, NID_RESULT_REGIME):
        if candidate in ids:
            return candidate
    return None


def explain_series(graph: dict) -> dict:
    """The rendered explanation for one series, or an honest refusal."""
    root = primary_root(graph)
    if root is None:
        return {"explanation_available": False,
                "reason_code": "no_explainable_root",
                "reason": ("no mechanism refusal, t50 result or descriptive regime "
                           "exists for this series — the engine emitted none of "
                           "them, so there is nothing to trace. This is an ENGINE "
                           "output gap, not a missing M17 adapter."),
                "series_id": graph.get("series_id")}
    return {"explanation_available": True, "root": root,
            "text": render_tree(graph, root), "series_id": graph.get("series_id")}


# ========================================================================== #
# VALIDATORS — the mechanical form of "M17 derives nothing"
# ========================================================================== #
class RawResolver:
    """Resolves a source_ref against artifacts RE-READ FROM DISK — deliberately
    NOT against the pruned in-memory bundle, so pruning can never hide a bad
    ref. JSONL files are indexed lazily on first use."""

    def __init__(self, indir):
        self.indir = Path(indir)
        self._cache = {}

    def _load(self, artifact):
        if artifact in self._cache:
            return self._cache[artifact]
        spec = ARTIFACTS.get(artifact)
        path = self.indir.joinpath(artifact)
        data = None
        if spec and spec["kind"] == "jsonl":
            rk = spec["record_key"]
            idx = {}
            for rec in _iter_jsonl(path):
                k = _record_index_key(rec, rk)
                if k is not None:
                    idx[k] = rec
            data = idx
        else:
            data = _read_json(path)
        self._cache[artifact] = data
        return data

    def resolve(self, ref, default_record_key=None):
        artifact = _artifact_name(ref)
        spec = ARTIFACTS.get(artifact)
        if spec is None:
            return _MISSING
        data = self._load(artifact)
        if data is None:
            return _MISSING
        if spec["kind"] == "jsonl":
            rk = ref.get("record_key", default_record_key)
            # A COMPOSITE record_key round-trips through JSON as a LIST, and the
            # index is keyed by TUPLES. Restore the type so a ref re-read from a
            # written artifact resolves exactly as the in-memory one did.
            if isinstance(rk, list):
                rk = tuple(rk)
            try:
                rec = data.get(rk)
            except TypeError:               # an unhashable record_key resolves nowhere
                return _MISSING
            if rec is None:
                return _MISSING
            return resolve_key_path(rec, ref.get("key_path"))
        return resolve_key_path(data, ref.get("key_path"))


def validate_vocabulary(graph: dict) -> list:
    """Every node kind, edge kind, confidence kind and causal licence must be a
    member of its CLOSED vocabulary."""
    issues = []
    for n in graph.get("nodes", []):
        if n.get("kind") not in NODE_KIND_SET:
            issues.append({"node_id": n.get("node_id"), "issue": "unknown_node_kind",
                           "value": n.get("kind")})
        if node_causal_license(n) not in CAUSAL_LICENSE_SET:
            issues.append({"node_id": n.get("node_id"),
                           "issue": "unknown_causal_license",
                           "value": n.get("causal_license")})
        conf = n.get("confidence")
        if conf is not None:
            if not isinstance(conf, list):
                issues.append({"node_id": n.get("node_id"),
                               "issue": "confidence_is_not_a_list",
                               "value": type(conf).__name__})
            else:
                for c in conf:
                    if not isinstance(c, dict) or c.get("kind") not in CONFIDENCE_KIND_SET:
                        issues.append({"node_id": n.get("node_id"),
                                       "issue": "unknown_confidence_kind",
                                       "value": (c or {}).get("kind")
                                       if isinstance(c, dict) else type(c).__name__})
                    elif set(c.keys()) != {"kind", "value", "basis", "valid_at"}:
                        issues.append({"node_id": n.get("node_id"),
                                       "issue": "confidence_shape",
                                       "value": sorted(c.keys())})
    for e in graph.get("edges", []):
        if e.get("kind") not in EDGE_KIND_SET:
            issues.append({"edge": [e.get("from"), e.get("to")],
                           "issue": "unknown_edge_kind", "value": e.get("kind")})
        if e.get("causal_license", "none") not in CAUSAL_LICENSE_SET:
            issues.append({"edge": [e.get("from"), e.get("to")],
                           "issue": "unknown_causal_license",
                           "value": e.get("causal_license")})
    return issues


def validate_source_refs(graph: dict, resolver) -> list:
    """Every source_ref must RESOLVE against the real artifact. A dangling ref
    is a hard failure: an untraceable explanation is worse than none."""
    dangling = []
    for n in graph.get("nodes", []):
        ref = n.get("source_ref")
        if not ref:
            if n.get("kind") not in ("assumption",):
                dangling.append({"node_id": n["node_id"], "issue": "no_source_ref"})
            continue
        art = ref.get("artifact") or ""
        # A source_ref names a BARE artifact filename; anything path-shaped
        # is a portability bug (a graph built on Windows must be byte-
        # identical to one built on Linux), and an unregistered artifact
        # means the ref can never be resolved by a consumer.
        if ("\\" in art) or ("/" in art) or (":" in art) or art not in ARTIFACTS:
            dangling.append({"node_id": n["node_id"],
                             "issue": "non_portable_or_unregistered_artifact",
                             "artifact": art})
        if resolver.resolve(ref, graph.get("series_id")) is _MISSING:
            dangling.append({"node_id": n["node_id"], "issue": "dangling_source_ref",
                             "ref": ref})
    for e in graph.get("edges", []):
        ref = e.get("source_ref")
        if ref and resolver.resolve(ref, graph.get("series_id")) is _MISSING:
            dangling.append({"edge": [e.get("from"), e.get("to")],
                             "issue": "dangling_edge_source_ref", "ref": ref})
    return dangling


def assert_no_derivation(graph: dict, resolver) -> list:
    """THE load-bearing check. Every node value that is not None must BYTE-EQUAL
    the artifact value at its declared key path, re-read from disk. A number M17
    computed cannot exist at any key path in any artifact, so it cannot pass."""
    violations = []
    for n in graph.get("nodes", []):
        if n.get("value") is None:
            continue
        ref = n.get("source_ref")
        if not ref:
            violations.append({"node_id": n["node_id"],
                               "issue": "value_without_source_ref",
                               "value": n.get("value")})
            continue
        got = resolver.resolve(ref, graph.get("series_id"))
        if got is _MISSING:
            violations.append({"node_id": n["node_id"],
                               "issue": "value_source_ref_does_not_resolve",
                               "ref": ref})
            continue
        if _canon_safe(got) != _canon_safe(n["value"]):
            violations.append({"node_id": n["node_id"],
                               "issue": "value_differs_from_artifact",
                               "artifact_value": got, "node_value": n["value"]})
    return violations


def check_adapter_contracts(bundle: dict, indir) -> dict:
    """THE CONTRACT TEST. Each adapter DECLARES the artifact key paths it reads;
    this asserts they exist in the LIVE artifacts. When an upstream module
    changes shape, the M17 suite fails LOUDLY here instead of silently degrading
    every explanation to `explanation_available: false`."""
    resolver = RawResolver(indir)
    results = []
    for ad in ADAPTERS:
        art = ad["artifact"]
        spec = ARTIFACTS.get(art) or {}
        data = resolver._load(art)
        for key in ad["declared_keys"]:
            present = 0
            checked = 0
            if data is None:
                results.append({"adapter": ad["name"], "artifact": art,
                                "key_path": key, "ok": False,
                                "reason": "artifact_missing"})
                continue
            if spec.get("kind") == "jsonl":
                for rec in data.values():
                    checked += 1
                    if resolve_key_path(rec, key) is not _MISSING:
                        present += 1
            else:
                checked = 1
                if resolve_key_path(data, key) is not _MISSING:
                    present = 1
            results.append({"adapter": ad["name"], "artifact": art,
                            "key_path": key, "ok": present > 0,
                            "n_records_present": present,
                            "n_records_checked": checked})
    failed = [r for r in results if not r["ok"]]
    return {"n_checked": len(results), "n_failed": len(failed),
            "all_ok": not failed, "failures": failed, "results": results}


# ========================================================================== #
# COVERAGE — MEASURED, NOT ASSERTED
#
# The DENOMINATOR is published in full (every M7-spine leaf path) so the
# headline percentage is auditable and a later Stage B can diff against it. A
# coverage number over a loosely-chosen denominator is unfalsifiable, which
# would defeat the point of measuring it.
#
# TWO populations are reported and NEVER blended:
#   * EXPLAINABLE series — those for which M5 produced a record (the engine
#     emitted something to explain);
#   * ALL series — the whole M7 spine.
# Blending them would mix "M17 has no adapter" with "the engine produced
# nothing", which are different failures with different fixes. Stage B only
# addresses the first.
# ========================================================================== #
# Which SPINE-OWNER module short-names (as returned by `field_owner`) have a
# Stage-A adapter. Declared explicitly rather than derived from
# `source_module`, because the spine's owner names ("m4") and the adapters'
# module names ("m4_features") are different namespaces and conflating them
# silently reports every module as adapter-less.
ADAPTER_OWNER_MODULES = frozenset(("m1", "m2", "m4", "m5", "m6", "m8", "m11"))
ADAPTER_SOURCE_MODULES = frozenset(a["source_module"] for a in ADAPTERS)

_UNEXPLAINED_REASONS = {
    "no_adapter": ("no Stage-A adapter reads this module's artifact; the field is "
                   "not traced and is therefore never narrated"),
    "not_declared": ("this module HAS a Stage-A adapter, but this particular field "
                     "is not declared in its spine mapping (no value-identity was "
                     "verified for it)"),
}


class CoverageAccumulator:
    """Counts M17's OWN nodes and fields. These integers are properties of the
    explanation layer, not engine results — they are the only numbers in this
    module that are not read from an upstream artifact (see `_frac`)."""

    def __init__(self):
        self.emitting_all = Counter()
        self.explained_all = Counter()
        self.emitting_expl = Counter()
        self.explained_expl = Counter()
        self.adapters_for_path = {}
        self.n_series_all = 0
        self.n_series_expl = 0
        self.inst_all = 0
        self.inst_expl_all = 0
        self.inst_explainable = 0
        self.inst_expl_explainable = 0
        self.n_series_with_root = 0
        self.value_identity_checked = 0
        self.value_identity_mismatch = []
        self.node_kind_counts = Counter()
        self.edge_kind_counts = Counter()
        self.confidence_kind_counts = Counter()
        self.causal_license_counts = Counter()

    def observe(self, spine_record: dict, graph: dict, explainable: bool) -> dict:
        leaves = sorted(set(leaf_paths(spine_record)))
        explained = {}
        for n in graph.get("nodes", []):
            self.node_kind_counts[n.get("kind")] += 1
            self.causal_license_counts[node_causal_license(n)] += 1
            for c in (n.get("confidence") or []):
                if isinstance(c, dict):
                    self.confidence_kind_counts[c.get("kind")] += 1
            for p in (n.get("explains") or ()):
                explained.setdefault(p, []).append(n)
        for e in graph.get("edges", []):
            self.edge_kind_counts[e.get("kind")] += 1

        self.n_series_all += 1
        if explainable:
            self.n_series_expl += 1
        if primary_root(graph) is not None:
            self.n_series_with_root += 1

        n_expl_here = 0
        for path in leaves:
            self.emitting_all[path] += 1
            self.inst_all += 1
            if explainable:
                self.emitting_expl[path] += 1
                self.inst_explainable += 1
            hits = explained.get(path)
            if not hits:
                continue
            n_expl_here += 1
            self.explained_all[path] += 1
            self.inst_expl_all += 1
            if explainable:
                self.explained_expl[path] += 1
                self.inst_expl_explainable += 1
            for n in hits:
                self.adapters_for_path.setdefault(path, set()).add(node_adapter(n))
            # CROSS-ARTIFACT IDENTITY: the node reads the ORIGINATING artifact,
            # so its value must equal the value M7 assembled into the spine. A
            # mismatch would mean M17 is explaining a different number from the
            # one the product shows — recorded, never smoothed over.
            spine_val = resolve_key_path(spine_record, path)
            self.value_identity_checked += 1
            for n in hits:
                if _canon_safe(n.get("value")) != _canon_safe(spine_val):
                    self.value_identity_mismatch.append({
                        "series_id": graph.get("series_id"), "field_path": path,
                        "node_id": n.get("node_id"), "node_value": n.get("value"),
                        "spine_value": spine_val})
        return {"n_spine_fields_emitted": len(leaves),
                "n_spine_fields_explained": n_expl_here,
                "explainable": explainable}

    def report(self) -> dict:
        paths = sorted(self.emitting_all)
        denominator = []
        by_module = {}
        stage_b = []
        n_paths_explained = 0
        for p in paths:
            owner = field_owner(p)
            n_emit = self.emitting_all[p]
            n_expl = self.explained_all[p]
            available = n_expl > 0
            if available:
                n_paths_explained += 1
            reason_code = None
            if not available:
                reason_code = ("not_declared" if owner in ADAPTER_OWNER_MODULES
                               else "no_adapter")
            row = {
                "field_path": p,
                "owner_module": owner,
                "n_series_emitting": n_emit,
                "n_series_explained": n_expl,
                "n_series_emitting_explainable": self.emitting_expl[p],
                "n_series_explained_explainable": self.explained_expl[p],
                "explanation_available": available,
                "adapters": sorted(self.adapters_for_path.get(p, ())),
            }
            if not available:
                row["reason_code"] = reason_code
                row["reason"] = _UNEXPLAINED_REASONS[reason_code]
                stage_b.append({"field_path": p, "owner_module": owner,
                                "n_series_emitting": n_emit,
                                "reason_code": reason_code})
            denominator.append(row)
            m = by_module.setdefault(owner, {"field_paths_total": 0,
                                             "field_paths_explained": 0,
                                             "field_instances_total": 0,
                                             "field_instances_explained": 0,
                                             "has_stage_a_adapter":
                                                 owner in ADAPTER_OWNER_MODULES})
            m["field_paths_total"] += 1
            m["field_instances_total"] += n_emit
            m["field_instances_explained"] += n_expl
            if available:
                m["field_paths_explained"] += 1
        for m in by_module.values():
            m["coverage_by_distinct_path"] = _frac(m["field_paths_explained"],
                                                   m["field_paths_total"])
            m["coverage_by_instance"] = _frac(m["field_instances_explained"],
                                              m["field_instances_total"])
        # (n_series_emitting DESC, field_path ASC) via two stable sorts, so no
        # negation is needed — arithmetic is confined to `_frac`.
        stage_b.sort(key=lambda r: r["field_path"])
        stage_b.sort(key=lambda r: r["n_series_emitting"], reverse=True)

        return {
            "m17_version": M17_VERSION,
            "honesty": honesty_block(),
            "denominator_definition": {
                "artifact": "/".join((ARTIFACT_DIR, SPINE_ARTIFACT)),
                "what_counts_as_an_emitted_field": (
                    "every LEAF key path of every record in the M7 per-series "
                    "spine, unioned over all records. A LIST is a leaf (its "
                    "elements are not enumerated). An empty dict, and a null, are "
                    "leaves at their own path — so a block that is null on a "
                    "compact record contributes its own parent path as well as "
                    "the child paths the full schema emits."),
                "why_published": (
                    "a coverage percentage over a loosely-chosen denominator is "
                    "unfalsifiable. The full enumeration is emitted below so the "
                    "headline number is auditable and Stage B can diff against it."),
                "n_distinct_field_paths": len(paths),
            },
            "populations": {
                "all_series": {
                    "n_series": self.n_series_all,
                    "definition": "every record in the M7 spine"},
                "explainable_series": {
                    "n_series": self.n_series_expl,
                    "definition": ("series for which M5 emitted a regimes.jsonl "
                                   "record, i.e. the engine produced something to "
                                   "explain"),
                    "note": ("the remainder are unfittable/suspect curves for which "
                             "the engine emitted no regime, no features and no "
                             "refusal. That is an ENGINE OUTPUT GAP, not a missing "
                             "M17 adapter, and Stage B cannot close it.")},
                "never_blended": (
                    "these two populations are reported separately and are never "
                    "averaged into one number"),
            },
            "totals": {
                "field_paths_total": len(paths),
                "field_paths_explained": n_paths_explained,
                "coverage_by_distinct_path": _frac(n_paths_explained, len(paths)),
                "field_instances_total_all_series": self.inst_all,
                "field_instances_explained_all_series": self.inst_expl_all,
                "coverage_by_instance_all_series": _frac(self.inst_expl_all,
                                                         self.inst_all),
                "field_instances_total_explainable_series": self.inst_explainable,
                "field_instances_explained_explainable_series":
                    self.inst_expl_explainable,
                "coverage_by_instance_explainable_series":
                    _frac(self.inst_expl_explainable, self.inst_explainable),
                "n_series_with_an_explainable_root": self.n_series_with_root,
            },
            "by_module": by_module,
            "graph_census": {
                "node_kinds": dict(self.node_kind_counts),
                "edge_kinds": dict(self.edge_kind_counts),
                "confidence_kinds": dict(self.confidence_kind_counts),
                "causal_licenses": dict(self.causal_license_counts),
                "note": ("confidence_kinds counts CONFIDENCE OBJECTS, not series. "
                         "Distinct kinds > 1 with no merge step is the machine-"
                         "readable form of 'confidence never collapses'."),
            },
            "cross_artifact_value_identity": {
                "n_checked": self.value_identity_checked,
                "n_mismatched": len(self.value_identity_mismatch),
                "note": ("M17 nodes read the ORIGINATING artifact, not the M7 "
                         "spine. This check proves the explained value is the same "
                         "value the product shows. A non-zero mismatch count would "
                         "mean M17 is explaining a different number."),
                "mismatches": self.value_identity_mismatch[:50],
            },
            "adapters": [
                {"name": a["name"], "source_module": a["source_module"],
                 "artifact": "/".join((ARTIFACT_DIR, a["artifact"])),
                 "scope": a["scope"],
                 "n_declared_keys": len(a["declared_keys"]),
                 "declared_keys": list(a["declared_keys"]),
                 "declared_prefixes": list(a.get("declared_prefixes", ())),
                 "explains_declared": list(a.get("explains_declared", ()))}
                for a in ADAPTERS],
            "stage_a_named_narrowings": [
                {"id": "conformal_interval_not_constructed",
                 "what": ("the conformal UNCERTAINTY node carries the MARGINAL "
                          "quantile verbatim; t50 +/- q is never built"),
                 "why": ("constructing the interval is arithmetic on engine values, "
                         "and selecting the Mondrian stratum needs an assay-endpoint "
                         "(amyloid vs generic) classification M17 is not licensed to "
                         "make. Both would be M17 deriving."),
                 "stage_b": ("have conformal.py emit the per-series stratum key and "
                             "the interval itself, so M17 can read them")},
                {"id": "m2_failure_reason_collapsed",
                 "what": ("m2_fit.py:152 writes ONE string for THREE distinct causes "
                          "(malformed arrays / fewer than 3 points / empty candidate "
                          "bank)"),
                 "why": "the cause exists only in that function's local scope and "
                        "never reaches an artifact, so no adapter can disambiguate it",
                 "stage_b": "widen the upstream record to carry the discriminated cause"},
                {"id": "m15_claims_not_joined",
                 "what": "M15 claim confidence is attached to a CORPUS node only",
                 "why": ("M15's ledger holds fixture-document claims with no "
                         "series_id; joining them to the kinetics corpus would be a "
                         "fabricated link"),
                 "stage_b": "a real PDF front end producing claims that carry a series key"},
                {"id": "fit_provenance_partial",
                 "what": ("node.method.fit_provenance is {available: true} with "
                          "an ARTIFACT ADDRESS for the fitting modules (M2), and "
                          "{available: false} with a reason for methods that fit "
                          "nothing"),
                 "why": ("§2.3 FitProvenance is now emitted by the modules that "
                         "run an optimizer (engine/fit_provenance.py), so M17 "
                         "publishes where the record lives rather than reporting "
                         "it missing. A triage/classify/join method has no "
                         "optimizer run and therefore genuinely has no fit "
                         "provenance — reported as such, not as a gap. M17 still "
                         "derives nothing: it gives the address, not the values."),
                 "stage_b": ("adapters that surface the basin verdict itself as a "
                             "gate node, so a knife-edge fit is visible in the "
                             "explanation tree and not only in the artifact")},
                {"id": "m6_propensity_join_outcome_fields_are_m7s",
                 "what": ("`propensity.available`, `propensity.construct_match` "
                          "and `propensity.reason` get NO node and are "
                          "re-attributed from m6 to m7 in FIELD_OWNER_PREFIXES"),
                 "why": ("all three are synthesized by M7's join gate "
                         "(m7_assemble.py:554-583) and exist at no key path in "
                         "propensity.jsonl — the `reason` string is composed with "
                         "an f-string in local scope and never reaches an "
                         "artifact, so no adapter can byte-compare it. Stated "
                         "plainly because the re-attribution FLATTERS m6: its "
                         "denominator drops 119 -> 116 and it reads 116/116 "
                         "instead of 116/119. The old grouping is recorded here "
                         "so the change is a visible correction, never a quiet "
                         "rebase; m7's denominator grows 7 -> 10 to match."),
                 "stage_b": ("have M7 write the join outcome to an artifact "
                             "(a per-curve join ledger), which would make all "
                             "three explainable and is also the fix for the "
                             "assay bleed named below")},
                {"id": "m6_assay_bleed_fixed_m17_mirrors_the_corrected_join",
                 "what": ("FIXED UPSTREAM. M17 reproduces M7's propensity join "
                          "EXACTLY — the (uniprot_id, assay, construct_id) triple "
                          "first, then the FIRST record for the uniprot alone "
                          "(m7_assemble.py:1045-1046) — and now mirrors its ASSAY "
                          "GATE as well as its join. Every M6 node's source_ref "
                          "names the record actually used, and the M6 templates "
                          "PRINT that record key"),
                 "why": ("this narrowing previously recorded a defect M17 was "
                         "MIRRORING rather than correcting: the uniprot-only "
                         "fallback ignored the assay, so M7 attached blocks "
                         "computed for a different one, and M17 reported them "
                         "because explaining anything other than what the product "
                         "shows is the one thing this module may never do. "
                         "join-policy-1.0 fixed the engine — the M6 grouping key "
                         "gained the assay axis and _propensity_block now refuses a "
                         "cross-assay record — so there is no longer a bleed to "
                         "mirror. The record is KEPT rather than deleted because the "
                         "withdrawn figures below are the only published memory of "
                         "the defect, and because M17 had to change too: mirroring "
                         "the join but not the new gate produced 5 verified "
                         "value-identity mismatches on the first rebuild, which is "
                         "exactly the contradiction the mirror exists to avoid"),
                 "measured": {
                     "population": ("the 1,194 full-schema series in "
                                    "protein_analysis.jsonl"),
                     "definition": (
                         "FALLBACK = the (uniprot_id, condition_vector.assay_type, "
                         "condition_vector.construct_id) triple read from "
                         "curves_triaged.jsonl is not a key of the propensity "
                         "triple index while uniprot_id alone is, i.e. the `or` at "
                         "m7_assemble.py:1045-1046 supplied the record. "
                         "DIFFERENT ASSAY = the resolved record's top-level "
                         "`assay` != the curve's condition_vector.assay_type. "
                         "CONTAMINATED = that record is nonetheless published "
                         "(available: true, construct_match: true). SEQUENCE-AXIS "
                         "BLEED = sequence_axis.available is true AND "
                         "sequence_axis.assay != the curve's assay_type."),
                     "n_fallback_resolved": 87,
                     "n_fallback_with_a_different_assay": 5,
                     "n_contaminated": 0,
                     "n_with_a_differing_sequence_axis_assay": 0,
                     "withdrawn_pre_fix_figures": {
                         "n_fallback_resolved": 96,
                         "n_fallback_with_a_different_assay": 14,
                         "n_presented_as_construct_match_true": 9,
                         "n_with_a_differing_sequence_axis_assay": 7,
                         "note": ("NESTED subsets, 96 > 14 > {9, 7}, never summed. "
                                  "An earlier revision read the 96 as the "
                                  "different-assay count, overstating the defect "
                                  "~7x while contradicting its own follow-on "
                                  "numbers. Retained as history; none of these "
                                  "reproduces against the current build.")},
                     "note": ("the 5 remaining different-assay fallbacks are IAPP "
                              "turbidity curves with no usable t50, so no "
                              "turbidity M6 record can exist for them; the gate "
                              "refuses all 5 and n_contaminated is 0. 8 of the "
                              "pre-fix 9 are now CORRECTLY joined, because the M6 "
                              "re-key created the assay-matched records they "
                              "needed — the fix removed the contamination without "
                              "costing an attachment (n_propensity_attached is "
                              "unchanged at 959)."),
                 },
                 "stage_b": ("none for the bleed itself — it is fixed and pinned by "
                             "a regression test asserting 0 construct-matched "
                             "cross-assay joins. What remains is upstream and "
                             "unrelated: gamma.jsonl carries a protein NAME rather "
                             "than a uniprot id, so M6's surface join is still "
                             "name-based")},
                {"id": "m6_two_author_fields_covered_conditionally",
                 "what": ("FIVE m6-owned spine paths get a node only where an M6 "
                          "record joined, and none on the 148 full-schema series "
                          "where none did: `sequence_axis.available`, "
                          "`sequence_axis.reason`, `propensity.uniprot_id`, "
                          "`propensity.prop_construct_id` and "
                          "`propensity.prop_assay`"),
                 "why": ("all five paths have TWO AUTHORS: M6's own fields when a "
                         "record joined, and M7's synthesis when none did — "
                         "m7_assemble.py:554-563 for the three `propensity.*` "
                         "identity fields and :719 for the two `sequence_axis.*` "
                         "ones. An adapter that claimed them unconditionally would "
                         "attach a source_ref pointing at a record that does not "
                         "exist for those 148 series. No node it cannot "
                         "byte-compare. This is the WHOLE of m6's instance "
                         "shortfall and the arithmetic closes exactly: "
                         "148 x 5 = 740, and m6 is explained on 78,440 of the "
                         "79,180 instances it emits"),
                 "measured": {
                     "population": ("the 1,194 full-schema series in "
                                    "protein_analysis.jsonl"),
                     "definition": (
                         "NO-JOIN = neither the (uniprot_id, "
                         "condition_vector.assay_type, "
                         "condition_vector.construct_id) triple nor uniprot_id "
                         "alone is a key of the propensity index, so "
                         "m7_assemble.py:1045-1046 supplied no record. TWO-AUTHOR "
                         "PATH = a spine leaf path under propensity/sequence_axis "
                         "that is emitted on the no-join series and whose "
                         "field_owner() is m6 (the three M7-authored join-outcome "
                         "paths are re-attributed by FIELD_OWNER_PREFIXES and are "
                         "excluded)."),
                     "n_series_with_no_m6_record": 148,
                     "two_author_paths": ["propensity.prop_assay",
                                          "propensity.prop_construct_id",
                                          "propensity.uniprot_id",
                                          "sequence_axis.available",
                                          "sequence_axis.reason"],
                     "n_series_with_a_cross_assay_refusal": 5,
                     "instance_shortfall": 750,
                     "shortfall_decomposition": {
                         "propensity.prop_assay": 148,
                         "propensity.prop_construct_id": 148,
                         "propensity.uniprot_id": 148,
                         "sequence_axis.available": 153,
                         "sequence_axis.reason": 153},
                     "note": ("an earlier revision named only the two "
                              "`sequence_axis.*` paths; the three `propensity.*` "
                              "identity fields are two-author on exactly the same "
                              "148 series and were missing. Behaviour was correct "
                              "on all five — only the enumeration was short. "
                              "join-policy-1.0 then added a THIRD author case, and "
                              "the shortfall stopped being uniform: 740 = 148 x 5 "
                              "became 750 = 148 x 3 + 153 x 2. On 5 further "
                              "full-schema series the joined M6 record is "
                              "cross-assay, so M7 publishes its OWN refusal block "
                              "and M17 mirrors the gate by emitting nothing — which "
                              "affects the two sequence_axis paths only, never the "
                              "three propensity identity fields. The decomposition "
                              "is published rather than the total alone precisely "
                              "because the total no longer factorises."),
                 },
                 "stage_b": ("have M7 record which author supplied the block, so "
                             "the synthesized half is traceable too")},
                {"id": "m6_surface_gamma_is_not_edged_to_m4",
                 "what": ("no `computed_from` edge is drawn from "
                          "`propensity.surface.gamma_*` to the M4-γ nodes, even "
                          "though M6's surface is a reshaped passthrough of a γ "
                          "record (m6_propensity.py:279-300)"),
                 "why": ("the two γs are usually not the same number: M6 keys γ by "
                         "PROTEIN NAME, last record wins "
                         "(m6_propensity.py:534-537), while M4-γ is keyed per "
                         "CONCENTRATION SERIES. MEASURED, definition below: 304 "
                         "series carry both, but on 99 of them NEITHER estimator "
                         "produced a value and the agreement is trivial. On the "
                         "204 where at least one did, the point estimates DIFFER "
                         "on 174 and agree on 30. The edge would assert a "
                         "provenance link that does not hold"),
                 "measured": {
                     "population": (
                         "spine series whose assembled "
                         "propensity.surface.available is true AND whose "
                         "curves_triaged concentration_series_id resolves to a "
                         "record in gamma.jsonl"),
                     "definition": (
                         "DIFFERS = propensity.surface.gamma_regression.value is "
                         "not byte-equal to that γ record's "
                         "gamma_regression.gamma, where an ABSENT key and a stored "
                         "null are BOTH read as `no value` and are therefore equal "
                         "to each other. The two `gamma_regression` BLOCKS are not "
                         "comparable — M6 renames the fields to {value, ci, "
                         "reliable} — so the comparison is scalar-to-scalar and "
                         "the field pair is named. INFORMATIVE SUB-POPULATION = "
                         "the series where at least one of the two scalars is "
                         "non-null; the rest agree only because neither estimator "
                         "resolved a γ, which is not evidence of a provenance "
                         "link and is reported separately rather than folded in."),
                     "n_carrying_both": 304,
                     "n_both_null_agree_trivially": 100,
                     "n_informative": 204,
                     "n_gamma_point_estimate_differs": 174,
                     "n_agree": 130,
                     "n_agree_on_a_real_value": 30,
                     "decomposition_of_the_304": {
                         "agree_both_null": 100, "agree_both_value": 30,
                         "differ_both_value": 133, "differ_m6_null_only": 38,
                         "differ_m4_null_only": 3},
                     "neighbouring_scalars_differ_of_304": {
                         "gamma_regression.ci": 179,
                         "gamma_regression.reliable": 95,
                         "gamma_global.value": 179,
                         "gamma_global.ci": 179,
                         "gamma_disagreement": 52},
                     "moved_by_join_policy_1_0": (
                         "re-measured after the M6 re-key and the assay-matched γ "
                         "join. Population 303 -> 304; differ 178 -> 174; agree "
                         "125 -> 130. The movement is concentrated in "
                         "differ_m6_null_only (42 -> 38) and agree_both_value "
                         "(26 -> 30): four groups that previously carried a "
                         "foreign-assay γ, or none, now carry their own. "
                         "differ_both_value is UNCHANGED at 133 and the informative "
                         "sub-population is UNCHANGED at 204, so the conclusion is "
                         "untouched — the two γs still disagree wherever both "
                         "resolve, and the edge stays refused."),
                     "note": ("TWO withdrawn figures, both from this narrowing. "
                              "(1) `192 of 337` does not reproduce under this or "
                              "any definition tried. (2) `267 of 303 differ / 36 "
                              "agree` was WRONG and was caught by an internal "
                              "contradiction: the definition promised null == null "
                              "while 99 series have both scalars null, so 36 was "
                              "impossible on its face. Cause: the comparison used "
                              "_canon_safe(resolve_key_path(...)), and 31 of the 52 "
                              "γ records carry gamma_regression.status "
                              "'insufficient_data' and OMIT the `gamma` key, "
                              "whereas M6's surface() writes reg.get('gamma') as an "
                              "explicit null. resolve_key_path returns the _MISSING "
                              "sentinel for the absent key, _canon_safe(_MISSING) is "
                              "Python None, and _canon_safe(None) is the STRING "
                              "'null' — so absent-vs-null compared UNEQUAL on 92 "
                              "series and flipped 89 of them from agree to differ. "
                              "It also made .ci coincide with .value at 267, which "
                              "was the tell. The conclusion (refuse the edge) holds "
                              "under every reading tried; only the integers moved."),
                 },
                 "stage_b": ("have M6 record the concentration_series_id its "
                             "surface γ came from, which makes the edge exact")},
                {"id": "explanations_no_longer_reach_fields_the_compact_schema_never_emits",
                 "what": ("FIXED. An adapter fires on whether ITS artifact "
                          "joined, never on whether the SPINE emits the block, "
                          "so a compact-schema series USED TO receive nodes "
                          "explaining field paths its own result does not "
                          "contain. The SPINE-EMISSION GATE in build_graph now "
                          "removes any node whose declared `explains` paths are "
                          "ALL absent from that series' own result. Measured "
                          "after: 0 orphan nodes corpus-wide"),
                 "why": ("460 of the 1,654 results carry `schema: compact` "
                         "(information_yield signal_only or uninformative) and "
                         "have NO `propensity` and NO `sequence_axis` key at all — "
                         "m7_assemble.py:624-637. The M6 adapters gate on "
                         "`ctx.record('propensity.jsonl') is not None`, which is "
                         "true for 382 of those 460, so they emit 26,547 nodes "
                         "whose `explains` path is absent from the result being "
                         "explained; 323 of the 382 receive a full SCORED "
                         "propensity block. CPAD-TK-1041 is a worked case: 103 M6 "
                         "nodes, 26 of them scored propensity, on a record with no "
                         "`propensity` key. COVERAGE IS NOT AFFECTED — "
                         "CoverageAccumulator.observe iterates only the spine "
                         "record's OWN leaves, so an orphan node is counted "
                         "nowhere and no published percentage moves; the defect is "
                         "in the emitted graph and the rendered explanation. HOW "
                         "IT WAS FIXED, and why the first attempt was rejected: the "
                         "obvious gate — have the ADAPTER suppress a node whose "
                         "explains path is absent — requires the adapter to read "
                         "the spine, and the spine is the coverage DENOMINATOR; an "
                         "adapter told which fields to explain can no longer be "
                         "measured on whether it found them. The alternative, "
                         "re-deriving the tier, was also rejected: no upstream "
                         "artifact CARRIES the schema (checked curves_triaged, "
                         "features, regimes, gamma, fits, metadata_quality, "
                         "propensity, regimes_series — zero such fields), it is "
                         "only RE-DERIVABLE by re-implementing "
                         "m7_assemble.information_yield and gamma_significant, and "
                         "M17 will not hold a second copy of an engine decision "
                         "rule that could drift. WHAT SHIPPED instead is a gate in "
                         "build_graph, NOT in any adapter: the builder receives the "
                         "series' own spine record and drops nodes whose declared "
                         "paths are all absent from it. No adapter can see the "
                         "spine, so the measurement property is intact; nothing is "
                         "derived, since the record decides only whether a node "
                         "SURVIVES, never what it says; and no coverage figure "
                         "moves, because those nodes were counted nowhere"),
                 "measured": {
                     "population": ("the 460 compact-schema series in "
                                    "protein_analysis.jsonl"),
                     "definition": (
                         "ORPHAN NODE = a node that declares at least one "
                         "`explains` path and for which resolve_key_path(spine "
                         "record, p) is _MISSING for EVERY declared p — i.e. it "
                         "explains a field this series does not emit. Counted by "
                         "building each compact series' graph and testing each "
                         "node against that series' own spine record."),
                     "n_compact_series": 460,
                     "n_series_with_a_joined_m6_record": {"full": 1046,
                                                          "compact": 382,
                                                          "total": 1428},
                     "orphan_nodes_remaining": 0,
                     "nodes_gated_by_adapter": {"m6_sequence_axis": 16959,
                                                "m4_gamma": 10177,
                                                "m6_propensity": 8701,
                                                "m1_triage": 5060,
                                                "m5_regimes": 624,
                                                "m4_features": 592},
                     "nodes_gated_total": 42113,
                     "m6_nodes_gated": 25660,
                     "m6_compact_series_affected": 382,
                     "m6_compact_series_with_a_scored_block": 323,
                     "gate_is_load_bearing": (
                         "the by-adapter table is what the gate REMOVES, measured "
                         "by rebuilding each compact series' graph with the gate "
                         "disabled. It is published rather than dropped to zero "
                         "because a fix whose effect is invisible cannot be "
                         "audited, and because the four non-M6 rows show the class "
                         "was M17-wide and pre-existing rather than an M6 defect."),
                     "moved_by_join_policy_1_0": (
                         "re-measured after the assay gate landed. m6_sequence_axis "
                         "17,882 -> 16,959 because the gate suppresses cross-assay "
                         "blocks; m6_propensity 8,665 -> 8,701 because the M6 re-key "
                         "created 4 more records, so more compact series now join. "
                         "Net m6 26,547 -> 25,660, total 43,000 -> 42,113, scored "
                         "series 321 -> 323. The four non-M6 rows are unchanged, "
                         "which is the check that the movement is this change's and "
                         "not drift."),
                     "coverage_impact": "none — observe() iterates spine leaves only",
                     "note": ("this CORRECTS the earlier claim in "
                              "`m6_two_author_fields_covered_conditionally` that M6 "
                              "nodes appear ONLY on the 1,046 series with a joined "
                              "M6 record. The real figure is 1,428 series: 1,046 "
                              "full-schema plus 382 compact-schema. The m4_gamma / "
                              "m1_triage / m5_regimes / m4_features rows are "
                              "PRE-EXISTING and unchanged by this change; they are "
                              "listed so the pattern is not read as an M6 defect."),
                 },
                 "stage_b": ("STAGE B, M17-WIDE, ITS OWN GOVERNED CHANGE: (1) have "
                             "M7 publish the per-series shape decision as an "
                             "upstream fact — a small schema/tier ledger keyed by "
                             "series_id, ideally the same per-curve ledger the "
                             "join-outcome narrowing already asks for — so an "
                             "adapter can gate on a STORED value with no "
                             "derivation and no spine read; (2) move the gate into "
                             "shared machinery (build_graph, not the individual "
                             "adapters) so it binds every adapter at once and the "
                             "m4_gamma / m1_triage / m5_regimes / m4_features "
                             "counts above go to zero with it; (3) add the "
                             "corresponding standing guard — zero orphan nodes "
                             "corpus-wide — so the class cannot come back. Doing "
                             "it inside the M6 adapters would fix 26,547 of 43,000 "
                             "and leave the invariant unstated"),
                 },
                {"id": "structure_features_not_read",
                 "what": "data/processed/structure_features.json (17 MB) is not read",
                 "why": "no Stage-A adapter needs it and streaming it would cost "
                        "memory for no traced field",
                 "stage_b": "an M16 adapter over the small associations artifact"},
            ],
            "stage_b_spec": stage_b,
        }


# ========================================================================== #
# THE BOUNDED SAMPLE
#
# `explanations.jsonl` is a ~90 MB derived view of artifacts already on disk, so
# it is written only under --dump-graphs. Everything that needs to READ a real
# graph reads this sample instead. The selection rule is DETERMINISTIC and
# stated here rather than buried, so the sample is reproducible and auditable:
#
#   1. the two ANCHOR series (the all-branches and typical worked examples);
#   2. one series per distinct M5 `gates_failed` signature — lexicographically
#      first — so every refusal shape in the corpus is represented;
#   3. the first series joined to a gamma record (the only source of the
#      `bootstrap_ci` confidence kind);
#   4. the first series with NO M5 record (the engine-output gap, 414 series).
#
# Ties are broken by sorting series ids lexicographically, so the sample does not
# depend on file order. Capped at SAMPLE_MAX series to keep the file under a
# megabyte.
# ========================================================================== #
SAMPLE_ANCHORS = ("CPAD-TK-1061", "CPAD-TK-1039")
SAMPLE_MAX = 10


def select_sample_series(bundle: dict) -> list:
    """The deterministic stratified sample described above."""
    arts = bundle.get("artifacts") or {}
    ct = (arts.get("curves_triaged.jsonl") or {}).get("data") or {}
    reg = (arts.get("regimes.jsonl") or {}).get("data") or {}
    gam = (arts.get("gamma.jsonl") or {}).get("data") or {}
    chosen = []

    def add(sid):
        if sid and sid in ct and sid not in chosen and len(chosen) < SAMPLE_MAX:
            chosen.append(sid)

    for anchor in SAMPLE_ANCHORS:
        add(anchor)
    signatures = {}
    for sid in sorted(reg):
        gf = resolve_key_path(reg[sid], "mechanistic.gates_failed")
        key = _canon_safe([] if gf is _MISSING else gf)
        if key is not None and key not in signatures:
            signatures[key] = sid
    for key in sorted(signatures):
        add(signatures[key])
    for sid in sorted(ct):
        cs = (ct[sid] or {}).get("concentration_series_id")
        if cs and cs in gam:
            add(sid)
            break
    for sid in sorted(ct):
        if sid not in reg:
            add(sid)
            break
    return chosen


# ========================================================================== #
# CLI
# ========================================================================== #
def _write_jsonl_header(fh, bundle, contracts, is_sample):
    """explanations.jsonl opens with a HEADER record carrying the machine-
    readable honesty block; every following line is one series explanation.
    Consumers should branch on `record_type`."""
    fh.write(_canon({
        "record_type": "header",
        "m17_version": M17_VERSION,
        "is_sample": is_sample,
        "sample_selection_rule": (
            "anchors + one series per distinct M5 gates_failed signature + first "
            "gamma-joined series + first series with no M5 record; ties broken by "
            "lexicographic series_id. See m17_explain.select_sample_series."
            if is_sample else None),
        "regenerate": (
            "every graph here is DERIVED from data/processed artifacts; rebuild "
            "with `python engine/m17_explain.py --dump-graphs` for the full corpus"),
        "engine_build_id": bundle.get("engine_build_id"),
        "node_kinds": list(NODE_KINDS),
        "edge_kinds": list(EDGE_KINDS),
        "confidence_kinds": list(CONFIDENCE_KINDS),
        "causal_licenses": list(CAUSAL_LICENSES),
        "projections": {k: list(v) for k, v in PROJECTIONS.items()},
        "artifact_dir": ARTIFACT_DIR,
        "source_ref_shape": (
            "{artifact, record_key, key_path}; `artifact` is a BARE filename, "
            "resolved against `artifact_dir`"),
        "omitted_defaults": {
            "note": ("these keys are omitted from a node/edge when they hold "
                     "their default value; ABSENT MUST BE READ AS THE DEFAULT"),
            "node.causal_license": "none",
            "edge.causal_license": "none",
            "edge.template": "the edge kind",
            "node.value_kind": "artifact_verbatim",
            "node.label": "the slug segment of node_id",
            "node.method": ("replaced by node.method_ref, which resolves in the "
                            "record's `methods` table"),
        },
        "projections_are_not_stored": (
            "the four projections are VIEWS computed by "
            "m17_explain.project(graph, name) on read; a stored copy could drift "
            "from the graph, so they are deliberately absent from every record"),
        "edge_direction_convention": (
            "an edge points FROM the dependent node TO the thing it depends on"),
        "adapter_contract_ok": contracts.get("all_ok"),
        "missing_artifacts": bundle.get("missing_artifacts", []),
        "honesty": honesty_block(),
    }))
    fh.write("\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="PRISE M17 — Explanation, Provenance & Evidence Graph (Stage A)")
    proc_default = Path(__file__).resolve().parent.parent.joinpath("data", "processed")
    ap.add_argument("--input", type=Path, default=proc_default,
                    help="directory holding the input artifacts")
    ap.add_argument("--output", type=Path, default=None,
                    help="output dir (default: --input)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N spine records")
    ap.add_argument("--dump-graphs", action="store_true",
                    help=("ALSO write the full per-series graph dump "
                          "(explanations.jsonl, ~90 MB). Off by default: the "
                          "dump is a DERIVED view of artifacts already on disk, "
                          "and web/server.py indexes large JSONLs at startup."))
    ap.add_argument("--series", type=str, default=None,
                    help="explain ONE series and print its tree (writes nothing)")
    args = ap.parse_args(argv)

    indir = Path(args.input)
    outdir = Path(args.output) if args.output else indir
    bundle = load_bundle(indir)
    contracts = check_adapter_contracts(bundle, indir)

    # ---- single-series mode: print, write nothing --------------------------- #
    if args.series:
        ct = (bundle["artifacts"].get("curves_triaged.jsonl") or {}).get("data") or {}
        rec = ct.get(args.series) or {}
        # single-series mode must apply the SAME emission gate as the corpus run,
        # or the CLI would print explanations the written artifact does not contain
        spine_rec = None
        for _s in _iter_jsonl(indir.joinpath(SPINE_ARTIFACT)):
            if _s.get(SPINE_RECORD_KEY) == args.series:
                spine_rec = _s
                break
        graph = build_graph(args.series, bundle, join_keys_for_series(rec),
                            spine_record=spine_rec)
        rendered = explain_series(graph)
        print(f"[M17] series {args.series}: {graph['n_nodes']} nodes / "
              f"{graph['n_edges']} edges, acyclic={graph['acyclic']}")
        if rendered.get("explanation_available"):
            print(rendered["text"])
        else:
            print(_canon(rendered))
        return 0

    outdir.mkdir(parents=True, exist_ok=True)
    expl_path = outdir.joinpath("explanations.jsonl")
    sample_path = outdir.joinpath("explanations_sample.jsonl")
    cov_path = outdir.joinpath("explanation_coverage.json")

    ct_index = (bundle["artifacts"].get("curves_triaged.jsonl") or {}).get("data") or {}
    reg_index = (bundle["artifacts"].get("regimes.jsonl") or {}).get("data") or {}
    sample_ids = set(select_sample_series(bundle))
    acc = CoverageAccumulator()
    n = 0

    corpus_graph = build_graph("__corpus__", bundle, None, scope="corpus")
    corpus_graph["record_type"] = "corpus"

    sink = open(sample_path, "w", encoding="utf-8")
    dump = open(expl_path, "w", encoding="utf-8") if args.dump_graphs else None
    try:
        _write_jsonl_header(sink, bundle, contracts, True)
        sink.write(_canon(corpus_graph))
        sink.write("\n")
        if dump is not None:
            _write_jsonl_header(dump, bundle, contracts, False)
            dump.write(_canon(corpus_graph))
            dump.write("\n")
        # STREAM the spine so no more than one M7 record is ever held. Coverage
        # is accumulated over EVERY series regardless of what is written, so the
        # measured numbers do not depend on --dump-graphs.
        for spine in _iter_jsonl(indir.joinpath(SPINE_ARTIFACT)):
            sid = spine.get(SPINE_RECORD_KEY)
            if sid is None:
                continue
            if args.limit is not None and n >= args.limit:
                break
            n += 1
            triage = ct_index.get(sid) or {}
            # the spine record is passed for the EMISSION GATE only — no adapter
            # sees it, and it decides node survival, never a node's value
            graph = build_graph(sid, bundle, join_keys_for_series(triage),
                                spine_record=spine)
            explainable = sid in reg_index
            graph["record_type"] = "explanation"
            graph["coverage"] = acc.observe(spine, graph, explainable)
            # The rendered TEXT is derived from the graph by explain_series();
            # only its availability + root are stored, for the same reason the
            # projections are not stored.
            rendered = explain_series(graph)
            graph["explanation"] = {k: v for k, v in rendered.items()
                                    if k != "text"}
            graph["honesty_block_in"] = "/".join((ARTIFACT_DIR,
                                                  "explanation_coverage.json"))
            line = _canon(graph)
            if sid in sample_ids:
                sink.write(line)
                sink.write("\n")
            if dump is not None:
                dump.write(line)
                dump.write("\n")
    finally:
        sink.close()
        if dump is not None:
            dump.close()

    report = acc.report()
    report["adapter_contract_check"] = {
        "n_checked": contracts["n_checked"], "n_failed": contracts["n_failed"],
        "all_ok": contracts["all_ok"], "failures": contracts["failures"]}
    report["outputs"] = {
        "explanation_coverage.json": "always written; this file",
        "explanations_sample.jsonl": {
            "always_written": True,
            "series": sorted(sample_ids),
            "selection_rule": (
                "anchors + one series per distinct M5 gates_failed signature + "
                "first gamma-joined series + first series with no M5 record; "
                "ties broken by lexicographic series_id"),
        },
        "explanations.jsonl": {
            "written": bool(args.dump_graphs),
            "flag": "--dump-graphs",
            "why_not_default": (
                "a per-series graph is DERIVED deterministically from artifacts "
                "already on disk (~10 ms/series), so the full dump is a stored "
                "view of stored data — the same category error as serialising "
                "the projections. It is also ~90 MB, and web/server.py indexes "
                "large per-series JSONLs into memory at startup, so shipping it "
                "by default would add hundreds of MB to the boot footprint of a "
                "zero-install local demo. Per-series trees are built on demand."),
        },
    }
    report["inputs"] = {"input_dir": str(indir),
                        "missing_artifacts": bundle.get("missing_artifacts", []),
                        "engine_build_id": bundle.get("engine_build_id")}
    cov_path.write_text(json.dumps(report, indent=2, sort_keys=True),
                        encoding="utf-8")

    # ---- printed summary ---------------------------------------------------- #
    t = report["totals"]
    print(f"[M17] wrote {cov_path}")
    print(f"[M17] wrote {sample_path} ({len(sample_ids)} series, deterministic sample)")
    if args.dump_graphs:
        print(f"[M17] wrote {expl_path} (full dump, --dump-graphs)")
    else:
        print("[M17] full per-series dump NOT written (derived view; "
              "use --dump-graphs)")
    print(f"[M17] {M17_VERSION}  engine_build_id={bundle.get('engine_build_id')}")
    print(f"[M17] adapter contract: {contracts['n_checked']} declared keys checked, "
          f"{contracts['n_failed']} failed")
    print(f"\n  DENOMINATOR: {t['field_paths_total']} distinct M7-spine field paths "
          f"(published in full in explanation_coverage.json)")
    print(f"  series: {report['populations']['all_series']['n_series']} total, "
          f"{report['populations']['explainable_series']['n_series']} explainable "
          f"(M5 emitted a record)")
    print("\n  COVERAGE (two numbers, never blended):")
    print(f"    by distinct field path      : {t['field_paths_explained']}"
          f"/{t['field_paths_total']} = {t['coverage_by_distinct_path']}")
    print(f"    by field instance, ALL      : {t['field_instances_explained_all_series']}"
          f"/{t['field_instances_total_all_series']} = "
          f"{t['coverage_by_instance_all_series']}")
    print(f"    by field instance, EXPLAINABLE: "
          f"{t['field_instances_explained_explainable_series']}"
          f"/{t['field_instances_total_explainable_series']} = "
          f"{t['coverage_by_instance_explainable_series']}")
    print("\n  BY MODULE (distinct paths explained / total, adapter?):")
    for mod in sorted(report["by_module"]):
        m = report["by_module"][mod]
        print(f"    {mod:<12} {m['field_paths_explained']:>3}/{m['field_paths_total']:<3} "
              f"= {str(m['coverage_by_distinct_path']):<8} adapter={m['has_stage_a_adapter']}")
    gc = report["graph_census"]
    print(f"\n  GRAPH CENSUS: nodes {dict(sorted(gc['node_kinds'].items()))}")
    print(f"                edges {dict(sorted(gc['edge_kinds'].items()))}")
    print(f"                confidence kinds {dict(sorted(gc['confidence_kinds'].items()))}")
    ci = report["cross_artifact_value_identity"]
    print(f"\n  CROSS-ARTIFACT VALUE IDENTITY: {ci['n_mismatched']} mismatched of "
          f"{ci['n_checked']} checked")
    print(f"  STAGE B SPEC: {len(report['stage_b_spec'])} unexplained field paths "
          f"named, {len(report['stage_a_named_narrowings'])} named narrowings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
