"""
Tests for M17 — Explanation, Provenance & Evidence-Graph Layer, Stage A.

Deterministic, stdlib-only, no pytest (run it and read the exit code; 0 = pass).
Some tests are HERMETIC (synthetic in-memory bundles); the rest run against the
REAL corpus in data/processed, because the load-bearing claims of this module —
"every number byte-compares to its artifact", "the worked example resolves",
"coverage is measured" — are only meaningful on real artifacts.

What is pinned here:

  * M17 DERIVES NOTHING — every non-null node value byte-compares equal to the
    artifact value at its declared key path, re-read FROM DISK (not from the
    pruned in-memory bundle, so pruning cannot hide a bad ref); and `_frac` is
    the only function in the module containing arithmetic (checked with `ast`).
  * DANGLING source_ref -> FAIL. A deliberately corrupted ref must be caught.
  * MUTATION: change one artifact value -> the tree changes. Change nothing ->
    the rendered text is BYTE-IDENTICAL.
  * ACYCLICITY, and a UNIQUE (not merely valid) topological order.
  * CLOSED node / edge / confidence / causal-licence vocabularies.
  * A `refusal` node never renders an affirmative sentence, and never names a
    mechanism from the equivalence class it is refusing to choose within.
  * CONFIDENCE never collapses to a scalar; >= 4 incommensurable kinds appear.
  * COVERAGE matches reality, and the two populations are never blended.
  * THE WORKED EXAMPLE resolves end to end on the live corpus (CPAD-TK-1061),
    and the TYPICAL shape (CPAD-TK-1039) does NOT invent an assay blocker.
  * ADAPTER CONTRACTS: every declared key path exists in the live artifacts.

    python engine/test_m17_explain.py
"""
from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path

import m17_explain as m17

PROC = Path(__file__).resolve().parent.parent.joinpath("data", "processed")

# The two anchor series, chosen from the real corpus:
#   ALL-BRANCHES: turbidity, assay_reports_mass=false, no concentration series,
#                 agitation+seeding unknown -> every branch of the worked
#                 example is present.
#   TYPICAL:      ThT, assay_reports_mass=true -> the assay gate PASSES, so the
#                 assay blocker must be ABSENT. 1,209 of 1,240 series look like
#                 this, so inventing a blocker here is the failure mode that
#                 actually threatens M17.
SERIES_ALL_BRANCHES = "CPAD-TK-1061"
SERIES_TYPICAL = "CPAD-TK-1039"

_BUNDLE = None
_GRAPHS = {}


def _bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = m17.load_bundle(PROC)
    return _BUNDLE


def _graph(series_id):
    if series_id not in _GRAPHS:
        b = _bundle()
        ct = (b["artifacts"].get("curves_triaged.jsonl") or {}).get("data") or {}
        rec = ct.get(series_id) or {}
        _GRAPHS[series_id] = m17.build_graph(series_id, b,
                                             m17.join_keys_for_series(rec))
    return _GRAPHS[series_id]


def _corpus_available():
    return PROC.joinpath("regimes.jsonl").exists()


def _node(graph, node_id):
    for n in graph["nodes"]:
        if n["node_id"] == node_id:
            return n
    return None


def _edges_of(graph, node_id, kind=None):
    return [e for e in graph["edges"]
            if e["from"] == node_id and (kind is None or e["kind"] == kind)]


# --------------------------------------------------------------------------- #
# 1. closed vocabularies
# --------------------------------------------------------------------------- #
def test_node_vocabulary_is_closed_and_exact():
    assert m17.NODE_KINDS == (
        "result", "datum", "metadata_fact", "method", "assumption", "gate",
        "refusal", "uncertainty", "deferral", "ceiling")
    assert isinstance(m17.NODE_KINDS, tuple)      # ordered, not a set
    assert m17.NODE_KIND_SET == frozenset(m17.NODE_KINDS)


def test_edge_vocabulary_is_closed_and_exact():
    assert m17.EDGE_KINDS == (
        "computed_from", "assumes", "gated_by", "blocked_by",
        "uncertainty_from", "bounded_by", "unblocked_by")
    assert isinstance(m17.EDGE_KINDS, tuple)
    assert m17.EDGE_KIND_SET == frozenset(m17.EDGE_KINDS)


def test_confidence_and_causal_vocabularies_are_closed():
    assert m17.CONFIDENCE_KINDS == (
        "bootstrap_ci", "conformal_quantile", "claim_confidence_calibrated",
        "completeness_presence", "shape_confidence_ordinal")
    assert m17.CAUSAL_LICENSES == (
        "none", "deterministic_engine_logic", "statistical_only",
        "measured_causal")


def test_every_edge_kind_is_visible_in_at_least_one_projection():
    """Four VIEWS of one DAG. If an edge kind appeared in no projection it would
    be invisible to every consumer while still shaping the graph."""
    seen = set()
    for kinds in m17.PROJECTIONS.values():
        seen.update(kinds)
    assert seen == m17.EDGE_KIND_SET, sorted(m17.EDGE_KIND_SET - seen)


def test_projections_are_derived_not_stored():
    """A stored projection could drift from its graph; a function call cannot."""
    g = _graph(SERIES_TYPICAL) if _corpus_available() else None
    if g is None:
        return
    assert "projections" not in g
    for name in m17.PROJECTIONS:
        p = m17.project(g, name)
        assert p["acyclic"] is True
        for a, kind, b in p["edges"]:
            assert kind in m17.PROJECTIONS[name]


# --------------------------------------------------------------------------- #
# 2. M17 derives nothing — the mechanical guards
# --------------------------------------------------------------------------- #
def test_frac_is_the_only_arithmetic_in_the_module():
    """The mechanical form of 'M17 derives nothing': if the module cannot do
    arithmetic outside one audited counting helper, it cannot compute an engine
    value even by accident."""
    src = Path(m17.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = (ast.Div, ast.Mult, ast.Pow, ast.FloorDiv, ast.Mod, ast.MatMult,
              ast.Sub)
    offenders = []
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for node in ast.walk(fn):
            if isinstance(node, ast.BinOp) and isinstance(node.op, banned):
                if fn.name != "_frac":
                    offenders.append((fn.name, type(node.op).__name__,
                                      node.lineno))
    assert not offenders, offenders


def test_module_imports_no_numpy_or_scipy():
    """Every float in the graph must arrive via json.loads as a plain Python
    float — a numpy scalar's repr could drift between builds and silently break
    byte-identity."""
    src = Path(m17.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "numpy" not in imported and "scipy" not in imported, sorted(imported)
    assert not hasattr(m17, "np")


def test_render_node_cannot_compose_a_sentence_at_runtime():
    """`render_node` must be a table lookup + one format_map. No concatenation,
    no f-string, no join — so free-form prose is impossible by construction.

    The guard covers `_template_fields` TOO. Scoping it to the name `render_node`
    alone left the substitution table un-policed: a sentence assembled one frame
    down would satisfy a guard that only reads the caller, which is the blind spot
    rather than the invariant. Both frames, one rule."""
    src = Path(m17.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    guarded = ("render_node", "_template_fields")
    fns = {n.name: n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name in guarded}
    assert set(fns) == set(guarded), sorted(fns)   # a renamed frame must not slip out
    for name, fn in sorted(fns.items()):
        for node in ast.walk(fn):
            assert not isinstance(node, ast.JoinedStr), "f-string in " + name
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                raise AssertionError("string concatenation in " + name)
            if isinstance(node, ast.Attribute):
                assert node.attr != "join", "str.join in " + name


def test_no_derivation_on_the_live_corpus():
    """THE load-bearing check, on real artifacts: every non-null node value
    byte-compares equal to the artifact value at its declared key path, re-read
    from disk."""
    if not _corpus_available():
        return
    resolver = m17.RawResolver(PROC)
    for sid in (SERIES_ALL_BRANCHES, SERIES_TYPICAL):
        g = _graph(sid)
        violations = m17.assert_no_derivation(g, resolver)
        assert not violations, violations[:5]


def test_no_derivation_catches_a_planted_fabricated_value():
    """Negative control: if the guard cannot catch a number M17 made up, it is
    not a guard."""
    if not _corpus_available():
        return
    g = copy.deepcopy(_graph(SERIES_TYPICAL))
    victim = next(n for n in g["nodes"]
                  if isinstance(n.get("value"), float))
    victim["value"] = 123456.789            # a value present in no artifact
    violations = m17.assert_no_derivation(g, m17.RawResolver(PROC))
    assert any(v.get("issue") == "value_differs_from_artifact"
               for v in violations), violations[:5]


# --------------------------------------------------------------------------- #
# 3. source_ref traceability
# --------------------------------------------------------------------------- #
def test_every_source_ref_resolves_against_the_raw_artifact_file():
    if not _corpus_available():
        return
    resolver = m17.RawResolver(PROC)
    for sid in (SERIES_ALL_BRANCHES, SERIES_TYPICAL):
        dangling = m17.validate_source_refs(_graph(sid), resolver)
        assert not dangling, dangling[:5]


def test_dangling_source_ref_fails():
    """A ref that points nowhere must be REPORTED, not tolerated: an
    untraceable explanation is worse than no explanation."""
    if not _corpus_available():
        return
    g = copy.deepcopy(_graph(SERIES_TYPICAL))
    g["nodes"][0]["source_ref"] = {"artifact": "regimes.jsonl",
                                   "record_key": g["series_id"],
                                   "key_path": "no.such.key.path"}
    dangling = m17.validate_source_refs(g, m17.RawResolver(PROC))
    assert any(d.get("issue") == "dangling_source_ref" for d in dangling)


def test_unregistered_or_path_shaped_artifact_is_rejected():
    if not _corpus_available():
        return
    g = copy.deepcopy(_graph(SERIES_TYPICAL))
    g["nodes"][0]["source_ref"] = {"artifact": "C:\\data\\regimes.jsonl",
                                   "key_path": "series_id"}
    dangling = m17.validate_source_refs(g, m17.RawResolver(PROC))
    assert any(d.get("issue") == "non_portable_or_unregistered_artifact"
               for d in dangling)


def test_source_refs_are_bare_portable_filenames():
    if not _corpus_available():
        return
    for n in _graph(SERIES_TYPICAL)["nodes"]:
        ref = n.get("source_ref")
        if not ref:
            continue
        art = ref["artifact"]
        assert "\\" not in art and "/" not in art and ":" not in art
        assert art in m17.ARTIFACTS


def test_node_source_reassembles_series_and_study_provenance():
    """`source` is not stored per node (it is identical for every node in a
    record); it must still be reconstructible in full."""
    if not _corpus_available():
        return
    g = _graph(SERIES_TYPICAL)
    n = _node(g, m17.NID_META_AGITATION)
    src = m17.node_source(g, n)
    assert src["series_id"] == SERIES_TYPICAL
    assert src["artifact"] == "data/processed/curves_triaged.jsonl"
    assert src["key_path"] == "condition_vector.field_provenance.agitation"
    assert (src["source_study"] or {}).get("pmid")     # CPAD provenance present


# --------------------------------------------------------------------------- #
# 4. determinism + mutation
# --------------------------------------------------------------------------- #
def test_same_graph_renders_byte_identical_text():
    if not _corpus_available():
        return
    g = _graph(SERIES_ALL_BRANCHES)
    a = m17.render_tree(g, m17.NID_REFUSAL_MECHANISM)
    b = m17.render_tree(copy.deepcopy(g), m17.NID_REFUSAL_MECHANISM)
    assert a == b
    assert a.encode("utf-8") == b.encode("utf-8")


def test_rebuilding_from_the_same_bundle_is_byte_identical():
    if not _corpus_available():
        return
    b = _bundle()
    ct = (b["artifacts"]["curves_triaged.jsonl"]["data"])
    join = {"concentration_series_id":
            (ct.get(SERIES_ALL_BRANCHES) or {}).get("concentration_series_id")}
    g1 = m17.build_graph(SERIES_ALL_BRANCHES, b, join)
    g2 = m17.build_graph(SERIES_ALL_BRANCHES, b, join)
    assert m17._canon(g1) == m17._canon(g2)


def test_mutating_an_artifact_value_changes_the_tree():
    """The explanation must TRACK the engine. If flipping the agitation
    provenance to 'known' left the rendered tree unchanged, the tree would be
    decoration rather than evidence."""
    if not _corpus_available():
        return
    b = _bundle()
    mutated = copy.deepcopy(b)
    rec = mutated["artifacts"]["curves_triaged.jsonl"]["data"][SERIES_ALL_BRANCHES]
    before = rec["condition_vector"]["field_provenance"]["agitation"]
    rec["condition_vector"]["field_provenance"]["agitation"] = "known"
    ct = b["artifacts"]["curves_triaged.jsonl"]["data"]
    join = {"concentration_series_id":
            (ct.get(SERIES_ALL_BRANCHES) or {}).get("concentration_series_id")}
    g_before = m17.build_graph(SERIES_ALL_BRANCHES, b, join)
    g_after = m17.build_graph(SERIES_ALL_BRANCHES, mutated, join)
    t_before = m17.render_tree(g_before, m17.NID_REFUSAL_MECHANISM)
    t_after = m17.render_tree(g_after, m17.NID_REFUSAL_MECHANISM)
    assert before == "unknown"
    assert t_before != t_after
    assert "NOT RECORDED" in t_before


def test_topological_order_is_unique_not_merely_valid():
    """Kahn with a sorted ready queue: shuffling the input node order must not
    change the emitted order."""
    if not _corpus_available():
        return
    g = _graph(SERIES_ALL_BRANCHES)
    ids = [n["node_id"] for n in g["nodes"]]
    o1, ok1 = m17._topological_order(ids, g["edges"])
    o2, ok2 = m17._topological_order(list(reversed(ids)), g["edges"])
    assert ok1 and ok2
    assert o1 == o2
    pos = {nid: i for i, nid in enumerate(o1)}
    for e in g["edges"]:                       # every dependency precedes its user
        if e["from"] != e["to"]:
            assert pos[e["to"]] < pos[e["from"]], e


def test_graph_is_acyclic_and_has_no_dropped_or_conflicting_parts():
    if not _corpus_available():
        return
    for sid in (SERIES_ALL_BRANCHES, SERIES_TYPICAL):
        g = _graph(sid)
        assert g["acyclic"] is True
        assert g["dropped_edges"] == [], g["dropped_edges"][:3]
        assert g["node_conflicts"] == [], g["node_conflicts"][:3]
        assert g["adapter_errors"] == [], g["adapter_errors"][:3]
        assert not m17.validate_vocabulary(g)


# --------------------------------------------------------------------------- #
# 5. the refusal never speaks affirmatively
# --------------------------------------------------------------------------- #
_MECHANISM_NAMES = ("primary_nucleation_dominated", "secondary_nucleation_dominated",
                    "fragmentation_dominated", "saturating_secondary_nucleation")


def test_every_refusal_template_opens_with_a_refusal_marker():
    for key, tpl in m17.TEMPLATES.items():
        if key.startswith("refusal."):
            assert tpl.startswith("REFUSED:"), (key, tpl[:40])


def test_refusal_node_never_renders_an_affirmative_sentence():
    if not _corpus_available():
        return
    g = _graph(SERIES_ALL_BRANCHES)
    node = _node(g, m17.NID_REFUSAL_MECHANISM)
    text = m17.render_node(node)["text"]
    assert text.startswith("REFUSED:")
    # It must not name a mechanism: naming one inside a refusal to name one is
    # precisely the failure an over-eager narrator would produce.
    for name in _MECHANISM_NAMES:
        assert name not in text, name
    for affirmative in ("the mechanism is", "we conclude", "therefore the",
                        "demonstrates", "shows that", "is driven by"):
        assert affirmative not in text.lower()


def test_equivalence_class_is_framed_as_what_cannot_be_distinguished():
    """The mechanism names DO appear here — legitimately, and only inside an
    explicitly negative frame."""
    if not _corpus_available():
        return
    g = _graph(SERIES_ALL_BRANCHES)
    node = _node(g, "result:m5_regimes:mechanistic.equivalence_class")
    text = m17.render_node(node)["text"]
    assert "CANNOT DISTINGUISH" in text
    assert "narrowed = false" in text


def test_statistical_only_node_can_never_render_a_causal_verb():
    """No live node carries `statistical_only` in Stage A (M10/M16 have no
    adapter), so the guard is exercised on a synthetic node — otherwise it
    would be untested until the day it first matters."""
    node = m17.make_node(
        "result", "m10_crossmodal", "apr_vs_regime", "APR count vs regime",
        "result",                              # adapter asks for the plain one
        value=0.05, source_ref=m17._ref("m10_crossmodal.json", None, "p_value"),
        causal_license="statistical_only")
    text = m17.render_node(node)["text"]
    assert "ASSOCIATIVE ONLY" in text
    for verb in ("causes", "caused by", "drives", "leads to", "results in",
                 "due to", "because"):
        assert verb not in text.lower(), verb


def test_measured_causal_licence_is_declared_but_unused_on_the_live_corpus():
    """Nothing in this engine licenses a causal claim about nature."""
    if not _corpus_available():
        return
    report = json.loads(PROC.joinpath("explanation_coverage.json")
                        .read_text(encoding="utf-8"))
    counts = report["graph_census"]["causal_licenses"]
    assert counts.get("measured_causal", 0) == 0
    assert counts.get("statistical_only", 0) == 0


def test_every_node_and_edge_kind_has_at_least_one_live_instance():
    """STANDING GUARD, added because of a real incident during this build.

    A bug in `_assign_path` created a dict where a list was needed, which
    silently pruned every selector-addressed fact out of the loaded bundle. The
    consequence was that the ENTIRE `unblocked_by` edge kind disappeared from
    every graph in the corpus — the refusal no longer named what would unblock
    it — and NOTHING FAILED. The suite was green while the explanation layer had
    quietly lost a whole category of evidence.

    `test_every_edge_kind_is_visible_in_at_least_one_projection` does not catch
    this: it is STRUCTURAL, proving the vocabulary is fully viewable, not that
    the graph actually CONTAINS each kind. Silently losing an entire kind is
    precisely the quiet degradation this module exists to prevent, so the census
    is published in explanation_coverage.json and asserted here."""
    if not _corpus_available():
        return
    report = json.loads(PROC.joinpath("explanation_coverage.json")
                        .read_text(encoding="utf-8"))
    census = report["graph_census"]

    missing_nodes = [k for k in m17.NODE_KINDS
                     if not census["node_kinds"].get(k)]
    assert not missing_nodes, (
        "node kinds with ZERO live instances (see docstring — this is how the "
        "unblocked_by edges vanished silently): %r" % (missing_nodes,))

    missing_edges = [k for k in m17.EDGE_KINDS
                     if not census["edge_kinds"].get(k)]
    assert not missing_edges, (
        "edge kinds with ZERO live instances: %r" % (missing_edges,))

    # the census must describe the WHOLE corpus, not just what was dumped
    assert sum(census["node_kinds"].values()) > 100000
    assert census["edge_kinds"]["unblocked_by"] > 1000


def test_census_is_published_in_the_coverage_artifact():
    """The guard above is only as good as the data it reads, so the census must
    actually be in the shipped artifact rather than recomputed by the test."""
    if not _corpus_available():
        return
    report = json.loads(PROC.joinpath("explanation_coverage.json")
                        .read_text(encoding="utf-8"))
    census = report["graph_census"]
    for key in ("node_kinds", "edge_kinds", "confidence_kinds",
                "causal_licenses"):
        assert key in census and census[key]
    for k in census["node_kinds"]:
        assert k in m17.NODE_KIND_SET
    for k in census["edge_kinds"]:
        assert k in m17.EDGE_KIND_SET


# --------------------------------------------------------------------------- #
# 6. confidence never collapses
# --------------------------------------------------------------------------- #
def test_confidence_is_always_a_list_of_tagged_objects():
    if not _corpus_available():
        return
    for sid in (SERIES_ALL_BRANCHES, SERIES_TYPICAL):
        for n in _graph(sid)["nodes"]:
            conf = n.get("confidence")
            if conf is None:
                continue
            assert isinstance(conf, list), n["node_id"]
            for c in conf:
                assert isinstance(c, dict)
                assert set(c.keys()) == {"kind", "value", "basis", "valid_at"}
                assert c["kind"] in m17.CONFIDENCE_KIND_SET


def test_no_merged_confidence_scalar_anywhere_in_the_emitted_artifacts():
    """A merged scalar would fabricate a number across incommensurable
    guarantees and violate invariant §10.5."""
    if not _corpus_available():
        return
    banned = ("combined_confidence", "overall_confidence", "confidence_score",
              "mean_confidence", "merged_confidence", "confidence_total")
    text = PROC.joinpath("explanation_coverage.json").read_text(encoding="utf-8")
    for b in banned:
        assert b not in text, b
    with PROC.joinpath("explanations_sample.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            for b in banned:
                assert b not in line, b


def test_at_least_four_incommensurable_confidence_kinds_appear():
    if not _corpus_available():
        return
    report = json.loads(PROC.joinpath("explanation_coverage.json")
                        .read_text(encoding="utf-8"))
    kinds = {k for k, v in report["graph_census"]["confidence_kinds"].items()
             if v}
    assert len(kinds) >= 4, kinds
    assert "bootstrap_ci" in kinds            # asymptotic
    assert "conformal_quantile" in kinds      # finite-sample valid
    assert "completeness_presence" in kinds   # PRESENCE, not correctness


def test_m15_claim_confidence_exists_but_is_corpus_scoped_only():
    """Joining 59 fixture-document claims to the kinetics corpus would be a
    fabricated link, so the kind must appear ONLY on the corpus record."""
    if not _corpus_available():
        return
    with PROC.joinpath("explanations_sample.jsonl").open(encoding="utf-8") as fh:
        fh.readline()
        corpus = json.loads(fh.readline())
    assert corpus["record_type"] == "corpus"
    kinds = [c["kind"] for n in corpus["nodes"] for c in n.get("confidence", [])]
    assert "claim_confidence_calibrated" in kinds
    node = corpus["nodes"][0]
    valid_at = node["confidence"][0]["valid_at"]
    assert "FIXTURE" in valid_at and "NOT" in valid_at


# --------------------------------------------------------------------------- #
# 7. counterfactuals are gated on gain_basis
# --------------------------------------------------------------------------- #
def test_service_c_measured_edge_licenses_a_counterfactual():
    if not _corpus_available():
        return
    g = _graph(SERIES_ALL_BRANCHES)
    by_id = {n["node_id"]: n for n in g["nodes"]}
    edges = _edges_of(g, m17.NID_REFUSAL_MECHANISM, "unblocked_by")
    assert edges, "the refusal must name what would unblock it"
    measured = [e for e in edges if e.get("gain_basis") == "service_c_measured"]
    assert measured
    for e in measured:
        text = m17.render_edge(e, by_id)["text"]
        assert "MEASURED" in text and "quantitative counterfactual is licensed" in text


def test_structural_estimate_edge_renders_as_an_upper_bound():
    """A structural_estimate gain is a DAG upper bound, not a measurement, and
    must inherit M8's validity ceiling."""
    edge = m17.make_edge("a", "b", "unblocked_by", "m11_metadata",
                         gain_basis="structural_estimate",
                         template="unblocked_by.structural_estimate")
    text = m17.render_edge(edge, {"b": {"node_id": "b", "label": "record_pH"}})["text"]
    assert "UPPER BOUND" in text
    assert "not measured" in text
    assert "M8 validity ceiling" in text


def test_absent_gain_basis_licenses_no_counterfactual():
    edge = m17.make_edge("a", "b", "unblocked_by", "m11_metadata")
    text = m17.render_edge(edge, {"b": {"node_id": "b", "label": "x"}})["text"]
    assert "NO quantitative counterfactual is licensed" in text


def test_validity_ceiling_is_present_and_inherited():
    if not _corpus_available():
        return
    g = _graph(SERIES_ALL_BRANCHES)
    ceiling = _node(g, m17.NID_CEILING_M8)
    assert ceiling is not None and ceiling["kind"] == "ceiling"
    assert "UPPER bounds" in m17.render_node(ceiling)["text"]
    inheritors = [e for e in g["edges"]
                  if e["to"] == m17.NID_CEILING_M8 and e["kind"] == "bounded_by"]
    assert inheritors, "blockers/actions must inherit the ceiling"


# --------------------------------------------------------------------------- #
# 8. THE WORKED EXAMPLE — end to end on the live corpus
# --------------------------------------------------------------------------- #
def test_worked_example_resolves_end_to_end():
    """Every branch of the design's worked example, on a real series."""
    if not _corpus_available():
        return
    g = _graph(SERIES_ALL_BRANCHES)

    refusal = _node(g, m17.NID_REFUSAL_MECHANISM)
    assert refusal is not None and refusal["kind"] == "refusal"
    assert refusal["value"] is None                     # single_mechanism_call

    # gated_by  GATE mechanistic_inference_licensed = false
    assert _edges_of(g, m17.NID_REFUSAL_MECHANISM, "gated_by")
    gate = _node(g, m17.NID_GATE_MECH_LICENSE)
    assert gate["value"] is False

    # blocked_by  META agitation / seeded = "unknown", via the M8 blocker whose
    # scope is CORPUS_WIDE and which blocks the mechanistic_licensed rung
    agit_gate = _node(g, "gate:m5_regimes:"
                         "agitation_or_seeding_unknown_branch_undetermined")
    assert agit_gate is not None
    blocker = _node(g, m17.NID_BLOCKER_AGIT_SEED)
    assert blocker is not None
    detail = json.loads(blocker["detail"])
    assert detail["scope"] == "corpus_wide"
    assert detail["blocks_rung"] == "mechanistic_licensed"
    leaves = {e["to"] for e in _edges_of(g, m17.NID_BLOCKER_AGIT_SEED, "blocked_by")}
    assert m17.NID_META_AGITATION in leaves and m17.NID_META_SEEDED in leaves
    assert _node(g, m17.NID_META_AGITATION)["value"] == "unknown"
    assert _node(g, m17.NID_META_SEEDED)["value"] == "unknown"

    # blocked_by  META assay_reports_mass != true  -> assumes ThT-proportional
    assert _node(g, m17.NID_DATUM_ASSAY_MASS)["value"] is False
    assumes = {e["to"] for e in _edges_of(g, m17.NID_DATUM_ASSAY_MASS, "assumes")}
    assert m17.NID_ASSUMPTION_DYE_MASS in assumes
    assumption = _node(g, m17.NID_ASSUMPTION_DYE_MASS)
    assert assumption["kind"] == "assumption"
    assert assumption["value"] is None                  # never measured
    assert "PRISE_DESIGN.md §10.9" in assumption["assumptions"]

    # blocked_by  DATA no concentration series, blocking the SCALING rung
    prereqs = {e["to"] for e in _edges_of(g, m17.NID_REFUSAL_MECHANISM, "blocked_by")}
    cs = "metadata_fact:m11_metadata:blocking_prerequisite.concentration_series"
    assert cs in prereqs
    no_series = _node(g, m17.NID_BLOCKER_NO_SERIES)
    assert json.loads(no_series["detail"])["blocks_rung"] == "scaling"

    # => equivalence_class NOT narrowed
    eq = _node(g, "result:m5_regimes:mechanistic.equivalence_class")
    assert eq["value"] == list(_MECHANISM_NAMES)
    assert _node(g, "datum:m5_regimes:mechanistic.equivalence_class_narrowed"
                 )["value"] is False

    # blocked_by  DEFERRAL Service-C confusion matrix + FDR threshold (§4)
    assert m17.NID_DEFERRAL_SERVICE_C in prereqs
    deferral = _node(g, m17.NID_DEFERRAL_SERVICE_C)
    assert deferral["kind"] == "deferral"
    assert "Service-C confusion matrix" in deferral["value"]

    # unblocked_by  the two Service-C-MEASURED actions
    actions = {(_node(g, e["to"])["value"], e.get("gain_basis"))
               for e in _edges_of(g, m17.NID_REFUSAL_MECHANISM, "unblocked_by")}
    assert ("record_agitation_and_seeding", "service_c_measured") in actions
    assert ("add_concentration_series", "service_c_measured") in actions


def test_typical_series_does_NOT_invent_an_assay_blocker():
    """1,209 of 1,240 series have a PASSING assay gate. The failure mode that
    actually threatens M17 is inventing a blocker that never fired — far more
    likely than dropping one that did — so this is an explicit NEGATIVE
    assertion, not merely a check that the tree renders."""
    if not _corpus_available():
        return
    g = _graph(SERIES_TYPICAL)
    assert _node(g, m17.NID_DATUM_ASSAY_MASS)["value"] is True

    gates = _node(g, "datum:m5_regimes:mechanistic.gates_failed")["value"]
    assert "assay_not_mass_proportional" not in gates
    assert "assay_mass_proportionality_unknown" not in gates

    # the assay GATE node must not exist at all ...
    assert _node(g, "gate:m5_regimes:assay_not_mass_proportional") is None
    assert _node(g, "gate:m5_regimes:assay_mass_proportionality_unknown") is None
    # ... nor may any edge blame the non-mass-assay blocker ...
    for e in g["edges"]:
        assert e["to"] != m17.NID_BLOCKER_NON_MASS, e
    # ... and the rendered tree must never say the assay blocked anything.
    text = m17.render_tree(g, m17.NID_REFUSAL_MECHANISM)
    assert "assay_not_mass_proportional" not in text
    assert "non_mass_or_unknown_assay" not in text

    # The universal blocker IS still there, and so is the §10.9 assumption:
    # a PASSED gate is not a VERIFIED assumption.
    assert "agitation_or_seeding_unknown_branch_undetermined" in gates
    assert _node(g, m17.NID_ASSUMPTION_DYE_MASS) is not None


# --------------------------------------------------------------------------- #
# 9. adapter contracts
# --------------------------------------------------------------------------- #
def test_every_declared_adapter_key_exists_in_the_live_artifacts():
    """When an upstream module changes shape, M17 must FAIL LOUDLY here rather
    than silently degrade every explanation to explanation_available: false."""
    if not _corpus_available():
        return
    report = m17.check_adapter_contracts(_bundle(), PROC)
    assert report["all_ok"], report["failures"][:8]
    assert report["n_checked"] > 100


def test_adapter_registry_is_well_formed():
    names = [a["name"] for a in m17.ADAPTERS]
    assert len(names) == len(set(names))
    for a in m17.ADAPTERS:
        assert a["artifact"] in m17.ARTIFACTS
        assert a["scope"] in ("series", "corpus")
        assert callable(a["build"])
        assert a["declared_keys"], a["name"]


def test_contract_failure_is_detected_when_a_declared_key_disappears():
    """Negative control for the contract test itself."""
    if not _corpus_available():
        return
    original = m17.ADAPTERS[0]["declared_keys"]
    try:
        m17.ADAPTERS[0]["declared_keys"] = tuple(list(original) +
                                                 ["a.key.that.does.not.exist"])
        report = m17.check_adapter_contracts(_bundle(), PROC)
        assert not report["all_ok"]
        assert any(f["key_path"] == "a.key.that.does.not.exist"
                   for f in report["failures"])
    finally:
        m17.ADAPTERS[0]["declared_keys"] = original


def test_declared_spine_mappings_are_value_identical_to_the_m7_spine():
    """M17 nodes read the ORIGINATING artifact, not the assembled M7 spine. If
    those ever diverged, M17 would be explaining a different number from the one
    the product displays."""
    if not _corpus_available():
        return
    report = json.loads(PROC.joinpath("explanation_coverage.json")
                        .read_text(encoding="utf-8"))
    ident = report["cross_artifact_value_identity"]
    assert ident["n_checked"] > 100000
    assert ident["n_mismatched"] == 0, ident["mismatches"][:5]


# --------------------------------------------------------------------------- #
# 10. coverage is measured, and the two populations are never blended
# --------------------------------------------------------------------------- #
def test_coverage_report_matches_reality():
    if not _corpus_available():
        return
    report = json.loads(PROC.joinpath("explanation_coverage.json")
                        .read_text(encoding="utf-8"))
    t = report["totals"]
    denom = report["denominator_definition"]

    # the denominator is PUBLISHED, in full, and matches the totals
    rows = [r for r in report["stage_b_spec"]]
    assert denom["n_distinct_field_paths"] == t["field_paths_total"]
    assert t["field_paths_explained"] + len(rows) == t["field_paths_total"]

    # two populations, separately reported, never averaged
    pops = report["populations"]
    assert pops["all_series"]["n_series"] == 1654
    assert pops["explainable_series"]["n_series"] == 1240
    assert t["coverage_by_instance_all_series"] is not None
    assert t["coverage_by_instance_explainable_series"] is not None
    assert "never" in pops["never_blended"]

    # a field is 'explained' only if a node actually explained it somewhere
    for row in report["by_module"].values():
        assert row["field_paths_explained"] <= row["field_paths_total"]
        assert row["field_instances_explained"] <= row["field_instances_total"]


def test_unexplained_fields_carry_a_machine_readable_reason_and_are_never_narrated():
    if not _corpus_available():
        return
    report = json.loads(PROC.joinpath("explanation_coverage.json")
                        .read_text(encoding="utf-8"))
    unexplained = [r for r in report["stage_b_spec"]]
    assert unexplained, "a Stage-A build with nothing left to do would be a bug"
    for r in unexplained:
        assert r["reason_code"] in ("no_adapter", "not_declared")
        assert r["owner_module"]
    # sorted by impact so Stage B has a priority order
    counts = [r["n_series_emitting"] for r in unexplained]
    assert counts == sorted(counts, reverse=True)


def test_modules_without_an_adapter_report_zero_coverage_not_a_guess():
    if not _corpus_available():
        return
    report = json.loads(PROC.joinpath("explanation_coverage.json")
                        .read_text(encoding="utf-8"))
    for mod in ("m3", "m7", "m9"):
        row = report["by_module"][mod]
        assert row["has_stage_a_adapter"] is False
        assert row["field_paths_explained"] == 0
        assert row["coverage_by_distinct_path"] == 0.0


def test_series_with_no_engine_output_are_refused_not_narrated():
    """The 414 series with no M5 record must return explanation_available:
    false with a reason naming it an ENGINE gap, not an M17 gap."""
    if not _corpus_available():
        return
    b = _bundle()
    reg = b["artifacts"]["regimes.jsonl"]["data"]
    ct = b["artifacts"]["curves_triaged.jsonl"]["data"]
    missing = [sid for sid in ct if sid not in reg]
    assert missing, "expected unfittable series with no M5 record"
    sid = sorted(missing)[0]
    g = m17.build_graph(sid, b, {"concentration_series_id": None})
    assert _node(g, m17.NID_REFUSAL_MECHANISM) is None
    out = m17.explain_series(g)
    if not out["explanation_available"]:
        assert out["reason_code"] == "no_explainable_root"
        assert "ENGINE output gap" in out["reason"]


def test_default_run_does_not_write_the_90mb_dump():
    """The full per-series dump is a DERIVED view of artifacts already on disk,
    and web/server.py indexes large per-series JSONLs into memory at startup —
    so it must stay behind --dump-graphs. The sample is what ships."""
    if not _corpus_available():
        return
    report = json.loads(PROC.joinpath("explanation_coverage.json")
                        .read_text(encoding="utf-8"))
    outputs = report["outputs"]
    assert outputs["explanations.jsonl"]["flag"] == "--dump-graphs"
    assert outputs["explanations_sample.jsonl"]["always_written"] is True
    sample = PROC.joinpath("explanations_sample.jsonl")
    assert sample.exists()
    assert sample.stat().st_size < 1_000_000, sample.stat().st_size


def test_sample_selection_is_deterministic_and_covers_its_strata():
    if not _corpus_available():
        return
    b = _bundle()
    first = m17.select_sample_series(b)
    assert first == m17.select_sample_series(b)          # deterministic
    assert len(first) <= m17.SAMPLE_MAX
    for anchor in m17.SAMPLE_ANCHORS:
        assert anchor in first
    reg = b["artifacts"]["regimes.jsonl"]["data"]
    ct = b["artifacts"]["curves_triaged.jsonl"]["data"]
    gam = b["artifacts"]["gamma.jsonl"]["data"]
    # every distinct M5 refusal shape is represented
    sigs = {m17._canon(m17.resolve_key_path(reg[s], "mechanistic.gates_failed"))
            for s in first if s in reg}
    all_sigs = {m17._canon(m17.resolve_key_path(reg[s], "mechanistic.gates_failed"))
                for s in reg}
    assert sigs == all_sigs, sorted(all_sigs - sigs)
    # the gamma stratum (the only source of bootstrap_ci) is present
    assert any((ct[s] or {}).get("concentration_series_id") in gam for s in first)
    # so is the engine-output gap (no M5 record)
    assert any(s not in reg for s in first)


def test_sample_file_records_are_real_traceable_graphs():
    """The sample exists so tests and consumers can read REAL graphs without the
    90 MB dump; it must therefore survive the same guards."""
    if not _corpus_available():
        return
    resolver = m17.RawResolver(PROC)
    n_expl = 0
    with PROC.joinpath("explanations_sample.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("record_type") != "explanation":
                continue
            n_expl += 1
            assert rec["acyclic"] is True
            assert not m17.validate_vocabulary(rec)
            assert not m17.assert_no_derivation(rec, resolver)
            assert not m17.validate_source_refs(rec, resolver)
    assert n_expl >= 5


def test_honesty_block_is_present_and_complete():
    hb = m17.honesty_block()
    for key in ("derives_nothing", "confidence_never_merged",
                "deterministic_templates_only", "causal_license_is_inherited",
                "untraceable_is_refused_not_narrated",
                "counterfactuals_are_gated_on_gain_basis",
                "conformal_interval_is_refused",
                "fit_provenance_is_addressed_not_derived",
                "coverage_is_measured_not_asserted", "stage_a_edits_nothing"):
        assert key in hb and hb[key]
    if _corpus_available():
        report = json.loads(PROC.joinpath("explanation_coverage.json")
                            .read_text(encoding="utf-8"))
        assert report["honesty"] == hb


def test_fit_provenance_is_addressed_never_derived():
    """§2.3 FitProvenance is now emitted by the modules that run an optimizer.
    M17's job is to publish WHERE each record lives, never to copy or recompute
    it — and to keep saying `available: false`, with a reason, for methods that
    fit nothing (triage, classify, join). Both halves are asserted here: a
    blanket `true` would be as wrong as the blanket `false` this replaced."""
    assert m17.FIT_PROVENANCE_M2["available"] is True
    # M17 derives nothing: it may carry the ADDRESS but must not carry values.
    for banned in ("basin", "n_starts", "basin_spread", "rng_seed"):
        assert banned not in m17.FIT_PROVENANCE_M2, (
            f"M17 must not copy {banned} out of the artifact — it derives nothing")
    assert m17.FIT_PROVENANCE_M2["artifact"] == "fits.jsonl"
    assert m17.FIT_PROVENANCE_ABSENT["available"] is False
    assert m17.FIT_PROVENANCE_ABSENT["reason"]
    if not _corpus_available():
        return
    g = _graph(SERIES_TYPICAL)
    assert g["methods"], "the interned method table must exist"
    seen_true = seen_false = 0
    for mid, meth in g["methods"].items():
        fpv = meth["fit_provenance"]
        assert isinstance(fpv["available"], bool)
        if fpv["available"]:
            seen_true += 1
            assert fpv.get("artifact") and fpv.get("key_path")
        else:
            seen_false += 1
            assert fpv.get("reason")
    assert seen_true >= 1, "the M2 fitting method must report an available record"
    assert seen_false >= 1, "non-fitting methods must still report absence+reason"
    for n in g["nodes"]:
        assert n["method_ref"] in g["methods"]      # every node carries a method


def test_named_stage_a_narrowings_are_discoverable_from_the_artifact():
    """A limitation a user can only find by reading source is not disclosed."""
    if not _corpus_available():
        return
    report = json.loads(PROC.joinpath("explanation_coverage.json")
                        .read_text(encoding="utf-8"))
    ids = {n["id"] for n in report["stage_a_named_narrowings"]}
    assert "conformal_interval_not_constructed" in ids
    assert "m2_failure_reason_collapsed" in ids
    assert "m15_claims_not_joined" in ids
    assert "fit_provenance_partial" in ids
    for n in report["stage_a_named_narrowings"]:
        assert n["what"] and n["why"] and n["stage_b"]


def test_conformal_node_carries_the_marginal_quantile_and_refuses_the_interval():
    if not _corpus_available():
        return
    g = _graph(SERIES_TYPICAL)
    node = _node(g, "uncertainty:conformal:t50_marginal_quantile")
    assert node is not None
    raw = json.loads(PROC.joinpath("service_c_conformal.json")
                     .read_text(encoding="utf-8"))
    assert node["value"] == raw["calibration_quantiles"]["t50"]["marginal_quantile"]
    text = m17.render_node(node)["text"]
    assert "MARGINAL" in text
    assert "does not build" in text            # the refusal is in the rendered text


# --------------------------------------------------------------------------- #
# 10b. THE COMPOSITE (TRIPLE) JOIN — M6
#
# M6 records are keyed (uniprot_id, assay, condition_vector.construct_id). The
# failure this whole section exists to prevent is WT BLEED: resolving on a
# PARTIAL key and attaching Wild-Type's propensity to a mutant curve, which is
# the bug m7_assemble.py:548-552 was written to fix. M17 must not reintroduce it.
# --------------------------------------------------------------------------- #
# THE NEGATIVE-CONTROL FIXTURE. A real mutant series whose (uniprot, assay,
# construct) triple MISSES: the curve is P10997 'N 1 A' (turbidity), and the only
# M6 record reachable for P10997 is construct 'Wild Type'. M7 refuses to attach it
# (available: false, construct_match: false) and M17 must refuse too.
#
# IT MUST BE A `schema: full` SERIES, and the test asserts that. The previous
# fixture, CPAD-TK-1042, is `schema: compact`: its result carries NO `propensity`
# key at all, so "M17 emitted no scored propensity" was true there for a reason
# that has nothing to do with the construct gate, and the control could not
# distinguish a correct refusal from an accident. CPAD-TK-1062 emits the whole
# propensity block, so the only thing that can suppress the scored half is the
# construct mismatch this test is about.
SERIES_CONSTRUCT_MISMATCH = "CPAD-TK-1062"
_MISMATCH_TRIPLE = ("P10997", "ThT", "Wild Type")
_MISMATCH_CURVE_CONSTRUCT = "N 1 A"


def test_record_index_key_scalar_path_is_bit_for_bit_unchanged():
    """The composite key must be strictly ADDITIVE: a scalar record_key has to
    keep behaving exactly like the `rec.get(rk)` it replaced, including the
    None-means-skip contract."""
    rec = {"series_id": "S1", "uniprot_id": "P1", "assay": "ThT",
           "condition_vector": {"construct_id": "Wild Type"}}
    assert m17._record_index_key(rec, "series_id") == "S1"
    assert m17._record_index_key({"series_id": None}, "series_id") is None
    assert m17._record_index_key({}, "series_id") is None
    assert m17._record_index_key(["not", "a", "dict"], "series_id") is None


def test_record_index_key_composite_is_a_tuple_and_refuses_partial_keys():
    rec = {"uniprot_id": "P1", "assay": "ThT",
           "condition_vector": {"construct_id": "Wild Type"}}
    key = m17._record_index_key(rec, m17.PROPENSITY_RECORD_KEY)
    assert key == ("P1", "ThT", "Wild Type")
    assert isinstance(key, tuple)
    # a member that does not resolve makes the WHOLE key None — a two-thirds
    # triple is not a key, and indexing under one is how a lookup hits the
    # wrong record.
    for broken in ({"uniprot_id": "P1", "assay": "ThT"},
                   {"uniprot_id": "P1", "assay": None,
                    "condition_vector": {"construct_id": "Wild Type"}},
                   {"assay": "ThT", "condition_vector": {"construct_id": "WT"}}):
        assert m17._record_index_key(broken, m17.PROPENSITY_RECORD_KEY) is None


def test_fallback_key_must_be_a_leading_prefix_of_the_record_key():
    """The rule is LEADING PREFIX — same members, same order, from position 0 —
    and it is pinned because the docstring used to say prefix while the code
    tested subset-membership, so an out-of-order or gap-skipping declaration was
    silently accepted. The fallback index is built ONCE and shared by every
    lookup: ("assay",) against this triple would file 160 records under ~4 assay
    values and hand back an arbitrary protein's record. Only a coarsening of the
    SAME key is a legitimate fallback."""
    rk = m17.PROPENSITY_RECORD_KEY
    key = ("P1", "ThT", "Wild Type")
    assert m17._fallback_index_key(key, ("uniprot_id",), rk) == ("P1",)
    assert m17._fallback_index_key(key, ("uniprot_id", "assay"), rk) == ("P1", "ThT")
    assert m17._fallback_index_key(key, rk, rk) == key          # the whole key
    assert m17._fallback_index_key(key, (), rk) == ()           # degenerate, not an error
    # a member that is not part of the record key is a declaration error, and
    # must yield NO fallback rather than a guess
    assert m17._fallback_index_key(key, ("nope",), rk) is None
    assert m17._fallback_index_key("scalar", ("uniprot_id",), rk) is None
    # ...and so is a real member that is not LEADING: a suffix, a gap-skipping
    # subset, or a reordering. Each of these was accepted before.
    assert m17._fallback_index_key(key, ("assay",), rk) is None
    assert m17._fallback_index_key(key, ("condition_vector.construct_id",), rk) is None
    assert m17._fallback_index_key(
        key, ("uniprot_id", "condition_vector.construct_id"), rk) is None
    assert m17._fallback_index_key(key, ("assay", "uniprot_id"), rk) is None
    # a malformed index key (wrong arity for the record key) yields no fallback
    assert m17._fallback_index_key(("P1", "ThT"), ("uniprot_id",), rk) is None
    # the ONE fallback actually declared in ARTIFACTS is a leading prefix
    assert m17.PROPENSITY_RECORD_KEY_FALLBACK == m17.PROPENSITY_RECORD_KEY[:1]


def test_composite_join_does_not_bleed_across_constructs_HERMETIC():
    """THE NEGATIVE CONTROL, on a synthetic bundle so it cannot pass by luck.

    Two M6 records for ONE protein and ONE assay, differing only by construct.
    A lookup for construct B must return B's number — never A's — and a lookup
    for a construct present in NEITHER record must fall back to the FIRST
    record and say so via its record_key, not silently answer as if matched."""
    wt = {"uniprot_id": "P1", "assay": "ThT",
          "condition_vector": {"construct_id": "Wild Type"},
          "propensity": {"cohort": {"value": 11.0}}}
    mut = {"uniprot_id": "P1", "assay": "ThT",
           "condition_vector": {"construct_id": "E22K"},
           "propensity": {"cohort": {"value": 99.0}}}
    index = {}
    fallback = {}
    for rec in (wt, mut):
        k = m17._record_index_key(rec, m17.PROPENSITY_RECORD_KEY)
        index[k] = rec
        fk = m17._fallback_index_key(k, m17.PROPENSITY_RECORD_KEY_FALLBACK,
                                     m17.PROPENSITY_RECORD_KEY)
        fallback.setdefault(fk, k)
    bundle = {"artifacts": {"propensity.jsonl": {
        "kind": "jsonl", "record_key": m17.PROPENSITY_RECORD_KEY,
        "data": index, "fallback_index": fallback, "available": True}}}

    def cohort_for(construct):
        ctx = m17.Ctx("S", bundle, {"uniprot_id": "P1", "assay": "ThT",
                                    "condition_vector.construct_id": construct})
        val, ref = ctx.get("propensity.jsonl", "propensity.cohort.value")
        return val, ref["record_key"]

    assert cohort_for("E22K") == (99.0, ("P1", "ThT", "E22K"))
    assert cohort_for("Wild Type") == (11.0, ("P1", "ThT", "Wild Type"))
    # an unmatched construct falls back to the FIRST record (mirroring M7) and
    # the ref names THAT record, so the mis-join is auditable rather than hidden
    val, rk = cohort_for("A2T")
    assert val == 11.0 and rk == ("P1", "ThT", "Wild Type")
    # a MISSING member yields no record at all — never a partial-key match
    ctx = m17.Ctx("S", bundle, {"uniprot_id": "P1", "assay": None,
                                "condition_vector.construct_id": "E22K"})
    assert ctx.record_key_for("propensity.jsonl") is None
    assert ctx.record("propensity.jsonl") is None


def test_triple_join_resolves_on_the_live_corpus():
    if not _corpus_available():
        return
    b = _bundle()
    entry = b["artifacts"]["propensity.jsonl"]
    assert entry["available"] is True
    # 160 before join-policy-1.0; the M6 group key gained the ASSAY axis, splitting
    # the 4 groups that had pooled t50 across two assays into 8.
    assert len(entry["data"]) == 164, len(entry["data"])
    for key in entry["data"]:
        assert isinstance(key, tuple) and len(key) == 3, key
    # the fallback index is keyed by the uniprot PREFIX and is first-wins
    assert all(isinstance(k, tuple) and len(k) == 1 for k in entry["fallback_index"])

    ct = b["artifacts"]["curves_triaged.jsonl"]["data"]
    jk = m17.join_keys_for_series(ct[SERIES_TYPICAL])
    # join-key VALUES come from M1's triage record, never from the M7 spine
    for member in ("uniprot_id", "assay", "condition_vector.construct_id"):
        assert jk.get(member) is not None, member
    ctx = m17.Ctx(SERIES_TYPICAL, b, jk)
    key = ctx.record_key_for("propensity.jsonl")
    assert key in entry["data"], key
    assert ctx.record("propensity.jsonl") is not None


def test_no_adapter_reads_the_m7_spine():
    """The spine is the coverage DENOMINATOR. Explaining a spine value by citing
    the spine would be circular, so no adapter may name it as its artifact and
    it must not even be a registered readable artifact."""
    for a in m17.ADAPTERS:
        assert a["artifact"] != m17.SPINE_ARTIFACT, a["name"]
    assert m17.SPINE_ARTIFACT not in m17.ARTIFACTS
    for members in m17.JOIN_KEY_SOURCES.values():
        for _member, origin in members:
            assert origin and isinstance(origin, str)


def test_m6_adapters_emit_verbatim_nodes_that_byte_compare():
    if not _corpus_available():
        return
    g = _graph(SERIES_TYPICAL)
    m6 = [n for n in g["nodes"] if m17.node_adapter(n).startswith("m6_")]
    assert len(m6) > 50, len(m6)
    assert not m17.assert_no_derivation(g, m17.RawResolver(PROC))
    assert not m17.validate_source_refs(g, m17.RawResolver(PROC))
    # every M6 ref carries the COMPOSITE record key (it is never the series_id,
    # so build_graph must not have stripped it)
    for n in m6:
        rk = n["source_ref"]["record_key"]
        assert isinstance(rk, tuple) and len(rk) == 3, n["node_id"]
        assert n["source_ref"]["artifact"] == "propensity.jsonl"
    both = {m17.node_adapter(n) for n in m6}
    assert both == {"m6_propensity", "m6_sequence_axis"}, both


def test_m6_propensity_refuses_to_bleed_another_constructs_propensity():
    """THE NEGATIVE CONTROL on the LIVE corpus. CPAD-TK-1062 is a P10997 'N 1 A'
    mutant; the only M6 record reachable for P10997 is construct 'Wild Type'. M7
    refuses to attach it (m7_assemble.py:571-583) and M17 must refuse too — the
    WT-bleed bug must not re-enter through the explanation layer.

    THE CONTROL IS ONLY VALID ON A `schema: full` SERIES, so that is asserted
    FIRST. On a compact series the result has no `propensity` key whatsoever and
    "no scored propensity was emitted" would be true no matter what the construct
    gate did — a control that passes for the wrong reason is not a control."""
    if not _corpus_available():
        return
    # --- the fixture's preconditions, read straight from the spine ----------
    # (the TEST may read the spine; no ADAPTER may. This is the assertion that
    #  keeps the fixture honest if the corpus is ever regenerated.)
    spine = None
    with PROC.joinpath("protein_analysis.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if SERIES_CONSTRUCT_MISMATCH in line:
                rec = json.loads(line)
                if rec.get("series_id") == SERIES_CONSTRUCT_MISMATCH:
                    spine = rec
                    break
    assert spine is not None, SERIES_CONSTRUCT_MISMATCH
    assert spine["schema"] == "full", spine["schema"]
    assert "propensity" in spine        # the block IS emitted for this series...
    assert spine["propensity"]["construct_match"] is False   # ...and M7 refused it
    assert spine["propensity"]["prop_construct_id"] == _MISMATCH_TRIPLE[2]
    assert (spine["condition_vector"]["construct_id"]
            == _MISMATCH_CURVE_CONSTRUCT != _MISMATCH_TRIPLE[2])

    g = _graph(SERIES_CONSTRUCT_MISMATCH)
    prop = [n for n in g["nodes"] if m17.node_adapter(n) == "m6_propensity"]
    assert prop, "the join AUDIT must still be explained"

    # the scored propensity itself must be ENTIRELY absent
    for n in prop:
        for banned in ("propensity.surface", "propensity.intrinsic",
                       "propensity.cohort", "scored_feature", "feature_status",
                       "condition_match"):
            assert banned not in n["node_id"], n["node_id"]

    # only the three auditable join-outcome facts survive, and they report the
    # OTHER construct — which is exactly what M7 surfaces so the join is visible
    explained = {p for n in prop for p in n.get("explains", ())}
    assert explained == {"propensity.uniprot_id", "propensity.prop_construct_id",
                         "propensity.prop_assay"}, explained
    got = _node(g, "datum:m6_propensity:condition_vector.construct_id")
    assert got["value"] == _MISMATCH_TRIPLE[2]
    assert got["source_ref"]["record_key"] == _MISMATCH_TRIPLE
    # ...and the rendered line NAMES the record, so a reader sees the mismatch
    text = m17.render_node(got)["text"]
    assert "join audit" in text and _MISMATCH_TRIPLE[2] in text


_TWO_AUTHOR_PATHS = ("propensity.prop_assay", "propensity.prop_construct_id",
                     "propensity.uniprot_id", "sequence_axis.available",
                     "sequence_axis.reason")


def test_m6_two_author_fields_are_never_claimed_without_a_record():
    """FIVE m6-owned paths are M6's own fields when a record joined and M7's
    synthesis when none did (m7_assemble.py:554-563 and :719). A node on the
    second population would carry a FALSE source_ref.

    The narrowing used to name only the two `sequence_axis.*` paths. The three
    `propensity.*` identity fields are two-author on exactly the same series, and
    the arithmetic below is what proves the enumeration is now COMPLETE: if a
    sixth two-author path existed, or one of these five were not two-author,
    148 x 5 would not equal m6's instance shortfall."""
    if not _corpus_available():
        return
    b = _bundle()
    ct = b["artifacts"]["curves_triaged.jsonl"]["data"]
    idx = b["artifacts"]["propensity.jsonl"]["data"]
    fb = b["artifacts"]["propensity.jsonl"]["fallback_index"]
    unjoined = None
    for sid in sorted(ct):
        jk = m17.join_keys_for_series(ct[sid])
        if jk.get("uniprot_id") is None or (jk["uniprot_id"],) in fb:
            continue
        unjoined = sid
        break
    assert unjoined, "expected series with no M6 record at all"
    g = m17.build_graph(unjoined, b, m17.join_keys_for_series(ct[unjoined]))
    assert not [n for n in g["nodes"] if m17.node_adapter(n).startswith("m6_")]
    assert idx     # the artifact IS loaded; the refusal is about THIS series

    # and where a record DOES join, the paths ARE explained. Four of the five
    # show up on the TYPICAL series; `sequence_axis.reason` only exists where M6
    # itself declined the axis (`available: false`), so the corpus is walked in
    # series order until every one of the five has been seen at least once.
    explained = {p for n in _graph(SERIES_TYPICAL)["nodes"]
                 for p in n.get("explains", ())}
    assert "sequence_axis.uniprot_id" in explained
    missing = [p for p in _TWO_AUTHOR_PATHS if p not in explained]
    assert missing == ["sequence_axis.reason"], missing
    for sid in sorted(ct):
        if not missing:
            break
        g3 = m17.build_graph(sid, b, m17.join_keys_for_series(ct[sid]))
        seen = {p for n in g3["nodes"] for p in n.get("explains", ())}
        missing = [p for p in missing if p not in seen]
    assert not missing, missing

    # --- the enumeration is complete, and the arithmetic closes -------------
    # NO-JOIN = neither the triple nor uniprot_id alone hits the M6 index.
    # A two-author path is emitted by the spine on those series but explained by
    # no M6 node, so each one costs exactly one instance per no-join series.
    # A NULL uniprot is also a no-join: no M6 record carries one (verified), so
    # `prop_by_uniprot.get(None)` misses just as a real uniprot with no record
    # does. 146 of the 148 are this case — excluding them would silently shrink
    # the population to 2 and the arithmetic below would stop meaning anything.
    nojoin = []
    for sid in sorted(ct):
        up = m17.join_keys_for_series(ct[sid]).get("uniprot_id")
        if up is not None and (up,) in fb:
            continue
        nojoin.append(sid)
    assert not any((None,) == k for k in fb), "no M6 record may carry a null uniprot"
    emitting = {p: 0 for p in _TWO_AUTHOR_PATHS}
    n_full_nojoin = 0
    nojoin_set = set(nojoin)
    with PROC.joinpath("protein_analysis.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("series_id") not in nojoin_set or rec.get("schema") != "full":
                continue
            n_full_nojoin += 1
            leaves = set(m17.leaf_paths(rec))
            m6_owned = {p for p in leaves
                        if (p.startswith("propensity") or p.startswith("sequence_axis"))
                        and m17.field_owner(p) == "m6"}
            # EXACTLY these five, on every no-join full-schema series
            assert m6_owned == set(_TWO_AUTHOR_PATHS), sorted(m6_owned)
            for p in _TWO_AUTHOR_PATHS:
                emitting[p] += 1
    assert n_full_nojoin == 148, n_full_nojoin
    assert set(emitting.values()) == {148}, emitting

    report = json.loads(PROC.joinpath("explanation_coverage.json")
                        .read_text(encoding="utf-8"))
    m6 = report["by_module"]["m6"]
    shortfall = m6["field_instances_total"] - m6["field_instances_explained"]
    # 740 = 148 x 5 before join-policy-1.0. The shortfall is NO LONGER uniform:
    # the assay gate adds a THIRD author case on the two sequence_axis paths only.
    # On 5 further full-schema series (the cross-assay fallbacks) M7 publishes its
    # own refusal block, so those two paths are short on 153 series while the three
    # propensity identity paths remain short on the original 148.
    assert shortfall == 750, (m6, shortfall)          # 148 x 3 + 153 x 2
    assert n_full_nojoin * 3 + (n_full_nojoin + 5) * 2 == shortfall
    # and the narrowing PUBLISHES that decomposition rather than only the total
    named = {n["id"]: n for n in report["stage_a_named_narrowings"]}
    meas = named["m6_two_author_fields_covered_conditionally"]["measured"]
    assert meas["n_series_with_no_m6_record"] == 148
    assert meas["n_series_with_a_cross_assay_refusal"] == 5
    assert sorted(meas["two_author_paths"]) == sorted(_TWO_AUTHOR_PATHS)
    assert meas["instance_shortfall"] == 750
    assert meas["shortfall_decomposition"] == {
        "propensity.prop_assay": 148, "propensity.prop_construct_id": 148,
        "propensity.uniprot_id": 148,
        "sequence_axis.available": 153, "sequence_axis.reason": 153}


def test_m7_authored_join_outcome_fields_get_no_node_and_belong_to_m7():
    """`propensity.available|construct_match|reason` are synthesized by M7's
    join gate and exist at no key path in any artifact. No node may claim them,
    and they must be attributed to the module that can actually fix them."""
    if not _corpus_available():
        return
    authored = ("propensity.available", "propensity.construct_match",
                "propensity.reason")
    for path in authored:
        assert m17.field_owner(path) == "m7", path
    assert m17.field_owner("propensity.surface.value") == "m6"
    assert m17.field_owner("sequence_axis.available") == "m6"
    for sid in (SERIES_ALL_BRANCHES, SERIES_TYPICAL, SERIES_CONSTRUCT_MISMATCH):
        for n in _graph(sid)["nodes"]:
            for p in n.get("explains", ()):
                assert p not in authored, (sid, n["node_id"], p)


def test_m6_coverage_is_measured_and_the_denominator_change_is_disclosed():
    if not _corpus_available():
        return
    report = json.loads(PROC.joinpath("explanation_coverage.json")
                        .read_text(encoding="utf-8"))
    m6 = report["by_module"]["m6"]
    assert m6["has_stage_a_adapter"] is True
    assert m6["field_paths_explained"] == 116, m6
    assert m6["field_paths_total"] == 116, m6
    # the re-attributed paths landed in m7, so nothing was DROPPED from the
    # published denominator — it was regrouped, and both halves still sum.
    #
    # join-policy-1.0 EMITS THREE NEW FIELDS (propensity.assay_match,
    # sequence_axis.assay_match, sequence_axis.prop_assay), so the published
    # denominator GREW 329 -> 332 rather than being rebased. All three are M7's
    # gate output, so m7 goes 10 -> 13 and m6 stays 116/116.
    assert report["by_module"]["m7"]["field_paths_total"] == 13
    assert report["totals"]["field_paths_total"] == 332
    # a rebase that is not disclosed is a rebase; this one is named in the data
    ids = {n["id"] for n in report["stage_a_named_narrowings"]}
    assert "m6_propensity_join_outcome_fields_are_m7s" in ids
    assert "m6_assay_bleed_fixed_m17_mirrors_the_corrected_join" in ids
    assert "m6_two_author_fields_covered_conditionally" in ids
    assert "m6_surface_gamma_is_not_edged_to_m4" in ids
    assert "explanations_no_longer_reach_fields_the_compact_schema_never_emits" in ids


def test_every_published_integer_carries_the_definition_that_produced_it():
    """The lesson of the Stage-B audit: four narrowings published integers that
    did not reproduce, because no narrowing said what it had counted. A
    `measured` block is now the ONLY place a narrowing may put a number, and it
    must state its population and its predicate."""
    if not _corpus_available():
        return
    report = json.loads(PROC.joinpath("explanation_coverage.json")
                        .read_text(encoding="utf-8"))
    saw = 0
    for n in report["stage_a_named_narrowings"]:
        meas = n.get("measured")
        if meas is None:
            continue
        saw += 1
        assert meas.get("population"), n["id"]
        assert meas.get("definition"), n["id"]
        ints = [k for k, v in meas.items() if isinstance(v, int)]
        assert ints, ("a `measured` block with no integer is just prose", n["id"])
    assert saw >= 4, saw


def _m7_propensity_indexes():
    """M7's two propensity indexes, rebuilt VERBATIM (m7_assemble.py:1002-1009),
    so these tests measure the engine's join and not M17's mirror of it."""
    by_triple, by_uniprot = {}, {}
    with PROC.joinpath("propensity.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            up = p.get("uniprot_id")
            cv = p.get("condition_vector", {}) or {}
            by_triple[(up, p.get("assay"), cv.get("construct_id"))] = p
            by_uniprot.setdefault(up, p)
    return by_triple, by_uniprot


def _spine_records():
    with PROC.joinpath("protein_analysis.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


class _SpineIndex(dict):
    """series_id -> spine record, loaded once on first use. A per-series rescan of
    the 1,654-record spine inside a 382-iteration loop is O(n*m) and turned one
    test into a multi-minute run."""

    _loaded = False

    def get(self, key, default=None):
        if not self._loaded:
            for r in _spine_records():
                dict.__setitem__(self, r["series_id"], r)
            self._loaded = True
        return dict.get(self, key, default)

    def __getitem__(self, key):
        got = self.get(key)
        if got is None:
            raise KeyError(key)
        return got


_SPINE_BY_ID = _SpineIndex()


def test_no_construct_matched_cross_assay_join_survives_anywhere():
    """REGRESSION GUARD for the assay bleed, fixed by join-policy-1.0.

    WITHDRAWN HISTORICAL FIGURES, retained so the defect stays legible: before the
    fix this test pinned the nested quadruple 96 / 14 / 9 / 7 — 96 full-schema
    series resolved via the uniprot-only fallback, 14 of those to a record computed
    for a DIFFERENT assay, 9 of those published as `available: true,
    construct_match: true`, and 7 carrying a cross-assay `sequence_axis`. (An
    earlier revision of the narrowing read the 96 as the different-assay count and
    so overstated the defect ~7x while contradicting its own follow-on numbers.)

    Those integers are no longer reproducible and must not be: `_propensity_block`
    now refuses a cross-assay record and the M6 grouping key carries the assay, so
    8 of the 9 are correctly joined instead. What is pinned now is the INVARIANT,
    not a count — the counts were the symptom. Deleting the test would erase the
    only executable memory of the defect, so it is converted rather than removed."""
    if not _corpus_available():
        return
    by_triple, by_uniprot = _m7_propensity_indexes()
    # RAW, not the pruned bundle index: this recomputation must not share the
    # load/prune path with the code that produced the published integers.
    ct = {}
    with PROC.joinpath("curves_triaged.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                ct[r["series_id"]] = r
    fallback, diff_assay, clean, seq_bleed = [], [], [], []
    for rec in _spine_records():
        sid = rec["series_id"]
        if rec.get("schema") != "full":
            continue
        t = ct.get(sid) or {}
        cv = t.get("condition_vector", {}) or {}
        up = t.get("uniprot_id")
        if (up, cv.get("assay_type"), cv.get("construct_id")) in by_triple:
            continue
        prop = by_uniprot.get(up)
        if prop is None:
            continue
        fallback.append(sid)
        if prop.get("assay") == cv.get("assay_type"):
            continue
        diff_assay.append(sid)
        prop = rec.get("propensity") or {}
        if prop.get("construct_match") is True and prop.get("available") is True:
            clean.append(sid)
        sa = rec.get("sequence_axis") or {}
        if sa.get("available") is True and sa.get("assay") != cv.get("assay_type"):
            seq_bleed.append(sid)

    # THE INVARIANT. Not a count: no curve may carry a propensity computed for
    # another assay, and none may carry a sequence axis licensed against another
    # endpoint. Both were non-empty before join-policy-1.0 (9 and 7).
    assert clean == [], clean
    assert seq_bleed == [], seq_bleed

    # A cross-assay FALLBACK may still be REACHED — what must never happen is that
    # it is published. These are IAPP turbidity curves with no usable t50, so no
    # turbidity M6 record can exist for them; the gate refuses them by design and
    # they are the residual case the narrowing names.
    assert set(diff_assay) < set(fallback)
    for sid in diff_assay:
        prop = (_SPINE_BY_ID.get(sid) or {}).get("propensity") or {}
        assert prop.get("available") is False, (sid, prop)
        assert prop.get("assay_match") is False, (sid, prop)

    # and every refusal must be tallied in the rollup, in its own class
    rollup = json.loads(PROC.joinpath("m7_rollup.json").read_text(encoding="utf-8"))
    summary = rollup["propensity_join_summary"]
    assert summary["n_assay_mismatch_suppressed"] == len(diff_assay), summary


def _scalar(record, block, field):
    """`record[block][field]`, with an ABSENT key and a stored null both reduced
    to None — which is what "null == null" in the published definition means.

    DELIBERATELY PLAIN. This helper exists so the recomputation below shares NO
    code with the measurement that produced the published integers: no
    `resolve_key_path`, no `_canon_safe`. That is not fastidiousness — it is the
    bug this test now guards. The withdrawn `267 of 303` came from
    `_canon_safe(resolve_key_path(...))`, where an absent key becomes the
    `_MISSING` sentinel whose `_canon_safe` is Python None while a stored null
    canonicalises to the STRING "null", so absent-vs-null compared UNEQUAL. A
    guard that recomputes through the same helpers reproduces the same wrong
    number and certifies it."""
    b = record.get(block)
    if not isinstance(b, dict):
        return None
    return b.get(field)


def test_the_refused_gamma_edge_is_justified_by_a_reproducible_measurement():
    """The `m6_surface_gamma_is_not_edged_to_m4` integers, recomputed from the
    artifacts under the published definition, by an implementation independent of
    the one that produced them.

    Two figures have been withdrawn from this narrowing (192/337, then 267/36);
    the CONCLUSION (refuse the edge) survives every reading, which is why it is
    still refused. The FULL 2x2 is asserted, not just the totals: a decomposition
    that sums to the population is what makes a miscount visible instead of
    plausible."""
    if not _corpus_available():
        return
    b = _bundle()
    ct = b["artifacts"]["curves_triaged.jsonl"]["data"]
    gamma = {}
    with PROC.joinpath("gamma.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                g = json.loads(line)
                gamma[g["concentration_series_id"]] = g
    cells = {"agree_both_null": 0, "agree_both_value": 0, "differ_both_value": 0,
             "differ_m6_null_only": 0, "differ_m4_null_only": 0}
    neighbours = {"gamma_regression.ci": 0, "gamma_regression.reliable": 0,
                  "gamma_global.value": 0, "gamma_global.ci": 0,
                  "gamma_disagreement": 0}
    pairs = (("gamma_regression.ci", "gamma_regression", "ci",
              "gamma_regression", "gamma_ci"),
             ("gamma_regression.reliable", "gamma_regression", "reliable",
              "gamma_regression", "gamma_reliable"),
             ("gamma_global.value", "gamma_global", "value",
              "gamma_global", "gamma"),
             ("gamma_global.ci", "gamma_global", "ci",
              "gamma_global", "gamma_ci"))
    both = 0
    for rec in _spine_records():
        surf = ((rec.get("propensity") or {}).get("surface") or {})
        if surf.get("available") is not True:
            continue
        cs = (ct.get(rec["series_id"]) or {}).get("concentration_series_id")
        g = gamma.get(cs) if cs else None
        if g is None:
            continue
        both += 1
        m6 = _scalar(surf, "gamma_regression", "value")
        m4 = _scalar(g, "gamma_regression", "gamma")
        same = json.dumps(m6, sort_keys=True) == json.dumps(m4, sort_keys=True)
        if m6 is None and m4 is None:
            cells["agree_both_null"] += 1                 # equal by construction
        elif same:
            cells["agree_both_value"] += 1
        elif m6 is None:
            cells["differ_m6_null_only"] += 1
        elif m4 is None:
            cells["differ_m4_null_only"] += 1
        else:
            cells["differ_both_value"] += 1
        for name, b6, f6, b4, f4 in pairs:
            if (json.dumps(_scalar(surf, b6, f6), sort_keys=True)
                    != json.dumps(_scalar(g, b4, f4), sort_keys=True)):
                neighbours[name] += 1
        if (json.dumps(surf.get("gamma_disagreement"), sort_keys=True)
                != json.dumps(_scalar(g, "disagreement", "disagree"),
                              sort_keys=True)):
            neighbours["gamma_disagreement"] += 1

    # 303 / 99 / 26 / 42 / 178 before join-policy-1.0. The re-key plus the
    # assay-matched γ join moved four groups off a foreign-assay γ (or off none)
    # onto their own, which shifts differ_m6_null_only 42 -> 38 and
    # agree_both_value 26 -> 30. differ_both_value is UNCHANGED at 133 and the
    # informative sub-population is UNCHANGED at 204 — the conclusion does not move,
    # only the population it is measured over.
    assert both == 304, both
    assert sum(cells.values()) == both, cells          # the 2x2 must close
    differs = (cells["differ_both_value"] + cells["differ_m6_null_only"]
               + cells["differ_m4_null_only"])
    informative = both - cells["agree_both_null"]
    assert cells == {"agree_both_null": 100, "agree_both_value": 30,
                     "differ_both_value": 133, "differ_m6_null_only": 38,
                     "differ_m4_null_only": 3}, cells
    assert differs == 174 and informative == 204

    # THE CONTRADICTION THAT CAUGHT THE WITHDRAWN FIGURE, pinned as a rule: the
    # definition says absent == null, so every both-null pair MUST be an
    # agreement. Any published n_agree below that floor is impossible on its face.
    report = json.loads(PROC.joinpath("explanation_coverage.json")
                        .read_text(encoding="utf-8"))
    named = {n["id"]: n for n in report["stage_a_named_narrowings"]}
    meas = named["m6_surface_gamma_is_not_edged_to_m4"]["measured"]
    assert meas["n_agree"] >= cells["agree_both_null"], meas
    assert meas["n_carrying_both"] == both
    assert meas["n_both_null_agree_trivially"] == cells["agree_both_null"]
    assert meas["n_informative"] == informative
    assert meas["n_gamma_point_estimate_differs"] == differs
    assert meas["n_agree"] == both - differs
    assert meas["n_agree_on_a_real_value"] == cells["agree_both_value"]
    assert meas["decomposition_of_the_304"] == cells
    assert meas["neighbouring_scalars_differ_of_304"] == neighbours, neighbours
    # the tell that exposed the defect: .ci coinciding exactly with .value was an
    # artifact of the absent-vs-null miscompare, and must not silently return
    assert neighbours["gamma_regression.ci"] != differs

    # and NO such edge exists in a graph that has both halves
    for sid in (SERIES_ALL_BRANCHES, SERIES_TYPICAL):
        for e in _graph(sid)["edges"]:
            assert not ("m6_" in e["from"] and "m4_gamma" in e["to"]), e


def test_compact_schema_series_receive_nodes_for_fields_they_never_emit():
    """A KNOWN, PUBLISHED DEFECT, pinned so the published numbers stay true.

    An adapter fires on whether ITS artifact joined, never on whether the spine
    emits the block, so a compact-schema result — which has no `propensity` and
    no `sequence_axis` key at all — still receives M6 nodes. This test measures
    the defect rather than asserting it away, and it will FAIL when the M17-wide
    gate lands, which is the point: the narrowing's numbers must move with it.

    ORPHAN NODE = a node declaring >=1 `explains` path, none of which resolves in
    that series' own spine record."""
    if not _corpus_available():
        return
    b = _bundle()
    ct = b["artifacts"]["curves_triaged.jsonl"]["data"]
    by_adapter = {}
    m6_series, scored_series, m6_nodes = set(), set(), 0
    m6_nodes_structural = 0
    joined = {"full": 0, "compact": 0}
    for rec in _spine_records():
        sid = rec["series_id"]
        jk = m17.join_keys_for_series(ct.get(sid) or {})
        if m17.Ctx(sid, b, jk).record("propensity.jsonl") is not None:
            joined[rec.get("schema")] = joined.get(rec.get("schema"), 0) + 1
        if rec.get("schema") != "compact":
            continue
        assert "propensity" not in rec and "sequence_axis" not in rec, sid
        for n in m17.build_graph(sid, b, jk)["nodes"]:
            exp = [p for p in (n.get("explains") or ()) if p]
            # SECOND, INDEPENDENT COUNT of the M6 total that needs no
            # resolve_key_path: a compact record has neither top-level key
            # (asserted above), so EVERY M6 node with an `explains` path is an
            # orphan by inspection. If the two counts ever diverge, the
            # path-resolution predicate is what is wrong, not the graph.
            if exp and m17.node_adapter(n).startswith("m6_"):
                assert all(p.split(".")[0] in ("propensity", "sequence_axis")
                           for p in exp), n["node_id"]
                m6_nodes_structural += 1
            if not exp or any(m17.resolve_key_path(rec, p) is not m17._MISSING
                              for p in exp):
                continue
            ad = m17.node_adapter(n)
            by_adapter[ad] = by_adapter.get(ad, 0) + 1
            if ad.startswith("m6_"):
                m6_series.add(sid)
                m6_nodes += 1
                if any(k in n["node_id"] for k in ("propensity.surface",
                                                   "propensity.intrinsic",
                                                   "propensity.cohort")):
                    scored_series.add(sid)
    # THE INVARIANT: with the gate on — i.e. exactly what the corpus run writes —
    # NO node may explain a path its own spine record does not emit. The loop above
    # builds UNGATED graphs so it can measure what the gate REMOVES; re-running all
    # 382 gated as well doubled this suite's runtime past ten minutes. Instead the
    # real gate is exercised on a deterministic SAMPLE spanning every contributing
    # adapter, and the full population is covered by the accounting above (an orphan
    # that survived would show up there, because that count is what the narrowing
    # publishes and it is cross-checked against the artifact).
    sample = sorted(m6_series)[::40] or sorted(m6_series)
    sampled_adapters = set()
    for sid in sample:
        rec = _SPINE_BY_ID[sid]
        jk = m17.join_keys_for_series(ct.get(sid) or {})
        gated = m17.build_graph(sid, b, jk, spine_record=rec)
        for n in gated["nodes"]:
            exp = [p for p in (n.get("explains") or ()) if p]
            assert not exp or any(
                m17.resolve_key_path(rec, p) is not m17._MISSING for p in exp), (
                sid, n["node_id"], exp)
        # and the gate must RECORD what it removed, never drop it silently
        assert gated["spine_gated_nodes"], sid
        for g in gated["spine_gated_nodes"]:
            sampled_adapters.add(g["adapter"])
            assert g["reason"] == "spine_does_not_emit_these_paths", g
    assert {"m6_propensity", "m6_sequence_axis"} <= sampled_adapters, sampled_adapters

    assert joined == {"full": 1046, "compact": 382}, joined
    assert len(m6_series) == 382, len(m6_series)
    # 26,547 / 321 before join-policy-1.0. The assay gate suppressed cross-assay
    # sequence-axis blocks and the M6 re-key created 4 more records that more
    # compact series join, so m6_sequence_axis fell 17,882 -> 16,959 while
    # m6_propensity rose 8,665 -> 8,701. The four non-M6 adapter rows are unchanged,
    # which is what distinguishes a caused movement from drift.
    assert m6_nodes == 25660, m6_nodes
    assert m6_nodes_structural == m6_nodes, (m6_nodes_structural, m6_nodes)
    assert len(scored_series) == 323, len(scored_series)

    report = json.loads(PROC.joinpath("explanation_coverage.json")
                        .read_text(encoding="utf-8"))
    named = {n["id"]: n for n in report["stage_a_named_narrowings"]}
    meas = named["explanations_no_longer_reach_fields_the_compact_schema_never_emits"]["measured"]
    assert meas["m6_nodes_gated"] == m6_nodes
    assert meas["orphan_nodes_remaining"] == 0
    assert meas["m6_compact_series_affected"] == len(m6_series)
    assert meas["m6_compact_series_with_a_scored_block"] == len(scored_series)
    assert meas["n_series_with_a_joined_m6_record"] == {
        "full": 1046, "compact": 382, "total": 1428}
    # the PRE-EXISTING contributors are published too, so the pattern is not
    # mis-read as an M6 defect -- and the per-adapter table must be exact
    assert meas["nodes_gated_by_adapter"] == by_adapter, (
        meas["nodes_gated_by_adapter"], by_adapter)
    assert meas["nodes_gated_total"] == sum(by_adapter.values())
    assert set(by_adapter) - {"m6_propensity", "m6_sequence_axis"}, (
        "the class predates the M6 adapters; other adapters must appear")


def test_coverage_is_untouched_by_the_orphan_nodes():
    """The defect above is in the GRAPH, not in the numbers, and that separation
    is the reason it can be published rather than hot-fixed: `observe` iterates
    the spine record's OWN leaves, so a node explaining a path the record does
    not contain is counted in no numerator and no denominator."""
    if not _corpus_available():
        return
    acc = m17.CoverageAccumulator()
    spine_rec = {"series_id": "S", "schema": "compact", "a": {"b": 1}}
    graph = {"series_id": "S", "nodes": [
        {"node_id": "result:m6_propensity:propensity.cohort.value",
         "kind": "result", "value": 7, "explains": ["propensity.cohort.value"]},
        {"node_id": "datum:m1_triage:a.b", "kind": "datum", "value": 1,
         "explains": ["a.b"]}], "edges": []}
    out = acc.observe(spine_rec, graph, True)
    assert out["n_spine_fields_emitted"] == 3          # series_id, schema, a.b
    assert out["n_spine_fields_explained"] == 1        # only a.b
    assert acc.inst_all == out["n_spine_fields_emitted"]
    assert "propensity.cohort.value" not in acc.emitting_all
    assert "propensity.cohort.value" not in acc.explained_all
    assert not acc.value_identity_mismatch


def test_composite_source_refs_survive_the_json_round_trip():
    """A composite record_key serialises as a LIST and the index is keyed by
    TUPLES. If the resolver did not restore the type, every M6 ref in a written
    artifact would silently dangle."""
    if not _corpus_available():
        return
    resolver = m17.RawResolver(PROC)
    g = _graph(SERIES_TYPICAL)
    reread = json.loads(m17._canon(g))
    m6 = [n for n in reread["nodes"] if m17.node_adapter(n).startswith("m6_")]
    assert m6
    for n in m6:
        assert isinstance(n["source_ref"]["record_key"], list)   # JSON shape
    assert not m17.assert_no_derivation(reread, resolver)
    assert not m17.validate_source_refs(reread, resolver)


def test_m6_templates_exist_so_no_node_is_narrated_without_one():
    if not _corpus_available():
        return
    for sid in (SERIES_TYPICAL, SERIES_CONSTRUCT_MISMATCH):
        for n in _graph(sid)["nodes"]:
            if not m17.node_adapter(n).startswith("m6_"):
                continue
            out = m17.render_node(n)
            assert out["explanation_available"] is True, n["node_id"]
            assert "<unavailable>" not in out["text"], n["node_id"]


# --------------------------------------------------------------------------- #
# 11. key-path grammar + robustness
# --------------------------------------------------------------------------- #
def test_key_path_grammar():
    doc = {"a": {"b": [{"field": "x", "v": 1}, {"field": "y", "v": 2}]},
           "c": [10, 20, 30]}
    assert m17.resolve_key_path(doc, "a.b[field=y].v") == 2
    assert m17.resolve_key_path(doc, "a.b[0].field") == "x"
    assert m17.resolve_key_path(doc, "c[2]") == 30
    assert m17.resolve_key_path(doc, "a.b[field=zz].v") is m17._MISSING
    assert m17.resolve_key_path(doc, "c[9]") is m17._MISSING
    assert m17.resolve_key_path(doc, "nope") is m17._MISSING
    # a stored null is DISTINCT from an unresolvable path
    assert m17.resolve_key_path({"k": None}, "k") is None


def test_prune_roundtrip_preserves_selector_and_index_paths():
    """The bug this pins: creating a dict where a list was needed silently
    pruned selector-addressed facts out of the bundle, so their nodes never
    appeared and the graph quietly thinned."""
    rec = {"recs": [{"field": "agitation", "action": "record_it", "gain": 4.5},
                    {"field": "seeding", "action": "record_it2", "gain": 1.5}],
           "deps": {"MECHANISM": {"blocking": ["a", "b", "c"]}}}
    paths = ["recs[field=agitation].action", "recs[field=seeding].gain",
             "deps.MECHANISM.blocking[2]"]
    pruned = m17._prune(rec, paths)
    for p in paths:
        assert m17.resolve_key_path(pruned, p) == m17.resolve_key_path(rec, p), p


def test_leaf_paths_definition_matches_the_published_denominator():
    obj = {"a": 1, "b": {"c": None, "d": {}}, "e": [1, 2, 3], "f": {"g": {"h": 2}}}
    assert sorted(m17.leaf_paths(obj)) == ["a", "b.c", "b.d", "e", "f.g.h"]


def test_canonical_encoding_is_stable_and_rejects_nan():
    assert m17._canon({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert m17._canon_safe(float("nan")) is None
    assert m17._canon_safe(float("inf")) is None


def test_nothing_raises_on_empty_or_malformed_input():
    empty = {"artifacts": {}, "missing_artifacts": [], "engine_build_id": None}
    g = m17.build_graph("nope", empty, {})
    assert g["nodes"] == [] and g["edges"] == []
    assert g["acyclic"] is True
    assert m17.explain_series(g)["explanation_available"] is False
    assert m17.render_tree(g, "missing") == ""
    assert m17.project(g, "not_a_projection")["available"] is False
    assert m17.parse_key_path("a[unclosed") == ()
    assert m17.resolve_key_path(None, "a.b") is m17._MISSING
    assert m17.render_node({"template": "no_such_template"})[
        "explanation_available"] is False


def test_unknown_template_refuses_rather_than_improvising():
    out = m17.render_node({"node_id": "x:y:z", "template": "does.not.exist"})
    assert out["explanation_available"] is False
    assert out["reason_code"] == "no_template_for_node"


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
    print(f"test_m17_explain: {passed} passed")
    return passed


if __name__ == "__main__":
    _run_all()
    sys.exit(0)
