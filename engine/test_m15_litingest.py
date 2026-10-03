"""
Tests for M15 — Literature Ingestion & Claim Extraction (§3-M15).

Deterministic + fast. Operates on SYNTHETIC fixture Documents (does NOT read the
real data files except the CPAD round-trip, which is hermetic on a synthetic
mini-corpus). Pins the M15 contract:

  - exact extraction of concentration (unit-normalised to µM), pH, temperature,
    mutation, assay/dye, agitation, seeding from a Methods block + a table + a
    caption;
  - every claim's PROVENANCE resolves to the correct block / location_type /
    char_span + VERBATIM source_text;
  - APPEND-ONLY: a correction appends a new claim with supersedes set and the
    ORIGINAL claim is byte-identical afterwards (never mutated);
  - confidence: an implausible value (pH 74) -> low confidence + flag; table >
    prose ordering;
  - CONFLICT fires when two fixture papers report divergent gamma/t50 for the
    same protein+condition (both provenances retained);
  - CPAD round-trip: a fixture mirroring a real pmid -> high precision/recall; a
    deliberately-wrong field is caught;
  - FRONT-END SEAM (pdf_to_document / ocr_page / ner_extract /
    digitize_figure_curve): BOTH branches are covered without installing any
    dependency — dependency absent -> NotImplementedError naming the specific
    missing dep; implementation injected -> the front-end delegates and returns
    its result; honesty_block() reports the TRUE detected state either way; an
    injected NER hit is stamped at the lower 'ml_ner' tier, never promoted;
  - never raises on empty/malformed input.

    python engine/test_m15_litingest.py
    pytest engine/test_m15_litingest.py
"""
from __future__ import annotations

import contextlib
import json

import m15_litingest as m15


# --------------------------------------------------------------------------- #
# fixture builders
# --------------------------------------------------------------------------- #
def _methods_doc():
    """A single-page doc with a Methods paragraph + a conditions table + a caption."""
    spec = {
        "doi": "10.0000/test", "pmid": "99999999",
        "pages": [{"page_no": 1, "blocks": [
            {"block_id": "m-h", "type": "heading", "text": "Materials and Methods"},
            {"block_id": "m-1", "type": "paragraph",
             "text": ("Alpha-synuclein (UniProt P37840) carrying the A30P mutation "
                      "was prepared at 50 µM in 20 mM Tris buffer at pH 7.4 and "
                      "monitored by Thioflavin T fluorescence at 37 C under quiescent "
                      "conditions in unseeded reactions with 100 mM NaCl.")},
            {"block_id": "m-tab", "type": "table", "section": "methods",
             "header": ["Protein", "Conc (µM)", "Temp", "pH", "Assay"],
             "rows": [["alpha-synuclein", "50", "37 C", "7.4", "ThT"]]},
            {"block_id": "m-fig", "type": "figure_caption",
             "text": "Fig 2. ThT kinetics of alpha-synuclein at 10-70 µM, pH 7.4, 37°C, shaken."},
        ]}],
    }
    return m15.document_from_fixture(spec)


def _claims_for(field, claims):
    return [c for c in claims if c["field"] == field]


# --------------------------------------------------------------------------- #
# front-end seam helpers — both branches are exercised with NO real dependency
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _all_frontends_absent():
    """Force every front-end to resolve as UNAVAILABLE, hermetically: drop any
    injected implementation and pin the detection cache to 'nothing found'.

    WHY pin: the test must assert the dependency-absent branch on ANY host,
    including one where pdfminer/spaCy happen to be installed. State is fully
    restored on exit."""
    saved_impls = dict(m15._FRONTEND_IMPLS)
    saved_cache = dict(m15._BACKEND_CACHE)
    m15._FRONTEND_IMPLS.clear()
    m15._BACKEND_CACHE.update(
        {n: {"module": None, "detail": "pinned absent by test"}
         for n in m15.FRONTEND_NAMES})
    try:
        yield
    finally:
        m15._FRONTEND_IMPLS.clear()
        m15._FRONTEND_IMPLS.update(saved_impls)
        m15._BACKEND_CACHE.clear()
        m15._BACKEND_CACHE.update(saved_cache)


@contextlib.contextmanager
def _frontend(name, fn):
    """Inject `fn` as the implementation of one front-end, then restore exactly
    the previous implementation AND detection-cache state."""
    saved_cache = dict(m15._BACKEND_CACHE)
    prev = m15.register_frontend(name, fn)
    try:
        yield fn
    finally:
        m15.register_frontend(name, prev)
        m15._BACKEND_CACHE.clear()
        m15._BACKEND_CACHE.update(saved_cache)


# one representative argument per front-end (never actually parsed/OCR'd here)
_FRONTEND_ARGS = {"pdf_to_document": "paper.pdf", "ocr_page": "<page-image>",
                  "ner_extract": "Alpha-synuclein was prepared.",
                  "digitize_figure_curve": "<figure>"}


# --------------------------------------------------------------------------- #
# 1. exact extraction from Methods prose
# --------------------------------------------------------------------------- #
def test_extract_concentration_normalised_to_um():
    doc = _methods_doc()
    claims = m15.extract_document(doc).claims
    concs = _claims_for("concentration", claims)
    # 50 µM in prose (methods), 50 in table, 10-70 µM range in caption
    vals = {c["value"] for c in concs}
    assert 50.0 in vals
    # unit normalisation: nM would divide by 1000; here µM stays µM
    for c in concs:
        assert c["unit"] == "µM"


def test_concentration_unit_normalisation_nM_to_uM():
    spec = {"doi": "d", "pmid": "1", "pages": [{"page_no": 1, "blocks": [
        {"block_id": "b", "type": "paragraph", "section": "methods",
         "text": "Protein at 500 nM and also 2 mM were tested."}]}]}
    claims = m15.extract_document(m15.document_from_fixture(spec)).claims
    vals = sorted(c["value"] for c in _claims_for("concentration", claims))
    assert 0.5 in vals          # 500 nM -> 0.5 µM
    assert 2000.0 in vals       # 2 mM -> 2000 µM


def test_extract_pH_temperature_mutation():
    doc = _methods_doc()
    claims = m15.extract_document(doc).claims
    assert any(c["value"] == 7.4 for c in _claims_for("pH", claims))
    assert any(c["value"] == 37.0 for c in _claims_for("temperature", claims))
    assert any(c["value"] == "A30P" for c in _claims_for("mutation", claims))


def test_extract_assay_dye_agitation_seeding():
    doc = _methods_doc()
    claims = m15.extract_document(doc).claims
    assert any(c["value"] == "ThT" for c in _claims_for("assay", claims))
    assert any(c["value"] == "ThT" for c in _claims_for("dye", claims))
    assert any(c["value"] == "quiescent" for c in _claims_for("agitation", claims))
    assert any(c["value"] == "unseeded" for c in _claims_for("seeding", claims))


def test_extract_uniprot_and_protein_gazetteer():
    doc = _methods_doc()
    claims = m15.extract_document(doc).claims
    assert any(c["value"] == "P37840" for c in _claims_for("uniprot", claims))
    assert any(c["value"] == "Alpha-Synuclein" for c in _claims_for("protein", claims))


# --------------------------------------------------------------------------- #
# 2. provenance resolves correctly
# --------------------------------------------------------------------------- #
def test_provenance_resolves_verbatim_and_span():
    doc = _methods_doc()
    claims = m15.extract_document(doc).claims
    # the prose 50 µM claim
    prose = [c for c in _claims_for("concentration", claims)
             if c["location_type"] == "methods_prose" and c["value"] == 50.0]
    assert prose, "no methods-prose concentration claim"
    c = prose[0]
    assert c["doi"] == "10.0000/test" and c["pmid"] == "99999999"
    assert c["page"] == 1 and c["block_id"] == "m-1"
    start, end = c["char_span"]
    block_text = doc["pages"][0]["blocks"][1]["text"]
    # VERBATIM: source_text is exactly the char-span slice of the block text
    assert block_text[start:end] == c["source_text"]
    assert "50" in c["source_text"] and "µM" in c["source_text"]


def test_table_provenance_row_col():
    doc = _methods_doc()
    claims = m15.extract_document(doc).claims
    tbl = [c for c in claims if c["location_type"] == "table"]
    assert tbl
    for c in tbl:
        assert "table_cell" in c
        assert "row" in c["table_cell"] and "col" in c["table_cell"]
        assert c["extractor"]["method"] == "table_cell"


def test_caption_location_type():
    doc = _methods_doc()
    claims = m15.extract_document(doc).claims
    cap = [c for c in claims if c["block_id"] == "m-fig"]
    assert cap
    assert all(c["location_type"] == "figure_caption" for c in cap)
    assert any(c["field"] == "concentration" for c in cap)


def test_section_detection():
    doc = _methods_doc()
    sections = m15.detect_sections(doc)
    assert sections["m-1"] == "methods"
    assert sections["m-fig"] == "methods"   # inherits running section


# --------------------------------------------------------------------------- #
# 3. APPEND-ONLY ledger
# --------------------------------------------------------------------------- #
def test_append_only_supersede_never_mutates_original():
    doc = _methods_doc()
    ledger = m15.extract_document(doc)
    original = _claims_for("pH", ledger.claims)[0]
    original_id = original["claim_id"]
    original_bytes = json.dumps(original, sort_keys=True)

    # A correction: same field, corrected value, via supersede-by-append.
    corrected = dict(original)
    corrected["value"] = 7.35
    new_claim = ledger.supersede(original_id, corrected)

    # the NEW claim carries supersedes; it is a distinct entry
    assert new_claim["supersedes"] == original_id
    assert new_claim["claim_id"] != original_id
    # the ORIGINAL is byte-identical and still present (never mutated/deleted)
    still = ledger.get(original_id)
    assert json.dumps(still, sort_keys=True) == original_bytes
    # ledger grew by exactly one
    assert len(ledger.claims) == len({c["claim_id"] for c in ledger.claims})
    # active view drops the superseded original, keeps the correction
    active_ids = {c["claim_id"] for c in ledger.active_claims()}
    assert original_id not in active_ids
    assert new_claim["claim_id"] in active_ids


def test_append_is_idempotent():
    ledger = m15.ClaimLedger()
    c = m15.make_claim(field="pH", value=7.4, unit=None, doi="d", pmid="1", page=1,
                       location_type="methods_prose", block_id="b", char_span=[0, 3],
                       source_text="pH 7.4", stage="s", method="unit_grammar",
                       confidence=0.9, group="conditions",
                       plausibility={"plausible": True, "reason": "ok"})
    ledger.append_claim(c)
    ledger.append_claim(dict(c))       # same id -> no-op append
    assert len(ledger.claims) == 1


def test_claim_id_deterministic():
    kw = dict(field="pH", value=7.4, unit=None, doi="d", pmid="1", page=1,
              location_type="methods_prose", block_id="b", char_span=[0, 3],
              source_text="pH 7.4", stage="s", method="unit_grammar",
              confidence=0.9, group="conditions",
              plausibility={"plausible": True, "reason": "ok"})
    assert m15.make_claim(**kw)["claim_id"] == m15.make_claim(**kw)["claim_id"]


# --------------------------------------------------------------------------- #
# 4. confidence
# --------------------------------------------------------------------------- #
def test_implausible_pH_low_confidence_and_flag():
    spec = {"doi": "d", "pmid": "1", "pages": [{"page_no": 1, "blocks": [
        {"block_id": "b", "type": "paragraph", "section": "methods",
         "text": "The sample was at pH 74 which is impossible."}]}]}
    claims = m15.extract_document(m15.document_from_fixture(spec)).claims
    ph = _claims_for("pH", claims)
    assert ph
    c = ph[0]
    assert c["value"] == 74.0
    assert c["plausibility"]["plausible"] is False
    assert "implausible_value" in c["flags"]
    assert c["confidence"] < 0.3      # heavily down-weighted


def test_table_beats_prose_confidence():
    # same field (pH) in a table cell vs methods prose -> table has higher clarity
    conf_table, _, _ = m15.estimate_confidence(method="table_cell",
                                               location_type="table",
                                               field="pH", value=7.4)
    conf_prose, _, _ = m15.estimate_confidence(method="unit_grammar",
                                               location_type="methods_prose",
                                               field="pH", value=7.4)
    conf_caption, _, _ = m15.estimate_confidence(method="unit_grammar",
                                                 location_type="figure_caption",
                                                 field="pH", value=7.4)
    assert conf_table > conf_prose > conf_caption


def test_calibrate_confidence_ece_brier():
    labeled = [{"confidence": 0.9, "correct": True},
               {"confidence": 0.9, "correct": True},
               {"confidence": 0.1, "correct": False},
               {"confidence": 0.1, "correct": False}]
    cal = m15.calibrate_confidence(labeled)
    assert cal["n"] == 4
    assert 0.0 <= cal["ece"] <= 0.2       # well-calibrated -> low ECE
    assert cal["brier"] is not None
    # empty input -> nulls, never raises
    assert m15.calibrate_confidence([])["ece"] is None


# --------------------------------------------------------------------------- #
# 5. conflict detection
# --------------------------------------------------------------------------- #
def test_conflict_fires_on_divergent_gamma():
    result = m15.run_pipeline(cpad_curves=None)
    conflicts = result["conflicts"]
    gammas = [c for c in conflicts if c["field"] == "gamma"]
    assert gammas, "expected a gamma conflict between FIX-2 and FIX-3"
    cf = gammas[0]
    assert set(cf["values"]) == {0.5, 1.2}
    # both provenances retained
    pmids = {cl["pmid"] for cl in cf["claims"]}
    assert pmids == {"20000001", "20000002"}
    for cl in cf["claims"]:
        assert cl["source_text"] and cl["claim_id"]
    assert "never_auto_resolved" in cf["honesty"]


def test_no_conflict_within_tolerance():
    claims = [
        m15.make_claim(field="gamma", value=1.00, unit=None, doi="d1", pmid="A",
                       page=1, location_type="results_prose", block_id="b",
                       char_span=[0, 1], source_text="g", stage="s",
                       method="regex_exact", confidence=0.9, group="reported_results",
                       plausibility={"plausible": True, "reason": "ok"}),
        m15.make_claim(field="gamma", value=1.05, unit=None, doi="d2", pmid="B",
                       page=1, location_type="results_prose", block_id="b",
                       char_span=[0, 1], source_text="g", stage="s",
                       method="regex_exact", confidence=0.9, group="reported_results",
                       plausibility={"plausible": True, "reason": "ok"}),
    ]
    ctx = {"A": {"protein": "X", "pH": 7.4, "temperature": 37.0},
           "B": {"protein": "X", "pH": 7.4, "temperature": 37.0}}
    assert m15.detect_conflicts(claims, ctx) == []       # 5% spread < 15% tol


def test_conflicts_vs_computed():
    claims = [m15.make_claim(field="gamma", value=1.2, unit=None, doi="d", pmid="A",
                             page=1, location_type="results_prose", block_id="b",
                             char_span=[0, 1], source_text="g", stage="s",
                             method="regex_exact", confidence=0.9,
                             group="reported_results",
                             plausibility={"plausible": True, "reason": "ok"})]
    # engine computed gamma 0.6 for pmid A -> literature-vs-computed conflict
    out = m15.conflicts_vs_computed(claims, {"A": {"gamma": 0.6}})
    assert len(out) == 1
    assert out[0]["kind"] == "literature_vs_computed"
    # within tolerance -> no conflict
    assert m15.conflicts_vs_computed(claims, {"A": {"gamma": 1.25}}) == []


# --------------------------------------------------------------------------- #
# 6. CPAD round-trip validation
# --------------------------------------------------------------------------- #
def _synthetic_cpad_for_pmid(pmid):
    """A tiny synthetic CPAD corpus whose curated condition_vector mirrors FIX-1
    (Aβ40, 20 µM, pH 7.35, 37°C, ThT, phosphate). Hermetic — no real file read."""
    return [{
        "series_id": "SYN-CPAD-1", "protein_id": "Amyloid Beta peptide-ABeta40",
        "uniprot_id": "P05067",
        "source_study": {"pmid": pmid},
        "condition_vector": {
            "concentration": {"value_uM": 20.0, "unit": "microM"},
            "temperature_C": 37.0, "pH": 7.35, "assay_type": "ThT",
            "buffer": "NaH2PO4 50 mM", "construct_id": "Wild Type"},
    }]


def test_cpad_roundtrip_high_precision_recall():
    docs = m15.fixture_documents()
    fix1 = docs[0]        # mirrors pmid 18258258
    cpad = _synthetic_cpad_for_pmid("18258258")
    # retag the synthetic curve's pmid to match fix1
    v = m15.validate_against_cpad(fix1, "18258258", cpad)
    assert v["n_curated_curves"] == 1
    assert v["recall"] == 1.0
    assert v["precision"] >= 0.8
    # buffer alias: 'phosphate' matches 'NaH2PO4 50 mM'
    assert v["per_field"]["buffer"]["match"] is True
    assert v["per_field"]["concentration"]["match"] is True
    assert v["honesty"]["accuracy_is_measured_not_asserted"] is True


def test_cpad_roundtrip_catches_wrong_field():
    docs = m15.fixture_documents()
    fix1 = docs[0]
    # deliberately-wrong curated pH (2.0) that fix1 (pH 7.4) will NOT match
    cpad = _synthetic_cpad_for_pmid("18258258")
    cpad[0]["condition_vector"]["pH"] = 2.0
    v = m15.validate_against_cpad(fix1, "18258258", cpad)
    assert v["per_field"]["pH"]["match"] is False    # caught


# --------------------------------------------------------------------------- #
# 7. front-end seam: dependency ABSENT branch (honest failure)
# --------------------------------------------------------------------------- #
def test_frontend_absent_raises_naming_the_missing_dependency():
    """No implementation -> NotImplementedError that names the SPECIFIC dep and
    the way to plug one in. No degraded stand-in, ever."""
    dep_token = {"pdf_to_document": "pdfminer", "ocr_page": "pytesseract",
                 "ner_extract": "spaCy", "digitize_figure_curve": "digitizer"}
    with _all_frontends_absent():
        for name, token in dep_token.items():
            try:
                getattr(m15, name)(_FRONTEND_ARGS[name])
                assert False, f"{name} must raise when its dependency is absent"
            except NotImplementedError as e:
                msg = str(e)
                assert token in msg, f"{name} must name {token}: {msg}"
                assert "NOT INSTALLED" in msg          # the true reason
                assert "register_frontend" in msg      # the documented plug point
                assert "honesty_block" in msg
                assert "not about network access" in msg   # stale claim retracted


def test_frontend_absent_never_returns_a_degraded_substitute():
    """resolve_frontend() is the capability query: it raises rather than handing
    back any callable that could fabricate a result."""
    with _all_frontends_absent():
        for name in m15.FRONTEND_NAMES:
            try:
                m15.resolve_frontend(name)
                assert False, f"resolve_frontend({name!r}) must raise when absent"
            except NotImplementedError:
                pass
            st = m15.frontend_status(name)
            assert st["available"] is False and st["source"] is None


# --------------------------------------------------------------------------- #
# 7b. front-end seam: implementation PRESENT branch (real delegation)
# --------------------------------------------------------------------------- #
def test_injected_frontend_delegates_and_returns_its_result():
    """An injected implementation is actually CALLED, with the caller's argument,
    and its result is returned untouched — the seam is genuinely pluggable."""
    payloads = {
        "pdf_to_document": {"doi": "10.0/injected", "pmid": "7", "pages": []},
        "ocr_page": "OCR'D PAGE TEXT",
        "ner_extract": [{"text": "tau", "label": "PROTEIN", "start": 0, "end": 3}],
        "digitize_figure_curve": [[0.0, 0.0], [1.0, 0.42]],
    }
    seen = []
    with _all_frontends_absent():
        for name, out in payloads.items():
            def impl(arg, _name=name, _out=out):
                seen.append((_name, arg))
                return _out
            with _frontend(name, impl):
                got = getattr(m15, name)(_FRONTEND_ARGS[name])
                assert got is out, f"{name} did not return the injected result"
                st = m15.frontend_status(name)
                assert st["available"] is True and st["source"] == "injected"
                assert m15.resolve_frontend(name) is impl
            # withdrawing the injection restores the honest failure
            try:
                getattr(m15, name)(_FRONTEND_ARGS[name])
                assert False, f"{name} must fail again once unregistered"
            except NotImplementedError:
                pass
    assert [n for n, _ in seen] == list(payloads)          # every one delegated
    assert dict(seen) == _FRONTEND_ARGS                    # with the real argument


def test_register_frontend_validates_and_never_half_registers():
    try:
        m15.register_frontend("not_a_frontend", lambda x: x)
        assert False, "unknown front-end must raise"
    except ValueError:
        pass
    try:
        m15.register_frontend("ocr_page", "not-callable")
        assert False, "non-callable implementation must raise"
    except TypeError:
        pass
    assert m15.frontend_status("ocr_page")["source"] != "injected"


# --------------------------------------------------------------------------- #
# 7c. honesty reflects ACTUAL detection in both branches
# --------------------------------------------------------------------------- #
def test_honesty_block_reports_true_detected_frontend_state():
    with _all_frontends_absent():
        hb = m15.honesty_block()
        caps = hb["frontend_capabilities"]
        assert set(caps) == set(m15.FRONTEND_NAMES)
        assert all(c["available"] is False for c in caps.values())
        assert "LIVE now: none" in hb["stubs_are_honest"]
        assert "ocr_page" in hb["stubs_are_honest"]

        # flip ONE seam live: the reported state must change with it
        with _frontend("ocr_page", lambda image: "page text"):
            hb2 = m15.honesty_block()
            caps2 = hb2["frontend_capabilities"]
            assert caps2["ocr_page"]["available"] is True
            assert caps2["ocr_page"]["source"] == "injected"
            assert caps2["pdf_to_document"]["available"] is False
            assert "LIVE now: ocr_page" in hb2["stubs_are_honest"]
            assert "pdf_to_document (needs" in hb2["stubs_are_honest"]


def test_honesty_text_makes_no_network_claim():
    """REGRESSION: unavailability is attributed to an uninstalled dependency, not
    to a (false) claim that the machine is offline."""
    blob = json.dumps(m15.honesty_block(), default=str).lower()
    assert "no network" not in blob
    assert "offline env" not in blob
    assert "not installed" in blob
    assert "no network" not in (m15.__doc__ or "").lower()
    with _all_frontends_absent():
        try:
            m15.ocr_page(None)
            assert False
        except NotImplementedError as e:
            assert "no network" not in str(e).lower()


def test_import_and_detection_stay_dependency_free():
    """Importing m15 pulls in NO third-party module, and capability detection
    PROBES (find_spec) rather than executing a backing library."""
    import os
    import subprocess
    import sys
    engine_dir = os.path.dirname(os.path.abspath(m15.__file__))
    code = ("import sys, json\n"
            f"sys.path.insert(0, {engine_dir!r})\n"
            "import m15_litingest as m\n"
            "third = ('fitz', 'pdfminer', 'pytesseract', 'spacy', 'numpy', 'scipy')\n"
            "after_import = [x for x in third if x in sys.modules]\n"
            "m.honesty_block()\n"
            "after_detect = [x for x in ('fitz', 'pytesseract', 'spacy')\n"
            "                if x in sys.modules]\n"
            "print(json.dumps([after_import, after_detect]))\n")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True)
    assert proc.returncode == 0, proc.stderr
    after_import, after_detect = json.loads(proc.stdout.strip().splitlines()[-1])
    assert after_import == [], f"import pulled third-party modules: {after_import}"
    assert after_detect == [], f"detection executed a backing library: {after_detect}"


def test_builtin_pdf_adapter_delegates_to_a_detected_library():
    """The DETECTED-library branch, exercised with a stand-in module in place of
    PyMuPDF (no dependency installed, no PDF read). Proves the built-in adapter
    is wired to detection and really translates the library's output into the
    `Document` abstraction — text + bbox only, nothing invented."""
    import sys
    import types

    class _FakePage:
        def __init__(self, blocks):
            self._blocks = blocks

        def get_text(self, kind):
            assert kind == "blocks"
            return self._blocks

    class _FakeDoc:
        metadata = {"title": "A Fake Paper"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            # (x0, y0, x1, y1, text, block_no, block_type); type 1 = image
            return iter([_FakePage([
                (0, 0, 10, 5, "Materials and Methods", 0, 0),
                (0, 6, 10, 20, "Tau was prepared at 25 µM at pH 7.0.", 1, 0),
                (0, 21, 10, 30, "", 2, 1),
            ])])

    fake_fitz = types.ModuleType("fitz")
    fake_fitz.open = lambda path: _FakeDoc()
    saved_mod = sys.modules.get("fitz")
    sys.modules["fitz"] = fake_fitz
    try:
        with _all_frontends_absent():
            m15._BACKEND_CACHE["pdf_to_document"] = {"module": "fitz",
                                                     "detail": "stand-in (test)"}
            st = m15.frontend_status("pdf_to_document")
            assert st["available"] is True and st["source"] == "library"
            doc = m15.pdf_to_document("paper.pdf")
    finally:
        if saved_mod is None:
            sys.modules.pop("fitz", None)
        else:
            sys.modules["fitz"] = saved_mod

    blocks = doc["pages"][0]["blocks"]
    assert len(blocks) == 2                       # the empty image block is dropped
    assert blocks[0]["type"] == "heading"         # IMRaD heading, matched exactly
    assert blocks[1]["type"] == "paragraph"       # never guessed as table/figure
    assert blocks[1]["bbox"] == [0, 6, 10, 20]
    assert doc["_frontend"]["backend"] == "fitz"
    assert "doi_pmid_not_resolved_by_frontend" in doc["_flags"]
    # a parsed Document is the SAME abstraction: the deterministic pipeline runs
    # on it unchanged
    claims = m15.extract_document(doc).claims
    assert any(c["field"] == "pH" and c["value"] == 7.0 for c in claims)
    assert m15.detect_sections(doc)[blocks[1]["block_id"]] == "methods"


def test_pdf_block_typing_never_guesses_tables_or_figures():
    assert m15._pdf_block_type("Materials and Methods") == "heading"
    assert m15._pdf_block_type("  3. Results  ") == "heading"
    assert m15._pdf_block_type("Results show that the half-time was 8 h and the "
                               "reaction was seeded.") == "paragraph"
    # a SHORT line that merely STARTS with a section word is prose, not a heading
    assert m15._pdf_block_type("Results were reproducible.") == "paragraph"
    assert m15._pdf_block_type("Discussion of the fit follows.") == "paragraph"
    for text in ("Protein  Conc  pH\nTau  25  7.0", "Fig 1. ThT kinetics.", ""):
        assert m15._pdf_block_type(text) == "paragraph"


# --------------------------------------------------------------------------- #
# 7d. a LIVE NER front-end augments at its own LOWER reliability tier
# --------------------------------------------------------------------------- #
def _fake_ner(text):
    """What a bio-NER model returns: entities WITH char offsets into `text`."""
    if "Alpha-synuclein" not in text:
        return []
    s = text.index("Alpha-synuclein")
    return [
        {"text": "Alpha-synuclein", "label": "PROTEIN", "start": s,
         "end": s + len("Alpha-synuclein"), "backend": "fake-model:test"},
        {"text": "cytosol", "label": "CELLULAR_COMPONENT",     # no honest target
         "start": 0, "end": 5},
        {"text": "no offsets", "label": "PROTEIN"},            # no provenance
    ]


def test_ml_ner_tier_is_below_every_deterministic_method():
    assert m15.METHOD_RELIABILITY["ml_ner"] == 0.50
    for method in m15.DETERMINISTIC_METHODS:
        assert m15.METHOD_RELIABILITY["ml_ner"] < m15.METHOD_RELIABILITY[method]


def test_injected_ner_claims_stamped_ml_ner_not_regex_exact():
    doc = _methods_doc()
    block_text = doc["pages"][0]["blocks"][1]["text"]
    with _all_frontends_absent(), _frontend("ner_extract", _fake_ner):
        claims = m15.extract_document(doc, use_ner=True).claims
    ner = [c for c in claims if c["extractor"]["method"] == "ml_ner"]
    assert len(ner) == 1, "one mappable, span-carrying entity -> one claim"
    c = ner[0]
    assert c["extractor"]["method"] == "ml_ner"
    assert c["extractor"]["method"] not in ("regex_exact", "gazetteer", "table_cell")
    assert c["extractor"]["stage"] == "ml_ner_augment"
    assert c["field"] == "protein"
    # verbatim surface form, NOT laundered through the gazetteer into a canonical
    assert c["value"] == "Alpha-synuclein"
    assert c["source_text"] == block_text[c["char_span"][0]:c["char_span"][1]]
    assert "ml_ner_probabilistic_not_deterministic" in c["flags"]
    assert c["ner"]["label"] == "PROTEIN"
    # the SAME protein found deterministically outranks the model's hit
    gaz = [g for g in claims if g["field"] == "protein"
           and g["extractor"]["method"] == "gazetteer"
           and g["block_id"] == c["block_id"]]
    assert gaz and gaz[0]["confidence"] > c["confidence"]


def test_ner_augments_and_never_replaces_deterministic_claims():
    doc = _methods_doc()
    before = m15.extract_document(doc).claims
    before_ids = {b["claim_id"] for b in before}
    assert all(b["extractor"]["method"] in m15.DETERMINISTIC_METHODS for b in before)
    with _all_frontends_absent(), _frontend("ner_extract", _fake_ner):
        after = m15.extract_document(doc, use_ner=True).claims
    assert before_ids <= {a["claim_id"] for a in after}   # nothing displaced
    assert len(after) == len(before) + 1


def test_ner_unavailable_is_reported_not_silently_skipped():
    doc = _methods_doc()
    with _all_frontends_absent():
        assert m15.extract_document(doc).claims          # default path never raises
        try:
            m15.extract_document(doc, use_ner=True)
            assert False, "an explicit use_ner=True must not silently degrade"
        except NotImplementedError as e:
            assert "spaCy" in str(e)


# --------------------------------------------------------------------------- #
# 7e. never-raises
# --------------------------------------------------------------------------- #


def test_never_raises_on_empty_or_malformed():
    assert m15.document_from_fixture(None)["pages"] == []
    assert m15.document_from_fixture({})["pages"] == []
    assert m15.extract_document(None).claims == []
    assert m15.extract_document({"pages": [{"blocks": [None, 5]}]}).claims == []
    bad = m15.document_from_fixture({"pages": "notalist"})
    assert bad["pages"] == []
    # malformed block types are coerced, not fatal
    doc = m15.document_from_fixture({"doi": "d", "pmid": "1", "pages": [
        {"page_no": 1, "blocks": [{"type": "weird", "text": "pH 7.0"}]}]})
    claims = m15.extract_document(doc).claims
    assert any(c["field"] == "pH" for c in claims)


def test_honesty_block_present_on_outputs():
    result = m15.run_pipeline(cpad_curves=None)
    hb = result["honesty"]
    for key in ("claims_not_facts", "append_only", "fully_traceable",
                "confidence_is_measured", "stubs_are_honest",
                "accuracy_is_measured_not_asserted", "curator_approves"):
        assert key in hb


def test_jsonl_serialisable():
    ledger = m15.extract_document(_methods_doc())
    text = ledger.to_jsonl()
    for line in text.splitlines():
        json.loads(line)          # every line is valid JSON


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
    print(f"test_m15_litingest: {passed} passed")
    return passed


if __name__ == "__main__":
    _run_all()
