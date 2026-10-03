"""
PRISE — Module M15: Literature Ingestion & Claim Extraction
===========================================================

Implements PRISE_DESIGN.md §3-M15. M15 turns protein-aggregation papers into
STRUCTURED, TRACEABLE CLAIMS — never trusted facts. Every extraction is a
`Claim` with full provenance (doi + page + block + char-span + VERBATIM source
text), a MEASURED/calibrated confidence, and an entry in an APPEND-ONLY,
immutable ledger. A correction never overwrites: it APPENDS a new claim carrying
`supersedes=<old_id>`. Divergent values across papers surface as `conflict`
records (both provenances retained) — never auto-resolved. Extraction accuracy
is MEASURED against CPAD (the curated gold standard), not asserted.

Pipeline stages (honestly labelled deterministic-REAL vs capability-detected):
  * pdf_to_document / ocr_page / ner_extract / digitize_figure_curve -> REAL
    PLUG POINTS. Each resolves, at CALL time, either an implementation INJECTED
    by the deployment (`register_frontend`) or a BACKING LIBRARY detected on this
    interpreter, and delegates to it. When neither exists it raises a CLEAR
    NotImplementedError naming the SPECIFIC missing dependency — it never
    substitutes a heuristic, a regex imitation or an empty-but-plausible result.
    None of pdfminer/PyMuPDF/pytesseract/spaCy is installed in THIS checkout (a
    dependency-availability fact, not a statement about network access: PRISE
    deliberately depends on stdlib + numpy/scipy only), and no PDF corpus is
    bundled, so in a stock checkout all four report UNAVAILABLE. The detected
    state is reported truthfully by honesty_block()['frontend_capabilities'].
  * Everything downstream of the `Document` abstraction is deterministic-REAL,
    stdlib + (numpy optional) only, and operates on synthetic FIXTURE documents
    that mimic what a born-digital PDF parser WOULD emit.

What M15 does (deterministic-real):
  1. Document abstraction (`Document`) + `document_from_fixture` + 3 fixtures.
  2. IMRaD section detection (heading regex + cues). Methods = the goldmine.
  3. Deterministic extractors (regex + unit grammar + offline dictionaries) for
     the 15 targets, grouped by epistemic status:
       CONDITIONS (Methods/table/caption -> M11 ontology / condition_vector),
       REPORTED RESULTS (Results -> compared vs PRISE's own computed values),
       REPORTED SEMANTIC CLAIMS (lowest confidence, flagged narrative).
     Plus a TABLE parser (header -> field map -> per-row claims) and a CAPTION
     parser.
  4. Append-only claim ledger + JSONL persistence + supersede-by-append.
  5. Confidence estimation (method reliability x source clarity x plausibility x
     cross-field consistency) + a `calibrate_confidence` ECE/Brier hook.
  6. Conflict detection (group by protein+field+condition-context; divergent ->
     conflict, both provenances) + `conflicts_vs_computed`.
  7. CPAD-validation harness `validate_against_cpad` (field-level precision /
     recall / accuracy + confidence calibration on a fixture mirroring a real
     pmid).
  8. CLI over the fixtures -> lit_claims.jsonl / lit_conflicts.json /
     lit_validation.json + a readable summary.

=== HONESTY (mandatory — emitted as a machine-readable block on every output) ===
  * CLAIMS NOT FACTS: an extraction is a proposal, never a trusted fact.
  * APPEND-ONLY: the ledger is immutable; a correction appends a NEW claim with
    `supersedes` set; the original is NEVER mutated or deleted.
  * FULLY TRACEABLE: every claim resolves to doi + page + span/table/figure and
    carries the VERBATIM source text.
  * CONFIDENCE IS MEASURED: per-claim confidence is a calibrated estimate with
    ECE/Brier hooks — not an asserted number; implausible values are flagged.
  * FRONT-ENDS ARE HONEST: PDF-parse / OCR / ML-NER / figure-curve digitization
    are capability-detected plug points. They delegate when an implementation is
    injected or a backing library is installed, and otherwise raise
    NotImplementedError naming the missing dependency. What is reported is what
    was DETECTED — nothing is faked, and a degraded stand-in is never returned.
  * ACCURACY IS MEASURED, NOT ASSERTED: precision/recall are computed against
    CPAD's curated condition_vector, not claimed.
  * A CURATOR APPROVES: M15 PROPOSES. Only human-approved, high-confidence,
    non-conflicting claims would ever graduate into the analytical corpus —
    nothing is auto-trusted.

All outputs are pure, deterministic, JSON-serialisable. Nothing here raises on
empty/malformed input: a bad document yields an empty ledger + a flag.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

M15_VERSION = "m15-litingest-1.0"

# A FIXED, deterministic timestamp for `extracted_at`. WHY: reproducibility —
# the ledger must be byte-identical across runs, so we never call wall-clock
# time. This is the corpus/harness build stamp, not "now".
EXTRACTED_AT = "2026-07-01T00:00:00Z"

# The 15 extraction targets, grouped by epistemic status (see module docstring).
CONDITION_FIELDS = ("protein", "uniprot", "mutation", "construct", "assay", "dye",
                    "temperature", "pH", "ionic_strength", "buffer",
                    "reducing_agent", "agitation", "seeding", "concentration")
RESULT_FIELDS = ("gamma", "lag_time", "half_time", "reported_uncertainty")
SEMANTIC_FIELDS = ("mechanism", "model", "equation")


# ============================ HONESTY BLOCK ================================= #
def honesty_block() -> dict:
    """The machine-readable honesty stamp attached to every M15 output (§3-M15)."""
    return {
        "module": M15_VERSION,
        "claims_not_facts": (
            "every extraction is a CLAIM (a proposal with provenance + measured "
            "confidence), never a trusted fact"),
        "append_only": (
            "the ledger is immutable; a correction APPENDS a new claim with "
            "`supersedes` set; originals are never mutated or deleted"),
        "fully_traceable": (
            "every claim resolves to doi + page + block + char_span/table/figure "
            "and carries VERBATIM source_text"),
        "confidence_is_measured": (
            "per-claim confidence is a calibrated estimate (ECE/Brier hooks), not "
            "an asserted number; implausible values are flagged and down-weighted"),
        "accuracy_is_measured_not_asserted": (
            "field-level precision/recall are computed against CPAD's curated "
            "condition_vector — not claimed"),
        "stubs_are_honest": _frontend_honesty_sentence(),
        "frontend_capabilities": frontend_capabilities(),
        "runs_on_fixtures": (
            "no PDF corpus is bundled with PRISE, so the reference pipeline runs on "
            "synthetic FIXTURE documents mimicking what a born-digital parser would "
            "emit; a live pdf_to_document front-end feeds the SAME Document "
            "abstraction, so nothing downstream changes"),
        "curator_approves": (
            "M15 PROPOSES; only human-approved, high-confidence, non-conflicting "
            "claims would ever graduate into the analytical corpus — nothing auto-"
            "trusted"),
    }


# ============ PLUGGABLE FRONT-ENDS (capability-detected + injectable) ======= #
# These four are the ONLY seams between PRISE and the outside (PDF/pixel/ML)
# world. Each one is a REAL plug point, resolved in this order at CALL time:
#
#   1. an implementation INJECTED by the deployment/test via register_frontend();
#   2. a BACKING LIBRARY detected on this interpreter (guarded, lazy — detection
#      uses importlib.util.find_spec and executes no third-party module code;
#      the library is only imported when the front-end is actually called);
#   3. otherwise -> NotImplementedError naming the SPECIFIC missing dependency.
#
# WHY resolution is lazy: importing m15_litingest must stay stdlib-only and fast,
# and a deployment that installs pdfminer AFTER import must still be served.
# WHAT IS NEVER DONE: when no implementation exists we do NOT fall back to a
# regex/heuristic imitation and we do NOT return an empty-but-plausible result.
# A missing capability is REPORTED, never simulated — the deterministic pipeline
# below is a *different*, honestly-labelled path (Document + fixtures), not a
# stand-in for a parse/OCR/NER result.

# name -> {dep: human dependency name, candidates: import names probed in order,
#          returns: contract, note: extra honesty text}
_FRONTEND_SPECS = {
    "pdf_to_document": {
        "dep": "pdfminer.six / PyMuPDF (import name `pdfminer` / `fitz`)",
        "candidates": ("fitz", "pdfminer.high_level"),
        "returns": "a `Document` dict (see document_from_fixture)",
        "note": ("the built-in adapter recovers TEXT BLOCKS + bboxes only; it does "
                 "NOT segment tables or figures — a deployment needing those should "
                 "register its own camelot/layout-model-backed front-end"),
    },
    "ocr_page": {
        "dep": "pytesseract + the Tesseract OCR binary",
        "candidates": ("pytesseract",),
        "returns": "the page text as a str",
        "note": ("the Python package and the Tesseract BINARY are separate "
                 "dependencies; a missing binary is reported, never worked around"),
    },
    "ner_extract": {
        "dep": ("spaCy/scispaCy + a bio-NER model (NER_MODEL_NAME = "
                "%r)"),                      # filled in _dep_label()
        "candidates": ("spacy",),
        "returns": ("a list of {text,label,start,end} entity dicts with CHAR "
                    "OFFSETS into the input text"),
        "note": ("an ML-NER hit AUGMENTS the deterministic extractors, never "
                 "replaces them, and is stamped extractor.method='ml_ner' — a "
                 "LOWER reliability tier (0.50) than regex_exact (0.95)"),
    },
    "digitize_figure_curve": {
        "dep": "a figure-curve digitizer (e.g. WebPlotDigitizer, offline)",
        "candidates": (),                    # deliberately empty — see note
        "returns": "a list of [x, y] points in DATA coordinates",
        "note": ("INJECTION-ONLY: no importable Python library performs axis-"
                 "calibrated curve digitization, so there is nothing to detect. "
                 "Tracing pixels without a calibrated axis would FABRICATE data, "
                 "so no built-in adapter is provided"),
    },
}
FRONTEND_NAMES = tuple(_FRONTEND_SPECS)

# scispaCy's bio-NER model. A module-level knob, not a hidden constant: a
# deployment may point it at whatever model it actually installed.
NER_MODEL_NAME = "en_ner_bionlp13cg_md"

# name -> injected callable. Highest precedence; empty in a stock checkout.
_FRONTEND_IMPLS = {}
# name -> {"module": <import name or None>, "detail": str}. Detection cache.
_BACKEND_CACHE = {}
# (model_name) -> loaded spaCy pipeline, so we load a model at most once.
_NER_PIPELINES = {}

_MISSING_DEP_NOTE = (
    "PRISE front-end `{name}` has NO implementation available: it requires {dep}, "
    "which is NOT INSTALLED on this interpreter. (This is a statement about "
    "DEPENDENCY AVAILABILITY, not about network access — PRISE deliberately ships "
    "with stdlib + numpy/scipy only and adds no parsing/OCR/ML dependency.) "
    "Nothing was faked: no heuristic, regex imitation or empty-but-plausible "
    "result was substituted for {returns}. To make it live, either install the "
    "dependency or inject an implementation with "
    "register_frontend({name!r}, your_callable). "
    "See honesty_block()['frontend_capabilities'] for the detected state. "
    "The deterministic pipeline is unaffected: it runs on the Document "
    "abstraction + synthetic fixtures via document_from_fixture().")


def _dep_label(name: str) -> str:
    """Human dependency name for a front-end (NER's includes the model name)."""
    dep = _FRONTEND_SPECS[name]["dep"]
    return dep % (NER_MODEL_NAME,) if "%r" in dep else dep


def _detect_backend(name: str, refresh: bool = False) -> dict:
    """Locate a backing library for `name` WITHOUT executing it.

    Uses importlib.util.find_spec, so a heavyweight dep (spaCy) is *not* imported
    just to answer 'is it there?'. Cached per process; register_frontend() and
    refresh=True invalidate the cache. NEVER raises."""
    if not refresh and name in _BACKEND_CACHE:
        return _BACKEND_CACHE[name]
    candidates = _FRONTEND_SPECS[name]["candidates"]
    if candidates:
        found = {"module": None,
                 "detail": ("none of the candidate import names ("
                            + ", ".join(candidates) + ") is installed")}
    else:
        found = {"module": None,
                 "detail": "no auto-detectable backend exists; injection-only"}
    for mod in candidates:
        try:
            spec = importlib.util.find_spec(mod)
        except (ImportError, AttributeError, ValueError):
            spec = None            # missing parent package / broken installation
        if spec is not None:
            found = {"module": mod, "detail": f"found import spec for {mod!r}"}
            break
    _BACKEND_CACHE[name] = found
    return found


def register_frontend(name: str, fn):
    """INJECT a real implementation for one front-end seam; returns the previous
    implementation (or None) so a caller/test can restore it.

    `fn=None` removes the injection (same as unregister_frontend). Raises on an
    unknown name or a non-callable — a mis-registered seam must FAIL LOUDLY, not
    be silently ignored."""
    if name not in _FRONTEND_SPECS:
        raise ValueError(f"unknown front-end {name!r}; known: {sorted(_FRONTEND_SPECS)}")
    if fn is not None and not callable(fn):
        raise TypeError(f"front-end {name!r} implementation must be callable, "
                        f"got {type(fn).__name__}")
    prev = _FRONTEND_IMPLS.get(name)
    if fn is None:
        _FRONTEND_IMPLS.pop(name, None)
    else:
        _FRONTEND_IMPLS[name] = fn
    _BACKEND_CACHE.pop(name, None)          # capability state changed -> re-detect
    return prev


def unregister_frontend(name: str):
    """Remove an injected implementation; returns it (or None)."""
    return register_frontend(name, None)


def frontend_status(name: str, refresh: bool = False) -> dict:
    """The DETECTED state of one front-end — the ground truth the honesty block
    reports. Never asserts availability it has not checked. NEVER raises."""
    if name not in _FRONTEND_SPECS:
        return {"frontend": name, "available": False, "source": None,
                "error": "unknown front-end"}
    spec = _FRONTEND_SPECS[name]
    status = {"frontend": name, "available": False, "source": None, "backend": None,
              "dependency": _dep_label(name), "candidates": list(spec["candidates"]),
              "returns": spec["returns"], "detection": None, "note": spec["note"],
              # Straight from the module's own epistemics: the built-in adapters
              # cannot run without the dep, so in a dep-free checkout they are
              # UNTESTED code. Say so rather than imply a verified capability.
              "builtin_adapter_verified_here": False}
    impl = _FRONTEND_IMPLS.get(name)
    if impl is not None:
        status.update({
            "available": True, "source": "injected",
            "backend": getattr(impl, "__name__", None) or repr(impl),
            "detection": ("an implementation is injected via register_frontend(); "
                          "library detection not consulted. PRISE did NOT verify "
                          "what the injected implementation does")})
        return status
    det = _detect_backend(name, refresh=refresh)
    status.update({"available": det["module"] is not None,
                   "source": "library" if det["module"] else None,
                   "backend": det["module"], "detection": det["detail"]})
    return status


def frontend_capabilities(refresh: bool = False) -> dict:
    """{front-end name -> detected status} for all four seams. NEVER raises."""
    return {n: frontend_status(n, refresh=refresh) for n in FRONTEND_NAMES}


def _frontend_honesty_sentence(refresh: bool = False) -> str:
    """The prose honesty line about the front-ends, written from ACTUAL DETECTION.

    WHY generated, not hard-coded: the previous fixed text asserted the machine had
    'no network', which was (a) unverified and (b) false on some hosts, and it
    asserted every front-end was dead even where a dependency was installed. The
    only claim we can honestly make is what we just detected."""
    caps = frontend_capabilities(refresh=refresh)
    live = [f"{n} (via {caps[n]['source']}: {caps[n]['backend']})"
            for n in FRONTEND_NAMES if caps[n]["available"]]
    down = [f"{n} (needs {caps[n]['dependency']})"
            for n in FRONTEND_NAMES if not caps[n]["available"]]
    parts = [
        "pdf_to_document / ocr_page / ner_extract / digitize_figure_curve are "
        "CAPABILITY-DETECTED plug points: each resolves an injected implementation "
        "(register_frontend) or a detected backing library lazily at CALL time, and "
        "otherwise raises NotImplementedError naming the specific missing dependency "
        "— never a faked, heuristic or empty-but-plausible substitute."]
    parts.append("LIVE now: " + ("; ".join(live) if live else "none") + ".")
    if down:
        parts.append(
            "UNAVAILABLE now because the dependency is NOT INSTALLED on this "
            "interpreter (a dependency-availability fact — NOT a claim about network "
            "access; PRISE deliberately ships stdlib + numpy/scipy only): "
            + "; ".join(down) + ".")
    return " ".join(parts)


def _missing_dep_error(name: str) -> NotImplementedError:
    spec = _FRONTEND_SPECS[name]
    return NotImplementedError(_MISSING_DEP_NOTE.format(
        name=name, dep=_dep_label(name), returns=spec["returns"]))


def resolve_frontend(name: str):
    """Return the callable that will serve `name` (injected > detected library),
    or raise the honest, dependency-naming NotImplementedError if there is none.

    Callers can use this to ASK whether a capability exists before committing to
    a code path — it is the same resolution the four front-ends perform."""
    if name not in _FRONTEND_SPECS:
        raise ValueError(f"unknown front-end {name!r}; known: {sorted(_FRONTEND_SPECS)}")
    impl = _FRONTEND_IMPLS.get(name)
    if impl is not None:
        return impl
    backend = _detect_backend(name)["module"]
    if backend is None:
        raise _missing_dep_error(name)
    adapter = _BUILTIN_ADAPTERS[name]
    return lambda arg: adapter(arg, backend)


# ---- built-in adapters (used only when a backing library IS present) ------- #
# Each adapter delegates to the real library and translates its output into the
# PRISE contract. They are thin ON PURPOSE: an adapter that "enriched" a parse
# with guesses would reintroduce exactly the dishonesty this seam removes.

def _pdf_block_type(text: str) -> str:
    """Type a recovered PDF text block. Only two outcomes: a short line that is
    ENTIRELY an IMRaD heading (per the module's existing _SECTION_PATTERNS) is a
    'heading'; everything else stays 'paragraph'. Tables/figures are NOT guessed
    at — mis-typing a block would silently corrupt downstream provenance."""
    stripped = (text or "").strip()
    if not stripped or len(stripped) > 80:
        return "paragraph"
    core = stripped.rstrip(" .:;—-")
    for _, pat in _SECTION_PATTERNS:
        m = pat.search(stripped)
        # The heading regex must consume the WHOLE line: 'Results' is a heading,
        # 'Results show that ...' is prose. A partial match is left as prose,
        # because a mis-typed block silently corrupts every claim's provenance.
        if m and m.end() >= len(core):
            return "heading"
    return "paragraph"


def _adapter_pdf_to_document(path, backend: str) -> dict:
    """Real PDF -> `Document` via PyMuPDF or pdfminer.six. Text + bbox only."""
    pages = []
    title = None
    if backend == "fitz":                                  # PyMuPDF
        import fitz
        with fitz.open(path) as pdf:
            title = (pdf.metadata or {}).get("title") or None
            for pno, page in enumerate(pdf, start=1):
                blocks = []
                for bi, blk in enumerate(page.get_text("blocks")):
                    x0, y0, x1, y1, text = blk[0], blk[1], blk[2], blk[3], blk[4]
                    if len(blk) > 6 and blk[6] != 0:
                        continue                # image block: it carries no text
                    text = (text or "").strip()
                    if not text:
                        continue
                    blocks.append({"block_id": f"p{pno}-b{bi}",
                                   "type": _pdf_block_type(text), "text": text,
                                   "bbox": [x0, y0, x1, y1]})
                pages.append({"page_no": pno, "blocks": blocks})
    else:                                                  # pdfminer.six
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer
        for pno, layout in enumerate(extract_pages(path), start=1):
            blocks = []
            for bi, element in enumerate(layout):
                if not isinstance(element, LTTextContainer):
                    continue
                text = (element.get_text() or "").strip()
                if not text:
                    continue
                blocks.append({"block_id": f"p{pno}-b{bi}",
                               "type": _pdf_block_type(text), "text": text,
                               "bbox": list(element.bbox)})
            pages.append({"page_no": pno, "blocks": blocks})
    # Reuse the Document validator so a parsed PDF and a fixture are the SAME
    # abstraction downstream (nothing about the pipeline changes).
    doc = document_from_fixture({"doi": None, "pmid": None, "title": title,
                                 "pages": pages})
    doc["_frontend"] = {
        "name": "pdf_to_document", "backend": backend,
        "recovered": "text blocks + bboxes",
        "not_recovered": ("doi/pmid (supply them from the citation record), "
                          "table structure, figure images"),
    }
    doc["_flags"] = list(doc.get("_flags") or []) + ["doi_pmid_not_resolved_by_frontend"]
    return doc


def _adapter_ocr_page(image, backend: str) -> str:
    """Real OCR via pytesseract. A missing Tesseract BINARY is re-raised as the
    same honest missing-dependency error — never silently an empty string."""
    import pytesseract
    try:
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as exc:
        raise NotImplementedError(
            f"{_missing_dep_error('ocr_page')} [detail: the pytesseract PACKAGE is "
            f"installed but the Tesseract BINARY it drives is not on PATH: {exc}]"
        ) from exc


def _adapter_ner_extract(text, backend: str) -> list:
    """Real bio-NER via spaCy/scispaCy. Returns entities WITH char offsets, so
    every downstream claim keeps a verifiable span. spaCy present but the model
    absent is a MISSING DEPENDENCY, reported as such."""
    import spacy
    nlp = _NER_PIPELINES.get(NER_MODEL_NAME)
    if nlp is None:
        try:
            nlp = spacy.load(NER_MODEL_NAME)
        except (OSError, IOError) as exc:     # spaCy raises OSError for a missing model
            raise NotImplementedError(
                f"{_missing_dep_error('ner_extract')} [detail: spaCy is installed "
                f"but the model {NER_MODEL_NAME!r} is not: {exc}]") from exc
        _NER_PIPELINES[NER_MODEL_NAME] = nlp
    return [{"text": ent.text, "label": ent.label_, "start": ent.start_char,
             "end": ent.end_char, "backend": f"spacy:{NER_MODEL_NAME}"}
            for ent in nlp(text).ents]


_BUILTIN_ADAPTERS = {
    "pdf_to_document": _adapter_pdf_to_document,
    "ocr_page": _adapter_ocr_page,
    "ner_extract": _adapter_ner_extract,
    # digitize_figure_curve: intentionally absent — injection-only (see specs).
}


def pdf_to_document(path: str) -> dict:
    """Parse a born-digital PDF into a `Document` (pdfminer.six / PyMuPDF).

    Delegates to an injected or detected implementation; raises the honest,
    dependency-naming NotImplementedError when neither exists."""
    return resolve_frontend("pdf_to_document")(path)


def ocr_page(image) -> str:
    """OCR a scanned page image into text (pytesseract + Tesseract).

    Delegates to an injected or detected implementation; raises the honest,
    dependency-naming NotImplementedError when neither exists."""
    return resolve_frontend("ocr_page")(image)


def ner_extract(text: str) -> list:
    """ML named-entity recognition (spaCy / scispaCy bio-NER model).

    The deterministic regex+dictionary extractors below remain the DEPENDENCY-FREE
    path; a live ML-NER stage AUGMENTS (never replaces) them, and every claim it
    produces is stamped extractor.method='ml_ner' — a LOWER reliability tier than
    regex_exact (see METHOD_RELIABILITY / extract_ner_claims). Delegates to an
    injected or detected implementation; raises the honest, dependency-naming
    NotImplementedError when neither exists."""
    return resolve_frontend("ner_extract")(text)


def digitize_figure_curve(figure) -> list:
    """Digitize an x/y kinetic curve from a figure image.

    INJECTION-ONLY: there is no importable Python backend to detect, and tracing
    pixels without a calibrated axis would fabricate data — so without an injected
    implementation this raises the honest, dependency-naming NotImplementedError."""
    return resolve_frontend("digitize_figure_curve")(figure)


# ============================ DOCUMENT ABSTRACTION ========================= #
# A `Document` is what a born-digital PDF parser WOULD emit:
#   {doi, pmid, pages:[{page_no, blocks:[{block_id, type, text, bbox, rows?}]}]}
# block.type ∈ (heading|paragraph|table|figure_caption|equation).
BLOCK_TYPES = ("heading", "paragraph", "table", "figure_caption", "equation")


def document_from_fixture(spec: dict) -> dict:
    """Build a validated `Document` from a fixture dict (the deterministic path).

    Fills defaults, assigns deterministic block_ids where absent, and NEVER
    raises — a malformed spec yields a minimal document + a flag list."""
    flags = []
    if not isinstance(spec, dict):
        return {"doi": None, "pmid": None, "pages": [], "_flags": ["not_a_dict"],
                "honesty": honesty_block()}
    doc = {"doi": spec.get("doi"), "pmid": (str(spec["pmid"]) if spec.get("pmid")
                                            is not None else None),
           "title": spec.get("title"), "pages": [], "honesty": honesty_block()}
    pages = spec.get("pages") or []
    if not isinstance(pages, list):
        pages, flags = [], flags + ["pages_not_list"]
    for pi, page in enumerate(pages):
        if not isinstance(page, dict):
            flags.append(f"page_{pi}_not_dict")
            continue
        pno = page.get("page_no", pi + 1)
        blocks_out = []
        for bi, blk in enumerate(page.get("blocks") or []):
            if not isinstance(blk, dict):
                flags.append(f"block_{pi}_{bi}_not_dict")
                continue
            btype = blk.get("type")
            if btype not in BLOCK_TYPES:
                flags.append(f"block_{pi}_{bi}_bad_type_{btype}")
                btype = "paragraph"
            bid = blk.get("block_id") or f"p{pno}-b{bi}"
            out = {"block_id": bid, "type": btype,
                   "text": str(blk.get("text") or ""),
                   "bbox": blk.get("bbox"), "section": blk.get("section")}
            if btype == "table" and "rows" in blk:
                out["rows"] = blk["rows"]
                out["header"] = blk.get("header")
            blocks_out.append(out)
        doc["pages"].append({"page_no": pno, "blocks": blocks_out})
    if flags:
        doc["_flags"] = flags
    return doc


# ============================ SECTION / IMRaD DETECTION ==================== #
# Heading regexes for IMRaD sections. WHY: Methods is the metadata goldmine
# (buffer/pH/temp/agitation/seeding/concentration); Results holds reported γ/t50/
# mechanism claims. Section is used to weight source clarity in confidence.
_SECTION_PATTERNS = [
    ("abstract", re.compile(r"^\s*abstract\b", re.I)),
    ("introduction", re.compile(r"^\s*(1\.?\s*)?introduction\b", re.I)),
    ("methods", re.compile(r"^\s*(2\.?\s*)?(materials?\s+and\s+methods?|"
                           r"experimental(\s+(section|procedures?))?|methods?)\b", re.I)),
    ("results", re.compile(r"^\s*(3\.?\s*)?(results?(\s+and\s+discussion)?)\b", re.I)),
    ("discussion", re.compile(r"^\s*(4\.?\s*)?discussion\b", re.I)),
    ("references", re.compile(r"^\s*(references|bibliography)\b", re.I)),
]


def detect_sections(document: dict) -> dict:
    """Return {block_id -> section_label}. A heading opens a section; following
    blocks inherit it until the next heading. figure_caption/table/equation keep
    their own type-label as a 'section hint' but also inherit the running section.
    Explicit block['section'] overrides. NEVER raises."""
    labels = {}
    current = "front_matter"
    for page in (document.get("pages") or []):
        if not isinstance(page, dict):
            continue
        for blk in (page.get("blocks") or []):
            if not isinstance(blk, dict):
                continue
            bid = blk.get("block_id")
            if blk.get("section"):  # explicit annotation wins
                current = blk["section"]
                labels[bid] = current
                continue
            if blk.get("type") == "heading":
                text = blk.get("text", "")
                matched = None
                for label, pat in _SECTION_PATTERNS:
                    if pat.search(text):
                        matched = label
                        break
                current = matched or current
                labels[bid] = current
            else:
                labels[bid] = current
    return labels


# ============================ OFFLINE DICTIONARIES ========================= #
# Gazetteers/grammars are OFFLINE, deterministic, and curated from the CPAD
# corpus vocabulary (protein names + uniprot in the bundled corpus) + standard
# assay/buffer/mechanism/model terms. A real ML-NER stage would AUGMENT these.

# Protein gazetteer (surface form -> canonical). Built from CPAD protein_id +
# common literature aliases. Longest-match wins (see _find_proteins).
PROTEIN_GAZETTEER = {
    "amyloid beta": "Amyloid Beta peptide",
    "amyloid-beta": "Amyloid Beta peptide",
    "amyloid β": "Amyloid Beta peptide",
    "abeta42": "Amyloid Beta peptide-ABeta42",
    "aβ42": "Amyloid Beta peptide-ABeta42",
    "abeta40": "Amyloid Beta peptide-ABeta40",
    "aβ40": "Amyloid Beta peptide-ABeta40",
    "alpha-synuclein": "Alpha-Synuclein",
    "α-synuclein": "Alpha-Synuclein",
    "alpha synuclein": "Alpha-Synuclein",
    "asyn": "Alpha-Synuclein",
    "islet amyloid polypeptide": "Islet Amyloid Polypeptide",
    "iapp": "Islet Amyloid Polypeptide",
    "amylin": "Islet Amyloid Polypeptide",
    "lysozyme": "lysozyme",
    "insulin": "insulin",
    "prp": "PrP",
    "prion protein": "PrP",
    "tau": "Tau",
    "beta2-microglobulin": "beta-2-microglobulin",
    "β2-microglobulin": "beta-2-microglobulin",
    "β2m": "beta-2-microglobulin",
    "p53": "P53",
    "ure2": "Ure2 protein",
    "immunoglobulin light chain": "immunoglobulin light chain",
    "immunoglobin ki": "immunoglobin kI",
    "kappa i": "immunoglobin kI",
}

# UniProt accession regex (e.g. P05067, Q761V2). Standard UniProtKB pattern.
_UNIPROT_RE = re.compile(
    r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})\b")

# Point-mutation regex: wild-type AA + position + mutant AA (e.g. A30P, E22G).
_MUTATION_RE = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY]\d{1,4}[ACDEFGHIKLMNPQRSTVWY])\b")

# Assay + dye dictionary (surface -> canonical). ThT etc. are fibril-reporting;
# turbidity/CD/DLS are not fibril-specific (kept for completeness).
ASSAY_DYE = {
    "thioflavin t": ("ThT", "dye"),
    "thioflavin-t": ("ThT", "dye"),
    "tht": ("ThT", "dye"),
    "thioflavin s": ("ThS", "dye"),
    "ths": ("ThS", "dye"),
    "congo red": ("CongoRed", "dye"),
    "turbidity": ("turbidity", "assay"),
    "circular dichroism": ("CD", "assay"),
    " cd ": ("CD", "assay"),
    "dynamic light scattering": ("DLS", "assay"),
    "dls": ("DLS", "assay"),
}

# Buffer dictionary. Standard aggregation-assay buffers.
BUFFER_TERMS = {
    "tris": "Tris", "phosphate": "phosphate", "pbs": "PBS", "hepes": "HEPES",
    "glycine": "glycine", "acetate": "acetate", "citrate": "citrate",
    "mes": "MES", "mops": "MOPS", "sodium phosphate": "phosphate",
    "potassium phosphate": "phosphate",
}

# Reducing agents. Matched anywhere in Methods/table text.
_REDUCING_AGENT_RE = re.compile(
    r"\b(DTT|TCEP|BME|2-?ME|beta-?mercaptoethanol|2-mercaptoethanol|"
    r"dithiothreitol|tris\(2-carboxyethyl\)phosphine)\b", re.I)

# Mechanism dictionary (narrative, lowest-confidence semantic claims).
MECHANISM_TERMS = {
    "secondary nucleation": "secondary_nucleation",
    "primary nucleation": "primary_nucleation",
    "fragmentation": "fragmentation",
    "elongation": "elongation",
    "surface-catalysed": "secondary_nucleation",
    "surface catalyzed": "secondary_nucleation",
    "monomer-dependent secondary": "secondary_nucleation",
}

# Kinetic-model dictionary (narrative semantic claims).
MODEL_TERMS = {
    "sigmoidal": "sigmoidal",
    "finke-watzky": "Finke-Watzky",
    "finke watzky": "Finke-Watzky",
    "gompertz": "Gompertz",
    "amylofit": "AmyloFit",
    "knowles": "Knowles",
    "cohen": "Knowles",
    "boltzmann": "sigmoidal",
    "hill": "sigmoidal",
}


# ============================ UNIT GRAMMAR ================================= #
# Concentration unit normalisation to µM. Deterministic factor table.
_CONC_TO_UM = {"nm": 1e-3, "um": 1.0, "µm": 1.0, "microm": 1.0, "micromolar": 1.0,
               "mm": 1e3, "m": 1e6}
_CONC_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|to|–)?\s*(\d+(?:\.\d+)?)?\s*"
    r"(nM|µM|uM|microM|micromolar|mM|M)\b", re.I)
_PH_RE = re.compile(r"\bpH\s*(\d{1,3}(?:\.\d+)?)", re.I)
_TEMP_RE = re.compile(r"(-?\d{1,3}(?:\.\d+)?)\s*°?\s*C\b")
_IONIC_RE = re.compile(r"\b(NaCl|KCl)\b[^.\d]{0,12}?(\d+(?:\.\d+)?)\s*(mM|M|µM|uM)\b", re.I)
_GAMMA_RE = re.compile(r"(?:scaling exponent|γ|gamma)\s*(?:of|=|:|was|is)?\s*"
                       r"[-–]?\s*(-?\d+(?:\.\d+)?)", re.I)
_HALFTIME_RE = re.compile(r"\b(?:t\s?50|t1/2|t½|half[- ]?time)\b\s*(?:of|=|:|was|is)?\s*"
                          r"(\d+(?:\.\d+)?)\s*(h|hr|hours?|min|minutes?|s|sec)?", re.I)
_LAG_RE = re.compile(r"\blag(?:\s*(?:time|phase|period))?\b\s*(?:of|=|:|was|is)?\s*"
                     r"(\d+(?:\.\d+)?)\s*(h|hr|hours?|min|minutes?|s|sec)?", re.I)
# ± uncertainty / CI (attach to a reported result). Capture the ± value.
_UNCERT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:±|\+/-|\+-)\s*(\d+(?:\.\d+)?)")

# Agitation grammar. rpm value optional.
_AGIT_QUIESCENT_RE = re.compile(r"\b(quiescent|without agitation|no agitation|"
                                r"unstirred|non-agitated)\b", re.I)
_AGIT_MOVED_RE = re.compile(r"\b(shaken|shaking|stirred|stirring|orbital|agitat\w*|"
                            r"(\d+)\s*rpm)\b", re.I)
# Seeding grammar.
_SEED_YES_RE = re.compile(r"\b(seeded|preformed seeds?|preformed fibrils?|"
                          r"(\d+(?:\.\d+)?)\s*%?\s*seeds?|with seeds?)\b", re.I)
_SEED_NO_RE = re.compile(r"\b(unseeded|without seeds?|de novo|seed[- ]free)\b", re.I)


# ============================ CLAIM DATA MODEL ============================= #
def _claim_id(doi, pmid, block_id, char_span, field, value, method) -> str:
    """Deterministic claim id = short hash over the provenance-defining tuple.

    WHY deterministic: re-extracting the SAME text yields the SAME id, so the
    append-only ledger is idempotent per (source, field, value, method) and a
    genuine correction (different value) gets a genuinely different id."""
    payload = json.dumps([doi, pmid, block_id, char_span, field, value, method],
                         sort_keys=True, default=str)
    return "clm_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_claim(*, field, value, unit, doi, pmid, page, location_type, block_id,
               char_span, source_text, stage, method, confidence,
               group, plausibility, flags=None, supersedes=None,
               extra=None) -> dict:
    """Assemble one immutable Claim (§3-M15). char_span is [start,end) into the
    block text; source_text is the VERBATIM slice. NEVER raises."""
    claim = {
        "claim_id": _claim_id(doi, pmid, block_id, list(char_span or []), field,
                              value, method),
        "field": field, "value": value, "unit": unit,
        "group": group,                       # conditions|reported_results|semantic
        "doi": doi, "pmid": pmid, "page": page,
        "location_type": location_type,       # methods_prose|table|figure_caption|...
        "block_id": block_id, "char_span": list(char_span) if char_span else None,
        "source_text": source_text,           # VERBATIM
        "extractor": {"stage": stage, "method": method, "version": M15_VERSION},
        "confidence": round(float(confidence), 4),
        "plausibility": plausibility,         # {plausible: bool, reason: str}
        "flags": flags or [],
        "extracted_at": EXTRACTED_AT,
        "supersedes": supersedes,             # None, or the claim_id being corrected
    }
    if extra:
        claim.update(extra)
    return claim


# ======================= APPEND-ONLY CLAIM LEDGER ========================= #
class ClaimLedger:
    """An APPEND-ONLY, immutable claim ledger (§3-M15).

    Invariants (enforced + tested):
      * append_claim NEVER mutates or deletes an existing claim.
      * a duplicate (same claim_id) is a no-op APPEND (idempotent), not an edit.
      * a correction is a NEW claim carrying supersedes=<old_id>; the superseded
        claim remains byte-identical in the ledger forever.
    Persistence is append-only JSONL."""

    def __init__(self):
        self._claims = []          # ordered; never reordered or edited
        self._by_id = {}           # id -> index (first-seen wins)

    def append_claim(self, claim: dict) -> dict:
        """Append a claim. Returns the stored claim. If claim_id already present,
        this is an idempotent no-op (the original is NOT touched)."""
        if not isinstance(claim, dict) or "claim_id" not in claim:
            return {"error": "not_a_claim", "input": claim}
        cid = claim["claim_id"]
        if cid in self._by_id:
            return self._claims[self._by_id[cid]]  # original, untouched
        self._by_id[cid] = len(self._claims)
        self._claims.append(claim)
        return claim

    def supersede(self, old_id: str, new_claim: dict) -> dict:
        """Correct a claim by APPENDING a new one that points back via supersedes.
        The old claim is NOT modified. Returns the appended new claim."""
        nc = dict(new_claim)
        nc["supersedes"] = old_id
        # Recompute id so the correcting claim is a distinct ledger entry.
        nc["claim_id"] = _claim_id(nc.get("doi"), nc.get("pmid"),
                                   nc.get("block_id"), nc.get("char_span") or [],
                                   nc.get("field"), nc.get("value"),
                                   nc["extractor"]["method"] + "|supersede:" + str(old_id))
        return self.append_claim(nc)

    def get(self, claim_id: str):
        idx = self._by_id.get(claim_id)
        return self._claims[idx] if idx is not None else None

    @property
    def claims(self) -> list:
        return list(self._claims)          # a copy; callers cannot mutate internals

    def active_claims(self) -> list:
        """Claims not superseded by a later claim (the current 'proposed' view).
        The superseded originals REMAIN in the full ledger — this is a view, not a
        deletion."""
        superseded = {c["supersedes"] for c in self._claims if c.get("supersedes")}
        return [c for c in self._claims if c["claim_id"] not in superseded]

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(c, sort_keys=True, default=str)
                         for c in self._claims)

    def persist(self, path) -> str:
        """Write the FULL ledger (incl. superseded originals) as append-only JSONL."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = self.to_jsonl()
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text + ("\n" if text else ""))
        return str(p)


# ============================ CONFIDENCE ESTIMATION ======================= #
# Confidence = method_reliability x source_clarity x plausibility_factor x
# consistency_factor, all in (0,1]. These priors are a VERSIONED modeling choice
# and are CALIBRATED (ECE/Brier) against a labeled fixture set — never asserted.

# Method reliability: regex-exact > dictionary > ML-NER > inferred > OCR'd text.
# NOTE the tiering is DELIBERATE and load-bearing: a LIVE ML-NER hit sits at 0.50,
# BELOW every deterministic extractor, because a model's entity is a probabilistic
# guess while a regex/table match is a literal reading of the source. A live
# front-end therefore never PROMOTES a claim — see extract_ner_claims().
METHOD_RELIABILITY = {
    "regex_exact": 0.95, "unit_grammar": 0.93, "table_cell": 0.97,
    "dictionary": 0.85, "gazetteer": 0.82, "inferred": 0.55, "ocr_stub": 0.30,
    "ml_ner": 0.50,          # a LIVE ml-NER result (injected or library-backed)
    "ml_ner_stub": 0.50,     # retained alias: same tier, pre-plug-seam name
}
# The methods the dependency-free deterministic pipeline may ever stamp. Anything
# outside this set came from a front-end and must carry its own (lower) tier.
DETERMINISTIC_METHODS = ("regex_exact", "unit_grammar", "table_cell", "dictionary",
                         "gazetteer")
# Source clarity: table > methods-prose > caption > abstract > results-prose.
SOURCE_CLARITY = {
    "table": 0.97, "methods_prose": 0.90, "figure_caption": 0.75,
    "results_prose": 0.72, "abstract": 0.60, "equation": 0.65,
    "front_matter": 0.55, "discussion": 0.60, "introduction": 0.55,
}

# Physically-plausible ranges for numeric condition fields (unit-normalised).
PLAUSIBLE_RANGES = {
    "pH": (0.0, 14.0), "temperature": (-20.0, 100.0),
    "concentration": (1e-6, 1e7),   # µM; must be > 0
    "gamma": (-2.0, 5.0), "half_time": (0.0, 1e6), "lag_time": (0.0, 1e6),
}


def _plausibility(field: str, value) -> dict:
    """Range/plausibility check. Implausible -> {plausible:False} + reason.
    Non-numeric fields are trivially plausible (dictionary-gated already)."""
    rng = PLAUSIBLE_RANGES.get(field)
    if rng is None or value is None:
        return {"plausible": True, "reason": "no_numeric_range" if rng is None
                else "no_value"}
    try:
        v = float(value)
    except (TypeError, ValueError):
        return {"plausible": True, "reason": "non_numeric"}
    lo, hi = rng
    if v < lo or v > hi:
        return {"plausible": False,
                "reason": f"{field}={v} outside plausible [{lo},{hi}]"}
    if field == "concentration" and v <= 0:
        return {"plausible": False, "reason": "concentration must be > 0"}
    return {"plausible": True, "reason": "in_range"}


def estimate_confidence(*, method: str, location_type: str, field: str, value,
                        consistency: float = 1.0) -> tuple:
    """Return (confidence, plausibility_dict, flags). Confidence is a PRODUCT of
    method reliability, source clarity, a plausibility factor, and a cross-field
    consistency factor — a MEASURED estimate to be calibrated, not asserted."""
    flags = []
    mr = METHOD_RELIABILITY.get(method, 0.5)
    sc = SOURCE_CLARITY.get(location_type, 0.6)
    plaus = _plausibility(field, value)
    pf = 1.0
    if not plaus["plausible"]:
        pf = 0.15                         # implausible -> heavy down-weight
        flags.append("implausible_value")
    cf = max(0.2, min(1.0, float(consistency)))
    conf = mr * sc * pf * cf
    return max(0.0, min(1.0, conf)), plaus, flags


def calibrate_confidence(labeled: list, n_bins: int = 10) -> dict:
    """ECE/Brier on a labeled fixture set of (confidence, is_correct) pairs.

    Reuses the service_c reliability-diagram pattern. `labeled` is a list of
    {confidence: float, correct: bool}. Confidence is MEASURED here, not asserted.
    NEVER raises; empty input -> nulls."""
    pairs = [(float(x.get("confidence", 0.0)), bool(x.get("correct", False)))
             for x in (labeled or []) if isinstance(x, dict)]
    if not pairs:
        return {"ece": None, "brier": None, "n": 0, "bins": [],
                "note": "no labeled pairs"}
    N = len(pairs)
    brier = sum((p - (1.0 if c else 0.0)) ** 2 for p, c in pairs) / N
    edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    bins = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b < n_bins - 1:
            members = [(p, c) for p, c in pairs if lo <= p < hi]
        else:
            members = [(p, c) for p, c in pairs if lo <= p <= hi]
        nb = len(members)
        if nb == 0:
            continue
        acc_b = sum(1.0 for _, c in members if c) / nb
        conf_b = sum(p for p, _ in members) / nb
        ece += (nb / N) * abs(acc_b - conf_b)
        bins.append({"lo": lo, "hi": hi, "n": nb, "accuracy": round(acc_b, 4),
                     "confidence": round(conf_b, 4)})
    return {"ece": round(ece, 4), "brier": round(brier, 4), "n": N, "bins": bins,
            "note": "confidence is MEASURED/calibrated, not asserted"}


# ============================ EXTRACTORS =================================== #
def _loc_type_for(section: str, block_type: str) -> str:
    """Map (section, block_type) -> a location_type for source-clarity weighting."""
    if block_type == "table":
        return "table"
    if block_type == "figure_caption":
        return "figure_caption"
    if block_type == "equation":
        return "equation"
    if section == "methods":
        return "methods_prose"
    if section == "results":
        return "results_prose"
    if section == "abstract":
        return "abstract"
    return section if section in SOURCE_CLARITY else "results_prose"


def _find_proteins(text: str) -> list:
    """Longest-match protein gazetteer scan. Returns [(canonical, surface, span)]."""
    low = text.lower()
    hits = []
    # Sort surfaces longest-first so 'abeta42' wins over 'amyloid beta'.
    for surface in sorted(PROTEIN_GAZETTEER, key=len, reverse=True):
        start = 0
        while True:
            i = low.find(surface, start)
            if i < 0:
                break
            span = (i, i + len(surface))
            # skip if overlapping an already-found (longer) hit
            if not any(s < span[1] and span[0] < e for _, _, (s, e) in hits):
                hits.append((PROTEIN_GAZETTEER[surface], text[i:i + len(surface)], span))
            start = i + len(surface)
    return sorted(hits, key=lambda h: h[2][0])


def _norm_conc(value: float, unit: str) -> float:
    factor = _CONC_TO_UM.get(unit.lower().replace("μ", "µ").replace("µ", "µ"), None)
    if factor is None:
        factor = _CONC_TO_UM.get(unit.lower(), 1.0)
    return round(value * factor, 6)


def _mk(field, value, unit, group, method, m, blk, section, doc):
    """Helper: build a claim from a regex match `m` inside block `blk`."""
    span = list(m.span())
    src = blk["text"][span[0]:span[1]]
    loc = _loc_type_for(section, blk["type"])
    conf, plaus, flags = estimate_confidence(method=method, location_type=loc,
                                             field=field, value=value)
    return make_claim(field=field, value=value, unit=unit, doi=doc.get("doi"),
                      pmid=doc.get("pmid"), page=blk.get("_page"),
                      location_type=loc, block_id=blk["block_id"], char_span=span,
                      source_text=src, stage="deterministic_extract",
                      method=method, confidence=conf, group=group,
                      plausibility=plaus, flags=flags)


def extract_from_block(blk: dict, section: str, doc: dict) -> list:
    """Run all deterministic extractors over ONE text block. Returns a list of
    Claims. Table blocks are handled by parse_table instead. NEVER raises."""
    claims = []
    text = blk.get("text") or ""
    if not text.strip():
        return claims
    loc = _loc_type_for(section, blk["type"])

    # -- CONDITIONS ------------------------------------------------------- #
    # protein (gazetteer)
    for canon, surface, span in _find_proteins(text):
        conf, plaus, flags = estimate_confidence(method="gazetteer",
                                                 location_type=loc, field="protein",
                                                 value=canon)
        claims.append(make_claim(field="protein", value=canon, unit=None,
                                 doi=doc.get("doi"), pmid=doc.get("pmid"),
                                 page=blk.get("_page"), location_type=loc,
                                 block_id=blk["block_id"], char_span=list(span),
                                 source_text=surface, stage="deterministic_extract",
                                 method="gazetteer", confidence=conf,
                                 group="conditions", plausibility=plaus, flags=flags))
    # uniprot
    for m in _UNIPROT_RE.finditer(text):
        claims.append(_mk("uniprot", m.group(1), None, "conditions", "regex_exact",
                          m, blk, section, doc))
    # mutation
    for m in _MUTATION_RE.finditer(text):
        claims.append(_mk("mutation", m.group(1), None, "conditions", "regex_exact",
                          m, blk, section, doc))
    # assay + dye (dictionary)
    low = text.lower()
    for surface, (canon, kind) in ASSAY_DYE.items():
        i = low.find(surface)
        if i >= 0:
            span = (i, i + len(surface))
            field = "dye" if kind == "dye" else "assay"
            conf, plaus, flags = estimate_confidence(method="dictionary",
                                                     location_type=loc, field=field,
                                                     value=canon)
            claims.append(make_claim(field=field, value=canon, unit=None,
                                     doi=doc.get("doi"), pmid=doc.get("pmid"),
                                     page=blk.get("_page"), location_type=loc,
                                     block_id=blk["block_id"], char_span=list(span),
                                     source_text=text[i:i + len(surface)],
                                     stage="deterministic_extract", method="dictionary",
                                     confidence=conf, group="conditions",
                                     plausibility=plaus, flags=flags))
            # ThT/ThS also imply an assay claim (fibril-reporting)
            if kind == "dye" and canon in ("ThT", "ThS"):
                claims.append(make_claim(field="assay", value=canon, unit=None,
                                         doi=doc.get("doi"), pmid=doc.get("pmid"),
                                         page=blk.get("_page"), location_type=loc,
                                         block_id=blk["block_id"], char_span=list(span),
                                         source_text=text[i:i + len(surface)],
                                         stage="deterministic_extract",
                                         method="dictionary", confidence=conf,
                                         group="conditions", plausibility=plaus,
                                         flags=flags))
    # temperature
    for m in _TEMP_RE.finditer(text):
        val = float(m.group(1))
        claims.append(_mk("temperature", val, "C", "conditions", "unit_grammar",
                          m, blk, section, doc))
    # pH
    for m in _PH_RE.finditer(text):
        claims.append(_mk("pH", float(m.group(1)), None, "conditions", "unit_grammar",
                          m, blk, section, doc))
    # concentration (unit-normalised to µM; range endpoints -> two claims)
    for m in _CONC_RE.finditer(text):
        lo_v = float(m.group(1))
        hi_v = float(m.group(2)) if m.group(2) else None
        unit = m.group(3)
        span = list(m.span())
        src = text[span[0]:span[1]]
        loc2 = _loc_type_for(section, blk["type"])
        um = _norm_conc(lo_v, unit)
        conf, plaus, flags = estimate_confidence(method="unit_grammar",
                                                 location_type=loc2,
                                                 field="concentration", value=um)
        extra = {"raw": src, "unit_normalized_to": "µM"}
        if hi_v is not None:
            extra["range_uM"] = [um, _norm_conc(hi_v, unit)]
        claims.append(make_claim(field="concentration", value=um, unit="µM",
                                 doi=doc.get("doi"), pmid=doc.get("pmid"),
                                 page=blk.get("_page"), location_type=loc2,
                                 block_id=blk["block_id"], char_span=span,
                                 source_text=src, stage="deterministic_extract",
                                 method="unit_grammar", confidence=conf,
                                 group="conditions", plausibility=plaus,
                                 flags=flags, extra=extra))
    # buffer (dictionary)
    for surface in sorted(BUFFER_TERMS, key=len, reverse=True):
        i = low.find(surface)
        if i >= 0:
            span = (i, i + len(surface))
            conf, plaus, flags = estimate_confidence(method="dictionary",
                                                     location_type=loc, field="buffer",
                                                     value=BUFFER_TERMS[surface])
            claims.append(make_claim(field="buffer", value=BUFFER_TERMS[surface],
                                     unit=None, doi=doc.get("doi"),
                                     pmid=doc.get("pmid"), page=blk.get("_page"),
                                     location_type=loc, block_id=blk["block_id"],
                                     char_span=list(span),
                                     source_text=text[i:i + len(surface)],
                                     stage="deterministic_extract", method="dictionary",
                                     confidence=conf, group="conditions",
                                     plausibility=plaus, flags=flags))
            break            # first (longest) buffer term is enough
    # reducing agent
    for m in _REDUCING_AGENT_RE.finditer(text):
        claims.append(_mk("reducing_agent", m.group(1), None, "conditions",
                          "regex_exact", m, blk, section, doc))
    # ionic strength
    for m in _IONIC_RE.finditer(text):
        salt, val, unit = m.group(1), float(m.group(2)), m.group(3)
        span = list(m.span())
        loc2 = _loc_type_for(section, blk["type"])
        conf, plaus, flags = estimate_confidence(method="unit_grammar",
                                                 location_type=loc2,
                                                 field="ionic_strength", value=val)
        claims.append(make_claim(field="ionic_strength",
                                 value={"salt": salt, "value": val, "unit": unit},
                                 unit=unit, doi=doc.get("doi"), pmid=doc.get("pmid"),
                                 page=blk.get("_page"), location_type=loc2,
                                 block_id=blk["block_id"], char_span=span,
                                 source_text=text[span[0]:span[1]],
                                 stage="deterministic_extract", method="unit_grammar",
                                 confidence=conf, group="conditions",
                                 plausibility=plaus, flags=flags))
    # agitation (grammar; quiescent vs moved)
    mq = _AGIT_QUIESCENT_RE.search(text)
    mm = _AGIT_MOVED_RE.search(text)
    if mq:
        claims.append(_agit_seed_claim("agitation", "quiescent", mq, blk, section, doc))
    elif mm:
        claims.append(_agit_seed_claim("agitation", "agitated", mm, blk, section, doc))
    # seeding (grammar; seeded vs unseeded — check 'unseeded' first)
    ms_no = _SEED_NO_RE.search(text)
    ms_yes = _SEED_YES_RE.search(text)
    if ms_no:
        claims.append(_agit_seed_claim("seeding", "unseeded", ms_no, blk, section, doc))
    elif ms_yes:
        claims.append(_agit_seed_claim("seeding", "seeded", ms_yes, blk, section, doc))

    # -- REPORTED RESULTS (compared vs PRISE's own computed values later) --- #
    for m in _GAMMA_RE.finditer(text):
        claims.append(_mk("gamma", float(m.group(1)), None, "reported_results",
                          "regex_exact", m, blk, section, doc))
    for m in _HALFTIME_RE.finditer(text):
        unit = (m.group(2) or "").lower() or None
        claims.append(_mk("half_time", float(m.group(1)),
                          _norm_time_unit(unit), "reported_results",
                          "regex_exact", m, blk, section, doc))
    for m in _LAG_RE.finditer(text):
        unit = (m.group(2) or "").lower() or None
        claims.append(_mk("lag_time", float(m.group(1)),
                          _norm_time_unit(unit), "reported_results",
                          "regex_exact", m, blk, section, doc))
    for m in _UNCERT_RE.finditer(text):
        claims.append(_mk("reported_uncertainty",
                          {"value": float(m.group(1)), "pm": float(m.group(2))},
                          None, "reported_results", "regex_exact", m, blk, section, doc))

    # -- REPORTED SEMANTIC CLAIMS (lowest confidence, narrative) ----------- #
    for surface, canon in MECHANISM_TERMS.items():
        i = low.find(surface)
        if i >= 0:
            claims.append(_semantic_claim("mechanism", canon, surface, i, blk,
                                          section, doc))
    for surface, canon in MODEL_TERMS.items():
        i = low.find(surface)
        if i >= 0:
            claims.append(_semantic_claim("model", canon, surface, i, blk,
                                          section, doc))
    if blk["type"] == "equation":
        claims.append(_semantic_claim("equation", text.strip(), text.strip(), 0,
                                      blk, section, doc, method="regex_exact"))
    return claims


# ---- OPT-IN ML-NER augmentation (only when the front-end is live) --------- #
# Entity label -> PRISE ontology field. DELIBERATELY NARROW: a label with no
# honest ontology target (SIMPLE_CHEMICAL, CELL, ORGANISM, ...) is DROPPED rather
# than force-fitted into `buffer`/`dye`, which would manufacture a condition the
# paper never stated. A deployment whose model emits richer labels extends this
# map explicitly — the mapping is a curation decision, not an inference.
NER_LABEL_TO_FIELD = {
    "PROTEIN": "protein",
    "GENE_OR_GENE_PRODUCT": "protein",
    "AMINO_ACID_SEQUENCE": "construct",
}


def extract_ner_claims(blk: dict, section: str, doc: dict) -> list:
    """OPT-IN: run the ML-NER front-end over ONE block and turn its entities into
    Claims stamped extractor.method='ml_ner' (reliability 0.50 — BELOW every
    deterministic method; an NER hit is never promoted to regex_exact).

    Honesty rules enforced here:
      * source_text is sliced from the BLOCK text using the entity's own char
        offsets — verbatim by construction, never the model's echo of the text;
      * an entity without usable offsets is DROPPED (PRISE requires a resolvable
        span; a claim without provenance is not a claim);
      * an entity whose label has no honest ontology target is DROPPED, not
        force-fitted (see NER_LABEL_TO_FIELD);
      * the surface form is NOT canonicalised through the gazetteer — that would
        launder a model guess into a dictionary-tier result.

    Raises NotImplementedError (naming the missing dep) if no ML-NER front-end is
    available — this is an EXPLICIT request for a capability, so its absence is
    reported rather than silently skipped."""
    text = blk.get("text") or ""
    if not text.strip():
        return []
    entities = ner_extract(text)                 # raises when unavailable — by design
    claims = []
    loc = _loc_type_for(section, blk.get("type") or "paragraph")
    for ent in (entities or []):
        if not isinstance(ent, dict):
            continue
        field = NER_LABEL_TO_FIELD.get(str(ent.get("label") or "").upper())
        if field is None:
            continue                             # no honest target -> dropped
        start, end = ent.get("start"), ent.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue                             # no span -> no provenance -> dropped
        if not (0 <= start < end <= len(text)):
            continue
        value = text[start:end]                  # VERBATIM slice of the block text
        conf, plaus, flags = estimate_confidence(method="ml_ner", location_type=loc,
                                                 field=field, value=value)
        claims.append(make_claim(
            field=field, value=value, unit=None, doi=doc.get("doi"),
            pmid=doc.get("pmid"), page=blk.get("_page"), location_type=loc,
            block_id=blk.get("block_id"), char_span=[start, end], source_text=value,
            stage="ml_ner_augment", method="ml_ner", confidence=conf,
            group="conditions", plausibility=plaus,
            flags=list(flags) + ["ml_ner_probabilistic_not_deterministic",
                                 "surface_form_not_canonicalised"],
            extra={"ner": {"label": ent.get("label"),
                           "backend": ent.get("backend"),
                           "model_score": ent.get("score")}}))
    return claims


def _norm_time_unit(unit):
    if unit is None:
        return "h"          # aggregation half-times default to hours in the corpus
    if unit.startswith("min"):
        return "min"
    if unit.startswith("s"):
        return "s"
    return "h"


def _agit_seed_claim(field, value, m, blk, section, doc):
    span = list(m.span())
    loc = _loc_type_for(section, blk["type"])
    conf, plaus, flags = estimate_confidence(method="dictionary", location_type=loc,
                                             field=field, value=value)
    return make_claim(field=field, value=value, unit=None, doi=doc.get("doi"),
                      pmid=doc.get("pmid"), page=blk.get("_page"), location_type=loc,
                      block_id=blk["block_id"], char_span=span,
                      source_text=blk["text"][span[0]:span[1]],
                      stage="deterministic_extract", method="dictionary",
                      confidence=conf, group="conditions", plausibility=plaus,
                      flags=flags)


def _semantic_claim(field, value, surface, i, blk, section, doc, method="dictionary"):
    """Semantic (narrative) claim — lowest confidence, flagged narrative. WHY: a
    stated mechanism/model is the author's INTERPRETATION, not a measurement."""
    span = [i, i + len(surface)] if method == "dictionary" else [0, len(surface)]
    loc = _loc_type_for(section, blk["type"])
    conf, plaus, flags = estimate_confidence(method=method, location_type=loc,
                                             field=field, value=value,
                                             consistency=0.7)  # narrative discount
    flags = list(flags) + ["narrative_interpretation_not_measurement"]
    return make_claim(field=field, value=value, unit=None, doi=doc.get("doi"),
                      pmid=doc.get("pmid"), page=blk.get("_page"), location_type=loc,
                      block_id=blk["block_id"], char_span=span,
                      source_text=blk["text"][span[0]:span[1]] if method == "dictionary"
                      else blk["text"].strip(),
                      stage="deterministic_extract", method=method,
                      confidence=conf, group="semantic", plausibility=plaus,
                      flags=flags)


# ============================ TABLE PARSER ================================ #
# Header-token -> ontology field. Case-insensitive substring match on the header.
_TABLE_HEADER_MAP = [
    ("protein", "protein"), ("uniprot", "uniprot"), ("mutation", "mutation"),
    ("variant", "mutation"), ("construct", "construct"),
    ("conc", "concentration"), ("[m]", "concentration"), ("µm", "concentration"),
    ("um", "concentration"), ("temp", "temperature"), ("ph", "pH"),
    ("assay", "assay"), ("dye", "dye"), ("buffer", "buffer"),
    ("salt", "ionic_strength"), ("nacl", "ionic_strength"),
    ("ionic", "ionic_strength"), ("agitat", "agitation"), ("shak", "agitation"),
    ("seed", "seeding"), ("t50", "half_time"), ("t1/2", "half_time"),
    ("half-time", "half_time"), ("lag", "lag_time"), ("gamma", "gamma"),
    ("γ", "gamma"),
]


def _map_header(col: str):
    """Map a table header cell -> ontology field (longest, most-specific first)."""
    low = (col or "").strip().lower()
    for token, field in sorted(_TABLE_HEADER_MAP, key=lambda kv: -len(kv[0])):
        if token in low:
            return field
    return None


def _coerce_cell(field, cell):
    """Coerce a table cell string to a typed value + unit for the mapped field."""
    s = str(cell).strip()
    if field == "concentration":
        m = _CONC_RE.search(s) or re.search(r"(\d+(?:\.\d+)?)", s)
        if m:
            if m.re is _CONC_RE:
                return _norm_conc(float(m.group(1)), m.group(3)), "µM"
            return float(m.group(1)), "µM"   # bare number -> assume header unit µM
        return s, None
    if field in ("temperature", "pH", "gamma", "half_time", "lag_time"):
        m = re.search(r"(-?\d+(?:\.\d+)?)", s)
        if m:
            unit = {"temperature": "C", "half_time": "h", "lag_time": "h"}.get(field)
            return float(m.group(1)), unit
        return s, None
    if field == "seeding":
        return ("seeded" if re.search(r"seed|yes|\+", s, re.I)
                and not re.search(r"unseed|no\b|-", s, re.I) else
                "unseeded" if re.search(r"unseed|no\b|-", s, re.I) else s), None
    if field == "agitation":
        return ("quiescent" if re.search(r"quiesc|no|-", s, re.I)
                else "agitated" if re.search(r"shak|stir|orbit|rpm|yes|\+", s, re.I)
                else s), None
    return s, None


def parse_table(blk: dict, section: str, doc: dict) -> list:
    """Parse a table block into per-row condition claims (§3-M15). header row ->
    field mapping -> one claim per mapped cell, with table char-span provenance.
    Tables get the highest source-clarity. NEVER raises."""
    claims = []
    header = blk.get("header")
    rows = blk.get("rows") or []
    if header is None and rows:
        header, rows = rows[0], rows[1:]     # first row is the header
    if not header:
        return claims
    field_by_col = {ci: _map_header(h) for ci, h in enumerate(header)}
    for ri, row in enumerate(rows):
        if not isinstance(row, (list, tuple)):
            continue
        for ci, cell in enumerate(row):
            field = field_by_col.get(ci)
            if field is None or cell in (None, "", "-"):
                continue
            value, unit = _coerce_cell(field, cell)
            group = ("reported_results" if field in RESULT_FIELDS else "conditions")
            conf, plaus, flags = estimate_confidence(method="table_cell",
                                                     location_type="table",
                                                     field=field, value=value)
            claims.append(make_claim(
                field=field, value=value, unit=unit, doi=doc.get("doi"),
                pmid=doc.get("pmid"), page=blk.get("_page"), location_type="table",
                block_id=blk["block_id"],
                char_span=None,       # table provenance is (row,col), see extra
                source_text=str(cell), stage="deterministic_extract",
                method="table_cell", confidence=conf, group=group,
                plausibility=plaus, flags=flags,
                extra={"table_cell": {"row": ri, "col": ci,
                                      "header": header[ci] if ci < len(header) else None}}))
    return claims


def parse_caption(blk: dict, section: str, doc: dict) -> list:
    """Parse a figure caption into condition/result claims. Same extractors as
    prose, but with figure_caption location_type (lower source clarity)."""
    return extract_from_block(blk, section, doc)


# ============================ DOCUMENT-LEVEL PIPELINE ===================== #
def extract_document(document: dict, ledger: ClaimLedger = None,
                     use_ner: bool = False) -> ClaimLedger:
    """Run the full deterministic extraction over a Document, appending every
    Claim to the (append-only) ledger. NEVER raises.

    `use_ner=True` ADDITIONALLY runs the ML-NER front-end over every text block and
    appends its claims at the lower 'ml_ner' reliability tier (the deterministic
    claims are produced identically either way — NER augments, never replaces).
    It is the ONE argument that can make this function raise: asking for a
    capability that is not installed gets an honest NotImplementedError, not a
    silent skip. Default False keeps the pipeline dependency-free."""
    if ledger is None:
        ledger = ClaimLedger()
    if not isinstance(document, dict):
        return ledger
    sections = detect_sections(document)
    for page in (document.get("pages") or []):
        if not isinstance(page, dict):
            continue
        pno = page.get("page_no")
        for bi, blk in enumerate(page.get("blocks") or []):
            if not isinstance(blk, dict):
                continue
            blk = dict(blk)
            blk.setdefault("block_id", f"p{pno}-b{bi}")
            blk.setdefault("type", "paragraph")
            blk["_page"] = pno
            section = sections.get(blk.get("block_id"), "results_prose")
            if blk.get("type") == "table":
                blk_claims = parse_table(blk, section, document)
            elif blk.get("type") == "figure_caption":
                blk_claims = parse_caption(blk, section, document)
            else:
                blk_claims = extract_from_block(blk, section, document)
            # AUGMENT (never replace): NER claims are appended alongside the
            # deterministic ones, at their own lower 'ml_ner' reliability tier.
            if use_ner and blk.get("type") != "table":
                blk_claims = list(blk_claims) + extract_ner_claims(blk, section,
                                                                   document)
            for c in blk_claims:
                ledger.append_claim(c)
    return ledger


# ============================ CONFLICT DETECTION ========================== #
def _condition_context(claim: dict, doc_conditions: dict) -> tuple:
    """A coarse condition-context key so conflicts group like-with-like. WHY: two
    papers reporting different γ under DIFFERENT pH aren't in conflict."""
    return (doc_conditions.get("protein"),
            doc_conditions.get("pH"), doc_conditions.get("temperature"))


def _numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def detect_conflicts(claims: list, doc_conditions_by_pmid: dict = None,
                     rel_tol: float = 0.15) -> list:
    """Group active claims by (protein, field, condition-context); DIVERGENT
    numeric values across DIFFERENT papers -> a conflict record carrying BOTH
    claims' full provenance. Never auto-resolves. NEVER raises.

    `doc_conditions_by_pmid[pmid]` supplies the per-paper protein/pH/temp context
    (so conflicts are like-for-like). Result fields (gamma/half_time/lag_time)
    are the primary conflict targets."""
    doc_conditions_by_pmid = doc_conditions_by_pmid or {}
    conflict_fields = ("gamma", "half_time", "lag_time")
    groups = {}
    for c in claims:
        if c.get("field") not in conflict_fields:
            continue
        if c.get("supersedes") is not None:
            continue
        ctx = doc_conditions_by_pmid.get(c.get("pmid"), {})
        key = (ctx.get("protein"), c["field"], ctx.get("pH"), ctx.get("temperature"))
        groups.setdefault(key, []).append(c)
    conflicts = []
    for key, members in groups.items():
        # Only cross-PAPER divergence is a conflict (same paper = same value ok).
        by_val = {}
        for c in members:
            v = _numeric(c.get("value"))
            if v is None:
                continue
            by_val.setdefault(round(v, 6), []).append(c)
        if len(by_val) < 2:
            continue
        vals = sorted(by_val)
        pmids = {c.get("pmid") for c in members}
        if len(pmids) < 2:
            continue          # divergence within one paper is not a cross-source conflict
        lo, hi = vals[0], vals[-1]
        denom = max(abs(lo), abs(hi), 1e-9)
        if (hi - lo) / denom <= rel_tol:
            continue          # within tolerance -> agreement, not conflict
        protein, field, pH, temp = key
        conflicts.append({
            "conflict_id": "cfl_" + hashlib.sha256(
                json.dumps([protein, field, pH, temp, vals], default=str)
                .encode()).hexdigest()[:16],
            "protein": protein, "field": field,
            "condition_context": {"pH": pH, "temperature": temp},
            "values": vals, "rel_spread": round((hi - lo) / denom, 4),
            "resolution": "UNRESOLVED — surfaced, never auto-resolved",
            "claims": [{"claim_id": c["claim_id"], "pmid": c.get("pmid"),
                        "doi": c.get("doi"), "value": c.get("value"),
                        "unit": c.get("unit"), "page": c.get("page"),
                        "block_id": c.get("block_id"),
                        "location_type": c.get("location_type"),
                        "source_text": c.get("source_text"),
                        "confidence": c.get("confidence")}
                       for c in members],
            "honesty": {"never_auto_resolved": True,
                        "both_provenances_retained": True},
        })
    return conflicts


def conflicts_vs_computed(claims: list, computed: dict, rel_tol: float = 0.15) -> list:
    """Compare EXTRACTED reported-result claims (gamma/half_time) against PRISE's
    OWN computed values. A divergence beyond tolerance is a conflict between the
    LITERATURE CLAIM and the ENGINE — surfaced, never auto-resolved.

    `computed[pmid][field] = value`. NEVER raises."""
    out = []
    for c in claims:
        field = c.get("field")
        if field not in ("gamma", "half_time", "lag_time"):
            continue
        if c.get("supersedes") is not None:
            continue
        comp = (computed.get(c.get("pmid")) or {}).get(field)
        cv = _numeric(c.get("value"))
        if comp is None or cv is None:
            continue
        denom = max(abs(comp), abs(cv), 1e-9)
        if abs(comp - cv) / denom <= rel_tol:
            continue
        out.append({
            "conflict_id": "cvc_" + hashlib.sha256(
                json.dumps([c["claim_id"], field, comp], default=str).encode())
                .hexdigest()[:16],
            "kind": "literature_vs_computed",
            "field": field, "pmid": c.get("pmid"),
            "reported_claim": {"claim_id": c["claim_id"], "value": c.get("value"),
                               "source_text": c.get("source_text"),
                               "location_type": c.get("location_type"),
                               "confidence": c.get("confidence")},
            "computed_value": comp,
            "rel_diff": round(abs(comp - cv) / denom, 4),
            "resolution": "UNRESOLVED — literature CLAIM vs engine value, surfaced only",
        })
    return out


# ============================ CPAD VALIDATION HARNESS ==================== #
# Map an M15 field -> how to read the same quantity from a CPAD condition_vector.
def _cpad_field_value(field: str, cv: dict, curve: dict):
    if field == "protein":
        return curve.get("protein_id")
    if field == "uniprot":
        return curve.get("uniprot_id")
    if field == "concentration":
        c = cv.get("concentration") or {}
        return c.get("value_uM") if isinstance(c, dict) else None
    if field == "temperature":
        return cv.get("temperature_C")
    if field == "pH":
        return cv.get("pH")
    if field == "assay":
        return cv.get("assay_type")
    if field == "buffer":
        b = cv.get("buffer")
        return b
    if field == "mutation" or field == "construct":
        return cv.get("construct_id")
    return None


# Chemical aliases so a canonical buffer name matches CPAD's verbose formula
# (e.g. extractor 'phosphate' vs curated 'NaH2PO4 50 mM'). Offline, deterministic.
_BUFFER_ALIASES = {
    "phosphate": ("phosphate", "nah2po4", "na2hpo4", "kh2po4", "k2hpo4", "po4",
                  "pbs"),
    "tris": ("tris",), "hepes": ("hepes",), "glycine": ("glycine",),
    "acetate": ("acetate", "ch3coo"), "citrate": ("citrate",),
}


def _values_match(field: str, extracted, curated) -> bool:
    """Field-aware equality: numeric within tolerance; strings by canonical/substr.
    Buffers use a small chemical-alias table (canonical name <-> CPAD formula)."""
    if extracted is None or curated is None:
        return False
    if field in ("concentration", "temperature", "pH"):
        ev, cvv = _numeric(extracted), _numeric(curated)
        if ev is None or cvv is None:
            return False
        return abs(ev - cvv) <= max(0.05 * abs(cvv), 0.15)
    if field == "assay":
        return str(extracted).lower() == str(curated).lower()
    if field == "buffer":
        e, c = str(extracted).lower(), str(curated).lower()
        aliases = _BUFFER_ALIASES.get(e, (e,))
        return any(a in c for a in aliases)
    # protein / construct: canonical-substring match (curated names are verbose)
    e, c = str(extracted).lower(), str(curated).lower()
    return e in c or c in e or any(tok in c for tok in e.split() if len(tok) > 3)


def validate_against_cpad(document: dict, pmid: str, cpad_curves: list,
                          fields: tuple = None) -> dict:
    """Extract conditions from a fixture paper tagged with a REAL CPAD pmid and
    compare to CPAD's curated condition_vector for curves with that pmid ->
    field-level precision / recall / accuracy + confidence calibration (§3-M15).

    This MEASURES extraction accuracy against the curated gold standard — it does
    NOT assert it. Runs on fixtures mirroring a real pmid; real-paper validation
    is gated on PDFs. NEVER raises."""
    fields = fields or ("protein", "concentration", "temperature", "pH", "assay",
                        "buffer")
    curated = [c for c in (cpad_curves or [])
               if str((c.get("source_study") or {}).get("pmid")) == str(pmid)]
    # Curated gold values per field (a set of allowed values across the pmid's curves).
    gold = {f: set() for f in fields}
    for cv_curve in curated:
        cv = cv_curve.get("condition_vector") or {}
        for f in fields:
            val = _cpad_field_value(f, cv, cv_curve)
            if val is not None:
                gold[f].add(str(val))

    ledger = extract_document(document)
    extracted = ledger.active_claims()
    # Per field: best extracted value (highest confidence).
    ext_by_field = {}
    labeled = []       # (confidence, correct) for calibration
    for c in extracted:
        f = c.get("field")
        if f not in fields:
            continue
        prev = ext_by_field.get(f)
        if prev is None or c["confidence"] > prev["confidence"]:
            ext_by_field[f] = c

    per_field = {}
    tp = fp = fn = 0
    for f in fields:
        ext = ext_by_field.get(f)
        gold_vals = gold[f]
        if ext is None and not gold_vals:
            per_field[f] = {"status": "both_absent", "match": None}
            continue
        if ext is None:
            fn += 1
            per_field[f] = {"status": "missed", "curated": sorted(gold_vals),
                            "extracted": None, "match": False}
            continue
        matched = any(_values_match(f, ext["value"], g) for g in gold_vals) \
            if gold_vals else False
        labeled.append({"confidence": ext["confidence"], "correct": matched})
        if matched:
            tp += 1
        elif gold_vals:
            fp += 1      # extracted but wrong vs curated
        else:
            fp += 1      # extracted but nothing curated to confirm
        per_field[f] = {"status": "match" if matched else "mismatch",
                        "extracted": ext["value"], "curated": sorted(gold_vals),
                        "confidence": ext["confidence"], "match": matched,
                        "source_text": ext.get("source_text"),
                        "location_type": ext.get("location_type")}

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    n_eval = tp + fp + fn
    accuracy = tp / n_eval if n_eval else None
    return {
        "version": M15_VERSION, "pmid": str(pmid),
        "n_curated_curves": len(curated),
        "fields_evaluated": list(fields),
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "counts": {"tp": tp, "fp": fp, "fn": fn},
        "per_field": per_field,
        "confidence_calibration": calibrate_confidence(labeled),
        "honesty": {
            "accuracy_is_measured_not_asserted": True,
            "gold_standard": "CPAD curated condition_vector for this pmid",
            "runs_on_fixture_mirroring_real_pmid": True,
            "real_paper_validation_gated_on": (
                "a PDF corpus + a LIVE pdf_to_document front-end; currently "
                + ("LIVE" if frontend_status("pdf_to_document")["available"]
                   else "UNAVAILABLE (dependency not installed)"))},
    }


# ============================ FIXTURE DOCUMENTS =========================== #
# 3 realistic FIXTURE documents mimicking real aggregation papers. Fixture 1 is
# built to MIRROR real CPAD pmid 18258258 (Kim & Hecht, Aβ40) for the round-trip.
def fixture_documents() -> list:
    """Three synthetic fixture Documents (what a born-digital parser would emit).

    * FIX-1: mirrors real CPAD pmid 18258258 (Aβ40; 20 µM, pH 7.35→7.4, 37°C, ThT,
      phosphate, NaCl 100 mM) — used for the CPAD round-trip validation.
    * FIX-2: an Aβ42 paper reporting γ=1.2 (conflicts with FIX-3's γ for Aβ42).
    * FIX-3: an Aβ42 paper reporting γ=0.5 under the same protein context.
    """
    fix1 = {
        "doi": "10.1016/j.jmb.2008.01.041", "pmid": "18258258",
        "title": "Sequence determinants of Abeta40 amyloid formation",
        "pages": [{"page_no": 3, "blocks": [
            {"block_id": "f1-methods-h", "type": "heading",
             "text": "Materials and Methods"},
            {"block_id": "f1-methods-1", "type": "paragraph",
             "text": ("Amyloid beta (Abeta40, UniProt P05067) was dissolved to a final "
                      "concentration of 20 µM in 50 mM sodium phosphate buffer "
                      "containing 100 mM NaCl at pH 7.4. Aggregation was monitored by "
                      "Thioflavin T fluorescence at 37 C under quiescent conditions in "
                      "unseeded reactions.")},
            {"block_id": "f1-table", "type": "table", "section": "methods",
             "header": ["Protein", "Conc (µM)", "Temp", "pH", "Assay", "Buffer"],
             "rows": [["Abeta40", "20", "37 C", "7.4", "ThT", "phosphate"]]},
            {"block_id": "f1-results-h", "type": "heading", "text": "Results"},
            {"block_id": "f1-fig3", "type": "figure_caption",
             "text": ("Fig 3. ThT kinetics of Abeta40 at 10-70 µM, pH 7.4, 37°C, "
                      "quiescent. The half-time was 24 h.")},
        ]}],
    }
    fix2 = {
        "doi": "10.1073/pnas.aaaa", "pmid": "20000001",
        "title": "Secondary nucleation dominates Abeta42 aggregation",
        "pages": [{"page_no": 1, "blocks": [
            {"block_id": "f2-abs", "type": "heading", "text": "Abstract"},
            {"block_id": "f2-abs-1", "type": "paragraph",
             "text": ("Aggregation of Abeta42 proceeds via a secondary nucleation "
                      "mechanism described by the Knowles integrated rate law.")},
            {"block_id": "f2-methods-h", "type": "heading",
             "text": "Materials and Methods"},
            {"block_id": "f2-methods-1", "type": "paragraph",
             "text": ("Abeta42 was prepared at 5 µM in 20 mM Tris buffer at pH 7.4 and "
                      "aggregation followed by ThT fluorescence at 37 C with shaking "
                      "(300 rpm). Reactions were seeded with 1% preformed seeds. "
                      "The A2T mutation was included as a control.")},
            {"block_id": "f2-results-h", "type": "heading", "text": "Results"},
            {"block_id": "f2-res-1", "type": "paragraph",
             "text": ("The scaling exponent gamma was 1.2 for Abeta42, consistent with "
                      "secondary nucleation. A sigmoidal model gave t50 = 3.5 ± 0.2 h.")},
        ]}],
    }
    fix3 = {
        "doi": "10.1021/bbbb", "pmid": "20000002",
        "title": "Fragmentation-dominated Abeta42 kinetics",
        "pages": [{"page_no": 1, "blocks": [
            {"block_id": "f3-methods-h", "type": "heading",
             "text": "Experimental Procedures"},
            {"block_id": "f3-methods-1", "type": "paragraph",
             "text": ("Abeta42 was studied at 5 µM in 20 mM Tris, pH 7.4, at 37 C "
                      "under stirring, monitored by Thioflavin T.")},
            {"block_id": "f3-results-h", "type": "heading", "text": "Results"},
            {"block_id": "f3-res-1", "type": "paragraph",
             "text": ("Here the scaling exponent gamma was 0.5, indicating a "
                      "fragmentation-dominated regime. The half-time was 8 h.")},
        ]}],
    }
    return [document_from_fixture(s) for s in (fix1, fix2, fix3)]


def _fixture_conditions_by_pmid() -> dict:
    """The per-paper protein/pH/temperature context for conflict grouping (the
    fixtures share Aβ42 at pH 7.4 / 37°C so FIX-2 vs FIX-3 γ genuinely conflict)."""
    return {
        "18258258": {"protein": "Amyloid Beta peptide-ABeta40", "pH": 7.4,
                     "temperature": 37.0},
        "20000001": {"protein": "Amyloid Beta peptide-ABeta42", "pH": 7.4,
                     "temperature": 37.0},
        "20000002": {"protein": "Amyloid Beta peptide-ABeta42", "pH": 7.4,
                     "temperature": 37.0},
    }


# ============================ DRIVER / CLI ================================ #
def run_pipeline(cpad_curves: list = None) -> dict:
    """Run the full deterministic pipeline over the fixture documents and return
    {ledger, conflicts, validation, summary}. NEVER raises."""
    docs = fixture_documents()
    ledger = ClaimLedger()
    for d in docs:
        extract_document(d, ledger)
    active = ledger.active_claims()
    conflicts = detect_conflicts(active, _fixture_conditions_by_pmid())
    # CPAD round-trip on FIX-1 (mirrors real pmid 18258258).
    validation = None
    if cpad_curves is not None:
        validation = validate_against_cpad(docs[0], "18258258", cpad_curves)

    # Field coverage.
    coverage = {}
    for c in active:
        coverage.setdefault(c["field"], 0)
        coverage[c["field"]] += 1
    summary = {
        "n_documents": len(docs),
        "n_claims_total": len(ledger.claims),
        "n_claims_active": len(active),
        "field_coverage": dict(sorted(coverage.items())),
        "n_conflicts": len(conflicts),
        "groups": {g: sum(1 for c in active if c["group"] == g)
                   for g in ("conditions", "reported_results", "semantic")},
    }
    return {"ledger": ledger, "conflicts": conflicts, "validation": validation,
            "summary": summary, "honesty": honesty_block()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PRISE M15 — Literature Ingestion & "
                                             "Claim Extraction (runs on fixtures).")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--outdir", default=str(root / "data" / "processed"))
    ap.add_argument("--cpad", default=str(root / "data" / "processed" /
                                          "curves_triaged.jsonl"))
    args = ap.parse_args(argv)

    cpad_curves = []
    cpad_path = Path(args.cpad)
    if cpad_path.exists():
        with open(cpad_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        cpad_curves.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    result = run_pipeline(cpad_curves=cpad_curves or None)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    claims_path = result["ledger"].persist(outdir / "lit_claims.jsonl")
    with open(outdir / "lit_conflicts.json", "w", encoding="utf-8") as fh:
        json.dump({"version": M15_VERSION, "conflicts": result["conflicts"],
                   "honesty": honesty_block()}, fh, indent=2, default=str)
    with open(outdir / "lit_validation.json", "w", encoding="utf-8") as fh:
        json.dump({"version": M15_VERSION, "validation": result["validation"],
                   "honesty": honesty_block()}, fh, indent=2, default=str)

    s = result["summary"]
    print(f"[M15] {M15_VERSION} — CLAIMS not facts; append-only; fully traceable; "
          f"accuracy MEASURED vs CPAD.")
    print(f"[M15] wrote {claims_path}")
    print(f"[M15] wrote {outdir / 'lit_conflicts.json'}")
    print(f"[M15] wrote {outdir / 'lit_validation.json'}")
    print(f"\n  Documents processed (synthetic fixtures): {s['n_documents']}")
    print(f"  Claims in ledger (append-only): {s['n_claims_total']} "
          f"({s['n_claims_active']} active)")
    print(f"  Claim groups: {s['groups']}")
    print(f"  Field coverage: {s['field_coverage']}")

    # Example fully-provenanced claim.
    example = next((c for c in result["ledger"].active_claims()
                    if c["field"] == "concentration"), None)
    if example:
        print("\n  EXAMPLE fully-provenanced claim (a proposal, not a fact):")
        print(f"    field={example['field']} value={example['value']} "
              f"unit={example['unit']}")
        print(f"    doi={example['doi']} pmid={example['pmid']} "
              f"page={example['page']} location={example['location_type']} "
              f"block={example['block_id']} span={example['char_span']}")
        print(f"    source_text={example['source_text']!r}")
        print(f"    method={example['extractor']['method']} "
              f"confidence={example['confidence']} plausible="
              f"{example['plausibility']['plausible']}")

    # A detected conflict.
    if result["conflicts"]:
        cf = result["conflicts"][0]
        print(f"\n  CONFLICT (surfaced, never auto-resolved): {cf['field']} for "
              f"{cf['protein']} -> values {cf['values']} "
              f"(rel_spread {cf['rel_spread']})")
        for cl in cf["claims"]:
            print(f"    pmid={cl['pmid']} value={cl['value']} "
                  f"src={cl['source_text']!r}")

    if result["validation"]:
        v = result["validation"]
        print(f"\n  CPAD round-trip (pmid {v['pmid']}, {v['n_curated_curves']} "
              f"curated curves): precision={v['precision']} recall={v['recall']} "
              f"accuracy={v['accuracy']} counts={v['counts']}")
        print(f"    confidence calibration: ECE={v['confidence_calibration']['ece']} "
              f"Brier={v['confidence_calibration']['brier']}")

    print("\n  HONESTY: M15 PROPOSES claims; a human curator approves; only high-"
          "confidence,\n  non-conflicting claims graduate into the corpus. Nothing "
          "auto-trusted.")
    caps = frontend_capabilities()
    print("  FRONT-ENDS (detected now — PDF-parse / OCR / ML-NER / figure-curve):")
    for name in FRONTEND_NAMES:
        st = caps[name]
        if st["available"]:
            print(f"    {name}: LIVE via {st['source']} ({st['backend']})")
        else:
            print(f"    {name}: UNAVAILABLE — needs {st['dependency']}; calling it "
                  f"raises NotImplementedError, nothing is faked")
    print("  (UNAVAILABLE means the dependency is not installed here — PRISE stays "
          "stdlib+numpy/scipy;\n   it is NOT a claim about network access. Inject one "
          "with register_frontend(name, fn).)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
