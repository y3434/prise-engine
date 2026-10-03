#!/usr/bin/env python3
"""PRISE web app — stdlib-only backend.

A zero-install local web server over the completed PRISE protein-aggregation
analysis engine. Built on http.server.ThreadingHTTPServer + a tiny path router
(NO fastapi / flask / uvicorn). Loads the precomputed M1..M9 artifacts from
../data/processed at startup, caches them in memory, builds indices, and serves
DERIVED analytical results + single curves (with CPAD attribution).

Data residency (enforced): CPAD bulk is attribution-gated. We serve only derived
results + SINGLE curves (one series at a time). There is deliberately NO
bulk-dump / "download all curves" endpoint.

Launch:  python web/server.py            # serves http://127.0.0.1:8000
         python web/server.py 8123       # override port (arg)
         PRISE_PORT=8123 python web/server.py
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "processed")
STATIC = os.path.join(HERE, "static")
ENGINE = os.path.join(ROOT, "engine")

# Make the engine importable (the web app only IMPORTS engine functions).
if ENGINE not in sys.path:
    sys.path.insert(0, ENGINE)

# Engine imports are optional: the dashboard / browse / curve endpoints work on
# precomputed artifacts alone. /api/analyze and the fitted-overlay need the
# engine; if it can't import (e.g. scipy missing) we degrade gracefully.
ENGINE_OK = True
ENGINE_ERR = None
try:
    import m0_cohort as _m0
    import m1_ingest as _m1
    import m2_fit as _m2
    import m3_select as _m3
    import m4_features as _m4
    import m5_classify as _m5
    import mechanistic as _mech
    import models as _models
except Exception as exc:  # pragma: no cover - defensive
    ENGINE_OK = False
    ENGINE_ERR = f"{type(exc).__name__}: {exc}"
    _m0 = _m1 = _m2 = _m3 = _m4 = _m5 = _mech = _models = None

# conformal is imported separately: its pure test-time application functions
# (conformal_t50_interval / conformal_regime_set) need only the frozen JSON, so a
# host without scipy can still apply a pre-built calibration. Import is best-effort.
try:
    import conformal as _conformal
except Exception:  # pragma: no cover - defensive
    _conformal = None


# --------------------------------------------------------------------------- #
# In-memory cache of artifacts + indices (built once at startup)
# --------------------------------------------------------------------------- #
class LazyRecordIndex:
    """A dict-like view over a large per-series JSONL that holds only offsets.

    MEASURED, against a 512 MB container ceiling: held as parsed dicts,
    metadata_quality.jsonl (17.7 MB on disk) costs 59.6 MB resident,
    information_content.jsonl (13.3 MB) costs 36.2 MB and
    boed_recommendations.jsonl (17.1 MB) costs 41.8 MB -- 137.6 MB between
    them, on a process whose warm peak was 376 MB. Every reader of these three
    is a SINGLE-record lookup by series_id (/api/metadata, /api/information,
    /api/boed), so nothing needs them all resident. This keeps
    {series_id: (offset, nbytes)} and parses one line on demand.

    Deliberately dict-shaped (get / [] / in / len / bool / iter / items /
    values) so no call site or test changes. items() is a GENERATOR: iterating
    the whole index costs one record at a time, never the whole file.

    A record is served from disk on every read, so this trades a few hundred
    microseconds of I/O per request for ~137 MB of headroom. At the measured
    ~0.15 s per API response that is not a latency the user can perceive.
    """

    _LEAD_KEY = re.compile(br'^\s*\{\s*"series_id"\s*:\s*"([^"]+)"')

    def __init__(self, path):
        self.path = path
        self._offsets = {}
        self.load_error = None

    def build(self):
        """Scan once for line offsets. Never parses a full record."""
        off = 0
        try:
            with open(self.path, "rb") as fh:
                for raw in fh:
                    n = len(raw)
                    sid = None
                    m = self._LEAD_KEY.match(raw)
                    if m:
                        sid = m.group(1).decode("utf-8", "replace")
                    elif raw.strip():
                        # series_id is not the leading key on this line: fall
                        # back to a real parse rather than silently dropping
                        # the record.
                        try:
                            sid = json.loads(raw.decode("utf-8")).get("series_id")
                        except Exception:
                            sid = None
                    if sid:
                        self._offsets[sid] = (off, n)
                    off += n
        except Exception as exc:
            self.load_error = str(exc)
        return self

    def _read(self, off, n):
        with open(self.path, "rb") as fh:
            fh.seek(off)
            raw = fh.read(n)
        return json.loads(raw.decode("utf-8"))

    def get(self, sid, default=None):
        loc = self._offsets.get(sid)
        if loc is None:
            return default
        try:
            return self._read(*loc)
        except Exception:
            # a corrupt line must not 500 an endpoint that would otherwise
            # answer "available: false"
            return default

    def __getitem__(self, sid):
        return self._read(*self._offsets[sid])

    def __contains__(self, sid):
        return sid in self._offsets

    def __len__(self):
        return len(self._offsets)

    def __bool__(self):
        return bool(self._offsets)

    def __iter__(self):
        return iter(self._offsets)

    def keys(self):
        return self._offsets.keys()

    def items(self):
        for sid in self._offsets:
            rec = self.get(sid)
            if rec is not None:
                yield sid, rec

    def values(self):
        for _, rec in self.items():
            yield rec


class Store:
    """Loads and indexes the precomputed PRISE artifacts."""

    def __init__(self) -> None:
        self.m7_rollup = {}
        self.m8 = {}
        self.m9_rollup = {}
        self.m10 = {}                           # cross-modal payload (M10)
        self.conformal = {}                     # frozen conformal calibration + validation
        self.reality_check = {}                 # blind external reality check (§6C)
        # M11 metadata-quality: corpus rollup + per-series map (metadata_quality.*)
        self.metadata_corpus = {}               # metadata_quality.json {corpus, by_protein}
        self.metadata_by_series = {}            # series_id -> per-dataset M11 assess record
        # M12 information-content: per-series map (information_content.jsonl). The
        # headline per-curve information-geometry record (FIM spectrum, info-density
        # profile + phase fractions, observability, EIG, recommended measurement).
        self.information_by_series = {}          # series_id -> per-curve M12 record
        # M13 BOED: per-series map (boed_recommendations.jsonl). The optimal-design
        # capstone — ranked next-experiment recommendations, the (cost vs information)
        # Pareto frontier, and the greedy multi-step plan (keyed by series_id).
        self.boed_by_series = {}                 # series_id -> per-unit M13 record
        # M14 cross-study hierarchical meta-analysis: one JSON per MULTI-study
        # protein (data/processed/meta_analysis/*.json) keyed by the record's
        # `protein` field, PLUS the rollup's single_study statuses so an absent
        # protein returns an honest single_study note (not a 404/500).
        self.meta_by_protein = {}               # protein name -> full M14 record
        self.meta_rollup = {}                   # meta_analysis.json (summary + honesty)
        self.meta_single_by_protein = {}        # protein name -> single_study stub
        # M15 literature ingestion & claim extraction: the append-only claim ledger
        # (lit_claims.jsonl — every extraction is a CLAIM with provenance + measured
        # confidence, NOT a trusted fact), the cross-paper CONFLICTS (surfaced
        # UNRESOLVED), and the CPAD-validation accuracy (precision/recall MEASURED
        # against the curated gold standard). All OPTIONAL — /api/literature degrades
        # to {available: False} if the artifacts were not built.
        self.lit_claims = []                     # [claim record, ...] (append-only)
        self.lit_conflicts = {}                  # {version, conflicts:[...], honesty}
        self.lit_validation = {}                 # {version, validation:{...}, honesty}
        # M16 sequence<->structure<->kinetics bridge: per-protein structural PROXY
        # features (APR/sequence-derived, NOT 3D) keyed by uniprot, PLUS the corpus
        # structure->kinetics ASSOCIATION table (association-only, NEVER causal). Both
        # OPTIONAL — /api/structure degrades to {available: False} if not built.
        self.structure_by_uniprot = {}           # uniprot -> per-protein structure record
        self.structure_features_doc = {}         # structure_features.json top-level
        self.structure_assoc = {}                # structure_kinetics_associations.json
        # deterministic EB partial-pooling (pooling.json): per-curve shrinkage +
        # replicate meta, indexed series_id -> pooled record, protein -> [records]
        self.pooling = {}
        self.pooling_by_series = {}             # series_id -> per-curve pooled record
        self.pooling_by_protein = {}            # protein_id -> {records, replicates}
        # concentration_series_id -> {rows: [...]} — the per-series
        # (concentration, t50) table, precomputed by
        # `python engine/m4_features.py --t50-points`. /api/series-gamma used to
        # rebuild this live on every request (1-30 s per protein locally, far
        # worse on a 0.1-CPU container).
        self.t50_points_by_csid = {}
        # series_id -> {fits: {variant: fit_mechanistic() dict}} — the Tier-B
        # (Knowles/Cohen) bank, precomputed by `python engine/tierb_bank.py`.
        # /api/models fitted these four stiff ODE models live per request:
        # median 0.29 s but max 114 s (CPAD-TK-2185), on a 0.1-CPU container.
        self.mech_fits_by_series = {}
        # series_id -> {fits: {model: fit}} — registry models /api/models
        # compares but M2 did not select for that curve's shape (brain_cousens
        # on 1331 monotonic curves, the sigmoidal set on 377 non-monotonic
        # ones). Built by `python engine/tierb_bank.py --registry`.
        self.registry_gapfill_by_series = {}
        self.etl_manifest = {}
        # series_id -> ProteinAnalysisResult (M7)
        self.results_by_series = {}
        # protein_id -> [series_id, ...] (insertion order)
        self.series_by_protein = {}
        # series_id -> curve record (x_hours/y_intensity/m1/condition_vector)
        self.curve_by_series = {}
        # series_id -> M9 recommendation record
        self.m9_by_series = {}
        # series_id -> M2 fit record (fits.jsonl). THE PERFORMANCE FIX: /api/curve
        # and /api/models used to REFIT the whole candidate bank on every request
        # -- measured at 29 s and 38 s per curve after M2 gained genuine
        # multi-starts for all 13 models. Those fits are already computed and
        # published; recomputing them per page view was pure waste, and it also
        # meant the UI could show numbers that differed from the artifacts if the
        # serving host's stack differed. Serving the artifact is both faster and
        # more faithful.
        self.fits_by_series = {}
        # dual-γ payloads (gamma.jsonl), indexed several ways
        self.gamma_records = []                 # [dual_gamma payload, ...]
        self.gamma_by_protein = {}              # protein name -> [dual_gamma, ...]
        self.gamma_by_cs_id = {}                # concentration_series_id -> dual_gamma
        # concentration_series.json (membership), indexed by cs_id + protein
        self.conc_series_by_id = {}             # cs_id -> series meta (member ids)
        self.conc_series_by_protein = {}        # protein name -> [series meta, ...]
        # DOSE-RESPONSE fits (dose_response_fits.jsonl): k_agg-vs-concentration
        # model bank fitted to the CPAD R-rows, keyed by protein name. The raw
        # (concentration, k_agg) points come from rate_concentration.jsonl.
        self.dose_response_by_protein = {}      # protein name -> fit record
        self.dose_points_by_protein = {}        # protein name -> [(conc_uM, k_agg, pmid)]
        self.engine_build_id = None
        self.citation = None
        self.errors = []

    # -- loading helpers --------------------------------------------------- #
    @staticmethod
    def _load_json(name):
        with open(os.path.join(DATA, name), encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _iter_jsonl(name):
        path = os.path.join(DATA, name)
        with open(path, encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    # never let one bad record kill startup
                    continue

    def load(self) -> None:
        # small JSON rollups
        for attr, fname in (
            ("m7_rollup", "m7_rollup.json"),
            ("m8", "m8_reachability.json"),
            ("m9_rollup", "m9_rollup.json"),
            ("etl_manifest", "etl_manifest.json"),
        ):
            try:
                setattr(self, attr, self._load_json(fname))
            except Exception as exc:
                self.errors.append(f"{fname}: {exc}")

        # M10 cross-modal is OPTIONAL: the app degrades gracefully (the tab shows a
        # "not built" note) if data/processed/m10_crossmodal.json is absent.
        try:
            self.m10 = self._load_json("m10_crossmodal.json")
        except Exception as exc:
            self.m10 = {}
            self.errors.append(f"m10_crossmodal.json (optional): {exc}")

        # Conformal-prediction calibration is OPTIONAL: /api/conformal and the
        # per-curve interval display degrade gracefully if the frozen artifact
        # (service_c_conformal.json) was not built.
        try:
            self.conformal = self._load_json("service_c_conformal.json")
        except Exception as exc:
            self.conformal = {}
            self.errors.append(f"service_c_conformal.json (optional): {exc}")

        # Deterministic EB partial-pooling is OPTIONAL: /api/pooling and the
        # per-curve shrunk-t50 display degrade gracefully if pooling.json was not
        # built (run `python engine/pooling.py`).
        try:
            self.pooling = self._load_json("pooling.json")
            self._index_pooling()
        except Exception as exc:
            self.pooling = {}
            self.errors.append(f"pooling.json (optional): {exc}")

        # M14 cross-study meta-analysis is OPTIONAL: /api/meta/{protein} and the
        # protein-level "Cross-study meta-analysis (M14)" panel degrade gracefully
        # if the artifacts were not built (run `python engine/m14_metaanalysis.py`).
        # The per-protein multi-study records live in data/processed/meta_analysis/
        # (one JSON each, keyed by the record's `protein` field); the rollup
        # (meta_analysis.json) carries the summary + honesty + every protein's
        # status incl. single_study — indexed so an absent/single-study protein
        # returns an honest note, never a 500. Never let one bad file kill startup.
        self._load_meta_analysis()

        # Blind external reality check is OPTIONAL: /api/reality-check and the
        # Corpus-overview panel degrade gracefully if reality_check.json was not
        # built (run `python engine/reality_check.py`).
        try:
            self.reality_check = self._load_json("reality_check.json")
        except Exception as exc:
            self.reality_check = {}
            self.errors.append(f"reality_check.json (optional): {exc}")

        # M11 metadata-quality is OPTIONAL: /api/metadata/{series_id},
        # /api/metadata-overview and the metadata-quality panels degrade gracefully
        # ({available: False}) if the artifacts were not built (run
        # `python engine/m11_metadata.py`). The corpus rollup is a small JSON; the
        # per-series records are a large JSONL indexed by series_id at startup.
        try:
            self.metadata_corpus = self._load_json("metadata_quality.json")
        except Exception as exc:
            self.metadata_corpus = {}
            self.errors.append(f"metadata_quality.json (optional): {exc}")
        idx = LazyRecordIndex(os.path.join(DATA, "metadata_quality.jsonl")).build()
        if idx.load_error:
            self.errors.append(f"metadata_quality.jsonl (optional): {idx.load_error}")
        else:
            self.metadata_by_series = idx

        # M12 information-content is OPTIONAL: /api/information/{series_id} and the
        # per-curve information-content panel degrade gracefully ({available: False})
        # if the artifact was not built (run `python engine/m12_information.py`). It
        # is a large JSONL indexed by series_id at startup.
        idx = LazyRecordIndex(os.path.join(DATA, "information_content.jsonl")).build()
        if idx.load_error:
            self.errors.append(f"information_content.jsonl (optional): {idx.load_error}")
        else:
            self.information_by_series = idx

        # M13 BOED is OPTIONAL: /api/boed/{series_id} and the "Next experiment —
        # optimal design" panel degrade gracefully ({available: False}) if the
        # artifact was not built (run `python engine/m13_boed.py`). It is a large
        # JSONL indexed by series_id at startup.
        idx = LazyRecordIndex(os.path.join(DATA, "boed_recommendations.jsonl")).build()
        if idx.load_error:
            self.errors.append(f"boed_recommendations.jsonl (optional): {idx.load_error}")
        else:
            self.boed_by_series = idx

        # M15 literature ingestion is OPTIONAL: /api/literature and the Corpus-overview
        # "Literature ingestion — extracted claims (M15)" card degrade gracefully
        # ({available: False}) if the artifacts were not built (run
        # `python engine/m15_litingest.py`). The claim ledger is a JSONL (append-only,
        # one CLAIM per line with full provenance + measured confidence); the conflicts
        # + CPAD-validation are small JSONs. Never let one bad file kill startup.
        try:
            for r in self._iter_jsonl("lit_claims.jsonl"):
                self.lit_claims.append(r)
        except Exception as exc:
            self.errors.append(f"lit_claims.jsonl (optional): {exc}")
        try:
            self.lit_conflicts = self._load_json("lit_conflicts.json")
        except Exception as exc:
            self.lit_conflicts = {}
            self.errors.append(f"lit_conflicts.json (optional): {exc}")
        try:
            self.lit_validation = self._load_json("lit_validation.json")
        except Exception as exc:
            self.lit_validation = {}
            self.errors.append(f"lit_validation.json (optional): {exc}")

        # M16 sequence<->structure<->kinetics bridge is OPTIONAL: /api/structure/{protein_id}
        # and the protein-level "Sequence <-> structure <-> kinetics (M16)" panel degrade
        # gracefully ({available: False}) if the artifacts were not built (run
        # `python engine/m16_structure.py`). structure_features.json is a LARGE JSON
        # (per-protein APR/sequence PROXY features keyed by uniprot); the associations
        # table is a small JSON. Never let one bad file kill startup.
        try:
            self.structure_features_doc = self._load_json("structure_features.json")
            byu = (self.structure_features_doc or {}).get(
                "structural_features_by_uniprot", {}) or {}
            if isinstance(byu, dict):
                self.structure_by_uniprot = byu
        except Exception as exc:
            self.structure_features_doc = {}
            self.structure_by_uniprot = {}
            self.errors.append(f"structure_features.json (optional): {exc}")
        try:
            self.structure_assoc = self._load_json(
                "structure_kinetics_associations.json")
        except Exception as exc:
            self.structure_assoc = {}
            self.errors.append(
                f"structure_kinetics_associations.json (optional): {exc}")

        # Registry gap-fill. OPTIONAL: absent -> /api/models fits the
        # non-selected models live (correct, just slow).
        try:
            for r in self._iter_jsonl("registry_gapfill_fits.jsonl"):
                sid = r.get("series_id")
                if sid:
                    self.registry_gapfill_by_series[sid] = r
        except Exception as exc:
            self.errors.append(f"registry_gapfill_fits.jsonl (optional): {exc}")

        # Precomputed Tier-B mechanistic bank. OPTIONAL: absent -> /api/models
        # falls back to fitting live (correct, just slow).
        try:
            for r in self._iter_jsonl("mechanistic_fits.jsonl"):
                sid = r.get("series_id")
                if sid:
                    self.mech_fits_by_series[sid] = r
        except Exception as exc:
            self.errors.append(f"mechanistic_fits.jsonl (optional): {exc}")

        # Precomputed per-series t50 points. OPTIONAL: absent -> /api/series-gamma
        # falls back to computing them live (correct, just slow).
        try:
            for r in self._iter_jsonl("series_t50_points.jsonl"):
                csid = r.get("concentration_series_id")
                if csid:
                    self.t50_points_by_csid[csid] = r
        except Exception as exc:
            self.errors.append(f"series_t50_points.jsonl (optional): {exc}")

        self.engine_build_id = (
            self.m7_rollup.get("engine_build_id")
            or self.m8.get("versioning", {}).get("engine_build_id")
        )
        self.citation = self.etl_manifest.get("source_citation")

        # M7 per-curve results -> index by series_id + group by protein_id
        for r in self._iter_jsonl("protein_analysis.jsonl"):
            sid = r.get("series_id")
            pid = r.get("protein_id")
            if not sid:
                continue
            self.results_by_series[sid] = r
            self.series_by_protein.setdefault(pid, []).append(sid)

        # curve points + M1 triage block
        for c in self._iter_jsonl("curves_triaged.jsonl"):
            sid = c.get("series_id")
            if sid:
                self.curve_by_series[sid] = c

        # M2 candidate-bank fits, for the curve overlay and the model table
        for f in self._iter_jsonl("fits.jsonl"):
            sid = f.get("series_id")
            if sid:
                self.fits_by_series[sid] = f

        # M9 recommendations keyed by unit_id (== series_id)
        for m in self._iter_jsonl("m9_recommendations.jsonl"):
            sid = m.get("unit_id") or m.get("series_id")
            if sid:
                self.m9_by_series[sid] = m

        # dual-γ payloads -> index by protein name + concentration_series_id
        for g in self._iter_jsonl("gamma.jsonl"):
            self.gamma_records.append(g)
            prot = g.get("protein")
            csid = g.get("concentration_series_id")
            if prot:
                self.gamma_by_protein.setdefault(prot, []).append(g)
            if csid:
                self.gamma_by_cs_id[csid] = g

        # concentration-series membership (which curves belong to each series)
        try:
            cs = self._load_json("concentration_series.json")
            for meta in (cs or []):
                csid = meta.get("concentration_series_id")
                prot = meta.get("protein")
                if csid:
                    self.conc_series_by_id[csid] = meta
                if prot:
                    self.conc_series_by_protein.setdefault(prot, []).append(meta)
        except Exception as exc:
            self.errors.append(f"concentration_series.json: {exc}")

        # DOSE-RESPONSE (k_agg vs concentration) is OPTIONAL: /api/dose-response and
        # the protein-detail dose-response card degrade gracefully if the artifact
        # was not built (run `python engine/m2_fit.py --dose`).
        for d in self._iter_jsonl("dose_response_fits.jsonl"):
            prot = d.get("protein")
            if prot:
                self.dose_response_by_protein[prot] = d
        # raw (concentration, k_agg) points per protein, for the scatter overlay
        for r in self._iter_jsonl("rate_concentration.jsonl"):
            prot = r.get("protein")
            c = (r.get("concentration") or {}).get("value_uM")
            k = r.get("k_agg")
            if prot and c is not None and k is not None:
                self.dose_points_by_protein.setdefault(prot, []).append(
                    (float(c), float(k), r.get("pmid")))

    def _index_pooling(self):
        """Index the EB partial-pooling payload by series_id + protein_id so the
        per-curve display can show the shrunk t50 and the protein endpoint can return
        the replicate stochastic-nucleation signals. Best-effort; never raises."""
        pool = self.pooling or {}
        for r in pool.get("curve_records", []) or []:
            sid = r.get("series_id")
            pid = r.get("protein_id")
            if sid:
                self.pooling_by_series[sid] = r
            if pid:
                self.pooling_by_protein.setdefault(
                    pid, {"records": [], "replicates": []})["records"].append(r)
        # replicate meta keyed by uniprot — attach to every protein sharing it
        uni_to_pid = {}
        for r in pool.get("curve_records", []) or []:
            if r.get("uniprot_id") and r.get("protein_id"):
                uni_to_pid.setdefault(r["uniprot_id"], r["protein_id"])
        for m in pool.get("replicate_meta", []) or []:
            uni = (m.get("stratum") or {}).get("uniprot")
            pid = uni_to_pid.get(uni)
            if pid and pid in self.pooling_by_protein:
                self.pooling_by_protein[pid]["replicates"].append(m)

    def _load_meta_analysis(self):
        """Load the M14 cross-study meta-analysis artifacts. Each MULTI-study
        protein has its own JSON in data/processed/meta_analysis/, keyed here by
        the record's `protein` field (the real protein name that matches the
        /api/protein ids). The rollup (meta_analysis.json) is read for the summary
        + honesty + the single_study statuses so an absent protein can be answered
        with an honest single_study note instead of a 404. Best-effort — one bad
        file never kills startup."""
        meta_dir = os.path.join(DATA, "meta_analysis")
        n_multi = 0
        try:
            if os.path.isdir(meta_dir):
                for fn in sorted(os.listdir(meta_dir)):
                    if not fn.endswith(".json"):
                        continue
                    try:
                        with open(os.path.join(meta_dir, fn), encoding="utf-8") as fh:
                            rec = json.load(fh)
                    except Exception as exc:
                        self.errors.append(f"meta_analysis/{fn}: {exc}")
                        continue
                    prot = rec.get("protein")
                    if prot and rec.get("status") == "meta_analysis":
                        self.meta_by_protein[prot] = rec
                        n_multi += 1
                    elif prot:
                        # a single-study (or flagged) record shipped as its own file
                        self.meta_single_by_protein[prot] = rec
        except Exception as exc:
            self.errors.append(f"meta_analysis/ dir: {exc}")

        # rollup: summary + honesty + every protein's status (incl. single_study)
        try:
            self.meta_rollup = self._load_json("meta_analysis.json")
            for p in (self.meta_rollup.get("proteins") or []):
                prot = p.get("protein")
                if not prot:
                    continue
                if p.get("status") == "meta_analysis":
                    # prefer the fuller per-file record if we already have it
                    self.meta_by_protein.setdefault(prot, p)
                else:
                    self.meta_single_by_protein.setdefault(prot, p)
        except Exception as exc:
            self.meta_rollup = {}
            self.errors.append(f"meta_analysis.json (optional): {exc}")

        self._meta_multi_count = n_multi

    # -- derived views ----------------------------------------------------- #
    def protein_list(self, q=None):
        """[{protein_id, uniprot_id, n_curves, modal_regime, best_information_yield,
        has_concentration_series}] grouped from protein_analysis.jsonl."""
        yield_rank = {
            "mechanistic": 5, "scaling": 4, "descriptive": 3,
            "signal_only": 2, "uninformative": 1,
        }
        out = []
        ql = q.lower() if q else None
        for pid, sids in self.series_by_protein.items():
            if ql and ql not in (pid or "").lower():
                # also match on uniprot id
                first = self.results_by_series.get(sids[0], {})
                if ql not in (first.get("uniprot_id") or "").lower():
                    continue
            uniprot = None
            regimes = {}
            best_yield = None
            best_rank = -1
            has_series = False
            for sid in sids:
                r = self.results_by_series.get(sid, {})
                uniprot = uniprot or r.get("uniprot_id")
                reg = (r.get("descriptive_regime") or {}).get("regime")
                if reg:
                    regimes[reg] = regimes.get(reg, 0) + 1
                iy = r.get("information_yield")
                rk = yield_rank.get(iy, 0)
                if rk > best_rank:
                    best_rank, best_yield = rk, iy
                # concentration series membership from the curve record
                cur = self.curve_by_series.get(sid, {})
                if cur.get("concentration_series_id"):
                    has_series = True
            # authoritative: does this protein have a dual-γ / concentration series?
            if pid in self.gamma_by_protein or pid in self.conc_series_by_protein:
                has_series = True
            modal = max(regimes, key=regimes.get) if regimes else None
            out.append({
                "protein_id": pid,
                "uniprot_id": uniprot,
                "n_curves": len(sids),
                "modal_regime": modal,
                "best_information_yield": best_yield,
                "has_concentration_series": has_series,
                # join on protein name: does this protein have CPAD R-rows fitted
                # to a dose-response (k_agg vs concentration) model bank?
                "has_dose_response": pid in self.dose_response_by_protein,
            })
        out.sort(key=lambda d: (-d["n_curves"], d["protein_id"] or ""))
        return out

    def protein_detail(self, pid):
        sids = self.series_by_protein.get(pid)
        if not sids:
            return None
        units = [self.results_by_series[s] for s in sids if s in self.results_by_series]
        meta = units[0] if units else {}
        # M9 recommendations joined by series_id (unit)
        m9 = {}
        for s in sids:
            rec = self.m9_by_series.get(s)
            if rec:
                m9[s] = {
                    "top_recommendation": rec.get("top_recommendation"),
                    "n_recommendations": rec.get("n_recommendations"),
                    "recommendations": rec.get("recommendations", []),
                    "refusals": rec.get("refusals", []),
                    "state": rec.get("state", {}),
                }
        # representative unit for sequence axis + structure linkage: prefer a
        # full-schema unit that actually carries a sequence axis.
        #
        # THE CARD IS NOT PURELY PROTEIN-LEVEL. The predictor SCORES are, but the
        # licensing fields the card renders as badges (assay_endpoint_class,
        # endpoint_match_to_assay, comparison_licensed) are pure functions of the
        # ASSAY — so for a protein measured by several assays this shows whichever
        # unit sorted first. Since join-policy-1.0 the engine refuses to attach a
        # cross-assay sequence axis for exactly that reason; presenting one unit's
        # licensing as the protein's would reintroduce at the view layer the error
        # the engine now rejects. The assay is therefore published ALONGSIDE the
        # block, and the set of assays this protein actually spans, so the client
        # can say which endpoint the badges were computed against instead of
        # implying they hold for all of them.
        rep = None
        for u in units:
            if u.get("schema") == "full" and (u.get("sequence_axis") or {}).get("available"):
                rep = u
                break
        if rep is None:
            for u in units:
                if u.get("schema") == "full":
                    rep = u
                    break
        rep = rep or meta
        rep_assay = ((rep.get("condition_vector") or {}).get("assay_type")
                     if isinstance(rep, dict) else None)
        assays = sorted({(u.get("condition_vector") or {}).get("assay_type")
                         for u in units
                         if (u.get("condition_vector") or {}).get("assay_type")})
        return {
            "protein_id": pid,
            "uniprot_id": meta.get("uniprot_id"),
            "n_curves": len(sids),
            "engine_build_id": meta.get("engine_build_id") or self.engine_build_id,
            "units": units,
            "m9_recommendations": m9,
            "sequence_axis": rep.get("sequence_axis"),
            # which assay the licensing badges above were computed FOR, and every
            # assay this protein is measured by. Equal-length lists mean the card
            # speaks for the whole protein; a shorter first element does not.
            "sequence_axis_assay": rep_assay,
            "sequence_axis_assay_is_representative_only": len(assays) > 1,
            "assays_spanned": assays,
            "structure_linkage": self._structure_linkage(rep, units),
            "attribution": self._attribution(),
        }

    def _structure_linkage(self, rep, units):
        # gather distinct PDB ids across the protein's condition vectors
        pdbs = []
        for u in units:
            pdb = (u.get("condition_vector") or {}).get("pdb_id")
            if pdb and pdb not in pdbs:
                pdbs.append(pdb)
        return {"pdb_ids": pdbs, "relationship": "associative_only",
                "caveat": "structure linkage is associative, never causal; fibril "
                          "polymorphism can make 'the structure of protein P' ill-posed"}

    def _attribution(self):
        return {
            "source_dataset": self.etl_manifest.get("source_dataset"),
            "citation": self.citation,
            "note": "Derived analytical results + single curves only; raw bulk "
                    "CPAD data is not served (attribution-gated).",
            "engine_build_id": self.engine_build_id,
        }


STORE = Store()


# --------------------------------------------------------------------------- #
# Model predict helper (fitted-overlay) — reuses engine models registry
# --------------------------------------------------------------------------- #
def predict_model(model_name, params, xs):
    """Evaluate a fitted kinetic model on xs. Returns list[float] or None."""
    if not ENGINE_OK or _models is None:
        return None
    entry = _models.REGISTRY.get(model_name)
    if not entry or not isinstance(params, dict):
        return None
    try:
        pnames = entry["params"]
        args = [float(params[p]) for p in pnames]
        func = entry["func"]
        return [float(func(float(t), *args)) for t in xs]
    except Exception:
        return None


def _dense_grid(xs, n=120):
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return list(xs)
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


# --------------------------------------------------------------------------- #
# Endpoint implementations (return python objects; router serialises)
# --------------------------------------------------------------------------- #
def api_overview():
    m7 = STORE.m7_rollup
    m8 = STORE.m8
    m9 = STORE.m9_rollup

    ladder = m8.get("inference_ladder", {})
    boundary = m8.get("mechanistically_resolvable_boundary", {})
    blocker = m8.get("blocker_census", {})
    vc = m8.get("validity_ceiling", {})

    # top blocker census items (already impact-ranked)
    blockers = []
    for b in blocker.get("blockers_impact_ranked", [])[:5]:
        blockers.append({
            "blocker": b.get("blocker"),
            "count": b.get("count"),
            "fraction": b.get("fraction"),
            "blocks_rung": b.get("blocks_rung"),
            "detail": b.get("detail"),
            "scope": b.get("scope"),
        })

    top_action = None
    actions = m9.get("corpus_actions_ranked", [])
    if actions:
        top_action = actions[0]

    return {
        "engine_build_id": STORE.engine_build_id,
        "n_curves": m7.get("n_results"),
        "yield_tiers": m7.get("by_information_yield", {}),
        "descriptive_regimes": m7.get("by_descriptive_regime", {}),
        "gamma_counts": {
            "n_with_reliable_gamma_fit": m7.get("n_with_reliable_gamma_fit"),
            "n_with_significant_gamma": m7.get("n_with_significant_gamma"),
            "note": m7.get("gamma_count_note"),
        },
        "corpus_caveats": m7.get("corpus_caveats", []),
        "inference_ladder": {
            "n_curves": ladder.get("n_curves"),
            "n_proteins": ladder.get("n_proteins"),
            "rungs": ladder.get("rungs", []),
            "monotone_non_increasing": ladder.get("monotone_non_increasing"),
        },
        "blocker_census": {
            "headline": blocker.get("headline"),
            "top": blockers,
        },
        "boundary_gap": {
            "structural_reachable_proteins": boundary.get("structural_reachable_proteins_upper_bound"),
            "structural_reachable_series": boundary.get("structural_reachable_series_upper_bound"),
            "actual_mechanistically_licensed": boundary.get("actual_mechanistically_licensed"),
            "gap": boundary.get("gap_structural_minus_licensed"),
            "headline": boundary.get("headline"),
            "is_upper_bound": boundary.get("is_upper_bound"),
        },
        "validity_ceiling": {
            "clause": vc.get("clause"),
            "mechanism_misidentification_degradation": (
                vc.get("evidence_a_different_forward_model_degrades_resolution", {})
                .get("mechanism_misidentification_degradation")
            ),
            "deferred_external_validation": vc.get("deferred_external_validation", {}),
        },
        "m9": {
            "top_corpus_action": m9.get("top_corpus_action"),
            "headline": m9.get("headline"),
            "top_action": top_action,
        },
        "attribution": STORE._attribution(),
    }


def api_crossmodal():
    """GET /api/crossmodal: the M10 cross-modal Sequence × Kinetics × Structure
    payload (coverage/effective-n, P1 association cards, P2 APR→regime trend, P3
    discordance). Degrades gracefully: if the artifact was not built, returns a
    {available: False} block the frontend renders as a 'not built' note."""
    m10 = STORE.m10 or {}
    if not m10 or m10.get("status") == "error":
        return {
            "available": False,
            "reason": (m10.get("error") if m10.get("status") == "error" else
                       "m10_crossmodal.json not found — run "
                       "`python engine/m10_crossmodal.py` to build it"),
            "attribution": STORE._attribution(),
        }
    out = dict(m10)
    out["available"] = True
    out["attribution"] = STORE._attribution()
    return out


def api_conformal():
    """GET /api/conformal: the frozen conformal-prediction calibration + the
    realized-coverage validation summary (PRISE_DESIGN.md §4). Serves the version,
    target coverage, the per-stratum t50 quantiles the frontend applies client-side,
    and the validation table (target vs realized, width / set size, overall + per
    Mondrian stratum). Degrades gracefully: if service_c_conformal.json was not
    built, returns {available: False} with a reason."""
    cf = STORE.conformal or {}
    if not cf or "calibration_quantiles" not in cf:
        return {
            "available": False,
            "reason": ("service_c_conformal.json not found — run "
                       "`python engine/conformal.py` to build the calibration"),
            "attribution": STORE._attribution(),
        }
    cq = cf.get("calibration_quantiles", {})
    val = cf.get("validation", {})
    return {
        "available": True,
        "version": cf.get("version"),
        "alpha": cf.get("alpha"),
        "target_coverage": cf.get("target_coverage"),
        "studentize": cf.get("studentize"),
        "guarantee": cf.get("guarantee"),
        "honesty": cf.get("honesty"),
        # the per-stratum quantiles the per-curve display applies client-side
        "calibration_quantiles": {
            "alpha": cq.get("alpha"),
            "target_coverage": cq.get("target_coverage"),
            "t50": cq.get("t50"),
            "regime": cq.get("regime"),
            "mondrian_axes": cq.get("mondrian_axes"),
            "n_calibration_points_used": cq.get("n_calibration_points_used"),
        },
        # the proof: realized coverage vs target, overall + per Mondrian stratum
        "validation": val,
        "attribution": STORE._attribution(),
    }


def api_reality_check():
    """GET /api/reality-check: the blind external reality check (§6C falsification
    test). Serves the pre-registered consensus fixed points vs PRISE's blind output
    (descriptive regime + γ + mechanistic equivalence class), the per-protein
    CONSISTENT/CONTRADICTION/INSUFFICIENT verdicts, the overall non-contradiction
    result, the reserved-proteins/leakage discipline, and the honest framing.
    Degrades gracefully: if reality_check.json was not built, returns
    {available: False} with a reason."""
    rc = STORE.reality_check or {}
    if not rc or "reality_check_result" not in rc:
        return {
            "available": False,
            "reason": ("reality_check.json not found — run "
                       "`python engine/reality_check.py` to build it"),
            "attribution": STORE._attribution(),
        }
    out = dict(rc)
    out["available"] = True
    out["attribution"] = STORE._attribution()
    return out


def api_metadata(series_id):
    """GET /api/metadata/{series_id}: the per-dataset M11 assess result (metadata
    completeness score, uncertainty, mechanistic completeness + blockers, the ranked
    missing fields by impact with their consequence chains, and the top recommendation).

    Reads the precomputed metadata_quality.jsonl record (preferred); degrades
    gracefully to {available: False} if the artifact was not built or the series is
    not indexed. Never 500 (wrapped in try/except)."""
    try:
        if not STORE.metadata_by_series:
            return {
                "available": False,
                "reason": ("metadata_quality.jsonl not found — run "
                           "`python engine/m11_metadata.py` to build the M11 "
                           "metadata-quality layer"),
                "attribution": STORE._attribution(),
            }
        rec = STORE.metadata_by_series.get(series_id)
        if not rec:
            return {
                "available": False,
                "not_found": True,
                "reason": f"no metadata-quality record for series: {series_id}",
                "attribution": STORE._attribution(),
            }
        comp = rec.get("metadata_completeness", {}) or {}
        unc = rec.get("metadata_uncertainty", {}) or {}
        recs = rec.get("recommended_missing_metadata", []) or []
        return {
            "available": True,
            "series_id": series_id,
            "protein_id": rec.get("protein_id"),
            "uniprot_id": rec.get("uniprot_id"),
            "version": rec.get("version"),
            "ontology_version": rec.get("ontology_version"),
            # completeness (0..1) — PRESENCE, not correctness
            "metadata_score": comp.get("metadata_score"),
            "n_capturable_fields": comp.get("n_capturable_fields"),
            "not_captured_by_source": rec.get("not_captured_by_source", []),
            "metadata_completeness": comp,
            # a SEPARATE trust scalar (0 = trustworthy)
            "metadata_uncertainty": unc.get("metadata_uncertainty"),
            "uncertainty_detail": unc,
            # mechanistic completeness + the explicit blockers
            "mechanistic_completeness": rec.get("mechanistic_completeness"),
            "mechanistic_blockers": rec.get("mechanistic_blockers", []),
            "dependency_evaluation": rec.get("dependency_evaluation", {}),
            # ranked missing fields by impact, each with its consequence_chain
            "delta_inferential_power": rec.get("delta_inferential_power", []),
            "recommended_missing_metadata": recs,
            "top_recommendation": recs[0] if recs else None,
            "honesty": rec.get("honesty", {}),
            "engine_build_id": STORE.engine_build_id,
            "attribution": STORE._attribution(),
        }
    except Exception as exc:  # pragma: no cover - defensive, never 500
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}",
                "series_id": series_id, "attribution": STORE._attribution()}


def api_metadata_overview():
    """GET /api/metadata-overview: the M11 CORPUS metadata-quality summary — the
    median metadata_score, the mechanistic_completeness distribution, the
    WHY-mechanism≈0 universal-blocker explanation ({agitation, seeded}), and the
    top corpus recommendations. Degrades to {available: False}. Never 500."""
    try:
        mc = STORE.metadata_corpus or {}
        corpus = mc.get("corpus") if isinstance(mc, dict) else None
        if not corpus:
            return {
                "available": False,
                "reason": ("metadata_quality.json not found — run "
                           "`python engine/m11_metadata.py` to build it"),
                "attribution": STORE._attribution(),
            }
        ub = corpus.get("universal_blocker", {}) or {}
        return {
            "available": True,
            "version": corpus.get("version"),
            "ontology_version": corpus.get("ontology_version"),
            "n_curves": corpus.get("n_curves"),
            "n_proteins": corpus.get("n_proteins"),
            "metadata_score": corpus.get("metadata_score", {}),
            "mechanistic_completeness": corpus.get("mechanistic_completeness", {}),
            "metadata_uncertainty": corpus.get("metadata_uncertainty", {}),
            "n_mechanism_licensed": corpus.get("n_mechanism_licensed"),
            "universal_blocker": {
                "fields": ["agitation", "seeded"],
                "n_curves_blocked": ub.get("n_curves_agitation_or_seeding_blocked"),
                "fraction": ub.get("fraction"),
                "why_mechanism_licensed_is_zero":
                    ub.get("why_mechanism_licensed_is_zero"),
            },
            "top_recommendations":
                (corpus.get("corpus_recommendations_ranked", []) or [])[:3],
            "honesty": corpus.get("honesty", {}),
            "engine_build_id": STORE.engine_build_id,
            "attribution": STORE._attribution(),
        }
    except Exception as exc:  # pragma: no cover - defensive, never 500
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}",
                "attribution": STORE._attribution()}


def api_information(series_id):
    """GET /api/information/{series_id}: the per-curve M12 information-content record
    (the headline information-geometry deliverable). Returns the information-density
    profile D(t) aligned to the curve's OBSERVED time grid + total, the phase
    fractions (lag/growth/plateau) + phase boundaries, the FIM spectrum scalars
    (effective rank HARD vs ENTROPY — the sloppiness story, condition number,
    richness score, redundancy, EIG bits), the per-parameter observability table +
    stiff/sloppy identifiable combinations, the recommended additional measurement
    (time + phase + rationale, censoring tie-in), and the honesty block (FIM is
    local / linearized / model-conditional / post-selection).

    Reads the precomputed information_content.jsonl record. Degrades gracefully to
    {available: False} if the artifact was not built or the series is not indexed
    (or the record is a flagged no_fit/degenerate result). Never 500 (wrapped)."""
    try:
        if not STORE.information_by_series:
            return {
                "available": False,
                "reason": ("information_content.jsonl not found — run "
                           "`python engine/m12_information.py` to build the M12 "
                           "information-content layer"),
                "attribution": STORE._attribution(),
            }
        rec = STORE.information_by_series.get(series_id)
        if not rec:
            return {
                "available": False,
                "not_found": True,
                "reason": f"no information-content record for series: {series_id}",
                "attribution": STORE._attribution(),
            }
        status = rec.get("status")
        # a flagged minimal result (no converged model / degenerate FIM): surface the
        # scalars we DO have + the reason, but flag that the full geometry is absent.
        if status != "ok":
            return {
                "available": True,
                "computable": False,
                "series_id": series_id,
                "status": status,
                "selected_model": rec.get("selected_model"),
                "reason": rec.get("note") or f"information geometry not computable ({status})",
                "effective_rank_hard": rec.get("effective_rank_hard"),
                "effective_rank_entropy": rec.get("effective_rank_entropy"),
                "information_richness_score": rec.get("information_richness_score"),
                "honesty": rec.get("honesty", {}),
                "engine_build_id": STORE.engine_build_id,
                "attribution": STORE._attribution(),
            }
        dens = rec.get("information_density", {}) or {}
        obsv = rec.get("parameter_observability", {}) or {}
        spec = rec.get("spectrum", {}) or {}
        ms = rec.get("model_selection", {}) or {}
        return {
            "available": True,
            "computable": True,
            "series_id": series_id,
            "status": status,
            "version": rec.get("version"),
            "selected_model": rec.get("selected_model"),
            "selected_by": ms.get("selected_by"),
            "n_params": ms.get("n_params"),
            "n_points": ms.get("n_points"),
            "params": ms.get("params"),
            "censoring_class": rec.get("censoring_class"),
            # ---- THE HEADLINE: information-density D(t) aligned to the observed grid,
            # its phase fractions + boundaries (for the phase-band shading) ----
            "information_density": {
                "density": dens.get("density", []),
                "total_information": dens.get("total_information"),
                "cooperative_phases": dens.get("cooperative_phases"),
                "phase_fractions": dens.get("phase_fractions", {}),
                "phase_boundaries": dens.get("phase_boundaries", {}),
                "phase_note": dens.get("phase_note"),
            },
            # ---- the scalars: effective rank HARD vs ENTROPY (the sloppiness story) ----
            "effective_rank_hard": rec.get("effective_rank_hard"),
            "effective_rank_entropy": rec.get("effective_rank_entropy"),
            "condition_number": rec.get("condition_number"),
            "condition_number_correlation": rec.get("condition_number_correlation"),
            "spectral_entropy": spec.get("spectral_entropy"),
            "rank_deficient": spec.get("rank_deficient"),
            "information_richness_score": rec.get("information_richness_score"),
            "information_richness": rec.get("information_richness", {}),
            "redundancy": rec.get("redundancy", {}),
            "expected_information_gain": rec.get("expected_information_gain", {}),
            "signal_to_noise": rec.get("signal_to_noise", {}),
            # ---- parameter observability + identifiable combinations ----
            "parameter_observability": {
                "per_parameter": obsv.get("per_parameter", {}),
                "n_observable": obsv.get("n_observable"),
                "observable_cv_threshold": obsv.get("observable_cv_threshold"),
                "stiff_combinations": obsv.get("stiff_combinations", []),
                "sloppy_combinations": obsv.get("sloppy_combinations", []),
            },
            "relative_sensitivity_peak_time":
                (rec.get("sensitivity", {}) or {}).get("relative_sensitivity_peak_time", {}),
            # ---- recommended additional measurement (M9 action language) ----
            "recommended_additional_measurements":
                rec.get("recommended_additional_measurements", {}),
            "expected_variance_reduction": {
                "recommended_time": (rec.get("expected_variance_reduction", {}) or {}).get("recommended_time"),
                "observed_t_max": (rec.get("expected_variance_reduction", {}) or {}).get("observed_t_max"),
                "grid_max": (rec.get("expected_variance_reduction", {}) or {}).get("grid_max"),
            },
            # ---- honesty (FIM local / linearized / model-conditional / post-selection) ----
            "honesty": rec.get("honesty", {}),
            "engine_build_id": STORE.engine_build_id,
            "attribution": STORE._attribution(),
        }
    except Exception as exc:  # pragma: no cover - defensive, never 500
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}",
                "series_id": series_id, "attribution": STORE._attribution()}


def api_boed(series_id):
    """GET /api/boed/{series_id}: the per-unit M13 Bayesian Optimal Experimental
    Design record — the ranked next-experiment recommendations, the (cost vs expected
    information) Pareto frontier, and the greedy multi-step plan.

    Returns the full precomputed boed_recommendations.jsonl record (candidates[] with
    the four §3-M13 quantities per design, ranked[], pareto_optimal_sets[],
    multi_step_plan{}, current_state, top_recommendation, honesty). Degrades
    gracefully to {available: False} if the artifact was not built or the series is
    not indexed. A status=='INSUFFICIENT' record (no fitted forward model) is served
    as-is with available=True so the panel can show the M9 fallback. Never 500."""
    try:
        if not STORE.boed_by_series:
            return {
                "available": False,
                "reason": ("boed_recommendations.jsonl not found — run "
                           "`python engine/m13_boed.py` to build the M13 BOED "
                           "optimal-experimental-design layer"),
                "attribution": STORE._attribution(),
            }
        rec = STORE.boed_by_series.get(series_id)
        if not rec:
            return {
                "available": False,
                "not_found": True,
                "reason": f"no BOED record for series: {series_id}",
                "attribution": STORE._attribution(),
            }
        out = dict(rec)
        out["available"] = True
        out["engine_build_id"] = STORE.engine_build_id
        out["attribution"] = STORE._attribution()
        return out
    except Exception as exc:  # pragma: no cover - defensive, never 500
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}",
                "series_id": series_id, "attribution": STORE._attribution()}


def api_literature():
    """GET /api/literature: the M15 literature-ingestion & claim-extraction layer.

    CORE FRAMING (honesty-first): these are CLAIMS, not trusted facts. Every claim
    carries full provenance (doi + page + location_type + verbatim source_text +
    char_span) and a MEASURED confidence; cross-paper CONFLICTS are surfaced
    UNRESOLVED (never auto-resolved); extraction accuracy is MEASURED against CPAD's
    curated gold standard. M15 PROPOSES; a curator approves — nothing is auto-trusted.

    Returns {available, n_claims, claims_by_paper (grouped by doi/pmid), field_coverage
    (which of the target fields were extracted + counts), conflicts, validation,
    honesty}. Degrades gracefully to {available: False} if the ledger was not built.
    Wrapped in try/except — never 500."""
    try:
        claims = STORE.lit_claims or []
        if not claims:
            return {
                "available": False,
                "reason": ("lit_claims.jsonl not found — run "
                           "`python engine/m15_litingest.py` to build the M15 "
                           "literature-ingestion & claim-extraction layer"),
                "attribution": STORE._attribution(),
            }

        # group claims by paper (doi|pmid) — preserve first-seen order
        papers = {}
        order = []
        for c in claims:
            doi = c.get("doi")
            pmid = c.get("pmid")
            key = (doi or "") + "|" + (pmid or "")
            if key not in papers:
                papers[key] = {
                    "paper_key": key,
                    "doi": doi,
                    "pmid": pmid,
                    # author/year/title are OPTIONAL — carried only if the extractor
                    # emitted them (the offline fixtures do not).
                    "author": c.get("author"),
                    "year": c.get("year"),
                    "title": c.get("title"),
                    "claims": [],
                }
                order.append(key)
            papers[key]["claims"].append(c)
        claims_by_paper = [papers[k] for k in order]

        # field-coverage summary: which target fields were extracted + counts, split
        # by group so the UI can show conditions vs reported_results vs semantic.
        field_coverage = {}
        for c in claims:
            fld = c.get("field")
            if not fld:
                continue
            fc = field_coverage.setdefault(
                fld, {"field": fld, "count": 0, "group": c.get("group"),
                      "n_superseded": 0})
            fc["count"] += 1
            if c.get("supersedes"):
                fc["n_superseded"] += 1
        field_coverage = sorted(
            field_coverage.values(), key=lambda d: (-d["count"], d["field"]))

        conflicts_doc = STORE.lit_conflicts or {}
        validation_doc = STORE.lit_validation or {}
        # prefer the module-level honesty block from either doc (identical content).
        honesty = (conflicts_doc.get("honesty")
                   or validation_doc.get("honesty") or {})

        return {
            "available": True,
            "n_claims": len(claims),
            "n_papers": len(claims_by_paper),
            "n_superseded": sum(1 for c in claims if c.get("supersedes")),
            "claims_by_paper": claims_by_paper,
            "field_coverage": field_coverage,
            "conflicts": conflicts_doc.get("conflicts", []),
            "n_conflicts": len(conflicts_doc.get("conflicts", []) or []),
            "conflicts_version": conflicts_doc.get("version"),
            # CPAD-validation: extraction accuracy MEASURED against the curated gold.
            "validation": validation_doc.get("validation", {}),
            "honesty": honesty,
            "engine_build_id": STORE.engine_build_id,
            "attribution": STORE._attribution(),
        }
    except Exception as exc:  # pragma: no cover - defensive, never 500
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}",
                "attribution": STORE._attribution()}


def api_pooling(protein_id):
    """GET /api/pooling/{protein_id}: the deterministic empirical-Bayes partial-
    pooling view for a protein (PRISE_DESIGN.md §4 spine). Returns the per-curve
    shrunk t50 records (only those the pooling actually MOVED are interesting), the
    replicate random-effects meta-analyses + stochastic-nucleation signals, and an
    honest explainer. Degrades gracefully: {available: False} if pooling.json was
    not built or the protein has no pooled curves."""
    pool = STORE.pooling or {}
    if not pool or pool.get("status") != "ok":
        return {
            "available": False,
            "reason": ("pooling.json not found — run `python engine/pooling.py` "
                       "to build the empirical-Bayes partial-pooling layer"),
            "attribution": STORE._attribution(),
        }
    entry = STORE.pooling_by_protein.get(protein_id)
    if not entry:
        return {"available": False, "not_found": True,
                "reason": f"no pooled curves for protein: {protein_id}",
                "attribution": STORE._attribution()}
    records = entry.get("records", [])
    # surface which curves the pooling MOVED (shrunk toward the stratum mean)
    moved = [r for r in records
             if "stratum_too_small" not in (r.get("flags") or [])
             and (r.get("moved") or 0.0) > 1e-6]
    moved.sort(key=lambda r: -(r.get("moved") or 0.0))
    return {
        "available": True,
        "protein_id": protein_id,
        "pooling_version": pool.get("pooling_version"),
        "target": pool.get("target"),
        "scale": pool.get("scale"),
        "n_curves": len(records),
        "n_moved": len(moved),
        "curve_records": records,
        "moved": moved,
        "replicate_meta": entry.get("replicates", []),
        "weight_formula": "shrinkage_weight = τ²/(τ²+s²)",
        "explainer": ("Empirical-Bayes partial pooling (deterministic): shrinks a "
                      "noisy single-curve estimate toward its condition+assay-matched "
                      "stratum mean; weight = τ²/(τ²+s²). This is closed-form EB "
                      "shrinkage at the feature level (no MCMC), NOT a full Bayesian "
                      "latent-rate hierarchy."),
        "honesty": pool.get("honesty"),
        "attribution": STORE._attribution(),
    }


def api_meta(protein_id):
    """GET /api/meta/{protein_id}: the M14 cross-study hierarchical meta-analysis
    record for a protein. For a protein with >=2 studies returns the full record
    (forest_plot rows + pooled diamond + prediction interval, heterogeneity Q/I²/τ²,
    variance_decomposition with the lab-confounded / batch-unidentifiable honesty,
    leave_one_study_out, publication_bias funnel + Egger, moderators, honesty). For a
    single-study or absent protein returns {available: True, status: 'single_study'|...}
    with the honest note (NEVER a 500). Wrapped in try/except."""
    try:
        rec = STORE.meta_by_protein.get(protein_id)
        if rec:
            out = dict(rec)
            out["available"] = True
            out["engine_build_id"] = STORE.engine_build_id
            out["attribution"] = STORE._attribution()
            return out
        # single-study (or otherwise not-pooled) protein: honest one-liner, not a 500
        single = STORE.meta_single_by_protein.get(protein_id)
        honesty = (STORE.meta_rollup.get("honesty")
                   if isinstance(STORE.meta_rollup, dict) else None)
        if single:
            return {
                "available": True,
                "protein": protein_id,
                "status": single.get("status", "single_study"),
                "n_studies": single.get("n_studies", 1),
                "n_curves": single.get("n_curves"),
                "note": single.get("note",
                                   "Single study (n=1) — no cross-study pooling "
                                   "possible; reported as a single study with NO "
                                   "fabricated pooling (§3-M14 honesty)."),
                "honesty": single.get("honesty", honesty),
                "attribution": STORE._attribution(),
            }
        # protein not present in the M14 corpus at all (or artifacts not built)
        if not STORE.meta_by_protein and not STORE.meta_single_by_protein:
            return {
                "available": False,
                "protein": protein_id,
                "reason": ("meta_analysis artifacts not found — run "
                           "`python engine/m14_metaanalysis.py` to build the M14 "
                           "cross-study meta-analysis layer"),
                "attribution": STORE._attribution(),
            }
        return {
            "available": True,
            "protein": protein_id,
            "status": "single_study",
            "n_studies": 1,
            "note": ("Single study (n=1) — no cross-study pooling possible; this "
                     "protein is not in the multi-study meta-analysis set (§3-M14 "
                     "honesty)."),
            "honesty": honesty,
            "attribution": STORE._attribution(),
        }
    except Exception as exc:  # pragma: no cover - defensive, never 500
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}",
                "protein": protein_id, "attribution": STORE._attribution()}


def _resolve_uniprot(protein_id):
    """protein_id -> uniprot_id, reusing the same M7 index protein_detail uses.
    Returns None if the protein is unknown or carries no uniprot on any unit."""
    sids = STORE.series_by_protein.get(protein_id)
    if not sids:
        return None
    for s in sids:
        uni = (STORE.results_by_series.get(s, {}) or {}).get("uniprot_id")
        if uni:
            return uni
    return None


def _structure_corpus_block():
    """The corpus-level structure->kinetics ASSOCIATION table (identical across all
    proteins — it is a corpus result, not per-protein). Association-only, never
    causal; every card carries causal:false + a verdict. Small; assembled once per
    request from the dedicated associations artifact (falls back to the copy embedded
    in structure_features.json)."""
    doc = STORE.structure_assoc or {}
    assoc = doc.get("associations") or {}
    if not assoc.get("association_cards"):
        # fall back to the copy embedded in the features doc
        fdoc = STORE.structure_features_doc or {}
        assoc = fdoc.get("associations") or {}
        doc = fdoc or doc
    cards = assoc.get("association_cards", []) or []
    return {
        "n_cards": assoc.get("n_cards", len(cards)),
        "fdr_alpha": assoc.get("fdr_alpha"),
        "n_significant_after_fdr": assoc.get("n_significant_after_fdr"),
        "verdict_counts": assoc.get("verdict_counts", {}),
        "causal_invariant": assoc.get("causal_invariant"),
        "association_cards": cards,
        "coverage": doc.get("coverage", {}),
        "assoc_honesty": assoc.get("honesty"),
    }


def api_structure(protein_id):
    """GET /api/structure/{protein_id}: the M16 sequence<->structure<->kinetics bridge
    for one protein.

    CORE FRAMING (honesty-first): each of the 6 genuinely-3D features is a NESTED
    block {proxy, has_real_3d[, real_3d]} — the 1-D sequence/APR PROXY is ALWAYS
    present (derivation=sequence_apr_proxy, is_3d_derived=false), and a REAL 3D value
    (derivation=pdb_3d, is_3d_derived=true, per-feature `fidelity`) is added ONLY when
    a PDB was actually fetched + parsed for this protein. `has_real_3d_features` /
    `real_3d_pdb_id` / `real_3d_native_or_fibril` stamp that at the container level.
    A protein with no PDB keeps proxies only, FLAGGED (has_real_3d=false, NO real_3d
    key — never a fabricated 3D number). A native-monomer structure is NOT the
    aggregation-competent/fibril state; the structure->kinetics associations are
    STATISTICAL ONLY, NEVER causal (effective n <= 26 proteins -> mostly
    underpowered/null); the GNN graph is a forward-compatible REPRESENTATION for a
    future GNN (no GNN is built).

    Resolves protein_id -> uniprot (reusing the M7 index), then returns
    {available, uniprot, protein_name, n_aprs, apr_proxy_features, structural_features
    (the 8 features with honest status), graph_summary, structure_sources, the corpus
    association table (protein-level), variance_ceiling, honesty}. Never 500: an
    absent uniprot / missing features -> {available: False, note}. Wrapped."""
    try:
        fdoc = STORE.structure_features_doc or {}
        if not STORE.structure_by_uniprot and not fdoc:
            return {
                "available": False,
                "protein_id": protein_id,
                "note": ("structure_features.json not found — run "
                         "`python engine/m16_structure.py` to build the M16 "
                         "sequence<->structure<->kinetics bridge"),
                "attribution": STORE._attribution(),
            }
        uni = _resolve_uniprot(protein_id)
        # the corpus association table is a corpus result — surface it regardless so
        # even a protein with no APR-proxy features still sees the (all-null) table.
        corpus = _structure_corpus_block()
        honesty = (fdoc.get("honesty_constraints")
                   or (STORE.structure_assoc or {}).get("honesty_constraints") or [])
        variance_ceiling = (fdoc.get("variance_ceiling_log10_t50")
                            or (STORE.structure_assoc or {})
                            .get("variance_ceiling_log10_t50") or {})
        rec = STORE.structure_by_uniprot.get(uni) if uni else None
        if not rec:
            return {
                "available": False,
                "protein_id": protein_id,
                "uniprot": uni,
                "note": ("no APR/sequence structural-proxy features for this protein "
                         + (f"(uniprot {uni})" if uni else "(no uniprot resolved)")
                         + " — it is not in the M16 structural-feature set (of "
                         "382 sequences, 26 proteins in the kinetics corpus carry "
                         "APR-proxy features). The structure->kinetics associations "
                         "below are a CORPUS result (association-only, never causal)."),
                "association_table": corpus,
                "variance_ceiling": variance_ceiling,
                "honesty": honesty,
                "attribution": STORE._attribution(),
            }
        sf = rec.get("structural_features", {}) or {}
        return {
            "available": True,
            "protein_id": protein_id,
            "uniprot": uni,
            "protein_name": rec.get("protein_name"),
            "n_aprs": rec.get("n_aprs"),
            "has_apr_sequence": rec.get("has_apr_sequence"),
            # the APR/sequence PROXY features (β-propensity, hydrophobicity, charge,
            # the REAL aggregation hotspots) — this block is the 1-D sequence layer
            # ONLY: stamped derivation=sequence_apr_proxy, is_3d_derived=false. The
            # REAL 3D measurements (when a PDB exists) live in structural_features.
            "apr_proxy_features": rec.get("apr_proxy_features", {}),
            # the 8 structural features. The 6 genuinely-3D ones are NESTED blocks
            # {proxy, has_real_3d[, real_3d]} — proxy always present and stamped
            # is_3d_derived=false; real_3d present ONLY with a parsed PDB, stamped
            # derivation=pdb_3d / is_3d_derived=true + a per-feature `fidelity`.
            # contact_map is intrinsically 3D (no proxy): REAL or computable=false.
            "structural_features": {
                "derivation_summary": sf.get("derivation_summary"),
                # container-level REAL-3D stamp: did a PDB fetch+parse succeed, which
                # PDB, and is it a native monomer or a fibril (native != fibril).
                "has_real_3d_features": bool(sf.get("has_real_3d_features")),
                "real_3d_pdb_id": sf.get("real_3d_pdb_id"),
                "real_3d_native_or_fibril": sf.get("real_3d_native_or_fibril",
                                                   "unknown"),
                "aggregation_hotspots": sf.get("aggregation_hotspots"),
                "beta_sheet_content": sf.get("beta_sheet_content"),
                "solvent_accessibility": sf.get("solvent_accessibility"),
                "hydrophobic_patches": sf.get("hydrophobic_patches"),
                "electrostatic_surface": sf.get("electrostatic_surface"),
                "secondary_structure": sf.get("secondary_structure"),
                "contact_map": sf.get("contact_map"),
                "surface_curvature": sf.get("surface_curvature"),
            },
            # the GNN-ready graph SUMMARY (no node/edge dump — a representation only)
            "graph_summary": _graph_summary(rec.get("graph", {}) or {}),
            # PDB/AlphaFold/CryoEM/NMR refs with native-vs-fibril
            "structure_sources": sf.get("structure_sources", {}),
            # the corpus structure->kinetics association table (protein-level)
            "association_table": corpus,
            "variance_ceiling": variance_ceiling,
            "honesty": honesty,
            "engine_build_id": STORE.engine_build_id,
            "attribution": STORE._attribution(),
        }
    except Exception as exc:  # pragma: no cover - defensive, never 500
        return {"available": False, "protein_id": protein_id,
                "note": f"{type(exc).__name__}: {exc}",
                "attribution": STORE._attribution()}


def _graph_summary(graph):
    """Compact GNN-graph summary (NO node/edge dump — the graph is a forward-compatible
    REPRESENTATION for a future GNN; no GNN is built)."""
    return {
        "n_nodes": graph.get("n_nodes"),
        "n_edges": graph.get("n_edges"),
        "native_or_fibril": graph.get("native_or_fibril"),
        "structure_id": graph.get("structure_id"),
        "edge_types_present": graph.get("edge_types_present", []),
        "contact_edges_available": graph.get("contact_edges_available"),
        "node_feature_schema": graph.get("node_feature_schema", []),
        "note": graph.get("note"),
    }


def _conformal_for_curve(t50_hat, t50_status, censoring_class, regime_scores):
    """Apply the frozen conformal calibration to ONE live curve. Returns
    {t50_interval, regime_set} (or {available: False}) so /api/analyze can show the
    finite-sample-valid interval + prediction set alongside the point t50 / regime.
    Best-effort: needs the frozen artifact AND the pure application functions; any
    gap -> {available: False}, never a 500."""
    cf = STORE.conformal or {}
    quant = cf.get("calibration_quantiles")
    if not quant or _conformal is None:
        return {"available": False,
                "reason": "conformal calibration not loaded on this host"}
    out = {"available": True,
           "target_coverage": cf.get("target_coverage"),
           "version": cf.get("version")}
    try:
        # the stratum is (censoring | assay_endpoint | regime); the caller passes the
        # already-resolved key parts via a dict in regime_scores["_stratum"] if present
        stratum = (regime_scores or {}).get("_stratum")
        if stratum is None:
            # fall back to a marginal-only lookup (no stratum match) — still valid
            stratum = "|||"
        scores = {k: v for k, v in (regime_scores or {}).items()
                  if not k.startswith("_")}
        # conformal_t50_interval handles left-censoring suppression + one-sided
        # right-censoring internally from the censoring arg; we just hand it the point.
        out["t50_interval"] = _conformal.conformal_t50_interval(
            t50_hat, stratum, censoring_class or "none", quant)
        out["regime_set"] = _conformal.conformal_regime_set(
            scores, quant, regime_stratum=stratum)
    except Exception as exc:   # pragma: no cover - defensive
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return out


def api_proteins(q=None):
    lst = STORE.protein_list(q)
    return {"n": len(lst), "proteins": lst, "attribution": STORE._attribution()}


def api_protein(pid):
    detail = STORE.protein_detail(pid)
    if detail is None:
        return None
    return detail


def api_curve(series_id):
    """SINGLE curve only: x/y + M1 triage summary + fitted-model overlay +
    attribution. No bulk dump."""
    cur = STORE.curve_by_series.get(series_id)
    if cur is None:
        return None
    m1 = cur.get("m1", {})
    res = STORE.results_by_series.get(series_id, {})

    x = cur.get("x_hours") or []
    y = cur.get("y_intensity") or []
    y_proc = m1.get("y_processed")

    # fitted overlay: prefer recomputing via the engine (authoritative), else
    # fall back to the model name carried in the M7 result (no overlay curve).
    best_model = (res.get("model_selection") or {}).get("best_point") \
        or (res.get("model_selection") or {}).get("best_descriptive")
    overlay = None
    overlay_model = best_model
    # PREFER THE PUBLISHED FIT. This used to call _m2.fit_curve() on every
    # request, which refit all 13 models just to draw ONE overlay line -- 29 s per
    # curve once M2 gained genuine multi-starts. The same fit is already in
    # fits.jsonl, so serving it is both ~1000x faster and MORE faithful: the line
    # drawn is now exactly the published fit rather than a re-derivation that
    # could differ if the serving host's BLAS differs. Recomputation remains the
    # fallback for a series the artifact does not cover.
    stored = STORE.fits_by_series.get(series_id) or {}
    fit = stored if stored.get("fits") else None
    if fit is None and ENGINE_OK:
        try:
            fit = _m2.fit_curve(dict(cur))
        except Exception:
            fit = None
    if fit:
        try:
            bm = fit.get("best_by_aicc")
            if bm and bm in (fit.get("fits") or {}):
                params = (fit["fits"][bm] or {}).get("params", {})
                grid = _dense_grid([float(v) for v in x])
                yhat = predict_model(bm, params, grid)
                if yhat is not None:
                    overlay = {"model": bm, "x": grid, "y": yhat,
                               "r2": fit["fits"][bm].get("r2")}
                    overlay_model = bm
        except Exception:
            overlay = None

    # censored region (for shading): right-censored => after last point still
    # rising; left-censored => before first point. We surface the censoring class
    # and the observed window so the UI can badge/shade it.
    cclass = m1.get("censoring_class") or cur.get("censoring_class")

    return {
        "series_id": series_id,
        "protein_id": cur.get("protein_id"),
        "uniprot_id": cur.get("uniprot_id"),
        "x": [float(v) for v in x],
        "y": [float(v) for v in y],
        "y_processed": [float(v) for v in y_proc] if y_proc else None,
        "units": cur.get("units", {}),
        "signal_basis": m1.get("signal_basis"),
        "triage": {
            "fittability_class": m1.get("fittability_class") or cur.get("fittability_class"),
            "censoring_class": cclass,
            "recommended_handling": m1.get("recommended_handling"),
            "normalization_mode": m1.get("normalization_mode"),
            "plateau_reached": m1.get("plateau_reached"),
            "reasons": m1.get("reasons", []),
        },
        "fitted": overlay,
        "fitted_model_name": overlay_model,
        "t50_status": (res.get("curve_features") or {}).get("t50_status"),
        "condition_vector": cur.get("condition_vector", {}),
        "attribution": STORE._attribution(),
    }


# --------------------------------------------------------------------------- #
# Model comparison / behaviour classification (per selected curve)
# --------------------------------------------------------------------------- #
# Honest per-family notes surfaced in the payload (frontend renders these as a
# legend). These keep the comparison honesty-first: a high R² on a Tier-B/
# mechanistic fit is NOT a mechanism call.
MODEL_FAMILY = {
    "lnt": "descriptive", "logistic": "descriptive", "gompertz": "descriptive",
    "richards": "descriptive", "exponential": "descriptive",
    "scaling_law": "descriptive",
    "brain_cousens": "biphasic",
    "finke_watzky": "autocatalytic",
    # mechanistic variants are added dynamically (family 'mechanistic')
}

MODEL_NOTES = {
    "lnt": "linear no-threshold (a straight line; no cooperative onset).",
    "logistic": "symmetric sigmoid — cooperative / nucleation-dependent shape.",
    "gompertz": "asymmetric sigmoid — cooperative / nucleation-dependent shape.",
    "richards": "generalised-logistic sigmoid — cooperative / nucleation-dependent shape.",
    "exponential": "downhill saturation — no lag, no cooperative onset.",
    "scaling_law": "power-law growth (c + a·tⁿ); the exponent is descriptive, not a rate.",
    "brain_cousens": "biphasic / hormesis — natively a DOSE model, shown here on the "
                     "time axis for comparison only.",
    "finke_watzky": "2-step autocatalytic (slow nucleation + autocatalysis); its k1,k2 "
                    "are LUMPED constants, NOT amyloid nucleation/elongation rates.",
}

FAMILY_NOTES = {
    "descriptive": "Descriptive sigmoids (logistic/Gompertz/Richards) = "
                   "cooperative, nucleation-dependent curve SHAPE — not a mechanism.",
    "biphasic": "Brain–Cousens = biphasic/hormesis dose model, shown on the time "
                "axis for comparison.",
    "autocatalytic": "Finke–Watzky = 2-step autocatalytic; its constants are NOT "
                     "amyloid nucleation rates.",
    "mechanistic": "Tier-B (Knowles/Cohen nucleation) models are SLOPPY from a single "
                   "curve — rates are not uniquely identifiable. A high R² here is NOT "
                   "a mechanism call; M5 still refuses mechanism.",
}

# closed-form REGISTRY models we fit explicitly for the comparison (do NOT rely on
# M1 routing, which omits brain_cousens on monotonic curves)
_COMPARISON_REGISTRY_MODELS = [
    "lnt", "logistic", "gompertz", "richards", "exponential",
    "scaling_law", "brain_cousens", "finke_watzky",
]


def _mech_predict(variant, params, nc, n2, xs):
    """Evaluate a fitted mechanistic variant on xs. Returns list[float] or None."""
    if not ENGINE_OK or _mech is None or not isinstance(params, dict):
        return None
    try:
        frac = _mech.simulate_mass_fraction(
            list(xs),
            kn=params.get("kn", 0.0), kp=params.get("kp", 0.0),
            k2=params.get("k2", 0.0), kminus=params.get("kminus", 0.0),
            KM=params.get("KM", float("inf")), nc=nc, n2=n2, mtot=1.0)
        base = float(params.get("base", 0.0))
        amp = float(params.get("amp", 1.0))
        return [float(base + amp * f) for f in frac]
    except Exception:
        return None


# Per-series Tier-B mechanistic fits, memoised. Deterministic given the curve, so
# caching cannot change an answer -- it only stops the server refitting four stiff
# ODEs every time a user revisits a curve. Cleared wholesale when full rather than
# LRU-evicted: the working set is a browsing session's worth of curves, and a
# clear-and-refill is simpler to reason about than an eviction policy nobody will
# tune.
_TIERB_CACHE = {}
_TIERB_CACHE_MAX = 512


def api_models(series_id):
    """GET /api/models/{series_id}: fit the FULL candidate bank to the stored
    triaged curve and return a ranked behaviour-classification comparison.

    Wrapped so any failure returns {"error": ...} rather than a 500."""
    if not ENGINE_OK:
        return {"error": f"engine unavailable on this host: {ENGINE_ERR}"}
    cur = STORE.curve_by_series.get(series_id)
    if cur is None:
        return {"error": f"curve not found: {series_id}", "not_found": True}
    try:
        m1 = cur.get("m1", {})
        x = [float(v) for v in (cur.get("x_hours") or [])]
        y_src = m1.get("y_processed") or cur.get("y_intensity") or []
        y = [float(v) for v in y_src]
        signal_used = "m1.y_processed" if m1.get("y_processed") else "y_intensity"
        if len(x) != len(y) or len(x) < 3:
            return {"error": "curve has too few / malformed points to fit a model bank",
                    "series_id": series_id}

        models = []          # converged comparison rows
        non_converged = []   # names that did not converge (flagged honestly)

        # ---- DESCRIPTIVE + biphasic + autocatalytic (REGISTRY) ----
        # PREFER THE PUBLISHED FITS. This loop used to call _m2.fit_one() per
        # model on every request -- 38 s per curve once M2 gained genuine
        # multi-starts. fits.jsonl already holds every one of these fits for
        # every series, so the table is now READ rather than recomputed, which is
        # also the only way to guarantee the ranking shown matches the ranking the
        # engine published. A model missing from the artifact still falls back to
        # fitting it live, so a partial artifact degrades instead of hiding rows.
        stored_fits = ((STORE.fits_by_series.get(series_id) or {}).get("fits")
                       or {})
        # M2 publishes only the CANDIDATE SET it selected for this curve's shape
        # (a monotonic_fit curve gets no brain_cousens; a nonmonotonic_fit curve
        # gets no sigmoidal models). This endpoint deliberately shows the FULL
        # bank so the excluded models can be seen ranked alongside, so the rest
        # come from the gap-fill artifact rather than a live refit -- 9.6 s on
        # CPAD-TK-2185, and every one of the 1654 curves needs at least one.
        gapfill = ((STORE.registry_gapfill_by_series.get(series_id) or {})
                   .get("fits") or {})
        for name in _COMPARISON_REGISTRY_MODELS:
            r = stored_fits.get(name) or gapfill.get(name)
            if not r:
                r = _m2.fit_one(x, y, name)
            if r.get("converged") and r.get("aicc") is not None:
                models.append({
                    "name": name,
                    "family": MODEL_FAMILY.get(name, "descriptive"),
                    "n_params": r.get("n_params"),
                    "r2": r.get("r2"),
                    "aicc": r.get("aicc"),
                    "params": r.get("params") or {},
                    "kind": "registry",
                    "note": MODEL_NOTES.get(name, ""),
                })
            else:
                non_converged.append({"name": name, "family": MODEL_FAMILY.get(name),
                                      "reason": r.get("reason", "no_convergence")})

        # ---- TIER-B mechanistic (Knowles/Cohen) ----
        # These are NOT in any artifact -- no module publishes per-curve
        # mechanistic fits -- so they are the one thing still computed live, and
        # at ~1.2 s for four stiff-ODE fits they dominate what is left of this
        # request. Memoised per series because the fit is a deterministic
        # function of (x, y): revisiting a curve, or flipping between curves and
        # back, is then free. Bounded so a long session cannot grow without limit.
        # PREFER THE PUBLISHED BANK (mechanistic_fits.jsonl). Fitting these four
        # per request measured median 0.29 s but max 114 s (CPAD-TK-2185, 22
        # points), and the container runs on 0.1 CPU -- a tail curve never
        # finished in a browser. The in-memory memo below now serves only the
        # fallback path; it is also why a protein used to feel fast on a second
        # visit and slow on the first, and slow again once the cache cleared.
        published = (STORE.mech_fits_by_series.get(series_id) or {}).get("fits")
        if published:
            cached = published
        else:
            cached = _TIERB_CACHE.get(series_id)
            if cached is None:
                cached = {}
                for variant in _mech.VARIANT_RATES:
                    cached[variant] = _mech.fit_mechanistic(x, y, variant=variant)
                if len(_TIERB_CACHE) >= _TIERB_CACHE_MAX:
                    _TIERB_CACHE.clear()      # simplest bounded policy; see note
                _TIERB_CACHE[series_id] = cached
        for variant in _mech.VARIANT_RATES:
            r = cached.get(variant) or _mech.fit_mechanistic(x, y, variant=variant)
            if r.get("converged") and r.get("aicc") is not None:
                models.append({
                    "name": variant,
                    "family": "mechanistic",
                    "n_params": r.get("n_params"),
                    "r2": r.get("r2"),
                    "aicc": r.get("aicc"),
                    "params": r.get("params") or {},
                    "kind": "mechanistic",
                    "nc": r.get("nc"), "n2": r.get("n2"),
                    "note": "Tier-B Knowles/Cohen nucleation model — sloppy from a single "
                            "curve; rates not uniquely identifiable (NOT a mechanism call).",
                })
            else:
                non_converged.append({"name": variant, "family": "mechanistic",
                                      "reason": r.get("reason", "no_convergence")})

        if not models:
            return {"error": "no model converged on this curve",
                    "series_id": series_id, "non_converged": non_converged}

        # ---- ranking: ΔAICc + Akaike weights over the converged set ----
        aiccs = [m["aicc"] for m in models]
        amin = min(aiccs)
        # akaike weights (m3_select form) over converged comparison models
        dw = {m["name"]: math.exp(-0.5 * (m["aicc"] - amin)) for m in models}
        zw = sum(dw.values()) or 1.0
        for m in models:
            m["delta_aicc"] = m["aicc"] - amin
            m["akaike_weight"] = dw[m["name"]] / zw
        models.sort(key=lambda m: m["aicc"])
        for i, m in enumerate(models):
            m["winner"] = (i == 0)
            m["rank"] = i + 1

        winner = models[0]

        # ---- dense fitted curves for the top ~6 models (frontend overlay) ----
        grid = _dense_grid(x)
        for m in models[:6]:
            if m["kind"] == "registry":
                yg = predict_model(m["name"], m["params"], grid)
            else:
                yg = _mech_predict(m["name"], m["params"], m.get("nc", 2.0),
                                   m.get("n2", 2.0), grid)
            if yg is not None:
                m["x_grid"] = grid
                m["y_grid"] = yg

        # ---- identifiability on the WINNER (REGISTRY models only have an FIM) ----
        identifiability = None
        if winner["kind"] == "registry":
            try:
                identifiability = _m3.fim_identifiability(
                    winner["name"], winner["params"], x, y)
            except Exception as exc:
                identifiability = {"flag": "fim_failed", "error": str(exc)}
        else:
            identifiability = {
                "flag": "not_computed_mechanistic_winner",
                "note": "FIM identifiability here is reported for closed-form REGISTRY "
                        "winners; the Tier-B winner is sloppy by construction (rates "
                        "fit in log-space, not uniquely identifiable).",
            }

        # ---- Feature 3: bootstrap selection stability from the stored M7 result ----
        res = STORE.results_by_series.get(series_id, {})
        ms = res.get("model_selection") or {}
        bootstrap = None
        if ms.get("selection_frequencies") or ms.get("selection_stability") is not None:
            bootstrap = {
                "source": ms.get("source"),
                "selection_frequencies": ms.get("selection_frequencies"),
                "selection_stability": ms.get("selection_stability"),
                "t50_predictive_interval": ms.get("t50_predictive_interval"),
                "t50_multimodal": ms.get("t50_multimodal"),
                "best_point": ms.get("best_point"),
                "note": "M3-sampled bootstrap selection envelope (selection runs inside "
                        "the bootstrap; a build-time job).",
            }
        else:
            bootstrap = {
                "note": "point selection (AICc); bootstrap-enveloped selection is a "
                        "build-time job (M3) and is not available for this curve.",
            }

        return {
            "series_id": series_id,
            "protein_id": cur.get("protein_id"),
            "signal_used": signal_used,
            "n_points": len(x),
            "n_converged": len(models),
            "winner": winner["name"],
            "winner_family": winner["family"],
            "models": models,
            "non_converged": non_converged,
            "identifiability": identifiability,
            "bootstrap_selection": bootstrap,
            "family_notes": FAMILY_NOTES,
            "honesty": "Tier-B mechanistic / Finke–Watzky fits are shown for COMPARISON "
                       "only — a high R² is NOT a licensed mechanism. M5 still refuses a "
                       "mechanism call on a single curve.",
            "engine_build_id": STORE.engine_build_id,
            "attribution": STORE._attribution(),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "series_id": series_id}


# --------------------------------------------------------------------------- #
# Concentration-series scaling (dual-γ) view (per protein with a series)
# --------------------------------------------------------------------------- #
def api_series_gamma(protein_id):
    """GET /api/series-gamma/{protein_id}: the dual-γ payload(s) for a protein's
    concentration series + the per-member (concentration, t50) points for a
    log-log plot. A protein may have >1 series, so this returns a list.

    Wrapped so any failure returns {"error": ...} rather than a 500."""
    gammas = STORE.gamma_by_protein.get(protein_id)
    metas = STORE.conc_series_by_protein.get(protein_id)
    if not gammas and not metas:
        return {"error": f"no concentration series for protein: {protein_id}",
                "not_found": True}
    try:
        # join gamma payloads + membership by concentration_series_id
        out_series = []
        seen = set()
        # iterate over gamma records first (they carry the dual-γ payload)
        for g in (gammas or []):
            csid = g.get("concentration_series_id")
            seen.add(csid)
            meta = STORE.conc_series_by_id.get(csid, {})
            out_series.append(_build_gamma_series(g, meta))
        # include any series with membership but no gamma payload (rare)
        for meta in (metas or []):
            csid = meta.get("concentration_series_id")
            if csid in seen:
                continue
            out_series.append(_build_gamma_series(None, meta))

        return {
            "protein_id": protein_id,
            "n_series": len(out_series),
            "series": out_series,
            "explanation": "γ is the half-time scaling exponent (t50 ∝ [m]^−γ) — a "
                           "mechanistic CONSTRAINT. The two estimators (regression and "
                           "global collapse) are reported SEPARATELY and never merged; "
                           "disagreement means the curve shape is not concentration-"
                           "invariant.",
            "engine_build_id": STORE.engine_build_id,
            "attribution": STORE._attribution(),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "protein_id": protein_id}


def _build_gamma_series(gamma_payload, meta):
    """Assemble one concentration-series entry: the stored dual-γ payload + the
    per-member (concentration, t50, censoring) points for the log-log plot."""
    csid = (gamma_payload or {}).get("concentration_series_id") \
        or (meta or {}).get("concentration_series_id")
    entry = {
        "concentration_series_id": csid,
        "conditions": (gamma_payload or {}).get("conditions")
        or {k: (meta or {}).get(k) for k in ("pH", "temperature_C", "assay", "mutation")},
        "n_member_curves": (gamma_payload or {}).get("n_member_curves")
        or (meta or {}).get("n_curves"),
        "n_distinct_concentrations": (meta or {}).get("n_distinct_concentrations"),
    }
    if gamma_payload:
        entry["gamma_regression"] = gamma_payload.get("gamma_regression")
        entry["gamma_global"] = gamma_payload.get("gamma_global")
        entry["disagreement"] = gamma_payload.get("disagreement")
        entry["mechanism_note"] = gamma_payload.get("mechanism_note")
        entry["n_curves_with_t50"] = gamma_payload.get("n_curves_with_t50")
        # the shared-rate ODE global fit (third γ_mechanistic + identifiability),
        # joined in by M4 when engine/global_fit.py output is present (read-only)
        if gamma_payload.get("global_ode_fit"):
            entry["global_ode_fit"] = gamma_payload.get("global_ode_fit")
        # surface the honesty gate flags at the top level for the display
        reg = gamma_payload.get("gamma_regression") or {}
        dis = gamma_payload.get("disagreement") or {}
        entry["gamma_physical"] = reg.get("gamma_physical")
        entry["shape_not_concentration_invariant"] = dis.get(
            "shape_not_concentration_invariant")
    else:
        entry["gamma_regression"] = {"status": "not_computed"}
        entry["gamma_global"] = {"status": "not_computed"}

    # per-member points for the log10(t50) vs log10(conc) scatter
    points = _series_t50_points(meta)
    entry["points"] = points
    entry["n_points"] = len(points)
    # fitted -γ slope line (from gamma_regression intercept + slope) for the plot
    reg = entry.get("gamma_regression") or {}
    if (reg.get("status") == "ok" and reg.get("gamma") is not None
            and reg.get("intercept") is not None):
        entry["fit_line"] = {
            "intercept_log10t50": reg.get("intercept"),
            "slope": -float(reg.get("gamma")),         # log10 t50 = a - γ·log10 m
            "gamma": reg.get("gamma"),
            "note": "log10(t50) = intercept − γ·log10([m]); slope = −γ.",
        }
    return entry


def _series_t50_rows(meta):
    """The raw (concentration_uM, t50, censoring_class) rows for one series.

    Prefers the PUBLISHED table (series_t50_points.jsonl, built by
    `python engine/m4_features.py --t50-points`). Building it per request meant
    refitting every member curve: measured at 1-30 s per protein locally
    (AL-12: 30.5 s), and the deployment runs on 0.1 CPU.

    Falls back to the live computation when the artifact is absent, so a missing
    artifact degrades to SLOW, never to wrong or empty. The published rows are
    the same function's output, not a cheaper proxy -- the per-curve M4 t50 in
    protein_analysis.jsonl is None for curves this table still estimates, so
    substituting it would silently drop points from the log-log scatter."""
    if not meta:
        return []
    csid = meta.get("concentration_series_id")
    if csid:
        pub = STORE.t50_points_by_csid.get(csid)
        if pub is not None:
            return pub.get("rows") or []
    if not ENGINE_OK or _m4 is None:
        return []
    members = [STORE.curve_by_series.get(sid)
               for sid in (meta.get("member_series_ids") or [])]
    members = [m for m in members if m is not None]
    if not members:
        return []
    try:
        return _m4.series_t50_table(members)
    except Exception:
        return []


def _series_t50_points(meta):
    """Shape the rows for the log-log plot. Shaping is unchanged; only the
    SOURCE of the rows moved from per-request refitting to the published table."""
    rows = _series_t50_rows(meta)
    if not rows:
        return []
    out = []
    for r in rows:
        m = r.get("concentration_uM")
        t = r.get("t50")
        if m is None or t is None or m <= 0 or t <= 0:
            continue
        out.append({
            "series_id": r.get("series_id"),
            "concentration_uM": float(m),
            "t50": float(t),
            "log10_concentration": math.log10(float(m)),
            "log10_t50": math.log10(float(t)),
            "censoring_class": r.get("censoring_class"),
            "right_censored": bool(r.get("right_censored")),
            "left_censored": bool(r.get("left_censored")),
            "model": r.get("model"),
        })
    return out


# --------------------------------------------------------------------------- #
# Dose-response (k_agg vs concentration) view — per protein with CPAD R-rows
# --------------------------------------------------------------------------- #
# Honest per-model notes for the dose-response bank (frontend renders as a legend).
_DOSE_MODEL_NOTES = {
    "dr_lnt": "linear no-threshold (k_agg rises linearly with [monomer]; no onset).",
    "dr_threshold": "hockey-stick: flat below a threshold dose τ, then linear.",
    "dr_hill": "Hill / 4-parameter-logistic sigmoid (cooperative saturation).",
    "dr_brain_cousens": "Brain–Cousens hormesis (a low-dose bump then decline).",
    "dr_scaling": "power-law dose scaling (k_agg ∝ [monomer]ⁿ; n is the exponent).",
}


def api_dose_response(protein_id):
    """GET /api/dose-response/{protein_id}: the dose-response (k_agg vs
    concentration) model-bank fit for a protein's CPAD R-rows. Returns the ranked
    models (R²/AICc + the AICc-best), the raw (concentration, k_agg) points for the
    scatter, and an honest explainer. Degrades gracefully: {available: False} if the
    artifact was not built or the protein has no R-rows.

    Wrapped so any failure returns {"error": ...}/{available: False} rather than a 500."""
    fit = STORE.dose_response_by_protein.get(protein_id)
    if not fit:
        return {
            "available": False,
            "not_found": True,
            "reason": (f"no dose-response (R-row) fit for protein: {protein_id} — "
                       "needs CPAD rate-vs-concentration rows and "
                       "`python engine/m2_fit.py --dose`"),
            "attribution": STORE._attribution(),
        }
    try:
        fits = fit.get("fits", {}) or {}
        best = fit.get("best_by_aicc")
        # ranked table: converged models ascending by AICc (best first)
        ranked = []
        for name, r in fits.items():
            if not r.get("converged") or r.get("aicc") is None:
                continue
            ranked.append({
                "model": name,
                "r2": r.get("r2"),
                "aicc": r.get("aicc"),
                "n_params": r.get("n_params"),
                "params": r.get("params") or {},
                "best": (name == best),
                "note": _DOSE_MODEL_NOTES.get(name, ""),
            })
        ranked.sort(key=lambda m: (m["aicc"] if m["aicc"] is not None else float("inf")))

        # raw (concentration, k_agg) points for the scatter (sorted, de-duplicated
        # to match what m2_fit fit on — set() of (conc, k_agg) pairs)
        raw = STORE.dose_points_by_protein.get(protein_id, [])
        seen = set()
        points = []
        for c, k, pmid in sorted(raw):
            key = (c, k)
            if key in seen:
                continue
            seen.add(key)
            points.append({"concentration_uM": c, "k_agg": k, "pmid": pmid})

        # dense overlay of the AICc-best model over the observed concentration range
        overlay = None
        if best and best in fits and points and ENGINE_OK and _models is not None:
            params = (fits[best] or {}).get("params") or {}
            cs = [p["concentration_uM"] for p in points]
            grid = _dense_grid(cs)
            yhat = predict_model(best, params, grid)
            if yhat is not None:
                overlay = {"model": best, "x": grid, "y": yhat,
                           "r2": (fits[best] or {}).get("r2")}

        return {
            "available": True,
            "protein_id": protein_id,
            "n_points": fit.get("n_points", len(points)),
            "best_by_aicc": best,
            "models": ranked,
            "points": points,
            "overlay": overlay,
            "axis": fit.get("axis", "dose"),
            "explainer": (
                "How the apparent aggregation rate (k_agg) scales with monomer "
                "concentration — a complementary, endpoint-style view to the kinetic "
                "dual-γ. The bank (LNT / threshold / Hill / Brain–Cousens / power-law) "
                "is ranked by AICc; selection here is preliminary (M3 is the principled "
                "selector)."),
            "honesty": (
                "These are CPAD's PRECOMPUTED k_agg values (rate-vs-concentration "
                "R-rows), distinct from the engine's own fitted t50-scaling γ. They are "
                "literature-reported apparent rates aggregated across studies/conditions "
                "(see per-point PMIDs), NOT a single controlled concentration series, so "
                "treat this as a descriptive scaling view, not a mechanism call."),
            "engine_build_id": STORE.engine_build_id,
            "attribution": STORE._attribution(),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "protein_id": protein_id}


def api_cohort(body):
    """POST /api/cohort: {series_ids: [...]} -> M0 cohort_result over those triaged
    series (§3-M0). This is the MULTI-dataset entry point: the user selects one OR
    several real CPAD datasets for one protein and M0 gates comparability, detects the
    data_mode from what varies, and routes to the licensed analysis.

    Wrapped so any failure returns {"error": ...} rather than a 500. The (slow)
    shared-rate ODE global fit is gated off by default for web latency; the dual-γ
    still gives the concentration_series mechanism-constraint result — the client can
    request it with {run_global_ode: true}."""
    if not ENGINE_OK or _m0 is None:
        return {"error": f"engine unavailable on this host: {ENGINE_ERR}"}
    try:
        ids = body.get("series_ids")
        if not isinstance(ids, list) or not ids:
            return {"error": "series_ids must be a non-empty list of series ids"}
        if len(ids) > 200:
            return {"error": "too many series selected (limit 200)"}

        members = []
        missing = []
        for sid in ids:
            cur = STORE.curve_by_series.get(sid)
            if cur is None:
                missing.append(sid)
            else:
                members.append(cur)
        if not members:
            return {"error": "none of the requested series were found",
                    "missing_series_ids": missing}

        # If every member belongs to the SAME known concentration-series, hand M0 that
        # series' identity metadata (protein / pH / temp / assay) so the dual-γ +
        # global-fit payloads are correctly labelled; else M0 derives it from members.
        series_meta = None
        csids = {c.get("concentration_series_id") for c in members}
        csids.discard(None)
        if len(csids) == 1:
            series_meta = STORE.conc_series_by_id.get(next(iter(csids)))

        run_ode = bool(body.get("run_global_ode", False))
        # web latency: a lighter dual-γ bootstrap budget (point γ unchanged, only CI
        # resolution). A big series with B=200 is too slow for an interactive POST.
        result = _m0.assemble_cohort(members, series_meta=series_meta,
                                     run_global_ode=run_ode, b_reg=60, b_glob=30)
        result["missing_series_ids"] = missing
        result["engine_build_id"] = STORE.engine_build_id
        result["attribution"] = STORE._attribution()
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def api_analyze(body):
    """POST: {x, y, concentration_uM?, assay?} -> compact live-analysis result.

    Wrapped so a bad curve returns {"error": ...} instead of a 500.
    """
    if not ENGINE_OK:
        return {"error": f"engine unavailable on this host: {ENGINE_ERR}"}
    try:
        x = body.get("x")
        y = body.get("y")
        if not isinstance(x, list) or not isinstance(y, list):
            return {"error": "x and y must be lists of numbers"}
        try:
            x = [float(v) for v in x]
            y = [float(v) for v in y]
        except Exception:
            return {"error": "x and y must contain only numbers"}
        if len(x) != len(y):
            return {"error": f"x and y length mismatch ({len(x)} vs {len(y)})"}
        if len(x) < 3:
            return {"error": "need at least 3 (time, signal) points"}
        if len(x) > 5000:
            return {"error": "too many points (limit 5000)"}

        assay = body.get("assay") or "ThT"
        conc = body.get("concentration_uM")
        cv = {
            "assay_type": assay,
            "assay_reports_mass": assay in ("ThT", "ThS", "Congo Red", "Cytofluor"),
            "agitation": None,
            "seeded": None,
            "construct_id": "user-upload",
        }
        if conc is not None:
            try:
                cv["concentration"] = {"value_uM": float(conc)}
            except Exception:
                pass

        series = {
            "series_id": "user-upload",
            "x_hours": x,
            "y_intensity": y,
            "digitization_uncertainty": False,
            "condition_vector": cv,
        }
        # M1 triage -> M2 fit -> M4 features -> M5 classify
        series = _m1.triage_series(series)
        fit = _m2.fit_curve(series)
        feat = _m4.extract_features(series, fit_result=fit)
        cls = _m5.classify_curve(series, fit_result=fit)

        best = fit.get("best_by_aicc")
        overlay = None
        if best and best in fit.get("fits", {}):
            params = fit["fits"][best].get("params", {})
            grid = _dense_grid(x)
            yhat = predict_model(best, params, grid)
            if yhat is not None:
                overlay = {"model": best, "x": grid, "y": yhat,
                           "r2": fit["fits"][best].get("r2")}

        desc = cls.get("descriptive_regime", {}) or {}
        mech = cls.get("mechanistic", {}) or {}
        feats = feat.get("features", {}) or {}

        # ---- conformal prediction (finite-sample-valid t50 interval + regime set) ----
        # Apply the frozen Service-C calibration to THIS curve using the same engine
        # tags the calibration was Mondrian-conditioned on (censoring, assay endpoint,
        # descriptive regime). Best-effort: degrades to {available: False}.
        conformal_out = {"available": False, "reason": "conformal not loaded"}
        if _conformal is not None and STORE.conformal:
            try:
                cens = series["m1"].get("censoring_class", "none")
                assay_endpoint = ("amyloid" if cv.get("assay_reports_mass")
                                  else "generic")
                regime_tag = desc.get("regime", "unknown")
                stratum = _conformal._stratum_key(cens, assay_endpoint, regime_tag)
                rscores = _conformal._regime_scores(series, fit)
                rscores["_stratum"] = stratum
                conformal_out = _conformal_for_curve(
                    feats.get("t50"), feat.get("t50_status"), cens, rscores)
            except Exception as exc:   # pragma: no cover - defensive
                conformal_out = {"available": False,
                                 "reason": f"{type(exc).__name__}: {exc}"}

        # a lightweight "information yield"-ish badge for the live curve
        if not best:
            yield_tag = "signal_only"
        elif feats.get("has_interior_inflection"):
            yield_tag = "descriptive (cooperative)"
        else:
            yield_tag = "descriptive"

        return {
            "ok": True,
            "regime": desc.get("regime"),
            "regime_definition": desc.get("definition"),
            "information_yield": yield_tag,
            "confidence": cls.get("confidence", {}),
            "m1": {
                "fittability_class": series["m1"].get("fittability_class"),
                "censoring_class": series["m1"].get("censoring_class"),
                "recommended_handling": series["m1"].get("recommended_handling"),
                "normalization_mode": series["m1"].get("normalization_mode"),
            },
            "best_model": best,
            "model_r2": fit.get("fits", {}).get(best, {}).get("r2") if best else None,
            "features": {
                "t50": feats.get("t50"),
                "t50_status": feat.get("t50_status"),
                "lag_time": feats.get("lag_time"),
                "lag_status": feat.get("lag_status"),
                "lag_to_t50_ratio": feats.get("lag_to_t50_ratio"),
                "inflection_time": feats.get("inflection_time"),
                "transition_sharpness": feats.get("transition_sharpness"),
                "has_interior_inflection": feats.get("has_interior_inflection"),
            },
            "mechanistic": {
                "licensed": mech.get("mechanistic_inference_licensed"),
                "equivalence_class": mech.get("equivalence_class"),
                "gates_failed": mech.get("gates_failed", []),
                "degeneracy": mech.get("degeneracy"),
            },
            "conformal": conformal_out,
            "fitted": overlay,
            "x": x,
            "y": y,
            "y_processed": series["m1"].get("y_processed"),
            "caveats": _analyze_caveats(series, feat, mech),
            "engine_build_id": STORE.engine_build_id,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _analyze_caveats(series, feat, mech):
    out = []
    if not mech.get("mechanistic_inference_licensed", False):
        gates = mech.get("gates_failed", [])
        out.append("mechanistic inference not licensed" +
                   (f" ({', '.join(gates)})" if gates else ""))
    ts = feat.get("t50_status")
    if ts and ts != "point":
        out.append(f"t50 is censored/biased ({ts}); treat as a bound, not a point")
    if not series["m1"].get("y_processed"):
        out.append("no plateau validated; min-max normalization withheld")
    out.append("live single-curve fit: a γ scaling exponent needs a "
               "≥3-concentration series (single curve cannot resolve mechanism)")
    return out


# --------------------------------------------------------------------------- #
# HTTP handler / router
# --------------------------------------------------------------------------- #
MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "PRISE/1.0"
    protocol_version = "HTTP/1.1"

    # -- low-level senders ------------------------------------------------- #
    def _send_json(self, obj, status=200):
        try:
            payload = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        except Exception as exc:
            payload = json.dumps({"error": f"serialisation: {exc}"}).encode("utf-8")
            status = 500
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_file(self, path):
        if not os.path.isfile(path):
            return self._send_json({"error": "not found"}, 404)
        ext = os.path.splitext(path)[1].lower()
        ctype = MIME.get(ext, "application/octet-stream")
        with open(path, "rb") as fh:
            data = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 8 * 1024 * 1024:  # 8 MB upload cap
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    # -- routing ----------------------------------------------------------- #
    def do_GET(self):
        try:
            self._route_get()
        except Exception:
            self._send_json({"error": "internal error",
                             "trace": traceback.format_exc(limit=2)}, 500)

    do_HEAD = do_GET

    def do_POST(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path in ("/api/analyze", "/api/cohort"):
                try:
                    body = self._read_body()
                except Exception as exc:
                    return self._send_json({"error": f"bad request body: {exc}"}, 400)
                if parsed.path == "/api/cohort":
                    return self._send_json(api_cohort(body))
                return self._send_json(api_analyze(body))
            return self._send_json({"error": "not found"}, 404)
        except Exception:
            self._send_json({"error": "internal error",
                             "trace": traceback.format_exc(limit=2)}, 500)

    def _route_get(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            return self._send_file(os.path.join(HERE, "index.html"))

        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            # prevent path traversal
            safe = os.path.normpath(os.path.join(STATIC, rel))
            if not safe.startswith(STATIC):
                return self._send_json({"error": "forbidden"}, 403)
            return self._send_file(safe)

        if path == "/api/overview":
            return self._send_json(api_overview())

        if path == "/api/crossmodal":
            return self._send_json(api_crossmodal())

        if path == "/api/conformal":
            return self._send_json(api_conformal())

        if path == "/api/reality-check":
            return self._send_json(api_reality_check())

        if path == "/api/metadata-overview":
            return self._send_json(api_metadata_overview())

        if path == "/api/literature":
            return self._send_json(api_literature())

        m = re.match(r"^/api/metadata/(.+)$", path)
        if m:
            sid = urllib.parse.unquote(m.group(1))
            out = api_metadata(sid)
            status = 404 if out.get("not_found") else 200
            return self._send_json(out, status)

        m = re.match(r"^/api/information/(.+)$", path)
        if m:
            sid = urllib.parse.unquote(m.group(1))
            out = api_information(sid)
            status = 404 if out.get("not_found") else 200
            return self._send_json(out, status)

        m = re.match(r"^/api/boed/(.+)$", path)
        if m:
            sid = urllib.parse.unquote(m.group(1))
            out = api_boed(sid)
            status = 404 if out.get("not_found") else 200
            return self._send_json(out, status)

        m = re.match(r"^/api/pooling/(.+)$", path)
        if m:
            pid = urllib.parse.unquote(m.group(1))
            out = api_pooling(pid)
            status = 404 if out.get("not_found") else 200
            return self._send_json(out, status)

        m = re.match(r"^/api/meta/(.+)$", path)
        if m:
            pid = urllib.parse.unquote(m.group(1))
            # M14 always answers 200 with an honest available/status block (a
            # single-study or absent protein is a valid, honest answer, not a 404).
            return self._send_json(api_meta(pid))

        m = re.match(r"^/api/structure/(.+)$", path)
        if m:
            pid = urllib.parse.unquote(m.group(1))
            # M16 always answers 200 with an honest available block (an absent uniprot
            # / missing features is a valid, honest {available: False} answer, never a
            # 404/500) — the corpus association table is still surfaced.
            return self._send_json(api_structure(pid))

        if path == "/api/proteins":
            q = (qs.get("q") or [None])[0]
            return self._send_json(api_proteins(q))

        m = re.match(r"^/api/protein/(.+)$", path)
        if m:
            pid = urllib.parse.unquote(m.group(1))
            detail = api_protein(pid)
            if detail is None:
                return self._send_json({"error": f"protein not found: {pid}"}, 404)
            return self._send_json(detail)

        m = re.match(r"^/api/curve/(.+)$", path)
        if m:
            sid = urllib.parse.unquote(m.group(1))
            cur = api_curve(sid)
            if cur is None:
                return self._send_json({"error": f"curve not found: {sid}"}, 404)
            return self._send_json(cur)

        m = re.match(r"^/api/models/(.+)$", path)
        if m:
            sid = urllib.parse.unquote(m.group(1))
            out = api_models(sid)
            # not-found -> 404, everything else (incl. {"error":...}) -> 200 JSON
            status = 404 if out.get("not_found") else 200
            return self._send_json(out, status)

        m = re.match(r"^/api/series-gamma/(.+)$", path)
        if m:
            pid = urllib.parse.unquote(m.group(1))
            out = api_series_gamma(pid)
            status = 404 if out.get("not_found") else 200
            return self._send_json(out, status)

        m = re.match(r"^/api/dose-response/(.+)$", path)
        if m:
            pid = urllib.parse.unquote(m.group(1))
            out = api_dose_response(pid)
            status = 404 if out.get("not_found") else 200
            return self._send_json(out, status)

        if path == "/api/series-list":
            # proteins that have a concentration series (dual-γ available)
            names = sorted(set(STORE.gamma_by_protein) | set(STORE.conc_series_by_protein))
            return self._send_json({"n": len(names), "proteins": names,
                                    "attribution": STORE._attribution()})

        if path == "/api/health":
            return self._send_json({
                "ok": True,
                "engine_ok": ENGINE_OK,
                "engine_error": ENGINE_ERR,
                "n_results": len(STORE.results_by_series),
                "n_proteins": len(STORE.series_by_protein),
                "n_curves": len(STORE.curve_by_series),
                "engine_build_id": STORE.engine_build_id,
            })

        return self._send_json({"error": f"not found: {path}"}, 404)

    # quieter logging
    def log_message(self, fmt, *args):
        sys.stderr.write("[web] %s - %s\n" % (self.address_string(), fmt % args))


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def make_server(port, host="127.0.0.1"):
    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    port = 8000
    if argv:
        try:
            port = int(argv[0])
        except ValueError:
            pass
    port = int(os.environ.get("PRISE_PORT", os.environ.get("PORT", port)))
    # Deployment binds 0.0.0.0 (PRISE_HOST); local default stays loopback.
    host = os.environ.get("PRISE_HOST", "127.0.0.1")

    sys.stderr.write("[web] loading PRISE artifacts from %s ...\n" % DATA)
    STORE.load()
    if STORE.errors:
        for e in STORE.errors:
            sys.stderr.write("[web] WARN artifact load: %s\n" % e)
    _n_assoc = len(((STORE.structure_assoc or {}).get("associations") or {})
                   .get("association_cards", []) or [])
    sys.stderr.write(
        "[web] loaded: %d results / %d proteins / %d curves / %d m9 / %d m12 / "
        "%d m13-boed / %d m14-meta (multi-study) / %d m15-lit-claims (%d conflicts) / "
        "%d m16-structure-proxy (%d assoc cards) (engine_ok=%s)\n"
        % (len(STORE.results_by_series), len(STORE.series_by_protein),
           len(STORE.curve_by_series), len(STORE.m9_by_series),
           len(STORE.information_by_series), len(STORE.boed_by_series),
           len(STORE.meta_by_protein), len(STORE.lit_claims),
           len((STORE.lit_conflicts or {}).get("conflicts", []) or []),
           len(STORE.structure_by_uniprot), _n_assoc, ENGINE_OK)
    )
    if not ENGINE_OK:
        sys.stderr.write("[web] NOTE engine import failed (%s) — /api/analyze and "
                         "fitted overlays disabled; browse/dashboard still work\n"
                         % ENGINE_ERR)

    httpd = make_server(port, host)
    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    url = "http://%s:%d" % (shown, port)
    print("PRISE web app -> %s" % url, flush=True)
    sys.stderr.write("[web] serving at %s (Ctrl-C to stop)\n" % url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\n[web] shutting down\n")
        httpd.shutdown()


if __name__ == "__main__":
    main()
