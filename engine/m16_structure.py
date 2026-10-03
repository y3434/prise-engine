"""
PRISE — Module M16: Sequence ↔ Structure ↔ Kinetics Bridge (§3-M16)
====================================================================

M16 EXTENDS M10's cross-modal analysis with a **STRUCTURE-FEATURE axis**. Where
M10 mined sequence PREDICTORS (PASTA/Waltz) → kinetics, M16 asks: do
STRUCTURE-DERIVED features associate with the measured kinetics? — and answers it
with the SAME protein-level rigor (effective n = PROTEINS, not curves; LOPO CV;
permutation nulls; variance ceiling; BH-FDR).

REAL 3D FEATURES FROM ACTUAL PDB COORDINATES (v2.0 — the upgrade)
----------------------------------------------------------------
This build FETCHES real PDB coordinate files (files.rcsb.org) into a local cache
and computes GENUINE 3D structural features from them in PURE numpy — no
BioPython / DSSP / FreeSASA / APBS. Every extractor stamps derivation="pdb_3d",
the per-structure native_or_fibril state, and an EXACT-vs-APPROXIMATION fidelity
label so nothing is overclaimed:

  * contact_map      — Cβ (Cα for Gly) distance matrix < cutoff -> contact pairs,
                       contact order, per-residue contact number. fidelity=EXACT.
                       (These are the REAL GNN contact edges.)
  * real_sasa        — Shrake–Rupley rolling-ball SASA (per-atom VdW radii, 1.4 Å
                       probe, 96 golden-spiral sphere points, neighbour-pruned) ->
                       per-residue SASA + relative SASA (Tien 2013 maxima) ->
                       buried/exposed. fidelity=EXACT ALGORITHM (Shrake–Rupley).
  * secondary_struct — backbone φ/ψ dihedral geometry -> H/E/C + beta_sheet_content
                       + helix%. fidelity=APPROXIMATION (φ/ψ geometry, NOT DSSP
                       H-bonding).
  * hydrophobic_patches — SASA-exposed hydrophobic (Kyte–Doolittle) residues
                       spatially clustered via the contact map. fidelity=REAL
                       (SASA-exposed + contact clustering).
  * electrostatic_surface — formal charges on exposed residues -> net surface
                       charge, +/- patch counts, Coulomb/Debye–Hückel potential
                       summary. fidelity=APPROXIMATION (Coulombic/Debye–Hückel,
                       NOT APBS Poisson–Boltzmann).
  * surface_curvature — Cα neighbour-density convexity for exposed residues.
                       fidelity=APPROXIMATION (Cα neighbour-density convexity).

A protein WITHOUT a fetched PDB (or a failed fetch/parse) keeps the existing
sequence/APR PROXY features, FLAGGED. `structural_features` reports BOTH where
available so the real-3D-vs-proxy contrast is visible. The whole real-3D path is
DETERMINISTIC given the cached files and NEVER raises: any fetch/parse failure
falls back to the proxy + a flag, it does not throw.

TWO HONESTY PILLARS (stamped on every output)
---------------------------------------------
  (1) The amyloid **NATIVE-vs-FIBRIL paradox**: a native-monomer structure is NOT
      the aggregation-competent / fibril state, so any structure→kinetics link is
      an ASSOCIATION, often confounded. Each structure carries a `native_or_fibril`
      stamp (from the corpus `type`: 'Fibril' => the aggregated state) + the
      "native ≠ aggregation-competent" note; MANY amyloid PDBs here are FIBRILS.
  (2) **Association-ONLY, NEVER causal**: `causal:false` is an INVARIANT on every
      association; `experimentally_validated` can only be set by an EXTERNAL human
      flag, never from data.

At n ≤ 24 proteins power is LOW: M16 EXPECTS mostly `insufficient_power`/`null`
verdicts and reports them honestly (like M10's clean nulls). The GNN graph is a
forward-compatible REPRESENTATION for a later model — NO GNN is built here, but
the REAL contact edges are now populated where a PDB was parsed.

Pure / deterministic (given cached files) / JSON-serialisable. NEVER raises on bad
input (returns a status block). Reuses engine/m10_crossmodal.py for all
protein-level statistics.

Version: m16-structure-2.0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# REUSE M10's protein-level rigor (do not reinvent): the cross-modal join, the
# variance ceiling, LOPO skill, permutation nulls, and the ordinal regime ladder.
from m10_crossmodal import (
    N_PERM,
    REGIME_LADDER,
    REGIME_ORDER,
    _spearman,
    _perm_pvalue_corr,
    build_protein_table,
    lopo_ordinal_accuracy,
    lopo_r2,
    variance_ceiling,
)

VERSION = "m16-structure-2.0"


def _safe_print(s):
    """Print that never dies on a non-UTF-8 console (Windows cp1252)."""
    try:
        print(s)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(str(s).encode(enc, "replace").decode(enc))


# ============================================================================ #
# PUBLISHED SCALES (hard-coded + cited — no external table needed)
# ============================================================================ #
# Kyte & Doolittle (1982) J Mol Biol 157:105 — hydropathy. Positive = hydrophobic.
KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Chou & Fasman (1978) Adv Enzymol 47:45 — conformational parameters.
# Pβ = β-sheet former propensity; Pα = α-helix former propensity. >1 = former.
CHOU_FASMAN_PB = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19, "Q": 1.10, "E": 0.37,
    "G": 0.75, "H": 0.87, "I": 1.60, "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38,
    "P": 0.55, "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}
CHOU_FASMAN_PA = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70, "Q": 1.11, "E": 1.51,
    "G": 0.57, "H": 1.00, "I": 1.08, "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13,
    "P": 0.57, "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}
# Crude electrostatics at pH 7 (a SEQUENCE proxy, NOT a 3D surface potential):
# D,E = -1 ; K,R = +1 ; H = +0.1 (partially protonated near pH 7). All else 0.
CHARGE_AT_PH7 = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}

STANDARD_AAS = set(KYTE_DOOLITTLE)

# 3-letter -> 1-letter for PDB resName parsing (the 20 standard residues).
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # common modified / alt residues mapped to their parent for feature purposes
    "MSE": "M", "HSD": "H", "HSE": "H", "HSP": "H", "SEC": "C", "PYL": "K",
    "CSO": "C", "PTR": "Y", "SEP": "S", "TPO": "T",
}

# Element -> van der Waals radius (Å), for the Shrake–Rupley rolling-ball SASA.
VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20, "P": 1.80}
VDW_DEFAULT = 1.70                     # any un-tabulated element -> carbon-like
PROBE_RADIUS = 1.4                     # water probe (Å)
N_SPHERE_POINTS = 96                   # golden-spiral surface points per atom

# Tien et al. 2013 (PLoS ONE 8:e80635) theoretical Gly-X-Gly max SASA (Å²) per
# residue type — the denominator for RELATIVE SASA (a residue with SASA==max is
# fully solvent-exposed). Used to call buried (<0.25) vs exposed.
TIEN_MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0, "E": 223.0,
    "Q": 225.0, "G": 104.0, "H": 224.0, "I": 197.0, "L": 201.0, "K": 236.0,
    "M": 224.0, "F": 240.0, "P": 159.0, "S": 155.0, "T": 172.0, "W": 285.0,
    "Y": 263.0, "V": 174.0,
}
REL_SASA_EXPOSED_CUTOFF = 0.25         # rel-SASA >= 0.25 -> solvent-exposed
KD_HYDROPHOBIC_CUTOFF = 1.5            # Kyte–Doolittle >= 1.5 -> hydrophobic residue

# Debye–Hückel / Coulomb surface-potential constants (approximation, NOT APBS PB).
IONIC_STRENGTH_M = 0.150               # ~150 mM physiological ionic strength
WATER_DIELECTRIC = 78.5               # relative permittivity of water at ~25 °C
DEBYE_LENGTH_A = 3.04 / math.sqrt(IONIC_STRENGTH_M)   # Debye length (Å) at 150 mM ~7.85 Å

# The kinetics-linked PDBs to fetch (§ inputs). Fetched into data/pdb_cache/.
KINETICS_PDB_IDS = [
    "1AG2", "1GXT", "1HEM", "1IYT", "1L3N", "1LDS", "1UOL", "1XQ8", "2C9V",
    "2IFB", "2L86", "2OCT", "2V0A", "2WAR", "2YXF", "3DVF", "3DVI", "3FFN",
    "3KQ0", "3U79", "4NP8",
]

# Map each kinetics-linked PDB -> the corpus UniProt whose kinetics it informs, and
# the native/fibril stamp of THAT structure (from the corpus `type`, verified by the
# PDB TITLE). 'fibril'=the aggregated/amyloid state; 'native'=native monomer. This
# is the honest per-structure native-vs-fibril stamp the task requires; a UniProt is
# assigned its best (native-monomer preferred) available real structure downstream.
KINETICS_PDB_TO_UNIPROT = {
    "1AG2": ("P04925", "native"),   # mouse prion protein PrP(121-231), native fold
    "1GXT": ("P30131", "native"),   # HypF N-terminal domain, native fold
    "1HEM": ("P61626", "native"),   # hen egg-white lysozyme, native fold
    "1IYT": ("P05067", "native"),   # Alzheimer Abeta(1-42) monomer (NMR), native
    "1L3N": ("P00441", "native"),   # reduced dimeric Cu,Zn-SOD1, native fold
    "1LDS": ("P61769", "native"),   # monomeric human beta-2-microglobulin, native
    "1UOL": ("Q761V2", "native"),   # p53 core domain, native fold
    "1XQ8": ("P37840", "native"),   # micelle-bound alpha-synuclein, native (helical)
    "2C9V": ("P00441", "native"),   # atomic-res Cu,Zn human SOD1, native fold
    "2IFB": ("P02693", "native"),   # rat intestinal fatty-acid-binding protein, native
    "2L86": ("P10997", "native"),   # human amylin/IAPP in SDS micelles (NMR), native
    "2OCT": ("P04080", "native"),   # Stefin B (cystatin B) tetramer, native fold
    "2V0A": ("P00441", "native"),   # atomic-res human SOD1, native fold
    "2WAR": ("P61626", "native"),   # hen egg-white lysozyme E35Q, native fold
    "2YXF": ("P61769", "native"),   # beta-2-microglobulin high-res, native fold
    "3DVF": ("P01594", "native"),   # amyloidogenic kappa-1 Bence-Jones light chain, native fold
    "3DVI": ("P01594", "native"),   # kappa-1 amyloidogenic light chain VL, native fold
    "3FFN": ("P06396", "native"),   # calcium-free human gelsolin, native fold
    "3KQ0": ("P02763", "native"),   # human alpha-1-acid glycoprotein, native fold
    "3U79": ("P01594", "native"),   # AL light chain variant, native fold
    "4NP8": ("P10636", "fibril"),   # tau VQIVYK steric-zipper amyloid — the FIBRIL state
}

# BH-FDR level for the association batch.
FDR_ALPHA = 0.05
# Minimum proteins before an 'association' verdict is even licensed. Below this a
# significant permutation hit is NOT trustworthy (a handful of proteins can look
# perfectly monotone by chance), so we refuse to call it — insufficient_power.
MIN_N_FOR_ASSOCIATION = 8

# The honesty block reused across every deferred 3D feature.
_NATIVE_FIBRIL_NOTE = (
    "A native-monomer structure is NOT the aggregation-competent/fibril state; any "
    "structure->kinetics link is ASSOCIATIVE and often confounded (native != fibril)."
)


def _stub_error(feature, dep):
    """Build the honest NotImplementedError message for a deferred 3D feature.
    WHY: offline PRISE has no structural-biology stack, no coordinates, no network,
    so a genuinely-3D feature CANNOT be computed — we refuse rather than fake it."""
    return NotImplementedError(
        f"{feature} requires {dep} + 3D coordinates; deferred pluggable stage — "
        f"see honesty block. Not computable offline (no structural-biology tools, "
        f"no .pdb/.cif coordinates, no network bundled). {_NATIVE_FIBRIL_NOTE}"
    )


# ============================================================================ #
# 1. STRUCTURE-SOURCE ABSTRACTION (+ pluggable loader stub)
# ============================================================================ #
def make_structure(source, id, uniprot=None, is_fibril=None, reliability=None,
                   n_models=None, has_coordinates=False):
    """Normalise a structure descriptor into the M16 `Structure` schema (§3-M16).

    A `Structure` records its provenance so EVERY downstream feature can be stamped
    with source + native_or_fibril + the native!=aggregation-competent note. The
    reliability metric is source-specific: AlphaFold->pLDDT, CryoEM->resolution_A,
    NMR->ensemble_models, PDB->experimental. `has_coordinates` is False offline
    (no .pdb/.cif bundled), which is WHY every real 3D feature is a stub."""
    src = (source or "PDB").upper()
    src = {"CRYO-EM": "CryoEM", "CRYOEM": "CryoEM", "PDB": "PDB",
           "ALPHAFOLD": "AlphaFold", "NMR": "NMR"}.get(src, source)
    if is_fibril is True:
        nof = "fibril"
    elif is_fibril is False:
        nof = "native"
    else:
        nof = "unknown"
    # default reliability metric by source (value stays None offline — no coords)
    if reliability is None:
        metric = {"AlphaFold": "pLDDT", "CryoEM": "resolution_A",
                  "NMR": "ensemble_models"}.get(src, "experimental")
        reliability = {"metric": metric, "value": None}
    return {
        "source": src,
        "id": id,
        "uniprot": uniprot,
        "is_fibril": is_fibril,
        "native_or_fibril": nof,
        "reliability": reliability,
        "n_models": n_models,
        "has_coordinates": bool(has_coordinates),
        "native_vs_fibril_note": _NATIVE_FIBRIL_NOTE,
    }


def structure_from_pdb_record(rec, uniprot=None):
    """Adapt a `sequence_structure.json` structures[] entry to a `Structure`.
    Fibril-vs-native is inferred from the entry `type` ('Fibril' => fibril);
    reliability from the deposition method (resolution for EM/X-ray, ensemble for
    NMR). Still has_coordinates=False — we hold only the PDB ID, not the coords."""
    typ = (rec.get("type") or "").lower()
    method = (rec.get("method") or "").upper()
    is_fibril = True if "fibril" in typ else (False if typ else None)
    if "NMR" in method:
        source, reliability, n_models = "NMR", {"metric": "ensemble_models",
                                                "value": rec.get("n_models")}, rec.get("n_models")
    elif "MICROSCOPY" in method or "CRYO" in method or "EM" in method:
        source = "CryoEM"
        reliability = {"metric": "resolution_A", "value": rec.get("resolution")}
        n_models = None
    else:
        source = "PDB"
        reliability = {"metric": "resolution_A" if rec.get("resolution") else "experimental",
                       "value": rec.get("resolution")}
        n_models = None
    return make_structure(source, rec.get("pdb_id"), uniprot=uniprot,
                          is_fibril=is_fibril, reliability=reliability,
                          n_models=n_models, has_coordinates=False)


def _pdb_cache_dir():
    """The on-disk PDB coordinate cache (data/ is gitignored — safe to populate)."""
    return Path(__file__).resolve().parent.parent / "data" / "pdb_cache"


def fetch_pdb(pdb_id, cache_dir=None, timeout=30, force=False):
    """Download {pdb_id}.pdb from files.rcsb.org into the local cache; return its
    path (or None on failure — NEVER raises).

    Robust: skips if already cached, sets a User-Agent, retries once on error,
    30 s timeout. On any failure it returns None so the caller falls back to the
    sequence/APR proxy + a flag. Deterministic given the cache (a cached file is
    reused verbatim)."""
    if not pdb_id:
        return None
    pid = str(pdb_id).strip().upper()
    cache = Path(cache_dir) if cache_dir else _pdb_cache_dir()
    dst = cache / f"{pid}.pdb"
    try:
        if dst.exists() and dst.stat().st_size > 0 and not force:
            return dst
    except Exception:
        pass
    url = f"https://files.rcsb.org/download/{pid}.pdb"
    for attempt in range(2):                     # one retry
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PRISE-M16/2.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                raise ValueError("empty response")
            cache.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            return dst
        except Exception:                        # network/HTTP/parse — never raise
            if attempt == 0:
                try:
                    time.sleep(1.0)
                except Exception:
                    pass
                continue
            return None
    return None


def _atom_element(name_field, element_field):
    """Best-effort element symbol from a PDB ATOM record (element cols 77-78 first,
    else inferred from the atom name)."""
    el = (element_field or "").strip()
    if el:
        return el[0].upper() + el[1:].lower() if len(el) > 1 else el.upper()
    nm = (name_field or "").strip()
    # atom names like 'CA','CB','N','O','SG' — first alpha char is usually the element
    for ch in nm:
        if ch.isalpha():
            return ch.upper()
    return "C"


def parse_pdb(path, pdb_id=None):
    """Pure-stdlib PDB parser -> a Structure dict (NEVER raises; returns None on
    unreadable/empty input).

    Reads ATOM/HETATM records by FIXED COLUMNS (name 13-16, resName 18-20,
    chain 22, resSeq 23-26 + iCode 27, x/y/z 31-54, element 77-78). Multi-model NMR
    ensembles keep MODEL 1 only for geometry (the ensemble size is noted). HETATM
    waters/ions are dropped; modified residues in THREE_TO_ONE are kept.

    Structure = {pdb_id, models:[{chains:{chain:[residue]}}], n_models, n_chains,
    is_fibril, native_or_fibril, ...}, where residue =
    {resnum, resname(1-letter), atoms:[{name, element, xyz(np.ndarray)}]}."""
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return None
        pid = (pdb_id or p.stem).upper()
        # collect residues for model 1 only, preserving chain + residue order
        model_no = 0                              # 0 = pre-MODEL (single-model files)
        in_model_one = True                      # true until we pass ENDMDL of model 1
        chains = {}                              # chain -> list[residue]
        cur_key = {}                             # (chain,resnum,icode) -> residue dict
        n_models = 0
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                rec = line[:6].strip()
                if rec == "MODEL":
                    n_models += 1
                    model_no += 1
                    in_model_one = (n_models == 1)
                    continue
                if rec == "ENDMDL":
                    if n_models >= 1:
                        in_model_one = False     # done with model 1's atoms
                    continue
                if rec not in ("ATOM", "HETATM"):
                    continue
                if not in_model_one and n_models >= 1:
                    continue                     # geometry from model 1 only
                # altLoc: keep only blank or 'A' (avoid double-counting split atoms)
                altloc = line[16:17]
                if altloc not in (" ", "", "A"):
                    continue
                resname3 = line[17:20].strip().upper()
                aa = THREE_TO_ONE.get(resname3)
                if aa is None:
                    continue                     # skip waters/ions/ligands/unknowns
                chain = line[21:22] or " "
                try:
                    resnum = int(line[22:26])
                except Exception:
                    continue
                icode = line[26:27]
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except Exception:
                    continue
                name = line[12:16].strip()
                element = _atom_element(name, line[76:78])
                key = (chain, resnum, icode)
                res = cur_key.get(key)
                if res is None:
                    res = {"resnum": resnum, "resname": aa, "atoms": []}
                    cur_key[key] = res
                    chains.setdefault(chain, []).append(res)
                res["atoms"].append({"name": name, "element": element,
                                     "xyz": np.array([x, y, z], dtype=float)})
        chains = {c: rs for c, rs in chains.items() if rs}
        if not chains:
            return None
        n_chains = len(chains)
        n_residues = sum(len(rs) for rs in chains.values())
        # >=2 distinct protein chains each with a real fold (>=20 residues) reads as a
        # multi-chain assembly; the authoritative native/fibril stamp comes from the
        # corpus `type` (attached downstream), this is only a coordinate-level hint.
        big_chains = sum(1 for rs in chains.values() if len(rs) >= 20)
        assembly_hint = big_chains >= 2
        model1 = {"chains": chains}
        return {
            "pdb_id": pid,
            "models": [model1],
            "n_models": n_models if n_models else 1,
            "is_nmr_ensemble": n_models > 1,
            "n_chains": n_chains,
            "n_residues": n_residues,
            "chain_ids": sorted(chains.keys()),
            "assembly_multichain_hint": assembly_hint,
            "is_fibril": None,                   # set from corpus `type` downstream
            "native_or_fibril": "unknown",       # set from corpus `type` downstream
            "has_coordinates": True,
        }
    except Exception:
        return None                              # NEVER raise on a bad file


def load_structure(source, path_or_id, cache_dir=None):
    """Load 3D coordinates for a structure. For source=='PDB' this is now REAL:
    fetch_pdb() + parse_pdb(). Returns the parsed Structure dict, or None on any
    failure (caller falls back to the sequence/APR proxy + a flag — NEVER raises).

    Non-PDB sources (AlphaFold / CryoEM / NMR ensembles from other providers) remain
    deferred loaders: PRISE ships only the PDB path as REAL here."""
    src = (source or "PDB").upper()
    if src in ("PDB", "CRYOEM", "CRYO-EM", "NMR", "EM"):
        # all of these are downloadable as .pdb from RCSB by ID here
        path = fetch_pdb(path_or_id, cache_dir=cache_dir)
        if path is None:
            return None
        return parse_pdb(path, pdb_id=path_or_id)
    # deferred non-PDB loaders (e.g. AlphaFold model repos): not wired here
    return None


# ============================================================================ #
# 1b. REAL 3D FEATURE EXTRACTORS (pure numpy, from actual coordinates)
# ============================================================================ #
def _iter_residues(structure):
    """Yield (chain, residue) for model-1 residues in a parsed Structure."""
    if not structure or not structure.get("models"):
        return
    for chain, residues in structure["models"][0]["chains"].items():
        for res in residues:
            yield chain, res


def _residue_repr_atom(res):
    """The contact-map representative atom of a residue: Cβ, or Cα for Gly / when Cβ
    is missing. Returns an np.ndarray xyz or None."""
    ca = cb = None
    for a in res["atoms"]:
        if a["name"] == "CB":
            cb = a["xyz"]
        elif a["name"] == "CA":
            ca = a["xyz"]
    if res["resname"] == "G":
        return ca
    return cb if cb is not None else ca


def _residue_list(structure):
    """Flat ordered list of residues with a usable Cα/Cβ, plus parallel metadata."""
    residues, meta = [], []
    for chain, res in _iter_residues(structure):
        rep = _residue_repr_atom(res)
        if rep is None:
            continue
        residues.append(res)
        meta.append({"chain": chain, "resnum": res["resnum"], "aa": res["resname"],
                     "rep_xyz": rep})
    return residues, meta


def compute_contact_map(structure, cutoff_angstrom=8.0):
    """REAL residue–residue contact map from coordinates. Cβ (Cα for Gly) pairwise
    Euclidean distances; a contact = distance < cutoff for |i-j| >= 2 (skip trivial
    sequence neighbours). Returns contact pairs, per-residue contact number, and the
    relative contact order. These ARE the GNN contact edges.

    fidelity: EXACT (a deterministic distance threshold on the real coordinates)."""
    if not structure:
        return {"computable": False, "reason": "no structure"}
    _, meta = _residue_list(structure)
    n = len(meta)
    if n < 2:
        return {"computable": False, "reason": "need >=2 residues", "n_residues": n}
    coords = np.array([m["rep_xyz"] for m in meta], dtype=float)
    # full pairwise distance matrix (n small — PDB monomers/oligomers)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    contact = (dist < float(cutoff_angstrom))
    np.fill_diagonal(contact, False)
    seq_sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    contact_nontrivial = contact & (seq_sep >= 2)
    iu = np.triu_indices(n, k=2)
    pair_mask = contact_nontrivial[iu]
    pairs_i = iu[0][pair_mask]
    pairs_j = iu[1][pair_mask]
    n_contacts = int(pairs_i.size)
    contact_number = contact_nontrivial.sum(axis=1).astype(int)
    # relative contact order = mean sequence separation of contacts / (n * n_contacts)
    if n_contacts > 0:
        seps = np.abs(pairs_i - pairs_j)
        rel_co = float(np.sum(seps) / (n * n_contacts))
        abs_co = float(np.mean(seps))
    else:
        rel_co = 0.0
        abs_co = 0.0
    contact_edges = [{"i": int(i), "j": int(j),
                      "resnum_i": int(meta[i]["resnum"]), "resnum_j": int(meta[j]["resnum"]),
                      "chain_i": meta[i]["chain"], "chain_j": meta[j]["chain"]}
                     for i, j in zip(pairs_i.tolist(), pairs_j.tolist())]
    return {
        "computable": True,
        "derivation": "pdb_3d",
        "fidelity": "exact",
        "cutoff_angstrom": float(cutoff_angstrom),
        "n_residues": n,
        "n_contacts": n_contacts,
        "relative_contact_order": round(rel_co, 6),
        "absolute_contact_order": round(abs_co, 4),
        "mean_contact_number": round(float(np.mean(contact_number)), 4),
        "max_contact_number": int(np.max(contact_number)) if n else 0,
        "per_residue_contact_number": contact_number.tolist(),
        "contact_edges": contact_edges,
        "note": ("REAL contact map: Cbeta (Calpha for Gly) distance < cutoff, |i-j|>=2. "
                 "These are the GNN contact edges. fidelity=EXACT."),
    }


def _golden_spiral_points(n):
    """`n` roughly-equidistant unit-sphere points (Fibonacci/golden-spiral). Used as
    the Shrake–Rupley test directions — deterministic, no RNG."""
    idx = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n)             # polar angle
    theta = math.pi * (1.0 + 5.0 ** 0.5) * idx       # golden-angle azimuth
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def compute_real_sasa(structure, probe_radius=PROBE_RADIUS,
                      n_points=N_SPHERE_POINTS):
    """REAL solvent-accessible surface area via the SHRAKE–RUPLEY rolling-ball
    algorithm, in pure numpy. Each atom gets a VdW radius (C 1.70, N 1.55, O 1.52,
    S 1.80, H 1.20 Å) inflated by the 1.4 Å probe; ~96 golden-spiral test points on
    that inflated sphere are marked occluded if they fall inside any NEIGHBOUR atom's
    inflated sphere (neighbour-pruned by a distance cutoff). Per-atom SASA is summed
    to per-residue SASA and divided by the Tien 2013 Gly-X-Gly maximum for relative
    SASA -> buried (<0.25) vs exposed.

    fidelity: EXACT ALGORITHM (Shrake–Rupley) — an exact implementation of an
    established numerical method (its accuracy is bounded by n_points, reported)."""
    if not structure:
        return {"computable": False, "reason": "no structure"}
    # flatten all atoms of model 1 (with a residue index back-pointer)
    atom_xyz, atom_r, atom_res = [], [], []
    residues_meta = []                            # per-residue: chain,resnum,aa
    ridx = 0
    for chain, res in _iter_residues(structure):
        residues_meta.append({"chain": chain, "resnum": res["resnum"], "aa": res["resname"]})
        for a in res["atoms"]:
            atom_xyz.append(a["xyz"])
            atom_r.append(VDW_RADII.get(a["element"], VDW_DEFAULT))
            atom_res.append(ridx)
        ridx += 1
    n_atoms = len(atom_xyz)
    if n_atoms == 0:
        return {"computable": False, "reason": "no atoms"}
    xyz = np.array(atom_xyz, dtype=float)
    radii = np.array(atom_r, dtype=float) + probe_radius     # inflated radii
    atom_res = np.array(atom_res, dtype=int)
    sphere = _golden_spiral_points(n_points)                 # (P,3) unit directions
    per_atom_area = 4.0 * math.pi * radii * radii            # full-sphere area/atom
    max_reach = float(np.max(radii)) * 2.0
    per_atom_sasa = np.zeros(n_atoms, dtype=float)
    # neighbour search per atom (cutoff = ri + rj_max), then occlusion test on points
    for i in range(n_atoms):
        ri = radii[i]
        d = xyz - xyz[i]
        dist2 = np.sum(d * d, axis=1)
        cut = (ri + radii) ** 2
        neigh = np.where((dist2 < cut) & (np.arange(n_atoms) != i))[0]
        pts = xyz[i] + ri * sphere                           # (P,3) surface points
        if neigh.size == 0:
            per_atom_sasa[i] = per_atom_area[i]
            continue
        nxyz = xyz[neigh]                                    # (M,3)
        nr2 = radii[neigh] ** 2                              # (M,)
        # a point is buried if inside ANY neighbour's inflated sphere
        pd = pts[:, None, :] - nxyz[None, :, :]              # (P,M,3)
        pd2 = np.sum(pd * pd, axis=2)                        # (P,M)
        buried = np.any(pd2 < nr2[None, :], axis=1)          # (P,)
        exposed_frac = 1.0 - float(np.mean(buried))
        per_atom_sasa[i] = per_atom_area[i] * exposed_frac
    _ = max_reach
    # sum atom SASA to residues, compute relative SASA + buried/exposed
    n_res = len(residues_meta)
    res_sasa = np.zeros(n_res, dtype=float)
    for i in range(n_atoms):
        res_sasa[atom_res[i]] += per_atom_sasa[i]
    rel = []
    exposed_flags = []
    per_residue = []
    for k, m in enumerate(residues_meta):
        maxasa = TIEN_MAX_ASA.get(m["aa"])
        rsa = (res_sasa[k] / maxasa) if maxasa else None
        if rsa is not None:
            rsa = min(1.5, rsa)                               # cap tiny-frag artifacts
            rel.append(rsa)
            exposed_flags.append(rsa >= REL_SASA_EXPOSED_CUTOFF)
        per_residue.append({"chain": m["chain"], "resnum": m["resnum"], "aa": m["aa"],
                            "sasa_A2": round(float(res_sasa[k]), 3),
                            "rel_sasa": (round(rsa, 4) if rsa is not None else None),
                            "exposed": (bool(rsa >= REL_SASA_EXPOSED_CUTOFF)
                                        if rsa is not None else None)})
    total_sasa = float(np.sum(res_sasa))
    n_exposed = int(sum(1 for e in exposed_flags if e))
    n_scored = len(rel)
    return {
        "computable": True,
        "derivation": "pdb_3d",
        "fidelity": "exact_algorithm (Shrake-Rupley)",
        "probe_radius_A": probe_radius,
        "n_sphere_points": n_points,
        "n_residues": n_res,
        "total_sasa_A2": round(total_sasa, 2),
        "total_sasa_nm2": round(total_sasa / 100.0, 3),
        "mean_rel_sasa": (round(float(np.mean(rel)), 4) if rel else None),
        "median_rel_sasa": (round(float(np.median(rel)), 4) if rel else None),
        "frac_exposed": (round(n_exposed / n_scored, 4) if n_scored else None),
        "n_exposed": n_exposed,
        "n_buried": (n_scored - n_exposed),
        "per_residue": per_residue,
        "note": ("Shrake-Rupley rolling-ball SASA (VdW + 1.4A probe, %d golden-spiral "
                 "points, neighbour-pruned). rel-SASA vs Tien 2013 max; exposed if "
                 ">=%.2f. fidelity=EXACT ALGORITHM." % (n_points, REL_SASA_EXPOSED_CUTOFF)),
    }


def _backbone_atoms(structure):
    """Per-chain ordered list of (resnum, aa, N, CA, C) backbone coordinates for φ/ψ.
    Missing-backbone residues break the chain (dihedrals need i-1 .. i+1)."""
    chains = {}
    if not structure or not structure.get("models"):
        return chains
    for chain, residues in structure["models"][0]["chains"].items():
        seq = []
        for res in residues:
            N = CA = C = None
            for a in res["atoms"]:
                if a["name"] == "N":
                    N = a["xyz"]
                elif a["name"] == "CA":
                    CA = a["xyz"]
                elif a["name"] == "C":
                    C = a["xyz"]
            seq.append((res["resnum"], res["resname"], N, CA, C))
        chains[chain] = seq
    return chains


def _dihedral(p0, p1, p2, p3):
    """Signed dihedral angle (degrees) about the p1-p2 bond (standard formula)."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return math.degrees(math.atan2(y, x))


def _classify_ss(phi, psi):
    """φ/ψ -> H (helix) / E (sheet) / C (coil) by Ramachandran basin proximity.
    Helix basin ~(-60,-45); sheet basin ~(-120,+130). APPROXIMATION, NOT DSSP."""
    if phi is None or psi is None:
        return "C"
    # right-handed alpha-helix basin
    if -160.0 <= phi <= -20.0 and -90.0 <= psi <= 5.0:
        return "H"
    # extended / beta-sheet basin (psi positive, phi strongly negative)
    if -180.0 <= phi <= -40.0 and (90.0 <= psi <= 180.0 or -180.0 <= psi <= -150.0):
        return "E"
    return "C"


def compute_secondary_structure(structure):
    """Geometry-based secondary structure from backbone φ/ψ dihedrals (pure numpy).
    For each residue with a complete i-1,i,i+1 backbone we compute φ (C_{i-1}-N-CA-C)
    and ψ (N-CA-C-N_{i+1}), then assign H / E / C by Ramachandran basin. Reports the
    per-residue SS string, beta_sheet_content (%E) and helix_content (%H).

    fidelity: APPROXIMATION (backbone φ/ψ geometry, NOT DSSP H-bonding) — it does not
    detect the hydrogen-bond ladders DSSP uses, so it labels CONFORMATION, not true
    DSSP secondary structure."""
    if not structure:
        return {"computable": False, "reason": "no structure"}
    chains = _backbone_atoms(structure)
    ss_all = []
    per_residue = []
    for chain, seq in chains.items():
        for i in range(len(seq)):
            resnum, aa, N, CA, C = seq[i]
            phi = psi = None
            if i > 0 and seq[i - 1][4] is not None and N is not None and CA is not None and C is not None:
                Cprev = seq[i - 1][4]
                phi = _dihedral(Cprev, N, CA, C)
            if i + 1 < len(seq) and seq[i + 1][2] is not None and N is not None and CA is not None and C is not None:
                Nnext = seq[i + 1][2]
                psi = _dihedral(N, CA, C, Nnext)
            ss = _classify_ss(phi, psi)
            ss_all.append(ss)
            per_residue.append({"chain": chain, "resnum": resnum, "aa": aa,
                                "phi": (round(phi, 1) if phi is not None else None),
                                "psi": (round(psi, 1) if psi is not None else None),
                                "ss": ss})
    n = len(ss_all)
    if n == 0:
        return {"computable": False, "reason": "no scorable backbone"}
    c = Counter(ss_all)
    return {
        "computable": True,
        "derivation": "pdb_3d",
        "fidelity": "approximation (backbone phi/psi geometry, NOT DSSP H-bonding)",
        "n_residues": n,
        "beta_sheet_content": round(100.0 * c.get("E", 0) / n, 2),
        "helix_content": round(100.0 * c.get("H", 0) / n, 2),
        "coil_content": round(100.0 * c.get("C", 0) / n, 2),
        "ss_string": "".join(ss_all),
        "ss_counts": {"H": c.get("H", 0), "E": c.get("E", 0), "C": c.get("C", 0)},
        "per_residue": per_residue,
        "note": ("H/E/C from backbone phi/psi Ramachandran basins. beta_sheet_content "
                 "= %%E. fidelity=APPROXIMATION (geometry, not DSSP H-bond ladders)."),
    }


def _exposed_residue_index(sasa):
    """Map (chain,resnum) -> rel_sasa for residues the SASA pass marked exposed."""
    out = {}
    for r in (sasa or {}).get("per_residue", []):
        if r.get("exposed"):
            out[(r["chain"], r["resnum"])] = r.get("rel_sasa")
    return out


def compute_hydrophobic_surface_patches(structure, sasa, contact=None,
                                        cutoff_angstrom=8.0):
    """REAL 3D hydrophobic surface patches: take the SASA-EXPOSED residues whose
    Kyte–Doolittle hydrophobicity is high (>= 1.5), then spatially CLUSTER them via
    the contact map (two exposed-hydrophobic residues are in the same patch if they
    contact within the cutoff). Reports patch count + max patch size.

    fidelity: REAL (SASA-exposed residues + contact-map spatial clustering) — every
    ingredient (exposure, spatial adjacency) is computed from the real coordinates."""
    if not structure or not sasa or not sasa.get("computable"):
        return {"computable": False, "reason": "need structure + SASA"}
    _, meta = _residue_list(structure)
    n = len(meta)
    if n == 0:
        return {"computable": False, "reason": "no residues"}
    exposed = _exposed_residue_index(sasa)
    # exposed AND hydrophobic residues -> node set for clustering
    nodes = [k for k, m in enumerate(meta)
             if (m["chain"], m["resnum"]) in exposed
             and KYTE_DOOLITTLE.get(m["aa"], -9) >= KD_HYDROPHOBIC_CUTOFF]
    if not nodes:
        return {"computable": True, "derivation": "pdb_3d",
                "fidelity": "real (SASA-exposed + contact clustering)",
                "n_exposed_hydrophobic": 0, "n_patches": 0, "max_patch_size": 0,
                "patch_sizes": [], "note": "no exposed hydrophobic residues"}
    coords = np.array([meta[k]["rep_xyz"] for k in nodes], dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    adj = (dist < float(cutoff_angstrom))
    m = len(nodes)
    # union-find connected components over the exposed-hydrophobic adjacency graph
    parent = list(range(m))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(m):
        for b in range(a + 1, m):
            if adj[a, b]:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    comp = Counter(find(a) for a in range(m))
    sizes = sorted(comp.values(), reverse=True)
    return {
        "computable": True,
        "derivation": "pdb_3d",
        "fidelity": "real (SASA-exposed + contact clustering)",
        "kd_cutoff": KD_HYDROPHOBIC_CUTOFF,
        "n_exposed_hydrophobic": m,
        "n_patches": len(sizes),
        "max_patch_size": (sizes[0] if sizes else 0),
        "mean_patch_size": (round(sum(sizes) / len(sizes), 3) if sizes else 0),
        "patch_sizes": sizes,
        "note": ("exposed (rel-SASA>=%.2f) + hydrophobic (KD>=%.1f) residues clustered "
                 "by contact map. fidelity=REAL." % (REL_SASA_EXPOSED_CUTOFF,
                                                     KD_HYDROPHOBIC_CUTOFF)),
    }


def compute_electrostatic_surface(structure, sasa):
    """Surface electrostatics on the SASA-EXPOSED residues (approximation, NOT APBS).
    Formal charges (Asp/Glu −1, Lys/Arg +1, His +0.1) on exposed charged residues ->
    net surface charge and +/- surface-patch counts; a Coulomb / Debye–Hückel
    screened potential summary (κ from ~150 mM ionic strength) gives a
    surface-potential magnitude in kT/e units.

    fidelity: APPROXIMATION (Coulombic / Debye–Hückel screened point charges, NOT an
    APBS Poisson–Boltzmann solution of the full dielectric geometry)."""
    if not structure or not sasa or not sasa.get("computable"):
        return {"computable": False, "reason": "need structure + SASA"}
    _, meta = _residue_list(structure)
    exposed = _exposed_residue_index(sasa)
    charged = []                                  # (xyz, q, aa) for exposed charged res
    n_pos = n_neg = 0
    net = 0.0
    for m in meta:
        key = (m["chain"], m["resnum"])
        if key not in exposed:
            continue
        q = CHARGE_AT_PH7.get(m["aa"], 0.0)
        if q == 0.0:
            continue
        charged.append((m["rep_xyz"], q, m["aa"]))
        net += q
        if q > 0:
            n_pos += 1
        elif q < 0:
            n_neg += 1
    n_charged = len(charged)
    if n_charged == 0:
        return {"computable": True, "derivation": "pdb_3d",
                "fidelity": "approximation (Coulombic/Debye-Huckel, NOT APBS Poisson-Boltzmann)",
                "net_surface_charge": 0.0, "n_exposed_positive": 0, "n_exposed_negative": 0,
                "n_positive_patches": 0, "n_negative_patches": 0,
                "mean_abs_surface_potential_kT_e": 0.0,
                "note": "no exposed charged residues"}
    # screened (Debye–Hückel) pairwise potential energy per charge, in kT/e-like units:
    # phi_i = sum_{j!=i} q_j * exp(-r_ij / lambda_D) / (eps_r * r_ij).  Bjerrum-scaled
    # so the magnitude is a dimensionless surface-potential proxy (approximation).
    xyz = np.array([c[0] for c in charged], dtype=float)
    qs = np.array([c[1] for c in charged], dtype=float)
    diff = xyz[:, None, :] - xyz[None, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(r, np.inf)
    BJERRUM_A = 7.0                               # Bjerrum length in water ~7 Å (25 C)
    screen = np.exp(-r / DEBYE_LENGTH_A)
    phi = np.sum(qs[None, :] * screen * BJERRUM_A / r, axis=1)   # per-charge potential
    mean_abs_phi = float(np.mean(np.abs(phi)))
    # +/- surface patches: cluster same-sign exposed charged residues by contact (8 Å)
    def _cluster(sel):
        if sel.size < 1:
            return 0
        c = xyz[sel]
        d = np.sqrt(np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=-1))
        adj = d < 8.0
        mm = len(sel)
        parent = list(range(mm))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for a in range(mm):
            for b in range(a + 1, mm):
                if adj[a, b]:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
        return len({find(a) for a in range(mm)})
    n_pos_patch = _cluster(np.where(qs > 0)[0])
    n_neg_patch = _cluster(np.where(qs < 0)[0])
    return {
        "computable": True,
        "derivation": "pdb_3d",
        "fidelity": "approximation (Coulombic/Debye-Huckel, NOT APBS Poisson-Boltzmann)",
        "ionic_strength_M": IONIC_STRENGTH_M,
        "debye_length_A": round(DEBYE_LENGTH_A, 3),
        "net_surface_charge": round(net, 3),
        "n_exposed_charged": n_charged,
        "n_exposed_positive": n_pos,
        "n_exposed_negative": n_neg,
        "n_positive_patches": n_pos_patch,
        "n_negative_patches": n_neg_patch,
        "mean_abs_surface_potential_kT_e": round(mean_abs_phi, 4),
        "note": ("formal charges (D/E=-1,K/R=+1,H=+0.1) on exposed residues; screened "
                 "Debye-Huckel potential at %d mM. fidelity=APPROXIMATION (NOT APBS "
                 "Poisson-Boltzmann)." % int(1000 * IONIC_STRENGTH_M)),
    }


def compute_surface_curvature(structure, sasa, radius_A=10.0):
    """Local surface curvature proxy for the SASA-EXPOSED residues, from Cα
    neighbour density: for each exposed residue we count Cα atoms within a sphere of
    `radius_A`; FEWER neighbours => more convex / protruding (higher curvature),
    MORE neighbours => flatter / concave. The convexity score is normalised so higher
    = more convex.

    fidelity: APPROXIMATION (Cα neighbour-density convexity) — a coordinate-derived
    proxy for local convexity, NOT a true differential-geometry surface curvature."""
    if not structure or not sasa or not sasa.get("computable"):
        return {"computable": False, "reason": "need structure + SASA"}
    # all Cα coordinates (the density field)
    ca_all = []
    for _, res in _iter_residues(structure):
        for a in res["atoms"]:
            if a["name"] == "CA":
                ca_all.append(a["xyz"])
                break
    if len(ca_all) < 3:
        return {"computable": False, "reason": "need >=3 CA atoms"}
    ca_all = np.array(ca_all, dtype=float)
    _, meta = _residue_list(structure)
    exposed = _exposed_residue_index(sasa)
    scores = []
    per_residue = []
    for m in meta:
        key = (m["chain"], m["resnum"])
        if key not in exposed:
            continue
        d = ca_all - m["rep_xyz"]
        dist = np.sqrt(np.sum(d * d, axis=1))
        n_neigh = int(np.sum(dist < radius_A)) - 1     # exclude self
        # convexity = 1 - neighbours/local-max (fewer neighbours => more convex/protruding)
        scores.append(n_neigh)
        per_residue.append({"chain": m["chain"], "resnum": m["resnum"], "aa": m["aa"],
                            "n_ca_neighbours": n_neigh})
    if not scores:
        return {"computable": True, "derivation": "pdb_3d",
                "fidelity": "approximation (CA neighbour-density convexity)",
                "n_exposed": 0, "mean_convexity": None, "note": "no exposed residues"}
    arr = np.array(scores, dtype=float)
    mx = float(np.max(arr)) if arr.size else 1.0
    convex = 1.0 - arr / mx if mx > 0 else np.zeros_like(arr)   # 1=most convex
    return {
        "computable": True,
        "derivation": "pdb_3d",
        "fidelity": "approximation (CA neighbour-density convexity)",
        "radius_A": radius_A,
        "n_exposed": len(scores),
        "mean_ca_neighbours": round(float(np.mean(arr)), 3),
        "mean_convexity": round(float(np.mean(convex)), 4),
        "max_convexity": round(float(np.max(convex)), 4),
        "per_residue": per_residue,
        "note": ("exposed-residue local convexity from CA neighbour density within "
                 "%.0f A (fewer neighbours=more convex). fidelity=APPROXIMATION." % radius_A),
    }


def compute_real_structural_features(structure, native_or_fibril="unknown",
                                     cutoff_angstrom=8.0):
    """Run ALL real 3D extractors on a parsed Structure and bundle them with their
    per-feature fidelity + the native/fibril stamp. NEVER raises: any extractor that
    cannot run returns a computable:false block. Returns None if `structure` is None
    (caller then falls back to the proxy)."""
    if not structure:
        return None
    try:
        contact = compute_contact_map(structure, cutoff_angstrom=cutoff_angstrom)
        sasa = compute_real_sasa(structure)
        ss = compute_secondary_structure(structure)
        hpatch = compute_hydrophobic_surface_patches(structure, sasa, contact=contact,
                                                     cutoff_angstrom=cutoff_angstrom)
        elec = compute_electrostatic_surface(structure, sasa)
        curv = compute_surface_curvature(structure, sasa)
    except Exception as exc:                       # defensive — never raise
        return {"computable": False, "derivation": "pdb_3d",
                "error": f"{type(exc).__name__}: {exc}"}
    # protein-level scalar summary (the association predictors)
    summary = {
        "mean_rel_sasa": sasa.get("mean_rel_sasa") if sasa.get("computable") else None,
        "frac_exposed": sasa.get("frac_exposed") if sasa.get("computable") else None,
        "beta_sheet_content": ss.get("beta_sheet_content") if ss.get("computable") else None,
        "helix_content": ss.get("helix_content") if ss.get("computable") else None,
        "relative_contact_order": contact.get("relative_contact_order") if contact.get("computable") else None,
        "n_contacts": contact.get("n_contacts") if contact.get("computable") else None,
        "mean_contact_number": contact.get("mean_contact_number") if contact.get("computable") else None,
        "n_hydrophobic_patches": hpatch.get("n_patches") if hpatch.get("computable") else None,
        "max_hydrophobic_patch": hpatch.get("max_patch_size") if hpatch.get("computable") else None,
        "net_surface_charge": elec.get("net_surface_charge") if elec.get("computable") else None,
        "mean_convexity": curv.get("mean_convexity") if curv.get("computable") else None,
    }
    return {
        "computable": True,
        "derivation": "pdb_3d",
        "pdb_id": structure.get("pdb_id"),
        "native_or_fibril": native_or_fibril,
        "is_nmr_ensemble": structure.get("is_nmr_ensemble", False),
        "n_models_in_ensemble": structure.get("n_models", 1),
        "n_chains": structure.get("n_chains"),
        "n_residues": structure.get("n_residues"),
        "cutoff_angstrom": cutoff_angstrom,
        "summary": summary,
        "fidelity_by_feature": {
            "contact_map": "exact",
            "real_sasa": "exact_algorithm (Shrake-Rupley)",
            "secondary_structure": "approximation (backbone phi/psi geometry, NOT DSSP)",
            "hydrophobic_surface_patches": "real (SASA-exposed + contact clustering)",
            "electrostatic_surface": "approximation (Coulombic/Debye-Huckel, NOT APBS)",
            "surface_curvature": "approximation (CA neighbour-density convexity)",
        },
        "contact_map": contact,
        "real_sasa": sasa,
        "secondary_structure": ss,
        "hydrophobic_surface_patches": hpatch,
        "electrostatic_surface": elec,
        "surface_curvature": curv,
        "native_vs_fibril_note": _NATIVE_FIBRIL_NOTE,
    }


# ============================================================================ #
# 2. BUILDABLE-NOW APR / SEQUENCE PROXY FEATURES (deterministic, 1-D)
# ============================================================================ #
def _clean_peptide(seq):
    """Uppercase + keep only standard 20 AAs (drops gaps/X/unknowns)."""
    if not seq:
        return ""
    return "".join(c for c in str(seq).upper() if c in STANDARD_AAS)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def max_hydrophobic_patch(seq, window=3):
    """Max mean Kyte–Doolittle hydropathy over any contiguous `window` residues —
    a 1-D hydrophobic 'patch' proxy. WHY a proxy: a REAL hydrophobic surface patch
    needs 3D coordinates + SASA (a stub below); this is the sequence-local analogue
    and is stamped as such."""
    s = _clean_peptide(seq)
    if len(s) < window:
        return _mean([KYTE_DOOLITTLE[c] for c in s]) if s else None
    best = None
    for i in range(len(s) - window + 1):
        m = sum(KYTE_DOOLITTLE[c] for c in s[i:i + window]) / window
        best = m if best is None else max(best, m)
    return best


def peptide_proxy_features(seq):
    """Deterministic sequence proxies for ONE APR peptide (§3-M16 step 2).

    Returns Chou–Fasman Pβ/Pα (mean over residues), Kyte–Doolittle hydrophobicity
    (mean + max 3-residue patch), and a crude net charge / pI proxy. Every value is
    a 1-D SEQUENCE proxy — NOT a 3D-derived quantity."""
    s = _clean_peptide(seq)
    if not s:
        return {"n_residues": 0, "beta_propensity": None, "alpha_propensity": None,
                "hydrophobicity_mean": None, "hydrophobic_patch_max": None,
                "net_charge": None, "crude_pI_proxy": None}
    pb = _mean([CHOU_FASMAN_PB[c] for c in s])
    pa = _mean([CHOU_FASMAN_PA[c] for c in s])
    kd = _mean([KYTE_DOOLITTLE[c] for c in s])
    patch = max_hydrophobic_patch(s, window=3)
    net_charge = sum(CHARGE_AT_PH7.get(c, 0.0) for c in s)
    # crude pI proxy: acidic-heavy -> low, basic-heavy -> high, centred at 7.
    n_pos = sum(1 for c in s if c in ("K", "R")) + 0.1 * s.count("H")
    n_neg = sum(1 for c in s if c in ("D", "E"))
    pI = 7.0 + (n_pos - n_neg)  # 1 pH-unit per net charge — a crude monotone proxy
    return {
        "n_residues": len(s),
        "beta_propensity": round(pb, 4),
        "alpha_propensity": round(pa, 4),
        "hydrophobicity_mean": round(kd, 4),
        "hydrophobic_patch_max": round(patch, 4) if patch is not None else None,
        "net_charge": round(net_charge, 4),
        "crude_pI_proxy": round(pI, 4),
    }


def _parse_span(position):
    """Parse an APR position like '17-24' -> (17, 24); tolerant of junk -> None."""
    if not position:
        return None
    try:
        a, b = str(position).split("-", 1)
        return int(float(a)), int(float(b))
    except Exception:
        return None


def apr_proxy_features(aprs):
    """Aggregate the per-peptide proxies over ALL of a protein's APR peptides.

    Produces the protein-level sequence/APR proxy feature set: mean/max Pβ, Pα,
    hydrophobicity + patch, net charge, APR count / total length / (relative)
    coverage, plus the aggregation-hotspot summary (positions + categories).
    STAMPED derivation='sequence_apr_proxy' — explicitly NOT 3D."""
    aprs = aprs or []
    per = []
    hotspots = []
    total_len = 0
    cat_counts = Counter()
    max_end = 0
    for a in aprs:
        region = a.get("region")
        pf = peptide_proxy_features(region)
        cat = a.get("category") or "uncategorized"
        cat_counts[cat] += 1
        span = _parse_span(a.get("position"))
        length = a.get("length")
        try:
            length = int(float(length)) if length is not None else (
                len(_clean_peptide(region)) if region else None)
        except Exception:
            length = len(_clean_peptide(region)) if region else None
        if length:
            total_len += length
        if span:
            max_end = max(max_end, span[1])
        hotspots.append({
            "position": a.get("position"), "span": list(span) if span else None,
            "length": length, "category": cat,
            "region": region, "proxy": pf,
        })
        if pf["n_residues"] > 0:
            per.append(pf)

    def _agg(key, fn):
        vals = [p[key] for p in per if p.get(key) is not None]
        return round(fn(vals), 4) if vals else None

    coverage = None
    if max_end > 0 and total_len > 0:
        # relative coverage proxy: fraction of the APR-spanned region that is APR.
        # NOT protein-length coverage (full sequence unavailable) — labelled so.
        coverage = round(min(1.0, total_len / max_end), 4)

    feats = {
        "derivation": "sequence_apr_proxy",
        "is_3d_derived": False,
        "n_aprs": len(aprs),
        "n_aprs_with_sequence": len(per),
        "total_apr_length": total_len,
        "apr_span_max_end": max_end or None,
        "apr_coverage_proxy": coverage,
        "apr_coverage_note": ("fraction of the APR-spanned region occupied by APRs; "
                              "NOT full-protein coverage (full sequence unavailable)"),
        "beta_propensity_mean": _agg("beta_propensity", _mean),
        "beta_propensity_max": _agg("beta_propensity", max),
        "alpha_propensity_mean": _agg("alpha_propensity", _mean),
        "hydrophobicity_mean": _agg("hydrophobicity_mean", _mean),
        "hydrophobic_patch_max": _agg("hydrophobic_patch_max", max),
        "net_charge_total": round(sum(p["net_charge"] for p in per), 4) if per else None,
        "net_charge_mean": _agg("net_charge", _mean),
        "crude_pI_proxy_mean": _agg("crude_pI_proxy", _mean),
        "apr_category_counts": dict(cat_counts),
        "aggregation_hotspots": hotspots,
        "scales_cited": {
            "hydrophobicity": "Kyte & Doolittle 1982 (J Mol Biol 157:105)",
            "beta_alpha_propensity": "Chou & Fasman 1978 (Adv Enzymol 47:45)",
            "charge": "crude pH7 proxy D/E=-1, K/R=+1, H=+0.1 (electrostatics proxy)",
        },
    }
    return feats


# ============================================================================ #
# 3. THE 8 REQUESTED STRUCTURAL FEATURES (honest proxy / stub split)
# ============================================================================ #
def _proxy_marker(value, real_dep, real_feature, proxy_of):
    """A machine-readable proxy marker: the buildable-now value + the honest
    statement of what the REAL 3D feature would need (a deferred stub)."""
    return {
        "value": value,
        "derivation": "sequence_apr_proxy",
        "is_3d_derived": False,
        "proxy_of": proxy_of,
        "real_feature_deferred": real_feature,
        "real_feature_requires": real_dep,
        "native_vs_fibril_note": _NATIVE_FIBRIL_NOTE,
    }


def _real_3d_marker(real_summary, key, real_from, real_fidelity):
    """A machine-readable REAL-3D feature marker (derivation=pdb_3d). Present only
    when a PDB was fetched + parsed for this protein."""
    return {
        "value": (real_summary or {}).get(key),
        "derivation": "pdb_3d",
        "is_3d_derived": True,
        "fidelity": real_fidelity,
        "real_from": real_from,
        "native_vs_fibril_note": _NATIVE_FIBRIL_NOTE,
    }


def structural_features(aprs, structures=None, real_features=None):
    """The 8 requested structural features. When a REAL PDB was fetched + parsed
    (`real_features` = compute_real_structural_features output) the genuinely-3D
    values are reported (derivation=pdb_3d, per-feature fidelity, native/fibril
    stamp) ALONGSIDE the sequence/APR proxy, so the real-vs-proxy contrast is visible.
    Without a PDB (or a failed fetch/parse) only the proxy is reported, FLAGGED.

    REAL 3D (fidelity exact / approximation, when a PDB is present):
        contact_map (EXACT), real_sasa/solvent_accessibility (EXACT ALGORITHM
        Shrake-Rupley), secondary_structure/beta_sheet_content (APPROXIMATION φ/ψ),
        hydrophobic_patches (REAL SASA+contact), electrostatic_surface
        (APPROXIMATION Coulomb/Debye-Huckel), surface_curvature (APPROXIMATION
        Cα neighbour density).
    REAL from APRs: aggregation_hotspots.
    PROXY (1-D sequence, always present): beta/SASA/hydrophobic/electrostatic/
        secondary Chou-Fasman + Kyte-Doolittle + net-charge analogues."""
    ap = apr_proxy_features(aprs)
    src_stamp = summarize_structure_sources(structures or [])
    rf = real_features if (real_features and real_features.get("computable")) else None
    rsum = (rf or {}).get("summary", {})
    real_pdb_id = (rf or {}).get("pdb_id")
    real_nof = (rf or {}).get("native_or_fibril", "unknown")
    has_real = rf is not None

    def _feature(proxy_marker, real_key, real_from, real_fidelity):
        """Bundle: always the proxy; the REAL 3D value too when a PDB was parsed."""
        block = {"proxy": proxy_marker, "has_real_3d": has_real}
        if has_real:
            block["real_3d"] = _real_3d_marker(rsum, real_key, real_from, real_fidelity)
        return block

    out = {
        "derivation_summary": (
            "REAL-3D (pdb_3d, when a PDB is fetched): contact_map(EXACT), "
            "real_sasa(EXACT Shrake-Rupley), secondary_structure(APPROX phi/psi), "
            "hydrophobic_patches(REAL), electrostatic_surface(APPROX Coulomb/DH), "
            "surface_curvature(APPROX). REAL-APR: aggregation_hotspots. PROXY: 1-D "
            "sequence analogues (always present)."),
        "has_real_3d_features": has_real,
        "real_3d_pdb_id": real_pdb_id,
        "real_3d_native_or_fibril": real_nof,
        "structure_sources": src_stamp,
        # --- REAL (directly from the shipped APR peptides) ---
        "aggregation_hotspots": {
            "value": ap["aggregation_hotspots"],
            "derivation": "apr_real",
            "is_3d_derived": False,
            "note": "REAL: the shipped APR peptides (positions + categories).",
        },
        # --- the 6 genuinely-3D features (REAL when a PDB is present, else PROXY) ---
        "beta_sheet_content": _feature(
            _proxy_marker(ap["beta_propensity_mean"], "DSSP/STRIDE + coordinates",
                          "DSSP secondary-structure beta fraction",
                          "Chou-Fasman beta-sheet propensity (mean Pbeta over APR residues)"),
            "beta_sheet_content", "secondary_structure (%E from phi/psi)",
            "approximation (backbone phi/psi geometry, NOT DSSP H-bonding)"),
        "solvent_accessibility": _feature(
            _proxy_marker(ap["hydrophobicity_mean"], "freesasa/DSSP + coordinates",
                          "per-residue SASA",
                          "Kyte-Doolittle hydrophobicity (high hydrophobicity ~ low SASA; "
                          "INVERSE, sequence-local proxy only)"),
            "mean_rel_sasa", "real_sasa (Shrake-Rupley mean rel-SASA)",
            "exact_algorithm (Shrake-Rupley)"),
        "hydrophobic_patches": _feature(
            _proxy_marker(ap["hydrophobic_patch_max"], "freesasa surface + coordinates",
                          "3D surface hydrophobic patches",
                          "max 3-residue Kyte-Doolittle window (1-D patch)"),
            "n_hydrophobic_patches", "hydrophobic_surface_patches (n patches)",
            "real (SASA-exposed + contact clustering)"),
        "electrostatic_surface": _feature(
            _proxy_marker(ap["net_charge_total"], "APBS/pdb2pqr + coordinates",
                          "Poisson-Boltzmann surface potential",
                          "sequence net charge at pH7 (D/E=-1,K/R=+1,H=+0.1)"),
            "net_surface_charge", "electrostatic_surface (net exposed charge)",
            "approximation (Coulombic/Debye-Huckel, NOT APBS Poisson-Boltzmann)"),
        "secondary_structure": _feature(
            _proxy_marker({"beta_propensity": ap["beta_propensity_mean"],
                           "alpha_propensity": ap["alpha_propensity_mean"]},
                          "DSSP/STRIDE + coordinates", "DSSP 8/3-state secondary structure",
                          "Chou-Fasman Pbeta/Palpha propensities"),
            "helix_content", "secondary_structure (%H from phi/psi)",
            "approximation (backbone phi/psi geometry, NOT DSSP H-bonding)"),
        "contact_map": _contact_map_feature(rf),
        "surface_curvature": _feature(
            _proxy_marker(None, "3D surface mesh + coordinates",
                          "differential-geometry surface curvature",
                          "no 1-D sequence analogue (curvature is intrinsically 3D)"),
            "mean_convexity", "surface_curvature (CA neighbour-density convexity)",
            "approximation (CA neighbour-density convexity)"),
    }
    return out


def _contact_map_feature(real_features):
    """Contact map is intrinsically 3D — no proxy. REAL when a PDB was parsed, else
    a computable:false marker (fetch failed / no PDB reference)."""
    rf = real_features if (real_features and real_features.get("computable")) else None
    if rf is None:
        return {
            "has_real_3d": False,
            "computable": False,
            "is_3d_derived": True,
            "real_feature_requires": "3D coordinates + a distance cutoff (also the "
                                     "GNN contact edge set)",
            "note": "no fetched PDB for this protein (or fetch/parse failed); proxy "
                    "fallback has no contact-map analogue (intrinsically 3D).",
        }
    cm = rf.get("contact_map", {})
    return {
        "has_real_3d": True,
        "computable": True,
        "is_3d_derived": True,
        "derivation": "pdb_3d",
        "fidelity": "exact",
        "value": {"n_contacts": cm.get("n_contacts"),
                  "relative_contact_order": cm.get("relative_contact_order"),
                  "mean_contact_number": cm.get("mean_contact_number")},
        "native_vs_fibril_note": _NATIVE_FIBRIL_NOTE,
    }


# compute_dssp_secondary_structure was renamed to compute_secondary_structure (now a
# REAL geometry-based extractor, see section 1b). Keep a back-compatible alias whose
# name no longer implies DSSP H-bonding (fidelity is APPROXIMATION, stamped there).
compute_dssp_secondary_structure = compute_secondary_structure


# ============================================================================ #
# 4. GNN-READY GRAPH REPRESENTATION (forward-compatible; NO GNN built)
# ============================================================================ #
def build_graph(aprs, uniprot=None, structure=None, real_features=None):
    """Emit a stable GNN-consumable graph for a protein/structure (§3-M16 step 4).

    APR-peptide layer: nodes = residues of the shipped APR peptides; each carries
    node_features {hydrophobicity, charge, beta_propensity, alpha_propensity,
    apr_member}; edges = sequence adjacency within each APR. When a PDB was fetched +
    parsed (`real_features`), a SEPARATE `real_contact_graph` holds the REAL
    structural contact edges (Cβ distance) — the true GNN contact edge set. The
    schema is stable so a GNN could consume it later — NO GNN is built here."""
    nodes = []
    edges = []
    node_index = 0
    for apr_i, a in enumerate(aprs or []):
        s = _clean_peptide(a.get("region"))
        span = _parse_span(a.get("position"))
        start = span[0] if span else None
        first_node_of_apr = node_index
        for k, aa in enumerate(s):
            residue_index = (start + k) if start is not None else None
            nodes.append({
                "node_id": node_index,
                "residue_index": residue_index,
                "aa": aa,
                "apr_id": apr_i,
                "node_features": {
                    "hydrophobicity": KYTE_DOOLITTLE[aa],
                    "charge": CHARGE_AT_PH7.get(aa, 0.0),
                    "beta_propensity": CHOU_FASMAN_PB[aa],
                    "alpha_propensity": CHOU_FASMAN_PA[aa],
                    "apr_member": 1,
                },
            })
            # sequence-adjacency edge within this APR peptide
            if k > 0:
                edges.append({"i": node_index - 1, "j": node_index,
                              "type": "sequence_adjacency"})
            node_index += 1
        _ = first_node_of_apr  # (kept for clarity; APRs are not cross-linked offline)

    # REAL contact edges: when a PDB was fetched + parsed we build a SEPARATE
    # residue-level graph whose nodes are the STRUCTURE's residues and whose edges
    # are the REAL Cβ-distance contacts (compute_contact_map). This is the true GNN
    # contact edge set; it lives beside the APR-peptide graph above.
    real_graph = None
    contact_available = False
    if real_features and real_features.get("computable"):
        cm = real_features.get("contact_map", {})
        if cm.get("computable"):
            contact_available = True
            ss = real_features.get("secondary_structure", {})
            ss_by_res = {(r["chain"], r["resnum"]): r["ss"]
                         for r in ss.get("per_residue", [])} if ss.get("computable") else {}
            sasa = real_features.get("real_sasa", {})
            rsa_by_res = {(r["chain"], r["resnum"]): r
                          for r in sasa.get("per_residue", [])} if sasa.get("computable") else {}
            # residue nodes in contact-map index order (the edge indices reference these)
            edges_arr = cm.get("contact_edges", [])
            # collect the residue set that the contact edges touch, in first-seen order
            rnodes = []
            seen = {}
            cn = cm.get("per_residue_contact_number", [])
            for e in edges_arr:
                for side in ("i", "j"):
                    idx = e[side]
                    if idx not in seen:
                        seen[idx] = len(rnodes)
                        chain = e["chain_%s" % side]
                        resnum = e["resnum_%s" % side]
                        rr = rsa_by_res.get((chain, resnum), {})
                        rnodes.append({
                            "contact_index": idx, "chain": chain, "resnum": resnum,
                            "aa": rr.get("aa"),
                            "ss": ss_by_res.get((chain, resnum)),
                            "rel_sasa": rr.get("rel_sasa"), "exposed": rr.get("exposed"),
                            "contact_number": (cn[idx] if idx < len(cn) else None),
                        })
            real_graph = {
                "structure_pdb_id": real_features.get("pdb_id"),
                "native_or_fibril": real_features.get("native_or_fibril", "unknown"),
                "n_nodes": len(rnodes),
                "n_contact_edges": len(edges_arr),
                "nodes": rnodes,
                "contact_edges": edges_arr,
                "edge_type": "structural_contact_Cbeta_lt_%.1fA" % cm.get("cutoff_angstrom", 8.0),
                "relative_contact_order": cm.get("relative_contact_order"),
                "derivation": "pdb_3d",
                "fidelity": "exact",
                "note": ("REAL GNN contact graph: nodes=structure residues, "
                         "edges=Cbeta-distance contacts. " + _NATIVE_FIBRIL_NOTE),
            }
    return {
        "uniprot": uniprot,
        "structure_id": (structure or {}).get("id") if structure else None,
        "native_or_fibril": (structure or {}).get("native_or_fibril") if structure else "unknown",
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
        "edge_types_present": sorted({e["type"] for e in edges}),
        "contact_edges_available": contact_available,
        "real_contact_graph": real_graph,
        "note": ("GNN-ready REPRESENTATION only — NO GNN is built. APR-peptide nodes "
                 "carry sequence-adjacency edges. When a PDB was fetched, `real_contact_"
                 "graph` holds the REAL structural contact edges (Cbeta distance). "
                 + _NATIVE_FIBRIL_NOTE),
        "node_feature_schema": ["hydrophobicity", "charge", "beta_propensity",
                                "alpha_propensity", "apr_member"],
    }


def summarize_structure_sources(structures):
    """Fold a protein's structures[] into source + native/fibril stamps. (Coordinate
    availability is decided per-protein by the real-3D fetch path, not here.)"""
    structs = [structure_from_pdb_record(s) for s in (structures or [])]
    src_counts = Counter(s["source"] for s in structs)
    nof_counts = Counter(s["native_or_fibril"] for s in structs)
    return {
        "n_structures": len(structs),
        "has_pdb_reference": len(structs) > 0,
        "source_counts": dict(src_counts),
        "native_or_fibril_counts": dict(nof_counts),
        "has_fibril_structure": nof_counts.get("fibril", 0) > 0,
        "has_native_structure": nof_counts.get("native", 0) > 0,
        "structure_ids": [s["id"] for s in structs],
        "native_vs_fibril_note": _NATIVE_FIBRIL_NOTE,
    }


def _uniprot_to_kinetics_pdb():
    """Invert KINETICS_PDB_TO_UNIPROT -> {uniprot: [(pdb_id, native_or_fibril), ...]},
    preferring native-monomer structures first (they are the ones whose real-3D
    features are least confounded by the native-vs-fibril paradox)."""
    out = defaultdict(list)
    for pid, (up, nof) in KINETICS_PDB_TO_UNIPROT.items():
        out[up].append((pid, nof))
    for up in out:
        # native first, then by PDB id for determinism
        out[up].sort(key=lambda t: (0 if t[1] == "native" else 1, t[0]))
    return dict(out)


# ============================================================================ #
# PER-PROTEIN STRUCTURAL FEATURE + GRAPH BUILDER
# ============================================================================ #
def build_protein_structural_features(seq_lookup, restrict_uniprots=None,
                                      fetch=True, cutoff_angstrom=8.0,
                                      fetch_log=None) -> dict:
    """Per-protein: sequence/APR proxy features + the 8 structural features + a GNN
    graph. For a protein with a kinetics-linked PDB, ALSO fetch the coordinates and
    compute the REAL 3D features (derivation=pdb_3d, per-feature fidelity, native/
    fibril stamp) and populate the REAL contact edges in the graph. A protein without
    a PDB (or a failed fetch/parse) keeps the sequence/APR proxy, FLAGGED — never
    raises. `fetch_log` (a dict) collects the fetch outcomes for the CLI report."""
    by_up = (seq_lookup or {}).get("by_uniprot", {}) or {}
    up2pdb = _uniprot_to_kinetics_pdb()
    out = {}
    for up, rec in by_up.items():
        if restrict_uniprots is not None and up not in restrict_uniprots:
            continue
        rec = rec or {}
        aprs = rec.get("aprs") or []
        structures = rec.get("structures") or []
        proxy = apr_proxy_features(aprs)

        # --- REAL 3D path: fetch + parse the best kinetics-linked PDB (if any) ---
        real_features = None
        real_source = None
        for pid, nof in up2pdb.get(up, []):
            struct = load_structure("PDB", pid) if fetch else None
            if struct is None:
                if fetch_log is not None:
                    fetch_log.setdefault("failed", []).append({"uniprot": up, "pdb_id": pid})
                continue
            struct["native_or_fibril"] = nof
            struct["is_fibril"] = (nof == "fibril")
            rf = compute_real_structural_features(struct, native_or_fibril=nof,
                                                  cutoff_angstrom=cutoff_angstrom)
            if rf and rf.get("computable"):
                real_features = rf
                real_source = {"pdb_id": pid, "native_or_fibril": nof}
                if fetch_log is not None:
                    fetch_log.setdefault("ok", []).append(
                        {"uniprot": up, "pdb_id": pid, "native_or_fibril": nof})
                break
            elif fetch_log is not None:
                fetch_log.setdefault("parse_failed", []).append({"uniprot": up, "pdb_id": pid})

        out[up] = {
            "uniprot": up,
            "protein_name": rec.get("protein_name"),
            "n_aprs": len(aprs),
            "has_apr_sequence": proxy["n_aprs_with_sequence"] > 0,
            "has_real_3d_features": real_features is not None,
            "real_3d_source": real_source,
            "apr_proxy_features": proxy,
            "real_3d_features": real_features,       # None when no PDB / fetch failed
            "structural_features": structural_features(aprs, structures,
                                                       real_features=real_features),
            "graph": build_graph(aprs, uniprot=up, real_features=real_features),
        }
    return out


# ============================================================================ #
# 5. ASSOCIATION LAYER — structure features -> kinetics (protein-level)
# ============================================================================ #
# Features tested (protein-level). Each: (label, source, key) where source is
# "proxy" (from apr_proxy_features) or "real3d" (from real_3d_features.summary — the
# REAL coordinate-derived features). The real-3D predictors are the headline upgrade.
_ASSOC_FEATURES = [
    # sequence/APR PROXY predictors (retained for the real-vs-proxy contrast)
    ("beta_propensity_mean", "proxy", "beta_propensity_mean"),
    ("hydrophobicity_mean", "proxy", "hydrophobicity_mean"),
    ("net_charge_total", "proxy", "net_charge_total"),
    ("n_aprs", "proxy", "n_aprs"),
    # REAL 3D predictors (derivation=pdb_3d) — the requested new association inputs
    ("real_mean_rel_sasa", "real3d", "mean_rel_sasa"),
    ("real_beta_sheet_content", "real3d", "beta_sheet_content"),
    ("real_relative_contact_order", "real3d", "relative_contact_order"),
    ("real_n_hydrophobic_patches", "real3d", "n_hydrophobic_patches"),
    ("real_net_surface_charge", "real3d", "net_surface_charge"),
    ("real_mean_convexity", "real3d", "mean_convexity"),
]

# The kinetics targets: (label, protein-table key, kind).
_ASSOC_TARGETS = [
    ("lag_time", "median_lag_to_t50_ratio", "continuous"),
    ("median_log10_t50", "median_log10_t50_point", "continuous"),
    ("mechanistic_regime", "max_regime_ordinal", "ordinal"),
]


def _bh_fdr(pvals):
    """Benjamini–Hochberg q-values for a list of p (None passed through as None)."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    m = len(idx)
    q = [None] * len(pvals)
    if m == 0:
        return q
    order = sorted(idx, key=lambda i: pvals[i])
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = min(prev, pvals[i] * m / (rank + 1))
        q[i] = round(val, 6)
        prev = val
    return q


def _feature_value(sf, source, feat_key):
    """Pull one feature value from a struct_feats entry by source: proxy (from
    apr_proxy_features) or real3d (from real_3d_features.summary). None if absent."""
    if source == "real3d":
        rf = sf.get("real_3d_features")
        if not rf or not rf.get("computable"):
            return None
        return (rf.get("summary") or {}).get(feat_key)
    return (sf.get("apr_proxy_features") or {}).get(feat_key)


def _feature_vector(struct_feats, prot_table, source, feat_key, target_key):
    """Paired protein-level (x=structure feature, y=kinetics target) over proteins
    present in BOTH tables with finite values."""
    xs, ys, ups = [], [], []
    for up, d in prot_table.items():
        sf = struct_feats.get(up)
        if not sf:
            continue
        x = _feature_value(sf, source, feat_key)
        y = d.get(target_key)
        if x is None or y is None:
            continue
        try:
            xf, yf = float(x), float(y)
        except Exception:
            continue
        if not (math.isfinite(xf) and math.isfinite(yf)):
            continue
        xs.append(xf)
        ys.append(yf)
        ups.append(up)
    return xs, ys, ups


def _verdict(n, spearman_rho, perm_p, fdr_q, skill_for_verdict):
    """Blunt verdict: association | insufficient_power | null. At n<=24 EXPECT
    insufficient_power. Significance requires BOTH FDR survival AND positive LOPO
    out-of-sample skill (a permutation hit without skill is not an association)."""
    if n < 4 or spearman_rho is None or perm_p is None:
        return "insufficient_power"
    survives_fdr = (fdr_q is not None and fdr_q < FDR_ALPHA)
    has_skill = (skill_for_verdict is not None and skill_for_verdict > 0.0)
    # an 'association' is only licensed above the minimum-n floor: below it, even a
    # significant + skilful hit is untrustworthy (tiny n looks monotone by chance).
    if survives_fdr and has_skill and n >= MIN_N_FOR_ASSOCIATION:
        return "association"
    # small n with no signal at all -> insufficient_power; a clear non-signal -> null
    if n < 10:
        return "insufficient_power"
    return "null"


def _spearman_ci(rho, n):
    """Fisher-z 95% CI for a Spearman rho (approximate; honest effect-size band)."""
    if rho is None or n < 4 or abs(rho) >= 1.0:
        return None
    z = math.atanh(rho)
    se = 1.0 / math.sqrt(n - 3)
    lo, hi = math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
    return [round(lo, 4), round(hi, 4)]


def build_associations(struct_feats, prot_table, ceiling_by_target=None) -> dict:
    """Associate structure features (REAL 3D + sequence/APR PROXY) -> kinetics at the
    PROTEIN level (§3-M16 step 5), REUSING M10 (Spearman + LOPO + permutation null +
    variance ceiling + BH-FDR). causal:false is an INVARIANT on every card. At n<=24
    most verdicts are insufficient_power/null — reported honestly."""
    ceiling_by_target = ceiling_by_target or {}
    cards = []
    for feat_label, feat_source, feat_key in _ASSOC_FEATURES:
        for tgt_label, tgt_key, kind in _ASSOC_TARGETS:
            xs, ys, ups = _feature_vector(struct_feats, prot_table, feat_source,
                                          feat_key, tgt_key)
            n = len(xs)
            rho = _spearman(xs, ys) if n >= 3 else None
            _, perm_p, n_perm = (_perm_pvalue_corr(xs, ys, kind="spearman")
                                 if n >= 4 else (None, None, 0))
            if kind == "ordinal":
                skill = lopo_ordinal_accuracy(xs, ys) if n >= 4 else {"computable": False}
                lopo_r2_val = None
                skill_for_verdict = skill.get("skill_above_baseline")
            else:
                skill = lopo_r2(xs, ys) if n >= 4 else {"computable": False}
                lopo_r2_val = skill.get("lopo_r2")
                skill_for_verdict = lopo_r2_val
            ceil_frac = ceiling_by_target.get(tgt_key)
            is_real = (feat_source == "real3d")
            confounders = [
                "native structure != aggregation-competent/fibril state (many PDBs here "
                "are the NATIVE monomer, NOT the aggregated fibril)",
                "fibril polymorphism ('the structure of protein P' is condition-dependent)",
                "selection bias: corpus enriched for famous amyloids (no negatives)",
                "one sequence maps to many rates across conditions (variance-ceiling bound)",
            ]
            if is_real:
                confounders.append(
                    "REAL 3D feature from a single deposited structure (often a native "
                    "monomer under crystallisation/NMR conditions), NOT the fibril state")
            else:
                confounders.append(
                    "PROXY feature is 1-D sequence/APR-derived, NOT a 3D-measured quantity")
            cards.append({
                "feature": feat_label,
                "feature_source": feat_source,
                "derivation": ("pdb_3d" if is_real else "sequence_apr_proxy"),
                "target": tgt_label,
                "n_proteins": n,
                "unit": "protein",
                "spearman_rho": (round(rho, 4) if rho is not None else None),
                "permutation_p": (round(perm_p, 6) if perm_p is not None else None),
                "permutation_n_draws": n_perm,
                "fdr_q": None,  # filled after the batch (BH across all cards)
                "lopo_r2": lopo_r2_val,
                "lopo_skill_over_baseline": skill_for_verdict,
                "lopo_detail": skill,
                "variance_ceiling_fraction": (round(ceil_frac, 4)
                                              if ceil_frac is not None else None),
                "effect_ci": _spearman_ci(rho, n),
                "confounders": confounders,
                "causal": False,                 # INVARIANT — never true
                "basis": "statistical_association",
                "experimentally_validated": False,  # external human flag only
                "verdict": _verdict(n, rho, perm_p, None, skill_for_verdict),
            })
    # BH-FDR across the whole batch of tested feature x target pairs, then re-verdict
    qs = _bh_fdr([c["permutation_p"] for c in cards])
    for c, q in zip(cards, qs):
        c["fdr_q"] = q
        c["verdict"] = _verdict(c["n_proteins"], c["spearman_rho"],
                                c["permutation_p"], q, c["lopo_skill_over_baseline"])
    verdict_counts = Counter(c["verdict"] for c in cards)
    n_assoc_fdr = sum(1 for c in cards
                      if c["verdict"] == "association"
                      and c["fdr_q"] is not None and c["fdr_q"] < FDR_ALPHA)
    real_cards = [c for c in cards if c["feature_source"] == "real3d"]
    proxy_cards = [c for c in cards if c["feature_source"] == "proxy"]
    n_real_tested = sum(1 for c in real_cards if c["n_proteins"] >= 4)
    max_n_real = max((c["n_proteins"] for c in real_cards), default=0)
    return {
        "n_cards": len(cards),
        "n_real_3d_cards": len(real_cards),
        "n_proxy_cards": len(proxy_cards),
        "n_real_3d_cards_testable": n_real_tested,
        "max_n_proteins_real_3d": max_n_real,
        "fdr_alpha": FDR_ALPHA,
        "association_cards": cards,
        "verdict_counts": dict(verdict_counts),
        "n_significant_after_fdr": n_assoc_fdr,
        "causal_invariant": "causal:false on EVERY card (no card is ever causal)",
        "honesty": (
            "Effective n = PROTEINS (<=24), so power is LOW and most verdicts are "
            "insufficient_power/null (as expected). The REAL 3D predictors have even "
            "FEWER proteins (only those with a fetched PDB, n=%d max) so they are "
            "MORE underpowered, not less. Every association is ASSOCIATIVE, never "
            "causal; experimentally_validated is an external human flag only; REAL 3D "
            "features come from a single deposited (often native-monomer) structure, "
            "NOT the fibril state (native != aggregation-competent)." % max_n_real),
    }


# ============================================================================ #
# ASSEMBLER
# ============================================================================ #
def _ceilings_by_target(prot_table) -> dict:
    """Between-protein variance ceiling for each CONTINUOUS target (log10 t50).
    The ordinal regime target has no ANOVA ceiling; only t50 does here."""
    reps = {up: d.get("log10_t50_replicates") or []
            for up, d in prot_table.items() if d.get("log10_t50_replicates")}
    vc = variance_ceiling(reps)
    frac = vc.get("between_protein_fraction") if vc.get("computable") else None
    return {"median_log10_t50_point": frac, "_detail": vc}


def build_structure_bridge(triaged, features, regimes, seq_lookup, fetch=True) -> dict:
    """Assemble the full M16 payload. Pure (deterministic given the PDB cache); never
    raises on bad input. `fetch=True` downloads+parses the kinetics-linked PDBs and
    computes REAL 3D features; `fetch=False` runs proxy-only (for offline tests)."""
    try:
        prot_table = build_protein_table(triaged or [], features or [],
                                         regimes or [], seq_lookup or {})
    except Exception as exc:                                   # defensive, never raise
        return {"version": VERSION, "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "structural_features": {}, "associations": {}}
    kinetics_ups = set(prot_table.keys())
    fetch_log = {}
    struct_feats = build_protein_structural_features(
        seq_lookup, restrict_uniprots=None, fetch=fetch, fetch_log=fetch_log)
    # association is over the kinetics corpus (proteins present in both)
    ceilings = _ceilings_by_target(prot_table)
    ceiling_by_target = {"median_log10_t50_point": ceilings["median_log10_t50_point"]}
    associations = build_associations(
        {u: struct_feats[u] for u in struct_feats if u in kinetics_ups},
        prot_table, ceiling_by_target=ceiling_by_target)

    n_with_apr = sum(1 for u in kinetics_ups
                     if struct_feats.get(u, {}).get("has_apr_sequence"))
    n_with_pdb = sum(1 for u in kinetics_ups
                     if (struct_feats.get(u, {}) or {})
                     .get("structural_features", {})
                     .get("structure_sources", {}).get("n_structures", 0) > 0)
    n_with_real3d = sum(1 for u in kinetics_ups
                        if struct_feats.get(u, {}).get("has_real_3d_features"))
    # native/fibril breakdown of the fetched real structures actually used
    nof_used = Counter()
    real_pdbs_used = []
    for u in kinetics_ups:
        src = struct_feats.get(u, {}).get("real_3d_source")
        if src:
            nof_used[src["native_or_fibril"]] += 1
            real_pdbs_used.append({"uniprot": u, "pdb_id": src["pdb_id"],
                                   "native_or_fibril": src["native_or_fibril"]})
    n_fetch_ok = len({(x["uniprot"], x["pdb_id"]) for x in fetch_log.get("ok", [])})
    n_fetch_fail = len(fetch_log.get("failed", []))
    return {
        "version": VERSION,
        "status": "ok",
        "coverage": {
            "n_proteins_total_seq": len(struct_feats),
            "n_proteins_in_kinetics": len(kinetics_ups),
            "n_proteins_with_apr_proxy_in_kinetics": n_with_apr,
            "n_proteins_with_pdb_ref_in_kinetics": n_with_pdb,
            "n_proteins_with_real_3d_in_kinetics": n_with_real3d,
            "real_3d_native_count": nof_used.get("native", 0),
            "real_3d_fibril_count": nof_used.get("fibril", 0),
            "real_3d_structures_used": sorted(real_pdbs_used, key=lambda x: x["pdb_id"]),
            "n_pdb_fetch_ok": n_fetch_ok,
            "n_pdb_fetch_failed": n_fetch_fail,
            "effective_n_is_proteins_not_curves": True,
            "banner": (f"Effective n = {n_with_apr} PROTEINS with APR-proxy features "
                       f"(of {len(kinetics_ups)} in the kinetics corpus). REAL 3D "
                       f"features from fetched PDBs for {n_with_real3d} proteins "
                       f"({nof_used.get('native', 0)} native, {nof_used.get('fibril', 0)} "
                       f"fibril) — NOT the ~1600 curves. Power is low by design; the "
                       f"real-3D subset is even smaller."),
        },
        "pdb_fetch_log": {
            "ok": fetch_log.get("ok", []),
            "failed": fetch_log.get("failed", []),
            "parse_failed": fetch_log.get("parse_failed", []),
        },
        "variance_ceiling_log10_t50": ceilings["_detail"],
        "structural_features_by_uniprot": struct_feats,
        "associations": associations,
        "honesty_constraints": [
            "REAL 3D FEATURES (v2.0): contact map, Shrake-Rupley SASA, phi/psi "
            "secondary structure, hydrophobic surface patches, electrostatic surface, "
            "surface curvature — computed in PURE numpy from FETCHED PDB coordinates.",
            "EXACT vs APPROXIMATION stamped per feature: EXACT = contact_map (distance "
            "threshold on real coords); EXACT ALGORITHM = real_sasa (Shrake-Rupley). "
            "APPROXIMATION = secondary_structure (phi/psi geometry, NOT DSSP H-bonds), "
            "electrostatic_surface (Coulomb/Debye-Huckel, NOT APBS Poisson-Boltzmann), "
            "surface_curvature (CA neighbour-density convexity). hydrophobic_patches = "
            "REAL (SASA-exposed + contact clustering).",
            "A protein WITHOUT a fetched PDB (or a failed fetch/parse) keeps the "
            "sequence/APR PROXY features, FLAGGED (has_real_3d_features=False) — the "
            "fetch NEVER raises, it falls back.",
            "NATIVE-vs-FIBRIL: a native-monomer structure != the aggregation-competent/"
            "fibril state. Each real structure carries a per-structure native_or_fibril "
            "stamp; MOST fetched PDBs here are NATIVE monomers (e.g. 1IYT Abeta42 "
            "monomer, 1XQ8 micelle-bound aSyn), NOT the fibril — so structure->kinetics "
            "is ASSOCIATIVE and often confounded.",
            "ASSOCIATION-ONLY, NEVER causal: causal:false is an invariant on every "
            "association; experimentally_validated is an external human flag only.",
            "Effective n = PROTEINS (<=24); the REAL-3D subset is even smaller (only "
            "proteins with a fetched PDB), so those tests are MORE underpowered. Most "
            "verdicts are insufficient_power/null — reported honestly.",
            "The GNN graph is a forward-compatible REPRESENTATION; NO GNN is built, but "
            "the REAL structural contact edges are now populated where a PDB was parsed.",
        ],
    }


# ============================================================================ #
# I/O + CLI
# ============================================================================ #
def _load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _print_summary(payload: dict):
    _safe_print("\n=== M16 structure bridge (version %s) ===" % payload.get("version"))
    if payload.get("status") != "ok":
        _safe_print("status: %s (%s)" % (payload.get("status"), payload.get("error")))
        return
    cov = payload.get("coverage", {})
    _safe_print(cov.get("banner", ""))
    _safe_print("proteins with APR-proxy features (kinetics corpus): %d"
                % cov.get("n_proteins_with_apr_proxy_in_kinetics", 0))
    _safe_print("PDBs fetched OK: %d  failed: %d"
                % (cov.get("n_pdb_fetch_ok", 0), cov.get("n_pdb_fetch_failed", 0)))
    _safe_print("proteins with REAL 3D features (kinetics corpus): %d "
                "(native=%d, fibril=%d)"
                % (cov.get("n_proteins_with_real_3d_in_kinetics", 0),
                   cov.get("real_3d_native_count", 0), cov.get("real_3d_fibril_count", 0)))
    used = cov.get("real_3d_structures_used", [])
    if used:
        _safe_print("real 3D structures used: %s"
                    % ", ".join("%s->%s(%s)" % (u["uniprot"], u["pdb_id"],
                                                u["native_or_fibril"]) for u in used))
    # per-feature fidelity (EXACT vs APPROXIMATION) — the honesty centrepiece
    _safe_print("\n--- per-feature fidelity (EXACT vs APPROXIMATION) ---")
    for k, v in {
        "contact_map": "EXACT (Cbeta distance threshold on real coords)",
        "real_sasa": "EXACT ALGORITHM (Shrake-Rupley rolling-ball)",
        "hydrophobic_surface_patches": "REAL (SASA-exposed + contact clustering)",
        "secondary_structure": "APPROXIMATION (backbone phi/psi geometry, NOT DSSP H-bonds)",
        "electrostatic_surface": "APPROXIMATION (Coulomb/Debye-Huckel, NOT APBS PB)",
        "surface_curvature": "APPROXIMATION (CA neighbour-density convexity)",
    }.items():
        _safe_print("  %-28s %s" % (k, v))
    vc = payload.get("variance_ceiling_log10_t50", {})
    if vc.get("computable"):
        _safe_print("\nvariance ceiling (between-protein log10 t50): %.1f%% (%d proteins)"
                    % (100 * vc["between_protein_fraction"], vc["n_proteins"]))
    # real-3D feature examples: 1IYT (Abeta42) + 1XQ8 (alpha-synuclein)
    sf = payload.get("structural_features_by_uniprot", {})
    for up, tag in (("P05067", "1IYT Abeta42"), ("P37840", "1XQ8 alpha-synuclein")):
        e = sf.get(up)
        if not e or not e.get("real_3d_features"):
            continue
        rf = e["real_3d_features"]
        s = rf.get("summary", {})
        cm = rf.get("real_sasa", {})
        _safe_print("\n--- REAL 3D example: %s [%s, %s] ---"
                    % (tag, rf.get("pdb_id"), rf.get("native_or_fibril")))
        _safe_print("  n_residues=%s  n_chains=%s  ensemble=%s(%s models)"
                    % (rf.get("n_residues"), rf.get("n_chains"),
                       rf.get("is_nmr_ensemble"), rf.get("n_models_in_ensemble")))
        _safe_print("  mean_rel_SASA=%s (%%exposed=%s)  beta_sheet_content=%s%%  helix=%s%%"
                    % (s.get("mean_rel_sasa"), cm.get("frac_exposed"),
                       s.get("beta_sheet_content"), s.get("helix_content")))
        _safe_print("  n_contacts=%s  rel_contact_order=%s  n_hydrophobic_patches=%s  "
                    "net_surface_charge=%s  mean_convexity=%s"
                    % (s.get("n_contacts"), s.get("relative_contact_order"),
                       s.get("n_hydrophobic_patches"), s.get("net_surface_charge"),
                       s.get("mean_convexity")))
    assoc = payload.get("associations", {})
    _safe_print("\n--- STRUCTURE -> KINETICS associations (n cards=%d; %d real-3D, %d proxy) ---"
                % (assoc.get("n_cards", 0), assoc.get("n_real_3d_cards", 0),
                   assoc.get("n_proxy_cards", 0)))
    _safe_print("real-3D cards testable (n>=4): %d  (max n proteins=%d)"
                % (assoc.get("n_real_3d_cards_testable", 0),
                   assoc.get("max_n_proteins_real_3d", 0)))
    _safe_print("verdict counts: %s" % assoc.get("verdict_counts"))
    _safe_print("significant after FDR (q<%.2f, with LOPO skill): %d"
                % (assoc.get("fdr_alpha", 0.05), assoc.get("n_significant_after_fdr", 0)))
    _safe_print("causal invariant: %s" % assoc.get("causal_invariant"))
    # show the strongest few REAL-3D cards, then the strongest proxy cards
    real_cards = sorted([c for c in assoc.get("association_cards", [])
                         if c["feature_source"] == "real3d" and c["spearman_rho"] is not None],
                        key=lambda c: abs(c["spearman_rho"]), reverse=True)
    _safe_print("strongest REAL-3D association cards:")
    for c in real_cards[:6]:
        _safe_print("  [%s -> %s] n=%d rho=%s perm_p=%s q=%s LOPO=%s -> %s"
                    % (c["feature"], c["target"], c["n_proteins"], c["spearman_rho"],
                       c["permutation_p"], c["fdr_q"], c["lopo_skill_over_baseline"],
                       c["verdict"]))


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    proc = root / "data" / "processed"
    ap = argparse.ArgumentParser(description="PRISE M16 structure bridge")
    ap.add_argument("--triaged", type=Path, default=proc / "curves_triaged.jsonl")
    ap.add_argument("--features", type=Path, default=proc / "features.jsonl")
    ap.add_argument("--regimes", type=Path, default=proc / "regimes.jsonl")
    ap.add_argument("--sequence", type=Path, default=proc / "sequence_structure.json")
    ap.add_argument("--output", type=Path, default=proc / "structure_features.json")
    ap.add_argument("--assoc-output", type=Path,
                    default=proc / "structure_kinetics_associations.json")
    ap.add_argument("--per-protein-dir", type=Path,
                    default=proc / "structure_features_by_protein")
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip PDB fetch + real-3D features (proxy-only, offline)")
    args = ap.parse_args(argv)

    triaged = _load_jsonl(args.triaged)
    features = _load_jsonl(args.features)
    regimes = _load_jsonl(args.regimes)
    seq = _load_json(args.sequence, {"by_uniprot": {}, "absent_predictors": {}})

    payload = build_structure_bridge(triaged, features, regimes, seq,
                                     fetch=not args.no_fetch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[M16] wrote %s" % args.output)

    # split the associations into their own artifact (per §3-M16 step 6)
    assoc_payload = {
        "version": VERSION, "status": payload.get("status"),
        "coverage": payload.get("coverage", {}),
        "variance_ceiling_log10_t50": payload.get("variance_ceiling_log10_t50", {}),
        "associations": payload.get("associations", {}),
        "honesty_constraints": payload.get("honesty_constraints", []),
    }
    args.assoc_output.write_text(json.dumps(assoc_payload, indent=2), encoding="utf-8")
    print("[M16] wrote %s" % args.assoc_output)

    # per-protein feature files (small; deterministic)
    if payload.get("status") == "ok":
        args.per_protein_dir.mkdir(parents=True, exist_ok=True)
        for up, rec in payload.get("structural_features_by_uniprot", {}).items():
            (args.per_protein_dir / f"{up}.json").write_text(
                json.dumps(rec, indent=2), encoding="utf-8")

    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
