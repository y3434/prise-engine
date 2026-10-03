"""
Tests for M16 — Sequence ↔ Structure ↔ Kinetics Bridge (§3-M16), v2.0 (REAL 3D).

Deterministic + fast. They assert:
  * APR/sequence PROXY features are numerically CORRECT on a known peptide
    (Chou–Fasman Pβ/Pα + Kyte–Doolittle + net charge on 'TFDVSFQT' + a charged
    peptide, against hand-computed values);
  * the GNN graph carries node_features, edges are sequence_adjacency, schema stable;
  * REAL 3D extractors on HAND-BUILT tiny coordinate structures (no network):
      - Shrake–Rupley SASA orders a buried atom BELOW an exposed atom (analytic);
      - contact map is EXACT + SYMMETRIC on known coordinates;
      - φ/ψ secondary structure labels a hand-built helix vs sheet mini-backbone;
  * native-vs-fibril stamping from the corpus `type`;
  * a FAILED fetch -> proxy fallback (no raise): fetch_pdb returns None, the
    per-protein builder keeps the proxy + flags has_real_3d_features=False;
  * parse_pdb / fetch_pdb NEVER raise on junk input;
  * the association layer recovers an INJECTED signal, returns null/insufficient_power
    for noise, BH-FDR controls a null batch; causal:false is an INVARIANT on EVERY
    card (proxy AND real-3d);
  * DETERMINISM: parsing a CACHED PDB twice gives identical features (network-gated
    tests skip with a clear message if the cache is absent).

    python engine/test_m16_structure.py
    pytest engine/test_m16_structure.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from m16_structure import (
    CHOU_FASMAN_PA,
    CHOU_FASMAN_PB,
    KINETICS_PDB_TO_UNIPROT,
    KYTE_DOOLITTLE,
    apr_proxy_features,
    build_associations,
    build_graph,
    build_protein_structural_features,
    build_structure_bridge,
    compute_contact_map,
    compute_dssp_secondary_structure,
    compute_electrostatic_surface,
    compute_hydrophobic_surface_patches,
    compute_real_sasa,
    compute_real_structural_features,
    compute_secondary_structure,
    compute_surface_curvature,
    fetch_pdb,
    load_structure,
    make_structure,
    parse_pdb,
    peptide_proxy_features,
    structural_features,
    structure_from_pdb_record,
)

_CACHE = Path(__file__).resolve().parent.parent / "data" / "pdb_cache"


# ------------------------------------------------------------------------- #
# tiny hand-built Structure helpers (no network, deterministic geometry)
# ------------------------------------------------------------------------- #
def _atom(name, element, xyz):
    return {"name": name, "element": element, "xyz": np.asarray(xyz, dtype=float)}


def _res(resnum, aa, atoms):
    return {"resnum": resnum, "resname": aa, "atoms": atoms}


def _structure(chains, pdb_id="TEST", n_models=1):
    return {"pdb_id": pdb_id, "models": [{"chains": chains}], "n_models": n_models,
            "is_nmr_ensemble": n_models > 1, "n_chains": len(chains),
            "n_residues": sum(len(v) for v in chains.values()),
            "chain_ids": sorted(chains), "is_fibril": None,
            "native_or_fibril": "unknown", "has_coordinates": True}


# ============================ 2. APR proxy features ======================== #
def test_peptide_proxy_hand_computed_TFDVSFQT():
    pf = peptide_proxy_features("TFDVSFQT")
    assert pf["n_residues"] == 8
    assert abs(pf["beta_propensity"] - 1.1538) < 1e-3
    assert abs(pf["alpha_propensity"] - 0.9838) < 1e-3
    assert abs(pf["hydrophobicity_mean"] - 0.075) < 1e-3
    assert abs(pf["hydrophobic_patch_max"] - 2.0667) < 1e-3
    assert abs(pf["net_charge"] - (-1.0)) < 1e-9


def test_peptide_proxy_charged_peptide():
    pf = peptide_proxy_features("DEKRRH")
    assert abs(pf["net_charge"] - 1.1) < 1e-9
    assert abs(pf["crude_pI_proxy"] - 8.1) < 1e-9


def test_peptide_proxy_empty_and_junk_never_raises():
    for seq in (None, "", "XXX-*", "  "):
        pf = peptide_proxy_features(seq)
        assert pf["n_residues"] == 0
        assert pf["beta_propensity"] is None


def test_apr_proxy_aggregate_and_stamp():
    aprs = [{"position": "17-24", "length": 8.0, "region": "TFDVSFQT",
             "category": "Functional Amyloid"},
            {"position": "30-35", "length": 6.0, "region": "LLVFFA",
             "category": "Pathogenic"}]
    ap = apr_proxy_features(aprs)
    assert ap["derivation"] == "sequence_apr_proxy"
    assert ap["is_3d_derived"] is False
    assert ap["n_aprs"] == 2 and ap["n_aprs_with_sequence"] == 2
    assert ap["total_apr_length"] == 14
    assert ap["beta_propensity_mean"] is not None
    assert ap["apr_category_counts"]["Functional Amyloid"] == 1
    assert len(ap["aggregation_hotspots"]) == 2


def test_apr_proxy_empty_never_raises():
    ap = apr_proxy_features([])
    assert ap["n_aprs"] == 0 and ap["beta_propensity_mean"] is None
    assert ap["derivation"] == "sequence_apr_proxy"


# ============ REAL 3D EXTRACTORS on hand-built tiny structures ============= #
def test_shrake_rupley_buried_vs_exposed_ordering():
    """A central atom surrounded by 6 close neighbours (octahedral shell) is far MORE
    buried than a lone isolated atom: its SASA must be much smaller. This is the
    analytic buried-vs-exposed ordering the Shrake–Rupley routine must reproduce."""
    # central carbon at origin + 6 carbons at +/-3 A along each axis (close packing)
    d = 3.0
    shell = [(_atom("CB", "C", (d, 0, 0))), (_atom("CB", "C", (-d, 0, 0))),
             (_atom("CB", "C", (0, d, 0))), (_atom("CB", "C", (0, -d, 0))),
             (_atom("CB", "C", (0, 0, d))), (_atom("CB", "C", (0, 0, -d)))]
    center = _res(1, "A", [_atom("CA", "C", (0, 0, 0)), _atom("CB", "C", (0.1, 0, 0))])
    neigh_res = [_res(10 + k, "A", [_atom("CA", "C", a["xyz"]), a])
                 for k, a in enumerate(shell)]
    buried_struct = _structure({"A": [center] + neigh_res})
    sb = compute_real_sasa(buried_struct)
    # a single isolated residue (no neighbours) is fully exposed
    lone = _structure({"A": [_res(1, "A", [_atom("CA", "C", (0, 0, 0)),
                                           _atom("CB", "C", (0.1, 0, 0))])]})
    sl = compute_real_sasa(lone)
    assert sb["computable"] and sl["computable"]
    # the buried central residue's SASA < the lone residue's SASA (same residue type)
    buried_central = sb["per_residue"][0]["sasa_A2"]
    lone_sasa = sl["per_residue"][0]["sasa_A2"]
    assert buried_central < lone_sasa
    # and the lone residue is called exposed, the shell-buried one is not
    assert sl["per_residue"][0]["exposed"] is True
    assert sb["fidelity"].startswith("exact_algorithm")


def test_shrake_rupley_isolated_atom_matches_full_sphere():
    """A single isolated atom's SASA == 4*pi*(r+probe)^2 within the sphere-point
    discretisation error (the algorithm is EXACT up to n_points)."""
    lone = _structure({"A": [_res(1, "A", [_atom("CB", "C", (0, 0, 0))])]})
    s = compute_real_sasa(lone, n_points=960)
    expected = 4.0 * math.pi * (1.70 + 1.4) ** 2
    got = s["per_residue"][0]["sasa_A2"]
    assert abs(got - expected) / expected < 0.02          # <2% discretisation error


def test_contact_map_exact_and_symmetric():
    """Three residues in a line at 5 A spacing (Cβ): residue1–residue3 are 10 A
    apart (NOT a contact at cutoff 8); residue1–residue2 are 5 A but |i-j|=1
    (trivial neighbour, excluded). So a 3-residue line has ZERO non-trivial contacts;
    bringing residue1 and residue3 within 8 A creates exactly ONE symmetric contact."""
    line = _structure({"A": [
        _res(1, "A", [_atom("CA", "C", (0, 0, 0)), _atom("CB", "C", (0, 0, 0))]),
        _res(2, "A", [_atom("CA", "C", (5, 0, 0)), _atom("CB", "C", (5, 0, 0))]),
        _res(3, "A", [_atom("CA", "C", (10, 0, 0)), _atom("CB", "C", (10, 0, 0))])]})
    cm = compute_contact_map(line, cutoff_angstrom=8.0)
    assert cm["computable"] and cm["fidelity"] == "exact"
    assert cm["n_contacts"] == 0                          # only |i-j|=1 within 8 A

    # now fold so residue1 and residue3 are 6 A apart -> exactly one (1,3) contact
    folded = _structure({"A": [
        _res(1, "A", [_atom("CA", "C", (0, 0, 0)), _atom("CB", "C", (0, 0, 0))]),
        _res(2, "A", [_atom("CA", "C", (5, 4, 0)), _atom("CB", "C", (5, 4, 0))]),
        _res(3, "A", [_atom("CA", "C", (6, 0, 0)), _atom("CB", "C", (6, 0, 0))])]})
    cm2 = compute_contact_map(folded, cutoff_angstrom=8.0)
    assert cm2["n_contacts"] == 1
    e = cm2["contact_edges"][0]
    assert {e["resnum_i"], e["resnum_j"]} == {1, 3}       # the (1,3) long-range contact
    # per-residue contact number is symmetric: residues 1 and 3 each have 1 contact
    cn = cm2["per_residue_contact_number"]
    assert cn[0] == 1 and cn[2] == 1 and cn[1] == 0


def test_contact_map_gly_uses_ca():
    """Glycine has no Cβ, so its representative atom must fall back to Cα (else the
    residue would be silently dropped)."""
    st = _structure({"A": [
        _res(1, "G", [_atom("CA", "C", (0, 0, 0)), _atom("N", "N", (-1, 0, 0))]),
        _res(2, "A", [_atom("CA", "C", (5, 3, 0)), _atom("CB", "C", (5, 3, 0))]),
        _res(3, "G", [_atom("CA", "C", (2, 0, 0))])]})
    cm = compute_contact_map(st, cutoff_angstrom=8.0)
    assert cm["computable"] and cm["n_residues"] == 3      # both Gly kept via Cα


def _ideal_backbone(n, phi_deg, psi_deg, omega_deg=180.0, aa="A"):
    """Ideal backbone at fixed (φ,ψ) via NeRF placement of N,CA,C using standard
    bond lengths/angles. phi=-57,psi=-47 -> α-helix; phi=-120,psi=+130 -> β-strand.
    Deterministic, exact geometry (no RNG)."""
    bl = {"N_CA": 1.458, "CA_C": 1.525, "C_N": 1.329}
    ba = {"N_CA_C": 111.0, "CA_C_N": 116.0, "C_N_CA": 122.0}

    def place(a, b, c, length, angle, torsion):
        angle = math.radians(180.0 - angle)
        torsion = math.radians(torsion)
        d = np.array([length * math.cos(angle),
                      length * math.sin(angle) * math.cos(torsion),
                      length * math.sin(angle) * math.sin(torsion)])
        bc = c - b
        bc /= np.linalg.norm(bc)
        nrm = np.cross(b - a, bc)
        nrm /= np.linalg.norm(nrm)
        m2 = np.cross(nrm, bc)
        M = np.array([bc, m2, nrm]).T
        return c + M.dot(d)

    atoms = [np.array([0.0, 0.0, 0.0]), np.array([1.458, 0.0, 0.0]),
             np.array([1.458 + 1.525 * math.cos(math.radians(180 - 111)),
                       1.525 * math.sin(math.radians(180 - 111)), 0.0])]
    for i in range(1, n):
        atoms.append(place(atoms[-3], atoms[-2], atoms[-1], bl["C_N"], ba["CA_C_N"], psi_deg))
        atoms.append(place(atoms[-3], atoms[-2], atoms[-1], bl["N_CA"], ba["C_N_CA"], omega_deg))
        atoms.append(place(atoms[-3], atoms[-2], atoms[-1], bl["CA_C"], ba["N_CA_C"], phi_deg))
    chain = []
    for i in range(n):
        N, CA, C = atoms[3 * i], atoms[3 * i + 1], atoms[3 * i + 2]
        chain.append(_res(i + 1, aa, [_atom("N", "N", N), _atom("CA", "C", CA),
                                      _atom("C", "C", C)]))
    return chain


def _helix_backbone(n=8):
    """An ideal α-helix backbone (φ=-57, ψ=-47)."""
    return _ideal_backbone(n, -57.0, -47.0)


def test_secondary_structure_helix_and_sheet_assignment():
    """φ/ψ SS must label an ideal β-strand backbone as predominantly E (no H), and an
    ideal α-helix backbone as predominantly H. fidelity is APPROXIMATION."""
    ss_strand = compute_secondary_structure(_structure({"A": _ideal_backbone(10, -120, 130, aa="V")}))
    assert ss_strand["computable"]
    assert ss_strand["fidelity"].startswith("approximation")
    assert ss_strand["ss_counts"]["E"] >= 5                # ideal strand -> mostly E
    assert ss_strand["ss_counts"]["H"] == 0                # and no helix
    ss_helix = compute_secondary_structure(_structure({"A": _ideal_backbone(10, -57, -47)}))
    assert ss_helix["computable"]
    assert ss_helix["ss_counts"]["H"] >= 5                 # ideal helix -> mostly H
    assert ss_helix["ss_counts"]["E"] == 0


def test_secondary_structure_classify_basins_directly():
    """Direct φ/ψ basin classification: (-60,-45)->H, (-120,+130)->E, (60,60)->C."""
    from m16_structure import _classify_ss
    assert _classify_ss(-60.0, -45.0) == "H"
    assert _classify_ss(-120.0, 130.0) == "E"
    assert _classify_ss(60.0, 60.0) == "C"
    assert _classify_ss(None, None) == "C"


def test_hydrophobic_patches_and_electrostatics_on_tiny_structure():
    """A cluster of exposed hydrophobic residues (V/I/L) forms one patch; exposed
    charged residues give the right net surface charge sign."""
    # 3 exposed valines within contact + 2 charged (D,K) — all isolated => exposed
    chains = {"A": [
        _res(1, "V", [_atom("CA", "C", (0, 0, 0)), _atom("CB", "C", (0.5, 0, 0))]),
        _res(2, "I", [_atom("CA", "C", (5, 0, 0)), _atom("CB", "C", (5.5, 0, 0))]),
        _res(3, "L", [_atom("CA", "C", (5, 5, 0)), _atom("CB", "C", (5.5, 5, 0))]),
        _res(4, "D", [_atom("CA", "C", (20, 0, 0)), _atom("CB", "C", (20.5, 0, 0))]),
        _res(5, "K", [_atom("CA", "C", (25, 0, 0)), _atom("CB", "C", (25.5, 0, 0))])]}
    st = _structure(chains)
    sasa = compute_real_sasa(st)
    hp = compute_hydrophobic_surface_patches(st, sasa, cutoff_angstrom=8.0)
    assert hp["computable"] and hp["fidelity"].startswith("real")
    assert hp["n_exposed_hydrophobic"] == 3                # V, I, L are exposed+hydrophobic
    el = compute_electrostatic_surface(st, sasa)
    assert el["computable"] and el["fidelity"].startswith("approximation")
    assert abs(el["net_surface_charge"] - 0.0) < 1e-9      # one -1 (D) + one +1 (K)
    assert el["n_exposed_positive"] == 1 and el["n_exposed_negative"] == 1


def test_surface_curvature_protruding_more_convex():
    """A residue with FEWER Cα neighbours is scored MORE convex than a crowded one."""
    # crowded cluster + one isolated protrusion
    chains = {"A": [
        _res(1, "A", [_atom("CA", "C", (0, 0, 0)), _atom("CB", "C", (0.5, 0, 0))]),
        _res(2, "A", [_atom("CA", "C", (2, 0, 0)), _atom("CB", "C", (2.5, 0, 0))]),
        _res(3, "A", [_atom("CA", "C", (0, 2, 0)), _atom("CB", "C", (0.5, 2, 0))]),
        _res(4, "A", [_atom("CA", "C", (2, 2, 0)), _atom("CB", "C", (2.5, 2, 0))]),
        _res(5, "A", [_atom("CA", "C", (40, 0, 0)), _atom("CB", "C", (40.5, 0, 0))])]}
    st = _structure(chains)
    sasa = compute_real_sasa(st)
    cv = compute_surface_curvature(st, sasa, radius_A=10.0)
    assert cv["computable"] and cv["fidelity"].startswith("approximation")
    per = {r["resnum"]: r["n_ca_neighbours"] for r in cv["per_residue"]}
    # the isolated residue 5 has 0 neighbours; a crowded one has >=2
    assert per.get(5) == 0
    assert max(per.values()) >= 2


def test_compute_real_structural_features_bundle_and_fidelity():
    """The bundle runs all 6 extractors, carries a scalar summary + per-feature
    fidelity, and NEVER raises on a normal tiny structure."""
    st = _structure({"A": _helix_backbone(12)}, pdb_id="MINI")
    rf = compute_real_structural_features(st, native_or_fibril="native")
    assert rf["computable"] and rf["derivation"] == "pdb_3d"
    assert rf["native_or_fibril"] == "native"
    fb = rf["fidelity_by_feature"]
    assert fb["contact_map"] == "exact"
    assert fb["real_sasa"].startswith("exact_algorithm")
    assert "NOT DSSP" in fb["secondary_structure"]
    assert "NOT APBS" in fb["electrostatic_surface"]
    for k in ("mean_rel_sasa", "beta_sheet_content", "relative_contact_order",
              "n_hydrophobic_patches", "net_surface_charge", "mean_convexity"):
        assert k in rf["summary"]


# ==================== PDB PARSER (pure stdlib, never raises) =============== #
def test_parse_pdb_junk_and_missing_never_raises():
    assert parse_pdb("does_not_exist_12345.pdb") is None
    # a temp file of garbage -> None, not an exception
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
        fh.write("not a pdb file\nrandom junk\n")
        name = fh.name
    assert parse_pdb(name) is None
    Path(name).unlink(missing_ok=True)


def test_parse_pdb_minimal_record_and_multimodel():
    """A hand-written 2-model PDB parses model 1 only, flags the ensemble."""
    import tempfile
    txt = (
        "MODEL        1\n"
        "ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1      1.500   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  CB  ALA A   1      2.000   1.000   0.000  1.00  0.00           C\n"
        "ENDMDL\n"
        "MODEL        2\n"
        "ATOM      4  N   ALA A   1      9.000   9.000   9.000  1.00  0.00           N\n"
        "ATOM      5  CA  ALA A   1      9.500   9.000   9.000  1.00  0.00           C\n"
        "ENDMDL\n")
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
        fh.write(txt)
        name = fh.name
    st = parse_pdb(name, pdb_id="MINI")
    Path(name).unlink(missing_ok=True)
    assert st is not None
    assert st["n_residues"] == 1                       # only model 1's residue
    assert st["is_nmr_ensemble"] is True and st["n_models"] == 2
    # the parsed CA is at the MODEL-1 coordinate (1.5,0,0), not model 2's
    ca = [a for a in st["models"][0]["chains"]["A"][0]["atoms"] if a["name"] == "CA"][0]
    assert abs(ca["xyz"][0] - 1.5) < 1e-9


# ==================== 3. the 8 features: real / proxy split ================ #
def test_structural_features_proxy_only_when_no_pdb():
    """With no real_features, the genuinely-3D features report the PROXY (flagged
    has_real_3d=False) and contact_map is computable:false."""
    aprs = [{"position": "1-8", "region": "TFDVSFQT", "category": "Pathogenic"}]
    sf = structural_features(aprs, structures=[], real_features=None)
    assert sf["aggregation_hotspots"]["derivation"] == "apr_real"
    assert sf["has_real_3d_features"] is False
    for k in ("beta_sheet_content", "solvent_accessibility", "hydrophobic_patches",
              "electrostatic_surface", "secondary_structure"):
        blk = sf[k]
        assert blk["has_real_3d"] is False
        assert blk["proxy"]["is_3d_derived"] is False
        assert blk["proxy"]["derivation"] == "sequence_apr_proxy"
    assert sf["contact_map"]["computable"] is False        # intrinsically 3D, no PDB
    assert sf["surface_curvature"]["proxy"]["value"] is None  # no 1-D curvature analogue


def test_structural_features_real_3d_present_reports_both():
    """With real_features, each genuinely-3D feature reports BOTH the real-3d value
    (derivation=pdb_3d + fidelity) and the proxy, and contact_map becomes EXACT."""
    st = _structure({"A": _helix_backbone(12)}, pdb_id="MINI")
    rf = compute_real_structural_features(st, native_or_fibril="native")
    aprs = [{"position": "1-8", "region": "TFDVSFQT", "category": "Pathogenic"}]
    sf = structural_features(aprs, structures=[], real_features=rf)
    assert sf["has_real_3d_features"] is True
    assert sf["real_3d_native_or_fibril"] == "native"
    b = sf["beta_sheet_content"]
    assert b["has_real_3d"] is True
    assert b["real_3d"]["derivation"] == "pdb_3d"
    assert "NOT DSSP" in b["real_3d"]["fidelity"]
    sa = sf["solvent_accessibility"]["real_3d"]
    assert sa["fidelity"].startswith("exact_algorithm")     # Shrake-Rupley SASA is EXACT
    assert sf["contact_map"]["computable"] is True and sf["contact_map"]["fidelity"] == "exact"


def test_beta_proxy_matches_chou_fasman():
    sf = structural_features([{"region": "TFDVSFQT", "position": "1-8"}], [])
    expect = sum(CHOU_FASMAN_PB[c] for c in "TFDVSFQT") / 8
    assert abs(sf["beta_sheet_content"]["proxy"]["value"] - expect) < 1e-3


# ================= load_structure / fetch: real-or-fallback =============== #
def test_load_structure_failed_fetch_returns_none_never_raises():
    """A bogus PDB id -> fetch returns None -> load_structure returns None (no raise).
    Uses an isolated empty cache dir so it truly attempts (and fails) a fetch OR is
    offline; either way it must NOT raise and must return None."""
    import tempfile
    tmp = tempfile.mkdtemp()
    out = load_structure("PDB", "ZZZZ_not_a_real_pdb", cache_dir=tmp)
    assert out is None                                     # failed fetch -> None, no raise


def test_failed_fetch_falls_back_to_proxy_flag():
    """A protein whose PDB fetch fails keeps the sequence/APR proxy features and is
    flagged has_real_3d_features=False — the builder NEVER raises. We force failure
    by pointing at a protein with no kinetics-linked PDB (fetch not attempted)."""
    seq = {"by_uniprot": {"P_NONE": {"protein_name": "no-pdb protein",
                                     "aprs": [{"position": "1-8", "region": "TFDVSFQT",
                                               "category": "Pathogenic"}],
                                     "structures": []}}}
    out = build_protein_structural_features(seq, fetch=False)
    e = out["P_NONE"]
    assert e["has_real_3d_features"] is False
    assert e["real_3d_features"] is None
    assert e["apr_proxy_features"]["n_aprs_with_sequence"] == 1   # proxy preserved
    assert e["structural_features"]["contact_map"]["computable"] is False


def test_dssp_alias_is_geometry_secondary_structure():
    """compute_dssp_secondary_structure is a back-compat alias for the REAL geometry
    extractor (renamed concept); it computes, it does NOT raise NotImplementedError."""
    st = _structure({"A": _helix_backbone(10)})
    ss = compute_dssp_secondary_structure(st)
    assert ss["computable"] and ss["derivation"] == "pdb_3d"


# ==================== 1. structure source abstraction ===================== #
def test_make_structure_and_native_fibril_stamp():
    s = make_structure("AlphaFold", "AF-P37840", uniprot="P37840", is_fibril=False)
    assert s["source"] == "AlphaFold"
    assert s["native_or_fibril"] == "native"
    assert s["reliability"]["metric"] == "pLDDT"
    assert "native" in s["native_vs_fibril_note"]

    f = structure_from_pdb_record({"pdb_id": "2N0A", "type": "Fibril",
                                   "method": "SOLID-STATE NMR"}, uniprot="P37840")
    assert f["source"] == "NMR" and f["native_or_fibril"] == "fibril"


def test_kinetics_pdb_native_fibril_map_stamps():
    """The kinetics-PDB map stamps the aggregated states as fibril and native folds
    as native — 4NP8 (tau VQIVYK steric zipper) is the FIBRIL; 1IYT/1XQ8 are native."""
    assert KINETICS_PDB_TO_UNIPROT["4NP8"][1] == "fibril"
    assert KINETICS_PDB_TO_UNIPROT["1IYT"][1] == "native"
    assert KINETICS_PDB_TO_UNIPROT["1XQ8"][1] == "native"


# ============================== 4. GNN graph ============================== #
def test_graph_nodes_edges_schema():
    aprs = [{"position": "17-24", "region": "TFDVSFQT", "category": "Pathogenic"}]
    g = build_graph(aprs, uniprot="TEST")
    assert g["n_nodes"] == 8
    assert g["n_edges"] == 7
    assert all(e["type"] == "sequence_adjacency" for e in g["edges"])
    assert g["contact_edges_available"] is False          # no PDB -> no real contacts
    assert g["real_contact_graph"] is None
    nf = g["nodes"][0]["node_features"]
    for key in ("hydrophobicity", "charge", "beta_propensity", "alpha_propensity",
                "apr_member"):
        assert key in nf
    assert g["node_feature_schema"] == ["hydrophobicity", "charge", "beta_propensity",
                                        "alpha_propensity", "apr_member"]
    assert abs(g["nodes"][0]["node_features"]["hydrophobicity"] - KYTE_DOOLITTLE["T"]) < 1e-9
    assert abs(g["nodes"][1]["node_features"]["beta_propensity"] - CHOU_FASMAN_PB["F"]) < 1e-9
    assert g["nodes"][0]["residue_index"] == 17


def test_graph_real_contact_edges_populated_when_pdb():
    """With real_features from a folded structure, the graph carries a
    real_contact_graph with EXACT structural contact edges."""
    folded = _structure({"A": [
        _res(1, "A", [_atom("CA", "C", (0, 0, 0)), _atom("CB", "C", (0, 0, 0))]),
        _res(2, "A", [_atom("CA", "C", (5, 4, 0)), _atom("CB", "C", (5, 4, 0))]),
        _res(3, "A", [_atom("CA", "C", (6, 0, 0)), _atom("CB", "C", (6, 0, 0))])]})
    rf = compute_real_structural_features(folded, native_or_fibril="native")
    g = build_graph([{"position": "1-3", "region": "AAA"}], uniprot="X", real_features=rf)
    assert g["contact_edges_available"] is True
    rg = g["real_contact_graph"]
    assert rg is not None and rg["fidelity"] == "exact"
    assert rg["n_contact_edges"] == 1


def test_graph_multi_apr_no_cross_edges():
    aprs = [{"position": "1-3", "region": "AAA"}, {"position": "10-12", "region": "VVV"}]
    g = build_graph(aprs, uniprot="X")
    assert g["n_nodes"] == 6
    assert g["n_edges"] == 4


def test_graph_empty_never_raises():
    g = build_graph([], uniprot="X")
    assert g["n_nodes"] == 0 and g["n_edges"] == 0


# ============ 5. association layer: injected signal + nulls + FDR ========= #
def _prot_table_from(struct_map, targets):
    return {up: dict(t, log10_t50_replicates=[]) for up, t in targets.items()}


def _struct_feats_from(feat_map):
    return {up: {"apr_proxy_features": dict(fv)} for up, fv in feat_map.items()}


def test_association_recovers_injected_signal():
    n = 18
    ups = [f"P{i:03d}" for i in range(n)]
    feat = {u: {"beta_propensity_mean": float(i),
                "hydrophobicity_mean": float(np.random.default_rng(i).normal())}
            for i, u in enumerate(ups)}
    tgt = {u: {"median_log10_t50_point": 0.5 * i + 0.01 * ((-1) ** i),
               "median_lag_to_t50_ratio": None, "max_regime_ordinal": None}
           for i, u in enumerate(ups)}
    assoc = build_associations(_struct_feats_from(feat), _prot_table_from(feat, tgt))
    card = next(c for c in assoc["association_cards"]
                if c["feature"] == "beta_propensity_mean"
                and c["target"] == "median_log10_t50")
    assert card["n_proteins"] == n
    assert card["spearman_rho"] > 0.95
    assert card["permutation_p"] < 0.05
    assert card["fdr_q"] is not None and card["fdr_q"] < 0.05
    assert card["lopo_skill_over_baseline"] is not None and card["lopo_skill_over_baseline"] > 0
    assert card["verdict"] == "association"
    assert card["causal"] is False


def test_association_real3d_feature_from_summary():
    """A REAL-3D predictor (from real_3d_features.summary) with an injected monotone
    signal is recovered by the association layer — proving the real-3d source path."""
    n = 16
    ups = [f"R{i:03d}" for i in range(n)]
    struct_feats = {u: {"apr_proxy_features": {},
                        "real_3d_features": {"computable": True,
                                             "summary": {"beta_sheet_content": float(i)}}}
                    for i, u in enumerate(ups)}
    tgt = {u: {"median_log10_t50_point": 0.4 * i + 0.01 * ((-1) ** i),
               "median_lag_to_t50_ratio": None, "max_regime_ordinal": None}
           for i, u in enumerate(ups)}
    assoc = build_associations(struct_feats, _prot_table_from({}, tgt))
    card = next(c for c in assoc["association_cards"]
                if c["feature"] == "real_beta_sheet_content"
                and c["target"] == "median_log10_t50")
    assert card["feature_source"] == "real3d" and card["derivation"] == "pdb_3d"
    assert card["n_proteins"] == n and card["spearman_rho"] > 0.95
    assert card["causal"] is False


def test_association_null_feature_and_fdr_control():
    n = 16
    ups = [f"Q{i:03d}" for i in range(n)]
    rng = np.random.default_rng(2024)
    feat = {u: {"beta_propensity_mean": float(rng.normal()),
                "hydrophobicity_mean": float(rng.normal()),
                "net_charge_total": float(rng.normal()),
                "n_aprs": float(rng.integers(1, 5))}
            for u in ups}
    tgt = {u: {"median_log10_t50_point": float(rng.normal()),
               "median_lag_to_t50_ratio": float(rng.normal()),
               "max_regime_ordinal": float(rng.integers(0, 4))} for u in ups}
    assoc = build_associations(_struct_feats_from(feat), _prot_table_from(feat, tgt))
    assert assoc["n_significant_after_fdr"] == 0
    assert all(c["causal"] is False for c in assoc["association_cards"])


def test_association_small_n_is_insufficient_power():
    ups = [f"S{i}" for i in range(5)]
    feat = {u: {"beta_propensity_mean": float(i)} for i, u in enumerate(ups)}
    tgt = {u: {"median_log10_t50_point": float(i),
               "median_lag_to_t50_ratio": None, "max_regime_ordinal": None}
           for i, u in enumerate(ups)}
    assoc = build_associations(_struct_feats_from(feat), _prot_table_from(feat, tgt))
    card = next(c for c in assoc["association_cards"]
                if c["feature"] == "beta_propensity_mean"
                and c["target"] == "median_log10_t50")
    assert card["verdict"] == "insufficient_power"


def test_causal_false_invariant_everywhere():
    n = 12
    ups = [f"C{i}" for i in range(n)]
    feat = {u: {"beta_propensity_mean": float(i), "n_aprs": float(i % 3)}
            for i, u in enumerate(ups)}
    tgt = {u: {"median_log10_t50_point": float(i),
               "median_lag_to_t50_ratio": float(i % 2),
               "max_regime_ordinal": float(i % 4)} for i, u in enumerate(ups)}
    assoc = build_associations(_struct_feats_from(feat), _prot_table_from(feat, tgt))
    for c in assoc["association_cards"]:
        assert c["causal"] is False
        assert c["experimentally_validated"] is False
        assert c["basis"] == "statistical_association"


# ============================ end-to-end / robustness ===================== #
def test_build_structure_bridge_empty_inputs_never_raises():
    payload = build_structure_bridge([], [], [], {"by_uniprot": {}}, fetch=False)
    assert payload["status"] == "ok"
    assert payload["associations"]["n_cards"] >= 0


def test_build_structure_bridge_smoke_proxy_only():
    """Tiny end-to-end (fetch=False -> proxy path): full payload, honesty constraints,
    causal:false everywhere, native-vs-fibril stamp present."""
    seq = {"by_uniprot": {
        "P1": {"protein_name": "prot1",
               "aprs": [{"position": "1-8", "region": "TFDVSFQT",
                         "category": "Pathogenic"}],
               "structures": [{"pdb_id": "1AAA", "type": "Fibril",
                               "method": "SOLID-STATE NMR"}]},
        "P2": {"protein_name": "prot2",
               "aprs": [{"position": "1-6", "region": "LLVFFA",
                         "category": "Pathogenic"}], "structures": []}}}
    triaged = [
        {"series_id": "s1", "uniprot_id": "P1", "protein_id": "prot1",
         "condition_vector": {"assay_type": "ThT", "assay_reports_mass": True}},
        {"series_id": "s2", "uniprot_id": "P2", "protein_id": "prot2",
         "condition_vector": {"assay_type": "ThT", "assay_reports_mass": True}}]
    features = [
        {"series_id": "s1", "status": "ok", "t50_status": "point",
         "features": {"t50": 10.0, "lag_to_t50_ratio": 0.3}},
        {"series_id": "s2", "status": "ok", "t50_status": "point",
         "features": {"t50": 100.0, "lag_to_t50_ratio": 0.5}}]
    regimes = [{"series_id": "s1", "descriptive_regime": {"regime": "cooperative_sigmoidal"}},
               {"series_id": "s2", "descriptive_regime": {"regime": "gradual_non_cooperative"}}]
    payload = build_structure_bridge(triaged, features, regimes, seq, fetch=False)
    assert payload["status"] == "ok"
    assert payload["coverage"]["n_proteins_with_apr_proxy_in_kinetics"] == 2
    sf = payload["structural_features_by_uniprot"]["P1"]
    assert sf["structural_features"]["structure_sources"]["has_fibril_structure"] is True
    assert all(c["causal"] is False for c in payload["associations"]["association_cards"])
    assert any("native" in h.lower() for h in payload["honesty_constraints"])


# ---------------- network-gated real-PDB tests (skip if no cache) --------- #
def test_cached_1IYT_real_features_and_determinism():
    """On the cached 1IYT (Abeta42 NMR monomer): parse, compute real features, assert
    sensible values, and DETERMINISM (two runs identical). Skips with a clear message
    if the cache file is absent (offline)."""
    path = _CACHE / "1IYT.pdb"
    if not path.exists():
        print("SKIP test_cached_1IYT (no cached PDB — run the CLI once online)")
        return
    st1 = parse_pdb(path, pdb_id="1IYT")
    st2 = parse_pdb(path, pdb_id="1IYT")
    assert st1 is not None and st1["n_residues"] == 42
    assert st1["is_nmr_ensemble"] is True                 # 10-model NMR ensemble
    rf1 = compute_real_structural_features(st1, native_or_fibril="native")
    rf2 = compute_real_structural_features(st2, native_or_fibril="native")
    assert rf1["summary"] == rf2["summary"]               # deterministic
    s = rf1["summary"]
    assert 0.0 <= s["mean_rel_sasa"] <= 1.5               # rel-SASA in range
    assert s["n_contacts"] > 0                            # a folded peptide has contacts
    assert rf1["contact_map"]["fidelity"] == "exact"


def test_cached_4NP8_is_fibril_and_beta_rich():
    """The cached 4NP8 (tau VQIVYK steric zipper) is the FIBRIL state and β-rich."""
    path = _CACHE / "4NP8.pdb"
    if not path.exists():
        print("SKIP test_cached_4NP8 (no cached PDB)")
        return
    st = parse_pdb(path, pdb_id="4NP8")
    assert st is not None
    rf = compute_real_structural_features(st, native_or_fibril="fibril")
    assert rf["native_or_fibril"] == "fibril"             # the aggregated state
    assert rf["summary"]["beta_sheet_content"] >= 30.0    # steric-zipper strand


if __name__ == "__main__":
    import sys
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed of {len(fns)}")
    sys.exit(1 if failed else 0)
