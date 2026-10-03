"""
Tests for PRISE Module M14 — Cross-Study Hierarchical Meta-Analysis
(engine/m14_meta.py). Fast, fully deterministic (synthetic data with a seeded
LCG; closed-form inference). Every assertion is a property of the random-effects
meta-analysis math or the §3-M14 honesty contract.

Proves:
  * a PLANTED between-study τ²_between is recovered within tolerance;
  * Cochran's Q and Higgins I² match their closed forms;
  * leave-one-study-out FLAGS an injected outlier study (large influence);
  * a construct (WT vs mutant) moderator offset is recovered;
  * a single-study protein -> status "single_study", NO pooling, no crash;
  * Egger/funnel DETECTS an injected small-study bias (effect ∝ s.e.);
  * SCALE: a 2000-study protein fits in < 1s;
  * NEVER raises on empty / degenerate (single concentration -> stratum fallback,
    flagged);
  * determinism (byte-identical repeated fits).

    python engine/test_m14_meta.py
    pytest engine/test_m14_meta.py
"""
from __future__ import annotations

import json
import math
import time

import m14_meta as M


# --------------------------- helpers --------------------------------------- #
def _curve(series_id, t50, study, author="LabA", log10_conc=1.0, pH=7.4,
           temp=37.0, is_mutant=False, s2_log=0.01):
    """One poolable curve in fit_protein_meta's expected shape. s2_log is the
    per-curve sampling variance on the ln-t50 scale."""
    t50 = float(t50)
    return {
        "series_id": series_id,
        "t50": t50,
        "s2_t50": s2_log * t50 * t50,      # inverse delta -> t50-scale var
        "s2_log_default": s2_log,
        "study": study,
        "author": author,
        "year": "2020",
        "log10_conc": log10_conc,
        "pH": pH, "temp": temp, "is_mutant": is_mutant,
    }


def _lcg(seed):
    state = seed & 0x7FFFFFFF

    def _rand():
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    def _gauss():
        u1 = max(_rand(), 1e-12)
        u2 = _rand()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return _rand, _gauss


# --------------------------- 1. τ² recovery -------------------------------- #
def test_planted_tau2_recovered():
    """Inject a known between-study τ² across K studies with tiny sampling error;
    the estimator recovers it within tolerance."""
    _rand, _gauss = _lcg(20240601)
    tau2_true, mu_true = 0.30, 2.0
    tau = math.sqrt(tau2_true)
    curves = []
    K = 400
    for s in range(K):
        u = tau * _gauss()
        for c in range(4):
            # very small sampling noise so between-study variance dominates
            theta = mu_true + u + 0.02 * _gauss()
            curves.append(_curve(f"S{s}_{c}", math.exp(theta), f"PMID{s}",
                                 s2_log=0.0004))
    res = M.fit_protein_meta("P", curves)
    assert res["status"] == "meta_analysis"
    tau2 = res["heterogeneity"]["tau2_between"]
    assert abs(tau2 - tau2_true) < 0.06, (tau2, tau2_true)
    # pooled μ recovered
    assert abs(res["pooled_estimate"]["mu_log"] - mu_true) < 0.1


# --------------------------- 2. Q and I² closed forms ---------------------- #
def test_Q_and_I2_match_closed_forms():
    """Q = Σ wᵢ(θᵢ−μ_FE)² and I² = max(0,(Q−(k−1))/Q) at the STUDY level."""
    # one curve per study so study effect == curve effect, s2 known
    effs = [1.0, 1.4, 0.6, 2.2, 1.1, 0.9, 1.7]
    s2 = 0.02
    curves = [_curve(f"S{i}", math.exp(e), f"PMID{i}", s2_log=s2)
              for i, e in enumerate(effs)]
    res = M.fit_protein_meta("P", curves)
    het = res["heterogeneity"]
    # recompute Q by hand (single-curve studies: study s2 == curve s2)
    w = [1.0 / s2] * len(effs)
    mu_fe = sum(wi * e for wi, e in zip(w, effs)) / sum(w)
    Q_hand = sum(wi * (e - mu_fe) ** 2 for wi, e in zip(w, effs))
    assert abs(het["Q"] - Q_hand) < 1e-6, (het["Q"], Q_hand)
    k = len(effs)
    I2_hand = max(0.0, (Q_hand - (k - 1)) / Q_hand)
    assert abs(het["I2"] - I2_hand) < 1e-9
    assert het["df"] == k - 1


# --------------------------- 3. LOSO flags an outlier ---------------------- #
def test_loso_flags_injected_outlier():
    """A tight cluster of studies + one wildly-off study: LOSO marks the off study
    as the most influential and flags it an outlier."""
    curves = []
    for i in range(8):
        curves.append(_curve(f"S{i}", math.exp(1.0 + 0.01 * i), f"PMID{i}",
                             s2_log=0.001))
    # the outlier: effect far from the cluster, small SE (so it is influential)
    curves.append(_curve("SOUT", math.exp(4.0), "PMID_OUT", s2_log=0.001))
    res = M.fit_protein_meta("P", curves)
    loso = res["leave_one_study_out"]
    out_row = next(r for r in loso if r["pmid"] == "PMID_OUT")
    assert out_row["is_outlier"] is True, out_row
    # it must be THE most-influential (largest |Δμ|)
    top = next(r for r in loso if r.get("most_influential"))
    assert top["pmid"] == "PMID_OUT", top
    # none of the tight-cluster studies should be flagged an outlier
    assert all(not r["is_outlier"] for r in loso if r["pmid"] != "PMID_OUT")


# --------------------------- 4. construct moderator recovered -------------- #
def test_construct_moderator_recovered():
    """Mutant curves carry a +0.8 log-t50 offset; the construct fixed effect
    recovers it (and the WT/mutant fold-change)."""
    _rand, _gauss = _lcg(777)
    offset = 0.8
    curves = []
    for s in range(30):
        base = 1.5 + 0.15 * _gauss()          # study-level effect
        for c in range(3):
            wt = _curve(f"S{s}wt{c}", math.exp(base + 0.02 * _gauss()),
                        f"PMID{s}", is_mutant=False, s2_log=0.01)
            mut = _curve(f"S{s}mu{c}", math.exp(base + offset + 0.02 * _gauss()),
                         f"PMID{s}", is_mutant=True, s2_log=0.01)
            curves += [wt, mut]
    res = M.fit_protein_meta("P", curves)
    ce = res["moderators"]["construct_effect"]
    assert ce.get("estimable", True) is not False
    assert abs(ce["estimate_log"] - offset) < 0.1, ce
    # fold change ~ exp(0.8)
    assert abs(ce["fold_change_t50_mutant_vs_wt"] - math.exp(offset)) < 0.3


# --------------------------- 5. single-study honesty ----------------------- #
def test_single_study_no_pooling_no_crash():
    """A protein with curves from ONE study -> status single_study, no pooling."""
    curves = [_curve(f"S{i}", 10.0 + i, "PMID_ONLY") for i in range(5)]
    res = M.fit_protein_meta("P", curves)
    assert res["status"] == "single_study"
    assert res["n_studies"] == 1
    assert "pooled_estimate" not in res
    assert "single study" in res["note"].lower()


def test_empty_never_raises():
    """No curves -> no crash, honest no_data status."""
    res = M.fit_protein_meta("P", [])
    assert res["status"] == "no_data"
    assert res["n_studies"] == 0


# --------------------------- 6. Egger small-study bias --------------------- #
def test_egger_detects_small_study_bias():
    """Inject small-study bias: studies with LARGER s.e. carry LARGER effects
    (funnel asymmetry). Egger's intercept test must flag it."""
    _rand, _gauss = _lcg(4242)
    curves = []
    # 14 studies; se grows across studies and effect grows WITH se (bias)
    for i in range(14):
        se = 0.05 + 0.05 * i               # increasing sampling sd
        s2_log = se * se
        effect = 1.0 + 3.0 * se            # effect correlated with se -> asymmetry
        # single curve per study so study se == curve se
        curves.append(_curve(f"S{i}", math.exp(effect + 0.001 * _gauss()),
                             f"PMID{i}", s2_log=s2_log))
    res = M.fit_protein_meta("P", curves)
    egg = res["publication_bias"]["egger"]
    assert egg["testable"] is True
    assert egg["small_study_effect_flag"] is True, egg
    assert egg["p"] < 0.05


def test_egger_null_when_no_bias():
    """Symmetric funnel (effect independent of se) -> Egger does NOT flag."""
    _rand, _gauss = _lcg(9)
    curves = []
    for i in range(14):
        se = 0.05 + 0.05 * i
        s2_log = se * se
        effect = 1.5 + 0.02 * _gauss()     # effect NOT correlated with se
        curves.append(_curve(f"S{i}", math.exp(effect), f"PMID{i}", s2_log=s2_log))
    res = M.fit_protein_meta("P", curves)
    egg = res["publication_bias"]["egger"]
    assert egg["small_study_effect_flag"] is False, egg


# --------------------------- 7. SCALE: 2000 studies < 1s ------------------- #
def test_scale_2000_studies_under_1s():
    curves = M.synthetic_protein(n_studies=2000, tau2_true=0.25)
    t0 = time.perf_counter()
    res = M.fit_protein_meta("SYN", curves)
    dt = time.perf_counter() - t0
    assert res["status"] == "meta_analysis"
    assert res["n_studies"] == 2000
    assert dt < 1.0, f"2000-study fit took {dt:.3f}s (>=1s)"
    # τ² still recovered at scale
    assert abs(res["heterogeneity"]["tau2_between"] - 0.25) < 0.05


# --------------------------- 8. degenerate concentration -> fallback ------- #
def test_single_concentration_falls_back_to_stratum():
    """All curves at one concentration -> the concentration moderator is degenerate;
    the fit falls back to matched-condition pooling and SAYS SO (flag), no crash."""
    _rand, _gauss = _lcg(11)
    curves = []
    for s in range(6):
        base = 1.5 + 0.1 * _gauss()
        for c in range(3):
            curves.append(_curve(f"S{s}_{c}", math.exp(base + 0.02 * _gauss()),
                                 f"PMID{s}", log10_conc=1.0))  # SAME conc
    res = M.fit_protein_meta("P", curves)
    assert res["status"] == "meta_analysis"
    assert res["fallback_condition_stratum_pooling"] is True
    assert any("DEGENERATE" in n or "degenerate" in n
               for n in res["moderator_notes"])
    assert res["pooled_estimate"]["condition_adjusted"] is False


# --------------------------- 9. laboratory confounding honesty ------------- #
def test_laboratory_confounded_when_one_study_per_author():
    """Each study from a distinct author -> lab confounded with study (flagged)."""
    curves = [_curve(f"S{i}", math.exp(1.5), f"PMID{i}", author=f"Lab{i}")
              for i in range(5)]
    res = M.fit_protein_meta("P", curves)
    lab = res["laboratory"]
    assert lab["status"] == "confounded_with_study"
    assert lab["estimable"] is False


def test_laboratory_estimable_when_author_spans_studies():
    """An author group spanning >=2 studies -> lab effect estimable."""
    curves = []
    # LabX has 2 studies, others 1 each
    curves.append(_curve("S0", math.exp(1.4), "PMID0", author="LabX"))
    curves.append(_curve("S1", math.exp(1.6), "PMID1", author="LabX"))
    curves.append(_curve("S2", math.exp(1.5), "PMID2", author="LabY"))
    curves.append(_curve("S3", math.exp(1.7), "PMID3", author="LabZ"))
    res = M.fit_protein_meta("P", curves)
    lab = res["laboratory"]
    assert lab["status"] == "estimable"
    assert lab["estimable"] is True


# --------------------------- 10. batch always unidentifiable --------------- #
def test_batch_always_unidentifiable():
    curves = [_curve(f"S{i}", math.exp(1.5), f"PMID{i}") for i in range(5)]
    res = M.fit_protein_meta("P", curves)
    vd = res["variance_decomposition"]
    assert vd["batch"] == "unidentifiable_not_recorded"


# --------------------------- 11. prediction interval widens on τ² ---------- #
def test_prediction_interval_wider_than_ci():
    """The new-study prediction interval must be wider than the μ credible interval
    (it adds τ² to SE_μ²)."""
    _rand, _gauss = _lcg(5)
    curves = []
    for s in range(20):
        u = 0.4 * _gauss()
        for c in range(3):
            curves.append(_curve(f"S{s}_{c}", math.exp(1.5 + u + 0.02 * _gauss()),
                                 f"PMID{s}", s2_log=0.01))
    res = M.fit_protein_meta("P", curves)
    ci = res["pooled_estimate"]["ci95_log"]
    pi = res["prediction_interval_new_study"]["pi95_log"]
    assert (pi[1] - pi[0]) > (ci[1] - ci[0]), (pi, ci)


# --------------------------- 12. determinism ------------------------------- #
def test_determinism_byte_identical():
    curves = M.synthetic_protein(n_studies=50, tau2_true=0.2)
    a = json.dumps(M.fit_protein_meta("P", curves), sort_keys=True)
    b = json.dumps(M.fit_protein_meta("P", curves), sort_keys=True)
    assert a == b


# --------------------------- 13. build_meta_analysis integration ----------- #
def test_build_meta_analysis_join_and_counts():
    """End-to-end assembly from analysis + triaged shapes: join on series_id,
    multi-study protein -> meta, single-study -> single_study."""
    analysis = []
    triaged = []

    def _add(sid, protein, t50, pmid, author, conc=10.0):
        analysis.append({
            "series_id": sid, "protein_id": protein,
            "curve_features": {"status": "ok", "t50_status": "point",
                               "features": {"t50": t50}},
            "model_selection": {"t50_predictive_interval":
                                [t50 * 0.8, t50, t50 * 1.25]},
            "condition_vector": {"assay_type": "ThT", "pH": 7.4,
                                 "temperature_C": 37.0,
                                 "construct_id": "Wild Type",
                                 "concentration": {"value_uM": conc}},
        })
        triaged.append({
            "series_id": sid,
            "source_study": {"pmid": pmid, "author": author, "year": "2020"},
            "condition_vector": {"concentration": {"value_uM": conc}},
        })

    # multi-study protein A (3 studies, varied concentration)
    for i, (pmid, conc) in enumerate([("A1", 5), ("A2", 20), ("A3", 50)]):
        for c in range(3):
            _add(f"A{i}_{c}", "ProtA", 10.0 + i * 5 + c, pmid, f"Lab{i}", conc)
    # single-study protein B
    for c in range(3):
        _add(f"B_{c}", "ProtB", 30.0 + c, "B1", "LabB")

    payload = M.build_meta_analysis(analysis, triaged, gamma_records=[])
    assert payload["status"] == "ok"
    by = {r["protein"]: r for r in payload["proteins"]}
    assert by["ProtA"]["status"] == "meta_analysis"
    assert by["ProtA"]["n_studies"] == 3
    assert by["ProtB"]["status"] == "single_study"
    assert payload["summary"]["n_meta_analysis"] == 1
    assert payload["summary"]["n_single_study"] == 1


# --------------------------- 14. never raises on junk ---------------------- #
def test_never_raises_on_garbage():
    junk = [
        {"series_id": "x", "t50": None, "s2_t50": None, "s2_log_default": None,
         "study": None, "author": None, "log10_conc": None, "pH": None,
         "temp": None, "is_mutant": False},
        {"series_id": "y", "t50": float("nan"), "s2_t50": -1, "s2_log_default": 0.01,
         "study": "P1", "author": "L", "log10_conc": float("inf"), "pH": 7,
         "temp": 37, "is_mutant": False},
    ]
    res = M.fit_protein_meta("P", junk)   # must not raise
    assert res["status"] in ("single_study", "no_data", "meta_analysis", "error")


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
