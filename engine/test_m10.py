"""
Tests for M10 — cross-modal Sequence × Kinetics × Structure analysis.

Deterministic, fast, on SYNTHETIC small inputs (no real artifacts needed). They
assert the statistics are CORRECT on planted signals and ~null on noise, that the
PROTEIN is the unit (replicate curves of one protein cannot inflate n), and that
the module degrades gracefully on missing/empty inputs.

    python engine/test_m10.py
    pytest engine/test_m10.py
"""
from __future__ import annotations

import math

import numpy as np

from m10_crossmodal import (
    DISCORDANCE_ALPHA,
    binom_test_two_sided,
    build_crossmodal,
    build_protein_table,
    discordance_direction,
    jonckheere_permutation_p,
    jonckheere_statistic,
    kruskal_wallis,
    lopo_r2,
    p2_apr_regime,
    p3_discordance,
    variance_ceiling,
    _discordance_clause,
    _perm_pvalue_corr,
    _spearman,
)


# --------------------------- tiny synthetic-artifact builders -------------- #
def _triaged(series_id, uniprot, name, assay="ThT", assay_mass=True, censoring="none"):
    return {"series_id": series_id, "uniprot_id": uniprot, "protein_id": name,
            "censoring_class": censoring,
            "condition_vector": {"assay_type": assay, "assay_reports_mass": assay_mass}}


def _feat(series_id, t50=None, t50_status="point", lag_ratio=None, censoring="none",
          status="ok"):
    return {"series_id": series_id, "status": status, "t50_status": t50_status,
            "censoring_class": censoring,
            "features": {"t50": t50, "lag_to_t50_ratio": lag_ratio}}


def _regime(series_id, regime):
    return {"series_id": series_id, "descriptive_regime": {"regime": regime}}


def _seq(uniprot, name, pasta=None, waltz=0, n_aprs=0, n_struct=0, n_amyloid=0):
    aprs = [{"position": f"{i}-{i+5}", "category": "Pathogenic"} for i in range(n_aprs)]
    structs = ([{"pdb_id": f"A{i}", "amyloid": "Amyloid"} for i in range(n_amyloid)]
               + [{"pdb_id": f"N{i}", "amyloid": "Non-amyloid"}
                  for i in range(n_struct - n_amyloid)])
    return {
        "protein_name": name,
        "sequence_predictors": {
            "pasta": {"value": pasta, "available": pasta is not None},
            "waltz": {"value": None, "available": waltz > 0,
                      "n_waltz_amyloid_peptides": waltz},
        },
        "aprs": aprs, "structures": structs,
    }


# =============================== correlation =============================== #
def test_spearman_planted_perfect_monotone():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [2, 4, 6, 8, 10, 12, 14, 16]          # perfectly monotone in x
    assert abs(_spearman(x, y) - 1.0) < 1e-9


def test_perm_pvalue_significant_on_planted_signal():
    # strong linear signal across 12 PROTEINS -> small permutation p
    x = list(range(12))
    y = [2.0 * v + 0.05 * ((-1) ** v) for v in x]
    obs, p, n = _perm_pvalue_corr(x, y, kind="spearman")
    assert obs is not None and obs > 0.9
    assert p is not None and p < 0.01
    assert n == 10000


def test_perm_pvalue_null_on_noise():
    # deterministic pseudo-random noise (seeded) -> p should be large (no signal)
    rng = np.random.default_rng(0)
    x = list(range(20))
    y = list(rng.permutation(20))            # independent of x
    obs, p, n = _perm_pvalue_corr(x, y, kind="spearman")
    assert p is not None and p > 0.2         # not significant


def test_perm_pvalue_is_deterministic():
    x = list(range(15))
    y = [v * 1.3 + 0.2 * (v % 3) for v in x]
    a = _perm_pvalue_corr(x, y)
    b = _perm_pvalue_corr(x, y)
    assert a == b                             # seeded -> byte-identical


# ============================ variance ceiling ============================ #
def test_variance_ceiling_all_between():
    # each protein has identical replicates -> ALL variance is between-protein
    reps = {"P1": [0.0, 0.0, 0.0], "P2": [1.0, 1.0, 1.0], "P3": [2.0, 2.0]}
    vc = variance_ceiling(reps)
    assert vc["computable"]
    assert abs(vc["between_protein_fraction"] - 1.0) < 1e-9
    assert abs(vc["within_protein_fraction"] - 0.0) < 1e-9


def test_variance_ceiling_all_within():
    # all proteins share the SAME mean but vary internally -> ~0 between-protein
    reps = {"P1": [-1.0, 1.0], "P2": [-1.0, 1.0], "P3": [-1.0, 1.0]}
    vc = variance_ceiling(reps)
    assert vc["computable"]
    assert vc["between_protein_fraction"] < 1e-9


def test_variance_ceiling_known_value():
    # 2 proteins, means 0 and 2, each with +/-1 spread:
    # SS_within = 4*1 = 4 ; group means 0,2 grand 1 -> SS_between = 2*1+2*1 = 4
    # fraction = 4/8 = 0.5
    reps = {"A": [-1.0, 1.0], "B": [1.0, 3.0]}
    vc = variance_ceiling(reps)
    assert abs(vc["between_protein_fraction"] - 0.5) < 1e-9


def test_variance_ceiling_graceful_when_thin():
    assert variance_ceiling({})["computable"] is False
    assert variance_ceiling({"P1": [1.0]})["computable"] is False


# ============================== LOPO skill ================================ #
def test_lopo_r2_high_on_clean_linear():
    x = [float(v) for v in range(12)]
    y = [3.0 * v + 1.0 for v in x]            # exact line -> LOPO R^2 ~ 1
    r = lopo_r2(x, y)
    assert r["computable"] and r["lopo_r2"] > 0.95


def test_lopo_r2_nonpositive_on_noise():
    rng = np.random.default_rng(3)
    x = [float(v) for v in range(16)]
    y = list(rng.normal(0, 1, 16))           # no relation -> LOPO R^2 <= ~0
    r = lopo_r2(x, y)
    assert r["computable"] and r["lopo_r2"] <= 0.1


# ============================ Jonckheere trend ============================ #
def test_jonckheere_planted_increasing_trend():
    # three ORDERED groups with a clear increasing shift
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    jt = jonckheere_statistic(g)
    assert jt["computable"]
    # every cross-group pair is increasing -> J == max == sum_{i<j} n_i*n_j = 27
    assert jt["J"] == 27.0
    assert jt["direction"] == "increasing"
    assert jt["z"] is not None and jt["z"] > 0
    assert jt["p_one_sided_normal"] < 0.05


def test_jonckheere_no_trend_is_midrange():
    # identical distributions in each group -> J near its null mean, z ~ 0
    g = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    jt = jonckheere_statistic(g)
    assert abs(jt["J"] - jt["mean_J"]) < 1e-9
    assert abs(jt["z"]) < 1e-6


def test_jonckheere_decreasing_detected():
    g = [[7, 8, 9], [4, 5, 6], [1, 2, 3]]
    jt = jonckheere_statistic(g)
    assert jt["J"] == 0.0                      # no increasing pairs
    assert jt["direction"] == "decreasing"


def test_jonckheere_permutation_protein_level():
    # per-protein ordinal labels + values with a planted monotone trend
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    res = jonckheere_permutation_p(labels, values, n_perm=2000)
    assert res["computable"]
    assert res["unit"] == "protein"
    assert res["p_one_sided_permutation"] < 0.05


# ============================= Kruskal–Wallis ============================= #
def test_kruskal_wallis_detects_difference():
    g = [[1, 2, 3], [10, 11, 12], [20, 21, 22]]
    kw = kruskal_wallis(g)
    assert kw["computable"] and kw["p_value"] < 0.05


def test_kruskal_wallis_null():
    g = [[1, 2, 3, 4], [1, 2, 3, 4]]
    kw = kruskal_wallis(g)
    assert kw["computable"] and kw["p_value"] > 0.2


# ============================ binomial (P3) =============================== #
def test_binom_two_sided_known():
    # 10 trials, p=0.5, k=5 -> p-value 1.0 (the most likely outcome)
    assert abs(binom_test_two_sided(5, 10, 0.5) - 1.0) < 1e-9
    # extreme: 0/10 at p=0.5 is rare two-sided
    assert binom_test_two_sided(0, 10, 0.5) < 0.01


def test_binom_observed_below_null_is_significant():
    # observed discordance 1/20 vs a null rate of 0.4 -> significant surprise
    p = binom_test_two_sided(1, 20, 0.4)
    assert p is not None and p < 0.01


# ===================== PROTEIN-as-unit enforcement ======================== #
def _planted_corpus(n_curves_per_protein=50):
    """A corpus where the predictor is PERFECTLY anti/correlated at the protein
    level but each protein contributes MANY identical-predictor curves. The honest
    n is the #proteins; an analysis that counted curves would massively overstate
    significance."""
    triaged, feats, regimes = [], [], []
    # 6 proteins, predictor = pasta energy, target = a per-protein t50 (constant)
    proteins = [("U1", "p1", -2.0, 10.0), ("U2", "p2", -3.0, 20.0),
                ("U3", "p3", -4.0, 30.0), ("U4", "p4", -5.0, 40.0),
                ("U5", "p5", -6.0, 50.0), ("U6", "p6", -7.0, 60.0)]
    seq = {"by_uniprot": {}, "absent_predictors": {}}
    for up, nm, pasta, t50 in proteins:
        seq["by_uniprot"][up] = _seq(up, nm, pasta=pasta, waltz=2, n_aprs=2,
                                     n_struct=1, n_amyloid=1)
        for c in range(n_curves_per_protein):
            sid = f"{up}-{c}"
            # replicate t50 with tiny within-protein jitter (deterministic)
            jit = 0.5 * math.sin(c)
            triaged.append(_triaged(sid, up, nm))
            feats.append(_feat(sid, t50=t50 + jit, t50_status="point",
                               lag_ratio=0.3))
            regimes.append(_regime(sid, "cooperative_sigmoidal"))
    return triaged, feats, regimes, seq


def test_protein_table_collapses_curves_to_proteins():
    triaged, feats, regimes, seq = _planted_corpus(n_curves_per_protein=40)
    table = build_protein_table(triaged, feats, regimes, seq)
    assert len(table) == 6                     # 6 PROTEINS, not 240 curves
    for up, d in table.items():
        assert d["n_curves"] == 40             # curves counted but not the unit
        assert d["pasta_available"] is True


def test_unit_is_protein_in_permutation_n():
    # the permutation null sees n=6 (proteins), NOT 240 (curves)
    triaged, feats, regimes, seq = _planted_corpus(40)
    table = build_protein_table(triaged, feats, regimes, seq)
    xs = [d["pasta_energy"] for d in table.values()]
    ys = [d["median_log10_t50_point"] for d in table.values()]
    assert len(xs) == 6
    _, p, n = _perm_pvalue_corr(xs, ys)
    assert n == 10000
    # with only 6 proteins the smallest achievable two-sided p is bounded well
    # above 0 (1/(6!) territory), so a curve-count of 240 cannot manufacture
    # an impossibly tiny p here.
    assert p >= 1.0 / (math.factorial(6))


# =============================== P2 / P3 end-to-end ======================== #
def test_p2_apr_regime_planted_trend():
    # APR count increases with regime cooperativity across proteins
    triaged, feats, regimes, seq = [], [], [], {"by_uniprot": {}}
    plan = [("g", "no_detectable_aggregation", 0),
            ("g", "no_detectable_aggregation", 0),
            ("h", "gradual_non_cooperative", 2),
            ("h", "gradual_non_cooperative", 3),
            ("t", "threshold_driven", 6),
            ("t", "threshold_driven", 7),
            ("c", "cooperative_sigmoidal", 11),
            ("c", "cooperative_sigmoidal", 12)]
    for i, (pre, regime, naprs) in enumerate(plan):
        up = f"{pre}{i}"
        seq["by_uniprot"][up] = _seq(up, up, pasta=-4.0, n_aprs=naprs)
        sid = f"{up}-0"
        triaged.append(_triaged(sid, up, up))
        feats.append(_feat(sid, t50=10.0))
        regimes.append(_regime(sid, regime))
    table = build_protein_table(triaged, feats, regimes, seq)
    p2 = p2_apr_regime(table)
    assert p2["computable"]
    assert p2["trend_direction"] == "increasing"
    assert p2["p_value_protein_permutation"] < 0.1
    meds = p2["median_apr_by_regime"]
    # monotone non-decreasing medians along the ordered regimes
    seq_meds = [meds[k] for k in
                ["no_detectable_aggregation", "gradual_non_cooperative",
                 "threshold_driven", "cooperative_sigmoidal"] if k in meds]
    assert seq_meds == sorted(seq_meds)


def test_p3_discordance_confusion_and_censoring_exclusion():
    triaged, feats, regimes = [], [], []
    seq = {"by_uniprot": {}, "absent_predictors": {}}
    # A: seq-pos (pasta -6) + kin-pos (a cooperative mass curve)  -> concordant
    # B: seq-pos + kin-NEG but ONLY right-censored -> EXCLUDED (detection limit)
    # C: seq-pos + kin-pos -> concordant
    # D: seq-NEG (no predictor available won't enter; give pasta -1 = above thr,
    #    waltz 0) + kin-pos -> a TRUE discordance (seq says no, kinetics says yes)
    seq["by_uniprot"]["A"] = _seq("A", "A", pasta=-6.0, waltz=3)
    seq["by_uniprot"]["B"] = _seq("B", "B", pasta=-6.0, waltz=3)
    seq["by_uniprot"]["C"] = _seq("C", "C", pasta=-6.0, waltz=3)
    seq["by_uniprot"]["D"] = _seq("D", "D", pasta=-1.0, waltz=0)   # seq-NEGATIVE

    # A: concordant positive
    triaged.append(_triaged("A0", "A", "A"))
    feats.append(_feat("A0", t50=10.0, censoring="none"))
    regimes.append(_regime("A0", "cooperative_sigmoidal"))
    # B: kinetics negative only under right-censoring (no fittable mass curve)
    triaged.append(_triaged("B0", "B", "B", censoring="right"))
    feats.append(_feat("B0", t50=None, t50_status="lower_bound",
                       censoring="right", status="ok"))
    regimes.append(_regime("B0", "no_detectable_aggregation"))
    # C: concordant positive
    triaged.append(_triaged("C0", "C", "C"))
    feats.append(_feat("C0", t50=20.0))
    regimes.append(_regime("C0", "cooperative_sigmoidal"))
    # D: seq-negative but kinetics-positive -> discordant
    triaged.append(_triaged("D0", "D", "D"))
    feats.append(_feat("D0", t50=15.0))
    regimes.append(_regime("D0", "cooperative_sigmoidal"))

    table = build_protein_table(triaged, feats, regimes, seq)
    p3 = p3_discordance(table)
    conf = p3["confusion_2x2"]
    # B excluded by the censoring guard
    assert p3["n_excluded_censored"] == 1
    assert any(e["uniprot"] == "B" for e in p3["excluded_right_censored"])
    # A,C concordant positive; D is seq-neg / kin-pos discordance
    assert conf["seq_pos_kin_pos"] == 2
    assert conf["seq_neg_kin_pos"] == 1
    assert p3["n_discordant"] == 1
    assert 0.0 <= p3["null_disagreement_rate"] <= 1.0
    assert p3["binomial_p"] is not None


# ============ P3 DIRECTION vs SIGNIFICANCE (two SEPARATE axes) ============ #
# Regression guard for a shipped FALSE CLAIM: the verdict once read "Observed
# discordance EXCEEDS the null" for an observed rate 3.4x BELOW the null, because
# the direction word was taken from a (direction-blind) two-sided p-value. The
# direction word must come from the two RATES; the p-value gates only the word
# "significantly". These tests pin all three cases: over / under / not-significant.
def test_direction_comes_from_rates_not_from_the_p_value():
    # SAME p-value, opposite sides of the null -> opposite directions.
    over = discordance_direction(0.60, 0.30, 0.001)
    under = discordance_direction(0.09, 0.30, 0.001)
    assert over["direction"] == "above"
    assert under["direction"] == "below"
    assert over["significant"] is True and under["significant"] is True
    # SAME rates, different p -> SAME direction, different significance.
    assert discordance_direction(0.09, 0.30, 0.90)["direction"] == "below"
    assert discordance_direction(0.09, 0.30, 0.90)["significant"] is False
    # equal rates, and the no-data case
    assert discordance_direction(0.30, 0.30, 1.0)["direction"] == "equal"
    assert discordance_direction(None, 0.30, None)["direction"] is None
    assert discordance_direction(None, 0.30, None)["significant"] is None
    # the reported gap carries the sign
    assert under["observed_minus_null"] < 0 < over["observed_minus_null"]


def test_discordance_clause_all_three_cases():
    # (1) OVER the null + significant -> excess claim, never an agreement claim
    over = _discordance_clause("above", True, 22, 12)
    assert "EXCEEDS" in over
    assert "FALLS BELOW" not in over
    assert "falsifiable signal" in over

    # (2) UNDER the null + significant -> agreement claim, never an excess claim,
    #     and it must carry the conservative-null / easy-subset / thin-n caveat
    #     rather than claiming the predictors are good.
    under = _discordance_clause("below", True, 22, 2)
    assert "FALLS BELOW" in under
    assert "EXCEEDS" not in under
    assert "AGREE more often" in under
    assert "NOT evidence" in under          # no win claimed in the mirror direction
    assert "conservative" in under.lower()
    assert "2/22" in under                   # thin-count caveat is quantified

    # (3) NOT significant, from EITHER side -> no directional claim is made
    for direction in ("above", "below", "equal"):
        ns = _discordance_clause(direction, False, 22, 2)
        assert "NOT significantly" in ns
        assert "EXCEEDS" not in ns and "FALLS BELOW" not in ns
        assert "no falsifiable surprise" in ns

    # degenerate: nothing evaluable -> neither axis claims anything
    none_clause = _discordance_clause(None, None, 0, 0)
    assert "no direction" in none_clause and "no significance" in none_clause


def _discordance_corpus(n_concordant, n_discordant, n_extra_censored=0):
    """Synthetic corpus for the P3 scan.

    Concordant proteins are seq-positive (PASTA -6) AND kinetics-positive; the
    discordant ones are seq-NEGATIVE (PASTA -1, above the pre-registered
    threshold) but kinetics-positive. `n_extra_censored` right-censored curves are
    attached to concordant proteins to lift the corpus censoring rate (and hence
    the pre-registered null) without excluding anyone."""
    triaged, feats, regimes = [], [], []
    seq = {"by_uniprot": {}, "absent_predictors": {}}
    names = []
    for i in range(n_concordant):
        up = f"K{i}"
        names.append(up)
        seq["by_uniprot"][up] = _seq(up, up, pasta=-6.0, waltz=3)
        triaged.append(_triaged(f"{up}-0", up, up))
        feats.append(_feat(f"{up}-0", t50=10.0 + i))
        regimes.append(_regime(f"{up}-0", "cooperative_sigmoidal"))
    for i in range(n_discordant):
        up = f"D{i}"
        seq["by_uniprot"][up] = _seq(up, up, pasta=-1.0, waltz=0)   # seq-NEGATIVE
        triaged.append(_triaged(f"{up}-0", up, up))
        feats.append(_feat(f"{up}-0", t50=12.0 + i))
        regimes.append(_regime(f"{up}-0", "cooperative_sigmoidal"))
    for j in range(n_extra_censored):
        up = names[j % len(names)]                 # stays kinetics-positive
        sid = f"{up}-rc{j}"
        triaged.append(_triaged(sid, up, up, censoring="right"))
        feats.append(_feat(sid, t50=None, t50_status="lower_bound", censoring="right"))
        regimes.append(_regime(sid, "no_detectable_aggregation"))
    return build_protein_table(triaged, feats, regimes, seq)


def test_p3_verdict_under_null_reports_agreement_not_excess():
    """The shipped-bug shape: few discordances, a censoring-inflated null, a
    significant two-sided p. The verdict must say the observed rate falls BELOW
    the null (sequence and kinetics AGREE more than predicted) — never that it
    exceeds it — and must carry the don't-claim-a-win caveat."""
    table = _discordance_corpus(n_concordant=20, n_discordant=2, n_extra_censored=6)
    p3 = p3_discordance(table)
    assert p3["n_evaluated"] == 22 and p3["n_discordant"] == 2
    assert p3["observed_disagreement_rate"] < p3["null_disagreement_rate"]
    assert p3["binomial_p"] < DISCORDANCE_ALPHA          # significant, but BELOW
    assert p3["direction_vs_null"] == "below"
    assert p3["significant_at_0_05"] is True
    v = p3["verdict"]
    assert "EXCEEDS" not in v                            # the false claim, gone
    assert "FALLS BELOW" in v and "AGREE more often" in v
    assert "NOT evidence" in v


def test_p3_verdict_over_null_reports_excess():
    """Mirror case: every evaluable protein discordant -> the observed rate is far
    ABOVE the null and the verdict must say so."""
    table = _discordance_corpus(n_concordant=0, n_discordant=20)
    p3 = p3_discordance(table)
    assert p3["n_evaluated"] == 20 and p3["n_discordant"] == 20
    assert p3["observed_disagreement_rate"] > p3["null_disagreement_rate"]
    assert p3["direction_vs_null"] == "above"
    assert p3["significant_at_0_05"] is True
    v = p3["verdict"]
    assert "EXCEEDS" in v and "FALLS BELOW" not in v


def test_p3_verdict_not_significant_makes_no_directional_claim():
    """Below the null but nowhere near significant (2/22 against a ~0.15 null):
    no direction is claimed in either direction."""
    table = _discordance_corpus(n_concordant=20, n_discordant=2)
    p3 = p3_discordance(table)
    assert p3["direction_vs_null"] == "below"            # direction still recorded
    assert p3["binomial_p"] >= DISCORDANCE_ALPHA
    assert p3["significant_at_0_05"] is False
    v = p3["verdict"]
    assert "NOT significantly" in v
    assert "EXCEEDS" not in v and "FALLS BELOW" not in v


# =============================== assembler / graceful ===================== #
def test_build_crossmodal_graceful_on_empty():
    out = build_crossmodal([], [], [], {"by_uniprot": {}}, [])
    assert out["status"] == "ok"
    assert out["version"] == "m10-crossmodal-1.0"
    for k in ("coverage", "P1_sequence_to_kinetics", "P2_apr_to_regime",
              "P3_discordance", "P4_gamma_sidebar", "honesty_constraints"):
        assert k in out
    assert out["coverage"]["n_proteins_total"] == 0


def test_build_crossmodal_never_raises_on_garbage():
    # malformed records must not raise — the module returns a status block
    out = build_crossmodal([{"bad": 1}], [{"oops": 2}], [None] if False else [{}],
                           None, [{"junk": True}])
    assert out["status"] in ("ok", "error")
    assert out["version"] == "m10-crossmodal-1.0"


def test_assembler_effective_n_banner_says_proteins():
    triaged, feats, regimes, seq = _planted_corpus(30)
    out = build_crossmodal(triaged, feats, regimes, seq, [])
    cov = out["coverage"]
    assert cov["effective_n_is_proteins_not_curves"] is True
    assert cov["n_proteins_total"] == 6
    assert cov["n_curves_total"] == 180          # 6 * 30
    assert "PROTEINS, NOT" in cov["effective_n_banner"]
    # P1 ran on 6 proteins, not 180 curves
    assert out["P1_sequence_to_kinetics"]["n_proteins_pasta"] == 6


# --------------------------------- runner ---------------------------------- #
def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
    return passed


if __name__ == "__main__":
    n = _run_all()
    print(f"test_m10: OK — {n} tests passed")
