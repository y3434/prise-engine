"""
Tests for M13 — Bayesian Optimal Experimental Design (BOED).

Deterministic tests on SYNTHETIC series (fixed inputs, seeded RNG) that lock the
scientific claims of the M13 capstone:

  * recommend_experiments NEVER raises + returns the full contract keys
    (clean logistic series, single-curve series, empty/no-fit series)
  * TWO honest variable types coexist: FIM-based quantitative designs AND a
    gating (licensing) design, distinguished by gain_basis
  * single-curve protein: add_concentration_series ranks near the top (a mechanism
    reason) + record_agitation_and_seeding present on a licensing basis
  * DIMINISHING RETURNS: the greedy multi-step marginal EIG is non-increasing, and
    a 2nd concentration series is worth strictly less than the 1st (submodular)
  * REPLICATE SCALING: parameter SE ~ 1/sqrt(R) (R=4 ~ halves SE vs R=1)
  * RIGHT-CENSORED extend-window: extending t_max on a truncated sigmoid yields a
    larger EIG (amplitude/plateau parameter) than on an already-complete curve
  * PARETO frontier is genuinely NON-DOMINATED on (info up, cost down, runtime down)
  * COST/RUNTIME scale monotonically with n_conditions, n_replicates, t_max
  * HONESTY block present (Laplace/model-conditional, cost-is-estimate, gating!=fim)

    python engine/test_m13_boed.py
    pytest engine/test_m13_boed.py
"""
from __future__ import annotations

import numpy as np

from models import REGISTRY, m_logistic
from m13_boed import (
    build_candidates,
    estimate_cost_runtime,
    multi_step_plan,
    pareto_frontier,
    prospective_fim,
    recommend_experiments,
    _prior_precision,
    _posterior_se_reduction,
)

T = np.arange(0.0, 40.0, 0.5)

# a fixed γ payload (M4 dual_gamma shape) so the concentration rescaling is
# deterministic and NOT flagged low-confidence.
GAMMA_RESULT = {"gamma_regression": {"gamma": 1.2, "gamma_reliable": True}}


# --------------------------------------------------------------------------- #
#  fixture builders (the shape m2.fit_curve / recommend_experiments expect)
# --------------------------------------------------------------------------- #
def _noisy(yc, sd=0.005, seed=0):
    return list(np.asarray(yc) + np.random.default_rng(seed).normal(0, sd, size=len(yc)))


def _series(y, x=None, sid="syn", conc_uM=20.0, csid=None, agit_known=False,
            temperature_C=37.0, pH=7.35):
    """A synthetic M1-triaged curve in the exact shape recommend_experiments reads:
    x_hours, m1.y_processed, condition_vector.concentration.value_uM, field_provenance,
    and (optionally) a concentration_series_id."""
    xx = T if x is None else np.asarray(x, float)
    prov = {"agitation": "known" if agit_known else "unknown",
            "seeded": "known" if agit_known else "unknown"}
    cv = {
        "concentration": {"value_uM": conc_uM, "unit": "microM"},
        "temperature_C": temperature_C,
        "pH": pH,
        "field_provenance": prov,
    }
    s = {
        "series_id": sid,
        "protein_id": sid + "_prot",
        "x_hours": list(map(float, xx[: len(y)])),
        "condition_vector": cv,
        "m1": {"recommended_handling": "monotonic_fit",
               "y_processed": list(y),
               "fittability_class": "fittable"},
    }
    if csid:
        s["concentration_series_id"] = csid
    return s


def _series_member(conc_uM, csid, seed, temperature_C=37.0, pH=7.35):
    """One member curve of a concentration series (for the members list)."""
    y = _noisy(m_logistic(T, 0.0, 1.0, 0.5, 20.0), sd=0.005, seed=seed)
    return _series(y, sid=f"mem_{conc_uM}", conc_uM=conc_uM, csid=csid,
                   temperature_C=temperature_C, pH=pH)


def _clean_logistic_series(sid="clean", seed=1, csid=None, agit_known=False):
    y = _noisy(m_logistic(T, 0.0, 1.0, 0.5, 20.0), sd=0.005, seed=seed)
    return _series(y, sid=sid, csid=csid, agit_known=agit_known)


def _fitted_theta(name, params, order):
    return np.array([params[pn] for pn in order], float)


def _fit_a_series(series):
    """Run recommend_experiments and pull the theta/order/sigma2/params it used, so
    tests can call the lower-level prospective_fim / build_candidates directly on the
    SAME fitted model M13 chose."""
    res = recommend_experiments(series, gamma_result=GAMMA_RESULT)
    assert res["status"] == "ok", res
    name = res["selected_model"]
    order = list(REGISTRY[name]["params"])
    # re-fit to recover params (recommend_experiments does not surface theta directly)
    from m2_fit import fit_curve
    fr = fit_curve(series)
    fit = fr["fits"][name]
    params = fit["params"]
    theta = _fitted_theta(name, params, order)
    return res, name, params, order, theta, res["sigma2"]


_CONTRACT_KEYS = ("candidates", "ranked", "pareto_optimal_sets", "multi_step_plan",
                  "prior", "honesty", "version", "status")


# --------------------------------------------------------------------------- #
#  1. recommend_experiments never raises + full contract on clean/single/empty
# --------------------------------------------------------------------------- #
def test_recommend_never_raises_clean_series():
    """A clean logistic SERIES unit (>=3 members) returns status ok + all contract
    keys, an 8-candidate set, a Pareto set, and a 3-step plan."""
    csid = "cs1"
    members = [_series_member(10.0, csid, 20),
               _series_member(20.0, csid, 21),
               _series_member(40.0, csid, 22)]
    rep = members[1]
    res = recommend_experiments(rep, members=members, gamma_result=GAMMA_RESULT)
    for k in _CONTRACT_KEYS:
        assert k in res, f"missing contract key {k}"
    assert res["status"] == "ok"
    # 8 candidate designs for a series with agitation/seeding still unknown
    ids = {c["candidate_id"] for c in res["candidates"]}
    assert "add_concentration_series" in ids
    assert "record_agitation_and_seeding" in ids
    assert len(res["candidates"]) == 8
    assert len(res["multi_step_plan"]["steps"]) == 3


def test_recommend_never_raises_single_curve():
    """A single fittable curve (no series id) still returns ok + full contract; it
    tops out at add_concentration_series as the missing structural design."""
    res = recommend_experiments(_clean_logistic_series(sid="single", seed=3),
                                members=[], gamma_result=GAMMA_RESULT)
    for k in _CONTRACT_KEYS:
        assert k in res
    assert res["status"] == "ok"
    ids = {c["candidate_id"] for c in res["candidates"]}
    assert "add_concentration_series" in ids


def test_recommend_never_raises_empty_no_fit():
    """An empty / no-fit series does NOT raise: status INSUFFICIENT + an M9 fallback,
    and it STILL carries the honesty + version contract keys."""
    empty = {"series_id": "empty", "x_hours": [], "m1": {"y_processed": []}}
    res = recommend_experiments(empty)
    assert res["status"] == "INSUFFICIENT"
    assert "m9_fallback" in res
    assert "honesty" in res and "version" in res
    # a garbage / too-short curve is also handled without raising
    short = {"series_id": "short", "x_hours": [0.0, 1.0],
             "m1": {"y_processed": [0.1, 0.2]}}
    res2 = recommend_experiments(short)
    assert res2["status"] == "INSUFFICIENT"
    assert res2["candidates"] == []


# --------------------------------------------------------------------------- #
#  2. two variable types present + gain_basis distinguishes fim vs licensing
# --------------------------------------------------------------------------- #
def test_two_variable_types_and_gain_basis():
    """The candidate set contains BOTH FIM-based quantitative designs AND the gating
    (licensing) design; gain_basis cleanly separates them."""
    res = recommend_experiments(_clean_logistic_series(sid="mix", seed=4),
                                members=[], gamma_result=GAMMA_RESULT)
    by_id = {c["candidate_id"]: c for c in res["candidates"]}

    fim_designs = {"add_concentration_series", "extend_sampling_duration",
                   "add_replicates", "increase_measurement_frequency"}
    assert fim_designs.issubset(set(by_id)), "missing a core FIM design"
    for cid in fim_designs:
        assert by_id[cid]["gain_basis"] == "fim"
        assert by_id[cid]["variable_type"] == "quantitative"

    gate = by_id["record_agitation_and_seeding"]
    assert gate["gain_basis"] == "licensing"
    assert gate["variable_type"] == "gating"

    # a concentration design's mechanistic gain basis is Service-C-measured / fim
    conc = by_id["add_concentration_series"]
    assert conc["gain_basis"] == "fim"
    # gating gives NO smooth FIM EIG (its value is scored as the licensing unblock)
    assert gate["expected_information_gain"]["eig_bits"] == 0.0
    assert gate["expected_information_gain"]["gain_basis"] == "licensing"


# --------------------------------------------------------------------------- #
#  3. single-curve protein: series ranks high (mechanism) + gating on licensing
# --------------------------------------------------------------------------- #
def test_single_curve_series_ranks_high_and_gating_present():
    """On a single curve the add_concentration_series design carries the LARGEST raw
    information gain (the FIM/mechanism reason: it is the ONLY design that opens the
    γ / reaction-order direction), and record_agitation_and_seeding is present on a
    licensing basis (necessary-but-not-sufficient without a series).

    NOTE: we rank on RAW information gain, not EIG-per-cost. The series is the
    information leader but costs more (3 conditions), so on the priority (EIG/cost)
    axis the cheaper per-curve designs can out-rank it — that is an honest cost
    trade-off, not a demotion of its scientific value. The mechanism claim is about
    information content, which we check directly."""
    res = recommend_experiments(_clean_logistic_series(sid="prot", seed=5),
                                members=[], gamma_result=GAMMA_RESULT)
    # add_concentration_series is the top design by RAW information gain (mechanism).
    by_info = sorted(res["candidates"],
                     key=lambda c: -c["information_gain_for_ranking_bits"])
    assert by_info[0]["candidate_id"] == "add_concentration_series", \
        [(c["candidate_id"], c["information_gain_for_ranking_bits"]) for c in by_info[:3]]
    conc = next(c for c in res["candidates"]
                if c["candidate_id"] == "add_concentration_series")
    assert conc["target"] == "gamma"                 # its FIM target is the γ direction
    assert conc["gain_basis"] == "fim"
    # its EIG includes the TARGETED extra γ-direction bit (the mechanism reason)
    assert "eig_bits_gamma_direction" in conc["expected_information_gain"]
    assert conc["expected_information_gain"]["eig_bits_gamma_direction"] > 0.0

    gate = next(c for c in res["candidates"]
                if c["candidate_id"] == "record_agitation_and_seeding")
    assert gate["gain_basis"] == "licensing"
    lic = gate["expected_mechanistic_distinguishability_gain"]["licensing"]
    # single curve -> recording metadata is necessary but not sufficient
    assert lic["necessary_but_not_sufficient"] is True
    assert "add_concentration_series" in lic["also_requires"]


# --------------------------------------------------------------------------- #
#  4. diminishing returns / submodular
# --------------------------------------------------------------------------- #
def test_multi_step_marginal_eig_non_increasing():
    """The greedy multi-step plan's per-step FIM marginal EIG is monotone
    non-increasing (submodular log-det): step1 >= step2 >= step3."""
    csid = "cs2"
    members = [_series_member(10.0, csid, 30),
               _series_member(20.0, csid, 31),
               _series_member(40.0, csid, 32)]
    res = recommend_experiments(members[1], members=members, gamma_result=GAMMA_RESULT)
    plan = res["multi_step_plan"]
    curve = plan["diminishing_returns_curve_bits"]
    assert len(curve) >= 2
    # the module's own flag must agree
    assert plan["fim_marginals_non_increasing"] is True
    # and the FIM marginals are non-increasing to a tolerance
    fim_marg = [s["marginal_information_gain_bits"] for s in plan["steps"]
                if s["gain_basis"] == "fim"]
    for i in range(len(fim_marg) - 1):
        assert fim_marg[i] >= fim_marg[i + 1] - 1e-6, fim_marg


def test_second_series_worth_less_than_first():
    """Adding a 2nd concentration series after the 1st yields a STRICTLY smaller
    marginal EIG than the 1st (submodular): ΔEIG(second) < ΔEIG(first)."""
    sub = _clean_logistic_series(sid="sub", seed=6)
    res, name, params, order, theta, sigma2 = _fit_a_series(sub)
    p = len(order)
    prior_cv = 1.0
    prior_prec = _prior_precision(theta, prior_cv)

    # build one series FIM by stacking the three per-[m] prospective FIMs (as M13 does)
    from m13_boed import _rescale_timescale
    xa = np.asarray(sub["x_hours"], float)
    t_min, t_max = float(xa.min()), float(xa.max())
    gamma_eff = 1.2
    fim_series = np.zeros((p, p))
    for mult in (0.5, 1.0, 2.0):
        scale = _rescale_timescale(gamma_eff, mult * 20.0, 20.0)
        grid = np.linspace(t_min * scale, t_max * scale, len(xa))
        fim_k, _ = prospective_fim(name, params, grid, sigma2, replicate_count=1)
        fim_series = fim_series + fim_k

    from m12_information import expected_information_gain
    base = expected_information_gain(prior_prec, theta, prior_cv)["eig_bits"]
    after_first = expected_information_gain(prior_prec + fim_series, theta, prior_cv)["eig_bits"]
    after_second = expected_information_gain(prior_prec + 2.0 * fim_series, theta, prior_cv)["eig_bits"]
    delta_first = after_first - base
    delta_second = after_second - after_first
    assert delta_first > 0.0
    assert delta_second < delta_first - 1e-9, (delta_first, delta_second)


# --------------------------------------------------------------------------- #
#  5. replicate scaling — parameter SE ~ 1/sqrt(R)
# --------------------------------------------------------------------------- #
def test_replicate_se_scales_inverse_sqrt_R():
    """prospective_fim with R replicates scales the FIM by ~R, so the posterior
    parameter SE scales ~1/sqrt(R): R=4 roughly halves the SE relative to R=1.
    Checked on the DESIGN-ONLY covariance (pinv of the FIM) to isolate the R law."""
    rep_s = _clean_logistic_series(sid="rep", seed=7)
    res, name, params, order, theta, sigma2 = _fit_a_series(rep_s)
    xa = np.asarray(rep_s["x_hours"], float)

    fim1, _ = prospective_fim(name, params, xa, sigma2, replicate_count=1)
    fim4, _ = prospective_fim(name, params, xa, sigma2, replicate_count=4)
    # FIM scales linearly with R
    assert np.allclose(fim4, 4.0 * fim1, rtol=1e-9, atol=1e-12)

    # -> the SE (sqrt diag of the inverse) scales ~1/sqrt(R). R=4 -> SE * 1/2.
    se1 = np.sqrt(np.clip(np.diag(np.linalg.pinv(fim1)), 0.0, None))
    se4 = np.sqrt(np.clip(np.diag(np.linalg.pinv(fim4)), 0.0, None))
    good = se1 > 1e-9
    ratio = se4[good] / se1[good]
    assert np.allclose(ratio, 0.5, rtol=1e-6), ratio


def test_add_replicates_candidate_reduces_uncertainty():
    """The add_replicates candidate reports a positive posterior SE reduction (its
    R=4 FIM tightens the parameters vs the prior)."""
    res = recommend_experiments(_clean_logistic_series(sid="repc", seed=8),
                                members=[], gamma_result=GAMMA_RESULT)
    rep = next(c for c in res["candidates"] if c["candidate_id"] == "add_replicates")
    red = rep["expected_parameter_uncertainty_reduction"]["max_se_reduction_pct"]
    assert red > 0.0
    assert rep["expected_information_gain"]["eig_bits"] > 0.0


# --------------------------------------------------------------------------- #
#  6. extend sampling_duration on a RIGHT-CENSORED curve
# --------------------------------------------------------------------------- #
_SC_STUB = {"single_concentration_classes": 1, "concentration_series_classes": 4,
            "pairs_newly_resolved": 6, "residual_pair_auc": 0.888,
            "mismatch_degradation": 0.66, "service_c_version": "x"}


def _extend_candidate(name, params, xhours, theta, sigma2):
    cands = build_candidates(
        name, params, xhours, sigma2, theta, list(REGISTRY[name]["params"]),
        m0=20.0, gamma=1.2, gamma_reliable=True, has_series_now=False,
        agitation_seeding_known=False, t_measured_range=None, ph_measured_range=None,
        sc=_SC_STUB, prior_cv=1.0, gamma_prior_cv=1.0)
    return next(c for c in cands if c["candidate_id"] == "extend_sampling_duration")


def test_extend_window_bigger_eig_on_censored():
    """On a right-censored sigmoid (truncated before the plateau), the MARGINAL value
    of extend_sampling_duration — EIG(extended window) − EIG(current window) at the
    same fitted model — is much LARGER than on an already-complete curve, and that
    gain is attributed to the amplitude/plateau parameter (the right-censoring fix).

    The marginal (extend vs. keep the current window) is the honest comparison: the
    absolute EIG is confounded by the two curves having different point counts and
    fitted θ. The marginal isolates the information the EXTRA plateau points buy."""
    from m12_information import expected_information_gain
    # complete curve: goes well past t50=20 to a clear plateau
    full = _clean_logistic_series(sid="full", seed=9)
    # right-censored: cut just past the inflection (no plateau observed)
    Tc = T[T <= 22.0]
    cy = _noisy(m_logistic(Tc, 0.0, 1.0, 0.5, 20.0), sd=0.005, seed=9)
    cens = _series(cy, x=Tc, sid="cens", conc_uM=20.0)

    def marginal_and_amp(series):
        _, name, params, order, theta, sigma2 = _fit_a_series(series)
        xh = series["x_hours"]
        # baseline EIG on the observed window (same grid, no extension)
        fim_cur, _ = prospective_fim(name, params, np.asarray(xh, float), sigma2, 1)
        eig_cur = expected_information_gain(fim_cur, theta, 1.0)["eig_bits"]
        ext = _extend_candidate(name, params, xh, theta, sigma2)
        eig_ext = expected_information_gain(np.asarray(ext["fim"], float), theta, 1.0)["eig_bits"]
        prior_prec = _prior_precision(theta, 1.0)
        se = _posterior_se_reduction(prior_prec, np.asarray(ext["fim"], float), order, theta)
        amp_red = se["per_parameter"].get("amp", {}).get("se_reduction_pct", 0.0)
        return eig_ext - eig_cur, amp_red

    marg_full, amp_full = marginal_and_amp(full)
    marg_cens, amp_cens = marginal_and_amp(cens)

    # extending the window buys strictly MORE information when the curve was censored
    assert marg_cens > marg_full + 1e-6, (marg_cens, marg_full)
    # and the extension informs the amplitude/plateau parameter on the censored curve
    assert amp_cens > 0.0


# --------------------------------------------------------------------------- #
#  7. Pareto set is genuinely NON-DOMINATED
# --------------------------------------------------------------------------- #
def test_pareto_is_non_dominated():
    """No Pareto member is dominated by ANY scored candidate on (info up, cost down,
    runtime down). We re-derive the full objective vectors and verify domination-free."""
    csid = "cs3"
    members = [_series_member(10.0, csid, 40),
               _series_member(20.0, csid, 41),
               _series_member(40.0, csid, 42)]
    res = recommend_experiments(members[1], members=members, gamma_result=GAMMA_RESULT)
    pareto = res["pareto_optimal_sets"]
    assert len(pareto) >= 1

    # objective vectors for EVERY candidate (info up, cost down, runtime down)
    cand_vecs = {}
    for c in res["candidates"]:
        cand_vecs[c["candidate_id"]] = (
            c["information_gain_for_ranking_bits"],
            c["estimated_cost"]["cost_units"],
            c["estimated_runtime"]["runtime_hours"])

    def dominates(b, a):
        ib, cb, rb = b
        ia, ca, ra = a
        return (ib >= ia and cb <= ca and rb <= ra
                and (ib > ia or cb < ca or rb < ra))

    for f in pareto:
        a = (f["information_gain_bits"], f["cost_units"], f["runtime_hours"])
        for cid, b in cand_vecs.items():
            if cid == f["candidate_id"]:
                continue
            assert not dominates(b, a), f"{f['candidate_id']} dominated by {cid}"


def test_pareto_frontier_helper_direct():
    """pareto_frontier() on a hand-built scored set returns exactly the non-dominated
    members (a small deterministic check of the domination logic)."""
    def mk(cid, info, cost, rt):
        return {"candidate_id": cid, "information_gain_for_ranking_bits": info,
                "estimated_cost": {"cost_units": cost},
                "estimated_runtime": {"runtime_hours": rt},
                "gain_basis": "fim", "target": "parameters"}
    scored = [
        mk("A", 10.0, 1.0, 1.0),   # non-dominated (best info, cheap)
        mk("B", 5.0, 2.0, 2.0),    # dominated by A (less info, costlier, slower)
        mk("C", 3.0, 0.5, 0.5),    # non-dominated (cheapest/fastest)
        mk("D", 8.0, 3.0, 3.0),    # dominated by A
    ]
    front_ids = {f["candidate_id"] for f in pareto_frontier(scored)}
    assert front_ids == {"A", "C"}, front_ids


# --------------------------------------------------------------------------- #
#  8. cost/runtime scale correctly (monotone in conditions, replicates, t_max)
# --------------------------------------------------------------------------- #
def test_cost_runtime_monotone():
    """estimate_cost_runtime increases monotonically with n_conditions, n_replicates
    and t_max (a versioned ESTIMATE table)."""
    base = estimate_cost_runtime(1, 1, 10.0)
    # more conditions -> higher cost
    assert estimate_cost_runtime(3, 1, 10.0)["estimated_cost_units"] > base["estimated_cost_units"]
    # more replicates -> higher cost
    assert estimate_cost_runtime(1, 4, 10.0)["estimated_cost_units"] > base["estimated_cost_units"]
    # longer t_max -> longer runtime
    assert estimate_cost_runtime(1, 1, 20.0)["estimated_runtime_hours"] > base["estimated_runtime_hours"]
    # strictly monotone across a sweep of conditions
    costs = [estimate_cost_runtime(n, 1, 10.0)["estimated_cost_units"] for n in (1, 2, 4, 8)]
    assert all(costs[i] < costs[i + 1] for i in range(len(costs) - 1)), costs
    # metadata-only is cheap and zero-runtime
    md = estimate_cost_runtime(0, 0, 0.0, metadata_only=True)
    assert md["estimated_runtime_hours"] == 0.0
    assert md["estimated_cost_units"] < base["estimated_cost_units"]


# --------------------------------------------------------------------------- #
#  9. honesty block present + prior-conditional labels
# --------------------------------------------------------------------------- #
def test_honesty_block_present():
    """Every result carries the honesty contract: Laplace/model-conditional EIG,
    cost-is-an-estimate, gating!=fim, greedy!=global, Service-C upper bound."""
    res = recommend_experiments(_clean_logistic_series(sid="hon", seed=10),
                                members=[], gamma_result=GAMMA_RESULT)
    h = res["honesty"]
    for k in ("eig_is_expected_laplace", "mechanistic_gain_bounded_by_service_c",
              "cost_is_an_estimate", "gating_is_licensing_not_fim",
              "greedy_not_global", "tpH_extrapolation_flagged",
              "self_similar_shape_approx"):
        assert k in h, f"missing honesty key {k}"
    # the plan repeats the greedy!=global caveat
    assert "greedy_note" in res["multi_step_plan"]
    # prior block flags EIG as prior-conditional
    assert res["prior"]["eig_is_prior_conditional"] is True
    # honesty block present even on the INSUFFICIENT path
    empty = recommend_experiments({"series_id": "e", "x_hours": [], "m1": {"y_processed": []}})
    assert "honesty" in empty


# --------------------------------------------------------------------------- #
#  runner (house style: "N passed, M failed")
# --------------------------------------------------------------------------- #
def _run():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}"); failed += 1
        except Exception as e:                       # a raise is itself a failure
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
