#!/usr/bin/env python3
"""Fast, deterministic, stdlib-only smoke test for the PRISE web app.

Boots the real server on an ephemeral port in a background thread and exercises
the public endpoints, asserting HTTP 200 + the JSON keys the frontend renders.

Run:  python web/test_web.py        (exit 0 = pass, non-zero = fail)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import server  # noqa: E402  (the module under test)


def _get(base, path, timeout=10):
    url = base + path
    with urllib.request.urlopen(url, timeout=timeout) as r:
        body = r.read().decode("utf-8")
        return r.status, json.loads(body)


def _get_raw(base, path):
    url = base + path
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read().decode("utf-8")


def _get_with_status(base, path, timeout=10):
    """GET that returns (status, json) even on a 4xx (urlopen raises on >=400)."""
    url = base + path
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # noqa: PERF203
        return e.code, json.loads(e.read().decode("utf-8"))


def _post(base, path, obj, timeout=20):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(
        base + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main():
    # Load artifacts once (same path the entrypoint uses).
    server.STORE.load()

    # Bind an ephemeral port (0) and serve in a daemon thread.
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    port = httpd.server_address[1]
    base = "http://127.0.0.1:%d" % port
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    failures = []
    try:
        # --- GET / serves the SPA ---
        st, html = _get_raw(base, "/")
        _require(st == 200, "GET / status %s" % st)
        _require("/static/app.js" in html, "GET / missing app.js reference")

        # --- GET /static/app.js serves a non-empty, PARSEABLE frontend bundle ---
        # A frontend that references an undefined function (e.g. loadPooling) or
        # otherwise fails to parse silently breaks every analysis panel while the
        # API tests stay green. Parse the SERVED app.js so a broken frontend fails
        # the smoke test. Node in the deploy box is too old for modern JS, so we
        # downgrade `??`/`?.` to their non-optional equivalents purely for SYNTAX
        # validation and parse with esprima.
        st, appjs = _get_raw(base, "/static/app.js")
        _require(st == 200, "GET /static/app.js status %s" % st)
        _require(len(appjs.strip()) > 0, "GET /static/app.js is empty")
        try:
            import esprima  # type: ignore
        except ImportError:
            sys.stderr.write(
                "NOTE: esprima not installed — skipping app.js parse check "
                "(pip install esprima to enable it)\n")
        else:
            import re as _re
            _src = appjs.replace("??", "||")
            _src = _re.sub(r"\?\.", ".", _src)
            try:
                esprima.parseScript(_src)
            except Exception as _exc:  # noqa: BLE001
                raise AssertionError(
                    "/static/app.js failed to parse: %s" % _exc)

        # --- /api/health ---
        st, h = _get(base, "/api/health")
        _require(st == 200, "/api/health status %s" % st)
        _require(h.get("ok") is True, "/api/health not ok")
        _require(h.get("n_proteins", 0) > 0, "no proteins loaded")
        _require(h.get("n_curves", 0) > 0, "no curves loaded")

        # --- /api/overview ---
        st, ov = _get(base, "/api/overview")
        _require(st == 200, "/api/overview status %s" % st)
        for k in ("inference_ladder", "yield_tiers", "blocker_census",
                  "boundary_gap", "validity_ceiling", "attribution"):
            _require(k in ov, "/api/overview missing key %r" % k)
        _require((ov["inference_ladder"] or {}).get("rungs"),
                 "/api/overview inference_ladder has no rungs")

        # --- /api/crossmodal (M10): cross-modal sequence×kinetics×structure ---
        st, cm = _get(base, "/api/crossmodal")
        _require(st == 200, "/api/crossmodal status %s" % st)
        if cm.get("available"):
            for k in ("coverage", "P1_sequence_to_kinetics", "P2_apr_to_regime",
                      "P3_discordance", "honesty_constraints", "version"):
                _require(k in cm, "/api/crossmodal missing key %r" % k)
            cov = cm["coverage"]
            # the binding honesty constraint: effective n is PROTEINS, not curves
            _require(cov.get("effective_n_is_proteins_not_curves") is True,
                     "/api/crossmodal coverage must flag protein-as-unit")
            _require("PROTEINS, NOT" in (cov.get("effective_n_banner") or ""),
                     "/api/crossmodal missing effective-n banner")
            _require(cov.get("n_proteins_predictor_AND_point_t50", 0) > 0,
                     "/api/crossmodal no predictor+t50 proteins")
            # P1 cards carry a permutation p, LOPO skill, and a variance ceiling
            p1 = cm["P1_sequence_to_kinetics"]
            _require(p1.get("association_cards"),
                     "/api/crossmodal P1 has no association cards")
            c0 = p1["association_cards"][0]
            for k in ("predictor", "target", "spearman_rho", "permutation_p",
                      "variance_ceiling_fraction", "verdict_plain_english"):
                _require(k in c0, "/api/crossmodal P1 card missing %r" % k)
            _require(c0.get("unit") == "protein",
                     "/api/crossmodal P1 unit must be protein")
            # P3 fills the §3-M6 null_disagreement_rate (was None)
            _require(cm["P3_discordance"].get("null_disagreement_rate") is not None,
                     "/api/crossmodal P3 null_disagreement_rate not filled")
        else:
            _require("reason" in cm,
                     "/api/crossmodal unavailable must carry a reason")

        # --- / serves the SPA with the Cross-modal tab present ---
        _require("data-tab=\"crossmodal\"" in html or "crossmodal" in html,
                 "GET / missing Cross-modal tab")

        # --- /api/conformal: frozen conformal calibration + realized coverage ---
        st, conf = _get(base, "/api/conformal")
        _require(st == 200, "/api/conformal status %s" % st)
        if conf.get("available"):
            for k in ("version", "target_coverage", "calibration_quantiles",
                      "validation"):
                _require(k in conf, "/api/conformal missing key %r" % k)
            cq = conf["calibration_quantiles"]
            _require("t50" in cq and "regime" in cq,
                     "/api/conformal calibration_quantiles missing t50/regime")
            _require(cq["t50"].get("strata") is not None,
                     "/api/conformal t50 has no Mondrian strata")
            v = conf["validation"]
            # the proof: realized t50 coverage meets target (within slack)
            t50m = (v.get("t50") or {}).get("marginal") or {}
            tgt = conf.get("target_coverage") or 0.9
            _require(t50m.get("realized_coverage") is not None,
                     "/api/conformal validation has no realized t50 coverage")
            _require(t50m["realized_coverage"] >= tgt - 0.12,
                     "/api/conformal realized t50 coverage %.3f below target %.2f"
                     % (t50m["realized_coverage"], tgt))
        else:
            _require("reason" in conf,
                     "/api/conformal unavailable must carry a reason")

        # --- /api/reality-check: blind external reality check (§6C) ---
        st, rc = _get(base, "/api/reality-check")
        _require(st == 200, "/api/reality-check status %s" % st)
        if rc.get("available"):
            for k in ("version", "reality_check_result", "fixed_points",
                      "reserved_proteins_note", "falsification_note", "honesty_note"):
                _require(k in rc, "/api/reality-check missing key %r" % k)
            res = rc["reality_check_result"]
            for k in ("n_fixed_points_evaluated", "n_consistent",
                      "n_contradiction", "n_insufficient", "verdict_plain_english"):
                _require(k in res, "/api/reality-check result missing %r" % k)
            _require(isinstance(rc["fixed_points"], list) and rc["fixed_points"],
                     "/api/reality-check has no fixed points")
            fp0 = rc["fixed_points"][0]
            for k in ("protein", "consensus", "prise", "checks", "verdict"):
                _require(k in fp0, "/api/reality-check fixed point missing %r" % k)
            _require(fp0["verdict"] in
                     ("CONSISTENT", "CONTRADICTION", "INSUFFICIENT_DATA"),
                     "/api/reality-check bad verdict %r" % fp0.get("verdict"))
            # consensus must carry a citation (the pre-registered fixed point)
            _require(fp0["consensus"].get("citation"),
                     "/api/reality-check fixed point missing consensus citation")
            # reserved-proteins / leakage discipline must be surfaced
            _require("no_leakage" in rc["reserved_proteins_note"],
                     "/api/reality-check missing leakage discipline")
        else:
            _require("reason" in rc,
                     "/api/reality-check unavailable must carry a reason")

        # --- /api/metadata-overview (M11): corpus metadata-quality summary ---
        st, mo = _get(base, "/api/metadata-overview")
        _require(st == 200, "/api/metadata-overview status %s" % st)
        if mo.get("available"):
            for k in ("metadata_score", "mechanistic_completeness",
                      "universal_blocker", "top_recommendations", "honesty"):
                _require(k in mo, "/api/metadata-overview missing key %r" % k)
            ub = mo["universal_blocker"]
            _require(ub.get("fields") == ["agitation", "seeded"],
                     "/api/metadata-overview universal blocker must be {agitation, seeded}")
            _require("agitation" in (ub.get("why_mechanism_licensed_is_zero") or ""),
                     "/api/metadata-overview must explain the {agitation,seeded} blocker")
            _require(isinstance(mo["top_recommendations"], list)
                     and len(mo["top_recommendations"]) >= 1,
                     "/api/metadata-overview has no corpus recommendations")
            _require((mo["metadata_score"].get("distribution") or {}).get("median")
                     is not None,
                     "/api/metadata-overview has no median metadata_score")
        else:
            _require("reason" in mo,
                     "/api/metadata-overview unavailable must carry a reason")

        # --- /api/literature (M15): extracted claims + conflicts + CPAD-validation ---
        # CORE FRAMING: these are CLAIMS, not facts. Assert claims are present with
        # provenance, a cross-paper conflict is surfaced UNRESOLVED, the CPAD-validation
        # precision is present, and the claims-not-facts honesty note is visible.
        st, lit = _get(base, "/api/literature")
        _require(st == 200, "/api/literature status %s" % st)
        if lit.get("available"):
            for k in ("n_claims", "claims_by_paper", "field_coverage",
                      "conflicts", "validation", "honesty"):
                _require(k in lit, "/api/literature missing key %r" % k)
            # claims present, grouped by paper, each carrying full provenance
            _require(lit["n_claims"] > 0, "/api/literature has no claims")
            _require(isinstance(lit["claims_by_paper"], list)
                     and len(lit["claims_by_paper"]) >= 1,
                     "/api/literature claims_by_paper empty")
            p0 = lit["claims_by_paper"][0]
            _require(isinstance(p0.get("claims"), list) and p0["claims"],
                     "/api/literature paper has no claims")
            c0 = p0["claims"][0]
            # provenance trail + measured confidence + verbatim source_text
            for k in ("field", "value", "group", "doi", "page", "location_type",
                      "char_span", "source_text", "confidence"):
                _require(k in c0, "/api/literature claim missing provenance key %r" % k)
            _require(c0.get("source_text"),
                     "/api/literature claim missing VERBATIM source_text")
            _require(isinstance(c0.get("confidence"), (int, float)),
                     "/api/literature claim confidence is not measured")
            # field-coverage: which target fields were extracted + counts
            _require(isinstance(lit["field_coverage"], list)
                     and len(lit["field_coverage"]) >= 1,
                     "/api/literature has no field coverage")
            _require("count" in lit["field_coverage"][0],
                     "/api/literature field_coverage row missing count")
            # a cross-paper conflict is present + surfaced UNRESOLVED with both provenances
            _require(isinstance(lit["conflicts"], list) and len(lit["conflicts"]) >= 1,
                     "/api/literature has no conflicts")
            cf0 = lit["conflicts"][0]
            _require("UNRESOLVED" in (cf0.get("resolution") or ""),
                     "/api/literature conflict must be UNRESOLVED")
            _require(isinstance(cf0.get("claims"), list) and len(cf0["claims"]) >= 2,
                     "/api/literature conflict must retain both papers' claims")
            for cl in cf0["claims"][:2]:
                for k in ("doi", "page", "source_text", "value"):
                    _require(k in cl, "/api/literature conflict claim missing %r" % k)
            # CPAD-validation accuracy — MEASURED against the curated gold standard
            val = lit["validation"]
            _require(val.get("precision") is not None,
                     "/api/literature validation missing precision")
            _require(val.get("recall") is not None and val.get("accuracy") is not None,
                     "/api/literature validation missing recall/accuracy")
            _require(val.get("n_curated_curves") is not None,
                     "/api/literature validation missing n_curated_curves")
            # honesty: claims-not-facts + append-only must be visible
            hon = lit["honesty"]
            _require("claims_not_facts" in hon,
                     "/api/literature honesty missing claims-not-facts note")
            _require("append_only" in hon,
                     "/api/literature honesty missing append-only note")
            _require("CLAIM" in (hon.get("claims_not_facts") or "")
                     or "never a trusted fact" in (hon.get("claims_not_facts") or ""),
                     "/api/literature claims-not-facts note must say claims != facts")
        else:
            _require("reason" in lit,
                     "/api/literature unavailable must carry a reason")

        # --- /api/metadata/{series_id} (M11): per-dataset metadata-quality record ---
        if server.STORE.metadata_by_series:
            some_sid = next(iter(server.STORE.metadata_by_series))
            st, mq = _get(base, "/api/metadata/" + urllib.parse.quote(some_sid))
            _require(st == 200, "/api/metadata status %s" % st)
            _require(mq.get("available") is True,
                     "/api/metadata should be available for %s" % some_sid)
            for k in ("metadata_score", "metadata_uncertainty",
                      "mechanistic_completeness", "mechanistic_blockers",
                      "delta_inferential_power", "recommended_missing_metadata"):
                _require(k in mq, "/api/metadata missing key %r" % k)
            # ranked missing fields carry impact + a plain-English consequence chain
            for it in (mq["delta_inferential_power"] or [])[:1]:
                for k in ("field", "impact", "blocks", "consequence_chain"):
                    _require(k in it, "/api/metadata delta item missing %r" % k)
            # missing series -> {available: False} + 404
            st0, mq0 = _get_with_status(
                base, "/api/metadata/" + urllib.parse.quote("NoSuchSeries_zzz"))
            _require(st0 == 404, "/api/metadata missing series should 404, got %s" % st0)
            _require(mq0.get("available") is False,
                     "/api/metadata missing series should be unavailable")

        # --- /api/information/{series_id} (M12): per-curve information content ---
        # The headline deliverable: WHERE information concentrates across the curve.
        if server.STORE.information_by_series:
            # prefer a real computable ("ok") series so we can assert the geometry;
            # CPAD-TK-1039 is a known fit, fall back to the first ok record.
            info_sid = None
            if "CPAD-TK-1039" in server.STORE.information_by_series:
                info_sid = "CPAD-TK-1039"
            else:
                for sid, r in server.STORE.information_by_series.items():
                    if r.get("status") == "ok":
                        info_sid = sid
                        break
            _require(info_sid is not None, "no ok information-content record for /api/information test")
            st, ic = _get(base, "/api/information/" + urllib.parse.quote(info_sid))
            _require(st == 200, "/api/information status %s" % st)
            _require(ic.get("available") is True,
                     "/api/information should be available for %s" % info_sid)
            _require(ic.get("computable") is True,
                     "/api/information should be computable for an ok series")
            # the headline: information-density array + phase fractions
            idn = ic.get("information_density") or {}
            _require(isinstance(idn.get("density"), list) and len(idn["density"]) >= 1,
                     "/api/information missing information-density array")
            _require("phase_fractions" in idn,
                     "/api/information missing phase_fractions")
            _require(idn.get("total_information") is not None,
                     "/api/information missing total_information")
            # the sloppiness story: effective rank HARD vs ENTROPY
            for k in ("effective_rank_hard", "effective_rank_entropy",
                      "condition_number", "information_richness_score",
                      "parameter_observability", "recommended_additional_measurements",
                      "honesty"):
                _require(k in ic, "/api/information missing key %r" % k)
            po = ic["parameter_observability"]
            _require(isinstance(po.get("per_parameter"), dict) and po["per_parameter"],
                     "/api/information observability has no per-parameter table")
            # honesty: FIM is local / linearized / model-conditional / post-selection
            hon = ic["honesty"]
            _require("post_selection" in hon and "fim_is_local" in hon,
                     "/api/information honesty missing local/post-selection clauses")
            # missing series degrades to {available: False} + 404
            st0, ic0 = _get_with_status(
                base, "/api/information/" + urllib.parse.quote("NoSuchSeries_zzz"))
            _require(st0 == 404, "/api/information missing series should 404, got %s" % st0)
            _require(ic0.get("available") is False,
                     "/api/information missing series should be unavailable")

        # --- /api/boed/{series_id} (M13): Bayesian Optimal Experimental Design ---
        # The optimal-design capstone: ranked next-experiment recommendations, the
        # cost-vs-information Pareto frontier, and the greedy multi-step plan.
        if server.STORE.boed_by_series:
            # prefer a real ok series (CPAD-TK-1130 is a known ok record); else the
            # first status=='ok' record so we can assert the ranked/pareto/plan blocks.
            boed_sid = None
            if ("CPAD-TK-1130" in server.STORE.boed_by_series
                    and server.STORE.boed_by_series["CPAD-TK-1130"].get("status") == "ok"):
                boed_sid = "CPAD-TK-1130"
            else:
                for sid, r in server.STORE.boed_by_series.items():
                    if r.get("status") == "ok":
                        boed_sid = sid
                        break
            _require(boed_sid is not None, "no ok BOED record for /api/boed test")
            st, bd = _get(base, "/api/boed/" + urllib.parse.quote(boed_sid))
            _require(st == 200, "/api/boed status %s" % st)
            _require(bd.get("available") is True,
                     "/api/boed should be available for %s" % boed_sid)
            _require(bd.get("status") == "ok",
                     "/api/boed expected an ok record for %s" % boed_sid)
            # the three headline blocks: ranked[], pareto_optimal_sets[], multi_step_plan
            for k in ("candidates", "ranked", "pareto_optimal_sets", "multi_step_plan",
                      "top_recommendation", "current_state", "honesty", "selected_model"):
                _require(k in bd, "/api/boed missing key %r" % k)
            _require(isinstance(bd["ranked"], list) and len(bd["ranked"]) >= 1,
                     "/api/boed ranked has no recommendations")
            r0 = bd["ranked"][0]
            for k in ("rank", "candidate_id", "gain_basis",
                      "information_gain_for_ranking_bits",
                      "priority_score_bits_per_cost", "cost_units"):
                _require(k in r0, "/api/boed ranked row missing %r" % k)
            _require(r0["rank"] == 1, "/api/boed top ranked row is not rank 1")
            # ranked ascending by rank + descending by priority (EIG/cost)
            ranks = [r["rank"] for r in bd["ranked"]]
            _require(ranks == sorted(ranks), "/api/boed ranked not ordered by rank")
            _require(isinstance(bd["pareto_optimal_sets"], list)
                     and len(bd["pareto_optimal_sets"]) >= 1,
                     "/api/boed has no Pareto-optimal designs")
            pf0 = bd["pareto_optimal_sets"][0]
            for k in ("candidate_id", "information_gain_bits", "cost_units",
                      "runtime_hours", "gain_basis"):
                _require(k in pf0, "/api/boed pareto member missing %r" % k)
            # a candidate carries the four §3-M13 quantities the table renders
            _require(isinstance(bd["candidates"], list) and len(bd["candidates"]) >= 1,
                     "/api/boed has no candidates")
            c0 = bd["candidates"][0]
            for k in ("expected_information_gain",
                      "expected_parameter_uncertainty_reduction",
                      "expected_mechanistic_distinguishability_gain",
                      "estimated_cost", "estimated_runtime", "gain_basis", "target"):
                _require(k in c0, "/api/boed candidate missing key %r" % k)
            # the multi-step plan: steps + diminishing-returns curve + greedy note
            plan = bd["multi_step_plan"]
            for k in ("steps", "diminishing_returns_curve_bits",
                      "cumulative_gain_bits", "cumulative_cost_units", "greedy_note"):
                _require(k in plan, "/api/boed multi_step_plan missing key %r" % k)
            _require(isinstance(plan["steps"], list) and len(plan["steps"]) >= 1,
                     "/api/boed plan has no steps")
            s0 = plan["steps"][0]
            for k in ("step", "candidate_id", "marginal_information_gain_bits",
                      "cumulative_gain_bits", "cumulative_cost_units"):
                _require(k in s0, "/api/boed plan step missing %r" % k)
            _require(isinstance(plan["diminishing_returns_curve_bits"], list)
                     and len(plan["diminishing_returns_curve_bits"]) >= 1,
                     "/api/boed plan has no diminishing-returns curve")
            # honesty: the Laplace / gating≠FIM / cost-is-estimate / greedy≠global clauses
            hon = bd["honesty"]
            for k in ("eig_is_expected_laplace", "gating_is_licensing_not_fim",
                      "cost_is_an_estimate", "greedy_not_global"):
                _require(k in hon, "/api/boed honesty missing clause %r" % k)
            # missing series degrades to {available: False} + 404
            st0, bd0 = _get_with_status(
                base, "/api/boed/" + urllib.parse.quote("NoSuchSeries_zzz"))
            _require(st0 == 404, "/api/boed missing series should 404, got %s" % st0)
            _require(bd0.get("available") is False,
                     "/api/boed missing series should be unavailable")

        # --- /api/proteins ---
        st, pl = _get(base, "/api/proteins")
        _require(st == 200, "/api/proteins status %s" % st)
        _require(pl.get("n", 0) > 0 and pl.get("proteins"),
                 "/api/proteins returned no proteins")
        first_pid = pl["proteins"][0]["protein_id"]

        # --- /api/protein/{first} ---
        st, det = _get(base, "/api/protein/" + urllib.parse.quote(first_pid))
        _require(st == 200, "/api/protein status %s" % st)
        _require(det.get("units"), "/api/protein has no units")
        u0 = det["units"][0]
        _require("series_id" in u0, "unit missing series_id")
        _require("condition_vector" in u0, "unit missing condition_vector")
        _require("information_yield" in u0, "unit missing information_yield")

        # --- /api/curve/{first series of that protein} ---
        first_sid = u0["series_id"]
        st, cur = _get(base, "/api/curve/" + urllib.parse.quote(first_sid))
        _require(st == 200, "/api/curve status %s" % st)
        for k in ("series_id", "x", "y", "triage", "condition_vector"):
            _require(k in cur, "/api/curve missing key %r" % k)
        _require(isinstance(cur["x"], list) and len(cur["x"]) >= 1,
                 "/api/curve has no x points")

        # --- /api/models/{a real fittable series} : ranked behaviour comparison ---
        # find a fittable curve with enough points for the full bank (>= ~10 pts)
        fittable_sid = None
        for sid, cur in server.STORE.curve_by_series.items():
            m1 = cur.get("m1", {})
            if (m1.get("fittability_class") == "fittable"
                    and len(cur.get("x_hours") or []) >= 12):
                fittable_sid = sid
                break
        _require(fittable_sid is not None, "no fittable curve found for /api/models test")
        # the full bank incl. Tier-B ODE fits can be slow -> generous timeout
        st, mc = _get(base, "/api/models/" + urllib.parse.quote(fittable_sid),
                      timeout=90)
        _require(st == 200, "/api/models status %s" % st)
        if server.ENGINE_OK:
            _require("error" not in mc, "/api/models error: %s" % mc.get("error"))
            for k in ("models", "winner", "identifiability", "family_notes",
                      "bootstrap_selection", "honesty"):
                _require(k in mc, "/api/models missing key %r" % k)
            ms = mc["models"]
            _require(isinstance(ms, list) and len(ms) > 1,
                     "/api/models should return >1 ranked model")
            # ranked by AICc ascending + winner flagged
            aiccs = [m["aicc"] for m in ms]
            _require(aiccs == sorted(aiccs), "/api/models not ranked by AICc ascending")
            _require(ms[0].get("winner") is True, "/api/models top row not flagged winner")
            # >1 family present (descriptive + mechanistic at least)
            fams = {m["family"] for m in ms}
            _require(len(fams) > 1, "/api/models should span >1 family, got %s" % fams)
            _require("mechanistic" in fams,
                     "/api/models should include a Tier-B mechanistic family")
            for m in ms:
                for k in ("name", "family", "n_params", "r2", "aicc",
                          "delta_aicc", "akaike_weight", "params"):
                    _require(k in m, "/api/models row missing %r" % k)
            # identifiability block on the winner
            _require("flag" in mc["identifiability"],
                     "/api/models identifiability missing flag")
            # honesty: Tier-B is not a licensed mechanism call
            _require("not a licensed mechanism" in mc["honesty"].lower()
                     or "still refuses" in mc["honesty"].lower(),
                     "/api/models honesty note missing mechanism caveat")
        else:
            _require("error" in mc,
                     "/api/models should error when engine unavailable")

        # --- /api/series-gamma/{a protein with a series} : dual-γ + points ---
        gamma_protein = "Amyloid Beta peptide-ABeta40"
        st, sg = _get(base, "/api/series-gamma/" + urllib.parse.quote(gamma_protein),
                      timeout=120)
        _require(st == 200, "/api/series-gamma status %s" % st)
        _require("error" not in sg, "/api/series-gamma error: %s" % sg.get("error"))
        _require(sg.get("n_series", 0) > 0 and sg.get("series"),
                 "/api/series-gamma returned no series for %s" % gamma_protein)
        s0 = sg["series"][0]
        for k in ("gamma_regression", "gamma_global", "disagreement", "points"):
            _require(k in s0, "/api/series-gamma series missing key %r" % k)
        _require(isinstance(s0["points"], list),
                 "/api/series-gamma points is not a list")
        if server.ENGINE_OK:
            # at least one series for this protein must yield per-concentration points
            total_pts = sum(len(s.get("points") or []) for s in sg["series"])
            _require(total_pts > 0,
                     "/api/series-gamma produced no (concentration, t50) points")
            # a point carries concentration + t50 + censoring
            for s in sg["series"]:
                for p in (s.get("points") or []):
                    for k in ("concentration_uM", "t50", "censoring_class"):
                        _require(k in p, "/api/series-gamma point missing %r" % k)
                    break

        # --- /api/dose-response/{a protein with R-rows} : ranked k_agg-vs-conc bank ---
        # /api/proteins must flag which proteins have a dose-response (R-row) fit
        dr_flagged = [p["protein_id"] for p in pl["proteins"]
                      if p.get("has_dose_response")]
        _require(len(dr_flagged) > 0,
                 "/api/proteins must flag >=1 protein with has_dose_response")
        # Alpha-Synuclein has R-rows; use it (fall back to the first flagged protein)
        dr_protein = ("Alpha-Synuclein" if "Alpha-Synuclein" in dr_flagged
                      else dr_flagged[0])
        st, dr = _get(base, "/api/dose-response/" + urllib.parse.quote(dr_protein))
        _require(st == 200, "/api/dose-response status %s" % st)
        _require(dr.get("available") is True,
                 "/api/dose-response should be available for %s" % dr_protein)
        for k in ("models", "points", "best_by_aicc", "n_points", "honesty"):
            _require(k in dr, "/api/dose-response missing key %r" % k)
        ms = dr["models"]
        _require(isinstance(ms, list) and len(ms) >= 1,
                 "/api/dose-response returned no ranked models")
        # ranked ascending by AICc + the best flagged + carries R²/AICc
        aiccs = [m["aicc"] for m in ms]
        _require(aiccs == sorted(aiccs),
                 "/api/dose-response models not ranked by AICc ascending")
        _require(any(m.get("best") for m in ms),
                 "/api/dose-response has no best-flagged model")
        for m in ms:
            for k in ("model", "r2", "aicc", "best"):
                _require(k in m, "/api/dose-response model row missing %r" % k)
        # raw (concentration, k_agg) points present for the scatter
        _require(isinstance(dr["points"], list) and len(dr["points"]) >= 2,
                 "/api/dose-response has too few (conc, k_agg) points")
        for p in dr["points"][:1]:
            for k in ("concentration_uM", "k_agg"):
                _require(k in p, "/api/dose-response point missing %r" % k)
        # honesty: distinct from the engine's fitted t50-scaling γ
        _require("precomputed" in dr["honesty"].lower()
                 or "r-row" in dr["honesty"].lower(),
                 "/api/dose-response honesty must flag CPAD precomputed k_agg")
        # missing protein degrades to {available: False} + 404
        st, dr0 = _get_with_status(base, "/api/dose-response/" +
                                   urllib.parse.quote("NotARealProtein_zzz"))
        _require(st == 404, "/api/dose-response missing protein should 404, got %s" % st)
        _require(dr0.get("available") is False,
                 "/api/dose-response missing protein should be unavailable")

        # --- /api/meta/{protein} (M14): cross-study hierarchical meta-analysis ---
        # A MULTI-study protein returns the full record (forest plot rows + pooled
        # diamond + prediction interval, heterogeneity Q/I²/τ², variance decomposition
        # with the lab-confounded / batch-unidentifiable honesty, leave-one-out, funnel
        # + Egger). A SINGLE-study protein returns an honest single_study note (NEVER a
        # 500). Both answer HTTP 200 with available=True.
        meta_protein = "Amyloid Beta peptide-ABeta42"
        st, ma = _get(base, "/api/meta/" + urllib.parse.quote(meta_protein))
        _require(st == 200, "/api/meta status %s" % st)
        _require(ma.get("available") is True,
                 "/api/meta should be available for %s" % meta_protein)
        _require(ma.get("status") == "meta_analysis",
                 "/api/meta %s should be a meta_analysis" % meta_protein)
        for k in ("n_studies", "n_curves", "pooled_estimate", "forest_plot",
                  "heterogeneity", "variance_decomposition", "leave_one_study_out",
                  "publication_bias", "honesty"):
            _require(k in ma, "/api/meta missing key %r" % k)
        _require(ma["n_studies"] >= 2, "/api/meta multi-study must have >=2 studies")
        # the deliverable: forest plot rows (per-study effect + CI + weight) + the
        # pooled diamond + the new-study prediction interval
        fp = ma["forest_plot"]
        _require(isinstance(fp.get("rows"), list) and len(fp["rows"]) >= 2,
                 "/api/meta forest_plot has too few study rows")
        r0 = fp["rows"][0]
        for k in ("effect_t50", "ci95_t50", "weight_pct", "n_curves"):
            _require(k in r0, "/api/meta forest row missing %r" % k)
        _require(fp.get("pooled_diamond") and fp["pooled_diamond"].get("mu_t50") is not None,
                 "/api/meta forest_plot missing pooled diamond")
        _require(isinstance(fp.get("prediction_interval_t50"), list)
                 and len(fp["prediction_interval_t50"]) == 2,
                 "/api/meta forest_plot missing new-study prediction interval")
        # heterogeneity: Q (df, p) + I² + τ²_between
        het = ma["heterogeneity"]
        for k in ("Q", "df", "p", "I2", "tau2_between"):
            _require(k in het, "/api/meta heterogeneity missing %r" % k)
        _require(het["I2"] is not None, "/api/meta heterogeneity has no I2")
        # variance decomposition: the lab-confounded / batch-unidentifiable honesty
        vd = ma["variance_decomposition"]
        _require("fractions" in vd, "/api/meta variance_decomposition missing fractions")
        _require(vd.get("laboratory") == "confounded_with_study"
                 or isinstance(vd.get("laboratory"), (int, float)),
                 "/api/meta variance_decomposition laboratory must be confounded/value")
        _require(vd.get("batch") == "unidentifiable_not_recorded",
                 "/api/meta variance_decomposition batch must be unidentifiable")
        # leave-one-out influence rows
        _require(isinstance(ma["leave_one_study_out"], list)
                 and len(ma["leave_one_study_out"]) >= 2,
                 "/api/meta leave_one_study_out has too few rows")
        loo0 = ma["leave_one_study_out"][0]
        for k in ("pooled_mu_t50_hours", "I2", "is_outlier"):
            _require(k in loo0, "/api/meta leave-one-out row missing %r" % k)
        # publication bias: funnel points (effect vs se) + Egger
        pb = ma["publication_bias"]
        _require((pb.get("funnel") or {}).get("points"),
                 "/api/meta publication_bias has no funnel points")
        pt0 = pb["funnel"]["points"][0]
        for k in ("effect_log", "se"):
            _require(k in pt0, "/api/meta funnel point missing %r" % k)
        _require("egger" in pb, "/api/meta publication_bias missing egger")
        # honesty: condition-adjusted (never raw t50) + lab-confounded + batch clauses
        _require(ma["pooled_estimate"].get("condition_adjusted") is True,
                 "/api/meta pooled estimate must be condition-adjusted")
        hon = ma["honesty"]
        for k in ("condition_adjustment", "laboratory", "batch", "bayesian"):
            _require(k in hon, "/api/meta honesty missing clause %r" % k)

        # a SINGLE-study protein: honest single_study note, still 200 + available
        single_protein = None
        for name, rec in server.STORE.meta_single_by_protein.items():
            if rec.get("status") == "single_study":
                single_protein = name
                break
        if single_protein is not None:
            st, ms1 = _get(base, "/api/meta/" + urllib.parse.quote(single_protein))
            _require(st == 200, "/api/meta single-study status %s" % st)
            _require(ms1.get("available") is True,
                     "/api/meta single-study should be available for %s" % single_protein)
            _require(ms1.get("status") == "single_study",
                     "/api/meta %s should be single_study" % single_protein)
            _require(ms1.get("note"),
                     "/api/meta single-study must carry an honest note")
            _require("forest_plot" not in ms1,
                     "/api/meta single-study must NOT fabricate a forest plot")
        # an absent protein also degrades honestly to single_study (never 500)
        st, ms0 = _get(base, "/api/meta/" + urllib.parse.quote("NotARealProtein_zzz"))
        _require(st == 200, "/api/meta absent protein should 200, got %s" % st)
        _require(ms0.get("available") is True and ms0.get("status") == "single_study",
                 "/api/meta absent protein should be an honest single_study")

        # --- /api/structure/{protein} (M16): sequence<->structure<->kinetics bridge ---
        # CORE FRAMING: each of the 6 genuinely-3D features is a NESTED block
        # {proxy, has_real_3d[, real_3d]}. The 1-D sequence/APR PROXY is ALWAYS present
        # and ALWAYS stamped is_3d_derived=False; a REAL 3D value (derivation=pdb_3d,
        # is_3d_derived=True, per-feature `fidelity`) is added ONLY where a PDB was
        # actually fetched + parsed. A protein with NO PDB must carry NO real_3d key —
        # a fabricated 3D number is the exact failure mode this endpoint exists to
        # prevent. The structure->kinetics associations are STATISTICAL ONLY, never
        # causal (n<=26 -> mostly null). Never 500 — an absent protein degrades to
        # {available: False}.
        if server.STORE.structure_by_uniprot:
            # the 6 features that are PROXY-without-a-PDB and REAL-3D-with-one
            NESTED_3D_FEATURES = ("beta_sheet_content", "solvent_accessibility",
                                  "hydrophobic_patches", "electrostatic_surface",
                                  "secondary_structure", "surface_curvature")

            def _check_struct_schema(sx, want_real_3d, who):
                """Assert the FULL nested M16 structural-feature contract on one
                payload. `want_real_3d` selects the branch: True = a PDB was fetched
                and parsed, False = no PDB (proxy-only, and NOTHING may pretend
                otherwise)."""
                sf = sx["structural_features"]
                for k in ("aggregation_hotspots", "contact_map") + NESTED_3D_FEATURES:
                    _require(k in sf,
                             "/api/structure %s structural_features missing %r" % (who, k))
                # (0) container-level REAL-3D stamp
                _require(sf.get("has_real_3d_features") is want_real_3d,
                         "/api/structure %s has_real_3d_features must be %r"
                         % (who, want_real_3d))
                if want_real_3d:
                    _require(isinstance(sf.get("real_3d_pdb_id"), str)
                             and len(sf["real_3d_pdb_id"]) >= 4,
                             "/api/structure %s real 3D must name the PDB it measured"
                             % who)
                    _require(sf.get("real_3d_native_or_fibril")
                             in ("native", "fibril", "unknown"),
                             "/api/structure %s real 3D must stamp native-vs-fibril "
                             "(native != the aggregation-competent state)" % who)
                else:
                    _require(sf.get("real_3d_pdb_id") is None,
                             "/api/structure %s must NOT name a PDB it never fetched"
                             % who)
                # (1) aggregation_hotspots = REAL, from the shipped APR peptides — REAL
                # but sequence-level, so it must NOT claim to be 3D-derived.
                ah = sf["aggregation_hotspots"] or {}
                _require(ah.get("derivation") == "apr_real",
                         "/api/structure %s aggregation_hotspots must be REAL (apr_real)"
                         % who)
                _require(ah.get("is_3d_derived") is False,
                         "/api/structure %s aggregation_hotspots is REAL but SEQUENCE-"
                         "level — it must not claim is_3d_derived" % who)
                # (2) the 6 nested features
                for k in NESTED_3D_FEATURES:
                    blk = sf.get(k) or {}
                    # (2a) the PROXY sub-block is ALWAYS present and ALWAYS honest —
                    # a real 3D measurement must never overwrite or erase it.
                    px = blk.get("proxy")
                    _require(isinstance(px, dict),
                             "/api/structure %s %s must carry a nested proxy sub-block"
                             % (who, k))
                    _require(px.get("is_3d_derived") is False,
                             "/api/structure %s %s.proxy must be stamped "
                             "is_3d_derived=False" % (who, k))
                    _require(px.get("derivation") == "sequence_apr_proxy",
                             "/api/structure %s %s.proxy must be stamped "
                             "derivation=sequence_apr_proxy" % (who, k))
                    _require(bool(px.get("proxy_of")),
                             "/api/structure %s %s.proxy must state WHAT it is a proxy "
                             "of" % (who, k))
                    _require(bool(px.get("real_feature_requires")),
                             "/api/structure %s %s.proxy must name what the REAL 3D "
                             "feature would require" % (who, k))
                    # (2b) the per-feature flag must agree with the container stamp
                    _require(blk.get("has_real_3d") is want_real_3d,
                             "/api/structure %s %s.has_real_3d must be %r"
                             % (who, k, want_real_3d))
                    if want_real_3d:
                        # (2c) the REAL 3D sub-block: measured, stamped, and carrying
                        # its own fidelity caveat (exact / exact_algorithm / approx).
                        r3 = blk.get("real_3d")
                        _require(isinstance(r3, dict),
                                 "/api/structure %s %s must carry a real_3d sub-block "
                                 "when a PDB was parsed" % (who, k))
                        _require(r3.get("derivation") == "pdb_3d",
                                 "/api/structure %s %s.real_3d must be stamped "
                                 "derivation=pdb_3d" % (who, k))
                        _require(r3.get("is_3d_derived") is True,
                                 "/api/structure %s %s.real_3d must be stamped "
                                 "is_3d_derived=True" % (who, k))
                        _require(isinstance(r3.get("fidelity"), str)
                                 and bool(r3["fidelity"]),
                                 "/api/structure %s %s.real_3d must carry a fidelity "
                                 "caveat" % (who, k))
                        _require(r3.get("value") is not None,
                                 "/api/structure %s %s.real_3d claims a measurement but "
                                 "carries no value" % (who, k))
                        _require(bool(r3.get("real_from")),
                                 "/api/structure %s %s.real_3d must name the real "
                                 "computation it came from" % (who, k))
                    else:
                        # (2d) THE anti-fabrication invariant: no PDB -> no real_3d.
                        _require("real_3d" not in blk,
                                 "/api/structure %s %s must NOT invent a real_3d block "
                                 "with no fetched PDB" % (who, k))
                # (3) contact_map is intrinsically 3D: REAL when a PDB parsed, else an
                # honest computable:false — it must NEVER acquire a 1-D stand-in.
                cm = sf.get("contact_map") or {}
                _require(cm.get("is_3d_derived") is True,
                         "/api/structure %s contact_map is intrinsically 3D" % who)
                _require("proxy" not in cm,
                         "/api/structure %s contact_map must NOT gain a 1-D proxy" % who)
                _require(cm.get("has_real_3d") is want_real_3d,
                         "/api/structure %s contact_map.has_real_3d must be %r"
                         % (who, want_real_3d))
                if want_real_3d:
                    _require(cm.get("computable") is True
                             and cm.get("derivation") == "pdb_3d",
                             "/api/structure %s contact_map must be REAL (pdb_3d)" % who)
                    _require(cm.get("fidelity") == "exact",
                             "/api/structure %s contact_map fidelity must be exact"
                             % who)
                    _require((cm.get("value") or {}).get("n_contacts") is not None,
                             "/api/structure %s REAL contact_map must report n_contacts"
                             % who)
                else:
                    _require(cm.get("computable") is False,
                             "/api/structure %s contact_map must be an honest "
                             "not-computable stub" % who)
                    _require(cm.get("value") is None,
                             "/api/structure %s contact_map must NOT report a value "
                             "with no PDB" % who)
                    _require(bool(cm.get("real_feature_requires")),
                             "/api/structure %s deferred contact_map must say what it "
                             "needs" % who)

            def _has_real_3d(pid):
                """True/False if this protein is in the M16 set, else None."""
                uni = server._resolve_uniprot(pid)
                rec = server.STORE.structure_by_uniprot.get(uni) if uni else None
                if not rec:
                    return None
                return bool((rec.get("structural_features") or {})
                            .get("has_real_3d_features"))

            struct_protein = ("Alpha-Synuclein"
                              if server.STORE.series_by_protein.get("Alpha-Synuclein")
                              else first_pid)
            st, sx = _get(base, "/api/structure/" + urllib.parse.quote(struct_protein))
            _require(st == 200, "/api/structure status %s" % st)
            if sx.get("available"):
                for k in ("uniprot", "apr_proxy_features", "structural_features",
                          "graph_summary", "structure_sources", "association_table",
                          "variance_ceiling", "honesty"):
                    _require(k in sx, "/api/structure missing key %r" % k)
                # (c) APR/sequence PROXY features present + stamped not-3D
                apf = sx["apr_proxy_features"]
                for k in ("beta_propensity_mean", "hydrophobicity_mean",
                          "net_charge_total", "aggregation_hotspots"):
                    _require(k in apf, "/api/structure apr_proxy_features missing %r" % k)
                _require(apf.get("is_3d_derived") is False,
                         "/api/structure proxy features must be stamped is_3d_derived=False")
                # (b) aggregation hotspots = the REAL structural feature
                _require(isinstance(apf.get("aggregation_hotspots"), list)
                         and len(apf["aggregation_hotspots"]) >= 1,
                         "/api/structure has no APR aggregation hotspots")
                h0 = apf["aggregation_hotspots"][0]
                for k in ("position", "category", "region"):
                    _require(k in h0, "/api/structure hotspot missing %r" % k)
                # (d) the 8 structural features, each with an honest status. Assert the
                # FULL nested contract against whichever branch this protein is in.
                _check_struct_schema(sx, _has_real_3d(struct_protein) is True,
                                     struct_protein)
                # (e) GNN graph summary: nodes/edges + contact edges unavailable
                g = sx["graph_summary"]
                for k in ("n_nodes", "n_edges", "native_or_fibril",
                          "contact_edges_available"):
                    _require(k in g, "/api/structure graph_summary missing %r" % k)
                # (g) the corpus structure->kinetics association table
                at = sx["association_table"]
                _require(isinstance(at.get("association_cards"), list)
                         and len(at["association_cards"]) >= 1,
                         "/api/structure has no association cards")
                # EVERY card is association-only, never causal + carries a verdict
                for c in at["association_cards"]:
                    _require(c.get("causal") is False,
                             "/api/structure association card must be causal:false")
                    _require(c.get("verdict") in
                             ("association", "insufficient_power", "null"),
                             "/api/structure card bad verdict %r" % c.get("verdict"))
                    for k in ("feature", "target", "spearman_rho", "permutation_p",
                              "fdr_q", "n_proteins"):
                        _require(k in c, "/api/structure card missing %r" % k)
                # the all-null / n-limited read: 0 survive FDR at n<=26
                _require(at.get("n_significant_after_fdr") == 0,
                         "/api/structure expected 0 associations surviving FDR (all null)")
                _require((at.get("coverage") or {}).get(
                         "effective_n_is_proteins_not_curves") is True,
                         "/api/structure coverage must flag effective-n = proteins")
                # variance ceiling: sequence-only predictor is a per-protein constant
                _require(sx["variance_ceiling"].get("between_protein_fraction") is not None,
                         "/api/structure missing variance ceiling")
                # (h) honesty footer: association-only/never-causal + proxy-not-3D visible
                honesty_txt = " ".join(sx["honesty"]) if isinstance(sx["honesty"], list) \
                    else str(sx["honesty"])
                _require("NEVER causal" in honesty_txt or "never causal" in honesty_txt,
                         "/api/structure honesty must state association-only/never-causal")
                _require("PROXY" in honesty_txt or "is_3d_derived" in honesty_txt,
                         "/api/structure honesty must state proxy-not-3D")
            else:
                _require("note" in sx or "reason" in sx,
                         "/api/structure unavailable must carry a note")

            # (d2) BOTH BRANCHES of the nested schema must hold over the LIVE API, not
            # just whichever branch the headline protein happens to be in. Find one
            # corpus protein WITH a fetched PDB and one WITHOUT, and assert the full
            # contract on each: the with-PDB one must expose real_3d + fidelity; the
            # without-PDB one must expose ONLY proxies and must not fabricate a real_3d
            # block or a contact map anywhere.
            _with_3d = None
            _without_3d = None
            for _pid in server.STORE.series_by_protein:
                _flag = _has_real_3d(_pid)
                if _flag is True and _with_3d is None:
                    _with_3d = _pid
                elif _flag is False and _without_3d is None:
                    _without_3d = _pid
                if _with_3d and _without_3d:
                    break
            _require(_with_3d is not None and _without_3d is not None,
                     "/api/structure test needs one protein WITH real 3D and one "
                     "WITHOUT to cover both branches (found with=%r without=%r)"
                     % (_with_3d, _without_3d))
            for _pid, _want in ((_with_3d, True), (_without_3d, False)):
                _stb, _sxb = _get(base, "/api/structure/" + urllib.parse.quote(_pid))
                _require(_stb == 200, "/api/structure %s status %s" % (_pid, _stb))
                _require(_sxb.get("available") is True,
                         "/api/structure %s should be available" % _pid)
                _check_struct_schema(_sxb, _want, _pid)
            # and the two branches must be genuinely DIFFERENT — a renderer that showed
            # them identically would be hiding the real-vs-proxy distinction.
            _s_with = _get(base, "/api/structure/" +
                           urllib.parse.quote(_with_3d))[1]["structural_features"]
            _s_without = _get(base, "/api/structure/" +
                              urllib.parse.quote(_without_3d))[1]["structural_features"]
            _require(_s_with["beta_sheet_content"].get("real_3d", {}).get("value")
                     is not None
                     and "real_3d" not in _s_without["beta_sheet_content"],
                     "/api/structure the real-3D and proxy-only branches must be "
                     "distinguishable in the payload")

            # a protein with NO APR features degrades to {available: False} + honest note,
            # still surfacing the CORPUS association table (never a 500).
            st0, sx0 = _get(base, "/api/structure/" +
                            urllib.parse.quote("NotARealProtein_zzz"))
            _require(st0 == 200, "/api/structure absent protein should 200, got %s" % st0)
            _require(sx0.get("available") is False,
                     "/api/structure absent protein should be unavailable")
            _require("note" in sx0,
                     "/api/structure absent protein must carry an honest note")
            _require("association_table" in sx0,
                     "/api/structure absent protein must still surface the corpus table")

        # --- POST /api/cohort (M0): multi-dataset assembly + routing ---
        # (a) a REAL concentration-series member set -> concentration_series + dual-γ.
        # Prefer the SMALLEST eligible series (>=3 present members spanning >=3 DISTINCT
        # concentrations) so the interactive POST stays fast AND routes to conc-series.
        def _distinct_conc(ids):
            cs = set()
            for i in ids:
                c = server.STORE.curve_by_series.get(i, {})
                v = ((c.get("condition_vector") or {}).get("concentration") or {}).get("value_uM")
                if v is not None:
                    cs.add(round(float(v), 4))
            return cs
        cs_meta = None
        for csid, meta in sorted(
                server.STORE.conc_series_by_id.items(),
                key=lambda kv: len(kv[1].get("member_series_ids") or [])):
            ids = meta.get("member_series_ids") or []
            present = [i for i in ids if i in server.STORE.curve_by_series]
            if len(present) >= 3 and len(_distinct_conc(present)) >= 3:
                cs_meta = (csid, present)
                break
        _require(cs_meta is not None, "no concentration series found for /api/cohort test")
        st, coh = _post(base, "/api/cohort", {"series_ids": cs_meta[1]}, timeout=120)
        _require(st == 200, "/api/cohort (conc series) status %s" % st)
        if server.ENGINE_OK:
            _require("error" not in coh, "/api/cohort conc error: %s" % coh.get("error"))
            for k in ("data_mode", "comparability", "members", "assembled_analysis",
                      "window_of_validity", "version"):
                _require(k in coh, "/api/cohort missing key %r" % k)
            _require(coh["data_mode"] == "concentration_series",
                     "/api/cohort real conc series should be concentration_series, got %s"
                     % coh["data_mode"])
            _require(coh["comparability"].get("ok") is True,
                     "/api/cohort real conc series should be comparable")
            a = coh["assembled_analysis"]
            _require(a.get("route") == "concentration_series",
                     "/api/cohort route should be concentration_series")
            _require("dual_gamma" in a or "gamma_regression" in a,
                     "/api/cohort concentration_series must carry dual-γ")
            wov = coh["window_of_validity"].get("concentration_uM") or {}
            _require(wov.get("min") is not None,
                     "/api/cohort must report a concentration window of validity")

        # (b) a MIXED set (different construct) -> confounded / comparability mismatch
        from collections import defaultdict as _dd
        byprot = _dd(dict)
        mixed = None
        for sid, c in server.STORE.curve_by_series.items():
            cv = c.get("condition_vector") or {}
            uni = c.get("uniprot_id") or c.get("protein_id")
            con = cv.get("construct_id")
            if con and con not in byprot[uni]:
                byprot[uni][con] = sid
            if len(byprot[uni]) >= 2:
                mixed = list(byprot[uni].values())[:2]
                break
        _require(mixed is not None, "no construct-mismatch pair for /api/cohort test")
        st, coh2 = _post(base, "/api/cohort", {"series_ids": mixed})
        _require(st == 200, "/api/cohort (mixed) status %s" % st)
        if server.ENGINE_OK:
            comp = coh2.get("comparability", {})
            _require(comp.get("ok") is False,
                     "/api/cohort mixed construct must fail comparability")
            _require("construct_id" in (comp.get("mismatched_fields") or []),
                     "/api/cohort must flag construct_id mismatch")
            _require(coh2.get("data_mode") == "confounded",
                     "/api/cohort mixed construct -> confounded")
            _require(coh2["assembled_analysis"].get("mechanism_refused") is True,
                     "/api/cohort confound must refuse mechanism")

        # (c) empty selection -> flagged error, never 500
        st, coh3 = _post(base, "/api/cohort", {"series_ids": []})
        _require(st == 200, "/api/cohort empty should be 200 JSON, got %s" % st)
        _require("error" in coh3, "/api/cohort empty selection must carry an error")

        # --- POST /api/analyze with a synthetic logistic sigmoid ---
        import math
        xs = [i * 4.0 for i in range(24)]
        ys = [1.0 / (1.0 + math.exp(-0.18 * (t - 42.0))) for t in xs]
        st, an = _post(base, "/api/analyze",
                       {"x": xs, "y": ys, "concentration_uM": 10, "assay": "ThT"})
        _require(st == 200, "/api/analyze status %s" % st)
        if server.ENGINE_OK:
            _require(an.get("ok") is True,
                     "/api/analyze not ok: %s" % an.get("error"))
            for k in ("regime", "information_yield", "m1", "mechanistic",
                      "features", "engine_build_id"):
                _require(k in an, "/api/analyze missing key %r" % k)
            # honesty: mechanism must NOT be licensed on a single user curve
            _require(an["mechanistic"].get("licensed") in (False, None),
                     "single-curve analyze should not license mechanism")
            # conformal: when the frozen calibration is loaded, /api/analyze surfaces
            # a finite-sample-valid t50 interval + a regime prediction set
            _require("conformal" in an, "/api/analyze missing conformal block")
            cf = an["conformal"]
            if cf.get("available"):
                iv = cf.get("t50_interval") or {}
                rs = cf.get("regime_set") or {}
                if iv.get("available"):
                    _require(iv.get("lo") is not None,
                             "/api/analyze conformal t50 interval missing lo")
                    _require(iv.get("hi") is not None or iv.get("hi_is_infinite"),
                             "/api/analyze conformal t50 interval missing hi")
                _require(rs.get("available") and rs.get("set_size", 0) >= 1,
                         "/api/analyze conformal regime set empty")
        else:
            # documented graceful degradation when the engine can't import
            _require("error" in an,
                     "/api/analyze should return an error when engine unavailable")

    except AssertionError as exc:
        failures.append(str(exc))
    finally:
        httpd.shutdown()

    if failures:
        for f in failures:
            sys.stderr.write("FAIL: %s\n" % f)
        print("test_web: FAILED (%d)" % len(failures))
        return 1
    print("test_web: OK — all endpoints passed (engine_ok=%s)" % server.ENGINE_OK)
    return 0


if __name__ == "__main__":
    sys.exit(main())
