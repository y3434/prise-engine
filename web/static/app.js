/* PRISE web app — vanilla single-page app (no framework, no build step).
   Three tabs: Dashboard, Browse proteins, Analyze a curve.
   Plotly is used when available; everything degrades to an SVG/table fallback
   so the page is never blank without internet. */

"use strict";

// --------------------------------------------------------------------------- //
// tiny helpers
// --------------------------------------------------------------------------- //
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function fmt(v, digits = 3) {
  if (v == null || v === "") return "—";
  if (typeof v === "number") {
    if (!isFinite(v)) return "—";
    if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0)) return v.toExponential(2);
    return Number(v.toFixed(digits)).toString();
  }
  return String(v);
}

function pct(frac) {
  if (frac == null) return "—";
  return (frac * 100).toFixed(1) + "%";
}

async function api(path) {
  const r = await fetch(path);
  const j = await r.json();
  if (!r.ok && j && j.error) throw new Error(j.error);
  return j;
}

function plotlyAvailable() {
  return typeof window.Plotly !== "undefined" && !window.__PLOTLY_FAILED__;
}

let GLOBAL_ATTRIBUTION = null;

// --------------------------------------------------------------------------- //
// tab switching
// --------------------------------------------------------------------------- //
function initTabs() {
  $$(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab;
      $$(".view").forEach((v) => v.classList.add("hidden"));
      $("#view-" + tab).classList.remove("hidden");
      if (tab === "browse" && !browseLoaded) loadBrowse();
      if (tab === "dashboard" && !dashboardLoaded) loadDashboard();
      if (tab === "crossmodal" && !crossmodalLoaded) loadCrossmodal();
    });
  });
}

// --------------------------------------------------------------------------- //
// Plot helper: Plotly line+marker plot with SVG/table fallback
// --------------------------------------------------------------------------- //
// palette for overlaying multiple fitted model lines on one plot
const OVERLAY_COLORS = [
  "#5ec6a8", "#e0a458", "#c6a3ff", "#e06c75", "#6cc070", "#4ea1d3",
  "#d98ec0", "#9fb0c0",
];

function renderCurvePlot(container, opts) {
  // opts: { x, y, fitted:{x,y,model}, overlays:[{x,y,model,color}], title,
  //         censoring, xlabel, ylabel }
  container.innerHTML = "";
  const { x, y, fitted, overlays, title, censoring, xlabel, ylabel } = opts;
  // unify single `fitted` and multi `overlays` into one list of fitted lines
  const fitLines = [];
  if (fitted && fitted.x && fitted.y) fitLines.push(fitted);
  (overlays || []).forEach((o) => { if (o && o.x && o.y) fitLines.push(o); });

  if (plotlyAvailable()) {
    try {
      const traces = [{
        x, y, mode: "markers", type: "scatter", name: "observed",
        marker: { size: 7, color: "#4ea1d3" },
      }];
      fitLines.forEach((f, i) => {
        traces.push({
          x: f.x, y: f.y, mode: "lines", type: "scatter",
          name: "fit: " + (f.model || "model"),
          line: { color: f.color || OVERLAY_COLORS[i % OVERLAY_COLORS.length], width: 2 },
        });
      });
      const shapes = [];
      // shade censored region
      if (censoring && censoring.indexOf("right") >= 0 && x.length) {
        shapes.push(censorShape(Math.max(...x), Math.max(...x) * 1.0, x, "right"));
      }
      const layout = {
        title: { text: title || "", font: { size: 13, color: "#e6edf3" } },
        paper_bgcolor: "#1a212b", plot_bgcolor: "#11171f",
        font: { color: "#9fb0c0", size: 12 },
        margin: { l: 50, r: 16, t: title ? 34 : 12, b: 42 },
        xaxis: { title: xlabel || "time (hours)", gridcolor: "#2d3a48", zerolinecolor: "#2d3a48" },
        yaxis: { title: ylabel || "signal", gridcolor: "#2d3a48", zerolinecolor: "#2d3a48" },
        legend: { orientation: "h", y: -0.22, font: { size: 11 } },
        showlegend: true,
      };
      Plotly.newPlot(container, traces, layout, { displayModeBar: false, responsive: true });
      return;
    } catch (e) {
      // fall through to fallback
    }
  }
  renderCurveFallback(container, opts);
}

function censorShape() { return {}; } // placeholder (Plotly path uses inline shapes only if needed)

function renderCurveFallback(container, opts) {
  const { x, y, fitted, overlays, censoring, xlabel, ylabel } = opts;
  const wrap = el("div", { class: "plot-fallback" });
  const fitLines = [];
  if (fitted && fitted.x && fitted.y) fitLines.push(fitted);
  (overlays || []).forEach((o) => { if (o && o.x && o.y) fitLines.push(o); });
  // SVG scatter + fit line(s)
  const W = 520, H = 280, pad = 44;
  const xs = x.slice(), ys = y.slice();
  const allX = xs.slice(), allY = ys.slice();
  fitLines.forEach((f) => { allX.push(...f.x); allY.push(...f.y); });
  const xmin = Math.min(...allX), xmax = Math.max(...allX);
  const ymin = Math.min(...allY), ymax = Math.max(...allY);
  const sx = (v) => pad + ((v - xmin) / ((xmax - xmin) || 1)) * (W - 2 * pad);
  const sy = (v) => H - pad - ((v - ymin) / ((ymax - ymin) || 1)) * (H - 2 * pad);

  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="curve plot">`;
  svg += `<rect x="0" y="0" width="${W}" height="${H}" fill="#11171f"/>`;
  // axes
  svg += `<line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="#2d3a48"/>`;
  svg += `<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}" stroke="#2d3a48"/>`;
  // fit line(s)
  fitLines.forEach((f, k) => {
    const col = f.color || OVERLAY_COLORS[k % OVERLAY_COLORS.length];
    let d = "";
    for (let i = 0; i < f.x.length; i++) {
      d += (i === 0 ? "M" : "L") + sx(f.x[i]).toFixed(1) + " " + sy(f.y[i]).toFixed(1) + " ";
    }
    svg += `<path d="${d}" fill="none" stroke="${col}" stroke-width="2"/>`;
  });
  // observed points (mark censored points distinctly if flagged on the point)
  for (let i = 0; i < xs.length; i++) {
    const censored = ys.__censored__ && ys.__censored__[i];
    if (censored) {
      svg += `<rect x="${(sx(xs[i]) - 4).toFixed(1)}" y="${(sy(ys[i]) - 4).toFixed(1)}" width="8" height="8" fill="none" stroke="#e0a458" stroke-width="1.5"/>`;
    } else {
      svg += `<circle cx="${sx(xs[i]).toFixed(1)}" cy="${sy(ys[i]).toFixed(1)}" r="3.5" fill="#4ea1d3"/>`;
    }
  }
  // labels
  svg += `<text x="${W / 2}" y="${H - 8}" fill="#6b7d8f" font-size="11" text-anchor="middle">${esc(xlabel || "time (hours)")}</text>`;
  svg += `<text x="12" y="${H / 2}" fill="#6b7d8f" font-size="11" text-anchor="middle" transform="rotate(-90 12 ${H / 2})">${esc(ylabel || "signal")}</text>`;
  svg += `</svg>`;
  wrap.innerHTML = svg;

  const note = el("div", { class: "plot-note" },
    "Plotly CDN unavailable — showing an offline SVG render" +
    (fitLines.length ? " (" + fitLines.map((f) => esc(f.model)).join(", ") + ")" : "") +
    (censoring && censoring !== "none" ? " · censoring: " + esc(censoring) : ""));
  wrap.appendChild(note);

  // also a compact data table (first/last rows) for full transparency
  const tbl = el("details", {}, [el("summary", { class: "muted" }, "show data table")]);
  const table = el("table", { class: "data-table" });
  table.appendChild(el("tr", {}, [el("th", {}, "t (h)"), el("th", {}, "signal")]));
  for (let i = 0; i < xs.length; i++) {
    table.appendChild(el("tr", {}, [
      el("td", { class: "num" }, fmt(xs[i])),
      el("td", { class: "num" }, fmt(ys[i])),
    ]));
  }
  tbl.appendChild(table);
  wrap.appendChild(tbl);
  container.appendChild(wrap);
}

// --------------------------------------------------------------------------- //
// DASHBOARD
// --------------------------------------------------------------------------- //
let dashboardLoaded = false;

async function loadDashboard() {
  dashboardLoaded = true;
  const root = $("#dashboard-content");
  try {
    const d = await api("/api/overview");
    GLOBAL_ATTRIBUTION = d.attribution;
    renderFooter(d.attribution);
    $("#build-badge").textContent = "build " + (d.engine_build_id || "?");
    root.classList.remove("loading");
    root.innerHTML = "";

    // 0. Plain-language intro: what this view IS (and is NOT)
    const nCurves = (d.inference_ladder || {}).n_curves || d.n_curves || "all";
    const nProteins = (d.inference_ladder || {}).n_proteins || "—";
    root.appendChild(el("div", { class: "card overview-intro" }, [
      el("h2", {}, "Corpus overview — what PRISE can and cannot infer across the whole CPAD 2.0 corpus"),
      el("p", { class: "muted" }, [
        document.createTextNode("This is a "),
        el("strong", {}, "corpus-wide aggregate"),
        document.createTextNode(" over all " + nCurves + " curves (" + nProteins +
          " proteins) — the sum of the same per-dataset analyses you explore under "),
        el("strong", {}, "Browse & analyze datasets"),
        document.createTextNode(". It is "),
        el("strong", {}, "not"),
        document.createTextNode(" the analysis of any one dataset, and "),
        el("strong", {}, "not"),
        document.createTextNode(" invented numbers: every count below is computed by the " +
          "engine from the individual curve results. To see one concrete dataset's analysis, " +
          "open a protein under Browse and click one of its curves."),
      ]),
    ]));

    // 1. Headline GAP card
    const gap = d.boundary_gap || {};
    root.appendChild(el("div", { class: "card headline-gap" }, [
      el("h3", {}, "The headline gap"),
      el("div", { class: "kpi-row", style: "align-items:center; gap:2rem;" }, [
        el("div", { class: "kpi" }, [
          el("div", { class: "n", style: "color:var(--accent)" }, String(gap.structural_reachable_proteins ?? "—")),
          el("div", { class: "l" }, "structurally resolvable"),
        ]),
        el("div", { class: "kpi" }, [
          el("div", { class: "n", style: "color:var(--danger)" }, String(gap.actual_mechanistically_licensed ?? "—")),
          el("div", { class: "l" }, "actually licensed"),
        ]),
        el("div", { class: "kpi" }, [
          el("div", { class: "gap-number" }, "GAP " + (gap.gap ?? "—")),
          el("div", { class: "l" }, "closed by ONE metadata fix"),
        ]),
      ]),
      el("p", { class: "gap-sub" }, esc(gap.headline || "")),
    ]));

    // 2. Validity-ceiling / honesty banner
    const vc = d.validity_ceiling || {};
    root.appendChild(el("div", { class: "honesty-banner" }, [
      el("h3", {}, "Validity ceiling — reachability UNDER the PRISE forward model"),
      el("p", { class: "clause" }, esc(vc.clause || "")),
      el("p", { class: "clause" },
        "Mechanism is licensed on 0 curves — because agitation & seeding are " +
        "unrecorded corpus-wide, the primary/secondary-nucleation branch is " +
        "undetermined. A different (e.g. KMC/Smoluchowski) forward model degrades " +
        "mechanism-identification accuracy by ~" +
        fmt(vc.mechanism_misidentification_degradation, 2) +
        " (Service C C1) — these counts are upper bounds on reachability, not the field's data."),
    ]));

    // 3. Inference-ladder funnel
    const ladder = d.inference_ladder || {};
    const rungs = ladder.rungs || [];
    const maxC = Math.max(...rungs.map((r) => r.curves || 0), 1);
    const funnel = el("div", { class: "funnel" });
    rungs.forEach((r) => {
      const w = ((r.curves || 0) / maxC) * 100;
      funnel.appendChild(el("div", { class: "funnel-row" }, [
        el("div", { class: "funnel-label" }, esc(r.rung)),
        el("div", { class: "funnel-bar-track" }, [
          el("div", { class: "funnel-bar" + (r.curves === 0 ? " zero" : ""),
            style: "width:" + Math.max(w, r.curves === 0 ? 8 : 1) + "%" },
            [el("span", { class: "funnel-val", style: "padding:0 .5rem" }, String(r.curves))]),
        ]),
        el("div", { class: "funnel-frac" }, pct(r.curve_fraction)),
      ]));
    });
    root.appendChild(el("div", { class: "card" }, [
      el("h2", {}, "Inference ladder"),
      el("p", { class: "explainer" }, "What it means: of " + (ladder.n_curves || "?") +
        " curves, how many reach each successive level of analysis. Each rung is a stricter " +
        "evidentiary bar, so the count can only stay the same or fall (monotone non-increasing)."),
      el("p", { class: "muted" }, (ladder.n_curves || "?") + " curves · " + (ladder.n_proteins || "?") +
        " proteins."),
      funnel,
    ]));

    // 4. two-column: yield tiers + blocker census
    const grid = el("div", { class: "grid cols-2" });

    // yield tiers
    grid.appendChild(el("div", { class: "card" }, [
      el("h2", {}, "Information-yield tiers"),
      el("p", { class: "explainer" }, "What it means: how far each curve's evidence reaches — " +
        "mechanistic (a mechanism call is licensed) › scaling (a γ concentration-exponent " +
        "resolves) › descriptive (curve shape only) › signal_only (too sparse/noisy to fit) › " +
        "uninformative. Higher tiers are rarer; this is the corpus split."),
      buildBarList(d.yield_tiers, d.n_curves, "tier-"),
      el("p", { class: "section-note" }, "scaling requires a γ whose 95% CI excludes 0: " +
        fmt(d.gamma_counts.n_with_reliable_gamma_fit, 0) + " reliable γ fits, but only " +
        fmt(d.gamma_counts.n_with_significant_gamma, 0) + " resolved (CI excludes 0)."),
    ]));

    // blocker census
    const blk = d.blocker_census || {};
    const blkList = el("div", { class: "barlist" });
    (blk.top || []).forEach((b) => {
      blkList.appendChild(el("div", { class: "barlist-wrap" }, [
        el("div", {}, [
          el("div", { class: "barlist-label" }, [
            el("span", {}, esc(b.blocker)),
            el("span", { class: "badge danger" }, "blocks " + esc(b.blocks_rung || "")),
          ]),
          el("div", { class: "faint", style: "font-size:.74rem" }, esc(b.detail || "")),
        ]),
        el("div", {}, [
          el("div", { class: "barlist-track" }, [
            el("div", { class: "barlist-fill blk", style: "width:" + ((b.fraction || 0) * 100) + "%" }),
          ]),
          el("div", { class: "barlist-count" }, fmt(b.count, 0) + " · " + pct(b.fraction)),
        ]),
      ]));
    });
    grid.appendChild(el("div", { class: "card" }, [
      el("h2", {}, "Blocker census (impact-ranked)"),
      el("p", { class: "explainer" }, "What it means: the missing metadata or data limits that " +
        "stop curves from climbing to a higher rung, ranked by how many curves each one holds back. " +
        "These are the levers — fix the top blocker and the most curves advance."),
      blkList,
    ]));
    root.appendChild(grid);

    // 5. top corpus action (M9)
    const m9 = d.m9 || {};
    const ta = m9.top_action || {};
    root.appendChild(el("div", { class: "card" }, [
      el("h2", {}, "Highest-leverage next experiment (M9)"),
      el("p", { class: "explainer" }, "What it means: mechanism is licensed on 0 curves because no " +
        "curve in the corpus records the metadata (agitation / seeding) needed to license a " +
        "mechanism call — so PRISE reports the single experiment that would unlock the most curves, " +
        "rather than guessing. Open any protein in Browse to see its specific failed gates."),
      el("div", { class: "rec top" }, [
        el("div", { class: "rec-head" }, [
          el("span", { class: "rec-exp" }, esc(m9.top_corpus_action || "—")),
          el("span", { class: "badge ok" }, "cost: " + esc(ta.cost_tier || "?")),
          el("span", { class: "badge" }, "reduction ×" + fmt(ta.expected_degeneracy_reduction, 1)),
          el("span", { class: "badge" }, "top-ranked for " + fmt(ta.n_units_top_ranked, 0) + " units"),
        ]),
        el("div", { class: "rec-note" }, esc(m9.headline || "")),
      ]),
    ]));

    // 5b. Reality check (external validation) — blind §6C falsification test
    await renderRealityCheck(root);

    // 5c. Why inference fails (metadata) — M11 corpus metadata-quality card
    await renderMetadataOverview(root);

    // 5d. Literature ingestion — extracted claims (M15): claims (not facts) with
    // full provenance + measured confidence, cross-paper conflicts (unresolved),
    // and the CPAD-validation accuracy.
    await renderLiterature(root);

    // 6. corpus caveats
    const cav = el("ul", { class: "caveat-list" });
    (d.corpus_caveats || []).forEach((c) => cav.appendChild(el("li", {}, esc(c))));
    root.appendChild(el("div", { class: "card" }, [
      el("h2", {}, "Corpus-wide caveats"),
      cav,
    ]));

  } catch (e) {
    root.classList.remove("loading");
    root.innerHTML = "";
    root.appendChild(el("div", { class: "error-banner" }, "Failed to load dashboard: " + esc(e.message)));
  }
}

function buildBarList(counts, total, prefix) {
  const list = el("div", { class: "barlist" });
  const max = Math.max(...Object.values(counts || {}).map((v) => v || 0), 1);
  Object.entries(counts || {}).forEach(([k, v]) => {
    list.appendChild(el("div", { class: "barlist-wrap" }, [
      el("div", { class: "barlist-label" }, [
        el("span", { class: "badge " + prefix + k }, esc(k)),
      ]),
      el("div", {}, [
        el("div", { class: "barlist-track" }, [
          el("div", { class: "barlist-fill " + prefix + k, style: "width:" + ((v / max) * 100) + "%" }),
        ]),
        el("div", { class: "barlist-count" }, fmt(v, 0)),
      ]),
    ]));
  });
  return list;
}

// --------------------------------------------------------------------------- //
// Reality check (external validation) — the blind §6C falsification test.
// Fixed-points table (protein · consensus · PRISE regime/γ/equiv-class · verdict),
// the overall verdict, and the honest framing (few fixed points, non-contradiction
// test, reserved proteins). Degrades gracefully if the artifact was not built.
// --------------------------------------------------------------------------- //
const _VERDICT_BADGE = {
  CONSISTENT: "ok", CONTRADICTION: "danger", INSUFFICIENT_DATA: "",
};

async function renderRealityCheck(root) {
  let d;
  try {
    d = await api("/api/reality-check");
  } catch (e) {
    return;   // never break the dashboard on this optional panel
  }
  const card = el("div", { class: "card reality-check" });
  card.appendChild(el("h2", {}, "Reality check (external validation)"));

  if (!d || d.available === false) {
    card.appendChild(el("p", { class: "muted" },
      esc((d && d.reason) || "reality_check.json not built — run "
        + "`python engine/reality_check.py`.")));
    root.appendChild(card);
    return;
  }

  const res = d.reality_check_result || {};
  const overall = res.n_contradiction > 0 ? "danger"
    : (res.n_fixed_points_evaluated > 0 ? "ok" : "");

  card.appendChild(el("p", { class: "explainer" },
    "What it means: PRISE run BLIND on a handful of proteins the FIELD considers "
    + "mechanistically RESOLVED, its honest output (descriptive regime + γ + "
    + "mechanistic equivalence class) compared to published consensus. This is a "
    + "NON-CONTRADICTION test against a few fixed points — like a thermometer against "
    + "freezing/boiling points — NOT statistical calibration and NOT proof of "
    + "correctness. A CONTRADICTION on a strong-consensus case FALSIFIES the map."));

  // overall verdict banner
  card.appendChild(el("div", { class: "rc-overall " + overall }, [
    el("div", { class: "rc-tally" }, [
      el("span", { class: "badge ok" }, "CONSISTENT " + (res.n_consistent ?? "—")),
      el("span", { class: "badge danger" }, "CONTRADICTION " + (res.n_contradiction ?? "—")),
      el("span", { class: "badge" }, "INSUFFICIENT " + (res.n_insufficient ?? "—")),
      el("span", { class: "badge" }, (res.n_fixed_points_evaluated ?? "—") + " evaluated"),
    ]),
    el("p", { class: "rc-plain" }, esc(res.verdict_plain_english || "")),
  ]));

  // fixed-points table
  const rows = (d.fixed_points || []).map((p) => {
    const pr = p.prise || {};
    const cons = p.consensus || {};
    const gc = (p.checks || {}).gamma || {};
    const pg = (gc.prise_gamma != null) ? fmt(gc.prise_gamma, 2) : "—";
    const cg = (cons.gamma_consensus != null) ? fmt(cons.gamma_consensus, 2) : "n/a";
    const equiv = pr.equivalence_class
      ? pr.equivalence_class.length + "-member class"
      : "—";
    const consMember = equiv !== "—"
      && (pr.equivalence_class || []).indexOf(cons.mechanism) >= 0;
    return el("tr", {}, [
      el("td", {}, [
        el("strong", {}, esc(p.display_name || p.protein)),
        el("div", { class: "faint", style: "font-size:.72rem" },
          "consensus: " + esc(cons.consensus_strength || "")),
      ]),
      el("td", {}, [
        document.createTextNode(esc(cons.shape || "")),
        el("div", { class: "faint", style: "font-size:.72rem" },
          "mechanism: " + esc(cons.mechanism || "")),
      ]),
      el("td", {}, [
        el("span", { class: "badge tier-" + (pr.modal_regime || "") },
          esc(pr.modal_regime || "—")),
      ]),
      el("td", {}, "γ " + pg + " / " + cg
        + "  (" + esc(gc.status || "na") + ")"),
      el("td", {}, [
        document.createTextNode(esc(equiv)),
        el("div", { class: "faint", style: "font-size:.72rem" },
          consMember ? "consensus not excluded" : "—"),
      ]),
      el("td", {}, [
        el("span", { class: "badge " + (_VERDICT_BADGE[p.verdict] || "") },
          esc(p.verdict)),
      ]),
    ]);
  });

  const table = el("table", { class: "rc-table" }, [
    el("thead", {}, el("tr", {}, [
      el("th", {}, "Protein"),
      el("th", {}, "Field consensus"),
      el("th", {}, "PRISE regime"),
      el("th", {}, "γ (PRISE / consensus)"),
      el("th", {}, "Equivalence class"),
      el("th", {}, "Verdict"),
    ])),
    el("tbody", {}, rows),
  ]);
  card.appendChild(table);

  // honest framing + reserved-proteins discipline
  const lk = d.reserved_proteins_note || {};
  card.appendChild(el("p", { class: "section-note" },
    "Reserved-proteins discipline: these " + ((lk.reserved_proteins || []).length)
    + " proteins are reserved SOLELY for this §6C falsification role — NOT used for "
    + "anchoring / calibration / demonstrators. Leakage check: "
    + (lk.no_leakage ? "clean (no reserved name in the calibration data)."
      : "WARNING — a reserved name appears in the calibration data.")));
  card.appendChild(el("p", { class: "faint", style: "font-size:.74rem" },
    esc(d.honesty_note || "")));
  card.appendChild(el("p", { class: "faint", style: "font-size:.74rem" },
    "PRISE refuses to LICENSE a single mechanism corpus-wide (agitation/seeding "
    + "unknown) — that refusal is EXPECTED; the test is on regime + γ + "
    + "equivalence-class CONSISTENCY, never on a named mechanism."));

  root.appendChild(card);
}

// --------------------------------------------------------------------------- //
// WHY INFERENCE FAILS (METADATA) — M11 corpus metadata-quality card on the
// Corpus-overview tab. Median metadata_score, the "MECHANISM licensed 0/N —
// universal blocker {agitation, seeded}" explanation, and the top-3 corpus
// recommendations. NEVER breaks the dashboard (own try/catch).
// --------------------------------------------------------------------------- //
async function renderMetadataOverview(root) {
  let d;
  try {
    d = await api("/api/metadata-overview");
  } catch (e) {
    return;   // optional panel — never break the dashboard
  }
  const card = el("div", { class: "card metadata-overview" });
  card.appendChild(el("h2", {}, "Why inference fails (metadata) — M11"));

  if (!d || d.available === false) {
    card.appendChild(el("p", { class: "muted" },
      esc((d && d.reason) || "metadata_quality.json not built — run "
        + "`python engine/m11_metadata.py`.")));
    root.appendChild(card);
    return;
  }

  const md = (d.metadata_score || {}).distribution || {};
  const mc = (d.mechanistic_completeness || {}).distribution || {};
  const ub = d.universal_blocker || {};
  const nCurves = d.n_curves || "?";
  const nLicensed = (d.n_mechanism_licensed != null) ? d.n_mechanism_licensed : "?";

  card.appendChild(el("p", { class: "explainer" },
    "What it means: PRISE scores, per dataset, how much of the metadata inference "
    + "needs is actually PRESENT (completeness) and what its absence BLOCKS — then "
    + "rolls it up. The corpus records the conditions but NOT the mechanism-gating "
    + "fields, which is exactly why mechanism is licensed on 0 curves."));

  // median completeness vs mechanistic completeness — two labelled bars
  card.appendChild(el("div", { class: "mq-bars" }, [
    scoreBar("median metadata completeness (PRESENCE)", md.median, {}),
    scoreBar("median mechanistic completeness (M5 prereqs met)", mc.median, {}),
  ]));

  // the WHY-mechanism≈0 headline banner
  card.appendChild(el("div", { class: "mq-why-banner" }, [
    el("div", { class: "mq-why-head" }, [
      el("span", { class: "gap-number" }, "MECHANISM " + nLicensed + " / " + nCurves),
      el("span", { class: "l" }, "licensed — universal blocker {agitation, seeded}"),
    ]),
    el("p", { class: "clause" }, esc(ub.why_mechanism_licensed_is_zero
      || ("{agitation, seeded} are unknown on ~" + pct(ub.fraction)
        + " of curves corpus-wide, so the mechanism gate refuses for every curve."))),
  ]));

  // top-3 corpus recommendations (information gain ÷ effort)
  const recs = d.top_recommendations || [];
  if (recs.length) {
    const list = el("div", { class: "rec-list" });
    recs.forEach((r, i) => {
      const measured = r.action === "record_agitation_and_seeding"
        || r.action === "add_concentration_series";
      list.appendChild(el("div", { class: "rec" + (i === 0 ? " top" : "") }, [
        el("div", { class: "rec-head" }, [
          el("span", { class: "rec-exp" }, "#" + (r.rank || i + 1) + " " + esc(r.action)),
          el("span", { class: "badge " + (_MQ_IMPACT_BADGE[r.impact] || "ok") }, "impact: " + esc(r.impact || "?")),
          el("span", { class: "badge" }, "effort: " + esc(r.effort || "?")),
          el("span", { class: "badge" }, "gain/effort " + fmt(r.total_gain_over_effort, 1)),
          el("span", { class: "badge" }, fmt(r.curves_affected, 0) + " curves"),
          measured
            ? el("span", { class: "badge ok", title: "Service-C MEASURED γ 1→4 split" }, "measured")
            : el("span", { class: "badge", title: "structural upper bound (M8 ceiling)" }, "structural"),
        ]),
      ]));
    });
    card.appendChild(el("div", {}, [
      el("p", { class: "section-note" }, "Top corpus recommendations (information gain ÷ effort):"),
      list,
    ]));
  }

  // honesty footer — completeness ≠ correctness + structural-upper-bound
  card.appendChild(el("p", { class: "faint", style: "font-size:.74rem" },
    "Completeness ≠ correctness: metadata_score measures PRESENCE, never whether a "
    + "recorded value is correct. Information-gain numbers are STRUCTURAL upper bounds "
    + "under the dependency DAG (inheriting M8's validity ceiling), EXCEPT the "
    + "Service-C-measured ones (the γ-resolved 1→4 Tier-B split)."));

  root.appendChild(card);
}

// --------------------------------------------------------------------------- //
// LITERATURE INGESTION — EXTRACTED CLAIMS (M15) — Corpus-overview card.
// CORE FRAMING (honesty-first): these are CLAIMS, not trusted facts. Every claim
// carries full provenance (DOI · page · location · verbatim source_text · char-span)
// and a MEASURED confidence; cross-paper CONFLICTS are surfaced UNRESOLVED; the
// extraction accuracy is MEASURED against CPAD's curated gold standard. M15
// PROPOSES; a curator approves. NEVER breaks the dashboard (own try/catch).
// --------------------------------------------------------------------------- //
const _LIT_GROUP_BADGE = {
  conditions: "tier-descriptive",
  reported_results: "tier-scaling",
  semantic: "",
};

// confidence -> a 0..1 bar (label + track). confidence is MEASURED, not asserted.
function litConfidenceBar(conf) {
  const v = (typeof conf === "number" && isFinite(conf)) ? Math.max(0, Math.min(1, conf)) : 0;
  const hue = v >= 0.8 ? "high" : (v >= 0.6 ? "mid" : "low");
  return el("div", { class: "lit-conf" }, [
    el("div", { class: "lit-conf-track", title: "measured confidence (calibrated)" }, [
      el("div", { class: "lit-conf-fill " + hue, style: "width:" + (v * 100) + "%" }),
    ]),
    el("span", { class: "lit-conf-val" }, conf == null ? "—" : fmt(conf, 2)),
  ]);
}

// one claim row: field · value+unit · group badge · confidence bar · location_type
// + the VERBATIM source_text in quotes with the DOI·page·char-span provenance trail.
// A superseded claim (supersedes set) is dimmed + struck to show append-only history.
function litClaimRow(c) {
  const superseded = !!c.supersedes;
  const valueText = (c.value == null ? "—" : String(c.value))
    + (c.unit ? " " + c.unit : "");
  const span = Array.isArray(c.char_span) ? ("[" + c.char_span[0] + "–" + c.char_span[1] + "]") : "";
  // provenance trail: DOI · page · location_type · char-span (the full traceability)
  const provParts = [];
  if (c.doi) provParts.push("DOI " + c.doi);
  if (c.pmid) provParts.push("PMID " + c.pmid);
  if (c.page != null) provParts.push("p." + c.page);
  if (c.block_id) provParts.push(c.block_id);
  if (span) provParts.push("char " + span);

  const flags = Array.isArray(c.flags) ? c.flags : [];
  const plausible = (c.plausibility || {}).plausible;

  const head = el("div", { class: "lit-claim-head" }, [
    el("span", { class: "lit-claim-field" }, esc(c.field || "—")),
    el("span", { class: "lit-claim-value" }, esc(valueText)),
    el("span", { class: "badge " + (_LIT_GROUP_BADGE[c.group] || "") }, esc(c.group || "—")),
    el("span", { class: "badge muted-badge" }, esc(c.location_type || "—")),
    plausible === false
      ? el("span", { class: "badge danger", title: "value flagged implausible — down-weighted" }, "implausible")
      : null,
    superseded
      ? el("span", { class: "badge warn", title: "superseded by a later appended claim" }, "superseded")
      : null,
  ]);
  flags.forEach((f) => head.appendChild(
    el("span", { class: "badge warn" }, esc(String(f)))));

  return el("div", { class: "lit-claim" + (superseded ? " superseded" : "") }, [
    el("div", { class: "lit-claim-row" }, [
      head,
      litConfidenceBar(c.confidence),
    ]),
    // the VERBATIM source_text in quotes — the load-bearing traceability
    el("blockquote", { class: "lit-verbatim", title: "verbatim source text (never paraphrased)" },
      "“" + esc(c.source_text || "") + "”"),
    el("div", { class: "lit-prov" }, esc(provParts.join(" · "))),
  ]);
}

// one conflict card: field/protein/condition + the divergent values with BOTH
// papers' provenance, marked UNRESOLVED (never auto-resolved).
function litConflictCard(cf) {
  const ctx = cf.condition_context || {};
  const ctxParts = [];
  if (ctx.pH != null) ctxParts.push("pH " + fmt(ctx.pH, 2));
  if (ctx.temperature != null) ctxParts.push(fmt(ctx.temperature, 1) + "°C");
  const rows = (cf.claims || []).map((cl) => el("div", { class: "lit-conflict-side" }, [
    el("div", { class: "lit-conflict-val" }, [
      el("span", { class: "lit-conflict-num" },
        (cl.value == null ? "—" : String(cl.value)) + (cl.unit ? " " + cl.unit : "")),
      el("span", { class: "badge muted-badge" }, esc(cl.location_type || "—")),
    ]),
    el("blockquote", { class: "lit-verbatim" }, "“" + esc(cl.source_text || "") + "”"),
    el("div", { class: "lit-prov" },
      esc((cl.doi ? "DOI " + cl.doi : (cl.pmid ? "PMID " + cl.pmid : "?"))
        + (cl.page != null ? " · p." + cl.page : "")
        + (cl.confidence != null ? " · conf " + fmt(cl.confidence, 2) : ""))),
  ]));
  return el("div", { class: "lit-conflict" }, [
    el("div", { class: "lit-conflict-head" }, [
      el("span", { class: "lit-conflict-title" },
        esc((cf.protein ? cf.protein + " · " : "") + (cf.field || "?"))),
      ctxParts.length ? el("span", { class: "faint", style: "font-size:.74rem" },
        esc(ctxParts.join(" · "))) : null,
      el("span", { class: "badge danger" }, "UNRESOLVED"),
    ]),
    el("div", { class: "lit-conflict-body" }, rows),
    el("div", { class: "faint", style: "font-size:.72rem" },
      "Both provenances retained; never auto-resolved (relative spread "
      + pct(cf.rel_spread) + ")."),
  ]);
}

async function renderLiterature(root) {
  let d;
  try {
    d = await api("/api/literature");
  } catch (e) {
    return;   // optional panel — never break the dashboard
  }
  const card = el("div", { class: "card literature-ingest" });
  card.appendChild(el("h2", {}, "Literature ingestion — extracted claims (M15)"));

  if (!d || d.available === false) {
    card.appendChild(el("p", { class: "muted" },
      esc((d && d.reason) || "lit_claims.jsonl not built — run "
        + "`python engine/m15_litingest.py`.")));
    root.appendChild(card);
    return;
  }

  // (a) core-framing banner — claims not facts
  card.appendChild(el("div", { class: "lit-banner" },
    "EXTRACTED CLAIMS — not trusted facts. Every claim is traceable "
    + "(DOI · page · verbatim text) with a measured confidence; conflicts "
    + "are surfaced unresolved; accuracy is measured against CPAD."));

  card.appendChild(el("p", { class: "explainer" },
    "What it means: M15 reads papers and PROPOSES structured claims (protein, "
    + "conditions, reported results, mechanism) — each a proposal with provenance + "
    + "a measured confidence, NOT a fact. It never overwrites the corpus; a human "
    + "curator approves before anything graduates. "
    + fmt(d.n_claims, 0) + " claims across " + fmt(d.n_papers, 0) + " papers."));

  // (b) CPAD-VALIDATION ACCURACY up top — measured against the curated gold standard
  const v = d.validation || {};
  const cal = v.confidence_calibration || {};
  if (v.precision != null || v.accuracy != null) {
    const kpis = el("div", { class: "lit-val-kpis" }, [
      litValKpi("precision", v.precision),
      litValKpi("recall", v.recall),
      litValKpi("accuracy", v.accuracy),
      litValKpi("curated curves", v.n_curated_curves, true),
      (cal.ece != null) ? litValKpi("ECE (calibration)", cal.ece) : null,
    ]);
    const counts = v.counts || {};
    card.appendChild(el("div", { class: "lit-val" }, [
      el("div", { class: "lit-val-label" },
        "CPAD-validation — extraction accuracy MEASURED against the curated gold standard"),
      kpis,
      el("p", { class: "faint", style: "font-size:.74rem" },
        "pmid " + esc(v.pmid || "?") + " · fields evaluated: "
        + esc((v.fields_evaluated || []).join(", ") || "—")
        + " · tp " + fmt(counts.tp, 0) + " / fp " + fmt(counts.fp, 0)
        + " / fn " + fmt(counts.fn, 0)
        + (cal.ece != null ? " · confidence is MEASURED/calibrated, not asserted" : "")),
    ]));
  }

  // (c) claims browser — grouped by paper, with a field/group filter
  const browser = el("div", { class: "lit-browser" });
  const controls = el("div", { class: "lit-filter" });
  const fieldSel = el("select", { class: "lit-filter-sel" }, [el("option", { value: "" }, "all fields")]);
  const groupSel = el("select", { class: "lit-filter-sel" }, [el("option", { value: "" }, "all groups")]);
  (d.field_coverage || []).forEach((fc) =>
    fieldSel.appendChild(el("option", { value: fc.field }, esc(fc.field + " (" + fc.count + ")"))));
  ["conditions", "reported_results", "semantic"].forEach((g) =>
    groupSel.appendChild(el("option", { value: g }, esc(g))));
  const claimsHost = el("div", { class: "lit-papers" });

  function renderClaims() {
    claimsHost.innerHTML = "";
    const wantField = fieldSel.value;
    const wantGroup = groupSel.value;
    let shown = 0;
    (d.claims_by_paper || []).forEach((p) => {
      const kept = (p.claims || []).filter((c) =>
        (!wantField || c.field === wantField) && (!wantGroup || c.group === wantGroup));
      if (!kept.length) return;
      shown += kept.length;
      const label = p.doi || p.pmid || "unknown paper";
      const sub = [p.author, p.year].filter((x) => x != null).join(" ");
      const paperEl = el("div", { class: "lit-paper" }, [
        el("div", { class: "lit-paper-head" }, [
          el("span", { class: "lit-paper-id" }, esc(String(label))),
          p.pmid ? el("span", { class: "badge muted-badge" }, "PMID " + esc(p.pmid)) : null,
          sub ? el("span", { class: "faint", style: "font-size:.74rem" }, esc(sub)) : null,
          el("span", { class: "badge" }, kept.length + " claim" + (kept.length === 1 ? "" : "s")),
        ]),
      ]);
      kept.forEach((c) => paperEl.appendChild(litClaimRow(c)));
      claimsHost.appendChild(paperEl);
    });
    if (!shown) claimsHost.appendChild(el("p", { class: "muted" }, "No claims match this filter."));
  }
  fieldSel.addEventListener("change", renderClaims);
  groupSel.addEventListener("change", renderClaims);
  controls.appendChild(el("span", { class: "faint", style: "font-size:.78rem" }, "filter:"));
  controls.appendChild(fieldSel);
  controls.appendChild(groupSel);
  browser.appendChild(el("p", { class: "section-note" }, "Claims browser (grouped by paper):"));
  browser.appendChild(controls);
  browser.appendChild(claimsHost);
  renderClaims();
  card.appendChild(browser);

  // (d) conflicts — divergent values with BOTH provenances, UNRESOLVED
  const conflicts = d.conflicts || [];
  const conflictHost = el("div", { class: "lit-conflicts" });
  if (conflicts.length) {
    conflicts.forEach((cf) => conflictHost.appendChild(litConflictCard(cf)));
  } else {
    conflictHost.appendChild(el("p", { class: "muted" }, "No cross-paper conflicts detected."));
  }
  card.appendChild(el("div", {}, [
    el("p", { class: "section-note" },
      "Cross-paper conflicts (" + fmt(d.n_conflicts, 0)
      + ") — divergent values for the same field/condition, surfaced UNRESOLVED (never auto-resolved):"),
    conflictHost,
  ]));

  // (e) field-coverage summary — which target fields were extracted + counts
  const fc = d.field_coverage || [];
  if (fc.length) {
    const chips = el("div", { class: "lit-coverage" });
    fc.forEach((f) => chips.appendChild(el("span", { class: "lit-cov-chip" }, [
      el("span", { class: "lit-cov-field" }, esc(f.field)),
      el("span", { class: "lit-cov-count" }, String(f.count)),
    ])));
    card.appendChild(el("div", {}, [
      el("p", { class: "section-note" },
        "Field coverage — target fields extracted (" + fc.length + " distinct):"),
      chips,
    ]));
  }

  // (f) honesty footer — claims not facts; append-only; capability-detected front-end
  // seams; fixtures. The server-supplied text is generated by M15 from LIVE capability
  // detection, so it is always preferred; the fallbacks below are only reached when
  // the payload carries no honesty block, and they must therefore not assert anything
  // about this host that they have not checked.
  const h = d.honesty || {};
  card.appendChild(el("div", { class: "lit-honesty" }, [
    el("strong", {}, "Honesty: "),
    document.createTextNode(esc(h.claims_not_facts
      || "every extraction is a CLAIM (a proposal with provenance + measured confidence), never a trusted fact")
      + " " + esc(h.append_only
        || "The ledger is append-only, never overwritten; a correction APPENDS a new claim with `supersedes` set.")
      // NB: this fallback makes NO claim about network access and does NOT declare the
      // seams dead. pdf_to_document / ocr_page / ner_extract / digitize_figure_curve
      // are capability-detected at call time; a seam is unavailable when its DEPENDENCY
      // is not installed on this interpreter — a dependency-availability fact, not a
      // statement about connectivity. Without a payload the frontend cannot know which
      // seams resolved, and it says so rather than guessing.
      + " " + esc(h.stubs_are_honest
        || "pdf_to_document / ocr_page / ner_extract / digitize_figure_curve are "
        + "CAPABILITY-DETECTED plug points: each resolves an injected implementation or "
        + "a detected backing library at call time, and otherwise raises "
        + "NotImplementedError naming the specific missing dependency — never a faked or "
        + "empty-but-plausible substitute. A seam is unavailable when that dependency is "
        + "NOT INSTALLED on this interpreter (PRISE ships stdlib + numpy/scipy only). "
        + "The live per-seam state is NOT known here: this is a fallback shown because "
        + "the server sent no honesty block — see honesty.frontend_capabilities for the "
        + "detected state.")
      + " " + esc(h.runs_on_fixtures
        || "No PDF corpus is bundled with PRISE, so the reference pipeline runs on "
        + "synthetic FIXTURE documents mimicking what a born-digital parser would emit; "
        + "a live pdf_to_document front-end feeds the SAME Document abstraction, so "
        + "nothing downstream changes.")
      + " " + esc(h.curator_approves
        || "M15 PROPOSES; only a human-approved, high-confidence, non-conflicting claim would ever graduate — nothing auto-trusted.")),
  ]));

  // (f2) the MACHINE-READABLE per-seam detected state, when M15 supplied it. This is
  // ground truth (what was actually detected on this interpreter), so it is shown in
  // preference to any prose. Absent key -> nothing rendered, no error.
  const caps = h.frontend_capabilities;
  if (caps && typeof caps === "object") {
    const seamRow = el("div", { class: "pill-row lit-seams" });
    Object.keys(caps).forEach((name) => {
      const s = caps[name] || {};
      const live = s.available === true;
      seamRow.appendChild(el("span", {
        class: "badge " + (live ? "ok" : "muted-badge"),
        title: live
          ? ("resolved via " + (s.source || "?") + ": " + (s.backend || "?"))
          : ("dependency not installed on this interpreter: " + (s.dependency || "?")),
      }, name + (live ? " · live" : " · dependency not installed")));
    });
    card.appendChild(el("p", { class: "section-note" },
      "Front-end seams (detected on this interpreter, not asserted):"));
    card.appendChild(seamRow);
  }

  root.appendChild(card);
}

function litValKpi(label, value, integer) {
  return el("div", { class: "lit-val-kpi" }, [
    el("div", { class: "lit-val-n" }, value == null ? "—" : fmt(value, integer ? 0 : 3)),
    el("div", { class: "lit-val-l" }, label),
  ]);
}

// --------------------------------------------------------------------------- //
// CROSS-MODAL (M10): Sequence × Kinetics × Structure
// Honesty-first: every panel states the unit (PROTEIN), the null, and the caveat.
// --------------------------------------------------------------------------- //
let crossmodalLoaded = false;

function honestyBanner(text) {
  return el("div", { class: "honesty-banner cm-banner" }, [
    el("strong", {}, "Honesty: "), document.createTextNode(text),
  ]);
}

// horizontal labelled bars (Plotly when available, SVG fallback) for the
// median-APR-by-regime panel.
function renderMedianBars(container, labels, values, opts) {
  container.innerHTML = "";
  const ylabel = (opts && opts.ylabel) || "median APR count";
  if (plotlyAvailable()) {
    try {
      Plotly.newPlot(container, [{
        x: labels, y: values, type: "bar",
        marker: { color: "#5ec6a8" },
        text: values.map((v) => (v == null ? "" : String(v))), textposition: "outside",
      }], {
        paper_bgcolor: "#1a212b", plot_bgcolor: "#11171f",
        font: { color: "#9fb0c0", size: 12 },
        margin: { l: 50, r: 16, t: 12, b: 70 },
        xaxis: { tickangle: -18, gridcolor: "#2d3a48" },
        yaxis: { title: ylabel, gridcolor: "#2d3a48", zerolinecolor: "#2d3a48" },
      }, { displayModeBar: false, responsive: true });
      return;
    } catch (e) { /* fall through */ }
  }
  // SVG fallback
  const W = 520, H = 240, pad = 48;
  const max = Math.max(...values.map((v) => v || 0), 1);
  const bw = (W - 2 * pad) / labels.length;
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="median bars">`;
  svg += `<rect x="0" y="0" width="${W}" height="${H}" fill="#11171f"/>`;
  svg += `<line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="#2d3a48"/>`;
  labels.forEach((lab, i) => {
    const v = values[i] || 0;
    const h = ((v / max) * (H - 2 * pad));
    const x = pad + i * bw + bw * 0.18;
    const w = bw * 0.64;
    const y = H - pad - h;
    svg += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" fill="#5ec6a8"/>`;
    svg += `<text x="${(x + w / 2).toFixed(1)}" y="${(y - 5).toFixed(1)}" fill="#e6edf3" font-size="11" text-anchor="middle">${esc(values[i])}</text>`;
    svg += `<text x="${(x + w / 2).toFixed(1)}" y="${(H - pad + 14).toFixed(1)}" fill="#9fb0c0" font-size="9" text-anchor="middle">${esc(String(lab).replace(/_/g, " "))}</text>`;
  });
  svg += `<text x="12" y="${H / 2}" fill="#6b7d8f" font-size="11" text-anchor="middle" transform="rotate(-90 12 ${H / 2})">${esc(ylabel)}</text>`;
  svg += `</svg>`;
  container.innerHTML = svg;
}

function cmStat(label, value, accent) {
  return el("div", { class: "kpi" }, [
    el("div", { class: "n", style: accent ? "color:" + accent : null }, String(value)),
    el("div", { class: "l" }, label),
  ]);
}

async function loadCrossmodal() {
  crossmodalLoaded = true;
  const root = $("#crossmodal-content");
  let d;
  try {
    d = await api("/api/crossmodal");
  } catch (e) {
    root.classList.remove("loading");
    root.innerHTML = "";
    root.appendChild(el("div", { class: "error-banner" },
      "Cross-modal analysis unavailable: " + esc(e.message)));
    return;
  }
  if (d.attribution) renderFooter(d.attribution);
  root.classList.remove("loading");
  root.innerHTML = "";

  if (!d.available) {
    root.appendChild(el("div", { class: "card" }, [
      el("h2", {}, "Cross-modal analysis not built"),
      el("p", { class: "muted" }, esc(d.reason ||
        "Run `python engine/m10_crossmodal.py` to generate m10_crossmodal.json.")),
    ]));
    return;
  }

  // intro
  root.appendChild(el("div", { class: "card overview-intro" }, [
    el("h2", {}, "Cross-modal: does the SEQUENCE predict the MEASURED kinetics?"),
    el("p", { class: "muted" },
      "PRISE uniquely holds, for the SAME proteins, sequence aggregation predictors " +
      "(PASTA/Waltz), measured kinetic regimes & rates, and structure/APR data. This " +
      "tab mines that overlap — and reports the answer honestly, even when it is a " +
      "null result."),
  ]));

  // ---- COVERAGE / effective-n banner ----
  const cov = d.coverage || {};
  root.appendChild(el("div", { class: "honesty-banner cm-effective-n" }, [
    el("h3", {}, "Effective n = PROTEINS, not curves"),
    el("p", { class: "clause" }, esc(cov.effective_n_banner || "")),
    el("div", { class: "kpi-row", style: "gap:1.4rem; flex-wrap:wrap; margin-top:.6rem;" }, [
      cmStat("proteins (total)", cov.n_proteins_total ?? "—", "var(--accent)"),
      cmStat("curves (total)", cov.n_curves_total ?? "—"),
      cmStat("with predictor", cov.n_proteins_with_predictor ?? "—"),
      cmStat("with point t50", cov.n_proteins_with_point_t50 ?? "—"),
      cmStat("predictor + t50", cov.n_proteins_predictor_AND_point_t50 ?? "—", "var(--accent)"),
      cmStat("with struct/APR", cov.n_proteins_with_structure_or_apr ?? "—"),
    ]),
  ]));

  // ---- P1: sequence -> kinetics association cards ----
  const p1 = d.P1_sequence_to_kinetics || {};
  const vc = p1.variance_ceiling || {};
  const p1card = el("div", { class: "card" }, [
    el("h2", {}, "P1 · Sequence → kinetics association (the headline)"),
    el("p", { class: "explainer" },
      "Do the endpoint-matched predictors (PASTA best pairing energy; Waltz amyloid-" +
      "hexapeptide coverage) predict the MEASURED rate/regime? Protein-level Spearman/" +
      "Pearson, leave-one-protein-out predictive skill, and a protein-label permutation " +
      "null (10,000 draws)."),
  ]);
  if (vc.computable) {
    p1card.appendChild(el("div", { class: "ceiling-note" }, [
      el("strong", {}, "Variance ceiling: "),
      document.createTextNode(
        pct(vc.between_protein_fraction) + " of total log10(t50) variance is " +
        "BETWEEN-protein (one-way ANOVA over " + vc.n_proteins + " proteins, " +
        vc.n_curves + " curves). That is the ABSOLUTE max any sequence-only " +
        "predictor could explain — the rest is condition-driven and unreachable " +
        "by a per-protein constant."),
    ]));
  }
  (p1.association_cards || []).forEach((c) => {
    const sig = c.significant_at_0_05;
    const card = el("div", { class: "assoc-card " + (sig ? "sig" : "null") }, [
      el("div", { class: "assoc-head" }, [
        el("span", { class: "assoc-title" },
          esc(c.predictor) + "  →  " + esc(c.target)),
        el("span", { class: "badge " + (sig ? "danger" : "muted-badge") },
          sig ? "association" : "no association"),
      ]),
      el("div", { class: "assoc-stats" }, [
        el("span", {}, "ρ = " + fmt(c.spearman_rho, 3)),
        el("span", {}, "r = " + fmt(c.pearson_r, 3)),
        el("span", {}, "LOPO skill = " + fmt(c.lopo_skill_over_baseline, 3)),
        el("span", {}, "perm p = " + fmt(c.permutation_p, 3)),
        el("span", {}, "ceiling = " +
          (c.variance_ceiling_fraction == null ? "—" : pct(c.variance_ceiling_fraction))),
        el("span", { class: "muted" }, "n = " + c.n_proteins + " proteins"),
      ]),
      el("p", { class: "assoc-verdict" }, esc(c.verdict_plain_english)),
    ]);
    p1card.appendChild(card);
  });
  p1card.appendChild(honestyBanner(p1.honesty ||
    "unit = protein; permutation shuffles protein labels; skill is leave-one-protein-out."));
  root.appendChild(p1card);

  // ---- P2: APR -> regime trend (the clean positive) ----
  const p2 = d.P2_apr_to_regime || {};
  const p2card = el("div", { class: "card" }, [
    el("h2", {}, "P2 · APR count → regime cooperativity trend"),
    el("p", { class: "explainer" },
      "Do proteins with MORE aggregation-prone regions reach MORE cooperative measured " +
      "regimes? Jonckheere–Terpstra trend test across the ordered regimes, with a " +
      "protein-level permutation null (so replicate curves can't inflate it)."),
  ]);
  if (p2.computable) {
    const meds = p2.median_apr_by_regime || {};
    const order = ["no_detectable_aggregation", "gradual_non_cooperative",
      "threshold_driven", "cooperative_sigmoidal"];
    const labels = order.filter((k) => k in meds);
    const values = labels.map((k) => meds[k]);
    const plot = el("div", { class: "cm-plot" });
    p2card.appendChild(plot);
    renderMedianBars(plot, labels, values, { ylabel: "median APR count per protein" });
    p2card.appendChild(el("div", { class: "assoc-stats" }, [
      el("span", {}, "Jonckheere J = " + fmt(p2.jonckheere_statistic, 1)),
      el("span", {}, "protein-perm p = " + fmt(p2.p_value_protein_permutation, 4)),
      el("span", {}, "direction: " + esc(p2.trend_direction || "—")),
      el("span", { class: "muted" }, "n = " + p2.n_proteins + " proteins"),
    ]));
    p2card.appendChild(el("p", { class: "assoc-verdict" }, esc(p2.verdict)));
    (p2.caveats || []).forEach((cv2) =>
      p2card.appendChild(el("p", { class: "caveat-line" }, "• " + esc(cv2))));
  } else {
    p2card.appendChild(el("p", { class: "muted" },
      "Not computable: " + esc(p2.reason || "insufficient proteins")));
  }
  root.appendChild(p2card);

  // ---- P3: discordance table ----
  const p3 = d.P3_discordance || {};
  const conf = p3.confusion_2x2 || {};
  const p3card = el("div", { class: "card" }, [
    el("h2", {}, "P3 · Sequence↔kinetics discordance scan"),
    el("p", { class: "explainer" },
      "Per protein: SEQUENCE-positive = predictor flags amyloid; KINETICS-positive = " +
      "aggregates somewhere in its measured mass-assay conditions. Censoring-aware — a " +
      "protein 'no-detectable' only under right-censoring is excluded, not a discordance."),
  ]);
  // 2x2 confusion table
  const tbl = el("table", { class: "data-table cm-confusion" });
  tbl.appendChild(el("tr", {}, [
    el("th", {}, ""), el("th", {}, "kinetics +"), el("th", {}, "kinetics −")]));
  tbl.appendChild(el("tr", {}, [
    el("th", {}, "sequence +"),
    el("td", { class: "num concord" }, fmt(conf.seq_pos_kin_pos, 0)),
    el("td", { class: "num discord" }, fmt(conf.seq_pos_kin_neg, 0))]));
  tbl.appendChild(el("tr", {}, [
    el("th", {}, "sequence −"),
    el("td", { class: "num discord" }, fmt(conf.seq_neg_kin_pos, 0)),
    el("td", { class: "num concord" }, fmt(conf.seq_neg_kin_neg, 0))]));
  p3card.appendChild(tbl);
  p3card.appendChild(el("div", { class: "assoc-stats" }, [
    el("span", {}, "evaluated = " + p3.n_evaluated + " proteins"),
    el("span", {}, "censoring-excluded = " + (p3.n_excluded_censored ?? 0)),
    el("span", {}, "observed rate = " + fmt(p3.observed_disagreement_rate, 3)),
    el("span", {}, "null rate = " + fmt(p3.null_disagreement_rate, 3)),
    el("span", {}, "binomial p = " + fmt(p3.binomial_p, 3)),
  ]));
  // discordant proteins list
  if ((p3.discordant_proteins || []).length) {
    const dl = el("div", { class: "discord-list" }, [el("strong", {}, "Discordant proteins: ")]);
    p3.discordant_proteins.forEach((dp) => {
      dl.appendChild(el("span", { class: "badge muted-badge" },
        esc(dp.name || dp.uniprot) +
        " (seq " + (dp.sequence_positive ? "+" : "−") +
        " / kin " + (dp.kinetics_positive ? "+" : "−") + ")"));
    });
    p3card.appendChild(dl);
  }
  p3card.appendChild(el("p", { class: "assoc-verdict" }, esc(p3.verdict)));
  p3card.appendChild(honestyBanner(
    "null disagreement rate combines a PRE-REGISTERED predictor FP baseline (" +
    pct((p3.null_components || {}).pasta_literature_fp_rate) +
    ") with the corpus right-censoring rate (" +
    pct((p3.null_components || {}).corpus_censoring_rate) +
    "). Fills the §3-M6 deferred `null_disagreement_rate`."));
  root.appendChild(p3card);

  // ---- P4: gamma sidebar (thin) ----
  const p4 = d.P4_gamma_sidebar || {};
  const p4card = el("div", { class: "card thin-card" }, [
    el("h2", {}, "P4 · γ × predictor (honestly thin)"),
    el("p", { class: "muted" }, esc(p4.verdict || "")),
  ]);
  if ((p4.table || []).length) {
    const gt = el("table", { class: "data-table" });
    gt.appendChild(el("tr", {}, [
      el("th", {}, "protein"), el("th", {}, "γ"),
      el("th", {}, "PASTA energy"), el("th", {}, "Waltz cov."),
      el("th", {}, "max regime")]));
    p4.table.forEach((r) => {
      gt.appendChild(el("tr", {}, [
        el("td", {}, esc(r.protein || r.uniprot)),
        el("td", { class: "num" }, fmt(r.gamma, 3)),
        el("td", { class: "num" }, fmt(r.pasta_energy, 2)),
        el("td", { class: "num" }, fmt(r.waltz_coverage, 0)),
        el("td", {}, esc(r.max_regime || "—"))]));
    });
    p4card.appendChild(gt);
  }
  root.appendChild(p4card);

  // ---- global honesty constraints footer ----
  const hc = el("div", { class: "card honesty-list" }, [el("h3", {}, "Honesty constraints (enforced)")]);
  (d.honesty_constraints || []).forEach((h) =>
    hc.appendChild(el("p", { class: "caveat-line" }, "• " + esc(h))));
  root.appendChild(hc);

  $("#build-badge").textContent = "build " + ((d.attribution || {}).engine_build_id || "?");
}

// --------------------------------------------------------------------------- //
// BROWSE
// --------------------------------------------------------------------------- //
let browseLoaded = false;
let allProteins = [];
let selectedProtein = null;

async function loadBrowse() {
  browseLoaded = true;
  const listEl = $("#protein-list");
  try {
    const d = await api("/api/proteins");
    if (d.attribution) {
      renderFooter(d.attribution);
      if (d.attribution.engine_build_id) {
        $("#build-badge").textContent = "build " + d.attribution.engine_build_id;
      }
    }
    allProteins = d.proteins || [];
    renderProteinList(allProteins);
    $("#protein-search").addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase().trim();
      const filtered = !q ? allProteins : allProteins.filter((p) =>
        (p.protein_id || "").toLowerCase().includes(q) ||
        (p.uniprot_id || "").toLowerCase().includes(q));
      renderProteinList(filtered);
    });
  } catch (e) {
    listEl.classList.remove("loading");
    listEl.innerHTML = "";
    listEl.appendChild(el("div", { class: "error-banner" }, "Failed: " + esc(e.message)));
  }
}

function renderProteinList(list) {
  const listEl = $("#protein-list");
  listEl.classList.remove("loading");
  listEl.innerHTML = "";
  if (!list.length) {
    listEl.appendChild(el("div", { class: "empty-hint" }, "No matches."));
    return;
  }
  list.forEach((p) => {
    const item = el("div", { class: "protein-item" + (p.protein_id === selectedProtein ? " active" : "") }, [
      el("div", { class: "pname" }, esc(p.protein_id)),
      el("div", { class: "pmeta" }, [
        el("span", {}, p.uniprot_id ? esc(p.uniprot_id) : "no uniprot"),
        el("span", {}, p.n_curves + " curve" + (p.n_curves === 1 ? "" : "s")),
        el("span", { class: "badge tier-" + (p.best_information_yield || "uninformative") },
          esc(p.best_information_yield || "—")),
        p.has_concentration_series ? el("span", { class: "badge ok" }, "conc-series") : null,
      ]),
    ]);
    item.addEventListener("click", () => {
      selectedProtein = p.protein_id;
      $$(".protein-item").forEach((n) => n.classList.remove("active"));
      item.classList.add("active");
      loadProteinDetail(p.protein_id);
    });
    listEl.appendChild(item);
  });
}

let currentDetail = null;
let currentUnitIdx = 0;

// tier ordering for "most informative" auto-selection
const YIELD_RANK = {
  mechanistic: 5, scaling: 4, descriptive: 3, signal_only: 2, uninformative: 1,
};

function pickDefaultUnitIdx(units) {
  // prefer the most informative dataset (highest yield tier), else the first.
  let bestIdx = 0, bestRank = -1;
  (units || []).forEach((u, i) => {
    const rank = YIELD_RANK[u.information_yield] || 0;
    if (rank > bestRank) { bestRank = rank; bestIdx = i; }
  });
  return bestIdx;
}

async function loadProteinDetail(pid) {
  const panel = $("#protein-detail");
  panel.innerHTML = "";
  panel.appendChild(el("div", { class: "loading" }, "Loading " + esc(pid) + "…"));
  try {
    const d = await api("/api/protein/" + encodeURIComponent(pid));
    currentDetail = d;
    // a fresh protein starts with an empty cohort selection (no cross-protein bleed)
    cohortSelected = new Set();
    // auto-select the most informative dataset so the user instantly sees a
    // concrete analysis (never an empty/ambiguous panel).
    currentUnitIdx = pickDefaultUnitIdx(d.units || []);
    renderProteinDetail();
  } catch (e) {
    panel.innerHTML = "";
    panel.appendChild(el("div", { class: "error-banner" }, "Failed: " + esc(e.message)));
  }
}

function renderProteinDetail() {
  const d = currentDetail;
  const panel = $("#protein-detail");
  panel.innerHTML = "";

  // head
  const meta = (d.units && d.units[0]) || {};
  panel.appendChild(el("div", { class: "detail-head" }, [
    el("h2", {}, esc(d.protein_id)),
    d.uniprot_id ? el("a", { href: "https://www.uniprot.org/uniprotkb/" + encodeURIComponent(d.uniprot_id),
      target: "_blank", class: "badge" }, "UniProt " + esc(d.uniprot_id)) : null,
    el("span", { class: "badge" }, d.n_curves + " unit" + (d.n_curves === 1 ? "" : "s")),
    el("span", { class: "badge mono" }, "build " + esc(d.engine_build_id || "")),
  ]));

  // DATASET PICKER — a readable, scannable table of this protein's CPAD curves.
  panel.appendChild(buildDatasetPicker(d));

  const unit = (d.units || [])[currentUnitIdx] || {};
  renderUnit(panel, unit, d);

  // CONCENTRATION-SERIES SCALING (dual-γ) — protein-level, shown once per protein
  // when this protein has a concentration series (has_concentration_series flag).
  const plistEntry = (allProteins || []).find((p) => p.protein_id === d.protein_id);
  if (plistEntry && plistEntry.has_concentration_series) {
    const gammaCard = el("div", { class: "card gamma-card" }, [
      el("h3", {}, "Concentration-series scaling (dual-γ)"),
      el("p", { class: "explainer" },
        "γ is the half-time scaling exponent (t50 ∝ [monomer]^−γ) — a mechanistic " +
        "CONSTRAINT across this protein's concentration series. The two estimators are " +
        "reported separately and never merged; disagreement means the curve shape isn't " +
        "concentration-invariant."),
      el("div", { class: "gamma-body" }, [el("div", { class: "loading" }, "loading concentration-series scaling…")]),
    ]);
    panel.appendChild(gammaCard);
    loadSeriesGamma($(".gamma-body", gammaCard), d.protein_id);
  }

  // DOSE–RESPONSE (k_agg vs concentration) — protein-level, shown once per protein
  // when this protein has CPAD R-rows fitted to a dose-response bank.
  if (plistEntry && plistEntry.has_dose_response) {
    const drCard = el("div", { class: "card dose-card" }, [
      el("h3", {}, "Dose–response (k_agg vs concentration)"),
      el("p", { class: "explainer" },
        "How the apparent aggregation rate (k_agg) scales with monomer concentration — " +
        "a complementary, endpoint-style view to the kinetic dual-γ. The bank " +
        "(LNT / threshold / Hill / Brain–Cousens / power-law) is ranked by AICc."),
      el("div", { class: "dose-body" }, [el("div", { class: "loading" }, "loading dose–response…")]),
    ]);
    panel.appendChild(drCard);
    loadDoseResponse($(".dose-body", drCard), d.protein_id);
  }

  // CROSS-STUDY META-ANALYSIS (M14) — protein-level, shown once per protein. This
  // is where the forest plot / heterogeneity / variance decomposition / funnel live
  // (a cross-study result belongs on the PROTEIN view, above the per-curve view).
  const metaCard = el("div", { class: "card meta-card" }, [
    el("h3", {}, "Cross-study meta-analysis (M14)"),
    el("p", { class: "explainer" },
      "Hierarchical normal random-effects pooling of the log-t50 effect ACROSS studies " +
      "for this protein — condition-adjusted (meta-regression on log₁₀[concentration] " +
      "+ pH/temperature/construct), never raw t50 across concentrations. Shown only when " +
      "≥2 studies contribute; otherwise reported honestly as a single study."),
    el("div", { class: "meta-body" }, [el("div", { class: "loading" }, "loading cross-study meta-analysis…")]),
  ]);
  panel.appendChild(metaCard);
  loadMeta($(".meta-body", metaCard), d.protein_id);

  // SEQUENCE ↔ STRUCTURE ↔ KINETICS (M16) — protein-level, shown once per protein.
  // Structural features here are SEQUENCE/APR-derived PROXIES (not 3D); the
  // structure→kinetics associations are STATISTICAL ONLY, never causal.
  const structCard = el("div", { class: "card struct-card" }, [
    el("h3", {}, "Sequence ↔ structure ↔ kinetics (M16)"),
    el("p", { class: "explainer" },
      "Per-protein structural PROXY features (APR/sequence-derived, NOT 3D) + the " +
      "corpus structure→kinetics association table. Real 3D features are deferred; a " +
      "native-monomer structure is not the aggregation-competent/fibril state; every " +
      "association is statistical only — never causal."),
    el("div", { class: "struct-body" }, [el("div", { class: "loading" }, "loading sequence↔structure↔kinetics…")]),
  ]);
  panel.appendChild(structCard);
  loadStructure($(".struct-body", structCard), d.protein_id);

  // structure links (protein-level)
  const sl = d.structure_linkage || {};
  if (sl.pdb_ids && sl.pdb_ids.length) {
    const pills = el("div", { class: "pill-row" });
    sl.pdb_ids.forEach((id) => pills.appendChild(
      el("a", { href: "https://www.rcsb.org/structure/" + encodeURIComponent(id),
        target: "_blank", class: "badge" }, esc(id))));
    panel.appendChild(el("div", { class: "card" }, [
      el("h3", {}, "Structure linkage (associative only)"),
      pills,
      el("p", { class: "section-note" }, esc(sl.caveat || "")),
    ]));
  }
}

// --------------------------------------------------------------------------- //
// CONCENTRATION-SERIES SCALING (dual-γ) — per protein with a series
// --------------------------------------------------------------------------- //
async function loadSeriesGamma(container, proteinId) {
  try {
    const d = await api("/api/series-gamma/" + encodeURIComponent(proteinId));
    if (d.error) {
      container.innerHTML = "";
      container.appendChild(el("div", { class: "muted" }, "Scaling unavailable: " + esc(d.error)));
      return;
    }
    renderSeriesGamma(container, d);
  } catch (e) {
    container.innerHTML = "";
    container.appendChild(el("div", { class: "muted" }, "Scaling failed: " + esc(e.message)));
  }
}

function gammaEstimatorBlock(label, est) {
  est = est || {};
  // estimator can be insufficient_data / not_computed -> surface the status, no fake γ
  if (est.status && est.status !== "ok") {
    return el("div", { class: "gamma-estimator" }, [
      el("h4", {}, label),
      el("div", { class: "gamma-val muted" }, "γ = —"),
      el("div", { class: "section-note" }, "status: " + esc(est.status) +
        (est.n_distinct_concentrations != null
          ? " (" + est.n_distinct_concentrations + " distinct [m])" : "")),
    ]);
  }
  const g = est.gamma;
  const ci = est.gamma_ci;
  const ciStr = (Array.isArray(ci) && ci.length === 2)
    ? " [" + fmt(ci[0], 2) + ", " + fmt(ci[1], 2) + "]" : " (no CI)";
  const bits = [
    el("div", { class: "gamma-val" }, "γ = " + (g != null ? fmt(g, 3) : "—")),
    el("div", { class: "gamma-ci" }, "95% CI:" + ciStr),
  ];
  if (est.gamma_reliable != null) {
    bits.push(el("span", { class: "badge " + (est.gamma_reliable ? "ok" : "warn") },
      est.gamma_reliable ? "reliable" : "flagged unreliable"));
  }
  // honesty: negative-γ driving-force gate (non-physical t50-rises-with-[m])
  if (est.gamma_physical === false) {
    bits.push(el("span", { class: "badge danger" }, "γ<0 non-physical"));
    if (est.gamma_raw != null) {
      bits.push(el("div", { class: "section-note warn-note" },
        "raw slope γ = " + fmt(est.gamma_raw, 3) + " (kept for transparency, NOT emitted as an estimate)"));
    }
    if (est.gamma_physical_reason) {
      bits.push(el("div", { class: "section-note warn-note" }, esc(est.gamma_physical_reason)));
    }
  }
  if (est.collapse_r2 != null) {
    bits.push(el("div", { class: "section-note" }, "collapse R² = " + fmt(est.collapse_r2, 3)));
  }
  // relabel honesty: the collapse is phenomenological, not a shared-rate ODE fit
  if (est.is_shared_rate_ode_fit === false) {
    bits.push(el("div", { class: "section-note" },
      "phenomenological master-curve collapse (shared SHAPE only — not shared microscopic rates)"));
  }
  (est.reliability_reasons || []).forEach((r) =>
    bits.push(el("div", { class: "section-note warn-note" }, esc(r))));
  return el("div", { class: "gamma-estimator" }, [el("h4", {}, label), ...bits]);
}

// The shared-rate Knowles/Cohen ODE global fit (engine/global_fit.py): the THIRD
// γ_mechanistic from ESTIMATED reaction orders, mechanism-AICc ranking, and the
// identifiability / sloppiness report. Honesty-first: reaction-order CONSTRAINT +
// identifiable combinations, NOT unique rates; mechanism NOT licensed.
function globalOdeFitBlock(g) {
  const ro = g.reaction_orders || {};
  const n2txt = (ro.n_2 == null) ? "fixed (nuisance for this family)" : fmt(ro.n_2, 2);
  const bits = [
    el("h4", {}, "Shared-rate Knowles/Cohen ODE global fit (γ #3)"),
    el("div", { class: "gamma-val" },
      "γ_mechanistic = " + (g.gamma_mechanistic != null ? fmt(g.gamma_mechanistic, 3) : "—")),
    el("div", { class: "section-note" },
      "from ESTIMATED reaction orders — " + esc(g.gamma_mechanistic_formula || "")),
    el("div", { class: "section-note" },
      "n_c = " + (ro.n_c != null ? fmt(ro.n_c, 2) : "—") + " · n_2 = " + n2txt
      + (g.best_mechanism ? "  · best family: " + esc(g.best_mechanism) : "")),
  ];
  // sloppiness / identifiability — the honest finding
  if (g.sloppy === true) {
    bits.push(el("span", { class: "badge warn" }, "SLOPPY — rates not uniquely identifiable"));
  } else if (g.sloppy === false) {
    bits.push(el("span", { class: "badge ok" }, "shared FIM reasonably conditioned"));
  }
  if (g.fim_condition_number != null) {
    bits.push(el("div", { class: "section-note" },
      "FIM condition number ≈ " + fmt(g.fim_condition_number, 0)));
  }
  // identifiable combination(s): the stiff direction(s) the data actually constrains
  const combos = (g.identifiable_combinations || []).filter((c) => c.stiff);
  if (combos.length) {
    bits.push(el("div", { class: "section-note" },
      "identifiable combination: " + esc(combos[0].combination)));
  }
  // per-rate status (constrained vs sloppy/unconstrained)
  if (g.individual_rate_status) {
    const sl = Object.entries(g.individual_rate_status)
      .filter(([, v]) => v !== "constrained").map(([k]) => k);
    if (sl.length) {
      bits.push(el("div", { class: "section-note warn-note" },
        "sloppy/unconstrained: " + sl.map(esc).join(", ")));
    }
  }
  // mechanism AICc ranking
  const rank = g.mechanism_aicc_ranking || [];
  if (rank.length) {
    const rstr = rank.map((r) =>
      esc(r.mechanism) + " (ΔAICc " + fmt(r.delta_aicc, 1) + ")").join(" · ");
    bits.push(el("div", { class: "section-note" }, "mechanism AICc ranking: " + rstr));
  }
  if (g.honesty_note) {
    bits.push(el("div", { class: "section-note warn-note" }, esc(g.honesty_note)));
  } else {
    bits.push(el("div", { class: "section-note warn-note" },
      "Reaction-order CONSTRAINT + identifiable combinations — NOT a unique mechanism "
      + "(M5 still refuses: agitation/seeding unknown on this corpus)."));
  }
  return el("div", { class: "gamma-estimator gamma-ode" }, bits);
}

function renderSeriesGamma(container, d) {
  container.innerHTML = "";
  const seriesList = d.series || [];
  if (!seriesList.length) {
    container.appendChild(el("div", { class: "muted" }, "No concentration series resolved for this protein."));
    return;
  }
  seriesList.forEach((s, idx) => {
    const wrap = el("div", { class: "gamma-series" });
    const cond = s.conditions || {};
    const condBits = [];
    if (cond.assay) condBits.push(esc(cond.assay));
    if (cond.pH != null) condBits.push("pH " + fmt(cond.pH));
    if (cond.temperature_C != null) condBits.push(fmt(cond.temperature_C) + " °C");
    if (cond.mutation) condBits.push(esc(cond.mutation));
    wrap.appendChild(el("div", { class: "gamma-series-head" }, [
      el("span", { class: "badge mono" }, esc(s.concentration_series_id || ("series " + (idx + 1)))),
      el("span", { class: "muted", style: "font-size:.8rem" }, condBits.join(" · ")),
      el("span", { class: "badge" }, (s.n_member_curves ?? "?") + " curves"),
      el("span", { class: "badge" }, (s.n_distinct_concentrations ?? "?") + " distinct [m]"),
    ]));

    // two γ estimators side by side
    const ests = el("div", { class: "gamma-estimators" }, [
      gammaEstimatorBlock("γ regression (censored log–log)", s.gamma_regression),
      gammaEstimatorBlock("γ global (master-curve collapse)", s.gamma_global),
    ]);
    wrap.appendChild(ests);

    // disagreement verdict
    const dis = s.disagreement || {};
    if (dis.testable === false) {
      wrap.appendChild(el("p", { class: "section-note" },
        "Disagreement: not testable — " + esc(dis.reason || "")));
    } else if (dis.disagree != null) {
      wrap.appendChild(el("div", { class: "gamma-verdict" }, [
        el("span", { class: "badge " + (dis.disagree ? "danger" : "ok") },
          dis.disagree ? "γ estimators DISAGREE" : "γ estimators consistent"),
        el("span", { class: "section-note" }, esc(dis.interpretation || "")),
      ]));
    }

    // curvature test result
    const curv = (s.gamma_regression || {}).curvature_test || {};
    if (curv.testable === false) {
      wrap.appendChild(el("p", { class: "section-note" },
        "Curvature test: not testable — " + esc(curv.reason || "")));
    } else if (curv.testable) {
      wrap.appendChild(el("div", { class: "gamma-verdict" }, [
        el("span", { class: "badge " + (curv.significant ? "warn" : "ok") },
          curv.significant ? "log–log curvature SIGNIFICANT" : "no detectable curvature"),
        el("span", { class: "section-note" },
          "p = " + fmt(curv.p_value, 3) + " · " + esc(curv.interpretation || "")),
      ]));
    }

    // shape-not-concentration-invariant flag (poor master-curve collapse)
    if (s.shape_not_concentration_invariant === true) {
      wrap.appendChild(el("div", { class: "gamma-verdict" }, [
        el("span", { class: "badge danger" }, "shape NOT concentration-invariant"),
        el("span", { class: "section-note" },
          "the master curve does not collapse (collapse R² < 0.8) — the curve shape is "
          + "concentration-dependent; that is itself a mechanistic signal"),
      ]));
    }

    // shared-rate Knowles/Cohen ODE global fit — the THIRD γ + reaction orders +
    // identifiability (present only when engine/global_fit.py was run for this series)
    if (s.global_ode_fit) {
      wrap.appendChild(globalOdeFitBlock(s.global_ode_fit));
    }

    // log-log scatter plot of t50 vs [monomer] with the fitted −γ slope line
    const points = s.points || [];
    if (points.length >= 2) {
      const plotDiv = el("div", { class: "plot-wrap gamma-plot" });
      wrap.appendChild(plotDiv);
      renderGammaScatter(plotDiv, s);
    } else {
      wrap.appendChild(el("p", { class: "section-note" },
        "Not enough (concentration, t50) points to draw the log–log scatter."));
    }

    container.appendChild(wrap);
    if (idx < seriesList.length - 1) container.appendChild(el("hr", { class: "gamma-sep" }));
  });

  container.appendChild(el("p", { class: "section-note" }, esc(d.explanation || "")));
}

// log10(t50) vs log10([monomer]) scatter + fitted −γ slope line (Plotly + SVG fallback)
function renderGammaScatter(container, s) {
  const points = s.points || [];
  const obs = points.filter((p) => !p.right_censored && !p.left_censored);
  const cens = points.filter((p) => p.right_censored || p.left_censored);
  const fit = s.fit_line;

  // build the fitted line over the observed log10([m]) range
  const allLm = points.map((p) => p.log10_concentration);
  const lmMin = Math.min(...allLm), lmMax = Math.max(...allLm);
  let lineX = null, lineY = null;
  if (fit && fit.intercept_log10t50 != null && fit.slope != null) {
    lineX = [lmMin, lmMax];
    lineY = lineX.map((lm) => fit.intercept_log10t50 + fit.slope * lm);
  }

  if (plotlyAvailable()) {
    try {
      const traces = [{
        x: obs.map((p) => p.log10_concentration), y: obs.map((p) => p.log10_t50),
        mode: "markers", type: "scatter", name: "uncensored t50",
        marker: { size: 9, color: "#4ea1d3" },
        text: obs.map((p) => fmt(p.concentration_uM) + " µM"),
      }];
      if (cens.length) {
        traces.push({
          x: cens.map((p) => p.log10_concentration), y: cens.map((p) => p.log10_t50),
          mode: "markers", type: "scatter", name: "censored t50 (bound)",
          marker: { size: 11, color: "#e0a458", symbol: "triangle-up-open", line: { width: 2 } },
          text: cens.map((p) => fmt(p.concentration_uM) + " µM · " + (p.censoring_class || "")),
        });
      }
      if (lineX) {
        traces.push({
          x: lineX, y: lineY, mode: "lines", type: "scatter",
          name: "fit: slope −γ = " + fmt(fit.slope, 2),
          line: { color: "#5ec6a8", width: 2, dash: "solid" },
        });
      }
      const layout = {
        paper_bgcolor: "#1a212b", plot_bgcolor: "#11171f",
        font: { color: "#9fb0c0", size: 12 },
        margin: { l: 54, r: 16, t: 12, b: 46 },
        xaxis: { title: "log₁₀ [monomer] (µM)", gridcolor: "#2d3a48", zerolinecolor: "#2d3a48" },
        yaxis: { title: "log₁₀ t50 (h)", gridcolor: "#2d3a48", zerolinecolor: "#2d3a48" },
        legend: { orientation: "h", y: -0.25, font: { size: 11 } },
        showlegend: true,
      };
      Plotly.newPlot(container, traces, layout, { displayModeBar: false, responsive: true });
      return;
    } catch (e) { /* fall through to SVG */ }
  }
  // SVG fallback: reuse renderCurveFallback with log-space coords + censored marks
  const x = points.map((p) => p.log10_concentration);
  const y = points.map((p) => p.log10_t50);
  y.__censored__ = points.map((p) => p.right_censored || p.left_censored);
  renderCurveFallback(container, {
    x, y,
    fitted: (lineX ? { x: lineX, y: lineY, model: "slope −γ = " + fmt((fit || {}).slope, 2) } : null),
    censoring: "none",
    xlabel: "log10 [monomer] (uM)", ylabel: "log10 t50 (h)",
  });
}

// --------------------------------------------------------------------------- //
// CROSS-STUDY META-ANALYSIS (M14) — per protein, forest + heterogeneity +
// variance decomposition + leave-one-out + funnel + honesty footer.
// The forest plot is THE deliverable. Everything degrades to an SVG/table
// fallback so the panel is never blank without Plotly.
// --------------------------------------------------------------------------- //
async function loadMeta(container, proteinId) {
  try {
    const d = await api("/api/meta/" + encodeURIComponent(proteinId));
    renderMeta(container, d);
  } catch (e) {
    container.innerHTML = "";
    container.appendChild(el("div", { class: "muted" },
      "Cross-study meta-analysis failed: " + esc(e.message)));
  }
}

function renderMeta(container, d) {
  container.innerHTML = "";
  d = d || {};

  // artifacts not built at all
  if (d.available === false) {
    container.appendChild(el("div", { class: "muted" },
      "Cross-study meta-analysis unavailable: " + esc(d.reason || "not built")));
    return;
  }

  // SINGLE-STUDY (or absent) protein: an honest one-liner, no fabricated panel.
  if (d.status !== "meta_analysis") {
    const ns = d.n_studies != null ? d.n_studies : 1;
    container.appendChild(el("div", { class: "meta-single" }, [
      el("span", { class: "badge warn" }, "single study (n=" + esc(ns) + ")"),
      el("span", { class: "section-note" },
        "Single study (n=1) — no cross-study pooling possible. " + esc(d.note || "")),
    ]));
    return;
  }

  // ---- head: n_studies / n_curves + the pooled point estimate ----
  const pooled = d.pooled_estimate || {};
  const t50 = pooled.mu_t50_hours;
  const ciT = pooled.ci95_t50_hours || [];
  const headBits = [
    el("span", { class: "badge" }, (d.n_studies ?? "?") + " studies"),
    el("span", { class: "badge" }, (d.n_curves ?? "?") + " curves"),
  ];
  if (pooled.condition_adjusted) headBits.push(el("span", { class: "badge ok" }, "condition-adjusted"));
  container.appendChild(el("div", { class: "meta-head" }, headBits));
  if (t50 != null) {
    container.appendChild(el("p", { class: "meta-pooled-line" }, [
      el("strong", {}, "Pooled t50 = " + fmt(t50, 2) + " h"),
      document.createTextNode(
        (ciT.length === 2 ? "  (95% CI " + fmt(ciT[0], 2) + "–" + fmt(ciT[1], 2) + " h)" : "") +
        "  ·  μ_log = " + fmt(pooled.mu_log, 3) +
        " on the natural-log-t50 scale " +
        (pooled.moderators_used && pooled.moderators_used.length
          ? "(moderators: " + pooled.moderators_used.map(esc).join(", ") + ")" : "")),
    ]));
  }

  // ---- a) FOREST PLOT (the deliverable) ----
  container.appendChild(el("h4", { class: "meta-sub" }, "Forest plot"));
  const forestDiv = el("div", { class: "plot-wrap meta-forest" });
  container.appendChild(forestDiv);
  renderForestPlot(forestDiv, d);
  container.appendChild(el("p", { class: "section-note" },
    "Effect = back-transformed t50 (hours, log-spaced axis). Marker area ∝ study weight; " +
    "bars are 95% CIs. The diamond is the random-effects pooled estimate; the dashed " +
    "interval is the 95% PREDICTION interval for a NEW study (wider than the CI — it " +
    "carries the between-study spread τ)."));

  // ---- b) HETEROGENEITY ----
  container.appendChild(renderHeterogeneity(d.heterogeneity || {}));

  // ---- c) VARIANCE DECOMPOSITION ----
  container.appendChild(renderVarianceDecomposition(d.variance_decomposition || {}, d.laboratory || {}));

  // ---- d) LEAVE-ONE-STUDY-OUT ----
  container.appendChild(renderLeaveOneOut(d.leave_one_study_out || []));

  // ---- e) PUBLICATION BIAS (funnel + Egger + small-study flag) ----
  container.appendChild(renderPublicationBias(d.publication_bias || {}));

  // ---- f) HONESTY FOOTER ----
  container.appendChild(renderMetaHonesty(d.honesty || {}));
}

// (a) FOREST PLOT: one row per study (effect + 95% CI + weight-scaled marker),
// then the pooled diamond + the prediction interval. Plotly + SVG fallback.
function renderForestPlot(container, d) {
  const fp = d.forest_plot || {};
  const rows = (fp.rows || []).slice();
  if (!rows.length) {
    container.appendChild(el("p", { class: "section-note" }, "No per-study forest rows available."));
    return;
  }
  const diamond = fp.pooled_diamond || {};
  const piT = fp.prediction_interval_t50 || [];

  // label per study: prefer author/year (from study_deviations), else PMID
  const authorByPmid = {};
  (d.study_deviations || []).forEach((s) => {
    if (s.pmid) authorByPmid[String(s.pmid)] = s.author;
  });
  function labelFor(r) {
    const pmid = r.pmid != null ? String(r.pmid) : null;
    const auth = pmid ? authorByPmid[pmid] : null;
    const first = auth ? String(auth).split(",")[0].trim() : null;
    if (first) return first + (pmid ? " (PMID " + pmid + ")" : "");
    return pmid ? "PMID " + pmid : (r.study || "study");
  }

  // effect + CI on the t50 (hours) scale; guard against non-positive for log axis
  function eff(r) { return r.effect_t50; }
  function ciOf(r) { return r.ci95_t50 || []; }

  const weights = rows.map((r) => (r.weight_pct != null ? r.weight_pct : 1));
  const wMax = Math.max(...weights, 1e-9);

  if (plotlyAvailable()) {
    try {
      const yLabels = rows.map((r) => labelFor(r) + "  (" + (r.n_curves ?? "?") + "c, w=" +
        fmt(r.weight_pct, 1) + "%)");
      // rows plotted top→bottom in the order given (reverse the y so first row is top)
      const yPos = rows.map((r, i) => rows.length - i);
      const errPlus = rows.map((r) => { const c = ciOf(r); return (c.length === 2 ? Math.max(0, c[1] - eff(r)) : 0); });
      const errMinus = rows.map((r) => { const c = ciOf(r); return (c.length === 2 ? Math.max(0, eff(r) - c[0]) : 0); });
      const studyTrace = {
        x: rows.map(eff), y: yPos, mode: "markers", type: "scatter", name: "study",
        marker: {
          size: rows.map((r) => 8 + 22 * Math.sqrt((r.weight_pct || 1) / wMax)),
          color: "#4ea1d3", line: { color: "#0b1017", width: 1 },
        },
        error_x: { type: "data", symmetric: false, array: errPlus, arrayminus: errMinus,
          color: "#4ea1d3", thickness: 1.5, width: 3 },
        text: rows.map((r) => labelFor(r) + " · t50=" + fmt(eff(r), 2) + " h · w=" + fmt(r.weight_pct, 1) + "%"),
        hoverinfo: "text",
      };
      const traces = [studyTrace];
      // pooled diamond (as a single marked point + its CI whisker) at y=0
      if (diamond.mu_t50 != null) {
        const dc = diamond.ci95_t50 || [];
        traces.push({
          x: [diamond.mu_t50], y: [0], mode: "markers", type: "scatter", name: "pooled (RE)",
          marker: { size: 16, color: "#5ec6a8", symbol: "diamond", line: { color: "#0b1017", width: 1 } },
          error_x: (dc.length === 2 ? { type: "data", symmetric: false,
            array: [Math.max(0, dc[1] - diamond.mu_t50)], arrayminus: [Math.max(0, diamond.mu_t50 - dc[0])],
            color: "#5ec6a8", thickness: 2, width: 6 } : undefined),
          text: ["pooled t50 = " + fmt(diamond.mu_t50, 2) + " h"], hoverinfo: "text",
        });
      }
      // prediction interval (dashed, wider) at y=-1
      if (piT.length === 2) {
        traces.push({
          x: piT, y: [-1, -1], mode: "lines", type: "scatter", name: "95% prediction interval",
          line: { color: "#e0a458", width: 3, dash: "dash" },
          text: "new-study 95% PI: " + fmt(piT[0], 2) + "–" + fmt(piT[1], 2) + " h", hoverinfo: "text",
        });
      }
      const tickvals = yPos.concat(diamond.mu_t50 != null ? [0] : []).concat(piT.length === 2 ? [-1] : []);
      const ticktext = yLabels.concat(diamond.mu_t50 != null ? ["◆ pooled (random-effects)"] : [])
        .concat(piT.length === 2 ? ["— 95% prediction interval"] : []);
      const layout = {
        paper_bgcolor: "#1a212b", plot_bgcolor: "#11171f",
        font: { color: "#9fb0c0", size: 11 },
        margin: { l: 240, r: 20, t: 10, b: 46 },
        height: Math.max(260, 44 * rows.length + 120),
        xaxis: { title: "t50 (hours, log scale)", type: "log", gridcolor: "#2d3a48", zerolinecolor: "#2d3a48" },
        yaxis: { tickvals: tickvals, ticktext: ticktext, gridcolor: "#1c2731",
          zeroline: false, range: [-1.8, rows.length + 0.8] },
        showlegend: false,
      };
      // pooled reference line (vertical) at the pooled t50
      if (diamond.mu_t50 != null) {
        layout.shapes = [{ type: "line", x0: diamond.mu_t50, x1: diamond.mu_t50,
          y0: -1.8, y1: rows.length + 0.8, line: { color: "#5ec6a8", width: 1, dash: "dot" } }];
      }
      Plotly.newPlot(container, traces, layout, { displayModeBar: false, responsive: true });
      return;
    } catch (e) { /* fall through to SVG */ }
  }
  renderForestFallback(container, rows, labelFor, eff, ciOf, wMax, diamond, piT);
}

// SVG forest-plot fallback (log-t50 axis) — one row per study + pooled diamond +
// dashed prediction interval. Self-contained; no Plotly.
function renderForestFallback(container, rows, labelFor, eff, ciOf, wMax, diamond, piT) {
  const wrap = el("div", { class: "plot-fallback" });
  const W = 620, rowH = 30, padL = 250, padR = 24, padT = 14;
  const nExtra = 2; // pooled + PI rows
  const H = padT + rowH * (rows.length + nExtra) + 40;
  // collect all positive effect/CI values to set a log10 x-domain
  const vals = [];
  rows.forEach((r) => { const c = ciOf(r); [eff(r), c[0], c[1]].forEach((v) => { if (v > 0) vals.push(v); }); });
  if (diamond.mu_t50 > 0) vals.push(diamond.mu_t50);
  (diamond.ci95_t50 || []).forEach((v) => { if (v > 0) vals.push(v); });
  piT.forEach((v) => { if (v > 0) vals.push(v); });
  const lo = Math.log10(Math.min(...vals)), hi = Math.log10(Math.max(...vals));
  const span = (hi - lo) || 1;
  const sx = (v) => (v > 0
    ? padL + ((Math.log10(v) - lo) / span) * (W - padL - padR)
    : padL);

  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="forest plot">`;
  svg += `<rect x="0" y="0" width="${W}" height="${H}" fill="#11171f"/>`;
  // pooled reference vertical line
  if (diamond.mu_t50 > 0) {
    const px = sx(diamond.mu_t50);
    svg += `<line x1="${px.toFixed(1)}" y1="${padT}" x2="${px.toFixed(1)}" y2="${(H - 34).toFixed(1)}" stroke="#5ec6a8" stroke-width="1" stroke-dasharray="3 3"/>`;
  }
  // per-study rows
  rows.forEach((r, i) => {
    const y = padT + rowH * i + rowH / 2;
    const c = ciOf(r);
    const x0 = c.length === 2 ? sx(c[0]) : sx(eff(r));
    const x1 = c.length === 2 ? sx(c[1]) : sx(eff(r));
    const xc = sx(eff(r));
    const msz = 3 + 6 * Math.sqrt((r.weight_pct || 1) / wMax);
    svg += `<line x1="${x0.toFixed(1)}" y1="${y.toFixed(1)}" x2="${x1.toFixed(1)}" y2="${y.toFixed(1)}" stroke="#4ea1d3" stroke-width="1.5"/>`;
    svg += `<rect x="${(xc - msz).toFixed(1)}" y="${(y - msz).toFixed(1)}" width="${(2 * msz).toFixed(1)}" height="${(2 * msz).toFixed(1)}" fill="#4ea1d3"/>`;
    const lab = esc(labelFor(r)) + "  (" + (r.n_curves ?? "?") + "c, w=" + fmt(r.weight_pct, 1) + "%)";
    svg += `<text x="6" y="${(y + 4).toFixed(1)}" fill="#9fb0c0" font-size="11">${lab}</text>`;
    svg += `<text x="${(W - padR).toFixed(1)}" y="${(y + 4).toFixed(1)}" fill="#7e8ea0" font-size="10" text-anchor="end">${fmt(eff(r), 2)}h</text>`;
  });
  // pooled diamond row
  if (diamond.mu_t50 > 0) {
    const y = padT + rowH * rows.length + rowH / 2;
    const dc = diamond.ci95_t50 || [];
    const xc = sx(diamond.mu_t50);
    const dx0 = dc.length === 2 ? sx(dc[0]) : xc - 8;
    const dx1 = dc.length === 2 ? sx(dc[1]) : xc + 8;
    svg += `<polygon points="${dx0.toFixed(1)},${y.toFixed(1)} ${xc.toFixed(1)},${(y - 8).toFixed(1)} ${dx1.toFixed(1)},${y.toFixed(1)} ${xc.toFixed(1)},${(y + 8).toFixed(1)}" fill="#5ec6a8"/>`;
    svg += `<text x="6" y="${(y + 4).toFixed(1)}" fill="#5ec6a8" font-size="11">◆ pooled (random-effects)</text>`;
    svg += `<text x="${(W - padR).toFixed(1)}" y="${(y + 4).toFixed(1)}" fill="#5ec6a8" font-size="10" text-anchor="end">${fmt(diamond.mu_t50, 2)}h</text>`;
  }
  // prediction interval row (dashed)
  if (piT.length === 2 && piT[0] > 0 && piT[1] > 0) {
    const y = padT + rowH * (rows.length + 1) + rowH / 2;
    svg += `<line x1="${sx(piT[0]).toFixed(1)}" y1="${y.toFixed(1)}" x2="${sx(piT[1]).toFixed(1)}" y2="${y.toFixed(1)}" stroke="#e0a458" stroke-width="3" stroke-dasharray="6 4"/>`;
    svg += `<text x="6" y="${(y + 4).toFixed(1)}" fill="#e0a458" font-size="11">— 95% prediction interval (new study)</text>`;
  }
  // x-axis label
  svg += `<text x="${((padL + W - padR) / 2).toFixed(1)}" y="${(H - 8).toFixed(1)}" fill="#7e8ea0" font-size="11" text-anchor="middle">t50 (hours, log scale)</text>`;
  svg += `</svg>`;
  wrap.innerHTML = svg;
  container.appendChild(wrap);
}

// (b) HETEROGENEITY: Q (df, p), I² (%), τ²_between + a one-line read.
function renderHeterogeneity(h) {
  const card = el("div", { class: "meta-block" });
  card.appendChild(el("h4", { class: "meta-sub" }, "Heterogeneity"));
  const i2pct = h.I2 != null ? (h.I2 * 100) : null;
  const kv = el("div", { class: "meta-kv" }, [
    metaKvItem("Q", h.Q != null ? fmt(h.Q, 1) : "—"),
    metaKvItem("df", h.df != null ? String(h.df) : "—"),
    metaKvItem("p", h.p != null ? fmt(h.p, 3) : "—"),
    metaKvItem("I²", i2pct != null ? i2pct.toFixed(1) + "%" : "—"),
    metaKvItem("τ²_between", h.tau2_between != null ? fmt(h.tau2_between, 3) : "—"),
    metaKvItem("τ_between", h.tau_between != null ? fmt(h.tau_between, 3) : "—"),
  ]);
  card.appendChild(kv);
  // one-line read
  let read;
  if (i2pct != null && i2pct >= 90) {
    read = "I²=" + i2pct.toFixed(0) + "% → nearly all variation is BETWEEN-study (real " +
      "differences across labs/conditions), not within-study sampling noise. Pool the " +
      "effect with the random-effects model, not a fixed effect.";
  } else if (i2pct != null && i2pct >= 50) {
    read = "I²=" + i2pct.toFixed(0) + "% → substantial between-study heterogeneity; the " +
      "random-effects pooled estimate carries a real between-study spread τ.";
  } else if (i2pct != null) {
    read = "I²=" + i2pct.toFixed(0) + "% → most variation is within-study sampling; studies " +
      "are relatively consistent.";
  } else {
    read = "heterogeneity not computable for this protein.";
  }
  card.appendChild(el("p", { class: "section-note" }, read));
  return card;
}

// (c) VARIANCE DECOMPOSITION: between/within/measurement/construct as a stacked
// bar of fractions, + laboratory (value OR 'confounded with study') + batch
// ('unidentifiable — not recorded'). The honesty made visible.
function renderVarianceDecomposition(vd, lab) {
  const card = el("div", { class: "meta-block" });
  card.appendChild(el("h4", { class: "meta-sub" }, "Variance decomposition"));
  const fr = vd.fractions || {};
  const order = [
    ["between_study", "between-study (τ²)", "#4ea1d3"],
    ["within_study", "within-study residual", "#5ec6a8"],
    ["measurement", "measurement (digitization)", "#e0a458"],
    ["construct", "construct (WT/mutant)", "#b58cd6"],
  ];
  // stacked bar (percent width per component)
  const bar = el("div", { class: "meta-vd-bar" });
  order.forEach(([k, label, col]) => {
    const f = fr[k];
    if (f == null || f <= 0) return;
    bar.appendChild(el("div", {
      class: "meta-vd-seg",
      style: "width:" + (f * 100).toFixed(2) + "%;background:" + col + ";",
      title: label + ": " + (f * 100).toFixed(1) + "%",
    }));
  });
  card.appendChild(bar);
  // legend + fraction values
  const legend = el("div", { class: "meta-vd-legend" });
  order.forEach(([k, label, col]) => {
    const f = fr[k];
    legend.appendChild(el("div", { class: "meta-vd-key" }, [
      el("span", { class: "swatch", style: "background:" + col + ";" }),
      el("span", {}, label + ": " + (f != null ? (f * 100).toFixed(1) + "%" : "—")),
    ]));
  });
  card.appendChild(legend);
  // laboratory: value OR 'confounded with study'
  const labStatus = (lab && lab.status) || vd.laboratory;
  let labText;
  if (typeof vd.laboratory === "number") {
    labText = "laboratory variance = " + fmt(vd.laboratory, 3);
  } else if (labStatus === "confounded_with_study" || vd.laboratory === "confounded_with_study") {
    labText = "laboratory: CONFOUNDED WITH STUDY — no author/lab group spans ≥2 studies " +
      "for this protein, so a laboratory effect is not separable from the study effect " +
      "(flagged, NOT fabricated).";
  } else {
    labText = "laboratory: " + esc(String(vd.laboratory ?? labStatus ?? "—"));
  }
  card.appendChild(el("p", { class: "section-note warn-note" }, labText));
  // batch: unidentifiable — not recorded
  const batchText = (vd.batch === "unidentifiable_not_recorded" || vd.batch == null)
    ? "batch: UNIDENTIFIABLE — not recorded anywhere in the corpus, so a batch variance " +
      "component cannot be estimated (flagged, NOT fabricated)."
    : "batch: " + esc(String(vd.batch));
  card.appendChild(el("p", { class: "section-note warn-note" }, batchText));
  if (vd.note) card.appendChild(el("p", { class: "section-note" }, esc(vd.note)));
  return card;
}

// (d) LEAVE-ONE-STUDY-OUT: compact table of pooled μ when each study is dropped,
// flagging the most-influential and any outlier.
function renderLeaveOneOut(loo) {
  const card = el("div", { class: "meta-block" });
  card.appendChild(el("h4", { class: "meta-sub" }, "Leave-one-study-out (influence)"));
  if (!loo.length) {
    card.appendChild(el("p", { class: "section-note" }, "Not enough studies for a leave-one-out analysis."));
    return card;
  }
  const table = el("table", { class: "data-table meta-loo-table" });
  const thead = el("thead", {}, el("tr", {}, [
    el("th", {}, "study dropped"),
    el("th", {}, "pooled t50 (h)"),
    el("th", {}, "Δμ_log"),
    el("th", {}, "τ²"),
    el("th", {}, "I²"),
    el("th", {}, "flag"),
  ]));
  table.appendChild(thead);
  const tbody = el("tbody", {});
  loo.forEach((r) => {
    const flags = [];
    if (r.most_influential) flags.push(el("span", { class: "badge warn" }, "most influential"));
    if (r.is_outlier) flags.push(el("span", { class: "badge danger" }, "outlier"));
    const auth = r.author ? String(r.author).split(",")[0].trim() : null;
    const label = (auth ? auth + " " : "") + (r.pmid ? "(PMID " + r.pmid + ")" : (r.dropped_study || ""));
    const i2 = r.I2 != null ? (r.I2 * 100).toFixed(1) + "%" : "—";
    tbody.appendChild(el("tr", { class: (r.is_outlier ? "meta-loo-outlier" : "") }, [
      el("td", {}, esc(label)),
      el("td", {}, r.pooled_mu_t50_hours != null ? fmt(r.pooled_mu_t50_hours, 2) : "—"),
      el("td", {}, r.delta_mu_log != null ? fmt(r.delta_mu_log, 3) : "—"),
      el("td", {}, r.tau2_between != null ? fmt(r.tau2_between, 3) : "—"),
      el("td", {}, i2),
      el("td", {}, flags.length ? flags : el("span", { class: "muted" }, "—")),
    ]));
  });
  table.appendChild(tbody);
  card.appendChild(table);
  const mostInf = loo.find((r) => r.most_influential);
  const anyOut = loo.some((r) => r.is_outlier);
  card.appendChild(el("p", { class: "section-note" },
    (mostInf ? "Most-influential study: dropping it shifts the pooled μ_log by " +
      fmt(mostInf.delta_mu_log, 3) + ". " : "") +
    (anyOut ? "At least one study is flagged as an outlier." :
      "No single study is an outlier — the pooled estimate is not driven by one study.")));
  return card;
}

// (e) PUBLICATION BIAS: funnel plot (effect vs s.e.) + Egger p + small-study flag.
function renderPublicationBias(pb) {
  const card = el("div", { class: "meta-block" });
  card.appendChild(el("h4", { class: "meta-sub" }, "Publication bias"));
  const funnel = pb.funnel || {};
  const egger = pb.egger || {};
  const points = funnel.points || [];
  // Egger + small-study flag summary
  const flag = (pb.small_study_effect_flag != null ? pb.small_study_effect_flag
    : egger.small_study_effect_flag);
  const eggerBits = [
    el("span", { class: "badge" }, "Egger p = " +
      (egger.p != null ? fmt(egger.p, 3) : "n/a")),
    el("span", { class: "badge " + (flag ? "danger" : "ok") },
      flag ? "small-study effect FLAGGED" : "no small-study effect"),
  ];
  if (egger.intercept != null) {
    eggerBits.push(el("span", { class: "muted", style: "font-size:.8rem" },
      "intercept = " + fmt(egger.intercept, 2)));
  }
  if (egger.testable === false) {
    card.appendChild(el("p", { class: "section-note" },
      "Egger's test not testable for this protein (too few studies)."));
  }
  card.appendChild(el("div", { class: "meta-egger" }, eggerBits));
  // funnel plot
  if (points.length >= 2) {
    const funDiv = el("div", { class: "plot-wrap meta-funnel-plot" });
    card.appendChild(funDiv);
    renderFunnelPlot(funDiv, points, funnel.pooled_mu_log);
  } else {
    card.appendChild(el("p", { class: "section-note" }, "Not enough studies to draw a funnel plot."));
  }
  card.appendChild(el("p", { class: "section-note" },
    "Funnel: each study's log-t50 effect (x) vs its standard error (y, inverted so " +
    "precise studies sit at top). Asymmetry ⇒ possible small-study / publication bias; " +
    "the vertical line is the pooled μ_log."));
  return card;
}

// funnel scatter: effect_log (x) vs se (y, inverted). Plotly + SVG fallback.
function renderFunnelPlot(container, points, pooledMu) {
  const xs = points.map((p) => p.effect_log);
  const ses = points.map((p) => p.se);
  if (plotlyAvailable()) {
    try {
      const traces = [{
        x: xs, y: ses, mode: "markers", type: "scatter", name: "study",
        marker: { size: 9, color: "#4ea1d3", line: { color: "#0b1017", width: 1 } },
        text: points.map((p) => (p.study || "") + " · effect=" + fmt(p.effect_log, 2) + " · se=" + fmt(p.se, 3)),
        hoverinfo: "text",
      }];
      const layout = {
        paper_bgcolor: "#1a212b", plot_bgcolor: "#11171f",
        font: { color: "#9fb0c0", size: 12 },
        margin: { l: 60, r: 16, t: 12, b: 46 },
        xaxis: { title: "log-t50 effect", gridcolor: "#2d3a48", zerolinecolor: "#2d3a48" },
        yaxis: { title: "standard error", autorange: "reversed", gridcolor: "#2d3a48", zerolinecolor: "#2d3a48" },
        showlegend: false,
      };
      if (pooledMu != null) {
        const seMax = Math.max(...ses, 1e-9);
        layout.shapes = [{ type: "line", x0: pooledMu, x1: pooledMu, y0: 0, y1: seMax,
          line: { color: "#5ec6a8", width: 1.5, dash: "dash" } }];
      }
      Plotly.newPlot(container, traces, layout, { displayModeBar: false, responsive: true });
      return;
    } catch (e) { /* fall through */ }
  }
  // SVG fallback
  const wrap = el("div", { class: "plot-fallback" });
  const W = 520, H = 280, pad = 48;
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const smin = 0, smax = Math.max(...ses, 1e-9);
  const sx = (v) => pad + ((v - xmin) / ((xmax - xmin) || 1)) * (W - 2 * pad);
  const sy = (v) => pad + ((v - smin) / ((smax - smin) || 1)) * (H - 2 * pad); // inverted: se↑ => lower
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="funnel plot">`;
  svg += `<rect x="0" y="0" width="${W}" height="${H}" fill="#11171f"/>`;
  svg += `<line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="#2d3a48"/>`;
  svg += `<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}" stroke="#2d3a48"/>`;
  if (pooledMu != null) {
    const px = sx(pooledMu);
    svg += `<line x1="${px.toFixed(1)}" y1="${pad}" x2="${px.toFixed(1)}" y2="${H - pad}" stroke="#5ec6a8" stroke-width="1.5" stroke-dasharray="4 4"/>`;
  }
  points.forEach((p) => {
    svg += `<circle cx="${sx(p.effect_log).toFixed(1)}" cy="${sy(p.se).toFixed(1)}" r="4.5" fill="#4ea1d3" stroke="#0b1017"/>`;
  });
  svg += `<text x="${(W / 2).toFixed(1)}" y="${(H - 12).toFixed(1)}" fill="#7e8ea0" font-size="11" text-anchor="middle">log-t50 effect</text>`;
  svg += `<text x="14" y="${(H / 2).toFixed(1)}" fill="#7e8ea0" font-size="11" transform="rotate(-90 14 ${(H / 2).toFixed(1)})" text-anchor="middle">standard error (inverted)</text>`;
  svg += `</svg>`;
  wrap.innerHTML = svg;
  container.appendChild(wrap);
}

// (f) HONESTY FOOTER: condition-adjusted / lab-confounded / batch-unidentifiable /
// 'Bayesian'=deterministic hierarchical-normal RE with weak priors.
function renderMetaHonesty(h) {
  const card = el("div", { class: "meta-block meta-honesty honesty-banner" });
  card.appendChild(el("h4", { class: "meta-sub" }, "Honesty"));
  const list = el("ul", { class: "honesty-list" });
  const items = [
    ["Condition-adjusted", h.condition_adjustment ||
      "pooling is a random-effects meta-REGRESSION on log₁₀(concentration) + pH/temperature/" +
      "construct — NEVER raw t50 across concentrations."],
    ["Laboratory", h.laboratory ||
      "laboratory (author) effect estimable ONLY where an author group spans ≥2 studies for " +
      "the protein; else CONFOUNDED WITH STUDY (flagged, not fabricated)."],
    ["Batch", h.batch ||
      "batch is NOT RECORDED in the corpus → UNIDENTIFIABLE (flagged)."],
    ["“Bayesian”", h.bayesian ||
      "'Bayesian' = deterministic hierarchical-normal random-effects with weak priors " +
      "(half-normal on τ, wide normal on μ) → credible interval for μ + posterior-predictive " +
      "interval for a NEW study. Closed form, not MCMC."],
    ["Effect", h.effect ||
      "effect = natural-log t50 of POINT (uncensored) curves only."],
  ];
  items.forEach(([k, v]) => {
    list.appendChild(el("li", {}, [el("strong", {}, k + ": "), document.createTextNode(esc(v))]));
  });
  card.appendChild(list);
  return card;
}

// small helper: a labelled key/value chip for the meta panel
function metaKvItem(k, v) {
  return el("div", { class: "meta-kv-item" }, [
    el("span", { class: "meta-kv-k" }, k),
    el("span", { class: "meta-kv-v" }, String(v)),
  ]);
}

// --------------------------------------------------------------------------- //
// SEQUENCE ↔ STRUCTURE ↔ KINETICS (M16) — per protein
// Honesty-first: features are SEQUENCE/APR PROXIES (not 3D); native ≠ fibril;
// associations are STATISTICAL ONLY, never causal (n≤26 → mostly null).
// --------------------------------------------------------------------------- //

// honest status badge per structural feature: REAL 3D (measured from FETCHED PDB
// coordinates, derivation=pdb_3d) / REAL (the shipped APR peptides) / PROXY (a 1-D
// sequence stand-in, NOT 3D-measured) / DEFERRED (intrinsically 3D, no PDB → not
// computable at all).
const _STRUCT_STATUS = {
  real_3d: { cls: "ok", label: "REAL 3D" },
  real: { cls: "ok", label: "REAL" },
  proxy: { cls: "warn", label: "PROXY" },
  deferred: { cls: "danger", label: "DEFERRED" },
};

// The 3D tool a textbook measurement of each feature would classically use. This is
// CONTEXT for the PROXY/DEFERRED caveat only — the STATUS is always read off the
// payload block (has_real_3d / real_3d / computable), NEVER assumed here, so a real
// PDB-derived measurement can never render as a proxy and vice versa.
const _STRUCT_FEATURE_TOOL = {
  aggregation_hotspots: "the shipped APR peptides (positions + categories)",
  beta_sheet_content: "DSSP/STRIDE (β-sheet fraction)",
  solvent_accessibility: "freesasa/DSSP (per-residue SASA)",
  hydrophobic_patches: "freesasa surface (3D hydrophobic patches)",
  electrostatic_surface: "APBS/pdb2pqr (Poisson–Boltzmann surface)",
  secondary_structure: "DSSP/STRIDE (8/3-state secondary structure)",
  contact_map: "3D coordinates + a distance cutoff",
  surface_curvature: "3D surface mesh + coordinates",
};

// Read the honest status of ONE feature block straight from the payload.
//   aggregation_hotspots      → "real"      (APR-derived, derivation=apr_real)
//   {..., has_real_3d:true, real_3d:{...}}  → "real_3d" (nested 6 features)
//   contact_map w/ computable:true          → "real_3d" (marker is inline, no proxy)
//   {proxy:{...}, has_real_3d:false}        → "proxy"
//   contact_map w/ computable:false         → "deferred" (no proxy analogue exists)
function structFeatureStatus(key, block) {
  block = block || {};
  if (key === "aggregation_hotspots") return "real";
  if (block.has_real_3d === true &&
      (block.real_3d || block.derivation === "pdb_3d")) return "real_3d";
  if (block.proxy) return "proxy";
  return "deferred";
}

// Render a feature value: number → fixed, object (e.g. contact-map triple, the
// Chou–Fasman β/α pair) → "k = v · k = v", null/absent → "—".
function structValueText(v) {
  if (v == null) return "—";
  if (typeof v === "number") return fmt(v, 4);
  if (typeof v === "object") {
    const parts = Object.keys(v).map((k) => k.replace(/_/g, " ") + " = " +
      (typeof v[k] === "number" ? fmt(v[k], 4) : (v[k] == null ? "—" : String(v[k]))));
    return parts.length ? parts.join("  ·  ") : "—";
  }
  return String(v);
}

// verdict badge for the association table (association | insufficient_power | null)
const _ASSOC_VERDICT_BADGE = {
  association: "ok",
  insufficient_power: "warn",
  null: "muted-badge",
};

// a small labelled proxy-feature stat with the "sequence_apr_proxy · not 3D" badge.
function structProxyStat(label, value, note) {
  return el("div", { class: "struct-proxy-stat" }, [
    el("div", { class: "struct-proxy-n mono" }, value == null ? "—" : String(value)),
    el("div", { class: "struct-proxy-l" }, [
      document.createTextNode(label + " "),
      el("span", { class: "badge muted-badge struct-proxy-badge" }, "sequence_apr_proxy · not 3D"),
    ]),
    note ? el("div", { class: "section-note" }, esc(note)) : null,
  ]);
}

async function loadStructure(container, proteinId) {
  try {
    const d = await api("/api/structure/" + encodeURIComponent(proteinId));
    renderStructure(container, d);
  } catch (e) {
    container.innerHTML = "";
    container.appendChild(el("div", { class: "muted" },
      "Sequence↔structure↔kinetics unavailable: " + esc(e.message)));
  }
}

function renderStructure(container, d) {
  container.innerHTML = "";
  d = d || {};

  // (a) THE BANNER — states the truth for THIS protein: whether a real PDB backs the
  // 3D features or everything here is a 1-D proxy. Plus native≠fibril and
  // association-only, never causal.
  const _sfTop = d.structural_features || {};
  const _hasReal3d = _sfTop.has_real_3d_features === true;
  container.appendChild(el("div", { class: "honesty-banner struct-banner" }, [
    el("strong", {}, _hasReal3d
      ? ("Some features are MEASURED from real PDB 3D coordinates" +
         (_sfTop.real_3d_pdb_id ? " (" + esc(_sfTop.real_3d_pdb_id) + ")" : "") +
         "; the rest are SEQUENCE/APR PROXIES. Each row below says which. ")
      : "No PDB was fetched for this protein — every structural feature below is a SEQUENCE/APR-derived PROXY (not 3D). "),
    document.createTextNode(
      "Native structure ≠ the aggregation-competent/fibril state. " +
      "Associations are STATISTICAL ONLY — never causal."),
  ]));

  // The corpus association table is a CORPUS result, surfaced even when this protein
  // has no per-protein features (available === false).
  if (d.available === false) {
    container.appendChild(el("div", { class: "muted struct-none" },
      esc(d.note || "No structural-proxy features for this protein.")));
    if (d.association_table) {
      container.appendChild(el("h4", { class: "struct-sub" }, "Structure → kinetics associations (corpus)"));
      container.appendChild(renderStructAssoc(d.association_table, d.variance_ceiling));
    }
    container.appendChild(renderStructHonesty(d.honesty));
    return;
  }

  // ---- head: uniprot / protein_name / n_aprs ----
  const headBits = [];
  if (d.uniprot) {
    headBits.push(el("a", {
      href: "https://www.uniprot.org/uniprotkb/" + encodeURIComponent(d.uniprot),
      target: "_blank", class: "badge",
    }, "UniProt " + esc(d.uniprot)));
  }
  if (d.n_aprs != null) headBits.push(el("span", { class: "badge" }, esc(d.n_aprs) + " APR" + (d.n_aprs === 1 ? "" : "s")));
  headBits.push(el("span", { class: "badge ok" }, "APRs = REAL structural feature"));
  container.appendChild(el("div", { class: "struct-head" }, headBits));

  // (b) AGGREGATION HOTSPOTS (APRs) — the one REAL structural feature.
  container.appendChild(el("h4", { class: "struct-sub" }, "Aggregation hotspots (APRs) — REAL"));
  container.appendChild(renderStructHotspots(d.apr_proxy_features || {}));

  // (c) APR/SEQUENCE PROXY FEATURES — β-propensity, hydrophobicity, charge/pI.
  container.appendChild(el("h4", { class: "struct-sub" }, "APR / sequence proxy features"));
  container.appendChild(renderStructProxyFeatures(d.apr_proxy_features || {}));

  // (d) THE 8 STRUCTURAL FEATURES with honest REAL / PROXY / DEFERRED status.
  container.appendChild(el("h4", { class: "struct-sub" }, "Structural features (buildable-now vs deferred-3D)"));
  container.appendChild(renderStructFeatureGrid(d.structural_features || {}));

  // (e) GNN GRAPH SUMMARY — a representation for a FUTURE GNN (no GNN is built).
  container.appendChild(el("h4", { class: "struct-sub" }, "GNN graph summary (representation for a future GNN)"));
  container.appendChild(renderStructGraph(d.graph_summary || {}));

  // (f) STRUCTURE SOURCES — PDB/AlphaFold/CryoEM/NMR + native-vs-fibril.
  container.appendChild(el("h4", { class: "struct-sub" }, "Structure sources"));
  container.appendChild(renderStructSources(d.structure_sources || {}));

  // (g) STRUCTURE → KINETICS ASSOCIATION TABLE (corpus-level cards).
  container.appendChild(el("h4", { class: "struct-sub" }, "Structure → kinetics associations (corpus)"));
  container.appendChild(renderStructAssoc(d.association_table || {}, d.variance_ceiling || {}));

  // (h) HONESTY FOOTER.
  container.appendChild(renderStructHonesty(d.honesty));
}

// (b) Aggregation hotspots (APRs) — positions / categories / peptides.
function renderStructHotspots(apf) {
  const hotspots = apf.aggregation_hotspots || [];
  const catCounts = apf.apr_category_counts || {};
  const wrap = el("div", { class: "struct-hotspots" });
  // category count chips
  const cats = Object.keys(catCounts);
  if (cats.length) {
    const chips = el("div", { class: "pill-row" });
    cats.forEach((c) => chips.appendChild(
      el("span", { class: "badge" }, esc(c) + " ×" + esc(catCounts[c]))));
    wrap.appendChild(chips);
  }
  if (!hotspots.length) {
    wrap.appendChild(el("p", { class: "section-note" }, "No APR peptides shipped for this protein."));
    return wrap;
  }
  const table = el("table", { class: "data-table struct-hotspot-table" });
  table.appendChild(el("thead", {}, el("tr", {}, [
    el("th", {}, "Position"),
    el("th", {}, "Length"),
    el("th", {}, "Category"),
    el("th", {}, "Peptide (region)"),
  ])));
  const tbody = el("tbody", {});
  hotspots.forEach((h) => {
    tbody.appendChild(el("tr", {}, [
      el("td", { class: "mono" }, esc(h.position != null ? h.position : "—")),
      el("td", { class: "num" }, h.length != null ? esc(h.length) : "—"),
      el("td", {}, esc(h.category || "—")),
      el("td", { class: "mono" }, esc(h.region || "—")),
    ]));
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  wrap.appendChild(el("p", { class: "section-note" },
    "APR positions / categories / peptides are the one REAL structural feature (the shipped APR peptides)."));
  return wrap;
}

// (c) APR / sequence proxy features: β-propensity (Chou–Fasman), hydrophobicity +
// max patch (Kyte–Doolittle), net charge / pI (electrostatics proxy).
function renderStructProxyFeatures(apf) {
  const wrap = el("div", {});
  const grid = el("div", { class: "struct-proxy-grid" }, [
    structProxyStat("β-sheet propensity (Chou–Fasman, mean)", fmt(apf.beta_propensity_mean, 3)),
    structProxyStat("β-sheet propensity (max)", fmt(apf.beta_propensity_max, 3)),
    structProxyStat("hydrophobicity (Kyte–Doolittle, mean)", fmt(apf.hydrophobicity_mean, 3)),
    structProxyStat("max hydrophobic patch (K–D 3-res window)", fmt(apf.hydrophobic_patch_max, 3)),
    structProxyStat("net charge (pH7 proxy, total)", fmt(apf.net_charge_total, 2)),
    structProxyStat("crude pI proxy (mean)", fmt(apf.crude_pI_proxy_mean, 2)),
    structProxyStat("APR coverage proxy", fmt(apf.apr_coverage_proxy, 3), apf.apr_coverage_note),
  ]);
  wrap.appendChild(grid);
  const scales = apf.scales_cited || {};
  const scaleKeys = Object.keys(scales);
  if (scaleKeys.length) {
    wrap.appendChild(el("p", { class: "section-note" },
      "Scales: " + scaleKeys.map((k) => esc(scales[k])).join("  ·  ")));
  }
  wrap.appendChild(el("p", { class: "section-note" },
    "Every number in THIS panel is 1-D sequence/APR-derived (derivation = " +
    esc(apf.derivation || "sequence_apr_proxy") + ", is_3d_derived = " +
    esc(String(apf.is_3d_derived)) + ") — NOT 3D-measured. Any genuinely 3D " +
    "measurement for this protein appears in the structural-features table below, " +
    "labelled REAL 3D with its own fidelity."));
  return wrap;
}

// (d) the 8 structural features with honest REAL 3D / REAL / PROXY / DEFERRED status
// badges read PER FEATURE off the payload. A value MEASURED from fetched PDB
// coordinates is labelled as such and carries its `fidelity` caveat; a 1-D sequence
// stand-in is labelled NOT 3D-measured. Where both exist the proxy is kept visible
// beside the real value so the contrast is legible — never silently swapped.
function renderStructFeatureGrid(sf) {
  const wrap = el("div", {});
  const hasReal3d = sf.has_real_3d_features === true;

  // container-level stamp: which PDB (if any) backs the REAL-3D column, and whether
  // it is a native monomer or a fibril (native ≠ the aggregation-competent state).
  const stamp = el("div", { class: "pill-row struct-3d-stamp" });
  if (hasReal3d) {
    stamp.appendChild(el("span", { class: "badge ok" },
      "REAL 3D from fetched PDB coordinates"));
    if (sf.real_3d_pdb_id) {
      stamp.appendChild(el("a", {
        href: "https://www.rcsb.org/structure/" + encodeURIComponent(sf.real_3d_pdb_id),
        target: "_blank", class: "badge",
      }, "PDB " + esc(sf.real_3d_pdb_id)));
    }
    const nof = sf.real_3d_native_or_fibril || "unknown";
    stamp.appendChild(el("span", { class: "badge " + (nof === "fibril" ? "ok" : "warn") },
      "measured state: " + esc(nof)));
  } else {
    stamp.appendChild(el("span", { class: "badge warn" },
      "NO fetched PDB — every 3D feature below is a 1-D sequence PROXY, not measured"));
  }
  wrap.appendChild(stamp);

  if (sf.derivation_summary) {
    wrap.appendChild(el("p", { class: "section-note struct-derivation" }, esc(sf.derivation_summary)));
  }
  const order = ["aggregation_hotspots", "beta_sheet_content", "solvent_accessibility",
    "hydrophobic_patches", "electrostatic_surface", "secondary_structure",
    "contact_map", "surface_curvature"];
  const table = el("table", { class: "data-table struct-feature-table" });
  table.appendChild(el("thead", {}, el("tr", {}, [
    el("th", {}, "Feature"),
    el("th", {}, "Status"),
    el("th", {}, "Value"),
    el("th", {}, "How it was obtained / honesty caveat"),
  ])));
  const tbody = el("tbody", {});
  let nReal3d = 0;
  let nProxy = 0;
  order.forEach((key) => {
    const block = sf[key] || {};
    const status = structFeatureStatus(key, block);
    const badge = _STRUCT_STATUS[status] || _STRUCT_STATUS.proxy;
    const tool = _STRUCT_FEATURE_TOOL[key] || "3D coordinates";
    const proxy = block.proxy || null;
    const valueCell = el("td", { class: "struct-feature-value" });
    const detailCell = el("td", { class: "struct-feature-detail" });

    if (status === "real_3d") {
      nReal3d += 1;
      // contact_map carries its pdb_3d marker INLINE (it has no proxy analogue);
      // the other five nest it under `real_3d`.
      const r = block.real_3d || block;
      valueCell.appendChild(el("div", { class: "struct-val-real mono" },
        structValueText(r.value)));
      detailCell.appendChild(el("div", { class: "struct-real-line" }, [
        el("strong", {}, "MEASURED from PDB 3D coordinates"),
        document.createTextNode(" — derivation = " + esc(r.derivation || "pdb_3d") +
          ", is_3d_derived = true" +
          (r.real_from ? ", from " + esc(r.real_from) : "") + "."),
      ]));
      detailCell.appendChild(el("div", { class: "section-note struct-fidelity" }, [
        el("strong", {}, "fidelity: "),
        document.createTextNode(esc(r.fidelity || "unspecified — treat with caution")),
      ]));
      // keep the 1-D proxy VISIBLE beside the real value, explicitly marked not-3D,
      // so a reader can see how far the sequence stand-in is from the measurement.
      if (proxy) {
        valueCell.appendChild(el("div", { class: "struct-val-proxy mono" },
          "proxy: " + structValueText(proxy.value)));
        detailCell.appendChild(el("div", { class: "section-note" },
          "1-D proxy shown for contrast — " + esc(proxy.proxy_of || tool) +
          " (derivation = " + esc(proxy.derivation || "sequence_apr_proxy") +
          ", is_3d_derived = false): NOT 3D-measured."));
      }
      if (r.native_vs_fibril_note) {
        // NB: no esc() — el() puts strings through createTextNode (already safe), and
        // esc() would render the note's "structure->kinetics" as "structure-&gt;kinetics".
        detailCell.appendChild(el("div", { class: "section-note" },
          String(r.native_vs_fibril_note)));
      }
    } else if (status === "proxy") {
      nProxy += 1;
      const p = proxy || {};
      valueCell.appendChild(el("div", { class: "struct-val-proxy mono" },
        structValueText(p.value)));
      detailCell.appendChild(el("div", { class: "struct-proxy-line" }, [
        el("strong", {}, "NOT 3D-measured"),
        document.createTextNode(" — 1-D sequence/APR proxy" +
          (p.proxy_of ? ": " + esc(p.proxy_of) : "") +
          " (derivation = " + esc(p.derivation || "sequence_apr_proxy") +
          ", is_3d_derived = false)."),
      ]));
      if (p.real_feature_deferred) {
        detailCell.appendChild(el("div", { class: "section-note" },
          "the real 3D feature (" + esc(p.real_feature_deferred) + ") needs " +
          esc(p.real_feature_requires || tool) + " — no PDB fetched for this protein."));
      }
    } else if (status === "real") {
      // aggregation_hotspots: REAL, but from the shipped APR peptides — 1-D, and
      // that must stay explicit (real ≠ 3D-derived).
      const hs = block.value || [];
      valueCell.appendChild(el("div", { class: "struct-val-real mono" },
        (Array.isArray(hs) ? hs.length : 0) + " APR" +
        (Array.isArray(hs) && hs.length === 1 ? "" : "s")));
      detailCell.appendChild(el("div", {}, esc(block.note || tool)));
      detailCell.appendChild(el("div", { class: "section-note" },
        "derivation = " + esc(block.derivation || "apr_real") +
        ", is_3d_derived = false (REAL, but sequence-level — not a 3D measurement)."));
    } else {
      // DEFERRED: intrinsically 3D and no PDB → not computable, and there is no
      // proxy analogue to stand in for it. Show nothing rather than a number.
      valueCell.appendChild(el("div", { class: "struct-val-none mono" }, "—"));
      detailCell.appendChild(el("div", { class: "struct-deferred-line" }, [
        el("strong", {}, "NOT COMPUTABLE"),
        document.createTextNode(" — " + esc(block.note ||
          "no fetched PDB for this protein (or fetch/parse failed).")),
      ]));
      detailCell.appendChild(el("div", { class: "section-note" },
        "needs " + esc(block.real_feature_requires || tool) +
        "; no 1-D analogue exists (intrinsically 3D)."));
    }

    tbody.appendChild(el("tr", { class: "struct-feature-row status-" + status }, [
      el("td", {}, esc(key.replace(/_/g, " "))),
      el("td", {}, el("span", { class: "badge " + badge.cls }, badge.label)),
      valueCell,
      detailCell,
    ]));
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  wrap.appendChild(el("p", { class: "section-note" }, hasReal3d
    ? (nReal3d + " feature" + (nReal3d === 1 ? "" : "s") +
       " MEASURED from real PDB 3D coordinates (each stamped with its own fidelity — " +
       "exact / exact-algorithm / approximation); " + nProxy + " still 1-D PROXY. " +
       "aggregation_hotspots is REAL but sequence-level. A native structure is NOT " +
       "the aggregation-competent/fibril state.")
    : ("NO real 3D for this protein: aggregation_hotspots is REAL (sequence-level) " +
       "and the rest are 1-D sequence/APR PROXIES — NOT 3D-measured. contact_map is " +
       "intrinsically 3D, so it is reported as not computable rather than faked.")));
  return wrap;
}

// (e) GNN graph summary — n_nodes / n_edges / native_or_fibril + the contact-edges
// caveat. A representation for a FUTURE GNN (no GNN is built).
function renderStructGraph(g) {
  const wrap = el("div", { class: "struct-graph" });
  const chips = el("div", { class: "pill-row" }, [
    el("span", { class: "badge" }, (g.n_nodes != null ? g.n_nodes : "—") + " nodes"),
    el("span", { class: "badge" }, (g.n_edges != null ? g.n_edges : "—") + " edges"),
    el("span", { class: "badge" }, "native/fibril: " + esc(g.native_or_fibril || "unknown")),
    el("span", { class: "badge " + (g.contact_edges_available ? "ok" : "danger") },
      g.contact_edges_available ? "contact edges present" : "contact edges unavailable (need coordinates)"),
  ]);
  wrap.appendChild(chips);
  const edgeTypes = g.edge_types_present || [];
  if (edgeTypes.length) {
    wrap.appendChild(el("p", { class: "section-note" },
      "Edge types present: " + edgeTypes.map(esc).join(", ")));
  }
  const schema = g.node_feature_schema || [];
  if (schema.length) {
    wrap.appendChild(el("p", { class: "section-note" },
      "Node feature schema: " + schema.map(esc).join(", ")));
  }
  if (g.note) wrap.appendChild(el("p", { class: "section-note" }, esc(g.note)));
  return wrap;
}

// (f) Structure sources — PDB/AlphaFold/CryoEM/NMR refs + native-vs-fibril.
function renderStructSources(ss) {
  const wrap = el("div", { class: "struct-sources" });
  const ids = ss.structure_ids || [];
  const sourceCounts = ss.source_counts || {};
  const nvfCounts = ss.native_or_fibril_counts || {};
  if (!ss.has_pdb_reference && !ids.length && !(ss.n_structures > 0)) {
    wrap.appendChild(el("p", { class: "section-note" },
      "No structure reference for this protein (no PDB/AlphaFold/CryoEM/NMR ref; no coordinates)."));
    wrap.appendChild(el("p", { class: "section-note" }, esc(ss.native_vs_fibril_note || "")));
    return wrap;
  }
  const chips = el("div", { class: "pill-row" });
  ids.forEach((id) => chips.appendChild(el("a", {
    href: "https://www.rcsb.org/structure/" + encodeURIComponent(id),
    target: "_blank", class: "badge",
  }, esc(id))));
  Object.keys(sourceCounts).forEach((s) => chips.appendChild(
    el("span", { class: "badge" }, esc(s) + " ×" + esc(sourceCounts[s]))));
  Object.keys(nvfCounts).forEach((k) => chips.appendChild(
    el("span", { class: "badge " + (k === "fibril" ? "ok" : "warn") },
      esc(k) + " ×" + esc(nvfCounts[k]))));
  wrap.appendChild(chips);
  wrap.appendChild(el("p", { class: "section-note" },
    (ss.has_coordinates ? "" : "PDB ref only, no coordinates bundled. ") +
    esc(ss.native_vs_fibril_note || "")));
  return wrap;
}

// (g) Structure → kinetics association table (corpus cards). Every row shows
// causal:false; the headline reads "0/24 survive FDR at n=26 — no association".
function renderStructAssoc(at, vc) {
  const wrap = el("div", { class: "struct-assoc" });
  at = at || {};
  const cards = at.association_cards || [];
  const cov = at.coverage || {};
  const nSig = at.n_significant_after_fdr;
  const nCards = at.n_cards != null ? at.n_cards : cards.length;

  // the prominent read
  const nProteins = cov.n_proteins_with_apr_proxy_in_kinetics || 26;
  wrap.appendChild(el("div", { class: "struct-assoc-headline" }, [
    el("strong", {}, (nSig != null ? nSig : 0) + "/" + nCards +
      " survive FDR at n=" + nProteins + " proteins — no association; not causal."),
  ]));
  if (cov.banner) {
    wrap.appendChild(el("p", { class: "section-note" }, esc(cov.banner)));
  }

  if (!cards.length) {
    wrap.appendChild(el("p", { class: "section-note" }, "No association cards available."));
    return wrap;
  }

  const table = el("table", { class: "data-table struct-assoc-table" });
  table.appendChild(el("thead", {}, el("tr", {}, [
    el("th", {}, "Feature → target"),
    el("th", {}, "Spearman ρ"),
    el("th", {}, "perm p"),
    el("th", {}, "FDR q"),
    el("th", {}, "n"),
    el("th", {}, "LOPO skill"),
    el("th", {}, "causal"),
    el("th", {}, "Verdict"),
  ])));
  const tbody = el("tbody", {});
  cards.forEach((c) => {
    const vBadge = _ASSOC_VERDICT_BADGE[c.verdict] || "muted-badge";
    const skill = c.lopo_skill_over_baseline != null ? c.lopo_skill_over_baseline : c.lopo_r2;
    tbody.appendChild(el("tr", {}, [
      el("td", {}, [
        el("span", { class: "mono" }, esc(c.feature || "?")),
        document.createTextNode(" → "),
        el("span", { class: "mono" }, esc(c.target || "?")),
      ]),
      el("td", { class: "num" }, fmt(c.spearman_rho, 3)),
      el("td", { class: "num" }, fmt(c.permutation_p, 3)),
      el("td", { class: "num" }, fmt(c.fdr_q, 3)),
      el("td", { class: "num" }, c.n_proteins != null ? esc(c.n_proteins) : "—"),
      el("td", { class: "num" }, fmt(skill, 3)),
      el("td", {}, el("span", { class: "badge danger" }, "causal:false")),
      el("td", {}, el("span", { class: "badge " + vBadge }, esc(c.verdict || "null"))),
    ]));
  });
  table.appendChild(tbody);
  wrap.appendChild(table);

  // variance ceiling read
  vc = vc || {};
  if (vc.between_protein_fraction != null) {
    wrap.appendChild(el("p", { class: "section-note struct-vc" }, [
      el("strong", {}, "Variance ceiling: "),
      document.createTextNode(
        "a sequence-only predictor is a per-protein CONSTANT — it can explain at most " +
        pct(vc.between_protein_fraction) + " of log₁₀(t50) variance (" +
        pct(vc.within_protein_fraction) + " is condition-driven, within-protein). " +
        esc(vc.interpretation || "")),
    ]));
  }
  if (at.assoc_honesty) {
    wrap.appendChild(el("p", { class: "section-note" }, esc(at.assoc_honesty)));
  }
  return wrap;
}

// (h) Honesty footer: association-only/never-causal, proxy-not-3D, native-vs-fibril,
// n-limited, 3D deferred, GNN for later.
function renderStructHonesty(honesty) {
  const card = el("div", { class: "struct-honesty honesty-banner" });
  card.appendChild(el("h4", { class: "struct-sub" }, "Honesty"));
  const list = el("ul", { class: "honesty-list" });
  const items = (honesty && honesty.length) ? honesty : [
    "ASSOCIATION-ONLY, NEVER causal: causal:false is an invariant on every association; experimentally_validated is an external human flag only.",
    "REAL 3D where a PDB was fetched + parsed: contact map, Shrake-Rupley SASA, φ/ψ secondary structure, hydrophobic surface patches, electrostatic surface, surface curvature — stamped derivation=pdb_3d, is_3d_derived=True, with a per-feature fidelity (EXACT / EXACT ALGORITHM / APPROXIMATION).",
    "PROXY otherwise: with NO fetched PDB (has_real_3d=False) those features fall back to 1-D SEQUENCE/APR analogues, stamped is_3d_derived=False (NOT 3D-measured) — the fallback is FLAGGED, never presented as a measurement.",
    "NATIVE-vs-FIBRIL: a native-monomer structure ≠ the aggregation-competent/fibril state; MOST fetched PDBs here are native monomers, so structure→kinetics is ASSOCIATIVE and often confounded.",
    "Effective n = PROTEINS (≤26), and the REAL-3D subset is smaller still, so those tests are MORE underpowered: most verdicts are insufficient_power/null.",
    "contact_map is intrinsically 3D — with no PDB it is reported computable:false rather than given a fabricated 1-D stand-in.",
    "The GNN graph is a forward-compatible REPRESENTATION; NO GNN is built (real contact edges are populated only where a PDB was parsed).",
  ];
  items.forEach((t) => list.appendChild(el("li", {}, esc(t))));
  card.appendChild(list);
  return card;
}

// --------------------------------------------------------------------------- //
// DOSE–RESPONSE (k_agg vs concentration) — per protein with CPAD R-rows
// --------------------------------------------------------------------------- //
async function loadDoseResponse(container, proteinId) {
  try {
    const d = await api("/api/dose-response/" + encodeURIComponent(proteinId));
    if (d.error || d.available === false) {
      container.innerHTML = "";
      container.appendChild(el("div", { class: "muted" },
        "Dose–response unavailable: " + esc(d.error || d.reason || "not built")));
      return;
    }
    renderDoseResponse(container, d);
  } catch (e) {
    container.innerHTML = "";
    container.appendChild(el("div", { class: "muted" }, "Dose–response failed: " + esc(e.message)));
  }
}

function renderDoseResponse(container, d) {
  container.innerHTML = "";
  const models = d.models || [];
  const points = d.points || [];

  // header badges
  container.appendChild(el("div", { class: "dose-head" }, [
    el("span", { class: "badge" }, points.length + " (conc, k_agg) point" + (points.length === 1 ? "" : "s")),
    d.best_by_aicc ? el("span", { class: "badge ok" }, "best (AICc): " + esc(d.best_by_aicc)) : null,
  ]));

  // scatter of k_agg vs [monomer] with the AICc-best model overlaid
  if (points.length >= 2) {
    const plotDiv = el("div", { class: "plot-wrap dose-plot" });
    container.appendChild(plotDiv);
    renderDoseScatter(plotDiv, d);
  } else {
    container.appendChild(el("p", { class: "section-note" },
      "Not enough (concentration, k_agg) points to draw the scatter."));
  }

  // ranked model table: model · R² · AICc · best
  if (models.length) {
    const tbl = el("table", { class: "dose-table" });
    tbl.appendChild(el("tr", {}, ["model", "R²", "AICc", "best"].map((h) => el("th", {}, h))));
    models.forEach((m) => {
      tbl.appendChild(el("tr", { class: m.best ? "dose-best" : "", title: m.note || "" }, [
        el("td", { class: "mono" }, esc(m.model)),
        el("td", { class: "num" }, m.r2 != null ? fmt(m.r2, 3) : "—"),
        el("td", { class: "num" }, m.aicc != null ? fmt(m.aicc, 2) : "—"),
        el("td", {}, m.best ? el("span", { class: "badge ok" }, "✓") : ""),
      ]));
    });
    container.appendChild(tbl);
  }

  // honest explainer: these are CPAD's precomputed k_agg (R-rows), NOT the engine's γ
  container.appendChild(el("p", { class: "section-note" }, esc(d.explainer || "")));
  if (d.honesty) {
    container.appendChild(el("p", { class: "section-note warn-note" }, esc(d.honesty)));
  }
}

// k_agg vs [monomer] scatter + AICc-best dose model overlay (Plotly + SVG fallback)
function renderDoseScatter(container, d) {
  const points = (d.points || []).slice().sort(
    (a, b) => a.concentration_uM - b.concentration_uM);
  const ov = d.overlay;

  if (plotlyAvailable()) {
    try {
      const traces = [{
        x: points.map((p) => p.concentration_uM), y: points.map((p) => p.k_agg),
        mode: "markers", type: "scatter", name: "CPAD k_agg (R-rows)",
        marker: { size: 9, color: "#4ea1d3" },
        text: points.map((p) => (p.pmid ? "PMID " + p.pmid : "")),
      }];
      if (ov && ov.x && ov.y) {
        traces.push({
          x: ov.x, y: ov.y, mode: "lines", type: "scatter",
          name: "best fit: " + esc(ov.model) + (ov.r2 != null ? " (R²=" + fmt(ov.r2, 2) + ")" : ""),
          line: { color: "#5ec6a8", width: 2 },
        });
      }
      const layout = {
        paper_bgcolor: "#1a212b", plot_bgcolor: "#11171f",
        font: { color: "#9fb0c0", size: 12 },
        margin: { l: 60, r: 16, t: 12, b: 46 },
        xaxis: { title: "[monomer] (µM)", gridcolor: "#2d3a48", zerolinecolor: "#2d3a48" },
        yaxis: { title: "k_agg (/h)", gridcolor: "#2d3a48", zerolinecolor: "#2d3a48" },
        legend: { orientation: "h", y: -0.25, font: { size: 11 } },
        showlegend: true,
      };
      Plotly.newPlot(container, traces, layout, { displayModeBar: false, responsive: true });
      return;
    } catch (e) { /* fall through to SVG */ }
  }
  // SVG fallback
  renderCurveFallback(container, {
    x: points.map((p) => p.concentration_uM),
    y: points.map((p) => p.k_agg),
    fitted: (ov && ov.x ? { x: ov.x, y: ov.y, model: ov.model } : null),
    censoring: "none",
    xlabel: "[monomer] (uM)", ylabel: "k_agg (/h)",
  });
}

function unitNPoints(u) {
  // # observed points: full units carry shape_confidence.n_points; fall back to
  // the descriptive_regime confidence; compact units carry neither.
  const sc = u.shape_confidence || (u.descriptive_regime || {}).confidence;
  return sc && sc.n_points != null ? sc.n_points : null;
}

// Multi-select state (M0 cohort): the set of series_ids the user has ticked in the
// dataset picker for "analyze together". Reset whenever the protein changes.
let cohortSelected = new Set();

function buildDatasetPicker(d) {
  const units = d.units || [];
  const wrap = el("div", { class: "card dataset-picker" });
  wrap.appendChild(el("h3", {}, "CPAD datasets for this protein — click one, or tick several to analyze together"));
  wrap.appendChild(el("p", { class: "explainer" },
    units.length + " real CPAD 2.0 dataset" + (units.length === 1 ? "" : "s") +
    " for " + esc(d.protein_id) + ". CLICK any row for PRISE's single-curve analysis, or " +
    "TICK several and press “Analyze selected together” to assemble a cohort — M0 detects " +
    "the data mode (replicates / concentration-series / condition-series / confounded), " +
    "gates comparability, and routes to the licensed analysis. No upload needed."));

  // "Analyze selected together (N)" control bar
  const btn = el("button", { class: "cohort-btn", disabled: "true" },
    "Analyze selected together (0)");
  const selectAll = el("button", { class: "cohort-btn-secondary" }, "Select all");
  const clearBtn = el("button", { class: "cohort-btn-secondary" }, "Clear");
  const bar = el("div", { class: "cohort-bar" }, [btn, selectAll, clearBtn,
    el("span", { class: "cohort-hint muted" },
      "Tip: tick ≥2 identical-condition runs → replicates; ≥3 varying-concentration → concentration-series.")]);

  function refreshBtn() {
    const n = cohortSelected.size;
    btn.textContent = "Analyze selected together (" + n + ")";
    if (n >= 1) btn.removeAttribute("disabled");
    else btn.setAttribute("disabled", "true");
  }

  const scroll = el("div", { class: "picker-scroll" });
  const tbl = el("table", { class: "picker-table" });
  const head = el("tr", {}, ["", "", "conc (µM)", "pH", "temp (°C)", "assay", "# pts",
    "censoring", "yield", "regime"].map((h) => el("th", {}, h)));
  tbl.appendChild(head);

  units.forEach((u, i) => {
    const cv = u.condition_vector || {};
    const conc = (cv.concentration && cv.concentration.value_uM != null)
      ? fmt(cv.concentration.value_uM) : "—";
    const assayTxt = (cv.assay_type || "?") + (cv.assay_reports_mass ? " · mass" : " · non-mass");
    const np = unitNPoints(u);
    const iy = u.information_yield || "uninformative";
    const cclass = u.censoring_class && u.censoring_class !== "none" ? u.censoring_class : "none";
    const reg = (u.descriptive_regime || {}).regime || "—";
    const sid = u.series_id;

    // per-row checkbox (multi-select for the cohort). Clicking it must NOT trigger
    // the row's single-curve select (stopPropagation).
    const cb = el("input", { type: "checkbox", class: "cohort-cb", title: "add to cohort" });
    if (cohortSelected.has(sid)) cb.checked = true;
    cb.addEventListener("click", (ev) => ev.stopPropagation());
    cb.addEventListener("change", () => {
      if (cb.checked) cohortSelected.add(sid); else cohortSelected.delete(sid);
      refreshBtn();
    });

    const row = el("tr", {
      class: "picker-row" + (i === currentUnitIdx ? " active" : ""),
      title: u.series_id,
    }, [
      el("td", { class: "pick-cb" }, cb),
      el("td", { class: "pick-marker" }, i === currentUnitIdx ? "▶" : ""),
      el("td", { class: "num" }, conc),
      el("td", { class: "num" }, cv.pH != null ? fmt(cv.pH) : "—"),
      el("td", { class: "num" }, cv.temperature_C != null ? fmt(cv.temperature_C) : "—"),
      el("td", {}, esc(assayTxt)),
      el("td", { class: "num" }, np != null ? String(np) : "—"),
      el("td", {}, cclass === "none"
        ? el("span", { class: "badge ok" }, "none")
        : el("span", { class: "badge censor" }, esc(cclass))),
      el("td", {}, el("span", { class: "badge tier-" + iy }, esc(iy))),
      el("td", { class: "regime-cell" }, esc(reg)),
    ]);
    row.addEventListener("click", () => {
      if (i === currentUnitIdx) return;
      currentUnitIdx = i;
      renderProteinDetail();
    });
    tbl.appendChild(row);
  });

  btn.addEventListener("click", () => {
    const ids = units.map((u) => u.series_id).filter((s) => cohortSelected.has(s));
    if (ids.length) runCohort(ids, d.protein_id);
  });
  selectAll.addEventListener("click", () => {
    units.forEach((u) => cohortSelected.add(u.series_id));
    renderProteinDetail();
  });
  clearBtn.addEventListener("click", () => {
    cohortSelected.clear();
    renderProteinDetail();
  });

  refreshBtn();
  wrap.appendChild(bar);
  scroll.appendChild(tbl);
  wrap.appendChild(scroll);
  // the cohort analysis panel renders here (populated by runCohort)
  wrap.appendChild(el("div", { id: "cohort-panel" }));
  return wrap;
}

// --------------------------------------------------------------------------- //
// M0 cohort analysis panel (multi-dataset assembly)
// --------------------------------------------------------------------------- //
const COHORT_MODE_LABEL = {
  single_curve: "SINGLE CURVE", replicates: "REPLICATES",
  concentration_series: "CONCENTRATION SERIES", condition_series: "CONDITION SERIES",
  confounded: "CONFOUNDED", empty: "EMPTY",
};

async function runCohort(ids, protein) {
  const panel = $("#cohort-panel");
  if (!panel) return;
  panel.innerHTML = "";
  panel.appendChild(el("div", { class: "card cohort-card" }, [
    el("div", { class: "loading" }, "Assembling cohort of " + ids.length + " dataset(s)…"),
  ]));
  try {
    const r = await fetch("/api/cohort", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ series_ids: ids }),
    });
    const res = await r.json();
    renderCohort(res);
  } catch (e) {
    panel.innerHTML = "";
    panel.appendChild(el("div", { class: "card cohort-card" }, [
      el("div", { class: "error-banner" }, "Cohort assembly failed: " + esc(e.message)),
    ]));
  }
}

function renderCohort(res) {
  const panel = $("#cohort-panel");
  panel.innerHTML = "";
  const card = el("div", { class: "card cohort-card" });

  if (res.error) {
    card.appendChild(el("div", { class: "error-banner" }, "Cohort error: " + esc(res.error)));
    panel.appendChild(card);
    return;
  }

  const mode = res.data_mode || "empty";
  const modeLabel = COHORT_MODE_LABEL[mode] || mode.toUpperCase();
  const comp = res.comparability || {};

  // ---- header: data_mode badge + reason -----
  card.appendChild(el("div", { class: "cohort-head" }, [
    el("h3", {}, "Cohort analysis — " + (res.n_selected || 0) + " dataset(s) assembled"),
    el("span", { class: "cohort-mode-badge mode-" + mode }, modeLabel),
  ]));
  if (res.data_mode_reason)
    card.appendChild(el("p", { class: "cohort-reason" }, esc(res.data_mode_reason)));

  // ---- comparability verdict (honesty-first, always visible) -----
  card.appendChild(renderComparability(comp));

  // ---- the routed assembled analysis -----
  const a = res.assembled_analysis || {};
  card.appendChild(renderAssembled(mode, a, res));

  // ---- window of validity (always visible; inference not licensed beyond) -----
  card.appendChild(renderWindowOfValidity(res.window_of_validity || {}));

  // ---- members -----
  card.appendChild(renderCohortMembers(res.members || []));

  panel.appendChild(card);
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderComparability(comp) {
  const ok = comp.ok === true;
  const box = el("div", { class: "cohort-comp " + (ok ? "comp-ok" : "comp-fail") });
  box.appendChild(el("div", { class: "comp-verdict" }, [
    el("span", { class: "badge " + (ok ? "ok" : "danger") },
      ok ? "COMPARABILITY OK" : "COMPARABILITY CONFOUND"),
    el("span", { class: "muted" }, " " + esc(comp.note || "")),
  ]));
  const mm = comp.mismatches || [];
  if (mm.length) {
    const ul = el("ul", { class: "comp-list" });
    mm.forEach((m) => {
      ul.appendChild(el("li", {}, [
        el("strong", {}, esc(m.field)),
        document.createTextNode(" mixes " + (m.distinct_values || []).map(esc).join(", ") +
          " — " + esc(m.note || "")),
      ]));
    });
    box.appendChild(ul);
  }
  const deg = comp.degraded_fields || [];
  if (deg.length) {
    const ul = el("ul", { class: "comp-list degraded" });
    deg.forEach((dg) => {
      ul.appendChild(el("li", {}, esc(dg.field) + ": " + esc(dg.note || "")));
    });
    box.appendChild(el("div", { class: "muted", style: "margin-top:.3rem" },
      "Degraded (unknown) comparability fields — confidence widened, never silently merged:"));
    box.appendChild(ul);
  }
  return box;
}

function renderAssembled(mode, a, res) {
  const box = el("div", { class: "cohort-assembled" });
  if (a.status === "degraded" || a.status === "insufficient_data") {
    box.appendChild(el("div", { class: "warn-banner" },
      esc(a.route || mode) + ": " + esc(a.reason || a.status)));
    return box;
  }

  if (mode === "confounded") {
    // prominent refusal + descriptive-only
    const flag = a.confound_flag || {};
    box.appendChild(el("div", { class: "cohort-refusal" }, [
      el("div", { class: "refusal-title" }, "⚠ MECHANISM / SCALING REFUSED"),
      el("p", {}, esc(flag.message || a.refusal_reason ||
        "multiple conditions co-vary — γ is confounded")),
      el("p", { class: "muted" }, esc(flag.invariant || "Invariant 19")),
      (a.confound_axes && a.confound_axes.length)
        ? el("p", {}, [el("strong", {}, "Co-varying axes: "),
          document.createTextNode((a.confound_axes || []).join(", "))]) : null,
    ]));
    const rows = a.per_curve_descriptive || [];
    if (rows.length) {
      box.appendChild(el("h4", {}, "Descriptive-only per curve (no mechanism, no γ):"));
      const tbl = el("table", { class: "cohort-table" });
      tbl.appendChild(el("tr", {}, ["series", "conc (µM)", "pH", "regime", "model"].map((h) => el("th", {}, h))));
      rows.forEach((rr) => {
        const c = rr.conditions || {};
        tbl.appendChild(el("tr", {}, [
          el("td", { class: "mono" }, esc(rr.series_id)),
          el("td", { class: "num" }, c.concentration_uM != null ? fmt(c.concentration_uM) : "—"),
          el("td", { class: "num" }, c.pH != null ? fmt(c.pH) : "—"),
          el("td", {}, esc(rr.descriptive_regime || "—")),
          el("td", {}, esc(rr.best_model || "—")),
        ]));
      });
      box.appendChild(tbl);
    }
    return box;
  }

  if (mode === "replicates") {
    box.appendChild(el("h4", {}, "Replicate random-effects meta-analysis (§3-M1)"));
    box.appendChild(el("p", { class: "explainer" },
      "Replicates are fit INDIVIDUALLY then combined by DerSimonian–Laird random effects " +
      "(within-fit + between-replicate variance) — never curve-averaged. The between-replicate " +
      "t50 SCATTER is retained as a stochastic-nucleation signal."));
    const grid = el("div", { class: "cohort-kv" }, [
      kv("pooled t50", a.pooled_t50 != null ? fmt(a.pooled_t50) + " h" : "—"),
      kv("pooled SE (log₁₀)", fmt(a.pooled_se_log10)),
      kv("τ²_between", fmt(a.tau2_between)),
      kv("n usable replicates", a.n_usable_replicates != null ? String(a.n_usable_replicates) : "—"),
      kv("pooled SE < individual", a.pooled_se_lt_individual ? "yes ✓" : "no"),
    ]);
    box.appendChild(grid);
    const sn = a.stochastic_nucleation_signal || {};
    box.appendChild(el("div", { class: a.stochastic_nucleation_flag ? "cohort-signal fired" : "cohort-signal" }, [
      el("strong", {}, a.stochastic_nucleation_flag
        ? "STOCHASTIC-NUCLEATION SIGNAL FIRED" : "stochastic-nucleation scatter"),
      el("p", {}, "between-replicate t50 CV = " + fmt(sn.between_replicate_t50_cv) +
        " (threshold " + fmt(sn.threshold) + "). " + esc(sn.interpretation || "")),
    ]));
    return box;
  }

  if (mode === "concentration_series") {
    box.appendChild(el("h4", {}, "Dual-γ + shared-rate global ODE (mechanism constraint)"));
    box.appendChild(el("p", { class: "explainer" },
      "γ is the half-time scaling exponent (t50 ∝ [m]^−γ). The two γ estimators are reported " +
      "SEPARATELY and never merged; disagreement means the shape isn't concentration-invariant."));
    const dg = a.dual_gamma || {};
    // reuse the existing dual-γ rendering
    box.appendChild(renderGammaBlock(dg.gamma_regression || a.gamma_regression,
      dg.gamma_global || a.gamma_global, dg.disagreement || a.disagreement));
    const ode = a.global_ode_fit || {};
    if (ode.status === "ok") {
      box.appendChild(el("div", { class: "cohort-ode" }, [
        el("h4", {}, "Shared-rate Knowles/Cohen ODE global fit"),
        el("div", { class: "cohort-kv" }, [
          kv("best mechanism", ode.best_mechanism || "—"),
          kv("n_c", ode.reaction_orders ? fmt(ode.reaction_orders.n_c) : "—"),
          kv("n_2", (ode.reaction_orders && ode.reaction_orders.n_2 != null) ? fmt(ode.reaction_orders.n_2) : "fixed"),
          kv("γ_mechanistic", fmt(ode.gamma_mechanistic)),
          kv("sloppy", ode.sloppy ? "yes (only stiff combos identifiable)" : "conditioned"),
        ]),
        el("p", { class: "muted" }, esc(ode.honesty_note || "")),
      ]));
    } else if (ode.status === "skipped") {
      box.appendChild(el("p", { class: "muted" },
        "Shared-rate ODE fit skipped for latency (dual-γ above is the mechanism-constraint result)."));
    }
    box.appendChild(el("p", { class: "muted" }, esc(a.mechanism_note || "")));
    return box;
  }

  if (mode === "condition_series") {
    box.appendChild(el("h4", {}, "Feature-vs-condition surface (" + esc(a.surface_axis || "condition") + ")"));
    box.appendChild(el("p", { class: "explainer" },
      esc(a.surface_note || "feature values vs the varying condition over the MEASURED range; " +
        "no scaling exponent is licensed for a non-concentration axis.")));
    const surf = a.surface || [];
    const tbl = el("table", { class: "cohort-table" });
    tbl.appendChild(el("tr", {}, [esc(a.surface_axis || "condition"), "t50 (h)", "t50 status",
      "lag (h)", "max rate", "sharpness"].map((h) => el("th", {}, h))));
    surf.forEach((s) => {
      tbl.appendChild(el("tr", {}, [
        el("td", { class: "num" }, s.condition_value != null ? fmt(s.condition_value) : "—"),
        el("td", { class: "num" }, s.t50 != null ? fmt(s.t50) : "—"),
        el("td", {}, esc(s.t50_status || "—")),
        el("td", { class: "num" }, s.lag_time != null ? fmt(s.lag_time) : "—"),
        el("td", { class: "num" }, s.max_rate != null ? fmt(s.max_rate) : "—"),
        el("td", { class: "num" }, s.transition_sharpness != null ? fmt(s.transition_sharpness) : "—"),
      ]));
    });
    box.appendChild(tbl);
    return box;
  }

  // single_curve
  box.appendChild(el("h4", {}, "Single-curve descriptive result"));
  box.appendChild(el("div", { class: "cohort-kv" }, [
    kv("best model", a.best_model || "—"),
    kv("t50", a.features && a.features.t50 != null ? fmt(a.features.t50) + " h" : "—"),
    kv("t50 status", a.t50_status || "—"),
    kv("regime", a.descriptive_regime || "—"),
    kv("mechanism licensed", a.mechanistic_licensed ? "yes" : "no"),
  ]));
  if (a.information_note)
    box.appendChild(el("p", { class: "muted" }, esc(a.information_note)));
  return box;
}

function renderGammaBlock(reg, glob, dis) {
  reg = reg || {}; glob = glob || {}; dis = dis || {};
  const box = el("div", { class: "cohort-gamma" });
  const grid = el("div", { class: "cohort-kv" }, [
    kv("γ_regression", reg.gamma != null ? fmt(reg.gamma) : (reg.status || "—")),
    kv("γ_reg reliable", reg.gamma_reliable ? "yes" : "no"),
    kv("γ_reg physical", reg.gamma_physical === false ? "NO (γ<0 gated)" : "yes"),
    kv("γ_global (collapse)", glob.gamma != null ? fmt(glob.gamma) : (glob.status || "—")),
    kv("collapse R²", glob.collapse_r2 != null ? fmt(glob.collapse_r2) : "—"),
  ]);
  box.appendChild(grid);
  if (dis && (dis.disagree != null || dis.difference != null)) {
    box.appendChild(el("div", { class: dis.disagree ? "cohort-signal fired" : "cohort-signal" }, [
      el("strong", {}, dis.disagree ? "γ ESTIMATORS DISAGREE" : "γ estimators consistent"),
      el("p", {}, esc(dis.interpretation ||
        ("|Δγ| = " + fmt(dis.difference)))),
    ]));
  }
  if (reg.gamma_physical === false) {
    box.appendChild(el("div", { class: "warn-banner" },
      esc(reg.gamma_physical_reason || "γ<0 is non-physical (t50 rises with concentration) — gated")));
  }
  return box;
}

function renderWindowOfValidity(wov) {
  const box = el("div", { class: "cohort-wov" });
  box.appendChild(el("h4", {}, "Window of validity (inference not licensed beyond)"));
  const rng = (r, unit) => (r && r.min != null)
    ? fmt(r.min) + "–" + fmt(r.max) + " " + unit + " (" + (r.n_distinct || 0) + " distinct)" : "—";
  box.appendChild(el("div", { class: "cohort-kv" }, [
    kv("concentration", rng(wov.concentration_uM, "µM")),
    kv("pH", rng(wov.pH, "")),
    kv("temperature", rng(wov.temperature_C, "°C")),
    kv("assays", (wov.assays || []).join(", ") || "—"),
    kv("constructs", (wov.constructs || []).join(", ") || "—"),
  ]));
  box.appendChild(el("p", { class: "muted" }, esc(wov.note ||
    "inference is licensed only within these MEASURED ranges; PRISE does not extrapolate.")));
  return box;
}

function renderCohortMembers(members) {
  const box = el("div", { class: "cohort-members" });
  box.appendChild(el("h4", {}, "Members (" + members.length + ")"));
  const tbl = el("table", { class: "cohort-table" });
  tbl.appendChild(el("tr", {}, ["series", "conc (µM)", "pH", "temp", "assay", "construct", "censoring"].map((h) => el("th", {}, h))));
  members.forEach((m) => {
    const c = m.conditions || {};
    tbl.appendChild(el("tr", {}, [
      el("td", { class: "mono" }, esc(m.series_id)),
      el("td", { class: "num" }, c.concentration_uM != null ? fmt(c.concentration_uM) : "—"),
      el("td", { class: "num" }, c.pH != null ? fmt(c.pH) : "—"),
      el("td", { class: "num" }, c.temperature_C != null ? fmt(c.temperature_C) : "—"),
      el("td", {}, esc(c.assay_type || "—")),
      el("td", {}, esc(c.construct_id || "—")),
      el("td", {}, esc(m.censoring_class || "none")),
    ]));
  });
  box.appendChild(tbl);
  return box;
}

function kv(k, v) {
  return el("div", { class: "kv-item" }, [
    el("span", { class: "kv-k" }, k),
    el("span", { class: "kv-v" }, String(v)),
  ]);
}

function renderUnit(panel, unit, detail) {
  // condition + yield summary
  const cv = unit.condition_vector || {};
  const iy = unit.information_yield;

  // ----- IDENTITY HEADER: the chosen dataset's identity + conditions lead. -----
  const titleBits = [];
  if (cv.concentration && cv.concentration.value_uM != null)
    titleBits.push(fmt(cv.concentration.value_uM) + " µM");
  if (cv.pH != null) titleBits.push("pH " + fmt(cv.pH));
  if (cv.assay_type) titleBits.push(esc(cv.assay_type));
  titleBits.push(esc(unit.series_id));
  panel.appendChild(el("div", { class: "analysis-head" }, [
    el("h3", {}, "Analysis — " + esc(detail.protein_id) + " · " + titleBits.join(" · ")),
    el("p", { class: "section-note" },
      "PRISE's full analysis of this single CPAD 2.0 dataset. Pick another dataset above to switch."),
  ]));

  // ----- CURVE PLOT leads (fetch single curve) -----
  // a per-render plot context so the model-comparison card can overlay extra
  // fitted model lines on this same plot (multi-model overlay).
  const plotCtx = { plotDiv: null, series_id: unit.series_id, base: null, overlays: {} };
  const plotCard = el("div", { class: "card" }, [el("h3", {}, "Curve & fit")]);
  const plotDiv = el("div", { class: "plot-wrap" }, [el("div", { class: "loading" }, "loading curve…")]);
  plotCtx.plotDiv = plotDiv;
  plotCard.appendChild(plotDiv);
  panel.appendChild(plotCard);
  loadCurveInto(plotDiv, unit.series_id, plotCtx);

  // ----- MODEL COMPARISON / behaviour classification (fetch /api/models) -----
  const cmpCard = el("div", { class: "card" }, [
    el("h3", {}, "Model comparison — behaviour classification"),
    el("p", { class: "explainer" },
      "The full candidate bank fit explicitly to this curve and ranked by AICc — " +
      "descriptive sigmoids, the biphasic & autocatalytic forms, and the Tier-B " +
      "Knowles/Cohen nucleation models. This classifies the curve's BEHAVIOUR (shape); " +
      "it is not a mechanism call."),
    el("div", { class: "model-cmp-body" }, [el("div", { class: "loading" }, "fitting candidate bank…")]),
  ]);
  panel.appendChild(cmpCard);
  loadModelComparison($(".model-cmp-body", cmpCard), unit.series_id, plotCtx);

  // ----- condition + yield summary badges -----
  const summary = el("div", { class: "card" }, [
    el("div", { style: "display:flex; gap:.6rem; align-items:center; flex-wrap:wrap; margin-bottom:.6rem" }, [
      el("span", { class: "badge tier-" + (iy || "uninformative") }, "yield: " + esc(iy || "—")),
      unit.descriptive_regime && unit.descriptive_regime.regime
        ? el("span", { class: "badge tier-descriptive" }, esc(unit.descriptive_regime.regime)) : null,
      unit.censoring_class && unit.censoring_class !== "none"
        ? el("span", { class: "badge censor" }, "censoring: " + esc(unit.censoring_class)) : null,
      el("span", { class: "badge" }, esc(cv.assay_type || "?") +
        (cv.assay_reports_mass ? " (mass)" : " (non-mass)")),
      cv.concentration && cv.concentration.value_uM != null
        ? el("span", { class: "badge" }, fmt(cv.concentration.value_uM) + " µM") : null,
      cv.pH != null ? el("span", { class: "badge" }, "pH " + fmt(cv.pH)) : null,
      cv.temperature_C != null ? el("span", { class: "badge" }, fmt(cv.temperature_C) + " °C") : null,
      el("span", { class: "badge" }, "construct: " + esc(cv.construct_id || "?")),
    ]),
  ]);
  // shape confidence
  const sc = unit.shape_confidence || (unit.descriptive_regime || {}).confidence;
  if (sc) {
    summary.appendChild(el("p", { class: "muted", style: "font-size:.84rem" },
      "Shape confidence: " + esc(sc.level || "?") + " (R²=" + fmt(sc.r2) + ", n=" + (sc.n_points || "?") + "). " +
      "This is the M5 descriptive-shape confidence — NOT a mechanistic confidence."));
  }
  panel.appendChild(summary);

  // ----- METADATA QUALITY (M11) — completeness, uncertainty, mechanistic
  // completeness + blockers, ranked missing fields by impact, top recommendation.
  // Async slot; NEVER throws (a metadata hiccup can't break the analysis panel).
  const mqSlot = el("div", { class: "metadata-quality-slot" });
  panel.appendChild(mqSlot);
  loadMetadataQuality(mqSlot, unit.series_id);

  // ----- INFORMATION CONTENT (M12) — the headline: WHERE information concentrates
  // across the kinetic curve. Async slot; NEVER throws (an M12 hiccup can't break
  // the analysis panel). Needs the curve's raw (x_hours, y) which we fetch inside.
  const infoSlot = el("div", { class: "information-content-slot" });
  panel.appendChild(infoSlot);
  loadInformationContent(infoSlot, unit.series_id);

  // ----- NEXT EXPERIMENT — OPTIMAL DESIGN (M13 BOED) — the ranked next-experiment
  // recommendations, the (cost vs expected-information) Pareto frontier, and the
  // greedy multi-step plan. Async slot; NEVER throws (an M13 hiccup can't break the
  // analysis panel). Placed after the M12 information-content panel.
  const boedSlot = el("div", { class: "boed-slot" });
  panel.appendChild(boedSlot);
  loadBoed(boedSlot, unit.series_id);

  // FEATURES
  const cf = (unit.curve_features || {}).features || {};
  if (Object.keys(cf).length) {
    const featGrid = el("div", { class: "feature-grid" });
    const feats = [
      ["t50", cf.t50, (unit.curve_features || {}).t50_status],
      ["lag time", cf.lag_time, (unit.curve_features || {}).lag_status],
      ["lag/t50 ratio", cf.lag_to_t50_ratio, null],
      ["inflection time", cf.inflection_time, null],
      ["max rate", cf.max_rate, null],
      ["transition sharpness", cf.transition_sharpness, null],
      ["plateau", cf.plateau, null],
      ["dynamic range", cf.dynamic_range, null],
    ];
    feats.forEach(([label, val, tag]) => {
      featGrid.appendChild(el("div", { class: "feat" }, [
        el("div", { class: "fl" }, label),
        el("div", { class: "fv" }, fmt(val)),
        tag ? el("div", { class: "ft badge " + (tag === "point" ? "ok" : "censor") }, esc(tag)) : null,
      ]));
    });
    const gamma = [];
    if (unit.gamma_global) gamma.push("γ_global=" + fmt((unit.gamma_global || {}).value));
    if (unit.gamma_regression) gamma.push("γ_reg=" + fmt((unit.gamma_regression || {}).value));
    const featCard = el("div", { class: "card" }, [
      el("h3", {}, "Features (definition contract " + esc((unit.curve_features || {}).definition_contract || "") + ")"),
      featGrid,
      gamma.length ? el("p", { class: "section-note" }, gamma.join(" · ")) : null,
      cf.has_interior_inflection === false
        ? el("p", { class: "section-note" }, "Non-cooperative: no interior inflection → lag & inflection undefined (not faked).") : null,
    ]);
    // EMPIRICAL-BAYES PARTIAL POOLING — show the shrunk t50 + weight where the
    // pooling actually moved this curve's estimate toward its stratum mean (§4).
    const poolSlot = el("div", { class: "pooling-slot" });
    featCard.appendChild(poolSlot);
    loadPooling(poolSlot, detail.protein_id, unit.series_id);
    panel.appendChild(featCard);
  }

  // PROPENSITY
  const pr = unit.propensity;
  if (pr && pr.available) {
    const intr = pr.intrinsic || {};
    const coh = pr.cohort || {};
    const surf = pr.surface || {};
    const kv = el("div", { class: "kv" });
    const addKv = (k, v) => { kv.appendChild(el("div", { class: "k" }, k)); kv.appendChild(el("div", { class: "v" }, v)); };
    addKv("scored feature", esc(pr.scored_feature || "?") + " (" + esc(pr.feature_status || "?") + ")");
    addKv("condition match", esc(pr.condition_match || "?"));
    if (intr.anchor_kind) addKv("intrinsic anchor", esc(intr.anchor_kind) + " (eff n=" + (intr.anchor_effective_n ?? "?") + ")");
    if (intr.percentile_in_stratum != null) addKv("feature percentile", fmt(intr.percentile_in_stratum, 1));
    if (intr.propensity_percentile_in_stratum != null) addKv("propensity percentile", fmt(intr.propensity_percentile_in_stratum, 1) + " (" + esc(intr.direction || "") + ")");
    if (!coh.suppressed && coh.propensity_percentile != null) addKv("cohort propensity pct", fmt(coh.propensity_percentile, 1) + " (N=" + (coh.N ?? "?") + ")");
    if (surf.available && surf.gamma_global) addKv("surface γ_global", fmt((surf.gamma_global || {}).value));
    panel.appendChild(el("div", { class: "card" }, [el("h3", {}, "Propensity (three views, never collapsed)"), kv]));
  } else if (pr && pr.available === false) {
    panel.appendChild(el("div", { class: "card" }, [
      el("h3", {}, "Propensity"),
      el("p", { class: "muted" }, pr.construct_match === false
        ? "Refused: M6 has a record only for a different construct (no Wild-Type bleed onto this construct)."
        : "Not available for this unit (censored t50 is a bound, not rankable as a point)."),
    ]));
  }

  // SEQUENCE AXIS (from detail-level representative)
  const sa = detail.sequence_axis;
  if (sa && sa.available) {
    const tbl = el("table", { class: "subaxis-table" });
    tbl.appendChild(el("tr", {}, ["predictor", "value", "endpoint", "licensed vs assay"].map((h) => el("th", {}, h))));
    Object.entries(sa.sub_axes || {}).forEach(([name, ax]) => {
      tbl.appendChild(el("tr", {}, [
        el("td", {}, esc(name) + (ax.endpoint_class === "amyloid" ? "" : " (generic)")),
        el("td", { class: "num" }, ax.value != null ? fmt(ax.value) : (ax.n_waltz_amyloid_peptides != null ? ax.n_waltz_amyloid_peptides + " hexapep" : "—")),
        el("td", {}, esc(ax.endpoint_class || "?")),
        el("td", {}, ax.comparison_licensed
          ? el("span", { class: "badge ok" }, "licensed")
          : el("span", { class: "badge warn" }, "barred")),
      ]));
    });
    // The predictor SCORES are protein-level, but "licensed" / "barred" is a pure
    // function of the ASSAY. When this protein spans several assays the badges were
    // computed for one of them, and saying so is the difference between a
    // protein-level card and a mislabelled per-assay one.
    const saAssay = detail.sequence_axis_assay;
    const spanned = detail.assays_spanned || [];
    const repOnly = detail.sequence_axis_assay_is_representative_only;
    const heading = "Sequence axis (de-conflated · assay endpoint: "
      + esc(sa.assay_endpoint_class || "?")
      + (saAssay ? " · from " + esc(saAssay) : "") + ")";
    const cardKids = [
      el("h3", {}, heading),
      tbl,
      el("p", { class: "section-note" }, "A comparison is licensed only on endpoint match: amyloid-specific PASTA/Waltz vs an amyloid dye; generic TANGO/AGGRESCAN are barred."),
    ];
    if (repOnly) {
      cardKids.push(el("p", { class: "section-note warn" },
        "This protein is measured by " + spanned.length + " assays (" + esc(spanned.join(", "))
        + "). The predictor values above are protein-level, but the licensed/barred column was"
        + " computed for " + esc(saAssay || "the representative unit")
        + " only — it does not necessarily hold for the others."));
    }
    panel.appendChild(el("div", { class: "card" }, cardKids));
  }

  // M9 RECOMMENDATIONS
  const m9 = (detail.m9_recommendations || {})[unit.series_id];
  if (m9 && m9.recommendations) {
    const recList = el("div", { class: "rec-list" });
    (m9.recommendations || []).forEach((r, i) => {
      recList.appendChild(el("div", { class: "rec" + (i === 0 ? " top" : "") }, [
        el("div", { class: "rec-head" }, [
          el("span", { class: "rec-exp" }, "#" + (r.rank || i + 1) + " " + esc(r.experiment)),
          el("span", { class: "badge" }, "cost: " + esc(r.cost_tier || "?")),
          el("span", { class: "badge ok" }, "gain ×" + fmt(r.expected_degeneracy_reduction, 1)),
          r.necessary_but_not_sufficient ? el("span", { class: "badge warn" }, "necessary but not sufficient") : null,
        ]),
        el("div", { class: "rec-note" }, esc(r.expected_reduction_note || "")),
      ]));
    });
    (m9.refusals || []).forEach((rf) => {
      recList.appendChild(el("div", { class: "rec" }, [
        el("div", { class: "rec-head" }, [el("span", { class: "badge danger" }, "refused")]),
        el("div", { class: "rec-note" }, esc(typeof rf === "string" ? rf : JSON.stringify(rf))),
      ]));
    });
    panel.appendChild(el("div", { class: "card" }, [
      el("h3", {}, "Recommended next experiment (M9 · Service-C-quantified)"),
      recList,
    ]));
  }

  // MECHANISTIC GATES + CAVEATS (honesty-first, always last)
  const mech = unit.mechanistic || {};
  const honest = el("div", { class: "card" }, [el("h3", {}, "Honesty: why mechanism is not licensed + caveats")]);
  if (mech.mechanistic_inference_licensed === false || mech.gates_failed) {
    honest.appendChild(el("p", { class: "muted" },
      "Mechanistic inference licensed: " + (mech.mechanistic_inference_licensed ? "yes" : "NO") +
      ". Equivalence class: " + esc((mech.verdict_or_equivalence_class || mech.equivalence_class || []).join(", ") || "—") + "."));
    const gates = el("ul", { class: "caveat-list gate-list" });
    (mech.gates_failed || []).forEach((g) => gates.appendChild(el("li", {}, "gate failed: " + esc(g))));
    if ((mech.gates_failed || []).length) honest.appendChild(gates);
  }
  const caveats = el("ul", { class: "caveat-list" });
  (unit.caveats || []).forEach((c) => caveats.appendChild(el("li", {}, esc(c))));
  if ((unit.caveats || []).length) honest.appendChild(caveats);
  // compact (signal_only) units carry an m9_hook instead of full fields
  if (unit.m9_hook && unit.m9_hook.suggestion && !(m9 && m9.recommendations)) {
    honest.appendChild(el("p", { class: "section-note" }, "M9 hook: " + esc(unit.m9_hook.suggestion)));
  }
  panel.appendChild(honest);
}

// --------------------------------------------------------------------------- //
// EMPIRICAL-BAYES PARTIAL POOLING (per-curve shrunk t50). Renders a small
// section-note ONLY when the pooling actually moved this curve's estimate toward
// its stratum mean. MUST NEVER THROW — a pooling hiccup can never break the panel.
// --------------------------------------------------------------------------- //
async function loadPooling(container, proteinId, seriesId) {
  try {
    const d = await api("/api/pooling/" + encodeURIComponent(proteinId));
    if (!d || !d.available) return;
    const recs = d.curve_records || [];
    const rec = recs.find((r) => r && r.series_id === seriesId);
    if (!rec) return;
    const flags = rec.flags || [];
    const moved = rec.moved || 0;
    if (flags.indexOf("stratum_too_small") !== -1 || !(moved > 1e-6)) {
      // precise estimate or stratum too small — no meaningful shrinkage to show
      container.appendChild(el("p", { class: "section-note muted" },
        "not pooled — precise estimate or stratum too small"));
      return;
    }
    const stratum = rec.stratum || {};
    const stratumLabel = stratum.assay || rec.stratum_key || "matched";
    container.appendChild(el("p", { class: "section-note" },
      "Empirical-Bayes partial pooling: t50 shrunk " + fmt(rec.raw_value) + "→" + fmt(rec.pooled_value) +
      " h toward the " + esc(stratumLabel) + " mean " + fmt(rec.mu_stratum_value) + " h · weight w=" +
      fmt(rec.shrinkage_weight) + " (n_stratum=" + (rec.n_stratum ?? "?") + ")"));
    const explainer = [d.explainer, d.weight_formula].filter(Boolean).join(" · ");
    if (explainer) {
      container.appendChild(el("p", { class: "section-note muted", style: "font-size:.8rem" },
        esc(explainer)));
    }
  } catch (e) {
    // never let a pooling failure break the analysis panel — render nothing
  }
}

// --------------------------------------------------------------------------- //
// METADATA QUALITY (M11) — per-dataset panel. Renders the metadata_score (on a
// labelled 0–1 bar), the SEPARATE metadata_uncertainty, the mechanistic_completeness
// with its explicit blockers, the ranked missing fields by impact (each with its
// plain-English consequence_chain), and the top recommendation. MUST NEVER THROW.
// --------------------------------------------------------------------------- //
const _MQ_IMPACT_BADGE = { high: "danger", moderate: "warn", negligible: "" };

function scoreBar(label, value, opts) {
  // a labelled 0–1 bar. value in [0,1] or null. opts.danger flips the fill hue.
  const v = (typeof value === "number" && isFinite(value)) ? value : 0;
  const cls = "mq-bar-fill" + (opts && opts.danger ? " danger" : "");
  return el("div", { class: "mq-bar-row" }, [
    el("div", { class: "mq-bar-label" }, [
      el("span", {}, label),
      el("span", { class: "mq-bar-val" }, value == null ? "—" : fmt(value, 3) + " / 1"),
    ]),
    el("div", { class: "mq-bar-track" }, [
      el("div", { class: cls, style: "width:" + Math.max(0, Math.min(100, v * 100)) + "%" }),
    ]),
  ]);
}

function renderMetadataQuality(container, d) {
  container.innerHTML = "";
  const card = el("div", { class: "card metadata-quality" });
  card.appendChild(el("h3", {}, "Metadata quality & information completeness (M11)"));

  if (!d || d.available === false) {
    card.appendChild(el("p", { class: "muted" },
      esc((d && d.reason) || "metadata_quality.jsonl not built — run "
        + "`python engine/m11_metadata.py`.")));
    container.appendChild(card);
    return;
  }

  card.appendChild(el("p", { class: "explainer" },
    "How much of the metadata this dataset needs is actually PRESENT, and what its "
    + "absence blocks. Completeness ≠ correctness: this scores PRESENCE, it does NOT "
    + "verify the recorded values. Information-gain numbers are structural upper "
    + "bounds (except the Service-C-measured ones)."));

  // ----- the two scalars: completeness bar + SEPARATE uncertainty bar -----
  const bars = el("div", { class: "mq-bars" }, [
    scoreBar("metadata completeness (PRESENCE)", d.metadata_score, {}),
    scoreBar("metadata uncertainty (trust; 0 = trustworthy)", d.metadata_uncertainty, { danger: true }),
    scoreBar("mechanistic completeness (M5 prereqs met)", d.mechanistic_completeness, {}),
  ]);
  card.appendChild(bars);

  // ----- mechanistic completeness + explicit blockers -----
  const blockers = d.mechanistic_blockers || [];
  const blkWrap = el("div", { class: "mq-blockers" }, [
    el("span", { class: "mq-blockers-label" }, "mechanistic blockers:"),
  ]);
  if (blockers.length) {
    blockers.forEach((b) => blkWrap.appendChild(el("span", { class: "badge danger" }, esc(b))));
  } else {
    blkWrap.appendChild(el("span", { class: "badge ok" }, "none — mechanism prereqs met"));
  }
  card.appendChild(blkWrap);

  // ----- ranked missing fields by impact, each with its consequence chain -----
  const items = d.delta_inferential_power || [];
  if (items.length) {
    const list = el("div", { class: "mq-field-list" });
    items.forEach((it) => {
      const ig = it.information_gain || {};
      const basisBadge = it.gain_basis === "service_c_measured"
        ? el("span", { class: "badge ok", title: "Service-C MEASURED γ 1→4 split" }, "measured gain")
        : el("span", { class: "badge", title: "structural upper bound under the DAG (M8 ceiling)" }, "structural upper bound");
      list.appendChild(el("div", { class: "mq-field" }, [
        el("div", { class: "mq-field-head" }, [
          el("span", { class: "mq-field-name" }, esc(it.field)),
          el("span", { class: "badge " + (_MQ_IMPACT_BADGE[it.impact] || "") }, esc(it.impact || "?")),
          el("span", { class: "badge" }, "blocks: " + esc((it.blocks || []).join(", ") || "—")),
          basisBadge,
        ]),
        el("div", { class: "mq-field-chain" }, esc(it.consequence_chain || "")),
        ig.note ? el("div", { class: "faint", style: "font-size:.72rem" },
          "information gain: " + esc(ig.tier_lift || "") + " — " + esc(ig.note || "")) : null,
      ]));
    });
    card.appendChild(el("div", {}, [
      el("p", { class: "section-note" }, "Missing / uncertain fields, ranked by impact (high › moderate › negligible):"),
      list,
    ]));
  }

  // ----- top recommendation -----
  const rec = d.top_recommendation;
  if (rec) {
    card.appendChild(el("div", { class: "rec top", style: "margin-top:.7rem" }, [
      el("div", { class: "rec-head" }, [
        el("span", { class: "rec-exp" }, "Top fix: " + esc(rec.action || rec.field || "—")),
        el("span", { class: "badge ok" }, "impact: " + esc(rec.impact || "?")),
        el("span", { class: "badge" }, "effort: " + esc(rec.effort || "?")),
        el("span", { class: "badge" }, "gain/effort " + fmt(rec.gain_over_effort, 2)),
      ]),
      el("div", { class: "rec-note" }, esc(rec.rationale || "")),
    ]));
  }

  // ----- honesty footer -----
  const notCaptured = d.not_captured_by_source || [];
  card.appendChild(el("p", { class: "faint", style: "font-size:.74rem; margin-top:.6rem" },
    "Completeness ≠ correctness (a present value may still be wrong). Information-gain "
    + "numbers are STRUCTURAL upper bounds under the dependency DAG (inheriting M8's "
    + "validity ceiling), except the Service-C-measured γ 1→4 split."
    + (notCaptured.length
      ? " Not captured by the CPAD schema (excluded from the score, not the dataset's fault): "
        + esc(notCaptured.join(", ")) + "."
      : "")));

  container.appendChild(card);
}

async function loadMetadataQuality(container, seriesId) {
  try {
    const d = await api("/api/metadata/" + encodeURIComponent(seriesId));
    renderMetadataQuality(container, d);
  } catch (e) {
    // never let a metadata-quality failure break the analysis panel
    try {
      renderMetadataQuality(container, { available: false, reason: e.message });
    } catch (e2) { /* truly last-resort: render nothing */ }
  }
}

// --------------------------------------------------------------------------- //
// INFORMATION CONTENT (M12) — the headline deliverable: SHOW where information is
// concentrated across a kinetic curve. Renders:
//   (a) the HEADLINE viz — the curve with the information-density D(t) overlaid as a
//       filled area on a secondary y-axis + colour-graded markers, lag/growth/plateau
//       phase BANDS shaded behind, and a marker at the recommended next measurement;
//   (b) a phase-concentration bar (lag/growth/plateau info fractions);
//   (c) the scalars (effective rank HARD vs ENTROPY, condition number, richness, EIG);
//   (d) the per-parameter observability table + stiff/sloppy combinations;
//   (e) the recommended additional measurement;
//   (f) the honesty footer (FIM local / linearized / model-conditional / post-selection).
// Every function it calls is DEFINED here; every fetch is wrapped; MUST NEVER THROW.
// --------------------------------------------------------------------------- //

// phase -> accent colour (also used for the phase bands + the concentration bar)
const INFO_PHASE_COLORS = {
  lag: "#4ea1d3", growth: "#5ec6a8", plateau: "#e0a458", single_region: "#9fb0c0",
};
// a small blue->teal->amber ramp for the density heat mapping along the curve
function infoDensityColor(t) {
  // t in [0,1]; low = faint blue, mid = teal, high = warm amber (where info concentrates)
  const u = Math.max(0, Math.min(1, t));
  const stops = [
    [0.0, [78, 122, 160]],   // faint blue-grey (low leverage)
    [0.5, [94, 198, 168]],   // teal
    [1.0, [224, 164, 88]],   // amber (high leverage — information concentrates HERE)
  ];
  let a = stops[0], b = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (u >= stops[i][0] && u <= stops[i + 1][0]) { a = stops[i]; b = stops[i + 1]; break; }
  }
  const span = (b[0] - a[0]) || 1;
  const f = (u - a[0]) / span;
  const c = [0, 1, 2].map((k) => Math.round(a[1][k] + f * (b[1][k] - a[1][k])));
  return "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
}

function infoStat(label, value, sub, accent) {
  return el("div", { class: "info-stat" }, [
    el("div", { class: "info-stat-n", style: accent ? "color:" + accent : null },
      value == null ? "—" : String(value)),
    el("div", { class: "info-stat-l" }, label),
    sub ? el("div", { class: "info-stat-sub" }, sub) : null,
  ]);
}

// THE HEADLINE VIZ: curve + information-density overlay + phase bands + rec marker.
// Plotly path uses a filled area on a secondary y-axis (y2) for D(t), colour-graded
// markers along the curve, background shapes for the phase bands, and a vertical
// line + marker at the recommended next measurement. SVG fallback mirrors all of it.
function renderInfoDensityPlot(container, ctx) {
  container.innerHTML = "";
  // ctx: { x, y, density, phases:[{name,x0,x1,color}], recTime, tMax, ylabel }
  const { x, y, density, phases, recTime, tMax, ylabel } = ctx;
  const n = Math.min(x.length, y.length, density.length);
  if (n < 1) {
    container.appendChild(el("p", { class: "muted" }, "no curve/density points to plot"));
    return;
  }
  const dmax = Math.max(...density.slice(0, n), 1e-12);

  if (plotlyAvailable()) {
    try {
      const shapes = [];
      // phase BANDS (behind everything) — shade lag/growth/plateau
      (phases || []).forEach((ph) => {
        shapes.push({
          type: "rect", xref: "x", yref: "paper",
          x0: ph.x0, x1: ph.x1, y0: 0, y1: 1,
          fillcolor: ph.color, opacity: 0.12, line: { width: 0 }, layer: "below",
        });
      });
      // recommended-time vertical marker line
      if (recTime != null && isFinite(recTime)) {
        shapes.push({
          type: "line", xref: "x", yref: "paper",
          x0: recTime, x1: recTime, y0: 0, y1: 1,
          line: { color: "#e06c75", width: 2, dash: "dash" }, layer: "above",
        });
      }
      const traces = [];
      // D(t) filled area on the SECONDARY y-axis (the "where info is") band
      traces.push({
        x: x.slice(0, n), y: density.slice(0, n), yaxis: "y2",
        mode: "lines", type: "scatter", name: "information density D(t)",
        line: { color: "#5ec6a8", width: 1.5 }, fill: "tozeroy",
        fillcolor: "rgba(94,198,168,0.22)", hovertemplate: "t=%{x}<br>D=%{y:.3g}<extra></extra>",
      });
      // the curve itself with colour-graded markers (heat = local info leverage)
      traces.push({
        x: x.slice(0, n), y: y.slice(0, n), yaxis: "y1",
        mode: "lines+markers", type: "scatter", name: "curve (signal)",
        line: { color: "#6b7d8f", width: 1.5 },
        marker: {
          size: 9,
          color: density.slice(0, n),
          colorscale: [[0, infoDensityColor(0)], [0.5, infoDensityColor(0.5)], [1, infoDensityColor(1)]],
          cmin: 0, cmax: dmax,
          colorbar: { title: { text: "D(t)", side: "right" }, thickness: 10, len: 0.6, x: 1.02 },
          line: { color: "#11171f", width: 0.5 },
        },
        hovertemplate: "t=%{x}<br>signal=%{y:.3g}<extra></extra>",
      });
      // recommended-time marker point (on the curve baseline)
      if (recTime != null && isFinite(recTime)) {
        traces.push({
          x: [recTime], y: [Math.min(...y.slice(0, n))], yaxis: "y1",
          mode: "markers", type: "scatter", name: "recommended next measurement",
          marker: { size: 13, color: "#e06c75", symbol: "triangle-up",
            line: { color: "#11171f", width: 1 } },
          hovertemplate: "recommend measuring at t=%{x}<extra></extra>",
        });
      }
      const layout = {
        paper_bgcolor: "#1a212b", plot_bgcolor: "#11171f",
        font: { color: "#9fb0c0", size: 12 },
        margin: { l: 52, r: 60, t: 12, b: 42 },
        xaxis: { title: "time (hours)", gridcolor: "#2d3a48", zerolinecolor: "#2d3a48" },
        yaxis: { title: ylabel || "signal", gridcolor: "#2d3a48", zerolinecolor: "#2d3a48" },
        yaxis2: {
          title: { text: "information density D(t)", font: { color: "#5ec6a8" } },
          overlaying: "y", side: "right", showgrid: false,
          tickfont: { color: "#5ec6a8" }, rangemode: "tozero",
        },
        shapes,
        legend: { orientation: "h", y: -0.22, font: { size: 10 } },
        showlegend: true,
      };
      Plotly.newPlot(container, traces, layout, { displayModeBar: false, responsive: true });
      return;
    } catch (e) { /* fall through to SVG */ }
  }
  renderInfoDensityFallback(container, ctx, n, dmax);
}

// SVG fallback for the headline viz: phase bands + curve (heat markers) + D(t) area
// on a right axis + a dashed recommended-time line. No Plotly needed.
function renderInfoDensityFallback(container, ctx, n, dmax) {
  const { x, y, density, phases, recTime } = ctx;
  const W = 560, H = 300, padL = 46, padR = 52, padT = 14, padB = 46;
  const xs = x.slice(0, n), ys = y.slice(0, n), ds = density.slice(0, n);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const sx = (v) => padL + ((v - xmin) / ((xmax - xmin) || 1)) * (W - padL - padR);
  const sy = (v) => H - padB - ((v - ymin) / ((ymax - ymin) || 1)) * (H - padT - padB);
  const sd = (v) => H - padB - ((v / (dmax || 1)) * (H - padT - padB));

  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="information density">`;
  svg += `<rect x="0" y="0" width="${W}" height="${H}" fill="#11171f"/>`;
  // phase bands
  (phases || []).forEach((ph) => {
    const x0 = sx(ph.x0), x1 = sx(ph.x1);
    svg += `<rect x="${x0.toFixed(1)}" y="${padT}" width="${Math.max(0, x1 - x0).toFixed(1)}" height="${H - padT - padB}" fill="${ph.color}" opacity="0.13"/>`;
    svg += `<text x="${((x0 + x1) / 2).toFixed(1)}" y="${padT + 12}" fill="${ph.color}" font-size="9" text-anchor="middle">${esc(ph.name)}</text>`;
  });
  // axes
  svg += `<line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="#2d3a48"/>`;
  svg += `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" stroke="#2d3a48"/>`;
  // D(t) filled area on the right axis
  let area = `M${sx(xs[0]).toFixed(1)} ${(H - padB).toFixed(1)} `;
  for (let i = 0; i < n; i++) area += `L${sx(xs[i]).toFixed(1)} ${sd(ds[i]).toFixed(1)} `;
  area += `L${sx(xs[n - 1]).toFixed(1)} ${(H - padB).toFixed(1)} Z`;
  svg += `<path d="${area}" fill="rgba(94,198,168,0.22)" stroke="#5ec6a8" stroke-width="1.2"/>`;
  // curve line
  let cd = "";
  for (let i = 0; i < n; i++) cd += (i === 0 ? "M" : "L") + sx(xs[i]).toFixed(1) + " " + sy(ys[i]).toFixed(1) + " ";
  svg += `<path d="${cd}" fill="none" stroke="#6b7d8f" stroke-width="1.4"/>`;
  // recommended-time dashed line
  if (recTime != null && isFinite(recTime)) {
    const rx = sx(recTime);
    svg += `<line x1="${rx.toFixed(1)}" y1="${padT}" x2="${rx.toFixed(1)}" y2="${H - padB}" stroke="#e06c75" stroke-width="1.8" stroke-dasharray="5 4"/>`;
    svg += `<path d="M${(rx - 5).toFixed(1)} ${(H - padB).toFixed(1)} L${(rx + 5).toFixed(1)} ${(H - padB).toFixed(1)} L${rx.toFixed(1)} ${(H - padB - 9).toFixed(1)} Z" fill="#e06c75"/>`;
  }
  // colour-graded markers (heat = local info)
  for (let i = 0; i < n; i++) {
    svg += `<circle cx="${sx(xs[i]).toFixed(1)}" cy="${sy(ys[i]).toFixed(1)}" r="4.2" fill="${infoDensityColor(ds[i] / (dmax || 1))}" stroke="#11171f" stroke-width="0.6"/>`;
  }
  // labels
  svg += `<text x="${(padL + (W - padR)) / 2}" y="${H - 8}" fill="#6b7d8f" font-size="11" text-anchor="middle">time (hours)</text>`;
  svg += `<text x="12" y="${H / 2}" fill="#6b7d8f" font-size="11" text-anchor="middle" transform="rotate(-90 12 ${H / 2})">signal</text>`;
  svg += `<text x="${W - 10}" y="${H / 2}" fill="#5ec6a8" font-size="11" text-anchor="middle" transform="rotate(90 ${W - 10} ${H / 2})">information density D(t)</text>`;
  svg += `</svg>`;
  container.innerHTML = svg;
  container.appendChild(el("div", { class: "plot-note" },
    "Offline SVG render — warmer markers & the taller teal band mark where information concentrates."));
}

// PHASE CONCENTRATION BAR: lag / growth / plateau info fractions as a single stacked
// horizontal bar, with the dominant phase highlighted + a one-line read.
function renderInfoPhaseBar(fractions) {
  const wrap = el("div", { class: "info-phase-wrap" });
  const order = ["lag", "growth", "plateau"];
  const hasPhases = order.some((k) => fractions[k] != null);
  if (!hasPhases) {
    // non-cooperative -> single region (honest degradation, not faked phases)
    wrap.appendChild(el("p", { class: "muted", style: "font-size:.82rem" },
      "Non-cooperative curve (no interior inflection) — information is NOT split into "
      + "lag/growth/plateau phases; it is reported as a single region (not faked)."));
    return wrap;
  }
  const parts = order.map((k) => ({ name: k, frac: fractions[k] || 0 }));
  const dominant = parts.reduce((a, b) => (b.frac > a.frac ? b : a), parts[0]);
  const bar = el("div", { class: "info-phase-bar" });
  parts.forEach((p) => {
    const seg = el("div", {
      class: "info-phase-seg" + (p.name === dominant.name ? " dominant" : ""),
      style: "width:" + (p.frac * 100).toFixed(1) + "%; background:" + INFO_PHASE_COLORS[p.name],
      title: p.name + ": " + pct(p.frac),
    }, p.frac >= 0.12 ? [el("span", {}, pct(p.frac))] : []);
    bar.appendChild(seg);
  });
  wrap.appendChild(bar);
  // legend
  const leg = el("div", { class: "info-phase-legend" });
  parts.forEach((p) => {
    leg.appendChild(el("span", { class: "info-phase-key" }, [
      el("span", { class: "swatch", style: "background:" + INFO_PHASE_COLORS[p.name] }),
      el("span", {}, p.name + " " + pct(p.frac)),
    ]));
  });
  wrap.appendChild(leg);
  // one-line read
  wrap.appendChild(el("p", { class: "info-phase-read" }, [
    document.createTextNode("Information concentrates in the "),
    el("strong", { style: "color:" + INFO_PHASE_COLORS[dominant.name] }, dominant.name.toUpperCase()),
    document.createTextNode(" phase (" + pct(dominant.frac) + " of Σ D(t))."),
  ]));
  return wrap;
}

function renderInformationContent(container, d, curve) {
  container.innerHTML = "";
  const card = el("div", { class: "card information-content" });
  card.appendChild(el("h3", {}, "Information content (M12)"));

  if (!d || d.available === false) {
    card.appendChild(el("p", { class: "muted" },
      esc((d && d.reason) || "information_content.jsonl not built — run "
        + "`python engine/m12_information.py`.")));
    container.appendChild(card);
    return;
  }

  card.appendChild(el("p", { class: "explainer" },
    "WHERE information is concentrated across this kinetic curve. The Fisher-information "
    + "density D(t) marks which time-points constrain the fit; the phases (lag / growth / "
    + "plateau) show where it pools; the effective rank tells the sloppiness story; and the "
    + "recommended next measurement is the optimal-design argmax."));

  // ---- a flagged minimal (no_fit / degenerate) record: show what we have, honestly ----
  if (d.computable === false) {
    card.appendChild(el("p", { class: "muted" },
      "Information geometry not computable for this curve ("
      + esc(d.status || "flagged") + "): " + esc(d.reason || "") + "."));
    const mini = el("div", { class: "info-stat-row" }, [
      infoStat("effective rank (hard)", d.effective_rank_hard, "constrained directions"),
      infoStat("effective rank (entropy)", fmt(d.effective_rank_entropy, 2), "spectral erank"),
      infoStat("richness", fmt(d.information_richness_score, 3), "bounded 0–1"),
    ]);
    card.appendChild(mini);
    container.appendChild(card);
    return;
  }

  const dens = d.information_density || {};
  const density = dens.density || [];
  const bounds = dens.phase_boundaries || {};
  const fractions = dens.phase_fractions || {};

  // (a) THE HEADLINE VIZ — needs the curve's raw x/y (fetched by the loader).
  const cx = (curve && curve.x) || [];
  const cy = (curve && ((curve.y_processed && curve.y_processed.length) ? curve.y_processed
    : curve.y)) || [];
  const ylabel = (curve && curve.y_processed && curve.y_processed.length)
    ? "normalized signal" : "signal (a.u.)";
  // build phase bands from the boundaries (only when cooperative phases exist)
  const phases = [];
  if (dens.cooperative_phases && bounds.lag_end != null && bounds.growth_end != null) {
    phases.push({ name: "lag", x0: bounds.t0, x1: bounds.lag_end, color: INFO_PHASE_COLORS.lag });
    phases.push({ name: "growth", x0: bounds.lag_end, x1: bounds.growth_end, color: INFO_PHASE_COLORS.growth });
    phases.push({ name: "plateau", x0: bounds.growth_end, x1: bounds.t_end, color: INFO_PHASE_COLORS.plateau });
  }
  const recTime = (d.recommended_additional_measurements || {}).recommended_time;

  const vizHead = el("p", { class: "section-note" },
    "Curve (grey line) with the information density D(t) as the teal band on the right "
    + "axis; markers are colour-graded by local information (amber = high). "
    + (phases.length ? "Bands = lag/growth/plateau phases. " : "")
    + (recTime != null ? "Red dashed line = the recommended next measurement." : ""));
  card.appendChild(vizHead);
  const vizDiv = el("div", { class: "plot-wrap info-density-plot" });
  card.appendChild(vizDiv);
  if (cx.length && cy.length && density.length) {
    try {
      renderInfoDensityPlot(vizDiv, {
        x: cx, y: cy, density, phases, recTime,
        tMax: (d.expected_variance_reduction || {}).observed_t_max, ylabel,
      });
    } catch (e) {
      vizDiv.appendChild(el("p", { class: "muted" }, "density plot unavailable: " + esc(e.message)));
    }
  } else {
    vizDiv.appendChild(el("p", { class: "muted" },
      "curve points unavailable for the density overlay (density array has "
      + density.length + " points)."));
  }

  // (b) PHASE CONCENTRATION BAR
  card.appendChild(el("h4", { style: "margin-top:1rem" }, "Phase concentration"));
  card.appendChild(renderInfoPhaseBar(fractions));

  // (c) THE SCALARS — effective rank HARD vs ENTROPY (the sloppiness story) etc.
  const p = d.n_params;
  const erankH = d.effective_rank_hard;
  const erankE = d.effective_rank_entropy;
  const eig = d.expected_information_gain || {};
  const redun = d.redundancy || {};
  card.appendChild(el("h4", { style: "margin-top:1rem" }, "Information geometry"));
  const stats = el("div", { class: "info-stat-row" }, [
    infoStat("effective rank (hard)", erankH,
      p != null ? "of " + p + " params" : "constrained dirs", "var(--accent)"),
    infoStat("effective rank (entropy)", fmt(erankE, 2),
      "spectral erank", "var(--accent-2)"),
    infoStat("condition number", fmt(d.condition_number, 1),
      "raw FIM (scale-carrying)"),
    infoStat("richness", fmt(d.information_richness_score, 3),
      "bounded 0–1"),
    infoStat("redundancy", fmt(redun.redundancy, 3),
      "n_eff=" + fmt(redun.n_eff_participation_ratio, 1) + " / n=" + (redun.n_points ?? "?")),
    infoStat("EIG", fmt(eig.eig_bits, 2) + " bits",
      "prior-conditional (cv=" + fmt(eig.prior_cv, 1) + ")"),
  ]);
  card.appendChild(stats);
  // the sloppiness one-liner: "p parameters, but information in ~erank directions"
  if (p != null && erankE != null) {
    const sloppy = erankH != null && erankH < p;
    card.appendChild(el("p", { class: "info-sloppy-read" + (sloppy ? " sloppy" : "") }, [
      el("strong", {}, p + " parameter" + (p === 1 ? "" : "s")),
      document.createTextNode(", but information in ~" + fmt(erankE, 1)
        + " effective direction" + (erankE < 1.5 ? "" : "s") + " "),
      sloppy
        ? el("span", { class: "badge warn" }, "SLOPPY (erank < p)")
        : el("span", { class: "badge ok" }, "full rank"),
      document.createTextNode("."),
    ]));
  }

  // (d) PARAMETER OBSERVABILITY table + stiff / sloppy combinations
  const po = d.parameter_observability || {};
  const per = po.per_parameter || {};
  const paramNames = Object.keys(per);
  if (paramNames.length) {
    card.appendChild(el("h4", { style: "margin-top:1rem" }, "Parameter observability"));
    const tbl = el("table", { class: "subaxis-table info-obsv-table" });
    tbl.appendChild(el("tr", {}, ["parameter", "θ̂", "CRLB SE", "CV", "observable"].map((h) => el("th", {}, h))));
    paramNames.forEach((nm) => {
      const r = per[nm] || {};
      tbl.appendChild(el("tr", {}, [
        el("td", { class: "mono" }, esc(nm)),
        el("td", { class: "num" }, fmt(r.theta_hat)),
        el("td", { class: "num" }, fmt(r.crlb_se)),
        el("td", { class: "num" }, r.cv == null ? "—" : fmt(r.cv)),
        el("td", {}, r.observable
          ? el("span", { class: "badge ok" }, "observable")
          : el("span", { class: "badge warn" }, "unobservable")),
      ]));
    });
    card.appendChild(tbl);
    card.appendChild(el("p", { class: "section-note" },
      (po.n_observable ?? "?") + " / " + paramNames.length + " parameters observable "
      + "(CV < " + fmt(po.observable_cv_threshold, 1) + "; CV = SE/|θ̂|)."));

    // stiff vs sloppy identifiable combinations
    const stiff = po.stiff_combinations || [];
    const sloppyC = po.sloppy_combinations || [];
    const combos = el("div", { class: "info-combos" });
    const addCombos = (label, list, cls) => {
      if (!list.length) return;
      combos.appendChild(el("div", { class: "info-combo-group" }, [
        el("span", { class: "info-combo-label " + cls }, label),
        ...list.map((c) => el("span", { class: "info-combo mono" },
          esc(c.combination || c.dominant_parameter || "—")
          + " (λ=" + fmt(c.eigenvalue, 2) + ")")),
      ]));
    };
    addCombos("stiff (well-constrained):", stiff, "ok");
    addCombos("sloppy (poorly constrained):", sloppyC, "warn");
    if (stiff.length || sloppyC.length) {
      card.appendChild(el("p", { class: "section-note", style: "margin-top:.6rem" },
        "Identifiable combinations (FIM eigenvectors): STIFF directions are constrained; "
        + "SLOPPY directions are the unresolved parameter combinations."));
      card.appendChild(combos);
    }
  }

  // (e) RECOMMENDED ADDITIONAL MEASUREMENT
  const rec = d.recommended_additional_measurements || {};
  if (rec.experiment) {
    card.appendChild(el("h4", { style: "margin-top:1rem" }, "Recommended additional measurement"));
    const recHead = el("div", { class: "rec-head" }, [
      el("span", { class: "rec-exp" }, esc(rec.experiment)),
      rec.recommended_time != null
        ? el("span", { class: "badge ok" }, "at t = " + fmt(rec.recommended_time) + " h") : null,
      rec.recommended_time_phase
        ? el("span", { class: "badge" }, "phase: " + esc(rec.recommended_time_phase)) : null,
      rec.beyond_current_window
        ? el("span", { class: "badge warn" }, "beyond current window") : null,
      rec.tied_to_censoring
        ? el("span", { class: "badge censor" }, "tied to censoring") : null,
    ]);
    card.appendChild(el("div", { class: "rec top" }, [
      recHead,
      el("div", { class: "rec-note" }, esc(rec.rationale || "")),
      rec.action_language
        ? el("div", { class: "faint", style: "font-size:.72rem; margin-top:.3rem" },
          esc(rec.action_language)) : null,
    ]));
  }

  // (f) HONESTY FOOTER — FIM local / linearized / model-conditional / post-selection.
  const H = d.honesty || {};
  const honestyBits = [
    H.fim_is_local, H.fim_is_linearized, H.model_conditional, H.post_selection,
    H.eig_prior_conditional, H.richness_is_a_collapse, H.sloppy_is_reported,
  ].filter(Boolean);
  card.appendChild(el("div", { class: "info-honesty" }, [
    el("strong", {}, "Honesty: "),
    document.createTextNode(
      "the FIM here is LOCAL (curvature at θ̂), LINEARIZED (Laplace/Gaussian), "
      + "CONDITIONAL on the M2-selected model (" + esc(d.selected_model || "?")
      + ") + the σ²I noise model, and POST-SELECTION. "
      + "Richness is a bounded COLLAPSE of the spectrum (shown with it); EIG is "
      + "prior-conditional; a sloppy model (erank < p) is reported, not hidden."),
  ]));
  if (honestyBits.length) {
    const det = el("details", { style: "margin-top:.4rem" }, [
      el("summary", { class: "muted", style: "font-size:.76rem" }, "full honesty block"),
    ]);
    const ul = el("ul", { class: "caveat-list", style: "font-size:.74rem" });
    honestyBits.forEach((b) => ul.appendChild(el("li", {}, esc(b))));
    det.appendChild(ul);
    card.appendChild(det);
  }

  container.appendChild(card);
}

async function loadInformationContent(container, seriesId) {
  try {
    // fetch the M12 record and the raw curve (for the density overlay) in parallel
    let d = null, curve = null;
    try {
      d = await api("/api/information/" + encodeURIComponent(seriesId));
    } catch (e) {
      renderInformationContent(container, { available: false, reason: e.message });
      return;
    }
    // only fetch the curve when we have a computable record with a density array
    if (d && d.available && d.computable !== false
        && (d.information_density || {}).density && d.information_density.density.length) {
      try {
        curve = await api("/api/curve/" + encodeURIComponent(seriesId));
      } catch (e) {
        curve = null; // still render the rest of the panel without the overlay
      }
    }
    renderInformationContent(container, d, curve);
  } catch (e) {
    // never let an M12 failure break the analysis panel
    try {
      renderInformationContent(container, { available: false, reason: e.message });
    } catch (e2) { /* truly last-resort: render nothing */ }
  }
}

// --------------------------------------------------------------------------- //
// NEXT EXPERIMENT — OPTIMAL DESIGN (M13 BOED)
// The ranked next-experiment recommendations, the cost-vs-information Pareto
// frontier (Plotly scatter + SVG fallback), and the greedy multi-step plan with
// its diminishing-returns curve. Vanilla JS; reuses el/esc/fmt/pct/api + the
// Plotly-or-SVG fallback pattern. Every function defined here is called; every
// function called is defined (here or the shared helpers above).
// --------------------------------------------------------------------------- //

// friendly words for each candidate design (the candidate_id -> plain-English map)
const BOED_CANDIDATE_NAMES = {
  add_single_concentration: "Add one new protein concentration",
  add_concentration_series: "Add a ≥3-point concentration series",
  add_replicates: "Add replicate runs",
  increase_measurement_frequency: "Sample more frequently (denser Δt)",
  extend_sampling_duration: "Extend the observation window",
  vary_temperature: "Vary temperature",
  vary_pH: "Vary pH",
  record_agitation_and_seeding: "Record agitation & seeding (metadata)",
};
function boedFriendly(id) {
  return BOED_CANDIDATE_NAMES[id] || String(id || "—").replace(/_/g, " ");
}

// gain_basis -> badge class + label (FIM = a simulated Fisher-information direction;
// licensing = a categorical M5 license unblock; service_c_measured = mechanism gain).
const BOED_BASIS_BADGE = {
  fim: { cls: "ok", label: "FIM" },
  licensing: { cls: "warn", label: "licensing (gating)" },
  service_c_measured: { cls: "tier-scaling", label: "Service-C measured" },
  service_c_measured_structural_only: { cls: "tier-scaling", label: "Service-C (structural)" },
  none: { cls: "muted-badge", label: "none" },
};
function boedBasisBadge(basis) {
  const b = BOED_BASIS_BADGE[basis] || { cls: "", label: String(basis || "—") };
  return el("span", { class: "badge " + b.cls }, b.label);
}

// colour for a point by gain_basis (FIM designs vs the gating/licensing design)
function boedBasisColor(basis) {
  if (basis === "licensing") return "#e0a458";     // amber — the gating design
  if (basis === "fim") return "#4ea1d3";           // blue — a simulated FIM design
  return "#9fb0c0";
}

// map a runtime (hours) onto a marker size (bigger = longer to run)
function boedRuntimeSize(runtime, rmax) {
  const r = (runtime == null || !isFinite(runtime)) ? 0 : Math.max(0, runtime);
  const frac = rmax > 0 ? r / rmax : 0;
  return 9 + Math.round(frac * 16);                // 9..25 px
}

// Pull the per-candidate detail row we render in the table + plot from a full
// candidates[] entry. Reuses the ranked[] priority/EIG scalars keyed by candidate_id.
function boedCandidateRow(cand, rankInfo) {
  const eig = cand.expected_information_gain || {};
  const se = cand.expected_parameter_uncertainty_reduction || {};
  const mech = cand.expected_mechanistic_distinguishability_gain || {};
  const cost = cand.estimated_cost || {};
  const runtime = cand.estimated_runtime || {};
  // eig_bits_total exists on a series (parameters + gamma direction); else eig_bits
  const eigBits = (eig.eig_bits_total != null) ? eig.eig_bits_total : eig.eig_bits;
  return {
    candidate_id: cand.candidate_id,
    friendly: boedFriendly(cand.candidate_id),
    gain_basis: cand.gain_basis,
    target: cand.target,
    variable_type: cand.variable_type,
    eig_bits: eigBits,
    eig_gamma: eig.eig_bits_gamma_direction,
    se_max: se.max_se_reduction_pct,
    se_mean: se.mean_se_reduction_pct,
    mech_reduction: mech.class_size_reduction,
    mech_before: mech.equivalence_class_before,
    mech_after: mech.equivalence_class_after,
    mech_gain_x: mech.resolution_gain_x,
    mech_basis: mech.basis,
    cost_units: cost.cost_units,
    runtime_hours: runtime.runtime_hours,
    low_confidence: cand.low_confidence,
    low_confidence_reason: cand.low_confidence_reason,
    // ranking scalars (from ranked[]) — the priority = information/cost
    rank: rankInfo ? rankInfo.rank : null,
    priority: rankInfo ? rankInfo.priority_score_bits_per_cost : null,
    info_for_ranking: rankInfo ? rankInfo.information_gain_for_ranking_bits
      : (cand.information_gain_for_ranking_bits),
    description: cand.description,
  };
}

// THE PARETO FRONTIER: each candidate as a point at (x = estimated cost, y = expected
// information gain in bits). Pareto-optimal members are highlighted + connected as the
// frontier line; marker size encodes runtime; colour distinguishes FIM vs licensing
// designs. Plotly path + a full SVG fallback that mirrors it.
function renderParetoPlot(container, rows, paretoIds) {
  container.innerHTML = "";
  const pts = rows.filter((r) => r.cost_units != null && r.info_for_ranking != null);
  if (!pts.length) {
    container.appendChild(el("p", { class: "muted" }, "no scored candidates to plot"));
    return;
  }
  const rmax = Math.max(...pts.map((r) => (r.runtime_hours || 0)), 1e-9);
  const onFront = (id) => paretoIds.indexOf(id) >= 0;
  // frontier members sorted by cost so the connecting line is monotone
  const front = pts.filter((r) => onFront(r.candidate_id))
    .slice().sort((a, b) => (a.cost_units - b.cost_units));

  if (plotlyAvailable()) {
    try {
      const traces = [];
      // 1) the frontier line (connect the non-dominated members)
      if (front.length >= 2) {
        traces.push({
          x: front.map((r) => r.cost_units),
          y: front.map((r) => r.info_for_ranking),
          mode: "lines", type: "scatter", name: "Pareto frontier",
          line: { color: "#6cc070", width: 2, dash: "solid" },
          hoverinfo: "skip",
        });
      }
      // 2) all candidate points, coloured by gain_basis, sized by runtime, frontier
      //    members ringed. We split gating vs FIM into separate traces for the legend.
      const groups = [
        { key: "fim", name: "FIM design (simulated)", color: boedBasisColor("fim") },
        { key: "licensing", name: "licensing / gating", color: boedBasisColor("licensing") },
      ];
      groups.forEach((g) => {
        const gp = pts.filter((r) => (g.key === "fim" ? r.gain_basis !== "licensing"
          : r.gain_basis === "licensing"));
        if (!gp.length) return;
        traces.push({
          x: gp.map((r) => r.cost_units),
          y: gp.map((r) => r.info_for_ranking),
          text: gp.map((r) => r.friendly + (onFront(r.candidate_id) ? " ★" : "")),
          customdata: gp.map((r) => [r.candidate_id, r.runtime_hours, r.eig_bits]),
          mode: "markers", type: "scatter", name: g.name,
          marker: {
            size: gp.map((r) => boedRuntimeSize(r.runtime_hours, rmax)),
            color: g.color,
            symbol: gp.map((r) => (r.gain_basis === "licensing" ? "diamond" : "circle")),
            line: {
              color: gp.map((r) => (onFront(r.candidate_id) ? "#6cc070" : "#11171f")),
              width: gp.map((r) => (onFront(r.candidate_id) ? 2.5 : 0.6)),
            },
          },
          hovertemplate: "%{text}<br>cost=%{x} units<br>info=%{y:.2f} bits"
            + "<br>runtime=%{customdata[1]:.0f} h<extra></extra>",
        });
      });
      const layout = {
        paper_bgcolor: "#1a212b", plot_bgcolor: "#11171f",
        font: { color: "#9fb0c0", size: 12 },
        margin: { l: 56, r: 16, t: 12, b: 44 },
        xaxis: { title: "estimated cost (units)", gridcolor: "#2d3a48", zerolinecolor: "#2d3a48" },
        yaxis: { title: "expected information gain (bits)", gridcolor: "#2d3a48", zerolinecolor: "#2d3a48" },
        legend: { orientation: "h", y: -0.22, font: { size: 10 } },
        showlegend: true,
      };
      Plotly.newPlot(container, traces, layout, { displayModeBar: false, responsive: true });
      return;
    } catch (e) { /* fall through to SVG */ }
  }
  renderParetoFallback(container, pts, front, rmax, onFront);
}

// SVG fallback for the Pareto frontier: axes + connecting frontier line + all
// candidate points (colour by basis, size by runtime, frontier members ringed green).
function renderParetoFallback(container, pts, front, rmax, onFront) {
  const W = 560, H = 320, padL = 56, padR = 18, padT = 16, padB = 46;
  const costs = pts.map((r) => r.cost_units);
  const infos = pts.map((r) => r.info_for_ranking);
  const xmin = Math.min(...costs, 0), xmax = Math.max(...costs);
  const ymin = Math.min(...infos, 0), ymax = Math.max(...infos);
  const sx = (v) => padL + ((v - xmin) / ((xmax - xmin) || 1)) * (W - padL - padR);
  const sy = (v) => H - padB - ((v - ymin) / ((ymax - ymin) || 1)) * (H - padT - padB);

  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Pareto frontier">`;
  svg += `<rect x="0" y="0" width="${W}" height="${H}" fill="#11171f"/>`;
  svg += `<line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="#2d3a48"/>`;
  svg += `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" stroke="#2d3a48"/>`;
  // frontier connecting line
  if (front.length >= 2) {
    let d = "";
    for (let i = 0; i < front.length; i++) {
      d += (i === 0 ? "M" : "L") + sx(front[i].cost_units).toFixed(1) + " "
        + sy(front[i].info_for_ranking).toFixed(1) + " ";
    }
    svg += `<path d="${d}" fill="none" stroke="#6cc070" stroke-width="2"/>`;
  }
  // candidate points
  pts.forEach((r) => {
    const cx = sx(r.cost_units), cy = sy(r.info_for_ranking);
    const rad = boedRuntimeSize(r.runtime_hours, rmax) / 2.4;
    const col = boedBasisColor(r.gain_basis);
    const ring = onFront(r.candidate_id);
    if (r.gain_basis === "licensing") {
      // diamond for the gating design
      const s = rad;
      svg += `<path d="M${cx.toFixed(1)} ${(cy - s).toFixed(1)} L${(cx + s).toFixed(1)} ${cy.toFixed(1)} `
        + `L${cx.toFixed(1)} ${(cy + s).toFixed(1)} L${(cx - s).toFixed(1)} ${cy.toFixed(1)} Z" `
        + `fill="${col}" stroke="${ring ? "#6cc070" : "#11171f"}" stroke-width="${ring ? 2.5 : 0.8}"/>`;
    } else {
      svg += `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${rad.toFixed(1)}" `
        + `fill="${col}" stroke="${ring ? "#6cc070" : "#11171f"}" stroke-width="${ring ? 2.5 : 0.8}"/>`;
    }
  });
  svg += `<text x="${(padL + (W - padR)) / 2}" y="${H - 8}" fill="#6b7d8f" font-size="11" text-anchor="middle">estimated cost (units)</text>`;
  svg += `<text x="14" y="${H / 2}" fill="#6b7d8f" font-size="11" text-anchor="middle" transform="rotate(-90 14 ${H / 2})">expected information gain (bits)</text>`;
  svg += `</svg>`;
  container.innerHTML = svg;
  container.appendChild(el("div", { class: "plot-note" },
    "Offline SVG render — the green line connects the Pareto-optimal (non-dominated) "
    + "designs; larger markers = longer runtime; blue circles = FIM designs, "
    + "amber diamonds = the licensing/gating design."));
}

// THE DIMINISHING-RETURNS CURVE: the per-step marginal bits of the greedy plan as a
// small vertical bar chart (each step's marginal information gain). SVG only (small).
function renderDiminishingBars(container, curveBits, steps) {
  container.innerHTML = "";
  const vals = (curveBits || []).map((v) => (v == null || !isFinite(v)) ? 0 : v);
  if (!vals.length) {
    container.appendChild(el("p", { class: "muted" }, "no multi-step plan to chart"));
    return;
  }
  const W = 360, H = 150, padL = 40, padR = 12, padT = 14, padB = 30;
  const vmax = Math.max(...vals, 1e-9);
  const n = vals.length;
  const bw = (W - padL - padR) / n;
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="diminishing returns">`;
  svg += `<rect x="0" y="0" width="${W}" height="${H}" fill="#11171f"/>`;
  svg += `<line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="#2d3a48"/>`;
  svg += `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" stroke="#2d3a48"/>`;
  // trend line across bar tops (to show the fall-off)
  let trend = "";
  for (let i = 0; i < n; i++) {
    const h = (vals[i] / vmax) * (H - padT - padB);
    const x = padL + i * bw + bw * 0.5;
    const yTop = (H - padB) - h;
    // bar
    svg += `<rect x="${(padL + i * bw + bw * 0.18).toFixed(1)}" y="${yTop.toFixed(1)}" `
      + `width="${(bw * 0.64).toFixed(1)}" height="${h.toFixed(1)}" fill="#5ec6a8" opacity="0.75"/>`;
    // value label
    svg += `<text x="${x.toFixed(1)}" y="${(yTop - 3).toFixed(1)}" fill="#9fb0c0" font-size="9" text-anchor="middle">${esc(fmt(vals[i], 2))}</text>`;
    // step label
    const stepId = (steps && steps[i]) ? steps[i].candidate_id : ("step " + (i + 1));
    svg += `<text x="${x.toFixed(1)}" y="${(H - padB + 12).toFixed(1)}" fill="#6b7d8f" font-size="9" text-anchor="middle">${esc(i + 1)}</text>`;
    trend += (i === 0 ? "M" : "L") + x.toFixed(1) + " " + yTop.toFixed(1) + " ";
  }
  svg += `<path d="${trend}" fill="none" stroke="#e0a458" stroke-width="1.6" stroke-dasharray="4 3"/>`;
  svg += `<text x="14" y="${H / 2}" fill="#6b7d8f" font-size="10" text-anchor="middle" transform="rotate(-90 14 ${H / 2})">marginal bits</text>`;
  svg += `<text x="${(padL + (W - padR)) / 2}" y="${H - 4}" fill="#6b7d8f" font-size="10" text-anchor="middle">plan step</text>`;
  svg += `</svg>`;
  container.innerHTML = svg;
}

// the ranked recommendation table (rank, friendly experiment, EIG bits, SE reduction,
// mechanistic-distinguishability gain, cost, runtime, gain_basis badge, low_confidence).
function renderBoedTable(rows) {
  const tbl = el("table", { class: "subaxis-table boed-table" });
  tbl.appendChild(el("tr", {}, [
    "#", "experiment", "EIG (bits)", "param SE ↓", "mech. gain", "cost", "runtime (h)", "basis", "flags",
  ].map((h) => el("th", {}, h))));
  rows.forEach((r) => {
    const mechCell = (r.mech_reduction && r.mech_reduction > 0)
      ? (r.mech_before + "→" + r.mech_after + " classes (" + fmt(r.mech_gain_x, 1) + "×)")
      : "—";
    tbl.appendChild(el("tr", { class: r.rank === 1 ? "boed-top-row" : null }, [
      el("td", { class: "num" }, r.rank == null ? "—" : String(r.rank)),
      el("td", {}, [
        el("span", { class: "boed-exp" }, esc(r.friendly)),
        r.target ? el("span", { class: "faint", style: "font-size:.7rem; display:block" },
          "target: " + esc(r.target)) : null,
      ]),
      el("td", { class: "num" }, r.eig_bits == null ? "—" : fmt(r.eig_bits, 2)
        + (r.eig_gamma != null ? " (+" + fmt(r.eig_gamma, 2) + " γ)" : "")),
      el("td", { class: "num" }, r.se_max == null ? "—" : fmt(r.se_max, 1) + "%"),
      el("td", {}, mechCell),
      el("td", { class: "num" }, fmt(r.cost_units, 2)),
      el("td", { class: "num" }, fmt(r.runtime_hours, 1)),
      el("td", {}, boedBasisBadge(r.gain_basis)),
      el("td", {}, r.low_confidence
        ? el("span", { class: "badge warn", title: esc(r.low_confidence_reason || "") }, "low confidence")
        : el("span", { class: "faint", style: "font-size:.72rem" }, "—")),
    ]));
  });
  return tbl;
}

// the multi-step plan: step 1 → 2 → 3 (chosen experiment + marginal bits), cumulative
// gain vs cost, the diminishing-returns bar chart, and the greedy_note.
function renderBoedPlan(card, plan) {
  const steps = plan.steps || [];
  card.appendChild(el("h4", { style: "margin-top:1rem" }, "Multi-step plan (greedy, K = "
    + (plan.k_steps != null ? plan.k_steps : steps.length) + ")"));
  if (!steps.length) {
    card.appendChild(el("p", { class: "muted" },
      esc((plan.note || "no multi-step plan available."))));
    return;
  }
  // step chain: chosen experiment + marginal bits, arrow-connected
  const chain = el("div", { class: "boed-step-chain" });
  steps.forEach((s, i) => {
    if (i > 0) chain.appendChild(el("span", { class: "boed-arrow" }, "→"));
    chain.appendChild(el("div", { class: "boed-step" + (i === 0 ? " first" : "") }, [
      el("div", { class: "boed-step-n" }, "step " + (s.step != null ? s.step : (i + 1))),
      el("div", { class: "boed-step-exp" }, esc(boedFriendly(s.candidate_id))),
      el("div", { class: "boed-step-bits" }, "+" + fmt(s.marginal_information_gain_bits, 2) + " bits"),
      boedBasisBadge(s.gain_basis),
    ]));
  });
  card.appendChild(chain);

  // diminishing-returns bar chart + a one-line read on monotonicity
  card.appendChild(el("p", { class: "section-note", style: "margin-top:.7rem" },
    "Diminishing-returns curve — each step's MARGINAL information gain. Log-det EIG is "
    + "submodular, so the FIM marginals should fall off; the amber dashed line traces the drop."));
  const barDiv = el("div", { class: "boed-dim-bars" });
  card.appendChild(barDiv);
  renderDiminishingBars(barDiv, plan.diminishing_returns_curve_bits, steps);
  card.appendChild(el("p", { class: "section-note" }, [
    plan.fim_marginals_non_increasing
      ? el("span", { class: "badge ok" }, "FIM marginals non-increasing ✓")
      : el("span", { class: "badge warn" }, "FIM marginals not strictly non-increasing"),
    document.createTextNode("  cumulative: " + fmt(plan.cumulative_gain_bits, 2)
      + " bits for " + fmt(plan.cumulative_cost_units, 2) + " cost units."),
  ]));

  // cumulative gain vs cost table (per step)
  const cumTbl = el("table", { class: "subaxis-table boed-cum-table" });
  cumTbl.appendChild(el("tr", {}, ["step", "experiment", "marg. bits", "Σ bits", "Σ cost"]
    .map((h) => el("th", {}, h))));
  steps.forEach((s, i) => {
    cumTbl.appendChild(el("tr", {}, [
      el("td", { class: "num" }, String(s.step != null ? s.step : (i + 1))),
      el("td", {}, esc(boedFriendly(s.candidate_id))),
      el("td", { class: "num" }, fmt(s.marginal_information_gain_bits, 2)),
      el("td", { class: "num" }, fmt(s.cumulative_gain_bits, 2)),
      el("td", { class: "num" }, fmt(s.cumulative_cost_units, 2)),
    ]));
  });
  card.appendChild(cumTbl);

  if (plan.greedy_note) {
    card.appendChild(el("p", { class: "section-note", style: "margin-top:.5rem" },
      esc(plan.greedy_note)));
  }
}

// current_state (what's already known) — the context row above the recommendations.
function renderBoedCurrentState(card, cs) {
  const badges = el("div", { class: "boed-state-badges" });
  const add = (label, val, cls) => {
    if (val == null || val === "") return;
    badges.appendChild(el("span", { class: "badge " + (cls || "") }, label + ": " + val));
  };
  add("concentration", cs.protein_concentration_uM != null
    ? fmt(cs.protein_concentration_uM) + " µM" : null);
  add("γ used", cs.gamma_used != null ? fmt(cs.gamma_used, 3)
    + (cs.gamma_reliable ? "" : " (fallback)") : null, cs.gamma_reliable ? "ok" : "warn");
  badges.appendChild(el("span", { class: "badge " + (cs.has_concentration_series_now ? "ok" : "muted-badge") },
    "concentration series: " + (cs.has_concentration_series_now ? "present" : "none")));
  badges.appendChild(el("span", { class: "badge " + (cs.agitation_seeding_known ? "ok" : "warn") },
    "agitation/seeding: " + (cs.agitation_seeding_known ? "recorded" : "unknown")));
  if (cs.temperature_measured_range)
    add("T range", cs.temperature_measured_range.map((v) => fmt(v)).join("–") + " °C");
  if (cs.pH_measured_range)
    add("pH range", cs.pH_measured_range.map((v) => fmt(v)).join("–"));
  card.appendChild(badges);
}

function renderBoed(container, d) {
  container.innerHTML = "";
  const card = el("div", { class: "card boed" });
  card.appendChild(el("h3", {}, "Next experiment — optimal design (M13)"));

  if (!d || d.available === false) {
    card.appendChild(el("p", { class: "muted" },
      esc((d && d.reason) || "boed_recommendations.jsonl not built — run "
        + "`python engine/m13_boed.py`.")));
    container.appendChild(card);
    return;
  }

  card.appendChild(el("p", { class: "explainer" },
    "Bayesian Optimal Experimental Design: which next experiment adds the most "
    + "EXPECTED scientific information per unit cost. Candidate designs are scored by "
    + "expected information gain (EIG, bits) under the current fitted forward model, "
    + "ranked by EIG ÷ cost, laid out on the cost-vs-information Pareto frontier, and "
    + "sequenced into a greedy multi-step plan with diminishing returns."));

  // ---- INSUFFICIENT (no fitted forward model): show the M9 rule-table fallback ----
  if (d.status && d.status !== "ok") {
    const fb = d.m9_fallback || {};
    card.appendChild(el("p", { class: "muted" },
      "No converged forward model for this curve (" + esc(d.status) + "): "
      + esc(d.reason || "") + ". Falling back to the M9 heuristic rule table."));
    const sug = fb.suggested_experiments || [];
    if (sug.length) {
      const chain = el("div", { class: "boed-step-chain" });
      sug.forEach((id, i) => {
        if (i > 0) chain.appendChild(el("span", { class: "boed-arrow" }, "·"));
        chain.appendChild(el("div", { class: "boed-step" }, [
          el("div", { class: "boed-step-exp" }, esc(boedFriendly(id))),
        ]));
      });
      card.appendChild(chain);
    }
    if (fb.note) card.appendChild(el("p", { class: "section-note" }, esc(fb.note)));
    container.appendChild(card);
    return;
  }

  const cands = d.candidates || [];
  const ranked = d.ranked || [];
  const rankById = {};
  ranked.forEach((r) => { rankById[r.candidate_id] = r; });
  // build the detail rows (join candidates[] with ranked[]) sorted by rank
  const rows = cands.map((c) => boedCandidateRow(c, rankById[c.candidate_id]));
  rows.sort((a, b) => {
    if (a.rank == null && b.rank == null) return 0;
    if (a.rank == null) return 1;
    if (b.rank == null) return -1;
    return a.rank - b.rank;
  });

  // ---- (d) TOP RECOMMENDATION callout ----
  const topId = d.top_recommendation;
  const topRow = rows.find((r) => r.candidate_id === topId) || rows[0];
  if (topRow) {
    card.appendChild(el("div", { class: "boed-top-callout" }, [
      el("div", { class: "boed-top-label" }, "Top recommendation"),
      el("div", { class: "boed-top-exp" }, [
        el("span", {}, esc(topRow.friendly)),
        boedBasisBadge(topRow.gain_basis),
        topRow.low_confidence ? el("span", { class: "badge warn" }, "low confidence") : null,
      ]),
      el("div", { class: "boed-top-metrics" },
        "EIG " + fmt(topRow.eig_bits, 2) + " bits · "
        + fmt(topRow.priority, 2) + " bits/cost · cost " + fmt(topRow.cost_units, 2)
        + " · " + fmt(topRow.runtime_hours, 1) + " h"),
      topRow.description
        ? el("div", { class: "boed-top-desc" }, esc(topRow.description)) : null,
    ]));
  }

  // ---- (d) CURRENT STATE (what's already known) ----
  if (d.current_state) {
    card.appendChild(el("h4", { style: "margin-top:1rem" }, "Current state (already known)"));
    renderBoedCurrentState(card, d.current_state);
  }

  // ---- (a) THE PARETO FRONTIER plot ----
  card.appendChild(el("h4", { style: "margin-top:1rem" }, "Cost vs information — Pareto frontier"));
  card.appendChild(el("p", { class: "section-note" },
    "Each design is a point at (estimated cost, expected information gain). The green "
    + "line connects the Pareto-optimal (non-dominated) designs — no other design is "
    + "cheaper AND more informative AND faster. Marker size = runtime; blue circles = "
    + "FIM (simulated) designs, amber diamonds = the licensing/gating design."));
  const paretoIds = (d.pareto_optimal_sets || []).map((p) => p.candidate_id);
  const plotDiv = el("div", { class: "plot-wrap boed-pareto-plot" });
  card.appendChild(plotDiv);
  try {
    renderParetoPlot(plotDiv, rows, paretoIds);
  } catch (e) {
    plotDiv.appendChild(el("p", { class: "muted" }, "Pareto plot unavailable: " + esc(e.message)));
  }
  if (paretoIds.length) {
    card.appendChild(el("p", { class: "section-note" },
      paretoIds.length + " Pareto-optimal design" + (paretoIds.length === 1 ? "" : "s") + ": "
      + paretoIds.map((id) => boedFriendly(id)).join(", ") + "."));
  }

  // ---- (b) RANKED RECOMMENDATION TABLE ----
  card.appendChild(el("h4", { style: "margin-top:1rem" }, "Ranked recommendations (by EIG ÷ cost)"));
  card.appendChild(renderBoedTable(rows));

  // ---- (c) THE MULTI-STEP PLAN ----
  if (d.multi_step_plan) {
    renderBoedPlan(card, d.multi_step_plan);
  }

  // ---- (e) HONESTY FOOTER ----
  const H = d.honesty || {};
  const honestyBits = [
    H.eig_is_expected_laplace, H.mechanistic_gain_bounded_by_service_c,
    H.cost_is_an_estimate, H.gating_is_licensing_not_fim, H.greedy_not_global,
    H.tpH_extrapolation_flagged, H.self_similar_shape_approx,
  ].filter(Boolean);
  card.appendChild(el("div", { class: "info-honesty" }, [
    el("strong", {}, "Honesty: "),
    document.createTextNode(
      "EIG is EXPECTED under the fitted forward model (" + esc(d.selected_model || "?")
      + ") + a weak Gaussian prior + the assumed σ²I noise — Laplace-LINEARIZED, "
      + "inheriting M8's validity ceiling (not absolute information). Mechanistic gains "
      + "are BOUNDED by Service-C γ-separability (an UPPER bound). Cost & runtime are a "
      + "VERSIONED ESTIMATE table for RANKING, NOT real lab economics. GATING designs "
      + "(record agitation/seeding) give a LICENSING unblock, NOT a smooth FIM gain. "
      + "The greedy plan is NOT the global optimum (submodularity gives a (1−1/e) "
      + "guarantee); T/pH extrapolation beyond the measured window is flagged."),
  ]));
  if (honestyBits.length) {
    const det = el("details", { style: "margin-top:.4rem" }, [
      el("summary", { class: "muted", style: "font-size:.76rem" }, "full honesty block"),
    ]);
    const ul = el("ul", { class: "caveat-list", style: "font-size:.74rem" });
    honestyBits.forEach((b) => ul.appendChild(el("li", {}, esc(b))));
    det.appendChild(ul);
    card.appendChild(det);
  }

  container.appendChild(card);
}

async function loadBoed(container, seriesId) {
  try {
    let d = null;
    try {
      d = await api("/api/boed/" + encodeURIComponent(seriesId));
    } catch (e) {
      renderBoed(container, { available: false, reason: e.message });
      return;
    }
    renderBoed(container, d);
  } catch (e) {
    // never let an M13 failure break the analysis panel
    try {
      renderBoed(container, { available: false, reason: e.message });
    } catch (e2) { /* truly last-resort: render nothing */ }
  }
}

async function loadCurveInto(container, seriesId, plotCtx) {
  try {
    const c = await api("/api/curve/" + encodeURIComponent(seriesId));
    const x = c.x || [];
    const y = (c.y_processed && c.y_processed.length) ? c.y_processed : (c.y || []);
    const tri = c.triage || {};
    if (plotCtx) {
      // store base curve so the model-comparison card can re-render with overlays
      plotCtx.base = {
        x, y, fitted: c.fitted,
        censoring: tri.censoring_class,
        ylabel: c.y_processed ? "normalized signal" : "signal (a.u.)",
      };
      plotCtx.plotDiv = container;
      redrawPlotWithOverlays(plotCtx);
    } else {
      renderCurvePlot(container, {
        x, y, fitted: c.fitted, title: null,
        censoring: tri.censoring_class,
        ylabel: c.y_processed ? "normalized signal" : "signal (a.u.)",
      });
    }
    container.appendChild(el("p", { class: "plot-note" },
      "fittability: " + esc(tri.fittability_class || "?") +
      " · censoring: " + esc(tri.censoring_class || "none") +
      " · normalization: " + esc(tri.normalization_mode || "?") +
      (c.fitted ? " · fit: " + esc(c.fitted.model) + " (R²=" + fmt(c.fitted.r2) + ")" : " · no fit overlay") +
      (c.t50_status ? " · t50: " + esc(c.t50_status) : "")));
  } catch (e) {
    container.innerHTML = "";
    container.appendChild(el("div", { class: "error-banner" }, "curve load failed: " + esc(e.message)));
  }
}

// Re-render the curve plot with whatever model overlays are currently toggled on.
function redrawPlotWithOverlays(plotCtx) {
  if (!plotCtx || !plotCtx.base || !plotCtx.plotDiv) return;
  const b = plotCtx.base;
  const overlays = Object.values(plotCtx.overlays || {});
  renderCurvePlot(plotCtx.plotDiv, {
    x: b.x, y: b.y, fitted: b.fitted, overlays,
    title: null, censoring: b.censoring, ylabel: b.ylabel,
  });
}

// --------------------------------------------------------------------------- //
// MODEL COMPARISON / behaviour classification (per selected curve)
// --------------------------------------------------------------------------- //
async function loadModelComparison(container, seriesId, plotCtx) {
  try {
    const d = await api("/api/models/" + encodeURIComponent(seriesId));
    if (d.error) {
      container.innerHTML = "";
      container.appendChild(el("div", { class: "muted" }, "Model comparison unavailable: " + esc(d.error)));
      return;
    }
    renderModelComparison(container, d, plotCtx);
  } catch (e) {
    container.innerHTML = "";
    container.appendChild(el("div", { class: "muted" }, "Model comparison failed: " + esc(e.message)));
  }
}

const FAMILY_BADGE = {
  descriptive: "tier-descriptive", biphasic: "warn",
  autocatalytic: "tier-scaling", mechanistic: "tier-mechanistic",
};

function renderModelComparison(container, d, plotCtx) {
  container.innerHTML = "";
  const models = d.models || [];

  // ----- ranked table -----
  const tbl = el("table", { class: "model-cmp-table" });
  tbl.appendChild(el("tr", {}, ["", "model", "family", "R²", "AICc", "ΔAICc",
    "Akaike wt", "key params", "overlay"].map((h) => el("th", {}, h))));

  models.forEach((m, i) => {
    const isWin = m.winner;
    const color = OVERLAY_COLORS[i % OVERLAY_COLORS.length];
    const canOverlay = Array.isArray(m.x_grid) && Array.isArray(m.y_grid);

    // overlay toggle checkbox (only for the top models that carry a fitted grid)
    let toggle = null;
    if (canOverlay && plotCtx) {
      const cb = el("input", { type: "checkbox" });
      // show the winner overlaid by default
      if (isWin) {
        cb.checked = true;
        plotCtx.overlays[m.name] = { x: m.x_grid, y: m.y_grid, model: m.name, color };
      }
      cb.addEventListener("change", () => {
        if (cb.checked) {
          plotCtx.overlays[m.name] = { x: m.x_grid, y: m.y_grid, model: m.name, color };
        } else {
          delete plotCtx.overlays[m.name];
        }
        redrawPlotWithOverlays(plotCtx);
      });
      toggle = el("label", { class: "overlay-toggle", title: "overlay this model on the plot" }, [
        cb, el("span", { class: "swatch", style: "background:" + color }),
      ]);
    }

    const params = m.params || {};
    const keyParams = Object.entries(params).slice(0, 4)
      .map(([k, v]) => k + "=" + fmt(v, 3)).join(", ");

    const row = el("tr", { class: "model-cmp-row" + (isWin ? " winner" : "") }, [
      el("td", {}, isWin ? el("span", { class: "crown", title: "AICc winner" }, "♛") : ""),
      el("td", { class: "mono" }, esc(m.name)),
      el("td", {}, el("span", { class: "badge " + (FAMILY_BADGE[m.family] || "") }, esc(m.family))),
      el("td", { class: "num" }, fmt(m.r2, 4)),
      el("td", { class: "num" }, fmt(m.aicc, 1)),
      el("td", { class: "num" }, fmt(m.delta_aicc, 1)),
      el("td", { class: "num" }, fmt(m.akaike_weight, 3)),
      el("td", { class: "params-cell" }, esc(keyParams)),
      el("td", {}, toggle),
    ]);
    tbl.appendChild(row);
  });
  if ((d.non_converged || []).length) {
    container.appendChild(el("p", { class: "section-note" },
      "did not converge: " + d.non_converged.map((n) => esc(n.name)).join(", ")));
  }
  const scroll = el("div", { class: "model-cmp-scroll" }, [tbl]);
  container.appendChild(scroll);
  container.appendChild(el("p", { class: "section-note" },
    "Winner = lowest AICc. Tick the overlay swatches to show/hide each fitted model " +
    "on the curve plot above (the winner is shown by default)."));

  // redraw the plot so the winner overlay appears immediately
  redrawPlotWithOverlays(plotCtx);

  // ----- family legend (honesty notes, one line each) -----
  const legend = el("div", { class: "family-legend" });
  legend.appendChild(el("h4", {}, "What each family means"));
  Object.entries(d.family_notes || {}).forEach(([fam, note]) => {
    legend.appendChild(el("div", { class: "legend-row" }, [
      el("span", { class: "badge " + (FAMILY_BADGE[fam] || "") }, esc(fam)),
      el("span", { class: "legend-note" }, esc(note)),
    ]));
  });
  container.appendChild(legend);

  // ----- identifiability of the winner -----
  const id = d.identifiability || {};
  const flag = id.flag || "—";
  const flagBadge = (flag === "identifiable") ? "ok"
    : (flag === "practically_non_identifiable" || flag === "structurally_non_identifiable") ? "danger" : "warn";
  const idLine = el("div", { class: "ident-line" }, [
    el("span", { class: "badge " + flagBadge }, "identifiability: " + esc(flag)),
    id.condition_number != null
      ? el("span", { class: "badge" }, "cond № = " + fmt(id.condition_number, 1)) : null,
    id.log10_collinearity != null
      ? el("span", { class: "badge" }, "log₁₀ collinearity = " + fmt(id.log10_collinearity, 2)) : null,
  ]);
  container.appendChild(el("div", { class: "ident-card" }, [
    el("h4", {}, "Identifiability of the winner (" + esc(d.winner) + ")"),
    idLine,
    el("p", { class: "section-note" },
      id.note
        ? esc(id.note)
        : "A high condition number = a parameter COMBINATION is unconstrained by this " +
          "single curve (the sloppiest direction is the least-determined mix of parameters). " +
          "Practically non-identifiable means the fit is good but the parameters are not " +
          "pinned down individually."),
    id.sloppiest_direction
      ? el("p", { class: "section-note mono" }, "sloppiest direction: " +
          Object.entries(id.sloppiest_direction).map(([k, v]) => k + " " + fmt(v, 2)).join("  ·  ")) : null,
  ]));

  // ----- bootstrap selection stability (Feature 3) -----
  const bs = d.bootstrap_selection || {};
  const bsCard = el("div", { class: "card tight bootstrap-card" }, [el("h4", {}, "Bootstrap selection stability")]);
  if (bs.selection_frequencies || bs.selection_stability != null) {
    bsCard.appendChild(el("p", { class: "muted" },
      "From the M3-sampled bootstrap (selection runs inside the bootstrap): the winner " +
      "was selected in " + pct(bs.selection_stability) + " of resamples."));
    const freqs = bs.selection_frequencies || {};
    const fl = el("div", { class: "pill-row" });
    Object.entries(freqs).forEach(([k, v]) =>
      fl.appendChild(el("span", { class: "badge" }, esc(k) + ": " + pct(v))));
    bsCard.appendChild(fl);
    if (bs.t50_multimodal != null) {
      bsCard.appendChild(el("p", { class: "section-note" },
        "t50 selection " + (bs.t50_multimodal ? "is MULTIMODAL (unstable across models)" : "is stable across models") + "."));
    }
  } else {
    bsCard.appendChild(el("p", { class: "section-note" }, esc(bs.note ||
      "point selection (AICc); bootstrap-enveloped selection is a build-time job (M3).")));
  }
  container.appendChild(bsCard);

  // ----- honesty note (Tier-B not a mechanism call) -----
  container.appendChild(el("div", { class: "honesty-inline" }, [
    el("p", {}, esc(d.honesty ||
      "Tier-B mechanistic fits are shown for comparison only — a high R² is NOT a " +
      "licensed mechanism. M5 still refuses a mechanism call on a single curve.")),
  ]));
}

// --------------------------------------------------------------------------- //
// ANALYZE
// --------------------------------------------------------------------------- //
function initAnalyze() {
  $("#analyze-run").addEventListener("click", runAnalyze);
  $("#analyze-example").addEventListener("click", loadExample);
  $("#analyze-file").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => { $("#analyze-text").value = reader.result; };
    reader.readAsText(f);
  });
}

function loadExample() {
  // synthetic sigmoid (logistic) with a little noise
  const lines = [];
  for (let i = 0; i < 24; i++) {
    const t = i * 4;
    const s = 1 / (1 + Math.exp(-0.18 * (t - 42)));
    const noise = (Math.sin(t * 1.3) * 0.012);
    lines.push(t + "\t" + (s + noise).toFixed(4));
  }
  $("#analyze-text").value = lines.join("\n");
  $("#analyze-conc").value = "10";
  $("#analyze-assay").value = "ThT";
}

function parseColumns(text) {
  const x = [], y = [];
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    const s = line.trim();
    if (!s || /[a-zA-Z]{2,}/.test(s.replace(/e[-+]?\d/gi, ""))) {
      // skip header-ish / non-numeric lines (but allow scientific notation 'e')
      if (!/^[-+0-9.\s,;\te]+$/.test(s)) continue;
    }
    const parts = s.split(/[\s,;\t]+/).filter((p) => p !== "");
    if (parts.length < 2) continue;
    const a = parseFloat(parts[0]), b = parseFloat(parts[1]);
    if (isFinite(a) && isFinite(b)) { x.push(a); y.push(b); }
  }
  return { x, y };
}

async function runAnalyze() {
  const errBox = $("#analyze-error");
  errBox.classList.add("hidden");
  const resBox = $("#analyze-result");
  const { x, y } = parseColumns($("#analyze-text").value);
  if (x.length < 3) {
    errBox.textContent = "Need at least 3 valid (time, signal) rows. Parsed " + x.length + ".";
    errBox.classList.remove("hidden");
    return;
  }
  const body = { x, y };
  const conc = parseFloat($("#analyze-conc").value);
  if (isFinite(conc)) body.concentration_uM = conc;
  body.assay = $("#analyze-assay").value;

  resBox.innerHTML = "";
  resBox.appendChild(el("div", { class: "loading" }, "running engine…"));
  try {
    const r = await fetch("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (d.error) {
      resBox.innerHTML = "";
      resBox.appendChild(el("div", { class: "error-banner" }, "Engine: " + esc(d.error)));
      return;
    }
    renderAnalyzeResult(resBox, d);
  } catch (e) {
    resBox.innerHTML = "";
    resBox.appendChild(el("div", { class: "error-banner" }, "Request failed: " + esc(e.message)));
  }
}

// Conformal-prediction card: the finite-sample-valid t50 interval + regime
// prediction set, applied to THIS curve from the frozen Service-C calibration.
// Honest: marginal vs Mondrian, one-sided for censored, thin-stratum fallback.
function renderConformal(box, conf, t50_hat, regime) {
  if (!conf) return;
  const card = el("div", { class: "card" }, [
    el("h3", {}, "Conformal prediction interval"),
  ]);
  if (!conf.available) {
    card.appendChild(el("p", { class: "muted" },
      "Not available: " + esc(conf.reason || "conformal calibration not loaded.")));
    box.appendChild(card);
    return;
  }
  const tgt = conf.target_coverage != null ? Math.round(conf.target_coverage * 100) : 90;
  card.appendChild(el("p", { class: "section-note" },
    "Finite-sample-valid ≥" + tgt + "% coverage, distribution-free, calibrated on " +
    "Service C ground truth (Mondrian-conditional on censoring / assay / regime). " +
    esc(conf.version || "")));

  // t50 interval
  const iv = conf.t50_interval || {};
  if (iv.available) {
    const loS = fmt(iv.lo);
    const hiS = iv.hi_is_infinite ? "+∞" : fmt(iv.hi);
    const oneSided = iv.one_sided
      ? " (one-sided: a right-censored t50 is a lower bound, so the upper end is open)"
      : "";
    const fb = iv.marginal_fallback
      ? " · stratum too thin → fell back to the MARGINAL quantile (flagged)"
      : " · stratum: " + esc(iv.stratum_used || "—");
    card.appendChild(el("div", { class: "conformal-row" }, [
      el("span", { class: "badge ok" }, "t50 ∈ [" + loS + ", " + hiS + "] h"),
      el("span", { class: "muted", style: "font-size:.82rem" },
        "point t50=" + fmt(t50_hat) + " · q=" + fmt(iv.q) + oneSided + fb),
    ]));
  } else {
    card.appendChild(el("p", { class: "muted" },
      "t50 interval: " + esc(iv.reason || "no point t50 to anchor an interval.")));
  }

  // regime prediction set
  const rs = conf.regime_set || {};
  if (rs.available) {
    const setStr = (rs.set || []).join(", ") || "—";
    const ambiguous = rs.set_size > 1;
    const saturated = rs.regime_unresolvable_in_stratum;
    let note;
    if (saturated) {
      note = "all regimes — the engine cannot resolve the regime in this " +
        "(censoring/assay) stratum (still a valid ≥" + tgt + "% set)";
    } else if (ambiguous) {
      note = "size " + rs.set_size + " — the engine's evidence is genuinely split";
    } else {
      note = "singleton — confident regime call";
    }
    card.appendChild(el("div", { class: "conformal-row" }, [
      el("span", { class: "badge " + (ambiguous ? "warn" : "ok") },
        "regime set {" + esc(setStr) + "}"),
      el("span", { class: "muted", style: "font-size:.82rem" }, note),
    ]));
  }
  card.appendChild(el("p", { class: "section-note" },
    "Marginal vs Mondrian: the interval/set uses the matching " +
    "(censoring · assay · regime) stratum when it has enough calibration points, " +
    "else it falls back to the pooled marginal quantile (flagged above)."));
  box.appendChild(card);
}

function renderAnalyzeResult(box, d) {
  box.innerHTML = "";

  // summary badges
  const mech = d.mechanistic || {};
  box.appendChild(el("div", { class: "card" }, [
    el("div", { style: "display:flex; gap:.5rem; flex-wrap:wrap; align-items:center" }, [
      el("span", { class: "badge tier-descriptive" }, "regime: " + esc(d.regime || "—")),
      el("span", { class: "badge" }, "yield: " + esc(d.information_yield || "—")),
      el("span", { class: "badge" }, "best model: " + esc(d.best_model || "—")),
      d.model_r2 != null ? el("span", { class: "badge ok" }, "R²=" + fmt(d.model_r2)) : null,
      el("span", { class: "badge " + (mech.licensed ? "ok" : "danger") },
        mech.licensed ? "mechanism licensed" : "mechanism NOT licensed"),
    ]),
    d.regime_definition ? el("p", { class: "section-note" }, esc(d.regime_definition)) : null,
  ]));

  // plot
  const plotCard = el("div", { class: "card" }, [el("h3", {}, "Curve & fitted overlay")]);
  const plotDiv = el("div", { class: "plot-wrap" });
  plotCard.appendChild(plotDiv);
  box.appendChild(plotCard);
  const y = (d.y_processed && d.y_processed.length) ? d.y_processed : d.y;
  renderCurvePlot(plotDiv, {
    x: d.x, y, fitted: d.fitted,
    ylabel: d.y_processed ? "normalized signal" : "signal",
  });

  // features
  const f = d.features || {};
  const featGrid = el("div", { class: "feature-grid" });
  const items = [
    ["t50", f.t50, f.t50_status],
    ["lag time", f.lag_time, f.lag_status],
    ["lag/t50 ratio", f.lag_to_t50_ratio, null],
    ["inflection time", f.inflection_time, null],
    ["transition sharpness", f.transition_sharpness, null],
  ];
  items.forEach(([l, v, tag]) => featGrid.appendChild(el("div", { class: "feat" }, [
    el("div", { class: "fl" }, l),
    el("div", { class: "fv" }, fmt(v)),
    tag ? el("div", { class: "ft badge " + (tag === "point" ? "ok" : "censor") }, esc(tag)) : null,
  ])));
  box.appendChild(el("div", { class: "card" }, [el("h3", {}, "Features"), featGrid]));

  // CONFORMAL PREDICTION — finite-sample-valid t50 interval + regime set
  renderConformal(box, d.conformal, f.t50, d.regime);

  // M1 triage
  const m1 = d.m1 || {};
  box.appendChild(el("div", { class: "card tight" }, [
    el("h4", {}, "M1 triage"),
    el("div", { class: "pill-row" }, [
      el("span", { class: "badge" }, "fittability: " + esc(m1.fittability_class || "?")),
      el("span", { class: "badge " + (m1.censoring_class === "none" ? "ok" : "censor") }, "censoring: " + esc(m1.censoring_class || "?")),
      el("span", { class: "badge" }, "handling: " + esc(m1.recommended_handling || "?")),
      el("span", { class: "badge" }, esc(m1.normalization_mode || "?")),
    ]),
  ]));

  // mechanistic + caveats (honesty)
  const honest = el("div", { class: "card" }, [el("h3", {}, "Honesty: mechanism gates + caveats")]);
  honest.appendChild(el("p", { class: "muted" },
    "Equivalence class: " + esc((mech.equivalence_class || []).join(", ") || "—") +
    " · degeneracy: " + esc(mech.degeneracy || "—")));
  if ((mech.gates_failed || []).length) {
    const g = el("ul", { class: "caveat-list gate-list" });
    mech.gates_failed.forEach((x) => g.appendChild(el("li", {}, "gate failed: " + esc(x))));
    honest.appendChild(g);
  }
  const cav = el("ul", { class: "caveat-list" });
  (d.caveats || []).forEach((c) => cav.appendChild(el("li", {}, esc(c))));
  honest.appendChild(cav);
  honest.appendChild(el("p", { class: "section-note" }, "engine build " + esc(d.engine_build_id || "")));
  box.appendChild(honest);
}

// --------------------------------------------------------------------------- //
// footer / attribution
// --------------------------------------------------------------------------- //
function renderFooter(attr) {
  if (!attr) return;
  GLOBAL_ATTRIBUTION = attr;
  const inner = $("#footer-inner");
  inner.innerHTML = "";
  inner.appendChild(el("div", {}, [
    el("strong", {}, "Data: "),
    document.createTextNode((attr.source_dataset || "CPAD 2.0") + " — "),
    el("em", {}, esc(attr.citation || "")),
  ]));
  inner.appendChild(el("div", {}, esc(attr.note ||
    "Derived analytical results + single curves only; raw bulk data not served.")));
  inner.appendChild(el("div", {}, [
    el("strong", {}, "Engine build: "),
    document.createTextNode(attr.engine_build_id || "?"),
    document.createTextNode("  ·  Local MVP (stdlib server + vanilla JS). FastAPI/React is the documented production path."),
  ]));
}

// --------------------------------------------------------------------------- //
// boot
// --------------------------------------------------------------------------- //
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initAnalyze();
  // Browse is the default landing experience: choose & analyze a real CPAD dataset.
  loadBrowse();
});
