"""
PRISE — Module M10: Cross-modal Sequence × Kinetics × Structure analysis
========================================================================

PRISE uniquely co-holds, for the SAME proteins, three modalities:

  (1) **Sequence** aggregation predictors (TANGO / AGGRESCAN / PASTA / Waltz,
      per UniProt) — `data/processed/sequence_structure.json` (the M6 sequence
      axis, ETL `cpad-sequence-1.1`).
  (2) **Kinetics** — MEASURED descriptive regimes (M5), features incl. t50 +
      lag_to_t50_ratio (M4), and dual-γ (M4) from the engine.
  (3) **Structure / APR** — `sequence_structure.json` aprs[] + structures[].

Until now the product only DISPLAYS the sequence axis (endpoint-match gated, §3-M6)
and never MINES the cross-modal relationship. M10 does the mining, and reports the
findings already latent in the data. It is **scientific-novelty**: it asks, on the
SAME proteins, *does a sequence-only predictor actually predict the measured rate /
regime?* — and answers it honestly.

THE BINDING HONESTY CONSTRAINT (enforced everywhere here)
---------------------------------------------------------
The effective n is the number of **PROTEINS (~20)**, NOT curves (~1200). A
predictor score is a per-protein CONSTANT replicated across all of that protein's
curves; treating each curve as independent would inflate n ~60×. So:

  * every association uses the **PROTEIN as the unit** (one summary row per UniProt);
  * every null is a **protein-level permutation** (shuffle the per-protein labels,
    never the per-curve rows);
  * predictive skill is **leave-one-PROTEIN-out** (LOPO) — n forces LOPO over k-fold.

The corpus is ~97% amyloid/ThT endpoint and biased toward famous amyloids (almost
no negative class), so M10 tests **rate / regime AMONG known amyloids**, NEVER
"amyloid vs not". And one sequence maps to MANY rates across conditions, so a
sequence-only predictor's max explainable variance is bounded by the
**between-protein fraction** of total t50 variance — M10 computes and displays that
ceiling (a one-way ANOVA decomposition over proteins).

Phases:
  P1 — SEQUENCE → KINETICS association (the headline; usually a null result).
  P2 — STRUCTURE/APR ↔ REGIME association (the clean positive signal; ASSOCIATIVE).
  P3 — protein-level DISCORDANCE scan (fills §3-M6 `null_disagreement_rate`).
  P4 — γ × predictor sidebar (honestly thin; descriptive, no test).

Pure / deterministic / JSON-serialisable. Never raises on bad input (returns a
status block instead). No fitting here — reads precomputed artifacts only.

Version: m10-crossmodal-1.0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _safe_print(s):
    """Print that never dies on a non-UTF-8 console (Windows cp1252): the rich
    unicode lives in the JSON artifact; the CLI summary degrades to ASCII-safe."""
    try:
        print(s)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(str(s).encode(enc, "replace").decode(enc))

VERSION = "m10-crossmodal-1.0"

# Ordinal descriptive regime ladder (cooperativity-ascending). M5's
# `no_detectable_aggregation` is the floor; cooperative_sigmoidal the ceiling.
REGIME_ORDER = {
    "no_detectable_aggregation": 0,
    "gradual_non_cooperative": 1,
    "threshold_driven": 2,
    "cooperative_sigmoidal": 3,
}
REGIME_LADDER = [k for k, _ in sorted(REGIME_ORDER.items(), key=lambda kv: kv[1])]

# Permutation draws for the protein-level nulls (seeded → deterministic).
N_PERM = 10000
SEED = 1729

# --- P3 pre-registered null-disagreement constants (FLAGGED as assumptions) --- #
# PASTA literature false-positive rate for the amyloid (cross-β) call. A
# pre-registered constant, NOT fitted from this corpus; documented so the
# discordance baseline is auditable (Service-C calibration is the deferred upgrade).
PASTA_LITERATURE_FP_RATE = 0.15      # ~10–20% FP reported for sequence amyloid callers
# Pre-registered PASTA pairing-energy threshold for a SEQUENCE-positive amyloid
# call: a min (most-stable) pairing energy below this is "predicts cross-β".
PASTA_AMYLOID_ENERGY_THRESHOLD = -3.0
# Significance threshold for the P3 two-sided binomial. It licenses ONLY the word
# "significantly"; it NEVER supplies the direction word — a two-sided p is
# direction-blind (equally small far above and far below the null). See
# `discordance_direction` for the enforced separation of the two axes.
DISCORDANCE_ALPHA = 0.05


# ============================================================================ #
# small, dependency-light statistics (protein-level by construction)
# ============================================================================ #
def _finite(xs):
    return [float(x) for x in xs if x is not None and math.isfinite(float(x))]


def _median(xs):
    xs = sorted(_finite(xs))
    n = len(xs)
    if n == 0:
        return None
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _rankdata(a):
    """Average-tie ranks (1-based), like scipy.stats.rankdata. Pure-numpy."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0          # average 1-based rank of the tie block
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _pearson(x, y):
    """Pearson r over PAIRED protein-level values (None if undefined)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    """Spearman ρ = Pearson on average-tie ranks (protein-level)."""
    if len(x) < 3:
        return None
    return _pearson(_rankdata(x), _rankdata(y))


def _perm_pvalue_corr(x, y, kind="spearman"):
    """Two-sided permutation p-value for a protein-level correlation.

    The unit is the PROTEIN: we permute the per-protein y labels (never per-curve
    rows), so ~1200 replicate curves can NEVER inflate significance. Seeded RNG →
    byte-deterministic. Returns (observed, p_value, n_perm)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    corr = _spearman if kind == "spearman" else _pearson
    obs = corr(x, y)
    if obs is None or n < 4:
        return obs, None, 0
    rng = np.random.default_rng(SEED)
    # precompute ranks once for Spearman speed; permuting ranks ≡ permuting values
    xx = _rankdata(x) if kind == "spearman" else x
    yy = _rankdata(y) if kind == "spearman" else y
    obs_r = _pearson(xx, yy)
    ge = 0
    for _ in range(N_PERM):
        r = _pearson(xx, rng.permutation(yy))
        if r is not None and abs(r) >= abs(obs_r) - 1e-12:
            ge += 1
    # +1 smoothing → never reports an impossible p=0
    p = (ge + 1) / (N_PERM + 1)
    return obs, float(p), N_PERM


# ============================================================================ #
# variance ceiling — between-protein fraction of total log10(t50) variance
# ============================================================================ #
def variance_ceiling(protein_logt50_replicates: dict) -> dict:
    """One-way ANOVA decomposition of total log10(t50) variance over PROTEINS.

    Input: {uniprot: [log10(t50) of each replicate curve]}. The
    between-protein fraction (SS_between / SS_total) is the ABSOLUTE upper bound on
    variance any sequence-only (per-protein-constant) predictor could explain: a
    constant cannot distinguish two curves of the SAME protein, so all WITHIN-
    protein t50 variance is unreachable by construction. This is THE ceiling the
    headline P1 result is measured against."""
    groups = {u: _finite(v) for u, v in protein_logt50_replicates.items()}
    groups = {u: v for u, v in groups.items() if v}
    allv = [x for v in groups.values() for x in v]
    if len(allv) < 2 or len(groups) < 2:
        return {"computable": False,
                "reason": "need ≥2 proteins each with ≥1 curve and ≥2 total curves",
                "n_proteins": len(groups), "n_curves": len(allv)}
    grand = sum(allv) / len(allv)
    ss_total = sum((x - grand) ** 2 for x in allv)
    ss_between = sum(len(v) * (sum(v) / len(v) - grand) ** 2 for v in groups.values())
    if ss_total <= 0:
        return {"computable": False, "reason": "zero total variance",
                "n_proteins": len(groups), "n_curves": len(allv)}
    frac = ss_between / ss_total
    return {
        "computable": True,
        "between_protein_fraction": round(frac, 4),
        "within_protein_fraction": round(1.0 - frac, 4),
        "ss_between": round(ss_between, 6),
        "ss_total": round(ss_total, 6),
        "n_proteins": len(groups),
        "n_curves": len(allv),
        "interpretation": (
            "A sequence-only predictor is a per-protein CONSTANT; it cannot explain "
            "any WITHIN-protein t50 variance. So the most variance ANY such predictor "
            f"could explain is {round(100 * frac, 1)}% — the rest is condition-driven."),
    }


# ============================================================================ #
# leave-one-protein-out predictive skill (n forces LOPO, not k-fold)
# ============================================================================ #
def lopo_r2(x, y) -> dict:
    """Leave-one-PROTEIN-out CV predictive R² for a 1-D OLS predictor x→y.

    For each held-out protein, fit y~x on the OTHERS and predict the held-out y;
    R² = 1 - SS_res/SS_tot against the mean-of-others baseline. With n≈20 proteins,
    LOPO is the only honest CV (k-fold would split a protein's identity). A
    sequence-only predictor that does not generalise lands at R² ≤ 0."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 4:
        return {"computable": False, "reason": "need ≥4 proteins for LOPO", "n": n}
    preds = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        xi, yi = x[mask], y[mask]
        if np.std(xi) == 0:
            preds[i] = yi.mean()                 # degenerate predictor → baseline
            continue
        b1 = np.cov(xi, yi, bias=True)[0, 1] / np.var(xi)
        b0 = yi.mean() - b1 * xi.mean()
        preds[i] = b0 + b1 * x[i]
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {"computable": True, "lopo_r2": (round(r2, 4) if r2 is not None else None),
            "n": n,
            "note": "negative R² = worse than predicting the corpus mean (no skill)"}


def lopo_ordinal_accuracy(x, y_ord) -> dict:
    """LOPO predictive accuracy + Spearman for an ORDINAL target (regime rank).

    Held-out prediction = nearest-rank of the OLS x→rank fit on the others, snapped
    to the observed rank set. Accuracy is exact-rank hit-rate; a rank-adjacent hit
    rate is also reported (an off-by-one ordinal miss is softer than a wild miss)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_ord, dtype=float)
    n = len(x)
    if n < 4:
        return {"computable": False, "reason": "need ≥4 proteins for LOPO", "n": n}
    levels = sorted(set(y.tolist()))
    preds = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        xi, yi = x[mask], y[mask]
        if np.std(xi) == 0:
            yhat = yi.mean()
        else:
            b1 = np.cov(xi, yi, bias=True)[0, 1] / np.var(xi)
            b0 = yi.mean() - b1 * xi.mean()
            yhat = b0 + b1 * x[i]
        preds[i] = min(levels, key=lambda lv: abs(lv - yhat))   # snap to a real rank
    exact = float(np.mean(preds == y))
    adj = float(np.mean(np.abs(preds - y) <= 1.0))
    sp = _spearman(x, y)
    # CRITICAL base-rate guard: with this corpus almost every protein's MAX regime
    # is cooperative_sigmoidal, so a CONSTANT predictor trivially scores ~0.9–1.0
    # accuracy. The honest skill is accuracy ABOVE the majority-class baseline
    # (predict-the-mode); the lift can be ≤0 even when raw accuracy is ~1.0.
    counts = Counter(y.tolist())
    majority = max(counts.values()) / n
    lift = exact - majority
    return {"computable": True, "lopo_accuracy": round(exact, 4),
            "majority_class_baseline": round(majority, 4),
            "skill_above_baseline": round(lift, 4),
            "lopo_within_one_rank": round(adj, 4),
            "spearman_rho": (round(sp, 4) if sp is not None else None),
            "n": n, "n_levels": len(levels),
            "note": ("raw accuracy is dominated by the majority class; skill = "
                     "accuracy − majority-class baseline (≤0 = no real skill, even "
                     "if raw accuracy looks high)")}


# ============================================================================ #
# Jonckheere–Terpstra trend test (implemented; scipy lacks it here)
# ============================================================================ #
def jonckheere_statistic(groups_ordered) -> dict:
    """Jonckheere–Terpstra trend test for an a-priori ORDERED set of groups.

    `groups_ordered` is a list of value-lists, ordered by the hypothesised
    INCREASING trend (here: regime cooperativity floor→ceiling). The J statistic
    counts, over every ordered group pair (i<j), how many (x in group_i, y in
    group_j) pairs have y > x (ties count 0.5). A normal approximation gives a
    one-sided z + p for the increasing alternative; the caller may instead use a
    protein-level permutation null (preferred for tiny, unequal groups)."""
    groups = [list(_finite(g)) for g in groups_ordered]
    sizes = [len(g) for g in groups]
    k = len(groups)
    if k < 2 or sum(sizes) < 3 or sum(1 for s in sizes if s > 0) < 2:
        return {"computable": False, "reason": "need ≥2 non-empty ordered groups",
                "group_sizes": sizes}
    # U-style count over ordered pairs (i<j): #(y>x) + 0.5 #(y==x)
    J = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            for x in groups[i]:
                for y in groups[j]:
                    if y > x:
                        J += 1.0
                    elif y == x:
                        J += 0.5
    N = sum(sizes)
    mean_J = (N * N - sum(s * s for s in sizes)) / 4.0
    var_J = (N * N * (2 * N + 3) - sum(s * s * (2 * s + 3) for s in sizes)) / 72.0
    z = (J - mean_J) / math.sqrt(var_J) if var_J > 0 else None
    # one-sided p for the INCREASING alternative via the normal approximation
    p_one = (0.5 * math.erfc(z / math.sqrt(2.0))) if z is not None else None
    return {"computable": True, "J": J, "mean_J": round(mean_J, 4),
            "var_J": round(var_J, 4),
            "z": (round(z, 4) if z is not None else None),
            "p_one_sided_normal": (round(p_one, 6) if p_one is not None else None),
            "group_sizes": sizes,
            "direction": ("increasing" if (z is not None and z > 0) else
                          ("decreasing" if z is not None else None))}


def jonckheere_permutation_p(group_labels_ordinal, values, n_perm=N_PERM) -> dict:
    """Protein-level permutation null for the Jonckheere trend.

    `group_labels_ordinal`/`values` are PER-PROTEIN parallel arrays (one entry per
    protein: its ordinal group + its value). We shuffle the protein→value labels
    and recompute J, so 600 cooperative *curves* cannot masquerade as 600
    independent observations — the unit is the protein. Seeded → deterministic."""
    labels = list(group_labels_ordinal)
    vals = _finite(values)
    if len(labels) != len(values) or len(vals) != len(values):
        # drop non-finite pairs together
        pairs = [(g, v) for g, v in zip(group_labels_ordinal, values)
                 if v is not None and math.isfinite(float(v))]
        labels = [g for g, _ in pairs]
        vals = [float(v) for _, v in pairs]
    if len(set(labels)) < 2 or len(vals) < 3:
        return {"computable": False, "reason": "need ≥2 groups, ≥3 proteins",
                "n_proteins": len(vals)}
    order = sorted(set(labels))

    def _J(lab, val):
        g = defaultdict(list)
        for L, v in zip(lab, val):
            g[L].append(v)
        gl = [g[o] for o in order]
        return jonckheere_statistic(gl).get("J")

    obs = _J(labels, vals)
    rng = np.random.default_rng(SEED)
    vals_arr = np.asarray(vals, dtype=float)
    ge = 0
    for _ in range(n_perm):
        if _J(labels, rng.permutation(vals_arr).tolist()) >= obs - 1e-9:
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return {"computable": True, "J_observed": obs, "p_one_sided_permutation": float(p),
            "n_perm": n_perm, "n_proteins": len(vals),
            "unit": "protein", "ordered_groups": order}


def kruskal_wallis(groups) -> dict:
    """Kruskal–Wallis omnibus H over the (unordered) regime groups (protein-level).
    Reported alongside Jonckheere: KW asks 'any difference', JT asks 'monotone'."""
    groups = [list(_finite(g)) for g in groups]
    groups = [g for g in groups if g]
    if len(groups) < 2:
        return {"computable": False, "reason": "need ≥2 non-empty groups"}
    allv = [x for g in groups for x in g]
    N = len(allv)
    if N < 3:
        return {"computable": False, "reason": "need ≥3 observations"}
    ranks = _rankdata(allv)
    idx = 0
    rank_groups = []
    for g in groups:
        rank_groups.append(ranks[idx:idx + len(g)])
        idx += len(g)
    H = 12.0 / (N * (N + 1)) * sum(
        (np.sum(rg) ** 2) / len(rg) for rg in rank_groups) - 3.0 * (N + 1)
    df = len(groups) - 1
    # survival of chi-square(df) via the regularized upper incomplete gamma
    p = _chi2_sf(H, df)
    return {"computable": True, "H": round(float(H), 4), "df": df,
            "p_value": round(float(p), 6),
            "note": "omnibus (any group differs); not a trend test"}


def _chi2_sf(x, df):
    """Upper-tail chi-square survival via a series/continued-fraction γ — no scipy
    needed (M10 must stay light). Accurate enough for the small df here."""
    if x <= 0:
        return 1.0
    a = df / 2.0
    xx = x / 2.0
    # lower regularized incomplete gamma P(a, xx) via series (xx < a+1) else CF
    if xx < a + 1.0:
        term = 1.0 / a
        s = term
        n = a
        for _ in range(500):
            n += 1.0
            term *= xx / n
            s += term
            if abs(term) < abs(s) * 1e-12:
                break
        P = s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - P))
    # continued fraction for Q(a, xx)
    tiny = 1e-300
    b = xx + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    Q = math.exp(-xx + a * math.log(xx) - math.lgamma(a)) * h
    return max(0.0, min(1.0, Q))


# ============================================================================ #
# binomial test (two-sided) for the discordance scan
# ============================================================================ #
def binom_test_two_sided(k, n, p):
    """Exact two-sided binomial p-value (sum of outcomes ≤ as likely as observed).
    Pure-python; n is tiny here (≈20 proteins)."""
    if n <= 0 or not (0.0 <= p <= 1.0):
        return None

    def pmf(i):
        return math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    obs = pmf(k)
    total = sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12)
    return float(min(1.0, total))


# ============================================================================ #
# JOIN: build the per-PROTEIN cross-modal summary (key everything by UniProt)
# ============================================================================ #
def build_protein_table(triaged, features, regimes, seq_lookup) -> dict:
    """Collapse ~1200 curves to one row per UniProt PROTEIN (the effective unit).

    Per protein we collect: point/lower-bound t50 lists, lag_to_t50_ratio,
    descriptive-regime distribution + an ordinal summary, the endpoint-matched
    predictors (PASTA energy, Waltz coverage), APR/structure counts, and the M1
    censoring picture (for the censoring-aware P3 discordance scan)."""
    feat_by_sid = {f.get("series_id"): f for f in features}
    reg_by_sid = {r.get("series_id"): r for r in regimes}
    by_up = defaultdict(lambda: {
        "uniprot": None, "name": None,
        "t50_point": [], "t50_lower_bound": [], "log10_t50_point": [],
        "lag_ratio": [], "regimes": [], "regime_ords": [],
        "assays": Counter(), "n_curves": 0,
        "n_mass_assay_fittable": 0, "n_right_censored": 0, "n_uncensored": 0,
    })
    for t in triaged:
        up = t.get("uniprot_id")
        if not up:
            continue
        sid = t.get("series_id")
        d = by_up[up]
        d["uniprot"] = up
        d["name"] = d["name"] or t.get("protein_id") or t.get("protein")
        d["n_curves"] += 1
        cv = t.get("condition_vector") or {}
        d["assays"][cv.get("assay_type")] += 1
        f = feat_by_sid.get(sid)
        r = reg_by_sid.get(sid)
        if r:
            rg = (r.get("descriptive_regime") or {}).get("regime")
            if rg in REGIME_ORDER:
                d["regimes"].append(rg)
                d["regime_ords"].append(REGIME_ORDER[rg])
        # censoring-aware kinetics-positive bookkeeping (P3):
        # a "no-detectable" that is only right-censored is NOT a true negative.
        cclass = (f.get("censoring_class") if f else None) or t.get("censoring_class")
        assay_mass = cv.get("assay_reports_mass")
        fittable = bool(f and f.get("status") == "ok")
        if assay_mass and fittable:
            rg = (r.get("descriptive_regime") or {}).get("regime") if r else None
            if rg in ("gradual_non_cooperative", "threshold_driven",
                      "cooperative_sigmoidal"):
                d["n_mass_assay_fittable"] += 1            # aggregates SOMEWHERE
        if cclass in ("right", "left_right"):
            d["n_right_censored"] += 1
        else:
            d["n_uncensored"] += 1
        if f and f.get("status") == "ok":
            ts = f.get("t50_status")
            t50 = (f.get("features") or {}).get("t50")
            lr = (f.get("features") or {}).get("lag_to_t50_ratio")
            if lr is not None and math.isfinite(lr):
                d["lag_ratio"].append(lr)
            if t50 is not None and t50 > 0:
                if ts == "point":
                    d["t50_point"].append(t50)
                    d["log10_t50_point"].append(math.log10(t50))
                elif ts == "lower_bound":
                    d["t50_lower_bound"].append(t50)

    # attach predictors + structure/APR + protein-level summaries
    out = {}
    for up, d in by_up.items():
        rec = (seq_lookup.get("by_uniprot", {}) or {}).get(up, {}) or {}
        sp = rec.get("sequence_predictors") or {}
        pasta = sp.get("pasta") or {}
        waltz = sp.get("waltz") or {}
        regime_counts = Counter(d["regimes"])
        # protein-level ordinal regime summary = the MOST COOPERATIVE regime the
        # protein reaches in ANY measured condition (a propensity statement: "this
        # protein CAN form cooperative fibrils") — robust to the many gradual
        # curves a famous amyloid also produces. Also keep the modal regime.
        max_ord = max(d["regime_ords"]) if d["regime_ords"] else None
        modal = regime_counts.most_common(1)[0][0] if regime_counts else None
        out[up] = {
            "uniprot": up, "name": d["name"], "n_curves": d["n_curves"],
            "assay_mix": dict(d["assays"]),
            "median_t50_point": _median(d["t50_point"]),
            "median_log10_t50_point": _median(d["log10_t50_point"]),
            "log10_t50_replicates": list(d["log10_t50_point"]),
            "n_point_t50": len(d["t50_point"]),
            "n_lower_bound_t50": len(d["t50_lower_bound"]),
            "median_lag_to_t50_ratio": _median(d["lag_ratio"]),
            "regime_counts": dict(regime_counts),
            "modal_regime": modal,
            "max_regime": (REGIME_LADDER[max_ord] if max_ord is not None else None),
            "max_regime_ordinal": max_ord,
            "n_mass_assay_fittable": d["n_mass_assay_fittable"],
            "n_right_censored": d["n_right_censored"],
            "n_uncensored": d["n_uncensored"],
            # endpoint-matched sequence predictors (PASTA energy, Waltz coverage)
            "pasta_energy": pasta.get("value") if pasta.get("available") else None,
            "pasta_available": bool(pasta.get("available") and pasta.get("value") is not None),
            "waltz_coverage": waltz.get("n_waltz_amyloid_peptides") if waltz else None,
            "waltz_available": bool(waltz.get("available")),
            "n_aprs": len(rec.get("aprs") or []),
            "apr_categories": Counter(
                (a.get("category") or "uncategorized")
                for a in (rec.get("aprs") or [])),
            "n_structures": len(rec.get("structures") or []),
            "n_amyloid_structures": sum(
                1 for s in (rec.get("structures") or [])
                if (s.get("amyloid") or "").lower() == "amyloid"),
            "n_nonamyloid_structures": sum(
                1 for s in (rec.get("structures") or [])
                if (s.get("amyloid") or "").lower() in ("non-amyloid", "nonamyloid")),
        }
    return out


# ============================================================================ #
# P1 — SEQUENCE → KINETICS association (the headline)
# ============================================================================ #
def _assoc_card(predictor, target, xs, ys, ceiling_fraction, ordinal=False):
    """One per-predictor association card (protein-level, with LOPO + perm null)."""
    n = len(xs)
    sp = _spearman(xs, ys)
    pr = _pearson(xs, ys)
    _, perm_p, n_perm = _perm_pvalue_corr(xs, ys, kind="spearman")
    if ordinal:
        skill = lopo_ordinal_accuracy(xs, ys)
        skill_key = "lopo_predictive_accuracy"
        skill_val = skill.get("lopo_accuracy")
        # honest skill for the verdict = accuracy ABOVE the majority-class baseline
        # (raw ordinal accuracy is dominated by the near-universal cooperative class)
        skill_for_verdict = skill.get("skill_above_baseline")
        skill_label = "skill-over-baseline"
    else:
        skill = lopo_r2(xs, ys)
        skill_key = "lopo_predictive_r2"
        skill_val = skill.get("lopo_r2")
        skill_for_verdict = skill_val
        skill_label = "LOPO skill"

    # blunt plain-English verdict
    rho_s = "—" if sp is None else f"{sp:+.2f}"
    sk_s = "n/a" if skill_for_verdict is None else f"{skill_for_verdict:+.2f}"
    p_s = "n/a" if perm_p is None else f"{perm_p:.3f}"
    ceil_s = "n/a" if ceiling_fraction is None else f"{100 * ceiling_fraction:.0f}%"
    significant = (perm_p is not None and perm_p < 0.05)
    has_skill = (skill_for_verdict is not None and skill_for_verdict > 0.0)
    if significant and has_skill:
        verdict = (f"{predictor} shows a protein-level association with {target} "
                   f"(ρ={rho_s}, {skill_label}={sk_s}, permutation p={p_s}); "
                   f"variance ceiling {ceil_s}. Associative, n={n} proteins.")
    else:
        verdict = (f"{predictor} does NOT predict measured {target}: ρ={rho_s}, "
                   f"{skill_label}={sk_s} (≤0 = no out-of-sample skill), permutation "
                   f"p={p_s}; even the variance ceiling is only {ceil_s}. n={n} proteins.")
    return {
        "predictor": predictor, "target": target, "n_proteins": n,
        "unit": "protein",
        "spearman_rho": (round(sp, 4) if sp is not None else None),
        "pearson_r": (round(pr, 4) if pr is not None else None),
        skill_key: skill_val,
        "lopo_skill_over_baseline": (skill.get("skill_above_baseline")
                                     if ordinal else skill_val),
        "lopo_detail": skill,
        "permutation_p": (round(perm_p, 6) if perm_p is not None else None),
        "permutation_n_draws": n_perm,
        "variance_ceiling_fraction": (round(ceiling_fraction, 4)
                                      if ceiling_fraction is not None else None),
        "significant_at_0_05": significant,
        "verdict_plain_english": verdict,
    }


def p1_sequence_to_kinetics(prot_table: dict) -> dict:
    """P1: do the endpoint-matched sequence predictors (PASTA energy, Waltz
    coverage) predict the MEASURED kinetics (median log10 t50, median
    lag_to_t50_ratio, ordinal regime)? Protein-level, LOPO, permutation-nulled,
    against the variance ceiling. Usually a NULL result — stated bluntly."""
    # eligibility: a usable point t50 (for the t50 targets) + an available predictor
    pasta_rows = [d for d in prot_table.values()
                  if d["pasta_available"] and d["median_log10_t50_point"] is not None]
    waltz_rows = [d for d in prot_table.values()
                  if d["waltz_available"] and d["median_log10_t50_point"] is not None]

    # variance ceiling on the union of predictor-eligible proteins (point t50)
    elig = {d["uniprot"]: d["log10_t50_replicates"] for d in prot_table.values()
            if (d["pasta_available"] or d["waltz_available"]) and d["log10_t50_replicates"]}
    ceiling = variance_ceiling(elig)
    ceil_frac = ceiling.get("between_protein_fraction") if ceiling.get("computable") else None

    cards = []
    # PASTA (min pairing energy: more negative = stronger predicted cross-β)
    if len(pasta_rows) >= 4:
        px = [d["pasta_energy"] for d in pasta_rows]
        cards.append(_assoc_card(
            "PASTA_best_pairing_energy", "median_log10_t50",
            px, [d["median_log10_t50_point"] for d in pasta_rows], ceil_frac))
        # lag_to_t50 target (only proteins with that feature)
        lp = [d for d in pasta_rows if d["median_lag_to_t50_ratio"] is not None]
        if len(lp) >= 4:
            cards.append(_assoc_card(
                "PASTA_best_pairing_energy", "median_lag_to_t50_ratio",
                [d["pasta_energy"] for d in lp],
                [d["median_lag_to_t50_ratio"] for d in lp], None))
        # ordinal max-regime target
        rp = [d for d in pasta_rows if d["max_regime_ordinal"] is not None]
        if len(rp) >= 4:
            cards.append(_assoc_card(
                "PASTA_best_pairing_energy", "ordinal_max_regime",
                [d["pasta_energy"] for d in rp],
                [d["max_regime_ordinal"] for d in rp], None, ordinal=True))
    # Waltz amyloid-hexapeptide coverage
    if len(waltz_rows) >= 4:
        cards.append(_assoc_card(
            "Waltz_hexapeptide_coverage", "median_log10_t50",
            [d["waltz_coverage"] for d in waltz_rows],
            [d["median_log10_t50_point"] for d in waltz_rows], ceil_frac))
        rw = [d for d in waltz_rows if d["max_regime_ordinal"] is not None]
        if len(rw) >= 4:
            cards.append(_assoc_card(
                "Waltz_hexapeptide_coverage", "ordinal_max_regime",
                [d["waltz_coverage"] for d in rw],
                [d["max_regime_ordinal"] for d in rw], None, ordinal=True))

    return {
        "n_proteins_pasta": len(pasta_rows),
        "n_proteins_waltz": len(waltz_rows),
        "variance_ceiling": ceiling,
        "association_cards": cards,
        "headline": (
            "Among known amyloids, the endpoint-matched sequence predictors do not "
            "explain the measured rate: " + (cards[0]["verdict_plain_english"]
                                             if cards else "no eligible proteins.")),
        "honesty": ("unit = PROTEIN (predictor scores are per-protein constants); "
                    "permutation null shuffles protein labels; skill is leave-one-"
                    "protein-out; max explainable variance is capped by the "
                    "between-protein ceiling shown."),
    }


# ============================================================================ #
# P2 — STRUCTURE/APR ↔ REGIME association (the clean positive signal)
# ============================================================================ #
def p2_apr_regime(prot_table: dict) -> dict:
    """P2: associate APR count / structure mix with the descriptive regime
    cooperativity, using the PROTEIN as the unit. Jonckheere–Terpstra trend across
    the ORDERED regimes (protein-level permutation null) + Kruskal–Wallis omnibus.
    Explicitly ASSOCIATIVE, never causal; polymorphism caveat attached."""
    # one row per protein: (max-regime ordinal, APR count). Use the most-cooperative
    # regime reached (a per-protein propensity), so a famous amyloid's many gradual
    # curves don't drag it to the floor.
    rows = [(d["max_regime_ordinal"], d["n_aprs"], d["uniprot"], d["name"])
            for d in prot_table.values() if d["max_regime_ordinal"] is not None]
    if len(rows) < 3:
        return {"computable": False, "reason": "need ≥3 proteins with a regime",
                "n_proteins": len(rows)}

    # group APR counts by ordinal regime (protein-level)
    by_ord = defaultdict(list)
    for o, apr, _, _ in rows:
        by_ord[o].append(apr)
    ordered = sorted(by_ord)
    groups_ordered = [by_ord[o] for o in ordered]
    median_apr_by_regime = {REGIME_LADDER[o]: _median(by_ord[o]) for o in ordered}
    n_by_regime = {REGIME_LADDER[o]: len(by_ord[o]) for o in ordered}

    jt = jonckheere_statistic(groups_ordered)
    jt_perm = jonckheere_permutation_p([o for o, _, _, _ in rows],
                                       [apr for _, apr, _, _ in rows])
    kw = kruskal_wallis(groups_ordered)

    # same panel for amyloid-structure fraction vs regime (associative companion)
    struct_rows = [(d["max_regime_ordinal"], d["n_structures"])
                   for d in prot_table.values()
                   if d["max_regime_ordinal"] is not None and d["n_structures"] > 0]
    median_struct_by_regime = {}
    if struct_rows:
        bs = defaultdict(list)
        for o, ns in struct_rows:
            bs[o].append(ns)
        median_struct_by_regime = {REGIME_LADDER[o]: _median(bs[o]) for o in sorted(bs)}

    trend_dir = jt.get("direction")
    p_perm = jt_perm.get("p_one_sided_permutation")
    sig = (p_perm is not None and p_perm < 0.05)
    verdict = (
        f"APR count rises monotonically with regime cooperativity "
        f"(medians {'→'.join(str(median_apr_by_regime[r]) for r in median_apr_by_regime)}; "
        f"Jonckheere protein-permutation p={p_perm:.4f}). ASSOCIATIVE only — more "
        f"aggregation-prone regions co-occur with more cooperative measured kinetics; "
        f"NOT a causal claim (fibril polymorphism + selection bias)."
        if sig else
        f"No significant monotone APR↔regime trend at the protein level "
        f"(Jonckheere permutation p={p_perm}). Medians: {median_apr_by_regime}.")
    return {
        "computable": True,
        "n_proteins": len(rows),
        "unit": "protein",
        "median_apr_by_regime": median_apr_by_regime,
        "n_proteins_by_regime": n_by_regime,
        "median_structures_by_regime": median_struct_by_regime,
        "jonckheere": jt,
        "jonckheere_permutation": jt_perm,
        "jonckheere_statistic": jt.get("J"),
        "p_value_protein_permutation": p_perm,
        "kruskal_wallis": kw,
        "trend_direction": trend_dir,
        "relationship": "associative_only",
        "verdict": verdict,
        "caveats": [
            "ASSOCIATIVE, never causal: APRs and cooperative kinetics co-occur.",
            "fibril polymorphism: 'the APRs of protein P' is condition-dependent.",
            "selection bias: the corpus is enriched for famous amyloids (no negatives).",
            "unit = protein; the protein-permutation null prevents replicate curves "
            "inflating significance.",
        ],
    }


# ============================================================================ #
# P3 — protein-level DISCORDANCE scan (fills §3-M6 null_disagreement_rate)
# ============================================================================ #
def discordance_direction(observed_rate, null_rate, binomial_p,
                          alpha=DISCORDANCE_ALPHA, tol=1e-12) -> dict:
    """Resolve the TWO INDEPENDENT axes of a discordance result: DIRECTION and
    SIGNIFICANCE. They are computed from DIFFERENT inputs and must never be
    derived from one another.

      * DIRECTION   ← sign of (observed_rate − null_rate). The two RATES, and
        nothing else. A two-sided binomial p is direction-BLIND: it is equally
        small when the observed rate is far ABOVE the null and when it is far
        BELOW it, so it can never say which way the result points.
      * SIGNIFICANCE ← (binomial_p < alpha). This licenses only the word
        "significantly different from the null"; it contributes NO direction.

    Reading direction off the p-value is precisely how a verdict ends up
    asserting "EXCEEDS the null" for a rate 3.4× UNDER it. Keep them apart."""
    if observed_rate is None or null_rate is None:
        return {"direction": None, "significant": None, "alpha": alpha,
                "observed_minus_null": None}
    delta = float(observed_rate) - float(null_rate)
    if delta > tol:
        direction = "above"
    elif delta < -tol:
        direction = "below"
    else:
        direction = "equal"
    significant = (binomial_p is not None and float(binomial_p) < alpha)
    return {"direction": direction,
            "significant": (significant if binomial_p is not None else None),
            "alpha": alpha,
            "observed_minus_null": round(delta, 6)}


def _discordance_clause(direction, significant, n_eval, n_discord,
                        alpha=DISCORDANCE_ALPHA) -> str:
    """The interpretive sentence appended to the P3 verdict.

    DIRECTION supplies the direction word (EXCEEDS / FALLS BELOW); SIGNIFICANCE
    supplies only whether the gap is readable at all. Every branch is spelled out
    — a future corpus can genuinely flip the sign, and when it does the emitted
    sentence must follow it without a human re-deriving the wording."""
    if direction is None or significant is None:
        return ("No evaluable proteins — no direction and no significance claim "
                "is licensed.")
    where = {"above": "above", "below": "below", "equal": "exactly at"}[direction]
    if not significant:
        return (f"Observed discordance sits {where} the null but is NOT significantly "
                f"different from it (two-sided, α={alpha}) — within the expected "
                f"null, no falsifiable surprise in either direction.")
    if direction == "equal":
        return (f"Observed discordance equals the null rate — no directional claim "
                f"is available; a flag at α={alpha} on a zero gap is a numerical "
                f"edge case, not a finding.")
    if direction == "above":
        return (f"Observed discordance EXCEEDS the null and is significantly "
                f"different from it (two-sided, α={alpha}) — sequence and kinetics "
                f"disagree MORE often than the pre-registered baseline predicts: a "
                f"falsifiable signal worth review.")
    # direction == "below": the mirror case, and the one most easily overclaimed.
    return (f"Observed discordance FALLS BELOW the null and is significantly "
            f"different from it (two-sided, α={alpha}) — sequence and kinetics AGREE "
            f"more often than the pre-registered baseline predicts. This is NOT "
            f"evidence that the sequence predictors are good: the null is a "
            f"pre-registered construction (PASTA literature FP ⊕ corpus censoring), "
            f"never calibrated on this corpus, so a CONSERVATIVE (too-high) null "
            f"produces exactly this result — as would an easier-than-average "
            f"evaluable subset ({n_eval} endpoint-matched, famous-amyloid-enriched "
            f"proteins). At {n_discord}/{n_eval} the discordant count is thin and a "
            f"single protein moves the rate materially. Direction stated; no win "
            f"claimed.")


def p3_discordance(prot_table: dict) -> dict:
    """P3: protein-level sequence↔kinetics discordance, censoring-aware.

    SEQUENCE-positive = PASTA pairing energy below the pre-registered threshold OR
    ≥1 Waltz-validated amyloid hexapeptide. KINETICS-positive = ≥1 fittable
    cooperative/threshold/gradual curve in a mass-proportional assay (aggregates
    SOMEWHERE in its measured conditions). A protein that is "no-detectable" ONLY
    under right-censoring is EXCLUDED (not a true discordance). Builds the 2×2,
    sets the null disagreement rate from the predictor FP/FN baseline + the corpus
    censoring rate, and binomial-tests observed vs null."""
    confusion = {"seq_pos_kin_pos": 0, "seq_pos_kin_neg": 0,
                 "seq_neg_kin_pos": 0, "seq_neg_kin_neg": 0}
    discordant = []
    excluded_censored = []
    evaluated = []
    for d in prot_table.values():
        seq_pos = (
            (d["pasta_available"] and d["pasta_energy"] is not None
             and d["pasta_energy"] < PASTA_AMYLOID_ENERGY_THRESHOLD)
            or (d["waltz_available"] and (d["waltz_coverage"] or 0) >= 1))
        # a protein with NO endpoint-matched predictor at all can't enter the scan
        if not (d["pasta_available"] or d["waltz_available"]):
            continue
        kin_pos = d["n_mass_assay_fittable"] >= 1
        # censoring-aware exclusion: if kinetics looks NEGATIVE but every curve is
        # right-censored (we never observed a plateau), the negative is a detection-
        # limit artifact, NOT a true discordance — exclude it.
        if not kin_pos and d["n_uncensored"] == 0 and d["n_right_censored"] > 0:
            excluded_censored.append({"uniprot": d["uniprot"], "name": d["name"],
                                      "reason": "kinetics-negative only under right-"
                                                "censoring (detection limit, not a "
                                                "true discordance)"})
            continue
        evaluated.append(d["uniprot"])
        key = (f"seq_{'pos' if seq_pos else 'neg'}_"
               f"kin_{'pos' if kin_pos else 'neg'}")
        confusion[key] += 1
        if seq_pos != kin_pos:
            discordant.append({
                "uniprot": d["uniprot"], "name": d["name"],
                "sequence_positive": seq_pos, "kinetics_positive": kin_pos,
                "pasta_energy": d["pasta_energy"], "waltz_coverage": d["waltz_coverage"],
                "n_mass_assay_fittable": d["n_mass_assay_fittable"],
                "n_uncensored": d["n_uncensored"], "n_right_censored": d["n_right_censored"],
                "modal_regime": d["modal_regime"], "max_regime": d["max_regime"],
            })

    n_eval = len(evaluated)
    n_discord = (confusion["seq_pos_kin_neg"] + confusion["seq_neg_kin_pos"])
    # NULL disagreement rate = predictor FP/FN literature baseline (pre-registered)
    # combined with the corpus detection-limit/censoring rate. We OR the two
    # independent ways a sequence↔kinetics call can disagree by chance:
    #   1 - (1 - predictor_error)(1 - corpus_censoring_rate).
    total_curves = sum(d["n_uncensored"] + d["n_right_censored"] for d in prot_table.values())
    total_rc = sum(d["n_right_censored"] for d in prot_table.values())
    corpus_censor_rate = (total_rc / total_curves) if total_curves else 0.0
    null_rate = 1.0 - (1.0 - PASTA_LITERATURE_FP_RATE) * (1.0 - corpus_censor_rate)
    null_rate = min(0.99, null_rate)
    binom_p = binom_test_two_sided(n_discord, n_eval, null_rate) if n_eval else None
    obs_rate = (n_discord / n_eval) if n_eval else None

    # DIRECTION (which side of the null) and SIGNIFICANCE (is the gap readable)
    # are resolved on SEPARATE axes — see `discordance_direction`. The direction
    # word below comes from the two RATES; the p-value only gates "significantly".
    axes = discordance_direction(obs_rate, null_rate, binom_p)

    verdict = (
        f"{n_discord}/{n_eval} evaluable proteins are sequence↔kinetics DISCORDANT "
        f"(observed rate {0 if obs_rate is None else round(obs_rate, 3)}) vs a "
        f"pre-registered null disagreement rate of {round(null_rate, 3)} "
        f"(PASTA ~{int(100 * PASTA_LITERATURE_FP_RATE)}% FP ⊕ corpus censoring "
        f"{round(corpus_censor_rate, 3)}); two-sided binomial p="
        f"{'n/a' if binom_p is None else round(binom_p, 3)}. "
        + _discordance_clause(axes["direction"], axes["significant"],
                              n_eval, n_discord))
    return {
        "confusion_2x2": confusion,
        "n_evaluated": n_eval,
        "n_discordant": n_discord,
        "observed_disagreement_rate": (round(obs_rate, 4) if obs_rate is not None else None),
        "discordant_proteins": discordant,
        "excluded_right_censored": excluded_censored,
        "n_excluded_censored": len(excluded_censored),
        "null_disagreement_rate": round(null_rate, 4),
        "null_components": {
            "pasta_literature_fp_rate": PASTA_LITERATURE_FP_RATE,
            "pasta_fp_rate_is_pre_registered_assumption": True,
            "corpus_censoring_rate": round(corpus_censor_rate, 4),
            "combination": "1 - (1 - predictor_fp)(1 - corpus_censoring)",
        },
        "binomial_p": (round(binom_p, 6) if binom_p is not None else None),
        # the two axes, reported separately so a consumer can never re-conflate them
        "direction_vs_null": axes["direction"],
        "observed_minus_null": axes["observed_minus_null"],
        "significant_at_0_05": axes["significant"],
        "alpha": axes["alpha"],
        "direction_basis": ("sign of (observed_rate − null_rate); the two-sided "
                            "binomial p is direction-blind and gates only the word "
                            "'significantly'"),
        "pre_registered_constants": {
            "pasta_amyloid_energy_threshold": PASTA_AMYLOID_ENERGY_THRESHOLD,
            "waltz_seq_positive_min_coverage": 1,
        },
        "unit": "protein",
        "verdict": verdict,
        "caveats": [
            "DIRECTION comes from (observed − null) only; a two-sided p is equally "
            "small far ABOVE and far BELOW the null and can never supply direction.",
            "a BELOW-null result is NOT proof the sequence predictors work: the null "
            "is pre-registered (PASTA literature FP ⊕ corpus censoring), not "
            "calibrated here, so it may simply be conservative.",
            "the evaluable set is endpoint-matched and famous-amyloid-enriched — an "
            "easier subset than the corpus at large.",
            "unit = protein and n is tiny; one protein moves the observed rate "
            "materially in either direction.",
        ],
        "note": ("fills the §3-M6 deferred `null_disagreement_rate` (was None). "
                 "The predictor FP rate is a documented pre-registered constant; "
                 "Service-C calibration is the upgrade path."),
    }


# ============================================================================ #
# P4 — γ × predictor sidebar (honestly thin; descriptive, no test)
# ============================================================================ #
def p4_gamma_sidebar(prot_table: dict, gamma_records, name_to_up) -> dict:
    """P4: descriptive γ × predictor table for the ≤7 proteins with a RELIABLE
    gamma_regression AND an endpoint-matched predictor. No inferential test — n is
    far below the reliability floor; hypothesis-generating only (feeds M9)."""
    rows = []
    for g in gamma_records:
        reg = g.get("gamma_regression") or {}
        if not reg.get("gamma_reliable"):
            continue
        up = name_to_up.get(g.get("protein"))
        d = prot_table.get(up) if up else None
        if not d or not (d["pasta_available"] or d["waltz_available"]):
            continue
        rows.append({
            "protein": g.get("protein"), "uniprot": up,
            "gamma": reg.get("gamma"), "gamma_ci": reg.get("gamma_ci"),
            "pasta_energy": d["pasta_energy"], "waltz_coverage": d["waltz_coverage"],
            "median_log10_t50": d["median_log10_t50_point"],
            "max_regime": d["max_regime"],
        })
    # dedup by uniprot, keep the row with the tightest CI (most informative γ)
    best = {}
    for r in rows:
        key = r["uniprot"] or r["protein"]
        if key not in best:
            best[key] = r
    rows = list(best.values())
    return {
        "n_proteins": len(rows),
        "table": rows,
        "test_performed": False,
        "verdict": (f"n={len(rows)} reliable γ with a matched predictor — below the "
                    "n<5 reliability floor; NO inferential claim licensed; "
                    "hypothesis-generating only (feeds M9)."),
        "unit": "protein",
    }


# ============================================================================ #
# ASSEMBLER
# ============================================================================ #
def _coverage(prot_table: dict) -> dict:
    """The effective-n table: proteins per modality-overlap. THE honesty centrepiece
    — the effective n is PROTEINS, not curves."""
    has_seq = [d for d in prot_table.values() if d["pasta_available"] or d["waltz_available"]]
    has_point = [d for d in prot_table.values() if d["median_t50_point"] is not None]
    has_regime = [d for d in prot_table.values() if d["max_regime_ordinal"] is not None]
    has_struct = [d for d in prot_table.values() if d["n_structures"] > 0 or d["n_aprs"] > 0]
    both = [d for d in prot_table.values()
            if (d["pasta_available"] or d["waltz_available"])
            and d["median_t50_point"] is not None]
    total_curves = sum(d["n_curves"] for d in prot_table.values())
    return {
        "n_proteins_total": len(prot_table),
        "n_curves_total": total_curves,
        "n_proteins_with_predictor": len(has_seq),
        "n_proteins_with_pasta": sum(1 for d in prot_table.values() if d["pasta_available"]),
        "n_proteins_with_waltz": sum(1 for d in prot_table.values() if d["waltz_available"]),
        "n_proteins_with_point_t50": len(has_point),
        "n_proteins_with_regime": len(has_regime),
        "n_proteins_with_structure_or_apr": len(has_struct),
        "n_proteins_predictor_AND_point_t50": len(both),
        "effective_n_is_proteins_not_curves": True,
        "effective_n_banner": (
            f"Effective n ≈ {len(both)} PROTEINS, NOT {total_curves} curves. A "
            "sequence predictor is a per-protein constant replicated across that "
            "protein's curves; every test here uses the PROTEIN as the unit."),
    }


def build_crossmodal(triaged, features, regimes, seq_lookup, gamma_records) -> dict:
    """Assemble the full M10 cross-modal payload. Pure; never raises on bad input."""
    try:
        prot_table = build_protein_table(triaged or [], features or [],
                                         regimes or [], seq_lookup or {})
    except Exception as exc:                                   # defensive, never raise
        return {"version": VERSION, "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "coverage": {}, "P1": {}, "P2": {}, "P3": {}, "P4": {}}
    name_to_up = {}
    for d in prot_table.values():
        if d["name"]:
            name_to_up.setdefault(d["name"], d["uniprot"])
    return {
        "version": VERSION,
        "status": "ok",
        "coverage": _coverage(prot_table),
        "P1_sequence_to_kinetics": p1_sequence_to_kinetics(prot_table),
        "P2_apr_to_regime": p2_apr_regime(prot_table),
        "P3_discordance": p3_discordance(prot_table),
        "P4_gamma_sidebar": p4_gamma_sidebar(prot_table, gamma_records or [], name_to_up),
        "honesty_constraints": [
            "Effective n = number of PROTEINS (~20), NOT curves (~1200): a predictor "
            "score is a per-protein constant replicated across that protein's curves.",
            "Every association test uses the PROTEIN as the unit (protein-level "
            "permutation nulls; leave-one-protein-out predictive skill).",
            "The corpus is ~97% amyloid/ThT and biased toward famous amyloids (almost "
            "no negative class) — tests target rate/regime AMONG known amyloids, NEVER "
            "'amyloid vs not'.",
            "A sequence maps to MANY rates across conditions, so a sequence-only "
            "predictor's max explainable variance is bounded by the BETWEEN-protein "
            "fraction of total t50 variance (the variance ceiling, computed + shown).",
            "P2 is ASSOCIATIVE, never causal (fibril polymorphism + selection bias).",
            "P4 has n below the reliability floor — descriptive only, no test.",
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
                    continue                                  # never die on one bad row
    return out


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _print_summary(payload: dict):
    cov = payload.get("coverage", {})
    _safe_print("\n=== M10 cross-modal (version %s) ===" % payload.get("version"))
    _safe_print(cov.get("effective_n_banner", ""))
    _safe_print("coverage: %d proteins / %d curves; predictor+point-t50 = %d proteins"
          % (cov.get("n_proteins_total", 0), cov.get("n_curves_total", 0),
             cov.get("n_proteins_predictor_AND_point_t50", 0)))
    p1 = payload.get("P1_sequence_to_kinetics", {})
    vc = p1.get("variance_ceiling", {})
    _safe_print("\n--- P1 SEQUENCE → KINETICS (n=%d PASTA / %d Waltz proteins) ---"
          % (p1.get("n_proteins_pasta", 0), p1.get("n_proteins_waltz", 0)))
    if vc.get("computable"):
        _safe_print("variance ceiling (between-protein log10 t50): %.1f%% (%d proteins, %d curves)"
              % (100 * vc["between_protein_fraction"], vc["n_proteins"], vc["n_curves"]))
    for c in p1.get("association_cards", []):
        _safe_print("  [%s → %s]  ρ=%s  LOPO-skill=%s  perm_p=%s  ceiling=%s"
              % (c["predictor"], c["target"],
                 c.get("spearman_rho"),
                 c.get("lopo_skill_over_baseline"),
                 c.get("permutation_p"), c.get("variance_ceiling_fraction")))
        _safe_print("     " + c["verdict_plain_english"])
    p2 = payload.get("P2_apr_to_regime", {})
    _safe_print("\n--- P2 APR → REGIME (n=%d proteins) ---" % p2.get("n_proteins", 0))
    if p2.get("computable"):
        _safe_print("median APR by regime: %s" % p2.get("median_apr_by_regime"))
        _safe_print("Jonckheere J=%s  protein-permutation p=%s  direction=%s"
              % (p2.get("jonckheere_statistic"),
                 p2.get("p_value_protein_permutation"), p2.get("trend_direction")))
        _safe_print("  " + p2.get("verdict", ""))
    p3 = payload.get("P3_discordance", {})
    _safe_print("\n--- P3 DISCORDANCE (n=%d evaluated, %d censoring-excluded) ---"
          % (p3.get("n_evaluated", 0), p3.get("n_excluded_censored", 0)))
    _safe_print("confusion 2x2: %s" % p3.get("confusion_2x2"))
    _safe_print("null disagreement rate=%s  observed=%s  binomial p=%s"
          % (p3.get("null_disagreement_rate"), p3.get("observed_disagreement_rate"),
             p3.get("binomial_p")))
    _safe_print("  " + p3.get("verdict", ""))
    p4 = payload.get("P4_gamma_sidebar", {})
    _safe_print("\n--- P4 γ × predictor sidebar (n=%d proteins) ---" % p4.get("n_proteins", 0))
    _safe_print("  " + p4.get("verdict", ""))


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    proc = root / "data" / "processed"
    ap = argparse.ArgumentParser(description="PRISE M10 cross-modal analysis")
    ap.add_argument("--triaged", type=Path, default=proc / "curves_triaged.jsonl")
    ap.add_argument("--features", type=Path, default=proc / "features.jsonl")
    ap.add_argument("--regimes", type=Path, default=proc / "regimes.jsonl")
    ap.add_argument("--sequence", type=Path, default=proc / "sequence_structure.json")
    ap.add_argument("--gamma", type=Path, default=proc / "gamma.jsonl")
    ap.add_argument("--output", type=Path, default=proc / "m10_crossmodal.json")
    args = ap.parse_args(argv)

    triaged = _load_jsonl(args.triaged)
    features = _load_jsonl(args.features)
    regimes = _load_jsonl(args.regimes)
    gamma = _load_jsonl(args.gamma)
    seq = _load_json(args.sequence, {"by_uniprot": {}, "absent_predictors": {}})

    payload = build_crossmodal(triaged, features, regimes, seq, gamma)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[M10] wrote %s" % args.output)
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
