# PRISE engine — analytical core

Modules that consume the canonical `AggregationSeries` produced by the ETL
(`../etl/`) and run the analysis pipeline. Built incrementally per
`../PRISE_DESIGN.md` §3.

Every corpus figure below is from the **calibrated, assay-corrected** build:
`engine_build_id` **`d6ffd19f850c`**, `corpus_version` `c09ac0cf398e`, over
**1,654 curves / 105 proteins**. Results from a different `engine_build_id` are
**not** comparable to these and the engine blocks such comparisons (§7). The
preceding build `e660cb452b41` (pre-`join-policy-1.0`) is retained under
`data/processed/_builds/` so the superseded numbers stay auditable.

## Implemented

### M1 — ingestion, normalization, artifact gate, censoring & QC
`m1_ingest.py` — faithful triage of each curve (no fitting/inference here):

- **artifact gate** → `fittability_class` ∈ {fittable, suspect, unfittable} + reasons
- **censoring** → `censoring_class` ∈ {none, left, right, left_right, interval}
  - `right` = still rising at the end (no plateau) → blocks min–max normalisation
  - `left` = no lag phase at t0 (possible left-censoring *or* downhill kinetics;
    M5 disambiguates) — flagged, not asserted
- **normalization discipline** → baseline subtraction always; min–max **only when a
  plateau is validated** (never plateau-manufacturing on right-censored curves);
  records `signal_basis` + `normalization_mode`
- **recommended_handling** ∈ {monotonic_fit, nonmonotonic_fit, descriptive_only, reject}
- **plate QC** = not applicable for digitized literature curves (no co-located controls)

All heuristics are **scale-robust** (CPAD intensities are digitized a.u. with
unknown normalisation): decisions use each curve's own dynamic range and
point-to-point noise, never absolute thresholds.

Run:
```
python engine/m1_ingest.py            # reads data/processed/curves.jsonl
                                      # writes data/processed/curves_triaged.jsonl
```

Triage distribution on the real corpus (1,654 curves), **Service-C-calibrated
(thresholds-1.0)**:
- fittable 1,240 · suspect 408 · unfittable 6  *(unchanged — the lag-slope gate
  only affects censoring, not fittability)*
- monotonic_fit 1,277 · nonmonotonic_fit 323 · descriptive_only 48 · reject 6
  *(unchanged)*
- censoring: none **736** · right **168** · left **607** · left_right **130** ·
  interval **13**

> **Thresholds are now calibrated, not first-pass.** As of the governed apply step
> (`engine/calibrated_thresholds.py`, `thresholds-1.0`, applied 2026-07-01), M1's
> `LAG_SLOPE_FRAC` runs Service C's **held-out** recommendation **0.45** (out-of-
> sample separation AUC 1.0), replacing the first-pass **0.30** (retained as
> `first_pass`, reversible). The higher bar corrects the first-pass over-flagging
> of left/no-lag: the **classification SHIFT** vs the first-pass triage is
> `none 634→736 (+102)`, `left 711→607 (−104)`, `right 156→168 (+12)`,
> `left_right 142→130 (−12)`, `interval 11→13 (+2)`. This is strictly more
> rigorous: the cutoff now has a measured out-of-sample guarantee, not a guess.
> The change bumps `engine_build_id` (old results non-comparable, §7).

### M2 — model fitting bank
Models live in `models.py` (closed-form registry) and `mechanistic.py` (the
Knowles/Cohen ODE family). `m2_fit.py` drives the fitting (bounded multi-start
scipy TRF, per-fit SSE/R²/AIC/AICc/BIC + parameter SEs, graceful
non-convergence). M2 only fits + reports; selection is M3, mechanism is M5.

Three axes / entry points:
- **Kinetic (signal vs time)** — `fit_curve(series)` — `lnt` (linear no-threshold),
  `logistic`, `gompertz`, `richards` (generalised-logistic sigmoid), `exponential`,
  `scaling_law` (power), `brain_cousens` (biphasic/hormesis), `finke_watzky`
  (Tier-B closed form). Candidates routed by M1 `recommended_handling` (a monotonic
  sigmoid is never forced onto a non-monotonic curve).
- **Dose–response (response vs concentration)** — `fit_dose_response(d, r)` —
  `dr_lnt`, `dr_threshold` (hockey-stick), `dr_hill` (4PL), `dr_brain_cousens`,
  `dr_scaling`. Fits the concentration-series / R-row (k_agg-vs-concentration) data.
- **Tier B mechanistic (signal vs time)** — `mechanistic.fit_mechanistic` — one ODE
  framework (P, M moment equations) covering `nucleation_elongation`,
  `secondary_nucleation`, `saturating_secondary`, `fragmentation`. Exact numerical
  integration (LSODA), fit in log-rate space. These models are *sloppy* (individual
  rates not uniquely identifiable from one curve) — M3/M4 (global concentration
  fits + γ) is where reaction orders are actually constrained.

> Scientific note: Brain–Cousens and LNT are dose–response models; their native
> home is the DOSE bank (and they are also exposed on the kinetic axis as a
> hormesis bump / a line).

Run:
```
python engine/m2_fit.py                 # kinetic: curves_triaged.jsonl -> fits.jsonl
python engine/m2_fit.py --limit 50      # quick subset
python engine/m2_fit.py --dose          # dose-response on R-rows by protein
```

### fit provenance — §2.3 `FitProvenance` and the §7 basin-stability gate
`fit_provenance.py` builds the record §2.3 specifies and §10.12 requires, which
nothing emitted until now. Two halves: a **frozen recipe** (optimizer, tolerances,
RNG applicability, version, `env_manifest_ref`) written once to
`data/processed/fit_recipes.json`, and a per-fit **basin ledger** referencing it by
id. Emitted by `m2_fit`, `global_fit`, `m3_select` and `m4_features`. **Measured
cost: `fits.jsonl` 4.78 → 6.71 MB (+40%, ~200 B per model-fit).** A first pass
inlining the explanatory prose grew it **76%** — fixed by a `reason_code` plus a
resolvable table (the prose alone was 860 KB, 10.2% of the file) and by omitting
`winning_basin`/`basin_spread` in the single-basin case where they are derivable.
Estimated +26%, measured +40%: the estimate was wrong, which is why it was
measured.

**Honesty, in both directions.** M2/`mechanistic`/`global_fit` draw **no random
number** — their starts are built deterministically from the data — so the record
says `rng: not applicable` **with a reason** rather than inventing `rng_seed: 0`.
(A fabricated provenance record is a lie that *passes*; `test_m2_never_fabricates_
an_rng_seed` is the control that prevents it.) `global_fit`'s `seed` argument was
**decorative** — `_fit_mechanism` never referenced it and the `np.random.seed()`
call seeded a RandomState `least_squares` does not consult — so it is now recorded
as `effective: false`. M3's and M4's bootstraps **do** draw and carry a real seed,
algorithm and `stream_policy`. M3's seed is now **derived per series**
(`estimation-policy-1.0`): it previously restarted from a constant 0, so the
**98.4%** of curves sharing a length with another received byte-identical weight
sequences. Verified in the artifact: 1,240 records, 1,240 distinct seeds.

**The §7 gate, now evaluable — and now actually evaluated.** Each fit gets
`stable` / `knife_edge` / `not_evaluable` / `no_fit`. Basins are clustered by
**parameter-vector** distance, not SSE — the whole point, since ~all same-basin SSE
gaps are ~1e-11 (two starts reaching the *same* optimum), whereas a knife-edge is
near-equal SSE at a *materially different* vector. Measured from the shipped
artifact (1,654 curves / 9,633 model-fits): **stable 9,085 (94.31%) · knife_edge
373 (3.87%) · no_fit 173 (1.80%) · not_evaluable 2 (0.02%)**. Among the 9,458 fits
that can be assessed the knife-edge rate is **3.9%**. Of the `stable` verdicts
5,739 are "all starts agree" and 3,346 "rivals decisively worse" — i.e. on 3,346
fits the multi-start genuinely changed which optimum was reported, which was
previously invisible. `no_fit` splits 162 `too_few_points` + 11
`no_start_converged`.

> **This census is the point of `estimation-policy-1.0`, and it replaces a much
> worse one.** Before it, `fit_one` built its extra starts by testing *parameter
> names* (`t0`, `k`), so `logistic`/`gompertz`/`richards` got 4 starts,
> `exponential` 2, and the other **9 of 13** registry models — including
> `finke_watzky` (rates named k1/k2) and every dose-response model — got exactly
> **one** and had nothing to be stable between. The census then read **stable
> 48.9% · not_evaluable 46.6% · knife_edge 2.70%**: nearly half the corpus's fits
> could not be assessed *at all*, so §7's gate ("anchors freezable only if
> basin-stable") was inapplicable to them and the engine was publishing fitted
> parameters whose basin status was unknown. `not_evaluable` is now **2 fits**, and
> both are honest — `brain_cousens` cases where only one start converged, reported
> `single_start_converged` and never upgraded to `stable`. §7 stays conservative:
> being unable to check is not permission. Note the knife-edge rate ROSE (2.70% →
> 3.87%): more starts find rival basins that were previously invisible, so the gate
> is refusing more fits than before, which is the gate working rather than failing.

### M3 — comparison, selection & identifiability (bootstrap-enveloped)
`m3_select.py` — the statistical spine. **Model selection runs inside the
bootstrap** (post-selection inference): each resample re-fits the candidates and
re-selects, so we report honest *selection-stability frequencies* rather than
pretending the winner was known. The resampler is a **BLOCKWISE wild bootstrap**
(Mammen weights held constant across `ceil(n**(1/3))` consecutive residuals) and
each curve draws from its **own** seed derived from its series id — the pointwise
form assumed serially independent residuals that half this corpus does not have,
and a shared constant seed made the bootstrap ensemble correlated across curves.
Both are `estimation-policy` corrections; see the DONE entries below for the
measurements.

- `analyze_curve(series)` →
  - `akaike_weights` + `point_aicc`; `best_descriptive` / `best_mechanistic`
    reported separately (a sloppy mechanistic fit can't masquerade as a clean win)
  - `selection.selection_frequencies` + `selection_stability` (how often each
    model wins across resamples)
  - `selection.t50_predictive_interval` — a *cross-model* functional (t50), pooled
    only over a model-transferable quantity, with a `t50_multimodal` flag when
    selection is unstable
  - `selection.per_model_conditional_CIs` — model-specific params reported only for
    the resamples that selected that model (never pooled into one number)
  - `identifiability_of_best` — condition number of the column-normalised
    sensitivity matrix (collinearity), the sloppiest parameter direction, and a
    practical-(non-)identifiability flag. **The non-identifiability cutoff
    `COND_NUMBER_NONIDENT` is now Service-C-calibrated (thresholds-1.0): 1000 →
    107** (held-out separation AUC 1.0) — the first-pass round guess of 10³ is
    superseded by the measured ~10² boundary; more sloppy/collinear fits are now
    correctly flagged practically non-identifiable. Previous value retained in
    `engine/calibrated_thresholds.py`; the change bumps `engine_build_id` (§7).
  - `ci_reliable` — false when too few points for a trustworthy bootstrap

Run (expensive — bootstrap × refit; a build-time / async job):
```
python engine/m3_select.py --limit 0 --bootstrap 150    # rebuilds the SHIPPED m3_sample.jsonl
```
> **Corrected 2026-08-06.** This line previously read `--limit 50`, which does
> **not** reproduce the shipped artifact — `m3_sample.jsonl` carries all **1,240**
> fittable curves, not 50. `--bootstrap 150` *is* correct: rebuilding at that value
> reproduces every stored value byte-identically (`artifact_invariance.py`, 0
> violations). The B is now also recorded **in the artifact** as
> `selection.bootstrap_provenance.requested_B`, so this command is no longer the
> only record of how the file was built.
> Deferred refinements (documented): block/sieve bootstrap for serial
> correlation (wild handles only heteroscedasticity), BCa intervals, a
> cross-validated descriptive-vs-mechanistic criterion, and corpus-scale FDR
> control. (M3's `identifiability_of_best` is a **sensitivity-collinearity
> condition number** — the column-normalised Jacobian's correlation matrix, NOT
> the Fisher Information Matrix; the true observed FIM for the ODE mechanistic
> family now lives in `global_fit.py`.)

### M4 — feature extractor, definition contract & dual-γ
`m4_features.py` — turns fitted/selected models into the interpretable quantities
the rest of the pipeline reasons about, under an explicit **versioned definition
contract** (`m4-defs-1.0`), because cross-protein comparison is only valid between
features computed the *same* way (a Gompertz t50 and a tangent-intercept lag from
two different definitions are not comparable).

- **Single-curve feature vector** (`extract_features`) — read off the *selected
  model* on a dense grid (definition-homogeneous across logistic/Gompertz/
  Richards/…): `t50` (primary), `lag_time` (tangent-intercept to baseline),
  **`lag_to_t50_ratio`** (dimensionless shape invariant — a rare single-curve
  mechanistic signal), `inflection_time`, `max_rate` (+ normalized), `t10`/`t90`,
  `transition_width`/`transition_sharpness`, `plateau`, `dynamic_range`.
  - **Censoring-aware status**, not silent numbers: right-censored → `t50` is a
    **lower bound**; left-censored → `t50` biased-early and **lag undefined**.
  - **Non-cooperative guard**: a line / decelerating saturation / power-law has no
    interior inflection → lag & inflection are returned `None` with a flag (never
    an arbitrary `np.gradient` argmax masquerading as an onset).
  - Optional per-feature **wild-bootstrap CIs** (conditional on the selected model;
    cross-model t50 uncertainty stays in M3).
- **Dual-γ** (`dual_gamma`) — the headline mechanistic constraint for a
  `concentration_series`. γ = half-time scaling exponent (t50 ∝ m^−γ). **Two
  estimators, separately provenanced, never merged:**
  - `gamma_regression` — **censored** log–log regression of t50 on concentration
    (right-censored curves contribute an inequality `P(Y ≥ bound)`, not a point;
    no-censoring fast path = OLS). Ships with a **curvature test** (concentration-
    dependent reaction order → saturating secondary nucleation / approach to
    m_crit / mechanism change → route to global fit), an **informative-censoring
    test** (blocks γ if censoring correlates with concentration, §S3), an **m_crit
    identifiability diagnostic** (nominal-m by default; supersaturation correction
    applied *only when m_crit is constrained* — otherwise a flagged downgrade, not
    silent, §K4), and a **reliability gate** (≥3 uncensored anchors, physical |γ|,
    finite CI) — a diverged fit is reported `gamma_reliable: false`, never faked.
    A **NEGATIVE-γ driving-force gate**: γ<0 (t50 rising with concentration) is
    non-physical → `gamma_physical: false` + `gamma_raw` kept for transparency +
    `gamma_reliable: false` (wrong driving force / informative censoring / m_crit
    unconstrained), never emitted as a clean estimate.
  - `gamma_global` — γ from a **phenomenological master-curve COLLAPSE** coupling
    every member curve through one shared logistic **shape** + one γ in rescaled
    time τ = t/t50(m); uses the whole curve shape, tolerates per-curve censoring,
    and is *different machinery* from the t50 regression. **It is NOT a shared-rate
    fit** — it shares only a shape, never microscopic Knowles/Cohen rate constants
    (`method="master_curve_collapse"`, `is_shared_rate_ode_fit=false`). **The real
    shared-rate Knowles/Cohen ODE global fit lives in `engine/global_fit.py`** (see
    below); when run, its third γ_mechanistic + identifiability is joined into the
    dual-γ payload as `global_ode_fit`.
  - **Operational disagreement test** (fixed so loud disagreements are no longer
    dropped): ALWAYS reports the raw pointwise `|γ_reg − γ_collapse|` + a
    `disagree_pointwise` flag beyond a pre-registered margin (0.3) **even when a CI
    is missing or the regression γ is unreliable**; adds a collapse-quality gate
    (collapse_r2 < 0.8 ⇒ `shape_not_concentration_invariant`); keeps the CI-overlap
    test when CIs exist. γ stays a **constraint, not a discriminator** (mechanism
    call is M5, shape-gated, γ a consistency check only).

### global_fit — shared-rate Knowles/Cohen ODE global fit (the real one)
`global_fit.py` (`global-fit-1.0`) — the mechanistic centrepiece §3-M2/§3-M4
promised: for a concentration series it fits **all** member curves
**simultaneously** with **shared microscopic parameters** {log10 k_n, k_+, k_2,
**n_c, n_2** (reaction orders ESTIMATED, not hard-fixed at 2.0), K_M for
saturating}, only the monomer concentration varying per curve (+ a free per-curve
base/amp). Residuals from every curve stack into one vector, each integrated via
`mechanistic.simulate_mass_fraction`; fit with `least_squares` (TRF, bounds, log-
rate space), seeded multi-start (capped), per mechanism family {nucleation_
elongation, secondary_nucleation, saturating_secondary, fragmentation}, AICc-
ranked. Outputs: shared params (bounds-aware), global R²/AICc, an **identifiability
report** (observed FIM eigen-decomposition → identifiable COMBINATIONS + which
individual rates are sloppy/unconstrained), an analytic **third γ_mechanistic**
from the fitted (n_c, n_2) (Meisl/Knowles: γ=(n_2+1)/2 secondary, (n_c+1)/2
primary, ½ fragmentation), and the mechanism-AICc ranking. **Honesty:** these fits
are SLOPPY on this corpus — only stiff rate *combinations* are identifiable, not
unique rates; the value is the reaction-order CONSTRAINT + identifiable combos. It
does **not** license a single mechanism (M5 still refuses — agitation/seeding
unknown). The stiff ODE × many curves is slow, so the CLI supports `--limit`.

> **That sloppiness is now a MEASURED VERDICT, not a claim in prose.** The fit
> emits §2.3 fit provenance (see "fit provenance" above), so the multi-start
> basin record §7 needs survives instead of being discarded. On a controlled
> 4-concentration series the shared-rate fit returns **`basin: knife_edge`** —
> 2 distinct basins, inter-basin SSE gap **2.7e-6** (far inside the 1e-3
> tolerance) at a parameter distance of **0.60**. In plain terms: two materially
> different microscopic-rate vectors fit the same data indistinguishably well,
> so which one "wins" is an accident of start ordering. That is precisely the
> sloppiness this module has always asserted — now expressed as a verdict §7 can
> act on ("anchors are freezable only if basin-stable; knife-edge fits are
> refused") rather than a sentence a reader has to take on trust. It is the
> clearest single illustration of what the FitProvenance change bought. The
> module also no longer advertises a **decorative seed**: `_fit_mechanism` never
> consumed the `seed` argument and the old `np.random.seed()` call seeded a
> generator `least_squares` does not consult, so the record now says
> `seed_argument.effective: false` and `rng.applicable: false` with a reason.
```
python engine/global_fit.py --limit 6          # real-corpus demo -> global_fit.json
python engine/global_fit.py --series CPAD-CS-...  # one series
```
Real corpus (first eligible series): e.g. ABeta40 CS-38932798 → best family
nucleation_elongation, n_c≈2.96, γ_mech≈1.98, R²≈0.99, **SLOPPY** (1 stiff / 2
sloppy directions; identifiable combo ≈ −0.75·n_c − 0.47·log k_n − 0.47·log k_+);
ABeta42 CS-59873462 → fragmentation, γ_mech=0.50, R²≈0.90, sloppy. The third
γ_mechanistic sits alongside the two phenomenological γ's, never merged.

Run:
```
python engine/m4_features.py                       # per-curve features -> features.jsonl
python engine/m4_features.py --limit 200 --bootstrap 150
python engine/m4_features.py --gamma --bootstrap 300   # dual-γ per series -> gamma.jsonl (batch job)
```
> **Corrected 2026-08-06 — this one had teeth.** The γ line previously read
> `--bootstrap 150`. Rebuilding at 150 produced **39 changed values** in
> `gamma.jsonl` (`gamma_ci` bounds and `n_bootstrap_rejected_divergent`); rebuilding
> at **300** — the `gamma_regression(B=300)` / `dual_gamma(B_reg=300)` function
> default — reproduces the shipped artifact with **0 violations**. So the shipped
> file was built at the default and the documented command never reproduced it.
> This mattered because `gamma_ci` feeds `m7_assemble.gamma_significant()`, which
> gates the `scaling` tier of the information-yield ladder — an unrecorded batch
> size sat upstream of a published count. (The significance verdict happened not to
> flip: 5 series before and after. That was luck, not design.) B, seed and recipe
> are now recorded in `gamma.jsonl` as `bootstrap_provenance`, so the artifact
> states how it was built instead of relying on this line.
>
> The middle line (`--limit 200 --bootstrap 150`, for `features.jsonl`) is **not
> verified** — `features.jsonl` has 1,240 records, so `--limit 200` looks wrong
> too, but it is outside the snapshot set and was not tested. Flagged rather than
> silently "corrected".
Real corpus — **per-curve features** (1,240 fittable): 1,217 full vectors, 23
flat. Censoring-honest `t50_status`: point 500 · biased_early 448 · lower_bound
260 · irregular 9. **690/1,217 curves are genuinely cooperative** (interior
inflection); across the 532 with a defined tangent-intercept lag the
**lag-to-t50 ratio clusters at median 0.700 (IQR 0.538–0.802)** — the textbook
signature of nucleation-dependent amyloid kinetics, surfaced as a corpus finding.

Real corpus — **dual-γ** over the 52 concentration series (build-time batch,
≈7 min): **21 return a γ_regression**, of which **8 pass the reliability gate**
once heavy left/right censoring and the ≥3-uncensored-anchor rule are honoured
(the rest are flagged, not faked); **9** are gated non-physical by the negative-γ
driving-force rule, the curvature test is significant on **2**, the disagreement
test fires on **13** series and **1** is blocked for informative censoring. The
reliability gate is doing real work here — most series in this corpus cannot
support a trustworthy exponent, and the module says so rather than emitting one.
Clean agreeing examples: Aβ42 γ_reg ≈ 1.07 (CI
0.94–1.14) vs γ_global 1.10; α-synuclein γ_reg ≈ 0.15 (CI 0.13–0.18) vs 0.18;
Ure2 0.31 = 0.31. Several immunoglobulin light-chain series come back γ ≈ 0 (no
strong concentration scaling) — itself a finding, not a failure.

> γ-batch runtime note: like M3 this is an async/build-time job (the global fit on
> a 60+-curve series is a 60+-parameter least-squares; bootstrap budget auto-scales
> down for large series). The full Knowles/Cohen **ODE** global fit is **no longer
> deferred** — it ships as `engine/global_fit.py` (above) and joins the dual-γ
> payload as `global_ode_fit`; the master-curve collapse remains the always-on v1
> global estimator because it is cheap and censoring-tolerant. Still deferred:
> separate **left-censoring algebra** (left-only t50 currently excluded from γ, not
> modelled), and an explicit errors-in-variables term for concentration uncertainty.

### M5 — classification & regime registry (degeneracy- & shape-aware)
`m5_classify.py` — consumes M1 triage + M2 fits + M4 features and emits three
things against a declarative, hierarchical, **versioned** `REGIME_REGISTRY`
(`m5-regimes-1.0`):

- **Descriptive regime** (always available, pure shape, never erased by anomaly):
  `cooperative_sigmoidal` / `gradual_non_cooperative` / `threshold_driven` /
  `non_monotonic_settling` / `no_detectable_aggregation` (assay-sensitivity-
  qualified) / `anomalous_unclassified` (reserved for genuinely unfittable curves).
- **Mechanistic verdict → equivalence class** — the honest part: M5 **never names a
  single mechanism**. It returns the equivalence class + every gate it failed:
  assay-licensed (mass-proportional ThT/ThS), **shape-gated** (γ a *consistency
  check only*, never sole evidence — K1), **hard-branched on agitation & seeding**
  (quiescent vs shaken), and **data-regime-conditional degeneracy** (single curve →
  broad; reliable γ → reduced; significant curvature → re-broadened by saturating-
  secondary / m_crit). The specific confusion-matrix class + the FDR single-
  mechanism threshold are **Service C deliverables (deferred, flagged)**.
- **Model-agnostic anomaly detector** — Wald–Wolfowitz runs test on best-fit
  residuals → **corpus Benjamini–Hochberg FDR**; a registry **proposal** additionally
  requires *material misfit* (R²<0.9) so structure-but-near-perfect curves don't flood
  the queue. Proposals are human-governed (`PROPOSED_PENDING_HUMAN_APPROVAL`,
  signed/dated/version-bound) and **never auto-mutate** the registry (§7).
- Plus a **structure-linkage** associative panel (CPAD/PDB cross-refs; polymorphism
  caveat; never a causal claim).

Run:
```
python engine/m5_classify.py                 # per-curve regimes -> regimes.jsonl
python engine/m5_classify.py --limit 150
python engine/m5_classify.py --series --bootstrap 80   # per-series mechanistic class
```
Real corpus (1,240 fittable curves): **cooperative_sigmoidal 679 ·
gradual_non_cooperative 502 · threshold_driven 11 · no_detectable 48** (0
anomalous — every fittable curve gets a real shape). Internal-consistency check:
679 + 11 = **690**, exactly M4's interior-inflection count — and it held across the
Service-C threshold recalibration, which re-partitioned curves *between*
`cooperative_sigmoidal` and `threshold_driven` (611+79 before, 679+11 now) without
changing the inflection-bearing set. That invariance is real evidence the
recalibration did what it claimed and nothing more. **Mechanistic inference
licensed on 0 curves** — correct, because agitation/seeding are `unknown` corpus-
wide (the quiescent-vs-shaken hard branch is undetermined). Anomaly scan: 498/1,097
show FDR-significant residual *structure*, but only **19** are *also* materially
misfit (R²<0.9) → **19 actionable proposals** (the effect-size gate strips ~480
mild-misfit/digitization-autocorrelation flags). Both counts are reported; the
runs-test threshold stays first-pass, pending Service C calibration.

> **Regime-shape thresholds are now Service-C-calibrated (thresholds-1.0).**
> `THRESHOLD_SHARPNESS` 3.0 → **5.7** (held-out separation AUC 1.0) and
> `THRESHOLD_LAG_RATIO` 0.80 → **0.97** now source their value from the governed
> `engine/calibrated_thresholds.py`; `ANOMALY_FDR_ALPHA` stays **0.05** (Service C
> confirmed FDR-null control holds). A higher sharpness/lag-ratio bar shifts the
> threshold-vs-cooperative call (fewer curves called `threshold_driven`, more
> remain `cooperative_sigmoidal`). First-pass values retained + reversible; the
> change bumps `engine_build_id` (§7).
>
> Deferred (documented): Service-C confusion-matrix equivalence classes + the FDR
> single-mechanism degeneracy threshold; anomaly-threshold calibration (the
> runs-test p-threshold itself is still first-pass).

### M6 — propensity scorer & sequence axis
`m6_propensity.py` — assembles a **propensity** statement (three views, never
collapsed into one number) plus a de-conflated **sequence axis** and an associative
**structure-linkage** panel (consumes M4 features + dual-γ, M5 regimes, and the
CPAD sequence layer via `etl/cpad_sequence_etl.py`):

- **surface** — feature-vs-driving-force + γ_global/γ_regression (PRIMARY where a
  concentration series exists; from M4 dual-γ; passes through `window_of_validity`).
- **intrinsic** — a position on a *reference-anchored* scale: a **leakage-free
  corpus** anchor (query excluded from its own assay-matched stratum) where effective
  n ≥ 8, else a **versioned reference scale** (`m6-refscale-1.0`, with `clamped` /
  `out_of_range` flags). The **95% stratum reference interval** uses proper quantile
  interpolation and is emitted only when n ≥ 20 (else null + flag). `anchor_kind`
  records which; valid only at its condition vector.
- **cohort** — **Hazen mid-rank** percentile (rank and percentile reconcile by
  construction, ties split evenly), **self-excluded N**, **suppressed for singletons**.
  Because low t50 = high propensity, every record carries an explicit `direction` and
  a sign-corrected **`propensity_percentile`** so a consumer can't sort backwards.
- **Censoring honoured**: only **point** t50 enters the ranked pools; right-censored
  (`lower_bound`) proteins are emitted but intrinsic/cohort **suppressed** (a ≥ value
  is not rankable as a point).
- **Portability + condition discipline**: only time-domain features scored
  (amplitude/a.u. barred); the protein key includes **construct/mutation** (mutants
  not merged into WT); each record flags **`condition_match: "assay_only"`** — a fully
  licensed cross-protein ranking needs condition+assay matching (Service A).
- **Sequence axis (de-conflated)** — TANGO / AGGRESCAN / PASTA / Waltz as **separate
  named sub-axes** (granularity, endpoint, version, explicit reduction, `condition_inputs`).
  Only **PASTA & Waltz are amyloid-specific**; **TANGO & AGGRESCAN are generic**
  β-aggregation/hotspot predictors. A comparison is **licensed only on endpoint match**:
  vs an **amyloid dye** (ThT/ThS/**Congo Red**/Cytofluor) only PASTA/Waltz are licensed;
  an **unknown/None assay licenses nothing** (never silently "generic"). **Zyggregator
  & CamSol declared absent**; **NuAPRpred declared intentionally-excluded** (auditable,
  not silently dropped). Predictor reductions are **WT-only** (mutants excluded), and
  PASTA's `+10000` "no-pairing" sentinel is treated as missing. **Waltz is coverage**
  (validated-hexapeptide count; absence = untested, not non-amyloid), not a propensity
  magnitude.
- **Structure linkage** — associative CPAD/PDB panel (PDB cross-refs, APR + amyloid-
  structure counts); polymorphism a first-class caveat; never causal.

Run:
```
python etl/cpad_sequence_etl.py          # one-time: sequence_structure.json (v1.1)
python engine/m6_propensity.py           # propensity.jsonl (per protein × construct × assay)
```
Real corpus: **164 protein×construct×assay records** — 127 with a sequence axis, 135
with structure linkage, **121 corpus-anchored** (8 reference-scale), **35
censored-only suppressed**, 69 with a γ surface. E.g. Aβ40 WT (ThT): cohort
feature-percentile 17.3 → **propensity percentile 82.7** (`low_is_high_propensity`,
self-excluded N=110); sequence axis — **PASTA (−8.9, mean −4.4 after sentinel-strip) & Waltz
(6 validated hexapeptides) licensed vs ThT; TANGO & AGGRESCAN barred** (generic);
48 structures (43 amyloid) + 86 APRs.

> Deferred (documented): reference scales
> (`m6-refscale-1.0`) and the n≥20 interval floor are first-pass; AGGRESCAN/TANGO
> reductions are fragment-level maxima over deposited WT peptides, not whole-sequence
> native runs.
>
> **No longer deferred:** `null_disagreement_rate` — M10 phase P3 (below) now
> computes it at the protein level against a pre-registered null. It remains
> first-pass in one respect: the predictor false-positive rate feeding the null is
> a documented pre-registered constant, not a Service-C-calibrated one.

### M7 — output assembler (graceful degradation built in)
`m7_assemble.py` — joins the M1–M6 artifacts into one **`ProteinAnalysisResult`** per
curve (no new inference) plus an FDR-aware corpus rollup (`m7-assembler-1.0`):

- **`information_yield`** — a sortable honesty tier: `mechanistic` > `scaling` >
  `descriptive` > `signal_only` > `uninformative`. **`scaling` requires a
  *significant* γ** (reliable AND 95% CI excludes 0), so a reliable-but-CI-spans-zero
  γ falls to `descriptive` with a caveat — no borrowed scaling claim.
- **Degenerate-case contract** — for `signal_only`/`uninformative` (the common case)
  it emits a **compact** result (regime + yield + one ranked **M9 hook** + ≤3
  impact-ranked caveats), *not* a caveat avalanche; heavy fields are omitted.
- **No manufactured certainty** — `mechanistic.confidence` is **null when the verdict
  isn't licensed** (the M5 shape confidence is preserved separately as
  `shape_confidence`); a **construct mismatch refuses the propensity** (no Wild-Type
  bleed onto a mutant curve); censoring/`t50_status`, `gates_failed`, `anchor_kind`,
  `condition_match` are carried verbatim.
- **`engine_build_id`** — a deterministic hash over **all** comparability axes
  (M1–M7 + anchor + reference-scale + ETL schemas + a `corpus_version` folding every
  CPAD source digest incl. APRs & structures + harness + env + the **governed
  `thresholds_version`** and the governed **`join_policy_version`**);
  `comparable()` gates cross-build comparison on equality. Applying Service C's
  calibration (thresholds-1.0) folded a new axis and flipped the build id
  **`bc442463dfdf` → `e660cb452b41`**; correcting the M6/M7 assay axis
  (join-policy-1.0/1.1) flipped it again **`e660cb452b41` → `29bb01bedd97` →
  `3e5fa43cd1a8` → `6baf9f251541` → `bcceb97a46a1` → `271f46f057e6` → `d6ffd19f850c`** — older
  results are correctly marked non-comparable (§7).
- **Rollup** — tier/regime counts, **both** `n_with_reliable_gamma_fit` and
  `n_with_significant_gamma`, corpus-wide anomaly-FDR summary, and `corpus_caveats`
  naming the universal limitations (graph digitization; agitation/seeding unknown).

Run:
```
python engine/m7_assemble.py             # protein_analysis.jsonl + m7_rollup.json
```
Real corpus (1,654 curves): **1,194 full / 460 compact**; yield **descriptive 1,135 ·
signal_only 454 · scaling 59 · uninformative 6 · mechanistic 0**. The significance
gate is what separates `scaling` from `descriptive`: **77** curves have a reliable
γ fit but only **59** a γ distinguishable from zero — both counts are reported, and
the reliable-but-CI-spans-0 remainder is demoted with a caveat rather than
promoted. These counts carry `engine_build_id` **`d6ffd19f850c`** and `corpus_version`
`c09ac0cf398e`, so the whole M1–M9 artifact set here is post-recalibration,
post-assay-fix and internally comparable. The tier counts are UNCHANGED by
join-policy-1.0 (the yield ladder takes no propensity input); what moved is the
`propensity`/`sequence_axis` content and the build id. The prior builds
(`bc442463dfdf` first-pass thresholds, `e660cb452b41` pre-assay-fix) are correctly
marked non-comparable (§7) and their numbers are not reported above; the
`e660cb452b41` artifacts are retained under `data/processed/_builds/`.

### calibrated_thresholds — the GOVERNED apply artifact (thresholds-1.0)
`calibrated_thresholds.py` — the signed/dated/versioned record (§7/§8) applying
Service C's **held-out** calibration back into M1/M3/M5. For each threshold it
retains BOTH the previous `first_pass` value and the applied `calibrated` value
(+ the Service-C source: held-out AUC, n), plus an `APPLY_RECORD`
(`applied_at 2026-07-01`, `by "governed apply step"`, `rationale "Service C
held-out calibration (service-c-1.0)"`, `source_artifact "service_c_calibration.json"`).
M1/M3/M5 import the applied value via `calibrated_value(...)` (constant NAMES
unchanged). Applied: **M1 LAG_SLOPE_FRAC 0.30→0.45 · M3 COND_NUMBER_NONIDENT
1000→107 · M5 THRESHOLD_SHARPNESS 3.0→5.7 · M5 THRESHOLD_LAG_RATIO 0.80→0.97 ·
M5 ANOMALY_FDR_ALPHA 0.05 (unchanged)**. `THRESHOLDS_VERSION` folds into
`engine_build_id`, so the apply invalidates prior (first-pass) results per §7.
Reversible: restore `first_pass` + re-bump the version.

> **`env_manifest` is now a real stack fingerprint** — python micro, numpy,
> scipy, BLAS vendor+version, OS/machine — so two builds on the same Python
> minor but a different BLAS are correctly non-comparable. Still deferred: the
> live THREAD-COUNT environment is excluded on purpose (it can reorder BLAS
> reductions, but folding it in would make the build id depend on how the process
> was launched), so §7's pinned-container guarantee still needs a container.

### join_policy — the GOVERNED apply artifact for the M6/M7 assay axis (join-policy-1.1)
`join_policy.py` — the second signed/dated/versioned record in the
`calibrated_thresholds.py` mould, applying the M6/M7 **assay-axis** correction
(1.0 applied 2026-08-10, 1.1 on 2026-08-17). It carries, for each of five
decisions, the `previous` behaviour, the `applied` one, and the measurement or
citation that justified the change:

| decision | previous | applied |
|---|---|---|
| `M6_GROUP_KEY` | `(uniprot, construct)` | `(uniprot, construct, assay)` |
| `M6_GAMMA_JOIN` | protein name, last-wins | protein name **and** assay |
| `M6_ANCHOR_LEAKAGE` *(1.2)* | leave-one-**group**-out | leave-one-**study**-out |
| `PROPENSITY_ASSAY_GATE` | construct only | construct **and** assay |
| `SEQUENCE_AXIS_ASSAY_GATE` | any uniprot record | assay-matched only |
| `ASSAY_ENDPOINT_AMYLOID_ADDITIONS` *(1.1)* | `[]` — k114 was `unknown` | `["k114"]` — k114 is `amyloid` |

Two properties make this a governed change rather than a patch. **(1) It is folded
into `engine_build_id`.** `JOIN_POLICY_VERSION` is a 15th `build_components` axis,
added because none of the other 14 is a function of join code — without it the fix
would have shipped the *same* build id with *different* numbers, and `comparable()`
(which tests id equality alone) would have declared two materially different corpora
comparable. `e660cb452b41` → `29bb01bedd97` (join 1.0) → `3e5fa43cd1a8`
(join 1.1) → `6baf9f251541` (estimation 1.0) → **`bcceb97a46a1`**
(estimation 1.1) → `271f46f057e6` (join 1.2) → **`d6ffd19f850c`** (env axis).
**(2)** The revert path is real, not
decorative.** `m6_propensity` and `m7_assemble` **read** the policy at join time
rather than hard-coding the new behaviour, so restoring `applied` → `previous`
restores the prior corpus exactly. Two tests prove it by flipping the policy and
asserting the *original defect returns* — a retained `previous` that no consumer
reads is a comment, which is the one way this precedent can be copied badly.

**`k114`, corrected in 1.1.** 1.0 deliberately left
`_assay_endpoint_class("k114")` returning `"unknown"` — honest, since the call
needed a citation the join fix did not carry, but **wrong on the science**. K114 is
(trans,trans)-1-bromo-2,5-bis(3-hydroxycarbonyl-4-hydroxy)styrylbenzene, a
Congo-red/X-34-derived fluorophore introduced specifically to quantify amyloid
fibrillogenesis — it binds the cross-β fibril and fluoresces on binding, the same
detection principle and the same molecular target as ThT, ThS and Congo Red, all of
which this table already classes as amyloid (Crystal AS *et al.*, *J Neurochem*
2003;86(6):1359-1368). `unknown` was not the conservative choice: it **barred PASTA
and Waltz — the cross-β-specific predictors — from comparison against a
cross-β-specific assay**, which is precisely the endpoint match the licensing rule
exists to permit. The 9 k114 curves now license PASTA + Waltz and still bar
TANGO + AGGRESCAN, identical to the other three dyes. It is kept as its own
governed decision, with its own version bump and citation, rather than folded into
the join fix. A test pins the **entire corpus assay vocabulary**, so a new assay
cannot arrive and silently license nothing.

### bootstrap_policy — the GOVERNED apply artifact for estimation (estimation-policy-1.0)
`bootstrap_policy.py` — the third governed record, after `calibrated_thresholds`
and `join_policy`, covering the two decisions that determine how the corpus is
FITTED rather than how it is joined:

| decision | previous | applied |
|---|---|---|
| `M2_START_TOPUP` | name-keyed starts only (9 of 13 models got ONE) | deterministic spread within each parameter's bounds |
| `M3_SEED_STREAM` | constant seed 0 for every curve | per-series BLAKE2b derivation |
| `M3_BOOTSTRAP_BLOCKING` *(1.1)* | pointwise wild bootstrap | blockwise, `ceil(n**(1/3))` |

Same two properties as `join_policy`. **(1)** `BOOTSTRAP_POLICY_VERSION` is folded
into `engine_build_id` as the 16th axis — no other component is a function of the
fitting code, so without it a change to how every curve is fitted would ship under
an unchanged id. **(2)** `m2_fit` and `m3_select` **read** the policy, so each
correction stays one value from its predecessor; the tests exercise BOTH branches
of BOTH decisions, including one that flips the seed policy back and asserts the
original shared-stream defect returns exactly.

### external_forward_model — de-circularising Service C (external-smoluchowski-1.0)
`external_forward_model.py` — **the answer to the question "compared to what?"**.

Service C's validation was **circular**: its "mismatched generator" arm generated
from `mechanistic.simulate_mass_fraction`, *the same function the engine fits*, so
"the engine recovers the mechanism" reduced to "the engine recovers its own
model". No recovery number obtained that way is evidence about the world.

This module integrates the **size-resolved master equation** that model is derived
from — one state per aggregate size plus free monomer, with primary nucleation,
two-ended elongation, uniform-breakpoint fragmentation and surface-catalysed
secondary nucleation, and **no moment closure**. The engine's model tracks only
two moments (`P`, `M`) and represents no size distribution at all, so this is a
structural difference, not a reparametrisation.

**It agrees where the closure is valid and diverges where it is not** — both
halves are required, and both are pinned by test. Agreement licenses calling it
the same physics; divergence is what makes generating from it worth doing:

| regime | closure vs master equation, max abs mass fraction |
|---|---|
| primary nucleation + elongation | 0.006 |
| fragmentation (weak or strong) | ~0.007 |
| **secondary nucleation** | **0.114** |
| strong monomer depletion | 0.077 |

Mass is conserved to **~1e-16** and the size-axis truncation never binds
(occupancy < 1e-29), so the divergence is physics rather than numerics.

#### What it measured — the central result

Ground truth is known by construction, so the engine can be asked directly
whether AICc over its four mechanistic variants names the right one. 24 cases
(4 mechanism classes × 6 deterministic rate replicates), both arms paired on
identical rate constants so misspecification is not confounded with difficulty:

| arm | mechanism-ID accuracy (4 classes, chance 0.25) |
|---|---|
| **internal** — the engine's own generator (**a ceiling**) | **0.583** |
| **external** — the independent master equation | **0.542** |
| misspecification penalty | 0.042 |

**The internal figure is the headline, and it is a ceiling**: that data came from
the very model being selected, and the engine still names the wrong mechanism
~42% of the time. The shortfall is therefore an **identifiability limit, not a
misspecification penalty** — and the small internal→external gap says the same
thing, because there is little room left to lose. `fragmentation` systematically
absorbs the other classes.

This is the first time PRISE's standing refusal to license mechanistic inference
(**corpus-wide `mechanistic: 0`**) has been supported by a measurement against a
forward model the engine does not fit. Previously that refusal rested on gates
and argument; it now rests on a number.

> **The descriptive arm is reported but is NOT evidence of robustness.** The
> closed-form bank fits *both* generators to R² ≈ 0.9999 and selects the same
> model in every case, even where the underlying trajectories differ by 0.11 in
> mass fraction. That is not the engine being robust — it is R² being blind to
> mechanism, which is exactly why the mechanism arm above is measured directly.

Run `python engine/external_forward_model.py` → `data/processed/external_validation.json`.

#### The stochastic arm — how much of the spread is chemistry

The deterministic generator tests **misspecification**. It cannot test the other
thing every real experiment has and every model here lacks: **intrinsic
copy-number noise**. Amyloid nucleation is a rare-event process — a handful of
nuclei can set the lag time — so between-well variance is not all measurement
error. An **exact Gillespie SSA of the same reaction network** now measures it:

| system size N | RMSE of SSA mean vs the master equation | well-to-well t50 CV |
|---|---|---|
| 500 | 0.0282 | **0.179** |
| 2,000 | 0.0238 | 0.110 |
| 8,000 | 0.0157 | 0.043 |
| 20,000 | **0.0033** | 0.036 |

The RMSE column is the **correctness check** — the SSA mean must approach the
deterministic solution in the large-N limit, or the arm is simulating different
chemistry and its spread would mean nothing. It does, monotonically.

The CV column is the **finding**: at N=500, t50 varies **17.9% between nominally
identical wells from the chemistry alone**, with *zero* measurement error
anywhere in the generator. Every estimator in this engine treats residual scatter
as measurement noise. Where nucleation is rare, some of it is not — and no
residual diagnostic on a single curve can separate them, which is why this is
measured rather than assumed small.

> Still deferred, and now the honest remainder: the rate laws are a
> mechanistically grounded **choice**, not a claim about the true chemistry of any
> named protein; and neither arm is calibrated against real flagged traces — the
> synthetic→real transfer measurement and the artifact-realism distributional
> check are unchanged.

### Service C — validation & identifiability engine (de-circularized)
`service_c.py` — the synthetic-calibration backbone (`service-c-1.0`). It generates
curves from a *known* ground truth, runs the real engine on them, and measures what
the engine recovers — **pre-registered** (every noise/artifact/censoring constant is
fixed from physics/instrument specs, never tuned to the data it later judges, C2) and
**deterministic** (seeded → byte-identical re-runs). Sits on top of M1–M7; never edits
them — it emits *recommendations*, not in-place changes.

- **Synthetic generator** — closed-form *or* Tier-B ODE truth + heteroscedastic ThT
  noise, baseline drift, graph-digitization, and left/right censoring; `make_series`
  runs the real M1 triage so downstream modules see an authentic curve.
- **Parameter recovery** — bias / RMSE / Wald-95% coverage per parameter + t50.
- **Data-regime confusion + equivalence classes (K2)** — and it genuinely tests the
  concentration-series advantage: single-curve pools `[logistic, richards]` and
  collapses **all four Tier-B mechanisms into one class**, whereas a **multi-
  concentration family resolved by γ_global splits them into four singletons** (mean
  γ 0.81 saturating_secondary · 0.86 fragmentation · 0.96 nucleation_elongation ·
  1.20 secondary_nucleation) — the K2 "a clean series breaks sec-nuc/fragmentation
  degeneracy" claim, *measured*, not asserted. Five of the six pairwise separation
  AUCs are 1.0; the hardest pair, `fragmentation | saturating_secondary`, separates
  at **0.89** — still above the pre-registered 0.7 floor, but it is the weakest link
  and is reported as such rather than rounded up.
- **FDR-null with a power arm** — realized FDP 0.0 **and** power 1.00 on a bank-
  unrepresentable (double-sigmoid) alternative (control is only claimed when power is
  demonstrated).
- **Calibration** — ECE 0.033 / Brier 0.063 on M5 descriptive confidence; **held-out**
  threshold recommendations (train-chosen, out-of-sample AUC reported — no in-sample
  leakage; a test proves the leakage mechanism on pure noise). Current→recommended:
  M1 lag-slope 0.30→0.45 · M3 cond-number 1000→107 · M5 sharpness 3→5.7.
- **De-circularization (C1)** — a mismatched generator (Tier-B ODE truth vs closed-form
  fitter) reported via **mechanism-misidentification degradation** (0.67), not a
  vacuous R². The full external KMC/Smoluchowski generator + blind real-literature
  reality check are explicitly **deferred**.

Run (build-time batch, ≈2 min):
```
python engine/service_c.py               # -> data/processed/service_c_calibration.json
```
> Service C emits recommendations only. **Those recommendations have now been
> APPLIED** via the governed step (`engine/calibrated_thresholds.py`,
> thresholds-1.0, 2026-07-01): M1/M3/M5 run the calibrated cutoffs, the previous
> values are retained, and `engine_build_id` is bumped. Service C itself still never
> edits the modules — it reads the live constants (its recorded `current` now
> reflects the calibrated value once re-run). Deferred slices: external
> forward-model generator, blind real-literature check, artifact-realism
> distributional check, synthetic→real transfer, M6 null-disagreement.

### Conformal — split / Mondrian conformal prediction (calibrated coverage)
`conformal.py` — the finite-sample-valid coverage layer (`conformal-1.0`), closing
the §4 admission that the engine emits point estimates + wild-bootstrap percentile
CIs but no *calibrated* probabilities. It sits on top of Service C + M1/M2/M4/M5 and
never edits them. Calibration source = Service C synthetic curves with **known**
t50/regime (the only ground truth available); the real engine is run on each so the
nonconformity scores are exactly what the deployed pipeline produces.

- **Continuous t50** — split-conformal absolute-residual score `s = |t50_hat − t50_true|`;
  the `ceil((n+1)(1−α))` empirical quantile gives a marginal interval
  `[t50_hat − q, t50_hat + q]` with **guaranteed ≥ 1−α coverage**. **MONDRIAN**:
  the quantile is stratified by `(censoring_class, assay_endpoint amyloid|generic,
  descriptive-regime tier)` — the SAME tags the engine produces — so coverage holds
  within strata; thin strata fall back to the marginal quantile (flagged).
  **Right-censored t50 is a lower bound → ONE-SIDED** `[t50_hat − q, +∞)`; left-only
  t50 and non-monotonic curves are excluded from the t50 channel (consistent with M4).
- **Classification regime** — nonconformity `1 − normalized_akaike_weight(true_regime)`
  yields **prediction SETS** `{regime : 1 − score ≤ q}` with ≥ 1−α marginal coverage;
  set size > 1 only when the engine's evidence is genuinely split (a saturated all-label
  set is flagged `regime_unresolvable_in_stratum`). Mondrian-conditional on
  `(censoring|assay)`.
- **Validation** — a frozen, seeded proper-calibration / validation split; realized
  coverage (target vs achieved) + mean width / set size, overall and per Mondrian
  stratum. At `n=400, α=0.1`: **t50 marginal coverage 0.97** (mean width 7.8 h, 110
  one-sided), **regime marginal coverage 0.92** (mean set size 2.3) — both ≥ the 0.90
  target. Six t50 strata stand on their own (much tighter) quantile, e.g.
  `none|amyloid|cooperative_sigmoidal` q≈1.6 h vs the 7.6 h pooled marginal.

Deterministic: seeded generation + frozen split → byte-identical re-runs. Run
(build-time batch, ≈4 min):
```
python engine/conformal.py               # -> data/processed/service_c_conformal.json
```
> Honesty: calibrated on SYNTHETIC ground truth; coverage on real CPAD curves holds
> insofar as the pre-registered noise/censoring law matches the digitized traces — a
> stated, testable assumption (§6C), not a claim of validity on arbitrary real data.

### M8 — inferential reachability map (Phase 2)
`m8_reachability.py` — **consolidates** the existing artifacts (ETL census, M5/M6/M7
rollups, Service C) into a falsifiable map of *what is inferable from CPAD's bulk
kinetics, and what is not* — **no new inference**. Every headline carries the
conditioning clause "**reachability under the PRISE forward model**" and a mandatory
**validity ceiling** (counts are upper bounds on reachability / lower bounds on
degeneracy, conditional on forward-model adequacy — backed by Service C's 0.67
mismatched-generator degradation; the full external validation is deferred).

- **Inference ladder** (of 1,654 curves): signal_present **1,606 (97%)** → fittable
  **1,240 (75%)** → descriptive_regime **1,194 (72%)** → scaling/γ-significant **59
  (4%)** → mechanistic_licensed **0**. Monotone, and reconciles with the M7 rollup.
- **Reachability grid** — data_richness × condition_completeness, each cell tagged
  recoverable / degenerate / unconstrained (corpus: degenerate 1,002 · recoverable
  649 · unconstrained 3; partitions all 1,654).
- **Blocker census** (impact-ranked): agitation/seeding unknown **1,654 (100%)** ·
  no concentration series **1,005 (61%)** · censoring **918 (56%)** · non-mass/
  unknown assay 46 · low metadata completeness 3.
- **Mechanistically-resolvable boundary (the headline)** — STRUCTURAL upper bound
  **33 proteins / 48 series** (≥3-concentration mass-assay families that Service C
  shows a γ-series *could* resolve) vs **0 actually licensed** → **GAP = 33**, closed
  by a single metadata fix (recording agitation/seeding). *Protein unit here is
  `protein_id` (CPAD name; denominator 105 — finer than M6's UniProt keying, so
  Aβ40/Aβ42 count separately); the series count (48) is unit-free.*
- **Metadata explanation (M11 dependency — ADDITIVE + GRACEFUL)** — M8 *consumes*
  the M11 corpus artifact `data/processed/metadata_quality.json` to make the map
  **causal**: a `metadata_explanation` block, a `why_gap_unresolved` on the boundary
  (the ranked `why_blocked` metadata fields + their M11 information-gain — #1 is
  `record_agitation_and_seeding`, Service-C-measured), a per-blocker M11
  cross-reference on the blocker census (field(s) responsible + gain-to-unblock),
  and a top-level `metadata_completeness_summary` (median metadata_score **0.69** but
  median mechanistic_completeness **0.4** — the corpus records the *conditions* but
  not the mechanism-gating {agitation, seeded}, which is *why* the GAP is unresolved).
  **If `metadata_quality.json` is absent this degrades to `{m11_available: false}`
  and every other M8 key is unchanged.** Honesty carried through: completeness ≠
  correctness, and M11 information-gain numbers are structural upper bounds (this
  map's validity ceiling) except the Service-C-measured ones.

Run:
```
python engine/m11_metadata.py            # -> metadata_quality.json (+ .jsonl) — build FIRST for the causal block
python engine/m8_reachability.py         # -> data/processed/m8_reachability.json
```

### M9 — experimental-design recommender (Phase 2)
`m9_recommender.py` — for each analysis unit, recommends the experiment that would
most reduce **practical** (not structural) degeneracy. **Committed method (stated in
every output):** a **Service-C-distilled rule table whose gains are *quantified by
Service C's measured degeneracy reduction*** (a γ-resolved concentration series
splits the Tier-B mechanism class 1→4 — a measured ~4× resolution, carried with the
per-pair separation AUC). The full **EIG/optimal-design** computation is explicitly
**deferred** as research — M9 does not pretend to compute EIG. It elevates M7's
`m9_hook` stub into a standalone, cost-weighted recommender.

- **Candidates**, ranked by reduction ÷ cost: record agitation & seeding (cheap) ·
  add a concentration series (medium, Service-C-quantified 4×) · extend window /
  de-censor t50 · add replicates · use a mass-proportional assay.
- **Dependency-honest crediting** — recording agitation/seeding only *unblocks
  mechanism* where a concentration series **already exists** (444 series units → full
  gain, ranks #1); on the 1,209 single-curve units it's **discounted + flagged
  `necessary_but_not_sufficient`** (the binding prerequisite there is the series, so
  `add_concentration_series` ranks first). No over-credit.
- **Practical-vs-structural gate** — refuses to recommend an experiment for
  *structural* degeneracy (e.g. the Service-C-irreducible `{fragmentation,
  saturating_secondary}` pair, or M3 structural non-identifiability), saying so.
- **Validity ceiling** inherited from M8 (every gain an upper bound, conditional on
  forward-model adequacy; cites the 0.67 mismatch degradation).
- **Corpus rollup headline:** *recording agitation & seeding is the cheapest action
  that closes the M8 GAP — it unblocks the mechanistic license for the 33 series-
  bearing structural-reachable proteins (48 series); on single-curve units it only
  removes a future gate.*

Run:
```
python engine/m9_recommender.py          # -> m9_recommendations.jsonl + m9_rollup.json
```

### M0 — dataset selection & cohort assembly (the entry point)
`m0_cohort.py` — the §3-M0 entry point that was previously unbuilt: the product only
ever analysed one curve at a time. M0 lets a caller assemble **one or several** CPAD
datasets for a protein and routes the assembled set to the analysis its structure
actually licenses, behind a **comparability gate that runs before any fitting**.

- **`data_mode` routing** — `single_curve` (per-curve descriptive + features) ·
  `replicates` (identical condition vectors → `pooling.replicate_meta` random-effects
  + stochastic-nucleation scatter) · `concentration_series` (≥3 concentrations,
  everything else matched → dual-γ + the shared-rate ODE global fit) ·
  `condition_series` (pH/temperature varies at fixed concentration → a
  feature-vs-condition **surface**, explicitly **no γ**) · `confounded` (>1 condition
  axis co-varies → **REFUSED**, descriptive-only + the co-varying axes named).
- **Comparability gate** — the set must share protein/uniprot, `construct_id` and
  `assay_type`. A mismatch is a hard confound flag naming the offending field(s); an
  *unknown* comparability-critical field degrades and widens, never silently merges.
- **`window_of_validity`** — the measured condition ranges; inference is not licensed
  beyond them. No extrapolation, no synthesised curve for an unmeasured condition.
- **Honest framing** — M0 **orchestrates**; it re-implements no inference. It calls
  the existing tested M1/M2/M4/M5, `global_fit` and `pooling` entry points and never
  modifies them. A bad or empty selection degrades to a flagged minimal
  `cohort_result`; it never raises.

Run:
```
python engine/m0_cohort.py               # demo over a REAL concentration series
```
The bundled demo assembles `CPAD-CS-38932798` (Aβ40, 10 member curves): M0 reports
`data_mode = concentration_series` on **4 distinct concentrations**,
`comparability_ok = true`, and a window of validity of 10–70 µM — then hands off to
the licensed dual-γ route. There is no corpus-wide M0 artifact by design: M0 is a
per-selection router, and the selection is the user's.

### M10 — cross-modal sequence × kinetics × structure mining
`m10_crossmodal.py` (`m10-crossmodal-1.0`) — PRISE co-holds sequence predictors,
measured kinetics and structure/APR data for the *same* proteins; until M10 the
product only **displayed** the sequence axis and never asked whether it **predicts**
anything. M10 mines it and reports the answer honestly, including when the answer is
null. No fitting: it reads precomputed artifacts only.

**The binding honesty constraint:** the effective n is **proteins, not curves**. A
predictor score is a per-protein constant replicated across that protein's curves, so
every association uses the protein as the unit, every null is a protein-level
permutation, and predictive skill is leave-one-**protein**-out.

- **P1 sequence → kinetics** (the headline, and it is a null result). Over 45 proteins
  / 1,433 curves, 20 have both an endpoint-matched predictor and a point t50. First
  M10 computes the **variance ceiling**: a per-protein constant cannot explain
  *within*-protein variance, so the most any sequence-only predictor could explain is
  **32.0%** of t50 variance. Against that ceiling, PASTA vs median log₁₀ t50 gives
  ρ = −0.08, **LOPO R² = −0.17** (worse than predicting the mean) and permutation
  p = 0.74. Stated plainly: the endpoint-matched sequence predictors **do not** explain
  the measured rate among known amyloids.
- **P2 structure/APR ↔ regime** — the cleaner signal, still associative. Median APR
  count by regime over 44 proteins is 0 / 0 / 4 (no_detectable /
  gradual_non_cooperative / cooperative_sigmoidal), an increasing trend, but the
  protein-level Jonckheere permutation test gives **p = 0.050** — right on the line
  and *not* declared significant. `relationship: associative_only`, never causal.
- **P3 discordance** — fills M6's previously-null `null_disagreement_rate`:
  **2 / 22** evaluable proteins are sequence↔kinetics discordant (observed 0.091)
  against a **pre-registered** null of 0.312 (PASTA ~15% FP ⊕ 19.1% corpus censoring),
  two-sided binomial p = 0.022 — fewer discordances than the null predicts.
- **P4 γ × predictor** — an honestly thin descriptive sidebar; no test is run on it.

Because the corpus is ~97% amyloid/ThT and almost entirely positive-class, M10 tests
**rate/regime among known amyloids** — never "amyloid vs not".

Run:
```
python engine/m10_crossmodal.py          # -> data/processed/m10_crossmodal.json
```

### M11 — metadata quality & information completeness
`m11_metadata.py` (`m11-metadata-1.0`, ontology `m11-ontology-1.0`) — formalises and
quantifies the metadata layer the engine already licenses inference against. Every
gate M11 reasons about **mirrors** the real M5 mechanistic gates and the M8
reachability structure, read from the same `condition_vector` + `field_provenance`
M1 builds; M11 reads artifacts and mutates nothing.

Per series it emits a weighted `metadata_score`, a **separate**
`metadata_uncertainty` trust scalar, a per-missing-field `delta_inferential_power`
(which tiers it blocks + the consequence chain), a `mechanistic_completeness` from a
DAG derived from M5's gates, and missing fields ranked by information-gain ÷ effort in
M9's action language.

**The honest limitation, stated on every record: completeness ≠ correctness.** The
score measures how much metadata is *present*; it cannot verify a recorded value.
Information-gain numbers are **structural upper bounds** under the DAG, inheriting
M8's validity ceiling — except the ones stamped `service_c_measured`. The default
weights are a versioned, configurable modelling choice, never objective truth. And
`not_captured_by_source` (the field is not a first-class CPAD field) is kept distinct
from "the study failed to record it".

Real corpus (1,654 curves / 105 proteins): median `metadata_score` **0.685** but
median `mechanistic_completeness` **0.4** — the corpus records the *conditions* and
not the mechanism-gating {agitation, seeded}. **Mechanism is licensed on 0 / 1,654
curves**, and M11 names the universal blocker explicitly: agitation and seeding are
`unknown` corpus-wide, so the M5 quiescent-vs-shaken branch is undetermined for every
curve regardless of how complete the rest of the metadata is. This is the same gate
M5 applies and the same GAP M8 reports — M11 quantifies *why*, it does not invent a
parallel licensing logic. Top-ranked corpus action: `record_agitation_and_seeding`.

Run:
```
python engine/m11_metadata.py            # -> metadata_quality.json + metadata_quality.jsonl
python engine/m11_metadata.py --limit 200
```

### M12 — information content & experimental-design geometry
`m12_information.py` (`m12-information-1.0`) — unifies the information geometry the
pipeline already touches into one per-curve treatment. It is deliberately *consistent
with*, not a third opinion of, the existing modules: the sensitivity Jacobian is the
same finite-difference object M3 and `global_fit` build (M12's condition number on the
column-normalised FIM matches M3's collinearity number, cross-checked in the tests),
the observed FIM 𝓘 = SᵀΣ⁻¹S mirrors `global_fit._identifiability`, the lag/growth/
plateau segmentation reads M4's feature boundaries, and θ̂ comes from the M2-selected
best-AICc fit.

Per curve: the eigen-spectrum and effective rank (hard + entropy), estimator
differential entropy, per-parameter CRLB observability and stiff-vs-sloppy
identifiable combinations, the per-point information-density profile D(t) and its
fraction by phase, measurement redundancy, a Sherman–Morrison optimal-design scan for
the best next time-point, a prior-conditional EIG, and an M9-language recommended
measurement.

**Honesty attached to every result:** the FIM is **local, linearised** (a Laplace
curvature approximation at θ̂), **conditional** on the M2-selected model *and* the
σ²I noise model, and **post-selection** — model-averaging over M3's selection
posterior is a documented deferred extension. The `information_richness_score` is a
bounded collapse of the spectrum and is always returned *with* the spectrum it
summarises, never instead of it.

Real corpus (1,240 fittable curves, all 1,240 returning a result; 690 cooperative):
median hard effective rank **4** against a median of **4** parameters, with **66
curves (5.3%)** sloppy in the strict erank < n_params sense — but the median FIM
condition number is ~1.2×10⁴, so "full rank" here is not the same as "well
conditioned". Median information richness 0.570. The **growth phase carries the
largest median share of the information** (0.435 vs lag 0.264 / plateau 0.305) and
beats the plateau on 87% of curves; the modal recommendation is
`sample_denser_at_transition` (489 curves), then `extend_observation_window` (402).

Run:
```
python engine/m12_information.py         # -> information_content.jsonl + information_content.json
python engine/m12_information.py --limit 200
```

### M13 — Bayesian optimal experimental design (BOED)
`m13_boed.py` (`m13-boed-1.0`) — the rigorous capstone of the design-advice thread
(M9's heuristic rule table → M12's per-curve geometry → M13's multi-variable BOED).
It scores candidate next experiments ξ by expected information gain, ranks by EIG ÷
cost, exposes the non-dominated Pareto frontier, and plans a greedy K-step batch whose
marginal gains diminish (submodular log-det). It **reuses M12's exact FIM machinery**
on prospective designs rather than reimplementing it.

The design principle is that its 8 design variables are of **two honest kinds**, and
every candidate says which it is:
- **Quantitative** (`gain_basis='fim'`) — concentration (single new [m] *and* a ≥3-point
  series), temperature, pH, replicate count, measurement frequency Δt, sampling
  duration t_max. These can be *simulated*, so the EIG is a real D-optimality number.
- **Gating / categorical** (`gain_basis='licensing'`) — agitation, seeding. These are
  not a smooth FIM direction but a **licensing unblock**, anchored to Service C's
  measured 1→4 Tier-B split, and credited in full only where the other prerequisites
  exist (otherwise `necessary_but_not_sufficient`, exactly as M9 does).

**Honesty:** the prospective concentration curve is built by rescaling the fitted
timescale with γ — a stated **self-similar-shape approximation**. Extrapolation
beyond the measured T/pH window is flagged low-confidence. The cost/runtime table is a
versioned, configurable **estimate**, not real lab economics. And the mechanism gain
is an upper bound inherited from Service C.

Real corpus: **839 analysis units** (49 series units, 790 single-curve), all 839
scored. Top recommendation distribution: `increase_measurement_frequency` 482 ·
`extend_sampling_duration` 241 · `add_concentration_series` 86 ·
`record_agitation_and_seeding` 23 · `add_single_concentration` 4 · `vary_pH` 3.

Run:
```
python engine/m13_boed.py                # -> boed_recommendations.json + boed_recommendations.jsonl
python engine/m13_boed.py --limit 50
```

### M14 — cross-study hierarchical meta-analysis
`m14_meta.py` (`m14-metaanalysis-1.0`) — generalises `pooling.py`'s replicate-level
random-effects meta *up to the STUDY level*: one real publication = one study, and the
between-study variance τ²_between is the publication effect.

Built to the data reality rather than an idealisation: raw t50 cannot be pooled across
different concentrations (it scales with γ), so M14 is a **meta-regression** — it
regresses out the condition moderators (primarily log₁₀ concentration, plus pH /
temperature / construct where there is spread) and pools the study random effects of
the residual, on the **log t50** scale. Estimation is deterministic closed-form
(DerSimonian–Laird τ² with a Paule–Mandel/REML-style refinement) plus BLUP shrinkage,
reusing `pooling.py`'s own primitives; weak priors yield a credible interval for μ and
a **posterior-predictive interval for a new study**.

**Honesty:** pooling is always condition-**adjusted**, never raw across
concentrations; where a moderator is degenerate M14 falls back to
matched-condition-stratum pooling and says so. The concentration-moderator slope
β_logc should track −γ and is cross-checked against `gamma.jsonl`. Laboratory
(`author`) is estimable only where a group spans ≥2 studies for that protein, else
reported `CONFOUNDED_WITH_STUDY`. This is **not** a full latent-ODE-rate MCMC
hierarchy — that is a documented deferred extension.

Real corpus: of **60 proteins** reaching the meta layer, only **11** carry ≥2 studies
with poolable point t50 and are actually meta-analysed; the other **49** are reported
`status="single_study"` with **no fabricated pooling**.

Run:
```
python engine/m14_meta.py                # -> meta_analysis.json (+ per-protein dir)
python engine/m14_meta.py --selfcheck    # 2000-study scale timing demo
```

### M15 — literature ingestion & claim extraction
`m15_litingest.py` (`m15-litingest-1.0`) — turns protein-aggregation papers into
**structured, traceable claims — never trusted facts**. Every extraction is a `Claim`
with full provenance (doi + page + block + char-span + **verbatim** source text), a
measured confidence, and an entry in an **append-only, immutable ledger**. A
correction never overwrites: it appends a new claim carrying `supersedes=<old_id>`.
Divergent values across papers surface as `conflict` records with both provenances
retained — **never auto-resolved**. Extraction accuracy is *measured* against CPAD,
the curated gold standard, not asserted.

Downstream of the `Document` abstraction everything is deterministic and stdlib-only:
IMRaD section detection (Methods is the goldmine), regex + unit-grammar + offline-
dictionary extractors for the 15 targets grouped by epistemic status (CONDITIONS →
M11 ontology / condition_vector; REPORTED RESULTS → compared against PRISE's own
computed values; REPORTED SEMANTIC CLAIMS → lowest confidence, flagged narrative), a
table parser and a caption parser, the claim ledger + JSONL persistence, confidence
estimation with an ECE/Brier calibration hook, conflict detection, and the
`validate_against_cpad` round-trip harness.

The document-acquisition front end — PDF parsing, OCR, ML-NER and figure-curve
digitization — is a set of four **capability-detected pluggable seams**
(`pdf_to_document`, `ocr_page`, `ner_extract`, `digitize_figure_curve`), deliberately
separated from the deterministic core. Each resolves **at call time**, in order: an
implementation injected via `register_frontend` → a **detected** backing library
(built-in adapters dispatch to PyMuPDF / pdfminer / pytesseract / spaCy) → an honest
`NotImplementedError` **naming the missing dependency**. Detection uses
`importlib.util.find_spec`, so **module import stays stdlib-only and never imports, or
executes, a third-party module**.

A seam being unavailable is a **dependency-availability fact about this interpreter** —
PRISE ships stdlib + numpy/scipy only — and is **not** a claim about network access.
The load-bearing guarantee is that nothing is faked and nothing silently degrades: a
heuristic, a regex imitation, or an empty-but-plausible substitute is **never**
supplied in place of a real parse / OCR / NER result. Two consequences are deliberate:
**figure digitization is injection-only** — no importable library does axis-calibrated
digitization, and pixel-tracing without a calibrated axis would fabricate data — and
**ML-NER output is stamped `ml_ner` (0.50)**, below every deterministic method, never
laundered up to `regex_exact` via the gazetteer.

Fixtures and a live front end feed the *same* `Document` abstraction, so the
deterministic core is identical either way: in a stock checkout it runs on synthetic
**fixture documents** that mimic what a born-digital PDF parser would emit, and a
registered or detected front end substitutes real documents without the core changing.

**Honesty:** claims not facts; append-only; fully traceable; accuracy measured, not
asserted. **A curator approves** — M15 only ever *proposes*; nothing graduates into
the analytical corpus automatically.

Bundled fixture run: 3 fixture documents → **59 ledger claims** and **2 conflicts**;
the CPAD round-trip on the fixture mirroring pmid 18258258 scores field-level
precision / recall / accuracy **1.00** across 6 condition fields (ECE 0.059). That is
a clean result on a **6-field fixture** — it demonstrates the harness works end to
end, it is *not* evidence of extraction accuracy on real PDFs, and it should not be
read as one.

Run:
```
python engine/m15_litingest.py           # -> lit_claims.jsonl + lit_conflicts.json + lit_validation.json
```

### M16 — sequence ↔ structure ↔ kinetics bridge
`m16_structure.py` (`m16-structure-2.0`) — extends M10 with a **structure-feature
axis**: where M10 mined sequence predictors → kinetics, M16 asks whether
**structure-derived** features associate with the measured kinetics, under the same
protein-level rigour (effective n = proteins, LOPO CV, permutation nulls, variance
ceiling, BH-FDR).

This build **fetches real PDB coordinate files** into a local cache and computes
genuine 3D features in pure numpy — no BioPython / DSSP / FreeSASA / APBS — and each
extractor carries an explicit fidelity label so nothing is overclaimed: contact map /
contact order (**exact**), Shrake–Rupley SASA (**exact algorithm**), hydrophobic
patches (**real**, SASA-exposed + contact-clustered), secondary structure
(**approximation** — φ/ψ geometry, not DSSP H-bonding), electrostatic surface
(**approximation** — Coulombic/Debye–Hückel, not Poisson–Boltzmann), surface curvature
(**approximation**). A protein without a fetched PDB keeps the sequence/APR **proxy**
features, flagged; where both exist, both are reported so the real-3D-vs-proxy
contrast is visible. Any fetch or parse failure falls back to the proxy — it never
raises.

**The two honesty pillars, stamped on every output:** (1) the **native-vs-fibril
paradox** — a native-monomer structure is not the aggregation-competent state, so any
structure→kinetics link is an association and often a confounded one; every structure
carries a `native_or_fibril` stamp. (2) `causal: false` on **every** card, without
exception.

Real corpus: 45 proteins in the kinetics corpus, 26 with APR-proxy features, and
**real 3D features for 11 proteins from 11 successfully fetched PDBs (10 native,
1 fibril)** — which is exactly the point of pillar (1). Between-protein variance
ceiling on log₁₀ t50: **42.7%**. Of **30 association cards** (18 real-3D, 12 proxy),
**0 survive BH-FDR at α=0.05** — 24 null and 6 underpowered. The honest headline is a
null result at n≈11–26 proteins, reported as such.

Run:
```
python engine/m16_structure.py           # fetches PDBs -> structure_features.json
                                         #    + structure_kinetics_associations.json
python engine/m16_structure.py --no-fetch  # proxy-only, fully offline
```

### M17 — explanation, provenance & evidence graph (Stage A)
`m17_explain.py` (`m17-explain-1.0`) — every value the engine emits becomes a node
in a directed **acyclic evidence graph** whose edges are facts the engine *already
computed*. It answers "why does PRISE say this?" from artifacts on disk, never from
a fresh calculation.

**The load-bearing invariant: M17 DERIVES NOTHING.** No fitting, no statistics, no
thresholds, no numbers of its own — a harvester, a graph builder and a
deterministic renderer. The reason is not stylistic: if M17 could compute, its
explanations could disagree with the engine, and an explanation that contradicts
the thing it explains is worse than none. This is enforced **mechanically**, four
ways, so a violation *fails* rather than being discouraged:

- **Value provenance** — every non-null node value byte-compares equal to the
  artifact value at its declared key path, re-resolved against the artifact
  **re-read from disk** (not the pruned in-memory bundle, so pruning cannot hide a
  bad ref). A number M17 invented exists at no key path, so it cannot pass.
- **Arithmetic ban** — `_frac()` is the *only* function in the module containing a
  division/multiplication/subtraction/power, and it divides M17's own node counts,
  never an engine value. The suite parses the file with `ast` and asserts it.
- **Template-only rendering** — `render_node()` is one dict lookup plus one
  `str.format_map`, with no concatenation, f-string or `join`, so a sentence
  *cannot* be composed at runtime. Same graph ⇒ **byte-identical** text.
- **No numpy** — stdlib only, so every float arrives via `json.loads` as a plain
  Python float and no `numpy.float64` repr can drift between builds.

Node and edge vocabularies are **closed enums** (`result · datum · metadata_fact ·
method · assumption · gate · refusal · uncertainty · deferral · ceiling`;
`computed_from · assumes · gated_by · blocked_by · uncertainty_from · bounded_by ·
unblocked_by`). Assumption trees, evidence chains, uncertainty chains and prose are
**not four structures — they are four projections of the one DAG** (filter by edge
kind, topologically order), so they cannot contradict one another; they are
computed on read and never stored, because a stored view can drift from its graph.

**`confidence` is always a tagged LIST, never a merged scalar.** The engine carries
incommensurable confidences — M4's asymptotic **bootstrap CI**, `conformal.py`'s
finite-sample **conformal quantile**, M15's ECE/Brier-**calibrated claim
confidence**, M11's **completeness** (PRESENCE, explicitly not correctness) and
M5's **ordinal shape** level. Averaging them would fabricate a number and violate
invariant §10.5, so no code path reduces the list. Every node also carries a
**causal licence** the renderer inherits: a statistical-only source can never reach
a causal verb, and `measured_causal` is **declared but unused** — nothing in this
engine licenses a causal claim about nature (measured: 0 instances corpus-wide).

#### Coverage — measured, not asserted, and reported as TWO numbers

The denominator is **published in full** in `explanation_coverage.json`: every
**leaf key path** of the M7 per-series spine, unioned over all records (**332
distinct paths**), where a list is a leaf and a null or empty dict is a leaf at its
own path. A coverage percentage over a loosely-chosen denominator is
unfalsifiable, so the enumeration ships with the number.

```
by distinct field path            266 / 332       = 80.1%
by field instance, ALL series     210,446/260,151 = 80.9%   (1,654 series)
by field instance, EXPLAINABLE    198,502/244,011 = 81.3%   (1,240 series)
```

The two populations are **never blended**. "M17 has no adapter" and "the engine
produced nothing to explain" are different failures with different fixes, and only
the first is Stage B's to close: `regimes.jsonl`/`features.jsonl` hold **1,240**
records against the spine's **1,654**, so for **414 unfittable/suspect series there
is no refusal, no regime and no feature vector in existence**. Those return
`explanation_available: false` naming it an *engine* output gap.

> **The counter-intuitive result HAS NOW REVERSED — recorded, not quietly
> rewritten.** Through Stage A, instance coverage over *explainable* series
> (**49.5%**) was **lower** than over *all* series (**51.0%**): the 414 series with
> no M5 record emit a *compact* schema that is mostly `condition_vector`, which M17
> covers at 97.5%, so they out-scored full-schema series carrying the un-adapted
> M6/M3 blocks. **Stage B's M6 adapters inverted it** — explainable is now
> **81.3%** against all-series **80.9%** — because the full-schema series that were
> being dragged down by un-adapted `propensity`/`sequence_axis` blocks are exactly
> the ones M6 now explains. The mechanism above is still the correct history; only
> its *direction* has flipped, and it flipped for the expected reason. The point
> stands unchanged: a single blended figure would have hidden the effect in both
> eras, which is why it is still refused.

Per module (distinct paths explained / total):

| m1 | m4 | m5 | m3 | m6 | m7 | m9 |
|---|---|---|---|---|---|---|
| **39/40 · 97.5%** | **89/93 · 95.7%** | **22/31 · 71.0%** | 0/35 | **116/116 · 100%** | 0/13 | 0/4 |

> **m6 and m7 totals moved, and the direction matters.** Stage B re-attributed the
> three join-outcome fields `propensity.available`, `.construct_match` and `.reason`
> from m6 to m7: M7's `_propensity_block` *authors* them and they exist in no M6
> artifact. m6's denominator therefore **shrank 119 → 116** and m7's **grew 7 → 10**.
> The re-attribution flatters m6 (116/116 rather than 116/119), so it is called out
> here rather than left to be discovered: the total is **332** — it GREW by the
> three fields join-policy-1.0 emits (`propensity.assay_match`,
> `sequence_axis.assay_match`, `sequence_axis.prop_assay`, all M7's gate output),
> and was never rebased.

**The honest headline: where M17 has an adapter it is near-complete (m1, m4 and now
m6 all ≥ 95%); the gap is BREADTH, not depth.** With M6 closed, the highest-value
remaining Stage B target is **M3 — 35 paths at 0%**, then M7 (13) and M9 (4).
**Eleven** adapters now ship (M1, M2, M4-features, M4-γ, M5, M6-propensity,
M6-sequence-axis, M8, M11, conformal, M15); M3, M7, M9, M10, M12, M13, M14,
`pooling`, `global_fit` and `reality_check` have none and report zero rather than a
guess.

Real corpus: **236,653 nodes / 65,535 edges** (M6 emits nodes only, no edges), all
acyclic and zero adapter errors. **Cross-artifact value identity: 0 mismatched of
210,446 checked** — M17 nodes read the *originating* artifact, not the assembled
spine, so this proves the explained number is the number the product displays. The
adapter **contract test** checks **247 declared key paths** against the live
artifacts (0 failures): when an upstream module changes shape the M17 suite fails
*loudly* instead of silently degrading every explanation to
`explanation_available: false`.

> **Correction — "zero dropped edges" was never true corpus-wide.** Stage A
> published it on the strength of a test that checks **two hand-picked series**.
> Measured over all 1,654: **2,538 dropped edges across 437 series**, from
> `m11_metadata` and `conformal`, and **identical on HEAD before Stage B** — a
> pre-existing gap in the guard, not a Stage B regression. The standing guard is
> weaker than the claim it was supporting; widening it to the corpus is a Stage B
> item.

#### What M17 refuses

No prose beyond deterministic templates · no merged confidence scalar · no causal
verb from a statistical-only source · **no explanation it cannot trace** (→
`explanation_available: false` + a reason; never narrated around) · and **no
quantitative counterfactual unless the M11/M13 edge carries
`gain_basis: "service_c_measured"`** — a `structural_estimate` edge renders as an
**upper bound** inheriting M8's validity ceiling verbatim.

The design's worked example resolves end-to-end on the real corpus: the
`single_mechanism_call` refusal on **CPAD-TK-1061** → `gated_by` the failed
mechanistic licence → `blocked_by` the M8 corpus-wide blocker → the
`agitation`/`seeded` `"unknown"` metadata leaves → the §10.9 dye∝mass
**assumption** → the not-narrowed equivalence class → the §4 Service-C **deferral**
→ `unblocked_by` both Service-C-**measured** actions. The *typical* shape
(**CPAD-TK-1039**, ThT, assay gate **passing**) is tested with an explicit
**negative** assertion that no assay blocker is invented — 1,209 of 1,240 series
look like that, so fabricating a blocker that never fired is the failure mode that
actually threatens M17. The §10.9 assumption node is still present there: **a
passed gate is not a verified assumption.**

Run:
```
python engine/m17_explain.py              # -> explanation_coverage.json (~55 KB)
                                          #  + explanations_sample.jsonl (392 KB)
python engine/m17_explain.py --dump-graphs   # + explanations.jsonl (~90 MB)
python engine/m17_explain.py --series CPAD-TK-1061   # print one tree
```
> **Why the full dump is not the default.** A per-series graph is *derived*
> deterministically (~10 ms) from artifacts already on disk, so the 90 MB corpus
> dump is a stored view of stored data — the same category error as serialising the
> projections. It is also a practical hazard: `web/server.py` indexes large
> per-series JSONLs into memory **at startup**, and ~175k fully-traced nodes become
> several hundred MB of live objects at boot in an app whose point is being a
> zero-install local demo. The default therefore writes the coverage artifact plus
> a **deterministic stratified sample** (7 series: both anchors, one per distinct
> M5 `gates_failed` signature, the first γ-joined series, and the first series with
> no M5 record; ties broken lexicographically). Determinism is proven — consecutive
> full runs are byte-identical by SHA-256 — so a regenerable dump beats a stored
> one. Per-series trees for the web are built on demand.

> **⚠ M17 found an unenforced invariant — and it has now been fixed.** M17 went
> looking for the §2.3 `FitProvenance` record every parameter is supposed to carry
> and found **zero hits** across `engine/`, `etl/`, `web/` and every artifact, with
> **no test at all** behind invariant §10.12's `[test]` tag. Worse for **§7**: its
> basin-stability gate ("anchors freezable only if basin-stable; knife-edge fits
> refused") was **unevaluable**, because `m2_fit.py` kept only the best-SSE start and
> `continue`d past failures without recording them, destroying the spread at fit
> time. Recorded honestly in `a6c9f88`; **fixed in the FitProvenance change**
> (`engine/fit_provenance.py` + `engine/test_fit_provenance.py`). M17's role was
> always to *report*, never to derive, and that has not changed: it now publishes
> the record's **address** (artifact + key path + recipe table) for fitting methods
> and still reports `available: false` **with a reason** for methods that fit
> nothing. See "fit provenance" below for the measured basin breakdown.

> **Genuinely remaining / deferred (Stage B).** The coverage artifact names **66
> unexplained field paths** machine-readably, impact-ranked, so Stage B has a
> priority order rather than a vibe: **52 `no_adapter`** (M6's 119 are now closed;
> the leader is M3's 35, then M7's 10 and M9's 4) and **14 `not_declared`**. Stated plainly rather than quietly excluded:
> most of those 16 are an artifact of the *published* leaf definition — a block that
> is `null` on a compact record contributes its own parent path as a leaf, so
> `gamma_global`, `descriptive_regime` and `curve_features.features` appear as
> unexplained paths *alongside* their fully-explained children. They are counted
> because the denominator rule counts them, not because the value is untraced.
> **Ten** **named narrowings** ship in the artifact so a user meets them in the
> data, not by reading source — the five below, plus five added by Stage B's M6
> work: the join-outcome fields being M7's rather than M6's, the join mirroring
> M7's assay bleed (now retired — the bleed is FIXED by `join-policy-1.0`, and the
> record is kept as the fixed-defect history), the two-author fields covered
> conditionally, the surface-γ edge refused, and the compact-schema orphan nodes
> (also now FIXED, by the spine-emission gate; the record publishes what the gate
> removes so the fix is auditable rather than invisible). Every integer inside a
> narrowing sits in a `measured` block carrying the `population` and `definition`
> that produced it, pinned by a standing test — a rule adopted after review found
> four published figures that did not reproduce.
> **(1)** the conformal **interval is not constructed** —
> M17 attaches the *marginal* quantile verbatim, because building `t50 ± q` is
> arithmetic on engine values and selecting the Mondrian stratum needs an
> assay-endpoint classification M17 is not licensed to make; **(2)** `m2_fit.py:152`
> collapses **three** distinct causes (malformed arrays / fewer than 3 points /
> empty candidate bank) into **one** string — the cause exists only in that
> function's local scope and never reaches an artifact, so **no adapter can
> disambiguate it** and Stage B must widen the upstream record; **(3)** M15 claim
> confidence is **corpus-scoped only** — its ledger holds fixture-document claims
> with no `series_id`, and joining them to the kinetics corpus would be a fabricated
> link; **(4)** `fit_provenance` is **addressed, not derived** — M17 gives the record's artifact + key path and never copies its values, and a method that fits nothing reports `available: false` with a reason; **(5)** `structure_features.json`
> (17 MB) is deliberately **not read**. Also deferred: M17 is read-only over
> `data/processed` and Stage A edits **no** file in `m0..m16` and none in `web/`.

### pooling — deterministic empirical-Bayes partial pooling
`pooling.py` (`pooling-1.0`) — the §4 partial-pooling spine (replicates → protein →
stratum), previously unbuilt: every M4 estimate is per-curve frequentist. §4 also pins
the method — *deterministic* approximate inference, **not** free-running MCMC — and
requires that any down-weight be an explicit, frozen, versioned monotone function of a
named statistic. This module honours both.

Two-level Gaussian model per condition+assay-matched stratum, fit by closed-form
moment estimators (DerSimonian–Laird τ², no iteration, no MCMC):
`θ_pooled,i = wᵢ·θ̂ᵢ + (1−wᵢ)·μ_stratum` with `wᵢ = τ²/(τ² + sᵢ²)`. That weight **is**
the frozen monotone function §4 demands: a precise curve keeps its own value, a noisy
curve in a tight stratum is pulled toward μ. Strata with n<3 cannot estimate τ² → no
pooling (w=1), flagged `stratum_too_small`. Cross-condition pooling is **forbidden**
(Service A discipline): strata are keyed on protein/uniprot + construct + assay +
rounded pH/temp.

Where a stratum holds true **replicates**, per-feature estimates are combined by
DerSimonian–Laird random effects — and the replicate t50 **scatter is retained, not
averaged away**: high between-replicate CV is a *stochastic-nucleation signal* and
sets `stochastic_nucleation_flag`.

**Honest limitation:** this is empirical-Bayes/James–Stein shrinkage at the feature
level, **not** a full Bayesian latent-rate ODE hierarchy, and it makes no MCMC draws.

Real corpus (target `t50`, log₁₀ scale): 1,654 curves seen → **403 poolable point
estimates** (**789 excluded as censored** — a ≥ bound is not a poolable point) across
154 strata, of which **34 strata** are large enough to pool; **257 curves are
shrunk**, median shrinkage weight **0.886** (most curves largely keep their own
value) but a mean of 0.607 (a minority are pulled hard). **60 replicate groups**, of
which **23 carry a stochastic-nucleation flag**.

Run:
```
python engine/pooling.py                 # -> data/processed/pooling.json
python engine/pooling.py --target t50    # log10 t50 (default)
```

### reality_check — the blind external falsification test
`reality_check.py` (`reality-check-1.0`) — the §6C / decision-row-C1 test that
de-circularizes the engine: run blind on the handful of proteins the field considers
mechanistically **resolved**, and check whether PRISE's honest outputs **contradict**
the consensus.

**What this is not** matters more than what it is. It is a **non-contradiction test on
a handful of fixed points** — a thermometer against known freezing and boiling points,
not a statistical calibration. n is deliberately tiny. A pass is **not** a
"validated" stamp. And because PRISE honestly refuses to name a single mechanism
corpus-wide (agitation/seeding unrecorded), the test runs against PRISE's *honest*
outputs — descriptive regime, γ, and the mechanistic **equivalence class** — asking
only whether they contradict consensus, never whether PRISE named the "right"
mechanism.

Three checks per fixed point: **regime** (a gradual_non_cooperative call on a
field-resolved nucleated amyloid is a contradiction), **γ** (where a strong consensus
exists, is PRISE's γ inside a tolerance band), and **mechanism class** (PRISE passes
unless its evidence *positively excludes* the consensus mechanism). These proteins are
reserved **solely** for this role and never used for anchoring, calibration or
demonstrators; `leakage_check()` verifies their absence from the Service-C / anchor
calibration sets. The module only reads artifacts and performs no inference.

Result: **5 fixed points registered, 5 evaluated, 5 CONSISTENT, 0 contradictions, 0
insufficient-data** (Aβ42, Aβ40, α-synuclein, insulin, β2-microglobulin). Read this
as "the thermometer did not disagree with five fixed points", nothing stronger.

Run:
```
python engine/reality_check.py           # -> data/processed/reality_check.json
```

## Tests
**There is no pytest here.** PRISE depends on stdlib + numpy/scipy only, so every
test file is a **self-running stdlib script**: run it directly and read its exit code
(`0` = pass). There is no collector and no `pytest engine/` — to run the whole suite,
run every file, e.g.:
```
for f in engine/test_*.py; do python "$f" || echo "FAILED: $f"; done   # bash
Get-ChildItem engine/test_*.py | % { python $_.FullName }              # PowerShell
```
> **These counts are MEASURED, not hand-maintained.** Regenerate with
> `python engine/run_all_tests.py --roster`, which runs every suite and reports
> the count each one actually printed. The roster below drifted twice while it
> was hand-kept (`test_m10` sat at 25 after a commit raised it to 30;
> `test_fit_provenance` sat at 14 after a control was added) — a self-reported
> number diverging from reality is the same class of defect as the `[test]` tag
> §10.12 carried while having no test behind it.

All 28 engine test files, with the counts each one printed on the run that produced
this README:
```
python engine/test_calibrated_thresholds.py  # governed apply: version/values/record + M1/M3/M5 consume (11)
python engine/test_m1.py        # M1 triage: synthetic + real smoke (8)
python engine/test_m2.py        # M2 kinetic + dose recovery + routing (12)
python engine/test_fit_provenance.py # §2.3 FitProvenance + §7 basin gate + env fingerprint (19)
python engine/test_artifact_invariance.py # value-invariance WITHIN a build id + governed-change guard (8)
python engine/test_tierb.py     # Tier-B ODE: behaviour + fit + sloppiness (5)
python engine/test_m3.py        # M3 selection + identifiability + blockwise bootstrap + policy (18)
python engine/test_m4.py        # M4 features + definition contract + dual-γ (19)
python engine/test_global_fit.py # shared-rate Knowles/Cohen ODE fit: reaction-order + γ recovery, sloppiness (9)
python engine/test_m5.py        # M5 regimes + mechanistic gates + anomaly FDR (26)
python engine/test_m6.py        # M6 propensity + sequence axis + assay vocabulary + study leakage (39)
python engine/test_m7.py        # M7 assembler + yield ladder + build-id + assay gate (53)
python engine/test_service_c.py # Service C generator/recovery/confusion/calibration (29)
python engine/test_external_forward_model.py # external Smoluchowski + Gillespie: mass conservation, closure agreement AND divergence, mechanism-ID ceiling, intrinsic-noise scaling (14)
python engine/test_conformal.py # conformal coverage/width/one-sided/sets/Mondrian/determinism (22)
python engine/test_m8.py        # M8 ladder/grid/blockers/boundary/validity-ceiling + M11 metadata_explanation (19)
python engine/test_m9.py        # M9 recommender: rules/cost/dependency/structural-refusal (28)
python engine/test_m0.py        # M0 cohort: data_mode routing, comparability gate, confound refusal (11)
python engine/test_m10.py       # M10 cross-modal: protein-unit nulls, LOPO, variance ceiling, discordance (30)
python engine/test_m11_metadata.py   # M11 ontology/completeness/uncertainty/DAG/gain-ranking (14)
python engine/test_m12_information.py # M12 FIM/erank/CRLB/phase profile/Sherman–Morrison + post-selection (24)
python engine/test_m13_boed.py  # M13 BOED: EIG, two gain_basis kinds, Pareto, submodular batch (14)
python engine/test_m14_meta.py  # M14 meta-regression: τ², BLUP, single-study refusal, moderator fallback (17)
python engine/test_m15_litingest.py  # M15 ledger/append-only/provenance/conflicts/CPAD validation (36)
python engine/test_m16_structure.py  # M16 real-3D extractors, native-vs-fibril, LOPO, BH-FDR (37)
python engine/test_m17_explain.py    # M17 no-derivation, dangling refs, mutation, closed vocab, worked example, coverage (76)
python engine/test_pooling.py   # pooling: DL τ², monotone shrinkage weight, replicate meta, small-stratum (22)
python engine/test_reality_check.py  # reality check: fixed points, contradiction rules, leakage check (18)
```
The web app has its own stdlib test harness at `web/test_web.py` (boots the server on
an ephemeral port and exercises the API); it is not part of the engine suite.

## What is built, and what genuinely remains

### Built and shipping
- **Analytical engine M0–M17**, plus `global_fit.py`, `pooling.py`, `conformal.py`,
  `service_c.py`, `calibrated_thresholds.py` and `reality_check.py` — every module
  documented above, each with a self-running test file (27 in `engine/`).
- **The web app**, at `web/` — `server.py`, `index.html`, `static/app.js`,
  `static/style.css`, and its own harness `test_web.py`. It is a **stdlib-only**
  backend: `http.server.ThreadingHTTPServer` plus a small path router, with **no
  FastAPI, no Flask, no uvicorn**, and a vanilla-JS single-page front end with no
  framework and no build step. It loads the precomputed artifacts from
  `data/processed` at startup, caches them in memory, and serves derived results
  plus **single** curves. `python web/server.py` (default port 8000;
  `python web/server.py 8123` or `PRISE_PORT=8123` to override) — zero pip installs.
  It imports engine functions and never modifies them, and if the engine import fails
  (e.g. no SciPy) it **degrades gracefully**: browse and the corpus overview still
  work rather than the page going blank. Beyond the M1–M9 views it exposes the newer
  modules over HTTP — `/api/cohort` (M0), `/api/crossmodal` (M10),
  `/api/metadata-overview` + `/api/metadata/{series}` (M11),
  `/api/information/{series}` (M12), `/api/boed/{series}` (M13),
  `/api/meta/{protein}` (M14), `/api/literature` (M15), `/api/structure/{protein}`
  (M16), `/api/pooling/{protein}`, `/api/conformal` and `/api/reality-check`.
  **Data residency is enforced**: CPAD bulk is attribution-gated, so there is
  deliberately **no** bulk-dump endpoint — derived results and one series at a time
  only, with attribution carried on every response.

> A **FastAPI + React production deployment** is a documented *future* path, not the
> current implementation. Nothing in the shipped tree uses FastAPI. Describing the
> present web app as a FastAPI backend would be false.

### Genuinely remaining (documented within-module deferrals)
- **M3** — BCa intervals, a cross-validated descriptive-vs-mechanistic
  criterion, corpus-scale FDR control. *(The block bootstrap for serial
  correlation is no longer deferred — see `estimation-policy-1.1` below.)*
- **M4** — separate left-censoring algebra (left-only t50 is excluded from γ, not
  modelled) and an errors-in-variables term for concentration uncertainty.
- **M5** — Service-C confusion-matrix equivalence classes, the FDR single-mechanism
  degeneracy threshold, and calibration of the anomaly runs-test p-threshold (still
  first-pass).
- **M6** — the reference scales (`m6-refscale-1.0`) and the n≥20 interval floor
  are first-pass. *(Leave-one-**study**-out leakage control is no longer
  deferred — see `join-policy-1.2` below.)*
- **M7** — `env_manifest` now fingerprints the software stack (python micro,
  numpy, scipy, BLAS vendor+version, OS/machine). What remains is the live
  THREAD-COUNT environment, excluded on purpose: it can reorder BLAS reductions,
  but folding it into the build id would make the same corpus non-comparable with
  itself when rebuilt at a different degree of parallelism. §7's pinned-container
  guarantee therefore still needs a container, not a richer fingerprint.
- **M6/M7 assay axis — DONE** (`join-policy-1.0` applied 2026-08-10, `1.1`
  2026-08-17; found 2026-08-09 while building M17 Stage B's M6 adapters). See `engine/join_policy.py`
  and the "governed join policy" section above.

  *What was wrong, root cause first.* `m6_propensity.py` grouped curves on
  `(uniprot, construct)` — **assay was not in the key** — and labelled the record
  first-wins, so a protein×construct measured by two assays was pooled into one
  median under one arbitrary label. **4 of 133 groups** pooled across assays; the
  sharpest, `P01160 / Wild Type`, shipped as `assay: "CongoRed"` carrying t50
  **44.851 h** — the *ThT* median — while its CongoRed data sat at **586.870 h**,
  13× away. A wrong **number** in a wrong stratum under a wrong label. It is also
  why `k114` reached no M6 record at all: those curves were absorbed into
  ThT-labelled groups. Downstream, `m7_assemble.py:1045`'s uniprot-only fallback
  then attached such records across the assay axis while `_propensity_block` gated
  on construct alone — its comment asserting the triple key guaranteed assay
  equality was false on that path, and `want_assay` was accepted and never read.
  Measured: **96** full-schema fallbacks, **14** to a different assay, **9**
  published as clean joins (`available: true, construct_match: true`), and **7**
  carrying a cross-assay `sequence_axis` whose licensing had been computed for the
  wrong endpoint (**24 of 28** licensing decisions in those blocks were wrong — the
  licensing *flips*, it does not merely over-license).

  *The fix, and why both halves were needed.* The M6 re-key alone still left one
  contaminated series; the M7 gate alone would have converted 8 legitimate joins
  into refusals. Both: **0 contaminated, 8 correctly joined.** M6 now emits **164**
  records (was 160), `k114` exists as an assay, and the gate refuses cross-assay
  propensity *and* cross-assay `sequence_axis` with explicit reasons. `k114`'s
  endpoint class was left `unknown` by 1.0 and **corrected to `amyloid` in 1.1**
  with its citation (see the join_policy section) — those 9 curves now license
  PASTA + Waltz rather than nothing. Measured after: **0** cross-assay propensity and **0** cross-assay sequence axes
  corpus-wide; `n_propensity_attached` **unchanged at 959** — the contamination was
  removed without costing a single attachment. Only `CPAD-TK-1061` remains refused
  (IAPP turbidity has no usable t50, so no turbidity record can exist).

  *`engine_build_id` `e660cb452b41` → `29bb01bedd97` (1.0) → `3e5fa43cd1a8` (1.1).*
  It would **not** have moved
  on its own — all 14 prior components are version strings or ETL digests, none a
  function of the join code — so the fix would have shipped the same id with
  different numbers and `comparable()` would have called two different corpora
  comparable. `join_policy_version` is a 15th component added for exactly that,
  with `m6_anchor_version` bumped `1.0 → 1.1` because the re-key changes the
  composition of every anchoring stratum.
- **M9** — the full EIG/optimal-design computation. *(M13 now computes real
  D-optimality EIG for the quantitative design variables; M9 itself remains the
  Service-C-distilled rule table and does not pretend otherwise.)*
- **M12** — the per-parameter CRLB remains conditional on the selected model,
  which is inherent rather than deferred (see the DONE entry below); what is
  genuinely outstanding is a full model-averaged FIM SPECTRUM, not just the
  collapsed verdict.
- **M12 post-selection — DONE** (2026-08-18). The FIM, spectrum and per-parameter
  CRLB were computed on the **AICc winner** and reported without saying how much
  that choice mattered. Measured on M3's own selection posterior, that choice
  matters a great deal: the winner takes a median of only **0.770** of bootstrap
  resamples, **25.1%** of curves are flagged multimodal, and the median runner-up
  holds **0.247**.

  M12 now reads `m3_sample.jsonl` and re-asks the collapsed **richness verdict**
  of every model the bootstrap selects with weight ≥ 0.05, reporting the
  posterior-weighted value, the **spread** across candidates, and the weight of
  the model it conditioned on. Corpus-wide:

  | | |
  |---|---|
  | median weight of the conditioning model | **0.767** |
  | **conditioned on a *minority* model (< 0.5)** | **253 curves — 20.7%** |
  | richness spread across candidates | median 0.054 · p90 0.144 · max 0.438 |
  | verdicts robust to selection | **63.6%** |

  One curve in five had its information verdict computed on a model the bootstrap
  rejects more often than it picks. The spread is published beside the mean
  deliberately — a mean alone would hide whether the verdict depends on which
  model won — and "robust" is a **conjunction** (tight spread AND a real
  majority), because a confident winner is worthless if the runner-up disagrees,
  and a tight spread is worthless if the winner was nearly arbitrary.

  **Per-parameter CRLB is NOT averaged and stays conditional**, which is not a
  shortcut: a parameter of one model is not a parameter of another, so averaging
  their standard errors would be arithmetic over incommensurable quantities. Only
  the collapsed scalar is common to all candidates, so only it is averaged. A
  test pins that the averaged block contains nothing else.

- **M14** — a full latent-ODE-rate MCMC hierarchy; the shipped estimator is
  deterministic closed-form meta-regression.
- **M15** — the front-end *seams* now ship (capability-detected + injectable, with
  adapters for PyMuPDF/pdfminer/pytesseract/spaCy), but no backing library is
  installed here and no PDF corpus is bundled, so the deterministic core still runs
  on fixture documents and **real-document extraction accuracy remains unmeasured**.
  Figure digitization stays injection-only by design (see M15 above). What genuinely
  remains is a validated run against real PDFs, not the wiring.
- **M17 (Stage B)** — **M6 is DONE** (`m6_propensity` + `m6_sequence_axis`,
  116/116 paths; coverage 45.6% → **80.1%** of paths and 51.0% → **80.9%** of
  instances). Remaining adapters, ranked by the measured gap: **M3 (35 paths)**,
  then M7 (13), M9 (4), plus M10/M12/M13/M14. Also: widening `m2_fit.py:152` so its
  three collapsed failure causes become distinguishable; having `conformal.py` emit
  the per-series Mondrian stratum + interval so M17 can read rather than derive it;
  and on-demand per-series trees in `web/`.
- **M17 — the compact-schema orphan-node invariant — DONE** *(found 2026-08-09 in
  review, fixed 2026-08-10)*. `propensity`/`sequence_axis` exist only in M7's **full**
  schema, but the adapters gated on "did an upstream record join", never on whether
  the spine emits the block, so **382 of the 460 compact series received 25,660 M6
  nodes** explaining paths their own result does not carry — **42,113 M17-wide**
  across six adapters (`m4_gamma` 10,177, `m1_triage` 5,060 and others predate Stage
  B; M6 was merely the largest). `m17_explain --series CPAD-TK-1041` would print
  `propensity.cohort.N = 109` for a series the engine publishes no propensity for.
  Coverage was never inflated — the accumulator iterates only the spine record's own
  leaves — so this was a wrong-**graph** defect, not a wrong-number one.

  **The fix is a gate in `build_graph`, deliberately NOT in any adapter.** Two
  approaches were rejected first: having the adapter consult the spine breaks the
  measurement property (the spine is the coverage denominator, and an adapter told
  which fields to explain can no longer be measured on whether it found them), and
  re-deriving the tier would put a second copy of M7's yield ladder inside M17,
  which is the derivation this module exists to refuse. Instead the **builder**
  receives the series' own spine record and drops any node whose declared `explains`
  paths are all absent from it. No adapter can see the spine, nothing is derived
  (the record decides only whether a node *survives*, never what it says), and no
  published percentage moves. Measured after: **0 orphan nodes corpus-wide**;
  42,113 nodes gated across the 460 compact series; the per-adapter table of what
  the gate removes is published so the fix is auditable rather than invisible.
  Edges orphaned by the gate are recorded with their own reason
  (`endpoint_node_gated_not_in_spine`, 1,328) so they never contaminate the
  genuine-dangling signal, which stays at **2,538 across 437 series** — unchanged,
  confirming the gate introduced no new dangling references.
- **§2.3 `FitProvenance` / §10.12 / §7 basin-stability — DONE**, see
  `engine/fit_provenance.py`. What genuinely remains: **(a)** the §7 *pinned
  containerized environment* (BLAS/thread/libm fingerprint) — `env_manifest` is
  still a coarse interpreter tag, so "tolerance-reproducible under the pinned
  reference environment" is not yet assertable. **(b)** ≥2 genuine starts per model
  and **(c)** M3's constant-seed stream are both **DONE** — see the
  `estimation-policy-1.0` entry below.
- **M2 multi-start + M3 bootstrap stream — DONE** (`estimation-policy-1.0`,
  applied 2026-08-18; see `engine/bootstrap_policy.py`). Two defects that made
  published numbers *unattributable* rather than merely imprecise.

  **(1) M3 gave every curve the same random stream.** `analyze_curve` took
  `seed=0` and the driver never overrode it, so all curves drew wild-bootstrap
  resamples from `default_rng(0)`. Equal length + equal seed = **byte-identical
  resample index matrices**, and measured over the corpus **1,627 of the 1,654**
  curves carrying an x grid — **98.4%** — share their length with another (n=5: 88
  curves, n=11: 83, n=7: 81). Determinism was preserved; **independence was not**,
  and `pooling.py` and `m14_meta.py` both aggregate per-curve intervals as though
  their Monte-Carlo error were independent. The fix is a deterministic per-series
  derivation — BLAKE2b over the series id, *not* the builtin `hash()`, which is
  salted per process and would have traded a shared-stream defect for a
  non-determinism one. Verified in the artifact: **1,240 records, 1,240 distinct
  seeds, every one equal to `series_seed(series_id)`, none zero.**

  **(2) 9 of 13 registry models got a single optimisation start.** A single-start
  fit cannot be distinguished from a local optimum, so §7's basin gate was
  `not_evaluable` for most of the registry. `_spread_starts` now tops every model
  up to `n_starts` deterministic in-bounds probes, appended *after* the name-keyed
  starts so any model that already had enough is bit-for-bit unchanged. Result:
  `not_evaluable` **46.6% → 0.02% (2 fits)**, and the knife-edge rate rose
  2.70% → 3.87% because the gate can now see rival basins it previously could not.

  *Rebuild.* Both move fitted values, so this is a governed change:
  `engine_build_id` `3e5fa43cd1a8` → **`6baf9f251541`** via a new
  `estimation_policy_version` axis (the 16th) — without it a change to how the
  corpus is *fitted* would have shipped under an unchanged id. Full chain rebuilt
  M2→M3→M4→M5→M6→M7→M8→M9→pooling/M14→M17. M3 single-threaded measured ~35 min, so
  `m3_select` gained `--shards`/`--shard`/`--skip`; sharding is sound because
  curves are fitted independently and the seed is now per-curve, and it was
  **verified byte-identical to a single-process run (43/43)** before use. The
  M17 coverage figures, graph census, yield ladder and propensity join summary are
  all **unchanged** by this rebuild — what moved is the fits, their basin verdicts
  and the bootstrap intervals.
- **M3 serial correlation — DONE** (`estimation-policy-1.1`, applied 2026-08-18).
  The wild bootstrap resampled residuals **pointwise**, which is valid for
  heteroskedastic but **serially independent** errors. Kinetic residuals are not
  serially independent: a model that slightly misfits a sigmoid's shape leaves
  consecutive residuals sharing a sign. Measured over the 1,504 curves with a
  best fit and a usable grid — **median lag-1 autocorrelation +0.182, 48.2% above
  0.2, 24.9% above 0.5, median Durbin–Watson 1.483 with 30.3% below 1.0**. Half
  the corpus. Resampling those as independent destroys the dependence the data
  has and yields intervals that are **too narrow, worst on the most
  autocorrelated curves** — i.e. exactly the fits least worth trusting.

  Replaced by **Shao's blockwise wild bootstrap**: one Mammen draw per block of
  `ceil(n**(1/3))` consecutive residuals. Verified the sampler keeps the Mammen
  marginal (E=0, Var=1) while making the weight constant within a block
  (correlation 1.0) and independent across blocks (~0), and that `block=1`
  reproduces the pointwise sampler bit-for-bit. Block length is a function of
  **n alone, never of the estimated autocorrelation** — a data-chosen block would
  make the interval depend on a tuning parameter selected from the same residuals,
  which is the post-selection trap this module fights elsewhere.

  **The effect is a correction, not an inflation, and that is the evidence it is
  real.** Comparing the two builds curve by curve against each curve's own
  measured ρ:

  | residual lag-1 ρ | n | median interval width ratio |
  |---|---|---|
  | ρ ≤ 0 | 414 | **0.759** (correctly narrower) |
  | 0.0 – 0.2 | 167 | 0.929 |
  | 0.2 – 0.5 | 267 | 1.069 |
  | ρ > 0.5 | 318 | **1.342** (wider) |

  Rank correlation between ρ and log width ratio **+0.617 (p ≈ 1.7e-123)**. The
  interval moves in the direction each curve's own dependence structure dictates —
  widening positively autocorrelated curves and tightening anti-correlated ones,
  where the pointwise bootstrap had been *over*-stating uncertainty. A uniform
  widening would have been weaker evidence; the median overall ratio is 0.994
  because the correction redistributes rather than inflates.

  Recipe `m3/wild-bootstrap-mammen-1.0` → **`m3/blockwise-wild-bootstrap-mammen-2.0`**
  (the old id is retained in the table so records written under it still resolve),
  `M3_VERSION` → `m3-select-1.1`, and each record carries the `block_length` its
  own interval was built with. `engine_build_id` `6baf9f251541` → **`bcceb97a46a1`**.
- **M6 anchor leakage — DONE** (`join-policy-1.2`, applied 2026-08-18). The
  intrinsic **corpus anchor** excluded only the group being scored, so its pool
  was still full of other groups from the **same study**, sharing lab, protocol
  and batch effects. §8 asks the anchor to be leakage-free; it was not.

  This README previously argued the residual leakage was small *because "proteins
  rarely co-occur across studies"*. That premise is true — only **36 of 164**
  groups draw on more than one study — but it is **not the mechanism**. The leak
  runs the other way: one study contributing **several** groups to one stratum.
  Measured that way, **151 of 164 groups (92.1%)** shared at least one study with
  another group in their own anchoring stratum.

  Now **leave-one-STUDY-out**: a group's pool excludes every group sharing any of
  its `source_study.pmid` values. Measured cost — **10 of 129 scored groups
  (7.8%)** lose their corpus anchor and fall back to the versioned reference
  scale, **all 10 in ThS**, whose median pool drops 9 → 4 because that stratum is
  essentially one study's measurements. ThT, the bulk, drops only 110 → 107 and
  keeps its corpus anchor. So the cost lands precisely where the anchor was never
  independent evidence, which is the point rather than a side effect:
  `intrinsic_corpus_anchored` 121 → **111**, `intrinsic_reference_scale` 8 → **18**.

  The **cohort** view is deliberately untouched: it is an explicit "rank within
  the cohort under analysis", not a leakage-free reference, and `cohort_rank`
  already self-excludes. `ANCHOR_VERSION` → `m6-anchor-1.2` because the
  composition of every stratum changed, which is exactly what that axis describes.
  `engine_build_id` `bcceb97a46a1` → **`271f46f057e6`**.
- **Service C** — the artifact-realism distributional check, synthetic→real
  transfer, and M6 null-disagreement calibration. *(The external forward-model
  generator is no longer deferred — it ships as
  `engine/external_forward_model.py`, a size-resolved Smoluchowski master
  equation WITH an exact Gillespie stochastic arm; see the section above for what
  they measured.)* *(The blind real-literature reality check is no
  longer deferred — it ships as `engine/reality_check.py`.)*

> **DONE (no longer deferred):** *applying* Service C's calibrated thresholds back
> into M1/M3/M5 — completed as the governed, signed/dated/versioned apply step
> (`engine/calibrated_thresholds.py`, thresholds-1.0, 2026-07-01). M1/M3/M5 now run
> the calibrated cutoffs (0.30→0.45 · 1000→107 · 3.0→5.7 · 0.80→0.97; α unchanged),
> `engine_build_id` bumped `bc442463dfdf`→`e660cb452b41`, previous values retained
> + reversible.
>
> **Also no longer deferred:** the shared-rate Knowles/Cohen **ODE global fit**
> (`global_fit.py`); M6's **null_disagreement_rate** (M10 phase P3); the **blind
> reality check** (`reality_check.py`); and the **web app** itself.
