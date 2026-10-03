# PRISE — Reports

Point-in-time reports on the PRISE product. Each report is generated **after auditing the
running product** at a specific `engine_build_id`, so it describes **what was actually
functional** at that moment — not aspirational plan claims. Where the product differs from
the design, the report describes what is real and flags the gap.

> PRISE is under active development and **this is not the final product** — expect more
> reports to be added here as the engine and web app evolve.

## Index

| Date | Report | Build | Scope |
|------|--------|-------|-------|
| 2026-08-05 | [Full product report (M0–M17)](2026-08-05_full-product-report.html) | `e660cb452b41` | Engine + web audit through **M17**, with an unusual shape: **the new module audited the engine and found the engine wanting.** Covers M17 (explanation/provenance evidence graph — derives nothing, ~50% coverage over a *published* 329-path denominator, decomposed into two populations); the **`FitProvenance` finding** (§2.3 specified, §10.12 tagged `[test]`-enforced, emitted nowhere and asserted by no test — leaving §7's basin-stability gate *unevaluable*); the fix and its measured basin census (**2.70% knife-edge** corpus-wide, **11/16** for the shared-rate ODE fits); why `engine_build_id` deliberately does **not** bump; **574/574** tests; and a **corrections section** listing four numbers the 2026-07-21 report published wrongly |
| 2026-07-21 | [Full product report (M0–M16)](2026-07-21_full-product-report.html) | `e660cb452b41` | Up-to-date engine + web audit through **M16**: what PRISE is, the mathematics of every module (M0–M16 + Service C, conformal, pooling, global-fit, reality-check), **live** corpus results re-read from the product's own artifacts, the 477/477 test-suite run, and an honest gaps section (the M16 engine↔web 3D-schema desync, the M15 stubs, stale in-repo docs, and documented deferrals) |
| 2026-07-02 | [Full product report](2026-07-02_full-product-report.html) | `e660cb452b41` | Complete engine + web audit: what PRISE is, the full mathematics of every module (M0–M10, Service C, conformal, pooling, global-fit, reality-check), real corpus results, constraints, gaps vs the plan, and future improvements |

## Conventions

- **Filename:** `YYYY-MM-DD_<slug>.html`
- Each report names the **`engine_build_id`** it was generated against (the comparability
  gate — results from different build ids are not directly comparable).
- Reports are **self-contained HTML** (open locally in a browser; equations render via
  MathJax from CDN, so an internet connection is needed for the math to typeset).
- Reports are **descriptive**, not authoritative source: the code + tests + `PRISE_DESIGN.md`
  remain the source of truth.
