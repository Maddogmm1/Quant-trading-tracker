# Phase 5 Conclusion — Overnight vs. Intraday Return Decomposition

**Status: CLOSED.** This document concludes the Phase 5 overnight-gap research
branch. The hypothesis is judged **not supported** as a stable, generalisable
effect. No further work is authorised on this specific hypothesis without a new
pre-registered spec.

---

## 1. Pre-registered objective

Per `PHASE5_OVERNIGHT_GAP_SPECIFICATION.md`, the Phase 5 research question was
whether the decomposition

```
overnight_i(t) = ln(open_i(t) / close_i(t-1))
intraday_i(t)  = ln(close_i(t) / open_i(t))
```

shows a persistent, non-zero overnight return component for the equal-weighted
US large-cap proxy — distinguishable from zero and from the intraday component
over the same days — sufficient to clear a pre-declared 1.0 percentage-point
annualised economic-magnitude threshold (spec §8), before any consideration of
whether it would survive realistic Trading 212 execution constraints (spec §9).

This is a **Tier 0** result only (spec §6): a block-bootstrap distributional
test on the aggregate proxy series, not a per-security ranking model. Tier 1/2
(linear model, tree ensembles) were never reached — Tier 0 is the gate that
determines whether they're worth building at all, and it did not clear.

**Walk-forward split** (chronological, daily granularity, embargo = 5 trading
days at each boundary, via `src/ml/walk_forward.py::build_primary_split()`
unchanged from Phase 4): **train** 1,353 dates (2015-01-02 to 2020-05-18, 5
purged), **validation** 335 dates (2020-05-27 to 2021-09-22, 5 purged),
**locked test** 566 dates — **never accessed in this branch** (§5).

## 2. Tier 0 result: train, validation, and combined

All three block-bootstrap runs from `PHASE5_TIER0_EXPLORATORY_REPORT.json`
(circular moving-block bootstrap, 5,000 resamples, 95% CI, block length chosen
per-series from the autocorrelation structure):

| Window | n days | Overnight point estimate (daily) | Annualised | CI excludes zero (vs-zero / paired-vs-intraday) | Classification |
|---|---|---|---|---|---|
| Train | 1,353 | +0.0000171 | **+0.43pp** | No / No | `failure` |
| Validation | 335 | +0.00126 | **+31.76pp** | Yes / Yes | `genuine_effect_candidate` |
| Train + validation combined | 1,688 | +0.000264 | **+6.65pp** | No / No | `failure` |

Two of three windows fail outright. The one window that clears both the
statistical (CI excludes zero) and economic-magnitude (>1.0pp annualised)
bars is validation alone — and per the test module's own classification
language, `genuine_effect_candidate` is explicitly not a final result; spec
§8's robustness checks were the next required step, not a stopping point.
Applying the first and most basic of those checks — does the result survive
being folded back into the full sample it was carved from — it does not: the
combined estimate collapses to a statistically insignificant +6.65pp.

## 3. Regime diagnostic: why validation alone lit up

`PHASE5_VALIDATION_WINDOW_DIAGNOSTIC.json` breaks the validation window down
by calendar year, tests sensitivity to excluding the most extreme days, and
identifies the securities behind those extreme days.

**The validation window is not an arbitrary slice of history.** The
chronological split happens to place it at 2020-05-27 through 2021-09-22 —
the COVID-19 reopening/recovery period: vaccine trial readouts, unprecedented
fiscal/monetary stimulus, and violent sector rotation out of stay-at-home
winners and into travel/leisure/energy names.

**Per-year breakdown confirms rapid decay, not a stable level:**

| Year | n days | Annualised overnight estimate |
|---|---|---|
| 2020 | 153 | +46.65pp |
| 2021 | 182 | +19.25pp |

**Removal-sensitivity is large but the effect doesn't vanish from single
days** — excluding the top 1/3/5/10/20 most extreme days by |overnight
return| brings the annualised estimate down to a 23–28pp range, still well
above the 1.0pp threshold. So this is not a single-day fluke; it is a
regime-wide effect confined entirely to one historically extraordinary
16-month window, decaying by more than half from 2020 to 2021 within that
same window.

**The top extreme days are dominated by a recurring handful of high-beta,
reopening-sensitive names**, not a broad market-wide move. NCLH, CCL, RCL,
AAL, UAL, and DAL (cruise lines and airlines) appear as top-5 contributors on
the large majority of the 20 most extreme days. The single largest day,
2020-11-09 — the aggregate proxy moved +5.47% overnight — coincides with the
Pfizer/BioNTech Phase 3 vaccine trial results announcement (CCL +29.1%, RCL
+22.4%, AAL +22.3% overnight) alongside BIIB (−35.4%), plausibly the
unrelated Biogen aducanumab FDA advisory panel outcome landing the same
week. Two distinct, one-off news events driving a single extreme day is the
opposite of evidence for a repeatable, tradeable overnight premium.

## 4. Data-quality cross-check: not an artifact

Before attributing the validation-window effect to real (if regime-specific)
market behaviour rather than a data bug, the top-20 extreme days' top
contributing securities' price rows were checked against `validate_ohlc()`
(`src/validation/checks.py`), re-run fresh as part of this diagnostic
(3,681 rows flagged `suspicious` database-wide — a full-database scan, not
scoped to this window). **Zero** of the extreme-day contributors' relevant
price rows were flagged `suspicious`. The effect is not explained by known
bad-OHLC rows.

One documentation gap surfaced along the way, tracked separately (§6 below):
these rows carry `price_data_quality = "derived"`, a value not listed in
`schema.sql`'s own column comment (`'ok'|'flagged'|'suspicious'`).

## 5. Locked test set: never accessed

The locked test period (566 dates) was not read, queried, or evaluated at
any point in this branch. `run_phase5_tier0_test.py` and
`run_phase5_validation_window_diagnostic.py` both structurally pass only
`train_dates + validation_dates` to `overnight_targets.proxy_series_for_dates()`
— `test_dates` is resolved only for its count. **No confirmatory experiment
was registered or run for this hypothesis; gate (d) of spec §12 was never
reached, because the train/validation result did not warrant advancing to
it.**

## 6. Conclusion

**The overnight/intraday decomposition hypothesis is not supported as a
stable, tradeable structural effect for US large-caps.** Train alone (5.4
years, the large majority of the pre-registered sample) shows no
distinguishable overnight effect. The one window that did show a large
effect is fully explained by a specific, well-documented, non-repeating
16-month historical regime — the COVID reopening trade — concentrated in a
recurring small cluster of high-beta travel/energy names, and decaying
sharply even within that window. It does not survive being combined with
the rest of the sample.

This is a legitimate negative research result, consistent with the standing
instruction governing this research program (see `PHASE4_CONCLUSION_3M.md`
§10): the objective is to determine *whether* genuine, exploitable structure
exists, not to manufacture a positive backtest. The gate did exactly what it
was designed to do — it caught a non-generalisable regime anomaly at Tier 0,
before Tier 1/2 model-building effort or any locked-test evaluation was
spent on it.

## 7. Reproducibility record

| Artifact | Contents |
|---|---|
| `PHASE5_TIER0_EXPLORATORY_REPORT.json` | Split boundaries, train/validation/combined block-bootstrap results |
| `PHASE5_VALIDATION_WINDOW_DIAGNOSTIC.json` | Date-range breakdown, per-year decomposition, removal-sensitivity, extreme-day contributor detail, data-quality cross-check |
| `src/ml/overnight_targets.py` | Target/decomposition module (KEEP for any future Phase 5-adjacent hypothesis) |
| `src/ml/overnight_significance.py` | Tier 0 block-bootstrap test module (KEEP) |
| `tests/test_overnight_targets_leakage.py`, `tests/test_overnight_significance.py` | 16/16 passing leakage and statistical-correctness tests |

No `ml_experiments` row was written for this branch — Tier 0 is an
exploratory statistical test on an aggregate series, not a per-security
model fit, and nothing here was logged as `is_confirmatory=1` or
`touched_locked_test_set=1`.

## 8. What this document does not do

This document does not evaluate a different split, a different universe
definition, a shorter/longer horizon than one trading session, or a
sector-neutral version of the proxy (e.g. excluding travel/energy names to
test whether a residual effect exists in the rest of the market) — any of
those would be a new, separately pre-registered hypothesis, not a
continuation of this one. The v2-corrected `run_phase5_open_price_data_check.py`
and `run_phase5_sample_size_report.py` were delivered but never re-run
against the real database after their bug fixes; that follow-up is now moot
for this branch's conclusion (the Tier 0 result stands on its own, computed
directly from per-date proxy series rather than from those summary
diagnostics) and is not pursued further here. Per instruction, work on this
hypothesis stops here.
