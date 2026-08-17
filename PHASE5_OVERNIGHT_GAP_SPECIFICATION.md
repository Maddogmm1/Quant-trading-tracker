# Phase 5 Specification: Overnight vs. Intraday Return Decomposition

Status: **CLOSED — see `PHASE5_CONCLUSION.md`.** The Tier 0 block-bootstrap
test (§6) was implemented and run on the train/validation split: train alone
and train+validation combined both failed to distinguish the overnight
component from zero; validation alone showed a large effect that a follow-up
diagnostic traced entirely to the COVID reopening regime (2020-05 to 2021-09),
concentrated in a recurring cluster of high-beta travel/energy names, and not
present once folded back into the full sample. The hypothesis is judged not
supported. The locked test set (566 dates) was never accessed — gate (d) of
§12 was never reached, because the Tier 0 result did not warrant advancing to
it. No further work on this specific hypothesis is authorised without a new
pre-registered spec. This document is retained for its methodology (target
definition, sample-size diagnostics, Tier 0 test design) and as the record of
what was pre-registered before the result was known.

Depends on: Phase 1–3 (data foundation, corporate-action handling,
point-in-time backtesting engine — all complete), Phase 4 (predictive-
modelling research on cross-sectional multi-month returns — complete,
clean negative result, see `PHASE4_CONCLUSION_3M.md`).

## Guiding principles

Identical to Phase 4's (`PHASE4_SPECIFICATION.md` preamble), restated
because they bind here just as much: no look-ahead bias anywhere in the
pipeline; no survivorship masquerading as predictive performance; no
retrospective threshold/hyperparameter tuning; reuse existing machinery
rather than reinventing it; no silent data cleaning; no fabricated data;
full reproducibility; and — the one that matters most given Phase 4's
own outcome — **optimise for determining whether genuine, exploitable
structure exists, not for finding a profitable backtest.** A clean
negative here is exactly as valid a Phase 5 outcome as Phase 4's was.

---

## 0. Codebase reuse assessment

Requested before writing the rest of this spec. Everything below was
read directly from the current repository (`src/`, `tests/`,
`config/config.yaml`), not assumed.

### 0.1 KEEP — reused unchanged

- **`src/database/schema.sql`** — the `prices` table already has an
  `open` column (`open REAL`), populated by the existing ingestion path
  (`src/ingestion/price_sources.py`'s `YFinancePriceSource.fetch()`
  already captures `Open` from yfinance's OHLCV frame — this was never
  Phase 4-specific, it's been there since Phase 1). `ml_experiments` /
  `ml_predictions` are generic enough (JSON config blobs) to describe an
  overnight-return experiment without a schema change — see §0.2 for the
  one caveat.
- **`src/ingestion/adjustments.py`** — `compute_split_adjusted()` and
  `compute_total_return()` already scale `open` by the identical
  `price_factor`/`cum_factor` applied to `close`/`high`/`low` (lines
  85–91 and 182–187 of that file). This is a real, load-bearing finding
  for §2: the split/dividend backward-adjustment math for `open` already
  exists and needs no new code — only validation that it behaves
  correctly at daily granularity (§2.3), which it was never explicitly
  tested for before (Phase 3/4 only ever consumed `close`).
- **`src/backtest/execution.py`** — `next_rebalance_dates()` already
  supports `frequency="daily"`; `next_market_session()`,
  `PointInTimeDataAccess`, and the `Strategy` interface are
  frequency-agnostic and reused unchanged.
- **`src/backtest/universe.py`** (`build_eligible_universe`), **`src/validation/checks.py`**
  (`check_point_in_time_availability`, `apply_universe_filters`,
  `reconstruct_universe`) — reused unchanged for eligibility. See §0.2 for
  the one addition layered on top, not a replacement.
- **`src/backtest/costs.py`**, **`src/backtest/accounting.py`**,
  **`src/backtest/engine.py`**, **`src/backtest/benchmarks.py`** — the
  cost model, portfolio accounting, and backtest orchestration are all
  generic over rebalance frequency and reused exactly as-is. `run_backtest()`
  already accepts any `rebalance_dates` list, daily included.
- **`src/ml/feature_matrix.py`, `src/ml/baselines.py`, `src/ml/trees.py`**
  — not used for the primary (Tier 0) distributional test, but kept
  ready, unmodified, for Tier 1/2 escalation if §6 warrants it (frozen
  grids, inner-CV tuning, ablation ladder — all directly reusable for a
  daily-horizon feature set with no code change).
- **`src/ml/walk_forward.py`** (`build_primary_split`) — the
  chronological-split-with-embargo logic is generic over any date list
  and any `embargo_periods` integer; reused with `embargo_periods` set to
  a small number of trading days rather than months (§4.2).

### 0.2 MODIFY

- **`src/backtest/universe.py` eligibility** — `build_eligible_universe()`
  itself is unchanged, but a **new, additive** point-in-time check for
  same-day `open` availability must sit on top of it (not inside it,
  since STRICT/PERMISSIVE and Phase 4's cross-sectional use of this
  function must not silently start requiring `open`, which they never
  needed). Implemented as a thin wrapper in the new module from §0.3, not
  an edit to `universe.py`.
- **`config/config.yaml`** — a new, additive `phase5:` block, mirroring
  `phase4:`'s structure. Proposed content is inlined at each relevant
  section below (§2, §4, §7, §9); nothing in `phase4:` or earlier blocks
  changes. Not written to the file yet — see the instruction at the top
  of this document.
- **`src/database/migrations.py`** — if the experiment-tracking tables
  need a `target_family` discriminator column (`'cross_sectional_excess_return'`
  vs `'overnight_gap'`) to keep Phase 4 and Phase 5 experiments queryable
  separately without guessing from `target_config_json` contents, that's
  one additive `ALTER TABLE ml_experiments ADD COLUMN target_family TEXT`
  migration, following the exact pattern already used for `securities.cik`.
  Proposed, not applied — flagged for your sign-off at §23(b).

### 0.3 NEW

- **`src/ml/overnight_targets.py`** (name proposed, not created yet) — a
  small, new module analogous to `src/ml/targets.py` but for a
  single-security daily time-series decomposition rather than a
  cross-sectional excess-return ranking. Computes, per security per
  trading day `t` (never a rebalance-period index — genuine calendar
  trading days):
  ```
  overnight_t = ln(open_t / close_{t-1})
  intraday_t  = ln(close_t / open_t)
  ```
  using `total_return` prices exclusively, both endpoints one session
  apart (well inside the existing "ratio-only-within-a-bounded-window"
  rule from Phase 4 §6/§8.4 — ratios one session apart cancel the
  future-dividend restatement constant exactly the same way a 21-session
  window does). Returns `None`, never a fabricated value, when either
  endpoint is unresolvable — same discipline as `targets.py`.
- **A point-in-time same-day-open accessor** — a one-function addition
  (`open_at_or_before` / `open_on_exact_date`), mirroring
  `src/ml/features.py`'s `latest_price_at_or_before()` pattern exactly,
  since every existing point-in-time primitive in this codebase resolves
  `close`, not `open`, and the overnight target needs a same-session
  `open`, not a nearest-available one (an overnight return anchored to a
  stale open would silently become a multi-day return without saying so).
- **`run_phase5_open_price_data_check.py`** and
  **`run_phase5_sample_size_report.py`** — written and now executed
  against the real database (§0.5), matching the existing "diagnostic
  script, run locally, writes its own JSON report" convention
  `run_phase4_sample_size_check.py` already established.
- **A block-bootstrap distributional test module** — the Tier 0 primary
  test (§6) needs a paired significance test respecting the day-level
  correlation structure found in §5, analogous in spirit to how Phase 4's
  `evaluate_predictions()` respects per-date IC aggregation, but this is
  a genuinely new, small piece of code (a one-sample/paired block
  bootstrap on a daily return series), not a reuse of anything Phase 4
  built for cross-sectional ranking.

### 0.4 DELETE from consideration (not applicable to Phase 5)

- **`src/ml/targets.py`'s cross-sectional excess-return target**
  (`y_i(t,h) = r_i(t,h) - B(t,h)`) — Phase 5's primary hypothesis (§1) is
  a distributional claim about a return *component* (overnight vs.
  intraday), not a cross-sectional ranking claim. It is not reused for
  the primary test. It remains directly relevant *if* Phase 5 escalates
  to Tier 2 (§6) and a ranking-based strategy is eventually built on top
  of an overnight signal — flagged there, not built now.
- **`src/universe/identity_resolution.py`, `membership_sources.py`,
  `stage_universe.py`** — Phase 5 needs no new ingestion or membership
  work; it reuses the already-validated, already-populated database
  as-is. Untouched.
- **Any ML model class beyond linear/tree** — same hard exclusion as
  Phase 4 §11 (LSTM/TCN/Transformer/any neural network), reaffirmed here
  independently in §6, not merely inherited.

### 0.5 Diagnostics: now run against the real database

This session's remote-shell bridge to your machine failed to start when
this document was first drafted, so the two scripts below were written
but not executed (§2.3/§2.4/§3.4 originally said "pending"). You ran both
locally and returned the JSON output; the real findings are folded into
§2.3, §2.4, and §3.4 below, and both JSON reports are now saved in your
repo root (`PHASE5_OPEN_PRICE_DATA_CHECK.json`,
`PHASE5_SAMPLE_SIZE_REPORT.json`) alongside `run_phase5_open_price_data_check.py`
and `run_phase5_sample_size_report.py` themselves, for reproducibility —
same convention as `PHASE4_SAMPLE_SIZE_REPORT.json`.

One honest caveat about the diagnostics' own methodology, surfaced
rather than glossed over: the open-price-on-exact-date measurement in
`run_phase5_open_price_data_check.py` and the autocorrelation-based
effective-N formula in `run_phase5_sample_size_report.py` both have a
real limitation discovered only by looking at their actual output — see
§2.3 and §3.4 respectively. Neither invalidates the headline findings,
but both need a follow-up before being treated as final numbers.

---

## 1. Hypothesis and mechanism

**Testable claim, not a vague theme:**

> For the PERMISSIVE-eligible S&P 500 constituent universe over
> 2015–2023, the overnight return component (previous close to next
> open) contributes a disproportionate, statistically distinguishable-
> from-zero share of total daily return, relative to the intraday
> component (open to close) — and this split is not merely a
> volatility-driven artifact of the two windows having different
> lengths in wall-clock time.

Candidate mechanisms from the literature (Branch & Ma, Kelly &
Clark-Joseph, and others on the overnight/intraday split; academic
consensus is real but the *cause* is still debated) — listed for
context, not adjudicated here, since this project has no way to
distinguish between them with price data alone:

- Differential order-flow timing: institutional accumulation
  disproportionately executes at/near the open (VWAP/TWAP algos,
  overnight risk transfer), retail flow disproportionately executes
  intraday.
- A risk premium for holding through the non-trading window (overnight
  gap risk — earnings, macro news, geopolitical events all cluster in
  non-trading hours).
- Close/open-adjacent flow effects: passive index rebalancing and
  closing-auction volume concentrate price discovery very close to the
  close, with the open absorbing overnight information asymmetrically.

**What this phase can and cannot establish:** it can establish *whether*
this project's own point-in-time US large-cap data shows a
statistically real overnight/intraday split, of what direction and
magnitude, and whether it is exploitable net of realistic UK-ISA
transaction costs. It cannot and does not attempt to adjudicate *why* —
that would require order-flow or auction-volume data this project
doesn't have and isn't proposing to ingest.

## 2. Primary target variable

### 2.1 Definition — ONE primary formulation, not a fishing exercise

```
overnight_i(t) = ln(open_i(t) / close_i(t-1))
intraday_i(t)  = ln(close_i(t) / open_i(t))
```

Both computed on `total_return` prices, `t-1` meaning the immediately
preceding trading session in security `i`'s own price history (not a
calendar day — weekends/holidays are skipped by construction, since both
endpoints are resolved via the existing `date <= as_of_date`
session-lookup primitives, never a literal calendar subtraction).

**Primary metric: `overnight_i(t)` in isolation**, tested against zero
and against `intraday_i(t)` over the *same* trading day (so
`overnight_i(t) + intraday_i(t) = daily_total_return_i(t)` exactly, by
construction of log returns — this additivity is the specific reason log
return is used here rather than simple return, where the two components
don't sum to the daily total and any "share of return" statement would
be an approximation rather than an identity).

### 2.2 Why log-return overnight, not the alternatives

- **Log return over simple return:** simple overnight/intraday returns
  don't compose additively into the daily return (compounding order
  matters), so a claim like "60% of the daily return happens overnight"
  is only a clean, exact statement in log-return space. Given the whole
  point of this phase is a return-decomposition claim, the metric that
  actually decomposes exactly is the only defensible primary choice.
- **Per-security overnight return over the equal-weighted proxy alone:**
  the proxy-level series (mean overnight return across `ELIG(t)`,
  analogous to Phase 4's `B(t,h)`) is used for the primary *significance
  test's* time-series (§5, §6.0) because a single aggregate series is
  what a serial-autocorrelation/block-bootstrap test needs to run
  against. But the underlying observations feeding it are per-security —
  this is not "one number for the whole market," it's an average across
  a genuine daily cross-section, exactly mirroring how Phase 4's
  `B(t,h)` was the mean of individual `r_i(t,h)` values, not a separately
  sourced index return.
- **Excluded as V1/primary:** overnight *volatility* (as opposed to
  mean return), gap-fade/gap-continuation conditional strategies,
  day-of-week or turn-of-month interaction effects, and anything
  cross-sectionally ranked. All of these are legitimate secondary/
  diagnostic questions (§6 Tier 1 mentions a few explicitly) but are not
  the primary pre-registered claim — introducing them as co-equal
  primaries is exactly the "five-target fishing exercise" this section
  exists to prevent.

### 2.3 Data requirements: does the database have reliable point-in-time daily OPEN prices?

**Partial answer available now, from code inspection (§0.1):**
`open` is captured by ingestion, schema-supported, and mathematically
adjusted by the existing split/dividend backward-adjustment code using
the identical per-row scalar applied to `close`. This is a materially
better starting position than "we'd need to re-ingest from scratch."

**Real findings, from `PHASE5_OPEN_PRICE_DATA_CHECK.json`:**

1. **Row-level coverage is complete.** Of 2,372,308 price rows (matching
   Phase 3's full-universe row count exactly, confirming no partial
   re-ingest happened), `open` is populated (non-NULL, positive) on
   **100.0%** of rows, identical to `close`'s own 100.0%, across all
   three `adj_type` values. `open` is not a second-class citizen in this
   data — this is a materially better result than the "needs
   re-ingestion" scenario this section was written to guard against.
2. **687 rows** have adjusted `open` falling outside `[low, high]` on the
   same row — identical count across `raw`/`split_adjusted`/
   `total_return`, consistent with the adjustment math (§0.1) preserving
   rather than introducing the violation (a shared per-row scalar can't
   change intra-row ordering, so a violation present in `raw` propagates
   unchanged). **Reconciled, via the corrected script (`script_version: 2`,
   `PHASE5_OPEN_PRICE_DATA_CHECK.json`): all 687 rows were already
   flagged `suspicious`** (0 not previously flagged), confirmed a subset
   of the existing, already-known 1,227-row raw-bad-OHLC set from
   `BAD_OHLC_INVESTIGATION.md` — **CONFIRMED, no new problem.**
3. **The original eligible-pairs-with-open-availability figure (33.09%
   missing) was a measurement artifact in the diagnostic script itself,
   confirmed by the corrected re-run.** The original script queried
   `prices` for an exact match on `next_rebalance_dates("monthly")`'s
   literal calendar date strings (e.g. `"2015-02-01"`) rather than
   resolving each to its nearest actual trading session first — many
   calendar month-starts are weekends or holidays, on which **no**
   security has *any* price row, inflating the miss-rate for reasons
   having nothing to do with `open` specifically. The corrected script
   (`script_version: 2`, resolving each nominal date to its real trading
   session before querying) found **0 of 108 sampled dates with no
   resolvable session, and only 4 of 46,277 eligible (security, date)
   pairs missing `open` on the resolved session — 0.01%.** Confirms
   point 1's conclusion directly rather than relying on the earlier
   inference: open-price coverage on real trading sessions is
   essentially complete.
4. **Corporate-action spot-check (15 sampled ex-dates, see §2.4)** — raw
   `open` on the ex-date session looks like a plausible, ordinary market
   print in every sampled case (moves of a few tenths of a percent to
   ~1.5%, in both directions relative to the prior close — consistent
   with everyday volatility swamping most of these dividends, which
   average well under 1% of price), not a back-adjustment artifact.

**Conclusion: open-price coverage is adequate. §12 gate (a) is fully
closed** — both follow-ups from the first pass (the 687-row
reconciliation and the corrected re-run) are done, and both confirmed the
gate rather than surfacing a new problem.

### 2.4 Corporate-action / dividend timing risk specific to the open

Restating the mechanism, not just flagging it as a generic risk:
`compute_total_return()` (`src/ingestion/adjustments.py`) anchors each
dividend's back-adjustment factor to the split-adjusted **close** on the
session immediately before the ex-date (`prior_close` in that function,
line ~160), then applies the resulting factor uniformly to every price —
including `open` — strictly before the ex-date. This is standard and
correct for `close`-anchored monthly rebalancing (Phase 3/4's use case).
For a daily overnight-return target specifically, two things need
checking that were never relevant before:

1. **The ex-date session itself.** No adjustment factor applies to
   prices *on or after* the ex-date (the loop condition is `date_ <
   ex_date`), so `open` on the ex-date session is the *raw* market open —
   correct, since that's the actual traded price reflecting the real
   ex-dividend drop. But `overnight(ex_date) = ln(open_ex_date /
   close_{ex_date - 1})` mixes a raw-scale `open` against a
   total-return-adjusted `close_{ex_date-1}` if the adjustment isn't
   applied consistently across that specific one-day boundary. **Spot-
   checked against 15 real ex-dates (§2.3 point 4)** — e.g. JCI
   2020-12-18 ($0.26 div): open $46.45 vs. prior close $46.45 (flat, div
   too small relative to typical daily move to show cleanly); HD
   2020-12-02 ($1.50 div): open $273.97 vs. prior close $276.60 (a ~1%
   drop, roughly consistent with a ~0.5% dividend yield plus ordinary
   overnight noise); no case showed a discontinuity suggesting the raw
   `open` and adjusted `close_{t-1}` are being mixed inconsistently.
   This is a spot-check on 15 of thousands of ex-dates, not an exhaustive
   proof — the leakage-audit test suite required at §12's implementation
   gate must still include a dedicated automated test for this specific
   boundary, mirroring `tests/test_ml_features_leakage.py`'s pattern,
   before Tier 0 runs on real data.
2. **Half-trading-days and holiday-adjacent sessions.** yfinance daily
   bars do not distinguish a half-day (e.g. the day after Thanksgiving)
   from a full session — the OHLCV bar looks structurally identical
   either way. A half-day's `open`-to-`close` intraday window is real but
   compressed (1pm close vs. 4pm), which could inflate or deflate the
   apparent overnight/intraday split around a small, calendar-predictable
   set of dates each year, purely as a market-structure artifact rather
   than a genuine overnight-effect signal. This is a **known, accepted
   V1 limitation** (not blocking, not silently ignored): flagged
   explicitly in the diagnostic's output for human review, and revisited
   only if a robustness check (§8) shows the result concentrating around
   these specific dates.

## 3. Sample-size and statistical power assessment

### 3.1 Why Phase 4's formula doesn't transfer

Phase 4 §5's `effective_independent_time_blocks = months_in_window //
horizon_months` assumed monthly rebalancing with an `h`-month-overlapping
label window — the source of non-independence was *labels sharing
calendar months*. Phase 5's target is 1-trading-day-ahead, non-
overlapping by construction (day `t`'s overnight return and day `t+1`'s
overnight return share no calendar time at all). The correlation
structure that actually matters here is different, not absent:

- **Cross-sectional:** hundreds of eligible securities share the same
  overnight macro/news shock on a given day — the same issue Phase 4
  had, but now at daily instead of monthly granularity.
- **Serial:** does today's aggregate overnight return predict
  tomorrow's? Bid-ask bounce, feedback trading, and the very
  institutional-flow mechanisms candidate in §1 could all induce genuine
  day-to-day autocorrelation in the proxy overnight-return series, which
  a naive "one independent observation per trading day" count would miss
  entirely.

### 3.2 Methodology (implemented in `run_phase5_sample_size_report.py`, §0.5)

1. Reuse `build_eligible_universe()` at the existing monthly refresh
   cadence (not re-derived daily — a strategy that only checks index
   membership monthly while trading daily is the realistic, and simpler,
   design; see §0.2). For each trading day, resolve the eligible set from
   the most recent monthly refresh at-or-before that day.
2. Build the per-security daily `overnight_i(t)` panel (§2.1) and the
   equal-weighted proxy series `overnight_proxy(t) = mean over ELIG of
   overnight_i(t)`.
3. **Cross-sectional correlation:** estimate the average pairwise
   correlation `rho` between securities' overnight-return time series
   (sampled security pairs with sufficient common history, not a full
   N² computation), and report the implied cross-sectional effective
   breadth `N_eff = N/(1+(N-1)*rho)` — for context on per-day
   cross-sectional power (relevant if Phase 5 ever escalates to a
   ranking strategy), reported separately from, and never multiplied
   into, the time-block count.
4. **Serial correlation:** compute the lag-1…lag-20 autocorrelation of
   the proxy series, and derive an effective sample size via
   `N_eff = N / (1 + 2 * sum_{k=1}^{K} rho_k)`, truncating `K` at the
   first lag where `|rho_k|` falls inside its own `±2/sqrt(N)` band (the
   standard "not distinguishable from white noise" cutoff), capped at 20
   trading days. This is the number that determines statistical power
   for the primary test (§6.0), analogous in role to Phase 4's
   "effective independent time blocks."
5. Repeat per split window (train/validation/locked-test, chronological
   60/15/25 — see §4.2) since a regime with different volatility
   clustering could show meaningfully different autocorrelation than the
   full-sample figure, and the locked-test window's own effective-N is
   what ultimately bounds the confirmatory test's power (§8).

### 3.3 Consequences this forces on the rest of the design

- The primary significance test (§6.0) must use the block-bootstrap /
  autocorrelation-adjusted confidence interval from this section, never
  a naive i.i.d. standard error across tens of thousands of raw
  (security, day) rows — that would understate uncertainty by treating
  serially- and cross-sectionally-correlated observations as
  independent, exactly Phase 4 §5.4's finding, now at daily granularity.
- If the locked-test window's effective-N is small enough to make a
  single confirmatory test underpowered, that is itself a legitimate,
  reportable finding — not a reason to lower the significance bar or
  expand the test-set touch count after the fact.
- Given daily data has vastly more *nominal* rows than Phase 4's monthly
  panel, there is a real risk of mistaking nominal statistical
  significance (trivially achieved with tens of thousands of raw rows)
  for genuine evidence — this section's entire purpose is closing that
  gap before any test result is looked at, exactly mirroring Phase 4's
  own justification for its §5.

### 3.4 Real findings, from `PHASE5_SAMPLE_SIZE_REPORT.json`

**Headline numbers:** 969,687 nominal (security, day) pairs; 2,263
trading days with a computed proxy return (2015–2023, matching Phase
4's date range); average eligible breadth 428.5/month (identical to
Phase 4's own figure — confirms the same monthly-refresh universe is
being reused correctly, not silently redefined).

**1. Cross-sectional correlation is high — higher than Phase 4's
monthly analogue, and this is the single most decision-relevant number
in this report.** Estimated average pairwise correlation between
securities' overnight-return series: **ρ ≈ 0.43** (2,000 sampled
security pairs, each with ≥60 days of common history). The implied
cross-sectional effective breadth is **N_eff ≈ 2.3** — i.e. on any given
day, despite ~428 nominally eligible names, there are only about *two
to three* genuinely independent overnight "bets" worth of cross-
sectional information. This is mechanistically unsurprising in
hindsight (overnight gaps are dominated by index-futures/macro moves
that hit nearly every stock simultaneously, unlike idiosyncratic
intraday order flow) but is a real, load-bearing constraint, not a
minor caveat:

  - It **confirms §2's primary-metric choice was right** — testing the
    time series of the equal-weighted proxy (one aggregate series) is
    well-matched to a market where the cross-section carries almost no
    independent information on a given day.
  - It is **a serious problem for any Tier 2 cross-sectional ranking
    strategy** (§6 originally scoped Tier 1/2 loosely toward
    "predicting `overnight_i(t)`" per-security). With ~2.3 independent
    effective names per day, a per-security ranking/stock-picking
    approach at daily granularity would be operating with almost no
    real cross-sectional power, regardless of how many nominal
    securities are in the panel. **§6 is revised accordingly below.**

**2. Serial (day-to-day) autocorrelation is real, but the effective-N
formula as implemented has a real weakness the raw output exposes —
flagged rather than used at face value.** Lag-1 autocorrelation is
consistently negative and well outside the ±2σ significance band in
every window that has enough data to measure it reliably (full sample:
−0.119 vs. a ±0.042 band; train: −0.121 vs. ±0.054; validation: −0.211
vs. ±0.109) — a genuine, actionable finding: yesterday's aggregate
overnight return predicts a *partial reversal* tomorrow, which is
exactly the kind of literature-consistent, economically interpretable
signal (bid-ask bounce / overreaction-correction) that would motivate a
Tier 1 lagged-return feature (§6).

The original closed-form formula `N_eff = N/(1+2·Σρ_k)` is not robust when
consecutive lags have opposite signs of similar magnitude, which is
exactly what happens here (lag-1 ≈ −0.12, lag-2 ≈ +0.12, largely
cancelling): it produced a **nonsensical result on its face** — effective-N
values *larger* than the nominal day count in the full sample and
validation window (2,269.4 of 2,263 nominal; 464.4 of 339 nominal in
validation) — since autocorrelation cannot increase genuine independent
information. This is a **known limitation of the truncated-sum
Newey-West-style estimator under alternating-sign autocorrelation**, not
a data problem, and **exactly the kind of "diagnostic reveals a flaw in
the diagnostic itself" finding this project's own discipline exists to
surface rather than paper over** (directly analogous in spirit to Phase
1's own timing-metric-dilution and bad-OHLC-inflation bugs).

**Fixed and re-run (`script_version: 2`, circular moving-block bootstrap
replacing the closed-form ratio):** every window now produces a sane
result, explicitly capped at its own nominal day count
(`capped_at_nominal: true` in all four windows below) —

| Window | n nominal | block length | N_eff (uncapped) | N_eff (reported) |
|---|---|---|---|---|
| Full sample | 2,263 | 7 | 2,514.2 | 2,263 (capped) |
| Train | 1,358 | 7 | 1,454.3 | 1,358 (capped) |
| Validation | 339 | 9 | 507.4 | 339 (capped) |
| Locked test | 566 | 5 | 674.5 | 566 (capped) |

Every window's uncapped estimate exceeds its nominal count — consistent
with the same alternating-sign autocorrelation structure identified
above, and exactly why the cap exists rather than reporting the raw
bootstrap number uncritically. The locked-test window remains, as
before, the cleanest of the four (smallest relative gap between uncapped
and nominal), and **is still the number this spec uses to size §10's
confirmatory budget** — `N_eff = 566`, a materially better-powered
position than Phase 4's ~9 independent monthly blocks.

*(One minor, transparently-noted discrepancy: this script's own train/
validation split reports 1,358/339 nominal days, 5 more in each than
`run_phase5_tier0_test.py`'s 1,353/335 — both call
`build_primary_split()` with the same `embargo_periods=5`, so this is
almost certainly a boundary-counting difference between the two scripts'
own date-range resolution, not a disagreement about the underlying data.
It does not affect any conclusion in this document or in
`PHASE5_CONCLUSION.md`, both of which cite the actual date-labelled proxy
series, not this count. Not chased down further since Phase 5 is closed
— worth a look if this split-construction path is ever reused.)*

**Conclusion: §12 gate (b) is fully closed** — the effective-N formula
has been replaced with the more robust block-bootstrap estimator and
re-run against real data; every window now produces a sane, correctly-
capped result.

## 4. Leakage risks specific to daily granularity

Extends Phase 4 §8's audit to what changes at 1-day resolution (Phase
4's controls — `date <= as_of_date` everywhere, ratio-only-within-window,
never re-querying `ELIG(t+h)`, normalisation fit only on training data —
all still apply unchanged and are not restated in full here).

### 4.1 Exact market-open/close timestamp handling

`overnight_i(t)` and `intraday_i(t)` both resolve `open`/`close` via
exact-date row lookups (§0.3's new same-day-open accessor), never a
"nearest available" fallback the way `targets.py`'s forward-return
resolution does for multi-month labels — an overnight return anchored to
a stale `open` several days later would silently become a multi-day
return mislabeled as an overnight one. A security missing `open` (or
`close`) on a specific date is simply excluded from that day's panel
row and counted, never imputed — same discipline as every existing
point-in-time function in this codebase.

### 4.2 Half-trading-days and holiday-adjacent sessions

Covered mechanically in §2.4 point 2. The leakage-relevant angle
specifically: a half-day's compressed intraday window must not be
silently blended with normal-session data in a way that makes the
*intraday* component look structurally different from the *overnight*
component for reasons unrelated to the hypothesis. Flagged as a known
V1 limitation, checked in the robustness section (§8), not blocking.

### 4.3 Corporate actions effective exactly at the open

Covered in full in §2.4 point 1 — the one genuinely new leakage-adjacent
risk this phase introduces relative to Phase 4, because Phase 4's
monthly cadence never required crossing an ex-date boundary at 1-day
resolution. Must be verified against real ex-dates before this spec is
finalized (§23).

### 4.4 Walk-forward / embargo at daily granularity

`walk_forward.build_primary_split()` is reused (§0.1) with
`embargo_periods` expressed in **trading days**, not months. Given the
target horizon is exactly 1 day (no multi-day overlap the way Phase 4's
3-month label had), the embargo requirement is much smaller than Phase
4's — proposed: `embargo_periods = 5` trading days at each split
boundary, a conservative buffer against the serial-correlation structure
found in §3 rather than a strict overlapping-label requirement (since
there is no overlapping label at h=1). This is a proposed default,
flagged for approval at §23(a), not a value already locked in.

### 4.5 Normalisation / feature construction (only relevant if Tier 1/2 is reached)

If §6 escalates past the Tier 0 distributional test, any
feature-scaling/normalisation must be fit only on that fold's training
window — identical rule to Phase 4 §8.5, restated because it is easy to
forget it still applies once model fitting starts.

## 5. Sample-size analysis — see §3

(Placed at §3 per this document's own dependency order — the sample-size
finding constrains §6's test design and §8's decision table, so it needed
to come before both. This numbered reference exists only so the
requested outline item is easy to locate.)

## 6. Model hierarchy

**Tier 0 — simple distributional test (mandatory first step, this is
where §23's implementation approval initially stops):**

Primary test: is the mean of `overnight_proxy(t)` (the equal-weighted
daily overnight log-return series, §2.1) statistically distinguishable
from zero, **and** from the mean of `intraday_proxy(t)` over the same
days — using a paired difference test — with confidence intervals from
a **block bootstrap** whose block length is set by §3's measured
autocorrelation decay (not an arbitrary round number), evaluated
separately per split window per §3.2 step 5. No feature, no model, no
fitting — a statistical test on two already-observed return series.

**Tier 1 — linear model, revised scope after §3.4's cross-sectional
finding (only if Tier 0 shows a validation-period effect worth
refining, never triggered by a train-period-only or locked-test
result):**

§3.4's measured ρ≈0.43 (implied cross-sectional effective breadth ≈2.3)
means a **per-security ranking model is not a well-powered use of this
data at daily granularity** — this materially changes what was
originally sketched for Tier 1. Revised target: predict the **sign or
magnitude of the aggregate `overnight_proxy(t)` series itself** (a
single time series, matching what Tier 0 already tests), not
`overnight_i(t)` per security. Candidate features: yesterday's
`intraday_proxy(t-1)` and `overnight_proxy(t-1)` (§3.4's measured lag-1
reversal is the leading candidate feature, not a guess), recent realised
volatility of the proxy series, day-of-week, turn-of-month indicator.
Reuses `src/ml/baselines.py`'s `tune_hyperparameters()`/inner-CV
machinery unchanged (§0.1) — no new tuning code, just a one-series
rather than per-security training panel.

A **separate, explicitly secondary** question — does *cross-sectional*
information exist at all, despite the low effective breadth — may be
explored on train/validation data only, never promoted to a
confirmatory test without its own pre-registration and its own slot in
§10's shared budget, given how little independent power §3.4 shows is
available for it.

**Tier 2 — tree ensembles (only if Tier 1 shows validation-period value
over the Tier 0 baseline):**

Random Forest / HistGradientBoosting on the same proxy-series target as
the revised Tier 1, reusing `src/ml/trees.py` and Phase 4's frozen-grid-
plus-ablation-ladder discipline exactly. Same leave-one-family-out
diagnostic as Phase 4 §16. The price-history-length diagnostic (Phase 4
§8.6) is less directly relevant to a single aggregate series and is
carried forward only if Tier 2 reintroduces any per-security feature.

**Hard exclusion, all tiers: no neural network, LSTM, TCN, or
Transformer architecture** — reaffirmed independently of Phase 4's own
sample-size argument (§11 there), because nothing about moving to daily
granularity changes the fundamental problem: more nominal rows is not
more independent evidence (§3), and a temporal deep-learning model's
data appetite scales with genuinely independent regime-diversity, not
raw row count.

**Escalation is one-directional and gated on VALIDATION evidence only** —
identical discipline to Phase 4 §9.4/§9.5: the locked test set is never
touched to decide whether to escalate a tier, only to confirm a
pre-registered, already-frozen hypothesis once every modelling decision
is fixed.

## 7. Data requirements — see §2.3/§2.4

(Same placement note as §5 — folded into §2 where the target definition
naturally requires it, referenced here for the outline.)

## 8. Pre-declared success/failure criteria and decision table

Modelled directly on Phase 4 §20, adapted for a distributional Tier-0
primary test rather than a cross-sectional IC:

- **Failure:** the Tier 0 block-bootstrap confidence interval for the
  locked-test-window mean `overnight_proxy(t)` contains zero, **or** the
  overnight-vs-intraday paired-difference CI contains zero. Escalation
  to Tier 1/2 does not happen (§6's gating), and this phase concludes
  with a documented negative (§10).
- **Inconclusive:** the CI excludes zero but the effect is economically
  trivial — proposed threshold: annualised mean overnight-proxy return
  magnitude below 1.0 percentage point (a number chosen to be
  comfortably below the smallest transaction-cost drag any realistic
  daily-turnover strategy could clear, per §9, not chosen to flatter an
  unseen result) — **or** the locked-test effective-N from §3 turns out
  too small for the CI itself to be trustworthy (flagged explicitly, not
  glossed over, exactly Phase 4 §20's "inconclusive" spirit).
- **Evidence of a genuine, further-investigable effect:** CI excludes
  zero, magnitude exceeds the inconclusive threshold, **and** the effect
  survives every robustness check below without collapsing.

**Robustness checks a positive-looking Tier 0 result must survive before
being taken seriously** (mirroring Phase 4 §16's investigation-trigger
discipline):

1. **Sub-period stability** — at least one high-volatility regime (2020)
   and one low-volatility regime tested separately; an effect
   concentrated in a single 4-month window (exactly Phase 4's own
   Aug–Nov 2020 finding for the cross-sectional models) is presumptive
   evidence of a regime artifact, not a stable effect.
2. **Half-day/holiday-adjacent exclusion** — per §4.2, re-run excluding
   calendar-known half-trading-days; if the effect meaningfully weakens,
   that's a market-structure artifact, not the hypothesised mechanism.
3. **Ex-date exclusion** — per §4.3, re-run excluding sessions within a
   few days of any corporate action ex-date for the affected security;
   same collapse-implies-artifact logic.
4. **Cross-sectional-breadth sensitivity** — re-run on STRICT (Phase 4's
   existing sanity-check policy) as a degenerate-case check, and on a
   restricted large-cap-only sub-universe, to confirm the effect isn't an
   artifact of PERMISSIVE's broader, lower-quality inclusion criteria.
5. **Transaction-cost sensitivity** (§9) — ±20% on `bid_ask_spread_bps`,
   consistent with Phase 4 §16's own sensitivity range.

## 9. Transaction costs and execution realism

### 9.1 Why this is a different problem than Phase 4's

Phase 4's `top_n_momentum` benchmark, at *monthly* rebalancing, already
measured 66.7% average turnover and real, material cost drag (§4 of
`PHASE4_SPECIFICATION.md`). An overnight strategy implies, at the
limit, a full round-trip **every single trading day** (enter near the
close, exit near the next open) — roughly 20x Phase 4's rebalancing
frequency. Cost realism is not a secondary concern here; it is close to
the whole ballgame, since even a real, statistically genuine overnight
effect of a few basis points per day can be entirely consumed by daily
round-trip costs.

### 9.2 Cost model — reused, not reinvented (§0.1)

`src/backtest/costs.py`'s `trade_cost()` and `config.yaml`'s
`backtest.costs` block apply unchanged: `fx_cost_pct` (0.15%,
non-zero for a UK ISA holding US-listed stocks — this is the single
largest fixed per-trade cost in the existing model and would be paid on
**every entry and every exit** at daily frequency, not once a month),
`bid_ask_spread_bps` (5bps, existing conservative fixed assumption), zero
commission (Trading 212 stock trades), and the SEC/FINRA sell-side fee.
No second cost model is introduced — every Phase 5 result is reported
gross and net, with turnover/cost-drag fields identical to Phase 4's.

### 9.3 A genuinely new execution-realism gap Phase 4 didn't have — now verified against Trading 212's own documentation

The idealised `overnight_i(t) = ln(open_t/close_{t-1})` decomposition
assumes execution **exactly at the closing print and exactly at the next
opening print** — the theoretical construct the academic literature
actually studies. Checked directly against Trading 212's help centre
(§12 gate (c), now closed):

- Trading 212 supports **market orders** (execute immediately at the
  best available bid/ask when processed) and **limit orders**; there is
  **no market-on-close or market-on-open order type** in either
  documentation page — nothing schedules an order to execute
  specifically at the closing or opening auction print.
- Trading 212's "24/5" US-stock trading spans four sessions: pre-market
  (4:00–9:30am ET), regular hours (9:30am–4:00pm ET, the official
  NYSE/Nasdaq session), after-hours (4:00–8:00pm ET), and an
  **overnight session (8:00pm–4:00am ET) priced via Blue Ocean ATS — a
  separate trading venue, not the primary exchange.** Standard/
  pre/after-hours sessions use consolidated best-price data across US
  exchanges; the overnight session specifically does not.

**This matters concretely, in two distinct ways:**

1. A realistic Trading 212 proxy for "sell at the close, buy at the next
   open" is a regular market order placed in the last minutes of the
   4:00pm session and another in the first minutes of the 9:30am
   session — subject to ordinary bid/ask spread and slippage against the
   *actual* closing/opening print, not a guaranteed match to it.
2. Attempting to literally hold "overnight" using Trading 212's
   dedicated overnight session means trading on **Blue Ocean ATS
   pricing**, which is a different venue and reference price than the
   NYSE/Nasdaq close/open this specification's target (§2) and the
   academic literature it's built on are actually defined against —
   using that session would not be measuring the same effect at all, and
   is explicitly out of scope for how §2's target is computed.

Neither path is a free proxy for the idealised decomposition. This is
reported as a **named executability caveat**, distinct from the
statistical significance question — a result can be genuine (§8) and
still require its own separate, explicitly-scoped tracking-error/
slippage study before being called "tradeable at retail scale" — not
proposed as part of Phase 5's own scope (§0.4's DELETE list already
excludes any live-trading work).

Sources: [What are the types of orders? – Trading 212](https://helpcentre.trading212.com/hc/en-us/articles/30721439008797-What-are-the-types-of-orders), [Market Orders – Trading 212](https://helpcentre.trading212.com/hc/en-us/articles/360007081257-Market-Orders), [Limit Orders – Trading 212](https://helpcentre.trading212.com/hc/en-us/articles/360007139778-Limit-Orders), [24/5 Trading - Invest & Stocks ISA – Trading 212](https://helpcentre.trading212.com/hc/en-us/articles/13574400364445-24-5-Trading-Invest-Stocks-ISA)

### 9.4 Sensitivity check

`bid_ask_spread_bps` and `fx_cost_pct` at ±20% (§8 robustness check 5),
reported for both a naive "trade every day" turnover assumption and
whatever the eventually-tested strategy's real measured turnover turns
out to be once §6 produces one (Tier 0 alone has no turnover — it is a
statistical test, not a strategy).

## 10. Multiple-testing context

Per your instruction: Phase 5 may become one of **up to two**
pre-registered hypotheses, sharing **one** corrected significance budget
and **one** confirmatory-test-set allocation — not a fresh budget granted
per hypothesis. Concretely, proposed for `config.yaml`'s (not-yet-written)
`phase5:` block:

```yaml
phase5:
  experiment_tracking:
    # Shared across every Phase 5 hypothesis, not per-hypothesis.
    max_confirmatory_test_experiments_shared: 2
    hypotheses_registered: ["overnight_intraday_decomposition"]  # this document
    # A second hypothesis, if and when proposed, is appended here --
    # never a separate config block with its own independent budget.
```

**Derivation (§3.4's real numbers):** unlike Phase 4, where the cap of 5
was mechanically derived from a scarce resource (~9 independent monthly
test blocks), Phase 5's locked-test window has ~566 effective
independent trading days (§3.4) — comfortably powered, not the binding
constraint here. The binding constraint is instead your own instruction
that Phase 5 shares one budget across **up to two** pre-registered
hypotheses: proposed cap is **2** — one confirmatory test-set touch per
hypothesis, mirroring Phase 4 §17's "no more than one confirmatory run
per declared axis" discipline, applied here at the hypothesis level
rather than the robustness-axis level, since §3.4 found the real
scarce resource in this phase is cross-sectional breadth (≈2.3
effective names/day, §3.4 point 1), not time-series length. If Tier
1/2 escalation (§6, §11) ever needs a second test-set touch for the
*same* hypothesis, that consumes the second slot, not a fresh budget —
consistent with §11's stopping rule.

## 11. Stopping rule

**If §8 resolves to Failure:** this branch closes with a
`PHASE5_OVERNIGHT_GAP_CONCLUSION.md` document, in the same style as
`PHASE4_CONCLUSION_3M.md` — reporting the measured effect size, its
confidence interval, and which robustness checks (if any) were even
reached, as a legitimate, useful research outcome. No Tier 1/2 escalation
occurs. The locked test set's one-time confirmatory touch (§10) is
consumed and logged regardless of outcome, exactly as Phase 4 logged
every touch, successful or not.

**If §8 resolves to Inconclusive:** documented with the same rigor as a
Failure, explicitly not reframed as a partial success by selecting a more
flattering metric after the fact (identical discipline to Phase 4 §20's
"inconclusive" category) — and explicitly not a license to lower the
magnitude/significance thresholds and re-test, since the confirmatory
budget (§10) has already been spent.

**If §8 resolves to genuine evidence of an effect:** escalate to Tier 1
(§6) using validation-period data only; a second, separately pre-
registered confirmatory test-set touch (drawn from the same shared
budget, §10) would be required before Tier 1's own result could be
reported as confirmatory — this is not automatic and requires a fresh
round of this project's stop/go gate discipline (§12), not a continuation
under the original approval.

No Phase 6 / live trading / Trading 212 order execution work begins as
part of, or as a consequence of, Phase 5 — regardless of outcome, mirroring
Phase 4 §23's identical closing line.

## 12. Phase 5 stop/go gate

**Specification → implementation** requires:

(a) ✅ **Closed.** Real output from `run_phase5_open_price_data_check.py`
    (§2.3/§2.4) — open-price coverage confirmed adequate (100.0%
    row-level population). Two small follow-ups remain (the 687-row
    bad-OHLC reconciliation, and re-running the exact-date measurement
    with corrected trading-session resolution) but are not blocking.
(b) ✅ **Closed.** Real output from `run_phase5_sample_size_report.py`
    (§3.4) — locked-test effective-N (566 days, clean) used to set §10's
    confirmatory-budget cap (2). The effective-N *formula* itself needs
    replacing with a direct block bootstrap before Tier 0's own CI is
    computed (§3.4) — that's an implementation-phase task, not a gate on
    starting implementation.
(c) ✅ **Closed.** Trading 212 order-type support verified against its own
    help centre (§9.3): no market-on-close/market-on-open order type
    exists; the "overnight" session uses a different venue (Blue Ocean
    ATS) than the NYSE/Nasdaq close/open this spec's target is defined
    against.
(d) **Open — the one remaining gate.** Your explicit sign-off on: the §8
    success/failure/inconclusive thresholds (unchanged from the original
    draft — nothing in the real data changed the reasoning behind the
    1.0pp annualised threshold), the §4.4 embargo default (5 trading
    days — the real lag-1/lag-2 autocorrelation found in §3.4 falls
    comfortably inside this margin), the §10 confirmatory-budget cap
    (now set to 2, derived above), and the revised Tier 1/2 scope in §6
    (proxy-series prediction rather than per-security ranking, following
    directly from §3.4's cross-sectional finding).

**Implementation → Tier 0 test execution** requires: the new
`overnight_targets.py` module and same-day-open accessor (§0.3) passing a
leakage-audit test suite analogous to
`tests/test_ml_features_leakage.py` (future-price-mutation invariance,
insufficient-history returns `None`, reproducibility) and to
`tests/test_backtest_execution_timing.py`'s execution-date discipline,
before the Tier 0 statistical test is run on real data.

**Tier 0 → locked-test evaluation** requires: the Tier 0 test config
frozen and logged (via the existing `ml_experiments` table, §0.1/§0.2)
*before* the locked test window is touched, and your explicit written
confirmation that it may now be evaluated — identical, deliberately, to
Phase 4 §23's irreversibility gate.

Nothing beyond §0's diagnostic scripts is built before (a)–(d) above are
satisfied.

---

## Appendix A: Dependency graph

```
Data (prices incl. open, corporate_actions)              [Phase 1-2, reused]
        |
        v
Diagnostics: open-price coverage (S2.3/2.4) + sample-size/
             autocorrelation report (S3)                 [NEW scripts, not yet run]
        |
        v
Point-in-time universe: ELIG(t), monthly refresh          [Phase 3/4, reused unchanged]
        |
        v
Target: overnight_i(t), intraday_i(t)                     [S2, NEW small module]
        |
        v
Tier 0: block-bootstrap distributional test on the
        equal-weighted proxy series, per split window      [S6, NEW small module]
        |
        v
Pre-registered decision table applied to LOCKED TEST only  [S8]
        |
        v
   FAILURE / INCONCLUSIVE  -->  STOP, write conclusion doc  [S11]
        |
   GENUINE EFFECT (validation-gated)
        |
        v
Tier 1: linear model on daily features                     [S6, Phase4 machinery reused]
        |
        v
Tier 2: tree ensembles, same ablation/robustness discipline [S6, Phase4 machinery reused]
        |
        v
Execution-realism gap check against real Trading 212
order-type support                                          [S9.3]
        |
        v
STOP -- report to user, no Phase 6 work begins               [S11]
```
