# Phase 4 Specification: Predictive Modelling Research (Point-in-Time, Walk-Forward)

Status: DRAFT — awaiting approval. No implementation, feature computation, model
training, or evaluation has begun.

Depends on: Phase 1 (data foundation, complete), Phase 2 (corporate actions /
total-return, complete), Phase 3 (point-in-time backtesting engine + 200-seed
random benchmark, complete — see `PHASE3_SPECIFICATION.md`, `PHASE3_200SEED_FINAL_REPORT.md`).

## Guiding principles

These extend Phase 3's guiding principles to the ML-specific risks this phase
introduces, and take precedence over local convenience during implementation.

- No look-ahead bias, in any form — features, labels, model selection, or
  evaluation — anywhere in the pipeline.
- No survivorship masquerading as predictive performance.
- No future metadata reachable from a historical decision, matching Phase 3 §4's
  `as_of_date` invariant exactly.
- No retrospective threshold, hyperparameter, or portfolio-construction tuning —
  every choice with predictive consequences is fixed in configuration *before* it
  touches the locked test period.
- **Do not reinvent the Phase 3 backtesting machinery.** Phase 4 produces a new
  *signal* (a ranked prediction), not a new backtester. That signal plugs into
  Phase 3's existing strategy interface, universe construction, cost model, and
  persistence schema unchanged.
- No silent data cleaning. Every exclusion, imputation, or missing-data decision
  is auditable and attributable to a specific, logged rule.
- No fabricated data. If a feature needs data we don't have, it is labelled
  `REQUIRES NEW DATA` or `UNSAFE/DEFERRED` — never quietly approximated with an
  external dataset that wasn't identified up front.
- Every experiment is reproducible: identical config + seed must produce
  identical predictions and identical downstream backtest results.
- **Optimise for determining whether genuine predictive information exists, not
  for finding a profitable backtest.** A negative or inconclusive result is a
  valid, useful Phase 4 outcome and must be reported as such.

---

## 1. Research question

> Can information that was genuinely available at time *t* predict the
> subsequent cross-sectional performance of stocks over a defined future
> horizon, sufficiently well to construct a portfolio that outperforms
> appropriate Phase 3 benchmarks after realistic transaction costs?

Phase 4 is a research question, not a product commitment. The honest possible
outcomes are: **no evidence of predictive value**, **inconclusive** (data/power
insufficient to tell), or **evidence of genuine predictive value** — see §20 for
how each is defined in advance.

## 2. Scope

**In scope:** target definition, feature engineering from existing point-in-time
data, a formal leakage audit, chronological walk-forward training/validation,
simple interpretable baselines, tree-ensemble ML models (conditional on baseline
evidence), conversion of predictions into portfolio weights via Phase 3's
strategy interface, and evaluation against the four Phase 3 deterministic
benchmarks plus the 200-seed random distribution.

**Primary research universe: PERMISSIVE only.** STRICT is excluded from all
Phase 4 modelling, training, and evaluation. The 200-seed report confirmed
STRICT's filters leave essentially one eligible security for the entire
2015–2023 period (`num_distinct_securities_ever_selected: 1`,
`num_distinct_holding_sets: 1` across all 200 seeds) — there is no cross-section
to model. STRICT is retained solely as a degenerate-case sanity check (e.g. "does
the pipeline still run and produce a trivial, explainable result under an
almost-empty universe") — never as a parallel research track, and never
reported as if it were a second independent result.

**Out of scope (explicit non-goals):** Phase 5 / live trading / Trading 212
order execution / any real-money component; a second, parallel transaction-cost
model; cap-weighted benchmarking (still blocked — no market-cap data, see §7);
sector-relative modelling (no populated sector data, see §7); neural network
architectures (see §11, §22); optimising portfolio construction or thresholds
against observed test-period performance.

## 3. Target variable

### 3.1 Definitions

Let `ELIG(t)` be Phase 3's own point-in-time eligible universe under the
PERMISSIVE policy — literally `build_eligible_universe(t, PERMISSIVE, ...)`,
reused unchanged, never reimplemented. Let `t` and `t+h` be entries in Phase 3's
existing `rebalance_dates` sequence (monthly), so `h` is expressed in **rebalance
periods**, not calendar days — this sidesteps weekday/holiday ambiguity and
keeps every horizon aligned to a date the engine already resolves via
`next_market_session`.

For security `i ∈ ELIG(t)`, using `total_return`-adjusted closes exclusively
(never `raw` or `split_adjusted` — see §8.4 for why this specific series is
mandatory here):

```
r_i(t, h) = TR_i(t+h) / TR_i(t) - 1
```

Benchmark — the point-in-time equal-weighted return of the *same* eligible set,
i.e. exactly what Phase 3's `equal_weight_sp500` strategy already realises over
that period (reused, not redefined):

```
B(t, h) = mean over j ∈ ELIG(t) of r_j(t, h)
```

**Primary target — excess return over the point-in-time equal-weighted eligible
universe:**

```
y_i(t, h) = r_i(t, h) - B(t, h)
```

**Derived targets, computed from the same underlying label — never a second,
inconsistent definition:**

- `rank_i(t, h)` — cross-sectional percentile rank of `y_i(t,h)` (equivalently of
  `r_i(t,h)`, since `B(t,h)` is constant across `i` at fixed `t`) within
  `ELIG(t)`. Used for Spearman IC / rank-correlation evaluation (§14) and as an
  optional alternative training objective (learning-to-rank) in later
  experiments — not V1.
- `z_i(t, h) = 1[y_i(t,h) > 0]` — binary "outperformed the eligible-universe
  average" label, for the logistic-regression baseline required by §10.

### 3.2 Why excess return over the eligible universe, not literal "S&P 500"

ChatGPT's target list includes "future excess return relative to S&P 500." Our
schema has no ingested S&P 500 index-level price series — only per-security
`prices` rows (see §7). A literal cap-weighted index return is `REQUIRES NEW
DATA`. The only benchmark we can compute without inventing a data source is the
equal-weighted return of the same point-in-time eligible set — which is exactly
Phase 3's `equal_weight_sp500` benchmark, already validated, already persisted.
Using it as the label's benchmark keeps Phase 4 architecturally consistent with
Phase 3 rather than introducing a second, undeclared notion of "the market."

### 3.3 Why excess return over rank or absolute return

Absolute return `r_i(t,h)` has wildly different scale and variance across
months — a model trained on raw absolute returns across 2018 vs 2021 would
mostly be learning "which months were good," not genuine cross-sectional
stock-picking skill. Cross-sectional excess return removes the shared monthly
component by construction and is the most direct, mathematically simplest
target that still supports every downstream use (regression loss, rank
correlation, sign-based classification) from one definition. Absolute
return-based baselines (§10) are still reported for reference, not discarded.

### 3.4 Delisting / truncated-window labels — a specific, named leakage/bias risk

A security that leaves `ELIG(t)` (or loses price coverage entirely) between `t`
and `t+h` cannot have `TR_i(t+h)` computed normally. **Silently dropping such
securities from the training/evaluation sample is a real bias, not a
convenience** — delisting/bankruptcy correlates with poor realised returns, so
dropping them shifts the observed label distribution positive (this is the
mechanism behind classic survivorship bias, now reintroduced one label at a
time if handled carelessly). Required handling:

1. If a resolvable `total_return` price exists at or before `t+h` (partial
   coverage), use the *last available* price as a frozen terminal value and set
   an explicit `label_truncated = true` flag on that row.
2. If no resolvable price exists at all before `t+h`, exclude the row and log it
   — do not impute a return.
3. Truncated-window labels must be reportable separately (count, average
   truncated-vs-full-window label value) in every experiment's output — not
   silently blended into the main statistics. If a model's apparent edge
   disappears once truncated labels are excluded, that is a direct signal the
   "edge" was actually a survivorship artifact, which is exactly the class of
   confound §16 requires us to check for.

### 3.5 Horizons

See §4.

## 4. Prediction horizons

Candidates: 1, 3, and 6 rebalance periods ahead (≈ 1, 3, 6 months, given
Phase 3's monthly rebalance cadence — no other cadence exists in the engine).

**Primary: h = 3 months.** Reasoning, grounded in this project's own evidence
and standard cross-sectional literature, not an arbitrary pick:

- Phase 3's `top_n_momentum`/`bottom_n_momentum` benchmarks already use a
  252-day (~12M) *signal formation* lookback, suggesting multi-month windows are
  where this team already expects momentum-type information to live — but 1M
  forward returns are the horizon most associated with short-term *reversal*,
  not continuation, in the momentum literature, making 1M a poor primary target
  for a momentum-adjacent feature set.
- 1M horizon means monthly-rebalance-on-a-1M-signal — the exact configuration
  that produced PERMISSIVE `top_n_momentum`'s real, measured turnover (66.7%
  average) and transaction-cost drag ($19,915 total) in the 200-seed report.
  Repeating that turnover profile on a noisier 1M signal is a real, already-
  demonstrated cost risk, not a hypothetical one.
- 6M reduces turnover and noise further but roughly halves the number of
  independent, non-overlapping time blocks available in any fixed window
  (§5) — a real statistical-power cost.
- 3M is the standard middle ground: long enough to sit past the 1M reversal
  region and reduce turnover meaningfully, short enough to preserve a usable
  number of independent test-period blocks (§5).

1M and 6M are **secondary/diagnostic horizons**: 1M specifically for a
turnover/cost-sensitivity comparison against the 3M result (does the extra
turnover a shorter horizon implies ever pay for itself net of costs), 6M for a
robustness check on the 3M finding, run only if 3M shows a validation-period
signal worth chasing (§9, §16). Neither is scored against the locked test set
unless it is pre-registered to be, per §17.

## 5. Sample-size and statistical power assessment

This section did not exist in ChatGPT's original outline. It is added because
every later design decision — feature count, model count, number of ablations —
must be constrained by how much genuinely independent evidence the dataset can
actually supply, and that number turns out to be much smaller than the raw row
count suggests.

### 5.1 What we actually have

From the validated 200-seed PERMISSIVE run: 108 monthly rebalance dates
(2015-01 to 2023-12), eligible-universe size ranging 367–477 per date, mean
428.5. Raw `(security, date)` pairs across the full period ≈ 428.5 × 108 ≈
**46,300**.

### 5.2 Why 46,300 is not the real sample size

These rows are heavily non-independent in two ways that matter for anything
claiming statistical significance:

- **Cross-sectionally**, every security observed in the same month shares that
  month's market-wide shock. 428 securities in one month is much closer to one
  independent draw of "what the market did" than 428 independent draws.
- **Serially, for overlapping-horizon labels.** At `h=3`, month `t`'s label and
  month `t+1`'s label share two of their three underlying return months. At
  `h=6`, adjacent monthly labels share five of six months. This is the
  "overlapping future-return labels" issue §9 must handle explicitly, and it
  directly shrinks the *effective* number of independent time observations to
  roughly `(months in a window) / h`.

### 5.3 Illustrative calculation (to be recomputed exactly once split boundaries are fixed — see §9)

Using a rough 60/15/25 chronological split of the 108 months (≈ 65 / 16 / 27
months for train / validation / locked test — illustrative, not yet a
decision):

| Horizon | Months in locked test | Effective independent time-blocks in locked test |
|---|---|---|
| 1M | 27 | ≈ 27 |
| 3M (primary) | 27 | ≈ 9 |
| 6M | 27 | ≈ 4–5 |

At the primary 3-month horizon, the locked test period contains on the order of
**9 independent non-overlapping time blocks** — regardless of how many
thousands of `(security, date)` rows are inside them. Cross-sectional breadth
(≈428 securities/month) gives real power to ask "does the ranking work within a
given month," but very little power to ask "does this hold up across many
different macro regimes," because there are only a handful of genuinely
distinct macro periods in the test window at all.

### 5.4 Consequences this forces on the rest of the design

- Significance testing must respect the time-block structure (block
  bootstrap / clustering by rebalance date at minimum — not a naive i.i.d.
  confidence interval across all 46,300 rows, which would understate
  uncertainty by treating correlated rows as independent).
- The number of *confirmatory* hypotheses run against the locked test set must
  be small — single digits, not dozens (§17 sets the actual cap).
- Feature-set size and model complexity for V1 should stay conservative: a
  handful of well-motivated features per family, not an exhaustive kitchen-sink
  set, both because 9 independent blocks cannot support validating a
  high-dimensional model and because it directly limits overfitting surface
  area during the (larger, but still finite) validation-period search.
- This calculation must be **redone programmatically**, exactly, once §9's
  split boundaries are fixed — the numbers above are a grounded estimate from
  real Phase 3 coverage statistics, not a substitute for the real count.

## 6. Feature architecture

Every feature below states: definition, required price representation and why,
lookback, data source, leakage risk, and availability classification (§7 defines
the four classes). "V1" = included in the first experiment; "later" = valid
future work, not built now.

All return/momentum/volatility features are computed on `adj_type='total_return'`
prices. **Reason, not just convention:** `compute_total_return()`
(`src/ingestion/adjustments.py`) performs a full-history *backward
restatement* — every historical `total_return` row is multiplied by the
cumulative factor of **every dividend on record with a later ex-date**,
including dividends after the prediction date (standard CRSP/Yahoo Adj-Close
convention, correct for computing *returns*). This means a raw `total_return`
**level** at a historical date reflects information (all dividends between that
date and the ingestion cutoff) that would not have been knowable in real time.
**Rule: every feature must be expressed as a ratio or return over
`total_return` within its own bounded lookback window — never as a raw
absolute `total_return` level used in isolation, and never mixed with `raw` or
`split_adjusted` values.** Ratios/returns within a bounded window are safe
because the future-dividend multiplicative constant is shared by every term in
the window and cancels exactly (worked derivation in the accompanying design
notes, available on request) — this is a structural property of the
computation, not an assumption. This rule is elevated to a leakage control in
§8.4, not left as a footnote.

### 6.1 Price / momentum

| Feature | Definition | Lookback | Availability | Leakage | V1 |
|---|---|---|---|---|---|
| 1M return | `TR(t)/TR(t-21d)-1` | 21 trading days | AVAILABLE NOW | None (bounded ratio) | Yes |
| 3M return | `TR(t)/TR(t-63d)-1` | 63d | AVAILABLE NOW | None | Yes |
| 6M return | `TR(t)/TR(t-126d)-1` | 126d | AVAILABLE NOW | None | Yes |
| 12M return | `TR(t)/TR(t-252d)-1` (matches existing momentum benchmark lookback) | 252d | AVAILABLE NOW | None | Yes |
| Momentum acceleration | `[TR(t)/TR(t-63d)-1] - [TR(t-63d)/TR(t-126d)-1]` | 126d | AVAILABLE NOW | None | Yes |
| Distance from 200d MA | `TR(t)/mean(TR(t-199d..t)) - 1` | 200d | AVAILABLE NOW | None | Yes |
| Breakout (position in 252d range) | `(TR(t)-min(TR,252d))/(max(TR,252d)-min(TR,252d))` | 252d | AVAILABLE NOW | None | Later — likely collinear with the return features above; include only if ablation (§16) shows independent value |

### 6.2 Volatility / risk

| Feature | Definition | Lookback | Availability | Leakage | V1 |
|---|---|---|---|---|---|
| Realised volatility | stdev of daily `total_return` log-returns | 63d | AVAILABLE NOW | None | Yes |
| Downside volatility | stdev of negative daily returns only | 63d | AVAILABLE NOW | None | Yes |
| ATR-type range measure | mean daily `(high-low)/close` on `split_adjusted` (range measures should use `split_adjusted`, not `total_return` — dividend back-adjustment distorts intraday range interpretation) | 21d | AVAILABLE NOW | None if computed strictly ≤t | Later — secondary to realised vol, adds limited marginal information for a monthly-horizon target |
| Max drawdown (trailing) | max peak-to-trough `total_return` decline | 252d | AVAILABLE NOW | None | Yes |
| Volatility regime | current 21d realised vol vs its own 252d trailing average | 252d | DERIVABLE FROM EXISTING DATA | None | Later — useful mainly as a market-level regime feature (§6.5), redundant at security level for V1 |

### 6.3 Volume / liquidity

| Feature | Definition | Lookback | Availability | Leakage | V1 |
|---|---|---|---|---|---|
| Dollar volume | `close × volume`, per day | 1d | DERIVABLE FROM EXISTING DATA | None | Yes (component of below) |
| Rolling avg dollar volume | mean dollar volume over a **true rolling window** | 20d | DERIVABLE FROM EXISTING DATA, **with a caveat**: `apply_universe_filters()`'s existing `min_avg_dollar_volume_usd` check averages over the security's **entire history to date**, not a rolling 20-day window (despite the config comment referencing "20-day"). A genuine rolling-window feature needs new, small, point-in-time code — it must not silently reuse that function's whole-history average and call it "20-day." | None if window strictly ≤t | Yes |
| Volume trend | rolling 20d avg volume vs rolling 100d avg volume | 100d | DERIVABLE | None | Yes |
| Turnover proxy | dollar volume ÷ trailing price level (crude, since no shares-outstanding data exists) | 20d | DERIVABLE, weak proxy — see §7 | None | Later — flag as a weak proxy in any report that uses it |
| Float / short interest | — | — | REQUIRES NEW DATA | — | No |

### 6.4 Relative / cross-sectional

| Feature | Definition | Lookback | Availability | Leakage | V1 |
|---|---|---|---|---|---|
| Return relative to eligible-universe mean | `r_i(t,63d) - mean_j∈ELIG(t)(r_j(t,63d))` | 63d | DERIVABLE FROM EXISTING DATA | Must use `ELIG(t)` computed with **no knowledge of who is eligible at t+h** — see §8.2 | Yes |
| Cross-sectional percentile rank of momentum | rank of 3M/12M return within `ELIG(t)` | matches base feature | DERIVABLE | Same as above | Yes |
| Volatility percentile | rank of realised vol within `ELIG(t)` | 63d | DERIVABLE | Same | Yes |
| Sector-relative momentum | — | — | **UNSAFE/DEFERRED** — `securities.sector` exists as a schema column but is unpopulated (confirmed 0 populated rows in Phase 1/2; this is a pre-existing, documented data gap, not something Phase 4 can work around) | — | No |

### 6.5 Market / regime

No S&P 500 index-level price series is ingested (schema has no `index_prices`
table, only per-security `prices`). "Market" features must therefore be built
from the same `ELIG(t)` equal-weighted proxy used for the target's benchmark
(§3.2) — this is a **derived proxy, explicitly not the real S&P 500 index**,
and every artifact that uses it must say so.

| Feature | Definition | Lookback | Availability | Leakage | V1 |
|---|---|---|---|---|---|
| Proxy-index trend | trailing return of the equal-weighted `ELIG(t)` return series | 63d/252d | DERIVABLE FROM EXISTING DATA (reuses `equal_weight_sp500`'s own construction) | None | Yes |
| Proxy-index volatility | realised vol of the equal-weighted proxy series | 63d | DERIVABLE | None | Yes |
| Breadth proxy | % of `ELIG(t)` securities above their own 200d MA | 200d | DERIVABLE | None | Yes |
| Bull/bear regime indicator | sign/threshold on proxy-index trailing return | 252d | DERIVABLE (built from the above) | None | Later — redundant with trend/volatility features until ablation shows it adds independent value |
| True cap-weighted S&P 500 return/volatility | — | — | REQUIRES NEW DATA (no market cap, no true index series) | — | No |

### 6.6 Explicitly excluded from the feature set (not a family — a rule)

Data-quality / identity-resolution metadata (`identifier_quality`,
`identity_review_queue` flags, `price_data_quality`) must **never** be used as
predictive features, even though they are `AVAILABLE NOW`. A security's
"well-resolved identity" status plausibly correlates with being a large,
well-known company — using it as a feature risks a spurious pattern that is
really about how much attention Phase 1/2 gave that ticker, not the company's
prospects. This exclusion is itself the first concrete implementation of §16's
"if removing a feature collapses the strategy, suspect leakage or an artifact"
principle — these fields are excluded pre-emptively rather than discovered as
the cause of a collapse later.

## 7. Data requirements and availability classification

Every feature/data need above is labelled with one of four classes:

- **AVAILABLE NOW** — directly queryable from existing tables, no new code beyond
  a point-in-time-safe query.
- **DERIVABLE FROM EXISTING DATA** — computable from existing tables but needs
  new, explicitly-reviewed code (e.g. a true rolling window, not reuse of a
  whole-history average under a misleading name).
- **REQUIRES NEW DATA** — genuinely absent from the schema; would mean a new
  ingestion source, not proposed or approved here.
- **UNSAFE/DEFERRED** — a schema column exists but is unpopulated, or the
  concept can't be safely constructed with current data quality.

Summary of the notable gaps carried forward from Phase 1–3 (already documented
there, restated here because Phase 4 is the phase that would actually be hurt by
them): no populated `sector`/`industry`; no point-in-time market cap or shares
outstanding (blocks cap-weighting and genuine turnover-rate normalisation); no
true S&P 500 index-level series (only constituent prices); free
yfinance-sourced data with known survivorship and identity-resolution
limitations already characterised in `YHD_SWEEP.md` / `BAD_OHLC_INVESTIGATION.md`.
None of these are new problems — Phase 4 simply inherits them and must design
around them rather than quietly assuming they're solved.

## 8. Leakage controls

A formal audit, not a generic warning. For every feature and every pipeline
stage, the test is: *"if the model predicts using information available through
end-of-day t, could anything dated after t have reached it, directly or
indirectly?"*

### 8.1 Feature-level (per §6 tables above)

Every feature is already tagged with a leakage assessment in its table row.
General rule enforced by construction: every feature query filters `date <= t`,
reusing `check_point_in_time_availability()`'s pattern of never touching a row
dated after `as_of_date` — this is Phase 3's own tested invariant, inherited
unchanged.

### 8.2 Universe-membership leakage

A feature or the target must never be computed using `ELIG(t+h)` (the
eligibility set at the *future* date) — only `ELIG(t)`. Concretely: candidate
selection at time `t` is drawn from `ELIG(t)`; the label looks at what happened
to those specific securities by `t+h`, but never re-queries who *was* eligible
at `t+h` to decide who to include. This also means index additions/removals
between `t` and `t+h` do not affect the sample — the security stays in the
Phase 4 sample based on `t`'s membership, exactly mirroring how Phase 3's
backtest engine already treats mid-period membership changes.

### 8.3 Corporate-action / restatement leakage

Splits, dividends, ticker renames, ticker reuse, ticker changes: already
point-in-time-safe by construction through `compute_split_adjusted`/
`compute_total_return`'s use in Phase 3, **provided §8.4's ratio-only rule is
followed**. Restated historical data (revised OHLC after a correction):
`price_data_quality` flags any known-suspicious rows; a feature computed over a
window containing a flagged row must record that fact (percentage of flagged
rows in-window) as an auditable field, not silently proceed as if the window
were clean.

### 8.4 Total-return backward-restatement rule (project-specific, see §6 preamble)

Restated in full here as the leakage control it is: **no feature may use a raw
`total_return` price level in isolation; only ratios/returns within a bounded
lookback window are permitted.** This is enforced by the feature-family tables
in §6 and must be enforced in code review / feature-generation tests before any
model touches these features.

### 8.5 Rolling-window, normalisation, and cross-sectional operations

- Rolling windows: window boundaries are `[t-lookback, t]` inclusive of `t`,
  never extending past it.
- Cross-sectional ranking/percentile features (§6.4): computed only over
  `ELIG(t)`, using only data available at `t` for every member — a rank feature
  is only as leakage-safe as its least-safe input.
- Normalisation/scaling: any global mean/stdev used to scale a feature must be
  computed **only from the training period** (or, for a walk-forward fold, only
  from that fold's training window) — never fit on validation or test data,
  and never fit on the full 2015–2023 range and then "applied backward."
- PCA / dimensionality reduction, imputation, feature selection: **not part of
  V1** (§6 keeps the V1 feature count small enough that none of these are
  needed yet). If introduced later, each must be explicitly fit only on that
  fold's training window and is subject to the same audit as any other feature.

### 8.6 The price-history-length finding — a mandatory diagnostic, not an assumed bias

The 200-seed PERMISSIVE report measured a raw 0.386 correlation between random-
selection frequency and a security's price-history length, and — after
splitting opportunity-count from selection-rate — a residual ~0.20 correlation
in the selection *rate* itself that opportunity count alone did not explain.
**This is not treated as an established bias here.** It is a real,
project-specific, quantified finding that must be carried into Phase 4 as a
mandatory diagnostic, run before any feature using a long lookback window
(12M return, 200d MA, 252d drawdown — anything with a multi-month history
requirement) is trusted:

1. For every V1 feature with a lookback ≥ 63 trading days, compute the same
   opportunity-count-vs-price-history-length and selection/inclusion-rate-vs-
   price-history-length correlations, but for *that feature's* eligibility
   requirement specifically (a feature with a 252-day lookback effectively
   creates its own, stricter sub-universe of "securities with enough history to
   compute this feature" — this could differ from and compound with the
   existing PERMISSIVE eligibility pattern).
2. If a model's predictive edge is concentrated in securities with unusually
   long price histories, that must be reported explicitly and investigated as
   a candidate explanation for the "edge" before any other conclusion is drawn
   — per §16, if removing the affected feature collapses the result, that is
   evidence for an artifact, not for genuine signal.
3. This diagnostic's conditional analysis — genuine selection bias vs.
   eligibility structure vs. another artifact — is run once real Phase 4 data
   exists to analyse; this specification only commits to running it, not to a
   conclusion about what it will show.

## 9. Walk-forward methodology

**No random train/test split, anywhere, for any reason.**

### 9.1 Structure

Expanding-window walk-forward: `TRAIN → VALIDATE → TEST(fold) → move forward →
retrain`. Initial training window starts at the beginning of available data
(2015-01) and grows forward; validation and test windows are fixed-length
blocks that slide forward as the training window expands. Exact window lengths
are a decision for the approval step (§23 lists this explicitly as an open
decision) — the illustrative 60/15/25 split in §5.3 is a starting point, not a
commitment.

### 9.2 Retraining frequency

Retraining cadence must be coarser than the label horizon to avoid training on
still-unresolved (still-overlapping) labels — e.g. at `h=3` months, a retrain
cannot use a label for a period that hasn't finished yet. Proposed: retrain no
more often than once per `h`-month block, not every single monthly rebalance.

### 9.3 Embargo / purge

At every train/validation and validation/test boundary, the last `h` months of
the earlier window must be **purged** (excluded from training) because their
labels' return windows extend into the following period — this is the standard
control for overlapping-label leakage across a chronological split boundary,
and it is mandatory here given §5's finding that overlap already meaningfully
reduces the effective sample.

### 9.4 Hyperparameter selection

Only from the validation period, via walk-forward validation folds — never from
the locked test period, and never by re-running against the test period and
picking the best-looking configuration after the fact (that would be exactly
the "researching toward the best historical result" failure mode §17 exists to
prevent).

### 9.5 Protecting the final test period

The locked test period's labels and predictions are not computed, inspected, or
touched until every modelling decision (feature set, model type,
hyperparameters, portfolio construction rule) has been frozen based on
train/validation evidence alone. This is enforced procedurally in §17 via an
immutable experiment registry, not just stated as an intention.

## 10. Baseline models

Required, in this order, before any ML model is considered:

1. **Historical mean return** — no model at all; ranks securities by their own
   trailing mean return. The floor every later model must clear.
2. **Momentum-only ranking** — ranks by the single best-performing feature from
   §6.1 on the validation set. Establishes how much of any later model's
   apparent edge is just momentum re-discovered.
3. **Linear regression / ridge** on the full V1 feature set, predicting
   `y_i(t,h)` directly.
4. **Logistic regression** on the full V1 feature set, predicting `z_i(t,h)`
   (outperformance probability).
5. **Elastic net** — as a feature-selection-aware variant of (3), useful given
   §5's constraint that the feature count must stay small relative to the
   effective sample size.

### 10.1 Ablation

Mandatory, on the baselines above (not deferred to the ML stage): PRICE ONLY →
PRICE + VOLATILITY → PRICE + VOLUME → PRICE + MARKET REGIME → FULL V1 FEATURE
SET, evaluated at each step on the validation period. This is what determines
where predictive information (if any) actually concentrates before a single
tree-ensemble model is trained, directly serving the research question in §1
rather than the eventual return number.

## 11. ML models

**Only after §10's baselines and ablation are complete and show validation-
period evidence worth pursuing further.**

In scope, conditional on that evidence: Random Forest, gradient boosting
(scikit-learn's `GradientBoostingRegressor`/`Classifier` as a
zero-new-dependency starting point; XGBoost/LightGBM only if their environment
footprint is confirmed acceptable at implementation time — not assumed here).

**Out of scope for Phase 4 entirely: LSTM, TCN, Transformer, or any neural
network architecture.** This is a hard exclusion, not a "defer pending
evidence" — see §5.3: roughly 9 independent time-blocks in the locked test set
at the primary horizon is not a regime where a temporal deep-learning model can
be trained, validated, and tested without simply overfitting the walk-forward
folds' idiosyncrasies. A temporal neural network's theoretical advantage over
carefully engineered rolling features is learning nonlinear temporal
dependencies directly from raw sequences — that advantage requires far more
independent sequences than this dataset has, at any horizon. Listed formally as
deferred future work in §22, conditional on substantially more data (more
years, more securities, or both) and on tree-ensemble evidence justifying
further complexity — not proposed for this phase under any configuration.

## 12. Portfolio construction

```
MODEL OUTPUT → cross-sectional ranking within ELIG(t)
             → candidate selection (top-K)
             → risk controls
             → position sizing
             → portfolio
             → Phase 3 cost model
             → realised return
```

**Pre-specified, not tuned after seeing test results:**

- Portfolio size: **20 (V1 default)** — matches Phase 3's existing
  `top_n_momentum`/`bottom_n_momentum`/`random_selection` size exactly, for a
  direct, apples-to-apples comparison against every existing Phase 3 benchmark.
  30 and 50 are declared robustness variants (§16), each a separate,
  pre-registered run — never a post-hoc "what if we'd used 30 instead" after
  seeing the 20-stock result.
- Weighting: **equal-weight (V1 default)** — again for direct comparability with
  existing benchmarks and because it introduces no additional tuned parameter.
  Confidence-weighted, volatility-adjusted, and capped-weight variants are
  declared robustness variants (§16), not V1.
- Selection rule: top-20 by predicted `y_i(t,h)` (equivalently, by predicted
  rank) within `ELIG(t)` at each retraining/rebalance point.

Implementation note tying back to the "don't reinvent the machinery" principle:
this is realised as a new `Strategy` implementation (e.g. `MLRankedSelection`)
conforming to Phase 3's existing strategy interface — the same interface
`BuyAndHold`/`TopNMomentum`/`RandomSelection` already implement — so Phase 3's
engine, cost accounting, and persistence run completely unchanged underneath it.

## 13. Transaction costs

Phase 3's existing cost model (`src/backtest/costs.py`, `config.yaml`'s
`backtest.costs` block) is reused exactly as-is. No second cost implementation
is created. Every model result is reported both **gross** and **net** of costs,
alongside: turnover, number of trades, total transaction costs, cost as a
percentage of gross return, cost per rebalance, and the gross-to-net performance
degradation — mirroring exactly the fields already computed and persisted for
every Phase 3 benchmark, so ML results sit in the same reporting format as
`buy_and_hold`/`top_n_momentum`/etc. without a new schema.

## 14. Evaluation metrics

**Prediction quality:** MAE/RMSE (regression baselines), Spearman rank
correlation (information coefficient) between predicted and realised `y_i(t,h)`
— **primary prediction-quality metric**, since the eventual use is a ranking,
not a point estimate; hit rate (sign agreement); calibration (for the logistic
baseline); top-decile-minus-bottom-decile spread.

**Portfolio quality (primary):** net-of-cost cumulative return, CAGR, Sharpe
ratio, Sortino ratio, maximum drawdown, Calmar ratio, annualised volatility,
turnover, transaction costs — all computed by Phase 3's existing report
generation, unchanged.

**Statistical robustness:** confidence intervals and significance tests must
respect §5's time-block structure (block bootstrap resampling by rebalance-date
block, not row-level i.i.d. resampling); performance-distribution comparison
against the full 200-seed random distribution (percentile rank of the model's
result within that distribution, not just mean-vs-mean); sensitivity analysis
per §16.

**Primary vs secondary:** Spearman IC (prediction quality) and net-of-cost
Sharpe ratio (portfolio quality) are primary — both must be evaluated relative
to the 200-seed distribution's own dispersion, not treated as free-standing
numbers. Everything else is secondary/diagnostic.

## 15. Benchmark methodology

Every model is evaluated through Phase 3's own point-in-time backtesting
engine — same universe construction, same cost model, same execution-date
logic — against, at minimum: Buy & Hold, Equal-weight (eligible universe),
Top-20 momentum, and the full 200-seed random-selection distribution (never
collapsed to a single number — percentile rank within the distribution, not
"beats the median," is the comparison). No comparison against any future ML
strategy is made (there isn't one yet), and this phase does not build or
compare against an FFT/other forecasting strategy either, per Phase 3's own
stated non-goals.

## 16. Robustness testing

Experiments to determine whether any observed effect survives: different
portfolio sizes (20/30/50), different weighting schemes, PERMISSIVE vs the
STRICT sanity-check (expecting a degenerate, uninformative result — §2), the
1M/6M secondary horizons, different sub-periods within 2015–2023 (explicitly
including at least one high-volatility regime, e.g. 2020, and one low-volatility
regime), feature removal (leave-one-family-out), model simplification (does the
elastic-net baseline capture most of the tree-ensemble's edge), and reasonable
transaction-cost sensitivity (±20% on `bid_ask_spread_bps`/`fx_cost_pct`, not a
new cost model).

**Named investigation trigger, not a generic caveat:** if removing one feature
or one feature family causes an apparent strategy edge to collapse, this is
treated as presumptive evidence of leakage or a data artifact (starting with
§8.6's price-history-length diagnostic) and must be investigated and reported
before the result is described as genuine predictive value.

## 17. Multiple-testing / overfitting control

**Experiment registry, mirroring Phase 3's `backtest_runs` reproducibility
pattern exactly:** every experiment — baseline or ML, exploratory or
confirmatory — gets an immutable ID, an immutable JSON config (feature set,
target definition, horizon, training/validation/test window boundaries, model
type, hyperparameters, random seed, universe policy, portfolio construction,
cost config), and a logged outcome. Identical config + seed must reproduce
identical predictions, identical holdings, identical performance — same
guarantee Phase 3 already provides for backtests, extended to cover the model
that generates the signal.

**Exploratory vs confirmatory, explicitly separated:** any number of exploratory
experiments may run against train/validation data. **Confirmatory experiments —
the ones whose result is reported as evidence for or against the research
question — must be pre-registered (config frozen, written down) before touching
the locked test set, and per §5.4, capped at a small number.** Concretely: no
more than one confirmatory run per declared robustness axis in §16, and no more
than roughly 5 confirmatory hypotheses total against the locked test set for
the primary 3-month horizon, consistent with the ~9 independent time-blocks
available there — re-derived exactly once §9's split is finalised, not
guessed at implementation time.

This is how "researching toward the best historical result" is prevented: the
test set is touched a small, fixed, pre-declared number of times, by
pre-declared configurations, and every touch — successful or not — is logged
permanently, not just the ones that looked good.

## 18. Experiment tracking / reproducibility

New tables, additive to the schema, following `backtest_runs`/`backtest_results`'s
existing pattern exactly (not a new paradigm):

- `ml_experiments` — one row per experiment: experiment_id, exploratory/
  confirmatory flag, target definition, horizon, feature_config_json,
  train/validation/test window boundaries, model_type, hyperparameters_json,
  random_seed, universe_policy (always `PERMISSIVE`), portfolio_construction_json,
  cost_config reference, created_at, code_version.
- `ml_predictions` — per (experiment_id, security_id, as_of_date): predicted
  value, predicted rank, realised label (once resolvable), truncated-label flag
  (§3.4).
- The resulting backtest of an experiment's predictions is a normal
  `backtest_runs` row (via the `MLRankedSelection` strategy, §12) — no
  duplicate results storage.

## 19. Computational requirements

CPU-only. Linear/logistic/elastic-net and tree ensembles (Random Forest,
gradient boosting) on a feature matrix bounded by §5's sample-size finding
(tens of thousands of rows, single-digit-to-low-double-digit feature count for
V1) train in seconds to low minutes on an ordinary personal PC — no GPU
requirement, no cluster. Feature values are cached per `(security, as_of_date)`
rather than recomputed per experiment (same caching principle Phase 3 already
applies to `build_eligible_universe`, extended to feature computation).
Experiments run sequentially by default; parallelisation across experiment IDs
is a later optimisation, not required for V1 given the small experiment count
§17 imposes.

## 20. Pre-registered success framework

Defined now, before any locked-test data is touched, addressing predictive
performance and net portfolio performance jointly. Exact numeric thresholds
below are proposed for your approval, not chosen because they flatter an
already-known result (none exists yet):

- **Failure:** primary-horizon Spearman IC on the locked test set is not
  reliably different from zero (its block-bootstrap confidence interval
  contains zero), **or** net-of-cost Sharpe ratio falls at or below the 50th
  percentile of the 200-seed random distribution.
- **Inconclusive:** IC's confidence interval excludes zero but is small in
  magnitude (e.g. |IC| < 0.03), **or** net Sharpe lands between the 50th and
  75th percentile of the random distribution — directionally interesting, not
  strong enough to act on, and explicitly not to be reframed as success by
  picking a more flattering metric after the fact.
- **Evidence of genuine predictive value:** primary-horizon IC's confidence
  interval excludes zero with |IC| ≥ 0.03, **and** net-of-cost Sharpe exceeds
  the 75th percentile of the 200-seed random distribution, **and** the result
  survives the leave-one-feature-family-out and price-history-length
  diagnostics in §16/§8.6 without collapsing.

These thresholds are a proposal, explicitly flagged in §23 as requiring your
sign-off before they're locked — changing them after seeing test results is
exactly the failure mode this section exists to prevent, so they must be
agreed now or not used as the success framework at all.

## 21. Known limitations

Carried forward from Phase 1–3, restated because Phase 4 is where they bite:
free yfinance data with known survivorship/data-quality issues; no sector data;
no point-in-time market cap; no true index-level series (only a derived
equal-weight proxy); STRICT policy unusable for cross-sectional modelling;
small effective sample size at longer horizons (§5); dividend/corporate-action
data quality varies (`corporate_action_quality` frequently `unverified`);
identity resolution imperfect (`identifier_quality` frequently `unresolved`,
though excluded from features per §6.6, it still limits how many securities
have a clean enough history to be usable at all).

## 22. Explicitly deferred features/models

Sector-relative features (blocked on unpopulated sector data). Cap-weighted
benchmarking (blocked on no market-cap data). True S&P 500 index-level market
features (blocked on no ingested index series). Float/short-interest liquidity
features (requires new data). Breakout, ATR-range, volatility-regime, and
bull/bear-regime features (deferred to post-V1 ablation, not blocked on data —
just not justified as first-pass inclusions). LSTM/TCN/Transformer/any neural
network architecture (§11 — hard exclusion for Phase 4, conditional deferral to
a hypothetical future phase only with substantially more data and tree-ensemble
evidence). PCA, automated feature selection, imputation beyond §3.4's
truncated-label rule (deferred until the V1 feature count actually needs it).
Confidence-weighted/volatility-adjusted/capped portfolio weighting, 30- and
50-stock portfolios (declared robustness variants, not V1 default — §12).

## 23. Phase 4 stop/go gate

**Specification → implementation** requires your explicit sign-off on:
(a) the walk-forward window boundaries (§9, currently illustrative), (b) the V1
feature list as tagged in §6, (c) the §20 success-framework thresholds, (d) the
confirmatory-experiment cap in §17. Nothing in §6–§18 is built before this
sign-off.

**Implementation → model training** requires: the leakage audit in §8 passing
its own tests (point-in-time query tests analogous to Phase 3's
`test_backtest_execution_timing.py` pattern), the §5 sample-size calculation
re-derived exactly against the real database and confirmed to still support the
planned experiment count, and the baselines in §10 implemented and running
before any ML model code is written.

**Training → locked-test evaluation** requires: every confirmatory experiment
config frozen and logged in `ml_experiments` *before* the test period is
touched, and explicit written confirmation from you that the locked test period
may now be evaluated — this is the one step in the whole pipeline that is
irreversible (once the test set has been looked at once, its evidentiary value
for that configuration is spent), so it is gated separately from ordinary
implementation approval.

No Phase 5 / live trading / Trading 212 execution work begins as part of, or as
a consequence of, Phase 4 — regardless of what Phase 4 finds.

---

## Appendix A: Dependency graph

```
Data (prices, corporate_actions, index_membership, securities)
        |
        v
Point-in-time universe: ELIG(t) = build_eligible_universe(t, PERMISSIVE)   [Phase 3, reused]
        |
        v
Features: §6 families, computed only from data <= t, ratio/return-only on
          total_return per §8.4, cached per (security_id, as_of_date)
        |
        v
Target: y_i(t,h) = r_i(t,h) - B(t,h)   [§3, uses ELIG(t) and total_return(t..t+h)]
        |
        v
Walk-forward split: TRAIN -> VALIDATE -> [locked] TEST, with purge/embargo   [§9]
        |
        v
Training: baselines (§10) -> ablation -> (conditional) tree ensembles (§11)
        |
        v
Prediction: ranked y_i(t,h) per security, per as_of_date   [persisted: ml_predictions]
        |
        v
Ranking -> top-K candidate selection   [§12]
        |
        v
Portfolio construction: equal-weight, K=20 (V1)   [§12, MLRankedSelection strategy]
        |
        v
Backtest: Phase 3 engine, unchanged -- costs, execution timing, accounting   [§13]
        |
        v
Evaluation: prediction-quality + portfolio-quality metrics, vs Phase 3
            benchmarks + 200-seed distribution   [§14, §15]
        |
        v
Robustness / ablation / leakage re-checks   [§16, §8.6]
        |
        v
Pre-registered success framework applied to the LOCKED TEST result only   [§20]
        |
        v
STOP -- report to user, no Phase 5 work begins   [§23]
```
