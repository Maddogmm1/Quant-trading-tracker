# Phase 3 Specification: Point-in-Time Backtesting Engine

Status: DRAFT — awaiting approval. No implementation has begun.
Depends on: Phase 1 (data foundation, complete), Phase 2 (corporate actions, complete),
Stage 3 (full historical S&P 500 ingestion, closed AMBER).

## Guiding principles

These apply across every section below and take precedence over any local
convenience during implementation:

- No look-ahead bias, in any form, anywhere in the engine.
- No survivorship masquerading as predictive performance.
- No future metadata (identity, membership, corporate actions, or price) reachable
  from a historical decision.
- No retrospective threshold tuning — every quality/filter/cost parameter is fixed
  in configuration before any result is evaluated.
- No duplicated universe-construction logic — one authoritative interface, used by
  every benchmark and every strategy without exception.
- No silent data cleaning — every exclusion, substitution, or adjustment is
  auditable and attributable to a specific rule.
- No fabricated historical information (e.g. no retrospective market-cap
  reconstruction, no back-filling missing data with later observations).
- No hardcoded quality decisions hidden in Python — thresholds live in
  `config.yaml`, matching the design principle already established in
  `src/validation/checks.py::apply_universe_filters`.
- Every backtest run must be reproducible: identical inputs must produce identical
  outputs.
- Every exclusion must be auditable: for any security missing from an eligible
  universe on any date, there must be a queryable reason.

---

## 1. Purpose and scope

Phase 3 builds and validates the **backtesting engine** — the instrument used to
evaluate trading strategies against the Phase 1/2/Stage 3 historical dataset. The
engine itself is the deliverable of this phase, not any strategy's returns. Scope
covers: point-in-time universe construction, benchmark strategies, a pluggable
signal/strategy interface, execution and cost modeling, portfolio accounting,
performance measurement, and — most importantly — the validation suite that proves
the engine can be trusted before any predictive model is allowed near it.

The system under test throughout this phase is the **backtester**, not a strategy.
A strategy that "makes money" in an unvalidated backtester is not evidence of
anything.

## 2. Explicit non-goals

Phase 3 does **not** include:

- Machine learning models of any kind (no regression, no neural networks, no
  gradient boosting, no feature learning).
- Optimization of predictive features or hyperparameters.
- Ensemble models.
- Hierarchical regime/sector/stock models.
- The FFT forecasting work — only its future plug-in point is designed here (§9,
  §17 below via the strategy interface).
- Live trading, paper trading, or broker integration.
- Cap-weighted benchmarking (see §20 — no market-cap data exists yet).
- Optimizing transaction cost assumptions, rebalance frequency, or data-quality
  thresholds against observed performance. These are fixed before evaluation, full
  stop.

Phase 3 ends with a validation report and an explicit STOP (§23). Predictive
modeling is a separate, future-approved phase.

## 3. Data sources and existing Phase 2 outputs

The backtester consumes, read-only, the outputs already validated through Stage 3.
No new ingestion happens in Phase 3.

| Table / function | Role for the backtester |
|---|---|
| `securities` | Canonical security identity (`security_id`, `identifier_quality`, `active_flag`, `delisted_date`, `delisting_confidence`) |
| `index_membership` | Point-in-time S&P 500 membership claims (`raw_ticker`, `effective_date`, `removal_date`, `confidence`) |
| `prices` (`adj_type` = raw / split_adjusted / total_return) | Daily OHLCV, three parallel representations — raw is the source-of-truth for data-quality checks, `total_return` is the canonical return series (§10) |
| `corporate_actions` | Splits, dividends, reverse splits, spinoff-artifact flags (`corporate_action_quality`) |
| `identity_review_queue` | Unresolved ticker-reuse/identity-gap flags |
| `known_identifiers`, `known_ticker_renames` | CIK-based identity resolution and rename registry |
| `ingestion_attempts` | 4-state fetch outcome per ticker (used for provider-empty classification) |
| `src.validation.checks.check_point_in_time_availability()` | Existing point-in-time eligibility primitive — reused, not reimplemented (§5) |
| `src.validation.checks.reconstruct_universe()` | Existing point-in-time membership reconstruction — extended into the authoritative interface (§5), not duplicated |
| `src.validation.checks.apply_universe_filters()` | Existing config-driven price/liquidity/history filters — reused as one stage of the authoritative interface |
| `src.validation.stage_report.generate_survivorship_categorization()` | Existing 7-category classification — the pattern §15's per-run coverage report follows, scoped to `as_of_date` instead of globally |
| `src.ingestion.adjustments.compute_total_return_for_all()` | Canonical total-return calculation (§10) |
| YHD sweep results (`YHD_SWEEP.json`) | Security-level data-quality severity input to §6's policy (BMC/PTV/HAR/TIE and the rest of the 231-ticker set) |
| `config.yaml` (`universe_filters`) | Extended in Phase 3 with backtest-specific sections (§6, §12, §13, §19) |

## 4. Point-in-time data model

**The invariant:** for any decision made "as of" date `D`, no query anywhere in the
engine may read data dated after `D`, for any of: price, dividend, split, ticker
rename, index membership, identity resolution, or delisting metadata. This is
stricter than "don't use future prices" — metadata leakage (e.g. a delisting
record entered today revealing that a security won't survive, or an identity
resolution performed with current-day knowledge) is leakage too.

`as_of_date` is a first-class parameter threaded through every layer: universe
construction, data access, signal generation, and execution. There is no global
"current date" state anywhere in the engine.

The three `adj_type` price representations are used for distinct, non-overlapping
purposes and must not be conflated:
- `raw`: source-of-truth for data-quality checks (bad-OHLC detection, the YHD
  sweep) — never used directly for return calculation.
- `split_adjusted`: share-count-continuous price series — used for anything
  concerned with price level (e.g. minimum-price filters) but not total return.
- `total_return`: canonical return series (dividends reinvested) — used for all
  performance measurement (§14) and the canonical total-return benchmark. See §10.

## 5. Authoritative universe-construction interface

**This is the single most important architectural component of Phase 3.**

A new module, `src/backtest/universe.py`, exposes exactly one entry point for
determining eligibility:

```
build_eligible_universe(
    as_of_date: str,
    universe_definition: str = "SP500",
    data_quality_policy: dict,     # from config.yaml (§6)
    predeclared_filters: dict,     # from config.yaml universe_filters (existing)
) -> (eligible: list[security_id], exclusion_report: list[dict])
```

Internally this composes, rather than reimplements, existing functions:
1. `reconstruct_universe(conn, universe_definition, as_of_date, confidence_mode)` —
   who was a historical constituent as of this date, per real membership claims,
   with conflicting claims already excluded upstream.
2. `check_point_in_time_availability(conn, security_id, as_of_date, ...)` per
   candidate — has-any-data, sufficient-lookback, completeness, liquidity, all
   independently reported, never just a bare boolean.
3. `apply_universe_filters(conn, security_id, config, adj_type)` — the existing
   price/volume/history filters, unchanged.
4. The `data_quality_policy` (§6) — STRICT/PERMISSIVE/custom — determines how
   `identifier_quality`, `identity_review_queue` flags, and the new
   security-level YHD-severity flag affect eligibility.

**Architectural invariant (non-negotiable):** every benchmark (§7) and every
strategy (§8) obtains its candidate universe by calling
`build_eligible_universe()` and nothing else. No benchmark implementation may
query `index_membership` or `prices` directly to determine who's eligible. No
strategy may bypass this interface "for convenience." A benchmark or strategy may
apply *additional* selection logic *after* receiving the eligible set (e.g. "of
these eligible securities, pick the top 20 by momentum") — but it may never
independently reconstruct which securities were eligible in the first place.

The `exclusion_report` returned alongside `eligible` records, for every
non-eligible candidate, which specific check failed (no membership claim,
insufficient lookback, below liquidity threshold, identity unresolved under the
active policy, severe OHLC flag under the active policy, etc.) — this is what
feeds §15's per-run coverage report.

**Required test:** `test_benchmark_and_strategy_universe_construction_are_identical`
— given the same `as_of_date`, `universe_definition`, and `data_quality_policy`,
assert that a benchmark's internal universe call and a strategy's internal
universe call return byte-identical `eligible` lists. This is the direct
regression test for Addition 2.

## 6. Data-quality policies

Quality policy is a named, versioned block in `config.yaml`, never a Python
constant. At minimum, two presets:

```yaml
backtest:
  data_quality_policies:
    STRICT:
      min_completeness_pct: 0.95
      require_full_history: true        # excludes "partial_history" category
      exclude_unresolved_identity: true
      exclude_identity_review_flagged: true
      exclude_severe_ohlc_flagged: true   # security-level flag, see below
      severe_ohlc_bad_row_pct_threshold: null  # not used when exclude=true
    PERMISSIVE:
      min_completeness_pct: 0.80
      require_full_history: false        # partial_history allowed
      exclude_unresolved_identity: false  # included, but exclusion_report still records the flag
      exclude_identity_review_flagged: false
      exclude_severe_ohlc_flagged: false
      severe_ohlc_bad_row_pct_threshold: 0.10   # security-level flag surfaced, not auto-excluded, at this severity
```

**Security-level YHD-severity flag:** the YHD sweep found the row-level
`price_data_quality='suspicious'` flag insufficient on its own — BMC (17% of its
own rows bad) and CFC (12.5%) are qualitatively different from HAR (1.4%) or TIE
(0.6%), even though the row-level flag looks the same. Phase 3 adds a computed,
non-destructive security-level field: `bad_ohlc_row_pct` per security (bad raw
rows / total raw rows), read at universe-construction time, not stored as a
mutation to `securities`. Policies reference it via
`severe_ohlc_bad_row_pct_threshold`.

**By design:** BMC, PTV, HAR, and TIE are never hardcoded as
excluded. They are surfaced through this general mechanism like any other
security, and whether they're in or out is entirely a function of the active
`data_quality_policy` — this makes "run once under STRICT, once under PERMISSIVE,
compare" a first-class, config-only operation.

## 7. Benchmark definitions

All four benchmarks obtain their universe exclusively via §5's interface. All four
use the same rebalance dates, execution model (§9), and transaction cost model
(§12) as whatever strategy they're being compared against, in any given run —
apples to apples is enforced structurally, not by convention.

**A. Buy-and-hold.** Construct the eligible universe once, at the backtest start
date, equal-weighted. Hold without rebalancing (subject to forced exits on
delisting — handled per §10). This isolates "what if I'd just bought the starting
universe and done nothing," including the survivorship and identity exclusions
that applied on day one.

**B. Equal-weight S&P 500.** Rebalance to equal weight across the eligible
universe at every rebalance date (§13). This is the closest achievable proxy to
"the index" given the data available — see §20 for why it is *not* the S&P 500
index return, and why no cap-weighted benchmark exists in V1.

**C. Deterministic momentum benchmarks** (resolving Addition 3's ambiguity). Two
specific, separately-labeled benchmarks, not one vague one:

- **C1 — Top-N trailing momentum.** At each rebalance date, rank the eligible
  universe by trailing 12-month `total_return` series return (a fixed, long
  lookback — deliberately *not* tuned or matched to any parameter the eventual
  predictive model might use), equal-weight the top N (N fixed in config, e.g. 20).
- **C2 — Bottom-N trailing momentum (reversal control).** Same construction,
  bottom N by the same metric.

Rationale: momentum is a well-known, zero-fitting, fully deterministic effect —
using it as a control answers "does a trivial, publicly-known rule already
capture most of the value here?" before any model gets credit for anything.
Running both directions (C1 and C2) turns this into a genuine control rather than
a cherry-picked comparison: if a future predictive strategy can't beat C1, or
performs suspiciously similarly to C1, that's diagnostic. The lookback window (12
months) is fixed in `config.yaml` at spec-approval time and must never be
retuned to make a future model look better by comparison — if it's changed later,
that's a new benchmark, not the same one with better numbers.

**D. Random selection** — full methodology in §19, not repeated here. Same
universe/rebalance/portfolio-size/execution/cost parameters as whatever it's
compared against.

**No cap-weighted benchmark** exists in V1 (§20).

## 8. Strategy interface

A minimal, stable interface so any future signal — including the eventual FFT
work — plugs into the identical engine used for the benchmarks, with zero special
casing:

```
class Strategy:
    def generate_signal(self, as_of_date, eligible_universe, data_access) -> dict[security_id, weight]:
        """Must not access data dated after as_of_date. data_access is a
        point-in-time-bounded accessor, not a raw connection -- see §9."""
```

`eligible_universe` is always the output of §5's interface — a strategy never
constructs its own. `data_access` is a thin wrapper that enforces the §9 lookback
invariant structurally (queries dated after `as_of_date` raise, rather than
relying on the strategy author to remember). Benchmarks A–D are themselves trivial
implementations of this same interface, which is itself part of the validation
story: if the interface only worked for "real" strategies, that would be a design
smell.

The FFT predictor is explicitly out of scope for Phase 3 (§2) but must be
implementable against this interface without any change to the interface itself
when it's eventually approved.

## 9. Signal/execution timing

Three distinct, explicitly separated concepts, never conflated:

- **Signal date (T):** the date whose data is used to compute a decision. All
  data access for signal generation is bounded to `date <= T`.
- **Execution date:** the date the hypothetical trade occurs. For the initial
  benchmark engine, execution is at the **next available trading day's open**
  after T — i.e., a strategy computed from data through T's close trades at
  T+1's open. This is deliberately conservative and avoids the classic bug of
  using T's own close as both the signal input and the fill price (perfect
  foresight).
- **Holding period:** the interval a position is held, determined by rebalance
  frequency (§13).

`data_access` (§8) enforces `feature_data_end <= decision_date` structurally. A
dedicated test (§17) verifies `trade_execution_time >= decision_date` and that a
252-trading-day lookback strategy is rejected (not silently truncated or
back-filled) if fewer than 252 real observations exist as of T — missing history
is never filled from future observations.

## 10. Corporate-action handling

Splits, dividends, ticker changes, acquisitions, and delistings are already
handled at the ingestion layer (Phase 2) and are **not reimplemented** here.

- The canonical return series for all performance measurement is the existing
  `prices` table's `total_return` adj_type, produced by
  `compute_total_return_for_all()`. The backtester consumes this directly.
- Raw and derived representations stay separate per the existing schema — the
  backtester never re-derives split/dividend adjustments from raw prices itself.
- **Required tests** (extending the existing Phase 2 test coverage, not
  duplicating it): confirm the backtester correctly *interprets* a known
  total-return series around a known dividend and a known split (i.e., test the
  consumer, since the producer already has
  `test_total_return_reflects_dividend_reinvestment` and
  `test_reverse_split_scales_the_correct_direction` in `tests/test_phase1.py`).
- **Delisting/acquisition handling in the portfolio layer:** when a held position
  is delisted mid-holding-period, the position is force-closed at the last
  available price (or a configured delisting-return assumption if none exists —
  documented explicitly per security, never silently assumed to be zero or
  continued).
- If portfolio-layer accounting requires any calculation beyond consuming
  `total_return` directly (e.g. computing a period-over-period portfolio return
  from position-level total-return series), that calculation is implemented in
  `src/backtest/accounting.py` and explicitly documented as distinct from, and
  downstream of, the canonical security-level total-return calculation — never a
  second implementation of dividend/split math.

## 11. Portfolio accounting

Tracked at minimum, per rebalance date and cumulatively: cash, positions (by
security_id), position weights, shares, portfolio value, realized P&L, unrealized
P&L, dividends received, transaction costs paid, turnover.

**Required invariant, checked after every rebalance and every corporate-action
event:**
```
sum(position_value for all positions) + cash == portfolio_value   (within tolerance, e.g. 1e-6 relative)
```
A dedicated test constructs a scenario where this would silently drift (e.g. a
dividend received but not credited to cash, or a delisting not properly
force-closed) and asserts the invariant catches it.

## 12. Transaction-cost model

Implemented as an isolated module, `src/backtest/costs.py`, structurally separate
from signal generation and portfolio accounting, producing a
`gross_return -> costs -> net_return` waterfall reportable at both levels.

Cost components, and their V1 applicability for a UK-domiciled Trading 212 ISA
trading US-listed S&P 500 equities — each explicitly documented, not silently
defaulted:

| Component | V1 treatment | Rationale |
|---|---|---|
| Commission / platform fee | Configurable, default 0 | Trading 212 is commission-free on stock trades |
| FX cost | Configurable, default ~0.15% per conversion | GBP→USD conversion required for every US-stock trade in a GBP ISA — this is the most likely non-zero real cost and must not be silently omitted |
| Stamp duty / SDRT | 0, documented N/A | UK stamp duty applies to UK-listed equity purchases, not US-listed stocks |
| PTM levy | 0, documented N/A | Applies to LSE trades above a threshold, not applicable to US-listed equities |
| SEC/FINRA sell-side fees | Configurable, small default (e.g. a few basis points on sells) | US SEC Section 31 fee applies to US equity sales regardless of broker |
| Bid/ask spread | Configurable, default a conservative fixed bps assumption | No intraday/quote data exists to derive this empirically — must be an explicit, documented assumption, not fitted |
| Slippage | Configurable, default 0 for V1 (daily-bar backtest, no market-impact model) | Explicitly flagged as a known simplification (§20) |

All values are configuration, fixed before any run's results are evaluated (§6's
principle applies identically here) — never optimized to make a strategy look
better net of costs.

## 13. Rebalancing

Frequency is a config value: `daily`, `weekly`, or `monthly`, supported from V1.
Not optimized in Phase 3 — a single predetermined frequency is used per
validation run, chosen for testing convenience, not performance.

## 14. Performance metrics

Computed per run, at minimum: cumulative return, CAGR, annualized volatility,
Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio, win rate, turnover,
total transaction costs, number of trades, exposure (% invested vs. cash), cash
utilization. **Annual returns are reported as a full year-by-year breakdown, not
collapsed into a single headline CAGR** — a strategy that looks fine on CAGR but
is carried by one exceptional year needs to be visible as such.

## 15. Survivorship/coverage reporting

Every backtest run produces its own per-rebalance-date coverage report — the
global Stage 3 survivorship figure (34.2%, 2015–today window) is never used as a
proxy for what a specific backtest run actually saw, since coverage varies by
period, universe definition, and active `data_quality_policy`.

Per rebalance date, at minimum:
- historical eligible constituents (per real membership claims)
- securities with usable data (passed §5's availability check)
- securities excluded by the active quality policy, broken down by reason
  (provider-empty, identity-unresolved, partial-history, severe-OHLC — reusing
  the `generate_survivorship_categorization()` taxonomy, scoped to `as_of_date`
  rather than computed globally)
- final tradable universe size

This is a direct consumption of §5's `exclusion_report`, aggregated across the
run's rebalance dates — not a separately computed statistic that could drift from
what the engine actually did.

## 16. Look-ahead-bias prevention

Treated as a first-class test suite (`tests/test_backtest_lookahead.py`), not an
afterthought. At minimum:

1. **Future-price mutation test:** mutate prices dated after decision date T,
   re-run, assert all signals/portfolio states through T are byte-identical.
2. **Future-membership mutation test:** add a membership change effective after T,
   re-run, assert the universe at T is unchanged.
3. **Future-dividend mutation test:** mutate a dividend dated after T, assert
   portfolio decisions through T are unchanged.
4. **General future-data-cannot-alter-past test:** a parametrized version of the
   above across price/dividend/split/membership/identity tables in one pass, to
   catch any single table someone forgets to add a specific test for.
5. **Project-specific identity/ticker-substitution test (Addition 6, resolving
   the AMR → AAMRQ risk from BACKLOG.md item 2):** construct a synthetic security
   whose real ticker was `X` during an earlier historical period and was only
   ever labeled `Y` (its later-known ticker) starting from some later date —
   mirroring AMR/AAMRQ exactly. Assert:
   - `Y`'s price/identity data cannot be used to satisfy an availability check
     for a decision date that falls within `X`'s actual trading window.
   - Historical membership resolves against the correct underlying security
     identity, not whichever ticker string happens to appear in the source file
     for a given row.
   - A ticker change does not create *artificial* historical availability (i.e.
     the gap between `X`'s last real trading date and `Y`'s first data point is
     not silently bridged).
   - An identity that cannot be resolved is explicitly flagged (appears in the
     `exclusion_report` / `identity_review_queue`), never silently merged into
     continuity.
   - **The adversarial case:** construct the specific scenario where getting this
     wrong would produce an apparently valid, plausible-looking backtest result
     (e.g. a momentum strategy that appears to "catch" a rename-driven price
     discontinuity as a real return) — and confirm the correct system rejects or
     flags it rather than silently producing that number.

## 17. Synthetic validation tests

Constructed against known analytical answers — these must pass before any
result from real historical data is trusted, and are considered more important
than any attractive-looking strategy return:

1. Constant-price asset → zero return, exactly.
2. Asset with a known deterministic +10% return → engine reproduces 10.00%
   exactly (within floating-point tolerance).
3. Known 2-for-1 split, mid-holding-period → position value continuity verified
   (shares double, price halves, total value unchanged at the split boundary).
4. Known dividend → total-return reinvestment matches hand-calculated value.
5. Known ticker change (clean rename, CIK-linked, *not* the adversarial identity
   case in §16) → position continuity preserved correctly across the rename.
6. Delisted security mid-holding-period → forced closure per §10, portfolio
   accounting invariant (§11) still holds.
7. Missing-data period for one held security → explicit handling per policy (no
   silent fill), documented and asserted.
8. Portfolio with a known, hand-calculated transaction cost → net return matches
   exactly.
9. Signal generated on T, trade executed on T+1 → verified via a case where
   using T's close as the fill price would produce a different, wrong answer;
   assert the engine produces the T+1-open answer.

## 18. Reproducibility

Every run persists, at minimum: full configuration (quality policy, filters, cost
assumptions, execution assumptions, rebalance frequency), random seed(s) (§19),
start/end dates, universe definition, code/version identifier, results (all §14
metrics plus annual breakdown), and the §15 coverage report.

Proposed schema addition (new tables, `IF NOT EXISTS`, applied via the existing
`migrations.py` pattern — not a departure from how this project already handles
schema evolution):

- `backtest_runs` (run_id, created_at, config_json, random_seed,
  code_version, start_date, end_date, rebalance_frequency, universe_definition,
  data_quality_policy_name, cost_assumptions_json, execution_assumptions_json)
- `backtest_results` (run_id, metric_name, metric_value, period)
- `backtest_coverage` (run_id, as_of_date, eligible_constituents,
  usable_data_count, excluded_by_quality, provider_empty_count,
  identity_unresolved_count, partial_history_count, final_tradable_count)
- `backtest_positions` (run_id, as_of_date, security_id, shares, weight,
  position_value) — the audit trail behind §11's invariant check

**Required test:** run the same configuration twice, assert `backtest_results` is
identical between runs (byte-for-byte on every metric).

## 19. Random benchmark methodology (Addition 5, in full)

- One **fixed seed** for the headline, reported, reproducible run (e.g. seed
  `42`, or whichever value is fixed at spec-approval — the specific value
  doesn't matter, that it's fixed and documented does).
- A **separate, larger, predetermined set of seeds** for the distribution —
  proposed: **200 seeds**, fixed in `config.yaml` at spec-approval time, before
  any strategy has been evaluated against them. This number is chosen now and
  is not revisited later based on how favorable or unfavorable the resulting
  distribution turns out to be.
- Every seed's random-selection run uses the **identical** universe (via §5),
  rebalance dates, portfolio size, execution model (§9), and transaction cost
  model (§12) as whatever it's being compared against — the only thing that
  varies across the 200 runs is which eligible securities get picked.
- Reported: mean, median, standard deviation, min, max, and the full percentile
  distribution (at least 5th/25th/50th/75th/95th) of the chosen headline metric
  (e.g. CAGR or Sharpe) across the 200 seeds.
- A future strategy is compared against **where it falls in this distribution**
  (e.g. "above the 90th percentile of no-signal random outcomes"), not against a
  single random number — beating one lucky seed is not evidence of anything.
- Seeds are never selected or excluded post hoc because they produced a
  favorable or unfavorable comparison.

## 20. Known V1 limitations

Carried forward explicitly, not silently assumed fixed by reaching Phase 3:

- **No market-cap or shares-outstanding data exists** (Addition 1). No
  cap-weighted S&P 500 benchmark is possible in V1. Benchmark B (equal-weight) is
  the closest available proxy and is *not* equivalent to "the S&P 500 index" as
  commonly understood (e.g. SPY's actual return) — this must be labeled
  explicitly in every report the engine produces, not just in this document.
  Today's market caps are never used to retroactively weight historical
  portfolios.
- Survivorship bias: 34.2% of true historical constituents (2015–today window)
  are unrecoverable from free data — carried forward from Stage 3, now measured
  per-run via §15 rather than relied on as a single global figure.
- Entity identity remains genuinely unresolved for a subset of legacy tickers
  (backlog items 1–2), including the YHD/placeholder-identity pattern found
  across 231 tickers in the Stage 3 sweep — handled via configurable policy
  (§6), never silently assumed resolved.
- yfinance behavior under sustained real load at full 1,205-ticker scale remains
  validated only at Stage 1/2's ~50–200-ticker scale (~2.1–2.3s/security) — not a
  backtest-engine concern directly, but relevant if/when the universe needs
  re-fetching or extending.
- No intraday data — execution model (§9) and spread/slippage assumptions (§12)
  are necessarily daily-bar approximations, not empirically derived.
- S&P 400 and any full-universe (800–1,200+ stock) expansion beyond the current
  S&P 500 scope remains an interface stub only, per Stage 3's own backlog item 4.

## 21. Future extensions

- Historical market-cap/shares-outstanding ingestion, enabling a true
  cap-weighted benchmark — architecture in §5/§7 is deliberately extensible to
  add this as a new benchmark without touching the universe-construction
  interface's contract.
- The FFT predictor and any future ML strategy plug into §8's `Strategy`
  interface unchanged.
- Additional data-quality policy presets beyond STRICT/PERMISSIVE.
- Intraday/quote-level data for empirically-derived spread and slippage models.
- S&P 400 / broader universe expansion.

## 22. Phase 3 acceptance criteria

The engine is **not** considered validated merely because it runs without
errors. Explicit pass/fail:

- [ ] All §17 synthetic tests pass, producing the analytically-known correct
      result exactly (within floating-point tolerance).
- [ ] The §11 portfolio-accounting invariant holds across every synthetic and
      real-data test scenario, including delisting and dividend events.
- [ ] Point-in-time membership construction (§5) is verified correct against at
      least the real historical cases already characterized in Stage 3
      (AABA/AAMRQ, AAL's two membership periods, etc.).
- [ ] All §16 look-ahead tests pass, including the Addition-6
      identity-substitution adversarial case.
- [ ] The §5 "benchmark and strategy universe construction are identical" test
      passes.
- [ ] Transaction costs are correctly applied and reconcile exactly in the
      §12 gross → net waterfall for the §17 known-cost synthetic case.
- [ ] The §19 random-benchmark distribution is reproducible (same 200 seeds,
      same results, run twice) and correctly uses the shared universe/execution/
      cost pipeline.
- [ ] Identical configuration produces identical results on repeated runs (§18).
- [ ] A full validation report is produced covering all of the above, plus the
      four required benchmarks (§7) run once under STRICT and once under
      PERMISSIVE data-quality policy, with results and coverage reports for both.

## 23. STOP condition

Once the validation report (per §22) is produced, work stops. No predictive
modeling, no FFT integration, no strategy development beyond the four benchmarks
required for validation. The validation report is presented for explicit
approval before Phase 3 is considered complete or Phase 4 (predictive modeling)
begins.
