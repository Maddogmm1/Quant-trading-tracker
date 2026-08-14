# Data Pipeline Notes: Building the Ingestion Pipeline in Three Stages

This is a narrative summary of how the real-data ingestion pipeline (yfinance
OHLCV, dividends, splits, S&P 500 membership) was built and validated
incrementally: 50 securities, then 200, then the full ~1,206-security
historical universe. It replaces the individual per-stage README/report files
with the findings that are actually worth keeping.

Price source throughout: yfinance (Yahoo Finance, unofficial).

## Stage 1 — 50 securities, proving the mechanics

The first real-data run covered 50 hand-picked tickers spanning mega-caps
(AAPL, MSFT, GOOGL, AMZN, NVDA, META, JPM), large-caps, a handful of smaller
S&P 500 names, known ticker changes (ABC→COR, ABX→GOLD), known splits (TSLA
plus the mega-caps above), heavy-dividend payers (T, IBM, MMM, CVX), real
historically delisted names (MON, ABMD, AABA, AAMRQ), and one real spin-off
example (ABBV from Abbott, 2013).

Results: 157,845 raw price rows (2010-01-04 to 2023-12-29), 5 securities
resolved via known CIK, 0 identity-review flags, 0 bad-OHLC rows, 2,007
dividend events across 38 securities, and 24 split events. 4 securities
(MON, ABMD, AABA, AAMRQ) had zero price data — all real, sourced delistings
predating or excluded from Yahoo's coverage. Average time for a genuine
fresh fetch (not a resumed skip) was 0.915s/security, extrapolating to
roughly 3 minutes for 200 securities and 18 minutes for 1,200.

**Split reclassification.** Of the 24 detected "splits," 4 had non-round
ratios (PFE 1.054, LEN 1.017, T 1.324, IBM 1.046) landing exactly on
documented real corporate events unrelated to genuine share splits —
Pfizer/Viatris (2020-11-17), AT&T/WarnerMedia (2022-04-11), and similar.
yfinance's split field appears to pick up the residual price-adjustment
factor from spin-offs as a side effect. These were reclassified in place
(`classify_split_ratio()`, a whole-number-ratio heuristic with 3%
tolerance) and recorded as `corporate_action_quality='likely_spinoff_artifact'`
rather than genuine splits, without a full re-fetch — the reclassification
only touched local metadata already in hand.

## Stage 2 — 200 securities, where real-world messiness shows up

Stage 2 built directly on top of Stage 1's database (resumability meant only
the ~150 new tickers were fetched, not all 200), adding retry/rate-limit
telemetry and a Stage 1 vs Stage 2 comparison generator.

| Metric | Stage 1 | Stage 2 | Change |
|---|---|---|---|
| Securities | 50 | 200 | +150 |
| Raw price rows | 157,845 | 456,601 | +298,756 |
| Securities with zero price data | 4 | 52 | +48 |
| Bad OHLC rows flagged (raw) | 0 | 6 | +6 |
| Dividend events | 2,007 | 5,194 | +3,187 |
| Split events | 24 | 49 | +25 |
| Identity review flags | 0 | 8 | +8 |
| Avg seconds/security (genuine fresh fetch) | 0.915 | 2.126 | — |

At 200 real tickers, 26% (52/200) had no recoverable price data, consistent
with the survivorship-bias pattern already seen in Stage 1, now at larger
scale. `classify_split_ratio()` caught 11 new spinoff-artifact ratios beyond
the original 4, with no changes to the heuristic — a sign it generalizes.
One case, CPWR (two extreme-ratio "splits" in 2017 despite the company going
private in 2014), may be a real example of ticker reuse by an unrelated
entity, left as a flagged anomaly rather than acted on.

**Malformed data handling.** Three tickers (PZE, SVU, TWX) crashed ingestion
with `'>' not supported between instances of 'str' and 'int'` — yfinance
returned a non-numeric value in the splits/dividends Series for these
complex/long-delisted names. Fixed by coercing to float first and skipping
cleanly on failure rather than aborting the whole ticker, with a
`force_tickers` parameter added so previously-crashed tickers could be
selectively re-fetched without a full resume-skip.

**Timing-metric dilution.** The naive "avg seconds/security" metric was
silently diluted twice, by two different mechanisms: first, resume-skips
(near-zero-cost) pulled the average down and made a resumed run look faster
than it was; then, after fixing that, a second run's "freshly fetched" pool
turned out to consist almost entirely of chronic zero-data tickers (fast
failures), which also understated the real per-security cost by roughly 7x
versus the genuine ~2.3s/security figure. The report now separates
"including skips," "all fresh," and "fresh with data only," and
extrapolation uses only the last.

**Report-scoping bug.** `generate_stage_report()` originally queried the
database unconditionally rather than filtering by the requested ticker set —
invisible while each stage had its own database, but a real bug once stages
began sharing one growing database: regenerating a "Stage 1 report" after
Stage 2 had run would have silently reported Stage 2's full 200-security
numbers mislabeled as Stage 1. Fixed by resolving the `tickers` parameter to
`security_id`s and filtering explicitly; verified with a dedicated test and
a full Stage 1 → Stage 2 → regenerate-Stage-1-report simulation confirming
the regenerated report still shows 50, not 200.

**Bad-OHLC count inflation.** The initial "18 bad_ohlc_flagged" figure was
counting the same underlying raw-data problem up to 3x — once for the raw
row and once each for its derived split-adjusted/total-return copies, since
scaling by a factor doesn't fix an already-broken high<close relationship.
Fixed by scoping both the count and the sample to `adj_type='raw'` only.
The real, confirmed root causes in the flagged data: RYC shows a
low>close violation consistent with a frozen/stale quote, and PBG shows
high<low with high and low apparently swapped, a classic column-swap
error — genuine integrity problems in Yahoo's free data for obscure/legacy
tickers, not synthetic test artifacts. See `docs/BAD_OHLC_INVESTIGATION.md`
for the full investigation.

**Yahoo purges data for recently-retired tickers.** JNPR, SWN, and FLT
showed up in the "zero price data" set despite being fully active and
normally trading throughout the entire 2010-2023 window — unlike
MON/ABMD/AABA/AAMRQ, which were delisted years ago. A bare yfinance call,
bypassing the pipeline entirely, confirmed empty data for JNPR too, ruling
out a bug in this codebase. The distinguishing pattern: all three had their
ticker retired very recently relative to "now" (FLT renamed March 2024, SWN
merged October 2024, JNPR acquired July 2025). Working hypothesis: Yahoo's
backend purges historical data access for a ticker once it's retired, even
for data from well within the company's normal actively-traded years — a
forward-looking risk for any ongoing dataset built this way, not just a
historical backtesting limitation.

**Identity review queue duplication.** `flag_identity_review()` had no
duplicate check, so a persistent gap (AMD/PTC/FMC/PCG — real organic finds
with two disconnected ticker-usage periods and no confirming CIK) got
re-flagged on every ingestion run: 4 flags became 8 after a second run.
Fixed with a check-before-insert on `(ticker, period boundaries)`.

## Stage 3 — the full ~1,206-security historical universe

Two pieces of infrastructure were built before running Stage 3, both
motivated directly by Stage 2 findings:
1. An `ingestion_attempts` table with `SUCCESS_WITH_DATA` /
   `SUCCESS_EMPTY_PROVIDER` / `TRANSIENT_FAILURE` / `PERMANENT_FAILURE`
   classification per ticker per attempt — turning the JNPR/SWN/FLT finding
   from a manual-audit-only observation into queryable data.
2. `generate_survivorship_categorization()`: automated 7-category
   classification, replacing the individual per-ticker web research that
   worked at Stage 2's 52 zero-data tickers but wouldn't scale to Stage 3's
   full set.

**Schema migration gap.** `init_db()` had only ever applied `schema.sql` to
brand-new database files. When `ingestion_attempts` was added, the existing
Stage 1/2 database had no path to acquire the new table, producing a real
crash: `sqlite3.OperationalError: no such table: ingestion_attempts`. Fixed
with two mechanisms: every `CREATE TABLE`/`INDEX` in `schema.sql` now uses
`IF NOT EXISTS` and is always re-applied; and a new
`src/database/migrations.py` handles column-by-column `ALTER TABLE`
migrations for columns added to tables that already existed before the
column was introduced (`IF NOT EXISTS` on `CREATE TABLE` doesn't help here,
since it skips the whole statement if the table already exists). Verified
against a reconstruction of the real failure — an old-schema database with
real data, migrated in place, confirming new structures appear and original
data survives untouched.

**Results at full scale:**

| Metric | Value |
|---|---|
| Total securities | 1,205 |
| Raw price rows | 2,372,308 (2010-01-04 to 2023-12-29) |
| Securities with zero price data | 412 (34%) |
| Bad OHLC rows flagged (raw) | 1,227 |
| Identity review flags (unresolved) | 49 |
| Dividend events | 5,194 across 104 securities |
| Split events | 49 (same set as Stage 2 — no new fresh-fetch splits observed) |

Survivorship categorization breakdown (automated, 7 categories):
- `full_usable_history`: 729 (60.5%)
- `partial_history`: 28 (2.3%)
- `provider_empty`: 403 (33.4%) — the JNPR/SWN/FLT-style Yahoo-side gap,
  confirmed to be a much wider pattern than the original 3-ticker
  observation
- `identity_unresolved`: 45 (3.7%) — tickers with two disconnected usage
  periods and no confirming CIK (AMD, PTC, FMC, PCG, AAL, and 40 others)
- `no_historical_price_data`, `download_failure`, `other_unknown`: 0 each

Total wall time for the full fresh-fetch run was 144.5 seconds (794 of the
1,206 tickers were already present from Stages 1-2 and skipped). None of
the 412 freshly-fetched securities in this particular run returned data, so
the timing extrapolation for this run relies on the all-fresh average rather
than a fresh-with-data average, and is flagged as less reliable than the
Stage 1/2 extrapolations for that reason.

## Cross-cutting data-quality investigations

Two further investigations were run once bad-OHLC flags and Yahoo's
placeholder-identity ("YHD") tickers were understood better:
- `docs/BAD_OHLC_INVESTIGATION.md` — per-ticker root-cause analysis of the
  specific flagged bad-OHLC rows (TNB, CFC, RYC, PZE, PBG).
- `docs/YHD_SWEEP.md` — a full-universe sweep for securities carrying
  Yahoo's generic "YHD" exchange / numeric-shortName placeholder identity,
  and how much of the bad-OHLC problem they account for.
