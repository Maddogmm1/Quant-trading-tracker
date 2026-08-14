# Possible Future Extensions

Open items assessed as "fixable but not urgent at 22-ticker scale" or
"inherent to the free-data constraint." Worth revisiting before or during
a jump to the full ~800-1,200 stock universe.

## 1. Security identity resolution is ticker-only

`src/universe/identity_resolution.py` resolves via CIK where known (5
real, sourced entries: MON, ABC/COR, AABA, AAMRQ, ABX) and flags
suspicious ticker-reuse gaps for review rather than silently merging, but
there is no bulk/automated CIK resolution at scale — 5 entries is nowhere
near comprehensive coverage for 800-1,200 securities. One real organic
flag surfaced this way: AAL's two S&P 500 membership periods (1996-97,
2015-24) are 18 years apart with no confirming CIK, sitting unresolved in
`identity_review_queue` and worth investigating before scaling.

## 2. Ticker-rename registry doesn't scale, and source data can mislabel historical tickers

Two related problems:
- The `known_ticker_renames` registry only grows via manual, per-case
  verification (proven during Phase 1: 2 real corrections out of ~265
  losses moved the bias number by 0.3pp — not a shortcut).
- The `fja05680/sp500` membership file sometimes labels a security by a
  later-known ticker, not the one actually trading during its real
  historical membership window (confirmed 2/2 in a spot-check). AMR Corp
  traded as "AMR" (not "AAMRQ") until its 2012 SEC delisting; Yahoo
  traded as "YHOO" (not "AABA") until June 2017 — "AABA" only existed
  from the exact day it left the S&P 500, meaning it was never really the
  operative ticker during the actual membership window at all.

Fetching real price data by directly using tickers from the membership
file risks silently returning zero data for periods where the real
contemporaneous ticker was different — this looks identical to a genuine
data gap but isn't one. Before mass price-fetching at scale, it would be
worth cross-checking a sample of historical tickers against a secondary
source (e.g. SEC EDGAR `formerNames` + ticker history) to estimate how
common this is, rather than trusting the membership file's ticker column
as ground truth.

## 3. yfinance retry/backoff untested at real scale

Exponential backoff with jitter was added and reasoned through, but only
run against ~22-23 sequential tickers. There is no evidence yet on
whether it survives real sustained load (rate limiting, IP throttling) at
800-1,200 tickers — worth testing against a mid-size batch (e.g.
100-150 tickers) before committing to it at full scale.

## 4. S&P 400 and full-universe scale are both untested

S&P 400 membership ingestion is an interface stub only
(`SP400MembershipParser`), by design — a deliberate decision not to
bulk-scrape SPDJI press releases blindly. The "~800-1,200 stock universe"
figure has never been tested against anything beyond the 22-25 ticker
S&P 500 test subset. Both were always planned as a later, deliberate
step.

## 5. Spin-offs and rights issues are not handled

Phase 2 (database and corporate-action handling) implemented splits,
reverse splits, ticker changes, and dividends/total-return properly, each
validated against realistic data. Spin-offs and rights issues were
explicitly scoped out as rarer and structurally more complex (a spin-off
creates a new security with a cost-basis split from the parent; a rights
issue changes share count without a clean price-ratio adjustment). None
of the current 22-25 test securities had one, so this has never been
exercised even conceptually — worth revisiting if a security in the
working universe actually has one, rather than building speculative
handling ahead of a real example.

## 6. yfinance split field occasionally contains spinoff artifacts (fixed)

Real Stage 1 ingestion (50 securities, 2010-2023) surfaced 24 "splits"
from `yf.Ticker().splits`, 5 of which were non-round ratios (1.017,
1.046, 1.054, 1.324, 1.998) landing exactly on documented real spinoff
dates: Pfizer/Viatris (2020-11-17), IBM/Kyndryl (2021-11-04), AT&T/
WarnerMedia (2022-04-11). yfinance's split field appears to pick up the
residual price-adjustment factor from spinoffs as a side effect — these
are not genuine share-count splits.

Fixed by adding `classify_split_ratio()` (whole-number-ratio heuristic,
3% tolerance) to `src/ingestion/adjustments.py`. Genuine splits still
feed `compute_split_adjusted()` normally; artifacts are recorded for the
audit trail with `corporate_action_quality='likely_spinoff_artifact'`,
excluded from the adjustment math, and the security gets flagged via the
existing unsupported-corporate-action mechanism. The classification is a
heuristic, not a certainty — a genuine unusual split ratio could in
principle be misclassified as an artifact, so it's worth spot-checking
classified-as-artifact events at larger scale.

## 7. Stage 2 groundwork: retry telemetry, resumability, comparison tooling

Three gaps were closed based on Stage 2's own requirements before
building Stage 2 itself:
1. `YFinancePriceSource` tracks retry attempts, rate-limit-shaped errors,
   transient errors, and permanent (delisted-shaped) errors as
   instance-level stats, surfaced in the report.
2. `run_stage()` skips re-fetching any security that already has raw
   price rows in the target database, making a killed/restarted run
   genuinely resumable — tested with a simulated partial-run-then-resume
   scenario (10-of-20 tickers, then resumed to 20), confirming zero
   duplicates and exactly the right tickers skipped/fetched.
3. `compare_stages.py` generates a Stage 1 vs Stage 2 comparison table
   from saved JSON reports.

## 8. Real Stage 2 findings and fixes

Running Stage 2 (200 real securities) surfaced real bugs/gaps beyond what
Stage 1's 50 exercised:
- 3 tickers (PZE, SVU, TWX) crashed with "'>' not supported between
  instances of 'str' and 'int'" — yfinance returned a non-numeric value
  in the splits/dividends Series for these complex/long-delisted names.
  Fixed by coercing to float first and skipping cleanly (with a log line)
  on failure, rather than aborting the whole ticker. Reproduced via a
  mocked malformed pandas Series, with 2 new permanent tests.
- The naive "avg seconds/security" timing metric was silently diluted by
  resume-skips (46 near-zero-cost skips pulled Stage 2's reported average
  from ~2.3s down to ~1.2s, which looked like a speedup but wasn't real).
  The report now separates "including skips" from "fresh fetch only"
  explicitly, and extrapolation uses the fresh-fetch-only number.
- Added a `force_tickers` parameter to `run_stage()` — resumability's
  skip check only looks at whether raw price rows exist, so a ticker
  whose price fetch succeeded but whose dividend/split processing
  crashed (exactly PZE/SVU/TWX's situation) would be wrongly skipped on a
  naive resume. `force_tickers` lets specific tickers be force-refetched
  regardless.
- At 200 real tickers, 26% (52/200) had zero recoverable price data,
  consistent with the survivorship-bias pattern established since
  Phase 1, now at larger scale. `classify_split_ratio()` correctly caught
  11 new spinoff-artifact ratios beyond the original 4, validating that
  it generalizes. One interesting case (CPWR, two extreme-ratio "splits"
  in 2017 despite Compuware going private in 2014) may be a real example
  of ticker reuse by an unrelated entity — worth investigating at Stage 3
  rather than acted on immediately.

## 9. Report-scoping bug, found and fixed before it corrupted a comparison

`generate_stage_report()` previously queried the database
unconditionally — `SELECT COUNT(*) FROM securities` with no ticker
filter. This was invisible while each stage had its own database, but
stages share one growing database for resumability (Stage 2 builds on
Stage 1's DB). Regenerating a "Stage 1 report" after Stage 2 had run
would have silently reported Stage 2's full 200-security numbers,
mislabeled as Stage 1, making the Stage 1 vs Stage 2 comparison
meaningless. Caught before any corrupted report was generated or
trusted.

Fixed: every query now resolves the `tickers` parameter to `security_id`s
first and filters by them explicitly. Verified two ways: (1) a dedicated
test building a DB with both "Stage 1" and "Stage 2 only" securities,
confirming a Stage-1-scoped report only sees the Stage 1 subset; (2) a
full end-to-end simulation (Stage 1 -> Stage 2 on top -> regenerate
Stage 1 report) confirming the regenerated report still shows 50, not
200. Also fixed `run_stage1_real.py`, which always called `reset=True` —
that made sense as the first-ever script but not once Stage 2 had grown
the shared DB, so it now uses `reset=False` (query/ensure-present, not
reset). Added `retry_failed.py`, which force-retries specific tickers
(bypassing the resume-skip) for exactly the PZE/SVU/TWX situation.

## 10. Timing extrapolation bias, second occurrence

After the resume-skip dilution fix (item 8), a subtler version of the
same class of bug surfaced: on a resumed run where nearly everything with
real data was already ingested, the "freshly fetched" pool can end up
almost entirely composed of the chronic zero-data securities (fast
failures, retried every run since they never accumulate rows to trigger
the skip check). Observed directly: 52 "freshly fetched" == 52
"zero-price-data" securities exactly — meaning the reported 0.34s/
security average reflected only fast no-data lookups, not a single
genuine multi-year OHLCV+dividend+split fetch. Extrapolating from it
would have estimated ~6.8 minutes for 1,200 securities, roughly 7x too
optimistic versus the real ~2.3s/security cost.

Fixed: timing now splits "freshly fetched" into with-data vs zero-data-
fast-failure subsets, and extrapolation uses only the with-data average,
falling back with an explicit warning if a run has zero fresh successful
fetches to measure from.

## 11. Bad-OHLC count inflated by derived representations

The "18 bad_ohlc_flagged" figure and its sample were counting the same
underlying raw-data problem up to 3x — once for the raw row, once each
for its derived split_adjusted/total_return copies (scaling by a factor
doesn't fix an already-broken high<close relationship, so the violation
propagates). Confirmed by inspecting the flagged sample: PZE/RYC/PBG rows
appeared 2-3 times each, at different price scales matching split-
adjustment.

Fixed: both the count and the sample are now scoped to `adj_type='raw'`
only, since that's the actual source-data integrity question — derived
series inheriting the same flaw isn't a new finding. See
`docs/BAD_OHLC_INVESTIGATION.md` for the underlying data-quality
investigation.

## 12. Identity review queue was not idempotent (fixed)

`flag_identity_review()` had no duplicate check, unlike every other
ingestion function in this pipeline. Identity resolution runs for every
ticker on every stage run regardless of price-skip status, so a real,
persistent gap (AMD/PTC/FMC/PCG, all genuine organic finds) got
re-flagged and re-inserted on every run — 4 flags became 8 after a
second run. Fixed with a check-before-insert on
`(ticker, period boundaries)`, matching the pattern used everywhere else.
Added `dedupe_identity_review_queue.py` as a one-off cleanup for
databases that already accumulated duplicates before this fix.

## 13. Yahoo appears to purge data for recently-retired tickers

Stage 2's audit found JNPR, SWN, FLT among the "zero price data" set —
but unlike MON/ABMD/AABA/AAMRQ (delisted years ago), all three companies
were fully active and normally trading throughout the entire 2010-2023
window. Confirmed via a bare yfinance call, bypassing the pipeline
entirely, that this is an upstream Yahoo/yfinance limitation rather than
a bug in this codebase.

The distinguishing pattern: all three had their ticker retired very
recently relative to "now" (FLT renamed March 2024, SWN merged October
2024, JNPR acquired July 2025), not years ago. Working hypothesis: Yahoo's
backend appears to purge historical data access for a ticker once it's
retired, even for data from well within the company's normal
actively-traded years.

This means the "zero price data" category isn't limited to long-dead
companies — it can affect any currently-active security that later gets
delisted, acquired, or renamed, retroactively, even years after the fact.
This is a forward-looking risk for ongoing maintenance of this research
universe, not just a historical backtesting limitation: the zero-price-
data rate should be expected to creep upward over time for any dataset
built this way, as more currently-active securities eventually get
delisted or renamed and retroactively lose queryability.

## 14. Stage 3 groundwork: status taxonomy and automated survivorship categorization

Stage 2's zero-price-data audit (item 13) was done manually — individual
web research per ticker, which doesn't scale to Stage 3's 1,206. Two
things were built before running Stage 3:
1. An `ingestion_attempts` table with SUCCESS_WITH_DATA/
   SUCCESS_EMPTY_PROVIDER/TRANSIENT_FAILURE/PERMANENT_FAILURE
   classification, persisted per ticker per fetch attempt — previously
   this distinction existed only as a print statement, not queryable
   data.
2. `generate_survivorship_categorization()`: automated 7-category
   classification using the new status data plus existing sourced fields
   (delisting records, identity review queue), tested against
   reconstructions of the real patterns found in Stage 2 (JNPR-style
   silent-empty, MON-style sourced-delisted, clean data, identity-
   unresolved).

## 15. Schema migration gap (fixed)

`init_db()` only ever applied `schema.sql` to brand-new database files.
When `ingestion_attempts` was added for Stage 3, any existing database
(including the real Stage 1/2 database) had no path to acquire the new
table, causing a hard crash the moment Stage 3 tried to use it:
`sqlite3.OperationalError: no such table: ingestion_attempts`.

Fixed with two mechanisms, both required:
1. Every `CREATE TABLE`/`INDEX` in `schema.sql` now uses `IF NOT EXISTS`,
   and `init_db()` always applies the schema script — safe and
   idempotent, not just for new databases.
2. `src/database/migrations.py`: explicit column-by-column `ALTER TABLE`
   migrations for columns added to a table that already existed before
   the column was introduced (e.g. `securities.cik`) — `IF NOT EXISTS`
   on `CREATE TABLE` does nothing for this case, since it skips the whole
   statement if any table with that name is present, columns and all.

Verified against a reconstruction of the real failure: built a database
with the old, pre-`ingestion_attempts` schema and real data in it, then
ran the fixed `init_db()` against it (`reset=False`, exactly as the stage
scripts do). Confirmed the new table appears, new columns appear,
original data is completely untouched, and the exact query that crashed
the real run now succeeds. 2 new permanent tests. This class of bug
(schema evolving faster than the migration path) is inherent to the
project's design of long-lived, incrementally-reused databases across
stages — worth remembering for any future schema change: it migrates
automatically now, but a new migration type beyond "new table" or "new
column" (e.g. changing a column's type or constraint) will need a new
case added explicitly to `migrations.py`.
