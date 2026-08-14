
## Data-integrity and safety-net fixes

- `init_db()` now refuses to reset a database containing real
  (non-synthetic, non-test) price data unless `force=True` is passed
  explicitly. This prevents the synthetic demo script from silently
  destroying a real yfinance data pull. Tested both the refusal and the
  force path.
- `validate_ohlc()` was previously unproven — it had never actually fired
  on any run. Added a test that injects genuinely bad rows (high<low,
  negative price, negative volume) and confirms they get flagged
  `suspicious` while a genuinely good row does not.
- Added `delisting_confidence` and `delisting_source` columns to
  `securities`, and sourced MON (SEC DEFM14A) and ABMD (SEC 8-K) properly.
  While sourcing AABA and ABX, found both had been mis-recorded: their
  stored dates were actually the security's S&P 500 removal date, not
  when it stopped trading — a conflation bug. Corrected: AABA actually
  dissolved 2019-10-02 (not 2017-06-19), and ABX was never delisted at
  all — it changed its NYSE ticker to GOLD in 2019 (added to the rename
  registry, sourced via SEC 6-K) and kept trading continuously. AAMRQ's
  date was also wrong (stored as its 2003 index-removal date; corrected
  to its actual 2012-01-30 SEC delisting). This surfaced a deeper,
  unresolved finding: the ticker "AAMRQ" itself didn't exist during AMR's
  real 1996-2003 S&P 500 years (the real ticker was "AMR"; AAMRQ only
  came into use post-bankruptcy in 2012). This means the source
  membership file can label a security by a later-known ticker rather
  than the one actually in use historically — a structural risk flagged
  for awareness before scaling, not resolved here.
- `YFinancePriceSource` now has exponential backoff with jitter on
  transient failures, and fails fast (no wasted retries) on errors that
  look like genuine delistings rather than rate limits. This is
  meaningfully better than the flat sleep it replaces but has not been
  tested against real sustained load at hundreds of tickers.
- ISIN/CIK cross-referencing for security identity remains a larger,
  separate task — identity resolution is still ticker-only in most cases.
- 3 new tests added (26 total, all passing): safety-net refusal +
  force-override, and `validate_ohlc` catching real bad data. One
  existing test (`test_delisted_securities_not_deleted`) was corrected to
  match the fixed ABC/ABX data rather than the bug it had previously
  (accidentally) encoded as expected behavior.
