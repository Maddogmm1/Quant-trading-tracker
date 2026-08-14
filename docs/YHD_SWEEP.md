# Full-Universe YHD / Placeholder-Identity Sweep

A full-universe scan for securities carrying Yahoo's generic "YHD" exchange
code and/or a numeric (rather than real company-name) `shortName` — a
pattern that turned out to correlate strongly with data-quality problems
identified during the bad-OHLC investigation (`docs/BAD_OHLC_INVESTIGATION.md`).

## Method

Every security in the full ~1,206-ticker universe was checked via yfinance
for `exchange == 'YHD'` or a numeric-looking `shortName`, and cross-referenced
against existing price data, membership data, identity-review flags, and
bad-OHLC flag counts already in the database.

## Findings

- 231 of 1,205 securities (19.2%) are affected (YHD exchange or numeric
  shortName).
- Of those, 69 have actual price data, 231 have membership data, and only 4
  carry an identity-review flag — meaning the vast majority of affected
  tickers aren't otherwise distinguishable as suspicious without this check.
- 23 of the affected tickers carry at least one bad-OHLC flag.
- 81 of the 231 are relevant to the 2010-2023 backtest period specifically.
- Of the 1,227 bad-OHLC rows flagged system-wide, 1,222 (99.6%) come from
  YHD-flagged tickers — this placeholder-identity pattern accounts for
  essentially all of the bad-OHLC problem in the full universe, not just a
  coincidental subset of it.

## Top affected tickers by bad-OHLC row count

| Ticker | Exchange | shortName | Price rows | Bad-OHLC rows | Relevant to 2010-2023 |
|---|---|---|---|---|---|
| BMC | YHD | 1804073 (numeric) | 2,439 | 417 | Yes |
| CFC | YHD | 3847602 (numeric) | 2,337 | 293 | No |
| PTV | YHD | 980658 (numeric) | 1,602 | 261 | Yes |
| TNB | YHD | 303598 (numeric) | 3,004 | 73 | No |
| CIN | YHD | 985646 (numeric) | 2,033 | 46 | No |
| CNG | YHD | (none) | 2,572 | 33 | No |
| HAR | YHD | 906601 (numeric) | 2,285 | 31 | Yes |
| CMX | YHD | 2880468 (numeric) | 2,834 | 30 | No |
| TIE | YHD | 215914 (numeric) | 2,339 | 14 | Yes |
| KRI | YHD | 445461 (numeric) | 2,371 | 6 | No |
| PBG | YHD | 1090313 (numeric) | 2,009 | 4 | Yes |

A long tail of ~220 further YHD-flagged tickers each carry 0-2 bad-OHLC
rows, mostly with zero price data at all (delisted or never fetchable).

## Conclusion

`quoteType == 'MUTUALFUND'` combined with a `YHD` exchange code and a
numeric `shortName` is Yahoo's generic fallback identity for securities it
can no longer properly identify — largely long-delisted or historically
renamed names. This placeholder identity is a strong, cheap predictor of
both "no price data available" and, for the minority that do have data,
"elevated risk of raw OHLC integrity violations." It was not used to filter
or exclude securities automatically (that decision belongs to the
backtesting engine's configurable data-quality policy, not the ingestion
layer), but it materially informed where to focus the bad-OHLC root-cause
investigation.
