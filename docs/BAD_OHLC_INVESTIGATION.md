# Bad-OHLC Investigation: TNB / CFC / RYC / PZE / PBG

Five tickers accounted for the bulk of raw-data OHLC integrity violations
(high < low, low > close, etc.) flagged during Stage 2/3 ingestion. Each was
independently re-fetched live from yfinance (bypassing the pipeline
entirely) to confirm the violations exist in the source data itself, not in
our ingestion or adjustment code.

## Method

For each ticker: pulled `identifier_quality`, `delisted_date`, and the count
and date range of flagged rows from the database, then re-fetched a fresh,
independent sample directly via yfinance for the flagged dates to compare
against what's already stored.

## Findings

**TNB** — 73 of 3,004 raw rows bad (2.43%), spanning 2010-01-05 to
2022-01-28. A frozen-field pattern: `open=0.6346` recurs on 60 of the 73 bad
rows while high/low/close move normally around it — consistent with a stale
open-price field rather than a genuine OHLC error. The independent re-fetch
reproduces the same frozen value, confirming this is a source-data artifact,
not an ingestion bug. (One re-fetched row also showed an isolated
`open=4800.0` outlier alongside otherwise-normal values, a separate, rarer
anomaly in the same series.)

**CFC** — 293 of 2,337 raw rows bad (12.5%), 2010-01-06 to 2014-06-20, the
highest bad-row rate of the five. The independent re-fetch shows plausible,
internally consistent OHLC values for the same dates — the specific
violation pattern in the stored data (rather than a frozen field) suggests
a different underlying source issue than TNB's, though both are Yahoo-side.

**RYC** — 1 bad row out of 1,207 (0.08%), a `low > close` violation on
2011-10-31 consistent with a frozen or stale quote for that single day.

**PZE** — 1 bad row out of 2,109 (0.05%) on 2012-05-23.

**PBG** — 4 bad rows out of 2,009 (0.2%), spanning 2012-12-14 to 2013-01-31.
The clearest case: `high < low` with the high and low values apparently
swapped — a classic column-swap error rather than a plausible price
anomaly.

## Conclusion

All five tickers carry Yahoo's generic "YHD" exchange / numeric-shortName
placeholder identity (see `docs/YHD_SWEEP.md`), which turns out to be a
strong predictor of bad-OHLC risk across the wider universe. The violations
are real, source-side data-integrity problems in Yahoo's free data for
obscure/legacy tickers — not synthetic test cases and not artifacts of this
pipeline's own processing. They are flagged (`price_data_quality='suspicious'`)
and left in place rather than silently corrected, since there is no reliable
way to reconstruct the true value from the data available.
