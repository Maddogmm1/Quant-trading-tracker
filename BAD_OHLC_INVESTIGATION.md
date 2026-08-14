# Bad-OHLC Investigation: TNB / CFC / RYC / PZE / PBG

## TNB
- DB name: None
- identifier_quality: unresolved, active_flag: 1
- delisted_date: None (None)
- total_raw_rows: 3004, bad_row_count: 73 (2.43%)
- bad date range: 2010-01-05 to 2022-01-28 (73 distinct dates)
- SUSPECTED FROZEN FIELD(S): [{'field': 'open', 'value': 0.6346, 'occurrences': 60, 'of': 73}]
- yfinance current identity: shortName=303598, longName=None, exchange=YHD, quoteType=MUTUALFUND
- yfinance re-fetch sample (independent, live): [{'date': '2010-01-05', 'open': 0.6346, 'high': 0.7115, 'low': 0.7115, 'close': 0.7115}, {'date': '2010-01-06', 'open': 0.6346, 'high': 0.7019, 'low': 0.7019, 'close': 0.7019}, {'date': '2010-01-08', 'open': 4800.0, 'high': 4800.0, 'low': 4800.0, 'close': 4800.0}, {'date': '2010-01-11', 'open': 0.6346, 'high': 0.7019, 'low': 0.7019, 'close': 0.7019}, {'date': '2010-01-12', 'open': 0.6346, 'high': 0.6827, 'low': 0.6827, 'close': 0.6827}, {'date': '2010-01-13', 'open': 0.6346, 'high': 0.7019, 'low': 0.7019, 'close': 0.7019}, {'date': '2010-01-14', 'open': 0.6346, 'high': 0.7019, 'low': 0.7019, 'close': 0.7019}, {'date': '2010-01-15', 'open': 4700.0, 'high': 5000.0, 'low': 4700.0, 'close': 4700.0}]

## CFC
- DB name: None
- identifier_quality: unresolved, active_flag: 1
- delisted_date: None (None)
- total_raw_rows: 2337, bad_row_count: 293 (12.54%)
- bad date range: 2010-01-06 to 2014-06-20 (293 distinct dates)
- yfinance current identity: shortName=3847602, longName=None, exchange=YHD, quoteType=MUTUALFUND
- yfinance re-fetch sample (independent, live): [{'date': '2010-01-06', 'open': 47.0, 'high': 46.5, 'low': 46.5, 'close': 46.5}, {'date': '2010-01-07', 'open': 46.5, 'high': 46.5, 'low': 46.5, 'close': 46.5}, {'date': '2010-01-08', 'open': 46.5, 'high': 46.5, 'low': 46.5, 'close': 46.5}, {'date': '2010-01-11', 'open': 46.5, 'high': 46.5, 'low': 46.5, 'close': 46.5}, {'date': '2010-01-12', 'open': 46.5, 'high': 46.5, 'low': 43.5, 'close': 43.5}, {'date': '2010-01-13', 'open': 43.5, 'high': 46.0, 'low': 43.5, 'close': 43.75}, {'date': '2010-01-14', 'open': 43.75, 'high': 45.75, 'low': 43.5, 'close': 44.0}, {'date': '2010-01-15', 'open': 44.0, 'high': 46.25, 'low': 43.5, 'close': 43.5}, {'date': '2010-01-19', 'open': 43.0, 'high': 46.0, 'low': 43.0, 'close': 43.25}, {'date': '2010-01-20', 'open': 43.25, 'high': 45.0, 'low': 43.0, 'close': 43.25}]

## RYC
- DB name: None
- identifier_quality: unresolved, active_flag: 1
- delisted_date: None (None)
- total_raw_rows: 1207, bad_row_count: 1 (0.08%)
- bad date range: 2011-10-31 to 2011-10-31 (1 distinct dates)
- yfinance current identity: shortName=None, longName=None, exchange=YHD, quoteType=MUTUALFUND
- yfinance re-fetch sample (independent, live): [{'date': '2011-10-31', 'open': 35.24, 'high': 35.24, 'low': 35.24, 'close': 35.13}, {'date': '2011-11-01', 'open': 34.97, 'high': 34.97, 'low': 34.97, 'close': 34.97}, {'date': '2011-11-02', 'open': 33.525, 'high': 33.525, 'low': 33.525, 'close': 33.525}]

## PZE
- DB name: None
- identifier_quality: unresolved, active_flag: 1
- delisted_date: None (None)
- total_raw_rows: 2109, bad_row_count: 1 (0.05%)
- bad date range: 2012-05-23 to 2012-05-23 (1 distinct dates)
- yfinance current identity: shortName=145886, longName=None, exchange=YHD, quoteType=MUTUALFUND
- yfinance re-fetch sample (independent, live): [{'date': '2012-05-23', 'open': 4.63, 'high': 4.645, 'low': 4.54, 'close': 4.685}, {'date': '2012-05-24', 'open': 4.63, 'high': 4.63, 'low': 4.45, 'close': 4.46}, {'date': '2012-05-25', 'open': 4.45, 'high': 4.68, 'low': 4.45, 'close': 4.675}]

## PBG
- DB name: None
- identifier_quality: unresolved, active_flag: 1
- delisted_date: None (None)
- total_raw_rows: 2009, bad_row_count: 4 (0.2%)
- bad date range: 2012-12-14 to 2013-01-31 (4 distinct dates)
- yfinance current identity: shortName=1090313, longName=None, exchange=YHD, quoteType=MUTUALFUND
- yfinance re-fetch sample (independent, live): [{'date': '2012-12-14', 'open': 6.45, 'high': 6.25, 'low': 6.57, 'close': 6.28}, {'date': '2012-12-17', 'open': 6.45, 'high': 6.57, 'low': 6.25, 'close': 6.28}, {'date': '2012-12-18', 'open': 6.15, 'high': 6.29, 'low': 5.9, 'close': 5.9}, {'date': '2012-12-19', 'open': 5.85, 'high': 5.89, 'low': 5.5, 'close': 5.5}, {'date': '2012-12-20', 'open': 5.5, 'high': 5.5, 'low': 5.04, 'close': 5.2}, {'date': '2012-12-21', 'open': 5.19, 'high': 5.47, 'low': 5.1, 'close': 5.25}, {'date': '2012-12-24', 'open': 5.25, 'high': 5.25, 'low': 5.25, 'close': 5.25}, {'date': '2012-12-26', 'open': 5.25, 'high': 5.25, 'low': 5.25, 'close': 5.25}, {'date': '2012-12-27', 'open': 5.55, 'high': 5.59, 'low': 5.27, 'close': 5.42}, {'date': '2012-12-28', 'open': 5.38, 'high': 5.44, 'low': 5.22, 'close': 5.28}]
