"""
Pluggable OHLCV price-source interface.

The dev environment used to build this had no network access to yfinance,
Stooq, or any real market-data API, so SyntheticDemoPriceSource generates
deterministic, clearly labeled placeholder OHLCV data. It exists so the
pipeline mechanics (idempotency, missing-data detection, quality flags,
reproducibility) can be tested end-to-end without real data. It is not
real market data and must never be used for actual research or
backtesting -- every row it produces is tagged with a source_id pointing
to a data_sources row named 'SYNTHETIC_DEMO (placeholder — not real
market data)'.

A real adapter (YFinancePriceSource, StooqPriceSource) implements the
same interface and can be substituted with no change to the ingestion
pipeline or schema.
"""
from dataclasses import dataclass
from typing import List, Optional
import hashlib
import datetime


@dataclass
class PriceBar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class PriceSourceAdapter:
    source_name: str

    def fetch(self, ticker: str, start_date: str, end_date: str) -> List[PriceBar]:
        raise NotImplementedError


def _business_days(start_date: str, end_date: str):
    d = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)
    while d <= end:
        if d.weekday() < 5:
            yield d.isoformat()
        d += datetime.timedelta(days=1)


class SyntheticDemoPriceSource(PriceSourceAdapter):
    """Deterministic synthetic OHLCV generator, seeded per-ticker so results
    are reproducible across runs (required for the idempotency test)."""

    source_name = "SYNTHETIC_DEMO (placeholder — not real market data)"

    def __init__(self, gap_tickers=None, no_data_tickers=None, start_price=100.0):
        # gap_tickers: {ticker: (gap_start, gap_end)} -> deliberately missing window
        self.gap_tickers = gap_tickers or {}
        self.no_data_tickers = no_data_tickers or set()
        self.start_price = start_price
        self.stats = {"total_retry_attempts": 0, "rate_limit_events": 0,
                       "transient_errors": 0, "permanent_errors": 0, "tickers_with_any_retry": 0}
        self.last_call_status = None
        self.last_call_error_detail = None
        self.last_call_attempts = 0

    def fetch(self, ticker: str, start_date: str, end_date: str) -> List[PriceBar]:
        self.last_call_attempts = 1
        if ticker in self.no_data_tickers:
            self.last_call_status = "SUCCESS_EMPTY_PROVIDER"
            self.last_call_error_detail = None
            return []  # simulates a known constituent with unavailable price history

        seed = int(hashlib.sha256(ticker.encode()).hexdigest(), 16) % (2**32)
        rng_state = seed
        def next_rand():
            nonlocal rng_state
            rng_state = (1103515245 * rng_state + 12345) % (2**31)
            return (rng_state / (2**31))  # deterministic pseudo-random in [0,1)

        gap = self.gap_tickers.get(ticker)
        price = self.start_price + (seed % 200)
        bars = []
        for d in _business_days(start_date, end_date):
            if gap and gap[0] <= d <= gap[1]:
                continue  # deliberate missing window
            drift = (next_rand() - 0.5) * 0.02
            price = max(0.5, price * (1 + drift))
            o = price * (1 + (next_rand() - 0.5) * 0.005)
            h = max(o, price) * (1 + next_rand() * 0.005)
            l = min(o, price) * (1 - next_rand() * 0.005)
            c = price
            v = 1_000_000 + int(next_rand() * 5_000_000)
            bars.append(PriceBar(date=d, open=round(o, 4), high=round(h, 4),
                                  low=round(l, 4), close=round(c, 4), volume=v))
        self.last_call_status = "SUCCESS_WITH_DATA" if bars else "SUCCESS_EMPTY_PROVIDER"
        self.last_call_error_detail = None
        return bars

    def fetch_dividends(self, ticker, start_date, end_date):
        return []  # synthetic source has no dividend simulation -- always empty, honestly

    def fetch_splits(self, ticker, start_date, end_date):
        return []  # synthetic source has no split simulation -- always empty, honestly


class YFinancePriceSource(PriceSourceAdapter):
    """
    Real price data via yfinance. Requires: pip install yfinance

    Not tested end-to-end during development since the build environment
    couldn't reach Yahoo's endpoints. Worth running locally and checking
    what happens for delisted tickers (MON, ABMD, AABA, AAMRQ) in
    particular -- expect several of these to return empty, which is a
    real finding about data availability, not a bug.

    Includes basic retry with exponential backoff on transient failures
    (network errors, rate-limit-shaped exceptions). This is not a full
    production-grade rate limiter -- it hasn't been tested against real
    sustained load at hundreds of tickers, only reasoned about. Treat it
    as better than a flat sleep, not as a solved problem.
    """

    source_name = "yfinance (Yahoo Finance, unofficial)"

    def __init__(self, auto_adjust=False, verbose=True, max_retries=3, base_delay=1.0):
        self.auto_adjust = auto_adjust
        self.verbose = verbose
        self.max_retries = max_retries
        self.base_delay = base_delay
        # Telemetry accumulated across all fetch() calls on this instance --
        # not returned by fetch() itself (that would break the
        # PriceSourceAdapter interface contract), but readable afterward via
        # source.stats. The original retry logic had no way to tell whether
        # or how often it actually fired, so this tracks that.
        self.stats = {
            "total_retry_attempts": 0, "rate_limit_events": 0,
            "transient_errors": 0, "permanent_errors": 0, "tickers_with_any_retry": 0,
        }
        # Per-call status, readable immediately after fetch() returns -- a
        # 4-state taxonomy (success/empty/transient/permanent). Set fresh on
        # every call.
        self.last_call_status = None
        self.last_call_error_detail = None
        self.last_call_attempts = 0
        try:
            import yfinance  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "yfinance is not installed. Run: pip install yfinance"
            ) from e

    def fetch(self, ticker: str, start_date: str, end_date: str) -> List[PriceBar]:
        import yfinance as yf
        import time
        import random

        end_inclusive = (
            datetime.date.fromisoformat(end_date) + datetime.timedelta(days=1)
        ).isoformat()

        df = None
        last_error = None
        ticker_had_retry = False
        attempts_made = 0
        for attempt in range(self.max_retries):
            attempts_made += 1
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_inclusive,
                    auto_adjust=self.auto_adjust,
                    progress=False,
                    threads=False,
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                err_str = str(e)
                is_likely_delisted = "possibly delisted" in err_str or "No timezone found" in err_str
                is_rate_limit = "429" in err_str or "rate limit" in err_str.lower() or "Too Many Requests" in err_str
                if is_likely_delisted:
                    self.stats["permanent_errors"] += 1
                    # Retrying won't help here -- it's a data-availability
                    # fact, not a transient failure. Fail fast.
                    break
                if is_rate_limit:
                    self.stats["rate_limit_events"] += 1
                else:
                    self.stats["transient_errors"] += 1
                if attempt < self.max_retries - 1:
                    ticker_had_retry = True
                    self.stats["total_retry_attempts"] += 1
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    if self.verbose:
                        print(f"  [yfinance] {ticker}: attempt {attempt+1} failed ({e}), "
                              f"retrying in {delay:.1f}s")
                    time.sleep(delay)
        if ticker_had_retry:
            self.stats["tickers_with_any_retry"] += 1

        # Classify this call into the 4-state taxonomy, readable via
        # self.last_call_status right after this returns.
        self.last_call_attempts = attempts_made
        if df is not None and hasattr(df, "empty") and not df.empty:
            self.last_call_status = "SUCCESS_WITH_DATA"
            self.last_call_error_detail = None
        elif last_error is not None:
            err_str = str(last_error)
            is_likely_delisted = "possibly delisted" in err_str or "No timezone found" in err_str
            self.last_call_status = "PERMANENT_FAILURE" if is_likely_delisted else "TRANSIENT_FAILURE"
            self.last_call_error_detail = err_str
        else:
            # No exception at all, but also no data -- the JNPR/SWN/FLT
            # pattern, confirmed via a bare yfinance call bypassing this
            # pipeline. Distinct from a caught delisted-shaped error: this
            # is Yahoo returning nothing with no error signal at all.
            self.last_call_status = "SUCCESS_EMPTY_PROVIDER"
            self.last_call_error_detail = None

        if df is None or (hasattr(df, "empty") and df.empty):
            if self.verbose:
                reason = f" ({last_error})" if last_error else ""
                print(f"  [yfinance] {ticker}: no data returned{reason} "
                      f"(likely delisted / not covered by Yahoo's free history)")
            return []

        if isinstance(df.columns, __import__("pandas").MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Defensive: strip any timezone info from the index before formatting.
        # Daily-interval yfinance data has been tz-naive in testing, and
        # .strftime() on a tz-aware Timestamp formats its own stored
        # wall-clock date rather than converting to UTC first, so this isn't
        # fixing an observed bug -- it's cheap insurance against a yfinance
        # version where that's untrue.
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

        bars = []
        for idx, row in df.iterrows():
            try:
                o, h, l, c, v = row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]
                if any(x is None or (isinstance(x, float) and x != x) for x in (o, h, l, c, v)):
                    continue
                bars.append(PriceBar(
                    date=idx.strftime("%Y-%m-%d"),
                    open=round(float(o), 4), high=round(float(h), 4),
                    low=round(float(l), 4), close=round(float(c), 4),
                    volume=float(v),
                ))
            except Exception:
                continue

        if self.verbose:
            print(f"  [yfinance] {ticker}: {len(bars)} bars fetched "
                  f"({start_date} to {end_date})")

        time.sleep(self.base_delay + random.uniform(0, 0.3))  # politeness delay with jitter
        return bars

    def fetch_dividends(self, ticker: str, start_date: str, end_date: str):
        """Returns list of (ex_date, amount) tuples. Separate from OHLCV
        fetch since yfinance exposes dividends via a different code path
        (yf.Ticker(...).dividends), not the download() OHLCV call."""
        import yfinance as yf
        try:
            t = yf.Ticker(ticker)
            div_series = t.dividends
        except Exception as e:
            if self.verbose:
                print(f"  [yfinance] {ticker}: dividend fetch failed ({e})")
            return []
        if div_series is None or div_series.empty:
            return []
        out = []
        for idx, amount in div_series.items():
            # Defensive: yfinance can return non-numeric values for some
            # tickers with complex/legacy corporate histories (PZE, SVU, TWX
            # all crashed here before this fix). Coerce first, skip cleanly
            # if it's not a real number, rather than letting one bad row
            # abort the entire ticker's dividend/split processing.
            try:
                amount = float(amount)
            except (TypeError, ValueError):
                if self.verbose:
                    print(f"  [yfinance] {ticker}: skipping malformed dividend value {amount!r} on {idx}")
                continue
            date_str = idx.strftime("%Y-%m-%d")
            if start_date <= date_str <= end_date and amount > 0:
                out.append((date_str, amount))
        return out

    def fetch_splits(self, ticker: str, start_date: str, end_date: str):
        """Returns list of (ex_date, ratio) tuples. ratio > 1 = forward
        split (e.g. 4.0 for 4-for-1), ratio < 1 = reverse split (e.g. 0.1
        for 1-for-10). Same code path as dividends (yf.Ticker().splits).
        Corporate actions used to be entered manually for a single security
        (AAPL); at any real scale, every security's splits need to come
        from here automatically."""
        import yfinance as yf
        try:
            t = yf.Ticker(ticker)
            split_series = t.splits
        except Exception as e:
            if self.verbose:
                print(f"  [yfinance] {ticker}: split fetch failed ({e})")
            return []
        if split_series is None or split_series.empty:
            return []
        out = []
        for idx, ratio in split_series.items():
            # Same defensive coercion as fetch_dividends() -- this exact
            # bug (str value in the splits Series) crashed PZE/SVU/TWX
            # during real ingestion.
            try:
                ratio = float(ratio)
            except (TypeError, ValueError):
                if self.verbose:
                    print(f"  [yfinance] {ticker}: skipping malformed split value {ratio!r} on {idx}")
                continue
            date_str = idx.strftime("%Y-%m-%d")
            if start_date <= date_str <= end_date and ratio > 0 and ratio != 1.0:
                out.append((date_str, ratio))
        return out
