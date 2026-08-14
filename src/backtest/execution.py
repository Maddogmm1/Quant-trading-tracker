"""
Phase 3: point-in-time-bounded data access + signal/execution timing.

A strategy never gets a raw database connection -- it gets a
PointInTimeDataAccess bounded to a specific as_of_date, and any attempt to
query past that date raises. That way "no look-ahead" is enforced by the
code rather than relying on strategy authors to remember it.

Execution convention (fixed, not tunable per-strategy):
    signal computed from data through T's close
    trade executed at T+1's next available trading day's open
Using T's own close as both the signal input and the fill price would be
perfect-foresight bias, so this splits them.
"""
import datetime


class FutureDataAccessError(Exception):
    """Raised when code asks for data dated after the bound as_of_date."""


class PointInTimeDataAccess:
    def __init__(self, conn, as_of_date, adj_type="total_return"):
        self.conn = conn
        self.as_of_date = as_of_date
        self.adj_type = adj_type

    def _check_date(self, date):
        if date is not None and date > self.as_of_date:
            raise FutureDataAccessError(
                f"Requested data dated {date}, which is after as_of_date={self.as_of_date}."
            )

    def price_history(self, security_id, lookback_days, end_date=None):
        """Returns up to `lookback_days` rows of (date, close) with
        date <= as_of_date, most recent first. end_date, if given, must
        also be <= as_of_date."""
        end_date = end_date or self.as_of_date
        self._check_date(end_date)
        rows = self.conn.execute(
            """SELECT date, close FROM prices
               WHERE security_id=? AND adj_type=? AND date<=?
               ORDER BY date DESC LIMIT ?""",
            (security_id, self.adj_type, min(end_date, self.as_of_date), lookback_days),
        ).fetchall()
        return [dict(r) for r in rows]

    def trailing_return(self, security_id, lookback_days):
        """Total return over the trailing `lookback_days` observations,
        strictly bounded to as_of_date. Returns None if there isn't
        enough history rather than computing over a shorter window."""
        hist = self.price_history(security_id, lookback_days)
        if len(hist) < lookback_days:
            return None
        newest = hist[0]["close"]
        oldest = hist[-1]["close"]
        if not oldest:
            return None
        return (newest / oldest) - 1.0

    def latest_price(self, security_id):
        rows = self.price_history(security_id, lookback_days=1)
        return rows[0]["close"] if rows else None


def next_trading_day_open(conn, security_id, signal_date, adj_type="total_return"):
    """A single security's own next available trading day's open,
    strictly after signal_date. Useful for security-level questions, but
    not for resolving the portfolio-wide execution date -- see
    next_market_session() for why one security's data gap shouldn't
    determine when the whole portfolio trades."""
    row = conn.execute(
        """SELECT date, open FROM prices
           WHERE security_id=? AND adj_type=? AND date>?
           ORDER BY date ASC LIMIT 1""",
        (security_id, adj_type, signal_date),
    ).fetchone()
    return dict(row) if row else None


def next_market_session(conn, signal_date, adj_type="total_return"):
    """The portfolio-wide next trading session strictly after signal_date,
    resolved from the full database's trading calendar -- the earliest
    date, across every security, on which any security recorded a price.
    Returns None only if no security has a price after signal_date
    anywhere in the database (the true end of the dataset).

    This replaces an earlier, buggy approach that picked one arbitrary
    "representative" security from the portfolio and used its own next
    price as the execution date for everyone. That broke badly in
    practice: a real ~6-month gap in one security's price history (ABMD,
    security_id 16 in the test data -- see
    tests/test_backtest_execution_timing.py) silently delayed the entire
    portfolio's execution by 6 months whenever that security got picked
    as representative, and since the engine processes signal dates in
    order, the resulting stale trade landed in the middle of several
    later, correctly-timed trades' history entries -- state corruption,
    not just an inaccurate date.

    This doesn't introduce look-ahead bias because it only ever answers
    "was the market open on some date after signal_date" -- a calendar
    fact, not a price fact. Market calendars are published years in
    advance, so knowing the next session after a Friday close is the
    following Monday isn't privileged information the way a future price
    would be. The function never reads close/open/volume, only `date`.
    Per-security price availability on the resolved date is handled
    separately in accounting.Portfolio.rebalance_to, so a security
    missing data on the resolved session just sits out that period
    without affecting anyone else's execution date."""
    row = conn.execute(
        "SELECT MIN(date) AS d FROM prices WHERE adj_type=? AND date>?",
        (adj_type, signal_date),
    ).fetchone()
    return row["d"] if row and row["d"] else None


def next_rebalance_dates(start_date, end_date, frequency):
    """Deterministic rebalance date sequence. frequency: 'daily'|'weekly'|'monthly'.
    Uses calendar dates and doesn't check for actual trading data -- a
    non-trading date just yields no eligible universe/no execution rather
    than being silently shifted."""
    d = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)
    dates = []
    while d <= end:
        dates.append(d.isoformat())
        if frequency == "daily":
            d += datetime.timedelta(days=1)
        elif frequency == "weekly":
            d += datetime.timedelta(days=7)
        elif frequency == "monthly":
            # advance one calendar month
            if d.month == 12:
                d = d.replace(year=d.year + 1, month=1)
            else:
                d = d.replace(month=d.month + 1)
        else:
            raise ValueError(f"Unknown rebalance frequency: {frequency}")
    return dates


class Strategy:
    """Base interface every benchmark and every future predictive strategy
    implements identically. A strategy never constructs its own universe;
    eligible_universe always comes from
    src.backtest.universe.build_eligible_universe()."""

    name = "base_strategy"

    def generate_signal(self, data_access, as_of_date, eligible_universe):
        """Returns dict[security_id, weight] summing to <= 1.0 (remainder
        held as cash). Should use data_access, not a raw conn, for
        anything date-dependent."""
        raise NotImplementedError
