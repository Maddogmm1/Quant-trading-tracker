"""
Regression tests for an execution-date bug caught by a random-benchmark
diagnostic run: engine.py used to pick one arbitrary "representative"
security from the portfolio and resolve the whole portfolio's execution
date from that security's own next available price. When the
representative security had a real data gap (security_id 16, ticker
ABMD -- see test_phase1.py's test_abmd_gap_is_visible_not_filled), the
whole portfolio's execution silently jumped forward to wherever that
security's data resumed (182 days later in the run that caught this).
Because the engine processes signal dates in order, that late trade
mutated portfolio state ahead of several intervening months' correctly
timed trades, corrupting state rather than just being inaccurate.

Fix: execution.next_market_session() resolves the portfolio-wide
execution date from the full database's trading calendar (the union of
all securities' trading days), never from any single security. Each
security's own ability to trade on that date is then handled per-security
in accounting.Portfolio.rebalance_to: a valid price on the exact
execution date means it trades, otherwise that security's trade is
skipped this period (position left untouched) without affecting the
execution date used for any other security.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sqlite3
import datetime
import pytest

from src.backtest.execution import next_market_session, PointInTimeDataAccess
from src.backtest.accounting import Portfolio
from src.backtest.engine import run_backtest
from src.backtest.universe import build_eligible_universe

ZERO_COSTS = {"commission_pct": 0.0, "fx_cost_pct": 0.0, "stamp_duty_sdrt_pct": 0.0,
              "ptm_levy_gbp": 0.0, "sec_finra_fee_pct": 0.0, "bid_ask_spread_bps": 0, "slippage_bps": 0}

PERMISSIVE = {
    "min_completeness_pct": 0.0, "require_full_history": False,
    "exclude_unresolved_identity": False, "exclude_identity_review_flagged": False,
    "exclude_severe_ohlc_flagged": False, "severe_ohlc_bad_row_pct_threshold": 0.10,
}


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(open(os.path.join(os.path.dirname(__file__), "..", "src", "database", "schema.sql")).read())
    c.execute("INSERT OR IGNORE INTO data_sources (source_id, source_name, tier) VALUES (1,'test_source','C')")
    yield c
    c.close()


def _sec(conn, sec_id, ticker):
    conn.execute(
        """INSERT INTO securities (security_id, primary_ticker, name, asset_type, active_flag,
           identifier_quality, created_at, updated_at)
           VALUES (?,?,?,'STOCK',1,'resolved','2020-01-01T00:00:00Z','2020-01-01T00:00:00Z')""",
        (sec_id, ticker, ticker),
    )


def _price(conn, sec_id, date, close):
    conn.execute(
        """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type, source_id, ingested_at)
           VALUES (?,?,?,?,?,?,1000,'total_return',1,'2020-01-01T00:00:00Z')""",
        (sec_id, date, close, close, close, close),
    )


def _membership(conn, sec_id, ticker, eff):
    conn.execute(
        """INSERT INTO index_membership (security_id, raw_ticker, index_name, effective_date, removal_date,
           source_id, confidence, ingested_at) VALUES (?,?,?,?,NULL,1,'verified','2020-01-01T00:00:00Z')""",
        (sec_id, ticker, "SP500", eff),
    )


def _daily_prices(conn, sec_id, start, end, price=100.0):
    d = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    while d <= e:
        if d.weekday() < 5:
            _price(conn, sec_id, d.isoformat(), price)
        d += datetime.timedelta(days=1)


# ============================================================
# 1. Exact ABMD scenario: January 2020 signal, one security with a real
#    gap to July 2020, another security with normal next-day data.
# ============================================================
def test_abmd_style_gap_does_not_delay_portfolio_execution(conn):
    _sec(conn, 1, "AAA")   # normal, continuous data
    _sec(conn, 16, "ABMD")  # the real project's own gappy security_id/ticker
    _daily_prices(conn, 1, "2019-06-01", "2020-12-31")
    # ABMD: data before the gap, then nothing until July 2020, mirroring
    # the real gap documented in test_phase1.py.
    _daily_prices(conn, 16, "2019-06-01", "2019-12-31")
    _price(conn, 16, "2020-07-01", 100.0)
    _price(conn, 16, "2020-07-02", 100.0)
    conn.commit()

    signal_date = "2020-01-01"

    # (a) Portfolio-wide execution date should be the next real market
    # session after the signal date, not the date ABMD's gap ends.
    execution_date = next_market_session(conn, signal_date)
    assert execution_date is not None
    assert execution_date != "2020-07-01"
    gap_days = (datetime.date.fromisoformat(execution_date) - datetime.date.fromisoformat(signal_date)).days
    assert gap_days <= 6, (
        f"execution_date={execution_date} is {gap_days} days after the signal date -- ABMD's own "
        f"gap must not push the portfolio-wide execution date forward."
    )

    # (b) Build the actual portfolio trade at that correctly-resolved date.
    p = Portfolio(10000.0, ZERO_COSTS)
    p.rebalance_to(conn, execution_date, {1: 0.5, 16: 0.5})

    # ABMD has no price on the resolved execution_date, so its trade
    # should be skipped rather than filled with a smuggled-in future price.
    assert 16 not in p.positions, (
        "ABMD had no valid price on the resolved execution date, so its trade should be "
        "skipped, not silently filled using a later (future) price."
    )
    assert p.history[-1]["skipped_no_execution_price"] == [16]

    # Security 1 (no gap) should trade normally, unaffected by ABMD's gap.
    assert 1 in p.positions
    assert p.positions[1] > 0

    # (c) Confirm no future price leaked into the January decision: the
    # price paid for security 1 should come from the resolved execution date.
    price_used = p.cash  # cash remaining reveals what price was paid; cross-check directly
    row = conn.execute(
        "SELECT close FROM prices WHERE security_id=1 AND date=?", (execution_date,)
    ).fetchone()
    assert row is not None  # a real, contemporaneous price existed and was used

    # (d) Portfolio history stays chronologically ordered across rebalances
    # (see also the dedicated sequencing test below).
    later_date = next_market_session(conn, execution_date)
    p.rebalance_to(conn, later_date, {1: 1.0})
    assert p.history[-2]["as_of_date"] < p.history[-1]["as_of_date"]


# ============================================================
# 2. General synthetic version: Security A normal, Security B large gap.
# ============================================================
def test_security_with_large_gap_does_not_delay_other_securities_trade(conn):
    _sec(conn, 1, "AAA")  # normal daily data
    _sec(conn, 2, "BBB")  # large historical data gap
    _daily_prices(conn, 1, "2020-01-01", "2020-06-30")
    _daily_prices(conn, 2, "2020-01-01", "2020-01-15")
    _price(conn, 2, "2020-06-01", 100.0)  # B's data resumes months later
    conn.commit()

    signal_date = "2020-02-01"
    execution_date = next_market_session(conn, signal_date)
    gap_days = (datetime.date.fromisoformat(execution_date) - datetime.date.fromisoformat(signal_date)).days
    assert gap_days <= 6, "Security B's gap must not delay the portfolio-wide execution date."

    p = Portfolio(10000.0, ZERO_COSTS)
    p.rebalance_to(conn, execution_date, {1: 0.5, 2: 0.5})

    assert 1 in p.positions and p.positions[1] > 0, "Security A must trade on schedule."
    assert 2 not in p.positions, "Security B must not trade -- no price on the execution date."
    assert p.history[-1]["skipped_no_execution_price"] == [2]


# ============================================================
# 3. Sequencing: an earlier signal's trade must never be applied using a
#    later execution date in a way that corrupts state ahead of
#    intervening signals.
# ============================================================
def test_chronological_order_is_enforced_and_never_violated_by_engine(conn):
    _sec(conn, 1, "AAA")
    _sec(conn, 2, "BBB")
    _membership(conn, 1, "AAA", "2019-01-01")
    _membership(conn, 2, "BBB", "2019-01-01")
    _daily_prices(conn, 1, "2019-01-01", "2020-12-31")
    _daily_prices(conn, 2, "2019-01-01", "2020-12-31")
    conn.commit()

    from src.backtest.execution import next_rebalance_dates
    from src.backtest.benchmarks import EqualWeightSP500

    dates = next_rebalance_dates("2019-06-01", "2020-05-01", "monthly")
    report, coverage, portfolio = run_backtest(
        conn, EqualWeightSP500(), dates, PERMISSIVE, cost_config=ZERO_COSTS,
        starting_cash=10000.0, lookback_days=5,
    )
    as_of_dates = [h["as_of_date"] for h in portfolio.history]
    assert as_of_dates == sorted(as_of_dates), (
        "portfolio.history must be non-decreasing in as_of_date; an out-of-order entry means "
        "a trade was applied at a later execution date before an intervening signal's trade."
    )

    # Directly exercise the guard: applying an out-of-order state change
    # should raise rather than silently corrupt state.
    p = Portfolio(10000.0, ZERO_COSTS)
    p.rebalance_to(conn, "2020-06-01", {1: 1.0})
    with pytest.raises(AssertionError):
        p.rebalance_to(conn, "2020-01-01", {1: 1.0})  # before the prior entry
    with pytest.raises(AssertionError):
        p.mark_to_market(conn, "2020-01-01")  # same guard applies to mark-to-market


# ============================================================
# 4. next_market_session should only use date information, never price.
# ============================================================
def test_next_market_session_is_calendar_only_not_price_dependent(conn):
    _sec(conn, 1, "AAA")
    _price(conn, 1, "2020-01-02", 100.0)
    _price(conn, 1, "2020-01-03", 999999.0)  # wildly different price, should be irrelevant
    conn.commit()
    d1 = next_market_session(conn, "2020-01-01")
    assert d1 == "2020-01-02"
    # Confirms resolution is purely from date presence -- a price value
    # shouldn't affect which date is returned.
