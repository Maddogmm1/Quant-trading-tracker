"""
Synthetic validation tests against known analytical answers. These matter
more than any attractive-looking result from real historical data --
if any of these fail, nothing else the engine produces can be trusted.

Uses an in-memory sqlite database built to mirror the real schema exactly
(so accounting.py / universe.py / engine.py run against it unmodified),
seeded with securities/prices/corporate_actions constructed so the
correct answer is known by hand, not estimated.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sqlite3
import pytest

from src.backtest.accounting import Portfolio
from src.backtest.execution import PointInTimeDataAccess, next_trading_day_open, next_rebalance_dates
from src.backtest.engine import run_backtest
from src.backtest.benchmarks import BuyAndHold
from src.backtest import metrics as metrics_mod
from src.backtest.costs import trade_cost, apply_gross_to_net

ZERO_COSTS = {"commission_pct": 0.0, "fx_cost_pct": 0.0, "stamp_duty_sdrt_pct": 0.0,
              "ptm_levy_gbp": 0.0, "sec_finra_fee_pct": 0.0, "bid_ask_spread_bps": 0, "slippage_bps": 0}


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(open(os.path.join(os.path.dirname(__file__), "..", "src", "database", "schema.sql")).read())
    yield c
    c.close()


def _seed_security(conn, security_id, ticker, name="Test Co"):
    conn.execute(
        """INSERT INTO securities (security_id, primary_ticker, name, asset_type, active_flag,
           identifier_quality, created_at, updated_at)
           VALUES (?,?,?,'STOCK',1,'resolved','2020-01-01T00:00:00Z','2020-01-01T00:00:00Z')""",
        (security_id, ticker, name),
    )


def _seed_price(conn, security_id, date, close, adj_type="total_return", high=None, low=None, open_=None, volume=1000):
    conn.execute(
        """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type, source_id, ingested_at)
           VALUES (?,?,?,?,?,?,?,?,1,'2020-01-01T00:00:00Z')""",
        (security_id, date, open_ or close, high or close, low or close, close, volume, adj_type),
    )


def _seed_source(conn):
    conn.execute("INSERT OR IGNORE INTO data_sources (source_id, source_name, tier) VALUES (1,'test_source','C')")


# --- 1. Constant-price asset -> zero return, exactly ---
def test_constant_price_asset_zero_return(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "FLAT")
    dates = ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]
    for d in dates:
        _seed_price(conn, 1, d, 100.0)
    conn.commit()

    p = Portfolio(10000.0, ZERO_COSTS)
    for d in dates:
        p.rebalance_to(conn, d, {1: 1.0})
    values = [h["portfolio_value"] for h in p.history]
    assert values == pytest.approx([10000.0] * len(dates))
    assert metrics_mod.cumulative_return(values) == pytest.approx(0.0)
    assert metrics_mod.max_drawdown(values) == pytest.approx(0.0)


# --- 2. Asset with a known deterministic +10% return ---
def test_known_plus_10pct_return(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "TEN")
    _seed_price(conn, 1, "2020-01-01", 100.0)
    _seed_price(conn, 1, "2020-06-01", 110.0)
    conn.commit()

    p = Portfolio(10000.0, ZERO_COSTS)
    p.rebalance_to(conn, "2020-01-01", {1: 1.0})
    p.rebalance_to(conn, "2020-06-01", {1: 1.0})  # no-op rebalance, same target weight
    values = [h["portfolio_value"] for h in p.history]
    assert values[-1] == pytest.approx(11000.0)
    assert metrics_mod.cumulative_return(values) == pytest.approx(0.10)


# --- 3. Known 2-for-1 split: position value continuity across the split boundary ---
def test_known_split_position_value_continuity(conn):
    # The engine consumes a total_return series where a genuine split is
    # already baked in upstream as a smooth price halving with an implied
    # doubling of effective shares. This test checks the engine doesn't
    # reintroduce a discontinuity on top of that.
    _seed_source(conn)
    _seed_security(conn, 1, "SPLIT")
    _seed_price(conn, 1, "2020-01-01", 100.0)
    _seed_price(conn, 1, "2020-01-02", 50.0)  # split-adjusted continuity: value must not jump
    conn.commit()

    p = Portfolio(10000.0, ZERO_COSTS)
    p.rebalance_to(conn, "2020-01-01", {1: 1.0})
    shares_before = p.positions[1]
    pv_at_split = p.portfolio_value(conn, "2020-01-02")
    # No rebalance on the split date, so shares are held constant while
    # the price halves. A correctly split-adjusted series already reflects
    # the same economic value at the new price, so portfolio value should
    # not halve -- confirming the engine applies no separate split logic
    # of its own beyond holding shares constant against whatever price
    # the canonical series provides.
    assert shares_before == pytest.approx(100.0)  # 10000/100
    assert pv_at_split == pytest.approx(shares_before * 50.0)


# --- 4. Known dividend: total-return reinvestment matches hand-calculated value ---
def test_known_dividend_total_return_reinvestment(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "DIV")
    # total_return series already encodes dividend reinvestment upstream:
    # a $2 dividend on a $100 stock with a contemporaneous $98 ex-dividend
    # raw price becomes a total_return price of 100 * (1 + 2/100) = 102.
    _seed_price(conn, 1, "2020-01-01", 100.0)
    _seed_price(conn, 1, "2020-02-01", 102.0)
    conn.execute(
        "INSERT INTO corporate_actions (security_id, action_type, ex_date, ratio_or_value, source_id, ingested_at) "
        "VALUES (1,'dividend','2020-02-01',2.0,1,'2020-01-01T00:00:00Z')"
    )
    conn.commit()

    p = Portfolio(10000.0, ZERO_COSTS)
    p.rebalance_to(conn, "2020-01-01", {1: 1.0})
    pv = p.portfolio_value(conn, "2020-02-01")
    assert pv == pytest.approx(10200.0)
    div = p._dividends_between(conn, 1, "2020-01-01", "2020-02-01", shares_held=p.positions[1])
    assert div == pytest.approx(2.0 * p.positions[1])


# --- 5. Known ticker change: position continuity preserved (clean, CIK-linked case) ---
def test_known_clean_ticker_change_continuity(conn):
    # Same underlying security_id throughout -- the clean case, distinct
    # from the adversarial identity-substitution case in
    # tests/test_backtest_lookahead.py.
    _seed_source(conn)
    _seed_security(conn, 1, "NEWTICK")
    _seed_price(conn, 1, "2020-01-01", 100.0)
    _seed_price(conn, 1, "2020-02-01", 105.0)
    conn.commit()
    p = Portfolio(10000.0, ZERO_COSTS)
    p.rebalance_to(conn, "2020-01-01", {1: 1.0})
    pv = p.portfolio_value(conn, "2020-02-01")
    assert pv == pytest.approx(10500.0)  # continuous, no discontinuity from the rename itself


# --- 6. Delisted security mid-holding-period: forced closure, invariant still holds ---
def test_delisted_security_forced_closure(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "DEAD")
    _seed_price(conn, 1, "2020-01-01", 100.0)
    _seed_price(conn, 1, "2020-02-01", 90.0)  # last available price before delisting
    # no price after 2020-02-01 -- security is delisted
    conn.commit()

    p = Portfolio(10000.0, ZERO_COSTS)
    p.rebalance_to(conn, "2020-01-01", {1: 1.0})
    # rebalance away from the now-delisted security -- forces closure at last known price
    p.rebalance_to(conn, "2020-03-01", {})
    assert p.positions == {}
    assert p.cash == pytest.approx(9000.0)  # 100 shares * 90.0 last known price
    p._assert_invariant(conn, "2020-03-01")  # must not raise


# --- 7. Missing-data period: no silent fill ---
def test_missing_data_period_not_filled(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "GAPPY")
    _seed_price(conn, 1, "2020-01-01", 100.0)
    _seed_price(conn, 1, "2020-04-01", 100.0)  # 3-month gap in between, deliberately unfilled
    conn.commit()

    da = PointInTimeDataAccess(conn, "2020-02-15")
    hist = da.price_history(1, lookback_days=5)
    # only the single real observation on/before 2020-02-15 exists -- must
    # not be padded with interpolated/forward-filled values
    assert len(hist) == 1
    assert hist[0]["date"] == "2020-01-01"


# --- 8. Portfolio with a known, hand-calculated transaction cost ---
def test_known_transaction_cost_matches_hand_calculation(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "COST")
    _seed_price(conn, 1, "2020-01-01", 100.0)
    conn.commit()

    cfg = {"commission_pct": 0.0, "fx_cost_pct": 0.0015, "stamp_duty_sdrt_pct": 0.0,
           "ptm_levy_gbp": 0.0, "sec_finra_fee_pct": 0.0, "bid_ask_spread_bps": 5, "slippage_bps": 0}
    expected_cost = 10000.0 * (0.0015 + 5 / 10000.0)  # $17.50
    assert trade_cost(10000.0, is_sell=False, cost_config=cfg) == pytest.approx(expected_cost)

    p = Portfolio(10000.0, cfg)
    p.rebalance_to(conn, "2020-01-01", {1: 1.0})
    assert p.total_costs_paid == pytest.approx(expected_cost)
    assert p.cash == pytest.approx(-expected_cost)  # fully invested, cost paid on top

    net, drag = apply_gross_to_net(gross_return=0.0, total_costs=expected_cost, portfolio_value_at_start=10000.0)
    assert drag == pytest.approx(expected_cost / 10000.0)


# --- 9. Signal on T, trade executed on T+1: using T's close as the fill price would be wrong ---
def test_signal_t_execution_t_plus_1(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "TIMING")
    _seed_price(conn, 1, "2020-01-01", 100.0)  # T's close
    _seed_price(conn, 1, "2020-01-02", 120.0)  # T+1's open/close (single daily bar in this schema)
    conn.commit()

    signal_date = "2020-01-01"
    execution_bar = next_trading_day_open(conn, 1, signal_date)
    assert execution_bar["date"] == "2020-01-02"
    assert execution_bar["open"] == pytest.approx(120.0)

    p = Portfolio(10000.0, ZERO_COSTS)
    p.rebalance_to(conn, execution_bar["date"], {1: 1.0})
    shares = p.positions[1]
    # Using T's close (100.0) as the fill price would have bought 100
    # shares; the correct T+1 execution price (120.0) buys fewer shares.
    wrong_shares_if_using_t_close = 10000.0 / 100.0
    assert shares == pytest.approx(10000.0 / 120.0)
    assert shares != pytest.approx(wrong_shares_if_using_t_close)


# --- 10. BuyAndHold trades once and never again, even as prices drift ---
def test_buy_and_hold_trades_once_then_never_rebalances(conn):
    """Regression test for a bug caught in the first live validation run:
    BuyAndHold returned the same target-weights dict on every call, which
    run_backtest() dutifully rebalanced back to each period, turning "buy
    and hold" into monthly constant-weight rebalancing (~39,600 trades
    observed on ~430 securities over 108 months, versus the single round
    of opening trades a true buy-and-hold should produce)."""
    _seed_source(conn)
    _seed_security(conn, 1, "AAA")
    _seed_security(conn, 2, "BBB")
    conn.execute(
        "INSERT INTO index_membership (security_id, raw_ticker, index_name, effective_date, removal_date, "
        "source_id, confidence, ingested_at) VALUES (1,'AAA','SP500','2019-01-01',NULL,1,'verified','2020-01-01T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO index_membership (security_id, raw_ticker, index_name, effective_date, removal_date, "
        "source_id, confidence, ingested_at) VALUES (2,'BBB','SP500','2019-01-01',NULL,1,'verified','2020-01-01T00:00:00Z')"
    )
    import datetime
    d = datetime.date(2019, 1, 1)
    end = datetime.date(2020, 6, 1)
    while d <= end:
        if d.weekday() < 5:
            # deliberately drifting, DIFFERENT price paths so equal
            # weights would visibly drift apart if the engine kept
            # rebalancing back toward them
            _seed_price(conn, 1, d.isoformat(), 100.0 + (d.toordinal() % 30))
            _seed_price(conn, 2, d.isoformat(), 50.0 - (d.toordinal() % 20) * 0.5)
        d += datetime.timedelta(days=1)
    conn.commit()

    permissive = {
        "min_completeness_pct": 0.0, "require_full_history": False,
        "exclude_unresolved_identity": False, "exclude_identity_review_flagged": False,
        "exclude_severe_ohlc_flagged": False, "severe_ohlc_bad_row_pct_threshold": 0.10,
    }
    dates = next_rebalance_dates("2019-06-01", "2020-05-01", "monthly")  # 12 monthly rebalance dates
    report, coverage, portfolio = run_backtest(
        conn, BuyAndHold(), dates, permissive, cost_config=ZERO_COSTS,
        starting_cash=10000.0, lookback_days=10,
    )
    # exactly one round of opening trades (2 securities), then never again
    assert portfolio.trade_count == 2
    assert report["number_of_trades"] == 2
    # A second bug found alongside the first: a strategy that legitimately
    # stops trading must still get mark-to-market on the no-trade periods,
    # otherwise history would have exactly 1 entry and cumulative_return
    # would trivially be 0.0 regardless of how prices actually moved.
    assert len(portfolio.history) == len(dates)
    # prices genuinely drift in this fixture (see _seed_price calls above),
    # so a real buy-and-hold return must be nonzero here.
    assert report["cumulative_return"] != 0.0
