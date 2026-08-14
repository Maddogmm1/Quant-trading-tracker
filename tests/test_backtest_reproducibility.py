"""
Reproducibility and persistence tests.

Covers two things: (1) save_positions computes weight/position_value
correctly against a hand-calculable portfolio -- a regression test for a
bug caught during the first live validation run, where save_positions
originally inserted weight=NULL and crashed against the NOT NULL
constraint; (2) two runs with identical config and random_seed produce
identical backtest_results rows.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sqlite3
import pytest

from src.backtest.accounting import Portfolio
from src.backtest.engine import run_backtest
from src.backtest.execution import next_rebalance_dates
from src.backtest.benchmarks import EqualWeightSP500
from src.backtest.reproducibility import save_run, save_positions, load_results

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


# --- 1. save_positions computes weight and position_value correctly ---
def test_save_positions_weight_and_value_match_hand_calculation(conn):
    _sec(conn, 1, "AAA")
    _sec(conn, 2, "BBB")
    _price(conn, 1, "2020-01-01", 100.0)
    _price(conn, 2, "2020-01-01", 50.0)
    conn.commit()

    p = Portfolio(10000.0, ZERO_COSTS)
    p.rebalance_to(conn, "2020-01-01", {1: 0.6, 2: 0.4})
    # hand-calculated: 6000/100 = 60 shares of AAA, 4000/50 = 80 shares of BBB
    assert p.positions[1] == pytest.approx(60.0)
    assert p.positions[2] == pytest.approx(80.0)

    run_id = save_run(
        conn, run_label="test", config={"x": 1}, random_seed=None, start_date="2020-01-01",
        end_date="2020-01-01", rebalance_frequency="monthly", universe_definition="SP500",
        data_quality_policy_name="PERMISSIVE", strategy_name="test_strategy", cost_config=ZERO_COSTS,
        execution_config={}, report={"cumulative_return": 0.0, "annual_returns": {}}, coverage_report=[],
        created_at="2020-01-01T00:00:00Z",
    )
    save_positions(conn, run_id, p.history)

    rows = conn.execute(
        "SELECT security_id, shares, weight, position_value FROM backtest_positions WHERE run_id=? ORDER BY security_id",
        (run_id,),
    ).fetchall()
    assert len(rows) == 2

    aaa, bbb = rows[0], rows[1]
    assert aaa["shares"] == pytest.approx(60.0)
    assert aaa["position_value"] == pytest.approx(6000.0)  # 60 shares * 100.0
    assert aaa["weight"] == pytest.approx(0.6)              # 6000 / 10000 portfolio value

    assert bbb["shares"] == pytest.approx(80.0)
    assert bbb["position_value"] == pytest.approx(4000.0)   # 80 shares * 50.0
    assert bbb["weight"] == pytest.approx(0.4)


# --- 2. A position with no resolvable price is skipped, not fabricated ---
def test_save_positions_skips_position_with_no_price_at_date(conn):
    _sec(conn, 1, "AAA")
    _price(conn, 1, "2020-01-01", 100.0)
    conn.commit()

    p = Portfolio(10000.0, ZERO_COSTS)
    p.rebalance_to(conn, "2020-01-01", {1: 1.0})
    # Manually inject a phantom position for a security with NO price row
    # at all, to exercise the "no resolvable price" branch deliberately.
    p.history[-1]["positions"] = dict(p.history[-1]["positions"])
    p.history[-1]["positions"][999] = 5.0

    run_id = save_run(
        conn, run_label="test2", config={}, random_seed=None, start_date="2020-01-01", end_date="2020-01-01",
        rebalance_frequency="monthly", universe_definition="SP500", data_quality_policy_name="PERMISSIVE",
        strategy_name="test", cost_config=ZERO_COSTS, execution_config={},
        report={"cumulative_return": 0.0, "annual_returns": {}}, coverage_report=[], created_at="2020-01-01T00:00:00Z",
    )
    save_positions(conn, run_id, p.history)

    rows = conn.execute("SELECT security_id FROM backtest_positions WHERE run_id=?", (run_id,)).fetchall()
    sec_ids = {r["security_id"] for r in rows}
    assert 1 in sec_ids
    assert 999 not in sec_ids  # skipped, since it has no price -- not written with a fabricated value


# --- 3. Two identical-config/seed runs produce identical persisted results ---
def test_identical_config_and_seed_produce_identical_results(conn):
    _sec(conn, 1, "AAA")
    _sec(conn, 2, "BBB")
    _membership(conn, 1, "AAA", "2019-01-01")
    _membership(conn, 2, "BBB", "2019-01-01")
    import datetime
    d = datetime.date(2019, 1, 1)
    end = datetime.date(2020, 6, 1)
    while d <= end:
        if d.weekday() < 5:
            _price(conn, 1, d.isoformat(), 100.0 + d.toordinal() % 10)
            _price(conn, 2, d.isoformat(), 50.0 + d.toordinal() % 7)
        d += datetime.timedelta(days=1)
    conn.commit()

    dates = next_rebalance_dates("2019-06-01", "2020-05-01", "monthly")

    def _do_run(label):
        report, coverage, portfolio = run_backtest(
            conn, EqualWeightSP500(), dates, PERMISSIVE, cost_config=ZERO_COSTS,
            starting_cash=10000.0, lookback_days=10,
        )
        run_id = save_run(
            conn, run_label=label, config={"seed": 42}, random_seed=42, start_date=dates[0], end_date=dates[-1],
            rebalance_frequency="monthly", universe_definition="SP500", data_quality_policy_name="PERMISSIVE",
            strategy_name="equal_weight_sp500", cost_config=ZERO_COSTS, execution_config={},
            report=report, coverage_report=coverage, created_at="2020-01-01T00:00:00Z",
        )
        return load_results(conn, run_id)

    results_a = _do_run("run_a")
    results_b = _do_run("run_b")

    # load_results' output is already just metric_name/value/period, so
    # equality here means the two runs produced byte-for-byte identical metrics.
    assert results_a == results_b
