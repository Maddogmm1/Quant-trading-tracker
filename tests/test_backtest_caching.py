"""
Regression tests for the universe_cache optimization in engine.run_backtest.

build_eligible_universe() is a pure function of its arguments, so it's safe
to memoize across run_backtest() calls that share those arguments (e.g. the
4 benchmarks plus N random seeds at one data_quality_policy). These tests
prove cached and uncached runs produce byte-identical results.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sqlite3
import datetime
import pytest

from src.backtest.engine import run_backtest, _universe_cache_key
from src.backtest.execution import next_rebalance_dates
from src.backtest.benchmarks import TopNMomentum, RandomSelection

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
    for sec_id, ticker in [(1, "AAA"), (2, "BBB"), (3, "CCC")]:
        c.execute(
            """INSERT INTO securities (security_id, primary_ticker, name, asset_type, active_flag,
               identifier_quality, created_at, updated_at)
               VALUES (?,?,?,'STOCK',1,'resolved','2020-01-01T00:00:00Z','2020-01-01T00:00:00Z')""",
            (sec_id, ticker, ticker),
        )
        c.execute(
            """INSERT INTO index_membership (security_id, raw_ticker, index_name, effective_date, removal_date,
               source_id, confidence, ingested_at) VALUES (?,?,?,?,NULL,1,'verified','2020-01-01T00:00:00Z')""",
            (sec_id, ticker, "SP500", "2019-01-01"),
        )
        d = datetime.date(2019, 1, 1)
        end = datetime.date(2020, 6, 1)
        price = 90.0 + sec_id * 7  # distinct, deterministic price paths per security
        while d <= end:
            if d.weekday() < 5:
                price *= 1.0003
                c.execute(
                    """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
                       source_id, ingested_at) VALUES (?,?,?,?,?,?,1000,'total_return',1,'2020-01-01T00:00:00Z')""",
                    (sec_id, d.isoformat(), price, price, price, price),
                )
            d += datetime.timedelta(days=1)
    c.commit()
    yield c
    c.close()


def test_cached_and_uncached_runs_produce_identical_results(conn):
    dates = next_rebalance_dates("2019-06-01", "2020-05-01", "monthly")

    report_uncached, coverage_uncached, _ = run_backtest(
        conn, TopNMomentum(n=2, lookback_days=10), dates, PERMISSIVE, cost_config=ZERO_COSTS,
        starting_cash=10000.0, lookback_days=10, universe_cache=None,
    )
    cache = {}
    report_cached, coverage_cached, _ = run_backtest(
        conn, TopNMomentum(n=2, lookback_days=10), dates, PERMISSIVE, cost_config=ZERO_COSTS,
        starting_cash=10000.0, lookback_days=10, universe_cache=cache,
    )

    assert report_uncached == report_cached
    assert coverage_uncached == coverage_cached
    assert len(cache) > 0  # confirms the cache was actually populated/used


def test_cache_is_shared_correctly_across_different_strategies_same_policy(conn):
    """Multiple different strategies sharing one cache dict at the same
    policy should each get results identical to running uncached."""
    dates = next_rebalance_dates("2019-06-01", "2020-05-01", "monthly")
    shared_cache = {}

    results_cached = {}
    results_uncached = {}
    for label, strat_factory in [
        ("top_n", lambda: TopNMomentum(n=2, lookback_days=10)),
        ("random_seed_1", lambda: RandomSelection(portfolio_size=2, seed=1)),
        ("random_seed_2", lambda: RandomSelection(portfolio_size=2, seed=2)),
    ]:
        report, coverage, _ = run_backtest(
            conn, strat_factory(), dates, PERMISSIVE, cost_config=ZERO_COSTS,
            starting_cash=10000.0, lookback_days=10, universe_cache=shared_cache,
        )
        results_cached[label] = (report, coverage)

        report_u, coverage_u, _ = run_backtest(
            conn, strat_factory(), dates, PERMISSIVE, cost_config=ZERO_COSTS,
            starting_cash=10000.0, lookback_days=10, universe_cache=None,
        )
        results_uncached[label] = (report_u, coverage_u)

    for label in results_cached:
        assert results_cached[label] == results_uncached[label], f"mismatch for {label}"

    # Coverage is a pure function of date+policy, so every strategy's series
    # should match the others -- confirms the shared cache isn't leaking
    # strategy-specific state into what should be a universe-only cache.
    coverages = [results_cached[label][1] for label in results_cached]
    assert all(c == coverages[0] for c in coverages)


def test_universe_cache_key_distinguishes_policies_and_dates(conn):
    key_a = _universe_cache_key("2020-01-01", PERMISSIVE, {}, "SP500", 10, "total_return")
    key_b = _universe_cache_key("2020-02-01", PERMISSIVE, {}, "SP500", 10, "total_return")
    strict = {**PERMISSIVE, "min_completeness_pct": 0.95}
    key_c = _universe_cache_key("2020-01-01", strict, {}, "SP500", 10, "total_return")
    assert key_a != key_b  # different dates must not collide
    assert key_a != key_c  # different policies must not collide
