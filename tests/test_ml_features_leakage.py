"""
Leakage-control tests for src/ml/features.py: model training only begins
once these pass, since a feature that peeks at future data silently
poisons every downstream backtest result.

Uses the same in-memory-schema fixture pattern as
tests/test_backtest_synthetic.py, so features.py runs against the real
schema unmodified.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sqlite3
import pytest

from src.ml import features as F


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(open(os.path.join(os.path.dirname(__file__), "..", "src", "database", "schema.sql")).read())
    yield c
    c.close()


def _seed_source(conn):
    conn.execute("INSERT OR IGNORE INTO data_sources (source_id, source_name, tier) VALUES (1,'test_source','C')")


def _seed_security(conn, security_id, ticker):
    conn.execute(
        """INSERT INTO securities (security_id, primary_ticker, name, asset_type, active_flag,
           identifier_quality, created_at, updated_at)
           VALUES (?,?,?,'STOCK',1,'resolved','2020-01-01T00:00:00Z','2020-01-01T00:00:00Z')""",
        (security_id, ticker, ticker),
    )


def _seed_price(conn, security_id, date, close, volume=1_000_000, adj_type="total_return"):
    conn.execute(
        """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type, source_id, ingested_at)
           VALUES (?,?,?,?,?,?,?,?,1,'2020-01-01T00:00:00Z')""",
        (security_id, date, close, close, close, close, volume, adj_type),
    )


def _business_dates(start_year, start_month, n):
    """n consecutive weekday dates as ISO strings, starting from the 1st
    of start_year/start_month. Deterministic and doesn't need a real
    calendar library for a synthetic fixture."""
    import datetime
    d = datetime.date(start_year, start_month, 1)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return out


# --- 1. Core point-in-time invariant: a future price must never change a past feature ---

def test_future_price_row_never_changes_a_past_feature(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "AAA")
    dates = _business_dates(2020, 1, 300)
    for i, d in enumerate(dates):
        _seed_price(conn, 1, d, 100.0 + i * 0.1)  # smooth upward drift
    conn.commit()

    as_of = dates[250]
    before = {
        "return_1m": F.return_1m(conn, 1, as_of),
        "return_3m": F.return_3m(conn, 1, as_of),
        "return_12m": F.return_12m(conn, 1, as_of),
        "vol_63d": F.realised_volatility_63d(conn, 1, as_of),
        "mdd_252d": F.max_drawdown_252d(conn, 1, as_of),
        "dist_200dma": F.distance_from_200d_ma(conn, 1, as_of),
    }

    # Overwrite the future-most price with a wildly different value. If it
    # leaked backward into any of the above, every one of them would change.
    conn.execute("UPDATE prices SET close=999999.0 WHERE security_id=1 AND date=?", (dates[299],))
    conn.commit()

    after = {
        "return_1m": F.return_1m(conn, 1, as_of),
        "return_3m": F.return_3m(conn, 1, as_of),
        "return_12m": F.return_12m(conn, 1, as_of),
        "vol_63d": F.realised_volatility_63d(conn, 1, as_of),
        "mdd_252d": F.max_drawdown_252d(conn, 1, as_of),
        "dist_200dma": F.distance_from_200d_ma(conn, 1, as_of),
    }
    assert before == after


# --- 2. The total-return backward-restatement rule must not leak via a ratio feature ---

def test_dividend_added_after_the_fact_does_not_change_a_ratio_feature(conn):
    """compute_total_return() backward-restates every historical
    total_return row using every known dividend, even one dated after
    as_of_date, by design. This proves the math holds up: a ratio feature
    computed over a bounded window is invariant to a dividend added later
    with an ex_date after the window, because the future-dividend
    multiplicative constant is shared by every price in the window and
    cancels out."""
    _seed_source(conn)
    _seed_security(conn, 1, "DIVCO")
    dates = _business_dates(2020, 1, 300)
    for i, d in enumerate(dates):
        _seed_price(conn, 1, d, 100.0 + i * 0.1)
    conn.commit()

    as_of = dates[250]
    before = F.return_3m(conn, 1, as_of)

    # Simulate what compute_total_return() would do: a dividend with an
    # ex_date after as_of retroactively scales every price strictly
    # before that ex_date, uniformly, by the same factor.
    ex_date = dates[299]
    factor = 0.95
    for i, d in enumerate(dates):
        if d < ex_date:
            conn.execute(
                "UPDATE prices SET close=? WHERE security_id=1 AND date=? AND adj_type='total_return'",
                (round((100.0 + i * 0.1) * factor, 10), d),
            )
    conn.commit()

    after = F.return_3m(conn, 1, as_of)
    assert before == pytest.approx(after, rel=1e-9)


# --- 3. Insufficient history returns None, never a fabricated value ---

def test_insufficient_history_returns_none_not_a_fabricated_value(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "SHORT")
    dates = _business_dates(2020, 1, 10)  # far fewer than any lookback requires
    for i, d in enumerate(dates):
        _seed_price(conn, 1, d, 100.0 + i)
    conn.commit()

    as_of = dates[-1]
    assert F.return_1m(conn, 1, as_of) is None
    assert F.return_12m(conn, 1, as_of) is None
    assert F.realised_volatility_63d(conn, 1, as_of) is None
    assert F.max_drawdown_252d(conn, 1, as_of) is None
    assert F.distance_from_200d_ma(conn, 1, as_of) is None
    assert F.rolling_avg_dollar_volume_20d(conn, 1, as_of) is None


# --- 4. Reproducibility: identical inputs -> identical outputs ---

def test_features_are_reproducible(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "REPRO")
    dates = _business_dates(2020, 1, 300)
    import random
    rng = random.Random(7)
    price = 100.0
    for d in dates:
        price *= 1 + rng.uniform(-0.02, 0.02)
        _seed_price(conn, 1, d, price)
    conn.commit()

    as_of = dates[250]
    first = {name: getattr(F, name)(conn, 1, as_of) for name in
             ["return_1m", "return_3m", "return_6m", "return_12m", "momentum_acceleration",
              "distance_from_200d_ma", "realised_volatility_63d", "downside_volatility_63d",
              "max_drawdown_252d", "dollar_volume", "rolling_avg_dollar_volume_20d",
              "volume_trend_20d_100d"]}
    second = {name: getattr(F, name)(conn, 1, as_of) for name in first}
    assert first == second


# --- 5. Rolling dollar-volume window is a genuine rolling window, not a whole-history average ---

def test_rolling_dollar_volume_ignores_data_outside_its_window(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "VOL")
    dates = _business_dates(2020, 1, 120)
    for i, d in enumerate(dates):
        # Huge volume far in the past, tiny volume recently. A
        # whole-history average would be dominated by the huge early
        # volume; a true 20-day rolling window should not be.
        vol = 100_000_000 if i < 50 else 1_000
        _seed_price(conn, 1, d, 100.0, volume=vol)
    conn.commit()

    as_of = dates[-1]
    rolling20 = F.rolling_avg_dollar_volume_20d(conn, 1, as_of)
    assert rolling20 == pytest.approx(100.0 * 1_000, rel=1e-6)  # recent tiny volume only


# --- 6. Cross-sectional rank/mean never uses a security outside the map passed in ---

def test_cross_sectional_rank_only_uses_the_provided_eligible_map(conn):
    values = {1: 0.10, 2: 0.20, 3: 0.30}  # simulates ELIG(t) only
    rank_of_2 = F.cross_sectional_percentile_rank(2, values)
    # security 2 is the median of exactly these 3 values -> rank 0.5
    assert rank_of_2 == pytest.approx(0.5)

    # A security not in the map (e.g. only eligible at t+h, never at t)
    # must not silently participate: own value None means rank None.
    assert F.cross_sectional_percentile_rank(999, values) is None


def test_proxy_index_daily_returns_never_reads_past_as_of_date(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "A")
    _seed_security(conn, 2, "B")
    dates = _business_dates(2020, 1, 40)
    for i, d in enumerate(dates):
        _seed_price(conn, 1, d, 100.0 + i)
        _seed_price(conn, 2, d, 200.0 + i * 2)
    conn.commit()

    as_of = dates[20]
    before = F.proxy_index_volatility(conn, [1, 2], as_of, n_sessions=15)

    # Extreme future prices for both members, after as_of.
    for d in dates[21:]:
        conn.execute("UPDATE prices SET close=close*100 WHERE date=?", (d,))
    conn.commit()

    after = F.proxy_index_volatility(conn, [1, 2], as_of, n_sessions=15)
    assert before == after


def test_price_history_length_asof_is_point_in_time_not_all_time(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "A")
    dates = _business_dates(2020, 1, 40)
    for i, d in enumerate(dates):
        _seed_price(conn, 1, d, 100.0 + i)
    conn.commit()

    as_of = dates[19]  # the 20th trading session
    length_before = F.price_history_length_asof(conn, 1, as_of)
    assert length_before == 20  # dates[0..19] inclusive

    # Mutating rows already dated after as_of must not change the
    # point-in-time count. This is a diagnostic function, so if it
    # silently counted future rows it would itself be a leakage bug in a
    # leakage-detection tool.
    for d in dates[20:]:
        conn.execute("UPDATE prices SET close=close*100 WHERE date=?", (d,))
    conn.commit()
    length_after = F.price_history_length_asof(conn, 1, as_of)
    assert length_after == length_before == 20


def test_price_history_length_asof_is_zero_for_a_security_with_no_history_yet(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "GHOST")
    conn.commit()
    assert F.price_history_length_asof(conn, 1, "2020-01-01") == 0
