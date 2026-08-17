"""
Leakage-control tests for src/ml/overnight_targets.py
(PHASE5_OVERNIGHT_GAP_SPECIFICATION.md section 0.3/12). Mirrors
tests/test_ml_features_leakage.py's fixture pattern exactly, so
overnight_targets.py runs against the real schema unmodified, and is
required to pass before the Tier 0 statistical test touches real data
(spec section 12, "Implementation -> Tier 0 test execution").
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sqlite3
import math
import random
import datetime
import pytest

from src.ml import overnight_targets as OT


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


def _seed_price(conn, security_id, date, open_, close, high=None, low=None, volume=1_000_000,
                 adj_type="total_return"):
    high = high if high is not None else max(open_, close)
    low = low if low is not None else min(open_, close)
    conn.execute(
        """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type, source_id, ingested_at)
           VALUES (?,?,?,?,?,?,?,?,1,'2020-01-01T00:00:00Z')""",
        (security_id, date, open_, high, low, close, volume, adj_type),
    )


def _business_dates(start_year, start_month, n):
    d = datetime.date(start_year, start_month, 1)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return out


# --- 1. Core decomposition identity: overnight + intraday == daily total return ---

def test_decomposition_identity_holds(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "AAA")
    dates = _business_dates(2020, 1, 10)
    prices = [(100.0, 101.0), (101.0, 100.5), (100.5, 102.0), (102.0, 101.5), (101.5, 103.0),
              (103.0, 102.5), (102.5, 104.0), (104.0, 103.5), (103.5, 105.0), (105.0, 104.5)]
    for d, (o, c) in zip(dates, prices):
        _seed_price(conn, 1, d, o, c)
    conn.commit()

    for i in range(1, len(dates)):
        decomp = OT.daily_decomposition(conn, 1, dates[i])
        assert decomp is not None
        assert decomp["decomposition_identity_ok"] is True


# --- 2. Core point-in-time invariant: a future price row must never change a past value ---

def test_future_price_never_changes_a_past_overnight_return(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "AAA")
    dates = _business_dates(2020, 1, 20)
    for i, d in enumerate(dates):
        _seed_price(conn, 1, d, 100.0 + i, 100.5 + i)
    conn.commit()

    as_of = dates[10]
    before = OT.overnight_return(conn, 1, as_of)
    before_decomp = OT.daily_decomposition(conn, 1, as_of)

    conn.execute("UPDATE prices SET open=999999.0, close=999999.0 WHERE security_id=1 AND date=?", (dates[-1],))
    conn.commit()

    after = OT.overnight_return(conn, 1, as_of)
    after_decomp = OT.daily_decomposition(conn, 1, as_of)
    assert before == after
    assert before_decomp == after_decomp


# --- 3. Never a "nearest available" fallback -- a missing exact-date price returns None ---

def test_missing_exact_date_open_returns_none_not_a_fallback(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "GAPPY")
    dates = _business_dates(2020, 1, 10)
    for i, d in enumerate(dates):
        if i == 5:
            continue  # deliberate gap: no row at all for dates[5]
        _seed_price(conn, 1, d, 100.0 + i, 100.5 + i)
    conn.commit()

    assert OT.overnight_return(conn, 1, dates[5]) is None
    assert OT.intraday_return(conn, 1, dates[5]) is None
    assert OT.daily_decomposition(conn, 1, dates[5]) is None


def test_missing_previous_close_resolves_to_market_wide_previous_session(conn):
    """A security's own data gap at t-1 must not be silently treated as
    'no previous session exists' if OTHER securities traded that day --
    previous_trading_session() resolves market-wide, exactly mirroring
    execution.next_market_session()'s reasoning for why a single
    security's gap shouldn't determine the whole panel's calendar."""
    _seed_source(conn)
    _seed_security(conn, 1, "A")
    _seed_security(conn, 2, "B")  # trades every day, including dates[4]
    dates = _business_dates(2020, 1, 10)
    for i, d in enumerate(dates):
        _seed_price(conn, 2, d, 200.0 + i, 200.5 + i)
        if i == 4:
            continue  # security 1 has a gap at dates[4]
        _seed_price(conn, 1, d, 100.0 + i, 100.5 + i)
    conn.commit()

    prev_session = OT.previous_trading_session(conn, dates[5])
    assert prev_session == dates[4]  # security 2 traded, so the session still "exists"

    # Security 1's own overnight_return at dates[5] must be None (it has
    # no close on dates[4] itself), not silently resolved against an
    # earlier session -- this is the "no nearest-available fallback" rule
    # applying at the previous-session boundary too.
    assert OT.overnight_return(conn, 1, dates[5]) is None


# --- 4. Reproducibility: identical inputs -> identical outputs ---

def test_overnight_targets_are_reproducible(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "REPRO")
    dates = _business_dates(2020, 1, 30)
    rng = random.Random(7)
    price = 100.0
    for d in dates:
        o = price
        price *= 1 + rng.uniform(-0.02, 0.02)
        c = price
        _seed_price(conn, 1, d, o, c)
    conn.commit()

    as_of = dates[15]
    first = OT.daily_decomposition(conn, 1, as_of)
    second = OT.daily_decomposition(conn, 1, as_of)
    assert first == second


# --- 5. Ex-date boundary (spec sections 2.4 / 4.3): raw open on the ex-date session ---
# vs. an adjusted prior close must not silently mix scales.

def test_ex_date_boundary_does_not_mix_scales_inconsistently(conn):
    """Simulates compute_total_return()'s own behaviour at a dividend
    ex-date boundary: prices strictly BEFORE ex_date get scaled by the
    dividend factor, ex_date itself and after do not (matching
    src/ingestion/adjustments.py's `date_ < ex_date` loop condition
    exactly). overnight_return() at the ex-date session divides the
    ex-date's own (unscaled) open by the prior session's (scaled) close --
    this test locks in that division of labour (the scaling is
    upstream's job, in compute_total_return(), never this module's) by
    checking the arithmetic is exactly what's stored, no implicit
    rescaling inside this module."""
    _seed_source(conn)
    _seed_security(conn, 1, "DIVCO")
    dates = _business_dates(2020, 1, 5)
    ex_date = dates[3]
    # Rows exactly as a real total_return series would look: dates before
    # ex_date carry a simulated 0.95 dividend back-adjustment factor;
    # ex_date itself and after are unscaled raw prints.
    raw = [(100.0, 100.5), (100.5, 101.0), (101.0, 99.5), (95.0, 96.0), (96.0, 97.0)]
    for i, d in enumerate(dates):
        o, c = raw[i]
        _seed_price(conn, 1, d, o, c)
    conn.commit()

    result = OT.overnight_return(conn, 1, ex_date)
    expected = math.log(95.0 / 99.5)
    assert result == pytest.approx(expected)


# --- 6. proxy_series_for_dates never re-derives eligibility, only uses what's passed in ---

def test_proxy_series_only_uses_the_provided_eligible_set(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "A")
    _seed_security(conn, 2, "B")
    _seed_security(conn, 3, "C")  # deliberately excluded from the eligible map
    dates = _business_dates(2020, 1, 5)
    for i, d in enumerate(dates):
        _seed_price(conn, 1, d, 100.0 + i, 101.0 + i)
        _seed_price(conn, 2, d, 200.0 + i, 199.0 + i)
        _seed_price(conn, 3, d, 50.0 + i, 999.0)  # extreme value, must never leak into the proxy
    conn.commit()

    eligible_by_date = {d: [1, 2] for d in dates[1:]}
    result = OT.proxy_series_for_dates(conn, eligible_by_date, dates[1:])
    for d in dates[1:]:
        assert result[d]["n_securities"] <= 2
        # A proxy dominated by security 3's extreme close would be far
        # outside a plausible daily log-return range; confirm it never
        # appears.
        if result[d]["overnight_proxy"] is not None:
            assert abs(result[d]["overnight_proxy"]) < 1.0


# --- 7. A date with no resolvable previous session (start of history) is handled, not crashed ---

def test_first_date_in_history_has_no_previous_session(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "A")
    dates = _business_dates(2020, 1, 5)
    for i, d in enumerate(dates):
        _seed_price(conn, 1, d, 100.0 + i, 100.5 + i)
    conn.commit()

    assert OT.previous_trading_session(conn, dates[0]) is None
    assert OT.overnight_return(conn, 1, dates[0]) is None
    assert OT.daily_decomposition(conn, 1, dates[0]) is None
    # intraday_return has no previous-session dependency, so it should
    # still resolve on the very first date.
    assert OT.intraday_return(conn, 1, dates[0]) is not None
