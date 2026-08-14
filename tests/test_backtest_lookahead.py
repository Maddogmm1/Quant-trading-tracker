"""
Look-ahead-bias tests. Each test mutates data dated after a decision date
and asserts the decision is unchanged, since the only real way to catch
future leakage is to prove changing the future doesn't change the past.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sqlite3
import copy
import pytest

from src.backtest.universe import build_eligible_universe
from src.backtest.execution import PointInTimeDataAccess, FutureDataAccessError

PERMISSIVE = {
    "min_completeness_pct": 0.80, "require_full_history": False,
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


def _sec(conn, sec_id, ticker, identifier_quality="resolved"):
    conn.execute(
        """INSERT INTO securities (security_id, primary_ticker, name, asset_type, active_flag,
           identifier_quality, created_at, updated_at)
           VALUES (?,?,?,'STOCK',1,?,'2020-01-01T00:00:00Z','2020-01-01T00:00:00Z')""",
        (sec_id, ticker, ticker, identifier_quality),
    )


def _price(conn, sec_id, date, close, adj_type="total_return"):
    conn.execute(
        """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type, source_id, ingested_at)
           VALUES (?,?,?,?,?,?,1000,?,1,'2020-01-01T00:00:00Z')""",
        (sec_id, date, close, close, close, close, adj_type),
    )


def _membership(conn, sec_id, ticker, eff, removal=None, index_name="SP500"):
    conn.execute(
        """INSERT INTO index_membership (security_id, raw_ticker, index_name, effective_date, removal_date,
           source_id, confidence, ingested_at) VALUES (?,?,?,?,?,1,'verified','2020-01-01T00:00:00Z')""",
        (sec_id, ticker, index_name, eff, removal),
    )


def _seed_liquid_history(conn, sec_id, ticker, start="2019-01-01", end="2020-06-01", price=100.0):
    """A long, complete daily price series so the lookback/completeness/
    liquidity checks all pass -- keeps these tests exercising real
    eligibility logic rather than bailing out early on thin history."""
    import datetime
    d = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    while d <= e:
        if d.weekday() < 5:
            _price(conn, sec_id, d.isoformat(), price)
        d += datetime.timedelta(days=1)


# --- 1. Future-price mutation: signals through T must not change ---
def test_future_price_mutation_does_not_change_past_signal(conn):
    _sec(conn, 1, "AAA")
    _seed_liquid_history(conn, 1, "AAA", start="2019-01-01", end="2020-01-15")
    conn.commit()

    da = PointInTimeDataAccess(conn, "2020-01-10")
    before = da.trailing_return(1, 100)

    # mutate a price dated AFTER the decision date
    conn.execute("UPDATE prices SET close=999999 WHERE security_id=1 AND date>?", ("2020-01-10",))
    conn.commit()

    da2 = PointInTimeDataAccess(conn, "2020-01-10")
    after = da2.trailing_return(1, 100)
    assert before == after


# --- 2. Future-membership mutation: earlier universes must not change ---
def test_future_membership_mutation_does_not_change_earlier_universe(conn):
    _sec(conn, 1, "AAA")
    _seed_liquid_history(conn, 1, "AAA")
    _membership(conn, 1, "AAA", "2019-01-01", None)
    conn.commit()

    eligible_before, _ = build_eligible_universe(conn, "2019-06-01", PERMISSIVE, lookback_days=50)

    # add a NEW security whose membership only starts in the future relative to the decision date
    _sec(conn, 2, "BBB")
    _seed_liquid_history(conn, 2, "BBB")
    _membership(conn, 2, "BBB", "2020-03-01", None)  # starts AFTER our 2019-06-01 decision date
    conn.commit()

    eligible_after, _ = build_eligible_universe(conn, "2019-06-01", PERMISSIVE, lookback_days=50)
    assert eligible_before == eligible_after
    assert 2 not in eligible_after  # BBB correctly absent from the earlier universe


# --- 3. Future-dividend mutation: earlier portfolio decisions unchanged ---
def test_future_dividend_mutation_does_not_change_earlier_decision(conn):
    _sec(conn, 1, "AAA")
    _seed_liquid_history(conn, 1, "AAA")
    conn.commit()

    from src.backtest.accounting import Portfolio
    zero_costs = {"commission_pct": 0, "fx_cost_pct": 0, "stamp_duty_sdrt_pct": 0, "ptm_levy_gbp": 0,
                  "sec_finra_fee_pct": 0, "bid_ask_spread_bps": 0, "slippage_bps": 0}
    p = Portfolio(10000.0, zero_costs)
    pv_before = p.rebalance_to(conn, "2019-06-03", {1: 1.0})  # a real trading date in the seeded series

    # add a dividend dated in the future relative to this decision
    conn.execute(
        "INSERT INTO corporate_actions (security_id, action_type, ex_date, ratio_or_value, source_id, ingested_at) "
        "VALUES (1,'dividend','2020-05-01',50.0,1,'2020-01-01T00:00:00Z')"
    )
    conn.commit()

    p2 = Portfolio(10000.0, zero_costs)
    pv_after = p2.rebalance_to(conn, "2019-06-03", {1: 1.0})
    assert pv_before == pytest.approx(pv_after)


# --- 4. General parametrized future-data-cannot-alter-past test ---
@pytest.mark.parametrize("table,mutation_sql", [
    ("prices", "UPDATE prices SET close=888888 WHERE security_id=1 AND date>'2020-01-10'"),
    ("corporate_actions", "INSERT INTO corporate_actions (security_id, action_type, ex_date, ratio_or_value, "
                           "source_id, ingested_at) VALUES (1,'split','2020-06-01',2.0,1,'2020-01-01T00:00:00Z')"),
])
def test_general_future_mutation_does_not_alter_past_result(conn, table, mutation_sql):
    _sec(conn, 1, "AAA")
    _seed_liquid_history(conn, 1, "AAA", end="2020-01-15")
    conn.commit()

    da = PointInTimeDataAccess(conn, "2020-01-10")
    before = da.price_history(1, lookback_days=20)

    conn.execute(mutation_sql)
    conn.commit()

    da2 = PointInTimeDataAccess(conn, "2020-01-10")
    after = da2.price_history(1, lookback_days=20)
    assert before == after


# --- 5. Structural guard: asking for future data raises rather than just happening to return correctly ---
def test_data_access_raises_on_explicit_future_request(conn):
    _sec(conn, 1, "AAA")
    _seed_liquid_history(conn, 1, "AAA")
    conn.commit()
    da = PointInTimeDataAccess(conn, "2019-06-01")
    with pytest.raises(FutureDataAccessError):
        da.price_history(1, lookback_days=10, end_date="2019-12-01")


# --- 6. AMR -> AAMRQ-style identity substitution (see BACKLOG.md item 2) ---
def test_identity_substitution_adversarial_case_amr_aamrq_pattern(conn):
    """
    Mirrors a real finding: AMR Corp traded as 'AMR' until its 2012
    delisting, and the ticker 'AAMRQ' only existed afterward, during
    bankruptcy proceedings. A source file that labels AMR's whole
    historical membership window with the later ticker 'AAMRQ' would,
    if naively resolved, make it look like 'AAMRQ' had tradeable data
    back to 1996. Membership must resolve via security_id, not raw
    ticker string, so the later ticker's own (much later) price history
    can never be used to satisfy an earlier availability check.
    """
    # The real early-period entity traded as "OLDTICK" (stand-in for AMR)
    # with genuine daily data through its delisting date.
    _sec(conn, 1, "OLDTICK")
    _seed_liquid_history(conn, 1, "OLDTICK", start="1996-01-02", end="2003-01-01")
    _membership(conn, 1, "OLDTICK", "1996-01-02", "2003-01-01")

    # A different, later-created security record for "NEWTICK" (stand-in
    # for AAMRQ), whose own real data only starts in 2012.
    _sec(conn, 2, "NEWTICK")
    _seed_liquid_history(conn, 2, "NEWTICK", start="2012-01-30", end="2013-01-01")
    conn.commit()

    # The bug pattern: a membership source file mislabels the 1996-2003
    # window with the later ticker "NEWTICK" while still pointing at the
    # correct underlying security_id=1.
    _membership(conn, 1, "NEWTICK", "1996-01-02", "2003-01-01")
    conn.commit()

    # A decision date inside the real 1996-2003 window.
    decision_date = "1998-06-01"

    # Security 2 ("NEWTICK"'s own identity) has no data anywhere near
    # 1998, so if membership were incorrectly resolved through security
    # 2's own price history, this would silently look like a clean
    # "no data" case rather than exposing the bug.
    wrong_entity_check = build_eligible_universe(
        conn, decision_date, PERMISSIVE, lookback_days=50, universe_definition="SP500",
    )
    eligible, exclusion_report = wrong_entity_check

    assert 1 in eligible, (
        "Membership must resolve against the correct underlying security_id, "
        "not get silently lost because the raw_ticker string in the source "
        "file doesn't match that security's current primary_ticker."
    )

    # Security 2's real, independent 2012-2013 data must never be used to
    # satisfy this 1998 decision, even though it's named "NEWTICK" today.
    da = PointInTimeDataAccess(conn, decision_date)
    with pytest.raises(FutureDataAccessError):
        da.price_history(2, lookback_days=10, end_date="2012-01-30")

    hist_sec2_asof_1998 = PointInTimeDataAccess(conn, decision_date).price_history(2, lookback_days=10)
    assert hist_sec2_asof_1998 == [], (
        "Security 2's own 2012-era data must not appear as available history "
        "for a 1998 decision date -- there's no artificial bridging of the "
        "real gap between the two entities' trading windows."
    )
