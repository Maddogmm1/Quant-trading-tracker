"""
Target/label tests for src/ml/targets.py: delisting/truncation handling
and the rule that only the universe passed in for time t is ever used
for labels at t, never a wider set discovered later.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sqlite3
import pytest

from src.ml import targets as T


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


def _seed_price(conn, security_id, date, close, adj_type="total_return"):
    conn.execute(
        """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type, source_id, ingested_at)
           VALUES (?,?,?,?,?,?,?,?,1,'2020-01-01T00:00:00Z')""",
        (security_id, date, close, close, close, close, 1000, adj_type),
    )


# --- 1. Known deterministic return, no truncation ---

def test_known_return_and_excess_over_benchmark(conn):
    _seed_source(conn)
    for sid, ticker in [(1, "A"), (2, "B")]:
        _seed_security(conn, sid, ticker)
    _seed_price(conn, 1, "2020-01-01", 100.0)
    _seed_price(conn, 1, "2020-04-01", 110.0)  # +10%
    _seed_price(conn, 2, "2020-01-01", 100.0)
    _seed_price(conn, 2, "2020-04-01", 90.0)   # -10%
    conn.commit()

    result = T.compute_labels_for_universe(conn, [1, 2], "2020-01-01", "2020-04-01")
    assert result["benchmark_return"] == pytest.approx(0.0)  # mean(+10%, -10%)
    assert result["per_security"][1]["r"] == pytest.approx(0.10)
    assert result["per_security"][1]["y"] == pytest.approx(0.10)  # excess over 0% benchmark
    assert result["per_security"][2]["r"] == pytest.approx(-0.10)
    assert result["per_security"][2]["z"] == 0
    assert result["per_security"][1]["z"] == 1
    assert result["excluded_no_price"] == []
    assert result["truncated_count"] == 0


# --- 2. Delisting: security has no price at target_date but has an earlier one -> truncated, not dropped ---

def test_security_delisted_before_target_date_is_truncated_not_dropped(conn):
    _seed_source(conn)
    for sid, ticker in [(1, "SURVIVOR"), (2, "DELISTED")]:
        _seed_security(conn, sid, ticker)
    _seed_price(conn, 1, "2020-01-01", 100.0)
    _seed_price(conn, 1, "2020-04-01", 105.0)
    _seed_price(conn, 2, "2020-01-01", 100.0)
    _seed_price(conn, 2, "2020-02-15", 40.0)   # collapses and stops trading well before target_date
    conn.commit()

    result = T.compute_labels_for_universe(conn, [1, 2], "2020-01-01", "2020-04-01")
    assert 2 in result["per_security"]  # not dropped
    assert result["per_security"][2]["r"] == pytest.approx(-0.60)  # frozen at last available price
    assert result["per_security"][2]["label_truncated"] is True
    assert result["per_security"][1]["label_truncated"] is False
    assert result["truncated_count"] == 1
    assert result["excluded_no_price"] == []
    # The benchmark return must include the delisted security's real
    # (bad) outcome. Dropping it here would be exactly the kind of
    # one-label-at-a-time survivorship bias this function needs to avoid.
    assert result["benchmark_return"] == pytest.approx((0.05 + (-0.60)) / 2)


# --- 3. No resolvable price at all -> excluded, never a fabricated 0 ---

def test_security_with_no_price_at_or_before_target_is_excluded_not_fabricated(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "GHOST")
    # No price rows at all for security 1.
    _seed_security(conn, 2, "NORMAL")
    _seed_price(conn, 2, "2020-01-01", 100.0)
    _seed_price(conn, 2, "2020-04-01", 100.0)
    conn.commit()

    result = T.compute_labels_for_universe(conn, [1, 2], "2020-01-01", "2020-04-01")
    assert 1 not in result["per_security"]
    assert 1 in result["excluded_no_price"]
    assert 2 in result["per_security"]


# --- 4. Universe-membership leakage: only the passed-in list at t is ever used, never a later universe ---

def test_only_the_passed_in_universe_is_used_never_a_wider_set(conn):
    _seed_source(conn)
    for sid, ticker in [(1, "A"), (2, "B"), (3, "LATE_ARRIVAL")]:
        _seed_security(conn, sid, ticker)
    _seed_price(conn, 1, "2020-01-01", 100.0)
    _seed_price(conn, 1, "2020-04-01", 110.0)
    _seed_price(conn, 2, "2020-01-01", 100.0)
    _seed_price(conn, 2, "2020-04-01", 100.0)
    # security 3 only has data from 2020-04-01 onward, so it would be
    # eligible at the target date but wasn't eligible at t and must
    # never be used.
    _seed_price(conn, 3, "2020-04-01", 500.0)
    conn.commit()

    result = T.compute_labels_for_universe(conn, [1, 2], "2020-01-01", "2020-04-01")
    assert set(result["per_security"].keys()) == {1, 2}
    assert 3 not in result["per_security"]
    assert 3 not in result["excluded_no_price"]  # never considered, not just excluded after the fact


# --- 5b. A non-trading target_date (weekend/holiday, exactly what
# next_rebalance_dates() produces for calendar month-starts) must not be
# flagged as a truncated/delisting label just because no security has an
# exact price row on that exact date. Only a genuinely large gap (the
# security actually stopped trading well before target_date) should be
# flagged; this mirrors the anomaly threshold the execution-timing
# diagnostic elsewhere in the codebase uses to distinguish normal
# calendar noise from a real problem.

def test_weekend_target_date_is_not_flagged_as_truncated(conn):
    _seed_source(conn)
    for sid, ticker in [(1, "A"), (2, "B")]:
        _seed_security(conn, sid, ticker)
    # 2020-02-01 is a Saturday, so no security trades that day, but both
    # have a normal price 1-2 days later, same as any ordinary month-start.
    _seed_price(conn, 1, "2020-01-01", 100.0)
    _seed_price(conn, 1, "2020-02-03", 105.0)  # the following Monday
    _seed_price(conn, 2, "2020-01-01", 100.0)
    _seed_price(conn, 2, "2020-02-03", 95.0)
    conn.commit()

    result = T.compute_labels_for_universe(conn, [1, 2], "2020-01-01", "2020-02-01")
    assert result["per_security"][1]["label_truncated"] is False
    assert result["per_security"][2]["label_truncated"] is False
    assert result["truncated_count"] == 0
    assert result["per_security"][1]["r"] == pytest.approx(0.05)


def test_genuine_multi_month_gap_is_still_flagged_as_truncated(conn):
    _seed_source(conn)
    _seed_security(conn, 1, "DELISTED")
    _seed_price(conn, 1, "2020-01-01", 100.0)
    _seed_price(conn, 1, "2020-01-15", 40.0)  # collapses and stops trading, months before target_date
    conn.commit()

    result = T.compute_labels_for_universe(conn, [1], "2020-01-01", "2020-06-01")
    assert result["per_security"][1]["label_truncated"] is True
    assert result["truncated_count"] == 1


# --- 5. Rank is consistent with excess return ordering ---

def test_rank_matches_excess_return_ordering(conn):
    _seed_source(conn)
    for i, sid in enumerate([1, 2, 3, 4]):
        _seed_security(conn, sid, f"S{sid}")
        _seed_price(conn, sid, "2020-01-01", 100.0)
    _seed_price(conn, 1, "2020-04-01", 90.0)   # worst
    _seed_price(conn, 2, "2020-04-01", 100.0)
    _seed_price(conn, 3, "2020-04-01", 110.0)
    _seed_price(conn, 4, "2020-04-01", 120.0)  # best
    conn.commit()

    result = T.compute_labels_for_universe(conn, [1, 2, 3, 4], "2020-01-01", "2020-04-01")
    ranks = {sid: result["per_security"][sid]["rank"] for sid in [1, 2, 3, 4]}
    assert ranks[1] < ranks[2] < ranks[3] < ranks[4]
