"""
Feature-matrix assembly tests for src/ml/feature_matrix.py. Two
concerns: (1) FEATURE_FAMILIES in code stays byte-for-byte in sync with
config.yaml's phase4.v1_features, and (2) build_panel() never produces a
row using a date outside what it was given, and never fabricates a value
for a missing feature or label.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sqlite3
import yaml
import pytest

from src.ml.feature_matrix import FEATURE_FAMILIES, build_panel, all_v1_feature_names
from src.backtest.execution import next_rebalance_dates
from src.ml.walk_forward import build_primary_split


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")


def test_feature_families_match_config_yaml_exactly():
    cfg = yaml.safe_load(open(CONFIG_PATH))
    configured = cfg["phase4"]["v1_features"]
    for family, names in FEATURE_FAMILIES.items():
        assert family in configured, f"family {family!r} in code but not in config.yaml"
        assert configured[family] == names, (
            f"family {family!r} drifted: config.yaml has {configured[family]}, code has {names}"
        )
    for family in configured:
        if family == "explicitly_excluded":
            continue
        assert family in FEATURE_FAMILIES, f"family {family!r} in config.yaml but not in code"


def test_explicitly_excluded_features_are_never_in_the_v1_feature_list():
    cfg = yaml.safe_load(open(CONFIG_PATH))
    excluded = set(cfg["phase4"]["v1_features"]["explicitly_excluded"])
    assert not (excluded & set(all_v1_feature_names()))


# --- panel construction against a synthetic in-memory database ---

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


def _seed_price_series(conn, security_id, dates, start_price=100.0, drift=0.0002, vol=0.01, seed=1):
    import random
    rng = random.Random(seed + security_id)
    price = start_price
    for d in dates:
        price *= 1 + rng.gauss(drift, vol)
        price = max(price, 1.0)
        conn.execute(
            """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type, source_id, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,1,'2020-01-01T00:00:00Z')""",
            (security_id, d, price, price, price, price, 5_000_000, "total_return"),
        )


def _seed_membership(conn, security_id, ticker, effective_date):
    conn.execute("INSERT OR IGNORE INTO data_sources (source_id, source_name, tier) VALUES (2,'membership_source','A')")
    conn.execute(
        """INSERT INTO index_membership (security_id, raw_ticker, index_name, effective_date, source_id, confidence, membership_quality, ingested_at)
           VALUES (?,?,?,?,2,'verified','complete','2020-01-01T00:00:00Z')""",
        (security_id, ticker, "SP500", effective_date),
    )


def _daily_dates(n, start="2015-01-01"):
    import datetime
    d = datetime.date.fromisoformat(start)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return out


def _minimal_config():
    return {
        "backtest": {
            "data_quality_policies": {
                "PERMISSIVE": {
                    "min_completeness_pct": 0.80, "require_full_history": False,
                    "exclude_unresolved_identity": False, "exclude_identity_review_flagged": False,
                    "exclude_severe_ohlc_flagged": False, "severe_ohlc_bad_row_pct_threshold": 0.10,
                },
            },
            "execution": {"lookback_days_required": 252},
        },
    }


def test_build_panel_never_uses_a_date_outside_what_it_was_given(conn):
    _seed_source(conn)
    dates = _daily_dates(800)  # comfortably more than the 252-trading-day lookback several features need
    for sid, ticker in [(1, "A"), (2, "B"), (3, "C")]:
        _seed_security(conn, sid, ticker)
        _seed_price_series(conn, sid, dates, seed=sid)
        _seed_membership(conn, sid, ticker, dates[0])
    conn.commit()

    rebalance_dates = next_rebalance_dates(dates[0], dates[-1], "monthly")
    assert len(rebalance_dates) > 20

    cfg = _minimal_config()
    # A slice starting well past month 12 (so the 252-trading-day lookback
    # features are satisfiable), deliberately excluding the last few
    # rebalance dates so we can assert the panel never reaches past them.
    as_of_dates = rebalance_dates[15:20]

    rows, stats = build_panel(conn, cfg, "PERMISSIVE", as_of_dates, rebalance_dates, horizon_months=1,
                               predeclared_filters={}, log=None)

    assert stats["rows_built"] > 0
    used_dates = {r["as_of_date"] for r in rows}
    assert used_dates.issubset(set(as_of_dates))
    # every row's target_date must be a real, later rebalance date. If it's
    # beyond as_of_dates that's fine (labels legitimately look forward),
    # but the as_of_date itself must never exceed what was passed in.
    for r in rows:
        assert r["as_of_date"] in as_of_dates
        assert rebalance_dates.index(r["target_date"]) == rebalance_dates.index(r["as_of_date"]) + 1


def test_build_panel_drops_rows_with_missing_features_rather_than_fabricating(conn):
    _seed_source(conn)
    dates = _daily_dates(500)
    _seed_security(conn, 1, "SHORT_HISTORY")
    # Only seed a short tail of price history: not enough for a 252d lookback
    _seed_price_series(conn, 1, dates[-30:], seed=1)
    _seed_membership(conn, 1, "SHORT_HISTORY", dates[-30])
    conn.commit()

    rebalance_dates = next_rebalance_dates(dates[-30], dates[-1], "monthly")
    cfg = _minimal_config()
    if len(rebalance_dates) < 2:
        pytest.skip("not enough synthetic rebalance dates for this check")

    rows, stats = build_panel(conn, cfg, "PERMISSIVE", rebalance_dates[:-1], rebalance_dates, horizon_months=1,
                               predeclared_filters={}, log=None)
    # every row for this thin-history security should be dropped for missing features
    assert rows == []
    assert stats["rows_dropped_missing_feature"] > 0 or stats["rows_built"] == 0
