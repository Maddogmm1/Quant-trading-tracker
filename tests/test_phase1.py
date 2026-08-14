"""
Phase 1 acceptance tests. Run against a freshly-built demo database
(run_phase1_demo.run() must have been executed at least once, twice for
the idempotency assertions to be meaningful).

Run with: pytest tests/test_phase1.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
from src.database.db import get_connection
from src.validation import checks

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "database", "quant_trader.db")


@pytest.fixture(scope="module")
def conn():
    c = get_connection(DB_PATH)
    yield c
    c.close()


# ---------------------------------------------------------------
# 1. Duplicate prevention / idempotency
# ---------------------------------------------------------------
def test_no_duplicate_securities(conn):
    rows = conn.execute("SELECT primary_ticker, COUNT(*) c FROM securities GROUP BY primary_ticker HAVING c > 1").fetchall()
    assert rows == [], f"Duplicate securities found: {rows}"


def test_no_duplicate_price_rows(conn):
    rows = conn.execute("""
        SELECT security_id, date, adj_type, source_id, COUNT(*) c
        FROM prices GROUP BY security_id, date, adj_type, source_id HAVING c > 1
    """).fetchall()
    assert rows == [], f"Duplicate price rows found: {rows}"


def test_no_duplicate_membership_claims_from_same_source(conn):
    rows = conn.execute("""
        SELECT raw_ticker, index_name, effective_date, source_id, COUNT(*) c
        FROM index_membership GROUP BY raw_ticker, index_name, effective_date, source_id HAVING c > 1
    """).fetchall()
    assert rows == [], f"Duplicate membership claims from the same source: {rows}"


def test_ingestion_log_recorded_multiple_runs(conn):
    n_runs = conn.execute("SELECT COUNT(*) c FROM ingestion_log").fetchone()["c"]
    assert n_runs >= 2, "Expected at least 2 logged runs (proves the pipeline was actually re-run, not just idempotent by omission)"


# ---------------------------------------------------------------
# 2. Ticker change handling
# ---------------------------------------------------------------
def test_fb_meta_resolve_to_same_security(conn):
    fb = conn.execute("SELECT security_id FROM ticker_history WHERE ticker='FB'").fetchone()
    meta = conn.execute("SELECT security_id FROM ticker_history WHERE ticker='META'").fetchone()
    assert fb is not None and meta is not None
    assert fb["security_id"] == meta["security_id"], "FB and META must resolve to the same security_id (ticker change, not a new company)"


def test_ticker_history_windows_are_contiguous_for_fb_meta(conn):
    rows = conn.execute("""
        SELECT ticker, valid_from, valid_to FROM ticker_history
        WHERE ticker IN ('FB','META') ORDER BY valid_from
    """).fetchall()
    assert rows[0]["ticker"] == "FB" and rows[0]["valid_to"] == "2022-06-09"
    assert rows[1]["ticker"] == "META" and rows[1]["valid_from"] == "2022-06-09"


# ---------------------------------------------------------------
# 3. Missing data is detected, not silently filled
# ---------------------------------------------------------------
def test_abmd_gap_is_visible_not_filled(conn):
    sec_id = conn.execute("SELECT security_id FROM securities WHERE primary_ticker='ABMD'").fetchone()["security_id"]
    gap_rows = conn.execute(
        "SELECT COUNT(*) c FROM prices WHERE security_id=? AND date >= '2020-01-01' AND date <= '2020-06-30'",
        (sec_id,),
    ).fetchone()["c"]
    assert gap_rows == 0, "Deliberate ABMD gap should have zero rows, not be interpolated/filled"
    # but data OUTSIDE the gap must exist -> proves it's a real gap, not just missing entirely
    outside_rows = conn.execute(
        "SELECT COUNT(*) c FROM prices WHERE security_id=? AND date < '2020-01-01'", (sec_id,)
    ).fetchone()["c"]
    assert outside_rows > 0


def test_mon_has_zero_price_rows_and_is_visible_as_known_gap(conn):
    sec_id = conn.execute("SELECT security_id FROM securities WHERE primary_ticker='MON'").fetchone()["security_id"]
    price_rows = conn.execute("SELECT COUNT(*) c FROM prices WHERE security_id=?", (sec_id,)).fetchone()["c"]
    assert price_rows == 0
    # MUST still exist in securities table, not be silently absent
    sec = conn.execute("SELECT * FROM securities WHERE security_id=?", (sec_id,)).fetchone()
    assert sec is not None
    assert sec["delisted_date"] == "2018-06-07"
    assert sec["delisting_reason"] == "acquired"


# ---------------------------------------------------------------
# 4. Known delisted securities remain visible
# ---------------------------------------------------------------
def test_delisted_securities_not_deleted(conn):
    # ABC and ABX are correctly NOT delisted as of this review (both recovered
    # via the known-renames registry: ABC->COR, ABX->GOLD -- they kept trading).
    # This test previously asserted all 6 were delisted, which was itself the
    # bug this review fixed.
    for ticker in ["MON", "ABMD", "AABA", "AAMRQ"]:
        row = conn.execute("SELECT active_flag, delisted_date, delisting_confidence FROM securities WHERE primary_ticker=?", (ticker,)).fetchone()
        assert row is not None, f"{ticker} missing from securities table entirely"
        assert row["active_flag"] == 0
        assert row["delisted_date"] is not None
        assert row["delisting_confidence"] == "verified", f"{ticker} should now carry a sourced, verified delisting record"

    for ticker in ["ABC", "ABX"]:
        row = conn.execute("SELECT active_flag FROM securities WHERE primary_ticker=?", (ticker,)).fetchone()
        assert row is not None
        assert row["active_flag"] == 1, f"{ticker} was corrected: it renamed but kept trading, not delisted"


# ---------------------------------------------------------------
# 5. Point-in-time membership reconstruction
# ---------------------------------------------------------------
def test_universe_excludes_future_additions(conn):
    # ABNB added 2023-09-18: must NOT appear in the universe in 2016
    r = checks.reconstruct_universe(conn, "SP500", "2016-06-01")
    tickers = [d["ticker"] for d in r["detail"]]
    assert "ABNB" not in tickers


def test_universe_includes_past_removed_members_on_a_past_date(conn):
    # MON was a member until 2018-06-07; must appear as a member on 2016-06-01
    # even though it no longer exists today.
    r = checks.reconstruct_universe(conn, "SP500", "2016-06-01")
    tickers = [d["ticker"] for d in r["detail"]]
    assert "MON" in tickers


def test_universe_size_sane_band(conn):
    # With only 22 test securities this can't be ~500, but the MECHANISM
    # (no negative counts, no securities double counted) must hold.
    for d in ["2010-01-01", "2016-06-01", "2020-03-01"]:
        r = checks.reconstruct_universe(conn, "SP500", d)
        assert r["total_historical_constituents"] >= 0
        assert (r["constituents_with_usable_price_data"] + r["known_delisted_or_unavailable"] + r["unresolved_identifiers"]) == r["total_historical_constituents"]


# ---------------------------------------------------------------
# 6. Source hierarchy / confidence metadata preserved
# ---------------------------------------------------------------
def test_conflicting_claims_are_flagged_not_resolved(conn):
    rows = conn.execute("SELECT * FROM index_membership WHERE raw_ticker='AAL' AND confidence='conflicting'").fetchall()
    assert len(rows) == 2, "Both disagreeing AAL claims should be preserved and flagged, not merged into one"


def test_unresolved_identifier_preserved(conn):
    row = conn.execute("SELECT * FROM index_membership WHERE raw_ticker='ZZZTEST'").fetchone()
    assert row is not None, "Unresolved membership claim must not be silently dropped"
    assert row["security_id"] is None
    assert row["membership_quality"] == "unresolved"


def test_source_tier_recorded(conn):
    tiers = {r["tier"] for r in conn.execute("SELECT DISTINCT tier FROM data_sources").fetchall()}
    assert tiers.issuperset({"A", "B", "C"}), f"Expected sources across all 3 tiers, got {tiers}"


def test_no_confidence_values_outside_allowed_set(conn):
    rows = conn.execute("SELECT DISTINCT confidence FROM index_membership").fetchall()
    vals = {r["confidence"] for r in rows}
    assert vals.issubset({"verified", "unverified", "conflicting"})


# ---------------------------------------------------------------
# 7. Reproducibility
# ---------------------------------------------------------------
def test_schema_version_recorded(conn):
    row = conn.execute("SELECT schema_version FROM schema_meta").fetchone()
    assert row["schema_version"] == 1


def test_every_price_row_traceable_to_a_source(conn):
    orphans = conn.execute("""
        SELECT COUNT(*) c FROM prices p LEFT JOIN data_sources s ON p.source_id = s.source_id
        WHERE s.source_id IS NULL
    """).fetchone()["c"]
    assert orphans == 0


def test_every_membership_claim_traceable_to_a_source(conn):
    orphans = conn.execute("""
        SELECT COUNT(*) c FROM index_membership m LEFT JOIN data_sources s ON m.source_id = s.source_id
        WHERE s.source_id IS NULL
    """).fetchone()["c"]
    assert orphans == 0


# ---------------------------------------------------------------
# 8. Rename-fallback mechanism (added after real-data run surfaced FB/META)
# ---------------------------------------------------------------
def test_rename_fallback_recovers_known_rename_and_is_idempotent():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db
    from src.ingestion import pipeline as pl
    from src.ingestion.price_sources import SyntheticDemoPriceSource

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_rename_fallback.db")
    conn = init_db(test_db, _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql"), reset=True)
    seed_csv = _os.path.join(_os.path.dirname(__file__), "..", "data", "reference", "known_ticker_renames_seed.csv")
    pl.load_known_renames(conn, seed_csv)

    sec_id, _ = pl.get_or_create_security(conn, "META", name="Meta Platforms Inc")
    source = SyntheticDemoPriceSource(no_data_tickers={"FB"})

    res1 = pl.ingest_prices_with_rename_fallback(conn, sec_id, "FB", "2015-01-01", "2016-01-01", source)
    assert res1["redirect_used"] == "META"
    assert res1["bars_fetched"] > 0

    th = conn.execute("SELECT COUNT(*) c FROM ticker_history WHERE security_id=?", (sec_id,)).fetchone()["c"]
    assert th == 2  # FB window + META window, both recorded

    res2 = pl.ingest_prices_with_rename_fallback(conn, sec_id, "FB", "2015-01-01", "2016-01-01", source)
    assert res2["inserted"] == 0  # fully idempotent on re-run
    th2 = conn.execute("SELECT COUNT(*) c FROM ticker_history WHERE security_id=?", (sec_id,)).fetchone()["c"]
    assert th2 == 2  # no duplicate ticker_history rows created

    conn.close()
    _os.remove(test_db)


def test_rename_fallback_does_not_fabricate_redirect_for_unknown_ticker():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db
    from src.ingestion import pipeline as pl
    from src.ingestion.price_sources import SyntheticDemoPriceSource

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_rename_fallback2.db")
    conn = init_db(test_db, _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql"), reset=True)

    sec_id, _ = pl.get_or_create_security(conn, "MON", name="Monsanto Co")
    source = SyntheticDemoPriceSource(no_data_tickers={"MON"})  # genuinely delisted, no registry entry

    res = pl.ingest_prices_with_rename_fallback(conn, sec_id, "MON", "2015-01-01", "2016-01-01", source)
    assert res["redirect_used"] is None
    assert res["bars_fetched"] == 0

    conn.close()
    _os.remove(test_db)


# ---------------------------------------------------------------
# 9. Split adjustment (added after review flagged raw-only prices as a
#    correctness gap -- naive returns across a split date would be wrong)
# ---------------------------------------------------------------
def test_reverse_split_scales_the_correct_direction():
    """Previously unproven: reverse splits (ratio < 1) use the same formula
    as forward splits mathematically, but this was never actually tested."""
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db, now_iso
    from src.ingestion import pipeline as pl
    from src.ingestion.adjustments import compute_split_adjusted

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_reverse_split.db")
    conn = init_db(test_db, _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql"), reset=True)
    sec_id, _ = pl.get_or_create_security(conn, "RSCO", name="Reverse Split Test Co")
    source_id = pl.get_or_create_source(conn, "manual_realistic_test", "C")

    # 1-for-10 reverse split: pre-split raw price should be 1/10th of post-split
    raw_rows = [("2023-01-01", 2.00), ("2023-01-02", 20.00)]  # true 10x jump on reverse split day
    for date_, close in raw_rows:
        conn.execute(
            """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
               source_id, price_data_quality, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (sec_id, date_, close, close, close, close, 1_000_000, "raw", source_id, "ok", now_iso()),
        )
    conn.commit()
    pl.ingest_corporate_action(conn, sec_id, "reverse_split", "2023-01-02", 0.1, "1-for-10 reverse split",
                                "manual/test", "C", quality="verified")

    compute_split_adjusted(conn, sec_id)
    adj = dict(conn.execute(
        "SELECT date, close FROM prices WHERE security_id=? AND adj_type='split_adjusted' AND date IN (?,?)",
        (sec_id, "2023-01-01", "2023-01-02"),
    ).fetchall())

    # 2.00 raw pre-split -> should become 20.00 adjusted (2.00 / 0.1), matching post-split scale
    assert abs(adj["2023-01-01"] - 20.00) < 0.01, f"expected reverse-split-adjusted price ~20.00, got {adj['2023-01-01']}"
    assert abs(adj["2023-01-02"] - 20.00) < 0.01

    conn.close()
    _os.remove(test_db)


def test_total_return_reflects_dividend_reinvestment():
    """Dividend-adjusted (total-return) series: a flat raw price with
    dividends paid along the way should show earlier dates progressively
    discounted, since a real investor's total return exceeds the flat
    nominal price once dividends are reinvested."""
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db, now_iso
    from src.ingestion import pipeline as pl
    from src.ingestion.adjustments import compute_split_adjusted, compute_total_return

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_total_return.db")
    conn = init_db(test_db, _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql"), reset=True)
    sec_id, _ = pl.get_or_create_security(conn, "DIVCO", name="Dividend Test Co")
    source_id = pl.get_or_create_source(conn, "manual_realistic_test", "C")

    dates = [("2023-01-03", 100.0), ("2023-04-03", 100.0), ("2023-07-03", 100.0), ("2023-10-02", 100.0)]
    for date_, close in dates:
        conn.execute(
            """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
               source_id, price_data_quality, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (sec_id, date_, close, close, close, close, 1_000_000, "raw", source_id, "ok", now_iso()),
        )
    conn.commit()
    for ex_date in ("2023-04-03", "2023-07-03", "2023-10-02"):
        pl.ingest_corporate_action(conn, sec_id, "dividend", ex_date, 1.0, "$1 dividend",
                                    "manual/test", "C", quality="verified")

    compute_split_adjusted(conn, sec_id)
    result = compute_total_return(conn, sec_id)
    assert result["dividends_applied"] == 3

    tr = {r["date"]: r["close"] for r in conn.execute(
        "SELECT date, close FROM prices WHERE security_id=? AND adj_type='total_return' ORDER BY date", (sec_id,)
    ).fetchall()}

    # Most recent date should equal raw (nothing left to discount forward of it)
    assert abs(tr["2023-10-02"] - 100.0) < 0.01
    # Earliest date should be LOWER than the most recent -- reflecting reinvested dividends
    assert tr["2023-01-03"] < tr["2023-10-02"]
    # Monotonically non-decreasing as we move forward in time
    ordered = [tr[d] for d, _ in dates]
    assert ordered == sorted(ordered)

    conn.close()
    _os.remove(test_db)


def test_split_adjustment_removes_realistic_split_cliff():
    """This test deliberately skips SyntheticDemoPriceSource, since the
    synthetic generator is a split-unaware random walk and never produces
    a real split cliff (testing against it proved nothing during
    development and even gave a misleading result). Real, manually
    verified AAPL values are used instead so the adjustment math is
    checked independent of whichever price source supplies raw data."""
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db, now_iso
    from src.ingestion import pipeline as pl
    from src.ingestion.adjustments import compute_split_adjusted

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_split_adj.db")
    conn = init_db(test_db, _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql"), reset=True)
    sec_id, _ = pl.get_or_create_security(conn, "AAPL", name="Apple Inc")
    source_id = pl.get_or_create_source(conn, "manual_realistic_test", "C")

    # Real, confirmed pre/post-split AAPL close values (2020-08-31 4-for-1 split)
    realistic_raw = [
        ("2020-08-28", 499.23),
        ("2020-08-31", 129.04),
    ]
    for date_, close in realistic_raw:
        conn.execute(
            """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
               source_id, price_data_quality, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (sec_id, date_, close, close, close, close, 1_000_000, "raw", source_id, "ok", now_iso()),
        )
    conn.commit()
    pl.ingest_corporate_action(conn, sec_id, "split", "2020-08-31", 4.0, "4-for-1 split",
                                "manual/public record", "B", quality="verified")

    compute_split_adjusted(conn, sec_id)

    raw_return = 129.04 / 499.23 - 1  # ~ -74%, the fabricated-looking naive number
    adj = dict(conn.execute(
        "SELECT date, close FROM prices WHERE security_id=? AND adj_type='split_adjusted' AND date IN (?,?)",
        (sec_id, "2020-08-28", "2020-08-31"),
    ).fetchall())
    adjusted_return = adj["2020-08-31"] / adj["2020-08-28"] - 1

    assert raw_return < -0.5, "sanity check: the raw naive return really is a huge fake drop"
    assert abs(adjusted_return) < 0.10, (
        f"split-adjusted return across a split date should be small/realistic, got {adjusted_return:.1%}"
    )

    conn.close()
    _os.remove(test_db)


def test_universe_filters_use_config_not_hardcoded_values():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db, now_iso
    from src.ingestion import pipeline as pl
    from src.validation import checks
    from src.utils.config_loader import load_config

    config_path = _os.path.join(_os.path.dirname(__file__), "..", "config", "config.yaml")
    assert _os.path.exists(config_path), "config/config.yaml must exist -- this was a real gap flagged in review"
    config = load_config(config_path)
    assert "universe_filters" in config
    for key in ("min_price_usd", "min_avg_dollar_volume_usd", "min_historical_days"):
        assert key in config["universe_filters"]

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_filters.db")
    conn = init_db(test_db, _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql"), reset=True)
    sec_id, _ = pl.get_or_create_security(conn, "PENNY", name="Penny Stock Test")
    source_id = pl.get_or_create_source(conn, "test_source", "C")
    # One cheap, thin day of data -- should fail on price, volume, AND history-length
    conn.execute(
        """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
           source_id, price_data_quality, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (sec_id, "2023-01-01", 1.0, 1.0, 1.0, 1.0, 100, "raw", source_id, "ok", now_iso()),
    )
    conn.commit()

    result = checks.apply_universe_filters(conn, sec_id, config, adj_type="raw")
    assert result["passed"] is False
    assert result["checks"]["min_price_usd"]["passed"] is False

    conn.close()
    _os.remove(test_db)


# ---------------------------------------------------------------
# 10. Safety net: don't let a demo-data reset silently destroy real data
# ---------------------------------------------------------------
def test_init_db_refuses_to_reset_over_real_data_without_force():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db, now_iso
    from src.ingestion import pipeline as pl

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_safety_guard.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")

    conn = init_db(test_db, schema, reset=True)
    sec_id, _ = pl.get_or_create_security(conn, "REALCO", name="Real Data Co")
    source_id = pl.get_or_create_source(conn, "yfinance (Yahoo Finance, unofficial)", "C")
    conn.execute(
        """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
           source_id, price_data_quality, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (sec_id, "2023-01-01", 1, 1, 1, 1, 100, "raw", source_id, "ok", now_iso()),
    )
    conn.commit()
    conn.close()

    raised = False
    try:
        init_db(test_db, schema, reset=True)  # no force -- must refuse
    except RuntimeError:
        raised = True
    assert raised, "init_db must refuse to reset a database containing non-synthetic price data"

    conn2 = init_db(test_db, schema, reset=True, force=True)  # force -- must succeed
    n = conn2.execute("SELECT COUNT(*) c FROM prices").fetchone()["c"]
    assert n == 0
    conn2.close()
    _os.remove(test_db)


def test_init_db_allows_reset_over_synthetic_only_data():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db, now_iso
    from src.ingestion import pipeline as pl

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_safety_guard2.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")

    conn = init_db(test_db, schema, reset=True)
    sec_id, _ = pl.get_or_create_security(conn, "SYNCO", name="Synthetic Co")
    source_id = pl.get_or_create_source(conn, "SYNTHETIC_DEMO (placeholder — not real market data)", "C")
    conn.execute(
        """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
           source_id, price_data_quality, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (sec_id, "2023-01-01", 1, 1, 1, 1, 100, "raw", source_id, "ok", now_iso()),
    )
    conn.commit()
    conn.close()

    # Should NOT raise -- synthetic-only data is safe to blow away
    conn2 = init_db(test_db, schema, reset=True)
    n = conn2.execute("SELECT COUNT(*) c FROM prices").fetchone()["c"]
    assert n == 0
    conn2.close()
    _os.remove(test_db)


# ---------------------------------------------------------------
# 11. Validation checks actually catch bad data (previously unproven --
#     they had fired zero times against either synthetic or real runs)
# ---------------------------------------------------------------
def test_validate_ohlc_catches_genuinely_bad_rows():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db, now_iso
    from src.ingestion import pipeline as pl
    from src.validation import checks

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_bad_ohlc.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")
    conn = init_db(test_db, schema, reset=True)
    sec_id, _ = pl.get_or_create_security(conn, "BADCO", name="Bad Data Co")
    source_id = pl.get_or_create_source(conn, "test_source", "C")

    bad_rows = [
        # high < low: physically impossible
        ("2023-01-01", 10, 5, 8, 7, 1000),
        # high < close: impossible
        ("2023-01-02", 10, 12, 9, 15, 1000),
        # negative price
        ("2023-01-03", -5, -1, -6, -3, 1000),
        # negative volume
        ("2023-01-04", 10, 11, 9, 10, -500),
    ]
    good_row = ("2023-01-05", 10, 11, 9, 10.5, 1000)

    for date_, o, h, l, c, v in bad_rows + [good_row]:
        conn.execute(
            """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
               source_id, price_data_quality, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (sec_id, date_, o, h, l, c, v, "raw", source_id, "ok", now_iso()),
        )
    conn.commit()

    result = checks.validate_ohlc(conn)
    assert result["bad_ohlc_flagged"] >= 3, f"expected at least 3 bad OHLC rows flagged, got {result}"
    assert result["negative_volume_flagged"] >= 1

    flagged = {r["date"] for r in conn.execute(
        "SELECT date FROM prices WHERE security_id=? AND price_data_quality='suspicious'", (sec_id,)
    ).fetchall()}
    for bad_date, *_ in bad_rows:
        assert bad_date in flagged, f"{bad_date} should have been flagged suspicious"
    assert good_row[0] not in flagged, "the one genuinely good row must NOT be flagged"

    conn.close()
    _os.remove(test_db)


# ---------------------------------------------------------------
# 12. Stable identity resolution (CIK/ISIN) -- added per architecture review
# ---------------------------------------------------------------
def test_identity_resolution_unifies_renamed_entity_via_cik():
    """Positive control, using REAL data: ABC and COR share the same CIK
    (1140859, AmerisourceBergen/Cencora). Resolving both should return the
    SAME security_id -- proving CIK correctly unifies an entity across a
    ticker change, independent of the separate known_ticker_renames
    mechanism."""
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db
    from src.universe import identity_resolution as idres

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_identity_cik.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")
    conn = init_db(test_db, schema, reset=True)
    seed = _os.path.join(_os.path.dirname(__file__), "..", "data", "reference", "known_identifiers_seed.csv")
    idres.load_known_identifiers(conn, seed)

    abc_id, created1, method1 = idres.resolve_or_create_security(conn, "ABC", as_of_date="2020-01-01")
    cor_id, created2, method2 = idres.resolve_or_create_security(conn, "COR", as_of_date="2024-01-01")

    assert abc_id == cor_id, "ABC and COR share a CIK and must resolve to the SAME security_id"
    assert method1 == "cik_created_new"
    assert method2 == "cik_matched_existing"

    conn.close()
    _os.remove(test_db)


def test_identity_resolution_does_not_merge_unrelated_ticker_reuse():
    """The critical negative case: two UNRELATED companies that happened to
    use the same ticker string at different times (synthetic -- no real
    example in the current 22-security test set, but this is exactly the
    risk the architecture review flagged). Without CIK/ISIN info, the
    system must NOT silently merge them, and must flag the ambiguity."""
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db, now_iso
    from src.universe import identity_resolution as idres
    from src.ingestion import pipeline as pl

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_identity_reuse.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")
    conn = init_db(test_db, schema, reset=True)

    source_id = pl.get_or_create_source(conn, "manual_synthetic_test", "C")
    conn.execute(
        """INSERT INTO index_membership (security_id, raw_ticker, index_name, effective_date, removal_date,
           source_id, confidence, verification_status, membership_quality, ingested_at)
           VALUES (NULL, 'OLDCO', 'SP500', '1998-01-01', '2001-06-01', ?, 'unverified', 'synthetic test', 'unresolved', ?)""",
        (source_id, now_iso()),
    )
    conn.execute(
        """INSERT INTO index_membership (security_id, raw_ticker, index_name, effective_date, removal_date,
           source_id, confidence, verification_status, membership_quality, ingested_at)
           VALUES (NULL, 'OLDCO', 'SP500', '2015-01-01', '2020-01-01', ?, 'unverified', 'synthetic test', 'unresolved', ?)""",
        (source_id, now_iso()),
    )
    conn.commit()

    sec_id_1, created_1, method_1 = idres.resolve_or_create_security(conn, "OLDCO", as_of_date="1999-01-01")
    sec_id_2, created_2, method_2 = idres.resolve_or_create_security(conn, "OLDCO", as_of_date="2016-01-01")

    assert "flagged_for_review" in method_2, f"expected review flag on second resolution, got method={method_2}"
    flagged = conn.execute("SELECT * FROM identity_review_queue WHERE ticker='OLDCO'").fetchall()
    assert len(flagged) >= 1, "the gap between OLDCO's two usage periods must be flagged in identity_review_queue"
    gap_row = dict(flagged[0])
    assert gap_row["gap_days"] > 365 * 2
    assert gap_row["resolved"] == 0

    conn.close()
    _os.remove(test_db)


def test_unresolved_identity_still_gets_a_security_id_but_low_quality():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db
    from src.universe import identity_resolution as idres

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_identity_unresolved.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")
    conn = init_db(test_db, schema, reset=True)

    sec_id, created, method = idres.resolve_or_create_security(conn, "NOCIK", as_of_date="2020-01-01")
    row = conn.execute("SELECT identifier_quality FROM securities WHERE security_id=?", (sec_id,)).fetchone()
    assert row["identifier_quality"] == "unresolved"
    assert method == "ticker_fallback_new"

    conn.close()
    _os.remove(test_db)


# ---------------------------------------------------------------
# 13. Point-in-time price availability -- added per architecture review.
#     These tests specifically target look-ahead leakage.
# ---------------------------------------------------------------
def test_point_in_time_availability_never_sees_future_data():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db, now_iso
    from src.ingestion import pipeline as pl
    from src.validation import checks
    import datetime

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_pit_leakage.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")
    conn = init_db(test_db, schema, reset=True)
    sec_id, _ = pl.get_or_create_security(conn, "PITCO", name="Point in Time Test Co")
    source_id = pl.get_or_create_source(conn, "test_source", "C")

    as_of = "2023-06-01"
    d = datetime.date(2023, 5, 20)
    for i in range(10):
        conn.execute(
            """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
               source_id, price_data_quality, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (sec_id, d.isoformat(), 10, 10, 10, 10, 1000, "raw", source_id, "ok", now_iso()),
        )
        d += datetime.timedelta(days=1)
        if d.isoformat() >= as_of:
            break
    conn.commit()

    result_before = checks.check_point_in_time_availability(conn, sec_id, as_of, lookback_days=5)

    future_d = datetime.date(2023, 6, 2)
    for i in range(30):
        conn.execute(
            """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
               source_id, price_data_quality, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (sec_id, future_d.isoformat(), 999, 999, 999, 999, 999999, "raw", source_id, "ok", now_iso()),
        )
        future_d += datetime.timedelta(days=1)
    conn.commit()

    result_after = checks.check_point_in_time_availability(conn, sec_id, as_of, lookback_days=5)

    assert result_before == result_after, (
        f"Point-in-time result changed after adding FUTURE data -- LOOK-AHEAD LEAKAGE.\n"
        f"before={result_before}\nafter={result_after}"
    )

    future_count = conn.execute("SELECT COUNT(*) c FROM prices WHERE date > ?", (as_of,)).fetchone()["c"]
    assert future_count == 30

    conn.close()
    _os.remove(test_db)


def test_point_in_time_availability_dimensions():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db, now_iso
    from src.ingestion import pipeline as pl
    from src.validation import checks
    import datetime

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_pit_dims.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")
    conn = init_db(test_db, schema, reset=True)
    source_id = pl.get_or_create_source(conn, "test_source", "C")

    sec_none, _ = pl.get_or_create_security(conn, "NODATA", name="No Data Co")
    r = checks.check_point_in_time_availability(conn, sec_none, "2023-06-01", lookback_days=252)
    assert r["eligible"] is False
    assert r["has_any_historical_data"] is False

    sec_good, _ = pl.get_or_create_security(conn, "GOODCO", name="Good Data Co")
    d = datetime.date(2022, 1, 3)
    count = 0
    while count < 300:
        if d.weekday() < 5:
            conn.execute(
                """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
                   source_id, price_data_quality, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (sec_good, d.isoformat(), 10, 10, 10, 10, 100000, "raw", source_id, "ok", now_iso()),
            )
            count += 1
        d += datetime.timedelta(days=1)
    conn.commit()
    r2 = checks.check_point_in_time_availability(conn, sec_good, d.isoformat(), lookback_days=252)
    assert r2["has_any_historical_data"] is True
    assert r2["has_sufficient_lookback"] is True
    assert r2["completeness_passed"] is True
    assert r2["has_liquidity_info"] is True
    assert r2["eligible"] is True

    sec_gappy, _ = pl.get_or_create_security(conn, "GAPPYCO", name="Gappy Data Co")
    d = datetime.date(2022, 1, 3)
    while d < datetime.date(2023, 1, 3):
        if d.weekday() < 5 and not (datetime.date(2022, 6, 1) <= d <= datetime.date(2022, 9, 1)):
            conn.execute(
                """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
                   source_id, price_data_quality, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (sec_gappy, d.isoformat(), 10, 10, 10, 10, 100000, "raw", source_id, "ok", now_iso()),
            )
        d += datetime.timedelta(days=1)
    conn.commit()
    r3 = checks.check_point_in_time_availability(conn, sec_gappy, "2023-01-03", lookback_days=200,
                                                   min_completeness_pct=0.95)
    assert r3["completeness_passed"] is False, "a 3-month gap should fail a 95% completeness bar"
    assert r3["eligible"] is False

    conn.close()
    _os.remove(test_db)


# ---------------------------------------------------------------
# 14. S&P 400 modular parser -- real, sourced sample spanning multiple years
# ---------------------------------------------------------------
def test_sp400_parser_normalizes_into_same_schema_as_sp500():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db
    from src.universe.membership_sources import SP400MembershipParser
    from src.ingestion import pipeline as pl

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_sp400.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")
    conn = init_db(test_db, schema, reset=True)

    parser = SP400MembershipParser()
    real_sample = [
        {"ticker": "BRX", "effective_date": "2019-02-06",
         "source_reference": "https://www.spice-indices.com/idpfiles/spice-assets/resources/public/documents/864758_vectren4.pdf"},
        {"ticker": "FLEX", "effective_date": "2024-11-25", "announcement_date": "2024-11-19",
         "source_reference": "https://www.sdxcentral.com/press-releases/sp-dow-jones-indices-announces-changes-to-midcap-400-and-smallcap-600/"},
        {"ticker": "FTI", "effective_date": "2025-09-12", "announcement_date": "2025-09-02",
         "source_reference": "https://finance.yahoo.com/news/technipfmc-set-join-p-midcap-215400910.html"},
    ]
    records = parser.parse(manual_records=real_sample)
    assert len(records) == 3
    for r in records:
        assert r.index_name == "SP400"
        assert r.confidence == "verified"

    security_resolver = {}
    for r in real_sample:
        sec_id, _ = pl.get_or_create_security(conn, r["ticker"], name=f"{r['ticker']} (SP400 sample)")
        security_resolver[r["ticker"]] = sec_id

    result = pl.ingest_membership_records(conn, records, security_resolver)
    assert result["inserted"] == 3
    assert result["unresolved"] == 0

    rows = conn.execute("SELECT * FROM index_membership WHERE index_name='SP400'").fetchall()
    assert len(rows) == 3
    years = {r["effective_date"][:4] for r in rows}
    assert years == {"2019", "2024", "2025"}, "sample must span multiple different years, not one"

    from src.validation import checks
    universe_2025 = checks.reconstruct_universe(conn, "SP400", "2025-01-01")
    tickers_2025 = {d["ticker"] for d in universe_2025["detail"]}
    assert "BRX" in tickers_2025

    conn.close()
    _os.remove(test_db)


# ---------------------------------------------------------------
# 15. Unsupported corporate action flagging
# ---------------------------------------------------------------
def test_unsupported_corporate_action_flagging():
    """Real example: AbbVie (ABBV) was spun off from Abbott Laboratories
    on 2013-01-01/02 (confirmed via Abbott's own 8-K)."""
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db
    from src.ingestion import pipeline as pl

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_unsupported_ca.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")
    conn = init_db(test_db, schema, reset=True)

    sec_id, _ = pl.get_or_create_security(conn, "ABBV", name="AbbVie Inc")
    pl.flag_unsupported_corporate_action(
        conn, sec_id, "spinoff", "2013-01-01",
        "AbbVie spun off from Abbott Laboratories (ABT); pre-spinoff price history under ABT "
        "is not adjusted for this event. Source: Abbott 8-K Item 2.01, "
        "https://www.sec.gov/Archives/edgar/data/0000001800/000110465913001016/a13-2169_18k.htm",
        "SEC EDGAR (Abbott Laboratories 8-K)", "A",
    )

    sec = conn.execute("SELECT * FROM securities WHERE security_id=?", (sec_id,)).fetchone()
    assert sec["has_unsupported_corporate_action"] == 1
    assert "AbbVie" in sec["unsupported_corporate_action_note"]

    flagged_list = pl.securities_with_unsupported_corporate_actions(conn)
    assert len(flagged_list) == 1
    assert flagged_list[0]["primary_ticker"] == "ABBV"

    ca_row = conn.execute(
        "SELECT * FROM corporate_actions WHERE security_id=? AND action_type='spinoff'", (sec_id,)
    ).fetchone()
    assert ca_row is not None
    assert "UNSUPPORTED, FLAGGED NOT PROCESSED" in ca_row["detail"]

    conn.close()
    _os.remove(test_db)


# ---------------------------------------------------------------
# 16. Split-ratio classification (found via REAL Stage 1 data: yfinance's
#     split field picks up spinoff-driven price-adjustment factors that
#     are NOT genuine share-count splits -- e.g. PFE/Viatris, IBM/Kyndryl,
#     T/WarnerMedia all showed non-round ratios on their real spinoff dates)
# ---------------------------------------------------------------
def test_classify_split_ratio_distinguishes_genuine_from_spinoff_artifact():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.ingestion.adjustments import classify_split_ratio

    # Real, confirmed genuine splits from Stage 1 output
    for ratio in [2.0, 4.0, 5.0, 7.0, 10.0, 20.0, 1.998]:  # GOOGL 2014 was 1.998, true 2:1
        assert classify_split_ratio(ratio) == "genuine", f"{ratio} should classify as genuine"

    # Real spinoff-artifact ratios observed in actual Stage 1 yfinance data
    for ratio in [1.017, 1.046, 1.054, 1.324]:  # LEN, IBM, PFE, T respectively
        assert classify_split_ratio(ratio) == "likely_spinoff_artifact", \
            f"{ratio} should classify as a spinoff artifact, not a genuine split"

    # Reverse split reciprocal should also classify correctly
    assert classify_split_ratio(0.1) == "genuine"  # 1-for-10 reverse split


def test_spinoff_artifact_split_excluded_from_adjustment_math():
    """The critical downstream behavior: a spinoff-artifact ratio must NOT
    get applied by compute_split_adjusted(), even though it's still
    recorded in corporate_actions for the audit trail."""
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db, now_iso
    from src.ingestion import pipeline as pl
    from src.ingestion.adjustments import compute_split_adjusted

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_spinoff_artifact.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")
    conn = init_db(test_db, schema, reset=True)
    sec_id, _ = pl.get_or_create_security(conn, "PFETEST", name="Pfizer-like Test Co")
    source_id = pl.get_or_create_source(conn, "manual_realistic_test", "C")

    raw_rows = [("2020-11-16", 100.0), ("2020-11-17", 94.6)]  # ~5.4% drop, like the real PFE/Viatris date
    for date_, close in raw_rows:
        conn.execute(
            """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
               source_id, price_data_quality, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (sec_id, date_, close, close, close, close, 1_000_000, "raw", source_id, "ok", now_iso()),
        )
    conn.commit()

    # Record it with the 'likely_spinoff_artifact' quality flag, exactly as
    # run_stage_ingestion.py now does for non-round ratios
    conn.execute(
        """INSERT INTO corporate_actions
           (security_id, action_type, ex_date, ratio_or_value, detail, source_id,
            corporate_action_quality, ingested_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (sec_id, "reverse_split", "2020-11-17", 1.054, "spinoff artifact test",
         source_id, "likely_spinoff_artifact", now_iso()),
    )
    conn.commit()

    result = compute_split_adjusted(conn, sec_id)
    assert result["splits_applied"] == 0, "the spinoff-artifact ratio must NOT be applied as a real split"

    adj = dict(conn.execute(
        "SELECT date, close FROM prices WHERE security_id=? AND adj_type='split_adjusted' AND date IN (?,?)",
        (sec_id, "2020-11-16", "2020-11-17"),
    ).fetchall())
    # Split-adjusted should equal raw exactly, since no genuine split was applied
    assert abs(adj["2020-11-16"] - 100.0) < 0.01
    assert abs(adj["2020-11-17"] - 94.6) < 0.01

    conn.close()
    _os.remove(test_db)


# ---------------------------------------------------------------
# 17. Resumability: a partial failure followed by resume must not
#     re-download already-successful tickers, and must produce no
#     duplicates.
# ---------------------------------------------------------------
def test_stage_ingestion_resumes_without_reingesting_completed_tickers():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from run_stage_ingestion import run_stage
    from src.ingestion.price_sources import SyntheticDemoPriceSource

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_resume.db")
    tickers_10 = ["AAPL", "MSFT", "JNJ", "PG", "KO", "XOM", "JPM", "WMT", "HD", "DIS"]
    tickers_20 = tickers_10 + ["A", "ABBV", "FB", "META", "MON", "AAL", "ABMD", "ACGL", "ABNB", "AABA"]

    # Simulate a "crash after 10 of 20" partial run
    report1, timing1 = run_stage(1, SyntheticDemoPriceSource(), db_path=test_db, reset=True,
                                  tickers_override=tickers_10)
    assert all(not r["skipped_resume"] for r in timing1)

    # "Resume": run the full 20 against the SAME db
    report2, timing2 = run_stage(1, SyntheticDemoPriceSource(), db_path=test_db, reset=False,
                                  tickers_override=tickers_20)
    skipped = {r["ticker"] for r in timing2 if r["skipped_resume"]}
    fresh = {r["ticker"] for r in timing2 if not r["skipped_resume"]}

    assert skipped == set(tickers_10), "exactly the originally-completed tickers should be skipped"
    assert fresh == set(tickers_20) - set(tickers_10), "only the NEW tickers should be freshly fetched"

    import sqlite3
    conn = sqlite3.connect(test_db)
    dupes = conn.execute("""
        SELECT security_id, date, adj_type, source_id, COUNT(*) c FROM prices
        GROUP BY security_id, date, adj_type, source_id HAVING c > 1
    """).fetchall()
    assert len(dupes) == 0, "resume must not create duplicate price rows"
    conn.close()
    _os.remove(test_db)


# ---------------------------------------------------------------
# 18. Malformed yfinance data (found via real Stage 2 crash: PZE, SVU, TWX
#     all threw "'>' not supported between instances of 'str' and 'int'"
#     because yfinance returned a non-numeric value in the splits/dividends
#     Series for these complex/legacy-ticker securities)
# ---------------------------------------------------------------
def test_fetch_splits_skips_malformed_values_without_crashing():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from unittest.mock import patch, MagicMock
    import pandas as pd
    from src.ingestion.price_sources import YFinancePriceSource

    fake_splits = pd.Series({
        pd.Timestamp("2015-01-01"): 2.0,
        pd.Timestamp("2018-06-01"): "corrupt_value",  # the real observed failure mode
        pd.Timestamp("2020-01-01"): 4.0,
    })
    mock_ticker = MagicMock()
    mock_ticker.splits = fake_splits

    with patch("yfinance.Ticker", return_value=mock_ticker):
        src = YFinancePriceSource(verbose=False)
        result = src.fetch_splits("FAKETICKER", "2010-01-01", "2023-12-31")
        assert result == [("2015-01-01", 2.0), ("2020-01-01", 4.0)]


def test_fetch_dividends_skips_malformed_values_without_crashing():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from unittest.mock import patch, MagicMock
    import pandas as pd
    from src.ingestion.price_sources import YFinancePriceSource

    fake_dividends = pd.Series({
        pd.Timestamp("2016-01-01"): 0.5,
        pd.Timestamp("2019-01-01"): "also_corrupt",
    })
    mock_ticker = MagicMock()
    mock_ticker.dividends = fake_dividends

    with patch("yfinance.Ticker", return_value=mock_ticker):
        src = YFinancePriceSource(verbose=False)
        result = src.fetch_dividends("FAKETICKER", "2010-01-01", "2023-12-31")
        assert result == [("2016-01-01", 0.5)]


# ---------------------------------------------------------------
# 19. Report scoping -- generate_stage_report() previously queried the
#     whole database unconditionally, so a "Stage 1 report" generated
#     after Stage 2 had grown the shared database would silently report
#     Stage 2's numbers instead. Every query is now scoped to the
#     `tickers` parameter.
# ---------------------------------------------------------------
def test_stage_report_is_scoped_to_given_tickers_not_whole_database():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db, now_iso
    from src.ingestion import pipeline as pl
    from src.validation.stage_report import generate_stage_report

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_report_scoping.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")
    conn = init_db(test_db, schema, reset=True)
    source_id = pl.get_or_create_source(conn, "test_source", "C")

    # "Stage 1" tickers: 3 securities, each with 5 price rows and 1 dividend
    stage1_tickers = ["S1A", "S1B", "S1C"]
    # "Stage 2 only" tickers: 7 MORE securities, simulating a larger shared DB
    stage2_only_tickers = [f"S2X{i}" for i in range(7)]

    for ticker in stage1_tickers + stage2_only_tickers:
        sec_id, _ = pl.get_or_create_security(conn, ticker, name=ticker)
        for d in ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]:
            conn.execute(
                """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
                   source_id, price_data_quality, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (sec_id, d, 10, 10, 10, 10, 1000, "raw", source_id, "ok", now_iso()),
            )
        pl.ingest_corporate_action(conn, sec_id, "dividend", "2020-01-02", 0.5, "test div",
                                    "test_source", "C", quality="unverified")
    conn.commit()

    # Generate a report scoped ONLY to the 3 "Stage 1" tickers
    report = generate_stage_report(conn, stage1_tickers, [], "test_source", 1)

    assert report["security_counts"]["total_securities"] == 3, (
        f"Report should show 3 (Stage 1 scope), not 10 (whole DB) -- got "
        f"{report['security_counts']['total_securities']}"
    )
    assert report["price_coverage"]["total_raw_price_rows"] == 15  # 3 securities x 5 rows
    assert report["dividends"]["total_dividend_events"] == 3  # one per Stage 1 ticker, not all 10

    # Sanity check: the WHOLE database really does have 10 securities --
    # proving this test would have caught the original bug (which would
    # have returned 10 here, not 3).
    whole_db_count = conn.execute("SELECT COUNT(*) c FROM securities").fetchone()["c"]
    assert whole_db_count == 10

    conn.close()
    _os.remove(test_db)


# ---------------------------------------------------------------
# 20. Timing measurement bias -- found via real Stage 2 data: on a
#     resumed run, "freshly fetched" can be almost entirely fast
#     zero-data failures (securities retried every run since they never
#     accumulate rows), which wildly understates real per-security cost
#     if not separated from genuine successful fetches.
# ---------------------------------------------------------------
def test_timing_report_separates_zero_data_fast_failures_from_real_fetches():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.validation.stage_report import generate_stage_report
    from src.database.db import init_db
    from src.ingestion import pipeline as pl

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_timing_bias.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")
    conn = init_db(test_db, schema, reset=True)
    for t in ["REAL1", "REAL2", "ZERO1", "ZERO2", "ZERO3"]:
        pl.get_or_create_security(conn, t, name=t)
    conn.commit()

    # Simulate the exact real scenario: 2 genuine slow fetches (real data),
    # 3 fast zero-data failures, all "freshly fetched" (not resumed).
    timing_records = [
        {"ticker": "REAL1", "total_seconds": 2.3, "bars_fetched": 3500, "skipped_resume": False, "error": None},
        {"ticker": "REAL2", "total_seconds": 2.1, "bars_fetched": 3400, "skipped_resume": False, "error": None},
        {"ticker": "ZERO1", "total_seconds": 0.1, "bars_fetched": 0, "skipped_resume": False, "error": None},
        {"ticker": "ZERO2", "total_seconds": 0.1, "bars_fetched": 0, "skipped_resume": False, "error": None},
        {"ticker": "ZERO3", "total_seconds": 0.1, "bars_fetched": 0, "skipped_resume": False, "error": None},
    ]

    report = generate_stage_report(conn, ["REAL1", "REAL2", "ZERO1", "ZERO2", "ZERO3"], timing_records, "test", 1)
    t = report["timing"]

    # The naive "all fresh" average would be (2.3+2.1+0.1+0.1+0.1)/5 = 0.94 -- misleading
    naive_avg = t["avg_seconds_per_security_ALL_FRESH_STILL_MISLEADING_IF_MANY_ZERO_DATA"]
    assert abs(naive_avg - 0.94) < 0.01

    # The HONEST metric should only average the 2 real fetches: (2.3+2.1)/2 = 2.2
    honest_avg = t["avg_seconds_per_security_FRESH_WITH_DATA_ONLY_HONEST_METRIC"]
    assert abs(honest_avg - 2.2) < 0.01, f"expected honest avg ~2.2 (real fetches only), got {honest_avg}"

    # Extrapolation must use the HONEST number, not the misleading one
    extrap = report["timing"]["extrapolated_estimates_based_on_fresh_WITH_DATA_avg"]
    assert abs(extrap["200_securities_minutes"] - (2.2 * 200 / 60)) < 0.1

    conn.close()
    _os.remove(test_db)


def test_bad_ohlc_count_not_inflated_by_derived_representations():
    """A bad row in RAW data propagates the same violation into its
    derived split_adjusted/total_return copies -- the report must count
    the underlying problem once (scoped to adj_type='raw'), not up to 3x."""
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db, now_iso
    from src.ingestion import pipeline as pl
    from src.ingestion.adjustments import compute_split_adjusted, compute_total_return
    from src.validation.stage_report import generate_stage_report

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_ohlc_inflation.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")
    conn = init_db(test_db, schema, reset=True)
    sec_id, _ = pl.get_or_create_security(conn, "BADCO", name="Bad Data Co")
    source_id = pl.get_or_create_source(conn, "test_source", "C")

    # One genuinely bad raw row: high < close (impossible)
    conn.execute(
        """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
           source_id, price_data_quality, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (sec_id, "2020-01-01", 10, 10.5, 9, 11, 1000, "raw", source_id, "ok", now_iso()),
    )
    conn.commit()
    compute_split_adjusted(conn, sec_id)   # propagates the same broken relationship
    compute_total_return(conn, sec_id)     # propagates it again

    report = generate_stage_report(conn, ["BADCO"], [], "test_source", 1)
    assert report["data_quality"]["bad_ohlc_flagged"] == 1, (
        f"one bad raw row should count as ONE problem, not 3 (raw+split_adjusted+total_return), "
        f"got {report['data_quality']['bad_ohlc_flagged']}"
    )
    assert len(report["data_quality"]["flagged_rows_sample"]) == 1

    conn.close()
    _os.remove(test_db)


# ---------------------------------------------------------------
# 21. Identity review queue duplication -- found via real Stage 2 data:
#     AMD/PTC/FMC/PCG's flags doubled from 4 to 8 across two runs because
#     flag_identity_review() had no duplicate check, unlike every other
#     ingestion function in this pipeline.
# ---------------------------------------------------------------
def test_flag_identity_review_is_idempotent_across_repeated_calls():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db, now_iso
    from src.universe import identity_resolution as idres
    from src.ingestion import pipeline as pl

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_identity_dedup.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")
    conn = init_db(test_db, schema, reset=True)

    source_id = pl.get_or_create_source(conn, "test_source", "C")
    conn.execute(
        """INSERT INTO index_membership (security_id, raw_ticker, index_name, effective_date, removal_date,
           source_id, confidence, verification_status, membership_quality, ingested_at)
           VALUES (NULL, 'REUSED', 'SP500', '1998-01-01', '2001-01-01', ?, 'unverified', 'test', 'unresolved', ?)""",
        (source_id, now_iso()),
    )
    conn.execute(
        """INSERT INTO index_membership (security_id, raw_ticker, index_name, effective_date, removal_date,
           source_id, confidence, verification_status, membership_quality, ingested_at)
           VALUES (NULL, 'REUSED', 'SP500', '2015-01-01', '2020-01-01', ?, 'unverified', 'test', 'unresolved', ?)""",
        (source_id, now_iso()),
    )
    conn.commit()

    # Simulate what happens on every real stage run: resolve_or_create_security
    # gets called for the same ticker repeatedly across separate runs.
    idres.resolve_or_create_security(conn, "REUSED", as_of_date="2016-01-01")
    idres.resolve_or_create_security(conn, "REUSED", as_of_date="2016-01-01")
    idres.resolve_or_create_security(conn, "REUSED", as_of_date="2016-01-01")

    flags = conn.execute("SELECT * FROM identity_review_queue WHERE ticker='REUSED'").fetchall()
    assert len(flags) == 1, f"expected exactly 1 flag after 3 repeated calls, got {len(flags)}"

    conn.close()
    _os.remove(test_db)


# ---------------------------------------------------------------
# 22. Automated survivorship categorization (Stage 3) -- proves the
#     4-state taxonomy and 7-category classification correctly reproduce
#     the manual Stage 2 audit findings automatically, at any scale.
# ---------------------------------------------------------------
def test_survivorship_categorization_matches_real_stage2_patterns():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db, now_iso
    from src.ingestion import pipeline as pl
    from src.validation.stage_report import generate_survivorship_categorization

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_survivorship_cat.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")
    conn = init_db(test_db, schema, reset=True)
    source_id = pl.get_or_create_source(conn, "yfinance (Yahoo Finance, unofficial)", "C")

    # Case 1: JNPR-style -- SUCCESS_EMPTY_PROVIDER, no error, currently-fine
    # company. Must land in category 4 (provider_empty), NOT category 3.
    jnpr_id, _ = pl.get_or_create_security(conn, "JNPRTEST", name="Juniper-like Test Co")
    conn.execute(
        """INSERT INTO ingestion_attempts (ticker, security_id, provider, requested_start, requested_end,
           attempts, status, rows_returned, error_detail, attempted_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
        ("JNPRTEST", jnpr_id, "yfinance (Yahoo Finance, unofficial)", "2010-01-01", "2023-12-31",
         3, "SUCCESS_EMPTY_PROVIDER", 0, None, now_iso()),
    )

    # Case 2: MON-style -- sourced, verified delisting record. Category 3.
    mon_id, _ = pl.get_or_create_security(conn, "MONTEST", name="Monsanto-like Test Co")
    conn.execute(
        "UPDATE securities SET active_flag=0, delisted_date='2018-06-07', delisting_reason='acquired', "
        "delisting_confidence='verified' WHERE security_id=?", (mon_id,),
    )
    conn.execute(
        """INSERT INTO ingestion_attempts (ticker, security_id, provider, requested_start, requested_end,
           attempts, status, rows_returned, error_detail, attempted_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
        ("MONTEST", mon_id, "yfinance (Yahoo Finance, unofficial)", "2010-01-01", "2023-12-31",
         1, "PERMANENT_FAILURE", 0, "possibly delisted", now_iso()),
    )

    # Case 3: full, complete real data -- category 1
    good_id, _ = pl.get_or_create_security(conn, "GOODTEST", name="Good Data Test Co")
    import datetime
    d = datetime.date(2020, 1, 1)
    count = 0
    while count < 200:
        if d.weekday() < 5:
            conn.execute(
                """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
                   source_id, price_data_quality, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (good_id, d.isoformat(), 10, 10, 10, 10, 1000, "raw", source_id, "ok", now_iso()),
            )
            count += 1
        d += datetime.timedelta(days=1)
    conn.commit()

    # Case 4: identity-unresolved with a real review-queue flag -- category 6
    reuse_id, _ = pl.get_or_create_security(conn, "REUSETEST", name="Reuse Test Co")
    conn.execute("UPDATE securities SET identifier_quality='unresolved' WHERE security_id=?", (reuse_id,))
    conn.execute(
        """INSERT INTO identity_review_queue (ticker, flag_reason, resolved, flagged_at)
           VALUES (?,?,0,?)""", ("REUSETEST", "test gap", now_iso()),
    )
    conn.commit()

    result = generate_survivorship_categorization(
        conn, ["JNPRTEST", "MONTEST", "GOODTEST", "REUSETEST"]
    )

    assert "JNPRTEST" in result["provider_empty"]["example_tickers"], "JNPR-pattern must land in provider_empty, not no_historical_price_data"
    assert "MONTEST" in result["no_historical_price_data"]["example_tickers"]
    assert "GOODTEST" in result["full_usable_history"]["example_tickers"]
    assert "REUSETEST" in result["identity_unresolved"]["example_tickers"]

    conn.close()
    _os.remove(test_db)


# ---------------------------------------------------------------
# 23. Schema migration -- found via a real crash: an existing Stage 1/2
#     database predated the ingestion_attempts table (added for Stage 3)
#     and init_db() had no way to add it to an already-existing database.
# ---------------------------------------------------------------
def test_init_db_adds_new_tables_to_existing_database_without_data_loss():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db, now_iso
    from src.ingestion import pipeline as pl

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_migration.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")

    conn = init_db(test_db, schema, reset=True)
    sec_id, _ = pl.get_or_create_security(conn, "AAPL", name="Apple Inc")
    source_id = pl.get_or_create_source(conn, "yfinance (Yahoo Finance, unofficial)", "C")
    conn.execute(
        """INSERT INTO prices (security_id, date, open, high, low, close, volume, adj_type,
           source_id, price_data_quality, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (sec_id, "2020-01-01", 100, 100, 100, 100, 1000, "raw", source_id, "ok", now_iso()),
    )
    conn.commit()
    # Simulate "predates this table" by dropping it
    conn.execute("DROP TABLE ingestion_attempts")
    conn.commit()
    conn.close()

    # Re-open exactly as a real script would (reset=False)
    conn2 = init_db(test_db, schema, reset=False)
    tables = {r[0] for r in conn2.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "ingestion_attempts" in tables, "missing table must be added to an existing database"

    price_count = conn2.execute("SELECT COUNT(*) c FROM prices").fetchone()["c"]
    assert price_count == 1, "existing data must survive the schema update"
    sec_count = conn2.execute("SELECT COUNT(*) c FROM securities").fetchone()["c"]
    assert sec_count == 1

    conn2.close()
    _os.remove(test_db)


# ---------------------------------------------------------------
# 23. Schema migration on existing databases. init_db() previously only
#     applied schema.sql to brand-new database files, so a schema change
#     made after a database already existed (e.g. ingestion_attempts,
#     added for Stage 3) never reached it, and a real Stage 3 run crashed
#     with "no such table: ingestion_attempts". Fixed by always
#     re-applying schema.sql (IF NOT EXISTS, safe for new tables) plus
#     explicit column-level migrations for columns added to existing
#     tables.
# ---------------------------------------------------------------
def test_migration_adds_missing_tables_and_columns_without_losing_data():
    import sys, os as _os, sqlite3
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_migration.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")

    if _os.path.exists(test_db):
        _os.remove(test_db)

    # Build a deliberately OLD schema, missing later tables/columns,
    # with real data that must survive.
    conn = sqlite3.connect(test_db)
    conn.executescript("""
        CREATE TABLE schema_meta (schema_version INTEGER NOT NULL, applied_at TEXT NOT NULL);
        CREATE TABLE securities (
            security_id INTEGER PRIMARY KEY AUTOINCREMENT,
            primary_ticker TEXT NOT NULL, name TEXT, active_flag INTEGER NOT NULL DEFAULT 1,
            delisted_date TEXT, delisting_reason TEXT,
            identifier_quality TEXT NOT NULL DEFAULT 'unresolved',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX idx_securities_ticker_unique ON securities(primary_ticker);
    """)
    conn.execute(
        "INSERT INTO securities (primary_ticker, name, active_flag, created_at, updated_at) "
        "VALUES ('TESTCO', 'Test Company', 1, 'x', 'x')"
    )
    conn.commit()
    conn.close()

    # This is exactly what a real stage script does: open existing DB, reset=False
    conn = init_db(test_db, schema, reset=False)

    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "ingestion_attempts" in tables
    assert "known_identifiers" in tables
    assert "identity_review_queue" in tables

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(securities)").fetchall()}
    for c in ("cik", "has_unsupported_corporate_action", "delisting_confidence", "delisting_source"):
        assert c in cols

    # Original data must be untouched
    row = conn.execute("SELECT * FROM securities WHERE primary_ticker='TESTCO'").fetchone()
    assert row is not None
    assert row["name"] == "Test Company"

    # The exact query that crashed the real run must now work
    result = conn.execute(
        "SELECT * FROM ingestion_attempts WHERE ticker=? ORDER BY attempted_at DESC LIMIT 1", ("TESTCO",)
    ).fetchone()
    assert result is None  # no attempts logged yet, but the query itself must not raise

    conn.close()
    _os.remove(test_db)


def test_migration_is_idempotent_on_second_call():
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.database.db import init_db

    test_db = _os.path.join(_os.path.dirname(__file__), "..", "data", "database", "_test_migration2.db")
    schema = _os.path.join(_os.path.dirname(__file__), "..", "src", "database", "schema.sql")

    conn = init_db(test_db, schema, reset=True)  # fresh, already up to date
    conn.close()

    # Calling init_db again on an already-current database must not error
    conn2 = init_db(test_db, schema, reset=False)
    cols = {r["name"] for r in conn2.execute("PRAGMA table_info(securities)").fetchall()}
    assert "cik" in cols
    conn2.close()
    _os.remove(test_db)


# ---------------------------------------------------------------
# 12. run_stage_ingestion.py path resolution must not depend on cwd
# ---------------------------------------------------------------
def test_stage_ingestion_paths_are_independent_of_cwd():
    """Real bug found via a Stage 3 run: DB_PATH/SCHEMA_PATH/CSV_PATH in
    run_stage_ingestion.py used to be bare relative strings, resolved
    against the process's cwd rather than the project's own location.
    Running the script from a different cwd could then silently open or
    create a different physical database/schema/CSV file with no error,
    defeating every idempotency check that assumes "same DB across runs"
    (likely why identity_review_queue picked up duplicate flags across
    two Stage 3 attempts run from different directories).

    Invokes a fresh Python subprocess from three different working
    directories and asserts the three paths resolve identically each time.
    """
    import subprocess, sys, os as _os, tempfile

    project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    script = _os.path.join(project_root, "run_stage_ingestion.py")

    probe = (
        "import importlib.util, sys; "
        f"spec = importlib.util.spec_from_file_location('run_stage_ingestion', {script!r}); "
        "mod = importlib.util.module_from_spec(spec); "
        "sys.modules['run_stage_ingestion'] = mod; "
        "spec.loader.exec_module(mod); "
        "print(mod.DB_PATH); print(mod.SCHEMA_PATH); print(mod.CSV_PATH)"
    )

    candidate_cwds = [
        project_root,
        _os.path.dirname(project_root),
        tempfile.mkdtemp(),
    ]

    # The real entry points (run_stage1_real.py etc.) each do
    # `sys.path.insert(0, os.path.dirname(__file__))` before importing
    # run_stage_ingestion, which is what makes `from src...` importable
    # regardless of cwd. Replicate that via PYTHONPATH so this test stays
    # focused on DB_PATH/SCHEMA_PATH/CSV_PATH resolution rather than the
    # separate sys.path/import mechanism.
    env = dict(_os.environ, PYTHONPATH=project_root)

    resolved = []
    for cwd in candidate_cwds:
        result = subprocess.run(
            [sys.executable, "-c", probe], cwd=cwd, capture_output=True, text=True, timeout=30, env=env,
        )
        assert result.returncode == 0, (
            f"Import from cwd={cwd} failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        )
        lines = result.stdout.strip().splitlines()
        assert len(lines) == 3, f"Expected 3 printed paths, got: {lines!r}"
        resolved.append(tuple(_os.path.normpath(p) for p in lines))

    assert len(set(resolved)) == 1, (
        f"DB_PATH/SCHEMA_PATH/CSV_PATH differ depending on cwd -- they must not. "
        f"Got, per cwd {candidate_cwds}: {resolved}"
    )

    db_path, schema_path, csv_path = resolved[0]
    assert db_path == _os.path.normpath(
        _os.path.join(project_root, "data", "database", "quant_trader_stage.db")
    )
    assert schema_path == _os.path.normpath(
        _os.path.join(project_root, "src", "database", "schema.sql")
    )
    assert csv_path == _os.path.normpath(
        _os.path.join(project_root, "data", "raw", "sp500-master", "sp500_ticker_start_end.csv")
    )
    assert _os.path.exists(schema_path)
    assert _os.path.exists(csv_path)
