"""
Phase 5 pre-registration diagnostic #1: does the existing database have
reliable point-in-time daily OPEN prices, not just close/adjusted-close?

This is a read-only diagnostic, matching the pattern of
run_phase4_sample_size_check.py: it trains nothing, computes no feature,
touches no label, and writes only its own report file.

Answers, per adj_type ('raw', 'split_adjusted', 'total_return'):
  1. What fraction of price rows have a non-NULL, positive `open`?
  2. Is that coverage materially worse than `close`'s coverage (i.e. is
     `open` a second-class citizen in the ingested data)?
  3. Does the split/dividend back-adjustment math applied to `open` in
     src/ingestion/adjustments.py (same price_factor / cum_factor as
     `close`) produce internally consistent series -- e.g. does
     total_return open ever exceed total_return high, or fall below
     total_return low, which would indicate the OHLC relationship breaks
     under adjustment even if it held for the raw row?
  4. How many PERMISSIVE-eligible (security, date) pairs across the full
     2015-2023 window would be excluded if a daily open-price-availability
     filter were added on top of the existing eligibility checks (which
     today only require `close` and `volume`)?
  5. A sample of corporate-action ex-dates: does the raw `open` on the
     ex_date itself look consistent with the ex-dividend/split adjustment
     the market would apply (a sanity spot-check, not a proof) -- flagged
     for human review, not auto-corrected.

Run this locally, in the same folder as the real database:
    python3 run_phase5_open_price_data_check.py
"""
import sys
import os
import json
import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import yaml

from src.database.db import init_db
from src.backtest.universe import build_eligible_universe
from src.backtest.execution import next_rebalance_dates

DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "quant_trader_stage.db")
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "src", "database", "schema.sql")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")


def _now():
    return datetime.datetime.utcnow().isoformat() + "Z"


def _open_vs_close_coverage(conn):
    out = {}
    for adj_type in ("raw", "split_adjusted", "total_return"):
        row = conn.execute(
            """SELECT COUNT(*) total,
                      SUM(CASE WHEN open IS NOT NULL AND open > 0 THEN 1 ELSE 0 END) open_ok,
                      SUM(CASE WHEN close IS NOT NULL AND close > 0 THEN 1 ELSE 0 END) close_ok
               FROM prices WHERE adj_type=?""",
            (adj_type,),
        ).fetchone()
        total = row["total"] or 0
        out[adj_type] = {
            "total_rows": total,
            "open_populated": row["open_ok"] or 0,
            "open_populated_pct": round((row["open_ok"] or 0) / total, 4) if total else None,
            "close_populated": row["close_ok"] or 0,
            "close_populated_pct": round((row["close_ok"] or 0) / total, 4) if total else None,
        }
    return out


def _adjusted_open_relationship_violations(conn):
    """After split/dividend adjustment, does open ever fall outside
    [low, high] for the SAME row it was derived from? adjustments.py
    scales open/high/low/close by the identical factor per row, so a
    violation here would mean the row was already broken before scaling
    (i.e. this re-surfaces raw bad-OHLC rows) OR that the per-row scaling
    itself introduced an inconsistency (it should not, since one shared
    scalar can't change intra-row ordering) -- distinguishing the two is
    the point of comparing against the raw-row count."""
    out = {}
    for adj_type in ("raw", "split_adjusted", "total_return"):
        row = conn.execute(
            """SELECT COUNT(*) c FROM prices
               WHERE adj_type=? AND open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL
                 AND (open > high OR open < low)""",
            (adj_type,),
        ).fetchone()
        out[adj_type] = row["c"]
    return out


def _resolve_to_trading_session(conn, nominal_date, adj_type="total_return"):
    """Resolves a nominal (possibly non-trading) calendar date to the
    market-wide next actual trading session on-or-after it -- the earliest
    date, across every security, on which any security recorded a price.
    Mirrors src.backtest.execution.next_market_session()'s market-wide
    resolution logic, but inclusive of nominal_date itself (that function
    is strictly-after, meant for post-signal execution dates; here we want
    "the real session this nominal date actually refers to", which could
    be nominal_date itself if it happens to be a trading day).

    FIX (see PHASE5_OPEN_PRICE_DATA_CHECK.json's first real run): the
    original version of this function queried `prices` for an EXACT match
    on the raw rebalance-date string (e.g. "2015-02-01"), most of which
    are weekends/holidays -- days on which NO security has ANY price row
    at all. That inflated the "missing open" rate for reasons having
    nothing to do with open-price coverage specifically. Resolving to the
    actual session first, exactly once per date rather than once per
    security, fixes this without changing the eligibility logic itself."""
    row = conn.execute(
        "SELECT MIN(date) d FROM prices WHERE adj_type=? AND date>=?",
        (adj_type, nominal_date),
    ).fetchone()
    return row["d"] if row and row["d"] else None


def _eligible_pairs_with_and_without_open_filter(conn, cfg):
    """For a sample of monthly PERMISSIVE-eligible dates (reusing the
    existing eligibility mechanism unchanged), how many eligible
    (security, date) pairs would additionally fail a same-day open-price
    requirement -- measured against the REAL resolved trading session for
    each nominal date, not the literal calendar string. Sampled at Phase
    4's existing monthly rebalance cadence, since re-deriving eligibility
    at daily cadence is exactly the kind of new code this diagnostic
    exists to justify (or not) -- not assumed here."""
    bt_cfg = cfg["backtest"]
    p4_cfg = cfg["phase4"]
    predeclared_filters = cfg.get("universe_filters", {})
    start_date = cfg["data"]["price_start_date"]
    end_date = cfg["data"]["price_end_date"]
    rebalance_dates = next_rebalance_dates(start_date, end_date, "monthly")
    policy = bt_cfg["data_quality_policies"][p4_cfg["primary_universe_policy"]]
    lookback_days = bt_cfg["execution"]["lookback_days_required"]

    total_eligible_pairs = 0
    missing_open_on_date_pairs = 0
    dates_sampled = 0
    dates_with_no_resolvable_session = 0
    for d in rebalance_dates:
        eligible, _ = build_eligible_universe(
            conn, d, policy, predeclared_filters=predeclared_filters,
            universe_definition="SP500", lookback_days=lookback_days,
        )
        if not eligible:
            continue
        resolved_date = _resolve_to_trading_session(conn, d)
        if resolved_date is None:
            dates_with_no_resolvable_session += 1
            continue
        dates_sampled += 1
        total_eligible_pairs += len(eligible)
        placeholders = ",".join("?" * len(eligible))
        rows = conn.execute(
            f"""SELECT security_id FROM prices
                WHERE security_id IN ({placeholders}) AND adj_type='total_return' AND date=?
                  AND open IS NOT NULL AND open > 0""",
            (*eligible, resolved_date),
        ).fetchall()
        has_open = {r["security_id"] for r in rows}
        missing_open_on_date_pairs += len(eligible) - len(has_open)

    return {
        "dates_sampled": dates_sampled,
        "dates_with_no_resolvable_session": dates_with_no_resolvable_session,
        "total_eligible_security_date_pairs": total_eligible_pairs,
        "pairs_missing_open_on_resolved_session": missing_open_on_date_pairs,
        "pairs_missing_open_pct": round(missing_open_on_date_pairs / total_eligible_pairs, 4)
        if total_eligible_pairs else None,
        "note": "FIXED vs the first run: dates are resolved to the real trading session "
                "on-or-after each nominal monthly date before checking open availability, "
                "instead of matching the literal (often non-trading) calendar date string.",
    }


def _reconcile_open_violations_against_known_bad_ohlc(conn):
    """The 687 adjusted-open-outside-[low,high] rows found in the first
    run were expected to be a subset of the already-known, already-
    documented raw bad-OHLC rows (BAD_OHLC_INVESTIGATION.md, ~1,227 raw
    rows flagged price_data_quality='suspicious' by validate_ohlc() in
    src/validation/checks.py) rather than a new, separate problem -- this
    function checks that directly instead of leaving it asserted."""
    row = conn.execute(
        """SELECT COUNT(*) c FROM prices
           WHERE adj_type='raw' AND open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL
             AND (open > high OR open < low)"""
    ).fetchone()
    total_open_violations_raw = row["c"]

    row2 = conn.execute(
        """SELECT COUNT(*) c FROM prices
           WHERE adj_type='raw' AND open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL
             AND (open > high OR open < low) AND price_data_quality='suspicious'"""
    ).fetchone()
    already_flagged = row2["c"]

    row3 = conn.execute(
        "SELECT COUNT(*) c FROM prices WHERE adj_type='raw' AND price_data_quality='suspicious'"
    ).fetchone()
    total_known_bad_ohlc_raw = row3["c"]

    return {
        "total_open_outside_high_low_raw": total_open_violations_raw,
        "already_flagged_suspicious": already_flagged,
        "not_previously_flagged": total_open_violations_raw - already_flagged,
        "total_known_bad_ohlc_rows_raw_all_reasons": total_known_bad_ohlc_raw,
        "conclusion": (
            "CONFIRMED subset of the known/documented bad-OHLC set -- no new problem."
            if already_flagged == total_open_violations_raw else
            "NOT fully a subset -- some open-outside-[low,high] rows were not already "
            "flagged by validate_ohlc(); these are a genuinely new finding and need "
            "their own root-cause look, same treatment as BAD_OHLC_INVESTIGATION.md's "
            "original TNB/CFC/RYC/PZE/PBG cases."
        ),
    }


def _corporate_action_open_spotcheck(conn, sample_n=15):
    """Sample of ex-dates for splits/dividends: report the raw open on the
    ex_date and the prior session's raw close, for human eyeballing. Not
    an automated pass/fail -- the point is surfacing whether OPEN on an
    ex-date session looks like a plausible, already-adjusted market print
    (which it should, since it's a real traded price) rather than an
    ingestion artifact."""
    rows = conn.execute(
        """SELECT ca.security_id, s.primary_ticker, ca.action_type, ca.ex_date, ca.ratio_or_value
           FROM corporate_actions ca JOIN securities s ON ca.security_id = s.security_id
           WHERE ca.action_type IN ('split', 'reverse_split', 'dividend', 'special_dividend')
             AND ca.corporate_action_quality != 'likely_spinoff_artifact'
           ORDER BY RANDOM() LIMIT ?""",
        (sample_n,),
    ).fetchall()
    out = []
    for r in rows:
        ex_open = conn.execute(
            "SELECT date, open FROM prices WHERE security_id=? AND adj_type='raw' AND date>=? "
            "ORDER BY date ASC LIMIT 1",
            (r["security_id"], r["ex_date"]),
        ).fetchone()
        prior_close = conn.execute(
            "SELECT date, close FROM prices WHERE security_id=? AND adj_type='raw' AND date<? "
            "ORDER BY date DESC LIMIT 1",
            (r["security_id"], r["ex_date"]),
        ).fetchone()
        out.append({
            "ticker": r["primary_ticker"], "action_type": r["action_type"],
            "ex_date": r["ex_date"], "ratio_or_value": r["ratio_or_value"],
            "resolved_ex_session_date": ex_open["date"] if ex_open else None,
            "raw_open_on_ex_session": ex_open["open"] if ex_open else None,
            "raw_close_prior_session": prior_close["close"] if prior_close else None,
        })
    return out


def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: database not found at {DB_PATH}")
        sys.exit(1)

    cfg = yaml.safe_load(open(CONFIG_PATH))
    conn = init_db(DB_PATH, SCHEMA_PATH, reset=False, force=False)

    report = {
        "generated_at": _now(),
        "script_version": 2,  # v2: fixed exact-date resolution bug + added bad-OHLC reconciliation
        "coverage_by_adj_type": _open_vs_close_coverage(conn),
        "adjusted_ohlc_open_outside_high_low_count": _adjusted_open_relationship_violations(conn),
        "bad_ohlc_reconciliation": _reconcile_open_violations_against_known_bad_ohlc(conn),
        "eligible_pairs_open_availability": _eligible_pairs_with_and_without_open_filter(conn, cfg),
        "corporate_action_open_spotcheck_sample": _corporate_action_open_spotcheck(conn),
    }
    conn.close()

    out_json = os.path.join(PROJECT_ROOT, "PHASE5_OPEN_PRICE_DATA_CHECK.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nWrote {out_json}")
    for adj_type, stats in report["coverage_by_adj_type"].items():
        print(f"  {adj_type}: open_populated_pct={stats['open_populated_pct']}, "
              f"close_populated_pct={stats['close_populated_pct']}")
    print(f"  bad-OHLC reconciliation: {report['bad_ohlc_reconciliation']['conclusion']}")
    print(f"  pairs missing open on resolved session: "
          f"{report['eligible_pairs_open_availability']['pairs_missing_open_pct']}")
    print("This script computes no feature and trains no model.")


if __name__ == "__main__":
    main()
