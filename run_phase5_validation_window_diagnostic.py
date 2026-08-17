"""
Phase 5 follow-up diagnostic: why did the validation window alone show a
large overnight effect (+31.76pp annualised) that neither train alone
(+0.43pp) nor train+validation combined (+6.65pp) showed?

This does NOT touch the locked test window at any point (it only ever
resolves train_dates and validation_dates from build_primary_split(), and
test_dates is used for its count only, exactly like
run_phase5_tier0_test.py).

It answers three questions about the validation window specifically:
  1. What calendar period does it actually cover? A large aggregate effect
     concentrated in a short, unusual window (e.g. a volatility event) is a
     different animal from a small, steady effect spread evenly across
     years.
  2. Is the effect spread evenly across days, or driven by a handful of
     extreme days? If excluding the top few most extreme days collapses
     the annualised estimate, the "genuine_effect_candidate" classification
     is fragile, not robust.
  3. Do the securities behind the most extreme days have any price rows
     already flagged (or newly flaggable) as data-quality 'suspicious' by
     the existing validate_ohlc() check (src/validation/checks.py) --
     i.e. is this plausibly a data artifact rather than a market effect?

Run this locally, in the same folder as the real database:
    python3 run_phase5_validation_window_diagnostic.py
"""
import sys
import os
import json
import math
import datetime
import statistics

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import yaml

from src.database.db import init_db
from src.backtest.universe import build_eligible_universe
from src.backtest.execution import next_rebalance_dates
from src.ml.walk_forward import build_primary_split
from src.ml import overnight_targets as OT
from src.validation.checks import validate_ohlc

DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "quant_trader_stage.db")
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "src", "database", "schema.sql")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")

# Must match run_phase5_tier0_test.py exactly -- this diagnostic re-derives
# the identical split so "the validation window" means the same set of
# dates in both scripts.
EMBARGO_TRADING_DAYS = 5

# How many of the most extreme single days (by |overnight_proxy|) to
# inspect and to test removal-sensitivity for.
TOP_N_EXTREME_DAYS = 20
EXCLUDE_K_VALUES = [1, 3, 5, 10, 20]


def _now():
    return datetime.datetime.utcnow().isoformat() + "Z"


def _monthly_eligible_sets(conn, cfg):
    """Identical to run_phase5_tier0_test.py's own version -- reused
    unchanged so both scripts agree on ELIG(t)."""
    bt_cfg = cfg["backtest"]
    p4_cfg = cfg["phase4"]
    predeclared_filters = cfg.get("universe_filters", {})
    start_date = cfg["data"]["price_start_date"]
    end_date = cfg["data"]["price_end_date"]
    monthly_dates = next_rebalance_dates(start_date, end_date, "monthly")
    policy = bt_cfg["data_quality_policies"][p4_cfg["primary_universe_policy"]]
    lookback_days = bt_cfg["execution"]["lookback_days_required"]

    sets_by_month_start = {}
    for d in monthly_dates:
        eligible, _ = build_eligible_universe(
            conn, d, policy, predeclared_filters=predeclared_filters,
            universe_definition="SP500", lookback_days=lookback_days,
        )
        sets_by_month_start[d] = eligible
    return sets_by_month_start


def _eligible_set_for_date(sets_by_month_start, date):
    month_starts = sorted(sets_by_month_start.keys())
    applicable = [m for m in month_starts if m <= date]
    if not applicable:
        return []
    return sets_by_month_start[applicable[-1]]


def _trading_days_in_db(conn, start_date, end_date):
    rows = conn.execute(
        "SELECT DISTINCT date FROM prices WHERE adj_type='total_return' AND date>=? AND date<=? ORDER BY date",
        (start_date, end_date),
    ).fetchall()
    return [r["date"] for r in rows]


def _per_security_overnight_on_date(conn, eligible_ids, date, prev_date, adj_type="total_return"):
    """Per-security overnight_i(t) for a single date, given prev_date
    already resolved by the caller. FIX: originally called
    OT.overnight_return() once per security, which internally re-resolves
    previous_trading_session() (an unindexed full-table scan) on every
    single call -- with ~20 extreme days x a few hundred securities each,
    that was several thousand full scans, the same N+1 pattern that made
    the original proxy_series_for_dates() slow. Since every security on a
    given date shares the same prev_date, this now takes it as a
    parameter and does two bulk 'security_id IN (...)' queries (one for
    `date`, one for `prev_date`) instead of resolving it per security."""
    if not eligible_ids or prev_date is None:
        return {}
    placeholders = ",".join("?" * len(eligible_ids))
    open_rows = conn.execute(
        f"SELECT security_id, open FROM prices WHERE adj_type=? AND date=? AND security_id IN ({placeholders})",
        (adj_type, date, *eligible_ids),
    ).fetchall()
    close_prev_rows = conn.execute(
        f"SELECT security_id, close FROM prices WHERE adj_type=? AND date=? AND security_id IN ({placeholders})",
        (adj_type, prev_date, *eligible_ids),
    ).fetchall()
    open_by_sid = {r["security_id"]: r["open"] for r in open_rows}
    close_prev_by_sid = {r["security_id"]: r["close"] for r in close_prev_rows}

    out = {}
    for sid in eligible_ids:
        o = open_by_sid.get(sid)
        c_prev = close_prev_by_sid.get(sid)
        if o is None or o <= 0 or c_prev is None or c_prev <= 0:
            continue
        out[sid] = math.log(o / c_prev)
    return out


def _security_price_quality(conn, security_id, date, adj_type="total_return"):
    row = conn.execute(
        "SELECT price_data_quality FROM prices WHERE security_id=? AND adj_type=? AND date=?",
        (security_id, adj_type, date),
    ).fetchone()
    return row["price_data_quality"] if row else None


def _ticker_for(conn, security_id):
    row = conn.execute("SELECT primary_ticker FROM securities WHERE security_id=?", (security_id,)).fetchone()
    return row["primary_ticker"] if row else str(security_id)


def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: database not found at {DB_PATH}")
        sys.exit(1)

    cfg = yaml.safe_load(open(CONFIG_PATH))
    conn = init_db(DB_PATH, SCHEMA_PATH, reset=False, force=False)

    print("Re-running validate_ohlc() so price_data_quality flags are current "
          "(flags in place, never deletes, per src/validation/checks.py)...")
    ohlc_result = validate_ohlc(conn)
    print(f"  -> {ohlc_result}")

    start_date = cfg["data"]["price_start_date"]
    end_date = cfg["data"]["price_end_date"]

    print("Computing monthly PERMISSIVE eligibility sets...")
    sets_by_month_start = _monthly_eligible_sets(conn, cfg)

    print("Resolving trading-day calendar and re-deriving the same split as "
          "run_phase5_tier0_test.py...")
    trading_days = _trading_days_in_db(conn, start_date, end_date)
    split = build_primary_split(
        trading_days,
        train_fraction=0.60, validation_fraction=0.15, test_fraction=0.25,
        embargo_periods=EMBARGO_TRADING_DAYS,
    )
    train_dates = split["train_dates"]
    validation_dates = split["validation_dates"]
    test_dates = split["test_dates"]  # count only, never touched below

    print(f"train: {train_dates[0]} .. {train_dates[-1]} ({len(train_dates)} dates)")
    print(f"validation: {validation_dates[0]} .. {validation_dates[-1]} ({len(validation_dates)} dates)")
    print(f"locked test: {len(test_dates)} dates -- NOT resolved to a range or queried by this script")

    eligible_by_date = {d: _eligible_set_for_date(sets_by_month_start, d) for d in validation_dates}
    print(f"Building overnight/intraday proxy series for the {len(validation_dates)} validation dates...")
    proxy = OT.proxy_series_for_dates(conn, eligible_by_date, validation_dates)

    # --- Q1: per-calendar-year breakdown of the validation window ---
    by_year = {}
    for d in validation_dates:
        row = proxy.get(d)
        if row is None or row["overnight_proxy"] is None:
            continue
        year = d[:4]
        by_year.setdefault(year, []).append(row["overnight_proxy"])
    year_breakdown = {
        year: {
            "n_days": len(vals),
            "mean_overnight_proxy": statistics.mean(vals),
            "annualised_pct": statistics.mean(vals) * 252 * 100,
            "sum_overnight_proxy": sum(vals),
        }
        for year, vals in sorted(by_year.items())
    }

    # --- Q2: top extreme days + removal sensitivity ---
    day_values = [
        (d, proxy[d]["overnight_proxy"], proxy[d]["n_securities"])
        for d in validation_dates
        if proxy.get(d) and proxy[d]["overnight_proxy"] is not None
    ]
    day_values_sorted_by_magnitude = sorted(day_values, key=lambda t: abs(t[1]), reverse=True)
    top_extreme_days = day_values_sorted_by_magnitude[:TOP_N_EXTREME_DAYS]

    all_vals_in_order = [v for (_, v, _) in day_values]  # not sorted by date necessarily; order doesn't matter for a mean
    full_mean = statistics.mean(all_vals_in_order) if all_vals_in_order else None
    exclusion_sensitivity = {}
    excluded_dates_sorted_by_magnitude = [d for (d, _, _) in day_values_sorted_by_magnitude]
    for k in EXCLUDE_K_VALUES:
        excluded = set(excluded_dates_sorted_by_magnitude[:k])
        remaining_vals = [v for (d, v, _) in day_values if d not in excluded]
        if remaining_vals:
            m = statistics.mean(remaining_vals)
            exclusion_sensitivity[f"exclude_top_{k}"] = {
                "n_days_remaining": len(remaining_vals),
                "mean_overnight_proxy": m,
                "annualised_pct": m * 252 * 100,
            }
        else:
            exclusion_sensitivity[f"exclude_top_{k}"] = None

    # --- Q3: data-quality cross-reference for the top extreme days ---
    extreme_day_detail = []
    for date, agg_value, n_sec in top_extreme_days:
        eligible_ids = eligible_by_date.get(date, [])
        prev_date = OT.previous_trading_session(conn, date)
        per_sec = _per_security_overnight_on_date(conn, eligible_ids, date, prev_date)
        if not per_sec:
            extreme_day_detail.append({
                "date": date, "aggregate_overnight_proxy": agg_value,
                "n_securities": n_sec, "contributing_securities": [],
            })
            continue
        # The securities most responsible for this day's aggregate move --
        # largest |individual overnight return| that day.
        ranked = sorted(per_sec.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        contributors = []
        for sid, val in ranked:
            open_quality = _security_price_quality(conn, sid, date)
            close_prev_quality = _security_price_quality(conn, sid, prev_date) if prev_date else None
            contributors.append({
                "ticker": _ticker_for(conn, sid),
                "overnight_return": val,
                "open_row_quality_on_date": open_quality,
                "close_row_quality_on_prev_date": close_prev_quality,
            })
        extreme_day_detail.append({
            "date": date,
            "prev_date": prev_date,
            "aggregate_overnight_proxy": agg_value,
            "n_securities_in_proxy": n_sec,
            "top_contributing_securities": contributors,
        })

    n_flagged_suspicious_among_contributors = sum(
        1
        for day in extreme_day_detail
        for c in day.get("top_contributing_securities", [])
        if c["open_row_quality_on_date"] == "suspicious" or c["close_row_quality_on_prev_date"] == "suspicious"
    )

    conn.close()

    report = {
        "generated_at": _now(),
        "purpose": "Explains the validation-only genuine_effect_candidate result from "
                   "PHASE5_TIER0_EXPLORATORY_REPORT.json -- does not itself constitute a "
                   "Tier 0 test result, and does not touch the locked test window.",
        "validate_ohlc_rerun_result": ohlc_result,
        "split_date_ranges": {
            "train": {"first": train_dates[0], "last": train_dates[-1], "n_dates": len(train_dates)},
            "validation": {"first": validation_dates[0], "last": validation_dates[-1], "n_dates": len(validation_dates)},
            "test_LOCKED": {"n_dates": len(test_dates), "note": "date range deliberately not resolved/printed by this script"},
        },
        "validation_full_sample_annualised_pct": full_mean * 252 * 100 if full_mean is not None else None,
        "validation_by_calendar_year": year_breakdown,
        "validation_removal_sensitivity": exclusion_sensitivity,
        "validation_top_extreme_days": [
            {"date": d, "overnight_proxy": v, "n_securities": n} for (d, v, n) in top_extreme_days
        ],
        "validation_extreme_day_detail": extreme_day_detail,
        "n_extreme_day_contributor_price_rows_flagged_suspicious": n_flagged_suspicious_among_contributors,
    }

    out_json = os.path.join(PROJECT_ROOT, "PHASE5_VALIDATION_WINDOW_DIAGNOSTIC.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nWrote {out_json}")
    print(f"\nValidation window: {validation_dates[0]} .. {validation_dates[-1]}")
    print("Per-year annualised overnight estimate:")
    for year, stats in year_breakdown.items():
        print(f"  {year}: {stats['n_days']} days, annualised {stats['annualised_pct']:.2f}pp")
    print("\nRemoval sensitivity (annualised pct after excluding the K most extreme days):")
    for k, v in exclusion_sensitivity.items():
        if v is not None:
            print(f"  {k}: {v['annualised_pct']:.2f}pp ({v['n_days_remaining']} days remaining)")
    print(f"\n{n_flagged_suspicious_among_contributors} of the top-{TOP_N_EXTREME_DAYS} extreme days' top-5 "
          f"contributing (security, date) price rows are flagged price_data_quality='suspicious'.")
    print(f"\n{len(test_dates)} locked-test dates exist and were NOT touched by this script.")


if __name__ == "__main__":
    main()
