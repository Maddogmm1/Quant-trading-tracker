"""
Phase 5 Tier 0 primary test -- runs ONLY against train+validation dates.

Per PHASE5_OVERNIGHT_GAP_SPECIFICATION.md section 12 ("Tier 0 ->
locked-test evaluation requires... explicit written confirmation"), this
script does not compute, query, or touch a single overnight/intraday
return for any date in the locked test window: `test_dates` from
`walk_forward.build_primary_split()` is resolved and its COUNT is
reported (for context), but the dates themselves are never passed to the
proxy-series builder. This mirrors PHASE4_CONCLUSION_3M.md section 11's
discipline exactly ("The locked test period was not read, queried, or
evaluated at any point in this branch").

This is an EXPLORATORY run -- no `ml_experiments` row is written by this
script, and no result here is a confirmatory finding (spec sections
12/18: that requires a separately pre-registered, frozen config, logged
before the locked test set is ever touched). Its purpose is to produce
the first real train/validation read of whether the Tier 0 test shows
anything worth taking to a confirmatory locked-test evaluation.

Run this locally, in the same folder as the real database:
    python3 run_phase5_tier0_test.py
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
from src.ml.walk_forward import build_primary_split
from src.ml import overnight_targets as OT
from src.ml import overnight_significance as SIG

DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "quant_trader_stage.db")
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "src", "database", "schema.sql")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")

# Spec section 4.4 proposed default -- open item pending your sign-off
# (spec section 12 gate (d)). Change here (and re-run) if you approve a
# different value.
EMBARGO_TRADING_DAYS = 5


def _now():
    return datetime.datetime.utcnow().isoformat() + "Z"


def _monthly_eligible_sets(conn, cfg):
    """PERMISSIVE eligibility, refreshed monthly -- identical cadence and
    mechanism to run_phase5_sample_size_report.py's own
    _monthly_eligible_sets(), reused here rather than re-derived, since
    both scripts must agree on ELIG(t) for their numbers to be comparable."""
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


def _series_for(proxy, dates):
    overnight, intraday, used_dates, n_missing_days = [], [], [], 0
    for d in dates:
        row = proxy.get(d)
        if row is None or row["overnight_proxy"] is None or row["intraday_proxy"] is None:
            n_missing_days += 1
            continue
        overnight.append(row["overnight_proxy"])
        intraday.append(row["intraday_proxy"])
        used_dates.append(d)
    return overnight, intraday, used_dates, n_missing_days


def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: database not found at {DB_PATH}")
        sys.exit(1)

    cfg = yaml.safe_load(open(CONFIG_PATH))
    conn = init_db(DB_PATH, SCHEMA_PATH, reset=False, force=False)

    start_date = cfg["data"]["price_start_date"]
    end_date = cfg["data"]["price_end_date"]

    print("Computing monthly PERMISSIVE eligibility sets...")
    sets_by_month_start = _monthly_eligible_sets(conn, cfg)

    print("Resolving trading-day calendar...")
    trading_days = _trading_days_in_db(conn, start_date, end_date)

    # Chronological 60/15/25 split with an embargo at each boundary (spec
    # section 4.4), reusing build_primary_split() unchanged (spec section
    # 0.1) -- trading_days plays the role Phase 4's rebalance_dates
    # played, just at daily instead of monthly granularity.
    split = build_primary_split(
        trading_days,
        train_fraction=0.60, validation_fraction=0.15, test_fraction=0.25,
        embargo_periods=EMBARGO_TRADING_DAYS,
    )
    train_dates = split["train_dates"]
    validation_dates = split["validation_dates"]
    test_dates = split["test_dates"]  # NEVER passed to proxy_series_for_dates below

    exploratory_dates = train_dates + validation_dates
    eligible_by_date = {d: _eligible_set_for_date(sets_by_month_start, d) for d in exploratory_dates}

    print(f"Building overnight/intraday proxy series for {len(exploratory_dates)} "
          f"train+validation dates ONLY -- {len(test_dates)} locked-test dates exist "
          f"and are NOT queried by this script.")
    proxy = OT.proxy_series_for_dates(conn, eligible_by_date, exploratory_dates)

    conn.close()  # no further DB access below -- everything from here works on already-fetched data

    train_overnight, train_intraday, train_used, train_missing = _series_for(proxy, train_dates)
    val_overnight, val_intraday, val_used, val_missing = _series_for(proxy, validation_dates)
    combined_overnight = train_overnight + val_overnight
    combined_intraday = train_intraday + val_intraday

    print("Running Tier 0 block-bootstrap test on TRAIN...")
    train_result = SIG.tier0_test(train_overnight, train_intraday)
    print("Running Tier 0 block-bootstrap test on VALIDATION...")
    val_result = SIG.tier0_test(val_overnight, val_intraday)
    print("Running Tier 0 block-bootstrap test on TRAIN+VALIDATION combined...")
    combined_result = SIG.tier0_test(combined_overnight, combined_intraday)

    report = {
        "generated_at": _now(),
        "status": "EXPLORATORY -- train/validation only. The locked test window "
                  "was never queried or computed by this script.",
        "embargo_trading_days": EMBARGO_TRADING_DAYS,
        "split_sizes": {
            "train_dates": len(train_dates), "train_purged": len(split["train_purged_dates"]),
            "validation_dates": len(validation_dates), "validation_purged": len(split["validation_purged_dates"]),
            "test_LOCKED_dates_NOT_QUERIED": len(test_dates),
        },
        "train": {"result": train_result, "n_dates_used": len(train_used), "n_dates_missing_proxy": train_missing},
        "validation": {"result": val_result, "n_dates_used": len(val_used), "n_dates_missing_proxy": val_missing},
        "train_plus_validation_combined": {
            "result": combined_result,
            "n_dates_used": len(train_used) + len(val_used),
        },
    }

    out_json = os.path.join(PROJECT_ROOT, "PHASE5_TIER0_EXPLORATORY_REPORT.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nWrote {out_json}")
    print(f"TRAIN classification: {train_result['classification']} ({train_result['classification_reason']})")
    print(f"VALIDATION classification: {val_result['classification']} ({val_result['classification_reason']})")
    print(f"COMBINED classification: {combined_result['classification']} ({combined_result['classification_reason']})")
    print(f"\n{len(test_dates)} locked-test dates exist and were NOT touched by this script.")
    print("This is an EXPLORATORY result. No confirmatory experiment has been registered or run.")


if __name__ == "__main__":
    main()
