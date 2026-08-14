"""
Sample-size / statistical-power gate, run before any Phase 4 model
training begins. The design doc's section 5 numbers were an illustrative
estimate from the 200-seed report's aggregate coverage stats (367-477
range, mean 428.5); this script recomputes them exactly, per rebalance
date, using the same build_eligible_universe() Phase 3 already validated
-- no new universe logic, no new eligibility rule.

Doesn't train anything, compute any feature, or touch any label beyond
counting how many (security, date) pairs and how many independent
time-blocks would be available. Read-only against prices/securities/
index_membership -- writes only its own report file.

Run this locally, in the same folder as the real database:
    python3 run_phase4_sample_size_check.py
"""
import sys
import os
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import yaml

from src.database.db import init_db
from src.backtest.universe import build_eligible_universe
from src.backtest.execution import next_rebalance_dates
from src.ml.walk_forward import build_primary_split

DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "quant_trader_stage.db")
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "src", "database", "schema.sql")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")


def _now():
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"


def _eligible_counts_per_date(conn, cfg, policy_name, rebalance_dates, predeclared_filters, universe_definition):
    policy = cfg["backtest"]["data_quality_policies"][policy_name]
    lookback_days = cfg["backtest"]["execution"]["lookback_days_required"]
    universe_cache = {}
    counts = {}
    for d in rebalance_dates:
        eligible, _ = build_eligible_universe(
            conn, d, policy, predeclared_filters=predeclared_filters,
            universe_definition=universe_definition, lookback_days=lookback_days,
        )
        counts[d] = len(eligible)
    return counts


def _window_stats(dates, counts_per_date, horizon_months):
    raw_pairs = sum(counts_per_date[d] for d in dates)
    effective_blocks = len(dates) // horizon_months if horizon_months > 0 else None
    return {
        "months_in_window": len(dates),
        "raw_security_date_pairs": raw_pairs,
        "effective_independent_time_blocks": effective_blocks,
    }


def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: database not found at {DB_PATH}")
        sys.exit(1)

    cfg = yaml.safe_load(open(CONFIG_PATH))
    bt_cfg = cfg["backtest"]
    p4_cfg = cfg["phase4"]
    predeclared_filters = cfg.get("universe_filters", {})
    start_date = cfg["data"]["price_start_date"]
    end_date = cfg["data"]["price_end_date"]
    rebalance_dates = next_rebalance_dates(start_date, end_date, bt_cfg["rebalance_frequency"])
    universe_definition = "SP500"

    primary_horizon = p4_cfg["target"]["primary_horizon_months"]
    secondary_horizons = p4_cfg["target"]["secondary_horizon_months"]
    embargo = primary_horizon  # config.yaml: embargo_purge_months == "primary_horizon_months"

    conn = init_db(DB_PATH, SCHEMA_PATH, reset=False, force=False)

    report = {"generated_at": _now(), "rebalance_date_count": len(rebalance_dates), "policies": {}}

    for policy_name in [p4_cfg["primary_universe_policy"], p4_cfg["sanity_check_policy"]]:
        print(f"=== {policy_name} ===")
        t0 = time.time()
        counts_per_date = _eligible_counts_per_date(conn, cfg, policy_name, rebalance_dates,
                                                      predeclared_filters, universe_definition)
        elapsed = time.time() - t0
        print(f"  eligible-universe count computed for all {len(rebalance_dates)} dates in {elapsed:.1f}s "
              f"(min={min(counts_per_date.values())}, max={max(counts_per_date.values())}, "
              f"mean={sum(counts_per_date.values())/len(counts_per_date):.1f})")

        split = build_primary_split(
            rebalance_dates,
            p4_cfg["walk_forward"]["train_fraction"],
            p4_cfg["walk_forward"]["validation_fraction"],
            p4_cfg["walk_forward"]["test_fraction"],
            embargo_periods=embargo,
        )

        horizons_block = {}
        for h in [primary_horizon] + list(secondary_horizons):
            horizons_block[str(h)] = {
                "train": _window_stats(split["train_dates"], counts_per_date, h),
                "validation": _window_stats(split["validation_dates"], counts_per_date, h),
                "test_LOCKED": _window_stats(split["test_dates"], counts_per_date, h),
            }

        max_confirmatory = p4_cfg["experiment_tracking"]["max_confirmatory_test_experiments"]
        primary_test_blocks = horizons_block[str(primary_horizon)]["test_LOCKED"]["effective_independent_time_blocks"]
        gate_ok = primary_test_blocks is not None and max_confirmatory <= primary_test_blocks

        report["policies"][policy_name] = {
            "eligible_count_min": min(counts_per_date.values()),
            "eligible_count_max": max(counts_per_date.values()),
            "eligible_count_mean": round(sum(counts_per_date.values()) / len(counts_per_date), 1),
            "split": {
                "train_dates_count": len(split["train_dates"]),
                "train_purged_count": len(split["train_purged_dates"]),
                "validation_dates_count": len(split["validation_dates"]),
                "validation_purged_count": len(split["validation_purged_dates"]),
                "test_dates_count": len(split["test_dates"]),
                "test_date_range": [split["test_dates"][0], split["test_dates"][-1]] if split["test_dates"] else None,
            },
            "by_horizon_months": horizons_block,
            "confirmatory_experiment_cap_check": {
                "configured_max_confirmatory_test_experiments": max_confirmatory,
                "effective_independent_test_blocks_at_primary_horizon": primary_test_blocks,
                "cap_is_within_available_evidence": gate_ok,
                "note": ("PASS: the configured confirmatory-experiment cap does not exceed the number of "
                         "independent test blocks available." if gate_ok else
                         "FAIL: the configured cap exceeds the independent test blocks available at the "
                         "primary horizon -- config.yaml's phase4.experiment_tracking.max_confirmatory_test_experiments "
                         "must be reduced, or this is not a valid gate pass, before any confirmatory experiment "
                         "touches the locked test set."),
            },
        }

    conn.close()

    out_json = os.path.join(PROJECT_ROOT, "PHASE4_SAMPLE_SIZE_REPORT.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)

    primary_policy_result = report["policies"][p4_cfg["primary_universe_policy"]]
    gate_pass = primary_policy_result["confirmatory_experiment_cap_check"]["cap_is_within_available_evidence"]
    print(f"\nWrote {out_json}")
    print(f"Gate check ({p4_cfg['primary_universe_policy']}, primary horizon={primary_horizon}m): "
          f"{'PASS' if gate_pass else 'FAIL -- see note in report'}")
    print("This script trains nothing and computes no feature/label.")


if __name__ == "__main__":
    main()
