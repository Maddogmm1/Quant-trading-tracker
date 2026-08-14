"""
Phase 3 validation run: the benchmark suite Phase 3 stops at (see
PHASE3_SPECIFICATION.md for the full rationale). Runs the 4 required
benchmarks -- buy-and-hold, equal-weight-S&P500, top-20-momentum,
bottom-20-momentum -- plus the random-selection distribution, once under
the STRICT data-quality policy and once under PERMISSIVE, over the full
priced date range in config.yaml (data.price_start_date ..
data.price_end_date), rebalanced monthly. Persists every run to
backtest_runs/backtest_results/backtest_coverage/backtest_positions and
writes a combined markdown + JSON report.

This does not build or evaluate any predictive model -- that's out of
scope for this script; a strategy beyond the four benchmarks + random
selection belongs in a separate script.

Run this LOCALLY, in the same folder as your real database:
    python3 run_phase3_validation.py

By default this runs the FULL random-benchmark distribution (200 seeds,
per config.yaml's backtest.random_benchmark.distribution_seed_count) for
BOTH policies -- with ~1,200 securities and ~9 years of monthly
rebalancing this can take a while (a per-run progress line prints so you
can gauge pace). If you want a fast first pass to sanity-check the
mechanics before committing to the full distribution, pass --quick-random
to cut the distribution down to 20 seeds (still enough for a rough
mean/median, just not the full spec-required 200) -- the report notes
which mode produced it.

Nothing here mutates prices/, securities/, index_membership/, or any
Phase 1/2 table. It only INSERTs into the 4 new backtest_* tables.
"""
import sys
import os
import json
import time
import argparse
import statistics

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import sqlite3
import yaml

from src.database.db import init_db
from src.backtest.universe import build_eligible_universe
from src.backtest.execution import next_rebalance_dates
from src.backtest.engine import run_backtest
from src.backtest.benchmarks import BuyAndHold, EqualWeightSP500, TopNMomentum, BottomNMomentum, RandomSelection
from src.backtest.reproducibility import save_run, save_positions

DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "quant_trader_stage.db")
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "src", "database", "schema.sql")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")


def _now():
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"


def _connect():
    # init_db re-applies schema.sql (every CREATE TABLE/INDEX uses IF NOT
    # EXISTS) and runs apply_migrations() -- this is what actually adds
    # the 4 new backtest_* tables and the new prices composite index to
    # your existing database, without touching any existing data. Same
    # idempotent pattern run_stage_ingestion.py already relies on.
    conn = init_db(DB_PATH, SCHEMA_PATH, reset=False, force=False)
    return conn


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _run_one(conn, cfg, policy_name, policy, strategy, rebalance_dates, predeclared_filters,
             starting_cash, universe_definition, run_label, universe_cache=None):
    t0 = time.time()
    report, coverage_report, portfolio = run_backtest(
        conn, strategy, rebalance_dates, policy,
        cost_config=cfg["backtest"]["costs"],
        predeclared_filters=predeclared_filters,
        starting_cash=starting_cash,
        lookback_days=cfg["backtest"]["execution"]["lookback_days_required"],
        universe_definition=universe_definition,
        universe_cache=universe_cache,
    )
    elapsed = time.time() - t0

    run_id = save_run(
        conn, run_label=run_label, config=cfg, random_seed=getattr(strategy, "seed", None),
        start_date=rebalance_dates[0], end_date=rebalance_dates[-1],
        rebalance_frequency=cfg["backtest"]["rebalance_frequency"],
        universe_definition=universe_definition, data_quality_policy_name=policy_name,
        strategy_name=strategy.name, cost_config=cfg["backtest"]["costs"],
        execution_config=cfg["backtest"]["execution"], report=report,
        coverage_report=coverage_report, created_at=_now(),
    )
    save_positions(conn, run_id, portfolio.history)

    return {
        "run_id": run_id, "strategy": strategy.name, "policy": policy_name,
        "elapsed_seconds": round(elapsed, 1), "report": report,
        "coverage_summary": _summarize_coverage(coverage_report),
    }


def _summarize_coverage(coverage_report):
    if not coverage_report:
        return {}
    final = [c["final_tradable_count"] for c in coverage_report]
    return {
        "num_rebalance_dates": len(coverage_report),
        "min_final_tradable_count": min(final),
        "max_final_tradable_count": max(final),
        "mean_final_tradable_count": round(statistics.mean(final), 1),
        "dates_with_zero_eligible": sum(1 for f in final if f == 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick-random", action="store_true",
                     help="Use 20 random-benchmark seeds instead of the full 200 for a fast sanity pass.")
    args = ap.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"ERROR: database not found at {DB_PATH}")
        print("This script expects the real quant_trader_stage.db in data/database/, same as run_stage3_real.py.")
        sys.exit(1)

    cfg = _load_config()
    bt_cfg = cfg["backtest"]
    predeclared_filters = cfg.get("universe_filters", {})
    start_date = cfg["data"]["price_start_date"]
    end_date = cfg["data"]["price_end_date"]
    rebalance_dates = next_rebalance_dates(start_date, end_date, bt_cfg["rebalance_frequency"])
    universe_definition = "SP500"
    starting_cash = 100000.0

    seed_count = 20 if args.quick_random else bt_cfg["random_benchmark"]["distribution_seed_count"]
    # Deterministic, pre-declared seed set derived from the fixed headline
    # seed -- generated once here, never re-picked after seeing results.
    rng_for_seeds = __import__("random").Random(bt_cfg["random_benchmark"]["headline_seed"])
    distribution_seeds = [rng_for_seeds.randint(0, 2**31 - 1) for _ in range(seed_count)]
    headline_seed = bt_cfg["random_benchmark"]["headline_seed"]
    portfolio_size = bt_cfg["portfolio"]["default_size"]

    print(f"Database: {DB_PATH}")
    print(f"Backtest period: {start_date} .. {end_date}, {bt_cfg['rebalance_frequency']} rebalancing "
          f"({len(rebalance_dates)} rebalance dates)")
    print(f"Random benchmark: {seed_count} seeds "
          f"({'QUICK mode -- 20 seeds, not the full 200' if args.quick_random else 'full spec-required 200'})\n")

    conn = _connect()
    all_results = {"STRICT": {}, "PERMISSIVE": {}}

    for policy_name in ["STRICT", "PERMISSIVE"]:
        policy = bt_cfg["data_quality_policies"][policy_name]
        print(f"=== Policy: {policy_name} ===")
        # build_eligible_universe(date, policy, ...) is a pure function of
        # date and policy, independent of strategy or seed, so one cache
        # shared across every strategy/seed at this policy is a safe,
        # results-preserving speedup (see tests/test_backtest_caching.py).
        # A fresh cache per policy avoids any STRICT/PERMISSIVE key collision.
        universe_cache = {}

        strategies = [
            BuyAndHold(),
            EqualWeightSP500(),
            TopNMomentum(n=bt_cfg["benchmarks"]["top_n_momentum"]["n"],
                         lookback_days=bt_cfg["benchmarks"]["top_n_momentum"]["lookback_days"]),
            BottomNMomentum(n=bt_cfg["benchmarks"]["bottom_n_momentum"]["n"],
                             lookback_days=bt_cfg["benchmarks"]["bottom_n_momentum"]["lookback_days"]),
        ]

        for strat in strategies:
            print(f"  Running {strat.name} ...", end="", flush=True)
            res = _run_one(conn, cfg, policy_name, policy, strat, rebalance_dates,
                            predeclared_filters, starting_cash, universe_definition,
                            run_label=f"{strat.name}_{policy_name}", universe_cache=universe_cache)
            all_results[policy_name][strat.name] = res
            print(f" done in {res['elapsed_seconds']}s "
                  f"(cumulative_return={res['report']['cumulative_return']:.4f}, "
                  f"final_tradable_count(mean)={res['coverage_summary'].get('mean_final_tradable_count')})")

        # --- Random selection: headline seed (reported individually) + distribution ---
        print(f"  Running random_selection distribution ({seed_count} seeds) ...")
        random_runs = []
        failed_seeds = []
        t_dist_start = time.time()
        for i, seed in enumerate(distribution_seeds):
            strat = RandomSelection(portfolio_size=portfolio_size, seed=seed)
            label = f"random_selection_seed{seed}_{policy_name}"
            try:
                res = _run_one(conn, cfg, policy_name, policy, strat, rebalance_dates,
                                predeclared_filters, starting_cash, universe_definition, run_label=label,
                                universe_cache=universe_cache)
                random_runs.append(res)
            except Exception as e:
                # A single seed's failure must not abort a multi-hour run.
                # Record it verbatim (seed, label, exception type + message)
                # and continue -- this changes NOTHING about how successful
                # seeds are computed, it only prevents one bad seed from
                # losing all prior work. conn is NOT rolled back here since
                # save_run/save_positions for this seed either didn't run
                # or partially ran; if conn is left in a bad transaction
                # state after an exception we re-open it so subsequent
                # seeds aren't affected.
                import traceback
                err_text = f"{type(e).__name__}: {e}"
                print(f"    !! seed {seed} FAILED ({i + 1}/{len(distribution_seeds)}): {err_text}")
                failed_seeds.append({
                    "seed": seed, "index": i, "run_label": label,
                    "error": err_text, "traceback": traceback.format_exc(),
                })
                try:
                    conn.rollback()
                except Exception:
                    pass
            if (i + 1) % max(1, len(distribution_seeds) // 10) == 0:
                print(f"    ... {i + 1}/{len(distribution_seeds)} seeds attempted "
                      f"({len(random_runs)} succeeded, {len(failed_seeds)} failed) "
                      f"({time.time() - t_dist_start:.0f}s elapsed)")

        if failed_seeds:
            print(f"  WARNING: {len(failed_seeds)}/{len(distribution_seeds)} seeds failed for {policy_name} "
                  f"-- see failed_seeds in the JSON report for details.")

        cum_returns = [r["report"]["cumulative_return"] for r in random_runs]
        cagrs = [r["report"]["cagr"] for r in random_runs]
        sharpes = [r["report"]["sharpe_ratio"] for r in random_runs]
        dist_summary = {
            "seed_count": seed_count,
            "successful_seed_count": len(random_runs),
            "failed_seed_count": len(failed_seeds),
            "failed_seeds": failed_seeds,
            "headline_seed": headline_seed,
            "cumulative_return": {
                "mean": statistics.mean(cum_returns), "median": statistics.median(cum_returns),
                "stdev": statistics.stdev(cum_returns) if len(cum_returns) > 1 else 0.0,
                "min": min(cum_returns), "max": max(cum_returns),
            } if cum_returns else None,
            "cagr": {
                "mean": statistics.mean(cagrs), "median": statistics.median(cagrs),
                "stdev": statistics.stdev(cagrs) if len(cagrs) > 1 else 0.0,
            } if cagrs else None,
            "sharpe_ratio": {
                "mean": statistics.mean(sharpes), "median": statistics.median(sharpes),
                "stdev": statistics.stdev(sharpes) if len(sharpes) > 1 else 0.0,
            } if sharpes else None,
            "individual_run_ids": [r["run_id"] for r in random_runs],
        }
        all_results[policy_name]["random_selection_distribution"] = dist_summary
        print(f"  Random distribution done in {time.time() - t_dist_start:.0f}s "
              f"({len(random_runs)} succeeded, {len(failed_seeds)} failed)\n")

    conn.close()

    # --- Write combined report ---
    out_json = os.path.join(PROJECT_ROOT, "PHASE3_VALIDATION_REPORT.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    out_md = os.path.join(PROJECT_ROOT, "PHASE3_VALIDATION_REPORT.md")
    with open(out_md, "w") as f:
        f.write(_format_markdown(all_results, start_date, end_date, bt_cfg["rebalance_frequency"],
                                  seed_count, args.quick_random))

    print(f"\nDone. Wrote {out_json} and {out_md}.")


def _format_markdown(all_results, start_date, end_date, frequency, seed_count, quick_random):
    lines = []
    lines.append("# Phase 3 Validation Report\n")
    lines.append(f"Backtest period: {start_date} .. {end_date}, {frequency} rebalancing\n")
    lines.append(f"Random benchmark seeds: {seed_count} "
                  f"({'QUICK mode, not full spec-required 200' if quick_random else 'full spec-required 200'})\n")

    for policy_name in ["STRICT", "PERMISSIVE"]:
        lines.append(f"\n## Policy: {policy_name}\n")
        lines.append("| Strategy | Cumulative Return | CAGR | Sharpe | Max Drawdown | Trades | Mean Tradable Count |")
        lines.append("|---|---|---|---|---|---|---|")
        for strat_name, res in all_results[policy_name].items():
            if strat_name == "random_selection_distribution":
                continue
            r = res["report"]
            cs = res["coverage_summary"]
            lines.append(f"| {strat_name} | {r['cumulative_return']:.4f} | {r['cagr']:.4f} | "
                          f"{r['sharpe_ratio']:.3f} | {r['max_drawdown']:.4f} | {r['number_of_trades']} | "
                          f"{cs.get('mean_final_tradable_count')} |")

        dist = all_results[policy_name].get("random_selection_distribution")
        if dist:
            lines.append(f"\n**Random selection distribution** ({dist['seed_count']} seeds attempted, "
                          f"{dist.get('successful_seed_count', dist['seed_count'])} succeeded, "
                          f"{dist.get('failed_seed_count', 0)} failed, "
                          f"headline seed {dist['headline_seed']}):\n")
            if dist.get('failed_seed_count'):
                lines.append(f"- **{dist['failed_seed_count']} seed(s) failed** -- see `failed_seeds` in the JSON "
                              f"report for the seed, run label, and error for each.\n")
            if dist['cumulative_return']:
                lines.append(f"- Cumulative return: mean={dist['cumulative_return']['mean']:.4f}, "
                              f"median={dist['cumulative_return']['median']:.4f}, "
                              f"stdev={dist['cumulative_return']['stdev']:.4f}, "
                              f"min={dist['cumulative_return']['min']:.4f}, max={dist['cumulative_return']['max']:.4f}")
                lines.append(f"- CAGR: mean={dist['cagr']['mean']:.4f}, median={dist['cagr']['median']:.4f}, "
                              f"stdev={dist['cagr']['stdev']:.4f}")
                lines.append(f"- Sharpe: mean={dist['sharpe_ratio']['mean']:.3f}, "
                              f"median={dist['sharpe_ratio']['median']:.3f}, stdev={dist['sharpe_ratio']['stdev']:.3f}")
            else:
                lines.append("- No successful seeds -- no distribution statistics available.")

        lines.append("\n### Coverage detail\n")
        for strat_name, res in all_results[policy_name].items():
            if strat_name == "random_selection_distribution":
                continue
            cs = res["coverage_summary"]
            lines.append(f"- {strat_name}: {cs.get('num_rebalance_dates')} rebalance dates, "
                          f"tradable count range [{cs.get('min_final_tradable_count')}, "
                          f"{cs.get('max_final_tradable_count')}], "
                          f"{cs.get('dates_with_zero_eligible')} dates with zero eligible securities")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
