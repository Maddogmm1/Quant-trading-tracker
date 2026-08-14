"""
Phase 3: comprehensive post-hoc report over the FULL 200-seed random
benchmark run.

Run this LOCALLY, after `python3 run_phase3_validation.py` (no
--quick-random) has completed and persisted its results to
data/database/quant_trader_stage.db, and after you've copied
PHASE3_VALIDATION_REPORT.json into this folder (this script reads it for
the failed-seed bookkeeping the validation script records, plus per-run
elapsed_seconds for runtime totals -- neither of those lives in the
database).

Mostly read-only against the database: it queries backtest_runs/
backtest_results/backtest_coverage/backtest_positions exactly like
run_phase3_diagnostic.py did, and does the same one-sample-seed live
reproducibility spot-check (2 fresh runs, not 200 -- re-running all 200
seeds twice just to prove reproducibility would double the runtime for a
question the diagnostic already answered "yes" to at 20 seeds; this is a
cheap confirmation it still holds for the seed actually used here, not a
full re-verification).

It doesn't change the eligibility rules, RNG, selection mechanism, or
backtest methodology in response to anything it finds here, including the
price-history-length correlation, which is analyzed (splitting out
opportunity count vs. selection rate) but never "corrected."

It doesn't compare against a future ML strategy (doesn't exist yet) and
doesn't begin Phase 4 / predictive modelling. It stops after writing the
report.

Usage:
    python3 run_phase3_200seed_report.py
"""
import sys
import os
import json
import time
import statistics
import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import yaml

from src.database.db import init_db
from src.backtest.execution import next_rebalance_dates, next_market_session
from src.backtest.engine import run_backtest
from src.backtest.benchmarks import RandomSelection
from src.backtest.reproducibility import save_run, save_positions

DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "quant_trader_stage.db")
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "src", "database", "schema.sql")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
VALIDATION_JSON_PATH = os.path.join(PROJECT_ROOT, "PHASE3_VALIDATION_REPORT.json")

KNOWN_FLAGGED_TICKERS = {"TNB", "CFC", "RYC", "PZE", "PBG"}


def _now():
    return datetime.datetime.utcnow().isoformat() + "Z"


def _percentiles(values, pcts):
    s = sorted(values)
    n = len(s)
    out = {}
    for p in pcts:
        if n == 1:
            out[p] = s[0]
            continue
        k = (n - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, n - 1)
        out[p] = s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)
    return out


def _corr(xs, ys):
    if len(xs) < 2 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    return cov / (statistics.pstdev(xs) * statistics.pstdev(ys))


def _load_random_runs(conn, policy_name, run_ids):
    # Neither strategy_name='random_selection' nor a run_label pattern is a
    # safe filter on its own. Besides diagnostic/spot-check reruns polluting
    # the table, run_phase3_validation.py derives its N distribution seeds
    # from random.Random(headline_seed), so a 20-seed --quick-random run and
    # a later full 200-seed run share their first 20 seeds byte-for-byte
    # (same RNG, same sequence, just fewer/more draws). If both runs' rows
    # sit in the same database, matching by run_label would silently
    # double-count those 20 seeds. The only reliable way to scope to
    # exactly this run is the explicit run_id list the validation script
    # itself recorded in individual_run_ids in its JSON output -- that's
    # what run_ids must be here (passed in from main()).
    if not run_ids:
        return []
    placeholders = ",".join("?" * len(run_ids))
    rows = conn.execute(
        f"""SELECT run_id, random_seed, config_json
           FROM backtest_runs
           WHERE strategy_name='random_selection' AND data_quality_policy_name=?
             AND run_id IN ({placeholders})
           ORDER BY run_id ASC""",
        (policy_name, *run_ids),
    ).fetchall()
    found_ids = {r["run_id"] for r in rows}
    missing = [rid for rid in run_ids if rid not in found_ids]
    runs = []
    for r in rows:
        metrics = {
            m["metric_name"]: m["metric_value"]
            for m in conn.execute(
                "SELECT metric_name, metric_value FROM backtest_results WHERE run_id=? AND period IS NULL",
                (r["run_id"],),
            ).fetchall()
        }
        runs.append({"run_id": r["run_id"], "random_seed": r["random_seed"],
                      "config_json": r["config_json"], "metrics": metrics})
    if missing:
        print(f"  WARNING: {len(missing)} run_id(s) listed in the validation JSON's individual_run_ids "
              f"were not found in the database: {missing}")
    return runs


def _load_deterministic_run(conn, strategy_name, policy_name):
    row = conn.execute(
        """SELECT run_id, config_json FROM backtest_runs
           WHERE strategy_name=? AND data_quality_policy_name=? ORDER BY run_id DESC LIMIT 1""",
        (strategy_name, policy_name),
    ).fetchone()
    if not row:
        return None
    metrics = {
        m["metric_name"]: m["metric_value"]
        for m in conn.execute(
            "SELECT metric_name, metric_value FROM backtest_results WHERE run_id=? AND period IS NULL",
            (row["run_id"],),
        ).fetchall()
    }
    return {"run_id": row["run_id"], "config_json": row["config_json"], "metrics": metrics}


def _holdings_by_date(conn, run_id):
    rows = conn.execute(
        "SELECT as_of_date, security_id FROM backtest_positions WHERE run_id=? ORDER BY as_of_date",
        (run_id,),
    ).fetchall()
    by_date = {}
    for r in rows:
        by_date.setdefault(r["as_of_date"], set()).add(r["security_id"])
    return by_date


def _duplicate_holding_sets(conn, run_ids, sample_date):
    """Across ALL persisted runs for this policy (not a single sample
    date), how many DISTINCT full-portfolio holding sets exist on the
    date nearest sample_date. A count of 1 across many seeds under
    PERMISSIVE would indicate the RNG isn't actually varying the
    portfolio; under STRICT a count of 1 is a KNOWN, already-documented
    small-universe degeneracy (often only 1 eligible security exists),
    not a new finding."""
    sets_seen = {}
    for rid in run_ids:
        by_date = _holdings_by_date(conn, rid)
        if not by_date:
            continue
        actual_date = min(by_date.keys(), key=lambda d: abs(
            datetime.date.fromisoformat(d) - datetime.date.fromisoformat(sample_date)))
        key = frozenset(by_date[actual_date])
        sets_seen.setdefault(key, []).append(rid)
    return {
        "sample_date": sample_date,
        "num_runs_checked": len(run_ids),
        "num_distinct_holding_sets": len(sets_seen),
        "run_ids_per_duplicate_group": [v for v in sets_seen.values() if len(v) > 1],
    }


def _execution_timing_check(conn, rebalance_dates):
    """Calendar-only (never reads price values) -- confirms the
    execution-date fix (next_market_session) still produces no anomalous
    gaps over the full backtest period. This does not depend on seed at
    all (next_market_session takes no strategy/seed input), so it only
    needs to run once per policy run, not once per seed."""
    max_gap = 0
    gaps_over_6d = []
    for signal_date in rebalance_dates:
        execution_date = next_market_session(conn, signal_date, adj_type="total_return")
        if not execution_date:
            continue
        gap = (datetime.date.fromisoformat(execution_date) - datetime.date.fromisoformat(signal_date)).days
        if gap > 6:
            gaps_over_6d.append({"signal_date": signal_date, "execution_date": execution_date, "gap_days": gap})
        max_gap = max(max_gap, gap)
    return {"dates_checked": len(rebalance_dates), "anomalous_gap_count": len(gaps_over_6d),
            "anomalous_gaps": gaps_over_6d, "max_gap_days": max_gap}


def _fairness_check(conn, policy_name, random_runs):
    """Checks ALL persisted random runs (not a 5-run sample -- cheap now
    since it's pure SQL + set arithmetic against already-persisted
    positions, no backtest re-execution)."""
    ew = _load_deterministic_run(conn, "equal_weight_sp500", policy_name)
    if not ew:
        return {"checked": False, "reason": "no equal_weight_sp500 run found"}
    ew_holdings = _holdings_by_date(conn, ew["run_id"])
    violations = []
    dates_checked = 0
    for rr in random_runs:
        rand_holdings = _holdings_by_date(conn, rr["run_id"])
        for date, secs in rand_holdings.items():
            dates_checked += 1
            not_subset = secs - ew_holdings.get(date, set())
            if not_subset:
                violations.append({"run_id": rr["run_id"], "date": date, "extra_securities": sorted(not_subset)})
    return {"checked": True, "equal_weight_run_id": ew["run_id"], "dates_checked": dates_checked,
            "violations": violations, "fair": len(violations) == 0}


def _randomness_sanity(conn, run_ids):
    selection_counts = {}
    for rid in run_ids:
        by_date = _holdings_by_date(conn, rid)
        for secs in by_date.values():
            for sec_id in secs:
                selection_counts[sec_id] = selection_counts.get(sec_id, 0) + 1
    if not selection_counts:
        return {"error": "no positions found for these run_ids"}

    sec_ids = list(selection_counts.keys())
    rows = conn.execute(
        f"SELECT security_id, primary_ticker FROM securities WHERE security_id IN "
        f"({','.join('?' * len(sec_ids))})", sec_ids,
    ).fetchall()
    ticker_by_id = {r["security_id"]: r["primary_ticker"] for r in rows}

    price_counts = {}
    for sec_id in sec_ids:
        c = conn.execute(
            "SELECT COUNT(*) c FROM prices WHERE security_id=? AND adj_type='total_return'", (sec_id,)
        ).fetchone()["c"]
        price_counts[sec_id] = c

    counts = list(selection_counts.values())
    ids_sorted_by_id = sorted(sec_ids)
    ids_sorted_alpha = sorted(sec_ids, key=lambda s: ticker_by_id.get(s, ""))
    id_rank = {sid: i for i, sid in enumerate(ids_sorted_by_id)}
    alpha_rank = {sid: i for i, sid in enumerate(ids_sorted_alpha)}
    freq_list = [selection_counts[s] for s in sec_ids]

    return {
        "num_distinct_securities_ever_selected": len(sec_ids),
        "selection_count_mean": statistics.mean(counts),
        "selection_count_stdev": statistics.stdev(counts) if len(counts) > 1 else 0.0,
        "selection_count_min": min(counts),
        "selection_count_max": max(counts),
        "correlation_selection_freq_vs_security_id_order": round(_corr([id_rank[s] for s in sec_ids], freq_list), 4),
        "correlation_selection_freq_vs_alphabetical_ticker_order": round(
            _corr([alpha_rank[s] for s in sec_ids], freq_list), 4),
        "correlation_selection_freq_vs_price_history_length": round(
            _corr([price_counts[s] for s in sec_ids], freq_list), 4),
        "sector_bias_checkable": False,
        "sector_bias_note": "securities.sector is unpopulated in this database (Phase 1/2 never sourced sector "
                             "data) -- documented limitation, not attempted here.",
    }


def _opportunity_vs_selection_rate_analysis(conn, policy_name, run_ids):
    """Splits the raw selection-count correlation seen in the diagnostic
    into two distinct causes:
    (A) a security being selected more often PER OPPORTUNITY (a genuine
        per-draw preference/bias in the RNG or selection code), vs
    (B) a security simply having more OPPORTUNITIES because it stays
        eligible for more rebalance dates (a downstream consequence of
        universe eligibility, not a selection bias).

    'Opportunity' on a given date = eligible that date. equal_weight_sp500
    holds the FULL eligible universe on every date (already established
    by build_eligible_universe's single-source-of-truth design and
    confirmed by the fairness check), so its holdings-by-date IS the
    eligible set for that policy -- no need to recompute
    build_eligible_universe here.

    Per security: opportunity_count = (# dates eligible) * (# seeds),
    selection_count = (# times actually held across all seeds x dates),
    selection_rate = selection_count / opportunity_count.

    We correlate SELECTION RATE (not raw selection count) against
    price-history length. Do not infer (A) exists unless this
    rate-based correlation itself is materially non-zero.
    """
    ew = _load_deterministic_run(conn, "equal_weight_sp500", policy_name)
    if not ew:
        return {"checked": False, "reason": "no equal_weight_sp500 run found"}
    ew_holdings = _holdings_by_date(conn, ew["run_id"])  # date -> eligible set (proxy)

    opportunity_count = {}
    for secs in ew_holdings.values():
        for sec_id in secs:
            opportunity_count[sec_id] = opportunity_count.get(sec_id, 0) + len(run_ids)

    selection_count = {}
    for rid in run_ids:
        by_date = _holdings_by_date(conn, rid)
        for secs in by_date.values():
            for sec_id in secs:
                selection_count[sec_id] = selection_count.get(sec_id, 0) + 1

    sec_ids = [s for s in opportunity_count.keys() if opportunity_count[s] > 0]
    if not sec_ids:
        return {"checked": False, "reason": "no eligible securities found"}

    selection_rate = {s: selection_count.get(s, 0) / opportunity_count[s] for s in sec_ids}

    price_counts = {}
    for sec_id in sec_ids:
        c = conn.execute(
            "SELECT COUNT(*) c FROM prices WHERE security_id=? AND adj_type='total_return'", (sec_id,)
        ).fetchone()["c"]
        price_counts[sec_id] = c

    rate_list = [selection_rate[s] for s in sec_ids]
    price_list = [price_counts[s] for s in sec_ids]
    opp_list = [opportunity_count[s] for s in sec_ids]

    corr_rate_vs_length = round(_corr(price_list, rate_list), 4)
    corr_opportunity_vs_length = round(_corr(price_list, opp_list), 4)

    return {
        "checked": True,
        "num_securities_analyzed": len(sec_ids),
        "correlation_selection_RATE_vs_price_history_length": corr_rate_vs_length,
        "correlation_opportunity_count_vs_price_history_length": corr_opportunity_vs_length,
        "interpretation": (
            "If correlation_opportunity_count_vs_price_history_length is large but "
            "correlation_selection_RATE_vs_price_history_length is near zero, the raw "
            "selection-count correlation seen in the diagnostic is explained by (B) -- "
            "longer-history securities simply stay eligible for more dates -- not by any "
            "per-opportunity selection preference (A). A materially non-zero RATE "
            "correlation would indicate (A) actually exists and would warrant "
            "investigation (not a methodology change during this run)."
        ),
        "selection_rate_summary": {
            "mean": round(statistics.mean(rate_list), 4),
            "median": round(statistics.median(rate_list), 4),
            "min": round(min(rate_list), 4),
            "max": round(max(rate_list), 4),
        },
    }


def _reproducibility_spot_check(conn, cfg, policy_name, policy, predeclared_filters, rebalance_dates, seed):
    results = []
    for i in range(2):
        report, coverage, portfolio = run_backtest(
            conn, RandomSelection(portfolio_size=cfg["backtest"]["portfolio"]["default_size"], seed=seed),
            rebalance_dates, policy, cost_config=cfg["backtest"]["costs"],
            predeclared_filters=predeclared_filters,
            lookback_days=cfg["backtest"]["execution"]["lookback_days_required"],
            universe_definition="SP500",
        )
        run_id = save_run(
            conn, run_label=f"repro_spotcheck_{policy_name}_{seed}_{i}", config=cfg, random_seed=seed,
            start_date=rebalance_dates[0], end_date=rebalance_dates[-1],
            rebalance_frequency=cfg["backtest"]["rebalance_frequency"], universe_definition="SP500",
            data_quality_policy_name=policy_name, strategy_name="random_selection",
            cost_config=cfg["backtest"]["costs"], execution_config=cfg["backtest"]["execution"],
            report=report, coverage_report=coverage, created_at=_now(),
        )
        save_positions(conn, run_id, portfolio.history)
        results.append(report)
    identical = results[0] == results[1]
    return {"seed": seed, "identical": identical, "report_a": results[0], "report_b": results[1]}


def _transaction_cost_sanity(det_runs):
    rows = [{"strategy": name, "avg_turnover_pct": r["metrics"].get("avg_turnover_pct"),
             "total_transaction_costs": r["metrics"].get("total_transaction_costs")}
            for name, r in det_runs.items() if r]
    rows.sort(key=lambda x: x["avg_turnover_pct"] or 0)
    monotonic = all(rows[i]["total_transaction_costs"] <= rows[i + 1]["total_transaction_costs"]
                     for i in range(len(rows) - 1))
    return {"ordered_by_turnover": rows, "cost_increases_monotonically_with_turnover": monotonic}


def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: database not found at {DB_PATH}")
        sys.exit(1)
    if not os.path.exists(VALIDATION_JSON_PATH):
        print(f"ERROR: {VALIDATION_JSON_PATH} not found -- run run_phase3_validation.py (full, no "
              f"--quick-random) first and keep its JSON output in this folder.")
        sys.exit(1)

    cfg = yaml.safe_load(open(CONFIG_PATH))
    bt_cfg = cfg["backtest"]
    predeclared_filters = cfg.get("universe_filters", {})
    start_date = cfg["data"]["price_start_date"]
    end_date = cfg["data"]["price_end_date"]
    rebalance_dates = next_rebalance_dates(start_date, end_date, bt_cfg["rebalance_frequency"])

    with open(VALIDATION_JSON_PATH) as f:
        validation_json = json.load(f)

    conn = init_db(DB_PATH, SCHEMA_PATH, reset=False, force=False)
    report = {"generated_at": _now(), "source_validation_report": VALIDATION_JSON_PATH, "policies": {}}

    for policy_name in ["STRICT", "PERMISSIVE"]:
        print(f"\n=== Reporting policy: {policy_name} ===")
        policy = bt_cfg["data_quality_policies"][policy_name]

        # --- individual_run_ids from THIS validation run's own JSON output is
        # the only reliable way to scope to exactly this run (see _load_random_runs
        # docstring-comment for why run_label/strategy_name alone are not safe). ---
        val_dist = validation_json.get(policy_name, {}).get("random_selection_distribution", {})
        expected_run_ids = val_dist.get("individual_run_ids", [])
        if not expected_run_ids:
            print(f"  No individual_run_ids found in the validation JSON for {policy_name} -- run the full "
                  f"run_phase3_validation.py first.")
            continue

        random_runs = _load_random_runs(conn, policy_name, expected_run_ids)
        if not random_runs:
            print(f"  None of the {len(expected_run_ids)} expected run_ids for {policy_name} were found in "
                  f"the database -- is this the same database the validation run wrote to?")
            continue
        run_ids = [r["run_id"] for r in random_runs]
        print(f"  {len(run_ids)} persisted random runs found for {policy_name} "
              f"(of {len(expected_run_ids)} expected from the validation JSON)")

        # --- successful/failed seed counts, straight from the validation JSON ---
        seed_bookkeeping = {
            "seeds_attempted": val_dist.get("seed_count"),
            "successful_seed_count": val_dist.get("successful_seed_count", len(expected_run_ids)),
            "failed_seed_count": val_dist.get("failed_seed_count", 0),
            "failed_seeds": val_dist.get("failed_seeds", []),
            "persisted_run_count_in_db": len(random_runs),
            "expected_run_id_count_from_validation_json": len(expected_run_ids),
        }
        if len(random_runs) != len(expected_run_ids):
            seed_bookkeeping["note"] = ("MISMATCH: not every run_id the validation JSON says it persisted was "
                                         "found in the database for this policy -- investigate before trusting "
                                         "the distribution stats below (see missing run_id warning above, if any).")
            print(f"  WARNING: {seed_bookkeeping['note']}")

        # --- required distribution statistics ---
        cum_returns = [r["metrics"]["cumulative_return"] for r in random_runs]
        cagrs = [r["metrics"]["cagr"] for r in random_runs]
        sharpes = [r["metrics"]["sharpe_ratio"] for r in random_runs]
        max_dds = [r["metrics"]["max_drawdown"] for r in random_runs]
        costs = [r["metrics"]["total_transaction_costs"] for r in random_runs]
        turnovers = [r["metrics"]["avg_turnover_pct"] for r in random_runs]
        trade_counts = [r["metrics"]["number_of_trades"] for r in random_runs]

        stats_block = {
            "seed_count": len(random_runs),
            "cumulative_return": {
                "mean": statistics.mean(cum_returns), "median": statistics.median(cum_returns),
                "stdev": statistics.stdev(cum_returns) if len(cum_returns) > 1 else 0.0,
                "min": min(cum_returns), "max": max(cum_returns),
                "percentiles": _percentiles(cum_returns, [5, 25, 75, 95]),
            },
            "cagr": {"mean": statistics.mean(cagrs), "median": statistics.median(cagrs)},
            "sharpe_ratio": {"mean": statistics.mean(sharpes), "median": statistics.median(sharpes)},
            "max_drawdown": {"mean": statistics.mean(max_dds), "median": statistics.median(max_dds)},
            "total_transaction_costs": {"mean": statistics.mean(costs), "median": statistics.median(costs)},
            "avg_turnover_pct_mean": statistics.mean(turnovers),
            "number_of_trades_mean": statistics.mean(trade_counts),
        }

        det_runs = {name: _load_deterministic_run(conn, name, policy_name)
                    for name in ["buy_and_hold", "equal_weight_sp500", "top_n_momentum", "bottom_n_momentum"]}
        det_comparison = {name: (r["metrics"] if r else None) for name, r in det_runs.items()}

        sample_date = rebalance_dates[len(rebalance_dates) // 2]
        duplicate_holdings = _duplicate_holding_sets(conn, run_ids, sample_date)

        print("  Running execution-timing check (calendar-only, whole period)...")
        execution_timing = _execution_timing_check(conn, rebalance_dates)

        print(f"  Running fairness check across all {len(run_ids)} runs...")
        fairness = _fairness_check(conn, policy_name, random_runs)

        cost_sanity = _transaction_cost_sanity(det_runs)

        print("  Running randomness-sanity (RNG/order bias) diagnostics...")
        randomness = _randomness_sanity(conn, run_ids)

        print("  Running opportunity-vs-selection-rate analysis (price-history-length A-vs-B split)...")
        opportunity_analysis = _opportunity_vs_selection_rate_analysis(conn, policy_name, run_ids)

        min_run = min(random_runs, key=lambda r: r["metrics"]["cumulative_return"])
        max_run = max(random_runs, key=lambda r: r["metrics"]["cumulative_return"])

        sample_seed = random_runs[0]["random_seed"]
        print(f"  Running reproducibility spot-check (seed={sample_seed}, 2 fresh runs)...")
        t0 = time.time()
        repro = _reproducibility_spot_check(conn, cfg, policy_name, policy, predeclared_filters,
                                             rebalance_dates, sample_seed)
        repro_elapsed = time.time() - t0
        print(f"    done in {repro_elapsed:.1f}s -- identical: {repro['identical']}")

        # --- runtime, from the validation script's own elapsed_seconds ---
        det_elapsed = [r["elapsed_seconds"] for name, r in validation_json.get(policy_name, {}).items()
                       if name != "random_selection_distribution" and isinstance(r, dict) and "elapsed_seconds" in r]
        # NOTE: per-seed elapsed_seconds aren't in the validation JSON today (only aggregate distribution
        # timing is printed to stdout, not persisted) -- report what's available and flag the gap rather
        # than fabricating a number.
        runtime_block = {
            "deterministic_strategy_elapsed_seconds": det_elapsed,
            "note": ("Per-seed elapsed_seconds for the 200-run random distribution is not persisted in "
                     "PHASE3_VALIDATION_REPORT.json (only console output shows periodic progress) -- "
                     "recording per-seed timing would need a small follow-up change to "
                     "run_phase3_validation.py, out of scope here."),
        }

        # --- cache hit/miss statistics (analytical estimate, not instrumented) ---
        cache_stats = {
            "note": "Computed analytically, not instrumented in engine.py (would require a code change "
                     "out of scope for this run).",
            "total_universe_lookups_estimate": 5 * len(rebalance_dates) + len(run_ids) * len(rebalance_dates),
            "unique_cache_entries_estimate": len(rebalance_dates),
            "explanation": ("build_eligible_universe(date, policy) is a pure function of (date, policy) at a "
                             "fixed policy, so the cache has exactly one entry per rebalance date; every "
                             "strategy/seed after the first to reach a given date is a cache hit."),
        }

        report["policies"][policy_name] = {
            "seed_bookkeeping": seed_bookkeeping,
            "distribution_statistics": stats_block,
            "deterministic_benchmark_comparison": det_comparison,
            "duplicate_holding_sets": duplicate_holdings,
            "execution_timing": execution_timing,
            "strategy_comparison_fairness": fairness,
            "transaction_cost_sanity": cost_sanity,
            "randomness_sanity": randomness,
            "price_history_length_opportunity_vs_selection_rate_analysis": opportunity_analysis,
            "performance_distribution_outliers": {
                "min_return_run_id": min_run["run_id"], "min_return": min_run["metrics"]["cumulative_return"],
                "max_return_run_id": max_run["run_id"], "max_return": max_run["metrics"]["cumulative_return"],
            },
            "reproducibility_spot_check": repro,
            "runtime": runtime_block,
            "cache_hit_miss_statistics": cache_stats,
        }

    conn.close()

    out_json = os.path.join(PROJECT_ROOT, "PHASE3_200SEED_REPORT.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nDone. Wrote {out_json}.")
    print("\nThis script does not begin Phase 4 / ML work -- Phase 3 stops here.")


if __name__ == "__main__":
    main()
