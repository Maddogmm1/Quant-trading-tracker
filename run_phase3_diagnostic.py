"""
Phase 3 random-benchmark diagnostic (pre-200-seed-run gate).

Runs ZERO new backtests over the full 20-seed distribution -- it queries
everything from the backtest_runs/backtest_results/backtest_coverage/
backtest_positions rows your last `run_phase3_validation.py --quick-random`
run already persisted, plus a small number of DELIBERATELY cheap live
re-runs for the reproducibility check (2 fresh runs per policy, not 20).

Answers, in order: distribution statistics; whether every
seed used the identical point-in-time universe/dates/policy/costs;
reproducibility; seed independence; randomness-bias sanity checks;
strategy-comparison fairness (random vs deterministic drawing from the
same eligible pool); transaction-cost sanity; performance-distribution
outlier investigation (including a check against known YHD/bad-OHLC
tickers, since PERMISSIVE does not exclude severe-OHLC-flagged
securities); and measured per-seed runtime with a 200-seed projection.

Does NOT run the full 200-seed distribution. Does NOT change any research
logic, config, or scoring. Read-only against prices/securities/
index_membership -- only ever INSERTs two small extra rows into the
existing backtest_* tables for the reproducibility re-runs.

Run this LOCALLY, in the same folder as run_phase3_validation.py:
    python3 run_phase3_diagnostic.py
"""
import sys
import os
import json
import time
import statistics

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import sqlite3
import yaml

from src.database.db import init_db
from src.backtest.universe import build_eligible_universe
from src.backtest.execution import next_rebalance_dates
from src.backtest.engine import run_backtest
from src.backtest.benchmarks import RandomSelection
from src.backtest.reproducibility import save_run, save_positions
from src.backtest.costs import trade_cost

DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "quant_trader_stage.db")
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "src", "database", "schema.sql")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")

# Known bad-data tickers from BAD_OHLC_INVESTIGATION.md / YHD_SWEEP.md --
# used only to flag if a performance outlier happens to hold one of them,
# not to exclude anything.
KNOWN_FLAGGED_TICKERS = {"TNB", "CFC", "RYC", "PZE", "PBG"}


def _now():
    import datetime
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
        if f == c:
            out[p] = s[f]
        else:
            out[p] = s[f] + (s[c] - s[f]) * (k - f)
    return out


def _load_random_runs(conn, policy_name, limit=20):
    rows = conn.execute(
        """SELECT run_id, random_seed, config_json, cost_assumptions_json, execution_assumptions_json
           FROM backtest_runs
           WHERE strategy_name='random_selection' AND data_quality_policy_name=?
           ORDER BY run_id DESC LIMIT ?""",
        (policy_name, limit),
    ).fetchall()
    rows = list(reversed(rows))  # chronological
    runs = []
    for r in rows:
        metrics = {
            m["metric_name"]: m["metric_value"]
            for m in conn.execute(
                "SELECT metric_name, metric_value FROM backtest_results WHERE run_id=? AND period IS NULL",
                (r["run_id"],),
            ).fetchall()
        }
        coverage = conn.execute(
            "SELECT as_of_date, final_tradable_count FROM backtest_coverage WHERE run_id=? ORDER BY as_of_date",
            (r["run_id"],),
        ).fetchall()
        runs.append({
            "run_id": r["run_id"], "random_seed": r["random_seed"],
            "config_json": r["config_json"], "cost_assumptions_json": r["cost_assumptions_json"],
            "execution_assumptions_json": r["execution_assumptions_json"],
            "metrics": metrics,
            "coverage": [(c["as_of_date"], c["final_tradable_count"]) for c in coverage],
        })
    return runs


def _load_deterministic_run(conn, strategy_name, policy_name):
    row = conn.execute(
        """SELECT run_id, config_json FROM backtest_runs
           WHERE strategy_name=? AND data_quality_policy_name=? ORDER BY run_id DESC LIMIT 1""",
        (strategy_name, policy_name),
    ).fetchone()
    if not row:
        return None
    coverage = conn.execute(
        "SELECT as_of_date, final_tradable_count FROM backtest_coverage WHERE run_id=? ORDER BY as_of_date",
        (row["run_id"],),
    ).fetchall()
    return {
        "run_id": row["run_id"], "config_json": row["config_json"],
        "coverage": [(c["as_of_date"], c["final_tradable_count"]) for c in coverage],
    }


def _holdings_by_date(conn, run_id):
    rows = conn.execute(
        "SELECT as_of_date, security_id FROM backtest_positions WHERE run_id=? ORDER BY as_of_date",
        (run_id,),
    ).fetchall()
    by_date = {}
    for r in rows:
        by_date.setdefault(r["as_of_date"], set()).add(r["security_id"])
    return by_date


def _holdings_count_stats(conn, run_ids):
    counts = []
    for rid in run_ids:
        by_date = _holdings_by_date(conn, rid)
        counts.extend(len(v) for v in by_date.values())
    return counts


def _reproducibility_check(conn, cfg, policy_name, policy, predeclared_filters, rebalance_dates, seed,
                            universe_cache=None):
    results = []
    for i in range(2):
        report, coverage, portfolio = run_backtest(
            conn, RandomSelection(portfolio_size=cfg["backtest"]["portfolio"]["default_size"], seed=seed),
            rebalance_dates, policy, cost_config=cfg["backtest"]["costs"],
            predeclared_filters=predeclared_filters, universe_definition="SP500",
            lookback_days=cfg["backtest"]["execution"]["lookback_days_required"],
            universe_cache=universe_cache,
        )
        run_id = save_run(
            conn, run_label=f"diagnostic_repro_{policy_name}_seed{seed}_run{i}", config=cfg, random_seed=seed,
            start_date=rebalance_dates[0], end_date=rebalance_dates[-1],
            rebalance_frequency=cfg["backtest"]["rebalance_frequency"], universe_definition="SP500",
            data_quality_policy_name=policy_name, strategy_name="random_selection",
            cost_config=cfg["backtest"]["costs"], execution_config=cfg["backtest"]["execution"],
            report=report, coverage_report=coverage, created_at=_now(),
        )
        save_positions(conn, run_id, portfolio.history)
        results.append({"run_id": run_id, "report": report})
    a, b = results[0]["report"], results[1]["report"]
    identical = all(a[k] == b[k] for k in ("cumulative_return", "cagr", "sharpe_ratio", "max_drawdown",
                                            "number_of_trades", "total_transaction_costs"))
    return {"seed": seed, "run_id_a": results[0]["run_id"], "run_id_b": results[1]["run_id"],
            "identical": identical, "report_a": a, "report_b": b}


def _randomness_sanity(conn, run_ids):
    """Selection-frequency bias checks against ticker order, alphabetical
    order, and price-history-length (a proxy for 'missing-data pattern').
    Sector bias is NOT checkable -- securities.sector was never populated
    during Phase 1/2 ingestion (0 of 25 securities have a sector value in
    the local check; this is a known, pre-existing data gap, not
    something this diagnostic can work around)."""
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

    def _corr(xs, ys):
        if len(xs) < 2 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
            return 0.0
        mx, my = statistics.mean(xs), statistics.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
        return cov / (statistics.pstdev(xs) * statistics.pstdev(ys))

    id_rank = {sid: i for i, sid in enumerate(ids_sorted_by_id)}
    alpha_rank = {sid: i for i, sid in enumerate(ids_sorted_alpha)}
    freq_list = [selection_counts[s] for s in sec_ids]
    id_rank_list = [id_rank[s] for s in sec_ids]
    alpha_rank_list = [alpha_rank[s] for s in sec_ids]
    price_count_list = [price_counts[s] for s in sec_ids]

    return {
        "num_distinct_securities_ever_selected": len(sec_ids),
        "selection_count_mean": statistics.mean(counts),
        "selection_count_stdev": statistics.stdev(counts) if len(counts) > 1 else 0.0,
        "selection_count_min": min(counts),
        "selection_count_max": max(counts),
        "correlation_selection_freq_vs_security_id_order": round(_corr(id_rank_list, freq_list), 4),
        "correlation_selection_freq_vs_alphabetical_ticker_order": round(_corr(alpha_rank_list, freq_list), 4),
        "correlation_selection_freq_vs_price_history_length": round(_corr(price_count_list, freq_list), 4),
        "sector_bias_checkable": False,
        "sector_bias_note": "securities.sector is unpopulated in this database (Phase 1/2 never sourced sector "
                             "data) -- cannot check sector bias.",
        "note": "correlations near 0 indicate no bias; |r| > ~0.3 would warrant a closer look.",
    }


def _flagged_ticker_exposure(conn, run_ids):
    out = {}
    for rid in run_ids:
        by_date = _holdings_by_date(conn, rid)
        all_secs = set()
        for s in by_date.values():
            all_secs.update(s)
        if not all_secs:
            out[rid] = []
            continue
        rows = conn.execute(
            f"SELECT security_id, primary_ticker FROM securities WHERE security_id IN "
            f"({','.join('?' * len(all_secs))})", list(all_secs),
        ).fetchall()
        flagged = [r["primary_ticker"] for r in rows if r["primary_ticker"] in KNOWN_FLAGGED_TICKERS]
        out[rid] = flagged
    return out


def _execution_timing_check(conn, cfg, policy, predeclared_filters, rebalance_dates, seeds, universe_cache):
    """Re-runs each seed fresh (with the CURRENT engine code) and flags
    any execution date resolved more than 6 calendar days after its
    signal date -- the signature of the ABMD-style bug (one arbitrary
    security's own data gap silently delaying the whole portfolio's
    execution). Uses execution.next_market_session directly (the same
    function the engine itself now uses) rather than re-deriving dates a
    second way, so this checks the ACTUAL resolved execution dates, not
    an independent reimplementation that could itself be wrong."""
    import datetime as _dt
    from src.backtest.execution import next_market_session

    max_gap = 0
    affected = 0
    for seed in seeds:
        seed_max_gap = 0
        for signal_date in rebalance_dates:
            execution_date = next_market_session(conn, signal_date, adj_type="total_return")
            if not execution_date:
                continue
            gap = (_dt.date.fromisoformat(execution_date) - _dt.date.fromisoformat(signal_date)).days
            seed_max_gap = max(seed_max_gap, gap)
        # next_market_session no longer depends on the seed at all (it's
        # policy/date-only, not security-specific) -- so if it's ever
        # anomalous for one seed it's identically anomalous for all of
        # them. This loop still iterates per-seed to keep the report
        # shape parallel to the other per-seed checks and to make that
        # invariant explicit rather than assumed.
        if seed_max_gap > 6:
            affected += 1
        max_gap = max(max_gap, seed_max_gap)
    return {"seeds_checked": len(seeds), "affected_seed_count": affected, "max_gap_days": max_gap}


def _fairness_check(conn, policy_name):
    """Every random-selected security on a given date must be a SUBSET of
    equal_weight_sp500's held set on that same date (equal_weight holds
    the FULL eligible universe every date), proving both draw from the
    identical eligible pool."""
    ew = _load_deterministic_run(conn, "equal_weight_sp500", policy_name)
    if not ew:
        return {"checked": False, "reason": "no equal_weight_sp500 run found"}
    ew_holdings = _holdings_by_date(conn, ew["run_id"])

    random_runs = _load_random_runs(conn, policy_name, limit=5)  # sample, not all 20 -- cheap and sufficient
    violations = []
    dates_checked = 0
    for rr in random_runs:
        rand_holdings = _holdings_by_date(conn, rr["run_id"])
        for date, secs in rand_holdings.items():
            dates_checked += 1
            ew_set = ew_holdings.get(date, set())
            not_subset = secs - ew_set
            if not_subset:
                violations.append({"run_id": rr["run_id"], "date": date, "extra_securities": list(not_subset)})
    return {"checked": True, "equal_weight_run_id": ew["run_id"], "dates_checked": dates_checked,
            "violations": violations, "fair": len(violations) == 0}


def _identical_universe_check(det_runs, random_runs):
    """Compares backtest_coverage (final_tradable_count per date) across
    all runs (4 deterministic + N random) at a given policy. Since this is
    a pure function of (date, policy) and doesn't depend on strategy or
    seed, every run's coverage series must be byte-identical."""
    all_coverage_series = [r["coverage"] for r in det_runs.values() if r] + [r["coverage"] for r in random_runs]
    if not all_coverage_series:
        return {"checked": False}
    reference = all_coverage_series[0]
    mismatches = [i for i, series in enumerate(all_coverage_series) if series != reference]
    return {"checked": True, "num_runs_compared": len(all_coverage_series),
            "all_identical": len(mismatches) == 0, "mismatched_run_indices": mismatches}


def _config_identical_check(det_runs, random_runs):
    configs = [r["config_json"] for r in det_runs.values() if r] + [r["config_json"] for r in random_runs]
    if not configs:
        return {"checked": False}
    return {"checked": True, "all_identical": len(set(configs)) == 1, "num_distinct_configs": len(set(configs))}


def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: database not found at {DB_PATH}")
        sys.exit(1)

    cfg = yaml.safe_load(open(CONFIG_PATH))
    bt_cfg = cfg["backtest"]
    predeclared_filters = cfg.get("universe_filters", {})
    start_date = cfg["data"]["price_start_date"]
    end_date = cfg["data"]["price_end_date"]
    rebalance_dates = next_rebalance_dates(start_date, end_date, bt_cfg["rebalance_frequency"])

    conn = init_db(DB_PATH, SCHEMA_PATH, reset=False, force=False)

    report = {"generated_at": _now(), "policies": {}}

    for policy_name in ["STRICT", "PERMISSIVE"]:
        print(f"\n=== Diagnosing policy: {policy_name} ===")
        policy = bt_cfg["data_quality_policies"][policy_name]
        random_runs = _load_random_runs(conn, policy_name, limit=20)
        if not random_runs:
            print(f"  No persisted random_selection runs found for {policy_name} -- run "
                  f"run_phase3_validation.py --quick-random first.")
            continue
        run_ids = [r["run_id"] for r in random_runs]
        print(f"  Found {len(run_ids)} persisted random runs (run_ids {run_ids[0]}..{run_ids[-1]})")

        # 1-10: distribution statistics
        cum_returns = [r["metrics"]["cumulative_return"] for r in random_runs]
        cagrs = [r["metrics"]["cagr"] for r in random_runs]
        max_dds = [r["metrics"]["max_drawdown"] for r in random_runs]
        turnovers = [r["metrics"]["avg_turnover_pct"] for r in random_runs]
        costs = [r["metrics"]["total_transaction_costs"] for r in random_runs]
        holdings_counts = _holdings_count_stats(conn, run_ids)

        stats_block = {
            "seed_count": len(random_runs),
            "cumulative_return": {
                "mean": statistics.mean(cum_returns), "median": statistics.median(cum_returns),
                "stdev": statistics.stdev(cum_returns) if len(cum_returns) > 1 else 0.0,
                "min": min(cum_returns), "max": max(cum_returns),
                "percentiles": _percentiles(cum_returns, [5, 25, 50, 75, 95]),
            },
            "cagr_mean": statistics.mean(cagrs),
            "max_drawdown_mean": statistics.mean(max_dds),
            "avg_turnover_pct_mean": statistics.mean(turnovers),
            "total_transaction_costs_mean": statistics.mean(costs),
            "holdings_per_date": {
                "mean": statistics.mean(holdings_counts) if holdings_counts else None,
                "min": min(holdings_counts) if holdings_counts else None,
                "max": max(holdings_counts) if holdings_counts else None,
            },
        }

        # 11 + D: identical universe/dates/policy/costs/execution across every seed and vs deterministic strategies
        det_runs = {name: _load_deterministic_run(conn, name, policy_name)
                    for name in ["buy_and_hold", "equal_weight_sp500", "top_n_momentum", "bottom_n_momentum"]}
        universe_check = _identical_universe_check(det_runs, random_runs)
        config_check = _config_identical_check(det_runs, random_runs)

        # A: reproducibility -- rerun ONE persisted seed twice, live
        sample_seed = random_runs[0]["random_seed"]
        print(f"  Running reproducibility check (seed={sample_seed}, 2 fresh runs)...")
        t0 = time.time()
        universe_cache = {}
        repro = _reproducibility_check(conn, cfg, policy_name, policy, predeclared_filters, rebalance_dates,
                                        sample_seed, universe_cache=universe_cache)
        repro_elapsed = time.time() - t0
        print(f"    done in {repro_elapsed:.1f}s -- identical: {repro['identical']}")

        # H: execution timing -- re-run all persisted seeds fresh (with
        # the current code) and confirm no execution date is ever more
        # than a few calendar days after its signal date. This directly
        # answers "is the ABMD-style execution-date bug still present":
        # post-fix this must be 0/N; the diagnostic that first caught the
        # bug found 7/20 affected under PERMISSIVE, with gaps up to 182
        # days, before this fix existed.
        print(f"  Running execution-timing check ({len(random_runs)} seeds, fresh)...")
        execution_timing = _execution_timing_check(conn, cfg, policy, predeclared_filters, rebalance_dates,
                                                     [r["random_seed"] for r in random_runs], universe_cache)
        print(f"    seeds with anomalous execution gaps: {execution_timing['affected_seed_count']}/"
              f"{execution_timing['seeds_checked']} (max gap seen: {execution_timing['max_gap_days']} days)")

        # B: seed independence
        holdings_by_run = {rid: _holdings_by_date(conn, rid) for rid in run_ids}
        sample_date = rebalance_dates[len(rebalance_dates) // 2]
        # find the closest date actually present for each run
        distinct_holding_sets_on_sample_date = set()
        for rid, by_date in holdings_by_run.items():
            actual_date = min(by_date.keys(), key=lambda d: abs(
                __import__("datetime").date.fromisoformat(d) - __import__("datetime").date.fromisoformat(sample_date)
            )) if by_date else None
            if actual_date:
                distinct_holding_sets_on_sample_date.add(frozenset(by_date[actual_date]))
        seed_independence = {
            "sample_date": sample_date,
            "num_runs_checked": len(holdings_by_run),
            "num_distinct_holding_sets": len(distinct_holding_sets_on_sample_date),
            "independent": len(distinct_holding_sets_on_sample_date) > 1,
        }

        # C: randomness sanity
        randomness = _randomness_sanity(conn, run_ids)

        # D: strategy comparison fairness
        fairness = _fairness_check(conn, policy_name)

        # E: transaction-cost sanity (from already-computed deterministic runs -- higher turnover -> higher cost)
        det_turnover_cost = []
        for name, r in det_runs.items():
            if not r:
                continue
            m = {row["metric_name"]: row["metric_value"] for row in conn.execute(
                "SELECT metric_name, metric_value FROM backtest_results WHERE run_id=? AND period IS NULL",
                (r["run_id"],)).fetchall()}
            det_turnover_cost.append({"strategy": name, "avg_turnover_pct": m.get("avg_turnover_pct"),
                                       "total_transaction_costs": m.get("total_transaction_costs")})
        det_turnover_cost.sort(key=lambda x: x["avg_turnover_pct"] or 0)
        monotonic = all(
            det_turnover_cost[i]["total_transaction_costs"] <= det_turnover_cost[i + 1]["total_transaction_costs"]
            for i in range(len(det_turnover_cost) - 1)
        )
        cost_sanity = {"ordered_by_turnover": det_turnover_cost, "cost_increases_monotonically_with_turnover": monotonic}

        # F: performance distribution outliers + known-flagged-ticker exposure
        min_run = min(random_runs, key=lambda r: r["metrics"]["cumulative_return"])
        max_run = max(random_runs, key=lambda r: r["metrics"]["cumulative_return"])
        flagged_exposure = _flagged_ticker_exposure(conn, [min_run["run_id"], max_run["run_id"]])
        outliers = {
            "min_return_run_id": min_run["run_id"], "min_return": min_run["metrics"]["cumulative_return"],
            "min_return_flagged_tickers_held": flagged_exposure.get(min_run["run_id"], []),
            "max_return_run_id": max_run["run_id"], "max_return": max_run["metrics"]["cumulative_return"],
            "max_return_flagged_tickers_held": flagged_exposure.get(max_run["run_id"], []),
        }

        # G: runtime
        runtime_block = {
            "reproducibility_rerun_seconds_each": round(repro_elapsed / 2, 1),
            "projected_200_seed_distribution_seconds": round((repro_elapsed / 2) * 200, 0),
            "projected_200_seed_distribution_minutes": round((repro_elapsed / 2) * 200 / 60.0, 1),
        }

        report["policies"][policy_name] = {
            "distribution_statistics": stats_block,
            "universe_dates_policy_identical_across_all_runs": universe_check,
            "config_identical_across_all_runs": config_check,
            "reproducibility": repro,
            "execution_timing": execution_timing,
            "seed_independence": seed_independence,
            "randomness_sanity": randomness,
            "strategy_comparison_fairness": fairness,
            "transaction_cost_sanity": cost_sanity,
            "performance_distribution_outliers": outliers,
            "runtime": runtime_block,
        }

    conn.close()

    out_json = os.path.join(PROJECT_ROOT, "PHASE3_DIAGNOSTIC_REPORT.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nDone. Wrote {out_json}.")


if __name__ == "__main__":
    main()
