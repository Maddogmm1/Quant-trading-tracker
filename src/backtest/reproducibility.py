"""
Phase 3 reproducibility. Persists everything needed to reproduce a run
byte-for-byte: config, seed, dates, universe definition, quality policy,
cost/execution assumptions, code version, results, and the coverage
report. Two runs with identical config_json + random_seed should produce
identical backtest_results rows -- see
tests/test_backtest_reproducibility.py.
"""
import json
import subprocess


def _code_version():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
    except Exception:
        return "unknown"


def save_run(conn, *, run_label, config, random_seed, start_date, end_date,
             rebalance_frequency, universe_definition, data_quality_policy_name,
             strategy_name, cost_config, execution_config, report, coverage_report,
             created_at):
    cur = conn.execute(
        """INSERT INTO backtest_runs
           (run_label, created_at, config_json, random_seed, code_version, start_date, end_date,
            rebalance_frequency, universe_definition, data_quality_policy_name, strategy_name,
            cost_assumptions_json, execution_assumptions_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_label, created_at, json.dumps(config, sort_keys=True, default=str), random_seed, _code_version(),
         start_date, end_date, rebalance_frequency, universe_definition, data_quality_policy_name, strategy_name,
         json.dumps(cost_config, sort_keys=True), json.dumps(execution_config, sort_keys=True)),
    )
    run_id = cur.lastrowid

    for metric_name, value in report.items():
        if metric_name == "annual_returns":
            for year, ret in value.items():
                conn.execute(
                    "INSERT INTO backtest_results (run_id, metric_name, metric_value, period) VALUES (?,?,?,?)",
                    (run_id, "annual_return", ret, str(year)),
                )
        else:
            conn.execute(
                "INSERT INTO backtest_results (run_id, metric_name, metric_value, period) VALUES (?,?,?,NULL)",
                (run_id, metric_name, value),
            )

    for c in coverage_report:
        conn.execute(
            """INSERT INTO backtest_coverage
               (run_id, as_of_date, eligible_constituents, usable_data_count, excluded_by_quality,
                provider_empty_count, identity_unresolved_count, partial_history_count, final_tradable_count)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (run_id, c["as_of_date"], c["eligible_constituents"], c["usable_data_count"], c["excluded_by_quality"],
             c["provider_empty_count"], c["identity_unresolved_count"], c["partial_history_count"], c["final_tradable_count"]),
        )

    conn.commit()
    return run_id


def _price_asof(conn, security_id, as_of_date, adj_type):
    row = conn.execute(
        "SELECT close FROM prices WHERE security_id=? AND adj_type=? AND date<=? ORDER BY date DESC LIMIT 1",
        (security_id, adj_type, as_of_date),
    ).fetchone()
    return row["close"] if row else None


def save_positions(conn, run_id, portfolio_history, adj_type="total_return"):
    """weight and position_value are computed here rather than carried on
    the Portfolio snapshot, since Portfolio.history only tracks shares.
    Both are NOT NULL columns in backtest_positions, so a position with
    no resolvable price (e.g. delisted with no data at all as of this
    date) is skipped rather than written with a fabricated value."""
    for snap in portfolio_history:
        pv = snap.get("portfolio_value") or 0.0
        for sec_id, shares in snap["positions"].items():
            price = _price_asof(conn, sec_id, snap["as_of_date"], adj_type)
            if price is None:
                continue
            position_value = shares * price
            weight = (position_value / pv) if pv else 0.0
            conn.execute(
                """INSERT INTO backtest_positions (run_id, as_of_date, security_id, shares, weight, position_value)
                   VALUES (?,?,?,?,?,?)""",
                (run_id, snap["as_of_date"], sec_id, shares, weight, position_value),
            )
    conn.commit()


def load_results(conn, run_id):
    rows = conn.execute(
        "SELECT metric_name, metric_value, period FROM backtest_results WHERE run_id=? ORDER BY metric_name, period",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]
