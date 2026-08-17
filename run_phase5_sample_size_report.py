"""
Phase 5 pre-registration diagnostic #2: empirical sample-size / statistical
power assessment for the overnight-vs-intraday decomposition hypothesis, at
DAILY granularity -- a fundamentally different correlation structure than
Phase 4's monthly-horizon cross-sectional problem, so Phase 4's block-count
formula (months_in_window // horizon_months) is not reused blindly here.

Read-only, like run_phase4_sample_size_check.py and
run_phase5_open_price_data_check.py: trains nothing, computes no predictive
feature, writes only its own report file.

What "effective independent sample size" means here:

  Nominal count: raw (security, trading day) pairs across the eligible
  universe -- large (hundreds of thousands), but two separate correlation
  structures inflate this above the genuine information content:

  1. CROSS-SECTIONAL: on any single day, hundreds of securities' overnight
     returns share the same market-wide overnight shock (earnings season
     clustering aside, most overnight gaps are dominated by
     after-hours/pre-market news that moves the whole market). This
     diagnostic estimates the average pairwise correlation of eligible
     securities' overnight log-returns on the same day and reports an
     effective-breadth estimate (Ns/(1+(Ns-1)*rho), the standard
     equal-correlation effective-N formula), not just the raw eligible
     count.

  2. SERIAL: does today's equal-weighted-proxy overnight return predict
     tomorrow's? Autocorrelation in the overnight-return time series
     (whether from bid-ask bounce, feedback effects, or the overlapping
     nature of some institutional order-flow patterns documented in the
     literature) would mean adjacent trading days are not independent
     draws either. This diagnostic computes the lag-1..lag-10 autocorrelation
     of the equal-weighted proxy overnight-return series for reporting, and
     derives the actual effective sample size via a **circular moving-block
     bootstrap** of the sample mean (v2 -- see script_version below), not the
     closed-form Newey-West-style ratio `N/(1+2*sum(rho_k))` the first
     version of this script used.

     FIX, found by looking at this script's own first real output rather
     than trusting the formula: that closed-form estimator is not robust
     when consecutive lags alternate sign with similar magnitude, which is
     exactly what the real proxy series showed (lag-1 approx -0.12, lag-2
     approx +0.12, nearly cancelling). It produced two nonsensical results
     -- effective-N values LARGER than the nominal day count -- in the
     full-sample and validation windows of the first run
     (PHASE5_SAMPLE_SIZE_REPORT.json). The block bootstrap estimates
     Var(sample mean) empirically instead of via a truncated infinite sum,
     and the reported effective-N is capped at the nominal day count (the
     uncapped figure is reported alongside, not hidden) since this
     diagnostic's purpose is a conservative bound on genuinely independent
     evidence, not a claim of more observations than were actually
     collected.

  Final effective independent block count = (serial effective-N in trading
  days) -- the cross-sectional effective breadth is reported separately
  since it affects a DIFFERENT question (how much cross-sectional power
  exists per day if this were ever extended to a per-security ranking
  strategy) and must not be multiplied into the time-block count, which
  would double-count the same shared-shock structure twice.

Run this locally, in the same folder as the real database:
    python3 run_phase5_sample_size_report.py
"""
import sys
import os
import json
import datetime
import math

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


def _monthly_eligible_sets(conn, cfg):
    """PERMISSIVE eligibility, refreshed monthly (Phase 4's existing
    cadence, reused unchanged) -- NOT re-derived daily. Phase 5's daily
    return series is computed for whichever securities were eligible as of
    the most recent monthly refresh, exactly mirroring how a real monthly-
    membership-check / daily-trading strategy would actually operate."""
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


def _trading_days_in_db(conn, start_date, end_date):
    rows = conn.execute(
        "SELECT DISTINCT date FROM prices WHERE adj_type='total_return' AND date>=? AND date<=? ORDER BY date",
        (start_date, end_date),
    ).fetchall()
    return [r["date"] for r in rows]


def _overnight_log_return_panel(conn, sets_by_month_start, trading_days):
    """For every trading day t (after the first), for every security
    eligible under the most recent monthly refresh at-or-before t,
    overnight_t = ln(open_t / close_{t-1}), using total_return prices
    (consistent with the rest of the codebase's ratio-only-within-window
    rule -- both endpoints are one session apart, well within any bounded
    window). Returns:
        pairs: list of (date, security_id, overnight_log_return)
        proxy_by_date: {date: mean overnight_log_return across the day's panel}
    """
    month_starts = sorted(sets_by_month_start.keys())

    def eligible_set_for(date):
        applicable = [m for m in month_starts if m <= date]
        if not applicable:
            return []
        return sets_by_month_start[applicable[-1]]

    # Pull all total_return open/close rows once, indexed by (security_id, date).
    all_sec_ids = sorted({sid for s in sets_by_month_start.values() for sid in s})
    if not all_sec_ids:
        return [], {}
    placeholders = ",".join("?" * len(all_sec_ids))
    rows = conn.execute(
        f"SELECT security_id, date, open, close FROM prices "
        f"WHERE security_id IN ({placeholders}) AND adj_type='total_return' "
        f"AND date>=? AND date<=?",
        (*all_sec_ids, trading_days[0], trading_days[-1]),
    ).fetchall()
    by_sec = {}
    for r in rows:
        by_sec.setdefault(r["security_id"], {})[r["date"]] = (r["open"], r["close"])

    pairs = []
    proxy_by_date = {}
    for i in range(1, len(trading_days)):
        prev_day, day = trading_days[i - 1], trading_days[i]
        elig = eligible_set_for(day)
        day_returns = []
        for sid in elig:
            hist = by_sec.get(sid, {})
            today = hist.get(day)
            prev = hist.get(prev_day)
            if not today or not prev:
                continue
            open_t, _ = today
            _, close_prev = prev
            if not open_t or not close_prev or open_t <= 0 or close_prev <= 0:
                continue
            r = math.log(open_t / close_prev)
            pairs.append((day, sid, r))
            day_returns.append(r)
        if day_returns:
            proxy_by_date[day] = sum(day_returns) / len(day_returns)

    return pairs, proxy_by_date


def _average_pairwise_cross_sectional_correlation(pairs, sample_days=250):
    """Estimates rho (average pairwise correlation across securities'
    same-day overnight returns) from a sample of days, via the identity
    var(cross-sectional mean) relationship: for N equally-correlated
    variables each with variance sigma^2, var(mean) = sigma^2 * (1 + (N-1)*rho) / N.
    Rearranged: rho = (N * var(mean)/sigma^2 - 1) / (N - 1), estimated
    per-day then averaged across sampled days with N>=10, since fewer than
    10 names makes the per-day sigma^2/rho estimate too noisy to trust."""
    from collections import defaultdict
    import statistics

    by_date = defaultdict(list)
    for date, sid, r in pairs:
        by_date[date].append(r)

    dates = sorted(by_date.keys())
    if len(dates) > sample_days:
        step = len(dates) // sample_days
        dates = dates[::step][:sample_days]

    # Direct average pairwise correlation via covariance of a random
    # sub-sample of security pairs across the FULL date range -- a
    # multi-day panel per security is needed for a meaningful correlation
    # estimate, which a single day's cross-sectional mean/variance alone
    # cannot supply. Build a wide panel: security -> list of (date, r)
    # over the dates all securities share, then sample pairs.
    from collections import defaultdict as dd
    by_sec = dd(dict)
    for date, sid, r in pairs:
        by_sec[sid][date] = r

    sec_ids = list(by_sec.keys())
    import random
    rng = random.Random(42)
    sample_pairs = []
    max_pairs = 2000
    attempts = 0
    while len(sample_pairs) < max_pairs and attempts < max_pairs * 20 and len(sec_ids) >= 2:
        attempts += 1
        a, b = rng.sample(sec_ids, 2)
        common_dates = set(by_sec[a].keys()) & set(by_sec[b].keys())
        if len(common_dates) < 60:
            continue
        xs = [by_sec[a][d] for d in sorted(common_dates)]
        ys = [by_sec[b][d] for d in sorted(common_dates)]
        sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
        if sx == 0 or sy == 0:
            continue
        mx, my = statistics.mean(xs), statistics.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
        sample_pairs.append(cov / (sx * sy))

    if not sample_pairs:
        return None, 0
    return statistics.mean(sample_pairs), len(sample_pairs)


def _autocorrelation(series, max_lag=20):
    import statistics
    n = len(series)
    mean = statistics.mean(series)
    var = sum((x - mean) ** 2 for x in series)
    if var == 0:
        return []
    out = []
    for k in range(1, max_lag + 1):
        if k >= n:
            break
        cov_k = sum((series[i] - mean) * (series[i + k] - mean) for i in range(n - k))
        out.append(cov_k / var)
    return out


def _choose_block_length(acf, two_sigma):
    """Block length for the bootstrap below: the last lag whose |rho_k|
    exceeds its own significance band, times 2, plus a margin -- long
    enough to span the full measured decorrelation range rather than an
    arbitrary round number, floored at 5 and capped at 20 trading days
    (same cap the original ACF truncation used, kept for continuity)."""
    k = 0
    for i, rho_k in enumerate(acf, start=1):
        if two_sigma is not None and abs(rho_k) < two_sigma:
            break
        k = i
    return max(5, min(20, k * 2 + 3))


def _block_bootstrap_n_eff(series, block_length, n_boot=1000, seed=42):
    """Empirical effective sample size via a circular moving-block
    bootstrap of the mean -- see the module docstring's "FIX" note for why
    this replaces the closed-form N/(1+2*sum(rho_k)) estimator.

    Method: resample the series in circular blocks of `block_length`,
    estimate Var(sample mean) empirically across n_boot resamples, then
    invert the definition Var(mean) = population_variance / N_eff, i.e.
    N_eff = population_variance / block_bootstrap_var_of_mean. The
    reported N_eff is capped at the nominal N -- see module docstring for
    why -- with the uncapped figure reported alongside for transparency.
    """
    import statistics
    import random

    n = len(series)
    if n < block_length * 4 or n < 8:
        return None
    pop_var = statistics.pvariance(series)
    if pop_var == 0:
        return None

    rng = random.Random(seed)
    n_blocks_needed = math.ceil(n / block_length)
    boot_means = []
    for _ in range(n_boot):
        sample = []
        for _ in range(n_blocks_needed):
            start = rng.randrange(0, n)  # circular block start, wraps around the series
            sample.extend(series[(start + j) % n] for j in range(block_length))
        sample = sample[:n]
        boot_means.append(sum(sample) / len(sample))

    var_of_mean_bootstrap = statistics.pvariance(boot_means)
    if var_of_mean_bootstrap == 0:
        return None

    n_eff_uncapped = pop_var / var_of_mean_bootstrap
    return {
        "block_length_used": block_length,
        "n_bootstrap_resamples": n_boot,
        "n_effective_uncapped": round(n_eff_uncapped, 1),
        "n_effective_independent_days": round(min(n_eff_uncapped, n), 1),
        "capped_at_nominal": n_eff_uncapped > n,
    }


def _effective_n_report(series):
    """Combines the ACF (reported for interpretability -- e.g. the real
    lag-1 reversal finding is itself an actionable Tier-1 feature
    candidate, not just a nuisance parameter) with the block-bootstrap
    effective-N (used for actually sizing confidence intervals/budgets,
    per the module docstring's FIX note)."""
    n = len(series)
    acf = _autocorrelation(series, max_lag=min(20, n // 4 if n >= 8 else 1))
    two_sigma = 2.0 / math.sqrt(n) if n > 0 else None
    block_length = _choose_block_length(acf, two_sigma)
    bootstrap = _block_bootstrap_n_eff(series, block_length)

    out = {
        "n_nominal_days": n,
        "acf_lag_1_to_10": [round(v, 4) for v in acf[:10]],
        "two_sigma_band": round(two_sigma, 4) if two_sigma else None,
        "block_bootstrap": bootstrap,
        "n_effective_independent_days": bootstrap["n_effective_independent_days"] if bootstrap else n,
    }
    return out


def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: database not found at {DB_PATH}")
        sys.exit(1)

    cfg = yaml.safe_load(open(CONFIG_PATH))
    conn = init_db(DB_PATH, SCHEMA_PATH, reset=False, force=False)

    start_date = cfg["data"]["price_start_date"]
    end_date = cfg["data"]["price_end_date"]

    print("Computing monthly PERMISSIVE eligibility sets (reusing build_eligible_universe)...")
    sets_by_month_start = _monthly_eligible_sets(conn, cfg)

    print("Resolving trading-day calendar from the database...")
    trading_days = _trading_days_in_db(conn, start_date, end_date)

    print(f"Building overnight-return panel across {len(trading_days)} trading days...")
    pairs, proxy_by_date = _overnight_log_return_panel(conn, sets_by_month_start, trading_days)

    proxy_dates_sorted = sorted(proxy_by_date.keys())
    proxy_series = [proxy_by_date[d] for d in proxy_dates_sorted]

    print("Estimating average pairwise cross-sectional correlation (sampled security pairs)...")
    rho, n_pairs_sampled = _average_pairwise_cross_sectional_correlation(pairs)

    print("Estimating serial autocorrelation / effective independent days (equal-weighted proxy series)...")
    serial = _effective_n_report(proxy_series)

    # Simple chronological 60/15/25 split of the proxy date range, for
    # reporting effective-N per split window using the same autocorrelation
    # methodology applied to each sub-series independently (train-window
    # autocorrelation need not match the full-sample figure).
    n_dates = len(proxy_dates_sorted)
    train_end = round(n_dates * 0.60)
    val_end = round(n_dates * 0.75)
    windows = {
        "train": proxy_series[:train_end],
        "validation": proxy_series[train_end:val_end],
        "test_LOCKED": proxy_series[val_end:],
    }
    window_stats = {name: _effective_n_report(series) for name, series in windows.items()}

    avg_eligible_breadth = (
        sum(len(s) for s in sets_by_month_start.values()) / len(sets_by_month_start)
        if sets_by_month_start else None
    )
    cross_sectional_effective_breadth = None
    if rho is not None and avg_eligible_breadth:
        n_s = avg_eligible_breadth
        cross_sectional_effective_breadth = n_s / (1 + (n_s - 1) * rho) if (1 + (n_s - 1) * rho) != 0 else None

    report = {
        "generated_at": _now(),
        "script_version": 2,  # v2: effective-N via block bootstrap, replacing the fragile closed-form formula
        "date_range": [start_date, end_date],
        "nominal_security_date_pairs": len(pairs),
        "trading_days_with_a_computed_proxy_return": len(proxy_series),
        "average_eligible_breadth_per_month": round(avg_eligible_breadth, 1) if avg_eligible_breadth else None,
        "cross_sectional_correlation": {
            "estimated_average_pairwise_rho": round(rho, 4) if rho is not None else None,
            "n_security_pairs_sampled": n_pairs_sampled,
            "implied_cross_sectional_effective_breadth": round(cross_sectional_effective_breadth, 1)
            if cross_sectional_effective_breadth else None,
            "note": "Reported for context (per-day cross-sectional power), NOT multiplied into the "
                    "time-block count below -- that would double-count the same shared-shock structure.",
        },
        "serial_autocorrelation_full_sample": serial,
        "serial_autocorrelation_by_split_window": window_stats,
    }

    conn.close()

    out_json = os.path.join(PROJECT_ROOT, "PHASE5_SAMPLE_SIZE_REPORT.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nWrote {out_json}")
    print(f"Full-sample effective independent trading days: "
          f"{serial['n_effective_independent_days']} (of {serial['n_nominal_days']} nominal)")
    print(f"Locked-test-window effective independent trading days: "
          f"{window_stats['test_LOCKED']['n_effective_independent_days']} "
          f"(of {window_stats['test_LOCKED']['n_nominal_days']} nominal)")
    print("This script computes no predictive feature and trains no model.")


if __name__ == "__main__":
    main()
