"""
Phase 5 Tier 0 primary significance test
(PHASE5_OVERNIGHT_GAP_SPECIFICATION.md section 6, "Tier 0"). A one-sample
and a paired-difference test on the equal-weighted overnight/intraday
proxy series (src.ml.overnight_targets.proxy_series_for_dates), using a
circular moving-block bootstrap confidence interval rather than a naive
i.i.d. standard error -- required by section 3's finding that daily
(security, day) pairs are neither cross-sectionally nor serially
independent (section 3.4: cross-sectional rho~=0.43, and real, though
formula-sensitive, serial autocorrelation).

This module computes NO feature, fits NO model, and is deliberately
separate from src/ml/baselines.py's evaluate_predictions() -- that
function is built for per-date Spearman IC on a cross-sectional ranking
problem (Phase 4's target), not a one-sample/paired test on a single
aggregate time series (Phase 5's Tier 0 target). Uses the same
block-bootstrap approach as run_phase5_sample_size_report.py's
_block_bootstrap_n_eff() (reimplemented here, not imported, since src/
modules should not depend on a repo-root run_*.py script meant to be run
standalone), generalized to bootstrap the statistic of interest (a mean,
or a paired mean difference) directly rather than just an effective N.
"""
import math
import random
import statistics


def _circular_block_resample(series, block_length, rng):
    n = len(series)
    n_blocks_needed = math.ceil(n / block_length)
    sample = []
    for _ in range(n_blocks_needed):
        start = rng.randrange(0, n)
        sample.extend(series[(start + j) % n] for j in range(block_length))
    return sample[:n]


def choose_block_length(series, max_lag=20):
    """Same block-length-selection logic as
    run_phase5_sample_size_report.py's _choose_block_length(): derive from
    the last lag whose autocorrelation exceeds its own significance band,
    doubled plus a margin, floored at 5 and capped at 20 trading days."""
    n = len(series)
    if n < 8:
        return 5
    mean = statistics.mean(series)
    var = sum((x - mean) ** 2 for x in series)
    two_sigma = 2.0 / math.sqrt(n)
    k = 0
    if var > 0:
        for lag in range(1, min(max_lag, n // 4) + 1):
            cov_k = sum((series[i] - mean) * (series[i + lag] - mean) for i in range(n - lag))
            rho_k = cov_k / var
            if abs(rho_k) < two_sigma:
                break
            k = lag
    return max(5, min(20, k * 2 + 3))


def block_bootstrap_mean_ci(series, block_length, n_boot=5000, ci=0.95, seed=42):
    """Block-bootstrap confidence interval for the mean of `series`.
    Returns None if series is too short for the given block_length (needs
    at least 4 blocks' worth of data, matching the sample-size script's
    own minimum) -- an explicit None, not a fabricated wide interval, so
    callers can distinguish "not enough data" from "a real, wide CI"."""
    n = len(series)
    if n < block_length * 4:
        return None
    point_estimate = statistics.mean(series)
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = _circular_block_resample(series, block_length, rng)
        boot_means.append(statistics.mean(sample))
    boot_means.sort()
    alpha = 1 - ci
    lo_idx = max(0, min(int(round((alpha / 2) * n_boot)), n_boot - 1))
    hi_idx = max(0, min(int(round((1 - alpha / 2) * n_boot)) - 1, n_boot - 1))
    return {
        "n_nominal": n,
        "block_length_used": block_length,
        "n_bootstrap_resamples": n_boot,
        "point_estimate": point_estimate,
        "ci_level": ci,
        "ci_low": boot_means[lo_idx],
        "ci_high": boot_means[hi_idx],
        "ci_excludes_zero": not (boot_means[lo_idx] <= 0.0 <= boot_means[hi_idx]),
    }


def block_bootstrap_paired_difference_ci(series_a, series_b, block_length, n_boot=5000, ci=0.95, seed=42):
    """Block-bootstrap CI for mean(series_a - series_b), resampling the
    PAIRED difference series (not each series independently -- that would
    discard the same-day correlation between the two components, which is
    exactly the structure a paired test needs to respect). series_a and
    series_b must already be the same length and date-aligned by the
    caller -- never silently re-aligned here."""
    if len(series_a) != len(series_b):
        raise ValueError(
            f"series_a and series_b must be the same length and already date-aligned; "
            f"got {len(series_a)} vs {len(series_b)}"
        )
    diffs = [a - b for a, b in zip(series_a, series_b)]
    return block_bootstrap_mean_ci(diffs, block_length, n_boot=n_boot, ci=ci, seed=seed)


def tier0_test(overnight_series, intraday_series, n_boot=5000, ci=0.95, seed=42):
    """The full Tier 0 primary test (spec sections 6 and 8): is the mean
    overnight_proxy(t) distinguishable from zero, AND from
    intraday_proxy(t) over the same days. Both inputs must already be
    date-aligned, equal-length, with no None entries (the caller excludes
    dates where either proxy is unresolvable -- see
    overnight_targets.proxy_series_for_dates, which reports "n_missing"
    precisely so that exclusion is auditable, not silent).

    Returns a dict with both CIs plus the decision-table classification
    from spec section 8 (failure / inconclusive /
    genuine_effect_candidate), using the annualised-magnitude threshold
    from that section. This function does NOT know whether it is being
    run on train, validation, or the locked test set -- that discipline
    belongs to the caller (spec section 9/12: the locked test set may
    only be touched once, after every other decision is frozen). A
    "genuine_effect_candidate" classification is explicitly NOT a final
    "success" -- spec section 8's robustness checks still apply before
    that label can be upgraded.
    """
    if len(overnight_series) != len(intraday_series):
        raise ValueError("overnight_series and intraday_series must be the same length and date-aligned")
    if not overnight_series:
        return {
            "n_days": 0, "block_length_used": None,
            "overnight_vs_zero_ci": None, "overnight_vs_intraday_paired_ci": None,
            "annualised_overnight_pct": None,
            "classification": "inconclusive",
            "classification_reason": "no resolvable days in the requested window",
        }

    block_length = choose_block_length(overnight_series)
    overnight_ci = block_bootstrap_mean_ci(overnight_series, block_length, n_boot=n_boot, ci=ci, seed=seed)
    paired_ci = block_bootstrap_paired_difference_ci(
        overnight_series, intraday_series, block_length, n_boot=n_boot, ci=ci, seed=seed
    )

    # Annualise the overnight point estimate (mean daily log return * 252
    # trading days) for the section 8 economic-magnitude threshold -- 1.0
    # percentage point, chosen per the spec to sit comfortably below the
    # smallest realistic daily-turnover cost drag, not to flatter this
    # specific result.
    trading_days_per_year = 252
    annualised_pct = None
    if overnight_ci is not None:
        annualised_pct = overnight_ci["point_estimate"] * trading_days_per_year * 100.0

    if overnight_ci is None or paired_ci is None:
        classification = "inconclusive"
        reason = "insufficient data for a trustworthy block-bootstrap CI at the chosen block length"
    elif not overnight_ci["ci_excludes_zero"] or not paired_ci["ci_excludes_zero"]:
        classification = "failure"
        reason = "overnight-vs-zero CI or overnight-vs-intraday paired CI contains zero"
    elif abs(annualised_pct) < 1.0:
        classification = "inconclusive"
        reason = f"CI excludes zero but annualised magnitude ({annualised_pct:.3f}pp) is below the 1.0pp threshold"
    else:
        classification = "genuine_effect_candidate"
        reason = ("CI excludes zero on both tests and magnitude clears the 1.0pp threshold -- "
                  "spec section 8 ROBUSTNESS CHECKS NOT YET APPLIED, this is not a final result")

    return {
        "n_days": len(overnight_series),
        "block_length_used": block_length,
        "overnight_vs_zero_ci": overnight_ci,
        "overnight_vs_intraday_paired_ci": paired_ci,
        "annualised_overnight_pct": annualised_pct,
        "classification": classification,
        "classification_reason": reason,
    }
