"""
Phase 3 backtest engine: orchestrates universe construction, signal
generation, execution timing, portfolio accounting, and coverage
reporting into one run. This is the only place these pieces are wired
together -- everything it calls is independently testable in isolation.
"""
from src.backtest.universe import build_eligible_universe
from src.backtest.execution import PointInTimeDataAccess, next_market_session, next_rebalance_dates
from src.backtest.accounting import Portfolio
from src.backtest import metrics as metrics_mod


def _universe_cache_key(signal_date, data_quality_policy, predeclared_filters,
                         universe_definition, lookback_days, adj_type):
    return (
        signal_date,
        frozenset(data_quality_policy.items()),
        frozenset((predeclared_filters or {}).items()),
        universe_definition, lookback_days, adj_type,
    )


def run_backtest(conn, strategy, rebalance_dates, data_quality_policy, cost_config,
                  predeclared_filters=None, starting_cash=100000.0, lookback_days=252,
                  adj_type="total_return", universe_definition="SP500",
                  accounting_tolerance=1e-6, universe_cache=None):
    """
    Runs one full backtest and returns (metrics_report, coverage_report,
    portfolio). Doesn't persist to the database itself -- that's
    src.backtest.reproducibility's job, which keeps this function
    testable without any DB write side effects.

    universe_cache: optional dict, shared by the caller across multiple
    run_backtest() calls (e.g. the 4 benchmarks plus N random seeds at
    one data_quality_policy). build_eligible_universe() is a pure
    function of its arguments, so memoizing it across calls that share
    the same (date, policy, filters, universe_definition, lookback_days,
    adj_type) is a speed optimization only -- it can't change results.
    Leave it None (the default) to skip caching. See
    tests/test_backtest_caching.py for the cached-vs-uncached equivalence
    test.
    """
    portfolio = Portfolio(starting_cash, cost_config, accounting_tolerance)
    coverage_report = []

    for signal_date in rebalance_dates:
        if universe_cache is not None:
            cache_key = _universe_cache_key(signal_date, data_quality_policy, predeclared_filters,
                                             universe_definition, lookback_days, adj_type)
            if cache_key in universe_cache:
                eligible, exclusion_report = universe_cache[cache_key]
            else:
                eligible, exclusion_report = build_eligible_universe(
                    conn, signal_date, data_quality_policy, predeclared_filters=predeclared_filters,
                    universe_definition=universe_definition, lookback_days=lookback_days, adj_type=adj_type,
                )
                universe_cache[cache_key] = (eligible, exclusion_report)
        else:
            eligible, exclusion_report = build_eligible_universe(
                conn, signal_date, data_quality_policy, predeclared_filters=predeclared_filters,
                universe_definition=universe_definition, lookback_days=lookback_days, adj_type=adj_type,
            )

        reasons = {}
        for e in exclusion_report:
            reasons.setdefault(e["reason"], 0)
            reasons[e["reason"]] += 1

        coverage_report.append({
            "as_of_date": signal_date,
            "eligible_constituents": len(eligible) + len(exclusion_report),
            "usable_data_count": len(eligible),
            "excluded_by_quality": sum(v for k, v in reasons.items() if "policy" in k or "severe_ohlc" in k),
            "provider_empty_count": reasons.get("provider_empty_no_data", 0),
            "identity_unresolved_count": reasons.get("identity_unresolved_no_security_row", 0)
                                          + reasons.get("identity_unresolved_excluded_by_policy", 0),
            "partial_history_count": reasons.get("insufficient_full_history_required_by_policy", 0),
            "final_tradable_count": len(eligible),
        })

        if not eligible:
            # No eligible universe this period -- if we're already holding
            # positions from earlier, still mark them to market so the
            # value series has no gap (see Portfolio.mark_to_market).
            if portfolio.positions:
                portfolio.mark_to_market(conn, signal_date, adj_type=adj_type)
            continue

        data_access = PointInTimeDataAccess(conn, signal_date, adj_type=adj_type)
        signal_weights = strategy.generate_signal(data_access, signal_date, eligible)
        if not signal_weights:
            # Strategy deliberately chose not to trade this period (e.g.
            # BuyAndHold after its initial purchase) -- mark existing
            # positions to market rather than leaving a gap in history.
            if portfolio.positions:
                portfolio.mark_to_market(conn, signal_date, adj_type=adj_type)
            continue

        # Trade at the next valid market session after the signal date,
        # not the signal date's own close. Resolved from the full
        # database's trading calendar rather than any single security's
        # own price availability (an earlier version picked a
        # "representative" security for this and let its data gaps delay
        # the whole portfolio -- see execution.next_market_session).
        # Per-security execution eligibility on the resolved date is
        # handled separately, inside rebalance_to.
        execution_date = next_market_session(conn, signal_date, adj_type=adj_type) or signal_date

        portfolio.rebalance_to(conn, execution_date, signal_weights, adj_type=adj_type)

    report = metrics_mod.full_report(
        portfolio.history, portfolio.total_costs_paid, portfolio.trade_count,
    )
    return report, coverage_report, portfolio
