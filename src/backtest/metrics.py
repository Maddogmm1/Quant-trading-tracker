"""
Phase 3 performance metrics. Computed from a portfolio value time
series -- pure functions, no database access, testable against
hand-calculated values.
"""
import math
import statistics


def _period_returns(values):
    return [values[i] / values[i - 1] - 1.0 for i in range(1, len(values)) if values[i - 1]]


def cumulative_return(values):
    if not values or values[0] == 0:
        return 0.0
    return values[-1] / values[0] - 1.0


def cagr(values, dates):
    if len(values) < 2 or values[0] <= 0:
        return 0.0
    import datetime
    years = (datetime.date.fromisoformat(dates[-1]) - datetime.date.fromisoformat(dates[0])).days / 365.25
    if years <= 0:
        return 0.0
    return (values[-1] / values[0]) ** (1.0 / years) - 1.0


def annualized_volatility(values, periods_per_year=12):
    rets = _period_returns(values)
    if len(rets) < 2:
        return 0.0
    return statistics.stdev(rets) * math.sqrt(periods_per_year)


def sharpe_ratio(values, periods_per_year=12, risk_free_rate=0.0):
    rets = _period_returns(values)
    if len(rets) < 2:
        return 0.0
    excess = [r - risk_free_rate / periods_per_year for r in rets]
    mean_excess = statistics.mean(excess)
    sd = statistics.stdev(excess)
    if sd == 0:
        return 0.0
    return (mean_excess / sd) * math.sqrt(periods_per_year)


def sortino_ratio(values, periods_per_year=12, risk_free_rate=0.0):
    rets = _period_returns(values)
    if len(rets) < 2:
        return 0.0
    excess = [r - risk_free_rate / periods_per_year for r in rets]
    downside = [min(0, r) for r in excess]
    downside_sd = math.sqrt(sum(d ** 2 for d in downside) / len(downside)) if downside else 0.0
    if downside_sd == 0:
        return 0.0
    return (statistics.mean(excess) / downside_sd) * math.sqrt(periods_per_year)


def max_drawdown(values):
    if not values:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            dd = (v - peak) / peak
            max_dd = min(max_dd, dd)
    return max_dd


def calmar_ratio(values, dates):
    mdd = max_drawdown(values)
    if mdd == 0:
        return 0.0
    return cagr(values, dates) / abs(mdd)


def win_rate(values):
    rets = _period_returns(values)
    if not rets:
        return 0.0
    return sum(1 for r in rets if r > 0) / len(rets)


def annual_returns(values, dates):
    """Year-by-year return breakdown, not just a single headline CAGR."""
    import datetime
    by_year = {}
    for v, d in zip(values, dates):
        year = datetime.date.fromisoformat(d).year
        by_year.setdefault(year, []).append(v)
    result = {}
    for year, vals in sorted(by_year.items()):
        if len(vals) >= 2 and vals[0]:
            result[year] = vals[-1] / vals[0] - 1.0
    return result


def full_report(portfolio_history, cost_total, trade_count, periods_per_year=12):
    """portfolio_history: list of {'as_of_date':..., 'portfolio_value':...}
    in chronological order, as produced by accounting.Portfolio.history."""
    values = [h["portfolio_value"] for h in portfolio_history]
    dates = [h["as_of_date"] for h in portfolio_history]
    turnovers = [h.get("turnover_pct", 0.0) for h in portfolio_history]

    return {
        "cumulative_return": cumulative_return(values),
        "cagr": cagr(values, dates),
        "annualized_volatility": annualized_volatility(values, periods_per_year),
        "sharpe_ratio": sharpe_ratio(values, periods_per_year),
        "sortino_ratio": sortino_ratio(values, periods_per_year),
        "max_drawdown": max_drawdown(values),
        "calmar_ratio": calmar_ratio(values, dates),
        "win_rate": win_rate(values),
        "avg_turnover_pct": statistics.mean(turnovers) if turnovers else 0.0,
        "total_transaction_costs": cost_total,
        "number_of_trades": trade_count,
        "annual_returns": annual_returns(values, dates),
    }
