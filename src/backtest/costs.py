"""
Phase 3 transaction cost model, isolated from signal generation and
portfolio accounting so a strategy's signal never sees costs and costs
never get tuned against observed performance. All values come from
config.yaml's backtest.costs block.
"""


def trade_cost(trade_value_abs, is_sell, cost_config):
    """Returns the dollar cost of a single trade of absolute value
    trade_value_abs (always positive, regardless of buy/sell direction).
    is_sell matters only for the SEC/FINRA fee, which applies to sale
    proceeds, not purchases."""
    commission = trade_value_abs * cost_config.get("commission_pct", 0.0)
    fx = trade_value_abs * cost_config.get("fx_cost_pct", 0.0)
    stamp_duty = trade_value_abs * cost_config.get("stamp_duty_sdrt_pct", 0.0)  # N/A for US equities, documented as 0
    ptm = cost_config.get("ptm_levy_gbp", 0.0)  # N/A for US equities, documented as 0
    sec_fee = trade_value_abs * cost_config.get("sec_finra_fee_pct", 0.0) if is_sell else 0.0
    spread = trade_value_abs * (cost_config.get("bid_ask_spread_bps", 0) / 10000.0)
    slippage = trade_value_abs * (cost_config.get("slippage_bps", 0) / 10000.0)
    return commission + fx + stamp_duty + ptm + sec_fee + spread + slippage


def apply_gross_to_net(gross_return, total_costs, portfolio_value_at_start):
    """Gross -> costs -> net waterfall. Returns (net_return, cost_drag_pct)."""
    if portfolio_value_at_start == 0:
        return gross_return, 0.0
    cost_drag_pct = total_costs / portfolio_value_at_start
    return gross_return - cost_drag_pct, cost_drag_pct
