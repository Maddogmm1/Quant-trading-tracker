"""
Target/label construction for the Phase 4 model (PHASE4_SPECIFICATION.md
section 3).

y_i(t,h) = r_i(t,h) - B(t,h)
    r_i(t,h) = TR_i(t+h) / TR_i(t) - 1
    B(t,h)   = mean over j in ELIG(t) of r_j(t,h)   -- Phase 3's own
               equal_weight_sp500 construction, reused not redefined.

rank_i(t,h)  = cross-sectional percentile rank of y_i(t,h) within ELIG(t)
z_i(t,h)     = 1[y_i(t,h) > 0]

A security that loses price coverage between t and t+h is never silently
dropped -- that would reintroduce survivorship bias one label at a time --
and never gets a fabricated return. Instead it gets a frozen terminal
value at the last resolvable price, with label_truncated=True recorded
alongside; both are required output fields, not optional.

`t` and `t+h` must both be entries in Phase 3's own rebalance_dates
sequence (src.backtest.execution.next_rebalance_dates), so horizons are
counted in rebalance periods, never raw calendar days. This module
doesn't choose t+h itself -- the caller resolves it from rebalance_dates
and passes it in, so there's exactly one definition of "3 months ahead"
in the whole codebase.
"""
from src.ml.features import latest_price_at_or_before


def resolve_forward_price(conn, security_id, target_date, adj_type="total_return", normal_gap_days=10):
    """Returns (resolved_date, price, truncated) for security_id's price
    at target_date.

    target_date is one of Phase 3's own rebalance_dates -- a literal
    calendar date (e.g. a month's 1st), not guaranteed to be an actual
    trading session (next_rebalance_dates() leaves confirming that to the
    caller). Requiring an exact row match against that calendar string
    would flag every ordinary weekend/holiday month-start as a
    "truncated" (delisted) label, since almost no security has a literal
    trading row on a literal month-1st. So target_date resolves to the
    nearest actual trading session on or after it (mirroring
    src.backtest.execution.next_market_session's "next session at/after a
    nominal date" convention for execution dates); a session landing a
    few days later is normal calendar noise, not a truncation.

    truncated=True only when either (a) no trading session exists
    on-or-after target_date within normal_gap_days at all, or (b) the
    nearest one on-or-after is suspiciously far away. Both cases fall
    back to the last available price at-or-before target_date (a frozen
    terminal value), which is the genuine "this security stopped
    trading" case. Returns None if no resolvable price exists at all,
    before or after."""
    import datetime as _dt

    on_or_after = conn.execute(
        "SELECT date, close FROM prices WHERE security_id=? AND adj_type=? AND date>=? "
        "ORDER BY date ASC LIMIT 1",
        (security_id, adj_type, target_date),
    ).fetchone()
    if on_or_after and on_or_after["close"] is not None:
        gap_days = (_dt.date.fromisoformat(on_or_after["date"]) - _dt.date.fromisoformat(target_date)).days
        if gap_days <= normal_gap_days:
            return (on_or_after["date"], on_or_after["close"], False)

    fallback = latest_price_at_or_before(conn, security_id, target_date, adj_type)
    if fallback is None:
        return None
    return (fallback[0], fallback[1], True)


def forward_return(conn, security_id, t_date, target_date, adj_type="total_return"):
    """r_i(t,h) = TR_i(t+h) / TR_i(t) - 1, plus the truncation flag from
    resolve_forward_price. Returns None if either endpoint is
    unresolvable, never a fabricated return. The anchor at t uses the
    same "nearest price at-or-before" rule as feature construction (via
    latest_price_at_or_before) -- one point-in-time price-resolution rule
    shared by both."""
    anchor = latest_price_at_or_before(conn, security_id, t_date, adj_type)
    if anchor is None or not anchor[1]:
        return None
    resolved = resolve_forward_price(conn, security_id, target_date, adj_type)
    if resolved is None:
        return None
    resolved_date, price, truncated = resolved
    return {
        "return": price / anchor[1] - 1.0,
        "t_price_date": anchor[0],
        "target_price_date": target_date,
        "resolved_price_date": resolved_date,
        "label_truncated": truncated,
    }


def compute_labels_for_universe(conn, eligible_security_ids, t_date, target_date, adj_type="total_return"):
    """Computes r_i(t,h) for every i in eligible_security_ids (ELIG(t),
    passed in by the caller and never recomputed here as ELIG(t+h) --
    that would leak future universe membership into the label), then
    derives B(t,h), y_i, rank_i, and z_i from that single pass.
    Securities with no resolvable return (missing entirely, not even a
    fallback price) are omitted from the returned dict and reported
    separately in 'excluded_no_price', never silently folded in as a
    zero return.

    Returns a dict:
        {
          "per_security": {security_id: {
              "r": float, "y": float, "rank": float, "z": int,
              "label_truncated": bool,
              "t_price_date": str, "resolved_price_date": str,
          }, ...},
          "benchmark_return": float or None,   # B(t,h)
          "excluded_no_price": [security_id, ...],
          "truncated_count": int,
        }
    """
    raw = {}
    excluded = []
    for sec_id in eligible_security_ids:
        fr = forward_return(conn, sec_id, t_date, target_date, adj_type)
        if fr is None:
            excluded.append(sec_id)
            continue
        raw[sec_id] = fr

    returns = {sid: fr["return"] for sid, fr in raw.items()}
    if returns:
        benchmark_return = sum(returns.values()) / len(returns)
    else:
        benchmark_return = None

    per_security = {}
    if benchmark_return is not None:
        sorted_returns = sorted(returns.values())
        n = len(sorted_returns)
        for sid, fr in raw.items():
            r = fr["return"]
            y = r - benchmark_return
            below = sum(1 for v in sorted_returns if v < r)
            equal = sum(1 for v in sorted_returns if v == r)
            rank = (below + 0.5 * equal) / n if n > 1 else None
            per_security[sid] = {
                "r": r,
                "y": y,
                "rank": rank,
                "z": 1 if y > 0 else 0,
                "label_truncated": fr["label_truncated"],
                "t_price_date": fr["t_price_date"],
                "resolved_price_date": fr["resolved_price_date"],
            }

    return {
        "per_security": per_security,
        "benchmark_return": benchmark_return,
        "excluded_no_price": excluded,
        "truncated_count": sum(1 for fr in raw.values() if fr["label_truncated"]),
    }
