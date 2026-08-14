"""
Point-in-time-safe feature computation for the Phase 4 model (see
PHASE4_SPECIFICATION.md section 6 for the full feature list).

Two invariants hold across every function here. First, inherited from
Phase 3's check_point_in_time_availability(): every query filters on
`date <= as_of_date`, so nothing in this module can see a price dated
after as_of_date (tested in tests/test_ml_features_leakage.py). Second,
every feature is expressed as a ratio or return of total_return prices
within its own bounded lookback window, never a raw total_return level
used in isolation. That's because compute_total_return()
(src/ingestion/adjustments.py) backward-restates historical total_return
rows using every dividend on record, including ones dated after
as_of_date. That's fine for return ratios, since the future-dividend
multiplicative constant is shared across a bounded window and cancels
exactly, but it makes a raw level unsafe to treat as "the price known at
time t" in isolation. So no function below returns a raw level, only
ratios/returns/rank-eligible derived quantities.

Every function takes (conn, security_id, as_of_date, ...) and returns
None, not a fabricated value, when there isn't enough point-in-time
history to compute it. Callers must handle None rather than treat it as
zero.
"""
import statistics


def _price_n_sessions_before(conn, security_id, as_of_date, n, adj_type="total_return"):
    """The n-th most recent trading session at-or-before as_of_date
    (n=0 -> the most recent session <= as_of_date). Returns (date, close)
    or None if fewer than n+1 sessions exist. The `date <= as_of_date`
    filter is the entire point-in-time guarantee."""
    row = conn.execute(
        "SELECT date, close FROM prices WHERE security_id=? AND adj_type=? AND date<=? "
        "ORDER BY date DESC LIMIT 1 OFFSET ?",
        (security_id, adj_type, as_of_date, n),
    ).fetchone()
    if not row or row["close"] is None:
        return None
    return (row["date"], row["close"])


def _closes_window(conn, security_id, as_of_date, n_sessions, adj_type="total_return"):
    """The n_sessions most recent closes at-or-before as_of_date, oldest
    first. Returns None if fewer than n_sessions exist -- no partial-window
    computation, matching Phase 3's require_full_history convention of
    treating an incomplete lookback as ineligible rather than a smaller
    sample."""
    rows = conn.execute(
        "SELECT close FROM (SELECT date, close FROM prices WHERE security_id=? AND adj_type=? AND date<=? "
        "ORDER BY date DESC LIMIT ?) ORDER BY date ASC",
        (security_id, adj_type, as_of_date, n_sessions),
    ).fetchall()
    if len(rows) < n_sessions:
        return None
    closes = [r["close"] for r in rows]
    if any(c is None for c in closes):
        return None
    return closes


def latest_price_at_or_before(conn, security_id, as_of_date, adj_type="total_return"):
    """The single 'nearest price at-or-before as_of_date' primitive,
    reused by src/ml/targets.py so label construction and feature
    construction share one point-in-time price-resolution rule instead of
    two subtly different ones."""
    return _price_n_sessions_before(conn, security_id, as_of_date, 0, adj_type)


def return_between_offsets(conn, security_id, as_of_date, end_offset_sessions, window_sessions,
                            adj_type="total_return"):
    """Return over a window ending end_offset_sessions before as_of_date:
    price(end_offset) / price(end_offset + window_sessions) - 1. Both
    endpoints are resolved via _price_n_sessions_before, so both are
    <= as_of_date. end_offset_sessions=0 gives the return ending "now";
    a nonzero end_offset lets a caller reach further back, e.g. for
    momentum_acceleration below."""
    end = _price_n_sessions_before(conn, security_id, as_of_date, end_offset_sessions, adj_type)
    start = _price_n_sessions_before(conn, security_id, as_of_date, end_offset_sessions + window_sessions, adj_type)
    if not end or not start or not start[1]:
        return None
    return end[1] / start[1] - 1.0


# --- 6.1 Price / momentum ------------------------------------------------

def return_1m(conn, security_id, as_of_date):
    return return_between_offsets(conn, security_id, as_of_date, 0, 21)


def return_3m(conn, security_id, as_of_date):
    return return_between_offsets(conn, security_id, as_of_date, 0, 63)


def return_6m(conn, security_id, as_of_date):
    return return_between_offsets(conn, security_id, as_of_date, 0, 126)


def return_12m(conn, security_id, as_of_date):
    return return_between_offsets(conn, security_id, as_of_date, 0, 252)


def momentum_acceleration(conn, security_id, as_of_date):
    """[3M return ending now] - [3M return ending 3M ago]."""
    recent = return_between_offsets(conn, security_id, as_of_date, 0, 63)
    prior = return_between_offsets(conn, security_id, as_of_date, 63, 63)
    if recent is None or prior is None:
        return None
    return recent - prior


def distance_from_200d_ma(conn, security_id, as_of_date):
    """TR(t) / mean(TR(t-199d..t)) - 1 -- a ratio of the anchor price to a
    200-session simple moving average, both from data <= as_of_date."""
    closes = _closes_window(conn, security_id, as_of_date, 200)
    if closes is None:
        return None
    anchor = _price_n_sessions_before(conn, security_id, as_of_date, 0)
    if not anchor:
        return None
    ma = sum(closes) / len(closes)
    if ma == 0:
        return None
    return anchor[1] / ma - 1.0


# --- 6.2 Volatility / risk ------------------------------------------------

def _daily_returns_window(conn, security_id, as_of_date, n_sessions, adj_type="total_return"):
    """n_sessions daily returns ending at as_of_date -- needs n_sessions+1
    closes. Returns None if unavailable."""
    closes = _closes_window(conn, security_id, as_of_date, n_sessions + 1, adj_type)
    if closes is None:
        return None
    out = []
    for i in range(1, len(closes)):
        if not closes[i - 1]:
            return None
        out.append(closes[i] / closes[i - 1] - 1.0)
    return out


def realised_volatility_63d(conn, security_id, as_of_date):
    rets = _daily_returns_window(conn, security_id, as_of_date, 63)
    if rets is None or len(rets) < 2:
        return None
    return statistics.stdev(rets)


def downside_volatility_63d(conn, security_id, as_of_date):
    rets = _daily_returns_window(conn, security_id, as_of_date, 63)
    if rets is None:
        return None
    negative = [r for r in rets if r < 0]
    if len(negative) < 2:
        return None
    return statistics.stdev(negative)


def max_drawdown_252d(conn, security_id, as_of_date):
    closes = _closes_window(conn, security_id, as_of_date, 252)
    if closes is None:
        return None
    peak = closes[0]
    max_dd = 0.0
    for c in closes:
        peak = max(peak, c)
        if peak > 0:
            max_dd = min(max_dd, (c - peak) / peak)
    return max_dd


# --- 6.3 Volume / liquidity ------------------------------------------------
# These deliberately don't reuse apply_universe_filters()'s dollar-volume
# average, which averages over a security's entire history to date rather
# than a true rolling window. These are fresh, genuinely-rolling computations.

def _dollar_volumes_window(conn, security_id, as_of_date, n_sessions, adj_type="total_return"):
    rows = conn.execute(
        "SELECT close, volume FROM (SELECT date, close, volume FROM prices "
        "WHERE security_id=? AND adj_type=? AND date<=? ORDER BY date DESC LIMIT ?) ORDER BY date ASC",
        (security_id, adj_type, as_of_date, n_sessions),
    ).fetchall()
    if len(rows) < n_sessions:
        return None
    vals = []
    for r in rows:
        if r["close"] is None or r["volume"] is None:
            return None
        vals.append(r["close"] * r["volume"])
    return vals


def dollar_volume(conn, security_id, as_of_date):
    """Single-day dollar volume at (or nearest before) as_of_date."""
    row = conn.execute(
        "SELECT close, volume FROM prices WHERE security_id=? AND adj_type='total_return' AND date<=? "
        "ORDER BY date DESC LIMIT 1",
        (security_id, as_of_date),
    ).fetchone()
    if not row or row["close"] is None or row["volume"] is None:
        return None
    return row["close"] * row["volume"]


def rolling_avg_dollar_volume_20d(conn, security_id, as_of_date):
    vals = _dollar_volumes_window(conn, security_id, as_of_date, 20)
    if vals is None:
        return None
    return sum(vals) / len(vals)


def volume_trend_20d_100d(conn, security_id, as_of_date):
    """20d rolling avg dollar volume vs 100d rolling avg dollar volume,
    expressed as a ratio-minus-one so it's scale-free like every other
    feature here."""
    v20 = _dollar_volumes_window(conn, security_id, as_of_date, 20)
    v100 = _dollar_volumes_window(conn, security_id, as_of_date, 100)
    if v20 is None or v100 is None:
        return None
    avg100 = sum(v100) / len(v100)
    if avg100 == 0:
        return None
    avg20 = sum(v20) / len(v20)
    return avg20 / avg100 - 1.0


# --- 6.4 Relative / cross-sectional ------------------------------------------------
# These take a precomputed {security_id: value} map for the eligible
# universe at this as_of_date, rather than re-querying it -- callers
# compute the per-security base feature (e.g. return_3m) for every member
# of ELIG(t) once, then pass that dict in here. Keeps these functions
# pure/cheap and avoids N^2 re-querying across a rebalance date's full
# eligible set.

def cross_sectional_mean(values_by_security):
    """Mean of non-None values in a {security_id: value_or_None} map.
    Returns None if there are no usable values."""
    usable = [v for v in values_by_security.values() if v is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def cross_sectional_percentile_rank(security_id, values_by_security):
    """Percentile rank (0..1) of values_by_security[security_id] among all
    non-None values in the map, INCLUDING itself. Returns None if the
    security's own value is None or there are fewer than 2 usable values
    (a rank of 1 security against itself is meaningless)."""
    own = values_by_security.get(security_id)
    if own is None:
        return None
    usable = [v for v in values_by_security.values() if v is not None]
    if len(usable) < 2:
        return None
    usable_sorted = sorted(usable)
    # average-rank handling for ties, expressed as a fraction in [0, 1]
    below = sum(1 for v in usable_sorted if v < own)
    equal = sum(1 for v in usable_sorted if v == own)
    return (below + 0.5 * equal) / len(usable_sorted)


def return_relative_to_eligible_mean(security_id, returns_by_security):
    """return_i(t, window) - mean_j in ELIG(t)(return_j(t, window)).
    returns_by_security must already be computed only from ELIG(t)
    members, never ELIG(t+h)."""
    own = returns_by_security.get(security_id)
    mean = cross_sectional_mean(returns_by_security)
    if own is None or mean is None:
        return None
    return own - mean


# --- 6.5 Market / regime (derived equal-weighted proxy -- NOT the real S&P 500) ---
# No S&P 500 index-level series is ingested. Every function below is a
# derived proxy built from ELIG(t) member prices, and must always be
# labelled as such wherever it's used or reported.

def proxy_index_trend(returns_by_security):
    """The equal-weighted eligible-universe return over a window -- this
    is Phase 3's equal_weight_sp500 construction, reused rather than
    redefined. Callers pass in the per-security
    {security_id: return_over_window(...)} map for ELIG(t)."""
    return cross_sectional_mean(returns_by_security)


def proxy_index_daily_returns(conn, eligible_security_ids, as_of_date, n_sessions, adj_type="total_return"):
    """Builds the equal-weighted proxy index's own daily return series --
    not the average of members' individual volatilities, which would
    overstate "market" volatility by ignoring diversification. Uses the
    set of trading dates present across eligible_security_ids' own price
    history (date <= as_of_date only), and on each date averages the
    daily return across whichever eligible members have prices on both
    that date and the prior date. Membership is fixed at as_of_date's
    ELIG(t) rather than reconstructed day-by-day through history, which
    is an approximation worth calling out rather than assuming exact."""
    if not eligible_security_ids:
        return None
    placeholders = ",".join("?" * len(eligible_security_ids))
    rows = conn.execute(
        f"SELECT DISTINCT date FROM prices WHERE security_id IN ({placeholders}) AND adj_type=? AND date<=? "
        f"ORDER BY date DESC LIMIT ?",
        (*eligible_security_ids, adj_type, as_of_date, n_sessions + 1),
    ).fetchall()
    dates = sorted(r["date"] for r in rows)
    if len(dates) < 2:
        return None

    price_rows = conn.execute(
        f"SELECT security_id, date, close FROM prices WHERE security_id IN ({placeholders}) AND adj_type=? "
        f"AND date IN ({','.join('?' * len(dates))})",
        (*eligible_security_ids, adj_type, *dates),
    ).fetchall()
    by_date = {}
    for r in price_rows:
        by_date.setdefault(r["date"], {})[r["security_id"]] = r["close"]

    daily_returns = []
    for i in range(1, len(dates)):
        prev_prices = by_date.get(dates[i - 1], {})
        cur_prices = by_date.get(dates[i], {})
        member_returns = [
            cur_prices[sid] / prev_prices[sid] - 1.0
            for sid in eligible_security_ids
            if sid in prev_prices and sid in cur_prices and prev_prices[sid]
        ]
        if member_returns:
            daily_returns.append(sum(member_returns) / len(member_returns))
    return daily_returns if daily_returns else None


def proxy_index_volatility(conn, eligible_security_ids, as_of_date, n_sessions=63, adj_type="total_return"):
    rets = proxy_index_daily_returns(conn, eligible_security_ids, as_of_date, n_sessions, adj_type)
    if rets is None or len(rets) < 2:
        return None
    return statistics.stdev(rets)


def breadth_proxy(distance_from_ma_by_security):
    """Fraction of ELIG(t) trading above their own 200d MA. Callers pass
    in the per-security {security_id: distance_from_200d_ma(...)} map."""
    usable = [v for v in distance_from_ma_by_security.values() if v is not None]
    if not usable:
        return None
    return sum(1 for v in usable if v > 0) / len(usable)


# --- section 6.6: explicit denylist, enforced in code as well as config ---
EXPLICITLY_EXCLUDED_FEATURES = frozenset({
    "identifier_quality", "identity_review_queue_flag", "price_data_quality",
})


def price_history_length_asof(conn, security_id, as_of_date, adj_type="total_return"):
    """Diagnostic-only, never a V1 feature or model input: the number of
    total_return price rows on record for security_id at or before
    as_of_date (point-in-time, same date<=as_of_date invariant as every
    other function in this module).

    This is the Phase 4 analogue of Phase 3's own price-history-length
    diagnostic (run_phase3_diagnostic.py's _randomness_sanity, which
    correlated selection frequency against COUNT(*) of all price rows for
    a security). Phase 3 could use the full, all-time count because it
    was analysing a completed backtest's selections after the fact; here
    a model predicts live on as_of_date, so the all-time count would leak
    future history into what's supposed to be a point-in-time diagnostic.
    Used by run_phase4_trees.py's price-history-length correlation check,
    which guards against a model implicitly learning "more history means
    a more established, safer pick" rather than any real signal."""
    row = conn.execute(
        "SELECT COUNT(*) c FROM prices WHERE security_id=? AND adj_type=? AND date<=?",
        (security_id, adj_type, as_of_date),
    ).fetchone()
    return row["c"]
