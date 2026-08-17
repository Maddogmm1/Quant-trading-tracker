"""
Phase 5 overnight/intraday return decomposition
(PHASE5_OVERNIGHT_GAP_SPECIFICATION.md section 2). A new, small module --
deliberately NOT built on top of src/ml/targets.py's cross-sectional
excess-return machinery, since Phase 5's primary hypothesis is a
distributional claim about a return *component* for a single aggregate
time series, not a cross-sectional ranking claim (see the spec's section
0.4 DELETE-from-consideration entry, and section 3.4's real finding that
cross-sectional breadth is only ~2.3 effective names/day -- far too
little to support a per-security ranking approach at V1).

    overnight_i(t) = ln(open_i(t) / close_i(t-1))
    intraday_i(t)  = ln(close_i(t) / open_i(t))

Both computed on total_return prices exclusively, both endpoints exactly
one trading session apart -- well inside Phase 4's "ratio-only-within-a-
bounded-window" rule (PHASE4_SPECIFICATION.md section 6/8.4): the
future-dividend back-restatement constant compute_total_return() applies
is shared by both endpoints of a one-session window and cancels exactly,
the same way it does for a 21-session momentum window.

Every function here returns None, never a fabricated value, when either
endpoint is unresolvable -- same discipline as src/ml/targets.py and
src/ml/features.py. Nothing here uses a "nearest available" fallback the
way targets.py's multi-month forward-return resolution does: an
overnight return anchored to a stale open several sessions later would
silently become a multi-day return mislabeled as an overnight one, so a
missing exact-date open/close simply excludes that (security, date) row
-- counted by the caller, never imputed.
"""
import math


def open_on_exact_date(conn, security_id, date, adj_type="total_return"):
    """The exact-date open price for security_id on `date`, or None if no
    price row exists for that exact date. Deliberately NOT a
    nearest-available lookup (contrast with
    src.ml.features.latest_price_at_or_before, which is designed for
    lookback-window features where "nearest available" is the right
    behaviour) -- an overnight return needs the literal session's own
    open, or nothing."""
    row = conn.execute(
        "SELECT open FROM prices WHERE security_id=? AND adj_type=? AND date=?",
        (security_id, adj_type, date),
    ).fetchone()
    if not row or row["open"] is None or row["open"] <= 0:
        return None
    return row["open"]


def close_on_exact_date(conn, security_id, date, adj_type="total_return"):
    """Same discipline as open_on_exact_date, for close. Used here instead
    of src.ml.features.latest_price_at_or_before for symmetry -- both
    endpoints of overnight_i(t) must be the literal session's own price,
    not a lookback fallback."""
    row = conn.execute(
        "SELECT close FROM prices WHERE security_id=? AND adj_type=? AND date=?",
        (security_id, adj_type, date),
    ).fetchone()
    if not row or row["close"] is None or row["close"] <= 0:
        return None
    return row["close"]


def previous_trading_session(conn, date, adj_type="total_return"):
    """The market-wide most recent trading session strictly before `date`
    -- the latest date, across every security, on which any security
    recorded a price. Mirrors src.backtest.execution.next_market_session's
    market-wide resolution logic exactly, but looking backward instead of
    forward, for the identical reason that function gives: a single
    security's own data gap must not silently determine which session
    counts as "yesterday" for the whole panel. Used to resolve t-1 for
    overnight_i(t) without a literal calendar-day subtraction (weekends/
    holidays are skipped by construction, matching every other date
    primitive in this codebase)."""
    row = conn.execute(
        "SELECT MAX(date) d FROM prices WHERE adj_type=? AND date<?",
        (adj_type, date),
    ).fetchone()
    return row["d"] if row and row["d"] else None


def overnight_return(conn, security_id, date, adj_type="total_return"):
    """overnight_i(t) = ln(open_i(t) / close_i(t-1)), t-1 resolved via
    previous_trading_session (market-wide calendar, not this security's
    own history). Returns None if either endpoint, or the previous
    session itself, is unresolvable -- never a fabricated value."""
    prev_date = previous_trading_session(conn, date, adj_type)
    if prev_date is None:
        return None
    open_t = open_on_exact_date(conn, security_id, date, adj_type)
    close_prev = close_on_exact_date(conn, security_id, prev_date, adj_type)
    if open_t is None or close_prev is None:
        return None
    return math.log(open_t / close_prev)


def intraday_return(conn, security_id, date, adj_type="total_return"):
    """intraday_i(t) = ln(close_i(t) / open_i(t)), same-session only -- no
    previous-session resolution needed."""
    open_t = open_on_exact_date(conn, security_id, date, adj_type)
    close_t = close_on_exact_date(conn, security_id, date, adj_type)
    if open_t is None or close_t is None:
        return None
    return math.log(close_t / open_t)


def daily_decomposition(conn, security_id, date, adj_type="total_return"):
    """Both components together, plus the additivity identity check
    overnight + intraday == ln(close_t/close_{t-1}) (the full daily total
    return, PHASE5_OVERNIGHT_GAP_SPECIFICATION.md section 2.1) -- computed
    independently here as a genuine consistency check, not assumed to
    hold. Returns None if either component is unresolvable (no partial
    result is ever returned)."""
    prev_date = previous_trading_session(conn, date, adj_type)
    if prev_date is None:
        return None
    overnight = overnight_return(conn, security_id, date, adj_type)
    intraday = intraday_return(conn, security_id, date, adj_type)
    if overnight is None or intraday is None:
        return None
    close_prev = close_on_exact_date(conn, security_id, prev_date, adj_type)
    close_t = close_on_exact_date(conn, security_id, date, adj_type)
    daily_total = math.log(close_t / close_prev) if close_prev and close_t else None
    return {
        "date": date,
        "prev_date": prev_date,
        "overnight": overnight,
        "intraday": intraday,
        "daily_total_return": daily_total,
        "decomposition_identity_ok": (
            abs((overnight + intraday) - daily_total) < 1e-9
            if daily_total is not None else None
        ),
    }


def _all_dates_with_price_data(conn, adj_type):
    """The full market-wide calendar of dates that have at least one price
    row for `adj_type`, sorted ascending. One query, run once per
    proxy_series_for_dates() call -- this is what makes the batched
    previous-session resolution below possible without repeating a
    MAX(date)-WHERE-date< query per (security, date) pair."""
    rows = conn.execute(
        "SELECT DISTINCT date FROM prices WHERE adj_type=? ORDER BY date",
        (adj_type,),
    ).fetchall()
    return [r["date"] for r in rows]


def _previous_trading_session_cache(conn, dates, adj_type):
    """previous_trading_session(), batched: computes the market-wide
    previous session for every date in `dates` from a single sorted
    calendar fetched once, via binary search, instead of one
    'SELECT MAX(date) WHERE date<?' query per date. Semantics are
    identical to calling previous_trading_session() for each date
    individually -- this is a fetch-strategy substitution only."""
    import bisect
    all_dates = _all_dates_with_price_data(conn, adj_type)
    cache = {}
    for d in dates:
        idx = bisect.bisect_left(all_dates, d)
        cache[d] = all_dates[idx - 1] if idx > 0 else None
    return cache


def _bulk_open_close(conn, dates, adj_type):
    """(security_id, open, close) for every security with a price row on
    any date in `dates`, keyed {date: {security_id: (open, close)}}.
    Fetched via a small number of chunked 'date IN (...)' queries instead
    of one query per (security_id, date) pair -- the same open/close
    values open_on_exact_date()/close_on_exact_date() would return
    individually, just fetched in bulk. Chunked to stay well under
    SQLite's default host-parameter limit."""
    unique_dates = sorted(set(dates))
    out = {d: {} for d in unique_dates}
    if not unique_dates:
        return out
    CHUNK = 500
    for i in range(0, len(unique_dates), CHUNK):
        chunk = unique_dates[i:i + CHUNK]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT security_id, date, open, close FROM prices "
            f"WHERE adj_type=? AND date IN ({placeholders})",
            (adj_type, *chunk),
        ).fetchall()
        for r in rows:
            out[r["date"]][r["security_id"]] = (r["open"], r["close"])
    return out


def proxy_series_for_dates(conn, eligible_security_ids_by_date, dates, adj_type="total_return"):
    """Builds the equal-weighted overnight/intraday proxy series across
    `dates`, for whichever securities are eligible on each date (passed in
    by the caller as {date: [security_id, ...]} -- mirroring
    src.ml.targets.compute_labels_for_universe's convention of never
    recomputing eligibility inside this module; ELIG(t) is always
    supplied, never re-derived here).

    Returns {date: {"overnight_proxy", "intraday_proxy", "n_securities",
    "n_missing"}}. A date's proxy is None (not zero, not silently
    skipped) if no security in its eligible set had a resolvable
    decomposition that day.

    Performance note: this function is the one place in the module that
    fans out to every (security, date) pair in a caller-supplied universe
    -- potentially hundreds of securities times hundreds of dates. Rather
    than call daily_decomposition() per pair (which issues ~8 individual
    SQL queries each, several of them a full-table-scanning
    'MAX(date) WHERE date<?' with no usable index), it fetches the
    previous-session calendar once and the needed open/close prices in a
    handful of bulk queries, then computes the same
    overnight_i(t)=ln(open_i(t)/close_i(t-1)),
    intraday_i(t)=ln(close_i(t)/open_i(t)) arithmetic, with the same
    exact-date/no-fallback/positive-price validity rules as
    open_on_exact_date()/close_on_exact_date()/previous_trading_session().
    Single-pair callers (tests, spot checks) should keep using
    daily_decomposition() directly -- that function is unchanged."""
    prev_session_by_date = _previous_trading_session_cache(conn, dates, adj_type)
    needed_dates = set(dates) | {d for d in prev_session_by_date.values() if d is not None}
    price_cache = _bulk_open_close(conn, needed_dates, adj_type)

    out = {}
    for date in dates:
        eligible = eligible_security_ids_by_date.get(date, [])
        prev_date = prev_session_by_date.get(date)
        overnight_vals, intraday_vals = [], []
        missing = 0
        for sid in eligible:
            if prev_date is None:
                missing += 1
                continue
            open_t, close_t = price_cache.get(date, {}).get(sid, (None, None))
            _, close_prev = price_cache.get(prev_date, {}).get(sid, (None, None))
            if (
                open_t is None or open_t <= 0
                or close_t is None or close_t <= 0
                or close_prev is None or close_prev <= 0
            ):
                missing += 1
                continue
            overnight_vals.append(math.log(open_t / close_prev))
            intraday_vals.append(math.log(close_t / open_t))
        out[date] = {
            "overnight_proxy": sum(overnight_vals) / len(overnight_vals) if overnight_vals else None,
            "intraday_proxy": sum(intraday_vals) / len(intraday_vals) if intraday_vals else None,
            "n_securities": len(overnight_vals),
            "n_missing": missing,
        }
    return out
