"""
Computes a split-adjusted price series from raw prices + recorded splits.

This is the one place in the pipeline that deletes and regenerates rows
rather than only appending. That's fine here because 'split_adjusted' rows
are a derived, deterministic transform of raw prices + corporate_actions,
not a second measurement of reality. The "never silently overwrite data"
rule applies to source data (prices, membership, corporate actions
themselves); a derived series should be fully recomputed whenever its
inputs change, or it goes stale silently.

Standard backward-adjustment convention:
  - For any date t before a split's ex_date, adjusted_price(t) = raw_price(t) / ratio
  - Volume is scaled the opposite direction: adjusted_volume(t) = raw_volume(t) * ratio
  - Multiple splits compose multiplicatively.
  - Reverse splits use the same formula with ratio < 1 (e.g. 1-for-10 = ratio 0.1).
"""
from src.database.db import now_iso

# Common genuine whole-share-count split ratios (and their reverse-split
# reciprocals). A ratio close to one of these is very likely a real split.
# A ratio close to 1.0 but not matching this list (e.g. 1.017, 1.046, 1.054,
# 1.324) is more likely a spinoff-driven residual price-adjustment factor
# that yfinance's split field picked up as a side effect, rather than a
# genuine share-count split. Found by inspecting real data: LEN/PFE/IBM/T
# all showed small non-round ratios on dates matching documented real
# spinoffs (Viatris from Pfizer, Kyndryl from IBM, WarnerMedia from AT&T).
_GENUINE_SPLIT_RATIOS = [1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 50, 100]
_GENUINE_SPLIT_TOLERANCE = 0.03  # 3% -- covers e.g. GOOGL's 1.998 (true 2:1)


def classify_split_ratio(ratio):
    """
    Returns 'genuine' if `ratio` (or its reciprocal, for reverse splits)
    is close to a common whole-share-count split ratio, else
    'likely_spinoff_artifact'. Not a certainty claim -- a heuristic to
    route ambiguous cases to human review instead of applying them blindly.
    """
    candidates = _GENUINE_SPLIT_RATIOS + [1 / r for r in _GENUINE_SPLIT_RATIOS]
    for c in candidates:
        if abs(ratio - c) / c <= _GENUINE_SPLIT_TOLERANCE:
            return "genuine"
    return "likely_spinoff_artifact"


def compute_split_adjusted(conn, security_id):
    raw_rows = conn.execute(
        "SELECT * FROM prices WHERE security_id=? AND adj_type='raw' ORDER BY date",
        (security_id,),
    ).fetchall()
    if not raw_rows:
        return {"security_id": security_id, "rows_written": 0, "splits_applied": 0}

    # Only apply ratios classified as genuine splits -- artifacts get
    # recorded but excluded from the price adjustment math itself.
    splits = conn.execute(
        """SELECT ex_date, ratio_or_value FROM corporate_actions
           WHERE security_id=? AND action_type IN ('split','reverse_split')
             AND corporate_action_quality != 'likely_spinoff_artifact'
           ORDER BY ex_date""",
        (security_id,),
    ).fetchall()

    source_id = raw_rows[0]["source_id"]  # derived series inherits provenance of its raw input

    # Derived data: safe to delete-and-regenerate.
    conn.execute("DELETE FROM prices WHERE security_id=? AND adj_type='split_adjusted'", (security_id,))

    written = 0
    for row in raw_rows:
        date_ = row["date"]
        price_factor = 1.0
        vol_factor = 1.0
        for s in splits:
            if date_ < s["ex_date"] and s["ratio_or_value"]:
                ratio = s["ratio_or_value"]
                price_factor /= ratio
                vol_factor *= ratio

        conn.execute(
            """INSERT INTO prices
               (security_id, date, open, high, low, close, volume, adj_type,
                source_id, price_data_quality, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (security_id, date_,
             row["open"] * price_factor if row["open"] is not None else None,
             row["high"] * price_factor if row["high"] is not None else None,
             row["low"] * price_factor if row["low"] is not None else None,
             row["close"] * price_factor if row["close"] is not None else None,
             row["volume"] * vol_factor if row["volume"] is not None else None,
             "split_adjusted", source_id, "derived", now_iso()),
        )
        written += 1

    conn.commit()
    return {"security_id": security_id, "rows_written": written, "splits_applied": len(splits)}


def compute_split_adjusted_for_all(conn):
    """Convenience: recompute split-adjusted series for every security that
    has raw price data. Called after any price or corporate-action ingest."""
    sec_ids = [r["security_id"] for r in conn.execute("SELECT DISTINCT security_id FROM prices WHERE adj_type='raw'").fetchall()]
    results = []
    for sec_id in sec_ids:
        results.append(compute_split_adjusted(conn, sec_id))
    return results


def compute_total_return(conn, security_id):
    """
    Computes a dividend-adjusted (total-return) price series, built on top
    of the split-adjusted series so it accounts for both splits and
    dividends -- matches the "Adj Close" convention used by Yahoo/CRSP.

    Standard back-adjustment: for each dividend of amount D with ex-date
    ex, and P = the split-adjusted close on the trading day immediately
    before ex, all split-adjusted prices at dates t < ex get multiplied by
    (1 - D/P). Multiple dividends compound multiplicatively, most-recent
    first -- same approach as compute_split_adjusted(), with a per-dividend
    factor instead of a fixed ratio.

    Requires compute_split_adjusted() to have already been run for this
    security, since it uses that output as the base series. Also derived
    data, so safe to delete-and-regenerate.
    """
    adj_rows = conn.execute(
        "SELECT * FROM prices WHERE security_id=? AND adj_type='split_adjusted' ORDER BY date",
        (security_id,),
    ).fetchall()
    if not adj_rows:
        return {"security_id": security_id, "rows_written": 0, "dividends_applied": 0}

    dividends = conn.execute(
        """SELECT ex_date, ratio_or_value FROM corporate_actions
           WHERE security_id=? AND action_type IN ('dividend','special_dividend')
           ORDER BY ex_date""",
        (security_id,),
    ).fetchall()

    if not dividends:
        # No dividends recorded -- total_return equals split_adjusted. Still
        # write it as a real copy so downstream code can always rely on
        # adj_type='total_return' existing.
        dividends = []

    price_by_date = {r["date"]: r["close"] for r in adj_rows}
    dates_sorted = sorted(price_by_date.keys())

    # Precompute each dividend's per-event factor using the split-adjusted
    # close on the closest available trading day BEFORE its ex_date.
    div_factors = []  # list of (ex_date, factor)
    for d in dividends:
        ex_date = d["ex_date"]
        amount = d["ratio_or_value"]
        if not amount or amount <= 0:
            continue
        prior_dates = [dt for dt in dates_sorted if dt < ex_date]
        if not prior_dates:
            continue  # dividend predates our price coverage -- can't anchor it, skip rather than guess
        prior_close = price_by_date[prior_dates[-1]]
        if prior_close <= 0:
            continue
        factor = max(1 - (amount / prior_close), 0.01)  # floor to avoid a pathological/negative factor
        div_factors.append((ex_date, factor))

    source_id = adj_rows[0]["source_id"]
    conn.execute("DELETE FROM prices WHERE security_id=? AND adj_type='total_return'", (security_id,))

    written = 0
    for row in adj_rows:
        date_ = row["date"]
        cum_factor = 1.0
        for ex_date, factor in div_factors:
            if date_ < ex_date:
                cum_factor *= factor

        conn.execute(
            """INSERT INTO prices
               (security_id, date, open, high, low, close, volume, adj_type,
                source_id, price_data_quality, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (security_id, date_,
             row["open"] * cum_factor if row["open"] is not None else None,
             row["high"] * cum_factor if row["high"] is not None else None,
             row["low"] * cum_factor if row["low"] is not None else None,
             row["close"] * cum_factor if row["close"] is not None else None,
             row["volume"] / cum_factor if row["volume"] is not None and cum_factor else row["volume"],
             "total_return", source_id, "derived", now_iso()),
        )
        written += 1

    conn.commit()
    return {"security_id": security_id, "rows_written": written, "dividends_applied": len(div_factors)}


def compute_total_return_for_all(conn):
    """Convenience: recompute total-return series for every security with
    a split-adjusted series already computed. Run AFTER compute_split_adjusted_for_all()."""
    sec_ids = [r["security_id"] for r in conn.execute("SELECT DISTINCT security_id FROM prices WHERE adj_type='split_adjusted'").fetchall()]
    results = []
    for sec_id in sec_ids:
        results.append(compute_total_return(conn, sec_id))
    return results
