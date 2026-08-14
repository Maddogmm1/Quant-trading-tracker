"""
One-off investigation: characterize the 1,227 bad-OHLC observations found
in Stage 3, focused on 5 named tickers (TNB, CFC, RYC, PZE, PBG), to see
whether the problem is concentrated in a few legacy tickers and what kind
of problem each one is.

For each ticker this pulls every flagged raw row (not just the 30-row
report sample), looks for nearby corporate_actions (splits/dividends) that
might explain a one-off adjustment artifact, and does a live yfinance
re-fetch + .info lookup as an independent cross-check against what's
already in the database (catches stale-cache-style issues and shows what
company the ticker currently/historically maps to).

Read-only: does not correct, delete, or re-flag anything.

Run: python3 investigate_bad_ohlc.py
Writes: BAD_OHLC_INVESTIGATION.md and BAD_OHLC_INVESTIGATION.json
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from run_stage_ingestion import DB_PATH
from src.database.db import get_connection

TARGET_TICKERS = ["TNB", "CFC", "RYC", "PZE", "PBG"]

BAD_OHLC_CONDITION = (
    "high < low OR high < open OR high < close OR low > open OR low > close "
    "OR open <= 0 OR close <= 0"
)


def investigate_ticker(conn, ticker):
    sec = conn.execute(
        "SELECT * FROM securities WHERE primary_ticker=?", (ticker,)
    ).fetchone()
    result = {"ticker": ticker}
    if not sec:
        result["error"] = "not found in securities table"
        return result

    sec_id = sec["security_id"]
    result["security_id"] = sec_id
    result["db_name"] = sec["name"]
    result["identifier_quality"] = sec["identifier_quality"]
    result["active_flag"] = sec["active_flag"]
    result["delisted_date"] = sec["delisted_date"]
    result["delisting_reason"] = sec["delisting_reason"]

    total_raw_rows = conn.execute(
        "SELECT COUNT(*) c FROM prices WHERE security_id=? AND adj_type='raw'", (sec_id,)
    ).fetchone()["c"]
    result["total_raw_rows"] = total_raw_rows

    bad_rows = conn.execute(
        f"""SELECT date, open, high, low, close, volume FROM prices
            WHERE security_id=? AND adj_type='raw' AND ({BAD_OHLC_CONDITION})
            ORDER BY date""",
        (sec_id,),
    ).fetchall()
    result["bad_row_count"] = len(bad_rows)
    result["bad_row_pct_of_total"] = round(len(bad_rows) / total_raw_rows * 100, 2) if total_raw_rows else None
    result["bad_rows"] = [dict(r) for r in bad_rows]

    if bad_rows:
        dates = [r["date"] for r in bad_rows]
        result["first_bad_date"] = dates[0]
        result["last_bad_date"] = dates[-1]
        # crude "is this one continuous stretch or scattered?" signal
        result["distinct_bad_dates"] = len(set(dates))

        # Look for a frozen/stale field: does one OHLC field repeat an
        # identical value across many DIFFERENT dates while others move?
        for field in ("open", "high", "low", "close"):
            vals = [r[field] for r in bad_rows]
            most_common = max(set(vals), key=vals.count) if vals else None
            freq = vals.count(most_common) if most_common is not None else 0
            if freq >= 5 and freq / len(vals) >= 0.3:
                result.setdefault("suspected_frozen_fields", []).append(
                    {"field": field, "value": most_common, "occurrences": freq, "of": len(vals)}
                )

    # Nearby corporate actions (any type) within +/- 10 days of each bad date
    nearby_actions = []
    for r in bad_rows:
        rows = conn.execute(
            """SELECT action_type, ex_date, ratio_or_value, corporate_action_quality FROM corporate_actions
               WHERE security_id=? AND date(ex_date) BETWEEN date(?, '-10 days') AND date(?, '+10 days')""",
            (sec_id, r["date"], r["date"]),
        ).fetchall()
        for a in rows:
            nearby_actions.append({"bad_date": r["date"], **dict(a)})
    # de-dupe
    seen = set()
    dedup_actions = []
    for a in nearby_actions:
        key = (a["bad_date"], a["action_type"], a["ex_date"])
        if key in seen:
            continue
        seen.add(key)
        dedup_actions.append(a)
    result["nearby_corporate_actions"] = dedup_actions

    # Identity review queue flags for this ticker
    identity_flags = conn.execute(
        "SELECT gap_days, period_1_start, period_1_end, period_2_start, period_2_end, resolved "
        "FROM identity_review_queue WHERE ticker=?", (ticker,)
    ).fetchall()
    result["identity_review_flags"] = [dict(r) for r in identity_flags]

    return result


def yfinance_cross_check(ticker, sample_dates):
    """Independent live re-fetch as a cross-check. Best-effort -- network
    issues here should not crash the whole investigation."""
    out = {"ticker": ticker}
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        try:
            info = tk.info
            out["current_shortName"] = info.get("shortName")
            out["current_longName"] = info.get("longName")
            out["current_exchange"] = info.get("exchange")
            out["current_quoteType"] = info.get("quoteType")
        except Exception as e:
            out["info_error"] = str(e)

        if sample_dates:
            start = min(sample_dates)
            end = max(sample_dates)
            import datetime
            end_plus = (datetime.date.fromisoformat(end) + datetime.timedelta(days=3)).isoformat()
            hist = tk.history(start=start, end=end_plus, auto_adjust=False)
            out["refetch_rows"] = len(hist)
            out["refetch_sample"] = [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "open": round(float(row["Open"]), 4),
                    "high": round(float(row["High"]), 4),
                    "low": round(float(row["Low"]), 4),
                    "close": round(float(row["Close"]), 4),
                }
                for idx, row in hist.head(10).iterrows()
            ]
    except ImportError:
        out["error"] = "yfinance not installed -- run: pip install yfinance"
    except Exception as e:
        out["fetch_error"] = str(e)
    return out


if __name__ == "__main__":
    conn = get_connection(DB_PATH)
    print(f"Using database: {DB_PATH}\n")

    all_results = {}
    for ticker in TARGET_TICKERS:
        print(f"Investigating {ticker}...")
        r = investigate_ticker(conn, ticker)
        bad_dates = [row["date"] for row in r.get("bad_rows", [])]
        r["yfinance_cross_check"] = yfinance_cross_check(ticker, bad_dates[:5] if bad_dates else [])
        all_results[ticker] = r

    conn.close()

    with open("BAD_OHLC_INVESTIGATION.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    lines = ["# Bad-OHLC Investigation: TNB / CFC / RYC / PZE / PBG\n"]
    for ticker, r in all_results.items():
        lines.append(f"## {ticker}")
        if r.get("error"):
            lines.append(f"- ERROR: {r['error']}\n")
            continue
        lines.append(f"- DB name: {r.get('db_name')}")
        lines.append(f"- identifier_quality: {r.get('identifier_quality')}, active_flag: {r.get('active_flag')}")
        lines.append(f"- delisted_date: {r.get('delisted_date')} ({r.get('delisting_reason')})")
        lines.append(f"- total_raw_rows: {r.get('total_raw_rows')}, bad_row_count: {r.get('bad_row_count')} "
                      f"({r.get('bad_row_pct_of_total')}%)")
        if r.get("bad_rows"):
            lines.append(f"- bad date range: {r.get('first_bad_date')} to {r.get('last_bad_date')} "
                         f"({r.get('distinct_bad_dates')} distinct dates)")
        if r.get("suspected_frozen_fields"):
            lines.append(f"- SUSPECTED FROZEN FIELD(S): {r['suspected_frozen_fields']}")
        if r.get("nearby_corporate_actions"):
            lines.append(f"- nearby corporate actions: {r['nearby_corporate_actions']}")
        if r.get("identity_review_flags"):
            lines.append(f"- identity_review_queue flags: {r['identity_review_flags']}")
        yc = r.get("yfinance_cross_check", {})
        lines.append(f"- yfinance current identity: shortName={yc.get('current_shortName')}, "
                     f"longName={yc.get('current_longName')}, exchange={yc.get('current_exchange')}, "
                     f"quoteType={yc.get('current_quoteType')}")
        if yc.get("refetch_sample"):
            lines.append(f"- yfinance re-fetch sample (independent, live): {yc['refetch_sample']}")
        if yc.get("fetch_error") or yc.get("info_error") or yc.get("error"):
            lines.append(f"- yfinance cross-check issues: {yc.get('fetch_error') or yc.get('info_error') or yc.get('error')}")
        lines.append("")

    with open("BAD_OHLC_INVESTIGATION.md", "w") as f:
        f.write("\n".join(lines))

    print("\n\nDone. Wrote BAD_OHLC_INVESTIGATION.md and BAD_OHLC_INVESTIGATION.json.")
