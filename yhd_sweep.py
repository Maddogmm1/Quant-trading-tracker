"""
Stage 3 close-out check: full-universe YHD / placeholder-identity sweep.

Applies the same mechanical signal that flagged TNB and CFC during the
bad-OHLC investigation -- current yfinance identity resolves to
exchange=='YHD' (Yahoo's internal code for archived/delisted-ticker
records) and/or a purely numeric "shortName" (a placeholder, not a real
company name) -- across all 1,205+ securities in the Stage 3 universe.

Deliberately mechanical, not investigative: no per-ticker web research, no
manual classification of why a given ticker looks this way. For every
flagged ticker this only cross-references data already in the database
(price coverage, membership windows, identity_review_queue, bad-OHLC
flags) so every existing signal is visible in one place. Read-only.

Resumable: writes incrementally to YHD_SWEEP_RAW.json (one yfinance
lookup per ticker) and skips tickers already checked on a re-run, same
resumability pattern as the stage scripts -- this touches ~1,205 tickers
and yfinance .info calls are not cheap, so an interruption should not
mean starting over.

Run: python3 yhd_sweep.py
Writes: YHD_SWEEP_RAW.json (checkpoint), YHD_SWEEP.md, YHD_SWEEP.json
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

from run_stage_ingestion import DB_PATH, PRICE_START, PRICE_END
from src.database.db import get_connection

RAW_CHECKPOINT_PATH = "YHD_SWEEP_RAW.json"


def load_checkpoint():
    if os.path.exists(RAW_CHECKPOINT_PATH):
        with open(RAW_CHECKPOINT_PATH) as f:
            return json.load(f)
    return {}


def save_checkpoint(data):
    with open(RAW_CHECKPOINT_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def yfinance_identity(ticker):
    import yfinance as yf
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        return {
            "exchange": info.get("exchange"),
            "quoteType": info.get("quoteType"),
            "shortName": info.get("shortName"),
            "longName": info.get("longName"),
            "error": None,
        }
    except Exception as e:
        return {"exchange": None, "quoteType": None, "shortName": None, "longName": None, "error": str(e)}


def is_placeholder_shortname(name):
    if name is None:
        return False
    return name.strip().isdigit()


def run_sweep(conn):
    securities = conn.execute(
        "SELECT security_id, primary_ticker FROM securities ORDER BY primary_ticker"
    ).fetchall()
    total_securities = len(securities)
    print(f"Total securities to check: {total_securities}")

    raw = load_checkpoint()
    already_done = len(raw)
    if already_done:
        print(f"Resuming: {already_done} already checked in a prior run.")

    for i, sec in enumerate(securities):
        ticker = sec["primary_ticker"]
        if ticker in raw:
            continue
        identity = yfinance_identity(ticker)
        raw[ticker] = {"security_id": sec["security_id"], **identity}
        if (i + 1) % 25 == 0 or (i + 1) == total_securities:
            save_checkpoint(raw)
            print(f"  [{i+1}/{total_securities}] checked, last={ticker}")

    save_checkpoint(raw)
    return raw, total_securities


def cross_reference(conn, raw):
    affected = []
    lookup_errors = []
    for ticker, r in raw.items():
        if r.get("error"):
            lookup_errors.append({"ticker": ticker, "error": r["error"]})
            continue

        is_yhd = (r.get("exchange") == "YHD")
        is_placeholder = is_placeholder_shortname(r.get("shortName"))
        if not (is_yhd or is_placeholder):
            continue

        sec_id = r["security_id"]
        price_count = conn.execute(
            "SELECT COUNT(*) c FROM prices WHERE security_id=? AND adj_type='raw'", (sec_id,)
        ).fetchone()["c"]
        membership_rows = conn.execute(
            "SELECT effective_date, removal_date FROM index_membership WHERE raw_ticker=?", (ticker,)
        ).fetchall()
        identity_flags = conn.execute(
            "SELECT COUNT(*) c FROM identity_review_queue WHERE ticker=? AND resolved=0", (ticker,)
        ).fetchone()["c"]
        bad_ohlc = conn.execute(
            """SELECT COUNT(*) c FROM prices
               WHERE security_id=? AND adj_type='raw' AND price_data_quality='suspicious'""",
            (sec_id,),
        ).fetchone()["c"]

        relevant = False
        for m in membership_rows:
            eff = m["effective_date"]
            rem = m["removal_date"] or "9999-12-31"
            if eff <= PRICE_END and rem >= PRICE_START:
                relevant = True
                break

        affected.append({
            "ticker": ticker,
            "security_id": sec_id,
            "exchange": r.get("exchange"),
            "quoteType": r.get("quoteType"),
            "shortName": r.get("shortName"),
            "is_yhd": is_yhd,
            "is_placeholder_shortname": is_placeholder,
            "has_price_data": price_count > 0,
            "price_row_count": price_count,
            "has_membership_data": len(membership_rows) > 0,
            "membership_row_count": len(membership_rows),
            "has_identity_review_flag": identity_flags > 0,
            "bad_ohlc_flagged_count": bad_ohlc,
            "relevant_to_2010_2023_backtest_period": relevant,
        })

    return affected, lookup_errors


if __name__ == "__main__":
    conn = get_connection(DB_PATH)
    print(f"Using database: {DB_PATH}\n")

    raw, total_securities = run_sweep(conn)
    print("\nSweep fetch complete. Cross-referencing against database...\n")

    affected, lookup_errors = cross_reference(conn, raw)

    total_bad_ohlc_system_wide = conn.execute(
        "SELECT COUNT(*) c FROM prices WHERE adj_type='raw' AND price_data_quality='suspicious'"
    ).fetchone()["c"]

    affected_with_price = sum(1 for a in affected if a["has_price_data"])
    affected_with_membership = sum(1 for a in affected if a["has_membership_data"])
    affected_with_identity_flag = sum(1 for a in affected if a["has_identity_review_flag"])
    affected_with_bad_ohlc = sum(1 for a in affected if a["bad_ohlc_flagged_count"] > 0)
    affected_relevant = sum(1 for a in affected if a["relevant_to_2010_2023_backtest_period"])
    bad_ohlc_from_affected = sum(a["bad_ohlc_flagged_count"] for a in affected)

    affected_sorted = sorted(affected, key=lambda a: -a["bad_ohlc_flagged_count"])

    summary = {
        "total_securities_checked": total_securities,
        "lookup_errors": len(lookup_errors),
        "total_affected_yhd_or_placeholder": len(affected),
        "pct_of_universe_affected": round(len(affected) / total_securities * 100, 2) if total_securities else 0,
        "affected_with_price_data": affected_with_price,
        "affected_with_membership_data": affected_with_membership,
        "affected_with_identity_review_flag": affected_with_identity_flag,
        "affected_with_bad_ohlc_flags": affected_with_bad_ohlc,
        "affected_relevant_to_2010_2023_backtest_period": affected_relevant,
        "total_bad_ohlc_flagged_system_wide": total_bad_ohlc_system_wide,
        "bad_ohlc_flagged_contributed_by_affected_tickers": bad_ohlc_from_affected,
        "pct_of_system_bad_ohlc_from_affected_tickers": (
            round(bad_ohlc_from_affected / total_bad_ohlc_system_wide * 100, 2)
            if total_bad_ohlc_system_wide else 0
        ),
        "affected_tickers": affected_sorted,
        "lookup_error_detail": lookup_errors,
    }

    with open("YHD_SWEEP.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    lines = ["# Full-Universe YHD / Placeholder-Identity Sweep\n"]
    lines.append(f"- total_securities_checked: {summary['total_securities_checked']}")
    lines.append(f"- lookup_errors: {summary['lookup_errors']}")
    lines.append(f"- total_affected (exchange==YHD or numeric shortName): {summary['total_affected_yhd_or_placeholder']} "
                 f"({summary['pct_of_universe_affected']}% of universe)")
    lines.append(f"- affected_with_price_data: {summary['affected_with_price_data']}")
    lines.append(f"- affected_with_membership_data: {summary['affected_with_membership_data']}")
    lines.append(f"- affected_with_identity_review_flag: {summary['affected_with_identity_review_flag']}")
    lines.append(f"- affected_with_bad_ohlc_flags: {summary['affected_with_bad_ohlc_flags']}")
    lines.append(f"- affected_relevant_to_2010_2023_backtest_period: {summary['affected_relevant_to_2010_2023_backtest_period']}")
    lines.append(f"- total_bad_ohlc_flagged_system_wide: {summary['total_bad_ohlc_flagged_system_wide']}")
    lines.append(f"- bad_ohlc_flagged_contributed_by_affected_tickers: {summary['bad_ohlc_flagged_contributed_by_affected_tickers']} "
                 f"({summary['pct_of_system_bad_ohlc_from_affected_tickers']}% of all flagged rows)")
    lines.append("\n## Affected tickers (sorted by bad-OHLC row count, descending)\n")
    for a in affected_sorted:
        lines.append(
            f"- {a['ticker']} (security_id={a['security_id']}): exchange={a['exchange']}, "
            f"quoteType={a['quoteType']}, shortName={a['shortName']} | "
            f"price_data={a['has_price_data']} ({a['price_row_count']} rows), "
            f"membership_data={a['has_membership_data']} ({a['membership_row_count']} claims), "
            f"identity_flag={a['has_identity_review_flag']}, "
            f"bad_ohlc={a['bad_ohlc_flagged_count']}, "
            f"relevant_2010_2023={a['relevant_to_2010_2023_backtest_period']}"
        )
    if lookup_errors:
        lines.append(f"\n## Lookup errors ({len(lookup_errors)})\n")
        for e in lookup_errors[:50]:
            lines.append(f"- {e['ticker']}: {e['error']}")

    with open("YHD_SWEEP.md", "w") as f:
        f.write("\n".join(lines))

    conn.close()
    print(f"\nDone. {len(affected)} affected out of {total_securities} checked "
          f"({summary['pct_of_universe_affected']}%).")
    print("Wrote YHD_SWEEP.md and YHD_SWEEP.json.")
