"""
Run Phase 1 against REAL yfinance data instead of the synthetic placeholder.

Requires network access to Yahoo Finance.

Setup:
    pip install yfinance
    cd quant_trader
    python3 run_phase1_real_data.py

This reuses every piece of the Phase 1 pipeline unchanged (schema, security
resolution, membership ingestion, validation) -- the only thing that changes
is which PriceSourceAdapter gets passed in. That's the point of having built
it as a swappable interface.

Expected differences from the synthetic run, worth watching for:
  - MON, ABMD, AABA, AAMRQ, ABX (delisted names) may return 0 bars each.
    That's expected -- it's the actual finding this run is testing for.
  - FB may or may not return data post-2022 depending on how Yahoo handled
    the FB->META ticker migration on their end.
  - Real gaps (holidays, halts, data holes) will show up where the synthetic
    run had none -- that's real signal, not a defect in the pipeline.
  - Some of tests/test_phase1.py will legitimately fail against real data,
    specifically test_abmd_gap_is_visible_not_filled (that gap was synthetic)
    and the exact row-count assumptions. Re-read what actually failed and
    why before treating it as broken.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.database.db import init_db
from src.ingestion.price_sources import YFinancePriceSource
from src.ingestion import pipeline as pl
from run_phase1_demo import (
    DB_PATH, SCHEMA_PATH, CSV_PATH, TEST_SUBSET, PRICE_START, PRICE_END,
    run as run_membership_and_setup,
)
from src.ingestion.adjustments import compute_split_adjusted_for_all, compute_total_return_for_all


def run_real_price_ingest():
    # Reuse steps 1-8 (membership, security master, delisted flags) from the
    # synthetic demo unchanged — only price ingestion differs.
    result = run_membership_and_setup(reset_db=True)
    security_resolver = result["security_resolver"]

    from src.database.db import get_connection
    conn = get_connection(DB_PATH)

    print("\n=== Fetching REAL price data via yfinance (with rename fallback) ===")
    price_source = YFinancePriceSource(auto_adjust=False, verbose=True)

    # Load the known-renames registry -- explicit, curated, not auto-detected.
    rename_result = pl.load_known_renames(conn, "data/reference/known_ticker_renames_seed.csv")
    print(f"Loaded rename registry: {rename_result}")

    price_windows = {
        "AAPL": (PRICE_START, PRICE_END), "MSFT": (PRICE_START, PRICE_END), "JNJ": (PRICE_START, PRICE_END),
        "PG": (PRICE_START, PRICE_END), "KO": (PRICE_START, PRICE_END), "XOM": (PRICE_START, PRICE_END),
        "JPM": (PRICE_START, PRICE_END), "WMT": (PRICE_START, PRICE_END), "HD": (PRICE_START, PRICE_END),
        "DIS": (PRICE_START, PRICE_END), "A": (PRICE_START, PRICE_END), "ABBV": (PRICE_START, PRICE_END),
        "FB": (PRICE_START, "2022-06-09"), "META": ("2022-06-09", PRICE_END),
        "MON": ("2015-01-01", "2018-06-07"),
        "AAL": ("2015-03-23", PRICE_END),
        "ABMD": (PRICE_START, "2022-12-22"),
        "ACGL": ("2022-11-01", PRICE_END),
        "ABNB": ("2023-09-18", PRICE_END),
        "AABA": (PRICE_START, "2019-10-02"),
        "ABC": (PRICE_START, "2023-08-30"),
        "ABX": (PRICE_START, PRICE_END),
        # AAMRQ, ABX intentionally have no window — pre-date any reasonable
        # free-data coverage; left as a documented pre-coverage gap.
    }

    totals = {"inserted": 0, "skipped_duplicate": 0, "zero_data_tickers": [], "redirected_tickers": {}}
    for ticker, (start, end) in price_windows.items():
        sec_id = security_resolver[ticker]
        res = pl.ingest_prices_with_rename_fallback(conn, sec_id, ticker, start, end, price_source)
        if res["bars_fetched"] == 0:
            totals["zero_data_tickers"].append(ticker)
        if res["redirect_used"]:
            totals["redirected_tickers"][ticker] = res["redirect_used"]
        totals["inserted"] += res["inserted"]
        totals["skipped_duplicate"] += res["skipped_duplicate"]

    print("\n=== REAL DATA INGEST SUMMARY ===")
    print(f"Rows inserted: {totals['inserted']}")
    print(f"Tickers recovered via known-rename redirect: {totals['redirected_tickers']}")
    print(f"Tickers with ZERO data returned (even after rename check): {totals['zero_data_tickers']}")
    print("\nCompare the zero-data list against MON/ABMD/AABA/AAMRQ/ABX from before --")
    print("anything that dropped OFF that list is now recovered via the rename registry.")

    print("\n=== Computing split-adjusted prices (THIS run has real data, so this is meaningful) ===")
    adj_results = compute_split_adjusted_for_all(conn)
    print(f"Split-adjustment computed for {len(adj_results)} securities.")
    aapl_id = security_resolver.get("AAPL")
    if aapl_id:
        check = conn.execute(
            "SELECT date, close FROM prices WHERE security_id=? AND adj_type='split_adjusted' "
            "AND date IN ('2020-08-28','2020-08-31')", (aapl_id,)
        ).fetchall()
        if len(check) == 2:
            vals = {r["date"]: r["close"] for r in check}
            ret = vals["2020-08-31"] / vals["2020-08-28"] - 1
            print(f"Sanity check -- AAPL split-adjusted return across the 2020-08-31 split date: {ret:.1%} "
                  f"(should be small/realistic; raw would show ~-75%)")

    print("\n=== Fetching REAL dividend data and computing total-return series ===")
    div_totals = {"tickers_with_dividends": 0, "total_dividend_events": 0}
    for ticker, sec_id in security_resolver.items():
        if ticker == "META":
            continue
        start, end = price_windows.get(ticker, (PRICE_START, PRICE_END))
        divs = price_source.fetch_dividends(ticker, start, end)
        if divs:
            div_totals["tickers_with_dividends"] += 1
            div_totals["total_dividend_events"] += len(divs)
            for ex_date, amount in divs:
                pl.ingest_corporate_action(conn, sec_id, "dividend", ex_date, amount,
                                            f"${amount} dividend (yfinance)", "yfinance", "C",
                                            quality="unverified")
    print(f"Dividend ingest: {div_totals}")
    tr_results = compute_total_return_for_all(conn)
    print(f"Total-return series computed for {len(tr_results)} securities.")

    conn.close()
    return totals


if __name__ == "__main__":
    run_real_price_ingest()
