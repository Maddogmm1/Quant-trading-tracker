"""
Retries ONLY specific failed tickers (e.g. PZE, SVU, TWX from the real
Stage 2 crash) against the existing database, without re-fetching
everything else. Uses force_tickers to bypass the resume-skip check,
since these tickers already have price rows but never got their
dividends/splits processed due to the (now-fixed) malformed-data bug.

Run: python3 retry_failed.py PZE SVU TWX
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from run_stage_ingestion import run_stage
from src.ingestion.price_sources import YFinancePriceSource

if __name__ == "__main__":
    failed_tickers = sys.argv[1:]
    if not failed_tickers:
        print("Usage: python3 retry_failed.py TICKER1 TICKER2 ...")
        print("Example: python3 retry_failed.py PZE SVU TWX")
        sys.exit(1)

    print(f"Retrying {len(failed_tickers)} tickers: {failed_tickers}")
    print("(force_tickers bypasses resume-skip, so these WILL be re-fetched "
          "even though they already have price data)\n")

    source = YFinancePriceSource(verbose=True)
    report, timing = run_stage(
        stage=2,  # stage number only affects report labeling here, not scope
        price_source=source,
        reset=False,
        force_reset=False,
        tickers_override=failed_tickers,
        force_tickers=set(failed_tickers),
    )

    print("\n\nRetry complete. Results:")
    for r in timing:
        status = "ERROR: " + r["error"] if r.get("error") else "OK"
        print(f"  {r['ticker']:8s} bars={r['bars_fetched']} divs={r.get('dividends_fetched')} "
              f"splits={r.get('splits_fetched')}  {status}")
