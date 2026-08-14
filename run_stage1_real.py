"""
Stage 1: real-data ingestion for ~50 representative S&P 500 securities.

Requires network access to Yahoo Finance.

Setup:
    pip install yfinance
    cd quant_trader
    python3 run_stage1_real.py

This will take a while -- expect roughly 1-2 seconds per ticker (OHLCV +
dividends + splits + retry/backoff overhead) x 50 tickers, so probably
2-5 minutes total, possibly more if Yahoo rate-limits. The script prints
progress per ticker as it goes, and reports exact timing at the end,
including extrapolated estimates for 200/500/1200 securities -- that
extrapolation is the actual point of this stage.

The full validation report prints to the console and also gets saved to
STAGE1_REPORT.md.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from run_stage_ingestion import run_stage
from src.ingestion.price_sources import YFinancePriceSource
from src.validation.stage_report import format_report_markdown

if __name__ == "__main__":
    print("Starting Stage 1 real-data ingestion (~50 securities)...")
    print("This uses the ACTUAL yfinance API and will take several minutes.\n")

    source = YFinancePriceSource(verbose=True)
    # reset=False: this may be running against a database that already has
    # Stage 2 (or later) data on top of Stage 1's -- resetting would destroy
    # that. Stage 1's tickers are always a guaranteed subset (see
    # stage_universe.py), so this just ensures they're present/up to date
    # and generates a report SCOPED to just the Stage 1 ticker list.
    report, timing = run_stage(1, source, reset=False, force_reset=False)

    md = format_report_markdown(report)
    print("\n\n" + md)

    with open("STAGE1_REPORT.md", "w") as f:
        f.write(md)
    import json
    with open("STAGE1_REPORT.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n\nFull report saved to STAGE1_REPORT.md and STAGE1_REPORT.json.")
