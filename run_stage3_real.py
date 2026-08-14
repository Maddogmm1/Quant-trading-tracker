"""
Stage 3: full historical S&P 500 ingestion (~1,206 securities).

Run this LOCALLY, in the SAME folder as your existing Stage 1/2 database --
Stage 1+2 already cover ~250 of these 1,206, so resumability means only
~950 need fresh fetching.

Setup:
    python3 run_stage3_real.py

Expected: ~34-45 minutes for the fresh-fetch portion. Prints progress per
ticker -- this is a long run, best left uninterrupted. If it is
interrupted, just re-run this script: resumability means already-completed
tickers are skipped, not re-fetched.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from run_stage_ingestion import run_stage
from src.ingestion.price_sources import YFinancePriceSource
from src.validation.stage_report import format_report_markdown

if __name__ == "__main__":
    print("Starting Stage 3: full historical S&P 500 ingestion (~1,206 securities).")
    print("This is a LONG run (est. 34-45 minutes for fresh fetches). Progress prints per ticker.")
    print("If interrupted, just re-run this script -- resumability skips already-completed tickers.\n")

    source = YFinancePriceSource(verbose=True)
    report, timing = run_stage(3, source, reset=False, force_reset=False)

    md = format_report_markdown(report)
    print("\n\n" + md)

    with open("STAGE3_REPORT.md", "w") as f:
        f.write(md)
    with open("STAGE3_REPORT.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n\nFull report saved to STAGE3_REPORT.md and STAGE3_REPORT.json.")
