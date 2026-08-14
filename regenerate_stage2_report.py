"""
Regenerates a fully-corrected STAGE2_REPORT.md/json against your EXISTING
database -- does NOT re-fetch all 200 tickers (almost all already have
data and would just get skipped again, which is exactly why a plain
re-run wouldn't have fixed the timing measurement).

Instead:
1. Force-refetches a small sample of ~10 KNOWN-good tickers (already
   confirmed to have real data) to get a genuine, honest "cost of a real
   fetch" timing sample under the latest report-generation code.
2. Generates the full data-quality/dividend/split/identity report against
   all 200 Stage 2 tickers using your EXISTING ingested data -- no
   unnecessary re-fetching, so this is fast.

Run: python3 regenerate_stage2_report.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from run_stage_ingestion import run_stage
from src.ingestion.price_sources import YFinancePriceSource
from src.universe.stage_universe import get_stage_tickers
from src.validation.stage_report import format_report_markdown

DB_PATH = "data/database/quant_trader_stage.db"
CSV_PATH = "data/raw/sp500-master/sp500_ticker_start_end.csv"

# Known-good tickers already confirmed to have real, substantial price
# history -- forcing a fresh fetch of these (small, fast) gives an honest
# timing sample without re-fetching all 200.
TIMING_SAMPLE_TICKERS = {"AAPL", "MSFT", "JNJ", "PG", "KO", "XOM", "JPM", "WMT", "HD", "DIS"}

if __name__ == "__main__":
    print("Regenerating Stage 2 report with the latest fixes...")
    print(f"Forcing a fresh timing sample from {len(TIMING_SAMPLE_TICKERS)} known-good tickers "
          f"(fast, ~20-30s) to get an honest per-security cost estimate.")
    print("The other ~190 tickers use your EXISTING data -- not re-fetched.\n")

    all_200 = get_stage_tickers(2, CSV_PATH)
    source = YFinancePriceSource(verbose=True)

    report, timing = run_stage(
        stage=2,
        price_source=source,
        db_path=DB_PATH,
        reset=False,
        force_reset=False,
        tickers_override=all_200,
        force_tickers=TIMING_SAMPLE_TICKERS,
    )

    md = format_report_markdown(report)
    print("\n\n" + md)

    with open("STAGE2_REPORT.md", "w") as f:
        f.write(md)
    with open("STAGE2_REPORT.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n\nSaved to STAGE2_REPORT.md and STAGE2_REPORT.json.")
