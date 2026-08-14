"""
Stage 2: real-data ingestion for ~200 S&P 500 securities.

Run this LOCALLY. Requires the SAME database used for Stage 1 to be
either fresh or already contain Stage 1's data -- Stage 2's ticker list
is Stage 1's ~50 PLUS ~150 more (deterministic, seeded), so re-running
against the existing quant_trader_stage.db will correctly SKIP the ~50
Stage 1 tickers (resume behavior) and only fetch the ~150 new ones --
meaningfully faster than a from-scratch run, and a direct test of the
resumability requirement.

Setup:
    python3 run_stage2_real.py

Expect roughly 200 x ~2.3s ~ 7-8 minutes MINUS however much Stage 1
already covered (if run against the same DB) -- watch the "resumability"
section of the output to see how much was actually skipped vs fetched fresh.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from run_stage_ingestion import run_stage
from src.ingestion.price_sources import YFinancePriceSource
from src.validation.stage_report import format_report_markdown

if __name__ == "__main__":
    print("Starting Stage 2 real-data ingestion (~200 securities)...")
    print("If your Stage 1 database already exists, Stage 1's ~50 tickers will be")
    print("SKIPPED (already ingested) -- only ~150 new tickers get freshly fetched.\n")

    source = YFinancePriceSource(verbose=True)
    # reset=False: build ON TOP of the existing Stage 1 database, not a fresh one --
    # this is what makes the resumability/skip behavior actually apply.
    report, timing = run_stage(2, source, reset=False, force_reset=False)

    md = format_report_markdown(report)
    print("\n\n" + md)

    with open("STAGE2_REPORT.md", "w") as f:
        f.write(md)
    with open("STAGE2_REPORT.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n\nFull report saved to STAGE2_REPORT.md and STAGE2_REPORT.json.")
