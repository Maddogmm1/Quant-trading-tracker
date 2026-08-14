"""
Applies the classify_split_ratio() fix to an EXISTING Stage 1 database,
in place -- no re-fetching from yfinance required. Run this on the
database you already built with the previous version of the code.

Setup: none needed beyond what Stage 1 already required.
Run:
    python3 reclassify_stage1_splits.py

This will:
1. Re-check every recorded split/reverse_split against the (now fixed)
   whole-number-ratio heuristic.
2. Flag any spinoff-artifact ratios (excluding them from split-adjustment
   math) and mark the affected security.
3. Recompute split_adjusted and total_return series for everyone.
4. Regenerate the full validation report.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.database.db import get_connection, now_iso
from src.ingestion.adjustments import classify_split_ratio, compute_split_adjusted_for_all, compute_total_return_for_all
from src.validation.stage_report import generate_stage_report, format_report_markdown

DB_PATH = "data/database/quant_trader_stage.db"


def reclassify(db_path=DB_PATH):
    conn = get_connection(db_path)

    rows = conn.execute("""
        SELECT ca.action_id, ca.security_id, ca.action_type, ca.ex_date, ca.ratio_or_value,
               ca.corporate_action_quality, s.primary_ticker
        FROM corporate_actions ca JOIN securities s ON ca.security_id = s.security_id
        WHERE ca.action_type IN ('split','reverse_split')
    """).fetchall()

    reclassified = []
    for row in rows:
        classification = classify_split_ratio(row["ratio_or_value"])
        new_quality = "unverified" if classification == "genuine" else "likely_spinoff_artifact"
        if row["corporate_action_quality"] == new_quality:
            continue  # already correctly classified, nothing to do

        conn.execute(
            "UPDATE corporate_actions SET corporate_action_quality=? WHERE action_id=?",
            (new_quality, row["action_id"]),
        )
        reclassified.append({
            "ticker": row["primary_ticker"], "ex_date": row["ex_date"],
            "ratio": row["ratio_or_value"], "old_quality": row["corporate_action_quality"],
            "new_quality": new_quality,
        })

        if new_quality == "likely_spinoff_artifact":
            note = (f"yfinance reported a 'split' ratio of {row['ratio_or_value']} on {row['ex_date']}, "
                    f"which does not match common whole-share-count split ratios. Likely a spinoff or "
                    f"other corporate action's residual price-adjustment factor, not a genuine split -- "
                    f"flagged for manual investigation. (Reclassified after Stage 1 initial run.)")
            conn.execute(
                "UPDATE securities SET has_unsupported_corporate_action=1, "
                "unsupported_corporate_action_note=COALESCE(unsupported_corporate_action_note || ' | ', '') || ?, "
                "updated_at=? WHERE security_id=?",
                (note, now_iso(), row["security_id"]),
            )
    conn.commit()

    print(f"Reclassified {len(reclassified)} corporate action rows:")
    for r in reclassified:
        print(f"  {r['ticker']:8s} {r['ex_date']}  ratio={r['ratio']}  "
              f"{r['old_quality']} -> {r['new_quality']}")

    print("\nRecomputing split-adjusted and total-return series (derived data -- safe to regenerate)...")
    compute_split_adjusted_for_all(conn)
    compute_total_return_for_all(conn)
    conn.commit()

    tickers = [r["primary_ticker"] for r in conn.execute("SELECT DISTINCT primary_ticker FROM securities").fetchall()]
    report = generate_stage_report(
        conn, tickers, [], "yfinance (Yahoo Finance, unofficial) [reclassified in-place, no re-fetch]", 1
    )
    md = format_report_markdown(report)

    with open("STAGE1_REPORT_RECLASSIFIED.md", "w") as f:
        f.write(md)
    print("\n\n" + md)
    print("\n\nSaved to STAGE1_REPORT_RECLASSIFIED.md")

    conn.close()


if __name__ == "__main__":
    reclassify()
