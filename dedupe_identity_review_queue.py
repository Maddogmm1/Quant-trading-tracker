"""
One-off cleanup: removes duplicate identity_review_queue rows created
before flag_identity_review() had a duplicate check. Keeps the earliest
row per (ticker, period boundaries), deletes the rest. Safe to run
multiple times (idempotent -- a clean table stays clean).

Run: python3 dedupe_identity_review_queue.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.database.db import get_connection

DB_PATH = "data/database/quant_trader_stage.db"

if __name__ == "__main__":
    conn = get_connection(DB_PATH)

    before = conn.execute("SELECT COUNT(*) c FROM identity_review_queue").fetchone()["c"]

    conn.execute("""
        DELETE FROM identity_review_queue
        WHERE review_id NOT IN (
            SELECT MIN(review_id) FROM identity_review_queue
            GROUP BY ticker, IFNULL(period_1_start,''), IFNULL(period_1_end,''),
                     IFNULL(period_2_start,''), IFNULL(period_2_end,'')
        )
    """)
    conn.commit()

    after = conn.execute("SELECT COUNT(*) c FROM identity_review_queue").fetchone()["c"]
    print(f"identity_review_queue: {before} rows -> {after} rows ({before - after} duplicates removed)")

    remaining = conn.execute("SELECT ticker, gap_days FROM identity_review_queue").fetchall()
    print("\nRemaining flags:")
    for r in remaining:
        print(f"  {r['ticker']}: {r['gap_days']} day gap")

    conn.close()
