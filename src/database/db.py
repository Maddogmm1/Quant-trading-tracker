"""Database connection and initialization utilities."""
import sqlite3
import os
from datetime import datetime, timezone
from src.database.migrations import apply_migrations

SCHEMA_VERSION = 1


def get_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path, schema_path, reset=False, force=False):
    """Create the database from schema.sql. If reset=True, delete existing file first.

    Refuses to reset a database that already contains real (non-synthetic,
    non-test) price data unless force=True is passed explicitly. The
    synthetic demo script and the real-data script share a DB_PATH by
    default, so without this check, re-running the synthetic demo after a
    real data pull would silently destroy real work.
    """
    if reset and os.path.exists(db_path):
        if not force:
            conn = get_connection(db_path)
            try:
                real_data_rows = conn.execute(
                    """SELECT COUNT(*) c FROM prices p JOIN data_sources s ON p.source_id = s.source_id
                       WHERE s.source_name NOT LIKE 'SYNTHETIC_DEMO%'
                         AND s.source_name NOT LIKE 'test_%' AND s.source_name NOT LIKE 'manual_%'"""
                ).fetchone()["c"]
            except sqlite3.OperationalError:
                real_data_rows = 0  # table doesn't exist yet, e.g. corrupt/partial DB -- safe to proceed
            conn.close()
            if real_data_rows > 0:
                raise RuntimeError(
                    f"Refusing to reset '{db_path}': it contains {real_data_rows} price rows from a "
                    f"non-synthetic source (looks like real data, e.g. from run_phase1_real_data.py). "
                    f"Call init_db(..., force=True) if you really want to destroy this data."
                )
        os.remove(db_path)

    is_new = not os.path.exists(db_path)
    conn = get_connection(db_path)
    # Apply the schema every time, not just for brand-new databases.
    # Every CREATE TABLE in schema.sql uses IF NOT EXISTS, so this safely
    # adds any new table to an existing database, but it won't add new
    # columns to a table that already existed before the column was
    # introduced (e.g. securities.cik) -- IF NOT EXISTS skips the whole
    # CREATE TABLE if the table's already there. apply_migrations()
    # covers that gap with explicit ALTER TABLE ADD COLUMN calls, checked
    # column-by-column. Together these let an older database catch up to
    # the current schema without losing data.
    with open(schema_path, "r") as f:
        conn.executescript(f.read())
    migrations_applied = apply_migrations(conn)
    if migrations_applied:
        print(f"[db] Applied {len(migrations_applied)} schema migration(s) to existing database: {migrations_applied}")
    if is_new:
        conn.execute(
            "INSERT INTO schema_meta (schema_version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
        )
    conn.commit()
    return conn


def now_iso():
    return datetime.now(timezone.utc).isoformat()
