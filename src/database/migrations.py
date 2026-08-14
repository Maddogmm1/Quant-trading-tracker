"""
Column-level migrations. New tables are handled by re-running schema.sql
on every init_db() call (every CREATE TABLE uses IF NOT EXISTS, safe on
an existing database), but that doesn't add new columns to a table that
already existed before the column was introduced. This module covers
that gap: securities.cik, securities.delisting_confidence, etc. were all
added after the securities table itself already existed in earlier
databases.

Each check is column-by-column via PRAGMA table_info(), so it's safe to
run against any database state every time a connection opens.
"""


def _column_exists(conn, table_name, column_name):
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(r["name"] == column_name for r in rows)


def apply_migrations(conn):
    applied = []

    column_migrations = [
        ("securities", "cik", "TEXT"),
        ("securities", "has_unsupported_corporate_action", "INTEGER NOT NULL DEFAULT 0"),
        ("securities", "unsupported_corporate_action_note", "TEXT"),
        ("securities", "delisting_confidence", "TEXT NOT NULL DEFAULT 'unverified'"),
        ("securities", "delisting_source", "TEXT"),
    ]

    for table, column, coltype in column_migrations:
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not table_exists:
            continue
        if not _column_exists(conn, table, column):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
            applied.append(f"{table}.{column}")

    if applied:
        conn.commit()

    return applied
