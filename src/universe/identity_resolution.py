"""
Stable security identity resolution.

A ticker string isn't a permanent company identity -- the same ticker can
be reused by an unrelated company years later once the original is
delisted, and naively resolving "ticker -> security_id" risks merging two
different companies into one record.

Preferred fix: a stable identifier (CIK, ISIN) via the known_identifiers
registry. Fall back to ticker-only resolution when no identifier is
available, and in that case check for suspicious reuse patterns (a large
temporal gap between two periods of the same ticker) and flag them for
human review instead of assuming continuity.

known_identifiers is a small, explicit, sourced registry rather than
automated bulk resolution -- same approach as known_ticker_renames.
"""
from src.database.db import now_iso

# A gap this large between two periods of the same ticker is treated as
# suspicious enough to flag for review rather than assume continuity.
# Deliberately conservative -- most genuine corporate continuity (e.g. a
# brief trading halt) is much shorter than this.
SUSPICIOUS_GAP_DAYS = 365 * 2


def load_known_identifiers(conn, csv_path):
    """Load the CIK/ISIN registry into the database. Idempotent."""
    import csv as csv_module
    inserted, skipped = 0, 0
    with open(csv_path, "r") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            existing = conn.execute(
                "SELECT identifier_id FROM known_identifiers WHERE ticker=? AND valid_from=?",
                (row["ticker"], row["valid_from"]),
            ).fetchone()
            if existing:
                skipped += 1
                continue
            conn.execute(
                """INSERT INTO known_identifiers
                   (ticker, cik, isin, valid_from, valid_to, source, confidence, notes)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (row["ticker"], row.get("cik") or None, row.get("isin") or None,
                 row["valid_from"], row.get("valid_to") or None,
                 row.get("source"), row.get("confidence", "unverified"), row.get("notes")),
            )
            inserted += 1
    conn.commit()
    return {"inserted": inserted, "skipped_duplicate": skipped}


def lookup_known_identifier(conn, ticker, as_of_date=None):
    """Return the known_identifiers row covering `ticker` at `as_of_date`
    (or the most recent row for that ticker if as_of_date is None)."""
    if as_of_date:
        row = conn.execute(
            """SELECT * FROM known_identifiers WHERE ticker=?
               AND valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?)
               ORDER BY valid_from DESC LIMIT 1""",
            (ticker, as_of_date, as_of_date),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM known_identifiers WHERE ticker=? ORDER BY valid_from DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    return dict(row) if row else None


def find_security_by_cik(conn, cik):
    row = conn.execute("SELECT * FROM securities WHERE cik=?", (cik,)).fetchone()
    return dict(row) if row else None


def flag_identity_review(conn, ticker, reason, period_1=None, period_2=None, gap_days=None):
    """Idempotent: identity resolution runs for every ticker on every stage
    run regardless of whether price data was skipped, so this gets called
    repeatedly for the same real gap across runs. Without a duplicate
    check, AMD/PTC/FMC/PCG's flags doubled from 4 to 8 after a second run.
    Checks for an existing unresolved flag with the same ticker + period
    boundaries before inserting."""
    p1s = period_1[0] if period_1 else None
    p1e = period_1[1] if period_1 else None
    p2s = period_2[0] if period_2 else None
    p2e = period_2[1] if period_2 else None
    existing = conn.execute(
        """SELECT review_id FROM identity_review_queue
           WHERE ticker=? AND IFNULL(period_1_start,'')=IFNULL(?,'')
             AND IFNULL(period_1_end,'')=IFNULL(?,'') AND IFNULL(period_2_start,'')=IFNULL(?,'')
             AND IFNULL(period_2_end,'')=IFNULL(?,'')""",
        (ticker, p1s, p1e, p2s, p2e),
    ).fetchone()
    if existing:
        return  # already flagged for this exact gap -- don't duplicate
    conn.execute(
        """INSERT INTO identity_review_queue
           (ticker, flag_reason, period_1_start, period_1_end, period_2_start, period_2_end,
            gap_days, resolved, flagged_at)
           VALUES (?,?,?,?,?,?,?,0,?)""",
        (ticker, reason, p1s, p1e, p2s, p2e, gap_days, now_iso()),
    )
    conn.commit()


def detect_ticker_reuse_risk(conn, ticker):
    """
    Scans this ticker's historical usage periods (from index_membership,
    since that's what we have dated windows for) for a gap large enough
    to be suspicious. Doesn't decide the answer, just flags it for review.
    Returns the flag info if one was raised, else None.
    """
    rows = conn.execute(
        """SELECT DISTINCT effective_date, removal_date FROM index_membership
           WHERE raw_ticker=? ORDER BY effective_date""",
        (ticker,),
    ).fetchall()
    if len(rows) < 2:
        return None

    for i in range(len(rows) - 1):
        end_1 = rows[i]["removal_date"]
        start_2 = rows[i + 1]["effective_date"]
        if end_1 is None:
            continue  # still active per this claim -- no gap to measure
        import datetime
        try:
            d1 = datetime.date.fromisoformat(end_1)
            d2 = datetime.date.fromisoformat(start_2)
        except ValueError:
            continue
        gap_days = (d2 - d1).days
        if gap_days >= SUSPICIOUS_GAP_DAYS:
            flag_identity_review(
                conn, ticker,
                f"Ticker '{ticker}' has two usage periods separated by {gap_days} days "
                f"with no known CIK/ISIN to confirm they're the same entity -- could be "
                f"the same company (e.g. reorganization) or an unrelated company reusing "
                f"the ticker after the original was delisted.",
                period_1=(rows[i]["effective_date"], end_1),
                period_2=(start_2, rows[i + 1]["removal_date"]),
                gap_days=gap_days,
            )
            return {"ticker": ticker, "gap_days": gap_days}
    return None


def resolve_or_create_security(conn, ticker, as_of_date=None, name=None, exchange=None,
                                country=None, currency=None, asset_type="STOCK",
                                first_seen_date=None):
    """
    The main identity resolution entry point. Order of preference:
    1. known_identifiers registry (CIK) -- if the CIK already maps to an
       existing security, return that security_id (correctly links across
       renames/ticker changes even without a known_ticker_renames entry).
    2. known_identifiers registry (CIK), no existing security yet -- create
       one, tagged identifier_quality='resolved'.
    3. No registry entry at all -- fall back to ticker-based lookup, but
       run the reuse-risk check and flag for review rather than trusting
       continuity. identifier_quality='unresolved' in this case, always.

    Returns (security_id, created: bool, identity_method: str)
    """
    identifier = lookup_known_identifier(conn, ticker, as_of_date)

    if identifier and identifier.get("cik"):
        existing = find_security_by_cik(conn, identifier["cik"])
        if existing:
            return existing["security_id"], False, "cik_matched_existing"

        ts = now_iso()
        cur = conn.execute(
            """INSERT INTO securities
               (primary_ticker, name, exchange, country, currency, asset_type, cik,
                first_seen_date, active_flag, identifier_quality, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,1,'resolved',?,?)""",
            (ticker, name, exchange, country, currency, asset_type, identifier["cik"],
             first_seen_date, ts, ts),
        )
        conn.commit()
        return cur.lastrowid, True, "cik_created_new"

    # No known identifier -- fall back to ticker, but check for reuse risk first
    reuse_flag = detect_ticker_reuse_risk(conn, ticker)

    row = conn.execute("SELECT security_id FROM securities WHERE primary_ticker=?", (ticker,)).fetchone()
    if row:
        method = "ticker_fallback_flagged_for_review" if reuse_flag else "ticker_fallback"
        return row["security_id"], False, method

    ts = now_iso()
    cur = conn.execute(
        """INSERT INTO securities
           (primary_ticker, name, exchange, country, currency, asset_type,
            first_seen_date, active_flag, identifier_quality, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,1,'unresolved',?,?)""",
        (ticker, name, exchange, country, currency, asset_type, first_seen_date, ts, ts),
    )
    conn.commit()
    method = "ticker_fallback_new_flagged_for_review" if reuse_flag else "ticker_fallback_new"
    return cur.lastrowid, True, method
