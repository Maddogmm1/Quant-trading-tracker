"""Core Phase 1 ingestion pipeline. Every ingest_* function is idempotent:
running it twice on the same input must not create duplicate rows."""
from src.database.db import now_iso


def get_or_create_source(conn, source_name, tier, description="", url=""):
    row = conn.execute("SELECT source_id FROM data_sources WHERE source_name = ?", (source_name,)).fetchone()
    if row:
        return row["source_id"]
    cur = conn.execute(
        "INSERT INTO data_sources (source_name, tier, description, url) VALUES (?,?,?,?)",
        (source_name, tier, description, url),
    )
    conn.commit()
    return cur.lastrowid


def get_or_create_security(conn, ticker, name=None, exchange=None, country=None,
                            currency=None, sector=None, asset_type="STOCK",
                            first_seen_date=None, identifier_quality="resolved"):
    """Resolve a ticker to a security_id. Idempotent: same ticker -> same id.
    Phase-1-simple resolution strategy (ticker as the resolution key when no
    ISIN/CIK is available) -- real cross-referencing via those is future work."""
    row = conn.execute("SELECT security_id FROM securities WHERE primary_ticker = ?", (ticker,)).fetchone()
    if row:
        return row["security_id"], False
    ts = now_iso()
    cur = conn.execute(
        """INSERT INTO securities
           (primary_ticker, name, exchange, country, currency, sector, asset_type,
            first_seen_date, active_flag, identifier_quality, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,1,?,?,?)""",
        (ticker, name, exchange, country, currency, sector, asset_type,
         first_seen_date, identifier_quality, ts, ts),
    )
    conn.commit()
    return cur.lastrowid, True


def link_ticker_history(conn, security_id, ticker, valid_from, valid_to, source):
    existing = conn.execute(
        "SELECT ticker_history_id FROM ticker_history WHERE ticker=? AND valid_from=?",
        (ticker, valid_from),
    ).fetchone()
    if existing:
        return existing["ticker_history_id"], False
    cur = conn.execute(
        "INSERT INTO ticker_history (security_id, ticker, valid_from, valid_to, source) VALUES (?,?,?,?,?)",
        (security_id, ticker, valid_from, valid_to, source),
    )
    conn.commit()
    return cur.lastrowid, True


def ingest_membership_records(conn, records, security_resolver):
    """records: list[MembershipRecord]. security_resolver: dict[ticker]->security_id
    (None if unresolved). Idempotent on (raw_ticker, index_name, effective_date, source).
    Duplicate claims from the same source/run are skipped; claims from
    different sources that disagree are kept as separate rows rather than
    merged."""
    inserted, skipped_dupe, unresolved = 0, 0, 0
    for r in records:
        source_id = get_or_create_source(conn, r.source_name, r.source_tier)
        existing = conn.execute(
            """SELECT membership_id FROM index_membership
               WHERE raw_ticker=? AND index_name=? AND effective_date=? AND source_id=?""",
            (r.raw_ticker, r.index_name, r.effective_date, source_id),
        ).fetchone()
        if existing:
            skipped_dupe += 1
            continue

        security_id = security_resolver.get(r.raw_ticker)
        quality = "complete" if security_id else "unresolved"
        if security_id is None:
            unresolved += 1

        conn.execute(
            """INSERT INTO index_membership
               (security_id, raw_ticker, index_name, effective_date, removal_date,
                announcement_date, source_id, source_reference, confidence,
                verification_status, membership_quality, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (security_id, r.raw_ticker, r.index_name, r.effective_date, r.removal_date,
             r.announcement_date, source_id, r.source_reference, r.confidence,
             r.verification_status, quality, now_iso()),
        )
        inserted += 1
    conn.commit()
    return {"inserted": inserted, "skipped_duplicate": skipped_dupe, "unresolved": unresolved}


def ingest_prices(conn, security_id, ticker, bars, source_name, adj_type="raw"):
    """Idempotent on (security_id, date, adj_type, source_id) via schema UNIQUE constraint."""
    source_id = get_or_create_source(conn, source_name, "C",
                                      description="Synthetic placeholder — see price_sources.py")
    inserted, skipped_dupe = 0, 0
    for bar in bars:
        try:
            conn.execute(
                """INSERT INTO prices
                   (security_id, date, open, high, low, close, volume, adj_type,
                    source_id, price_data_quality, ingested_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (security_id, bar.date, bar.open, bar.high, bar.low, bar.close,
                 bar.volume, adj_type, source_id, "ok", now_iso()),
            )
            inserted += 1
        except Exception as e:
            if "UNIQUE" in str(e):
                skipped_dupe += 1
            else:
                raise
    conn.commit()
    return {"inserted": inserted, "skipped_duplicate": skipped_dupe}


def ingest_corporate_action(conn, security_id, action_type, ex_date, ratio_or_value,
                             detail, source_name, source_tier, quality="unverified"):
    source_id = get_or_create_source(conn, source_name, source_tier)
    existing = conn.execute(
        "SELECT action_id FROM corporate_actions WHERE security_id=? AND action_type=? AND ex_date=?",
        (security_id, action_type, ex_date),
    ).fetchone()
    if existing:
        return existing["action_id"], False
    cur = conn.execute(
        """INSERT INTO corporate_actions
           (security_id, action_type, ex_date, ratio_or_value, detail, source_id,
            corporate_action_quality, ingested_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (security_id, action_type, ex_date, ratio_or_value, detail, source_id, quality, now_iso()),
    )
    conn.commit()
    return cur.lastrowid, True


def load_known_renames(conn, csv_path):
    """Load the manually curated rename registry into the database.
    Idempotent on (old_ticker, new_ticker, effective_date)."""
    import csv as csv_module
    inserted, skipped = 0, 0
    with open(csv_path, "r") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            existing = conn.execute(
                "SELECT rename_id FROM known_ticker_renames WHERE old_ticker=? AND new_ticker=? AND effective_date=?",
                (row["old_ticker"], row["new_ticker"], row["effective_date"]),
            ).fetchone()
            if existing:
                skipped += 1
                continue
            conn.execute(
                """INSERT INTO known_ticker_renames
                   (old_ticker, new_ticker, effective_date, source, confidence, notes)
                   VALUES (?,?,?,?,?,?)""",
                (row["old_ticker"], row["new_ticker"], row["effective_date"],
                 row.get("source"), row.get("confidence", "unverified"), row.get("notes")),
            )
            inserted += 1
    conn.commit()
    return {"inserted": inserted, "skipped_duplicate": skipped}


def lookup_rename(conn, ticker):
    """Return the new_ticker if `ticker` is a known old ticker, else None.
    Only consults the explicit registry; never guesses."""
    row = conn.execute(
        "SELECT new_ticker, effective_date, confidence FROM known_ticker_renames WHERE old_ticker=?",
        (ticker,),
    ).fetchone()
    return dict(row) if row else None


def ingest_prices_with_rename_fallback(conn, security_id, ticker, start_date, end_date,
                                        price_source, adj_type="raw"):
    """
    Fetch price data for `ticker`. If the source returns nothing and `ticker`
    is a known rename (per the explicit registry, never auto-detected),
    retry under the new ticker for the same window, record the redirect
    (ticker_history + corporate_actions), and proceed on the recovered data.

    If the ticker is not a known rename, this behaves like a normal fetch:
    an empty result stays empty and visible in the diagnostics rather than
    being retried against arbitrary alternative tickers.
    """
    bars = price_source.fetch(ticker, start_date, end_date)
    redirect_used = None

    if len(bars) == 0:
        rename = lookup_rename(conn, ticker)
        if rename:
            new_ticker = rename["new_ticker"]
            retry_bars = price_source.fetch(new_ticker, start_date, end_date)
            if len(retry_bars) > 0:
                bars = retry_bars
                redirect_used = new_ticker

                # Record the redirect explicitly via ticker_history + corporate_actions
                # rather than substituting data silently under the old ticker's name.
                link_ticker_history(conn, security_id, ticker, "1900-01-01",
                                     rename["effective_date"], "known_ticker_renames registry")
                link_ticker_history(conn, security_id, new_ticker, rename["effective_date"],
                                     None, "known_ticker_renames registry")
                ingest_corporate_action(
                    conn, security_id, "ticker_change", rename["effective_date"], None,
                    f"{ticker} -> {new_ticker} (recovered via known-renames registry, "
                    f"confidence={rename['confidence']})",
                    "known_ticker_renames registry", "B",
                    quality=rename["confidence"],
                )

    result = ingest_prices(conn, security_id, ticker, bars, price_source.source_name, adj_type=adj_type)
    result["redirect_used"] = redirect_used
    result["bars_fetched"] = len(bars)
    return result


def flag_unsupported_corporate_action(conn, security_id, action_type, action_date, note, source_name, source_tier):
    """
    Records that a security has undergone a corporate action type we don't
    process (spin-offs, rights issues -- see BACKLOG.md).

    Doesn't attempt to adjust prices for it. Marks the security so the
    eventual backtester can exclude it or analyze it separately instead of
    treating a price series with an unadjusted discontinuity as clean.
    """
    source_id = get_or_create_source(conn, source_name, source_tier)
    conn.execute(
        "UPDATE securities SET has_unsupported_corporate_action=1, "
        "unsupported_corporate_action_note=?, updated_at=? WHERE security_id=?",
        (note, now_iso(), security_id),
    )
    # Also record it as a corporate_actions row for a full audit trail;
    # corporate_action_quality makes clear it's flagged, not processed.
    ingest_corporate_action(
        conn, security_id, action_type, action_date, None,
        f"UNSUPPORTED, FLAGGED NOT PROCESSED: {note}", source_name, source_tier,
        quality="unverified",
    )
    conn.commit()


def securities_with_unsupported_corporate_actions(conn):
    """For the eventual backtester: which securities need special handling
    (exclusion or separate analysis) due to an unprocessed corporate action."""
    rows = conn.execute(
        "SELECT security_id, primary_ticker, unsupported_corporate_action_note "
        "FROM securities WHERE has_unsupported_corporate_action=1"
    ).fetchall()
    return [dict(r) for r in rows]


def log_run(conn, run_type, source_id, requested, succeeded, failed, inserted, skipped, warnings=""):
    conn.execute(
        """INSERT INTO ingestion_log
           (run_timestamp, run_type, source_id, securities_requested, securities_succeeded,
            securities_failed, rows_inserted, rows_skipped_duplicate, warnings, schema_version)
           VALUES (?,?,?,?,?,?,?,?,?,1)""",
        (now_iso(), run_type, source_id, requested, succeeded, failed, inserted, skipped, warnings),
    )
    conn.commit()
