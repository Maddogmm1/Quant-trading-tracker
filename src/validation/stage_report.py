"""
Generates the validation report covering coverage, missing data, dividends,
splits, identity resolution, point-in-time availability, errors/timing, and
survivorship-bias assessment.

Since stages share a single growing database (Stage 2 builds on top of
Stage 1's DB for resumability), every query here must be scoped to the
`tickers` list passed in -- otherwise a "Stage 1 report" generated after
Stage 2 has run would report Stage 2's full numbers instead. This was a
real bug, caught before it corrupted a report: fixed by resolving
`tickers` to security_ids up front and filtering every query by them.
"""
from src.validation import checks


def generate_survivorship_categorization(conn, tickers):
    """
    Automated version of the manual audit originally done for Stage 2's 52
    zero-price-data securities. That audit required individual web
    research per ticker, which doesn't scale to 1,206. This uses data
    already on hand (ingestion_attempts status, sourced delisting records,
    identity resolution quality) to bucket every security automatically
    into 7 categories:
        1. Full usable history
        2. Partial history
        3. No historical price data (confirmed delisted/permanent)
        4. Provider-empty (JNPR/SWN/FLT pattern -- no error, no data)
        5. Download failure (transient, retry-able)
        6. Identity unresolved
        7. Other/unknown

    It's a heuristic classification based on available signals, not the
    same research-grade confidence as the manual audit -- it will
    mis-classify anything where the signals don't tell the full story
    (e.g. it can't know a security is really a ticker-reuse case unless
    identity_review_queue already flagged it).
    """
    placeholders = ",".join("?" for _ in tickers)
    sec_rows = conn.execute(
        f"SELECT * FROM securities WHERE primary_ticker IN ({placeholders})", tickers
    ).fetchall()

    categorized = {i: [] for i in range(1, 8)}
    CATEGORY_NAMES = {
        1: "full_usable_history", 2: "partial_history", 3: "no_historical_price_data",
        4: "provider_empty", 5: "download_failure", 6: "identity_unresolved", 7: "other_unknown",
    }

    for sec in sec_rows:
        ticker = sec["primary_ticker"]
        sec_id = sec["security_id"]

        price_count = conn.execute(
            "SELECT COUNT(*) c FROM prices WHERE security_id=? AND adj_type='raw'", (sec_id,)
        ).fetchone()["c"]
        date_range = conn.execute(
            "SELECT MIN(date) mn, MAX(date) mx FROM prices WHERE security_id=? AND adj_type='raw'", (sec_id,)
        ).fetchone()

        latest_attempt = conn.execute(
            "SELECT * FROM ingestion_attempts WHERE ticker=? ORDER BY attempted_at DESC LIMIT 1", (ticker,)
        ).fetchone()

        identity_flagged = conn.execute(
            "SELECT COUNT(*) c FROM identity_review_queue WHERE ticker=? AND resolved=0", (ticker,)
        ).fetchone()["c"] > 0

        category = None
        if sec["identifier_quality"] == "unresolved" and identity_flagged:
            category = 6
        elif price_count > 0:
            expected_days = _business_day_count_safe(date_range["mn"], date_range["mx"])
            completeness = price_count / expected_days if expected_days > 0 else 0
            category = 1 if completeness >= 0.90 else 2
        elif latest_attempt:
            status = latest_attempt["status"]
            if status == "SUCCESS_EMPTY_PROVIDER":
                category = 4
            elif status == "PERMANENT_FAILURE":
                category = 3
            elif status == "TRANSIENT_FAILURE":
                category = 5
            else:
                category = 7
        elif sec["active_flag"] == 0 and sec["delisting_confidence"] == "verified":
            # No ingestion_attempts row but we have a sourced delisting
            # record (e.g. from Stage 1/2's manually curated citations) --
            # treat as confirmed no-data, not unknown.
            category = 3
        else:
            category = 7  # never attempted, or genuinely unclear signals

        categorized[category].append(ticker)

    total = len(sec_rows)
    result = {}
    for i in range(1, 8):
        result[CATEGORY_NAMES[i]] = {
            "count": len(categorized[i]),
            "pct_of_total": round(len(categorized[i]) / total * 100, 1) if total else 0,
            "example_tickers": categorized[i][:10],
        }
    return result


def _business_day_count_safe(start_date, end_date):
    if not start_date or not end_date:
        return 0
    import datetime
    try:
        d = datetime.date.fromisoformat(start_date)
        e = datetime.date.fromisoformat(end_date)
    except ValueError:
        return 0
    n = 0
    while d <= e:
        if d.weekday() < 5:
            n += 1
        d += datetime.timedelta(days=1)
    return n


def generate_stage_report(conn, tickers, timing_records, price_source_name, stage_number):
    report = {"stage": stage_number, "price_source": price_source_name}

    # Resolve the ticker list to security_ids once -- every query below
    # filters by this set, not by everything in the database.
    placeholders = ",".join("?" for _ in tickers)
    sec_id_rows = conn.execute(
        f"SELECT security_id FROM securities WHERE primary_ticker IN ({placeholders})", tickers
    ).fetchall()
    sec_ids = [r["security_id"] for r in sec_id_rows]
    sec_id_placeholders = ",".join("?" for _ in sec_ids) if sec_ids else "NULL"

    # --- 1. Universe / security counts ---
    total_securities = len(sec_ids)
    resolved_via_cik = conn.execute(
        f"SELECT COUNT(*) c FROM securities WHERE identifier_quality='resolved' "
        f"AND security_id IN ({sec_id_placeholders})", sec_ids
    ).fetchone()["c"]
    unresolved_identity = conn.execute(
        f"SELECT COUNT(*) c FROM securities WHERE identifier_quality='unresolved' "
        f"AND security_id IN ({sec_id_placeholders})", sec_ids
    ).fetchone()["c"]
    report["security_counts"] = {
        "total_securities": total_securities,
        "resolved_via_known_identifier_cik": resolved_via_cik,
        "unresolved_identity": unresolved_identity,
    }

    # --- 2. Identity review queue (ticker reuse risk) ---
    review_rows = conn.execute(
        f"SELECT * FROM identity_review_queue WHERE ticker IN ({placeholders})", tickers
    ).fetchall()
    report["identity_review_queue"] = {
        "total_flags": len(review_rows),
        "unresolved_flags": sum(1 for r in review_rows if r["resolved"] == 0),
        "flagged_tickers": [dict(r) for r in review_rows],
    }

    # --- 3. Membership / conflicts ---
    total_membership = conn.execute(
        f"SELECT COUNT(*) c FROM index_membership WHERE raw_ticker IN ({placeholders})", tickers
    ).fetchone()["c"]
    conflicting = conn.execute(
        f"SELECT COUNT(*) c FROM index_membership WHERE confidence='conflicting' "
        f"AND raw_ticker IN ({placeholders})", tickers
    ).fetchone()["c"]
    unresolved_membership = conn.execute(
        f"SELECT COUNT(*) c FROM index_membership WHERE security_id IS NULL "
        f"AND raw_ticker IN ({placeholders})", tickers
    ).fetchone()["c"]
    report["membership"] = {
        "total_membership_claims": total_membership,
        "conflicting_claims": conflicting,
        "unresolved_membership_claims": unresolved_membership,
    }

    # --- 4. Price coverage ---
    total_price_rows = conn.execute(
        f"SELECT COUNT(*) c FROM prices WHERE adj_type='raw' AND security_id IN ({sec_id_placeholders})", sec_ids
    ).fetchone()["c"]
    date_range = conn.execute(
        f"SELECT MIN(date) mn, MAX(date) mx FROM prices WHERE adj_type='raw' "
        f"AND security_id IN ({sec_id_placeholders})", sec_ids
    ).fetchone()
    zero_data_securities = conn.execute(f"""
        SELECT s.primary_ticker FROM securities s
        WHERE s.security_id IN ({sec_id_placeholders})
        AND NOT EXISTS (SELECT 1 FROM prices p WHERE p.security_id = s.security_id AND p.adj_type='raw')
    """, sec_ids).fetchall()
    report["price_coverage"] = {
        "total_raw_price_rows": total_price_rows,
        "earliest_date": date_range["mn"],
        "latest_date": date_range["mx"],
        "securities_with_zero_price_data": [r["primary_ticker"] for r in zero_data_securities],
        "count_zero_price_data": len(zero_data_securities),
    }

    # --- 5. Data quality (OHLC validation results) ---
    # validate_ohlc() itself operates DB-wide (it's a mutating pass that
    # flags rows), which is correct -- it should check everything. But the
    # counts reported here must be scoped to this stage's tickers, or a
    # Stage 1 report would show Stage 2's flagged-row count.
    #
    # Also scope to adj_type='raw' only: a bad raw row propagates the same
    # OHLC violation into its derived split_adjusted/total_return copies
    # (scaling by a factor doesn't fix an already-broken high<close
    # relationship), so without this the same underlying problem gets
    # counted/sampled up to 3x -- once per representation.
    checks.validate_ohlc(conn)
    bad_ohlc_scoped = conn.execute(f"""
        SELECT COUNT(*) c FROM prices
        WHERE security_id IN ({sec_id_placeholders}) AND adj_type='raw'
        AND (high < low OR high < open OR high < close OR low > open OR low > close OR open <= 0 OR close <= 0)
    """, sec_ids).fetchone()["c"]
    neg_volume_scoped = conn.execute(f"""
        SELECT COUNT(*) c FROM prices WHERE security_id IN ({sec_id_placeholders}) AND adj_type='raw' AND volume < 0
    """, sec_ids).fetchone()["c"]
    report["data_quality"] = {"bad_ohlc_flagged": bad_ohlc_scoped, "negative_volume_flagged": neg_volume_scoped}
    flagged_sample = conn.execute(f"""
        SELECT s.primary_ticker, p.date, p.open, p.high, p.low, p.close, p.volume
        FROM prices p JOIN securities s ON p.security_id = s.security_id
        WHERE p.price_data_quality = 'suspicious' AND p.security_id IN ({sec_id_placeholders}) AND p.adj_type='raw'
        ORDER BY p.date LIMIT 30
    """, sec_ids).fetchall()
    report["data_quality"]["flagged_rows_sample"] = [dict(r) for r in flagged_sample]
    report["data_quality"]["note"] = ("Scoped to adj_type='raw' only -- the same root problem in raw data "
                                       "also propagates to its derived split_adjusted/total_return copies, "
                                       "which would inflate this count up to 3x for the same issue otherwise.")

    # --- 6. Dividends ---
    div_count = conn.execute(f"""
        SELECT COUNT(*) c FROM corporate_actions
        WHERE action_type IN ('dividend','special_dividend') AND security_id IN ({sec_id_placeholders})
    """, sec_ids).fetchone()["c"]
    div_securities = conn.execute(f"""
        SELECT COUNT(DISTINCT security_id) c FROM corporate_actions
        WHERE action_type IN ('dividend','special_dividend') AND security_id IN ({sec_id_placeholders})
    """, sec_ids).fetchone()["c"]
    report["dividends"] = {
        "total_dividend_events": div_count,
        "securities_with_dividends": div_securities,
        "securities_without_dividends": total_securities - div_securities,
    }

    # --- 7. Splits ---
    split_count = conn.execute(f"""
        SELECT COUNT(*) c FROM corporate_actions
        WHERE action_type IN ('split','reverse_split') AND security_id IN ({sec_id_placeholders})
    """, sec_ids).fetchone()["c"]
    split_rows = conn.execute(f"""
        SELECT s.primary_ticker, ca.ex_date, ca.ratio_or_value FROM corporate_actions ca
        JOIN securities s ON ca.security_id = s.security_id
        WHERE ca.action_type IN ('split','reverse_split') AND ca.security_id IN ({sec_id_placeholders})
        ORDER BY ca.ex_date
    """, sec_ids).fetchall()
    report["splits"] = {
        "total_split_events": split_count,
        "details": [dict(r) for r in split_rows],
    }

    # --- 8. Unsupported corporate actions ---
    unsupported_all = conn.execute(f"""
        SELECT primary_ticker, unsupported_corporate_action_note FROM securities
        WHERE has_unsupported_corporate_action=1 AND security_id IN ({sec_id_placeholders})
    """, sec_ids).fetchall()
    report["unsupported_corporate_actions"] = [dict(r) for r in unsupported_all]

    # --- 9. Delisted securities ---
    delisted = conn.execute(f"""
        SELECT primary_ticker, delisted_date, delisting_reason, delisting_confidence
        FROM securities WHERE active_flag=0 AND security_id IN ({sec_id_placeholders})
    """, sec_ids).fetchall()
    report["delisted_securities"] = [dict(r) for r in delisted]

    # --- 10. Survivorship-bias style breakdown ---
    with_data = conn.execute(f"""
        SELECT COUNT(DISTINCT s.security_id) c FROM securities s
        JOIN prices p ON p.security_id = s.security_id AND p.adj_type='raw'
        WHERE s.security_id IN ({sec_id_placeholders})
    """, sec_ids).fetchone()["c"]
    delisted_no_data = conn.execute(f"""
        SELECT COUNT(*) c FROM securities s
        WHERE s.active_flag = 0 AND s.security_id IN ({sec_id_placeholders})
        AND NOT EXISTS (SELECT 1 FROM prices p WHERE p.security_id = s.security_id AND p.adj_type='raw')
    """, sec_ids).fetchone()["c"]
    report["survivorship_style_breakdown"] = {
        "total_securities_in_universe": total_securities,
        "securities_with_usable_price_data": with_data,
        "known_delisted_with_no_price_data": delisted_no_data,
        "known_delisted_WITH_price_data": len(delisted) - delisted_no_data,
    }

    # --- 11. Point-in-time availability spot checks ---
    spot_check_dates = ["2016-01-04", "2020-01-02", "2023-01-03"]
    pit_results = []
    for ticker in tickers[:5]:  # spot-check first 5 tickers only, not exhaustive
        row = conn.execute("SELECT security_id FROM securities WHERE primary_ticker=?", (ticker,)).fetchone()
        if not row:
            continue
        for d in spot_check_dates:
            r = checks.check_point_in_time_availability(conn, row["security_id"], d, lookback_days=100)
            pit_results.append({"ticker": ticker, "as_of_date": d, "eligible": r["eligible"],
                                 "has_any_historical_data": r["has_any_historical_data"]})
    report["point_in_time_spot_checks"] = pit_results

    # --- 12. Timing / errors ---
    if timing_records:
        # Three timing categories, not two: on a resumed run, "freshly
        # fetched" can end up almost entirely made of tickers that were
        # fresh only because they have zero data (fast failures, retried
        # every run since they never accumulate rows to trigger the skip
        # check), not genuine successful fetches. Mixing those into the
        # "fresh fetch average" badly understates real per-security cost
        # (observed: 0.34s/security when real fetches cost ~2.3s/security --
        # a 7x understatement that would have corrupted the time estimate
        # for later stages if not caught).
        fresh_records = [r for r in timing_records if not r.get("skipped_resume")]
        skipped_records = [r for r in timing_records if r.get("skipped_resume")]
        fresh_with_data = [r for r in fresh_records if (r.get("bars_fetched") or 0) > 0]
        fresh_zero_data = [r for r in fresh_records if (r.get("bars_fetched") or 0) == 0]

        all_times = [r["total_seconds"] for r in timing_records if r.get("total_seconds") is not None]
        fresh_times = [r["total_seconds"] for r in fresh_records if r.get("total_seconds") is not None]
        fresh_with_data_times = [r["total_seconds"] for r in fresh_with_data if r.get("total_seconds") is not None]
        errors = [r for r in timing_records if r.get("error")]

        report["timing"] = {
            "securities_processed": len(timing_records),
            "securities_freshly_fetched": len(fresh_records),
            "securities_freshly_fetched_WITH_data": len(fresh_with_data),
            "securities_freshly_fetched_ZERO_data_fast_failures": len(fresh_zero_data),
            "securities_skipped_resume": len(skipped_records),
            "total_wall_seconds_including_skips": round(sum(all_times), 2),
            "total_wall_seconds_fresh_fetch_only": round(sum(fresh_times), 2),
            "avg_seconds_per_security_INCLUDING_skips_MISLEADING": (
                round(sum(all_times) / len(all_times), 4) if all_times else 0
            ),
            "avg_seconds_per_security_ALL_FRESH_STILL_MISLEADING_IF_MANY_ZERO_DATA": (
                round(sum(fresh_times) / len(fresh_times), 4) if fresh_times else 0
            ),
            "avg_seconds_per_security_FRESH_WITH_DATA_ONLY_HONEST_METRIC": (
                round(sum(fresh_with_data_times) / len(fresh_with_data_times), 4) if fresh_with_data_times else None
            ),
            "min_seconds_fresh_fetch": min(fresh_times) if fresh_times else 0,
            "max_seconds_fresh_fetch": max(fresh_times) if fresh_times else 0,
            "errors_encountered": len(errors),
            "error_details": errors,
        }
        # Extrapolation uses the fresh-with-data average -- the only honest
        # proxy for the cost of a real successful fetch. Falls back to the
        # all-fresh average with a warning if this run had zero successful
        # fresh fetches to measure from (can happen on a fully resumed run).
        avg = report["timing"]["avg_seconds_per_security_FRESH_WITH_DATA_ONLY_HONEST_METRIC"]
        if avg is None:
            avg = report["timing"]["avg_seconds_per_security_ALL_FRESH_STILL_MISLEADING_IF_MANY_ZERO_DATA"]
            report["timing"]["extrapolation_warning"] = (
                "No freshly-fetched security in this run actually returned data -- "
                "extrapolation below falls back to the all-fresh average, which may "
                "still be biased toward fast zero-data lookups. Treat with caution; "
                "prefer a stage's estimate that had genuine fresh successful fetches."
            )
        report["timing"]["extrapolated_estimates_based_on_fresh_WITH_DATA_avg"] = {
            "200_securities_minutes": round(avg * 200 / 60, 1),
            "500_securities_minutes": round(avg * 500 / 60, 1),
            "1200_securities_minutes": round(avg * 1200 / 60, 1),
        }

    # --- 13. Automated survivorship categorization (7-category) ---
    report["survivorship_categorization"] = generate_survivorship_categorization(conn, tickers)

    return report


def format_report_markdown(report):
    lines = [f"# Stage {report['stage']} Validation Report", f"Price source: {report['price_source']}", ""]

    lines.append("## Security counts")
    for k, v in report["security_counts"].items():
        lines.append(f"- {k}: {v}")

    lines.append("\n## Identity review queue")
    lines.append(f"- Total flags: {report['identity_review_queue']['total_flags']}")
    lines.append(f"- Unresolved: {report['identity_review_queue']['unresolved_flags']}")
    for f in report["identity_review_queue"]["flagged_tickers"]:
        lines.append(f"  - {f['ticker']}: {f['flag_reason']}")

    lines.append("\n## Membership")
    for k, v in report["membership"].items():
        lines.append(f"- {k}: {v}")

    lines.append("\n## Price coverage")
    for k, v in report["price_coverage"].items():
        if k != "securities_with_zero_price_data":
            lines.append(f"- {k}: {v}")
    lines.append(f"- securities with zero price data: {report['price_coverage']['securities_with_zero_price_data']}")

    lines.append("\n## Data quality")
    for k, v in report["data_quality"].items():
        if k not in ("flagged_rows_sample", "note"):
            lines.append(f"- {k}: {v}")
    if report["data_quality"].get("note"):
        lines.append(f"- note: {report['data_quality']['note']}")
    if report["data_quality"].get("flagged_rows_sample"):
        lines.append("- flagged rows sample:")
        for r in report["data_quality"]["flagged_rows_sample"]:
            lines.append(f"  - {r['primary_ticker']} {r['date']}: O={r['open']} H={r['high']} "
                         f"L={r['low']} C={r['close']} V={r['volume']}")

    lines.append("\n## Dividends")
    for k, v in report["dividends"].items():
        lines.append(f"- {k}: {v}")

    lines.append("\n## Splits")
    lines.append(f"- total split events: {report['splits']['total_split_events']}")
    for d in report["splits"]["details"]:
        lines.append(f"  - {d['primary_ticker']}: {d['ex_date']} ratio={d['ratio_or_value']}")

    lines.append("\n## Unsupported corporate actions")
    for u in report["unsupported_corporate_actions"]:
        lines.append(f"- {u['primary_ticker']}: {u['unsupported_corporate_action_note']}")

    lines.append("\n## Delisted securities")
    for d in report["delisted_securities"]:
        lines.append(f"- {d['primary_ticker']}: {d['delisted_date']} ({d['delisting_reason']}, "
                      f"confidence={d['delisting_confidence']})")

    lines.append("\n## Survivorship-style breakdown")
    for k, v in report["survivorship_style_breakdown"].items():
        lines.append(f"- {k}: {v}")

    lines.append("\n## Point-in-time availability spot checks")
    for p in report["point_in_time_spot_checks"]:
        lines.append(f"- {p['ticker']} as of {p['as_of_date']}: eligible={p['eligible']}, "
                      f"has_any_data={p['has_any_historical_data']}")

    if "timing" in report:
        lines.append("\n## Timing")
        for k, v in report["timing"].items():
            if k not in ("error_details",):
                lines.append(f"- {k}: {v}")
        if report["timing"]["error_details"]:
            lines.append("- error details:")
            for e in report["timing"]["error_details"]:
                lines.append(f"  - {e}")

    if "source_reliability_stats" in report:
        lines.append("\n## Source reliability (retries / rate-limits / errors)")
        for k, v in report["source_reliability_stats"].items():
            lines.append(f"- {k}: {v}")

    if "resumability" in report:
        lines.append("\n## Resumability")
        for k, v in report["resumability"].items():
            lines.append(f"- {k}: {v}")

    if "survivorship_categorization" in report:
        lines.append("\n## Survivorship categorization (automated, 7-category)")
        for cat, data in report["survivorship_categorization"].items():
            lines.append(f"- {cat}: {data['count']} ({data['pct_of_total']}%) e.g. {data['example_tickers']}")

    return "\n".join(lines)
