"""Data quality validation and point-in-time diagnostics.
Flags issues; never silently deletes or fills data."""


def validate_ohlc(conn):
    """Detect impossible OHLC relationships and non-positive prices/volume.
    Flags rows in place (price_data_quality='suspicious'), never deletes."""
    bad_ohlc = conn.execute("""
        SELECT price_id FROM prices
        WHERE high < low OR high < open OR high < close
           OR low > open OR low > close OR open <= 0 OR close <= 0
    """).fetchall()
    for row in bad_ohlc:
        conn.execute("UPDATE prices SET price_data_quality='suspicious' WHERE price_id=?", (row["price_id"],))
    neg_vol = conn.execute("SELECT price_id FROM prices WHERE volume < 0").fetchall()
    for row in neg_vol:
        conn.execute("UPDATE prices SET price_data_quality='suspicious' WHERE price_id=?", (row["price_id"],))
    conn.commit()
    return {"bad_ohlc_flagged": len(bad_ohlc), "negative_volume_flagged": len(neg_vol)}


def detect_missing_dates(conn, security_id, expected_dates):
    """expected_dates: list of ISO business-day strings the security SHOULD
    have a price for (typically its membership-window intersected with
    overall trading calendar). Returns missing dates — does NOT fill them."""
    have = set(
        r["date"] for r in conn.execute(
            "SELECT DISTINCT date FROM prices WHERE security_id=?", (security_id,)
        ).fetchall()
    )
    missing = [d for d in expected_dates if d not in have]
    return missing


def check_duplicate_securities_by_name(conn):
    rows = conn.execute("""
        SELECT name, COUNT(*) c FROM securities WHERE name IS NOT NULL
        GROUP BY name HAVING c > 1
    """).fetchall()
    return [dict(r) for r in rows]


def detect_and_flag_conflicts(conn, index_name):
    """
    Find membership claims for the same raw_ticker + index whose active
    windows overlap but whose effective_date/removal_date DISAGREE across
    sources. Mark all involved rows confidence='conflicting'.

    This never deletes or silently merges — it only relabels rows so that
    downstream universe reconstruction can choose to exclude them rather
    than pick one arbitrarily. Idempotent: re-running does not double-flag
    or un-flag rows other than recomputing the same set.
    """
    rows = conn.execute(
        "SELECT * FROM index_membership WHERE index_name=? AND confidence != 'conflicting'",
        (index_name,),
    ).fetchall()

    by_ticker = {}
    for r in rows:
        by_ticker.setdefault(r["raw_ticker"], []).append(dict(r))

    flagged_ids = set()
    for ticker, claims in by_ticker.items():
        if len(claims) < 2:
            continue
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                a, b = claims[i], claims[j]
                if a["source_id"] == b["source_id"]:
                    continue  # same source repeating itself is not a conflict
                a_end = a["removal_date"] or "9999-12-31"
                b_end = b["removal_date"] or "9999-12-31"
                overlaps = a["effective_date"] <= b_end and b["effective_date"] <= a_end
                agrees = (a["effective_date"] == b["effective_date"] and
                          (a["removal_date"] or None) == (b["removal_date"] or None))
                if overlaps and not agrees:
                    flagged_ids.add(a["membership_id"])
                    flagged_ids.add(b["membership_id"])

    for mid in flagged_ids:
        conn.execute(
            "UPDATE index_membership SET confidence='conflicting', "
            "verification_status = verification_status || ' [AUTO-FLAGGED: conflicts with another source]' "
            "WHERE membership_id=? AND confidence != 'conflicting'",
            (mid,),
        )
    conn.commit()
    return {"claims_examined": len(rows), "newly_flagged_conflicting": len(flagged_ids)}


def check_point_in_time_availability(conn, security_id, as_of_date, lookback_days=252,
                                      min_completeness_pct=0.95, near_date_tolerance_days=5,
                                      adj_type="raw"):
    """
    Reusable point-in-time eligibility check for eventual backtesting.
    Answers: "For security X, as of date Y, with required lookback N
    trading days, is the security eligible for modelling?"

    Every query in this function filters on `date <= as_of_date` -- no
    code path here can see or use a price observation dated after
    as_of_date. That's the one property this function must never violate,
    and it's directly tested by
    test_point_in_time_availability_never_sees_future_data.

    Returns a dict with each of 5 dimensions evaluated independently so a
    caller can see why something failed:
        A. has_any_historical_data
        B. has_data_on_or_around_date  (within near_date_tolerance_days before as_of_date)
        C. has_sufficient_lookback     (calendar span covers lookback_days)
        D. completeness_over_lookback  (% of expected trading days present)
        E. has_liquidity_info          (volume data present, not just price)
        eligible: bool -- True only if all of A-E pass
    """
    # A. Any historical data at all, strictly before/on as_of_date
    any_data = conn.execute(
        "SELECT COUNT(*) c FROM prices WHERE security_id=? AND adj_type=? AND date <= ?",
        (security_id, adj_type, as_of_date),
    ).fetchone()["c"]
    has_any_historical_data = any_data > 0

    if not has_any_historical_data:
        return {
            "security_id": security_id, "as_of_date": as_of_date, "lookback_days_required": lookback_days,
            "has_any_historical_data": False, "has_data_on_or_around_date": False,
            "has_sufficient_lookback": False, "completeness_over_lookback": 0.0,
            "completeness_passed": False, "has_liquidity_info": False, "eligible": False,
            "reason": "no_price_data_on_or_before_as_of_date",
        }

    # B. Data on/around the requested date (within tolerance, still <= as_of_date)
    near_date_rows = conn.execute(
        "SELECT COUNT(*) c FROM prices WHERE security_id=? AND adj_type=? AND date <= ? "
        "AND date > date(?, '-' || ? || ' days')",
        (security_id, adj_type, as_of_date, as_of_date, near_date_tolerance_days),
    ).fetchone()["c"]
    has_data_on_or_around_date = near_date_rows > 0

    # C/D. Lookback window: strictly bounded by as_of_date on the upper end
    window_rows = conn.execute(
        "SELECT date, volume FROM prices WHERE security_id=? AND adj_type=? AND date <= ? ORDER BY date DESC LIMIT ?",
        (security_id, adj_type, as_of_date, lookback_days),
    ).fetchall()
    actual_days_found = len(window_rows)
    has_sufficient_lookback = actual_days_found >= lookback_days

    if actual_days_found > 0:
        earliest_in_window = window_rows[-1]["date"]
        # Expected trading days = business days between earliest observed and as_of_date
        expected = _business_day_count(earliest_in_window, as_of_date)
        completeness = actual_days_found / expected if expected > 0 else 0.0
    else:
        completeness = 0.0
    completeness_passed = completeness >= min_completeness_pct

    # E. Liquidity info: do we have volume (not null, not all zero) over the window
    has_liquidity_info = any(r["volume"] is not None and r["volume"] > 0 for r in window_rows)

    eligible = (has_any_historical_data and has_data_on_or_around_date and
                has_sufficient_lookback and completeness_passed and has_liquidity_info)

    return {
        "security_id": security_id, "as_of_date": as_of_date, "lookback_days_required": lookback_days,
        "has_any_historical_data": has_any_historical_data,
        "has_data_on_or_around_date": has_data_on_or_around_date,
        "has_sufficient_lookback": has_sufficient_lookback,
        "actual_lookback_days_found": actual_days_found,
        "completeness_over_lookback": round(completeness, 4),
        "completeness_passed": completeness_passed,
        "has_liquidity_info": has_liquidity_info,
        "eligible": eligible,
    }


def _business_day_count(start_date, end_date):
    import datetime
    d = datetime.date.fromisoformat(start_date)
    e = datetime.date.fromisoformat(end_date)
    n = 0
    while d <= e:
        if d.weekday() < 5:
            n += 1
        d += datetime.timedelta(days=1)
    return n


def apply_universe_filters(conn, security_id, config, adj_type="raw"):
    """
    Evaluate one security against config["universe_filters"]. Returns a
    dict with pass/fail per criterion plus the measured values, not just
    a single boolean, so a human can see why something was excluded.

    Thresholds come only from config.yaml and should never be tuned based
    on backtest results.
    """
    f = config["universe_filters"]
    rows = conn.execute(
        "SELECT date, close, volume FROM prices WHERE security_id=? AND adj_type=? ORDER BY date",
        (security_id, adj_type),
    ).fetchall()

    if not rows:
        return {
            "security_id": security_id, "passed": False, "reason": "no_price_data",
            "historical_days": 0,
        }

    historical_days = len(rows)
    latest_price = rows[-1]["close"]
    avg_dollar_volume = sum((r["close"] or 0) * (r["volume"] or 0) for r in rows) / historical_days

    checks_result = {
        "min_price_usd": {"threshold": f["min_price_usd"], "value": round(latest_price, 2),
                           "passed": latest_price >= f["min_price_usd"]},
        "min_avg_dollar_volume_usd": {"threshold": f["min_avg_dollar_volume_usd"],
                                       "value": round(avg_dollar_volume, 2),
                                       "passed": avg_dollar_volume >= f["min_avg_dollar_volume_usd"]},
        "min_historical_days": {"threshold": f["min_historical_days"], "value": historical_days,
                                 "passed": historical_days >= f["min_historical_days"]},
    }
    passed = all(c["passed"] for c in checks_result.values())
    return {
        "security_id": security_id,
        "passed": passed,
        "checks": checks_result,
        "historical_days": historical_days,
    }


def reconstruct_universe(conn, index_name, as_of_date, confidence_mode="verified_and_unverified"):
    """
    Returns a point-in-time universe report:
        - total historical constituents (per membership claims active on date)
        - constituents with usable price data
        - known delisted/unavailable securities
        - other unresolved securities

    confidence_mode: 'verified_only' | 'verified_and_unverified'
    'conflicting' rows are always excluded from the resolved universe count
    (they can't be resolved automatically) but are reported separately.
    """
    if confidence_mode == "verified_only":
        conf_clause = "confidence = 'verified'"
    else:
        conf_clause = "confidence IN ('verified','unverified')"

    active_claims = conn.execute(f"""
        SELECT * FROM index_membership
        WHERE index_name = ?
          AND effective_date <= ?
          AND (removal_date IS NULL OR removal_date > ?)
          AND {conf_clause}
    """, (index_name, as_of_date, as_of_date)).fetchall()

    conflicting_claims = conn.execute("""
        SELECT * FROM index_membership
        WHERE index_name = ? AND effective_date <= ? AND (removal_date IS NULL OR removal_date > ?)
          AND confidence = 'conflicting'
    """, (index_name, as_of_date, as_of_date)).fetchall()

    # De-duplicate by resolved security (or raw_ticker if unresolved): multiple
    # sources independently confirming the same membership fact should count
    # as one constituent, not one per corroborating source. If sources
    # disagree, they were already routed to 'conflicting' and excluded above.
    seen = set()
    distinct_claims = []
    for claim in active_claims:
        key = claim["security_id"] if claim["security_id"] is not None else f"unresolved:{claim['raw_ticker']}"
        if key in seen:
            continue
        seen.add(key)
        distinct_claims.append(claim)

    total = len(distinct_claims)
    usable_price = 0
    delisted_unavailable = 0
    unresolved = 0
    detail = []

    for claim in distinct_claims:
        sec_id = claim["security_id"]
        if sec_id is None:
            unresolved += 1
            detail.append({"ticker": claim["raw_ticker"], "status": "unresolved_identifier"})
            continue

        sec = conn.execute("SELECT * FROM securities WHERE security_id=?", (sec_id,)).fetchone()
        price_count = conn.execute(
            "SELECT COUNT(*) c FROM prices WHERE security_id=? AND date <= ?", (sec_id, as_of_date)
        ).fetchone()["c"]

        if price_count > 0:
            usable_price += 1
            detail.append({"ticker": claim["raw_ticker"], "status": "usable_price_data"})
        elif sec["active_flag"] == 0 or sec["delisted_date"]:
            delisted_unavailable += 1
            detail.append({"ticker": claim["raw_ticker"], "status": "known_delisted_unavailable_price"})
        else:
            delisted_unavailable += 1
            detail.append({"ticker": claim["raw_ticker"], "status": "known_constituent_no_price_data"})

    return {
        "index_name": index_name,
        "as_of_date": as_of_date,
        "confidence_mode": confidence_mode,
        "total_historical_constituents": total,
        "constituents_with_usable_price_data": usable_price,
        "known_delisted_or_unavailable": delisted_unavailable,
        "unresolved_identifiers": unresolved,
        "conflicting_claims_excluded": len(conflicting_claims),
        "detail": detail,
    }
