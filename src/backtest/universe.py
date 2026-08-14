"""
Phase 3: the one authoritative universe-construction interface.

Every benchmark and strategy gets its eligible securities through
build_eligible_universe() and nothing else -- no querying
index_membership or prices directly to determine eligibility. This is
the single biggest guard against survivorship/look-ahead bugs creeping
in strategy by strategy.

This doesn't reimplement point-in-time logic; it composes existing,
already-tested primitives:
    - src.validation.checks.reconstruct_universe()     (who was a member)
    - src.validation.checks.check_point_in_time_availability()  (usable data)
    - src.validation.checks.apply_universe_filters()    (price/volume/history)
plus a Phase-3-specific data_quality_policy layer (identity resolution
quality, identity_review_queue flags, and a security-level bad-OHLC-row
severity signal found during a later data sweep).
"""
from src.validation import checks


def _bad_ohlc_row_pct(conn, security_id):
    """Security-level severity signal: the row-level
    price_data_quality='suspicious' flag alone doesn't distinguish a
    security with 17% bad rows from one with 1.4%. Computed on the fly,
    not persisted to securities."""
    row = conn.execute(
        """SELECT
               COUNT(*) AS total,
               SUM(CASE WHEN price_data_quality='suspicious' THEN 1 ELSE 0 END) AS bad
           FROM prices WHERE security_id=? AND adj_type='raw'""",
        (security_id,),
    ).fetchone()
    if not row or not row["total"]:
        return 0.0
    return (row["bad"] or 0) / row["total"]


def build_eligible_universe(conn, as_of_date, data_quality_policy, predeclared_filters=None,
                             universe_definition="SP500", lookback_days=252,
                             price_start=None, price_end=None, adj_type="total_return"):
    """
    Returns (eligible: list[int security_id], exclusion_report: list[dict]).

    data_quality_policy: one of the dicts under config.yaml's
        backtest.data_quality_policies (STRICT / PERMISSIVE / custom).
    predeclared_filters: config.yaml's existing universe_filters block
        (min_price_usd, min_avg_dollar_volume_usd, min_historical_days),
        unchanged from Phase 1/2.

    Every excluded candidate gets exactly one exclusion_report entry with
    a specific reason -- this feeds the per-run coverage report. Nothing
    here mutates the database.
    """
    predeclared_filters = predeclared_filters or {}
    membership = checks.reconstruct_universe(conn, universe_definition, as_of_date,
                                              confidence_mode="verified_and_unverified")

    eligible = []
    exclusion_report = []

    for item in membership["detail"]:
        ticker = item["ticker"]

        if item["status"] == "unresolved_identifier":
            exclusion_report.append({
                "ticker": ticker, "security_id": None, "reason": "identity_unresolved_no_security_row",
            })
            continue

        row = conn.execute("SELECT security_id, identifier_quality FROM securities WHERE primary_ticker=?",
                            (ticker,)).fetchone()
        if not row:
            exclusion_report.append({"ticker": ticker, "security_id": None, "reason": "security_row_not_found"})
            continue
        sec_id = row["security_id"]

        # --- identity-quality gate (policy-driven, never hardcoded) ---
        if data_quality_policy.get("exclude_unresolved_identity") and row["identifier_quality"] == "unresolved":
            exclusion_report.append({"ticker": ticker, "security_id": sec_id, "reason": "identity_unresolved_excluded_by_policy"})
            continue

        if data_quality_policy.get("exclude_identity_review_flagged"):
            flagged = conn.execute(
                "SELECT COUNT(*) c FROM identity_review_queue WHERE ticker=? AND resolved=0", (ticker,)
            ).fetchone()["c"] > 0
            if flagged:
                exclusion_report.append({"ticker": ticker, "security_id": sec_id, "reason": "identity_review_flagged_excluded_by_policy"})
                continue

        # --- point-in-time data availability (existing, richer than a bare "has data" check) ---
        pit = checks.check_point_in_time_availability(
            conn, sec_id, as_of_date, lookback_days=lookback_days,
            min_completeness_pct=data_quality_policy.get("min_completeness_pct", 0.95),
            adj_type=adj_type,
        )
        if not pit["has_any_historical_data"]:
            exclusion_report.append({"ticker": ticker, "security_id": sec_id, "reason": "provider_empty_no_data"})
            continue
        if data_quality_policy.get("require_full_history") and not pit["eligible"]:
            exclusion_report.append({"ticker": ticker, "security_id": sec_id, "reason": "insufficient_full_history_required_by_policy",
                                      "detail": pit})
            continue
        if not pit["completeness_passed"]:
            exclusion_report.append({"ticker": ticker, "security_id": sec_id, "reason": "below_min_completeness_pct",
                                      "detail": pit})
            continue
        if not pit["has_sufficient_lookback"]:
            exclusion_report.append({"ticker": ticker, "security_id": sec_id, "reason": "insufficient_lookback",
                                      "detail": pit})
            continue
        if not pit["has_liquidity_info"]:
            exclusion_report.append({"ticker": ticker, "security_id": sec_id, "reason": "no_liquidity_info"})
            continue

        # --- severe OHLC severity (security-level, YHD-sweep-derived signal) ---
        threshold = data_quality_policy.get("severe_ohlc_bad_row_pct_threshold")
        bad_pct = _bad_ohlc_row_pct(conn, sec_id)
        if data_quality_policy.get("exclude_severe_ohlc_flagged") and threshold is not None and bad_pct > threshold:
            exclusion_report.append({"ticker": ticker, "security_id": sec_id, "reason": "severe_ohlc_excluded_by_policy",
                                      "bad_ohlc_row_pct": bad_pct})
            continue

        # --- existing Phase 1/2 price/volume/history filters, unchanged ---
        if predeclared_filters:
            filt = checks.apply_universe_filters(conn, sec_id, {"universe_filters": predeclared_filters}, adj_type=adj_type)
            if not filt["passed"]:
                exclusion_report.append({"ticker": ticker, "security_id": sec_id, "reason": "failed_predeclared_filters",
                                          "detail": filt["checks"]})
                continue

        eligible.append(sec_id)

    return eligible, exclusion_report
