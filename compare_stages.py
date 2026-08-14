"""
Compares two stage JSON reports (e.g. STAGE1_REPORT.json vs STAGE2_REPORT.json)
and produces a side-by-side comparison table.

Run: python3 compare_stages.py STAGE1_REPORT.json STAGE2_REPORT.json
"""
import json
import sys


def load(path):
    with open(path, "r") as f:
        return json.load(f)


def compare(report1, report2):
    rows = []

    def add(label, v1, v2, note=""):
        try:
            change = v2 - v1 if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) else "n/a"
        except TypeError:
            change = "n/a"
        rows.append((label, v1, v2, change, note))

    add("Securities", report1["security_counts"]["total_securities"], report2["security_counts"]["total_securities"])
    add("Resolved via known CIK", report1["security_counts"]["resolved_via_known_identifier_cik"],
        report2["security_counts"]["resolved_via_known_identifier_cik"])
    add("Unresolved identity", report1["security_counts"]["unresolved_identity"],
        report2["security_counts"]["unresolved_identity"])
    add("Identity review flags", report1["identity_review_queue"]["total_flags"],
        report2["identity_review_queue"]["total_flags"])
    add("Membership claims", report1["membership"]["total_membership_claims"],
        report2["membership"]["total_membership_claims"])
    add("Conflicting membership claims", report1["membership"]["conflicting_claims"],
        report2["membership"]["conflicting_claims"])
    add("Raw price rows", report1["price_coverage"]["total_raw_price_rows"],
        report2["price_coverage"]["total_raw_price_rows"])
    add("Securities with zero price data", report1["price_coverage"]["count_zero_price_data"],
        report2["price_coverage"]["count_zero_price_data"])
    add("Bad OHLC rows flagged", report1["data_quality"]["bad_ohlc_flagged"],
        report2["data_quality"]["bad_ohlc_flagged"])
    add("Dividend events", report1["dividends"]["total_dividend_events"],
        report2["dividends"]["total_dividend_events"])
    add("Securities without dividends", report1["dividends"]["securities_without_dividends"],
        report2["dividends"]["securities_without_dividends"])
    add("Split events", report1["splits"]["total_split_events"], report2["splits"]["total_split_events"])
    add("Unsupported corporate action flags", len(report1["unsupported_corporate_actions"]),
        len(report2["unsupported_corporate_actions"]))
    add("Delisted securities", len(report1["delisted_securities"]), len(report2["delisted_securities"]))

    if "timing" in report1 and "timing" in report2:
        t1, t2 = report1["timing"], report2["timing"]
        # Timing field names changed twice as earlier versions of the
        # ingestion script were found to dilute the average with skipped
        # or zero-data lookups. Prefer the most-corrected field name
        # available and fall back to older names for old saved reports.
        def get_timing(t, *names):
            for n in names:
                if n in t:
                    return t[n]
            return None

        wall_1 = get_timing(t1, "total_wall_seconds_fresh_fetch_only", "total_wall_seconds")
        wall_2 = get_timing(t2, "total_wall_seconds_fresh_fetch_only", "total_wall_seconds")
        add("Total wall seconds (fresh fetch only if available, else raw)", wall_1, wall_2)

        avg_1 = get_timing(t1, "avg_seconds_per_security_FRESH_WITH_DATA_ONLY_HONEST_METRIC",
                            "avg_seconds_per_security_FRESH_FETCH_ONLY", "avg_seconds_per_security")
        avg_2 = get_timing(t2, "avg_seconds_per_security_FRESH_WITH_DATA_ONLY_HONEST_METRIC",
                            "avg_seconds_per_security_FRESH_FETCH_ONLY", "avg_seconds_per_security")
        note = "Stage 1 baseline was ~2.3s/security (genuine fresh-with-data fetches)."
        if "avg_seconds_per_security_FRESH_WITH_DATA_ONLY_HONEST_METRIC" not in t1 or \
           "avg_seconds_per_security_FRESH_WITH_DATA_ONLY_HONEST_METRIC" not in t2:
            note += (" Note: at least one report predates the fresh-with-data fix, so its average may "
                     "still be diluted by resume-skips or fast zero-data lookups and isn't a reliable "
                     "basis for extrapolating full-universe ingestion time. Re-run that stage to regenerate.")
        add("Avg seconds/security", round(avg_1, 3) if isinstance(avg_1, (int, float)) else avg_1,
            round(avg_2, 3) if isinstance(avg_2, (int, float)) else avg_2, note=note)
        add("Errors encountered", t1.get("errors_encountered"), t2.get("errors_encountered"))

    if "source_reliability_stats" in report1 and "source_reliability_stats" in report2:
        for k in report1["source_reliability_stats"]:
            add(f"Reliability: {k}", report1["source_reliability_stats"][k],
                report2["source_reliability_stats"].get(k))

    if "resumability" in report2:
        rows.append(("Tickers skipped (resumed)", "n/a", report2["resumability"]["tickers_skipped_as_already_ingested"],
                     "n/a", "Stage 2 built on Stage 1's DB -- this many were NOT re-fetched"))

    return rows


def format_comparison_markdown(rows):
    lines = ["# Stage 1 vs Stage 2 Comparison\n", "| Metric | Stage 1 | Stage 2 | Change | Note |",
             "|---|---|---|---|---|"]
    for label, v1, v2, change, note in rows:
        lines.append(f"| {label} | {v1} | {v2} | {change} | {note} |")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 compare_stages.py STAGE1_REPORT.json STAGE2_REPORT.json")
        sys.exit(1)
    r1 = load(sys.argv[1])
    r2 = load(sys.argv[2])
    rows = compare(r1, r2)
    md = format_comparison_markdown(rows)
    print(md)
    with open("STAGE1_VS_STAGE2_COMPARISON.md", "w") as f:
        f.write(md)
    print("\n\nSaved to STAGE1_VS_STAGE2_COMPARISON.md")
