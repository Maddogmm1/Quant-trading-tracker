"""
Stage-agnostic real-data ingestion pipeline.

run_stage(stage, price_source, ...) below takes a ticker list and a
price_source object. Nothing in this function knows or cares whether that
list has 50 or 1,200 tickers, or whether the price_source is real
(YFinancePriceSource) or synthetic (for dry-run testing). Scaling to
Stage 2/3/4 means changing the ticker list via src/universe/stage_universe.py
-- this function itself doesn't change.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) if __name__ == "__main__" else ".")

from src.database.db import init_db, now_iso
from src.universe.membership_sources import SP500GithubWikipediaParser
from src.universe import identity_resolution as idres
from src.universe.stage_universe import get_stage_tickers, get_stage1_category_map, get_stage2_selection_detail
from src.ingestion import pipeline as pl
from src.ingestion.adjustments import compute_split_adjusted_for_all, compute_total_return_for_all, classify_split_ratio
from src.validation import checks
from src.validation.stage_report import generate_stage_report, format_report_markdown

# All data/schema paths are resolved relative to THIS FILE's location, not
# the process's current working directory. Found via a real bug: the
# previous bare relative paths (e.g. "data/database/quant_trader_stage.db")
# meant that invoking this module (directly, or via run_stage3_real.py)
# from a different working directory would silently open or CREATE a
# different physical database/schema/CSV file with no error -- defeating
# every idempotency/dedup check that assumes "same DB across runs" (this is
# the likely root cause of identity_review_queue picking up duplicate flags
# across two Stage 3 attempts). PROJECT_ROOT is the directory containing
# this script, so these resolve identically regardless of cwd.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "quant_trader_stage.db")
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "src", "database", "schema.sql")
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "sp500-master", "sp500_ticker_start_end.csv")
KNOWN_IDENTIFIERS_PATH = os.path.join(PROJECT_ROOT, "data", "reference", "known_identifiers_seed.csv")
KNOWN_RENAMES_PATH = os.path.join(PROJECT_ROOT, "data", "reference", "known_ticker_renames_seed.csv")

PRICE_START = "2010-01-01"  # deliberately longer than Phase 1's window, to
PRICE_END = "2023-12-31"    # actually exercise long-history behavior


def run_stage(stage, price_source, db_path=None, reset=True, force_reset=False,
              price_start=PRICE_START, price_end=PRICE_END, tickers_override=None,
              skip_if_already_ingested=True, force_tickers=None):
    """
    tickers_override: explicit ticker list, bypassing get_stage_tickers().
        Used for (a) testing resumability, (b) retrying only specific
        failed securities independently of a full stage run.
    skip_if_already_ingested: if a security already has raw price rows in
        the target DB, skip re-fetching its prices/dividends/splits. This
        is what makes a killed/restarted run resumable without wasting
        API calls re-downloading tickers that already succeeded. Best-
        effort: it checks for ANY existing raw price rows, not a proof of
        complete coverage -- a partially-ingested ticker will look "done"
        and get skipped too. The validation report's coverage stats are
        what actually catch that, not this check.
    force_tickers: set of tickers to ALWAYS re-fetch regardless of
        skip_if_already_ingested -- e.g. tickers whose price fetch
        succeeded but whose dividend/split processing crashed partway
        (real Stage 2 case: PZE, SVU, TWX had prices but never got their
        dividends/splits processed due to the malformed-data bug). The
        resume-skip check alone can't distinguish "fully done" from
        "prices done, dividends/splits never attempted" since it only
        checks for the presence of raw price rows -- this is the
        explicit override for that gap.
    """
    force_tickers = force_tickers or set()
    db_path = db_path or DB_PATH
    conn = init_db(db_path, SCHEMA_PATH, reset=reset, force=force_reset)

    tickers = tickers_override if tickers_override is not None else get_stage_tickers(stage, CSV_PATH)
    category_map = get_stage1_category_map() if stage == 1 else {}
    print(f"Stage {stage}: {len(tickers)} tickers selected.")

    # --- Membership (real data, same as Phase 1) ---
    parser = SP500GithubWikipediaParser()
    membership_records = parser.parse(CSV_PATH, ticker_whitelist=set(tickers))
    print(f"Parsed {len(membership_records)} real membership records for {len(tickers)} tickers.")

    # --- Identity resolution (CIK-preferred, same module as Phase 1) ---
    id_reg_result = idres.load_known_identifiers(conn, KNOWN_IDENTIFIERS_PATH)
    rename_reg_result = pl.load_known_renames(conn, KNOWN_RENAMES_PATH)
    print(f"Loaded identifier registry: {id_reg_result}; rename registry: {rename_reg_result}")

    security_resolver = {}
    identity_methods = {}
    for ticker in tickers:
        sec_id, created, method = idres.resolve_or_create_security(conn, ticker)
        security_resolver[ticker] = sec_id
        identity_methods[ticker] = method

    m_result = pl.ingest_membership_records(conn, membership_records, security_resolver)
    print(f"Membership ingest: {m_result}")

    # --- Delisting metadata + unsupported corporate action flags ---
    # Reusing the same sourced citations established during Phase 1 rather
    # than re-verifying facts already confirmed. Only applied to tickers
    # actually present in this stage.
    delisted_meta = {
        "MON": ("2018-06-07", "acquired", "verified",
                "SEC EDGAR DEFM14A (CIK 1110783): stock delisted from NYSE and deregistered under the "
                "Exchange Act upon Bayer acquisition closing. "
                "https://www.sec.gov/Archives/edgar/data/1110783/000119312516765991/d252304ddefm14a.htm"),
        "ABMD": ("2022-12-22", "acquired", "verified",
                 "SEC EDGAR 8-K (Johnson & Johnson, CIK 200406): tender offer for Abiomed completed "
                 "2022-12-22, converting shares to cash + CVR. "
                 "https://www.sec.gov/Archives/edgar/data/200406/000119312522311072/d428734dex991.htm"),
        "AABA": ("2019-10-02", "dissolved", "verified",
                 "Yahoo! Inc renamed to Altaba Inc in 2017 but kept trading under ticker AABA for two "
                 "more years; Altaba stopped trading 2019-10-02 and filed its certificate of dissolution "
                 "2019-10-04. https://en.wikipedia.org/wiki/Altaba"),
        "AAMRQ": ("2012-01-30", "bankrupt", "verified",
                  "AMR Corp (ticker AMR on NYSE) filed Chapter 11 on 2011-11-29, NYSE trading suspended "
                  "2012-01-05, formally delisted by the SEC 2012-01-30; traded OTC under AAMRQ until its "
                  "2013 merger into American Airlines Group (ticker AAL). CAVEAT: ticker 'AAMRQ' did not "
                  "exist during AMR's actual 1996-2003 S&P 500 membership window -- see BACKLOG.md item 2. "
                  "https://www.sec.gov/Archives/edgar/data/6201/000000620113000023/amr-10kx20121231.htm"),
    }
    for ticker, (delisted_date, reason, confidence, source) in delisted_meta.items():
        if ticker in security_resolver:
            conn.execute(
                "UPDATE securities SET active_flag=0, delisted_date=?, delisting_reason=?, "
                "delisting_confidence=?, delisting_source=?, updated_at=? WHERE security_id=?",
                (delisted_date, reason, confidence, source, now_iso(), security_resolver[ticker]),
            )
    if "ABBV" in security_resolver:
        pl.flag_unsupported_corporate_action(
            conn, security_resolver["ABBV"], "spinoff", "2013-01-01",
            "AbbVie spun off from Abbott Laboratories (ABT). Source: Abbott 8-K Item 2.01, "
            "https://www.sec.gov/Archives/edgar/data/0000001800/000110465913001016/a13-2169_18k.htm",
            "SEC EDGAR (Abbott Laboratories 8-K)", "A",
        )
    conn.commit()

    # --- Price / dividend / split ingestion with timing instrumentation ---
    timing_records = []
    print(f"\nFetching real price data via {price_source.source_name} for {len(tickers)} tickers...")
    print(f"Date range: {price_start} to {price_end}\n")

    for i, ticker in enumerate(tickers):
        sec_id = security_resolver[ticker]
        t0 = time.time()
        error = None
        bars_count, div_count, split_count = 0, 0, 0

        if skip_if_already_ingested and ticker not in force_tickers:
            existing = conn.execute(
                "SELECT COUNT(*) c FROM prices WHERE security_id=? AND adj_type='raw'", (sec_id,)
            ).fetchone()["c"]
            if existing > 0:
                total_seconds = time.time() - t0
                timing_records.append({
                    "ticker": ticker, "price_seconds": 0, "dividend_seconds": 0, "split_seconds": 0,
                    "total_seconds": round(total_seconds, 3), "bars_fetched": existing,
                    "dividends_fetched": None, "splits_fetched": None, "error": None, "skipped_resume": True,
                })
                print(f"  [{i+1}/{len(tickers)}] {ticker:8s} SKIPPED (already has {existing} raw price rows -- resume)")
                continue

        try:
            t_price_start = time.time()
            price_result = pl.ingest_prices_with_rename_fallback(
                conn, sec_id, ticker, price_start, price_end, price_source, adj_type="raw"
            )
            price_seconds = time.time() - t_price_start
            bars_count = price_result["bars_fetched"]

            # Persist the 4-state status taxonomy -- one row per ticker per
            # run, not deduplicated, so repeated attempts across resumes
            # are all visible in the audit trail.
            status = price_source.last_call_status or (
                "SUCCESS_WITH_DATA" if bars_count > 0 else "SUCCESS_EMPTY_PROVIDER"
            )
            redirect_note = f" (redirected to {price_result['redirect_used']})" if price_result.get("redirect_used") else ""
            conn.execute(
                """INSERT INTO ingestion_attempts
                   (ticker, security_id, provider, requested_start, requested_end,
                    attempts, status, rows_returned, error_detail, attempted_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (ticker, sec_id, price_source.source_name, price_start, price_end,
                 price_source.last_call_attempts or 1, status, bars_count,
                 (price_source.last_call_error_detail or "") + redirect_note or None, now_iso()),
            )
            conn.commit()

            t_div_start = time.time()
            divs = price_source.fetch_dividends(ticker, price_start, price_end)
            for ex_date, amount in divs:
                pl.ingest_corporate_action(conn, sec_id, "dividend", ex_date, amount,
                                            f"${amount} dividend", price_source.source_name, "C",
                                            quality="unverified")
            div_seconds = time.time() - t_div_start
            div_count = len(divs)

            t_split_start = time.time()
            splits = price_source.fetch_splits(ticker, price_start, price_end)
            for ex_date, ratio in splits:
                classification = classify_split_ratio(ratio)
                if classification == "genuine":
                    action_type = "split" if ratio > 1 else "reverse_split"
                    pl.ingest_corporate_action(conn, sec_id, action_type, ex_date, ratio,
                                                f"{ratio}x {action_type}", price_source.source_name, "C",
                                                quality="unverified")
                else:
                    # Non-round ratio -- much more likely a spinoff-driven
                    # residual price-adjustment factor than a genuine share
                    # split (see src/ingestion/adjustments.py). Record it
                    # for the audit trail, but with a quality flag that
                    # EXCLUDES it from the split-adjustment math, and flag
                    # the security so a human can investigate what the
                    # underlying corporate event actually was.
                    action_type = "split" if ratio > 1 else "reverse_split"
                    src_id = pl.get_or_create_source(conn, price_source.source_name, "C")
                    existing_ca = conn.execute(
                        "SELECT action_id FROM corporate_actions WHERE security_id=? AND action_type=? AND ex_date=?",
                        (sec_id, action_type, ex_date),
                    ).fetchone()
                    if not existing_ca:
                        conn.execute(
                            """INSERT INTO corporate_actions
                               (security_id, action_type, ex_date, ratio_or_value, detail, source_id,
                                corporate_action_quality, ingested_at)
                               VALUES (?,?,?,?,?,?,?,?)""",
                            (sec_id, action_type, ex_date, ratio,
                             f"Ratio {ratio} does not match a common whole-share split ratio -- "
                             f"likely a spinoff-driven price-adjustment artifact, NOT a genuine "
                             f"share-count split. Excluded from split-adjustment math.",
                             src_id, "likely_spinoff_artifact", now_iso()),
                        )
                    ticker_note = (f"yfinance reported a 'split' ratio of {ratio} on {ex_date}, which "
                                    f"does not match common whole-share-count split ratios. Likely a "
                                    f"spinoff or other corporate action's residual price-adjustment "
                                    f"factor, not a genuine split -- flagged for manual investigation.")
                    conn.execute(
                        "UPDATE securities SET has_unsupported_corporate_action=1, "
                        "unsupported_corporate_action_note=COALESCE(unsupported_corporate_action_note || ' | ', '') || ?, "
                        "updated_at=? WHERE security_id=?",
                        (ticker_note, now_iso(), sec_id),
                    )
            conn.commit()
            split_seconds = time.time() - t_split_start
            split_count = len(splits)

        except Exception as e:
            error = str(e)
            price_seconds = div_seconds = split_seconds = 0

        total_seconds = time.time() - t0
        timing_records.append({
            "ticker": ticker, "price_seconds": round(price_seconds, 2),
            "dividend_seconds": round(div_seconds, 2), "split_seconds": round(split_seconds, 2),
            "total_seconds": round(total_seconds, 2), "bars_fetched": bars_count,
            "dividends_fetched": div_count, "splits_fetched": split_count, "error": error,
            "skipped_resume": False,
        })
        print(f"  [{i+1}/{len(tickers)}] {ticker:8s} bars={bars_count:5d} divs={div_count:3d} "
              f"splits={split_count} time={total_seconds:.1f}s"
              f"{'  ERROR: ' + error if error else ''}")

    # --- Corporate action derived series ---
    print("\nComputing split-adjusted and total-return series...")
    compute_split_adjusted_for_all(conn)
    compute_total_return_for_all(conn)

    # --- Validation ---
    print("Running validation checks...")
    checks.validate_ohlc(conn)
    checks.detect_and_flag_conflicts(conn, "SP500")

    pl.log_run(conn, "stage_ingestion", None, len(tickers), len(security_resolver), 0,
               sum(r["bars_fetched"] for r in timing_records),
               sum(1 for r in timing_records if r["error"]),
               warnings=f"stage={stage}")

    # --- Report ---
    report = generate_stage_report(conn, tickers, timing_records, price_source.source_name, stage)
    report["source_reliability_stats"] = dict(price_source.stats)  # retry/rate-limit/error telemetry
    report["resumability"] = {
        "tickers_skipped_as_already_ingested": sum(1 for r in timing_records if r.get("skipped_resume")),
        "tickers_freshly_fetched": sum(1 for r in timing_records if not r.get("skipped_resume")),
    }
    if stage == 2:
        report["stage2_selection_detail"] = get_stage2_selection_detail(CSV_PATH)
    conn.close()
    return report, timing_records


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data (dry-run / plumbing test only)")
    args = p.parse_args()

    if args.synthetic:
        from src.ingestion.price_sources import SyntheticDemoPriceSource
        source = SyntheticDemoPriceSource()
    else:
        from src.ingestion.price_sources import YFinancePriceSource
        source = YFinancePriceSource(verbose=True)

    report, timing = run_stage(args.stage, source, reset=True, force_reset=args.synthetic)
    print("\n\n" + format_report_markdown(report))
