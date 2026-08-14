"""
Phase 1 test-subset demonstration.

Builds a deliberately diverse 25-security test universe from REAL S&P 500
historical membership data (github.com/fja05680/sp500), covering:
  - continuously-listed members since 1996
  - a member added later (A, ABBV)
  - a genuine ticker-change pair (FB -> META, real historical event)
  - a genuine acquisition/delisting (MON / Monsanto, acquired by Bayer 2018)
  - a member removed then re-added then removed again (AAL)
  - members that predate our synthetic price coverage window (AAMRQ, ABX)
  - recently-added members (ACGL, ABNB)
  - one deliberately unresolved membership record (ZZZTEST)
  - one deliberately conflicting cross-source membership claim (AAL)

Price data is SYNTHETIC (see src/ingestion/price_sources.py) because this
sandbox cannot reach yfinance/Stooq. This is clearly labeled throughout.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from src.database.db import init_db, now_iso
from src.utils.config_loader import load_config
from src.universe.membership_sources import (
    SP500GithubWikipediaParser, SPDJIOfficialPressReleaseParser, SP400MembershipParser
)
from src.universe import identity_resolution as idres
from src.ingestion.price_sources import SyntheticDemoPriceSource
from src.ingestion import pipeline as pl
from src.ingestion.adjustments import compute_split_adjusted_for_all, compute_total_return_for_all
from src.validation import checks

CONFIG = load_config()
DB_PATH = "data/database/quant_trader.db"
SCHEMA_PATH = "src/database/schema.sql"
CSV_PATH = "data/raw/sp500-master/sp500_ticker_start_end.csv"

TEST_SUBSET = [
    "AAPL", "MSFT", "JNJ", "PG", "KO", "XOM", "JPM", "WMT", "HD", "DIS",   # continuous since 1996
    "A", "ABBV",                                                            # added later, continuous since
    "FB", "META",                                                           # ticker-change pair
    "MON",                                                                  # acquired/delisted 2018
    "AAL",                                                                  # removed, re-added, removed again
    "ABMD",                                                                 # delisted 2022 (acquired)
    "ACGL", "ABNB",                                                         # recently added
    "AABA",                                                                 # delisted 2017 (Yahoo/Altaba)
    "ABC",                                                                  # delisted 2023 (renamed/Cencora)
    "AAMRQ", "ABX",                                                         # delisted pre-2015 (before our price window)
]

PRICE_START, PRICE_END = CONFIG["data"]["price_start_date"], CONFIG["data"]["price_end_date"]


def run(reset_db=True):
    log = {"steps": []}
    def step(msg):
        print(f"\n=== {msg} ===")
        log["steps"].append(msg)

    step("1. Initialize database from schema")
    conn = init_db(DB_PATH, SCHEMA_PATH, reset=reset_db)

    step("2. Parse REAL S&P 500 historical membership data (Tier B, github.com/fja05680/sp500)")
    tier_b_parser = SP500GithubWikipediaParser()
    tier_b_records = tier_b_parser.parse(CSV_PATH, ticker_whitelist=set(TEST_SUBSET))
    print(f"Parsed {len(tier_b_records)} real Tier-B membership records for {len(TEST_SUBSET)} whitelisted tickers")
    for r in tier_b_records:
        print(f"  {r.raw_ticker:8s} {r.effective_date} -> {r.removal_date or '(active)'}  conf={r.confidence}")

    step("3. Add a demonstration Tier-A (official) record — verified confidence")
    tier_a_parser = SPDJIOfficialPressReleaseParser()
    tier_a_records = tier_a_parser.parse([{
        "ticker": "META", "effective_date": "2022-06-09",
        "source_reference": "Meta Platforms ticker/name change effective 2022-06-09 (publicly documented corporate event)",
    }])

    step("4. Inject ONE deliberately conflicting cross-source claim (AAL second period)")
    conflicting_records = tier_b_parser.parse.__self__.__class__.__mro__  # no-op placeholder to keep structure obvious
    from src.universe.membership_sources import MembershipRecord
    conflicting_records = [
        MembershipRecord(
            raw_ticker="AAL", index_name="SP500", effective_date="2015-03-20",
            removal_date="2024-09-23", announcement_date=None,
            source_name="Synthetic Tier-C demo source (deliberately disagrees with Tier B)",
            source_tier="C", source_reference=None, confidence="unverified",
            verification_status="Deliberately injected to demonstrate conflicting-claim handling; "
                                 "disagrees with Tier-B effective_date (2015-03-23) by 3 days.",
        )
    ]

    step("5. Inject ONE deliberately unresolved membership record (ticker with no security master match)")
    unresolved_records = [
        MembershipRecord(
            raw_ticker="ZZZTEST", index_name="SP500", effective_date="2010-01-01",
            removal_date=None, announcement_date=None,
            source_name="Synthetic Tier-C demo source (deliberately unresolvable)",
            source_tier="C", source_reference=None, confidence="unverified",
            verification_status="Deliberately injected: no corresponding security exists in the "
                                 "security master, to test unresolved-identifier handling.",
        )
    ]

    all_records = tier_b_records + tier_a_records + conflicting_records + unresolved_records

    step("6. Resolve securities (ticker -> security_id), build security master + ticker_history")
    # Load the CIK/ISIN registry first -- identity resolution prefers this
    # over ticker-only matching, since tickers get reused and renamed.
    id_reg_result = idres.load_known_identifiers(conn, "data/reference/known_identifiers_seed.csv")
    print(f"Loaded known-identifiers registry: {id_reg_result}")

    # Minimal manually-curated metadata for the test subset (name/sector) — in production this
    # would come from the T212 instruments endpoint / another metadata source, not hardcoded.
    known_names = {
        "AAPL": "Apple Inc", "MSFT": "Microsoft Corp", "JNJ": "Johnson & Johnson", "PG": "Procter & Gamble",
        "KO": "Coca-Cola Co", "XOM": "Exxon Mobil Corp", "JPM": "JPMorgan Chase & Co", "WMT": "Walmart Inc",
        "HD": "Home Depot Inc", "DIS": "Walt Disney Co", "A": "Agilent Technologies", "ABBV": "AbbVie Inc",
        "FB": "Facebook Inc / Meta Platforms Inc (former name)", "META": "Meta Platforms Inc",
        "MON": "Monsanto Co", "AAL": "American Airlines Group", "ABMD": "Abiomed Inc",
        "ACGL": "Arch Capital Group", "ABNB": "Airbnb Inc", "AABA": "Altaba Inc (formerly Yahoo! Inc)",
        "ABC": "AmerisourceBergen Corp", "AAMRQ": "AMR Corp (American Airlines, bankruptcy ticker)",
        "ABX": "Barrick Gold Corp",
    }

    security_resolver = {}   # ticker -> security_id, for THIS ingest run's membership rows
    identity_methods = {}    # ticker -> how it was resolved (for the report)
    fb_meta_security_id = None

    for ticker in TEST_SUBSET:
        # FB and META are the SAME security (ticker change) — resolve to one security_id
        if ticker == "META" and fb_meta_security_id is not None:
            security_resolver["META"] = fb_meta_security_id
            continue
        sec_id, created, method = idres.resolve_or_create_security(
            conn, ticker, name=known_names.get(ticker), exchange="NASDAQ/NYSE (unspecified in test data)",
            country="US", currency="USD", asset_type="STOCK",
        )
        security_resolver[ticker] = sec_id
        identity_methods[ticker] = method
        if ticker == "FB":
            fb_meta_security_id = sec_id
        print(f"  {'created' if created else 'found existing'}: {ticker} -> security_id={sec_id} (method={method})")

    # ticker_history: FB then META on the SAME security_id
    pl.link_ticker_history(conn, security_resolver["FB"], "FB", "1996-01-01", "2022-06-09", "manual/demo")
    pl.link_ticker_history(conn, security_resolver["FB"], "META", "2022-06-09", None, "manual/demo")
    for t in TEST_SUBSET:
        if t in ("FB", "META"):
            continue
        pl.link_ticker_history(conn, security_resolver[t], t, "1900-01-01", None, "manual/demo")

    # Note: ZZZTEST is deliberately NOT in security_resolver -> stays unresolved
    step("7. Ingest membership records (idempotent)")
    m_result_run1 = pl.ingest_membership_records(conn, all_records, security_resolver)
    print(f"Run 1: {m_result_run1}")

    step("8. Mark delisted securities in the security master")
    # Several of these dates were originally the security's S&P 500 removal
    # date (an index-membership event), which is not the same as when the
    # security actually stopped trading. Fixed below with primary-source
    # citations for each one.
    delisted_meta = {
        # (delisted_date, reason, confidence, source)
        "MON": ("2018-06-07", "acquired", "verified",
                "SEC EDGAR DEFM14A (CIK 1110783): stock delisted from NYSE and deregistered under the "
                "Exchange Act upon Bayer acquisition closing. "
                "https://www.sec.gov/Archives/edgar/data/1110783/000119312516765991/d252304ddefm14a.htm"),
        "ABMD": ("2022-12-22", "acquired", "verified",
                 "SEC EDGAR 8-K (Johnson & Johnson, CIK 200406): tender offer for Abiomed completed "
                 "2022-12-22, converting shares to cash + CVR. "
                 "https://www.sec.gov/Archives/edgar/data/200406/000119312522311072/d428734dex991.htm"),
        "AABA": ("2019-10-02", "dissolved", "verified",
                 "Originally dated to the 2017-06-19 S&P 500 removal date and labeled 'renamed', which "
                 "was wrong on both counts. Yahoo! Inc renamed to Altaba Inc in 2017 but kept trading "
                 "under ticker AABA for two more years; Altaba stopped trading 2019-10-02 and filed its "
                 "certificate of dissolution 2019-10-04. "
                 "https://en.wikipedia.org/wiki/Altaba ; SEC EDGAR CIK 1011006 8-K filings confirm the timeline."),
        "ABC": (None, None, None, None),  # not delisted -- see known_ticker_renames registry (ABC->COR)
        "AAMRQ": ("2012-01-30", "bankrupt", "verified",
                  "Originally dated to the 2003-03-14 S&P 500 removal date. The real story: AMR Corp "
                  "(ticker AMR on NYSE) filed Chapter 11 on 2011-11-29, NYSE trading was suspended "
                  "2012-01-05 and formally delisted by the SEC 2012-01-30; AMR then traded OTC under "
                  "ticker AAMRQ until its 2013 merger into American Airlines Group (ticker AAL). Caveat: "
                  "the ticker 'AAMRQ' itself did not exist during AMR's actual 1996-2003 S&P 500 "
                  "membership window -- the real ticker then was 'AMR'. The source membership file labels "
                  "the security by a later-known ticker rather than the one in use at the time, which is "
                  "a structural risk beyond this single record. "
                  "https://www.sec.gov/Archives/edgar/data/6201/000000620113000023/amr-10kx20121231.htm"),
        "ABX": (None, None, None, None),  # CORRECTED: not delisted at all -- see known_ticker_renames (ABX->GOLD, 2019); Barrick kept trading continuously
        "FB": (None, None, None, None),  # not delisted -- renamed in place, security remains active as META
    }
    for ticker, (delisted_date, reason, confidence, source) in delisted_meta.items():
        if delisted_date is None:
            continue  # ABC, ABX, FB: not genuinely delisted, handled via rename registry instead
        sec_id = security_resolver[ticker]
        conn.execute(
            "UPDATE securities SET active_flag=0, delisted_date=?, delisting_reason=?, "
            "delisting_confidence=?, delisting_source=?, updated_at=? WHERE security_id=?",
            (delisted_date, reason, confidence, source, now_iso(), sec_id),
        )
    conn.commit()

    step("9. Ingest SYNTHETIC price data (idempotent) — see honesty note in price_sources.py")
    price_source = SyntheticDemoPriceSource(
        gap_tickers={"ABMD": ("2020-01-01", "2020-06-30")},  # deliberate partial-data test case
        no_data_tickers={"MON"},                              # deliberate "no price data at all" test case
    )

    price_windows = {
        "AAPL": (PRICE_START, PRICE_END), "MSFT": (PRICE_START, PRICE_END), "JNJ": (PRICE_START, PRICE_END),
        "PG": (PRICE_START, PRICE_END), "KO": (PRICE_START, PRICE_END), "XOM": (PRICE_START, PRICE_END),
        "JPM": (PRICE_START, PRICE_END), "WMT": (PRICE_START, PRICE_END), "HD": (PRICE_START, PRICE_END),
        "DIS": (PRICE_START, PRICE_END), "A": (PRICE_START, PRICE_END), "ABBV": (PRICE_START, PRICE_END),
        "FB": (PRICE_START, "2022-06-09"), "META": ("2022-06-09", PRICE_END),
        "MON": ("2015-01-01", "2018-06-07"),          # window defined but no_data_tickers -> yields zero rows
        "AAL": ("2015-03-23", PRICE_END),
        "ABMD": (PRICE_START, "2022-12-22"),
        "ACGL": ("2022-11-01", PRICE_END),
        "ABNB": ("2023-09-18", PRICE_END),
        "AABA": (PRICE_START, "2019-10-02"),
        "ABC": (PRICE_START, "2023-08-30"),
        "ABX": (PRICE_START, PRICE_END),  # CORRECTED: genuinely traded throughout; recoverable via ABX->GOLD rename
        # AAMRQ: still deliberately no window -- genuine dead end (bankruptcy, no
        # recoverable successor ticker), though now for the RIGHT documented reason.
    }

    price_totals = {"inserted": 0, "skipped_duplicate": 0}
    for ticker, (start, end) in price_windows.items():
        sec_id = security_resolver[ticker]
        bars = price_source.fetch(ticker, start, end)
        res = pl.ingest_prices(conn, sec_id, ticker, bars, price_source.source_name, adj_type="raw")
        price_totals["inserted"] += res["inserted"]
        price_totals["skipped_duplicate"] += res["skipped_duplicate"]
        print(f"  {ticker:8s} window {start}..{end}: {len(bars):5d} bars fetched, {res}")
    print(f"Price ingest run 1 totals: {price_totals}")

    step("10. Ingest corporate actions")
    # Real, well-documented events for our test subset
    pl.ingest_corporate_action(conn, security_resolver["AAPL"], "split", "2020-08-31", 4.0,
                                "4-for-1 stock split", "manual/public record", "B", quality="unverified")
    pl.ingest_corporate_action(conn, security_resolver["FB"], "ticker_change", "2022-06-09", None,
                                "FB -> META ticker/name change", "manual/public record", "B", quality="unverified")
    pl.ingest_corporate_action(conn, security_resolver["MON"], "acquisition", "2018-06-07", None,
                                "Acquired by Bayer AG; stock delisted from NYSE and deregistered under "
                                "the Exchange Act per Monsanto's own DEFM14A merger proxy -- confirmed "
                                "no successor public ticker exists. Source: SEC EDGAR CIK 1110783, "
                                "https://www.sec.gov/Archives/edgar/data/1110783/000119312516765991/d252304ddefm14a.htm",
                                "SEC EDGAR (primary source)", "A", quality="verified")

    conn.commit()

    step("10b. Compute split-adjusted price series from raw prices + corporate actions")
    adj_results = compute_split_adjusted_for_all(conn)
    print(f"Split-adjustment computed for {len(adj_results)} securities.")
    print("The synthetic price generator used in this demo is a split-unaware random walk, so this")
    print("step runs correctly but the resulting split_adjusted series isn't meaningfully different")
    print("from raw beyond the exact math. The adjustment logic itself is validated against realistic")
    print("AAPL split values in tests/test_phase1.py::test_split_adjustment_removes_realistic_split_cliff.")
    print("Split-adjusted values only become meaningful once run against real yfinance data")
    print("(run_phase1_real_data.py), where raw prices genuinely contain the split cliff.")

    step("10c. Compute total-return (dividend-adjusted) price series")
    tr_results = compute_total_return_for_all(conn)
    print(f"Total-return series computed for {len(tr_results)} securities.")
    print("Same caveat as split-adjustment: this demo's synthetic/corporate-action data has no real")
    print("dividends recorded, so total_return == split_adjusted here. Validated separately against")
    print("realistic manual dividend data -- see test_total_return_reflects_dividend_reinvestment.")

    step("10d. Flag unsupported corporate actions (spin-offs, rights issues)")
    # Real example: AbbVie (ABBV) was spun off from Abbott Laboratories on
    # 2013-01-01/02, confirmed via Abbott's own 8-K. Spin-offs aren't
    # processed yet (BACKLOG.md #5), so this must be visibly flagged rather
    # than silently treated as a clean, unaffected price series.
    pl.flag_unsupported_corporate_action(
        conn, security_resolver["ABBV"], "spinoff", "2013-01-01",
        "AbbVie spun off from Abbott Laboratories (ABT). Our price history for ABBV starts "
        "at its independent listing (2013-01-02 per real S&P 500 membership data), so this "
        "doesn't corrupt ABBV's own series -- but it IS the kind of event that would need "
        "special handling (cost-basis split from parent) if Abbott (ABT) were ever added to "
        "this universe. Source: Abbott 8-K Item 2.01, "
        "https://www.sec.gov/Archives/edgar/data/0000001800/000110465913001016/a13-2169_18k.htm",
        "SEC EDGAR (Abbott Laboratories 8-K)", "A",
    )
    flagged = pl.securities_with_unsupported_corporate_actions(conn)
    print(f"Securities flagged with unsupported corporate actions: {[f['primary_ticker'] for f in flagged]}")

    step("10e. Ingest a real, sourced S&P 400 sample (3 events spanning 2019/2024/2025)")
    # Deliberately not bulk-scraping SPDJI press releases here. This just
    # demonstrates that the modular parser normalizes real S&P 400 data into
    # the same index_membership schema as S&P 500 -- proving the
    # architecture works, not claiming complete S&P 400 coverage.
    sp400_parser = SP400MembershipParser()
    sp400_sample = [
        {"ticker": "BRX", "effective_date": "2019-02-06",
         "source_reference": "https://www.spice-indices.com/idpfiles/spice-assets/resources/public/documents/864758_vectren4.pdf"},
        {"ticker": "FLEX", "effective_date": "2024-11-25", "announcement_date": "2024-11-19",
         "source_reference": "https://www.sdxcentral.com/press-releases/sp-dow-jones-indices-announces-changes-to-midcap-400-and-smallcap-600/"},
        {"ticker": "FTI", "effective_date": "2025-09-12", "announcement_date": "2025-09-02",
         "source_reference": "https://finance.yahoo.com/news/technipfmc-set-join-p-midcap-215400910.html"},
    ]
    sp400_records = sp400_parser.parse(manual_records=sp400_sample)
    sp400_resolver = {}
    for rec in sp400_sample:
        sec_id, _, _ = idres.resolve_or_create_security(conn, rec["ticker"], name=f"{rec['ticker']} (SP400 sample)")
        sp400_resolver[rec["ticker"]] = sec_id
    sp400_result = pl.ingest_membership_records(conn, sp400_records, sp400_resolver)
    print(f"S&P 400 sample ingest: {sp400_result} "
          f"(BRX 2019, FLEX 2024, FTI 2025 -- real SPDJI press-release-sourced events)")

    step("10c. Apply configured universe filters (config/config.yaml, NOT hardcoded)")
    filter_summary = {"passed": 0, "failed": 0, "details": []}
    for ticker, sec_id in security_resolver.items():
        if ticker == "META":  # same security as FB, already evaluated
            continue
        result = checks.apply_universe_filters(conn, sec_id, CONFIG, adj_type="raw")
        filter_summary["details"].append({"ticker": ticker, **result})
        if result["passed"]:
            filter_summary["passed"] += 1
        else:
            filter_summary["failed"] += 1
    print(f"Universe filter results: {filter_summary['passed']} passed, {filter_summary['failed']} failed "
          f"(thresholds from config.yaml: {CONFIG['universe_filters']})")
    for d in filter_summary["details"]:
        status = "PASS" if d["passed"] else "FAIL"
        reason = d.get("reason", "")
        print(f"  {d['ticker']:8s} {status}  {reason}")

    step("11. Log ingestion run")
    identity_review = conn.execute("SELECT COUNT(*) c FROM identity_review_queue WHERE resolved=0").fetchone()["c"]
    print(f"Identity review queue (unresolved ticker-reuse-risk flags): {identity_review}")
    print(f"Identity resolution methods used: {identity_methods}")
    pl.log_run(conn, "membership", None, len(all_records), len(security_resolver), 1,
               m_result_run1["inserted"], m_result_run1["skipped_duplicate"])
    pl.log_run(conn, "prices", None, len(price_windows), len(price_windows), 0,
               price_totals["inserted"], price_totals["skipped_duplicate"])

    conn.close()
    return {"security_resolver": security_resolver, "all_records_count": len(all_records)}


if __name__ == "__main__":
    result = run(reset_db=True)
    print("\n\nFirst run complete.")
    print(json.dumps({"securities_resolved": len(result["security_resolver"])}, indent=2))
