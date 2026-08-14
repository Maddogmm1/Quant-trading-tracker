"""
Modular membership-source ingestion.

Pipeline shape:

    raw source  -->  normalised MembershipRecord list  -->  validation
                 -->  security identifier resolution   -->  database

S&P 500 and S&P 400 share the same downstream schema/pipeline even though
their upstream sources are completely different. Only the parser at the
front changes.
"""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class MembershipRecord:
    """Normalised record — the common currency between any source parser
    and the database layer. Every parser must emit these."""
    raw_ticker: str
    index_name: str            # 'SP500' | 'SP400'
    effective_date: str        # ISO date
    removal_date: Optional[str]
    announcement_date: Optional[str]
    source_name: str
    source_tier: str           # 'A' | 'B' | 'C'
    source_reference: Optional[str]
    confidence: str            # 'verified' | 'unverified' | 'conflicting'
    verification_status: str


class MembershipSourceParser:
    """Base interface every index-membership source must implement."""

    source_name: str
    source_tier: str

    def parse(self, **kwargs) -> List[MembershipRecord]:
        raise NotImplementedError


class SP500GithubWikipediaParser(MembershipSourceParser):
    """
    Tier B source: community-maintained reconstruction of S&P 500
    membership derived from Wikipedia (current list + revision/selected-
    changes history), cross-referenced by the maintainer against an
    original 1996-2019 file sourced from 'Trading Evolved' (Clenow).

    Source: github.com/fja05680/sp500, file 'sp500_ticker_start_end.csv'
    (pre-derived per-ticker membership windows).

    Known, maintainer-documented limitation (from the repo's own README):
    coverage in the first ~5 years (1996-2000) is less certain — the
    constituent count sits visibly below ~500 and the maintainer states
    they have "no way to independently check it" for that period. Encoded
    here as a lower confidence for effective_dates < 2001-01-01.
    """

    source_name = "fja05680/sp500 (GitHub, Wikipedia-derived)"
    source_tier = "B"
    EARLY_PERIOD_CUTOFF = "2001-01-01"  # per source's own documented caveat

    def parse(self, csv_path: str, ticker_whitelist=None) -> List[MembershipRecord]:
        records = []
        with open(csv_path, "r") as f:
            header = f.readline()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                ticker, start_date = parts[0], parts[1]
                end_date = parts[2] if len(parts) > 2 and parts[2] else None

                if ticker_whitelist is not None and ticker not in ticker_whitelist:
                    continue

                confidence = "unverified"
                note = "Not cross-checked against an official SPDJI press release."
                if start_date < self.EARLY_PERIOD_CUTOFF:
                    note = (
                        "Falls within source's own documented low-confidence window "
                        "(pre-2001); source maintainer states constituent counts are "
                        "visibly below ~500 in this period and could not be independently "
                        "verified."
                    )

                records.append(
                    MembershipRecord(
                        raw_ticker=ticker,
                        index_name="SP500",
                        effective_date=start_date,
                        removal_date=end_date,
                        announcement_date=None,  # not provided by this source
                        source_name=self.source_name,
                        source_tier=self.source_tier,
                        source_reference="https://github.com/fja05680/sp500",
                        confidence=confidence,
                        verification_status=note,
                    )
                )
        return records


class SPDJIOfficialPressReleaseParser(MembershipSourceParser):
    """
    Tier A source: official S&P Dow Jones Indices press releases
    (press.spglobal.com). These are ground truth but are published as
    individual PDFs/pages per change, not a bulk file — there is no
    free bulk historical export.

    Deliberately not bulk-scraped. This class defines the interface and,
    for Phase 1, is seeded with a small number of manually verified records
    (well-documented public index changes) to demonstrate that a Tier-A
    source resolves with 'verified' confidence and can coexist with, or
    override, Tier-B claims about the same security. Bulk automated
    retrieval is future work.
    """

    source_name = "S&P Dow Jones Indices official press releases"
    source_tier = "A"

    def parse(self, manual_records: List[dict]) -> List[MembershipRecord]:
        records = []
        for r in manual_records:
            records.append(
                MembershipRecord(
                    raw_ticker=r["ticker"],
                    index_name=r.get("index_name", "SP500"),
                    effective_date=r["effective_date"],
                    removal_date=r.get("removal_date"),
                    announcement_date=r.get("announcement_date"),
                    source_name=self.source_name,
                    source_tier=self.source_tier,
                    source_reference=r.get("source_reference"),
                    confidence="verified",
                    verification_status="Manually verified against a specific, dated SPDJI press release.",
                )
            )
        return records


class SP400MembershipParser(MembershipSourceParser):
    """
    Interface stub only.

    No free bulk historical S&P 400 membership file exists. The only free
    ground-truth source is SPDJI press releases (Tier A), published one
    change at a time, with materially weaker findability/coverage the
    further back you go.

    This class exists to prove the S&P 500 and S&P 400 pipelines share a
    schema: it emits the same MembershipRecord type as the S&P 500
    parsers, just with index_name='SP400'. Not populated with real data in
    Phase 1 -- that would require the press-release scraping work this
    project isn't doing yet.
    """

    source_name = "S&P 400 (not yet implemented — interface only)"
    source_tier = "A"

    def parse(self, manual_records: Optional[List[dict]] = None) -> List[MembershipRecord]:
        if not manual_records:
            return []
        records = []
        for r in manual_records:
            records.append(
                MembershipRecord(
                    raw_ticker=r["ticker"],
                    index_name="SP400",
                    effective_date=r["effective_date"],
                    removal_date=r.get("removal_date"),
                    announcement_date=r.get("announcement_date"),
                    source_name=self.source_name,
                    source_tier=self.source_tier,
                    source_reference=r.get("source_reference"),
                    confidence="verified",
                    verification_status="Manually seeded demonstration record only.",
                )
            )
        return records
