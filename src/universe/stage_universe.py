"""
Staged universe selection. The ingestion pipeline itself (run_stage_ingestion.py)
takes a plain ticker list and is agnostic to how that list was built -- this
module is the only thing that changes between stages.

Stage 1 (~50): hand-curated for representativeness (mega-cap, large-cap,
smaller S&P 500, long histories, ticker changes, splits, dividends,
historical delistings). Category labels below are a best-knowledge
categorization for test design purposes -- the actual membership dates,
corporate actions, and financial facts for each name come from the real
data sources (membership CSV + yfinance) during ingestion, not asserted
here. If a ticker turns out not to be in the real S&P 500 membership
file, the parser just returns zero membership records for it -- a real,
informative finding, not a crash.

Stage 2/3/4: derived directly from the real S&P 500 historical membership
file (github.com/fja05680/sp500), no manual curation -- proving the
pipeline scales without a rewrite.
"""
import csv


STAGE1_TICKERS = {
    # Mega-cap, currently trading, huge market cap
    "mega_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM"],
    # Large-cap, well-known, currently trading
    "large_cap": ["XOM", "JNJ", "PG", "KO", "WMT", "HD", "DIS", "UNH",
                   "V", "MA", "PFE", "CSCO", "BAC", "GS", "CAT", "BA", "NKE", "MCD"],
    # Smaller/lower-profile S&P 500 constituents (still large companies by
    # most standards, but far less prominent than the above)
    "smaller_sp500": ["IEX", "JKHY", "CHRW", "POOL", "EXPD", "WAT", "CHD",
                       "NDSN", "ALGN", "TXT", "ROL", "LEN", "PAYX"],
    # Known for a real historical ticker change (verified earlier this project)
    "ticker_changes": ["ABC", "ABX"],  # ABC->COR, ABX->GOLD (both in known_ticker_renames registry)
    # Known for at least one stock split in recent years (exact dates/ratios
    # come from the real yfinance split fetch, not hardcoded here)
    "splits": ["TSLA"],  # AAPL, NVDA, GOOGL already covered above and also have real splits
    # Known for substantial/long-running dividends
    "dividends": ["T", "IBM", "MMM", "CVX"],
    # Historically LEFT the S&P 500 (delisted/dissolved) -- all four have
    # real, sourced citations (see BACKLOG.md)
    "historically_delisted": ["MON", "ABMD", "AABA", "AAMRQ"],
    # Real, sourced example of an unsupported corporate action (spin-off) --
    # AbbVie spun off from Abbott Laboratories 2013-01-01/02, confirmed via
    # Abbott's own 8-K
    "spinoff_example": ["ABBV"],
}


def get_stage1_tickers():
    """Flat, deduplicated list for Stage 1 (~50 securities)."""
    seen = []
    for category, tickers in STAGE1_TICKERS.items():
        for t in tickers:
            if t not in seen:
                seen.append(t)
    return seen


def get_stage1_category_map():
    """ticker -> list of categories it was chosen for (a ticker can be in
    multiple), for the validation report to show WHY each was included."""
    out = {}
    for category, tickers in STAGE1_TICKERS.items():
        for t in tickers:
            out.setdefault(t, []).append(category)
    return out


def get_stage_tickers(stage, membership_csv_path, seed=42):
    """
    Stage 1: hand-curated list above.
    Stage 2: ~200, stratified pseudo-random sample from the REAL S&P 500
             membership file (deterministic via seed), guaranteed to
             include every Stage 1 ticker as a subset (monotonic staging).
    Stage 3: every distinct ticker that has EVER been a real S&P 500
             member per the membership file (full historical S&P 500).
    Stage 4: Stage 3 + whatever S&P 400 tickers are available (currently
             only the small real sample from Phase 1 -- see BACKLOG.md
             item 4 on S&P 400 coverage limits).

    Note on Stage 2 "representativeness": the additional ~150 tickers
    beyond Stage 1 are a uniform random sample (seeded, so reproducible)
    from the full real S&P 500 historical membership list. It's not
    stratified by sector (no free bulk sector-classification source
    exists -- see BACKLOG.md) or explicitly balanced by market cap, but it
    does naturally span a wide range of historical membership dates since
    it's drawn from the full 1996-2024 membership file, and it guarantees
    every Stage 1 category (mega-cap, splits, dividends, delistings, etc.)
    is present via the Stage 1 subset. Call get_stage2_selection_detail()
    for the exact per-ticker breakdown.
    """
    if stage == 1:
        return get_stage1_tickers()

    all_tickers = []
    with open(membership_csv_path, "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if parts and parts[0]:
                all_tickers.append(parts[0])
    all_tickers = sorted(set(all_tickers))

    if stage == 3:
        return all_tickers

    if stage == 2:
        import random
        rng = random.Random(seed)
        stage1 = set(get_stage1_tickers())
        remaining_pool = [t for t in all_tickers if t not in stage1]
        rng.shuffle(remaining_pool)
        target_additional = max(0, 200 - len(stage1))
        return sorted(stage1) + remaining_pool[:target_additional]

    if stage == 4:
        # S&P 400 bulk historical data is NOT available for free (see
        # BACKLOG.md #4) -- this returns Stage 3 plus only whatever S&P 400
        # sample tickers exist in the database already, not a claim of
        # full S&P 400 coverage.
        return all_tickers  # caller should separately merge in known SP400 sample tickers

    raise ValueError(f"Unknown stage: {stage}")


def get_stage2_selection_detail(membership_csv_path, seed=42):
    """
    Per-ticker breakdown of why each Stage 2 ticker was selected, so it's
    always possible to see exactly which securities were included and why.
    Returns list of {ticker, selection_reason, categories}.
    """
    stage1_tickers = get_stage1_tickers()
    stage1_categories = get_stage1_category_map()
    stage2_tickers = get_stage_tickers(2, membership_csv_path, seed=seed)

    detail = []
    for t in stage2_tickers:
        if t in stage1_tickers:
            detail.append({
                "ticker": t, "selection_reason": "stage1_hand_curated",
                "categories": stage1_categories.get(t, []),
            })
        else:
            detail.append({
                "ticker": t, "selection_reason": f"stage2_random_addition_seed_{seed}",
                "categories": [],
            })
    return detail
