"""
Measures survivorship-bias coverage for the S&P 500 using REAL historical
membership data (github.com/fja05680/sp500).

Question this answers: for a given backtest window, what fraction of
securities that were EVER an S&P 500 member during that window are still
listed today (and therefore fetchable from free sources like yfinance)?

The gap (1 - coverage) is the proportion of the true historical universe
that a free-data pipeline will silently be unable to price — i.e. the
survivorship bias, as a number, not a guess.

Approximation used: "still listed today" is approximated by "this source's
membership record has no end_date" (still an active constituent as of the
source's last update). This is a reasonable proxy, not a certainty — a
stock can leave the S&P 500 while remaining listed elsewhere (in which case
this UNDERSTATES fetchability), or a ticker can be reused by an unrelated
company decades later (which this script does not attempt to detect).
Treat this as a first-order estimate, not a precise figure.
"""
import csv
from datetime import date


def load_renames_map(csv_path):
    """old_ticker -> new_ticker, from the curated registry. Only tickers
    with an EXPLICIT, sourced entry count -- this is not guessed."""
    renames = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            renames[row["old_ticker"]] = row["new_ticker"]
    return renames


def load_membership(csv_path):
    records = []
    with open(csv_path, "r") as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            ticker = parts[0]
            start = parts[1]
            end = parts[2] if len(parts) > 2 and parts[2] else None
            records.append((ticker, start, end))
    return records


def coverage_for_window(records, window_start, window_end, renames_map=None):
    """Securities that overlapped the window at all, vs. how many are
    recoverable -- either still active (no end_date) OR confirmed recoverable
    via a known, sourced ticker rename (e.g. FB->META). Everything else is
    a genuine loss. renames_map=None reproduces the original (pre-registry)
    numbers for comparison."""
    renames_map = renames_map or {}
    overlapping = {}  # ticker -> 'still_active' | 'recovered_rename' | 'lost'
    for ticker, start, end in records:
        if start > window_end:
            continue
        if end is not None and end < window_start:
            continue

        if end is None:
            status = "still_active"
        elif ticker in renames_map:
            status = "recovered_rename"
        else:
            status = "lost"

        if ticker not in overlapping:
            overlapping[ticker] = status
        else:
            # if ANY period for this ticker is still active, that wins
            if overlapping[ticker] != "still_active":
                overlapping[ticker] = status if status == "still_active" else overlapping[ticker]

    total = len(overlapping)
    still_listed = sum(1 for v in overlapping.values() if v == "still_active")
    recovered = sum(1 for v in overlapping.values() if v == "recovered_rename")
    lost = sum(1 for v in overlapping.values() if v == "lost")
    coverage = (still_listed + recovered) / total if total else 0.0
    return {
        "window": f"{window_start} to {window_end}",
        "total_distinct_constituents": total,
        "still_listed_today": still_listed,
        "recovered_via_known_rename": recovered,
        "genuinely_lost": lost,
        "coverage_ratio": round(coverage, 4),
        "survivorship_bias_gap": round(1 - coverage, 4),
    }


def removed_ticker_detail(records, window_start, window_end, limit=None):
    """List of tickers that overlapped the window but are no longer active,
    with their removal date, so you can see WHEN the losses cluster."""
    seen = {}
    for ticker, start, end in records:
        if start > window_end:
            continue
        if end is not None and end < window_start:
            continue
        if end is None:
            continue  # still listed, not a loss
        if ticker not in seen or end > seen[ticker]:
            seen[ticker] = end
    items = sorted(seen.items(), key=lambda x: x[1])
    return items[:limit] if limit else items


if __name__ == "__main__":
    CSV_PATH = "data/raw/sp500-master/sp500_ticker_start_end.csv"
    RENAMES_PATH = "data/reference/known_ticker_renames_seed.csv"
    records = load_membership(CSV_PATH)
    renames_map = load_renames_map(RENAMES_PATH)
    print(f"Loaded {len(records)} raw membership period records "
          f"({len(set(r[0] for r in records))} distinct tickers ever seen)")
    print(f"Loaded {len(renames_map)} known ticker renames from the curated registry: {renames_map}\n")

    today = date.today().isoformat()

    windows = [
        ("1996-01-01", today),
        ("2010-01-01", today),
        ("2015-01-01", today),
        ("2018-01-01", today),
        ("2020-01-01", today),
        ("2022-01-01", today),
    ]

    print("=== WITHOUT rename registry (original numbers) ===")
    print(f"{'Window':<28} {'Total':>7} {'StillListed':>12} {'Lost':>6} {'Coverage':>9} {'BiasGap':>8}")
    print("-" * 76)
    for w_start, w_end in windows:
        r = coverage_for_window(records, w_start, w_end, renames_map=None)
        print(f"{r['window']:<28} {r['total_distinct_constituents']:>7} "
              f"{r['still_listed_today']:>12} {r['genuinely_lost']:>6} "
              f"{r['coverage_ratio']*100:>8.1f}% {r['survivorship_bias_gap']*100:>7.1f}%")

    print("\n=== WITH rename registry applied (corrected numbers) ===")
    print(f"{'Window':<28} {'Total':>7} {'StillListed':>12} {'Recovered':>10} {'Lost':>6} {'Coverage':>9} {'BiasGap':>8}")
    print("-" * 90)
    results = []
    for w_start, w_end in windows:
        r = coverage_for_window(records, w_start, w_end, renames_map=renames_map)
        results.append(r)
        print(f"{r['window']:<28} {r['total_distinct_constituents']:>7} "
              f"{r['still_listed_today']:>12} {r['recovered_via_known_rename']:>10} "
              f"{r['genuinely_lost']:>6} {r['coverage_ratio']*100:>8.1f}% {r['survivorship_bias_gap']*100:>7.1f}%")

    print(f"\nNote: with only {len(renames_map)} entries in the registry, this correction is small "
          f"and will UNDERSTATE true coverage at full scale -- the registry only fixes what's been "
          f"individually verified so far, it does not systematically find every rename.")

    print("\n--- Detail: securities lost for the 2015-today window (first 25 by removal date, renames excluded) ---")
    lost = removed_ticker_detail(records, "2015-01-01", today, limit=None)
    lost = [x for x in lost if x[0] not in renames_map]
    for ticker, removal_date in lost[:25]:
        print(f"  {ticker:10s} removed {removal_date}")
    print(f"\n  ... {max(0, len(lost) - 25)} more not shown")
