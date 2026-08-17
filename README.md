# Quant Trader — UK Stock-Ranking Research (Phase 1–5)

A from-scratch research pipeline that asks a simple question: does information
that was genuinely available at time *t* let you predict which UK/US large-cap
stocks will outperform their peers, in a way that would survive real
transaction costs inside a Trading 212 Stocks & Shares ISA? Built end-to-end
on free data (`yfinance`), with £0/month running cost.

**Headline result:** two independent hypotheses tested, both pre-registered
with frozen thresholds and a locked out-of-sample set, both came back
negative.

- **Phase 4** (cross-sectional, 3-month and 1-month horizons): across seven
  model families — historical mean, momentum, ridge, elastic net, logistic
  regression, random forest, and gradient boosting — none showed predictive
  value that survived robustness checks. The apparent edge from the tree
  models turned out to be overfitting to a single four-month market regime
  (COVID recovery, Aug–Nov 2020), not a stable signal. See
  [`PHASE4_CONCLUSION_3M.md`](PHASE4_CONCLUSION_3M.md).
- **Phase 5** (overnight vs. intraday return decomposition): a large
  apparent overnight effect in the validation window traced back to the
  same COVID-reopening regime — a handful of travel/energy names swinging
  10–30% overnight around vaccine-trial news — and disappeared once folded
  back into the full sample. Caught at the first statistical gate, before
  any model-building or locked-test time was spent on it. See
  [`PHASE5_CONCLUSION.md`](PHASE5_CONCLUSION.md).

The negative results are the deliverable, not a consolation prize. Anyone
can show a backtest that "worked" — that usually just means overfitting went
unchecked. The point of this repository is the process that catches that
before it turns into a live trading decision: pre-registration, a locked
test set that's actually never touched, and regime-robustness checks applied
to anything that looks promising before it's believed.

## Why this project is worth reading, even though both branches "failed"

Most hobby trading-model repos report a backtest and stop. Nearly all of them
are wrong in ways that don't show up until the strategy is live, because
free/cheap data pipelines have well-known failure modes: survivorship bias in
the universe, look-ahead bias in point-in-time features, and hyperparameter
tuning quietly performed on the test set. This project was built to eliminate
those specific failure modes and to produce an honest answer, not a flattering
backtest:

- **Point-in-time correctness everywhere.** Every feature and target is
  computed strictly from data available as of the decision date, checked
  directly by a dedicated leakage-control test suite
  (`tests/test_ml_features_leakage.py`, `tests/test_backtest_lookahead.py`).
- **A walk-forward split with embargo/purge**, not a random train/test split —
  train, validation, and a locked test set separated in time, with an embargo
  region between them equal to the prediction horizon so overlapping labels
  can't leak across the boundary.
- **A pre-registered success framework**, written before any model was fit:
  what would count as a meaningful result, what robustness checks a result
  would have to pass, and what to do in each possible outcome — including
  "no evidence of predictive value," which was treated as a legitimate result
  from the outset, not a failure to explain away.
- **Frozen hyperparameter grids and decision tables**, fixed before the
  corresponding experiments ran and never edited afterward based on what came
  back.
- **Locked test sets that were never touched.** 27 dates in Phase 4, 566 in
  Phase 5. Every experiment in this repository ran against train/validation
  data only, enforced structurally (the locked dates are never passed to
  the functions that compute returns or features for them, not just
  excluded by convention). No result in this repository is a confirmatory
  test-set number.
- **122 automated tests**, including synthetic panels with a known planted
  relationship (so "does the model recover a real signal it's given" is
  checked directly) and dedicated leakage tests (so "does this function ever
  see the future" is checked directly, not just assumed).
- **A 200-seed random-benchmark validation** (Phase 3) confirming the
  backtesting engine itself — costs, execution timing, accounting — behaves
  correctly before any predictive model was ever fit on top of it.

## Project structure

| Phase | What it does | Status |
|---|---|---|
| **Phase 1** | Data ingestion: prices, splits/dividends adjustment, universe membership, identity resolution | Complete |
| **Phase 2** | Staged, validated database of point-in-time security history | Complete |
| **Phase 3** | Point-in-time backtesting engine (execution timing, transaction costs, accounting) plus a 200-seed random-portfolio benchmark to validate the engine itself | Complete |
| **Phase 4** | Predictive-modelling research: feature engineering, walk-forward evaluation, linear and tree-ensemble models, regime-robustness diagnostics | Complete — see [`PHASE4_CONCLUSION_3M.md`](PHASE4_CONCLUSION_3M.md) |
| **Phase 5** | Overnight vs. intraday return decomposition, daily-granularity walk-forward split, block-bootstrap significance test | Complete — see [`PHASE5_CONCLUSION.md`](PHASE5_CONCLUSION.md) |

Phases 4 and 5 are the headline: they're where the pre-registration
discipline, the robustness gates, and the honest negative results all come
together. Phases 1–3 are the infrastructure that makes those results
trustworthy — without a leakage-free, validated data pipeline and
backtesting engine, a "no predictive value" conclusion wouldn't mean
anything.

Source layout:

```
src/
  ingestion/    price/adjustment/pipeline logic (yfinance-backed)
  universe/     historical index membership, identity resolution
  database/     schema and migrations for the staged database
  validation/   data-quality checks run against the staged database
  backtest/     point-in-time backtesting engine (execution, costs, accounting)
  ml/           features, targets, walk-forward splits, models (Phase 4),
                overnight/intraday decomposition + significance test (Phase 5)
tests/          138 tests: unit, synthetic-recovery, and leakage-control
```

## Setup and reproduction

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -q          # 138 tests, no network or database required
```

The database and raw price/membership data are not included in this
repository (see Data below). To regenerate them and reproduce the Phase 4/5
results:

```bash
python3 run_stage_ingestion.py           # builds the staged database from yfinance + reference data
python3 run_phase3_200seed_report.py     # validates the backtesting engine
python3 run_phase4_baselines.py          # linear models, 3-month horizon (default)
python3 run_phase4_baselines.py --horizon 1   # 1-month horizon
python3 run_phase4_trees.py              # tree-ensemble stage
python3 run_phase5_tier0_test.py         # overnight/intraday block-bootstrap test, train+validation only
```

Each script writes a JSON report to the project root; the numbers in
`PHASE4_CONCLUSION_3M.md` and `PHASE5_CONCLUSION.md` come directly from
these reports and are cited by experiment ID (Phase 4) or report filename
(Phase 5) for traceability.

## Data and licensing

No price data or database is included in this repository — it's regenerated
locally from free sources:

- **Prices, splits, dividends:** [`yfinance`](https://github.com/ranaroussi/yfinance).
- **Historical S&P 500 membership:** a vendored copy of
  [fja05680/sp500](https://github.com/fja05680/sp500) (MIT licensed,
  Copyright 2019–2020 Farrell J. Aultman) lives under
  `data/raw/sp500-master/` for attribution; the large CSVs and notebooks
  themselves are excluded from this repository (see `.gitignore`) — clone the
  upstream repository directly if you need them.

This project's own code is MIT licensed — see [`LICENSE`](LICENSE).

## What this project is not

This is a research pipeline, not a trading system, and not investment advice.
No model or signal in this repository has ever been used to place a live or
paper trade. Phase 4's conclusion is that the V1 feature set showed no
robust cross-sectional predictive value at the horizons tested; Phase 5's is
that the apparent overnight effect doesn't survive outside one historical
regime. The honest reading of both is "don't trade this," not "needs more
tuning until it works." Nothing here should be read as a recommendation to
buy, sell, or hold any security.
