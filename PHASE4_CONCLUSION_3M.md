# Phase 4 Conclusion — 3-Month Horizon Modelling Branch

**Status: CLOSED.** This document concludes the 3-month-horizon predictive-modelling
branch of Phase 4. The 1-month and 6-month horizons are explicitly **not** covered
here and are **not** authorised — see §13.

---

## 1. Pre-registered objective and horizon

Per `PHASE4_SPECIFICATION.md`, the Phase 4 research question was:

> Can information that was genuinely available at time *t* predict the subsequent
> cross-sectional performance of stocks over a defined future horizon, sufficiently
> well to construct a portfolio that outperforms appropriate Phase 3 benchmarks
> after realistic transaction costs?

This branch evaluated the **primary horizon: 3 months** (rebalance periods, not
calendar days), on the **PERMISSIVE** universe (the sole primary Phase 4 research
universe; STRICT was retained only as a degenerate-case sanity check and was
confirmed elsewhere to have 0–1 eligible securities per date — not usable for
modelling).

The target was the benchmark-relative excess return
`y_i(t,h) = r_i(t,h) − B(t,h)`, `B(t,h)` = the equal-weighted mean return across
`ELIG(t)`, evaluated via per-date Spearman IC (predicted vs. realised `y`),
averaged across dates — never pooled across all (security, date) rows at once.

**Walk-forward split** (chronological, embargo = horizon = 3 months at each
boundary): **train** 62 dates (2015-01–2020-02, 3 purged), **validation** 13 dates
(2020-06–2021-06, 3 purged), **locked test** 27 dates (2021-10–2023-12) —
**never accessed in this branch** (§11).

**Sample-size context** (`PHASE4_SAMPLE_SIZE_REPORT.json`): at the 3-month horizon,
raw (security, date) pairs (train 25,184 / validation 5,811) substantially overstate
the effective independent sample once cross-sectional and serial correlation are
accounted for — **20 independent time-blocks in train, 4 in validation, 9 in the
locked test set**. This is the standing constraint behind every threshold and gate
used below.

## 2. Models evaluated

Seven model families, all fit/evaluated on **train + validation only**:

- **Historical mean** (no fitting; ranks by trailing `return_12m / 12`)
- **Momentum-only** (best single price-momentum feature, selected on train,
  reported on validation)
- **Ridge regression**
- **Elastic Net**
- **Logistic Regression** (continuous score via `predict_proba − 0.5`)
- **Random Forest** (`RandomForestRegressor`)
- **HistGradientBoosting** (`HistGradientBoostingRegressor`)

Ridge/Logistic/Elastic Net and the two tree models were all hyperparameter-tuned
via an expanding-window inner cross-validation carved entirely out of TRAIN's own
dates (never touching validation or test) — for the tree models, via two **frozen,
pre-declared candidate grids** (Random Forest: 12 candidates; HistGradientBoosting:
16 candidates), fixed before any tree experiment ran and not modified afterward.
No neural network, LSTM, TCN, or Transformer model was used or considered, per the
spec's explicit exclusion.

## 3. Feature ablation structure

The same cumulative ablation ladder was applied to every fitted model family:

`price_only → +volatility_risk → +volume_liquidity → +market_regime → full_v1_feature_set`

(`price_momentum`, `volatility_risk`, `volume_liquidity`, `relative_cross_sectional`,
`market_regime` — the five V1 feature families defined in `config.yaml`'s
`phase4.v1_features` / `src/ml/feature_matrix.py::FEATURE_FAMILIES`.) For the tree
stage, one additional configuration was run: `full_v1_feature_set` with the
`volume_liquidity` family removed (leave-one-family-out; §8).

## 4. Validation-period results — linear models

No linear model showed robust predictive value at any ablation step. Mean IC was
negative across the board; median IC was near zero only for the full-feature-set
Ridge/Elastic Net configurations, with the mean/median gap traced to a specific
regime window (§5, §6 diagnostic below applies to trees, but the same 4-month
window was independently confirmed to explain most of the linear models' mean/
median gap in the earlier `EXPLORATORY_VALIDATION_REGIME_DIAGNOSTIC`).

| Model | Best config | exp_id | Validation mean IC | Median IC |
|---|---|---|---|---|
| Historical mean | — | 35 | −0.195 | −0.218 |
| Momentum-only | `distance_from_200d_ma` | 36 | −0.055 | −0.078 |
| Ridge | full_v1_feature_set, α=0.3 | 49 | −0.120 | +0.004 |
| Elastic Net | full_v1_feature_set, α=3e-05 | 51 | −0.117 | +0.005 |
| Logistic | full_v1_feature_set, C=0.1 | 50 | −0.185 | −0.231 |

Every ridge/logistic/elastic-net × ablation-step combination (experiment IDs
37–51, 15 experiments) showed the same pattern: mean IC in the −0.11 to −0.22
range, with logistic notably remaining clearly negative even after the Aug–Nov
2020 exploratory exclusion diagnostic — the one linear-model result that did
**not** resolve to "regime-driven," and is recorded here as an open, unexplained
oddity rather than glossed over.

## 5. Tree-ensemble result: all 10 combinations, all failed the regime gate

Random Forest and HistGradientBoosting were each fit across the 5 ablation steps
(experiment IDs 52–61, 10 experiments), plus the leave-one-family-out check
(experiment IDs 62–63). Per the pre-declared decision table (§6 of the prior
message thread; frozen before any tree experiment ran):

**Pre-declared "meaningful improvement" criterion:** full-13-month validation mean
IC must exceed the best like-for-like linear baseline at the same ablation step by
≥ 0.02.

**All 10 of 10 combinations nominally cleared this threshold** — deltas over the
matching linear baseline ranged from +0.026 (Random Forest, price_only) to +0.133
(HistGB, full_v1_feature_set). Taken alone, this would look like a broad,
consistent improvement from tree models.

**All 10 of 10 failed the pre-declared Aug–Nov 2020 regime-robustness gate.** The
fraction of each combination's apparent improvement attributable to that single
four-month window ranged from **112% to 269%** (over 100% means the tree model
actually underperformed the linear baseline outside that window — the regime
window's contribution more than accounts for the entire nominal gain). No
combination survived this gate, so **the decision-table outcome is
`NO_IMPROVEMENT_OR_FAILS_ROBUSTNESS`, with zero survivors** — mechanically
equivalent to "no improvement" for the purpose of allocating a confirmatory
test-set slot.

## 6. The two strongest individual results, in full

**HistGradientBoosting, full_v1_feature_set (exp_id 61):** +0.016 mean IC across
all 13 validation months (the single best number produced anywhere in this
branch) — falling to **−0.011** once Aug–Nov 2020 is excluded.

**Random Forest, full_v1_feature_set (exp_id 60):** −0.008 mean IC across all 13
months, falling further to **−0.027** excluding the same four months.

Both patterns are consistent with the same underlying story: a more flexible
model fitting one anomalous, well-documented market-disruption period (the
COVID-recovery cross-sectional rotation) more closely than a linear model can,
without that fit generalising to any other period in the validation window.

## 7. Price-history-length diagnostic — trees did not reproduce the Phase 3 artifact

Phase 3's own diagnostic found a raw pooled correlation of ~0.386 and an
opportunity-adjusted residual-rate correlation of ~0.20 between security
selection frequency and price-history length — a live concern given trees are
generally well-suited to exploiting exactly this kind of "how much data exists"
proxy. The Phase 4 analogue (`price_history_length_asof`, point-in-time-safe by
construction — never counts rows dated after `as_of_date`) was computed for every
tree/ablation-step combination, both as a raw pooled Pearson correlation
(prediction vs. history length) and as a per-date Spearman IC averaged across
dates (the point-in-time, per-date analogue of Phase 3's opportunity-adjusted
control).

Representative result (full_v1_feature_set): Random Forest raw = **0.012**,
residual = **0.021**; HistGB raw = **−0.056**, residual = **−0.015**. Every
combination stayed far below the 90%-of-Phase-3-benchmark failure thresholds
(≈0.347 raw / ≈0.18 residual) used as the pre-declared gate. **The tree models
are not reproducing the Phase 3 price-history-length artifact.**

## 8. Volume/liquidity leave-one-family-out — not a data-availability proxy

Dropping `volume_liquidity` from the full V1 feature set (experiment IDs 62–63)
removed only **19.9%** (Random Forest) and **28.2%** (HistGB) of the full-feature
model's improvement over the linear baseline — well under the pre-declared 80%
failure threshold. If the volume/liquidity family were acting as a coverage- or
availability-driven proxy rather than a genuine feature, removing it would have
been expected to erase most of the gain; it did not. Combined with §7, there is no
evidence the tree models' apparent (regime-confined) edge is an artifact of data
availability.

## 9. Conclusion

**No model evaluated in this branch — historical mean, momentum-only, Ridge,
Elastic Net, Logistic Regression, Random Forest, or HistGradientBoosting —
demonstrated robust, regime-independent predictive value for the V1 feature set
at the 3-month horizon.** The linear models showed uniformly negative-to-flat
validation IC. The tree models showed an apparent edge that, on inspection,
resolved entirely to overfitting a single four-month historical regime rather
than to a stable cross-sectional signal, and this was true of every ablation
step tested for both tree algorithms.

## 10. This is a legitimate negative research result

Per the Phase 4 pre-registered success framework and the standing instruction
governing this research program, "no evidence of predictive value" is an
explicitly valid and successful research outcome — not a failure of the project,
the pipeline, or the modelling effort. The objective throughout has been to
determine *whether* genuine predictive information exists in this feature set at
this horizon, not to manufacture a positive backtest. This branch answers that
question, for the 3-month horizon and the V1 feature set, in the negative.

## 11. Locked test set: never accessed

The locked test period (27 dates, 2021-10 through 2023-12) was not read, queried,
or evaluated at any point in this branch. **No confirmatory test-set result
exists for the 3-month horizon, and none of the 5-slot confirmatory experiment
budget (`max_confirmatory_test_experiments`, §17) has been consumed.**

## 12. Reproducibility record

All results in this document are reproducible from the persisted database tables
(`ml_experiments`, `ml_predictions`) and the JSON reports below. Every experiment
listed was logged with `is_confirmatory=0, touched_locked_test_set=0`.

| Artifact | Contents |
|---|---|
| `PHASE4_SAMPLE_SIZE_REPORT.json` | Split boundaries, effective independent block counts |
| `PHASE4_BASELINES_REPORT.json` | Historical mean, momentum-only, ridge/logistic/elastic-net results (exp_id 35–51) |
| `PHASE4_ELASTIC_NET_DIAGNOSTIC.json` | Diagnosis of the original untuned elastic-net alpha=0.01 collapse (pre-tuning; superseded by exp_id 39/42/45/48/51) |
| `PHASE4_EXPLORATORY_VALIDATION_REGIME_DIAGNOSTIC.json` | Aug–Nov 2020 exclusion re-aggregation across all linear-model configurations |
| `PHASE4_TREES_REPORT.json` | Random Forest / HistGB results, frozen grids, decision-table application (exp_id 52–63) |

Experiment IDs by configuration:

- `historical_mean`: 35 | `momentum_only`: 36
- Linear ablation ladder (ridge / logistic / elastic_net), price_only → full_v1_feature_set: 37–51
- Tree ablation ladder (random_forest / hist_gb), price_only → full_v1_feature_set: 52–61
- Leave-one-family-out (volume_liquidity removed), random_forest / hist_gb: 62, 63

---

## 13. What this document does not do

This document adds no new model, feature, horizon, or experiment. The 1-month and
6-month horizons remain **entirely unauthorised, separate experiment families**
per the standing instruction, and were not run, checked, or referenced for any
result above. Per instruction, work stops here pending explicit approval before
any new experiment family — including the 1-month/6-month horizons — begins.
