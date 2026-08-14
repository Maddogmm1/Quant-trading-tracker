"""
Tree-ensemble stage of the Phase 4 model -- a deliberately constrained
scope, staged only after the linear baselines had been reviewed. The
following were pre-declared and frozen before this script was written:

  - Algorithms: RandomForestRegressor and HistGradientBoostingRegressor
    only (src/ml/trees.py). No neural networks, no XGBoost/LightGBM.
  - Candidate grids: src.ml.trees.FROZEN_RF_GRID (12) and
    FROZEN_HISTGB_GRID (16). Not edited after seeing any result.
  - "Meaningful improvement" criterion: a tree/ablation-step combination's
    full-13-month validation mean IC must beat the best same-ablation-step
    linear baseline's mean IC (from PHASE4_BASELINES_REPORT.json) by
    >= MEANINGFUL_IC_IMPROVEMENT.
  - Robustness gates (all must pass for a combination to "survive"):
      (a) regime robustness -- the improvement over the linear baseline
          must not be >=80% attributable to the Aug-Nov 2020 window
          (EXPLORATORY_VALIDATION_REGIME_DIAGNOSTIC.json's own finding).
      (b) price-history-length robustness -- prediction correlation with
          point-in-time price-history length must stay well below the
          Phase 3 selection-rate benchmarks (~0.386 raw / ~0.20 residual).
      (c) leave-one-family-out robustness (full_v1_feature_set only) --
          the improvement must not be >=80% attributable to the
          volume_liquidity family alone.
  - Decision table (frozen, applied mechanically below, not judged
    per-combination after the fact): no survivors -> NO_IMPROVEMENT;
    one survivor -> GENUINELY_PROMISING; more than one -> MULTIPLE_PROMISING
    (reported for manual review, never auto-selected).

Every experiment here is exploratory (is_confirmatory=0,
touched_locked_test_set=0), so the confirmatory-experiment cap is
untouched by this script and the locked test period is never read.

Requires PHASE4_BASELINES_REPORT.json (from run_phase4_baselines.py) to
already exist in this folder, for the like-for-like linear-baseline
comparison. This can take noticeably longer to run than the baselines
script -- roughly (12+16) candidates x 3 inner-CV folds x 6 feature-set
contexts (5 ablation steps + 1 leave-one-out) plus final refits.

Run this locally, in the same folder as the real database:
    python3 run_phase4_trees.py
"""
import sys
import os
import json
import time
import statistics

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import yaml

from src.database.db import init_db
from src.backtest.execution import next_rebalance_dates
from src.ml.walk_forward import build_primary_split
from src.ml.feature_matrix import build_panel, FEATURE_FAMILIES, ABLATION_STEPS, feature_names_for_families
from src.ml import features as F
from src.ml import baselines as B
from src.ml import trees as TR

DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "quant_trader_stage.db")
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "src", "database", "schema.sql")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
BASELINES_REPORT_PATH = os.path.join(PROJECT_ROOT, "PHASE4_BASELINES_REPORT.json")
REGIME_DIAGNOSTIC_PATH = os.path.join(PROJECT_ROOT, "PHASE4_EXPLORATORY_VALIDATION_REGIME_DIAGNOSTIC.json")

MEANINGFUL_IC_IMPROVEMENT = 0.02
REGIME_EXCLUDED_MONTHS = ["2020-08-01", "2020-09-01", "2020-10-01", "2020-11-01"]
# Phase 3's own selection-rate benchmarks (PHASE3_200SEED_FINAL_REPORT.md):
# raw pooled correlation ~0.386, opportunity-adjusted "residual" rate
# correlation ~0.20. "Approaching" these magnitudes is treated as a
# robustness failure per the pre-declared gate.
PHASE3_RAW_BENCHMARK = 0.386
PHASE3_RESIDUAL_BENCHMARK = 0.20
PRICE_HISTORY_RAW_FAIL_THRESHOLD = 0.9 * PHASE3_RAW_BENCHMARK
PRICE_HISTORY_RESIDUAL_FAIL_THRESHOLD = 0.9 * PHASE3_RESIDUAL_BENCHMARK
REGIME_ATTRIBUTION_FAIL_FRACTION = 0.8
LEAVE_ONE_OUT_ATTRIBUTION_FAIL_FRACTION = 0.8

FIT_FUNCS = {"random_forest": TR.fit_random_forest, "hist_gb": TR.fit_hist_gb}
FROZEN_GRIDS = {"random_forest": TR.FROZEN_RF_GRID, "hist_gb": TR.FROZEN_HISTGB_GRID}


def _now():
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"


def _log_experiment(conn, *, label, robustness_axis, universe_policy, target_config, feature_config,
                     split_config, model_type, hyperparameters, portfolio_construction, cost_config, notes):
    cur = conn.execute(
        """INSERT INTO ml_experiments
           (experiment_label, created_at, is_confirmatory, touched_locked_test_set, robustness_axis,
            universe_policy, target_config_json, feature_config_json, split_config_json, model_type,
            hyperparameters_json, portfolio_construction_json, cost_config_json, notes)
           VALUES (?,?,0,0,?,?,?,?,?,?,?,?,?,?)""",
        (label, _now(), robustness_axis, universe_policy, json.dumps(target_config), json.dumps(feature_config),
         json.dumps(split_config), model_type, json.dumps(hyperparameters),
         json.dumps(portfolio_construction), json.dumps(cost_config), notes),
    )
    conn.commit()
    return cur.lastrowid


def _persist_predictions(conn, experiment_id, rows, predictions):
    for r, pred in zip(rows, predictions):
        conn.execute(
            """INSERT OR REPLACE INTO ml_predictions
               (experiment_id, as_of_date, security_id, predicted_value, realised_label, label_truncated)
               VALUES (?,?,?,?,?,?)""",
            (experiment_id, r["as_of_date"], r["security_id"], float(pred), r["y"], int(r["label_truncated"])),
        )
    conn.commit()


def _excl_regime(per_date_dict):
    """mean/median IC over the same per-date dict evaluate_predictions()
    already produced, with REGIME_EXCLUDED_MONTHS removed -- a read-only
    re-aggregation, never a new fit."""
    ics = [v["ic"] for d, v in per_date_dict.items() if d not in REGIME_EXCLUDED_MONTHS and v["ic"] is not None]
    return {
        "n_months": sum(1 for d in per_date_dict if d not in REGIME_EXCLUDED_MONTHS),
        "mean_ic": statistics.mean(ics) if ics else None,
        "median_ic": statistics.median(ics) if ics else None,
    }


def _price_history_length_diagnostic(conn, rows, predictions):
    """Raw pooled Pearson correlation (mirrors Phase 3's raw pooled
    selection-frequency-vs-history-length check) and a per-date Spearman
    IC of prediction-vs-history-length averaged across dates (the Phase 4
    analogue of Phase 3's opportunity-adjusted "residual rate" control).
    Both remove a shared, mechanical across-the-board confound rather than
    treating the raw pooled number as the whole story. Diagnostic only:
    history length is never a V1 feature and never enters any model."""
    lengths = [F.price_history_length_asof(conn, r["security_id"], r["as_of_date"]) for r in rows]
    raw_corr = B._pearson(list(predictions), lengths)

    by_date_preds, by_date_lengths = {}, {}
    for r, pred, length in zip(rows, predictions, lengths):
        by_date_preds.setdefault(r["as_of_date"], []).append(pred)
        by_date_lengths.setdefault(r["as_of_date"], []).append(length)
    per_date_ics = []
    for d in by_date_preds:
        ic = B.spearman(by_date_preds[d], by_date_lengths[d])
        if ic is not None:
            per_date_ics.append(ic)
    residual_corr = statistics.mean(per_date_ics) if per_date_ics else None

    return {
        "raw_pooled_correlation": raw_corr,
        "residual_per_date_ic": residual_corr,
        "phase3_raw_benchmark": PHASE3_RAW_BENCHMARK,
        "phase3_residual_benchmark": PHASE3_RESIDUAL_BENCHMARK,
        "approaches_phase3_raw_benchmark": (raw_corr is not None and abs(raw_corr) >= PRICE_HISTORY_RAW_FAIL_THRESHOLD),
        "approaches_phase3_residual_benchmark": (residual_corr is not None
                                                  and abs(residual_corr) >= PRICE_HISTORY_RESIDUAL_FAIL_THRESHOLD),
    }


def _run_one(conn, model_type, feature_names, train_rows, val_rows, universe_policy, target_config,
             split_config, portfolio_construction, cost_config, families_label, robustness_axis, label):
    fit_func = FIT_FUNCS[model_type]
    grid = FROZEN_GRIDS[model_type]
    t0 = time.time()
    best_candidate, tuning_diagnostics = B.tune_hyperparameters(
        train_rows, feature_names, model_type, grid, n_folds=3, embargo_periods=3, fit_func=fit_func,
    )
    fitted = fit_func(train_rows, feature_names, **best_candidate)
    val_preds = B.predict_fitted(fitted, val_rows)
    result = B.evaluate_predictions(val_rows, val_preds)
    excl = _excl_regime(result["per_date"])
    history_diag = _price_history_length_diagnostic(conn, val_rows, val_preds)

    exp_id = _log_experiment(
        conn, label=label, robustness_axis=robustness_axis, universe_policy=universe_policy,
        target_config=target_config, feature_config={"families": families_label, "features": feature_names},
        split_config=split_config, model_type=model_type,
        hyperparameters={**fitted["hyperparameters"], "tuning_diagnostics": tuning_diagnostics},
        portfolio_construction=portfolio_construction, cost_config=cost_config,
        notes=json.dumps({k: v for k, v in result.items() if k != "per_date"}),
    )
    _persist_predictions(conn, exp_id, val_rows, val_preds)
    print(f"  [{time.time()-t0:.0f}s] selected={best_candidate}  "
          f"mean_ic_full13={result['mean_ic']}  mean_ic_excl4={excl['mean_ic']}")

    return {
        "experiment_id": exp_id, "selected_hyperparameters": best_candidate,
        "full_13_months": {"mean_ic": result["mean_ic"], "median_ic": result["median_ic"]},
        "excluding_aug_nov_2020": excl,
        "price_history_length_diagnostic": history_diag,
        "per_date": result["per_date"],
    }


def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: database not found at {DB_PATH}")
        sys.exit(1)
    if not os.path.exists(BASELINES_REPORT_PATH):
        print(f"ERROR: {BASELINES_REPORT_PATH} not found -- run run_phase4_baselines.py first "
              "(needed for the like-for-like linear-baseline comparison).")
        sys.exit(1)

    with open(BASELINES_REPORT_PATH) as f:
        baselines_report = json.load(f)

    cfg = yaml.safe_load(open(CONFIG_PATH))
    bt_cfg = cfg["backtest"]
    p4_cfg = cfg["phase4"]
    predeclared_filters = cfg.get("universe_filters", {})
    start_date = cfg["data"]["price_start_date"]
    end_date = cfg["data"]["price_end_date"]
    rebalance_dates = next_rebalance_dates(start_date, end_date, bt_cfg["rebalance_frequency"])
    policy_name = p4_cfg["primary_universe_policy"]
    horizon = p4_cfg["target"]["primary_horizon_months"]
    embargo = horizon

    split = build_primary_split(
        rebalance_dates,
        p4_cfg["walk_forward"]["train_fraction"], p4_cfg["walk_forward"]["validation_fraction"],
        p4_cfg["walk_forward"]["test_fraction"], embargo_periods=embargo,
    )
    print(f"Train: {len(split['train_dates'])} dates ({split['train_dates'][0]}..{split['train_dates'][-1]})")
    print(f"Validation: {len(split['validation_dates'])} dates "
          f"({split['validation_dates'][0]}..{split['validation_dates'][-1]})")
    print(f"[NOT TOUCHED] Locked test: {len(split['test_dates'])} dates "
          f"({split['test_dates'][0]}..{split['test_dates'][-1]})\n")

    conn = init_db(DB_PATH, SCHEMA_PATH, reset=False, force=False)

    print("Building TRAIN panel...")
    train_rows, train_stats = build_panel(conn, cfg, policy_name, split["train_dates"], rebalance_dates,
                                           horizon, predeclared_filters)
    print("Building VALIDATION panel...")
    val_rows, val_stats = build_panel(conn, cfg, policy_name, split["validation_dates"], rebalance_dates,
                                       horizon, predeclared_filters)
    print(f"train rows={train_stats['rows_built']}  validation rows={val_stats['rows_built']}\n")

    if not train_rows or not val_rows:
        print("ERROR: empty train or validation panel -- cannot proceed.")
        sys.exit(1)

    target_config = {"benchmark": p4_cfg["target"]["benchmark"], "horizon_months": horizon}
    split_config = {"train_dates_count": len(split["train_dates"]), "validation_dates_count": len(split["validation_dates"]),
                     "embargo_periods": embargo}
    portfolio_construction = {"note": "not applied at the tree-evaluation stage -- prediction-quality metrics only"}
    cost_config = {"note": "not applied at the tree-evaluation stage"}

    report = {"generated_at": _now(), "policy": policy_name, "horizon_months": horizon,
              "meaningful_ic_improvement_threshold": MEANINGFUL_IC_IMPROVEMENT,
              "frozen_grids": {"random_forest": TR.FROZEN_RF_GRID, "hist_gb": TR.FROZEN_HISTGB_GRID},
              "train_stats": train_stats, "validation_stats": val_stats, "results": {}}

    # --- Main ablation ladder, both tree algorithms, per pre-declared grids ---
    for step_name, families in ABLATION_STEPS:
        feature_names = feature_names_for_families(families)
        report["results"].setdefault(step_name, {})
        for model_type in FIT_FUNCS:
            print(f"=== {model_type}_{step_name} ({len(feature_names)} features) ===")
            report["results"][step_name][model_type] = _run_one(
                conn, model_type, feature_names, train_rows, val_rows, policy_name, target_config,
                split_config, portfolio_construction, cost_config, families_label=families,
                robustness_axis=None, label=f"{model_type}_{step_name}",
            )

    # --- Leave-one-family-out (volume_liquidity), full feature set minus that family ---
    loo_families = ["price_momentum", "volatility_risk", "relative_cross_sectional", "market_regime"]
    loo_feature_names = feature_names_for_families(loo_families)
    report["results"]["full_v1_minus_volume_liquidity"] = {}
    for model_type in FIT_FUNCS:
        print(f"=== {model_type}_full_v1_minus_volume_liquidity ({len(loo_feature_names)} features) ===")
        report["results"]["full_v1_minus_volume_liquidity"][model_type] = _run_one(
            conn, model_type, loo_feature_names, train_rows, val_rows, policy_name, target_config,
            split_config, portfolio_construction, cost_config, families_label=loo_families,
            robustness_axis="leave_one_family_out_volume_liquidity",
            label=f"{model_type}_full_v1_minus_volume_liquidity",
        )

    conn.close()

    # --- Apply the pre-declared, frozen decision table mechanically ---
    decisions = {}
    for step_name, families in ABLATION_STEPS:
        linear_step = baselines_report["results"][step_name]
        best_linear_full = max(linear_step[m]["mean_ic"] for m in ("ridge", "logistic", "elastic_net")
                                if linear_step[m]["mean_ic"] is not None)
        best_linear_excl = None  # computed below only if needed for the regime gate

        for model_type in FIT_FUNCS:
            entry = report["results"][step_name][model_type]
            combo_label = f"{model_type}_{step_name}"
            delta_full = entry["full_13_months"]["mean_ic"] - best_linear_full
            clears_threshold = delta_full >= MEANINGFUL_IC_IMPROVEMENT

            gates = {"clears_threshold": clears_threshold}
            if clears_threshold:
                hist = entry["price_history_length_diagnostic"]
                gates["price_history_length_ok"] = not (hist["approaches_phase3_raw_benchmark"]
                                                          or hist["approaches_phase3_residual_benchmark"])
                # Regime gate: compare the SAME delta, computed excluding
                # Aug-Nov 2020 on both sides, against the full-period delta.
                best_linear_models_excl = []
                for m in ("ridge", "logistic", "elastic_net"):
                    pd_ = linear_step[m].get("per_date")
                    if pd_:
                        best_linear_models_excl.append(_excl_regime(pd_)["mean_ic"])
                best_linear_excl = max([v for v in best_linear_models_excl if v is not None], default=None)
                if best_linear_excl is not None and entry["excluding_aug_nov_2020"]["mean_ic"] is not None:
                    delta_excl = entry["excluding_aug_nov_2020"]["mean_ic"] - best_linear_excl
                    fraction_from_regime = 1.0 - (delta_excl / delta_full) if delta_full != 0 else None
                    gates["regime_robust"] = (fraction_from_regime is None
                                               or fraction_from_regime < REGIME_ATTRIBUTION_FAIL_FRACTION)
                    gates["fraction_of_gain_from_regime_window"] = fraction_from_regime
                else:
                    gates["regime_robust"] = None

                if step_name == "full_v1_feature_set":
                    loo_entry = report["results"]["full_v1_minus_volume_liquidity"][model_type]
                    delta_loo = loo_entry["full_13_months"]["mean_ic"] - best_linear_full
                    fraction_from_volume = 1.0 - (delta_loo / delta_full) if delta_full != 0 else None
                    gates["leave_one_out_robust"] = (fraction_from_volume is None
                                                      or fraction_from_volume < LEAVE_ONE_OUT_ATTRIBUTION_FAIL_FRACTION)
                    gates["fraction_of_gain_from_volume_liquidity_family"] = fraction_from_volume
                else:
                    gates["leave_one_out_robust"] = None

                survives = all(v is not False for v in
                                (gates.get("price_history_length_ok"), gates.get("regime_robust"),
                                 gates.get("leave_one_out_robust")))
            else:
                survives = False

            decisions[combo_label] = {
                "delta_vs_best_linear_baseline_full13": delta_full,
                "gates": gates, "survives": survives,
            }

    survivors = [k for k, v in decisions.items() if v["survives"]]
    if not survivors:
        outcome = "NO_IMPROVEMENT_OR_FAILS_ROBUSTNESS"
    elif len(survivors) == 1:
        outcome = "GENUINELY_PROMISING"
    else:
        outcome = "MULTIPLE_PROMISING"

    report["decision_table_application"] = {
        "meaningful_ic_improvement_threshold": MEANINGFUL_IC_IMPROVEMENT,
        "regime_attribution_fail_fraction": REGIME_ATTRIBUTION_FAIL_FRACTION,
        "leave_one_out_attribution_fail_fraction": LEAVE_ONE_OUT_ATTRIBUTION_FAIL_FRACTION,
        "per_combination": decisions,
        "survivors": survivors,
        "outcome": outcome,
        "note": ("This classification is mechanical, applying rules frozen before any tree experiment "
                 "was run. It does not constitute evidence of genuine predictive value on its own -- "
                 "this stage only determines whether validation evidence is strong enough to justify "
                 "requesting a confirmatory test-set slot. The locked test set has not been accessed."),
    }

    out_json = os.path.join(PROJECT_ROOT, "PHASE4_TREES_REPORT.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {out_json}.")
    print(f"Decision-table outcome: {outcome} (survivors: {survivors})")
    print("No test-period data was touched.")


if __name__ == "__main__":
    main()
