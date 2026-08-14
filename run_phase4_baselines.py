"""
Baseline models + ablation for the Phase 4 ranking problem. Builds the
TRAIN and VALIDATION panels (never the locked test set -- see the split
boundaries printed below), fits the five baselines, and runs the
PRICE ONLY -> ... -> FULL ablation for the three fitted models
(ridge/logistic/elastic-net). Every run here is exploratory
(ml_experiments.is_confirmatory=0, touched_locked_test_set=0), so nothing
here counts against the confirmatory-experiment cap and nothing touches
the locked test period (2021-10 onward, per PHASE4_SAMPLE_SIZE_REPORT.json).

ridge/logistic/elastic_net hyperparameters are selected via
tune_hyperparameters()'s inner cross-validation on TRAIN's own dates,
rather than left at sklearn's bare defaults -- an earlier run of this
script used the untuned defaults and elastic_net's alpha=0.01 turned out
to zero every coefficient (PHASE4_ELASTIC_NET_DIAGNOSTIC.json), a
hyperparameter artefact rather than a finding about the features.
Selected hyperparameters and the full per-candidate inner-CV diagnostics
are logged into ml_experiments/the JSON report so the choice stays
reproducible and auditable.

This does not build or evaluate any tree-ensemble model beyond these
baselines -- that's a separate step (run_phase4_trees.py) taken only
after these results are reviewed.

--horizon overrides which rebalance-period horizon to evaluate. It must be
one of the horizons already declared in config.yaml's phase4.target
(primary_horizon_months=3, secondary_horizon_months=[1, 6]); the script
refuses any other value, since running an undeclared horizon would be a
new experiment family rather than a pre-registered one. The output report
is written to PHASE4_BASELINES_REPORT.json for the primary horizon, or
PHASE4_BASELINES_REPORT_<N>M.json for a secondary horizon.

Run this locally, in the same folder as the real database (after
`pip install -r requirements.txt` to pick up numpy/scikit-learn):
    python3 run_phase4_baselines.py                # primary horizon (3 months)
    python3 run_phase4_baselines.py --horizon 1     # secondary horizon (1 month)
"""
import argparse
import sys
import os
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import yaml

from src.database.db import init_db
from src.backtest.execution import next_rebalance_dates
from src.ml.walk_forward import build_primary_split
from src.ml.feature_matrix import build_panel, FEATURE_FAMILIES, ABLATION_STEPS, feature_names_for_families
from src.ml import baselines as B

DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "quant_trader_stage.db")
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "src", "database", "schema.sql")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")


def _now():
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"


def _log_experiment(conn, *, label, universe_policy, target_config, feature_config, split_config,
                     model_type, hyperparameters, portfolio_construction, cost_config, notes):
    cur = conn.execute(
        """INSERT INTO ml_experiments
           (experiment_label, created_at, is_confirmatory, touched_locked_test_set, robustness_axis,
            universe_policy, target_config_json, feature_config_json, split_config_json, model_type,
            hyperparameters_json, portfolio_construction_json, cost_config_json, notes)
           VALUES (?,?,0,0,NULL,?,?,?,?,?,?,?,?,?)""",
        (label, _now(), universe_policy, json.dumps(target_config), json.dumps(feature_config),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=None,
                         help="Rebalance-period horizon in months. Must be one of config.yaml's "
                              "phase4.target.primary_horizon_months or secondary_horizon_months. "
                              "Defaults to the primary horizon.")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"ERROR: database not found at {DB_PATH}")
        sys.exit(1)

    cfg = yaml.safe_load(open(CONFIG_PATH))
    bt_cfg = cfg["backtest"]
    p4_cfg = cfg["phase4"]
    predeclared_filters = cfg.get("universe_filters", {})
    start_date = cfg["data"]["price_start_date"]
    end_date = cfg["data"]["price_end_date"]
    rebalance_dates = next_rebalance_dates(start_date, end_date, bt_cfg["rebalance_frequency"])
    policy_name = p4_cfg["primary_universe_policy"]

    primary_horizon = p4_cfg["target"]["primary_horizon_months"]
    secondary_horizons = p4_cfg["target"]["secondary_horizon_months"]
    allowed_horizons = [primary_horizon] + list(secondary_horizons)
    horizon = args.horizon if args.horizon is not None else primary_horizon
    if horizon not in allowed_horizons:
        print(f"ERROR: horizon={horizon} is not one of the pre-declared horizons {allowed_horizons} "
              f"in config.yaml's phase4.target. Running an undeclared horizon would be a new "
              f"experiment family -- refusing.")
        sys.exit(1)
    embargo = horizon
    print(f"Horizon: {horizon} month(s) ({'primary' if horizon == primary_horizon else 'secondary'})\n")

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
    t0 = time.time()
    train_rows, train_stats = build_panel(conn, cfg, policy_name, split["train_dates"], rebalance_dates,
                                           horizon, predeclared_filters)
    print(f"  {train_stats['rows_built']} rows in {time.time()-t0:.0f}s "
          f"(dropped: {train_stats['rows_dropped_missing_feature']} missing-feature, "
          f"{train_stats['rows_dropped_no_label']} no-label; "
          f"{train_stats['truncated_total']} truncated labels)")

    print("Building VALIDATION panel...")
    t0 = time.time()
    val_rows, val_stats = build_panel(conn, cfg, policy_name, split["validation_dates"], rebalance_dates,
                                       horizon, predeclared_filters)
    print(f"  {val_stats['rows_built']} rows in {time.time()-t0:.0f}s "
          f"(dropped: {val_stats['rows_dropped_missing_feature']} missing-feature, "
          f"{val_stats['rows_dropped_no_label']} no-label; "
          f"{val_stats['truncated_total']} truncated labels)\n")

    if not train_rows or not val_rows:
        print("ERROR: empty train or validation panel -- cannot proceed.")
        sys.exit(1)

    target_config = {"benchmark": p4_cfg["target"]["benchmark"], "horizon_months": horizon}
    split_config = {"train_dates_count": len(split["train_dates"]), "validation_dates_count": len(split["validation_dates"]),
                     "embargo_periods": embargo}
    portfolio_construction = {"note": "not applied at the baseline-evaluation stage -- prediction-quality metrics only"}
    cost_config = {"note": "not applied at the baseline-evaluation stage -- applies once a strategy is backtested"}

    report = {"generated_at": _now(), "policy": policy_name, "horizon_months": horizon,
              "train_stats": train_stats, "validation_stats": val_stats, "results": {}}

    # --- Baseline 1: historical mean ---
    print("=== historical_mean ===")
    val_preds = B.predict_historical_mean(val_rows)
    result = B.evaluate_predictions(val_rows, val_preds)
    exp_id = _log_experiment(conn, label="baseline_historical_mean", universe_policy=policy_name,
                              target_config=target_config, feature_config={"features": ["return_12m"]},
                              split_config=split_config, model_type="historical_mean", hyperparameters={},
                              portfolio_construction=portfolio_construction, cost_config=cost_config,
                              notes=json.dumps({k: v for k, v in result.items() if k != "per_date"}))
    _persist_predictions(conn, exp_id, val_rows, val_preds)
    report["results"]["historical_mean"] = {"experiment_id": exp_id, **result}
    print(f"  mean_ic={result['mean_ic']}, mean_hit_rate={result['mean_hit_rate']}")

    # --- Baseline 2: momentum-only (selected on TRAIN, reported on VALIDATION) ---
    print("=== momentum_only ===")
    best_feature, train_ic = B.select_best_momentum_feature(train_rows)
    val_preds = B.predict_momentum_only(val_rows, best_feature)
    result = B.evaluate_predictions(val_rows, val_preds)
    exp_id = _log_experiment(conn, label="baseline_momentum_only", universe_policy=policy_name,
                              target_config=target_config,
                              feature_config={"features": [best_feature], "selected_on": "train", "train_ic": train_ic},
                              split_config=split_config, model_type="momentum_only", hyperparameters={},
                              portfolio_construction=portfolio_construction, cost_config=cost_config,
                              notes=json.dumps({k: v for k, v in result.items() if k != "per_date"}))
    _persist_predictions(conn, exp_id, val_rows, val_preds)
    report["results"]["momentum_only"] = {"experiment_id": exp_id, "selected_feature": best_feature,
                                            "train_ic": train_ic, **result}
    print(f"  selected feature: {best_feature} (train IC={train_ic:.4f}), "
          f"validation mean_ic={result['mean_ic']}, mean_hit_rate={result['mean_hit_rate']}")

    # --- Baselines 3-5, across the ablation ladder ---
    # Candidate grids for the inner-CV tuning. Ridge/logistic defaults
    # (alpha=1.0/C=1.0) were already reasonable and stay in their grids as
    # a sanity check; elastic_net's grid specifically spans well below its
    # old default (0.01) since that default zeroed every coefficient
    # outright.
    fit_funcs = {"ridge": B.fit_ridge, "logistic": B.fit_logistic, "elastic_net": B.fit_elastic_net}
    candidate_grids = {
        "ridge": [{"alpha": a} for a in [10.0, 3.0, 1.0, 0.3, 0.1]],
        "logistic": [{"C": c} for c in [0.03, 0.1, 0.3, 1.0, 3.0]],
        "elastic_net": [{"alpha": a, "l1_ratio": 0.5} for a in [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]],
    }
    for step_name, families in ABLATION_STEPS:
        feature_names = feature_names_for_families(families)
        report["results"].setdefault(step_name, {})
        for model_type, fit_func in fit_funcs.items():
            label = f"{model_type}_{step_name}"
            print(f"=== {label} ({len(feature_names)} features) ===")
            best_candidate, tuning_diagnostics = B.tune_hyperparameters(
                train_rows, feature_names, model_type, candidate_grids[model_type],
                n_folds=3, embargo_periods=embargo,
            )
            print(f"  selected hyperparameters (inner-CV on train only): {best_candidate}")
            fitted = fit_func(train_rows, feature_names, **best_candidate)
            val_preds = B.predict_fitted(fitted, val_rows)
            result = B.evaluate_predictions(val_rows, val_preds)
            exp_id = _log_experiment(
                conn, label=label, universe_policy=policy_name, target_config=target_config,
                feature_config={"families": families, "features": feature_names},
                split_config=split_config, model_type=model_type,
                hyperparameters={**fitted["hyperparameters"], "tuning_diagnostics": tuning_diagnostics},
                portfolio_construction=portfolio_construction,
                cost_config=cost_config, notes=json.dumps({k: v for k, v in result.items() if k != "per_date"}),
            )
            _persist_predictions(conn, exp_id, val_rows, val_preds)
            report["results"][step_name][model_type] = {
                "experiment_id": exp_id, "selected_hyperparameters": best_candidate,
                "tuning_diagnostics": tuning_diagnostics, **result,
            }
            print(f"  mean_ic={result['mean_ic']}, mean_hit_rate={result['mean_hit_rate']}")

    conn.close()

    horizon_suffix = "" if horizon == primary_horizon else f"_{horizon}M"
    out_json = os.path.join(PROJECT_ROOT, f"PHASE4_BASELINES_REPORT{horizon_suffix}.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {out_json}.")
    print("No test-period data was touched. No tree-ensemble model was trained.")


if __name__ == "__main__":
    main()
