"""
Diagnostic for the elastic_net null-IC pattern seen in an early
PHASE4_BASELINES_REPORT.json run (identical mean_hit_rate=0.5455 across
every ablation step, mean_ic=null everywhere).

An identical hit rate regardless of which feature family is included is
the fingerprint of a model whose predictions are a constant -- every
coefficient shrunk to exactly zero by L1 regularisation, leaving only the
intercept. A constant prediction has zero within-date variance, so
Spearman IC is undefined (None) for every date, which is exactly what
evaluate_predictions() returns.

This script doesn't decide anything by policy -- it fits the same
full_v1_feature_set elastic net at several alpha values on the real
TRAIN panel and reports, for each one, how many of the (up to 20)
coefficients are non-zero and whether validation predictions actually
vary. That distinguishes two possibilities:
  (a) genuine finding -- any alpha down to a very small value still
      collapses to all-zero coefficients because the real signal is
      indistinguishable from noise at this sample size, or
  (b) a bug -- some alpha well above the current default (0.01) already
      produces non-trivial coefficients, meaning 0.01 was simply an
      unreasonably strong default for standardised features with this
      few independent time-blocks, and "no signal" would be a premature
      conclusion at the default alone.

Run this locally, same folder as the real database:
    python3 run_phase4_diagnose_elastic_net.py
"""
import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import yaml
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from src.database.db import init_db
from src.backtest.execution import next_rebalance_dates
from src.ml.walk_forward import build_primary_split
from src.ml.feature_matrix import build_panel, ABLATION_STEPS, feature_names_for_families
from src.ml import baselines as B

DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "quant_trader_stage.db")
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "src", "database", "schema.sql")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")

ALPHAS_TO_TRY = [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]


def main():
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
    horizon = p4_cfg["target"]["primary_horizon_months"]

    split = build_primary_split(
        rebalance_dates,
        p4_cfg["walk_forward"]["train_fraction"], p4_cfg["walk_forward"]["validation_fraction"],
        p4_cfg["walk_forward"]["test_fraction"], embargo_periods=horizon,
    )

    conn = init_db(DB_PATH, SCHEMA_PATH, reset=False, force=False)

    print("Building TRAIN panel...")
    train_rows, train_stats = build_panel(conn, cfg, policy_name, split["train_dates"], rebalance_dates,
                                           horizon, predeclared_filters)
    print("Building VALIDATION panel...")
    val_rows, val_stats = build_panel(conn, cfg, policy_name, split["validation_dates"], rebalance_dates,
                                       horizon, predeclared_filters)
    print(f"train rows={train_stats['rows_built']}  validation rows={val_stats['rows_built']}\n")

    # full_v1_feature_set is the last entry in ABLATION_STEPS.
    step_name, families = ABLATION_STEPS[-1]
    feature_names = feature_names_for_families(families)
    print(f"Diagnosing on '{step_name}' ({len(feature_names)} features): {feature_names}\n")

    X_train, y_train, _ = B._rows_to_arrays(train_rows, feature_names)
    scaler = StandardScaler().fit(X_train)
    Xs_train = scaler.transform(X_train)

    X_val, _, _ = B._rows_to_arrays(val_rows, feature_names)
    Xs_val = scaler.transform(X_val)

    report = {"n_features": len(feature_names), "feature_names": feature_names, "runs": []}

    for alpha in ALPHAS_TO_TRY:
        model = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000).fit(Xs_train, y_train)
        n_nonzero = int(np.count_nonzero(model.coef_))
        val_preds = model.predict(Xs_val)
        pred_std = float(np.std(val_preds))
        result = B.evaluate_predictions(val_rows, list(val_preds))
        row = {
            "alpha": alpha,
            "n_nonzero_coefficients": n_nonzero,
            "intercept": float(model.intercept_),
            "validation_prediction_stdev": pred_std,
            "validation_mean_ic": result["mean_ic"],
            "validation_mean_hit_rate": result["mean_hit_rate"],
        }
        report["runs"].append(row)
        print(f"alpha={alpha:<10} nonzero_coefs={n_nonzero:2d}/{len(feature_names)}  "
              f"pred_stdev={pred_std:.6f}  mean_ic={result['mean_ic']}  "
              f"mean_hit_rate={result['mean_hit_rate']}")

    conn.close()

    out_json = os.path.join(PROJECT_ROOT, "PHASE4_ELASTIC_NET_DIAGNOSTIC.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {out_json}.")


if __name__ == "__main__":
    main()
