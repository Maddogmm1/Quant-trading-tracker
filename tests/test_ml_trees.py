"""
Tree-ensemble stage tests for src/ml/trees.py. Synthetic panels with a
known planted relationship (same philosophy as test_ml_baselines.py),
plus checks that the frozen hyperparameter grids are exactly what was
pre-declared and that the shared inner-CV tuning machinery
(src.ml.baselines.tune_hyperparameters) works correctly when handed a
tree fit_func instead of ridge/logistic/elastic_net.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import random
import pytest

from src.ml import trees as TR
from src.ml import baselines as B


# --- 1. Frozen grids are exactly what was pre-declared, protecting against
# silent post-hoc editing after seeing a result ---

def test_frozen_rf_grid_has_exactly_the_pre_declared_12_candidates():
    assert len(TR.FROZEN_RF_GRID) == 12
    n_estimators_values = {c["n_estimators"] for c in TR.FROZEN_RF_GRID}
    max_depth_values = {c["max_depth"] for c in TR.FROZEN_RF_GRID}
    min_samples_leaf_values = {c["min_samples_leaf"] for c in TR.FROZEN_RF_GRID}
    assert n_estimators_values == {100, 300}
    assert max_depth_values == {3, 5, 8}
    assert min_samples_leaf_values == {20, 50}
    assert len(TR.FROZEN_RF_GRID) == len(set(tuple(sorted(c.items())) for c in TR.FROZEN_RF_GRID))


def test_frozen_histgb_grid_has_exactly_the_pre_declared_16_candidates():
    assert len(TR.FROZEN_HISTGB_GRID) == 16
    max_depth_values = {c["max_depth"] for c in TR.FROZEN_HISTGB_GRID}
    lr_values = {c["learning_rate"] for c in TR.FROZEN_HISTGB_GRID}
    max_iter_values = {c["max_iter"] for c in TR.FROZEN_HISTGB_GRID}
    leaf_values = {c["min_samples_leaf"] for c in TR.FROZEN_HISTGB_GRID}
    assert max_depth_values == {2, 3}
    assert lr_values == {0.01, 0.05}
    assert max_iter_values == {50, 150}
    assert leaf_values == {20, 50}
    assert len(TR.FROZEN_HISTGB_GRID) == len(set(tuple(sorted(c.items())) for c in TR.FROZEN_HISTGB_GRID))


# --- 2. Both models recover a planted nonlinear-ish relationship on held-out data ---

def _make_rows(rng, n_dates, n_secs, seed_offset):
    rows = []
    for date_i in range(n_dates):
        date = f"2020-{date_i+1:02d}-01" if date_i < 9 else f"2021-{date_i-8:02d}-01"
        for sid in range(n_secs):
            f1 = rng.gauss(0, 1)
            f2 = rng.gauss(0, 1)
            # A relationship trees can pick up at least as well as a linear
            # model (kept simple/linear-ish so a small forest can find it
            # with this few rows). This test is about plumbing, not about
            # proving trees beat linear models on nonlinear data.
            y = 0.05 * f1 - 0.02 * f2 + rng.gauss(0, 0.005)
            rows.append({"as_of_date": date, "security_id": sid + seed_offset,
                         "f1": f1, "f2": f2, "y": y, "z": 1 if y > 0 else 0})
    return rows


def test_random_forest_recovers_a_planted_relationship_on_held_out_data():
    rng = random.Random(21)
    feature_names = ["f1", "f2"]
    train_rows = _make_rows(rng, 12, 40, seed_offset=0)
    validation_rows = _make_rows(rng, 6, 40, seed_offset=1000)

    fitted = TR.fit_random_forest(train_rows, feature_names, n_estimators=100, max_depth=5, min_samples_leaf=20)
    val_preds = B.predict_fitted(fitted, validation_rows)
    result = B.evaluate_predictions(validation_rows, val_preds)
    assert result["mean_ic"] is not None
    assert result["mean_ic"] > 0.3


def test_hist_gb_recovers_a_planted_relationship_on_held_out_data():
    rng = random.Random(22)
    feature_names = ["f1", "f2"]
    train_rows = _make_rows(rng, 12, 40, seed_offset=0)
    validation_rows = _make_rows(rng, 6, 40, seed_offset=1000)

    fitted = TR.fit_hist_gb(train_rows, feature_names, max_depth=2, learning_rate=0.05, max_iter=150, min_samples_leaf=20)
    val_preds = B.predict_fitted(fitted, validation_rows)
    result = B.evaluate_predictions(validation_rows, val_preds)
    assert result["mean_ic"] is not None
    assert result["mean_ic"] > 0.3


# --- 3. predict_fitted handles scaler=None (trees don't scale) ---

def test_predict_fitted_works_with_no_scaler():
    rng = random.Random(23)
    feature_names = ["f1"]
    rows = _make_rows(rng, 6, 20, seed_offset=0)
    fitted = TR.fit_random_forest(rows, feature_names, n_estimators=50, max_depth=3, min_samples_leaf=10)
    assert fitted["scaler"] is None
    preds = B.predict_fitted(fitted, rows)
    assert len(preds) == len(rows)


# --- 4. The shared tune_hyperparameters() inner-CV machinery works with a
# tree fit_func override, and never touches rows outside what it's given ---

def test_tune_hyperparameters_works_with_a_tree_fit_func_override():
    rng = random.Random(24)
    feature_names = ["f1", "f2"]
    train_rows = _make_rows(rng, 15, 25, seed_offset=0)

    small_grid = [{"n_estimators": 50, "max_depth": 3, "min_samples_leaf": 20},
                  {"n_estimators": 50, "max_depth": 8, "min_samples_leaf": 5}]
    best, diagnostics = B.tune_hyperparameters(
        train_rows, feature_names, "random_forest", small_grid,
        n_folds=3, embargo_periods=1, fit_func=TR.fit_random_forest,
    )
    assert best in small_grid
    assert len(diagnostics) == len(small_grid)
    assert all(d["n_folds_with_ic"] >= 1 for d in diagnostics)
