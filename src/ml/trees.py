"""
Tree-ensemble stage of the Phase 4 model -- a deliberately constrained
scope, staged only after the linear baselines had been reviewed.

Two algorithms only, both scikit-learn (no new dependency footprint),
both CPU-only, no neural networks:

  RandomForestRegressor      -- bagging, structurally resistant to
                                 overfitting a small effective sample
                                 (~9 independent test blocks / ~4
                                 independent validation blocks).
  HistGradientBoostingRegressor -- boosting, more expressive, grid is
                                 deliberately biased toward shallow/
                                 low-iteration configurations for the
                                 same sample-size reason.

Both are fit as regressors on y (the benchmark-relative excess return),
matching ridge/elastic_net's framing rather than logistic's classification
framing -- logistic didn't add anything the regressors didn't already
show in the baselines run.

Neither model needs feature scaling -- tree splits are invariant to any
monotonic per-feature transform -- so fit_random_forest/fit_hist_gb store
scaler=None; src.ml.baselines.predict_fitted() already handles that.

The candidate grids below are frozen: pre-declared before any tree
experiment was run, and not to be added to, removed from, or edited after
seeing a result. A grid change is a new experiment, not an edit to this
one.
"""
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from src.ml.baselines import _rows_to_arrays

# random_state is fixed (not tuned, not varied) purely for exact
# reproducibility of a given hyperparameter candidate's fit -- it is not
# itself a hyperparameter candidate and was never part of the grid search.
_RANDOM_STATE = 20240613

FROZEN_RF_GRID = [
    {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
    for n in (100, 300)
    for d in (3, 5, 8)
    for leaf in (20, 50)
]  # 2 x 3 x 2 = 12 candidates

FROZEN_HISTGB_GRID = [
    {"max_depth": d, "learning_rate": lr, "max_iter": it, "min_samples_leaf": leaf}
    for d in (2, 3)
    for lr in (0.01, 0.05)
    for it in (50, 150)
    for leaf in (20, 50)
]  # 2 x 2 x 2 x 2 = 16 candidates


def fit_random_forest(train_rows, feature_names, n_estimators=100, max_depth=5, min_samples_leaf=20):
    X, y, _ = _rows_to_arrays(train_rows, feature_names)
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        random_state=_RANDOM_STATE, n_jobs=-1,
    ).fit(X, y)
    return {"model": model, "scaler": None, "feature_names": feature_names, "model_type": "random_forest",
            "hyperparameters": {"n_estimators": n_estimators, "max_depth": max_depth,
                                 "min_samples_leaf": min_samples_leaf}}


def fit_hist_gb(train_rows, feature_names, max_depth=3, learning_rate=0.05, max_iter=100, min_samples_leaf=20):
    X, y, _ = _rows_to_arrays(train_rows, feature_names)
    model = HistGradientBoostingRegressor(
        max_depth=max_depth, learning_rate=learning_rate, max_iter=max_iter,
        min_samples_leaf=min_samples_leaf, random_state=_RANDOM_STATE,
    ).fit(X, y)
    return {"model": model, "scaler": None, "feature_names": feature_names, "model_type": "hist_gb",
            "hyperparameters": {"max_depth": max_depth, "learning_rate": learning_rate,
                                 "max_iter": max_iter, "min_samples_leaf": min_samples_leaf}}
