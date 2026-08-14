"""
Baseline models for the Phase 4 ranking problem, plus the PRICE ONLY ->
... -> FULL ablation over feature families.

All fitting happens on a TRAIN panel; every reported evaluation number
comes from a separate VALIDATION panel the model never saw during
fitting. This module is never called with test-period rows -- callers are
responsible for only ever passing train_dates/validation_dates panels
(see run_phase4_baselines.py).

tune_hyperparameters() below selects hyperparameters via an inner
cross-validation carved entirely out of the TRAIN panel's own dates,
never the validation or test panel, since selection must never touch the
data the final evaluation is reported on. fit_ridge/fit_logistic/
fit_elastic_net still accept bare sklearn defaults when called directly
(e.g. for a quick one-off check), but run_phase4_baselines.py always goes
through tune_hyperparameters() first.

Uses numpy + scikit-learn, CPU-only. Deliberately avoids scipy: Spearman
correlation is implemented directly as a Pearson correlation of
within-date ranks, keeping the dependency footprint to just
linear/ridge/logistic/elastic-net/tree-ensembles.
"""
import statistics
from collections import defaultdict

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNet
from sklearn.preprocessing import StandardScaler

from src.ml.feature_matrix import FEATURE_FAMILIES


def _ranks(values):
    """Average-rank handling for ties, 0-indexed float ranks."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs, ys):
    if len(xs) < 2:
        return None
    sx = statistics.pstdev(xs)
    sy = statistics.pstdev(ys)
    if sx == 0 or sy == 0:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    return cov / (sx * sy)


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _rows_to_arrays(rows, feature_names):
    X = np.array([[r[name] for name in feature_names] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=float)
    z = np.array([r["z"] for r in rows], dtype=float)
    return X, y, z


def evaluate_predictions(rows, predictions, min_securities_per_date=5):
    """Per-as_of_date Spearman IC (predicted vs realised y) and hit rate
    (sign agreement), then averaged across dates. Never a single pooled
    correlation across all (security, date) rows at once, which would
    conflate cross-sectional skill within a date with the shared
    across-date market component -- the same reasoning behind
    benchmark-relative target construction, applied here to evaluation."""
    by_date = defaultdict(list)
    for r, pred in zip(rows, predictions):
        by_date[r["as_of_date"]].append((pred, r["y"]))

    ics, hit_rates = [], []
    per_date = {}
    for date, pairs in sorted(by_date.items()):
        if len(pairs) < min_securities_per_date:
            continue
        preds = [p for p, _ in pairs]
        actuals = [a for _, a in pairs]
        ic = spearman(preds, actuals)
        hits = sum(1 for p, a in pairs if (p > 0) == (a > 0))
        hit_rate = hits / len(pairs)
        if ic is not None:
            ics.append(ic)
        hit_rates.append(hit_rate)
        per_date[date] = {"ic": ic, "hit_rate": hit_rate, "n_securities": len(pairs)}

    return {
        "dates_evaluated": len(per_date),
        "dates_skipped_too_few_securities": len(by_date) - len(per_date),
        "mean_ic": statistics.mean(ics) if ics else None,
        "median_ic": statistics.median(ics) if ics else None,
        "stdev_ic": statistics.stdev(ics) if len(ics) > 1 else None,
        "mean_hit_rate": statistics.mean(hit_rates) if hit_rates else None,
        "per_date": per_date,
    }


# --- Baseline 1: historical mean return -------------------------------------------

def predict_historical_mean(rows):
    """No fitting -- ranks by the security's own trailing mean monthly
    return, approximated as return_12m / 12 (already a computed V1
    feature, so this introduces no new leakage surface)."""
    return [r["return_12m"] / 12.0 if r["return_12m"] is not None else 0.0 for r in rows]


# --- Baseline 2: momentum-only ranking (best single price/momentum feature, chosen on TRAIN, reported on VALIDATION) ---

def select_best_momentum_feature(train_rows):
    """Evaluates each price_momentum family feature's own mean IC on the
    TRAIN panel and returns the name of the best one. Selection happens
    on train, never on validation or test, so the validation-set number
    reported for this baseline is a genuine out-of-sample check of that
    choice."""
    best_name, best_ic = None, -2.0
    for name in FEATURE_FAMILIES["price_momentum"]:
        preds = [r[name] for r in train_rows]
        result = evaluate_predictions(train_rows, preds)
        ic = result["mean_ic"]
        if ic is not None and ic > best_ic:
            best_ic = ic
            best_name = name
    return best_name, best_ic


def predict_momentum_only(rows, feature_name):
    return [r[feature_name] for r in rows]


# --- Baselines 3-5: ridge / logistic / elastic net --------------------------------

def fit_ridge(train_rows, feature_names, alpha=1.0):
    X, y, _ = _rows_to_arrays(train_rows, feature_names)
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=alpha).fit(scaler.transform(X), y)
    return {"model": model, "scaler": scaler, "feature_names": feature_names,
            "model_type": "ridge", "hyperparameters": {"alpha": alpha}}


def fit_logistic(train_rows, feature_names, C=1.0):
    X, y, z = _rows_to_arrays(train_rows, feature_names)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=C, max_iter=1000).fit(scaler.transform(X), z)
    return {"model": model, "scaler": scaler, "feature_names": feature_names,
            "model_type": "logistic", "hyperparameters": {"C": C}}


def fit_elastic_net(train_rows, feature_names, alpha=0.01, l1_ratio=0.5):
    X, y, _ = _rows_to_arrays(train_rows, feature_names)
    scaler = StandardScaler().fit(X)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000).fit(scaler.transform(X), y)
    return {"model": model, "scaler": scaler, "feature_names": feature_names,
            "model_type": "elastic_net", "hyperparameters": {"alpha": alpha, "l1_ratio": l1_ratio}}


def predict_fitted(fitted, rows):
    X, _, _ = _rows_to_arrays(rows, fitted["feature_names"])
    # Tree models (src/ml/trees.py) don't scale features -- splits are
    # invariant to monotonic per-feature transforms, so a scaler would be
    # pure overhead, and fit_random_forest/fit_hist_gb store scaler=None.
    Xs = X if fitted["scaler"] is None else fitted["scaler"].transform(X)
    if fitted["model_type"] == "logistic":
        # predict_proba's positive-class probability -- a continuous
        # ranking signal, not just the 0/1 class label.
        return list(fitted["model"].predict_proba(Xs)[:, 1] - 0.5)
    return list(fitted["model"].predict(Xs))


# --- Hyperparameter tuning: selection must happen from data the final
# reported evaluation never also sees. Fitting once with a bare sklearn
# default isn't a neutral choice -- elastic_net's default alpha=0.01
# turned out to zero every coefficient outright
# (PHASE4_ELASTIC_NET_DIAGNOSTIC.json), which is a hyperparameter
# artefact, not a finding about the features. Tuning on VALIDATION and
# reporting that same validation score would be leakage (validation stops
# being an honest out-of-sample check); tuning against the locked TEST
# set is forbidden outright. So selection here runs an expanding-window
# inner cross-validation carved entirely out of TRAIN's own dates, with
# the same embargo/purge convention build_primary_split uses at the outer
# boundaries applied at each inner fold boundary too -- otherwise
# overlapping labels would leak across an inner fold split the same way
# they would across the outer train/validation split. -----------

def inner_cv_splits(train_dates, n_folds=3, embargo_periods=1):
    """n_folds expanding-window (inner_train, inner_validation) date
    splits, built only from train_dates (never validation_dates or
    test_dates). Fold k's inner_train is the first k/(n_folds+1) share of
    train_dates; its inner_validation is the next equal-sized share,
    after skipping embargo_periods dates immediately following
    inner_train (purging the overlapping-label boundary, same rule as the
    outer split). Folds with an empty inner_train or inner_validation are
    skipped, not padded or fabricated."""
    n = len(train_dates)
    chunk = n // (n_folds + 1)
    if chunk < 1:
        raise ValueError(f"not enough train dates ({n}) for {n_folds} inner folds")
    splits = []
    for k in range(1, n_folds + 1):
        train_end = k * chunk
        val_start = train_end + embargo_periods
        val_end = val_start + chunk
        if val_end > n:
            break
        inner_train = train_dates[:train_end]
        inner_val = train_dates[val_start:val_end]
        if inner_train and inner_val:
            splits.append((inner_train, inner_val))
    return splits


def tune_hyperparameters(train_rows, feature_names, model_type, candidates,
                          n_folds=3, embargo_periods=1, min_securities_per_date=5, fit_func=None):
    """Selects the best candidate hyperparameter dict for model_type using
    only train_rows, via inner_cv_splits. For each candidate, fits on each
    fold's inner_train and evaluates mean IC on that fold's
    inner_validation, never touching the real held-out validation set or
    the locked test set; the candidate with the best average IC across
    folds wins. If every candidate collapses to an undefined IC in every
    fold (e.g. every alpha tried still zeros every coefficient), falls
    back to the first candidate and flags that in the returned
    diagnostics rather than picking silently.

    model_type selects fit_ridge/fit_logistic/fit_elastic_net when
    fit_func is not given (backward-compatible with the original three
    baselines). Pass fit_func explicitly (e.g. src.ml.trees.fit_random_forest)
    to reuse this same leakage-safe inner-CV selection for any other
    model family -- model_type is then just a label for logging/diagnostics.

    Returns (best_candidate: dict, diagnostics: list of
    {"candidate", "mean_ic_across_folds", "n_folds_with_ic"})."""
    if fit_func is None:
        fit_func = {"ridge": fit_ridge, "logistic": fit_logistic, "elastic_net": fit_elastic_net}[model_type]
    train_dates = sorted(set(r["as_of_date"] for r in train_rows))
    folds = inner_cv_splits(train_dates, n_folds=n_folds, embargo_periods=embargo_periods)
    if not folds:
        raise ValueError("could not build any inner CV folds from the given train dates")

    diagnostics = []
    for candidate in candidates:
        fold_ics = []
        for inner_train_dates, inner_val_dates in folds:
            inner_train_set, inner_val_set = set(inner_train_dates), set(inner_val_dates)
            inner_train_rows = [r for r in train_rows if r["as_of_date"] in inner_train_set]
            inner_val_rows = [r for r in train_rows if r["as_of_date"] in inner_val_set]
            if not inner_train_rows or not inner_val_rows:
                continue
            fitted = fit_func(inner_train_rows, feature_names, **candidate)
            preds = predict_fitted(fitted, inner_val_rows)
            result = evaluate_predictions(inner_val_rows, preds, min_securities_per_date=min_securities_per_date)
            if result["mean_ic"] is not None:
                fold_ics.append(result["mean_ic"])
        diagnostics.append({
            "candidate": candidate,
            "mean_ic_across_folds": statistics.mean(fold_ics) if fold_ics else None,
            "n_folds_with_ic": len(fold_ics),
        })

    scored = [d for d in diagnostics if d["mean_ic_across_folds"] is not None]
    if not scored:
        best = candidates[0]
        for d in diagnostics:
            d["fallback_used"] = True
    else:
        best = max(scored, key=lambda d: d["mean_ic_across_folds"])["candidate"]

    return best, diagnostics
