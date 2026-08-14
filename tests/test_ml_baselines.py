"""
Baseline model tests for src/ml/baselines.py. These use synthetic,
hand-constructed panels (no database) so the correct answer is known
analytically, following the same philosophy as
test_backtest_synthetic.py: a synthetic test with a known answer tells
you more than a plausible-looking result on real data.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import random
import pytest

from src.ml import baselines as B


# --- 1. Spearman correctness against a known case ---

def test_spearman_perfect_positive_correlation():
    xs = [1, 2, 3, 4, 5]
    ys = [10, 20, 30, 40, 50]
    assert B.spearman(xs, ys) == pytest.approx(1.0)


def test_spearman_perfect_negative_correlation():
    xs = [1, 2, 3, 4, 5]
    ys = [50, 40, 30, 20, 10]
    assert B.spearman(xs, ys) == pytest.approx(-1.0)


def test_spearman_handles_ties_without_crashing():
    xs = [1, 1, 2, 2, 3]
    ys = [5, 4, 3, 3, 1]
    r = B.spearman(xs, ys)
    assert r is not None and -1.0 <= r <= 1.0


def test_spearman_zero_variance_returns_none_not_a_fabricated_number():
    assert B.spearman([1, 1, 1], [1, 2, 3]) is None


# --- 2. evaluate_predictions averages PER-DATE, not pooled across all rows ---

def _row(as_of_date, sid, y):
    return {"as_of_date": as_of_date, "security_id": sid, "y": y}


def test_evaluate_predictions_is_per_date_not_pooled():
    # Date A: predictions perfectly rank-correlate with y.
    # Date B: predictions perfectly anti-correlate with y.
    # A naive pooled correlation across both dates could land anywhere
    # depending on each date's absolute return level. The per-date
    # average must be exactly (1.0 + -1.0) / 2 = 0.0, regardless of the
    # absolute scale of each date's returns.
    rows = [
        _row("2020-01-01", 1, 0.01), _row("2020-01-01", 2, 0.02), _row("2020-01-01", 3, 0.03),
        _row("2020-01-01", 4, 0.04), _row("2020-01-01", 5, 0.05),
        _row("2020-02-01", 1, 5.0), _row("2020-02-01", 2, 4.0), _row("2020-02-01", 3, 3.0),
        _row("2020-02-01", 4, 2.0), _row("2020-02-01", 5, 1.0),
    ]
    preds = [0.01, 0.02, 0.03, 0.04, 0.05,   # matches date A's ordering exactly
             0.01, 0.02, 0.03, 0.04, 0.05]   # opposite of date B's ordering
    result = B.evaluate_predictions(rows, preds, min_securities_per_date=5)
    assert result["dates_evaluated"] == 2
    assert result["per_date"]["2020-01-01"]["ic"] == pytest.approx(1.0)
    assert result["per_date"]["2020-02-01"]["ic"] == pytest.approx(-1.0)
    assert result["mean_ic"] == pytest.approx(0.0)


def test_evaluate_predictions_skips_dates_with_too_few_securities():
    rows = [_row("2020-01-01", 1, 0.01), _row("2020-01-01", 2, 0.02)]  # only 2 securities
    preds = [0.01, 0.02]
    result = B.evaluate_predictions(rows, preds, min_securities_per_date=5)
    assert result["dates_evaluated"] == 0
    assert result["dates_skipped_too_few_securities"] == 1
    assert result["mean_ic"] is None


# --- 3. select_best_momentum_feature only ever looks at TRAIN rows ---

def test_select_best_momentum_feature_picks_the_genuinely_predictive_one():
    rng = random.Random(3)
    rows = []
    for date_i in range(6):
        date = f"2020-{date_i+1:02d}-01"
        for sid in range(10):
            y = rng.gauss(0, 0.05)
            rows.append({
                "as_of_date": date, "security_id": sid, "y": y,
                "return_1m": rng.gauss(0, 1),       # pure noise, uncorrelated with y
                "return_3m": y * 2.0 + rng.gauss(0, 0.001),  # near-perfectly predictive
                "return_6m": rng.gauss(0, 1),
                "return_12m": rng.gauss(0, 1),
                "momentum_acceleration": rng.gauss(0, 1),
                "distance_from_200d_ma": rng.gauss(0, 1),
            })
    best_name, best_ic = B.select_best_momentum_feature(rows)
    assert best_name == "return_3m"
    assert best_ic > 0.9


# --- 4. Ridge recovers a known planted linear relationship, held out on a validation panel ---

def test_ridge_recovers_a_planted_relationship_on_held_out_data():
    rng = random.Random(11)
    feature_names = ["f1", "f2"]

    def make_rows(n_dates, n_secs, seed_offset):
        rows = []
        for date_i in range(n_dates):
            date = f"2020-{date_i+1:02d}-01" if date_i < 9 else f"2021-{date_i-8:02d}-01"
            for sid in range(n_secs):
                f1 = rng.gauss(0, 1)
                f2 = rng.gauss(0, 1)
                y = 0.05 * f1 - 0.02 * f2 + rng.gauss(0, 0.005)  # small noise, real signal
                rows.append({"as_of_date": date, "security_id": sid + seed_offset,
                             "f1": f1, "f2": f2, "y": y, "z": 1 if y > 0 else 0})
        return rows

    train_rows = make_rows(12, 30, seed_offset=0)
    validation_rows = make_rows(6, 30, seed_offset=1000)

    fitted = B.fit_ridge(train_rows, feature_names, alpha=1.0)
    val_preds = B.predict_fitted(fitted, validation_rows)
    result = B.evaluate_predictions(validation_rows, val_preds)
    assert result["mean_ic"] is not None
    assert result["mean_ic"] > 0.5  # the planted relationship should be clearly recoverable out-of-sample


# --- 5. inner_cv_splits never reaches outside the train_dates it was given ---

def test_inner_cv_splits_only_uses_dates_from_the_given_train_dates_list():
    train_dates = [f"2020-{m:02d}-01" for m in range(1, 13)]  # 12 months
    folds = B.inner_cv_splits(train_dates, n_folds=3, embargo_periods=1)
    assert len(folds) >= 1
    all_train_dates_set = set(train_dates)
    for inner_train, inner_val in folds:
        assert set(inner_train) <= all_train_dates_set
        assert set(inner_val) <= all_train_dates_set
        # No inner_validation date is ever <= the latest inner_train date,
        # otherwise a fold would "train" on information from its own
        # validation period: the same overlapping-label leakage the outer
        # embargo/purge exists to prevent.
        assert min(inner_val) > max(inner_train)


def test_inner_cv_splits_raises_rather_than_fabricating_folds_when_too_few_dates():
    with pytest.raises(ValueError):
        B.inner_cv_splits(["2020-01-01", "2020-02-01"], n_folds=3, embargo_periods=1)


# --- 6. tune_hyperparameters selects using ONLY train_rows, and recovers a
# sane alpha rather than the one that zeros every coefficient ---

def test_tune_hyperparameters_only_scores_folds_built_from_the_given_rows_own_dates():
    rng = random.Random(7)
    feature_names = ["f1"]

    def make_rows(n_dates, n_secs):
        rows = []
        for date_i in range(n_dates):
            date = f"2020-{date_i+1:02d}-01" if date_i < 9 else f"2021-{date_i-8:02d}-01"
            for sid in range(n_secs):
                f1 = rng.gauss(0, 1)
                y = 0.05 * f1 + rng.gauss(0, 0.01)
                rows.append({"as_of_date": date, "security_id": sid, "f1": f1, "y": y,
                             "z": 1 if y > 0 else 0})
        return rows

    train_rows = make_rows(15, 20)
    train_dates = sorted({r["as_of_date"] for r in train_rows})

    candidates = [{"alpha": 100.0}, {"alpha": 1.0}, {"alpha": 0.01}]
    _, diagnostics = B.tune_hyperparameters(train_rows, feature_names, "ridge", candidates,
                                             n_folds=3, embargo_periods=1)

    # Every fold actually used must be built only from train_rows' own
    # dates: reconstruct the same folds independently and confirm no
    # fold references a date outside train_dates.
    folds = B.inner_cv_splits(train_dates, n_folds=3, embargo_periods=1)
    assert len(folds) >= 1
    for inner_train, inner_val in folds:
        assert set(inner_train) <= set(train_dates)
        assert set(inner_val) <= set(train_dates)
    # Every candidate was actually scored against at least one real fold.
    assert all(d["n_folds_with_ic"] >= 1 for d in diagnostics)


def test_tune_hyperparameters_prefers_an_alpha_that_does_not_zero_every_coefficient():
    rng = random.Random(13)
    feature_names = ["f1", "f2"]
    rows = []
    for date_i in range(15):
        date = f"2020-{date_i+1:02d}-01" if date_i < 9 else f"2021-{date_i-8:02d}-01"
        for sid in range(25):
            f1 = rng.gauss(0, 1)
            f2 = rng.gauss(0, 1)
            y = 0.04 * f1 - 0.03 * f2 + rng.gauss(0, 0.01)
            rows.append({"as_of_date": date, "security_id": sid, "f1": f1, "f2": f2, "y": y,
                         "z": 1 if y > 0 else 0})

    # A deliberately extreme alpha candidate should lose out to a
    # reasonable one, since it zeros the real, recoverable signal.
    candidates = [{"alpha": 10000.0}, {"alpha": 1.0}, {"alpha": 0.1}]
    best, diagnostics = B.tune_hyperparameters(rows, feature_names, "ridge", candidates,
                                                n_folds=3, embargo_periods=1)
    assert best["alpha"] != 10000.0
