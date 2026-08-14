"""
Walk-forward split tests for src/ml/walk_forward.py. No random split is
exercised here because none exists in the module; these tests check the
chronological split and embargo/purge logic, not randomness.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest

from src.ml.walk_forward import build_primary_split, generate_expanding_retrain_checkpoints


def _dates(n):
    """n synthetic monthly ISO dates, 2015-01 onward. Matches the shape
    of the real 108-month PERMISSIVE period closely enough for these tests."""
    import datetime
    out = []
    d = datetime.date(2015, 1, 1)
    for _ in range(n):
        out.append(d.isoformat())
        # step forward one month
        if d.month == 12:
            d = d.replace(year=d.year + 1, month=1)
        else:
            d = d.replace(month=d.month + 1)
    return out


# --- 1. Fractions must sum to 1.0 ---

def test_fractions_must_sum_to_one(conn=None):
    with pytest.raises(ValueError):
        build_primary_split(_dates(108), 0.6, 0.2, 0.3, embargo_periods=3)  # sums to 1.1


# --- 2. The three windows are chronologically contiguous and non-overlapping ---

def test_windows_are_chronological_and_non_overlapping():
    dates = _dates(108)
    split = build_primary_split(dates, 0.60, 0.15, 0.25, embargo_periods=3)

    all_returned = split["train_dates"] + split["train_purged_dates"] + \
        split["validation_dates"] + split["validation_purged_dates"] + split["test_dates"]
    # every date accounted for exactly once, in original chronological order
    assert sorted(all_returned) == sorted(dates)
    assert len(all_returned) == len(dates)

    # no set overlaps between the reported "usable" windows
    assert not (set(split["train_dates"]) & set(split["validation_dates"]))
    assert not (set(split["validation_dates"]) & set(split["test_dates"]))
    assert not (set(split["train_dates"]) & set(split["test_dates"]))

    # everything in train_dates is chronologically before everything in validation_dates
    assert max(split["train_dates"] + split["train_purged_dates"]) < min(split["validation_dates"] + split["validation_purged_dates"])
    assert max(split["validation_dates"] + split["validation_purged_dates"]) < min(split["test_dates"])


# --- 3. Embargo/purge removes exactly embargo_periods dates from the tail of train and validation ---

def test_embargo_purges_exactly_the_configured_tail_length():
    dates = _dates(108)
    split = build_primary_split(dates, 0.60, 0.15, 0.25, embargo_periods=3)
    assert len(split["train_purged_dates"]) == 3
    assert len(split["validation_purged_dates"]) == 3
    # the purged dates are the latest dates in their raw window, not an arbitrary subset
    raw_train_end_idx = round(108 * 0.60)
    assert split["train_purged_dates"] == dates[raw_train_end_idx - 3:raw_train_end_idx]
    # test window is never purged since nothing follows it
    raw_validation_end_idx = round(108 * 0.75)
    assert split["test_dates"] == dates[raw_validation_end_idx:]


def test_zero_embargo_purges_nothing():
    dates = _dates(108)
    split = build_primary_split(dates, 0.60, 0.15, 0.25, embargo_periods=0)
    assert split["train_purged_dates"] == []
    assert split["validation_purged_dates"] == []


# --- 4. Embargo larger than a window raises rather than silently emptying it ---

def test_embargo_larger_than_window_raises_not_silently_empties():
    dates = _dates(108)  # validation window at 15% of 108 is ~16 dates
    with pytest.raises(ValueError):
        build_primary_split(dates, 0.60, 0.15, 0.25, embargo_periods=20)


# --- 5. Real-data-shaped sanity check using the actual 108-month PERMISSIVE period length ---

def test_matches_expected_illustrative_counts_from_the_spec():
    dates = _dates(108)  # the real PERMISSIVE period is exactly 108 rebalance dates
    split = build_primary_split(dates, 0.60, 0.15, 0.25, embargo_periods=3)
    # expected window sizes at these fractions of 108 months: ~65/16/27
    assert len(split["train_dates"]) + len(split["train_purged_dates"]) == pytest.approx(65, abs=1)
    assert len(split["validation_dates"]) + len(split["validation_purged_dates"]) == pytest.approx(16, abs=1)
    assert len(split["test_dates"]) == pytest.approx(27, abs=1)


# --- 6. Expanding retrain checkpoints: every checkpoint's training window ends before its own labels could resolve ---

def test_retrain_checkpoints_never_train_on_still_unresolved_labels():
    dates = _dates(108)
    horizon = 3
    checkpoints = generate_expanding_retrain_checkpoints(dates, period_start_idx=65, horizon_periods=horizon)

    assert checkpoints  # non-empty
    for cp in checkpoints:
        checkpoint_idx = dates.index(cp["checkpoint_date"])
        train_end_idx = dates.index(cp["train_window_end_date"])
        # the training window must end at least `horizon` periods before
        # the checkpoint, otherwise a label used in training wouldn't
        # have resolved yet at prediction time.
        assert checkpoint_idx - train_end_idx >= horizon or train_end_idx == 0

    # predict_dates across all checkpoints exactly covers the period from
    # period_start_idx to the end, with no gaps and no overlaps
    covered = [d for cp in checkpoints for d in cp["predict_dates"]]
    assert covered == dates[65:]


def test_retrain_checkpoints_step_no_more_often_than_the_horizon():
    dates = _dates(108)
    horizon = 3
    checkpoints = generate_expanding_retrain_checkpoints(dates, period_start_idx=65, horizon_periods=horizon)
    checkpoint_dates = [cp["checkpoint_date"] for cp in checkpoints]
    idxs = [dates.index(d) for d in checkpoint_dates]
    gaps = [b - a for a, b in zip(idxs, idxs[1:])]
    assert all(g >= horizon for g in gaps)
