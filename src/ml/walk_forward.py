"""
Walk-forward split for the Phase 4 model. No random train/test split
anywhere -- this module only ever slices a chronologically sorted date
list.

Two things are built here:

1. build_primary_split() -- the single outer TRAIN / VALIDATION / [LOCKED]
   TEST split, at the fractions fixed in config.yaml's phase4.walk_forward
   block (60/15/25). Embargo/purge is applied at both the
   train/validation and validation/test boundaries: the last
   `embargo_periods` rebalance dates of each earlier window are dropped,
   because their forward-return labels extend into the following window.

2. generate_expanding_retrain_checkpoints() -- within the test (or
   validation) period, retraining checkpoints spaced no more often than
   the label horizon, each using an expanding training window that only
   ever includes dates whose labels have already fully resolved by that
   checkpoint, so there's no leakage across a checkpoint boundary either.
"""


def build_primary_split(rebalance_dates, train_fraction, validation_fraction, test_fraction,
                         embargo_periods):
    """rebalance_dates: chronologically sorted list of ISO date strings --
    Phase 3's own next_rebalance_dates() output, never re-derived here.

    Returns a dict:
        {
          "train_dates": [...],             # purged
          "train_purged_dates": [...],      # dropped from train's tail
          "validation_dates": [...],        # purged
          "validation_purged_dates": [...], # dropped from validation's tail
          "test_dates": [...],              # the locked test set -- never purged (nothing follows it)
        }

    Raises ValueError rather than silently producing an empty or
    misleading split if the fractions don't sum to ~1, or if
    embargo_periods would purge an entire window.
    """
    if abs((train_fraction + validation_fraction + test_fraction) - 1.0) > 1e-6:
        raise ValueError(
            f"train/validation/test fractions must sum to 1.0, got "
            f"{train_fraction}+{validation_fraction}+{test_fraction}={train_fraction+validation_fraction+test_fraction}"
        )
    if embargo_periods < 0:
        raise ValueError("embargo_periods must be >= 0")

    n = len(rebalance_dates)
    train_end_idx = round(n * train_fraction)
    validation_end_idx = round(n * (train_fraction + validation_fraction))

    raw_train = rebalance_dates[:train_end_idx]
    raw_validation = rebalance_dates[train_end_idx:validation_end_idx]
    test_dates = rebalance_dates[validation_end_idx:]

    if embargo_periods >= len(raw_train):
        raise ValueError(
            f"embargo_periods={embargo_periods} would purge the ENTIRE train window "
            f"({len(raw_train)} dates) -- reduce embargo_periods or increase train_fraction."
        )
    if embargo_periods >= len(raw_validation):
        raise ValueError(
            f"embargo_periods={embargo_periods} would purge the ENTIRE validation window "
            f"({len(raw_validation)} dates) -- reduce embargo_periods or increase validation_fraction."
        )
    if not test_dates:
        raise ValueError("test_fraction produced an empty locked test window -- check the fractions.")

    train_purged = raw_train[len(raw_train) - embargo_periods:] if embargo_periods else []
    train_dates = raw_train[:len(raw_train) - embargo_periods] if embargo_periods else raw_train

    validation_purged = raw_validation[len(raw_validation) - embargo_periods:] if embargo_periods else []
    validation_dates = raw_validation[:len(raw_validation) - embargo_periods] if embargo_periods else raw_validation

    return {
        "train_dates": train_dates,
        "train_purged_dates": train_purged,
        "validation_dates": validation_dates,
        "validation_purged_dates": validation_purged,
        "test_dates": test_dates,
    }


def generate_expanding_retrain_checkpoints(all_dates_up_to_and_including_period, period_start_idx,
                                            horizon_periods):
    """Within one window (validation or the locked test set), generates
    retraining checkpoints no more often than every `horizon_periods`
    rebalance dates, each with an expanding training window that stops
    `horizon_periods` before the checkpoint itself -- so every label used
    for that checkpoint's (re)training has already fully resolved by the
    time the checkpoint's predictions are made. Same embargo logic as
    build_primary_split, applied again at every internal retrain
    boundary, not just the two outer ones.

    all_dates_up_to_and_including_period: the full chronological date
        list (train + validation + test dates, in order) -- needed so an
        expanding window can look all the way back to the start.
    period_start_idx: index into that list where the window under
        evaluation (validation or test) begins.
    horizon_periods: the label horizon, in rebalance periods (matches
        config.yaml's phase4.target.primary_horizon_months for the
        primary horizon).

    Returns a list of dicts, one per checkpoint:
        {"checkpoint_date": str, "train_window_end_date": str,
         "predict_dates": [str, ...]}
    predict_dates covers this checkpoint's model until the next checkpoint
    (or the end of the period), so every rebalance date in the window is
    covered by exactly one checkpoint.
    """
    if horizon_periods <= 0:
        raise ValueError("horizon_periods must be positive")
    n = len(all_dates_up_to_and_including_period)
    if period_start_idx <= 0 or period_start_idx >= n:
        raise ValueError("period_start_idx must point inside all_dates_up_to_and_including_period, after some training history")

    checkpoints = []
    idx = period_start_idx
    while idx < n:
        # Train up to (but not including) the last horizon_periods dates
        # before this checkpoint, so no label used in training this
        # checkpoint's model could still be unresolved at the checkpoint date.
        train_window_end_idx = idx - horizon_periods
        if train_window_end_idx < 0:
            train_window_end_idx = 0
        checkpoint_date = all_dates_up_to_and_including_period[idx]
        next_idx = min(idx + horizon_periods, n)
        checkpoints.append({
            "checkpoint_date": checkpoint_date,
            "train_window_end_date": all_dates_up_to_and_including_period[train_window_end_idx],
            "predict_dates": all_dates_up_to_and_including_period[idx:next_idx],
        })
        idx = next_idx
    return checkpoints
