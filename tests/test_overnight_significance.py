"""
Tests for src/ml/overnight_significance.py -- the Tier 0 block-bootstrap
test (PHASE5_OVERNIGHT_GAP_SPECIFICATION.md section 6/8). These are
statistical-correctness tests on synthetic series with known properties,
not leakage tests (this module never touches the database) -- the
leakage-relevant work is in overnight_targets.py and its own test suite.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import random
import pytest

from src.ml import overnight_significance as SIG


def test_white_noise_around_zero_does_not_exclude_zero():
    """A series with a genuine zero mean should, in the large majority of
    draws, produce a CI that contains zero -- the test uses a fixed seed
    so this is deterministic, not flaky."""
    rng = random.Random(123)
    series = [rng.gauss(0, 1) for _ in range(500)]
    result = SIG.block_bootstrap_mean_ci(series, block_length=5, seed=1)
    assert result is not None
    assert result["ci_low"] < 0 < result["ci_high"]
    assert result["ci_excludes_zero"] is False


def test_series_with_a_real_nonzero_mean_excludes_zero():
    rng = random.Random(456)
    series = [rng.gauss(0.02, 0.1) for _ in range(500)]  # clear positive mean, small noise
    result = SIG.block_bootstrap_mean_ci(series, block_length=5, seed=1)
    assert result is not None
    assert result["ci_excludes_zero"] is True
    assert result["ci_low"] > 0


def test_paired_difference_uses_the_same_day_pairing_not_independent_resampling():
    """If overnight and intraday are IDENTICAL series (paired difference
    is exactly zero every day, zero variance), the CI must be a point mass
    at zero and must NOT exclude zero -- this would fail if the two series
    were resampled independently rather than as a paired difference,
    since independent resampling of two identical-but-separately-drawn
    series could spuriously show a nonzero difference."""
    rng = random.Random(789)
    series = [rng.gauss(0, 1) for _ in range(200)]
    result = SIG.block_bootstrap_paired_difference_ci(series, list(series), block_length=5, seed=1)
    assert result is not None
    assert result["point_estimate"] == pytest.approx(0.0)
    assert result["ci_excludes_zero"] is False


def test_too_short_a_series_returns_none_not_a_fabricated_ci():
    result = SIG.block_bootstrap_mean_ci([0.01, 0.02, -0.01], block_length=5, seed=1)
    assert result is None


def test_tier0_classification_failure_when_ci_contains_zero():
    rng = random.Random(11)
    overnight = [rng.gauss(0, 1) for _ in range(400)]
    intraday = [rng.gauss(0, 1) for _ in range(400)]
    result = SIG.tier0_test(overnight, intraday, n_boot=500)
    assert result["classification"] == "failure"


def test_tier0_classification_genuine_effect_candidate_when_clearly_nonzero_and_distinct():
    rng = random.Random(22)
    # A large, unambiguous overnight effect, clearly different from a
    # near-zero intraday series -- constructed to clear both the CI and
    # the 1.0pp annualised-magnitude threshold with room to spare.
    overnight = [rng.gauss(0.002, 0.001) for _ in range(400)]  # ~50pp/yr annualised
    intraday = [rng.gauss(0.0, 0.001) for _ in range(400)]
    result = SIG.tier0_test(overnight, intraday, n_boot=500)
    assert result["classification"] == "genuine_effect_candidate"
    assert abs(result["annualised_overnight_pct"]) > 1.0


def test_tier0_rejects_mismatched_length_inputs():
    with pytest.raises(ValueError):
        SIG.tier0_test([0.01, 0.02], [0.01], n_boot=100)


def test_tier0_handles_empty_input_without_crashing():
    result = SIG.tier0_test([], [], n_boot=100)
    assert result["classification"] == "inconclusive"
    assert result["n_days"] == 0
