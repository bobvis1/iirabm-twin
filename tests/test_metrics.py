import numpy as np
import pytest

from twin import evaluate
from twin.metrics import (
    auroc,
    bootstrap_ci,
    brier,
    interval_coverage,
    log_loss,
    paired_bootstrap_diff,
    rank_uniformity,
    reliability,
    wilson,
)


def test_scores_on_known_values():
    p, y = np.array([0.9, 0.2, 0.7, 0.1]), np.array([1, 0, 1, 0])
    assert brier(p, y) == pytest.approx((0.01 + 0.04 + 0.09 + 0.01) / 4)
    assert log_loss(p, y) == pytest.approx(-np.mean(np.log([0.9, 0.8, 0.7, 0.9])))
    assert auroc(p, y) == 1.0
    assert auroc(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5  # tie counts one half
    assert auroc(np.array([0.1, 0.9]), np.array([1, 0])) == 0.0
    assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([1, 1])))
    assert log_loss(np.array([0.0]), np.array([1])) < 10  # clipped, finite


def test_bootstrap_resamples_patients_not_values():
    """Resampling must keep each patient's row together: with y = p the Brier score is 0 in every
    resample, but pairing broken would give a nonzero one."""
    rng = np.random.default_rng(0)
    p = rng.random(50)
    est, lo, hi = bootstrap_ci(brier, p, p.copy(), n_boot=200)
    assert est == lo == hi == 0.0
    seen = []

    def stat(idx_vals, other):
        seen.append(len(idx_vals))
        return float(np.mean(idx_vals))

    bootstrap_ci(stat, np.arange(30.0), np.arange(30.0), n_boot=20)
    assert set(seen[1:]) == {30}  # every resample has as many patients as the original

    # CI width shrinks with more patients
    def width(n):
        _, lo, hi = bootstrap_ci(lambda v: v.mean(), rng.normal(size=n), n_boot=300)
        return hi - lo

    assert width(400) < width(25)


def test_paired_bootstrap_difference_of_a_better_method_excludes_zero():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200).astype(float)
    good = np.clip(y * 0.8 + 0.1 + rng.normal(0, 0.05, 200), 0, 1)
    bad = np.full(200, y.mean())
    est, lo, hi = paired_bootstrap_diff(brier, (bad, y), (good, y))
    assert est > 0 and lo > 0


def test_reliability_bins_partition_the_patients():
    rng = np.random.default_rng(2)
    p = rng.random(101)
    y = (rng.random(101) < p).astype(float)
    bins = reliability(p, y, bins=5)
    assert sum(b["n"] for b in bins) == 101 and len(bins) == 5
    assert [b["mean_predicted"] for b in bins] == sorted(b["mean_predicted"] for b in bins)


def test_wilson_interval():
    lo, hi = wilson(80, 100)
    assert 0.70 < lo < 0.80 < hi < 0.88
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # NaN


def test_coverage_and_ranks_on_a_synthetic_case_with_a_known_answer():
    """Normal-normal model: theta ~ N(0,1), data y ~ N(theta, s^2). The exact posterior has
    nominal coverage and uniform ranks; an over-confident one (sd halved) does not."""
    rng = np.random.default_rng(3)
    n, s = 4000, 1.0
    theta = rng.normal(size=n)
    y = theta + rng.normal(scale=s, size=n)
    post_var = 1 / (1 + 1 / s**2)
    mean = y * post_var / s**2
    sd = np.sqrt(post_var)
    z80 = 1.2815515655446004
    exact_cov = interval_coverage(mean - z80 * sd, mean + z80 * sd, theta)
    over_cov = interval_coverage(mean - z80 * sd / 2, mean + z80 * sd / 2, theta)
    assert abs(exact_cov - 0.8) < 0.02
    assert over_cov < 0.55
    from scipy.stats import norm

    ranks_exact = norm.cdf((theta - mean) / sd)
    ranks_over = norm.cdf((theta - mean) / (sd / 2))
    assert rank_uniformity(ranks_exact)["p_value"] > 0.01
    assert rank_uniformity(ranks_over)["p_value"] < 1e-6
    assert rank_uniformity(ranks_over)["counts"][0] > 2 * rank_uniformity(ranks_exact)["counts"][0]


def test_tuning_refuses_test_patients_and_evaluation_refuses_development_patients_by_default():
    evaluate.assert_development_only(["dev-001", "dev-040"])
    with pytest.raises(ValueError, match="development patients"):
        evaluate.assert_development_only(["dev-001", "test-001"])
    with pytest.raises(ValueError, match="development patients"):
        evaluate.evaluate_cohort(["dev-001"], {}, None, None)  # not allowed unless asked
