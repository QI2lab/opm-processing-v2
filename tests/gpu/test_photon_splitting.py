"""Photon-count distribution tests for production splitting."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("fractional", [False, True])
def test_matlab_split_matches_independent_distribution_moments(
    photon_splitter, cupy_gpu, fractional
):
    # Integer photons and calibrated fractional weights across the count range.
    """Match the reference split mean, variance and fractional-count behavior."""
    levels = np.array([0, 1, 4, 15, 64, 128, 149, 150, 256, 257, 2048], np.float32)
    if fractional:
        levels = np.r_[
            levels + np.float32(0.24),
            np.array(
                [0.49, 0.5, 0.51, 0.96, 1.44, 2.88, 12.24, 149.49, 149.5], np.float32
            ),
            [np.nextafter(np.float32(0.5), np.float32(0))],
        ]
    count = 200000
    observed = np.broadcast_to(levels[:, None], (levels.size, count))
    first = photon_splitter.split(observed)
    np.testing.assert_array_equal(first, photon_splitter.split(observed))
    if not fractional:
        reference = cupy_gpu.random.default_rng(19).binomial(
            cupy_gpu.asarray(np.where(observed >= 150, 1, observed)).astype(
                cupy_gpu.int64
            ),
            0.5,
        )
        np.testing.assert_array_equal(
            first[levels < 150], cupy_gpu.asnumpy(reference)[levels < 150]
        )
    second = observed - first
    np.testing.assert_allclose(
        first + second, observed, rtol=2 * np.finfo(np.float32).eps, atol=0
    )
    whole = np.floor(levels.astype(np.float64) + 0.5)
    residual = levels - whole
    # Binomial n/4 below 150; Gaussian n/4 above it; uniform remainder r²/12.
    variance = whole / 4 + residual**2 / 12
    fourth = 3 * variance**2 - np.where(whole < 150, whole / 8, 0) - residual**4 / 120
    variance_se = np.sqrt(
        np.maximum(fourth - (count - 3) / (count - 1) * variance**2, 0) / count
    )
    for half in (first, second):
        # Six standard errors of the independently specified count distribution,
        # rather than a fixed percentage or empirical probability tolerance.
        np.testing.assert_array_less(
            np.abs(half.astype(np.float64).mean(axis=1) - levels / 2),
            6 * np.sqrt(variance / count) + 1e-12,
        )
        np.testing.assert_array_less(
            np.abs(half.astype(np.float64).var(axis=1, ddof=1) - variance),
            6 * variance_se + 1e-12,
        )
    probability = np.array([1, 4, 6, 4, 1]) / 16
    measured = np.bincount(np.floor(first[2]).astype(int), minlength=5) / count
    np.testing.assert_array_less(
        np.abs(measured - probability),
        6 * np.sqrt(probability * (1 - probability) / count),
    )


def test_poisson_thinning_produces_independent_half_rate_processes(photon_splitter):
    """Recover independent half-rate Poisson samples through binomial thinning."""
    count = 400000
    observed = np.random.default_rng(73).poisson(20, count).astype(np.float32)
    first = photon_splitter.split(observed, seed=131).astype(np.float64)
    second = observed - first
    for half in [first, second]:
        assert abs(half.mean() - 10) < 6 * np.sqrt(10 / count)
        # Poisson fourth central moment is lambda + 3 lambda^2.
        variance_se = np.sqrt((10 + 300 - (count - 3) / (count - 1) * 100) / count)
        assert abs(half.var(ddof=1) - 10) < 6 * variance_se
    assert abs(np.corrcoef(first, second)[0, 1]) < 6 / np.sqrt(count)
