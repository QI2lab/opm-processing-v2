"""Photon-count distribution tests for production splitting."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("fractional", [False, True])
@pytest.mark.parametrize("array_backend", ["gpu", "host"])
def test_count_split_matches_prescribed_distribution_moments(
    cupy_gpu, array_backend, fractional
):
    """Check integer thinning and symmetric fractional-count splitting."""
    from opm_processing.imageprocessing.rlgc import _split_observed_counts

    xp = cupy_gpu if array_backend == "gpu" else np
    to_numpy = cupy_gpu.asnumpy if array_backend == "gpu" else np.asarray
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
    counts = xp.asarray(observed, dtype=xp.float32)
    first = to_numpy(_split_observed_counts(counts, xp.random.default_rng(19)))
    if not fractional:
        np.testing.assert_array_equal(first, np.floor(first))
    second = observed - first
    assert first.min() >= 0 and second.min() >= 0
    np.testing.assert_allclose(
        first + second, observed, rtol=2 * np.finfo(np.float32).eps, atol=0
    )
    whole = np.floor(levels.astype(np.float64))
    residual = levels - whole
    variance = (whole + residual**2) / 4
    fourth = 3 * variance**2 - (whole + residual**4) / 8
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


@pytest.mark.parametrize("array_backend", ["gpu", "host"])
def test_poisson_thinning_produces_independent_half_rate_processes(
    cupy_gpu, ideal_photon_camera, array_backend
):
    """Recover independent half-rate Poisson samples through binomial thinning."""
    from opm_processing.imageprocessing.rlgc import _split_observed_counts
    from tests.simulation.camera import digitize

    xp = cupy_gpu if array_backend == "gpu" else np
    to_numpy = cupy_gpu.asnumpy if array_backend == "gpu" else np.asarray
    count = 400000
    observed = digitize(np.full(count, 20.0), ideal_photon_camera, seed=73).astype(
        np.float32
    )
    first = to_numpy(
        _split_observed_counts(xp.asarray(observed), xp.random.default_rng(131))
    ).astype(np.float64)
    second = observed - first
    for half in [first, second]:
        assert abs(half.mean() - 10) < 6 * np.sqrt(10 / count)
        # Poisson fourth central moment is lambda + 3 lambda^2.
        variance_se = np.sqrt((10 + 300 - (count - 3) / (count - 1) * 100) / count)
        assert abs(half.var(ddof=1) - 10) < 6 * variance_se
    assert abs(np.corrcoef(first, second)[0, 1]) < 6 / np.sqrt(count)
