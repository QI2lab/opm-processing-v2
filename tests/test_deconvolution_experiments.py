"""Distribution and optical-kernel tests for deconvolution audit candidates."""

import math

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def experiment(cupy_gpu):
    """Load the experimental operators after requiring actual CUDA execution.

    Parameters
    ----------
    cupy_gpu
        Fixture proving CUDA execution before importing the audit kernels.

    Returns
    -------
    module
        Audit script containing the experimental count sampler and PSF window.
    """
    from scripts import audit_deconvolution

    return audit_deconvolution


@pytest.mark.unit
def test_lookup_cdf_matches_exact_binomial_probabilities(experiment, cupy_gpu):
    """Bound every CDF entry by independently summed integer binomial coefficients.

    Parameters
    ----------
    experiment
        Fixture providing the experimental operators.
    cupy_gpu
        Fixture providing CuPy on a working CUDA device.
    """
    actual = cupy_gpu.asnumpy(experiment.binomial_split_cdf(256))
    for n in range(257):
        cumulative = 0
        expected = []
        for k in range(n + 1):
            cumulative += math.comb(n, k)
            expected.append(cumulative / 2**n)
        # A nonzero absolute tolerance would conceal lost low-probability tails.
        np.testing.assert_allclose(actual[n, : n + 1], expected, rtol=2e-12, atol=0)
        np.testing.assert_array_equal(actual[n, n:], 1)
    assert np.all(np.diff(actual, axis=1) >= 0)


@pytest.mark.unit
@pytest.mark.parametrize("fractional", (False, True))
@pytest.mark.parametrize("assign_fractional_remainder", (False, True))
def test_lookup_splits_preserve_distribution_and_fallback(
    experiment, cupy_gpu, fractional, assign_fractional_remainder
):
    """Check counts, small-count probabilities, unbiased moments, and independence.

    Parameters
    ----------
    experiment
        Fixture providing the experimental operators.
    cupy_gpu
        Fixture providing CuPy on a working CUDA device.
    fractional : bool
        Add a known fractional count to exercise weighted residual assignment.
    assign_fractional_remainder : bool
        Reference residual placement or the undersampled weighted-count model.
    """
    cp = cupy_gpu
    levels = np.asarray((0, 1, 4, 15, 64, 128, 256, 257, 2048), np.float32)
    if fractional:
        levels += np.float32(0.24)
    observed = cp.asarray(np.broadcast_to(levels[:, None], (len(levels), 200_000)))
    table = experiment.binomial_split_cdf()
    first = experiment.split_counts_lookup(
        observed,
        cp.random.default_rng(19),
        table,
        assign_fractional_remainder=assign_fractional_remainder,
    )
    repeated = experiment.split_counts_lookup(
        observed,
        cp.random.default_rng(19),
        table,
        assign_fractional_remainder=assign_fractional_remainder,
    )
    np.testing.assert_array_equal(cp.asnumpy(first), cp.asnumpy(repeated))
    second = observed - first
    np.testing.assert_array_equal(cp.asnumpy(first + second), cp.asnumpy(observed))
    assert bool(cp.all(first >= 0)) and bool(cp.all(second >= 0))
    expected_variance = (
        np.floor(levels) + ((levels % 1) ** 2 if assign_fractional_remainder else 0)
    ) / 4
    means = (
        (levels / 2, levels / 2)
        if assign_fractional_remainder
        else (np.floor(levels) / 2, levels - np.floor(levels) / 2)
    )
    for half, mean in zip((first, second), means):
        # Six standard errors, plus float32 rounding at the largest count.
        mean_error = np.abs(cp.asnumpy(half.mean(axis=1)) - mean)
        assert np.all(mean_error <= 6 * np.sqrt(expected_variance / 200_000) + 2e-5)
        np.testing.assert_allclose(
            cp.asnumpy(half.var(axis=1)), expected_variance, rtol=0.02, atol=0.002
        )
    integer_split = np.rint(cp.asnumpy(first[2])).astype(int)
    np.testing.assert_allclose(
        np.bincount(integer_split, minlength=5) / integer_split.size,
        np.asarray((1, 4, 6, 4, 1)) / 16,
        atol=0.004,
    )
    pixels = cp.asnumpy(first[3])
    assert abs(np.corrcoef(pixels[:-1], pixels[1:])[0, 1]) < 0.01
    rng = cp.random.default_rng(31)
    a = cp.asnumpy(experiment.split_counts_lookup(observed[3], rng, table))
    b = cp.asnumpy(experiment.split_counts_lookup(observed[3], rng, table))
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.01


@pytest.mark.unit
def test_lookup_poisson_thinning_has_independent_halves(experiment, cupy_gpu):
    """A Poisson count split must retain two independent half-rate Poisson processes.

    Parameters
    ----------
    experiment
        Fixture providing the experimental operators.
    cupy_gpu
        Fixture providing CuPy on a working CUDA device.
    """
    cp = cupy_gpu
    observed = cp.asarray(
        np.random.default_rng(73).poisson(20, 400_000).astype(np.float32)
    )
    first = cp.asnumpy(
        experiment.split_counts_lookup(
            observed, cp.random.default_rng(73), experiment.binomial_split_cdf(64)
        )
    )
    second = cp.asnumpy(observed) - first
    for half in (first, second):
        assert abs(half.mean() - 10) < 0.03
        assert abs(half.var() - 10) < 0.12
    assert abs(np.corrcoef(first, second)[0, 1]) < 0.01


@pytest.mark.unit
@pytest.mark.parametrize("shape", ((1, 7, 9), (5, 7, 9)))
def test_taper_retains_center_origin_and_normalized_optical_gain(experiment, shape):
    """Check a known separable edge window, energy accounting, and unit DC response.

    Parameters
    ----------
    experiment
        Fixture providing the experimental operators.
    shape : tuple of int
        Planar or volumetric acquisition-grid PSF dimensions.
    """
    psf = np.ones(shape, np.float32)
    original = psf.copy()
    sampling = (0.2, 0.115, 0.115)
    widths = (0.2, 0.23, 0.115)
    tapered, removed = experiment.taper_psf(psf, sampling, widths)
    expected = np.ones(shape, np.float64)
    for axis, (size, pitch, width) in enumerate(
        zip(shape, sampling, widths, strict=True)
    ):
        if size == 1:
            continue
        profile = np.ones(size)
        profile[0] = profile[-1] = 0
        if width == 2 * pitch:
            profile[1] = profile[-2] = 0.5
        window_shape = [1, 1, 1]
        window_shape[axis] = size
        expected *= profile.reshape(window_shape)
    assert removed == pytest.approx(1 - expected.sum() / psf.sum())
    np.testing.assert_allclose(tapered, expected / expected.sum(), rtol=1e-6)
    np.testing.assert_array_equal(psf, original)
    np.testing.assert_allclose(np.fft.fftn(tapered)[0, 0, 0], 1, atol=1e-7)
    for axis in range(3):
        coordinates = np.arange(shape[axis]) - (shape[axis] - 1) / 2
        marginal = tapered.sum(axis=tuple(i for i in range(3) if i != axis))
        assert abs(np.sum(coordinates * marginal)) < 1e-7
