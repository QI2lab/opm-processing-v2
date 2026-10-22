"""Exact photon likelihood, Gaussian readout and quantized camera probabilities."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import j1
from scipy.stats import norm, poisson

from tests.simulation.camera_likelihood import camera_log_likelihood
from tests.simulation.metrics import (
    airy_rayleigh,
    camera_score_difference,
    first_minimum,
    poisson_log_ratio_moments,
    predictive_deviance,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("sigma", [0, 0.7, 1.6])
@pytest.mark.parametrize("rate", [0, 0.2, 2, 100])
def test_adc_likelihood_normalizes_and_predicts_photon_statistics(
    sigma, rate, camera_calibration
):
    # A 10-bit camera retains negligible saturation here; the likelihood still
    # integrates both censored endpoint codes, rather than discarding them.
    """Integrate quantized camera probabilities, including censored ADC endpoints."""
    camera = replace(camera_calibration, read_noise_e=sigma, bit_depth=10)
    codes = np.arange(2**camera.bit_depth)
    probability = np.exp(camera_log_likelihood(codes, rate, camera))
    np.testing.assert_allclose(probability.sum(), 1, rtol=0, atol=2e-13)
    if sigma == 0:
        n = np.arange(300)
        adc = np.clip(np.rint(100 + n / 0.24), 0, 1023).astype(int)
        expected = np.bincount(adc, weights=poisson.pmf(n, rate), minlength=1024)
        np.testing.assert_allclose(probability, expected, rtol=1e-11, atol=2e-13)
    if rate == 0 and sigma > 0:
        expected = norm.cdf((100.5 - 100) * 0.24 / sigma) - norm.cdf(
            (99.5 - 100) * 0.24 / sigma
        )
        assert probability[100] == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("rate", [0.2, 2, 100], scope="module")
@pytest.mark.parametrize("rate_factor", [0.5, 2])
def test_held_out_camera_likelihood_favors_true_rate_without_brightness_fit(
    rate, rate_factor, camera_calibration, camera_likelihood_sample
):
    """Prefer the true photon rate over fixed alternative rates on independent data."""
    raw, truth_score = camera_likelihood_sample
    difference = truth_score - camera_log_likelihood(
        raw, rate * rate_factor, camera_calibration
    )
    # Five standard errors test the likelihood ordering on independent observations.
    assert difference.mean() > 5 * difference.std(ddof=1) / np.sqrt(raw.size)


@pytest.mark.parametrize("kernel_shape", [(1, 1, 1), (3, 3, 3), (4, 2, 3)])
def test_blurred_object_likelihood_uses_exact_reflected_optical_model(kernel_shape):
    """Match the analytic Poisson likelihood of twofold exposure on reflected edges."""
    from scipy import ndimage

    z, y, x = np.indices((7, 9, 11), dtype=float)
    density = 1 + z + 2 * y + 3 * x
    psf = np.arange(1, np.prod(kernel_shape) + 1, dtype=float).reshape(kernel_shape)
    psf /= psf.sum()
    blurred = ndimage.convolve(density, psf, mode="reflect")
    expected = 2 * (2 * np.log(2) - 1) * blurred.sum()
    assert predictive_deviance(
        density, 2 * blurred, psf, boundary="reflect"
    ) == pytest.approx(expected, rel=1e-13)


def test_integer_camera_score_ratio_matches_independent_adc_sum(camera_calibration):
    """Recover the Poisson score ratio for an ideal integer-count camera."""
    camera = replace(
        camera_calibration, read_noise_e=0, electrons_per_adu=1, offset_adu=0
    )
    observed = np.array([0, 1, 30, 3000, 1000], dtype=np.uint16)
    prediction = np.array([0, 0.1, 30, 3000, 0.1])
    baseline = np.array([1, 0.2, 25, 3100, 0.2])
    np.testing.assert_allclose(
        camera_log_likelihood(observed, prediction, camera),
        poisson.logpmf(observed, prediction),
        rtol=3e-13,
        atol=3e-13,
    )
    expected = camera_log_likelihood(
        observed, prediction, camera
    ) - camera_log_likelihood(observed, baseline, camera)
    np.testing.assert_allclose(
        camera_score_difference(observed, prediction, baseline, camera),
        expected,
        rtol=3e-11,
        atol=3e-11,
    )


def test_registration_likelihood_preserves_dark_support():
    """Retain exact deterministic dark support and impossible-event limits."""
    assert poisson_log_ratio_moments([0, 2], [0, 2]) == (0, 0)
    assert poisson_log_ratio_moments([0, 0], [0, 3]) == (3, 0)
    assert poisson_log_ratio_moments([0, 2], [3, 0]) == (np.inf, np.inf)


@pytest.mark.parametrize("exposure", [0.01, 1, 100])
def test_registration_likelihood_moments_predict_independent_photon_draws(exposure):
    """Match analytic log-likelihood moments to independent photon realizations."""
    truth = exposure * np.array([0.2, 2, 20])
    prediction = exposure * np.array([0.1, 3, 16])
    loss, variance = poisson_log_ratio_moments(truth, prediction)
    base_loss, base_variance = poisson_log_ratio_moments(
        truth / exposure, prediction / exposure
    )
    # KL and variance scale with exposure; distinguishability scales as sqrt(T).
    np.testing.assert_allclose(
        [loss, variance], exposure * np.array([base_loss, base_variance]), rtol=1e-13
    )
    counts = np.random.default_rng(31).poisson(truth, size=(50000, 3))
    measured = (poisson.logpmf(counts, truth) - poisson.logpmf(counts, prediction)).sum(
        axis=1
    )
    n = measured.size
    assert abs(measured.mean() - loss) <= 5 * np.sqrt(variance / n)
    # Fourth cumulant of a weighted Poisson sum is sum(lambda*weight^4).
    fourth = 3 * variance**2 + np.sum(truth * np.log(truth / prediction) ** 4)
    variance_se = np.sqrt((fourth - (n - 3) / (n - 1) * variance**2) / n)
    assert abs(measured.var(ddof=1) - variance) <= 5 * variance_se


def test_directional_rayleigh_measurement_matches_airy_first_minimum():
    """Recover the circular-pupil Rayleigh separation and pair contrast."""
    separation, dip = airy_rayleigh(0.637, 1.35)
    radius = np.linspace(0, 2 * separation, 20001)
    argument = 2 * np.pi * 1.35 * radius[1:] / 0.637
    profile = np.r_[1, (2 * j1(argument) / argument) ** 2]
    assert first_minimum(radius, profile) == pytest.approx(separation, abs=radius[1])
    assert separation == pytest.approx(0.6098349456 * 0.637 / 1.35, rel=1e-9)
    assert dip == pytest.approx(0.26497, abs=5e-5)
    with pytest.raises(ValueError, match="first minimum"):
        first_minimum(radius, np.exp(-(radius**2)))
