"""Exact photon likelihood, Gaussian readout and quantized camera probabilities."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import norm, poisson

from tests.simulation.camera_likelihood import camera_log_likelihood

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


def test_integer_camera_likelihood_matches_poisson_probability(camera_calibration):
    """Recover exact Poisson probabilities for an ideal integer-count camera."""
    camera = replace(
        camera_calibration, read_noise_e=0, electrons_per_adu=1, offset_adu=0
    )
    observed = np.array([0, 1, 30, 3000, 1000], dtype=np.uint16)
    prediction = np.array([0, 0.1, 30, 3000, 0.1])
    np.testing.assert_allclose(
        camera_log_likelihood(observed, prediction, camera),
        poisson.logpmf(observed, prediction),
        rtol=3e-13,
        atol=3e-13,
    )
