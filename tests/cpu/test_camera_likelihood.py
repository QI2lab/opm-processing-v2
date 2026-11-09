"""Exact photon likelihood, Gaussian readout and quantized camera probabilities."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import norm, poisson

from tests.simulation.camera_likelihood import camera_log_likelihood
from tests.simulation.metrics import poisson_log_ratio_moments

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


@pytest.mark.parametrize("rate", [0.2, 2, 100])
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


@pytest.mark.parametrize(
    "artifact",
    [
        "correct",
        "correct-zero-bound",
        "gain",
        "dim",
        "stripes",
        "zero",
    ],
)
def test_camera_sampling_bounds_detect_gain_and_spatial_errors(
    camera_sampling_case, artifact
):
    """Known detector noise cannot conceal a twenty-percent or mean-preserving stripe error."""
    from opm_processing.imageprocessing.camera import camera_correct
    from tests.simulation.camera_likelihood import (
        SamplingEnvelope,
        camera_sampling_envelope,
    )

    case = camera_sampling_case
    expected, camera = case.expectation, case.camera
    envelope = camera_sampling_envelope(
        expected, case.raw, camera, radii=case.radii, footprint=case.footprint
    )
    # Identity sampling is an allowed positive interpolation: its central
    # coefficient is one. Its noise is the actual physical camera draw.
    prediction = camera_correct(
        case.raw, camera.offset_adu, camera.electrons_per_adu
    ).astype(np.float64)
    original_flux = prediction.sum(dtype=np.float64)
    if artifact == "correct-zero-bound":
        envelope = SamplingEnvelope(np.zeros_like(expected), envelope.upper)
    if artifact == "gain":
        prediction *= 1.2
    elif artifact == "dim":
        prediction *= 0.8
    elif artifact == "stripes":
        prediction += np.where(np.arange(prediction.shape[-1]) % 2, 1, -1) * (
            0.2 * case.nominal
        )
        assert prediction.sum(dtype=np.float64) == original_flux
    elif artifact == "zero":
        prediction.fill(0)
    correct = artifact in ("correct", "correct-zero-bound")
    inside = np.all(
        np.isfinite(prediction)
        & (prediction >= envelope.lower)
        & (prediction <= envelope.upper)
    )
    assert inside == correct
