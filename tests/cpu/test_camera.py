"""Known camera transfer functions and physical optical correction checks."""

import numpy as np
import pytest
from scipy.stats import norm

from opm_processing.imageprocessing.camera import (
    camera_correct,
    illumination_correct,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("offset", [100, 1000])
@pytest.mark.parametrize("qe", [1.0, 0.9])
def test_camera_calibration_transforms_read_noise_units(camera_codes, offset, qe):
    """Gain scales noise RMS; subtracting offset does not, including ADC rounding.

    Isolate Gaussian read noise around 96 fixed photoelectrons so clipping and
    shot noise cannot conceal a units error. Integrate exact normal ADC-bin
    probabilities rather than approximate quantization noise as uniform.
    """
    electrons_per_adu, sigma_e, signal_e = 0.24, 1.6, 96.0
    sigma_adu = sigma_e / electrons_per_adu
    relative_code = (
        camera_codes.astype(np.float64) - offset - signal_e / electrons_per_adu
    )
    probability = norm.cdf((relative_code + 0.5) / sigma_adu) - norm.cdf(
        (relative_code - 0.5) / sigma_adu
    )
    np.testing.assert_allclose(probability.sum(), 1, atol=2e-15, rtol=0)
    observed = camera_correct(camera_codes, offset, electrons_per_adu / qe).astype(
        np.float64
    )
    mean = np.sum(observed * probability)
    variance = np.sum((observed - mean) ** 2 * probability)
    adc_variance = np.sum(relative_code**2 * probability)
    np.testing.assert_allclose(mean, signal_e / qe, rtol=2 * np.finfo(np.float32).eps)
    np.testing.assert_allclose(
        variance,
        adc_variance * (electrons_per_adu / qe) ** 2,
        rtol=32 * np.finfo(np.float32).eps,
    )


@pytest.mark.parametrize(
    ("conversion", "anchor_counts"),
    [
        (0.24, (0, 0, 0, 0.18, 0.42, 0.66, 95.94, 96.18, 215.94, 15704.34)),
        (
            0.24 / 0.9,
            (
                0,
                0,
                0,
                1 / 5,
                7 / 15,
                11 / 15,
                533 / 5,
                1603 / 15,
                3599 / 15,
                261739 / 15,
            ),
        ),
    ],
)
def test_camera_transfer_dark_saturation_and_rounding(
    camera_codes, conversion, anchor_counts
):
    """Cover dark subtraction, saturation and rounding of every uint16 code."""
    raw = camera_codes
    actual = camera_correct(raw, 100.25, conversion)
    # Prescribed photon counts provide independent transfer anchors. Checking
    # every remaining ADC increment needs no copy of the calibration function.
    anchors = (0, 99, 100, 101, 102, 103, 500, 501, 1000, 65535)
    np.testing.assert_allclose(
        actual.ravel()[list(anchors)],
        anchor_counts,
        rtol=2 * np.finfo(np.float32).eps,
        atol=0,
    )
    # Two float32 operations bound each sample's error by gamma_2 times the
    # full ADC-range signal; adjacent errors add in the first difference.
    error = 2 * np.finfo(np.float32).eps
    roundoff = error / (1 - error) * 65536 * conversion
    np.testing.assert_allclose(
        np.diff(actual.ravel()[101:].astype(np.float64)),
        conversion,
        rtol=0,
        atol=2 * roundoff,
    )
    assert np.all(actual.ravel()[:101] == 0)
    assert actual.ravel()[-1] > 0
    assert not np.shares_memory(actual, raw)


@pytest.mark.parametrize("stage_gain", (False, True))
@pytest.mark.parametrize("strided", (False, True))
def test_detector_response_and_camera_transfer(
    detector_response_sample, stage_gain, strided
):
    """Invert known detector response and gain within the ADC rounding interval."""
    case = detector_response_sample
    original = case.raw.copy()
    calibrated = camera_correct(case.raw, case.offset, case.conversion, **case.options)
    reference = calibrated.copy()
    corrected = illumination_correct(calibrated, case.illumination)
    # Independent physical expectation: remove illumination/detector response from
    # observed photons; allow only the camera's half-ADU quantization uncertainty.
    physical = case.measured / (
        case.illumination.astype(np.float64) * case.gain.astype(np.float64)
    )
    tolerance = (
        case.conversion
        * 0.5
        / (case.illumination.astype(np.float64) * case.gain.astype(np.float64))
        + 2e-4
    )
    assert np.all(np.abs(corrected - physical) <= tolerance)
    np.testing.assert_array_equal(case.raw, original)
    np.testing.assert_array_equal(calibrated, reference)
    assert not np.shares_memory(corrected, calibrated)


def test_uniform_illumination_preserves_calibrated_intensities():
    """Uniform unit illumination preserves the calibrated photon values."""
    photons = np.linspace(0, 1000, 120, dtype=np.float32).reshape(3, 5, 8)
    np.testing.assert_array_equal(
        illumination_correct(photons, np.ones((5, 8), np.float32)), photons
    )


@pytest.mark.parametrize("shape", [(5, 8), (2, 3, 5, 8)])
def test_camera_images_and_broadcast_illumination(shape):
    """Broadcast known ADC transfer and illumination over leading image axes."""
    raw = np.full(shape, 110, np.uint16)
    actual = camera_correct(raw, 100, 0.25)
    np.testing.assert_array_equal(actual, np.full(shape, 2.5, np.float32))
    corrected = illumination_correct(actual, np.full((1, 8), 0.5, np.float32))
    np.testing.assert_array_equal(corrected, np.full(shape, 5, np.float32))
    np.testing.assert_array_equal(raw, 110)
    np.testing.assert_array_equal(actual, 2.5)
