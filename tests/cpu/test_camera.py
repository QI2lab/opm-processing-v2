"""Known camera transfer functions and physical optical correction checks."""

import numpy as np
import pytest

from opm_processing.imageprocessing.camera import (
    camera_correct,
    illumination_correct,
)


@pytest.mark.unit
@pytest.mark.parametrize("conversion", [0.24, 0.24 / 0.9])
def test_camera_transfer_dark_saturation_and_rounding(camera_codes, conversion):
    """Cover dark subtraction, saturation and rounding of every uint16 code.

    Parameters
    ----------
    conversion : float
        Camera conversion per ADU, with or without quantum-efficiency scaling.
    """
    raw = camera_codes
    expected = np.maximum(
        (raw.astype(np.float32) - np.float32(100.25)) * np.float32(conversion),
        np.float32(0),
    )
    actual = camera_correct(raw, 100.25, conversion)
    np.testing.assert_array_equal(actual, expected)
    assert np.all(actual.ravel()[:101] == 0)
    assert actual.ravel()[-1] > 0
    assert not np.shares_memory(actual, raw)


@pytest.mark.unit
@pytest.mark.parametrize("stage_gain", (False, True))
@pytest.mark.parametrize("strided", (False, True))
def test_optical_illumination_and_camera_transfer(
    camera_optical_sample, stage_gain, strided
):
    """Restore a diffraction emitter with known illumination and detector gain.

    Parameters
    ----------
    stage_gain : bool
        Include the measured detector response in acquisition and correction.
    strided : bool
        Process a noncontiguous scan and detector-Y crop.
    """
    case = camera_optical_sample
    original = case.raw.copy()
    calibrated = camera_correct(case.raw, case.offset, case.conversion, **case.options)
    reference = calibrated.copy()
    corrected = illumination_correct(calibrated, case.illumination)
    np.testing.assert_array_equal(corrected, reference / case.illumination)
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


@pytest.mark.unit
def test_uniform_illumination_preserves_calibrated_intensities():
    """Uniform unit illumination preserves the calibrated photon values."""
    photons = np.linspace(0, 1000, 120, dtype=np.float32).reshape(3, 5, 8)
    np.testing.assert_array_equal(
        illumination_correct(photons, np.ones((5, 8), np.float32)), photons
    )


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(5, 8), (2, 3, 5, 8)])
def test_camera_images_and_broadcast_illumination(shape):
    """Broadcast known ADC transfer and illumination over leading image axes.

    Parameters
    ----------
    shape : tuple of int
        Image dimensions, including optional leading acquisition axes.
    """
    raw = np.full(shape, 110, np.uint16)
    actual = camera_correct(raw, 100, 0.25)
    np.testing.assert_array_equal(actual, np.full(shape, 2.5, np.float32))
    corrected = illumination_correct(actual, np.full((1, 8), 0.5, np.float32))
    np.testing.assert_array_equal(corrected, np.full(shape, 5, np.float32))
    np.testing.assert_array_equal(raw, 110)
    np.testing.assert_array_equal(actual, 2.5)
