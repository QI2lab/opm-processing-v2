"""Known camera transfer functions and physical optical correction checks."""

import numpy as np
import pytest

from opm_processing.imageprocessing.camera import (
    camera_correct,
    illumination_correct,
    qi2lab_stage_scan_camera_gain,
)
from tests.physics_point_sources import sample_point_source


@pytest.mark.unit
@pytest.mark.parametrize("conversion", [0.24, 0.24 / 0.9])
def test_camera_transfer_dark_saturation_and_rounding(conversion):
    """Cover dark subtraction, saturation and rounding of every uint16 code.

    Parameters
    ----------
    conversion : float
        Camera conversion per ADU, with or without quantum-efficiency scaling.
    """
    raw = np.arange(65536, dtype=np.uint16).reshape(16, 16, 256)
    expected = np.maximum(
        (raw.astype(np.float32) - np.float32(100.25)) * np.float32(conversion),
        np.float32(0),
    )
    actual = camera_correct(raw, 100.25, conversion)
    np.testing.assert_array_equal(actual, expected)
    assert np.all(actual.ravel()[:101] == 0)
    assert actual.ravel()[-1] > 0
    assert not np.shares_memory(actual, raw)


@pytest.mark.integration
@pytest.mark.parametrize("stage_gain,strided", [(False, False), (True, True)])
def test_optical_illumination_and_camera_transfer(stage_gain, strided):
    """Restore a diffraction emitter with known illumination and detector gain.

    Parameters
    ----------
    stage_gain : bool
        Include the measured detector response in acquisition and correction.
    strided : bool
        Process a noncontiguous scan and detector-Y crop.
    """
    photons = sample_point_source(shape=(31, 65, 73), scan_step=0.4)
    photons *= np.float32(1000 / photons.max())
    photons += np.float32(20)
    y = np.linspace(-1, 1, 65, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, 73, dtype=np.float32)[None, :]
    illumination = np.asarray(0.5 + 0.5 * np.exp(-x * x - y * y), dtype=np.float32)
    gain = (
        qi2lab_stage_scan_camera_gain()[1040:1113]
        if stage_gain
        else np.ones(73, np.float32)
    )
    measured = np.random.default_rng(47).poisson(photons * illumination * gain)
    offset, conversion = 100.0, 0.24 / 0.9
    raw = np.rint(measured / conversion + offset).astype(np.uint16)
    if strided:
        raw, illumination = raw[::2, ::2, :], illumination[::2, :]
        measured = measured[::2, ::2, :]
    original = raw.copy()
    options = dict(detector_x_offset=1040, apply_stage_scan_gain=stage_gain)
    calibrated = camera_correct(raw, offset, conversion, **options)
    reference = calibrated.copy()
    corrected = illumination_correct(calibrated, illumination)
    np.testing.assert_array_equal(corrected, reference / illumination)
    # Independent physical expectation: remove illumination/detector response from
    # observed photons; allow only the camera's half-ADU quantization uncertainty.
    physical = measured / (illumination.astype(np.float64) * gain.astype(np.float64))
    tolerance = (
        conversion * 0.5 / (illumination.astype(np.float64) * gain.astype(np.float64))
        + 2e-4
    )
    assert np.all(np.abs(corrected - physical) <= tolerance)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(calibrated, reference)
    assert not np.shares_memory(corrected, calibrated)


@pytest.mark.unit
def test_uniform_illumination_and_uncorrected_input_contracts():
    """Uniform unit illumination preserves photons; raw counts require calibration."""
    photons = np.linspace(0, 1000, 120, dtype=np.float32).reshape(3, 5, 8)
    np.testing.assert_array_equal(
        illumination_correct(photons, np.ones((5, 8), np.float32)), photons
    )
    with pytest.raises(TypeError):
        camera_correct(photons, 100, 0.24)
    with pytest.raises(TypeError):
        illumination_correct(photons.astype(np.uint16), np.ones((5, 8), np.float32))


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
