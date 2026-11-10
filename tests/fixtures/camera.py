"""Calibrated camera codes and post-blur detector response operator fixtures."""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def camera_gain_profile():
    """Provide a prescribed nonuniform detector response for calibration tests."""
    return np.linspace(0.7, 1.3, 1900, dtype=np.float32)


@pytest.fixture
def camera_codes():
    """All recorded uint16 values for physical gain, offset and saturation checks."""
    return np.arange(65536, dtype=np.uint16).reshape(16, 16, 256)


@pytest.fixture
def detector_response_sample(
    stage_gain, strided, camera_gain_profile, linear_camera_field
):
    """Apply detector sensitivity and ADC conversion to prescribed detected counts."""
    coordinates = tuple(
        np.rint(np.linspace(0, len(axis) - 1, size)).astype(int)
        for axis, size in zip(linear_camera_field.axes, (31, 65, 73), strict=True)
    )
    # The shared physical concentration ramp ranges from 50 to 150. A fixed
    # exposure gives a prescribed 20..1020 detected-photon calibration object.
    photons = 20 + 10 * (linear_camera_field.field[np.ix_(*coordinates)] - 50)
    y = np.linspace(-1, 1, 65, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, 73, dtype=np.float32)[None, :]
    illumination = np.asarray(0.5 + 0.5 * np.exp(-x * x - y * y), dtype=np.float32)
    gain = camera_gain_profile[1040:1113] if stage_gain else np.ones(73, np.float32)
    measured = np.rint(photons * illumination * gain)
    offset, conversion = 100.0, 0.24 / 0.9
    raw = np.rint(measured / conversion + offset).astype(np.uint16)
    if strided:
        raw, illumination = raw[::2, ::2, :], illumination[::2, :]
        measured = measured[::2, ::2, :]
    options = {
        "detector_x_offset": 1040,
        "detector_gain_x": camera_gain_profile if stage_gain else None,
    }
    return SimpleNamespace(
        raw=raw,
        illumination=illumination,
        gain=gain,
        measured=measured,
        offset=offset,
        conversion=conversion,
        options=options,
    )
