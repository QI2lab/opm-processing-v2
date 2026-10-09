"""Calibrated camera codes and physical illumination forward models."""

from types import SimpleNamespace

import numpy as np
import pytest

from opm_processing.imageprocessing.camera import qi2lab_stage_scan_camera_gain
from tests.reference.point_sources import sample_point_source


@pytest.fixture(scope="session")
def camera_codes():
    """Return every uint16 detector code to check offset, gain and saturation."""
    return np.arange(65536, dtype=np.uint16).reshape(16, 16, 256)


@pytest.fixture
def camera_optical_sample(stage_gain, strided):
    """Forward simulate a diffraction emitter with shot noise and detector sensitivity.

    Parameters
    ----------
    stage_gain : bool
        Include the measured detector response in the forward image formation.
    strided : bool
        Return a noncontiguous scan and detector row crop.

    Returns
    -------
    types.SimpleNamespace
        Known illumination and gain, measured photons, uint16 camera data and
        calibration. Only independent camera quantization contributes extra error.
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
    options = {"detector_x_offset": 1040, "apply_stage_scan_gain": stage_gain}
    return SimpleNamespace(
        raw=raw,
        illumination=illumination,
        gain=gain,
        measured=measured,
        offset=offset,
        conversion=conversion,
        options=options,
    )
