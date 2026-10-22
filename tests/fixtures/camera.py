"""Calibrated camera codes and post-blur detector response operator fixtures."""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="session")
def camera_gain_profile():
    """Provide a prescribed nonuniform detector response for calibration tests."""
    return np.linspace(0.7, 1.3, 1900, dtype=np.float32)


@pytest.fixture(scope="session")
def camera_exposure():
    """Set exposure by peak expected detected photons per camera pixel.

    Each detected photon generates one photoelectron. The supplied expectation
    already includes QE, optical throughput and sensor integration. Scaling it
    changes the source exposure before the Poisson camera draw, not noisy data.
    """

    def at_peak(expectation, peak_detected_photons):
        """Set the declared mean detector exposure before the Poisson draw."""
        expectation = np.asarray(expectation, dtype=np.float64)
        maximum = float(expectation.max())
        if (
            not np.isfinite(expectation).all()
            or np.any(expectation < 0)
            or maximum <= 0
            or not np.isfinite(peak_detected_photons)
            or peak_detected_photons <= 0
        ):
            raise ValueError("Require positive exposure and finite nonnegative rates")
        return expectation * (peak_detected_photons / maximum)

    return at_peak


@pytest.fixture(scope="session")
def camera_codes():
    """All recorded uint16 values for physical gain, offset and saturation checks."""
    return np.arange(65536, dtype=np.uint16).reshape(16, 16, 256)


@pytest.fixture
def photon_splitter(cupy_gpu):
    """Run MATLAB's unclipped first-half split with a reproducible GPU generator."""
    from opm_processing.imageprocessing import rlgc

    def split(observed, seed=19):
        """Split prescribed observations with a reproducible GPU random generator."""
        counts = cupy_gpu.asarray(observed, dtype=cupy_gpu.float32)
        rng = cupy_gpu.random.default_rng(seed)
        first = rlgc._split_observed_counts(counts, rng)
        return cupy_gpu.asnumpy(first)

    return SimpleNamespace(split=split)


@pytest.fixture
def detector_response_sample(stage_gain, strided, camera_gain_profile):
    """Apply detector sensitivity and ADC conversion to prescribed detected counts."""
    photons = np.random.default_rng(47).integers(20, 1021, (31, 65, 73))
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


@pytest.fixture(scope="module")
def camera_likelihood_sample(camera_calibration, rate):
    """Share independent camera data and its true-rate score across hypotheses."""
    from tests.simulation.camera import digitize
    from tests.simulation.camera_likelihood import camera_log_likelihood

    raw = digitize(np.full(20000, rate), camera_calibration, seed=31)
    return raw, camera_log_likelihood(raw, rate, camera_calibration)
