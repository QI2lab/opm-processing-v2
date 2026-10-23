"""Unsigned camera digitization after photon detection and Gaussian read noise."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    """Electron-domain read noise and explicit digitization calibration."""

    read_noise_e: float
    electrons_per_adu: float
    offset_adu: float
    bit_depth: int


def digitize(
    expected_electrons: np.ndarray,
    camera: Camera,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Apply photon statistics, electron read noise, gain/offset and integer ADC.

    Parameters
    ----------
    expected_electrons
        Photoelectron expectation after sensor integration and QE.
    camera
        Camera calibration and read-noise parameters.
    seed
        Seed for one complete camera-noise realization.

    Returns
    -------
    numpy.ndarray
        Unsigned integer ADC pixels. The camera offset remains in the image;
        read noise may produce values below it. No signed image is exported.
    """
    if (
        not np.isfinite(expected_electrons).all()
        or np.any(expected_electrons < 0)
        or camera.electrons_per_adu <= 0
        or camera.read_noise_e < 0
        or not 1 <= camera.bit_depth <= 16
    ):
        raise ValueError("Require nonnegative means/noise and positive camera gain")
    rng = np.random.default_rng(seed)
    detected = rng.poisson(expected_electrons)
    charge = detected + rng.normal(0, camera.read_noise_e, detected.shape)
    raw = np.clip(
        np.rint(camera.offset_adu + charge / camera.electrons_per_adu),
        0,
        2**camera.bit_depth - 1,
    ).astype(np.uint16)
    return raw
