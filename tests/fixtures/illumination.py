"""Known depth-dependent illumination and detector stripes for BaSiC recovery."""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def depth_illumination():
    """Simulate known fluorescence multiplied by physical illumination and camera gain.

    Returns
    -------
    types.SimpleNamespace
        Ground fluorescence, detector sensitivity and quantized camera data.
        Geometry, spectral channels and camera calibration are fixed together.
    """
    rng = np.random.default_rng(37)
    height, width = 32, 64
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing="ij"
    )
    fields = np.stack((1 + 0.25 * xx + 0.1 * yy, 1 - 0.2 * xx + 0.15 * yy))
    specimen = rng.uniform(750, 1250, (2, 12, height, width))
    raw = np.rint(specimen * fields[:, None] / 0.25 + 100).astype(np.uint16)
    return SimpleNamespace(
        fields=fields, height=height, raw=raw, specimen=specimen, width=width
    )


@pytest.fixture
def tiled_illumination(
    tensorstore_dataset,
):
    """Simulate known fluorescence multiplied by physical illumination and camera gain.

    Parameters
    ----------
    tensorstore_dataset : callable
        Shared dataset double holding simulated raw camera pixels.

    Returns
    -------
    types.SimpleNamespace
        Ground fluorescence, detector sensitivity and quantized camera data.
        Geometry, spectral channels and camera calibration are fixed together.
    """
    rng = np.random.default_rng(7)
    positions, channels, scan_planes = 6, 2, 12
    height, width = 64, 128
    camera_offset = 100.0
    camera_conversion = 0.25

    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, np.newaxis]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[np.newaxis, :]
    illuminations = np.stack(
        (
            1.0 + 0.28 * yy + 0.12 * xx + 0.08 * yy * xx,
            1.0 - 0.22 * yy + 0.16 * xx - 0.06 * yy * xx,
        )
    )
    detector_x = 78
    detector_distance = (np.arange(width, dtype=np.float32) - detector_x) / 3.0
    detector_stripe = 1.0 - 0.12 * np.exp(-0.5 * detector_distance**2)
    illuminations *= detector_stripe[np.newaxis, np.newaxis, :]
    illuminations /= illuminations.mean(axis=(1, 2), keepdims=True)

    specimen = rng.uniform(
        750,
        1250,
        size=(positions, channels, scan_planes, height, width),
    ).astype(np.float32)
    raw = np.rint(
        specimen * illuminations[np.newaxis, :, np.newaxis, :, :] / camera_conversion
        + camera_offset
    ).astype(np.uint16)
    datastore = tensorstore_dataset(raw[np.newaxis, ...])
    stage_positions = np.zeros((positions, 3))
    stage_positions[:, 1] = np.arange(positions) * 100.0

    return SimpleNamespace(
        camera_conversion=camera_conversion,
        camera_offset=camera_offset,
        channels=channels,
        datastore=datastore,
        detector_x=detector_x,
        height=height,
        illuminations=illuminations,
        raw=raw,
        specimen=specimen,
        stage_positions=stage_positions,
        width=width,
    )
