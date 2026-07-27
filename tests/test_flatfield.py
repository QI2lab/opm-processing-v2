"""Ground-truth integration test for illumination correction."""

import warnings

import numpy as np

from opm_processing.imageprocessing.flatfield import (
    _flatfield_working_shape,
    estimate_illuminations,
)


def test_flatfield_correction_recovers_multitile_multichannel_truth():
    """Recover known rectangular illumination fields from a tiled scan."""
    rng = np.random.default_rng(7)
    positions, channels, scan_planes = 6, 2, 12
    height, width = 64, 128
    camera_offset = 100.0
    camera_conversion = 0.25
    assert _flatfield_working_shape((height, width)) == (32, 64)

    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, np.newaxis]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[np.newaxis, :]
    illuminations = np.stack(
        (
            1.0 + 0.28 * yy + 0.12 * xx + 0.08 * yy * xx,
            1.0 - 0.22 * yy + 0.16 * xx - 0.06 * yy * xx,
        )
    )
    illuminations /= illuminations.mean(axis=(1, 2), keepdims=True)

    specimen = rng.uniform(
        750,
        1250,
        size=(positions, channels, scan_planes, height, width),
    ).astype(np.float32)
    raw = np.rint(
        specimen * illuminations[np.newaxis, :, np.newaxis, :, :]
        / camera_conversion
        + camera_offset
    ).astype(np.uint16)
    datastore = raw[np.newaxis, ...]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="sort_intensity=False while is_timelapse=False.*",
            category=UserWarning,
        )
        estimated = estimate_illuminations(
            datastore,
            camera_offset,
            camera_conversion,
        )

    camera_corrected = (
        raw.astype(np.float32) - camera_offset
    ) * camera_conversion
    corrected = camera_corrected / estimated[np.newaxis, :, np.newaxis, :, :]
    for channel in range(channels):
        truth = specimen[:, channel]
        before = camera_corrected[:, channel]
        after = corrected[:, channel]
        scale = np.median(truth, axis=(-2, -1)) / np.median(
            after,
            axis=(-2, -1),
        )
        after = after * scale[..., np.newaxis, np.newaxis]
        before_error = np.mean(np.abs(before - truth)) / np.mean(truth)
        after_error = np.mean(np.abs(after - truth)) / np.mean(truth)

        assert after_error < 0.35 * before_error
        assert (
            np.corrcoef(
                estimated[channel].ravel(),
                illuminations[channel].ravel(),
            )[0, 1]
            > 0.95
        )
