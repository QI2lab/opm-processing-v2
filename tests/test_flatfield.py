"""Ground-truth integration test for illumination correction."""

import warnings
from pathlib import Path

import numpy as np
import tensorstore as ts

from opm_processing.imageprocessing.flatfield import (
    _flatfield_tile_indices,
    _flatfield_working_shape,
    _stage_z_groups,
    estimate_illuminations,
)
from opm_processing.imageprocessing.camera import (
    correct_qi2lab_stage_scan_camera,
    qi2lab_stage_scan_camera_gain,
)


def test_stage_z_groups_use_repeated_xy_depth_not_absolute_tilted_z():
    """Repeated XY visits define depth despite coverslip-dependent absolute Z."""
    levels, groups = _stage_z_groups(
        6,
        np.asarray(
            [
                [10.0, 0.0, 0.0],
                [11.0, 1.0, 0.0],
                [12.5, 0.0, 0.0],
                [13.5, 1.0, 0.0],
                [15.0, 0.0, 0.0],
                [16.0, 1.0, 0.0],
            ]
        ),
    )

    assert levels == (0.0, 1.0, 2.0)
    assert groups == ((0, 1), (2, 3), (4, 5))


def test_flatfield_tiles_are_evenly_subsampled_within_each_depth():
    """Large depth levels use a bounded, deterministic span of tiles."""
    selected = _flatfield_tile_indices(tuple(range(100)), max_tiles_per_level=8)

    assert len(selected) == 8
    assert selected[0] == 0
    assert selected[-1] == 99


def test_empty_check_runs_once_and_filters_illumination_candidates(monkeypatch):
    """The full-volume TPC mask limits each channel before tile subsampling."""
    from opm_processing import process as process_module
    from opm_processing.imageprocessing import flatfield as flatfield_module

    class FakeBasic:
        def __init__(self, **_kwargs):
            self.flatfield = None

        def autotune(self, _images):
            return None

        def fit(self, images):
            self.flatfield = np.ones(images.shape[-2:], dtype=np.float32)

    observed_candidates: list[tuple[int, ...]] = []

    def record_candidates(positions, *, max_tiles_per_level=32):
        observed_candidates.append(positions)
        return positions[:max_tiles_per_level]

    monkeypatch.setattr(flatfield_module, "BaSiC", FakeBasic)
    monkeypatch.setattr(
        flatfield_module,
        "_flatfield_tile_indices",
        record_candidates,
    )
    raw = np.full((1, 4, 2, 3, 4, 8), 100, dtype=np.uint16)
    raw[0, 0, 0, :, :, :4] = 130
    raw[0, 2, 0, :, :, :4] = 130
    datastore = ts.array(raw)
    signal_decisions = process_module._build_illumination_signal_decisions(
        datastore,
        np.zeros(4, dtype=np.int64),
        camera_offset=100.0,
        camera_conversion=1.0,
        threshold=20.0,
        min_signal_fraction=0.01,
        apply_stage_scan_gain=False,
    )

    assert signal_decisions is not None
    np.testing.assert_array_equal(
        signal_decisions,
        np.asarray([[[1, 0], [0, 0], [1, 0], [0, 0]]], dtype=np.int8),
    )

    estimated = estimate_illuminations(
        datastore,
        camera_offset=100.0,
        camera_conversion=1.0,
        stage_positions_zxy=np.asarray(
            [[0.0, 0.0, float(position)] for position in range(4)]
        ),
        signal_mask=signal_decisions == 1,
    )

    assert observed_candidates == [(0, 2), ()]
    np.testing.assert_array_equal(estimated[0, 1], 1.0)


def test_disabled_empty_check_does_not_prescan_tiles(monkeypatch):
    """Without a threshold, processing moves on without reading raw tiles."""
    from opm_processing import process as process_module

    class FailOnRead:
        shape = (1, 2, 3, 4, 5, 6)

        def __getitem__(self, _selection):
            raise AssertionError("disabled empty checking must not read the datastore")

    monkeypatch.setattr(
        process_module,
        "_camera_calibrated_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )
    assert (
        process_module._build_illumination_signal_decisions(
            FailOnRead(),
            np.zeros(2, dtype=np.int64),
            camera_offset=100.0,
            camera_conversion=1.0,
            threshold=None,
            min_signal_fraction=0.01,
            apply_stage_scan_gain=False,
        )
        is None
    )


def test_illumination_signal_check_stops_after_32_nonempty_candidates(monkeypatch):
    """Dense acquisitions avoid a full-position signal pre-scan."""
    from opm_processing import process as process_module

    progress_calls = []

    class RecordingProgress:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.progress = 0
            self.closed = False
            progress_calls.append(self)

        def update(self, amount=1):
            self.progress += amount

        def close(self):
            self.closed = True

    monkeypatch.setattr(process_module, "tqdm", RecordingProgress)
    raw = np.full((1, 40, 1, 2, 3, 4), 130, dtype=np.uint16)
    decisions = process_module._build_illumination_signal_decisions(
        ts.array(raw),
        np.zeros(40, dtype=np.int64),
        camera_offset=100.0,
        camera_conversion=1.0,
        threshold=20.0,
        min_signal_fraction=0.01,
        apply_stage_scan_gain=False,
    )

    assert decisions is not None
    assert np.count_nonzero(decisions == 1) == 32
    assert np.count_nonzero(decisions == -1) == 8
    assert len(progress_calls) == 1
    assert progress_calls[0].kwargs == {
        "total": 32,
        "desc": "Finding illumination tiles",
        "unit": "nonzero tile",
    }
    assert progress_calls[0].progress == 32
    assert progress_calls[0].closed

    unchecked_position = int(np.flatnonzero(decisions[0, :, 0] == -1)[0])
    assert not process_module._tile_is_empty(
        decisions,
        0,
        unchecked_position,
        0,
        np.full((2, 3, 4), 30.0, dtype=np.float32),
        20.0,
        0.01,
    )
    assert decisions[0, unchecked_position, 0] == 1


def test_qi2lab_stage_camera_gain_has_fixed_unity_baseline():
    """The saved camera line profile changes only the measured detector trough."""
    gain = qi2lab_stage_scan_camera_gain()

    assert gain.shape == (1900,)
    np.testing.assert_array_equal(gain[:1046], 1.0)
    np.testing.assert_array_equal(gain[1104:], 1.0)
    assert np.argmin(gain) == 1079
    assert gain[1079] == np.float32(0.717388)


def test_qi2lab_stage_camera_gain_broadcasts_over_scan_and_y():
    """One detector-X gain vector corrects every scan/Y pixel in a tile or ROI."""
    gain = qi2lab_stage_scan_camera_gain()
    expected = np.full((3, 5, 1900), 42.0, dtype=np.float32)
    measured = expected * gain[np.newaxis, np.newaxis, :]

    corrected = correct_qi2lab_stage_scan_camera(measured)
    np.testing.assert_allclose(corrected, expected, rtol=1e-6)

    roi = measured[..., 1040:1110]
    corrected_roi = correct_qi2lab_stage_scan_camera(roi, x_offset=1040)
    np.testing.assert_allclose(corrected_roi, expected[..., 1040:1110], rtol=1e-6)


def test_processing_contract_uses_uint16_raw_float32_intermediates_and_final_cast():
    """Camera, nonlinear gain, illumination, and output boundaries are explicit."""
    from opm_processing.process import (
        _apply_illumination_correction,
        _camera_calibrated_image,
        _format_processed_output,
    )

    raw = np.full((2, 3, 1900), 110, dtype=np.uint16)
    camera_corrected = _camera_calibrated_image(
        raw,
        100.0,
        1.0,
        apply_stage_scan_gain=True,
    )
    assert camera_corrected.dtype == np.float32
    np.testing.assert_allclose(camera_corrected[..., 100], 10.0)
    np.testing.assert_allclose(
        camera_corrected[..., 1079],
        10.0 / qi2lab_stage_scan_camera_gain()[1079],
    )

    illumination = np.full((3, 1900), 2.0, dtype=np.float32)
    corrected = _apply_illumination_correction(camera_corrected, illumination)
    assert corrected.dtype == np.float32
    assert _format_processed_output(corrected, False).dtype == np.uint16
    assert _format_processed_output(corrected, True).dtype == np.float32

    with np.testing.assert_raises(TypeError):
        _camera_calibrated_image(raw.astype(np.float32), 100.0, 1.0)


def test_stage_z_flatfield_round_trips_exact_values(
    tmp_path: Path,
):
    """The reusable illumination retains every stage/channel field exactly."""
    from opm_processing.process import (
        _read_current_flatfield,
        _write_flatfield,
    )

    for stage_level_count in (1, 2):
        path = tmp_path / f"flatfield-{stage_level_count}.ome.tif"
        flatfields = np.ones(
            (stage_level_count, 3, 8, 12),
            dtype=np.float32,
        )
        if stage_level_count > 1:
            flatfields[1] *= 1.25
        _write_flatfield(path, flatfields, 0.115)

        actual = _read_current_flatfield(path, flatfields.shape)
        assert actual is not None
        np.testing.assert_array_equal(actual, flatfields)


def test_estimator_fits_physical_stage_z_levels_independently(monkeypatch):
    """Stage-Z groups produce distinct fields while scan planes remain samples."""
    from opm_processing.imageprocessing import flatfield as flatfield_module

    class FakeBasic:
        def __init__(self, **_kwargs):
            self.flatfield = None

        def autotune(self, _images):
            return None

        def fit(self, images):
            field = np.median(images, axis=0)
            self.flatfield = field / np.mean(field)

    monkeypatch.setattr(flatfield_module, "BaSiC", FakeBasic)
    height, width = 12, 20
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    fields = np.stack((1.0 + 0.25 * x, 1.0 - 0.25 * x))
    raw = np.empty((1, 4, 1, 3, height, width), dtype=np.uint16)
    for position in range(4):
        raw[0, position, 0] = np.rint(1000.0 * fields[position // 2]).astype(np.uint16)

    estimated = estimate_illuminations(
        ts.array(raw),
        camera_offset=0.0,
        camera_conversion=1.0,
        stage_positions_zxy=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 1.0, 0.0],
                [5.0, 0.0, 0.0],
                [5.1, 1.0, 0.0],
            ]
        ),
    )

    assert estimated.shape == (2, 1, height, width)
    assert estimated[0, 0, :, -1].mean() > estimated[0, 0, :, 0].mean()
    assert estimated[1, 0, :, -1].mean() < estimated[1, 0, :, 0].mean()


def test_stage_flatfield_estimation_applies_camera_contract_per_channel(
    monkeypatch,
):
    """Each channel reaches stage-gain fitting as scalar-corrected float32."""
    from opm_processing.imageprocessing import flatfield as flatfield_module

    class FakeBasic:
        def __init__(self, **_kwargs):
            self.flatfield = None

        def autotune(self, _images):
            return None

        def fit(self, images):
            self.flatfield = np.ones(images.shape[-2:], dtype=np.float32)

    observed: list[tuple[tuple[int, ...], np.dtype, float]] = []

    def record_stage_gain(images, *, copy=False):
        assert copy is False
        observed.append((images.shape, images.dtype, float(images.mean())))
        return images

    monkeypatch.setattr(flatfield_module, "BaSiC", FakeBasic)
    monkeypatch.setattr(
        flatfield_module,
        "correct_qi2lab_stage_scan_camera",
        record_stage_gain,
    )

    raw = np.empty((1, 2, 2, 3, 4, 1900), dtype=np.uint16)
    raw[:, :, 0] = 110
    raw[:, :, 1] = 120
    estimate_illuminations(
        ts.array(raw),
        camera_offset=100.0,
        camera_conversion=0.5,
        stage_positions_zxy=np.asarray([[10.0, 0.0, 0.0], [11.0, 1.0, 0.0]]),
        apply_stage_scan_gain=True,
    )

    assert observed
    assert all(shape == (3, 4, 1900) for shape, _dtype, _mean in observed)
    assert all(dtype == np.dtype(np.float32) for _shape, dtype, _mean in observed)
    assert sorted({round(mean, 3) for _shape, _dtype, mean in observed}) == [
        5.0,
        10.0,
    ]


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
    datastore = ts.array(raw[np.newaxis, ...])

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

    camera_corrected = (raw.astype(np.float32) - camera_offset) * camera_conversion
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
        before_gain = np.median(before / truth, axis=(0, 1, 2))
        after_gain = np.median(after / truth, axis=(0, 1, 2))
        before_stripe_error = abs(
            before_gain[detector_x]
            - np.median(
                np.concatenate(
                    (
                        before_gain[detector_x - 12 : detector_x - 7],
                        before_gain[detector_x + 8 : detector_x + 13],
                    )
                )
            )
        )
        after_stripe_error = abs(
            after_gain[detector_x]
            - np.median(
                np.concatenate(
                    (
                        after_gain[detector_x - 12 : detector_x - 7],
                        after_gain[detector_x + 8 : detector_x + 13],
                    )
                )
            )
        )
        assert after_stripe_error < 0.4 * before_stripe_error, (
            f"channel {channel}: stripe error changed from "
            f"{before_stripe_error:.6f} to {after_stripe_error:.6f}"
        )
