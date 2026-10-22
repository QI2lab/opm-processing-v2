"""Illumination correction units against known detector response."""

import numpy as np
import pytest
import tensorstore as ts

from opm_processing.imageprocessing.camera import (
    camera_correct,
    illumination_correct,
)
from opm_processing.imageprocessing.flatfield import (
    _flatfield_sample_indices,
    _separable_residual_calibration,
    estimate_illuminations,
)


@pytest.mark.unit
def test_stage_z_groups_use_repeated_xy_depth_not_absolute_tilted_z():
    """Repeated XY visits define depth despite coverslip-dependent absolute Z."""
    from opm_processing.imageprocessing.coordinates import stage_z_level_indices

    indices = stage_z_level_indices(
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

    np.testing.assert_array_equal(indices, (0, 0, 1, 1, 2, 2))


@pytest.mark.unit
def test_flatfield_tiles_are_evenly_subsampled_within_each_depth():
    """Large depth levels use a bounded, deterministic span of tiles."""
    samples = _flatfield_sample_indices(tuple(range(100)), 12, max_tiles_per_level=8)
    selected = [position for position, _planes in samples]

    assert len(selected) == 8
    assert selected[0] == 0
    assert selected[-1] == 99
    assert len(set(selected)) == 8
    assert set(np.diff(selected)) <= {14, 15}
    assert samples == _flatfield_sample_indices(
        tuple(range(100)), 12, max_tiles_per_level=8
    )
    for _position, planes in samples:
        assert len(planes) == len(set(planes)) == 10
        assert 0 <= min(planes) <= max(planes) < 12


@pytest.mark.unit
def test_signal_selection_excludes_empty_channels_and_positions():
    """The full-volume TPC mask limits each channel before tile subsampling."""
    from opm_processing import process as process_module

    raw = np.full((1, 4, 2, 3, 4, 8), 100, dtype=np.uint16)
    raw[0, 0, 0, :, :, :4] = 130
    raw[0, 2, 0, :, :, :4] = 130
    datastore = ts.array(raw)
    signal_decisions = process_module.build_illumination_signal_decisions(
        datastore,
        np.zeros(4, dtype=np.int64),
        camera_offset=100.0,
        camera_conversion=1.0,
        threshold=20.0,
        min_signal_fraction=0.01,
        detector_gain_x=None,
    )

    assert signal_decisions is not None
    np.testing.assert_array_equal(
        signal_decisions,
        np.asarray([[[1, 0], [0, 0], [1, 0], [0, 0]]], dtype=np.int8),
    )


@pytest.mark.unit
def test_illumination_signal_check_stops_after_32_nonempty_candidates():
    """Dense acquisitions avoid a full-position signal pre-scan."""
    from opm_processing import process as process_module

    raw = np.full((1, 40, 1, 2, 3, 4), 130, dtype=np.uint16)
    decisions = process_module.build_illumination_signal_decisions(
        ts.array(raw),
        np.zeros(40, dtype=np.int64),
        camera_offset=100.0,
        camera_conversion=1.0,
        threshold=20.0,
        min_signal_fraction=0.01,
        detector_gain_x=None,
    )

    assert decisions is not None
    assert np.count_nonzero(decisions == 1) == 32
    assert np.count_nonzero(decisions == -1) == 8
    unchecked_position = int(np.flatnonzero(decisions[0, :, 0] == -1)[0])
    assert not process_module.tile_is_empty(
        decisions,
        0,
        unchecked_position,
        0,
        np.full((2, 3, 4), 30.0, dtype=np.float32),
        20.0,
        0.01,
    )
    assert decisions[0, unchecked_position, 0] == 1


@pytest.mark.unit
def test_processing_contract_uses_uint16_raw_float32_intermediates_and_final_cast(
    camera_gain_profile,
):
    """Camera, nonlinear gain, illumination, and output boundaries are explicit."""
    from opm_processing.process import format_processed_output

    raw = np.full((2, 3, 1900), 110, dtype=np.uint16)
    camera_corrected = camera_correct(
        raw,
        100.0,
        1.0,
        detector_gain_x=camera_gain_profile,
    )
    assert camera_corrected.dtype == np.float32
    np.testing.assert_allclose(
        camera_corrected[..., 100], 10.0 / camera_gain_profile[100]
    )
    np.testing.assert_allclose(
        camera_corrected[..., 1079],
        10.0 / camera_gain_profile[1079],
    )

    illumination = np.full((3, 1900), 2.0, dtype=np.float32)
    corrected = illumination_correct(camera_corrected, illumination)
    assert corrected.dtype == np.float32
    np.testing.assert_allclose(corrected, camera_corrected / 2)
    boundary = np.array([-1, 0, 0.24, 5.9, 65535, 70000], dtype=np.float32)
    integers = format_processed_output(boundary, False)
    assert integers.dtype == np.uint16
    np.testing.assert_array_equal(integers, [0, 0, 0, 5, 65535, 65535])
    floats = format_processed_output(boundary, True)
    assert floats.dtype == np.float32
    np.testing.assert_array_equal(floats, boundary)


@pytest.mark.unit
@pytest.mark.parametrize("decision", (None, -1), ids=("all", "unchecked-excluded"))
@pytest.mark.parametrize("topology", ("single", "tiles", "depths"))
@pytest.mark.parametrize("storage", (False, True), ids=("array", "store"))
def test_fit_samples_preserve_calibration_and_depth_groups(
    illumination_fit_case, topology, decision
):
    """Keep time-zero photon units and odd detector dimensions through fit preparation."""
    case = illumination_fit_case
    mask = None
    selected = np.arange(len(case.levels))
    if decision is not None:
        mask = np.ones(case.raw.shape[:3], np.int8)
        if len(selected) > 1:
            mask[0, 1] = decision
            selected = selected[selected != 1]
        else:
            mask[1] = decision  # Later times are not used for estimation.
    actual = estimate_illuminations(
        case.datastore,
        100,
        0.5,
        case.coordinates,
        detector_gain_x=case.gain,
        signal_mask=mask,
    )
    expected_levels = [
        np.median(case.levels[selected, channel], axis=1)
        if topology == "tiles"
        else case.levels[position, channel]
        for position in (selected if topology == "depths" else (0,))
        for channel in range(2)
    ]
    for images, levels in zip(case.calls, expected_levels, strict=True):
        np.testing.assert_allclose(
            images,
            np.broadcast_to(levels[:, None, None], images.shape),
            rtol=4 * np.finfo(np.float32).eps,
            atol=0,
        )
    assert actual.shape == ((2, 5, 7) if case.coordinates is None else (1, 2, 5, 7))
    np.testing.assert_allclose(actual, 1, rtol=4 * np.finfo(np.float32).eps, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("level", (100, 110), ids=("empty", "one-image"))
def test_empty_or_insufficient_samples_are_handled_before_fitting(level):
    """Empty channels use unity; a single informative image cannot constrain a fit."""
    raw = np.full((1, 1, 1, 1, 5, 7), level, np.uint16)
    if level == 100:
        np.testing.assert_array_equal(
            estimate_illuminations(raw, 100, 1), np.ones((1, 5, 7))
        )
    else:
        with pytest.raises(ValueError, match="at least two nonempty"):
            estimate_illuminations(raw, 100, 1)


@pytest.mark.unit
@pytest.mark.parametrize(
    "option,value",
    [
        ("camera_conversion", 0),
        ("camera_offset", np.nan),
        ("stage_positions_zxy", np.zeros((2, 3))),
        ("signal_mask", np.ones((1, 1, 2))),
    ],
)
def test_invalid_illumination_calibration_is_rejected(option, value):
    """Reject inconsistent acquisition metadata before loading samples or fitting."""
    options = {"camera_offset": 100, "camera_conversion": 1, option: value}
    with pytest.raises(ValueError):
        estimate_illuminations(np.full((1, 1, 1, 3, 5, 7), 120, np.uint16), **options)


@pytest.mark.unit
@pytest.mark.parametrize("level", (0, -1, np.nan))
def test_nonpositive_or_nonfinite_fits_are_rejected(level):
    """Prevent unusable fitted fields from entering residual calibration."""
    with pytest.raises(ValueError, match="invalid illumination field"):
        _separable_residual_calibration(
            np.full((5, 7), level, np.float32), np.ones((3, 5, 7), np.float32)
        )
