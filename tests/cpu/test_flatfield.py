"""Ground-truth integration test for illumination correction."""

from pathlib import Path

import numpy as np
import pytest

from opm_processing.imageprocessing.camera import (
    camera_correct,
    illumination_correct,
    qi2lab_stage_scan_camera_gain,
)
from opm_processing.imageprocessing.flatfield import (
    _flatfield_tile_indices,
    _stage_z_groups,
    estimate_illuminations,
)


@pytest.mark.integration
@pytest.mark.parametrize("output_flatfield", (False, True))
def test_processing_uses_existing_flatfield_with_separate_output(
    opm_v2_projection_zarr,
    tmp_path: Path,
    output_flatfield: bool,
) -> None:
    """Verify source reuse and output precedence through saved pixel values.

    Parameters
    ----------
    opm_v2_projection_zarr : OpmV2ProjectionFixture
        Simulated acquisition with camera calibration and raw images on disk.
    tmp_path : pathlib.Path
        Temporary directory containing the acquisition and processed output.
    output_flatfield : bool
        Write an output-side field that must take precedence over the source field.

    Returns
    -------
    None
        Reopened processed intensities match the selected calibration field.
    """
    from opm_processing.dataio.acquisition import acquisition_stem
    from opm_processing.dataio.position_collection import open_position_collection
    from opm_processing.process import process, write_flatfield

    acquisition = opm_v2_projection_zarr
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    filename = f"{acquisition_stem(acquisition.path)}_flatfield.ome.tif"
    field_shape = (1, acquisition.raw_data.shape[2], *acquisition.raw_data.shape[-2:])
    source_field = np.full(field_shape, 2.0, dtype=np.float32)
    selected_field = source_field
    write_flatfield(
        acquisition.path.parent / filename, source_field, acquisition.pixel_size_um
    )
    if output_flatfield:
        selected_field = np.full(field_shape, 1.25, dtype=np.float32)
        write_flatfield(
            output_dir / filename, selected_field, acquisition.pixel_size_um
        )

    process(
        root_path=acquisition.path,
        output=output_dir,
        flatfield_correction=True,
        deconvolve=False,
        save_float32=True,
        write_fused_max_projection_tiff=False,
    )
    saved = (
        open_position_collection(
            output_dir / f"{acquisition_stem(acquisition.path)}_projection.ome.zarr"
        )
        .arrays[0]
        .read()
        .result()
    )
    calibrated = (
        acquisition.raw_data[:, 0].astype(np.float32) - acquisition.camera_offset
    ) * acquisition.camera_conversion
    expected = (calibrated / selected_field[0])[:, :, np.newaxis]
    np.testing.assert_allclose(saved, expected, rtol=1e-6)


@pytest.mark.unit
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


@pytest.mark.unit
def test_flatfield_tiles_are_evenly_subsampled_within_each_depth():
    """Large depth levels use a bounded, deterministic span of tiles."""
    selected = _flatfield_tile_indices(tuple(range(100)), max_tiles_per_level=8)

    assert len(selected) == 8
    assert selected[0] == 0
    assert selected[-1] == 99
    assert len(set(selected)) == 8
    assert set(np.diff(selected)) <= {14, 15}


@pytest.mark.unit
def test_signal_selection_excludes_empty_channels_and_positions(
    tensorstore_dataset,
):
    """The full-volume TPC mask limits each channel before tile subsampling."""
    from opm_processing import process as process_module

    raw = np.full((1, 4, 2, 3, 4, 8), 100, dtype=np.uint16)
    raw[0, 0, 0, :, :, :4] = 130
    raw[0, 2, 0, :, :, :4] = 130
    datastore = tensorstore_dataset(raw)
    signal_decisions = process_module.build_illumination_signal_decisions(
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


@pytest.mark.unit
def test_illumination_signal_check_stops_after_32_nonempty_candidates(
    tensorstore_dataset,
):
    """Dense acquisitions avoid a full-position signal pre-scan."""
    from opm_processing import process as process_module

    raw = np.full((1, 40, 1, 2, 3, 4), 130, dtype=np.uint16)
    decisions = process_module.build_illumination_signal_decisions(
        tensorstore_dataset(raw),
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
def test_qi2lab_stage_camera_gain_has_fixed_unity_baseline():
    """The saved camera line profile changes only the measured detector trough."""
    gain = qi2lab_stage_scan_camera_gain()

    assert gain.shape == (1900,)
    np.testing.assert_array_equal(gain[:1046], 1.0)
    np.testing.assert_array_equal(gain[1104:], 1.0)
    assert np.argmin(gain) == 1079
    assert gain[1079] == np.float32(0.717388)


@pytest.mark.unit
def test_processing_contract_uses_uint16_raw_float32_intermediates_and_final_cast():
    """Camera, nonlinear gain, illumination, and output boundaries are explicit."""
    from opm_processing.process import format_processed_output

    raw = np.full((2, 3, 1900), 110, dtype=np.uint16)
    camera_corrected = camera_correct(
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
def test_stage_z_flatfield_round_trips_exact_values(
    tmp_path: Path,
):
    """The reusable illumination retains every stage/channel field exactly."""
    from opm_processing.process import (
        read_current_flatfield,
        write_flatfield,
    )

    for stage_level_count in (1, 2):
        path = tmp_path / f"flatfield-{stage_level_count}.ome.tif"
        flatfields = np.ones(
            (stage_level_count, 3, 8, 12),
            dtype=np.float32,
        )
        if stage_level_count > 1:
            flatfields[1] *= 1.25
        write_flatfield(path, flatfields, 0.115)

        actual = read_current_flatfield(path, flatfields.shape)
        assert actual is not None
        np.testing.assert_array_equal(actual, flatfields)


@pytest.mark.unit
def test_flatfield_sample_loading_applies_detector_calibration(
    tensorstore_dataset,
):
    """The estimator's loader must preserve each channel's calibrated pixels."""
    from opm_processing.imageprocessing.flatfield import _camera_corrected_images

    for value in (110, 120):
        raw = np.full((3, 4, 1900), value, dtype=np.uint16)
        raw[:, 0, 0] = 90
        actual = _camera_corrected_images(
            tensorstore_dataset(raw),
            camera_offset=100.0,
            camera_conversion=0.5,
            apply_stage_scan_gain=True,
        )
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual[:, 0, 0], 0)
        np.testing.assert_array_equal(actual[:, 1:, :1046], (value - 100) * 0.5)
        np.testing.assert_allclose(actual[..., 1079], (value - 100) * 0.5 / 0.717388)
        np.testing.assert_array_equal(actual[..., 1104:], (value - 100) * 0.5)


@pytest.mark.unit
def test_single_tile_depth_fields_recover_known_illumination(
    tensorstore_dataset, depth_illumination
):
    """Fit separate depth fields from scan planes with the real BaSiC solver."""
    case = depth_illumination
    estimated = estimate_illuminations(
        tensorstore_dataset(case.raw[None, :, None]),
        camera_offset=100.0,
        camera_conversion=0.25,
        stage_positions_zxy=np.asarray(((10, 20, 30), (0, 20, 30))),
    )
    assert estimated.shape == (2, 1, case.height, case.width)
    assert np.all(np.isfinite(estimated))
    assert np.all(estimated > 0)
    for depth in range(2):
        assert (
            np.corrcoef(estimated[depth, 0].ravel(), case.fields[depth].ravel())[0, 1]
            > 0.95
        )
        calibrated = (case.raw[depth].astype(float) - 100) * 0.25
        corrected = calibrated / estimated[depth, 0]
        before = np.mean(np.abs(calibrated - case.specimen[depth]))
        after = np.mean(np.abs(corrected - case.specimen[depth]))
        assert after < 0.35 * before


@pytest.mark.unit
def test_flatfield_correction_recovers_multitile_multichannel_truth(tiled_illumination):
    """Recover known rectangular illumination fields from a tiled scan."""
    case = tiled_illumination
    estimated = estimate_illuminations(
        case.datastore,
        case.camera_offset,
        case.camera_conversion,
        stage_positions_zxy=case.stage_positions,
        apply_stage_scan_gain=False,
        signal_mask=None,
    )[0]

    camera_corrected = (
        case.raw.astype(np.float32) - case.camera_offset
    ) * case.camera_conversion
    corrected = camera_corrected / estimated[np.newaxis, :, np.newaxis, :, :]
    for channel in range(case.channels):
        truth = case.specimen[:, channel]
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
                case.illuminations[channel].ravel(),
            )[0, 1]
            > 0.95
        )
        before_gain = np.median(before / truth, axis=(0, 1, 2))
        after_gain = np.median(after / truth, axis=(0, 1, 2))
        before_stripe_error = abs(
            before_gain[case.detector_x]
            - np.median(
                np.concatenate(
                    (
                        before_gain[case.detector_x - 12 : case.detector_x - 7],
                        before_gain[case.detector_x + 8 : case.detector_x + 13],
                    )
                )
            )
        )
        after_stripe_error = abs(
            after_gain[case.detector_x]
            - np.median(
                np.concatenate(
                    (
                        after_gain[case.detector_x - 12 : case.detector_x - 7],
                        after_gain[case.detector_x + 8 : case.detector_x + 13],
                    )
                )
            )
        )
        assert after_stripe_error < 0.4 * before_stripe_error, (
            f"channel {channel}: stripe error changed from "
            f"{before_stripe_error:.6f} to {after_stripe_error:.6f}"
        )
