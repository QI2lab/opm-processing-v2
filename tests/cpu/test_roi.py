"""Physical ROI interchange and world-to-skew coordinate coverage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from yaozarrs import open_group

from opm_processing.dataio.position_collection import (
    create_position_collection,
)
from opm_processing.dataio.processing_state import ProcessingState
from opm_processing.dataio.roi import (
    PhysicalRoi,
    align_bounds_to_grid,
    roi_from_image_pixel_rectangle,
    roi_from_world_rectangle,
    world_roi_to_skewed_bounds,
)
from opm_processing.imageprocessing.maxtilefusion import regenerate_fused_max_projection
from opm_processing.imageprocessing.tilefusion import TileFusion


@pytest.mark.unit
@pytest.mark.parametrize("factor,z_bin", [(1, 1), (3, 2)])
def test_roi_plan_keeps_raw_origin_clipped_halos_and_refined_extent(factor, z_bin):
    """Plan the same physical rectangle from native and refined scan samples."""
    from opm_processing.process import plan_roi_tiles

    roi = PhysicalRoi(
        bounds_yx_um=(103, 107, 202, 214),
        source_path=Path("preview.ome.zarr"),
        grid_origin_yx_um=(100, 200),
        pixel_size_yx_um=(1, 1),
        position_indices=(4, 5),
        tile_footprints=tuple(
            {
                "time_index": 1,
                "position_index": position,
                "origin_zyx_um": (7, y + 4, 201),
                "raw_origin_zyx_um": (7, y, 200),
                "bounds_yx_um": (y, y + 24, 200, 212),
            }
            for position, y in ((4, 100), (5, 1000))
        ),
    )
    options = {
        "channel_count": 2,
        "pixel_size_um": 1,
        "scan_step_um": 2,
        "angle_deg": 30,
        "z_downsample_level": z_bin,
        "halo_scan": 1,
        "halo_x": 2,
        "scan_upsample_factor": factor,
    }
    (plan,) = plan_roi_tiles(roi, (0, 1), (4, 5), (10, 4, 12), **options)
    assert (plan["time_index"], plan["position_index"]) == (1, 4)
    assert plan["skewed_bounds_sx"] == [0, 6, 0, 12]
    assert plan["skewed_shape_syx"] == [6, 4, 12]
    assert plan["deskew_crop_yx"] == [3, 7, 2, 12]
    assert plan["shape_tczyx"] == [1, 2, 2 // z_bin, 4, 10]
    assert plan["origin_zyx_um"] == [7 + (z_bin - 1) / 2, 103, 202]
    if factor == 3:
        assert plan["reconstruction_shape_syx"] == [16, 4, 12]
    else:
        assert "reconstruction_shape_syx" not in plan
    with pytest.raises(ValueError, match="does not contain"):
        plan_roi_tiles(roi, (0,), (4, 5), (10, 4, 12), **options)


@pytest.mark.unit
@pytest.mark.parametrize("separation", ["paired", "coincident", "time"])
def test_roi_registration_context_keeps_selection_limits(
    acquisition_metadata, separation
):
    """Use the selected overlap before adding search and SSIM read margins."""
    from opm_processing.process_roi import roi_with_registration_context

    acquisition = replace(
        acquisition_metadata,
        shape=(2, 2, 1, 100, 40, 200),
        pixel_size_um=1,
        scan_axis_step_um=1,
        angle_deg=30,
    )
    roi = PhysicalRoi(
        bounds_yx_um=(40, 60, 100, 120),
        source_path=Path("preview.ome.zarr"),
        grid_origin_yx_um=(0, 0),
        pixel_size_yx_um=(1, 1),
        position_indices=(0, 1),
        tile_footprints=tuple(
            {
                "time_index": index if separation == "time" else 0,
                "position_index": index,
                "origin_zyx_um": (0, 0, x),
                "bounds_yx_um": (0, 100, x, x + 120),
            }
            for index, x in enumerate((0, 80 if separation != "coincident" else 0))
        ),
    )
    expanded, limits = roi_with_registration_context(roi, acquisition, 1)
    assert roi.bounds_yx_um == (40, 60, 100, 120)
    if separation == "paired":
        # Each selected overlap axis is 20 pixels; retain at least half.
        # The 15-sample SSIM window at fivefold sampling adds 40 pixels.
        assert limits == (10, 10, 10)
        assert expanded.bounds_yx_um == (-10, 110, 50, 170)
        assert expanded.tile_footprints == roi.tile_footprints
    else:
        assert expanded is roi and limits is None


@pytest.mark.unit
def test_napari_world_rectangle_round_trips_grid_and_all_z(
    record_registration, fused_store_factory, acquisition_factory, tmp_path: Path
) -> None:
    """Save a pyramid-independent ROI plus every overlapping stage-Z level."""
    acquisition_factory(np.zeros((1, 10, 1, 1, 2, 2), np.uint16), mode="projection")
    source = tmp_path / "sample_max_z_fused.ome.zarr"
    fused_store_factory(
        np.zeros((1, 1, 1, 8, 8), np.uint16),
        name="sample_max_z_fused",
        spacing=(1, 0.5, 0.25),
        origin=(3, 10, 20),
    )
    tiles = [
        {"time_index": 0, "position_index": 2, "origin_zyx_um": [0, 10, 20]},
        {"time_index": 0, "position_index": 5, "origin_zyx_um": [4, 10, 20]},
        {"time_index": 0, "position_index": 9, "origin_zyx_um": [0, 30, 40]},
    ]
    processed = create_position_collection(
        tmp_path / "sample_projection.ome.zarr",
        (1, 10, 1, 1, 6, 8),
        (1.0, 0.5, 0.25),
        stage_positions=((0, 0, 0),) * 10,
    )
    del processed
    record_registration(tmp_path / "sample_projection.ome.zarr", tiles)

    roi = roi_from_world_rectangle(
        source,
        np.asarray(
            [
                [0.0, 0.0, 0.0, 10.6, 20.3],
                [0.0, 0.0, 0.0, 12.1, 21.1],
            ]
        ),
    )
    assert roi.bounds_yx_um == (10.5, 12.5, 20.25, 21.25)
    assert roi.position_indices == (2, 5)
    assert len(roi.tile_footprints) == 2

    path = roi.write(tmp_path / "selection.json")
    reopened = PhysicalRoi.read(path)
    assert reopened == roi


@pytest.mark.unit
def test_napari_image_pixels_are_converted_to_ngff_physical_coordinates(
    record_registration,
    fused_store_factory,
    acquisition_factory,
    tmp_path: Path,
) -> None:
    """Apply NGFF translation after mapping the Shapes layer to Image data."""
    acquisition_factory(np.zeros((1, 1, 1, 1, 2, 2), np.uint16), mode="projection")
    source = tmp_path / "sample_max_z_fused.ome.zarr"
    fused_store_factory(
        np.zeros((1, 1, 1, 5000, 5000), np.uint16),
        name="sample_max_z_fused",
        spacing=(1, 0.115, 0.115),
        origin=(0, -8192.189, 213.58),
    )
    processed_path = tmp_path / "sample_projection.ome.zarr"
    create_position_collection(
        processed_path,
        (1, 1, 1, 1, 5000, 5000),
        (1.0, 0.115, 0.115),
        stage_positions=((0, 0, 0),),
    )
    record_registration(
        processed_path,
        [
            {
                "time_index": 0,
                "position_index": 0,
                "origin_zyx_um": [0.0, -8192.189, 213.58],
            }
        ],
    )

    roi = roi_from_image_pixel_rectangle(
        source,
        np.asarray([[0, 0, 1808.0, 3554.0], [0, 0, 2148.0, 4262.0]]),
    )

    assert roi.bounds_yx_um == (-7984.269, -7945.169, 622.29, 703.71)


@pytest.mark.unit
def test_world_roi_maps_to_skewed_scan_x_and_keeps_all_camera_y() -> None:
    """Enclose the diagonal lab-Y preimage without defining a camera-Y crop."""
    bounds = world_roi_to_skewed_bounds(
        (103.0, 107.0, 202.0, 204.0),
        (100.0, 200.0),
        (10, 5, 12),
        pixel_size_um=1.0,
        scan_step_um=2.0,
        angle_deg=30.0,
        halo_scan=1,
        halo_x=2,
    )
    assert bounds is not None
    assert bounds.scan_start == 0
    assert bounds.scan_stop == 6
    assert bounds.x_start == 0
    assert bounds.x_stop == 6

    # Every skewed sample mapping into the requested lab-Y range is enclosed,
    # for every camera-Y row (which spans lab Z).
    for scan in range(10):
        for camera_y in range(5):
            lab_y = scan * 2.0 + camera_y * np.cos(np.deg2rad(30.0))
            if 3.0 <= lab_y < 7.0:
                assert bounds.scan_start <= scan < bounds.scan_stop


@pytest.mark.unit
def test_registered_tile_origin_is_selected_per_timepoint() -> None:
    """Use registered placement for each timepoint when mapping back to a tile."""
    roi = PhysicalRoi(
        bounds_yx_um=(0.0, 2.0, 0.0, 2.0),
        source_path=Path("preview.ome.zarr"),
        grid_origin_yx_um=(0.0, 0.0),
        pixel_size_yx_um=(1.0, 1.0),
        position_indices=(4,),
        tile_footprints=(
            {
                "time_index": 0,
                "position_index": 4,
                "bounds_yx_um": [10.0, 20.0, 30.0, 40.0],
            },
            {
                "time_index": 1,
                "position_index": 4,
                "bounds_yx_um": [11.0, 21.0, 32.0, 42.0],
            },
        ),
    )
    assert roi.tile_origin_yx_um(0, 4, (99.0, 99.0)) == (10.0, 30.0)
    assert roi.tile_origin_yx_um(1, 4, (99.0, 99.0)) == (11.0, 32.0)


@pytest.mark.unit
@pytest.mark.parametrize("external", [False, True])
def test_roi_fusion_keeps_multiple_stage_z_levels_and_crops_only_yx(
    record_registration,
    tmp_path: Path,
    external,
    acquisition_factory,
) -> None:
    """Filter arbitrary source positions while preserving their complete Z span."""
    processed_path = tmp_path / "sample_projection.ome.zarr"
    collection = create_position_collection(
        processed_path,
        (1, 3, 1, 1, 4, 4),
        (1.0, 1.0, 1.0),
        stage_positions=((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (0.0, 100.0, 100.0)),
    )
    observed = np.arange(100, 116, dtype=np.uint16).reshape(1, 4, 4)
    for array in collection.arrays:
        array.write(observed[None, None]).result()
    tiles = [
        {"time_index": 0, "position_index": 0, "origin_zyx_um": [0, 0, 0]},
        {"time_index": 0, "position_index": 1, "origin_zyx_um": [5, 0, 0]},
        {"time_index": 0, "position_index": 2, "origin_zyx_um": [0, 100, 100]},
    ]
    acquisition_factory(
        np.broadcast_to(observed, (1, 3, 1, 1, 4, 4)),
        mode="projection",
        pixel_size_um=1.0,
        scan_axis_step_um=1.0,
        angle_deg=45.0,
        camera_offset=0.0,
        camera_conversion=1.0,
        stage_positions_zxy=((0.0, 0.0, 0.0),) * 3,
        orientations=(("camera_Zstage_orientation", "negative"),),
    )
    record_registration(processed_path, tiles)
    roi = PhysicalRoi(
        bounds_yx_um=(-2.0, 3.0, -1.0, 3.0),
        source_path=processed_path,
        grid_origin_yx_um=(0.0, 0.0),
        pixel_size_yx_um=(1.0, 1.0),
        position_indices=(0, 1),
        tile_footprints=(
            {
                "time_index": 0,
                "position_index": 0,
                "origin_zyx_um": [0.0, 0.0, 0.0],
                "bounds_yx_um": [0.0, 4.0, 0.0, 4.0],
            },
            {
                "time_index": 0,
                "position_index": 1,
                "origin_zyx_um": [5.0, 0.0, 0.0],
                "bounds_yx_um": [0.0, 4.0, 0.0, 4.0],
            },
        ),
    )

    fusion = TileFusion(
        processed_path,
        output_dir=tmp_path / "rendered" if external else None,
        roi_selection=roi,
        max_workers=1,
        reverse_stage_y=False,
        reverse_stage_z=False,
    )
    assert fusion._source_position_indices == (0, 1)
    assert fusion._reuse_registered_roi_placements is True
    assert fusion.position_dim == 2
    assert len(fusion.position_arrays) == 2
    assert len(fusion._tile_positions) == 2
    assert {position[0] for position in fusion._tile_positions} == {0.0, 5.0}

    fusion._compute_fused_image_space()
    assert fusion.offset_um == (0.0, -2.0, -1.0)
    assert fusion.unpadded_shape == (6, 5, 4)
    fusion.run()
    fused = fusion.output_dir / "sample_fused.ome.zarr"
    written = open_group(fused)["0"].to_tensorstore()
    expected = np.zeros((1, 1, 6, 5, 4), dtype=np.uint16)
    expected[..., (0, 5), 2:5, 1:4] = observed[..., :3, :3]
    np.testing.assert_array_equal(written.read().result()[..., :6, :5, :4], expected)
    state = ProcessingState.read(tmp_path / "sample.processing.json")
    assert state.registered_output_for_fused(fused) == processed_path
    assert state.registration(processed_path)["tiles"] == tiles
    maximum = fusion.output_dir / "sample_max_z_fused.ome.zarr"
    regenerate_fused_max_projection(fused, maximum, max_workers=1)
    np.testing.assert_array_equal(
        open_group(maximum)["0"].to_tensorstore().read().result(),
        written.read().result().max(axis=2, keepdims=True),
    )


@pytest.mark.unit
@pytest.mark.parametrize("index", [0, 100_000])
@pytest.mark.parametrize("epsilon", [1e-10, 1e-5])
def test_outward_roi_rounding_has_an_absolute_pixel_tolerance(index, epsilon):
    """A large image index must not erase a real fractional-pixel intersection."""
    origin, pitch = -8000.0, 0.115
    bounds = origin + pitch * np.array([index + 10 - epsilon, index + 12 + epsilon])
    expected = (10, 12) if epsilon < 1e-7 else (9, 13)
    aligned = align_bounds_to_grid((*bounds, *bounds), (origin, origin), (pitch, pitch))
    expected = origin + pitch * (index + np.tile(expected, 2))
    np.testing.assert_allclose(aligned, expected, atol=1e-10, rtol=0)
