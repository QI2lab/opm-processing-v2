"""Keep stage-anchored preview footprints invertible to retained raw scans."""

from __future__ import annotations

import json

import numpy as np
import pytest

from opm_processing.dataio.position_collection import create_position_collection
from opm_processing.dataio.processing_state import ProcessingState
from opm_processing.dataio.roi import (
    PhysicalRoi,
    align_bounds_to_grid,
    projection_tile_footprints,
)
from opm_processing.imageprocessing.maxtilefusion import MaxTileFusion


@pytest.mark.unit
def test_registered_stage_roi_keeps_endpoint_and_removes_only_crop(
    stage_coordinate_case, record_registration, fused_store_factory
):
    """Registered stage placement preserves the retained raw scan's endpoint."""
    case = stage_coordinate_case
    processed = case.directory / "sample_deskewed.ome.zarr"
    create_position_collection(
        processed,
        (1, 2, 1, 4, 12, 5),
        (1, 1, 1),
        stage_positions=case.positions,
        spatial_offset_um=(0, -7, 2),
        attributes={"opm_processing_coordinates": case.marker},
    )
    preview = fused_store_factory(
        np.zeros((1, 1, 1, 12, 5), np.uint16),
        name="sample_max_z_fused",
        attributes={"opm_processing_coordinates": case.marker},
    )
    record_registration(
        processed,
        [
            {"time_index": 0, "position_index": 0, "origin_zyx_um": [-3, -107, 22]},
            {"time_index": 0, "position_index": 1, "origin_zyx_um": [-3, -157, 22]},
        ],
    )

    tiles = projection_tile_footprints(preview)

    assert tiles[0]["origin_zyx_um"] == [-3, -107, 22]
    assert tiles[0]["raw_origin_zyx_um"] == [-3, -110, 20]
    assert tiles[1]["raw_origin_zyx_um"] == [-3, -160, 20]


@pytest.mark.unit
@pytest.mark.parametrize("selected_positions", [(0, 1), (1,)])
def test_stage_preview_footprints_match_fused_pixel_origins(
    stage_coordinate_case, selected_positions
):
    """A stage preview retains both the endpoint and post-deskew XY crop."""
    case = stage_coordinate_case
    full_path = case.directory / "sample_deskewed.ome.zarr"
    create_position_collection(
        full_path,
        (1, 2, 1, 4, 12, 5),
        (1, 1, 1),
        stage_positions=case.positions,
        spatial_offset_um=(0, -7, 2),
        attributes={"opm_processing_coordinates": case.marker},
    )
    maximum = create_position_collection(
        case.directory / "sample_max_z_deskewed.ome.zarr",
        (1, 2, 1, 1, 12, 5),
        (1, 1, 1),
        stage_positions=case.positions,
        spatial_offset_um=(1.5, -7, 2),
        attributes={"opm_processing_coordinates": case.marker},
    )
    preview = case.directory / "sample_max_z_fused.ome.zarr"
    fusion = MaxTileFusion(
        [maximum.arrays[position] for position in selected_positions],
        [case.positions[position] for position in selected_positions],
        preview,
        (1, 1, 1),
        spatial_offsets_zyx_um=[
            maximum.spatial_origins_zyx_um[position] for position in selected_positions
        ],
        source_collection=maximum.path,
        source_position_indices=selected_positions,
    )
    state = ProcessingState.create(
        case.directory / "sample.processing.json", case.dataset.path
    )
    state.initialize_run(full_path, configuration={}, overwrite=True)

    tiles = projection_tile_footprints(preview)

    np.testing.assert_array_equal(
        np.array([tile["origin_zyx_um"][-2:] for tile in tiles]), fusion.tile_positions
    )
    assert tuple(tile["position_index"] for tile in tiles) == selected_positions
    if selected_positions == (0, 1):
        assert tiles[0]["bounds_yx_um"] == [-107, -95, 22, 27]
        assert tiles[0]["raw_origin_zyx_um"] == [-3, -110, 20]
    else:
        assert tiles[0]["bounds_yx_um"] == [-157, -145, 22, 27]
        assert tiles[0]["raw_origin_zyx_um"] == [-3, -160, 20]


@pytest.mark.unit
def test_roi_grid_retains_three_decimal_spatial_values(tmp_path):
    """ROI metadata follows the requested three-decimal spatial precision."""
    origin = (8192.123456789, -213.123456789)
    spacing = (0.1154, 0.11537)
    bounds = (
        origin[0] + 12000.25 * spacing[0],
        origin[0] + 12002.25 * spacing[0],
        origin[1] + 3.25 * spacing[1],
        origin[1] + 5.25 * spacing[1],
    )

    aligned = align_bounds_to_grid(bounds, origin, spacing)

    assert aligned == tuple(
        round(value, 3)
        for value in (
            origin[0] + 12000 * spacing[0],
            origin[0] + 12003 * spacing[0],
            origin[1] + 3 * spacing[1],
            origin[1] + 6 * spacing[1],
        )
    )
    roi = PhysicalRoi(
        bounds_yx_um=bounds,
        source_path=tmp_path / "preview.ome.zarr",
        grid_origin_yx_um=origin,
        pixel_size_yx_um=spacing,
        tile_footprints=({"origin_zyx_um": [1.234567, *origin]},),
    )
    document = json.loads(roi.write(tmp_path / "selection.json").read_text())
    assert document["source_grid"] == {
        "origin_yx_um": [8192.123, -213.123],
        "pixel_size_yx_um": [0.115, 0.115],
    }
    assert document["tile_footprints"][0]["origin_zyx_um"] == [
        1.235,
        8192.123,
        -213.123,
    ]
