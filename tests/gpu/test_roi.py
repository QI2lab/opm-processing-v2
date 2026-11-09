"""GPU registration of physical tile overlaps and stored acquisition workflows."""

from __future__ import annotations

import numpy as np
import pytest

from opm_processing.dataio.position_collection import (
    create_variable_position_collection,
    open_position_collection,
)
from opm_processing.dataio.roi import (
    PhysicalRoi,
)
from opm_processing.imageprocessing.tilefusion import TileFusion

pytestmark = [pytest.mark.usefixtures("cupy_gpu")]


@pytest.mark.unit
def test_variable_roi_tiles_are_stored_cropped_and_reopened(
    record_registration,
    tmp_path,
    acquisition_factory,
) -> None:
    """ROI series retain their independent crop shapes and physical origins."""
    processed_path = tmp_path / "sample_decon_deskewed.ome.zarr"
    shapes = ((1, 1, 2, 4, 4), (1, 1, 2, 5, 3))
    records = [
        {
            "time_index": 0,
            "position_index": 7,
            "origin_zyx_um": [0.0, 10.0, 20.0],
            "shape_tczyx": list(shapes[0]),
            "skewed_shape_syx": [16, 4, 4],
            "deskew_crop_yx": [4, 8, 0, 4],
        },
        {
            "time_index": 0,
            "position_index": 8,
            "origin_zyx_um": [0.0, 12.0, 22.0],
            "shape_tczyx": list(shapes[1]),
            "skewed_shape_syx": [16, 4, 3],
            "deskew_crop_yx": [6, 11, 0, 3],
        },
    ]
    collection = create_variable_position_collection(
        processed_path,
        shapes,
        (1.0, 1.0, 1.0),
        spatial_origins_zyx_um=(
            (0.0, 10.0, 20.0),
            (0.0, 12.0, 22.0),
        ),
    )
    observed = tuple(
        np.arange(np.prod(shape[-3:]), dtype=np.uint16).reshape(shape[-3:])
        + 100 * (index + 1)
        for index, shape in enumerate(shapes)
    )
    for array, pixels in zip(collection.arrays, observed, strict=True):
        array.write(pixels[None, None]).result()
    # Z=0 has no valid deskew interpolation; Z=1 is supported throughout
    # both interior Y crops. Store physically valid synthetic tile data.
    for array in collection.arrays:
        array[:, :, 0].write(np.uint16(0)).result()

    reopened = open_position_collection(processed_path)
    assert [tuple(array.shape) for array in reopened.arrays] == list(shapes)
    assert not any(key.startswith("opm_") for key in reopened.attributes)

    roi_series = tuple(
        {
            key: record[key]
            for key in (
                "time_index",
                "position_index",
                "skewed_shape_syx",
                "deskew_crop_yx",
            )
        }
        for record in records
    )
    acquisition_factory(
        np.ones((1, 9, 1, 16, 4, 4), np.uint16),
        pixel_size_um=1.0,
        scan_axis_step_um=1.0,
        camera_offset=0.0,
        camera_conversion=1.0,
        stage_positions_zxy=((0.0, 0.0, 0.0),) * 9,
    )
    record_registration(
        processed_path,
        records,
        roi_series=roi_series,
    )
    roi = PhysicalRoi(
        bounds_yx_um=(10.0, 17.0, 20.0, 25.0),
        source_path=tmp_path / "sample_max_z_fused.ome.zarr",
        grid_origin_yx_um=(10.0, 20.0),
        pixel_size_yx_um=(1.0, 1.0),
        position_indices=(7, 8),
    )
    fusion = TileFusion(
        processed_path,
        roi_selection=roi,
        max_workers=1,
        multiscale_factors=(2,),
        chunk_shape_yx=(4, 4),
        blend_pixels=(0, 0, 0),
    )
    assert fusion._variable_roi_tiles is True
    assert fusion._reuse_registered_roi_placements is False
    assert fusion._tile_shapes == [(2, 4, 4), (2, 5, 3)]
    assert fusion._tiles_by_time == ((0, 1),)
    fusion._compute_fused_image_space()
    assert fusion.offset_um == (0.0, 10.0, 20.0)
    assert fusion.unpadded_shape == (2, 7, 5)

    fusion.run()
    fused = np.asarray(fusion.fused_ts.read().result())
    assert fused.shape == (1, 1, 2, 8, 6)
    expected = np.zeros((1, 1, 2, 8, 6), np.uint16)
    expected[..., :4, :4] = observed[0]
    expected[..., 2:7, 2:5] = observed[1]
    expected[..., 2:4, 2:4] = (
        (observed[0][:, 2:4, 2:4].astype(np.float32) + observed[1][:, :2, :2]) / 2
    ).astype(np.uint16)
    expected[:, :, 0] = 0
    np.testing.assert_array_equal(fused, expected)
