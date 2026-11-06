"""Validate NGFF tissue calibration and saved whole-tile AO assignments."""

import numpy as np
import pytest
from scripts.ao_tissue_geometry import (
    _compose_transform,
    _intended_geometry,
    read_tissue_geometry,
)

pytestmark = pytest.mark.unit


def test_projection_transform_composes_dataset_then_parent_in_order():
    """A parent scale also scales the dataset translation before adding its own."""
    scale, origin = _compose_transform(
        {
            "coordinateTransformations": [
                {"type": "scale", "scale": [1, 1, 1, 0.4, 0.8]},
                {"type": "translation", "translation": [0, 0, 0, 0.15, 0.3]},
            ]
        },
        {
            "coordinateTransformations": [
                {"type": "scale", "scale": [1, 1, 1, 2, 3]},
                {"type": "translation", "translation": [0, 0, 0, 20, 30]},
            ]
        },
    )
    np.testing.assert_allclose(scale, [1, 1, 1, 0.8, 2.4], rtol=0, atol=1e-12)
    np.testing.assert_allclose(origin, [0, 0, 0, 20.3, 30.9], rtol=0, atol=1e-12)


@pytest.mark.parametrize("full_resolution", [False, True])
def test_tissue_centers_follow_saved_assignments_and_selected_level_calibration(
    ao_geometry_case, full_resolution
):
    """Actual native and pyramid arrays retain physical registered tile centers."""
    case = ao_geometry_case
    pixels, native_pixels, regions = case.pixels, case.native, case.regions
    geometry = read_tissue_geometry(
        case.root,
        None,
        0,
        0,
        regions,
        planned_grid=True,
        full_resolution=full_resolution,
    )
    if full_resolution:
        assert geometry.image.shape == (2400, 3200)
        assert geometry.image[0, 0] == geometry.image[-1, -1] == 123
        np.testing.assert_array_equal(geometry.image, native_pixels)
    else:
        np.testing.assert_array_equal(geometry.image, pixels)
    assert geometry.shape_yx == (2400, 3200)
    assert geometry.origin_yx_um == (20, 30)
    assert geometry.preview_origin_yx_um == (
        (20, 30) if full_resolution else (20.3, 30.9)
    )
    np.testing.assert_allclose(
        geometry.preview_scale_yx_um,
        (0.2, 0.6) if full_resolution else (0.8, 2.4),
        rtol=0,
        atol=1e-12,
    )
    assert geometry.provenance["full_resolution"] is full_resolution
    assert geometry.provenance["native_shape_yx"] == [2400, 3200]
    assert geometry.provenance["preview_dataset"] == ("0" if full_resolution else "1")
    np.testing.assert_allclose(
        geometry.provenance["native_scale_yx_um"], (0.2, 0.6), rtol=0, atol=1e-12
    )
    assert geometry.extent_yx_um == (19.9, 499.9, 29.7, 1949.7)
    assert geometry.depth_offsets == {0: 0, 1: 11.76}
    for depth in range(2):
        centers = geometry.centers_by_depth_index[depth]
        assert set(centers) == {0, 2, 3, 5, 6, 8}
        assert centers[0] == (round(100 + 1.234 * depth, 3), 210)
        assert centers[8] == (round(150 + 1.234 * depth, 3), 250)
        assert geometry.provenance["depths"][depth]["assigned_tile_counts"] == {
            0: 3,
            1: 0,
            2: 3,
            3: 1,
            4: 0,
            5: 1,
            6: 3,
            7: 0,
            8: 3,
        }
        planned = geometry.planned_centers_by_depth_index[depth]
        assert set(planned) == set(range(9))
        assert planned[0] == (419.9, 349.7)
        assert planned[2] == (99.9, 349.7)
        assert planned[4] == (259.9, 989.7)
        assert planned[6] == (419.9, 1629.7)
        assert planned[8] == (99.9, 1629.7)
        assert geometry.planned_grid_by_depth_index[depth] == {
            "horizontal_y_um": [179.9, 339.9],
            "vertical_x_um": [669.7, 1309.7],
        }


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        (None, "recorded requested grid dimensions"),
        ((2, 3), "match the recorded requested"),
    ],
)
def test_intended_geometry_requires_recorded_dimensions_and_matching_corrections(
    shape, message
):
    """Never infer a requested grid or invent a correction for an empty cell."""
    with pytest.raises(ValueError, match=message):
        _intended_geometry(
            np.asarray([(x, y) for y in (15, 30, 45) for x in (35, 60, 85)]),
            list(range(9)),
            shape,
            (0, 90, 0, 90),
        )
