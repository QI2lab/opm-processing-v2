"""Known display objects and real production SVGs for AO rendering units."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scripts.ao_tissue_overlay import write_tissue_overlay


@dataclass
class OverlayCase:
    """Production geometry and region records with prescribed visual landmarks."""

    geometry: Any
    regions: list
    pupil: Any = None


@pytest.fixture
def ao_overlay_case(ao_geometry_factory, ao_region_factory):
    """Provide asymmetric pupils at known physical centers in four layers."""
    pupil = np.ma.array(
        [[-np.pi / 4, np.pi / 2], [3 * np.pi / 4, 0]],
        mask=[[False, False], [False, True]],
    )
    centers = {
        depth: {0: point, 1: (point[0] + 3, point[1] - 3)}
        for depth, point in enumerate(((-35, 83), (-20, 95), (-10, 100), (0, 110)))
    }
    geometry = ao_geometry_factory(
        np.arange(25 * 50).reshape(25, 50),
        extent_yx_um=(-40, 10, 80, 120),
        centers_by_depth_index=centers,
        depth_offsets={depth: depth * 11.76 for depth in range(4)},
        provenance={"sampling": "fourfold thumbnail of anisotropic source"},
    )
    regions = [
        ao_region_factory(depth=depth, index=index, pupil=pupil.copy())
        for depth in range(4)
        for index in range(2)
    ]
    return OverlayCase(geometry, regions, pupil)


@pytest.fixture
def ao_planned_overlay_case(ao_geometry_factory, ao_region_factory):
    """Provide nine correction centers on a calibrated rectangular grid."""
    planned = {
        index: (y, x)
        for index, (y, x) in enumerate(
            (y, x) for x in (100, 300, 500) for y in (250, 150, 50)
        )
    }
    geometry = ao_geometry_factory(
        np.arange(30 * 60).reshape(30, 60),
        extent_yx_um=(0, 300, 0, 600),
        centers_by_depth_index={0: {0: (210, 500), 8: (250, 550)}},
        planned_centers_by_depth_index={0: planned},
        planned_grid_by_depth_index={
            0: {"horizontal_y_um": [100, 200], "vertical_x_um": [200, 400]}
        },
        depth_offsets={0: 0},
    )
    return OverlayCase(
        geometry,
        [
            ao_region_factory(index=index, pupil=np.ma.array([[0, 1], [2, 3]]))
            for index in range(9)
        ],
    )


@pytest.fixture
def ao_local_overlay_case(ao_geometry_factory, ao_region_factory):
    """Provide distinct known structures and intensity ranges for two layers."""
    first = np.full((16, 24), 2.0)
    first[:, 3:6] = 12
    second = np.full((16, 24), 100.0)
    second[10:13] = 400
    geometry = ao_geometry_factory(
        np.empty((0, 0)),
        images_by_depth={0: first, 1: second},
        extent_yx_um=(0, 16, 0, 24),
        centers_by_depth_index={0: {0: (8, 12)}, 1: {0: (8, 12)}},
        depth_offsets={0: 0, 1: 11.76},
        layer_projection_rois_by_depth={
            0: {"z_bounds_um": [-20, -10]},
            1: {"z_bounds_um": [-40, -30]},
        },
    )
    return OverlayCase(
        geometry,
        [
            ao_region_factory(depth=depth, pupil=np.ma.zeros((4, 4)))
            for depth in range(2)
        ],
    )


@pytest.fixture
def ao_partial_preview(ao_memmap_source):
    """Write samples whose four partial-bin display averages are analytical."""
    pixels = np.zeros((5, 7), np.uint16)
    pixels[0, 0] = 1000
    pixels[:4, 4:] = 200
    pixels[4, :4] = 300
    pixels[4, 4:] = (0, 200, 1000)
    return (
        ao_memmap_source(pixels),
        ao_memmap_source(np.full((5, 7), 1000, np.uint16)),
    )


@pytest.fixture(params=[False, True], ids=["gray", "rgb"])
def ao_checkerboard_preview(request, ao_memmap_source):
    """Write known checkerboards crossing source blocks and partial edge bins."""
    rows, columns = np.ogrid[:1031, :1043]
    checkerboard = (rows + columns) % 2
    if request.param:
        colors = np.asarray(((10, 30, 90), (250, 130, 170)), dtype=np.uint8)
        pixels = colors[checkerboard]
        expected = np.empty((516, 522, 3), np.uint8)
        expected[:] = (130, 80, 130)
        expected[-1, -1] = (10, 30, 90)
        limits = {}
    else:
        pixels = (checkerboard * 1000).astype(np.uint16)
        expected = np.full((516, 522, 3), 128, np.uint8)
        expected[-1, -1] = 0
        limits = {0: (0, 1000)}
    return ao_memmap_source(pixels), limits, expected


@dataclass
class SavedOverlay:
    """A production SVG and its independently prescribed RGB object."""

    path: Path
    pixels: np.ndarray
    page: dict


@pytest.fixture
def saved_overlay_svg(tmp_path, ao_geometry_factory, ao_region_factory):
    """Render known RGB samples through the actual pupil and SVG writers."""
    height, width = 192, 256
    y, x = np.indices((height, width))
    pixels = np.stack((x * 3, y * 5, (x * 2 + y) % 256), axis=-1).astype(np.uint8)
    centers = {
        index: (32 + 64 * row, 40 + 84 * column)
        for index, (row, column) in enumerate(np.ndindex(3, 3))
    }
    geometry = ao_geometry_factory(
        pixels,
        extent_yx_um=(0, height, 0, width),
        centers_by_depth_index={0: centers},
        depth_offsets={0: 0},
    )
    regions = [
        ao_region_factory(index=index, pupil=np.ma.array([[0.0, 1.0], [2.0, -1.0]]))
        for index in centers
    ]
    page = write_tissue_overlay(
        regions,
        geometry,
        tmp_path,
        timepoint=0,
        pupil_diameter_um=32,
        scale_bar_um=40,
        output_shape_yx=(height, width),
    )["pages"][0]
    return SavedOverlay(tmp_path / "ao_tissue_overlay_t000_z000.svg", pixels, page)


@pytest.fixture
def ao_quadrant_pupil():
    """Give four known signed phase quadrants inside a circular aperture."""
    phase = np.repeat(
        np.repeat([[-np.pi, -np.pi / 2], [np.pi / 2, np.pi]], 4, axis=0), 4, axis=1
    )
    coordinate = np.linspace(-1, 1, 8)
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    return np.ma.array(phase, mask=xx * xx + yy * yy > 1)


@pytest.fixture
def ao_rgb_field():
    """Construct the shared known RGB concentration ramp at requested dimensions."""

    def create(shape):
        """Evaluate three prescribed spatial frequencies without rendering pixels."""
        yy, xx = np.indices(shape)
        return np.stack(
            (
                (7 * xx + 3 * yy) % 256,
                (5 * xx + 11 * yy + 17) % 256,
                (13 * xx + 2 * yy + 41) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)

    return create


@pytest.fixture
def ao_native_canvas_case(ao_geometry_factory, ao_region_factory):
    """Share native tissue pixels, physical landmarks and independently known pupils."""
    centers = {
        index: (y, x)
        for index, (y, x) in enumerate(
            (y, x) for x in (70, 210, 350) for y in (25, 15, 5)
        )
    }
    geometry = ao_geometry_factory(
        np.zeros((60, 2100), dtype=np.uint16),
        shape_yx=(60, 2100),
        preview_scale_yx_um=(0.5, 0.2),
        preview_origin_yx_um=(0.25, 0.1),
        extent_yx_um=(0, 30, 0, 420),
        planned_centers_by_depth_index={0: centers},
        planned_grid_by_depth_index={
            0: {"horizontal_y_um": [10, 20], "vertical_x_um": [140, 280]}
        },
        depth_offsets={0: 0},
        provenance={},
    )
    regions = [
        ao_region_factory(depth=0, index=index, pupil=np.ma.zeros((8, 8)))
        for index in range(9)
    ]
    return SimpleNamespace(
        centers=centers,
        geometry=geometry,
        regions=regions,
    )


@pytest.fixture
def ao_native_orientation_case(
    ao_geometry_factory, ao_region_factory, ao_quadrant_pupil
):
    """Share native tissue pixels, physical landmarks and independently known pupils."""
    pupil = ao_quadrant_pupil.copy()
    phase, mask = pupil.data.copy(), pupil.mask.copy()
    geometry = ao_geometry_factory(
        np.zeros((120, 240), dtype=np.uint16),
        shape_yx=(120, 240),
        preview_scale_yx_um=(1, 0.5),
        preview_origin_yx_um=(0.5, 0.25),
        extent_yx_um=(0, 120, 0, 120),
        centers_by_depth_index={0: {0: (60, 60)}},
        depth_offsets={0: 0},
        provenance={},
    )
    regions = [ao_region_factory(depth=0, index=0, pupil=pupil)]
    return SimpleNamespace(
        phase=phase,
        mask=mask,
        pupil=pupil,
        geometry=geometry,
        regions=regions,
    )
