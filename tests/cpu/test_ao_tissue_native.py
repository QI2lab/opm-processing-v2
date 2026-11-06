"""Native tissue exports preserve source pixels and physical pupil geometry."""

import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import matplotlib
import numpy as np
import pytest
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image
from scripts.ao_tissue_overlay import write_tissue_overlay

pytestmark = pytest.mark.unit


def test_native_canvas_retains_full_anisotropic_background_and_grid(
    tmp_path, monkeypatch
):
    """A 2100-pixel-wide source remains native in PNG and embedded SVG images."""
    centers = {
        index: (y, x)
        for index, (y, x) in enumerate(
            (y, x) for x in (70, 210, 350) for y in (25, 15, 5)
        )
    }
    geometry = SimpleNamespace(
        image=np.zeros((60, 2100), dtype=np.uint16),
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
        SimpleNamespace(depth=0, index=index, pupil=np.ma.zeros((8, 8)))
        for index in range(9)
    ]
    figures = []
    original_savefig = Figure.savefig

    def capture_figure(figure, *args, **kwargs):
        if figure not in figures:
            figures.append(figure)
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    record = write_tissue_overlay(
        regions,
        geometry,
        tmp_path,
        timepoint=0,
        pupil_diameter_um=4,
        scale_bar_um=10,
        planned_grid=True,
        full_resolution=True,
    )
    assert record["full_resolution"] is True
    assert record["native_pixel_shape_yx"] == [60, 2100]
    page = record["pages"][0]
    assert page["pixel_shape_yx"] == [60, 2100]
    with Image.open(page["png"]) as image:
        assert image.size == (2100, 60)
        pixels = np.asarray(image.convert("RGB"))
    embedded_shapes = []
    for element in ElementTree.parse(page["svg"]).iter():
        if not element.tag.endswith("image"):
            continue
        href = element.get(
            "{http://www.w3.org/1999/xlink}href", element.get("href", "")
        )
        if href.startswith("data:image/png;base64,"):
            with Image.open(
                BytesIO(base64.b64decode(href.partition(",")[2]))
            ) as embedded:
                embedded_shapes.append(embedded.size)
    assert (2100, 60) in embedded_shapes

    axis = next(axis for axis in figures[0].axes if axis.images)
    np.testing.assert_allclose(
        axis.images[0].get_window_extent().size, (2100, 60), atol=1e-9
    )
    expected_rgb = np.asarray(
        matplotlib.colormaps[matplotlib.rcParams["image.cmap"]](0.5, bytes=True)[:3]
    )
    for index, pupil_image in enumerate(axis.images[1:]):
        y, x = centers[index]
        row = round((y - 0.25) / 0.5)
        column = round((x - 0.1) / 0.2)
        np.testing.assert_allclose(pixels[row, column], expected_rgb, rtol=0, atol=1)
        # Four physical micrometers cover 20 X pixels and eight Y pixels.
        np.testing.assert_allclose(
            pupil_image.get_window_extent().size, (20, 8), atol=1e-9
        )
    assert record["grid_line_alpha"] == 0.3
    expected_grid_rgb = np.full(3, round(255 * 0.3))
    # Probe line centers away from pupils, other grid lines, and the scale bar.
    # Thirty-percent white over the known black tissue is approximately 76/255.
    for row in (20, 40):
        np.testing.assert_allclose(pixels[row, 10], expected_grid_rgb, rtol=0, atol=1)
        np.testing.assert_array_equal(pixels[row - 3, 10], (0, 0, 0))
    for column in (700, 1400):
        np.testing.assert_allclose(pixels[2, column], expected_grid_rgb, rtol=0, atol=1)
        np.testing.assert_array_equal(pixels[2, column - 3], (0, 0, 0))


def test_native_pupil_colors_keep_sjs_orientation_without_changing_phase(tmp_path):
    """Asymmetric phase quadrants retain their linear colors in the native PNG."""
    phase = np.repeat(
        np.repeat([[-np.pi, -np.pi / 2], [np.pi / 2, np.pi]], 4, axis=0), 4, axis=1
    )
    coordinate = np.linspace(-1, 1, 8)
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    mask = xx * xx + yy * yy > 1
    pupil = np.ma.array(phase.copy(), mask=mask.copy())
    geometry = SimpleNamespace(
        image=np.zeros((120, 240), dtype=np.uint16),
        shape_yx=(120, 240),
        preview_scale_yx_um=(1, 0.5),
        preview_origin_yx_um=(0.5, 0.25),
        extent_yx_um=(0, 120, 0, 120),
        centers_by_depth_index={0: {0: (60, 60)}},
        depth_offsets={0: 0},
        provenance={},
    )
    record = write_tissue_overlay(
        [SimpleNamespace(depth=0, index=0, pupil=pupil)],
        geometry,
        tmp_path,
        timepoint=0,
        pupil_diameter_um=40,
        scale_bar_um=10,
        planned_grid=False,
        full_resolution=True,
    )
    with Image.open(Path(record["pages"][0]["png"])) as image:
        assert image.size == (240, 120)
        pixels = np.asarray(image.convert("RGB"))
    normalize = Normalize(-2 * np.pi, 2 * np.pi)
    colormap = matplotlib.colormaps[matplotlib.rcParams["image.cmap"]]
    for row, column, phase_value in (
        (65, 110, -np.pi),
        (65, 130, -np.pi / 2),
        (55, 110, np.pi / 2),
        (55, 130, np.pi),
    ):
        expected_rgb = np.asarray(colormap(normalize(phase_value), bytes=True)[:3])
        np.testing.assert_allclose(pixels[row, column], expected_rgb, rtol=0, atol=1)
    np.testing.assert_array_equal(pupil.data, phase)
    np.testing.assert_array_equal(pupil.mask, mask)
