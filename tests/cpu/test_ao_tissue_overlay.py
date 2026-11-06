"""Check physical tissue scale and unchanged SJS pupil orientation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from scripts.ao_tissue_overlay import add_scale_bar, write_tissue_overlay

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("thumbnail_factor", (1, 4))
def test_scale_bar_uses_horizontal_physical_spacing_after_resizing(thumbnail_factor):
    """A 10-micrometer bar spans 50 native X pixels with anisotropic sampling."""
    # The original 100-by-200 tissue samples have dy=0.5 and dx=0.2 micrometers.
    # A thumbnail changes the number of samples without changing their extent.
    image = np.zeros((100 // thumbnail_factor, 200 // thumbnail_factor))
    extent = (-40, 10, 80, 120)
    axis = Figure().subplots()
    axis.imshow(image, origin="upper", extent=(80, 120, 10, -40))
    record = add_scale_bar(axis, extent, 10)
    endpoints = np.asarray(record["endpoints_yx_um"])
    assert record["length_um"] == 10
    assert record["label"] == "10 \u00b5m"
    assert endpoints[0, 0] == endpoints[1, 0]
    assert endpoints[1, 1] - endpoints[0, 1] == 10
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), endpoints[:, 1])
    horizontal_spacing = 40 / image.shape[1]
    assert (
        endpoints[1, 1] - endpoints[0, 1]
    ) / horizontal_spacing == 50 / thumbnail_factor


def test_tissue_overlay_preserves_pupil_orientation_centers_and_full_extent(
    tmp_path, monkeypatch
):
    """Asymmetric pupils keep SJS orientation over an inverted tissue Y axis."""
    pupil = np.ma.array(
        [[-np.pi / 4, np.pi / 2], [3 * np.pi / 4, 0]],
        mask=[[False, False], [False, True]],
    )
    regions = [
        SimpleNamespace(depth=depth, index=index, pupil=pupil.copy())
        for depth in range(4)
        for index in range(2)
    ]
    centers = {
        0: {0: (-35, 83)},
        1: {0: (-20, 95)},
        2: {0: (-10, 100)},
        3: {0: (0, 110)},
    }
    for positions in centers.values():
        y, x = positions[0]
        positions[1] = (y + 3, x - 3)
    geometry = SimpleNamespace(
        image=np.arange(25 * 50).reshape(25, 50),
        extent_yx_um=(-40, 10, 80, 120),
        centers_by_depth_index=centers,
        depth_offsets={depth: depth * 11.76 for depth in range(4)},
        provenance={"sampling": "fourfold thumbnail of anisotropic source"},
    )
    figures = []
    original_savefig = Figure.savefig

    def capture_figure(figure, *args, **kwargs):
        if figure not in figures:
            figures.append(figure)
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    record = write_tissue_overlay(
        regions, geometry, tmp_path, timepoint=0, pupil_diameter_um=20, scale_bar_um=10
    )
    assert "png" not in record and "svg" not in record
    assert len(record["pages"]) == len(figures) == 4
    assert len(list(tmp_path.glob("*.png"))) == 4
    assert len(list(tmp_path.glob("*.svg"))) == 4
    for depth, (figure, page) in enumerate(zip(figures, record["pages"], strict=True)):
        assert Path(page["png"]).name == f"ao_tissue_overlay_t000_z{depth:03d}.png"
        assert Path(page["svg"]).name == f"ao_tissue_overlay_t000_z{depth:03d}.svg"
        assert Path(page["png"]).is_file() and Path(page["svg"]).is_file()
        assert figure._suptitle is None
        tissue_axes = [axis for axis in figure.axes if axis.images]
        assert len(tissue_axes) == 1
        axis = tissue_axes[0]
        assert not axis.patches
        np.testing.assert_array_equal(axis.get_xlim(), (80, 120))
        np.testing.assert_array_equal(axis.get_ylim(), (10, -40))
        background, *displayed_pupils = axis.images
        assert background.origin == "upper"
        assert len(displayed_pupils) == len(axis.child_axes) == 2
        assert not axis.texts
        assert axis.get_title() == ""
        for index, (displayed_pupil, colorbar_axis) in enumerate(
            zip(displayed_pupils, axis.child_axes, strict=True)
        ):
            assert displayed_pupil.origin == "lower"
            assert type(displayed_pupil.norm) is Normalize
            np.testing.assert_allclose(
                displayed_pupil.norm(np.asarray((-2, -1, 0, 1, 2)) * np.pi),
                (0, 0.25, 0.5, 0.75, 1),
            )
            np.testing.assert_array_equal(displayed_pupil.get_array().data, pupil.data)
            np.testing.assert_array_equal(displayed_pupil.get_array().mask, pupil.mask)
            np.testing.assert_array_equal(
                regions[2 * depth + index].pupil.data, pupil.data
            )
            left, right, bottom, top = displayed_pupil.get_extent()
            y, x = centers[depth][index]
            assert (left + right) / 2 == x
            assert (bottom + top) / 2 == y
            assert right - left == 20
            assert abs(top - bottom) == 20
            # Test visible orientation through Matplotlib's actual transform.
            # SJS pupil row zero stays below row one despite inverted tissue Y.
            row_zero_y = bottom + 0.25 * (top - bottom)
            row_one_y = bottom + 0.75 * (top - bottom)
            assert (
                axis.transData.transform((x, row_zero_y))[1]
                < axis.transData.transform((x, row_one_y))[1]
            )
            assert not colorbar_axis.get_xticks().size
            assert not colorbar_axis.get_yticks().size
            assert colorbar_axis.get_xlabel() == colorbar_axis.get_ylabel() == ""
            assert not any(text.get_text() for text in colorbar_axis.texts)
            bar_bounds = colorbar_axis.get_window_extent()
            assert bar_bounds.height > bar_bounds.width
            assert bar_bounds.height < axis.get_window_extent().height
            assert page["regions"][index]["center_yx_um"] == [y, x]
        assert page["scale_bar"]["length_um"] == 10


def test_planned_grid_evenly_divides_the_full_acquisition_area(tmp_path, monkeypatch):
    """All nine corrections occupy cell centers across the full tissue canvas."""
    planned = {
        index: (y, x)
        for index, (y, x) in enumerate(
            (y, x) for x in (100, 300, 500) for y in (250, 150, 50)
        )
    }
    boundaries = {"horizontal_y_um": [100, 200], "vertical_x_um": [200, 400]}
    geometry = SimpleNamespace(
        image=np.arange(30 * 60).reshape(30, 60),
        extent_yx_um=(0, 300, 0, 600),
        centers_by_depth_index={0: {0: (210, 500), 8: (250, 550)}},
        planned_centers_by_depth_index={0: planned},
        planned_grid_by_depth_index={0: boundaries},
        depth_offsets={0: 0},
        provenance={},
    )
    regions = [
        SimpleNamespace(depth=0, index=index, pupil=np.ma.array([[0, 1], [2, 3]]))
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
        pupil_diameter_um=30,
        scale_bar_um=50,
        planned_grid=True,
    )
    axis = next(axis for axis in figures[0].axes if axis.images)
    assert len(axis.images) == 10  # Tissue plus all nine saved corrections.
    assert len(axis.child_axes) == 9
    assert not axis.texts and axis.get_title() == ""
    for index, image in enumerate(axis.images[1:]):
        left, right, bottom, top = image.get_extent()
        assert ((bottom + top) / 2, (left + right) / 2) == planned[index]
        row, column = 2 - index % 3, index // 3
        center_y, center_x = planned[index]
        assert 100 * row < center_y < 100 * (row + 1)
        assert 200 * column < center_x < 200 * (column + 1)
        assert center_y == 100 * row + 50
        assert center_x == 200 * column + 100
        np.testing.assert_array_equal(image.get_array().data, regions[index].pupil.data)
        assert record["pages"][0]["regions"][index]["center_yx_um"] == list(
            planned[index]
        )
    grid_lines = [line for line in axis.lines if line.get_gid() != "tissue-scale-bar"]
    assert len(grid_lines) == 4
    segments = {
        (tuple(line.get_xdata()), tuple(line.get_ydata())) for line in grid_lines
    }
    assert segments == {
        ((0, 600), (100, 100)),
        ((0, 600), (200, 200)),
        ((200, 200), (0, 300)),
        ((400, 400), (0, 300)),
    }
    assert record["pages"][0]["planned_grid"] == boundaries
    for colorbar_axis in axis.child_axes:
        assert not colorbar_axis.get_xticks().size
        assert not colorbar_axis.get_yticks().size
        assert colorbar_axis.get_xlabel() == colorbar_axis.get_ylabel() == ""


def test_each_depth_uses_its_local_projection_and_reports_its_roi(
    tmp_path, monkeypatch
):
    """Different layer structures must survive per-layer display normalization."""
    first = np.full((16, 24), 2.0)
    first[:, 3:6] = 12
    second = np.full((16, 24), 100.0)
    second[10:13] = 400
    rois = {0: {"z_bounds_um": [-20, -10]}, 1: {"z_bounds_um": [-40, -30]}}
    geometry = SimpleNamespace(
        image=np.empty((0, 0)),  # A shared projection must never be consulted.
        images_by_depth={0: first, 1: second},
        extent_yx_um=(0, 16, 0, 24),
        centers_by_depth_index={0: {0: (8, 12)}, 1: {0: (8, 12)}},
        depth_offsets={0: 0, 1: 11.76},
        layer_projection_rois_by_depth=rois,
        provenance={},
    )
    regions = [
        SimpleNamespace(depth=depth, index=0, pupil=np.ma.zeros((4, 4)))
        for depth in range(2)
    ]
    figures = []
    original_savefig = Figure.savefig

    def capture_figure(figure, *args, **kwargs):
        if figure not in figures:
            figures.append(figure)
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    record = write_tissue_overlay(
        regions, geometry, tmp_path, timepoint=0, pupil_diameter_um=2, scale_bar_um=5
    )
    assert len(figures) == len(record["pages"]) == 2
    for depth, (figure, expected, limits) in enumerate(
        zip(figures, (first == 12, second == 400), ([2, 12], [100, 400]), strict=True)
    ):
        axis = next(axis for axis in figure.axes if axis.images)
        np.testing.assert_array_equal(axis.images[0].get_array(), expected)
        assert not axis.texts and axis.get_title() == ""
        assert record["pages"][depth]["projection_roi"] == rois[depth]
        assert record["pages"][depth]["tissue_display_limits"] == limits

    geometry.images_by_depth = {0: first}
    with pytest.raises(ValueError, match="Missing local Z projection for layer 1"):
        write_tissue_overlay(
            regions[1:],
            geometry,
            tmp_path / "missing",
            timepoint=0,
            pupil_diameter_um=2,
            scale_bar_um=5,
        )
