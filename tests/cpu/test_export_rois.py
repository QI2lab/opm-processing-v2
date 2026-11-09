"""Physical GUI selections and raw-to-OME ROI processing integration."""

from pathlib import Path

import numpy as np
import pytest

from opm_processing import export_rois


@pytest.mark.unit
@pytest.mark.parametrize("windows_paths", [False, True])
@pytest.mark.parametrize(
    ("raw_name", "preview_name", "matches"),
    [
        ("sample.ome.zarr", "sample_max_z_fused.ome.zarr", True),
        ("other.ome.zarr", "sample_max_z_fused.ome.zarr", False),
        ("sample.ome.zarr", "other_max_z_fused.ome.zarr", False),
    ],
)
def test_saved_selection_relocation_requires_same_stores(
    tmp_path, windows_paths, raw_name, preview_name, matches
):
    """Allow root relocation while rejecting a different raw store or preview."""
    from pathlib import PureWindowsPath

    old = PureWindowsPath(r"F:\old") if windows_paths else tmp_path / "old"
    manifest = {
        "source_path": str(old / raw_name),
        "preview_path": str(old / preview_name),
    }
    assert (
        export_rois.selection_paths_match(
            manifest,
            tmp_path / "current" / "sample.ome.zarr",
            tmp_path / "current" / "sample_max_z_fused.ome.zarr",
        )
        == matches
    )


@pytest.mark.unit
def test_selections_with_separate_accessible_raw_directory(tmp_path):
    """Moving previews alone can retain an unchanged external raw-data source."""
    raw = tmp_path / "raw" / "sample.ome.zarr"
    manifest = {
        "source_path": str(raw),
        "preview_path": str(tmp_path / "old" / "sample_max_z_fused.ome.zarr"),
    }
    preview = tmp_path / "current" / "sample_max_z_fused.ome.zarr"
    assert export_rois.selection_paths_match(manifest, raw, preview)
    assert not export_rois.selection_paths_match(
        manifest, Path(tmp_path / "different" / raw.name), preview
    )


@pytest.mark.unit
@pytest.mark.parametrize("origin", [(-2.3, 4.6, 1.15), (-2.3, 1.8, 1.15)])
def test_single_position_gui_keeps_time_and_global_coordinates(
    origin, fused_store_factory, napari_model, monkeypatch
):
    """Draw rectangles on plugin-loaded timelapse layers in stage space."""
    import napari

    data = np.arange(2 * 2 * 32 * 16, dtype=np.uint16).reshape(2, 2, 1, 32, 16)
    source = fused_store_factory(data, spacing=(0.115,) * 3, origin=origin)
    viewer = napari_model
    monkeypatch.setattr(napari, "Viewer", lambda: viewer)

    def draw():
        """Add a rectangle after inspecting the second acquired timepoint."""
        viewer.dims.set_current_step(0, 1)
        layer = viewer.layers[0]
        np.testing.assert_allclose(layer.translate[-2:], origin[-2:])
        vertices = np.asarray(((3, 2), (3, 10), (12, 10), (12, 2)))
        viewer.layers.selection.active.add(vertices, shape_type="rectangle")
        assert viewer.dims.current_step[0] == 1

    monkeypatch.setattr(napari, "run", draw)
    assert export_rois.select_rois(source, data.shape) == [(3, 12, 2, 10)]


@pytest.mark.unit
@pytest.mark.parametrize("z_origin", [0.0, -2441.811])
def test_draw_rois_on_thick_projection(
    monkeypatch, z_origin, napari_model, fused_store_factory
):
    """Draw level-zero rectangles on a physical full-depth maximum projection."""
    import napari
    from napari.utils._proxies import ReadOnlyWrapper
    from napari.utils.events import Event
    from napari.utils.interactions import (
        mouse_move_callbacks,
        mouse_press_callbacks,
        mouse_release_callbacks,
    )

    viewer = napari_model
    time, z, row, column = np.indices((2, 226, 32, 48), dtype=np.uint16)
    data = 100 + time * 20000 + z * 100 + row * 3 + column
    source = fused_store_factory(
        data[:, None],
        spacing=(0.23, 0.115, 0.115),
        origin=(z_origin, -5330.254, -91.32),
        factors=(1, 2),
    )
    monkeypatch.setattr(napari, "Viewer", lambda: viewer)

    def draw_rectangles():
        """Deliver physical mouse events to the real rectangle-drawing callbacks."""
        image = viewer.layers[0]
        rois = viewer.layers.selection.active
        rois.scale_factor = 0.001
        for time, level, corners in (
            (0, 1, ((3.2, 4.1), (9.8, 12.6))),
            (1, 0, ((12.2, 20.1), (19.8, 28.6))),
        ):
            image.locked_data_level = level
            viewer.dims.set_current_step(0, time)
            if time:
                viewer.dims.set_point(1, viewer.dims.point[1] + 0.17)
                viewer.dims.thickness = (0, 2.3, 0, 0)
            rois._slice_dims(viewer.dims)
            world = [
                (*viewer.dims.point[:2], *image.data_to_world((0, 0, *xy))[-2:])
                for xy in corners
            ]
            for kind, position, callback in (
                ("mouse_press", world[0], mouse_press_callbacks),
                ("mouse_move", world[1], mouse_move_callbacks),
                ("mouse_release", world[1], mouse_release_callbacks),
            ):
                event = ReadOnlyWrapper(
                    Event(
                        kind,
                        position=position,
                        is_dragging=kind == "mouse_move",
                        button=1,
                        modifiers=(),
                        pos=np.zeros(2),
                        dims_displayed=(2, 3),
                    )
                )
                callback(rois, event)
        viewer.dims.set_current_step(0, 0)
        viewer.dims.set_current_step(1, 0)
        rois._slice_dims(viewer.dims)
        assert rois._data_view._displayed.all()

    monkeypatch.setattr(napari, "run", draw_rectangles)
    assert export_rois.select_rois(source, data.shape) == [
        (3, 10, 4, 13),
        (12, 20, 20, 29),
    ]


@pytest.mark.unit
def test_multiscale_transform_uses_level_zero(napari_model):
    """Map world coordinates to level zero independently of the displayed scale."""
    from napari.layers import Image

    layer = Image(
        [np.zeros((32, 48)), np.zeros((8, 12))],
        multiscale=True,
        scale=(0.115, 0.119),
        translate=(20.25, -11.75),
    )
    pixels = np.array([[3.2, 4.1], [3.2, 12.6], [9.8, 12.6], [9.8, 4.1]])
    world = [layer.data_to_world(vertex) for vertex in pixels]
    for level in (0, 1):
        layer.data_level = level
        recovered = np.array([layer.world_to_data(vertex) for vertex in world])
        np.testing.assert_allclose(recovered, pixels)
        assert export_rois.rectangle_bounds(recovered, (32, 48)) == (3, 10, 4, 13)


@pytest.mark.unit
@pytest.mark.parametrize("z_size", [5, 6])
def test_thick_projection_preserves_time_and_multiscale_coordinates(
    z_size, napari_model
):
    """Retain time selection and physical coordinates across projected scales."""
    data = np.arange(2 * z_size * 16 * 24, dtype=np.uint16).reshape(2, z_size, 16, 24)
    pyramid = [data, data[:, ::2, ::2, ::2]]
    viewer = napari_model
    layers = [
        viewer.add_image(
            [level + channel * 2000 for level in pyramid],
            multiscale=True,
            scale=(1, 0.23, 0.115, 0.119),
            translate=(0, -2447.283, -5330.16, -91.32),
        )
        for channel in range(3)
    ]
    export_rois.configure_roi_projection(viewer, layers)
    pixels = np.array([[3.2, 4.1], [3.2, 12.6], [9.8, 12.6], [9.8, 4.1]])
    for time in range(2):
        viewer.dims.set_current_step(0, time)
        for layer in layers:
            for level, array in enumerate(layer.data):
                layer.locked_data_level = level
                layer.corner_pixels = np.array(
                    [np.zeros(4, dtype=int), np.array(array.shape) - 1]
                )
                layer._slice_dims(viewer.dims)
                np.testing.assert_array_equal(
                    layer._slice.image.raw, np.max(array[time], axis=0)
                )
                vertices = np.column_stack(
                    [np.full(4, time), np.full(4, z_size // 2), pixels]
                )
                world = [layer.data_to_world(vertex) for vertex in vertices]
                recovered = np.array([layer.world_to_data(v) for v in world])
                assert export_rois.rectangle_bounds(recovered, data.shape) == (
                    3,
                    10,
                    4,
                    13,
                )


@pytest.mark.unit
def test_bounds_clip_and_reject_empty():
    """Clip intersecting rectangles and reject selections outside the image."""
    assert export_rois.rectangle_bounds(
        np.array([[-2, -3], [-2, 8], [10, 8], [10, -3]]), (6, 7)
    ) == (0, 6, 0, 7)
    with pytest.raises(ValueError):
        export_rois.rectangle_bounds(
            np.array([[10, 10], [10, 12], [12, 12], [12, 10]]), (6, 7)
        )
