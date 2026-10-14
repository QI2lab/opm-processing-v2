"""Full-resolution ROI exports and multiscale selection coordinates."""

import json

import numpy as np
import pytest
import tifffile
from ome_types import from_xml
from typer.testing import CliRunner

from opm_processing import export_rois
from opm_processing.dataio.position_collection import create_position_collection
from opm_processing.dataio.processing_state import ProcessingState


@pytest.mark.integration
@pytest.mark.parametrize("suffix", ["_decon_deskewed", "_deskewed"])
@pytest.mark.parametrize("direct", [True, False])
def test_single_position_timelapse_exports(tmp_path, monkeypatch, suffix, direct):
    truth = np.arange(3 * 2 * 4 * 17 * 19, dtype=np.uint16).reshape(3, 2, 4, 17, 19)
    source = tmp_path / f"sample{suffix}.ome.zarr"
    spacing = (0.23, 0.115, 0.119)
    stage = (-40.5, 20.25, -11.75)
    collection = create_position_collection(
        source,
        (3, 1, 2, 4, 17, 19),
        spacing,
        stage_positions=[stage],
        channels=["488nm", "637nm"],
        time_interval_s=8.5,
        multiscale_factors_yx=(2,),
    )
    collection.arrays[0].write(truth).result()
    create_position_collection(
        tmp_path / f"sample_max_z{suffix}.ome.zarr",
        (3, 1, 2, 1, 17, 19),
        spacing,
    )
    companion = source / "OME" / "METADATA.ome.xml"
    template = from_xml(companion.read_text(encoding="utf-8"))
    template.images[0].pixels.channels[0].emission_wavelength = 525
    companion.write_text(template.to_xml(), encoding="utf-8")
    # The processed run has no registration/fusion record.
    state = ProcessingState.create(tmp_path / "sample.processing.json", source)
    state.initialize_run(source, configuration={}, overwrite=True)
    crops = [(1, 8, 2, 10), (3, 9, 4, 12)]

    def choose_rois(image_path, shape):
        assert image_path == source / "0"
        assert shape == truth.shape
        return crops

    monkeypatch.setattr(export_rois, "select_rois", choose_rois)
    result = CliRunner().invoke(export_rois.app, [str(source if direct else tmp_path)])
    assert result.exit_code == 0, result.exception
    for index, (y0, y1, x0, x1) in enumerate(crops, 1):
        target = tmp_path / "roi_exports" / f"sample_roi_{index:03d}.ome.tiff"
        expected = truth[:, :, :, y0:y1, x0:x1]
        with tifffile.TiffFile(target) as tif:
            np.testing.assert_array_equal(
                tif.asarray().reshape(expected.shape), expected
            )
            assert tif.is_bigtiff and tif.is_ome
            assert tif.pages[0].compression == tifffile.COMPRESSION.ADOBE_DEFLATE
            ome = from_xml(tif.ome_metadata, validate=True)
            image = ome.images[0]
            assert image.pixels.size_t == 3
            assert image.pixels.size_c == 2
            assert image.pixels.size_z == 4
            assert image.pixels.time_increment == 8.5
            assert [c.name for c in image.pixels.channels] == ["488nm", "637nm"]
            assert image.pixels.channels[0].emission_wavelength == 525
            assert image.stage_label.x == pytest.approx(stage[2] + x0 * spacing[2])
            assert image.stage_label.y == pytest.approx(stage[1] + y0 * spacing[1])
            for plane in image.pixels.planes:
                assert plane.position_x == image.stage_label.x
                assert plane.position_y == image.stage_label.y
                assert plane.position_z == pytest.approx(
                    stage[0] + plane.the_z * spacing[0]
                )
                assert plane.delta_t == pytest.approx(plane.the_t * 8.5)
            provenance = ome.structured_annotations.map_annotations[-1].value
            assert "ProcessingState" in provenance
            assert json.loads(provenance["ROIBoundsLevelZeroYX"]) == [y0, y1, x0, x1]


@pytest.mark.integration
def test_standalone_timelapse_exports(tmp_path, fused_store_factory, monkeypatch):
    truth = np.arange(3 * 2 * 4 * 17 * 19, dtype=np.uint16).reshape(3, 2, 4, 17, 19)
    source = fused_store_factory(
        truth,
        name="timelapse",
        spacing=(0.23, 0.115, 0.119),
        origin=(-40.5, 20.25, -11.75),
        time_step_s=8.5,
        time_origin_s=2.5,
        factors=(1, 2),
    )
    monkeypatch.setattr(
        export_rois, "select_rois", lambda source, shape: [(1, 8, 2, 10)]
    )
    result = CliRunner().invoke(export_rois.app, [str(source)])
    assert result.exit_code == 0, result.exception
    with tifffile.TiffFile(
        tmp_path / "roi_exports" / "timelapse_roi_001.ome.tiff"
    ) as tif:
        expected = truth[:, :, :, 1:8, 2:10]
        np.testing.assert_array_equal(tif.asarray().reshape(expected.shape), expected)
        pixels = from_xml(tif.ome_metadata, validate=True).images[0].pixels
        for plane in pixels.planes:
            assert plane.position_z == pytest.approx(-40.5 + plane.the_z * 0.23)
            assert plane.position_y == pytest.approx(20.25 + 0.115)
            assert plane.position_x == pytest.approx(-11.75 + 2 * 0.119)
            assert plane.delta_t == pytest.approx(2.5 + plane.the_t * 8.5)


@pytest.mark.integration
def test_standalone_collection_global_origin(tmp_path, monkeypatch):
    source = tmp_path / "sample_decon_deskewed.ome.zarr"
    origin = (-40.5, 20.25, -11.75)
    collection = create_position_collection(
        source,
        (2, 1, 1, 3, 5, 7),
        (0.23, 0.115, 0.119),
        stage_positions=[origin],
        spatial_offset_um=origin,
        channels=["488nm"],
        time_interval_s=2.5,
    )
    collection.arrays[0].write(np.ones((2, 1, 3, 5, 7), dtype=np.uint16)).result()
    monkeypatch.setattr(
        export_rois, "select_rois", lambda source, shape: [(1, 4, 2, 6)]
    )
    result = CliRunner().invoke(export_rois.app, [str(tmp_path)])
    assert result.exit_code == 0, result.exception
    with tifffile.TiffFile(tmp_path / "roi_exports" / "sample_roi_001.ome.tiff") as tif:
        ome = from_xml(tif.ome_metadata, validate=True)
        image = ome.images[0]
        assert image.stage_label.z == origin[0]
        assert image.stage_label.y == pytest.approx(origin[1] + 0.115)
        assert image.stage_label.x == pytest.approx(origin[2] + 2 * 0.119)
        assert image.pixels.channels[0].name == "488nm"
        assert image.pixels.time_increment == 2.5
        assert "SourceOME" in ome.structured_annotations.map_annotations[-1].value


@pytest.mark.integration
def test_multi_position_requires_fusion(tmp_path, monkeypatch):
    source = tmp_path / "sample_deskewed.ome.zarr"
    create_position_collection(source, (2, 2, 1, 3, 5, 7), (0.23, 0.115, 0.115))

    def unexpected_viewer(*args):
        pytest.fail(
            "A multi-position collection must be rejected before opening napari"
        )

    monkeypatch.setattr(export_rois, "select_rois", unexpected_viewer)
    result = CliRunner().invoke(export_rois.app, [str(tmp_path)])
    assert isinstance(result.exception, ValueError)
    assert "has 2 positions" in str(result.exception)


@pytest.mark.unit
@pytest.mark.parametrize("z_origin", [0.0, -2441.811])
def test_draw_rois_on_thick_projection(monkeypatch, z_origin):
    from pathlib import Path
    from types import SimpleNamespace

    import napari
    from napari.components import ViewerModel
    from napari.utils._proxies import ReadOnlyWrapper
    from napari.utils.interactions import (
        mouse_move_callbacks,
        mouse_press_callbacks,
        mouse_release_callbacks,
    )

    viewer = ViewerModel()
    data = np.zeros((2, 226, 32, 48), dtype=np.uint16)
    image = viewer.add_image(
        [data, data[:, ::2, ::4, ::4]],
        multiscale=True,
        scale=(1, 0.23, 0.115, 0.119),
        translate=(0, z_origin, -5330.254, -91.32),
    )
    monkeypatch.setattr(napari, "Viewer", lambda: viewer)
    monkeypatch.setattr(ViewerModel, "open", lambda *args, **kwargs: [image])

    def draw_rectangles():
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
                    SimpleNamespace(
                        type=kind,
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
    assert export_rois.select_rois(Path("sample_fused.ome.zarr"), data.shape) == [
        (3, 10, 4, 13),
        (12, 20, 20, 29),
    ]


@pytest.mark.integration
@pytest.mark.parametrize("source_xml", [True, False])
def test_multiple_roi_exports(tmp_path, fused_export, monkeypatch):
    crops = [(1, 8, 2, 10), (3, 9, 4, 12)]
    monkeypatch.setattr(export_rois, "select_rois", lambda source, shape: crops)
    result = CliRunner().invoke(export_rois.app, [str(tmp_path)])
    assert result.exit_code == 0, result.exception
    for i, (y0, y1, x0, x1) in enumerate(crops, 1):
        target = tmp_path / "roi_exports" / f"sample_roi_{i:03d}.ome.tiff"
        expected = fused_export.truth[:, :, :, y0:y1, x0:x1]
        with tifffile.TiffFile(target) as tif:
            np.testing.assert_array_equal(
                tif.asarray().reshape(expected.shape), expected
            )
            assert tif.is_bigtiff and tif.is_ome
            assert tif.pages[0].compression == tifffile.COMPRESSION.ADOBE_DEFLATE
            image = from_xml(tif.ome_metadata, validate=True).images[0]
            x = fused_export.origin[2] + x0 * fused_export.spacing[2]
            y = fused_export.origin[1] + y0 * fused_export.spacing[1]
            assert image.stage_label.x == pytest.approx(x)
            assert image.stage_label.y == pytest.approx(y)
            for plane in image.pixels.planes:
                assert plane.position_x == pytest.approx(x)
                assert plane.position_y == pytest.approx(y)
                assert plane.position_z == pytest.approx(
                    fused_export.origin[0] + plane.the_z * fused_export.spacing[0]
                )
    result = CliRunner().invoke(export_rois.app, [str(tmp_path)])
    assert isinstance(result.exception, FileExistsError)


@pytest.mark.unit
def test_multiscale_transform_uses_level_zero():
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
def test_thick_projection_preserves_time_and_multiscale_coordinates(z_size):
    from napari.components import ViewerModel

    rng = np.random.default_rng(42)
    data = rng.integers(0, 100, size=(2, z_size, 16, 24), dtype=np.uint16)
    data[0, 0, 1, 1] = 200
    data[0, -1, 2, 2] = 300
    data[1] += 1000
    pyramid = [data, data[:, ::2, ::2, ::2]]
    viewer = ViewerModel()
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
    assert export_rois.rectangle_bounds(
        np.array([[-2, -3], [-2, 8], [10, 8], [10, -3]]), (6, 7)
    ) == (0, 6, 0, 7)
    with pytest.raises(ValueError):
        export_rois.rectangle_bounds(
            np.array([[10, 10], [10, 12], [12, 12], [12, 10]]), (6, 7)
        )
