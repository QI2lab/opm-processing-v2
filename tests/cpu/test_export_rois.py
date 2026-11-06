"""Physical GUI selections and raw-to-OME ROI processing integration."""

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from ome_types import from_xml
from typer.testing import CliRunner
from yaozarrs import open_group, v05

from opm_processing import export_rois
from opm_processing.dataio.position_collection import open_image_array
from opm_processing.dataio.processing_state import ProcessingState


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


@pytest.mark.integration
@pytest.mark.parametrize(
    "roi_preview",
    [
        ("mirror", 1, False),
        ("stage", 1, False),
        ("mirror", 2, False),
        ("stage", 2, False),
        ("stage", 2, True),
    ],
    indirect=True,
)
@pytest.mark.parametrize("tiff", [False, True])
def test_raw_roi_exports_and_resume(
    roi_preview,
    tiff,
    monkeypatch,
    relocate_acquisition,
    invoke_cli,
    assert_roi_point_accuracy,
):
    """Reconstruct multiple rectangles with all times/channels and global origins."""
    case = roi_preview
    directory = case.preview.parent
    preview_name = case.preview.name
    selections = [(16, 27, 3, 15), (24, 36, 16, 29)]
    calls = []
    checkpoints = []
    complete_channel = ProcessingState.complete_channel
    from opm_processing.imageprocessing.tilefusion import TileFusion

    register = TileFusion.register_with_retries

    def register_selected_channel(fusion):
        """Inspect the real registration object and execute its actual algorithm."""
        if fusion._variable_roi_tiles:
            assert fusion.channel_to_use == 1
        return register(fusion)

    monkeypatch.setattr(TileFusion, "register_with_retries", register_selected_channel)

    def observe_checkpoint(state, *args, **kwargs):
        """Record actual writes so relocation must retain completed channels."""
        checkpoints.append(args)
        return complete_channel(state, *args, **kwargs)

    monkeypatch.setattr(ProcessingState, "complete_channel", observe_checkpoint)

    def choose(source, shape):
        """Supply user interaction while exercising actual processing and storage."""
        calls.append(source)
        assert shape[:3] == (2, 2, 1)
        return selections

    monkeypatch.setattr(export_rois, "select_rois", choose)
    missing_channel = CliRunner().invoke(export_rois.app, [str(directory)])
    assert missing_channel.exit_code == 2
    assert not calls
    completed = None
    for attempt in range(2):
        # Cover fresh selection after a move and saved-selection/checkpoint
        # resume after a copy, even while the old acquisition remains available.
        if attempt == int(tiff):
            directory = relocate_acquisition(directory, copy=tiff)
            if attempt:
                # Independently invoked commands must find raw data from a
                # nested moved ROI journal before export_rois resumes it.
                roi_directory = directory / "roi_exports" / "roi_001"
                invoke_cli(
                    "fuse",
                    roi_directory,
                    roi=roi_directory / "roi.json",
                    registration_channel=1,
                    max_workers=1,
                )
                fused = (
                    open_image_array(roi_directory / "sample_fused.ome.zarr")
                    .read()
                    .result()
                )
                y0, y1, x0, x1 = selections[0]
                native_z = int(
                    np.ceil(
                        case.truth.pixels.shape[-2]
                        * np.sin(np.deg2rad(case.truth.microscope.angle_deg))
                    )
                )
                height, width = y1 - y0, x1 - x0
                # Refusion may pad its default multiscale grid. Padding is not
                # an acquired region in which to search for another emitter.
                np.testing.assert_array_equal(fused[..., native_z:, :, :], 0)
                np.testing.assert_array_equal(fused[..., :native_z, height:, :], 0)
                np.testing.assert_array_equal(fused[..., :native_z, :height, width:], 0)
                fused = fused[..., :native_z, :height, :width]
                metadata = (
                    open_group(roi_directory / "sample_fused.ome.zarr")
                    .ome_metadata()
                    .multiscales[0]
                    .datasets[0]
                )
                assert_roi_point_accuracy(
                    fused,
                    case.truth,
                    metadata.translation_transform.translation[-3:],
                    metadata.scale_transform.scale[-3:],
                    raw_origin=case.raw_origin_zyx_um,
                )
        invoke_cli(
            "export-rois",
            directory,
            registration_channel=1,
            save_float32=True,
            tiff=tiff,
        )
        assert len(calls) == 1
        if completed is not None:
            assert len(checkpoints) == completed
        completed = len(checkpoints)
        for index, (y0, y1, x0, x1) in enumerate(selections, 1):
            roi_directory = directory / "roi_exports" / f"roi_{index:03d}"
            image = roi_directory / "sample_fused.ome.zarr"
            root = open_group(image)
            assert isinstance(root.ome_metadata(), v05.Image)
            data = open_image_array(image).read().result()
            assert data.shape[:2] == (2, 2)
            assert data.shape[-2:] == (y1 - y0, x1 - x0)
            assert np.all(data.max(axis=(2, 3, 4)) > 0)
            dataset = root.ome_metadata().multiscales[0].datasets[0]
            origin = dataset.translation_transform.translation
            assert_roi_point_accuracy(
                data,
                case.truth,
                origin[-3:],
                dataset.scale_transform.scale[-3:],
                raw_origin=case.raw_origin_zyx_um,
            )
            pixel = case.dataset.metadata.pixel_size_um
            assert origin[-2:] == pytest.approx(
                (case.origin[1] + y0 * pixel, case.origin[2] + x0 * pixel)
            )
            state = ProcessingState.read(roi_directory / "sample.processing.json")
            assert state.source_path() == directory / case.dataset.path.name
            tiles = roi_directory / "sample_deskewed.ome.zarr"
            assert len(state.completed_channels(tiles)) == 2 * 2 * case.positions
            if case.positions > 1:
                assert state.registration(tiles)["pairwise_metrics"]
            target = roi_directory / f"sample_roi_{index:03d}.ome.tiff"
            assert target.is_file() == tiff
            if tiff:
                with tifffile.TiffFile(target) as tif:
                    np.testing.assert_array_equal(
                        tif.asarray().reshape(data.shape), data
                    )
                    assert tif.is_ome and tif.is_bigtiff
                    assert (
                        tif.pages[0].compression == tifffile.COMPRESSION.ADOBE_DEFLATE
                    )
                    ome = from_xml(tif.ome_metadata, validate=True).images[0]
                    assert (
                        ome.pixels.time_increment
                        == case.dataset.metadata.volume_interval_s
                    )
                    assert (
                        ome.stage_label.z,
                        ome.stage_label.y,
                        ome.stage_label.x,
                    ) == pytest.approx(origin[-3:])
    # The directory form of process_roi must resolve the relocated main journal
    # too; export_rois normally passes the already-resolved raw store directly.
    first_roi = directory / "roi_exports" / "roi_001"
    export_rois.process_roi(
        directory,
        first_roi / "roi.json",
        registration_channel=1,
        output=first_roi,
        save_float32=True,
        z_downsample_level=1,
    )
    assert len(checkpoints) == completed
    manifest = json.loads(
        (directory / "roi_exports" / "selections.json").read_text(encoding="utf-8")
    )
    assert len(manifest["rois"]) == 2
    assert calls[0].name == preview_name


@pytest.mark.integration
@pytest.mark.parametrize(
    "mosaic_acquisition", [("mirror", 0.4, 0.001), ("stage", 1.2, 0.001)], indirect=True
)
def test_roi_maps_extended_specimen_to_raw_tiles(
    mosaic_acquisition, monkeypatch, invoke_cli, pipeline_geometry
):
    """Keep the oblique preimage and measure fresh links across a real tile seam."""
    from opm_processing.dataio.position_collection import open_position_collection
    from opm_processing.imageprocessing.tilefusion import TileFusion
    from opm_processing.process import process_skewed
    from tests.simulation.camera_likelihood import (
        camera_sampling_accuracy,
        camera_sampling_envelope,
        sampling_stencil_footprint,
        sampling_stencil_radii,
    )
    from tests.simulation.metrics import camera_object_apertures, camera_volume_samples

    case = mosaic_acquisition
    process_skewed(
        case.dataset.path,
        acquisition=case.dataset.metadata,
        max_projection=True,
        create_fused_max_projection=True,
        save_float32=True,
        z_downsample_level=1,
    )
    preview = case.dataset.path.parent / "mosaic_max_z_fused.ome.zarr"
    microscope = case.microscope
    pixel = microscope.pixel_size_um
    scans, rows = case.expectation.shape[:2]
    angle = np.deg2rad(microscope.angle_deg)
    # The original fluorescent lines are centered at laboratory Y=0. Choose
    # their center from acquisition geometry, before inspecting processed pixels.
    cy = round(
        ((scans - 1) * microscope.scan_step_um + (rows - 1) * pixel * np.cos(angle))
        / (2 * pixel)
    )
    offset = case.parameters["tile_offset_x_pixels"]
    bounds = (cy - 16, cy + 16, offset - 8, case.parameters["camera_shape_yx"][1] + 8)
    monkeypatch.setattr(export_rois, "select_rois", lambda source, shape: [bounds])
    measurements = []
    register = TileFusion.register_and_score

    def measure(*args, **kwargs):
        """Observe fresh local image measurements without changing their scores."""
        result = register(*args, **kwargs)
        measurements.append(result)
        return result

    monkeypatch.setattr(TileFusion, "register_and_score", staticmethod(measure))
    invoke_cli("export-rois", preview, registration_channel=0, save_float32=True)
    directory = preview.parent / "roi_exports" / "roi_001"
    state = ProcessingState.read(directory / "mosaic.processing.json")
    tiles = directory / "mosaic_deskewed.ome.zarr"
    records = state.roi_series(tiles)
    assert len(records) == 2
    assert all(record["skewed_bounds_sx"][0] > 0 for record in records)
    assert measurements and all(np.isfinite(score) for _, score in measurements)
    # This sparse ROI may not meet the unchanged production SSIM gate. The
    # interaction contract requires new local measurements, not forced links.
    assert state.registration(tiles)["fused_path"] == "mosaic_fused.ome.zarr"
    saved = open_position_collection(tiles)
    assert all(np.any(array.read().result() > 0) for array in saved.arrays)
    final = open_image_array(directory / "mosaic_fused.ome.zarr").read().result()
    assert final.shape[-2:] == (bounds[1] - bounds[0], bounds[3] - bounds[2])
    assert np.isfinite(final).all() and np.any(final > 0)
    metadata = (
        open_group(directory / "mosaic_fused.ome.zarr")
        .ome_metadata()
        .multiscales[0]
        .datasets[0]
    )
    raw = case.dataset.raw_data[0, :, 0]
    stage = case.dataset.metadata.scan_axis_reversed
    if stage:
        raw = raw[:, ::-1]
    shape = raw.shape[-3:]
    width = shape[-1]
    means = [case.expectation[..., p * offset : p * offset + width] for p in range(2)]
    gain = 2 * pixel / microscope.scan_step_um
    geometry = pipeline_geometry(shape, microscope)
    native_support = camera_volume_samples(
        geometry.support.astype(float),
        (0, 0, 0),
        (pixel, pixel, pixel),
        (0, 0, 0),
        shape,
        microscope,
    )
    for fused in (False, True):
        spacing = np.asarray(
            metadata.scale_transform.scale[-3:] if fused else saved.voxel_size_um
        )
        # Fractional stage placement adds one further interpolation stencil.
        pitch = spacing * (2 if fused else 1)
        for position, mean in enumerate(means):
            image = (
                final[0, 0] if fused else saved.arrays[position][0, 0].read().result()
            )
            origin = (
                metadata.translation_transform.translation[-3:]
                if fused
                else saved.spatial_origins_zyx_um[position]
            )
            # These are the simulator's actual placements, before the prescribed
            # one-pixel X and 0.001-um scan-encoder errors in acquisition metadata.
            frame_origin = (
                -2.3,
                4.6 - ((scans - 1) * microscope.scan_step_um if stage else 0),
                1.15 + position * offset * pixel,
            )
            sample = (origin, spacing, frame_origin, shape, microscope)
            support = camera_volume_samples(np.ones(image.shape), *sample)
            support = (
                np.minimum(support, native_support) >= 1 - 32 * np.finfo(float).eps
            )
            prediction = camera_volume_samples(image / gain, *sample)
            envelope = camera_sampling_envelope(
                np.asarray(means) if fused else mean,
                raw if fused else raw[position],
                microscope.camera,
                radii=sampling_stencil_radii(microscope, pitch),
                footprint=sampling_stencil_footprint(
                    microscope, pitch, origin_y_error_um=0.0005
                ),
                views=2 if fused else 1,
                view_offsets_syx=np.outer((np.arange(2) - position) * offset, (0, 0, 1))
                if fused
                else None,
            )
            apertures = camera_object_apertures(
                case.optical_data.ground_truth,
                microscope,
                shape,
                camera_offset_zyx_um=(
                    0,
                    0,
                    (position * offset + (width - case.expectation.shape[-1]) / 2)
                    * pixel,
                ),
            )
            assert all(np.any(aperture & support) for aperture in apertures[:2])
            for aperture in apertures:
                selected = aperture & support
                if np.any(selected):
                    accuracy = camera_sampling_accuracy(
                        prediction, mean, envelope, selected
                    )
                    assert accuracy.accepted, accuracy


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
