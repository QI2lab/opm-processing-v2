"""GPU registration of physical tile overlaps and stored acquisition workflows."""

import json

import numpy as np
import pytest
import tifffile
from ome_types import from_xml
from typer.testing import CliRunner
from yaozarrs import open_group, v05

from opm_processing import export_rois
from opm_processing.dataio.position_collection import open_image_array
from opm_processing.dataio.processing_state import ProcessingState

pytestmark = [pytest.mark.usefixtures("cupy_gpu")]


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
    microscope = case.truth.microscope
    pixel = microscope.pixel_size_um
    centers = (
        case.truth.centers_zyx_um[:, 1:]
        + np.asarray(case.raw_origin_zyx_um)[1:]
        - np.asarray(case.origin)[1:]
    ) / pixel
    margins = np.ceil(np.asarray(microscope.psf_half_extent_zyx_um)[1:] / pixel)
    selections = [
        tuple(np.array((low[0], high[0], low[1], high[1]), dtype=int))
        for low, high in zip(
            np.floor(centers - margins), np.ceil(centers + margins) + 1, strict=True
        )
    ]
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
                    required_point=0,
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
                required_point=index - 1,
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
    from opm_processing.dataio.roi import projection_tile_footprints
    from opm_processing.imageprocessing.opmpsf import camera_grid_center
    from opm_processing.imageprocessing.tilefusion import TileFusion
    from opm_processing.process import process_skewed
    from tests.simulation.camera_likelihood import (
        SamplingEnvelope,
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
    footprints = projection_tile_footprints(preview)
    microscope = case.microscope
    pixel = microscope.pixel_size_um
    native_step = case.optical_data.ground_truth.microscope.scan_step_um
    reference_center = camera_grid_center(
        (case.optical_data.native_scan_planes, *case.expectation.shape[1:]),
        pixel,
        native_step,
        microscope.angle_deg,
    )
    reference_center[1] -= case.optical_data.first_native_scan_index * native_step
    # The original fluorescent lines are centered at laboratory Y=0. Choose
    # their center from acquisition geometry, before inspecting processed pixels.
    cy = round(reference_center[1] / pixel)
    offset = case.parameters["tile_offset_x_pixels"]
    bounds = (cy - 16, cy + 16, offset - 8, case.dataset.raw_data.shape[-1] + 8)
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
    stage_z, stage_scan, stage_x = case.dataset.metadata.stage_positions_zxy[0]
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
            # Unfused tiles retain their reported camera frame. Registered
            # fusion must recover the simulator's independent true placement.
            frame_origin = (
                (
                    -stage_z,
                    -stage_scan
                    - ((shape[0] - 1) * microscope.scan_step_um if stage else 0),
                    stage_x + position * offset * pixel,
                )
                if fused
                else next(
                    item["raw_origin_zyx_um"]
                    for item in footprints
                    if item["position_index"] == position and item["time_index"] == 0
                )
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
            record = next(
                item for item in records if item["position_index"] == position
            )
            start, stop, x0, x1 = record["skewed_bounds_sx"]
            selection = (slice(start, stop), slice(None), slice(x0, x1))
            local_shape = (stop - start, shape[1], x1 - x0)
            local_center = camera_grid_center(
                local_shape, pixel, microscope.scan_step_um, microscope.angle_deg
            )
            local_center += np.asarray(
                (0, start * microscope.scan_step_um, (x0 + position * offset) * pixel)
            )
            apertures = camera_object_apertures(
                case.optical_data.ground_truth,
                microscope,
                local_shape,
                camera_offset_zyx_um=local_center - reference_center,
            )
            local_envelope = SamplingEnvelope(
                envelope.lower[selection], envelope.upper[selection]
            )
            for index, aperture in enumerate(apertures):
                selected = aperture & support[selection]
                if index < 2:
                    assert selected.any()
                if np.any(selected):
                    actual = prediction[selection][selected]
                    assert np.all(
                        np.isfinite(actual)
                        & (actual >= local_envelope.lower[selected])
                        & (actual <= local_envelope.upper[selected])
                    ), (fused, position, index)
