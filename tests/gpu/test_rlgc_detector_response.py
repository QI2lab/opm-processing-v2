"""Detector operator units and calibrated acquisition processing integration."""

import importlib
from contextlib import nullcontext
from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest
from scipy.special import kl_div


@pytest.mark.integration
@pytest.mark.parametrize(
    "mosaic_acquisition",
    [("stage", 1.2, 0.001, "tiled")],
    indirect=True,
)
def test_roi_deconvolves_recorded_sampling(
    cupy_gpu,
    mosaic_acquisition,
    monkeypatch,
    invoke_cli,
    production_forward_predictions,
):
    """Crop raw camera data with optical support before reconstructing the ROI."""
    import tifffile
    from typer.testing import CliRunner
    from yaozarrs import open_group

    from opm_processing import export_rois
    from opm_processing.dataio.position_collection import (
        open_image_array,
        open_position_collection,
        update_group_attributes,
    )
    from opm_processing.dataio.processing_state import ProcessingState
    from opm_processing.imageprocessing.camera import camera_correct
    from opm_processing.imageprocessing.maxtilefusion import (
        regenerate_fused_max_projection,
    )
    from opm_processing.imageprocessing.opmpsf import camera_grid_center
    from opm_processing.imageprocessing.opmtools import deskew_shape_estimator
    from opm_processing.imageprocessing.tilefusion import TileFusion
    from opm_processing.process_roi import process_roi

    case = mosaic_acquisition
    original_files = {
        path.relative_to(case.dataset.path): sha256(path.read_bytes()).digest()
        for path in case.dataset.path.rglob("*")
        if path.is_file()
    }
    invoke_cli(
        "process",
        case.dataset.path,
        max_projection=False,
        create_fused_max_projection=False,
        z_downsample_level=1,
        crop_after_deskew=False,
    )
    full_tiles = case.dataset.path.parent / "mosaic_deskewed.ome.zarr"
    full_fusion = TileFusion(full_tiles, multiscale_factors=())
    full_fusion.run()
    preview = case.dataset.path.parent / "mosaic_max_z_fused.ome.zarr"
    regenerate_fused_max_projection(
        case.dataset.path.parent / "mosaic_fused.ome.zarr", preview
    )
    full_state = ProcessingState.read(
        case.dataset.path.parent / "mosaic.processing.json"
    )
    saved_placements = {
        (tile["time_index"], tile["position_index"]): np.asarray(tile["origin_zyx_um"])
        for tile in full_state.registration(full_tiles)["tiles"]
    }
    full_collection = open_position_collection(full_tiles)
    microscope = case.microscope
    native_shape = case.dataset.raw_data.shape[-3:]
    # The original line object is at laboratory Y=0 on the centered sensor.
    # Place that physical site in the saved preview grid without reading pixels.
    native_step = microscope.scan_step_um / round(microscope.scan_step_um / 0.4)
    source_y = (
        -case.dataset.metadata.stage_positions_zxy[0][1]
        + camera_grid_center(
            (case.optical_data.native_scan_planes, *native_shape[1:]),
            microscope.pixel_size_um,
            native_step,
            microscope.angle_deg,
        )[1]
        - case.optical_data.first_native_scan_index * native_step
    )
    if case.dataset.metadata.scan_axis_reversed:
        source_y -= (native_shape[0] - 1) * microscope.scan_step_um
    grid = open_group(preview).ome_metadata().multiscales[0].datasets[0]
    center = round(
        (source_y - grid.translation_transform.translation[-2])
        / grid.scale_transform.scale[-2]
    )
    source_x = (case.expectation.shape[-1] - 1) // 2
    bounds = (center - 256, center + 256, source_x - 32, source_x + 32)
    monkeypatch.setattr(
        export_rois, "select_rois", lambda source, shape, figure_path=None: [bounds]
    )
    processing = importlib.import_module("opm_processing.process")
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    stages = []
    deconvolved = []
    predictions = []
    seeds = []
    corrections = []
    measurements = []
    rendered = []
    raw_crops = []
    pending_cores = []
    delivered_cores = completed_cores = 0
    open_raw = processing.open_acquisition_datastore
    generate_psfs = export_rois.load_deconvolution_psfs
    deconvolve = solver.rlgc
    channel_writer = processing.deskew_channel_writer
    register = TileFusion.register_with_retries
    measure = TileFusion.register_and_score
    render = TileFusion._render_fusion_block

    def observe_raw(acquisition):
        """Check real camera frames from the producer store before reconstruction."""
        result = open_raw(acquisition)
        assert acquisition.path == case.dataset.path
        for position in range(2):
            np.testing.assert_array_equal(
                result[0, position, 0, :]
                .read()
                .result()
                .reshape(case.dataset.raw_data.shape[-3:]),
                case.dataset.raw_data[0, position, 0],
            )
        stages.append("raw")
        return result

    def observe_psfs(*args, **kwargs):
        """Generate production optical kernels without measured PSF files."""
        assert args[1] is None
        result = generate_psfs(*args, **kwargs)
        assert result[1] is not None
        stages.append("psfs")
        return result

    def observe_deconvolution(*args, **kwargs):
        """Execute the actual CUDA solver and record input/output identity."""
        raw_crops.append(
            (args[0].shape, sha256(memoryview(np.ascontiguousarray(args[0]))).digest())
        )
        assert args[0].shape[1] == case.dataset.raw_data.shape[-2]
        assert args[0].size < np.prod(case.dataset.raw_data.shape[-3:])
        consume = kwargs["on_reconstructed_chunk"]

        def observe_core(core, start, stop):
            nonlocal delivered_cores
            delivered_cores += 1
            pending_cores.append((start, stop, id(core)))
            consume(core, start, stop)
            assert completed_cores >= delivered_cores - 1

        stages.append("deconvolve")
        result = deconvolve(*args, **dict(kwargs, on_reconstructed_chunk=observe_core))
        assert result is None
        measured = production_forward_predictions[-1]
        assert len(measured["chunks"]) > 1
        assert all(record["shared"] for record in measured["chunks"])
        predictions.append(measured)
        return result

    def observe_writer(shape, target, theta, distance, *args, **kwargs):
        """Verify the production deskewer consumes every finalized core unchanged."""
        assert distance == pytest.approx(0.4)
        consume, projection = channel_writer(
            shape, target, theta, distance, *args, **kwargs
        )
        deconvolved.append(shape)
        next_plane = 0

        def observe_deskew(core, start, stop):
            nonlocal next_plane, completed_cores
            assert start == next_plane
            assert core.shape == (stop - start, *shape[1:])
            assert pending_cores.pop(0) == (start, stop, id(core))
            result = consume(core, start, stop)
            completed_cores += 1
            next_plane = stop
            if stop == shape[0]:
                stages.append("deskew")
            return result

        return observe_deskew, projection

    def observe_registration(fusion):
        """Capture crop-displaced saved transforms before fresh ROI refinement."""
        assert fusion._variable_roi_tiles
        assert not fusion._reuse_registered_roi_placements
        seeds.extend(zip(fusion._roi_tile_records, fusion._tile_positions, strict=True))
        stages.append("registration")
        result = register(fusion)
        corrections.extend(np.array(fusion.global_offsets, copy=True))
        return result

    def observe_measurement(*args, **kwargs):
        """Measure actual processed ROI overlaps without changing registration."""
        result = measure(*args, **kwargs)
        measurements.append(result)
        return result

    def observe_render(*args, **kwargs):
        """Record actual rendered block identity without retaining its pixels."""
        result = render(*args, **kwargs)
        rendered.append(
            (
                result[0],
                result[1].shape,
                sha256(memoryview(np.ascontiguousarray(result[1]))).digest(),
            )
        )
        return result

    monkeypatch.setattr(processing, "open_acquisition_datastore", observe_raw)
    monkeypatch.setattr(export_rois, "load_deconvolution_psfs", observe_psfs)
    monkeypatch.setattr(solver, "rlgc", observe_deconvolution)
    monkeypatch.setattr(processing, "deskew_channel_writer", observe_writer)
    monkeypatch.setattr(TileFusion, "register_with_retries", observe_registration)
    monkeypatch.setattr(
        TileFusion, "register_and_score", staticmethod(observe_measurement)
    )
    monkeypatch.setattr(TileFusion, "_render_fusion_block", observe_render)
    result = CliRunner().invoke(
        export_rois.app,
        [
            str(preview),
            "--registration-channel",
            "0",
            "--deconvolve",
            "--save-float32",
            "--tiff",
            "--decon-verbose",
            "0",
            "--decon-crop-scan",
            "64",
        ],
    )
    if result.exception:
        raise result.exception
    assert result.exit_code == 0
    directory = preview.parent / "roi_exports" / "roi_001"
    state = ProcessingState.read(directory / "mosaic.processing.json")
    tiles = directory / "mosaic_decon_deskewed.ome.zarr"
    configuration = state.run(tiles)
    assert stages[:2] == ["psfs", "raw"]
    assert stages[2:-1] == ["deconvolve", "deskew"] * len(seeds)
    assert stages[-1] == "registration"
    assert not pending_cores
    assert delivered_cores == completed_cores
    assert len(seeds) == 2 and measurements
    # The stage fixture records the first acquisition position and positive
    # scan step. Reordering starts at its final acquired plane: image Y is
    # minus stage X. This endpoint displacement belongs to the canonical raw
    # frame; full factor-one processing introduces no Z-bin or deskew crop.
    canonical_scan_offset = np.zeros(3)
    if "stage" in case.dataset.metadata.mode:
        canonical_scan_offset[1] = (
            -(case.dataset.raw_data.shape[-3] - 1) * case.microscope.scan_step_um
        )
    np.testing.assert_allclose(
        full_collection.spatial_origins_zyx_um,
        np.broadcast_to(canonical_scan_offset, (len(full_collection.arrays), 3)),
        rtol=0,
        atol=1e-6,
    )
    for record, initial_origin in seeds:
        position = record["position_index"]
        saved = saved_placements[record["time_index"], position]
        raw_origin = saved
        scan_start, _stop, x_start, _x_stop = record["skewed_bounds_sx"]
        y_crop, _y_stop, x_crop, _x_stop = record["deskew_crop_yx"]
        pixel = case.microscope.pixel_size_um
        displaced = raw_origin + np.array(
            (
                0,
                scan_start * case.microscope.scan_step_um + y_crop * pixel,
                (x_start + x_crop) * pixel,
            )
        )
        np.testing.assert_allclose(initial_origin, displaced, rtol=0, atol=1e-6)
    assert all(np.isfinite(score) for _, score in measurements)
    registration = state.registration(tiles)
    assert registration["fused_path"] == "mosaic_fused.ome.zarr"
    for (_record, seed), correction, final in zip(
        seeds, corrections, registration["tiles"], strict=True
    ):
        np.testing.assert_allclose(
            final["origin_zyx_um"],
            np.asarray(seed) + correction * np.asarray(full_collection.voxel_size_um),
            rtol=0,
            atol=1e-6,
        )
    factor = round(case.microscope.scan_step_um / 0.4)
    for index, ((record, _seed), recovered_shape, actual_input) in enumerate(
        zip(seeds, deconvolved, raw_crops, strict=True)
    ):
        start, stop, x0, x1 = record["skewed_bounds_sx"]
        original = case.dataset.raw_data[
            record["time_index"], record["position_index"], 0
        ]
        if case.dataset.metadata.scan_axis_reversed:
            original = original[::-1]
        calibrated = camera_correct(
            original[start:stop, :, x0:x1],
            case.microscope.camera.offset_adu,
            case.microscope.camera.electrons_per_adu,
        )
        assert actual_input == (
            calibrated.shape,
            sha256(memoryview(np.ascontiguousarray(calibrated))).digest(),
        )
        raw_crops[index] = (start, stop, x0, x1)
        assert recovered_shape == (
            (stop - start - 1) * factor + 1,
            case.dataset.raw_data.shape[-2],
            x1 - x0,
        )
    cropped = open_position_collection(tiles)
    written = []
    for position, (
        recovered_shape,
        raw_crop,
        measured,
    ) in enumerate(zip(deconvolved, raw_crops, predictions, strict=True)):
        start, stop, x0, x1 = raw_crop
        x0 += position * case.parameters["tile_offset_x_pixels"]
        x1 += position * case.parameters["tile_offset_x_pixels"]
        expected_native = case.expectation[start:stop, :, x0:x1]
        baseline = measured["baseline_prediction"]
        assert (
            kl_div(expected_native, measured["prediction"]).sum()
            < kl_div(expected_native, baseline).sum()
        )
        volume = cropped.arrays[position][0, 0].read().result()
        written.append((volume.shape, sha256(memoryview(volume)).digest()))
        np.testing.assert_allclose(cropped.voxel_size_um, pixel, rtol=0, atol=0.0005)
        raw_origin = np.add(
            saved_placements[0, position],
            (0, start * microscope.scan_step_um, raw_crop[2] * pixel),
        )
        offset = (
            np.asarray(cropped.spatial_origins_zyx_um[position]) - raw_origin
        ) / pixel
        np.testing.assert_allclose(
            offset, np.rint(offset), rtol=0, atol=0.0005 / pixel + 1e-12
        )
        output_origin = np.rint(offset).astype(int)
        assert np.all(output_origin >= 0)
        native_shape, *_ = deskew_shape_estimator(
            recovered_shape,
            microscope.angle_deg,
            microscope.scan_step_um / factor,
            pixel,
            crop_after_deskew=False,
        )
        y0, y1, x0, x1 = seeds[position][0]["deskew_crop_yx"]
        np.testing.assert_array_equal(output_origin, (0, y0, x0))
        assert 0 <= y0 < y1 <= native_shape[1]
        assert 0 <= x0 < x1 <= native_shape[2]
        assert volume.shape == (native_shape[0], y1 - y0, x1 - x0)
        assert volume.dtype == np.float32
    assert (
        configuration["configuration"]["deconvolution"].get("scan_upsample_factor", 1)
        == factor
    )
    if factor > 1:
        assert configuration["reconstruction"]["scan_axis_step_um"] == pytest.approx(
            0.4
        )
    for tile in state.roi_series(tiles):
        start, stop = tile["skewed_bounds_sx"][:2]
        assert 0 <= start < stop <= case.dataset.raw_data.shape[-3]
        assert tile["skewed_shape_syx"][0] == stop - start
        shape = tile.get("reconstruction_shape_syx", tile["skewed_shape_syx"])
        assert shape[0] == (stop - start - 1) * factor + 1
    image = open_image_array(directory / "mosaic_fused.ome.zarr").read().result()
    assert image.shape[-2:] == (bounds[1] - bounds[0], bounds[3] - bounds[2])
    assert np.isfinite(image).all() and np.any(image > 0)
    # Fractional registration is rendered by the actual production fuser.
    # This checks persistence; optical truth is checked above in camera space.
    assert rendered
    for selection, shape, digest in rendered:
        block = image[selection]
        assert block.shape == shape
        assert sha256(memoryview(np.ascontiguousarray(block))).digest() == digest
    with tifffile.TiffFile(directory / "mosaic_roi_001.ome.tiff") as tif:
        assert tif.is_ome and tif.is_bigtiff
        np.testing.assert_array_equal(tif.asarray().reshape(image.shape), image)

    settings = configuration["configuration"]["deconvolution"]
    assert settings["read_noise"] == microscope.camera.read_noise_e
    assert settings["psf_parameters"] is not None and settings["psf_sha256"] is None
    monkeypatch.setattr(processing, "open_acquisition_datastore", open_raw)
    monkeypatch.setattr(export_rois, "load_deconvolution_psfs", generate_psfs)
    monkeypatch.setattr(solver, "rlgc", deconvolve)
    monkeypatch.setattr(processing, "deskew_channel_writer", channel_writer)
    monkeypatch.setattr(TileFusion, "register_with_retries", register)
    monkeypatch.setattr(TileFusion, "register_and_score", staticmethod(measure))
    monkeypatch.setattr(TileFusion, "_render_fusion_block", render)
    options = {
        "roi_json": directory / "roi.json",
        "output": directory,
        "registration_channel": 0,
        "deconvolve": True,
        "save_float32": True,
        "z_downsample_level": 1,
        "decon_crop_scan": 64,
        "resume": True,
    }
    before = len(production_forward_predictions)
    process_roi(case.dataset.path, **options)
    assert len(production_forward_predictions) == before
    assert {
        path.relative_to(case.dataset.path): sha256(path.read_bytes()).digest()
        for path in case.dataset.path.rglob("*")
        if path.is_file()
    } == original_files
    parameters = deepcopy(settings["psf_parameters"])
    parameters["support_fwhms"] *= 1.2
    update_group_attributes(
        case.dataset.path,
        {"opm_processing": {"deconvolution": {"psf_parameters": parameters}}},
    )
    with pytest.raises(ValueError, match=r"incompatible|configuration"):
        process_roi(case.dataset.path, **options)
    resumed = open_position_collection(tiles)
    assert resumed.spatial_origins_zyx_um == cropped.spatial_origins_zyx_um
    assert resumed.voxel_size_um == cropped.voxel_size_um
    for array, persisted in zip(resumed.arrays, written, strict=True):
        volume = array[0, 0].read().result()
        assert (volume.shape, sha256(memoryview(volume)).digest()) == persisted


@pytest.mark.unit
@pytest.mark.parametrize(
    "factor,pad_yx,nonuniform,kl_history,read_noise,storage",
    [
        (None, True, True, True, 0, "gpu"),
        (None, True, True, True, 1.6, "gpu"),
        (None, True, True, True, 1.6, "host"),
        (None, False, True, True, 1.6, "gpu"),
        (None, False, True, True, 1.6, "host"),
        (None, True, False, True, 1.6, "gpu"),
        (None, True, True, False, 1.6, "gpu"),
        (1, True, True, True, 1.6, "gpu"),
        (1, True, True, True, 1.6, "host"),
        (3, True, True, True, 1.6, "gpu"),
        (3, True, True, True, 1.6, "host"),
        (3, True, True, False, 1.6, "host"),
    ],
    ids=[
        "photons",
        "read-noise",
        "host",
        "unpadded-yx",
        "host-unpadded-yx",
        "unity-response",
        "iteration-cap",
        "zero-boundary",
        "host-zero-boundary",
        "sparse3",
        "host-sparse3",
        "host-sparse3-iteration-cap",
    ],
)
def test_tiled_consensus_recovers_known_blurred_object(
    cupy_gpu,
    factor,
    pad_yx,
    nonuniform,
    kl_history,
    read_noise,
    storage,
    discrete_optical_case,
    production_forward_predictions,
):
    """Three GC updates reduce known-object camera loss on each storage layout."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    case = discrete_optical_case
    options = {
        "use_kl": kl_history,
        "max_iterations": 3,
        "limit": 0,
        "max_delta": 0,
        "detector_response": case.response,
        "read_noise": read_noise,
        "rng_seed": 42,
        "scan_upsample_factor": factor or 1,
        "boundary": "reflect" if factor is None else "zero",
        "pad_yx": pad_yx,
        "storage": storage,
    }
    for chunk in (None, 2):
        recovered = solver.rlgc(case.measured, case.psf, crop_scan=chunk, **options)
        assert recovered.shape == case.truth.shape
        measured = production_forward_predictions[-1]
        baseline = measured["baseline_prediction"]
        expected = case.expectation[:: (factor or 1)]
        assert (
            kl_div(expected, measured["prediction"]).sum()
            < kl_div(expected, baseline).sum()
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "scan_planes,detector_shape,crop_scan,storage,gpu_budget,allocator",
    [
        (1, (16, 4), 3, "gpu", None, None),
        (1, (16, 4), 3, "host", None, None),
        (7, (16, 4), 3, "gpu", None, None),
        (7, (16, 4), 3, "host", None, None),
        (7, (16, 4), 20, "gpu", None, None),
        (32, (256, 256), 3, "gpu", None, None),
        (32, (256, 256), 20, "gpu", None, None),
        (32, (256, 256), 20, "host", None, None),
        (32, (256, 256), 20, "gpu", 128 * 1024**2, "isolated"),
        (32, (256, 256), 20, "gpu", 128 * 1024**2, "default"),
    ],
    ids=[
        "single-plane",
        "host-single-plane",
        "short",
        "host-short",
        "whole-short",
        "small-chunks",
        "large-chunks",
        "host-large-chunks",
        "limited-cuda-pool",
        "limited-default-pool",
    ],
)
def test_nonuniform_response_preserves_uniform_fluorescence(
    cupy_gpu,
    scan_planes,
    crop_scan,
    detector_shape,
    storage,
    gpu_budget,
    allocator,
    monkeypatch,
    uniform_response_case,
):
    """Preserve known fluorescence while bounding actual per-chunk solver state."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    states = []
    plan_keys = set()
    iterate = solver._iterate_rlgc

    def observe_state(observed, shape, *args, **kwargs):
        """Inspect real allocations and execute the unmodified numerical engine."""
        cache = cupy_gpu.fft.config.get_plan_cache()
        if states and shape != states[-1][1]:
            assert plan_keys.isdisjoint(key for key, _ in cache)
        assert (cupy_gpu.get_array_module(observed) is np) == (storage == "host")
        result = iterate(observed, shape, *args, **kwargs)
        states.append((observed.shape, shape))
        # Keep keys alone: retaining plan objects would itself retain GPU memory.
        plan_keys.clear()
        plan_keys.update(key for key, _ in cache)
        return result

    monkeypatch.setattr(solver, "_iterate_rlgc", observe_state)

    case = uniform_response_case
    response, truth, psf, measured = case.response, case.truth, case.psf, case.measured
    restored = np.empty_like(truth)
    written = 0

    def collect_chunk(core, start, stop):
        """Consume actual finalized cores and require complete ordered coverage."""
        nonlocal written
        assert start == written
        assert core.shape == (stop - start, *truth.shape[1:])
        restored[start:stop] = core
        written = stop

    successful_crops = []
    pool = (
        cupy_gpu.get_default_memory_pool()
        if allocator == "default"
        else cupy_gpu.cuda.MemoryPool()
        if gpu_budget is not None
        else None
    )
    original_limit = pool.get_limit() if pool is not None else None
    if pool is not None:
        pool.set_limit(size=gpu_budget)
    allocator = (
        cupy_gpu.cuda.using_allocator(pool.malloc)
        if pool is not None
        else nullcontext()
    )
    try:
        with allocator:
            result = solver.rlgc(
                measured,
                psf,
                crop_scan=crop_scan,
                detector_response=response,
                verbose=0,
                storage=storage,
                on_successful_crop_scan=successful_crops.append,
                on_reconstructed_chunk=collect_chunk,
            )
    finally:
        if pool is not None:
            assert pool.get_limit() == gpu_budget
            pool.free_all_blocks()
            pool.set_limit(size=original_limit)
    assert result is None
    assert written == truth.shape[0]
    np.testing.assert_allclose(restored, truth, rtol=2e-6, atol=2e-3)
    (successful_crop,) = successful_crops
    if gpu_budget is not None and scan_planes == 32 and crop_scan == 20:
        assert successful_crop < crop_scan
        assert len({shape for _, shape in states}) > 1
    chunk_shape = (min(scan_planes, crop_scan + 2 * psf.shape[0]), *detector_shape)
    padding = solver._linear_fft_pad_width(chunk_shape, psf.shape)
    state_bound = tuple(n + sum(p) for n, p in zip(chunk_shape, padding, strict=True))
    # Short inputs can fit one haloed chunk. Larger inputs must still use
    # multiple bounded states, including after real allocation fallback.
    assert states
    if chunk_shape[0] < scan_planes:
        assert len(states) > 1
    assert all(
        np.all(np.asarray(observed) <= state_bound)
        and np.all(np.asarray(reconstruction) <= state_bound)
        for observed, reconstruction in states
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "scan_factor", [None, 1, 3], ids=["planar", "native", "refined"]
)
@pytest.mark.parametrize("crop_scan", [None, 2], ids=["full", "chunked"])
def test_calibrated_input_preserves_subcount_intensity(
    cupy_gpu, scan_factor, crop_scan
):
    """Retain calibrated values and acquired-plane coordinates under identity optics."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")

    observed = np.broadcast_to(
        np.array((-0.4, 31.9, 64.1, 80000.2), np.float32), (8, 4)
    )
    expected = np.broadcast_to(np.array((0, 31.9, 64.1, 80000.2), np.float32), (8, 4))
    psf = np.ones((1, 1), np.float32)
    if scan_factor is not None:
        # Distinct planes expose a shifted chunk or sampling phase while
        # retaining the original calibrated-count range and float32 bound.
        observed = np.linspace(0.5, 1, 11, dtype=np.float32)[:, None, None] * observed
        target = np.zeros(((observed.shape[0] - 1) * scan_factor + 1, 8, 4), np.float32)
        target[::scan_factor] = np.maximum(observed, 0)
        expected = target
        psf = psf[None]
    actual = solver.rlgc(
        observed,
        psf,
        crop_scan=crop_scan,
        scan_upsample_factor=scan_factor or 1,
        boundary="reflect" if scan_factor is None else "zero",
        gradient_consensus=False,
        max_iterations=1,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0.02)


@pytest.mark.integration
def test_saved_fluorescence_with_detector_response(
    cupy_gpu, uniform_detector_acquisition, production_forward_predictions, monkeypatch
):
    """Recover known detector-space fluorescence and persist the accepted estimate."""
    from opm_processing.dataio.position_collection import open_position_collection
    from opm_processing.process import format_processed_output, process

    case = uniform_detector_acquisition
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    reconstruct = solver.rlgc
    accepted = []

    def observe_reconstruction(*args, **kwargs):
        """Retain the actual accepted planar estimate before production casts it."""
        result = reconstruct(*args, **kwargs)
        accepted.append(result)
        return result

    monkeypatch.setattr(solver, "rlgc", observe_reconstruction)
    process(case.dataset.path, **case.options)
    assert len(production_forward_predictions) == 1
    measured = production_forward_predictions[0]
    truth = case.expectation[0]
    assert kl_div(truth, measured["prediction"]).sum(dtype=np.float64) < kl_div(
        truth, measured["baseline_prediction"]
    ).sum(dtype=np.float64)
    saved = open_position_collection(case.output).arrays[0][0, 0].read().result()
    assert saved.dtype == np.uint16 and saved.shape == case.scene.shape
    np.testing.assert_array_equal(
        saved, format_processed_output(accepted[0], False).reshape(saved.shape)
    )
