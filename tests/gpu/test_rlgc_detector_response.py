"""Detector operator units and calibrated acquisition processing integration."""

import importlib
from contextlib import nullcontext

import numpy as np
import pytest
from scipy.special import kl_div


@pytest.mark.integration
@pytest.mark.parametrize(
    "mosaic_acquisition",
    [("mirror", 0.4, 0.001), ("stage", 1.2, 0.001)],
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
    )
    from opm_processing.dataio.processing_state import ProcessingState
    from opm_processing.imageprocessing.maxtilefusion import (
        regenerate_fused_max_projection,
    )
    from opm_processing.imageprocessing.opmpsf import camera_grid_center
    from opm_processing.imageprocessing.opmtools import (
        deskew_shape_estimator,
        interpolate_rows,
    )
    from opm_processing.imageprocessing.tilefusion import TileFusion
    from opm_processing.process import format_processed_output

    case = mosaic_acquisition
    invoke_cli(
        "process",
        case.dataset.path,
        max_projection=True,
        create_fused_max_projection=True,
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
    bounds = (center - 16, center + 16, source_x - 32, source_x + 32)
    monkeypatch.setattr(export_rois, "select_rois", lambda source, shape: [bounds])
    processing = importlib.import_module("opm_processing.process")
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    stages = []
    deconvolved = []
    predictions = []
    deskew_inputs = []
    seeds = []
    corrections = []
    measurements = []
    rendered = []
    raw_crops = []
    open_raw = processing.open_acquisition_datastore
    generate_psfs = processing.load_deconvolution_psfs
    deconvolve = solver.rlgc
    deskew = processing.orthogonal_deskew
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
        """Execute the actual CUDA solver and retain its output for deskew checks."""
        expected = case.dataset.raw_data[0, len(deconvolved), 0]
        if case.dataset.metadata.scan_axis_reversed:
            expected = expected[::-1]
        from opm_processing.imageprocessing.camera import camera_correct

        expected = camera_correct(
            expected,
            case.microscope.camera.offset_adu,
            case.microscope.camera.electrons_per_adu,
        )
        # Locate the actual read in the independently retained producer frames.
        # A unique photon realization identifies its scan/X preimage exactly.
        candidates = np.argwhere(expected[:, 0, :] == args[0][0, 0, 0])
        matches = [
            (int(scan), int(column))
            for scan, column in candidates
            if scan + args[0].shape[0] <= expected.shape[0]
            and column + args[0].shape[-1] <= expected.shape[-1]
            and np.array_equal(
                expected[
                    scan : scan + args[0].shape[0],
                    :,
                    column : column + args[0].shape[-1],
                ],
                args[0],
            )
        ]
        assert len(matches) == 1
        scan, column = matches[0]
        raw_crops.append(
            (scan, scan + args[0].shape[0], column, column + args[0].shape[-1])
        )
        assert args[0].shape[1] == expected.shape[1]
        assert args[0].size < expected.size
        result = deconvolve(*args, **kwargs)
        deconvolved.append(result)
        predictions.append(production_forward_predictions[-1])
        stages.append("deconvolve")
        return result

    def observe_deskew(data, *args, **kwargs):
        """Verify deskew receives the reconstruction just returned by RLGC."""
        np.testing.assert_array_equal(data, deconvolved[-1])
        deskew_inputs.append(data.copy())
        result = deskew(data, *args, **kwargs)
        stages.append("deskew")
        return result

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
        """Retain actual rendered blocks to verify persistence after ROI fusion."""
        result = render(*args, **kwargs)
        rendered.append((result[0], result[1].copy()))
        return result

    monkeypatch.setattr(processing, "open_acquisition_datastore", observe_raw)
    monkeypatch.setattr(processing, "load_deconvolution_psfs", observe_psfs)
    monkeypatch.setattr(solver, "rlgc", observe_deconvolution)
    monkeypatch.setattr(processing, "orthogonal_deskew", observe_deskew)
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
        ],
    )
    if result.exception:
        raise result.exception
    assert result.exit_code == 0
    directory = preview.parent / "roi_exports" / "roi_001"
    state = ProcessingState.read(directory / "mosaic.processing.json")
    tiles = directory / "mosaic_decon_deskewed.ome.zarr"
    configuration = state.run(tiles)
    assert stages[:2] == ["raw", "psfs"]
    assert stages[2:-1] == ["deconvolve", "deskew"] * len(seeds)
    assert stages[-1] == "registration"
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
    for (record, _seed), recovered, retained, raw_crop in zip(
        seeds, deconvolved, deskew_inputs, raw_crops, strict=True
    ):
        start, stop, x0, x1 = record["skewed_bounds_sx"]
        assert (start, stop, x0, x1) == raw_crop
        assert recovered.shape == (
            (stop - start - 1) * factor + 1,
            case.dataset.raw_data.shape[-2],
            x1 - x0,
        )
        np.testing.assert_array_equal(retained, recovered)
    cropped = open_position_collection(tiles)
    for position, (recovered, raw_crop, measured) in enumerate(
        zip(deconvolved, raw_crops, predictions, strict=True)
    ):
        start, stop, x0, x1 = raw_crop
        x0 += position * case.parameters["tile_offset_x_pixels"]
        x1 += position * case.parameters["tile_offset_x_pixels"]
        expected_native = case.expectation[start:stop, :, x0:x1]
        baseline = measured["input_prediction"]
        if baseline is None:
            # Sparse progress is measured on acquired planes against the
            # actual production initializer, without test-side interpolation.
            baseline = measured["initial_prediction"]
        assert (
            kl_div(expected_native, measured["prediction"]).sum()
            < kl_div(expected_native, baseline).sum()
        )
        volume = cropped.arrays[position][0, 0].read().result()
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
            recovered.shape,
            microscope.angle_deg,
            microscope.scan_step_um / factor,
            pixel,
            crop_after_deskew=False,
            divisible_by=1,
        )
        expected = np.zeros(volume.shape, np.float32)
        interpolate_rows(
            recovered,
            expected,
            native_shape[1],
            native_shape[0],
            microscope.angle_deg,
            microscope.scan_step_um / factor,
            pixel,
            1,
            False,
            tuple(output_origin),
        )
        np.testing.assert_allclose(
            volume,
            format_processed_output(expected, save_float32=volume.dtype.kind == "f"),
            rtol=8 * np.finfo(np.float32).eps,
            atol=0,
        )
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
    for selection, block in rendered:
        np.testing.assert_array_equal(image[selection], block)
    with tifffile.TiffFile(directory / "mosaic_roi_001.ome.tiff") as tif:
        assert tif.is_ome and tif.is_bigtiff
        np.testing.assert_array_equal(tif.asarray().reshape(image.shape), image)


@pytest.mark.unit
@pytest.mark.parametrize(
    "kl_history", [False, True], ids=["iteration-cap", "both-halves-KL"]
)
@pytest.mark.parametrize("read_noise", [0, 1.6], ids=["photons", "read-noise"])
@pytest.mark.parametrize(
    "crop_scan,storage", [(2, "gpu"), (2, "host")], ids=["tiled-fft", "host-tiled-fft"]
)
@pytest.mark.parametrize(
    "factor,pad_yx",
    [(None, True), (None, False), (1, True), (3, True)],
    ids=["native", "native-unpadded-yx", "sparse1", "sparse3"],
)
@pytest.mark.parametrize(
    "nonuniform", [False, True], ids=["unity", "detector-response"]
)
def test_tiled_consensus_recovers_known_blurred_object(
    cupy_gpu,
    factor,
    pad_yx,
    nonuniform,
    kl_history,
    read_noise,
    crop_scan,
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
    for chunk in (None, crop_scan):
        recovered = solver.rlgc(case.measured, case.psf, crop_scan=chunk, **options)
        assert recovered.shape == case.truth.shape
        measured = production_forward_predictions[-1]
        baseline = measured["input_prediction"]
        if baseline is None:
            # Sparse progress is measured against the actual initializer on
            # acquired planes, without interpolation or a second projector.
            baseline = measured["initial_prediction"]
        expected = case.expectation[:: (factor or 1)]
        assert (
            kl_div(expected, measured["prediction"]).sum()
            < kl_div(expected, baseline).sum()
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "scan_planes,detector_shape", [(1, (16, 4)), (7, (16, 4)), (32, (256, 256))]
)
@pytest.mark.parametrize("crop_scan", (3, 20))
@pytest.mark.parametrize(
    "storage,gpu_budget,allocator",
    [
        ("gpu", None, None),
        ("host", None, None),
        ("gpu", 128 * 1024**2, "isolated"),
        ("gpu", 128 * 1024**2, "default"),
    ],
    ids=["gpu", "host", "limited-cuda-pool", "limited-default-pool"],
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
    iterate = solver._iterate_rlgc

    def observe_state(observed, shape, *args, **kwargs):
        """Inspect real allocations and execute the unmodified numerical engine."""
        states.append((observed.shape, shape))
        assert (cupy_gpu.get_array_module(observed) is np) == (storage == "host")
        return iterate(observed, shape, *args, **kwargs)

    monkeypatch.setattr(solver, "_iterate_rlgc", observe_state)

    case = uniform_response_case
    response, truth, psf, measured = case.response, case.truth, case.psf, case.measured
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
            restored = solver.rlgc(
                measured,
                psf,
                crop_scan=crop_scan,
                detector_response=response,
                verbose=0,
                storage=storage,
                on_successful_crop_scan=successful_crops.append,
            )
    finally:
        if pool is not None:
            assert pool.get_limit() == gpu_budget
            pool.free_all_blocks()
            pool.set_limit(size=original_limit)
    np.testing.assert_allclose(restored, truth, rtol=2e-6, atol=2e-3)
    (successful_crop,) = successful_crops
    if gpu_budget is not None and scan_planes == 32 and crop_scan == 20:
        assert successful_crop < crop_scan
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
@pytest.mark.parametrize(
    ("uniform_detector_acquisition", "roi_mode"),
    (
        ("projection", False),
        ("mirror", False),
        ("stage", False),
        ("mirror", True),
        ("stage", True),
        *[
            pytest.param((mode, "uint16"), False, id=f"uint16-{mode}")
            for mode in ("mirror", "stage", "projection")
        ],
    ),
    indirect=["uniform_detector_acquisition"],
)
def test_saved_fluorescence_with_detector_response(
    cupy_gpu, uniform_detector_acquisition
):
    """Persist calibrated processing, geometric support and optical-model resume checks."""
    import copy

    from opm_processing.dataio.position_collection import (
        open_position_collection,
        update_group_attributes,
    )
    from opm_processing.dataio.processing_state import (
        ProcessingState,
        processing_state_path,
    )
    from opm_processing.process import process
    from opm_processing.process_roi import process_roi

    case = uniform_detector_acquisition
    run = process_roi if case.roi_mode else process
    original = {
        path.relative_to(case.dataset.path): path.read_bytes()
        for path in case.dataset.path.rglob("*")
        if path.is_file()
    }
    run(case.dataset.path, **case.options)
    saved = open_position_collection(case.output).arrays[0][0, 0].read().result()
    assert saved.dtype == (np.float32 if case.options["save_float32"] else np.uint16)
    assert saved.shape[:2] == case.support.shape
    np.testing.assert_array_equal(saved[~case.support], 0)
    state = ProcessingState.read(
        processing_state_path(
            case.output.parent, case.dataset.path.stem.removesuffix(".ome")
        )
    )
    settings = state.run(case.output)["configuration"]["deconvolution"]
    assert settings["read_noise"] == case.scene.microscope.camera.read_noise_e
    assert settings["psf_parameters"] is not None
    assert settings["psf_sha256"] is None
    run(case.dataset.path, **case.options, resume=True)
    assert {
        path.relative_to(case.dataset.path): path.read_bytes()
        for path in case.dataset.path.rglob("*")
        if path.is_file()
    } == original
    # Deliberately change optical calibration only after the successful real
    # acquisition/processing round trip; resume must reject the new model.
    parameters = copy.deepcopy(settings["psf_parameters"])
    parameters["support_fwhms"] *= 1.2
    update_group_attributes(
        case.dataset.path,
        {"opm_processing": {"deconvolution": {"psf_parameters": parameters}}},
    )
    with pytest.raises(ValueError, match=r"incompatible|configuration"):
        run(case.dataset.path, **case.options, resume=True)
    np.testing.assert_array_equal(
        open_position_collection(case.output).arrays[0][0, 0].read().result(), saved
    )
