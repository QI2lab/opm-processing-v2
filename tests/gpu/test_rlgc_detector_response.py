"""Detector operator units and calibrated acquisition processing integration."""

import importlib
from contextlib import nullcontext

import numpy as np
import pytest

from tests.simulation.camera import digitize
from tests.simulation.metrics import (
    camera_flux_accuracy,
    camera_partition_apertures,
    camera_prediction,
    predictive_deviance,
)


@pytest.mark.integration
@pytest.mark.parametrize(
    "mosaic_acquisition",
    [("mirror", 0.4, 0.001, 2), ("stage", 1.2, 0.001, 2)],
    indirect=True,
)
def test_roi_deconvolves_recorded_sampling(
    cupy_gpu, mosaic_acquisition, monkeypatch, invoke_cli
):
    """Crop raw camera data with optical support before reconstructing the ROI."""
    import tifffile
    from typer.testing import CliRunner

    from opm_processing import export_rois
    from opm_processing.dataio.position_collection import (
        open_image_array,
        open_position_collection,
    )
    from opm_processing.dataio.processing_state import ProcessingState
    from opm_processing.imageprocessing.maxtilefusion import (
        regenerate_fused_max_projection,
    )
    from opm_processing.imageprocessing.tilefusion import TileFusion

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
    projection = open_image_array(preview).read().result()[0, 0, 0]
    center = int(np.argmax(projection.max(axis=1)))
    bounds = (center - 16, center + 16, 40, 88)
    monkeypatch.setattr(export_rois, "select_rois", lambda source, shape: [bounds])
    processing = importlib.import_module("opm_processing.process")
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    stages = []
    deconvolved = []
    reconstruction_models = []
    deskew_inputs = []
    seeds = []
    corrections = []
    measurements = []
    raw_crops = []
    open_raw = processing.open_acquisition_datastore
    generate_psfs = processing.load_deconvolution_psfs
    deconvolve = solver.rlgc
    deskew = processing.orthogonal_deskew
    register = TileFusion.register_with_retries
    measure = TileFusion.register_and_score

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
        reconstruction_models.append(
            (args[0].copy(), np.asarray(args[1]).copy(), kwargs["boundary"])
        )
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

    monkeypatch.setattr(processing, "open_acquisition_datastore", observe_raw)
    monkeypatch.setattr(processing, "load_deconvolution_psfs", observe_psfs)
    monkeypatch.setattr(solver, "rlgc", observe_deconvolution)
    monkeypatch.setattr(processing, "orthogonal_deskew", observe_deskew)
    monkeypatch.setattr(TileFusion, "register_with_retries", observe_registration)
    monkeypatch.setattr(
        TileFusion, "register_and_score", staticmethod(observe_measurement)
    )
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
    from scipy.interpolate import interp1d

    for position, (recovered, raw_crop, model) in enumerate(
        zip(deconvolved, raw_crops, reconstruction_models, strict=True)
    ):
        observed, kernel, boundary = model
        start, stop, x0, x1 = raw_crop
        x0 += position * case.parameters["tile_offset_x_pixels"]
        x1 += position * case.parameters["tile_offset_x_pixels"]
        expected_native = case.expectation[start:stop, :, x0:x1]
        expected = np.zeros_like(recovered)
        expected[::factor] = expected_native
        baseline = observed
        if factor > 1:
            baseline = interp1d(
                np.arange(observed.shape[0]) * factor, observed, axis=0
            )(np.arange(recovered.shape[0]))
        kernel /= kernel.sum()
        selection = np.s_[::factor]
        boundary = "constant" if boundary == "zero" else "reflect"
        assert predictive_deviance(
            recovered, expected, kernel, boundary=boundary, selection=selection
        ) < predictive_deviance(
            baseline, expected, kernel, boundary=boundary, selection=selection
        )
        prediction = camera_prediction(recovered, kernel, boundary=boundary)[::factor]
        for aperture in camera_partition_apertures(expected_native.shape):
            assert camera_flux_accuracy(
                prediction, expected_native, case.microscope.camera, aperture
            ), "ROI reconstruction changed known original photon counts"
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
    with tifffile.TiffFile(directory / "mosaic_roi_001.ome.tiff") as tif:
        assert tif.is_ome and tif.is_bigtiff
        np.testing.assert_array_equal(tif.asarray().reshape(image.shape), image)


@pytest.mark.integration
@pytest.mark.parametrize(
    "roi_preview", [("mirror", 1, False), ("stage", 2, False)], indirect=True
)
def test_roi_cli_deconvolves_every_timepoint_and_channel(
    cupy_gpu,
    roi_preview,
    invoke_cli,
    monkeypatch,
    tmp_path,
    assert_roi_point_locations,
    assert_reconstruction_storage,
):
    """Retain all measured spectral/time samples through physical-PSF ROI RLGC."""
    from yaozarrs import open_group

    from opm_processing import export_rois
    from opm_processing.dataio.deconvolution import scan_upsample_from_metadata
    from opm_processing.dataio.position_collection import open_image_array
    from opm_processing.dataio.processing_state import ProcessingState
    from tests.simulation.metrics import (
        camera_noise_accuracy,
        camera_object_apertures,
        camera_prediction,
    )

    case = roi_preview
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    solve_chunk = solver._rlgc_once
    chunks = []
    solutions = []
    reconstruct = solver.rlgc

    def observe_solution(image, psf, *args, **kwargs):
        """Retain real inverse solutions and their generated optical kernels."""
        result = reconstruct(image, psf, *args, **kwargs)
        solutions.append((result.copy(), np.asarray(psf).copy(), kwargs["boundary"]))
        return result

    def observe_chunk(image, psf, *args, **kwargs):
        """Complete the real chunk solver and check its raw camera-space extent."""
        result = solve_chunk(image, psf, *args, **kwargs)
        chunks.append((image.shape, psf.shape, kwargs.get("storage")))
        return result

    monkeypatch.setattr(solver, "_rlgc_once", observe_chunk)
    monkeypatch.setattr(solver, "rlgc", observe_solution)
    # Both independently prescribed emitters and their surrounding optical
    # background lie in this selection; the old corner missed both sources.
    bounds = (8, 44, 0, 32)
    monkeypatch.setattr(export_rois, "select_rois", lambda source, shape: [bounds])
    destination = tmp_path / "selected"
    invoke_cli(
        "export-rois",
        case.preview,
        output=destination,
        deconvolve=True,
        save_float32=True,
        decon_verbose=0,
        decon_crop_scan=4 if case.positions > 1 else None,
        registration_channel=1,
    )
    directory = destination / "roi_001"
    image = open_image_array(directory / "sample_fused.ome.zarr").read().result()
    assert image.shape[:2] == case.dataset.metadata.shape[:3:2]
    assert image.shape[-2:] == (bounds[1] - bounds[0], bounds[3] - bounds[2])
    assert image.dtype == np.float32 and np.isfinite(image).all()
    assert np.all(image.max(axis=(2, 3, 4)) > 0)
    state = ProcessingState.read(directory / "sample.processing.json")
    tiles = directory / "sample_decon_deskewed.ome.zarr"
    assert state.completed_channels(tiles) == set(np.ndindex(2, case.positions, 2))
    factor = scan_upsample_from_metadata(case.dataset.metadata)
    raw_shape = case.dataset.raw_data.shape[-3:]
    assert len(chunks) >= 2 * case.positions * 2
    for shape, psf_shape, storage in chunks:
        assert storage == "gpu"
        assert shape[1] == raw_shape[1]
        assert shape[2] <= raw_shape[2]
        if case.positions > 1:
            limit = max(1, 4 // factor) + 2 * int(np.ceil(psf_shape[0] / factor))
            assert shape[0] <= min(raw_shape[0], limit + int(factor > 1))
    settings = state.run(tiles)["configuration"]["deconvolution"]
    assert settings["read_noise"] == 0
    records = state.roi_series(tiles)
    assert len(solutions) == 2 * case.positions * 2
    apertures = camera_object_apertures(
        case.truth.ground_truth, case.truth.microscope, raw_shape
    )
    metadata = open_group(directory / "sample_fused.ome.zarr").ome_metadata()
    saved_grid = metadata.multiscales[0].datasets[0]
    origin = saved_grid.translation_transform.translation[-3:]
    spacing = saved_grid.scale_transform.scale[-3:]
    assert_roi_point_locations(
        image, case.truth, origin, spacing, raw_origin=case.raw_origin_zyx_um
    )
    for _, generated, _ in solutions:
        np.testing.assert_array_equal(generated, solutions[0][1])
    for (record, channel), (recovered, kernel, boundary) in zip(
        ((record, channel) for record in records for channel in range(2)),
        solutions,
        strict=True,
    ):
        time, position = record["time_index"], record["position_index"]
        start, stop, x0, x1 = record["skewed_bounds_sx"]
        truth = case.truth.means[time, position, channel, start:stop, :, x0:x1]
        kernel /= kernel.sum()
        boundary = "constant" if boundary == "zero" else boundary
        prediction = camera_prediction(recovered, kernel, boundary=boundary)[::factor]
        for index, aperture in enumerate(apertures):
            selected = aperture[start:stop, :, x0:x1]
            if index < 2:
                assert selected.any(), "ROI must retain known source and background"
            if selected.any():
                accuracy = camera_noise_accuracy(
                    prediction, truth, case.truth.microscope.camera, selected
                )
                assert accuracy.accepted, accuracy
    expected_origin = np.add(
        case.raw_origin_zyx_um,
        (0, bounds[0] * case.truth.microscope.pixel_size_um, 0),
    )
    np.testing.assert_allclose(
        [tile["origin_zyx_um"] for tile in state.registration(tiles)["tiles"]],
        np.broadcast_to(expected_origin, (len(records), 3)),
        rtol=0,
        atol=1e-6,
    )
    recovered_by_channel = {
        (record["time_index"], record["position_index"], channel): result[0]
        for (record, channel), result in zip(
            ((record, channel) for record in records for channel in range(2)),
            solutions,
            strict=True,
        )
    }
    assert factor == 1
    assert all(
        tuple(record["skewed_bounds_sx"]) == (0, raw_shape[0], 0, raw_shape[2])
        for record in records
    )
    # Link final stored pixels to the independently scored inverses without
    # treating lossy interpolation as a second ideal optical inverse.
    for time, channel in np.ndindex(image.shape[:2]):
        assert_reconstruction_storage(
            image[time, channel],
            tuple(
                recovered_by_channel[time, position, channel]
                for position in range(case.positions)
            ),
            origin,
            spacing,
            case.raw_origin_zyx_um,
            case.truth.microscope,
            apertures[:2],
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "kl_history", [False, True], ids=["iteration-cap", "both-halves-KL"]
)
@pytest.mark.parametrize("read_noise", [0, 1.6], ids=["photons", "read-noise"])
@pytest.mark.parametrize(
    "crop_scan,storage", [(2, "gpu"), (2, "host")], ids=["tiled-fft", "host-tiled-fft"]
)
@pytest.mark.parametrize("factor", [None, 1, 3], ids=["native", "sparse1", "sparse3"])
@pytest.mark.parametrize(
    "nonuniform", [False, True], ids=["unity", "detector-response"]
)
def test_tiled_consensus_recovers_known_blurred_object(
    cupy_gpu,
    factor,
    nonuniform,
    kl_history,
    read_noise,
    crop_scan,
    storage,
    discrete_optical_case,
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
    }
    whole = solver.rlgc(case.measured, case.psf, storage=storage, **options)
    tiled = solver.rlgc(
        case.measured, case.psf, crop_scan=crop_scan, storage=storage, **options
    )
    assert tiled.shape == case.truth.shape
    baseline = case.measured / case.response
    if factor is not None:
        from scipy.interpolate import interp1d

        baseline = interp1d(np.arange(baseline.shape[0]) * factor, baseline, axis=0)(
            np.arange(case.truth.shape[0])
        )
    baseline_loss = predictive_deviance(
        baseline,
        case.expectation,
        case.psf,
        boundary=case.boundary,
        response=case.response,
    )
    # The GC gate is discontinuous at zero: single-precision FFT roundoff can
    # change an accepted update. This bounded-iteration unit checks update
    # progress against original optical truth. Converged photon accuracy is
    # covered by the point/detector units and complete acquisition workflows.
    for recovered in (whole, tiled):
        assert (
            predictive_deviance(
                recovered,
                case.expectation,
                case.psf,
                boundary=case.boundary,
                response=case.response,
            )
            < baseline_loss
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
    # Failed allocation attempts may reach the engine before retrying. A complete
    # successful attempt must still run every retained chunk on the same backend.
    assert len(states) >= int(np.ceil(scan_planes / successful_crop))
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
        observed = np.broadcast_to(observed, (11, *observed.shape))
        target = np.zeros(((observed.shape[0] - 1) * scan_factor + 1, 8, 4), np.float32)
        target[::scan_factor] = expected
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


@pytest.mark.unit
@pytest.mark.parametrize("crop_scan", (4, 20))
@pytest.mark.parametrize("seed", (31, 83))
def test_detector_response_recovers_point_flux(
    cupy_gpu, point_response_model, ideal_photon_camera, crop_scan, seed
):
    """Recover known emitters from integer Poisson draws under a varying response."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    case = point_response_model
    psf, response, expectation = case.psf, case.response, case.expectation
    measured = digitize(expectation, ideal_photon_camera, seed=seed).astype(np.float32)
    actual = solver.rlgc(
        measured, psf, crop_scan=crop_scan, detector_response=response, verbose=0
    )
    prediction = camera_prediction(actual, psf, boundary="reflect", response=response)
    for aperture in case.apertures:
        assert camera_flux_accuracy(
            prediction, expectation, ideal_photon_camera, aperture
        ), "Point reconstruction changed known absolute photon counts"
    assert predictive_deviance(
        actual, expectation, psf, boundary="reflect", response=response
    ) < predictive_deviance(
        measured / response, expectation, psf, boundary="reflect", response=response
    )
    for point in case.points:
        neighborhood = actual[tuple(slice(p - 1, p + 2) for p in point)]
        assert np.unravel_index(np.argmax(neighborhood), neighborhood.shape) == (
            1,
            1,
            1,
        )


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
    cupy_gpu, uniform_detector_acquisition, monkeypatch, stored_file_bytes
):
    """Check calibrated processing orchestration against known absolute fluorescence."""
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
    original = stored_file_bytes(case.dataset.path)
    reconstructions = []
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    reconstruct = solver.rlgc

    def observe(image, psf, *args, **kwargs):
        """Retain actual solutions and their generated optical measurement model."""
        result = reconstruct(image, psf, *args, **kwargs)
        reconstructions.append(
            (
                result.copy(),
                np.asarray(psf).copy(),
                kwargs.get("detector_response", 1),
                kwargs.get("boundary", "reflect"),
            )
        )
        return result

    monkeypatch.setattr(solver, "rlgc", observe)
    run(case.dataset.path, **case.options)
    saved = open_position_collection(case.output).arrays[0][0, 0].read().result()
    assert saved.dtype == (np.float32 if case.options["save_float32"] else np.uint16)
    assert saved.shape[:2] == case.support.shape
    np.testing.assert_array_equal(saved[~case.support], 0)
    # In a uniform physical object the photon-volume Jacobian fixes absolute
    # fluorescence. Average a fixed fully acquired region, not individual
    # inverse-filtered pixels; the strict deterministic operator unit remains.
    np.testing.assert_allclose(
        saved[case.support].mean(), case.expected, rtol=0.005, atol=1
    )
    state = ProcessingState.read(
        processing_state_path(
            case.output.parent, case.dataset.path.stem.removesuffix(".ome")
        )
    )
    settings = state.run(case.output)["configuration"]["deconvolution"]
    assert settings["read_noise"] == 0
    assert settings["psf_parameters"] is not None
    assert settings["psf_sha256"] is None
    records = state.roi_series(case.output)
    assert len(reconstructions) == 1
    recovered, kernel, response, boundary = reconstructions[0]
    expected = case.means
    if records:
        start, stop, x0, x1 = records[0]["skewed_bounds_sx"]
        expected = expected[start:stop, :, x0:x1]
    if recovered.ndim == 2:
        expected = expected[0]
        kernel = kernel[kernel.shape[0] // 2]
    kernel /= kernel.sum()
    assert recovered.shape == expected.shape
    from tests.simulation.metrics import camera_noise_accuracy, camera_prediction

    boundary = "constant" if boundary == "zero" else boundary
    prediction = camera_prediction(
        recovered, kernel, boundary=boundary, response=response
    )
    camera = case.scene.microscope.camera
    # This is a declared high-count measurement-noise accuracy target, not
    # an uncertainty claim about individual nonlinear reconstructed pixels.
    # Fixed detector rows and columns prevent local stripes from cancelling
    # or being diluted by the whole-camera photon count.
    for mask in camera_partition_apertures(expected.shape):
        accuracy = camera_noise_accuracy(prediction, expected, camera, mask)
        assert accuracy.accepted, accuracy
    # D(0 || prediction) = 2*sum(prediction). Reuse the independent SciPy
    # camera predictor in the existing metric, then test total measured flux
    # against original photon means. This avoids any per-pixel noise claim
    # about deconvolution or fitting a gain to the recovered image.
    predicted_flux = (
        predictive_deviance(
            recovered,
            np.zeros_like(recovered),
            kernel,
            boundary=boundary,
            response=response,
        )
        / 2
    )
    variance = expected.sum(dtype=np.float64) + expected.size * (
        camera.read_noise_e**2 + camera.electrons_per_adu**2 / 12
    )
    assert abs(predicted_flux - expected.sum(dtype=np.float64)) <= 6 * np.sqrt(variance)
    run(case.dataset.path, **case.options, resume=True)
    assert stored_file_bytes(case.dataset.path) == original
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
