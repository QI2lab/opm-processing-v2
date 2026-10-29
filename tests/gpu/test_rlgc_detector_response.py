"""Detector operator units and calibrated acquisition processing integration."""

import importlib
from contextlib import nullcontext

import numpy as np
import pytest
from scipy import ndimage

from tests.simulation.metrics import predictive_deviance


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
        expected = (
            np.maximum(
                expected.astype(np.float32) - case.microscope.camera.offset_adu, 0
            )
            * case.microscope.camera.electrons_per_adu
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
    for record, initial_origin in seeds:
        position = record["position_index"]
        saved = saved_placements[record["time_index"], position]
        raw_origin = saved - full_collection.spatial_origins_zyx_um[position]
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
    cupy_gpu, roi_preview, invoke_cli, monkeypatch, tmp_path
):
    """Retain all measured spectral/time samples through physical-PSF ROI RLGC."""
    from opm_processing import export_rois
    from opm_processing.dataio.deconvolution import scan_upsample_from_metadata
    from opm_processing.dataio.position_collection import open_image_array
    from opm_processing.dataio.processing_state import ProcessingState

    case = roi_preview
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    solve_chunk = solver._rlgc_once
    chunks = []

    def observe_chunk(image, psf, *args, **kwargs):
        """Complete the real chunk solver and check its raw camera-space extent."""
        result = solve_chunk(image, psf, *args, **kwargs)
        chunks.append((image.shape, psf.shape, kwargs.get("storage")))
        return result

    monkeypatch.setattr(solver, "_rlgc_once", observe_chunk)
    bounds = (3, 12, 2, 10)
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
    """Full and tiled GC improve the likelihood of a known blurred 3D object."""
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
    # change an accepted update. Verify physical accuracy against truth for each
    # layout; forward/adjoint units separately bound convolution seam error.
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

    response = np.broadcast_to(
        np.tile(np.array((0.25, 0.5, 1, 2), np.float32), detector_shape[1] // 4),
        detector_shape,
    )
    truth = np.full((scan_planes, *response.shape), 1200, dtype=np.float32)
    psf = np.ones((1 if scan_planes == 1 else 3, 3, 3), dtype=np.float32)
    psf /= psf.sum()
    measured = ndimage.convolve(truth, psf, mode="reflect") * response
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
def test_detector_response_recovers_point_flux(cupy_gpu, crop_scan, seed):
    """Recover known emitters from integer Poisson draws under a varying response."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    z, y, x = np.mgrid[-2:3, -3:4, -3:4]
    psf = np.exp(-0.5 * (z**2 + (y / 1.5) ** 2 + (x / 1.5) ** 2)).astype(np.float32)
    psf /= psf.sum()
    truth = np.zeros((9, 35, 37), np.float32)
    truth[3, 10, 10], truth[6, 25, 27] = 20000, 30000
    response = np.broadcast_to(np.linspace(0.25, 2, 37, dtype=np.float32), (35, 37))
    expectation = response * (ndimage.convolve(truth, psf, mode="reflect") + 0.25)
    measured = np.random.default_rng(seed).poisson(expectation).astype(np.float32)
    actual = solver.rlgc(
        measured, psf, crop_scan=crop_scan, detector_response=response, verbose=0
    )
    assert predictive_deviance(
        actual, expectation, psf, boundary="reflect", response=response
    ) < predictive_deviance(
        measured / response, expectation, psf, boundary="reflect", response=response
    )
    for point in ((3, 10, 10), (6, 25, 27)):
        neighborhood = actual[tuple(slice(p - 1, p + 2) for p in point)]
        assert np.unravel_index(np.argmax(neighborhood), neighborhood.shape) == (
            1,
            1,
            1,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("uniform_detector_acquisition", "roi_mode"),
    (
        ("projection", False),
        ("mirror", False),
        ("stage", False),
        ("mirror", True),
        ("stage", True),
        *[
            pytest.param((mode, "generated-PSF"), False, id=f"generated-psf-{mode}")
            for mode in ("mirror", "stage", "projection")
        ],
    ),
    indirect=["uniform_detector_acquisition"],
)
def test_saved_fluorescence_with_detector_response(
    cupy_gpu, uniform_detector_acquisition
):
    """Check calibrated processing orchestration against known absolute fluorescence."""
    from opm_processing.dataio.position_collection import open_position_collection
    from opm_processing.dataio.processing_state import (
        ProcessingState,
        processing_state_path,
    )
    from opm_processing.process import process
    from opm_processing.process_roi import process_roi

    case = uniform_detector_acquisition
    run = process_roi if case.roi_mode else process
    run(case.dataset.path, **case.options)
    saved = open_position_collection(case.output).arrays[0][0, 0].read().result()
    np.testing.assert_allclose(saved, case.expected, rtol=0.005, atol=1)
    state = ProcessingState.read(
        processing_state_path(
            case.output.parent, case.dataset.path.stem.removesuffix(".ome")
        )
    )
    settings = state.run(case.output)["configuration"]["deconvolution"]
    assert settings["read_noise"] == 0
    if "decon_psf_paths" in case.options:
        assert settings["psf_parameters"] is None
        np.save(case.psf_path, np.full((3, 3, 3), 2 / 27, np.float32))
    else:
        from yaozarrs import open_group

        from opm_processing.dataio.position_collection import update_group_attributes

        root = open_group(case.dataset.path)
        calibration = root.attrs["opm_processing"]
        assert (
            settings["psf_parameters"] == calibration["deconvolution"]["psf_parameters"]
        )
        calibration["deconvolution"]["psf_parameters"]["support_fwhms"] *= 1.2
        update_group_attributes(case.dataset.path, {"opm_processing": calibration})
    with pytest.raises(ValueError, match=r"incompatible|configuration"):
        run(case.dataset.path, **case.options, resume=True)
    np.testing.assert_array_equal(
        open_position_collection(case.output).arrays[0][0, 0].read().result(), saved
    )
