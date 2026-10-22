"""Shared qi2lab acquisitions for complete processing and fusion integration."""

import importlib
from dataclasses import replace
from functools import lru_cache, partial
from types import SimpleNamespace

import numpy as np
import pytest

from opm_processing.dataio.deconvolution import scan_upsample_from_metadata
from opm_processing.imageprocessing.tilefusion import TileFusion
from tests.simulation.camera import digitize
from tests.simulation.forward_model import Acquisition, camera_psf, forward_model
from tests.simulation.scenes import prepare_scene


@pytest.fixture
def pipeline_runner(monkeypatch, processing_options, invoke_cli):
    """Run real processing and fusion with shared storage and worker boundaries.

    Returns
    -------
    callable
        Process a raw acquisition into an independently selected output directory,
        optionally configuring deskew output tiles, RLGC and fusion block sizes.
    """
    from opm_processing.dataio.acquisition import acquisition_stem
    from opm_processing.dataio.position_collection import open_position_collection
    from opm_processing.imageprocessing import opmtools
    from opm_processing.imageprocessing.opmtools import orthogonal_deskew

    module = importlib.import_module("opm_processing.process")
    fuse = importlib.import_module("opm_processing.fuse")

    def run(
        dataset,
        *,
        deconvolve=False,
        illumination=False,
        crop_scan=None,
        psf_path=None,
        output=None,
        deskew_tile=None,
        decon_storage=None,
        fusion_chunk=(16, 16),
        blend=(0, 0, 0),
    ):
        """Persist calibrated tiles and registered multiscales using real algorithms."""
        options = dict(
            processing_options,
            deconvolve=deconvolve,
            flatfield_correction=illumination,
            save_float32=True,
            output=output,
        )
        if deconvolve:
            options.update(
                decon_psf_paths=[psf_path] * dataset.raw_data.shape[2],
                decon_crop_scan=crop_scan,
                decon_verbose=0,
            )
        deskew_tiles = set()
        reconstructions = []
        with monkeypatch.context() as execution:
            if deconvolve:
                solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
                reconstruct = solver.rlgc

                def observe_reconstruction(*args, **kwargs):
                    """Retain actual calibrated reconstructions for physical truth checks."""
                    if decon_storage is not None:
                        kwargs["storage"] = decon_storage
                    result = reconstruct(*args, **kwargs)
                    reconstructions.append(result.copy())
                    return result

                execution.setattr(solver, "rlgc", observe_reconstruction)
            if deskew_tile is not None:
                interpolate = opmtools.interpolate_rows

                def record_tile(*args):
                    """Record executed global output tiles and run the real kernel."""
                    deskew_tiles.add((args[-1], args[1].shape))
                    return interpolate(*args)

                execution.setattr(opmtools, "interpolate_rows", record_tile)
                execution.setattr(
                    module,
                    "orthogonal_deskew",
                    partial(orthogonal_deskew, tile_shape=deskew_tile),
                )
            invoke_cli("process", dataset.path, **options)
        directory = dataset.path.parent if output is None else output
        stem = acquisition_stem(dataset.path)
        processed = (
            directory / f"{stem}_{'decon_' if deconvolve else ''}deskewed.ome.zarr"
        )
        field = (
            module.read_current_flatfield(
                directory / f"{stem}_flatfield.ome.tif",
                (1, dataset.raw_data.shape[2], *dataset.raw_data.shape[-2:]),
            )
            if illumination
            else None
        )
        created = []

        def observe_fusion(*args, **kwargs):
            """Keep the actual CLI-created solver and its nominal stage positions."""
            fusion = TileFusion(*args, **kwargs)
            created.append((fusion, np.asarray(fusion._tile_positions).copy()))
            return fusion

        with monkeypatch.context() as execution:
            execution.setattr(fuse, "TileFusion", observe_fusion)
            invoke_cli(
                "fuse",
                processed,
                blend_pixels=blend,
                downsample_factors=(1, 1, 1),
                registration_channel=dataset.raw_data.shape[2] - 1,
                ssim_window=3,
                registration_threshold=0.1,
                max_registration_shift_zyx=(2, 2, 2),
                chunk_shape_yx=fusion_chunk,
                max_workers=1,
                normalize_depth_intensity=False,
            )
        ((fusion, nominal),) = created
        return SimpleNamespace(
            options=options,
            processed=processed,
            field=field,
            fusion=fusion,
            nominal=nominal,
            collection=open_position_collection(processed),
            output=directory / f"{stem}_fused.ome.zarr",
            deskew_tiles=deskew_tiles,
            reconstructions=reconstructions,
        )

    return run


@pytest.fixture(scope="session")
def pipeline_optical_data(
    microscope_parameters, simulation_parameters, acquisition_metadata
):
    """Cache production optics and camera means for repeated fields and mosaics.

    Returns
    -------
    callable
        Return microscope, noiseless camera means and the matching production PSF
        for a scan step, illumination selection and extended-mosaic flag.
    """

    @lru_cache(maxsize=6)
    def create(step, illumination=False, *, mosaic=False):
        """Render one calibrated physical specimen before independent camera draws."""
        microscope = replace(microscope_parameters, scan_step_um=step)
        settings = simulation_parameters["pipeline_mosaic" if mosaic else "pipeline"]
        kind = "line-pairs" if mosaic else "points"
        specimen = simulation_parameters["specimens"][kind]
        if mosaic:
            specimen = {
                "half_extent_zyx_um": settings["half_extent_zyx_um"],
                "features": dict(
                    specimen["features"], length_um=settings["line_length_um"]
                ),
            }
        scene = prepare_scene(kind, microscope, parameters=specimen)
        height, width = settings["camera_shape_yx"]
        shape = (
            round(settings["scan_extent_um"] / step) + 1,
            height,
            width + settings.get("tile_offset_x_pixels", 0),
        )
        expected = forward_model(
            scene.density,
            Acquisition(
                scene.axes,
                shape,
                illumination_profile="edge-taper" if illumination else "uniform",
                camera_photon_budget=tuple(simulation_parameters["photon_budgets"][0]),
                output="expectation",
            ),
            microscope,
        )
        factor = scan_upsample_from_metadata(
            replace(acquisition_metadata, scan_axis_step_um=step)
        )
        psf, _ = camera_psf(replace(microscope, scan_step_um=step / factor))
        return microscope, expected, psf.astype(np.float32)

    return create


@pytest.fixture
def pipeline_case(
    request,
    deconvolve,
    pipeline_optical_data,
    simulation_parameters,
    acquisition_factory,
    tmp_path,
    pipeline_runner,
    small_cpu_kernels,
):
    """Process independent repeat exposures with a known stage-placement error.

    Indirect parameters select scan mode, illumination, scan step, FFT scan
    chunk and whether read-noise calibration is recorded.

    Returns
    -------
    types.SimpleNamespace
        Raw acquisition, reopened processed collection, fitted illumination,
        fusion object and the independent expected support.

    Notes
    -----
    Both positions observe the same physical field with independent camera
    noise. Position 1 reports an erroneous one-pixel lateral stage shift.
    BaSiC runs in its production worker process. Camera calibration, RLGC,
    orthogonal interpolation, registration and fusion execute normally, with
    native TensorStore storage in pytest temporary directories.
    """
    from opm_processing.imageprocessing.opmtools import orthogonal_deskew

    mode, illumination, step, crop, record_noise = request.param
    microscope, expectation, psf = pipeline_optical_data(step, illumination)
    raw = np.empty((2, 2, 2, *expectation.shape), dtype=np.uint16)
    for time, position, channel in np.ndindex(raw.shape[:3]):
        raw[time, position, channel] = digitize(
            expectation * (1 + time / 4 + channel / 2),
            microscope.camera,
            seed=100 * time + 10 * position + channel,
        )
    if mode == "stage":
        raw = raw[:, :, :, ::-1].copy()
    calibration = (
        {"read_noise": {"rms": microscope.camera.read_noise_e, "units": "electrons"}}
        if record_noise
        else None
    )
    pixel = microscope.pixel_size_um
    dataset = acquisition_factory(
        raw,
        name="pipeline",
        mode=mode,
        camera_offset=microscope.camera.offset_adu,
        camera_conversion=microscope.camera.electrons_per_adu,
        pixel_size_um=pixel,
        scan_axis_step_um=step,
        angle_deg=microscope.angle_deg,
        stage_positions_zxy=((2.3, -4.6, 1.15), (2.3, -4.6, 1.15 + pixel)),
        deconvolution=calibration,
        acquired_plane_count=raw.shape[-3],
        plane_period_s=simulation_parameters["pipeline"]["plane_period_s"],
    )
    psf_path = None
    if deconvolve:
        psf_path = tmp_path / "measured_psf.npy"
        np.save(psf_path, psf)
    result = pipeline_runner(
        dataset,
        deconvolve=deconvolve,
        illumination=illumination,
        crop_scan=crop,
        psf_path=psf_path,
    )
    factor = scan_upsample_from_metadata(dataset.metadata) if deconvolve else 1
    reconstruction_shape = (
        (expectation.shape[0] - 1) * factor + 1,
        *expectation.shape[1:],
    )
    support = (
        orthogonal_deskew(
            np.full(reconstruction_shape, 1, dtype=np.float32),
            distance=step / factor,
            pixel_size=pixel,
            theta=microscope.angle_deg,
            downsample_factor=1,
        )
        > 0
    )
    return SimpleNamespace(
        **vars(result),
        dataset=dataset,
        microscope=microscope,
        support=support,
    )


@pytest.fixture
def mosaic_acquisition(
    request,
    pipeline_optical_data,
    simulation_parameters,
    acquisition_factory,
    tmp_path,
    small_cpu_kernels,
):
    """Persist independent adjacent views of the same extended fluorescent object.

    Returns
    -------
    types.SimpleNamespace
        Raw qi2lab acquisition, physical camera expectation, PSF and the shared
        dimensions used to check scan chunks, deskew tiles and mosaic seams.
    """
    mode, step = request.param
    microscope, expectation, psf = pipeline_optical_data(step, mosaic=True)
    parameters = simulation_parameters["pipeline_mosaic"]
    height, width = parameters["camera_shape_yx"]
    offset = parameters["tile_offset_x_pixels"]
    raw = np.empty((1, 2, 1, expectation.shape[0], height, width), np.uint16)
    for position in range(2):
        start = position * offset
        raw[0, position, 0] = digitize(
            expectation[..., start : start + width],
            microscope.camera,
            seed=simulation_parameters["camera_seed"] + position,
        )
    if mode == "stage":
        raw = raw[:, :, :, ::-1].copy()
    pixel = microscope.pixel_size_um
    dataset = acquisition_factory(
        raw,
        name="mosaic",
        mode=mode,
        pixel_size_um=pixel,
        scan_axis_step_um=step,
        angle_deg=microscope.angle_deg,
        camera_offset=microscope.camera.offset_adu,
        camera_conversion=microscope.camera.electrons_per_adu,
        stage_positions_zxy=(
            (2.3, -4.6, 1.15),
            (2.3, -4.6, 1.15 + (offset + 1) * pixel),
        ),
        deconvolution={
            "read_noise": {"rms": microscope.camera.read_noise_e, "units": "electrons"}
        },
    )
    factor = scan_upsample_from_metadata(dataset.metadata)
    fine_expectation = pipeline_optical_data(step / factor, mosaic=True)[1]
    psf_path = tmp_path / "mosaic_psf.npy"
    np.save(psf_path, psf)
    return SimpleNamespace(
        dataset=dataset,
        microscope=microscope,
        expectation=expectation,
        psf=psf,
        psf_path=psf_path,
        parameters=parameters,
        reconstruction_expectation=fine_expectation,
    )
