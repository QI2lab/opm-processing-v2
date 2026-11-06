"""Shared qi2lab acquisitions for complete processing and fusion integration."""

import importlib
from dataclasses import replace
from functools import lru_cache, partial, wraps
from inspect import signature
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from opm_processing.dataio.deconvolution import scan_upsample_from_metadata
from opm_processing.imageprocessing.tilefusion import TileFusion
from tests.simulation.camera import digitize
from tests.simulation.forward_model import Acquisition, camera_psf, forward_model
from tests.simulation.scenes import prepare_scene


@pytest.fixture
def pipeline_geometry(deskew_geometric_support):
    """Define the independent physical aperture and padded orthogonal canvas."""

    def create(shape, microscope, *, scan_factor=1):
        """Cover the oblique scan planes without evaluating an image interpolator."""
        scans, rows, columns = shape
        fine_shape = ((scans - 1) * scan_factor + 1, rows, columns)
        step = microscope.scan_step_um / scan_factor
        pixel = microscope.pixel_size_um
        angle = np.deg2rad(microscope.angle_deg)
        output_shape = (
            int(np.ceil(rows * np.sin(angle))),
            (int(np.ceil(fine_shape[0] * step / pixel + rows * np.cos(angle))) + 3)
            // 4
            * 4,
            (columns + 3) // 4 * 4,
        )
        support = deskew_geometric_support(
            fine_shape,
            output_shape,
            {"theta": microscope.angle_deg, "distance": step, "pixel_size": pixel},
        )
        aperture = (support[..., None] > 0) & (np.arange(output_shape[2]) < columns)
        return SimpleNamespace(shape=output_shape, support=aperture)

    return create


@pytest.fixture
def pipeline_camera_accuracy(pipeline_geometry):
    """Score persisted CPU images against the original optical camera means."""
    from scipy.special import kl_div
    from yaozarrs import open_group

    from opm_processing.dataio.acquisition import inspect_acquisition
    from tests.simulation.forward_model import sensor_integral
    from tests.simulation.metrics import airy_rayleigh

    def check(
        result,
        means,
        microscope,
        mode,
        *,
        source_y_half_um,
        offsets_x=None,
        regions=None,
    ):
        """Reject omitted/doubled photon Jacobians and Rayleigh-scale displacement."""
        shape = means.shape[-3:]
        scans, rows, columns = shape
        pixel, step = microscope.pixel_size_um, microscope.scan_step_um
        angle = np.deg2rad(microscope.angle_deg)
        span = (scans - 1) * step
        local_origin = np.array((0, -span if mode == "stage" else 0, 0))
        center = np.array(
            (
                (rows - 1) * pixel * np.sin(angle) / 2,
                (span + (rows - 1) * pixel * np.cos(angle)) / 2,
                (columns - 1) * pixel / 2,
            )
        )
        geometry = pipeline_geometry(shape, microscope)
        native_axes = tuple(
            (np.arange(size) - (size - 1) / 2) * spacing
            for size, spacing in zip(shape, (step, pixel, pixel), strict=True)
        )
        sampler = replace(microscope, pixel_samples=1)
        support_axes = tuple(
            np.arange(size) * pixel - coordinate
            for size, coordinate in zip(geometry.shape, center, strict=True)
        )
        coverage = sensor_integral(
            geometry.support.astype(float),
            support_axes,
            shape,
            sampler,
            sensor_axes=native_axes,
            area_integral=False,
        )
        mask = coverage >= 1 - 32 * np.finfo(float).eps
        # A fixed region containing the original object avoids empty scan tails;
        # it is defined in physical coordinates before inspecting any output.
        camera_y = native_axes[0][:, None] + native_axes[1][None] * np.cos(angle)
        mask &= (np.abs(camera_y) <= source_y_half_um)[..., None]
        rayleigh, _ = airy_rayleigh(
            microscope.wavelength_um, microscope.numerical_aperture
        )
        displacement = int(np.ceil(rayleigh / pixel))
        mask[..., :displacement] = False
        mask[..., -displacement:] = False
        gain = 2 * pixel / step
        positions = means.shape[1]
        offsets_x = np.zeros(positions) if offsets_x is None else np.asarray(offsets_x)
        source_metadata = inspect_acquisition(
            result.fusion.processing_state.source_path()
        )
        stage_z, stage_scan, stage_x = source_metadata.stage_positions_zxy[0]
        anchor = np.array((-stage_z, -stage_scan, stage_x))
        fused = open_group(result.output)["0"].to_tensorstore().read().result()

        def score(image, origin, frame_origin, expected, response, selections):
            """Resample only; never apply a second optical PSF or fit intensity."""
            axes = tuple(
                coordinate + np.arange(size) * pixel - target
                for coordinate, size, target in zip(
                    origin, image.shape, frame_origin + center, strict=True
                )
            )
            prediction = (
                sensor_integral(
                    image / gain,
                    axes,
                    shape,
                    sampler,
                    sensor_axes=native_axes,
                    area_integral=False,
                )
                * response
            )
            alternatives = [prediction * gain, prediction / gain]
            alternatives.extend(
                np.roll(prediction, shift, axis=-1)
                for shift in (-displacement, displacement)
            )
            for selection in selections:
                selected = mask.copy()
                selected[..., : selection.start] = False
                selected[..., selection.stop :] = False
                assert selected.any() and expected[selected].sum() > 0
                rates = expected[selected]
                correct = kl_div(
                    rates, np.maximum(prediction[selected], np.finfo(float).tiny)
                ).sum()
                for alternative in alternatives:
                    incorrect = kl_div(
                        rates, np.maximum(alternative[selected], np.finfo(float).tiny)
                    ).sum()
                    assert correct < incorrect, (correct, incorrect, selection)

        for time, position, channel in np.ndindex(means.shape[:3]):
            expected = means[time, position, channel]
            response = 1 if result.field is None else result.field[0, channel]
            selections = [slice(0, columns)] if regions is None else regions[position]
            image = result.collection.arrays[position][time, channel].read().result()
            assert image.shape == geometry.shape
            score(
                image,
                np.asarray(result.collection.spatial_origins_zyx_um[position]),
                local_origin,
                expected,
                response,
                selections,
            )
            frame_origin = anchor + local_origin + np.array((0, 0, offsets_x[position]))
            score(
                fused[time, channel],
                np.asarray(result.fusion.offset_um),
                frame_origin,
                expected,
                response,
                selections,
            )

    return check


@pytest.fixture(autouse=True)
def integration_pipeline_records(request):
    """Require real producer inputs and durable processing results for integrations.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Current test's classification.

    Yields
    ------
    dict
        Actual producer sources and filesystem relocations of those acquisitions.
    """
    integration = request.node.get_closest_marker("integration") is not None
    tmp_path = request.getfixturevalue("tmp_path") if integration else None
    sources = []
    relocations = []
    processed_sources = set()
    generated_optics_sources = set()
    if integration:
        processing = importlib.import_module("opm_processing.process")
        roi_processing = importlib.import_module("opm_processing.process_roi")
        patches = request.getfixturevalue("monkeypatch")
        generate_psfs = processing.load_deconvolution_psfs

        def observe_psfs(acquisition, measured_paths=None, **kwargs):
            """Require actual generated kernels rather than descriptive output tags."""
            result = generate_psfs(acquisition, measured_paths, **kwargs)
            if measured_paths is None and result[1] is not None:
                generated_optics_sources.add(acquisition.path.resolve())
            return result

        patches.setattr(processing, "load_deconvolution_psfs", observe_psfs)

        def observe_processor(processor):
            """Record successful real processing without changing its arguments or output."""
            parameters = signature(processor)

            @wraps(processor)
            def execute(*args, **kwargs):
                """Run the complete production operation before recording its source."""
                arguments = parameters.bind(*args, **kwargs).arguments
                acquisition = arguments.get("acquisition")
                source = (
                    acquisition.path
                    if acquisition is not None
                    else arguments["root_path"]
                )
                result = processor(*args, **kwargs)
                from pathlib import Path

                processed_sources.add(Path(source).resolve())
                return result

            return execute

        for name in ("process_skewed", "process_projection"):
            observed = observe_processor(getattr(processing, name))
            patches.setattr(processing, name, observed)
            if hasattr(roi_processing, name):
                patches.setattr(roi_processing, name, observed)
    yield {"sources": sources, "relocations": relocations}
    if not integration:
        return
    if any(
        report is not None and (report.failed or report.skipped)
        for report in (
            getattr(request.node, "pipeline_report_setup", None),
            getattr(request.node, "pipeline_report_call", None),
        )
    ):
        return
    from opm_processing.dataio.position_collection import open_position_collection
    from opm_processing.dataio.processing_state import ProcessingState

    assert sources, "Integration never finalized a real opm-v2 acquisition"
    states = [
        ProcessingState.read(path) for path in tmp_path.rglob("*.processing.json")
    ]
    for source in sources:
        locations = {source.resolve()}
        for before, after in relocations:
            locations.update(
                after / location.relative_to(before)
                for location in tuple(locations)
                if location.is_relative_to(before)
            )
        assert locations & processed_sources, (
            f"Integration bypassed actual processing for {source}"
        )
        completed = []
        for state in states:
            if state.source_path() not in locations:
                continue
            for name, record in state.document["outputs"].items():
                path = state.path.parent / name
                if record.get("completed_tiles") and path.is_dir():
                    optics = record["configuration"].get("deconvolution")
                    if optics is not None:
                        assert locations & generated_optics_sources, (
                            "Integration bypassed production PSF generation"
                        )
                    collection = open_position_collection(path)
                    for array in collection.arrays:
                        pixels = array.read().result()
                        assert pixels.size and np.isfinite(pixels).all()
                    if record.get("roi_series"):
                        expected = {
                            (tile["time_index"], tile["position_index"])
                            for tile in record["roi_series"]
                        }
                    else:
                        selection = record["configuration"].get("selection", {})
                        times = selection.get("time") or (0, collection.shape[0])
                        positions = selection.get("position") or (
                            0,
                            collection.shape[1],
                        )
                        expected = set(product(range(*times), range(*positions)))
                    if state.completed_tiles(path) == expected:
                        completed.append(path)
        assert completed, f"Integration never completed a final datastore from {source}"


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
                decon_crop_scan=crop_scan,
                decon_verbose=0,
            )
        deskew_tiles = set()
        reconstructions = []
        deconvolution_runs = []
        chunks = []
        with monkeypatch.context() as execution:
            if deconvolve:
                solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
                reconstruct = solver.rlgc
                reconstruct_chunk = solver._rlgc_once

                def observe_chunk(image, psf, *args, **kwargs):
                    """Record each actual chunk only after its solver completes."""
                    result = reconstruct_chunk(image, psf, *args, **kwargs)
                    chunks.append(
                        {
                            "input_shape": image.shape,
                            "output_shape": result.shape,
                            "psf_shape": psf.shape,
                            "storage": kwargs.get("storage", "gpu"),
                        }
                    )
                    return result

                def observe_reconstruction(*args, **kwargs):
                    """Retain actual calibrated reconstructions for physical truth checks."""
                    if decon_storage is not None:
                        kwargs["storage"] = decon_storage
                    before = len(chunks)
                    result = reconstruct(*args, **kwargs)
                    reconstructions.append(result.copy())
                    deconvolution_runs.append(
                        {
                            "input_shape": args[0].shape,
                            "crop_scan": kwargs.get("crop_scan"),
                            "chunks": chunks[before:],
                        }
                    )
                    return result

                execution.setattr(solver, "rlgc", observe_reconstruction)
                execution.setattr(solver, "_rlgc_once", observe_chunk)
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
            deconvolution_runs=deconvolution_runs,
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
    def create(step, illumination=False, *, mosaic=False, scan_extent_factor=1):
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
            round(settings["scan_extent_um"] * scan_extent_factor / step) + 1,
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
    opm_v2_acquisition_factory,
    tmp_path,
    pipeline_runner,
    pipeline_geometry,
    pipeline_camera_accuracy,
    small_cpu_kernels,
):
    """Process independent repeat exposures with a known stage-placement error.

    Indirect parameters select scan mode, illumination, scan step, FFT scan
    chunk size. All deconvolution uses production-generated optical kernels.

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
    mode, illumination, step, crop = request.param
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
    pixel = microscope.pixel_size_um
    dataset = opm_v2_acquisition_factory(
        raw,
        name="pipeline",
        mode=mode,
        camera_offset=microscope.camera.offset_adu,
        camera_conversion=microscope.camera.electrons_per_adu,
        pixel_size_um=pixel,
        scan_axis_step_um=step,
        angle_deg=microscope.angle_deg,
        stage_positions_zxy=((2.3, -4.6, 1.15), (2.3, -4.6, 1.15 + pixel)),
        acquired_plane_count=raw.shape[-3],
        plane_period_s=simulation_parameters["pipeline"]["plane_period_s"],
    )
    result = pipeline_runner(
        dataset,
        deconvolve=deconvolve,
        illumination=illumination,
        crop_scan=crop,
    )
    factor = scan_upsample_from_metadata(dataset.metadata) if deconvolve else 1
    geometry = pipeline_geometry(expectation.shape, microscope, scan_factor=factor)
    means = np.broadcast_to(
        expectation[None, None, None]
        * (1 + np.arange(2)[:, None, None] / 4 + np.arange(2)[None, None] / 2)[
            ..., None, None, None
        ],
        raw.shape,
    )
    if not deconvolve:
        pipeline_camera_accuracy(
            result,
            means,
            microscope,
            mode,
            source_y_half_um=simulation_parameters["specimens"]["points"][
                "half_extent_zyx_um"
            ][1]
            - microscope.psf_half_extent_zyx_um[1],
        )
    return SimpleNamespace(
        **vars(result),
        dataset=dataset,
        microscope=microscope,
        support=geometry.support,
        expectation=expectation,
        psf=psf,
    )


@pytest.fixture
def mosaic_acquisition(
    request,
    pipeline_optical_data,
    simulation_parameters,
    opm_v2_acquisition_factory,
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
    mode, step, *encoder_errors = request.param
    scan_encoder_error_um = encoder_errors[0] if encoder_errors else 0.0
    scan_extent_factor = encoder_errors[1] if len(encoder_errors) > 1 else 1
    microscope, expectation, psf = pipeline_optical_data(
        step, mosaic=True, scan_extent_factor=scan_extent_factor
    )
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
    dataset = opm_v2_acquisition_factory(
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
            (2.3, -4.6 + scan_encoder_error_um, 1.15 + (offset + 1) * pixel),
        ),
    )
    factor = scan_upsample_from_metadata(dataset.metadata)
    fine_expectation = pipeline_optical_data(
        step / factor, mosaic=True, scan_extent_factor=scan_extent_factor
    )[1]
    return SimpleNamespace(
        dataset=dataset,
        microscope=microscope,
        expectation=expectation,
        psf=psf,
        parameters=parameters,
        reconstruction_expectation=fine_expectation,
    )
