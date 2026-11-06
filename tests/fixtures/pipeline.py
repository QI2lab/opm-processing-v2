"""Shared qi2lab acquisitions for complete processing and fusion integration."""

import importlib
from dataclasses import dataclass, replace
from functools import lru_cache, partial, wraps
from inspect import signature
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from opm_processing.dataio.deconvolution import scan_upsample_from_metadata
from opm_processing.imageprocessing.tilefusion import TileFusion
from tests.simulation.camera import digitize
from tests.simulation.forward_model import (
    Acquisition,
    Microscope,
    camera_psf,
    forward_model,
)
from tests.simulation.scenes import OpticalScene, prepare_scene


@dataclass(frozen=True)
class PipelineOpticalObject:
    """One physical emission dose, shared unchanged by every acquisition grid."""

    microscope: Microscope
    axes: tuple[np.ndarray, ...]
    density: np.ndarray
    source_density: np.ndarray
    ground_truth: OpticalScene

    def __post_init__(self):
        """Protect cached optical truth from individual processing scenarios."""
        for array in (*self.axes, self.density, self.source_density):
            array.setflags(write=False)


@dataclass(frozen=True)
class PipelineOpticalData:
    """Immutable camera expectations and inverse kernel for one sensor grid."""

    microscope: Microscope
    expectation: np.ndarray
    source_expectation: np.ndarray
    psf: np.ndarray
    optical_object: PipelineOpticalObject

    def __post_init__(self):
        """Keep repeated camera draws independent of cached optical data."""
        for array in (self.expectation, self.source_expectation, self.psf):
            array.setflags(write=False)


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
    """Bound persisted camera predictions by original optical means and detector residuals."""
    from scipy.ndimage import shift as shift_bound
    from yaozarrs import open_group

    from opm_processing.dataio.acquisition import (
        inspect_acquisition,
        open_acquisition_datastore,
    )
    from opm_processing.dataio.ngff import SPATIAL_DECIMALS
    from tests.simulation.camera_likelihood import (
        SamplingEnvelope,
        camera_sampling_accuracy,
        camera_sampling_envelope,
        sampling_stencil_footprint,
        sampling_stencil_radii,
    )
    from tests.simulation.metrics import camera_volume_samples

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
        """Check fixed source/background apertures without fitting photons or spatial shifts."""
        shape = means.shape[-3:]
        scans, rows, columns = shape
        pixel, step = microscope.pixel_size_um, microscope.scan_step_um
        angle = np.deg2rad(microscope.angle_deg)
        span = (scans - 1) * step
        local_origin = np.array((0, -span if mode == "stage" else 0, 0))
        spacing = result.collection.voxel_size_um
        geometry = pipeline_geometry(shape, microscope)
        coverage = camera_volume_samples(
            geometry.support.astype(float),
            local_origin,
            spacing,
            local_origin,
            shape,
            microscope,
        )
        mask = coverage >= 1 - 32 * np.finfo(float).eps
        native_scan = (np.arange(scans) - (scans - 1) / 2) * step
        native_row = (np.arange(rows) - (rows - 1) / 2) * pixel
        camera_y = native_scan[:, None] + native_row[None] * np.cos(angle)
        # The physical analysis window comes from the original object and PSF
        # support. No reconstructed value selects the scored samples.
        mask &= (np.abs(camera_y) <= source_y_half_um)[..., None]
        positions = means.shape[1]
        offsets_x = np.zeros(positions) if offsets_x is None else np.asarray(offsets_x)
        source_metadata = inspect_acquisition(
            result.fusion.processing_state.source_path()
        )
        original = open_acquisition_datastore(source_metadata).read().result()
        if source_metadata.scan_axis_reversed:
            original = original[:, :, :, ::-1]
        stage_z, stage_scan, stage_x = source_metadata.stage_positions_zxy[0]
        anchor = np.array((-stage_z, -stage_scan, stage_x))
        fused = open_group(result.output)["0"].to_tensorstore().read().result()
        gain = 2 * pixel / step
        radii = sampling_stencil_radii(microscope, spacing)
        # OME-NGFF origins are serialized to the library's declared precision.
        origin_error = 0.5 * 10**-SPATIAL_DECIMALS
        footprint = sampling_stencil_footprint(
            microscope, spacing, origin_y_error_um=origin_error
        )
        grid_offsets = (
            np.asarray(result.fusion._tile_positions) - result.fusion.offset_um
        ) / spacing
        integer_grid = np.allclose(
            grid_offsets, np.rint(grid_offsets), rtol=0, atol=1e-8
        )
        # Fractional fusion placement adds one further orthogonal-grid stencil.
        fused_radii = sampling_stencil_radii(
            microscope, np.asarray(spacing) * (1 if integer_grid else 2)
        )
        fused_footprint = sampling_stencil_footprint(
            microscope,
            np.asarray(spacing) * (1 if integer_grid else 2),
            origin_y_error_um=origin_error,
        )

        def response_for(channel):
            return np.broadcast_to(
                1 if result.field is None else result.field[0, channel], shape
            )

        def score(
            image, origin, frame_origin, expected, response, envelope, selections
        ):
            prediction = (
                camera_volume_samples(
                    image / gain,
                    origin,
                    spacing,
                    frame_origin,
                    shape,
                    microscope,
                )
                * response
            )
            for selection in selections:
                selected = mask.copy()
                selected[..., : selection.start] = False
                selected[..., selection.stop :] = False
                measured = camera_sampling_accuracy(
                    prediction, expected, envelope, selected
                )
                assert measured.accepted, measured

        for time in range(means.shape[0]):
            for channel in range(means.shape[2]):
                response = response_for(channel)
                source_bounds = [
                    camera_sampling_envelope(
                        means[time, position, channel],
                        original[time, position, channel],
                        microscope.camera,
                        radii=fused_radii,
                        footprint=fused_footprint,
                        response=response,
                        views=positions,
                    )
                    for position in range(positions)
                ]
                for position in range(positions):
                    expected = means[time, position, channel]
                    selections = (
                        [slice(0, columns)] if regions is None else regions[position]
                    )
                    image = (
                        result.collection.arrays[position][time, channel]
                        .read()
                        .result()
                    )
                    assert image.shape == geometry.shape
                    own_bounds = camera_sampling_envelope(
                        expected,
                        original[time, position, channel],
                        microscope.camera,
                        radii=radii,
                        footprint=footprint,
                        response=response,
                    )
                    score(
                        image,
                        np.asarray(result.collection.spatial_origins_zyx_um[position]),
                        local_origin,
                        expected,
                        response,
                        own_bounds,
                        selections,
                    )
                    # Fusion is a positive mixture of ALL independently exposed
                    # views. Align interval bounds in corrected-photon units,
                    # then convert back with the target detector response.
                    low = np.full(shape, np.nan)
                    high = np.full(shape, np.nan)
                    rounding = np.full(shape, np.nan)
                    for source_position, bounds in enumerate(source_bounds):
                        delta = (
                            offsets_x[position] - offsets_x[source_position]
                        ) / pixel
                        assert np.isclose(delta, np.rint(delta), rtol=0, atol=1e-8)
                        shift = (0, 0, -int(np.rint(delta)))
                        aligned = [
                            shift_bound(
                                array / response,
                                shift,
                                order=0,
                                mode="constant",
                                cval=np.nan,
                                prefilter=False,
                            )
                            for array in (bounds.lower, bounds.upper, bounds.roundoff)
                        ]
                        low = np.fmin(low, aligned[0])
                        high = np.fmax(high, aligned[1])
                        rounding = np.fmax(rounding, aligned[2])
                    fused_bounds = SamplingEnvelope(
                        low * response, high * response, rounding * response
                    )
                    frame_origin = (
                        anchor + local_origin + np.array((0, 0, offsets_x[position]))
                    )
                    score(
                        fused[time, channel],
                        np.asarray(result.fusion.offset_um),
                        frame_origin,
                        expected,
                        response,
                        fused_bounds,
                        selections,
                    )

    return check


@pytest.fixture
def pipeline_inverse_accuracy():
    """Score actual inverse solutions against the original physical camera means."""
    from tests.simulation.metrics import (
        camera_flux_accuracy,
        camera_noise_accuracy,
        camera_object_apertures,
        camera_prediction,
    )

    def check(result, means, microscope, ground_truth, *, offsets_x=None):
        """Use captured real kernels and fixed source, background and detector apertures."""
        assert len(result.reconstructions) == len(result.deconvolution_runs)
        assert len(result.reconstructions) == np.prod(means.shape[:3])
        offsets = np.zeros(means.shape[1]) if offsets_x is None else offsets_x
        for index, density, run in zip(
            np.ndindex(*means.shape[:3]),
            result.reconstructions,
            result.deconvolution_runs,
            strict=True,
        ):
            expected = means[index]
            kernel = np.asarray(run["psf"], np.float64)
            if run["normalize_psf"]:
                kernel = kernel / kernel.sum(dtype=np.float64)
            boundary = "constant" if run["boundary"] == "zero" else run["boundary"]
            prediction = camera_prediction(
                density,
                kernel,
                boundary=boundary,
                response=run["detector_response"],
            )[:: run["scan_upsample_factor"]]
            assert prediction.shape == expected.shape
            apertures = camera_object_apertures(
                ground_truth,
                microscope,
                expected.shape,
                camera_offset_zyx_um=(0, 0, offsets[index[1]]),
            )
            for aperture in apertures:
                accuracy = camera_noise_accuracy(
                    prediction, expected, microscope.camera, aperture
                )
                assert accuracy.accepted, (index, accuracy)
                assert camera_flux_accuracy(
                    prediction, expected, microscope.camera, aperture
                ), index

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
                            "psf": np.asarray(args[1]).copy(),
                            "normalize_psf": kwargs.get("normalize_psf", True),
                            "boundary": kwargs.get("boundary", "reflect"),
                            "scan_upsample_factor": kwargs.get(
                                "scan_upsample_factor", 1
                            ),
                            "detector_response": np.asarray(
                                kwargs.get("detector_response", 1)
                            ).copy(),
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


@pytest.fixture
def pipeline_illumination_calibration(
    opm_v2_acquisition_factory,
    uniform_optical_calibration,
    invoke_cli,
):
    """Fit an identifiable real calibration acquisition before processing tissue."""
    from opm_processing.dataio.acquisition import acquisition_stem
    from opm_processing.process import read_current_flatfield, write_flatfield
    from tests.simulation.camera_likelihood import (
        SamplingEnvelope,
        camera_sampling_accuracy,
        camera_sampling_envelope,
    )

    def calibrate(dataset, microscope, shape, mode):
        """Use known uniform emission, real camera draws and the production BaSiC worker."""
        shape = tuple(shape)
        stem = acquisition_stem(dataset.path)
        working_width = shape[2] // 2
        assert working_width > 1
        # align_corners=True places the half-size samples at this native pitch.
        # The antialiased bilinear reduction and bilinear expansion each span
        # one such pitch; round outward before fitting any field.
        native_reach = int(np.ceil(2 * (shape[2] - 1) / (working_width - 1)))
        radii = (shape[0] - 1, shape[1] - 1, native_reach)
        # The calibration's physical dose is declared before any camera draw.
        # All channels share the same 637-nm optics in these pipeline cases.
        means = uniform_optical_calibration(
            microscope,
            shape,
            "edge-taper",
            (5000 / microscope.quantum_efficiency, 0),
        )
        channels = dataset.metadata.shape[2]
        levels = (0.75, 1.25)
        raw = np.empty((1, len(levels), channels, *shape), np.uint16)
        for position, channel in np.ndindex(raw.shape[1:3]):
            raw[0, position, channel] = digitize(
                means * levels[position] * (1 + channel / 4),
                microscope.camera,
                seed=903 + 10 * position + channel,
            )
        assert raw.max() < np.iinfo(raw.dtype).max
        canonical = raw.copy()
        if mode == "stage":
            raw = raw[:, :, :, ::-1].copy()
        calibration = opm_v2_acquisition_factory(
            raw,
            name=f"illumination-calibration/{stem}_calibration",
            mode=mode,
            camera_offset=microscope.camera.offset_adu,
            camera_conversion=microscope.camera.electrons_per_adu,
            pixel_size_um=microscope.pixel_size_um,
            scan_axis_step_um=microscope.scan_step_um,
            angle_deg=microscope.angle_deg,
            stage_positions_zxy=((0, 0, 0),) * len(levels),
        )
        output = calibration.path.parent / "processed"
        invoke_cli(
            "process",
            calibration.path,
            output=output,
            deconvolve=False,
            flatfield_correction=True,
            save_float32=True,
            z_downsample_level=1,
            max_projection=False,
            create_fused_max_projection=False,
            write_fused_max_projection_tiff=False,
        )
        field = read_current_flatfield(
            output / f"{acquisition_stem(calibration.path)}_flatfield.ome.tif",
            (1, channels, *shape[-2:]),
        )
        assert field is not None
        # BaSiC's working image has half the native detector dimensions.
        # Permit the declared reduction/expansion sampling aperture, with the
        # original camera residuals as the absolute noise allowance. This
        # criterion is specified before fitting and cannot cancel a wrong
        # field against the subsequent tissue correction. It is an accuracy
        # requirement for the nonlinear estimator, not a claim that BaSiC's
        # optimization is itself a local convex interpolation.
        for channel in range(channels):
            truth = means * (1 + channel / 4)
            view_bounds = [
                camera_sampling_envelope(
                    truth * level,
                    canonical[0, position, channel],
                    microscope.camera,
                    radii=radii,
                    response=level,
                    views=len(levels),
                )
                for position, level in enumerate(levels)
            ]
            envelope = SamplingEnvelope(
                np.minimum.reduce(
                    [
                        bound.lower / level
                        for bound, level in zip(view_bounds, levels, strict=True)
                    ]
                ),
                np.maximum.reduce(
                    [
                        bound.upper / level
                        for bound, level in zip(view_bounds, levels, strict=True)
                    ]
                ),
                np.maximum.reduce(
                    [
                        bound.roundoff / level
                        for bound, level in zip(view_bounds, levels, strict=True)
                    ]
                ),
            )
            prediction = np.broadcast_to(field[0, channel] * truth.mean(), truth.shape)
            mask = np.ones(truth.shape, bool)
            result = camera_sampling_accuracy(prediction, truth, envelope, mask)
            assert result.accepted, result
            # The prescribed illumination is measurable: uncorrected unit
            # sensitivity must fail the same absolute aperture, not merely
            # score a little worse than the fitted result.
            uncorrected = camera_sampling_accuracy(
                np.full_like(truth, truth.mean()), truth, envelope, mask
            )
            assert not uncorrected.accepted, uncorrected
        write_flatfield(
            dataset.path.parent / f"{stem}_flatfield.ome.tif",
            field,
            microscope.pixel_size_um,
        )
        return field

    return calibrate


@pytest.fixture(scope="session")
def pipeline_optical_data(
    microscope_parameters, simulation_parameters, acquisition_metadata
):
    """Cache production optics and camera means for repeated fields and mosaics.

    Returns
    -------
    callable
        Return immutable camera means, source-only means and production optics
        for a scan step, illumination selection and extended-mosaic flag.
    """

    @lru_cache(maxsize=4)
    def dosed_object(illumination, mosaic, scan_extent_factor):
        """Set exposure once, before querying coarse or refined sensor grids."""
        reference_step = simulation_parameters["microscope"]["scan_step_um"]
        microscope = replace(microscope_parameters, scan_step_um=reference_step)
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
            round(settings["scan_extent_um"] * scan_extent_factor / reference_step) + 1,
            height,
            width + settings.get("tile_offset_x_pixels", 0),
        )
        acquisition = Acquisition(
            scene.axes,
            shape,
            illumination_profile="edge-taper" if illumination else "uniform",
            # A declared high-background calibration exposure makes photon-gain
            # and stripe accuracy measurable above the original detector noise.
            # This dose is fixed before digitization and never refitted to a
            # coarse grid, a noisy observation or a reconstruction.
            camera_photon_budget=(3000, 4000),
            output="object",
        )
        dose = forward_model(scene.density, acquisition, microscope)
        source = forward_model(
            scene.density,
            replace(acquisition, camera_photon_budget=(3000, 0)),
            microscope,
        )
        return PipelineOpticalObject(microscope, scene.axes, dose, source, scene)

    @lru_cache(maxsize=6)
    def create(step, illumination=False, *, mosaic=False, scan_extent_factor=1):
        """Sample one unchanged physical emission field at the requested scan pitch."""
        optical_object = dosed_object(illumination, mosaic, scan_extent_factor)
        microscope = replace(optical_object.microscope, scan_step_um=step)
        settings = simulation_parameters["pipeline_mosaic" if mosaic else "pipeline"]
        height, width = settings["camera_shape_yx"]
        shape = (
            round(settings["scan_extent_um"] * scan_extent_factor / step) + 1,
            height,
            width + settings.get("tile_offset_x_pixels", 0),
        )
        acquisition = Acquisition(optical_object.axes, shape, output="expectation")
        expected = forward_model(optical_object.density, acquisition, microscope)
        source = forward_model(optical_object.source_density, acquisition, microscope)
        factor = scan_upsample_from_metadata(
            replace(acquisition_metadata, scan_axis_step_um=step)
        )
        psf, _ = camera_psf(replace(microscope, scan_step_um=step / factor))
        return PipelineOpticalData(
            microscope, expected, source, psf.astype(np.float32), optical_object
        )

    return create


@pytest.fixture
def pipeline_source_image(pipeline_geometry):
    """Project only the original physical emitters onto a laboratory camera grid."""

    def project(data, *, scan_factor=1):
        """Identify bright physical seams without deskewing production input pixels."""
        microscope = data.microscope
        geometry = pipeline_geometry(
            data.expectation.shape, microscope, scan_factor=scan_factor
        )
        scans, rows, columns = data.expectation.shape
        pixel = microscope.pixel_size_um
        angle = np.deg2rad(microscope.angle_deg)
        center = (
            (rows - 1) * pixel * np.sin(angle) / 2,
            ((scans - 1) * microscope.scan_step_um + (rows - 1) * pixel * np.cos(angle))
            / 2,
            (columns - 1) * pixel / 2,
        )
        axes = tuple(
            np.arange(size) * pixel - origin
            for size, origin in zip(geometry.shape, center, strict=True)
        )
        return forward_model(
            data.optical_object.source_density,
            Acquisition(
                data.optical_object.axes,
                geometry.shape,
                sensor_axes=axes,
                geometry="cartesian",
                output="expectation",
            ),
            microscope,
        )

    return project


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
    pipeline_illumination_calibration,
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
    optical_data = pipeline_optical_data(step, illumination)
    microscope, expectation, psf = (
        optical_data.microscope,
        optical_data.expectation,
        optical_data.psf,
    )
    raw = np.empty((2, 2, 2, *expectation.shape), dtype=np.uint16)
    for time, position, channel in np.ndindex(raw.shape[:3]):
        raw[time, position, channel] = digitize(
            expectation * (1 + time / 4 + channel / 2),
            microscope.camera,
            seed=100 * time + 10 * position + channel,
        )
    if mode == "stage":
        raw = raw[:, :, :, ::-1].copy()
    assert raw.max() < 2**microscope.camera.bit_depth - 1
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
    if illumination:
        pipeline_illumination_calibration(dataset, microscope, expectation.shape, mode)
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
        camera_means=means,
        source_expectation=optical_data.source_expectation,
        psf=psf,
        optical_data=optical_data,
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
    optical_data = pipeline_optical_data(
        step, mosaic=True, scan_extent_factor=scan_extent_factor
    )
    microscope, expectation, psf = (
        optical_data.microscope,
        optical_data.expectation,
        optical_data.psf,
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
    assert raw.max() < 2**microscope.camera.bit_depth - 1
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
    fine_data = pipeline_optical_data(
        step / factor, mosaic=True, scan_extent_factor=scan_extent_factor
    )
    return SimpleNamespace(
        dataset=dataset,
        microscope=microscope,
        expectation=expectation,
        source_expectation=optical_data.source_expectation,
        psf=psf,
        parameters=parameters,
        reconstruction_expectation=fine_data.expectation,
        reconstruction_source_expectation=fine_data.source_expectation,
        optical_data=optical_data,
    )
