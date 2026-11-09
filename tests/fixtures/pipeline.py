"""Shared qi2lab acquisitions for complete processing and fusion integration."""

import importlib
from dataclasses import dataclass, replace
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest

from opm_processing.dataio.deconvolution import scan_upsample_from_metadata
from opm_processing.imageprocessing.opmtools import (
    deskew_coverage,
    deskew_shape_estimator,
)
from opm_processing.imageprocessing.tilefusion import (
    DEFAULT_REGISTRATION_DOWNSAMPLE,
    TileFusion,
)
from tests.simulation.camera import digitize
from tests.simulation.forward_model import Microscope
from tests.simulation.scenes import OpticalScene


@dataclass(frozen=True)
class PipelineOpticalData:
    """Immutable camera expectations and source truth for one sensor grid."""

    microscope: Microscope
    expectation: np.ndarray
    source_expectation: np.ndarray | None
    ground_truth: OpticalScene
    native_scan_planes: int
    first_native_scan_index: int


@pytest.fixture
def pipeline_geometry():
    """Use production geometry for the physical aperture and orthogonal canvas."""

    def create(shape, microscope, *, scan_factor=1):
        """Cover the oblique scan planes without evaluating an image interpolator."""
        scans, rows, columns = shape
        fine_shape = ((scans - 1) * scan_factor + 1, rows, columns)
        step = microscope.scan_step_um / scan_factor
        pixel = microscope.pixel_size_um
        output_shape, _, _, _ = deskew_shape_estimator(
            fine_shape, microscope.angle_deg, step, pixel, crop_after_deskew=False
        )
        output_shape = tuple(output_shape)
        support = deskew_coverage(
            fine_shape,
            output_shape,
            theta=microscope.angle_deg,
            distance=step,
            pixel_size=pixel,
        )
        aperture = (support[..., None] > 0) & (np.arange(output_shape[2]) < columns)
        return SimpleNamespace(shape=output_shape, support=aperture)

    return create


@pytest.fixture
def pipeline_camera_accuracy(pipeline_geometry):
    """Bound persisted camera predictions by original optical means and detector residuals."""
    from yaozarrs import open_group

    from opm_processing.dataio.acquisition import (
        inspect_acquisition,
        open_acquisition_datastore,
    )
    from opm_processing.dataio.ngff import SPATIAL_DECIMALS
    from tests.simulation.camera_likelihood import (
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
        scan_center_offset_um=0.0,
    ):
        """Check fixed source/background apertures without fitting photons or spatial shifts."""
        shape = means.shape[-3:]
        scans, rows = shape[:2]
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
        camera_y = (
            native_scan[:, None]
            + native_row[None] * np.cos(angle)
            + scan_center_offset_um
        )
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

        def score(image, origin, frame_origin, response, envelope):
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
            # Noisy ADC samples may be zero at positive true rates. Their
            # certified sampling/noise intervals, including zeros, apply.
            assert mask.any()
            assert np.isfinite(prediction[mask]).all()
            assert np.all(prediction[mask] >= envelope.lower[mask])
            assert np.all(prediction[mask] <= envelope.upper[mask])

        for time in range(means.shape[0]):
            for channel in range(means.shape[2]):
                response = np.broadcast_to(
                    1 if result.field is None else result.field[0, channel], shape
                )
                for position in range(positions):
                    expected = means[time, position, channel]
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
                        response,
                        own_bounds,
                    )
                    native_offsets_x = (offsets_x - offsets_x[position]) / pixel
                    np.testing.assert_allclose(
                        native_offsets_x, np.rint(native_offsets_x), rtol=0, atol=1e-8
                    )
                    fused_bounds = camera_sampling_envelope(
                        means[time, :, channel],
                        original[time, :, channel],
                        microscope.camera,
                        radii=fused_radii,
                        footprint=fused_footprint,
                        response=response,
                        views=positions,
                        view_offsets_syx=np.outer(np.rint(native_offsets_x), (0, 0, 1)),
                    )
                    frame_origin = (
                        anchor + local_origin + np.array((0, 0, offsets_x[position]))
                    )
                    score(
                        fused[time, channel],
                        np.asarray(result.fusion.offset_um),
                        frame_origin,
                        response,
                        fused_bounds,
                    )

    return check


@pytest.fixture
def pipeline_runner(
    monkeypatch, processing_options, invoke_cli, production_forward_predictions
):
    """Run real processing and fusion with shared storage and worker boundaries.

    Returns
    -------
    callable
        Process a raw acquisition into an independently selected output directory,
        optionally configuring RLGC and fusion block sizes. Deskew uses the
        normal production tile planner.
    """
    from opm_processing.dataio.acquisition import acquisition_stem
    from opm_processing.dataio.position_collection import open_position_collection
    from opm_processing.imageprocessing import opmtools

    module = importlib.import_module("opm_processing.process")
    fuse = importlib.import_module("opm_processing.fuse")

    def run(
        dataset,
        *,
        deconvolve=False,
        illumination=False,
        crop_scan=None,
        output=None,
        fusion_chunk=None,
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
        deskew_runs = []
        reconstructions = []
        deconvolution_runs = []
        with monkeypatch.context() as execution:
            if deconvolve:
                solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
                reconstruct = solver.rlgc

                def observe_reconstruction(*args, **kwargs):
                    """Retain actual calibrated reconstructions for physical truth checks."""
                    result = reconstruct(*args, **kwargs)
                    reconstructions.append(result.copy())
                    deconvolution_runs.append(
                        {
                            "input_shape": args[0].shape,
                            "crop_scan": kwargs.get("crop_scan"),
                            "forward": production_forward_predictions[-1],
                        }
                    )
                    return result

                execution.setattr(solver, "rlgc", observe_reconstruction)
            interpolate = opmtools.interpolate_rows
            deskew = module.orthogonal_deskew

            def record_tile(*args):
                """Record actual global output tiles while running the real kernel."""
                deskew_tiles.add((args[-1], args[1].shape))
                return interpolate(*args)

            def observe_deskew(*args, **kwargs):
                """Observe the production planner without replacing its choices."""
                result = deskew(*args, **kwargs)
                deskew_runs.append(
                    {
                        "output_shape": result.shape,
                        "tile_shape": kwargs["tile_shape"],
                    }
                )
                return result

            execution.setattr(opmtools, "interpolate_rows", record_tile)
            execution.setattr(module, "orthogonal_deskew", observe_deskew)
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
                registration_channel=dataset.raw_data.shape[2] - 1,
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
            deskew_runs=deskew_runs,
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
            )
            prediction = np.broadcast_to(field[0, channel] * truth.mean(), truth.shape)
            assert np.isfinite(prediction).all()
            assert np.all(prediction >= envelope.lower)
            assert np.all(prediction <= envelope.upper)
            # The prescribed illumination is measurable: uncorrected unit
            # sensitivity must fail the same absolute aperture, not merely
            # score a little worse than the fitted result.
            uncorrected = np.full_like(truth, truth.mean())
            assert np.any(uncorrected < envelope.lower) or np.any(
                uncorrected > envelope.upper
            )
        write_flatfield(
            dataset.path.parent / f"{stem}_flatfield.ome.tif",
            field,
            microscope.pixel_size_um,
        )
        return field

    return calibrate


@pytest.fixture(scope="session")
def pipeline_optical_data(
    microscope_parameters, simulation_parameters, simulated_camera
):
    """Reuse native optical camera fields for repeated exposures and mosaics.

    Returns
    -------
    callable
        Return immutable camera means and original optical truth, with source-only
        fields for extended mosaics.
    """

    @lru_cache(maxsize=2)
    def native_data(illumination, mosaic, tiled, kind):
        """Set exposure once before selecting acquired scan planes."""
        reference_step = simulation_parameters["microscope"]["scan_step_um"]
        kind = kind or ("line-pairs" if mosaic else "points")
        height, width = simulation_parameters["camera_shape_yx"]
        scans = simulation_parameters[
            "tiled_scan_planes" if tiled else "whole_scan_planes"
        ]
        shape = (
            scans,
            height,
            width
            + (
                simulation_parameters["pipeline_mosaic"]["tile_offset_x_pixels"]
                if mosaic
                else 0
            ),
        )
        options = {
            "kind": kind,
            "large": True,
            "integration": True,
            "illumination_profile": "edge-taper" if illumination else "uniform",
            "step": reference_step,
            "conversion": microscope_parameters.camera.electrons_per_adu,
        }
        # Both fields use the same source exposure, fixed before camera noise.
        data = simulated_camera(shape, budget=(3000, 4000), **options)
        source = (
            simulated_camera(shape, budget=(3000, 0), **options).expectation
            if mosaic
            else None
        )
        return data, source

    @lru_cache(maxsize=2)
    def create(
        step, illumination=False, *, mosaic=False, tiled=False, kind=None, phase=0
    ):
        """Sample one unchanged physical emission field at the requested scan pitch."""
        data, native_source = native_data(illumination, mosaic, tiled, kind)
        microscope = replace(data.microscope, scan_step_um=step)
        native_step = simulation_parameters["microscope"]["scan_step_um"]
        factor = round(step / native_step)
        assert factor >= 1 and np.isclose(step, factor * native_step)
        assert 0 <= phase < factor
        expected = data.expectation[phase::factor]
        source = None if native_source is None else native_source[phase::factor]
        return PipelineOpticalData(
            microscope,
            expected,
            source,
            data.ground_truth,
            data.expectation.shape[0],
            phase,
        )

    return create


@pytest.fixture
def pipeline_case(
    request,
    deconvolve,
    pipeline_optical_data,
    simulation_parameters,
    opm_v2_acquisition_factory,
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
    noise. Position 1 reports an erroneous lateral shift of one registration-grid
    pixel (five native detector pixels).
    BaSiC runs in its production worker process. Camera calibration, RLGC,
    orthogonal interpolation, registration and fusion execute normally, with
    native TensorStore storage in pytest temporary directories.
    """
    mode, illumination, step, crop = request.param
    native_step = simulation_parameters["microscope"]["scan_step_um"]
    native_scans = simulation_parameters[
        "tiled_scan_planes" if crop is not None else "whole_scan_planes"
    ]
    phase = (native_scans - 1) % round(step / native_step) if mode == "stage" else 0
    optical_data = pipeline_optical_data(
        step,
        illumination,
        tiled=crop is not None,
        kind="line-pairs" if deconvolve else None,
        phase=phase,
    )
    microscope, expectation = (
        optical_data.microscope,
        optical_data.expectation,
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
    registration_error_pixels = DEFAULT_REGISTRATION_DOWNSAMPLE[-1]
    native_step = simulation_parameters["microscope"]["scan_step_um"]
    missing_span = (optical_data.native_scan_planes - 1) * native_step - (
        raw.shape[-3] - 1
    ) * step
    stage_scan = -4.6
    dataset = opm_v2_acquisition_factory(
        raw,
        name="pipeline",
        mode=mode,
        camera_offset=microscope.camera.offset_adu,
        camera_conversion=microscope.camera.electrons_per_adu,
        camera_read_noise_e=microscope.camera.read_noise_e,
        pixel_size_um=pixel,
        scan_axis_step_um=step,
        angle_deg=microscope.angle_deg,
        stage_positions_zxy=(
            (2.3, stage_scan, 1.15),
            (2.3, stage_scan, 1.15 + registration_error_pixels * pixel),
        ),
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
    if not deconvolve:
        means = np.broadcast_to(
            expectation[None, None, None]
            * (1 + np.arange(2)[:, None, None] / 4 + np.arange(2)[None, None] / 2)[
                ..., None, None, None
            ],
            raw.shape,
        )
        pipeline_camera_accuracy(
            result,
            means,
            microscope,
            mode,
            source_y_half_um=simulation_parameters["specimens"]["points"][
                "half_extent_zyx_um"
            ][1]
            - microscope.psf_half_extent_zyx_um[1],
            scan_center_offset_um=phase * native_step - missing_span / 2,
        )
    return SimpleNamespace(
        **vars(result),
        dataset=dataset,
        microscope=microscope,
        support=geometry.support,
        expectation=expectation,
        optical_data=optical_data,
        registration_error_pixels=registration_error_pixels,
    )


@pytest.fixture
def large_noisy_acquisition(
    request,
    noisy_extended_scene,
    simulation_parameters,
    opm_v2_acquisition_factory,
    small_cpu_kernels,
):
    """Write independent full-field views of a shared noisy extended specimen."""
    mode, factor = request.param
    microscope, expected, raw, source, _dark = noisy_extended_scene
    native_scan_planes = expected.shape[0]
    phase = (native_scan_planes - 1) % factor if mode == "stage" else 0
    microscope = replace(microscope, scan_step_um=microscope.scan_step_um * factor)
    expected = expected[phase::factor]
    pixels = np.stack(
        (
            raw[phase::factor],
            digitize(
                expected,
                microscope.camera,
                seed=simulation_parameters["camera_seed"] + 1,
            ),
        )
    )[None, :, None]
    if mode == "stage":
        pixels = pixels[:, :, :, ::-1].copy()
    pixel = microscope.pixel_size_um
    registration_error_pixels = DEFAULT_REGISTRATION_DOWNSAMPLE[-1]
    stage_scan = -4.6
    dataset = opm_v2_acquisition_factory(
        pixels,
        name="large_scene",
        mode=mode,
        pixel_size_um=pixel,
        scan_axis_step_um=microscope.scan_step_um,
        angle_deg=microscope.angle_deg,
        camera_offset=microscope.camera.offset_adu,
        camera_conversion=microscope.camera.electrons_per_adu,
        camera_read_noise_e=microscope.camera.read_noise_e,
        stage_positions_zxy=(
            (2.3, stage_scan, 1.15),
            (2.3, stage_scan, 1.15 + registration_error_pixels * pixel),
        ),
    )
    microscope = replace(microscope, scan_step_um=dataset.metadata.scan_axis_step_um)
    return SimpleNamespace(
        dataset=dataset,
        microscope=microscope,
        expectation=expected,
        source=source,
        sampling_factor=factor,
        sampling_phase=phase,
        native_scan_planes=native_scan_planes,
    )


@pytest.fixture
def mosaic_acquisition(
    request,
    pipeline_optical_data,
    simulation_parameters,
    opm_v2_acquisition_factory,
    small_cpu_kernels,
):
    """Persist independent adjacent views of the same extended fluorescent object.

    Returns
    -------
    types.SimpleNamespace
        Raw qi2lab acquisition, physical camera expectation and the shared
        dimensions used to check scan chunks, deskew tiles and mosaic seams.
    """
    mode, step, *encoder_errors = request.param
    scan_encoder_error_um = encoder_errors[0] if encoder_errors else 0.0
    layout = encoder_errors[1] if len(encoder_errors) > 1 else "tiled"
    assert layout in ("whole", "tiled")
    native_step = simulation_parameters["microscope"]["scan_step_um"]
    native_scans = simulation_parameters[f"{layout}_scan_planes"]
    phase = (native_scans - 1) % round(step / native_step) if mode == "stage" else 0
    optical_data = pipeline_optical_data(
        step, mosaic=True, tiled=layout == "tiled", phase=phase
    )
    microscope, expectation = (
        optical_data.microscope,
        optical_data.expectation,
    )
    parameters = simulation_parameters["pipeline_mosaic"]
    height, width = simulation_parameters["camera_shape_yx"]
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
    stage_scan = -4.6
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
            (2.3, stage_scan, 1.15),
            (
                2.3,
                stage_scan + scan_encoder_error_um,
                1.15 + (offset + DEFAULT_REGISTRATION_DOWNSAMPLE[-1]) * pixel,
            ),
        ),
        camera_read_noise_e=microscope.camera.read_noise_e,
    )
    return SimpleNamespace(
        dataset=dataset,
        microscope=microscope,
        expectation=expectation,
        source_expectation=optical_data.source_expectation,
        parameters=parameters,
        optical_data=optical_data,
        tiled=layout == "tiled",
    )
