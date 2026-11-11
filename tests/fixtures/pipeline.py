"""Shared qi2lab acquisitions for complete processing and fusion integration."""

import importlib
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

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
from tests.simulation.metrics import camera_source_mask
from tests.simulation.scenes import OpticalScene


@dataclass(frozen=True)
class PipelineOpticalData:
    """Immutable camera expectations and source truth for one sensor grid."""

    microscope: Microscope
    expectation: np.ndarray
    source_mask: np.ndarray | None
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
        live=None,
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
            live=live,
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
        deconvolution_runs = []
        with monkeypatch.context() as execution:
            if deconvolve:
                solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
                reconstruct = solver.rlgc

                def observe_reconstruction(*args, **kwargs):
                    """Observe real reconstruction shapes and acquired predictions."""
                    result = reconstruct(*args, **kwargs)
                    deconvolution_runs.append(
                        {
                            "input_shape": args[0].shape,
                            "output_shape": result.shape,
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
                """Record actual output bytes before production persists them."""
                result = deskew(*args, **kwargs)
                assert result.dtype == np.float32 and result.flags.c_contiguous
                deskew_runs.append(
                    {
                        "output_shape": result.shape,
                        "tile_shape": kwargs["tile_shape"],
                        "pixel_sha256": sha256(memoryview(result)).hexdigest(),
                    }
                )
                return result

            execution.setattr(opmtools, "interpolate_rows", record_tile)
            execution.setattr(module, "orthogonal_deskew", observe_deskew)
            source = dataset.path.parent if live is not None else dataset.path
            invoke_cli("process", source, **options)
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
            deconvolution_runs=deconvolution_runs,
        )

    return run


@pytest.fixture
def pipeline_illumination_calibration(
    opm_v2_acquisition_factory,
    uniform_optical_calibration,
):
    """Prepare a real independent calibration through the production worker."""
    from opm_processing.dataio.acquisition import (
        acquisition_stem,
        open_acquisition_datastore,
    )
    from opm_processing.process import call_estimate_illuminations, write_flatfield

    def calibrate(dataset, microscope, shape, mode):
        """Fit the acquired uniform source without processing unused image volumes."""
        shape = tuple(shape)
        stem = acquisition_stem(dataset.path)
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
        metadata = calibration.metadata
        field = call_estimate_illuminations(
            open_acquisition_datastore(metadata),
            metadata.camera_offset,
            metadata.camera_conversion,
            np.asarray(metadata.stage_positions_zxy),
            None,
            None,
        )
        write_flatfield(
            dataset.path.parent / f"{stem}_flatfield.ome.tif",
            field,
            microscope.pixel_size_um,
        )
        return field

    return calibrate


@pytest.fixture(scope="session")
def pipeline_optical_data(simulation_parameters, optical_camera_data):
    """Reuse native optical camera fields for repeated exposures and mosaics.

    Returns
    -------
    callable
        Return immutable camera means and fixed physical source masks for mosaics.
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
        scene, microscope, expectation = optical_camera_data(
            shape,
            kind=kind,
            large=True,
            integration=True,
            illumination_profile="edge-taper" if illumination else "uniform",
            step=reference_step,
            geometry="oblique",
            budget=(3000, 4000),
        )
        source = camera_source_mask(scene, microscope, shape) if mosaic else None
        if source is not None:
            source.setflags(write=False)
        return scene, microscope, expectation, source

    @lru_cache(maxsize=2)
    def create(
        step, illumination=False, *, mosaic=False, tiled=False, kind=None, phase=0
    ):
        """Sample one unchanged physical emission field at the requested scan pitch."""
        scene, microscope, expectation, native_source = native_data(
            illumination, mosaic, tiled, kind
        )
        microscope = replace(microscope, scan_step_um=step)
        native_step = simulation_parameters["microscope"]["scan_step_um"]
        factor = round(step / native_step)
        assert factor >= 1 and np.isclose(step, factor * native_step)
        assert 0 <= phase < factor
        expected = expectation[phase::factor]
        source = None if native_source is None else native_source[phase::factor]
        return PipelineOpticalData(
            microscope,
            expected,
            source,
            scene,
            expectation.shape[0],
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
    pipeline_illumination_calibration,
):
    """Process time/channel exposures and retain paired views for CLI exports.

    Indirect parameters select scan mode, illumination, scan step, FFT scan
    chunk size. All deconvolution uses production-generated optical kernels.

    Returns
    -------
    types.SimpleNamespace
        Raw acquisition, reopened processed collection, fitted illumination,
        fusion object and actual processing observations.

    Notes
    -----
    Deconvolution uses one position; the non-deconvolution CLI case uses two
    independent exposures of the same field. Its second position reports an
    erroneous lateral shift of one registration-grid pixel.
    BaSiC runs in its production worker process. Camera calibration, RLGC,
    orthogonal interpolation, registration and fusion execute normally, with
    native TensorStore storage in pytest temporary directories.
    """
    mode, illumination, step, crop = request.param
    if illumination:
        request.getfixturevalue("torch_gpu")
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
    positions = 1 if deconvolve else 2
    raw = np.empty((2, positions, 2, *expectation.shape), dtype=np.uint16)
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
        )[:positions],
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
    return SimpleNamespace(
        **vars(result),
        dataset=dataset,
        microscope=microscope,
        expectation=expectation,
    )


@pytest.fixture
def large_noisy_acquisition(
    request,
    noisy_extended_scene,
    simulation_parameters,
    opm_v2_acquisition_factory,
    opm_v2_live_acquisition_factory,
):
    """Write independent full-field views of a shared noisy extended specimen."""
    from opm_processing.dataio.acquisition import ChannelMetadata

    mode, factor, positions, live = request.param
    microscope, expected, raw, source, _dark = noisy_extended_scene
    native_scan_planes = expected.shape[0]
    phase = (native_scan_planes - 1) % factor if mode == "stage" else 0
    microscope = replace(microscope, scan_step_um=microscope.scan_step_um * factor)
    expected = expected[phase::factor]
    exposures = [raw[phase::factor]]
    if positions == 2:
        exposures.append(
            digitize(
                expected,
                microscope.camera,
                seed=simulation_parameters["camera_seed"] + 1,
            )
        )
    pixels = np.stack(exposures)[None, :, None]
    if mode == "stage":
        pixels = pixels[:, :, :, ::-1].copy()
    pixel = microscope.pixel_size_um
    registration_error_pixels = DEFAULT_REGISTRATION_DOWNSAMPLE[-1]
    stage_scan = -4.6
    writer = opm_v2_live_acquisition_factory if live else opm_v2_acquisition_factory
    dataset = writer(
        pixels,
        name="large_scene",
        mode=mode,
        pixel_size_um=pixel,
        scan_axis_step_um=microscope.scan_step_um,
        angle_deg=microscope.angle_deg,
        camera_offset=microscope.camera.offset_adu,
        camera_conversion=microscope.camera.electrons_per_adu,
        camera_read_noise_e=microscope.camera.read_noise_e,
        channels=(ChannelMetadata(0, "488 + 637", None, 10.0, 10.0),),
        orientations=(
            ("camera_XYstage_orientation", "positive"),
            ("camera_Zstage_orientation", "positive"),
            ("camera_mirror_orientation", "positive"),
        ),
        stage_positions_zxy=(
            (2.3, stage_scan, 1.15),
            (2.3, stage_scan, 1.15 + registration_error_pixels * pixel),
        )[:positions],
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
        registration_error_pixels=registration_error_pixels,
        live=live,
    )


@pytest.fixture
def mosaic_acquisition(
    request,
    pipeline_optical_data,
    simulation_parameters,
    opm_v2_acquisition_factory,
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
        source_mask=optical_data.source_mask,
        parameters=parameters,
        optical_data=optical_data,
        tiled=layout == "tiled",
    )
