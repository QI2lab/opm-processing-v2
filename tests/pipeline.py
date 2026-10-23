"""Shared complete processing scenarios collected by the CPU and CUDA suites."""

import importlib
from functools import partial

import numpy as np
import tensorstore as ts
from yaozarrs import open_group, v05

from opm_processing.dataio.acquisition import inspect_acquisition
from opm_processing.dataio.deconvolution import scan_upsample_from_metadata
from opm_processing.dataio.processing_state import ProcessingState
from opm_processing.imageprocessing.camera import camera_correct, illumination_correct
from opm_processing.imageprocessing.opmtools import orthogonal_deskew
from tests.simulation.metrics import predictive_deviance


def test_qi2lab_acquisition_to_fused_ngff_v05(pipeline_case, deconvolve):
    """Retain calibrated pixels, registered coordinates and pyramids across stages."""
    case = pipeline_case
    raw = open_group(case.dataset.path)
    assert "opm_v2" in raw.attrs
    assert raw["0"].attrs["ome_writers"]["frame_metadata"]
    metadata = inspect_acquisition(case.dataset.path)
    assert metadata.shape == case.dataset.raw_data.shape
    assert metadata.camera_conversion == case.microscope.camera.electrons_per_adu
    assert metadata.scan_axis_step_um == case.microscope.scan_step_um
    assert case.collection.shape[:3] == (2, 2, 2)
    np.testing.assert_allclose(
        case.collection.voxel_size_um, case.microscope.pixel_size_um
    )
    if case.options["flatfield_correction"]:
        assert case.field is not None and case.field.shape == (
            1,
            2,
            *metadata.shape[-2:],
        )
        assert np.all(np.isfinite(case.field) & (case.field > 0))
    for position, array in enumerate(case.collection.arrays):
        assert isinstance(
            open_group(case.processed / str(position)).ome_metadata(), v05.Image
        )
        for time, channel in np.ndindex(2, 2):
            saved = array[time, channel].read().result()
            assert saved.dtype == np.float32
            assert np.isfinite(saved).all() and np.all(saved >= 0)
            assert np.any(saved[case.support] > 0)
            np.testing.assert_array_equal(saved[~case.support], 0)
            if not deconvolve:
                calibrated = camera_correct(
                    case.dataset.raw_data[time, position, channel],
                    metadata.camera_offset,
                    metadata.camera_conversion,
                )
                if case.field is not None:
                    calibrated = illumination_correct(
                        calibrated, case.field[0, channel]
                    )
                if metadata.mode == "stage":
                    calibrated = calibrated[::-1].copy()
                expected = orthogonal_deskew(
                    calibrated,
                    theta=metadata.angle_deg,
                    distance=metadata.scan_axis_step_um,
                    pixel_size=metadata.pixel_size_um,
                    downsample_factor=1,
                )
                np.testing.assert_array_equal(saved, expected)

    fusion = case.fusion
    # Stage Z is converted to laboratory Z; the detector-X stage coordinate
    # keeps its physical value. Registration must retain the anchored origin.
    stage_z, _scan, stage_x = metadata.stage_positions_zxy[0]
    np.testing.assert_allclose(
        np.asarray(fusion._tile_positions)[0, (0, 2)],
        (-stage_z, stage_x),
        atol=1e-8,
    )
    # Repeated fields must coincide after removing the deliberately wrong
    # one-pixel stage displacement, for both independent timepoints.
    for time in range(2):
        indices = np.flatnonzero(np.asarray(fusion._tile_time_indices) == time)
        np.testing.assert_allclose(
            np.diff(np.asarray(fusion._tile_positions)[indices], axis=0),
            0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            np.diff(case.nominal[indices], axis=0),
            ((0, 0, metadata.pixel_size_um),),
            atol=1e-8,
        )
        assert tuple(indices) in fusion.optimized_pairwise_metrics
    state = ProcessingState.read(fusion.processing_state.path)
    assert state.completed_tiles(case.processed) == set(np.ndindex(2, 2))
    registration = state.registration(case.processed)
    assert len(registration["tiles"]) == 4
    assert registration["fused_path"].endswith(case.output.name)
    if deconvolve:
        settings = state.run(case.processed)["configuration"]["deconvolution"]
        expected_noise = (
            case.microscope.camera.read_noise_e if metadata.deconvolution else 0
        )
        assert settings["gc_stopping"]["read_noise"] == expected_noise
        assert settings["gc_stopping"]["kl_enabled"] is True

    base = assert_fused_ngff(case)
    # Coincident independently exposed tiles have equal weights. This checks
    # every fused pixel, including block boundaries, unsupported edges and
    # pyramid padding, without calling the production fusion renderer.
    expected_fused = np.zeros_like(base)
    for time in range(2):
        indices = np.flatnonzero(np.asarray(fusion._tile_time_indices) == time)
        origin = np.asarray(fusion._tile_positions)[indices[0]]
        start = np.rint(
            (origin - fusion.offset_um) / case.collection.voxel_size_um
        ).astype(int)
        slices = tuple(
            slice(a, a + n) for a, n in zip(start, case.support.shape, strict=True)
        )
        exposures = np.stack(
            [array[time].read().result() for array in case.collection.arrays]
        )
        expected_fused[(time, slice(None), *slices)] = exposures.mean(axis=0)
    np.testing.assert_allclose(base, expected_fused, rtol=5e-6, atol=5e-6)


def assert_fused_ngff(case):
    """Validate persisted v0.5 axes, global coordinates, pixels and every scale.

    Returns
    -------
    numpy.ndarray
        Reopened full-resolution TCZYX pixels.
    """
    fusion = case.fusion
    reopened = open_group(case.output)
    assert reopened.metadata.zarr_format == 3
    assert reopened.attrs["ome"]["version"] == "0.5"
    image = reopened.ome_metadata()
    assert isinstance(image, v05.Image)
    multiscale = image.multiscales[0]
    assert [axis.name for axis in multiscale.axes] == list("tczyx")
    assert [axis.unit for axis in multiscale.axes[-3:]] == ["micrometer"] * 3
    factors = (1, *fusion.multiscale_factors)
    assert [dataset.path for dataset in multiscale.datasets] == [
        str(i) for i in range(len(factors))
    ]
    store = reopened["0"].to_tensorstore()
    assert isinstance(store, ts.TensorStore)
    specification = store.spec().to_json()
    assert specification["driver"] == "zarr3"
    assert specification["kvstore"]["driver"] == "file"
    base = store.read().result()
    assert base.shape[:2] == (case.collection.shape[0], case.collection.shape[2])
    assert base.dtype == np.float32
    assert np.isfinite(base).all() and base.max() > 0
    for factor, dataset in zip(factors, multiscale.datasets, strict=True):
        np.testing.assert_allclose(
            dataset.scale_transform.scale[-3:],
            factor * np.asarray(case.collection.voxel_size_um),
        )
        np.testing.assert_allclose(
            dataset.translation_transform.translation[-3:], fusion.offset_um
        )
        np.testing.assert_array_equal(
            reopened[dataset.path].to_tensorstore().read().result(),
            base[:, :, ::factor, ::factor, ::factor],
        )
    return base


def test_extended_mosaic_full_and_tiled_agree(
    mosaic_acquisition,
    deconvolve,
    pipeline_runner,
    monkeypatch,
    tmp_path,
):
    """Check automatic chunks and foreground seams against identical full processing."""
    case = mosaic_acquisition
    parameters = case.parameters
    shape = case.dataset.raw_data.shape[-3:]
    factor = scan_upsample_from_metadata(case.dataset.metadata) if deconvolve else 1
    shape = ((shape[0] - 1) * factor + 1, *shape[1:])
    expectation = case.reconstruction_expectation if deconvolve else case.expectation
    scan_step = case.microscope.scan_step_um / factor
    offset = parameters["tile_offset_x_pixels"]
    tile_limits = np.asarray(parameters["deskew_tile_shape_zyx"])
    # The noiseless specimen image crosses Y and X output tile borders, so the
    # numerical comparison exercises foreground seams as well as dark regions.
    signal = orthogonal_deskew(
        expectation,
        theta=case.microscope.angle_deg,
        distance=scan_step,
        pixel_size=case.microscope.pixel_size_um,
        downsample_factor=1,
    )
    bright = signal[..., : shape[-1]] > signal.max() / 2
    for axis in (0, 1, 2):
        boundaries = np.arange(tile_limits[axis], bright.shape[axis], tile_limits[axis])
        assert np.any(
            np.take(bright, boundaries - 1, axis=axis)
            & np.take(bright, boundaries, axis=axis)
        )
    layouts = []
    if deconvolve:
        solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
        create_convolution = solver._Convolution

        def observe_layout(*args, **kwargs):
            """Record actual FFT plans while returning the real numerical operator."""
            operator = create_convolution(*args, **kwargs)
            layouts.append((operator.shape, operator.tile_shape, operator.crop_scan))
            return operator

        monkeypatch.setattr(solver, "_Convolution", observe_layout)
        pads = solver._linear_fft_pad_width(shape, case.psf.shape)
        padded_shape = tuple(n + sum(p) for n, p in zip(shape, pads, strict=True))
        # Configure the public planner's memory fraction for a sub-volume budget.
        # Query actual free memory; do not replace the CUDA runtime or allocator.
        memory_budget = int(
            np.prod(padded_shape) * 26 * np.dtype(np.float32).itemsize * 0.76
        )

    options = {
        "deconvolve": deconvolve,
        "psf_path": case.psf_path,
        "blend": tuple(parameters["blend_pixels_zyx"]),
        # Keep the same real RNG and arithmetic backend for the two layouts.
        # The public host-storage mode still performs every convolution on CUDA.
        # Backend-specific random streams differ even with the same seed.
        "decon_storage": "host" if deconvolve else None,
    }
    full = pipeline_runner(
        case.dataset,
        **options,
        crop_scan=shape[0],
        output=tmp_path / "full",
        fusion_chunk=(256, 256),
    )
    if deconvolve:
        assert layouts and all(
            full_shape == tile_shape for full_shape, tile_shape, _ in layouts
        )
        layouts.clear()
    with monkeypatch.context() as resources:
        if deconvolve:
            free_bytes, _ = solver.cp.cuda.runtime.memGetInfo()
            resources.setattr(
                solver,
                "determine_rlgc_crop_scan",
                partial(
                    solver.determine_rlgc_crop_scan,
                    memory_fraction=min(0.8, memory_budget / free_bytes),
                ),
            )
        tiled = pipeline_runner(
            case.dataset,
            **options,
            output=tmp_path / "tiled",
            deskew_tile=tuple(parameters["deskew_tile_shape_zyx"]),
            fusion_chunk=tuple(parameters["fusion_chunk_shape_yx"]),
        )
    if deconvolve:
        assert layouts and all(
            tile_shape[0] < full_shape[0] for full_shape, tile_shape, _ in layouts
        )
        crops = {crop for _, _, crop in layouts}
        assert len(crops) == 1
        crop = crops.pop()
        assert 1 < crop < shape[0]
        foreground_scan = expectation[..., : shape[-1]].max(axis=(1, 2))
        foreground_scan = foreground_scan > foreground_scan.max() / 2
        boundaries = np.arange(crop, padded_shape[0], crop) - pads[0][0]
        boundaries = boundaries[(boundaries > 0) & (boundaries < shape[0])]
        assert (
            np.count_nonzero(
                foreground_scan[boundaries - 1] & foreground_scan[boundaries]
            )
            >= 2
        )
        assert tiled.options["decon_crop_scan"] is None
        # Compare actual reconstructions with known optical camera means, rather
        # than treating a different FFT layout as physical ground truth.
        from scipy.interpolate import interp1d

        for result in (full, tiled):
            assert len(result.reconstructions) == case.dataset.raw_data.shape[1]
            for position, recovered in enumerate(result.reconstructions):
                start = position * offset
                expected = case.reconstruction_expectation[
                    ..., start : start + shape[-1]
                ]
                raw = case.dataset.raw_data[0, position, 0]
                if case.dataset.metadata.scan_axis_reversed:
                    raw = raw[::-1]
                baseline = camera_correct(
                    raw,
                    case.microscope.camera.offset_adu,
                    case.microscope.camera.electrons_per_adu,
                )
                if factor > 1:
                    baseline = interp1d(
                        np.arange(baseline.shape[0]) * factor, baseline, axis=0
                    )(np.arange(shape[0]))
                boundary = "constant" if factor > 1 else "reflect"
                assert predictive_deviance(
                    recovered,
                    expected,
                    case.psf,
                    boundary=boundary,
                ) < predictive_deviance(baseline, expected, case.psf, boundary=boundary)

    # A numerical layout comparison uses the same unsigned camera pixels,
    # recorded noise, measured PSF, random seed and unmodified GC stopping rules.
    # Tolerance covers accumulated single-precision FFT roundoff, not a change
    # in exposure, normalization, likelihood or reconstruction parameters.
    tolerance = {"rtol": 2e-4, "atol": 2e-3} if deconvolve else {"rtol": 0, "atol": 0}
    for reference, blocked in zip(
        full.collection.arrays, tiled.collection.arrays, strict=True
    ):
        np.testing.assert_allclose(
            blocked.read().result(), reference.read().result(), **tolerance
        )

    support = (
        orthogonal_deskew(
            np.full(shape, 1, dtype=np.float32),
            distance=scan_step,
            pixel_size=case.microscope.pixel_size_um,
            theta=case.microscope.angle_deg,
            downsample_factor=1,
        )
        > 0
    )
    assert len(tiled.deskew_tiles) > 1
    assert all(
        np.all(np.asarray(tile_shape) <= tile_limits)
        for _, tile_shape in tiled.deskew_tiles
    )
    assert any(
        np.any(np.asarray(tile_shape) < tile_limits)
        for _, tile_shape in tiled.deskew_tiles
    )
    fused = []
    for result in (full, tiled):
        actual = assert_fused_ngff(result)
        fused.append(actual)
        fusion = result.fusion
        origins = np.asarray(fusion._tile_positions)
        np.testing.assert_allclose(
            origins[1] - origins[0],
            (0, 0, offset * case.microscope.pixel_size_um),
            atol=1e-8,
        )
        assert (0, 1) in fusion.optimized_pairwise_metrics
        # Independent linear feathering: retain both exclusive camera regions
        # and normalize overlapping measurements of the same physical specimen.
        numerator = np.zeros_like(actual, dtype=np.float64)
        denominator = np.zeros_like(actual, dtype=np.float64)
        x = np.arange(shape[-1])
        blend = parameters["blend_pixels_zyx"][-1]
        weights = np.minimum(1, np.minimum(x + 1, shape[-1] - x) / blend)
        for position, array in enumerate(result.collection.arrays):
            start = np.rint(
                (origins[position] - fusion.offset_um) / result.collection.voxel_size_um
            ).astype(int)
            selection = tuple(
                slice(a, a + n) for a, n in zip(start, support.shape, strict=True)
            )
            numerator[(slice(None), slice(None), *selection)] += (
                array.read().result() * weights
            )
            denominator[(slice(None), slice(None), *selection)] += support * weights
        expected = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
        )
        np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=5e-6)
        # Bright lines occur in the left-only, shared and right-only regions.
        for region in (
            slice(0, offset),
            slice(offset, shape[-1]),
            slice(shape[-1], None),
        ):
            field = case.expectation[..., region]
            assert field.max() > case.expectation.max() / 2
        if result is tiled:
            block_y, block_x = fusion._fusion_block_shape(
                fusion.write_block_shape[2],
                *fusion.padded_shape[-2:],
            )
            assert block_y < actual.shape[-2] and block_x < actual.shape[-1]
            # Locate foreground from the known optical specimen, independently
            # of whether deconvolution narrows a line onto one camera column.
            foreground = signal > signal.max() / 2
            x_borders = np.arange(block_x, actual.shape[-1], block_x)
            y_borders = np.arange(block_y, actual.shape[-2], block_y)
            assert np.any(foreground[..., x_borders - 1] & foreground[..., x_borders])
            assert np.any(foreground[:, y_borders - 1] & foreground[:, y_borders])
    np.testing.assert_allclose(fused[1], fused[0], **tolerance)
