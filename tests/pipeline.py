"""Complete acquisition processing scenarios using production numerical operators."""

import importlib
from functools import partial
from itertools import pairwise

import numpy as np
import tensorstore as ts
from skimage.measure import block_reduce
from yaozarrs import open_group, v05

from opm_processing.dataio.acquisition import (
    inspect_acquisition,
    open_acquisition_datastore,
)
from opm_processing.dataio.deconvolution import scan_upsample_from_metadata
from opm_processing.dataio.processing_state import ProcessingState
from opm_processing.imageprocessing.camera import camera_correct, illumination_correct
from opm_processing.imageprocessing.opmtools import orthogonal_deskew


def test_qi2lab_acquisition_to_fused_ngff_v05(pipeline_case, deconvolve):
    """Retain calibrated pixels, registered coordinates and pyramids across stages."""
    case = pipeline_case
    raw = open_group(case.dataset.path)
    assert "opm_v2" in raw.attrs
    assert raw.attrs["opm_v2"]["storage_backend"] == "tensorstore"
    assert raw["0"].attrs["ome_writers"]["frame_metadata"]
    metadata = inspect_acquisition(case.dataset.path)
    np.testing.assert_array_equal(
        open_acquisition_datastore(metadata).read().result(), case.dataset.raw_data
    )
    assert "psf_parameters" not in (metadata.deconvolution or {})
    assert "emission_wavelengths_nm" not in (metadata.deconvolution or {})
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
            else:
                reconstruction_index = (
                    time * metadata.shape[1] + position
                ) * metadata.shape[2] + channel
                expected = orthogonal_deskew(
                    case.reconstructions[reconstruction_index],
                    theta=metadata.angle_deg,
                    distance=metadata.scan_axis_step_um
                    / scan_upsample_from_metadata(metadata),
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
    # one-registration-pixel stage displacement, for both independent timepoints.
    for time in range(2):
        indices = np.flatnonzero(np.asarray(fusion._tile_time_indices) == time)
        np.testing.assert_allclose(
            np.diff(np.asarray(fusion._tile_positions)[indices], axis=0),
            0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            np.diff(case.nominal[indices], axis=0),
            ((0, 0, case.registration_error_pixels * metadata.pixel_size_um),),
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
        assert settings["read_noise"] == case.microscope.camera.read_noise_e
        assert "decon_psf_paths" not in case.options
        assert "psf_parameters" not in (metadata.deconvolution or {})
        assert settings["emission_wavelengths_nm"] == [637.0, 637.0]
        assert settings["psf_parameters"]["lens"]["NA"] == 1.35
        assert settings["psf_sha256"] is None
        # Score actual in-loop production predictions against simulator truth.
        from scipy.special import kl_div

        factor = scan_upsample_from_metadata(metadata)
        assert len(case.reconstructions) == np.prod(metadata.shape[:3])
        assert len(case.deconvolution_runs) == len(case.reconstructions)
        for reconstruction, run in zip(
            case.reconstructions, case.deconvolution_runs, strict=True
        ):
            assert run["input_shape"] == metadata.shape[-3:]
            assert reconstruction.shape == (
                (metadata.shape[-3] - 1) * factor + 1,
                *metadata.shape[-2:],
            )
        for index, run in zip(
            np.ndindex(*metadata.shape[:3]),
            case.deconvolution_runs,
            strict=True,
        ):
            time, _position, channel = index
            observed = run["forward"]
            baseline = observed["input_prediction"]
            if baseline is None:
                baseline = observed["initial_prediction"]
            expected = case.expectation * (1 + time / 4 + channel / 2)
            assert observed["prediction"].shape == expected.shape
            assert (
                kl_div(expected, observed["prediction"]).sum()
                < kl_div(expected, baseline).sum()
            )

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
        if factor > 1:
            np.testing.assert_array_equal(
                reopened[dataset.path].to_tensorstore().read().result(),
                base[:, :, ::factor, ::factor, ::factor],
            )
    return base


def test_large_noisy_scene_acquisition_to_fusion(
    large_noisy_acquisition,
    deconvolve,
    pipeline_runner,
    pipeline_geometry,
    tmp_path,
):
    """Process full noisy specimens across real RLGC, deskew and storage tiles."""
    from scipy.special import kl_div

    case = large_noisy_acquisition
    dataset, microscope = case.dataset, case.microscope
    metadata = inspect_acquisition(dataset.path)
    assert metadata.shape[-2:] == (256, 1900)
    assert case.native_scan_planes in (50, 200)
    assert metadata.shape[-3] == len(
        range(case.sampling_phase, case.native_scan_planes, case.sampling_factor)
    )
    tiled_scan = case.native_scan_planes == 200
    crop_scan = 39 if tiled_scan else case.native_scan_planes
    np.testing.assert_array_equal(
        open_acquisition_datastore(metadata).read().result(), dataset.raw_data
    )
    result = pipeline_runner(
        dataset,
        deconvolve=deconvolve,
        crop_scan=crop_scan if deconvolve else None,
        output=tmp_path / "processed",
        fusion_chunk=(128, 128),
    )
    factor = scan_upsample_from_metadata(metadata) if deconvolve else 1
    geometry = pipeline_geometry(metadata.shape[-3:], microscope, scan_factor=factor)
    for position, array in enumerate(result.collection.arrays):
        saved = array[0, 0].read().result()
        assert saved.shape == geometry.shape and saved.dtype == np.float32
        np.testing.assert_array_equal(saved[~geometry.support], 0)
        calibrated = camera_correct(
            dataset.raw_data[0, position, 0],
            microscope.camera.offset_adu,
            microscope.camera.electrons_per_adu,
        )
        if metadata.scan_axis_reversed:
            calibrated = calibrated[::-1].copy()
        if deconvolve:
            run = result.deconvolution_runs[position]
            observed = run["forward"]
            chunks = sorted(
                observed["chunks"], key=lambda chunk: chunk["destination"].start
            )
            if tiled_scan:
                assert len(chunks) > 2
                assert chunks[0]["destination"].start == 0
                assert chunks[-1]["destination"].stop == metadata.shape[-3]
                assert all(
                    left["destination"].stop == right["destination"].start
                    for left, right in pairwise(chunks)
                )
                assert (
                    chunks[-1]["destination"].stop - chunks[-1]["destination"].start
                    < chunks[0]["destination"].stop
                )
                source = case.source[case.sampling_phase :: case.sampling_factor]
                borders = [chunk["destination"].stop for chunk in chunks[:-1]]
                assert (
                    sum(source[b - 1].any() and source[b].any() for b in borders) >= 2
                )
            else:
                assert len(chunks) == 1 and chunks[0]["destination"] == slice(None)
            assert all(chunk["shared"] == tiled_scan for chunk in chunks)
            for chunk in chunks:
                assert 1 <= chunk["visits"] <= 2
                assert chunk["updates"] == list(range(1, len(chunk["updates"]) + 1))
                assert (
                    observed["iterations"]
                    <= len(chunk["updates"])
                    <= observed["iterations"] + 2
                )
            baseline = observed["input_prediction"]
            if baseline is None:
                baseline = observed["initial_prediction"]
            source = case.source[case.sampling_phase :: case.sampling_factor]
            assert source.any()
            assert (
                kl_div(case.expectation[source], observed["prediction"][source]).sum()
                < kl_div(case.expectation[source], baseline[source]).sum()
            )
            calibrated = result.reconstructions[position]
        expected = orthogonal_deskew(
            calibrated,
            theta=microscope.angle_deg,
            distance=microscope.scan_step_um / factor,
            pixel_size=microscope.pixel_size_um,
            downsample_factor=1,
        )
        np.testing.assert_array_equal(saved, expected)

    # Only fixed source geometry selects foreground for execution coverage.
    source = (
        case.source[
            case.sampling_phase : case.sampling_phase
            + (metadata.shape[-3] - 1) * factor
            + 1
        ]
        if deconvolve
        else case.source[case.sampling_phase :: case.sampling_factor]
    )
    foreground = (
        orthogonal_deskew(
            source.astype(np.float32),
            theta=microscope.angle_deg,
            distance=microscope.scan_step_um / factor,
            pixel_size=microscope.pixel_size_um,
            downsample_factor=1,
        )
        > 0
    )
    covered = np.zeros(geometry.shape, bool)
    tile_shape = result.deskew_runs[0]["tile_shape"]
    for origin, shape in result.deskew_tiles:
        assert np.all(np.asarray(shape) <= tile_shape)
        covered[tuple(slice(a, a + n) for a, n in zip(origin, shape, strict=True))] = (
            True
        )
    assert covered.all() and len(result.deskew_tiles) > 8
    crossings = 0
    axis_borders = []
    for axis in range(3):
        borders = {
            origin[axis] for origin, _shape in result.deskew_tiles if origin[axis]
        }
        axis_borders.append(sorted(borders))
        crossings += sum(
            np.any(
                np.take(foreground, border - 1, axis=axis)
                & np.take(foreground, border, axis=axis)
            )
            for border in borders
        )
    assert crossings >= 2, (crossings, axis_borders, geometry.shape, tile_shape)
    assert any(
        origin[axis] % tile_shape[axis]
        and origin[axis] + shape[axis] == geometry.shape[axis]
        for origin, shape in result.deskew_tiles
        for axis in range(3)
    )
    fused = assert_fused_ngff(result)
    np.testing.assert_allclose(
        np.diff(np.asarray(result.fusion._tile_positions), axis=0), 0, atol=1e-8
    )
    origin = np.asarray(result.fusion._tile_positions)[0]
    start = np.rint(
        (origin - result.fusion.offset_um) / result.collection.voxel_size_um
    ).astype(int)
    selection = tuple(
        slice(a, a + n) for a, n in zip(start, geometry.shape, strict=True)
    )
    expected = (
        sum(array[0, 0].read().result() for array in result.collection.arrays) / 2
    )
    np.testing.assert_allclose(
        fused[(0, 0, *selection)], expected, rtol=8 * np.finfo(np.float32).eps
    )
    state = ProcessingState.read(result.fusion.processing_state.path)
    assert state.completed_tiles(result.processed) == {(0, 0), (0, 1)}


def test_extended_mosaic_acquisition_to_fusion(
    mosaic_acquisition,
    deconvolve,
    pipeline_runner,
    pipeline_geometry,
    pipeline_optical_data,
    monkeypatch,
    tmp_path,
):
    """Check realistic camera fields, automatic chunks and independent mosaic truth."""
    case = mosaic_acquisition
    parameters = case.parameters
    raw_shape = case.dataset.raw_data.shape[-3:]
    factor = scan_upsample_from_metadata(case.dataset.metadata) if deconvolve else 1
    shape = ((raw_shape[0] - 1) * factor + 1, *raw_shape[1:])
    assert raw_shape[1:] == (256, 1900)
    assert case.optical_data.native_scan_planes == (200 if case.tiled else 50)
    phase = case.optical_data.first_native_scan_index
    native_step = case.optical_data.ground_truth.microscope.scan_step_um
    offset = parameters["tile_offset_x_pixels"]
    layouts = []
    with monkeypatch.context() as resources:
        if deconvolve:
            solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
            create_convolution = solver._Convolution

            def observe_layout(*args, **kwargs):
                """Record real FFT operators without replacing their calculations."""
                operator = create_convolution(*args, **kwargs)
                layouts.append((operator.shape, operator.tile_shape))
                return operator

            resources.setattr(solver, "_Convolution", observe_layout)
            if case.tiled:
                resources.setattr(
                    solver,
                    "determine_rlgc_crop_scan",
                    partial(solver.determine_rlgc_crop_scan, memory_fraction=0.6),
                )
        result = pipeline_runner(
            case.dataset,
            deconvolve=deconvolve,
            blend=tuple(parameters["blend_pixels_zyx"]),
            crop_scan=None if case.tiled else shape[0],
            output=tmp_path / "processed",
            fusion_chunk=tuple(parameters["fusion_chunk_shape_yx"]),
        )
    if deconvolve:
        from scipy.special import kl_div

        assert layouts and all(state == fft for state, fft in layouts)
        fine_data = pipeline_optical_data(native_step, mosaic=True, tiled=case.tiled)
        source_mask = fine_data.source_mask[phase : phase + shape[0]]
        if case.tiled:
            assert result.options["decon_crop_scan"] is None
        for position, run in enumerate(result.deconvolution_runs):
            observed = run["forward"]
            chunks = sorted(
                observed["chunks"], key=lambda chunk: chunk["destination"].start
            )
            assert len(chunks) > 1 if case.tiled else len(chunks) == 1
            for chunk in chunks:
                if case.tiled:
                    assert chunk["observed_shape"][0] < raw_shape[0]
            start = position * offset
            seam_scans = np.ones(raw_shape[0], bool)
            if case.tiled:
                boundaries = np.asarray(
                    [chunk["destination"].stop for chunk in chunks[:-1]]
                )
                foreground = source_mask[::factor, :, start : start + shape[-1]].any(
                    axis=(1, 2)
                )
                assert sum(foreground[b - 1] and foreground[b] for b in boundaries) >= 2
                seam_scans[:] = False
                seam_scans[boundaries - 1] = True
                seam_scans[boundaries] = True
            expected = case.expectation[..., start : start + shape[-1]]
            predicted = observed["prediction"]
            baseline = observed["input_prediction"]
            if baseline is None:
                baseline = observed["initial_prediction"]
            assert predicted.shape == expected.shape
            assert kl_div(expected, predicted).sum() < kl_div(expected, baseline).sum()
            source = source_mask[::factor, :, start : start + shape[-1]]
            selected = source & seam_scans[:, None, None]
            assert selected.any()
            assert (
                kl_div(expected[selected], predicted[selected]).sum()
                < kl_div(expected[selected], baseline[selected]).sum()
            )

    support = pipeline_geometry(raw_shape, case.microscope, scan_factor=factor).support
    assert len(result.deskew_tiles) > 1
    tile_limits = np.asarray(result.deskew_runs[0]["tile_shape"])
    coverage = np.zeros(result.deskew_runs[0]["output_shape"], bool)
    for origin, tile_shape in result.deskew_tiles:
        assert np.all(np.asarray(tile_shape) <= tile_limits)
        coverage[
            tuple(slice(a, a + n) for a, n in zip(origin, tile_shape, strict=True))
        ] = True
    assert coverage.all()
    actual = assert_fused_ngff(result)
    fusion = result.fusion
    origins = np.asarray(fusion._tile_positions)
    np.testing.assert_allclose(
        origins[1] - origins[0],
        (0, 0, offset * case.microscope.pixel_size_um),
        atol=1e-8,
    )
    assert (0, 1) in fusion.optimized_pairwise_metrics
    width = shape[-1]
    blend = parameters["blend_pixels_zyx"][-1]
    midpoint = (width - offset) // 2
    assert blend <= midpoint < width - blend
    assert blend <= offset + midpoint < width - blend
    first, second = [array.read().result() for array in result.collection.arrays]
    start = np.rint(
        (origins[0] - fusion.offset_um) / result.collection.voxel_size_um
    ).astype(int)
    zy = tuple(
        slice(a, a + n) for a, n in zip(start[:2], support.shape[:2], strict=True)
    )
    for column, expected in (
        (offset // 2, first[..., offset // 2]),
        (
            offset + midpoint,
            (first[..., offset + midpoint] + second[..., midpoint]) / 2,
        ),
        (width + offset // 2, second[..., width - offset + offset // 2]),
    ):
        np.testing.assert_allclose(
            actual[(slice(None), slice(None), *zy, start[2] + column)],
            expected,
            rtol=5e-6,
            atol=5e-6,
        )
    block_y, block_x = fusion._fusion_block_shape(
        fusion.write_block_shape[2], *fusion.padded_shape[-2:]
    )
    assert block_y < actual.shape[-2] and block_x < actual.shape[-1]
    camera_source = source_mask if deconvolve else case.source_mask
    signal = orthogonal_deskew(
        camera_source,
        theta=case.microscope.angle_deg,
        distance=case.microscope.scan_step_um / factor,
        pixel_size=case.microscope.pixel_size_um,
        downsample_factor=1,
    )
    foreground = signal > signal.max() / 2
    block_shape = (fusion.write_block_shape[2], block_y, block_x)
    crossings = 0
    for axis, step in enumerate(block_shape):
        other_axes = [index for index in range(3) if index != axis]
        for border in range(step, actual.shape[axis + 2], step):
            local = border - start[axis]
            if 0 < local < foreground.shape[axis]:
                shared = np.take(foreground, local - 1, axis=axis) & np.take(
                    foreground, local, axis=axis
                )
                shared = np.pad(
                    shared, [(int(start[index]), 0) for index in other_axes]
                )
                # Distinct block faces in the same plane are distinct borders.
                crossings += np.count_nonzero(
                    block_reduce(
                        shared,
                        tuple(block_shape[index] for index in other_axes),
                        np.any,
                    )
                )
    assert crossings >= 2
