"""Simulated multi-channel acquisitions and physical regions for ROI workflows."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def roi_preview(
    request,
    opm_v2_acquisition_factory,
    small_cpu_kernels,
    roi_camera_scene,
    simulation_parameters,
):
    """Create a calibrated raw timelapse and a real processed max-Z preview."""
    from types import SimpleNamespace

    from opm_processing.dataio.roi import projection_tile_footprints
    from opm_processing.imageprocessing.maxtilefusion import (
        regenerate_fused_max_projection,
    )
    from opm_processing.imageprocessing.tilefusion import TileFusion
    from opm_processing.process import process_skewed

    mode, positions, registered = request.param
    scene = roi_camera_scene(times=2, positions=positions, channels=2)
    scene.stage_positions_zxy = ((2.3, -4.6, 1.15),) * positions
    pixels = scene.pixels
    if mode == "stage":
        pixels = pixels[:, :, :, ::-1].copy()
    dataset = opm_v2_acquisition_factory(
        pixels,
        mode=mode,
        camera_conversion=1.0,
        stage_positions_zxy=scene.stage_positions_zxy,
        acquired_plane_count=pixels.shape[-3],
        plane_period_s=simulation_parameters["pipeline"]["plane_period_s"],
    )
    process_skewed(
        dataset.path,
        acquisition=dataset.metadata,
        deconvolve=False,
        max_projection=True,
        create_fused_max_projection=positions > 1,
        save_float32=True,
        z_downsample_level=1,
    )
    preview = dataset.path.parent / (
        "sample_max_z_fused.ome.zarr"
        if positions > 1
        else "sample_max_z_deskewed.ome.zarr"
    )
    if registered:
        fusion = TileFusion(
            dataset.path.parent / "sample_deskewed.ome.zarr",
            multiscale_factors=(),
            resolution_multiples=((1, 1, 1),),
        )
        fusion.run()
        regenerate_fused_max_projection(
            dataset.path.parent / "sample_fused.ome.zarr", preview
        )
    footprints = projection_tile_footprints(preview)
    stage_anchor = np.asarray((-2.3, 4.6, 1.15))
    raw_origin = stage_anchor.copy()
    if mode == "stage":
        raw_origin[1] -= (pixels.shape[-3] - 1) * scene.microscope.scan_step_um
    return SimpleNamespace(
        dataset=dataset,
        preview=preview,
        positions=positions,
        origin=footprints[0]["origin_zyx_um"],
        truth=scene,
        raw_origin_zyx_um=tuple(raw_origin),
    )


@pytest.fixture
def napari_model(monkeypatch, tmp_path_factory):
    """Share headless viewer setup with isolated runtime caches and settings."""
    from napari.resources import _icons
    from napari.utils import _platformdirs

    directory = tmp_path_factory.mktemp("napari-runtime")
    monkeypatch.setattr(_icons, "user_cache_dir", lambda: str(directory))
    monkeypatch.setattr(_platformdirs, "user_config_dir", lambda: str(directory))
    from napari.components import ViewerModel
    from napari.plugins import _initialize_plugins

    _initialize_plugins()
    model = ViewerModel()
    yield model
    model.layers.clear()


@pytest.fixture
def roi_run(tmp_path: Path, opm_v2_acquisition_factory, roi_camera_scene):
    """Build a three-channel acquisition with two selected nonzero positions."""
    import importlib
    from types import SimpleNamespace

    from opm_processing.dataio.roi import PhysicalRoi

    process = importlib.import_module("opm_processing.process")
    scene = roi_camera_scene(positions=3, channels=3, vary_positions=True)
    scene.stage_positions_zxy = ((0.0, 0.0, 0.0),) * 3
    pixels = scene.pixels
    dataset = opm_v2_acquisition_factory(
        pixels, camera_conversion=1.0, stage_positions_zxy=scene.stage_positions_zxy
    )
    source, metadata = dataset.path, dataset.metadata
    roi = PhysicalRoi(
        bounds_yx_um=(2.07, 4.14, 0.46, 3.22),
        source_path=tmp_path / "sample_max_z_fused.ome.zarr",
        grid_origin_yx_um=(0.0, 0.0),
        pixel_size_yx_um=(0.115, 0.115),
        position_indices=(1, 2),
        tile_footprints=tuple(
            {
                "time_index": 0,
                "position_index": position,
                "origin_zyx_um": [0.0, 0.0, 0.0],
                "bounds_yx_um": [0.0, 6.4, 0.0, 3.68],
            }
            for position in (1, 2)
        ),
    )
    roi_path = roi.write(tmp_path / "sample_roi.json")

    return SimpleNamespace(
        process=process,
        source=source,
        metadata=metadata,
        raw_arrays=dataset.raw_arrays,
        roi=roi,
        roi_path=roi_path,
        output_dir=tmp_path / "sample_roi",
        truth=scene,
    )


@pytest.fixture
def roi_camera_scene(simulated_camera):
    """Share actual optical observations and original physical point locations.

    Returns
    -------
    callable
        Build calibrated timelapse/channel camera observations. Both original
        emitters remain inside the detector field and the selected ROI.
    """
    from types import SimpleNamespace

    from opm_processing.imageprocessing.opmpsf import camera_grid_center

    def create(
        *, times=1, positions=1, channels=1, vary_positions=False, shape=(8, 32, 32)
    ):
        """Render the existing physical point object without a new image simulator."""
        pixels = np.empty((times, positions, channels, *shape), np.uint16)
        means = np.empty(pixels.shape, np.float64)
        for time, position, channel in np.ndindex(pixels.shape[:3]):
            observation = simulated_camera(
                pixels.shape[-3:],
                conversion=1.0,
                seed=42 + 100 * time + 10 * position + channel,
                exposure=1
                + time / 4
                + channel / 2
                + (position / 4 if vary_positions else 0),
                budget=(3000, 4000),
            )
            pixels[time, position, channel] = observation.raw
            means[time, position, channel] = observation.expectation
        microscope = observation.microscope
        pixel = microscope.pixel_size_um
        # The simulator centers all sensor axes. Translate the original object
        # into the acquisition's first-pixel coordinate frame, independently
        # of any processing metadata or reconstructed image.
        center = np.asarray(
            camera_grid_center(
                pixels.shape[-3:],
                pixel_size_um=pixel,
                scan_axis_step_um=microscope.scan_step_um,
                theta_deg=microscope.angle_deg,
            )
        )
        centers = (
            center
            + np.asarray(observation.ground_truth.features["centers_xyz_um"])[:, ::-1]
        )
        return SimpleNamespace(
            pixels=pixels,
            means=means,
            centers_zyx_um=centers,
            microscope=microscope,
            ground_truth=observation.ground_truth,
        )

    return create


@pytest.fixture
def assert_roi_point_accuracy(assert_roi_point_locations):
    """Process original camera bounds with the production numerical kernels."""
    from opm_processing.imageprocessing.camera import illumination_correct
    from opm_processing.imageprocessing.opmpsf import camera_grid_center
    from opm_processing.imageprocessing.opmtools import (
        deskew_coverage,
        deskew_shape_estimator,
        interpolate_rows,
    )
    from opm_processing.imageprocessing.tilefusion import (
        _accumulate_tile_block,
        _normalize_block,
    )
    from opm_processing.process import format_processed_output
    from tests.simulation.camera_likelihood import (
        SamplingEnvelope,
        camera_sampling_accuracy,
        camera_sampling_envelope,
        sampling_stencil_radii,
    )

    def check(
        data,
        scene,
        origin,
        spacing,
        *,
        raw_origin=(0, 0, 0),
        channels=None,
        illumination=1.0,
        positions=None,
        fused=False,
    ):
        """Check fixed original-object apertures including partly supported Z bins."""
        assert_roi_point_locations(
            data, scene, origin, spacing, raw_origin=raw_origin, channels=channels
        )
        microscope, camera = scene.microscope, scene.microscope.camera
        pixel, scan_step = microscope.pixel_size_um, microscope.scan_step_um
        origin, raw_origin = np.asarray(origin), np.asarray(raw_origin)
        spacing = np.asarray(spacing)
        factor = round(spacing[0] / pixel)
        assert factor >= 1
        np.testing.assert_allclose(spacing[1:], pixel, rtol=0, atol=0.0005)
        selected_positions = (
            tuple(range(scene.pixels.shape[1]))
            if positions is None
            else tuple(positions)
        )
        assert selected_positions
        if fused and len(selected_positions) > 1:
            # These simulations explicitly expose the same physical camera field;
            # identically placed equal-support tiles have equal fusion weights.
            planned = np.asarray(scene.stage_positions_zxy)[list(selected_positions)]
            np.testing.assert_array_equal(
                planned, np.broadcast_to(planned[0], planned.shape)
            )
        native_shape = scene.pixels.shape[-3:]
        native_output_shape, *_ = deskew_shape_estimator(
            native_shape,
            microscope.angle_deg,
            scan_step,
            pixel,
            crop_after_deskew=False,
            divisible_by=1,
        )
        output_shape = data.shape[-3:]
        offset = (origin - raw_origin) / pixel
        first_z = round(offset[0] - (factor - 1) / 2)
        assert abs(offset[0] - first_z - (factor - 1) / 2) <= 0.0005 / pixel + 1e-12
        # The fixtures have coincident integer output grids. This explicitly
        # rules out an unaccounted fractional fusion resampling pitch.
        np.testing.assert_allclose(
            offset[1:], np.rint(offset[1:]), rtol=0, atol=0.0005 / pixel + 1e-12
        )
        assert first_z % factor == 0
        start = (first_z // factor, *np.rint(offset[1:]).astype(int))
        assert np.all(np.asarray(start) >= 0)
        coverage = deskew_coverage(
            native_shape,
            output_shape,
            microscope.angle_deg,
            scan_step,
            pixel,
            downsample_factor=factor,
            origin_zy=start[:2],
        )
        support = coverage[..., None] * (
            start[2] + np.arange(output_shape[2]) < native_shape[2]
        )
        profiles = [np.ones(length, np.float32) for length in output_shape]
        radii = sampling_stencil_radii(microscope, (pixel, pixel, pixel))
        field = np.asarray(illumination, np.float32)
        source_centers = scene.centers_zyx_um + raw_origin
        centers_y = origin[1] + np.arange(output_shape[1]) * spacing[1]
        centers_x = origin[2] + np.arange(output_shape[2]) * spacing[2]
        distance = np.full(output_shape[1:], np.inf)
        for point in source_centers:
            distance = np.minimum(
                distance,
                np.hypot(centers_y[:, None] - point[1], centers_x[None] - point[2]),
            )
        rayleigh = 0.61 * microscope.wavelength_um / microscope.numerical_aperture
        # The finite phantom's interior is fixed from its original domain and
        # the optical halo, never from the reconstructed signal or contrast.
        camera_center = (
            np.asarray(
                camera_grid_center(
                    native_shape,
                    pixel_size_um=pixel,
                    scan_axis_step_um=scan_step,
                    theta_deg=microscope.angle_deg,
                )
            )
            + raw_origin
        )
        domain = scene.ground_truth.axes
        interior = (
            (
                centers_y[:, None] - camera_center[1]
                >= domain[1][0] + microscope.psf_half_extent_zyx_um[1]
            )
            & (
                centers_y[:, None] - camera_center[1]
                <= domain[1][-1] - microscope.psf_half_extent_zyx_um[1]
            )
            & (
                centers_x[None] - camera_center[2]
                >= domain[2][0] + microscope.psf_half_extent_zyx_um[2]
            )
            & (
                centers_x[None] - camera_center[2]
                <= domain[2][-1] - microscope.psf_half_extent_zyx_um[2]
            )
        )
        apertures = (
            interior & (distance <= rayleigh),
            interior & (distance >= 2 * rayleigh),
        )
        selected_channels = range(data.shape[1]) if channels is None else channels
        for time in range(data.shape[0]):
            for channel in selected_channels:
                means = scene.means[time, selected_positions, channel].mean(axis=0)
                view_bounds = [
                    camera_sampling_envelope(
                        scene.means[time, position, channel],
                        scene.pixels[time, position, channel],
                        camera,
                        radii=radii,
                        response=field,
                        views=len(selected_positions),
                        z_bin_factor=factor,
                    )
                    for position in selected_positions
                ]
                if not fused:
                    if len(selected_positions) > 1:
                        np.testing.assert_array_equal(
                            scene.means[time, selected_positions, channel],
                            np.broadcast_to(
                                means, (len(selected_positions), *native_shape)
                            ),
                        )
                    sources = [
                        (
                            np.minimum.reduce([bound.lower for bound in view_bounds]),
                            np.maximum.reduce([bound.upper for bound in view_bounds]),
                            means,
                        )
                    ]
                else:
                    sources = [
                        (bound.lower, bound.upper, scene.means[time, position, channel])
                        for position, bound in zip(
                            selected_positions, view_bounds, strict=True
                        )
                    ]
                processed = np.zeros((3, *output_shape), np.float32)
                weight = np.zeros((1, *output_shape), np.float32)
                for source in sources:
                    deskewed = np.zeros_like(processed)
                    for target, values in zip(deskewed, source, strict=True):
                        interpolate_rows(
                            illumination_correct(np.asarray(values, np.float32), field),
                            target,
                            native_output_shape[1],
                            native_output_shape[0],
                            microscope.angle_deg,
                            scan_step,
                            pixel,
                            factor,
                            False,
                            start,
                        )
                    deskewed = format_processed_output(
                        deskewed, save_float32=data.dtype.kind == "f"
                    )
                    if fused:
                        _accumulate_tile_block(
                            processed,
                            weight,
                            deskewed,
                            np.ones(3, bool),
                            coverage,
                            *profiles,
                            0,
                            0,
                            0,
                        )
                    else:
                        processed[:] = deskewed
                if fused:
                    _normalize_block(processed, weight, processed)
                lower, upper, reference = processed
                quantum = 0 if data.dtype.kind == "f" else 2
                numerical = (
                    np.maximum(upper, 1) * 8 * np.finfo(np.float32).eps + quantum
                )
                envelope = SamplingEnvelope(
                    np.maximum(lower - numerical, 0), upper + numerical
                )
                for aperture in apertures:
                    mask = (support > 0) & aperture[None]
                    result = camera_sampling_accuracy(
                        data[time, channel], reference, envelope, mask
                    )
                    assert result.accepted, result

    return check


@pytest.fixture
def assert_reconstruction_storage():
    """Check storage with production kernels; callers score optical truth separately."""
    from opm_processing.imageprocessing.opmtools import (
        deskew_coverage,
        deskew_shape_estimator,
        interpolate_rows,
    )
    from opm_processing.imageprocessing.tilefusion import (
        _accumulate_tile_block,
        _normalize_block,
    )
    from opm_processing.process import format_processed_output

    def check(volume, reconstructions, origin, spacing, raw_origin, microscope):
        """Compare coincident inverse views with their persisted orthogonal samples."""
        shape = reconstructions[0].shape
        assert all(reconstruction.shape == shape for reconstruction in reconstructions)
        pixel = microscope.pixel_size_um
        np.testing.assert_allclose(spacing, pixel, rtol=0, atol=0.0005)
        offset = (np.asarray(origin) - raw_origin) / pixel
        np.testing.assert_allclose(
            offset, np.rint(offset), rtol=0, atol=0.0005 / pixel + 1e-12
        )
        start = np.rint(offset).astype(int)
        assert np.all(start >= 0)
        geometry = {
            "theta": microscope.angle_deg,
            "distance": microscope.scan_step_um,
            "pixel_size": pixel,
        }
        native_output_shape, *_ = deskew_shape_estimator(
            shape, **geometry, crop_after_deskew=False, divisible_by=1
        )
        coverage = deskew_coverage(
            shape, volume.shape, **geometry, origin_zy=tuple(start[:2])
        )
        expected = np.zeros((1, *volume.shape), np.float32)
        weight = np.zeros_like(expected)
        profiles = [np.ones(length, np.float32) for length in volume.shape]
        for reconstruction in reconstructions:
            deskewed = np.zeros_like(expected)
            interpolate_rows(
                reconstruction,
                deskewed[0],
                native_output_shape[1],
                native_output_shape[0],
                **geometry,
                downsample_factor=1,
                zero_initialized=False,
                output_origin=tuple(start),
            )
            _accumulate_tile_block(
                expected,
                weight,
                format_processed_output(
                    deskewed, save_float32=volume.dtype.kind == "f"
                ),
                np.ones(1, bool),
                coverage,
                *profiles,
                0,
                0,
                0,
            )
        _normalize_block(expected, weight, expected)
        np.testing.assert_allclose(
            volume, expected[0], rtol=8 * np.finfo(np.float32).eps, atol=0
        )

    return check


@pytest.fixture
def assert_roi_point_locations():
    """Compare reconstructed peak locations with the original physical emitters.

    Returns
    -------
    callable
        Check physical peak locations in the two independently prescribed
        source half-fields and the declared time/channel dose ordering.
    """

    def check(
        data,
        scene,
        origin,
        spacing,
        *,
        raw_origin=(0, 0, 0),
        channels=None,
    ):
        """Measure actual stored values against fixed physical source locations."""
        origin, spacing = np.asarray(origin), np.asarray(spacing)
        expected = scene.centers_zyx_um + np.asarray(raw_origin)
        split = expected[:, 2].mean()
        columns = origin[2] + np.arange(data.shape[-1]) * spacing[2]
        checked = 0
        for time, channel in np.ndindex(data.shape[:2]):
            if channels is not None and channel not in channels:
                continue
            for point, target in enumerate(expected):
                if np.any(target[1:] < origin[1:]) or np.any(
                    target[1:] >= origin[1:] + np.asarray(data.shape[-2:]) * spacing[1:]
                ):
                    continue
                checked += 1
                # Each emitter has its own physical half-field. A peak search
                # is not restricted to a small circle around the expected site.
                selection = columns < split if point == 0 else columns >= split
                volume = data[time, channel][..., selection]
                peak = np.unravel_index(np.argmax(volume), volume.shape)
                coordinate = origin + np.asarray(peak) * spacing
                coordinate[2] = columns[selection][peak[2]]
                # One output sample plus the 0.001-um serialization interval.
                np.testing.assert_allclose(
                    (coordinate - target) / spacing, 0, atol=1.01
                )
                assert volume[peak] > 0
        # The same physical object has explicitly increasing photon doses in
        # each channel/time. Preserve their ordering as well as their locations;
        # duplicated or swapped output channels cannot satisfy this invariant.
        expected_counts = scene.means.sum(axis=(1, 3, 4, 5))
        observed_counts = data.sum(axis=(-3, -2, -1), dtype=np.float64)
        selected_channels = (
            list(range(data.shape[1])) if channels is None else list(channels)
        )
        for left, right in pairwise(selected_channels):
            increasing = expected_counts[:, right] > expected_counts[:, left]
            assert np.all(
                observed_counts[increasing, right] > observed_counts[increasing, left]
            )
        if data.shape[0] > 1:
            for channel in selected_channels:
                increasing = np.diff(expected_counts[:, channel]) > 0
                assert np.all(np.diff(observed_counts[:, channel])[increasing] > 0)
        assert checked, "ROI excludes every original optical emitter"

    return check
