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

    from opm_processing.dataio.position_collection import open_position_collection
    from opm_processing.dataio.roi import projection_tile_footprints
    from opm_processing.imageprocessing.maxtilefusion import (
        regenerate_fused_max_projection,
    )
    from opm_processing.imageprocessing.tilefusion import TileFusion
    from opm_processing.process import process_skewed

    mode, positions, registered = request.param
    scene = roi_camera_scene(times=2, positions=positions, channels=2)
    pixels = scene.pixels
    if mode == "stage":
        pixels = pixels[:, :, :, ::-1].copy()
    dataset = opm_v2_acquisition_factory(
        pixels,
        mode=mode,
        camera_conversion=1.0,
        stage_positions_zxy=((2.3, -4.6, 1.15),) * positions,
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
    return SimpleNamespace(
        dataset=dataset,
        preview=preview,
        positions=positions,
        origin=footprints[0]["origin_zyx_um"],
        truth=scene,
        raw_origin_zyx_um=(-2.3, 1.8 if mode == "stage" else 4.6, 1.15),
        collection=open_position_collection(
            dataset.path.parent / "sample_deskewed.ome.zarr"
        ),
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
    pixels = scene.pixels
    dataset = opm_v2_acquisition_factory(
        pixels, camera_conversion=1.0, stage_positions_zxy=((0.0, 0.0, 0.0),) * 3
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
        raw_origin_zyx_um=(0.0, 0.0, 0.0),
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
            )
            pixels[time, position, channel] = observation.raw
            means[time, position, channel] = observation.expectation
        microscope = observation.microscope
        pixel = microscope.pixel_size_um
        angle = np.deg2rad(microscope.angle_deg)
        scans, rows, columns = pixels.shape[-3:]
        # The simulator centers all sensor axes. Translate the original object
        # into the acquisition's first-pixel coordinate frame, independently
        # of any processing metadata or reconstructed image.
        center = np.asarray(
            (
                (rows - 1) * pixel * np.sin(angle) / 2,
                (
                    (scans - 1) * microscope.scan_step_um
                    + (rows - 1) * pixel * np.cos(angle)
                )
                / 2,
                (columns - 1) * pixel / 2,
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
        )

    return create


@pytest.fixture
def assert_roi_point_accuracy():
    """Compare reconstructed photons and coordinates with the original emitters.

    Returns
    -------
    callable
        Check physical peak locations in the two independently prescribed
        source half-fields. Six shot/read-noise standard deviations bound the
        peak; positive interpolation and averaging cannot amplify it above
        the calibrated camera maximum and the known photon-volume Jacobian.
    """

    def check(
        data,
        scene,
        origin,
        spacing,
        *,
        raw_origin=(0, 0, 0),
        channels=None,
        illumination=1.0,
    ):
        """Measure actual stored values against fixed physical source locations."""
        origin, spacing = np.asarray(origin), np.asarray(spacing)
        expected = scene.centers_zyx_um + np.asarray(raw_origin)
        split = expected[:, 2].mean()
        columns = origin[2] + np.arange(data.shape[-1]) * spacing[2]
        gain = (
            2
            * scene.microscope.pixel_size_um
            / (scene.microscope.scan_step_um * illumination)
        )
        camera = scene.microscope.camera
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
                mean = scene.means[time, :, channel].max()
                sigma = np.sqrt(
                    mean + camera.read_noise_e**2 + camera.electrons_per_adu**2 / 12
                )
                assert 0 < volume[peak] <= gain * (mean + 6 * sigma)
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


@pytest.fixture
def assert_roi_observations_preserved():
    """Check crop consistency per camera realization, before fusion combines views.

    Returns
    -------
    callable
        Compare each cropped tile with the same position's full processing.
        This checks preservation of observed photons; scientific accuracy is
        separately checked against the original object and optical model.
    """

    def check(cropped, references, spacing, scan_step, state):
        """Compare raw-indexed crops for each actual time/position camera draw."""
        series = state.roi_series(cropped.path)
        assert len(series) == len(cropped.arrays)
        for tile, record in zip(cropped.arrays, series, strict=True):
            scan_start, _scan_stop, x_start, _x_stop = record["skewed_bounds_sx"]
            y0, _y1, x0, _x1 = record["deskew_crop_yx"]
            offset = np.asarray(
                (0, scan_start * scan_step / spacing[1] + y0, x_start + x0)
            )
            np.testing.assert_allclose(offset, np.rint(offset), atol=1e-8)
            offset = np.rint(offset).astype(int)
            actual = tile.read().result()
            time, position = record["time_index"], record["position_index"]
            reference = references[position][time : time + 1]
            lower = np.maximum(0, offset)
            upper = np.minimum(reference.shape[-3:], offset + actual.shape[-3:])
            assert np.all(upper > lower)
            source = tuple(slice(a, b) for a, b in zip(lower, upper, strict=True))
            selected = tuple(
                slice(a - shift, b - shift)
                for a, b, shift in zip(lower, upper, offset, strict=True)
            )
            np.testing.assert_allclose(
                actual[(..., *selected)],
                reference[(..., *source)],
                rtol=2e-6,
                atol=2e-3,
            )

    return check
