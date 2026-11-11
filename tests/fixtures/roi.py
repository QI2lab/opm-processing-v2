"""Simulated multi-channel acquisitions and physical regions for ROI workflows."""

from __future__ import annotations

from functools import lru_cache
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
    pixels = scene.pixels
    if mode == "stage":
        pixels = pixels[:, :, :, ::-1].copy()
    dataset = opm_v2_acquisition_factory(
        pixels,
        mode=mode,
        camera_conversion=1.0,
        camera_read_noise_e=scene.microscope.camera.read_noise_e,
        stage_positions_zxy=((2.3, -4.6, 1.15),) * positions,
        acquired_plane_count=pixels.shape[-3],
        plane_period_s=simulation_parameters["pipeline"]["plane_period_s"],
    )
    process_skewed(
        dataset.path,
        acquisition=dataset.metadata,
        deconvolve=False,
        max_projection=not registered,
        create_fused_max_projection=positions > 1 and not registered,
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
def roi_run(request, tmp_path: Path, opm_v2_acquisition_factory, roi_camera_scene):
    """Build a multi-channel acquisition with two selected nonzero positions."""
    import importlib
    from types import SimpleNamespace

    from opm_processing.dataio.roi import PhysicalRoi
    from opm_processing.imageprocessing.opmtools import deskew_shape_estimator

    process = importlib.import_module("opm_processing.process")
    scene = roi_camera_scene(
        positions=3, channels=getattr(request, "param", 3), vary_positions=True
    )
    pixels = scene.pixels
    dataset = opm_v2_acquisition_factory(
        pixels,
        camera_conversion=1.0,
        camera_read_noise_e=scene.microscope.camera.read_noise_e,
        stage_positions_zxy=((0.0, 0.0, 0.0),) * 3,
    )
    source, metadata = dataset.path, dataset.metadata
    pixel = scene.microscope.pixel_size_um
    halo = np.asarray(scene.microscope.psf_half_extent_zyx_um)[1:]
    low = np.floor((scene.centers_zyx_um[:, 1:].min(axis=0) - halo) / pixel)
    high = np.ceil((scene.centers_zyx_um[:, 1:].max(axis=0) + halo) / pixel)
    height, width = deskew_shape_estimator(
        pixels.shape[-3:],
        scene.microscope.angle_deg,
        scene.microscope.scan_step_um,
        pixel,
        crop_after_deskew=False,
    )[0][1:]
    roi = PhysicalRoi(
        bounds_yx_um=tuple(np.array((low[0], high[0], low[1], high[1])) * pixel),
        source_path=tmp_path / "sample_max_z_fused.ome.zarr",
        grid_origin_yx_um=(0.0, 0.0),
        pixel_size_yx_um=(pixel, pixel),
        position_indices=(1, 2),
        tile_footprints=tuple(
            {
                "time_index": 0,
                "position_index": position,
                "origin_zyx_um": [0.0, 0.0, 0.0],
                "bounds_yx_um": [0.0, height * pixel, 0.0, width * pixel],
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
        options={
            "acquisition": metadata,
            "roi_selection": roi,
            "output_dir": tmp_path / "sample_roi",
            "max_projection": False,
            "create_fused_max_projection": False,
        },
        truth=scene,
    )


@pytest.fixture(scope="session")
def roi_camera_scene(simulated_camera, simulation_parameters):
    """Cache immutable optical observations and original physical point locations.

    Returns
    -------
    callable
        Build calibrated timelapse/channel camera observations. Both original
        emitters remain inside the detector field and the selected ROI.
    """
    from types import SimpleNamespace

    from opm_processing.imageprocessing.opmpsf import camera_grid_center

    @lru_cache(maxsize=2)
    def create(*, times=1, positions=1, channels=1, vary_positions=False, shape=None):
        """Render the existing physical point object without a new image simulator."""
        if shape is None:
            shape = (
                simulation_parameters["whole_scan_planes"],
                *simulation_parameters["camera_shape_yx"],
            )
        pixels = np.empty((times, positions, channels, *shape), np.uint16)
        exposures = np.empty((times, positions, channels), np.float32)
        for time, channel, position in np.ndindex(times, channels, positions):
            exposure = (
                1 + time / 4 + channel / 2 + (position / 4 if vary_positions else 0)
            )
            observation = simulated_camera(
                pixels.shape[-3:],
                conversion=1.0,
                seed=42 + 100 * time + 10 * position + channel,
                exposure=exposure,
                budget=(3000, 4000),
            )
            pixels[time, position, channel] = observation.raw
            exposures[time, position, channel] = exposure
            if (time, position, channel) == (0, 0, 0):
                expectation = observation.expectation
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
        for values in (pixels, exposures, centers):
            values.setflags(write=False)
        return SimpleNamespace(
            pixels=pixels,
            expectation=expectation,
            exposures=exposures,
            centers_zyx_um=centers,
            microscope=microscope,
        )

    return create


@pytest.fixture
def assert_roi_point_accuracy():
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
        required_point=None,
    ):
        """Measure actual stored values against fixed physical source locations."""
        origin, spacing = np.asarray(origin), np.asarray(spacing)
        expected = scene.centers_zyx_um + np.asarray(raw_origin)
        halo = np.asarray(scene.microscope.psf_half_extent_zyx_um)
        upper = origin + (np.asarray(data.shape[-3:]) - 1) * spacing
        eligible = np.all(
            (expected - halo >= origin) & (expected + halo <= upper), axis=1
        )
        if required_point is not None:
            assert eligible[required_point], (
                "Intended emitter's PSF is clipped by its ROI"
            )
        split = expected[:, 2].mean()
        columns = origin[2] + np.arange(data.shape[-1]) * spacing[2]
        checked = 0
        for time, channel in np.ndindex(data.shape[:2]):
            if channels is not None and channel not in channels:
                continue
            for point, target in enumerate(expected):
                if not eligible[point]:
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
        expected_counts = scene.exposures.sum(axis=1)
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
        assert checked, "ROI contains no complete original optical emitter aperture"

    return check
