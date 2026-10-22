"""Simulated multi-channel acquisitions and physical regions for ROI workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def roi_preview(
    request,
    acquisition_factory,
    small_cpu_kernels,
    simulated_camera,
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
    pixels = np.empty((2, positions, 2, 8, 12, 14), np.uint16)
    for time, position, channel in np.ndindex(pixels.shape[:3]):
        pixels[time, position, channel] = simulated_camera(
            pixels.shape[-3:],
            conversion=1.0,
            exposure=1 + time / 4 + channel / 2,
        ).raw
    if mode == "stage":
        pixels = pixels[:, :, :, ::-1].copy()
    dataset = acquisition_factory(
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
def roi_psf(roi_preview, pipeline_optical_data, tmp_path):
    """Store the same physical PSF used by the shared forward simulator.

    Returns
    -------
    pathlib.Path
        Measured-kernel input sampled at the acquisition scan spacing.
    """
    step = roi_preview.dataset.metadata.scan_axis_step_um
    psf = pipeline_optical_data(step)[2]
    path = tmp_path / "roi_psf.npy"
    np.save(path, psf)
    return path


@pytest.fixture
def roi_run(tmp_path: Path, acquisition_factory, simulated_camera):
    """Build a three-channel acquisition with two selected nonzero positions."""
    import importlib
    from types import SimpleNamespace

    from opm_processing.dataio.roi import PhysicalRoi
    from opm_processing.imageprocessing.opmtools import orthogonal_deskew

    process = importlib.import_module("opm_processing.process")
    pixels = np.empty((1, 3, 3, 8, 12, 14), np.uint16)
    for position in range(3):
        for channel in range(3):
            pixels[0, position, channel] = simulated_camera(
                pixels.shape[-3:],
                conversion=1.0,
                exposure=1 + position / 4 + channel / 2,
            ).raw
    dataset = acquisition_factory(
        pixels, camera_conversion=1.0, stage_positions_zxy=((0.0, 0.0, 0.0),) * 3
    )
    source, raw, metadata = dataset.path, dataset.collection, dataset.metadata
    roi = PhysicalRoi(
        bounds_yx_um=(0.23, 2.3, 0.23, 1.38),
        source_path=tmp_path / "sample_max_z_fused.ome.zarr",
        grid_origin_yx_um=(0.0, 0.0),
        pixel_size_yx_um=(0.115, 0.115),
        position_indices=(1, 2),
        tile_footprints=tuple(
            {
                "time_index": 0,
                "position_index": position,
                "origin_zyx_um": [0.0, 0.0, 0.0],
                "bounds_yx_um": [0.0, 4.0, 0.0, 1.61],
            }
            for position in (1, 2)
        ),
    )
    roi_path = roi.write(tmp_path / "sample_roi.json")

    return SimpleNamespace(
        process=process,
        source=source,
        metadata=metadata,
        raw=raw,
        roi=roi,
        roi_path=roi_path,
        output_dir=tmp_path / "sample_roi",
        expected_tiles=tuple(
            np.stack(
                [
                    orthogonal_deskew(
                        np.maximum(
                            pixels[0, position, channel].astype(np.float32) - 100, 0
                        ),
                        theta=30.0,
                        distance=0.4,
                        pixel_size=0.115,
                        downsample_factor=2,
                    )[:, 2:20, 2:12]
                    for channel in range(3)
                ]
            )[None].astype(np.uint16)
            for position in (1, 2)
        ),
        overwritten_tile=np.broadcast_to(
            orthogonal_deskew(
                np.full((8, 12, 14), 801, dtype=np.float32),
                theta=30.0,
                distance=0.4,
                pixel_size=0.115,
                downsample_factor=2,
            )[None, None, :, 2:20, 2:12],
            (1, 3, 3, 18, 10),
        ).astype(np.uint16),
    )
