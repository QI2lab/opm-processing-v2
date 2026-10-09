"""Simulated multi-channel acquisitions and physical regions for ROI workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def roi_run(tmp_path: Path, acquisition_factory):
    """Build a three-channel acquisition with two selected nonzero positions.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the source, ROI and processed images.
    acquisition_factory : callable
        Shared writer for calibrated upstream camera data.

    Returns
    -------
    types.SimpleNamespace
        Known camera volumes, physical ROI and independently derived cropped
        fluorescence for original and overwritten data. Numerical processing,
        metadata inspection and fusion remain unmodified.
    """
    import importlib
    from types import SimpleNamespace

    from opm_processing.dataio.roi import PhysicalRoi
    from tests.reference.deskew import constant_deskew

    process = importlib.import_module("opm_processing.process")
    pixels = np.empty((1, 3, 3, 8, 12, 14), np.uint16)
    for position in range(3):
        for channel in range(3):
            pixels[0, position, channel] = 110 + 20 * position + channel
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
                    constant_deskew(
                        (8, 12, 14),
                        10 + 20 * position + channel,
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
            constant_deskew(
                (8, 12, 14),
                801,
                distance=0.4,
                pixel_size=0.115,
                downsample_factor=2,
            )[None, None, :, 2:20, 2:12],
            (1, 3, 3, 18, 10),
        ).astype(np.uint16),
    )
