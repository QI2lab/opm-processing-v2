"""Real image stores and prescribed trajectories for coordinate metadata units."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opm_processing.imageprocessing.coordinates import STAGE_COORDINATE_VERSION


@pytest.fixture
def stage_trajectory_metadata(acquisition_metadata):
    """Provide prescribed per-plane trajectories for isolated coordinate units."""

    def create(*, steps=(0.4,), leading=0, trailing=0):
        """Construct exact per-plane stage trajectories without accessing pixels."""
        starts = tuple(
            (102.123456789 + i * 100, -9.123456789, 70.0) for i in range(len(steps))
        )
        ends = tuple(
            (x + step * 10, y, z) for (x, y, z), step in zip(starts, steps, strict=True)
        )
        return replace(
            acquisition_metadata,
            path=Path("stage.ome.zarr"),
            storage_format="test",
            mode="stage",
            axes=("t", "p", "c", "z", "y", "x"),
            shape=(1, len(steps), 1, 11, 4, 4),
            array_paths=(),
            acquisition_order=("t", "p", "z", "c"),
            channels=(),
            stage_positions_zxy=tuple((z, x, y) for x, y, z in starts),
            scan_start_positions_xyz=starts,
            scan_end_positions_xyz=ends,
            scan_axis="x",
            scan_axis_step_um=0.4,
            pixel_size_um=0.115,
            angle_deg=30,
            camera_offset=100,
            camera_conversion=0.24,
            excess_scan_positions=0,
            excess_scan_start_positions=leading,
            excess_scan_end_positions=trailing,
            orientations=(),
            sidecar_paths=(),
        )

    return create


@pytest.fixture
def stage_coordinate_case(acquisition_factory, tmp_path):
    """Create two physical scan trajectories with a known ten-micrometer span."""
    positions = ((3.0, 100.0, 20.0), (3.0, 150.0, 20.0))
    dataset = acquisition_factory(
        np.zeros((1, 2, 1, 6, 4, 5), np.uint16),
        mode="stage",
        pixel_size_um=1.0,
        scan_axis_step_um=2.0,
        stage_positions_zxy=positions,
        scan_start_positions_xyz=((100.0, 20.0, 3.0), (150.0, 20.0, 3.0)),
        scan_end_positions_xyz=((110.0, 20.0, 3.0), (160.0, 20.0, 3.0)),
    )
    return SimpleNamespace(
        dataset=dataset,
        directory=tmp_path,
        positions=positions,
        marker={
            "version": STAGE_COORDINATE_VERSION,
            "placement_stage_positions_zxy_um": positions,
        },
    )
