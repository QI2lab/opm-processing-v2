"""Verify deskew origins against independently indexed stage trajectories."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from opm_processing.dataio.acquisition import AcquisitionMetadata
from opm_processing.imageprocessing.coordinates import deskewed_stage_scan_offsets


def _acquisition(*, steps=(0.4,), leading=0, trailing=0):
    """Construct exact per-plane stage trajectories without accessing pixels."""
    starts = tuple(
        (102.123456789 + i * 100, -9.123456789, 70.0) for i in range(len(steps))
    )
    ends = tuple(
        (x + step * 10, y, z) for (x, y, z), step in zip(starts, steps, strict=True)
    )
    return AcquisitionMetadata(
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


@pytest.mark.unit
@pytest.mark.parametrize("step", [0.4, -0.4])
@pytest.mark.parametrize(("leading", "trailing"), [(0, 0), (2, 1), (1, 3)])
def test_reordered_scan_landmarks_keep_recorded_stage_coordinates(
    step, leading, trailing
):
    """First, middle, and last retained planes retain their stage coordinates."""
    acquisition = _acquisition(steps=(step,), leading=leading, trailing=trailing)
    original = acquisition.stage_positions_zxy
    assert acquisition.scan_axis_reversed is (step > 0)
    order = np.arange(11)[::-1] if step > 0 else np.arange(11)
    retained = order[leading : len(order) - trailing]
    offset = deskewed_stage_scan_offsets(acquisition)[0]
    start_x = acquisition.scan_start_positions_xyz[0][0]
    for local_index in (0, len(retained) // 2, len(retained) - 1):
        raw_index = retained[local_index]
        stage_x = start_x + raw_index * step
        image_y = -start_x + offset[1] + local_index * abs(step)
        assert image_y == pytest.approx(-stage_x, abs=1e-12)
    assert acquisition.stage_positions_zxy == original
    assert original[0][1] == 102.123456789
    np.testing.assert_array_equal(offset[[0, 2]], (0, 0))


@pytest.mark.unit
def test_origins_use_individual_scan_spans_without_rounding():
    """Distinct recorded trajectory spans produce distinct physical origins."""
    acquisition = _acquisition(steps=(0.4123456789, 0.6123456789))
    np.testing.assert_allclose(
        deskewed_stage_scan_offsets(acquisition)[:, 1],
        (-4.123456789, -6.123456789),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.unit
def test_legacy_positive_scan_without_endpoints_uses_recorded_step():
    """Legacy stage records retain positive scanning with recorded spacing."""
    acquisition = replace(
        _acquisition(leading=2),
        scan_start_positions_xyz=(),
        scan_end_positions_xyz=(),
    )
    assert acquisition.scan_axis_reversed is True
    assert deskewed_stage_scan_offsets(acquisition)[0, 1] == -3.2


@pytest.mark.unit
def test_nonstage_geometry_has_no_stage_scan_displacement():
    """Mirror scans do not inherit stage endpoint offsets."""
    acquisition = replace(_acquisition(), mode="mirror")
    np.testing.assert_array_equal(deskewed_stage_scan_offsets(acquisition), [[0, 0, 0]])


@pytest.mark.unit
def test_mixed_stage_directions_are_rejected():
    """A global scan reorder cannot safely process mixed trajectory directions."""
    acquisition = _acquisition(steps=(0.4, -0.4))
    with pytest.raises(ValueError, match="mixed scan directions"):
        _ = acquisition.scan_axis_reversed
    with pytest.raises(ValueError, match="mixed scan directions"):
        deskewed_stage_scan_offsets(acquisition)


@pytest.mark.unit
def test_scan_crop_must_leave_retained_samples():
    """Reject a nominal origin when every scan sample has been removed."""
    acquisition = _acquisition(leading=7, trailing=4)
    with pytest.raises(ValueError, match="retain at least one"):
        deskewed_stage_scan_offsets(acquisition)
