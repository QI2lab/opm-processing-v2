"""Verify deskew origins against independently indexed stage trajectories."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from opm_processing.imageprocessing.coordinates import deskewed_stage_scan_offsets


@pytest.mark.unit
@pytest.mark.parametrize("step", [0.4, -0.4])
@pytest.mark.parametrize(("leading", "trailing"), [(0, 0), (2, 1), (1, 3)])
def test_reordered_scan_landmarks_keep_recorded_stage_coordinates(
    stage_trajectory_metadata, step, leading, trailing
):
    """First, middle, and last retained planes retain their stage coordinates."""
    acquisition = stage_trajectory_metadata(
        steps=(step,), leading=leading, trailing=trailing
    )
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
def test_origins_use_individual_scan_spans_without_rounding(stage_trajectory_metadata):
    """Distinct recorded trajectory spans produce distinct physical origins."""
    acquisition = stage_trajectory_metadata(steps=(0.4123456789, 0.6123456789))
    np.testing.assert_allclose(
        deskewed_stage_scan_offsets(acquisition)[:, 1],
        (-4.123456789, -6.123456789),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.unit
def test_legacy_positive_scan_without_endpoints_uses_recorded_step(
    stage_trajectory_metadata,
):
    """Legacy stage records retain positive scanning with recorded spacing."""
    acquisition = replace(
        stage_trajectory_metadata(leading=2),
        scan_start_positions_xyz=(),
        scan_end_positions_xyz=(),
    )
    assert acquisition.scan_axis_reversed is True
    assert deskewed_stage_scan_offsets(acquisition)[0, 1] == -3.2


@pytest.mark.unit
def test_nonstage_geometry_has_no_stage_scan_displacement(stage_trajectory_metadata):
    """Mirror scans do not inherit stage endpoint offsets."""
    acquisition = replace(stage_trajectory_metadata(), mode="mirror")
    np.testing.assert_array_equal(deskewed_stage_scan_offsets(acquisition), [[0, 0, 0]])


@pytest.mark.unit
def test_mixed_stage_directions_are_rejected(stage_trajectory_metadata):
    """A global scan reorder cannot safely process mixed trajectory directions."""
    acquisition = stage_trajectory_metadata(steps=(0.4, -0.4))
    with pytest.raises(ValueError, match="mixed scan directions"):
        _ = acquisition.scan_axis_reversed
    with pytest.raises(ValueError, match="mixed scan directions"):
        deskewed_stage_scan_offsets(acquisition)


@pytest.mark.unit
def test_scan_crop_must_leave_retained_samples(stage_trajectory_metadata):
    """Reject a nominal origin when every scan sample has been removed."""
    acquisition = stage_trajectory_metadata(leading=7, trailing=4)
    with pytest.raises(ValueError, match="retain at least one"):
        deskewed_stage_scan_offsets(acquisition)
