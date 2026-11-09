"""Geometry controls automatic search envelopes without moving tile origins."""

from dataclasses import replace

import numpy as np
import pytest

from opm_processing.dataio.acquisition import inspect_acquisition
from opm_processing.imageprocessing.opmtools import deskew_shape_estimator
from opm_processing.imageprocessing.tilefusion import (
    _infer_registration_limits,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("z_downsample", [1, 2, 4])
def test_recorded_overlap_uses_deskewed_voxel_spacing(
    current_opm_v2_stage_scan, z_downsample
):
    """The physical allowance stays constant across actual deskew output grids."""
    acquisition = replace(
        inspect_acquisition(current_opm_v2_stage_scan),
        shape=(1, 2, 2, 64, 256, 16),
        scan_axis_overlap_um=30,
        tile_axis_overlap=0.15,
        z_axis_overlap=0.2,
    )
    shape, *_ = deskew_shape_estimator(
        (64, 256, 16), 30, 0.4, 0.115, False, 4, z_downsample
    )
    spacing = (0.115 * z_downsample, 0.115, 0.115)
    limits = _infer_registration_limits(
        [(0, 0, 0), (10.14, 0, 0)],
        [shape, shape],
        spacing,
        (1, 1, 1),
        30,
        overlap_zyx_um=acquisition.registration_overlap_zyx_um,
    )
    # A 20% overlap of a 14.72-um laboratory-Z footprint is 2.944 um.
    # Integer output-grid rounding may add less than one deskewed Z voxel.
    allowance_um = limits[0, 1][0] * spacing[0]
    assert 2.944 <= allowance_um < 2.944 + spacing[0]
    assert limits[0, 1][0] == {1: 26, 2: 13, 4: 7}[z_downsample]


def test_stage_grid_allows_depth_footprint_drift_and_preserves_overlap():
    """Check geometric drift bounds against independent stage-grid dimensions."""
    positions = np.asarray(
        [(-z, -y, x) for z in (0, 11.76) for y in (0, 642.28) for x in (0, 183.89)]
    )
    original = positions.copy()
    # Recorded Z/tile fractions and the stage-scan overlap in micrometers.
    overlap = (256 * 0.115 * 0.5 * 0.2, 30, 1900 * 0.115 * 0.15)
    limits = _infer_registration_limits(
        positions,
        [(64, 6288, 1900)] * 8,
        (0.23, 0.115, 0.115),
        (3, 5, 5),
        30,
        overlap_zyx_um=overlap,
    )
    depth_limits = limits[0, 4]
    assert depth_limits[0] == 13  # Actual matched support gates depth corrections.
    assert depth_limits == (13, 439, 285)
    assert all(
        abs(shift) <= limit
        for shift, limit in zip((1, -182, 9), depth_limits, strict=False)
    )
    assert limits[0, 1][1] < depth_limits[1]
    np.testing.assert_array_equal(positions, original)
    reordered = _infer_registration_limits(
        positions[::-1],
        [(64, 6288, 1900)] * 8,
        (0.23, 0.115, 0.115),
        (3, 5, 5),
        30,
        overlap_zyx_um=overlap,
    )
    assert reordered[3, 7] == depth_limits


def test_projection_limits_ignore_depth_and_disjoint_tiles():
    """Exclude inactive Z corrections and nonoverlapping projection pairs."""
    limits = _infer_registration_limits(
        [(0, 0, 0), (100, 0, 80), (0, 0, 500)],
        [(1, 100, 100)] * 3,
        (1, 1, 1),
        (3, 5, 5),
        30,
        is_2d=True,
    )
    # With no recorded settings, preserve half the actual overlap; do not
    # multiply the geometric overlap or the untiled Y extent by ten percent.
    assert limits == {(0, 1): (0, 50, 10)}


def test_variable_roi_limits_use_intersection_not_common_shape():
    """Bound corrections using the physical intersection of differently sized tiles."""
    limits = _infer_registration_limits(
        [(0, 0, 0), (20, 90, 10)],
        [(32, 100, 80), (16, 50, 40)],
        (1, 1, 1),
        (3, 5, 5),
        45,
    )
    # The cropped Y intersection is only ten pixels despite a large Z/Y allowance.
    assert limits[0, 1] == (6, 5, 20)


def test_recorded_overlap_changes_allowance_without_changing_pair_geometry():
    """Recorded overlap supplies a physical allowance before support caps apply."""
    arguments = (
        [(0, 0, 0), (0, 0, 60)],
        [(64, 96, 96)] * 2,
        (1, 1, 1),
        (3, 2, 2),
        30,
    )
    assert _infer_registration_limits(*arguments, overlap_zyx_um=(12.8, 24, 24)) == {
        (0, 1): (13, 24, 18)
    }
    assert _infer_registration_limits(*arguments, overlap_zyx_um=(6, 8, 8)) == {
        (0, 1): (6, 8, 8)
    }
    scaled = (
        np.asarray(arguments[0]) * 1000,
        arguments[1],
        (1000, 1000, 1000),
        arguments[3],
        arguments[4],
    )
    assert _infer_registration_limits(
        *scaled, overlap_zyx_um=(12800, 24000, 24000)
    ) == {(0, 1): (13, 24, 18)}


def test_geometry_is_invariant_to_physical_unit_conversion():
    """Preserve pixel correction limits under a micrometer-to-nanometer conversion."""
    positions = np.asarray([(0, 0, 0), (10, 60, 5)])
    arguments = ([(32, 100, 80)] * 2, (2, 3, 3), 30)
    actual = _infer_registration_limits(
        positions, arguments[0], (1, 1, 1), arguments[1], arguments[2]
    )
    scaled = _infer_registration_limits(
        positions * 1000, arguments[0], (1000, 1000, 1000), arguments[1], arguments[2]
    )
    assert actual == scaled


def test_no_registration_limits_for_missing_or_disjoint_tiles():
    """Return no search bounds when there are no physically overlapping tiles."""
    assert _infer_registration_limits([], [], (1, 1, 1), (1, 1, 1), 30) == {}
    assert (
        _infer_registration_limits(
            [(0, 0, 0), (100, 100, 100)],
            [(10, 10, 10)] * 2,
            (1, 1, 1),
            (1, 1, 1),
            30,
        )
        == {}
    )
