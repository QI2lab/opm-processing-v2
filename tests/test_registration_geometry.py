"""Geometry controls automatic search envelopes without moving tile origins."""

import numpy as np
import pytest
import tensorstore as ts
from scipy.ndimage import gaussian_filter

from opm_processing.imageprocessing.tilefusion import (
    TileFusion,
    _aligned_registration_views,
    _infer_registration_limits,
    _ssim,
    xp,
)


@pytest.mark.unit
def test_stage_grid_allows_depth_footprint_drift_and_preserves_overlap():
    # Independent dimensions and spacing from the failed four-depth stage scan.
    """Check geometric drift bounds against independent stage-grid dimensions."""
    positions = np.asarray(
        [(-z, -y, x) for z in (0, 11.76) for y in (0, 642.28) for x in (0, 183.89)]
    )
    original = positions.copy()
    limits = _infer_registration_limits(
        positions,
        [(64, 6288, 1900)] * 8,
        (0.23, 0.115, 0.115),
        (3, 5, 5),
        30,
    )
    depth_limits = limits[0, 4]
    assert depth_limits[0] == 6  # Preserve half of the 13-voxel Z overlap.
    assert 240 <= depth_limits[1] <= 260
    assert depth_limits[2] == 31
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
    )
    assert reordered[3, 7] == depth_limits


@pytest.mark.unit
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
    assert limits == {(0, 1): (0, 10, 10)}


@pytest.mark.unit
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
    assert limits[0, 1] == (6, 5, 10)


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
@pytest.mark.parametrize("drift", (180, 270))
def test_inferred_depth_limits_recover_or_reject_measured_drift(drift):
    # A fixed random specimen sampled by two slabs with an independently
    # imposed Y displacement, including one outside the geometric allowance.
    """Recover a known slab displacement and reject drift beyond geometric bounds."""
    rng = np.random.default_rng(51)
    truth = gaussian_filter(rng.uniform(0, 1000, (115, 512 + drift, 64)), (1, 2, 2))
    truth = np.rint(truth).astype(np.uint16)
    volumes = (truth[:64, drift:], truth[51:, :512])
    fusion = TileFusion.__new__(TileFusion)
    fusion._tile_support_zy = [
        np.ones(volume.shape[:2], np.float32) for volume in volumes
    ]
    fusion.downsample_factors = (3, 2, 2)
    fusion.ssim_window = 15
    fusion.threshold = 0.7
    fusion.pairwise_metrics = {}
    fusion.position_dim = 2
    fusion.time_dim = 1
    fusion.z_dim, fusion.y_dim, fusion.x_dim = volumes[0].shape
    fusion._pixel_size = (0.23, 0.115, 0.115)
    fusion._tile_positions = [(0, 0, 0), (51 * 0.23, 0, 0)]
    fusion._tile_shapes = [volumes[0].shape] * 2
    fusion._tiles_by_time = ((0, 1),)
    fusion._tile_time_indices = [0, 0]
    fusion._is_2d = False
    fusion._max_workers = 1
    fusion._pair_registration_limits = _infer_registration_limits(
        fusion._tile_positions,
        fusion._tile_shapes,
        fusion._pixel_size,
        fusion.downsample_factors,
        30,
    )
    fusion.max_registration_shift_zyx = fusion._pair_registration_limits[0, 1]
    fusion._debug = False
    fusion._tile_channel_nonzero = {(0, 0): True, (1, 0): True}
    fusion.position_arrays = tuple(ts.array(volume[None, None]) for volume in volumes)
    fusion.refine_tile_positions_with_cross_correlation()
    if drift == 180:
        assert set(fusion.pairwise_metrics) == {(0, 1)}
        np.testing.assert_allclose(
            fusion.pairwise_metrics[0, 1][:3], (0, -180, 0), atol=1
        )
    else:
        assert fusion.pairwise_metrics == {}


@pytest.mark.unit
@pytest.mark.parametrize("offsets", ((0.5, -1.3), (0.5, -1.3, 2.4), (1, -2, 3)))
def test_aligned_registration_views_preserve_fractional_translation(offsets):
    """Recover an analytic intensity ramp without rounding shifts or scoring padding.

    Parameters
    ----------
    offsets : tuple of float
        Applied moving-image correction in two or three array dimensions.
    """
    shape = (11, 13, 17)[-len(offsets) :]
    coordinates = np.indices(shape, dtype=np.float32)
    slopes = np.arange(1, len(offsets) + 1, dtype=np.float32)
    fixed = 100 + np.einsum("i,i...->...", slopes, coordinates)
    moving = fixed + np.dot(slopes, offsets)
    reference, aligned = _aligned_registration_views(
        xp.asarray(fixed), xp.asarray(moving), offsets
    )
    # For a linear field, interpolation recovers the exact continuous object.
    assert reference.shape == tuple(
        length - int(np.ceil(abs(offset)))
        for length, offset in zip(shape, offsets, strict=False)
    )
    assert bool(xp.allclose(reference, aligned, atol=2e-5, rtol=0))


@pytest.mark.unit
def test_similarity_scores_common_geometric_support_and_rejects_unrelated_signal():
    """Exclude opposing wedge padding while retaining an independent mismatch check."""
    rng = np.random.default_rng(38)
    truth = rng.uniform(0, 1000, (21, 45, 37)).astype(np.float32)
    truth[:, 23:26, 17:20] = 0  # Supported dark specimen pixels remain eligible.
    z, y, _x = np.indices(truth.shape)
    fixed_support = y >= z
    moving_support = y < 45 - z
    common = xp.asarray(fixed_support & moving_support)
    fixed = xp.asarray(truth * fixed_support)
    moving = xp.asarray(truth * moving_support)
    assert _ssim(fixed, moving, 7) < 0.7
    assert _ssim(fixed, moving, 7, valid_mask=common) == pytest.approx(1, abs=1e-6)
    unrelated = xp.asarray(rng.uniform(0, 1000, truth.shape).astype(np.float32))
    assert _ssim(fixed, unrelated, 7, valid_mask=common) < 0.7


@pytest.mark.unit
@pytest.mark.parametrize("unrelated", (False, True))
def test_thin_depth_registration_scores_only_common_planes(unrelated):
    """Use matched XY samples when geometric gaps prevent a 3D SSIM window.

    Parameters
    ----------
    unrelated : bool
        Replace the second observed field with independent signal to check
        that projecting common planes does not manufacture a good match.
    """
    rng = np.random.default_rng(83)
    truth = gaussian_filter(rng.uniform(0, 1000, (5, 45, 37)), (0, 1, 1))
    support = np.zeros(truth.shape, bool)
    support[[0, 2, 4], 4:-4, 4:-4] = True
    fixed = xp.asarray(truth.astype(np.float32))
    moving = xp.asarray(
        gaussian_filter(rng.uniform(0, 1000, truth.shape), (0, 1, 1)).astype(np.float32)
        if unrelated
        else truth.astype(np.float32)
    )
    masks = (xp.asarray(support), xp.asarray(support))
    assert not np.isfinite(_ssim(fixed, moving, 3, valid_mask=masks[0]))
    measured_shift, score = TileFusion.register_and_score(
        fixed,
        moving,
        3,
        max_shift=(0, 2, 2),
        registration_axis=0,
        support_masks=masks,
    )
    if unrelated:
        assert score < 0.7
    else:
        np.testing.assert_allclose(measured_shift, 0, atol=0.1)
        assert score == pytest.approx(1, abs=1e-6)
