"""Geometry controls automatic search envelopes without moving tile origins."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from opm_processing.imageprocessing.tilefusion import (
    TileFusion,
    _aligned_registration_views,
    _infer_registration_limits,
    _refine_depth_candidate,
    _ssim,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("sampling", [(1, 1, 1), (1, 2, 2)])
def test_orthogonal_camera_pairs_recover_known_stage_errors(
    orthogonal_registration_pair,
    sampling,
):
    """Recover prescribed stage errors from independently formed oblique images."""
    from skimage.measure import block_reduce

    pair = orthogonal_registration_pair
    fixed, moving = (
        block_reduce(image, sampling, np.mean) for image in (pair.fixed, pair.moving)
    )
    masks = tuple(block_reduce(mask, sampling, np.min) for mask in pair.masks)
    shift, score = TileFusion.register_and_score(
        fixed,
        moving,
        3,
        max_shift=np.array((2, 4, 4)) / sampling,
        registration_axis=pair.axis,
        shift_sampling=sampling,
        support_masks=masks,
    )
    assert np.isfinite(score)
    np.testing.assert_array_equal(np.rint(np.asarray(shift) * sampling), pair.expected)


@pytest.mark.parametrize("shape", [(17, 19), (7, 17, 19)])
@pytest.mark.parametrize(
    ("blank_side", "level"),
    [
        ("fixed", 0),
        ("moving", 0),
        ("both", 0),
        ("fixed", 100),
        ("moving", 100),
        ("both", 100),
    ],
)
def test_empty_or_uninformative_overlap_is_rejected_before_histogram_matching(
    shape, blank_side, level
):
    """Reject empty or featureless overlaps before histogram matching."""
    rng = np.random.default_rng(292)
    fixed = rng.uniform(100, 1000, shape).astype(np.float32)
    moving = fixed.copy()
    if blank_side in ("fixed", "both"):
        fixed.fill(level)
    if blank_side in ("moving", "both"):
        moving.fill(level)

    shift, score = TileFusion.register_and_score(fixed, moving, 3)
    np.testing.assert_array_equal(shift, (0, 0, 0))
    assert score == -np.inf


@pytest.mark.parametrize("ramp_in_fixed", [False, True])
def test_zero_energy_directional_edges_keep_valid_intensity_registration(
    ramp_in_fixed,
):
    """Retain valid intensity registration when directional edges have zero energy."""
    rng = np.random.default_rng(293)
    fixed = np.broadcast_to(rng.uniform(100, 1000, (25, 31)), (7, 25, 31)).copy()
    moving = np.roll(fixed, 3, axis=2).astype(np.float32)
    if ramp_in_fixed:
        fixed += 2 * np.arange(7)[:, None, None]
    fixed = fixed.astype(np.float32)
    shift, score = TileFusion.register_and_score(
        fixed, moving, 3, max_shift=(2, 4, 8), registration_axis=0
    )
    # The moving image is invariant in Z, so the ramp case cannot constrain a
    # unique Z displacement. Its textured XY plane still has a known shift.
    np.testing.assert_allclose(shift[1:], (0, -3), atol=0.1)
    assert abs(shift[0]) <= 2
    if not ramp_in_fixed:
        assert shift[0] == pytest.approx(0, abs=0.1)
    assert score > 0.99


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
    assert depth_limits[0] == 13  # Actual matched support gates depth corrections.
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


@pytest.mark.parametrize("drift", (180, 270))
def test_inferred_depth_limits_recover_or_reject_measured_drift(
    registration_pair, shifted_slab_pair, drift
):
    # A fixed random specimen sampled by two slabs with an independently
    # imposed Y displacement, including one outside the geometric allowance.
    """Recover a known slab displacement and reject drift beyond geometric bounds."""
    pair = shifted_slab_pair((64, 512, 64), 51, (0, -drift, 0))
    fusion = registration_pair(pair.volumes, pair.positions)
    fusion.refine_tile_positions_with_cross_correlation()
    if drift == 180:
        assert set(fusion.pairwise_metrics) == {(0, 1)}
        np.testing.assert_allclose(
            fusion.pairwise_metrics[0, 1][:3], (0, -180, 0), atol=1
        )
    else:
        assert fusion.pairwise_metrics == {}


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("z_limit", (None, 6))
@pytest.mark.parametrize("depth_error", (7, 11))
def test_context_recovers_depth_error_beyond_half_overlap(
    registration_pair, shifted_slab_pair, reverse, z_limit, depth_error
):
    """Recover Z errors from a thirteen-plane nominal intersection.

    Both signs preserve the specimen's Z direction. An explicit six-pixel
    limit rejects the measured correction instead of clamping it.
    """
    pair = shifted_slab_pair((64, 200, 152), 51, (-depth_error, -40, 8), seed=20261009)
    volumes, positions = list(pair.volumes), pair.positions
    if reverse:
        volumes.reverse()
        positions.reverse()
    fusion = registration_pair(volumes, positions)
    if z_limit is not None:
        fusion._pair_registration_limits[0, 1] = (z_limit, 100, 100)
    fusion.max_registration_shift_zyx = fusion._pair_registration_limits[0, 1]
    fusion.refine_tile_positions_with_cross_correlation()
    if z_limit is not None:
        assert fusion.pairwise_metrics == {}
    else:
        expected = np.array((-depth_error, -40, 8)) * (-1 if reverse else 1)
        np.testing.assert_allclose(fusion.pairwise_metrics[0, 1][:3], expected, atol=1)
        fusion.optimize_shifts()
        np.testing.assert_allclose(fusion.global_offsets[1], expected, atol=1)


@pytest.mark.parametrize("offsets", ((0.5, -1.3), (0.5, -1.3, 2.4), (1, -2, 3)))
def test_aligned_registration_views_preserve_fractional_translation(offsets):
    """Recover an analytic intensity ramp without rounding shifts or scoring padding."""
    shape = (11, 13, 17)[-len(offsets) :]
    coordinates = np.indices(shape, dtype=np.float32)
    slopes = np.arange(1, len(offsets) + 1, dtype=np.float32)
    fixed = 100 + np.einsum("i,i...->...", slopes, coordinates)
    moving = fixed + np.dot(slopes, offsets)
    reference, aligned = _aligned_registration_views(
        np.asarray(fixed), np.asarray(moving), offsets
    )
    # For a linear field, interpolation recovers the exact continuous object.
    assert reference.shape == tuple(
        length - int(np.ceil(abs(offset)))
        for length, offset in zip(shape, offsets, strict=False)
    )
    assert bool(np.allclose(reference, aligned, atol=2e-5, rtol=0))


@pytest.mark.parametrize("uniform", (True, False))
def test_context_rejects_background_without_matching_specimen(uniform):
    """Similar background levels must not manufacture a depth match."""
    rng = np.random.default_rng(913)
    fields = [np.full((25, 48, 40), 1000, np.float32) for _ in range(2)]
    if not uniform:
        for field in fields:
            field += gaussian_filter(rng.normal(0, 5, field.shape), 1)
    masks = tuple(np.ones_like(field) for field in fields)
    _, score = _refine_depth_candidate(*fields, masks, 12, (0, 0), 9, 7)
    assert not np.isfinite(score)


@pytest.mark.parametrize("noise", [0.0, 0.3])
def test_high_ssim_shared_xy_background_cannot_establish_a_z_registration(noise):
    """Shared shading can have almost perfect SSIM without shared axial features."""
    rng = np.random.default_rng(297)
    background = gaussian_filter(rng.uniform(100, 1000, (48, 40)), 2)
    fixed = np.broadcast_to(background, (48, 48, 40)).copy()
    moving = fixed.copy()
    fixed += rng.normal(0, noise, fixed.shape)
    moving += rng.normal(0, noise, moving.shape)
    fixed, moving = fixed.astype(np.float32), moving.astype(np.float32)
    assert _ssim(fixed, moving, 7) > 0.99
    masks = (np.ones_like(fixed), np.ones_like(moving))
    _shift, score = _refine_depth_candidate(fixed, moving, masks, -20, (0, 0), 13, 7)
    assert score == -np.inf


def test_identical_line_crops_preserve_zero_displacement():
    """Identical line objects have a valid zero correction along their contrast."""
    specimen = np.full((6, 4, 16), 100, dtype=np.float32)
    specimen[..., 6] += 500
    shift, score = TileFusion.register_and_score(
        specimen, specimen.copy(), 3, max_shift=(3, 2, 8)
    )
    np.testing.assert_array_equal(shift, (0, 0, 0))
    assert score == pytest.approx(1, abs=1e-6)


def test_similarity_scores_common_geometric_support_and_rejects_unrelated_signal():
    """Exclude opposing wedge padding while retaining an independent mismatch check."""
    rng = np.random.default_rng(38)
    truth = rng.uniform(0, 1000, (21, 45, 37)).astype(np.float32)
    truth[:, 23:26, 17:20] = 0  # Supported dark specimen pixels remain eligible.
    z, y, _x = np.indices(truth.shape)
    fixed_support = y >= z
    moving_support = y < 45 - z
    common = np.asarray(fixed_support & moving_support)
    fixed = np.asarray(truth * fixed_support)
    moving = np.asarray(truth * moving_support)
    assert _ssim(fixed, moving, 7) < 0.7
    assert _ssim(fixed, moving, 7, valid_mask=common) == pytest.approx(1, abs=1e-6)
    unrelated = np.asarray(rng.uniform(0, 1000, truth.shape).astype(np.float32))
    assert _ssim(fixed, unrelated, 7, valid_mask=common) < 0.7


@pytest.mark.parametrize("unrelated", (False, True))
def test_thin_depth_registration_scores_only_common_planes(unrelated):
    """Use matched XY samples when geometric gaps prevent a 3D SSIM window."""
    rng = np.random.default_rng(83)
    truth = gaussian_filter(rng.uniform(0, 1000, (5, 45, 37)), (0, 1, 1))
    support = np.zeros(truth.shape, bool)
    support[[0, 2, 4], 4:-4, 4:-4] = True
    fixed = np.asarray(truth.astype(np.float32))
    moving = np.asarray(
        gaussian_filter(rng.uniform(0, 1000, truth.shape), (0, 1, 1)).astype(np.float32)
        if unrelated
        else truth.astype(np.float32)
    )
    masks = (np.asarray(support), np.asarray(support))
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
