"""Geometry controls automatic search envelopes without moving tile origins."""

from dataclasses import replace

import numpy as np
import pytest

from opm_processing.dataio.acquisition import inspect_acquisition
from opm_processing.imageprocessing.opmtools import deskew_shape_estimator
from opm_processing.imageprocessing.tilefusion import (
    TileFusion,
    _aligned_registration_views,
    _infer_registration_limits,
    _refine_depth_candidate,
    _ssim,
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
    registration_scene, shape, blank_side, level
):
    """Reject empty or featureless overlaps before histogram matching."""
    fixed = registration_scene((1, *shape) if len(shape) == 2 else shape).raw.astype(
        np.float32
    )
    if len(shape) == 2:
        fixed = fixed[0]
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
    registration_scene,
    ramp_in_fixed,
):
    """Retain valid intensity registration when directional edges have zero energy."""
    plane = registration_scene((1, 25, 31)).expectation[0]
    fixed = np.broadcast_to(plane, (7, 25, 31)).astype(np.float32).copy()
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


def test_recorded_depth_overlap_recovers_lateral_pair_beyond_old_z_budget(
    registration_pair,
    registration_observation_pair,
):
    """Recover independent slab sampling with a nine-pixel axial stage error."""
    observation = registration_observation_pair((64, 96, 96), (9, 0, 64), seed=186)
    volumes = observation.volumes
    positions = [(0, 0, 0), (0, 0, 60)]
    limits = _infer_registration_limits(
        positions,
        [volume.shape for volume in volumes],
        (1, 1, 1),
        (3, 2, 2),
        30,
        overlap_zyx_um=(12.8, 24, 24),
    )
    fusion = registration_pair(
        volumes,
        positions,
        spacing=(1, 1, 1),
        sampling=(3, 2, 2),
        ssim_window=7,
        max_registration_shift_zyx=limits[0, 1],
    )
    fusion._pair_registration_limits = limits
    fusion.refine_tile_positions_with_cross_correlation()
    np.testing.assert_allclose(fusion.pairwise_metrics[0, 1][:3], (9, 0, 4), atol=1)
    assert fusion.pairwise_metrics[0, 1][3] > 0.95


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
def test_context_rejects_background_without_matching_specimen(
    uniform, camera_calibration
):
    """Similar background levels must not manufacture a depth match."""
    from tests.simulation.camera import digitize

    fields = [np.full((25, 48, 40), 1000, np.float32) for _ in range(2)]
    if not uniform:
        camera = replace(camera_calibration, electrons_per_adu=100, offset_adu=0)
        fields = [
            digitize(field * 100, camera, seed=seed).astype(np.float32)
            for seed, field in enumerate(fields)
        ]
    masks = tuple(np.ones_like(field) for field in fields)
    _, score = _refine_depth_candidate(*fields, masks, 12, (0, 0), 9, 7)
    assert not np.isfinite(score)


@pytest.mark.parametrize("noise", [0.0, 0.3])
def test_high_ssim_shared_xy_background_cannot_establish_a_z_registration(
    noise, registration_scene, camera_calibration
):
    """Shared shading can have almost perfect SSIM without shared axial features."""
    from tests.simulation.camera import digitize

    background = registration_scene((1, 48, 40)).expectation[0]
    fixed = np.broadcast_to(background, (48, 48, 40)).copy()
    moving = fixed.copy()
    if noise:
        dose = float(background.mean() / noise**2)
        camera = replace(camera_calibration, electrons_per_adu=dose, offset_adu=0)
        fixed = digitize(fixed * dose, camera, seed=297)
        moving = digitize(moving * dose, camera, seed=298)
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


def test_similarity_scores_common_geometric_support_and_rejects_unrelated_signal(
    registration_scene,
):
    """Exclude opposing wedge padding while retaining an independent mismatch check."""
    truth = registration_scene((21, 45, 37), seed=38).expectation.astype(np.float32)
    truth[:, 23:26, 17:20] = 0  # Supported dark specimen pixels remain eligible.
    z, y, _x = np.indices(truth.shape)
    fixed_support = y >= z
    moving_support = y < 45 - z
    common = np.asarray(fixed_support & moving_support)
    fixed = np.asarray(truth * fixed_support)
    moving = np.asarray(truth * moving_support)
    assert _ssim(fixed, moving, 7) < 0.7
    assert _ssim(fixed, moving, 7, valid_mask=common) == pytest.approx(1, abs=1e-6)
    unrelated = registration_scene(truth.shape, seed=39).expectation.astype(np.float32)
    assert _ssim(fixed, unrelated, 7, valid_mask=common) < 0.7


@pytest.mark.parametrize("unrelated", (False, True))
def test_thin_depth_registration_scores_only_common_planes(
    unrelated, registration_scene
):
    """Use matched XY samples when geometric gaps prevent a 3D SSIM window."""
    truth = registration_scene((5, 45, 37), seed=83).expectation.astype(np.float32)
    support = np.zeros(truth.shape, bool)
    support[[0, 2, 4], 4:-4, 4:-4] = True
    fixed = np.asarray(truth.astype(np.float32))
    moving = np.asarray(
        registration_scene(truth.shape, seed=84).expectation.astype(np.float32)
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
