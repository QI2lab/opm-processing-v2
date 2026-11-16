"""CuPy/cuCIM registration on known signals and physical overlap supports."""

from dataclasses import replace

import numpy as np
import pytest

from opm_processing.imageprocessing.tilefusion import (
    TileFusion,
    _aligned_registration_views,
    _infer_registration_limits,
    _refine_depth_candidate,
    _ssim,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("cupy_gpu")]


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
    aperiodic_registration_object, shape, blank_side, level
):
    """Reject empty or featureless overlaps before histogram matching."""
    fixed = aperiodic_registration_object(
        (1, *shape) if len(shape) == 2 else shape, seed=51
    ).astype(np.float32)
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
    aperiodic_registration_object,
    ramp_in_fixed,
):
    """Retain valid intensity registration when directional edges have zero energy."""
    # Original background: 30 incident photons * QE=.8, before camera gain/offset.
    plane = 24 + aperiodic_registration_object((1, 25, 31), seed=51)[0]
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
    # The nuisance ramp changes intensity, so correct XY alignment need not
    # have near-unit SSIM. It must still retain a valid registration score.
    assert np.isfinite(score)


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
def test_aligned_registration_views_preserve_fractional_translation(cupy_gpu, offsets):
    """Recover an analytic intensity ramp without rounding shifts or scoring padding."""
    shape = (11, 13, 17)[-len(offsets) :]
    coordinates = np.indices(shape, dtype=np.float32)
    slopes = np.arange(1, len(offsets) + 1, dtype=np.float32)
    fixed = 100 + np.einsum("i,i...->...", slopes, coordinates)
    moving = fixed + np.dot(slopes, offsets)
    reference, aligned = _aligned_registration_views(
        cupy_gpu.asarray(fixed), cupy_gpu.asarray(moving), offsets
    )
    # For a linear field, interpolation recovers the exact continuous object.
    assert reference.shape == tuple(
        length - int(np.ceil(abs(offset)))
        for length, offset in zip(shape, offsets, strict=False)
    )
    np.testing.assert_allclose(
        cupy_gpu.asnumpy(reference), cupy_gpu.asnumpy(aligned), atol=2e-5, rtol=0
    )


@pytest.mark.parametrize("uniform", (True, False))
def test_context_rejects_background_without_matching_specimen(
    cupy_gpu, uniform, camera_calibration
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
    fields = [cupy_gpu.asarray(field) for field in fields]
    masks = tuple(cupy_gpu.ones_like(field) for field in fields)
    _, score = _refine_depth_candidate(*fields, masks, 12, (0, 0), 9, 7)
    assert not np.isfinite(score)


@pytest.mark.parametrize("noise", [0.0, 0.3])
def test_high_ssim_shared_xy_background_cannot_establish_a_z_registration(
    cupy_gpu, noise, aperiodic_registration_object, camera_calibration
):
    """Shared shading can have almost perfect SSIM without shared axial features."""
    from tests.simulation.camera import digitize

    background = aperiodic_registration_object((1, 48, 40), seed=51)[0]
    fixed = np.broadcast_to(background, (48, 48, 40)).copy()
    moving = fixed.copy()
    if noise:
        dose = float(background.mean() / noise**2)
        camera = replace(camera_calibration, electrons_per_adu=dose, offset_adu=0)
        fixed = digitize(fixed * dose, camera, seed=297)
        moving = digitize(moving * dose, camera, seed=298)
    fixed, moving = (
        cupy_gpu.asarray(field, dtype=cupy_gpu.float32) for field in (fixed, moving)
    )
    assert _ssim(fixed, moving, 7) > 0.99
    masks = (cupy_gpu.ones_like(fixed), cupy_gpu.ones_like(moving))
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
    cupy_gpu,
    aperiodic_registration_object,
):
    """Exclude opposing wedge padding while retaining an independent mismatch check."""
    # The original 30-photon background at QE=.8 distinguishes supported
    # specimen from zero padding; the central dark patch remains deliberate.
    truth = 24 + aperiodic_registration_object((21, 45, 37), seed=38).astype(np.float32)
    truth[:, 23:26, 17:20] = 0  # Supported dark specimen pixels remain eligible.
    z, y, _x = np.indices(truth.shape)
    fixed_support = y >= z
    moving_support = y < 45 - z
    common = cupy_gpu.asarray(fixed_support & moving_support)
    fixed = cupy_gpu.asarray(truth * fixed_support)
    moving = cupy_gpu.asarray(truth * moving_support)
    assert _ssim(fixed, moving, 7) < 0.7
    assert _ssim(fixed, moving, 7, valid_mask=common) == pytest.approx(1, abs=1e-6)
    unrelated = 24 + aperiodic_registration_object(truth.shape, seed=39).astype(
        np.float32
    )
    assert _ssim(fixed, cupy_gpu.asarray(unrelated), 7, valid_mask=common) < 0.7


@pytest.mark.parametrize("unrelated", (False, True))
def test_thin_depth_registration_scores_only_common_planes(
    cupy_gpu, unrelated, aperiodic_registration_object
):
    """Use matched XY samples when geometric gaps prevent a 3D SSIM window."""
    truth = aperiodic_registration_object((5, 45, 37), seed=83).astype(np.float32)
    support = np.zeros(truth.shape, bool)
    support[[0, 2, 4], 4:-4, 4:-4] = True
    fixed = cupy_gpu.asarray(truth.astype(np.float32))
    moving = cupy_gpu.asarray(
        aperiodic_registration_object(truth.shape, seed=84).astype(np.float32)
        if unrelated
        else truth.astype(np.float32)
    )
    masks = (cupy_gpu.asarray(support), cupy_gpu.asarray(support))
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
