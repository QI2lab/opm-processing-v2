"""Shared physical registration units for the actual CPU and CUDA backends."""

import numpy as np
import pytest

from opm_processing.imageprocessing import tilefusion


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-4])
@pytest.mark.parametrize("seed", [485, 492])
def test_bounded_peak_matches_upstream_phase_normalization(
    request, aperiodic_registration_object, scale, seed
):
    """Use the upstream weak-frequency floor even for faint unrelated objects."""
    if request.node.get_closest_marker("gpu"):
        tilefusion.require_gpu_backend()
    backend = tilefusion.xp
    fixed, moving = (
        aperiodic_registration_object((9, 43, 55), seed=value).astype(np.float32)
        for value in (seed, seed + 1)
    )
    fixed = backend.asarray((fixed / fixed.max() * np.float32(scale))[:7, :35, :47])
    moving = backend.asarray(
        (moving / moving.max() * np.float32(scale))[1:8, 4:39, 3:50]
    )
    expected, _, _ = tilefusion.phase_cross_correlation(
        fixed, moving, normalization="phase", upsample_factor=1
    )
    actual = tilefusion._bounded_phase_correlation_peak(
        fixed, moving, np.asarray(fixed.shape) // 2
    )
    np.testing.assert_array_equal(actual.tolist(), backend.asarray(expected).tolist())


@pytest.mark.parametrize("sampling", [(1, 1, 1), (1, 2, 2)])
def test_orthogonal_camera_pairs_recover_known_stage_errors(
    request, orthogonal_registration_pair, sampling
):
    """Recover prescribed stage errors from independently formed oblique images."""
    if request.node.get_closest_marker("gpu"):
        tilefusion.require_gpu_backend()
    backend = tilefusion.xp
    pair = orthogonal_registration_pair
    fixed, moving = (
        tilefusion.block_reduce(backend.asarray(image), sampling, backend.mean)
        for image in (pair.fixed, pair.moving)
    )
    masks = tuple(
        tilefusion.block_reduce(backend.asarray(mask), sampling, backend.min)
        for mask in pair.masks
    )
    shift, score = tilefusion.TileFusion.register_and_score(
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
