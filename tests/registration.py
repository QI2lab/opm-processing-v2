"""Shared physical registration units for the actual CPU and CUDA backends."""

import numpy as np
import pytest

from opm_processing.imageprocessing import tilefusion


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
