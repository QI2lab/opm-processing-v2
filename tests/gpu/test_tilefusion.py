"""GPU registration of physical tile overlaps and stored acquisition workflows."""

import numpy as np
import pytest

from opm_processing.imageprocessing import tilefusion as tilefusion_module

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("cupy_gpu")]


@pytest.mark.parametrize("half_width", [1, 3, 5])
def test_registration_similarity_uses_acquired_window_support(
    cupy_gpu, half_width, aperiodic_registration_object
):
    """Identical oblique samples have unit similarity only with a valid window."""
    image = aperiodic_registration_object((17, 19, 21), seed=321).astype(np.float32)
    z, y, _ = np.indices(image.shape)
    mask = np.abs(z - y) < half_width
    image, mask = cupy_gpu.asarray(image), cupy_gpu.asarray(mask)
    score = tilefusion_module._registration_similarity(
        image, image, (0, 0, 0), 15, masks=(mask, mask), registration_axis=1
    )
    if half_width == 1:
        assert score == -np.inf
    else:
        assert score == pytest.approx(1.0)


def test_zero_registration_channel_is_excluded_before_patch_reads(
    tile_fusion,
) -> None:
    """Do not schedule registration reads for a pair containing a zero tile."""
    fusion = tile_fusion(
        [np.ones((1, 4, 4, 4), np.uint16), np.zeros((1, 4, 4, 4), np.uint16)],
        [(0, 0, 0), (0, 0, 2)],
        downsample_factors=(1, 1, 1),
        ssim_window=3,
        threshold=0.0,
        max_registration_shift_zyx=(2, 2, 2),
    )

    fusion.refine_tile_positions_with_cross_correlation(ch_idx=0)

    assert fusion.pairwise_metrics == {}


@pytest.mark.parametrize("limit_y", (None, 50))
@pytest.mark.parametrize("depth_overlap", (20, 13))
def test_depth_registration_handles_large_xy_drift(
    registration_pair, shifted_slab_pair, block_fusion, limit_y, depth_overlap
):
    """Recover a thin-Z overlap with 60-pixel Y drift within the prescribed limit."""
    pair = shifted_slab_pair(
        (32, 180, 200), 32 - depth_overlap, (0, -60, 70), seed=71, spacing=(1, 1, 1)
    )
    fusion = registration_pair(
        pair.volumes,
        pair.positions,
        spacing=pair.spacing,
        sampling=(3, 5, 5),
        max_registration_shift_zyx=(20, 100 if limit_y is None else limit_y, 100),
    )
    del fusion._pair_registration_limits  # Exercise explicit, shared bounds.

    fusion.refine_tile_positions_with_cross_correlation()
    fusion.optimize_shifts()
    if limit_y is None:
        assert set(fusion.pairwise_metrics) == {(0, 1)}
        np.testing.assert_allclose(fusion.global_offsets[1], (0, -60, 70), atol=1)
        origins = (
            np.asarray(pair.positions) / pair.spacing
            + fusion.global_offsets
            + pair.starts[0]
        )
        renderer = block_fusion(
            [volume[None] for volume in pair.volumes],
            origins,
            pair.truth.shape,
            block_shape=[1, 1, 7, 31, 37],
            output_dtype=np.dtype(np.float32),
        )
        renderer._fuse_by_blocks()
        actual = renderer.fused_ts.read().result()[0, 0]
        np.testing.assert_allclose(
            actual,
            pair.expected_pixels,
            rtol=2e-6,
            atol=0,
        )
        from opm_processing.imageprocessing.camera import camera_correct

        camera = pair.observation.scene.microscope.camera
        photons = camera_correct(actual, camera.offset_adu, camera.electrons_per_adu)
        selected, envelope = pair.accuracy_mask, pair.expected_envelope
        assert selected.any()
        assert np.all(
            np.isfinite(photons[selected])
            & (photons[selected] >= envelope.lower[selected])
            & (photons[selected] <= envelope.upper[selected])
        )
    else:
        assert fusion.pairwise_metrics == {}
