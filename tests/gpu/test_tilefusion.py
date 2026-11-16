"""GPU registration of physical tile overlaps and stored acquisition workflows."""

import weakref

import numpy as np
import pytest

from opm_processing.imageprocessing import tilefusion as tilefusion_module

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("cupy_gpu")]


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("blank", [False, True])
def test_registration_releases_each_pairs_gpu_arrays_and_fft_workspace(
    cupy_gpu,
    registration_pair,
    aperiodic_registration_object,
    monkeypatch,
    planar,
    blank,
):
    """Release accepted and rejected pairs before processing the next overlap."""
    shape = (1 if planar else 19, 71, 145)
    truth = aperiodic_registration_object(shape, seed=482).astype(np.float32)
    volumes = [
        truth[..., start:stop].copy() for start, stop in ((0, 65), (40, 110), (80, 145))
    ]
    if blank:
        volumes[1].fill(0)
    fusion = registration_pair(
        volumes,
        [(0, 0, x) for x in (0, 40, 80)],
        spacing=(1, 1, 1),
        sampling=(3, 2, 2),
        ssim_window=3,
        max_registration_shift_zyx=(0 if planar else 3, 6, 6),
    )
    pool = cupy_gpu.get_default_memory_pool()
    plans = cupy_gpu.fft.config.get_plan_cache()
    plans.clear()
    pool.free_all_blocks()
    # Cleanup must preserve GPU arrays owned by the caller.
    sentinel = cupy_gpu.arange(17, dtype=cupy_gpu.float32)
    baseline = pool.used_bytes()
    pair_arrays = []
    reductions = 0
    reduce = tilefusion_module.block_reduce

    def observe_reduce(image, *args, **kwargs):
        """Observe the real GPU reductions without retaining their allocations."""
        nonlocal reductions
        assert isinstance(image, cupy_gpu.ndarray)
        if kwargs.get("func") is cupy_gpu.mean:
            overlap_shape = (shape[0], shape[1], 25 if reductions < 2 else 30)
            assert image.shape == (overlap_shape[1:] if planar else overlap_shape)
            if reductions % 2 == 0:
                assert all(reference() is None for reference in pair_arrays), (
                    "Previous registration overlap is still on the GPU"
                )
                assert plans.get_curr_size() == 0
                assert pool.free_bytes() == 0
            result = reduce(image, *args, **kwargs)
            pair_arrays.extend((weakref.ref(image), weakref.ref(result)))
            reductions += 1
            return result
        return reduce(image, *args, **kwargs)

    monkeypatch.setattr(tilefusion_module, "block_reduce", observe_reduce)
    fusion.refine_tile_positions_with_cross_correlation()
    assert reductions == 4  # Two differently shaped physical overlaps.
    assert all(reference() is None for reference in pair_arrays)
    assert plans.get_curr_size() == 0
    assert pool.used_bytes() == baseline
    assert pool.total_bytes() == baseline
    np.testing.assert_array_equal(cupy_gpu.asnumpy(sentinel), np.arange(17))
    if blank:
        assert fusion.pairwise_metrics == {}
    else:
        assert set(fusion.pairwise_metrics) == {(0, 1), (1, 2)}
        for metric in fusion.pairwise_metrics.values():
            np.testing.assert_array_equal(metric[:3], (0, 0, 0))
            assert metric[3] == pytest.approx(1)


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
            pair.union_shape,
            block_shape=[1, 1, 7, 31, 37],
            output_dtype=np.dtype(np.float32),
        )
        renderer._fuse_by_blocks()
        actual = renderer.fused_ts.read().result()[0, 0]
        # Unity feathering gives the arithmetic mean of the acquired operands.
        total = np.zeros(pair.union_shape, np.float64)
        count = np.zeros(pair.union_shape, np.uint8)
        for start, raw in zip(pair.starts, pair.volumes, strict=True):
            selection = tuple(
                slice(offset, offset + length)
                for offset, length in zip(start, raw.shape, strict=True)
            )
            total[selection] += raw
            count[selection] += 1
        expected = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=2e-6,
            atol=0,
        )
    else:
        assert fusion.pairwise_metrics == {}
