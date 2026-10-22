"""Adaptive registration must recompute a consistent complete link graph."""

import numpy as np
import pytest
import tensorstore as ts

from opm_processing.imageprocessing.tilefusion import (
    TileFusion,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "correction,expected,automatic,score",
    [
        ((15, -284, 0), (17, 299, 33), True, 0.9),
        ((-15, 46, 0), (17, 198, 33), True, 0.9),
        ((0, 0, 34), (13, 198, 39), True, 0.9),
        ((0, 420, 0), (13, 441, 33), True, 0.9),
        ((15, -284, 0), None, False, 0.9),
        ((15, -284, 0), None, True, 0.69),
        ((15, -284, 0), None, True, float("nan")),
        ((15, -284, 0), (17, 299, 33), True, 0.7),
    ],
    ids=(
        "zy",
        "negative-z",
        "x",
        "large-y",
        "explicit",
        "low-score",
        "invalid-score",
        "threshold",
    ),
)
def test_limits_follow_rejected_peak_and_sampling_without_widening_other_pairs(
    retry_fusion, correction, expected, automatic, score
):
    """Use measured peaks only when automatic limits and similarity permit it."""
    fusion = retry_fusion
    fusion._automatic_registration_limits = automatic
    original = fusion._pair_registration_limits.copy()
    fusion._rejected_registration_links = {
        (1, 2): (correction, original[1, 2], score),
        (3, 4): ((9999, 9999, 9999), original[3, 4], 0.69),
    }
    assert bool(fusion._expand_registration_limits()) == (expected is not None)
    assert fusion._pair_registration_limits == (
        original if expected is None else {**original, (1, 2): expected}
    )
    assert not fusion._expand_registration_limits()


@pytest.mark.unit
@pytest.mark.parametrize("failed_retry", [False, True])
def test_selective_registration_reads_only_target_pair_and_preserves_raw_outliers(
    fusion_operator, monkeypatch, failed_retry
):
    """Read selected pairs while preserving raw links outside the selection."""
    rng = np.random.default_rng(481)
    specimen = rng.uniform(0, 1000, (1, 1, 16, 16)).astype(np.float32)
    source_tiles = [specimen.copy() for _ in range(6)]
    if failed_retry:
        source_tiles[5] = rng.uniform(0, 1000, specimen.shape).astype(np.float32)
    raw = {
        (0, 1): (0, 2, 3, 0.99),
        (3, 4): (0, 1, 2, 0.99),
        (3, 5): (0, 7, 1, 0.8),  # A prior global outlier remains available.
        (4, 5): (0, 8, 8, 0.99),  # This stale target must be replaced or removed.
    }
    fusion = fusion_operator(
        TileFusion,
        source_tiles=source_tiles[:3],
        position_arrays=tuple(
            ts.array(np.stack((source_tiles[position], source_tiles[position + 3])))
            for position in range(3)
        ),
        time_dim=2,
        position_dim=3,
        _tile_positions=[(0, 0, 0)] * 6,
        _tile_shapes=[(1, 16, 16)] * 6,
        _tiles_by_time=((0, 1, 2), (3, 4, 5)),
        _pixel_size=(1, 1, 1),
        _tile_support_zy=[np.ones((1, 16), np.float32)] * 6,
        _is_2d=True,
        _max_workers=1,
        _debug=False,
        _tile_channel_nonzero={(index, 0): True for index in range(6)},
        downsample_factors=(1, 1, 1),
        ssim_window=3,
        threshold=0.7,
        max_registration_shift_zyx=(0, 8, 8),
        pairwise_metrics=raw.copy(),
        optimized_pairwise_metrics={(0, 1): raw[0, 1], (3, 4): raw[3, 4]},
        _rejected_registration_links={(0, 2): ((0, 10, 0), (0, 8, 8), 0.6)},
        _registration_pair_sampling={(0, 1): (1, 2, 2)},
    )
    reads = []
    read_patch = fusion._read_registration_patch

    def read(tile, *args):
        """Record which source tiles are read for a selective retry."""
        reads.append(tile)
        assert tile in (4, 5)
        return read_patch(tile, *args)

    monkeypatch.setattr(fusion, "_read_registration_patch", read)
    fusion.refine_tile_positions_with_cross_correlation(pairs_to_register={(4, 5)})
    assert sorted(reads) == [4, 5]
    assert {
        pair: value for pair, value in fusion.pairwise_metrics.items() if pair != (4, 5)
    } == {pair: value for pair, value in raw.items() if pair != (4, 5)}
    assert fusion._registration_pair_sampling[0, 1] == (1, 2, 2)
    assert fusion._rejected_registration_links[0, 2][2] == 0.6
    if failed_retry:
        assert (4, 5) not in fusion.pairwise_metrics
    else:
        np.testing.assert_allclose(
            fusion.pairwise_metrics[4, 5][:3], (0, 0, 0), atol=1e-6
        )
        assert fusion.pairwise_metrics[4, 5][3] == pytest.approx(1)


@pytest.mark.unit
@pytest.mark.parametrize(
    "depth_error,retries,automatic",
    [(15, 2, True), (16, 2, True), (15, 0, True), (15, 2, False)],
    ids=("15px", "16px", "no-retries", "explicit-limits"),
)
def test_real_depth_peak_outside_guard_band_is_remeasured_and_recovered(
    registration_pair,
    shifted_slab_pair,
    registration_cache,
    depth_error,
    retries,
    automatic,
):
    """Recover a known physical drift only when retries and automatic limits permit it."""
    pair = shifted_slab_pair((64, 160, 88), 51, (-depth_error, -40, 8), seed=20261017)
    fusion = registration_pair(
        pair.volumes,
        pair.positions,
        ssim_window=7,
        _automatic_registration_limits=automatic,
        registration_retries=retries,
        optimization_rel_threshold=0.5,
        optimization_abs_threshold=1.5,
        channel_to_use=0,
    )
    registration_cache(fusion)
    assert fusion.max_registration_shift_zyx[0] == 13
    fusion.register_with_retries()
    if automatic and retries:
        assert fusion._registration_retry_count >= 1
        np.testing.assert_allclose(
            fusion.pairwise_metrics[0, 1][:3], pair.expected, atol=1
        )
        np.testing.assert_allclose(fusion.global_offsets[1], pair.expected, atol=1)
    else:
        assert fusion._registration_retry_count == 0
        assert (0, 1) not in fusion.pairwise_metrics
        np.testing.assert_array_equal(fusion.global_offsets, np.zeros((2, 3)))
