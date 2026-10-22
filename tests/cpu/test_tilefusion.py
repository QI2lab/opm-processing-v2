"""Numerical and storage-boundary unit tests for tile fusion."""

import numpy as np
import pytest
import tensorstore as ts

from opm_processing.imageprocessing import tilefusion as tilefusion_module
from opm_processing.imageprocessing.coordinates import (
    stage_positions_to_image_coordinates,
)
from opm_processing.imageprocessing.maxtilefusion import MaxTileFusion
from opm_processing.imageprocessing.tilefusion import TileFusion

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("half_width", [1, 3, 5])
def test_registration_similarity_uses_acquired_window_support(half_width):
    """Identical oblique samples have unit similarity only with a valid window."""
    rng = np.random.default_rng(321)
    image = rng.uniform(10, 100, size=(17, 19, 21)).astype(np.float32)
    z, y, _ = np.indices(image.shape)
    mask = np.abs(z - y) < half_width
    score = tilefusion_module._registration_similarity(
        image, image, (0, 0, 0), 15, masks=(mask, mask), registration_axis=1
    )
    if half_width == 1:
        assert score == -np.inf
    else:
        assert score == pytest.approx(1.0)


@pytest.mark.parametrize("dtype", (np.uint16, np.float32))
@pytest.mark.parametrize("strided", (False, True))
@pytest.mark.parametrize("shared", (False, True))
@pytest.mark.parametrize("partial", (False, True))
def test_weighted_fusion_rows_preserve_masks_offsets_and_output_range(
    weighted_fusion_case, dtype, strided, shared
):
    """Compare strided weighted pixels with independent separable-weight truth."""
    case = weighted_fusion_case
    for source, pixels, channels, gains in zip(
        (case.first, case.second),
        case.source_views,
        case.present,
        case.depth_gains,
        strict=True,
    ):
        tilefusion_module._accumulate_tile_block(
            case.accumulated,
            case.weights,
            pixels if strided else source,
            channels,
            case.support,
            case.z_weights,
            case.y_weights,
            case.x_weights,
            1,
            2,
            3,
            *case.source_origin,
            gains,
        )
        contribution = case.spatial_weights[None] * channels[:, None, None, None]
        corrected = pixels if gains is None else pixels * gains[:, None, None, None]
        case.expected_sum[case.selection] += corrected * contribution
        case.expected_weight[case.selection] += (
            contribution * case.support[None, ..., None]
        )
    np.testing.assert_array_equal(case.accumulated, case.expected_sum)
    np.testing.assert_array_equal(
        case.weights, case.expected_weight[:1] if shared else case.expected_weight
    )
    expected = np.divide(
        case.expected_sum,
        case.expected_weight,
        out=np.zeros_like(case.expected_sum),
        where=case.expected_weight > 0,
    )
    if dtype == np.uint16:
        expected = np.clip(expected, 0, 65535).astype(np.uint16)
    destination = np.full((3, 4, 7, 12), 173, dtype=dtype)
    if strided:
        tilefusion_module._normalize_block(
            case.accumulated, case.weights, destination[..., 1:11]
        )
    else:
        tilefusion_module._normalize_block(
            case.accumulated, case.weights, destination, 0, 0, 1
        )
    np.testing.assert_array_equal(destination[..., 1:11], expected)
    np.testing.assert_array_equal(destination[..., (0, 11)], 173)
    np.testing.assert_array_equal(case.accumulated, case.expected_sum)


def test_uint16_weighted_fusion_preserves_integer_boundary_rounding() -> None:
    """Retain mixed-precision accumulation before truncating camera pixels."""
    profile = np.linspace(0.1, 1, 128, dtype=np.float32)
    accumulated = np.zeros((1, 1, 1, 1), np.float32)
    weights = np.zeros_like(accumulated)
    expected_sum = np.float32(0)
    expected_weight = np.float32(0)
    for photon_count, x_weight in ((272, profile[73]), (2690, profile[9])):
        source = np.full(accumulated.shape, photon_count, np.uint16)
        tilefusion_module._accumulate_tile_block(
            accumulated,
            weights,
            source,
            np.ones(1, bool),
            np.ones((1, 1), bool),
            np.asarray((0.1,), np.float32),
            np.asarray((profile[24],), np.float32),
            np.asarray((x_weight,), np.float32),
            0,
            0,
            0,
        )
        feather = np.float32(np.float32(0.1) * profile[24]) * x_weight
        expected_sum = np.float32(
            np.float64(expected_sum) + np.float64(photon_count) * np.float64(feather)
        )
        expected_weight = np.float32(expected_weight + feather)
        np.testing.assert_array_equal(source, photon_count)
    np.testing.assert_array_equal(accumulated, expected_sum)
    np.testing.assert_array_equal(weights, expected_weight)
    expected = np.float32(expected_sum / expected_weight)
    assert expected < 779
    output = np.zeros(accumulated.shape, np.uint16)
    tilefusion_module._normalize_block(accumulated, weights, output)
    np.testing.assert_array_equal(output, 778)


def test_max_projection_pyramid_clamps_partial_source_edge_chunks(
    fusion_operator,
) -> None:
    """Never request factor-rounded source bounds beyond level-zero shape."""
    source = np.arange(1 * 3 * 1 * 17 * 19, dtype=np.uint16).reshape(1, 3, 1, 17, 19)
    target = np.zeros((1, 3, 1, 2, 2), dtype=np.uint16)
    fusion = fusion_operator(
        MaxTileFusion,
        chunk_size=512,
        multiscale_factors_yx=(1, 16),
        fused_ts=ts.array(source),
    )
    fusion.multiscale_arrays = (fusion.fused_ts, ts.array(target))

    fusion.write_multiscales()

    np.testing.assert_array_equal(
        fusion.multiscale_arrays[1].read().result(),
        source[..., ::16, ::16],
    )


@pytest.mark.parametrize("angle", (None, 30.0, 45.0))
@pytest.mark.parametrize("reverse_z", (False, True))
def test_stage_z_is_reversed_only_for_lab_placement(angle, reverse_z) -> None:
    """A depth-only stage move preserves both orthogonal XY coordinates."""
    stage_positions = np.asarray(
        ((10.0, 20.0, 30.0), (14.0, 20.0, 30.0)),
        dtype=np.float64,
    )
    original = stage_positions.copy()

    coordinates = stage_positions_to_image_coordinates(
        stage_positions,
        reverse_y=True,
        reverse_z=reverse_z,
        opm_angle_deg=angle,
    )

    np.testing.assert_allclose(
        coordinates[1] - coordinates[0],
        (-4.0 if reverse_z else 4.0, 0.0, 0.0),
    )
    np.testing.assert_array_equal(stage_positions, original)


@pytest.mark.parametrize(
    ("dtype", "values"),
    (
        (np.uint16, (10, 30, 50)),
        (np.float32, (0.25, 0.75, 1.25)),
    ),
)
def test_max_projection_fusion_is_chunked_and_sparse(projection_fusion, dtype, values):
    """Fuse distant tiles without allocating arrays the size of the full canvas."""
    tile_shape = (4, 4)
    tile_positions = np.asarray(((0, 0), (0, 2), (0, 10_000)), dtype=float)
    fused_shape = (4, 10_004)
    tiles = tuple(
        np.full((1, 1, 1, *tile_shape), value, dtype=dtype) for value in values
    )
    fusion = projection_fusion(tiles, tile_positions, fused_shape)

    fusion.fuse_tiles()

    expected = np.zeros((1, 1, 1, *fused_shape), dtype=dtype)
    expected[0, 0, 0, :, :2] = values[0]
    expected[0, 0, 0, :, 2:4] = (values[0] + values[1]) / 2
    expected[0, 0, 0, :, 4:6] = values[1]
    expected[0, 0, 0, :, 10_000:10_004] = values[2]
    np.testing.assert_allclose(fusion.fused_ts.read().result(), expected)


def test_zero_max_projection_tile_does_not_reduce_weighted_fusion(
    projection_fusion,
) -> None:
    """Exclude a globally zero projection channel from its neighbor's weights."""
    tile_shape = (3, 4)
    fusion = projection_fusion(
        [
            np.full((1, 1, 1, *tile_shape), 100, np.uint16),
            np.zeros((1, 1, 1, *tile_shape), np.uint16),
        ],
        ((0, 0), (0, 0)),
        tile_shape,
        chunk_size=4,
    )

    fusion.fuse_tiles()

    np.testing.assert_array_equal(
        fusion.fused_ts.read().result(),
        np.full((1, 1, 1, *tile_shape), 100, dtype=np.uint16),
    )


def test_registration_pairs_include_every_physical_overlap() -> None:
    """Select every physical face, edge and corner overlap."""
    positions = [(z, y, x) for z in (0.0, 8.0) for y in (0.0, 8.0) for x in (0.0, 8.0)]

    pairs = tilefusion_module._select_registration_pairs(
        positions,
        tile_shape_zyx=(10, 10, 10),
        pixel_size_zyx=(1.0, 1.0, 1.0),
    )

    expected = {
        (i, j)
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
        if np.all(np.abs(np.asarray(positions[j]) - np.asarray(positions[i])) < 10.0)
    }
    assert set(pairs) == expected


def test_registration_pairs_use_each_cropped_tile_shape() -> None:
    """Detect ROI overlap from crop-adjusted origins and variable extents."""
    positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 6.0), (0.0, 0.0, 10.0)]
    shapes = [(1, 4, 10), (1, 4, 4), (1, 4, 3)]

    pairs = tilefusion_module._select_registration_pairs(
        positions,
        tile_shape_zyx=shapes,
        pixel_size_zyx=(1.0, 1.0, 1.0),
        is_2d=True,
    )

    assert pairs == [(0, 1)]


def test_shift_optimization_supports_variable_tiles_per_timepoint(
    fusion_operator,
) -> None:
    """Optimize global tile IDs without assuming rectangular T-by-P storage."""
    fusion = fusion_operator(
        TileFusion,
        _tile_positions=[(0.0, 0.0, 0.0)] * 5,
        _tiles_by_time=((0, 1), (2, 3, 4)),
        pairwise_metrics={(0, 1): (0, 0, 2, 1.0), (2, 4): (0, 3, 0, 1.0)},
    )

    fusion.optimize_shifts(method="ONE_ROUND")

    np.testing.assert_allclose(fusion.global_offsets[0], (0, 0, 0), atol=1e-12)
    np.testing.assert_allclose(fusion.global_offsets[1], (0, 0, 2), atol=1e-12)
    np.testing.assert_allclose(fusion.global_offsets[2], (0, 0, 0), atol=1e-12)
    np.testing.assert_allclose(fusion.global_offsets[3], (0, 0, 0), atol=1e-12)
    np.testing.assert_allclose(fusion.global_offsets[4], (0, 3, 0), atol=1e-12)


def test_block_fusion_does_not_infer_support_from_pixel_values(block_fusion):
    """Include legitimate zero-valued pixels when geometry marks them valid."""
    tiles = [
        np.full((1, 2, 3, 4), 10.0, dtype=np.float32),
        np.full((1, 2, 3, 4), 30.0, dtype=np.float32),
    ]
    # These dark pixels are inside valid geometry and must retain their weight.
    tiles[1][:, 0, :2, :2] = 0
    tiles[1][:, 1, :1, :2] = 0

    fusion = block_fusion(
        tiles,
        [(0, 0, 0), (0, 0, 2)],
        (2, 3, 6),
        block_shape=[1, 1, 1, 2, 2],
        chunk_y=2,
        chunk_x=2,
        max_in_flight_writes=2,
        _max_workers=2,
    )
    fusion._fuse_by_blocks()

    expected = np.full((1, 1, 2, 3, 6), 10, dtype=np.uint16)
    expected[..., 2:4] = 20
    expected[..., 4:6] = 30
    expected[:, :, 0, :2, 2:4] = 5
    expected[:, :, 1, :1, 2:4] = 5
    np.testing.assert_array_equal(fusion.fused_ts.read().result(), expected)


def test_block_fusion_has_no_zero_weight_line_at_masked_wedge_boundary(
    block_fusion,
):
    """Keep constant signal continuous where a deskew wedge meets a tile edge."""
    tiles = [
        np.full((1, 2, 4, 3), 100.0, dtype=np.float32),
        np.full((1, 2, 4, 3), 100.0, dtype=np.float32),
    ]
    for tile in tiles:
        tile[:, 0, :2, :] = 0
        tile[:, 1, :1, :] = 0

    fusion = block_fusion(
        tiles,
        [(0, 0, 0), (0, 2, 0)],
        (2, 6, 3),
        support=np.array([[False, False, True, True], [False, True, True, True]]),
        y_profile=TileFusion._make_1d_profile(4, 2),
    )

    fusion._fuse_by_blocks()

    expected = np.full((1, 1, 2, 6, 3), 100, dtype=np.uint16)
    expected[:, :, 0, :2] = 0
    expected[:, :, 1, :1] = 0
    np.testing.assert_array_equal(fusion.fused_ts.read().result(), expected)


@pytest.mark.parametrize("channels", [1, 2], ids=["zero-tile", "zero-channel"])
def test_zero_channels_keep_single_contributors_and_blend_valid_overlap(
    block_fusion, channels
):
    """Exclude absent channels while retaining independent weights for valid ones."""
    tiles = [
        np.full((channels, 1, 2, 4), 100, np.float32),
        np.zeros((channels, 1, 2, 4), np.float32),
    ]
    if channels == 2:
        tiles[1][1] = 200
    fusion = block_fusion(
        tiles,
        [(0, 0, 0)] * 2,
        (1, 2, 4),
        present={(0, c): True for c in range(channels)}
        | {(1, c): c > 0 for c in range(channels)},
    )
    fusion._fuse_by_blocks()
    expected = np.broadcast_to(
        np.array([100, 150][:channels])[None, :, None, None, None],
        (1, channels, 1, 2, 4),
    )
    np.testing.assert_array_equal(fusion.fused_ts.read().result(), expected)


def test_zero_registration_channel_is_excluded_before_patch_reads(
    fusion_operator,
) -> None:
    """Do not schedule registration reads for a pair containing a zero tile."""
    fusion = fusion_operator(
        TileFusion,
        downsample_factors=(1, 1, 1),
        ssim_window=3,
        threshold=0.0,
        pairwise_metrics={},
        position_dim=2,
        time_dim=1,
        z_dim=4,
        y_dim=4,
        x_dim=4,
        _pixel_size=(1.0, 1.0, 1.0),
        _tile_positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)],
        _is_2d=False,
        _max_workers=1,
        max_registration_shift_zyx=(2, 2, 2),
        _debug=False,
        _tile_channel_nonzero={(0, 0): True, (1, 0): False},
        position_arrays=(
            ts.array(np.ones((1, 1, 4, 4, 4), np.uint16)),
            ts.array(np.zeros((1, 1, 4, 4, 4), np.uint16)),
        ),
    )

    fusion.refine_tile_positions_with_cross_correlation(ch_idx=0)

    assert fusion.pairwise_metrics == {}


@pytest.mark.parametrize("limit_y", (None, 50))
@pytest.mark.parametrize("depth_overlap", (20, 13))
def test_depth_registration_handles_large_xy_drift(
    registration_pair, shifted_slab_pair, block_fusion, limit_y, depth_overlap, capsys
):
    """Recover a thin-Z overlap with 60-pixel Y drift, or explain rejection."""
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
    fusion.report_registration_connectivity()

    output = capsys.readouterr().out
    if limit_y is None:
        assert set(fusion.pairwise_metrics) == {(0, 1)}
        np.testing.assert_allclose(fusion.global_offsets[1], (0, -60, 70), atol=1)
        assert "Registration warning" not in output
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
            fused_ts=ts.array(np.zeros((1, 1, *pair.truth.shape), np.float32)),
            output_dtype=np.dtype(np.float32),
        )
        renderer._fuse_by_blocks()
        covered = np.zeros(pair.truth.shape, bool)
        for start, volume in zip(pair.starts, pair.volumes, strict=True):
            covered[
                tuple(slice(a, a + n) for a, n in zip(start, volume.shape, strict=True))
            ] = True
        np.testing.assert_allclose(
            renderer.fused_ts.read().result()[0, 0],
            np.where(covered, pair.truth, 0),
            rtol=2e-6,
            atol=0,
        )
    else:
        assert fusion.pairwise_metrics == {}
        assert "exceeded the ZYX shift limits" in output
        assert "disconnected components" in output
        assert "not mutually registered" in output


@pytest.mark.parametrize("dtype", (np.float32, np.uint16))
@pytest.mark.parametrize("block_shape,workers", [(None, 1), ((5, 11, 13), 2)])
def test_fusion_is_independent_of_blocks_and_workers(
    fusion_mosaic, block_fusion, dtype, block_shape, workers
):
    """Recover one physical field across crop, coverage, channel and time boundaries."""
    case = fusion_mosaic
    block_shape = case.shape if block_shape is None else block_shape
    fusion = block_fusion(
        case.tiles,
        case.origins,
        case.shape,
        support=case.supports[0],
        present=case.present,
        block_shape=[1, 1, *block_shape],
        fused_ts=ts.array(np.zeros(case.expected.shape, dtype)),
        output_dtype=np.dtype(dtype),
        _variable_roi_tiles=True,
        _tile_shapes=case.shapes,
        _tile_support_zy=case.supports,
        _tile_profiles=case.profiles,
        _depth_intensity_gains=case.gains,
        _tiles_by_time=((0, 1, 2), (3, 4, 5)),
        time_dim=2,
        chunk_y=block_shape[1],
        chunk_x=block_shape[2],
        _max_workers=workers,
        max_in_flight_writes=2,
        fusion_ram_fraction=0.1 if workers == 2 else 1,
    )
    fusion._fuse_by_blocks()
    actual = fusion.fused_ts.read().result()
    expected = case.expected.astype(dtype)
    if dtype == np.uint16:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(
            actual, expected, rtol=4 * np.finfo(dtype).eps, atol=0
        )
