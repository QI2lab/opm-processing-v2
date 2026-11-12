"""Numerical and storage-boundary unit tests for tile fusion."""

import math
from itertools import combinations

import numpy as np
import pytest

from opm_processing.dataio.position_collection import (
    create_variable_position_collection,
    open_position_collection,
)
from opm_processing.dataio.roi import PhysicalRoi
from opm_processing.imageprocessing import tilefusion as tilefusion_module
from opm_processing.imageprocessing.coordinates import (
    stage_positions_to_image_coordinates,
)
from opm_processing.imageprocessing.tilefusion import TileFusion

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", (np.uint16, np.float32))
@pytest.mark.parametrize("strided", (False, True))
@pytest.mark.parametrize("shared", (False, True))
@pytest.mark.parametrize("partial", (False, True))
def test_weighted_fusion_rows_preserve_masks_offsets_and_output_range(
    weighted_fusion_case, dtype, strided, shared
):
    """Preserve known affine samples, channel masks and scalar weighted counts."""
    case = weighted_fusion_case
    for source, pixels, channels in zip(
        (case.first, case.second),
        case.source_views,
        case.present,
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
        )
    # Both views have the same prescribed feather weights. Their normalized
    # mixture has closed-form scalar values: at the first location, a real
    # black pixel and ten photons give five, not ten. The other
    # channels exercise absent contributors and a value above uint16 range.
    np.testing.assert_array_equal(
        case.accumulated[:, 1, 2, 3],
        np.asarray(case.origin_sum, np.float32) * case.coverage_factor,
    )
    negative_position = (1, 2, 3 + case.negative_x)
    assert case.accumulated[(0, *negative_position)] == (
        case.negative_sum * case.coverage_factor
    )
    assert case.weights[(0, *negative_position)] == (
        case.negative_weight * case.coverage_factor
    )
    for coordinate, _float_values, _uint_values, weight_values, coverage in case.probes:
        np.testing.assert_array_equal(
            case.weights[(slice(None), *coordinate)],
            np.asarray(weight_values, np.float32) * np.float32(coverage),
        )
    np.testing.assert_array_equal(case.accumulated[:, case.untouched], 0)
    np.testing.assert_array_equal(case.weights[:, case.untouched], 0)
    unchanged = case.accumulated.copy()
    destination = np.full((3, 4, 7, 12), 173, dtype=dtype)
    if strided:
        tilefusion_module._normalize_block(
            case.accumulated, case.weights, destination[..., 1:11]
        )
    else:
        tilefusion_module._normalize_block(
            case.accumulated, case.weights, destination, 0, 0, 1
        )
    normalized = destination[..., 1:11]
    for coordinate, float_values, uint_values, _weights, _coverage in case.probes:
        expected = uint_values if dtype == np.uint16 else float_values
        np.testing.assert_array_equal(normalized[(slice(None), *coordinate)], expected)
    assert normalized[(0, *negative_position)] == (0 if dtype == np.uint16 else -10)
    np.testing.assert_array_equal(normalized[:, 1, 3, 3:7], 0)
    np.testing.assert_array_equal(normalized[:, 2, 2, 3:7], 0)
    np.testing.assert_array_equal(normalized[:, case.untouched], 0)
    np.testing.assert_array_equal(destination[..., (0, 11)], 173)
    np.testing.assert_array_equal(case.accumulated, unchanged)


def test_uint16_weighted_fusion_preserves_integer_boundary_rounding() -> None:
    """Retain mixed-precision accumulation before truncating camera pixels."""
    profile = np.linspace(0.1, 1, 128, dtype=np.float32)
    accumulated = np.zeros((1, 1, 1, 1), np.float32)
    weights = np.zeros_like(accumulated)
    photon_counts = (272, 2690)
    x_weights = (profile[73], profile[9])
    feathers = [np.float32(np.float32(0.1) * profile[24]) * x for x in x_weights]
    expected_sum = np.float32(
        math.fsum(
            float(a) * float(w) for a, w in zip(photon_counts, feathers, strict=True)
        )
    )
    expected_weight = np.float32(math.fsum(map(float, feathers)))
    for photon_count, x_weight in zip(photon_counts, x_weights, strict=True):
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
        np.testing.assert_array_equal(source, photon_count)
    np.testing.assert_array_equal(accumulated, expected_sum)
    np.testing.assert_array_equal(weights, expected_weight)
    expected = np.float32(expected_sum / expected_weight)
    assert expected < 779
    output = np.zeros(accumulated.shape, np.uint16)
    tilefusion_module._normalize_block(accumulated, weights, output)
    np.testing.assert_array_equal(output, 778)


def test_max_projection_pyramid_clamps_partial_source_edge_chunks(
    projection_fusion,
) -> None:
    """Never request factor-rounded source bounds beyond level-zero shape."""
    source = np.arange(1 * 3 * 1 * 17 * 19, dtype=np.uint16).reshape(1, 3, 1, 17, 19)
    fusion = projection_fusion(
        (source,),
        ((0, 0),),
        (17, 19),
        chunk_size=512,
    )
    fusion.fused_ts.write(source).result()

    fusion.write_multiscales()

    np.testing.assert_array_equal(
        fusion.multiscale_arrays[-1].read().result(),
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

    # All eight length-ten boxes share the central [8,10]^3 cube.
    expected = set(combinations(range(8), 2))
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
    tile_fusion,
) -> None:
    """Optimize global tile IDs without assuming rectangular T-by-P storage."""
    fusion = tile_fusion(
        [np.ones((1, 2, 4, 4), np.uint16)] * 5,
        [(0, 0, 0)] * 5,
        time_indices=(0, 0, 1, 1, 1),
    )
    fusion.pairwise_metrics = {(0, 1): (0, 0, 2, 1.0), (2, 4): (0, 3, 0, 1.0)}

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
        chunk_shape_yx=(2, 2),
        max_in_flight_writes=2,
        max_workers=2,
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
    )
    assert fusion._tile_channel_nonzero == {
        (tile, channel): tile == 0 or channel > 0
        for tile in range(2)
        for channel in range(channels)
    }
    fusion._fuse_by_blocks()
    expected = np.broadcast_to(
        np.array([100, 150][:channels])[None, :, None, None, None],
        (1, channels, 1, 2, 4),
    )
    np.testing.assert_array_equal(fusion.fused_ts.read().result(), expected)


@pytest.mark.parametrize("dtype", (np.float32, np.uint16))
@pytest.mark.parametrize("block_shape,workers", [(None, 1), ((5, 11, 13), 2)])
def test_fusion_is_independent_of_blocks_and_workers(
    fusion_mosaic, block_fusion, dtype, block_shape, workers
):
    """Recover one known object across crop, coverage, channel and time boundaries."""
    case = fusion_mosaic
    block_shape = case.shape if block_shape is None else block_shape
    fusion = block_fusion(
        case.tiles,
        case.origins,
        case.shape,
        support=case.supports,
        present=case.present,
        block_shape=[1, 1, *block_shape],
        output_dtype=np.dtype(dtype),
        profiles=case.profiles,
        time_indices=(0, 0, 0, 1, 1, 1),
        chunk_shape_yx=block_shape[1:],
        max_workers=workers,
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


def test_variable_roi_tiles_are_stored_cropped_and_reopened(
    record_registration,
    tmp_path,
    acquisition_factory,
) -> None:
    """ROI series retain their independent crop shapes and physical origins."""
    processed_path = tmp_path / "sample_decon_deskewed.ome.zarr"
    shapes = ((1, 1, 2, 4, 4), (1, 1, 2, 5, 3))
    records = [
        {
            "time_index": 0,
            "position_index": 7,
            "origin_zyx_um": [0.0, 10.0, 20.0],
            "shape_tczyx": list(shapes[0]),
            "skewed_shape_syx": [16, 4, 4],
            "deskew_crop_yx": [4, 8, 0, 4],
        },
        {
            "time_index": 0,
            "position_index": 8,
            "origin_zyx_um": [0.0, 12.0, 22.0],
            "shape_tczyx": list(shapes[1]),
            "skewed_shape_syx": [16, 4, 3],
            "deskew_crop_yx": [6, 11, 0, 3],
        },
    ]
    collection = create_variable_position_collection(
        processed_path,
        shapes,
        (1.0, 1.0, 1.0),
        spatial_origins_zyx_um=(
            (0.0, 10.0, 20.0),
            (0.0, 12.0, 22.0),
        ),
    )
    observed = tuple(
        np.arange(np.prod(shape[-3:]), dtype=np.uint16).reshape(shape[-3:])
        + 100 * (index + 1)
        for index, shape in enumerate(shapes)
    )
    for array, pixels in zip(collection.arrays, observed, strict=True):
        array.write(pixels[None, None]).result()
    # Z=0 has no valid deskew interpolation; Z=1 is supported throughout
    # both interior Y crops. Store physically valid synthetic tile data.
    for array in collection.arrays:
        array[:, :, 0].write(np.uint16(0)).result()

    reopened = open_position_collection(processed_path)
    assert [tuple(array.shape) for array in reopened.arrays] == list(shapes)
    assert not any(key.startswith("opm_") for key in reopened.attributes)

    roi_series = tuple(
        {
            key: record[key]
            for key in (
                "time_index",
                "position_index",
                "skewed_shape_syx",
                "deskew_crop_yx",
            )
        }
        for record in records
    )
    acquisition_factory(
        np.ones((1, 9, 1, 16, 4, 4), np.uint16),
        pixel_size_um=1.0,
        scan_axis_step_um=1.0,
        camera_offset=0.0,
        camera_conversion=1.0,
        stage_positions_zxy=((0.0, 0.0, 0.0),) * 9,
    )
    record_registration(
        processed_path,
        records,
        roi_series=roi_series,
    )
    roi = PhysicalRoi(
        bounds_yx_um=(10.0, 17.0, 20.0, 25.0),
        source_path=tmp_path / "sample_max_z_fused.ome.zarr",
        grid_origin_yx_um=(10.0, 20.0),
        pixel_size_yx_um=(1.0, 1.0),
        position_indices=(7, 8),
    )
    fusion = TileFusion(
        processed_path,
        roi_selection=roi,
        max_workers=1,
        multiscale_factors=(2,),
        chunk_shape_yx=(4, 4),
        blend_pixels=(0, 0, 0),
    )
    assert fusion._variable_roi_tiles is True
    assert fusion._reuse_registered_roi_placements is False
    assert fusion._tile_shapes == [(2, 4, 4), (2, 5, 3)]
    assert fusion._tiles_by_time == ((0, 1),)
    fusion._compute_fused_image_space()
    assert fusion.offset_um == (0.0, 10.0, 20.0)
    assert fusion.unpadded_shape == (2, 7, 5)

    fusion._pad_to_multiscale_multiple()
    fusion.fused_ts, fusion.write_block_shape = fusion._prepare_fused_image(
        tmp_path / "sample_fused.ome.zarr"
    )
    fusion._fuse_by_blocks()
    fusion._write_multiscales()
    fused = np.asarray(fusion.fused_ts.read().result())
    assert fused.shape == (1, 1, 2, 8, 6)
    expected = np.zeros((1, 1, 2, 8, 6), np.uint16)
    expected[..., :4, :4] = observed[0]
    expected[..., 2:7, 2:5] = observed[1]
    expected[..., 2:4, 2:4] = (
        (observed[0][:, 2:4, 2:4].astype(np.float32) + observed[1][:, :2, :2]) / 2
    ).astype(np.uint16)
    expected[:, :, 0] = 0
    np.testing.assert_array_equal(fused, expected)
