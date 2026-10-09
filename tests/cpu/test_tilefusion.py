"""Numerical unit and storage integration tests for tile fusion."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce as block_reduce_cpu
from typer.testing import CliRunner
from yaozarrs import open_group, v05
from yaozarrs.write.v05 import prepare_image

from opm_processing.dataio.position_collection import (
    create_position_collection,
    open_image_array,
    open_position_collection,
)
from opm_processing.dataio.processing_state import ProcessingState
from opm_processing.fuse import app as fuse_app
from opm_processing.imageprocessing import maxtilefusion as maxtilefusion_module
from opm_processing.imageprocessing import tilefusion as tilefusion_module
from opm_processing.imageprocessing.coordinates import (
    stage_positions_to_image_coordinates,
)
from opm_processing.imageprocessing.maxtilefusion import MaxTileFusion
from opm_processing.imageprocessing.tilefusion import TileFusion


@pytest.mark.unit
@pytest.mark.parametrize("dtype", (np.uint16, np.float32))
@pytest.mark.parametrize("strided", (False, True))
@pytest.mark.parametrize("shared", (False, True))
@pytest.mark.parametrize("partial", (False, True))
def test_weighted_fusion_rows_preserve_masks_offsets_and_output_range(
    dtype, strided, shared, partial
):
    """Compare strided weighted pixels with independent separable-weight truth.

    Parameters
    ----------
    dtype : numpy.dtype
        Destination type, retaining float32 values or clipping uint16 values.
    strided : bool
        Use cropped strided views, or contiguous blocks with explicit crop offsets.
    shared : bool
        Share one denominator across complete channels, or handle missing channels.
    partial : bool
        Include a half-filled interpolation bin with known geometric coverage.
    """
    first = np.arange(3 * 4 * 5 * 10, dtype=np.float32).reshape(3, 4, 5, 10)
    first[0, 1, 1, 1] = 0
    first[0, 1, 1, 3] = -20
    first[1, 1, 1, 1] = 90000
    second = first * np.float32(0.5) + np.float32(10)
    attenuation_per_um = np.asarray((0.025, 0.05, 0.075), np.float32)
    optical_path_um = np.float32(8)
    second *= np.exp(-attenuation_per_um * optical_path_um)[:, None, None, None]
    depth_gains = (None, np.exp(attenuation_per_um * optical_path_um))
    crop = np.s_[:, 1:3, 1:4, 1:9:2] if strided else np.s_[:, 1:3, 1:4, 1:5]
    source_views = (first[crop], second[crop])
    present = (
        (np.ones(3, bool), np.ones(3, bool))
        if shared
        else (np.asarray((True, True, False)), np.asarray((True, False, True)))
    )
    support = np.asarray(((True, False, True), (False, True, True)))
    if partial:
        support = support.astype(np.float32)
        support[0, 0] = 0.5
        for pixels in source_views:
            pixels[:, 0, 0] *= np.float32(0.5)
    z_weights = np.asarray((0.5, 1), np.float32)
    y_weights = np.asarray((1, 0.25, 0.5), np.float32)
    x_weights = np.asarray((1, 0.5, 0.25, 1), np.float32)
    spatial_weights = (z_weights[:, None, None] * y_weights[None, :, None]) * x_weights[
        None, None, :
    ]
    spatial_weights *= (support > 0)[..., None]
    accumulated = np.zeros((3, 4, 7, 10), np.float32)
    weights = np.zeros((1 if shared else 3, *accumulated.shape[1:]), np.float32)
    expected_sum = np.zeros_like(accumulated)
    expected_weight = np.zeros_like(accumulated)
    selection = np.s_[:, 1:3, 2:5, 3:7]
    source_origin = (0, 0, 0) if strided else (1, 1, 1)
    for source, pixels, channels, gains in zip(
        (first, second), source_views, present, depth_gains, strict=True
    ):
        tilefusion_module._accumulate_tile_block(
            accumulated,
            weights,
            pixels if strided else source,
            channels,
            support,
            z_weights,
            y_weights,
            x_weights,
            1,
            2,
            3,
            *source_origin,
            gains,
        )
        contribution = spatial_weights[None] * channels[:, None, None, None]
        corrected = pixels if gains is None else pixels * gains[:, None, None, None]
        expected_sum[selection] += corrected * contribution
        expected_weight[selection] += contribution * support[None, ..., None]
    np.testing.assert_array_equal(accumulated, expected_sum)
    np.testing.assert_array_equal(
        weights, expected_weight[:1] if shared else expected_weight
    )
    expected = np.divide(
        expected_sum,
        expected_weight,
        out=np.zeros_like(expected_sum),
        where=expected_weight > 0,
    )
    if dtype == np.uint16:
        expected = np.clip(expected, 0, 65535).astype(np.uint16)
    destination = np.full((3, 4, 7, 12), 173, dtype=dtype)
    if strided:
        tilefusion_module._normalize_block(accumulated, weights, destination[..., 1:11])
    else:
        tilefusion_module._normalize_block(accumulated, weights, destination, 0, 0, 1)
    np.testing.assert_array_equal(destination[..., 1:11], expected)
    np.testing.assert_array_equal(destination[..., (0, 11)], 173)
    np.testing.assert_array_equal(accumulated, expected_sum)


@pytest.mark.unit
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


@pytest.mark.unit
def test_max_projection_pyramid_clamps_partial_source_edge_chunks(
    tensorstore_dataset,
    fusion_operator,
) -> None:
    """Never request factor-rounded source bounds beyond level-zero shape."""
    source = np.arange(1 * 3 * 1 * 17 * 19, dtype=np.uint16).reshape(1, 3, 1, 17, 19)
    target = np.zeros((1, 3, 1, 2, 2), dtype=np.uint16)
    fusion = fusion_operator(
        MaxTileFusion,
        chunk_size=512,
        multiscale_factors_yx=(1, 16),
        fused_ts=tensorstore_dataset(source),
    )
    fusion.multiscale_arrays = (fusion.fused_ts, tensorstore_dataset(target))

    fusion.write_multiscales()

    np.testing.assert_array_equal(
        fusion.multiscale_arrays[1].read().result(),
        source[..., ::16, ::16],
    )


@pytest.mark.unit
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


@pytest.mark.integration
def test_max_projection_depth_tiles_share_xy_footprint(tmp_path) -> None:
    """Depth tiles with fixed stage XY must fuse into a single tile footprint."""
    source = np.arange(64, dtype=np.uint16).reshape(1, 1, 1, 8, 8)
    source_path = tmp_path / "depth_phantom.ome.zarr"
    collection = create_position_collection(
        source_path, (1, 2, 1, 1, 8, 8), (0.23, 0.115, 0.115)
    )
    for array in collection.arrays:
        array.write(source).result()
    fusion = MaxTileFusion(
        ts_dataset=open_position_collection(source_path).arrays,
        tile_positions=((3939.54, -7456.06, 4207.16), (3800.09, -7456.06, 4207.16)),
        output_path=tmp_path / "depth_max_z_fused.ome.zarr",
        pixel_size=(0.23, 0.115, 0.115),
        opm_angle_deg=30.0,
        blend_pixels=(0, 0),
        chunk_size=4,
    )
    fusion.run()

    np.testing.assert_array_equal(fusion.tile_positions[0], fusion.tile_positions[1])
    written = open_image_array(tmp_path / "depth_max_z_fused.ome.zarr")
    np.testing.assert_array_equal(written.read().result(), source)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dtype", "values"),
    (
        (np.uint16, (10, 30, 50)),
        (np.float32, (0.25, 0.75, 1.25)),
    ),
)
def test_max_projection_fusion_is_chunked_and_sparse(
    tensorstore_dataset, memory_store, fusion_operator, monkeypatch, dtype, values
):
    """Fuse distant tiles without allocating arrays the size of the full canvas."""
    tile_shape = (4, 4)
    tile_positions = np.asarray(((0, 0), (0, 2), (0, 10_000)), dtype=float)
    fused_shape = (4, 10_004)
    tiles = tuple(
        tensorstore_dataset(np.full((1, 1, 1, *tile_shape), value, dtype=dtype))
        for value in values
    )
    fusion = fusion_operator(
        MaxTileFusion,
        pad_y=0,
        pad_x=0,
        ts_dataset=tiles,
        tile_positions=tile_positions,
        pixel_size=np.asarray((1.0, 1.0)),
        offset=(0.0, 0.0),
        time_dim=1,
        channels=1,
        tile_shape=tile_shape,
        fused_shape=fused_shape,
        time_range=None,
        chunk_size=3,
        weight_mask=np.ones(tile_shape, dtype=np.float32),
        output_dtype=np.dtype(dtype),
        fused_ts=memory_store((1, 1, 1, *fused_shape), dtype=dtype),
    )

    real_zeros = maxtilefusion_module.np.zeros

    def reject_full_canvas(shape, *args, **kwargs):
        if tuple(shape) == (fusion.channels, 1, *fusion.fused_shape):
            raise AssertionError("fusion allocated the full mosaic canvas")
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(maxtilefusion_module.np, "zeros", reject_full_canvas)

    fusion.fuse_tiles()

    expected = np.zeros((1, 1, 1, *fused_shape), dtype=dtype)
    expected[0, 0, 0, :, :2] = values[0]
    expected[0, 0, 0, :, 2:4] = (values[0] + values[1]) / 2
    expected[0, 0, 0, :, 4:6] = values[1]
    expected[0, 0, 0, :, 10_000:10_004] = values[2]
    np.testing.assert_allclose(fusion.fused_ts.read().result(), expected)


@pytest.mark.unit
def test_zero_max_projection_tile_does_not_reduce_weighted_fusion(
    tensorstore_dataset,
    memory_store,
    fusion_operator,
) -> None:
    """Exclude a globally zero projection channel from its neighbor's weights."""
    tile_shape = (3, 4)
    fusion = fusion_operator(
        MaxTileFusion,
        pad_y=0,
        pad_x=0,
        ts_dataset=(
            tensorstore_dataset(np.full((1, 1, 1, *tile_shape), 100, dtype=np.uint16)),
            tensorstore_dataset(np.zeros((1, 1, 1, *tile_shape), dtype=np.uint16)),
        ),
        tile_positions=np.asarray(((0, 0), (0, 0)), dtype=float),
        pixel_size=np.asarray((1.0, 1.0)),
        offset=(0.0, 0.0),
        time_dim=1,
        channels=1,
        tile_shape=tile_shape,
        fused_shape=tile_shape,
        time_range=None,
        chunk_size=4,
        weight_mask=np.ones(tile_shape, dtype=np.float32),
        output_dtype=np.dtype(np.uint16),
        fused_ts=memory_store((1, 1, 1, *tile_shape), dtype=np.uint16),
    )

    fusion.fuse_tiles()

    np.testing.assert_array_equal(
        fusion.fused_ts.read().result(),
        np.full((1, 1, 1, *tile_shape), 100, dtype=np.uint16),
    )


@pytest.mark.integration
def test_max_projection_fusion_writes_centered_multiscales_and_offsets(
    tmp_path,
) -> None:
    """Persist every fused max level with rounded physical coordinates."""
    source = np.arange(64, dtype=np.uint16).reshape(1, 1, 1, 8, 8)
    output_path = tmp_path / "max_z_fused.ome.zarr"
    source_path = tmp_path / "ramp_phantom.ome.zarr"
    collection = create_position_collection(
        source_path, (1, 1, 1, 1, 8, 8), (0.34567, 0.1164, 0.1184)
    )
    collection.arrays[0].write(source).result()
    fusion = MaxTileFusion(
        ts_dataset=open_position_collection(source_path).arrays,
        tile_positions=((1.23456, 2.34567, 3.45678),),
        output_path=output_path,
        pixel_size=(0.34567, 0.1164, 0.1184),
        spatial_offset_z_um=4.56789,
        blend_pixels=(0, 0),
        chunk_size=4,
        reverse_stage_y=True,
        reverse_stage_z=False,
    )
    fusion.run()

    reopened = open_group(output_path)
    metadata = reopened.ome_metadata()
    assert isinstance(metadata, v05.Image)
    datasets = metadata.multiscales[0].datasets
    assert [dataset.path for dataset in datasets] == ["0", "1", "2", "3"]
    base_scale = np.asarray(datasets[0].scale_transform.scale[-3:])
    assert datasets[0].translation_transform is not None
    base_translation = np.asarray(datasets[0].translation_transform.translation[-3:])
    expected = source
    for factor, dataset in zip((1, 2, 4, 8), datasets, strict=False):
        assert dataset.scale_transform.scale == [
            1.0,
            1.0,
            0.346,
            round(0.116 * factor, 3),
            round(0.118 * factor, 3),
        ]
        assert dataset.translation_transform is not None
        assert dataset.translation_transform.translation == [
            0.0,
            0.0,
            4.568,
            -2.346,
            3.457,
        ]
        output_index = np.asarray([0, min(2, 7 // factor), min(3, 7 // factor)])
        output_coordinate = np.asarray(
            dataset.translation_transform.translation[-3:]
        ) + output_index * np.asarray(dataset.scale_transform.scale[-3:])
        source_index = output_index * np.asarray([1, factor, factor])
        np.testing.assert_allclose(
            output_coordinate,
            base_translation + source_index * base_scale,
        )
        if factor > 1:
            expected = source[..., ::factor, ::factor]
        np.testing.assert_array_equal(
            reopened[dataset.path].to_tensorstore().read().result(),
            expected,
        )


@pytest.mark.unit
def test_registration_pairs_include_every_physical_overlap() -> None:
    """Match main by retaining face, edge, and corner overlaps."""
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_block_fusion_does_not_infer_support_from_pixel_values(
    memory_store, fusion_operator, monkeypatch
):
    """Include legitimate zero-valued pixels when geometry marks them valid."""
    tiles = [
        np.full((1, 2, 3, 4), 10.0, dtype=np.float32),
        np.full((1, 2, 3, 4), 30.0, dtype=np.float32),
    ]
    # These dark pixels are inside valid geometry and must retain their weight.
    tiles[1][:, 0, :2, :2] = 0
    tiles[1][:, 1, :1, :2] = 0

    fusion = fusion_operator(
        TileFusion,
        source_tiles=tiles,
        fused_ts=memory_store((1, 1, 2, 3, 6)),
        write_block_shape=[1, 1, 1, 2, 2],
        offset_um=(0.0, 0.0, 0.0),
        padded_shape=(2, 3, 6),
        _pixel_size=(1.0, 1.0, 1.0),
        _tile_positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)],
        chunk_y=2,
        chunk_x=2,
        fusion_ram_fraction=1.0,
        max_in_flight_writes=2,
        _max_workers=2,
        _debug=False,
        _deskew_support_zy=np.ones((2, 3), dtype=bool),
        _tile_channel_nonzero={(0, 0): True, (1, 0): True},
    )
    monkeypatch.setattr(
        tilefusion_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=256),
    )

    fusion._fuse_by_blocks()

    expected = np.full((1, 1, 2, 3, 6), 10, dtype=np.uint16)
    expected[..., 2:4] = 20
    expected[..., 4:6] = 30
    expected[:, :, 0, :2, 2:4] = 5
    expected[:, :, 1, :1, 2:4] = 5
    np.testing.assert_array_equal(fusion.fused_ts.read().result(), expected)


@pytest.mark.unit
def test_block_fusion_has_no_zero_weight_line_at_masked_wedge_boundary(
    memory_store,
    fusion_operator,
    monkeypatch,
):
    """Keep constant signal continuous where a deskew wedge meets a tile edge."""
    tiles = [
        np.full((1, 2, 4, 3), 100.0, dtype=np.float32),
        np.full((1, 2, 4, 3), 100.0, dtype=np.float32),
    ]
    for tile in tiles:
        tile[:, 0, :2, :] = 0
        tile[:, 1, :1, :] = 0

    fusion = fusion_operator(
        TileFusion,
        source_tiles=tiles,
        fused_ts=memory_store((1, 1, 2, 6, 3)),
        write_block_shape=[1, 1, 2, 6, 3],
        offset_um=(0.0, 0.0, 0.0),
        padded_shape=(2, 6, 3),
        _pixel_size=(1.0, 1.0, 1.0),
        _tile_positions=[(0.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
        y_profile=TileFusion._make_1d_profile(4, 2),
        x_profile=np.ones(3, dtype=np.float32),
        chunk_y=6,
        chunk_x=3,
        fusion_ram_fraction=1.0,
        max_in_flight_writes=1,
        _max_workers=1,
        _debug=False,
        _deskew_support_zy=np.asarray(
            [[False, False, True, True], [False, True, True, True]], dtype=bool
        ),
        _tile_channel_nonzero={(0, 0): True, (1, 0): True},
    )
    monkeypatch.setattr(
        tilefusion_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=4096),
    )

    fusion._fuse_by_blocks()

    expected = np.full((1, 1, 2, 6, 3), 100, dtype=np.uint16)
    expected[:, :, 0, :2] = 0
    expected[:, :, 1, :1] = 0
    np.testing.assert_array_equal(fusion.fused_ts.read().result(), expected)


@pytest.mark.unit
def test_zero_tile_channel_contributes_neither_signal_nor_fusion_weight(
    memory_store,
    fusion_operator,
    monkeypatch,
):
    """An explicitly zeroed tile must not attenuate an overlapping valid tile."""
    tiles = [
        np.full((1, 1, 2, 4), 100.0, dtype=np.float32),
        np.zeros((1, 1, 2, 4), dtype=np.float32),
    ]

    fusion = fusion_operator(
        TileFusion,
        source_tiles=tiles,
        fused_ts=memory_store((1, 1, 1, 2, 4)),
        write_block_shape=[1, 1, 1, 2, 4],
        offset_um=(0.0, 0.0, 0.0),
        padded_shape=(1, 2, 4),
        _pixel_size=(1.0, 1.0, 1.0),
        _tile_positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        chunk_y=2,
        chunk_x=4,
        fusion_ram_fraction=1.0,
        max_in_flight_writes=1,
        _max_workers=1,
        _debug=False,
        _deskew_support_zy=np.ones((1, 2), dtype=bool),
        _tile_channel_nonzero={(0, 0): True, (1, 0): False},
    )
    monkeypatch.setattr(
        tilefusion_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=4096),
    )

    fusion._fuse_by_blocks()

    np.testing.assert_array_equal(
        fusion.fused_ts.read().result(),
        np.full((1, 1, 1, 2, 4), 100, dtype=np.uint16),
    )


@pytest.mark.unit
def test_single_nonzero_channel_contributor_is_copied_in_overlap(
    memory_store, fusion_operator, monkeypatch
):
    """Copy one valid channel unchanged while blending another channel."""
    tiles = [
        np.full((2, 1, 2, 4), 100.0, dtype=np.float32),
        np.stack(
            (
                np.zeros((1, 2, 4), dtype=np.float32),
                np.full((1, 2, 4), 200.0, dtype=np.float32),
            )
        ),
    ]

    fusion = fusion_operator(
        TileFusion,
        source_tiles=tiles,
        fused_ts=memory_store((1, 2, 1, 2, 4)),
        write_block_shape=[1, 1, 1, 2, 4],
        offset_um=(0.0, 0.0, 0.0),
        padded_shape=(1, 2, 4),
        _pixel_size=(1.0, 1.0, 1.0),
        _tile_positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        chunk_y=2,
        chunk_x=4,
        fusion_ram_fraction=1.0,
        max_in_flight_writes=1,
        _max_workers=1,
        _debug=False,
        _deskew_support_zy=np.ones((1, 2), dtype=bool),
        _tile_channel_nonzero={(0, 0): True, (0, 1): True, (1, 0): False, (1, 1): True},
    )
    monkeypatch.setattr(
        tilefusion_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=4096),
    )

    fusion._fuse_by_blocks()

    np.testing.assert_array_equal(
        fusion.fused_ts.read().result()[:, 0],
        np.full_like(fusion.fused_ts.read().result()[:, 0], 100),
    )
    np.testing.assert_array_equal(
        fusion.fused_ts.read().result()[:, 1],
        np.full_like(fusion.fused_ts.read().result()[:, 1], 150),
    )


@pytest.mark.unit
def test_zero_registration_channel_is_excluded_before_patch_reads(
    tensorstore_dataset,
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
            tensorstore_dataset(np.ones((1, 1, 4, 4, 4), np.uint16)),
            tensorstore_dataset(np.zeros((1, 1, 4, 4, 4), np.uint16)),
        ),
    )

    fusion.refine_tile_positions_with_cross_correlation(ch_idx=0)

    assert fusion.pairwise_metrics == {}


@pytest.mark.unit
@pytest.mark.parametrize("limit_y", (None, 50))
@pytest.mark.parametrize("depth_overlap", (20, 13))
def test_depth_registration_handles_large_xy_drift(
    tensorstore_dataset, fusion_operator, limit_y, depth_overlap, capsys
):
    """Recover a thin-Z overlap with 60-pixel Y drift, or explain rejection."""
    rng = np.random.default_rng(71)
    depth_step = 32 - depth_overlap
    truth = gaussian_filter(
        rng.uniform(0, 1000, (32 + depth_step, 240, 270)), (1, 2, 2)
    )
    truth = np.rint(truth).astype(np.uint16)
    volumes = (truth[:32, 60:240, :200], truth[depth_step:, :180, 70:270])
    fusion = fusion_operator(
        TileFusion,
        _tile_support_zy=[np.ones(volume.shape[:2], np.float32) for volume in volumes],
        downsample_factors=(3, 5, 5),
        ssim_window=15,
        threshold=0.7,
        pairwise_metrics={},
        position_dim=2,
        time_dim=1,
    )
    fusion.z_dim, fusion.y_dim, fusion.x_dim = volumes[0].shape
    fusion._pixel_size = (1.0, 1.0, 1.0)
    fusion._tile_positions = [(0.0, 0.0, 0.0), (float(depth_step), 0.0, 0.0)]
    fusion._tile_shapes = [volumes[0].shape] * 2
    fusion._tiles_by_time = ((0, 1),)
    fusion._tile_time_indices = [0, 0]
    fusion._is_2d = False
    fusion._max_workers = 1
    fusion.max_registration_shift_zyx = (20, 100, 100)
    if limit_y is not None:
        fusion.max_registration_shift_zyx = (20, limit_y, 100)
    fusion._debug = False
    fusion._tile_channel_nonzero = {(0, 0): True, (1, 0): True}
    fusion.position_arrays = tuple(
        tensorstore_dataset(volume[None, None]) for volume in volumes
    )

    fusion.refine_tile_positions_with_cross_correlation()
    fusion.optimize_shifts()
    fusion.report_registration_connectivity()

    output = capsys.readouterr().out
    if limit_y is None:
        assert set(fusion.pairwise_metrics) == {(0, 1)}
        np.testing.assert_allclose(fusion.global_offsets[1], (0, -60, 70), atol=1)
        assert "Registration warning" not in output
    else:
        assert fusion.pairwise_metrics == {}
        assert "exceeded the ZYX shift limits" in output
        assert "disconnected components" in output
        assert "not mutually registered" in output


@pytest.mark.unit
@pytest.mark.parametrize(
    ("downsample_method", "is_2d", "padded_shape"),
    (
        ("stride", False, (4, 8, 8)),
        ("block_mean", False, (4, 8, 8)),
        ("stride", True, (1, 8, 8)),
    ),
    ids=("volume-stride", "volume-block-mean", "projection-stride"),
)
def test_multiscale_storage_round_trip_matches_reference(
    fusion_operator,
    tmp_path,
    downsample_method,
    is_2d,
    padded_shape,
):
    """Reopen exact pyramid data and verify its common physical-space center."""
    fusion = fusion_operator(
        TileFusion,
        padded_shape=padded_shape,
        offset_um=(4.51234, -3.01234, 12.25123),
        _pixel_size=(0.81234, 0.51234, 0.25123),
        time_dim=1,
        channels=1,
        multiscale_factors=(2, 4),
        _is_2d=is_2d,
        chunk_y=4,
        chunk_x=4,
        multiscale_downsample=downsample_method,
        _max_workers=2,
        fusion_ram_fraction=0.4,
        max_in_flight_writes=2,
        _input_chunk_zyx=(2, 4, 4),
        output_dtype=np.dtype(np.uint16),
    )

    path = tmp_path / "multiscale.ome.zarr"
    scale0, fusion.write_block_shape = fusion._prepare_fused_image(path)
    source_shape = (1, 1, *padded_shape)
    source = np.arange(np.prod(source_shape), dtype=np.uint16).reshape(source_shape)
    scale0.write(source).result()
    fusion._write_multiscales()

    reopened = open_group(path)
    metadata = reopened.ome_metadata()
    assert metadata is not None
    datasets = metadata.multiscales[0].datasets
    assert [dataset.path for dataset in datasets] == ["0", "1", "2"]

    expected = source
    rounded_offset = np.asarray([round(float(value), 3) for value in fusion.offset_um])
    rounded_pixel_size = np.asarray(
        [round(float(value), 3) for value in fusion._pixel_size]
    )
    base_center = (
        rounded_offset
        + (np.asarray(padded_shape, dtype=np.float64) - 1.0) * rounded_pixel_size / 2.0
    )
    for absolute_factor, dataset in zip((1, 2, 4), datasets, strict=False):
        z_factor = 1 if is_2d else absolute_factor
        expected_scale = np.asarray(
            [
                round(float(value), 3)
                for value in (
                    1.0,
                    1.0,
                    rounded_pixel_size[0] * z_factor,
                    rounded_pixel_size[1] * absolute_factor,
                    rounded_pixel_size[2] * absolute_factor,
                )
            ]
        )
        center_shift = 0.5 if downsample_method == "block_mean" else 0.0
        expected_translation = np.asarray(
            [
                round(float(value), 3)
                for value in (
                    0.0,
                    0.0,
                    rounded_offset[0]
                    + center_shift * (z_factor - 1) * rounded_pixel_size[0],
                    rounded_offset[1]
                    + center_shift * (absolute_factor - 1) * rounded_pixel_size[1],
                    rounded_offset[2]
                    + center_shift * (absolute_factor - 1) * rounded_pixel_size[2],
                )
            ]
        )
        np.testing.assert_allclose(
            dataset.scale_transform.scale,
            expected_scale,
        )
        assert dataset.translation_transform is not None
        np.testing.assert_allclose(
            dataset.translation_transform.translation,
            expected_translation,
        )
        reopened_array = reopened[dataset.path].to_tensorstore()

        if downsample_method == "stride":
            # A pyramid index must have exactly the same physical coordinate
            # as the source voxel selected at index * absolute_factor.
            output_index = np.asarray(
                [
                    min(1, int(reopened_array.shape[-3]) - 1),
                    min(2, int(reopened_array.shape[-2]) - 1),
                    min(3, int(reopened_array.shape[-1]) - 1),
                ],
                dtype=np.float64,
            )
            source_index = output_index * np.asarray(
                [z_factor, absolute_factor, absolute_factor],
                dtype=np.float64,
            )
            output_coordinate = np.asarray(
                dataset.translation_transform.translation[-3:]
            ) + output_index * np.asarray(dataset.scale_transform.scale[-3:])
            source_coordinate = rounded_offset + source_index * rounded_pixel_size
            np.testing.assert_allclose(
                output_coordinate,
                source_coordinate,
            )

        spatial_shape = np.asarray(reopened_array.shape[-3:], dtype=np.float64)
        physical_center = (
            expected_translation[-3:]
            + (spatial_shape - 1.0) * expected_scale[-3:] / 2.0
        )
        if downsample_method == "block_mean":
            np.testing.assert_allclose(physical_center, base_center, atol=0.002)

        if dataset.path == "0":
            np.testing.assert_array_equal(
                reopened_array.read().result(),
                expected,
            )
            continue

        z_relative_factor = 1 if is_2d else 2
        if downsample_method == "stride":
            expected = expected[..., ::z_relative_factor, ::2, ::2]
        else:
            expected = block_reduce_cpu(
                expected,
                block_size=(1, 1, z_relative_factor, 2, 2),
                func=np.mean,
            ).astype(np.uint16)
        np.testing.assert_array_equal(
            reopened_array.read().result(),
            expected,
        )

    assert not any(key.startswith("opm_") for key in reopened.attrs)


@pytest.mark.unit
def test_fused_multiscale_storage_preserves_float32_source_contract(
    fusion_operator, tmp_path
):
    """Float32 processed tiles create float32 fused output at every level."""
    fusion = fusion_operator(
        TileFusion,
        padded_shape=(2, 4, 4),
        offset_um=(0.0, 0.0, 0.0),
        _pixel_size=(1.0, 1.0, 1.0),
        time_dim=1,
        channels=1,
        multiscale_factors=(2,),
        _is_2d=False,
        chunk_y=2,
        chunk_x=2,
        multiscale_downsample="stride",
        _max_workers=1,
        fusion_ram_fraction=0.4,
        max_in_flight_writes=1,
        _input_chunk_zyx=(1, 2, 2),
        output_dtype=np.dtype(np.float32),
    )

    path = tmp_path / "float32-fused.ome.zarr"
    scale0, fusion.write_block_shape = fusion._prepare_fused_image(path)
    source = np.full((1, 1, 2, 4, 4), 0.25, dtype=np.float32)
    scale0.write(source).result()
    fusion._write_multiscales()

    reopened = open_group(path)
    for level in ("0", "1"):
        result = reopened[level].to_tensorstore().read().result()
        assert result.dtype == np.float32
        factor = 2 ** int(level)
        np.testing.assert_array_equal(
            result,
            source[..., ::factor, ::factor, ::factor],
        )


@pytest.mark.integration
def test_regenerate_max_z_flag_preserves_multiscale_pyramid_round_trip(
    tmp_path,
) -> None:
    """Project every fused scale and retain its exact spatial coordinates."""
    raw_path = tmp_path / "sample.zarr"
    fused_path = tmp_path / "sample_fused.ome.zarr"
    output_path = tmp_path / "sample_max_z_fused.ome.zarr"
    shape = (2, 2, 5, 7, 9)
    scales = (
        [1.0, 1.0, 0.81234, 0.51234, 0.25123],
        [1.0, 1.0, 1.62468, 1.02468, 0.50246],
    )
    translations = (
        [0.0, 0.0, 4.51234, -3.01234, 12.25123],
        [0.0, 0.0, 4.91876, -2.75678, 12.37987],
    )
    image = v05.Image(
        multiscales=[
            v05.Multiscale(
                name="registered-fused",
                axes=[
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                datasets=[
                    v05.Dataset(
                        path=str(level),
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=scales[level]),
                            v05.TranslationTransformation(
                                translation=translations[level]
                            ),
                        ],
                    )
                    for level in range(2)
                ],
            )
        ]
    )

    level_shapes = (shape, (shape[0], shape[1], 3, 4, 5))
    level_specs = [(level_shape, np.uint16) for level_shape in level_shapes]

    _, raw_arrays = prepare_image(
        raw_path,
        image,
        level_specs,
        chunks=(1, 1, 2, 4, 5),
        writer="tensorstore",
        overwrite=True,
    )
    for level, level_shape in enumerate(level_shapes):
        raw_arrays[str(level)].write(np.zeros(level_shape, dtype=np.uint16)).result()

    rng = np.random.default_rng(281)
    registered = rng.integers(0, 60_000, size=shape, dtype=np.uint16)
    registered_levels = (registered, registered[..., ::2, ::2, ::2])
    _, fused_arrays = prepare_image(
        fused_path,
        image,
        level_specs,
        extra_attributes={},
        chunks=(1, 1, 2, 4, 5),
        writer="tensorstore",
        overwrite=True,
    )
    for level, level_data in enumerate(registered_levels):
        fused_arrays[str(level)].write(level_data).result()

    sentinel_specs = [
        ((level_shape[0], level_shape[1], 1, level_shape[3], level_shape[4]), np.uint16)
        for level_shape in level_shapes
    ]
    _, sentinel_arrays = prepare_image(
        output_path,
        image,
        sentinel_specs,
        chunks=(1, 1, 1, 4, 5),
        writer="tensorstore",
        overwrite=True,
    )
    for level, (sentinel_shape, _dtype) in enumerate(sentinel_specs):
        sentinel_arrays[str(level)].write(
            np.zeros(sentinel_shape, dtype=np.uint16)
        ).result()
    del raw_arrays, fused_arrays, sentinel_arrays

    processed_path = tmp_path / "sample_projection.ome.zarr"
    state = ProcessingState.create(tmp_path / "sample.processing.json", raw_path)
    state.initialize_run(processed_path, configuration={}, overwrite=True)
    state.save_registration(
        processed_path,
        configuration={},
        pairwise_metrics={},
    )
    state.complete_registration(
        processed_path,
        fused_path=fused_path,
        tiles=(),
    )

    result = CliRunner().invoke(
        fuse_app,
        [str(raw_path), "--regenerate-max-z", "--max-workers", "2"],
    )
    assert result.exit_code == 0, result.output

    reopened = open_group(output_path)
    metadata = reopened.ome_metadata()
    assert isinstance(metadata, v05.Image)
    datasets = metadata.multiscales[0].datasets
    assert [dataset.path for dataset in datasets] == ["0", "1"]
    for level, (dataset, level_data) in enumerate(
        zip(datasets, registered_levels, strict=False)
    ):
        actual = reopened[dataset.path].to_tensorstore().read().result()
        np.testing.assert_array_equal(
            actual,
            level_data.max(axis=2, keepdims=True),
        )
        expected_scale = [
            scales[level][0],
            scales[level][1],
            *(round(value, 3) for value in scales[level][2:]),
        ]
        np.testing.assert_allclose(dataset.scale_transform.scale, expected_scale)
        assert dataset.translation_transform is not None
        expected_translation = [
            translations[level][0],
            translations[level][1],
            *(round(value, 3) for value in translations[level][2:]),
        ]
        expected_translation[2] += 0.5 * (level_data.shape[2] - 1) * expected_scale[2]
        expected_translation[2:] = [
            round(value, 3) for value in expected_translation[2:]
        ]
        np.testing.assert_allclose(
            dataset.translation_transform.translation,
            expected_translation,
        )
    assert not any(key.startswith("opm_") for key in reopened.attrs)
    state = ProcessingState.read(tmp_path / "sample.processing.json")
    assert state.registered_output_for_max_projection(output_path) == processed_path
