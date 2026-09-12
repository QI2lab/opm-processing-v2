"""Numerical unit and storage integration tests for tile fusion."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import tensorstore as ts
from skimage.measure import block_reduce as block_reduce_cpu
from typer.testing import CliRunner
from yaozarrs import open_group, v05
from yaozarrs.write.v05 import prepare_image

from opm_processing.fuse import app as fuse_app
from opm_processing.dataio.processing_state import ProcessingState
from opm_processing.imageprocessing import maxtilefusion as maxtilefusion_module
from opm_processing.imageprocessing import tilefusion as tilefusion_module
from opm_processing.imageprocessing.coordinates import (
    stage_positions_to_image_coordinates,
)
from opm_processing.imageprocessing.maxtilefusion import MaxTileFusion
from opm_processing.imageprocessing.tilefusion import TileFusion


class _ImmediateFuture:
    """Completed write compatible with the fusion writer."""

    def result(self):
        """Return after an already-completed in-memory write."""


class _ArrayView:
    """Writable selection into an in-memory array."""

    def __init__(self, store, key):
        self.store = store
        self.key = key

    def write(self, value):
        """Write the selected values exactly."""
        self.store.data[self.key] = value
        return _ImmediateFuture()


class _ArrayStore:
    """Small in-memory TensorStore substitute for numerical fusion."""

    def __init__(self, shape, dtype=np.uint16):
        self.data = np.zeros(shape, dtype=dtype)

    def __getitem__(self, key):
        return _ArrayView(self, key)


class _ReadResult:
    """Completed read compatible with an in-memory TensorStore substitute."""

    def __init__(self, value):
        self.value = value

    def result(self):
        """Return the selected array."""
        return self.value


class _ReadView:
    """Readable selection into an in-memory array."""

    def __init__(self, value):
        self.value = value

    def read(self):
        """Return an immediately completed read."""
        return _ReadResult(self.value)


class _ReadArray:
    """Small read-only TensorStore substitute for projection fusion."""

    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape = self.data.shape
        self.dtype = SimpleNamespace(numpy_dtype=self.data.dtype)

    def __getitem__(self, key):
        return _ReadView(self.data[key])


@pytest.mark.integration
def test_max_projection_pyramid_clamps_partial_source_edge_chunks() -> None:
    """Never request factor-rounded source bounds beyond level-zero shape."""
    source = np.arange(1 * 3 * 1 * 17 * 19, dtype=np.uint16).reshape(1, 3, 1, 17, 19)
    target = np.zeros((1, 3, 1, 2, 2), dtype=np.uint16)
    fusion = MaxTileFusion.__new__(MaxTileFusion)
    fusion.chunk_size = 512
    fusion.multiscale_factors_yx = (1, 16)
    fusion.fused_ts = ts.array(source)
    fusion.multiscale_arrays = (fusion.fused_ts, ts.array(target))

    fusion.write_multiscales()

    np.testing.assert_array_equal(
        fusion.multiscale_arrays[1].read().result(),
        source[..., ::16, ::16],
    )


@pytest.mark.unit
def test_stage_z_is_reversed_only_for_lab_placement() -> None:
    """Negate physical stage Z and its Y shear without touching input pixels."""
    stage_positions = np.asarray(
        ((10.0, 20.0, 30.0), (14.0, 20.0, 30.0)),
        dtype=np.float64,
    )
    original = stage_positions.copy()

    coordinates = stage_positions_to_image_coordinates(
        stage_positions,
        reverse_y=True,
        reverse_z=True,
        opm_angle_deg=45.0,
    )

    np.testing.assert_allclose(coordinates[1] - coordinates[0], (-4.0, -4.0, 0.0))
    np.testing.assert_array_equal(stage_positions, original)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dtype", "values"),
    (
        (np.uint16, (10, 30, 50)),
        (np.float32, (0.25, 0.75, 1.25)),
    ),
)
def test_max_projection_fusion_is_chunked_and_sparse(monkeypatch, dtype, values):
    """Fuse distant tiles without allocating arrays the size of the full canvas."""
    tile_shape = (4, 4)
    tile_positions = np.asarray(((0, 0), (0, 2), (0, 10_000)), dtype=float)
    fused_shape = (4, 10_004)
    tiles = tuple(
        _ReadArray(np.full((1, 1, 1, *tile_shape), value, dtype=dtype))
        for value in values
    )
    fusion = MaxTileFusion.__new__(MaxTileFusion)
    fusion.pad_y = 0
    fusion.pad_x = 0
    fusion.ts_dataset = tiles
    fusion.tile_positions = tile_positions
    fusion.pixel_size = np.asarray((1.0, 1.0))
    fusion.offset = (0.0, 0.0)
    fusion.time_dim = 1
    fusion.channels = 1
    fusion.tile_shape = tile_shape
    fusion.fused_shape = fused_shape
    fusion.time_range = None
    fusion.chunk_size = 3
    fusion.weight_mask = np.ones(tile_shape, dtype=np.float32)
    fusion.output_dtype = np.dtype(dtype)
    fusion.fused_ts = _ArrayStore((1, 1, 1, *fused_shape), dtype=dtype)

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
    np.testing.assert_allclose(fusion.fused_ts.data, expected)


@pytest.mark.unit
def test_zero_max_projection_tile_does_not_reduce_weighted_fusion() -> None:
    """Exclude a globally zero projection channel from its neighbor's weights."""
    tile_shape = (3, 4)
    fusion = MaxTileFusion.__new__(MaxTileFusion)
    fusion.pad_y = 0
    fusion.pad_x = 0
    fusion.ts_dataset = (
        _ReadArray(np.full((1, 1, 1, *tile_shape), 100, dtype=np.uint16)),
        _ReadArray(np.zeros((1, 1, 1, *tile_shape), dtype=np.uint16)),
    )
    fusion.tile_positions = np.asarray(((0, 0), (0, 0)), dtype=float)
    fusion.pixel_size = np.asarray((1.0, 1.0))
    fusion.offset = (0.0, 0.0)
    fusion.time_dim = 1
    fusion.channels = 1
    fusion.tile_shape = tile_shape
    fusion.fused_shape = tile_shape
    fusion.time_range = None
    fusion.chunk_size = 4
    fusion.weight_mask = np.ones(tile_shape, dtype=np.float32)
    fusion.output_dtype = np.dtype(np.uint16)
    fusion.fused_ts = _ArrayStore((1, 1, 1, *tile_shape), dtype=np.uint16)

    fusion.fuse_tiles()

    np.testing.assert_array_equal(
        fusion.fused_ts.data,
        np.full((1, 1, 1, *tile_shape), 100, dtype=np.uint16),
    )


@pytest.mark.integration
def test_max_projection_fusion_writes_centered_multiscales_and_offsets(
    tmp_path,
) -> None:
    """Persist every fused max level with rounded physical coordinates."""
    source = np.arange(64, dtype=np.uint16).reshape(1, 1, 1, 8, 8)
    output_path = tmp_path / "max_z_fused.ome.zarr"
    fusion = MaxTileFusion(
        ts_dataset=(_ReadArray(source),),
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
    for factor, dataset in zip((1, 2, 4, 8), datasets):
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
def test_shift_optimization_supports_variable_tiles_per_timepoint() -> None:
    """Optimize global tile IDs without assuming rectangular T-by-P storage."""
    fusion = TileFusion.__new__(TileFusion)
    fusion._tile_positions = [(0.0, 0.0, 0.0)] * 5
    fusion._tiles_by_time = ((0, 1), (2, 3, 4))
    fusion.pairwise_metrics = {
        (0, 1): (0, 0, 2, 1.0),
        (2, 4): (0, 3, 0, 1.0),
    }

    fusion.optimize_shifts(method="ONE_ROUND")

    np.testing.assert_allclose(fusion.global_offsets[0], (0, 0, 0), atol=1e-12)
    np.testing.assert_allclose(fusion.global_offsets[1], (0, 0, 2), atol=1e-12)
    np.testing.assert_allclose(fusion.global_offsets[2], (0, 0, 0), atol=1e-12)
    np.testing.assert_allclose(fusion.global_offsets[3], (0, 0, 0), atol=1e-12)
    np.testing.assert_allclose(fusion.global_offsets[4], (0, 3, 0), atol=1e-12)


@pytest.mark.unit
def test_block_fusion_does_not_infer_support_from_pixel_values(monkeypatch):
    """Include legitimate zero-valued pixels when geometry marks them valid."""
    fusion = TileFusion.__new__(TileFusion)
    fusion.fused_ts = _ArrayStore((1, 1, 2, 3, 6))
    fusion.write_block_shape = [1, 1, 1, 2, 2]
    fusion.offset_um = (0.0, 0.0, 0.0)
    fusion.padded_shape = (2, 3, 6)
    fusion._pixel_size = (1.0, 1.0, 1.0)
    fusion._tile_positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]
    fusion.time_dim = 1
    fusion.position_dim = 2
    fusion.channels = 1
    fusion.z_dim = 2
    fusion.y_dim = 3
    fusion.x_dim = 4
    fusion.z_profile = np.ones(2, dtype=np.float32)
    fusion.y_profile = np.ones(3, dtype=np.float32)
    fusion.x_profile = np.ones(4, dtype=np.float32)
    fusion.chunk_y = 2
    fusion.chunk_x = 2
    fusion.fusion_ram_fraction = 1.0
    fusion.max_in_flight_writes = 2
    fusion._max_workers = 2
    fusion._debug = False
    fusion._deskew_support_zy = np.ones((2, 3), dtype=bool)
    fusion._tile_channel_nonzero = {(0, 0): True, (1, 0): True}
    tiles = [
        np.full((1, 2, 3, 4), 10.0, dtype=np.float32),
        np.full((1, 2, 3, 4), 30.0, dtype=np.float32),
    ]
    # These dark pixels are inside valid geometry and must retain their weight.
    tiles[1][:, 0, :2, :2] = 0
    tiles[1][:, 1, :1, :2] = 0

    def read_tile(self, tile_idx, ch_sel, z_slice, y_slice, x_slice, dtype=np.float32):
        selected = tiles[tile_idx][ch_sel, z_slice, y_slice, x_slice]
        return selected if dtype is None else selected.astype(dtype, copy=False)

    fusion._read_tile_volume = MethodType(read_tile, fusion)
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
    np.testing.assert_array_equal(fusion.fused_ts.data, expected)


@pytest.mark.unit
def test_block_fusion_has_no_zero_weight_line_at_masked_wedge_boundary(
    monkeypatch,
):
    """Keep constant signal continuous where a deskew wedge meets a tile edge."""
    fusion = TileFusion.__new__(TileFusion)
    fusion.fused_ts = _ArrayStore((1, 1, 2, 6, 3))
    fusion.write_block_shape = [1, 1, 2, 6, 3]
    fusion.offset_um = (0.0, 0.0, 0.0)
    fusion.padded_shape = (2, 6, 3)
    fusion._pixel_size = (1.0, 1.0, 1.0)
    fusion._tile_positions = [(0.0, 0.0, 0.0), (0.0, 2.0, 0.0)]
    fusion.time_dim = 1
    fusion.position_dim = 2
    fusion.channels = 1
    fusion.z_dim = 2
    fusion.y_dim = 4
    fusion.x_dim = 3
    fusion.z_profile = np.ones(2, dtype=np.float32)
    fusion.y_profile = fusion._make_1d_profile(4, 2)
    fusion.x_profile = np.ones(3, dtype=np.float32)
    fusion.chunk_y = 6
    fusion.chunk_x = 3
    fusion.fusion_ram_fraction = 1.0
    fusion.max_in_flight_writes = 1
    fusion._max_workers = 1
    fusion._debug = False
    fusion._deskew_support_zy = np.asarray(
        [[False, False, True, True], [False, True, True, True]],
        dtype=bool,
    )
    fusion._tile_channel_nonzero = {(0, 0): True, (1, 0): True}
    tiles = [
        np.full((1, 2, 4, 3), 100.0, dtype=np.float32),
        np.full((1, 2, 4, 3), 100.0, dtype=np.float32),
    ]
    for tile in tiles:
        tile[:, 0, :2, :] = 0
        tile[:, 1, :1, :] = 0

    def read_tile(self, tile_idx, ch_sel, z_slice, y_slice, x_slice, dtype=np.float32):
        selected = tiles[tile_idx][ch_sel, z_slice, y_slice, x_slice]
        return selected if dtype is None else selected.astype(dtype, copy=False)

    fusion._read_tile_volume = MethodType(read_tile, fusion)
    monkeypatch.setattr(
        tilefusion_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=4096),
    )

    fusion._fuse_by_blocks()

    expected = np.full((1, 1, 2, 6, 3), 100, dtype=np.uint16)
    expected[:, :, 0, :2] = 0
    expected[:, :, 1, :1] = 0
    np.testing.assert_array_equal(fusion.fused_ts.data, expected)


@pytest.mark.unit
def test_zero_tile_channel_contributes_neither_signal_nor_fusion_weight(
    monkeypatch,
):
    """An explicitly zeroed tile must not attenuate an overlapping valid tile."""
    fusion = TileFusion.__new__(TileFusion)
    fusion.fused_ts = _ArrayStore((1, 1, 1, 2, 4))
    fusion.write_block_shape = [1, 1, 1, 2, 4]
    fusion.offset_um = (0.0, 0.0, 0.0)
    fusion.padded_shape = (1, 2, 4)
    fusion._pixel_size = (1.0, 1.0, 1.0)
    fusion._tile_positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    fusion.time_dim = 1
    fusion.position_dim = 2
    fusion.channels = 1
    fusion.z_dim = 1
    fusion.y_dim = 2
    fusion.x_dim = 4
    fusion.z_profile = np.ones(1, dtype=np.float32)
    fusion.y_profile = np.ones(2, dtype=np.float32)
    fusion.x_profile = np.ones(4, dtype=np.float32)
    fusion.chunk_y = 2
    fusion.chunk_x = 4
    fusion.fusion_ram_fraction = 1.0
    fusion.max_in_flight_writes = 1
    fusion._max_workers = 1
    fusion._debug = False
    fusion._deskew_support_zy = np.ones((1, 2), dtype=bool)
    fusion._tile_channel_nonzero = {(0, 0): True, (1, 0): False}
    tiles = [
        np.full((1, 1, 2, 4), 100.0, dtype=np.float32),
        np.zeros((1, 1, 2, 4), dtype=np.float32),
    ]

    def read_tile(self, tile_idx, ch_sel, z_slice, y_slice, x_slice, dtype=np.float32):
        selected = tiles[tile_idx][ch_sel, z_slice, y_slice, x_slice]
        return selected if dtype is None else selected.astype(dtype, copy=False)

    fusion._read_tile_volume = MethodType(read_tile, fusion)
    monkeypatch.setattr(
        tilefusion_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=4096),
    )

    fusion._fuse_by_blocks()

    np.testing.assert_array_equal(
        fusion.fused_ts.data,
        np.full((1, 1, 1, 2, 4), 100, dtype=np.uint16),
    )


@pytest.mark.unit
def test_single_nonzero_channel_contributor_is_copied_in_overlap(monkeypatch):
    """Copy one valid channel unchanged while blending another channel."""
    fusion = TileFusion.__new__(TileFusion)
    fusion.fused_ts = _ArrayStore((1, 2, 1, 2, 4))
    fusion.write_block_shape = [1, 1, 1, 2, 4]
    fusion.offset_um = (0.0, 0.0, 0.0)
    fusion.padded_shape = (1, 2, 4)
    fusion._pixel_size = (1.0, 1.0, 1.0)
    fusion._tile_positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    fusion.time_dim = 1
    fusion.position_dim = 2
    fusion.channels = 2
    fusion.z_dim = 1
    fusion.y_dim = 2
    fusion.x_dim = 4
    fusion.z_profile = np.ones(1, dtype=np.float32)
    fusion.y_profile = np.ones(2, dtype=np.float32)
    fusion.x_profile = np.ones(4, dtype=np.float32)
    fusion.chunk_y = 2
    fusion.chunk_x = 4
    fusion.fusion_ram_fraction = 1.0
    fusion.max_in_flight_writes = 1
    fusion._max_workers = 1
    fusion._debug = False
    fusion._deskew_support_zy = np.ones((1, 2), dtype=bool)
    fusion._tile_channel_nonzero = {
        (0, 0): True,
        (0, 1): True,
        (1, 0): False,
        (1, 1): True,
    }
    tiles = [
        np.full((2, 1, 2, 4), 100.0, dtype=np.float32),
        np.stack(
            (
                np.zeros((1, 2, 4), dtype=np.float32),
                np.full((1, 2, 4), 200.0, dtype=np.float32),
            )
        ),
    ]

    def read_tile(self, tile_idx, ch_sel, z_slice, y_slice, x_slice, dtype=np.float32):
        selected = tiles[tile_idx][ch_sel, z_slice, y_slice, x_slice]
        return selected if dtype is None else selected.astype(dtype, copy=False)

    fusion._read_tile_volume = MethodType(read_tile, fusion)
    monkeypatch.setattr(
        tilefusion_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=4096),
    )

    fusion._fuse_by_blocks()

    np.testing.assert_array_equal(
        fusion.fused_ts.data[:, 0],
        np.full_like(fusion.fused_ts.data[:, 0], 100),
    )
    np.testing.assert_array_equal(
        fusion.fused_ts.data[:, 1],
        np.full_like(fusion.fused_ts.data[:, 1], 150),
    )


@pytest.mark.unit
def test_zero_registration_channel_is_excluded_before_patch_reads() -> None:
    """Do not schedule registration reads for a pair containing a zero tile."""
    fusion = TileFusion.__new__(TileFusion)
    fusion.downsample_factors = (1, 1, 1)
    fusion.ssim_window = 3
    fusion.threshold = 0.0
    fusion.pairwise_metrics = {}
    fusion.position_dim = 2
    fusion.time_dim = 1
    fusion.z_dim = 4
    fusion.y_dim = 4
    fusion.x_dim = 4
    fusion._pixel_size = (1.0, 1.0, 1.0)
    fusion._tile_positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]
    fusion._is_2d = False
    fusion._max_workers = 1
    fusion.max_registration_shift_zyx = (2, 2, 2)
    fusion._debug = False
    fusion._tile_channel_nonzero = {(0, 0): True, (1, 0): False}
    fusion.position_arrays = (
        SimpleNamespace(dtype=SimpleNamespace(numpy_dtype=np.dtype(np.uint16))),
        SimpleNamespace(dtype=SimpleNamespace(numpy_dtype=np.dtype(np.uint16))),
    )

    def fail_read(*_args, **_kwargs):
        raise AssertionError("zero registration tiles must be filtered before reads")

    fusion._read_registration_patch = fail_read
    fusion.refine_tile_positions_with_cross_correlation(ch_idx=0)

    assert fusion.pairwise_metrics == {}


@pytest.mark.integration
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
    tmp_path,
    downsample_method,
    is_2d,
    padded_shape,
):
    """Reopen exact pyramid data and verify its common physical-space center."""
    fusion = TileFusion.__new__(TileFusion)
    fusion.padded_shape = padded_shape
    fusion.offset_um = (4.51234, -3.01234, 12.25123)
    fusion._pixel_size = (0.81234, 0.51234, 0.25123)
    fusion.time_dim = 1
    fusion.channels = 1
    fusion.multiscale_factors = (2, 4)
    fusion._is_2d = is_2d
    fusion.chunk_y = 4
    fusion.chunk_x = 4
    fusion.multiscale_downsample = downsample_method
    fusion._max_workers = 2
    fusion.fusion_ram_fraction = 0.4
    fusion.max_in_flight_writes = 2
    fusion._input_chunk_zyx = (2, 4, 4)
    fusion.output_dtype = np.dtype(np.uint16)

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
    for absolute_factor, dataset in zip((1, 2, 4), datasets):
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


@pytest.mark.integration
def test_fused_multiscale_storage_preserves_float32_source_contract(tmp_path):
    """Float32 processed tiles create float32 fused output at every level."""
    fusion = TileFusion.__new__(TileFusion)
    fusion.padded_shape = (2, 4, 4)
    fusion.offset_um = (0.0, 0.0, 0.0)
    fusion._pixel_size = (1.0, 1.0, 1.0)
    fusion.time_dim = 1
    fusion.channels = 1
    fusion.multiscale_factors = (2,)
    fusion._is_2d = False
    fusion.chunk_y = 2
    fusion.chunk_x = 2
    fusion.multiscale_downsample = "stride"
    fusion._max_workers = 1
    fusion.fusion_ram_fraction = 0.4
    fusion.max_in_flight_writes = 1
    fusion._input_chunk_zyx = (1, 2, 2)
    fusion.output_dtype = np.dtype(np.float32)

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
    for level, (dataset, level_data) in enumerate(zip(datasets, registered_levels)):
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
