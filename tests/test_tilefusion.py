"""Numerical unit and storage integration tests for tile fusion."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from skimage.measure import block_reduce as block_reduce_cpu
from yaozarrs import open_group

from opm_processing.imageprocessing import tilefusion as tilefusion_module
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

    def __init__(self, shape):
        self.data = np.zeros(shape, dtype=np.uint16)

    def __getitem__(self, key):
        return _ArrayView(self, key)


def test_registration_pairs_are_exact_face_neighbors() -> None:
    """Select every overlapping face neighbor and no diagonal pair."""
    positions = [
        (z, y, x)
        for z in (0.0, 8.0)
        for y in (0.0, 8.0)
        for x in (0.0, 8.0)
    ]

    pairs = tilefusion_module._select_registration_pairs(
        positions,
        tile_shape_zyx=(10, 10, 10),
        pixel_size_zyx=(1.0, 1.0, 1.0),
    )

    expected = {
        (i, j)
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
        if np.count_nonzero(
            np.asarray(positions[j]) - np.asarray(positions[i])
        )
        == 1
        and np.max(np.abs(np.asarray(positions[j]) - np.asarray(positions[i])))
        == 8.0
    }
    assert set(pairs) == expected


def test_registration_overlap_crop_selects_same_physical_region() -> None:
    """Return bounded source regions describing the same physical overlap."""
    first, second = tilefusion_module._crop_registration_overlap(
        bounds_i=[(0, 128), (0, 5292), (1599, 1900)],
        bounds_j=[(0, 128), (0, 5292), (0, 301)],
        downsample_factors_zyx=(3, 5, 5),
        max_downsampled_shape_zyx=(64, 512, 512),
    )

    assert [hi - lo for lo, hi in first] == [128, 2560, 301]
    assert [hi - lo for lo, hi in second] == [128, 2560, 301]
    assert first[0] == second[0] == (0, 128)
    assert first[1] == second[1] == (1366, 3926)
    assert first[2] == (1599, 1900)
    assert second[2] == (0, 301)


def test_registration_downsampling_matches_independent_block_means() -> None:
    """Match an independent block-reduction reference at every output voxel."""
    data = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8)
    actual = tilefusion_module._mean_downsample_registration(data, (2, 2, 2))
    expected = block_reduce_cpu(
        data.astype(np.float32),
        block_size=(2, 2, 2),
        func=np.mean,
    )

    np.testing.assert_allclose(actual, expected)


def test_registration_score_views_match_shifted_overlap() -> None:
    """Select exactly the mutually supported voxels after a known shift."""
    fixed = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    moving = fixed + 1000

    fixed_view, moving_view = tilefusion_module._aligned_registration_views(
        fixed,
        moving,
        (1.0, -2.0, 0.0),
    )

    np.testing.assert_array_equal(fixed_view, fixed[1:, :3, :])
    np.testing.assert_array_equal(moving_view, moving[:3, 2:, :])


def test_bounded_disambiguation_recovers_known_wrapped_shift() -> None:
    """Recover a known shift lying beyond half of a small Fourier patch."""
    rng = np.random.default_rng(42)
    fixed = rng.normal(size=(7, 11, 13)).astype(np.float32)
    moving = np.zeros_like(fixed)
    moving[4:] = fixed[:-4]

    resolved = tilefusion_module._bounded_disambiguate_shift(
        fixed,
        moving,
        periodic_shift=(3.0, 0.0, 0.0),
        max_shift=(6.67, 1.0, 1.0),
    )

    np.testing.assert_allclose(resolved, (-4.0, 0.0, 0.0))


def test_block_fusion_recovers_exact_constant_tile_mosaic(monkeypatch):
    """Fuse overlapping tiles and validate every output voxel."""
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
    tiles = [
        np.ones((1, 2, 3, 4), dtype=np.float32),
        np.full((1, 2, 3, 4), 3.0, dtype=np.float32),
    ]

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

    expected = np.ones((1, 1, 2, 3, 6), dtype=np.uint16)
    expected[..., 2:4] = 2
    expected[..., 4:6] = 3
    np.testing.assert_array_equal(fusion.fused_ts.data, expected)


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
    fusion.offset_um = (4.5, -3.0, 12.25)
    fusion._pixel_size = (0.8, 0.5, 0.25)
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
    base_center = np.asarray(fusion.offset_um) + (
        np.asarray(padded_shape, dtype=np.float64) - 1.0
    ) * np.asarray(fusion._pixel_size) / 2.0
    for absolute_factor, dataset in zip((1, 2, 4), datasets):
        z_factor = 1 if is_2d else absolute_factor
        expected_scale = np.array(
            [
                1.0,
                1.0,
                fusion._pixel_size[0] * z_factor,
                fusion._pixel_size[1] * absolute_factor,
                fusion._pixel_size[2] * absolute_factor,
            ]
        )
        expected_translation = np.array(
            [
                0.0,
                0.0,
                fusion.offset_um[0] + 0.5 * (z_factor - 1) * fusion._pixel_size[0],
                fusion.offset_um[1]
                + 0.5 * (absolute_factor - 1) * fusion._pixel_size[1],
                fusion.offset_um[2]
                + 0.5 * (absolute_factor - 1) * fusion._pixel_size[2],
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
        spatial_shape = np.asarray(reopened_array.shape[-3:], dtype=np.float64)
        physical_center = expected_translation[-3:] + (
            spatial_shape - 1.0
        ) * expected_scale[-3:] / 2.0
        np.testing.assert_allclose(physical_center, base_center)

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
