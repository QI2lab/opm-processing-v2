import json
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from skimage.measure import block_reduce as block_reduce_cpu
from yaozarrs import open_group

from opm_processing.imageprocessing import tilefusion as tilefusion_module
from opm_processing.imageprocessing.maxtilefusion import MaxTileFusion
from opm_processing.imageprocessing.tilefusion import TileFusion


class _TrackedFuture:
    def __init__(self, tracker):
        self.tracker = tracker
        self.pending = True
        tracker["active"] += 1
        tracker["maximum"] = max(tracker["maximum"], tracker["active"])

    def result(self):
        if self.pending:
            self.pending = False
            self.tracker["active"] -= 1


class _ArrayView:
    def __init__(self, store, key):
        self.store = store
        self.key = key

    def write(self, value):
        self.store.data[self.key] = value
        self.store.tracker["writes"] += 1
        return _TrackedFuture(self.store.tracker)


class _ArrayStore:
    def __init__(self, shape):
        self.data = np.zeros(shape, dtype=np.uint16)
        self.tracker = {"active": 0, "maximum": 0, "writes": 0}

    def __getitem__(self, key):
        return _ArrayView(self, key)


def test_prepare_fused_image_uses_ngff_multiscales(tmp_path):
    fusion = TileFusion.__new__(TileFusion)
    fusion.padded_shape = (8, 16, 20)
    fusion.offset_um = (3.0, 4.0, 5.0)
    fusion._pixel_size = (1.5, 0.3, 0.3)
    fusion.time_dim = 2
    fusion.channels = 3
    fusion.multiscale_factors = (2, 4)
    fusion._is_2d = False
    fusion.chunk_y = 8
    fusion.chunk_x = 8

    path = tmp_path / "fused.ome.zarr"
    scale0, write_block_shape = fusion._prepare_fused_image(path)

    assert tuple(scale0.shape) == (2, 3, 8, 16, 20)
    assert tuple(fusion._multiscale_arrays["1"].shape) == (2, 3, 4, 8, 10)
    assert tuple(fusion._multiscale_arrays["2"].shape) == (2, 3, 2, 4, 5)
    assert write_block_shape == [1, 1, 4, 16, 16]

    scale0_metadata = json.loads((path / "0" / "zarr.json").read_text())
    assert scale0_metadata["chunk_grid"]["configuration"]["chunk_shape"] == [
        1,
        1,
        1,
        8,
        8,
    ]
    assert all(codec["name"] != "sharding_indexed" for codec in scale0_metadata["codecs"])

    multiscale = open_group(path).ome_metadata().multiscales[0]
    assert [dataset.path for dataset in multiscale.datasets] == ["0", "1", "2"]


def test_max_projection_fusion_uses_chunks_without_sharding(tmp_path):
    fusion = MaxTileFusion.__new__(MaxTileFusion)
    fusion.time_range = None
    fusion.time_dim = 1
    fusion.channels = 2
    fusion.fused_shape = (16, 20)
    fusion.pixel_size = (0.3, 0.3)
    fusion.chunk_size = 8
    fusion.output_path = tmp_path / "max-fused.ome.zarr"

    fusion.create_fused_image()

    metadata = json.loads((fusion.output_path / "0" / "zarr.json").read_text())
    assert metadata["chunk_grid"]["configuration"]["chunk_shape"] == [
        1,
        1,
        1,
        8,
        8,
    ]
    assert all(codec["name"] != "sharding_indexed" for codec in metadata["codecs"])


def test_max_projection_fusion_operational_settings_are_configurable():
    fusion = MaxTileFusion.__new__(MaxTileFusion)
    fusion.tile_shape = (20, 30)
    fusion.blend_pixels = (3, 5)

    weights = fusion.generate_blending_weights()

    assert weights.shape == fusion.tile_shape
    assert np.all(weights > 0)
    np.testing.assert_allclose(weights[10, 15], 1.0)


def test_fusion_block_shape_respects_memory_budget(monkeypatch):
    fusion = TileFusion.__new__(TileFusion)
    fusion.fusion_ram_fraction = 0.5
    fusion.chunk_y = 4
    fusion.chunk_x = 4
    monkeypatch.setattr(
        tilefusion_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=1024),
    )

    block_y, block_x = fusion._fusion_block_shape(2, 100, 100)

    assert block_y * block_x * 2 * 16 <= 512


def test_fusion_uses_spatial_blocks_and_bounded_writes(monkeypatch):
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
    tiles = [
        np.ones((1, 2, 3, 4), dtype=np.float32),
        np.full((1, 2, 3, 4), 3.0, dtype=np.float32),
    ]

    def read_tile(self, tile_idx, ch_sel, z_slice, y_slice, x_slice):
        return tiles[tile_idx][ch_sel, z_slice, y_slice, x_slice]

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
    assert fusion.fused_ts.tracker["writes"] > 1
    assert fusion.fused_ts.tracker["maximum"] <= 2
    assert fusion.fused_ts.tracker["active"] == 0


@pytest.mark.parametrize("downsample_method", ["stride", "block_mean"])
def test_multiscales_are_generated_in_spatial_blocks(tmp_path, downsample_method):
    fusion = TileFusion.__new__(TileFusion)
    fusion.padded_shape = (4, 8, 8)
    fusion.offset_um = (0.0, 0.0, 0.0)
    fusion._pixel_size = (1.0, 1.0, 1.0)
    fusion.time_dim = 1
    fusion.channels = 1
    fusion.multiscale_factors = (2, 4)
    fusion._is_2d = False
    fusion.chunk_y = 4
    fusion.chunk_x = 4
    fusion.multiscale_downsample = downsample_method

    path = tmp_path / "multiscale.ome.zarr"
    scale0, fusion.write_block_shape = fusion._prepare_fused_image(
        path,
        z_slices_per_write=2,
    )
    source = np.arange(1 * 1 * 4 * 8 * 8, dtype=np.uint16).reshape(1, 1, 4, 8, 8)
    scale0.write(source).result()

    fusion._write_multiscales()

    for level, factor in (("1", 2), ("2", 4)):
        if downsample_method == "stride":
            expected = source[..., ::factor, ::factor, ::factor]
        else:
            expected = block_reduce_cpu(
                source,
                block_size=(1, 1, factor, factor, factor),
                func=np.mean,
            ).astype(np.uint16)
        np.testing.assert_array_equal(
            fusion._multiscale_arrays[level].read().result(),
            expected,
        )


def test_iterative_optimization_handles_all_links_rejected():
    fusion = TileFusion.__new__(TileFusion)
    links = [
        {
            "i": 0,
            "j": 1,
            "t": np.array([10.0, 0.0, 0.0]),
            "w": 1.0,
        }
    ]

    shifts = fusion._two_round_opt(
        links,
        n_tiles=2,
        fixed_indices=[0, 1],
        rel_thresh=0.5,
        abs_thresh=1.0,
        iterative=True,
    )

    np.testing.assert_array_equal(shifts, np.zeros((2, 3)))


def test_register_and_score_identical_3d_patches():
    rng = np.random.default_rng(42)
    patch = rng.normal(size=(9, 17, 17)).astype(np.float32)

    shift, score = TileFusion.register_and_score(patch, patch, win_size=7)

    np.testing.assert_allclose(shift, (0.0, 0.0, 0.0), atol=0.1)
    assert score > 0.99
