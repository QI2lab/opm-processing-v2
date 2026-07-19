"""Test tiled registration, fusion, and multiscale writing."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from skimage.measure import block_reduce as block_reduce_cpu

from opm_processing.imageprocessing import tilefusion as tilefusion_module
from opm_processing.imageprocessing.tilefusion import TileFusion


class _TrackedFuture:
    """Track pending asynchronous writes for backpressure tests."""

    def __init__(self, tracker):
        """Register a newly pending write.

        Parameters
        ----------
        tracker : object
            Value supplied for ``tracker``.

        Returns
        -------
        None
            No value is returned.
        """
        self.tracker = tracker
        self.pending = True
        tracker["active"] += 1
        tracker["maximum"] = max(tracker["maximum"], tracker["active"])

    def result(self):
        """Resolve the write and update the pending-write count.

        Parameters
        ----------
        None
            This callable has no parameters.

        Returns
        -------
        None
            No value is returned.
        """
        if self.pending:
            self.pending = False
            self.tracker["active"] -= 1


class _ArrayView:
    """Writable view into an in-memory array store."""

    def __init__(self, store, key):
        """Initialize a keyed array view.

        Parameters
        ----------
        store : object
            Value supplied for ``store``.
        key : object
            Value supplied for ``key``.

        Returns
        -------
        None
            No value is returned.
        """
        self.store = store
        self.key = key

    def write(self, value):
        """Write a value and return a tracked future.

        Parameters
        ----------
        value : object
            Value supplied for ``value``.

        Returns
        -------
        object
            Result produced by the callable.
        """
        self.store.data[self.key] = value
        self.store.tracker["writes"] += 1
        return _TrackedFuture(self.store.tracker)


class _ArrayStore:
    """In-memory TensorStore test double."""

    def __init__(self, shape):
        """Allocate the array and write counters.

        Parameters
        ----------
        shape : object
            Value supplied for ``shape``.

        Returns
        -------
        None
            No value is returned.
        """
        self.data = np.zeros(shape, dtype=np.uint16)
        self.tracker = {"active": 0, "maximum": 0, "writes": 0}

    def __getitem__(self, key):
        """Return a writable view for an array selection.

        Parameters
        ----------
        key : object
            Value supplied for ``key``.

        Returns
        -------
        object
            Result produced by the callable.
        """
        return _ArrayView(self, key)


def test_fusion_uses_spatial_blocks_and_bounded_writes(monkeypatch):
    """Verify fusion uses spatial blocks and bounded pending writes.

    Parameters
    ----------
    monkeypatch : object
        Value supplied for ``monkeypatch``.

    Returns
    -------
    object
        Result produced by the callable.
    """
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
        """Read a selected region from an in-memory tile.

        Parameters
        ----------
        tile_idx : object
            Value supplied for ``tile idx``.
        ch_sel : object
            Value supplied for ``ch sel``.
        z_slice : object
            Value supplied for ``z slice``.
        y_slice : object
            Value supplied for ``y slice``.
        x_slice : object
            Value supplied for ``x slice``.

        Returns
        -------
        object
            Result produced by the callable.
        """
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
    """Verify pyramid levels are generated correctly in spatial blocks.

    Parameters
    ----------
    tmp_path : object
        Value supplied for ``tmp path``.
    downsample_method : object
        Value supplied for ``downsample method``.

    Returns
    -------
    None
        No value is returned.
    """
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
