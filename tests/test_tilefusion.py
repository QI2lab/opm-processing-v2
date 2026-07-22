"""Test tiled registration, fusion, and multiscale writing."""

import threading
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


def test_fused_shape_uses_minimal_multiscale_padding() -> None:
    """A one-pixel registration correction must not double a tile dimension."""
    fusion = TileFusion.__new__(TileFusion)
    fusion.unpadded_shape = (97, 4553, 6366)
    fusion.multiscale_factors = (2, 4, 8, 16, 32)
    fusion._is_2d = False

    fusion._pad_to_multiscale_multiple()

    assert fusion.padded_shape == (128, 4576, 6368)
    assert all(
        padded - original < 32
        for padded, original in zip(fusion.padded_shape, fusion.unpadded_shape)
    )


def test_require_gpu_reports_captured_backend_error(monkeypatch) -> None:
    """A requested CUDA backend must fail visibly instead of silently falling back."""
    monkeypatch.setattr(tilefusion_module, "USING_GPU", False)
    monkeypatch.setattr(
        tilefusion_module,
        "GPU_IMPORT_ERROR",
        ModuleNotFoundError("No module named 'cupy'"),
    )

    with pytest.raises(RuntimeError, match="No module named 'cupy'"):
        tilefusion_module.require_gpu_backend()


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
    fusion._max_workers = 2
    fusion._debug = False
    tiles = [
        np.ones((1, 2, 3, 4), dtype=np.float32),
        np.full((1, 2, 3, 4), 3.0, dtype=np.float32),
    ]
    first_reads = threading.Barrier(2)
    read_lock = threading.Lock()
    read_count = 0
    read_threads = set()

    def read_tile(
        self, tile_idx, ch_sel, z_slice, y_slice, x_slice, dtype=np.float32
    ):
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
        nonlocal read_count
        with read_lock:
            read_count += 1
            wait_for_peer = read_count <= 2
            read_threads.add(threading.get_ident())
        if wait_for_peer:
            first_reads.wait(timeout=5)
        selected = tiles[tile_idx][ch_sel, z_slice, y_slice, x_slice]
        return selected if dtype is None else selected.astype(dtype, copy=False)

    fusion._read_tile_volume = MethodType(read_tile, fusion)
    monkeypatch.setattr(
        tilefusion_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=256),
    )
    progress_calls = []
    real_tqdm = tilefusion_module.tqdm

    def recording_tqdm(*args, **kwargs):
        progress_calls.append(dict(kwargs))
        return real_tqdm(*args, disable=True, **kwargs)

    monkeypatch.setattr(tilefusion_module, "tqdm", recording_tqdm)

    fusion._fuse_by_blocks()

    expected = np.ones((1, 1, 2, 3, 6), dtype=np.uint16)
    expected[..., 2:4] = 2
    expected[..., 4:6] = 3
    np.testing.assert_array_equal(fusion.fused_ts.data, expected)
    assert fusion.fused_ts.tracker["writes"] > 1
    assert fusion.fused_ts.tracker["maximum"] <= 2
    assert fusion.fused_ts.tracker["active"] == 0
    assert fusion._fusion_stats["direct_regions"] > 0
    assert fusion._fusion_stats["blended_regions"] > 0
    assert fusion._fusion_stats["empty_blocks"] == 0
    assert len(read_threads) == 2
    assert progress_calls[0] == {
        "total": 1,
        "desc": "scale0",
        "leave": True,
        "unit": "timepoint",
    }
    assert progress_calls[1]["desc"] == "scale0 chunks"
    assert progress_calls[1]["leave"] is False
    assert progress_calls[1]["unit"] == "chunk"
    assert fusion.fused_ts.tracker["writes"] <= progress_calls[1]["total"]


@pytest.mark.parametrize("downsample_method", ["stride", "block_mean"])
def test_multiscales_are_generated_in_spatial_blocks(
    tmp_path, downsample_method, monkeypatch
):
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

    progress_calls = []
    real_tqdm = tilefusion_module.tqdm

    def recording_tqdm(*args, **kwargs):
        progress_calls.append(dict(kwargs))
        return real_tqdm(*args, disable=True, **kwargs)

    monkeypatch.setattr(tilefusion_module, "tqdm", recording_tqdm)
    fusion._write_multiscales()

    assert progress_calls == [
        {"total": 1, "desc": "scale1", "leave": True, "unit": "timepoint"},
        {"total": 8, "desc": "scale1 chunks", "leave": False, "unit": "chunk"},
        {"total": 1, "desc": "scale2", "leave": True, "unit": "timepoint"},
        {"total": 4, "desc": "scale2 chunks", "leave": False, "unit": "chunk"},
    ]

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
