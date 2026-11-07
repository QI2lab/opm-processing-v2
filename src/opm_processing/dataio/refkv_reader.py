"""Read referenced legacy raw chunks through a normal TPCZYX TensorStore.

OME-Zarr metadata remains in the mapped acquisition directory. Compressed
pixel bytes live only in the original legacy store; ``refkv`` resolves each
chunk reference when TensorStore requests it. Neither source nor mapping can
be written through the returned arrays.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import tensorstore as ts
from numcodecs import Blosc

if TYPE_CHECKING:
    from pathlib import Path

    from opm_processing.dataio.acquisition import AcquisitionMetadata

_MAX_OPEN_ARCHIVES = 4
_ARCHIVES: OrderedDict[tuple[int, str], tuple[Any, threading.RLock]] = OrderedDict()
_ARCHIVE_LOCK = threading.Lock()


def _archive_bytes(archive: str, key: str) -> bytes:
    """Resolve one chunk with a bounded, process-local, synchronized reader."""
    cache_key = (os.getpid(), archive)
    with _ARCHIVE_LOCK:
        if cache_key not in _ARCHIVES:
            import refkv

            _ARCHIVES[cache_key] = (refkv.open(archive), threading.RLock())
            while len(_ARCHIVES) > _MAX_OPEN_ARCHIVES:
                _ARCHIVES.popitem(last=False)
        reader, lock = _ARCHIVES[cache_key]
        _ARCHIVES.move_to_end(cache_key)
    # Reader transports and their cached archive indexes are not shared
    # concurrently. Decompression happens outside this lock.
    with lock:
        encoded = reader.get(key)
    if encoded is None:
        raise FileNotFoundError(f"Missing referenced raw chunk {key!r} in {archive}")
    return encoded


@dataclass(frozen=True)
class _NativeChunkReader:
    """A spawn-picklable callback containing paths and codec metadata only."""

    archive: str
    array_path: str
    chunk_shape: tuple[int, ...]
    dtype: str
    compressor: dict[str, Any]

    def __call__(
        self,
        domain: ts.IndexDomain,
        output: np.ndarray,
        _parameters: ts.VirtualChunkedReadParameters,
    ) -> None:
        """Decode one original Blosc chunk and retain its requested valid edge."""
        starts = tuple(int(value) for value in domain.inclusive_min)
        stops = tuple(int(value) for value in domain.exclusive_max)
        indices = tuple(
            start // length
            for start, length in zip(starts, self.chunk_shape, strict=True)
        )
        local_slices = tuple(
            slice(start - index * length, stop - index * length)
            for start, stop, index, length in zip(
                starts, stops, indices, self.chunk_shape, strict=True
            )
        )
        if any(
            selection.start < 0 or selection.stop > length
            for selection, length in zip(local_slices, self.chunk_shape, strict=True)
        ):
            raise ValueError("TensorStore callback crossed a native raw chunk boundary")
        key = f"{self.array_path}/c/" + "/".join(map(str, indices))
        encoded = _archive_bytes(self.archive, key)
        codec = Blosc(**{k: v for k, v in self.compressor.items() if k != "id"})
        decoded = codec.decode(encoded)
        dtype = np.dtype(self.dtype)
        expected_bytes = int(np.prod(self.chunk_shape)) * dtype.itemsize
        if len(decoded) != expected_bytes:
            raise ValueError(
                f"Referenced raw chunk {key!r} decoded to {len(decoded)} bytes; "
                f"expected {expected_bytes}"
            )
        native = np.frombuffer(decoded, dtype=dtype).reshape(self.chunk_shape)
        output[...] = native[local_slices]


def _array_descriptor(
    path: Path, expected_shape: tuple[int, ...]
) -> tuple[tuple[int, ...], str]:
    """Reject metadata that cannot describe this unchanged legacy chunk stream."""
    descriptor = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
    if (
        descriptor.get("zarr_format") != 3
        or descriptor.get("node_type") != "array"
        or descriptor.get("data_type") != "uint16"
        or tuple(descriptor.get("shape", ())) != expected_shape
        or descriptor.get("dimension_names") != list("tczyx")
    ):
        raise ValueError(f"Unsupported referenced raw array metadata at {path}")
    grid = descriptor.get("chunk_grid", {})
    chunks = tuple(grid.get("configuration", {}).get("chunk_shape", ()))
    if (
        grid.get("name") != "regular"
        or len(chunks) != 5
        or any(not isinstance(value, int) or value < 1 for value in chunks)
    ):
        raise ValueError(f"Invalid referenced raw chunk shape at {path}")
    if descriptor.get("chunk_key_encoding") != {
        "name": "default",
        "configuration": {"separator": "/"},
    }:
        raise ValueError(f"Unsupported referenced raw chunk key encoding at {path}")
    codecs = descriptor.get("codecs", [])
    if (
        len(codecs) != 2
        or codecs[0] != {"name": "bytes", "configuration": {"endian": "little"}}
        or codecs[1].get("name") != "blosc"
    ):
        raise ValueError(f"Unsupported referenced raw codec pipeline at {path}")
    return chunks, "<u2"


def open_refkv_datastore(acquisition: AcquisitionMetadata) -> ts.TensorStore:
    """Open a mapped acquisition as a read-only logical TPCZYX TensorStore.

    Parameters
    ----------
    acquisition : AcquisitionMetadata
        Inspected OME-Zarr mapping whose root contains ``opm_processing_refkv``.

    Returns
    -------
    tensorstore.TensorStore
        Per-position native TCZYX views stacked along position. References are
        resolved lazily, with no image or compressed-chunk copies on disk.
    """
    root_path = acquisition.path.absolute()
    root = json.loads((root_path / "zarr.json").read_text(encoding="utf-8"))
    manifest = root.get("attributes", {}).get("opm_processing_refkv", {})
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported or missing refkv acquisition mapping version")
    archive_name = manifest.get("archive")
    if not isinstance(archive_name, str) or not archive_name:
        raise ValueError("Referenced acquisition has no archive path")
    archive = (root_path / archive_name).absolute()
    if not archive.is_file():
        raise FileNotFoundError(
            f"Referenced acquisition archive does not exist: {archive}"
        )
    compressor = manifest.get("legacy_compressor", {})
    if not isinstance(compressor, dict) or compressor.get("id") != "blosc":
        raise ValueError("Only unchanged legacy Blosc raw chunks are supported")
    # Validate constructor options at open, before a later callback receives them.
    Blosc(**{key: value for key, value in compressor.items() if key != "id"})
    if acquisition.axes != tuple("tpczyx") or len(acquisition.shape) != 6:
        raise ValueError("Referenced acquisition must have logical TPCZYX axes")
    if len(acquisition.array_paths) != acquisition.shape[1]:
        raise ValueError(
            "Referenced position count does not match acquisition metadata"
        )
    expected_shape = (acquisition.shape[0], *acquisition.shape[2:])
    arrays = []
    for array_path in acquisition.array_paths:
        chunks, dtype = _array_descriptor(root_path / array_path, expected_shape)
        reader = _NativeChunkReader(str(archive), array_path, chunks, dtype, compressor)
        arrays.append(
            ts.virtual_chunked(
                reader,
                dtype=np.dtype(dtype),
                domain=ts.IndexDomain(shape=expected_shape, labels=list("tczyx")),
                chunk_layout=ts.ChunkLayout(
                    read_chunk_shape=chunks, grid_origin=(0,) * 5
                ),
            )
        )
    return ts.stack(arrays, axis=1)
