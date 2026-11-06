"""Read unchanged compressed legacy pixels through refkv and TensorStore."""

from __future__ import annotations

import itertools
import json
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from numcodecs import Blosc

from opm_processing.dataio.refkv_reader import open_refkv_datastore

refkv = pytest.importorskip("refkv")


def _mapping(tmp_path, *, missing_key=None, corrupt_key=None):
    """Make asymmetric, independently numbered pixels and compressed references."""
    shape = (2, 3, 2, 5, 7, 9)
    chunks = (1, 1, 2, 4, 5)
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    compressor = {
        "id": "blosc",
        "cname": "lz4",
        "clevel": 5,
        "shuffle": -1,
        "blocksize": 0,
    }
    codec = Blosc(**{key: value for key, value in compressor.items() if key != "id"})
    root = tmp_path / "mapped.ome.zarr"
    root.mkdir()
    sources_root = tmp_path / "legacy.zarr"
    sources_root.mkdir()
    manifest = {
        "format_version": 1,
        "archive": "references.vzip",
        "legacy_compressor": compressor,
        "source": str(sources_root),
    }
    (root / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"opm_processing_refkv": manifest},
            }
        )
    )
    descriptor = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [shape[0], *shape[2:]],
        "data_type": "uint16",
        "dimension_names": list("tczyx"),
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunks}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {
                "name": "blosc",
                "configuration": {
                    "cname": "lz4",
                    "clevel": 5,
                    "shuffle": "shuffle",
                    "typesize": 2,
                },
            },
        ],
        "attributes": {},
    }
    references, sources, source_files = {}, [], []
    for position in range(shape[1]):
        array_path = f"{position}/0"
        target = root / array_path
        target.mkdir(parents=True)
        metadata_bytes = json.dumps(descriptor).encode()
        (target / "zarr.json").write_bytes(metadata_bytes)
        references[f"{array_path}/zarr.json"] = metadata_bytes
        dimensions = descriptor["shape"]
        counts = tuple(
            (length + chunk - 1) // chunk
            for length, chunk in zip(dimensions, chunks, strict=True)
        )
        for indices in itertools.product(*(range(value) for value in counts)):
            starts = tuple(i * chunk for i, chunk in zip(indices, chunks, strict=True))
            stops = tuple(
                min(a + b, c)
                for a, b, c in zip(starts, chunks, dimensions, strict=True)
            )
            selections = tuple(slice(a, b) for a, b in zip(starts, stops, strict=True))
            native = np.zeros(chunks, dtype="<u2")
            native[
                tuple(slice(0, b - a) for a, b in zip(starts, stops, strict=True))
            ] = data[:, position][selections]
            key = f"{array_path}/c/" + "/".join(map(str, indices))
            raw_file = sources_root / str(len(source_files))
            raw_file.write_bytes(
                codec.encode(
                    native if key != corrupt_key else np.ones(3, dtype=np.uint16)
                )
            )
            source_files.append(raw_file)
            if key != missing_key:
                references[key] = refkv.Range(
                    source=len(sources), offset=0, length=raw_file.stat().st_size
                )
                sources.append(
                    refkv.Source(url=raw_file.as_uri(), size=raw_file.stat().st_size)
                )
    refkv.ReferenceStore(references, sources).save(root / "references.vzip")
    metadata = SimpleNamespace(
        path=root,
        axes=tuple("tpczyx"),
        shape=shape,
        array_paths=tuple(f"{p}/0" for p in range(shape[1])),
    )
    return metadata, data, source_files


@pytest.mark.cpu
@pytest.mark.unit
def test_refkv_native_pixels_axes_edge_chunks_and_read_only(tmp_path):
    """Preserve positions, time, channels, Z and padded edge chunks exactly."""
    metadata, expected, _ = _mapping(tmp_path)
    store = open_refkv_datastore(metadata)
    assert tuple(store.shape) == expected.shape
    assert store.readable and not store.writable
    np.testing.assert_array_equal(store.read().result(), expected)
    selection = (slice(None), slice(1, 3), 1, slice(1, 5), slice(2, 7), slice(3, 9))
    np.testing.assert_array_equal(store[selection].read().result(), expected[selection])
    with pytest.raises(ValueError, match=r"[Ww]rit|read.only"):
        store[0, 0, 0, 0, 0, 0].write(999).result()


@pytest.mark.cpu
@pytest.mark.unit
def test_refkv_tensorstore_pickles_into_a_fresh_process(tmp_path):
    """Reopen references after spawn without serializing locks or reader handles."""
    metadata, expected, _ = _mapping(tmp_path)
    store = open_refkv_datastore(metadata)
    # Populate the process-local cache before serializing, to test separation.
    assert int(store[0, 1, 0, 2, 3, 4].read().result()) == expected[0, 1, 0, 2, 3, 4]
    serialized = tmp_path / "store.pkl"
    serialized.write_bytes(pickle.dumps(store))
    code = (
        "import pickle,sys; from pathlib import Path; "
        "a=pickle.loads(Path(sys.argv[1]).read_bytes()); "
        "print(int(a[1,2,1,4,6,8].read().result()))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(serialized)],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert int(result.stdout.strip()) == expected[1, 2, 1, 4, 6, 8]


@pytest.mark.cpu
@pytest.mark.unit
def test_refkv_missing_reference_fails_instead_of_filling_raw_pixels(tmp_path):
    """Missing uncompleted legacy frames must never masquerade as black tissue."""
    metadata, _, _ = _mapping(tmp_path, missing_key="0/0/c/0/0/0/0/0")
    store = open_refkv_datastore(metadata)
    with pytest.raises(Exception, match="Missing referenced raw chunk"):
        store[0, 0, 0, 0].read().result()


@pytest.mark.cpu
@pytest.mark.unit
def test_refkv_missing_source_file_fails(tmp_path):
    """Keep missing original bytes visible even when the archive key exists."""
    metadata, _, files = _mapping(tmp_path)
    files[0].unlink()
    store = open_refkv_datastore(metadata)
    with pytest.raises(Exception, match=r"legacy.zarr|No such|exist"):
        store[0, 0, 0, 0].read().result()


@pytest.mark.cpu
@pytest.mark.unit
def test_refkv_wrong_decoded_chunk_size_fails(tmp_path):
    """Reject a valid Blosc payload describing a truncated native raw frame."""
    metadata, _, _ = _mapping(tmp_path, corrupt_key="0/0/c/0/0/0/0/0")
    store = open_refkv_datastore(metadata)
    with pytest.raises(Exception, match=r"decoded to .* bytes; expected"):
        store[0, 0, 0, 0].read().result()


@pytest.mark.cpu
@pytest.mark.unit
@pytest.mark.parametrize(
    "field,value",
    [
        ("data_type", "float32"),
        ("dimension_names", list("tzyxc")),
        ("codecs", [{"name": "gzip"}]),
    ],
)
def test_refkv_unsupported_array_metadata_fails_before_pixel_reads(
    tmp_path, field, value
):
    """Reject dtype, axis and codec changes that would reinterpret raw bytes."""
    metadata, _, files = _mapping(tmp_path)
    descriptor_path = metadata.path / "0/0/zarr.json"
    descriptor = json.loads(descriptor_path.read_text())
    descriptor[field] = value
    descriptor_path.write_text(json.dumps(descriptor))
    # Pixel reads cannot accidentally make the failing metadata check succeed.
    files[0].unlink()
    with pytest.raises(ValueError, match="Unsupported referenced raw"):
        open_refkv_datastore(metadata)
