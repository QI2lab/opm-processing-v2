"""Read unchanged compressed legacy pixels through refkv and TensorStore."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys

import numpy as np
import pytest

from opm_processing.dataio.refkv_reader import open_refkv_datastore


@pytest.mark.unit
def test_refkv_native_pixels_axes_edge_chunks_and_read_only(refkv_mapping_factory):
    """Preserve positions, time, channels, Z and padded edge chunks exactly."""
    metadata, expected, _ = refkv_mapping_factory()
    store = open_refkv_datastore(metadata)
    assert tuple(store.shape) == expected.shape
    assert store.readable and not store.writable
    np.testing.assert_array_equal(store.read().result(), expected)
    selection = (slice(None), slice(1, 3), 1, slice(1, 5), slice(2, 7), slice(3, 9))
    np.testing.assert_array_equal(store[selection].read().result(), expected[selection])
    with pytest.raises(ValueError, match=r"[Ww]rit|read.only"):
        store[0, 0, 0, 0, 0, 0].write(999).result()


@pytest.mark.unit
def test_refkv_tensorstore_pickles_into_a_fresh_process(
    tmp_path, refkv_mapping_factory
):
    """Reopen references after spawn without serializing locks or reader handles."""
    metadata, expected, _ = refkv_mapping_factory()
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


@pytest.mark.unit
def test_refkv_missing_reference_fails_instead_of_filling_raw_pixels(
    refkv_mapping_factory,
):
    """Missing uncompleted legacy frames must never masquerade as black tissue."""
    metadata, _, _ = refkv_mapping_factory(missing_key="0/0/c/0/0/0/0/0")
    store = open_refkv_datastore(metadata)
    with pytest.raises(Exception, match="Missing referenced raw chunk"):
        store[0, 0, 0, 0].read().result()


@pytest.mark.unit
def test_refkv_missing_source_file_fails(refkv_mapping_factory):
    """Keep missing original bytes visible even when the archive key exists."""
    metadata, _, files = refkv_mapping_factory()
    files[0].unlink()
    store = open_refkv_datastore(metadata)
    with pytest.raises(Exception, match=r"legacy.zarr|No such|exist"):
        store[0, 0, 0, 0].read().result()


@pytest.mark.unit
def test_refkv_wrong_decoded_chunk_size_fails(refkv_mapping_factory):
    """Reject a valid Blosc payload describing a truncated native raw frame."""
    metadata, _, _ = refkv_mapping_factory(corrupt_key="0/0/c/0/0/0/0/0")
    store = open_refkv_datastore(metadata)
    with pytest.raises(Exception, match=r"decoded to .* bytes; expected"):
        store[0, 0, 0, 0].read().result()


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
    refkv_mapping_factory, field, value
):
    """Reject dtype, axis and codec changes that would reinterpret raw bytes."""
    metadata, _, files = refkv_mapping_factory()
    descriptor_path = metadata.path / "0/0/zarr.json"
    descriptor = json.loads(descriptor_path.read_text())
    descriptor[field] = value
    descriptor_path.write_text(json.dumps(descriptor))
    # Pixel reads cannot accidentally make the failing metadata check succeed.
    files[0].unlink()
    with pytest.raises(ValueError, match="Unsupported referenced raw"):
        open_refkv_datastore(metadata)
