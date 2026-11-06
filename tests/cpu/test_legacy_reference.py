"""Bio-Formats2Raw virtualization preserves legacy raw image bytes and geometry."""

import hashlib
import itertools
import json

import numpy as np
import pytest
import refkv
import refkv.zarr
import zarr
from numcodecs import Blosc
from yaozarrs import open_group, v05

from opm_processing.dataio.acquisition import (
    inspect_acquisition,
    open_acquisition_datastore,
    resolve_acquisition_path,
)
from opm_processing.dataio.inspection import inspect_dataset
from opm_processing.dataio.legacy_reference import map_legacy_acquisition
from tests.cpu.test_legacy_metadata import _write_raw


def _source(tmp_path):
    raw, _ = _write_raw(tmp_path / "legacy.zarr", channels=2, interleaved=True)
    shape = (1, 2, 2, 3, 4, 5)
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) + 100
    codec = Blosc(cname="lz4", clevel=5, shuffle=-1)
    for t, p, c, z in itertools.product(*(range(n) for n in shape[:4])):
        path = raw / f"{t}/{p}/{c}/{z}/0/0"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(codec.encode(data[t, p, c, z]))
    return raw, data


def _hash_files(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.unit
def test_bioformats2raw_mapping_works_through_pipeline_and_refkv(tmp_path):
    """Standard virtual hierarchy and the pipeline read identical native pixels."""
    raw, data = _source(tmp_path)
    original = _hash_files(raw)
    mapped = map_legacy_acquisition(raw, workers=2)
    root = open_group(mapped)
    assert isinstance(root.ome_metadata(), v05.Bf2Raw)
    assert root["OME"].ome_metadata().series == ["0", "1"]
    assert (mapped / "OME/METADATA.ome.xml").is_file()
    assert resolve_acquisition_path(tmp_path) == mapped
    acquisition = inspect_acquisition(tmp_path)
    assert inspect_dataset(tmp_path)["raw"]["path"] == str(mapped)
    assert acquisition.shape == data.shape
    assert acquisition.channel_names == ("488nm", "561nm")
    assert [channel.laser_power for channel in acquisition.channels] == [20, 35]
    assert acquisition.stage_positions_zxy == (
        (30.123, 100.123, 200.988),
        (30.123, 120.123, 200.988),
    )
    acquisition.validate_processing_calibration()
    np.testing.assert_array_equal(
        open_acquisition_datastore(acquisition).read().result(), data
    )
    virtual = zarr.open_group(
        refkv.zarr.store(refkv.open(str(mapped / "references.vzip"))), mode="r"
    )
    assert virtual.attrs["ome"]["bioformats2raw.layout"] == 3
    np.testing.assert_array_equal(virtual["1/0"][:], data[:, 1])
    # Ordinary filesystem readers cannot silently turn missing pixel files into zero images.
    with pytest.raises(
        (ValueError, NotImplementedError), match=r"[Ss]torage|transform"
    ):
        zarr.open_array(str(mapped / "0/0"), mode="r")
    assert not list(mapped.glob("*/0/c"))
    assert _hash_files(raw) == original
    report = json.loads((mapped / "mapping_report.json").read_text())
    assert report["pixel_payload_bytes_copied"] == 0
    assert report["referenced_chunks"] == 12


@pytest.mark.unit
def test_mapper_rejects_missing_raw_chunk_without_publishing(tmp_path):
    """Missing data with legacy null fill must never become invented zeros."""
    raw, _ = _source(tmp_path)
    (raw / "0/1/1/2/0/0").unlink()
    with pytest.raises(FileNotFoundError):
        map_legacy_acquisition(raw, workers=2)
    assert not (tmp_path / "legacy.ome.zarr").exists()


@pytest.mark.unit
def test_mapper_does_not_overwrite_existing_mapping(tmp_path):
    """An existing destination stays unchanged on an accidental second import."""
    raw, _ = _source(tmp_path)
    mapped = tmp_path / "legacy.ome.zarr"
    mapped.mkdir()
    sentinel = mapped / "keep"
    sentinel.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        map_legacy_acquisition(raw)
    assert sentinel.read_bytes() == b"original"
