"""Bio-Formats2Raw virtualization preserves legacy raw image bytes and geometry."""

import json

import numpy as np
import pytest
import refkv
import refkv.zarr
import zarr
from yaozarrs import open_group, v05

from opm_processing.dataio.acquisition import (
    inspect_acquisition,
    open_acquisition_datastore,
    resolve_acquisition_path,
)
from opm_processing.dataio.inspection import inspect_dataset
from opm_processing.dataio.legacy_reference import map_legacy_acquisition


@pytest.mark.unit
def test_reference_archive_preserves_native_pixels_and_bioformats2raw_geometry(
    tmp_path, legacy_raw_factory, stored_file_bytes
):
    """Standard virtual hierarchy and the pipeline read identical native pixels."""
    case = legacy_raw_factory(channels=2, interleaved=True)
    raw, data = case.path, case.data
    original = stored_file_bytes(raw)
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
    assert stored_file_bytes(raw) == original
    report = json.loads((mapped / "mapping_report.json").read_text())
    assert report["pixel_payload_bytes_copied"] == 0
    assert report["referenced_chunks"] == 12


@pytest.mark.unit
def test_mapper_rejects_missing_raw_chunk_without_publishing(
    tmp_path, legacy_raw_factory
):
    """Missing data with legacy null fill must never become invented zeros."""
    raw = legacy_raw_factory(channels=2, interleaved=True).path
    (raw / "0/1/1/2/0/0").unlink()
    with pytest.raises(FileNotFoundError):
        map_legacy_acquisition(raw, workers=2)
    assert not (tmp_path / "legacy.ome.zarr").exists()


@pytest.mark.unit
def test_mapper_does_not_overwrite_existing_mapping(tmp_path, legacy_raw_factory):
    """An existing destination stays unchanged on an accidental second import."""
    raw = legacy_raw_factory(channels=2, interleaved=True).path
    mapped = tmp_path / "legacy.ome.zarr"
    mapped.mkdir()
    sentinel = mapped / "keep"
    sentinel.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        map_legacy_acquisition(raw)
    assert sentinel.read_bytes() == b"original"
