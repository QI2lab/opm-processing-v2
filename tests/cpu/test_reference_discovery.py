"""Discover a mapped acquisition beside the untouched legacy array."""

import json

import pytest

from opm_processing.dataio.acquisition import resolve_acquisition_path
from opm_processing.dataio.inspection import inspect_dataset


def _bare_array(path):
    path.mkdir()
    (path / ".zarray").write_text(json.dumps({"zarr_format": 2}))
    (path / ".zattrs").write_text("{}")


@pytest.mark.unit
def test_resolver_prefers_acquisition_group_over_legacy_array(tmp_path):
    """The original source array cannot hide its mapped acquisition group."""
    _bare_array(tmp_path / "raw.zarr")
    mapped = tmp_path / "raw.ome.zarr"
    mapped.mkdir()
    (mapped / "zarr.json").write_text(
        json.dumps(
            {"zarr_format": 3, "node_type": "group", "attributes": {"opm_v2": {}}}
        )
    )
    assert resolve_acquisition_path(tmp_path) == mapped


@pytest.mark.unit
def test_inspector_direct_legacy_array_requests_mapping(tmp_path):
    """The inspector explains unsupported raw arrays without loading journals."""
    raw = tmp_path / "raw.zarr"
    _bare_array(raw)
    with pytest.raises(ValueError, match="map-legacy-opm"):
        inspect_dataset(raw)


@pytest.mark.unit
def test_inspector_ignores_legacy_array_next_to_current_raw(current_opm_v2_stage_scan):
    """Normal OME-Zarr inspection survives an adjacent legacy source array."""
    raw = current_opm_v2_stage_scan
    _bare_array(raw.parent / "legacy.zarr")
    assert inspect_dataset(raw.parent)["raw"]["path"] == str(raw)
