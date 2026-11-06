"""Discover a mapped acquisition beside the untouched legacy array."""

import pytest

from opm_processing.dataio.acquisition import resolve_acquisition_path
from opm_processing.dataio.inspection import inspect_dataset


@pytest.mark.unit
def test_resolver_prefers_acquisition_group_over_legacy_array(
    current_opm_v2_stage_scan, legacy_raw_factory
):
    """Select the real OME acquisition beside a native legacy raw array."""
    raw = current_opm_v2_stage_scan
    legacy_raw_factory()
    assert resolve_acquisition_path(raw.parent) == raw


@pytest.mark.unit
def test_inspector_direct_legacy_array_requests_mapping(legacy_raw_factory):
    """The inspector explains unsupported raw arrays without loading journals."""
    raw = legacy_raw_factory().path
    with pytest.raises(ValueError, match="map-legacy-opm"):
        inspect_dataset(raw)


@pytest.mark.unit
def test_inspector_ignores_legacy_array_next_to_current_raw(
    current_opm_v2_stage_scan, legacy_raw_factory
):
    """Normal OME-Zarr inspection survives an adjacent legacy source array."""
    raw = current_opm_v2_stage_scan
    legacy_raw_factory()
    assert inspect_dataset(raw.parent)["raw"]["path"] == str(raw)
