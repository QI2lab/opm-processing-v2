"""Recorded external stores remain distinct from stale same-named local stores."""

from __future__ import annotations

from pathlib import Path

import pytest

from opm_processing.dataio.processing_state import resolve_store_path

pytestmark = pytest.mark.unit


def test_existing_absolute_store_wins_over_same_named_local_store(tmp_path):
    """A split-root fusion link must not bind an abandoned local fusion."""
    destination = tmp_path / "F" / "sample_fused.ome.zarr"
    journal_directory = tmp_path / "G"
    stale = journal_directory / destination.name
    destination.mkdir(parents=True)
    stale.mkdir(parents=True)
    (destination / "identity").write_text("completed fusion", "utf-8")
    (stale / "identity").write_text("abandoned fusion", "utf-8")

    resolved = resolve_store_path(str(destination), journal_directory)

    assert resolved == destination.resolve()
    assert (resolved / "identity").read_text("utf-8") == "completed fusion"


def test_unavailable_absolute_store_falls_back_to_relocated_local_store(tmp_path):
    """A moved acquisition remains readable after its original path disappears."""
    recorded = tmp_path / "unavailable" / "sample_fused.ome.zarr"
    current = tmp_path / "relocated"
    local = current / recorded.name
    local.mkdir(parents=True)

    assert resolve_store_path(recorded, current) == local.resolve()


def test_unavailable_absolute_store_without_local_copy_retains_recorded_path(tmp_path):
    """The caller receives the explicit missing path for actionable validation."""
    recorded = tmp_path / "unavailable" / "sample_fused.ome.zarr"

    assert resolve_store_path(recorded, tmp_path / "current") == recorded.resolve()


@pytest.mark.parametrize("relative", ["sample.ome.zarr", "rendered/sample.ome.zarr"])
def test_relative_store_links_resolve_against_current_directory(tmp_path, relative):
    """Existing relative and nested links keep their recorded directory structure."""
    target = tmp_path / relative
    target.mkdir(parents=True)

    assert resolve_store_path(relative, tmp_path) == target.resolve()


def test_unavailable_nested_relative_store_falls_back_to_local_basename(tmp_path):
    """Legacy relocation fallback still finds a store moved out of a subfolder."""
    local = tmp_path / "sample.ome.zarr"
    local.mkdir()

    assert resolve_store_path("old/nested/sample.ome.zarr", tmp_path) == local.resolve()


def test_unavailable_windows_absolute_link_preserves_portable_basename(tmp_path):
    """Windows drive-qualified metadata can find a relocated store on any host."""
    local = tmp_path / "sample.ome.zarr"
    local.mkdir()

    assert (
        resolve_store_path(Path(r"Z:\unavailable\sample.ome.zarr"), tmp_path)
        == local.resolve()
    )
