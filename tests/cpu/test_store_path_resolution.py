"""Store relocation and split-root links resolve against prescribed provenance."""

from pathlib import Path

import pytest

from opm_processing.dataio.processing_state import resolve_store_path

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "recorded,available,prefer_local,expected",
    [
        ("original", ("original", "local"), False, "original"),
        ("original", ("original", "local"), True, "local"),
        ("original", ("local",), False, "local"),
        ("original", (), False, "original"),
        ("relative", ("local",), False, "local"),
        ("nested", ("nested",), False, "nested"),
        ("nested", ("local",), False, "local"),
        ("windows", ("local",), False, "local"),
    ],
)
def test_store_link_resolution_respects_provenance_and_local_copy_policy(
    fs, recorded, available, prefer_local, expected
):
    """Select existing original or copied stores without losing portable basenames."""
    directory = Path("/store-links/current").resolve()
    paths = {
        "original": directory.parent / "original/sample.ome.zarr",
        "local": directory / "sample.ome.zarr",
        "nested": directory / "nested/sample.ome.zarr",
    }
    links = {
        "original": paths["original"],
        "relative": Path("sample.ome.zarr"),
        "nested": Path("nested/sample.ome.zarr"),
        "windows": Path(r"Z:\unavailable\sample.ome.zarr"),
    }
    fs.create_dir(directory)
    for name in available:
        fs.create_dir(paths[name])
        fs.create_file(paths[name] / "identity", contents=name)
    resolved = resolve_store_path(links[recorded], directory, prefer_local=prefer_local)
    assert resolved == paths[expected]
    if available:
        assert (resolved / "identity").read_text("utf-8") == expected
    else:
        assert not resolved.exists()
