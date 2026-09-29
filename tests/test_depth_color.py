"""Depth-color unit checks and mocked-store export integration."""

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from opm_processing.imageprocessing.depth_color import (
    depth_projection,
    depth_palette,
    depth_legends,
)
from opm_processing import export_projections as exporter


@pytest.mark.unit
def test_anisotropic_point_source_depth_and_brightness():
    spacing = (0.4, 0.2, 0.1)
    shape = (21, 31, 41)
    center = (6, 19, 28)
    coordinates = np.indices(shape)
    squared_distance = sum(
        ((coordinates[a] - center[a]) * spacing[a]) ** 2 for a in range(3)
    )
    volume = 80 * np.exp(-squared_distance / (2 * 0.3**2))
    for axis in range(3):
        projection = depth_projection(volume, axis, (0, 100))
        pixel = tuple(center[a] for a in range(3) if a != axis)
        expected = (depth_palette(shape[axis])[center[axis]] * 0.8).astype(np.uint8)
        np.testing.assert_array_equal(projection[pixel], expected)
    legends = depth_legends(shape, spacing)
    assert [entry["axis"] for entry in legends] == ["Z", "Y", "X"]
    np.testing.assert_allclose([entry["max_um"] for entry in legends], [8, 6, 4])


@pytest.mark.unit
def test_raw_maximum_precedes_display_clipping():
    volume = np.array([[[100]], [[200]]], dtype=float)
    np.testing.assert_array_equal(
        depth_projection(volume, 0, (0, 50))[0, 0], depth_palette(2)[1]
    )
    np.testing.assert_array_equal(depth_projection(np.zeros((2, 2, 2)), 0, (0, 1)), 0)
    np.testing.assert_array_equal(
        depth_projection(np.ones((2, 1, 1)), 0, (0, 1))[0, 0], depth_palette(2)[0]
    )


@pytest.mark.integration
def test_color_tiff_export_from_mocked_store(tmp_path, monkeypatch):
    class Store:
        def __init__(self, data):
            self.data, self.shape = data, data.shape

        def __getitem__(self, item):
            return Store(self.data[item])

        def read(self):
            future = Future()
            future.set_result(self.data)
            return future

    data = np.zeros((4, 1, 21, 31, 41), dtype=np.float32)
    data[:, 0, 6, 19, 28] = 100
    data[1:, 0, 12, 10, 20] = 50
    collection = SimpleNamespace(
        arrays=[Store(data)], voxel_size_um=(0.4, 0.2, 0.1), channel_names=["488"]
    )
    monkeypatch.setattr(exporter, "open_position_collection", lambda _: collection)
    acquisition = SimpleNamespace(
        scan_position_count=21, channels=[SimpleNamespace(exposure_ms=5)]
    )
    exporter.export_dataset(
        Path("sample_decon_deskewed.ome.zarr"), tmp_path, acquisition, depth_color=True
    )
    files = list(tmp_path.rglob("*.tiff"))
    assert len(files) == 4
    for path in files:
        assert "depth_color" in path.parts
        with tifffile.TiffFile(path) as tif:
            assert tif.pages[0].photometric.name == "RGB"
            assert tif.pages[0].compression.name in ("DEFLATE", "ADOBE_DEFLATE")
            metadata = tif.shaped_metadata[0]
            assert metadata["axes"] == "YXS"
            assert len(metadata["depth_color"]["legends"]) == 3
            assert metadata["volume_interval_ms"] == 105
            rgb = tif.asarray()
            assert rgb.shape[-1] == 3
            assert np.any(rgb[..., 0] != rgb[..., 1])

    serial = tmp_path / "serial"
    exporter.export_dataset(
        Path("sample_decon_deskewed.ome.zarr"),
        serial,
        acquisition,
        depth_color=True,
        workers=1,
    )
    for path in files:
        other = serial / path.relative_to(tmp_path)
        np.testing.assert_array_equal(tifffile.imread(path), tifffile.imread(other))
        with tifffile.TiffFile(path) as a, tifffile.TiffFile(other) as b:
            assert a.shaped_metadata == b.shaped_metadata


@pytest.mark.unit
def test_timepoint_workers_overlap_and_propagate_failure():
    from threading import Barrier, Lock

    barrier = Barrier(3, timeout=5)
    lock = Lock()
    active = 0
    maximum = 0

    def work(t):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        barrier.wait()
        with lock:
            active -= 1

    assert sorted(exporter.run_timepoints(work, range(6), 3)) == list(range(6))
    assert maximum == 3

    def fail(t):
        raise RuntimeError("read failed")

    with pytest.raises(RuntimeError, match="read failed"):
        list(exporter.run_timepoints(fail, range(100), 2))
