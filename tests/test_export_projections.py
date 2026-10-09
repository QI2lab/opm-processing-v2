"""Physical projection units and simulated disk-to-disk export tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from opm_processing import export_projections as exporter
from opm_processing.dataio.position_collection import create_position_collection


@pytest.mark.unit
def test_physical_sphere_projections_and_scale():
    """A sphere on anisotropic voxels must display circular in all three views."""
    spacing = (0.4, 0.2, 0.1)
    shape = (31, 61, 121)
    z, y, x = np.meshgrid(
        *[
            (np.arange(n) - (n - 1) / 2) * d
            for n, d in zip(shape, spacing, strict=False)
        ],
        indexing="ij",
    )
    sphere = (x * x + y * y + z * z <= 4**2).astype(float)
    images, display_pixel_um, legends = exporter.projection_panels(
        sphere, spacing, (0, 1)
    )
    canvas, pixel_um, _ = exporter.assemble_canvas(
        *images, display_pixel_um, 1234, 2, legends=legends
    )
    assert pixel_um == 0.1
    panels = [canvas[:122, :121], canvas[134:258, :121], canvas[:122, 133:257]]
    for panel in panels:
        # Exclude the upper-left panel label from the object geometry measurement.
        panel = panel.copy()
        panel[:30, :40] = 0
        rows, cols = np.where(panel > 127)
        assert abs(np.ptp(rows) - np.ptp(cols)) <= 4
        assert abs(np.ptp(rows) * pixel_um - 8) <= 0.4
    # The physical two-micron bar occupies exactly twenty display pixels.
    assert np.count_nonzero(canvas[216, 149:]) == 20


@pytest.mark.unit
def test_acquired_volume_timing():
    """Use acquired scan counts and all channel exposures, with millisecond labels."""
    acquisition = SimpleNamespace(
        scan_position_count=25, channels=[SimpleNamespace(exposure_ms=10)] * 3
    )
    assert exporter.volume_interval_ms(acquisition) == 750
    assert exporter.timestamp(80 * 750 + 123) == "01:00:123"
    acquisition.channels[0] = SimpleNamespace(exposure_ms=None)
    with pytest.raises(ValueError, match="exposure"):
        exporter.volume_interval_ms(acquisition)


@pytest.mark.integration
def test_disk_export_preserves_brightness_and_separates_series(tmp_path):
    """Known linear intensities retain the same mapping over time in each series."""
    first = np.linspace(0, 100, 10 * 20 * 30).reshape(10, 20, 30)
    data = np.stack([np.stack([first, first * 2]), np.stack([first * 0.5, first])])
    source_path = tmp_path / "sample_decon_deskewed.ome.zarr"
    collection = create_position_collection(
        source_path,
        (2, 2, 2, 10, 20, 30),
        (0.1, 0.1, 0.1),
        channels=("488", "637"),
        dtype=np.float32,
    )
    for array in collection.arrays:
        array.write(data.astype(np.float32)).result()
    acquisition = SimpleNamespace(
        scan_position_count=25, channels=[SimpleNamespace(exposure_ms=10)] * 2
    )
    exporter.export_dataset(source_path, tmp_path, acquisition, 1)
    files = sorted(tmp_path.rglob("*.tiff"))
    assert len(files) == 8
    first_file = (
        tmp_path / "sample_decon_deskewed/p000/c000/sample_decon_deskewed_t0000.tiff"
    )
    second_file = first_file.with_name("sample_decon_deskewed_t0001.tiff")
    with tifffile.TiffFile(second_file) as tif:
        metadata = tif.shaped_metadata[0]
        np.testing.assert_allclose(metadata["contrast_limits"], (0.001, 99.999))
        assert metadata["elapsed_ms"] == 500
        assert metadata["volume_interval_ms"] == 500
    assert (
        tifffile.imread(first_file)[:20, :30].mean()
        > tifffile.imread(second_file)[:20, :30].mean()
    )

    selected = tmp_path / "selected"
    exporter.export_dataset(
        source_path,
        selected,
        acquisition,
        1,
        timepoints=(1, 2),
    )
    subset = sorted(selected.rglob("*.tiff"))
    assert len(subset) == 4
    assert all(path.name.endswith("_t0001.tiff") for path in subset)
    for path in subset:
        full = tmp_path / path.relative_to(selected)
        np.testing.assert_array_equal(tifffile.imread(path), tifffile.imread(full))
        with tifffile.TiffFile(path) as tif:
            assert tif.shaped_metadata[0]["elapsed_ms"] == 500
            assert tif.shaped_metadata[0]["timepoint"] == 1
    for bounds in ((-1, 2), (1, 1), (0, 3)):
        with pytest.raises(ValueError, match="START < STOP"):
            exporter.export_dataset(
                source_path,
                tmp_path / "invalid",
                acquisition,
                timepoints=bounds,
            )
    assert not (tmp_path / "invalid").exists()
