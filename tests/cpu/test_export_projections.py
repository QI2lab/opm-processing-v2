"""Physical projection and export units with isolated file structures."""

from dataclasses import replace

import numpy as np
import pytest
import tifffile

from opm_processing import export_projections as exporter


@pytest.mark.unit
def test_physical_sphere_projections_and_scale(projection_sphere):
    """A sphere on anisotropic voxels must display circular in all three views."""
    case = projection_sphere
    sphere = case.sphere
    spacing = case.spacing
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
@pytest.mark.parametrize("planes, period", [(25, 0.03), (21, 0.005), (None, None)])
def test_acquired_volume_timing(acquisition_metadata, planes, period):
    """Hardware cadence is independent of exposure, channels and processed Z size."""
    acquisition = replace(
        acquisition_metadata,
        acquired_plane_count=planes,
        plane_period_s=period,
    )
    if planes is None:
        with pytest.raises(ValueError, match="hardware volume timing"):
            exporter.volume_interval_ms(acquisition)
    else:
        assert exporter.volume_interval_ms(acquisition) == pytest.approx(
            1000 * planes * period
        )
    assert exporter.timestamp(80 * 750 + 123) == "01:00:123"


@pytest.mark.unit
@pytest.mark.parametrize("stored_timing", [False, True])
def test_disk_export_preserves_brightness_and_separates_series(
    projection_export_case, tmp_path
):
    """Known linear intensities retain the same mapping over time in each series."""
    case = projection_export_case
    data = case.data
    source_path = case.source_path
    acquisition = case.acquisition
    exporter.export_dataset(source_path, tmp_path, acquisition, 1)
    files = sorted(tmp_path.rglob("*.tiff"))
    assert len(files) == 8
    for path in files:
        with tifffile.TiffFile(path) as tif:
            metadata = tif.shaped_metadata[0]
            channel = int(path.parent.name[1:])
            time = metadata["timepoint"]
            # Contrast stays fixed to the first object in each channel. Check
            # actual XY pixels outside the annotation against that known ramp.
            low, high = np.percentile(data[0, channel], (0.001, 99.999))
            projection = data[time, channel].max(axis=0)
            expected = np.rint(np.clip((projection - low) / (high - low), 0, 1) * 255)
            np.testing.assert_allclose(
                tif.asarray()[40:48, 40:64], expected[40:48, 40:64], rtol=0, atol=1
            )
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
