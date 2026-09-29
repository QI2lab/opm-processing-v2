"""Physical geometry and mocked-store integration checks for projection export."""

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from opm_processing import export_projections as exporter


@pytest.mark.unit
def test_physical_sphere_projections_and_scale():
    """A sphere on anisotropic voxels must display circular in all three views."""
    spacing = (0.4, 0.2, 0.1)
    shape = (31, 61, 121)
    z, y, x = np.meshgrid(
        *[(np.arange(n) - (n - 1) / 2) * d for n, d in zip(shape, spacing)],
        indexing="ij",
    )
    sphere = (x * x + y * y + z * z <= 4**2).astype(float)
    canvas, pixel_um, _ = exporter.make_canvas(sphere, spacing, (0, 1), 1234, 2)
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
def test_mocked_store_export_preserves_brightness_and_separates_series(
    tmp_path, monkeypatch
):
    """Known linear intensities retain the same mapping over time in each series."""

    class Store:
        def __init__(self, data):
            self.data = data
            self.shape = data.shape

        def __getitem__(self, item):
            return Store(self.data[item])

        def read(self):
            result = Future()
            result.set_result(self.data)
            return result

    first = np.linspace(0, 100, 10 * 20 * 30).reshape(10, 20, 30)
    data = np.stack([np.stack([first, first * 2]), np.stack([first * 0.5, first])])
    collection = SimpleNamespace(
        arrays=[Store(data), Store(data)],
        voxel_size_um=(0.1, 0.1, 0.1),
        channel_names=("488", "637"),
    )
    monkeypatch.setattr(exporter, "open_position_collection", lambda _: collection)
    acquisition = SimpleNamespace(
        scan_position_count=25, channels=[SimpleNamespace(exposure_ms=10)] * 2
    )
    exporter.export_dataset(
        Path("sample_decon_deskewed.ome.zarr"), tmp_path, acquisition, 1
    )
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

    from opm_processing import encode_projections as encoder

    encoded = []
    monkeypatch.setattr(
        encoder, "encode_sequence", lambda *args, **kwargs: encoded.append(args)
    )
    selected = tmp_path / "selected"
    exporter.export_dataset(
        Path("sample_decon_deskewed.ome.zarr"),
        selected,
        acquisition,
        1,
        timepoints=(1, 2),
        video=True,
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
    assert len(encoded) == 4
    assert all(len(args[0]) == 1 for args in encoded)
    assert all(args[1].name.endswith("_t0001-t0001.mp4") for args in encoded)
    for bounds in ((-1, 2), (1, 1), (0, 3)):
        with pytest.raises(ValueError, match="START < STOP"):
            exporter.export_dataset(
                Path("sample_decon_deskewed.ome.zarr"),
                tmp_path / "invalid",
                acquisition,
                timepoints=bounds,
            )
    assert not (tmp_path / "invalid").exists()


@pytest.mark.integration
def test_root_only_cli_resolution(tmp_path, monkeypatch):
    """Both CLIs select the full processed store and only its corresponding frames."""
    from typer.testing import CliRunner
    from opm_processing import encode_projections as encoder

    dataset = tmp_path / "sample_decon_deskewed.ome.zarr"
    dataset.mkdir()
    (tmp_path / "sample_max_z_decon_deskewed.ome.zarr").mkdir()
    (tmp_path / "sample_deskewed.ome.zarr").mkdir()
    called = []
    options = []
    monkeypatch.setattr(exporter, "acquisition_for", lambda *args: "metadata")
    monkeypatch.setattr(
        exporter,
        "export_dataset",
        lambda *args, **kwargs: (called.append(args), options.append(kwargs)),
    )
    runner = CliRunner()
    result = runner.invoke(exporter.app, [str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (
        runner.invoke(exporter.app, [str(tmp_path), "--downsample", "4"]).exit_code != 0
    )
    ranged = runner.invoke(exporter.app, [str(tmp_path), "--timepoints", "100", "200"])
    assert ranged.exit_code == 0, ranged.output
    assert options[-1]["timepoints"] == (100, 200)
    assert called[0][:3] == (dataset, tmp_path / "projection_frames", "metadata")
    assert runner.invoke(exporter.app, [str(dataset)]).exit_code != 0
    frames = (
        tmp_path
        / "projection_frames"
        / dataset.name.removesuffix(".ome.zarr")
        / "p000/c000"
    )
    frames.mkdir(parents=True)
    frame = frames / "sample_decon_deskewed_t0000.tiff"
    frame.touch()
    reduced_frame = frames / "downsample_4x" / frame.name
    reduced_frame.parent.mkdir()
    reduced_frame.touch()
    unrelated = tmp_path / "projection_frames/unrelated"
    unrelated.mkdir()
    (unrelated / "other_t0000.tiff").touch()
    encoded = []
    monkeypatch.setattr(
        encoder, "encode_sequence", lambda *args, **kwargs: encoded.append(args)
    )
    result = runner.invoke(encoder.app, [str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert len(encoded) == 1
    assert encoded[0][0] == [frame]
    assert runner.invoke(encoder.app, [str(frames)]).exit_code != 0
    (tmp_path / "another_decon_deskewed.ome.zarr").mkdir()
    with pytest.raises(ValueError, match="Expected one"):
        exporter.find_datasets(tmp_path)
