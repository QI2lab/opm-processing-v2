"""Unit tests of CLI validation and dispatch with mocked acquisition access."""

import importlib

import pytest
from typer.testing import CliRunner

from tests.undersampled_test_support import mock_acquisition


@pytest.mark.unit
@pytest.mark.parametrize(
    "options, message",
    [
        (["--decon-scan-upsample", "2"], "requires --deconvolve"),
        (["--deconvolve", "--decon-scan-upsample", "0"], "Invalid value"),
        (["--deconvolve", "--decon-scan-upsample", "1.5"], "Invalid value"),
        (
            ["--deconvolve", "--decon-scan-upsample", "2", "--decon-crop-scan", "10"],
            "unsupported",
        ),
        (
            [
                "--deconvolve",
                "--decon-scan-upsample",
                "2",
                "--decon-fallback-step-scan",
                "2",
            ],
            "unsupported",
        ),
    ],
)
def test_invalid_experimental_cli_options(options, message):
    """Reject invalid combinations before opening or modifying any dataset."""
    process = importlib.import_module("opm_processing.process")
    result = CliRunner().invoke(process.app, ["unused.zarr", *options])
    assert result.exit_code == 2, result.output
    assert message in result.output


@pytest.mark.unit
def test_experimental_cli_rejects_projection(monkeypatch):
    """Require an acquired scan axis rather than inventing missing 2D planes."""
    process = importlib.import_module("opm_processing.process")
    metadata = mock_acquisition("projection", (1, 1, 1, 1, 65, 49))
    output = metadata.path.parent / "output"
    monkeypatch.setattr(process, "inspect_acquisition", lambda _: metadata)
    monkeypatch.setattr(process, "_resolve_output_directory", lambda *args: output)
    result = CliRunner().invoke(
        process.app,
        [
            str(metadata.path),
            "--deconvolve",
            "--decon-scan-upsample",
            "2",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "requires a 3D" in result.output


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["mirror", "stage"])
def test_default_cli_keeps_existing_solver(mode, monkeypatch):
    """Ordinary --deconvolve must not silently select the experiment."""
    process = importlib.import_module("opm_processing.process")
    calls = []
    metadata = mock_acquisition(mode)
    monkeypatch.setattr(process, "inspect_acquisition", lambda _: metadata)
    monkeypatch.setattr(
        process, "_resolve_output_directory", lambda *args: metadata.path.parent
    )
    monkeypatch.setattr(
        process, "process_skewed", lambda **kwargs: calls.append(kwargs)
    )
    result = CliRunner().invoke(process.app, [str(metadata.path), "--deconvolve"])
    assert result.exit_code == 0, result.output
    assert calls[0]["deconvolve"] is True
    assert calls[0]["decon_scan_upsample"] is None


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["mirror", "stage"])
def test_live_cli_forwards_experimental_factor(mode, monkeypatch):
    """Pass the opt-in through live processing's separate dispatch branch."""
    process = importlib.import_module("opm_processing.process")
    metadata = mock_acquisition(mode)
    calls = []
    monkeypatch.setattr(
        process, "_resolve_output_directory", lambda *args: metadata.path.parent
    )
    monkeypatch.setattr(
        process, "_open_live_acquisition", lambda _: (metadata, object(), None)
    )
    monkeypatch.setattr(
        process, "process_skewed", lambda **kwargs: calls.append(kwargs)
    )
    result = CliRunner().invoke(
        process.app,
        [
            str(metadata.path),
            "--deconvolve",
            "--decon-scan-upsample",
            "3",
            "--live",
            "illumination.tif",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["decon_scan_upsample"] == 3
