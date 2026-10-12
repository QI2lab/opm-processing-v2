"""The Typer inspector preserves positional input, JSON and plain errors."""

import json
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from opm_processing.dataio import acquisition


@pytest.mark.unit
def test_inspector_help_and_required_path():
    """Keep one positional path and standard help/error exit codes."""
    runner = CliRunner()
    help_result = runner.invoke(acquisition.app, ["--help"])
    assert help_result.exit_code == 0
    assert "path" in help_result.output.lower()
    assert "Zarr store or containing directory" in help_result.output
    missing = runner.invoke(acquisition.app, [])
    assert missing.exit_code == 2
    assert "Missing argument" in missing.output


@pytest.mark.integration
def test_inspector_emits_actual_metadata_without_opening_pixels(
    tmp_path, current_opm_v2_stage_scan, monkeypatch
):
    """Emit the existing SDK manifest without invoking image array access."""
    expected = acquisition.inspect_acquisition(current_opm_v2_stage_scan).to_dict()

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata inspector opened pixel arrays")

    monkeypatch.setattr(acquisition, "open_acquisition_datastore", forbidden)
    result = CliRunner().invoke(acquisition.app, [str(current_opm_v2_stage_scan)])
    assert result.exit_code == 0, result.output
    assert result.output == json.dumps(expected, indent=2) + "\n"


@pytest.mark.integration
def test_inspector_main_emits_plain_python_traceback(tmp_path):
    """Exercise the installed entry-point callable in a separate interpreter."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from opm_processing.dataio.acquisition import main; main()",
            str(tmp_path / "absent.ome.zarr"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert result.stderr.startswith("Traceback (most recent call last):")
    assert "absent.ome.zarr" in result.stderr
    assert "╭" not in result.stderr and "╰" not in result.stderr
