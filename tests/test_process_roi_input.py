"""Resolve ROI processing inputs across separate raw and output directories."""

from pathlib import Path

import pytest
import zarr
from typer.testing import CliRunner

from opm_processing import process_roi as process_roi_module
from opm_processing.dataio.acquisition import acquisition_stem
from opm_processing.dataio.processing_state import ProcessingState


def _processed_directory(tmp_path: Path, stem: str) -> Path:
    """Create four derived stores without raw acquisition metadata."""
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    for label in ("deskewed", "fused", "max_z_deskewed", "max_z_fused"):
        zarr.open_group(output_dir / f"{stem}_{label}.ome.zarr", mode="w")
    return output_dir


@pytest.mark.integration
@pytest.mark.parametrize("explicit_paths", (False, True))
def test_process_roi_uses_raw_source_and_processed_directory_defaults(
    tmp_path: Path, roi_run, explicit_paths: bool
) -> None:
    """Resolve CLI inputs and persist the calibrated cropped raw pixels."""
    from dataclasses import replace

    import numpy as np

    from opm_processing.dataio.position_collection import open_position_collection

    run = roi_run
    source = run.source.resolve()
    stem = acquisition_stem(source)
    processed = _processed_directory(tmp_path, stem)
    ProcessingState.create(processed / f"{stem}.processing.json", source)
    roi = replace(run.roi, source_path=processed / f"{stem}_max_z_fused.ome.zarr")
    roi_path = (
        tmp_path / "custom_roi.json"
        if explicit_paths
        else processed / f"{stem}_roi.json"
    )
    roi.write(roi_path)
    expected_output = (
        tmp_path / "custom_output" if explicit_paths else processed / f"{stem}_roi"
    )
    args = [str(processed)]
    if explicit_paths:
        args.extend([str(roi_path), "--output", str(expected_output)])
    result = CliRunner().invoke(process_roi_module.app, args)

    assert result.exit_code == 0, result.exception
    output = open_position_collection(
        expected_output / f"{stem}_decon_deskewed.ome.zarr"
    )
    for array, expected in zip(output.arrays, run.expected_tiles, strict=False):
        np.testing.assert_array_equal(array.read().result(), expected)
    state = ProcessingState.read(expected_output / f"{stem}.processing.json")
    assert state.document["source"]["path"] == str(source)
    assert state.completed_channels(
        expected_output / f"{stem}_decon_deskewed.ome.zarr"
    ) == {(0, position, channel) for position in (1, 2) for channel in range(3)}
    assert not (source.parent / f"{stem}_roi").exists()
