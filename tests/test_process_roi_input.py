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
    for array, expected in zip(output.arrays, run.expected_tiles):
        np.testing.assert_array_equal(array.read().result(), expected)
    state = ProcessingState.read(expected_output / f"{stem}.processing.json")
    assert state.document["source"]["path"] == str(source)
    assert state.completed_channels(
        expected_output / f"{stem}_decon_deskewed.ome.zarr"
    ) == {(0, position, channel) for position in (1, 2) for channel in range(3)}
    assert not (source.parent / f"{stem}_roi").exists()


@pytest.mark.integration
@pytest.mark.parametrize("use_directory", (False, True))
def test_process_roi_still_accepts_raw_acquisition(
    opm_v2_skewed_zarr, use_directory: bool
) -> None:
    """Raw store and containing-directory inputs retain their original defaults."""
    source = opm_v2_skewed_zarr.path.resolve()
    acquisition, context = process_roi_module._resolve_roi_context(
        source.parent if use_directory else source
    )
    assert acquisition.path == source
    assert context == source.parent


@pytest.mark.integration
def test_process_roi_reports_missing_recorded_raw_source(tmp_path: Path) -> None:
    """Unavailable raw data is identified before creating any ROI outputs."""
    processed = _processed_directory(tmp_path, "sample")
    source = tmp_path / "unmounted" / "sample.ome.zarr"
    state_path = processed / "sample.processing.json"
    ProcessingState.create(state_path, source)

    with pytest.raises(
        FileNotFoundError, match="Raw acquisition.*unavailable"
    ) as error:
        process_roi_module.process_roi(processed)

    assert str(source) in str(error.value)
    assert str(state_path) in str(error.value)
    assert not (processed / "sample_roi").exists()


@pytest.mark.integration
def test_process_roi_rejects_ambiguous_processing_states(tmp_path: Path) -> None:
    """Multiple state documents require the caller to select the raw source."""
    processed = _processed_directory(tmp_path, "sample")
    for stem in ("sample", "other"):
        ProcessingState.create(
            processed / f"{stem}.processing.json", tmp_path / f"{stem}.ome.zarr"
        )

    with pytest.raises(ValueError, match="Expected one processing-state file") as error:
        process_roi_module._resolve_roi_context(processed)

    assert "sample.processing.json" in str(error.value)
    assert "other.processing.json" in str(error.value)


@pytest.mark.integration
def test_process_roi_explains_missing_source_state(tmp_path: Path) -> None:
    """Derived stores alone cannot provide the raw data needed for deconvolution."""
    processed = _processed_directory(tmp_path, "sample")
    with pytest.raises(ValueError, match="Pass the raw acquisition path"):
        process_roi_module._resolve_roi_context(processed)


@pytest.mark.integration
@pytest.mark.parametrize("recorded_source", (None, "", 123))
def test_process_roi_rejects_missing_source_identity(
    tmp_path: Path, recorded_source
) -> None:
    """Malformed source paths must never resolve to the working directory."""
    processed = _processed_directory(tmp_path, "sample")
    state = ProcessingState.create(
        processed / "sample.processing.json", tmp_path / "sample.ome.zarr"
    )
    state.document["source"]["path"] = recorded_source
    state.save()

    with pytest.raises(ValueError, match="lacks a source acquisition path"):
        process_roi_module._resolve_roi_context(processed)
