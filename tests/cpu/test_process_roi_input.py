"""Resolve ROI processing inputs across separate raw and output directories."""

from pathlib import Path

import pytest

from opm_processing import process_roi as process_roi_module
from opm_processing.dataio.acquisition import acquisition_stem
from opm_processing.dataio.processing_state import ProcessingState

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("explicit_paths", (False, True))
def test_process_roi_uses_raw_source_and_processed_directory_defaults(
    tmp_path: Path,
    roi_run,
    processed_directory_factory,
    explicit_paths: bool,
    relocate_acquisition,
) -> None:
    """Resolve CLI inputs and persist the calibrated cropped raw pixels."""
    from dataclasses import replace

    import numpy as np

    from opm_processing.dataio.position_collection import open_position_collection

    run = roi_run
    source = run.source.resolve()
    stem = acquisition_stem(source)
    processed = processed_directory_factory(stem)
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
    relocated = relocate_acquisition(tmp_path, copy=explicit_paths)
    source = relocated / source.relative_to(tmp_path)
    processed = relocated / processed.relative_to(tmp_path)
    roi_path = relocated / roi_path.relative_to(tmp_path)
    expected_output = relocated / expected_output.relative_to(tmp_path)
    process_roi_module.process_roi(
        processed,
        roi_path,
        registration_channel=1,
        deconvolve=False,
        output=expected_output,
    )
    output = open_position_collection(expected_output / f"{stem}_deskewed.ome.zarr")
    for array, expected in zip(output.arrays, run.expected_tiles, strict=False):
        np.testing.assert_array_equal(array.read().result(), expected)
    state = ProcessingState.read(expected_output / f"{stem}.processing.json")
    assert state.document["source"]["path"] == str(source)
    assert state.completed_channels(expected_output / f"{stem}_deskewed.ome.zarr") == {
        (0, position, channel) for position in (1, 2) for channel in range(3)
    }
    assert not (source.parent / f"{stem}_roi").exists()
