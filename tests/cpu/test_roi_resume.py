"""Interrupted ROI processing must retain only durably completed channels."""

import importlib

import numpy as np
import pytest

from opm_processing.dataio.position_collection import (
    open_position_collection,
)
from opm_processing.dataio.processing_state import ProcessingState
from tests.simulation.camera import digitize


@pytest.mark.integration
def test_empty_roi_channel_is_checkpointed_and_skipped(
    roi_run, monkeypatch, assert_roi_point_accuracy
):
    """Persist an empty channel before interruption and preserve it on resume."""
    run = roi_run
    camera = run.truth.microscope.camera
    dark = digitize(np.zeros_like(run.truth.means[0, 1, 0]), camera, seed=73)
    run.raw_arrays[1][0, 0].write(dark).result()
    real_complete = ProcessingState.complete_channel

    def stop_after_empty(self, *args, **kwargs):
        """Interrupt after recording a deliberately empty channel."""
        real_complete(self, *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(ProcessingState, "complete_channel", stop_after_empty)
    options = {
        "acquisition": run.metadata,
        "roi_selection": run.roi,
        "output_dir": run.output_dir,
        "max_projection": False,
        "create_fused_max_projection": False,
        # Zero exposure retains read noise and ADC quantization. Six calibrated
        # noise standard deviations separate it from the bright optical sample.
        "skip_empty_below": 6
        * np.sqrt(camera.read_noise_e**2 + camera.electrons_per_adu**2 / 12),
        "resume": True,
    }
    with pytest.raises(KeyboardInterrupt):
        run.process.process_skewed(run.source, **options)
    output = run.output_dir / "sample_deskewed.ome.zarr"
    state_path = run.output_dir / "sample.processing.json"
    state = ProcessingState.read(state_path)
    assert state.completed_channels(output) == {(0, 1, 0)}
    assert state.zero_channels(output) == {(0, 1, 0)}
    assert state.completed_tiles(output) == set()

    changed = digitize(2 * run.truth.means[0, 1, 0], camera, seed=131)
    assert changed.max() < np.iinfo(np.uint16).max
    run.raw_arrays[1][0, 0].write(changed).result()
    monkeypatch.setattr(ProcessingState, "complete_channel", real_complete)
    run.process.process_skewed(run.source, **options)
    result = open_position_collection(output)
    assert not np.any(result.arrays[0][0, 0].read().result())
    for index, array in enumerate(result.arrays):
        assert_roi_point_accuracy(
            array.read().result(),
            run.truth,
            result.spatial_origins_zyx_um[index],
            result.voxel_size_um,
            channels=(1, 2) if index == 0 else None,
            positions=(index + 1,),
        )
    assert ProcessingState.read(state_path).zero_channels(output) == {(0, 1, 0)}


@pytest.mark.unit
def test_recreated_output_does_not_reuse_stale_checkpoints(tmp_path):
    """An absent output store cannot be skipped using surviving JSON checkpoints."""
    process = importlib.import_module("opm_processing.process")
    source = tmp_path / "sample.ome.zarr"
    output = tmp_path / "sample_deskewed.ome.zarr"
    state = ProcessingState.create(tmp_path / "sample.processing.json", source)
    state.initialize_run(output, configuration={}, overwrite=True)
    state.complete_channel(output, 0, 1, 0)
    state.complete_tile(output, 0, 1)
    assert not output.exists()

    reopened = process.initialize_processing_state(
        output_dir=tmp_path,
        source_path=source,
        output_path=output,
        configuration={},
        resume=True,
        output_preexisting=False,
    )
    assert reopened.completed_channels(output) == set()
    assert reopened.completed_tiles(output) == set()
