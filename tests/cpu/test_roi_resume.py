"""Interrupted ROI processing must retain only durably completed channels."""

import importlib
from copy import copy

import numpy as np
import pytest
import tensorstore as ts

from opm_processing.dataio.position_collection import (
    open_position_collection,
)
from opm_processing.dataio.processing_state import ProcessingState
from tests.simulation.camera import digitize


@pytest.mark.integration
@pytest.mark.parametrize("roi_run", [2], indirect=True)
@pytest.mark.parametrize(
    "interruption",
    ("after_channel", "after_final_channel", "during_write", "before_checkpoint"),
)
def test_roi_resumes_at_last_durable_channel(
    roi_run, monkeypatch, interruption, assert_roi_point_accuracy
):
    """Restarting preserves completed data and retries uncommitted channel writes."""
    run = roi_run
    output = run.output_dir / "sample_deskewed.ome.zarr"
    state_path = run.output_dir / "sample.processing.json"
    real_complete = ProcessingState.complete_channel
    real_write = run.process.write_processed_channel
    channels = run.truth.pixels.shape[2]
    stop_key = (
        (0, 2, channels - 1) if interruption == "after_final_channel" else (0, 2, 0)
    )

    def checkpoint(self, path, time, position, channel, **kwargs):
        """Record a channel checkpoint only after the associated pixels are durable."""
        key = (time, position, channel)
        if key == stop_key and interruption == "before_checkpoint":
            raise KeyboardInterrupt
        real_complete(self, path, time, position, channel, **kwargs)
        if key == stop_key:
            raise KeyboardInterrupt

    def failing_write(target, value):
        """Inject a partial write failure before the channel checkpoint."""
        completed = ProcessingState.read(state_path).completed_channels(output)
        if completed == {(0, 1, channel) for channel in range(channels)}:
            # Leave a partial channel and fail the write future before checkpointing.
            real_write(target[:1], value[:1])
            readonly = ts.open(target.spec(), read=True, write=False).result()
            return real_write(readonly, value)
        return real_write(target, value)

    if interruption == "during_write":
        monkeypatch.setattr(run.process, "write_processed_channel", failing_write)
        expected_error = ValueError
    else:
        monkeypatch.setattr(ProcessingState, "complete_channel", checkpoint)
        expected_error = KeyboardInterrupt
    with pytest.raises(expected_error):
        run.process.process_skewed(run.source, **run.options, resume=True)

    state = ProcessingState.read(state_path)
    saved = {(0, 1, channel) for channel in range(channels)}
    if interruption == "after_channel":
        saved.add((0, 2, 0))
    elif interruption == "after_final_channel":
        saved.update((0, 2, channel) for channel in range(channels))
    assert state.completed_channels(output) == saved
    assert state.completed_tiles(output) == {(0, 1)}

    partial = open_position_collection(output)
    durable = {
        (position, channel): partial.arrays[position - 1][0, channel].read().result()
        for _, position, channel in saved
    }
    # A replay of any completed channel would now produce different pixels.
    changed = digitize(
        2 * (run.truth.expectation * run.truth.exposures[..., None, None, None]),
        run.truth.microscope.camera,
        seed=73,
    )
    assert changed.max() < np.iinfo(np.uint16).max
    for _, position, channel in saved:
        run.raw_arrays[position][0, channel].write(
            changed[0, position, channel]
        ).result()
    monkeypatch.setattr(ProcessingState, "complete_channel", real_complete)
    monkeypatch.setattr(run.process, "write_processed_channel", real_write)
    run.process.process_skewed(run.source, **run.options, resume=True)

    resumed = open_position_collection(output)
    for index, array in enumerate(resumed.arrays):
        pixels = array.read().result()
        assert_roi_point_accuracy(
            pixels,
            run.truth,
            resumed.spatial_origins_zyx_um[index],
            resumed.voxel_size_um,
        )
        for _, position, channel in saved:
            if position == index + 1:
                np.testing.assert_array_equal(
                    pixels[0, channel], durable[position, channel]
                )
    state = ProcessingState.read(state_path)
    assert state.completed_tiles(output) == {(0, 1), (0, 2)}
    assert state.completed_channels(output) == {
        (0, position, channel) for position in (1, 2) for channel in range(channels)
    }


@pytest.mark.integration
@pytest.mark.parametrize("roi_run", [2], indirect=True)
@pytest.mark.parametrize("resume", (True, False))
def test_roi_processing_respects_resume_choice(
    roi_run, resume, assert_roi_point_accuracy
):
    """Resume preserves saved photons; an explicit restart recalibrates input."""
    run = roi_run
    run.process.process_skewed(run.source, **run.options, resume=resume)
    output_path = run.output_dir / "sample_deskewed.ome.zarr"
    collection = open_position_collection(output_path)
    origins = collection.spatial_origins_zyx_um
    spacing = collection.voxel_size_um
    before = collection.arrays[0].read().result()
    unchanged = collection.arrays[1].read().result()
    for index, pixels in enumerate((before, unchanged)):
        assert_roi_point_accuracy(
            pixels,
            run.truth,
            collection.spatial_origins_zyx_um[index],
            collection.voxel_size_um,
        )
    changed = copy(run.truth)
    changed.exposures = run.truth.exposures.copy()
    changed.exposures[:, 1] *= 2
    pixels = digitize(
        2 * (run.truth.expectation * run.truth.exposures[:, 1, :, None, None, None]),
        changed.microscope.camera,
        seed=73,
    )
    assert pixels.max() < np.iinfo(np.uint16).max
    run.raw_arrays[1].write(pixels).result()
    run.process.process_skewed(run.source, **run.options, resume=resume)
    collection = open_position_collection(output_path)
    np.testing.assert_array_equal(collection.spatial_origins_zyx_um, origins)
    np.testing.assert_array_equal(collection.voxel_size_um, spacing)
    after = collection.arrays[0].read().result()
    np.testing.assert_array_equal(
        collection.arrays[1].read().result(),
        unchanged,
    )
    if resume is not False:
        np.testing.assert_array_equal(after, before)
    else:
        assert not np.array_equal(after, before)
        assert_roi_point_accuracy(
            after,
            changed,
            collection.spatial_origins_zyx_um[0],
            collection.voxel_size_um,
        )


@pytest.mark.integration
def test_empty_roi_channel_is_checkpointed_and_skipped(
    roi_run, monkeypatch, assert_roi_point_accuracy
):
    """Persist an empty channel before interruption and preserve it on resume."""
    run = roi_run
    camera = run.truth.microscope.camera
    dark = digitize(np.zeros_like(run.truth.expectation), camera, seed=73)
    run.raw_arrays[1][0, 0].write(dark).result()
    real_complete = ProcessingState.complete_channel

    def stop_after_empty(self, *args, **kwargs):
        """Interrupt after recording a deliberately empty channel."""
        real_complete(self, *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(ProcessingState, "complete_channel", stop_after_empty)
    options = {
        **run.options,
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

    changed = digitize(
        2 * (run.truth.expectation * run.truth.exposures[0, 1, 0]), camera, seed=131
    )
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
