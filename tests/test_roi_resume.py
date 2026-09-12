"""Interrupted ROI processing must retain only durably completed channels."""

from dataclasses import replace
import importlib

import numpy as np
import pytest
from typer.testing import CliRunner

from opm_processing import process_roi as roi_command
from opm_processing.dataio.position_collection import (
    open_position_collection,
)
from opm_processing.dataio.processing_state import ProcessingState


@pytest.mark.integration
@pytest.mark.parametrize(
    "interruption",
    ("after_channel", "after_final_channel", "during_write", "before_checkpoint"),
)
def test_roi_resumes_at_last_durable_channel(roi_run, monkeypatch, interruption):
    """Restarting preserves completed data and retries uncommitted channel writes."""
    run = roi_run
    output = run.output_dir / "sample_decon_deskewed.ome.zarr"
    state_path = run.output_dir / "sample.processing.json"
    reference_dir = run.output_dir.parent / "reference"
    roi_command.process_roi(run.source, run.roi_path, output=reference_dir)
    reference = open_position_collection(
        reference_dir / "sample_decon_deskewed.ome.zarr"
    )
    expected = [array.read().result() for array in reference.arrays]
    run.decon_calls.clear()
    real_complete = ProcessingState.complete_channel
    real_write = run.process._write_checkpointed_roi_channel
    stop_key = (0, 2, 2) if interruption == "after_final_channel" else (0, 2, 0)

    def checkpoint(self, path, time, position, channel, **kwargs):
        key = (time, position, channel)
        if key == stop_key and interruption == "before_checkpoint":
            raise KeyboardInterrupt
        real_complete(self, path, time, position, channel, **kwargs)
        if key == stop_key:
            raise KeyboardInterrupt

    def failing_write(target, value, state, path, key, **kwargs):
        if key == stop_key:

            class FailedTarget:
                def write(self, value):
                    # Leave a partially written channel, then fail its write future.
                    target[0].write(value[0]).result()
                    return self

                def result(self):
                    raise OSError("interrupted channel write")

            return real_write(FailedTarget(), value, state, path, key, **kwargs)
        return real_write(target, value, state, path, key, **kwargs)

    if interruption == "during_write":
        monkeypatch.setattr(
            run.process, "_write_checkpointed_roi_channel", failing_write
        )
        expected_error = OSError
    else:
        monkeypatch.setattr(ProcessingState, "complete_channel", checkpoint)
        expected_error = KeyboardInterrupt
    with pytest.raises(expected_error):
        roi_command.process_roi(run.source, run.roi_path)

    state = ProcessingState.read(state_path)
    saved = {(0, 1, channel) for channel in range(3)}
    if interruption == "after_channel":
        saved.add((0, 2, 0))
    elif interruption == "after_final_channel":
        saved.update((0, 2, channel) for channel in range(3))
    assert state.completed_channels(output) == saved
    assert state.completed_tiles(output) == {(0, 1)}

    # A replay of any completed channel would now produce different pixels.
    for _, position, channel in saved:
        run.raw.arrays[position][0, channel].write(np.uint16(900)).result()
    monkeypatch.setattr(ProcessingState, "complete_channel", real_complete)
    monkeypatch.setattr(run.process, "_write_checkpointed_roi_channel", real_write)
    run.decon_calls.clear()
    roi_command.process_roi(run.source, run.roi_path)

    assert len(run.decon_calls) == 6 - len(saved)
    resumed = open_position_collection(output)
    for array, expected_array in zip(resumed.arrays, expected):
        np.testing.assert_array_equal(array.read().result(), expected_array)
    state = ProcessingState.read(state_path)
    assert state.completed_tiles(output) == {(0, 1), (0, 2)}
    assert state.completed_channels(output) == {
        (0, position, channel) for position in (1, 2) for channel in range(3)
    }


@pytest.mark.integration
@pytest.mark.parametrize("known_empty", (False, True))
def test_empty_roi_channel_is_checkpointed_and_skipped(
    roi_run, monkeypatch, known_empty
):
    """Both empty-channel paths persist their zero decision before interruption."""
    run = roi_run
    run.raw.arrays[1][0, 0].write(np.uint16(100)).result()
    if known_empty:
        monkeypatch.setattr(
            run.process,
            "_known_empty_tile",
            lambda mask, time, position, channel: (
                (time, position, channel) == (0, 1, 0)
            ),
        )
    real_complete = ProcessingState.complete_channel

    def stop_after_empty(self, *args, **kwargs):
        real_complete(self, *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(ProcessingState, "complete_channel", stop_after_empty)
    options = dict(
        acquisition=run.metadata,
        roi_selection=run.roi,
        output_dir=run.output_dir,
        max_projection=False,
        create_fused_max_projection=False,
        skip_empty_below=1.0,
        resume=True,
    )
    with pytest.raises(KeyboardInterrupt):
        run.process.process_skewed(run.source, **options)
    output = run.output_dir / "sample_deskewed.ome.zarr"
    state_path = run.output_dir / "sample.processing.json"
    state = ProcessingState.read(state_path)
    assert state.completed_channels(output) == {(0, 1, 0)}
    assert state.zero_channels(output) == {(0, 1, 0)}
    assert state.completed_tiles(output) == set()

    run.raw.arrays[1][0, 0].write(np.uint16(900)).result()
    monkeypatch.setattr(ProcessingState, "complete_channel", real_complete)
    monkeypatch.setattr(run.process, "_known_empty_tile", lambda *args: False)
    run.process.process_skewed(run.source, **options)
    result = open_position_collection(output)
    expected = run.expected_tiles[0].copy()
    expected[:, 0] = 0
    np.testing.assert_array_equal(result.arrays[0].read().result(), expected)
    np.testing.assert_array_equal(
        result.arrays[1].read().result(), run.expected_tiles[1]
    )
    assert ProcessingState.read(state_path).zero_channels(output) == {(0, 1, 0)}


@pytest.mark.integration
def test_roi_resumes_existing_tile_only_checkpoints(roi_run, monkeypatch):
    """Runs created before channel checkpoints retain their completed tiles."""
    run = roi_run
    real_complete = ProcessingState.complete_tile

    def stop_after_tile(self, *args, **kwargs):
        real_complete(self, *args, **kwargs)
        self.run(args[0]).pop("completed_channels", None)
        self.save()
        raise KeyboardInterrupt

    monkeypatch.setattr(ProcessingState, "complete_tile", stop_after_tile)
    with pytest.raises(KeyboardInterrupt):
        roi_command.process_roi(run.source, run.roi_path)
    run.raw.arrays[1].write(np.uint16(900)).result()
    monkeypatch.setattr(ProcessingState, "complete_tile", real_complete)
    run.decon_calls.clear()
    roi_command.process_roi(run.source, run.roi_path)
    assert run.decon_calls == [50.0, 51.0, 52.0]
    output = open_position_collection(run.output_dir / "sample_decon_deskewed.ome.zarr")
    for array, expected in zip(output.arrays, run.expected_tiles):
        np.testing.assert_array_equal(array.read().result(), expected)


@pytest.mark.integration
def test_roi_resume_rejects_changed_tile_mapping_and_settings(roi_run):
    """Compatible shapes alone cannot justify reusing differently mapped tiles."""
    run = roi_run
    roi_command.process_roi(run.source, run.roi_path)
    state_path = run.output_dir / "sample.processing.json"
    saved_state = state_path.read_bytes()
    shifted = replace(
        run.roi,
        tile_footprints=tuple(
            {**footprint, "origin_zyx_um": [1.0, 0.0, 0.0]}
            for footprint in run.roi.tile_footprints
        ),
    )
    shifted.write(run.roi_path)
    with pytest.raises(ValueError, match="incompatible"):
        roi_command.process_roi(run.source, run.roi_path)
    assert state_path.read_bytes() == saved_state

    run.roi.write(run.roi_path)
    with pytest.raises(ValueError, match="incompatible"):
        roi_command.process_roi(run.source, run.roi_path, decon_crop_scan=4)
    assert state_path.read_bytes() == saved_state


@pytest.mark.integration
def test_roi_no_resume_overwrites_completed_channels(roi_run):
    """Explicit restart recomputes data instead of using previous checkpoints."""
    run = roi_run
    roi_command.process_roi(run.source, run.roi_path)
    run.decon_calls.clear()
    run.raw.arrays[1].write(np.uint16(900)).result()
    roi_command.process_roi(run.source, run.roi_path, resume=False)
    assert run.decon_calls == [800.0, 800.0, 800.0, 50.0, 51.0, 52.0]
    output = open_position_collection(run.output_dir / "sample_decon_deskewed.ome.zarr")
    np.testing.assert_array_equal(
        output.arrays[1].read().result(), run.expected_tiles[1]
    )
    changed = output.arrays[0].read().result()
    np.testing.assert_array_equal(changed, run.overwritten_tile)


@pytest.mark.integration
@pytest.mark.parametrize(
    "flag, expected", ((None, True), ("--resume", True), ("--no-resume", False))
)
def test_roi_cli_resumes_by_default(roi_run, flag, expected):
    """CLI resume preserves saved pixels; explicit restart replaces them."""
    run = roi_run
    args = [str(run.source)] + ([] if flag is None else [flag])
    first = CliRunner().invoke(roi_command.app, args)
    assert first.exit_code == 0, first.output
    output_path = run.output_dir / "sample_decon_deskewed.ome.zarr"
    before = open_position_collection(output_path).arrays[0].read().result()
    np.testing.assert_array_equal(before, run.expected_tiles[0])
    run.raw.arrays[1].write(np.uint16(900)).result()
    second = CliRunner().invoke(roi_command.app, args)
    assert second.exit_code == 0, second.output
    after = open_position_collection(output_path).arrays[0].read().result()
    if expected:
        np.testing.assert_array_equal(after, before)
    else:
        np.testing.assert_array_equal(after, run.overwritten_tile)
        assert not np.array_equal(after, before)


@pytest.mark.integration
def test_recreated_output_does_not_reuse_stale_checkpoints(tmp_path):
    """An absent output store cannot be skipped using surviving JSON checkpoints."""
    process = importlib.import_module("opm_processing.process")
    source = tmp_path / "sample.ome.zarr"
    output = tmp_path / "sample_decon_deskewed.ome.zarr"
    state = ProcessingState.create(tmp_path / "sample.processing.json", source)
    state.initialize_run(output, configuration={}, overwrite=True)
    state.complete_channel(output, 0, 1, 0)
    state.complete_tile(output, 0, 1)
    assert not output.exists()

    reopened = process._initialize_processing_state(
        output_dir=tmp_path,
        source_path=source,
        output_path=output,
        configuration={},
        resume=True,
        output_preexisting=False,
    )
    assert reopened.completed_channels(output) == set()
    assert reopened.completed_tiles(output) == set()


@pytest.mark.integration
def test_roi_command_fuses_selected_tiles_and_projects_pixels(roi_run, monkeypatch):
    """Run ROI processing, fusion, and projection through real persisted arrays."""
    from opm_processing.imageprocessing import tilefusion
    from opm_processing.imageprocessing.maxtilefusion import (
        regenerate_fused_max_projection,
    )
    from opm_processing.dataio.position_collection import open_image_array

    run = roi_run
    monkeypatch.setattr(tilefusion, "inspect_acquisition", lambda path: run.metadata)
    monkeypatch.setattr(
        roi_command,
        "TileFusion",
        lambda **kwargs: tilefusion.TileFusion(
            **kwargs,
            blend_pixels=(0, 0, 0),
            max_workers=1,
            multiscale_factors=(2,),
            chunk_shape_yx=(8, 8),
        ),
    )
    # The synthetic tiles are exactly co-located. Registration is a separate
    # GPU integration test; this fixture supplies that known displacement.
    monkeypatch.setattr(
        tilefusion.TileFusion,
        "register_and_score",
        staticmethod(lambda *args, **kwargs: ((0.0, 0.0, 0.0), 1.0)),
    )
    monkeypatch.setattr(
        roi_command, "regenerate_fused_max_projection", regenerate_fused_max_projection
    )
    roi_command.process_roi(run.source)
    fused = open_image_array(run.output_dir / "sample_fused.ome.zarr").read().result()
    projected = (
        open_image_array(run.output_dir / "sample_max_z_fused.ome.zarr").read().result()
    )
    expected = np.zeros_like(fused)
    mean = (run.expected_tiles[0].astype(np.float32) + run.expected_tiles[1]) / 2
    expected[..., :3, :18, :10] = mean.astype(np.uint16)
    np.testing.assert_array_equal(fused, expected)
    np.testing.assert_array_equal(projected, expected.max(axis=2, keepdims=True))
