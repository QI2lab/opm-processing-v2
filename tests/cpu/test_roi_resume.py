"""Interrupted ROI processing must retain only durably completed channels."""

import importlib
from dataclasses import replace

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
    output = run.output_dir / "sample_deskewed.ome.zarr"
    state_path = run.output_dir / "sample.processing.json"
    reference_dir = run.output_dir.parent / "reference"
    roi_command.process_roi(
        run.source, run.roi_path, output=reference_dir, deconvolve=False
    )
    reference = open_position_collection(reference_dir / "sample_deskewed.ome.zarr")
    expected = [array.read().result() for array in reference.arrays]
    real_complete = ProcessingState.complete_channel
    real_write = run.process.write_checkpointed_roi_channel
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
            run.process, "write_checkpointed_roi_channel", failing_write
        )
        expected_error = OSError
    else:
        monkeypatch.setattr(ProcessingState, "complete_channel", checkpoint)
        expected_error = KeyboardInterrupt
    with pytest.raises(expected_error):
        roi_command.process_roi(run.source, run.roi_path, deconvolve=False)

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
        run.raw.arrays[position][0, channel].write(np.uint16(901)).result()
    monkeypatch.setattr(ProcessingState, "complete_channel", real_complete)
    monkeypatch.setattr(run.process, "write_checkpointed_roi_channel", real_write)
    roi_command.process_roi(run.source, run.roi_path, deconvolve=False)

    resumed = open_position_collection(output)
    for array, expected_array in zip(resumed.arrays, expected, strict=False):
        np.testing.assert_array_equal(array.read().result(), expected_array)
    state = ProcessingState.read(state_path)
    assert state.completed_tiles(output) == {(0, 1), (0, 2)}
    assert state.completed_channels(output) == {
        (0, position, channel) for position in (1, 2) for channel in range(3)
    }


@pytest.mark.integration
def test_empty_roi_channel_is_checkpointed_and_skipped(roi_run, monkeypatch):
    """Persist an empty channel before interruption and preserve it on resume."""
    run = roi_run
    run.raw.arrays[1][0, 0].write(np.uint16(100)).result()
    real_complete = ProcessingState.complete_channel

    def stop_after_empty(self, *args, **kwargs):
        real_complete(self, *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(ProcessingState, "complete_channel", stop_after_empty)
    options = {
        "acquisition": run.metadata,
        "roi_selection": run.roi,
        "output_dir": run.output_dir,
        "max_projection": False,
        "create_fused_max_projection": False,
        "skip_empty_below": 1.0,
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

    run.raw.arrays[1][0, 0].write(np.uint16(901)).result()
    monkeypatch.setattr(ProcessingState, "complete_channel", real_complete)
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
def test_roi_no_resume_overwrites_completed_channels(roi_run):
    """Explicit restart recomputes data instead of using previous checkpoints."""
    run = roi_run
    roi_command.process_roi(run.source, run.roi_path, deconvolve=False)
    run.raw.arrays[1].write(np.uint16(901)).result()
    roi_command.process_roi(run.source, run.roi_path, resume=False, deconvolve=False)
    output = open_position_collection(run.output_dir / "sample_deskewed.ome.zarr")
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
    args = [str(run.source), "--no-deconvolve"] + ([] if flag is None else [flag])
    first = CliRunner().invoke(roi_command.app, args)
    assert first.exit_code == 0, first.output
    output_path = run.output_dir / "sample_deskewed.ome.zarr"
    before = open_position_collection(output_path).arrays[0].read().result()
    np.testing.assert_array_equal(before, run.expected_tiles[0])
    run.raw.arrays[1].write(np.uint16(901)).result()
    second = CliRunner().invoke(roi_command.app, args)
    assert second.exit_code == 0, second.output
    after = open_position_collection(output_path).arrays[0].read().result()
    if expected:
        np.testing.assert_array_equal(after, before)
    else:
        np.testing.assert_array_equal(after, run.overwritten_tile)
        assert not np.array_equal(after, before)


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


@pytest.mark.integration
def test_roi_command_fuses_selected_tiles_and_projects_pixels(tmp_path):
    """Recover a known line through real ROI processing, fusion and projection.

    Parameters
    ----------
    tmp_path
        Directory for the simulated acquisition, physical ROI and saved images.

    Notes
    -----
    Two colocated views contain the same fluorescent line on a constant
    background in three spectral channels. Full deskew gain is independently
    known to be two. Fusion must preserve that gain even at half-filled Z bins.
    No metadata, reconstruction, registration or storage boundary is mocked.
    """
    from opm_processing.dataio.acquisition import ChannelMetadata
    from opm_processing.dataio.position_collection import open_image_array
    from opm_processing.dataio.roi import PhysicalRoi
    from tests.fixtures.acquisition import (
        simulated_acquisition_metadata,
        write_simulated_acquisition,
    )

    shape = (1, 2, 3, 64, 64, 32)
    specimen = np.full(shape[-3:], 100.0)
    specimen[..., 10] += 500
    spectral_gains = np.arange(1, 4)
    raw = np.broadcast_to(
        specimen[None, None, None] * spectral_gains[None, None, :, None, None, None],
        shape,
    ).astype(np.uint16)
    metadata = replace(
        simulated_acquisition_metadata(tmp_path / "sample.ome.zarr", "mirror", shape),
        channels=tuple(
            ChannelMetadata(index, f"{wavelength}nm", wavelength, 10, None)
            for index, wavelength in enumerate((488, 561, 637))
        ),
        array_paths=("0/0", "1/0"),
        stage_positions_zxy=((0, 0, 0), (0, 0, 0)),
        scan_axis_step_um=1.0,
        pixel_size_um=1.0,
        camera_offset=0.0,
        camera_conversion=1.0,
    )
    write_simulated_acquisition(metadata, raw)
    roi = PhysicalRoi(
        bounds_yx_um=(56.0, 60.0, 4.0, 20.0),
        source_path=tmp_path / "sample_max_z_fused.ome.zarr",
        grid_origin_yx_um=(0.0, 0.0),
        pixel_size_yx_um=(1.0, 1.0),
        position_indices=(0, 1),
        tile_footprints=tuple(
            {
                "time_index": 0,
                "position_index": position,
                "origin_zyx_um": [0.0, 0.0, 0.0],
                "bounds_yx_um": [0.0, 120.0, 0.0, 32.0],
            }
            for position in (0, 1)
        ),
    )
    roi_path = roi.write(tmp_path / "sample_roi.json")
    roi_command.process_roi(
        metadata.path,
        roi_path,
        deconvolve=False,
        flatfield_correction=False,
        save_float32=True,
    )
    output_dir = tmp_path / "sample_roi"
    fused = open_image_array(output_dir / "sample_fused.ome.zarr").read().result()
    projected = (
        open_image_array(output_dir / "sample_max_z_fused.ome.zarr").read().result()
    )
    expected = np.zeros_like(fused)
    expected[0, :, :16, :4, :16] = (
        2 * spectral_gains[:, None, None, None] * specimen[0, 0, 4:20]
    )
    np.testing.assert_allclose(fused, expected, rtol=1e-6)
    np.testing.assert_allclose(
        projected, expected.max(axis=2, keepdims=True), rtol=1e-6
    )
