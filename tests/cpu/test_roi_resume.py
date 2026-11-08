"""Interrupted ROI processing must retain only durably completed channels."""

import importlib
from copy import copy

import numpy as np
import pytest
import tensorstore as ts

from opm_processing import process_roi as roi_command
from opm_processing.dataio.position_collection import (
    open_position_collection,
)
from opm_processing.dataio.processing_state import ProcessingState
from tests.simulation.camera import digitize


@pytest.mark.integration
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
    real_write = run.process.write_checkpointed_roi_channel
    stop_key = (0, 2, 2) if interruption == "after_final_channel" else (0, 2, 0)

    def checkpoint(self, path, time, position, channel, **kwargs):
        """Record a channel checkpoint only after the associated pixels are durable."""
        key = (time, position, channel)
        if key == stop_key and interruption == "before_checkpoint":
            raise KeyboardInterrupt
        real_complete(self, path, time, position, channel, **kwargs)
        if key == stop_key:
            raise KeyboardInterrupt

    def failing_write(target, value, state, path, key, **kwargs):
        """Inject a partial write failure before the channel checkpoint."""
        if key == stop_key:
            # Leave a partial channel and fail the write future before checkpointing.
            run.process.write_processed_channel(target[:1], value[:1])
            readonly = ts.open(target.spec(), read=True, write=False).result()
            return real_write(readonly, value, state, path, key, **kwargs)
        return real_write(target, value, state, path, key, **kwargs)

    if interruption == "during_write":
        monkeypatch.setattr(
            run.process, "write_checkpointed_roi_channel", failing_write
        )
        expected_error = ValueError
    else:
        monkeypatch.setattr(ProcessingState, "complete_channel", checkpoint)
        expected_error = KeyboardInterrupt
    with pytest.raises(expected_error):
        roi_command.process_roi(
            run.source, run.roi_path, registration_channel=1, deconvolve=False
        )

    state = ProcessingState.read(state_path)
    saved = {(0, 1, channel) for channel in range(3)}
    if interruption == "after_channel":
        saved.add((0, 2, 0))
    elif interruption == "after_final_channel":
        saved.update((0, 2, channel) for channel in range(3))
    assert state.completed_channels(output) == saved
    assert state.completed_tiles(output) == {(0, 1)}

    partial = open_position_collection(output)
    durable = {
        (position, channel): partial.arrays[position - 1][0, channel].read().result()
        for _, position, channel in saved
    }
    # A replay of any completed channel would now produce different pixels.
    changed = digitize(2 * run.truth.means, run.truth.microscope.camera, seed=73)
    assert changed.max() < np.iinfo(np.uint16).max
    for _, position, channel in saved:
        run.raw_arrays[position][0, channel].write(
            changed[0, position, channel]
        ).result()
    monkeypatch.setattr(ProcessingState, "complete_channel", real_complete)
    monkeypatch.setattr(run.process, "write_checkpointed_roi_channel", real_write)
    roi_command.process_roi(
        run.source, run.roi_path, registration_channel=1, deconvolve=False
    )

    resumed = open_position_collection(output)
    for index, array in enumerate(resumed.arrays):
        pixels = array.read().result()
        assert_roi_point_accuracy(
            pixels,
            run.truth,
            resumed.spatial_origins_zyx_um[index],
            resumed.voxel_size_um,
            positions=(index + 1,),
        )
        for _, position, channel in saved:
            if position == index + 1:
                np.testing.assert_array_equal(
                    pixels[0, channel], durable[position, channel]
                )
    state = ProcessingState.read(state_path)
    assert state.completed_tiles(output) == {(0, 1), (0, 2)}
    assert state.completed_channels(output) == {
        (0, position, channel) for position in (1, 2) for channel in range(3)
    }


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


@pytest.mark.integration
@pytest.mark.parametrize("resume", (None, True, False))
def test_roi_processing_resumes_by_default(roi_run, resume, assert_roi_point_accuracy):
    """Default resume preserves saved photons; an explicit restart recalibrates input."""
    run = roi_run
    options = {} if resume is None else {"resume": resume}
    roi_command.process_roi(
        run.source, run.roi_path, registration_channel=1, deconvolve=False, **options
    )
    output_path = run.output_dir / "sample_deskewed.ome.zarr"
    before = open_position_collection(output_path).arrays[0].read().result()
    collection = open_position_collection(output_path)
    unchanged = collection.arrays[1].read().result()
    for index, array in enumerate(collection.arrays):
        assert_roi_point_accuracy(
            array.read().result(),
            run.truth,
            collection.spatial_origins_zyx_um[index],
            collection.voxel_size_um,
            positions=(index + 1,),
        )
    changed = copy(run.truth)
    changed.means = run.truth.means.copy()
    changed.means[:, 1] *= 2
    changed.pixels = run.truth.pixels.copy()
    changed.pixels[:, 1] = digitize(
        changed.means[:, 1], changed.microscope.camera, seed=73
    )
    assert changed.pixels.max() < np.iinfo(np.uint16).max
    run.raw_arrays[1].write(changed.pixels[:, 1]).result()
    roi_command.process_roi(
        run.source, run.roi_path, registration_channel=1, deconvolve=False, **options
    )
    after = open_position_collection(output_path).arrays[0].read().result()
    np.testing.assert_array_equal(
        open_position_collection(output_path).arrays[1].read().result(),
        unchanged,
    )
    if resume is not False:
        np.testing.assert_array_equal(after, before)
    else:
        assert not np.array_equal(after, before)
    collection = open_position_collection(output_path)
    for index, array in enumerate(collection.arrays):
        assert_roi_point_accuracy(
            array.read().result(),
            changed if resume is False else run.truth,
            collection.spatial_origins_zyx_um[index],
            collection.voxel_size_um,
            positions=(index + 1,),
        )


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
def test_roi_command_fuses_selected_tiles_and_projects_pixels(
    roi_run, assert_roi_point_accuracy
):
    """Recover the original optical emitters through ROI fusion and max projection."""
    from yaozarrs import open_group

    from opm_processing.dataio.position_collection import open_image_array

    run = roi_run
    roi_command.process_roi(
        run.source,
        run.roi_path,
        registration_channel=1,
        deconvolve=False,
        flatfield_correction=False,
        save_float32=True,
    )
    fused_path = run.output_dir / "sample_fused.ome.zarr"
    fused = open_image_array(fused_path).read().result()
    projected = (
        open_image_array(run.output_dir / "sample_max_z_fused.ome.zarr").read().result()
    )
    dataset = open_group(fused_path).ome_metadata().multiscales[0].datasets[0]
    state = ProcessingState.read(run.output_dir / "sample.processing.json")
    source_tiles = run.output_dir / "sample_deskewed.ome.zarr"
    records = state.roi_series(source_tiles)
    for field in ("skewed_bounds_sx", "deskew_crop_yx"):
        assert all(record[field] == records[0][field] for record in records)
    # The prescribed source plans are coincident and every view has identical
    # crop/support. Correct registration must retain that known placement,
    # proving equal contributor weights for the original-dose mean below.
    expected_origin = (
        round(run.metadata.pixel_size_um / 2, 3),
        run.roi.bounds_yx_um[0],
        run.roi.bounds_yx_um[2],
    )
    registration = state.registration(source_tiles)
    np.testing.assert_allclose(
        [tile["origin_zyx_um"] for tile in registration["tiles"]],
        np.broadcast_to(expected_origin, (len(records), 3)),
        rtol=0,
        atol=1e-6,
    )
    assert_roi_point_accuracy(
        fused,
        run.truth,
        dataset.translation_transform.translation[-3:],
        dataset.scale_transform.scale[-3:],
        positions=run.roi.position_indices,
        fused=True,
    )
    # Projection is a separate numerical reduction, checked with NumPy after
    # the stored fused volume passes the original-object accuracy requirement.
    np.testing.assert_array_equal(projected, np.max(fused, axis=2, keepdims=True))
    assert state.completed_tiles(run.output_dir / "sample_deskewed.ome.zarr") == {
        (0, 1),
        (0, 2),
    }
