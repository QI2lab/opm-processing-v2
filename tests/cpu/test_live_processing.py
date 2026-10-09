"""Coverage for on-the-fly OPM tile discovery and illumination loading."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from opm_processing.dataio.acquisition import (
    inspect_acquisition,
)
from opm_processing.dataio.live import (
    LiveManifest,
    ZarrTileReadiness,
    iter_live_tiles,
    read_lifecycle_event,
)
from opm_processing.dataio.position_collection import (
    open_position_collection,
)
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)
from opm_processing.imageprocessing.opmtools import orthogonal_deskew
from opm_processing.process import (
    app,
    is_empty_tile,
    process,
)
from tests.fixtures.live import manifest_document
from tests.reference.deskew import constant_deskew


@pytest.mark.unit
def test_manifest_overlays_metadata_available_before_frame_metadata(
    tmp_path, live_acquisition_factory
) -> None:
    """Use the immutable plan when per-frame metadata has not been flushed."""
    dataset = live_acquisition_factory(completed=False)
    _collection, manifest = dataset.collection, dataset.manifest
    data_path = dataset.path
    acquisition = manifest.apply(inspect_acquisition(data_path))

    assert acquisition.mode == "stage"
    assert acquisition.camera_offset == pytest.approx(100.0)
    assert acquisition.camera_conversion == pytest.approx(0.24)
    assert acquisition.stage_positions_zxy == ((30.0, 100.0, 200.0),)
    assert acquisition.channel_names == ("488nm", "561nm")


@pytest.mark.unit
def test_chunk_presence_marks_only_a_fully_written_tile_ready(
    tmp_path, live_acquisition_factory
) -> None:
    """Require every C/Z/Y/X chunk before publishing one T/P tile."""
    dataset = live_acquisition_factory(
        name="chunks",
        stage_positions=((0.0, 0.0, 0.0),),
        chunks=(1, 1, 1, 2, 3),
        completed=False,
    )
    collection, manifest = dataset.collection, dataset.manifest
    data_path = dataset.path
    acquisition = manifest.apply(inspect_acquisition(data_path))
    readiness = ZarrTileReadiness(acquisition)

    assert readiness.ready_tiles() == set()
    collection.arrays[0][0, :, :2].write(
        np.ones((2, 2, 4, 5), dtype=np.uint16)
    ).result()
    assert readiness.ready_tiles() == set()

    collection.arrays[0][0, :, 2].write(np.ones((2, 4, 5), dtype=np.uint16)).result()
    assert readiness.ready_tiles() == {(0, 0)}


@pytest.mark.unit
def test_live_iterator_polls_then_stops_at_completed_log(tmp_path) -> None:
    """Poll only while caught up and stop after all completed tiles are yielded."""
    data_path = tmp_path / "iterator.ome.zarr"
    manifest_path = tmp_path / "iterator.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_document(data_path, timepoints=1)), encoding="utf-8"
    )
    manifest = LiveManifest.read(manifest_path)
    log_path = tmp_path / "iterator.log.jsonl"
    log_path.write_text(
        json.dumps({"event": "completed", "acquisition_id": manifest.acquisition_id})
        + "\n",
        encoding="utf-8",
    )

    class Readiness:
        calls = 0

        def ready_tiles(self):
            self.calls += 1
            return set() if self.calls == 1 else {(0, 0)}

    sleeps = []
    tiles = list(
        iter_live_tiles(
            Readiness(),
            manifest,
            log_path,
            poll_interval=30.0,
            sleeper=sleeps.append,
        )
    )

    assert tiles == [(0, 0)]
    assert sleeps == [30.0]


@pytest.mark.unit
def test_live_iterator_skips_tiles_completed_before_restart(tmp_path) -> None:
    """Seed iterator state with processed tiles discovered from existing output."""
    data_path = tmp_path / "resume.ome.zarr"
    manifest_path = tmp_path / "resume.manifest.json"
    document = manifest_document(data_path, timepoints=1)
    document["index_sizes"]["p"] = 2
    document["stage_positions_zxy"] = [
        [30.0, 100.0, 200.0],
        [30.0, 100.0, 202.0],
    ]
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    manifest = LiveManifest.read(manifest_path)
    log_path = tmp_path / "resume.log.jsonl"
    log_path.write_text(
        json.dumps({"event": "completed", "acquisition_id": manifest.acquisition_id})
        + "\n",
        encoding="utf-8",
    )

    class Readiness:
        def ready_tiles(self):
            return {(0, 0), (0, 1)}

    tiles = list(
        iter_live_tiles(
            Readiness(),
            manifest,
            log_path,
            completed_tiles={(0, 0)},
        )
    )

    assert tiles == [(0, 1)]


@pytest.mark.unit
def test_lifecycle_reader_ignores_an_incomplete_last_record(tmp_path) -> None:
    """Do not parse a JSONL record while the controller is appending it."""
    log_path = tmp_path / "sample.log.jsonl"
    log_path.write_text(
        '{"event":"started","acquisition_id":"id"}\n{"event":"compl',
        encoding="utf-8",
    )
    assert read_lifecycle_event(log_path, "id") == "started"


@pytest.mark.unit
def test_empty_tile_detection_uses_global_occupancy() -> None:
    """Use global occupancy to ignore sparse noise and hot pixels."""
    stack = np.zeros((5, 4, 4), dtype=np.float32)
    stack[:, 0, 0] = 10.0
    assert is_empty_tile(
        stack,
        threshold=2.0,
        min_signal_fraction=0.1,
    )
    stack[-1, 0, :4] = 2.0
    assert not is_empty_tile(
        stack,
        threshold=2.0,
        min_signal_fraction=0.1,
    )
    assert not is_empty_tile(
        stack,
        threshold=None,
        min_signal_fraction=0.1,
    )

    sparse_artifacts = np.zeros((100, 20, 20), dtype=np.float32)
    sparse_artifacts[:10, :10, :10] = 10.0
    assert is_empty_tile(
        sparse_artifacts,
        threshold=2.0,
        min_signal_fraction=0.05,
    )
    sparse_artifacts[:20, :10, :10] = 10.0
    assert not is_empty_tile(
        sparse_artifacts,
        threshold=2.0,
        min_signal_fraction=0.05,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("save_float32", "expected_dtype"),
    ((False, np.dtype(np.uint16)), (True, np.dtype(np.float32))),
)
def test_live_process_matches_direct_deskew_without_estimating(
    tmp_path, live_acquisition_factory, save_float32, expected_dtype
) -> None:
    """Process a complete live acquisition using only the supplied illumination."""
    raw = np.full((1, 1, 1, 4, 8, 5), 137, np.uint16)
    dataset = live_acquisition_factory(
        raw, name="live", illumination=np.full((1, 8, 5), 2.0, np.float32)
    )
    illumination_path = dataset.illumination_path
    expected_float = constant_deskew((4, 8, 5), 37 * 0.24 / 2, downsample_factor=1)
    args = [
        str(tmp_path),
        "--live",
        str(illumination_path),
        "--no-max-projection",
        "--no-create-fused-max-projection",
        "--z-downsample-level",
        "1",
    ]
    if save_float32:
        args.append("--save-float32")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output

    output = open_position_collection(tmp_path / "live_deskewed.ome.zarr")
    actual = output.arrays[0][0, 0].read().result()
    expected = (
        expected_float
        if save_float32
        else np.clip(expected_float, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    )
    assert np.dtype(actual.dtype) == expected_dtype
    # Bound float32 calibration and the four interpolation products against
    # analytical intensity; uint16 values still compare exactly at this scale.
    np.testing.assert_allclose(actual, expected, rtol=4 * np.finfo(np.float32).eps)
    if save_float32:
        positive = actual[actual > 0]
        assert positive.size > 0
        assert np.any(positive != np.floor(positive))
    assert not any(key.startswith("opm_") for key in output.attributes)


@pytest.mark.integration
def test_live_occupancy_filter_preserves_sample_signal_and_writes_zero(
    tmp_path,
    live_acquisition_factory,
) -> None:
    """Ignore a persistent hot pixel while processing a channel with real signal."""
    raw = np.full((2, 4, 8, 5), 100, dtype=np.uint16)
    raw[0, :, 0, 0] = 200  # One persistent detector hot pixel.
    raw[0, 0, 0, 1:5] = 104  # Sub-threshold signal before illumination correction.
    raw[1, 2] = 200  # A sample-like occupied detector frame.
    illumination = np.ones((2, 8, 5), dtype=np.float32)
    illumination[0] = (
        0.1  # Amplifying detector artifacts must not rescue an empty channel.
    )
    dataset = live_acquisition_factory(
        raw[None, None], name="live_empty", illumination=illumination
    )
    illumination_path = dataset.illumination_path
    process(
        root_path=tmp_path,
        live=illumination_path,
        deconvolve=False,
        skip_empty_below=2.0,
        skip_empty_min_signal_fraction=0.1,
        max_projection=True,
        create_fused_max_projection=False,
        z_downsample_level=1,
    )

    output = open_position_collection(tmp_path / "live_empty_deskewed.ome.zarr")
    max_output = open_position_collection(
        tmp_path / "live_empty_max_z_deskewed.ome.zarr"
    )
    assert np.count_nonzero(output.arrays[0][0, 0].read().result()) == 0
    assert np.count_nonzero(max_output.arrays[0][0, 0].read().result()) == 0
    assert np.count_nonzero(output.arrays[0][0, 1].read().result()) > 0
    expected = orthogonal_deskew(
        (np.flip(raw[1], axis=0).astype(np.float32) - 100) * np.float32(0.24),
        distance=0.4,
        pixel_size=0.115,
        downsample_factor=1,
    ).astype(np.uint16)
    np.testing.assert_array_equal(output.arrays[0][0, 1].read().result(), expected)
    state = ProcessingState.read(processing_state_path(tmp_path, "live_empty"))
    assert state.zero_channels(tmp_path / "live_empty_deskewed.ome.zarr") == {(0, 0, 0)}
    assert len(max_output.multiscale_factors_yx) > 1
    for factor, level_arrays in zip(
        max_output.multiscale_factors_yx, max_output.multiscale_arrays, strict=False
    ):
        assert np.count_nonzero(level_arrays[0][0, 0].read().result()) == 0
        np.testing.assert_array_equal(
            level_arrays[0][0, 1].read().result(),
            expected.max(axis=0, keepdims=True)[..., ::factor, ::factor],
        )
    assert not any(key.startswith("opm_") for key in output.attributes)


@pytest.mark.integration
def test_offline_resume_skips_durably_completed_tile_and_default_overwrites(
    tmp_path,
    monkeypatch,
    acquisition_factory,
) -> None:
    """Checkpoint after writes, resume the next tile, and overwrite by default."""
    process_module = importlib.import_module("opm_processing.process")
    pixels = np.empty((1, 2, 1, 4, 8, 5), np.uint16)
    pixels[:, 0] = 110
    pixels[:, 1] = 120
    dataset = acquisition_factory(
        pixels,
        name="offline_resume",
        mode="stage",
        camera_conversion=1.0,
        stage_positions_zxy=((30.0, 100.0, 200.0), (30.0, 100.0, 202.0)),
    )
    data_path, collection, metadata = dataset.path, dataset.collection, dataset.metadata
    real_complete_tile = process_module.ProcessingState.complete_tile
    completion_count = 0

    def interrupt_after_first(self, *args, **kwargs):
        nonlocal completion_count
        real_complete_tile(self, *args, **kwargs)
        completion_count += 1
        if completion_count == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(
        process_module.ProcessingState,
        "complete_tile",
        interrupt_after_first,
    )
    with pytest.raises(KeyboardInterrupt):
        process_module.process_skewed(
            data_path,
            acquisition=metadata,
            output_dir=tmp_path,
            max_projection=False,
            create_fused_max_projection=False,
            z_downsample_level=1,
        )

    output_path = tmp_path / "offline_resume_deskewed.ome.zarr"
    interrupted = open_position_collection(output_path)
    completed_tile = interrupted.arrays[0][0, 0].read().result()
    np.testing.assert_array_equal(
        completed_tile,
        constant_deskew(
            (4, 8, 5),
            10,
            distance=0.4,
            pixel_size=0.115,
            downsample_factor=1,
        ).astype(np.uint16),
    )
    collection.arrays[0][0].write(np.full((1, 4, 8, 5), 501, dtype=np.uint16)).result()

    monkeypatch.setattr(
        process_module.ProcessingState,
        "complete_tile",
        real_complete_tile,
    )
    process_module.process_skewed(
        data_path,
        acquisition=metadata,
        output_dir=tmp_path,
        max_projection=False,
        create_fused_max_projection=False,
        z_downsample_level=1,
        resume=True,
    )
    resumed = open_position_collection(output_path)
    np.testing.assert_array_equal(
        resumed.arrays[0][0, 0].read().result(),
        completed_tile,
    )
    np.testing.assert_array_equal(
        resumed.arrays[1][0, 0].read().result(),
        constant_deskew(
            (4, 8, 5),
            20,
            distance=0.4,
            pixel_size=0.115,
            downsample_factor=1,
        ).astype(np.uint16),
    )

    process_module.process_skewed(
        data_path,
        acquisition=metadata,
        output_dir=tmp_path,
        max_projection=False,
        create_fused_max_projection=False,
        z_downsample_level=1,
    )
    overwritten = open_position_collection(output_path)
    np.testing.assert_array_equal(
        overwritten.arrays[0][0, 0].read().result(),
        constant_deskew(
            (4, 8, 5),
            401,
            distance=0.4,
            pixel_size=0.115,
            downsample_factor=1,
        ).astype(np.uint16),
    )


@pytest.mark.integration
def test_live_processing_resumes_completed_output_tiles(
    tmp_path, monkeypatch, live_acquisition_factory
) -> None:
    """Preserve completed tiles and safely redo a partially written next tile."""
    pixels = np.empty((1, 2, 2, 4, 8, 5), np.uint16)
    pixels[:, 0] = 100
    pixels[:, 1] = 111
    dataset = live_acquisition_factory(pixels, name="live_resume")
    illumination_path = dataset.illumination_path
    output_path = tmp_path / "live_resume_deskewed.ome.zarr"
    max_output_path = tmp_path / "live_resume_max_z_deskewed.ome.zarr"

    real_complete = ProcessingState.complete_tile

    def interrupt_after_first(self, *args, **kwargs):
        real_complete(self, *args, **kwargs)
        # A live tile is durable only after both the volume and projection
        # checkpoints are committed. Interrupt after the second output.
        if args[0] == max_output_path:
            raise KeyboardInterrupt

    monkeypatch.setattr(ProcessingState, "complete_tile", interrupt_after_first)
    with pytest.raises(KeyboardInterrupt):
        process(
            root_path=tmp_path,
            live=illumination_path,
            max_projection=True,
            create_fused_max_projection=False,
            z_downsample_level=1,
        )

    first_output = open_position_collection(output_path)
    first_max_output = open_position_collection(max_output_path)
    completed_before_restart = first_output.arrays[0].read().result()
    completed_max_before_restart = first_max_output.arrays[0].read().result()
    first_output.arrays[1][0, 0, 0, 0, 0].write(np.uint16(999)).result()

    dataset.collection.arrays[0].write(np.uint16(901)).result()
    monkeypatch.setattr(ProcessingState, "complete_tile", real_complete)
    process(
        root_path=tmp_path,
        live=illumination_path,
        max_projection=True,
        create_fused_max_projection=False,
        z_downsample_level=1,
    )

    resumed_output = open_position_collection(output_path)
    resumed_max_output = open_position_collection(max_output_path)
    np.testing.assert_array_equal(
        resumed_output.arrays[0].read().result(),
        completed_before_restart,
    )
    np.testing.assert_array_equal(
        resumed_max_output.arrays[0].read().result(),
        completed_max_before_restart,
    )
    assert np.all(resumed_output.arrays[1].read().result() != 999)
    expected_tile = constant_deskew(
        (4, 8, 5),
        11 * 0.24,
        distance=0.4,
        pixel_size=0.115,
        downsample_factor=1,
    ).astype(np.uint16)
    for channel in range(2):
        np.testing.assert_array_equal(
            resumed_output.arrays[1][0, channel].read().result(), expected_tile
        )
        np.testing.assert_array_equal(
            resumed_max_output.arrays[1][0, channel].read().result(),
            expected_tile.max(axis=0, keepdims=True),
        )
    acquisition_records = [
        json.loads(line)
        for line in dataset.log.read_text(encoding="utf-8").splitlines()
    ]
    assert acquisition_records == [
        {"event": "completed", "acquisition_id": "synthetic-acquisition"}
    ]
    state = ProcessingState.read(processing_state_path(tmp_path, "live_resume"))
    assert state.completed_tiles(output_path) == {(0, 0), (0, 1)}
    assert state.completed_tiles(max_output_path) == {(0, 0), (0, 1)}
