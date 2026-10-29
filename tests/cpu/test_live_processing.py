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
    open_live_acquisition,
    process,
)
from tests.fixtures.live import manifest_document, write_manifest


@pytest.mark.unit
@pytest.mark.parametrize("event", ["canceled", "errored"])
def test_live_wait_stops_before_processing_on_acquisition_failure(
    tmp_path, live_acquisition_factory, event
):
    """Abort live waiting before processing a canceled or failed acquisition."""
    dataset = live_acquisition_factory(completed=False)
    dataset.illumination_path.unlink()
    dataset.log.write_text(
        json.dumps({"event": event, "acquisition_id": dataset.manifest.acquisition_id})
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "processed"
    result = CliRunner().invoke(
        app,
        [
            str(tmp_path),
            "--live-wait",
            "--flatfield-correction",
            "--output",
            str(destination),
        ],
    )
    assert isinstance(result.exception, RuntimeError)
    assert event in str(result.exception)
    assert not destination.exists()


@pytest.mark.unit
def test_live_wait_accepts_single_position_root_image(
    tmp_path, current_single_position_mirror_timelapse
):
    """Discover completed live data stored directly as a root NGFF image."""
    path = current_single_position_mirror_timelapse
    original = inspect_acquisition(path)
    document = manifest_document(path, timepoints=3)
    document["index_sizes"] = {"t": 3, "p": 1, "c": 1, "z": 4}
    document["channels"] = document["channels"][:1]
    document["mode"] = "mirror"
    manifest = write_manifest(path, document=document)
    acquisition, actual_manifest, _log = open_live_acquisition(tmp_path)
    assert acquisition.path == path
    assert acquisition.shape == original.shape
    assert actual_manifest == manifest
    assert ZarrTileReadiness(acquisition).ready_tiles() == {(0, 0), (1, 0), (2, 0)}


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
@pytest.mark.parametrize("resume", [False, True], ids=["poll", "resume"])
def test_live_iterator_waits_for_unprocessed_tiles_then_stops(
    live_acquisition_factory, resume
):
    """Observe real chunk arrivals, skip durable tiles and stop on completion."""
    dataset = live_acquisition_factory(
        shape=(1, 2 if resume else 1, 2, 3, 4, 5),
        completed=True,
    )
    readiness = ZarrTileReadiness(
        dataset.manifest.apply(inspect_acquisition(dataset.path))
    )
    sleeps = []

    def acquire(seconds=None):
        """Publish native TensorStore chunks when the iterator polls acquisition."""
        if seconds is not None:
            sleeps.append(seconds)
        for array in dataset.collection.arrays:
            array.write(np.ones(array.shape, np.uint16)).result()

    if resume:
        acquire()
    tiles = list(
        iter_live_tiles(
            readiness,
            dataset.manifest,
            dataset.log,
            completed_tiles={(0, 0)} if resume else set(),
            poll_interval=30.0,
            sleeper=acquire,
        )
    )
    assert tiles == [(0, 1)] if resume else tiles == [(0, 0)]
    assert sleeps == ([] if resume else [30.0])


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
@pytest.mark.parametrize("live_mode", ["live", "live-wait"])
@pytest.mark.parametrize(
    ("save_float32", "expected_dtype"),
    ((False, np.dtype(np.uint16)), (True, np.dtype(np.float32))),
)
def test_live_process_matches_direct_deskew_without_estimating(
    tmp_path,
    opm_v2_live_acquisition_factory,
    save_float32,
    expected_dtype,
    simulated_camera,
    relocate_acquisition,
    live_mode,
) -> None:
    """Process a complete live acquisition using only the supplied illumination."""
    observed = simulated_camera((4, 8, 5), conversion=0.24).raw
    raw = observed[None, None, None, ::-1].copy()
    dataset = opm_v2_live_acquisition_factory(
        raw, name="live", illumination=np.full((1, 8, 5), 2.0, np.float32)
    )
    root = relocate_acquisition(tmp_path, copy=save_float32)
    assert dataset.path.exists() == save_float32
    illumination_path = root / dataset.illumination_path.name
    expected_float = orthogonal_deskew(
        np.maximum(observed.astype(np.float32) - 100, 0) * 0.24 / 2,
        theta=30.0,
        distance=0.4,
        pixel_size=0.115,
        downsample_factor=1,
    )
    args = [
        str(root),
        "--no-max-projection",
        "--no-create-fused-max-projection",
        "--z-downsample-level",
        "1",
    ]
    if live_mode == "live":
        args.extend(("--live", str(illumination_path)))
    else:
        # Offline live-wait finds the supplied field in the current directory.
        field = root / "live_flatfield.ome.tif"
        processing = importlib.import_module("opm_processing.process")
        processing.write_flatfield(field, np.full((1, 1, 8, 5), 2.0, np.float32), 0.115)
        args.extend(("--live-wait", "--flatfield-correction"))
    if save_float32:
        args.append("--save-float32")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output

    output = open_position_collection(root / "live_deskewed.ome.zarr")
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
    assert "opm_v2" not in output.attributes


@pytest.mark.unit
def test_live_occupancy_filter_preserves_sample_signal_and_writes_zero(
    tmp_path,
    live_acquisition_factory,
) -> None:
    """Ignore a persistent hot pixel while processing a channel with real signal."""
    raw = np.full((2, 4, 8, 5), 100, dtype=np.uint16)
    raw[0, :, 0, 0] = 200  # One persistent detector hot pixel.
    raw[0, 0, 0, 1:5] = 104  # Sub-threshold signal before illumination correction.
    raw[1] = 200  # A uniformly occupied fluorescent volume.
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
        np.full((4, 8, 5), 100 * 0.24, dtype=np.float32),
        theta=30.0,
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


@pytest.mark.unit
@pytest.mark.parametrize("action", ["resume", "overwrite"])
def test_offline_resume_skips_durably_completed_tile_and_default_overwrites(
    tmp_path,
    monkeypatch,
    acquisition_factory,
    action,
) -> None:
    """Resume preserves completed pixels; explicit overwrite recomputes them."""
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
        """Interrupt after a durable write to exercise restart behavior."""
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
        orthogonal_deskew(
            np.full((4, 8, 5), 10, dtype=np.float32),
            theta=30.0,
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
    result = CliRunner().invoke(
        app,
        [
            str(data_path),
            "--output",
            str(tmp_path),
            "--no-max-projection",
            "--no-create-fused-max-projection",
            "--z-downsample-level",
            "1",
        ],
        input=f"{action}\n",
    )
    assert result.exit_code == 0, result.exception
    assert "Resume / Overwrite / Cancel" in result.output
    resumed = open_position_collection(output_path)
    np.testing.assert_array_equal(
        resumed.arrays[0][0, 0].read().result(),
        completed_tile
        if action == "resume"
        else orthogonal_deskew(
            np.full((4, 8, 5), 401, dtype=np.float32),
            theta=30.0,
            distance=0.4,
            pixel_size=0.115,
            downsample_factor=1,
        ).astype(np.uint16),
    )
    np.testing.assert_array_equal(
        resumed.arrays[1][0, 0].read().result(),
        orthogonal_deskew(
            np.full((4, 8, 5), 20, dtype=np.float32),
            theta=30.0,
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
        orthogonal_deskew(
            np.full((4, 8, 5), 401, dtype=np.float32),
            theta=30.0,
            distance=0.4,
            pixel_size=0.115,
            downsample_factor=1,
        ).astype(np.uint16),
    )


@pytest.mark.unit
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
        """Interrupt after a durable write to exercise restart behavior."""
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
    # The first tile is exactly the camera offset, hence zero calibrated photons.
    np.testing.assert_array_equal(first_output.arrays[0].read().result(), 0)
    np.testing.assert_array_equal(first_max_output.arrays[0].read().result(), 0)
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
        0,
    )
    np.testing.assert_array_equal(
        resumed_max_output.arrays[0].read().result(),
        0,
    )
    assert np.all(resumed_output.arrays[1].read().result() != 999)
    expected_tile = orthogonal_deskew(
        np.full((4, 8, 5), 11 * 0.24, dtype=np.float32),
        theta=30.0,
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
