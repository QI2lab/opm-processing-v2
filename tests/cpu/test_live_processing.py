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
from opm_processing.imageprocessing.coordinates import STAGE_COORDINATE_VERSION
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
def test_live_processing_preserves_original_optical_emitters(
    tmp_path,
    opm_v2_live_acquisition_factory,
    save_float32,
    expected_dtype,
    roi_camera_scene,
    assert_roi_point_accuracy,
    deskew_uniform_expectation,
    relocate_acquisition,
    live_mode,
) -> None:
    """Process a complete live acquisition using only the supplied illumination."""
    scene = roi_camera_scene()
    raw = scene.pixels[..., ::-1, :, :].copy()
    dataset = opm_v2_live_acquisition_factory(
        raw,
        name="live",
        illumination=np.full((1, 32, 32), 2.0, np.float32),
        camera_conversion=1.0,
    )
    root = relocate_acquisition(tmp_path, copy=save_float32)
    assert dataset.path.exists() == save_float32
    illumination_path = root / dataset.illumination_path.name
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
        processing.write_flatfield(
            field, np.full((1, 1, 32, 32), 2.0, np.float32), 0.115
        )
        args.extend(("--live-wait", "--flatfield-correction"))
    if save_float32:
        args.append("--save-float32")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output

    output = open_position_collection(root / "live_deskewed.ome.zarr")
    actual = output.arrays[0][0, 0].read().result()
    assert np.dtype(actual.dtype) == expected_dtype
    assert actual.shape == deskew_uniform_expectation(raw.shape[-3:], 1).shape
    stage_anchor = np.asarray((-30.0, -100.0, 200.0))
    assert_roi_point_accuracy(
        actual[None, None],
        scene,
        stage_anchor + output.spatial_origins_zyx_um[0],
        output.voxel_size_um,
        raw_origin=(-30.0, -102.8, 200.0),
        illumination=2.0,
    )
    assert output.spatial_origins_zyx_um[0] == (0.0, -2.8, 0.0)
    state = ProcessingState.read(root / "live.processing.json")
    assert state.completed_tiles(root / "live_deskewed.ome.zarr") == {(0, 0)}
    if save_float32:
        positive = actual[actual > 0]
        assert positive.size > 0
        assert np.any(positive != np.floor(positive))
    assert "opm_v2" not in output.attributes


@pytest.mark.integration
@pytest.mark.parametrize("signal_field", (1.0, 2.0))
@pytest.mark.parametrize("save_float32", (False, True))
def test_live_occupancy_filter_preserves_sample_signal_and_writes_zero(
    tmp_path, live_occupancy_case, save_float32, assert_uniform_photon_accuracy
):
    """Preserve a known optical sample while rejecting physical detector artifacts."""
    case = live_occupancy_case
    process(
        root_path=tmp_path,
        live=case.dataset.illumination_path,
        deconvolve=False,
        save_float32=save_float32,
        skip_empty_below=case.threshold,
        skip_empty_min_signal_fraction=0.1,
        max_projection=True,
        create_fused_max_projection=False,
        z_downsample_level=1,
    )
    path = tmp_path / "live_empty_deskewed.ome.zarr"
    output = open_position_collection(path)
    maximum = open_position_collection(tmp_path / "live_empty_max_z_deskewed.ome.zarr")
    data = output.arrays[0].read().result()
    assert data.dtype == (np.float32 if save_float32 else np.uint16)
    np.testing.assert_array_equal(data[:, 0], 0)
    assert_uniform_photon_accuracy(data, case.scene, channels=(1,))
    np.testing.assert_array_equal(
        maximum.arrays[0].read().result(), data.max(axis=2, keepdims=True)
    )
    assert len(maximum.multiscale_factors_yx) > 1
    for factor, arrays in zip(
        maximum.multiscale_factors_yx, maximum.multiscale_arrays, strict=True
    ):
        np.testing.assert_array_equal(
            arrays[0].read().result(),
            data.max(axis=2, keepdims=True)[..., ::factor, ::factor],
        )
    state = ProcessingState.read(processing_state_path(tmp_path, "live_empty"))
    assert state.zero_channels(path) == {(0, 0, 0)}
    assert state.completed_tiles(path) == {(0, 0)}
    assert "opm_v2" not in output.attributes
    coordinates = output.attributes["opm_processing_coordinates"]
    assert coordinates["version"] == STAGE_COORDINATE_VERSION
    assert coordinates["stage_positions_zxy_um"] == [[30.0, 100.0, 200.0]]


@pytest.mark.integration
@pytest.mark.parametrize("action", ["resume", "overwrite"])
def test_offline_resume_skips_durably_completed_tile_and_default_overwrites(
    tmp_path,
    resume_acquisition,
    interrupt_completed_tile,
    action,
    assert_uniform_photon_accuracy,
    relocate_acquisition,
):
    """Recover partial real acquisitions across relocation without changing completed photons."""
    process_module = importlib.import_module("opm_processing.process")
    case = resume_acquisition()
    dataset = case.dataset
    options = {
        "max_projection": False,
        "create_fused_max_projection": False,
        "z_downsample_level": 1,
    }
    with interrupt_completed_tile(case.output), pytest.raises(KeyboardInterrupt):
        process_module.process_skewed(dataset.path, dataset.metadata, **options)
    first = open_position_collection(case.output).arrays[0].read().result()
    assert_uniform_photon_accuracy(first, case.scene, position=0)
    dataset.raw_arrays[0].write(case.changed.pixels[:, 0, :, ::-1].copy()).result()
    root = relocate_acquisition(tmp_path, copy=action == "resume")
    source = root / dataset.path.name
    result = CliRunner().invoke(
        app,
        [
            str(source),
            "--output",
            str(root),
            "--no-max-projection",
            "--no-create-fused-max-projection",
            "--z-downsample-level",
            "1",
        ],
        input=f"{action}\n",
    )
    assert result.exit_code == 0, result.exception
    assert "Resume / Overwrite / Cancel" in result.output
    output = open_position_collection(root / case.output.name)
    actual = output.arrays[0].read().result()
    assert_uniform_photon_accuracy(
        actual, case.scene if action == "resume" else case.changed, position=0
    )
    if action == "resume":
        np.testing.assert_array_equal(actual, first)
    else:
        assert not np.array_equal(actual, first)
    assert_uniform_photon_accuracy(
        output.arrays[1].read().result(), case.scene, position=1
    )
    process_module.process_skewed(
        source, inspect_acquisition(source), output_dir=root, **options
    )
    overwritten = open_position_collection(root / case.output.name)
    assert_uniform_photon_accuracy(
        overwritten.arrays[0].read().result(), case.changed, position=0
    )
    assert_uniform_photon_accuracy(
        overwritten.arrays[1].read().result(), case.scene, position=1
    )


@pytest.mark.integration
def test_live_processing_resumes_completed_output_tiles(
    tmp_path,
    resume_acquisition,
    interrupt_completed_tile,
    assert_uniform_photon_accuracy,
    relocate_acquisition,
):
    """Preserve durable dark tiles and recompute an interrupted physical sample."""
    case = resume_acquisition(live=True)
    dataset = case.dataset
    log_path = tmp_path / "live_resume.log.jsonl"
    lifecycle = log_path.read_bytes()
    options = {
        "max_projection": True,
        "create_fused_max_projection": False,
        "z_downsample_level": 1,
    }
    with interrupt_completed_tile(case.maximum), pytest.raises(KeyboardInterrupt):
        process(root_path=tmp_path, live=dataset.illumination_path, **options)
    output = open_position_collection(case.output)
    maximum = open_position_collection(case.maximum)
    np.testing.assert_array_equal(output.arrays[0].read().result(), 0)
    np.testing.assert_array_equal(maximum.arrays[0].read().result(), 0)
    output.arrays[1][0, 0, 0, 0, 0].write(np.uint16(999)).result()
    dataset.raw_arrays[0].write(case.changed.pixels[:, 0, :, ::-1].copy()).result()
    root = relocate_acquisition(tmp_path, copy=True)
    process(root_path=root, live=root / dataset.illumination_path.name, **options)
    resumed = open_position_collection(root / case.output.name)
    projection = open_position_collection(root / case.maximum.name)
    np.testing.assert_array_equal(resumed.arrays[0].read().result(), 0)
    np.testing.assert_array_equal(projection.arrays[0].read().result(), 0)
    actual = resumed.arrays[1].read().result()
    assert_uniform_photon_accuracy(actual, case.scene, position=1)
    np.testing.assert_array_equal(
        projection.arrays[1].read().result(), actual.max(axis=2, keepdims=True)
    )
    assert (root / log_path.name).read_bytes() == lifecycle
    state = ProcessingState.read(processing_state_path(root, "live_resume"))
    assert state.completed_tiles(root / case.output.name) == {(0, 0), (0, 1)}
    assert state.completed_tiles(root / case.maximum.name) == {(0, 0), (0, 1)}
