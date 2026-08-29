"""Coverage for on-the-fly OPM tile discovery and illumination loading."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from tifffile import imwrite
from typer.main import get_command

from opm_processing.dataio.acquisition import (
    AcquisitionMetadata,
    ChannelMetadata,
    inspect_acquisition,
)
from opm_processing.dataio.live import (
    LIVE_MANIFEST_SCHEMA,
    LiveManifest,
    ZarrTileReadiness,
    iter_live_tiles,
    read_lifecycle_event,
    resolve_live_sidecars,
)
from opm_processing.dataio.position_collection import create_position_collection
from opm_processing.dataio.position_collection import open_position_collection
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)
from opm_processing.imageprocessing.opmtools import orthogonal_deskew
from opm_processing.process import (
    _is_empty_tile,
    _load_provided_illumination,
    _open_live_acquisition,
    _validate_empty_tile_options,
    app,
    process,
)


def _manifest_document(data_path: Path, *, timepoints: int = 2) -> dict:
    """Return a minimal valid live-acquisition manifest."""
    return {
        "schema": LIVE_MANIFEST_SCHEMA,
        "schema_version": "1.0",
        "acquisition_id": "synthetic-acquisition",
        "data_path": data_path.name,
        "mode": "stage",
        "index_sizes": {"t": timepoints, "p": 1, "c": 2, "z": 3},
        "acquisition_order": ["t", "p", "z", "c"],
        "channels": [
            {
                "name": "488nm",
                "wavelength_nm": 488.0,
                "exposure_ms": 10.0,
                "laser_power": 12.0,
            },
            {
                "name": "561nm",
                "wavelength_nm": 561.0,
                "exposure_ms": 15.0,
                "laser_power": 18.0,
            },
        ],
        "stage_positions_zxy": [[30.0, 100.0, 200.0]],
        "scan_axis": "x",
        "scan_axis_step_um": 0.4,
        "pixel_size_um": 0.115,
        "angle_deg": 30.0,
        "camera_offset": 100.0,
        "camera_e_to_adu": 0.24,
        "excess_scan_positions": 0,
        "excess_scan_start_positions": 0,
        "excess_scan_end_positions": 0,
        "orientations": {
            "camera_XYstage_orientation": "positive",
            "camera_Zstage_orientation": "negative",
            "camera_mirror_orientation": "positive",
        },
    }


def _write_manifest(data_path: Path, *, timepoints: int = 2) -> LiveManifest:
    sidecars = resolve_live_sidecars(data_path)
    sidecars.manifest.write_text(
        json.dumps(_manifest_document(data_path, timepoints=timepoints)),
        encoding="utf-8",
    )
    return LiveManifest.read(sidecars.manifest)


def test_manifest_overlays_metadata_available_before_frame_metadata(tmp_path) -> None:
    """Use the immutable plan when per-frame metadata has not been flushed."""
    data_path = tmp_path / "sample.ome.zarr"
    create_position_collection(
        data_path,
        (2, 1, 2, 3, 4, 5),
        (0.4, 0.115, 0.115),
        stage_positions=((30.0, 100.0, 200.0),),
        channels=("488nm", "561nm"),
        attributes={
            "opm_v2": {
                "index_sizes": {"t": 2, "p": 1, "c": 2, "z": 3},
                "acquisition_order": ["t", "p", "z", "c"],
                "configuration": {"acq_config": {"opm_mode": "stage"}},
            }
        },
        chunks=(1, 1, 1, 4, 5),
    )
    manifest = _write_manifest(data_path)

    acquisition = manifest.apply(inspect_acquisition(data_path))

    assert acquisition.mode == "stage"
    assert acquisition.camera_offset == pytest.approx(100.0)
    assert acquisition.camera_conversion == pytest.approx(0.24)
    assert acquisition.stage_positions_zxy == ((30.0, 100.0, 200.0),)
    assert acquisition.channel_names == ("488nm", "561nm")


def test_live_acquisition_directory_resolves_manifest_data_path(tmp_path) -> None:
    """Accept the acquisition directory and resolve its manifest OME-Zarr."""
    acquisition_dir = tmp_path / "timestamped_acquisition"
    acquisition_dir.mkdir()
    data_path = acquisition_dir / "sample.ome.zarr"
    create_position_collection(
        data_path,
        (2, 1, 2, 3, 4, 5),
        (0.4, 0.115, 0.115),
        stage_positions=((30.0, 100.0, 200.0),),
        channels=("488nm", "561nm"),
        attributes={
            "opm_v2": {
                "index_sizes": {"t": 2, "p": 1, "c": 2, "z": 3},
                "acquisition_order": ["t", "p", "z", "c"],
                "configuration": {"acq_config": {"opm_mode": "stage"}},
            }
        },
        chunks=(1, 1, 1, 4, 5),
    )
    manifest = _write_manifest(data_path)

    acquisition, resolved_manifest, log_path = _open_live_acquisition(acquisition_dir)

    assert acquisition.path == data_path.resolve()
    assert resolved_manifest == manifest
    assert log_path == acquisition_dir / "sample.log.jsonl"

    with pytest.raises(ValueError, match="containing acquisition directory"):
        _open_live_acquisition(data_path)


def test_chunk_presence_marks_only_a_fully_written_tile_ready(tmp_path) -> None:
    """Require every C/Z/Y/X chunk before publishing one T/P tile."""
    data_path = tmp_path / "chunks.ome.zarr"
    collection = create_position_collection(
        data_path,
        (2, 1, 2, 3, 4, 5),
        (0.4, 0.115, 0.115),
        stage_positions=((0.0, 0.0, 0.0),),
        channels=("488nm", "561nm"),
        attributes={
            "opm_v2": {
                "index_sizes": {"t": 2, "p": 1, "c": 2, "z": 3},
                "acquisition_order": ["t", "p", "z", "c"],
                "configuration": {"acq_config": {"opm_mode": "stage"}},
            }
        },
        chunks=(1, 1, 1, 2, 3),
    )
    manifest = _write_manifest(data_path)
    acquisition = manifest.apply(inspect_acquisition(data_path))
    readiness = ZarrTileReadiness(acquisition)

    assert readiness.ready_tiles() == set()
    collection.arrays[0][0, :, :2].write(
        np.ones((2, 2, 4, 5), dtype=np.uint16)
    ).result()
    assert readiness.ready_tiles() == set()

    collection.arrays[0][0, :, 2].write(np.ones((2, 4, 5), dtype=np.uint16)).result()
    assert readiness.ready_tiles() == {(0, 0)}


def test_live_iterator_polls_then_stops_at_completed_log(tmp_path) -> None:
    """Poll only while caught up and stop after all completed tiles are yielded."""
    data_path = tmp_path / "iterator.ome.zarr"
    manifest_path = tmp_path / "iterator.manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest_document(data_path, timepoints=1)), encoding="utf-8"
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


def test_live_iterator_skips_tiles_completed_before_restart(tmp_path) -> None:
    """Seed iterator state with processed tiles discovered from existing output."""
    data_path = tmp_path / "resume.ome.zarr"
    manifest_path = tmp_path / "resume.manifest.json"
    document = _manifest_document(data_path, timepoints=1)
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


def test_lifecycle_reader_ignores_an_incomplete_last_record(tmp_path) -> None:
    """Do not parse a JSONL record while the controller is appending it."""
    log_path = tmp_path / "sample.log.jsonl"
    log_path.write_text(
        '{"event":"started","acquisition_id":"id"}\n{"event":"compl',
        encoding="utf-8",
    )
    assert read_lifecycle_event(log_path, "id") == "started"


def test_provided_illumination_is_strictly_validated(tmp_path) -> None:
    """Accept a matching image and reject invalid values without estimation."""
    path = tmp_path / "illumination.ome.tif"
    expected = np.ones((2, 4, 5), dtype=np.float32)
    imwrite(path, expected, metadata={"axes": "CYX"})
    np.testing.assert_array_equal(
        _load_provided_illumination(path, expected.shape), expected
    )

    invalid_path = tmp_path / "invalid.ome.tif"
    invalid = expected.copy()
    invalid[0, 0, 0] = 0
    imwrite(invalid_path, invalid, metadata={"axes": "CYX"})
    with pytest.raises(ValueError, match="strictly positive"):
        _load_provided_illumination(invalid_path, expected.shape)


def test_live_cli_uses_one_path_valued_option() -> None:
    """Expose the illumination path through the single requested live option."""
    parameter = next(item for item in get_command(app).params if item.name == "live")
    assert parameter.opts == ["--live"]
    assert parameter.type.name == "path"
    assert parameter.is_flag is False


def test_save_float32_cli_is_an_opt_in_flag() -> None:
    """Keep uint16 as the default while exposing float32 output explicitly."""
    parameter = next(
        item for item in get_command(app).params if item.name == "save_float32"
    )
    assert parameter.opts == ["--save-float32"]
    assert parameter.is_flag is True
    assert parameter.default is False


def test_resume_cli_is_an_opt_in_flag() -> None:
    """Overwrite by default and resume only when explicitly requested."""
    parameter = next(item for item in get_command(app).params if item.name == "resume")
    assert parameter.opts == ["--resume"]
    assert parameter.is_flag is True
    assert parameter.default is False


def test_empty_tile_cli_and_hot_pixel_resistant_detection() -> None:
    """Use global occupancy to ignore sparse noise and hot pixels."""
    command = get_command(app)
    threshold_option = next(
        item for item in command.params if item.name == "skip_empty_below"
    )
    fraction_option = next(
        item for item in command.params if item.name == "skip_empty_min_signal_fraction"
    )
    assert threshold_option.opts == ["--skip-empty-below"]
    assert threshold_option.default is None
    assert fraction_option.opts == ["--skip-empty-min-signal-fraction"]
    assert fraction_option.default == 0.01

    stack = np.zeros((5, 4, 4), dtype=np.float32)
    stack[:, 0, 0] = 10.0
    assert _is_empty_tile(
        stack,
        threshold=2.0,
        min_signal_fraction=0.1,
    )
    stack[-1, 0, :4] = 2.0
    assert not _is_empty_tile(
        stack,
        threshold=2.0,
        min_signal_fraction=0.1,
    )
    assert not _is_empty_tile(
        stack,
        threshold=None,
        min_signal_fraction=0.1,
    )

    sparse_artifacts = np.zeros((100, 20, 20), dtype=np.float32)
    sparse_artifacts[:10, :10, :10] = 10.0
    assert _is_empty_tile(
        sparse_artifacts,
        threshold=2.0,
        min_signal_fraction=0.05,
    )
    sparse_artifacts[:20, :10, :10] = 10.0
    assert not _is_empty_tile(
        sparse_artifacts,
        threshold=2.0,
        min_signal_fraction=0.05,
    )


@pytest.mark.parametrize(
    ("threshold", "signal_fraction", "message"),
    (
        (-1.0, 0.01, "finite and nonnegative"),
        (float("nan"), 0.01, "finite and nonnegative"),
        (1.0, 0.0, "greater than 0"),
        (1.0, 1.1, "at most 1"),
    ),
)
def test_empty_tile_options_are_validated(threshold, signal_fraction, message) -> None:
    """Reject thresholds and occupancy fractions with undefined behavior."""
    with pytest.raises(ValueError, match=message):
        _validate_empty_tile_options(threshold, signal_fraction)


@pytest.mark.parametrize(
    ("save_float32", "expected_dtype"),
    ((False, np.dtype(np.uint16)), (True, np.dtype(np.float32))),
)
def test_live_process_matches_direct_deskew_without_estimating(
    tmp_path, monkeypatch, save_float32, expected_dtype
) -> None:
    """Process a complete live acquisition using only the supplied illumination."""
    data_path = tmp_path / "live.ome.zarr"
    raw = (np.arange(4 * 8 * 5, dtype=np.uint16).reshape(4, 8, 5) % 7) + 100
    corrected = np.maximum(
        (np.flip(raw, axis=0).astype(np.float32) - 100.0) * 0.24,
        0,
    )
    expected_float = orthogonal_deskew(
        corrected,
        theta=30.0,
        distance=0.4,
        pixel_size=0.115,
        downsample_factor=1,
    )
    collection = create_position_collection(
        data_path,
        (1, 1, 1, 4, 8, 5),
        (0.4, 0.115, 0.115),
        stage_positions=((30.0, 100.0, 200.0),),
        channels=("488nm",),
        attributes={
            "opm_v2": {
                "index_sizes": {"t": 1, "p": 1, "c": 1, "z": 4},
                "acquisition_order": ["t", "p", "z", "c"],
                "configuration": {"acq_config": {"opm_mode": "stage"}},
            }
        },
        chunks=(1, 1, 1, 8, 5),
    )
    collection.arrays[0][0, 0].write(raw).result()

    manifest_document = _manifest_document(data_path, timepoints=1)
    manifest_document["index_sizes"] = {"t": 1, "p": 1, "c": 1, "z": 4}
    manifest_document["channels"] = manifest_document["channels"][:1]
    sidecars = resolve_live_sidecars(data_path)
    sidecars.manifest.write_text(json.dumps(manifest_document), encoding="utf-8")
    sidecars.log.write_text(
        json.dumps({"event": "completed", "acquisition_id": "synthetic-acquisition"})
        + "\n",
        encoding="utf-8",
    )
    illumination_path = tmp_path / "illumination.ome.tif"
    imwrite(
        illumination_path,
        np.ones((1, 8, 5), dtype=np.float32),
        metadata={"axes": "CYX"},
    )
    monkeypatch.setattr(
        "opm_processing.process.call_estimate_illuminations",
        lambda *_args, **_kwargs: pytest.fail("live mode estimated illumination"),
    )

    process(
        root_path=tmp_path,
        live=illumination_path,
        max_projection=False,
        create_fused_max_projection=False,
        z_downsample_level=1,
        save_float32=save_float32,
    )

    output = open_position_collection(tmp_path / "live_deskewed.ome.zarr")
    actual = output.arrays[0][0, 0].read().result()
    expected = (
        expected_float
        if save_float32
        else np.clip(expected_float, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    )
    assert np.dtype(actual.dtype) == expected_dtype
    np.testing.assert_allclose(actual, expected)
    if save_float32:
        positive = actual[actual > 0]
        assert positive.size > 0
        assert np.any(positive != np.floor(positive))
    assert not any(key.startswith("opm_") for key in output.attributes)


def test_live_deconvolution_builds_each_channel_psf_once(tmp_path, monkeypatch) -> None:
    """Prepare PSFs before tile iteration and reuse them across positions."""
    data_path = tmp_path / "live_decon.ome.zarr"
    collection = create_position_collection(
        data_path,
        (1, 2, 2, 4, 8, 5),
        (0.4, 0.115, 0.115),
        stage_positions=((30.0, 100.0, 200.0), (30.0, 100.0, 202.0)),
        channels=("488nm", "561nm"),
        attributes={
            "opm_v2": {
                "index_sizes": {"t": 1, "p": 2, "c": 2, "z": 4},
                "acquisition_order": ["t", "p", "z", "c"],
                "configuration": {"acq_config": {"opm_mode": "stage"}},
            }
        },
        chunks=(1, 1, 1, 8, 5),
    )
    raw = np.full((2, 4, 8, 5), 110, dtype=np.uint16)
    for position in range(2):
        collection.arrays[position][0].write(raw).result()

    manifest_document = _manifest_document(data_path, timepoints=1)
    manifest_document["index_sizes"] = {"t": 1, "p": 2, "c": 2, "z": 4}
    manifest_document["stage_positions_zxy"] = [
        [30.0, 100.0, 200.0],
        [30.0, 100.0, 202.0],
    ]
    sidecars = resolve_live_sidecars(data_path)
    sidecars.manifest.write_text(json.dumps(manifest_document), encoding="utf-8")
    sidecars.log.write_text(
        json.dumps({"event": "completed", "acquisition_id": "synthetic-acquisition"})
        + "\n",
        encoding="utf-8",
    )
    illumination_path = tmp_path / "illumination.ome.tif"
    imwrite(
        illumination_path,
        np.ones((2, 8, 5), dtype=np.float32),
        metadata={"axes": "CYX"},
    )

    generated_wavelengths = []
    deconvolution_calls = []
    progress_calls = []

    def fake_generate_skewed_psf(*, em_wvl, **_kwargs):
        generated_wavelengths.append(em_wvl)
        return np.ones((3, 3, 3), dtype=np.float32)

    class FakeChunkState:
        def __init__(self, _crop_scan):
            pass

        def determine_once(self, _image_shape, _psf_shapes, *, gpu_id):
            del gpu_id
            return 4

        def remember_successful_crop(self, _crop_scan):
            pass

    def fake_chunked_rlgc(image, psf, **_kwargs):
        deconvolution_calls.append(id(psf))
        return np.asarray(image, dtype=np.float32)

    process_module = importlib.import_module("opm_processing.process")
    rlgc_module = importlib.import_module("opm_processing.imageprocessing.rlgc")

    def recording_tqdm(iterable, **kwargs):
        progress_calls.append(kwargs)
        return iterable

    monkeypatch.setattr(process_module, "tqdm", recording_tqdm)
    monkeypatch.setattr(
        process_module,
        "generate_skewed_psf",
        fake_generate_skewed_psf,
    )
    monkeypatch.setattr(rlgc_module, "RlgcChunkState", FakeChunkState)
    monkeypatch.setattr(rlgc_module, "chunked_rlgc", fake_chunked_rlgc)

    process(
        root_path=tmp_path,
        live=illumination_path,
        deconvolve=True,
        max_projection=False,
        create_fused_max_projection=False,
        z_downsample_level=1,
    )

    assert generated_wavelengths == [pytest.approx(0.488), pytest.approx(0.561)]
    assert len(deconvolution_calls) == 4
    assert len(set(deconvolution_calls)) == 2
    position_progress = [call for call in progress_calls if call.get("desc") == "p"]
    assert position_progress == [
        {"total": 2, "initial": 0, "desc": "p", "unit": "tile"}
    ]


def test_live_empty_channel_skips_deconvolution_and_writes_zero(
    tmp_path, monkeypatch
) -> None:
    """Ignore a persistent hot pixel while processing a channel with real signal."""
    data_path = tmp_path / "live_empty.ome.zarr"
    collection = create_position_collection(
        data_path,
        (1, 1, 2, 4, 8, 5),
        (0.4, 0.115, 0.115),
        stage_positions=((30.0, 100.0, 200.0),),
        channels=("488nm", "561nm"),
        attributes={
            "opm_v2": {
                "index_sizes": {"t": 1, "p": 1, "c": 2, "z": 4},
                "acquisition_order": ["t", "p", "z", "c"],
                "configuration": {"acq_config": {"opm_mode": "stage"}},
            }
        },
        chunks=(1, 1, 1, 8, 5),
    )
    raw = np.full((2, 4, 8, 5), 100, dtype=np.uint16)
    raw[0, :, 0, 0] = 200  # One persistent detector hot pixel.
    raw[0, 0, 0, 1:5] = 104  # Sub-threshold signal before illumination correction.
    raw[1, 2] = 200  # A sample-like occupied detector frame.
    collection.arrays[0][0].write(raw).result()

    manifest_document = _manifest_document(data_path, timepoints=1)
    manifest_document["index_sizes"] = {"t": 1, "p": 1, "c": 2, "z": 4}
    sidecars = resolve_live_sidecars(data_path)
    sidecars.manifest.write_text(json.dumps(manifest_document), encoding="utf-8")
    sidecars.log.write_text(
        json.dumps({"event": "completed", "acquisition_id": "synthetic-acquisition"})
        + "\n",
        encoding="utf-8",
    )
    illumination_path = tmp_path / "illumination.ome.tif"
    illumination = np.ones((2, 8, 5), dtype=np.float32)
    illumination[0] = 0.1  # Would amplify the four weak pixels above threshold.
    imwrite(illumination_path, illumination, metadata={"axes": "CYX"})

    deconvolution_calls = []

    class FakeChunkState:
        def __init__(self, _crop_scan):
            pass

        def determine_once(self, _image_shape, _psf_shapes, *, gpu_id):
            del gpu_id
            return 4

        def remember_successful_crop(self, _crop_scan):
            pass

    def fake_chunked_rlgc(image, _psf, **_kwargs):
        deconvolution_calls.append(np.asarray(image).copy())
        return np.asarray(image, dtype=np.float32)

    process_module = importlib.import_module("opm_processing.process")
    rlgc_module = importlib.import_module("opm_processing.imageprocessing.rlgc")
    monkeypatch.setattr(
        process_module,
        "generate_skewed_psf",
        lambda **_kwargs: np.ones((3, 3, 3), dtype=np.float32),
    )
    monkeypatch.setattr(rlgc_module, "RlgcChunkState", FakeChunkState)
    monkeypatch.setattr(rlgc_module, "chunked_rlgc", fake_chunked_rlgc)

    process(
        root_path=tmp_path,
        live=illumination_path,
        deconvolve=True,
        skip_empty_below=2.0,
        skip_empty_min_signal_fraction=0.1,
        max_projection=True,
        create_fused_max_projection=False,
        z_downsample_level=1,
    )

    assert len(deconvolution_calls) == 1
    output = open_position_collection(tmp_path / "live_empty_decon_deskewed.ome.zarr")
    max_output = open_position_collection(
        tmp_path / "live_empty_max_z_decon_deskewed.ome.zarr"
    )
    assert np.count_nonzero(output.arrays[0][0, 0].read().result()) == 0
    assert np.count_nonzero(max_output.arrays[0][0, 0].read().result()) == 0
    assert np.count_nonzero(output.arrays[0][0, 1].read().result()) > 0
    state = ProcessingState.read(processing_state_path(tmp_path, "live_empty"))
    assert state.zero_channels(tmp_path / "live_empty_decon_deskewed.ome.zarr") == {
        (0, 0, 0)
    }
    assert len(max_output.multiscale_factors_yx) > 1
    for level_arrays in max_output.multiscale_arrays:
        assert np.count_nonzero(level_arrays[0][0, 0].read().result()) == 0
        assert np.count_nonzero(level_arrays[0][0, 1].read().result()) > 0
    assert not any(key.startswith("opm_") for key in output.attributes)


def test_offline_resume_skips_durably_completed_tile_and_default_overwrites(
    tmp_path,
    monkeypatch,
) -> None:
    """Checkpoint after writes, resume the next tile, and overwrite by default."""
    process_module = importlib.import_module("opm_processing.process")
    data_path = tmp_path / "offline_resume.ome.zarr"
    collection = create_position_collection(
        data_path,
        (1, 2, 1, 4, 8, 5),
        (0.4, 0.115, 0.115),
        stage_positions=((30.0, 100.0, 200.0), (30.0, 100.0, 202.0)),
        channels=("488nm",),
    )
    collection.arrays[0][0].write(np.full((1, 4, 8, 5), 110, dtype=np.uint16)).result()
    collection.arrays[1][0].write(np.full((1, 4, 8, 5), 120, dtype=np.uint16)).result()
    metadata = AcquisitionMetadata(
        path=data_path,
        storage_format="opm-v2-ome-zarr-v3",
        mode="stage",
        axes=("t", "p", "c", "z", "y", "x"),
        shape=(1, 2, 1, 4, 8, 5),
        array_paths=("0/0", "1/0"),
        acquisition_order=("t", "p", "z", "c"),
        channels=(ChannelMetadata(0, "488nm", 488.0, 2.0, 10.0),),
        stage_positions_zxy=((30.0, 100.0, 200.0), (30.0, 100.0, 202.0)),
        scan_start_positions_xyz=(),
        scan_end_positions_xyz=(),
        scan_axis="x",
        scan_axis_step_um=0.4,
        pixel_size_um=0.115,
        angle_deg=30.0,
        camera_offset=100.0,
        camera_conversion=1.0,
        excess_scan_positions=0,
        excess_scan_start_positions=0,
        excess_scan_end_positions=0,
        orientations=(),
        sidecar_paths=(),
    )

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
    collection.arrays[0][0].write(np.full((1, 4, 8, 5), 500, dtype=np.uint16)).result()

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
    assert np.any(resumed.arrays[1][0, 0].read().result())

    process_module.process_skewed(
        data_path,
        acquisition=metadata,
        output_dir=tmp_path,
        max_projection=False,
        create_fused_max_projection=False,
        z_downsample_level=1,
    )
    overwritten = open_position_collection(output_path)
    assert not np.array_equal(
        overwritten.arrays[0][0, 0].read().result(),
        completed_tile,
    )


def test_live_processing_resumes_completed_output_tiles(tmp_path, monkeypatch) -> None:
    """Preserve completed tiles and safely redo a partially written next tile."""
    data_path = tmp_path / "live_resume.ome.zarr"
    collection = create_position_collection(
        data_path,
        (1, 2, 2, 4, 8, 5),
        (0.4, 0.115, 0.115),
        stage_positions=((30.0, 100.0, 200.0), (30.0, 100.0, 202.0)),
        channels=("488nm", "561nm"),
        attributes={
            "opm_v2": {
                "index_sizes": {"t": 1, "p": 2, "c": 2, "z": 4},
                "acquisition_order": ["t", "p", "z", "c"],
                "configuration": {"acq_config": {"opm_mode": "stage"}},
            }
        },
        chunks=(1, 1, 1, 8, 5),
    )
    for position, raw_value in enumerate((100, 111)):
        collection.arrays[position][0].write(
            np.full((2, 4, 8, 5), raw_value, dtype=np.uint16)
        ).result()

    manifest_document = _manifest_document(data_path, timepoints=1)
    manifest_document["index_sizes"] = {"t": 1, "p": 2, "c": 2, "z": 4}
    manifest_document["stage_positions_zxy"] = [
        [30.0, 100.0, 200.0],
        [30.0, 100.0, 202.0],
    ]
    sidecars = resolve_live_sidecars(data_path)
    sidecars.manifest.write_text(json.dumps(manifest_document), encoding="utf-8")
    sidecars.log.write_text(
        json.dumps({"event": "completed", "acquisition_id": "synthetic-acquisition"})
        + "\n",
        encoding="utf-8",
    )
    illumination_path = tmp_path / "illumination.ome.tif"
    imwrite(
        illumination_path,
        np.ones((2, 8, 5), dtype=np.float32),
        metadata={"axes": "CYX"},
    )

    process_module = importlib.import_module("opm_processing.process")

    def interrupted_tiles(_readiness, _manifest, _log_path, *, completed_tiles):
        assert completed_tiles == set()
        yield 0, 0
        raise KeyboardInterrupt

    monkeypatch.setattr(process_module, "iter_live_tiles", interrupted_tiles)
    with pytest.raises(KeyboardInterrupt):
        process(
            root_path=tmp_path,
            live=illumination_path,
            max_projection=True,
            create_fused_max_projection=False,
            z_downsample_level=1,
        )

    output_path = tmp_path / "live_resume_deskewed.ome.zarr"
    max_output_path = tmp_path / "live_resume_max_z_deskewed.ome.zarr"
    first_output = open_position_collection(output_path)
    first_max_output = open_position_collection(max_output_path)
    completed_before_restart = first_output.arrays[0].read().result()
    completed_max_before_restart = first_max_output.arrays[0].read().result()
    first_output.arrays[1][0, 0, 0, 0, 0].write(np.uint16(999)).result()

    observed_completed = []

    def resumed_tiles(_readiness, _manifest, _log_path, *, completed_tiles):
        observed_completed.append(set(completed_tiles))
        yield 0, 1

    monkeypatch.setattr(process_module, "iter_live_tiles", resumed_tiles)
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
    assert observed_completed == [{(0, 0)}]
    assert np.all(resumed_output.arrays[1].read().result() != 999)
    acquisition_records = [
        json.loads(line)
        for line in sidecars.log.read_text(encoding="utf-8").splitlines()
    ]
    assert acquisition_records == [
        {"event": "completed", "acquisition_id": "synthetic-acquisition"}
    ]
    state = ProcessingState.read(processing_state_path(tmp_path, "live_resume"))
    assert state.completed_tiles(output_path) == {(0, 0), (0, 1)}
    assert state.completed_tiles(max_output_path) == {(0, 0), (0, 1)}
