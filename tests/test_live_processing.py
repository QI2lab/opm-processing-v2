"""Coverage for on-the-fly OPM tile discovery and illumination loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tifffile import imwrite
from typer.main import get_command

from opm_processing.dataio.acquisition import inspect_acquisition
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
from opm_processing.imageprocessing.opmtools import orthogonal_deskew
from opm_processing.process import _load_provided_illumination, app, process


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


@pytest.mark.parametrize(
    ("save_float32", "expected_dtype"),
    ((False, np.dtype(np.uint16)), (True, np.dtype(np.float32))),
)
def test_live_process_matches_direct_deskew_without_estimating(
    tmp_path, monkeypatch, save_float32, expected_dtype
) -> None:
    """Process a complete live acquisition using only the supplied illumination."""
    data_path = tmp_path / "live.ome.zarr"
    raw = (
        np.arange(4 * 8 * 5, dtype=np.uint16).reshape(4, 8, 5) % 7
    ) + 100
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
        output_dtype=np.float32,
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
        root_path=data_path,
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
    illumination_step = output.attributes["opm_processing"]["steps"][2]
    assert illumination_step["parameters"]["source"] == "provided"
    assert illumination_step["parameters"]["estimated"] is False
