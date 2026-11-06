"""Manifest, lifecycle and illumination fixtures for streaming acquisitions."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tifffile import imwrite

from opm_processing.dataio.live import (
    LIVE_MANIFEST_SCHEMA,
    LiveManifest,
)
from opm_processing.dataio.position_collection import create_position_collection
from opm_processing.imageprocessing.camera import camera_correct, illumination_correct

if TYPE_CHECKING:
    from pathlib import Path


def manifest_document(data_path: Path, *, timepoints: int = 2) -> dict:
    """Describe a calibrated two-channel stage acquisition before any frames arrive."""
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


def write_manifest(
    data_path: Path, *, timepoints: int = 2, document: dict | None = None
) -> LiveManifest:
    """Persist a live acquisition plan and reopen it through the real reader."""
    manifest_path = data_path.with_name(
        data_path.name.removesuffix(".ome.zarr") + ".manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            document
            if document is not None
            else manifest_document(data_path, timepoints=timepoints)
        ),
        encoding="utf-8",
    )
    return LiveManifest.read(manifest_path)


@pytest.fixture
def opm_v2_live_acquisition_factory(tmp_path, opm_v2_acquisition_factory):
    """Write complete live camera acquisitions and lifecycle files with opm-v2.

    Returns
    -------
    callable
        Persist simulated observations and the illumination supplied to processing.
    """
    from opm_processing.dataio.acquisition import ChannelMetadata

    def create(
        raw_data,
        *,
        name="sample",
        illumination=None,
        camera_conversion=0.24,
    ):
        """Deliver actual frames through the upstream publisher and TensorStore writer."""
        path = tmp_path / f"{name}.ome.zarr"
        document = manifest_document(path, timepoints=raw_data.shape[0])
        document["data_path"] = str(path)
        document["camera_e_to_adu"] = camera_conversion
        document["index_sizes"] = dict(zip("tpcz", raw_data.shape[:4], strict=True))
        document["stage_positions_zxy"] = [
            [30.0, 100.0, 200.0 + 2 * index] for index in range(raw_data.shape[1])
        ]
        document["channels"] = [
            {
                "name": f"637nm-{index}",
                "wavelength_nm": 637.0,
                "exposure_ms": 10.0,
                "laser_power": 12.0,
            }
            for index in range(raw_data.shape[2])
        ]
        dataset = opm_v2_acquisition_factory(
            raw_data,
            name=name,
            mode="stage",
            live_manifest=document,
            channels=tuple(
                ChannelMetadata(index, f"637nm-{index}", 637.0, 10.0, 12.0)
                for index in range(raw_data.shape[2])
            ),
            stage_positions_zxy=tuple(map(tuple, document["stage_positions_zxy"])),
            camera_conversion=document["camera_e_to_adu"],
            orientations=tuple(document["orientations"].items()),
        )
        field_path = tmp_path / f"{name}_illumination.ome.tif"
        field = (
            np.ones((raw_data.shape[2], *raw_data.shape[-2:]), np.float32)
            if illumination is None
            else illumination
        )
        imwrite(field_path, field, metadata={"axes": "CYX"})
        dataset.illumination_path = field_path
        return dataset

    return create


@pytest.fixture
def live_acquisition_factory(tmp_path):
    """Return a writer for complete or partially acquired live camera objects."""

    def create(
        *,
        name="sample",
        shape=(2, 1, 2, 3, 4, 5),
        stage_positions=None,
        chunks=None,
        completed=True,
    ):
        """Persist a stage scan, lifecycle plan and measured illumination field."""
        data_path = tmp_path / f"{name}.ome.zarr"
        document = manifest_document(data_path, timepoints=shape[0])
        document["index_sizes"] = dict(zip("tpcz", shape[:4], strict=True))
        document["channels"] = document["channels"][: shape[2]]
        document["stage_positions_zxy"] = (
            list(stage_positions)
            if stage_positions is not None
            else [(30.0, 100.0, 200.0 + 2 * position) for position in range(shape[1])]
        )
        collection = create_position_collection(
            data_path,
            shape,
            (0.4, 0.115, 0.115),
            channels=tuple(channel["name"] for channel in document["channels"]),
            stage_positions=document["stage_positions_zxy"],
            attributes={
                "opm_v2": {
                    "index_sizes": document["index_sizes"],
                    "acquisition_order": document["acquisition_order"],
                    "configuration": {"acq_config": {"opm_mode": "stage"}},
                }
            },
            chunks=chunks if chunks is not None else (1, 1, 1, *shape[-2:]),
        )
        manifest = write_manifest(data_path, document=document)
        log_path = tmp_path / f"{name}.log.jsonl"
        if completed:
            log_path.write_text(
                json.dumps(
                    {"event": "completed", "acquisition_id": manifest.acquisition_id}
                )
                + "\n",
                encoding="utf-8",
            )
        illumination_path = tmp_path / f"{name}_illumination.ome.tif"
        field = np.ones((shape[2], *shape[-2:]), np.float32)
        imwrite(illumination_path, field, metadata={"axes": "CYX"})
        return SimpleNamespace(
            path=data_path,
            collection=collection,
            manifest=manifest,
            log=log_path,
            illumination_path=illumination_path,
        )

    return create


@pytest.fixture
def live_occupancy_case(
    signal_field,
    uniform_camera_scene,
    opm_v2_live_acquisition_factory,
    microscope_parameters,
):
    """Publish a dark detector-defect channel and independently observed sample."""
    threshold = 6 * microscope_parameters.camera.read_noise_e
    fields = np.broadcast_to(
        np.asarray((0.1, signal_field), np.float32)[:, None, None], (2, 24, 256)
    ).copy()
    scene = uniform_camera_scene(
        channels=2,
        fields=fields,
        exposures=np.asarray((0.0, 0.25))[None, None],
        detector_artifacts=True,
    )
    camera = scene.microscope.camera
    detector = camera_correct(
        scene.pixels[0, 0, 0], camera.offset_adu, camera.electrons_per_adu
    ).astype(np.float64)
    assert np.count_nonzero(detector >= threshold) / detector.size < 0.1
    wrong_order = illumination_correct(detector, fields[0])
    assert np.count_nonzero(wrong_order >= threshold) / detector.size >= 0.1
    dataset = opm_v2_live_acquisition_factory(
        scene.pixels[..., ::-1, :, :].copy(),
        name="live_empty",
        illumination=fields,
        camera_conversion=scene.microscope.camera.electrons_per_adu,
    )
    return SimpleNamespace(
        dataset=dataset,
        scene=scene,
        threshold=threshold,
    )


@pytest.fixture
def resume_acquisition(
    tmp_path,
    uniform_camera_scene,
    opm_v2_acquisition_factory,
    opm_v2_live_acquisition_factory,
):
    """Share real publisher inputs and a second declared optical source state."""

    def create(*, live=False):
        """Acquire independent photons before testing durable partial output recovery."""
        exposures = (
            np.asarray(((0.0, 0.0), (0.2, 0.25)))[None]
            if live
            else np.asarray((0.15, 0.17))[None, :, None]
        )
        parameters = {
            "positions": 2,
            "channels": exposures.shape[-1],
            "exposures": exposures,
            "read_noise_e": 0 if live else 1.6,
        }
        scene = uniform_camera_scene(**parameters)
        changed_exposures = exposures.copy()
        changed_exposures[:, 0] = 0.4
        changed = uniform_camera_scene(**dict(parameters, exposures=changed_exposures))
        raw = scene.pixels[..., ::-1, :, :].copy()
        name = "live_resume" if live else "offline_resume"
        if live:
            dataset = opm_v2_live_acquisition_factory(
                raw,
                name=name,
                camera_conversion=scene.microscope.camera.electrons_per_adu,
            )
        else:
            dataset = opm_v2_acquisition_factory(
                raw,
                name=name,
                mode="stage",
                camera_conversion=scene.microscope.camera.electrons_per_adu,
                stage_positions_zxy=((30.0, 100.0, 200.0), (30.0, 100.0, 202.0)),
            )
        return SimpleNamespace(
            dataset=dataset,
            scene=scene,
            changed=changed,
            output=tmp_path / f"{name}_deskewed.ome.zarr",
            maximum=tmp_path / f"{name}_max_z_deskewed.ome.zarr",
        )

    return create


@pytest.fixture
def interrupt_completed_tile(monkeypatch):
    """Inject an explicit interruption only after the real requested checkpoint."""
    from contextlib import contextmanager

    from opm_processing.dataio.processing_state import ProcessingState

    @contextmanager
    def interrupt(output):
        """Interrupt the first fully written tile of the selected real output."""
        complete = ProcessingState.complete_tile

        def checkpoint(state, path, time, position, **kwargs):
            complete(state, path, time, position, **kwargs)
            if path == output and position == 0:
                raise KeyboardInterrupt

        with monkeypatch.context() as patch:
            patch.setattr(ProcessingState, "complete_tile", checkpoint)
            yield

    return interrupt


@pytest.fixture(params=["estimate", "pending-chunks", "cached"])
def live_wait_case(
    request,
    tmp_path,
    opm_v2_live_acquisition_factory,
    monkeypatch,
    roi_camera_scene,
):
    """Delay real published chunks and lifecycle state for simulated optical input."""
    from opm_processing import process as processing

    kind = request.param
    if kind == "estimate":
        ensemble = request.getfixturevalue("illumination_calibration_ensemble")
        scene, pixels = ensemble.scene, ensemble.raw
        sample_timepoint = ensemble.sample_timepoint
    else:
        scene = roi_camera_scene(shape=(24, 33, 33))
        pixels, sample_timepoint = scene.pixels, 0
    raw = pixels[..., ::-1, :, :].copy()
    produced = opm_v2_live_acquisition_factory(raw, camera_conversion=1.0)
    acquisition, manifest, log = processing.open_live_acquisition(tmp_path)
    produced.manifest, produced.log = manifest, log
    produced.illumination_path.unlink()
    destination = tmp_path / "processed"
    field_path = tmp_path / "sample_flatfield.ome.tif"
    if kind == "cached":
        processing.write_flatfield(
            field_path, np.full((1, 1, *raw.shape[-2:]), 2, np.float32), 0.115
        )
    completed_log = log.read_bytes()
    if kind == "estimate":
        log.write_text(
            json.dumps(
                {
                    "event": "started",
                    "acquisition_id": manifest.acquisition_id,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    pending = None
    if kind != "cached":
        array_root = produced.path / acquisition.array_paths[0]
        chunk = next(
            path
            for path in array_root.rglob("*")
            if path.is_file() and path.name != "zarr.json"
        )
        pending = tmp_path / "pending_camera_chunk"
        chunk.replace(pending)
    sleeps = []

    def finish(seconds):
        """Publish actual writer bytes and the controller's real completion event."""
        assert not destination.exists()
        assert not (destination / field_path.name).exists()
        sleeps.append(seconds)
        if len(sleeps) == 1 and pending is not None:
            pending.replace(chunk)
        elif len(sleeps) == 2 and kind == "estimate":
            log.write_bytes(completed_log)
        else:
            pytest.fail("Processing kept waiting after acquisition completion")

    monkeypatch.setattr(processing.time, "sleep", finish)
    return SimpleNamespace(
        kind=kind,
        dataset=produced,
        raw=raw,
        scene=scene,
        sample_timepoint=sample_timepoint,
        calibration_photons=ensemble.calibration_photons
        if kind == "estimate"
        else None,
        destination=destination,
        sleeps=sleeps,
        field_path=field_path if kind == "cached" else destination / field_path.name,
    )
