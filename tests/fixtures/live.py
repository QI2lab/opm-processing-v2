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
from opm_processing.imageprocessing.opmtools import orthogonal_deskew

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
    sidecars = SimpleNamespace(
        manifest=data_path.with_name(
            data_path.name.removesuffix(".ome.zarr") + ".manifest.json"
        ),
        log=data_path.with_name(
            data_path.name.removesuffix(".ome.zarr") + ".log.jsonl"
        ),
    )
    sidecars.manifest.write_text(
        json.dumps(
            document
            if document is not None
            else manifest_document(data_path, timepoints=timepoints)
        ),
        encoding="utf-8",
    )
    return LiveManifest.read(sidecars.manifest)


@pytest.fixture
def live_acquisition_factory(tmp_path):
    """Return a writer for complete or partially acquired live camera objects."""

    def create(
        raw_data=None,
        *,
        name="sample",
        shape=(2, 1, 2, 3, 4, 5),
        stage_positions=None,
        chunks=None,
        completed=True,
        illumination=None,
    ):
        """Persist a stage scan, lifecycle plan and measured illumination field."""
        if raw_data is not None:
            shape = raw_data.shape
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
        if raw_data is not None:
            for position, array in enumerate(collection.arrays):
                array.write(raw_data[:, position]).result()
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
        field = (
            np.ones((shape[2], *shape[-2:]), np.float32)
            if illumination is None
            else illumination
        )
        imwrite(illumination_path, field, metadata={"axes": "CYX"})
        return SimpleNamespace(
            path=data_path,
            collection=collection,
            manifest=manifest,
            log=log_path,
            illumination_path=illumination_path,
        )

    return create


@pytest.fixture(params=["estimate", "pending-chunks", "cached"])
def live_wait_case(
    request, tmp_path, live_acquisition_factory, monkeypatch, simulated_camera
):
    """Share camera input and controlled completion for offline live-wait decisions."""
    from opm_processing import process as processing

    kind = request.param
    shape = (
        1,
        1,
        1,
        12 if kind == "estimate" else 4,
        8,
        16 if kind == "estimate" else 5,
    )
    raw = (
        simulated_camera(shape[-3:], conversion=0.24).raw[None, None, None, ::-1].copy()
    )
    dataset = live_acquisition_factory(
        raw if kind == "cached" else None, shape=shape, completed=kind != "estimate"
    )
    dataset.illumination_path.unlink()
    destination = tmp_path / "processed"
    field_path = tmp_path / "sample_flatfield.ome.tif"
    if kind == "cached":
        processing.write_flatfield(
            field_path, np.full((1, 1, *shape[-2:]), 2, np.float32), 0.115
        )
    sleeps = []

    def finish(seconds):
        """Write the acquisition completion event during the wait callback."""
        assert not destination.exists()
        assert not (destination / field_path.name).exists()
        sleeps.append(seconds)
        if len(sleeps) == 1:
            dataset.collection.arrays[0].write(raw[:, 0]).result()
            assert kind != "cached"
        elif len(sleeps) == 2 and kind == "estimate":
            dataset.log.write_text(
                json.dumps(
                    {
                        "event": "completed",
                        "acquisition_id": dataset.manifest.acquisition_id,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            pytest.fail("Processing kept waiting after acquisition completion")

    monkeypatch.setattr(processing.time, "sleep", finish)
    calibrated = np.maximum(raw[0, 0, 0, ::-1].astype(np.float32) - 100, 0) * 0.24
    return SimpleNamespace(
        kind=kind,
        dataset=dataset,
        raw=raw,
        destination=destination,
        sleeps=sleeps,
        expected=orthogonal_deskew(
            calibrated / (2 if kind == "cached" else 1),
            theta=30.0,
            distance=0.4,
            pixel_size=0.115,
            downsample_factor=1,
        ),
        field_path=field_path if kind == "cached" else destination / field_path.name,
    )
