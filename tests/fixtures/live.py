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

if TYPE_CHECKING:
    from pathlib import Path


def manifest_document(data_path: Path, *, timepoints: int = 2) -> dict:
    """Describe a calibrated two-channel stage acquisition before any frames arrive.

    Parameters
    ----------
    data_path : pathlib.Path
        Planned OME-Zarr location recorded relative to its sidecar.
    timepoints : int
        Planned acquisition time count.

    Returns
    -------
    dict
        Controller plan with known stage geometry, channels and camera calibration.
    """
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
    """Persist a live acquisition plan and reopen it through the real reader.

    Parameters
    ----------
    data_path : pathlib.Path
        Planned OME-Zarr acquisition path.
    timepoints : int
        Time count when using the standard two-channel stage plan.
    document : dict or None
        Explicit physical plan; otherwise use the common manifest configuration.

    Returns
    -------
    LiveManifest
        Acquisition plan read from its persisted sidecar.
    """
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
    """Return a writer for complete or partially acquired live camera objects.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the camera store and controller sidecars.

    Returns
    -------
    callable
        Writer accepting a known camera object or an incomplete acquisition plan.
    """

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
        """Persist a stage scan, lifecycle plan and measured illumination field.

        Parameters
        ----------
        raw_data : numpy.ndarray or None
            TPCZYX camera object; None leaves chunks unwritten for readiness tests.
        name : str
            Acquisition basename in the temporary directory.
        shape : tuple of int
            Planned TPCZYX dimensions when pixels are not yet available.
        stage_positions : sequence or None
            Recorded ZXY stage coordinates; defaults to separated lateral fields.
        chunks : tuple of int or None
            TCZYX storage chunk shape; defaults to one complete detector frame.
        completed : bool
            Append the controller's completed acquisition event.
        illumination : numpy.ndarray or None
            Known CYX sensitivity field; None writes uniform unity sensitivity.

        Returns
        -------
        types.SimpleNamespace
            Acquisition, collection, manifest, lifecycle log and illumination paths.
        """
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
