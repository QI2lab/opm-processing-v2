"""Self-contained processed snapshot metadata and lightweight TensorStore access.

This module imports metadata/IO dependencies only. It reads observed yaozarrs
axes/transforms before opening array handles; legacy reader defaults never enter
the PTV contract. Publishing metadata does not process or rewrite pixel arrays.
"""

import copy
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from ome_types import from_xml
from yaozarrs import open_group

from opm_processing.dataio.position_collection import update_group_attributes
from opm_processing.dataio.timing import hardware_volume_interval, read_time_interval

_LENGTH_TO_UM = {
    "meter": 1e6,
    "centimeter": 1e4,
    "millimeter": 1e3,
    "micrometer": 1.0,
    "nanometer": 1e-3,
    "picometer": 1e-6,
}
_STATUSES = {"valid", "acquired_empty", "invalid", "incomplete", "missing"}
_KINDS = {"deskewed", "deconvolved_deskewed", "fused_volume"}


@dataclass(frozen=True)
class ProcessedSnapshot:
    """Selected image handle and copied authoritative acquisition provenance."""

    array: Any
    metadata: dict[str, Any]


def _image_paths(root: Any) -> list[str]:
    """Resolve explicit image paths from the declared OME layout.

    Parameters
    ----------
    root : ZarrGroup
        Opened root image or position collection.

    Returns
    -------
    list of str
        Image group paths in stored series order.
    """
    metadata = root.ome_metadata()
    if hasattr(metadata, "multiscales"):
        return [""]
    return list(root["OME"].ome_metadata().series)


def _contract(root: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate self-contained PTV provenance and hardware timing.

    Parameters
    ----------
    root : ZarrGroup
        Opened processed OME-Zarr root.
    manifest : dict
        Recorded PTV acquisition and processing contract.

    Returns
    -------
    dict
        Validated producer hardware timing.
    """
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Missing supported self-contained opm_processing.ptv metadata")
    for name in ("dataset_id", "producer_version"):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise ValueError(f"PTV metadata requires {name}")
    if manifest.get("product_kind") not in _KINDS:
        raise ValueError(
            "PTV requires a full processed volume; raw/projection products rejected"
        )
    history = manifest.get("processing_history")
    if (
        not isinstance(history, list)
        or not history
        or any(not isinstance(row, dict) or not row for row in history)
    ):
        raise ValueError("PTV requires explicit processing history")
    if (
        not isinstance(manifest.get("registration"), dict)
        or type(manifest["registration"].get("applied")) is not bool
    ):
        raise ValueError("PTV requires explicit applied-registration provenance")
    if not isinstance(manifest.get("calibration"), dict) or not manifest["calibration"]:
        raise ValueError("PTV requires explicit calibration provenance")
    images = manifest.get("images")
    if (
        not isinstance(images, list)
        or any(not isinstance(image, dict) for image in images)
        or [image.get("path") for image in images] != _image_paths(root)
    ):
        raise ValueError("PTV image mapping must match the stored series order")
    timing = root.attrs.get("opm_processing", {}).get("timing", {})
    if timing.get("schema_version") != 1 or not timing.get("source"):
        raise ValueError("PTV requires recorded original hardware timing")
    interval = hardware_volume_interval(
        timing.get("n_acquired_planes"), timing.get("dt_plane_s")
    )
    if interval is None or not math.isclose(
        timing.get("dt_volume_s", float("nan")), interval, rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError("Producer hardware timing is missing or contradictory")
    return timing


def _snapshot(
    path: Path, root: Any, manifest: dict[str, Any], series: int, channel: int
) -> ProcessedSnapshot:
    """Open a selected processed image using its authoritative metadata.

    Parameters
    ----------
    path : pathlib.Path
        Processed OME-Zarr root path.
    root : ZarrGroup
        Opened root group.
    manifest : dict
        Validated self-contained PTV manifest.
    series : int
        Stored series index.
    channel : int
        Stored channel index.

    Returns
    -------
    ProcessedSnapshot
        Selected array handle and copied physical provenance.
    """
    timing = _contract(root, manifest)
    if (
        type(series) is not int
        or series < 0
        or series >= len(manifest["images"])
        or type(channel) is not int
        or channel < 0
    ):
        raise ValueError("Selected series/channel is outside the stored images")
    image_record = manifest["images"][series]
    image_path = image_record["path"]
    image_group = root if not image_path else root[image_path]
    scales = image_group.ome_metadata().multiscales
    if len(scales) != 1:
        raise ValueError("PTV requires one unambiguous multiscale image per series")
    multiscale = scales[0]
    if [axis.name for axis in multiscale.axes] != ["t", "c", "z", "y", "x"]:
        raise ValueError("PTV requires named TCZYX processed snapshot axes")
    if [axis.type for axis in multiscale.axes] != [
        "time",
        "channel",
        "space",
        "space",
        "space",
    ]:
        raise ValueError("Stored axis types disagree with processed TCZYX")
    factors = []
    for axis in multiscale.axes[2:]:
        if axis.unit not in _LENGTH_TO_UM:
            raise ValueError("Missing or unsupported observed spatial units")
        factors.append(_LENGTH_TO_UM[axis.unit])
    dt = hardware_volume_interval(timing["n_acquired_planes"], timing["dt_plane_s"])
    stored_dt = read_time_interval(multiscale)
    if stored_dt is None or not math.isclose(
        stored_dt, dt, rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError("Observed NGFF T scale/units contradict hardware cadence")
    datasets = multiscale.datasets
    spacings, origins = [], []
    for dataset in datasets:
        scale = dataset.scale_transform
        translation = dataset.translation_transform
        if (
            scale is None
            or translation is None
            or len(scale.scale) != 5
            or len(translation.translation) != 5
        ):
            raise ValueError(
                "Explicit observed TCZYX scale and translation are required"
            )
        if any(not math.isfinite(value) or value <= 0 for value in scale.scale) or any(
            not math.isfinite(value) for value in translation.translation
        ):
            raise ValueError("Stored transforms must be finite with positive scales")
        spacings.append(
            tuple(
                value * factor
                for value, factor in zip(scale.scale[2:], factors, strict=True)
            )
        )
        origins.append(
            tuple(
                value * factor
                for value, factor in zip(
                    translation.translation[2:], factors, strict=True
                )
            )
        )
    finest = [
        i
        for i, spacing in enumerate(spacings)
        if all(
            all(a <= b for a, b in zip(spacing, other, strict=True))
            for other in spacings
        )
    ]
    if len(finest) != 1:
        raise ValueError("Highest physical resolution must be unique and unambiguous")
    level = finest[0]
    array = image_group[datasets[level].path].to_tensorstore()
    if len(array.shape) != 5 or array.shape[2] <= 1 or channel >= array.shape[1]:
        raise ValueError("PTV requires full 3D TCZYX with an existing channel")
    indices = image_record.get("source_indices")
    statuses = image_record.get("volume_status_by_channel")
    if (
        not isinstance(indices, list)
        or len(indices) != array.shape[0]
        or any(type(k) is not int or k < 0 for k in indices)
        or any(b <= a for a, b in pairwise(indices))
    ):
        raise ValueError(
            "One increasing original source index is required per stored snapshot"
        )
    if (
        not isinstance(statuses, list)
        or len(statuses) != array.shape[1]
        or any(
            not isinstance(rows, list)
            or len(rows) != len(indices)
            or any(status not in _STATUSES for status in rows)
            for rows in statuses
        )
    ):
        raise ValueError(
            "Explicit valid/empty/invalid/incomplete/missing status is required per channel/snapshot"
        )
    channel_name = None
    xml_path = path / "OME/METADATA.ome.xml"
    if xml_path.is_file():
        ome = from_xml(xml_path.read_text(encoding="utf-8"))
        pixels = ome.images[series].pixels
        channel_name = pixels.channels[channel].name
        if pixels.time_increment is not None and (
            str(pixels.time_increment_unit.value) != "s"
            or not math.isclose(pixels.time_increment, dt, rel_tol=1e-10, abs_tol=1e-12)
        ):
            raise ValueError(
                "OME TimeIncrement contradicts observed hardware/NGFF timing"
            )
    normalized_timing = {
        name: timing[name] for name in ("n_acquired_planes", "dt_plane_s", "source")
    }
    for name in ("t_reference_s", "k_reference", "clock_precision_s"):
        if name in timing:
            normalized_timing[name] = timing[name]
    stored_times = image_record.get("stored_timestamps_s")
    if stored_times is not None:
        precision = timing.get("clock_precision_s")
        if (
            not isinstance(stored_times, list)
            or len(stored_times) != len(indices)
            or any(
                type(t) not in (int, float) or not math.isfinite(t)
                for t in stored_times
            )
            or any(b <= a for a, b in pairwise(stored_times))
            or type(precision) not in (int, float)
            or not math.isfinite(precision)
            or precision < 0
        ):
            raise ValueError(
                "Explicit snapshot timestamps require increasing finite values and clock precision"
            )
        reference_t = timing.get("t_reference_s", stored_times[0])
        reference_k = (
            timing.get("k_reference", indices[0])
            if "t_reference_s" in timing
            else indices[0]
        )
        if any(
            abs(t - (reference_t + (k - reference_k) * dt)) > precision
            for k, t in zip(indices, stored_times, strict=True)
        ):
            raise ValueError("Stored timestamps contradict original hardware cadence")
    metadata = {
        "schema_version": 1,
        "dataset_id": manifest["dataset_id"],
        "producer_version": manifest["producer_version"],
        "series": series,
        "channel": channel,
        "array_path": "/".join(
            part for part in (image_path, datasets[level].path) if part
        ),
        "shape_zyx": tuple(int(value) for value in array.shape[2:]),
        "spacing_zyx_um": spacings[level],
        "origin_zyx_um": origins[level],
        "timing": normalized_timing,
        "source_indices": indices,
        "volume_status": statuses[channel],
        "processing_history": manifest["processing_history"],
        "registration": manifest["registration"],
        "calibration": manifest["calibration"],
        "stored_ngff": {
            "time_unit": multiscale.axes[0].unit,
            "time_scale": datasets[level].scale_transform.scale[0],
            "channel_name": channel_name,
            "multiscale": multiscale.model_dump(mode="json"),
            "product_kind": manifest["product_kind"],
        },
    }
    if "stored_timestamps_s" in image_record:
        metadata["stored_timestamps_s"] = image_record["stored_timestamps_s"]
    return ProcessedSnapshot(array, copy.deepcopy(metadata))


def read_processed_snapshot(
    path: str | Path, *, series: int = 0, channel: int = 0
) -> ProcessedSnapshot:
    """Read authoritative metadata and open the highest-resolution processed array.

    Parameters
    ----------
    path : str or Path
        Self-contained processed OME-Zarr 0.5 image or position collection.
    series, channel : int
        Existing series and channel indices, without acquisition overrides.

    Returns
    -------
    ProcessedSnapshot
        Lazy TensorStore array and normalized copied evidence. No pixels are read.
    """
    path = Path(path).expanduser().resolve()
    root = open_group(path)
    manifest = root.attrs.get("opm_processing", {}).get("ptv")
    return _snapshot(path, root, manifest, series, channel)


def publish_ptv_metadata(path: str | Path, manifest: dict[str, Any]) -> None:
    """Validate and atomically embed explicit processing provenance without pixels.

    Parameters
    ----------
    path : str or Path
        Existing filesystem-backed processed OME-Zarr 0.5 output.
    manifest : dict
        Producer-owned version-one PTV history, calibration, registration and
        original source-index/status mapping. Timing must already be recorded.
    """
    path = Path(path).expanduser().resolve()
    root = open_group(path)
    _contract(root, manifest)
    for series, image in enumerate(manifest["images"]):
        for channel in range(max(1, len(image["volume_status_by_channel"]))):
            _snapshot(path, root, manifest, series, channel)
    processing = copy.deepcopy(root.attrs.get("opm_processing", {}))
    processing["ptv"] = manifest
    update_group_attributes(path, {"opm_processing": processing})


def publish_processing_state_ptv(path: str | Path, state: Any) -> None:
    """Embed current processing evidence and durable completion from its journal.

    Parameters
    ----------
    path : str or Path
        Processed collection recorded in the journal; timing must be known.
    state : ProcessingState
        Producer journal holding exact configuration and durable T/P/C writes.
        Intentionally zeroed processing channels are invalid, not acquired-empty.
    """
    from opm_processing import __version__

    path = Path(path).resolve()
    run = state.run(path)
    configuration = run.get("configuration")
    if not configuration or configuration.get("kind") not in _KINDS:
        raise ValueError("PTV export needs exact full-volume processing configuration")
    calibration = configuration.get("calibration")
    if not calibration:
        raise ValueError(
            "PTV export needs the actual applied camera/illumination calibration"
        )
    completed_tiles = state.completed_tiles(path)
    completed_channels = state.completed_channels(path)
    zero_channels = state.zero_channels(path)
    roi_series = state.roi_series(path)
    root = open_group(path)
    images = []
    for series, name in enumerate(_image_paths(root)):
        group = root if not name else root[name]
        dataset = group.ome_metadata().multiscales[0].datasets[0]
        shape = group[dataset.path].to_tensorstore().shape
        position = (
            series if not roi_series else int(roi_series[series]["position_index"])
        )
        indices = (
            list(range(shape[0]))
            if not roi_series
            else [int(roi_series[series]["time_index"])]
        )
        statuses = []
        for channel in range(shape[1]):
            rows = []
            for index in indices:
                key = (index, position, channel)
                complete = (
                    index,
                    position,
                ) in completed_tiles or key in completed_channels
                rows.append(
                    "incomplete"
                    if not complete
                    else "invalid"
                    if key in zero_channels
                    else "valid"
                )
            statuses.append(rows)
        images.append(
            {
                "path": name,
                "source_position_index": position,
                "source_indices": indices,
                "volume_status_by_channel": statuses,
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset_id": run["dataset_id"],
        "producer_version": __version__,
        "product_kind": configuration["kind"],
        "processing_history": [
            {
                "operation": configuration["kind"],
                "configuration": configuration,
                "configuration_sha256": run["configuration_sha256"],
            }
        ],
        "registration": {
            "applied": False,
            "coordinate_frame": "stored_processed",
            "deskew_applied": True,
        },
        "calibration": calibration,
        "images": images,
    }
    publish_ptv_metadata(path, manifest)


def mark_ptv_tile(
    path: str | Path,
    time_index: int,
    position_index: int,
    *,
    zero_channels: tuple[int, ...] = (),
) -> None:
    """Checkpoint self-contained status after every channel write is durable.

    Parameters
    ----------
    path : str or Path
        Processed collection with an already validated PTV header.
    time_index, position_index : int
        Original acquisition indices of the completed tile.
    zero_channels : tuple of int
        Deliberately zeroed processing channels, recorded as invalid.
    """
    processing = copy.deepcopy(open_group(path).attrs["opm_processing"])
    manifest = processing["ptv"]
    matched = False
    for image in manifest["images"]:
        if (
            image["source_position_index"] != position_index
            or time_index not in image["source_indices"]
        ):
            continue
        matched = True
        row = image["source_indices"].index(time_index)
        for channel, statuses in enumerate(image["volume_status_by_channel"]):
            statuses[row] = "invalid" if channel in zero_channels else "valid"
    if not matched:
        raise ValueError(
            "Completed tile does not correspond to the processed PTV mapping"
        )
    update_group_attributes(path, {"opm_processing": processing})
