"""Normalize old TPCZYX frame journals for a virtual OME-Zarr acquisition.

Legacy ``Stage.*_pos`` fields use micrometers. Their mapped values retain the
project's three-decimal stage precision. The complete journal is streamed once
to verify index coverage, calibration and recorded stage trajectories; only the
first two planes per channel and position are retained for the current reader.
The source metadata and image chunks are never modified.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

import ijson

if TYPE_CHECKING:
    from pathlib import Path
    from typing import BinaryIO


class _HashingReader:
    """Hash the same bytes consumed by the streaming JSON parser."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.digest.update(data)
        self.bytes_read += len(data)
        return data

    def readinto(self, target: Any) -> int:
        count = self.stream.readinto(target)
        if count:
            self.digest.update(memoryview(target)[:count])
            self.bytes_read += count
        return count


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    """Require a finite recorded calibration or stage value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing or invalid legacy {label}")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"Missing or invalid legacy {label}")
    return result


def _index(frame: dict[str, Any], shape: list[int]) -> tuple[int, int, int, int]:
    """Validate a legacy frame's complete logical T/P/C/Z address."""
    raw = frame.get("mda_event", {}).get("index", {})
    if not isinstance(raw, dict):
        raise ValueError("Invalid legacy frame index")
    values = []
    for axis, count in zip("tpcz", shape[:4], strict=True):
        value = raw.get(axis, 0 if count == 1 else None)
        if type(value) is not int or not 0 <= value < count:
            raise ValueError(f"Invalid legacy frame index {axis}={value!r}")
        values.append(value)
    return tuple(values)  # type: ignore[return-value]


def _frame_calibration(frame: dict[str, Any], shape: list[int]) -> dict[str, Any]:
    """Read recorded spatial, detector and channel calibration without defaults."""
    metadata = frame.get("mda_event", {}).get("metadata", {})
    daq, camera, opm = (metadata.get(name, {}) for name in ("DAQ", "Camera", "OPM"))
    if not all(isinstance(value, dict) for value in (daq, camera, opm)):
        raise ValueError("Invalid legacy instrument metadata")
    mode = str(daq.get("mode", "")).strip().lower()
    if mode != "stage":
        raise ValueError("The legacy reference mapper currently requires stage scans")
    pixel = _number(frame.get("pixel_size_um"), "pixel_size_um", positive=True)
    step = abs(_number(daq.get("scan_axis_step_um"), "scan_axis_step_um"))
    if step == 0:
        raise ValueError("Missing or invalid legacy scan_axis_step_um")
    angle = _number(opm.get("angle_deg"), "angle_deg", positive=True)
    if angle >= 90:
        raise ValueError("Legacy angle_deg must lie between zero and 90 degrees")
    for axis, size in (("x", shape[-1]), ("y", shape[-2])):
        if camera.get(f"camera_crop_{axis}") != size:
            raise ValueError(
                f"Legacy Camera.camera_crop_{axis} disagrees with raw shape"
            )
    states = daq.get("active_channels")
    powers = daq.get("laser_powers")
    exposures = daq.get("exposure_channels_ms")
    if (
        not isinstance(states, list)
        or not states
        or any(type(state) is not bool for state in states)
        or sum(states) != shape[2]
    ):
        raise ValueError("Legacy DAQ.active_channels disagrees with stored channels")
    for label, values in (
        ("laser_powers", powers),
        ("exposure_channels_ms", exposures),
    ):
        if not isinstance(values, list) or len(values) != len(states):
            raise ValueError(f"Legacy DAQ.{label} does not match active_channels")
        for value in values:
            _number(value, f"DAQ.{label}", positive=label == "exposure_channels_ms")
    orientations = {}
    for key in ("camera_Zstage_orientation", "camera_XYstage_orientation"):
        if opm.get(key) not in ("positive", "negative"):
            raise ValueError(f"Missing or invalid legacy OPM.{key}")
        orientations[key] = opm[key]
    for key in (
        "excess_scan_positions",
        "excess_scan_start_positions",
        "excess_scan_end_positions",
    ):
        if key in opm and (type(opm[key]) is not int or not 0 <= opm[key] < shape[3]):
            raise ValueError(f"Invalid legacy OPM.{key}")
    return {
        "mode": mode,
        "pixel_size_um": pixel,
        "scan_axis_step_um": step,
        "angle_deg": angle,
        "camera_offset": _number(camera.get("offset"), "Camera.offset"),
        "camera_conversion": _number(
            camera.get("e_to_ADU"), "Camera.e_to_ADU", positive=True
        ),
        "active_channels": states,
        "laser_powers": powers,
        "exposure_channels_ms": exposures,
        "orientations": orientations,
        "excess_scan_counts": {
            key: value for key, value in opm.items() if key.startswith("excess_scan_")
        },
    }


def _acquisition_order(shape: list[int], strides: dict[str, int]) -> list[str]:
    """Infer moving axes from frame order, placing singleton axes canonically."""
    canonical = "tpcz"
    order = sorted(strides, key=strides.__getitem__, reverse=True)
    for axis, size in zip(canonical, shape[:4], strict=True):
        if size != 1:
            if axis not in strides:
                raise ValueError(f"Cannot infer legacy acquisition order for {axis}")
            continue
        insertion = next(
            (
                i
                for i, other in enumerate(order)
                if canonical.index(other) > canonical.index(axis)
            ),
            len(order),
        )
        order.insert(insertion, axis)
    return order


def read_legacy_metadata(raw_path: Path) -> dict[str, Any]:
    """Validate legacy raw metadata and return the virtual OME-Zarr header recipe.

    Parameters
    ----------
    raw_path : pathlib.Path
        Existing Zarr v2 TPCZYX array containing ``.zattrs.frame_metadatas``.

    Returns
    -------
    dict
        Ordered initial frame sets, channel names, normalized root attributes,
        native sampling and shape, and a source audit report. No independent
        hardware plane period is asserted; recorded exposure fields remain intact.

    Raises
    ------
    ValueError
        Metadata is incomplete, contradictory, not a supported stage scan, or
        contains multiple timepoints, which this initial mapper does not support.
    """
    raw_path = raw_path.expanduser().absolute()
    descriptor_bytes = (raw_path / ".zarray").read_bytes()
    descriptor = json.loads(descriptor_bytes)
    shape = descriptor.get("shape")
    if (
        descriptor.get("zarr_format") != 2
        or not isinstance(shape, list)
        or len(shape) != 6
        or any(type(size) is not int or size < 1 for size in shape)
    ):
        raise ValueError("Expected a legacy Zarr v2 TPCZYX raw array")
    if shape[0] != 1:
        raise ValueError("Legacy reference mapping currently requires one timepoint")
    if shape[3] < 2:
        raise ValueError("Legacy stage scan metadata needs at least two scan planes")
    _, positions, channels, planes, _, _ = shape
    expected = math.prod(shape[:4])
    present = bytearray(expected)
    retained: dict[tuple[int, int, int], dict[str, Any]] = {}
    starts: dict[tuple[int, int], tuple[float, float, float]] = {}
    steps: dict[tuple[int, int], tuple[float, float, float]] = {}
    ends: dict[tuple[int, int], tuple[float, float, float]] = {}
    channel_names: dict[int, str] = {}
    channel_exposures: dict[int, tuple[float | None, float]] = {}
    strides: dict[str, int] = {}
    previous_z: dict[tuple[int, int], int] = {}
    calibration = None
    maximum_stage_rounding_um = 0.0
    count = 0
    attributes_path = raw_path / ".zattrs"
    before = attributes_path.stat()
    with attributes_path.open("rb") as stream:
        reader = _HashingReader(stream)
        for ordinal, frame in enumerate(
            ijson.items(reader, "frame_metadatas.item", use_float=True)
        ):
            if not isinstance(frame, dict):
                raise ValueError("Legacy frame metadata must contain objects")
            t, p, c, z = _index(frame, shape)
            flat = ((t * positions + p) * channels + c) * planes + z
            if present[flat]:
                raise ValueError(f"Duplicate legacy frame index {(t, p, c, z)}")
            present[flat] = 1
            count += 1
            current = _frame_calibration(frame, shape)
            if calibration is None:
                calibration = current
            elif calibration != current:
                raise ValueError(
                    f"Legacy acquisition calibration changes at {(p, c, z)}"
                )
            index_values = (t, p, c, z)
            if sum(value != 0 for value in index_values) == 1:
                for axis, value in zip("tpcz", index_values, strict=True):
                    if value == 1:
                        strides.setdefault(axis, ordinal)
            metadata = frame["mda_event"]["metadata"]
            name = metadata["DAQ"].get("current_channel")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Missing legacy DAQ.current_channel")
            if c in channel_names and channel_names[c] != name:
                raise ValueError(f"Legacy channel label changes for stored channel {c}")
            channel_names[c] = name
            camera_exposure = _number(
                metadata["Camera"].get("exposure_ms"),
                "Camera.exposure_ms",
                positive=True,
            )
            hardware_exposure = (
                _number(frame["exposure_ms"], "exposure_ms", positive=True)
                if frame.get("exposure_ms") is not None
                else None
            )
            exposure_pair = hardware_exposure, camera_exposure
            if c in channel_exposures and channel_exposures[c] != exposure_pair:
                raise ValueError(f"Legacy exposure changes for stored channel {c}")
            channel_exposures[c] = exposure_pair
            stage = metadata.get("Stage", {})
            if not isinstance(stage, dict):
                raise ValueError("Missing legacy Stage metadata")
            xyz_raw = tuple(
                _number(stage.get(f"{a}_pos"), f"Stage.{a}_pos") for a in "xyz"
            )
            xyz = tuple(round(value, 3) for value in xyz_raw)
            maximum_stage_rounding_um = max(
                maximum_stage_rounding_um,
                *(abs(a - b) for a, b in zip(xyz_raw, xyz, strict=True)),
            )
            for axis, value in zip("xyz", xyz, strict=True):
                stage[f"{axis}_pos"] = value
            key = p, c
            if z != previous_z.get(key, -1) + 1:
                raise ValueError(
                    "Legacy scan records must increase Z within each position/channel"
                )
            previous_z[key] = z
            if z == 0:
                starts[key] = xyz
            elif z == 1:
                step = tuple(
                    round(b - a, 3) for a, b in zip(starts[key], xyz, strict=True)
                )
                if step[0] == 0 or step[1:] != (0.0, 0.0):
                    raise ValueError("Legacy stage scans must move along stage X only")
                if not math.isclose(
                    abs(step[0]),
                    current["scan_axis_step_um"],
                    rel_tol=1e-3,
                    abs_tol=1e-6,
                ):
                    raise ValueError(
                        "Recorded legacy stage step disagrees with scan_axis_step_um"
                    )
                steps[key] = step
            else:
                expected_xyz = tuple(
                    round(start + z * delta, 3)
                    for start, delta in zip(starts[key], steps[key], strict=True)
                )
                if xyz != expected_xyz:
                    raise ValueError(
                        f"Nonlinear legacy stage trajectory at {(p, c, z)}"
                    )
            if z < 2:
                retained[p, c, z] = frame
            if z == planes - 1:
                ends[key] = xyz
        # Consume any buffered remainder through the hashing wrapper too.
        while reader.read(65536):
            pass
    after = attributes_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Legacy metadata changed during reference mapping")
    if count != expected or not all(present):
        raise ValueError(f"Incomplete legacy frame coverage: {count}/{expected}")
    assert calibration is not None
    if len({steps[p, c] for p in range(positions) for c in range(channels)}) != 1:
        raise ValueError(
            "Legacy stage scans have inconsistent scan steps or directions"
        )
    for p in range(positions):
        if len({starts[p, c] for c in range(channels)}) != 1:
            raise ValueError("Legacy channels disagree on a position's stage origin")
    order = _acquisition_order(shape, strides)
    daq_config = {
        "channel_states": calibration["active_channels"],
        "channel_powers": calibration["laser_powers"],
        "channel_exposures_ms": calibration["exposure_channels_ms"],
        "scan_axis_step_um": calibration["scan_axis_step_um"],
    }
    trajectories = [
        {
            "p": p,
            "start_xyz_um": list(starts[p, 0]),
            "second_xyz_um": [
                retained[p, 0, 1]["mda_event"]["metadata"]["Stage"][f"{a}_pos"]
                for a in "xyz"
            ],
            "end_xyz_um": list(ends[p, 0]),
            "step_xyz_um": list(steps[p, 0]),
        }
        for p in range(positions)
    ]
    return {
        "frame_sets": [
            [retained[p, c, z] for z in range(2) for c in range(channels)]
            for p in range(positions)
        ],
        "channel_names": [channel_names[c] for c in range(channels)],
        "root_attributes": {
            "opm_v2": {
                "configuration": {
                    "acq_config": {"opm_mode": calibration["mode"], "DAQ": daq_config}
                },
                "acquisition_order": order,
            }
        },
        "sampling_tczyx": [
            1.0,
            1.0,
            calibration["scan_axis_step_um"],
            calibration["pixel_size_um"],
            calibration["pixel_size_um"],
        ],
        "shape_tpczyx": shape,
        "report": {
            "schema_version": 1,
            "source": str(raw_path),
            "source_metadata_sha256": {
                ".zarray": hashlib.sha256(descriptor_bytes).hexdigest(),
                ".zattrs": reader.digest.hexdigest(),
            },
            "metadata_bytes_read": reader.bytes_read,
            "frame_count": count,
            "coverage_complete": True,
            "all_frame_trajectories_validated": True,
            "calibration": calibration,
            "channel_exposures_ms": {
                str(c): {"hardware": pair[0], "camera_setting": pair[1]}
                for c, pair in channel_exposures.items()
            },
            "stage_coordinate_unit": "micrometer; legacy Stage.*_pos schema",
            "stage_coordinate_decimals": 3,
            "maximum_stage_rounding_um": maximum_stage_rounding_um,
            "stage_trajectories": trajectories,
            "acquisition_order": order,
            "acquisition_order_source": "inferred from frame index sequence; singleton axes placed in canonical TPCZ order",
            "axis_stride_frames": strides,
            "timing": {
                "independent_plane_period_recorded": False,
                "hardware_exposure_fields_preserved": True,
            },
            "source_pixels_modified": False,
            "source_metadata_modified": False,
        },
    }
