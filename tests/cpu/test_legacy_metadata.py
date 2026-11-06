"""Legacy acquisition metadata is mapped without inventing geometry or pixels."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from opm_processing.dataio.acquisition import (
    _channel_metadata,
    _positions_from_frame_sets,
)
from opm_processing.dataio.legacy_metadata import read_legacy_metadata

if TYPE_CHECKING:
    from pathlib import Path


def _write_raw(
    path: Path, *, channels: int = 1, direction: int = 1, interleaved: bool = False
) -> tuple[Path, list[dict]]:
    """Write only a small legacy journal plus an untouched pixel sentinel."""
    path.mkdir()
    shape = [1, 2, channels, 3, 4, 5]
    states = [False, True, False, channels == 2, False]
    names = ["488nm", "561nm"][:channels]
    descriptor = {
        "zarr_format": 2,
        "shape": shape,
        "chunks": [1, 1, 1, 1, 4, 5],
        "dtype": "<u2",
        "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": -1},
        "fill_value": None,
        "order": "C",
        "filters": None,
        "dimension_separator": "/",
    }
    (path / ".zarray").write_text(json.dumps(descriptor))
    frames = []
    for p in range(2):
        pairs = (
            [(c, z) for z in range(3) for c in range(channels)]
            if interleaved
            else [(c, z) for c in range(channels) for z in range(3)]
        )
        for c, z in pairs:
            frames.append(
                {
                    "format": "frame-dict",
                    "exposure_ms": (10.0303823529, 15.02)[c],
                    "pixel_size_um": 0.115,
                    "mda_event": {
                        "index": {"t": 0, "p": p, "c": c, "z": z},
                        "metadata": {
                            "DAQ": {
                                "mode": "stage",
                                "scan_axis_step_um": 0.4,
                                "active_channels": states,
                                "laser_powers": [0, 20, 0, 35, 0],
                                "exposure_channels_ms": [5, 10.03, 5, 15.02, 5],
                                "current_channel": names[c],
                            },
                            "Camera": {
                                "exposure_ms": (10.03, 15.02)[c],
                                "camera_crop_x": 5,
                                "camera_crop_y": 4,
                                "offset": 100,
                                "e_to_ADU": 0.24,
                            },
                            "OPM": {
                                "angle_deg": 30,
                                "camera_Zstage_orientation": "negative",
                                "camera_XYstage_orientation": "positive",
                                "camera_mirror_orientation": "positive",
                                "excess_scan_positions": 0,
                            },
                            "Stage": {
                                "x_pos": 100.123456 + p * 20 + direction * z * 0.4,
                                "y_pos": 200.987654,
                                "z_pos": 30.123456,
                            },
                        },
                    },
                }
            )
    (path / ".zattrs").write_text(json.dumps({"frame_metadatas": frames}))
    (path / "pixel_sentinel").write_bytes(b"original compressed pixels")
    return path, frames


@pytest.mark.unit
def test_legacy_frame_mapping_preserves_calibration_geometry_and_source(tmp_path: Path):
    """Correctly label stored C0 from enabled hardware channel 1, without a copy."""
    raw, original_frames = _write_raw(tmp_path / "legacy.zarr")
    original_attrs = (raw / ".zattrs").read_bytes()
    result = read_legacy_metadata(raw)
    assert result["shape_tpczyx"] == [1, 2, 1, 3, 4, 5]
    assert result["sampling_tczyx"] == [1, 1, 0.4, 0.115, 0.115]
    assert result["channel_names"] == ["488nm"]
    assert result["root_attributes"]["opm_v2"]["acquisition_order"] == [
        "t",
        "p",
        "c",
        "z",
    ]
    config = result["root_attributes"]["opm_v2"]["configuration"]["acq_config"]["DAQ"]
    enabled_powers = [
        v
        for v, enabled in zip(
            config["channel_powers"], config["channel_states"], strict=True
        )
        if enabled
    ]
    mapped_channels = _channel_metadata(
        result["channel_names"], result["frame_sets"][0], enabled_powers
    )
    assert mapped_channels[0].laser_power == 20
    assert mapped_channels[0].exposure_ms == 10.03
    positions, starts, ends, axis, step = _positions_from_frame_sets(
        result["frame_sets"], scan_position_count=3
    )
    assert positions == ((30.123, 100.123, 200.988), (30.123, 120.123, 200.988))
    assert starts[0] == (100.123, 200.988, 30.123)
    assert ends[0] == pytest.approx((100.923, 200.988, 30.123))
    assert axis == "x" and step == pytest.approx(0.4)
    assert (
        result["frame_sets"][0][0]["exposure_ms"] == original_frames[0]["exposure_ms"]
    )
    assert result["report"]["timing"] == {
        "independent_plane_period_recorded": False,
        "hardware_exposure_fields_preserved": True,
    }
    assert result["report"]["frame_count"] == 6
    assert (
        result["report"]["source_metadata_sha256"][".zattrs"]
        == hashlib.sha256(original_attrs).hexdigest()
    )
    assert result["report"]["metadata_bytes_read"] == len(original_attrs)
    assert (raw / ".zattrs").read_bytes() == original_attrs
    assert (raw / "pixel_sentinel").read_bytes() == b"original compressed pixels"


@pytest.mark.unit
@pytest.mark.parametrize("interleaved", [False, True])
def test_legacy_mapping_infers_actual_multichannel_acquisition_order(
    tmp_path: Path, interleaved
):
    """Preserve signed scan motion and derive moving-axis order from event order."""
    raw, _ = _write_raw(
        tmp_path / "legacy.zarr", channels=2, direction=-1, interleaved=interleaved
    )
    result = read_legacy_metadata(raw)
    order = ["t", "p", "z", "c"] if interleaved else ["t", "p", "c", "z"]
    assert result["root_attributes"]["opm_v2"]["acquisition_order"] == order
    assert result["channel_names"] == ["488nm", "561nm"]
    assert len(result["frame_sets"]) == 2
    assert [f["mda_event"]["index"] for f in result["frame_sets"][0]] == [
        {"t": 0, "p": 0, "c": c, "z": z} for z in range(2) for c in range(2)
    ]
    assert result["report"]["stage_trajectories"][0]["step_xyz_um"] == [-0.4, 0, 0]
    assert result["report"]["stage_trajectories"][0]["end_xyz_um"] == [
        99.323,
        200.988,
        30.123,
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "Incomplete legacy frame coverage"),
        ("duplicate", "Duplicate legacy frame index"),
        ("out_of_range", "Invalid legacy frame index"),
        ("calibration", "calibration changes"),
        ("missing_calibration", "Camera.offset"),
        ("trajectory", "Nonlinear legacy stage trajectory"),
        ("channel_count", "active_channels disagrees"),
    ],
)
def test_legacy_mapping_rejects_metadata_that_cannot_describe_existing_pixels(
    tmp_path: Path, mutation, message
):
    """Do not silently publish a plausible but incorrect acquisition mapping."""
    raw, frames = _write_raw(tmp_path / "legacy.zarr")
    if mutation == "missing":
        frames.pop()
    elif mutation == "duplicate":
        frames.append(copy.deepcopy(frames[0]))
    elif mutation == "out_of_range":
        frames[-1]["mda_event"]["index"]["p"] = 2
    elif mutation == "calibration":
        frames[-1]["mda_event"]["metadata"]["Camera"]["offset"] += 1
    elif mutation == "missing_calibration":
        del frames[0]["mda_event"]["metadata"]["Camera"]["offset"]
    elif mutation == "trajectory":
        frames[-1]["mda_event"]["metadata"]["Stage"]["x_pos"] += 0.01
    elif mutation == "channel_count":
        frames[0]["mda_event"]["metadata"]["DAQ"]["active_channels"] = [False] * 5
    (raw / ".zattrs").write_text(json.dumps({"frame_metadatas": frames}))
    with pytest.raises(ValueError, match=message):
        read_legacy_metadata(raw)


@pytest.mark.unit
def test_legacy_mapping_rejects_unimplemented_multiple_timepoints(tmp_path: Path):
    """Stop explicitly instead of assigning one timepoint's geometry to another."""
    raw, _ = _write_raw(tmp_path / "legacy.zarr")
    descriptor = json.loads((raw / ".zarray").read_text())
    descriptor["shape"][0] = 2
    (raw / ".zarray").write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match="requires one timepoint"):
        read_legacy_metadata(raw)
