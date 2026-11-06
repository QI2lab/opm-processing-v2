"""Legacy acquisition metadata is mapped without inventing geometry or pixels."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from opm_processing.dataio.acquisition import (
    _channel_metadata,
    _positions_from_frame_sets,
)
from opm_processing.dataio.legacy_metadata import read_legacy_metadata


@pytest.mark.unit
def test_legacy_frame_mapping_preserves_calibration_geometry_and_source(
    legacy_raw_factory,
):
    """Correctly label stored C0 from enabled hardware channel 1, without a copy."""
    case = legacy_raw_factory()
    raw, original_frames = case.path, case.frames
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
    np.testing.assert_array_equal(case.array.read().result(), case.data)


@pytest.mark.unit
@pytest.mark.parametrize("interleaved", [False, True])
def test_legacy_mapping_infers_actual_multichannel_acquisition_order(
    legacy_raw_factory, interleaved
):
    """Preserve signed scan motion and derive moving-axis order from event order."""
    raw = legacy_raw_factory(channels=2, direction=-1, interleaved=interleaved).path
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
    legacy_raw_factory, mutation, message
):
    """Do not silently publish a plausible but incorrect acquisition mapping."""
    case = legacy_raw_factory()
    raw, frames = case.path, case.frames
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
def test_legacy_mapping_rejects_unimplemented_multiple_timepoints(legacy_raw_factory):
    """Stop explicitly instead of assigning one timepoint's geometry to another."""
    raw = legacy_raw_factory().path
    descriptor = json.loads((raw / ".zarray").read_text())
    descriptor["shape"][0] = 2
    (raw / ".zarray").write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match="requires one timepoint"):
        read_legacy_metadata(raw)
