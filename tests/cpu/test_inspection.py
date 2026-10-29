"""Analytical metadata units and CLI inspection of simulated OPM acquisitions."""

import json

import numpy as np
import pytest
import zarr
from yaozarrs import open_group

from opm_processing.dataio.acquisition import inspect_acquisition
from opm_processing.dataio.inspection import (
    hardware_timing,
    inspect_dataset,
    processed_report,
    raw_report,
)


@pytest.mark.unit
@pytest.mark.parametrize("field,factor", [("exposure_ms", 1), ("exposure_time", 1000)])
@pytest.mark.parametrize("planes", [1, 17])
def test_hardware_timing_uses_camera_exposures(field, factor, planes):
    """Sum unequal camera exposures once per stored channel and scan position."""
    frames = [
        {
            "event_index": {"c": channel},
            field: exposure / factor,
            "runner_time_ms": 123456,
            "event_metadata": {"Camera": {"exposure_ms": 999}},
        }
        for channel, exposure in enumerate((5.002102941, 15.0))
        for _ in range(2)
    ]
    assert hardware_timing(frames, 2, planes) == {
        "frame_exposure_ms": [5, 15],
        "volume_time_ms": 20 * planes,
    }


@pytest.mark.unit
@pytest.mark.parametrize("exposure", [None, -1, 0, float("nan"), True, "invalid"])
def test_missing_or_invalid_hardware_exposure_is_not_inferred(exposure):
    """Keep unknown camera timing unknown and reject nonphysical exposures."""
    frames = [
        {"mda_event": {"index": {"c": 0}, "metadata": {"Camera": {"exposure_ms": 10}}}}
    ]
    if exposure is None:
        assert hardware_timing(frames, 1, 17) == {
            "frame_exposure_ms": [None],
            "volume_time_ms": None,
        }
    else:
        frames[0]["exposure_ms"] = exposure
        with pytest.raises(ValueError, match="exposure"):
            hardware_timing(frames, 1, 17)


@pytest.mark.unit
@pytest.mark.parametrize(
    "unit,factor", [("micrometer", 1), ("nanometer", 1000), (None, None)]
)
@pytest.mark.parametrize("global_transform", [False, True])
def test_processed_volume_dimensions_use_recorded_lab_grid(
    fused_store_factory, unit, factor, global_transform
):
    """Report known voxel extents and composed transforms without reading pixels."""
    path = fused_store_factory(
        np.arange(60, dtype=np.uint16).reshape(1, 1, 3, 4, 5),
        spacing=(0.23, 0.115, 0.115),
        origin=(1, 2, 3),
        factors=(1, 2),
    )
    ome = dict(open_group(path).attrs["ome"])
    multiscale = ome["multiscales"][0]
    for axis in multiscale["axes"][2:]:
        if unit is None:
            axis.pop("unit", None)
        else:
            axis["unit"] = unit
    for dataset in multiscale["datasets"]:
        for transform in dataset["coordinateTransformations"]:
            key = transform["type"]
            transform[key][2:] = [value * (factor or 1) for value in transform[key][2:]]
    if global_transform:
        multiscale["coordinateTransformations"] = [
            {"type": "scale", "scale": [1, 1, 2, 2, 2]},
            {"type": "translation", "translation": [0, 0, *([5 * (factor or 1)] * 3)]},
        ]
    zarr.open_group(path, mode="r+").attrs.update({"ome": ome})
    image = processed_report(path)["images"][0]
    gain = 2 if global_transform else 1
    for axis, size, spacing, origin in zip(
        "zyx", (3, 4, 5), (0.23, 0.115, 0.115), (1, 2, 3), strict=True
    ):
        if unit is None:
            assert image["sampling_um"][axis] is None
            assert image["volume_dimensions_um"][axis] is None
        else:
            assert image["sampling_um"][axis] == pytest.approx(gain * spacing)
            assert image["volume_dimensions_um"][axis] == pytest.approx(
                size * gain * spacing
            )
            assert image["origin_um"][axis] == pytest.approx(
                gain * origin + (5 if global_transform else 0)
            )
    assert image["pyramid_levels"] == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    "selection",
    ["raw-only", "directory", "raw", "processed", "custom-processed", "processed-only"],
)
def test_inspect_cli_reports_simulated_raw_and_processed_data(
    inspection_case,
    selection,
    invoke_cli,
):
    """Inspect real processing products of an optically simulated oblique object."""
    case, collection = inspection_case.case, inspection_case.collection
    if collection is not None:
        assert any(np.any(array.read().result() > 0) for array in collection.arrays)
    report = json.loads(invoke_cli("inspect-opm", inspection_case.selected).output)
    if selection == "processed-only":
        assert report["raw"] is None
    else:
        raw = report["raw"]
        assert raw["axes"] == ["t", "p", "c", "s", "y'", "x"]
        assert raw["index_sizes"]["s"] == case.raw_data.shape[3]
        assert "z" not in raw["index_sizes"]
        assert raw["sampling_um"] == {
            "s": case.metadata.scan_axis_step_um,
            "y'": case.metadata.pixel_size_um,
            "x": case.metadata.pixel_size_um,
        }
        assert raw["y_prime_sampling_projected_y_um"] == pytest.approx(
            case.metadata.pixel_size_um * np.sqrt(3) / 2
        )
        assert raw["timing"] == {
            "frame_exposure_ms": [10],
            "volume_time_ms": 10 * case.raw_data.shape[3],
        }
    assert len(report["processed"]) == int(selection != "raw-only")
    if collection is not None:
        assert report["processed"][0]["timing"] == (
            None if selection == "processed-only" else report["raw"]["timing"]
        )
        assert len(report["processed"][0]["images"]) == case.raw_data.shape[1]
        for image, array in zip(
            report["processed"][0]["images"], collection.arrays, strict=True
        ):
            assert image["axes"] == list("tczyx")
            assert image["shape"] == list(array.shape)
            assert image["channels"] == list(case.metadata.channel_names)
            for axis, size in zip("zyx", array.shape[-3:], strict=True):
                assert image["volume_dimensions_um"][axis] == pytest.approx(
                    size * case.metadata.pixel_size_um
                )


@pytest.mark.unit
def test_missing_dataset_is_reported(tmp_path):
    """Report absent input instead of inventing acquisition metadata."""
    with pytest.raises(ValueError, match="No raw or processed"):
        inspect_dataset(tmp_path)


@pytest.mark.unit
def test_raw_report_prefers_actual_camera_exposures(current_opm_v2_stage_scan):
    """Ignore stale requested exposure and file-writing timestamps in raw timing."""
    path = current_opm_v2_stage_scan
    for position in ("0", "1"):
        group = zarr.open_group(path / position, mode="r+")
        writer = dict(group.attrs["ome_writers"])
        for frame in writer["frame_metadata"]:
            frame["event_metadata"]["Camera"]["exposure_ms"] = 999
            frame["delta_t"] = 123456
        group.attrs.update({"ome_writers": writer})
    report = raw_report(inspect_acquisition(path))
    assert [channel["exposure_ms"] for channel in report["channels"]] == [10, 15]
    assert report["timing"] == {"frame_exposure_ms": [10, 15], "volume_time_ms": 75}
    assert report["camera_field_of_view_um"] == {"y'": 4 * 0.115, "x": 5 * 0.115}


@pytest.mark.unit
@pytest.mark.parametrize(
    "problem", ["missing-channel", "varying-exposure", "out-of-range-channel"]
)
def test_hardware_timing_requires_consistent_channel_measurements(problem):
    """Reject contradictory records and leave incomplete channel timing unknown."""
    frames = [{"event_index": {"c": 0}, "exposure_ms": 5}]
    if problem == "missing-channel":
        assert hardware_timing(frames, 2, 17) == {
            "frame_exposure_ms": [5, None],
            "volume_time_ms": None,
        }
    else:
        frames.append(
            {
                "event_index": {"c": 2 if problem == "out-of-range-channel" else 0},
                "exposure_ms": 10,
            }
        )
        with pytest.raises(ValueError, match=r"channel|exposures vary"):
            hardware_timing(frames, 2, 17)
