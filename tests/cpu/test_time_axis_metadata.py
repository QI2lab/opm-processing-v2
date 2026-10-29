"""Physical time-axis units and hardware cadence through the actual producer writer."""

from dataclasses import replace

import pytest
from yaozarrs import open_group

from opm_processing.dataio.acquisition import _camera_timing, inspect_acquisition
from opm_processing.dataio.position_collection import (
    create_position_collection,
    create_variable_position_collection,
    open_position_collection,
    update_group_attributes,
)


@pytest.mark.unit
@pytest.mark.parametrize("cropped", [False, True])
def test_written_time_axis_interval_and_units(tmp_path, cropped):
    """Write the measured temporal interval with explicit NGFF units."""
    path = tmp_path / "timed.ome.zarr"
    if cropped:
        create_variable_position_collection(
            path,
            [(3, 1, 4, 8, 8)],
            (0.23, 0.115, 0.115),
            spatial_origins_zyx_um=[(0, 0, 0)],
            time_interval_s=0.085,
        )
    else:
        create_position_collection(
            path,
            (3, 1, 1, 128, 8, 8),
            (0.23, 0.115, 0.115),
            multiscale_factors_yx=(2,),
            time_interval_s=0.085,
        )
    image = open_group(path)["0"].ome_metadata()
    assert image.multiscales[0].axes[0].unit == "second"
    assert all(
        dataset.scale_transform.scale[0] == 0.085
        for dataset in image.multiscales[0].datasets
    )
    assert open_position_collection(path).time_interval_s == 0.085


@pytest.mark.unit
def test_hardware_cadence_uses_acquired_count(current_single_position_mirror_timelapse):
    """Compute volume cadence from the acquired plane count and trigger period."""
    source = inspect_acquisition(current_single_position_mirror_timelapse)
    acquired = replace(source, acquired_plane_count=17, plane_period_s=0.005)
    assert acquired.volume_interval_s == pytest.approx(0.085)
    assert acquired.scan_position_count != 17
    assert (
        replace(
            source, acquired_plane_count=None, plane_period_s=None
        ).volume_interval_s
        is None
    )


@pytest.mark.unit
@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_time_scale_rejected(tmp_path, interval):
    """Reject nonphysical recorded time scales."""
    with pytest.raises(ValueError, match="time_interval_s"):
        create_position_collection(
            tmp_path / "invalid.ome.zarr",
            (2, 1, 1, 4, 8, 8),
            (0.23, 0.115, 0.115),
            time_interval_s=interval,
        )


@pytest.mark.unit
def test_unknown_time_does_not_become_one_second(tmp_path):
    """Preserve unknown timing without inventing a one-second interval."""
    path = tmp_path / "unknown.ome.zarr"
    create_position_collection(path, (2, 1, 1, 4, 8, 8), (0.23, 0.115, 0.115))
    assert open_group(path)["0"].ome_metadata().multiscales[0].axes[0].unit is None
    assert open_position_collection(path).time_interval_s is None


@pytest.mark.integration
def test_processing_propagates_hardware_cadence(
    calibrated_mirror_timelapse, invoke_cli
):
    """Preserve hardware cadence through processing and projection output."""
    raw = calibrated_mirror_timelapse.path
    invoke_cli(
        "process",
        raw,
        max_projection=True,
        create_fused_max_projection=False,
        z_downsample_level=1,
    )
    for name in (
        "single_mirror_deskewed.ome.zarr",
        "single_mirror_max_z_deskewed.ome.zarr",
    ):
        result = open_position_collection(raw.parent / name)
        import numpy as np

        for time, expected in enumerate(calibrated_mirror_timelapse.expected):
            expected = expected.astype(np.uint16)
            if "max_z" in name:
                expected = expected.max(axis=0, keepdims=True)
            np.testing.assert_array_equal(
                result.arrays[0][time, 0].read().result(), expected
            )
        assert result.time_interval_s == pytest.approx(0.020)
        timing = result.attributes["opm_processing"]["timing"]
        assert timing["n_acquired_planes"] == 4
        assert timing["dt_plane_s"] == 0.005
        assert timing["dt_volume_s"] == 0.020
        assert timing["source"] == "recorded camera exposure_time; rounded to 0.01 ms"


@pytest.mark.unit
def test_confirmed_period_cannot_override_recorded_timing(
    current_single_position_mirror_timelapse,
):
    """Reject a supplied period that contradicts recorded hardware timing."""
    acquired = inspect_acquisition(current_single_position_mirror_timelapse)
    confirmed = acquired.with_plane_period(0.005)
    assert confirmed.acquired_plane_count == acquired.scan_position_count
    assert confirmed.with_plane_period(None) is confirmed
    assert (
        confirmed.with_plane_period(0.005).volume_interval_s
        == confirmed.volume_interval_s
    )
    with pytest.raises(ValueError, match="contradicts"):
        confirmed.with_plane_period(0.006)
    with pytest.raises(ValueError, match="time_interval_s"):
        acquired.with_plane_period(float("nan"))


@pytest.mark.unit
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("planes", [1, 5])
@pytest.mark.parametrize("field", ["exposure_time", "exposure_ms"])
def test_hardware_exposures_define_cadence(
    acquisition_factory, channels, planes, field
):
    """Use actual camera exposures and retain raw discarded scan planes."""
    import numpy as np

    dataset = acquisition_factory(
        np.zeros((2, 2, channels, planes, 4, 6), np.uint16),
        acquired_plane_count=planes,
        plane_period_s=0.03,
        excess_scan_start_positions=int(planes > 1),
        excess_scan_end_positions=int(planes > 1),
    )
    root = open_group(dataset.path)
    assert "timing" not in root.attrs["opm_v2"]
    for position in range(2):
        writer = dict(root[str(position)].attrs["ome_writers"])
        for frame in writer["frame_metadata"]:
            if field == "exposure_time":
                frame[field] = frame.pop("exposure_ms") / 1000
            frame["delta_t"] = -123.456
            frame["camera_metadata"] = {"ElapsedTime-ms": -1}
        update_group_attributes(dataset.path / str(position), {"ome_writers": writer})
    acquired = inspect_acquisition(dataset.path)
    assert acquired.acquired_plane_count == planes
    assert acquired.plane_period_s == pytest.approx(0.03)
    assert acquired.volume_interval_s == pytest.approx(planes * 0.03)
    assert acquired.with_plane_period(0.03).volume_interval_s == pytest.approx(
        planes * 0.03
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "exposures, message",
    [
        ([0], "Invalid camera exposure"),
        ([float("nan")], "Invalid camera exposure"),
        ([float("inf")], "Invalid camera exposure"),
        ([-1], "Invalid camera exposure"),
        ([True], "Invalid camera exposure"),
        (["invalid"], "Invalid camera exposure"),
        ([0.001], "rounds to zero"),
        ([5, 10], "hardware exposures change"),
    ],
)
def test_invalid_hardware_exposures_rejected(exposures, message):
    """Reject nonphysical hardware exposures and changes in hardware cadence."""
    frames = [
        {
            "event_index": {"t": 0, "p": 0, "c": 0, "z": scan},
            "exposure_ms": exposure,
        }
        for scan, exposure in enumerate(exposures)
    ]
    with pytest.raises(ValueError, match=message):
        _camera_timing([frames], 1)


@pytest.mark.unit
def test_hardware_timing_ignores_software_timestamps():
    """Use rounded actual hardware exposure rather than arrival or requested times."""
    frames = [
        {
            "event_index": {"z": scan},
            "runner_time_ms": scan * 5,
            "delta_t": scan * 0.116,
            "camera_metadata": {"ElapsedTime-ms": -scan},
            "event_metadata": {"Camera": {"exposure_ms": 150}},
        }
        for scan in range(2)
    ]
    assert _camera_timing([frames], 1) == (None, None)
    for frame in frames:
        frame["exposure_time"] = 0.0050021029411764706
        frame["exposure_ms"] = 150  # Prefer the current serialized camera value.
    assert _camera_timing([frames], 1) == (
        0.005,
        "recorded camera exposure_time; rounded to 0.01 ms",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "exposures, period", [([5.0021], 0.005), ([5.0021, 7.018], 0.01202)]
)
def test_hardware_channel_periods_sum_once(exposures, period):
    """Count each stored camera channel once, rounding before summing exposures."""
    frames = [
        {"event_index": {"c": channel}, "exposure_ms": exposure}
        for channel, exposure in enumerate(exposures)
    ]
    assert _camera_timing([frames], len(exposures))[0] == pytest.approx(period)
    assert _camera_timing([frames], len(exposures) + 1) == (None, None)


@pytest.mark.unit
@pytest.mark.parametrize("projection", [False, True])
def test_fused_writer_preserves_time_axis(
    tmp_path, fusion_operator, position_store_factory, projection
):
    """Retain the source temporal axis in fused output."""
    import numpy as np

    from opm_processing.imageprocessing.maxtilefusion import MaxTileFusion

    path = tmp_path / "fused.ome.zarr"
    if projection:
        collection = position_store_factory(
            np.zeros((3, 1, 1, 1, 8, 8), np.uint16),
            name="projection",
            voxel_size_um=(0.23, 0.115, 0.115),
            time_interval_s=0.085,
        )
        operator = MaxTileFusion(
            collection.arrays,
            ((0, 0, 0),),
            path,
            collection.voxel_size_um,
            time_interval_s=collection.time_interval_s,
            chunk_size=8,
            source_collection=collection.path,
        )
        provenance = open_group(path).attrs["opm_processing_projection"]
        assert provenance["source_collection"] == collection.path.name
        assert provenance["source_position_indices"] == [0]
    else:
        operator = fusion_operator(
            time_interval_s=0.085,
            multiscale_factors=(2,),
            multiscale_downsample="stride",
            _pixel_size=(0.23, 0.115, 0.115),
            offset_um=(0, 0, 0),
            padded_shape=(4, 8, 8),
            _is_2d=False,
            time_dim=3,
            channels=1,
            chunk_y=4,
            chunk_x=4,
        )
        operator._prepare_fused_image(path, z_slices_per_write=2)
    image = open_group(path).ome_metadata()
    assert image.multiscales[0].axes[0].unit == "second"
    assert all(
        d.scale_transform.scale[0] == 0.085 for d in image.multiscales[0].datasets
    )
