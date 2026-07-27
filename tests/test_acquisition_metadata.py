"""Metadata-only coverage for the current opm-v2 OME-Zarr writer layout."""

from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np
import zarr
from tifffile import imread

from opm_processing.dataio.acquisition import (
    inspect_acquisition,
    open_acquisition_datastore,
)
from opm_processing.dataio.convert_timelapse_data import convert_timelapse
from opm_processing.dataio.position_collection import create_position_collection


@pytest.fixture
def current_opm_v2_stage_scan(tmp_path: Path) -> Path:
    """Create a small Bio-Formats2Raw collection matching current opm-v2."""
    path = tmp_path / "current_stage.ome.zarr"
    shape = (1, 2, 2, 3, 4, 5)  # T, P, C, Z, Y, X
    stage_positions_zxy = ((30.0, 100.0, 200.0), (30.0, 100.0, 220.0))
    root_opm_metadata = {
        "index_sizes": {"t": 1, "p": 2, "c": 2, "z": 3},
        "acquisition_order": ["t", "p", "z", "c"],
        "configuration": {
            "acq_config": {
                "opm_mode": "stage",
                "DAQ": {
                    "channel_states": [True, True, False],
                    "channel_powers": [12.0, 18.0, 0.0],
                    "channel_exposures_ms": [10.0, 15.0, 0.0],
                    "scan_axis_step_um": 0.4,
                },
            }
        },
    }
    create_position_collection(
        path,
        shape,
        (0.4, 0.115, 0.115),
        stage_positions=stage_positions_zxy,
        channels=("488nm", "561nm"),
        attributes={"opm_v2": root_opm_metadata},
        chunks=(1, 1, 1, 4, 5),
    )

    root = zarr.open_group(path, mode="a")
    for position in range(shape[1]):
        frames = []
        for scan in range(shape[3]):
            for channel, name in enumerate(("488nm", "561nm")):
                frames.append(
                    {
                        "event_index": {
                            "t": 0,
                            "p": position,
                            "c": channel,
                            "z": scan,
                        },
                        "exposure_time": (10.0, 15.0)[channel] / 1000.0,
                        "event_metadata": {
                            "DAQ": {
                                "mode": "stage",
                                "scan_axis_step_um": 0.4,
                                "laser_powers": [12.0, 18.0, 0.0],
                                "current_channel": name,
                            },
                            "Camera": {
                                "exposure_ms": (10.0, 15.0)[channel],
                                "offset": 100.0,
                                "e_to_ADU": 0.24,
                            },
                            "OPM": {
                                "angle_deg": 30.0,
                                "camera_Zstage_orientation": "negative",
                                "camera_XYstage_orientation": "positive",
                                "camera_mirror_orientation": "positive",
                                "excess_scan_positions": 0,
                                "excess_scan_start_positions": 0,
                                "excess_scan_end_positions": 0,
                            },
                            "Stage": {
                                "x_pos": 100.0 + 0.4 * scan,
                                "y_pos": stage_positions_zxy[position][2],
                                "z_pos": stage_positions_zxy[position][0],
                                "excess_image": False,
                            },
                        },
                        "storage_index": [0, channel, scan],
                    }
                )
        root[str(position)].attrs["ome_writers"] = {"frame_metadata": frames}

        data = np.empty((1, shape[2], shape[3], shape[4], shape[5]), dtype=np.uint16)
        zz, yy, xx = np.mgrid[: shape[3], : shape[4], : shape[5]]
        for channel in range(shape[2]):
            data[0, channel] = (
                1000 * position + 100 * channel + 10 * zz + 2 * yy + xx
            )
        root[str(position)]["0"][:] = data
    return path


def test_current_stage_metadata_is_discovered_without_array_open(
    current_opm_v2_stage_scan: Path,
) -> None:
    """Recover all acquisition metadata written into a synthetic collection."""
    metadata = inspect_acquisition(current_opm_v2_stage_scan.parent)

    assert metadata.storage_format == "opm-v2-ome-zarr-v3"
    assert metadata.mode == "stage"
    assert metadata.axes == ("t", "p", "c", "z", "y", "x")
    assert metadata.shape == (1, 2, 2, 3, 4, 5)
    assert metadata.tile_count == 2
    assert metadata.scan_position_count == 3
    assert metadata.channel_names == ("488nm", "561nm")
    assert [channel.wavelength_nm for channel in metadata.channels] == [488.0, 561.0]
    assert [channel.exposure_ms for channel in metadata.channels] == [10.0, 15.0]
    assert [channel.laser_power for channel in metadata.channels] == [12.0, 18.0]
    assert metadata.stage_positions_zxy == (
        (30.0, 100.0, 200.0),
        (30.0, 100.0, 220.0),
    )
    assert metadata.scan_axis == "x"
    assert metadata.scan_axis_step_um == pytest.approx(0.4)
    assert metadata.pixel_size_um == pytest.approx(0.115)
    assert metadata.angle_deg == pytest.approx(30.0)
    assert metadata.camera_offset == pytest.approx(100.0)
    assert metadata.camera_conversion == pytest.approx(0.24)
    assert metadata.stage_axis_flips_xyz == (False, False, True)
    assert metadata.scan_axis_reversed is True


def test_current_stage_collection_opens_as_virtual_tpczyx(
    current_opm_v2_stage_scan: Path,
) -> None:
    """Recover exact pixels from every virtualized position and channel."""
    metadata = inspect_acquisition(current_opm_v2_stage_scan)
    datastore = open_acquisition_datastore(metadata)

    assert datastore.rank == 6
    assert tuple(datastore.shape) == metadata.shape
    assert tuple(datastore.domain.labels) == ("t", "", "c", "z", "y", "x")

    zz, yy, xx = np.mgrid[:3, :4, :5]
    for position in range(2):
        for channel in range(2):
            expected = 1000 * position + 100 * channel + 10 * zz + 2 * yy + xx
            actual = datastore[0, position, channel].read().result()
            np.testing.assert_array_equal(actual, expected)


def test_timelapse_converter_accepts_current_collection(
    current_opm_v2_stage_scan: Path,
) -> None:
    """Converted TIFF pixels must exactly match the requested source data."""
    output_dir = current_opm_v2_stage_scan.parent / "converted"
    written = convert_timelapse(
        current_opm_v2_stage_scan.parent,
        output_dir=output_dir,
        time_range=(0, 1),
        stage_range=(0, 1),
        scan_range=(0, 1),
        create_tiff=True,
    )

    assert written == [output_dir / "pos_0_scan_0.tiff"]
    converted = imread(written[0])
    yy, xx = np.mgrid[:4, :5]
    expected = np.stack((2 * yy + xx, 100 + 2 * yy + xx))[np.newaxis, ...]
    np.testing.assert_array_equal(converted, expected)
