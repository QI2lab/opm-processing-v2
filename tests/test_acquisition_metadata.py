"""Metadata-only coverage for the current opm-v2 OME-Zarr writer layout."""

from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np
import zarr
from tifffile import imread
from yaozarrs import DimSpec, v05
from yaozarrs.write.v05 import prepare_image

from opm_processing.dataio.acquisition import (
    inspect_acquisition,
    open_acquisition_datastore,
)
from opm_processing.dataio.convert_timelapse_data import convert_timelapse
from opm_processing.dataio.position_collection import (
    create_position_collection,
    open_position_collection,
)
from opm_processing.process import process


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
            data[0, channel] = 1000 * position + 100 * channel + 10 * zz + 2 * yy + xx
        root[str(position)]["0"][:] = data
    return path


@pytest.fixture
def current_single_position_mirror_timelapse(tmp_path: Path) -> Path:
    """Create the root-Image layout ome-writers uses for one position."""
    path = tmp_path / "single_mirror.ome.zarr"
    shape = (3, 1, 4, 5, 6)  # T, C, Z, Y, X
    dims = [
        DimSpec(name="t", size=shape[0], scale=1.0, unit="second"),
        DimSpec(name="c", size=shape[1], scale=1.0),
        DimSpec(name="z", size=shape[2], scale=0.4, unit="micrometer"),
        DimSpec(name="y", size=shape[3], scale=0.115, unit="micrometer"),
        DimSpec(name="x", size=shape[4], scale=0.115, unit="micrometer"),
    ]
    image = v05.Image(
        multiscales=[v05.Multiscale.from_dims(dims, name="single-position")]
    )
    _, arrays = prepare_image(
        path,
        image,
        datasets=[(shape, np.dtype(np.uint16))],
        extra_attributes={
            "opm_v2": {
                "index_sizes": {"t": 3, "p": 1, "c": 1, "z": 4},
                "acquisition_order": ["t", "p", "z", "c"],
                "configuration": {
                    "acq_config": {
                        "opm_mode": "mirror",
                        "DAQ": {
                            "channel_states": [True],
                            "channel_powers": [10.0],
                            "channel_exposures_ms": [2.0],
                        },
                    }
                },
            }
        },
        chunks=(1, 1, 1, shape[-2], shape[-1]),
        writer="tensorstore",
        overwrite=True,
    )
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) + 200
    arrays["0"].write(data).result()

    frames = []
    for time in range(shape[0]):
        for scan in range(shape[2]):
            frames.append(
                {
                    "event_index": {"t": time, "p": 0, "c": 0, "z": scan},
                    "event_metadata": {
                        "DAQ": {
                            "mode": "mirror",
                            "image_mirror_step_um": 0.4,
                            "current_channel": "488nm",
                            "laser_powers": [10.0],
                        },
                        "Camera": {
                            "exposure_ms": 2.0,
                            "offset": 100.0,
                            "e_to_ADU": 0.25,
                        },
                        "OPM": {
                            "angle_deg": 30.0,
                            "camera_mirror_orientation": "positive",
                            "camera_XYstage_orientation": "positive",
                            "camera_Zstage_orientation": "positive",
                        },
                        "Stage": {
                            "x_pos": 100.0,
                            "y_pos": 200.0,
                            "z_pos": 30.0,
                        },
                    },
                    "storage_index": [time, 0, scan],
                }
            )
    root = zarr.open_group(path, mode="a")
    root.attrs["ome_writers"] = {"frame_metadata": frames}
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


def test_single_position_root_image_opens_as_virtual_tpczyx(
    current_single_position_mirror_timelapse: Path,
) -> None:
    """Normalize ome-writers' one-position Image layout with a P=1 axis."""
    path = current_single_position_mirror_timelapse
    metadata = inspect_acquisition(path)
    datastore = open_acquisition_datastore(metadata)

    assert metadata.storage_format == "opm-v2-ome-zarr-v3"
    assert metadata.mode == "mirror"
    assert metadata.axes == ("t", "p", "c", "z", "y", "x")
    assert metadata.shape == (3, 1, 1, 4, 5, 6)
    assert metadata.array_paths == ("0",)
    assert metadata.scan_axis_step_um == pytest.approx(0.4)
    assert metadata.stage_positions_zxy == ((30.0, 100.0, 200.0),)
    assert tuple(datastore.shape) == metadata.shape
    expected = (
        np.arange(3 * 1 * 4 * 5 * 6, dtype=np.uint16).reshape(3, 1, 4, 5, 6) + 200
    )
    np.testing.assert_array_equal(datastore[:, 0].read().result(), expected)


def test_single_position_mirror_timelapse_processes_every_timepoint(
    current_single_position_mirror_timelapse: Path,
) -> None:
    """Deskew every timepoint from a one-position root Image acquisition."""
    path = current_single_position_mirror_timelapse
    process(
        root_path=path,
        max_projection=False,
        create_fused_max_projection=False,
        z_downsample_level=1,
    )

    output = open_position_collection(path.parent / "single_mirror_deskewed.ome.zarr")
    assert output.shape[:3] == (3, 1, 1)
    for time_index in range(3):
        assert np.any(output.arrays[0][time_index].read().result())


def test_single_position_mirror_timelapse_can_save_float32(
    current_single_position_mirror_timelapse: Path,
) -> None:
    """Preserve fractional calibrated values for the reported acquisition layout."""
    path = current_single_position_mirror_timelapse
    process(
        root_path=path,
        save_float32=True,
        max_projection=False,
        create_fused_max_projection=False,
        z_downsample_level=1,
    )

    output = open_position_collection(path.parent / "single_mirror_deskewed.ome.zarr")
    values = np.asarray(output.arrays[0][0, 0].read().result())
    assert values.dtype == np.float32
    assert np.any((values > 0) & (values != np.floor(values)))
    conversion = output.attributes["opm_processing"]["steps"][-1]
    assert conversion["name"] == "float32_output"
    assert conversion["parameters"]["quantized"] is False


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
