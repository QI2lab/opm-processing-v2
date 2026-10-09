"""Calibrated simulated acquisitions in the supported upstream OME-Zarr layouts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr
from yaozarrs import DimSpec, v05
from yaozarrs.write.v05 import prepare_image

from opm_processing.dataio.acquisition import AcquisitionMetadata, ChannelMetadata
from opm_processing.dataio.position_collection import (
    create_position_collection,
    open_position_collection,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class OpmV2ProjectionFixture:
    """A projection object stored in the upstream OPMMirrorHandler layout.

    Attributes
    ----------
    path : pathlib.Path
        Persisted raw acquisition.
    raw_data : numpy.ndarray
        Known camera values in TPCYX order.
    camera_offset, camera_conversion : float
        Detector offset in ADU and conversion to photon counts.
    pixel_size_um : float
        Detector pixel pitch in micrometers.
    channel_names : tuple of str
        Spectral labels in storage order.
    stage_positions_zxy : numpy.ndarray
        Recorded stage positions in micrometers.
    """

    path: Path
    raw_data: np.ndarray
    camera_offset: float
    camera_conversion: float
    pixel_size_um: float
    channel_names: tuple[str, ...]
    stage_positions_zxy: np.ndarray


def opm_v2_frame_metadata(
    *,
    index: dict[str, int],
    daq_metadata: dict,
    opm_metadata: dict,
    stage_metadata: dict,
    camera_shape_yx: tuple[int, int],
    pixel_size_um: float,
    camera_offset: float,
    camera_conversion: float,
    runner_time_ms: float,
    hardware_triggered: bool | None = None,
    additional_event_metadata: dict | None = None,
) -> dict:
    """Build one upstream-compatible ``FrameMetaV1`` dictionary.

    Parameters
    ----------
    index : dict
        Logical time, position, channel and optional scan indices.
    daq_metadata, opm_metadata, stage_metadata : dict
        Acquisition controller settings, optical geometry and physical stage pose.
    camera_shape_yx : tuple of int
        Detector row and column counts.
    pixel_size_um : float
        Detector pitch in micrometers.
    camera_offset, camera_conversion : float
        Offset in ADU and calibrated photon conversion.
    runner_time_ms : float
        Acquisition runner timestamp in milliseconds.
    hardware_triggered : bool or None
        Recorded trigger mode; None omits the optional field.
    additional_event_metadata : dict or None
        Additional upstream controller metadata attached to this frame.

    Returns
    -------
    dict
        Physical frame metadata containing its logical acquisition index.
    """
    event_metadata = {
        "DAQ": daq_metadata,
        "Camera": {
            "exposure_ms": 10.0,
            "camera_center_x": 128,
            "camera_center_y": 128,
            "camera_crop_x": camera_shape_yx[1],
            "camera_crop_y": camera_shape_yx[0],
            "offset": camera_offset,
            "e_to_ADU": camera_conversion,
        },
        "OPM": opm_metadata,
        "Stage": stage_metadata,
        **(additional_event_metadata or {}),
    }
    frame = {
        "format": "frame-dict",
        "version": "1.0",
        "pixel_size_um": pixel_size_um,
        "camera_device": "SyntheticCamera",
        "exposure_ms": 10.0,
        "property_values": [],
        "runner_time_ms": runner_time_ms,
        "mda_event": {"index": index, "metadata": event_metadata},
    }
    if hardware_triggered is not None:
        frame["hardware_triggered"] = hardware_triggered
    return frame


def write_opm_v2_zarr(
    *,
    path: Path,
    raw_data: np.ndarray,
    labels: tuple[str, ...],
    chunks: tuple[int, ...],
    frame_metadatas: list[dict],
) -> None:
    """Write a current group-based OME-Zarr acquisition fixture.

    Parameters
    ----------
    path : pathlib.Path
        Destination acquisition directory.
    raw_data : numpy.ndarray
        Known camera pixels in TPCZYX or TPCYX order.
    labels : tuple of str
        Logical labels matching raw_data dimensions.
    chunks : tuple of int
        Storage chunks in the same dimension order.
    frame_metadatas : list of dict
        Upstream frame records giving channel labels and physical calibration.

    Returns
    -------
    None
        Persist the real per-position image arrays and their frame records.
    """
    normalized = raw_data if "z" in labels else raw_data[:, :, :, None, :, :]
    normalized_chunks = chunks if "z" in labels else (*chunks[:3], 1, *chunks[-2:])
    time_count, position_count, channel_count, z_count, _y_count, _x_count = (
        int(value) for value in normalized.shape
    )
    frames_by_position: list[list[dict]] = [[] for _ in range(position_count)]
    channel_names = [f"channel-{index}" for index in range(channel_count)]
    stage_positions = [(0.0, 0.0, 0.0)] * position_count
    pixel_size_um = 1.0
    for frame in frame_metadatas:
        event = frame["mda_event"]
        index = event["index"]
        metadata = event["metadata"]
        position = int(index.get("p", 0))
        channel = int(index.get("c", 0))
        frames_by_position[position].append(frame)
        channel_names[channel] = str(metadata["DAQ"]["current_channel"])
        stage = metadata["Stage"]
        stage_positions[position] = (
            float(stage["z_pos"]),
            float(stage["x_pos"]),
            float(stage["y_pos"]),
        )
        pixel_size_um = float(frame["pixel_size_um"])

    index_sizes = {"t": time_count, "p": position_count, "c": channel_count}
    if z_count > 1:
        index_sizes["z"] = z_count
    collection = create_position_collection(
        path,
        normalized.shape,
        (1.0, pixel_size_um, pixel_size_um),
        stage_positions=stage_positions,
        channels=channel_names,
        attributes={
            "opm_v2": {
                "index_sizes": index_sizes,
                "acquisition_order": [
                    axis for axis in labels if axis not in {"y", "x"}
                ],
                "configuration": {"acq_config": {}},
            }
        },
        chunks=(
            normalized_chunks[0],
            normalized_chunks[2],
            normalized_chunks[3],
            normalized_chunks[4],
            normalized_chunks[5],
        ),
    )
    for position, array in enumerate(collection.arrays):
        array.write(normalized[:, position]).result()
    root = zarr.open_group(path, mode="r+")
    for position, frames in enumerate(frames_by_position):
        root[str(position)].attrs["ome_writers"] = {"frame_metadata": frames}


@pytest.fixture
def opm_v2_projection_zarr(tmp_path) -> OpmV2ProjectionFixture:
    """Create a small, faithful OPMMirrorHandler projection acquisition.

    Schema provenance (QI2lab/opm-v2 main, commit ee354a1421732218d1128332c42a93aef106aa33):
    - ``handlers/opm_mirror_handler.py`` defines shape/chunks/labels and .zattrs.
    - ``engine/setup_events_v2.py`` defines projection event metadata.
    - pymmcore-plus ``FrameMetaV1`` defines the enclosing frame metadata.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory receiving the acquisition.

    Returns
    -------
    OpmV2ProjectionFixture
        Saved multi-timepoint, multi-channel camera ramp and its calibration.
    """
    path = tmp_path / "opm_v2_projection.zarr"
    shape = (2, 1, 2, 16, 18)  # T, P, C, Y, X; projection has no Z index.
    chunks = (1, 1, 1, shape[-2], shape[-1])
    yy, xx = np.mgrid[: shape[-2], : shape[-1]]
    raw_data = np.empty(shape, dtype=np.uint16)
    for time in range(shape[0]):
        for channel in range(shape[2]):
            raw_data[time, 0, channel] = 200 + 50 * time + 25 * channel + 2 * yy + xx

    camera_offset = 100.0
    camera_conversion = 0.25
    pixel_size_um = 0.115
    channel_names = ("488nm", "561nm")
    stage_positions_zxy = np.array([[4.0, 20.0, 30.0]])
    frame_metadatas = []
    for time in range(shape[0]):
        for channel, channel_name in enumerate(channel_names):
            frame_metadatas.append(
                opm_v2_frame_metadata(
                    index={"t": time, "p": 0, "c": channel},
                    daq_metadata={
                        "mode": "projection",
                        "image_mirror_position": None,
                        "image_mirror_range_um": 40.0,
                        "image_mirror_step_um": None,
                        "channel_states": [channel == 0, channel == 1],
                        "exposure_channels_ms": [10.0, 10.0],
                        "laser_powers": [10.0, 12.0],
                        "interleaved": False,
                        "blanking": True,
                        "current_channel": channel_name,
                    },
                    opm_metadata={
                        "angle_deg": 30.0,
                        "camera_Zstage_orientation": "normal",
                        "camera_XYstage_orientation": "normal",
                        "camera_mirror_orientation": "normal",
                    },
                    stage_metadata={
                        "x_pos": stage_positions_zxy[0, 1],
                        "y_pos": stage_positions_zxy[0, 2],
                        "z_pos": stage_positions_zxy[0, 0],
                    },
                    camera_shape_yx=shape[-2:],
                    pixel_size_um=pixel_size_um,
                    camera_offset=camera_offset,
                    camera_conversion=camera_conversion,
                    runner_time_ms=float(time * 100 + channel),
                    additional_event_metadata={
                        "AO_mirror": {"modal_coeffs": None, "voltages": None}
                    },
                )
            )

    write_opm_v2_zarr(
        path=path,
        raw_data=raw_data,
        labels=("t", "p", "c", "y", "x"),
        chunks=chunks,
        frame_metadatas=frame_metadatas,
    )
    return OpmV2ProjectionFixture(
        path=path,
        raw_data=raw_data,
        camera_offset=camera_offset,
        camera_conversion=camera_conversion,
        pixel_size_um=pixel_size_um,
        channel_names=channel_names,
        stage_positions_zxy=stage_positions_zxy,
    )


def simulated_acquisition_metadata(
    path: Path, mode: str, shape: tuple[int, ...]
) -> AcquisitionMetadata:
    """Describe a calibrated 0.8 um combined-laser scan.

    Parameters
    ----------
    path : pathlib.Path
        On-disk acquisition store for the simulated specimen.
    mode : str
        Mirror or stage scanning mode.
    shape : tuple of int
        Acquisition dimensions in TPCZYX order.

    Returns
    -------
    AcquisitionMetadata
        Physical scan and camera calibration for the simulated acquisition.
    """
    return AcquisitionMetadata(
        path=path,
        storage_format="opm-v2-ome-zarr-v3",
        mode=mode,
        axes=("t", "p", "c", "z", "y", "x"),
        shape=shape,
        array_paths=tuple(f"{position}/0" for position in range(shape[1])),
        acquisition_order=("t", "p", "z", "c"),
        channels=(ChannelMetadata(0, "488 + 637", None, 10.0, None),),
        stage_positions_zxy=((4.0, 20.0, 30.0),) * shape[1],
        scan_start_positions_xyz=(),
        scan_end_positions_xyz=(),
        scan_axis="x",
        scan_axis_step_um=0.8,
        pixel_size_um=0.115,
        angle_deg=30.0,
        camera_offset=100.0,
        camera_conversion=0.5,
        excess_scan_positions=0,
        excess_scan_start_positions=0,
        excess_scan_end_positions=0,
        orientations=(),
        sidecar_paths=(),
    )


def write_simulated_acquisition(metadata: AcquisitionMetadata, raw: np.ndarray) -> None:
    """Persist simulated camera pixels and their physical acquisition metadata.

    Parameters
    ----------
    metadata : AcquisitionMetadata
        Scan geometry and camera calibration for the specimen.
    raw : numpy.ndarray
        Simulated uint16 camera data in TPCZYX order.

    Returns
    -------
    None
        Pixels and scan metadata are persisted at the metadata path; no image,
        metadata inspection, or processing-state boundary is mocked.
    """
    frames = []
    for position, (stage_z, stage_x, stage_y) in enumerate(
        metadata.stage_positions_zxy
    ):
        for time in range(metadata.shape[0]):
            for scan in range(min(2, metadata.shape[3])):
                for channel in metadata.channels:
                    frames.append(
                        opm_v2_frame_metadata(
                            index={
                                "t": time,
                                "p": position,
                                "c": channel.index,
                                "z": scan,
                            },
                            daq_metadata={
                                "mode": metadata.mode,
                                "channel_states": [True] * len(metadata.channels),
                                "exposure_channels_ms": [10.0] * len(metadata.channels),
                                "laser_powers": [10.0] * len(metadata.channels),
                                "interleaved": True,
                                "blanking": True,
                                "current_channel": channel.name,
                                "scan_axis_step_um": metadata.scan_axis_step_um,
                                "image_mirror_position": float(scan),
                                "image_mirror_step_um": metadata.scan_axis_step_um,
                                "image_mirror_range_um": metadata.shape[3]
                                * metadata.scan_axis_step_um,
                            },
                            opm_metadata={
                                "angle_deg": metadata.angle_deg,
                                **dict(metadata.orientations),
                                "excess_scan_positions": metadata.excess_scan_positions,
                                "excess_scan_start_positions": metadata.excess_scan_start_positions,
                                "excess_scan_end_positions": metadata.excess_scan_end_positions,
                            },
                            stage_metadata={
                                "x_pos": stage_x,
                                "y_pos": stage_y,
                                "z_pos": stage_z,
                                "excess_image": False,
                            },
                            camera_shape_yx=metadata.shape[-2:],
                            pixel_size_um=metadata.pixel_size_um,
                            camera_offset=metadata.camera_offset,
                            camera_conversion=metadata.camera_conversion,
                            runner_time_ms=float(len(frames)),
                            hardware_triggered=True,
                        )
                    )
    write_opm_v2_zarr(
        path=metadata.path,
        raw_data=raw,
        labels=metadata.axes,
        chunks=(1, 1, 1, 1, *metadata.shape[-2:]),
        frame_metadatas=frames,
    )


@pytest.fixture
def current_opm_v2_stage_scan(tmp_path: Path) -> Path:
    """Create a small Bio-Formats2Raw collection matching current opm-v2.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory receiving the stage acquisition.

    Returns
    -------
    pathlib.Path
        Collection containing known pixels, event-indexed metadata and stage poses.
    """
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
    """Create the root-Image layout ome-writers uses for one position.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory receiving the mirror acquisition.

    Returns
    -------
    pathlib.Path
        Root image containing three camera timepoints and upstream scan metadata.
    """
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


@pytest.fixture
def acquisition_factory(tmp_path):
    """Return a factory that persists camera pixels and complete physical metadata.

    The factory accepts a TPCZYX array, a dataset name, a scan mode and keyword
    overrides of AcquisitionMetadata fields. Defaults describe a 30 degree OPM
    with 115 nm detector pixels, 400 nm scans, 100 ADU offset and 0.25 photons/ADU.
    It returns the path, metadata, original pixels and reopened disk collection.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory receiving each independently named acquisition.

    Returns
    -------
    callable
        Dataset writer accepting camera pixels and physical metadata overrides.
    """

    def create(raw_data, *, name="sample", mode="mirror", **overrides):
        """Write one independent dataset with overrides of its physical calibration.

        Parameters
        ----------
        raw_data : numpy.ndarray
            Simulated camera values in time, position, channel, scan, Y, X order.
        name : str
            Dataset basename inside the test's temporary directory.
        mode : str
            Acquisition mode recorded in upstream frame metadata.
        **overrides
            AcquisitionMetadata fields replacing the common OPM calibration.

        Returns
        -------
        types.SimpleNamespace
            Dataset path, metadata, raw_data and reopened position collection.
        """
        metadata = replace(
            simulated_acquisition_metadata(
                tmp_path / f"{name}.ome.zarr", mode, raw_data.shape
            ),
            channels=tuple(
                ChannelMetadata(index, f"{wavelength}nm", wavelength, 10.0, 10.0)
                for index, wavelength in enumerate((488, 561, 637)[: raw_data.shape[2]])
            ),
            scan_axis_step_um=0.4,
            camera_conversion=0.25,
        )
        metadata = replace(metadata, **overrides)
        write_simulated_acquisition(metadata, raw_data)
        return SimpleNamespace(
            path=metadata.path,
            metadata=metadata,
            raw_data=raw_data,
            collection=open_position_collection(metadata.path),
        )

    return create


@pytest.fixture
def processing_options():
    """Return common settings for a full-resolution isolated volume workflow.

    Returns
    -------
    dict
        Disable illumination fitting, tile projections and automatic fusion;
        preserve native laboratory Z sampling. Tests override other options
        explicitly and validate their written fluorescence.
    """
    return {
        "flatfield_correction": False,
        "max_projection": False,
        "create_fused_max_projection": False,
        "z_downsample_level": 1,
    }
