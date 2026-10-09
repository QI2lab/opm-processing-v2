"""Persist simulated combined-laser acquisitions for reconstruction tests."""

from pathlib import Path

import numpy as np

from opm_processing.dataio.acquisition import AcquisitionMetadata, ChannelMetadata
from tests.conftest import _opm_v2_frame_metadata, _write_opm_v2_zarr


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
        array_paths=("0/0",),
        acquisition_order=("t", "p", "z", "c"),
        channels=(ChannelMetadata(0, "488 + 637", None, 10.0, None),),
        stage_positions_zxy=((4.0, 20.0, 30.0),),
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
        for scan in range(2):
            for channel in metadata.channels:
                frames.append(
                    _opm_v2_frame_metadata(
                        index={"t": 0, "p": position, "c": channel.index, "z": scan},
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
                        opm_metadata={"angle_deg": metadata.angle_deg},
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
    _write_opm_v2_zarr(
        path=metadata.path,
        raw_data=raw,
        labels=metadata.axes,
        chunks=(1, 1, 1, 1, *metadata.shape[-2:]),
        frame_metadatas=frames,
    )
