"""Actual producer acquisitions and synthetic layouts for isolated metadata units."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
from yaozarrs import DimSpec, open_group, v05
from yaozarrs.write.v05 import prepare_image

from opm_processing.dataio.acquisition import AcquisitionMetadata, ChannelMetadata
from opm_processing.dataio.position_collection import (
    create_position_collection,
    open_position_collection,
    update_group_attributes,
)


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
    exposure_ms: float | None = 10.0,
    additional_event_metadata: dict | None = None,
    frame_format: Literal["mda_event", "event_index"] = "mda_event",
) -> dict:
    """Build an upstream frame record in either supported metadata format."""
    event_metadata = {
        "DAQ": daq_metadata,
        "Camera": {
            "exposure_ms": exposure_ms,
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
    if frame_format == "event_index":
        frame = {
            "event_index": index,
            "event_metadata": event_metadata,
            "storage_index": [index.get(axis, 0) for axis in ("t", "c", "z")],
        }
        if exposure_ms is not None:
            frame["exposure_time"] = exposure_ms / 1000
        return frame
    frame = {
        "format": "frame-dict",
        "version": "1.0",
        "pixel_size_um": pixel_size_um,
        "camera_device": "SyntheticCamera",
        "property_values": [],
        "runner_time_ms": runner_time_ms,
        "mda_event": {"index": index, "metadata": event_metadata},
    }
    if exposure_ms is not None:
        frame["exposure_ms"] = exposure_ms
    if hardware_triggered is not None:
        frame["hardware_triggered"] = hardware_triggered
    return frame


def write_synthetic_acquisition_zarr(
    *,
    path: Path,
    raw_data: np.ndarray,
    labels: tuple[str, ...],
    chunks: tuple[int, ...],
    frame_metadatas: list[dict],
) -> None:
    """Write a synthetic group layout for isolated metadata and storage checks.

    This does not execute opm-v2. Acquisition pipeline integrations must use
    ``opm_v2_acquisition_factory`` instead.
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
    for position, frames in enumerate(frames_by_position):
        update_group_attributes(
            path / str(position), {"ome_writers": {"frame_metadata": frames}}
        )


@pytest.fixture
def opm_v2_projection_zarr(
    simulated_camera, opm_v2_acquisition_factory
) -> OpmV2ProjectionFixture:
    """Simulate spectral/time observations and persist them with OpmDataHandler."""
    shape = (2, 1, 2, 16, 18)
    raw_data = np.empty(shape, dtype=np.uint16)
    for time in range(shape[0]):
        for channel in range(shape[2]):
            raw_data[time, 0, channel] = simulated_camera(
                (1, *shape[-2:]),
                geometry="cartesian",
                exposure=1 + time / 4 + channel / 2,
            ).raw[0]
    dataset = opm_v2_acquisition_factory(
        raw_data[:, :, :, None],
        name="opm_v2_projection",
        mode="projection",
        suffix=".zarr",
        stage_positions_zxy=((4.0, 20.0, 30.0),),
    )
    return OpmV2ProjectionFixture(
        path=dataset.path,
        raw_data=raw_data,
        camera_offset=dataset.metadata.camera_offset,
        camera_conversion=dataset.metadata.camera_conversion,
        pixel_size_um=dataset.metadata.pixel_size_um,
        channel_names=dataset.metadata.channel_names,
        stage_positions_zxy=np.asarray(dataset.metadata.stage_positions_zxy),
    )


def simulated_acquisition_metadata(
    path: Path, mode: str, shape: tuple[int, ...]
) -> AcquisitionMetadata:
    """Describe a calibrated 0.8 um combined-laser scan."""
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
    """Persist simulated camera pixels and their physical acquisition metadata."""
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
                            exposure_ms=(
                                1000 * metadata.plane_period_s / len(metadata.channels)
                                if metadata.plane_period_s is not None
                                else None
                            ),
                            hardware_triggered=True,
                        )
                    )
    write_synthetic_acquisition_zarr(
        path=metadata.path,
        raw_data=raw,
        labels=metadata.axes,
        chunks=(1, 1, 1, 1, *metadata.shape[-2:]),
        frame_metadatas=frames,
    )

    if metadata.deconvolution is not None:
        update_group_attributes(
            metadata.path, {"opm_processing": {"deconvolution": metadata.deconvolution}}
        )


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

    for position in range(shape[1]):
        frames = []
        for scan in range(shape[3]):
            for channel, name in enumerate(("488nm", "561nm")):
                frames.append(
                    opm_v2_frame_metadata(
                        frame_format="event_index",
                        index={"t": 0, "p": position, "c": channel, "z": scan},
                        daq_metadata={
                            "mode": "stage",
                            "scan_axis_step_um": 0.4,
                            "laser_powers": [12.0, 18.0, 0.0],
                            "current_channel": name,
                        },
                        opm_metadata={
                            "angle_deg": 30.0,
                            "camera_Zstage_orientation": "negative",
                            "camera_XYstage_orientation": "positive",
                            "camera_mirror_orientation": "positive",
                            "excess_scan_positions": 0,
                            "excess_scan_start_positions": 0,
                            "excess_scan_end_positions": 0,
                        },
                        stage_metadata={
                            "x_pos": 100.0 + 0.4 * scan,
                            "y_pos": stage_positions_zxy[position][2],
                            "z_pos": stage_positions_zxy[position][0],
                            "excess_image": False,
                        },
                        camera_shape_yx=shape[-2:],
                        pixel_size_um=0.115,
                        camera_offset=100.0,
                        camera_conversion=0.24,
                        runner_time_ms=float(len(frames)),
                        exposure_ms=(10.0, 15.0)[channel],
                    )
                )
        update_group_attributes(
            path / str(position), {"ome_writers": {"frame_metadata": frames}}
        )

        data = np.empty((1, shape[2], shape[3], shape[4], shape[5]), dtype=np.uint16)
        zz, yy, xx = np.mgrid[: shape[3], : shape[4], : shape[5]]
        for channel in range(shape[2]):
            data[0, channel] = 1000 * position + 100 * channel + 10 * zz + 2 * yy + xx
        open_group(path)[f"{position}/0"].to_tensorstore().write(data).result()
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
                opm_v2_frame_metadata(
                    frame_format="event_index",
                    index={"t": time, "p": 0, "c": 0, "z": scan},
                    daq_metadata={
                        "mode": "mirror",
                        "image_mirror_step_um": 0.4,
                        "current_channel": "488nm",
                        "laser_powers": [10.0],
                    },
                    opm_metadata={
                        "angle_deg": 30.0,
                        "camera_mirror_orientation": "positive",
                        "camera_XYstage_orientation": "positive",
                        "camera_Zstage_orientation": "positive",
                    },
                    stage_metadata={
                        "x_pos": 100.0,
                        "y_pos": 200.0,
                        "z_pos": 30.0,
                    },
                    camera_shape_yx=shape[-2:],
                    pixel_size_um=0.115,
                    camera_offset=100.0,
                    camera_conversion=0.25,
                    runner_time_ms=float(len(frames)),
                    exposure_ms=None,
                )
            )
    update_group_attributes(path, {"ome_writers": {"frame_metadata": frames}})
    return path


@pytest.fixture
def opm_v2_acquisition_factory(
    tmp_path, acquisition_psf_cache, integration_pipeline_records
):
    """Persist simulated camera frames through the installed OpmDataHandler.

    The real acquisition writer owns layout, NGFF and frame serialization.
    Processing-specific calibration is never added to its output.

    Returns
    -------
    callable
        Write TPCZYX camera pixels with explicitly supplied acquisition settings.
    """
    from opm_v2.handlers.opm_data_handler import OpmDataHandler
    from useq import MDAEvent, MDASequence

    def create(
        raw_data,
        *,
        name="sample",
        mode="mirror",
        suffix=".ome.zarr",
        frame_metadatas=None,
        frame_exposure_ms=10.0,
        live_manifest=None,
        **overrides,
    ):
        """Deliver every planned camera frame through the real writer lifecycle."""
        metadata = replace(
            simulated_acquisition_metadata(
                tmp_path / f"{name}{suffix}", mode, raw_data.shape
            ),
            channels=tuple(
                ChannelMetadata(index, label, None, 10.0, 10.0)
                for index, label in enumerate(
                    ("488 + 637", "561 + 637", "637nm")[: raw_data.shape[2]]
                )
            ),
            scan_axis_step_um=0.4,
            camera_conversion=0.25,
        )
        metadata = replace(metadata, **overrides)
        if metadata.deconvolution:
            raise ValueError(
                "As-acquired fixtures cannot contain processing calibration"
            )
        sizes = dict(zip("tpcz", raw_data.shape[:4], strict=True))
        order = metadata.acquisition_order
        events = []
        recorded_frames = {
            tuple(frame["mda_event"]["index"].get(axis, 0) for axis in order): frame
            for frame in (frame_metadatas or ())
        }
        for coordinates in np.ndindex(*(sizes[axis] for axis in order)):
            index = dict(zip(order, coordinates, strict=True))
            stage_z, stage_x, stage_y = metadata.stage_positions_zxy[index["p"]]
            events.append(
                MDAEvent(
                    index=index,
                    exposure=(
                        metadata.plane_period_s * 1000 / len(metadata.channels)
                        if metadata.plane_period_s is not None
                        else frame_exposure_ms
                    ),
                    metadata={
                        "DAQ": {
                            "mode": mode,
                            "current_channel": metadata.channels[index["c"]].name,
                            "scan_axis_step_um": metadata.scan_axis_step_um,
                            "image_mirror_step_um": metadata.scan_axis_step_um,
                        },
                        "Camera": {
                            "offset": metadata.camera_offset,
                            "e_to_ADU": metadata.camera_conversion,
                        },
                        "OPM": {
                            "angle_deg": metadata.angle_deg,
                            **dict(metadata.orientations),
                        },
                        "Stage": {"x_pos": stage_x, "y_pos": stage_y, "z_pos": stage_z},
                    },
                )
            )
            if recorded_frames:
                events[-1] = events[-1].model_copy(
                    update={
                        "metadata": recorded_frames[coordinates]["mda_event"][
                            "metadata"
                        ]
                    }
                )
        writer = OpmDataHandler(
            path=metadata.path,
            index_sizes=sizes,
            acquisition_order=order,
            events=events,
            acquisition_metadata={"acq_config": {"opm_mode": mode}},
            live_manifest=live_manifest,
        )
        sequence = MDASequence()
        writer.sequenceStarted(
            sequence, {"image_infos": [{"pixel_size_um": metadata.pixel_size_um}]}
        )
        try:
            for event in events:
                index = tuple(event.index[axis] for axis in "tpcz")
                writer.frameReady(
                    raw_data[index],
                    event,
                    {} if event.exposure is None else {"exposure_ms": event.exposure},
                )
            writer.sequenceFinished(sequence)
        finally:
            writer.close()
        assert writer.is_finalized
        from opm_processing.dataio.acquisition import (
            inspect_acquisition,
            open_acquisition_datastore,
        )

        metadata = inspect_acquisition(metadata.path)
        source = open_acquisition_datastore(metadata)
        kernels, parameters = acquisition_psf_cache(metadata)
        integration_pipeline_records["sources"].append(metadata.path)
        return SimpleNamespace(
            path=metadata.path,
            metadata=metadata,
            raw_data=raw_data,
            raw_arrays=tuple(
                source[:, position] for position in range(raw_data.shape[1])
            ),
            psfs=kernels,
            psf_parameters=parameters,
        )

    return create


@pytest.fixture(scope="session")
def acquisition_psf_cache():
    """Cache immutable production optical kernels resolved from real raw metadata.

    Returns
    -------
    callable
        Resolve every acquisition's metadata and reuse only identical optics.
        CUDA processing still generates and consumes its own production PSFs.
    """
    import json

    from opm_processing.dataio.deconvolution import (
        emission_wavelengths_from_metadata,
        load_deconvolution_psfs,
        psf_parameters_from_metadata,
        scan_upsample_from_metadata,
    )

    cache = {}

    def generate(metadata):
        """Evaluate the actual PSF loader once per complete physical configuration."""
        parameters = psf_parameters_from_metadata(metadata)
        wavelengths = emission_wavelengths_from_metadata(metadata)
        step = (
            None
            if metadata.is_2d
            else metadata.scan_axis_step_um / scan_upsample_from_metadata(metadata)
        )
        key = (
            metadata.pixel_size_um,
            metadata.angle_deg,
            metadata.is_2d,
            step,
            wavelengths,
            json.dumps(parameters, sort_keys=True),
        )
        if key not in cache:
            kernels, parameters = load_deconvolution_psfs(
                metadata,
                projection=metadata.is_2d,
                reconstruction_step_um=step,
            )
            for kernel in kernels:
                kernel.setflags(write=False)
            cache[key] = (kernels, parameters)
        return cache[key]

    return generate


@pytest.fixture
def acquisition_factory(tmp_path):
    """Build synthetic acquisition layouts for isolated component tests.

    The factory accepts a TPCZYX array, a dataset name, a scan mode and keyword
    overrides of AcquisitionMetadata fields. Defaults describe a 30 degree OPM
    with 115 nm detector pixels, 400 nm scans, 100 ADU offset and 0.25 photons/ADU.
    It returns the path, metadata, original pixels and reopened disk collection.
    This fixture does not establish compatibility with the actual opm-v2 writer.
    """

    def create(raw_data, *, name="sample", mode="mirror", **overrides):
        """Write one independent dataset with overrides of its physical calibration."""
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


@pytest.fixture(scope="session")
def acquisition_metadata():
    """Provide frozen physical calibration for checks that do not read pixels."""
    return simulated_acquisition_metadata(
        Path("calibration.ome.zarr"), "mirror", (1, 1, 1, 3, 4, 5)
    )


@pytest.fixture
def deconvolution_acquisition(acquisition_factory):
    """Provide a tiny raw store with independently variable calibration attributes."""
    return acquisition_factory(np.zeros((1, 1, 1, 3, 4, 5), dtype=np.uint16))


@pytest.fixture(params=("mirror", "stage"))
def scan_camera_acquisition(request, acquisition_factory):
    """Known numbered planes with an explicit excess-plane and scan convention."""
    expected = np.broadcast_to(
        np.arange(1, 17, dtype=np.uint16)[:, None, None], (16, 32, 32)
    ).copy()
    stage = request.param == "stage"
    ordered = (
        np.concatenate((np.zeros((1, 32, 32), np.uint16), expected))
        if stage
        else expected
    )
    raw = ordered[::-1] if stage else ordered
    dataset = acquisition_factory(
        raw[None, None, None],
        mode=request.param,
        camera_offset=0,
        camera_conversion=1,
        excess_scan_positions=int(stage),
        excess_scan_start_positions=int(stage),
    )
    return SimpleNamespace(
        path=dataset.path, raw_data=dataset.raw_data, expected=expected
    )


@pytest.fixture
def processing_options():
    """Return common settings for a full-resolution isolated volume workflow."""
    return {
        "flatfield_correction": False,
        "max_projection": False,
        "create_fused_max_projection": False,
        "z_downsample_level": 1,
    }


@pytest.fixture(params=("mirror", "stage"))
def inspection_case(
    request,
    simulated_camera,
    opm_v2_acquisition_factory,
    processing_options,
    selection,
    small_cpu_kernels,
):
    """Prepare CLI inputs from a simulated object and real deskew processing.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Mirror or stage scan mode.
    simulated_camera : callable
        Shared physical forward model for a known fluorescent object.
    opm_v2_acquisition_factory : callable
        Native writer for calibrated opm-v2 acquisition metadata and pixels.
    processing_options : dict
        Shared settings preserving the laboratory output grid.
    selection : str
        Raw, directory, processed, custom-name or unavailable-source scenario.
    small_cpu_kernels : None
        Production CPU kernels configured for these small simulated volumes.

    Returns
    -------
    types.SimpleNamespace
        Simulated acquisition, actual processed collection and CLI input path.
    """
    from opm_processing.dataio.acquisition import acquisition_stem
    from opm_processing.dataio.processing_state import ProcessingState
    from opm_processing.process import process

    observed = simulated_camera((16, 32, 32))
    pixels = np.broadcast_to(observed.raw, (1, 2, 1, *observed.raw.shape)).copy()
    if request.param == "stage":
        pixels = pixels[:, :, :, ::-1].copy()
    case = opm_v2_acquisition_factory(
        pixels,
        name="inspection",
        mode=request.param,
        acquired_plane_count=pixels.shape[3],
        plane_period_s=0.01,
    )
    path = case.path
    output = path.parent / f"{acquisition_stem(path)}_deskewed.ome.zarr"
    collection = None
    if selection != "raw-only":
        process(path, **processing_options)
        if selection == "custom-processed":
            output = output.rename(path.parent / "custom-output.ome.zarr")
            state = ProcessingState.create(path.parent / "custom.processing.json", path)
            state.initialize_run(output, configuration={}, overwrite=True)
        collection = open_position_collection(output)
    if selection == "processed-only":
        path.rename(path.parent / "offline-raw")
    selected = (
        path.parent
        if selection == "directory"
        else output
        if "processed" in selection
        else path
    )
    return SimpleNamespace(case=case, collection=collection, selected=selected)


@pytest.fixture
def calibrated_mirror_timelapse(simulated_camera, opm_v2_acquisition_factory):
    """Write three simulated exposures through opm-v2's single-position layout."""
    from opm_processing.imageprocessing.opmtools import orthogonal_deskew

    raw = np.stack(
        [simulated_camera((4, 5, 6), exposure=1 + time / 4).raw for time in range(3)]
    )[:, None, None]
    dataset = opm_v2_acquisition_factory(
        raw,
        name="single_mirror",
        frame_exposure_ms=5.0021029411764706,
    )
    expected = [
        orthogonal_deskew(
            np.maximum(volume.astype(np.float32) - 100, 0) * 0.25,
            theta=30.0,
            distance=0.4,
            pixel_size=0.115,
            downsample_factor=1,
        )
        for volume in raw[:, 0, 0]
    ]
    return SimpleNamespace(path=dataset.path, expected=expected)
