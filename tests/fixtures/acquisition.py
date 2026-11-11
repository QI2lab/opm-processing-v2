"""Actual producer acquisitions and synthetic layouts for isolated metadata units."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
from yaozarrs import open_group

from opm_processing.dataio.acquisition import AcquisitionMetadata, ChannelMetadata
from opm_processing.dataio.position_collection import (
    create_position_collection,
    open_position_collection,
    update_group_attributes,
)


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
    camera_read_noise_e: float | None = None,
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
            **(
                {"read_noise_e": camera_read_noise_e}
                if camera_read_noise_e is not None
                else {}
            ),
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


@pytest.fixture
def opm_v2_projection_zarr(
    request,
    optical_camera_data,
    opm_v2_acquisition_factory,
    simulation_parameters,
    uniform_optical_calibration,
):
    """Simulate spectral/time observations and persist them with OpmDataHandler."""
    mode, illumination_factor, *counts = getattr(request, "param", ("projection", 1.0))
    positions = counts[0] if counts else 1
    from tests.simulation.camera import digitize

    shape = (2, positions, 2, *simulation_parameters["camera_shape_yx"])
    raw_data = np.empty(shape, dtype=np.uint16)
    expectation = np.empty(shape, dtype=np.float64)
    _, microscope, source = optical_camera_data(
        (1, *shape[-2:]),
        kind="points",
        large=False,
        integration=False,
        illumination_profile="uniform",
        step=0.4,
        geometry="cartesian",
        budget=(1000, 0),
    )
    microscope = replace(
        microscope, camera=replace(microscope.camera, electrons_per_adu=0.25)
    )
    background = uniform_optical_calibration(
        microscope,
        (1, *shape[-2:]),
        geometry="cartesian",
        photon_budget=(2000, 0),
    )[0]
    for time, position, channel in np.ndindex(shape[:3]):
        exposure = (1 + time / 4 + channel / 2 + position / 8) * illumination_factor
        # The uniform fluorescent background spans the full physical detector,
        # independently of the compact point object's finite storage domain.
        optical = source[0] * exposure + exposure * background
        expectation[time, position, channel] = optical
        raw_data[time, position, channel] = digitize(
            optical,
            microscope.camera,
            seed=42 + 100 * time + 10 * position + channel,
        )
    assert raw_data.max() < np.iinfo(np.uint16).max
    dataset = opm_v2_acquisition_factory(
        raw_data[:, :, :, None],
        name="opm_v2_projection",
        mode=mode,
        suffix=".zarr",
        camera_read_noise_e=microscope.camera.read_noise_e,
        stage_positions_zxy=tuple(
            (
                4.0,
                20.0,
                30.0 + p * (shape[-1] + 2) * microscope.pixel_size_um,
            )
            for p in range(positions)
        ),
    )
    dataset.expectation = expectation[:, :, :, None]
    dataset.microscope = microscope
    return dataset


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
    """Write prescribed camera metadata and pixels for isolated component units."""
    index_sizes = dict(zip("tpc", raw.shape[:3], strict=True))
    if raw.shape[3] > 1:
        index_sizes["z"] = raw.shape[3]
    collection = create_position_collection(
        metadata.path,
        raw.shape,
        (1.0, metadata.pixel_size_um, metadata.pixel_size_um),
        stage_positions=metadata.stage_positions_zxy,
        channels=tuple(channel.name for channel in metadata.channels),
        attributes={
            "opm_v2": {
                "index_sizes": index_sizes,
                "acquisition_order": [
                    axis for axis in metadata.axes if axis not in {"y", "x"}
                ],
                "configuration": {"acq_config": {}},
            }
        },
        chunks=(1, 1, 1, *raw.shape[-2:]),
    )
    frame_count = 0
    for position, (stage_z, stage_x, stage_y) in enumerate(
        metadata.stage_positions_zxy
    ):
        frames = []
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
                            runner_time_ms=float(frame_count),
                            exposure_ms=(
                                1000 * metadata.plane_period_s / len(metadata.channels)
                                if metadata.plane_period_s is not None
                                else None
                            ),
                            hardware_triggered=True,
                        )
                    )
                    frame_count += 1
        collection.arrays[position].write(raw[:, position]).result()
        update_group_attributes(
            metadata.path / str(position), {"ome_writers": {"frame_metadata": frames}}
        )

    if metadata.deconvolution is not None:
        update_group_attributes(
            metadata.path, {"opm_processing": {"deconvolution": metadata.deconvolution}}
        )


@pytest.fixture
def current_opm_v2_stage_scan(opm_v2_acquisition_factory) -> Path:
    """Write the actual producer layout, then prescribe metadata-reader edge cases."""
    position, channel, scan, row, column = np.indices((2, 2, 3, 4, 5))
    pixels = (1000 * position + 100 * channel + 10 * scan + 2 * row + column).astype(
        np.uint16
    )[None]
    channels = (
        ChannelMetadata(0, "488nm", 488.0, 10.0, 12.0),
        ChannelMetadata(1, "561nm", 561.0, 15.0, 18.0),
    )
    dataset = opm_v2_acquisition_factory(
        pixels,
        name="current_stage",
        mode="stage",
        channels=channels,
        camera_conversion=0.24,
        stage_positions_zxy=((30.0, 100.0, 200.0), (30.0, 100.0, 220.0)),
        orientations=(
            ("camera_Zstage_orientation", "negative"),
            ("camera_XYstage_orientation", "positive"),
            ("camera_mirror_orientation", "positive"),
        ),
    )
    root = open_group(dataset.path)
    opm = dict(root.attrs["opm_v2"])
    # These prescribed configuration/trajectory records exercise reader precedence;
    # the producer above owns all image layout, pixel writes and frame serialization.
    opm["configuration"]["acq_config"]["DAQ"] = {
        "channel_states": [True, True, False],
        "channel_powers": [12.0, 18.0, 0.0],
        "channel_exposures_ms": [10.0, 15.0, 0.0],
        "scan_axis_step_um": 0.4,
    }
    update_group_attributes(dataset.path, {"opm_v2": opm})
    for position in range(2):
        writer = dict(root[str(position)].attrs["ome_writers"])
        for frame in writer["frame_metadata"]:
            index, event = frame["event_index"], frame["event_metadata"]
            exposure = channels[index["c"]].exposure_ms
            frame["exposure_time"] = exposure / 1000
            event["Camera"]["exposure_ms"] = exposure
            event["Stage"]["x_pos"] = 100.0 + 0.4 * index["z"]
        update_group_attributes(dataset.path / str(position), {"ome_writers": writer})
    return dataset.path


@pytest.fixture
def current_single_position_mirror_timelapse(opm_v2_acquisition_factory) -> Path:
    """Write the actual root Image with prescribed missing camera timing."""
    pixels = np.arange(3 * 4 * 5 * 6, dtype=np.uint16).reshape(3, 1, 1, 4, 5, 6) + 200
    dataset = opm_v2_acquisition_factory(
        pixels,
        name="single_mirror",
        mode="mirror",
        frame_exposure_ms=2.0,
        channels=(ChannelMetadata(0, "488nm", 488.0, 2.0, 10.0),),
        stage_positions_zxy=((30.0, 100.0, 200.0),),
        orientations=(
            ("camera_mirror_orientation", "positive"),
            ("camera_XYstage_orientation", "positive"),
            ("camera_Zstage_orientation", "positive"),
        ),
    )
    root = open_group(dataset.path)
    writer = dict(root.attrs["ome_writers"])
    # The writer emits exposure_time. Removing it is an explicit missing-metadata
    # unit input, not a claim that the current producer omitted its camera timing.
    for frame in writer["frame_metadata"]:
        frame.pop("exposure_time")
    opm = dict(root.attrs["opm_v2"])
    opm["configuration"]["acq_config"]["DAQ"] = {
        "channel_states": [True],
        "channel_powers": [10.0],
        "channel_exposures_ms": [2.0],
    }
    update_group_attributes(dataset.path, {"ome_writers": writer, "opm_v2": opm})
    return dataset.path


@pytest.fixture
def opm_v2_acquisition_factory(tmp_path):
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
        camera_read_noise_e=None,
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
                            **(
                                {"read_noise_e": camera_read_noise_e}
                                if camera_read_noise_e is not None
                                else {}
                            ),
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
        return SimpleNamespace(
            path=metadata.path,
            metadata=metadata,
            raw_data=raw_data,
            raw_arrays=tuple(
                source[:, position] for position in range(raw_data.shape[1])
            ),
        )

    return create


@pytest.fixture
def acquisition_factory(tmp_path):
    """Build synthetic acquisition layouts for isolated component tests.

    The factory accepts a TPCZYX array, a dataset name, a scan mode and keyword
    overrides of AcquisitionMetadata fields. Defaults describe a 30 degree OPM
    with 115 nm detector pixels, 400 nm scans, 100 ADU offset and 0.25 photons/ADU.
    It returns the path, metadata and original pixels.
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
    opm_v2_acquisition_factory,
    position_store_factory,
    selection,
):
    """Write prescribed camera and Cartesian grids for metadata/path inspection."""
    from opm_processing.dataio.acquisition import acquisition_stem
    from opm_processing.dataio.processing_state import ProcessingState

    pixels = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(1, 2, 1, 3, 4, 5) + 100
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
        position_store_factory(
            pixels,
            name=output.name.removesuffix(".ome.zarr"),
            voxel_size_um=(case.metadata.pixel_size_um,) * 3,
            channels=case.metadata.channel_names,
            stage_positions=case.metadata.stage_positions_zxy,
            time_interval_s=case.metadata.volume_interval_s,
        )
        journal = path.parent / "inspection.processing.json"
        if selection == "custom-processed":
            output = output.rename(path.parent / "custom-output.ome.zarr")
            journal = path.parent / "custom.processing.json"
        state = ProcessingState.create(journal, path)
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
def calibrated_mirror_timelapse(roi_camera_scene, opm_v2_acquisition_factory):
    """Write independently noisy exposures containing both original optical emitters."""
    from opm_processing.imageprocessing.opmtools import deskew_shape_estimator

    scene = roi_camera_scene(times=3)
    dataset = opm_v2_acquisition_factory(
        scene.pixels,
        name="single_mirror",
        camera_conversion=1.0,
        camera_read_noise_e=scene.microscope.camera.read_noise_e,
        stage_positions_zxy=((0.0, 0.0, 0.0),),
        frame_exposure_ms=5.0021029411764706,
    )
    metadata = dataset.metadata
    shape, *_ = deskew_shape_estimator(
        scene.pixels.shape[-3:],
        metadata.angle_deg,
        metadata.scan_axis_step_um,
        metadata.pixel_size_um,
        crop_after_deskew=False,
    )
    return SimpleNamespace(
        path=dataset.path,
        truth=scene,
        shape_zyx=tuple(shape),
    )
