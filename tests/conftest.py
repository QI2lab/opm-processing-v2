"""Shared fixtures representing upstream acquisition formats."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts
from scipy import ndimage

from opm_processing.imageprocessing.opmtools import deskew_shape_estimator


@pytest.fixture(scope="module")
def cupy_gpu():
    """Return CuPy after proving CUDA execution, or honor GPU skip policy.

    Parameters
    ----------
    None
        This callable has no parameters.

    Returns
    -------
    object
        Result produced by the callable.
    """

    def unavailable(message: str) -> None:
        """Fail required GPU runs or skip optional GPU runs.

        Parameters
        ----------
        message : str
            Value supplied for ``message``.

        Returns
        -------
        None
            No value is returned.
        """
        if os.environ.get("OPM_REQUIRE_GPU") == "1":
            pytest.fail(message, pytrace=False)
        pytest.skip(message)

    try:
        import cupy as cp
    except ImportError:
        unavailable("CuPy is not installed; install the project's gpu extra")

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            unavailable("CuPy found no CUDA devices")
        device = cp.cuda.Device(0)
        device.use()
        probe = cp.arange(32, dtype=cp.float32)
        probe = probe * probe + cp.float32(3)
        device.synchronize()
        np.testing.assert_array_equal(
            cp.asnumpy(probe),
            np.arange(32, dtype=np.float32) ** 2 + 3,
        )
    except pytest.skip.Exception:
        raise
    except Exception as error:  # CUDA errors vary with runtime/driver versions.
        unavailable(f"CUDA execution failed: {error}")

    yield cp

    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


@dataclass(frozen=True)
class OpmV2ProjectionFixture:
    """An acquisition written in QI2lab/opm-v2 OPMMirrorHandler layout."""

    path: Path
    raw_data: np.ndarray
    camera_offset: float
    camera_conversion: float
    pixel_size_um: float
    channel_names: tuple[str, ...]
    stage_positions_zxy: np.ndarray


@dataclass(frozen=True)
class OpmV2SkewedFixture:
    """A mirror- or stage-scanned OPMMirrorHandler acquisition."""

    path: Path
    mode: str
    raw_data: np.ndarray
    camera_offset: float
    camera_conversion: float
    pixel_size_um: float
    scan_axis_step_um: float
    excess_scan_positions: int
    channel_names: tuple[str, ...]
    stage_positions_zxy: np.ndarray


@dataclass(frozen=True)
class OpmV2TiledGroundTruthFixture:
    """A blurred multi-tile acquisition rendered from a known 3D object."""

    path: Path
    configuration: str
    mode: str
    ground_truth: np.ndarray
    raw_data: np.ndarray
    true_stage_positions_zxy: np.ndarray
    recorded_stage_positions_zxy: np.ndarray
    tile_offsets_zyx_px: tuple[tuple[int, int, int], ...]
    recorded_position_errors_zyx_px: np.ndarray
    ellipsoids_zyx_radii: tuple[tuple[float, float, float, float, float, float], ...]
    pixel_size_um: float
    scan_axis_step_um: float
    psf_path: Path


@dataclass(frozen=True)
class TiledAcquisitionConfig:
    """Immutable spatial and acquisition configuration for a synthetic run."""

    name: str
    mode: str
    tile_offsets_zyx_px: tuple[tuple[int, int, int], ...]
    recorded_position_errors_zyx_px: tuple[tuple[int, int, int], ...]
    camera_shape_zyx: tuple[int, int, int] = (16, 32, 32)
    pixel_size_um: float = 0.115
    scan_axis_step_um: float = 0.4
    theta_deg: float = 30.0
    rng_seed: int = 2468


def _tiled_acquisition_config(
    name: str, *, mode: str = "mirror"
) -> TiledAcquisitionConfig:
    """Build one named spatial configuration without module-level constants.

    Parameters
    ----------
    name : str
        Value supplied for ``name``.
    mode : str
        Value supplied for ``mode``.

    Returns
    -------
    TiledAcquisitionConfig
        Result produced by the callable.
    """
    configurations = {
        "x_overlap": (
            ((0, 0, 0), (0, 0, 14)),
            ((0, 0, 0), (0, 0, 2)),
        ),
        "yx_grid": (
            ((0, 0, 0), (0, 0, 14), (0, 42, 0), (0, 42, 14)),
            ((0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2)),
        ),
        "z_staggered": (
            ((0, 0, 0), (4, 0, 0), (8, 0, 0)),
            ((0, 0, 0), (1, 0, 0), (1, 0, 0)),
        ),
        "yx_grid_z_staggered": (
            ((0, 0, 0), (2, 0, 14), (4, 42, 0), (6, 42, 14)),
            ((0, 0, 0), (1, 0, 2), (1, 2, 0), (1, 2, 2)),
        ),
    }
    try:
        offsets, errors = configurations[name]
    except KeyError as error:
        raise ValueError(f"Unknown tiled acquisition configuration: {name}") from error
    return TiledAcquisitionConfig(
        name=name,
        mode=mode,
        tile_offsets_zyx_px=offsets,
        recorded_position_errors_zyx_px=errors,
    )


def _opm_v2_frame_metadata(
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
    index : dict[str, int]
        Value supplied for ``index``.
    daq_metadata : dict
        Value supplied for ``daq metadata``.
    opm_metadata : dict
        Value supplied for ``opm metadata``.
    stage_metadata : dict
        Value supplied for ``stage metadata``.
    camera_shape_yx : tuple[int, int]
        Value supplied for ``camera shape yx``.
    pixel_size_um : float
        Value supplied for ``pixel size um``.
    camera_offset : float
        Value supplied for ``camera offset``.
    camera_conversion : float
        Value supplied for ``camera conversion``.
    runner_time_ms : float
        Value supplied for ``runner time ms``.
    hardware_triggered : bool | None
        Value supplied for ``hardware triggered``.
    additional_event_metadata : dict | None
        Value supplied for ``additional event metadata``.

    Returns
    -------
    dict
        Result produced by the callable.
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


def _write_opm_v2_zarr(
    *,
    path: Path,
    raw_data: np.ndarray,
    labels: tuple[str, ...],
    chunks: tuple[int, ...],
    frame_metadatas: list[dict],
) -> None:
    """Write image data and handler-style root metadata as Zarr v2.

    Parameters
    ----------
    path : Path
        Value supplied for ``path``.
    raw_data : np.ndarray
        Value supplied for ``raw data``.
    labels : tuple[str, ...]
        Value supplied for ``labels``.
    chunks : tuple[int, ...]
        Value supplied for ``chunks``.
    frame_metadatas : list[dict]
        Value supplied for ``frame metadatas``.

    Returns
    -------
    None
        No value is returned.
    """
    store = ts.open(
        {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(path)},
        },
        create=True,
        delete_existing=True,
        dtype=ts.uint16,
        shape=raw_data.shape,
        chunk_layout=ts.ChunkLayout(chunk_shape=chunks),
        domain=ts.IndexDomain(shape=raw_data.shape, labels=labels),
    ).result()
    store.write(raw_data).result()
    (path / ".zattrs").write_text(
        json.dumps({"frame_metadatas": frame_metadatas}),
        encoding="utf-8",
    )


@pytest.fixture
def opm_v2_projection_zarr(tmp_path) -> OpmV2ProjectionFixture:
    """Create a small, faithful OPMMirrorHandler projection acquisition.

    Schema provenance (QI2lab/opm-v2 main, commit ee354a1421732218d1128332c42a93aef106aa33):
    - ``handlers/opm_mirror_handler.py`` defines shape/chunks/labels and .zattrs.
    - ``engine/setup_events_v2.py`` defines projection event metadata.
    - pymmcore-plus ``FrameMetaV1`` defines the enclosing frame metadata.

    Parameters
    ----------
    tmp_path : object
        Value supplied for ``tmp path``.

    Returns
    -------
    OpmV2ProjectionFixture
        Result produced by the callable.
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
                _opm_v2_frame_metadata(
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

    _write_opm_v2_zarr(
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


@pytest.fixture(params=("mirror", "stage"), ids=("mirror-scan", "stage-scan"))
def opm_v2_skewed_zarr(request, tmp_path) -> OpmV2SkewedFixture:
    """Create normal skewed OPM-v2 acquisitions for both scan mechanisms.

    Parameters
    ----------
    request : object
        Value supplied for ``request``.
    tmp_path : object
        Value supplied for ``tmp path``.

    Returns
    -------
    OpmV2SkewedFixture
        Result produced by the callable.
    """
    mode = str(request.param)
    path = tmp_path / f"opm_v2_{mode}.zarr"
    shape = (1, 1, 1, 8, 12, 14)  # T, P, C, Z, Y, X
    chunks = (1, 1, 1, 1, shape[-2], shape[-1])
    zz, yy, xx = np.mgrid[: shape[-3], : shape[-2], : shape[-1]]
    raw_data = np.empty(shape, dtype=np.uint16)
    raw_data[0, 0, 0] = 220 + 8 * zz + 3 * yy + 2 * xx

    camera_offset = 100.0
    camera_conversion = 0.25
    pixel_size_um = 0.115
    scan_axis_step_um = 0.4
    excess_scan_positions = 1 if mode == "stage" else 0
    channel_names = ("488nm",)
    stage_positions_zxy = np.array([[4.0, 20.0, 30.0]])
    frame_metadatas = []
    for scan in range(shape[3]):
        daq_metadata = {
            "mode": mode,
            "channel_states": [True],
            "exposure_channels_ms": [10.0],
            "laser_powers": [10.0],
            "interleaved": True,
            "blanking": True,
            "current_channel": channel_names[0],
        }
        opm_metadata = {
            "angle_deg": 30.0,
            "camera_Zstage_orientation": "normal",
            "camera_XYstage_orientation": "normal",
            "camera_mirror_orientation": "normal",
        }
        stage_x = stage_positions_zxy[0, 1]
        if mode == "mirror":
            daq_metadata.update(
                {
                    "image_mirror_position": float(scan),
                    "image_mirror_range_um": shape[3] * scan_axis_step_um,
                    "image_mirror_step_um": scan_axis_step_um,
                }
            )
        else:
            daq_metadata["scan_axis_step_um"] = scan_axis_step_um
            opm_metadata.update(
                {
                    "excess_scan_positions": excess_scan_positions,
                    "excess_scan_start_positions": excess_scan_positions,
                    "excess_scan_end_positions": excess_scan_positions,
                }
            )
            stage_x += scan * scan_axis_step_um

        frame_metadatas.append(
            _opm_v2_frame_metadata(
                index={"t": 0, "p": 0, "c": 0, "z": scan},
                daq_metadata=daq_metadata,
                opm_metadata=opm_metadata,
                stage_metadata={
                    "x_pos": stage_x,
                    "y_pos": stage_positions_zxy[0, 2],
                    "z_pos": stage_positions_zxy[0, 0],
                    "excess_image": (mode == "stage" and scan < excess_scan_positions),
                },
                camera_shape_yx=shape[-2:],
                pixel_size_um=pixel_size_um,
                camera_offset=camera_offset,
                camera_conversion=camera_conversion,
                runner_time_ms=float(scan),
                hardware_triggered=True,
            )
        )

    _write_opm_v2_zarr(
        path=path,
        raw_data=raw_data,
        labels=("t", "p", "c", "z", "y", "x"),
        chunks=chunks,
        frame_metadatas=frame_metadatas,
    )
    return OpmV2SkewedFixture(
        path=path,
        mode=mode,
        raw_data=raw_data,
        camera_offset=camera_offset,
        camera_conversion=camera_conversion,
        pixel_size_um=pixel_size_um,
        scan_axis_step_um=scan_axis_step_um,
        excess_scan_positions=excess_scan_positions,
        channel_names=channel_names,
        stage_positions_zxy=stage_positions_zxy,
    )


def _create_opm_v2_tiled_ground_truth_zarr(
    tmp_path: Path,
    *,
    config: TiledAcquisitionConfig,
) -> OpmV2TiledGroundTruthFixture:
    """Render an overlapping tiled OPM-v2 acquisition from a known object.

    The forward model samples a laboratory-frame ground truth on the tilted
    camera planes used by OPM and convolves each skewed stack with the same
    compact PSF supplied to processing.

    Parameters
    ----------
    tmp_path : Path
        Value supplied for ``tmp path``.
    config : TiledAcquisitionConfig
        Value supplied for ``config``.

    Returns
    -------
    OpmV2TiledGroundTruthFixture
        Result produced by the callable.
    """
    if len(config.tile_offsets_zyx_px) != len(config.recorded_position_errors_zyx_px):
        raise ValueError("Each tile must have one recorded position error")
    mode = config.mode
    path = tmp_path / f"opm_v2_{mode}_{config.name}.zarr"
    pixel_size_um = config.pixel_size_um
    scan_axis_step_um = config.scan_axis_step_um
    theta_deg = config.theta_deg
    camera_shape = config.camera_shape_zyx
    excess_scan_positions = 1 if mode == "stage" else 0
    stored_scan_count = camera_shape[0] + excess_scan_positions
    shape = (
        1,
        len(config.tile_offsets_zyx_px),
        1,
        stored_scan_count,
        camera_shape[1],
        camera_shape[2],
    )
    tile_zyx, _, _, _ = deskew_shape_estimator(
        camera_shape,
        theta=theta_deg,
        distance=scan_axis_step_um,
        pixel_size=pixel_size_um,
        crop_after_deskew=False,
    )
    max_offsets = np.max(np.asarray(config.tile_offsets_zyx_px), axis=0)
    ground_truth_shape = tuple(
        int(tile_size + offset) for tile_size, offset in zip(tile_zyx, max_offsets)
    )

    rng = np.random.default_rng(config.rng_seed)
    ground_truth = np.zeros(ground_truth_shape, dtype=np.float32)
    ellipsoids = []
    for z_index, center_z in enumerate(
        np.arange(4.0, ground_truth_shape[0] - 2.0, 5.0)
    ):
        for y_index, center_y in enumerate(
            np.arange(12.0, ground_truth_shape[1] - 5.0, 18.0)
        ):
            for x_index, center_x in enumerate(
                np.arange(7.0, ground_truth_shape[2] - 4.0, 11.0)
            ):
                if (z_index + y_index + x_index) % 2 == 0:
                    ellipsoids.append(
                        (
                            center_z,
                            center_y,
                            center_x,
                            2.5 + 0.2 * (x_index % 2),
                            5.5 + 0.5 * (z_index % 2),
                            4.0 + 0.4 * (y_index % 2),
                        )
                    )
    ellipsoids_zyx_radii = tuple(ellipsoids)
    zz, yy, xx = np.indices(ground_truth_shape, dtype=np.float32)
    for (
        center_z,
        center_y,
        center_x,
        radius_z,
        radius_y,
        radius_x,
    ) in ellipsoids_zyx_radii:
        elliptical_radius = np.sqrt(
            ((zz - center_z) / radius_z) ** 2
            + ((yy - center_y) / radius_y) ** 2
            + ((xx - center_x) / radius_x) ** 2
        )
        ground_truth += 6500.0 * np.exp(-0.5 * ((elliptical_radius - 1.0) / 0.13) ** 2)
    ground_truth = np.clip(ground_truth, 0.0, 40_000.0)

    scan, camera_y, camera_x = np.indices(camera_shape, dtype=np.float32)
    theta_rad = np.deg2rad(theta_deg)
    sample_z = camera_y * np.sin(theta_rad)
    sample_y = scan * (scan_axis_step_um / pixel_size_um) + camera_y * np.cos(theta_rad)
    psf_z, psf_y, psf_x = np.mgrid[-2:3, -3:4, -3:4]
    psf = np.exp(
        -(psf_z**2 / 1.0**2 + psf_y**2 / 1.6**2 + psf_x**2 / 1.6**2) / 2
    ).astype(np.float32)
    psf /= psf.sum()
    psf_path = tmp_path / f"{mode}_synthetic_psf.npy"
    np.save(psf_path, psf)

    raw_data = np.zeros(shape, dtype=np.uint16)
    for position, (tile_z_offset, tile_y_offset, tile_x_offset) in enumerate(
        config.tile_offsets_zyx_px
    ):
        tile_sample_z = sample_z + tile_z_offset
        tile_sample_y = sample_y + tile_y_offset
        sample_x = camera_x + tile_x_offset
        ideal_skewed = ndimage.map_coordinates(
            ground_truth,
            (tile_sample_z, tile_sample_y, sample_x),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        blurred = ndimage.convolve(ideal_skewed, psf, mode="constant", cval=0.0)
        blurred = rng.poisson(np.clip(blurred, 0.0, None)).astype(np.uint16)
        if mode == "stage":
            # process_skewed reverses stage scans and then removes the leading
            # excess plane. Arrange stored data so that operation recovers the
            # same physical scan order as the mirror acquisition.
            post_flip = np.concatenate((np.zeros_like(blurred[:1]), blurred), axis=0)
            blurred = post_flip[::-1]
        raw_data[0, position, 0] = blurred

    true_offsets = np.asarray(config.tile_offsets_zyx_px, dtype=np.float64)
    recorded_errors = np.asarray(
        config.recorded_position_errors_zyx_px, dtype=np.float64
    )
    true_stage_positions_zxy = true_offsets * pixel_size_um
    recorded_stage_positions_zxy = (true_offsets + recorded_errors) * pixel_size_um
    frame_metadatas = []
    for position in range(shape[1]):
        for stored_scan in range(shape[3]):
            daq_metadata = {
                "mode": mode,
                "channel_states": [True],
                "exposure_channels_ms": [10.0],
                "laser_powers": [10.0],
                "interleaved": True,
                "blanking": True,
                "current_channel": "488nm",
            }
            opm_metadata = {
                "angle_deg": theta_deg,
                "camera_Zstage_orientation": "normal",
                "camera_XYstage_orientation": "normal",
                "camera_mirror_orientation": "normal",
            }
            stage_x = recorded_stage_positions_zxy[position, 1]
            if mode == "mirror":
                daq_metadata.update(
                    {
                        "image_mirror_position": float(stored_scan),
                        "image_mirror_range_um": (camera_shape[0] * scan_axis_step_um),
                        "image_mirror_step_um": scan_axis_step_um,
                    }
                )
            else:
                daq_metadata["scan_axis_step_um"] = scan_axis_step_um
                opm_metadata.update(
                    {
                        "excess_scan_positions": excess_scan_positions,
                        "excess_scan_start_positions": excess_scan_positions,
                        "excess_scan_end_positions": excess_scan_positions,
                    }
                )
                stage_x += stored_scan * scan_axis_step_um

            frame_metadatas.append(
                _opm_v2_frame_metadata(
                    index={
                        "t": 0,
                        "p": position,
                        "c": 0,
                        "z": stored_scan,
                    },
                    daq_metadata=daq_metadata,
                    opm_metadata=opm_metadata,
                    stage_metadata={
                        "x_pos": stage_x,
                        "y_pos": recorded_stage_positions_zxy[position, 2],
                        "z_pos": recorded_stage_positions_zxy[position, 0],
                        "excess_image": (
                            mode == "stage" and stored_scan < excess_scan_positions
                        ),
                    },
                    camera_shape_yx=camera_shape[1:],
                    pixel_size_um=pixel_size_um,
                    camera_offset=0.0,
                    camera_conversion=1.0,
                    runner_time_ms=float(position * shape[3] + stored_scan),
                    hardware_triggered=True,
                )
            )

    _write_opm_v2_zarr(
        path=path,
        raw_data=raw_data,
        labels=("t", "p", "c", "z", "y", "x"),
        chunks=(1, 1, 1, 1, camera_shape[1], camera_shape[2]),
        frame_metadatas=frame_metadatas,
    )
    return OpmV2TiledGroundTruthFixture(
        path=path,
        configuration=config.name,
        mode=mode,
        ground_truth=ground_truth,
        raw_data=raw_data,
        true_stage_positions_zxy=true_stage_positions_zxy,
        recorded_stage_positions_zxy=recorded_stage_positions_zxy,
        tile_offsets_zyx_px=config.tile_offsets_zyx_px,
        recorded_position_errors_zyx_px=recorded_errors,
        ellipsoids_zyx_radii=ellipsoids_zyx_radii,
        pixel_size_um=pixel_size_um,
        scan_axis_step_um=scan_axis_step_um,
        psf_path=psf_path,
    )


@pytest.fixture(params=("mirror", "stage"), ids=("mirror-tiled", "stage-tiled"))
def opm_v2_tiled_ground_truth_zarr(request, tmp_path) -> OpmV2TiledGroundTruthFixture:
    """Create the original two-tile X-overlap acquisition in both scan modes.

    Parameters
    ----------
    request : object
        Value supplied for ``request``.
    tmp_path : object
        Value supplied for ``tmp path``.

    Returns
    -------
    OpmV2TiledGroundTruthFixture
        Result produced by the callable.
    """
    return _create_opm_v2_tiled_ground_truth_zarr(
        tmp_path,
        config=_tiled_acquisition_config("x_overlap", mode=str(request.param)),
    )


@pytest.fixture(
    params=("yx_grid", "z_staggered", "yx_grid_z_staggered"),
    ids=("yx-grid", "z-staggered", "yx-grid-z-staggered"),
)
def opm_v2_spatial_tiling_ground_truth_zarr(
    request,
    tmp_path,
) -> OpmV2TiledGroundTruthFixture:
    """Create each distinct multi-axis tiling configuration as a test case.

    Parameters
    ----------
    request : object
        Value supplied for ``request``.
    tmp_path : object
        Value supplied for ``tmp path``.

    Returns
    -------
    OpmV2TiledGroundTruthFixture
        Result produced by the callable.
    """
    return _create_opm_v2_tiled_ground_truth_zarr(
        tmp_path,
        config=_tiled_acquisition_config(str(request.param)),
    )
