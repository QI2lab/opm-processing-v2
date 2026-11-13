"""Estimate illumination fields for OPM image correction."""

from __future__ import annotations

import gc
import io
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from opm_processing.cuda import preload_cuda_libraries
from opm_processing.imageprocessing.camera import (
    camera_correct,
)
from opm_processing.imageprocessing.coordinates import stage_z_level_indices

preload_cuda_libraries()

import torch
import torch.nn.functional as F
from basicpy import BaSiC

if TYPE_CHECKING:
    import tensorstore as ts


def _flatfield_sample_indices(
    positions: tuple[int, ...],
    n_scan_planes: int,
    *,
    max_tiles_per_level: int = 32,
    planes_per_position: int = 10,
) -> list[tuple[int, list[int]]]:
    """Choose distributed tiles and reproducible scan samples within one depth.

    Parameters
    ----------
    positions : tuple of int
        Eligible acquisition position indices for this channel and depth.
    n_scan_planes : int
        Acquired scan planes per tile.
    max_tiles_per_level : int
        Maximum spatially distributed tiles sampled per depth.
    planes_per_position : int
        Maximum distinct scan planes sampled from each selected tile.

    Returns
    -------
    list of tuple[int, list[int]]
        Acquisition positions paired with sorted, unique scan-plane indices.
    """
    if min(n_scan_planes, max_tiles_per_level, planes_per_position) < 1:
        raise ValueError("Illumination sampling counts must be positive")
    if len(positions) > max_tiles_per_level:
        relative = np.linspace(
            0, len(positions) - 1, max_tiles_per_level, dtype=np.int64
        )
        positions = tuple(positions[int(index)] for index in relative)
    rng = np.random.default_rng(0)
    return [
        (
            int(position),
            sorted(
                int(index)
                for index in rng.choice(
                    n_scan_planes,
                    size=min(n_scan_planes, planes_per_position),
                    replace=False,
                )
            ),
        )
        for position in positions
    ]


def _resize_image_stack(
    images: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Resize an image stack using BaSiCPy's interpolation convention.

    Parameters
    ----------
    images : np.ndarray
        Calibrated float32 image or stack with camera-X as its last axis.
    output_shape : tuple[int, int]
        Requested detector YX working dimensions for the fit.

    Returns
    -------
    np.ndarray
        Float32 image stack resized with the BaSiCPy interpolation convention.
    """
    resized = F.interpolate(
        torch.from_numpy(images[:, np.newaxis, :, :]),
        size=output_shape,
        mode="bilinear",
        align_corners=True,
        antialias=True,
    )
    return resized[:, 0].cpu().numpy()


def estimate_illuminations(
    datastore: ts.TensorStore | np.ndarray,
    camera_offset: float,
    camera_conversion: float,
    stage_positions_zxy: np.ndarray | None = None,
    *,
    detector_gain_x: np.ndarray | None = None,
    signal_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fit shared per-channel illumination using samples from every stage depth.

    Parameters
    ----------
    datastore : tensorstore.TensorStore or numpy.ndarray
        Nonempty TPCZYX acquisition, in raw camera ADU. Only time zero is fitted.
    camera_offset : float
        Recorded camera baseline in ADU, subtracted before calibration.
    camera_conversion : float
        Recorded calibrated intensity per ADU; must be finite and positive.
    stage_positions_zxy : numpy.ndarray or None
        Acquisition-ordered stage Z, X and Y coordinates in micrometers.
        Repeated XY visits define depth groups. None uses one group.
    detector_gain_x : numpy.ndarray or None
        Recorded detector-X response divided out before illumination fitting.
    signal_mask : numpy.ndarray or None
        TPC decisions: 1 selects a tile, zero excludes it and -1 is
        unchecked and excluded. Boolean masks are also accepted.

    Returns
    -------
    numpy.ndarray
        Dimensionless fields in CYX order without stage positions, or 1CYX order
        when stage positions are supplied. One BaSiC fit per channel uses all
        selected depths and is applied at every depth.

    Raises
    ------
    ValueError
        Calibration or dimensions are invalid, fewer than two nonempty images
        constrain a selected fit, or the fitted response is unusable.
    """
    if len(datastore.shape) != 6 or min(datastore.shape) < 1:
        raise ValueError(
            "Illumination estimation requires a nonempty TPCZYX acquisition"
        )
    if (
        not np.isfinite(camera_offset)
        or not np.isfinite(camera_conversion)
        or camera_conversion <= 0
    ):
        raise ValueError(
            "Camera offset must be finite and conversion finite and positive"
        )
    n_positions = int(datastore.shape[1])
    if stage_positions_zxy is None:
        depth_indices = np.zeros(n_positions, dtype=int)
    else:
        positions = np.asarray(stage_positions_zxy, dtype=float)
        if positions.shape != (n_positions, 3) or not np.isfinite(positions).all():
            raise ValueError(
                "Stage positions must contain one finite ZXY coordinate per tile"
            )
        depth_indices = stage_z_level_indices(positions)
    position_groups = tuple(
        tuple(int(value) for value in np.flatnonzero(depth_indices == level))
        for level in range(int(depth_indices.max()) + 1)
    )
    # Retain one shared camera correction per channel, irrespective of depth.
    flatfields = np.zeros(
        (
            1,
            datastore.shape[2],
            datastore.shape[-2],
            datastore.shape[-1],
        ),
        dtype=np.float32,
    )
    camera_shape = (datastore.shape[-2], datastore.shape[-1])
    working_shape = tuple(max(1, int(size) // 2) for size in camera_shape)
    n_channels = int(datastore.shape[2])
    if signal_mask is not None:
        signal_mask = np.asarray(signal_mask)
        if (
            signal_mask.shape != tuple(datastore.shape[:3])
            or not np.isin(signal_mask, (-1, 0, 1)).all()
        ):
            raise ValueError(
                "Illumination signal mask must have TPC shape and contain -1, 0 or 1"
            )
        signal_mask = signal_mask > 0

    with tqdm(
        total=n_channels,
        desc="illumination fitting",
        unit="channel",
        leave=False,
    ) as progress:
        for chan_idx in range(n_channels):
            progress.set_postfix(channel=chan_idx)
            sample_indices = []
            for positions in position_groups:
                eligible_positions = (
                    positions
                    if signal_mask is None
                    else tuple(
                        position
                        for position in positions
                        if signal_mask[0, position, chan_idx]
                    )
                )
                sample_indices.extend(
                    _flatfield_sample_indices(eligible_positions, datastore.shape[-3])
                )
            if not sample_indices:
                flatfields[0, chan_idx] = 1.0
                progress.update()
                continue
            # A lone tile must retain its scan frames as independent observations.
            single_tile = len(sample_indices) == 1
            n_fit_images = (
                len(sample_indices[0][1]) if single_tile else len(sample_indices)
            )
            fit_images = np.empty((n_fit_images, *working_shape), dtype=np.float32)
            for image_index, (pos_idx, scan_indices) in enumerate(sample_indices):
                selection = datastore[0, pos_idx, chan_idx, scan_indices, :, :]
                raw = (
                    selection.read().result()
                    if hasattr(selection, "read")
                    else np.asarray(selection)
                )
                temp_images = camera_correct(
                    raw,
                    camera_offset,
                    camera_conversion,
                    detector_gain_x=detector_gain_x,
                )
                if not np.isfinite(temp_images).all():
                    raise ValueError(
                        "Illumination samples must contain finite calibrated intensities"
                    )
                resized = _resize_image_stack(temp_images, working_shape)
                if single_tile:
                    fit_images[:] = resized
                else:
                    fit_images[image_index] = np.median(resized, axis=0)

            fit_images = fit_images[np.any(fit_images > 0, axis=(1, 2))]
            if not len(fit_images):
                flatfields[0, chan_idx] = 1.0
                progress.update()
                continue
            if len(fit_images) < 2:
                raise ValueError(
                    "Flatfield estimation requires at least two nonempty sampled images"
                )
            basic = BaSiC(
                get_darkfield=False,
                working_size=list(working_shape),
                smoothness_flatfield=2.0,
            )
            with redirect_stdout(io.StringIO()):
                basic.fit(fit_images)
            fitted = np.asarray(basic.flatfield, dtype=np.float32)
            if fitted.shape != working_shape or not np.all(
                np.isfinite(fitted) & (fitted > 0)
            ):
                raise ValueError("BaSiC returned an invalid illumination field")
            flatfields[0, chan_idx] = _resize_image_stack(
                fitted[np.newaxis, :, :], camera_shape
            )[0]
            del basic, fitted, fit_images
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            progress.update()
    if not np.all(np.isfinite(flatfields) & (flatfields > 0)):
        raise ValueError(
            "Illumination could not be estimated over the complete detector"
        )
    return flatfields[0] if stage_positions_zxy is None else flatfields
