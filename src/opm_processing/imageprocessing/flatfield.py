"""Estimate illumination fields for OPM image correction."""

import gc
import io
from contextlib import redirect_stdout

import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from opm_processing.cuda import preload_cuda_libraries
from opm_processing.imageprocessing.camera import (
    correct_qi2lab_stage_scan_camera,
)
from opm_processing.imageprocessing.coordinates import stage_z_level_indices


preload_cuda_libraries()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from basicpy import BaSiC  # noqa: E402


def _flatfield_sample_indices(
    n_positions: int,
    n_scan_planes: int,
    *,
    planes_per_position: int = 10,
) -> list[tuple[int, list[int]]]:
    """Select reproducible scan planes distributed across the acquisition.

    BaSiCPy needs independent images containing varied specimen content.  In a
    tiled stage scan, sampling only adjacent tiles can make specimen structure
    common to the fit and therefore indistinguishable from illumination.
    """
    samples_per_position = min(n_scan_planes, planes_per_position)

    rng = np.random.default_rng(0)
    return [
        (
            int(position),
            sorted(
                int(index)
                for index in rng.choice(
                    n_scan_planes,
                    size=samples_per_position,
                    replace=False,
                )
            ),
        )
        for position in range(n_positions)
    ]


def _stage_z_groups(
    n_positions: int,
    stage_positions_zxy: np.ndarray | None,
) -> tuple[tuple[float, ...], tuple[tuple[int, ...], ...]]:
    """Group tiles by repeated acquisition depth at the same stage XY."""
    if stage_positions_zxy is None:
        return (0.0,), (tuple(range(n_positions)),)
    positions = np.asarray(stage_positions_zxy, dtype=float)
    if positions.shape != (n_positions, 3):
        raise ValueError("stage positions must have shape (positions, 3)")
    indices = stage_z_level_indices(positions)
    groups = tuple(
        tuple(int(value) for value in np.flatnonzero(indices == level))
        for level in range(int(indices.max()) + 1)
    )
    return tuple(float(level) for level in range(len(groups))), groups


def _flatfield_tile_indices(
    positions: tuple[int, ...],
    *,
    max_tiles_per_level: int = 32,
) -> tuple[int, ...]:
    """Select evenly distributed tiles independently within one depth level."""
    if max_tiles_per_level < 1:
        raise ValueError("max_tiles_per_level must be positive")
    if len(positions) <= max_tiles_per_level:
        return positions
    relative = np.unique(
        np.linspace(0, len(positions) - 1, max_tiles_per_level, dtype=np.int64)
    )
    return tuple(positions[int(index)] for index in relative)


def _camera_corrected_images(
    selection,
    *,
    camera_offset: float,
    camera_conversion: float,
    apply_stage_scan_gain: bool,
) -> np.ndarray:
    """Load uint16 raw images and apply detector calibration in float32."""
    images = _read_images(selection)
    images -= np.float32(camera_offset)
    images *= np.float32(camera_conversion)
    np.maximum(images, np.float32(0), out=images)
    if apply_stage_scan_gain:
        images = correct_qi2lab_stage_scan_camera(images, copy=False)
    return images


def _read_images(selection) -> np.ndarray:
    """Read uint16 raw acquisition data and convert it once to float32."""
    try:
        images = selection.read().result()
    except (AttributeError, TypeError):
        images = np.asarray(selection)
    images = np.asarray(images)
    if images.dtype != np.dtype(np.uint16):
        raise TypeError(
            f"Illumination estimation requires uint16 raw data; received {images.dtype}"
        )
    images = images.astype(np.float32)
    if images.ndim == 2:
        images = images[np.newaxis, ...]
    return images


def _resize_image_stack(
    images: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Resize an image stack using BaSiCPy's interpolation convention."""
    resized = F.interpolate(
        torch.from_numpy(images[:, np.newaxis, :, :]),
        size=output_shape,
        mode="bilinear",
        align_corners=True,
        antialias=True,
    )
    return resized[:, 0].cpu().numpy()


def _flatfield_working_shape(image_shape: tuple[int, int]) -> tuple[int, int]:
    """Use a rectangular working field downsampled twofold on each axis."""
    return tuple(max(1, int(size) // 2) for size in image_shape)


def _separable_residual_calibration(
    flatfield: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Calibrate residual detector-coordinate gain on independent tile summaries."""
    corrected = observations / flatfield[np.newaxis, :, :]
    residual = np.median(corrected, axis=0)
    residual /= np.median(residual)

    sigma = max(4.0, observations.shape[-1] / 120.0)
    x_profile = np.median(residual, axis=0)
    x_profile = gaussian_filter1d(
        x_profile,
        sigma=sigma,
        mode="nearest",
    )
    x_profile /= np.mean(x_profile)

    y_profile = np.median(
        residual / x_profile[np.newaxis, :],
        axis=1,
    )
    y_profile = gaussian_filter1d(
        y_profile,
        sigma=sigma,
        mode="nearest",
    )
    y_profile /= np.mean(y_profile)

    calibrated = flatfield * y_profile[:, np.newaxis] * x_profile[np.newaxis, :]
    calibrated /= np.mean(calibrated)
    return calibrated.astype(np.float32, copy=False)


def estimate_illuminations(
    datastore,
    camera_offset,
    camera_conversion,
    stage_positions_zxy=None,
    *,
    apply_stage_scan_gain: bool = False,
    signal_mask: np.ndarray | None = None,
):
    """Estimate per-channel illumination fields from sampled images.

    Parameters
    ----------
    datastore
        Array-like TPCZYX acquisition datastore.
    camera_offset
        Camera offset subtracted from each sampled image.
    camera_conversion
        Multiplicative conversion from camera units to intensity units.
    stage_positions_zxy
        Stage ZYX coordinates used to group repeated XY visits by depth.
    apply_stage_scan_gain
        Apply the fixed detector-X gain correction before fitting.
    signal_mask
        Optional boolean TPC mask computed before processing. At time zero,
        false entries are excluded before tile subsampling for each channel.

    Returns
    -------
    numpy.ndarray
        Illumination fields in CYX order for a global estimate or ZCYX order
        when stage positions are supplied.
    """
    stage_z_levels, position_groups = _stage_z_groups(
        int(datastore.shape[1]),
        stage_positions_zxy,
    )
    # flatfields shape: physical stage z, c, y, x
    flatfields = np.zeros(
        (
            len(stage_z_levels),
            datastore.shape[2],
            datastore.shape[-2],
            datastore.shape[-1],
        ),
        dtype=np.float32,
    )
    camera_shape = (datastore.shape[-2], datastore.shape[-1])
    working_shape = _flatfield_working_shape(camera_shape)
    n_channels = int(datastore.shape[2])
    if signal_mask is not None:
        signal_mask = np.asarray(signal_mask, dtype=bool)
        expected_mask_shape = tuple(int(value) for value in datastore.shape[:3])
        if signal_mask.shape != expected_mask_shape:
            raise ValueError(
                f"signal_mask must have TPC shape {expected_mask_shape}; "
                f"received {signal_mask.shape}"
            )

    progress = tqdm(
        total=len(position_groups) * n_channels,
        desc="illumination fitting",
        unit="channel",
        leave=False,
    )
    for stage_level_index, positions in enumerate(position_groups):
        for chan_idx in range(n_channels):
            progress.set_postfix(
                stage_level=stage_level_index,
                channel=chan_idx,
            )
            eligible_positions = (
                positions
                if signal_mask is None
                else tuple(
                    position
                    for position in positions
                    if signal_mask[0, position, chan_idx]
                )
            )
            selected_positions = _flatfield_tile_indices(eligible_positions)
            if not selected_positions:
                flatfields[stage_level_index, chan_idx] = 1.0
                progress.update()
                continue
            samples_per_position = _flatfield_sample_indices(
                len(selected_positions),
                datastore.shape[-3],
            )
            sample_indices = [
                (selected_positions[group_position], scan_indices)
                for group_position, scan_indices in samples_per_position
            ]
            n_fit_images = len(sample_indices)
            fit_images = np.empty(
                (n_fit_images, *working_shape),
                dtype=np.float32,
            )
            calibration_images = np.empty_like(fit_images)
            for image_index, (pos_idx, scan_indices) in enumerate(sample_indices):
                temp_images = _camera_corrected_images(
                    datastore[0, pos_idx, chan_idx, scan_indices, :, :],
                    camera_offset=camera_offset,
                    camera_conversion=camera_conversion,
                    apply_stage_scan_gain=apply_stage_scan_gain,
                )
                resized = _resize_image_stack(temp_images, working_shape)
                fit_images[image_index] = np.median(resized, axis=0)
                calibration_images[image_index] = np.percentile(
                    resized,
                    75,
                    axis=0,
                )

            basic = BaSiC(
                get_darkfield=False,
                sort_intensity=True,
                working_size=list(working_shape),
            )

            with redirect_stdout(io.StringIO()):
                basic.autotune(fit_images)
                basic.smoothness_flatfield = 2.0
                basic.fit(fit_images)
            fitted_flatfield = _separable_residual_calibration(
                np.asarray(basic.flatfield, dtype=np.float32),
                calibration_images,
            )
            flatfields[stage_level_index, chan_idx, :] = _resize_image_stack(
                fitted_flatfield[np.newaxis, :, :],
                camera_shape,
            )[0]

            del basic, fitted_flatfield, fit_images, calibration_images

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            progress.update()

    progress.close()

    return flatfields[0] if stage_positions_zxy is None else flatfields
