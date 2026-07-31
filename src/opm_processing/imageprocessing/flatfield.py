"""Estimate illumination fields for OPM image correction."""

import builtins
import gc

import numpy as np
from scipy.ndimage import gaussian_filter1d

from opm_processing.cuda import preload_cuda_libraries


preload_cuda_libraries()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from basicpy import BaSiC  # noqa: E402


def no_op(*args, **kwargs):
    """Suppress output when temporarily substituted for :func:`print`.

    Parameters
    ----------
    args: Any
        positional arguments
    kwargs: Any
        keyword arguments

    Returns
    -------
    None
        No value is returned.
    """
    pass


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


def _read_images(selection) -> np.ndarray:
    """Read a TensorStore-like selection as an image stack."""
    try:
        images = selection.read().result()
    except (AttributeError, TypeError):
        images = np.asarray(selection)
    images = np.asarray(images, dtype=np.float32)
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

    calibrated = (
        flatfield
        * y_profile[:, np.newaxis]
        * x_profile[np.newaxis, :]
    )
    calibrated /= np.mean(calibrated)
    return calibrated.astype(np.float32, copy=False)


def estimate_illuminations(
    datastore,
    camera_offset,
    camera_conversion,
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

    Returns
    -------
    numpy.ndarray
        Per-channel illumination fields in CYX order.
    """
    # flatfields shape: c, y, x
    flatfields = np.zeros(
        (datastore.shape[2], datastore.shape[-2], datastore.shape[-1]), dtype=np.float32
    )
    sample_indices = _flatfield_sample_indices(
        datastore.shape[1],
        datastore.shape[-3],
    )
    n_fit_images = len(sample_indices)
    camera_shape = (datastore.shape[-2], datastore.shape[-1])
    working_shape = _flatfield_working_shape(camera_shape)

    n_channels = int(datastore.shape[2])
    fit_images = np.empty(
        (n_channels, n_fit_images, *working_shape),
        dtype=np.float32,
    )
    calibration_images = np.empty_like(fit_images)
    for image_index, (pos_idx, scan_indices) in enumerate(sample_indices):
        temp_images = _read_images(
            datastore[0, pos_idx, :, scan_indices, :, :]
        )
        if temp_images.shape[:2] == (len(scan_indices), n_channels):
            temp_images = np.swapaxes(temp_images, 0, 1)
        temp_images -= camera_offset
        temp_images *= camera_conversion
        np.clip(temp_images, 0, 2**16 - 1, out=temp_images)
        resized = _resize_image_stack(
            temp_images.reshape(-1, *camera_shape),
            working_shape,
        ).reshape(n_channels, len(scan_indices), *working_shape)
        fit_images[:, image_index] = np.median(resized, axis=1)
        calibration_images[:, image_index] = np.percentile(
            resized,
            75,
            axis=1,
        )

    del temp_images, resized

    for chan_idx in range(n_channels):
        basic = BaSiC(
            get_darkfield=False,
            sort_intensity=True,
            working_size=list(working_shape),
        )

        original_print = builtins.print
        builtins.print = no_op
        try:
            basic.autotune(fit_images[chan_idx])
            basic.smoothness_flatfield = 2.0
            basic.fit(fit_images[chan_idx])
        finally:
            builtins.print = original_print
        fitted_flatfield = _separable_residual_calibration(
            np.asarray(basic.flatfield, dtype=np.float32),
            calibration_images[chan_idx],
        )
        flatfields[chan_idx, :] = _resize_image_stack(
            fitted_flatfield[np.newaxis, :, :],
            camera_shape,
        )[0]

        del basic, fitted_flatfield

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    del fit_images, calibration_images

    return flatfields
