"""Estimate illumination fields for OPM image correction."""

import builtins
import gc

import numpy as np

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
    n_fit_images = sum(len(indices) for _, indices in sample_indices)
    camera_shape = (datastore.shape[-2], datastore.shape[-1])
    working_shape = _flatfield_working_shape(camera_shape)

    for chan_idx in range(datastore.shape[2]):
        basic = BaSiC(
            get_darkfield=False,
            working_size=list(working_shape),
        )
        images = np.empty(
            (n_fit_images, *working_shape),
            dtype=np.float32,
        )
        image_index = 0
        for pos_idx, scan_indices in sample_indices:
            temp_images = _read_images(
                datastore[0, pos_idx, chan_idx, scan_indices, :]
            )
            temp_images -= camera_offset
            temp_images *= camera_conversion
            np.clip(temp_images, 0, 2**16 - 1, out=temp_images)
            temp_images = _resize_image_stack(temp_images, working_shape)
            next_index = image_index + len(temp_images)
            images[image_index:next_index] = temp_images
            image_index = next_index

        original_print = builtins.print
        builtins.print = no_op
        try:
            basic.autotune(images)
            basic.fit(images)
        finally:
            builtins.print = original_print
        fitted_flatfield = np.asarray(basic.flatfield, dtype=np.float32)
        flatfields[chan_idx, :] = _resize_image_stack(
            fitted_flatfield[np.newaxis, :, :],
            camera_shape,
        )[0]

        del basic, images, temp_images

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return flatfields
