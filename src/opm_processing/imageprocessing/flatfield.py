"""Estimate illumination fields for OPM image correction."""

import builtins
import gc

import numpy as np

from opm_processing.cuda import preload_cuda_libraries


preload_cuda_libraries()

import torch  # noqa: E402
from basicpy import BaSiC  # noqa: E402


def no_op(*args, **kwargs):
    """Suppress output when temporarily substituted for :func:`print`.

    Parameters
    ----------
    args: Any
        positional arguments
    kwargs: Any
        keyword arguments
    """
    pass


def estimate_illuminations(
    datastore,
    camera_offset,
    camera_conversion,
    *,
    max_images: int = 5000,
    image_batches: int = 25,
    position_samples: int = 15,
    position_margin: int = 5,
    model_downsample: int = 4,
    rng_seed: int | None = None,
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
    max_images
        Maximum number of scan images sampled per position.
    image_batches
        Number of batches averaged before illumination fitting.
    position_samples
        Maximum number of positions sampled.
    position_margin
        Additional central positions eligible for random sampling.
    model_downsample
        Spatial downsampling used to initialize the BaSiC model.
    rng_seed
        Optional random seed for reproducible sampling.

    Returns
    -------
    numpy.ndarray
        Per-channel illumination fields in CYX order.
    """
    if min(max_images, image_batches, position_samples, model_downsample) < 1:
        raise ValueError("flatfield sampling values must be positive")
    if position_margin < 0:
        raise ValueError("position_margin must be nonnegative")
    rng = np.random.default_rng(rng_seed)
    # flatfields shape: c, y, x
    flatfields = np.zeros(
        (datastore.shape[2], datastore.shape[-2], datastore.shape[-1]), dtype=np.float32
    )

    # Define the number of z-images for flat field calculation
    if datastore.shape[-3] > max_images:
        n_rand_images = max_images
    elif datastore.shape[-3] == 1:
        n_rand_images = 1
    else:
        n_rand_images = datastore.shape[-3]
    n_image_batches = min(image_batches, n_rand_images)
    n_rand_images -= n_rand_images % n_image_batches
    n_images_to_max = n_rand_images // n_image_batches

    # Define the number of position samples for flat field calculation
    n_pos_samples = min(position_samples, datastore.shape[1])
    if datastore.shape[1] > n_pos_samples + position_margin:
        selection_width = n_pos_samples + position_margin
        selection_start = max(0, datastore.shape[1] // 2 - selection_width // 2)
        selection_stop = min(datastore.shape[1], selection_start + selection_width)
        flatfield_pos_iterator = list(
            rng.choice(
                range(selection_start, selection_stop),
                size=n_pos_samples,
                replace=False,
            )
        )
    else:
        n_pos_samples = datastore.shape[1]
        flatfield_pos_iterator = range(datastore.shape[1])

    for chan_idx in range(datastore.shape[2]):
        images = []
        for pos_idx in flatfield_pos_iterator:
            sample_indices = list(
                rng.choice(datastore.shape[-3], size=n_rand_images, replace=False)
            )
            try:
                temp_images = (
                    (
                        (
                            np.squeeze(
                                datastore[0, pos_idx, chan_idx, sample_indices, :]
                                .read()
                                .result()
                            ).astype(np.float32)
                            - camera_offset
                        )
                        * camera_conversion
                    )
                    .clip(0, 2**16 - 1)
                    .astype(np.uint16)
                )
            except Exception:
                temp_images = (
                    (
                        np.squeeze(
                            np.asarray(
                                datastore[0, pos_idx, chan_idx, sample_indices, :],
                                dtype=np.float32,
                            )
                            - camera_offset
                        )
                        * camera_conversion
                    )
                    .clip(0, 2**16 - 1)
                    .astype(np.uint16)
                )
            temp_images = temp_images.reshape(
                n_image_batches,
                n_images_to_max,
                temp_images.shape[-2],
                temp_images.shape[-1],
            )
            temp_images = np.squeeze(np.mean(temp_images, axis=1))
            images.append(temp_images)
        images = np.asarray(images, dtype=np.float32)
        images = images.reshape(
            n_pos_samples * n_image_batches, images.shape[-2], images.shape[-1]
        )
        original_print = builtins.print
        builtins.print = no_op
        try:
            basic = BaSiC(
                get_darkfield=False,
                darkfield=np.zeros(
                    (
                        temp_images.shape[-2] // model_downsample,
                        temp_images.shape[-1] // model_downsample,
                    ),
                    dtype=np.float64,
                ),
                flatfield=np.zeros(
                    (
                        temp_images.shape[-2] // model_downsample,
                        temp_images.shape[-1] // model_downsample,
                    ),
                    dtype=np.float64,
                ),
            )
            basic.autotune(images)
            basic.fit(images)
        finally:
            builtins.print = original_print
        flatfields[chan_idx, :] = np.squeeze(basic.flatfield).astype(np.float32)

        del basic, images, temp_images

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return flatfields
