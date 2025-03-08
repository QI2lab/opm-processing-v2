"""
Image processing functions for qi2lab 3D MERFISH.

This module includes various utilities for image processing, such as
downsampling, padding, and chunked GPU-based deconvolution.

History:
---------
- **2025/03**: Updated for new qi2lab OPM processing pipeline.
- **2024/12**: Refactored repo structure.
- **2024/07**: Added numba-accelerated downsampling, padding helper functions,
               and chunked GPU deconvolution.
"""

import numpy as np
import gc
from numpy.typing import NDArray
from numba import njit, prange

# GPU
CUPY_AVIALABLE = True
try:
    import cupy as cp  # type: ignore
except ImportError:
    xp = np
    CUPY_AVIALABLE = False
    from scipy import ndimage  # type: ignore
else:
    xp = cp
    from cupyx.scipy import ndimage  # type: ignore


def replace_hot_pixels(
    noise_map: NDArray, 
    data: NDArray, 
    threshold: float = 375.0
) -> NDArray:
    """Replace hot pixels with median values surrounding them.

    Parameters
    ----------
    noise_map: NDArray
        darkfield image collected at long exposure time to get hot pixels
    data: NDArray
        ND data [broadcast_dim,z,y,x]

    Returns
    -------
    data: NDArray
        hotpixel corrected data
    """

    data = xp.asarray(data, dtype=xp.float32)
    noise_map = xp.asarray(noise_map, dtype=xp.float32)

    # threshold darkfield_image to generate bad pixel matrix
    hot_pixels = xp.squeeze(xp.asarray(noise_map))
    hot_pixels[hot_pixels <= threshold] = 0
    hot_pixels[hot_pixels > threshold] = 1
    hot_pixels = hot_pixels.astype(xp.float32)
    inverted_hot_pixels = xp.ones_like(hot_pixels) - hot_pixels.copy()

    data = xp.asarray(data, dtype=xp.float32)
    for z_idx in prange(data.shape[0]):
        median = ndimage.median_filter(data[z_idx, :, :], size=3)
        data[z_idx, :] = inverted_hot_pixels * data[z_idx, :] + hot_pixels * median

    data[data < 0] = 0

    if CUPY_AVIALABLE:
        data = xp.asnumpy(data).astype(np.uint16)
        gc.collect()
        cp.clear_memo()
        cp._default_memory_pool.free_all_blocks()
    else:
        data = data.astype(np.uint16)

    return data

@njit(parallel=True)
def correct_shading(
    shading_image: NDArray, 
    data: NDArray,
    z_axis: int = 0
) -> NDArray:
    """Perform illumination shading correction.

    I_corrected = (I_raw - I_dark) / (I_bright - I_dark).
    Here, we assume I_bright is not normalized or background corrected.

    Parameters
    ----------
    shading_image: NDArray
        illumination shading correction
    data: NDArray
        ND data [broadcast_dim,z,y,x]
    z_axis: int, default = 0
        Axis along which to correct shading

    Returns
    -------
    data: NDArray
        shading corrected data
    """

    shading_image = np.squeeze(np.asarray(shading_image, dtype=np.float32))
    shading_image /= np.max(shading_image, axis=(0, 1))
    data = np.asarray(data, dtype=np.float32)
    data = np.moveaxis(data, z_axis, 0)
    
    for z_idx in prange(data.shape[z_axis]):
        data[z_idx] /= shading_image

    return np.moveaxis(data, 0, z_axis).astype(np.uint16)

def downsample_image_yx(image: NDArray, level: int = 2) -> NDArray:
    """Numba accelerated 2D plane downsampling

    Parameters
    ----------
    image: NDArray
        3D image to be downsampled
    level: int
        isotropic downsampling level

    Returns
    -------
    downsampled_image: NDArray
        downsampled 3D image
    """

    downsampled_image = downsample_axis(
        downsample_axis(image, level, 1),
        level, 2
    )

    return downsampled_image

def downsample_image_isotropic(image: NDArray, level: int = 2) -> NDArray:
    """Numba accelerated isotropic downsampling

    Parameters
    ----------
    image: NDArray
        3D image to be downsampled
    level: int
        isotropic downsampling level

    Returns
    -------
    downsampled_image: NDArray
        downsampled 3D image
    """

    downsampled_image = downsample_axis(
        downsample_axis(
            downsample_axis(image, level, 0), 
            level, 1), 
        level, 2
    )

    return downsampled_image

@njit(parallel=True)
def downsample_axis(
    image: NDArray, 
    level: int = 2, 
    axis: int = 0
) -> NDArray:
    """Numba accelerated downsampling for 3D images along a specified axis.

    Parameters
    ----------
    image: NDArray
        3D image to be downsampled.
    level: int
        Amount of downsampling.
    axis: int
        Axis along which to downsample (0, 1, or 2).

    Returns
    -------
    downsampled_image: NDArray
        3D downsampled image.

    """
    if axis == 0:
        new_length = image.shape[0] // level + (1 if image.shape[0] % level != 0 else 0)
        downsampled_image = np.zeros(
            (new_length, image.shape[1], image.shape[2]), dtype=image.dtype
        )

        for y in prange(image.shape[1]):
            for x in range(image.shape[2]):
                for z in range(new_length):
                    sum_value = 0.0
                    count = 0
                    for j in range(level):
                        original_index = z * level + j
                        if original_index < image.shape[0]:
                            sum_value += image[original_index, y, x]
                            count += 1
                    if count > 0:
                        downsampled_image[z, y, x] = sum_value / count

    elif axis == 1:
        new_length = image.shape[1] // level + (1 if image.shape[1] % level != 0 else 0)
        downsampled_image = np.zeros(
            (image.shape[0], new_length, image.shape[2]), dtype=image.dtype
        )

        for z in prange(image.shape[0]):
            for x in range(image.shape[2]):
                for y in range(new_length):
                    sum_value = 0.0
                    count = 0
                    for j in range(level):
                        original_index = y * level + j
                        if original_index < image.shape[1]:
                            sum_value += image[z, original_index, x]
                            count += 1
                    if count > 0:
                        downsampled_image[z, y, x] = sum_value / count

    elif axis == 2:
        new_length = image.shape[2] // level + (1 if image.shape[2] % level != 0 else 0)
        downsampled_image = np.zeros(
            (image.shape[0], image.shape[1], new_length), dtype=image.dtype
        )

        for z in prange(image.shape[0]):
            for y in range(image.shape[1]):
                for x in range(new_length):
                    sum_value = 0.0
                    count = 0
                    for j in range(level):
                        original_index = x * level + j
                        if original_index < image.shape[2]:
                            sum_value += image[z, y, original_index]
                            count += 1
                    if count > 0:
                        downsampled_image[z, y, x] = sum_value / count

    return downsampled_image

@njit(parallel=True)
def max_z_projection(data: NDArray, z_axis: int = 0) -> NDArray:
    """Numba accelerated max z projection of 3D image.

    Parameters
    ----------
    data: NDArray
        3D image to be projected.

    Returns
    -------
    max_projection: NDArray
        2D max projection of 3D image.
    """

    max_projection = np.max(data,axis=z_axis,keepdims=True)

    return max_projection