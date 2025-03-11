"""
qi2lab OPM data handling tools.

This module provides tools and utilities specifically for handling
oblique plane microscopy (OPM) data.

History:
---------
- **2025/03**: Updated for new qi2lab OPM processing pipeline.
- **2024/12**: Refactored repo structure.
- **2024/07**: Initial commit.
"""

from opm_processing.imageprocessing.utils import downsample_axis
from tqdm import tqdm
import numpy as np
from numpy.typing import ArrayLike
from typing import Sequence, Tuple
from numba import njit, prange
import gc

@njit
def deskew_shape_estimator(
    input_shape: Sequence[int],
    theta: float = 30.0,
    distance: float = 0.4,
    pixel_size: float = 0.115,
    divisble_by: int = 4
):
    """Generate shape of orthogonal interpolation output array.
    
    This function automatically pads the YX dimensions to be 
    an integer divisble by `divisble_by`.

    Parameters
    ----------
    input_shape: Sequence[int]
        shape of oblique array
    theta: float
        angle relative to coverslip
    distance: float
        step between image planes along coverslip
    pixel_size: float
        in-plane camera pixel size in OPM coordinates

    Returns
    -------
    output_shape: Sequence[int]
        shape of deskewed array
    """

    # change step size from physical space (nm) to camera space (pixels)
    pixel_step = distance / pixel_size  # (pixels)

    # calculate the number of pixels scanned during stage scan
    scan_end = input_shape[0] * pixel_step  # (pixels)

    # calculate properties for final image
    final_ny = np.int64(
        np.ceil(scan_end + input_shape[1] * np.cos(theta * np.pi / 180))
    )  # (pixels)
    final_nz = np.int64(
        np.ceil(input_shape[1] * np.sin(theta * np.pi / 180))
    )  # (pixels)
    final_nx = np.int64(input_shape[2])
    
    # pad YX array size to make sure it is divisble by 4
    pad_y = (divisble_by - (final_ny % divisble_by)) % divisble_by  
    pad_x = (divisble_by - (final_nx % divisble_by)) % divisble_by
    padded_final_ny = final_ny + pad_y 
    padded_final_nx = final_nx + pad_x

    return [final_nz, padded_final_ny, padded_final_nx], pad_y, pad_x


@njit(parallel=True)
def deskew(
    data: ArrayLike,
    theta: float = 30.0,
    distance: float = 0.4,
    pixel_size: float = 0.115,
    flip_scan = False,
    reverse_deskewed_z = False,
    divisible_by: int = 4,
    downsample_factor: int = 2
):
    """Numba accelerated orthogonal interpolation for oblique data.
    
    This function automatically pads the YX array dimensions to be 
    integer divisble by `divisble_by`.

    Parameters
    ----------
    data: ArrayLike
        image stack of uniformly spaced OPM planes
    theta: float, default = 30
        angle relative to coverslip in degrees
    distance: float, default = 0.4
        step between image planes along coverslip in microns
    pixel_size: float, default = 0.115
        in-plane camera pixel size in OPM coordinates in microns
    flip_scan: bool, default = False
        flip direction of scan stack w.r.t deskew direction
    reverse_deskewed_z: bool, default = False
        flip output z direction to match camera <-> stage orientation 
    divisible_by: int, default = 4
        amount to ensure data is divisible by for chunked storage
    downsample_factor: int, default = 2
        amount to downsample the output Z axis by

    Returns
    -------
    output: ArrayLike
        image stack of deskewed OPM planes on uniform grid
    """
    
    if flip_scan:
        data = np.flipud(data)

    num_images, ny, nx = data.shape
    pixel_step = distance / pixel_size
    scan_end = num_images * pixel_step

    # Convert angles once
    theta_rad = np.radians(theta)
    tantheta = np.tan(theta_rad)
    sintheta = np.sin(theta_rad)
    costheta = np.cos(theta_rad)

    # Compute final image size
    final_ny = np.int64(np.ceil(scan_end + ny * costheta))
    final_nz = np.int64(np.ceil(ny * sintheta))
    final_nz_downsampled = max(1, final_nz // downsample_factor)
    final_nx = np.int64(nx)

    # Pad dimensions to be divisible by `divisible_by`
    pad_y = (divisible_by - (final_ny % divisible_by)) % divisible_by
    pad_x = (divisible_by - (final_nx % divisible_by)) % divisible_by
    padded_final_ny = final_ny + pad_y
    padded_final_nx = final_nx + pad_x

    # Allocate output array
    output = np.zeros((final_nz_downsampled, padded_final_ny, padded_final_nx), dtype=np.float32)

    # Precompute division to avoid redundant division in the loop
    inv_pixel_step = 1 / pixel_step

    # Perform deskewing with integrated downsampling
    for z_ds in prange(final_nz_downsampled):
        z_start = z_ds * downsample_factor
        z_end = min(z_start + downsample_factor, final_nz)

        temp_buffer = np.zeros((padded_final_ny, padded_final_nx), dtype=np.float32)

        for z in range(z_start, z_end):
            for y in prange(final_ny):  
                virtual_plane = y - z / tantheta
                plane_before = int(np.floor(virtual_plane * inv_pixel_step))
                plane_after = plane_before + 1

                # Strict boundary check to prevent invalid memory accesses
                if plane_before < 0 or plane_after >= num_images:
                    continue  # Skip invalid interpolation points

                za = z / sintheta
                virtual_pos_before = za + (virtual_plane - plane_before * pixel_step) * costheta
                virtual_pos_after = za - (pixel_step - (virtual_plane - plane_before * pixel_step)) * costheta

                pos_before = int(np.floor(virtual_pos_before))
                pos_after = int(np.floor(virtual_pos_after))

                # Strict position index check
                if pos_before < 0 or pos_after >= ny - 1:
                    continue  # Skip out-of-bounds pixels

                dz_before = virtual_pos_before - pos_before
                dz_after = virtual_pos_after - pos_after

                # Fetch pixel values safely, ensuring they are within valid range
                pixel_1 = data[plane_after, pos_after + 1, :final_nx]
                pixel_2 = data[plane_after, pos_after, :final_nx]
                pixel_3 = data[plane_before, pos_before + 1, :final_nx]
                pixel_4 = data[plane_before, pos_before, :final_nx]

                # **Fix: If all surrounding pixels are zero, skip accumulation**
                if (
                    np.all(pixel_1 == 0) and np.all(pixel_2 == 0) and
                    np.all(pixel_3 == 0) and np.all(pixel_4 == 0)
                ):
                    continue  # Prevents division by zero and artifacts

                # Compute interpolated values
                new_values = (
                    dz_after * pixel_1
                    + (1 - dz_after) * pixel_2
                    + dz_before * pixel_3
                    + (1 - dz_before) * pixel_4
                ) * inv_pixel_step

                # Prevent small floating-point errors from accumulating
                new_values = np.clip(new_values, 0, 65534)  

                # Accumulate safely
                temp_buffer[y, :final_nx] = np.clip(temp_buffer[y, :final_nx] + new_values, 0, 65534)

        # Store the averaged downsampled z-slice
        output[z_ds] = np.clip(temp_buffer / downsample_factor, 0, 65534)  # Prevent overflow after division

    # Explicitly zero out padding before conversion
    if pad_y > 0:
        output[:, -pad_y:, :] = 0
    if pad_x > 0:
        output[:, :, -pad_x:] = 0

    # Convert to uint16 safely
    output_uint16 = output.astype(np.uint16)

    if reverse_deskewed_z:
        return np.flipud(output_uint16)
    else:
        return output_uint16


def lab2cam(
    x: int, y: int, z: int, theta: float = 30.0 * (np.pi / 180.0)
) -> Tuple[int, int, int]:
    """Convert xyz coordinates to camera coordinates sytem, x', y', and stage position.

    Parameters
    ----------
    x: int
        coverslip x coordinate
    y: int
        coverslip y coordinate
    z: int
        coverslip z coordinate
    theta: float
        OPM angle in radians


    Returns
    -------
    xp: int
        xp coordinate
    yp: int
        yp coordinate
    stage_pos: int
        distance of leading edge of camera frame from the y-axis
    """

    xp = x
    stage_pos = y - z / np.tan(theta)
    yp = z / np.sin(theta)
    return xp, yp, stage_pos


def chunk_indices(length: int, chunk_size: int) -> Sequence[int]:
    """Calculate indices for evenly distributed chunks.

    Parameters
    ----------
    length: int
        axis array length
    chunk_size: int
        size of chunks

    Returns
    -------
    indices: Sequence[int,...]
        chunk indices
    """

    indices = []
    for i in range(0, length - chunk_size, chunk_size):
        indices.append((i, i + chunk_size))
    if length % chunk_size != 0:
        indices.append((length - chunk_size, length))
    return indices


def chunked_orthogonal_deskew(
    oblique_image: ArrayLike,
    chunk_size: int = 15000,
    overlap_size: int = 550,
    scan_crop: int = 700,
    camera_bkd: int = 100,
    camera_cf: float = 0.24,
    camera_qe: float = 0.9,
    z_downsample_level=2
) -> ArrayLike:
    """Chunked orthogonal deskew of oblique data.

    Optionally performs deconvolution on each chunk.
    
    Parameters
    ----------
    oblique_image: ArrayLike
        oblique image stack
    psf_data: ArrayLike
        PSF data for deconvolution
    chunk_size: int
        size of chunks
    overlap_size: int
        overlap size
    scan_crop: int
        crop size
    camera_bkd: int
        camera background
    camera_cf: float
        camera conversion factor
    camera_qe: float
        camera quantum efficiency
    z_downsample_level: int
        z downsample level
    
    Returns
    -------
    deskewed_image: ArrayLike
        deskewed image stack
    """

    output_shape = deskew_shape_estimator(oblique_image.shape)
    output_shape[0] = output_shape[0] // z_downsample_level
    output_shape[1] = output_shape[1] - scan_crop
    deskewed_image = np.zeros(output_shape, dtype=np.uint16)

    if chunk_size < output_shape[1]:
        idxs = chunk_indices(output_shape[1], chunk_size)
    else:
        idxs = [(0, output_shape[1])]
        overlap_size = 0

    for idx in tqdm(idxs):
        if idx[0] > 0:
            tile_px_start = idx[0] - overlap_size
            crop_start = True
        else:
            tile_px_start = idx[0]
            crop_start = False

        if idx[1] < output_shape[1]:
            tile_px_end = idx[1] + overlap_size
            crop_end = True
        else:
            if overlap_size == 0:
                tile_px_end = idx[1] + scan_crop
                crop_end = False
            else:
                tile_px_end = idx[1]
                crop_end = False

        xp, yp, sp_start = lab2cam(
            oblique_image.shape[2], tile_px_start, 0, 30.0 * np.pi / 180.0
        )

        xp, yp, sp_stop = lab2cam(
            oblique_image.shape[2], tile_px_end, 0, 30.0 * np.pi / 180.0
        )
        scan_px_start = np.maximum(0, np.int64(np.ceil(sp_start * (0.115 / 0.4))))
        scan_px_stop = np.minimum(
            oblique_image.shape[0], np.int64(np.ceil(sp_stop * (0.115 / 0.4)))
        )

        raw_data = np.array(oblique_image[scan_px_start:scan_px_stop, :]).astype(
            np.float32
        )
        raw_data = raw_data - camera_bkd
        raw_data[raw_data < 0.0] = 0.0
        raw_data = ((raw_data * camera_cf) / camera_qe).astype(np.uint16)
        temp_deskew = deskew(raw_data).astype(np.uint16)

        if crop_start and crop_end:
            crop_deskew = temp_deskew[:, overlap_size:-overlap_size, :]
        elif crop_start:
            crop_deskew = temp_deskew[:, overlap_size:-1, :]
        elif crop_end:
            crop_deskew = temp_deskew[:, 0:-overlap_size, :]
        else:
            crop_deskew = temp_deskew[:, 0:-scan_crop, :]

        if crop_deskew.shape[1] > (chunk_size):
            diff = crop_deskew.shape[1] - (chunk_size)
            crop_deskew = crop_deskew[:, :-diff, :]
        elif crop_deskew.shape[1] < (chunk_size):
            diff = (chunk_size) - crop_deskew.shape[1]

            if crop_start and crop_end:
                crop_deskew = temp_deskew[:, overlap_size : -overlap_size + diff, :]
            elif crop_start:
                crop_deskew = temp_deskew[:, overlap_size - diff : -1, :]

        if z_downsample_level > 1:
            deskewed_image[:, idx[0] : idx[1], :] = downsample_axis(
                image=crop_deskew, level=z_downsample_level, axis=0
            )
        else:
            deskewed_image[:, idx[0] : idx[1], :] = crop_deskew

    del temp_deskew, oblique_image
    gc.collect()

    return deskewed_image