"""
OPM-specific data handling tools.

This module provides tools and utilities specifically for handling
oblique plane microscopy (OPM) data.

History:
---------
- **2024/12**: Refactored repo structure.
- **2024/07**: Initial commit.
"""

from opm_processing.imageprocessing.utils import (
    replace_hot_pixels,
    downsample_axis
)
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

    return [final_nz, padded_final_ny, padded_final_nx]


@njit(parallel=True)
def deskew(
    data: ArrayLike,
    theta: float = 30.0,
    distance: float = 0.4,
    pixel_size: float = 0.115,
    flip_scan = False,
    reverse_deskewed_z = False,
    divisble_by: int = 4
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

    Returns
    -------
    output: ArrayLike
        image stack of deskewed OPM planes on uniform grid
    """
    
    if flip_scan:
        data = np.flipud(data)

    # unwrap parameters
    [num_images, ny, nx] = data.shape  # (pixels)

    # change step size from physical space (nm) to camera space (pixels)
    pixel_step = distance / pixel_size  # (pixels)

    # calculate the number of pixels scanned during stage scan
    scan_end = num_images * pixel_step  # (pixels)

    # calculate properties for final image
    final_ny = np.int64(
        np.ceil(scan_end + ny * np.cos(theta * np.pi / 180))
    )  # (pixels)
    final_nz = np.int64(np.ceil(ny * np.sin(theta * np.pi / 180)))  # (pixels)
    final_nx = np.int64(nx)  # (pixels)
    
    # pad YX array to make sure it is divisble by 4
    pad_y = (divisble_by - (final_ny % divisble_by)) % divisble_by
    pad_x = (divisble_by- (final_nx % divisble_by)) % divisble_by
    padded_final_ny = final_ny + pad_y 
    padded_final_nx = final_nx + pad_x


    # create final image
    output = np.zeros(
        (final_nz, padded_final_ny, padded_final_nx), dtype=np.float32
    ) 

    # precalculate trig functions for scan angle
    tantheta = np.float32(np.tan(theta * np.pi / 180))  # (float32)
    sintheta = np.float32(np.sin(theta * np.pi / 180))  # (float32)
    costheta = np.float32(np.cos(theta * np.pi / 180))  # (float32)

    # perform orthogonal interpolation

    # loop through output z planes
    # defined as parallel loop in numba
    for z in prange(0, final_nz):
        # calculate range of output y pixels to populate
        y_range_min = np.minimum(0, np.int64(np.floor(np.float32(z) / tantheta)))
        y_range_max = np.maximum(
            final_ny, np.int64(np.ceil(scan_end + np.float32(z) / tantheta + 1))
        )

        # loop through final y pixels
        # defined as parallel loop in numba
        for y in prange(y_range_min, y_range_max):
            # find the virtual tilted plane that intersects the interpolated plane
            virtual_plane = y - z / tantheta

            # find raw data planes that surround the virtual plane
            plane_before = np.int64(np.floor(virtual_plane / pixel_step))
            plane_after = np.int64(plane_before + 1)

            # continue if raw data planes are within the data range
            if (plane_before >= 0) and (plane_after < num_images):
                # find distance of a point on the  interpolated plane to plane_before and plane_after
                l_before = virtual_plane - plane_before * pixel_step
                l_after = pixel_step - l_before

                # determine location of a point along the interpolated plane
                za = z / sintheta
                virtual_pos_before = za + l_before * costheta
                virtual_pos_after = za - l_after * costheta

                # determine nearest data points to interpoloated point in raw data
                pos_before = np.int64(np.floor(virtual_pos_before))
                pos_after = np.int64(np.floor(virtual_pos_after))

                # continue if within data bounds
                if (
                    (pos_before >= 0)
                    and (pos_after >= 0)
                    and (pos_before < ny - 1)
                    and (pos_after < ny - 1)
                ):
                    # determine points surrounding interpolated point on the virtual plane
                    dz_before = virtual_pos_before - pos_before
                    dz_after = virtual_pos_after - pos_after

                    # compute final image plane using orthogonal interpolation
                    output[z, y, :final_nx] = (
                        l_before * dz_after * data[plane_after, pos_after + 1, :]
                        + l_before * (1 - dz_after) * data[plane_after, pos_after, :]
                        + l_after * dz_before * data[plane_before, pos_before + 1, :]
                        + l_after * (1 - dz_before) * data[plane_before, pos_before, :]
                    ) / pixel_step
                    
    if reverse_deskewed_z:
        return np.flipud(output).astype(np.uint16)
    else:
        return output.astype(np.uint16)


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