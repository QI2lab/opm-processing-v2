"""Estimate deskew dimensions and interpolate calibrated oblique scans to ZYX grids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike


@njit
def deskew_shape_estimator(
    input_shape: Sequence[int],
    theta: float = 30.0,
    distance: float = 0.4,
    pixel_size: float = 0.115,
    crop_after_deskew: bool = True,
    divisible_by: int = 4,
) -> tuple[list[int], int, int, int]:
    """Generate shape of orthogonal interpolation output array.

    This function automatically pads the YX dimensions to be
    an integer divisible by `divisible_by`.

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

    crop_after_deskew : bool
        Remove the lateral triangular deskew margins from the output shape.
    divisible_by : int
        Pad output YX dimensions to multiples of this integer.

    Returns
    -------
    output_shape : list[int]
        Deskewed ZYX dimensions, including YX padding.
    pad_y : int
        Added rows along laboratory Y.
    pad_x : int
        Added columns along laboratory X.
    crop_y : int
        Rows removed from each end of laboratory Y before padding.
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

    if crop_after_deskew:
        crop_y = int(np.ceil(input_shape[1] * np.cos(theta * np.pi / 180)))
        final_ny = final_ny - int(crop_y * 2)
    else:
        crop_y = 0

    # Pad YX dimensions to the configured multiple.
    pad_y = (divisible_by - (final_ny % divisible_by)) % divisible_by
    pad_x = (divisible_by - (final_nx % divisible_by)) % divisible_by
    padded_final_ny = final_ny + pad_y
    padded_final_nx = final_nx + pad_x

    return [final_nz, padded_final_ny, padded_final_nx], pad_y, pad_x, crop_y


@njit(inline="always")
def deskew_row(
    z: int,
    y: int,
    planes: int,
    ny: int,
    final_nz: int,
    step: float,
    tangent: float,
    sine: float,
    cosine: float,
) -> tuple:
    """Locate four source rows without reducing coordinate precision.

    Parameters
    ----------
    z, y : int
        Laboratory coordinates in camera-pixel units.
    planes, ny : int
        Scan length and detector height.
    final_nz : int
        Laboratory height before Z averaging.
    step : float
        Scan spacing divided by camera pixel size.
    tangent, sine, cosine : float
        Trigonometric functions of the acquisition angle.

    Returns
    -------
    tuple
        Validity, preceding scan, preceding/following detector rows, and four
        float32 interpolation weights. Invalid rows have zero weights.
    """
    virtual_plane = y - z / tangent
    before = int(np.floor(virtual_plane * (1 / step)))
    if z >= final_nz or before < 0 or before + 1 >= planes:
        return (
            False,
            0,
            0,
            0,
            np.float32(0),
            np.float32(0),
            np.float32(0),
            np.float32(0),
        )
    za = z / sine
    position_before = za + (virtual_plane - before * step) * cosine
    position_after = za - (step - (virtual_plane - before * step)) * cosine
    rb, ra = int(np.floor(position_before)), int(np.floor(position_after))
    valid = rb >= 0 and ra >= 0 and rb + 1 < ny and ra + 1 < ny
    db, da = position_before - rb, position_after - ra
    return (
        valid,
        before,
        rb,
        ra,
        np.float32(da),
        np.float32(1 - da),
        np.float32(db),
        np.float32(1 - db),
    )


@njit(inline="always")
def deskew_sample(
    data: np.ndarray, x: int, geometry: tuple, gain: np.float32
) -> np.float32:
    """Interpolate one pixel from a previously located source quartet.

    Parameters
    ----------
    data : np.ndarray
        Oblique scan, detector-Y, detector-X volume.
    x : int
        Detector column, unchanged by deskewing.
    geometry : tuple
        Source indices and weights returned by ``deskew_row``.
    gain : np.float32
        Reciprocal scan spacing in camera-pixel units.

    Returns
    -------
    np.float32
        Nonnegative interpolated intensity, or zero outside valid support.
    """
    valid, before, rb, ra, a, b, c, d = geometry
    if not valid:
        return np.float32(0)
    value = (
        a * data[before + 1, ra + 1, x]
        + b * data[before + 1, ra, x]
        + c * data[before, rb + 1, x]
        + d * data[before, rb, x]
    ) * gain
    return np.float32(np.maximum(value, np.float32(0)))


@njit(parallel=True)
def interpolate_rows(
    data: ArrayLike,
    output: np.ndarray,
    final_ny: int,
    final_nz: int,
    theta: float = 30.0,
    distance: float = 0.4,
    pixel_size: float = 0.115,
    downsample_factor: int = 2,
    zero_initialized: bool = True,
) -> np.ndarray:
    """Parallelize orthogonal interpolation over independent output rows.

    Parameters
    ----------
    data : ArrayLike
        Oblique volume in scan, detector-Y, detector-X order.
    output : np.ndarray
        Fresh float32 output allocation, filled in place.
    final_ny, final_nz : int
        Unpadded laboratory dimensions before Z averaging.
    theta : float
        Angle to the coverslip, in degrees.
    distance : float
        Scan step in micrometers.
    pixel_size : float
        Camera pixel size in micrometers.
    downsample_factor : int
        Number of adjacent laboratory Z planes to average; defaults to two.
    zero_initialized : bool
        Whether output was allocated with zeros. Skip unsupported rows in that
        case; otherwise write every row, including padding and empty support.

    Returns
    -------
    np.ndarray
        Float32 laboratory ZYX volume with zero padding.

    Notes
    -----
    Coordinates retain float64 precision; weights and intensities use float32.
    Factors one and two write each output pixel once. Other factors accumulate
    into the output row, preserving the existing incomplete-Z-bin convention.
    """
    planes, ny, nx = data.shape
    nz, padded_y, padded_x = output.shape
    angle = np.radians(theta)
    tangent, sine, cosine = np.tan(angle), np.sin(angle), np.cos(angle)
    step = distance / pixel_size
    gain = np.float32(1 / step)
    for task in prange(nz * padded_y):
        row = np.int64(task)
        z_ds, y = row // padded_y, row % padded_y
        if y >= final_ny:
            if not zero_initialized:
                for x in range(padded_x):
                    output[z_ds, y, x] = np.float32(0)
            continue
        g0 = deskew_row(
            z_ds * downsample_factor,
            y,
            planes,
            ny,
            final_nz,
            step,
            tangent,
            sine,
            cosine,
        )
        if downsample_factor == 1:
            if zero_initialized and not g0[0]:
                continue
            for x in range(nx):
                output[z_ds, y, x] = deskew_sample(data, x, g0, gain)
        elif downsample_factor == 2:
            g1 = deskew_row(
                z_ds * 2 + 1, y, planes, ny, final_nz, step, tangent, sine, cosine
            )
            if zero_initialized and not g0[0] and not g1[0]:
                continue
            for x in range(nx):
                total = np.float32(
                    deskew_sample(data, x, g0, gain) + deskew_sample(data, x, g1, gain)
                )
                output[z_ds, y, x] = total / np.float32(2)
        else:
            for x in range(nx):
                output[z_ds, y, x] = np.float32(0)
            for z in range(
                z_ds * downsample_factor, min((z_ds + 1) * downsample_factor, final_nz)
            ):
                geometry = deskew_row(
                    z, y, planes, ny, final_nz, step, tangent, sine, cosine
                )
                if geometry[0]:
                    for x in range(nx):
                        output[z_ds, y, x] += deskew_sample(data, x, geometry, gain)
            for x in range(nx):
                output[z_ds, y, x] /= np.float32(downsample_factor)
        if not zero_initialized:
            for x in range(nx, padded_x):
                output[z_ds, y, x] = np.float32(0)
    return output


def orthogonal_deskew(
    data: ArrayLike,
    theta: float = 30.0,
    distance: float = 0.4,
    pixel_size: float = 0.115,
    reverse_deskewed_z: bool = False,
    divisible_by: int = 4,
    downsample_factor: int = 2,
    *,
    zero_initialized: bool = True,
) -> np.ndarray:
    """Deskew oblique data into a float32 laboratory volume.

    Parameters
    ----------
    data : ArrayLike
        Image stack in scan, detector-Y, detector-X order.
    theta : float
        Acquisition angle relative to the coverslip, in degrees.
    distance : float
        Distance between scan planes in micrometers.
    pixel_size : float
        In-plane camera pixel size in micrometers.
    reverse_deskewed_z : bool
        Reverse the laboratory Z axis after interpolation.
    divisible_by : int
        Pad Y and X to multiples of this integer.
    downsample_factor : int
        Average this many laboratory Z planes; the default remains two.
    zero_initialized : bool
        Use a zero-filled allocation; False writes every voxel explicitly for chunks.

    Returns
    -------
    np.ndarray
        Float32 ZYX volume with sampling
        ``(pixel_size * downsample_factor, pixel_size, pixel_size)``.
    """
    data = np.asarray(data)
    shape, pad_y, _, _ = deskew_shape_estimator(
        data.shape, theta, distance, pixel_size, False, divisible_by
    )
    final_nz, final_ny = shape[0], shape[1] - pad_y
    shape[0] = max(1, final_nz // downsample_factor)
    output = (np.zeros if zero_initialized else np.empty)(tuple(shape), np.float32)
    interpolate_rows(
        data,
        output,
        final_ny,
        final_nz,
        theta,
        distance,
        pixel_size,
        downsample_factor,
        zero_initialized,
    )
    return output[::-1] if reverse_deskewed_z else output
