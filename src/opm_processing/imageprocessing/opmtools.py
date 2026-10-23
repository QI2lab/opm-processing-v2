"""Orthogonal interpolation of oblique camera scans onto laboratory ZYX grids.

References
----------
Vincent Albert Maioli, "High-speed 3-D fluorescence imaging by oblique plane
microscopy: multi-well plate-reader development, biological applications and
image analysis", PhD thesis, Imperial College London (awarded 2017):
https://doi.org/10.25560/68022 . This is the orthogonal-interpolation reference
cited for the original qi2lab stage-scanning implementation in Sapoznik et al.
(2020), https://doi.org/10.7554/eLife.57681 .

Notes
-----
Interpolation projects each output point perpendicularly onto its two adjacent
acquired planes, interpolates between two detector rows on each plane, then
combines those four source samples. It does not apply an affine image warp.

Compared with the earlier OPM code's equal-plane contribution, plane values
are now weighted by d1/(d0+d1) and d0/(d0+d1), where d0,d1 are perpendicular
distances. These normalized weights preserve constants and linear laboratory
fields. The supported 30-degree photon-volume gain is 2*pixel_size/scan_step;
it is applied separately from the normalized interpolation. Optional Z averaging
retains the full bin divisor for partial bins, with coverage handled in fusion.

Output tiling evaluates this same kernel at global ZYX coordinates, without
rebasing the oblique geometry or resetting Z-averaging bins at tile boundaries.
The complete final Z bin is retained, with unsupported samples contributing zero.

Geometry uses float64 coordinates; weights and output intensities use float32.
Numba parallelizes independent output rows and reuses their quartet geometry
across detector X. Unsupported samples and padding are zero. Singleton/paired
Z bins use direct writes; larger bins accumulate without intermediate image
allocations. Float output is retained here; final output casting belongs to the
processing workflow. Output shape, orientation and padding conventions remain
explicit in the public functions below.
"""

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
    theta: float,
    distance: float,
    pixel_size: float,
    crop_after_deskew: bool = True,
    divisible_by: int = 4,
    downsample_factor: int = 1,
) -> tuple[list[int], int, int, int]:
    """Generate shape of orthogonal interpolation output array.

    This function automatically pads the YX dimensions to be
    an integer divisible by `divisible_by`.

    Parameters
    ----------
    input_shape : Sequence[int]
        Oblique scan, camera-Y and camera-X dimensions.
    theta : float
        Recorded angle relative to the coverslip, in degrees.
    distance : float
        Recorded scan spacing along the coverslip, in micrometers.
    pixel_size : float
        Recorded in-plane camera pixel pitch, in micrometers.

    crop_after_deskew : bool
        Remove the lateral triangular deskew margins from the output shape.
    divisible_by : int
        Pad output YX dimensions to multiples of this integer.
    downsample_factor : int
        Number of laboratory Z planes per output bin, including a partial last bin.

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
    if len(input_shape) != 3 or min(input_shape) < 1:
        raise ValueError("Deskew requires a nonempty scan-YX volume")
    if (
        not np.isfinite(theta)
        or not np.isfinite(distance)
        or not np.isfinite(pixel_size)
        or not 0 < theta < 90
        or distance <= 0
        or pixel_size <= 0
    ):
        raise ValueError("Deskew requires finite positive sampling and 0 < theta < 90")
    if (
        divisible_by < 1
        or int(divisible_by) != divisible_by
        or downsample_factor < 1
        or int(downsample_factor) != downsample_factor
    ):
        raise ValueError("Deskew padding and Z averaging must be positive integers")
    # Convert recorded micrometer spacing to camera-pixel units.
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

    if final_ny < 1:
        raise ValueError("Deskew cropping leaves no laboratory-Y extent")

    # Pad YX dimensions to the configured multiple.
    pad_y = (divisible_by - (final_ny % divisible_by)) % divisible_by
    pad_x = (divisible_by - (final_nx % divisible_by)) % divisible_by
    padded_final_ny = final_ny + pad_y
    padded_final_nx = final_nx + pad_x

    averaged_nz = (final_nz + int(downsample_factor) - 1) // int(downsample_factor)
    return [averaged_nz, padded_final_ny, padded_final_nx], pad_y, pad_x, crop_y


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
        float32 interpolation weights combining detector-row interpolation with
        perpendicular-distance weighting between planes. Valid weights sum to
        one; invalid rows have zero weights.
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
    displacement = virtual_plane - before * step
    position_before = za + displacement * cosine
    position_after = za - (step - displacement) * cosine
    rb, ra = int(np.floor(position_before)), int(np.floor(position_after))
    valid = rb >= 0 and ra >= 0 and rb + 1 < ny and ra + 1 < ny
    db, da = position_before - rb, position_after - ra
    # Perpendicular distances are displacement*sin(theta) and
    # (step-displacement)*sin(theta). Their common sine cancels in d_other/sum(d).
    after_weight = min(1.0, max(0.0, displacement / step))
    before_weight = 1.0 - after_weight
    return (
        valid,
        before,
        rb,
        ra,
        np.float32(after_weight * da),
        np.float32(after_weight * (1 - da)),
        np.float32(before_weight * db),
        np.float32(before_weight * (1 - db)),
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
        Photon-volume gain, ``2 / step`` at the supported 30-degree angle.

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
    theta: float,
    distance: float,
    pixel_size: float,
    downsample_factor: int = 2,
    zero_initialized: bool = True,
    output_origin: tuple[int, int, int] = (0, 0, 0),
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
    output_origin : tuple[int, int, int]
        Global averaged-Z, Y and X indices of the output window's first voxel.
        Source data and unpadded dimensions always describe the full input volume.

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
    z_origin, y_origin, x_origin = output_origin
    valid_x = min(padded_x, max(0, nx - x_origin))
    angle = np.radians(theta)
    tangent, sine, cosine = np.tan(angle), np.sin(angle), np.cos(angle)
    step = distance / pixel_size
    gain = np.float32(2 / step)
    for task in prange(nz * padded_y):
        row = np.int64(task)
        z_local, y_local = row // padded_y, row % padded_y
        z_ds, y = z_local + z_origin, y_local + y_origin
        if y >= final_ny:
            if not zero_initialized:
                for x in range(padded_x):
                    output[z_local, y_local, x] = np.float32(0)
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
            for x in range(valid_x):
                output[z_local, y_local, x] = deskew_sample(
                    data, x + x_origin, g0, gain
                )
        elif downsample_factor == 2:
            g1 = deskew_row(
                z_ds * 2 + 1, y, planes, ny, final_nz, step, tangent, sine, cosine
            )
            if zero_initialized and not g0[0] and not g1[0]:
                continue
            for x in range(valid_x):
                total = np.float32(
                    deskew_sample(data, x + x_origin, g0, gain)
                    + deskew_sample(data, x + x_origin, g1, gain)
                )
                output[z_local, y_local, x] = total / np.float32(2)
        else:
            for x in range(valid_x):
                output[z_local, y_local, x] = np.float32(0)
            for z in range(
                z_ds * downsample_factor, min((z_ds + 1) * downsample_factor, final_nz)
            ):
                geometry = deskew_row(
                    z, y, planes, ny, final_nz, step, tangent, sine, cosine
                )
                if geometry[0]:
                    for x in range(valid_x):
                        output[z_local, y_local, x] += deskew_sample(
                            data, x + x_origin, geometry, gain
                        )
            for x in range(valid_x):
                output[z_local, y_local, x] /= np.float32(downsample_factor)
        if not zero_initialized:
            for x in range(valid_x, padded_x):
                output[z_local, y_local, x] = np.float32(0)
    return output


def orthogonal_deskew(
    data: ArrayLike,
    theta: float,
    distance: float,
    pixel_size: float,
    reverse_deskewed_z: bool = False,
    divisible_by: int = 4,
    downsample_factor: int = 2,
    *,
    zero_initialized: bool = True,
    tile_shape: tuple[int, int, int] | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Deskew oblique data on one global grid, optionally writing output tiles.

    Parameters
    ----------
    data : ArrayLike
        Full image stack in scan, detector-Y, detector-X order.
    theta : float
        Recorded acquisition angle relative to the coverslip, in degrees.
    distance : float
        Recorded distance between scan planes in micrometers.
    pixel_size : float
        Recorded in-plane camera pixel size in micrometers.
    reverse_deskewed_z : bool
        Reverse the laboratory Z axis after interpolation.
    divisible_by : int
        Pad Y and X to multiples of this positive integer.
    downsample_factor : int
        Average this many laboratory Z planes. A partial final bin uses the full
        divisor, with unsupported source samples contributing zero.
    zero_initialized : bool
        Use a zero-filled internal allocation; False explicitly writes every voxel.
        Supplied output buffers always receive explicit writes.
    tile_shape : tuple[int, int, int] or None
        Maximum averaged-Z, Y and X output tile dimensions. None evaluates the
        full output at once. All tiles use the same global interpolation grid.
    out : numpy.ndarray or None
        Optional writable float32 output array with the exact deskewed shape.
        NumPy views and memory-mapped arrays are supported. A memory map together
        with tile_shape bounds resident output storage; input data stays available
        as a full volume. The buffer must not overlap the source data.

    Returns
    -------
    numpy.ndarray
        Float32 ZYX volume, or the supplied output buffer, with spacing
        ``(pixel_size * downsample_factor, pixel_size, pixel_size)``.

    Raises
    ------
    ValueError
        Input dimensions, physical calibration, averaging, tile dimensions or
        output buffer are invalid.
    """
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError("Deskew requires a nonempty scan-YX volume")
    shape, pad_y, _, _ = deskew_shape_estimator(
        data.shape, theta, distance, pixel_size, False, divisible_by, downsample_factor
    )
    downsample_factor = int(downsample_factor)
    final_nz = int(np.ceil(data.shape[1] * np.sin(np.radians(theta))))
    final_ny = shape[1] - pad_y
    shape = tuple(shape)
    if tile_shape is None:
        tile_shape = shape
    if len(tile_shape) != 3 or any(
        value < 1 or int(value) != value for value in tile_shape
    ):
        raise ValueError("Deskew tile_shape must contain three positive integers")
    tile_shape = tuple(int(value) for value in tile_shape)
    if out is not None:
        if (
            not isinstance(out, np.ndarray)
            or out.shape != shape
            or out.dtype != np.float32
            or not out.flags.writeable
            or np.may_share_memory(out, data)
        ):
            raise ValueError(
                "Deskew out must be writable float32, match the output shape and not overlap data"
            )
        zero_initialized = False
    else:
        out = (np.zeros if zero_initialized else np.empty)(shape, np.float32)
    target = out[::-1] if reverse_deskewed_z else out
    for z in range(0, shape[0], tile_shape[0]):
        for y in range(0, shape[1], tile_shape[1]):
            for x in range(0, shape[2], tile_shape[2]):
                interpolate_rows(
                    data,
                    target[
                        z : z + tile_shape[0],
                        y : y + tile_shape[1],
                        x : x + tile_shape[2],
                    ],
                    final_ny,
                    final_nz,
                    theta,
                    distance,
                    pixel_size,
                    downsample_factor,
                    zero_initialized,
                    (z, y, x),
                )
    return out
