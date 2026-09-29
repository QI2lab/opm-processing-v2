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

from tqdm import tqdm
import numpy as np
from numpy.typing import ArrayLike
from typing import Sequence, Tuple
from numba import njit, prange
import gc

from opm_processing.imageprocessing.camera import camera_correct, illumination_correct


@njit
def deskew_shape_estimator(
    input_shape: Sequence[int],
    theta: float = 30.0,
    distance: float = 0.4,
    pixel_size: float = 0.115,
    crop_after_deskew: bool = True,
    divisble_by: int = 4,
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

    crop_after_deskew : bool
        Value supplied for ``crop after deskew``.
    divisble_by : int
        Value supplied for ``divisble by``.

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

    if crop_after_deskew:
        crop_y = int(np.ceil(input_shape[1] * np.cos(theta * np.pi / 180)))
        final_ny = final_ny - int(crop_y * 2)
    else:
        crop_y = 0

    # pad YX array size to make sure it is divisble by 4
    pad_y = (divisble_by - (final_ny % divisble_by)) % divisble_by
    pad_x = (divisble_by - (final_nx % divisble_by)) % divisble_by
    padded_final_ny = final_ny + pad_y
    padded_final_nx = final_nx + pad_x

    return [final_nz, padded_final_ny, padded_final_nx], pad_y, pad_x, crop_y


@njit(inline="always")
def _deskew_row(
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
def _deskew_sample(
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
        Source indices and weights returned by ``_deskew_row``.
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
def _interpolate_rows(
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
        g0 = _deskew_row(
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
                output[z_ds, y, x] = _deskew_sample(data, x, g0, gain)
        elif downsample_factor == 2:
            g1 = _deskew_row(
                z_ds * 2 + 1, y, planes, ny, final_nz, step, tangent, sine, cosine
            )
            if zero_initialized and not g0[0] and not g1[0]:
                continue
            for x in range(nx):
                total = np.float32(
                    _deskew_sample(data, x, g0, gain)
                    + _deskew_sample(data, x, g1, gain)
                )
                output[z_ds, y, x] = total / np.float32(2)
        else:
            for x in range(nx):
                output[z_ds, y, x] = np.float32(0)
            for z in range(
                z_ds * downsample_factor, min((z_ds + 1) * downsample_factor, final_nz)
            ):
                geometry = _deskew_row(
                    z, y, planes, ny, final_nz, step, tangent, sine, cosine
                )
                if geometry[0]:
                    for x in range(nx):
                        output[z_ds, y, x] += _deskew_sample(data, x, geometry, gain)
            for x in range(nx):
                output[z_ds, y, x] /= np.float32(downsample_factor)
        if not zero_initialized:
            for x in range(nx, padded_x):
                output[z_ds, y, x] = np.float32(0)
    return output


def _orthogonal_deskew_float32(
    data: ArrayLike,
    theta: float = 30.0,
    distance: float = 0.4,
    pixel_size: float = 0.115,
    reverse_deskewed_z: bool = False,
    divisible_by: int = 4,
    downsample_factor: int = 2,
    zero_initialized: bool = True,
) -> np.ndarray:
    """Allocate a deskew output and invoke the shared row kernel.

    Parameters
    ----------
    data : ArrayLike
        Oblique volume in scan, detector-Y, detector-X order.
    theta : float
        Angle relative to the coverslip in degrees.
    distance, pixel_size : float
        Scan spacing and detector pixel size in micrometers.
    reverse_deskewed_z : bool
        Reverse the output Z axis.
    divisible_by : int
        Pad output Y and X to multiples of this integer.
    downsample_factor : int
        Laboratory Z averaging factor.
    zero_initialized : bool
        Allocate zeros for direct volumes; allocate empty for chunked volumes.

    Returns
    -------
    np.ndarray
        Float32 laboratory ZYX data.
    """
    shape, pad_y, _, _ = deskew_shape_estimator(
        data.shape, theta, distance, pixel_size, False, divisible_by
    )
    final_nz, final_ny = shape[0], shape[1] - pad_y
    shape[0] = max(1, final_nz // downsample_factor)
    output = (np.zeros if zero_initialized else np.empty)(tuple(shape), np.float32)
    _interpolate_rows(
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


def orthogonal_deskew(
    data: ArrayLike,
    theta: float = 30.0,
    distance: float = 0.4,
    pixel_size: float = 0.115,
    reverse_deskewed_z: bool = False,
    divisible_by: int = 4,
    downsample_factor: int = 2,
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

    Returns
    -------
    np.ndarray
        Float32 ZYX volume with sampling
        ``(pixel_size * downsample_factor, pixel_size, pixel_size)``.
    """
    return _orthogonal_deskew_float32(
        np.asarray(data),
        theta,
        distance,
        pixel_size,
        reverse_deskewed_z,
        divisible_by,
        downsample_factor,
    )


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


def chunk_indices(length: int, chunk_size: int) -> list[tuple[int, int]]:
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
    if length <= 0:
        raise ValueError("length must be greater than 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    return [
        (start, min(start + chunk_size, length))
        for start in range(0, length, chunk_size)
    ]


def _deconvolve_oblique_chunk(
    image: np.ndarray,
    psf: np.ndarray,
    crop_scan: int,
    on_successful_crop_scan=None,
) -> np.ndarray:
    """Run scan-axis RLGC without importing CuPy for deskew-only use.

    Parameters
    ----------
    image : np.ndarray
        Value supplied for ``image``.
    psf : np.ndarray
        Value supplied for ``psf``.
    crop_scan : int
        Retained tile size along axis 0, the acquisition scan axis.
    on_successful_crop_scan : callable or None
        Callback used to retain a successful fallback crop.

    Returns
    -------
    np.ndarray
        Result produced by the callable.
    """
    from opm_processing.imageprocessing.rlgc import chunked_rlgc

    return chunked_rlgc(
        image=image,
        psf=psf,
        crop_scan=crop_scan,
        on_successful_crop_scan=on_successful_crop_scan,
    )


def chunked_orthogonal_deskew(
    oblique_image: ArrayLike,
    psf_data: ArrayLike | None = None,
    deconvolve: bool = False,
    decon_chunk_size: int | None = None,
    chunk_size: int = 15000,
    overlap_size: int = 550,
    scan_crop: int = 700,
    camera_bkd: int = 100,
    camera_cf: float = 0.24,
    camera_qe: float = 0.9,
    illumination: ArrayLike | None = None,
    apply_stage_scan_gain: bool = False,
    z_downsample_level: int = 2,
    theta_deg: float = 30.0,
    scan_axis_step_um: float = 0.4,
    pixel_size_um: float = 0.115,
) -> ArrayLike:
    """Chunked orthogonal deskew of oblique data.

    Optionally performs nested scan-axis deconvolution on each outer deskew
    chunk. Deconvolution always receives the full camera Y and X extents.

    Parameters
    ----------
    oblique_image: ArrayLike
        oblique image stack
    psf_data: ArrayLike
        PSF data for deconvolution. Required when ``deconvolve`` is True.
    deconvolve: bool
        Run RLGC before deskewing each outer chunk.
    decon_chunk_size: int
        Retained scan-axis size for nested RLGC chunks. This is independent of
        the outer deskew ``chunk_size``.
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
    theta_deg: float
        OPM tilt angle in degrees.
    scan_axis_step_um: float
        Physical scan-axis step in micrometers.
    pixel_size_um: float
        Camera pixel size in micrometers.

    Returns
    -------
    deskewed_image: ArrayLike
        deskewed image stack
    """
    if deconvolve and psf_data is None:
        raise ValueError("psf_data is required when deconvolve=True")
    if decon_chunk_size is not None and decon_chunk_size <= 0:
        raise ValueError("decon_chunk_size must be greater than 0")
    if scan_axis_step_um <= 0 or pixel_size_um <= 0:
        raise ValueError("scan_axis_step_um and pixel_size_um must be positive")
    if camera_qe <= 0:
        raise ValueError("camera_qe must be positive")
    if deconvolve:
        from opm_processing.imageprocessing.rlgc import RlgcChunkState

        decon_chunk_state = RlgcChunkState(decon_chunk_size)

    estimated_shape, _, _, _ = deskew_shape_estimator(
        oblique_image.shape,
        theta=theta_deg,
        distance=scan_axis_step_um,
        pixel_size=pixel_size_um,
        crop_after_deskew=False,
    )
    output_shape = list(estimated_shape)
    output_shape[0] = output_shape[0] // z_downsample_level
    output_shape[1] = output_shape[1] - scan_crop
    if output_shape[1] <= 0:
        raise ValueError("scan_crop must be smaller than the deskewed Y size")
    deskewed_image = np.zeros(output_shape, dtype=np.float32)

    if chunk_size < output_shape[1]:
        idxs = chunk_indices(output_shape[1], chunk_size)
    else:
        idxs = [(0, output_shape[1])]
        overlap_size = 0

    for idx in tqdm(idxs):
        if idx[0] > 0:
            tile_px_start = idx[0] - overlap_size
        else:
            tile_px_start = idx[0]

        if idx[1] < output_shape[1]:
            tile_px_end = idx[1] + overlap_size
        else:
            if overlap_size == 0:
                tile_px_end = idx[1] + scan_crop
            else:
                tile_px_end = idx[1]

        xp, yp, sp_start = lab2cam(
            oblique_image.shape[2], tile_px_start, 0, np.deg2rad(theta_deg)
        )

        xp, yp, sp_stop = lab2cam(
            oblique_image.shape[2], tile_px_end, 0, np.deg2rad(theta_deg)
        )
        camera_to_scan = pixel_size_um / scan_axis_step_um
        scan_px_start = np.maximum(0, np.int64(np.ceil(sp_start * camera_to_scan)))
        scan_px_stop = np.minimum(
            oblique_image.shape[0],
            np.int64(np.ceil(sp_stop * camera_to_scan)),
        )

        raw_uint16 = np.asarray(oblique_image[scan_px_start:scan_px_stop, :])
        if raw_uint16.dtype != np.dtype(np.uint16):
            raise TypeError(
                f"chunked deskew requires uint16 raw input; received {raw_uint16.dtype}"
            )
        raw_data = camera_correct(
            raw_uint16,
            camera_bkd,
            camera_cf / camera_qe,
            apply_stage_scan_gain=apply_stage_scan_gain,
        )
        if illumination is not None:
            raw_data = illumination_correct(raw_data, illumination)
        if deconvolve:
            effective_crop_scan = decon_chunk_state.determine_once(
                tuple(int(size) for size in raw_data.shape),
                [tuple(int(size) for size in np.asarray(psf_data).shape)],
            )
            raw_data = _deconvolve_oblique_chunk(
                raw_data,
                np.asarray(psf_data),
                crop_scan=effective_crop_scan,
                on_successful_crop_scan=(decon_chunk_state.remember_successful_crop),
            )
        temp_deskew = _orthogonal_deskew_float32(
            raw_data,
            theta=theta_deg,
            distance=scan_axis_step_um,
            pixel_size=pixel_size_um,
            downsample_factor=z_downsample_level,
            zero_initialized=False,
        )

        target_size = idx[1] - idx[0]
        pixel_step = scan_axis_step_um / pixel_size_um
        chunk_global_y_origin = float(scan_px_start) * pixel_step
        local_y_start = int(np.rint(float(idx[0]) - chunk_global_y_origin))
        local_y_start = max(0, local_y_start)
        local_y_stop = local_y_start + target_size
        crop_deskew = temp_deskew[:, local_y_start:local_y_stop, :]
        if crop_deskew.shape[1] < target_size:
            crop_deskew = np.pad(
                crop_deskew,
                ((0, 0), (0, target_size - crop_deskew.shape[1]), (0, 0)),
            )

        deskewed_image[:, idx[0] : idx[1], :] = crop_deskew

    del temp_deskew, oblique_image
    gc.collect()

    return deskewed_image
