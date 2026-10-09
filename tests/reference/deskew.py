"""Analytical deskew intensity and support for a uniform fluorescent volume."""

import numpy as np


def constant_deskew(
    shape, value, *, distance=0.4, pixel_size=0.115, theta=30.0, downsample_factor=2
):
    """Evaluate constant fluorescence without production deskew kernels.

    Parameters
    ----------
    shape : tuple of int
        Scan, detector row and detector column counts.
    value : float
        Calibrated constant fluorescence in the acquired volume.
    distance, pixel_size : float
        Scan spacing and detector pixel pitch in micrometers.
    theta : float
        Camera plane angle to the coverslip in degrees.
    downsample_factor : int
        Number of laboratory Z planes averaged per stored output plane.

    Returns
    -------
    numpy.ndarray
        Float32 ZYX fluorescence including unsupported voxels and YX padding.

    Notes
    -----
    Two interpolated scan planes give gain ``2 * pixel_size / distance``.
    Both bracketing scans and both detector row pairs must lie inside the
    acquisition. Missing support contributes zero to a complete Z-bin divisor.
    """
    planes, rows, columns = shape
    angle = np.deg2rad(theta)
    step = distance / pixel_size
    nz = int(np.ceil(rows * np.sin(angle)))
    ny = int(np.ceil(planes * step + rows * np.cos(angle)))
    z, y = np.indices((nz, ny), dtype=np.float64)
    scan_position = y - z / np.tan(angle)
    before = np.floor(scan_position / step)
    displacement = scan_position - before * step
    row_before = z / np.sin(angle) + displacement * np.cos(angle)
    row_after = z / np.sin(angle) - (step - displacement) * np.cos(angle)
    supported = (
        (before >= 0)
        & (before + 1 < planes)
        & (row_before >= 0)
        & (row_before < rows - 1)
        & (row_after >= 0)
        & (row_after < rows - 1)
    )
    padded_z = ((nz + downsample_factor - 1) // downsample_factor) * downsample_factor
    intensity = np.zeros(
        (padded_z, ((ny + 3) // 4) * 4, ((columns + 3) // 4) * 4), np.float32
    )
    intensity[:nz, :ny, :columns] = supported[..., None] * (2 * value / step)
    return intensity.reshape(-1, downsample_factor, *intensity.shape[1:]).mean(axis=1)
