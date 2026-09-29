"""Fine Cartesian optical fields sampled onto skew acquisition coordinates.

Experimental only: neither the processing CLI nor production PSF/deskew code
imports this module. Generate and blur objects on the Cartesian grid first,
then sample the resulting image at raw camera-pixel centers. The diffraction
parameters match the existing 100x silicone model. Illumination, pixel-area
integration, scanner dynamics, motion, and empirical apodization are excluded.
"""

import numpy as np
import psfmodels
from scipy.interpolate import RegularGridInterpolator


def cartesian_psf(
    spacing_um: float,
    half_extent_zyx_um=(1.5, 0.9, 0.9),
    *,
    wavelength_um: float = 0.637,
    numerical_aperture: float = 1.35,
    depth_um: float = 0.0,
):
    """Return centered physical ZYX axes and a normalized fine optical PSF.

    The grid covers the requested physical extent at the supplied isotropic
    spacing, with an odd number of samples on each axis. XY extents must match
    because the optical library evaluates a square detector field. Do not
    change the physical extent when testing convergence with finer spacing.
    """
    extent = np.asarray(half_extent_zyx_um, dtype=float)
    if (
        not np.isfinite(spacing_um)
        or spacing_um <= 0
        or extent.shape != (3,)
        or not np.isfinite(extent).all()
        or np.any(extent <= 0)
        or extent[1] != extent[2]
    ):
        raise ValueError("Require positive spacing and ZYX extents with equal XY")
    if not np.isfinite(wavelength_um) or wavelength_um <= 0:
        raise ValueError("wavelength_um must be positive and finite")
    if not np.isfinite(depth_um) or depth_um < 0:
        raise ValueError("depth_um must be nonnegative and finite")
    if not np.isfinite(numerical_aperture) or not 0 < numerical_aperture <= 1.4:
        raise ValueError(
            "numerical_aperture must be positive and at most the immersion RI 1.4"
        )
    radii = np.ceil(extent / spacing_um).astype(int)
    axes = tuple(np.arange(-r, r + 1) * spacing_um for r in radii)
    psf = psfmodels.vectorial_psf(
        zv=axes[0] + depth_um,
        nx=axes[2].size,
        dxy=spacing_um,
        pz=depth_um,
        wvl=wavelength_um,
        params={
            "NA": numerical_aperture,
            "ni0": 1.4,
            "ni": 1.4,
            "ns": 1.38,
            "tg0": 170,
            "tg": 170,
            "ti0": 300,
        },
    )
    psf = np.asarray(psf, dtype=np.float64)
    psf /= psf.sum()
    return axes, psf


def sample_skewed(
    field: np.ndarray,
    axes_zyx_um,
    shape_syx,
    *,
    pixel_size_um: float = 0.115,
    scan_step_um: float = 0.8,
    angle_deg: float = 30.0,
    center_xyz_um=(0.0, 0.0, 0.0),
):
    """Trilinearly sample a Cartesian image at physical raw pixel centers.

    Output order is scan, camera-row, camera-column, centered at the given XYZ
    coordinate. Both odd and even shapes retain their physical half-pixel
    centers. Samples outside the provided Cartesian field are explicitly zero.
    No intensity normalization or voxel-volume conversion is applied: the
    returned values have the same units as the input field.
    """
    shape = tuple(shape_syx)
    if len(shape) != 3 or any(
        isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1
        for n in shape
    ):
        raise ValueError("shape_syx must contain three positive integers")
    if (
        not np.isfinite(pixel_size_um)
        or pixel_size_um <= 0
        or not np.isfinite(scan_step_um)
        or scan_step_um <= 0
        or not np.isfinite(angle_deg)
        or not 0 < angle_deg < 90
    ):
        raise ValueError("Require positive sampling and an angle between 0 and 90")
    center = np.asarray(center_xyz_um, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center_xyz_um must contain three finite coordinates")
    theta = np.deg2rad(angle_deg)
    scan = (np.arange(shape[0]) - (shape[0] - 1) / 2) * scan_step_um
    row = (np.arange(shape[1]) - (shape[1] - 1) / 2) * pixel_size_um
    col = (np.arange(shape[2]) - (shape[2] - 1) / 2) * pixel_size_um
    interpolate = RegularGridInterpolator(
        axes_zyx_um,
        field,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    result = np.empty(
        shape, dtype=np.float32 if field.dtype == np.float32 else np.float64
    )
    # A fine sphere can contain millions of camera samples. Bound temporary
    # coordinate arrays while keeping the same trilinear interpolation.
    for start in range(0, shape[0], 8):
        end = min(start + 8, shape[0])
        z, y, x = np.broadcast_arrays(
            center[2] + row[None, :, None] * np.sin(theta),
            center[1]
            + scan[start:end, None, None]
            + row[None, :, None] * np.cos(theta),
            center[0] + col[None, None, :],
        )
        result[start:end] = interpolate(np.stack((z, y, x), axis=-1))
    return result
