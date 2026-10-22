"""Generate silicone-objective OPM PSFs for planar and oblique acquisitions.

Sample the vectorial optical model in physical coordinates and interpolate its
Cartesian planes onto the acquisition scan grid used by production deconvolution.
The oblique kernel integrates the optical field over each physical camera pixel
using midpoint quadrature. Numerical oversampling controls quadrature and optical
grid resolution, while the camera pitch sets pixel area. Radial tapers share the
optical kernel center.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import psfmodels as psfm
from scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from collections.abc import Sequence


def _lens_parameters(parameters: dict) -> dict:
    """Require recorded physical lens parameters before evaluating the optics.

    Parameters
    ----------
    parameters : dict
        PSF settings containing the recorded ``lens`` calibration.

    Returns
    -------
    dict
        Lens calibration, with no implicit instrument parameters supplied.

    Raises
    ------
    ValueError
        A required physical lens parameter is missing.
    """
    lens = parameters.get("lens", {})
    missing = {"NA", "ni0", "ni", "ns", "ng0", "ng", "tg0", "tg", "ti0"} - lens.keys()
    if missing:
        raise ValueError(f"Missing PSF lens calibration: {', '.join(sorted(missing))}")
    return lens


def integrate_camera_pixels(
    optical_field: np.ndarray,
    axes: Sequence[np.ndarray],
    sensor_axes: Sequence[np.ndarray],
    *,
    pixel_size_um: float,
    theta_deg: float,
    samples: int,
    geometry: str = "oblique",
    area_integral: bool = True,
) -> np.ndarray:
    """Integrate a Cartesian optical field over physical camera pixels.

    Parameters
    ----------
    optical_field : numpy.ndarray
        Optical intensity in Cartesian ZYX order.
    axes : sequence of numpy.ndarray
        Cartesian ZYX coordinates in micrometers.
    sensor_axes : sequence of numpy.ndarray
        Pixel-center coordinates: scan, camera row and column for oblique
        acquisitions, or Cartesian ZYX for orthogonal acquisitions.
    pixel_size_um : float
        Physical camera-pixel width in object space.
    theta_deg : float
        Oblique camera angle to the coverslip, in degrees.
    samples : int
        Positive midpoint-quadrature count along each pixel dimension.
    geometry : {"oblique", "cartesian"}
        Camera orientation relative to the optical field.
    area_integral : bool
        Include physical pixel area; False returns the mean sampled intensity.

    Returns
    -------
    numpy.ndarray
        Pixel-integrated field on the requested sensor grid, with zero extension.

    Raises
    ------
    ValueError
        Pixel calibration, quadrature or coordinate arrays are invalid.
    """
    if (
        isinstance(samples, bool)
        or not isinstance(samples, (int, np.integer))
        or samples < 1
    ):
        raise ValueError("Pixel quadrature samples must be positive integers")
    if (
        not np.isfinite(pixel_size_um)
        or pixel_size_um <= 0
        or not np.isfinite(theta_deg)
    ):
        raise ValueError("Require positive pixel width and a finite camera angle")
    if geometry not in ("oblique", "cartesian"):
        raise ValueError("Require oblique or cartesian sensor geometry")
    if len(sensor_axes) != 3 or any(
        np.ndim(a) != 1 or not len(a) or not np.isfinite(a).all() for a in sensor_axes
    ):
        raise ValueError("Sensor axes must be three nonempty finite coordinate arrays")
    integrate = RegularGridInterpolator(
        axes, optical_field, bounds_error=False, fill_value=0
    )
    offsets = ((np.arange(samples) + 0.5) / samples - 0.5) * pixel_size_um
    theta = np.deg2rad(theta_deg)
    shape = tuple(map(len, sensor_axes))
    result = np.zeros(shape, dtype=np.result_type(optical_field.dtype, np.float32))
    a, b, c = sensor_axes
    for start in range(0, shape[0], 8):
        for dy in offsets:
            for dx in offsets:
                if geometry == "oblique":
                    z, y, x = np.broadcast_arrays(
                        (b[None, :, None] + dy) * np.sin(theta),
                        a[start : start + 8, None, None]
                        + (b[None, :, None] + dy) * np.cos(theta),
                        c[None, None, :] + dx,
                    )
                else:
                    z, y, x = np.broadcast_arrays(
                        a[start : start + 8, None, None],
                        b[None, :, None] + dy,
                        c[None, None, :] + dx,
                    )
                result[start : start + 8] += integrate(np.stack((z, y, x), axis=-1))
    result /= samples**2
    if area_integral:
        result *= pixel_size_um**2
    return result


# ROI tools
def get_skewed_roi_size(
    sizes: Sequence[float],
    theta: float,
    dc: float,
    dstep: float,
    ensure_odd: bool = True,
) -> list[int]:
    """Calculate the ROI size in the OPM matrix to include sufficient xy and z points.

    Parameters
    ----------
    sizes : list of float
        [z-size, y-size, x-size] in the same units as `dc` and `dstep`.
    theta : float
        Angle in radians.
    dc : float
        Camera pixel size.
    dstep : float
        Step size.
    ensure_odd : bool, optional
        Whether to ensure the ROI dimensions are odd. Default is True.

    Returns
    -------
    list of int
        [n0, n1, n2]: Integer size of ROI in skewed coordinates.
    """
    # x-size determines n2 size
    n2 = int(np.ceil(sizes[2] / dc))

    # z-size determines n1
    n1 = int(np.ceil(sizes[0] / dc / np.sin(theta)))

    # set so that @ top and bottom z-points, ROI includes the full y-size
    n0 = int(np.ceil(((0.5 * (n1 + 1)) * dc * np.cos(theta) + sizes[1]) / dstep))

    if ensure_odd:
        if np.mod(n2, 2) == 0:
            n2 += 1

        if np.mod(n1, 2) == 0:
            n1 += 1

        if np.mod(n0, 2) == 0:
            n0 += 1

    return [n0, n1, n2]


# coordinate transformations between OPM and coverslip frames
def get_skewed_coords(
    sizes: Sequence[int],
    dc: float,
    ds: float,
    theta: float,
    scan_direction: str = "lateral",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get laboratory coordinates (i.e., coverslip coordinates) for a stage-scanning OPM set.

    Parameters
    ----------
    sizes : tuple of int
        A tuple (n0, n1, n2) where:
        - n0: Number of images (nimgs).
        - n1: Number of camera pixels in the y-direction (ny_cam).
        - n2: Number of camera pixels in the x-direction (nx_cam).
    dc : float
        Camera pixel size.
    ds : float
        Stage step size.
    theta : float
        Angle in radians.
    scan_direction : {'lateral', 'axial'}, optional
        Direction of the scan. Must be either 'lateral' or 'axial'. Default is 'lateral'.

    Returns
    -------
    x : ndarray
        X-coordinates of the laboratory frame.
    y : ndarray
        Y-coordinates of the laboratory frame.
    z : ndarray
        Z-coordinates of the laboratory frame.

    Raises
    ------
    ValueError
        If `scan_direction` is not 'lateral' or 'axial'.

    Notes
    -----
    - For `scan_direction='lateral'`, the stage moves laterally, and the y-coordinates are computed
      based on the stage position and the camera pixel size.
    - For `scan_direction='axial'`, the stage moves axially, and the z-coordinates are computed
      based on the stage position and the camera pixel size.
    """
    nimgs, ny_cam, nx_cam = sizes

    if scan_direction == "lateral":
        x = dc * np.arange(nx_cam)[None, None, :]
        # y = stage_pos[:, None, None] + dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
        y = (
            ds * np.arange(nimgs)[:, None, None]
            + dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
        )
        z = dc * np.sin(theta) * np.arange(ny_cam)[None, :, None]
    elif scan_direction == "axial":
        x = dc * np.arange(nx_cam)[None, None, :]
        y = dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
        z = (
            ds * np.arange(nimgs)[:, None, None]
            + dc * np.sin(theta) * np.arange(ny_cam)[None, :, None]
        )
    else:
        raise ValueError(
            f"scan_direction must be `lateral` or `axial` but was `{scan_direction}`"
        )

    return x, y, z


def create_psf_silicone_100x(
    dxy: float,
    dz: float,
    nxy: int,
    nz: int,
    em_wvl: float,
    pz: float,
    *,
    parameters: dict,
) -> np.ndarray:
    """Evaluate the vectorial emission model with the existing lateral mask.

    The Gaussian mask width is supplied in Cartesian grid pixels. This preserves the
    established model; it does not introduce a measured light-sheet profile.
    The masked result is normalized by the deconvolution solver.

    Parameters
    ----------
    dxy : float
        Spacing of xy pixels.
    dz : float
        Spacing of z planes.
    nxy : int
        Number of xy pixels on a side.
    nz : int
        Number of z planes.
    em_wvl : float
        Emission wavelength in microns.
    pz : float
        Microns above coverslip.
    parameters : dict
        Lens calibration and the supplied three-dimensional taper width.

    Returns
    -------
    tot_psf : ndarray
        Theoretical PSF in coverslip coordinates.
    """
    em_lens = _lens_parameters(parameters)

    lim = (nz - 1) * dz / 2
    zv = np.linspace(-lim + pz, lim + pz, nz)

    psf = psfm.vectorial_psf(zv=zv, nx=nxy, dxy=dxy, pz=pz, wvl=em_wvl, params=em_lens)
    psf = psf / np.sum(psf)
    y, x = np.ogrid[: psf.shape[1], : psf.shape[2]]
    r = np.hypot(y - (psf.shape[1] - 1) / 2, x - (psf.shape[2] - 1) / 2)
    sigma = parameters["taper_sigma_px_3d"]
    radial_taper = np.exp(-np.power(r, 2) / (2 * sigma**2))
    radial_taper = radial_taper / np.max(radial_taper)

    psf = psf * radial_taper

    return psf


def generate_proj_psf(
    em_wvl: float, pixel_size_um: float, *, parameters: dict
) -> np.ndarray:
    """Generate a central optical-plane kernel with the existing lateral mask.

    The solver normalizes the kernel after masking.

    Parameters
    ----------
    em_wvl
        Emission wavelength in micrometers.
    pixel_size_um
        Camera-plane pixel size in micrometers.
    parameters
        Explicit lens calibration, emitter depth, grid dimensions and taper widths.

    Returns
    -------
    numpy.ndarray
        Masked central PSF plane in ZYX order with a singleton Z dimension.
    """
    em_lens = _lens_parameters(parameters)
    pz = parameters["projection_depth_um"]
    psf = psfm.vectorial_psf_centered(
        nz=parameters["projection_nz"],
        dz=parameters["projection_dz_um"],
        nx=parameters["projection_nxy"],
        dxy=pixel_size_um,
        pz=pz,
        wvl=em_wvl,
        params=em_lens,
    )
    psf = psf / np.sum(psf)
    # A projection acquisition contains one observed YX plane. Use the central
    # optical plane rather than integrating the 3D PSF along Z, which broadens
    # the 2D deconvolution kernel.
    psf = np.asarray(psf[psf.shape[0] // 2], dtype=np.float32)
    psf = psf / np.sum(psf)
    if psf.ndim == 2:
        psf = psf[np.newaxis, :, :]
    y, x = np.ogrid[: psf.shape[1], : psf.shape[2]]
    r = np.hypot(y - (psf.shape[1] - 1) / 2, x - (psf.shape[2] - 1) / 2)
    sigma = parameters["taper_sigma_px_2d"]
    radial_taper = np.exp(-np.power(r, 2) / (2 * sigma**2))
    radial_taper = radial_taper / np.max(radial_taper)

    psf = psf * radial_taper

    return psf


def generate_skewed_psf(
    em_wvl: float,
    pixel_size_um: float,
    scan_axis_step_um: float,
    theta_deg: float,
    *,
    parameters: dict,
) -> np.ndarray:
    """Integrate the optical PSF over pixels in the oblique camera plane.

    Parameters
    ----------
    em_wvl: float
        emission wavelength in microns
    pixel_size_um: float
        pixel size in microns
    scan_axis_step_um: float
        step size in microns
    theta_deg: float
        OPM tilt angle in degrees.
    parameters: dict
        Lens calibration, source depth, support and taper. ``grid_oversampling``
        is a positive integer controlling optical-grid spacing and midpoint
        quadrature samples along each camera-pixel dimension.

    Returns
    -------
    skewed_psf: np.ndarray
        OPM PSF in skewed coordinates
    """
    dc = pixel_size_um
    lens = _lens_parameters(parameters)
    na = lens["NA"]
    ni = lens["ni"]
    pz = parameters["skewed_depth_um"]
    dstage = scan_axis_step_um
    theta = np.deg2rad(theta_deg)

    xy_res = 1.6163399561827614 / np.pi * em_wvl / na
    z_res = 2.355 * (np.sqrt(6) / np.pi * ni * em_wvl / na**2)
    z_res_mod = (
        z_res
        + np.divide(pz + 1e-12, parameters["depth_scale_um"])
        / parameters["depth_width_scale"]
    )

    roi_skewed_size = get_skewed_roi_size(
        [
            z_res_mod * parameters["support_fwhms"],
            xy_res * parameters["support_fwhms"],
            xy_res * parameters["support_fwhms"],
        ],
        theta,
        dc,
        dstage,
        ensure_odd=True,
    )
    # make square
    roi_skewed_size[2] = roi_skewed_size[1]

    samples = parameters["grid_oversampling"]
    if isinstance(samples, bool) or int(samples) != samples or samples < 1:
        raise ValueError("grid_oversampling must be a positive integer")
    samples = int(samples)
    x, y, _ = get_skewed_coords(roi_skewed_size, dc, dstage, theta)
    x, y = x - x.mean(), y - y.mean()
    dxy = dc * np.cos(theta) / samples
    dz = dc * np.sin(theta) / samples
    # Cover the complete pixel footprints, with one Z guard plane on each side.
    half_xy = max(
        np.max(np.abs(x)) + dc / 2, np.max(np.abs(y)) + dc * np.cos(theta) / 2
    )
    nxy = 2 * int(np.ceil(half_xy / dxy)) + 1
    nz = roi_skewed_size[1] * samples + 2
    axis_xy = (np.arange(nxy) - nxy // 2) * dxy
    # psfmodels otherwise integrates Cartesian XY cells of width dxy. Evaluate
    # point intensities here; the camera pixel is tilted and has width dc.
    optical_parameters = dict(
        parameters, lens={**parameters["lens"], "sf": 1, "mode": 1}
    )
    psf_grid = create_psf_silicone_100x(
        dxy, dz, nxy, nz, em_wvl, pz, parameters=optical_parameters
    )
    sensor_axes = tuple(
        (np.arange(n) - (n - 1) / 2) * spacing
        for n, spacing in zip(roi_skewed_size, (dstage, dc, dc), strict=True)
    )
    axis_z = (np.arange(nz) - (nz - 1) / 2) * dz
    return integrate_camera_pixels(
        psf_grid,
        (axis_z, axis_xy, axis_xy),
        sensor_axes,
        pixel_size_um=dc,
        theta_deg=theta_deg,
        samples=samples,
    )
