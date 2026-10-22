"""Ground-truth construction only; no optics, detector sampling or noise."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def centered_axis(half_extent: float, spacing: float) -> np.ndarray:
    """Cover a physical half extent with an odd grid centered on zero.

    Parameters
    ----------
    half_extent
        Physical distance from the centered grid origin to its edge.
    spacing
        Physical voxel spacing in ZYX order, in micrometers.

    Returns
    -------
    np.ndarray
        Odd-length physical coordinate axis centered at zero.
    """
    radius = int(np.ceil(half_extent / spacing))
    return np.arange(-radius, radius + 1) * spacing


def meridian_sphere(
    axes_zyx: Sequence[np.ndarray],
    diameter_um: float,
    tube_diameter_um: float,
    meridians: int,
) -> np.ndarray:
    """Construct uniform fluorescent tubes around meridian great-circle arcs.

    Diameter refers to the sphere traced by the tube centerlines; the outer
    diameter includes one tube diameter. An even number of pole-to-pole arcs
    forms meridians/2 full circles. Overlapping tubes form a union, not a sum.

    Parameters
    ----------
    axes_zyx
        Cartesian ZYX physical coordinate arrays, in micrometers.
    diameter_um
        Diameter of the meridian sphere, in micrometers.
    tube_diameter_um
        Diameter of each fluorescent meridian tube, in micrometers.
    meridians
        Number of equally spaced fluorescent meridian tubes.

    Returns
    -------
    np.ndarray
        Float32 fluorescence density for the union of the meridian tubes.
    """
    if not 0 < tube_diameter_um < diameter_um:
        raise ValueError("Require 0 < tube diameter < sphere diameter")
    if not isinstance(meridians, int) or meridians < 2 or meridians % 2:
        raise ValueError("meridians must be a positive even integer of at least 2")
    z, y, x = np.meshgrid(
        *[a.astype(np.float32) for a in axes_zyx], indexing="ij", sparse=True
    )
    fluorescent = np.zeros(tuple(len(a) for a in axes_zyx), dtype=bool)
    for phi in np.arange(meridians // 2) * 2 * np.pi / meridians:
        perpendicular = -x * np.sin(phi) + y * np.cos(phi)
        in_plane = x * np.cos(phi) + y * np.sin(phi)
        distance_squared = (
            perpendicular**2 + (np.sqrt(in_plane**2 + z**2) - diameter_um / 2) ** 2
        )
        fluorescent |= distance_squared <= (tube_diameter_um / 2) ** 2
    return fluorescent.astype(np.float32)


def deposit_point(
    density: np.ndarray,
    axes: Sequence[np.ndarray],
    center_xyz: Sequence[float],
    photons: float,
) -> None:
    """Deposit a point conservatively with its specified mass and centroid.

    Parameters
    ----------
    density : numpy.ndarray
        Mutable emitted-photon density on a uniform Cartesian ZYX grid.
    axes : sequence of numpy.ndarray
        Physical ZYX grid axes in micrometers.
    center_xyz : sequence of float
        Emitter position in physical XYZ micrometers.
    photons : float
        Total emitted photon mass deposited into neighboring voxels.
    """
    spacing = float(axes[0][1] - axes[0][0])
    location = [
        (c - a[0]) / spacing for c, a in zip(center_xyz[::-1], axes, strict=True)
    ]
    lower = np.floor(location).astype(int)
    fraction = np.asarray(location) - lower
    for offset in np.ndindex(2, 2, 2):
        weight = np.prod(np.where(offset, fraction, 1 - fraction))
        density[tuple(lower + offset)] += photons * weight / spacing**3


def siemens_star(
    axes_zyx: Sequence[np.ndarray],
    radius_um: float,
    thickness_um: float,
    cycles: int,
    center_radius_fraction: float,
) -> np.ndarray:
    """MATLAB's sinusoidal Siemens target as a thin fluorescent XY sheet.

    Siemens_rdnoise_plot.m uses r=256, cos(100*atan2(Y,X)), and a constant
    half-bright center of radius 30*sqrt(2) pixels. The pattern fills its square
    field, including the corners. Here r=256 fine samples at 0.05 um gives
    radius_um=12.8; the center radius retains the MATLAB ratio. Optical image
    formation and camera noise belong to the shared forward_model, not here.

    Parameters
    ----------
    axes_zyx : sequence of numpy.ndarray
        Cartesian ZYX grid axes in micrometers.
    radius_um : float
        Physical half-width used to define the target center.
    thickness_um : float
        Fluorescent sheet thickness in micrometers.
    cycles : int
        Number of angular sinusoidal cycles.
    center_radius_fraction : float
        Half-bright central disk radius divided by radius_um.

    Returns
    -------
    numpy.ndarray
        Nonnegative fluorescent-sheet density on the supplied grid.
    """
    if radius_um <= 0 or thickness_um <= 0 or cycles < 1 or int(cycles) != cycles:
        raise ValueError("Require positive radius/thickness and integer cycles")
    z, y, x = np.meshgrid(*axes_zyx, indexing="ij", sparse=True)
    pattern = 0.5 * (1 + np.cos(cycles * np.arctan2(y, x)))
    pattern = np.where(
        np.hypot(y, x) <= radius_um * center_radius_fraction, 0.5, pattern
    )
    return (pattern * (np.abs(z) < thickness_um / 2)).astype(np.float32)
