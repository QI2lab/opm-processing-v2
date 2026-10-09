"""Independent vectorial-diffraction point emitters in laboratory microns."""

from functools import lru_cache

import numpy as np
import psfmodels
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks


@lru_cache(maxsize=1)
def optical_field():
    """Evaluate a 637 nm, NA 1.35 silicone objective on a Cartesian grid."""
    spacing = 0.04
    xy = (np.arange(101) - 50) * spacing
    z = (np.arange(151) - 75) * spacing
    field = psfmodels.vectorial_psf(
        zv=z,
        nx=xy.size,
        dxy=spacing,
        pz=0,
        wvl=0.637,
        params={
            "NA": 1.35,
            "ni0": 1.4,
            "ni": 1.4,
            "ns": 1.38,
            "tg0": 170,
            "tg": 170,
            "ti0": 300,
        },
    )
    return RegularGridInterpolator((z, xy, xy), field, bounds_error=False, fill_value=0)


def sample_point_source(
    shape=(81, 97, 49),
    scan_step=0.2,
    pixel=0.115,
    angle=30,
    center_xyz=(0, 0, 0),
    velocity_xyz=(0, 0, 0),
    scan_speed=0.8 / 0.003,
    exposure=0.003,
):
    """Integrate a stationary/moving emitter over the exposure of each plane.

    Coordinates and time are evaluated independently of production deskew/PSF
    helpers. Motion is constant velocity in um/s; exposure is in seconds.
    The camera plane is held at its nominal position during exposure, isolating
    specimen motion from any unmodeled motion of the scanning actuator.
    """
    scan, row, col = np.meshgrid(
        (np.arange(shape[0]) - (shape[0] - 1) / 2) * scan_step,
        (np.arange(shape[1]) - (shape[1] - 1) / 2) * pixel,
        (np.arange(shape[2]) - (shape[2] - 1) / 2) * pixel,
        indexing="ij",
        sparse=True,
    )
    theta = np.deg2rad(angle)
    x, y, z = col, scan + row * np.cos(theta), row * np.sin(theta)
    times = scan / scan_speed
    nodes, weights = np.polynomial.legendre.leggauss(5)
    result = np.zeros(shape)
    for node, weight in zip(nodes, weights / 2, strict=False):
        time = times + node * exposure / 2
        coords = np.broadcast_arrays(
            z - center_xyz[2] - velocity_xyz[2] * time,
            y - center_xyz[1] - velocity_xyz[1] * time,
            x - center_xyz[0] - velocity_xyz[0] * time,
        )
        points = np.stack(coords, axis=-1)
        result += weight * optical_field()(points)
    return result.astype(np.float32)


def point_moments(volume, voxel_size, center, radius=1.3):
    """Measure XYZ covariance and YZ principal-axis tilt in a physical sphere."""
    z, y, x = np.ogrid[: volume.shape[0], : volume.shape[1], : volume.shape[2]]
    xyz = np.broadcast_arrays(x * voxel_size[2], y * voxel_size[1], z * voxel_size[0])
    region = (
        sum((coord - c) ** 2 for coord, c in zip(xyz, center, strict=False))
        <= radius**2
    )
    mass = np.maximum(volume, 0).astype(np.float64) * region
    mass /= mass.sum()
    centroid = np.array([np.sum(mass * coord) for coord in xyz])
    covariance = np.array(
        [
            [
                np.sum(mass * (a - ca) * (b - cb))
                for b, cb in zip(xyz, centroid, strict=False)
            ]
            for a, ca in zip(xyz, centroid, strict=False)
        ]
    )
    yz_tilt = np.rad2deg(
        0.5 * np.arctan2(2 * covariance[1, 2], covariance[2, 2] - covariance[1, 1])
    )
    return {
        "centroid_xyz": centroid.tolist(),
        "covariance_xyz": covariance.tolist(),
        "yz_tilt_deg": float(yz_tilt),
        "axis_misalignment_deg": float(min(abs(yz_tilt), 90 - abs(yz_tilt))),
        "yz_correlation": float(
            covariance[1, 2] / np.sqrt(covariance[1, 1] * covariance[2, 2])
        ),
    }


def skewed_coordinates(shape, scan_step=0.2, offset=(0, 0, 0)):
    """Map centered acquisition indices to laboratory XYZ in microns.

    Parameters
    ----------
    shape : tuple of int
        Scan, detector-row, and detector-column dimensions.
    scan_step : float
        Scan displacement per plane in micrometers, defaulting to 0.2.
    offset : tuple of float
        Subvoxel offsets along the three acquisition axes, defaulting to zero.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Broadcastable laboratory X, Y, Z coordinates for 115 nm camera pixels
        and a 30 degree OPM angle.
    """
    scan, row, col = np.meshgrid(
        *[np.arange(n) - (n - 1) / 2 + d for n, d in zip(shape, offset, strict=False)],
        indexing="ij",
        sparse=True,
    )
    angle = np.deg2rad(30)
    return (
        col * 0.115,
        scan * scan_step + row * 0.115 * np.cos(angle),
        row * 0.115 * np.sin(angle),
    )


def fluorescent_specimen(shape, scan_step):
    """Integrate beads and a finite filament on the requested oblique grid.

    Parameters
    ----------
    shape : tuple of int
        Scan, detector-row, and detector-column dimensions.
    scan_step : float
        Scan displacement per plane in micrometers.

    Returns
    -------
    tuple[numpy.ndarray, list[tuple[float, float, float]]]
        Float32 fluorescence sampled with 27 subvoxel quadrature points and
        bead centers in laboratory XYZ micrometers. Beads have 380 nm diameter
        and 10,000 fluorescence units; the 220 nm filament has 5,000 units.
    """
    truth = np.zeros(shape, dtype=np.float64)
    # (X,Y,Z) centers in um; different Y phases include missing scan planes.
    beads = [(-1.25, -3.1, -0.55), (1.0, 2.7, 0.65), (0.6, -0.65, -0.65)]
    start = np.array([-0.9, -1.6, 0.45])
    end = np.array([0.65, 1.7, -0.15])
    direction = end - start
    offsets = (-1 / 3, 0.0, 1 / 3)
    for ds in offsets:
        for dy in offsets:
            for dx in offsets:
                xyz = skewed_coordinates(
                    shape, scan_step=scan_step, offset=(ds, dy, dx)
                )
                for center in beads:
                    distance2 = sum(
                        (coord - c) ** 2 for coord, c in zip(xyz, center, strict=False)
                    )
                    truth += 10000 * (distance2 <= 0.19**2) / 27
                projection = sum(
                    (coord - s) * d
                    for coord, s, d in zip(xyz, start, direction, strict=False)
                )
                projection = np.clip(projection / np.dot(direction, direction), 0, 1)
                distance2 = sum(
                    (coord - (s + projection * d)) ** 2
                    for coord, s, d in zip(xyz, start, direction, strict=False)
                )
                truth += 5000 * (distance2 <= 0.11**2) / 27
    return truth.astype(np.float32), beads


def two_point_profile(volume, center_syx, offset_syx, scan_step=0.2):
    """Measure the line profile through two known acquisition-grid point positions.

    Parameters
    ----------
    volume : numpy.ndarray
        Fluorescence in scan, camera-row, camera-column order.
    center_syx : array_like
        Midpoint of the two emitters in acquisition indices.
    offset_syx : array_like
        Displacement from the midpoint to the positive emitter in index units.
        The other emitter is at the opposite displacement.
    scan_step : float, default=0.2
        Scan displacement in micrometers. Camera pixels are 115 nm and the
        OPM angle is 30 degrees, matching the shared specimen model.

    Returns
    -------
    dict
        Distances and linear-interpolated profile samples, known separation,
        measured peak positions and separation, and fractional valley depth
        relative to the weaker peak. Distances are laboratory micrometers
        relative to the pair midpoint. Peak measurements are None when two
        local maxima are absent from the respective emitter neighborhoods.
        Positive valley depth alone does not define a resolution criterion.

    Notes
    -----
    Search within half a pair separation around each known emitter and select
    the strongest local maximum in each neighborhood. Four profile samples
    per acquisition-grid interval interpolate the measured image; they do
    not provide additional optical information. Intensities are not fitted.
    """
    offset = np.asarray(offset_syx, dtype=np.float64)
    displacement_xyz = np.array(
        (
            offset[2] * 0.115,
            offset[0] * scan_step + offset[1] * 0.115 * np.cos(np.pi / 6),
            offset[1] * 0.115 * np.sin(np.pi / 6),
        )
    )
    separation = 2 * np.linalg.norm(displacement_xyz)
    positions = np.linspace(-2, 2, int(np.ceil(16 * np.max(np.abs(offset)))) + 1)
    distance = positions * separation / 2
    coordinates = np.asarray(center_syx)[:, None] + offset[:, None] * positions
    profile = map_coordinates(volume, coordinates, order=1, mode="constant", cval=0)
    peaks, _ = find_peaks(profile)
    selected = []
    for expected in (-separation / 2, separation / 2):
        candidates = peaks[np.abs(distance[peaks] - expected) < separation / 4]
        if candidates.size:
            selected.append(int(candidates[np.argmax(profile[candidates])]))
    result = {
        "distance_um": distance,
        "profile": profile,
        "true_separation_um": float(separation),
        "peak_positions_um": None,
        "peak_separation_um": None,
        "valley_depth": None,
    }
    if len(selected) == 2:
        left, right = selected
        weaker_peak = min(profile[left], profile[right])
        if weaker_peak > 0:
            result.update(
                peak_positions_um=distance[selected].tolist(),
                peak_separation_um=float(distance[right] - distance[left]),
                valley_depth=float(1 - profile[left : right + 1].min() / weaker_peak),
            )
    return result
