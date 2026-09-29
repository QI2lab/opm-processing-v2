"""Independent vectorial-diffraction point emitters in laboratory microns."""

from functools import lru_cache

import numpy as np
import psfmodels
from scipy.interpolate import RegularGridInterpolator


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
    for node, weight in zip(nodes, weights / 2):
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
    region = sum((coord - c) ** 2 for coord, c in zip(xyz, center)) <= radius**2
    mass = np.maximum(volume, 0).astype(np.float64) * region
    mass /= mass.sum()
    centroid = np.array([np.sum(mass * coord) for coord in xyz])
    covariance = np.array(
        [
            [np.sum(mass * (a - ca) * (b - cb)) for b, cb in zip(xyz, centroid)]
            for a, ca in zip(xyz, centroid)
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
