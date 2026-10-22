"""Shared physical specimens, acquisition grids and fixed analysis apertures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tests.simulation.forward_model import (
    Microscope,
    emission_psf,
)
from tests.simulation.objects import (
    centered_axis,
    deposit_point,
    meridian_sphere,
    siemens_star,
)


@dataclass
class OpticalScene:
    """Known fine object and optical calibration for reconstruction comparisons."""

    kind: str
    microscope: Microscope
    axes: tuple[np.ndarray, ...]
    density: np.ndarray
    psf_axes: tuple[np.ndarray, ...]
    psf_irradiance: np.ndarray
    illumination_profile: str
    illumination_half_width_um: float
    features: dict


def prepare_scene(
    kind: str,
    microscope: Microscope,
    *,
    parameters: dict,
    illumination_profile: str = "uniform",
) -> OpticalScene:
    """Build fine-grid points, paired lines, a hollow sphere or the MATLAB Siemens target.

    Parameters
    ----------
    kind : str
        Points, paired lines, meridian sphere or Siemens star.
    microscope : Microscope
        Explicit optical and acquisition calibration.
    parameters : dict
        Specimen extent, sampling and feature dimensions.
    illumination_profile : str
        Uniform illumination or a positive long-axis edge taper.

    Returns
    -------
    OpticalScene
        Fine-grid density, physical axes and shared optical calibration.
    """
    spacing = microscope.fine_spacing_um
    if spacing <= 0 or spacing >= microscope.pixel_size_um:
        raise ValueError("Fine grid must be positive and finer than sensor pixels")
    half_extent = parameters["half_extent_zyx_um"]
    features = dict(parameters["features"])
    if kind == "line-pairs":
        gaps = np.asarray(features["gaps_um"])
        starts = np.r_[0, np.cumsum(gaps[:-1] + features.pop("pair_gap_um"))]
        positions = np.column_stack((starts, starts + gaps)).ravel()
        positions -= (positions[0] + positions[-1]) / 2
        features["positions_x_um"] = positions.tolist()
        if kind == "line-pairs":
            features["pair_centers_x_um"] = (
                positions.reshape(-1, 2).mean(axis=1).tolist()
            )
    elif kind not in ("points", "sphere", "siemens-star"):
        raise ValueError("Unknown specimen kind")
    axes = tuple(centered_axis(extent, spacing) for extent in half_extent)
    density = np.zeros(tuple(len(a) for a in axes), np.float32)
    z, y, x = np.meshgrid(*axes, indexing="ij", sparse=True)
    if kind == "points":
        for center in features["centers_xyz_um"]:
            deposit_point(density, axes, center, 1.0)
    elif kind == "line-pairs":
        # Volume-average the finite cylinders before fixing their known mass.
        for center in features["positions_x_um"]:
            line = np.zeros_like(density)
            samples = features["voxel_samples"]
            if not isinstance(samples, int) or samples < 1:
                raise ValueError("Specimen voxel_samples must be a positive integer")
            for dz, dy, dx in np.ndindex(samples, samples, samples):
                line += (
                    (z + (dz - (samples - 1) / 2) * spacing / samples) ** 2
                    + (x + (dx - (samples - 1) / 2) * spacing / samples - center) ** 2
                    <= (features["diameter_um"] / 2) ** 2
                ) & (
                    np.abs(y + (dy - (samples - 1) / 2) * spacing / samples)
                    <= features["length_um"] / 2
                )
            line *= features["length_um"] / (line.sum(dtype=np.float64) * spacing**3)
            density += line
    elif kind == "siemens-star":
        density = siemens_star(axes, **features)
    else:
        density = meridian_sphere(axes, **features)
    psf_axes, psf = emission_psf(microscope, spacing)
    half_width = (
        np.ceil(axes[2][-1] / microscope.pixel_size_um) * microscope.pixel_size_um
    )
    return OpticalScene(
        kind,
        microscope,
        axes,
        density,
        psf_axes,
        psf,
        illumination_profile,
        half_width,
        features,
    )


def acquisition_shape(scene: OpticalScene) -> tuple[int, int, int]:
    """Cover the full optical field with shared endpoints for factors 1, 3 and 4.

    Parameters
    ----------
    scene : OpticalScene
        Specimen support and declared acquisition sampling.

    Returns
    -------
    tuple[int, int, int]
        Scan-YX dimensions covering the scene with aligned scan endpoints.
    """
    p = scene.microscope
    theta = np.deg2rad(p.angle_deg)
    hz, hy, hx = (axis[-1] for axis in scene.axes)
    if p.scan_endpoint_divisor < 2 or p.scan_endpoint_divisor % 2:
        raise ValueError("Scan endpoint divisor must be a positive even integer")
    half_divisor = p.scan_endpoint_divisor // 2
    scan_radius = half_divisor * int(
        np.ceil((hy + hz / np.tan(theta)) / p.scan_step_um / half_divisor)
    )
    row_radius = int(np.ceil(hz / (p.pixel_size_um * np.sin(theta))))
    col_radius = int(np.ceil(hx / p.pixel_size_um))
    return 2 * scan_radius + 1, 2 * row_radius + 1, 2 * col_radius + 1
