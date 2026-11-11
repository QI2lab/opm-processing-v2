"""Likelihood and physical sampling measurements for simulated camera acquisitions."""

import numpy as np
from scipy.special import kl_div

from opm_processing.imageprocessing.opmpsf import (
    camera_grid_center,
    centered_sensor_axes,
    get_skewed_coords,
    integrate_camera_pixels,
)


def poisson_log_ratio_moments(expectation, prediction):
    """Exact mean and variance of log p(N|lambda)/p(N|mu), N~Pois(lambda).

    Pixels are independent. Preserve impossible hypotheses and deterministic
    dark pixels without introducing a photon floor. SciPy evaluates the KL loss.

    Parameters
    ----------
    expectation : array-like
        True nonnegative photon rates for independent pixels.
    prediction : array-like
        Alternative rates on the same detector grid.

    Returns
    -------
    tuple[float, float]
        Expected total log-likelihood loss and its shot-noise variance.
    """
    truth, predicted = np.broadcast_arrays(
        np.asarray(expectation, dtype=np.float64),
        np.asarray(prediction, dtype=np.float64),
    )
    if (
        not np.isfinite(truth).all()
        or not np.isfinite(predicted).all()
        or np.any(truth < 0)
        or np.any(predicted < 0)
    ):
        raise ValueError("Photon rates must be finite and nonnegative")
    if np.any((truth > 0) & (predicted == 0)):
        return float("inf"), float("inf")
    positive = truth > 0
    rate, alternative = truth[positive], predicted[positive]
    log_ratio = np.log(rate) - np.log(alternative)
    return float(kl_div(truth, predicted).sum()), float(np.sum(rate * log_ratio**2))


def camera_volume_samples(
    volume,
    origin_zyx,
    spacing_zyx,
    frame_origin_zyx,
    native_shape,
    microscope,
):
    """Sample one stored laboratory volume on its original physical camera grid.

    This applies the existing sensor quadrature in point-sampling mode. It
    neither applies a second optical PSF nor reconstructs or normalizes an image.

    Parameters
    ----------
    volume : numpy.ndarray
        Three-dimensional laboratory-grid intensities or geometric coverage.
    origin_zyx : sequence of float
        Laboratory coordinate of the first stored voxel center.
    spacing_zyx : sequence of float
        Stored voxel-center spacing in micrometers.
    frame_origin_zyx : sequence of float
        Laboratory coordinate of the first canonical camera pixel.
    native_shape : sequence of int
        Original camera SYX dimensions.
    microscope : Microscope
        Physical scan step, detector pitch and oblique angle.

    Returns
    -------
    numpy.ndarray
        Intensities at the original camera centers in SYX order.
    """
    shape = tuple(native_shape)
    pixel, step = microscope.pixel_size_um, microscope.scan_step_um
    center = camera_grid_center(shape, pixel, step, microscope.angle_deg)
    reference = np.asarray(frame_origin_zyx) + center
    axes = tuple(
        float(origin) + np.arange(size) * float(spacing) - target
        for origin, spacing, size, target in zip(
            origin_zyx, spacing_zyx, volume.shape, reference, strict=True
        )
    )
    native_axes = centered_sensor_axes(shape, (step, pixel, pixel))
    return integrate_camera_pixels(
        volume,
        axes,
        native_axes,
        pixel_size_um=pixel,
        theta_deg=microscope.angle_deg,
        samples=1,
        area_integral=False,
    )


def camera_source_mask(ground_truth, microscope, shape):
    """Select the fixed source region from original physical axes.

    The finite known object's PSF-eroded interior and Rayleigh-radius emitter
    neighborhoods are fixed independently of reconstructed values.
    """
    from tests.simulation.forward_model import camera_psf

    pixel, step = microscope.pixel_size_um, microscope.scan_step_um
    angle = np.deg2rad(microscope.angle_deg)
    x, y, z = get_skewed_coords(shape, pixel, step, angle)
    center = camera_grid_center(shape, pixel, step, microscope.angle_deg)
    coordinates = tuple(
        coordinate - midpoint
        for coordinate, midpoint in zip((z, y, x), center, strict=True)
    )
    radius = 0.61 * microscope.wavelength_um / microscope.numerical_aperture
    interior = np.ones(shape, bool)
    for coordinate, axis, halo in zip(
        coordinates,
        ground_truth.axes,
        microscope.psf_half_extent_zyx_um,
        strict=True,
    ):
        interior &= (coordinate >= axis[0] + halo) & (coordinate <= axis[-1] - halo)
    # A valid physical reference also needs the full declared optical support
    # inside this observed camera ROI, not just inside the original specimen.
    kernel, _ = camera_psf(microscope)
    for axis, (length, support) in enumerate(zip(shape, kernel.shape, strict=True)):
        indices = np.arange(length)
        dimensions = [1] * len(shape)
        dimensions[axis] = length
        interior &= (
            (indices >= support // 2) & (indices < length - support // 2)
        ).reshape(dimensions)
    distance = np.full(shape, np.inf)
    if ground_truth.kind == "line-pairs":
        features = ground_truth.features
        interior &= np.abs(coordinates[1]) <= (
            features["length_um"] / 2 - microscope.psf_half_extent_zyx_um[1]
        )
        axial_distance = np.maximum(
            np.abs(coordinates[1]) - features["length_um"] / 2, 0
        )
        for center in features["positions_x_um"]:
            distance = np.minimum(
                distance,
                np.maximum(
                    np.sqrt(
                        coordinates[0] ** 2
                        + (coordinates[2] - center) ** 2
                        + axial_distance**2
                    )
                    - features["diameter_um"] / 2,
                    0,
                ),
            )
        points = features.get("centers_xyz_um", ())
    else:
        points = ground_truth.features["centers_xyz_um"]
    for point in np.asarray(points).reshape(-1, 3)[:, ::-1]:
        distance = np.minimum(
            distance,
            np.sqrt(
                sum(
                    (axis - center) ** 2
                    for axis, center in zip(coordinates, point, strict=True)
                )
            ),
        )
    source = interior & (distance <= radius)
    assert source.any(), "The physical camera field must acquire the source"
    return source
