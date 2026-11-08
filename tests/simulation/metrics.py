"""Likelihood and physical sampling measurements for simulated camera acquisitions."""

import numpy as np
from scipy.special import kl_div

from opm_processing.imageprocessing.opmpsf import (
    camera_grid_center,
    centered_sensor_axes,
    get_skewed_coords,
    integrate_camera_pixels,
)


def camera_partition_apertures(shape):
    """Provide whole-camera and fixed detector row/column photon-count regions."""
    masks = [np.ones(shape, bool)]
    for axis in (-2, -1):
        for index in range(shape[axis]):
            mask = np.zeros(shape, bool)
            selector = [slice(None)] * len(shape)
            selector[axis] = index
            mask[tuple(selector)] = True
            masks.append(mask)
    return tuple(masks)


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


def camera_object_apertures(
    ground_truth, microscope, shape, *, camera_offset_zyx_um=(0, 0, 0)
):
    """Select source and background camera apertures from original physical axes.

    The finite known object's PSF-eroded interior and Rayleigh-radius emitter
    neighborhoods are fixed independently of reconstructed values. Detector
    row/column subsets preserve sensitivity to broad compensating stripes.
    """
    pixel, step = microscope.pixel_size_um, microscope.scan_step_um
    angle = np.deg2rad(microscope.angle_deg)
    x, y, z = get_skewed_coords(shape, pixel, step, angle)
    center = camera_grid_center(shape, pixel, step, microscope.angle_deg)
    coordinates = tuple(
        coordinate + offset
        for coordinate, offset in zip(
            (
                np.broadcast_to(z - center[0], shape),
                np.broadcast_to(y - center[1], shape),
                np.broadcast_to(x - center[2], shape),
            ),
            camera_offset_zyx_um,
            strict=True,
        )
    )
    radius = 0.61 * microscope.wavelength_um / microscope.numerical_aperture
    interior = np.ones(shape, bool)
    for index, (coordinate, axis, halo) in enumerate(
        zip(
            coordinates,
            ground_truth.axes,
            microscope.psf_half_extent_zyx_um,
            strict=True,
        )
    ):
        if ground_truth.kind == "line-pairs" and index == 0:
            # These thin known cylinders have a finite background slab rather
            # than a full axial PSF halo. Score the original physical slab near
            # the cylinders, including their two-Rayleigh background region.
            extent = ground_truth.features["diameter_um"] / 2 + 2 * radius
            interior &= (coordinate >= max(axis[0], -extent)) & (
                coordinate <= min(axis[-1], extent)
            )
            continue
        interior &= (coordinate >= axis[0] + halo) & (coordinate <= axis[-1] - halo)
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
        points = ()
    else:
        points = np.asarray(ground_truth.features["centers_xyz_um"])[:, ::-1]
    for point in points:
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
    background = interior & (distance >= 2 * radius)
    if not source.any() or not background.any():
        raise ValueError("The physical camera field must acquire source and background")
    masks = [source, background]
    for partition in camera_partition_apertures(shape)[1:]:
        selected = partition & interior
        if selected.any():
            masks.append(selected)
    return tuple(masks)
