"""Likelihood and resolution measurements for simulated camera acquisitions."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from scipy.special import j1, jn_zeros, kl_div, xlogy
from scipy.stats import chi2

from opm_processing.imageprocessing.opmpsf import (
    camera_grid_center,
    centered_sensor_axes,
    get_skewed_coords,
)


@dataclass(frozen=True)
class CameraNoiseAccuracy:
    """Absolute camera-space error relative to a declared high-count noise target."""

    deviance: float
    deviance_limit: float
    pearson: float
    pearson_limit: float

    @property
    def accepted(self):
        """Require finite pointwise Poisson and calibrated-noise accuracy scores."""
        return (
            np.isfinite(
                (self.deviance, self.deviance_limit, self.pearson, self.pearson_limit)
            ).all()
            and self.deviance <= self.deviance_limit
            and self.pearson <= self.pearson_limit
        )


def camera_noise_accuracy(prediction, expectation, camera, mask):
    """Judge camera-prediction error against a predeclared measurement-noise target.

    This is a metrology target motivated by one independent high-count camera
    observation, not a coverage theorem for nonlinear reconstructed pixels.
    The reference chi-square has one degree of freedom per original camera
    sample. Its variance includes Poisson, read and ADC quantization terms.
    The fixed upper-tail probability is 1e-6.
    No gain, covariance or degrees of freedom are fitted to the reconstruction.

    Parameters
    ----------
    prediction : array-like
        Actual reconstruction's forward-predicted detector electron rates.
    expectation : array-like
        Original noiseless optical camera rates.
    camera : Camera
        Independently declared detector read noise and electron/ADU calibration.
    mask : array-like
        Fixed physical aperture, selected without using reconstructed values.

    Returns
    -------
    CameraNoiseAccuracy
        Absolute likelihood and calibrated-noise residual scores and limits.
    """
    predicted, expected = np.broadcast_arrays(
        np.asarray(prediction, np.float64), np.asarray(expectation, np.float64)
    )
    selected = np.broadcast_to(np.asarray(mask, bool), expected.shape)
    if (
        not selected.any()
        or not np.isfinite(expected[selected]).all()
        or np.min(expected[selected]) < 1000
        or not np.isfinite((camera.read_noise_e, camera.electrons_per_adu)).all()
        or camera.read_noise_e < 0
        or camera.electrons_per_adu <= 0
    ):
        raise ValueError("Require a nonempty fixed high-count camera aperture")
    truth, actual = expected[selected], predicted[selected]
    if not np.isfinite(actual).all() or np.any(actual < 0):
        return CameraNoiseAccuracy(np.inf, 0, np.inf, 0)
    variance = truth + camera.read_noise_e**2 + camera.electrons_per_adu**2 / 12
    limit = float(chi2.isf(1e-6, truth.size))
    return CameraNoiseAccuracy(
        float(2 * kl_div(truth, actual).sum(dtype=np.float64)),
        limit * float(np.max(variance / truth)),
        float(np.sum((actual - truth) ** 2 / variance, dtype=np.float64)),
        limit,
    )


def camera_flux_accuracy(prediction, expectation, camera, mask):
    """Require absolute fixed-aperture photons within six declared noise sigmas.

    The count target is fixed before reconstruction. Its variance is the known
    Poisson total plus declared read/ADC variance; ideal integer-photon cameras
    have no ADC rounding error. This is a metrology target, not a claim about
    the distribution of nonlinear inverse-filtered pixels.
    """
    actual, expected = np.broadcast_arrays(
        np.asarray(prediction, np.float64), np.asarray(expectation, np.float64)
    )
    selected = np.broadcast_to(np.asarray(mask, bool), expected.shape)
    if (
        not selected.any()
        or not np.isfinite(expected[selected]).all()
        or np.any(expected[selected] < 0)
        or not np.isfinite((camera.read_noise_e, camera.electrons_per_adu)).all()
        or camera.read_noise_e < 0
        or camera.electrons_per_adu <= 0
    ):
        raise ValueError("Require a nonempty physical photon aperture and camera")
    integral_codes = (
        camera.read_noise_e == 0
        and float(camera.offset_adu).is_integer()
        and float(1 / camera.electrons_per_adu).is_integer()
    )
    quantization = 0 if integral_codes else camera.electrons_per_adu**2 / 12
    target = expected[selected].sum(dtype=np.float64)
    variance = target + selected.sum() * (camera.read_noise_e**2 + quantization)
    return bool(
        np.isfinite(actual[selected]).all()
        and np.all(actual[selected] >= 0)
        and abs(actual[selected].sum(dtype=np.float64) - target)
        <= 6 * np.sqrt(variance)
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


def camera_score_difference(raw, prediction, baseline, camera):
    """Score independent camera observations without a brightness-dependent grid.

    Ideal integer observations use the exact Poisson log ratio; its factorial
    cancels. Gaussian/ADC observations use an independent CPU summation of
    the physical detector probabilities. This is a validation metric only.

    Parameters
    ----------
    raw : array-like
        Independent digitized camera observations.
    prediction : array-like
        Predicted detected photoelectrons before digitization.
    baseline : array-like
        Reference prediction in the same units.
    camera : Camera
        Explicit physical camera calibration.

    Returns
    -------
    numpy.ndarray
        Pixelwise log-likelihood advantage over the baseline.
    """
    if (
        camera.read_noise_e == 0
        and camera.electrons_per_adu == 1
        and camera.offset_adu == 0
        and np.all(np.asarray(raw) < 2**camera.bit_depth - 1)
    ):
        return xlogy(raw, prediction) - xlogy(raw, baseline) - prediction + baseline
    from tests.simulation.camera_likelihood import camera_log_likelihood

    return camera_log_likelihood(raw, prediction, camera) - camera_log_likelihood(
        raw, baseline, camera
    )


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
    from dataclasses import replace

    from tests.simulation.forward_model import sensor_integral

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
    return sensor_integral(
        volume,
        axes,
        shape,
        replace(microscope, pixel_samples=1),
        sensor_axes=native_axes,
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


def reconstruction_storage_accuracy(
    prediction, reconstructions, coverage, microscope, spacing, mask
):
    """Certify positive storage/resampling relative to independently scored inverses.

    This numerical consistency bound never widens the separate original-photon
    accuracy target. All views must share the same independently verified grid.
    Fixed physical apertures require complete geometric coverage; local extrema
    over the physical contributing stencil bound every positive interpolation.
    """
    from tests.simulation.camera_likelihood import sampling_stencil_radii

    values = np.asarray(reconstructions, np.float64)
    if values.ndim != 4 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Require actual nonnegative reconstructed camera volumes")
    radii = sampling_stencil_radii(microscope, spacing)
    window = (1, *(2 * radius + 1 for radius in radii))
    lower = ndimage.minimum_filter(values, size=window, mode="nearest").min(axis=0)
    upper = ndimage.maximum_filter(values, size=window, mode="nearest").max(axis=0)
    error = 64 * (len(values) + 1) * np.finfo(np.float32).eps
    numerical = error / (1 - error) * np.maximum(upper, 1)
    selected = np.asarray(mask) & np.isclose(
        coverage, 1, rtol=0, atol=64 * np.finfo(np.float32).eps
    )
    if not selected.any():
        raise ValueError("Require a nonempty completely acquired physical aperture")
    actual = np.asarray(prediction)[selected]
    minimum, maximum = (lower - numerical)[selected], (upper + numerical)[selected]
    return bool(
        np.isfinite(actual).all()
        and np.isfinite(minimum).all()
        and np.isfinite(maximum).all()
        and np.all(minimum <= actual)
        and np.all(actual <= maximum)
    )


def camera_prediction(density, psf, *, boundary="constant", response=1):
    """Apply the independent installed SciPy optical camera predictor.

    Parameters
    ----------
    density : array-like
        Reconstructed object-grid intensities.
    psf : numpy.ndarray
        Centered optical kernel on the object grid.
    boundary : str
        SciPy boundary mode; constant uses zero-extended FFT convolution.
    response : array-like or float
        Detector response applied after blur.

    Returns
    -------
    numpy.ndarray
        Detector photon rates, including the original measured sensitivity.
    """
    if boundary == "constant":
        blurred = signal.fftconvolve(density, psf, mode="same")
    elif boundary == "reflect":
        # Symmetric extension is SciPy ndimage's reflect boundary. Including
        # the full kernel halo makes valid FFT convolution exactly equivalent,
        # without the direct convolution's per-voxel full-PSF work.
        pads = tuple(((size - 1) // 2, size // 2) for size in psf.shape)
        blurred = signal.fftconvolve(
            np.pad(density, pads, mode="symmetric"), psf, mode="valid"
        )
    else:
        blurred = ndimage.convolve(density, psf, mode=boundary)
    return np.maximum(blurred * response, np.finfo(float).tiny)


def predictive_deviance(
    density, expectation, psf, *, boundary="constant", response=1, selection=None
):
    """Score absolute expected camera counts, including background, without a fit.

    Finite float32 kernels and FFT roundoff may create numerically zero tails.
    The lower bound is float64's smallest positive normal, not a fitted floor.

    Parameters
    ----------
    density : array-like
        Reconstructed object-grid intensities.
    expectation : array-like
        Independently simulated expected detector counts.
    psf : numpy.ndarray
        Centered optical kernel on the object grid.
    boundary : str
        SciPy convolution boundary convention.
    response : array-like or float
        Detector response applied after blur.
    selection : array-like or slice or None
        Optional original-camera region scored after complete optical blur.

    Returns
    -------
    float
        Summed absolute Poisson deviance of the detector prediction.
    """
    prediction = camera_prediction(density, psf, boundary=boundary, response=response)
    divergence = kl_div(expectation, prediction)
    if selection is not None:
        divergence = divergence[selection]
    return float(2 * divergence.sum(dtype=np.float64))


def first_minimum(distance, intensity):
    """Return the first sampled minimum after a centered PSF's principal maximum.

    A profile without a minimum inside its modeled support has no measurable
    Rayleigh separation; it must not receive the lateral Airy value by default.
    Samples must be ordered physical distances from the central maximum.

    Parameters
    ----------
    distance : array-like
        Increasing physical distances from the PSF center.
    intensity : array-like
        Single-emitter optical intensity along that direction.

    Returns
    -------
    float
        Distance to the first sampled local minimum.
    """
    distance, intensity = np.asarray(distance), np.asarray(intensity)
    if (
        distance.ndim != 1
        or intensity.shape != distance.shape
        or np.any(np.diff(distance) <= 0)
    ):
        raise ValueError("Require a sampled PSF along increasing physical distance")
    peaks, _ = find_peaks(-intensity)
    candidates = peaks[distance[peaks] > 0]
    if not candidates.size:
        raise ValueError("PSF support does not contain a directional first minimum")
    index = int(candidates[0])
    return float(distance[index])


def airy_rayleigh(wavelength_um, numerical_aperture):
    """Scalar circular-pupil reference: exact Bessel first zero and pair contrast.

    Parameters
    ----------
    wavelength_um : float
        Emission wavelength in micrometers.
    numerical_aperture : float
        Circular-pupil numerical aperture.

    Returns
    -------
    tuple[float, float]
        Rayleigh separation in micrometers and equal-pair valley contrast.
    """
    root = float(jn_zeros(1, 1)[0])
    separation = root * wavelength_um / (2 * np.pi * numerical_aperture)
    # Two equal incoherent points at separation r: midpoint 2 h(r/2), peak 1.
    dip = 1 - 2 * (2 * j1(root / 2) / (root / 2)) ** 2
    return separation, float(dip)


def two_point_profile(volume, center_syx, offset_syx, microscope):
    """Measure peaks and valley along known laboratory-coordinate emitter positions.

    Search within half a pair separation of each source. Four interpolated
    samples per grid interval add no optical information; intensities are
    never fitted. The profile alone does not define a resolution criterion.

    Parameters
    ----------
    volume : numpy.ndarray
        Intensity image in oblique scan-YX coordinates.
    center_syx : sequence of float
        Pair midpoint in array-index coordinates.
    offset_syx : sequence of float
        Index displacement from the midpoint to either emitter.
    microscope : Microscope
        Recorded pixel pitch, scan step and oblique angle.

    Returns
    -------
    dict
        Physical profile coordinates, source positions, peak spacing and valley dip.
    """
    offset = np.asarray(offset_syx, dtype=np.float64)
    theta = np.deg2rad(microscope.angle_deg)
    pixel_size = microscope.pixel_size_um
    displacement_xyz = np.array(
        (
            offset[2] * pixel_size,
            offset[0] * microscope.scan_step_um
            + offset[1] * pixel_size * np.cos(theta),
            offset[1] * pixel_size * np.sin(theta),
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
