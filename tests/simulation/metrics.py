"""Likelihood and resolution measurements for simulated camera acquisitions."""

import numpy as np
from scipy import ndimage, signal
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from scipy.special import j1, jn_zeros, kl_div, xlogy


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


def predictive_deviance(density, expectation, psf, *, boundary="constant", response=1):
    """Score absolute expected camera counts, including background, without a fit.

    Finite float32 kernels and FFT roundoff may create numerically zero tails.
    The lower bound is float64's smallest positive *normal*, not a fitted floor.
    A positive ground-truth rate predicted as zero receives its full log penalty.

    Parameters
    ----------
    density : array-like
        Reconstructed object-grid intensities.
    expectation : array-like
        Independently simulated expected detector counts.
    psf : numpy.ndarray
        Centered optical kernel on the object grid.
    boundary : str
        SciPy boundary mode; constant uses zero-extended FFT convolution.
    response : array-like or float
        Detector response applied after blur.

    Returns
    -------
    float
        Summed absolute Poisson deviance of the detector prediction.
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
    prediction = np.maximum(blurred * response, np.finfo(float).tiny)
    return float(2 * kl_div(expectation, prediction).sum(dtype=np.float64))


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
