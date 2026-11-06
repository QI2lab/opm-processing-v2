"""Independent camera probabilities for physical validation, in recorded ADU.

Photon arrival is Poisson; electron read noise is Gaussian; the ADC rounds and
saturates. Likelihoods retain the offset and include both censored edge codes.
They are never evaluated on clipped, dark-subtracted pseudo-photon counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.special import gammaln, kl_div, log_ndtr, logsumexp, ndtri, xlogy
from scipy.stats import poisson

from opm_processing.imageprocessing.camera import camera_correct

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from tests.simulation.camera import Camera
    from tests.simulation.forward_model import Microscope


@dataclass(frozen=True)
class SamplingEnvelope:
    """Certified camera-rate interval from known means and original detector noise."""

    lower: np.ndarray
    upper: np.ndarray
    roundoff: np.ndarray

    def __post_init__(self) -> None:
        """Keep shared numerical bounds immutable across individual checks."""
        for array in (self.lower, self.upper, self.roundoff):
            array.setflags(write=False)


@dataclass(frozen=True)
class SamplingAccuracy:
    """Absolute camera-likelihood and photon-flux checks on one fixed aperture."""

    inside_envelope: bool
    deviance: float
    deviance_bound: float
    flux: float
    flux_lower: float
    flux_upper: float

    @property
    def accepted(self) -> bool:
        """Require local, likelihood and total-photon bounds simultaneously."""
        return (
            self.inside_envelope
            and np.isfinite(self.deviance)
            and np.isfinite(self.deviance_bound)
            and self.deviance <= self.deviance_bound
            and self.flux_lower <= self.flux <= self.flux_upper
        )


def sampling_stencil_radii(
    microscope: Microscope,
    output_spacing_zyx: Sequence[float],
    *,
    z_bin_factor: int = 1,
) -> tuple[int, int, int]:
    """Bound contributing native camera pixels by physical geometry alone.

    An inverse camera sample lies within one orthogonal output-cell pitch.
    A Z bin adds half its native-plane width. Perpendicular projection onto
    either adjacent acquired plane adds at most one scan interval and one
    detector-row interval. These bounds contain every contributing sample;
    they do not reproduce interpolation indices or weights.

    Parameters
    ----------
    microscope : Microscope
        Recorded camera pitch, scan spacing and OPM angle.
    output_spacing_zyx : sequence of float
        Physical spacing of the saved orthogonal image.
    z_bin_factor : int
        Number of native Z planes averaged into an output bin.

    Returns
    -------
    tuple of int
        Conservative native SYX radii for installed min/max filter operators.
    """
    spacing = np.asarray(output_spacing_zyx, np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0) or not np.isfinite(spacing).all():
        raise ValueError("Require three positive orthogonal spacings")
    if z_bin_factor < 1 or int(z_bin_factor) != z_bin_factor:
        raise ValueError("Require a positive integer Z-bin factor")
    pixel = microscope.pixel_size_um
    step = microscope.scan_step_um
    angle = np.deg2rad(microscope.angle_deg)
    z_spread = spacing[0] + (z_bin_factor - 1) * pixel / 2
    return (
        int(np.ceil((spacing[1] + z_spread / np.tan(angle)) / step)),
        int(np.ceil(z_spread / (pixel * np.sin(angle)) + step / pixel * np.cos(angle))),
        int(np.ceil(spacing[2] / pixel)),
    )


def sampling_stencil_footprint(
    microscope: Microscope,
    output_spacing_zyx: Sequence[float],
    *,
    z_bin_factor: int = 1,
    origin_y_error_um: float = 0,
) -> np.ndarray:
    """Exclude physically unreachable corners of the native-camera stencil.

    A perpendicular projection onto an adjacent scan plane changes laboratory
    Y by at most ``step*sin(theta)**2``. Detector-row interpolation adds at
    most ``pixel*cos(theta)`` and inverse camera sampling one output Y pitch.
    Native Z averaging leaves Y fixed. This necessary geometric condition
    retains every positive contributor without reproducing interpolation
    indices or weights.

    Parameters
    ----------
    microscope : Microscope
        Recorded camera pitch, scan spacing and OPM angle.
    output_spacing_zyx : sequence of float
        Saved-grid spacing, including any extra fractional-fusion stencil.
    z_bin_factor : int
        Native planes per output Z bin.
    origin_y_error_um : float
        Independently declared Y-origin serialization error bound.

    Returns
    -------
    numpy.ndarray
        Immutable centered SYX footprint for installed min/max filters.
    """
    radii = sampling_stencil_radii(
        microscope, output_spacing_zyx, z_bin_factor=z_bin_factor
    )
    if not np.isfinite(origin_y_error_um) or origin_y_error_um < 0:
        raise ValueError("Require a finite nonnegative coordinate-error bound")
    pixel, step = microscope.pixel_size_um, microscope.scan_step_um
    angle = np.deg2rad(microscope.angle_deg)
    offsets = np.meshgrid(
        *(np.arange(-radius, radius + 1) for radius in radii), indexing="ij"
    )
    distance_y = offsets[0] * step + offsets[1] * pixel * np.cos(angle)
    reach_y = (
        float(output_spacing_zyx[1])
        + step * np.sin(angle) ** 2
        + pixel * abs(np.cos(angle))
        + origin_y_error_um
    )
    arithmetic = 32 * np.finfo(float).eps * max(reach_y, step, pixel)
    support = np.abs(distance_y) <= reach_y + arithmetic
    support.setflags(write=False)
    return support


def camera_sampling_envelope(
    expectation: ArrayLike,
    raw: ArrayLike,
    camera: Camera,
    *,
    radii: Sequence[int],
    footprint: ArrayLike | None = None,
    response: ArrayLike = 1,
    views: int = 1,
    z_bin_factor: int = 1,
) -> SamplingEnvelope:
    """Bound positive resampling using original optical truth and detector residuals.

    Write each calibrated input as ``mu/response + residual/response``.
    Positive normalized interpolation and fusion weights keep each component
    inside its contributing-pixel extrema. Installed SciPy min/max filters
    evaluate those extrema over a conservative physical stencil. The measured
    residual contains the actual Poisson, read-noise, ADC and clipping effects;
    it is determined before processing, never fitted to its output. This also
    remains valid when a fitted flatfield depends on the observations.

    Parameters
    ----------
    expectation : array-like
        Original noiseless camera photoelectron means, before digitization.
    raw : array-like
        Original unsigned camera observations on the same grid.
    camera : Camera
        Explicit detector gain and offset.
    radii : sequence of int
        Native-camera stencil radii supplied by physical geometry.
    footprint : array-like or None
        Optional correlated physical support within the rectangular radii.
    response : array-like
        Positive response divided out before image interpolation.
    views : int
        Maximum number of independently acquired views mixed by fusion.
    z_bin_factor : int
        Native planes per output bin, used to bound arithmetic roundoff.

    Returns
    -------
    SamplingEnvelope
        Immutable lower/upper camera-rate bounds and float32 arithmetic margin.
    """
    means, codes = np.broadcast_arrays(
        np.asarray(expectation, np.float64), np.asarray(raw, np.float64)
    )
    sensitivity = np.broadcast_to(np.asarray(response, np.float64), means.shape)
    radius = np.asarray(radii)
    if (
        radius.shape != (means.ndim,)
        or np.any(radius < 0)
        or np.any(radius != radius.astype(int))
        or not np.isfinite(means).all()
        or np.any(means < 0)
        or not np.isfinite(codes).all()
        or not np.isfinite(sensitivity).all()
        or np.any(sensitivity <= 0)
        or views < 1
        or z_bin_factor < 1
    ):
        raise ValueError(
            "Require finite original camera data and positive sampling settings"
        )
    observed = camera_correct(
        codes, camera.offset_adu, camera.electrons_per_adu
    ).astype(np.float64)
    signal = means / sensitivity
    residual = (observed - means) / sensitivity
    window = tuple(2 * radius.astype(int) + 1)
    support = None if footprint is None else np.asarray(footprint, bool)
    if support is not None and (
        support.shape != window or not support[tuple(radius.astype(int))]
    ):
        raise ValueError("Require a centered physical footprint within the stencil")
    filters = {"size": window} if support is None else {"footprint": support}
    lower = (
        minimum_filter(signal, mode="nearest", **filters)
        + minimum_filter(residual, mode="nearest", **filters)
    ) * sensitivity
    upper = (
        maximum_filter(signal, mode="nearest", **filters)
        + maximum_filter(residual, mode="nearest", **filters)
    ) * sensitivity
    # Per view: calibration, four-sample orthogonal interpolation, eight-sample
    # inverse camera interpolation and normalized fusion take fewer than 64
    # positive arithmetic operations. Each extra native Z plane adds eight.
    operations = (64 + 8 * (z_bin_factor - 1)) * (views + 1)
    error = operations * np.finfo(np.float32).eps
    if error >= 1:
        raise ValueError("Sampling arithmetic exceeds the supported precision bound")
    roundoff = (
        error / (1 - error) * np.maximum.reduce((means, np.abs(lower), np.abs(upper)))
    )
    return SamplingEnvelope(
        np.maximum(lower - roundoff, 0),
        np.maximum(upper + roundoff, 0),
        roundoff,
    )


def camera_sampling_accuracy(
    prediction: ArrayLike,
    expectation: ArrayLike,
    envelope: SamplingEnvelope,
    mask: ArrayLike,
) -> SamplingAccuracy:
    """Compare absolute camera rates with certified likelihood and flux bounds.

    Poisson KL is convex in its predicted rate, so its maximum over a known
    interval occurs at an endpoint. A local interval check remains mandatory
    even when an endpoint is zero and gives an infinite KL bound. Fixed spatial
    apertures keep opposing stripe errors from cancelling in total photon flux.

    Parameters
    ----------
    prediction : array-like
        Camera-rate prediction sampled from the actual saved reconstruction.
    expectation : array-like
        Original simulated noiseless camera means.
    envelope : SamplingEnvelope
        Bounds derived solely from original truth and observations.
    mask : array-like
        Fixed physical source or background aperture on the camera grid.

    Returns
    -------
    SamplingAccuracy
        Joint local, absolute-likelihood and integrated-flux result.
    """
    predicted, expected = np.broadcast_arrays(
        np.asarray(prediction, np.float64), np.asarray(expectation, np.float64)
    )
    selected = np.broadcast_to(np.asarray(mask, bool), expected.shape)
    if not selected.any() or expected[selected].sum() <= 0:
        raise ValueError("Accuracy requires a nonempty positive known-camera aperture")
    low, high = envelope.lower[selected], envelope.upper[selected]
    rates, actual = expected[selected], predicted[selected]
    inside = bool(
        np.isfinite(actual).all() and np.all(actual >= low) and np.all(actual <= high)
    )
    divergence = float(2 * kl_div(rates, actual).sum(dtype=np.float64))
    bound = float(
        2 * np.maximum(kl_div(rates, low), kl_div(rates, high)).sum(dtype=np.float64)
    )
    return SamplingAccuracy(
        inside,
        divergence,
        bound,
        float(actual.sum(dtype=np.float64)),
        float(low.sum(dtype=np.float64)),
        float(high.sum(dtype=np.float64)),
    )


def camera_log_likelihood(
    raw: ArrayLike,
    expectation: ArrayLike,
    camera: Camera,
    *,
    tail_probability: float = 1e-13,
) -> np.ndarray:
    """Evaluate log P(ADC=raw | expected photoelectrons), pixel by pixel.

    For code a, sum Pois(n|lambda) times the Gaussian probability in its
    quantization interval [(a-.5-offset)g, (a+.5-offset)g]. Code zero has an
    infinite lower interval; the maximum code has an infinite upper interval.
    The only approximation is a Poisson tail of total probability <= the
    specified bound. Summation in log space preserves rare-event likelihoods.

    Parameters
    ----------
    raw : array-like
        Unsigned digitized ADC values.
    expectation : array-like
        Expected detected photoelectrons before read noise and digitization.
    camera : Camera
        Gain, offset, read-noise RMS and ADC depth.
    tail_probability : float
        Maximum omitted Poisson probability in the latent-count sum.

    Returns
    -------
    numpy.ndarray
        Pixelwise log probabilities of the observed ADC values.
    """
    codes, rates = np.broadcast_arrays(
        np.asarray(raw), np.asarray(expectation, np.float64)
    )
    maximum = 2**camera.bit_depth - 1
    if (
        np.any(codes != np.rint(codes))
        or np.any(codes < 0)
        or np.any(codes > maximum)
        or not np.isfinite(rates).all()
        or np.any(rates < 0)
        or camera.read_noise_e < 0
        or camera.electrons_per_adu <= 0
        or not 0 < tail_probability < 1
    ):
        raise ValueError(
            "Require recorded ADC codes, nonnegative rates and valid camera parameters"
        )
    result = np.empty(codes.size, dtype=np.float64)
    # Detector intervals depend on the recorded code, not the predicted rate.
    # Group similar rate/code scales to reuse probabilities. Each group's
    # Poisson upper limit uses its largest rate, so the SAME absolute tail bound
    # holds for every pixel by Poisson stochastic ordering. Every sum still
    # starts at zero, retaining rare low-code events even under a bright model.
    # Include the recorded interval even under a dim prediction: a rare bright
    # code can require n far beyond that prediction's Poisson upper quantile.
    adc_upper = np.maximum(
        (codes.ravel().astype(np.float64) + 0.5 - camera.offset_adu)
        * camera.electrons_per_adu,
        0,
    )
    exponents = np.frexp(np.maximum(rates.ravel(), adc_upper))[1]
    order = np.lexsort((codes.ravel(), exponents))
    boundaries = np.flatnonzero(np.diff(exponents[order])) + 1
    for group in np.split(order, boundaries):
        stop = (
            int(poisson.isf(tail_probability, float(np.max(rates.ravel()[group])))) + 1
        )
        stop = max(
            stop,
            int(
                np.ceil(
                    np.max(adc_upper[group])
                    - camera.read_noise_e * ndtri(tail_probability)
                )
            ),
        )
        electrons = np.arange(stop + 1, dtype=np.float64)
        log_factorial = gammaln(electrons + 1)
        batch = max(1, 262144 // electrons.size)
        for start in range(0, group.size, batch):
            indices = group[start : start + batch]
            adc, inverse = np.unique(codes.ravel()[indices], return_inverse=True)
            adc = adc[:, None]
            rate = rates.ravel()[indices, None]
            if camera.read_noise_e == 0:
                digitized = np.clip(
                    np.rint(camera.offset_adu + electrons / camera.electrons_per_adu),
                    0,
                    maximum,
                )
                log_interval = np.where(adc == digitized, 0, -np.inf)
            else:
                lower = (adc - 0.5 - camera.offset_adu) * camera.electrons_per_adu
                upper = (adc + 0.5 - camera.offset_adu) * camera.electrons_per_adu
                lower = np.where(adc == 0, -np.inf, lower)
                upper = np.where(adc == maximum, np.inf, upper)
                lo = (lower - electrons) / camera.read_noise_e
                hi = (upper - electrons) / camera.read_noise_e
                # Reflect positive intervals to avoid subtracting CDFs near one.
                log_hi = log_ndtr(np.where(lo > 0, -lo, hi))
                log_lo = log_ndtr(np.where(lo > 0, -hi, lo))
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_interval = log_hi + np.log(-np.expm1(log_lo - log_hi))
            log_poisson = xlogy(electrons, rate) - rate - log_factorial
            result[indices] = logsumexp(log_poisson + log_interval[inverse], axis=1)
    return result.reshape(codes.shape)
