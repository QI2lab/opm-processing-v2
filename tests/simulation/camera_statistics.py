"""Quantized camera moments from the existing exact Poisson/read-noise likelihood."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, poisson

from tests.simulation.camera import Camera
from tests.simulation.camera_likelihood import camera_log_likelihood


@dataclass(frozen=True)
class AdcMoments:
    """ADC moments and certified omitted-tail bounds, without simulating a camera."""

    mean: float
    variance: float
    fourth_moment_upper: float
    mean_error: float
    variance_error: float
    omitted_probability: float
    summation_roundoff: float
    codes: np.ndarray
    probabilities: np.ndarray


def adc_moments(rate: float, camera: Camera) -> AdcMoments:
    """Sum exact ADC probabilities on a compact interval and both censored edges.

    A Poisson upper quantile and two Gaussian tail quantiles exclude at most
    three tail probabilities. The likelihood's latent Poisson truncation adds
    one further tail probability. Omitted first and second moments use the
    finite ADC range. A gamma bound covers float64 summation of the retained
    probabilities; no truncated distribution is renormalized.

    The fourth central moment is bounded independently of the compact support:
    Poisson plus Gaussian noise has fourth moment lambda + 3(lambda+sigma^2)^2.
    Rounding adds at most half an ADU; censoring is nonexpansive about any
    center inside the ADC range. Minkowski's inequality gives the stated bound.
    """
    tail = 1e-14
    maximum = 2**camera.bit_depth - 1
    gain = camera.electrons_per_adu
    deviation = camera.read_noise_e * norm.isf(tail)
    lo = max(1, int(np.floor(camera.offset_adu - deviation / gain - 0.5)))
    hi = min(
        maximum - 1,
        int(
            np.ceil(
                camera.offset_adu + (poisson.isf(tail, rate) + deviation) / gain + 0.5
            )
        ),
    )
    codes = np.r_[0, np.arange(lo, hi + 1), maximum]
    probability = np.exp(
        camera_log_likelihood(codes, rate, camera, tail_probability=tail)
    )
    # Summation error is gamma_n times the sum of absolute terms. The stable
    # log-probability implementation evaluates each code through a logsumexp;
    # include the largest latent-electron sum as well as the final ADC sum.
    latent_terms = (
        int(np.ceil((maximum + 0.5 - camera.offset_adu) * gain + deviation)) + 2
    )
    operations = latent_terms + codes.size
    gamma = operations * np.finfo(np.float64).eps
    gamma /= 1 - gamma
    omitted = 4 * tail
    mean = float(probability @ codes)
    mean_error = maximum * omitted + gamma * max(1, mean)
    centered = codes - mean
    variance = float(probability @ centered**2)
    variance_error = (
        max(mean, maximum - mean) ** 2 * omitted
        + mean_error**2
        + gamma * max(1, variance)
    )
    fourth = (
        (rate + 3 * (rate + camera.read_noise_e**2) ** 2) ** 0.25 / gain
        + 0.5
        + abs(camera.offset_adu + rate / gain - mean)
        + mean_error
    ) ** 4
    codes.setflags(write=False)
    probability.setflags(write=False)
    return AdcMoments(
        mean,
        variance,
        fourth,
        mean_error,
        variance_error,
        omitted,
        gamma,
        codes,
        probability,
    )
