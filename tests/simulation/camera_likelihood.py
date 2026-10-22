"""Independent camera probabilities for physical validation, in recorded ADU.

Photon arrival is Poisson; electron read noise is Gaussian; the ADC rounds and
saturates. Likelihoods retain the offset and include both censored edge codes.
They are never evaluated on clipped, dark-subtracted pseudo-photon counts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln, log_ndtr, logsumexp, ndtri, xlogy
from scipy.stats import poisson

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from tests.simulation.camera import Camera


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
