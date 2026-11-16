"""Richardson-Lucy reconstruction with gradient consensus.

The mathematical reference is Liu and Stallinga (2026), Biomedical Optics
Express 17, 2999-3015, https://doi.org/10.1364/BOE.597710 . Equations (3)/(11)
define the RL update, Eq. (4) smooths the product of additive object changes,
and Eq. (5) applies that update only for strictly positive consensus. Ordinary
RL and RLGC share the same full-data update; consensus only gates that update.

The Python notebook at
https://colab.research.google.com/drive/1mfVNSCaYHz1g56g92xBkIoa8190XNJpJ
is a secondary implementation reference. Its unweighted multiplier consensus
and zero-consensus gate differ from the paper and are not used here.

Camera-calibrated observations are clipped at zero. The shifted-count read-noise
approximation uses the full-data RL rule in Liu et al. (2025), supplementary
Eq. (12), https://doi.org/10.1038/s41467-025-56241-x . Split-noise compensation
is approximate; it is not an exact Poisson-Gaussian likelihood.

Native observations use symmetric padding and a free padded reconstruction
with periodic convolution. Sparse scans use zero extension and selection of
acquired planes, with its matching scatter/convolution adjoint. Detector response
uses sensitivity normalization. Ryomen schedules overlapping scan chunks with
independent initialization and random halves. Tiled KL processing continues all
chunks to the latest first-local-KL trigger, parking state in system RAM without
replaying updates. This shared budget is distinct from pooled whole-volume KL.

KL stopping follows the notebook: normalize the acquired camera prediction Ax
and each fresh binomial half, stop when both divergences increase, and restore
the preceding estimate. This measurement-space comparison applies to native
and refined scan grids. Accepted-fraction and relative-change limits apply when
KL is disabled. Empty observations return zero. MATLAB parity has not been verified.
"""

import gc
import logging
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType

import numpy as np
import psutil
from ryomen import Slicer
from tqdm import tqdm

from opm_processing.cuda import preload_cuda_libraries

preload_cuda_libraries()

import cupy as cp
from cupy import ElementwiseKernel
from cupyx.scipy.fft import irfftn
from cupyx.scipy.fft import next_fast_len as next_gpu_fft_size

# -----------------------------------------------------------------------------
# Equation (5): apply the ordinary RL step only for positive consensus.
# -----------------------------------------------------------------------------
filter_update = ElementwiseKernel(
    "float32 recon, float32 HTratio, float32 consensus_map",
    "float32 out",
    """
    bool skip = consensus_map <= 0;
    out = skip ? recon : recon * HTratio
    """,
    "filter_update",
)


Array = np.ndarray | cp.ndarray

RLGC_SOLVER_REVISION = "shared-kl-checkpoints-ryomen-v2"
GC_UPDATED_FRACTION_LIMIT = 0.001
GC_RELATIVE_CHANGE_LIMIT = 0.001


class _RlgcStagingError(MemoryError):
    """Host staging cannot fit; smaller GPU chunks would duplicate more halos."""


@dataclass
class _RlgcCheckpoint:
    """Park a chunk's continuation in RAM, retaining only its owned rollback cores.

    CuPy 14's Generator has no public host-state serialization. Preserve its
    actual XORWOW and binomial-cache bytes rather than reseeding or replacing
    its generator. All transfers finish before device storage is released.
    """

    retained_scan: slice
    estimate: np.ndarray | None = None
    rng: object = None
    rng_buffers: tuple[np.ndarray, np.ndarray] | None = None
    iterations: int = 0
    first_kl_stop: int | None = None
    accepted: np.ndarray | None = None
    provisional: np.ndarray | None = None

    def restore(self, xp: ModuleType) -> tuple[Array, object]:
        """Restore the same reconstruction and random stream on the backend."""
        if xp is cp:
            self.rng.bit_generator._state = cp.asarray(
                self.rng_buffers[0], blocking=True
            )
            self.rng._binomial_state = cp.asarray(self.rng_buffers[1], blocking=True)
            return cp.asarray(self.estimate, blocking=True), self.rng
        return self.estimate, self.rng

    def park(self, recon: Array, rng: object, completed: int) -> np.ndarray:
        """Offload all mutable device state after a completed iteration."""
        self.iterations = completed
        self.rng = rng
        if cp.get_array_module(recon) is cp:
            buffers = (rng.bit_generator._state, rng._binomial_state)
            if self.rng_buffers is None:
                self.rng_buffers = tuple(cp.asnumpy(value) for value in buffers)
            else:
                for value, host in zip(buffers, self.rng_buffers, strict=True):
                    value.get(out=host, blocking=True)
            rng.bit_generator._state = None
            rng._binomial_state = None
            if self.estimate is None:
                self.estimate = cp.asnumpy(recon)
            else:
                recon.get(out=self.estimate, blocking=True)
        else:
            self.estimate = recon
        return self.estimate


def _host_slices(shape: tuple[int, ...]) -> Iterator[slice]:
    """Yield slabs with approximately one million elements of scratch space.

    Parameters
    ----------
    shape : tuple of int
        Global host-array dimensions.

    Yields
    ------
    slice
        First-axis slab; the split sequence is independent of FFT crop size.
    """
    step = max(1, 2**20 // math.prod(shape[1:]))
    for start in range(0, shape[0], step):
        yield slice(start, min(start + step, shape[0]))


def _gc_stopping_metrics(
    previous: cp.ndarray,
    recon: cp.ndarray,
    consensus: cp.ndarray,
    limit: float,
    max_delta: float,
) -> tuple[float, float, str | None]:
    """Evaluate optional accepted-fraction and relative-change stopping limits.

    The accepted fraction follows the strictly positive gate in paper Eq. (5).
    These additional limits are distinct from the paper's KL stopping rule.

    Parameters
    ----------
    previous : array-like
        Object estimate before the current iteration.
    recon : array-like
        Current nonnegative object estimate.
    consensus : array-like
        Consensus field; strictly positive entries permit an update.
    limit : float
        Minimum accepted-update fraction.
    max_delta : float
        Minimum maximum-absolute change relative to the current peak.

    Returns
    -------
    tuple[float, float, str or None]
        Accepted fraction, relative change and selected stopping reason.
    """
    xp = cp.get_array_module(recon)
    if xp is np:
        rejected = 0
        peak, change = np.float32(0), np.float32(0)
        for slab in _host_slices(recon.shape):
            rejected += np.count_nonzero(consensus[slab] <= 0)
            peak = np.maximum(peak, np.max(recon[slab]))
            change = np.maximum(change, np.max(np.abs(recon[slab] - previous[slab])))
        fraction = float(recon.size - rejected) / recon.size
        delta = float(change / np.maximum(peak, 1e-12))
    else:
        fraction = float(recon.size - xp.count_nonzero(consensus <= 0)) / recon.size
        denominator = xp.maximum(xp.max(recon), 1e-12)
        delta = float(xp.max(xp.abs(recon - previous)) / denominator)
    reason = (
        "consensus_fraction"
        if fraction < limit
        else "relative_change"
        if delta < max_delta
        else None
    )
    return fraction, delta, reason


def clear_rlgc_caches(clear_memory_pool: bool = False) -> None:
    """Clear cached FFT resources used by RLGC helper functions.

    Parameters
    ----------
    clear_memory_pool : bool, default=False
        If True, synchronize the current CUDA stream and release CuPy device
        and pinned memory pools in addition to clearing CuPy FFT plans.

    Returns
    -------
    None
    """
    try:
        cp.fft.config.get_plan_cache().clear()
    except Exception:
        pass
    if clear_memory_pool:
        gc.collect()
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def _axis_linear_fft_padding(
    length: int,
    psf_support: int,
    *,
    halo_multiplier: int = 1,
) -> tuple[int, int]:
    """Calculate symmetric linear-convolution padding for one axis.

    Parameters
    ----------
    length : int
        Input image length along the axis.
    psf_support : int
        PSF support length along the same axis.
    halo_multiplier : int, default=1
        Multiplier applied to the PSF half-width when constructing the
        reflected boundary halo.

    Returns
    -------
    tuple[int, int]
        Padding before and after the axis. The padding includes the PSF halo
        and any extra samples needed to reach an FFT-friendly length.
    """
    halo = max((int(psf_support) // 2) * int(halo_multiplier), 0)
    length_with_halo = length + 2 * halo
    new_length = next_gpu_fft_size(length_with_halo)
    fft_extra = new_length - length_with_halo
    pad_before = halo + fft_extra // 2
    pad_after = halo + fft_extra - fft_extra // 2
    return pad_before, pad_after


def _linear_fft_pad_width(
    image_shape: tuple[int, int, int],
    psf_shape: tuple[int, int, int],
    pad_yx: bool = True,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Calculate per-axis linear FFT padding without allocating an image.

    Parameters
    ----------
    image_shape : tuple[int, int, int]
        Input image dimensions in scan, camera-Y, camera-X order.
    psf_shape : tuple[int, int, int]
        Point-spread function support dimensions in scan, camera-Y, camera-X order.
    pad_yx : bool
        Include reflected detector YX halos in addition to scan-axis padding.

    Returns
    -------
    tuple
        Per-axis reflected boundary and FFT expansion widths.
    """
    pad_scan = _axis_linear_fft_padding(image_shape[0], psf_shape[0])
    if pad_yx:
        pad_y = _axis_linear_fft_padding(image_shape[1], psf_shape[1])
        pad_x = _axis_linear_fft_padding(image_shape[2], psf_shape[2])
    else:
        pad_y = (0, 0)
        pad_x = (0, 0)
    return pad_scan, pad_y, pad_x


def pad_psf(
    psf_temp: cp.ndarray,
    image_shape: tuple[int, int, int],
    normalize: bool = True,
) -> cp.ndarray:
    """Pad and center a PSF to match the target image shape.

    Parameters
    ----------
    psf_temp : cupy.ndarray
        Original PSF (Z, Y, X).
    image_shape : tuple of int
        Target shape (Z, Y, X).
    normalize : bool, default=True
        If True, normalize the padded PSF to unit sum. Set False only for
        diagnostics that intentionally preserve the input PSF scale.

    Returns
    -------
    cupy.ndarray
        Padded, centered, nonnegative PSF.
    """
    if psf_temp.ndim == 2:
        psf_temp = cp.expand_dims(psf_temp, axis=0)

    psf = cp.zeros(image_shape, dtype=cp.float32)
    psf[: psf_temp.shape[0], : psf_temp.shape[1], : psf_temp.shape[2]] = psf_temp

    psf = cp.roll(psf, tuple(-(size // 2) for size in psf_temp.shape), axis=(0, 1, 2))
    if normalize:
        s = cp.sum(psf)
        psf = psf / (s if s != 0 else 1.0)
    return psf.astype(cp.float32, copy=False)


def fft_conv(
    image: cp.ndarray, H: cp.ndarray, shape: tuple[int, int, int]
) -> cp.ndarray:
    """
    Convolution using the transform output as its frequency workspace.

    This computes ``irfftn(rfftn(image) * H, s=shape)`` without a redundant
    spectrum buffer or copy. No clipping is applied here-this matches the reference
    implementation used to compute predictions, ratios and consensus.

    Parameters
    ----------
    image : cupy.ndarray
        Input array in object space.
    H : cupy.ndarray
        Frequency-domain transfer function (RFFTN of PSF or its conjugate).
    shape : tuple of int
        Target inverse FFT shape (Z, Y, X).

    Returns
    -------
    cupy.ndarray
        Convolved array in object space (float32).
    """
    spectrum = cp.fft.rfftn(image)
    spectrum *= H
    return irfftn(spectrum, s=shape, overwrite_x=True).astype(cp.float32, copy=False)


def _observed_region_slices(
    shape: tuple[int, int, int],
    pad_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> tuple[slice, slice, slice]:
    """Return slices selecting the unpadded observed image region.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Full padded image shape.
    pad_width : tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        Per-axis padding widths.

    Returns
    -------
    tuple[slice, slice, slice]
        Slices selecting the original image inside the padded volume.
    """
    slices = []
    for axis, (pad_before, pad_after) in enumerate(pad_width):
        stop = shape[axis] - pad_after if pad_after > 0 else None
        slices.append(slice(pad_before, stop))
    return tuple(slices)


def _kl_div_into(
    p: cp.ndarray,
    q: cp.ndarray,
    scratch: cp.ndarray,
) -> float:
    """Compute normalized KLD with the local reference's float32 operation order.

    Parameters
    ----------
    p : cupy.ndarray
        First nonnegative distribution.
    q : cupy.ndarray
        Second nonnegative distribution.
    scratch : cupy.ndarray
        Float32 workspace with the same shape as ``p`` and ``q``.

    Returns
    -------
    float
        Float32 sum of directed prediction-to-observation KLD terms, represented
        as a Python float. NaN terms are zeroed as in the reference.
    """
    xp = cp.get_array_module(p)
    if xp is np:
        return _host_kl_div_into(p, q, scratch)
    probability_p = p + 1e-4
    probability_q = q + 1e-4
    probability_p /= xp.sum(probability_p)
    probability_q /= xp.sum(probability_q)
    # Preserve float32 operation order while reusing the dead q buffer. The
    # second KL evaluation also retains a rollback estimate; materializing every
    # logarithm and arithmetic result can exceed an otherwise fitting chunk.
    xp.log(probability_p, out=scratch)
    xp.log(probability_q, out=probability_q)
    scratch -= probability_q
    scratch *= probability_p
    scratch[xp.isnan(scratch)] = 0
    return float(xp.sum(scratch))


def _host_kl_div_into(
    p: np.ndarray,
    q: np.ndarray,
    scratch: np.ndarray,
    *,
    clip_prediction: bool = False,
) -> float:
    """Evaluate the same global normalized KL with slab-sized temporaries.

    Parameters
    ----------
    p, q : numpy.ndarray
        Global prediction and observation distributions, including strided views.
    scratch : numpy.ndarray
        Existing global float32 KL workspace.
    clip_prediction : bool
        Clip prediction roundoff for KL only, without changing the RL update.

    Returns
    -------
    float
        Prediction-to-observation KL using global float32 normalization and sum.
    """
    if clip_prediction:
        np.maximum(p, 0, out=scratch)
        scratch += np.float32(1e-4)
    else:
        np.add(p, np.float32(1e-4), out=scratch)
    mass_p = np.sum(scratch)
    np.add(q, np.float32(1e-4), out=scratch)
    mass_q = np.sum(scratch)
    for slab in _host_slices(p.shape):
        values = np.maximum(p[slab], 0) if clip_prediction else p[slab]
        probability_p = (values + np.float32(1e-4)) / mass_p
        probability_q = (q[slab] + np.float32(1e-4)) / mass_q
        terms = probability_p * (np.log(probability_p) - np.log(probability_q))
        terms[np.isnan(terms)] = 0
        scratch[slab] = terms
    return float(np.sum(scratch))


def split_kl_scores(
    predicted: cp.ndarray,
    split1: cp.ndarray,
    split2: cp.ndarray,
    scratch: cp.ndarray,
) -> tuple[float, float]:
    """Evaluate two directed, normalized KL divergences on equal-sized grids.

    Score the acquired camera prediction against each data half. Clip negative
    FFT roundoff only for this statistic; reconstruction is unchanged.

    Parameters
    ----------
    predicted : array-like
        Camera prediction on the same grid as the data halves.
    split1 : array-like
        First measured half image in the same geometry.
    split2 : array-like
        Second measured half image in the same geometry.
    scratch : array-like
        Reusable workspace matching the common array shape.

    Returns
    -------
    tuple[float, float]
        Normalized prediction-to-half KL divergences.
    """
    xp = cp.get_array_module(predicted)
    if xp is np:
        return tuple(
            _host_kl_div_into(predicted, half, scratch, clip_prediction=True)
            for half in (split1, split2)
        )
    predicted = xp.maximum(predicted, 0)
    return tuple(_kl_div_into(predicted, half, scratch) for half in (split1, split2))


@dataclass
class RlgcChunkState:
    """Reuse an allocation-tested scan crop across one tile's channels."""

    crop_scan: int | None = None


def _split_observed_counts(
    observed: cp.ndarray, rng: cp.random.Generator
) -> cp.ndarray:
    """Split integer events binomially and allocate fractional counts symmetrically.

    Integer photon counts follow exact Poisson thinning at every count level.
    For S=n+f, draw Binomial(n, 0.5) and independently assign f to either half
    with equal probability. Both halves have mean S/2, preserve S by subtraction,
    and stay nonnegative. Their conditional variance is (n+f**2)/4. This is an
    approximation for fractional camera counts, not exact Poisson thinning.

    Parameters
    ----------
    observed : array-like
        Nonnegative calibrated counts, including any read-noise variance shift.
    rng : numpy.random.Generator or cupy.random.Generator
        Generator on the same backend as observed.

    Returns
    -------
    array-like
        Nonnegative first half; the caller forms observed minus this draw.
    """
    xp = cp.get_array_module(observed)
    if xp is np and observed.ndim:
        first = np.empty_like(observed, dtype=np.float32)
        for slab in _host_slices(observed.shape):
            first[slab] = rng.binomial(observed[slab].astype(np.int64), p=0.5)
    else:
        first = rng.binomial(observed.astype(xp.int64), p=0.5).astype(xp.float32)
    slabs = _host_slices(observed.shape) if xp is np and observed.ndim else (Ellipsis,)
    for slab in slabs:
        remainder = observed[slab] - xp.floor(observed[slab])
        if bool(xp.any(remainder)):
            first[slab] += remainder * (
                rng.random(remainder.shape, dtype=xp.float32) < 0.5
            )
    return first


def _read_noise_variance(read_noise: float) -> np.float32:
    """Convert calibrated RMS noise to the shifted-count variance.

    Parameters
    ----------
    read_noise : float
        RMS read noise in calibrated image units.

    Returns
    -------
    float
        Variance used in the shifted-count approximation.

    Raises
    ------
    ValueError
        RMS noise is negative, nonfinite or not representable.
    """
    sigma = float(read_noise)
    if not np.isfinite(sigma) or sigma < 0 or sigma > np.sqrt(np.finfo(np.float32).max):
        raise ValueError("read_noise must be finite, nonnegative and representable")
    return np.float32(sigma * sigma)


def _split_calibrated_counts(
    observed: cp.ndarray, rng: cp.random.Generator, variance: np.float32
) -> tuple[cp.ndarray, cp.ndarray]:
    """Split the nonnegative shifted counts for the read-noise approximation.

    Rieger et al. (2024), https://doi.org/10.1364/OE.524683 , split M+variance
    and subtract variance/2 from each half. For an RL half exposure the variance
    is also variance/2, so the shifted RL ratio simplifies to 2*Q/(Ax+variance),
    where Q is a shifted half. Retaining Q avoids negative intermediate counts
    and the bias from independently clipping variance-subtracted half images.

    Parameters
    ----------
    observed : array-like
        Nonnegative calibrated detector estimates.
    rng : numpy.random.Generator or cupy.random.Generator
        Backend-compatible split generator.
    variance : float
        Read-noise variance in squared calibrated units.

    Returns
    -------
    tuple of array-like
        Nonnegative complementary halves of observed plus variance.
    """
    shifted = observed if variance == 0 else observed + variance
    first = _split_observed_counts(shifted, rng)
    return first, shifted - first


def _calibrated_observations(image: np.ndarray) -> np.ndarray:
    """Retain finite calibrated counts, clipping negative estimates at zero.

    Parameters
    ----------
    image : array-like
        Calibrated detector values before solver preparation.

    Returns
    -------
    numpy.ndarray
        Finite nonnegative float32 observations without count quantization.
    """
    observed = np.asarray(image, dtype=np.float32)
    if not np.isfinite(observed).all():
        raise ValueError("Camera-calibrated counts must be finite")
    return np.maximum(observed, 0)


def _detector_response(
    response: np.ndarray | None, image_shape: tuple[int, ...]
) -> np.ndarray | None:
    """Validate a positive detector-YX response, retaining compact storage.

    Parameters
    ----------
    response : array-like or None
        Positive detector-YX response; None selects uniform sensitivity.
    image_shape : tuple of int
        Acquired image dimensions ending in YX.

    Returns
    -------
    numpy.ndarray or None
        Validated compact detector response, or None.
    """
    if response is None:
        return None
    field = np.asarray(response, dtype=np.float32)
    if field.ndim == 3 and field.shape[0] == 1:
        field = field[0]
    field = np.broadcast_to(field, image_shape[-2:])
    if not np.isfinite(field).all() or np.any(field <= 0):
        raise ValueError("Detector response must be finite and strictly positive")
    return None if np.all(field == 1) else field


class _Convolution:
    """One periodic convolution, optionally evaluated by exact overlap-save tiles.

    Only FFT workspace is tiled. The object, random split, consensus and stopping
    decision remain global, so a chunk boundary is never an optical boundary.
    """

    def __init__(
        self,
        psf: np.ndarray,
        shape: tuple[int, int, int],
        crop_scan: int | None = None,
        normalize: bool = True,
    ) -> None:
        """Prepare the optical transfer function and optional FFT tiles.

        Parameters
        ----------
        psf : numpy.ndarray
            Centered optical kernel in scan-YX order.
        shape : tuple of int
            Periodic computational grid dimensions.
        crop_scan : int or None
            Retained scan planes per FFT tile; None uses the full grid.
        normalize : bool
            Normalize the kernel to unit mass before transforming.
        """
        self.shape = tuple(shape)
        self.crop_scan = shape[0] if crop_scan is None else int(crop_scan)
        if self.crop_scan < 1:
            raise ValueError("crop_scan must be positive")
        self.radius = psf.shape[0] // 2
        self.tile_shape = (
            (
                next_gpu_fft_size(min(self.crop_scan, shape[0]) + 2 * self.radius),
                *shape[1:],
            )
            if self.crop_scan < shape[0]
            else self.shape
        )
        self.otf = cp.fft.rfftn(
            pad_psf(cp.asarray(psf), self.tile_shape, normalize=normalize)
        )
        self.adjoint_otf = self.otf.conj()

    def __call__(self, values: Array, *, adjoint: bool = False) -> Array:
        """Convolve a global estimate using shared GPU FFT workspace.

        Parameters
        ----------
        values : array-like
            Object-grid values on the host or GPU.
        adjoint : bool
            Conjugate the transfer function to apply the adjoint.

        Returns
        -------
        array-like
            Convolution on the same backend and grid as values.
        """
        xp = cp.get_array_module(values)
        otf = self.adjoint_otf if adjoint else self.otf
        if self.tile_shape == self.shape:
            result = fft_conv(cp.asarray(values), otf, self.shape)
            return result if xp is cp else cp.asnumpy(result)
        result = xp.empty_like(values)
        for start in range(0, self.shape[0], self.crop_scan):
            stop = min(start + self.crop_scan, self.shape[0])
            indices = xp.arange(start - self.radius, stop + self.radius) % self.shape[0]
            padded = cp.zeros(self.tile_shape, dtype=cp.float32)
            padded[: indices.size] = cp.asarray(values[indices])
            blurred = fft_conv(padded, otf, self.tile_shape)[
                self.radius : self.radius + stop - start
            ]
            result[start:stop] = blurred if xp is cp else cp.asnumpy(blurred)
        return result


def _full_data_update(
    observed: Array,
    denominator: Array,
    adjoint: Callable,
    sensitivity: Array,
    variance: np.float32,
) -> Array:
    """Compute the same full-data RL multiplier with or without consensus.

    Parameters
    ----------
    observed : array-like
        Calibrated measured counts.
    denominator : array-like
        Positive forward prediction including the read-noise variance shift.
    adjoint : callable
        Adjoint of the measurement operator.
    sensitivity : array-like
        Positive adjoint image of unit measured values.
    variance : float
        Read-noise variance in calibrated units.

    Returns
    -------
    array-like
        Nonnegative sensitivity-normalized multiplicative RL update.
    """
    xp = cp.get_array_module(observed)
    if xp is np:
        measured_ratio = np.empty_like(observed)
        for slab in _host_slices(observed.shape):
            np.add(observed[slab], variance, out=measured_ratio[slab])
            measured_ratio[slab] /= denominator[slab]
    else:
        measured_ratio = (observed + variance) / denominator
    update = adjoint(measured_ratio)
    update /= sensitivity
    xp.maximum(update, 0, out=update)
    return update


def _iterate_rlgc(
    observed: Array,
    shape: tuple[int, int, int],
    forward: Callable,
    adjoint: Callable,
    smooth: Callable,
    region: tuple[slice, slice, slice],
    *,
    sensitivity: Array,
    gradient_consensus: bool,
    max_iterations: int | None,
    limit: float,
    max_delta: float,
    rng_seed: int | None,
    use_kl: bool,
    read_noise: float,
    initial_estimate: np.ndarray | None,
    logger: logging.Logger | None,
    log_prefix: str,
    callback: Callable | None,
    _checkpoint: _RlgcCheckpoint | None = None,
    _retained_region: tuple[slice, ...] | None = None,
    _finalize: bool = False,
) -> Array:
    """Iterate RL/RLGC independently of geometry and FFT tiling.

    Parameters
    ----------
    observed : array-like
        Calibrated measurements on the acquired grid.
    shape : tuple of int
        Full reconstruction dimensions in scan-YX order.
    forward : callable
        Map object estimates to acquired detector intensities.
    adjoint : callable
        Apply the exact adjoint of the supplied forward operator.
    smooth : callable
        Apply the optical smoothing used by the consensus gate.
    region : tuple of slice
        Unpadded acquired-pixel region used for KL scoring.
    sensitivity : array-like
        Adjoint image of unit detector values for this optical geometry.
    gradient_consensus : bool
        Use the reference GC gate and split-image stopping when True.
    max_iterations : int or None
        Optional iteration safety cap; None leaves GC stopping uncapped.
    limit : float
        Minimum accepted-update fraction.
    max_delta : float
        Minimum maximum-absolute relative object change.
    rng_seed : int or None
        Seed for fresh half-image splits.
    use_kl : bool
        Enable the notebook's prediction-to-half KL rule and rollback.
    read_noise : float
        RMS read noise in calibrated observation units.
    initial_estimate : numpy.ndarray or None
        Initial object; None selects the constant mean of acquired observations.
    logger : logging.Logger or None
        Destination for solver diagnostics.
    log_prefix : str
        Context prepended to diagnostic messages.
    callback : callable or None
        Observer receiving provisional iterates, including any later rejected by KL.
    _checkpoint : _RlgcCheckpoint or None
        Park or resume this chunk for the shared KL-selected iteration budget.
    _retained_region : tuple of slice or None
        Owned object-grid window saved at the first KL trigger.
    _finalize : bool
        Return only owned output at the known shared budget, discarding state.

    Returns
    -------
    array-like
        Final nonnegative reconstruction on the selected backend.
    """
    xp = cp.get_array_module(observed)
    variance = _read_noise_variance(read_noise)
    if logger is not None:
        logger.info("%spreparing sensitivity and reconstruction state", log_prefix)
    if _checkpoint is not None and _checkpoint.estimate is not None:
        recon, rng = _checkpoint.restore(xp)
        completed = _checkpoint.iterations
    else:
        recon = (
            xp.full(shape, xp.mean(observed[region]), dtype=xp.float32)
            if initial_estimate is None
            else xp.asarray(initial_estimate).copy()
        )
        rng = xp.random.default_rng(rng_seed) if gradient_consensus else None
        completed = 0
    use_kl = bool(use_kl and gradient_consensus)
    if _checkpoint is not None and _checkpoint.first_kl_stop is not None:
        use_kl = False
        limit = max_delta = 0
    scratch = xp.empty(observed[region].shape, dtype=xp.float32) if use_kl else None
    previous_scores = (np.inf, np.inf)
    previous_recon = None
    # Iteration two keeps both current and rollback estimates. Reserve that
    # extra live array before the first update so allocation fallback happens
    # before any completed work, through the caller's actual CUDA allocator.
    rollback_reserve = (
        xp.empty_like(recon) if xp is cp and use_kl and completed == 0 else None
    )
    pause_after_update = False
    reason = "max_iterations"
    while max_iterations is None or completed < max_iterations:
        if logger is not None:
            logger.info(
                "%siteration=%d starting forward projection", log_prefix, completed + 1
            )
        predicted = forward(recon)
        if gradient_consensus:
            if logger is not None:
                logger.info(
                    "%siteration=%d forming count splits",
                    log_prefix,
                    completed + 1,
                )
            split1, split2 = _split_calibrated_counts(observed, rng, variance)
            if use_kl:
                scores = split_kl_scores(
                    predicted[region],
                    xp.maximum(split1[region] - 0.5 * variance, 0),
                    xp.maximum(split2[region] - 0.5 * variance, 0),
                    scratch,
                )
                if scores[0] > previous_scores[0] and scores[1] > previous_scores[1]:
                    if _checkpoint is None:
                        recon = previous_recon
                        completed -= 1
                        reason = "split_kl"
                        break
                    owned = (
                        _retained_region
                        if _retained_region is not None
                        else np.s_[:, :, :]
                    )
                    accepted = previous_recon[owned]
                    provisional = recon[owned]
                    _checkpoint.accepted = (
                        cp.asnumpy(accepted) if xp is cp else accepted.copy()
                    )
                    _checkpoint.provisional = (
                        cp.asnumpy(provisional) if xp is cp else provisional.copy()
                    )
                    del accepted, provisional
                    _checkpoint.first_kl_stop = completed - 1
                    # Complete this already prepared step once, preserving its
                    # exact split and RNG position for a later continuation.
                    pause_after_update = True
                previous_scores = scores
                previous_recon = recon
            if logger is not None:
                logger.info(
                    "%siteration=%d backprojecting and evaluating consensus",
                    log_prefix,
                    completed + 1,
                )
            # Consume the prediction and halves after KL scoring, preserving
            # float32 operation order without extra full-volume GPU buffers.
            denominator = predicted
            xp.add(denominator, variance, out=denominator)
            xp.maximum(denominator, xp.float32(1e-12), out=denominator)
            del predicted
            split1 *= 2
            split1 /= denominator
            half1 = adjoint(split1)
            del split1
            half1 /= sensitivity
            split2 *= 2
            split2 /= denominator
            half2 = adjoint(split2)
            del split2
            half2 /= sensitivity
            # Eq. (4): smooth the product of ADDITIVE object changes. The
            # spatially varying x**2 factor must be inside the blurring operator.
            half1 -= 1
            half1 *= recon
            half2 -= 1
            half2 *= recon
            half1 *= half2
            del half2
            consensus = smooth(half1, sensitivity)
            del half1
            ratio = _full_data_update(
                observed, denominator, adjoint, sensitivity, variance
            )
            if xp is np:
                updated = ratio
                for slab in _host_slices(recon.shape):
                    updated[slab] *= recon[slab]
                    np.copyto(updated[slab], recon[slab], where=consensus[slab] <= 0)
            else:
                updated = filter_update(recon, ratio, consensus)
            if (use_kl or (limit <= 0 and max_delta <= 0)) and (
                logger is None or not logger.isEnabledFor(logging.DEBUG)
            ):
                fraction, delta, reason = None, None, None
            else:
                fraction, delta, reason = _gc_stopping_metrics(
                    recon, updated, consensus, limit, max_delta
                )
                if use_kl:
                    reason = None
            del denominator, ratio, consensus
        else:
            denominator = xp.maximum(predicted + variance, xp.float32(1e-12))
            ratio = _full_data_update(
                observed, denominator, adjoint, sensitivity, variance
            )
            xp.maximum(ratio, 0, out=ratio)
            updated = recon * ratio
            fraction = 1.0
            delta = float(
                xp.max(xp.abs(updated - recon)) / xp.maximum(xp.max(updated), 1e-12)
            )
            reason = "relative_change" if delta < max_delta else None
            del predicted, denominator, ratio
        recon = updated
        completed += 1
        if rollback_reserve is not None:
            rollback_reserve = None
        if callback is not None:
            callback(completed, recon)
        if logger is not None:
            logger.debug(
                "%siteration=%d updated_fraction=%.6g max_relative_delta=%.6g",
                log_prefix,
                completed,
                fraction,
                delta,
            )
        if pause_after_update or reason is not None:
            break
    else:
        reason = "max_iterations"
    if _checkpoint is not None:
        if _finalize:
            _checkpoint.iterations = completed
            _checkpoint.estimate = _checkpoint.rng = _checkpoint.rng_buffers = None
            owned = recon[_retained_region]
            return cp.asnumpy(owned) if xp is cp else owned.copy()
        return _checkpoint.park(recon, rng, completed)
    if logger is not None:
        label = {"consensus_fraction": "limit", "relative_change": "max_delta"}.get(
            reason, reason
        )
        logger.info("%sstop=%s iterations=%d", log_prefix, label, completed)
    return recon


def _rlgc_once(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    safe_mode: bool = True,
    limit: float = GC_UPDATED_FRACTION_LIMIT,
    max_delta: float = GC_RELATIVE_CHANGE_LIMIT,
    pad_yx: bool = True,
    rng_seed: int | None = 42,
    normalize_psf: bool = True,
    release_memory: bool = True,
    logger: logging.Logger | None = None,
    log_prefix: str = "",
    max_iterations: int | None = None,
    detector_response: np.ndarray | None = None,
    use_kl: bool = True,
    read_noise: float = 0.0,
    *,
    scan_upsample_factor: int = 1,
    boundary: str = "reflect",
    crop_scan: int | None = None,
    gradient_consensus: bool = True,
    initial_estimate: np.ndarray | None = None,
    callback: Callable | None = None,
    storage: str = "gpu",
    _checkpoint: _RlgcCheckpoint | None = None,
    _finalize: bool = False,
    _optical_state: dict | None = None,
) -> np.ndarray:
    """Deconvolve calibrated data with the paper's RL and consensus equations.

    Parameters
    ----------
    image, psf : numpy.ndarray
        Nonnegative calibrated 2D/3D measurements and a centered, odd-sized PSF.
        The PSF must already be sampled on the requested reconstruction grid.
    read_noise : float
        RMS noise in the calibrated measurement units; zero is ideal photon data.
    scan_upsample_factor : int
        Acquired scan spacing divided by reconstructed scan spacing. Missing
        planes are unobserved, not zero-valued measurements. Requires zero
        boundary extension when greater than one.
    boundary : {"reflect", "zero"}
        Symmetric acquisition padding, or a zero-extended object with exact
        convolution transpose and acquired-plane sensitivity normalization.
    crop_scan : int or None
        Maximum retained scan planes per FFT tile. Overlap-save evaluates the
        same global operator; splits and stopping never run independently by tile.
    limit, max_delta : float
        Existing accepted-voxel fraction and maximum-relative-change thresholds.
    max_iterations : int or None
        Optional safety cap. None uses reference GC stopping without a cap.
        Ordinary RL requires an explicit cap or a positive max_delta.
    rng_seed : int or None
        Seed for the reference random halves, shared across all FFT tiles.
    detector_response : numpy.ndarray or None
        Positive YX sensitivity in the measurement operator, not input division.
    use_kl : bool
        Enable the notebook's both-halves KL stopping and rollback on the
        acquired camera prediction.
    gradient_consensus : bool
        False selects ordinary RL for controlled comparisons using the same optics.
    initial_estimate : numpy.ndarray or None
        Positive reconstruction-grid initialization. None uses mean observations,
        adjusted to object units for nonunit detector response or boundary operators.
    callback : callable or None
        Receives each provisional iteration number and its device array. KL
        stopping can roll back the last iterate. Copy state that must outlive
        the callback; it does not control convergence. Native arrays include
        symmetric padding.
    storage : str
        Backend for reconstruction state, gpu or host; FFTs always execute on CUDA.
    _checkpoint : _RlgcCheckpoint or None
        Per-chunk continuation for the shared KL-selected iteration budget.
    _finalize : bool
        Return owned output instead of parking another continuation.
    _optical_state : dict or None
        Convolution and sensitivity reused for one chunk geometry within a channel.
    gpu_id, safe_mode, pad_yx, normalize_psf, release_memory, logger, log_prefix
        Device, compatibility, padding, normalization and execution diagnostics.
        safe_mode is retained for compatibility; KL always requires both halves.

    Returns
    -------
    numpy.ndarray
        Float32 reconstruction. A 2D input retains the legacy singleton-Z output.
    """
    image = np.asarray(image, dtype=np.float32)
    psf = np.asarray(psf, dtype=np.float32)
    if image.ndim == 2:
        image = image[None]
    if psf.ndim == 2:
        psf = psf[None]
    for name, array in (("image", image), ("psf", psf)):
        if array.ndim != 3 or not array.size:
            raise ValueError(f"{name} must be a nonempty 2D or 3D array")
    if (
        not np.isfinite(psf).all()
        or np.any(psf < 0)
        or psf.sum(dtype=np.float64) <= 0
        or any(n % 2 == 0 for n in psf.shape)
    ):
        raise ValueError(
            "psf must be finite and nonnegative with positive sum and odd dimensions"
        )
    for name, value in (
        ("scan_upsample_factor", scan_upsample_factor),
        ("max_iterations", max_iterations),
        ("crop_scan", crop_scan),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if boundary not in ("reflect", "zero") or (
        boundary == "reflect" and scan_upsample_factor != 1
    ):
        raise ValueError("scan upsampling requires boundary='zero'")
    if (
        not np.isfinite(limit)
        or limit < 0
        or not np.isfinite(max_delta)
        or max_delta < 0
    ):
        raise ValueError("Convergence thresholds must be finite and nonnegative")
    if not gradient_consensus and max_iterations is None and max_delta == 0:
        raise ValueError("Ordinary RL requires a bound or relative-change threshold")
    if (
        gradient_consensus
        and not use_kl
        and max_iterations is None
        and limit == 0
        and max_delta == 0
    ):
        raise ValueError("RLGC requires a stopping threshold or iteration bound")
    if storage not in ("gpu", "host"):
        raise ValueError("storage must be gpu or host")
    xp = cp if storage == "gpu" else np
    response = _detector_response(detector_response, image.shape)
    image = _calibrated_observations(image)
    factor = int(scan_upsample_factor)
    fine_shape = ((image.shape[0] - 1) * factor + 1, *image.shape[1:])
    if _optical_state is None:
        _optical_state = {}
    try:
        with cp.cuda.Device(gpu_id):
            if not np.any(image):
                if logger is not None:
                    logger.info("%sstop=empty_observation iterations=0", log_prefix)
                return np.zeros(fine_shape, dtype=np.float32)
            pads = _linear_fft_pad_width(
                fine_shape, psf.shape, pad_yx=pad_yx or boundary == "zero"
            )
            padded_shape = tuple(
                n + sum(p) for n, p in zip(fine_shape, pads, strict=True)
            )
            geometry = (fine_shape, padded_shape)
            previous = _optical_state.get("geometry")
            if geometry != previous:
                _optical_state.clear()
                # Drop the previous FFT workspace before allocating a new shape.
                if previous is not None and padded_shape != previous[1]:
                    clear_rlgc_caches(clear_memory_pool=True)
                _optical_state["geometry"] = geometry
            if storage == "host":
                # Explicit host storage still needs room for this chunk's state
                # and temporaries. The outer scan-chunk loop bounds this shape.
                estimated_host_bytes = (
                    math.prod(padded_shape) * 10 * np.dtype(np.float32).itemsize
                )
                available_host_bytes = psutil.virtual_memory().available
                if estimated_host_bytes > 0.8 * available_host_bytes:
                    raise MemoryError(
                        f"RLGC host state for padded shape {padded_shape} needs approximately "
                        f"{estimated_host_bytes / 1024**3:.1f} GiB; only "
                        f"{available_host_bytes / 1024**3:.1f} GiB RAM is available. "
                        "Reduce --decon-crop-scan or free host RAM."
                    )
            core = _observed_region_slices(padded_shape, pads)
            if "convolution" not in _optical_state:
                _optical_state["convolution"] = _Convolution(
                    psf, padded_shape, crop_scan, normalize_psf
                )
            convolve = _optical_state["convolution"]
            if logger is not None:
                logger.info(
                    "%sRLGC object_shape=%s padded_shape=%s FFT_tile=%s storage=%s (%.2f GiB per float32 state array)",
                    log_prefix,
                    fine_shape,
                    padded_shape,
                    convolve.tile_shape,
                    storage,
                    math.prod(padded_shape) * np.dtype(np.float32).itemsize / 1024**3,
                )
            field = None if response is None else xp.asarray(response)
            reflected = boundary == "reflect"
            if reflected:
                observed = xp.asarray(np.pad(image, pads, mode="symmetric"))
                if response is not None:
                    field = xp.asarray(np.pad(response, pads[-2:], mode="symmetric"))
                shape = padded_shape
                region = core
            else:
                observed = xp.asarray(image)
                shape = fine_shape
                region = np.s_[:, :, :]

            unsupported_scan = None
            if not reflected and factor > 1:
                # Determine exact zero columns of the acquired-plane operator
                # from PSF support, without a floating-point sensitivity cutoff.
                # FFT roundoff there must not be divided by the 1e-12 floor.
                visible = np.zeros(fine_shape[0], dtype=bool)
                acquired = np.arange(image.shape[0]) * factor
                offsets = (
                    np.flatnonzero(np.any(psf > 0, axis=(1, 2))) - psf.shape[0] // 2
                )
                for offset in offsets:
                    indices = acquired - offset
                    visible[indices[(indices >= 0) & (indices < fine_shape[0])]] = True
                if not visible.all():
                    unsupported_scan = xp.asarray(np.flatnonzero(~visible))

            interaction_otf = (
                convolve.otf * convolve.adjoint_otf
                if gradient_consensus
                and field is None
                and xp is cp
                and convolve.tile_shape == convolve.shape
                else None
            )

            def blur(values: Array, adjoint: bool = False) -> Array:
                """Convolve on the selected reflected or zero-extended object grid.

                Parameters
                ----------
                values : array-like
                    Object-grid values before acquired-plane selection.
                adjoint : bool
                    Apply the conjugate optical transfer function when True.

                Returns
                -------
                array-like
                    Convolution on the selected reconstruction grid.
                """
                if reflected:
                    return convolve(values, adjoint=adjoint)
                padded = xp.zeros(padded_shape, dtype=xp.float32)
                padded[core] = values
                return convolve(padded, adjoint=adjoint)[core].copy()

            def forward(estimate: Array) -> Array:
                """Predict acquired detector intensities from the object estimate.

                Parameters
                ----------
                estimate : array-like
                    Nonnegative reconstruction on the object grid.

                Returns
                -------
                array-like
                    Blurred, sampled and response-weighted detector prediction.
                """
                result = blur(estimate)
                if not reflected:
                    result = result[::factor].copy()
                return result if field is None else result * field

            def adjoint(values: Array) -> Array:
                """Backproject acquired values through the exact optical adjoint.

                Parameters
                ----------
                values : array-like
                    Values on the acquired detector grid.

                Returns
                -------
                array-like
                    Sensitivity-weighted backprojection on the object grid.
                """
                weighted = values if field is None else values * field
                if reflected:
                    return blur(weighted, adjoint=True)
                scattered = xp.zeros(fine_shape, dtype=xp.float32)
                scattered[::factor] = weighted
                result = blur(scattered, adjoint=True)
                if unsupported_scan is not None:
                    result[unsupported_scan] = 0
                return result

            def smooth(product: Array, sensitivity: Array) -> Array:
                """Apply optical consensus smoothing for the selected geometry.

                Parameters
                ----------
                product : array-like
                    Product of the two additive half-image object changes.
                sensitivity : array-like
                    Adjoint image of unit detector values.

                Returns
                -------
                array-like
                    Smoothed consensus field on the object grid.
                """
                if field is None:
                    if reflected:
                        if xp is cp and convolve.tile_shape == convolve.shape:
                            return fft_conv(
                                product,
                                interaction_otf,
                                padded_shape,
                            )
                    else:
                        # Retain padded H*H smoothing: cropping between these
                        # convolutions would change the operator at the boundary.
                        padded = xp.zeros(padded_shape, dtype=xp.float32)
                        padded[core] = product
                        if xp is cp and convolve.tile_shape == convolve.shape:
                            return fft_conv(padded, interaction_otf, padded_shape)[
                                core
                            ].copy()
                        return convolve(convolve(padded), adjoint=True)[core].copy()
                return adjoint(forward(product)) / sensitivity

            if initial_estimate is not None:
                initial_estimate = np.asarray(initial_estimate, dtype=np.float32)
                if (
                    initial_estimate.shape != shape
                    or not np.isfinite(initial_estimate).all()
                    or np.any(initial_estimate <= 0)
                ):
                    raise ValueError(
                        "initial_estimate must be finite, positive and match the reconstruction grid"
                    )
            elif (_checkpoint is None or _checkpoint.estimate is None) and (
                field is not None or not reflected or not normalize_psf
            ):
                # Normalized KL cannot identify an intensity-scale error.
                # Calibrate the constant initial object through A1 before KL
                # can roll back to it. Broadcasting retains only a scalar here.
                unit_mass = xp.sum(
                    forward(xp.ones(shape, dtype=xp.float32)), dtype=xp.float64
                )
                level = xp.sum(observed, dtype=xp.float64) / unit_mass
                initial_estimate = xp.broadcast_to(level.astype(xp.float32), shape)
            retained_region = None
            if _checkpoint is not None:
                start, stop, _ = _checkpoint.retained_scan.indices(fine_shape[0])
                offset = core[0].start if reflected else 0
                retained_region = (
                    slice(start + offset, stop + offset),
                    *(core[1:] if reflected else (slice(None), slice(None))),
                )
            if "sensitivity" not in _optical_state:
                _optical_state["sensitivity"] = xp.maximum(
                    adjoint(xp.ones_like(observed)), xp.float32(1e-12)
                )
            recon = _iterate_rlgc(
                observed,
                shape,
                forward,
                adjoint,
                smooth,
                region,
                sensitivity=_optical_state["sensitivity"],
                gradient_consensus=gradient_consensus,
                max_iterations=max_iterations,
                limit=limit,
                max_delta=max_delta,
                rng_seed=rng_seed,
                use_kl=use_kl,
                read_noise=read_noise,
                initial_estimate=initial_estimate,
                logger=logger,
                log_prefix=log_prefix,
                callback=callback,
                _checkpoint=_checkpoint,
                _retained_region=retained_region,
                _finalize=_finalize,
            )
            if _finalize:
                return recon
            result = recon[core] if reflected else recon
            if _checkpoint is not None:
                return result
            return cp.asnumpy(result) if xp is cp else result.copy()
    finally:
        if release_memory:
            with cp.cuda.Device(gpu_id):
                clear_rlgc_caches(clear_memory_pool=True)


def _is_gpu_memory_error(exc: BaseException) -> bool:
    """Identify CUDA allocation failures that should trigger chunk fallback.

    Parameters
    ----------
    exc : BaseException
        Exception raised during an RLGC attempt.

    Returns
    -------
    bool
        True if the exception appears to be a GPU or host memory allocation
        failure; otherwise False.
    """
    if isinstance(exc, _RlgcStagingError):
        return False
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, cp.cuda.memory.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "oom" in message


@contextmanager
def _gpu_memory_budget(gpu_id: int) -> Iterator[None]:
    """Bound default-pool allocations to physical device memory during an attempt.

    Parameters
    ----------
    gpu_id : int
        CUDA device used by the solver.

    Yields
    ------
    None
        Device state and FFT workspaces use the available physical memory,
        including this process's reusable allocations. Respect a caller's tighter
        limit and restore it on success or failure. This makes
        allocation fallback effective on systems that otherwise spill into shared
        GPU memory instead of raising an allocation error.
    """
    with cp.cuda.Device(gpu_id):
        pool = cp.get_default_memory_pool()
        original_limit = pool.get_limit()
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        budget = max(1, min(total_bytes, free_bytes + pool.total_bytes()))
        if original_limit:
            budget = min(budget, original_limit)
        pool.set_limit(size=budget)
        try:
            yield
        finally:
            pool.set_limit(size=original_limit)


def _chunked_rlgc_once(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    crop_scan: int,
    rng_seed: int | None,
    logger: logging.Logger | None,
    log_prefix: str,
    on_reconstructed_chunk: Callable[[np.ndarray, int, int], None] | None = None,
    **solver_options: object,
) -> np.ndarray | None:
    """Schedule haloed scan chunks with Ryomen and bounded device state.

    Parameters
    ----------
    image, psf : numpy.ndarray
        Acquired camera data and PSF sampled on the reconstruction grid.
    crop_scan : int
        Requested retained scan extent on the reconstruction grid. Chunk origins
        align with acquired planes when scan refinement is enabled.
    rng_seed : int or None
        Base random seed, incremented by chunk index as in the original workflow.
    logger : logging.Logger or None
        Destination for chunk and iteration diagnostics.
    log_prefix : str
        Context prepended to diagnostic messages.
    on_reconstructed_chunk : callable or None
        Consume finalized owned cores and their fine-scan bounds in ascending order.
        None assembles and returns the full reconstruction.
    solver_options : keyword arguments
        Optical, numerical, device and storage options for the same RLGC engine.

    Returns
    -------
    numpy.ndarray or None
        Full float32 output assembled from disjoint retained chunk interiors.
        A callback receives those interiors instead, without allocating that output.
        KL-enabled chunks park continuation state in system RAM.
    """
    factor = solver_options.get("scan_upsample_factor", 1)
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, np.integer))
        or factor < 1
    ):
        raise ValueError("scan_upsample_factor must be a positive integer")
    work = image[None] if image.ndim == 2 else image
    fine_scan = (work.shape[0] - 1) * factor + 1
    # Reuse FFT plans and device allocations between chunks. The public wrapper
    # releases resources after this frame returns, and clears them before OOM
    # retries. Per-chunk cleanup repeatedly rebuilds the same expensive plans.
    solver_options["release_memory"] = False
    if crop_scan >= fine_scan or not np.any(work):
        result = _rlgc_once(
            image,
            psf,
            rng_seed=rng_seed,
            logger=logger,
            log_prefix=log_prefix,
            **solver_options,
        )
        if on_reconstructed_chunk is not None:
            on_reconstructed_chunk(result, 0, fine_scan)
            return None
        return result
    if solver_options.get("initial_estimate") is not None:
        raise ValueError(
            "An explicit initial_estimate requires full-frame deconvolution"
        )
    # Preserve the original discarded halo of one complete PSF scan support.
    # Round outward to actual acquired planes, retaining the exact fine-grid phase.
    psf_scan = 1 if np.ndim(psf) == 2 else psf.shape[0]
    halo = math.ceil(psf_scan / factor)
    retained = max(1, int(crop_scan) // factor)
    if retained + 2 * halo >= work.shape[0]:
        # The entire input fits one haloed chunk. Ryomen clamps larger crops to
        # the image size, which can leave no retained interior for short inputs.
        result = _rlgc_once(
            image,
            psf,
            rng_seed=rng_seed,
            logger=logger,
            log_prefix=log_prefix,
            **solver_options,
        )
        if on_reconstructed_chunk is not None:
            on_reconstructed_chunk(result, 0, fine_scan)
            return None
        return result
    slices = Slicer(
        work[:, :1, :1],
        crop_size=(retained + 2 * halo, 1, 1),
        overlap=(halo, 0, 0),
        pad=True,
    )
    # Only scan coordinates are needed; avoid padding full camera planes just
    # to obtain Ryomen's layout. Reconstruction reads the original work below.
    layouts = []
    owned_stop = 0
    for tile_index, (crop, source, destination) in enumerate(slices):
        start = max(destination[-3].start, owned_stop)
        stop = destination[-3].stop
        if start >= stop:
            continue
        origin = destination[-3].start - source[-3].indices(crop.shape[0])[0]
        source_start = max(origin, 0)
        source_stop = min(origin + crop.shape[0], work.shape[0])
        if factor > 1 and source_stop < work.shape[0]:
            source_stop += 1
        layouts.append(
            (
                tile_index,
                source_start,
                source_stop,
                start * factor,
                min(stop * factor, fine_scan),
            )
        )
        owned_stop = stop
    tile_count = len(slices)
    del crop, slices
    if solver_options.get("gradient_consensus", True) and solver_options.get(
        "use_kl", True
    ):
        return _shared_kl_chunks(
            work,
            psf,
            layouts,
            fine_scan,
            factor,
            rng_seed,
            logger,
            log_prefix,
            tile_count,
            solver_options,
            on_reconstructed_chunk=on_reconstructed_chunk,
        )
    output = (
        np.empty((fine_scan, *work.shape[1:]), dtype=np.float32)
        if on_reconstructed_chunk is None
        else None
    )
    for tile_index, source_start, source_stop, dest_start, dest_stop in tqdm(
        layouts, desc="decon", unit="chunk", leave=False
    ):
        prefix = f"{log_prefix}chunk={tile_index + 1}/{tile_count} "
        if logger is not None:
            logger.info(
                "%sraw_scan=[%d:%d] retained_fine_scan=[%d:%d] storage=%s",
                prefix,
                source_start,
                source_stop,
                dest_start,
                dest_stop,
                solver_options.get("storage", "gpu"),
            )
        reconstructed = _rlgc_once(
            work[source_start:source_stop],
            psf,
            rng_seed=None if rng_seed is None else rng_seed + tile_index,
            logger=logger,
            log_prefix=prefix,
            **solver_options,
        )
        local_start = dest_start - source_start * factor
        core = reconstructed[local_start : local_start + dest_stop - dest_start]
        if on_reconstructed_chunk is None:
            output[dest_start:dest_stop] = core
        else:
            on_reconstructed_chunk(core, dest_start, dest_stop)
        del core, reconstructed
    return output


def _shared_kl_chunks(
    work: np.ndarray,
    psf: np.ndarray,
    layouts: list[tuple[int, int, int, int, int]],
    fine_scan: int,
    factor: int,
    rng_seed: int | None,
    logger: logging.Logger | None,
    log_prefix: str,
    tile_count: int,
    solver_options: dict,
    on_reconstructed_chunk: Callable[[np.ndarray, int, int], None] | None = None,
) -> np.ndarray | None:
    """Continue parked Ryomen chunks to the latest first-local-KL trigger.

    Each initial visit records the accepted and provisional owned cores and
    finishes the already prepared update before parking the full haloed state.
    A second visit advances only remaining iterations. No update is replayed;
    there is no device transfer after every iteration and no pooled KL rule.
    """
    storage = solver_options.get("storage", "gpu")
    reflected = solver_options.get("boundary", "reflect") == "reflect"
    psf_shape = (1, *psf.shape) if np.ndim(psf) == 2 else psf.shape
    shapes = []
    for _, start, stop, _, _ in layouts:
        fine_shape = ((stop - start - 1) * factor + 1, *work.shape[1:])
        pads = _linear_fft_pad_width(
            fine_shape,
            psf_shape,
            pad_yx=solver_options.get("pad_yx", True) or not reflected,
        )
        shapes.append(tuple(n + sum(p) for n, p in zip(fine_shape, pads, strict=True)))
    rng_bytes = 0
    if storage == "gpu":
        with cp.cuda.Device(solver_options.get("gpu_id", 0)):
            probe = cp.random.default_rng(0)
            probe.binomial(1, 0.5, size=1)
            try:
                rng_bytes = (
                    probe.bit_generator._state.nbytes + probe._binomial_state.nbytes
                )
            except AttributeError as exc:
                raise RuntimeError(
                    "This CuPy Generator cannot checkpoint its device state"
                ) from exc
            del probe
    state_bytes = sum(
        math.prod(
            shape if reflected else ((stop - start - 1) * factor + 1, *work.shape[1:])
        )
        * 4
        + (dest_stop - dest_start) * math.prod(work.shape[1:]) * 8
        + rng_bytes
        for shape, (_, start, stop, dest_start, dest_stop) in zip(
            shapes, layouts, strict=True
        )
    )
    output_bytes = (
        fine_scan * math.prod(work.shape[1:]) * 4
        if on_reconstructed_chunk is None
        else 0
    )
    workspace_bytes = (
        max(math.prod(shape) for shape in shapes) * 4 * (10 if storage == "host" else 3)
    )
    available = psutil.virtual_memory().available
    if state_bytes + output_bytes + workspace_bytes > 0.8 * available:
        raise _RlgcStagingError(
            f"RLGC chunk checkpoints and workspace need approximately "
            f"{(state_bytes + output_bytes + workspace_bytes) / 1024**3:.1f} GiB; "
            f"only {available / 1024**3:.1f} GiB RAM is available. Free host RAM "
            "or increase the retained chunk size to reduce duplicated halos."
        )
    if logger is not None:
        logger.info(
            "%sshared_kl checkpoint_bytes=%d workspace_bytes=%d",
            log_prefix,
            state_bytes,
            workspace_bytes,
        )
    output = (
        np.zeros((fine_scan, *work.shape[1:]), dtype=np.float32)
        if on_reconstructed_chunk is None
        else None
    )
    checkpoints = {}
    completed_updates = 0
    observer = solver_options.get("callback")

    def on_update(iteration: int, recon: Array) -> None:
        nonlocal completed_updates
        completed_updates += 1
        if observer is not None:
            observer(iteration, recon)

    options = dict(solver_options, callback=on_update, _optical_state={})
    # Allocate the largest optical working set first. Startup allocation fallback
    # remains available, but a later failure must never replay completed updates.
    order = sorted(
        range(len(layouts)), key=lambda i: math.prod(shapes[i]), reverse=True
    )
    common_stop = 0
    try:
        for index in tqdm(order, desc="decon KL", unit="chunk", leave=False):
            tile, start, stop, dest_start, dest_stop = layouts[index]
            crop = work[start:stop]
            local_start = dest_start - start * factor
            state = _RlgcCheckpoint(
                slice(local_start, local_start + dest_stop - dest_start)
            )
            prefix = f"{log_prefix}chunk={tile + 1}/{tile_count} "
            if logger is not None:
                logger.info(
                    "%sraw_scan=[%d:%d] retained_fine_scan=[%d:%d] storage=%s",
                    prefix,
                    start,
                    stop,
                    dest_start,
                    dest_stop,
                    storage,
                )
            _rlgc_once(
                crop,
                psf,
                rng_seed=None if rng_seed is None else rng_seed + tile,
                logger=logger,
                log_prefix=prefix,
                _checkpoint=state,
                **options,
            )
            if state.estimate is None:
                # The engine clips finite negative camera values to zero and
                # returns empty chunks before constructing numerical state.
                continue
            checkpoints[index] = state
            accepted = (
                state.iterations if state.first_kl_stop is None else state.first_kl_stop
            )
            common_stop = max(common_stop, accepted)
            if logger is not None:
                logger.info(
                    "%sfirst_kl_stop=%s checkpoint_iteration=%d",
                    prefix,
                    state.first_kl_stop,
                    state.iterations,
                )
        reason = (
            "max_iterations"
            if any(state.first_kl_stop is None for state in checkpoints.values())
            else "shared_split_kl"
            if checkpoints
            else "empty_observation"
        )
        finish_order = (
            range(len(layouts)) if on_reconstructed_chunk is not None else order
        )
        for index in tqdm(finish_order, desc="decon finish", unit="chunk", leave=False):
            tile, start, stop, dest_start, dest_stop = layouts[index]
            state = checkpoints.pop(index, None)
            if state is None:
                if on_reconstructed_chunk is not None:
                    on_reconstructed_chunk(
                        np.zeros((dest_stop - dest_start, *work.shape[1:]), np.float32),
                        dest_start,
                        dest_stop,
                    )
                continue
            prefix = f"{log_prefix}chunk={tile + 1}/{tile_count} "
            if state.first_kl_stop == common_stop:
                result = state.accepted
            elif (
                state.first_kl_stop is not None
                and state.first_kl_stop + 1 == common_stop
            ):
                result = state.provisional
            else:
                if state.iterations < common_stop:
                    state.accepted = state.provisional = None
                    result = _rlgc_once(
                        work[start:stop],
                        psf,
                        rng_seed=None if rng_seed is None else rng_seed + tile,
                        logger=logger,
                        log_prefix=prefix,
                        _checkpoint=state,
                        _finalize=True,
                        **dict(options, max_iterations=common_stop),
                    )
                elif reflected:
                    fine_shape = ((stop - start - 1) * factor + 1, *work.shape[1:])
                    pads = _linear_fft_pad_width(
                        fine_shape,
                        psf_shape,
                        pad_yx=solver_options.get("pad_yx", True),
                    )
                    core = _observed_region_slices(shapes[index], pads)
                    result = state.estimate[core][state.retained_scan]
                else:
                    result = state.estimate[state.retained_scan]
            if on_reconstructed_chunk is None:
                output[dest_start:dest_stop] = result
            else:
                on_reconstructed_chunk(result, dest_start, dest_stop)
            if logger is not None:
                logger.info("%sstop=%s iterations=%d", prefix, reason, common_stop)
            del state, result
    except Exception as exc:
        if completed_updates and _is_gpu_memory_error(exc):
            raise RuntimeError(
                "RLGC allocation failed after reconstruction updates; refusing "
                "to restart completed chunk iterations. Free device memory or "
                "select a smaller initial crop."
            ) from exc
        raise
    if logger is not None:
        logger.info("%sstop=%s iterations=%d", log_prefix, reason, common_stop)
    return output


def rlgc(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    crop_scan: int | None = None,
    safe_mode: bool = True,
    limit: float = GC_UPDATED_FRACTION_LIMIT,
    max_delta: float = GC_RELATIVE_CHANGE_LIMIT,
    rng_seed: int | None = 42,
    normalize_psf: bool = True,
    verbose: int = 0,
    release_memory: bool = True,
    logger: logging.Logger | None = None,
    log_prefix: str = "",
    on_successful_crop_scan: Callable[[int], None] | None = None,
    fallback_step_scan: int | None = None,
    detector_response: np.ndarray | None = None,
    read_noise: float = 0.0,
    on_reconstructed_chunk: Callable[[np.ndarray, int, int], None] | None = None,
    **solver_options: object,
) -> np.ndarray | None:
    """Deconvolve raw scan chunks with bounded device state and shared KL stopping.

    The solver first attempts the requested ``crop_scan``. If CUDA memory
    allocation fails before any update, it retries with smaller scan-axis chunks.
    KL-enabled chunks park in system RAM and resume to the latest first local
    KL trigger without replaying iterations. Every solver
    call retains the full camera Y and X extents. If provided,
    ``on_successful_crop_scan`` is called with the crop size that completed
    successfully.

    Parameters
    ----------
    image : numpy.ndarray
        2D or 3D image to deconvolve.
    psf : numpy.ndarray
        2D or 3D point-spread function.
    gpu_id : int, default=0
        CUDA device ID to use.
    crop_scan : int or None, default=None
        Requested retained scan-axis tile size. None attempts the full scan
        and uses actual allocation failures to reduce it. Values at least as
        large as the reconstructed scan-axis length select full-frame processing.
        Refined scans round retained extents to at least one acquired scan interval.
    safe_mode : bool, default=True
        Retained for compatibility. The reference KL rule always requires both
        halves to worsen; this wrapper option has no effect.
    limit : float, default=0.001
        Minimum updated fraction when KL stopping is disabled.
    max_delta : float, default=0.001
        Relative-change stopping threshold when KL stopping is disabled.
    rng_seed : int or None, default=42
        Seed for per-iteration 50:50 data splits, incremented by scan chunk index.
    normalize_psf : bool, default=True
        If True, normalize the padded PSF to unit sum before deconvolution.
    verbose : int, default=0
        Chunked processing shows a decon progress bar. Zero emits no solver
        diagnostics. Set to 1 to report storage, FFT geometry, iterations and
        stopping reasons. An explicit logger opts into diagnostics with its own
        logging controls; DEBUG includes consensus and change statistics.
    release_memory : bool, default=True
        If True, release CuPy memory pools and FFT caches before returning.
    logger : logging.Logger or None, default=None
        Optional logger for route, retry, and per-iteration diagnostics.
    log_prefix : str, default=""
        Structured prefix prepended to emitted log lines.
    on_successful_crop_scan : Callable[[int], None] or None, default=None
        Optional callback receiving the crop size that completed successfully.
    fallback_step_scan : int or None, default=None
        Number of retained scan planes removed after each allocation failure.
        If None, reduce the current crop by one quarter.
    detector_response : numpy.ndarray or None
        Positive detector-YX illumination and gain response retained for every
        scan chunk. Supply camera-calibrated counts without dividing by it.

    read_noise : float, default=0
        RMS read noise in the same calibrated units as image. Zero selects
        the photon-only branch; processing supplies the camera calibration.

    on_reconstructed_chunk : callable or None
        Consume each finalized float32 scan-YX core and its fine-grid start/stop
        bounds synchronously, in ascending scan order. Shared-KL output starts
        only after all chunks have selected the common iteration budget. Delivered
        arrays are never modified again, but may view a chunk's checkpoint;
        retain only needed context to bound memory. None assembles the output array.

    solver_options : keyword arguments
        Numerical parameters forwarded to the single engine: ``use_kl`` enables
        reference KL stopping; ``max_iterations`` is an optional safety cap;
        ``scan_upsample_factor`` and ``boundary="zero"`` select acquired-plane
        reconstruction; ``initial_estimate`` overrides calibrated initialization;
        ``gradient_consensus=False`` selects ordinary RL. ``callback`` observes
        provisional chunk iterates (iteration numbering restarts per chunk).
        Native callback arrays include padding; KL can reject the last iterate.
        ``storage`` chooses GPU or host chunk state; FFTs use CUDA in both cases.
        ``pad_yx`` controls symmetric camera-axis padding; zero boundaries always
        retain linear-convolution padding to prevent detector-edge wrap. An explicit
        ``initial_estimate`` requires full-frame processing.

    Returns
    -------
    numpy.ndarray or None
        Deconvolved image as float32, or None when finalized chunks were consumed
        by on_reconstructed_chunk without assembling a full reconstruction.
    """
    image_arr = np.asarray(image)
    if logger is None and verbose > 0:
        logger = logging.Logger("opm_processing.rlgc", level=logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    if image_arr.ndim == 2:
        image_scan = 1
    elif image_arr.ndim == 3:
        image_scan = image_arr.shape[0]
    else:
        raise ValueError(f"Expected a 2D or 3D image, got shape {image_arr.shape}")
    if crop_scan is not None and (
        isinstance(crop_scan, bool)
        or not isinstance(crop_scan, (int, np.integer))
        or crop_scan < 1
    ):
        raise ValueError("crop_scan must be a positive integer")

    if (image_arr.ndim == 2 or image_arr.shape[0] == 1) and solver_options.get(
        "boundary", "reflect"
    ) == "reflect":
        psf = np.asarray(psf)
        if psf.ndim == 3:
            psf = psf[psf.shape[0] // 2]
    original_ndim = image_arr.ndim
    image_scan = (image_scan - 1) * int(
        solver_options.get("scan_upsample_factor", 1)
    ) + 1
    if crop_scan is None:
        crop_scan = image_scan
    if fallback_step_scan is not None and fallback_step_scan < 1:
        raise ValueError("fallback_step_scan must be at least 1")
    min_crop_scan = 1
    attempted_crop_scan = min(int(crop_scan), int(image_scan))
    solver_options.setdefault("storage", "gpu")
    delivered_chunk = False

    def deliver_chunk(core: np.ndarray, start: int, stop: int) -> None:
        """Do not restart an attempt after its output consumer has advanced."""
        nonlocal delivered_chunk
        delivered_chunk = True
        on_reconstructed_chunk(core, start, stop)

    if logger is not None:
        logger.info(
            "%sRLGC acquired_shape=%s PSF_shape=%s retained_scan=%d storage=%s",
            log_prefix,
            image_arr.shape,
            np.asarray(psf).shape,
            attempted_crop_scan,
            solver_options["storage"],
        )

    while True:
        try:
            if logger is not None and logger.isEnabledFor(logging.INFO):
                logger.info(
                    "%sattempt_rlgc_crop_scan=%d requested_crop_scan=%d",
                    f"{log_prefix} " if log_prefix else "",
                    attempted_crop_scan,
                    crop_scan,
                )
            with _gpu_memory_budget(gpu_id):
                output = _chunked_rlgc_once(
                    image=image_arr,
                    psf=psf,
                    gpu_id=gpu_id,
                    crop_scan=attempted_crop_scan,
                    safe_mode=safe_mode,
                    limit=limit,
                    max_delta=max_delta,
                    rng_seed=rng_seed,
                    normalize_psf=normalize_psf,
                    release_memory=release_memory,
                    logger=logger,
                    log_prefix=log_prefix,
                    detector_response=detector_response,
                    read_noise=read_noise,
                    on_reconstructed_chunk=(
                        deliver_chunk if on_reconstructed_chunk is not None else None
                    ),
                    **solver_options,
                )
            if release_memory:
                # The chunk engine has now returned, so its device arrays are
                # no longer held by the engine's frame when pools are released.
                with cp.cuda.Device(gpu_id):
                    clear_rlgc_caches(clear_memory_pool=True)
            if on_successful_crop_scan is not None:
                on_successful_crop_scan(attempted_crop_scan)
            if logger is not None and logger.isEnabledFor(logging.INFO):
                logger.info(
                    "%ssuccessful_rlgc_crop_scan=%d requested_crop_scan=%d",
                    f"{log_prefix} " if log_prefix else "",
                    attempted_crop_scan,
                    crop_scan,
                )
            return output[0] if output is not None and original_ndim == 2 else output
        except Exception as exc:
            allocation_error = _is_gpu_memory_error(exc)
            if not allocation_error or delivered_chunk:
                if release_memory:
                    with cp.cuda.Device(gpu_id):
                        clear_rlgc_caches(clear_memory_pool=True)
                if allocation_error:
                    raise RuntimeError(
                        "RLGC allocation failed after delivering finalized chunks; "
                        "refusing to replay reconstruction or output writes."
                    ) from exc
                raise

            if attempted_crop_scan == min_crop_scan:
                if release_memory:
                    with cp.cuda.Device(gpu_id):
                        clear_rlgc_caches(clear_memory_pool=True)
                raise RuntimeError(
                    "RLGC exhausted allocation fallback even at the "
                    f"minimum scan-axis crop size {attempted_crop_scan}."
                ) from exc
            # A failed chunk's traceback owns its arrays. Release those frames
            # before trying a smaller chunk, otherwise fallback retains exactly
            # the state whose allocation exhausted memory.
            exc.__traceback__ = None
            with cp.cuda.Device(gpu_id):
                clear_rlgc_caches(clear_memory_pool=True)
            retry_step = (
                max(1, attempted_crop_scan // 4)
                if fallback_step_scan is None
                else fallback_step_scan
            )
            next_crop_scan = max(min_crop_scan, attempted_crop_scan - retry_step)

            if logger is not None and logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "%sretry_after_gpu_oom previous_crop_scan=%d next_crop_scan=%d",
                    f"{log_prefix} " if log_prefix else "",
                    attempted_crop_scan,
                    next_crop_scan,
                )
            attempted_crop_scan = next_crop_scan
