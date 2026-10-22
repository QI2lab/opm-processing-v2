"""Richardson–Lucy deconvolution with gradient consensus (RLGC).

References
----------
James Manton's original RLGC implementation (2023), developed with Andrew York,
Jan Becker and Craig Russell: https://github.com/jdmanton/rlgc .
Andrew York, Gradient Consensus (2023): https://doi.org/10.5281/zenodo.10278919 .
The later shared RLGC notebook is available at
https://colab.research.google.com/drive/1mfVNSCaYHz1g56g92xBkIoa8190XNJpJ .

Yiming Liu and Sjoerd Stallinga (2026), "Application of gradient-consensus
Richardson–Lucy deconvolution to noisy undersampled brightfield microscopy
data", Biomedical Optics Express 17, 2999: https://doi.org/10.1364/BOE.597710 .
Associated MATLAB code: https://gitlab.tudelft.nl/imphys/ci/upsample-gc-rl ,
gc_deconv.m at revision 0340af0d7a4c5d92a97a1b443dfedb367dfb9f3d.

OPM adaptations
---------------
The algorithm follows gc_deconv.m, including its read-noise compensation.
Initialize with constant mean(J). Split J+sigma^2 using do_binosplit.m's rounding,
binomial/Gaussian split and uniform remainder; subtract sigma^2/2 from each half
and clip independently at zero. The full and half ratios use sigma^2 in both
numerators and the predicted-image denominator, exactly as in gc_deconv.m.
Blur (u1-1)*(u2-1), with no intensity weighting. Negative consensus rejects an
update and zero permits it. Compute the full RL ratio directly from J, as in
MATLAB; clipped halves need not sum to J and must not define that full ratio.

Default KL stopping follows gc_deconv.m with use_KL=1: compute normalized
D(prediction || split) for each fresh half, and stop when BOTH scores exceed
their preceding-iteration scores. As in MATLAB, retain the current estimate;
do not roll back an iteration. Score Ax on the acquired-pixel grid, including
detector response, rather than compare differently sized object and data grids.
Native reflected padding is excluded from KL scores because it is not acquired.
This is MATLAB's measurement-space convention; the paper's equations (6)-(7)
write the object estimate instead. Its 1e-4 pseudocount is retained. The existing
accepted-fraction and maximum-relative-change thresholds remain 0.001; callers
can disable KL stopping with use_kl=False. No PSNR-derived threshold is used.

Camera gain and offset are calibrated before reconstruction. read_noise is RMS
in those calibrated input units; zero selects the photon-only reference branch.
Processing supplies the explicitly configured camera RMS read noise. This is the
reference's shifted-count approximation, not a new camera-likelihood algorithm.
The split implementation keeps
MATLAB's signed rounding remainder and Gaussian approximation at rounded counts
of 150 or higher, including its out-of-range fallback. CuPy and MATLAB random
generators differ, so the same numeric seed does not imply identical halves.

OPM-specific differences are the 3D oblique PSF, conjugate-OTF adjoint, optional
detector response with sensitivity normalization, and symmetric image halos.
Convergence metrics include the padded native volume, while KL uses acquired
pixels only. GPU execution uses float32, efficient FFT dimensions and cached
workspaces; negative FFT roundoff in the nonnegative full RL multiplier is
clipped at zero. Scan chunking has allocation fallback. The acquired-plane
operator uses zero extension and scan-plane selection,
rather than MATLAB's 2D pixel averaging. These geometric/execution adaptations
are not a claim of identical boundary conditions or bitwise MATLAB results.

FFT tiles use overlap-save with the PSF support at every convolution. They share
one object, initialization, count split and stopping decision. GPU-memory fallback
can keep this global state in host RAM and transfer only haloed FFT tiles. NumPy
and CuPy random generators follow the same split distribution but differ bitwise.
"""

import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from opm_processing.cuda import preload_cuda_libraries

preload_cuda_libraries()

import cupy as cp
from cupy import ElementwiseKernel

# -----------------------------------------------------------------------------
# CUDA kernel: multiplicative RL step gated by the existing consensus sign rule
# -----------------------------------------------------------------------------
filter_update = ElementwiseKernel(
    "float32 recon, float32 HTratio, float32 consensus_map",
    "float32 out",
    """
    bool skip = consensus_map < 0;
    out = skip ? recon : recon * HTratio
    """,
    "filter_update",
)


Array = np.ndarray | cp.ndarray

_fft_cache_3d: dict[tuple[int, int, int, int], cp.ndarray] = {}

GC_UPDATED_FRACTION_LIMIT = 0.001
GC_RELATIVE_CHANGE_LIMIT = 0.001


def _gc_stopping_metrics(
    previous: cp.ndarray,
    recon: cp.ndarray,
    consensus: cp.ndarray,
    limit: float,
    max_delta: float,
) -> tuple[float, float, str | None]:
    """Evaluate gc_deconv.m's stopping conditions for native and sparse solvers.

    Reference: gc_deconv.m, revision 0340af0d7a4c5d92a97a1b443dfedb367dfb9f3d,
    stopping metrics and final OR condition. Siemens_rdnoise_plot.m supplies
    limit1=limit2=0.001. Negative consensus values reject updates; zero accepts.
    The strict inequalities and 1e-12 denominator floor are from that code.

    Parameters
    ----------
    previous : array-like
        Object estimate before the current iteration.
    recon : array-like
        Current nonnegative object estimate.
    consensus : array-like
        Consensus field; nonnegative entries permit an update.
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
    fraction = float(recon.size - xp.count_nonzero(consensus < 0)) / recon.size
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
    _fft_cache_3d.clear()
    try:
        cp.fft.config.get_plan_cache().clear()
    except Exception:
        pass
    try:
        import cupyx

        cupyx.scipy.fft.clear_plan_cache()
    except Exception:
        pass
    if clear_memory_pool:
        gc.collect()
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def next_gpu_fft_size(x: int) -> int:
    """
    Return the smallest FFT-friendly size >= ``x`` with prime factors in {2, 3}.

    Parameters
    ----------
    x : int
        Minimum desired length.

    Returns
    -------
    int
        Next 2-3-smooth length >= ``x``.
    """
    if x <= 1:
        return 1
    n = x
    while True:
        m = n
        while (m % 2) == 0:
            m //= 2
        while (m % 3) == 0:
            m //= 3
        if m == 1:
            return n
        n += 1


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

    # Center the PSF
    for axis, axis_size in enumerate(psf.shape):
        psf = cp.roll(psf, int(axis_size / 2), axis=axis)
    for axis, axis_size in enumerate(psf_temp.shape):
        psf = cp.roll(psf, -int(axis_size / 2), axis=axis)

    psf = cp.fft.ifftshift(psf)
    if normalize:
        s = cp.sum(psf)
        psf = psf / (s if s != 0 else 1.0)
    return psf.astype(cp.float32)


def fft_conv(
    image: cp.ndarray, H: cp.ndarray, shape: tuple[int, int, int]
) -> cp.ndarray:
    """
    Linear convolution via FFT with cached buffers (no clipping).

    This computes ``irfftn(rfftn(image) * H, s=shape)`` with preallocated
    work buffers. No clipping is applied here-this matches the reference
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
    device_id = int(cp.cuda.Device().id)
    cache_key = (device_id, *shape)
    if cache_key not in _fft_cache_3d:
        z, y, x = shape
        freq_shape = (z, y, x // 2 + 1)
        _fft_cache_3d[cache_key] = cp.empty(freq_shape, dtype=cp.complex64)

    fft_buf = _fft_cache_3d[cache_key]
    fft_buf[...] = cp.fft.rfftn(image)
    fft_buf[...] *= H
    return cp.fft.irfftn(fft_buf, s=shape).astype(cp.float32, copy=False)


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
    probability_p = p + 1e-4
    probability_q = q + 1e-4
    probability_p = probability_p / xp.sum(probability_p)
    probability_q = probability_q / xp.sum(probability_q)
    scratch[...] = probability_p * (xp.log(probability_p) - xp.log(probability_q))
    scratch[xp.isnan(scratch)] = 0
    return float(xp.sum(scratch))


def split_kl_scores(
    predicted: cp.ndarray,
    split1: cp.ndarray,
    split2: cp.ndarray,
    scratch: cp.ndarray,
) -> tuple[float, float]:
    """MATLAB's normalized D(Ax || half), evaluated on acquired pixels.

    Store these two scores and compare with the preceding iteration's scores,
    as gc_deconv.m does. The new halves are fresh draws, not held-out data.
    Clip negative FFT roundoff only for this statistic; predictions used in
    the multiplicative update are unchanged.

    Parameters
    ----------
    predicted : array-like
        Predicted intensities at acquired pixels.
    split1 : array-like
        First measured half image in the same geometry.
    split2 : array-like
        Second measured half image in the same geometry.
    scratch : array-like
        Reusable workspace matching the acquired-pixel shape.

    Returns
    -------
    tuple[float, float]
        Normalized prediction-to-half KL divergences.
    """
    xp = cp.get_array_module(predicted)
    predicted = xp.maximum(predicted, 0)
    return tuple(_kl_div_into(predicted, half, scratch) for half in (split1, split2))


def determine_rlgc_crop_scan(
    image_shape: tuple[int, int, int],
    psf_shapes: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
    gpu_id: int = 0,
    memory_fraction: float = 0.8,
) -> int:
    """Choose the largest scan crop estimated to fit available GPU memory.

    The estimate includes scan processing halos, linear-convolution padding,
    FFT workspaces, solver state, and iteration temporaries. All channel PSFs
    are considered so the returned crop can be reused across channels.

    Parameters
    ----------
    image_shape : tuple[int, int, int]
        Unpadded input shape in scan, camera-Y, camera-X order.
    psf_shapes : list[tuple[int, ...]] or tuple[tuple[int, ...], ...]
        Shapes of every PSF that will be used during processing.
    gpu_id : int, default=0
        CUDA device used by RLGC.
    memory_fraction : float, default=0.8
        Fraction of currently free device memory available to the solver.

    Returns
    -------
    int
        Retained scan-axis crop size.
    """
    if len(image_shape) != 3 or any(int(size) < 1 for size in image_shape):
        raise ValueError(f"Expected a positive scan-Y-X shape, got {image_shape}")
    if not psf_shapes:
        raise ValueError("At least one PSF shape is required")
    if not 0 < memory_fraction < 1:
        raise ValueError("memory_fraction must be between 0 and 1")

    normalized_psf_shapes = [
        (1, *shape) if len(shape) == 2 else tuple(int(size) for size in shape)
        for shape in psf_shapes
    ]
    if any(len(shape) != 3 for shape in normalized_psf_shapes):
        raise ValueError("PSF shapes must be two- or three-dimensional")
    psf_shape = tuple(
        max(shape[axis] for shape in normalized_psf_shapes) for axis in range(3)
    )

    cp.cuda.Device(gpu_id).use()
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    memory_budget = int(free_bytes * memory_fraction)

    # Reserve optical state, OTFs, FFT workspaces, split/reduction temporaries
    # and optional detector-response buffers, before memory_fraction is applied.
    estimated_bytes_per_voxel = 26 * np.dtype(np.float32).itemsize
    scan_size, camera_y, camera_x = (int(size) for size in image_shape)
    pad_y = _axis_linear_fft_padding(camera_y, psf_shape[1])
    pad_x = _axis_linear_fft_padding(camera_x, psf_shape[2])
    padded_y = camera_y + sum(pad_y)
    padded_x = camera_x + sum(pad_x)

    for retained_scan in range(scan_size, 0, -1):
        if retained_scan == scan_size:
            processing_scan = scan_size
        else:
            processing_scan = min(
                scan_size,
                retained_scan + 2 * psf_shape[0],
            )
        pad_scan = _axis_linear_fft_padding(processing_scan, psf_shape[0])
        padded_scan = processing_scan + sum(pad_scan)
        estimated_bytes = padded_scan * padded_y * padded_x * estimated_bytes_per_voxel
        if estimated_bytes <= memory_budget:
            return retained_scan
    return 1


@dataclass
class RlgcChunkState:
    """Store one process-local scan crop and reuse it across deconvolutions."""

    crop_scan: int | None = None

    def determine_once(
        self,
        image_shape: tuple[int, int, int],
        psf_shapes: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
        gpu_id: int = 0,
    ) -> int:
        """Determine the crop only when this state has no stored value.

        Parameters
        ----------
        image_shape : tuple[int, int, int]
            Input image dimensions in scan, camera-Y, camera-X order.
        psf_shapes : list[tuple[int, ...]] | tuple[tuple[int, ...], ...]
            Point-spread function dimensions for the processed channels.
        gpu_id : int
            CUDA device index used for reconstruction or memory estimation.

        Returns
        -------
        int
            Shared retained scan size selected from the image and channel PSF dimensions.
        """
        if self.crop_scan is None:
            self.crop_scan = determine_rlgc_crop_scan(
                image_shape,
                psf_shapes,
                gpu_id=gpu_id,
            )
        return self.crop_scan


def _split_observed_counts(
    observed: cp.ndarray, rng: cp.random.Generator
) -> cp.ndarray:
    """Return do_binosplit.m's first, unclipped half for psplit=0.5.

    MATLAB rounds halfway away from zero. Below rounded count 150 it uses a
    binomial draw; otherwise it uses a Gaussian with mean n/2 and variance n/4,
    replacing out-of-range draws by n/2. Every pixel also receives U*(J-n),
    U uniform on [0,1), including negative rounding remainders. Both dummy
    distributions and the uniform draw are generated every call, as in MATLAB.
    The caller forms J-first BEFORE independently clipping the two halves.

    Parameters
    ----------
    observed : array-like
        Nonnegative counts including the read-noise variance shift, when used.
    rng : numpy.random.Generator or cupy.random.Generator
        Random generator on the same backend as observed.

    Returns
    -------
    array-like
        Unclipped first half, including the signed fractional remainder.
    """
    xp = cp.get_array_module(observed)
    whole = xp.floor(observed)
    counts = (whole + (observed - whole >= 0.5)).astype(xp.int64)
    high_counts = counts >= 150
    low_n = xp.where(high_counts, 1, counts)
    first_low = rng.binomial(low_n, p=0.5).astype(xp.float32)
    high_n = xp.where(high_counts, counts, 1).astype(xp.float32)
    first_high = high_n * 0.5 + xp.sqrt(high_n * 0.25) * rng.standard_normal(
        observed.shape, dtype=xp.float32
    )
    first_high = xp.where(
        (first_high >= high_n) | (first_high < 0), high_n * 0.5, first_high
    )
    first = xp.where(high_counts, first_high, first_low)
    return first + rng.random(observed.shape, dtype=xp.float32) * (
        observed - counts.astype(xp.float32)
    )


def _read_noise_variance(read_noise: float) -> np.float32:
    """Convert calibrated RMS noise to the variance used by gc_deconv.m.

    Parameters
    ----------
    read_noise : float
        RMS read noise in calibrated image units.

    Returns
    -------
    float
        Variance used in the MATLAB shifted-count approximation.

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
    """Apply gc_deconv.m's shifted split, half-variance subtraction and clipping.

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
        Two independently clipped, variance-corrected half images.
    """
    xp = cp.get_array_module(observed)
    shifted = observed if variance == 0 else observed + variance
    first = _split_observed_counts(shifted, rng)
    second = shifted - first
    return xp.maximum(first - 0.5 * variance, 0), xp.maximum(second - 0.5 * variance, 0)


def _calibrated_observations(image: np.ndarray) -> np.ndarray:
    """Retain finite nonnegative calibrated estimates without quantization loss.

    Parameters
    ----------
    image : array-like
        Calibrated detector values before solver preparation.

    Returns
    -------
    numpy.ndarray
        Finite, nonnegative float32 observations.
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
        otf = self.otf.conj() if adjoint else self.otf
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


def _iterate_rlgc(
    observed: Array,
    shape: tuple[int, int, int],
    forward: Callable,
    adjoint: Callable,
    smooth: Callable,
    region: tuple[slice, slice, slice],
    *,
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
        Unpadded acquired-pixel region used for KL and stopping metrics.
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
        Enable the reference two-half KL stopping comparison.
    read_noise : float
        RMS read noise in calibrated observation units.
    initial_estimate : numpy.ndarray or None
        Initial object; None selects the constant mean observation.
    logger : logging.Logger or None
        Destination for solver diagnostics.
    log_prefix : str
        Context prepended to diagnostic messages.
    callback : callable or None
        Observer receiving the iteration index and completed estimate.

    Returns
    -------
    array-like
        Final nonnegative reconstruction on the selected backend.
    """
    xp = cp.get_array_module(observed)
    variance = _read_noise_variance(read_noise)
    sensitivity = xp.maximum(adjoint(xp.ones_like(observed)), xp.float32(1e-12))
    recon = (
        xp.full(shape, xp.mean(observed), dtype=xp.float32)
        if initial_estimate is None
        else xp.asarray(initial_estimate).copy()
    )
    rng = xp.random.default_rng(rng_seed) if gradient_consensus else None
    use_kl = bool(use_kl and gradient_consensus)
    scratch = xp.empty(observed[region].shape, dtype=xp.float32) if use_kl else None
    previous_kl = (np.inf, np.inf)
    completed = 0
    reason = "max_iterations"
    while max_iterations is None or completed < max_iterations:
        predicted = forward(recon)
        if gradient_consensus:
            split1, split2 = _split_calibrated_counts(observed, rng, variance)
            if use_kl:
                scores = split_kl_scores(
                    predicted[region], split1[region], split2[region], scratch
                )
                if scores[0] > previous_kl[0] and scores[1] > previous_kl[1]:
                    reason = "split_kl"
                    break
                previous_kl = scores
            denominator = xp.maximum(predicted + variance, xp.float32(1e-12))
            half1 = adjoint(2 * (split1 + variance) / denominator) / sensitivity
            half2 = adjoint(2 * (split2 + variance) / denominator) / sensitivity
            consensus = smooth((half1 - 1) * (half2 - 1), sensitivity)
            ratio = adjoint((observed + variance) / denominator) / sensitivity
            xp.maximum(ratio, 0, out=ratio)
            updated = (
                filter_update(recon, ratio, consensus)
                if xp is cp
                else np.where(consensus < 0, recon, recon * ratio)
            )
            fraction, delta, reason = _gc_stopping_metrics(
                recon, updated, consensus, limit, max_delta
            )
        else:
            ratio = (
                adjoint(observed / xp.maximum(predicted, xp.float32(1e-12)))
                / sensitivity
            )
            xp.maximum(ratio, 0, out=ratio)
            updated = recon * ratio
            fraction = 1.0
            delta = float(
                xp.max(xp.abs(updated - recon)) / xp.maximum(xp.max(updated), 1e-12)
            )
            reason = "relative_change" if delta < max_delta else None
        recon = updated
        completed += 1
        if callback is not None:
            callback(completed, recon)
        if logger is not None:
            logger.info(
                "%siteration=%d updated_fraction=%.6g max_relative_delta=%.6g",
                log_prefix,
                completed,
                fraction,
                delta,
            )
        if reason is not None:
            break
    else:
        reason = "max_iterations"
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
) -> np.ndarray:
    """Deconvolve calibrated data with the MATLAB RLGC update and stopping rules.

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
        Reference accepted-voxel fraction and maximum-relative-change thresholds.
    max_iterations : int or None
        Optional safety cap. None uses reference GC stopping without a cap.
        Ordinary RL requires an explicit cap or a positive max_delta.
    rng_seed : int or None
        Seed for the reference random halves, shared across all FFT tiles.
    detector_response : numpy.ndarray or None
        Positive YX sensitivity in the measurement operator, not input division.
    use_kl : bool
        Enable the reference both-halves prediction-to-split KL stopping rule.
    gradient_consensus : bool
        False selects ordinary RL for controlled comparisons using the same optics.
    initial_estimate : numpy.ndarray or None
        Positive reconstruction-grid initialization. None uses mean observations.
    callback : callable or None
        Receives each completed iteration number and its device array. Copy any
        state that must outlive the callback; it does not control convergence.
    storage : str
        Backend for reconstruction state, gpu or host; FFTs always execute on CUDA.
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
        if array.ndim != 3 or not array.size or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite nonempty 2D or 3D array")
    if (
        np.any(psf < 0)
        or psf.sum(dtype=np.float64) <= 0
        or any(n % 2 == 0 for n in psf.shape)
    ):
        raise ValueError("psf must be nonnegative with positive sum and odd dimensions")
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
    if storage not in ("gpu", "host"):
        raise ValueError("storage must be gpu or host")
    xp = cp if storage == "gpu" else np
    response = _detector_response(detector_response, image.shape)
    image = _calibrated_observations(image)
    factor = int(scan_upsample_factor)
    fine_shape = ((image.shape[0] - 1) * factor + 1, *image.shape[1:])
    try:
        with cp.cuda.Device(gpu_id):
            pads = _linear_fft_pad_width(fine_shape, psf.shape, pad_yx=pad_yx)
            padded_shape = tuple(
                n + sum(p) for n, p in zip(fine_shape, pads, strict=True)
            )
            core = _observed_region_slices(padded_shape, pads)
            convolve = _Convolution(psf, padded_shape, crop_scan, normalize_psf)
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
                return blur(scattered, adjoint=True)

            def smooth(product: Array, sensitivity: Array) -> Array:
                """Apply optical consensus smoothing for the selected geometry.

                Parameters
                ----------
                product : array-like
                    Product of the two relative half-image update directions.
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
                                convolve.otf * convolve.otf.conj(),
                                padded_shape,
                            )
                    else:
                        # Retain padded H*H smoothing: cropping between these
                        # convolutions would change the operator at the boundary.
                        padded = xp.zeros(padded_shape, dtype=xp.float32)
                        padded[core] = product
                        if xp is cp and convolve.tile_shape == convolve.shape:
                            return fft_conv(
                                padded, convolve.otf * convolve.otf.conj(), padded_shape
                            )[core].copy()
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
            recon = _iterate_rlgc(
                observed,
                shape,
                forward,
                adjoint,
                smooth,
                region,
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
            )
            result = recon[core] if boundary == "reflect" else recon
            return cp.asnumpy(result) if xp is cp else result.copy()
    finally:
        if release_memory:
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
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, cp.cuda.memory.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "oom" in message


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
    verbose: int = 1,
    release_memory: bool = True,
    logger: logging.Logger | None = None,
    log_prefix: str = "",
    on_successful_crop_scan: Callable[[int], None] | None = None,
    fallback_step_scan: int | None = None,
    detector_response: np.ndarray | None = None,
    read_noise: float = 0.0,
    **solver_options: object,
) -> np.ndarray:
    """Deconvolve calibrated data with the single RLGC engine and FFT tiling.

    The solver first attempts the requested ``crop_scan``. If CUDA memory
    allocation fails, it retries with smaller scan-axis chunks. Every solver
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
        Requested retained scan-axis tile size. If None, determine a crop from
        current free GPU memory and this PSF. Values at least as large as the
        scan-axis length select full-frame processing.
    safe_mode : bool, default=True
        Retained for compatibility. The reference KL rule always requires both
        halves to worsen; this wrapper option has no effect.
    limit : float, default=0.001
        Minimum fraction of pixels updated per iteration before early stopping.
    max_delta : float, default=0.001
        Stop when the largest relative update falls below this threshold.
    rng_seed : int or None, default=42
        Seed for the per-iteration 50:50 data split. Every FFT tile shares the global
        split and convergence history.
    normalize_psf : bool, default=True
        If True, normalize the padded PSF to unit sum before deconvolution.
    verbose : int, default=1
        If at least 1, show a progress bar over scan-axis tiles.
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

    solver_options : keyword arguments
        Numerical parameters forwarded to the single engine: ``use_kl`` enables
        reference KL stopping; ``max_iterations`` is an optional safety cap;
        ``scan_upsample_factor`` and ``boundary="zero"`` select acquired-plane
        reconstruction; ``initial_estimate`` overrides mean-data initialization;
        ``gradient_consensus=False`` selects ordinary RL. ``callback`` observes
        completed iterates. ``storage`` chooses GPU or host state; FFTs use CUDA
        in both cases. ``pad_yx`` controls symmetric camera-axis padding.

    Returns
    -------
    numpy.ndarray
        Deconvolved image as float32.
    """
    image_arr = np.asarray(image)
    if image_arr.ndim == 2:
        image_scan = 1
    elif image_arr.ndim == 3:
        image_scan = image_arr.shape[0]
    else:
        raise ValueError(f"Expected a 2D or 3D image, got shape {image_arr.shape}")

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
        if image_arr.ndim == 2:
            crop_scan = 1
        else:
            crop_scan = determine_rlgc_crop_scan(
                (image_scan, *image_arr.shape[-2:]),
                [tuple(int(size) for size in np.asarray(psf).shape)],
                gpu_id=gpu_id,
            )
    if fallback_step_scan is not None and fallback_step_scan < 1:
        raise ValueError("fallback_step_scan must be at least 1")
    min_crop_scan = 1
    attempted_crop_scan = min(int(crop_scan), int(image_scan))
    solver_options.setdefault(
        "storage", "host" if attempted_crop_scan < image_scan else "gpu"
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
            output = _rlgc_once(
                image=image_arr,
                psf=psf,
                gpu_id=gpu_id,
                crop_scan=None
                if attempted_crop_scan >= image_scan
                else attempted_crop_scan,
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
                **solver_options,
            )
            if on_successful_crop_scan is not None:
                on_successful_crop_scan(attempted_crop_scan)
            if logger is not None and logger.isEnabledFor(logging.INFO):
                logger.info(
                    "%ssuccessful_rlgc_crop_scan=%d requested_crop_scan=%d",
                    f"{log_prefix} " if log_prefix else "",
                    attempted_crop_scan,
                    crop_scan,
                )
            return output[0] if original_ndim == 2 else output
        except Exception as exc:
            if not _is_gpu_memory_error(exc):
                raise

            clear_rlgc_caches(clear_memory_pool=True)
            already_on_host = solver_options["storage"] == "host"
            solver_options["storage"] = "host"
            if attempted_crop_scan == min_crop_scan:
                if not already_on_host:
                    continue
                raise RuntimeError(
                    "RLGC failed due to GPU memory constraints even at the "
                    f"minimum scan-axis crop size {attempted_crop_scan}."
                ) from exc
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
