"""
Richardson-Lucy Gradient Consensus (RLGC) deconvolution (Manton-style core).

Original idea for Gradient Consensus deconvolution:
James Manton and Andrew York, https://zenodo.org/records/10278919

The multiplicative update and gradient consensus follow the local
expansion-processing implementation. OPM uses geometry-specific padding,
data-derived initialization and same-split stopping. Validation is recorded
in docs/deconvolution_optimization_audit.md. That local implementation cites:
https://colab.research.google.com/drive/1mfVNSCaYHz1g56g92xBkIoa8190XNJpJ
"""

import gc
import logging
import timeit
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

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
    """Return the smallest cuFFT-friendly size at least as large as ``x``.

    Parameters
    ----------
    x : int
        Minimum desired length.

    Returns
    -------
    int
        Next length whose prime factors are in 2, 3, 5, and 7.
    """
    if x <= 1:
        return 1
    n = x
    while True:
        m = n
        for factor in (2, 3, 5, 7):
            while (m % factor) == 0:
                m //= factor
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


def remove_padding_zyx(
    padded_image: cp.ndarray | np.ndarray,
    pad_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> cp.ndarray | np.ndarray:
    """Remove reflected boundary and FFT padding from a reconstructed volume.

    Parameters
    ----------
    padded_image : cupy.ndarray or numpy.ndarray
        Padded image in Z, Y, X order.
    pad_width : tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        Reflected boundary and FFT padding widths for each axis.

    Returns
    -------
    cupy.ndarray or numpy.ndarray
        View of the input with padding removed.
    """
    slices = []
    for axis, (pad_before, pad_after) in enumerate(pad_width):
        start = pad_before
        stop = padded_image.shape[axis] - pad_after if pad_after > 0 else None
        slices.append(slice(start, stop))
    return padded_image[tuple(slices)]


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
    """Convolve using the transform output as the frequency workspace.

    This computes ``irfftn(rfftn(image) * H, s=shape)`` without an additional
    frequency buffer or copy. CuPy reuses FFT plans. No clipping is applied.

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
    return cp.fft.irfftn(spectrum, s=shape).astype(cp.float32, copy=False)


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
    probability_p = p + 1e-4
    probability_q = q + 1e-4
    probability_p = probability_p / cp.sum(probability_p)
    probability_q = probability_q / cp.sum(probability_q)
    scratch[...] = probability_p * (cp.log(probability_p) - cp.log(probability_q))
    scratch[cp.isnan(scratch)] = 0
    return float(cp.sum(scratch))


def split_kl_changes(
    predicted: cp.ndarray,
    previous_prediction: cp.ndarray,
    split1: cp.ndarray,
    split2: cp.ndarray,
    scratch: cp.ndarray,
) -> tuple[float, float]:
    """Compare two predictions against the same complementary observations.

    Compute D(split || predicted) - D(split || previous_prediction) for each
    half. The split entropy cancels, so shared prediction logs are evaluated
    once. Positive changes indicate a worse estimate. Normalizations and
    weighted reductions use float64; the logarithm workspace remains float32.
    Negative FFT roundoff is clipped only for this stopping statistic.

    Parameters
    ----------
    predicted : cupy.ndarray
        Current nonnegative prediction on the measured grid.
    previous_prediction : cupy.ndarray
        Previous prediction on the same grid.
    split1 : cupy.ndarray
        First fresh photon-count half.
    split2 : cupy.ndarray
        Complementary fresh photon-count half.
    scratch : cupy.ndarray
        Float32 workspace with the same shape as the predictions.

    Returns
    -------
    tuple[float, float]
        Changes in normalized observation-to-prediction KLD for both halves.
    """
    current = cp.maximum(predicted, 0) + 1e-4
    previous = cp.maximum(previous_prediction, 0) + 1e-4
    new_mass = cp.sum(current, dtype=cp.float64)
    old_mass = cp.sum(previous, dtype=cp.float64)
    scratch[...] = cp.log(previous / current) + cp.log(new_mass / old_mass)
    return tuple(
        float(
            cp.sum((split + 1e-4) * scratch, dtype=cp.float64)
            / cp.sum(split + 1e-4, dtype=cp.float64)
        )
        for split in (split1, split2)
    )


def _child_log_prefix(base_prefix: str, suffix: str) -> str:
    """Append one structured suffix to an RLGC log prefix.

    Parameters
    ----------
    base_prefix : str
        Existing structured log prefix, or an empty string.
    suffix : str
        Suffix to append as a separate token.

    Returns
    -------
    str
        Combined log prefix. If ``base_prefix`` is empty, returns ``suffix``.
    """
    return suffix if not base_prefix else f"{base_prefix} {suffix}"


def _resolve_tiled_axis_geometry(
    requested_crop: int,
    image_size: int,
    psf_support: int,
    axis_name: str,
) -> tuple[int, int]:
    """Resolve retained crop size and discarded processing halo for one axis.

    Parameters
    ----------
    requested_crop : int
        Requested retained crop size along the axis.
    image_size : int
        Full image size along the axis.
    psf_support : int
        PSF support length along the axis.
    axis_name : str
        Name used in validation error messages.

    Returns
    -------
    tuple[int, int]
        Retained tile size and hidden processing halo for the axis.
    """
    if requested_crop <= 0:
        raise ValueError(f"{axis_name} must be greater than 0 for tiled 3D RLGC.")

    retained_size = min(int(requested_crop), int(image_size))
    if retained_size >= image_size:
        return retained_size, 0

    tile_pad = int(psf_support)
    return retained_size, tile_pad


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

    # An isolated-pool measurement on the 219x243x490 workload peaked at
    # 24.2 float32-equivalent buffers per padded voxel. Use 26 before applying
    # memory_fraction to cover object state, OTFs, FFT workspaces, retained
    # predictions, normalization and binomial/reduction temporaries.
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

    def remember_successful_crop(self, crop_scan: int) -> None:
        """Retain a successful fallback crop for all later solver calls.

        Parameters
        ----------
        crop_scan : int
            Retained scan-plane count used for chunked deconvolution.
        """
        self.crop_scan = int(crop_scan)


def _axis_retained_bounds(retained_size: int, image_size: int) -> list[tuple[int, int]]:
    """Build non-overlapping retained tile bounds that exactly cover one axis.

    Parameters
    ----------
    retained_size : int
        Number of output samples retained from each tile along the axis.
    image_size : int
        Full image size along the axis.

    Returns
    -------
    list[tuple[int, int]]
        Inclusive-exclusive retained bounds covering ``[0, image_size)``.
    """
    if retained_size <= 0:
        raise ValueError("retained_size must be greater than 0.")
    bounds = []
    start = 0
    while start < image_size:
        stop = min(start + retained_size, image_size)
        bounds.append((start, stop))
        start = stop
    return bounds


def central_psf_plane(psf: np.ndarray) -> np.ndarray:
    """Return a normalized 2D central plane from a 2D or skewed 3D PSF.

    Parameters
    ----------
    psf : numpy.ndarray
        A YX PSF or a skewed ZYX PSF. For an even Z extent, index ``Z // 2``
        is used consistently with the PSF-centering convention in this module.

    Returns
    -------
    numpy.ndarray
        Nonnegative, unit-sum float32 PSF in YX order.
    """
    psf_array = np.asarray(psf, dtype=np.float32)
    if psf_array.ndim == 3:
        psf_array = psf_array[psf_array.shape[0] // 2]
    elif psf_array.ndim != 2:
        raise ValueError(f"Expected a 2D or 3D PSF, got shape {psf_array.shape}")
    if not np.all(np.isfinite(psf_array)):
        raise ValueError("PSF values must be finite")
    psf_array = np.maximum(psf_array, 0)
    psf_sum = float(np.sum(psf_array, dtype=np.float64))
    if not psf_sum > 0:
        raise ValueError("The central PSF plane must contain positive signal")
    return (psf_array / psf_sum).astype(np.float32, copy=False)


def _split_observed_counts(
    observed: cp.ndarray,
    rng: cp.random.Generator,
    *,
    assign_fractional_remainder: bool = True,
) -> cp.ndarray:
    """Draw a binomial half, optionally assigning fractional weighted counts.

    Whole counts follow the usual binomial split. Assign each fractional
    remainder entirely to either half with equal probability, instead of
    always putting it in the second half. The complementary half is
    ``observed - result``, preserving the measured data. For fractional data
    this is a weighted-count approximation, not an exact Poisson noise model.
    Integer-only inputs retain the original binomial samples and RNG sequence.
    Both native and undersampled RLGC use weighted residual assignment.

    Parameters
    ----------
    observed : cp.ndarray
        Nonnegative measured photon counts split into independent observations.
    rng : cp.random.Generator
        Random generator used for reproducible count splitting.
    assign_fractional_remainder : bool, default=True
        Randomly assign fractional weighted counts to either half. If False,
        return only integer binomial draws; the complementary half contains
        every fractional remainder, matching the local reference.

    Returns
    -------
    cupy.ndarray
        One random float32 count split; subtract it from observed to obtain the other split.
    """
    counts = observed.astype(cp.int64)
    split = rng.binomial(counts, p=0.5).astype(cp.float32)
    if not assign_fractional_remainder:
        return split
    remainder = observed - counts.astype(cp.float32)
    if bool(cp.any(remainder)):
        remainder *= rng.random(observed.shape, dtype=cp.float32) < 0.5
        split += remainder
    return split


def rlgc(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    safe_mode: bool = True,
    limit: float = 0.001,
    max_delta: float = 0.001,
    pad_yx: bool = True,
    rng_seed: int | None = 42,
    normalize_psf: bool = True,
    release_memory: bool = True,
    logger: logging.Logger | None = None,
    log_prefix: str = "",
    max_iterations: int = 100,
) -> np.ndarray:
    """
    Richardson-Lucy Gradient Consensus deconvolution.

    The multiplicative update and consensus sign rule follow the local
    expansion-processing reference. Initialization backprojects the measured
    image. Stopping compares successive predictions against the same fresh
    split, preventing changes in the split itself from triggering rollback.
    Fractional counts are split without favoring either half.

    OPM retains per-axis PSF halos and 2/3/5/7-smooth FFT padding. Symmetric
    input padding is applied once; padded values participate in the loop.

    Parameters
    ----------
    image : numpy.ndarray
        2D or 3D image to be deconvolved. 2D input is treated as a single-z
        stack internally.
    psf : numpy.ndarray
        2D or 3D point-spread function. This PSF is padded and transformed on
        the GPU to form the forward and adjoint OTFs internally.
    gpu_id : int, default=0
        Which GPU to use.
    safe_mode : bool, default=True
        Compare observation-to-prediction KLD against the same fresh halves.
        If True, stop when either half worsens; otherwise require both halves.
    limit : float, default=0.001
        Minimum fraction of pixels that must be updated per iteration before
        early stopping is triggered.
    max_delta : float, default=0.001
        Maximum allowed relative update magnitude before early stopping is
        triggered.
    pad_yx : bool, default=True
        If True, pad Y/X by the PSF support and expand them to FFT-friendly
        sizes. Z is always padded by the PSF support. Padding is removed before
        returning the result.
    rng_seed : int or None, default=42
        Seed for the per-iteration 50:50 data split. Set to None for
        nondeterministic splits.
    normalize_psf : bool, default=True
        If True, normalize the PSF to unit sum before deconvolution. Set False
        only for diagnostics that intentionally preserve PSF scale.
    release_memory : bool, default=True
        If True, release GPU memory pools after each call. Set False when
        calling in tight loops to avoid allocator thrashing.
    logger : logging.Logger or None, default=None
        Optional logger for per-iteration RLGC diagnostics.
    log_prefix : str, default=""
        Structured prefix prepended to every emitted log line.
    max_iterations : int, default=100
        Upper bound on consensus updates when stochastic stopping has not fired.

    Returns
    -------
    numpy.ndarray
        Deconvolved 3D image (float32). A 2D input is returned as a single-z
        stack.
    """
    cp.cuda.Device(gpu_id).use()
    logging_enabled = logger is not None and logger.isEnabledFor(logging.INFO)
    log_tag = f"{log_prefix} " if log_prefix else ""
    solver_start_time = timeit.default_timer() if logging_enabled else None
    cleanup_memory_pool = bool(release_memory)

    try:
        rng = cp.random.default_rng(rng_seed)

        if psf.ndim == 2:
            psf = np.expand_dims(psf, axis=0)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        pad_width = _linear_fft_pad_width(
            tuple(int(v) for v in image.shape),
            tuple(int(v) for v in psf.shape),
            pad_yx=pad_yx,
        )
        image_gpu_np = np.pad(image, pad_width, mode="symmetric")
        image_gpu = cp.asarray(image_gpu_np, dtype=cp.float32)
        del image_gpu_np
        psf_gpu = pad_psf(
            cp.asarray(psf, dtype=cp.float32), image_gpu.shape, normalize=normalize_psf
        )

        otf = cp.fft.rfftn(psf_gpu)
        otfT = cp.conjugate(otf)
        otfotfT = otf * otfT
        del psf_gpu

        num_z = image_gpu.shape[0]
        num_y = image_gpu.shape[1]
        num_x = image_gpu.shape[2]
        num_pixels = num_z * num_y * num_x
        num_iters = 0
        recon = cp.maximum(fft_conv(image_gpu, otfT, image_gpu.shape), 0)
        recon = cp.maximum(recon, cp.max(recon) * cp.float32(1e-7))
        previous_recon = recon
        previous_prediction = None
        kld_scratch = cp.empty_like(image_gpu)

        if logging_enabled:
            logger.info(
                "%ssolver_started image_shape=%s padded_shape=%s psf_shape=%s initialization=backprojection stopping=same_split safe_mode=%s pad_yx=%s",
                log_tag,
                tuple(int(v) for v in image.shape),
                tuple(int(v) for v in image_gpu.shape),
                tuple(int(v) for v in psf.shape),
                safe_mode,
                pad_yx,
            )

        while num_iters < max_iterations:
            iter_start_time = timeit.default_timer() if logging_enabled else None

            split1 = _split_observed_counts(image_gpu, rng)
            split2 = image_gpu - split1

            Hu = fft_conv(recon, otf, image_gpu.shape)

            if logging_enabled:
                kldim = _kl_div_into(Hu, image_gpu, kld_scratch)
            kld1 = kld2 = -np.inf
            if previous_prediction is not None:
                kld1, kld2 = split_kl_changes(
                    Hu, previous_prediction, split1, split2, kld_scratch
                )
            should_restore = (
                (kld1 > 0 or kld2 > 0) if safe_mode else (kld1 > 0 and kld2 > 0)
            )
            if should_restore:
                recon[...] = previous_recon
                if logging_enabled:
                    logger.info(
                        "%sstop=restore_previous_recon best_iteration=%d elapsed_s=%.2f safe_mode=%s kld_image=%.6f kld_split1_change=%.6f kld_split2_change=%.6f",
                        log_tag,
                        max(num_iters - 1, 0),
                        timeit.default_timer() - solver_start_time,
                        safe_mode,
                        float(kldim),
                        float(kld1),
                        float(kld2),
                    )
                break

            HTratio1 = fft_conv(
                cp.divide(split1, 0.5 * (Hu + 1e-12), dtype=cp.float32),
                otfT,
                image_gpu.shape,
            )
            del split1
            HTratio2 = fft_conv(
                cp.divide(split2, 0.5 * (Hu + 1e-12), dtype=cp.float32),
                otfT,
                image_gpu.shape,
            )
            del split2
            HTratio = cp.float32(0.5) * (HTratio1 + HTratio2)
            previous_prediction = Hu

            consensus_map = fft_conv(
                (HTratio1 - 1) * (HTratio2 - 1), otfotfT, recon.shape
            )

            previous_recon = recon
            recon = filter_update(recon, HTratio, consensus_map)

            num_updated = num_pixels - cp.sum(consensus_map < 0)
            recon_max = cp.maximum(cp.max(recon), cp.float32(1e-12))
            updated_fraction = float(num_updated) / float(num_pixels)
            max_relative_delta = float(
                cp.max(cp.abs(recon - previous_recon) / recon_max)
            )
            num_iters += 1
            if logging_enabled:
                min_HTratio = cp.min(HTratio)
                max_HTratio = cp.max(HTratio)
                logger.info(
                    "%siteration=%03d elapsed_s=%.2f kld_image=%.6f kld_split1_change=%.6f kld_split2_change=%.6f update_min=%.3f update_max=%.3f updated_fraction=%.5f max_relative_delta=%.5f",
                    log_tag,
                    num_iters,
                    timeit.default_timer() - iter_start_time,
                    float(kldim),
                    float(kld1),
                    float(kld2),
                    float(min_HTratio),
                    float(max_HTratio),
                    updated_fraction,
                    max_relative_delta,
                )

            del HTratio, HTratio1, HTratio2, consensus_map

            if updated_fraction < limit:
                if logging_enabled:
                    logger.info(
                        "%sstop=limit iteration=%03d updated_fraction=%.5f limit=%.5f",
                        log_tag,
                        num_iters,
                        updated_fraction,
                        limit,
                    )
                break

            if max_relative_delta < max_delta:
                if logging_enabled:
                    logger.info(
                        "%sstop=max_delta iteration=%03d max_relative_delta=%.5f threshold=%.5f",
                        log_tag,
                        num_iters,
                        max_relative_delta,
                        max_delta,
                    )
                break

        else:
            if logging_enabled:
                logger.info("%sstop=max_iterations iterations=%d", log_tag, num_iters)

        recon = remove_padding_zyx(recon, pad_width)
        recon_cpu = cp.asnumpy(recon).astype(np.float32)

        if logging_enabled:
            logger.info(
                "%ssolver_completed iterations=%d elapsed_s=%.2f output_shape=%s",
                log_tag,
                num_iters,
                timeit.default_timer() - solver_start_time,
                tuple(int(v) for v in recon_cpu.shape),
            )
        return recon_cpu
    except Exception:
        cleanup_memory_pool = True
        raise
    finally:
        if cleanup_memory_pool:
            cp.cuda.Stream.null.synchronize()
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


def _chunked_rlgc_once(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    crop_scan: int = 128,
    safe_mode: bool = True,
    limit: float = 0.001,
    max_delta: float = 0.001,
    rng_seed: int | None = 42,
    normalize_psf: bool = True,
    verbose: int = 1,
    release_memory: bool = True,
    logger: logging.Logger | None = None,
    log_prefix: str = "",
) -> np.ndarray:
    """Run one chunked RLGC attempt without GPU-memory fallback retries.

    This function performs either full-frame RLGC or scan-axis tiling for the
    requested crop size. Every tile keeps the full camera Y and X extents.

    Parameters
    ----------
    image : numpy.ndarray
        2D or 3D image to deconvolve.
    psf : numpy.ndarray
        2D or 3D point-spread function.
    gpu_id : int, default=0
        CUDA device ID to use.
    crop_scan : int, default=128
        Retained tile size along the scan axis (axis 0 of normalized ZYX).
    safe_mode : bool, default=True
        Compare observation-to-prediction KLD against the same fresh halves.
        If True, stop when either half worsens; otherwise require both halves.
    limit : float, default=0.001
        Minimum fraction of pixels updated per iteration before early stopping.
    max_delta : float, default=0.001
        Stop when the largest relative update falls below this threshold.
    rng_seed : int or None, default=42
        Seed for the per-iteration 50:50 data split.
    normalize_psf : bool, default=True
        If True, normalize the padded PSF to unit sum before deconvolution.
    verbose : int, default=1
        If at least 1, show a progress bar over scan-axis tiles.
    release_memory : bool, default=True
        If True, release CuPy memory pools and FFT caches before returning.
    logger : logging.Logger or None, default=None
        Optional logger for route and per-iteration diagnostics.
    log_prefix : str, default=""
        Structured prefix prepended to emitted log lines.

    Returns
    -------
    numpy.ndarray
        Deconvolved image as float32.
    """
    cp.cuda.Device(gpu_id).use()
    if crop_scan <= 0:
        raise ValueError("crop_scan must be greater than 0.")

    image_arr = np.asarray(image)
    original_ndim = image_arr.ndim
    if original_ndim == 2:
        image_work = image_arr[np.newaxis, ...]
    elif original_ndim == 3:
        image_work = image_arr
    else:
        raise ValueError(f"Expected a 2D or 3D image, got shape {image_arr.shape}")

    psf_arr = np.asarray(psf)
    if psf_arr.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D PSF, got shape {psf_arr.shape}")
    psf_shape = psf_arr.shape if psf_arr.ndim == 3 else (1, *psf_arr.shape)

    # Full-frame path if tiling not needed
    if crop_scan >= image_work.shape[0]:
        if logger is not None and logger.isEnabledFor(logging.INFO):
            logger.info(
                "%spath=3d_fullframe image_shape=%s psf_shape=%s",
                f"{log_prefix} " if log_prefix else "",
                tuple(int(v) for v in image_arr.shape),
                tuple(int(v) for v in psf_shape),
            )
        output = rlgc(
            image_arr,
            psf,
            gpu_id,
            safe_mode=safe_mode,
            limit=limit,
            max_delta=max_delta,
            pad_yx=True,
            rng_seed=rng_seed,
            normalize_psf=normalize_psf,
            release_memory=False,
            logger=logger,
            log_prefix=_child_log_prefix(log_prefix, "path=3d_fullframe"),
        )
        if original_ndim == 2 and output.ndim == 3 and output.shape[0] == 1:
            output = np.squeeze(output, axis=0)

    # Scan-axis tiled deconvolution with discarded processing halos. The scan
    # axis is axis 0 in the normalized ZYX volume; camera Y and X remain intact.
    # The halo is wider than a single convolution radius because RLGC is
    # iterative, so boundary influence can propagate farther than one PSF
    # half-width.
    else:
        full_shape = image_work.shape
        retained_scan, tile_pad_scan = _resolve_tiled_axis_geometry(
            crop_scan,
            full_shape[0],
            int(psf_shape[-3]),
            "crop_scan",
        )
        output = np.zeros_like(image_work, dtype=np.float32)

        retained_bounds_scan = _axis_retained_bounds(retained_scan, full_shape[0])
        num_tiles = len(retained_bounds_scan)

        if logger is not None and logger.isEnabledFor(logging.INFO):
            logger.info(
                "%spath=3d_tiled image_shape=%s psf_shape=%s retained_shape=%s processing_halo=%s num_tiles=%d",
                f"{log_prefix} " if log_prefix else "",
                tuple(int(v) for v in image_work.shape),
                tuple(int(v) for v in psf_shape),
                (retained_scan, full_shape[1], full_shape[2]),
                (tile_pad_scan, 0, 0),
                num_tiles,
            )

        if verbose >= 1 and num_tiles > 1:
            iterator = tqdm(
                enumerate(retained_bounds_scan),
                desc="Decon chunks",
                total=num_tiles,
                leave=False,
                unit="chunk",
            )
        else:
            iterator = enumerate(retained_bounds_scan)

        for tile_idx, (scan_dest_start, scan_dest_stop) in iterator:
            scan_crop_start = max(scan_dest_start - tile_pad_scan, 0)
            scan_crop_stop = min(scan_dest_stop + tile_pad_scan, full_shape[0])

            crop = image_work[scan_crop_start:scan_crop_stop, :, :]
            crop_array = rlgc(
                crop,
                psf,
                gpu_id,
                safe_mode=safe_mode,
                limit=limit,
                max_delta=max_delta,
                rng_seed=None if rng_seed is None else rng_seed + tile_idx,
                normalize_psf=normalize_psf,
                release_memory=False,
                logger=logger,
                log_prefix=_child_log_prefix(
                    log_prefix, f"path=3d_tiled tile={tile_idx:04d}"
                ),
            )

            scan_source_start = scan_dest_start - scan_crop_start
            scan_source_stop = scan_source_start + (scan_dest_stop - scan_dest_start)
            crop_sub = crop_array[scan_source_start:scan_source_stop, :, :]
            output[scan_dest_start:scan_dest_stop, :, :] = crop_sub

        del crop_sub
        gc.collect()

        if original_ndim == 2:
            output = np.squeeze(output, axis=0)

    if release_memory:
        clear_rlgc_caches(clear_memory_pool=True)

    return output


def chunked_rlgc(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    crop_scan: int | None = None,
    safe_mode: bool = True,
    limit: float = 0.001,
    max_delta: float = 0.001,
    rng_seed: int | None = 42,
    normalize_psf: bool = True,
    verbose: int = 1,
    release_memory: bool = True,
    logger: logging.Logger | None = None,
    log_prefix: str = "",
    on_successful_crop_scan: Callable[[int], None] | None = None,
    fallback_step_scan: int | None = None,
) -> np.ndarray:
    """Scan-axis chunked RLGC deconvolution with automatic fallback.

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
        Compare observation-to-prediction KLD against the same fresh halves.
        If True, stop when either half worsens; otherwise require both halves.
    limit : float, default=0.001
        Minimum fraction of pixels updated per iteration before early stopping.
    max_delta : float, default=0.001
        Stop when the largest relative update falls below this threshold.
    rng_seed : int or None, default=42
        Seed for the per-iteration 50:50 data split. Tiled calls offset this
        seed by tile index.
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

    if crop_scan is None:
        if image_arr.ndim == 2:
            crop_scan = 1
        else:
            crop_scan = determine_rlgc_crop_scan(
                tuple(int(size) for size in image_arr.shape),
                [tuple(int(size) for size in np.asarray(psf).shape)],
                gpu_id=gpu_id,
            )
    if fallback_step_scan is not None and fallback_step_scan < 1:
        raise ValueError("fallback_step_scan must be at least 1")
    min_crop_scan = 1
    attempted_crop_scan = min(int(crop_scan), int(image_scan))

    while True:
        try:
            if logger is not None and logger.isEnabledFor(logging.INFO):
                logger.info(
                    "%sattempt_rlgc_crop_scan=%d requested_crop_scan=%d",
                    f"{log_prefix} " if log_prefix else "",
                    attempted_crop_scan,
                    crop_scan,
                )
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
                verbose=verbose,
                release_memory=release_memory,
                logger=logger,
                log_prefix=log_prefix,
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
            return output
        except Exception as exc:
            if not _is_gpu_memory_error(exc):
                raise

            clear_rlgc_caches(clear_memory_pool=True)
            if attempted_crop_scan == min_crop_scan:
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


def rlgc_2d(
    image: np.ndarray,
    skewed_psf: np.ndarray,
    gpu_id: int = 0,
    safe_mode: bool = True,
    limit: float = 0.001,
    max_delta: float = 0.001,
    rng_seed: int | None = 42,
    verbose: int = 1,
    release_memory: bool = True,
    logger: logging.Logger | None = None,
    log_prefix: str = "",
) -> np.ndarray:
    """Deconvolve one acquired YX image using the central plane of its PSF.

    Non-spatial dimensions such as time, stage position, and channel must be
    iterated by the caller. A singleton Z dimension is accepted and restored
    on return so storage code can retain explicit TCZYX axes.

    Parameters
    ----------
    image : numpy.ndarray
        A YX image or singleton-Z ZYX image.
    skewed_psf : numpy.ndarray
        A YX PSF or skewed ZYX PSF. Only its central Z plane is used.
    gpu_id : int
        CUDA device index used for reconstruction or memory estimation.
    safe_mode : bool
        Roll back consensus iterations when either split likelihood worsens.
    limit : float, default=0.001
        Stop when the fraction of voxels receiving consensus updates falls below this value.
    max_delta : float, default=0.001
        Stop when the maximum relative reconstruction change falls below this value.
    rng_seed : int | None
        Seed for independent count splitting; None uses an unseeded generator.
    verbose : int
        Deconvolution diagnostic verbosity.
    release_memory : bool
        Clear shared FFT caches and unused GPU allocations on completion.
    logger : logging.Logger | None
        Optional logger for iteration diagnostics.
    log_prefix : str
        Text prepended to deconvolution diagnostics for this tile or channel.

    Returns
    -------
    numpy.ndarray
        Float32 deconvolution with the same rank as ``image``.
    """
    image_array = np.asarray(image)
    restore_z = image_array.ndim == 3 and image_array.shape[0] == 1
    if restore_z:
        image_yx = image_array[0]
    elif image_array.ndim == 2:
        image_yx = image_array
    else:
        raise ValueError(
            "2D RLGC expects a YX image or singleton-Z ZYX image; "
            f"got shape {image_array.shape}"
        )

    psf_yx = central_psf_plane(skewed_psf)
    result = chunked_rlgc(
        image=image_yx,
        psf=psf_yx,
        gpu_id=gpu_id,
        crop_scan=1,
        safe_mode=safe_mode,
        limit=limit,
        max_delta=max_delta,
        rng_seed=rng_seed,
        normalize_psf=False,
        verbose=verbose,
        release_memory=release_memory,
        logger=logger,
        log_prefix=_child_log_prefix(log_prefix, "path=2d_central_psf"),
    )
    result = np.asarray(result, dtype=np.float32)
    return result[np.newaxis] if restore_z else result
