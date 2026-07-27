"""
Richardson-Lucy Gradient Consensus (RLGC) deconvolution (Manton-style core).

Original idea for Gradient Consensus deconvolution:
James Manton and Andrew York, https://zenodo.org/records/10278919

Reference RLGC loop based on James Manton's implementation:
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

import cupy as cp  # noqa: E402
from cupy import ElementwiseKernel  # noqa: E402

# -----------------------------------------------------------------------------
# CUDA kernel: multiplicative RL step gated by consensus (reference-accurate)
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

_kld_terms = ElementwiseKernel(
    "float32 p, float32 q, float32 p_sum, float32 q_sum",
    "float32 out",
    """
    const float eps = 1e-4f;
    const float p_norm = (p + eps) / p_sum;
    const float q_norm = (q + eps) / q_sum;
    out = p_norm * (logf(p_norm) - logf(q_norm));
    """,
    "kld_terms",
)

# -----------------------------------------------------------------------------
# FFT caches (performance)
# -----------------------------------------------------------------------------
_fft_cache_3d: dict[tuple[int, int, int, int], cp.ndarray] = {}


def clear_rlgc_caches(clear_memory_pool: bool = False) -> None:
    """Clear cached FFT resources used by RLGC helper functions.

    Parameters
    ----------
    clear_memory_pool : bool, default=False
        If True, synchronize the current CUDA stream and release CuPy device
        and pinned memory pools in addition to clearing local FFT buffers and
        CuPy FFT plans.

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
    """Calculate per-axis linear FFT padding without allocating an image."""
    pad_scan = _axis_linear_fft_padding(image_shape[0], psf_shape[0])
    if pad_yx:
        pad_y = _axis_linear_fft_padding(image_shape[1], psf_shape[1])
        pad_x = _axis_linear_fft_padding(image_shape[2], psf_shape[2])
    else:
        pad_y = (0, 0)
        pad_x = (0, 0)
    return pad_scan, pad_y, pad_x


def pad_for_linear_fft(
    image: np.ndarray,
    psf_shape: tuple[int, int, int],
    pad_yx: bool = True,
) -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """Pad a 3D image for linear FFT convolution with ``ndimage(..., mode="reflect")`` edges.

    Z is always padded by the PSF support. Y/X are padded by the PSF support and
    expanded to FFT-friendly sizes only when ``pad_yx`` is True.

    Parameters
    ----------
    image : numpy.ndarray
        3D input image in Z, Y, X order.
    psf_shape : tuple[int, int, int]
        PSF shape in Z, Y, X order.
    pad_yx : bool, default=True
        If True, pad and FFT-expand Y/X. If False, only pad Z.

    Returns
    -------
    tuple[numpy.ndarray, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]
        Padded image and the per-axis padding widths needed to remove padding
        after deconvolution.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D input, got shape {image.shape!r}")

    pad_width = _linear_fft_pad_width(
        tuple(int(size) for size in image.shape),
        psf_shape,
        pad_yx=pad_yx,
    )
    padded_image = np.pad(image, pad_width, mode="symmetric")
    return padded_image, pad_width


def remove_padding_zyx(
    padded_image: cp.ndarray | np.ndarray,
    pad_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> cp.ndarray | np.ndarray:
    """Remove per-axis padding added by :func:`pad_for_linear_fft`.

    Parameters
    ----------
    padded_image : cupy.ndarray or numpy.ndarray
        Padded image in Z, Y, X order.
    pad_width : tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        Per-axis padding widths returned by :func:`pad_for_linear_fft`.

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


def _symmetric_padded_axis_indices(
    length: int,
    pad_before: int,
    pad_after: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source indices for a symmetric extension of one padded axis.

    Parameters
    ----------
    length : int
        Full padded axis length.
    pad_before : int
        Number of samples padded before the observed region.
    pad_after : int
        Number of samples padded after the observed region.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Source indices for the left and right padded regions.
    """
    observed = np.arange(pad_before, length - pad_after, dtype=np.int64)
    padded = np.pad(observed, (pad_before, pad_after), mode="symmetric")
    return padded[:pad_before], padded[length - pad_after :]


def enforce_symmetric_boundary(
    image: cp.ndarray,
    pad_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> None:
    """Constrain padded samples to be a symmetric extension of observed samples.

    Parameters
    ----------
    image : cupy.ndarray
        Padded 3D image in Z, Y, X order. The array is modified in place.
    pad_width : tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        Per-axis padding widths returned by :func:`pad_for_linear_fft`.

    Returns
    -------
    None
    """
    for axis, (pad_before, pad_after) in enumerate(pad_width):
        if pad_before == 0 and pad_after == 0:
            continue
        left_indices, right_indices = _symmetric_padded_axis_indices(
            image.shape[axis],
            pad_before,
            pad_after,
        )
        if pad_before > 0:
            destination = [slice(None)] * image.ndim
            destination[axis] = slice(0, pad_before)
            image[tuple(destination)] = cp.take(
                image,
                cp.asarray(left_indices),
                axis=axis,
            )
        if pad_after > 0:
            destination = [slice(None)] * image.ndim
            destination[axis] = slice(image.shape[axis] - pad_after, None)
            image[tuple(destination)] = cp.take(
                image,
                cp.asarray(right_indices),
                axis=axis,
            )


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
    """Linear convolution via FFT with cached buffers (no clipping).

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


def _observed_region_mask(
    shape: tuple[int, int, int],
    pad_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> cp.ndarray:
    """Build a mask that is one in the original image and zero in padding.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Full padded image shape.
    pad_width : tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        Per-axis padding widths returned by :func:`pad_for_linear_fft`.

    Returns
    -------
    cupy.ndarray
        Float32 mask with observed voxels equal to one.
    """
    mask = cp.zeros(shape, dtype=cp.float32)
    slices = []
    for axis, (pad_before, pad_after) in enumerate(pad_width):
        start = pad_before
        stop = shape[axis] - pad_after if pad_after > 0 else None
        slices.append(slice(start, stop))
    mask[tuple(slices)] = 1
    return mask


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


def kl_div(p: cp.ndarray, q: cp.ndarray, mask: cp.ndarray | None = None) -> float:
    """Compute Kullback-Leibler divergence between two distributions.

    Parameters
    ----------
    p : cupy.ndarray
        First distribution (nonnegative).
    q : cupy.ndarray
        Second distribution (nonnegative).
    mask : cupy.ndarray or None, default=None
        Optional mask selecting the observed image region. Values outside the
        mask are excluded before normalization.

    Returns
    -------
    float
        Sum over all elements of ``p * (log(p) - log(q))``, with NaNs set to 0.
    """
    eps = cp.float32(1e-4)
    if mask is not None:
        p = (p + eps) * mask
        q = (q + eps) * mask
    else:
        p = p + eps
        q = q + eps
    p = p / cp.sum(p)
    q = q / cp.sum(q)
    kldiv = p * (cp.log(p) - cp.log(q))
    kldiv[cp.isnan(kldiv)] = 0
    return float(cp.sum(kldiv))


def _kl_div_into(
    p: cp.ndarray,
    q: cp.ndarray,
    scratch: cp.ndarray,
) -> float:
    """Compute KLD using a caller-owned scratch array.

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
        Sum of the elementwise KLD terms.
    """
    eps_total = cp.float32(1e-4 * p.size)
    p_sum = cp.sum(p, dtype=cp.float32) + eps_total
    q_sum = cp.sum(q, dtype=cp.float32) + eps_total
    _kld_terms(p, q, p_sum, q_sum, scratch)
    return float(cp.sum(scratch, dtype=cp.float64))


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

    # The optimized solver peaks at approximately 18 float32-equivalent
    # buffers per padded voxel: object-space state, three OTFs, the reusable
    # FFT workspace, normalization, split data, ratios, consensus, and CuPy
    # FFT/random work areas. Keeping this derivation local avoids a global
    # hardware-specific crop setting.
    estimated_bytes_per_voxel = 18 * np.dtype(np.float32).itemsize
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
        """Determine the crop only when this state has no stored value."""
        if self.crop_scan is None:
            self.crop_scan = determine_rlgc_crop_scan(
                image_shape,
                psf_shapes,
                gpu_id=gpu_id,
            )
        return self.crop_scan

    def remember_successful_crop(self, crop_scan: int) -> None:
        """Retain a successful fallback crop for all later solver calls."""
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


def rlgc(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    safe_mode: bool = True,
    limit: float = 0.1,
    max_delta: float = 0.01,
    pad_yx: bool = True,
    rng_seed: int | None = 42,
    normalize_psf: bool = True,
    release_memory: bool = True,
    logger: logging.Logger | None = None,
    log_prefix: str = "",
) -> np.ndarray:
    """Richardson-Lucy Gradient Consensus deconvolution.

    The implementation follows the non-accelerated reference loop with
    split-KLD stopping and the consensus-gated multiplicative update.

    Parameters
    ----------
    image : numpy.ndarray
        2D or 3D image to deconvolve. 2D input is treated as a single-z stack.
    psf : numpy.ndarray
        2D or 3D point-spread function. The PSF is padded to the processing
        shape, centered using the reference RLGC convention, and transformed on
        the GPU to form the forward and adjoint OTFs.
    gpu_id : int, default=0
        CUDA device ID to use.
    safe_mode : bool, default=True
        If True, stop when either split KLD increases. If False, stop only when
        both split KLDs increase.
    limit : float, default=0.1
        Minimum fraction of pixels updated per iteration before early stopping.
    max_delta : float, default=0.01
        Stop when the largest relative update falls below this threshold.
    pad_yx : bool, default=True
        If True, pad Y/X by PSF support and expand to FFT-friendly sizes. Z is
        always padded by PSF support. Padding is removed before returning.
    rng_seed : int or None, default=42
        Seed for the per-iteration 50:50 data split. Set to None for
        nondeterministic splits.
    normalize_psf : bool, default=True
        If True, normalize the padded PSF to unit sum before deconvolution.
        Set False only for diagnostics that intentionally preserve PSF scale.
    release_memory : bool, default=True
        If True, release CuPy memory pools and FFT caches before returning. On
        exceptions, cleanup is always forced.
    logger : logging.Logger or None, default=None
        Optional logger for per-iteration diagnostics.
    log_prefix : str, default=""
        Structured prefix prepended to emitted log lines.

    Returns
    -------
    numpy.ndarray
        Deconvolved image as float32. A 2D input returns with singleton z
        removed by the caller when routed through :func:`chunked_rlgc`.
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

        image_shape = tuple(int(size) for size in image.shape)
        psf_shape = tuple(int(size) for size in psf.shape)
        pad_width = _linear_fft_pad_width(
            image_shape,
            psf_shape,
            pad_yx=pad_yx,
        )
        padded_shape = tuple(
            size + sum(axis_pad) for size, axis_pad in zip(image_shape, pad_width)
        )

        target_cache_key = (int(cp.cuda.Device().id), *padded_shape)
        stale_cache_keys = [key for key in _fft_cache_3d if key != target_cache_key]
        if stale_cache_keys:
            for key in stale_cache_keys:
                del _fft_cache_3d[key]
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()

        observed_slices = _observed_region_slices(padded_shape, pad_width)
        observed_core = cp.asarray(image, dtype=cp.float32)
        psf_gpu = pad_psf(
            cp.asarray(psf, dtype=cp.float32),
            padded_shape,
            normalize=normalize_psf,
        )

        otf = cp.fft.rfftn(psf_gpu)
        otfT = cp.conjugate(otf)
        otfotfT = otf * otfT
        del psf_gpu

        observed_support = cp.zeros(padded_shape, dtype=cp.float32)
        observed_support[observed_slices] = 1
        update_norm = fft_conv(observed_support, otfT, padded_shape)
        cp.maximum(update_norm, cp.float32(1e-6), out=update_norm)
        del observed_support

        num_pixels = int(observed_core.size)
        num_iters = 0
        prev_kld1 = np.inf
        prev_kld2 = np.inf

        recon = cp.full(
            padded_shape,
            cp.mean(observed_core),
            dtype=cp.float32,
        )
        next_recon = cp.empty_like(recon)
        kld_scratch = cp.empty_like(observed_core)

        if logging_enabled:
            logger.info(
                "%ssolver_started image_shape=%s padded_shape=%s psf_shape=%s reference_core=non_accelerated safe_mode=%s pad_yx=%s",
                log_tag,
                tuple(int(v) for v in image.shape),
                padded_shape,
                tuple(int(v) for v in psf.shape),
                safe_mode,
                pad_yx,
            )

        while True:
            iter_start_time = timeit.default_timer() if logging_enabled else None

            observed_counts = observed_core.astype(cp.int64)
            split1_core = rng.binomial(observed_counts, p=0.5)
            del observed_counts
            split1 = cp.zeros(padded_shape, dtype=cp.float32)
            split1[observed_slices] = split1_core
            del split1_core
            split2_core = observed_core - split1[observed_slices]

            Hu = fft_conv(recon, otf, padded_shape)

            predicted_core = Hu[observed_slices]
            kldim = _kl_div_into(predicted_core, observed_core, kld_scratch)
            kld1 = _kl_div_into(
                predicted_core,
                split1[observed_slices],
                kld_scratch,
            )
            kld2 = _kl_div_into(predicted_core, split2_core, kld_scratch)

            if safe_mode:
                should_restore = (kld1 > prev_kld1) or (kld2 > prev_kld2)
            else:
                should_restore = (kld1 > prev_kld1) and (kld2 > prev_kld2)
            if should_restore:
                recon = next_recon
                if logging_enabled:
                    logger.info(
                        "%sstop=restore_previous_recon best_iteration=%d elapsed_s=%.2f safe_mode=%s kld_image=%.6f kld_split1=%.6f prev_kld_split1=%.6f kld_split2=%.6f prev_kld_split2=%.6f",
                        log_tag,
                        max(num_iters - 1, 0),
                        timeit.default_timer() - solver_start_time,
                        safe_mode,
                        float(kldim),
                        float(kld1),
                        float(prev_kld1),
                        float(kld2),
                        float(prev_kld2),
                    )
                break

            prev_kld1 = kld1
            prev_kld2 = kld2

            Hu *= cp.float32(0.5)
            Hu += cp.float32(1e-12)
            cp.divide(split1, Hu, out=split1)
            HTratio1 = fft_conv(split1, otfT, padded_shape)
            HTratio1 /= update_norm
            split1.fill(0)
            cp.divide(
                split2_core,
                Hu[observed_slices],
                out=split1[observed_slices],
            )
            del split2_core
            HTratio2 = fft_conv(split1, otfT, padded_shape)
            HTratio2 /= update_norm

            cp.subtract(HTratio1, cp.float32(1), out=split1)
            cp.subtract(HTratio2, cp.float32(1), out=Hu)
            cp.multiply(split1, Hu, out=split1)
            del Hu
            consensus_map = fft_conv(split1, otfotfT, padded_shape)
            del split1

            cp.add(HTratio1, HTratio2, out=HTratio1)
            del HTratio2
            filter_update(recon, HTratio1, consensus_map, next_recon)
            enforce_symmetric_boundary(next_recon, pad_width)

            next_core = next_recon[observed_slices]
            recon_core = recon[observed_slices]
            num_updated = cp.count_nonzero(consensus_map[observed_slices] >= 0)
            recon_max = cp.maximum(cp.max(next_core), cp.float32(1e-12))
            updated_fraction = float(num_updated) / float(num_pixels)
            min_HTratio = cp.min(HTratio1)
            max_HTratio = cp.max(HTratio1)
            max_relative_delta = float(
                cp.max(cp.abs(next_core - recon_core)) / recon_max
            )
            recon, next_recon = next_recon, recon

            num_iters += 1
            if logging_enabled:
                logger.info(
                    "%siteration=%03d elapsed_s=%.2f kld_image=%.6f kld_split1=%.6f kld_split2=%.6f update_min=%.3f update_max=%.3f updated_fraction=%.5f max_relative_delta=%.5f",
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

            del HTratio1, consensus_map

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
        gc.collect()
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
    limit: float = 0.1,
    max_delta: float = 0.01,
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
        If True, stop when either split KLD increases. If False, stop only when
        both split KLDs increase.
    limit : float, default=0.1
        Minimum fraction of pixels updated per iteration before early stopping.
    max_delta : float, default=0.01
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
    limit: float = 0.1,
    max_delta: float = 0.01,
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
        If True, stop when either split KLD increases. If False, stop only when
        both split KLDs increase.
    limit : float, default=0.1
        Minimum fraction of pixels updated per iteration before early stopping.
    max_delta : float, default=0.01
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
