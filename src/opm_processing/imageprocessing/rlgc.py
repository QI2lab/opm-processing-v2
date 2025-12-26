import gc
import timeit
from typing import Tuple

import cupy as cp
import numpy as np
from cupy import ElementwiseKernel
from ryomen import Slicer

DEBUG = False

# -----------------------------------------------------------------------------#
# CUDA kernel: multiplicative RL step gated by consensus (reference-accurate)
# -----------------------------------------------------------------------------#
filter_update_ba = ElementwiseKernel(
    "float32 recon, float32 HTratio, float32 consensus_map",
    "float32 out",
    r"""
    bool skip = consensus_map < 0;
    out = skip ? recon : recon * HTratio;
    out = out < 0 ? 0 : out;
    """,
    "filter_update_ba",
)

# -----------------------------------------------------------------------------#
# FFT work-buffer cache (performance)
# -----------------------------------------------------------------------------#
_fft_cache: dict[Tuple[int, int, int], Tuple[cp.ndarray, cp.ndarray]] = {}


def next_gpu_fft_size(x: int) -> int:
    """
    Return the smallest FFT-friendly size ≥ ``x`` with prime factors in {2, 3}.

    Parameters
    ----------
    x : int
        Minimum desired length.

    Returns
    -------
    int
        Next 2–3–smooth length ≥ ``x``.
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


def pad_axis_to_fft(
    image: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, int, int]:
    """
    Pad one axis of a 3D array to the next 2–3–smooth length (ZYX order).

    Padding is symmetric and uses ``mode="reflect"`` to reduce boundary artifacts.

    Parameters
    ----------
    image : numpy.ndarray
        3D image to pad.
    axis : int
        Axis index to pad (0=z, 1=y, 2=x).

    Returns
    -------
    padded_image : numpy.ndarray
        Padded image.
    pad_before : int
        Padding added at the start of ``axis``.
    pad_after : int
        Padding added at the end of ``axis``.
    """
    if image.ndim != 3:
        raise ValueError(f"pad_axis_to_fft expects a 3D array, got ndim={image.ndim}.")

    length = image.shape[axis]
    new_len = next_gpu_fft_size(length)
    pad_amt = new_len - length
    pad_before = pad_amt // 2
    pad_after = pad_amt - pad_before

    pad_width = [(0, 0), (0, 0), (0, 0)]
    pad_width[axis] = (pad_before, pad_after)

    padded_image = np.pad(image, pad_width=tuple(pad_width), mode="reflect")
    return padded_image, pad_before, pad_after


def pad_zyx_to_fft(
    image: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    Pad Z, Y, X to FFT-friendly 2–3–smooth lengths using reflect padding.

    Parameters
    ----------
    image : numpy.ndarray
        3D image to pad, shaped (z, y, x).

    Returns
    -------
    padded_image : numpy.ndarray
        Padded image.
    pad_z : tuple of int
        (pad_before, pad_after) for Z.
    pad_y : tuple of int
        (pad_before, pad_after) for Y.
    pad_x : tuple of int
        (pad_before, pad_after) for X.
    """
    padded, pz0, pz1 = pad_axis_to_fft(image, axis=0)
    padded, py0, py1 = pad_axis_to_fft(padded, axis=1)
    padded, px0, px1 = pad_axis_to_fft(padded, axis=2)
    return padded, (pz0, pz1), (py0, py1), (px0, px1)


def remove_padding_axis(
    padded_image: np.ndarray | cp.ndarray,
    axis: int,
    pad_before: int,
    pad_after: int,
) -> np.ndarray | cp.ndarray:
    """
    Remove symmetric padding from one axis.

    Parameters
    ----------
    padded_image : numpy.ndarray or cupy.ndarray
        Padded array.
    axis : int
        Axis to unpad (0=z, 1=y, 2=x).
    pad_before : int
        Padding at the start of the axis.
    pad_after : int
        Padding at the end of the axis.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        Unpadded array.
    """
    if pad_before == 0 and pad_after == 0:
        return padded_image

    slc = [slice(None), slice(None), slice(None)]
    end = -pad_after if pad_after > 0 else None
    slc[axis] = slice(pad_before, end)
    return padded_image[tuple(slc)]


def make_feather_weight_z(shape: tuple[int, int, int], feather_px: int = 64) -> np.ndarray:
    """
    Create a feathered weight mask using a cosine taper on Z only.

    Y and X are uniform. Feather taper width is explicitly specified in pixels.

    Parameters
    ----------
    shape : tuple of int
        Crop shape as (z, y, x).
    feather_px : int
        Number of pixels to taper at each Z edge.

    Returns
    -------
    numpy.ndarray
        Feather mask of shape (z, y, x), values in [0, 1].
    """
    z, y, x = shape
    weight_z = np.ones(z, dtype=np.float32)
    if feather_px > 0:
        feather_px = min(feather_px, z // 2)
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 2 * feather_px, dtype=np.float32)))
        weight_z[:feather_px] = ramp[:feather_px]
        weight_z[-feather_px:] = ramp[feather_px:]

    weight = weight_z[:, None, None]
    return np.broadcast_to(weight, shape).astype(np.float32)


def pad_psf(psf_temp: cp.ndarray, image_shape: tuple[int, int, int]) -> cp.ndarray:
    """
    Pad and center a PSF to match the target image shape; normalize to unit sum.

    Parameters
    ----------
    psf_temp : cupy.ndarray
        Original PSF (Z, Y, X).
    image_shape : tuple of int
        Target shape (Z, Y, X).

    Returns
    -------
    cupy.ndarray
        Padded, centered, nonnegative PSF normalized to unit sum.
    """
    psf = cp.zeros(image_shape, dtype=cp.float32)
    psf[: psf_temp.shape[0], : psf_temp.shape[1], : psf_temp.shape[2]] = psf_temp

    # Center the PSF: roll padded field by +N//2 then undo by psf_temp//2
    for axis, axis_size in enumerate(psf.shape):
        psf = cp.roll(psf, axis_size // 2, axis=axis)
    for axis, axis_size in enumerate(psf_temp.shape):
        psf = cp.roll(psf, -(axis_size // 2), axis=axis)

    # Put the "impulse" at the origin for FFT-based convolution
    psf = cp.fft.ifftshift(psf)

    s = cp.sum(psf)
    psf = psf / (s if s != 0 else 1.0)
    return psf.astype(cp.float32)


def fft_conv(image: cp.ndarray, H: cp.ndarray, shape: tuple[int, int, int]) -> cp.ndarray:
    """
    Linear convolution via FFT with cached buffers (no clipping).

    This computes ``irfftn(rfftn(image) * H, s=shape)`` with preallocated
    work buffers. No clipping is applied here—this matches the reference
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
    if shape not in _fft_cache:
        z, y, x = shape
        freq_shape = (z, y, x // 2 + 1)
        fft_buf = cp.empty(freq_shape, dtype=cp.complex64)
        ifft_buf = cp.empty(shape, dtype=cp.float32)
        _fft_cache[shape] = (fft_buf, ifft_buf)

    fft_buf, ifft_buf = _fft_cache[shape]
    fft_buf[...] = cp.fft.rfftn(image)
    fft_buf[...] *= H
    ifft_buf[...] = cp.fft.irfftn(fft_buf, s=shape)
    return ifft_buf


def kl_div(p: cp.ndarray, q: cp.ndarray) -> float:
    """
    Compute Kullback–Leibler divergence between two distributions.

    Parameters
    ----------
    p : cupy.ndarray
        First distribution (nonnegative).
    q : cupy.ndarray
        Second distribution (nonnegative).

    Returns
    -------
    float
        Sum over all elements of ``p * (log(p) - log(q))``, with NaNs set to 0.
    """
    p = p + 1e-4
    q = q + 1e-4
    p = p / cp.sum(p)
    q = q / cp.sum(q)
    kldiv = p * (cp.log(p) - cp.log(q))
    kldiv[cp.isnan(kldiv)] = 0
    kldiv = cp.sum(kldiv)
    return kldiv


def rlgc_biggs_ba(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    otf: cp.ndarray | None = None,
    otfT: cp.ndarray | None = None,
    safe_mode: bool = True,
    init_value: float = 1,
    limit: float = 0.1,
    max_delta: float = 0.01,
) -> np.ndarray:
    """
    Biggs–Andrews accelerated Richardson–Lucy Gradient Consensus.

    Parameters
    ----------
    image : numpy.ndarray
        3D image (Z, Y, X) to be deconvolved.
    psf : numpy.ndarray
        3D point-spread function. If ``otf`` and ``otfT`` are None, this PSF
        will be padded and transformed to form the OTF internally.
    gpu_id : int, default=0
        Which GPU to use.
    otf, otfT : cupy.ndarray, optional
        Precomputed OTF and its conjugate in RFFTN layout.
    safe_mode : bool, default=True
        If True, stop when EITHER split KLD increases (play-it-safe).
        If False, stop only when BOTH split KLDs increase.
    init_value : float, default=1
        Constant initializer value for the reconstruction.
    limit : float, default=0.01
        Minimum fraction of pixels that must be updated per iteration before
        early stopping is triggered.
    max_delta : float, default=0.01
        Maximum allowed relative update magnitude before early stopping is
        triggered.

    Returns
    -------
    numpy.ndarray
        Deconvolved 3D image (float32).
    """
    cp.cuda.Device(gpu_id).use()
    rng = cp.random.default_rng(42)

    # Ensure 3D inputs
    if psf.ndim == 2:
        psf = np.expand_dims(psf, axis=0)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    if image.ndim != 3:
        raise ValueError(f"rlgc_biggs_ba expects a 3D image, got ndim={image.ndim}.")

    # Pad all axes to FFT-friendly sizes (Z, Y, X)
    image_padded, (pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1) = pad_zyx_to_fft(image)
    image_gpu = cp.asarray(image_padded, dtype=cp.float32)

    # OTFs
    if (otf is None) or (otfT is None):
        psf_gpu = pad_psf(cp.asarray(psf, dtype=cp.float32), image_gpu.shape)
        otf = cp.fft.rfftn(psf_gpu)
        otfT = cp.conjugate(otf)
        del psf_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        otfT = cp.conjugate(otf) if otfT is None else otfT

    otfotfT = otf * otfT
    shape = image_gpu.shape
    z, y, x = shape
    num_pixels = z * y * x

    recon = cp.full((z, y, x), cp.float32(init_value), dtype=cp.float32)
    previous_recon = recon.copy()

    recon_next = cp.empty_like(recon)
    Hu = cp.empty_like(recon)

    g1 = cp.zeros_like(recon)
    g2 = cp.zeros_like(recon)

    prev_kld1 = np.inf
    prev_kld2 = np.inf
    num_iters = 0
    if DEBUG:
        start_time = timeit.default_timer()

    while True:
        if DEBUG:
            iter_start_time = timeit.default_timer()

        split1 = rng.binomial(image_gpu.astype(cp.int64), p=0.5).astype(cp.float32)
        split2 = image_gpu - split1

        if num_iters >= 1:
            numerator = cp.sum(g1 * g2)
            denominator = cp.sum(g2 * g2)
            alpha = numerator / denominator
            alpha = cp.clip(alpha, 0.0, 1.0)
            if cp.isnan(alpha):
                alpha = 0.0
            alpha = float(alpha)
            alpha = 0.0
        else:
            alpha = 0.0

        y_vec = recon + alpha * (recon - previous_recon)

        Hu[...] = fft_conv(y_vec, otf, shape)

        kldim = kl_div(Hu, image_gpu)
        kld1 = kl_div(Hu, split1)
        kld2 = kl_div(Hu, split2)

        if safe_mode:
            if (kld1 > prev_kld1) or (kld2 > prev_kld2) or (kld1 < 1e-4) or (kld2 < 1e-4):
                recon[...] = previous_recon
                if DEBUG:
                    total_time = timeit.default_timer() - start_time
                    print(f"Optimum after {num_iters - 1} iters in {total_time:.1f} s.")
                break
        else:
            if ((kld1 > prev_kld1) and (kld2 > prev_kld2)) or (kld1 < 1e-4) or (kld2 < 1e-4):
                recon[...] = previous_recon
                if DEBUG:
                    total_time = timeit.default_timer() - start_time
                    print(f"Optimum after {num_iters - 1} iters in {total_time:.1f} s.")
                break

        prev_kld1 = kld1
        prev_kld2 = kld2

        previous_recon[...] = recon
        recon[...] = y_vec

        eps = 1e-12
        HTratio1 = fft_conv(cp.divide(split1, 0.5 * (Hu + eps), dtype=cp.float32), otfT, shape)
        HTratio2 = fft_conv(cp.divide(split2, 0.5 * (Hu + eps), dtype=cp.float32), otfT, shape)
        HTratio = HTratio1 + HTratio2

        consensus_map = fft_conv((HTratio1 - 1.0) * (HTratio2 - 1.0), otfotfT, recon.shape)

        filter_update_ba(recon, HTratio, consensus_map, recon_next)

        g2[...] = g1
        g1[...] = recon_next - y_vec

        recon[...] = recon_next

        num_updated = num_pixels - cp.sum(consensus_map < 0)
        max_relative_delta = cp.max((recon - previous_recon) / cp.max(recon))

        num_iters += 1
        if DEBUG:
            calc_time = timeit.default_timer() - iter_start_time
            min_HTratio = cp.min(HTratio)
            max_HTratio = cp.max(HTratio)
            max_relative_delta_dbg = cp.max((recon - previous_recon) / cp.max(recon))
            print(
                f"Iteration {num_iters:03d} completed in {calc_time:.2f}s. "
                f"KLDs: {kldim:.6f} (image), {kld1:.6f} (split1), {kld2:.6f} (split2). "
                f"Update range: {min_HTratio:.3f} to {max_HTratio:.3f}. "
                f"Largest relative delta = {max_relative_delta_dbg:.5f}."
            )

        del split1, split2, HTratio1, HTratio2, HTratio, consensus_map

        if num_updated / num_pixels < limit:
            if DEBUG:
                print("Hit limit")
            break

        if max_relative_delta < max_delta:
            if DEBUG:
                print("Hit max delta")
            break

        if max_relative_delta < 5.0 / cp.max(image_gpu):
            if DEBUG:
                print("Hit auto delta")
            break

    recon = cp.maximum(recon, 0.0)

    # Unpad back to original (Z, Y, X)
    recon = remove_padding_axis(recon, axis=0, pad_before=pad_z0, pad_after=pad_z1)
    recon = remove_padding_axis(recon, axis=1, pad_before=pad_y0, pad_after=pad_y1)
    recon = remove_padding_axis(recon, axis=2, pad_before=pad_x0, pad_after=pad_x1)

    recon_cpu = cp.asnumpy(recon).astype(np.float32)

    del g1, g2, recon, previous_recon, Hu, otf, otfT, otfotfT, image_gpu
    gc.collect()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return recon_cpu


def chunked_rlgc(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    crop_z: int = 512,
    overlap_z: int = 128,
    safe_mode: bool = True,
    verbose: int = 0,
) -> np.ndarray:
    """
    Chunked RLGC deconvolution with Z-slab tiling and feathered blending.

    This variant is intended for skewed-scanning microscopes where Z is the
    largest axis. The volume is tiled along Z only; Y and X are processed
    full-frame within each slab.

    Parameters
    ----------
    image : numpy.ndarray
        3D image to be deconvolved, shaped (z, y, x).
    psf : numpy.ndarray
        Point-spread function (PSF) to use for deconvolution.
    gpu_id : int, default=0
        Which GPU to use.
    crop_z : int, default=512
        Slab size in Z.
    overlap_z : int, default=128
        Overlap width in pixels between slabs (for feathering).
    safe_mode : bool, default=True
        RLGC stopping: play-it-safe if True.
    verbose : int, default=0
        If ≥ 1, show a progress bar over slabs.

    Returns
    -------
    numpy.ndarray
        Deconvolved image (float32).
    """
    if image.ndim != 3:
        raise ValueError(f"chunked_rlgc expects a 3D image, got ndim={image.ndim}.")

    cp.cuda.Device(gpu_id).use()
    cp.fft._cache.PlanCache(memsize=0)

    z, y, x = image.shape

    # Full-frame path if tiling not needed
    if crop_z >= z:
        output = rlgc_biggs_ba(
            image,
            psf,
            gpu_id,
            safe_mode=safe_mode,
            init_value=float(np.median(image)),
        )
    else:
        init_value = float(np.mean(image))
        output_sum = np.zeros_like(image, dtype=np.float32)
        output_weight = np.zeros_like(image, dtype=np.float32)

        crop_size = (crop_z, y, x)
        overlap = (overlap_z, 0, 0)
        slices = Slicer(image, crop_size=crop_size, overlap=overlap, pad=True)

        if verbose >= 1:
            from rich.progress import track

            iterator = track(
                enumerate(slices),
                description="Z slabs",
                total=len(slices),
                transient=True,
            )
        else:
            iterator = enumerate(slices)

        for _, (slab, source, destination) in iterator:
            slab_array = rlgc_biggs_ba(
                slab,
                psf,
                gpu_id,
                safe_mode=safe_mode,
                init_value=init_value,
            )

            # Resolve slab edge status in Z to decide feathering
            z_slice = source[0]

            def resolve_indexer(s: slice | int | type(Ellipsis), dim: int) -> tuple[int, int]: # type: ignore
                """
                Resolve an indexer (slice/int/Ellipsis) into [start, stop) bounds.

                Parameters
                ----------
                s : slice or int or Ellipsis
                    Indexer for a single axis.
                dim : int
                    Axis length.

                Returns
                -------
                tuple of int
                    (start, stop) bounds in the axis coordinate system.
                """
                if s is Ellipsis:
                    return 0, dim
                if isinstance(s, int):
                    # Single-index selection
                    if s < 0:
                        s = dim + s
                    return s, s + 1
                if not isinstance(s, slice):
                    raise TypeError(f"Unsupported indexer type: {type(s)!r} ({s!r})")

                start = 0 if s.start is None else s.start
                stop = dim if s.stop is None else s.stop

                # Normalize negative bounds
                if start < 0:
                    start = dim + start
                if stop < 0:
                    stop = dim + stop

                return start, stop

            z_start, z_stop = resolve_indexer(z_slice, slab.shape[0])
            is_z_edge = (z_start == 0) or (z_stop == slab.shape[0])

            if is_z_edge:
                feather_weight = np.ones_like(slab_array, dtype=np.float32)
            else:
                feather_weight = make_feather_weight_z(slab.shape, feather_px=overlap_z)

            weighted_slab = slab_array * feather_weight
            weighted_sub = weighted_slab[source]
            weight_sub = feather_weight[source]

            output_sum[destination] += weighted_sub
            output_weight[destination] += weight_sub

        del feather_weight, weighted_slab, weighted_sub, weight_sub
        gc.collect()

        nonzero = output_weight > 0
        output = np.zeros_like(output_sum, dtype=output_sum.dtype)
        output[nonzero] = output_sum[nonzero] / output_weight[nonzero]

        del output_sum, output_weight, nonzero
        gc.collect()

    _fft_cache.clear()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()

    return output
