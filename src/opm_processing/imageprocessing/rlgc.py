"""Richardson-Lucy Gradient Consensus deconvolution.

Original idea for gradient consensus deconvolution from: James Maton and Andrew York, https://zenodo.org/records/10278919

RLGC code based on James Manton's implementation, https://colab.research.google.com/drive/1mfVNSCaYHz1g56g92xBkIoa8190XNJpJ

Biggs-Andrews acceleration based on their 1997 paper, https://doi.org/10.1364/AO.36.001766
"""

import cupy as cp
import numpy as np
from cupy import ElementwiseKernel
from ryomen import Slicer
from tqdm import tqdm
import timeit

rng = cp.random.default_rng(42)
DEBUG = False

gc_update_kernel = ElementwiseKernel(
    'float32 recon, float32 gradient, float32 consensus_map, float32 H_T_ones',
    'float32 recon_next',
    '''
    float step = recon / H_T_ones;
    float upd  = recon + gradient * step;
    if (consensus_map < 0) {
        upd = recon;
    }
    if (upd < 0) {
        upd = 0;
    }
    recon_next = upd;
    ''',
    'gc_update_kernel'
)

_fft_cache: dict[tuple[int,int,int], tuple[cp.ndarray, cp.ndarray]] = {}
_H_T_cache: dict[tuple[int,int,int], cp.ndarray] = {}

def next_multiple_of_64(x: int) -> int:
    """Determine the next multiple of 64 greater than or equal to x.
    
    Parameters
    ----------
    x: int
        The input integer to round up to the next multiple of 64.
        
    Returns
    -------
    next_64_x: int
        The next multiple of 64 that is greater than or equal to x.
    """
    next_64_x = int(np.ceil((x + 31) / 31)) * 32
    return next_64_x

def pad_y(image: cp.ndarray, bkd: int) -> tuple[cp.ndarray, int, int]:
    """Pad y-axis of 3D array by the next multiple of 64 (zyx order).

    Parameters
    ----------
    image: cp.ndarray
        3D image to pad.
    bkd: int
        Background value to subtract before padding.

    Returns
    -------
    padded_image: cp.ndarray
        Padded 3D image.
    pad_y_before: int
        Amount of padding at the start of the y-axis.
    pad_y_after: int
        Amount of padding at the end of the y-axis.
    """
    z, y, x = image.shape
    new_y = next_multiple_of_64(y)
    pad_y = new_y - y
    pad_y_before = pad_y // 2
    pad_y_after = pad_y - pad_y_before
    pad_width = ((0, 0), (pad_y_before, pad_y_after), (0, 0))
    padded_image = cp.pad(
        (image.astype(cp.float32) - float(bkd)).clip(0, 2**16 - 1).astype(cp.uint16),
        pad_width, mode="reflect"
    )
    return padded_image, pad_y_before, pad_y_after

def remove_padding_y(
    padded_image: cp.ndarray,
    pad_y_before: int,
    pad_y_after: int
) -> cp.ndarray:
    """Remove y-axis padding added by pad_y.

    Parameters
    ----------
    padded_image: cp.ndarray
        Padded 3D image.
    pad_y_before: int
        Amount of padding at the start of the y-axis.
    pad_y_after: int
        Amount of padding at the end of the y-axis.

    Returns
    -------
    image: cp.ndarray
        Unpadded 3D image.
    """
    image = padded_image[:, pad_y_before:-pad_y_after, :]
    return image

def pad_psf(psf_temp: cp.ndarray, image_shape: tuple[int, int, int]) -> cp.ndarray:
    """Pad and center a PSF to match the target image shape.

    Parameters
    ----------
    psf_temp: cp.ndarray
        Original PSF array.
    image_shape: tuple[int, int, int]
        Desired shape for the padded PSF.

    Returns
    -------
    psf: cp.ndarray
        Padded and centered PSF, normalized to unit sum.
    """
    psf = cp.zeros(image_shape, dtype=cp.float32)
    psf[:psf_temp.shape[0], :psf_temp.shape[1], :psf_temp.shape[2]] = psf_temp
    for axis, axis_size in enumerate(psf.shape):
        psf = cp.roll(psf, int(axis_size / 2), axis=axis)
    for axis, axis_size in enumerate(psf_temp.shape):
        psf = cp.roll(psf, -int(axis_size / 2), axis=axis)
    psf = cp.fft.ifftshift(psf)
    psf = psf / cp.sum(psf)
    return cp.clip(psf,a_min=1e-12, a_max=2**16-1).astype(cp.float32)

def fft_conv(image: cp.ndarray, OTF: cp.ndarray, shape) -> cp.ndarray:
    """Perform convolution via FFT: irfftn(rfftn(image) * OTF).

    Parameters
    ----------
    image: cp.ndarray
        Input image in object space.
    OTF: cp.ndarray
        Frequency-domain transfer function (rfftn of PSF).
    shape: tuple[int, int, int]
        Target shape for the inverse FFT.

    Returns
    -------
    result: cp.ndarray
        Convolved image in object space.
    """
    if shape not in _fft_cache:
        z, y, x = shape
        freq_shape = (z, y, x // 2 + 1)
        fft_buf = cp.empty(freq_shape, dtype=cp.complex64)
        ifft_buf = cp.empty(shape, dtype=cp.float32)
        _fft_cache[shape] = (fft_buf, ifft_buf)
    fft_buf, ifft_buf = _fft_cache[shape]
    fft_buf[...] = cp.fft.rfftn(image)
    fft_buf[...] *= OTF
    ifft_buf[...] = cp.fft.irfftn(fft_buf, s=shape)
    return cp.clip(ifft_buf,a_min=1e-12, a_max=2**16-1)

def kl_div(p: cp.ndarray, q: cp.ndarray) -> float:
    """Compute Kullback–Leibler divergence between two distributions.

    Parameters
    ----------
    p: cp.ndarray
        First distribution (nonnegative).
    q: cp.ndarray
        Second distribution (nonnegative).

    Returns
    -------
    kldiv: float
        Sum over all elements of p * (log(p) - log(q)), with NaNs set to zero.
    """
    p = p + 1e-4
    q = q + 1e-4
    p = p / cp.sum(p)
    q = q / cp.sum(q)
    kldiv = p * (cp.log(p) - cp.log(q))
    kldiv[cp.isnan(kldiv)] = 0
    kldiv = cp.sum(kldiv)
    return float(kldiv)

def _get_H_T_ones(otf: cp.ndarray, otfT: cp.ndarray, shape: tuple[int,int,int]) -> cp.ndarray:
    if shape in _H_T_cache:
        return _H_T_cache[shape]
    ones = cp.ones(shape, dtype=cp.float32)
    freq_shape = (shape[0], shape[1], shape[2] // 2 + 1)
    fft_buf = cp.empty(freq_shape, dtype=cp.complex64)
    ifft_buf = cp.empty(shape, dtype=cp.float32)
    fft_buf[...] = cp.fft.rfftn(ones)
    fft_buf[...] *= otf
    ifft_buf[...] = cp.fft.irfftn(fft_buf, s=shape)
    fft_buf[...] = cp.fft.rfftn(ifft_buf)
    fft_buf[...] *= otfT
    ifft_buf[...] = cp.fft.irfftn(fft_buf, s=shape)
    _H_T_cache[shape] = ifft_buf
    return cp.clip(ifft_buf,a_min=1e-12, a_max=2**16-1).astype(cp.float32)

def rlgc_biggs(
    image: np.ndarray,
    psf: np.ndarray,
    bkd: int = 0,
    otf: cp.ndarray = None,
    otfT: cp.ndarray = None
) -> np.ndarray:
    """
    Andrew–Biggs accelerated RLGC deconvolution with Gradient Consensus.

    This routine performs Richardson–Lucy + Gradient Consensus (GC)
    deconvolution on a 3D image stack, using a Biggs–Andrews momentum
    term and an early stopping criterion based on Kullback–Leibler divergence.

    Parameters
    ----------
    image: np.ndarray
        3D image (zyx) to be deconvolved.
    psf: np.ndarray
        3D point-spread function. If `otf` and `otfT` are None, this PSF
        will be padded and transformed to form the OTF internally.
    bkd: int, optional
        Constant background to subtract before deconvolution (default=0).
    otf: cp.ndarray, optional
        Precomputed rfftn of the padded PSF. If provided, `psf` is ignored.
    otfT: cp.ndarray, optional
        Conjugate of `otf`. Must match shape of `otf`.

    Returns
    -------
    output: np.ndarray
        Deconvolved 3D image, clipped to [0, 2^16-1] and cast to uint16.
    """
    if image.ndim == 3:
        image_gpu, pad_y_before, pad_y_after = pad_y(
            cp.asarray(image, dtype=cp.float32), bkd
        )
    else:
        image_gpu = cp.asarray(image, dtype=cp.float32)
        image_gpu = image_gpu[cp.newaxis, ...]
    if isinstance(psf, np.ndarray) and otf is None and otfT is None:
        psf_gpu = pad_psf(cp.asarray(psf, dtype=cp.float32), image_gpu.shape)
        otf = cp.fft.rfftn(psf_gpu)
        otfT = cp.conjugate(otf)
        del psf_gpu
        cp.get_default_memory_pool().free_all_blocks()
    otfotfT = cp.clip(cp.real(otf * otfT).astype(cp.float32),a_min=1e-12, a_max=2**16-1).astype(cp.float32)
    shape = image_gpu.shape
    z, y, x = shape
    recon = cp.full(shape, cp.mean(image_gpu), dtype=cp.float32)
    previous_recon = cp.empty_like(recon)
    previous_recon[...] = recon
    recon_next = cp.empty_like(recon)
    split1 = cp.empty_like(recon)
    split2 = cp.empty_like(recon)
    Hu = cp.empty_like(recon)
    Hu_safe = cp.empty_like(recon)
    HTratio1 = cp.empty_like(recon)
    HTratio2 = cp.empty_like(recon)
    HTratio = cp.empty_like(recon)
    consensus_map = cp.empty_like(recon)
    g1 = cp.zeros_like(recon)
    g2 = cp.zeros_like(recon)
    H_T_ones = _get_H_T_ones(otf, otfT, shape)
    prev_kld1 = np.inf
    prev_kld2 = np.inf
    num_iters = 0
    start_time = timeit.default_timer()
    while True:
        iter_start_time = timeit.default_timer()
        split1[...] = rng.binomial(image_gpu.astype(cp.int64), p=0.5).astype(cp.float32)
        cp.subtract(image_gpu, split1, out=split2)
        if num_iters >= 2:
            numerator = cp.sum(g1 * g2)
            denominator = cp.sum(g2 * g2) + 1e-12
            alpha = numerator / denominator
            alpha = cp.clip(alpha, 0.0, 1.0)
            alpha = float(alpha)
        else:
            alpha = 0.0
        temp = recon - previous_recon
        cp.multiply(temp, alpha, out=temp)
        cp.add(recon, temp, out=recon_next)
        recon, recon_next = recon_next, recon
        previous_recon[...] = recon
        Hu[...] = fft_conv(recon, otf, shape)
        kld1 = kl_div(Hu, split1)
        kld2 = kl_div(Hu, split2)
        if (kld1 > prev_kld1) or (kld2 > prev_kld2) or (kld1 < 1e-1) or (kld2 < 1e-1):
            recon[...] = previous_recon
            if DEBUG:
                total_time = timeit.default_timer() - start_time
                print(
                    f"Optimum result obtained after {num_iters - 1} iterations "
                    f"in {total_time:.1f} seconds."
                )
            break
        prev_kld1 = kld1
        prev_kld2 = kld2
        cp.add(Hu, 1e-12, out=Hu_safe)
        cp.divide(split1, Hu_safe, out=split1)
        cp.subtract(split1, 0.5, out=split1)
        HTratio1[...] = fft_conv(split1, otfT, shape)
        cp.divide(split2, Hu_safe, out=split2)
        cp.subtract(split2, 0.5, out=split2)
        HTratio2[...] = fft_conv(split2, otfT, shape)
        cp.add(HTratio1, HTratio2, out=HTratio)
        cp.multiply(HTratio1, HTratio2, out=split1)
        consensus_map[...] = fft_conv(split1, otfotfT, shape)
        gc_update_kernel(
            recon,
            HTratio,
            consensus_map,
            H_T_ones,
            recon_next
        )
        temp_g2 = recon_next - recon
        g2[...] = g1
        g1[...] = temp_g2
        recon[...] = recon_next
        num_iters += 1
        if DEBUG:
            calc_time = timeit.default_timer() - iter_start_time
            print(
                f"Iteration {num_iters:03d} completed in {calc_time:.3f}s. "
                f"KLDs: {kld1:.4f} (split1), {kld2:.4f} (split2)."
            )
    recon = cp.clip(recon, 0, 2**16 - 1).astype(cp.uint16)
    if image.ndim == 3:
        recon = remove_padding_y(recon, pad_y_before, pad_y_after)
    else:
        recon = cp.squeeze(recon)
    del recon_next, g1, g2, H_T_ones
    cp.get_default_memory_pool().free_all_blocks()
    return cp.asnumpy(recon)

def chunked_rlgc(
    image: np.ndarray, 
    psf: np.ndarray, 
    scan_chunk_size: int = 384,
    scan_overlap_size: int = 64,
    bkd: int = 0
) -> np.ndarray:
    """Chunked RLGC deconvolution.
    
    Parameters
    ----------
    image: np.ndarray
        3D image to be deconvolved.
    psf: np.ndarray
        point spread function (PSF) to use for deconvolution.
    scan_chunk_size: int
        size of the chunk to process at a time.
    scan_overlap_size: int
        size of the overlap between chunks.
    bkd: int
        background value to subtract from the image.
        
    Returns
    -------
    output: np.ndarray
        deconvolved image.
    """
    
    cp.fft._cache.PlanCache(memsize=0)
    output = np.zeros_like(image)
    crop_size = (scan_chunk_size, image.shape[-2], image.shape[-1])
    overlap = (scan_overlap_size, 0, 0)
    slices = Slicer(image, crop_size=crop_size, overlap=overlap)
    for crop, source, destination in tqdm(slices,desc="decon chunk:",leave=False):
        crop_array = rlgc_biggs(crop, psf, bkd)
        cp.get_default_memory_pool().free_all_blocks()
        output[destination] = crop_array[source]
    return output
