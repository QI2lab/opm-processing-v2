"""Richardson-Lucy Gradient Consensus deconvolution.

Original idea for gradient consensus deconvolution from:
James Maton and Andrew York, https://zenodo.org/records/10278919

RLGC code based on James Manton's implementation,
https://colab.research.google.com/drive/1mfVNSCaYHz1g56g92xBkIoa8190XNJpJ

Biggs-Andrews acceleration based on their 1997 paper,
https://doi.org/10.1364/AO.36.001766
"""

import cupy as cp
import numpy as np
from cupy import ElementwiseKernel
from ryomen import Slicer
from tqdm import tqdm
import timeit

rng = cp.random.default_rng(42)
DEBUG = False

# ───────── Vendored padding routines ─────────

def get_pad_size(
    img_shape: tuple[int, int, int],
    psf_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Compute zero-pad size on each axis to avoid circular FFT."""
    return tuple(img_shape[i] + 2 * (psf_shape[i] // 2)
                 for i in range(3))


def pad(
    img: cp.ndarray,
    paddedsize: tuple[int, int, int],
    mode: str
) -> tuple[cp.ndarray, tuple[tuple[int, int], ...]]:
    """
    Pad a CuPy array on all three axes to `paddedsize`.

    Returns
    -------
    padded : cp.ndarray
    padding : tuple of (before, after) per axis
    """
    padding = tuple(
        (int(np.ceil((paddedsize[i] - img.shape[i]) / 2)),
         int(np.floor((paddedsize[i] - img.shape[i]) / 2)))
        for i in range(3)
    )
    return cp.pad(img, padding, mode=mode), padding


def unpad(
    padded: cp.ndarray,
    imgsize: tuple[int, int, int]
) -> cp.ndarray:
    """
    Crop a padded CuPy array back down to `imgsize`.
    """
    padding = tuple(
        (int(np.ceil((padded.shape[i] - imgsize[i]) / 2)),
         int(np.floor((padded.shape[i] - imgsize[i]) / 2)))
        for i in range(3)
    )
    slices = tuple(
        slice(padding[i][0], padding[i][0] + imgsize[i])
        for i in range(3)
    )
    return padded[slices]

# ───────── End vendored padding ─────────


# Existing FFT‐cache and kernels

_fft_cache: dict[tuple[int, int, int], tuple[cp.ndarray, cp.ndarray]] = {}
_H_T_cache: dict[tuple[int, int, int], cp.ndarray] = {}

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

def fft_conv(image: cp.ndarray, OTF: cp.ndarray, shape):
    """irfftn(rfftn(image)*OTF) with no wrap-around beyond shape."""
    if shape not in _fft_cache:
        z, y, x = shape
        freq_shape = (z, y, x // 2 + 1)
        _fft_cache[shape] = (
            cp.empty(freq_shape, dtype=cp.complex64),
            cp.empty(shape,    dtype=cp.float32),
        )
    fft_buf, ifft_buf = _fft_cache[shape]
    fft_buf[...] = cp.fft.rfftn(image)
    fft_buf[...] *= OTF
    ifft_buf[...] = cp.fft.irfftn(fft_buf, s=shape)
    return cp.clip(ifft_buf, a_min=1e-12, a_max=2**16-1)

def kldiv(p: cp.ndarray, q: cp.ndarray, HTones: cp.ndarray) -> float:
    """Masked KL divergence: sum_{HTones>0} p*(log p – log q)."""
    eps = 1e-4
    p = (p + eps) * HTones
    q = (q + eps) * HTones
    p /= cp.sum(p)
    q /= cp.sum(q)
    div = p * (cp.log(p) - cp.log(q))
    div[cp.isnan(div)] = 0
    div[HTones == 0] = 0
    return float(cp.sum(div))

def _get_H_T_ones(
    otf: cp.ndarray,
    otfT: cp.ndarray,
    shape: tuple[int, int, int]
) -> cp.ndarray:
    if shape in _H_T_cache:
        return _H_T_cache[shape]
    ones = cp.ones(shape, dtype=cp.float32)
    freq_shape = (shape[0], shape[1], shape[2] // 2 + 1)
    fft_buf = cp.empty(freq_shape, dtype=cp.complex64)
    ifft_buf = cp.empty(shape, dtype=cp.float32)

    # Hᵀ(H(1))
    fft_buf[...] = cp.fft.rfftn(ones)
    fft_buf[...] *= otf
    ifft_buf[...] = cp.fft.irfftn(fft_buf, s=shape)
    fft_buf[...] = cp.fft.rfftn(ifft_buf)
    fft_buf[...] *= otfT
    ifft_buf[...] = cp.fft.irfftn(fft_buf, s=shape)

    _H_T_cache[shape] = ifft_buf
    return cp.clip(ifft_buf, a_min=1e-12, a_max=2**16-1).astype(cp.float32)


def rlgc_biggs_nc(
    image: np.ndarray,
    psf: np.ndarray,
    bkd: int = 0,
    otf: cp.ndarray = None,
    otfT: cp.ndarray = None,
    eager_mode: bool = False
) -> np.ndarray:
    """
    Andrew–Biggs accelerated RLGC deconvolution with Gradient Consensus,
    enforcing non‐circulant (zero) boundary conditions.
    """
    # 1) Move data to GPU and subtract background
    img = cp.asarray(image, dtype=cp.float32) - float(bkd)
    if img.ndim == 2:
        img = img[cp.newaxis, ...]
    original_shape = img.shape

    # 2) Zero-pad image and HTones on all axes
    pad_shape = get_pad_size(original_shape, psf.shape)
    image_gpu, _ = pad(img, pad_shape, mode='constant')
    HTones,    _ = pad(cp.ones_like(img), pad_shape, mode='constant')

    # 3) Zero-pad PSF on all axes, shift to center
    psf_gpu, _ = pad(cp.asarray(psf, dtype=cp.float32),
                     pad_shape, mode='constant')
    psf_gpu = cp.fft.ifftshift(psf_gpu)

    # 4) Build OTFs
    if otf is None or otfT is None:
        otf  = cp.fft.rfftn(psf_gpu)
        otfT = cp.conjugate(otf)
    del psf_gpu

    otfotfT = cp.real(otf * otfT).astype(cp.float32)

    # 5) Precompute stabilized Hᵀ(1)
    HT1 = fft_conv(HTones, otfT, image_gpu.shape)
    HT1[HT1 < 1e-6] = 1

    # 6) Allocate buffers
    shape = image_gpu.shape
    recon = cp.full(shape, cp.mean(image_gpu), dtype=cp.float32)
    prev_recon = recon.copy()
    split1 = cp.empty_like(recon)
    split2 = cp.empty_like(recon)
    Hu = cp.empty_like(recon)
    HTr1 = cp.empty_like(recon)
    HTr2 = cp.empty_like(recon)
    HTr  = cp.empty_like(recon)
    cons = cp.empty_like(recon)
    H_T_ones = _get_H_T_ones(otf, otfT, shape)

    prev_kld1 = np.inf
    prev_kld2 = np.inf
    stop_iter = -1

    # 7) Iterations
    for i in range(100 if False else 1000000):  # use your total_iters
        # random split
        split1[...] = rng.binomial(image_gpu.astype(cp.int64),
                                   p=0.5).astype(cp.float32)
        cp.subtract(image_gpu, split1, out=split2)

        # forward
        Hu[...] = fft_conv(recon, otf, shape)

        # KL divergences
        kld1 = kldiv(Hu,    split1, HTones)
        kld2 = kldiv(Hu,    split2, HTones)

        # early stop
        if ((not eager_mode and (kld1 > prev_kld1 and kld2 > prev_kld2))
            or (eager_mode and (kld1 > prev_kld1 or kld2 > prev_kld2))):
            stop_iter = i
            recon[...] = prev_recon
            break

        prev_kld1, prev_kld2 = kld1, kld2

        # ratio updates
        HTr1[...] = fft_conv(split1 / (Hu + 1e-12), otfT, shape) / HT1
        HTr2[...] = fft_conv(split2 / (Hu + 1e-12), otfT, shape) / HT1
        HTr[...]  = 0.5 * (HTr1 + HTr2)

        # consensus blur
        cons[...] = fft_conv((HTr1 - 1) * (HTr2 - 1),
                             otfotfT, shape) * HTones

        # update
        recon, prev_recon = (gc_update_kernel(
            recon, HTr, cons, H_T_ones, recon), recon)

    # 8) Crop back to original shape
    recon = cp.clip(recon, 0, 2**16 - 1).astype(cp.uint16)
    recon = unpad(recon, original_shape)

    cp.get_default_memory_pool().free_all_blocks()
    return cp.asnumpy(recon)


def chunked_rlgc_nc(
    image: np.ndarray,
    psf: np.ndarray,
    scan_chunk_size: int = 384,
    scan_overlap_size: int = 64,
    bkd: int = 0,
    eager_mode: bool = False
) -> np.ndarray:
    """Chunked RLGC deconvolution (wraps `rlgc_biggs`)."""
    cp.fft._cache.PlanCache(memsize=0)
    output = np.zeros_like(image)
    crop_size = (scan_chunk_size,
                 image.shape[-2],
                 image.shape[-1])
    overlap = (scan_overlap_size, 0, 0)
    slices = Slicer(image,
                    crop_size=crop_size,
                    overlap=overlap)

    for crop, src, dst in tqdm(slices,
                               desc="decon chunk:",
                               leave=False):
        out_crop = rlgc_biggs_nc(crop, psf,
                              bkd=bkd,
                              eager_mode=eager_mode)
        cp.get_default_memory_pool().free_all_blocks()
        output[dst] = out_crop[src]

    return output
