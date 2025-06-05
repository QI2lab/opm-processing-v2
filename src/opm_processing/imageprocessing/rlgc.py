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

# Custom CUDA kernel provided for gradient consensus
filter_update_kernel = ElementwiseKernel(
    'float32 recon, float32 HTratio, float32 consensus_map',
    'float32 out',
    '''
    bool skip = consensus_map < 0;
    out = skip ? recon : recon * HTratio;
    out = out < 0 ? 0 : out
    ''',
    'filter_update'
)

def next_multiple_of_64(x: int) -> int:
    """Calculate next multiple of 64 for the given integer.

    Parameters
    ----------
    x: int
        value to check.

    Returns
    -------
    next_64_x: int
        next multiple of 64 above x.
    """

    next_64_x = int(np.ceil((x + 63) / 64)) * 64

    return next_64_x


def pad_y(image: cp.ndarray, bkd: int) -> tuple[cp.ndarray, int, int]:
    """Pad y-axis of 3D array by 32 (zyx order).

    Parameters
    ----------
    image: cp.ndarray
        3D image to pad.


    Returns
    -------
    padded_image: cp.ndarray
        padded 3D image
    pad_y_before: int
        amount of padding at beginning
    pad_y_after: int
        amount of padding at end
    """

    z, y, x = image.shape

    new_y = next_multiple_of_64(y)
    pad_y = new_y - y

    # Distribute padding evenly on both sides
    pad_y_before = pad_y // 2
    pad_y_after = pad_y - pad_y_before

    # Padding configuration for numpy.pad
    pad_width = ((0,0), (pad_y_before, pad_y_after), (0, 0))

    padded_image = cp.pad(
        (image.astype(cp.float32)-float(bkd)).clip(0,2**16-1).astype(cp.uint16), 
        pad_width, mode="reflect"
    )

    return padded_image, pad_y_before, pad_y_after

def remove_padding_y(
    padded_image: cp.ndarray, 
    pad_y_before: int, 
    pad_y_after: int
) -> cp.ndarray:
    """Removing y-axis padding of 3D array (zyx order).

    Parameters
    ----------
    padded_image: cp.ndarray
        padded 3D image
    pad_y_before: int
        amount of padding at beginning
    pad_y_after: int
        amount of padding at end


    Returns
    -------
    image: ArrayLike
        unpadded 3D image
    """

    image = padded_image[:, pad_y_before:-pad_y_after, :]

    return image

def pad_psf(psf_temp: cp.ndarray, image_shape: tuple[int, int, int]) -> cp.ndarray:
    """Pad PSF to match the image shape.
    
    Parameters
    ----------
    psf_temp: cp.ndarray
        PSF to pad.
    image_shape: tuple[int, int, int]
        shape of the image to match.
    
    Returns
    -------
    psf: cp.ndarray
        padded PSF.
    """
    
    psf = cp.zeros(image_shape,dtype=cp.float32)
    psf[:psf_temp.shape[0], :psf_temp.shape[1], :psf_temp.shape[2]] = psf_temp
    for axis, axis_size in enumerate(psf.shape):
        psf = cp.roll(psf, int(axis_size / 2), axis=axis)
    for axis, axis_size in enumerate(psf_temp.shape):
        psf = cp.roll(psf, -int(axis_size / 2), axis=axis)
    psf = cp.fft.ifftshift(psf)
    psf = psf / cp.sum(psf)
    
    return psf

def fft_conv(image: cp.ndarray, OTF: cp.ndarray, shape) -> cp.ndarray:
    return cp.fft.irfftn(cp.fft.rfftn(image) * OTF, s=shape)

def kl_div(p: cp.ndarray, q: cp.ndarray) -> float:
    """Kullback-Leibler divergence.
    
    Parameters
    ----------
    p: cp.ndarray
        first distribution
    q: cp.ndarray
        second distribution
        
    Returns
    -------
    kldiv: float
        Kullback-Leibler metric
    """
    
    p = p + 1e-4
    q = q + 1e-4
    p = p / cp.sum(p)
    q = q / cp.sum(q)
    kldiv = p * (cp.log(p) - cp.log(q))
    kldiv[cp.isnan(kldiv)] = 0
    kldiv = cp.sum(kldiv)
    return kldiv

def rlgc_biggs(
    image: np.ndarray,
    psf: np.ndarray,
    bkd: int = 0,
    otf: cp.ndarray = None,
    otfT: cp.ndarray = None
) -> np.ndarray:
    """Andrew-Biggs accelerated RLGC deconvolution.
    
    Parameters
    ----------
    image: np.ndarray
        3D image to be deconvolved.
    psf: np.ndarray
        point spread function (PSF) to use for deconvolution.
    bkd: int
        background value to subtract from the image.
    otf: cp.ndarray
        optional pre-computed OTF.
    otfT: cp.ndarray
        optional pre-computed OTF conjugate.

    Returns
    -------
    output: np.ndarray
        deconvolved image.
    """
    image_gpu, pad_y_before, pad_y_after = pad_y(cp.asarray(image, dtype=cp.float32),bkd)

    if isinstance(psf, np.ndarray) and otf is None and otfT is None:
        psf_gpu = pad_psf(cp.asarray(psf, dtype=cp.float32), image_gpu.shape)
        otf = cp.fft.rfftn(psf_gpu)
        otfT = cp.conjugate(otf)
        otfotfT = otf * otfT
        del psf_gpu
        cp.get_default_memory_pool().free_all_blocks()

    shape = image_gpu.shape
    recon = cp.full(shape, cp.mean(image_gpu), dtype=cp.float32)
    previous_recon = cp.empty_like(recon)
    previous_recon[:] = recon

    split1 = cp.empty_like(recon)
    split2 = cp.empty_like(recon)

    Hu = cp.empty_like(recon)
    Hu_safe = cp.empty_like(recon)

    HTratio = cp.empty_like(recon)
    HTratio1 = cp.empty_like(recon)
    HTratio2 = cp.empty_like(recon)

    consensus_map = cp.zeros_like(recon)
    g1 = cp.zeros_like(recon)
    g2 = cp.zeros_like(recon)
    recon_next = cp.empty_like(recon)

    prev_kld1 = np.inf
    prev_kld2 = np.inf
    num_iters = 0
    start_time = timeit.default_timer()

    while True:
        iter_start_time = timeit.default_timer()

        split1[:] = rng.binomial(image_gpu.astype(cp.int64), p=0.5).astype(cp.float32)
        cp.subtract(image_gpu, split1, out=split2)

        if num_iters >= 2:
            numerator = cp.sum(g1 * g2)
            denominator = cp.sum(g2 * g2) + 1e-12
            alpha = numerator / denominator
            alpha = cp.clip(alpha, 0.0, 1.0)
        else:
            alpha = 0.0

        recon_next[:] = recon + alpha * (recon - previous_recon)

        Hu[:] = fft_conv(recon_next, otf, shape)

        kld1 = kl_div(Hu, split1)
        kld2 = kl_div(Hu, split2)

        if (kld1 > prev_kld1) or (kld2 > prev_kld2) or (kld1 < 1e-2) or (kld2 < 1e-2):
            recon[:] = previous_recon
            if DEBUG:
                total_time = timeit.default_timer() - start_time
                print(
                    f"Optimum result obtained after {num_iters - 1} iterations "
                
                    f"in {total_time:.1f} seconds."
                )
            break

        prev_kld1 = kld1
        prev_kld2 = kld2

        previous_recon[:], recon[:] = recon, recon_next

        cp.add(Hu, 1e-12, out=Hu_safe)

        HTratio1[:] = fft_conv(cp.divide(split1, 0.5 * Hu_safe), otfT, shape)
        HTratio2[:] = fft_conv(cp.divide(split2, 0.5 * Hu_safe), otfT, shape)
        HTratio[:] = HTratio1 + HTratio2

        consensus_map[:] = fft_conv((HTratio1 - 1) * (HTratio2 - 1), otfotfT, recon.shape)
        recon_next[:] = filter_update_kernel(recon, HTratio, consensus_map)

        g2[:], g1[:] = g1, recon_next - recon
        recon[:] = recon_next

        num_iters += 1

        if DEBUG:
            calc_time = timeit.default_timer() - iter_start_time
            print(
                f"Iteration {num_iters:03d} completed in {calc_time:.3f}s. "
                f"KLDs: {kld1:.4f} (split1), {kld2:.4f} (split2)."
            )

        cp.get_default_memory_pool().free_all_blocks()

    recon = cp.clip(recon, 0, 2**16 - 1).astype(cp.uint16)
    recon = remove_padding_y(recon, pad_y_before, pad_y_after)

    cp.get_default_memory_pool().free_all_blocks()

    return cp.asnumpy(recon)

def chunked_rlgc(
    image: np.ndarray, 
    psf: np.ndarray, 
    scan_chunk_size: int = 384,
    scan_overlap_size: int = 64,
    bkd: int = 10
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
    
    output = np.zeros_like(image)
    
    crop_size = (scan_chunk_size, image.shape[-2], image.shape[-1])
    overlap = (scan_overlap_size, 0, 0)
    slices = Slicer(image, crop_size=crop_size, overlap=overlap)
    
    for crop, source, destination in tqdm(slices,desc="decon chunk:",leave=False):
        
        crop_array = rlgc_biggs(crop,psf,bkd)
        cp.get_default_memory_pool().free_all_blocks()
        output[destination] = crop_array[source]
        
    return output
