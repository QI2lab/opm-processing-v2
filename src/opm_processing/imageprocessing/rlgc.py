import cupy as cp
import numpy as np
from cupy import ElementwiseKernel
from ryomen import Slicer
from tqdm import tqdm
import timeit

rng = cp.random.default_rng(42)

DEBUG = False

# Custom CUDA kernel provided for gradient consensus
gradient_consensus = ElementwiseKernel(
    'float32 recon, float32 ratio, float32 r1, float32 r2',
    'float32 out',
    '''
    bool skip = (r1 - 1.0f)*(r2 - 1.0f) < 0;
    out = skip ? recon : recon * ratio;
    ''',
    'gradient_consensus'
)

def next_multiple_of_32(x: int) -> int:
    """Calculate next multiple of 32 for the given integer.

    Parameters
    ----------
    x: int
        value to check.

    Returns
    -------
    next_32_x: int
        next multiple of 32 above x.
    """

    next_32_x = int(np.ceil((x + 255) / 256)) * 256

    return next_32_x


def pad_y(image: cp.ndarray) -> tuple[cp.ndarray, int, int]:
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

    new_y = next_multiple_of_32(y)
    pad_y = new_y - y

    # Distribute padding evenly on both sides
    pad_y_before = pad_y // 2
    pad_y_after = pad_y - pad_y_before

    # Padding configuration for numpy.pad
    pad_width = ((0,0), (pad_y_before, pad_y_after), (0, 0))

    padded_image = cp.pad(image, pad_width, mode="reflect")

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
    p = p + 1e-4
    q = q + 1e-4
    p = p / cp.sum(p)
    q = q / cp.sum(q)
    kldiv = p * (cp.log(p) - cp.log(q))
    kldiv[cp.isnan(kldiv)] = 0
    kldiv = cp.sum(kldiv)
    return kldiv

def rlgc(
    image: np.ndarray, 
    psf: np.ndarray,
    otf: cp.ndarray = None,
    otfT: cp.ndarray = None
) -> np.ndarray:
    image_gpu, pad_y_before, pad_y_after = pad_y(
        cp.asarray(image, dtype=cp.float32)
    )
    
    if isinstance(psf, np.ndarray) and otf is None and otfT is None:
        psf_gpu = pad_psf(cp.asarray(psf, dtype=cp.float32), image_gpu.shape)
        otf = cp.fft.rfftn(psf_gpu)
        otfT = cp.conjugate(otf)
        del psf_gpu
    
    
    num_z = image_gpu.shape[0]
    num_y = image_gpu.shape[1]
    num_x = image_gpu.shape[2]

    
    recon = cp.mean(image_gpu) * cp.ones((num_z, num_y, num_x), dtype=cp.float32)
    previous_recon = recon
    
    num_iters = 0
    prev_kld1 = np.inf
    prev_kld2 = np.inf

    start_time = timeit.default_timer()
    while True:
        iter_start_time = timeit.default_timer()
        
        split1 = rng.binomial(image_gpu.astype('int64'), p=0.5)
        split2 = image_gpu - split1
        
        Hu = fft_conv(recon, otf, image_gpu.shape)
        
        kldim = kl_div(Hu, image_gpu)
        kld1 = kl_div(Hu, split1)
        kld2 = kl_div(Hu, split2)
        if ((kld1 > prev_kld1) & (kld2 > prev_kld2)):
            recon = previous_recon
            if DEBUG:
                print("Optimum result obtained after %d iterations with a total time of %1.1f seconds." % (num_iters - 1, timeit.default_timer() - start_time))
            break
        #del previous_recon
        prev_kld1 = kld1
        prev_kld2 = kld2
        
        # Calculate updates for split images and full images (H^T (d / Hu))
        HTratio1 = fft_conv(cp.divide(split1, 0.5 * (Hu + 1E-12), dtype=cp.float32), otfT, image_gpu.shape)
        #del split1
        HTratio2 = fft_conv(cp.divide(split2, 0.5 * (Hu + 1E-12), dtype=cp.float32), otfT, image_gpu.shape)
        #del split2
        HTratio = fft_conv(image_gpu / (Hu + 1E-12), otfT, image_gpu.shape)
        #del Hu

        # Save previous estimate in case KLDs increase after this iteration
        previous_recon = recon

        # Update estimate
        recon = gradient_consensus(recon, HTratio, HTratio1, HTratio2)
        min_HTratio = cp.min(HTratio)
        max_HTratio = cp.max(HTratio)
        max_relative_delta = cp.max((recon - previous_recon) / cp.max(recon))
        #del HTratio

        calc_time = timeit.default_timer() - iter_start_time
        if DEBUG:
            print("Iteration %03d completed in %1.3f s. KLDs = %1.4f (image), %1.4f (split 1), %1.4f (split 2). Update range: %1.2f to %1.2f. Largest relative delta = %1.5f." % (num_iters + 1, calc_time, kldim, kld1, kld2, min_HTratio, max_HTratio, max_relative_delta))

        num_iters = num_iters + 1

        #cp.get_default_memory_pool().free_all_blocks()

    recon = cp.clip(recon, 0, 2**16 - 1).astype(cp.uint16)
    recon = remove_padding_y(recon, pad_y_before, pad_y_after)

    return cp.asnumpy(recon)

def rlgc_biggs(
    image: np.ndarray, 
    psf: np.ndarray,
    otf: cp.ndarray = None,
    otfT: cp.ndarray = None
) -> np.ndarray:
    image_gpu, pad_y_before, pad_y_after = pad_y(
        cp.asarray(image, dtype=cp.float32)
    )

    if isinstance(psf, np.ndarray) and otf is None and otfT is None:
        psf_gpu = pad_psf(cp.asarray(psf, dtype=cp.float32), image_gpu.shape)
        otf = cp.fft.rfftn(psf_gpu)
        otfT = cp.conjugate(otf)
        del psf_gpu
        cp.get_default_memory_pool().free_all_blocks()

    recon = cp.full(image_gpu.shape, cp.mean(image_gpu), dtype=cp.float32)
    previous_recon = recon.copy()

    prev_kld1 = np.inf
    prev_kld2 = np.inf
    num_iters = 0

    recon_prev1 = recon.copy()
    G_prev1 = cp.zeros_like(recon, dtype=cp.float32)

    start_time = timeit.default_timer()

    while True:
        iter_start_time = timeit.default_timer()

        split1 = rng.binomial(image_gpu.astype(cp.int64), p=0.5).astype(cp.float32)
        split2 = image_gpu - split1

        Hu = fft_conv(recon, otf, image_gpu.shape)

        kld1 = kl_div(Hu, split1)
        kld2 = kl_div(Hu, split2)

        if (kld1 > prev_kld1) and (kld2 > prev_kld2):
            recon = previous_recon.copy()
            break

        prev_kld1 = kld1
        prev_kld2 = kld2
        previous_recon[:] = recon

        Hu_safe = Hu + 1e-12
        del Hu
        cp.get_default_memory_pool().free_all_blocks()

        HTratio1 = fft_conv(split1 / (0.5 * Hu_safe), otfT, image_gpu.shape)
        del split1
        cp.get_default_memory_pool().free_all_blocks()

        HTratio2 = fft_conv(split2 / (0.5 * Hu_safe), otfT, image_gpu.shape)
        del split2
        cp.get_default_memory_pool().free_all_blocks()

        HTratio = fft_conv(image_gpu / Hu_safe, otfT, image_gpu.shape)
        del Hu_safe
        cp.get_default_memory_pool().free_all_blocks()

        recon_next = gradient_consensus(recon, HTratio, HTratio1, HTratio2)
        del HTratio
        del HTratio1
        del HTratio2
        cp.get_default_memory_pool().free_all_blocks()

        if num_iters >= 2:
            G_current = recon_next - recon
            numerator = cp.sum(G_current * G_prev1)
            denominator = cp.sum(G_prev1 * G_prev1) + 1e-8

            lambda_factor = numerator / denominator
            lambda_factor = cp.clip(lambda_factor, 0.0, 1.0)

            recon_accel = recon_next + lambda_factor * (recon_next - recon_prev1)

            G_prev1[:] = G_current
            recon_prev1[:] = recon_next
            del G_current
            cp.get_default_memory_pool().free_all_blocks()
        else:
            recon_accel = recon_next
            G_prev1[:] = recon_next - recon
            recon_prev1[:] = recon_next

        recon[:] = cp.clip(recon_accel, 0, None)
        del recon_next
        del recon_accel
        cp.get_default_memory_pool().free_all_blocks()

        num_iters += 1

        calc_time = timeit.default_timer() - iter_start_time
        if DEBUG:
            print(
                f"Iter {num_iters:03d}, "
                f"time: {calc_time:.3f}s, "
                f"KLD1: {kld1:.4f}, "
                f"KLD2: {kld2:.4f}"
            )

    recon = cp.clip(recon, 0, 2**16 - 1).astype(cp.uint16)
    recon = remove_padding_y(recon, pad_y_before, pad_y_after)

    cp.get_default_memory_pool().free_all_blocks()

    return cp.asnumpy(recon)

def chunked_rlgc(
    image: np.ndarray, 
    psf: np.ndarray, 
    scan_chunk_size: int = 256,
    scan_overlap_size: int = 32
) -> np.ndarray:
    
    output = np.zeros_like(image)
    
    crop_size = (scan_chunk_size, image.shape[-2], image.shape[-1])
    overlap = (scan_overlap_size, 0, 0)
    slices = Slicer(image, crop_size=crop_size, overlap=overlap)
    
    for crop, source, destination in tqdm(slices,desc="decon chunk:",leave=False):
        
        crop_array = rlgc_biggs(crop,psf)
        cp.get_default_memory_pool().free_all_blocks()
        output[destination] = crop_array[source]
        
    return output