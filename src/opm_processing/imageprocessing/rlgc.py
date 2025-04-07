"""
Richardson-Lucy using gradient consesus deconvolution. 

Original idea and code: James Manton and Andrew York

Optimized code here: Douglas Shepherd
"""

import cupy as cp
import numpy as np

def pad_psf(psf: cp.ndarray,image_shape: list[int,int,int]) -> cp.ndarray:
    """Pad PSF to size of image
    
    Parameters
    ----------
    psf: cp.ndarray
        point spread function
    image_shape: list[int,int,int]
        shape of image to deconvolve
        
    Returns
    -------
    psf_padded: cp.ndarray
        point spread function padded to shape of image
    """
    
    padded_psf = cp.zeros(image_shape, dtype=cp.float32)
    insert_slices = tuple(slice(0, s) for s in psf.shape)
    padded_psf[insert_slices] = psf
    padded_psf = cp.fft.ifftshift(padded_psf)
    padded_psf = padded_psf / cp.sum(padded_psf)

    return padded_psf

def fft_conv(image: cp.ndarray, OTF: cp.ndarray) -> cp.ndarray:
    """Convolution using FFT
    
    Parameters
    ----------
    image: cp.ndarray
        real space image to convolve
    OTF: cp.ndarray
        fourier space transfer function to convolve
        
    Return
    ------
    convolved_image: cp.ndarray
        convolved image
    """
    
    return cp.real(cp.fft.ifftn(cp.fft.fftn(image) * OTF))

def kl_div(p: cp.ndarray, q: cp.ndarray) -> float:
    """Kullbeck-Liebler divergence (KL div).
    
    Parameters
    ----------
    p: cp.ndarray
        first distribution
    q: cp.ndarray
        second distribution
        
    Returns
    -------
    kldiv: float
        KL-div metric
    """
    
    eps = 1e-6  # smaller epsilon for numeric stability
    p = (p + eps) / cp.sum(p)
    q = (q + eps) / cp.sum(q)
    kldiv = cp.sum(p * (cp.log(p) - cp.log(q)))
    return float(cp.asnumpy(kldiv))

def rlgc(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Richardson-Lucy deconvolution by gradient concesus.
    
    Parameters
    ----------
    image: np.ndarray
        3D image volume to deconvolve
    psf: np.ndarray
        Point spread function for the data
        
    Returns
    -------
    recon: np.ndarray
        Deconvolved image
    """
    
    # Load data and PSF onto GPU
    image_gpu = cp.array(image, dtype=cp.float32)
    psf_gpu = pad_psf(cp.array(psf, dtype=cp.float32))

    # Calculate OTF and transpose
    otf = cp.fft.fftn(psf_gpu)
    otfT = cp.conjugate(otf)
    del psf, psf_gpu

    # Get dimensions of data
    num_z = image_gpu.shape[0]
    num_y = image_gpu.shape[1]
    num_x = image_gpu.shape[2]

    # Calculate Richardson-Lucy iterations
    HTones = fft_conv(cp.ones_like(image_gpu), otfT)
    recon = cp.mean(image_gpu) * cp.ones((num_z, num_y, num_x), dtype=cp.float32)
    previous_recon = recon

    prev_kld1 = np.inf
    prev_kld2 = np.inf
    split1 = cp.empty_like(image_gpu, dtype=cp.float32)
    split2 = cp.empty_like(image_gpu, dtype=cp.float32)
    
    # Split recorded image into 50:50 images
    # In the reference implementation, this split is done every loop
    # Test only doing it once to speed up
    image_gpu_int = image_gpu.astype(cp.int64)
    split1[:] = cp.rng.binomial(image_gpu_int, p=0.5).astype(cp.float32)    
    split2[:] = (image - split1).astype(cp.float32)
    
    del image_gpu_int
    cp.get_default_memory_pool().free_all_blocks()
    
    while True:

        # Calculate prediction
        Hu = fft_conv(recon, otf)

        # Calculate KL divergences and stop iterations if both have increased
        kld1, kld2 = kl_div(Hu, split1), kl_div(Hu, split2)
        if kld1 > prev_kld1 and kld2 > prev_kld2:
            recon = previous_recon  # revert
            break

        prev_kld1, prev_kld2 = kld1, kld2

        previous_recon[:] = recon  # reuse buffer, no reallocation

        Hu_safe = Hu + 1e-12

        # Compute update ratios efficiently
        common_ratio = image_gpu / Hu_safe
        split_ratio1 = split1 / (0.5 * Hu_safe)
        split_ratio2 = split2 / (0.5 * Hu_safe)

        del split1, split2  # explicitly free memory

        HTratio_full = fft_conv(common_ratio, otfT) / HTones
        HTratio1 = fft_conv(split_ratio1, otfT) / HTones
        HTratio2 = fft_conv(split_ratio2, otfT) / HTones

        del split_ratio1, split_ratio2, common_ratio, Hu_safe, Hu

        # In-place conditional update
        should_update = (HTratio1 - 1) * (HTratio2 - 1) >= 0
        HTratio_full[~should_update] = 1.0

        recon = recon * HTratio_full

        # Free temporary arrays immediately
        del HTratio_full, HTratio1, HTratio2, should_update
        
    cp.get_default_memory_pool().free_all_blocks()

    recon = cp.clip(recon.clip,0,2**16-1).astype(cp.uint16)

    return cp.asnumpy(recon)