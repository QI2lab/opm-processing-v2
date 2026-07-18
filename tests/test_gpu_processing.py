"""Correctness tests for the CUDA image-processing paths.

These tests use small synthetic microscopy volumes, but they are not smoke
tests: every test compares GPU output with an independent result or checks a
quantitative reconstruction/registration property.

Set ``OPM_REQUIRE_GPU=1`` in a GPU CI job so an unavailable GPU stack fails the
suite instead of skipping it.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy import ndimage


pytestmark = pytest.mark.gpu


def _direct_circular_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Small, deliberately direct reference for FFT circular convolution."""
    result = np.zeros_like(image, dtype=np.float64)
    for kernel_index in np.argwhere(kernel != 0):
        index = tuple(int(value) for value in kernel_index)
        result += float(kernel[index]) * np.roll(
            image,
            shift=index,
            axis=(0, 1, 2),
        )
    return result


def _ssim_nearest_reference(
    image_a: np.ndarray,
    image_b: np.ndarray,
    win_size: int,
) -> float:
    """NumPy/SciPy reference matching the CUDA kernel's nearest-edge boxes."""
    if image_a.ndim == 2:
        image_a = image_a[None]
        image_b = image_b[None]

    image_a = image_a.astype(np.float32)
    image_b = image_b.astype(np.float32)
    box_size = (win_size,) * 3
    mean_a = ndimage.uniform_filter(image_a, size=box_size, mode="nearest")
    mean_b = ndimage.uniform_filter(image_b, size=box_size, mode="nearest")
    mean_aa = ndimage.uniform_filter(image_a * image_a, size=box_size, mode="nearest")
    mean_bb = ndimage.uniform_filter(image_b * image_b, size=box_size, mode="nearest")
    mean_ab = ndimage.uniform_filter(image_a * image_b, size=box_size, mode="nearest")

    sample_count = win_size**3
    covariance_scale = sample_count / (sample_count - 1)
    variance_a = (mean_aa - mean_a * mean_a) * covariance_scale
    variance_b = (mean_bb - mean_b * mean_b) * covariance_scale
    covariance = (mean_ab - mean_a * mean_b) * covariance_scale
    data_range = float(image_a.max() - image_a.min())
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = (
        (2 * mean_a * mean_b + c1) * (2 * covariance + c2)
        / (
            (mean_a * mean_a + mean_b * mean_b + c1)
            * (variance_a + variance_b + c2)
        )
    )
    return float(np.mean(score))


def test_fft_convolution_matches_direct_reference(cupy_gpu):
    """Exercise cuFFT and validate all voxels against direct convolution."""
    cp = cupy_gpu
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    rng = np.random.default_rng(8)
    image = rng.normal(size=(6, 8, 10)).astype(np.float32)
    psf = np.zeros((3, 3, 3), dtype=np.float32)
    psf[1, 1, 1] = 5
    psf[0, 1, 1] = 2
    psf[1, 2, 1] = 1

    image_gpu = cp.asarray(image)
    padded_psf_gpu = rlgc.pad_psf(cp.asarray(psf), image.shape)
    transfer = cp.fft.rfftn(padded_psf_gpu)
    actual_gpu = rlgc.fft_conv(image_gpu, transfer, image.shape)
    assert isinstance(actual_gpu, cp.ndarray)
    assert actual_gpu.device.id == cp.cuda.Device().id

    padded_psf = cp.asnumpy(padded_psf_gpu)
    expected = _direct_circular_convolution(image, padded_psf)
    np.testing.assert_allclose(cp.asnumpy(actual_gpu), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(padded_psf.sum(), 1.0, atol=1e-6)
    rlgc.clear_rlgc_caches()


@pytest.mark.parametrize("shape", [(31, 37), (7, 23, 29)])
def test_custom_cuda_ssim_matches_scipy_reference(cupy_gpu, shape):
    """Validate the custom RawModule kernels for both 2D and 3D images."""
    cp = cupy_gpu
    ssim_module = importlib.import_module("opm_processing.imageprocessing.ssim_cuda")
    rng = np.random.default_rng(14)
    reference = rng.random(shape, dtype=np.float32)
    comparison = np.clip(
        reference * np.float32(0.91)
        + rng.normal(0, 0.035, shape).astype(np.float32),
        0,
        1,
    )

    score = ssim_module.structural_similarity_cupy_sep_shared(
        cp.asarray(reference),
        cp.asarray(comparison),
        win_size=5,
    )
    expected = _ssim_nearest_reference(reference, comparison, win_size=5)

    assert np.isfinite(score)
    # The CUDA kernel uses naive float32 window sums while SciPy uses a running
    # separable filter, so cancellation in the variance terms differs slightly.
    np.testing.assert_allclose(score, expected, rtol=3e-4, atol=3e-4)
    identical = ssim_module.structural_similarity_cupy_sep_shared(
        cp.asarray(reference), cp.asarray(reference), win_size=5
    )
    np.testing.assert_allclose(identical, 1.0, atol=2e-6)


def test_gpu_hot_pixel_replacement_matches_synthetic_sample(cupy_gpu):
    """Exercise cupyx median filtering on a representative camera defect."""
    cp = cupy_gpu
    utils = importlib.import_module("opm_processing.imageprocessing.utils")
    if not utils.CUPY_AVIALABLE or utils.xp is not cp:
        pytest.fail("imageprocessing.utils silently selected its CPU backend")

    yy, xx = np.mgrid[:21, :25]
    clean_plane = (100 + 2 * yy + 3 * xx).astype(np.uint16)
    sample = np.stack((clean_plane, clean_plane + 20))
    sample[:, 10, 12] = 60_000
    noise_map = np.zeros((21, 25), dtype=np.float32)
    noise_map[10, 12] = 1000

    corrected = utils.replace_hot_pixels(noise_map, sample, threshold=375)

    expected = sample.copy()
    # The 3x3 median includes the defective high pixel, so the fifth sorted
    # neighborhood value is one count above the clean linear-ramp center.
    expected[:, 10, 12] = [157, 177]
    np.testing.assert_array_equal(corrected, expected)


def test_gpu_registration_recovers_known_3d_translation(cupy_gpu):
    """Run cuCIM registration plus CUDA SSIM on a translated sample volume."""
    cp = cupy_gpu
    try:
        import cucim  # noqa: F401
    except ImportError:
        pytest.fail("cuCIM is unavailable; install the project's gpu extra")

    tilefusion = importlib.import_module("opm_processing.imageprocessing.tilefusion")
    if not tilefusion.USING_GPU or tilefusion.xp is not cp:
        pytest.fail("TileFusion silently selected its CPU registration backend")

    rng = np.random.default_rng(22)
    fixed = ndimage.gaussian_filter(
        rng.normal(size=(11, 39, 43)).astype(np.float32),
        sigma=(0.8, 1.2, 1.2),
    )
    applied_shift = np.array((1.0, -2.0, 3.0), dtype=np.float32)
    moving = ndimage.shift(
        fixed,
        shift=applied_shift,
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    unregistered_score = tilefusion._ssim(
        cp.asarray(fixed), cp.asarray(moving), win_size=5
    )

    recovered, score = tilefusion.TileFusion.register_and_score(
        fixed, moving, win_size=5
    )

    np.testing.assert_allclose(recovered, -applied_shift, atol=0.15)
    assert score > unregistered_score


def test_rlgc_gpu_deconvolution_improves_synthetic_point_sample(cupy_gpu):
    """End-to-end GPU deconvolution must improve recovery of point emitters."""
    del cupy_gpu  # The fixture has already proved that device 0 executes CUDA.
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    rng = np.random.default_rng(31)

    zz, yy, xx = np.mgrid[-2:3, -3:4, -3:4]
    psf = np.exp(-(zz**2 / 1.1**2 + yy**2 / 2.0**2 + xx**2 / 2.0**2) / 2)
    psf = (psf / psf.sum()).astype(np.float32)
    truth = np.zeros((9, 35, 37), dtype=np.float32)
    truth[2, 9, 10] = 1800
    truth[4, 18, 27] = 2400
    truth[6, 26, 17] = 2100
    noiseless = ndimage.convolve(truth, psf, mode="reflect")
    observed = rng.poisson(noiseless + 0.25).astype(np.float32)

    restored = rlgc.rlgc(
        observed,
        psf,
        gpu_id=0,
        rng_seed=17,
        limit=0.02,
        max_delta=0.005,
        release_memory=False,
    )

    assert restored.shape == truth.shape
    assert restored.dtype == np.float32
    assert np.isfinite(restored).all()
    assert restored.min() >= -1e-4

    def scale_invariant_rmse(candidate):
        scale = float(np.vdot(candidate, truth) / np.vdot(candidate, candidate))
        return float(np.sqrt(np.mean((candidate * scale - truth) ** 2)))

    assert scale_invariant_rmse(restored) < 0.85 * scale_invariant_rmse(observed)
    rlgc.clear_rlgc_caches(clear_memory_pool=True)
