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
    """Small, deliberately direct reference for FFT circular convolution.

    Parameters
    ----------
    image : np.ndarray
        Value supplied for ``image``.
    kernel : np.ndarray
        Value supplied for ``kernel``.

    Returns
    -------
    np.ndarray
        Result produced by the callable.
    """
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
    """NumPy/SciPy reference matching the CUDA kernel's nearest-edge boxes.

    Parameters
    ----------
    image_a : np.ndarray
        Value supplied for ``image a``.
    image_b : np.ndarray
        Value supplied for ``image b``.
    win_size : int
        Value supplied for ``win size``.

    Returns
    -------
    float
        Result produced by the callable.
    """
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
        (2 * mean_a * mean_b + c1)
        * (2 * covariance + c2)
        / ((mean_a * mean_a + mean_b * mean_b + c1) * (variance_a + variance_b + c2))
    )
    return float(np.mean(score))


@pytest.mark.unit
def test_fft_convolution_matches_direct_reference(cupy_gpu):
    """Exercise cuFFT and validate all voxels against direct convolution.

    Parameters
    ----------
    cupy_gpu : object
        Value supplied for ``cupy gpu``.

    Returns
    -------
    None
        No value is returned.
    """
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

    # Construct the impulse response independently of the padding under test.
    expected_kernel = np.zeros(image.shape, dtype=np.float64)
    expected_kernel[0, 0, 0] = 5 / 8
    expected_kernel[-1, 0, 0] = 2 / 8
    expected_kernel[0, 1, 0] = 1 / 8
    np.testing.assert_allclose(cp.asnumpy(padded_psf_gpu), expected_kernel)
    expected = _direct_circular_convolution(image, expected_kernel)
    np.testing.assert_allclose(cp.asnumpy(actual_gpu), expected, rtol=2e-5, atol=2e-5)
    rlgc.clear_rlgc_caches()


@pytest.mark.unit
@pytest.mark.parametrize(
    "p_values,q_values",
    [([0, 1, 3, 20], [4, 0, 2, 1]), ([0, 0, 0], [0, 0, 0]), ([2, 2, 2], [9, 9, 9])],
)
def test_kld_matches_independent_probability_definition(cupy_gpu, p_values, q_values):
    """Both CUDA paths must implement normalized, directed relative entropy."""
    cp = cupy_gpu
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    p = cp.asarray(p_values, dtype=cp.float32)
    q = cp.asarray(q_values, dtype=cp.float32)
    scratch = cp.empty_like(p)

    probability_p = np.asarray(p_values, dtype=np.float64) + 1e-4
    probability_q = np.asarray(q_values, dtype=np.float64) + 1e-4
    probability_p /= probability_p.sum()
    probability_q /= probability_q.sum()
    expected = np.sum(probability_p * np.log(probability_p / probability_q))
    for actual in (rlgc.kl_div(p, q), rlgc._kl_div_into(p, q, scratch)):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(cp.asnumpy(p), p_values)
    np.testing.assert_array_equal(cp.asnumpy(q), q_values)


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(31, 37), (7, 23, 29)])
def test_custom_cuda_ssim_matches_scipy_reference(cupy_gpu, shape):
    """Validate the custom RawModule kernels for both 2D and 3D images.

    Parameters
    ----------
    cupy_gpu : object
        Value supplied for ``cupy gpu``.
    shape : object
        Value supplied for ``shape``.

    Returns
    -------
    None
        No value is returned.
    """
    cp = cupy_gpu
    ssim_module = importlib.import_module("opm_processing.imageprocessing.ssim_cuda")
    rng = np.random.default_rng(14)
    reference = rng.random(shape, dtype=np.float32)
    comparison = np.clip(
        reference * np.float32(0.91) + rng.normal(0, 0.035, shape).astype(np.float32),
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
    constant = cp.zeros(shape, dtype=cp.float32)
    constant_score = ssim_module.structural_similarity_cupy_sep_shared(
        constant, constant, win_size=5
    )
    np.testing.assert_allclose(constant_score, 1.0, atol=2e-6)


@pytest.mark.unit
def test_gpu_hot_pixel_replacement_matches_synthetic_sample(cupy_gpu):
    """Exercise cupyx median filtering on a representative camera defect.

    Parameters
    ----------
    cupy_gpu : object
        Value supplied for ``cupy gpu``.

    Returns
    -------
    None
        No value is returned.
    """
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


@pytest.mark.integration
def test_gpu_registration_recovers_known_3d_translation(cupy_gpu):
    """Run cuCIM registration plus CUDA SSIM on a translated sample volume.

    Parameters
    ----------
    cupy_gpu : object
        Value supplied for ``cupy gpu``.

    Returns
    -------
    None
        No value is returned.
    """
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


@pytest.mark.integration
def test_rlgc_gpu_deconvolution_improves_synthetic_point_sample(cupy_gpu):
    """End-to-end GPU deconvolution must improve recovery of point emitters.

    Parameters
    ----------
    cupy_gpu : object
        Value supplied for ``cupy gpu``.

    Returns
    -------
    object
        Result produced by the callable.
    """
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
        """Measure RMSE after fitting a scalar intensity correction.

        Parameters
        ----------
        candidate : object
            Value supplied for ``candidate``.

        Returns
        -------
        object
            Result produced by the callable.
        """
        scale = float(np.vdot(candidate, truth) / np.vdot(candidate, candidate))
        return float(np.sqrt(np.mean((candidate * scale - truth) ** 2)))

    assert scale_invariant_rmse(restored) < 0.85 * scale_invariant_rmse(observed)
    # Shape recovery alone can conceal a factor-of-two photon gain.
    np.testing.assert_allclose(restored.sum(), observed.sum(), rtol=0.1)
    assert np.mean((restored - truth) ** 2) < np.mean((observed - truth) ** 2)
    rlgc.clear_rlgc_caches(clear_memory_pool=True)


@pytest.mark.integration
def test_rlgc_preserves_stationary_intensity(cupy_gpu, monkeypatch):
    """A balanced split of an exact constant prediction must have unit update."""
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")

    class BalancedSplit:
        """Provide the exact half-count observations for this fixed-point case."""

        def binomial(self, counts, p):
            """Split even counts equally so both gradients are exactly zero."""
            return counts // 2

    monkeypatch.setattr(rlgc.cp.random, "default_rng", lambda seed: BalancedSplit())
    observed = np.full((1, 8, 8), 32, dtype=np.float32)
    restored = rlgc.rlgc(observed, np.ones((1, 1, 1), dtype=np.float32))

    np.testing.assert_allclose(restored, observed, rtol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize("safe_mode", [True, False])
def test_rlgc_stopping_is_invariant_to_split_labels(cupy_gpu, monkeypatch, safe_mode):
    """Swapping the two halves on alternate iterations cannot change recovery."""
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    zz, yy, xx = np.mgrid[-2:3, -3:4, -3:4]
    psf = np.exp(-(zz**2 / 1.1**2 + (yy**2 + xx**2) / 4) / 2)
    psf = (psf / psf.sum()).astype(np.float32)
    truth = np.zeros((9, 35, 37), np.float32)
    truth[2, 9, 10] = 1800
    truth[4, 18, 27] = 2400
    truth[6, 26, 17] = 2100
    observed = (
        np.random.default_rng(31)
        .poisson(ndimage.convolve(truth, psf, mode="reflect") + 0.25)
        .astype(np.float32)
    )
    options = dict(safe_mode=safe_mode, rng_seed=17, limit=0.02, max_delta=0.005)
    baseline = rlgc.rlgc(observed, psf, **options)
    original_split = rlgc._split_observed_counts
    calls = 0

    def relabeled_split(image, rng):
        nonlocal calls
        split = original_split(image, rng)
        calls += 1
        return image - split if calls % 2 == 0 else split

    monkeypatch.setattr(rlgc, "_split_observed_counts", relabeled_split)
    actual = rlgc.rlgc(observed, psf, **options)
    assert calls > 2  # Exercise stopping beyond the first relabeling.
    np.testing.assert_allclose(actual, baseline, rtol=1e-6, atol=1e-5)
    assert np.mean((actual - truth) ** 2) < np.mean((observed - truth) ** 2)


@pytest.mark.integration
@pytest.mark.parametrize("scan_planes", [1, 5])
def test_rlgc_preserves_smooth_low_count_background(cupy_gpu, scan_planes):
    """Local stopping must preserve resolved background instead of flat patches."""
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    yy, xx = np.mgrid[:64, :96]
    background = 1.5 + 0.3 * np.sin(2 * np.pi * xx / 96)
    background += 0.15 * np.cos(2 * np.pi * yy / 64)
    truth = np.broadcast_to(background, (scan_planes, 64, 96)).astype(np.float32)
    zz, py, px = np.mgrid[-1:2, -5:6, -5:6]
    psf = np.exp(-(zz**2 + (py**2 + px**2) / 2.5**2) / 2)
    if scan_planes == 1:
        psf = psf[1:2]
    psf = (psf / psf.sum()).astype(np.float32)
    observed = ndimage.convolve(truth, psf, mode="reflect")

    restored = rlgc.rlgc(observed, psf, max_delta=0.1)

    assert np.isfinite(restored).all()
    assert restored.min() >= 0
    assert float(np.sqrt(np.mean((restored - truth) ** 2))) < 0.04
    assert np.mean(np.abs(restored - observed.mean()) < 1e-5) < 0.001
    np.testing.assert_allclose(restored.mean(), truth.mean(), rtol=0.01)


@pytest.mark.unit
def test_rlgc_fractional_split_is_unbiased_and_preserves_counts(cupy_gpu):
    """Both halves must have equal expected signal, including sub-count pixels."""
    cp = cupy_gpu
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    levels = np.array([0, 0.24, 0.96, 1, 1.44, 2.88, 12.24], dtype=np.float32)
    observed = cp.asarray(np.broadcast_to(levels[:, None], (len(levels), 200_000)))
    split1 = rlgc._split_observed_counts(observed, cp.random.default_rng(73))
    split2 = observed - split1

    assert bool(cp.all(split1 >= 0)) and bool(cp.all(split2 >= 0))
    np.testing.assert_allclose(
        cp.asnumpy(split1 + split2), cp.asnumpy(observed), atol=1e-6
    )
    for split in (split1, split2):
        np.testing.assert_allclose(
            cp.asnumpy(split.mean(axis=1)), levels / 2, atol=0.01
        )
        # Independently assigned unit counts and the residual weighted count
        # have variance sum(weight**2)/4. Deterministic halving must fail.
        expected_variance = (np.floor(levels) + (levels % 1) ** 2) / 4
        np.testing.assert_allclose(
            cp.asnumpy(split.var(axis=1)), expected_variance, rtol=0.02, atol=1e-5
        )


@pytest.mark.unit
def test_rlgc_integer_split_has_binomial_distribution(cupy_gpu):
    """Four photons split into 0..4 counts with binomial probabilities."""
    cp = cupy_gpu
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    observed = cp.full(200_000, 4, dtype=cp.float32)
    actual = cp.asnumpy(
        rlgc._split_observed_counts(observed, cp.random.default_rng(19))
    )
    np.testing.assert_array_equal(actual, actual.astype(np.int64))
    np.testing.assert_allclose(
        np.bincount(actual.astype(np.int64), minlength=5) / actual.size,
        np.array([1, 4, 6, 4, 1]) / 16,
        atol=0.004,
    )


@pytest.mark.integration
@pytest.mark.parametrize("seed", [7, 31, 83])
@pytest.mark.parametrize(
    "read_noise_e", [0.7, 1.0, 1.6], ids=["ultra-quiet", "standard", "fast"]
)
def test_rlgc_recovers_dim_emitter_with_camera_noise(cupy_gpu, seed, read_noise_e):
    """Recover an emitter with ORCA-Fusion BT shot, read, and quantization noise."""
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[-3:4, -6:7, -6:7]
    psf = np.exp(-0.5 * ((zz / 1.2) ** 2 + ((yy + 2 * zz) / 2) ** 2 + (xx / 2) ** 2))
    psf = (psf / psf.sum()).astype(np.float32)
    truth = np.zeros((9, 48, 64), dtype=np.float32)
    truth[4, 24, 32] = 100
    # Hamamatsu C15440-20UP: 0.24 electrons/ADU, 100 ADU offset,
    # 0.7/1.0/1.6 electrons RMS read noise across the three scan modes.
    # See the manufacturer's
    # ORCA-Fusion/ORCA-Fusion BT technical note, specifications, page 15:
    # https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/sys/SCAS0138E_C14440-20UP_tec.pdf
    # Shot noise acts on electrons before ADC conversion. Multiplying a
    # Poisson draw by 0.24 instead would incorrectly reduce its variance.
    background_electrons = rng.poisson(1.5, truth.shape)
    background_electrons = background_electrons + rng.normal(
        0, read_noise_e, truth.shape
    )
    injected = rng.poisson(ndimage.convolve(truth, psf, mode="reflect")).astype(
        np.float32
    )

    def calibrate_camera(electrons):
        """Digitize electron charge, then apply the pipeline's camera calibration."""
        adu = np.clip(np.rint(100 + electrons / 0.24), 0, 65535).astype(np.uint16)
        return np.maximum((adu.astype(np.float32) - 100) * 0.24, 0)

    background = calibrate_camera(background_electrons)
    observed = calibrate_camera(background_electrons + injected)
    restored_background = rlgc.rlgc(background, psf, rng_seed=seed)
    restored = rlgc.rlgc(observed, psf, rng_seed=seed)
    response = restored - restored_background

    # Score the complete estimate against the expected object, not a sampled
    # noise realization. Paired subtraction below cancels retained background
    # grain and cannot establish whole-image reconstruction accuracy by itself.
    expected_object = truth.astype(np.float64) + 1.5
    observed_mse = np.mean((observed - expected_object) ** 2)
    restored_mse = np.mean((restored - expected_object) ** 2)
    assert restored_mse < observed_mse, {
        "restored_mse": restored_mse,
        "observed_mse": observed_mse,
    }
    assert np.mean((restored_background.astype(np.float64) - 1.5) ** 2) < np.mean(
        (background.astype(np.float64) - 1.5) ** 2
    )

    # Paired background subtraction isolates the response to the known emitter.
    # The core contains the emitter's true location, so useful deconvolution
    # must move signal into it. A no-op cannot pass this strict improvement.
    core = (slice(3, 6), slice(22, 27), slice(30, 35))
    # Compare concentration with the actually measured blurred signal. Keep
    # the flux bound against the injected electron count, including any signal
    # lost during camera calibration's clipping of negative measurements.
    measured_signal = observed - background
    core_gain = float(response[core].sum() / measured_signal[core].sum())
    flux_ratio = float(response.sum() / injected.sum())
    assert core_gain > 1.0, {"core_gain": core_gain, "flux_ratio": flux_ratio}
    np.testing.assert_allclose(flux_ratio, 1.0, rtol=0.2)
    assert np.isfinite(restored).all()
    assert restored.min() >= 0


@pytest.mark.integration
def test_rlgc_empty_image_remains_empty(cupy_gpu):
    """Data-derived initialization must not introduce signal into an empty image."""
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    observed = np.zeros((3, 12, 16), dtype=np.float32)
    psf = np.ones((3, 3, 3), dtype=np.float32)
    restored = rlgc.rlgc(observed, psf)
    np.testing.assert_array_equal(restored, observed)


@pytest.mark.integration
@pytest.mark.parametrize("singleton_z", [False, True])
def test_rlgc_2d_recovers_points_using_central_psf(cupy_gpu, singleton_z):
    """Run the real wrapper and solver against independently blurred YX truth."""
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    yy, xx = np.mgrid[-3:4, -3:4]
    central = np.exp(-(yy**2 + xx**2) / 8).astype(np.float32)
    central /= central.sum()
    skewed_psf = np.stack(
        (np.ones_like(central), central * 7, np.ones_like(central) * 100)
    )
    truth = np.zeros((41, 43), np.float32)
    truth[12, 13] = 2000
    truth[28, 30] = 3000
    observed = (
        np.random.default_rng(31)
        .poisson(ndimage.convolve(truth, central, mode="reflect") + 0.25)
        .astype(np.float32)
    )
    if singleton_z:
        observed = observed[None]
        truth = truth[None]
    restored = rlgc.rlgc_2d(observed, skewed_psf, limit=0.02, max_delta=0.005)
    assert restored.shape == truth.shape
    assert restored.dtype == np.float32
    assert np.mean((restored - truth) ** 2) < np.mean((observed - truth) ** 2)
    np.testing.assert_allclose(restored.sum(), observed.sum(), rtol=0.1)
    np.testing.assert_array_equal(
        np.unravel_index(np.argmax(restored), restored.shape),
        np.unravel_index(np.argmax(truth), truth.shape),
    )
