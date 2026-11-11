"""Correctness tests for the CUDA image-processing paths.

These unit tests use known analytical operands and simulated camera data. Each compares
GPU output with an independent result or checks a
quantitative reconstruction/registration property.

Unavailable CUDA hardware skips these tests.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy import ndimage
from scipy.stats import entropy

from opm_processing.imageprocessing import tilefusion

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("cupy_gpu")]


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-4])
@pytest.mark.parametrize("seed", [485, 492])
def test_bounded_peak_matches_upstream_phase_normalization(
    aperiodic_registration_object, scale, seed
):
    """Use the upstream weak-frequency floor even for faint unrelated objects."""
    tilefusion.require_gpu_backend()
    backend = tilefusion.xp
    fixed, moving = (
        aperiodic_registration_object((9, 43, 55), seed=value).astype(np.float32)
        for value in (seed, seed + 1)
    )
    fixed = backend.asarray((fixed / fixed.max() * np.float32(scale))[:7, :35, :47])
    moving = backend.asarray(
        (moving / moving.max() * np.float32(scale))[1:8, 4:39, 3:50]
    )
    expected, _, _ = tilefusion.phase_cross_correlation(
        fixed, moving, normalization="phase", upsample_factor=1
    )
    actual = tilefusion._bounded_phase_correlation_peak(
        fixed, moving, np.asarray(fixed.shape) // 2
    )
    np.testing.assert_array_equal(actual.tolist(), backend.asarray(expected).tolist())


def test_orthogonal_camera_pairs_recover_known_stage_errors(
    orthogonal_registration_pair,
):
    """Recover prescribed stage errors from independently formed oblique images."""
    tilefusion.require_gpu_backend()
    backend = tilefusion.xp
    pair = orthogonal_registration_pair
    sampling = tilefusion.DEFAULT_REGISTRATION_DOWNSAMPLE
    fixed, moving = (
        tilefusion.block_reduce(backend.asarray(image), sampling, backend.mean)
        for image in (pair.fixed, pair.moving)
    )
    masks = tuple(
        tilefusion.block_reduce(backend.asarray(mask), sampling, backend.min)
        for mask in pair.masks
    )
    shift, score = tilefusion.TileFusion.register_and_score(
        fixed,
        moving,
        3,
        max_shift=np.array((2, 4, 4)) / sampling,
        registration_axis=pair.axis,
        shift_sampling=sampling,
        support_masks=masks,
    )
    assert np.isfinite(score)
    np.testing.assert_array_equal(np.rint(np.asarray(shift) * sampling), pair.expected)


def test_fft_convolution_matches_direct_reference(
    cupy_gpu, aperiodic_registration_object
):
    """Exercise cuFFT and validate all voxels against direct convolution."""
    cp = cupy_gpu
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    image = aperiodic_registration_object((6, 8, 10), seed=8).astype(np.float32)
    image /= image.max()
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
    expected = ndimage.convolve(
        image.astype(np.float64),
        expected_kernel,
        mode="wrap",
        origin=tuple(-(np.array(image.shape) // 2)),
    )
    np.testing.assert_allclose(cp.asnumpy(actual_gpu), expected, rtol=2e-5, atol=2e-5)
    rlgc.clear_rlgc_caches()


@pytest.mark.parametrize(
    "p_values,q_values",
    [([0, 1, 3, 20], [4, 0, 2, 1]), ([0, 0, 0], [0, 0, 0]), ([2, 2, 2], [9, 9, 9])],
)
@pytest.mark.parametrize("array_backend", ["gpu", "host"])
def test_kld_matches_independent_probability_definition(
    cupy_gpu, p_values, q_values, array_backend
):
    """The production CUDA divergence implements normalized directed relative entropy."""
    cp = cupy_gpu if array_backend == "gpu" else np
    to_numpy = cupy_gpu.asnumpy if array_backend == "gpu" else np.asarray
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    p = cp.asarray(p_values, dtype=cp.float32)
    q = cp.asarray(q_values, dtype=cp.float32)
    scratch = cp.empty_like(p)

    expected = entropy(
        np.asarray(p_values, dtype=np.float64) + 1e-4,
        np.asarray(q_values, dtype=np.float64) + 1e-4,
    )
    actual = rlgc._kl_div_into(p, q, scratch)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(to_numpy(p), p_values)
    np.testing.assert_array_equal(to_numpy(q), q_values)


@pytest.mark.parametrize("applied_shift", ((1.0, -2.0, 3.0), (0.6, -2.3, 3.4)))
def test_exact_signal_registration_recovers_known_3d_translation(
    cupy_gpu, aperiodic_registration_object, applied_shift
):
    """Register known analytical objects under declared exact signal transforms."""
    cp = cupy_gpu
    try:
        import cucim  # noqa: F401
    except ImportError:
        pytest.fail("cuCIM is unavailable; install the project's gpu extra")

    tilefusion = importlib.import_module("opm_processing.imageprocessing.tilefusion")
    assert tilefusion.USING_GPU and tilefusion.xp is cp

    fixed = aperiodic_registration_object((11, 39, 43), seed=22).astype(np.float32)
    applied_shift = np.array(applied_shift, dtype=np.float32)
    if np.any(applied_shift != np.rint(applied_shift)):
        # The Fourier shift defines an exact periodic fractional translation;
        # repeated spatial interpolation would change the sampled signal.
        moving = np.fft.ifftn(
            ndimage.fourier_shift(np.fft.fftn(fixed), applied_shift)
        ).real.astype(np.float32)
    else:
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
        fixed, moving, win_size=5, max_shift=(2, 4, 5)
    )

    np.testing.assert_allclose(recovered, -applied_shift, atol=0.15)
    assert score > unregistered_score


@pytest.mark.parametrize("applied_shift", ((1.0, -2.0, 3.0), (0.6, -2.3, 3.4)))
def test_registration_recovers_displaced_cartesian_observations(
    cupy_gpu, cartesian_registration_pair, applied_shift
):
    """Register independent optical observations in the consumed Cartesian frame."""
    tilefusion = importlib.import_module("opm_processing.imageprocessing.tilefusion")
    fixed, moving, expected = cartesian_registration_pair(applied_shift)
    unregistered = tilefusion._ssim(
        cupy_gpu.asarray(fixed), cupy_gpu.asarray(moving), 5
    )
    measured, score = tilefusion.TileFusion.register_and_score(
        fixed, moving, win_size=5
    )
    np.testing.assert_allclose(measured, expected, atol=0.15)
    assert score > unregistered


@pytest.mark.parametrize("crop_scan", [None, 1, 2])
@pytest.mark.parametrize("camera_value", (0, -1))
def test_rlgc_empty_image_remains_empty(cupy_gpu, crop_scan, camera_value, monkeypatch):
    """Preserve empty data and release solver state."""
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    clear = rlgc.clear_rlgc_caches
    cleanups = []

    def observe_cleanup(*args, **kwargs):
        """Count actual resource release without replacing its implementation."""
        cleanups.append(True)
        return clear(*args, **kwargs)

    monkeypatch.setattr(rlgc, "clear_rlgc_caches", observe_cleanup)
    observed = np.full((11, 12, 16), camera_value, dtype=np.float32)
    psf = np.ones((3, 3), dtype=np.float32)
    restored = rlgc.rlgc(observed, psf, crop_scan=crop_scan)
    np.testing.assert_array_equal(restored, np.zeros_like(observed))
    assert len(cleanups) == 1
    pool = cupy_gpu.get_default_memory_pool()
    assert pool.total_bytes() == pool.used_bytes()


@pytest.mark.parametrize("scale", (0.01, 1, 1000))
@pytest.mark.parametrize("array_backend", ["gpu", "host"])
def test_split_kl_scores_match_direct_probability_definition(
    cupy_gpu, scale, array_backend
):
    """Check prescribed prediction-to-split relative entropy and pseudocount."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    xp = cupy_gpu if array_backend == "gpu" else np
    rng = np.random.default_rng(51)
    prediction = (rng.uniform(0, 30, (3, 5, 7)) * scale).astype(np.float32)
    prediction[0] = 0
    halves = [rng.poisson(3, prediction.shape).astype(np.float32) for _ in range(2)]
    expected = [
        entropy(
            (prediction.astype(np.float64) + 1e-4).ravel(),
            (half.astype(np.float64) + 1e-4).ravel(),
        )
        for half in halves
    ]
    actual = solver.split_kl_scores(
        *(xp.asarray(value) for value in (prediction, *halves)),
        xp.empty_like(xp.asarray(prediction)),
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
