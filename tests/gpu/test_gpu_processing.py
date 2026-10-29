"""Correctness tests for the CUDA image-processing paths.

These unit tests use small synthetic microscopy volumes. Every test compares
GPU output with an independent result or checks a
quantitative reconstruction/registration property.

Set ``OPM_REQUIRE_GPU=1`` in a GPU CI job so an unavailable GPU stack fails the
suite instead of skipping it.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy import ndimage

from tests.simulation.metrics import (
    predictive_deviance,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("sampling", [(1, 1, 1), (1, 2, 2)])
def test_orthogonal_camera_pairs_recover_known_stage_errors(
    cupy_gpu, orthogonal_registration_pair, sampling
):
    """Recover prescribed stage errors from independently formed oblique images."""
    from cucim.skimage.measure import block_reduce

    from opm_processing.imageprocessing.tilefusion import TileFusion

    pair = orthogonal_registration_pair
    fixed, moving = (
        block_reduce(cupy_gpu.asarray(image), sampling, cupy_gpu.mean)
        for image in (pair.fixed, pair.moving)
    )
    masks = tuple(
        block_reduce(cupy_gpu.asarray(mask), sampling, cupy_gpu.min)
        for mask in pair.masks
    )
    shift, score = TileFusion.register_and_score(
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

    probability_p = np.asarray(p_values, dtype=np.float64) + 1e-4
    probability_q = np.asarray(q_values, dtype=np.float64) + 1e-4
    probability_p /= probability_p.sum()
    probability_q /= probability_q.sum()
    expected = np.sum(probability_p * np.log(probability_p / probability_q))
    actual = rlgc._kl_div_into(p, q, scratch)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(to_numpy(p), p_values)
    np.testing.assert_array_equal(to_numpy(q), q_values)


@pytest.mark.parametrize("applied_shift", ((1.0, -2.0, 3.0), (0.6, -2.3, 3.4)))
def test_gpu_registration_recovers_known_3d_translation(cupy_gpu, applied_shift):
    """Recover integer and fractional translations using cuCIM and CUDA SSIM."""
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
        fixed, moving, win_size=5
    )

    np.testing.assert_allclose(recovered, -applied_shift, atol=0.15)
    assert score > unregistered_score


@pytest.mark.parametrize("crop_scan", [None, 1, 2, 0, -1, 1.5, True])
@pytest.mark.parametrize(
    "verbosity", [None, 0, 1], ids=["default", "no-diagnostics", "diagnostics"]
)
def test_rlgc_empty_image_remains_empty(
    cupy_gpu, crop_scan, verbosity, capsys, monkeypatch
):
    """Preserve empty data and release solver state with opt-in diagnostics."""
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    clear = rlgc.clear_rlgc_caches
    cleanups = []

    def observe_cleanup(*args, **kwargs):
        """Count actual resource release without replacing its implementation."""
        cleanups.append(True)
        return clear(*args, **kwargs)

    monkeypatch.setattr(rlgc, "clear_rlgc_caches", observe_cleanup)
    observed = np.zeros((3, 12, 16), dtype=np.float32)
    psf = np.ones((3, 3, 3), dtype=np.float32)
    if crop_scan is not None and (
        isinstance(crop_scan, bool) or crop_scan in (0, -1, 1.5)
    ):
        with pytest.raises(ValueError, match="crop_scan must be a positive integer"):
            rlgc.rlgc(observed, psf, crop_scan=crop_scan)
        return
    options = {} if verbosity is None else {"verbose": verbosity}
    restored = rlgc.rlgc(observed, psf, crop_scan=crop_scan, **options)
    np.testing.assert_array_equal(restored, observed)
    assert len(cleanups) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert ("decon:" in captured.err) == (crop_scan in (1, 2))
    if verbosity:
        assert "RLGC acquired_shape=" in captured.err
        assert "iteration=" in captured.err
        assert "stop=" in captured.err
    else:
        assert "RLGC acquired_shape=" not in captured.err
        assert "iteration=" not in captured.err
        assert "stop=" not in captured.err
        if crop_scan is None:
            assert not captured.err
    assert not rlgc._fft_cache_3d
    pool = cupy_gpu.get_default_memory_pool()
    assert pool.total_bytes() == pool.used_bytes()


@pytest.mark.parametrize("singleton_z", [False, True])
def test_rlgc_2d_recovers_points_using_central_psf(
    cupy_gpu, planar_point_model, singleton_z
):
    """Run the real wrapper and solver against independently blurred YX truth."""
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    truth, central, skewed_psf = planar_point_model
    observed = (
        np.random.default_rng(31)
        .poisson(ndimage.convolve(truth, central, mode="reflect") + 0.25)
        .astype(np.float32)
    )
    if singleton_z:
        observed = observed[None]
        truth = truth[None]
    restored = rlgc.rlgc(observed, skewed_psf, limit=0.02, max_delta=0.005)
    assert restored.shape == truth.shape
    assert restored.dtype == np.float32
    psf = central[None] if singleton_z else central
    expectation = ndimage.convolve(truth, psf, mode="reflect") + 0.25
    assert predictive_deviance(
        restored, expectation, psf, boundary="reflect"
    ) < predictive_deviance(observed, expectation, psf, boundary="reflect")
    np.testing.assert_array_equal(
        np.unravel_index(np.argmax(restored), restored.shape),
        np.unravel_index(np.argmax(truth), truth.shape),
    )


@pytest.mark.parametrize("scale", (0.01, 1, 1000))
@pytest.mark.parametrize("array_backend", ["gpu", "host"])
def test_split_kl_scores_match_direct_probability_definition(
    cupy_gpu, scale, array_backend
):
    """Lock MATLAB's prediction-to-split direction, normalization and pseudocount."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    xp = cupy_gpu if array_backend == "gpu" else np
    rng = np.random.default_rng(51)
    prediction = (rng.uniform(0, 30, (3, 5, 7)) * scale).astype(np.float32)
    prediction[0] = 0
    halves = [rng.poisson(3, prediction.shape).astype(np.float32) for _ in range(2)]
    probability = prediction.astype(np.float64) + 1e-4
    probability /= probability.sum()
    expected = []
    for half in halves:
        target = half.astype(np.float64) + 1e-4
        target /= target.sum()
        expected.append(np.sum(probability * np.log(probability / target)))
    actual = solver.split_kl_scores(
        *(xp.asarray(value) for value in (prediction, *halves)),
        xp.empty_like(xp.asarray(prediction)),
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
