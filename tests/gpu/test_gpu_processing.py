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
from scipy.special import kl_div
from scipy.stats import entropy

from tests.registration import (
    test_bounded_peak_matches_upstream_phase_normalization as test_bounded_peak_matches_upstream_phase_normalization,
)
from tests.registration import (
    test_orthogonal_camera_pairs_recover_known_stage_errors as test_orthogonal_camera_pairs_recover_known_stage_errors,
)
from tests.simulation.camera import digitize
from tests.simulation.metrics import camera_flux_accuracy

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("cupy_gpu")]


def test_fft_convolution_matches_direct_reference(cupy_gpu, registration_scene):
    """Exercise cuFFT and validate all voxels against direct convolution."""
    cp = cupy_gpu
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    image = registration_scene((6, 8, 10), seed=8).raw.astype(np.float32)
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
    cupy_gpu, registration_scene, applied_shift
):
    """Register noiseless optical means under declared exact signal transforms."""
    cp = cupy_gpu
    try:
        import cucim  # noqa: F401
    except ImportError:
        pytest.fail("cuCIM is unavailable; install the project's gpu extra")

    tilefusion = importlib.import_module("opm_processing.imageprocessing.tilefusion")
    if not tilefusion.USING_GPU or tilefusion.xp is not cp:
        pytest.fail("TileFusion silently selected its CPU registration backend")

    fixed = registration_scene((11, 39, 43), seed=22).expectation.astype(np.float32)
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
def test_unbounded_registration_recovers_displaced_physical_observations(
    cupy_gpu, physical_translation_pair, applied_shift
):
    """Register independently exposed, aperiodic points without a displacement prior."""
    tilefusion = importlib.import_module("opm_processing.imageprocessing.tilefusion")
    fixed, moving, expected = physical_translation_pair(applied_shift)
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


@pytest.mark.parametrize("singleton_z", [False, True])
def test_rlgc_2d_recovers_points_using_central_psf(
    cupy_gpu,
    planar_point_model,
    ideal_photon_camera,
    singleton_z,
    production_forward_predictions,
):
    """Run the real wrapper and solver against independently blurred YX truth."""
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    case = planar_point_model
    truth, skewed_psf = case.truth, case.psf
    observed = digitize(
        case.expectation,
        ideal_photon_camera,
        seed=31,
    ).astype(np.float32)
    if singleton_z:
        observed = observed[None]
        truth = truth[None]
    restored = rlgc.rlgc(observed, skewed_psf, limit=0.02, max_delta=0.005)
    assert restored.shape == truth.shape
    assert restored.dtype == np.float32
    expectation = case.expectation[None] if singleton_z else case.expectation
    measured = production_forward_predictions[-1]
    assert (
        kl_div(expectation, measured["prediction"]).sum()
        < kl_div(expectation, measured["input_prediction"]).sum()
    )
    for aperture in case.apertures:
        assert camera_flux_accuracy(
            measured["prediction"], expectation, ideal_photon_camera, aperture
        )
    np.testing.assert_array_equal(
        np.unravel_index(np.argmax(restored), restored.shape),
        np.unravel_index(np.argmax(truth), truth.shape),
    )
    plane = restored[0] if singleton_z else restored
    for point, field in zip(case.points, case.point_fields, strict=True):
        candidates = np.where(field, plane, -np.inf)
        peak = np.unravel_index(np.argmax(candidates), plane.shape)
        # Localize both original emitters to one output sample, without a
        # peak-dependent search window or fitted amplitude.
        np.testing.assert_allclose(peak, point, rtol=0, atol=1)


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
