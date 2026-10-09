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


@pytest.mark.unit
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
    expected = _direct_circular_convolution(image, expected_kernel)
    np.testing.assert_allclose(cp.asnumpy(actual_gpu), expected, rtol=2e-5, atol=2e-5)
    rlgc.clear_rlgc_caches()


@pytest.mark.unit
@pytest.mark.parametrize(
    "p_values,q_values",
    [([0, 1, 3, 20], [4, 0, 2, 1]), ([0, 0, 0], [0, 0, 0]), ([2, 2, 2], [9, 9, 9])],
)
def test_kld_matches_independent_probability_definition(cupy_gpu, p_values, q_values):
    """The production CUDA divergence implements normalized directed relative entropy."""
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
    actual = rlgc._kl_div_into(p, q, scratch)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(cp.asnumpy(p), p_values)
    np.testing.assert_array_equal(cp.asnumpy(q), q_values)


@pytest.mark.unit
@pytest.mark.parametrize("applied_shift", ((1.0, -2.0, 3.0), (0.6, -2.3, 3.4)))
def test_gpu_registration_recovers_known_3d_translation(cupy_gpu, applied_shift):
    """Recover integer and fractional translations using cuCIM and CUDA SSIM.

    Parameters
    ----------
    cupy_gpu
        CuPy module verified against the available CUDA device.
    applied_shift : tuple of float
        Known moving-image translation in ZYX pixels.
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


@pytest.mark.unit
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
        """Measure RMSE after fitting a scalar intensity correction."""
        scale = float(np.vdot(candidate, truth) / np.vdot(candidate, candidate))
        return float(np.sqrt(np.mean((candidate * scale - truth) ** 2)))

    assert scale_invariant_rmse(restored) < 0.85 * scale_invariant_rmse(observed)
    # Shape recovery alone can conceal a factor-of-two photon gain.
    np.testing.assert_allclose(restored.sum(), observed.sum(), rtol=0.1)
    assert np.mean((restored - truth) ** 2) < np.mean((observed - truth) ** 2)
    rlgc.clear_rlgc_caches(clear_memory_pool=True)


@pytest.mark.unit
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


@pytest.mark.unit
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
def test_reference_fractional_split_retains_remainders_in_complement(cupy_gpu):
    """Verify the local reference's fractional-count mean and complementary data.

    Parameters
    ----------
    cupy_gpu
        Fixture providing CuPy on a working CUDA device.
    """
    cp = cupy_gpu
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    levels = np.asarray((0.24, 0.96, 1.44, 2.88, 12.24), np.float32)
    observed = cp.asarray(np.broadcast_to(levels[:, None], (len(levels), 200_000)))
    first = solver._split_observed_counts(
        observed, cp.random.default_rng(73), assign_fractional_remainder=False
    )
    second = observed - first
    np.testing.assert_array_equal(cp.asnumpy(first + second), cp.asnumpy(observed))
    np.testing.assert_array_equal(cp.asnumpy(first), cp.asnumpy(cp.floor(first)))
    np.testing.assert_allclose(
        cp.asnumpy(first.mean(axis=1)), np.floor(levels) / 2, atol=0.01
    )
    np.testing.assert_allclose(
        cp.asnumpy(second.mean(axis=1)), levels - np.floor(levels) / 2, atol=0.01
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "read_noise_e", [0.7, 1.0, 1.6], ids=["ultra-quiet", "standard", "fast"]
)
def test_rlgc_recovers_dim_emitter_with_camera_noise(cupy_gpu, read_noise_e):
    """Recover a known emitter across calibrated camera-noise realizations.

    Parameters
    ----------
    cupy_gpu
        Fixture requiring a working CUDA device.
    read_noise_e : float
        RMS electron read noise before ADC conversion. Three independent shot
        and read-noise draws check individual object error, concentration and
        absolute fluorescence, then the mean differential fluorescence.
    """
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    flux_ratios = []
    for seed in (7, 31, 83):
        rng = np.random.default_rng(seed)
        zz, yy, xx = np.mgrid[-3:4, -6:7, -6:7]
        psf = np.exp(
            -0.5 * ((zz / 1.2) ** 2 + ((yy + 2 * zz) / 2) ** 2 + (xx / 2) ** 2)
        )
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
        flux_ratios.append(flux_ratio)
        np.testing.assert_allclose(restored.sum(), observed.sum(), rtol=0.01)
        np.testing.assert_allclose(
            restored_background.sum(), background.sum(), rtol=0.01
        )
        assert np.isfinite(restored).all()
        assert restored.min() >= 0
    # Independently stopped nonlinear reconstructions need not conserve the
    # flux of their difference for each draw. Check its expected response
    # across realizations, without fitting a gain to any reconstruction.
    np.testing.assert_allclose(np.mean(flux_ratios), 1.0, rtol=0.2)


@pytest.mark.unit
def test_rlgc_empty_image_remains_empty(cupy_gpu):
    """Data-derived initialization must not introduce signal into an empty image."""
    rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    observed = np.zeros((3, 12, 16), dtype=np.float32)
    psf = np.ones((3, 3, 3), dtype=np.float32)
    restored = rlgc.rlgc(observed, psf)
    np.testing.assert_array_equal(restored, observed)


@pytest.mark.unit
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
    restored = rlgc.rlgc_2d(observed, skewed_psf, limit=0.02, max_delta=0.005)
    assert restored.shape == truth.shape
    assert restored.dtype == np.float32
    assert np.mean((restored - truth) ** 2) < np.mean((observed - truth) ** 2)
    np.testing.assert_allclose(restored.sum(), observed.sum(), rtol=0.1)
    np.testing.assert_array_equal(
        np.unravel_index(np.argmax(restored), restored.shape),
        np.unravel_index(np.argmax(truth), truth.shape),
    )


@pytest.mark.unit
@pytest.mark.parametrize("scale", (0.01, 1, 1000))
def test_same_split_kl_change_matches_direct_scores(cupy_gpu, scale):
    """Check cancellation against independently normalized CPU KL scores.

    Parameters
    ----------
    cupy_gpu
        Fixture requiring a working CUDA device.
    scale : float
        Photon-rate scale covering fractional, ordinary and bright measurements.
    """
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    rng = np.random.default_rng(51)
    current = (rng.uniform(0, 30, (3, 5, 7)) * scale).astype(np.float32)
    previous = (rng.uniform(0, 30, current.shape) * scale).astype(np.float32)
    current[0] = 0
    previous[1] = 0
    first = rng.poisson(3, current.shape).astype(np.float32)
    second = rng.poisson(3, current.shape).astype(np.float32)
    expected = []
    for split in (first, second):
        q = split.astype(np.float64) + 1e-4
        q /= q.sum()
        scores = []
        for prediction in (current, previous):
            p = prediction.astype(np.float64) + 1e-4
            p /= p.sum()
            scores.append(np.sum(q * np.log(q / p)))
        expected.append(scores[0] - scores[1])
    actual = solver.split_kl_changes(
        *(cupy_gpu.asarray(value) for value in (current, previous, first, second)),
        cupy_gpu.empty_like(cupy_gpu.asarray(current)),
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
