"""Reconstruction physics using the shared fine-Cartesian camera forward model.

Numerical assertions use production operators, SciPy and the shared simulator.
"""

import importlib
import logging
from functools import partial

import numpy as np
import pytest
from scipy.signal import fftconvolve
from scipy.special import kl_div

from opm_processing.imageprocessing.camera import camera_correct
from tests.simulation.camera import digitize

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("array_backend", ["gpu", "host"])
@pytest.mark.parametrize(
    "variance, observed_count, expected_multiplier",
    [(0, 90, 2), (4, 94, 2), (2.56, 0, 2.56 / 47.56)],
)
def test_full_data_rl_update_preserves_uniform_photon_rates(
    cupy_gpu,
    optical_setup,
    array_backend,
    variance,
    observed_count,
    expected_multiplier,
):
    """Apply the shared optical adjoint to analytically known uniform rates."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    xp = cupy_gpu if array_backend == "gpu" else np
    # A power-of-two constant field has only a DC Fourier coefficient. This
    # isolates scalar calibration arithmetic from mixed-radix FFT roundoff;
    # nonconstant optical-adjoint tests cover the general convolution operator.
    shape = tuple(1 << (2 * size).bit_length() for size in optical_setup.psf.shape)
    convolution = solver._Convolution(optical_setup.psf, shape)

    adjoint = partial(convolution, adjoint=True)
    observed = xp.full(shape, observed_count, dtype=xp.float32)
    denominator = xp.full(shape, 45 + variance, dtype=xp.float32)
    sensitivity = adjoint(xp.ones(shape, dtype=xp.float32))
    actual = solver._full_data_update(
        observed, denominator, adjoint, sensitivity, np.float32(variance)
    )
    first, second = solver._split_calibrated_counts(
        observed, xp.random.default_rng(42), np.float32(variance)
    )
    assert float(first.min()) >= 0 and float(second.min()) >= 0
    half_average = (
        adjoint(2 * first / denominator) + adjoint(2 * second / denominator)
    ) / (2 * sensitivity)
    np.testing.assert_allclose(
        cupy_gpu.asnumpy(half_average),
        cupy_gpu.asnumpy(actual),
        rtol=8 * np.finfo(np.float32).eps,
    )
    if array_backend == "gpu":
        actual = cupy_gpu.asnumpy(actual)
    np.testing.assert_allclose(
        actual, expected_multiplier, rtol=8 * np.finfo(np.float32).eps
    )


@pytest.mark.parametrize("factor", (None, 3), ids=("native", "sparse"))
@pytest.mark.parametrize("nonuniform", (True,))
@pytest.mark.parametrize("storage", ("gpu", "host"))
def test_gc_checkpoint_continues_same_optical_reconstruction(
    cupy_gpu, discrete_optical_case, camera_calibration, factor, nonuniform, storage
):
    """Resume the real optical inverse and random stream without repeated updates."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    case, camera = discrete_optical_case, camera_calibration
    observed = camera_correct(
        digitize(case.measured, camera, seed=31),
        camera.offset_adu,
        camera.electrons_per_adu,
    )
    options = {
        "scan_upsample_factor": factor or 1,
        "boundary": "reflect" if factor is None else "zero",
        "detector_response": case.response,
        "read_noise": camera.read_noise_e,
        "storage": storage,
    }
    steps = []
    state = solver._RlgcCheckpoint(slice(3, 7))
    solver._rlgc_once(
        observed,
        case.psf,
        _checkpoint=state,
        callback=lambda step, _: steps.append(step),
        **options,
    )
    assert state.first_kl_stop is not None
    assert state.iterations == state.first_kl_stop + 2
    assert isinstance(state.estimate, np.ndarray)
    if storage == "gpu":
        assert state.rng.bit_generator._state is None
        assert state.rng._binomial_state is None
    budget = state.iterations + 3
    resumed = solver._rlgc_once(
        observed,
        case.psf,
        _checkpoint=state,
        _finalize=True,
        max_iterations=budget,
        callback=lambda step, _: steps.append(step),
        **options,
    )
    uninterrupted = solver.rlgc(
        observed,
        case.psf,
        use_kl=False,
        limit=0,
        max_delta=0,
        max_iterations=budget,
        **options,
    )
    assert steps == list(range(1, budget + 1))
    np.testing.assert_array_equal(resumed, uninterrupted[state.retained_scan])
    assert state.estimate is state.rng is state.rng_buffers is None


@pytest.mark.parametrize("array_backend", ["host", "gpu"])
def test_consensus_uses_additive_object_changes(cupy_gpu, array_backend):
    """Paper Eqs. (4)–(5): weight before smoothing and freeze zero consensus."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    xp = np if array_backend == "host" else cupy_gpu
    observed = xp.asarray([[[3, 101, 101]]], dtype=xp.float32)
    first, second = solver._split_calibrated_counts(
        observed, xp.random.default_rng(42), np.float32(0)
    )
    # Identity optics make each additive half-step exactly 2*half - x.
    # Odd counts ensure unequal halves. Choose x between, on, and below the
    # two half-step targets to exercise negative, zero and positive consensus.
    initial = observed.copy()
    initial[..., 1] = 2 * first[..., 1]
    initial[..., 2] = xp.minimum(first[..., 2], second[..., 2])
    assert float(initial.min()) > 0
    expected_product = (2 * first - initial) * (2 * second - initial)
    assert float(expected_product[..., 0].item()) < 0
    assert float(expected_product[..., 1].item()) == 0
    assert float(expected_product[..., 2].item()) > 0

    def smooth(product, sensitivity):
        np.testing.assert_allclose(
            cupy_gpu.asnumpy(product), cupy_gpu.asnumpy(expected_product), rtol=2e-6
        )
        return product

    actual = solver._iterate_rlgc(
        observed,
        observed.shape,
        xp.copy,
        xp.copy,
        smooth,
        np.s_[:, :, :],
        gradient_consensus=True,
        max_iterations=1,
        limit=0,
        max_delta=0,
        rng_seed=42,
        use_kl=False,
        read_noise=0,
        initial_estimate=initial,
        logger=None,
        log_prefix="",
        callback=None,
    )
    expected = initial.copy()
    expected[..., 2] = observed[..., 2]
    np.testing.assert_allclose(
        cupy_gpu.asnumpy(actual), cupy_gpu.asnumpy(expected), rtol=2e-6
    )


@pytest.mark.parametrize("storage", ["host", "gpu"])
@pytest.mark.parametrize("normalize_psf", [True, False])
@pytest.mark.parametrize("boundary", ["reflect", "zero"])
def test_kl_preserves_finite_uniform_object(cupy_gpu, storage, normalize_psf, boundary):
    """KL rollback preserves object units at reflected and zero boundaries."""
    from opm_processing.imageprocessing.rlgc import rlgc

    measured = np.asarray([[[800, 1200, 800]]], dtype=np.float32)
    if boundary == "reflect":
        measured.fill(1200)
    if not normalize_psf:
        measured *= 3
    actual = rlgc(
        measured,
        np.ones((1, 1, 3), dtype=np.float32),
        boundary=boundary,
        storage=storage,
        normalize_psf=normalize_psf,
        max_iterations=100,
    )
    np.testing.assert_allclose(actual, 1200, rtol=2e-6, atol=2e-3)


@pytest.mark.parametrize("storage", ["host", "gpu"])
def test_gc_kl_scores_forward_prediction(
    cupy_gpu, storage, caplog, production_forward_predictions
):
    """Score Ax and restore the preceding iterate when both split KL scores rise."""
    from opm_processing.imageprocessing.rlgc import rlgc

    # Starting at M puts normalized x near both high-count data halves.
    # An RL step sharpens x away from M while bringing its blurred prediction
    # closer to M. Object KL and predicted-image KL therefore disagree here.
    initial = np.asarray([[[10000, 20000, 50000]]], dtype=np.float32)
    kernel = np.asarray([[[0.25, 0.5, 0.25]]], dtype=np.float32)
    steps = []
    actual = rlgc(
        initial,
        kernel,
        initial_estimate=initial,
        boundary="zero",
        storage=storage,
        max_iterations=4,
        limit=0,
        max_delta=0,
        rng_seed=42,
        callback=lambda number, values: steps.append(cupy_gpu.asnumpy(values).copy()),
    )
    assert len(steps) == 4
    assert not np.array_equal(steps[0], initial)
    np.testing.assert_array_equal(actual, steps[-1])
    prediction = production_forward_predictions[-1]
    assert (
        kl_div(initial, prediction["prediction"]).sum()
        < kl_div(initial, prediction["input_prediction"]).sum()
    )
    # Exercise termination with realizable optical data. The deliberately
    # incompatible data above distinguish prediction KL from object KL, but
    # their positive inverse need not reach a KL minimum.
    observed = fftconvolve(initial, kernel, mode="same").astype(np.float32)
    steps.clear()
    with caplog.at_level(logging.INFO, logger=__name__):
        actual = rlgc(
            observed,
            kernel,
            boundary="zero",
            storage=storage,
            max_iterations=100,
            limit=1,
            max_delta=1,
            rng_seed=42,
            logger=logging.getLogger(__name__),
            callback=lambda number, values: steps.append(
                cupy_gpu.asnumpy(values).copy()
            ),
        )
    assert any("stop=split_kl" in record.getMessage() for record in caplog.records)
    assert len(steps) >= 2
    np.testing.assert_array_equal(actual, steps[-2])


@pytest.mark.parametrize(
    "accepted,change,peak,reason",
    [
        (0, 2, 1000, "consensus_fraction"),
        (1, 2, 1000, None),
        (2, 2, 1000, None),
        (1000, 0.5, 1000, "relative_change"),
        (1000, 1, 1000, None),
        (1000, -2, 1000, None),
        (1, 0.5, 1000, "relative_change"),
        (1000, 1e-15, 1e-15, None),
        (1000, 5e-16, 5e-16, "relative_change"),
    ],
)
@pytest.mark.parametrize("array_backend", ["gpu", "host"])
def test_gc_stopping_uses_prescribed_fraction_and_change_limits(
    cupy_gpu, accepted, change, peak, reason, array_backend
):
    """Apply strict positive-consensus/absolute-change bounds and denominator floor."""
    from opm_processing.imageprocessing.rlgc import (
        GC_RELATIVE_CHANGE_LIMIT,
        GC_UPDATED_FRACTION_LIMIT,
        _gc_stopping_metrics,
    )

    xp = cupy_gpu if array_backend == "gpu" else np
    current = xp.full(1000, peak, dtype=xp.float64)
    previous = current.copy()
    previous[0] -= change
    consensus = xp.zeros(1000)
    consensus[:accepted] = 1
    fraction, delta, actual = _gc_stopping_metrics(
        previous,
        current,
        consensus,
        GC_UPDATED_FRACTION_LIMIT,
        GC_RELATIVE_CHANGE_LIMIT,
    )
    assert GC_UPDATED_FRACTION_LIMIT == GC_RELATIVE_CHANGE_LIMIT == 0.001
    assert fraction == accepted / 1000
    assert delta == pytest.approx(abs(change) / max(peak, 1e-12), rel=2e-15)
    assert actual == reason


@pytest.mark.parametrize("factor", [1, 3, 4])
@pytest.mark.parametrize("crop_scan", [None, 5])
def test_optical_forward_adjoint_matches_scipy(
    cupy_gpu, optical_setup, registration_scene, factor, crop_scan
):
    """Check the actual optical operator against independent CPU convolution."""
    cp = cupy_gpu
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    # Displace the calibrated optical origin by one scan sample to ensure the
    # adjoint test also detects using the unflipped, asymmetric PSF.
    psf = np.pad(optical_setup.psf, ((2, 0), (0, 0), (0, 0)))
    psf /= psf.sum()
    shape = (31, 17, 19)
    selection = (slice(None, None, factor), slice(None), slice(None))
    obj = registration_scene(shape, seed=34).raw.astype(np.float32)
    measured = registration_scene(obj[selection].shape, seed=35).raw.astype(np.float32)
    pads = solver._linear_fft_pad_width(shape, psf.shape)
    padded_shape = tuple(n + sum(p) for n, p in zip(shape, pads, strict=False))
    core = solver._observed_region_slices(padded_shape, pads)
    convolve = solver._Convolution(psf, padded_shape, crop_scan=crop_scan)
    padded = cp.zeros(padded_shape, dtype=cp.float32)
    padded[core] = cp.asarray(obj)
    forward = cp.asnumpy(convolve(padded)[core])[selection]
    scattered_gpu = cp.zeros(shape, dtype=cp.float32)
    scattered_gpu[selection] = cp.asarray(measured)
    padded[core] = scattered_gpu
    adjoint = cp.asnumpy(convolve(padded, adjoint=True)[core])
    np.testing.assert_allclose(
        forward, fftconvolve(obj, psf, mode="same")[selection], rtol=2e-5, atol=2e-6
    )
    scattered = np.zeros(shape)
    scattered[selection] = measured
    np.testing.assert_allclose(
        adjoint, fftconvolve(scattered, np.flip(psf), mode="same"), rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        np.sum(forward * measured), np.sum(obj * adjoint), rtol=2e-6
    )


@pytest.mark.parametrize("factor", (1, 3), ids=("native", "sparse3"))
@pytest.mark.parametrize(
    "crop_scan", (None, 12, 18), ids=("whole", "tiles12", "tiles18")
)
def test_tiled_gc_recovers_noisy_extended_objects_without_rl_noise(
    cupy_gpu,
    noisy_extended_scene,
    factor,
    crop_scan,
    caplog,
    production_forward_predictions,
):
    """Select actual KL stops in whole and tiled noisy objects before RL noise grows."""
    import re

    from opm_processing.imageprocessing.rlgc import rlgc
    from tests.simulation.metrics import camera_flux_accuracy

    microscope, expectation, psf, raw, source, dark = noisy_extended_scene
    camera = microscope.camera
    observed = camera_correct(
        raw[::factor], camera.offset_adu, camera.electrons_per_adu
    )
    options = {
        "crop_scan": source.shape[0] if crop_scan is None else crop_scan,
        "scan_upsample_factor": factor,
        "boundary": "reflect" if factor == 1 else "zero",
        "read_noise": camera.read_noise_e,
    }
    logger = logging.getLogger(__name__)
    caplog.set_level(logging.INFO, logger=__name__)
    recovered = rlgc(observed, psf, logger=logger, **options)
    measured = production_forward_predictions[-1]
    messages = "\n".join(
        record.getMessage() for record in caplog.records if record.name == __name__
    )
    stops = re.findall(r"stop=([^ ]+) iterations=(\d+)", messages)
    assert stops and {reason for reason, _ in stops} == {
        "split_kl" if crop_scan is None else "shared_split_kl"
    }
    iterations = {int(number) for _, number in stops}
    assert len(iterations) == 1
    iterations = iterations.pop()
    retained = sorted(
        {
            (int(start), int(stop))
            for start, stop in re.findall(
                r"retained_fine_scan=\[(\d+):(\d+)\]", messages
            )
        }
    )
    boundaries = {stop for _, stop in retained if stop < source.shape[0]}
    if crop_scan is not None:
        assert sum(source[b - 1].any() and source[b].any() for b in boundaries) >= 2
        assert retained[-1][1] - retained[-1][0] < retained[0][1] - retained[0][0]
    # Nearest acquired samples on either side, including every refined-grid
    # phase. PSF-support neighborhoods overlap throughout these small chunks.
    interface = np.zeros(source.shape[0], dtype=bool)
    for boundary_index in boundaries:
        interface[boundary_index - factor : boundary_index + factor] = True
    logger.info(
        "crop_scan=%s shape=%s psf_shape=%s boundaries=%s iterations=%d",
        crop_scan,
        source.shape,
        psf.shape,
        sorted(boundaries),
        iterations,
    )
    ordinary = rlgc(
        observed,
        psf,
        gradient_consensus=False,
        use_kl=False,
        max_iterations=iterations,
        limit=0,
        max_delta=0,
        **options,
    )
    predicted = measured["prediction"]
    baseline_prediction = measured["input_prediction"]
    if baseline_prediction is None:
        # Acquired-plane accuracy compares progress from the production
        # initializer, rather than a separately interpolated/reblurred image.
        baseline_prediction = measured["initial_prediction"]
    aperture = source[::factor]
    regions = (
        (("source", np.ones(source.shape[0], dtype=bool)),)
        if crop_scan is None
        else (("interface", interface), ("interior", ~interface))
    )
    for name, scan_mask in regions:
        region = aperture & scan_mask[::factor, None, None]
        assert region.any(), name
        true = expectation[::factor][region]
        actual = predicted[region]
        blurred = baseline_prediction[region]
        gc_error = kl_div(true, actual).sum(dtype=np.float64)
        baseline_error = kl_div(true, blurred).sum(dtype=np.float64)
        logger.info(
            "crop_scan=%s region=%s acquired_samples=%d kl_gc=%.9g kl_baseline=%.9g bias_e=%.9g rmse_e=%.9g",
            crop_scan,
            name,
            region.sum(),
            gc_error,
            baseline_error,
            np.mean(actual - true, dtype=np.float64),
            np.sqrt(np.mean((actual - true) ** 2, dtype=np.float64)),
        )
        assert gc_error < baseline_error, (name, gc_error, baseline_error)
    logger.info(
        "acquired_source_flux_6sigma=%s",
        camera_flux_accuracy(predicted, expectation[::factor], camera, aperture),
    )
    for phase in range(factor):
        aperture = source[phase::factor]
        assert aperture.any()
        background = dark[phase::factor]
        assert background.any()
        gc_noise = np.std(recovered[phase::factor][background], dtype=np.float64)
        rl_noise = np.std(ordinary[phase::factor][background], dtype=np.float64)
        logger.info(
            "phase=%d dark_std_gc=%.6g dark_std_rl=%.6g",
            phase,
            gc_noise,
            rl_noise,
        )
        assert gc_noise <= rl_noise, (phase, gc_noise, rl_noise)
