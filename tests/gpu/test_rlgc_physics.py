"""Reconstruction physics using the shared fine-Cartesian camera forward model.

Numerical assertions use production operators, SciPy and the shared simulator.
"""

import importlib

import numpy as np
import pytest
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

from opm_processing.imageprocessing.camera import camera_correct
from tests.simulation.camera import digitize
from tests.simulation.metrics import (
    camera_flux_accuracy,
    camera_prediction,
    camera_score_difference,
    predictive_deviance,
    two_point_profile,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("array_backend", ["gpu", "host"])
@pytest.mark.parametrize("variance, observed_count", [(0, 90), (4, 94)])
def test_full_data_rl_update_preserves_uniform_photon_rates(
    cupy_gpu, optical_setup, array_backend, variance, observed_count
):
    """Apply the shared optical adjoint to analytically known uniform rates."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    xp = cupy_gpu if array_backend == "gpu" else np
    # A power-of-two constant field has only a DC Fourier coefficient. This
    # isolates scalar calibration arithmetic from mixed-radix FFT roundoff;
    # nonconstant optical-adjoint tests cover the general convolution operator.
    shape = tuple(1 << (2 * size).bit_length() for size in optical_setup.psf.shape)
    convolution = solver._Convolution(optical_setup.psf, shape)

    def adjoint(values):
        """Execute the actual optical transpose on the selected state backend."""
        return convolution(values, adjoint=True)

    observed = xp.full(shape, observed_count, dtype=xp.float32)
    denominator = xp.full(shape, 45 + variance, dtype=xp.float32)
    sensitivity = adjoint(xp.ones(shape, dtype=xp.float32))
    actual = solver._full_data_update(
        observed, denominator, adjoint, sensitivity, np.float32(variance)
    )
    if array_backend == "gpu":
        actual = cupy_gpu.asnumpy(actual)
    np.testing.assert_allclose(actual, 2, rtol=8 * np.finfo(np.float32).eps)


def test_generated_psf_camera_reconstruction(cupy_gpu, production_point_acquisition):
    """Apply production GC using the same PSF as the full camera simulation."""
    from opm_processing.imageprocessing.rlgc import rlgc

    case = production_point_acquisition
    expected = case.expectations[-1]
    camera = case.microscope.camera
    calibrated = camera_correct(
        digitize(expected, camera, seed=42), camera.offset_adu, camera.electrons_per_adu
    )
    recovered = rlgc(
        calibrated,
        case.kernel,
        read_noise=camera.read_noise_e,
        boundary="zero",
        rng_seed=42,
    )
    assert np.isfinite(recovered).all() and recovered.min() >= 0
    assert predictive_deviance(recovered, expected, case.kernel) < predictive_deviance(
        calibrated, expected, case.kernel
    )
    prediction = camera_prediction(recovered, case.kernel, boundary="constant")
    for name, aperture in zip(
        ("whole", "source", "background"), case.apertures, strict=True
    ):
        assert camera_flux_accuracy(prediction, expected, camera, aperture), name
    score = camera_score_difference(
        digitize(expected, camera, seed=43),
        prediction,
        camera_prediction(calibrated, case.kernel),
        camera,
    )
    assert score.sum() > 0


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
def test_gc_stopping_matches_matlab_reference(
    cupy_gpu, accepted, change, peak, reason, array_backend
):
    """Lock MATLAB's strict bounds, absolute change, zero gate and denominator floor."""
    from opm_processing.imageprocessing.rlgc import (
        GC_RELATIVE_CHANGE_LIMIT,
        GC_UPDATED_FRACTION_LIMIT,
        _gc_stopping_metrics,
    )

    xp = cupy_gpu if array_backend == "gpu" else np
    current = xp.full(1000, peak, dtype=xp.float64)
    previous = current.copy()
    previous[0] -= change
    consensus = xp.full(1000, -1.0)
    consensus[:accepted] = 0
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


@pytest.mark.parametrize("factor", (None, 3, 4), ids=("native", "sparse3", "sparse4"))
@pytest.mark.parametrize("camera_mode", (False, True), ids=("ideal", "camera-3000"))
def test_native_gc_directional_rayleigh_pair(
    cupy_gpu,
    reconstruct,
    rayleigh_pair,
    camera_exposure,
    factor,
    camera_mode,
    camera_calibration,
    ideal_photon_camera,
):
    """Resolve noisy pairs placed at the directional optical first-minimum spacing."""
    case = rayleigh_pair
    expectation = case.expectation
    if camera_mode:
        expectation = camera_exposure(expectation, 3000)
        assert expectation.max() == pytest.approx(3000, rel=2e-15)
        camera = camera_calibration
        observed = digitize(expectation, camera, seed=31)
        calibrated = camera_correct(
            observed, camera.offset_adu, camera.electrons_per_adu
        )
    else:
        observed = digitize(expectation, ideal_photon_camera, seed=31).astype(
            np.float32
        )
        calibrated = observed
    recovered = reconstruct(
        calibrated,
        case.psf,
        factor,
        read_noise=camera.read_noise_e if camera_mode else 0,
    )
    baseline = calibrated
    if factor is not None:
        acquired = calibrated[::factor]
        positions = np.arange(acquired.shape[0]) * factor
        baseline = interp1d(positions, acquired, axis=0)(np.arange(positions[-1] + 1))
    center = (np.array(observed.shape) - 1) / 2
    measured = two_point_profile(recovered, center, case.offset, case.microscope)
    metrics = {
        "axis": case.axis,
        "scan_factor": factor,
        "rayleigh_separation_um": case.separation,
        "sampled_rayleigh_dip": case.acquisition_dip,
        "measured_dip": measured["valley_depth"],
        "measured_positions_um": measured["peak_positions_um"],
        "predictive_deviance": predictive_deviance(recovered, expectation, case.psf),
        "baseline_deviance": predictive_deviance(baseline, expectation, case.psf),
    }
    print(metrics)
    assert np.isfinite(recovered).all() and recovered.min() >= 0
    assert metrics["predictive_deviance"] < metrics["baseline_deviance"], metrics
    # Sparse scans need not preserve the information required by the Rayleigh
    # pair. Report their measured dip rather than assigning a fictitious limit.
    if factor is None:
        assert measured["peak_positions_um"] is not None, metrics
        # Rayleigh fixes source separation through the single-point first minimum.
        # It does not require a noisy reconstruction's dip to exceed the noiseless
        # acquisition's dip. Peak positions and likelihood test the recovered pair.
        # Profile interpolation cannot establish positions more finely than the
        # acquisition-grid projection along this direction.
        interval = case.separation / (2 * max(abs(case.offset)))
        np.testing.assert_allclose(
            measured["peak_positions_um"],
            [-case.separation / 2, case.separation / 2],
            rtol=0,
            atol=interval,
            err_msg=str(metrics),
        )


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
