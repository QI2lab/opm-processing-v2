"""Physical image-formation tests for the separate undersampled RL experiment.

The microscope is the existing 100x silicone model: NA 1.35, immersion RI 1.40,
sample RI 1.38, emission 610 nm, camera pixels 115 nm, and OPM angle 30 degrees.
The PSF comes from the vectorial diffraction model, never a Gaussian kernel.
Fluorescent spheres and a finite filament are defined in laboratory microns
and voxel-integrated in the skew acquisition coordinates. Independent SciPy
linear convolution produces photon expectations before scan-plane selection.

Full sampling is 0.2 um; integer undersampling is 0.6 or 0.8 um with unchanged
exposure per plane. Thus undersampling also reduces total collected photons.
We do not blur a decimated specimen: that would delete real fluorophores.
These are tests of the stated shift-invariant, zero-background optical model,
not a claim to model scattering, illumination variations or camera read noise.
"""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
from scipy import signal
from typer.testing import CliRunner

from opm_processing.dataio.position_collection import open_position_collection
from opm_processing.dataio.processing_state import ProcessingState
from tests.fixtures.acquisition import (
    simulated_acquisition_metadata,
    write_simulated_acquisition,
)
from tests.reference.point_sources import skewed_coordinates


def _blur(truth, psf):
    """Independent zero-exterior incoherent fluorescence image formation."""
    return np.maximum(signal.fftconvolve(truth.astype(np.float64), psf, mode="same"), 0)


def _interpolate_scan(image, factor):
    """Linear interpolation baseline with exactly the acquired endpoints."""
    positions = np.arange((image.shape[0] - 1) * factor + 1) / factor
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, image.shape[0] - 1)
    alpha = (positions - lower)[:, None, None]
    return image[lower] * (1 - alpha) + image[upper] * alpha


def _nrmse(actual, expected):
    """Measure absolute fluorescence error without fitting a gain or shift."""
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


@pytest.mark.unit
@pytest.mark.parametrize("factor", [3, 4])
@pytest.mark.parametrize("poisson", [False, True], ids=["noise_free", "photon_noise"])
def test_physical_full_and_undersampled_rl(
    cupy_gpu, optical_psf, specimen, factor, poisson
):
    """Recover physical features and missing planes relative to full-data RL."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc_undersampled")
    truth, beads = specimen
    expectation = _blur(truth, optical_psf)
    rng = np.random.default_rng(610)
    full_data = rng.poisson(expectation) if poisson else expectation
    # The same specimen, PSF, exposure and scan origin; no coarse-grid reblur.
    sparse_data = full_data[::factor].copy()
    kwargs = {"gradient_consensus": False, "max_iterations": 60, "max_delta": 0}
    dense = solver.rlgc_undersampled(full_data, optical_psf, **kwargs)
    recovered = solver.rlgc_undersampled(sparse_data, optical_psf, factor, **kwargs)
    # Sample the same optical PSF about its center at the coarse spacing.
    # This baseline performs conventional RL directly on the acquired grid.
    center = optical_psf.shape[0] // 2
    coarse_psf = optical_psf[center % factor :: factor].copy()
    coarse_psf /= coarse_psf.sum()
    native = solver.rlgc_undersampled(sparse_data, coarse_psf, **kwargs)
    native_interpolated = _interpolate_scan(native, factor)
    baseline = _interpolate_scan(sparse_data, factor)
    assert recovered.shape == truth.shape
    assert recovered.dtype == np.float32
    assert np.isfinite(recovered).all() and np.min(recovered) >= 0
    missing = np.arange(truth.shape[0]) % factor != 0
    reblurred = _blur(recovered, optical_psf)
    metrics = {
        "scan_step_um": 0.2 * factor,
        "dense_nrmse": _nrmse(dense, truth),
        "undersampled_nrmse": _nrmse(recovered, truth),
        "native_coarse_rl_nrmse": _nrmse(native_interpolated, truth),
        "interpolation_nrmse": _nrmse(baseline, truth),
        "difference_from_dense_rl": _nrmse(recovered, dense),
        "missing_planes_nrmse": _nrmse(recovered[missing], truth[missing]),
        "missing_planes_interpolation_nrmse": _nrmse(baseline[missing], truth[missing]),
        "missing_planes_reblur_nrmse": _nrmse(reblurred[missing], expectation[missing]),
        "dense_reblur_nrmse": _nrmse(_blur(dense, optical_psf), expectation),
        "measured_planes_reblur_nrmse": _nrmse(
            reblurred[::factor], expectation[::factor]
        ),
        "flux_ratio": float(recovered.sum() / truth.sum()),
    }
    # Feature localization is checked in physical coordinates, so swapping
    # scan/camera Y, changing origin, or a wrong shear cannot pass on MSE alone.
    xyz = skewed_coordinates(truth.shape)
    errors = []
    for bead in beads:
        region = (
            sum((coord - c) ** 2 for coord, c in zip(xyz, bead, strict=False)) < 0.6**2
        )
        mass = recovered * region
        centroid = np.array([np.sum(mass * coord) / mass.sum() for coord in xyz])
        truth_mass = truth * region
        truth_centroid = np.array(
            [np.sum(truth_mass * coord) / truth_mass.sum() for coord in xyz]
        )
        errors.append(float(np.linalg.norm(centroid - truth_centroid)))
    metrics["bead_centroid_errors_um"] = errors
    print(json.dumps(metrics))
    # These tolerances allow the expected loss of information, not a claim
    # that arbitrary missing high spatial frequencies can be recovered.
    assert metrics["dense_nrmse"] < metrics["interpolation_nrmse"] * 0.8
    assert metrics["undersampled_nrmse"] < metrics["interpolation_nrmse"] * 0.9
    assert metrics["undersampled_nrmse"] < metrics["native_coarse_rl_nrmse"] * 0.95
    assert metrics["undersampled_nrmse"] < metrics["dense_nrmse"] * 1.6
    assert metrics["difference_from_dense_rl"] < 0.15
    assert (
        metrics["missing_planes_nrmse"]
        < metrics["missing_planes_interpolation_nrmse"] * 0.9
    )
    # A finite-iteration full-data RL reconstruction also has residual blur.
    # Limit absolute error and additional error from withholding observations.
    assert metrics["missing_planes_reblur_nrmse"] < 0.1
    assert metrics["measured_planes_reblur_nrmse"] < 0.1
    assert (
        metrics["measured_planes_reblur_nrmse"] < 1.25 * metrics["dense_reblur_nrmse"]
    )
    assert 0.95 < metrics["flux_ratio"] < 1.05
    assert max(errors) < 0.12


@pytest.mark.unit
def test_physical_gradient_consensus(cupy_gpu, optical_psf, specimen):
    """Exercise the reused GC loop with photon noise and unmeasured planes."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc_undersampled")
    truth, _ = specimen
    data = np.random.default_rng(32).poisson(_blur(truth, optical_psf))
    results = []
    for factor in (1, 3):
        reconstructed = solver.rlgc_undersampled(
            data[::factor], optical_psf, factor, max_iterations=60
        )
        assert reconstructed.shape == truth.shape
        assert np.isfinite(reconstructed).all() and reconstructed.min() >= 0
        assert _nrmse(reconstructed, truth) < 0.9 * _nrmse(
            _interpolate_scan(data[::factor], factor), truth
        )
        results.append(reconstructed)
    assert _nrmse(results[1], truth) < 1.6 * _nrmse(results[0], truth)


@pytest.mark.unit
@pytest.mark.parametrize("factor", [1, 3, 4])
def test_optical_forward_adjoint_and_cpu_rl(cupy_gpu, optical_psf, factor):
    """Check the actual optical operator against independent CPU convolution."""
    cp = cupy_gpu
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc_undersampled")
    # Displace the calibrated optical origin by one scan sample to ensure the
    # adjoint test also detects using the unflipped, asymmetric PSF.
    psf = np.pad(optical_psf, ((2, 0), (0, 0), (0, 0)))
    psf /= psf.sum()
    shape = (13, 11, 9)
    rng = np.random.default_rng(34)
    obj = rng.uniform(1, 10, shape).astype(np.float32)
    measured = rng.uniform(1, 10, obj[::factor].shape).astype(np.float32)
    pads = solver._linear_fft_pad_width(shape, psf.shape)
    padded_shape = tuple(n + sum(p) for n, p in zip(shape, pads, strict=False))
    core = solver._observed_region_slices(padded_shape, pads)
    otf = cp.fft.rfftn(solver.pad_psf(cp.asarray(psf), padded_shape))
    forward = cp.asnumpy(
        solver._forward(cp.asarray(obj), otf, padded_shape, core, factor)
    )
    adjoint = cp.asnumpy(
        solver._adjoint(
            cp.asarray(measured), otf.conj(), padded_shape, core, shape, factor
        )
    )
    np.testing.assert_allclose(forward, _blur(obj, psf)[::factor], rtol=2e-5, atol=2e-6)
    scattered = np.zeros(shape)
    scattered[::factor] = measured
    np.testing.assert_allclose(
        adjoint, _blur(scattered, psf[::-1, ::-1, ::-1]), rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        np.sum(forward * measured), np.sum(obj * adjoint), rtol=2e-6
    )

    mask = np.zeros(shape)
    mask[::factor] = 1
    sensitivity = _blur(mask, psf[::-1, ::-1, ::-1])
    expected = _blur(scattered, psf[::-1, ::-1, ::-1]) / sensitivity
    for _ in range(3):
        ratio = np.zeros(shape)
        ratio[::factor] = measured / _blur(expected, psf)[::factor]
        expected *= _blur(ratio, psf[::-1, ::-1, ::-1]) / sensitivity
    actual = solver.rlgc_undersampled(
        measured, psf, factor, gradient_consensus=False, max_iterations=3, max_delta=0
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["mirror", "stage"])
def test_combined_channel_cli_reconstructs_physical_specimen(
    cupy_gpu, sparse_optical_volume, mode, tmp_path
):
    """Reconstruct physical beads from camera data on disk to deskewed data on disk."""
    process = importlib.import_module("opm_processing.process")
    truth, beads, known_psf, _expectation, photons = sparse_optical_volume
    # An ideal calibrated camera: 100 ADU offset, 0.5 photons/ADU.
    raw = (photons * 2 + 100).astype(np.uint16)
    if mode == "stage":
        raw = raw[::-1].copy()
    raw = raw[None, None, None]
    metadata = simulated_acquisition_metadata(
        tmp_path / "both_lasers.ome.zarr", mode, raw.shape
    )
    write_simulated_acquisition(metadata, raw)
    output = tmp_path
    psf_path = tmp_path / "independent_637nm_psf.npy"
    np.save(psf_path, known_psf)
    result = CliRunner().invoke(
        process.app,
        [
            str(metadata.path),
            "--deconvolve",
            "--decon-psf-paths",
            str(psf_path),
            "--decon-scan-upsample",
            "2",
            "--save-float32",
            "--no-max-projection",
            "--z-downsample-level",
            "1",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    processed_path = output / "both_lasers_decon_deskewed.ome.zarr"
    collection = open_position_collection(processed_path)
    state = ProcessingState.read(output / "both_lasers.processing.json")
    actual = collection.arrays[0][0, 0].read().result()
    assert np.isfinite(actual).all() and actual.min() >= 0
    assert collection.voxel_size_um == (0.115, 0.115, 0.115)
    np.testing.assert_array_equal(
        collection.stage_positions_zxy, metadata.stage_positions_zxy
    )
    assert state.completed_tiles(processed_path) == {(0, 0)}
    assert state.run(processed_path)["reconstruction"] == {
        "shape_syx": list(truth.shape),
        "scan_axis_step_um": 0.4,
    }
    # Independent Cartesian coordinates: each output voxel is 115 nm. Compare
    # bead centers in microns, not against a second call to the deskew routine.
    deskew_gain = 2 * 0.115 / 0.4
    saved_fluorescence = actual.sum(dtype=np.float64) * 0.115**3 / deskew_gain
    object_fluorescence = truth.sum(dtype=np.float64) * 0.4 * 0.115**2 * 0.5
    assert abs(saved_fluorescence / object_fluorescence - 1) < 0.1
    z, y, x = np.ogrid[: actual.shape[0], : actual.shape[1], : actual.shape[2]]
    xyz = (
        x * 0.115 - (truth.shape[2] - 1) / 2 * 0.115,
        y * 0.115
        - (
            (truth.shape[0] - 1) / 2 * 0.4
            + (truth.shape[1] - 1) / 2 * 0.115 * np.cos(np.pi / 6)
        ),
        z * 0.115 - (truth.shape[1] - 1) / 2 * 0.115 * np.sin(np.pi / 6),
    )
    for bead in beads:
        region = (
            sum((coord - c) ** 2 for coord, c in zip(xyz, bead, strict=False)) < 0.6**2
        )
        mass = actual * region
        assert mass.sum() > 0
        centroid = np.array([np.sum(mass * coord) / mass.sum() for coord in xyz])
        assert np.linalg.norm(centroid - bead) < 0.2


@pytest.mark.unit
def test_sparse_optical_beads_recover_flux_and_predict_withheld_planes(
    cupy_gpu, sparse_optical_volume
):
    """Check acquisition-grid accuracy independently of disk writing and deskewing."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc_undersampled")
    truth, _, known_psf, expectation, photons = sparse_optical_volume
    recovered = solver.rlgc_undersampled(photons.astype(np.float32), known_psf, 2)
    assert recovered.shape == truth.shape
    baseline = _interpolate_scan(photons, 2)
    metrics = {
        "truth_nrmse": _nrmse(recovered, truth),
        "interpolation_nrmse": _nrmse(baseline, truth),
        "withheld_prediction_nrmse": _nrmse(
            _blur(recovered, known_psf)[1::2], expectation[1::2]
        ),
        "flux_ratio": float(recovered.sum() / truth.sum()),
    }
    print(json.dumps(metrics))
    assert metrics["truth_nrmse"] < 0.9 * metrics["interpolation_nrmse"]
    assert metrics["withheld_prediction_nrmse"] < 0.2
    assert 0.9 < metrics["flux_ratio"] < 1.1
