"""Independent optical and boundary checks for production gradient consensus.

The optical field is evaluated on a 40 nm Cartesian grid at 637 nm and
integrated over detector pixels before sampling onto the 30 degree acquisition
grid. Fluorescent beads and a filament have known laboratory coordinates.
SciPy simulates image formation without production PSF or FFT helpers.
"""

import importlib
import json
import logging
from dataclasses import replace

import numpy as np
import pytest
from scipy import ndimage, signal

from opm_processing.dataio.position_collection import open_position_collection
from opm_processing.process import process
from tests.fixtures.acquisition import (
    simulated_acquisition_metadata,
    write_simulated_acquisition,
)
from tests.reference.point_sources import (
    skewed_coordinates,
    two_point_profile,
)


@pytest.mark.unit
@pytest.mark.parametrize("seed", (None, 31, 83))
@pytest.mark.parametrize(
    "factor", (None, 1, 3, 4), ids=("native", "dense", "sparse3", "sparse4")
)
def test_native_gc_recovers_independent_optical_specimen(
    cupy_gpu, physical_volume, seed, tmp_path, factor
):
    """Recover known object fluorescence and physical bead positions with native GC.

    Parameters
    ----------
    cupy_gpu
        Fixture requiring a working CUDA device.
    physical_volume
        Known specimen, bead centers, finite reconstruction PSF, and full optical data.
    seed : int | None
        Poisson realization, or None for noiseless photon expectations.
    tmp_path
        Directory for the measured object arrays and physical error report.
    factor : int | None
        Measured-plane interval for the sensitivity-corrected solver, or None
        for native reflected-boundary reconstruction.
    """
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    truth, beads, psf, expectation = physical_volume
    observed = (
        expectation
        if seed is None
        else np.random.default_rng(seed).poisson(expectation)
    )
    if factor is None:
        recovered = solver.rlgc(observed.astype(np.float32), psf)
    else:
        from opm_processing.imageprocessing.rlgc_undersampled import rlgc_undersampled

        recovered = rlgc_undersampled(
            observed[::factor].astype(np.float32),
            psf,
            factor,
        )
    reference_error = np.linalg.norm(observed - truth)
    metrics = {
        "seed": seed,
        "factor": factor,
        "object_nrmse": float(
            np.linalg.norm(recovered - truth) / np.linalg.norm(truth)
        ),
        "measured_nrmse": float(
            np.linalg.norm(observed - truth) / np.linalg.norm(truth)
        ),
        "recovery_error_ratio": float(
            np.linalg.norm(recovered - truth) / reference_error
        ),
        "flux_ratio": float(recovered.sum() / truth.sum()),
    }
    xyz = skewed_coordinates(truth.shape)
    localization_errors = []
    for center in beads:
        region = (
            sum((coord - c) ** 2 for coord, c in zip(xyz, center, strict=False))
            < 0.6**2
        )
        mass = np.maximum(recovered, 0) * region
        centroid = np.asarray([np.sum(mass * coord) / mass.sum() for coord in xyz])
        localization_errors.append(float(np.linalg.norm(centroid - center)))
    metrics["localization_errors_um"] = localization_errors
    print(json.dumps(metrics))
    np.savez(
        tmp_path / "ground-object.npz",
        truth=truth,
        observed=observed,
        recovered=recovered,
        psf=psf,
    )
    (tmp_path / "measurements.json").write_text(json.dumps(metrics), encoding="utf-8")
    assert metrics["recovery_error_ratio"] < 0.9, metrics
    assert abs(metrics["flux_ratio"] - 1) < 0.1, metrics
    assert max(localization_errors) < 0.15, metrics
    assert np.isfinite(recovered).all() and recovered.min() >= 0


@pytest.mark.unit
@pytest.mark.parametrize("axis", ("x", "y", "z"))
@pytest.mark.parametrize("photons_per_point", (10_000, 100_000))
@pytest.mark.parametrize("factor", (None, 3, 4), ids=("native", "sparse3", "sparse4"))
def test_native_gc_two_point_resolution(
    cupy_gpu, physical_volume, axis, photons_per_point, tmp_path, factor
):
    """Measure the same optical emitter pairs under full and sparse sampling.

    Parameters
    ----------
    cupy_gpu
        Fixture requiring a working CUDA device before reconstructing.
    physical_volume
        Independent detector-integrated optical PSF and acquisition-grid shape.
    axis : str
        Laboratory direction of the pair. Z pairs include compensating scan
        displacement and report their actual physical separation.
    photons_per_point : int
        Expected photons from each emitter before finite acquisition cropping.
    tmp_path
        Directory for physical line profiles and pair measurements.
    factor : int | None
        Select every third or fourth measured plane, or use native full sampling.

    Notes
    -----
    Measurements report raw and reconstructed peak positions, separation and
    valley depth across acquisition-grid and wider spacings. Closely
    spaced pairs may remain unresolved. Widest-pair measurements must retain
    two localized peaks and absolute fluorescence. No gain or translation is
    fitted, and no minimum resolution is assigned without a measured dip.
    """
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    specimen, _, psf, _ = physical_volume
    origin = np.array(specimen.shape) // 2
    displacements = {
        "x": (
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 3),
            (0, 0, 4),
            (0, 0, 6),
            (0, 0, 8),
            (0, 0, 12),
        ),
        "y": ((1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (6, 0, 0), (8, 0, 0)),
        "z": ((-2, 4, 0), (-4, 8, 0), (-6, 12, 0), (-8, 16, 0), (-12, 24, 0)),
    }[axis]
    for displacement in displacements:
        displacement = np.array(displacement)
        left = origin - displacement // 2
        right = left + displacement
        center = (left + right) / 2
        offset = displacement / 2
        truth = np.zeros(specimen.shape, np.float32)
        truth[tuple(left)] = photons_per_point
        truth[tuple(right)] = photons_per_point
        expectation = np.maximum(signal.fftconvolve(truth, psf, mode="same"), 0)
        observed = np.random.default_rng(31).poisson(expectation).astype(np.float32)
        if factor is None:
            recovered = solver.rlgc(observed, psf)
        else:
            from opm_processing.imageprocessing.rlgc_undersampled import (
                rlgc_undersampled,
            )

            recovered = rlgc_undersampled(
                observed[::factor],
                psf,
                factor,
            )
        profiles = {
            name: two_point_profile(volume, center, offset)
            for name, volume in (("raw", observed), ("reconstructed", recovered))
        }
        measurements = {
            "axis": axis,
            "factor": factor,
            "photons_per_point": photons_per_point,
            "seed": 31,
            "recovery_error_ratio": float(
                np.linalg.norm(recovered - truth) / np.linalg.norm(observed - truth)
            ),
            "flux_ratio": float(recovered.sum() / truth.sum()),
            "profiles": {
                name: {
                    key: value
                    for key, value in measured.items()
                    if key not in ("distance_um", "profile")
                }
                for name, measured in profiles.items()
            },
        }
        print(json.dumps(measurements))
        np.savez(
            tmp_path / f"{axis}-{max(map(abs, displacement))}.npz",
            distance_um=profiles["raw"]["distance_um"],
            raw_profile=profiles["raw"]["profile"],
            reconstructed_profile=profiles["reconstructed"]["profile"],
            recovered=recovered,
            measurements=json.dumps(measurements),
        )
        assert np.isfinite(recovered).all() and recovered.min() >= 0
        widest = np.array_equal(displacement, displacements[-1])
        if widest:
            np.testing.assert_allclose(recovered.sum(), truth.sum(), rtol=0.1)
            measured = profiles["reconstructed"]
            separation = measured["true_separation_um"]
            assert measured["peak_positions_um"] is not None, measurements
            acquisition_interval = separation / (2 * max(map(abs, offset)))
            np.testing.assert_allclose(
                measured["peak_positions_um"],
                (-separation / 2, separation / 2),
                atol=acquisition_interval,
                rtol=0,
                err_msg=str(measurements),
            )


@pytest.mark.unit
@pytest.mark.parametrize("safe_mode", (True, False))
@pytest.mark.parametrize("termination", ("rollback", "limit", "max_delta"))
def test_production_boundary_and_consensus_match_cpu_reference(
    cupy_gpu, monkeypatch, caplog, safe_mode, termination
):
    """Check the complete update and returned iterate against direct CPU convolution.

    Parameters
    ----------
    cupy_gpu
        Fixture requiring a working CUDA device.
    monkeypatch
        Fixture replacing random draws with independently prescribed splits.
    caplog
        Fixture capturing the solver's scientific stopping decisions.
    safe_mode : bool
        Roll back when either split worsens, or only when both worsen.
    termination : str
        Exercise rollback, updated-fraction stopping, or relative-delta stopping.
    """
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    observed = np.random.default_rng(7).integers(5, 80, (6, 8, 10)).astype(np.float32)
    psf = np.arange(1, 28, dtype=np.float64).reshape(3, 3, 3)
    psf /= psf.sum()
    splits = []
    rng = np.random.default_rng(17)

    padded = np.pad(observed, 1, mode="symmetric")

    def prescribed_split(values, generator, *, assign_fractional_remainder=False):
        """Supply known complementary binomial draws for both numerical operators.

        Parameters
        ----------
        values
            Measured device counts; only their independently known shape is used.
        generator
            Solver generator, superseded by the prescribed CPU draws.
        assign_fractional_remainder : bool
            Reference mode, unused because these prescribed measurements are integers.

        Returns
        -------
        cupy.ndarray
            Independent CPU binomial draw transferred to the device.
        """
        if termination == "rollback" and splits:
            delta = ndimage.convolve(padded, psf[::-1, ::-1, ::-1], mode="wrap")
            split = np.where(delta > delta.mean(), padded, 0)
        else:
            split = rng.binomial(padded.astype(np.int64), 0.5)
        splits.append(split)
        return cupy_gpu.asarray(split, dtype=cupy_gpu.float32)

    monkeypatch.setattr(solver, "_split_observed_counts", prescribed_split)
    limit = 1.01 if termination == "limit" else 0
    max_delta = 2 if termination == "max_delta" else 0
    logger = logging.getLogger("rlgc-rule-test")
    with caplog.at_level(logging.INFO, logger=logger.name):
        actual = solver.rlgc(
            observed,
            psf.astype(np.float32),
            safe_mode=safe_mode,
            limit=limit,
            max_delta=max_delta,
            logger=logger,
        )
    core = np.s_[1:-1, 1:-1, 1:-1]
    estimate = np.maximum(
        ndimage.convolve(padded, psf[::-1, ::-1, ::-1], mode="wrap"), 0
    )
    estimate = np.maximum(estimate, estimate.max() * 1e-7).astype(np.float64)
    previous = None
    previous_prediction = None
    for split in splits:
        predicted = ndimage.convolve(estimate, psf, mode="wrap")
        worsened = []
        for half in (split, padded - split):
            target = half + 1e-4
            target /= target.sum()
            probability = predicted + 1e-4
            probability /= probability.sum()
            score = np.sum(target * np.log(target / probability))
            old_score = np.inf
            if previous_prediction is not None:
                old = previous_prediction + 1e-4
                old /= old.sum()
                old_score = np.sum(target * np.log(target / old))
            worsened.append(score > old_score)
        if any(worsened) if safe_mode else all(worsened):
            estimate = previous
            expected_stop = "restore_previous_recon"
            break
        previous_prediction = predicted
        ratios = []
        for half in (split, padded - split):
            ratio = half / (0.5 * (predicted + 1e-12))
            ratios.append(ndimage.convolve(ratio, psf[::-1, ::-1, ::-1], mode="wrap"))
        consensus = ndimage.convolve(
            ndimage.convolve((ratios[0] - 1) * (ratios[1] - 1), psf, mode="wrap"),
            psf[::-1, ::-1, ::-1],
            mode="wrap",
        )
        updated = np.where(
            consensus >= 0, estimate * (ratios[0] + ratios[1]) * 0.5, estimate
        )
        previous, estimate = estimate, updated
        if np.mean(consensus >= 0) < limit:
            expected_stop = "limit"
            break
        if np.max(np.abs(estimate - previous) / max(estimate.max(), 1e-12)) < max_delta:
            expected_stop = "max_delta"
            break
    else:
        expected_stop = "max_iterations"
    np.testing.assert_allclose(actual, estimate[core], rtol=2e-4, atol=2e-4)
    assert f"stop={expected_stop}" in caplog.text
    assert expected_stop == (
        ("restore_previous_recon" if safe_mode else "max_iterations")
        if termination == "rollback"
        else termination
    )
    assert len(splits) > 1 if termination == "rollback" else len(splits) == 1


@pytest.mark.integration
@pytest.mark.parametrize("mode", ("mirror", "stage"))
def test_native_optical_deconvolution_disk_round_trip(
    processing_options, cupy_gpu, physical_volume, tmp_path, mode
):
    """Process camera stores and verify saved fluorescence and physical bead positions.

    Parameters
    ----------
    cupy_gpu
        Fixture requiring a working CUDA device.
    physical_volume
        Known specimen, bead coordinates, PSF, and photon expectations.
    tmp_path
        Temporary acquisition and processed stores.
    mode : str
        Mirror or reversed stage scan.
    """
    truth, beads, psf, expectation = physical_volume
    photons = np.random.default_rng(637).poisson(expectation)
    raw = (photons * 2 + 100).astype(np.uint16)
    if mode == "stage":
        raw = raw[::-1].copy()
    raw = raw[None, None, None]
    metadata = replace(
        simulated_acquisition_metadata(tmp_path / "physical.ome.zarr", mode, raw.shape),
        scan_axis_step_um=0.2,
    )
    write_simulated_acquisition(metadata, raw)
    psf_path = tmp_path / "independent-637nm-psf.npy"
    np.save(psf_path, psf)
    process(
        root_path=metadata.path,
        deconvolve=True,
        decon_psf_paths=[psf_path],
        decon_crop_scan=truth.shape[0],
        save_float32=True,
        **processing_options,
    )
    collection = open_position_collection(tmp_path / "physical_decon_deskewed.ome.zarr")
    actual = collection.arrays[0][0, 0].read().result()
    assert np.isfinite(actual).all() and actual.min() >= 0
    assert collection.voxel_size_um == (0.115, 0.115, 0.115)
    # The existing deskew sums two interpolated planes and divides by scan
    # spacing in pixels, giving constant-field gain 2 * pixel / scan_step.
    # Account for that known intensity convention, without fitting a scale.
    # The acquisition Jacobian is scan_step * pixel**2 * sin(angle).
    deskew_gain = 2 * 0.115 / 0.2
    output_fluorescence = actual.sum(dtype=np.float64) * 0.115**3 / deskew_gain
    known_fluorescence = truth.sum(dtype=np.float64) * 0.2 * 0.115**2 * 0.5
    assert abs(output_fluorescence / known_fluorescence - 1) < 0.1
    z, y, x = np.ogrid[: actual.shape[0], : actual.shape[1], : actual.shape[2]]
    xyz = (
        x * 0.115 - (truth.shape[2] - 1) * 0.115 / 2,
        y * 0.115
        - (
            (truth.shape[0] - 1) * 0.2
            + (truth.shape[1] - 1) * 0.115 * np.cos(np.pi / 6)
        )
        / 2,
        z * 0.115 - (truth.shape[1] - 1) * 0.115 * np.sin(np.pi / 6) / 2,
    )
    for center in beads:
        region = (
            sum((coord - c) ** 2 for coord, c in zip(xyz, center, strict=False))
            < 0.6**2
        )
        mass = actual * region
        centroid = np.asarray([np.sum(mass * coord) / mass.sum() for coord in xyz])
        assert np.linalg.norm(centroid - center) < 0.15
