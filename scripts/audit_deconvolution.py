"""Compare native RLGC, lookup-table count splitting, and finite-PSF edge tapering.

Read an NPZ containing image, psf, truth, and sampling_um arrays from an
independent optical simulation. Run warmed, alternating reconstruction trials
and write input/reconstruction OME-Zarr stores, candidate PSFs, and a JSON report
of timings, FFT counts, memory, absolute reconstruction error, flux, and measured
prediction error. An optional rlgc.py snapshot with the same OPM geometry
padding supplies a measured baseline. Use --require-pixel-identity only for
changes intended to preserve arithmetic and random draws.
The lookup sampler and taper are experiments; production defaults are unchanged.
Run from the repository root with python -m scripts.audit_deconvolution.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.special import bdtr

from opm_processing.dataio.position_collection import (
    create_position_collection,
    open_position_collection,
)
from opm_processing.imageprocessing import rlgc as solver
from opm_processing.imageprocessing.rlgc import _split_observed_counts

cp = solver.cp

_binomial_lookup = cp.ElementwiseKernel(
    "float32 observed, float64 uniform, float32 residual_uniform, raw float64 cdf, int32 stride",
    "float32 split",
    """
    const long long n = (long long)observed;
    if (n >= stride) {
        split = 0;
    } else {
        int lo = 0, hi = (int)n;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (cdf[n * stride + mid] > uniform) hi = mid;
            else lo = mid + 1;
        }
        split = (float)lo + (residual_uniform < 0.5f ? observed - (float)n : 0.0f);
    }
    """,
    "audit_binomial_lookup",
)


def binomial_split_cdf(max_count: int = 256) -> cp.ndarray:
    """Build a bounded float64 binomial CDF on the current CUDA device.

    Parameters
    ----------
    max_count : int
        Largest integer photon count represented, defaulting to 256.

    Returns
    -------
    cupy.ndarray
        Square table whose row n contains P(Binomial(n, 0.5) <= k).
        Entries beyond n equal one. The table occupies 8*(max_count+1)**2 bytes.
    """
    counts = np.arange(max_count + 1)
    probabilities = bdtr(
        np.minimum(counts[None], counts[:, None]), counts[:, None], 0.5
    )
    return cp.asarray(probabilities, dtype=cp.float64)


def split_counts_lookup(
    observed: cp.ndarray,
    rng: cp.random.Generator,
    cdf: cp.ndarray,
    *,
    assign_fractional_remainder: bool = True,
) -> cp.ndarray:
    """Split counts with fresh inverse-CDF draws and optional fractional assignment.

    Parameters
    ----------
    observed : cupy.ndarray
        Nonnegative float32 calibrated photon counts.
    rng : cupy.random.Generator
        Generator for independent float64 quantiles and fractional assignments.
    cdf : cupy.ndarray
        Float64 table from binomial_split_cdf on the same CUDA device.
    assign_fractional_remainder : bool, default=True
        Use weighted residual assignment for the undersampled model. False
        leaves every remainder in the complementary half, as in the reference.

    Returns
    -------
    cupy.ndarray
        One float32 half; subtracting it from observed gives the complementary
        half. Counts exceeding the table use the original binomial sampler.
    """
    split = _binomial_lookup(
        observed,
        rng.random(observed.shape, dtype=cp.float64),
        rng.random(observed.shape, dtype=cp.float32)
        if assign_fractional_remainder
        else cp.asarray(1, dtype=cp.float32),
        cdf,
        cdf.shape[0],
    )
    outside = observed >= cdf.shape[0]
    if bool(cp.any(outside)):
        split[outside] = _split_observed_counts(
            observed[outside],
            rng,
            assign_fractional_remainder=assign_fractional_remainder,
        )
    return split


def taper_psf(
    psf: np.ndarray,
    sampling_um: tuple[float, float, float],
    edge_widths_um: tuple[float, float, float],
) -> tuple[np.ndarray, float]:
    """Apply raised-cosine edge windows to the centered acquisition-grid PSF.

    Parameters
    ----------
    psf : numpy.ndarray
        Nonnegative centered SYX PSF, before FFT padding or shifting.
    sampling_um : tuple of float
        Scan, camera-row, and camera-column spacings in micrometers.
    edge_widths_um : tuple of float
        Taper widths from each face along SYX, in micrometers. Zero disables
        tapering along that axis; singleton axes retain unit weight.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Unit-sum float32 PSF and fraction of input energy removed by the window.
        The source is unchanged. Widths describe acquisition axes, not lab ZYX.
    """
    tapered = np.asarray(psf, dtype=np.float64).copy()
    original_energy = tapered.sum()
    for axis, (size, pitch, width) in enumerate(
        zip(psf.shape, sampling_um, edge_widths_um, strict=True)
    ):
        if size == 1 or width == 0:
            continue
        distance = np.minimum(np.arange(size), np.arange(size)[::-1]) * pitch
        profile = 0.5 * (1 - np.cos(np.pi * np.minimum(distance / width, 1)))
        shape = [1] * psf.ndim
        shape[axis] = size
        tapered *= profile.reshape(shape)
    retained_energy = tapered.sum()
    return (tapered / retained_energy).astype(np.float32), float(
        1 - retained_energy / original_energy
    )


def main() -> None:
    """Persist reconstruction comparisons from independently simulated input data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="NPZ with image, psf, truth, sampling_um"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--baseline", type=Path, help="rlgc.py snapshot using the same OPM padding"
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--lookup-count", type=int, default=256)
    parser.add_argument("--taper-um", type=float, nargs=3, default=(0.4, 0.46, 0.23))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver-logging", action="store_true")
    parser.add_argument("--require-pixel-identity", action="store_true")
    args = parser.parse_args()
    with np.load(args.input) as data:
        image, psf, truth = (data[name] for name in ("image", "psf", "truth"))
        sampling = tuple(float(value) for value in data["sampling_um"])
    image = image.astype(np.float32, copy=False)
    args.output.mkdir(parents=True, exist_ok=True)
    setup_start = time.perf_counter()
    lookup = binomial_split_cdf(args.lookup_count)
    cp.cuda.Stream.null.synchronize()
    lookup_setup_seconds = time.perf_counter() - setup_start
    tapered, removed_energy = taper_psf(psf, sampling, tuple(args.taper_um))
    variants = [
        ("current", solver, psf, False),
        ("lookup", solver, psf, True),
        ("taper", solver, tapered, False),
    ]
    if args.baseline is not None:
        spec = importlib.util.spec_from_file_location(
            "audit_rlgc_baseline", args.baseline
        )
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        variants.insert(0, ("baseline", baseline, psf, False))
    logger = logging.getLogger("deconvolution-audit") if args.solver_logging else None
    if logger is not None:
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())
    inputs = create_position_collection(
        args.output / "input.ome.zarr",
        (1, 1, 1, *image.shape),
        sampling,
        chunks=(1, 1, 8, 64, 64),
        dtype=np.float32,
    )
    inputs.arrays[0][0, 0].write(image.astype(np.float32)).result()
    original_split = solver._split_observed_counts
    fft_functions = {module: module.fft_conv for _, module, *_ in variants}
    counts = [0]

    def counted_fft(
        image: cp.ndarray, transfer: cp.ndarray, shape: tuple[int, ...]
    ) -> cp.ndarray:
        """Count optical convolutions during one reconstruction.

        Parameters
        ----------
        image
            Device input volume.
        transfer
            Forward or adjoint OTF.
        shape
            FFT transform dimensions.

        Returns
        -------
        cupy.ndarray
            Original solver convolution result.
        """
        counts[0] += 1
        return fft_function(image, transfer, shape)

    measurements = {name: [] for name, *_ in variants}
    metrics = {}
    try:
        for trial in range(args.trials + 1):
            for name, module, kernel, use_lookup in (
                variants if trial % 2 == 0 else variants[::-1]
            ):
                solver._split_observed_counts = (
                    partial(split_counts_lookup, cdf=lookup)
                    if use_lookup
                    else original_split
                )
                fft_function = fft_functions[module]
                module.fft_conv = counted_fft
                counts[0] = 0
                cp.cuda.Stream.null.synchronize()
                started = time.perf_counter()
                result = module.rlgc(
                    image,
                    kernel,
                    rng_seed=args.seed,
                    limit=0.01,
                    max_delta=0.001,
                    release_memory=False,
                    logger=logger,
                )
                cp.cuda.Stream.null.synchronize()
                elapsed = time.perf_counter() - started
                if trial:
                    measurements[name].append(elapsed)
                prediction = signal.fftconvolve(
                    result.astype(np.float64), psf, mode="same"
                )
                metrics[name] = {
                    "seconds_median": float(np.median(measurements[name]))
                    if trial
                    else None,
                    "seconds_trials": measurements[name],
                    "fft_convolutions": counts[0],
                    "cumulative_pool_reserved_bytes": cp.get_default_memory_pool().total_bytes(),
                    "truth_nrmse": float(
                        np.linalg.norm(result - truth) / np.linalg.norm(truth)
                    ),
                    "flux_ratio": float(result.sum() / truth.sum()),
                    "prediction_nrmse": float(
                        np.linalg.norm(prediction - image) / np.linalg.norm(image)
                    ),
                }
                if trial == args.trials:
                    output = create_position_collection(
                        args.output / f"{name}.ome.zarr",
                        (1, 1, 1, *result.shape),
                        sampling,
                        chunks=(1, 1, 8, 64, 64),
                        dtype=np.float32,
                    )
                    output.arrays[0][0, 0].write(result).result()
                    np.save(args.output / f"{name}-psf.npy", kernel)
                print(name, "trial", trial, "seconds", elapsed, flush=True)
    finally:
        solver._split_observed_counts = original_split
        for module, function in fft_functions.items():
            module.fft_conv = function
            module.clear_rlgc_caches(clear_memory_pool=True)
    report = {
        "shape_syx": list(image.shape),
        "sampling_um": sampling,
        "lookup_max_count": args.lookup_count,
        "lookup_table_bytes": lookup.nbytes,
        "lookup_setup_seconds": lookup_setup_seconds,
        "taper_widths_um": args.taper_um,
        "taper_removed_energy": removed_energy,
        "rng_seed": args.seed,
        "limit": 0.01,
        "max_delta": 0.001,
        "solver_logging": args.solver_logging,
        "memory_note": "Pool reservation accumulates across variants; this is not per-variant peak memory.",
        "variants": metrics,
    }
    if args.baseline is not None:
        previous = open_position_collection(args.output / "baseline.ome.zarr")
        current = open_position_collection(args.output / "current.ome.zarr")
        previous_pixels = previous.arrays[0][0, 0].read().result()
        current_pixels = current.arrays[0][0, 0].read().result()
        report["baseline_pixels_identical"] = bool(
            np.array_equal(previous_pixels, current_pixels)
        )
        if args.require_pixel_identity:
            np.testing.assert_array_equal(current_pixels, previous_pixels)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
