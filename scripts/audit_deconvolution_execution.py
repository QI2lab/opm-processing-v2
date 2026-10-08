"""Measure deconvolution execution costs with identical reconstruction rules.

Read independent simulated image and PSF arrays from an NPZ. Compare native
and scan-subsampled RLGC using the current FFT execution against the previous
extra frequency copy, per-call garbage collection, and repeated old-prediction
FFT. Run alternating warmed trials, require identical reconstructed pixels,
and write timings, transform counts and pool reservations to JSON. Optionally
tile the input for a larger fixed-iteration timing workload; that workload is
not a new physical validation. No acquisition stores are modified.
"""

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from opm_processing.imageprocessing import rlgc as native
from opm_processing.imageprocessing import rlgc_undersampled as sparse


def main() -> None:
    """Compare execution strategies on one supplied simulation and persist results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tile", type=int, nargs=3, default=(1, 1, 1))
    args = parser.parse_args()
    with np.load(args.input) as data:
        image = np.tile(data["image"], args.tile).astype(np.float32)
        psf = data["psf"]
    cp = native.cp
    frequency_buffers = {}
    calls = [0]
    original_fft = native.fft_conv

    def previous_fft(values, transfer, shape):
        """Reproduce the former extra frequency allocation and transform copy.

        Parameters
        ----------
        values : cupy.ndarray
            Object-space input.
        transfer : cupy.ndarray
            Optical transfer function.
        shape : tuple of int
            Inverse transform dimensions.

        Returns
        -------
        cupy.ndarray
            Unclipped float32 convolution.
        """
        calls[0] += 1
        if shape not in frequency_buffers:
            frequency_buffers[shape] = cp.empty(
                (*shape[:2], shape[2] // 2 + 1), cp.complex64
            )
        workspace = frequency_buffers[shape]
        workspace[...] = cp.fft.rfftn(values)
        workspace *= transfer
        return cp.fft.irfftn(workspace, s=shape).astype(cp.float32, copy=False)

    def current_fft(values, transfer, shape):
        """Count current convolution calls without changing execution.

        Parameters
        ----------
        values : cupy.ndarray
            Object-space input.
        transfer : cupy.ndarray
            Optical transfer function.
        shape : tuple of int
            Inverse transform dimensions.

        Returns
        -------
        cupy.ndarray
            Current unclipped convolution result.
        """
        calls[0] += 1
        return original_fft(values, transfer, shape)

    # Load a separate copy for the old redundant forward transform. The source
    # and scientific rules are otherwise identical to the current solver.
    spec = importlib.util.spec_from_file_location(
        "audit_sparse_previous", sparse.__file__
    )
    previous_sparse = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = previous_sparse
    source = (
        Path(sparse.__file__)
        .read_text()
        .replace(
            "predicted, previous_prediction, split1, split2, scratch",
            "predicted, cp.maximum(forward(previous), 1e-6), split1, split2, scratch",
        )
    )
    exec(compile(source, str(sparse.__file__), "exec"), previous_sparse.__dict__)
    memory_pool = cp.cuda.MemoryPool()
    peak_live = [0]

    def measured_allocator(size):
        """Allocate a device block and track peak live bytes in an isolated pool.

        Parameters
        ----------
        size : int
            Requested allocation in bytes.

        Returns
        -------
        cupy.cuda.MemoryPointer
            Allocation owned by the measurement pool.
        """
        allocation = memory_pool.malloc(size)
        peak_live[0] = max(peak_live[0], memory_pool.used_bytes())
        return allocation

    report = {
        "shape_syx": image.shape,
        "iterations_cap": args.iterations,
        "tiled_runtime_workload": tuple(args.tile) != (1, 1, 1),
        "variants": {},
    }
    try:
        for factor in (None, 3):
            measurements = {name: [] for name in ("previous", "current")}
            outputs = {}
            counts = {}
            memory = {}
            for trial in range(args.trials + 1):
                for name in (
                    ("previous", "current")
                    if trial % 2 == 0
                    else ("current", "previous")
                ):
                    module = (
                        native
                        if factor is None
                        else (previous_sparse if name == "previous" else sparse)
                    )
                    module.fft_conv = (
                        previous_fft if name == "previous" else current_fft
                    )
                    calls[0] = 0
                    cp.cuda.Stream.null.synchronize()
                    started = time.perf_counter()
                    kwargs = dict(
                        max_iterations=args.iterations,
                        limit=0,
                        max_delta=0,
                        rng_seed=42,
                        release_memory=False,
                    )
                    if factor is None:
                        result = module.rlgc(image, psf, **kwargs)
                    else:
                        result = module.rlgc_undersampled(
                            image[::factor], psf, factor, **kwargs
                        )
                    if name == "previous" and factor is None:
                        gc.collect()
                    cp.cuda.Stream.null.synchronize()
                    elapsed = time.perf_counter() - started
                    if trial:
                        measurements[name].append(elapsed)
                    outputs[name] = result
                    counts[name] = calls[0]
                    memory[name] = cp.get_default_memory_pool().total_bytes()
                    print(factor, name, trial, elapsed, flush=True)
            np.testing.assert_array_equal(outputs["previous"], outputs["current"])
            peaks = {}
            for name in ("previous", "current"):
                frequency_buffers.clear()
                native.clear_rlgc_caches(clear_memory_pool=True)
                memory_pool = cp.cuda.MemoryPool()
                peak_live[0] = 0
                module = (
                    native
                    if factor is None
                    else (previous_sparse if name == "previous" else sparse)
                )
                module.fft_conv = previous_fft if name == "previous" else current_fft
                with cp.cuda.using_allocator(measured_allocator):
                    if factor is None:
                        module.rlgc(image, psf, **kwargs)
                    else:
                        module.rlgc_undersampled(image[::factor], psf, factor, **kwargs)
                    cp.cuda.Stream.null.synchronize()
                peaks[name] = peak_live[0]
                frequency_buffers.clear()
                native.clear_rlgc_caches(clear_memory_pool=True)
                memory_pool.free_all_blocks()
            report["variants"]["native" if factor is None else "sparse3"] = {
                name: {
                    "seconds_trials": timings,
                    "seconds_median": float(np.median(timings)),
                    "fft_convolutions": counts[name],
                    "pool_reserved_bytes": memory[name],
                    "peak_live_pool_bytes": peaks[name],
                }
                for name, timings in measurements.items()
            }
        report["pixels_identical"] = True
        report["memory_note"] = (
            "Reservations include all warmed variants. Peak live bytes are measured separately "
            "using an isolated CuPy pool; allocations made directly by the driver are excluded."
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    finally:
        native.fft_conv = sparse.fft_conv = original_fft
        frequency_buffers.clear()
        native.clear_rlgc_caches(clear_memory_pool=True)


if __name__ == "__main__":
    main()
