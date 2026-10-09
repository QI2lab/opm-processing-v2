"""Experimental RL/gradient-consensus reconstruction of missing scan planes.

This separate module is opt-in through ``process --deconvolve
--decon-scan-upsample N``. Arrays are in (scan, camera_y, camera_x), not
laboratory ZYX. Supply the existing PSF
generator with ``acquired_scan_step_um / scan_upsample_factor``; its output is
the PSF required here. No PSF resampling or new PSF model is introduced.

The forward model is H = S C: convolve on the reconstruction grid, then select
acquired planes. Its adjoint is C^T S^T: insert zeros, then convolve with the
adjoint PSF. Missing planes are unobserved, not measured zeros. The RL update
includes H^T(1) normalization, which is essential for this sampling operator.

Unlike the production solver's reflected input padding, this experiment
assumes zero fluorescence outside the reconstruction volume. This gives an
explicit linear forward model and its exact transpose, including at edges.
"""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from opm_processing.imageprocessing.rlgc import (
    _linear_fft_pad_width,
    _observed_region_slices,
    _split_observed_counts,
    clear_rlgc_caches,
    cp,
    fft_conv,
    filter_update,
    pad_psf,
    split_kl_changes,
)

if TYPE_CHECKING:
    import logging


def _convolve_core(
    image: cp.ndarray,
    otf: cp.ndarray,
    padded_shape: tuple[int, int, int],
    core: tuple[slice, slice, slice],
) -> cp.ndarray:
    """Convolve a zero-extended object and crop to its original field of view.

    Parameters
    ----------
    image
        Nonnegative image samples on the operator input grid.
    otf
        Frequency-domain optical transfer function on the padded convolution grid.
    padded_shape
        FFT grid dimensions including zero padding.
    core
        Slices selecting the original image support from the FFT grid.

    Returns
    -------
    cupy.ndarray
        Convolved image on the original support after zero-padded FFT convolution.
    """
    padded = cp.zeros(padded_shape, dtype=cp.float32)
    padded[core] = image
    return fft_conv(padded, otf, padded_shape)[core].copy()


def _forward(
    image: cp.ndarray,
    otf: cp.ndarray,
    padded_shape: tuple[int, int, int],
    core: tuple[slice, slice, slice],
    factor: int,
) -> cp.ndarray:
    """Blur on the fine grid before selecting measured scan planes.

    Parameters
    ----------
    image
        Nonnegative image samples on the operator input grid.
    otf
        Frequency-domain optical transfer function on the padded convolution grid.
    padded_shape
        FFT grid dimensions including zero padding.
    core
        Slices selecting the original image support from the FFT grid.
    factor
        Integer spacing between measured scan planes on the fine reconstruction grid.

    Returns
    -------
    cupy.ndarray
        Predicted camera planes at the measured fine-grid scan indices.
    """
    return _convolve_core(image, otf, padded_shape, core)[::factor].copy()


def _adjoint(
    image: cp.ndarray,
    otf_adjoint: cp.ndarray,
    padded_shape: tuple[int, int, int],
    core: tuple[slice, slice, slice],
    fine_shape: tuple[int, int, int],
    factor: int,
) -> cp.ndarray:
    """Scatter measurements into a zero-filled fine grid and apply C transpose.

    Parameters
    ----------
    image
        Nonnegative image samples on the operator input grid.
    otf_adjoint
        Conjugate optical transfer function for adjoint convolution.
    padded_shape
        FFT grid dimensions including zero padding.
    core
        Slices selecting the original image support from the FFT grid.
    fine_shape
        Endpoint-preserving fine reconstruction dimensions in scan, Y, X order.
    factor
        Integer spacing between measured scan planes on the fine reconstruction grid.

    Returns
    -------
    cupy.ndarray
        Adjoint optical backprojection on the endpoint-preserving fine scan grid.
    """
    scattered = cp.zeros(fine_shape, dtype=cp.float32)
    scattered[::factor] = image
    return _convolve_core(scattered, otf_adjoint, padded_shape, core)


def rlgc_undersampled(
    image: np.ndarray,
    psf: np.ndarray,
    scan_upsample_factor: int = 1,
    *,
    gradient_consensus: bool = True,
    max_iterations: int = 100,
    gpu_id: int = 0,
    safe_mode: bool = True,
    limit: float = 0.001,
    max_delta: float = 0.001,
    rng_seed: int | None = 42,
    release_memory: bool = True,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Reconstruct a fine scan grid using the existing RLGC numerical helpers.

    Parameters
    ----------
    image : numpy.ndarray
        Nonnegative measured photon counts, shaped (scan, camera_y, camera_x).
        Measurements are taken at fine-grid indices 0, k, 2k, ... .
    psf : numpy.ndarray
        Nonnegative, centered, odd-sized 3D PSF sampled on the desired fine
        grid. It is normalized to unit sum. The caller controls physical
        sampling using the existing PSF generator.
    scan_upsample_factor : int, default=1
        Integer k relating acquired and reconstructed scan steps. Output has
        ``(image.shape[0] - 1) * k + 1`` planes, preserving endpoint positions.
    gradient_consensus : bool, default=True
        Use the existing count splitting, consensus gate and split-KLD
        rollback. False runs ordinary RL for controlled comparisons.
    max_iterations : int, default=100
        Maximum multiplicative updates after normalized-backprojection
        initialization. Also bounds the experimental GC run.
    gpu_id : int, default=0
        CUDA device.
    safe_mode : bool, default=True
        In GC mode, compare normalized observation-to-prediction KLD against
        the same fresh split for both estimates. Roll back if either half
        worsens; otherwise require both to worsen. Only acquired planes enter
        this statistic; the previous prediction is reused without another FFT.
    limit : float, default=0.001
        In GC mode, stop below this fraction of fine-grid voxels updated.
    max_delta : float, default=0.001
        Stop when the maximum change relative to the reconstruction peak is
        below this value. Use zero for fixed-iteration ordinary RL tests.
    rng_seed : int or None, default=42
        Seed for GC count splitting.
    release_memory : bool, default=True
        Clear shared FFT caches and free unused GPU pool allocations on exit.
    logger : logging.Logger or None, default=None
        Optional iteration diagnostics.

    Returns
    -------
    numpy.ndarray
        Float32 fluorescence estimate at the fine scan step. No deskewing,
        interpolation of measurements, chunking, or metadata changes occur.
    """
    for name, value in (
        ("scan_upsample_factor", scan_upsample_factor),
        ("max_iterations", max_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    image = np.asarray(image, dtype=np.float32)
    psf = np.asarray(psf, dtype=np.float32)
    for name, array in (("image", image), ("psf", psf)):
        if array.ndim != 3 or any(size == 0 for size in array.shape):
            raise ValueError(f"{name} must be a nonempty 3D array")
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(f"{name} must contain finite nonnegative values")
    if not np.sum(psf, dtype=np.float64) > 0 or any(s % 2 == 0 for s in psf.shape):
        raise ValueError("psf must have positive sum and odd dimensions")
    if not np.isfinite(limit) or not 0 <= limit <= 1:
        raise ValueError("limit must be between zero and one")
    if not np.isfinite(max_delta) or max_delta < 0:
        raise ValueError("max_delta must be finite and nonnegative")

    factor = int(scan_upsample_factor)
    fine_shape = ((image.shape[0] - 1) * factor + 1, *image.shape[1:])
    with cp.cuda.Device(gpu_id):
        try:
            pads = _linear_fft_pad_width(fine_shape, psf.shape)
            padded_shape = tuple(
                n + sum(p) for n, p in zip(fine_shape, pads, strict=False)
            )
            core = _observed_region_slices(padded_shape, pads)
            otf = cp.fft.rfftn(pad_psf(cp.asarray(psf), padded_shape))
            otf_adjoint = cp.conjugate(otf)
            observed = cp.asarray(image)

            def forward(estimate: cp.ndarray) -> cp.ndarray:
                """Blur the fine estimate and select only acquired scan planes.

                Parameters
                ----------
                estimate
                    Current fluorescence reconstruction on the fine scan grid.

                Returns
                -------
                array or scalar
                    Predicted measured camera planes from the fine estimate.
                """
                return _forward(estimate, otf, padded_shape, core, factor)

            def adjoint(values: cp.ndarray) -> cp.ndarray:
                """Scatter acquired-plane values and apply the adjoint optical convolution.

                Parameters
                ----------
                values
                    Values on acquired camera planes to backproject.

                Returns
                -------
                array or scalar
                    Backprojected measured-plane values on the fine reconstruction grid.
                """
                return _adjoint(
                    values, otf_adjoint, padded_shape, core, fine_shape, factor
                )

            sensitivity = cp.maximum(adjoint(cp.ones_like(observed)), 1e-6)
            recon = cp.maximum(adjoint(observed) / sensitivity, 0)
            # A positive seed avoids locking unmeasured voxels at zero.
            recon = cp.maximum(recon, cp.max(recon) * cp.float32(1e-7))
            previous = None
            previous_prediction = None
            rng = cp.random.default_rng(rng_seed) if gradient_consensus else None
            scratch = cp.empty_like(observed)
            for iteration in range(max_iterations):
                predicted = cp.maximum(forward(recon), cp.float32(1e-6))
                if gradient_consensus:
                    split1 = _split_observed_counts(observed, rng)
                    split2 = observed - split1
                    if previous is not None:
                        changes = split_kl_changes(
                            predicted, previous_prediction, split1, split2, scratch
                        )
                        worse = [change > 0 for change in changes]
                        if any(worse) if safe_mode else all(worse):
                            recon = previous
                            break
                    ratio1 = adjoint(2 * split1 / predicted) / sensitivity
                    ratio2 = adjoint(2 * split2 / predicted) / sensitivity
                    consensus = _convolve_core(
                        (ratio1 - 1) * (ratio2 - 1),
                        otf * otf_adjoint,
                        padded_shape,
                        core,
                    )
                    ratio = cp.maximum((ratio1 + ratio2) * 0.5, 0)
                    updated = cp.empty_like(recon)
                    filter_update(recon, ratio, consensus, updated)
                    updated_fraction = float(cp.mean(consensus >= 0))
                else:
                    ratio = cp.maximum(adjoint(observed / predicted) / sensitivity, 0)
                    updated = recon * ratio
                    updated_fraction = 1.0
                delta = float(
                    cp.max(cp.abs(updated - recon)) / cp.maximum(cp.max(updated), 1e-6)
                )
                previous_prediction = predicted
                previous, recon = recon, updated
                if logger is not None:
                    logger.info(
                        "undersampled_rl iteration=%d factor=%d gc=%s delta=%.6g updated_fraction=%.6g",
                        iteration + 1,
                        factor,
                        gradient_consensus,
                        delta,
                        updated_fraction,
                    )
                if delta < max_delta or (
                    gradient_consensus and updated_fraction < limit
                ):
                    break
            return cp.asnumpy(recon)
        finally:
            if release_memory:
                clear_rlgc_caches(clear_memory_pool=True)
