"""Deconvolve a saved camera-integrated, noisy sphere acquisition.

Read simulation.json, raw_electrons.tif, microscope comparison volumes, and the
fine Cartesian object/PSF from a saved simulation. Reconstruct the oblique and
Cartesian acquisitions with the production RLGC solvers, optionally inserting
scan planes with --scan-upsample, then deskew the oblique reconstruction.
Write reconstructed TIFFs, the sampled deconvolution PSF, and simulation.json
with truth-comparison metrics into decon_native or decon_upsample<N> below the
input directory. Acquisition is not rerun and no additional noise is added.

Run from the repository root with
``uv run python -m scripts.deconvolve_opm_simulation ACQUISITION``.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
from tifffile import imread, imwrite

from .opm_simulation import (
    centered_axis,
    comparison_metrics,
    pixel_average,
)
from opm_processing.imageprocessing.opmtools import orthogonal_deskew
from .psf_sampling_experiment import sample_skewed


def full_volume_error(image, truth):
    """Include both missing fluorescence and false fluorescence outside the tubes.

    Parameters
    ----------
    image
        Nonnegative image samples on the operator input grid.
    truth
        Known simulated fluorescence volume used as the reference.

    Returns
    -------
    float
        Full-volume relative L2 error against the known simulated fluorescence.
    """
    return float(np.linalg.norm(image - truth) / np.linalg.norm(truth))


def deconvolve_saved_simulation(directory, scan_upsample=None):
    """Keep exported data fixed and compare native or upsampled RLGC with truth.

    Require completed optical convolution, camera integration and detection
    noise. Use the saved detected electrons, never scale noiseless density into
    pseudo-counts. No new data or noise realization is generated here.

    Parameters
    ----------
    directory
        Saved simulation directory containing TIFFs and simulation.json.
    scan_upsample
        Integer scan-plane insertion factor, or None for native sampling.

    Returns
    -------
    dict
        Reconstruction truth-comparison metrics saved beside the output TIFFs.
    """
    directory = Path(directory)
    meta = json.loads((directory / "simulation.json").read_text(encoding="utf-8"))
    if (
        meta["acquisition"]["peak_electrons"] is None
        or meta["acquisition"]["camera_samples"] < 2
    ):
        raise ValueError(
            "GC requires the fully convolved, camera-integrated, noisy acquisition export"
        )
    if scan_upsample is not None and (
        not isinstance(scan_upsample, int) or scan_upsample < 1
    ):
        raise ValueError("scan_upsample must be a positive integer")
    from opm_processing.imageprocessing.rlgc import rlgc

    p, acquisition = meta["parameters"], meta["acquisition"]
    pixel = p["pixel_size_um"]
    theta = np.deg2rad(p["angle_deg"])
    step = acquisition["scan_step_um"] / (scan_upsample or 1)
    raw_path = directory / "raw_electrons.tif"
    counts = np.maximum(imread(raw_path) - acquisition["background_electrons"], 0)
    raw = imread(directory / "raw_skewed.tif")
    truth = imread(directory / "truth_sampled.ome.tif")
    normal = imread(directory / "normal.ome.tif")
    original_deskewed = imread(directory / "deskewed.ome.tif")
    fine_psf = imread(directory.parent / "psf_cartesian_fine.tif")
    counts_per_density = acquisition.get("counts_per_density")
    if counts_per_density is None:
        # Older exports did not persist their calibration gain. Recalculate
        # its exact definition from the saved fine truth and PSF, without
        # changing the acquired data or fitting gain to a reconstruction.
        fine_truth = imread(directory.parent / "ground_truth_fine.tif")
        peak_density = float(fftconvolve(fine_truth, fine_psf, mode="same").max())
        counts_per_density = acquisition["peak_electrons"] / peak_density
        del fine_truth
    np.testing.assert_allclose(counts / counts_per_density, raw, rtol=1e-6, atol=1e-7)
    psf_axes = tuple(
        a["origin_um"] + np.arange(a["size"]) * a["step_um"]
        for a in meta["cartesian_psf_axes_zyx"]
    )
    hz, hy, hx = (max(abs(a[0]), abs(a[-1])) for a in psf_axes)
    skewed_shape = tuple(
        len(centered_axis(e, d))
        for e, d in zip(
            (hy + hz / np.tan(theta), hz / np.sin(theta), hx),
            (step, pixel, pixel),
        )
    )
    # Same fine Cartesian optical field as the forward simulation. Only the
    # requested reconstruction-grid sampling changes; the measured raw data do not.
    skewed_psf = np.zeros(skewed_shape, dtype=np.float32)
    samples = acquisition["camera_samples"]
    offsets = ((np.arange(samples) + 0.5) / samples - 0.5) * pixel
    for dy in offsets:
        for dx in offsets:
            skewed_psf += sample_skewed(
                fine_psf,
                psf_axes,
                skewed_shape,
                pixel_size_um=pixel,
                scan_step_um=step,
                angle_deg=p["angle_deg"],
                center_xyz_um=(dx, dy * np.cos(theta), dy * np.sin(theta)),
            )
    skewed_psf /= skewed_psf.sum()
    print(
        f"Deconvolving fixed raw volume {raw.shape}; reconstruction scan step {step:g} um",
        flush=True,
    )
    if scan_upsample is None:
        reconstructed = rlgc(counts, skewed_psf)
    else:
        from opm_processing.imageprocessing.rlgc_undersampled import rlgc_undersampled

        reconstructed = rlgc_undersampled(
            counts, skewed_psf, scan_upsample_factor=scan_upsample
        )
    if not np.isfinite(reconstructed).all():
        raise FloatingPointError(
            "GC returned non-finite values; no deconvolution output was exported"
        )
    reconstructed /= counts_per_density
    deskewed = orthogonal_deskew(
        reconstructed,
        theta=p["angle_deg"],
        distance=step,
        pixel_size=pixel,
        downsample_factor=1,
        divisible_by=1,
    ) * (step / (2 * pixel))

    # Endpoint-preserving scan upsampling leaves the physical volume origin
    # unchanged. Select the exact pre-existing comparison grid, without shifts.
    raw_axes = meta["raw_axes_syx"]
    first_scan, first_row, first_col = (a["origin_um"] for a in raw_axes)
    origin = (
        first_row * np.sin(theta),
        first_scan + first_row * np.cos(theta),
        first_col,
    )
    offsets = [
        (a["origin_um"] - o) / pixel for a, o in zip(meta["output_axes_zyx"], origin)
    ]
    if not np.allclose(offsets, np.round(offsets), atol=1e-6):
        raise ValueError(
            "Reconstructed grid does not coincide with the recorded physical comparison grid"
        )
    crop = tuple(
        slice(int(round(o)), int(round(o)) + n) for o, n in zip(offsets, truth.shape)
    )
    deskewed = deskewed[crop].copy()
    if deskewed.shape != truth.shape:
        raise ValueError("Reconstruction does not cover the full comparison grid")

    normal_axes = tuple(centered_axis(e, pixel) for e in (hz, hy, hx))
    normal_psf = pixel_average(fine_psf, psf_axes, normal_axes, pixel, samples)
    normal_psf /= normal_psf.sum()
    print("Deconvolving fixed Cartesian microscope reference", flush=True)
    normal_reconstructed = (
        rlgc(normal * counts_per_density, normal_psf) / counts_per_density
    )
    if not np.isfinite(normal_reconstructed).all():
        raise FloatingPointError(
            "Cartesian GC returned non-finite values; no output was exported"
        )
    metrics = dict(
        opm_before_vs_truth=full_volume_error(original_deskewed, truth),
        opm_after_vs_truth=full_volume_error(deskewed, truth),
        normal_before_vs_truth=full_volume_error(normal, truth),
        normal_after_vs_truth=full_volume_error(normal_reconstructed, truth),
        deskew_vs_normal=comparison_metrics(deskewed, normal_reconstructed),
    )
    print(json.dumps(metrics), flush=True)
    output = directory / (
        "decon_native" if scan_upsample is None else f"decon_upsample{scan_upsample}"
    )
    output.mkdir(exist_ok=True)
    image_metadata = {"axes": "ZYX"}
    for axis in "XYZ":
        image_metadata[f"PhysicalSize{axis}"] = pixel
        image_metadata[f"PhysicalSize{axis}Unit"] = "µm"
    for name, image in (
        ("truth_sampled", truth),
        ("normal", normal_reconstructed),
        ("deskewed", deskewed),
    ):
        imwrite(
            output / f"{name}.ome.tif",
            image,
            metadata=image_metadata,
            compression="zlib",
        )
    imwrite(output / "deconvolved_skewed.tif", reconstructed, compression="zlib")
    imwrite(output / "deconvolution_psf_skewed.tif", skewed_psf, compression="zlib")
    meta.update(
        metrics=metrics,
        comparison_stage="deconvolution",
        deconvolution=dict(
            raw_source=str(raw_path.resolve()),
            raw_file_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            scan_upsample=scan_upsample,
            reconstruction_step_um=step,
            reconstruction_shape_syx=list(reconstructed.shape),
            counts_per_density=counts_per_density,
            added_noise=False,
            solver="rlgc" if scan_upsample is None else "rlgc_undersampled",
        ),
    )
    (output / "simulation.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return metrics


def main():
    """Run reconstruction on existing exported data without rerunning acquisition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--scan-upsample", type=int)
    args = parser.parse_args()
    deconvolve_saved_simulation(args.directory, args.scan_upsample)


if __name__ == "__main__":
    main()
