"""Experimental physical OPM simulation; production processing is unchanged.

Run with ``python -m opm_processing.imageprocessing.opm_simulation --output DIR``.
Coordinates are micrometers. Images before the camera represent fluorescence
density. Both microscopes use the same finite-support vectorial optical model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from tifffile import imwrite

from opm_processing.imageprocessing.opmtools import orthogonal_deskew
from opm_processing.imageprocessing.psf_sampling_experiment import (
    cartesian_psf,
    sample_skewed,
)


def centered_axis(half_extent, spacing):
    """Cover a physical half extent with an odd grid centered on zero."""
    radius = int(np.ceil(half_extent / spacing))
    return np.arange(-radius, radius + 1) * spacing


def meridian_sphere(axes_zyx, diameter_um=10.0, tube_diameter_um=1.0, meridians=12):
    """Construct uniform fluorescent tubes around meridian great-circle arcs.

    Diameter refers to the sphere traced by the tube centerlines; the outer
    diameter includes one tube diameter. An even number of pole-to-pole arcs
    forms meridians/2 full circles. Overlapping tubes form a union, not a sum.
    """
    if not 0 < tube_diameter_um < diameter_um:
        raise ValueError("Require 0 < tube diameter < sphere diameter")
    if not isinstance(meridians, int) or meridians < 2 or meridians % 2:
        raise ValueError("meridians must be a positive even integer of at least 2")
    z, y, x = np.meshgrid(
        *[a.astype(np.float32) for a in axes_zyx], indexing="ij", sparse=True
    )
    fluorescent = np.zeros(tuple(len(a) for a in axes_zyx), dtype=bool)
    for phi in np.arange(meridians // 2) * 2 * np.pi / meridians:
        perpendicular = -x * np.sin(phi) + y * np.cos(phi)
        in_plane = x * np.cos(phi) + y * np.sin(phi)
        distance_squared = (
            perpendicular**2 + (np.sqrt(in_plane**2 + z**2) - diameter_um / 2) ** 2
        )
        fluorescent |= distance_squared <= (tube_diameter_um / 2) ** 2
    return fluorescent.astype(np.float32)


def resample_grid(field, source_axes, target_axes):
    """Sample a uniform rectilinear field without allocating full XYZ meshes."""
    coords = [
        (target - source[0]) / (source[1] - source[0])
        for source, target in zip(source_axes, target_axes)
    ]
    result = np.empty(tuple(len(a) for a in target_axes), dtype=np.float32)
    yy, xx = np.meshgrid(coords[1], coords[2], indexing="ij")
    for i, zz in enumerate(coords[0]):
        result[i] = map_coordinates(
            field,
            np.stack((np.full_like(yy, zz), yy, xx)),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    return result


def pixel_average(field, source_axes, target_axes, pitch_um, samples=1):
    """Average uniformly spaced subpixel centers over a camera pixel's area.

    The first axis is plane index; only the two camera axes are integrated.
    samples=1 gives point sampling for the initial geometry validation. Photon
    conversion follows averaging, so fine-grid subdivisions do not add photons.
    """
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    result = np.zeros(tuple(len(a) for a in target_axes), dtype=np.float32)
    offsets = ((np.arange(samples) + 0.5) / samples - 0.5) * pitch_um
    for dy in offsets:
        for dx in offsets:
            result += resample_grid(
                field,
                source_axes,
                (target_axes[0], target_axes[1] + dy, target_axes[2] + dx),
            )
    return result / samples**2


def camera_noise(expected_electrons, read_noise_e=0.0, seed=42):
    """Apply Poisson detection then independent Gaussian read noise, in electrons."""
    if read_noise_e < 0 or not np.isfinite(read_noise_e):
        raise ValueError("read_noise_e must be nonnegative and finite")
    rng = np.random.default_rng(seed)
    return (
        rng.poisson(expected_electrons)
        + rng.normal(0.0, read_noise_e, np.shape(expected_electrons))
    ).astype(np.float32)


def prepare_sphere(
    *,
    diameter_um=10.0,
    tube_diameter_um=1.0,
    meridians=12,
    fine_spacing_um=0.05,
    pixel_size_um=0.115,
    fine_scan_step_um=0.1,
    angle_deg=30.0,
    numerical_aperture=1.35,
    wavelength_um=0.637,
):
    """Construct independent Cartesian and fine-skewed optical convolutions."""
    if not 0 < angle_deg < 90:
        raise ValueError("angle_deg must lie between 0 and 90")
    if min(fine_spacing_um, pixel_size_um, fine_scan_step_um) <= 0:
        raise ValueError("Grid spacings must be positive")
    theta = np.deg2rad(angle_deg)
    if fine_spacing_um >= pixel_size_um * min(1.0, np.sin(theta)):
        raise ValueError(
            "Cartesian fine spacing must be below the camera's projected Z sampling"
        )
    params = dict(
        diameter_um=diameter_um,
        tube_diameter_um=tube_diameter_um,
        meridians=meridians,
        fine_spacing_um=fine_spacing_um,
        pixel_size_um=pixel_size_um,
        fine_scan_step_um=fine_scan_step_um,
        angle_deg=angle_deg,
        numerical_aperture=numerical_aperture,
        wavelength_um=wavelength_um,
    )
    print("Generating fine Cartesian object and vectorial PSF", flush=True)
    psf_axes, psf = cartesian_psf(
        fine_spacing_um,
        (2.4, 1.2, 1.2),
        wavelength_um=wavelength_um,
        numerical_aperture=numerical_aperture,
    )
    psf = psf.astype(np.float32)
    half_extent = diameter_um / 2 + tube_diameter_um / 2 + 2.4
    axes = tuple(centered_axis(half_extent, fine_spacing_um) for _ in range(3))
    truth = meridian_sphere(axes, diameter_um, tube_diameter_um, meridians)
    print(f"Cartesian convolution: {truth.shape}, {fine_spacing_um:g} um", flush=True)
    cartesian_blurred = np.maximum(fftconvolve(truth, psf, mode="same"), 0)

    # The inverse acquisition map is r=Z/sin(theta), s=Y-Z*cot(theta).
    # Cover the entire Cartesian box, including optical halo, without cropping
    # corners that might be mistaken for a geometric failure.
    h = axes[0][-1]
    instrument_extents = (h + h / np.tan(theta), h / np.sin(theta), h)
    fine_camera_step = pixel_size_um / 2
    fine_axes = tuple(
        centered_axis(e, d)
        for e, d in zip(
            instrument_extents,
            (fine_scan_step_um, fine_camera_step, fine_camera_step),
        )
    )
    fine_shape = tuple(len(a) for a in fine_axes)
    print(f"Sampling object into fine instrument grid: {fine_shape}", flush=True)
    skewed_truth = sample_skewed(
        truth,
        axes,
        fine_shape,
        pixel_size_um=fine_camera_step,
        scan_step_um=fine_scan_step_um,
        angle_deg=angle_deg,
    )
    pz, py, px = (a[-1] for a in psf_axes)
    psf_extents = (py + pz / np.tan(theta), pz / np.sin(theta), px)
    skewed_psf_shape = tuple(
        len(centered_axis(e, d))
        for e, d in zip(
            psf_extents,
            (fine_scan_step_um, fine_camera_step, fine_camera_step),
        )
    )
    skewed_psf = sample_skewed(
        psf,
        psf_axes,
        skewed_psf_shape,
        pixel_size_um=fine_camera_step,
        scan_step_um=fine_scan_step_um,
        angle_deg=angle_deg,
    )
    skewed_psf /= skewed_psf.sum()
    print(f"Fine instrument convolution with skewed PSF {skewed_psf.shape}", flush=True)
    skewed_blurred = np.maximum(fftconvolve(skewed_truth, skewed_psf, mode="same"), 0)
    return dict(
        params=params,
        axes=axes,
        truth=truth,
        cartesian_blurred=cartesian_blurred,
        psf_axes=psf_axes,
        psf_cartesian=psf,
        fine_axes=fine_axes,
        skewed_psf=skewed_psf,
        skewed_blurred=skewed_blurred,
        instrument_extents=instrument_extents,
    )


def comparison_metrics(actual, reference):
    """Compare absolute density, without fitting a gain or registering images."""
    # Exclude the vast empty background from correlation.
    mask = reference > reference.max() * 0.01
    a, b = actual[mask].astype(np.float64), reference[mask].astype(np.float64)
    return dict(
        relative_l2=float(np.linalg.norm(a - b) / np.linalg.norm(b)),
        correlation=float(np.corrcoef(a, b)[0, 1]),
        total_signal_ratio=float(
            actual.sum(dtype=np.float64) / reference.sum(dtype=np.float64)
        ),
    )


def acquire_sphere(
    prepared,
    scan_step_um=0.2,
    camera_samples=1,
    peak_electrons=None,
    background_electrons=0.0,
    read_noise_e=0.0,
    seed=42,
):
    """Acquire, optionally integrate/noise, and run unchanged production deskew."""
    p = prepared["params"]
    if scan_step_um < p["fine_scan_step_um"]:
        raise ValueError("Scan step must be at least the fine instrument scan step")
    if peak_electrons is not None and peak_electrons <= 0:
        raise ValueError("peak_electrons must be positive")
    if background_electrons < 0:
        raise ValueError("background_electrons must be nonnegative")
    pixel = p["pixel_size_um"]
    theta = np.deg2rad(p["angle_deg"])
    raw_axes = tuple(
        centered_axis(e, d)
        for e, d in zip(
            prepared["instrument_extents"],
            (scan_step_um, pixel, pixel),
        )
    )
    print(
        f"Acquiring {scan_step_um:g} um scan, camera samples={camera_samples}",
        flush=True,
    )
    raw = pixel_average(
        prepared["skewed_blurred"],
        prepared["fine_axes"],
        raw_axes,
        pixel,
        camera_samples,
    )
    # An independent forward route: convolve on the Cartesian grid first,
    # then sample at exactly the same raw instrument pixel centers.
    raw_reference = np.zeros_like(raw)
    offsets = ((np.arange(camera_samples) + 0.5) / camera_samples - 0.5) * pixel
    for dy in offsets:
        for dx in offsets:
            raw_reference += sample_skewed(
                prepared["cartesian_blurred"],
                prepared["axes"],
                raw.shape,
                pixel_size_um=pixel,
                scan_step_um=scan_step_um,
                angle_deg=p["angle_deg"],
                center_xyz_um=(dx, dy * np.cos(theta), dy * np.sin(theta)),
            )
    raw_reference /= camera_samples**2
    forward_metrics = comparison_metrics(raw, raw_reference)
    raw_clean = raw.copy()
    raw_electrons = None
    gain = None
    if peak_electrons is not None:
        gain = peak_electrons / float(prepared["cartesian_blurred"].max())
        raw_electrons = camera_noise(
            raw * gain + background_electrons, read_noise_e, seed
        )
        raw = np.maximum((raw_electrons - background_electrons) / gain, 0)
    print("Orthogonal deskew (unchanged production implementation)", flush=True)
    deskewed = orthogonal_deskew(
        raw,
        theta=p["angle_deg"],
        distance=scan_step_um,
        pixel_size=pixel,
        downsample_factor=1,
        divisible_by=1,
    )
    # The simulator uses density, while deskew has a known 2*pixel/scan gain.
    # Undo that scalar for comparison; do not fit brightness to the reference.
    deskewed *= scan_step_um / (2 * pixel)
    output_axes = (
        np.arange(deskewed.shape[0]) * pixel - raw_axes[1][-1] * np.sin(theta),
        np.arange(deskewed.shape[1]) * pixel
        - raw_axes[0][-1]
        - raw_axes[1][-1] * np.cos(theta),
        np.arange(deskewed.shape[2]) * pixel - raw_axes[2][-1],
    )
    bounds = [
        np.flatnonzero((a >= prepared["axes"][i][0]) & (a <= prepared["axes"][i][-1]))
        for i, a in enumerate(output_axes)
    ]
    crop = tuple(slice(int(b[0]), int(b[-1]) + 1) for b in bounds)
    output_axes = tuple(a[s] for a, s in zip(output_axes, crop))
    deskewed = deskewed[crop].copy()
    reference = pixel_average(
        prepared["cartesian_blurred"],
        prepared["axes"],
        output_axes,
        pixel,
        camera_samples,
    )
    reference_clean = reference.copy()
    if gain is not None:
        reference_e = camera_noise(
            reference * gain + background_electrons, read_noise_e, seed + 1
        )
        reference = np.maximum((reference_e - background_electrons) / gain, 0)
    truth_sampled = resample_grid(prepared["truth"], prepared["axes"], output_axes)
    metrics = dict(
        forward_routes=forward_metrics,
        deskew_vs_normal=comparison_metrics(deskewed, reference),
        deskew_vs_noiseless_normal=comparison_metrics(deskewed, reference_clean),
    )
    print(json.dumps(metrics), flush=True)
    return dict(
        raw=raw,
        raw_clean=raw_clean,
        raw_electrons=raw_electrons,
        raw_axes=raw_axes,
        output_axes=output_axes,
        deskewed=deskewed,
        normal=reference,
        normal_clean=reference_clean,
        truth_sampled=truth_sampled,
        metrics=metrics,
        settings=dict(
            scan_step_um=scan_step_um,
            camera_samples=camera_samples,
            peak_electrons=peak_electrons,
            counts_per_density=gain,
            background_electrons=background_electrons,
            read_noise_e=read_noise_e,
            seed=seed,
        ),
    )


def save_simulation(prepared, acquired, output):
    """Write viewable volumes and complete physical axes/parameters, without plots."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pixel = prepared["params"]["pixel_size_um"]
    for name in ("deskewed", "normal", "normal_clean", "truth_sampled"):
        imwrite(
            output / f"{name}.ome.tif",
            acquired[name],
            metadata={
                "axes": "ZYX",
                "PhysicalSizeX": pixel,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel,
                "PhysicalSizeYUnit": "µm",
                "PhysicalSizeZ": pixel,
                "PhysicalSizeZUnit": "µm",
            },
            compression="zlib",
        )
    # Raw Z is the scan axis, not Cartesian Z; the explicit sidecar records SYX.
    imwrite(
        output / "raw_skewed.tif",
        acquired["raw"],
        metadata={"axes": "ZYX"},
        compression="zlib",
    )
    if acquired["raw_electrons"] is not None:
        imwrite(
            output / "raw_electrons.tif", acquired["raw_electrons"], compression="zlib"
        )

    def axes(values):
        return [
            dict(origin_um=float(a[0]), step_um=float(a[1] - a[0]), size=len(a))
            for a in values
        ]

    metadata = dict(
        parameters=prepared["params"],
        acquisition=acquired["settings"],
        metrics=acquired["metrics"],
        raw_axes_syx=axes(acquired["raw_axes"]),
        output_axes_zyx=axes(acquired["output_axes"]),
        fine_cartesian_axes_zyx=axes(prepared["axes"]),
        fine_instrument_axes_syx=axes(prepared["fine_axes"]),
        cartesian_psf_axes_zyx=axes(prepared["psf_axes"]),
        skewed_psf_axes_syx=[
            dict(origin_um=-(n - 1) / 2 * d, step_um=d, size=n)
            for n, d in zip(
                prepared["skewed_psf"].shape,
                (
                    prepared["params"]["fine_scan_step_um"],
                    pixel / 2,
                    pixel / 2,
                ),
            )
        ],
        fluorescence="union of tubes, diameter measured at meridian centerlines",
        comparison_gain="deskew multiplied by scan_step/(2*pixel), no fitted gain",
    )
    (output / "simulation.json").write_text(json.dumps(metadata, indent=2))


def main():
    """Run the reproducible sphere experiment, exporting each requested scan step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-steps", type=float, nargs="+", default=[0.2, 0.8])
    parser.add_argument("--fine-spacing", type=float, default=0.05)
    parser.add_argument("--fine-scan-step", type=float, default=0.1)
    parser.add_argument("--pixel-size", type=float, default=0.115)
    parser.add_argument("--angle", type=float, default=30.0)
    parser.add_argument("--na", type=float, default=1.35)
    parser.add_argument("--wavelength", type=float, default=0.637)
    parser.add_argument("--diameter", type=float, default=10.0)
    parser.add_argument("--tube-diameter", type=float, default=1.0)
    parser.add_argument("--meridians", type=int, default=12)
    parser.add_argument("--camera-samples", type=int, default=1)
    parser.add_argument("--peak-electrons", type=float)
    parser.add_argument("--background-electrons", type=float, default=0.0)
    parser.add_argument("--read-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepared = prepare_sphere(
        diameter_um=args.diameter,
        tube_diameter_um=args.tube_diameter,
        meridians=args.meridians,
        fine_spacing_um=args.fine_spacing,
        pixel_size_um=args.pixel_size,
        fine_scan_step_um=args.fine_scan_step,
        angle_deg=args.angle,
        numerical_aperture=args.na,
        wavelength_um=args.wavelength,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    fine_metadata = {"axes": "ZYX"}
    for axis in "XYZ":
        fine_metadata[f"PhysicalSize{axis}"] = args.fine_spacing
        fine_metadata[f"PhysicalSize{axis}Unit"] = "µm"
    imwrite(
        args.output / "ground_truth_fine.tif",
        prepared["truth"],
        compression="zlib",
        ome=True,
        metadata=fine_metadata,
    )
    imwrite(
        args.output / "psf_cartesian_fine.tif",
        prepared["psf_cartesian"],
        compression="zlib",
        ome=True,
        metadata=fine_metadata,
    )
    imwrite(
        args.output / "psf_skewed_fine.tif", prepared["skewed_psf"], compression="zlib"
    )
    for step in args.scan_steps:
        acquired = acquire_sphere(
            prepared,
            step,
            args.camera_samples,
            args.peak_electrons,
            args.background_electrons,
            args.read_noise,
            args.seed,
        )
        save_simulation(prepared, acquired, args.output / f"scan_{step:g}um")


if __name__ == "__main__":
    main()
