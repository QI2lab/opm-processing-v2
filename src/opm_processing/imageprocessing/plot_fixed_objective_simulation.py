"""Compare individual camera frames with a fixed plane and translating object."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tifffile import imread

from opm_processing.imageprocessing.opm_simulation import pixel_average, prepare_sphere


def main():
    """Translate the Cartesian optical field past a stationary detection plane."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acquisition", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.acquisition / "simulation.json").read_text())
    prepared = prepare_sphere(**meta["parameters"])
    p = prepared["params"]
    theta = np.deg2rad(p["angle_deg"])
    pitch = p["pixel_size_um"]
    scans, rows, columns = tuple(
        a["origin_um"] + np.arange(a["size"]) * a["step_um"]
        for a in meta["raw_axes_syx"]
    )
    indices = [int(np.argmin(abs(scans - s))) for s in (-8, -4, 0, 4, 8)]
    selected = scans[indices]
    # Independent route: shift the Cartesian blurred object in laboratory Y.
    # Detector points stay fixed for every frame, including all subpixels.
    samples = meta["acquisition"]["camera_samples"]
    offsets = ((np.arange(samples) + 0.5) / samples - 0.5) * pitch
    frames = []
    for s in selected:
        shifted_axes = (
            prepared["axes"][0],
            prepared["axes"][1] - s,
            prepared["axes"][2],
        )
        moving_object = RegularGridInterpolator(
            shifted_axes,
            prepared["cartesian_blurred"],
            bounds_error=False,
            fill_value=0,
        )
        frame = np.zeros((len(rows), len(columns)), dtype=np.float32)
        for dr in offsets:
            for dx in offsets:
                rr, xx = np.meshgrid(rows + dr, columns + dx, indexing="ij")
                fixed_detector = np.stack(
                    (rr * np.sin(theta), rr * np.cos(theta), xx), axis=-1
                )
                frame += moving_object(fixed_detector)
        frames.append(frame / samples**2)
    frames = np.asarray(frames)
    skew_route = pixel_average(
        prepared["skewed_blurred"],
        prepared["fine_axes"],
        (selected, rows, columns),
        pitch,
        samples,
    )
    errors = []
    for s, direct, skew in zip(selected, frames, skew_route):
        error = float(np.linalg.norm(direct - skew) / np.linalg.norm(direct))
        errors.append(dict(scan_position_um=float(s), relative_l2=error))
    raw = imread(args.acquisition / "raw_skewed.tif")[indices]
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axs = plt.subplots(3, 5, figsize=(12, 9), layout="constrained")
    rr = np.linspace(-25, 25, 100)
    extent = (
        columns[0] - pitch / 2,
        columns[-1] + pitch / 2,
        rows[0] - pitch / 2,
        rows[-1] + pitch / 2,
    )
    for col, s in enumerate(selected):
        ax = axs[0, col]
        ax.add_patch(
            Circle((-s, 0), p["diameter_um"] / 2, facecolor="0.9", edgecolor="0.3")
        )
        ax.plot(rr * np.cos(theta), rr * np.sin(theta), color="#0072B2", lw=2)
        ax.plot(-s, 0, "+", color="0.3")
        ax.set(
            title=f"Object translation: {-s:.1f} µm",
            xlabel="Laboratory Y (µm)",
            ylabel="Laboratory Z (µm)",
            xlim=(-15, 15),
            ylim=(-7, 7),
            aspect="equal",
        )
        for row, data in ((1, frames[col]), (2, raw[col])):
            axs[row, col].imshow(
                data,
                origin="lower",
                extent=extent,
                cmap="gray",
                vmin=0,
                vmax=0.5,
                interpolation="nearest",
                aspect="equal",
            )
            axs[row, col].set(
                xlim=(-6, 6),
                ylim=(-12, 12),
                xlabel="Camera column (µm)",
                ylabel="Camera row (µm)",
            )
        axs[1, col].set_title("Fixed-plane prediction")
        axs[2, col].set_title("Saved noisy camera frame")
    fig.suptitle(
        "Fixed oblique detection plane; object translates laterally\n"
        "Top: sphere outline and stationary plane | Middle/bottom: individual frames, not projections",
        fontsize=12,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output / f"fixed_objective_frames.{suffix}", dpi=300)
    plt.close(fig)
    report = dict(
        assumptions="Stationary oblique plane; lateral Y translation; spatially invariant Cartesian PSF",
        comparison="Noiseless fixed-plane prediction vs fine-skewed convolution, both pixel-integrated",
        frame_errors=errors,
        display_limits=[0, 0.5],
        scan_positions_um=selected.tolist(),
    )
    (args.output / "fixed_objective_check.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
