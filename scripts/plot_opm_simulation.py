"""Plot saved sphere acquisitions or deconvolved microscope comparisons.

Read simulation.json and truth_sampled, normal, and deskewed OME-TIFFs from the
input directory. Render physical maximum projections and central sections on
common intensity scales, plus a view of the raw oblique camera acquisition.
Write comparison.png, sections.png, and raw_instrument.png beside the inputs.
Requires matplotlib; run from the repository root with
``uv run --with matplotlib python -m scripts.plot_opm_simulation DIRECTORY``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread


def main() -> None:
    """Export common-scale physical projections and an instrument-frame view."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    path = args.directory
    meta = json.loads((path / "simulation.json").read_text())
    if meta.get("valid_for_gc_comparison") is False:
        raise ValueError(
            "This GC comparison was withdrawn; use the camera-integrated noisy acquisition"
        )
    fields = [
        imread(path / f"{name}.ome.tif")
        for name in ("truth_sampled", "normal", "deskewed")
    ]
    labels = [
        "Ground truth (sampled)",
        "Cartesian microscope",
        "OPM + orthogonal deskew",
    ]
    if meta.get("comparison_stage") == "deconvolution":
        labels[1] = "Cartesian deconvolution"
        labels[2] = "OPM deconvolution + deskew"
    axes = meta["output_axes_zyx"]
    a = meta["acquisition"]
    p = meta["parameters"]
    m = meta["metrics"]["deskew_vs_normal"]
    plot_radius = (p["diameter_um"] + p["tube_diameter_um"]) / 2 + 0.5
    for sections in (False, True):
        fig, axs = plt.subplots(3, 4, figsize=(12, 9), layout="constrained")
        for row, (projection_axis, horizontal, vertical, name) in enumerate(
            [
                (0, 2, 1, "XY"),
                (1, 2, 0, "XZ"),
                (2, 1, 0, "YZ"),
            ]
        ):
            section_index = int(
                np.argmin(
                    np.abs(
                        axes[projection_axis]["origin_um"]
                        + np.arange(axes[projection_axis]["size"])
                        * axes[projection_axis]["step_um"]
                    )
                )
            )
            projections = [
                np.take(a, section_index, axis=projection_axis)
                if sections
                else a.max(axis=projection_axis)
                for a in fields
            ]
            h, v = axes[horizontal], axes[vertical]
            extent = (
                h["origin_um"] - h["step_um"] / 2,
                h["origin_um"] + (h["size"] - 0.5) * h["step_um"],
                v["origin_um"] - v["step_um"] / 2,
                v["origin_um"] + (v["size"] - 0.5) * v["step_um"],
            )
            for col, projection in enumerate(projections):
                axs[row, col].imshow(
                    projection,
                    origin="lower",
                    extent=extent,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                axs[row, col].set_title(f"{labels[col]}: {name}", fontsize=10)
            # Projection of the absolute volumetric difference; this does not hide
            # opposing errors by subtracting two independently maximized images.
            difference = np.abs(fields[2] - fields[1])
            error = (
                np.take(difference, section_index, axis=projection_axis)
                if sections
                else difference.max(axis=projection_axis)
            )
            im = axs[row, 3].imshow(
                error, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=0.2
            )
            axs[row, 3].set_title(
                f"{'Slice' if sections else 'Max'} |OPM − reference|: {name}",
                fontsize=10,
            )
            for ax in axs[row]:
                ax.set(
                    xlim=(-plot_radius, plot_radius),
                    ylim=(-plot_radius, plot_radius),
                    xlabel=f"{'ZYX'[horizontal]} (µm)",
                    ylabel=f"{'ZYX'[vertical]} (µm)",
                )
        fig.colorbar(im, ax=axs[:, 3], shrink=0.7, label="Density difference (0–0.2)")
        detection = (
            "noiseless"
            if a["peak_electrons"] is None
            else f"{a['peak_electrons']:g} peak electrons"
        )
        reconstruction = meta.get("deconvolution")
        stage_label = (
            f" | GC, reconstruction step {reconstruction['reconstruction_step_um']:g} µm"
            if reconstruction
            else ""
        )
        fig.suptitle(
            ("Central sections | " if sections else "Maximum projections | ")
            + f"{p['diameter_um']:g} µm meridian sphere, {p['tube_diameter_um']:g} µm tubes | {p['angle_deg']:g}° OPM, {a['scan_step_um']:g} µm scan, "
            f"{p['pixel_size_um'] * 1000:g} nm camera\n"
            f"NA {p['numerical_aperture']:g}, {p['wavelength_um'] * 1000:g} nm | "
            f"camera quadrature {a['camera_samples']}×{a['camera_samples']}, {detection}{stage_label} | "
            f"relative error {m['relative_l2']:.1%}, correlation {m['correlation']:.4f}",
            fontsize=12,
        )
        fig.savefig(path / ("sections.png" if sections else "comparison.png"), dpi=150)
        plt.close(fig)

    if not (path / "raw_skewed.tif").exists():
        print(path / "comparison.png")
        return
    raw = imread(path / "raw_skewed.tif")
    scan, camera_y, _ = meta["raw_axes_syx"]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.imshow(
        raw.max(axis=2).T,
        origin="lower",
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=(
            scan["origin_um"],
            scan["origin_um"] + (scan["size"] - 1) * scan["step_um"],
            camera_y["origin_um"],
            camera_y["origin_um"] + (camera_y["size"] - 1) * camera_y["step_um"],
        ),
    )
    ax.set(
        xlabel="Scan position (µm)",
        ylabel="Camera row position (µm)",
        title="Raw OPM: projection over camera columns (instrument coordinates)",
    )
    fig.savefig(path / "raw_instrument.png", dpi=150)
    print(path / "comparison.png")


if __name__ == "__main__":
    main()
