"""Show saved zero-filled camera planes in instrument and Cartesian YZ views.

Read manifest.json and camera_*um_zero_filled.tif from the publication figure
directory. Verify the skew sampler against an analytic camera-plane equation
and draw the same samples in both coordinate systems. Write plane_geometry
figures in PNG, SVG, and PDF and plane_geometry_check.json beside the inputs.
Requires matplotlib; run from the repository root with
``uv run --with matplotlib python -m scripts.plot_opm_plane_geometry DIRECTORY``.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread

from .psf_sampling_experiment import sample_skewed


def main():
    """Check the sampler's plane equation and plot raw cells at physical positions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    manifest = json.loads((directory / "manifest.json").read_text())
    params = manifest["parameters"]
    theta = np.deg2rad(params["angle_deg"])
    cotangent = 1 / np.tan(theta)
    # A Cartesian field equal to Y-Z*cot(theta) must be constant within each
    # camera frame and equal that frame's scan position. Exercise the actual
    # sampler, not just the plotting transform, on this analytic field.
    axes = tuple(np.linspace(-6, 6, 61) for _ in range(3))
    zz, yy, xx = np.meshgrid(*axes, indexing="ij", sparse=True)
    field = np.broadcast_to(yy - zz * cotangent, (61, 61, 61)).copy()
    checks = []
    for step in (0.4, 0.8, 1.2):
        values = sample_skewed(
            field,
            axes,
            (5, 9, 7),
            pixel_size_um=0.115,
            scan_step_um=step,
            angle_deg=params["angle_deg"],
        )
        expected = np.broadcast_to(
            (np.arange(5) - 2)[:, None, None] * step, values.shape
        )
        error = float(np.max(np.abs(values - expected)))
        np.testing.assert_allclose(values, expected, atol=1e-12, rtol=0)
        checks.append(dict(scan_step_um=step, maximum_plane_equation_error_um=error))
    source_axes = manifest["camera_display_axes_syx"]
    edges = []
    for a in source_axes[:2]:
        edges.append(a["origin_um"] + (np.arange(a["size"] + 1) - 0.5) * a["step_um"])
    ss, rr = np.meshgrid(*edges, indexing="ij")
    cartesian_y = ss + rr * np.cos(theta)
    cartesian_z = rr * np.sin(theta)
    np.testing.assert_allclose(cartesian_y - cartesian_z * cotangent, ss, atol=1e-12)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axs = plt.subplots(2, 3, figsize=(9, 6.8), layout="constrained")
    for col, step in enumerate((0.4, 0.8, 1.2)):
        raw = imread(directory / f"camera_{step:.1f}um_zero_filled.tif")
        side = raw.max(axis=2)
        axs[0, col].pcolormesh(
            rr, ss, side, cmap="gray", vmin=0, vmax=1, shading="flat", rasterized=True
        )
        im = axs[1, col].pcolormesh(
            cartesian_y,
            cartesian_z,
            side,
            cmap="gray",
            vmin=0,
            vmax=1,
            shading="flat",
            rasterized=True,
        )
        axs[0, col].set(
            title=f"{step:g} µm acquisition",
            xlabel="Camera row r (µm)",
            ylabel="Scan position s (µm)",
            xlim=(-12, 12),
            ylim=(-12, 12),
        )
        axs[1, col].set(
            xlabel="Cartesian Y (µm)",
            ylabel="Cartesian Z (µm)",
            xlim=(-6, 6),
            ylim=(-6, 6),
        )
        for ax in axs[:, col]:
            ax.set_aspect("equal")
            ax.set_facecolor("black")
    fig.suptitle(
        "Same camera samples, two coordinate displays\n"
        "Top: instrument row–scan    |    Bottom: Cartesian YZ",
        fontsize=12,
    )
    fig.colorbar(im, ax=axs, shrink=0.65, label="Fluorescence density (a.u.)")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(directory / f"plane_geometry.{suffix}", dpi=400)
    plt.close(fig)
    report = dict(
        plane_equation="Z = tan(theta) * (Y - s)",
        angle_deg=params["angle_deg"],
        side_view_slope=float(np.tan(theta)),
        sampler_checks=checks,
        display="Affine placement of raw pixel cells; no intensity interpolation",
    )
    (directory / "plane_geometry_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
