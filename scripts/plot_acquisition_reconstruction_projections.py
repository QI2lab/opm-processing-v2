"""Compare acquisition and reconstructed Cartesian projections at four scan spacings.

Read the four saved sphere acquisitions supplied by --acquisitions, their
simulation metadata, and native or scan-upsampled deconvolution outputs.
Place zero-filled camera samples on the Cartesian grid and compare their
maximum projections with the final deskewed reconstructions on physical axes.
Write acquisition_reconstruction_mips figures in PNG, SVG, and PDF, an NPZ
of plotted source arrays, and a JSON provenance record to --output.
Requires matplotlib; run as
``uv run --with matplotlib python -m scripts.plot_acquisition_reconstruction_projections --acquisitions DIR1 DIR2 DIR3 DIR4 --output DIR``
from the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import map_coordinates
from tifffile import imread

from scripts.publication_fixed_plane_series import physical_axes

if TYPE_CHECKING:
    from collections.abc import Sequence


def acquisition_projections(
    raw: np.ndarray,
    axes: Sequence[np.ndarray],
    factor: int,
    theta: float,
    display_axis: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], int, int]:
    """Map raw cells into Cartesian space by nearest neighbor, preserving zero gaps.

    Parameters
    ----------
    raw
        Saved oblique camera image in scan, row, column order.
    axes
        Physical coordinate arrays for the source image, in storage-axis order.
    factor
        Integer spacing between measured scan planes on the fine reconstruction grid.
    theta
        Detector plane angle in radians.
    display_axis
        Physical Cartesian coordinates used for projection voxelization.

    Returns
    -------
    tuple
        Cartesian maximum projections and counts of measured and zero-filled scan planes.
    """
    scans, rows, columns = axes
    scan_step = (scans[1] - scans[0]) / factor
    assert np.isclose(scan_step, 0.4)
    filled = np.zeros(((len(scans) - 1) * factor + 1, *raw.shape[1:]), dtype=raw.dtype)
    filled[::factor] = raw
    measured = np.arange(filled.shape[0]) % factor == 0
    np.testing.assert_array_equal(filled[measured], raw)
    assert not np.any(filled[~measured])
    # Inverse physical map: row = Z/sin(theta), scan = Y-Z*cot(theta), column = X.
    # Order zero assigns each Cartesian display sample the value of its raw
    # instrument cell. No linear interpolation can fill a missing plane here.
    n = len(display_axis)
    xy = np.zeros((n, n), dtype=raw.dtype)
    xz = np.zeros((n, n), dtype=raw.dtype)
    yz = np.zeros((n, n), dtype=raw.dtype)
    x_index = (display_axis - columns[0]) / (columns[1] - columns[0])
    for zi, z in enumerate(display_axis):
        scan_index = (display_axis - z / np.tan(theta) - scans[0]) / scan_step
        row_index = (z / np.sin(theta) - rows[0]) / (rows[1] - rows[0])
        ss, xx = np.meshgrid(scan_index, x_index, indexing="ij")
        section = map_coordinates(
            filled,
            np.stack((ss, np.full_like(ss, row_index), xx)),
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        np.maximum(xy, section, out=xy)
        xz[zi] = section.max(axis=0)
        yz[zi] = section.max(axis=1)
    return (xy, xz, yz), int(measured.sum()), int((~measured).sum())


def main() -> None:
    """Render paired acquisition and reconstruction projections for all four steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisitions", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})
    # The saved Cartesian outputs cover approximately +/-7.8 um. Use the same
    # complete common physical box for projections of both acquisition and reconstruction.
    display_axis = np.arange(-156, 157) * 0.05
    source = {"acquisition_display_axis_um": display_axis}
    report = {
        "display_method": "Nearest-neighbor physical voxelization, no intensity interpolation",
        "acquisition_display_grid_um": 0.05,
        "zero_filled_scan_step_um": 0.4,
        "physical_projection_box_um": [-7.8, 7.8],
        "intensity_limits": [0, 1],
        "cases": [],
    }
    fig, axs = plt.subplots(4, 6, figsize=(15, 10.8), layout="constrained")
    orientations = ((0, 2, 1, "XY"), (1, 2, 0, "XZ"), (2, 1, 0, "YZ"))
    for row, directory in enumerate(args.acquisitions):
        factor = row + 1
        step = 0.4 * factor
        meta = json.loads((directory / "simulation.json").read_text())
        assert np.isclose(meta["acquisition"]["scan_step_um"], step)
        theta = np.deg2rad(meta["parameters"]["angle_deg"])
        raw = imread(directory / "raw_skewed.tif")
        print(f"Cartesian acquisition MIPs for {step:.1f} um", flush=True)
        projections, measured, zeros = acquisition_projections(
            raw, physical_axes(meta["raw_axes_syx"]), factor, theta, display_axis
        )
        decon = directory / (
            "decon_native" if factor == 1 else f"decon_upsample{factor}"
        )
        dm = json.loads((decon / "simulation.json").read_text())
        raw_hash = hashlib.sha256(
            (directory / "raw_electrons.tif").read_bytes()
        ).hexdigest()
        assert raw_hash == dm["deconvolution"]["raw_file_sha256"]
        assert np.isclose(dm["deconvolution"]["reconstruction_step_um"], 0.4)
        recon = imread(decon / "deskewed.ome.tif")
        assert np.isfinite(recon).all()
        rec_axes = physical_axes(dm["output_axes_zyx"])
        crop = []
        for axis in rec_axes:
            keep = np.flatnonzero((axis >= -7.8) & (axis <= 7.8))
            crop.append(slice(int(keep[0]), int(keep[-1]) + 1))
        recon = recon[tuple(crop)]
        rec_axes = tuple(a[s] for a, s in zip(rec_axes, crop, strict=False))
        for plane, (collapsed, horizontal, vertical, name) in enumerate(orientations):
            before = projections[plane]
            after = recon.max(axis=collapsed)
            source[f"acquisition_{step:.1f}um_{name}"] = before
            source[f"reconstruction_{step:.1f}um_{name}"] = after
            for stage, (data, axes) in enumerate(
                ((before, (display_axis,) * 3), (after, rec_axes))
            ):
                ax = axs[row, plane * 2 + stage]
                h, v = axes[horizontal], axes[vertical]
                dh, dv = h[1] - h[0], v[1] - v[0]
                extent = (h[0] - dh / 2, h[-1] + dh / 2, v[0] - dv / 2, v[-1] + dv / 2)
                ax.imshow(
                    data,
                    origin="lower",
                    extent=extent,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                    aspect="equal",
                )
                ax.set(
                    xlim=(-8, 8),
                    ylim=(-8, 8),
                    xlabel=f"{'ZYX'[horizontal]} (µm)",
                    ylabel=f"{'ZYX'[vertical]} (µm)",
                )
                ax.set_facecolor("black")
                ax.set_xticks([-8, 0, 8])
                ax.set_yticks([-8, 0, 8])
                if row == 0:
                    ax.set_title(
                        f"{name}: "
                        + ("acquisition" if stage == 0 else "reconstruction"),
                        fontsize=10,
                    )
                if plane == 0 and stage == 0:
                    ax.set_ylabel(f"{step:.1f} µm acquisition\nY (µm)", fontsize=9)
        for index, axis in enumerate(rec_axes):
            source[f"reconstruction_{step:.1f}um_axis{index}_um"] = axis
        report["cases"].append(
            {
                "acquisition": str(directory.resolve()),
                "reconstruction": str(decon.resolve()),
                "raw_electrons_sha256": raw_hash,
                "measured_planes": measured,
                "zero_planes": zeros,
            }
        )
    fig.suptitle(
        "Acquisition → reconstruction: Cartesian maximum-intensity projections\n"
        "0.4, 0.8, 1.2 and 1.6 µm acquisitions; all GC reconstructions at 0.4 µm, then orthogonal deskew",
        fontsize=13,
    )
    fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap="gray"),
        ax=axs,
        shrink=0.55,
        label="Fluorescence density (a.u.)",
        ticks=[0, 0.5, 1],
    )
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(args.output / f"acquisition_reconstruction_mips.{suffix}", dpi=600)
    plt.close(fig)
    np.savez_compressed(
        args.output / "acquisition_reconstruction_mips_source.npz", **source
    )
    (args.output / "acquisition_reconstruction_mips.json").write_text(
        json.dumps(report, indent=2)
    )
    print("Finished paired Cartesian MIP figure", flush=True)


if __name__ == "__main__":
    main()
