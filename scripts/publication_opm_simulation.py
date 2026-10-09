"""Generate publication figures for sphere imaging at three scan spacings.

Read saved 0.4, 0.8, and 1.2 micrometer acquisitions and their reconstruction
outputs, along with the fine Cartesian object in the acquisition parent folder.
Compare object sampling, optical blur, camera integration/noise, zero-filled
missing scan planes, and deconvolution using common physical/intensity scales.
Write PNG, SVG, PDF, and TIFF figures, zero-filled camera TIFFs, plotted source arrays
in projection_source_data.npz, and manifest.json provenance to --output.
Requires matplotlib; run as
``uv run --with matplotlib python -m scripts.publication_opm_simulation --acquisitions DIR1 DIR2 DIR3 --output DIR``
from the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from tifffile import imread, imwrite

from scripts.psf_sampling_experiment import sample_skewed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.gridspec import SubplotSpec

PLANES = ((0, 2, 1, "XY"), (1, 2, 0, "XZ"), (2, 1, 0, "YZ"))


def axes_from_metadata(items: Sequence[dict[str, float]]) -> tuple[np.ndarray, ...]:
    """Recover physical voxel-center coordinates from exported metadata.

    Parameters
    ----------
    items
        Saved physical axis records containing origin, step, and sample count.

    Returns
    -------
    tuple[np.ndarray, ...]
        Physical coordinate arrays reconstructed from saved axis metadata.
    """
    return tuple(a["origin_um"] + np.arange(a["size"]) * a["step_um"] for a in items)


def record(
    volume: np.ndarray, axes: Sequence[np.ndarray], instrument: bool = False
) -> dict[str, Any]:
    """Compute the three full-volume maximum-intensity projections.

    Parameters
    ----------
    volume
        ZYX image volume to project.
    axes
        Physical coordinate arrays for the source image, in storage-axis order.
    instrument
        Interpret axes as scan/row/column instead of laboratory ZYX.

    Returns
    -------
    dict
        Image samples and physical axes packaged for the publication projection panels.
    """
    return {
        "projections": [volume.max(axis=a) for a, _, _, _ in PLANES],
        "axes": axes,
        "instrument": instrument,
    }


def insert_missing_planes(
    raw: np.ndarray, raw_axes: Sequence[np.ndarray], display_axes: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Place measured planes by physical position; unmeasured planes remain zero.

    Parameters
    ----------
    raw
        Saved oblique camera image in scan, row, column order.
    raw_axes
        Physical coordinates of the measured camera sample centers.
    display_axes
        Fine instrument grid coordinates including unmeasured scan planes.

    Returns
    -------
    np.ndarray
        Camera samples inserted into the display grid with unmeasured planes left zero.
    """
    for source, target in zip(raw_axes[1:], display_axes[1:], strict=False):
        np.testing.assert_allclose(source, target, atol=1e-10)
    coordinates = (raw_axes[0] - display_axes[0][0]) / 0.4
    indices = np.rint(coordinates).astype(int)
    np.testing.assert_allclose(coordinates, indices, atol=1e-9)
    if indices.min() < 0 or indices.max() >= len(display_axes[0]):
        raise ValueError("Camera planes exceed the common display grid")
    result = np.zeros(tuple(map(len, display_axes)), dtype=raw.dtype)
    result[indices] = raw
    mask = np.zeros(len(display_axes[0]), dtype=bool)
    mask[indices] = True
    np.testing.assert_array_equal(result[indices], raw)
    assert not np.any(result[~mask])
    return result, mask


def image_panel(ax: Axes, item: dict[str, Any], row: int) -> None:
    """Display physical coordinates without smoothing or independent rescaling.

    Parameters
    ----------
    ax
        Matplotlib axes receiving the drawing.
    item
        Saved image record with data and physical coordinate axes.
    row
        Projection orientation index for the figure row.
    """
    _, horizontal, vertical, name = PLANES[row]
    axes = item["axes"]
    h, v = axes[horizontal], axes[vertical]
    dh, dv = h[1] - h[0], v[1] - v[0]
    extent = (h[0] - dh / 2, h[-1] + dh / 2, v[0] - dv / 2, v[-1] + dv / 2)
    ax.imshow(
        item["projections"][row],
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    if item["instrument"]:
        labels = ("Z′ (scan)", "Y′ (row)", "X′ (column)")
        ranges = (12, 12, 6)
    else:
        labels = ("Z", "Y", "X")
        ranges = (6, 6, 6)
    ax.set(
        xlim=(-ranges[horizontal], ranges[horizontal]),
        ylim=(-ranges[vertical], ranges[vertical]),
        xlabel=f"{labels[horizontal]} (µm)",
        ylabel=f"{labels[vertical]} (µm)",
    )
    ax.set_xticks([-ranges[horizontal], 0, ranges[horizontal]])
    ax.set_yticks([-ranges[vertical], 0, ranges[vertical]])
    ax.tick_params(length=2, width=0.5, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.text(
        0.04,
        0.96,
        "".join(letter + "′" for letter in name) if item["instrument"] else name,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=7,
    )


def draw_block(
    fig: Figure,
    spec: SubplotSpec,
    records: Sequence[dict[str, Any]],
    titles: Sequence[str],
    heading: str,
) -> None:
    """Lay out a labeled group of three projections for each volume.

    Parameters
    ----------
    fig
        Matplotlib figure to write and close.
    spec
        GridSpec region in which the figure block is drawn.
    records
        Image/axis records drawn in the comparison block.
    titles
        Column titles for the comparison images.
    heading
        Heading displayed above the comparison block.
    """
    sub = spec.subgridspec(
        4, len(records), height_ratios=[0.21, 1, 1, 1], hspace=0.43, wspace=0.58
    )
    header = fig.add_subplot(sub[0, :])
    header.axis("off")
    header.text(0, 1.05, heading, fontsize=10, weight="bold", va="top")
    for col, item in enumerate(records):
        for row in range(3):
            ax = fig.add_subplot(sub[row + 1, col])
            image_panel(ax, item, row)
            if row == 0:
                ax.set_title(titles[col], fontsize=8, pad=8)


def save_figure(fig: Figure, output: Path, stem: str) -> None:
    """Export vector text and high-resolution raster images for publication.

    Parameters
    ----------
    fig
        Matplotlib figure to write and close.
    output
        Directory or file receiving the generated images and figures.
    stem
        Output filename stem without the figure extension.
    """
    for suffix in ("pdf", "svg", "png", "tiff"):
        options = (
            {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tiff" else {}
        )
        fig.savefig(output / f"{stem}.{suffix}", dpi=600, facecolor="white", **options)
    plt.close(fig)


def main() -> None:
    """Validate saved acquisitions and render the forward and reconstruction panels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acquisitions",
        type=Path,
        nargs=3,
        required=True,
        help="Saved 0.4, 0.8 and 1.2 µm acquisitions, in that order",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    metas = [json.loads((d / "simulation.json").read_text()) for d in args.acquisitions]
    p = metas[0]["parameters"]
    for meta, step in zip(metas, (0.4, 0.8, 1.2), strict=False):
        assert meta["parameters"] == p
        assert np.isclose(meta["acquisition"]["scan_step_um"], step)
        for key in (
            "camera_samples",
            "peak_electrons",
            "background_electrons",
            "read_noise_e",
            "seed",
        ):
            assert meta["acquisition"][key] == metas[0]["acquisition"][key]
    fine_axes = axes_from_metadata(metas[0]["fine_cartesian_axes_zyx"])
    truth = imread(args.acquisitions[0].parent / "ground_truth_fine.tif")
    forward = [record(truth, fine_axes)]
    fine_raw_axes = axes_from_metadata(metas[0]["fine_instrument_axes_syx"])
    print("Sampling unblurred fine object in instrument coordinates", flush=True)
    skewed = sample_skewed(
        truth,
        fine_axes,
        tuple(map(len, fine_raw_axes)),
        pixel_size_um=p["pixel_size_um"] / 2,
        scan_step_um=p["fine_scan_step_um"],
        angle_deg=p["angle_deg"],
    )
    forward.append(record(skewed, fine_raw_axes, instrument=True))
    del truth, skewed
    display_axes = axes_from_metadata(metas[0]["raw_axes_syx"])
    manifest = {
        "display_scan_step_um": 0.4,
        "projection": "maximum intensity",
        "intensity_limits": [0, 1],
        "parameters": p,
        "cases": [],
    }
    reconstructions = []
    for directory, meta, factor in zip(
        args.acquisitions, metas, (1, 2, 3), strict=False
    ):
        raw_path = directory / "raw_skewed.tif"
        raw = imread(raw_path)
        raw_axes = axes_from_metadata(meta["raw_axes_syx"])
        zero_filled, mask = insert_missing_planes(raw, raw_axes, display_axes)
        forward.append(record(zero_filled, display_axes, instrument=True))
        imwrite(
            output / f"camera_{factor * 0.4:.1f}um_zero_filled.tif",
            zero_filled,
            compression="zlib",
            metadata={
                "axes": "ZYX",
                "description": "Stored ZYX denotes scan,row,column, not Cartesian; see manifest.json.",
            },
        )
        decon_dir = directory / (
            "decon_native" if factor == 1 else f"decon_upsample{factor}"
        )
        dm = json.loads((decon_dir / "simulation.json").read_text())
        assert np.isclose(dm["deconvolution"]["reconstruction_step_um"], 0.4)
        assert dm.get("valid_for_gc_comparison", True)
        electron_hash = hashlib.sha256(
            (directory / "raw_electrons.tif").read_bytes()
        ).hexdigest()
        assert electron_hash == dm["deconvolution"]["raw_file_sha256"]
        output_axes = axes_from_metadata(dm["output_axes_zyx"])
        reference = imread(decon_dir / "normal.ome.tif")
        if factor == 1:
            reference_fixed = reference
            reference_axes = output_axes
            reconstructions.append(record(reference, output_axes))
        else:
            np.testing.assert_allclose(reference, reference_fixed, rtol=1e-6, atol=1e-7)
            for a, b in zip(output_axes, reference_axes, strict=False):
                np.testing.assert_allclose(a, b, atol=1e-9)
        reconstructed = imread(decon_dir / "deskewed.ome.tif")
        assert np.isfinite(reconstructed).all()
        reconstructions.append(record(reconstructed, output_axes))
        manifest["cases"].append(
            {
                "acquisition": str(directory.resolve()),
                "reconstruction": str(decon_dir.resolve()),
                "acquisition_settings": meta["acquisition"],
                "raw_electrons_sha256": electron_hash,
                "measured_planes": int(mask.sum()),
                "zero_planes": int((~mask).sum()),
                "measured_scan_indices": np.flatnonzero(mask).tolist(),
                "metrics": dm["metrics"],
            }
        )
    manifest["camera_display_axes_syx"] = metas[0]["raw_axes_syx"]
    manifest["fine_instrument_axes_syx"] = metas[0]["fine_instrument_axes_syx"]
    manifest["fine_cartesian_axes_zyx"] = metas[0]["fine_cartesian_axes_zyx"]
    manifest["reconstruction_axes_zyx"] = metas[0]["output_axes_zyx"]
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    source = {}
    for group, records in (("forward", forward), ("reconstruction", reconstructions)):
        for col, item in enumerate(records):
            for row, (_, _, _, plane) in enumerate(PLANES):
                source[f"{group}_{col}_{plane}"] = item["projections"][row]
            for axis, values in enumerate(item["axes"]):
                source[f"{group}_{col}_axis{axis}_um"] = values
    np.savez_compressed(output / "projection_source_data.npz", **source)
    forward_titles = [
        "Ground truth",
        "→ Skewed object\n(before optical blur)",
        "→ Camera: 0.4 µm\nall planes measured",
        "Camera: 0.8 µm\n1 of 2 planes measured",
        "Camera: 1.2 µm\n1 of 3 planes measured",
    ]
    reconstruction_titles = [
        "Cartesian reference\nGC deconvolution",
        "OPM: 0.4 → 0.4 µm\nGC + deskew",
        "OPM: 0.8 → 0.4 µm\nGC + deskew",
        "OPM: 1.2 → 0.4 µm\nGC + deskew",
    ]
    headings = [
        "a  Forward model and sampled camera data",
        "b  Deconvolution and orthogonal deskew",
    ]
    for stem, records, titles, heading in (
        ("forward_model", forward, forward_titles, headings[0]),
        ("reconstructions", reconstructions, reconstruction_titles, headings[1]),
    ):
        fig = plt.figure(figsize=(10, 7.7))
        spec = fig.add_gridspec(1, 1, left=0.07, right=0.91, bottom=0.08, top=0.96)
        draw_block(fig, spec[0], records, titles, heading)
        cax = fig.add_axes((0.94, 0.37, 0.01, 0.22))
        fig.colorbar(
            ScalarMappable(norm=Normalize(0, 1), cmap="gray"),
            cax=cax,
            label="Fluorescence density (a.u.)",
            ticks=[0, 0.5, 1],
        )
        save_figure(fig, output, stem)
    fig = plt.figure(figsize=(10, 15.0))
    spec = fig.add_gridspec(
        2, 1, left=0.07, right=0.91, bottom=0.04, top=0.98, hspace=0.12
    )
    draw_block(fig, spec[0], forward, forward_titles, headings[0])
    draw_block(fig, spec[1], reconstructions, reconstruction_titles, headings[1])
    for y in (0.70, 0.20):
        cax = fig.add_axes((0.94, y, 0.01, 0.12))
        fig.colorbar(
            ScalarMappable(norm=Normalize(0, 1), cmap="gray"),
            cax=cax,
            label="Fluorescence density (a.u.)",
            ticks=[0, 0.5, 1],
        )
    save_figure(fig, output, "opm_sampling_reconstruction")
    print(f"Publication figures and source data: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
