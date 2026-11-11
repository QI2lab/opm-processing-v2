"""Render approved AO pupils with the sjs_opm_scripts publication grid style."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def write_publication_pupil_grid(
    regions: Sequence[Any],
    output: Path,
    *,
    timepoint: int,
    depth: int,
    reference_path: Path,
) -> dict[str, Any]:
    """Save a pupil grid using the reference's layout, extents, and colorbars.

    Pupils are supplied by the approved AO reader. This function reuses only
    the publication formatting from ``analysis/ao_grid_phase_plots.py`` and
    does not import or execute its older phase reconstruction code.
    """
    if not regions:
        raise ValueError("A publication pupil grid requires at least one region")
    phase_limit = 2 * np.pi

    reference_path = Path(reference_path).expanduser().resolve()
    reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    basename = f"sjs_pupil_grid_t{timepoint:03d}_z{depth:03d}"
    png = output / f"{basename}.png"
    svg = output / f"{basename}.svg"

    columns = 2
    rows = math.ceil(len(regions) / columns)
    figure = plt.figure(figsize=(12, 5 * rows))
    grid = figure.add_gridspec(nrows=rows, ncols=columns, hspace=0.2, wspace=0.2)
    ticks = [-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi]
    ticklabels = [r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"]
    statistics = []
    colormap = str(plt.rcParams["image.cmap"])
    try:
        for position, region in enumerate(regions):
            pupil = np.ma.asarray(region.pupil)
            values = pupil.compressed()
            if not values.size or not np.isfinite(values).all():
                raise ValueError(f"AO region {region.index} has no finite pupil phase")
            axis = figure.add_subplot(grid[position // columns, position % columns])
            axis.set_title(region.index)
            image = axis.imshow(
                pupil,
                origin="lower",
                extent=(-1, 1, -1, 1),
                vmin=-phase_limit,
                vmax=phase_limit,
            )
            axis.set_aspect("equal")
            colorbar = figure.colorbar(image, ax=axis, fraction=0.1, pad=0.05)
            colorbar.set_ticks(ticks)
            colorbar.set_ticklabels(ticklabels, fontsize=30)
            colorbar.set_label("rad.", rotation=270, labelpad=15, fontsize=30)
            statistics.append(
                {
                    "region_index": int(region.index),
                    "phase_min_rad": float(values.min()),
                    "phase_max_rad": float(values.max()),
                    "phase_absmax_rad": float(np.max(np.abs(values))),
                    "clipped_fraction": float(np.mean(np.abs(values) > phase_limit)),
                    "valid_pupil_samples": int(values.size),
                }
            )
        figure.savefig(png, dpi=160)
        figure.savefig(svg)
    finally:
        plt.close(figure)

    return {
        "timepoint": int(timepoint),
        "depth_index": int(depth),
        "png": str(png),
        "svg": str(svg),
        "reference": str(reference_path),
        "reference_sha256": reference_hash,
        "style": {
            "columns": columns,
            "rows": rows,
            "figure_size_inches": [12, 5 * rows],
            "grid_hspace": 0.2,
            "grid_wspace": 0.2,
            "pupil_origin": "lower",
            "pupil_extent": [-1, 1, -1, 1],
            "pupil_aspect": "equal",
            "colormap": colormap,
            "phase_limit_rad": float(phase_limit),
            "normalization": "linear",
            "colorbar_fraction": 0.1,
            "colorbar_pad": 0.05,
            "colorbar_ticks_rad": [float(value) for value in ticks],
            "colorbar_ticklabels": ticklabels,
            "colorbar_tick_fontsize": 30,
            "colorbar_label": "rad.",
            "colorbar_label_rotation": 270,
            "colorbar_labelpad": 15,
            "colorbar_label_fontsize": 30,
            "png_dpi": 160,
        },
        "regions": statistics,
    }
