"""Place SJS-styled pupils over calibrated tissue projections."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes


def add_scale_bar(
    axis: Axes,
    extent_yx_um: tuple[float, float, float, float],
    length_um: float | None = None,
    *,
    line_scale: float = 1,
) -> dict[str, Any]:
    """Draw an unlabeled scale bar in the tissue's micrometer coordinates."""
    y0, y1, x0, x1 = extent_yx_um
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError("Tissue extent must have positive width and height")
    if length_um is None:
        target = 0.18 * width
        magnitude = 10 ** math.floor(math.log10(target))
        length_um = max(
            value * magnitude
            for value in (0.1, 0.2, 0.5, 1, 2, 5)
            if value * magnitude <= target
        )
    if not np.isfinite(length_um) or not 0 < length_um <= 0.8 * width:
        raise ValueError("Scale bar must be positive and fit within the tissue image")
    right = x1 - 0.06 * width
    left = right - length_um
    y = y1 - 0.07 * height
    bar_pixels = abs(
        axis.transData.transform((right, y))[0] - axis.transData.transform((left, y))[0]
    )
    white_width = min(3 * line_scale, bar_pixels / 25 * 72 / axis.figure.dpi)
    stroke = [
        path_effects.withStroke(linewidth=white_width * 5 / 3, foreground="black")
    ]
    (line,) = axis.plot(
        (left, right),
        (y, y),
        color="white",
        linewidth=white_width,
        solid_capstyle="butt",
        path_effects=stroke,
    )
    line.set_gid("tissue-scale-bar")
    label = f"{length_um:g} µm"
    return {
        "length_um": float(length_um),
        "label": label,
        "label_visible": False,
        "endpoints_yx_um": [
            [round(float(y), 3), round(float(left), 3)],
            [round(float(y), 3), round(float(right), 3)],
        ],
    }


def write_tissue_overlay(
    regions: Sequence[Any],
    geometry: Any,
    output: Path,
    *,
    timepoint: int,
    pupil_diameter_um: float = 112.5,
    scale_bar_um: float | None = None,
    planned_grid: bool = False,
    full_resolution: bool = False,
    output_shape_yx: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Export one text-free tissue image per depth with a colorbar per pupil.

    Pupil colors follow the SJS publication reference. Local colorbars have no
    text. Tissue coordinates come from NGFF image transforms; pupil centers
    identify assigned source-tile groups rather than AO measurement sites.
    ``planned_grid`` instead divides the full acquisition canvas into the
    requested number of equal cells and centers each correction in its cell.
    ``full_resolution`` preserves one source image sample per output PNG pixel.
    ``output_shape_yx`` supplies explicit viewing-copy dimensions.
    """
    if not np.isfinite(pupil_diameter_um) or pupil_diameter_um <= 0:
        raise ValueError("Pupil display diameter must be positive and finite")
    layer_images = getattr(geometry, "images_by_depth", {})
    y0, y1, x0, x1 = geometry.extent_yx_um
    extent = (x0, x1, y1, y0)
    depths = sorted({region.depth for region in regions})
    if not depths:
        raise ValueError("No AO results selected for the tissue overlay")
    norm = Normalize(-2 * np.pi, 2 * np.pi)
    cmap = plt.rcParams["image.cmap"]
    grid_line_alpha = 0.3
    radius = pupil_diameter_um / 2
    pages = []
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    width, height = x1 - x0, y1 - y0
    for depth in depths:
        if layer_images:
            if depth not in layer_images:
                raise ValueError(f"Missing local Z projection for layer {depth}")
            image = np.asarray(layer_images[depth])
        else:
            image = np.asarray(geometry.image)
        if full_resolution and tuple(image.shape) != tuple(geometry.shape_yx):
            raise ValueError("Full-resolution overlays require the native image array")
        if image.ndim == 3 and image.shape[-1] in (3, 4) and image.dtype == np.uint8:
            background = image
            low, high = 0, 255
        else:
            limits = getattr(geometry, "display_limits_by_depth", {}).get(depth)
            if limits is None:
                finite = image[np.isfinite(image)]
                if not finite.size:
                    raise ValueError(
                        f"Layer {depth} projection has no finite intensities"
                    )
                low, high = np.percentile(finite, (1, 99.8))
            else:
                low, high = limits
            background = np.clip(
                (image - low) / max(high - low, np.finfo(float).eps), 0, 1
            )
        canvas_shape = tuple(output_shape_yx or image.shape)
        exact_canvas = full_resolution or output_shape_yx is not None
        line_scale = canvas_shape[1] / (8 * 160 * 0.96) if exact_canvas else 1
        if exact_canvas:
            figure = plt.figure(
                figsize=(canvas_shape[1] / 160, canvas_shape[0] / 160), dpi=160
            )
            axis = figure.add_axes((0, 0, 1, 1))
        else:
            figure, axis = plt.subplots(figsize=(8, 8 * height / width + 0.6))
            figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.94)
        basename = f"ao_tissue_overlay_t{timepoint:03d}_z{depth:03d}"
        png, svg = output / f"{basename}.png", output / f"{basename}.svg"
        try:
            axis.imshow(
                background,
                cmap="gray",
                vmin=0,
                vmax=1,
                origin="upper",
                extent=extent,
                interpolation="none" if exact_canvas else "auto",
            )
            boundaries = None
            if planned_grid:
                centers = geometry.planned_centers_by_depth_index.get(depth, {})
                boundaries = geometry.planned_grid_by_depth_index.get(depth)
                if not centers or boundaries is None:
                    raise ValueError(f"No intended AO grid at depth {depth}")
                for y in boundaries["horizontal_y_um"]:
                    axis.plot(
                        (x0, x1),
                        (y, y),
                        color="white",
                        linewidth=0.75 * line_scale,
                        alpha=grid_line_alpha,
                    )
                for x in boundaries["vertical_x_um"]:
                    axis.plot(
                        (x, x),
                        (y0, y1),
                        color="white",
                        linewidth=0.75 * line_scale,
                        alpha=grid_line_alpha,
                    )
            else:
                centers = geometry.centers_by_depth_index.get(depth, {})
            placements = []
            for region in (item for item in regions if item.depth == depth):
                if region.index not in centers:
                    continue
                y, x = centers[region.index]
                if not (y0 <= y <= y1 and x0 <= x <= x1):
                    raise ValueError(
                        f"AO region {region.index} display center is outside the tissue projection"
                    )
                # Tissue Y increases down the page. Reverse the physical Y extent
                # so the SJS origin='lower' pupil retains its visible orientation.
                pupil_image = axis.imshow(
                    region.pupil,
                    cmap=cmap,
                    norm=norm,
                    origin="lower",
                    extent=(x - radius, x + radius, y + radius, y - radius),
                )
                bar_width = 0.06 * pupil_diameter_um
                gap = 0.08 * pupil_diameter_um
                bar_left = x + radius + gap
                if bar_left + bar_width > x1:
                    bar_left = x - radius - gap - bar_width
                bar_bottom = y - radius
                colorbar_axis = axis.inset_axes(
                    (
                        (bar_left - x0) / width,
                        (y1 - y - radius) / height,
                        bar_width / width,
                        pupil_diameter_um / height,
                    )
                )
                colorbar = figure.colorbar(pupil_image, cax=colorbar_axis)
                colorbar.outline.set_linewidth(0.8 * line_scale)
                colorbar.set_ticks([])
                colorbar.set_ticks([], minor=True)
                colorbar.set_label("")
                colorbar.ax.yaxis.get_offset_text().set_visible(False)
                colorbar.ax.xaxis.get_offset_text().set_visible(False)
                placements.append(
                    {
                        "region_index": region.index,
                        "center_yx_um": [round(float(y), 3), round(float(x), 3)],
                        "colorbar_bounds_yx_um": [
                            round(float(bar_bottom), 3),
                            round(float(y + radius), 3),
                            round(float(bar_left), 3),
                            round(float(bar_left + bar_width), 3),
                        ],
                    }
                )
            if not placements:
                raise ValueError(f"No AO display positions found at depth {depth}")
            axis.set_xlim(x0, x1)
            axis.set_ylim(y1, y0)
            axis.set_aspect("auto" if exact_canvas else "equal")
            axis.set_axis_off()
            sample_depth = geometry.depth_offsets.get(depth)
            bar = add_scale_bar(
                axis, geometry.extent_yx_um, scale_bar_um, line_scale=line_scale
            )
            if exact_canvas:
                figure.savefig(png, dpi=160)
                figure.savefig(svg, dpi=160)
            else:
                figure.savefig(png, dpi=160, bbox_inches="tight", pad_inches=0.1)
                figure.savefig(svg, bbox_inches="tight", pad_inches=0.1)
            pages.append(
                {
                    "depth_index": depth,
                    "png": str(png),
                    "svg": str(svg),
                    "pixel_shape_yx": list(canvas_shape) if exact_canvas else None,
                    "sample_depth_um": sample_depth,
                    "projection_roi": getattr(
                        geometry, "layer_projection_rois_by_depth", {}
                    ).get(depth),
                    "tissue_display_limits": [float(low), float(high)],
                    "planned_grid": boundaries,
                    "scale_bar": bar,
                    "regions": placements,
                }
            )
        finally:
            plt.close(figure)
        print(f"Saved tissue overlay at depth {depth}: {png}", flush=True)
    return {
        "placement": (
            "centers of intended equal AO grid cells across the full acquisition canvas"
            if planned_grid
            else "representative centers of historically assigned source-tile groups"
        ),
        "pupil_display_diameter_um": pupil_diameter_um,
        "grid_line_alpha": grid_line_alpha,
        "full_resolution": bool(full_resolution),
        "native_pixel_shape_yx": list(geometry.shape_yx) if full_resolution else None,
        "pupil_phase_limit_rad": float(2 * np.pi),
        "pupil_normalization": "linear",
        "colorbars": "one beside each pupil; no ticks or text",
        "extent_yx_um": [round(float(value), 3) for value in geometry.extent_yx_um],
        "tissue_display_percentiles": [1, 99.8],
        "background": "local acquisition-layer Z projection"
        if layer_images
        else "full-volume maximum Z projection",
        "geometry": geometry.provenance,
        "pages": pages,
    }
