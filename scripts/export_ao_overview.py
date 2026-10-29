"""Export AO pupils/grid as transparent TIFF pages and per-region Zernike plots.

Run: uv run python scripts/export_ao_overview.py /path/to/acquisition-directory

The selected sjs_opm_scripts/ao_scripts/parse_ao_grid_results.py supplies the
WaveKit pupil calculation and coefficient plots directly. Each TIFF page is an
RGBA overlay on the level-zero fused maximum projection, with one page per AO
depth at the selected timepoint. The PNG overview includes the projection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib import font_manager
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from yaozarrs import open_group, v05

from opm_processing.dataio.acquisition import AcquisitionMetadata, inspect_acquisition
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)
from opm_processing.dataio.roi import (
    ROI_PREVIEW_SUFFIXES,
    projection_tile_footprints,
)
from opm_processing.export_rois import resolve_roi_source
from opm_processing.imageprocessing.coordinates import (
    stage_positions_to_image_coordinates,
    stage_z_level_indices,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


@dataclass
class Projection:
    """Projection pixels and laboratory coordinates from the current input."""

    path: Path
    group: Any
    multiscale: Any
    acquisition: AcquisitionMetadata
    footprints: tuple[dict[str, Any], ...]
    shape_yx: tuple[int, int]
    scale_yx_um: np.ndarray
    origin_yx_um: np.ndarray


@dataclass
class Region:
    """One planned AO region and its reference-reader result and pupil."""

    depth: int
    index: int
    result: Any
    stage_zxy: np.ndarray
    sample_depth_um: float | None
    pupil: np.ma.MaskedArray


def load_reference(path: Path) -> ModuleType:
    """Import the requested AO reader without running its interactive main.

    Parameters
    ----------
    path : pathlib.Path
        parse_ao_grid_results.py from sjs_opm_scripts.

    Returns
    -------
    types.ModuleType
        Reference implementation of AOOptimizationResult and pupil calculation.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"AO reader not found; pass --ao-reader: {path}")
    spec = importlib.util.spec_from_file_location("ao_overview_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import AO reader: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_projection(root: Path, path: Path | None) -> Projection:
    """Resolve a fused preview and read its calibrated level-zero coordinate grid.

    Parameters
    ----------
    root : pathlib.Path
        Current acquisition/output directory.
    path : pathlib.Path or None
        Explicit preview, or None for ordinary maximum-projection discovery.

    Returns
    -------
    Projection
        Metadata and pixel shape without reading the full-resolution image.
    """
    preview = resolve_roi_source(path if path is not None else root)
    group = open_group(preview)
    metadata = group.ome_metadata()
    if not isinstance(metadata, v05.Image):
        raise ValueError("AO overlays require a fused maximum-Z NGFF Image")
    multiscale = metadata.multiscales[0]
    if [axis.name for axis in multiscale.axes] != list("tczyx"):
        raise ValueError("Expected a TCZYX fused maximum projection")
    if any(axis.unit != "micrometer" for axis in multiscale.axes[-2:]):
        raise ValueError("Projection Y/X coordinates must be in micrometers")
    dataset = multiscale.datasets[0]
    shape = group[dataset.path].to_tensorstore().shape
    if shape[2] != 1:
        raise ValueError("Select a maximum-Z projection with singleton Z")
    suffix = next(s for s in ROI_PREVIEW_SUFFIXES if preview.name.endswith(s))
    state = ProcessingState.read(
        processing_state_path(preview.parent, preview.name.removesuffix(suffix))
    )
    return Projection(
        preview,
        group,
        multiscale,
        inspect_acquisition(state.source_path()),
        projection_tile_footprints(preview),
        tuple(map(int, shape[-2:])),
        np.asarray(dataset.scale_transform.scale[-2:], dtype=float),
        np.asarray(dataset.translation_transform.translation[-2:], dtype=float),
    )


def read_regions(
    root: Path, reader: ModuleType, args: argparse.Namespace
) -> list[Region]:
    """Read planned grid centers and full-precision final coefficient states.

    Parameters
    ----------
    root : pathlib.Path
        AO grid result directory beside the acquisition.
    reader : types.ModuleType
        Approved AO reference implementation.
    args : argparse.Namespace
        Time/depth selection, pupil state and wavelength.

    Returns
    -------
    list[Region]
        Depth-ordered regions with reference-normalized circular pupils.
    """
    regions = []
    for layer in sorted(root.glob(f"time_{args.timepoint}_z_*_ao_grid_results")):
        depth = int(layer.name.split("_z_", 1)[1].split("_", 1)[0])
        if args.depth_index is not None and depth != args.depth_index:
            continue
        planned = json.loads((layer / "ao_stage_positions.json").read_text("utf-8"))
        experiment = json.loads((layer / "exp_stage_positions.json").read_text("utf-8"))
        # Acquisition stores describe laboratory scan positions; the AO plan
        # records physical stage X, including depth/coverslip scan compensation.
        # Fit the recorded X-minus-lab_scan offset on the experiment's XY plane.
        references = [p for p in experiment if "lab_scan_um" in p]
        if references:
            xy = np.array([[p["x"], p["y"]] for p in references])
            center = xy.mean(axis=0)
            offsets = np.array([p["x"] - p["lab_scan_um"] for p in references])
            design = np.column_stack((np.ones(len(xy)), xy - center))
            fit = np.linalg.lstsq(design, offsets, rcond=None)[0]
        else:
            center, fit = np.zeros(2), np.zeros(3)
        for index, target in enumerate(planned):
            result = reader.AOOptimizationResult(
                layer / f"grid_pos_{index}" / "ao_results.zarr",
                args.timepoint,
                index,
            )
            xy_target = np.array([target["x"], target["y"]])
            scan_offset = np.r_[1.0, xy_target - center] @ fit
            stage = np.array([target["z"], target["x"] - scan_offset, target["y"]])
            pupil = result.get_pupil(
                coefficient_state=args.coefficient_state,
                samples=512,
                norm=True,
                wavelength=args.wavelength_nm / 1000,
                out="radians",
            )
            if not np.isfinite(pupil.compressed()).all():
                raise ValueError(f"Nonfinite pupil in {result.zarr_path}")
            regions.append(
                Region(
                    depth,
                    index,
                    result,
                    stage,
                    experiment[0].get("sample_depth_um"),
                    pupil,
                )
            )
    if not regions:
        raise ValueError(f"No AO regions for the requested time/depth in {root}")
    return sorted(regions, key=lambda r: (r.depth, r.index))


def stage_to_lab(points: np.ndarray, acquisition: AcquisitionMetadata) -> np.ndarray:
    """Apply the same stage-axis reflections and laboratory placement as fusion.

    Parameters
    ----------
    points : numpy.ndarray
        Stage ZXY points with laboratory scan values in the stage-X column.
    acquisition : AcquisitionMetadata
        Raw calibration used by the processed image.

    Returns
    -------
    numpy.ndarray
        Nominal laboratory ZYX coordinates in micrometers.
    """
    positions = np.asarray(acquisition.stage_positions_zxy, dtype=float)
    reflected = np.where(
        np.asarray(acquisition.stage_axis_flips_xyz[::-1]),
        positions.max(axis=0) - points,
        points,
    )
    return stage_positions_to_image_coordinates(
        reflected, reverse_z=not acquisition.stage_axis_flips_xyz[2]
    )


def registration_warp(
    projection: Projection,
    timepoint: int,
    depth: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Interpolate saved tile translations on one logical depth surface.

    Parameters
    ----------
    projection : Projection
        Preview footprints carrying the source tile's registered raw origin.
    timepoint : int
        Acquired time index.
    depth : int
        Repeated logical depth index at each stage-XY position.

    Returns
    -------
    callable
        Map nominal YX points to registered YX using piecewise-linear translation
        interpolation and nearest-tile extrapolation outside the stage-point hull.
    """
    acquisition = projection.acquisition
    positions = np.asarray(acquisition.stage_positions_zxy, dtype=float)
    nominal = stage_to_lab(positions, acquisition)
    levels = stage_z_level_indices(positions)
    selected = [
        tile
        for tile in projection.footprints
        if tile["time_index"] == timepoint and levels[tile["position_index"]] == depth
    ]
    if not selected:
        raise ValueError(
            f"No source tile placements for time {timepoint}, depth {depth}"
        )
    indices = [tile["position_index"] for tile in selected]
    coordinates = nominal[indices, 1:]
    registered = np.array(
        [tile.get("raw_origin_zyx_um", tile["origin_zyx_um"])[1:] for tile in selected]
    )
    offsets = registered - coordinates
    nearest = NearestNDInterpolator(coordinates, offsets)
    linear = (
        LinearNDInterpolator(coordinates, offsets)
        if len(coordinates) >= 3
        and np.linalg.matrix_rank(coordinates - coordinates[0]) == 2
        else None
    )

    def warp(points: np.ndarray) -> np.ndarray:
        """Apply the fitted translation field without changing source pixels."""
        correction = nearest(points) if linear is None else linear(points)
        missing = ~np.isfinite(correction).all(axis=1)
        if missing.any():
            correction[missing] = nearest(points[missing])
        return points + correction

    return warp


def draw_overlay(
    projection: Projection,
    regions: list[Region],
    args: argparse.Namespace,
    phase_limit: float,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Rasterize nearest-region grid boundaries and reference pupils with alpha.

    Parameters
    ----------
    projection : Projection
        Exact full-resolution destination coordinate grid.
    regions : list[Region]
        Correction regions for one depth.
    args : argparse.Namespace
        Pupil diameter and grid line thickness in physical units.
    phase_limit : float
        Shared symmetric pupil color limit in radians.

    Returns
    -------
    tuple
        Transparent RGBA image and region centers/coefficients for provenance.
    """
    height, width = projection.shape_yx
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    scale, origin = projection.scale_yx_um, projection.origin_yx_um
    nominal = stage_to_lab(
        np.stack([r.stage_zxy for r in regions]), projection.acquisition
    )[:, 1:]
    warp = registration_warp(projection, args.timepoint, regions[0].depth)
    axes = [np.unique(np.round(nominal[:, axis], 3)) for axis in (0, 1)]
    if len(axes[0]) * len(axes[1]) != len(regions):
        raise ValueError("AO targets must form a complete Cartesian lateral grid")
    line_width = max(1, round(args.line_width_um / min(scale)))
    # Nearest AO-region boundaries are midway between planned sample locations.
    # Extend outer cells to the image edge, as nearest-region assignment does.
    for axis in (0, 1):
        boundaries = (axes[axis][:-1] + axes[axis][1:]) / 2
        other = 1 - axis
        for boundary in boundaries:
            points = np.empty((256, 2))
            points[:, axis] = boundary
            points[:, other] = np.linspace(
                origin[other] - 100,
                origin[other] + projection.shape_yx[other] * scale[other] + 100,
                256,
            )
            pixels = (warp(points) - origin) / scale
            draw.line(
                [tuple(point[::-1]) for point in pixels], fill="white", width=line_width
            )
    diameter = np.maximum(2, np.rint(args.pupil_diameter_um / scale).astype(int))
    font = ImageFont.truetype(
        font_manager.findfont("DejaVu Sans"), max(12, int(diameter[1] / 6))
    )
    colormap = matplotlib.colormaps["RdBu_r"]
    normalize = Normalize(-phase_limit, phase_limit)
    records = []
    for region, center in zip(regions, warp(nominal), strict=True):
        y, x = (center - origin) / scale
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"AO region {region.index} center falls outside the projection"
            )
        # Reference imshow uses origin='lower'; TIFF rows run top to bottom.
        rgba = colormap(normalize(np.flipud(region.pupil)), bytes=True)
        rgba[np.flipud(np.ma.getmaskarray(region.pupil))] = 0
        pupil = Image.fromarray(rgba).resize(
            tuple(diameter[::-1]), Image.Resampling.LANCZOS
        )
        image.alpha_composite(
            pupil, (round(x - diameter[1] / 2), round(y - diameter[0] / 2))
        )
        draw.text(
            (x, y - diameter[0] / 2 - font.size / 4),
            f"R{region.index}",
            font=font,
            anchor="mb",
            fill="white",
            stroke_width=max(1, font.size // 25),
            stroke_fill="black",
        )
        records.append(
            {
                "region_index": region.index,
                "stage_zxy_lab_scan_um": region.stage_zxy.tolist(),
                "registered_center_yx_um": center.tolist(),
                "center_yx_pixels": [float(y), float(x)],
                "starting_coefficients": region.result.starting_coefficients.tolist(),
                "final_coefficients": region.result.final_coefficients.tolist(),
            }
        )
    return image, records


def overview_background(projection: Projection, args: argparse.Namespace) -> np.ndarray:
    """Read one overview scale while keeping the TIFF at native pixel resolution.

    Parameters
    ----------
    projection : Projection
        Calibrated maximum projection.
    args : argparse.Namespace
        Acquired timepoint, display channel and gamma.

    Returns
    -------
    numpy.ndarray
        Normalized grayscale display pixels, at most approximately 1800 per axis.
    """
    dataset = projection.multiscale.datasets[-1]
    for candidate in projection.multiscale.datasets:
        dataset = candidate
        if max(projection.group[candidate.path].to_tensorstore().shape[-2:]) <= 1800:
            break
    image = (
        projection.group[dataset.path]
        .to_tensorstore()[args.timepoint, args.channel, 0]
        .read()
        .result()
        .astype(float)
    )
    low, high = np.percentile(image[np.isfinite(image)], [1, 99.9])
    return (
        np.clip((image - low) / max(high - low, np.finfo(float).eps), 0, 1)
        ** args.gamma
    )


def main() -> None:
    """Generate the transparent AO overlay, overview, and regional coefficient plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--projection", type=Path)
    parser.add_argument("--ao-root", type=Path)
    parser.add_argument(
        "--ao-reader",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "sjs_opm_scripts/ao_scripts/parse_ao_grid_results.py",
    )
    parser.add_argument("--timepoint", type=int, default=0)
    parser.add_argument("--depth-index", type=int)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument(
        "--wavelength-nm",
        type=float,
        default=488.0,
        help="Reference reader default: 488 nm.",
    )
    parser.add_argument(
        "--coefficient-state", choices=("final", "change", "starting"), default="final"
    )
    parser.add_argument("--pupil-diameter-um", type=float, default=80.0)
    parser.add_argument("--line-width-um", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument(
        "--phase-limit", type=float, help="Shared symmetric pupil color limit, radians."
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name in ("wavelength_nm", "pupil_diameter_um", "line_width_um", "gamma"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.phase_limit is not None and (
        not np.isfinite(args.phase_limit) or args.phase_limit <= 0
    ):
        parser.error("--phase-limit must be positive and finite")
    root = args.root.expanduser().resolve()
    reader = load_reference(args.ao_reader)
    projection = read_projection(root, args.projection)
    if not 0 <= args.timepoint < projection.acquisition.shape[0]:
        parser.error("--timepoint is outside the acquired range")
    if not 0 <= args.channel < len(projection.acquisition.channels):
        parser.error("--channel is outside the acquired range")
    if args.ao_root is None:
        matches = sorted(projection.acquisition.path.parent.glob("*_ao_grid_results"))
        if len(matches) != 1:
            parser.error(
                "Pass --ao-root when the acquisition has zero or multiple AO result directories"
            )
        args.ao_root = matches[0]
    regions = read_regions(args.ao_root, reader, args)
    depths = sorted({r.depth for r in regions})
    phase_limit = args.phase_limit or max(
        float(np.max(np.abs(r.pupil.compressed()))) for r in regions
    )
    phase_limit = max(phase_limit, np.finfo(float).eps)
    output = (
        (args.output or projection.path.parent / "ao_overview").expanduser().resolve()
    )
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output is not empty; choose another --output or pass --overwrite: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    plots = output / "zernike_plots"
    plots.mkdir(exist_ok=True)
    background = overview_background(projection, args)
    columns = min(2, len(depths))
    rows = (len(depths) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(8 * columns, 8 * rows), squeeze=False
    )
    manifest = {
        "projection": str(projection.path),
        "shape_yx": projection.shape_yx,
        "origin_yx_um": projection.origin_yx_um.tolist(),
        "scale_yx_um": projection.scale_yx_um.tolist(),
        "timepoint": args.timepoint,
        "channel": args.channel,
        "coefficient_state": args.coefficient_state,
        "wavelength_nm": args.wavelength_nm,
        "phase_limit_rad": phase_limit,
        "wavekit_normalization": "RMS, piston-free modes 1..32",
        "reference": str(args.ao_reader.resolve()),
        "reference_sha256": hashlib.sha256(args.ao_reader.read_bytes()).hexdigest(),
        "placement": "planned AO centers; recorded lab-scan compensation; interpolated registered tile translations",
        "pages": [],
    }
    with tifffile.TiffWriter(output / "ao_grid_pupils.tiff", bigtiff=True) as writer:
        for page, depth in enumerate(depths):
            selected = [r for r in regions if r.depth == depth]
            overlay, records = draw_overlay(projection, selected, args, phase_limit)
            description = {
                "axes": "YXS",
                "depth_index": depth,
                "sample_depth_um": selected[0].sample_depth_um,
                "timepoint": args.timepoint,
                "origin_yx_um": projection.origin_yx_um.tolist(),
                "scale_yx_um": projection.scale_yx_um.tolist(),
                "phase_units": "rad",
            }
            writer.write(
                np.asarray(overlay),
                photometric="rgb",
                extrasamples="unassalpha",
                compression="deflate",
                tile=(256, 256),
                maxworkers=2,
                resolution=tuple((10000 / projection.scale_yx_um)[::-1]),
                resolutionunit="CENTIMETER",
                metadata=description,
            )
            thumbnail = overlay.resize(background.shape[::-1], Image.Resampling.LANCZOS)
            ax = axes.flat[page]
            ax.imshow(background, cmap="gray", vmin=0, vmax=1)
            ax.imshow(thumbnail)
            ax.set_title(
                f"Time {args.timepoint}, depth {depth} ({selected[0].sample_depth_um} um)"
            )
            ax.set_axis_off()
            manifest["pages"].append({"page": page, **description, "regions": records})
            for region in selected:
                fig, coefficient_axes = region.result.plot_coefficients(show=False)
                coefficient_axes.set_title(
                    f"Time {args.timepoint}, depth {depth}, region {region.index}\nStarting and end-of-iteration Zernike coefficients"
                )
                fig.tight_layout()
                fig.savefig(
                    plots
                    / f"t{args.timepoint:03d}_z{depth:03d}_region_{region.index:03d}.png",
                    dpi=180,
                    bbox_inches="tight",
                )
                plt.close(fig)
            print(
                f"Wrote transparent page {page}: depth {depth}, {len(selected)} pupils",
                flush=True,
            )
            del overlay
    for ax in list(axes.flat)[len(depths) :]:
        ax.set_visible(False)
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(-phase_limit, phase_limit), cmap="RdBu_r"),
        ax=list(axes.flat),
        fraction=0.02,
        pad=0.02,
    )
    colorbar.set_label(
        f"{args.coefficient_state.capitalize()} pupil phase (rad), wavelength {args.wavelength_nm:g} nm"
    )
    figure.savefig(output / "ao_overview.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    (output / "ao_overview.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Created {output / 'ao_grid_pupils.tiff'} and {len(regions)} Zernike plots",
        flush=True,
    )


if __name__ == "__main__":
    main()
