"""Export saved AO pupils with SJS formatting and optional tissue overlays.

Run: uv run python scripts/export_ao_overview.py /path/to/acquisition-directory
The portable local reader loads saved coefficients; the SDK-validated WaveKit
basis reconstructs their pupils. Publication formatting follows the SJS
two-column layout and linear plus/minus 2-pi colors. Explicit reader and
plot-reference paths retain support for external SJS files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from scripts.ao_publication_style import write_publication_pupil_grid
    from scripts.ao_wavekit_phase import BASIS_VERSION, phase_from_modal_coeff
else:
    from ao_publication_style import write_publication_pupil_grid
    from ao_wavekit_phase import BASIS_VERSION, phase_from_modal_coeff

if TYPE_CHECKING:
    from types import ModuleType

_LAYER = re.compile(r"^time_(\d+)(?:_z_(\d+))?_ao_grid(?:_results)?$")
_POSITION = re.compile(r"^grid_pos_(\d+)$")


@dataclass
class Region:
    """One saved AO result and its directly reconstructed pupil."""

    depth: int
    index: int
    result: Any
    pupil: np.ma.MaskedArray


def load_reference(path: Path) -> ModuleType:
    """Load the saved-result reader without running its command-line entrypoint."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"AO reader not found: {path}")
    spec = importlib.util.spec_from_file_location("ao_publication_reader", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import AO reader: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_ao_root(root: Path) -> Path:
    """Find saved AO results directly, without a fused image or processing journal."""
    root = root.expanduser().resolve()
    if (
        _LAYER.fullmatch(root.name)
        or root.name.endswith("_ao_grid_results")
        or any(root.glob("grid_pos_*/ao_results.zarr"))
    ):
        return root
    directory = root.parent if root.name.endswith(".zarr") else root
    candidates = sorted(p for p in directory.glob("*_ao_grid_results") if p.is_dir())
    if len(candidates) != 1:
        raise ValueError(
            "Select the AO result directory when its parent has zero or multiple AO grids"
        )
    return candidates[0]


def read_regions(
    root: Path, reader: ModuleType, args: argparse.Namespace
) -> list[Region]:
    """Read saved coefficients and reconstruct pupils in the validated SDK basis."""
    if args.coefficient_state not in {"starting", "final", "change"}:
        raise ValueError("Unknown saved AO coefficient state")
    candidates = [root] if _LAYER.fullmatch(root.name) else list(root.iterdir())
    layers = []
    if (
        not _LAYER.fullmatch(root.name)
        and args.timepoint == 0
        and args.depth_index in (None, 0)
        and any(root.glob("grid_pos_*/ao_results.zarr"))
    ):
        layers.append((0, root))
    for layer in candidates:
        match = _LAYER.fullmatch(layer.name)
        if not match or not layer.is_dir() or int(match[1]) != args.timepoint:
            continue
        depth = int(match[2] or 0)
        if args.depth_index is None or depth == args.depth_index:
            layers.append((depth, layer))
    regions = []
    for depth, layer in sorted(layers):
        positions = []
        for position in layer.iterdir():
            match = _POSITION.fullmatch(position.name)
            store = position / "ao_results.zarr"
            if match and store.is_dir():
                positions.append((int(match[1]), store))
        for index, store in sorted(positions):
            result = reader.AOOptimizationResult(store, args.timepoint, index)
            starting = np.asarray(result.starting_coefficients, dtype=float)
            final = np.asarray(result.final_coefficients, dtype=float)
            coefficients = {
                "starting": starting,
                "final": final,
                "change": final - starting,
            }[args.coefficient_state]
            pupil = phase_from_modal_coeff(
                coefficients,
                mode_indices=np.arange(1, coefficients.size + 1),
                samples=512,
                norm=True,
                wavelength=args.wavelength_nm / 1000,
                out="radians",
            )
            pupil = np.ma.asarray(pupil)
            if not pupil.count() or not np.isfinite(pupil.compressed()).all():
                raise ValueError(f"No finite pupil phase in {store}")
            regions.append(Region(depth, index, result, pupil))
    if not regions:
        raise ValueError("No saved AO results match the selected time and depth")
    return regions


def main() -> None:
    """Export publication pupils, optional tissue overlays, and native coefficients."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ao-root", type=Path)
    parser.add_argument(
        "--ao-reader",
        type=Path,
        default=Path(__file__).with_name("ao_saved_results.py"),
    )
    parser.add_argument("--plot-reference", type=Path)
    parser.add_argument("--timepoint", type=int, default=0)
    parser.add_argument("--depth-index", type=int)
    parser.add_argument(
        "--tissue-overlay",
        action="store_true",
        help="Also place pupils on the calibrated fused tissue projection with a scale bar.",
    )
    parser.add_argument(
        "--projection", type=Path, help="Explicit fused maximum-Z tissue projection."
    )
    parser.add_argument(
        "--planned-grid",
        action="store_true",
        help="Draw the intended equal AO grid over the full acquisition and center each correction in its cell.",
    )
    parser.add_argument(
        "--full-resolution",
        action="store_true",
        help="Read the native projection level and write one output PNG pixel per source pixel.",
    )
    parser.add_argument(
        "--local-z-projection",
        action="store_true",
        help="Calculate native local Z projections from acquisition-defined layer ROIs.",
    )
    parser.add_argument(
        "--native-bigtiff",
        action="store_true",
        help="Write tiled native BigTIFF overlays plus PNG/SVG viewing copies using a disk-backed projection cache.",
    )
    parser.add_argument("--preview-max-edge", type=int, default=6000)
    parser.add_argument(
        "--channel", type=int, default=0, help="Tissue display channel."
    )
    parser.add_argument(
        "--composite-channels",
        type=int,
        nargs=3,
        metavar=("MAGENTA", "CYAN", "ORANGE"),
        help="Render these three tissue channels in magenta, cyan, and orange with native BigTIFFs and PNG previews.",
    )
    parser.add_argument(
        "--pupil-diameter-um",
        type=float,
        default=112.5,
        help="Display diameter of pupil symbols on tissue.",
    )
    parser.add_argument(
        "--scale-bar-um",
        type=float,
        help="Tissue scale bar length; automatic when omitted.",
    )
    parser.add_argument("--wavelength-nm", type=float, default=488.0)
    parser.add_argument(
        "--coefficient-state", choices=("final", "change", "starting"), default="final"
    )
    parser.add_argument(
        "--coefficients",
        action="store_true",
        help="Also save coefficient plots with the selected reader (local default: starting/final/change bars).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.composite_channels is not None:
        if len(set(args.composite_channels)) != 3 or min(args.composite_channels) < 0:
            parser.error(
                "Composite channels must be three distinct nonnegative indices"
            )
        args.native_bigtiff = True
    if args.native_bigtiff:
        if not args.planned_grid:
            parser.error(
                "Native BigTIFF exports require --planned-grid, including --composite-channels"
            )
        args.local_z_projection = True
    if args.local_z_projection:
        args.full_resolution = True
    if args.preview_max_edge < 1:
        parser.error("--preview-max-edge must be positive")
    if not np.isfinite(args.wavelength_nm) or args.wavelength_nm <= 0:
        parser.error("--wavelength-nm must be positive and finite")
    if not np.isfinite(args.pupil_diameter_um) or args.pupil_diameter_um <= 0:
        parser.error("--pupil-diameter-um must be positive and finite")
    if args.scale_bar_um is not None and (
        not np.isfinite(args.scale_bar_um) or args.scale_bar_um <= 0
    ):
        parser.error("--scale-bar-um must be positive and finite")
    root = args.root.expanduser().resolve()
    ao_root = resolve_ao_root(args.ao_root or root)
    plot_reference = (
        args.plot_reference or Path(__file__).with_name("ao_publication_style.py")
    ).resolve()
    if not plot_reference.is_file():
        parser.error(f"Publication plotting reference not found: {plot_reference}")
    reader = load_reference(args.ao_reader)
    regions = read_regions(ao_root, reader, args)
    directory = root.parent if root.name.endswith(".zarr") else root
    output = (args.output or directory / "ao_publication").expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output is not empty; select a new directory or use --overwrite: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    phase_source = Path(phase_from_modal_coeff.__code__.co_filename).resolve()
    manifest = {
        "ao_results": str(ao_root),
        "timepoint": args.timepoint,
        "coefficient_state": args.coefficient_state,
        "wavelength_nm": args.wavelength_nm,
        "reader": str(args.ao_reader.resolve()),
        "reader_sha256": hashlib.sha256(args.ao_reader.read_bytes()).hexdigest(),
        "pupil_reconstruction": {
            "basis_version": BASIS_VERSION,
            "source": str(phase_source),
            "source_sha256": hashlib.sha256(phase_source.read_bytes()).hexdigest(),
            "quantity": "requested modal correction target from the selected saved coefficient state",
            "coefficient_state": args.coefficient_state,
            "coefficient_unit": "micrometer OPD",
            "phase_unit": "radians",
            "wavelength_nm": args.wavelength_nm,
            "normalization": "RMS",
            "mode_indices": "one-based piston-free WaveKit indices",
            "samples": 512,
            "phase_sign": "requested correction target; physical mirror-induced OPD is approximately opposite",
        },
        "grids": [],
        "regions": [],
    }
    for depth in sorted({region.depth for region in regions}):
        selected = [region for region in regions if region.depth == depth]
        manifest["grids"].append(
            write_publication_pupil_grid(
                selected,
                output / "pupil_grids" if args.native_bigtiff else output,
                timepoint=args.timepoint,
                depth=depth,
                reference_path=plot_reference,
            )
        )
        for region in selected:
            manifest["regions"].append(
                {
                    "depth_index": depth,
                    "region_index": region.index,
                    "source": str(region.result.zarr_path),
                    "starting_coefficients": region.result.starting_coefficients.tolist(),
                    "final_coefficients": region.result.final_coefficients.tolist(),
                }
            )
            if args.coefficients:
                figure, _ = region.result.plot_coefficients(
                    save_path=output
                    / "zernike_plots"
                    / f"t{args.timepoint:03d}_z{depth:03d}_region_{region.index:03d}.png",
                    show=False,
                )
                plt.close(figure)
        print(
            f"Wrote SJS publication pupil grid: time {args.timepoint}, depth {depth}",
            flush=True,
        )
    if (
        args.tissue_overlay
        or args.projection is not None
        or args.planned_grid
        or args.full_resolution
    ):
        if __package__:
            from scripts.ao_tissue_geometry import read_tissue_geometry
            from scripts.ao_tissue_overlay import write_tissue_overlay
        else:
            from ao_tissue_geometry import read_tissue_geometry
            from ao_tissue_overlay import write_tissue_overlay
        geometry = read_tissue_geometry(
            root,
            args.projection,
            args.timepoint,
            args.composite_channels[0] if args.composite_channels else args.channel,
            regions,
            planned_grid=args.planned_grid,
            full_resolution=args.full_resolution,
            load_image=not args.local_z_projection,
        )
        if args.local_z_projection:
            if __package__:
                from scripts.ao_layer_projections import read_layer_projections
            else:
                from ao_layer_projections import read_layer_projections
            if args.composite_channels:
                if __package__:
                    from scripts.ao_tissue_bigtiff import native_display_limits
                    from scripts.ao_tissue_composite import make_composite_layers
                else:
                    from ao_tissue_bigtiff import native_display_limits
                    from ao_tissue_composite import make_composite_layers
                images_by_channel, limits_by_channel, rois_by_channel = {}, {}, {}
                # Keep the original channel-one cache reusable after adding a composite.
                channel_order = sorted(
                    args.composite_channels, key=lambda c: (c != 1, c)
                )
                for channel in channel_order:
                    print(
                        f"Preparing composite channel {channel} local projections",
                        flush=True,
                    )
                    channel_images, channel_rois = read_layer_projections(
                        geometry,
                        timepoint=args.timepoint,
                        channel=channel,
                        depths=sorted({region.depth for region in regions}),
                        cache_dir=output
                        / (
                            "projection_cache"
                            if channel == 1
                            else f"projection_cache_channel_{channel}"
                        ),
                    )
                    images_by_channel[channel] = channel_images
                    rois_by_channel[channel] = channel_rois
                    limits_by_channel[channel] = {}
                    for depth, image in sorted(channel_images.items()):
                        limits_by_channel[channel][depth] = native_display_limits(image)
                        print(
                            f"Native channel {channel}, layer {depth} display limits: {limits_by_channel[channel][depth]}",
                            flush=True,
                        )
                colors = dict(
                    zip(
                        args.composite_channels,
                        ((255, 0, 255), (0, 255, 255), (255, 165, 0)),
                        strict=True,
                    )
                )
                images = make_composite_layers(
                    images_by_channel, colors, limits_by_channel
                )
                layer_rois = {}
                for depth in images:
                    reference_roi = dict(rois_by_channel[channel_order[0]][depth])
                    for key in (
                        "channel",
                        "mip_sha256",
                        "projection_cache",
                        "projection_cache_fingerprint",
                        "resumed_xy_blocks",
                        "shared_volume_read_bytes",
                    ):
                        reference_roi.pop(key, None)
                    layer_rois[depth] = {
                        **reference_roi,
                        "channels": args.composite_channels,
                        "mip_sha256_by_channel": {
                            str(channel): rois_by_channel[channel][depth]["mip_sha256"]
                            for channel in args.composite_channels
                        },
                        "projection_cache_by_channel": {
                            str(channel): rois_by_channel[channel][depth][
                                "projection_cache"
                            ]
                            for channel in args.composite_channels
                        },
                        "projection_cache_fingerprint_by_channel": {
                            str(channel): rois_by_channel[channel][depth][
                                "projection_cache_fingerprint"
                            ]
                            for channel in args.composite_channels
                        },
                        "resumed_xy_blocks_by_channel": {
                            str(channel): rois_by_channel[channel][depth][
                                "resumed_xy_blocks"
                            ]
                            for channel in args.composite_channels
                        },
                        "shared_volume_read_bytes_by_channel": {
                            str(channel): rois_by_channel[channel][depth][
                                "shared_volume_read_bytes"
                            ]
                            for channel in args.composite_channels
                        },
                    }
                geometry = replace(
                    geometry,
                    provenance={
                        **geometry.provenance,
                        "channel": None,
                        "tissue_composite": {
                            "channels": args.composite_channels,
                            "colors_rgb8": colors,
                            "color_names": dict(
                                zip(
                                    args.composite_channels,
                                    ("magenta", "cyan", "orange"),
                                    strict=True,
                                )
                            ),
                            "blend": "independent linear channel normalization, additive RGB, clipped to [0, 255]",
                            "display_percentiles": [1, 99.8],
                            "display_limits_by_channel_depth": limits_by_channel,
                        },
                    },
                )
            else:
                images, layer_rois = read_layer_projections(
                    geometry,
                    timepoint=args.timepoint,
                    channel=args.channel,
                    depths=sorted({region.depth for region in regions}),
                    cache_dir=output / "projection_cache"
                    if args.native_bigtiff
                    else None,
                )
            geometry = replace(
                geometry,
                image=images[min(images)],
                images_by_depth=images,
                layer_projection_rois_by_depth=layer_rois,
                provenance={**geometry.provenance, "layer_projection_rois": layer_rois},
            )
        if args.native_bigtiff:
            if __package__:
                from scripts.ao_tissue_bigtiff import (
                    native_display_limits,
                    write_tissue_bigtiffs,
                )
                from scripts.ao_tissue_preview import make_layer_previews
            else:
                from ao_tissue_bigtiff import (
                    native_display_limits,
                    write_tissue_bigtiffs,
                )
                from ao_tissue_preview import make_layer_previews
            limits = dict.fromkeys(geometry.images_by_depth, (0, 255))
            if not args.composite_channels:
                for depth, image in sorted(geometry.images_by_depth.items()):
                    limits[depth] = native_display_limits(image)
                    print(
                        f"Native layer {depth} display limits: {limits[depth]}",
                        flush=True,
                    )
            geometry = replace(geometry, display_limits_by_depth=limits)
            previews, factor = make_layer_previews(
                geometry.images_by_depth, limits, max_edge=args.preview_max_edge
            )
            preview_shape = next(iter(previews.values())).shape[:2]
            preview_geometry = replace(
                geometry,
                image=previews[min(previews)],
                images_by_depth=previews,
                display_limits_by_depth=dict.fromkeys(previews, (0, 255)),
                provenance={
                    **geometry.provenance,
                    "viewing_copy": True,
                    "viewing_copy_reduction": "area mean of native display-normalized RGB pixels",
                    "viewing_copy_downsample_factor": factor,
                    "viewing_copy_shape_yx": list(preview_shape),
                    "native_display_limits_by_depth": limits,
                },
            )
            manifest["tissue_overlay"] = write_tissue_overlay(
                regions,
                preview_geometry,
                output / "previews",
                timepoint=args.timepoint,
                pupil_diameter_um=args.pupil_diameter_um,
                scale_bar_um=args.scale_bar_um,
                planned_grid=args.planned_grid,
                full_resolution=False,
                output_shape_yx=preview_shape,
            )
            manifest["native_tissue_overlays"] = {"status": "rendering"}
            (output / "ao_publication.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            manifest["native_tissue_overlays"] = write_tissue_bigtiffs(
                regions,
                geometry,
                output / "native",
                timepoint=args.timepoint,
                pupil_diameter_um=args.pupil_diameter_um,
                scale_bar_um=manifest["tissue_overlay"]["pages"][0]["scale_bar"][
                    "length_um"
                ],
            )
        else:
            manifest["tissue_overlay"] = write_tissue_overlay(
                regions,
                geometry,
                output,
                timepoint=args.timepoint,
                pupil_diameter_um=args.pupil_diameter_um,
                scale_bar_um=args.scale_bar_um,
                planned_grid=args.planned_grid,
                full_resolution=args.full_resolution,
            )
        for page in manifest["tissue_overlay"]["pages"]:
            print(f"Wrote calibrated tissue overlay: {page['png']}", flush=True)
    (output / "ao_publication.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
