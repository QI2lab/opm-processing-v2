"""Finish native AO BigTIFFs from completed projection caches and viewing copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

if __package__:
    from scripts.ao_tissue_bigtiff import write_tissue_bigtiffs
    from scripts.ao_tissue_composite import make_composite_layers
    from scripts.ao_wavekit_phase import BASIS_VERSION, phase_from_modal_coeff
    from scripts.export_ao_overview import Region, load_reference
else:
    from ao_tissue_bigtiff import write_tissue_bigtiffs
    from ao_tissue_composite import make_composite_layers
    from ao_wavekit_phase import BASIS_VERSION, phase_from_modal_coeff
    from export_ao_overview import Region, load_reference


def load_cached_geometry(output: Path, manifest: dict[str, Any]) -> Any:
    """Reconstruct the exact renderer inputs without reading fused-volume pixels."""
    view = manifest["tissue_overlay"]
    provenance = view["geometry"]
    composite = provenance["tissue_composite"]
    channels = [int(channel) for channel in composite["channels"]]
    pages = {int(page["depth_index"]): page for page in view["pages"]}
    if len(pages) != len(view["pages"]) or not pages:
        raise ValueError("Expected distinct completed viewing-copy layers")
    colors = {int(c): tuple(color) for c, color in composite["colors_rgb8"].items()}
    limits = {
        int(c): {int(d): tuple(value) for d, value in layers.items()}
        for c, layers in composite["display_limits_by_channel_depth"].items()
    }
    images = {}
    common_canvas = None
    for channel in channels:
        directory = output / (
            "projection_cache"
            if channel == 1
            else f"projection_cache_channel_{channel}"
        )
        cache_path = directory / "layer_projection_cache.json"
        cache = json.loads(cache_path.read_bytes())
        context = cache["context"]
        if (
            cache.get("phase") != "complete"
            or not cache["completed_xy_blocks"]
            or not all(cache["completed_xy_blocks"])
            or context["channel"] != channel
            or context["timepoint"] != manifest["timepoint"]
            or set(context["depths"]) != set(pages)
        ):
            raise ValueError(f"Channel {channel} has an incomplete or mismatched cache")
        canvas = (
            tuple(context["shape_yx"]),
            tuple(context["canvas_origin_yx_um"]),
            tuple(context["canvas_scale_yx_um"]),
        )
        if common_canvas is not None and canvas != common_canvas:
            raise ValueError("Composite channel caches disagree on native calibration")
        common_canvas = canvas
        images[channel] = {}
        for depth, page in pages.items():
            roi = page["projection_roi"]
            if (
                Path(roi["projection_cache_by_channel"][str(channel)]).resolve()
                != cache_path.resolve()
                or roi["projection_cache_fingerprint_by_channel"][str(channel)]
                != cache["fingerprint"]
                or roi["mip_sha256_by_channel"][str(channel)]
                != cache["mip_sha256_by_depth"][str(depth)]
            ):
                raise ValueError(
                    "Viewing-copy provenance differs from its native cache"
                )
            path = (directory / cache["images"][str(depth)]["path"]).resolve()
            if path.parent != directory.resolve():
                raise ValueError(
                    "Native cache image points outside its cache directory"
                )
            image = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                image.shape != canvas[0]
                or image.dtype != np.uint16
                or path.stat().st_size != image.offset + image.nbytes
            ):
                raise ValueError("Native projection cache layout or size is invalid")
            images[channel][depth] = image
    if common_canvas is None:
        raise ValueError("No completed composite channels selected")
    native_images = make_composite_layers(images, colors, limits)
    return SimpleNamespace(
        shape_yx=common_canvas[0],
        origin_yx_um=common_canvas[1],
        scale_yx_um=common_canvas[2],
        extent_yx_um=tuple(view["extent_yx_um"]),
        planned_centers_by_depth_index={
            depth: {
                int(region["region_index"]): tuple(region["center_yx_um"])
                for region in page["regions"]
            }
            for depth, page in pages.items()
        },
        planned_grid_by_depth_index={
            depth: page["planned_grid"] for depth, page in pages.items()
        },
        images_by_depth=native_images,
        display_limits_by_depth=dict.fromkeys(native_images, (0, 255)),
        layer_projection_rois_by_depth={
            depth: page["projection_roi"] for depth, page in pages.items()
        },
        provenance=provenance,
    )


def load_saved_regions(manifest: dict[str, Any]) -> list[Region]:
    """Recreate pupils with their recorded basis and unchanged source coefficients."""
    path = Path(manifest["reader"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["reader_sha256"]:
        raise ValueError("The approved pupil reader has changed since preview export")
    basis_version = manifest.get("pupil_reconstruction", {}).get("basis_version")
    if basis_version and basis_version != BASIS_VERSION:
        raise ValueError(f"Unknown saved pupil reconstruction basis: {basis_version}")
    reader = load_reference(path)
    regions = []
    for record in manifest["regions"]:
        result = reader.AOOptimizationResult(
            Path(record["source"]),
            int(manifest["timepoint"]),
            int(record["region_index"]),
        )
        if not np.array_equal(
            result.final_coefficients, record["final_coefficients"]
        ) or not np.array_equal(
            result.starting_coefficients, record["starting_coefficients"]
        ):
            raise ValueError("Saved AO coefficients changed since the viewing copies")
        options = {
            "samples": 512,
            "norm": True,
            "wavelength": float(manifest["wavelength_nm"]) / 1000,
            "out": "radians",
        }
        if basis_version == BASIS_VERSION:
            states = {
                "starting": np.asarray(result.starting_coefficients),
                "final": np.asarray(result.final_coefficients),
                "change": np.asarray(result.final_coefficients)
                - result.starting_coefficients,
            }
            try:
                coefficients = states[manifest["coefficient_state"]]
            except KeyError as error:
                raise ValueError(
                    "coefficient_state must be 'starting', 'final', or 'change'"
                ) from error
            pupil = phase_from_modal_coeff(
                coefficients,
                mode_indices=np.arange(1, coefficients.size + 1),
                **options,
            )
        else:
            pupil = result.get_pupil(
                coefficient_state=manifest["coefficient_state"], **options
            )
        pupil = np.ma.asarray(pupil)
        if not pupil.count() or not np.isfinite(pupil.compressed()).all():
            raise ValueError("Saved AO pupil phase must contain finite samples")
        regions.append(
            Region(
                int(record["depth_index"]), int(record["region_index"]), result, pupil
            )
        )
    return regions


def finish_native(
    output: Path, *, depth_workers: int = 3, overwrite_native: bool = False
) -> dict[str, Any]:
    """Write native layers while preserving the completed PNG/SVG viewing copies.

    The original exporter must be stopped before invoking this continuation.
    """
    output = output.expanduser().resolve()
    manifest_path = output / "ao_publication.json"
    manifest_before = manifest_path.read_bytes()
    manifest = json.loads(manifest_before)
    if manifest.get("native_tissue_overlays", {}).get("status") != "rendering":
        raise ValueError(
            "Expected an interrupted native export with completed viewing copies"
        )
    for page in manifest["tissue_overlay"]["pages"]:
        for key in ("png", "svg"):
            path = Path(page[key]).resolve()
            if path.parent != output / "previews" or not path.is_file():
                raise ValueError(
                    "A completed viewing copy is missing or outside this export"
                )
    native_directory = output / "native"
    if (
        native_directory.exists()
        and any(native_directory.iterdir())
        and not overwrite_native
    ):
        raise FileExistsError(
            "Use --overwrite-native to replace interrupted native TIFFs"
        )
    geometry = load_cached_geometry(output, manifest)
    regions = load_saved_regions(manifest)
    expected = {
        (depth, index)
        for depth, centers in geometry.planned_centers_by_depth_index.items()
        for index in centers
    }
    actual = [(region.depth, region.index) for region in regions]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Saved pupils do not match the completed intended grid")
    view = manifest["tissue_overlay"]
    scale_bars = {float(page["scale_bar"]["length_um"]) for page in view["pages"]}
    if len(scale_bars) != 1:
        raise ValueError("Native continuation requires one consistent scale-bar length")
    native = write_tissue_bigtiffs(
        regions,
        geometry,
        native_directory,
        timepoint=int(manifest["timepoint"]),
        pupil_diameter_um=float(view["pupil_display_diameter_um"]),
        scale_bar_um=scale_bars.pop(),
        depth_workers=depth_workers,
    )
    if manifest_path.read_bytes() != manifest_before:
        raise ValueError(
            "AO manifest changed during native rendering; refusing to replace it"
        )
    manifest["native_tissue_overlays"] = native
    descriptor, temporary = tempfile.mkstemp(
        prefix=".ao-native-", suffix=".json", dir=output
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return native


def main() -> None:
    """Continue only native TIFF rendering from completed read-only projection caches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--depth-workers", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--overwrite-native", action="store_true")
    args = parser.parse_args()
    finish_native(
        args.output,
        depth_workers=args.depth_workers,
        overwrite_native=args.overwrite_native,
    )


if __name__ == "__main__":
    main()
