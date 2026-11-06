"""Render depth-dependent AO changes against one acquisition starting correction."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

if __package__:
    from scripts.ao_publication_style import write_publication_pupil_grid
    from scripts.ao_saved_viewing_background import read_saved_viewing_background
    from scripts.ao_tissue_bigtiff import NativeTissueRaster
    from scripts.ao_tissue_overlay import write_tissue_overlay
    from scripts.ao_tissue_rephase import rephase_tissue_bigtiff
    from scripts.ao_wavekit_phase import (
        BASIS_VERSION,
        WAVEKIT_MODE_NM,
        phase_from_modal_coeff,
    )
    from scripts.export_ao_overview import Region
    from scripts.finish_ao_native import load_cached_geometry, load_saved_regions
else:
    from ao_publication_style import write_publication_pupil_grid
    from ao_saved_viewing_background import read_saved_viewing_background
    from ao_tissue_bigtiff import NativeTissueRaster
    from ao_tissue_overlay import write_tissue_overlay
    from ao_tissue_rephase import rephase_tissue_bigtiff
    from ao_wavekit_phase import BASIS_VERSION, WAVEKIT_MODE_NM, phase_from_modal_coeff
    from export_ao_overview import Region
    from finish_ao_native import load_cached_geometry, load_saved_regions


def relative_regions(
    manifest: dict[str, Any],
    saved: list[Region],
    reader: Any,
    *,
    include_starting: bool = False,
) -> tuple[list[Region], dict[str, Any], list[dict[str, Any]]]:
    """Sum signed layer updates and subtract the same original starting vector."""
    if manifest["coefficient_state"] != "final":
        raise ValueError("Relative rendering requires original full final coefficients")
    records = {
        (int(record["depth_index"]), int(record["region_index"])): record
        for record in manifest["regions"]
    }
    if len(records) != len(manifest["regions"]) or not records:
        raise ValueError("Original AO records must have unique depth/position pairs")
    reference_depth = min(depth for depth, _ in records)
    reference_indices = sorted(
        index for depth, index in records if depth == reference_depth
    )
    reference = np.asarray(
        records[reference_depth, reference_indices[0]]["starting_coefficients"],
        dtype=float,
    )
    if reference.ndim != 1 or not np.isfinite(reference).all():
        raise ValueError("The common acquisition starting vector must be finite")
    for index in reference_indices:
        if not np.array_equal(
            reference, records[reference_depth, index]["starting_coefficients"]
        ):
            raise ValueError(
                "There is no single common acquisition starting correction"
            )
    expected = {(depth, index) for depth, _ in records for index in reference_indices}
    actual = [(int(region.depth), int(region.index)) for region in saved]
    if (
        set(records) != expected
        or len(set(actual)) != len(actual)
        or set(actual) != expected
    ):
        raise ValueError("Every depth must contain the same saved correction positions")
    wavelength = float(manifest["wavelength_nm"]) / 1000
    regions, plotted_records = [], []
    previous_final = {index: reference.copy() for index in reference_indices}
    cumulative = {index: np.zeros_like(reference) for index in reference_indices}
    for region in sorted(saved, key=lambda r: (r.depth, r.index)):
        record = records[region.depth, region.index]
        final = np.asarray(record["final_coefficients"], dtype=float)
        if final.shape != reference.shape or not np.isfinite(final).all():
            raise ValueError("Final corrections must match the common modal basis")
        starting = np.asarray(record["starting_coefficients"], dtype=float)
        if not np.array_equal(starting, previous_final[region.index]):
            raise ValueError(
                "Each layer must inherit the corresponding previous layer's final correction"
            )
        layer_delta = final - starting
        cumulative[region.index] += layer_delta
        change = final - reference
        if not np.allclose(cumulative[region.index], change, rtol=0, atol=1e-12):
            raise ValueError(
                "Cumulative signed layer changes disagree with final minus original start"
            )
        previous_final[region.index] = final.copy()
        coefficients = final if include_starting else change
        pupil = np.ma.asarray(
            reader.phase_from_modal_coeff(
                coefficients,
                mode_indices=np.arange(1, coefficients.size + 1),
                samples=512,
                norm=True,
                wavelength=wavelength,
                out="radians",
            )
        )
        if not pupil.count() or not np.isfinite(pupil.compressed()).all():
            raise ValueError("The reconstructed relative pupil has no finite phase")
        if pupil.shape != region.pupil.shape or not np.array_equal(
            np.ma.getmaskarray(pupil), np.ma.getmaskarray(region.pupil)
        ):
            raise ValueError("Relative pupils must preserve the original aperture mask")
        regions.append(Region(region.depth, region.index, region.result, pupil))
        plotted_records.append(
            {
                **record,
                "layer_delta_coefficients": layer_delta.tolist(),
                "cumulative_coefficients": cumulative[region.index].tolist(),
                "plotted_coefficients": coefficients.tolist(),
                "relative_rms_opd_nm": float(1000 * np.linalg.norm(change)),
                "plotted_rms_opd_nm": float(1000 * np.linalg.norm(coefficients)),
                "plotted_phase_rms_rad": float(np.std(pupil.compressed())),
                "plotted_phase_min_rad": float(pupil.min()),
                "plotted_phase_max_rad": float(pupil.max()),
                "relative_phase_rms_rad": float(np.std(pupil.compressed())),
                "relative_phase_min_rad": float(pupil.min()),
                "relative_phase_max_rad": float(pupil.max()),
            }
        )
        if include_starting:
            for key in (
                "relative_phase_rms_rad",
                "relative_phase_min_rad",
                "relative_phase_max_rad",
            ):
                plotted_records[-1].pop(key)
    phase_reference = {
        "kind": "common_acquisition_start",
        "formula": "final_coefficients[depth,position] - common_original_starting_coefficients",
        "accumulation": "sum of signed (final minus starting) vectors through the selected depth; every inherited start verified against the corresponding previous final",
        "reference_depth_index": reference_depth,
        "reference_region_indices": reference_indices,
        "reference_coefficients": reference.tolist(),
        "coefficient_unit": "micrometer, matching the approved reader's 0.488 wavelength convention",
        "quantity": "cumulative requested modal correction relative to the original acquisition start",
        "phase_sign": "requested correction target OPD; WaveKit's correction inverse produces approximately the opposite mirror-induced OPD",
        "measurement": "modeled requested correction, not a measurement of physical mirror shape or residual sample aberration",
        "basis": "one-based piston-free WaveKit modes with RMS-normalized Zernike polynomials",
        "wavelength_nm": float(manifest["wavelength_nm"]),
        "phase_limit_rad": float(2 * np.pi),
        "display_normalization": "linear, shared across every pupil and depth",
    }
    if include_starting:
        phase_reference.update(
            kind="configured_mirror_reference",
            formula="original_starting_coefficients + sum(signed layer updates) = full final_coefficients[depth,position]",
            reference_coefficients=np.zeros_like(reference).tolist(),
            original_starting_coefficients=reference.tolist(),
            starting_correction_included=True,
            quantity="full requested modal correction state including the inherited acquisition starting correction",
            reference_shape="zero modal coefficients about the configured non-flat voltage reference; physical reference shape is not reconstructed",
        )
    return regions, phase_reference, plotted_records


def export_relative(
    source: Path, output: Path, *, workers: int = 3, include_starting: bool = False
) -> dict[str, Any]:
    """Reuse clean viewing backgrounds and native tissue tiles; replace pupil phases."""
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError("Relative figures must be saved separately from their source")
    if workers not in (1, 2, 3):
        raise ValueError("Relative export supports one to three native workers")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Select an empty output directory: {output}")
    source_path = source / "ao_publication.json"
    source_bytes = source_path.read_bytes()
    original = json.loads(source_bytes)
    if not original.get("native_tissue_overlays", {}).get("full_resolution"):
        raise ValueError("Relative export requires completed original native overlays")
    saved = load_saved_regions(original)
    # The historical reader remains the unchanged data loader. New phase maps
    # use the modal basis independently checked against offline WaveKit.
    phase_reader = SimpleNamespace(phase_from_modal_coeff=phase_from_modal_coeff)
    regions, phase_reference, plotted_records = relative_regions(
        original, saved, phase_reader, include_starting=include_starting
    )
    phase_path = Path(__file__).with_name("ao_wavekit_phase.py")
    phase_reference["reconstruction"] = {
        "basis_version": BASIS_VERSION,
        "signed_nm_modes": [list(pair) for pair in WAVEKIT_MODE_NM],
        "source": str(phase_path),
        "source_sha256": hashlib.sha256(phase_path.read_bytes()).hexdigest(),
        "historical_reader_role": "saved coefficient loading and original aperture verification only",
    }
    geometry = load_cached_geometry(source, original)
    geometry.provenance = {**geometry.provenance, "pupil_reference": phase_reference}
    depths = sorted({region.depth for region in regions})
    old_pages = {
        int(page["depth_index"]): page for page in original["tissue_overlay"]["pages"]
    }
    native_pages = {
        int(page["depth_index"]): page
        for page in original["native_tissue_overlays"]["pages"]
    }
    if set(depths) != set(old_pages) or set(depths) != set(native_pages):
        raise ValueError("Saved figures and source pupils must contain the same depths")
    backgrounds, background_records = {}, {}
    for depth in depths:
        backgrounds[depth], background_records[depth] = read_saved_viewing_background(
            Path(old_pages[depth]["svg"]), tuple(old_pages[depth]["pixel_shape_yx"])
        )
    shapes = {tuple(image.shape[:2]) for image in backgrounds.values()}
    if len(shapes) != 1:
        raise ValueError("Viewing-copy layers must share one canvas")
    preview_geometry = SimpleNamespace(
        **{
            **vars(geometry),
            "image": backgrounds[depths[0]],
            "images_by_depth": backgrounds,
            "depth_offsets": {
                depth: old_pages[depth]["sample_depth_um"] for depth in depths
            },
            "provenance": {
                **geometry.provenance,
                "reused_viewing_backgrounds": background_records,
            },
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = copy.deepcopy(original)
    manifest["source_figure_manifest"] = {
        "path": str(source_path),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    manifest["source_coefficient_state"] = original["coefficient_state"]
    manifest["coefficient_state"] = (
        "final" if include_starting else "relative_to_acquisition_start"
    )
    if include_starting:
        manifest["pupil_reconstruction"] = {
            **phase_reference["reconstruction"],
            "quantity": phase_reference["quantity"],
            "coefficient_state": "final",
            "normalization": "RMS",
            "samples": 512,
            "wavelength_nm": float(original["wavelength_nm"]),
        }
    manifest["pupil_reference"] = phase_reference
    manifest["regions"] = plotted_records
    manifest["grids"] = []
    reference_path = Path(original["grids"][0]["reference"])
    for depth in depths:
        manifest["grids"].append(
            write_publication_pupil_grid(
                [region for region in regions if region.depth == depth],
                output / "pupil_grids",
                timepoint=original["timepoint"],
                depth=depth,
                reference_path=reference_path,
            )
        )
    manifest["tissue_overlay"] = write_tissue_overlay(
        regions,
        preview_geometry,
        output / "previews",
        timepoint=original["timepoint"],
        pupil_diameter_um=original["tissue_overlay"]["pupil_display_diameter_um"],
        scale_bar_um=old_pages[depths[0]]["scale_bar"]["length_um"],
        planned_grid=True,
        output_shape_yx=shapes.pop(),
    )
    manifest["native_tissue_overlays"] = {"status": "rendering"}
    manifest_path = output / "ao_publication.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rasters = {}
    # Build colormaps serially; only independent TIFF I/O runs in the pool.
    for depth in depths:
        old = native_pages[depth]
        with matplotlib.rc_context({"image.cmap": old["pupil_colormap"]}):
            raster = NativeTissueRaster(
                geometry,
                geometry.images_by_depth[depth],
                [region for region in regions if region.depth == depth],
                depth=depth,
                timepoint=original["timepoint"],
                pupil_diameter_um=old["pupil_display_diameter_um"],
                scale_bar_um=old["scale_bar"]["length_um"],
            )
            # Identical apertures are required for lossless background reuse.
            old_raster = NativeTissueRaster(
                geometry,
                geometry.images_by_depth[depth],
                [region for region in saved if region.depth == depth],
                depth=depth,
                timepoint=original["timepoint"],
                pupil_diameter_um=old["pupil_display_diameter_um"],
                scale_bar_um=old["scale_bar"]["length_um"],
            )
        for before, after in zip(old_raster._pupils, raster._pupils, strict=True):
            if (
                before["index"] != after["index"]
                or before["bounds"] != after["bounds"]
                or not np.array_equal(before["rgba"][..., 3], after["rgba"][..., 3])
            ):
                raise ValueError(
                    "Relative native figures must preserve pupil masks and placement"
                )
        rasters[depth] = raster

    def native_depth(depth: int) -> dict[str, Any]:
        old = native_pages[depth]
        destination = output / "native" / Path(old["tif"]).name
        result = rephase_tissue_bigtiff(
            Path(old["tif"]),
            destination,
            rasters[depth],
            phase_provenance=phase_reference,
        )
        return {**old, **result}

    with ThreadPoolExecutor(max_workers=min(workers, len(depths))) as executor:
        native = list(executor.map(native_depth, depths))
    if source_path.read_bytes() != source_bytes:
        raise ValueError("The original figure manifest changed during relative export")
    manifest["native_tissue_overlays"] = {
        **original["native_tissue_overlays"],
        "pages": native,
        "depth_workers": min(workers, len(depths)),
        "pupil_reference": phase_reference,
        "tissue_pixels": "original native RGB pixels retained exactly outside pupil masks",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved relative AO figures: {output}", flush=True)
    return manifest


def main() -> None:
    """Make separately saved relative AO figures from completed native overlays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Completed original ao_images directory"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--include-starting",
        action="store_true",
        help="Plot the full final requested modal state, including correction already present at acquisition start.",
    )
    args = parser.parse_args()
    suffix = "_full_corrected" if args.include_starting else "_relative"
    output = args.output or args.source.with_name(args.source.name + suffix)
    export_relative(
        args.source,
        output,
        workers=args.workers,
        include_starting=args.include_starting,
    )


if __name__ == "__main__":
    main()
