"""Export total and accumulated Grid-AO pupils with one publication colorbar."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from yaozarrs import open_group

from opm_processing.dataio.ao_results import (
    AOOptimizationResult,
    ao_wavelength_nm,
    read_ao_layers,
    select_roi_ao,
)
from opm_processing.imageprocessing.ao import BASIS_VERSION, phase_from_modal_coeff

if TYPE_CHECKING:
    from opm_processing.dataio.acquisition import AcquisitionMetadata
    from opm_processing.dataio.roi import PhysicalRoi


def export_roi_ao(
    acquisition: AcquisitionMetadata,
    roi: PhysicalRoi,
    output: Path,
    *,
    ao_root: Path | None = None,
    wavelength_nm: float | None = None,
    ao_layers: dict | None = None,
    colorbar_path: Path | None = None,
) -> dict[str, Any]:
    """Save total and original-start-relative phases with saved-plan provenance.

    Parameters
    ----------
    acquisition : AcquisitionMetadata
        Original acquisition, retaining its physical stage positions and settings.
    roi : PhysicalRoi
        Requested bounds and source-tile identities; processing halos are excluded.
    output : pathlib.Path
        ROI folder for the pupil, colorbar and AO provenance files.
    ao_root : pathlib.Path or None
        Explicit saved AO grid directory, otherwise discover beside acquisition.
    wavelength_nm : float or None
        Optional display wavelength override; otherwise use recorded AO illumination.
    ao_layers : dict or None
        Saved AO plans shared by the current ROI batch.
    colorbar_path : pathlib.Path or None
        Fixed publication colorbar already rendered for this ROI batch.

    Returns
    -------
    dict
        Saved provenance. Missing AO is explicitly recorded as unavailable.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    if ao_layers is None:
        ao_layers = read_ao_layers(acquisition, roi.source_path, ao_root)
    records = (
        select_roi_ao(acquisition, roi, ao_root, ao_layers=ao_layers)
        if ao_layers
        else []
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "exported" if records else "unavailable",
        "source_path": str(acquisition.path),
        "roi_bounds_yx_um": list(roi.bounds_yx_um),
        "coefficient_state": "full final state, including inherited starting correction",
        "relative_coefficient_state": "final minus the corresponding grid position's starting coefficients at the first saved AO time and depth",
        "quantity": "requested modal correction target, not measured mirror shape",
        "basis_version": BASIS_VERSION,
        "normalization": "linear",
        "phase_limit_rad": float(2 * np.pi),
        "corrections": records,
    }
    if not records:
        manifest["reason"] = (
            "No saved AO grid results found beside the raw acquisition or preview"
        )
    else:
        configuration = (
            open_group(acquisition.path)
            .attrs.get("opm_v2", {})
            .get("configuration", {})
        )
        colormap = str(mpl.rcParams["image.cmap"])
        manifest["colormap"] = colormap
        manifest["startup_flat_file"] = configuration.get("AOMirror", {}).get(
            "wfc_flat_path"
        )
        reference_time, reference_depth = min(ao_layers)
        reference_layer = ao_layers[reference_time, reference_depth][0]
        references = {}
        # SJS publication plots use Matplotlib's image cmap and shared +/-2pi.
        norm = Normalize(-2 * np.pi, 2 * np.pi)
        for record in records:
            result = AOOptimizationResult(
                Path(record["result_path"]),
                record["ao_time_index"],
                record["region_index"],
            )
            wavelength = ao_wavelength_nm(result, configuration, wavelength_nm)
            coefficients = result.final_coefficients
            index = record["region_index"]
            if index not in references:
                reference_path = (
                    reference_layer / f"grid_pos_{index}" / "ao_results.zarr"
                )
                references[index] = (
                    result
                    if reference_path == result.zarr_path
                    else AOOptimizationResult(reference_path, reference_time, index)
                )
            reference = references[index]
            relative_coefficients = coefficients - reference.starting_coefficients
            pupil = phase_from_modal_coeff(
                coefficients, samples=512, wavelength=wavelength / 1000, out="radians"
            )
            relative_pupil = phase_from_modal_coeff(
                relative_coefficients,
                samples=512,
                wavelength=wavelength / 1000,
                out="radians",
            )
            suffix = f"t{record['time_index']:03d}_z{record['depth_index']:03d}"
            pupil_name, colorbar_name = (
                f"ao_pupil_{suffix}.png",
                f"ao_colorbar_{suffix}.png",
            )
            relative_name = f"ao_pupil_grid_contribution_{suffix}.png"
            comparison_name = f"ao_pupil_comparison_{suffix}.png"
            for name, phase in ((pupil_name, pupil), (relative_name, relative_pupil)):
                plt.imsave(
                    output / name,
                    phase,
                    origin="lower",
                    cmap=colormap,
                    vmin=norm.vmin,
                    vmax=norm.vmax,
                )
            figure, axes = plt.subplots(1, 2, figsize=(12, 5))
            try:
                for axis, phase, title in zip(
                    axes,
                    (pupil, relative_pupil),
                    (
                        "Total correction\nRelative to startup reference",
                        "Accumulated Grid-AO contribution\nRelative to original acquisition start",
                    ),
                    strict=True,
                ):
                    axis.imshow(
                        phase,
                        origin="lower",
                        cmap=colormap,
                        norm=norm,
                        interpolation="none",
                    )
                    axis.set_axis_off()
                    axis.set_title(title, fontsize=17)
                figure.savefig(
                    output / comparison_name,
                    dpi=160,
                    bbox_inches="tight",
                    transparent=True,
                )
            finally:
                plt.close(figure)
            if colorbar_path is None:
                figure = plt.figure(figsize=(1.4, 5))
                try:
                    axis = figure.add_axes((0.1, 0.05, 0.3, 0.9))
                    bar = ColorbarBase(axis, cmap=mpl.colormaps[colormap], norm=norm)
                    bar.set_ticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
                    bar.set_ticklabels(
                        [r"$-2\pi$", r"$-\pi$", "$0$", r"$\pi$", r"$2\pi$"], fontsize=30
                    )
                    bar.set_label("rad.", rotation=270, labelpad=15, fontsize=30)
                    figure.savefig(
                        output / colorbar_name,
                        dpi=160,
                        bbox_inches="tight",
                        transparent=True,
                    )
                finally:
                    plt.close(figure)
                colorbar_path = output / colorbar_name
            else:
                shutil.copyfile(colorbar_path, output / colorbar_name)
            record.update(
                pupil=pupil_name,
                relative_pupil=relative_name,
                comparison=comparison_name,
                colorbar=colorbar_name,
                relative_reference_path=str(reference.zarr_path),
                relative_reference_time_index=reference_time,
                relative_reference_depth_index=reference_depth,
                original_starting_coefficients=reference.starting_coefficients.tolist(),
                relative_coefficients=relative_coefficients.tolist(),
                mirror_reference=result.metadata.get("initialization", {}).get(
                    "reference_state"
                ),
                wavelength_nm=wavelength,
                wavelength_source="explicit override"
                if wavelength_nm is not None
                else "recorded AO illumination",
                starting_coefficients=result.starting_coefficients.tolist(),
                final_coefficients=coefficients.tolist(),
                phase_absmax_rad=float(np.abs(pupil.compressed()).max()),
                clipped_fraction=float(np.mean(np.abs(pupil.compressed()) > 2 * np.pi)),
            )
    (output / "ao_corrections.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
