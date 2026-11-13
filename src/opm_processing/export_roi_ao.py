"""Export the dominant saved AO pupil and a separate publication colorbar per ROI."""

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
    """Save full final requested correction phases with saved-plan provenance.

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

    records = select_roi_ao(acquisition, roi, ao_root, ao_layers=ao_layers)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "exported" if records else "unavailable",
        "source_path": str(acquisition.path),
        "roi_bounds_yx_um": list(roi.bounds_yx_um),
        "coefficient_state": "full final state, including inherited starting correction",
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
            pupil = phase_from_modal_coeff(
                coefficients, samples=512, wavelength=wavelength / 1000, out="radians"
            )
            suffix = f"t{record['time_index']:03d}_z{record['depth_index']:03d}"
            pupil_name, colorbar_name = (
                f"ao_pupil_{suffix}.png",
                f"ao_colorbar_{suffix}.png",
            )
            plt.imsave(
                output / pupil_name,
                pupil,
                origin="lower",
                cmap=colormap,
                vmin=norm.vmin,
                vmax=norm.vmax,
            )
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
                colorbar=colorbar_name,
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
