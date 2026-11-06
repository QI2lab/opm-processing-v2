"""Render depth-dependent AO changes against one acquisition starting correction."""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

if __package__:
    from scripts.export_ao_overview import Region
else:
    from export_ao_overview import Region


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
