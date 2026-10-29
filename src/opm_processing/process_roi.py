"""Deconvolve and fuse a physical ROI selected on a napari max projection."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from opm_processing.dataio.acquisition import (
    AcquisitionMetadata,
    acquisition_stem,
    inspect_acquisition,
)
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)
from opm_processing.dataio.roi import PhysicalRoi
from opm_processing.imageprocessing.maxtilefusion import (
    regenerate_fused_max_projection,
)
from opm_processing.imageprocessing.opmtools import deskew_shape_estimator
from opm_processing.imageprocessing.tilefusion import (
    DEFAULT_REGISTRATION_DOWNSAMPLE,
    DEFAULT_REGISTRATION_SSIM_WINDOW,
    TileFusion,
    _infer_registration_limits,
)
from opm_processing.process import process_skewed


def roi_with_registration_context(
    roi: PhysicalRoi,
    acquisition: AcquisitionMetadata,
    z_downsample_level: int,
) -> PhysicalRoi:
    """Retain registration search and SSIM context around the selected rectangle.

    Parameters
    ----------
    roi : PhysicalRoi
        Original selection with saved full-acquisition tile placements.
    acquisition : AcquisitionMetadata
        Raw pixel calibration and camera height.
    z_downsample_level : int
        Deskewed Z reduction used by the subsequent ROI fusion.

    Returns
    -------
    PhysicalRoi
        Expanded processing rectangle. Context follows the same geometry-based
        search limits as fusion, including depth-dependent footprint shifts;
        nearly coincident tiles never request their entire overlap as a halo.
    """
    pixel = float(acquisition.pixel_size_um)
    spacing = np.array((pixel * z_downsample_level, pixel, pixel))
    deskew_shape, *_ = deskew_shape_estimator(
        acquisition.shape[-3:],
        theta=acquisition.angle_deg,
        distance=acquisition.scan_axis_step_um,
        pixel_size=pixel,
        downsample_factor=z_downsample_level,
        crop_after_deskew=False,
    )
    z_count = int(deskew_shape[0])
    support = (
        (DEFAULT_REGISTRATION_SSIM_WINDOW // 2 + 1)
        * np.asarray(DEFAULT_REGISTRATION_DOWNSAMPLE[1:])
        * pixel
    )
    bounds = np.asarray(roi.bounds_yx_um, dtype=float)
    margins = np.zeros(2)
    footprints = roi.tile_footprints
    times = sorted({tile["time_index"] for tile in footprints})
    for _ in range(32):
        expanded = bounds + np.array((-margins[0], margins[0], -margins[1], margins[1]))
        needed = np.zeros(2)
        for time in times:
            group = [tile for tile in footprints if tile["time_index"] == time]
            if len(group) < 2 or all(
                np.array_equal(tile["origin_zyx_um"], group[0]["origin_zyx_um"])
                for tile in group
            ):
                continue
            origins, shapes = [], []
            for tile in group:
                y0, y1, x0, x1 = tile["bounds_yx_um"]
                y0, y1 = max(y0, expanded[0]), min(y1, expanded[1])
                x0, x1 = max(x0, expanded[2]), min(x1, expanded[3])
                if y1 <= y0 or x1 <= x0:
                    continue
                origins.append((tile["origin_zyx_um"][0], y0, x0))
                shapes.append(
                    (
                        z_count,
                        math.ceil((y1 - y0) / pixel),
                        math.ceil((x1 - x0) / pixel),
                    )
                )
            limits = _infer_registration_limits(
                origins,
                shapes,
                spacing,
                DEFAULT_REGISTRATION_DOWNSAMPLE,
                acquisition.angle_deg,
            )
            for limit in limits.values():
                needed = np.maximum(needed, np.asarray(limit[1:]) * pixel + support)
        needed = np.ceil(needed / pixel - 1e-9) * pixel
        if np.all(needed <= margins):
            break
        margins = np.maximum(margins, needed)
    else:
        raise RuntimeError(
            "ROI registration context did not reach a stable search envelope"
        )
    print(f"ROI registration context: Y={margins[0]:.3f} um, X={margins[1]:.3f} um")
    return replace(
        roi,
        bounds_yx_um=tuple(
            bounds + np.array((-margins[0], margins[0], -margins[1], margins[1]))
        ),
    )


def process_roi(
    root_path: Path,
    roi_json: Path,
    *,
    registration_channel: int,
    deconvolve: bool = False,
    flatfield_correction: bool = False,
    save_float32: bool = False,
    z_downsample_level: int = 2,
    crop_after_deskew: bool = False,
    decon_crop_scan: int | None = None,
    decon_gpu_id: int = 0,
    decon_verbose: int = 0,
    decon_psf_paths: list[Path] | None = None,
    resume: bool = True,
    output: Path | None = None,
) -> Path:
    """Process a napari-selected world-space ROI and fuse its cropped tiles.

    Parameters
    ----------
    root_path : pathlib.Path
        Raw acquisition path or directory containing its processing state.
    roi_json : pathlib.Path
        Saved physical ROI selection.
    deconvolve : bool
        Deconvolve the selected raw regions before deskewing.
    flatfield_correction : bool
        Estimate or reuse illumination fields for camera-calibrated images.
    save_float32 : bool
        Save float32 intensities; False clips final output to uint16.
    z_downsample_level : int
        Integer reduction factor along deskewed Z.
    crop_after_deskew : bool
        General deskew crop option, which must be False for physical ROI crops.
    decon_crop_scan : int or None
        Retained scan planes per deconvolution chunk; None selects automatically.
    decon_gpu_id : int
        Zero-based CUDA device used for deconvolution.
    decon_verbose : int
        Deconvolution diagnostics: zero (default) disables them; one enables them.
    decon_psf_paths : list[pathlib.Path] or None
        Channel-ordered PSF files; None generates theoretical PSFs.
    registration_channel : int
        Channel used to align cropped tiles during fusion.
    resume : bool
        Continue completed tile and channel checkpoints; False overwrites output.
    output : pathlib.Path or None
        Output directory; None uses <acquisition>_roi beside the input.


    Returns
    -------
    pathlib.Path
        Final TCZYX OME-Zarr image; cropped tiles and checkpoints are retained.
    """
    requested_path = Path(root_path).expanduser().resolve()
    state_paths = tuple(requested_path.glob("*.processing.json"))
    if state_paths:
        if len(state_paths) != 1:
            raise ValueError(
                "Pass the raw acquisition path and ROI JSON explicitly when "
                "the input directory contains multiple processing-state files."
            )
        state = ProcessingState.read(state_paths[0])
        acquisition = inspect_acquisition(state.source_path())
        context_dir = requested_path
    else:
        acquisition = inspect_acquisition(requested_path)
        context_dir = acquisition.path.parent
    if acquisition.is_2d:
        raise ValueError(
            "ROI processing currently targets skewed 3D acquisitions; use process "
            "for native 2D data"
        )
    if not 0 <= registration_channel < len(acquisition.channels):
        raise ValueError("Registration channel is outside the acquired channel range")
    if crop_after_deskew:
        raise ValueError(
            "ROI processing already writes exact physical ROI crops; "
            "--crop-after-deskew is not applicable"
        )
    stem = acquisition_stem(acquisition.path)
    resolved_roi_json = (
        context_dir / f"{stem}_roi.json"
        if roi_json is None
        else Path(roi_json).expanduser().resolve()
    )
    roi = PhysicalRoi.read(resolved_roi_json)
    output_dir = (
        Path(output).expanduser().resolve()
        if output is not None
        else context_dir / f"{stem}_roi"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    processing_roi = roi_with_registration_context(roi, acquisition, z_downsample_level)
    state_path = processing_state_path(output_dir, stem)
    if resume and deconvolve and state_path.is_file():
        old_state = ProcessingState.read(state_path)
        old_output = output_dir / f"{stem}_decon_deskewed.ome.zarr"
        old_run = old_state.document["outputs"].get(old_output.name)
        if (
            old_run is not None
            and not old_state.completed_channels(old_output)
            and not old_state.completed_tiles(old_output)
        ):
            print("Rebuilding uncheckpointed ROI output; reusing saved selection.")
            resume = False
    process_skewed(
        root_path=acquisition.path,
        acquisition=acquisition,
        output_dir=output_dir,
        deconvolve=deconvolve,
        save_float32=save_float32,
        max_projection=False,
        flatfield_correction=flatfield_correction,
        create_fused_max_projection=False,
        z_downsample_level=z_downsample_level,
        crop_after_deskew=crop_after_deskew,
        decon_crop_scan=decon_crop_scan,
        decon_gpu_id=decon_gpu_id,
        decon_verbose=decon_verbose,
        decon_psf_paths=decon_psf_paths,
        roi_selection=processing_roi,
        resume=resume,
    )

    label = "decon_deskewed" if deconvolve else "deskewed"
    processed_path = output_dir / f"{stem}_{label}.ome.zarr"
    state = ProcessingState.read(processing_state_path(output_dir, stem))
    tile_records = state.roi_series(processed_path)
    selected_positions = tuple(
        dict.fromkeys(int(record["position_index"]) for record in tile_records)
    )
    # The shared fusion writer also assembles independently stored ROI timepoints
    # into TCZYX for one position; there are no pair measurements in that case.
    fusion_roi = replace(roi, position_indices=selected_positions)
    fusion = TileFusion(
        root_path=processed_path,
        channel_to_use=registration_channel,
        roi_selection=fusion_roi,
        multiscale_factors=(),
        resolution_multiples=((1, 1, 1),),
    )
    fusion.run()
    fused_path = output_dir / f"{stem}_fused.ome.zarr"
    max_z_path = output_dir / f"{stem}_max_z_fused.ome.zarr"
    regenerate_fused_max_projection(fused_path, max_z_path)
    print(f"ROI processing and fusion complete: {output_dir}")
    return fused_path
