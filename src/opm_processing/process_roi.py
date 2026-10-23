"""Deconvolve and fuse a physical ROI selected on a napari max projection."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from opm_processing.dataio.acquisition import (
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
from opm_processing.imageprocessing.tilefusion import TileFusion
from opm_processing.process import process_skewed


def process_roi(
    root_path: Path,
    roi_json: Path,
    deconvolve: bool = False,
    flatfield_correction: bool = False,
    save_float32: bool = False,
    z_downsample_level: int = 2,
    crop_after_deskew: bool = False,
    decon_crop_scan: int | None = None,
    decon_gpu_id: int = 0,
    decon_verbose: int = 1,
    decon_psf_paths: list[Path] | None = None,
    registration_channel: int = 0,
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
        Deconvolution diagnostic verbosity.
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
        acquisition = inspect_acquisition(Path(state.document["source"]["path"]))
        context_dir = requested_path
    else:
        acquisition = inspect_acquisition(requested_path)
        context_dir = acquisition.path.parent
    if acquisition.is_2d:
        raise ValueError(
            "ROI processing currently targets skewed 3D acquisitions; use process "
            "for native 2D data"
        )
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

    # Preserve geometric overlap outside the requested rectangle for local
    # registration. Crop to the exact requested bounds only during final fusion.
    margins = [0.0, 0.0]
    footprints = roi.tile_footprints
    for index, left in enumerate(footprints):
        for right in footprints[index + 1 :]:
            if (
                left["time_index"] != right["time_index"]
                or left["position_index"] == right["position_index"]
            ):
                continue
            a, b = left["bounds_yx_um"], right["bounds_yx_um"]
            overlap = [
                min(a[1], b[1]) - max(a[0], b[0]),
                min(a[3], b[3]) - max(a[2], b[2]),
            ]
            if min(overlap) <= 0:
                continue
            # Depth-neighbor alignment can move the common oblique-plane
            # footprint along Y by dz*cot(theta); retain that image context.
            depth = abs(left["origin_zyx_um"][0] - right["origin_zyx_um"][0])
            margins[0] = max(
                margins[0], depth / math.tan(math.radians(acquisition.angle_deg))
            )
            for axis in range(2):
                if left["origin_zyx_um"][axis + 1] != right["origin_zyx_um"][axis + 1]:
                    margins[axis] = max(margins[axis], overlap[axis])
    y0, y1, x0, x1 = roi.bounds_yx_um
    processing_roi = replace(
        roi,
        bounds_yx_um=(
            y0 - margins[0],
            y1 + margins[0],
            x0 - margins[1],
            x1 + margins[1],
        ),
    )
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
