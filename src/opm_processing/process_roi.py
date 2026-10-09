"""Deconvolve and fuse a physical ROI selected on a napari max projection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

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

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def process_roi(
    root_path: Annotated[
        Path,
        typer.Argument(
            help=(
                "Raw acquisition directory, OME-Zarr path, or processed output "
                "directory containing one <acquisition>.processing.json."
            )
        ),
    ],
    roi_json: Annotated[
        Path | None,
        typer.Argument(
            help=(
                "ROI JSON written by display; defaults to "
                "<acquisition>_roi.json in the input directory."
            )
        ),
    ] = None,
    deconvolve: Annotated[
        bool,
        typer.Option("--deconvolve/--no-deconvolve"),
    ] = True,
    flatfield_correction: bool = False,
    save_float32: bool = False,
    z_downsample_level: int = 2,
    crop_after_deskew: bool = False,
    decon_crop_scan: int | None = None,
    decon_gpu_id: int = 0,
    decon_verbose: int = 1,
    decon_psf_paths: list[Path] | None = None,
    registration_channel: int = 0,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help=(
                "Resume from completed tiles/channels by default. Use "
                "--no-resume to overwrite the processed ROI and start again."
            ),
        ),
    ] = True,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help=(
                "Output directory; defaults to <acquisition>_roi in the input "
                "directory (beside the source when a Zarr path is supplied)."
            ),
        ),
    ] = None,
) -> None:
    """Process a napari-selected world-space ROI and fuse its cropped tiles.

    Parameters
    ----------
    root_path : pathlib.Path
        Raw acquisition path or directory containing its processing state.
    roi_json : pathlib.Path or None
        ROI exported by display; None uses the input directory's default ROI.
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
    None
        Cropped tiles and checkpoints are saved, with fused outputs when the ROI
        spans multiple positions.
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
            "process-ROI currently targets skewed 3D acquisitions; use process "
            "for native 2D data"
        )
    if crop_after_deskew:
        raise ValueError(
            "process-ROI already writes exact physical ROI crops; "
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
        roi_selection=roi,
        resume=resume,
    )

    label = "decon_deskewed" if deconvolve else "deskewed"
    processed_path = output_dir / f"{stem}_{label}.ome.zarr"
    state = ProcessingState.read(processing_state_path(output_dir, stem))
    tile_records = state.roi_series(processed_path)
    selected_positions = tuple(
        dict.fromkeys(int(record["position_index"]) for record in tile_records)
    )
    if len(selected_positions) <= 1:
        print(
            "ROI processing complete; fusion skipped because the ROI intersects "
            "one position."
        )
        return

    print(f"Fusing {len(tile_records)} cropped ROI tiles...")
    fusion_roi = replace(roi, position_indices=selected_positions)
    fusion = TileFusion(
        root_path=processed_path,
        channel_to_use=registration_channel,
        roi_selection=fusion_roi,
    )
    fusion.run()
    fused_path = output_dir / f"{stem}_fused.ome.zarr"
    max_z_path = output_dir / f"{stem}_max_z_fused.ome.zarr"
    regenerate_fused_max_projection(fused_path, max_z_path)
    print(f"ROI processing and fusion complete: {output_dir}")


def main() -> None:
    """Run the ROI-processing command-line application.

    Returns
    -------
    None
        The Typer application parses arguments and processes the requested ROI.
    """
    app()


if __name__ == "__main__":
    main()
