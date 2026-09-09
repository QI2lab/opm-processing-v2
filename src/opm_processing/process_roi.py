"""Deconvolve and fuse a physical ROI selected on a napari max projection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from opm_processing.dataio.acquisition import acquisition_stem, inspect_acquisition
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)
from opm_processing.dataio.roi import (
    PhysicalRoi,
    validate_registered_max_projection,
)
from opm_processing.imageprocessing.tilefusion import TileFusion
from opm_processing.imageprocessing.maxtilefusion import (
    regenerate_fused_max_projection,
)
from opm_processing.process import process_skewed


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def process_roi(
    root_path: Annotated[
        Path,
        typer.Argument(help="Raw acquisition directory or OME-Zarr path."),
    ],
    roi_json: Annotated[
        Path | None,
        typer.Argument(
            help=(
                "ROI JSON written by display; defaults to "
                "<acquisition>_roi.json beside the acquisition."
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
            "--resume",
            help=(
                "Continue a compatible interrupted ROI run. Without this flag, "
                "processed ROI tiles are overwritten."
            ),
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="Output directory; defaults to <acquisition>_roi beside the source.",
        ),
    ] = None,
) -> None:
    """Process a napari-selected world-space ROI and fuse it when needed."""
    acquisition = inspect_acquisition(root_path)
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
        acquisition.path.parent / f"{stem}_roi.json"
        if roi_json is None
        else Path(roi_json).expanduser().resolve()
    )
    roi = PhysicalRoi.read(resolved_roi_json)
    validate_registered_max_projection(roi.source_path)
    output_dir = (
        Path(output).expanduser().resolve()
        if output is not None
        else acquisition.path.parent / f"{stem}_roi"
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
    if not tile_records:
        raise RuntimeError("ROI processing did not record any cropped tiles")
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
    """Run the ROI-processing command-line application."""
    app()


if __name__ == "__main__":
    main()
