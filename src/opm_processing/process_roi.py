"""Deconvolve and fuse a physical ROI selected on a napari max projection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from opm_processing.dataio.acquisition import (
    AcquisitionMetadata,
    acquisition_stem,
    inspect_acquisition,
)
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


def _resolve_roi_context(root_path: Path) -> tuple[AcquisitionMetadata, Path]:
    """Locate raw data while retaining the directory used for ROI defaults."""
    candidate = Path(root_path).expanduser().resolve()
    try:
        acquisition = inspect_acquisition(candidate)
    except ValueError as acquisition_error:
        if not candidate.is_dir() or any(
            (candidate / marker).is_file() for marker in ("zarr.json", ".zattrs")
        ):
            raise
        state_paths = sorted(candidate.glob("*.processing.json"))
        if not state_paths:
            raise ValueError(
                f"Cannot locate a raw acquisition in {candidate}, and no "
                "*.processing.json records its source. Pass the raw acquisition "
                "path as the first argument and the ROI JSON as the second."
            ) from acquisition_error
        if len(state_paths) != 1:
            matches = ", ".join(path.name for path in state_paths)
            raise ValueError(
                f"Expected one processing-state file in {candidate}, found "
                f"{len(state_paths)}: {matches}. Pass the raw acquisition path "
                "and ROI JSON explicitly to select an acquisition."
            ) from acquisition_error
        state = ProcessingState.read(state_paths[0])
        recorded_source = state.document["source"].get("path")
        if not isinstance(recorded_source, str) or not recorded_source.strip():
            raise ValueError(
                f"Processing state lacks a source acquisition path: {state.path}"
            )
        source_path = Path(recorded_source).expanduser().resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(
                f"Raw acquisition recorded in {state.path} is unavailable: "
                f"{source_path}. Mount its drive or pass the current raw "
                "acquisition path and ROI JSON explicitly."
            )
        acquisition = inspect_acquisition(source_path)
        return acquisition, candidate
    return acquisition, acquisition.path.parent


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
    """Process a napari-selected world-space ROI and fuse it when needed."""
    acquisition, context_dir = _resolve_roi_context(root_path)
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
    validate_registered_max_projection(roi.source_path)
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
