"""
Fuse qi2lab OPM data.

This file registers and fuses deskewed qi2lab OPM data.
"""

from pathlib import Path
from typing import Annotated

import typer
from opm_processing.dataio.acquisition import (
    acquisition_stem,
    resolve_acquisition_path,
)
from opm_processing.dataio.roi import PhysicalRoi
from opm_processing.imageprocessing.maxtilefusion import (
    regenerate_fused_max_projection,
)
from opm_processing.imageprocessing.tilefusion import (
    TileFusion,
    fusion_backend_status,
    require_gpu_backend,
    resolve_fusion_input,
)


app = typer.Typer()
app.pretty_exceptions_enable = False


@app.command()
def register_and_fuse(
    root_path: Path,
    registration_channel: Annotated[
        int,
        typer.Option(
            "--registration-channel",
            "--chan-idx",
            min=0,
            help="Zero-based channel index used to register tile positions.",
        ),
    ] = 0,
    blend_pixels: tuple[int, int, int] = (20, 600, 400),
    downsample_factors: tuple[int, int, int] = (3, 5, 5),
    ssim_window: int = 15,
    registration_threshold: float = 0.7,
    chunk_shape_yx: tuple[int, int] = (1024, 1024),
    fusion_ram_fraction: float = 0.4,
    max_workers: int | None = None,
    max_in_flight_writes: int = 2,
    optimization_rel_threshold: float = 0.5,
    optimization_abs_threshold: float = 1.5,
    max_registration_shift_zyx: Annotated[
        tuple[int, int, int] | None,
        typer.Option(
            help=(
                "Maximum ZYX corrections in pixels. Default: infer per-pair "
                "limits from stage spacing and scan geometry."
            ),
        ),
    ] = None,
    normalize_depth_intensity: Annotated[
        bool,
        typer.Option(
            "--normalize-depth-intensity/--no-normalize-depth-intensity",
            help=(
                "Match brightness between depth layers using one gain per "
                "channel shared by all XY tiles at each depth."
            ),
        ),
    ] = True,
    require_gpu: bool = False,
    regenerate_max_z: Annotated[
        bool,
        typer.Option(
            "--regenerate-max-z",
            help=(
                "Only overwrite the fused maximum-Z projection by projecting "
                "every scale of the existing registered fused image."
            ),
        ),
    ] = False,
    roi: Annotated[
        Path | None,
        typer.Option(
            "--roi",
            help="Physical ROI JSON created by display --roi-output.",
        ),
    ] = None,
) -> None:
    """Register processed OPM tiles, fuse them, and save a maximum projection.

    Parameters
    ----------
    root_path : pathlib.Path
        Raw acquisition path, processed tile store, or processing output directory.
    registration_channel : int
        Zero-based channel used to align overlapping tiles.
    blend_pixels : tuple[int, int, int]
        Feathering widths in Z, Y, and X pixels at each tile edge.
    downsample_factors : tuple[int, int, int]
        Z, Y, and X sampling factors used during overlap registration.
    ssim_window : int
        Window width used to score registered overlaps with structural similarity.
    registration_threshold : float
        Minimum structural similarity required to accept a pairwise registration.
    chunk_shape_yx : tuple[int, int]
        Y and X dimensions of output fusion blocks.
    fusion_ram_fraction : float
        Fraction of available host RAM used to size concurrent fusion blocks.
    max_workers : int or None
        CPU fusion workers; None uses up to eight physical cores.
    max_in_flight_writes : int
        Maximum queued fusion-block writes before waiting for disk output.
    optimization_rel_threshold : float
        Median-residual multiplier used to reject registration outliers.
    optimization_abs_threshold : float
        Minimum residual cutoff in effective registration sampling bins.
    max_registration_shift_zyx : tuple[int, int, int] or None
        Maximum ZYX corrections in pixels; None derives limits from stage spacing
        and scan geometry for each overlap.
    normalize_depth_intensity : bool
        Match registered depth layers using channel gains shared across XY tiles
        at each depth and timepoint. Save the gains alongside the fused image.
    require_gpu : bool
        Require CUDA registration instead of allowing the CPU backend.
    regenerate_max_z : bool
        Regenerate all maximum-Z pyramid levels from the existing fused image
        without registering or fusing tiles again.
    roi : pathlib.Path or None
        Display-exported ROI restricting positions and fused YX bounds while
        retaining every Z plane; None fuses the full acquisition.

    Returns
    -------
    None
        Registered fused data and its maximum-Z pyramid are written to disk.
    """
    if regenerate_max_z:
        try:
            output_dir, _processed_path, stem, _source_path = resolve_fusion_input(
                root_path
            )
        except FileNotFoundError:
            acquisition_path = resolve_acquisition_path(root_path)
            output_dir = acquisition_path.parent
            stem = acquisition_stem(acquisition_path)
    else:
        status = fusion_backend_status(max_workers=max_workers)
        if require_gpu:
            require_gpu_backend()
        print(
            "Fusion backends: "
            f"registration={status['registration_backend']}; "
            f"fusion={status['fusion_backend']} "
            f"({status['fusion_workers']} block workers)"
        )
        if not status["gpu_registration"]:
            print(f"GPU registration unavailable: {status['gpu_error']}")

        tile_fuser = TileFusion(
            root_path=root_path,
            channel_to_use=registration_channel,
            blend_pixels=blend_pixels,
            downsample_factors=downsample_factors,
            ssim_window=ssim_window,
            threshold=registration_threshold,
            chunk_shape_yx=chunk_shape_yx,
            fusion_ram_fraction=fusion_ram_fraction,
            max_workers=max_workers,
            max_in_flight_writes=max_in_flight_writes,
            optimization_rel_threshold=optimization_rel_threshold,
            optimization_abs_threshold=optimization_abs_threshold,
            max_registration_shift_zyx=max_registration_shift_zyx,
            roi_selection=None if roi is None else PhysicalRoi.read(roi),
            normalize_depth_intensity=normalize_depth_intensity,
        )
        tile_fuser.run()
        output_dir = tile_fuser.output_dir
        stem = tile_fuser.acquisition_name
    fused_path = output_dir / f"{stem}_fused.ome.zarr"
    max_z_path = output_dir / f"{stem}_max_z_fused.ome.zarr"
    regenerate_fused_max_projection(
        fused_path,
        max_z_path,
        max_workers=max_workers,
    )
    action = "Regenerated" if regenerate_max_z else "Created registered"
    print(f"{action} fused maximum-Z projection: {max_z_path}")


def main() -> None:
    """Run the registration and fusion command-line application.

    Returns
    -------
    None
        The Typer application parses arguments and runs the requested fusion.
    """
    app()


if __name__ == "__main__":
    main()
