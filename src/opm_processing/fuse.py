"""
Fuse qi2lab OPM data.

This file registers and fuses deskewed qi2lab OPM data.
"""

import multiprocessing as mp
import sys

if sys.platform.startswith("linux"):
    mp.set_start_method("forkserver", force=True)
elif sys.platform.startswith("win"):
    mp.set_start_method("spawn", force=True)

import warnings
from typing import Annotated

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)

from pathlib import Path
import typer
from opm_processing.dataio.acquisition import (
    acquisition_stem,
    resolve_acquisition_path,
)
from opm_processing.imageprocessing.maxtilefusion import (
    regenerate_fused_max_projection,
)
from opm_processing.imageprocessing.tilefusion import (
    TileFusion,
    fusion_backend_status,
    require_gpu_backend,
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
    max_registration_shift_zyx: tuple[int, int, int] = (20, 50, 100),
    require_gpu: bool = False,
    regenerate_max_z: Annotated[
        bool,
        typer.Option(
            "--regenerate-max-z",
            help=(
                "Only overwrite the fused maximum-Z projection using scale 0 "
                "of the existing registered full-resolution fused image."
            ),
        ),
    ] = False,
):
    """Register and fuse processed OPM data.

    This code assumes data is already processed and on disk.

    Usage: `fuse "/path/to/qi2lab_acquisition.zarr"

    Output will be in `/path/to/qi2lab_acquisition_fused.ome.zarr`

    <acq_type> will be either deskewed or projection depending on OPM mode.

    Parameters
    ----------
    root_path: Path
        Path to an OPM acquisition Zarr store or its containing directory.
    registration_channel: int, default = 0
        Zero-based channel index to use for registration.
        If there is only one channel, this should be 0.
        If there are multiple channels, this should be the index of the channel
        to use for registration.

    blend_pixels : tuple[int, int, int]
        Value supplied for ``blend pixels``.
    downsample_factors : tuple[int, int, int]
        Value supplied for ``downsample factors``.
    ssim_window : int
        Value supplied for ``ssim window``.
    registration_threshold : float
        Value supplied for ``registration threshold``.
    chunk_shape_yx : tuple[int, int]
        Value supplied for ``chunk shape yx``.
    fusion_ram_fraction : float
        Value supplied for ``fusion ram fraction``.
    max_workers : int or None
        Number of CPU fusion workers. By default, uses up to eight physical
        cores so each concurrent block retains efficient Z depth.
    max_in_flight_writes : int
        Value supplied for ``max in flight writes``.
    optimization_rel_threshold : float
        Value supplied for ``optimization rel threshold``.
    optimization_abs_threshold : float
        Value supplied for ``optimization abs threshold``.
    max_registration_shift_zyx : tuple[int, int, int]
        Value supplied for ``max registration shift zyx``.
    require_gpu : bool
        Fail instead of silently using CPU registration when CUDA is unavailable.
    regenerate_max_z : bool
        Regenerate only the fused maximum-Z projection from the existing
        registered full-resolution fused image.

    Returns
    -------
    None
        No value is returned.
    """
    if regenerate_max_z:
        acquisition_path = resolve_acquisition_path(root_path)
        base = acquisition_path.parent
        stem = acquisition_stem(acquisition_path)
        fused_path = base / f"{stem}_fused.ome.zarr"
        output_path = base / f"{stem}_max_z_fused.ome.zarr"
        regenerate_fused_max_projection(
            fused_path,
            output_path,
            max_workers=max_workers,
        )
        print(f"Regenerated fused maximum-Z projection: {output_path}")
        return

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
    )
    tile_fuser.run()


# entry for point for CLI
def main():
    """Run the registration and fusion command-line application.

    Parameters
    ----------
    None
        This callable has no parameters.

    Returns
    -------
    None
        No value is returned.
    """
    app()


if __name__ == "__main__":
    main()
