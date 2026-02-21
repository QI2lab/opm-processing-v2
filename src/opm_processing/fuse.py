"""
Fuse qi2lab OPM data

This file registers and fuses deskewed qi2lab OPM data.
"""

import multiprocessing as mp
import sys
if sys.platform.startswith("linux"):
    mp.set_start_method("forkserver", force=True)
elif sys.platform.startswith("win"):
    mp.set_start_method("spawn", force=True)
    
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)

from pathlib import Path
import typer
from opm_processing.imageprocessing.tilefusion import TileFusion


app = typer.Typer()
app.pretty_exceptions_enable = False

@app.command()
def register_and_fuse(
    root_path: Path,
    chan_idx: int = 0,
):
    """Register and fuse processed OPM data.
    
    This code assumes data is already processed and on disk.
    
    Usage: `fuse "/path/to/qi2lab_acquisition.zarr"
    
    Output will be in `/path/to/qi2lab_acquisition_fused_<acq_type>.ome.zarr`
    
    <acq_type> will be either deskewed or projection depending on OPM mode.
    
    Parameters
    ----------
    root_path: Path
        Path to OPM pymmcoregui zarr file.
    chan_idx: int, default = 0
        Channel index to use for registration and fusion.
        If there is only one channel, this should be 0.
        If there are multiple channels, this should be the index of the channel
        to use for registration.
    """
    
    # Open zarr using tensorstore
    tile_fuser = TileFusion(root_path=root_path,channel_to_use=chan_idx)    
    tile_fuser.run()
    
# entry for point for CLI        
def main():
    app()

if __name__ == "__main__":
    main()