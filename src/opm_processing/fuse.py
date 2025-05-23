"""
Fuse qi2lab OPM data

This file registers and fuses deskewed qi2lab OPM data.
"""

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
    saved_shifts: bool = False
):
    """Register and fuse deskewed OPM data.
    
    This code assumes data is already deskewed and on disk.
    
    Usage: `fuse "/path/to/qi2lab_acquisition.zarr"
    
    Output will be in `/path/to/qi2lab_acquisition_fused_deskewed.ome.zarr`
    
    Parameters
    ----------
    root_path: Path
        Path to OPM pymmcoregui zarr file.
    saved_shifts: bool
        If True, use existing shifts on disk if available.
        If False, compute shifts from the zarr file.
    """
    
    # Open zarr using tensorstore
    if saved_shifts:
        tile_fuser = TileFusion(root_path=root_path)
    else:
        tile_fuser = TileFusion(root_path=root_path,metrics_filename=Path("none"))
    tile_fuser.run()
    
    
# entry for point for CLI        
def main():
    app()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()