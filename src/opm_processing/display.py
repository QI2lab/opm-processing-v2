"""
Display qi2lab OPM data

This file displays deskewed qi2lab OPM data.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from pathlib import Path
import tensorstore as ts
import napari
import json
from opm_processing.dataio.metadata import find_key, extract_stage_positions
import numpy as np
from napari.experimental import link_layers
from cmap import Colormap
import typer
from tqdm import tqdm

app = typer.Typer()
app.pretty_exceptions_enable = False

@app.command()
def display(
    root_path: Path, 
    to_display: str = "max-z",
    time_range: tuple[int,int] = None,
    pos_range: tuple[int,int] = None
):
    """Display deskewed OPM data.
    
    This code assumes data is already deskewed and on disk.
    
    Usage: `display "/path/to/qi2lab_acquisition.zarr" --to-display DISPLAY_OPTION` \
    --time-range TSTART TEND --pos-range PSTART PEND
    
    `OPTION` is one of `{max-z, fused-max-z, full}`. 
        - `max-z` loads each maximum Z projection and places in the recorded \
            stage position in napari.
        - `full` loads each deskewed tile and place in the recorded stage \
            position in napari.
        - `fused-max-z` loads the maximum z projections fused using recorded \
            stage positions in napari.
        - `fused-full` loads the full registered and fused deskewed data if \
            available.
    `TSTART` and `TEND` are the start (inclusive) and stop (exclusive) time \
        indices to load and display
    `PSTART` and `PEND` are the start (inclusive) and stop (exclusive) position \
        indices to load and display
    
    
    Parameters
    ----------
    root_path: Path
        Path to OPM pymmcoregui zarr file.
    to_display: str, default = "max-z"
        Data type to display. Options are "max-z" for maximum z projection,
        "fused-max-z" for fused maximum z projction, or "full" for 3D data.
    time_range: list[int,int], default = None
        Range of timepoints to reconstruct
    pos_range: list[int,int], default = None
        Range of stage positions to reconstruct   
    """
    
    # account for flip between camera and stage in y direction
    stage_y_flipped = True
    stage_z_flipped = True
    
    # Read metadata
    zattrs_path = root_path / Path(".zattrs")
    with open(zattrs_path, "r") as f:
        zattrs = json.load(f)

    pixel_size_um = float(find_key(zattrs,"pixel_size_um"))
    stage_positions = extract_stage_positions(zattrs)
    
    if stage_y_flipped:
        stage_y_max = np.max(stage_positions[:,1])
        for pos_idx, _ in enumerate(stage_positions):
            stage_positions[pos_idx,1] = stage_y_max - stage_positions[pos_idx,1]

    if stage_z_flipped:
        stage_z_max = np.max(stage_positions[:,0])
        for pos_idx, _ in enumerate(stage_positions):
            stage_positions[pos_idx,0] = stage_z_max - stage_positions[pos_idx,0]
    
    if to_display == "max-z":
        data_path = root_path.parents[0] / Path(str(root_path.stem)+"_max_z_deskewed.zarr")
        scale_to_use = [pixel_size_um,pixel_size_um]
    if to_display == "fused-max-z":
        data_path = root_path.parents[0] / Path(str(root_path.stem)+"_max_z_fused.zarr")
        scale_to_use = [pixel_size_um,pixel_size_um]
    elif to_display == "full":
        data_path = root_path.parents[0] / Path(str(root_path.stem)+"_deskewed.zarr")
        scale_to_use = [pixel_size_um*2,pixel_size_um,pixel_size_um]
        
    # open datastore on disk
    spec = {
        "driver" : "zarr3",
        "kvstore" : {
            "driver" : "file",
            "path" : str(data_path)
        }
    }
    datastore = ts.open(spec).result()
        
    channel_layers = {ch: [] for ch in range(datastore.shape[2])}
    colormaps = [
        Colormap("chrisluts:bop_purple").to_napari(),
        Colormap("chrisluts:bop_blue").to_napari(),
        Colormap("chrisluts:bop_orange").to_napari(),
    ]
    viewer = napari.Viewer()
    
    
    if time_range is not None:
        time_iterator = tqdm(range(time_range[0],time_range[1]),desc="t")
    else:
        time_iterator = tqdm(range(datastore.shape[0]),desc="t")
    
    if not(to_display == "fused-max-z"):
        if pos_range is not None:
            pos_iterator = tqdm(range(pos_range[0],pos_range[1]),desc="p",leave=False)
        else:
            pos_iterator = tqdm(range(datastore.shape[1]),desc="p",leave=False)
    else:
        pos_iterator = tqdm(range(datastore.shape[1]),desc="p",leave=False)
        
    for time_idx in time_iterator:
        for pos_idx in pos_iterator:
            for chan_idx in range(datastore.shape[2]):
                if to_display == "full":
                    translate_to_use = [
                        stage_positions[pos_idx,0],
                        stage_positions[pos_idx,1],
                        stage_positions[pos_idx,2]
                    ]
                elif to_display == "max-z":
                    translate_to_use = [
                        stage_positions[pos_idx,1],
                        stage_positions[pos_idx,2]
                    ]
                elif to_display == "fused-max-z":
                    translate_to_use = [
                        (np.max(stage_positions[:,1]) - np.min(stage_positions[:,1]))/2,
                        (np.max(stage_positions[:,2]) - np.min(stage_positions[:,2]))/2
                    ]
                    
                layer = viewer.add_image(
                    datastore[time_idx,pos_idx,chan_idx,:],
                    scale=scale_to_use,
                    translate=translate_to_use,
                    name = "p"+str(pos_idx).zfill(3)+"_c"+str(chan_idx),
                    blending="additive",
                    colormap=colormaps[chan_idx],
                    contrast_limits = [10,500]
                )
                
                channel_layers[chan_idx].append(layer)
    
    if not(to_display == "fused-max-z"):
        for chan_idx in range(datastore.shape[2]):
            link_layers(channel_layers[chan_idx],("contrast_limits","gamma"))
            
    napari.run()
    
# entry for point for CLI        
def main():
    app()

if __name__ == "__main__":
    main()