"""
Display OPM data

This file is meant for quick deskew and display of qi2lab OPM results. No deconvolution and it assumes everything fits in memory.
"""

from pathlib import Path
import tensorstore as ts
import napari
import json
from opm_processing.dataio.metadata import find_key, extract_stage_positions
import numpy as np
from napari.experimental import link_layers
from cmap import Colormap
from tqdm import tqdm
from tifffile import TiffWriter

def display(root_path: Path):
    """Display deskewed OPM data.
    
    This code assumes data is already deskewed and on disk. Pass in the raw data
    folder for now to ensure metadata is read properly.
    
    Parameters
    ----------
    root_path: Path
        Path to OPM pymmcoregui zarr file.
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
    
    max_z_output_path = root_path.parents[0] / Path(str(root_path.stem)+"_max_zfused.zarr")
    # todo: ADD tif creation file here and displyy
    if stage_y_flipped:
        stage_y_max = np.max(stage_positions[:,1])
        for pos_idx, _ in enumerate(stage_positions):
            stage_positions[pos_idx,1] = stage_y_max - stage_positions[pos_idx,1]

    if stage_z_flipped:
        stage_z_max = np.max(stage_positions[:,0])
        for pos_idx, _ in enumerate(stage_positions):
            stage_positions[pos_idx,0] = stage_z_max - stage_positions[pos_idx,0]
    
    # open datastore on disk
    spec = {
        "driver" : "zarr3",
        "kvstore" : {
            "driver" : "file",
            "path" : str(max_z_output_path)
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
    for time_idx in range(datastore.shape[0]):
        for pos_idx in range(datastore.shape[1]):
            for chan_idx in range(datastore.shape[2]):
                layer = viewer.add_image(
                    np.squeeze((datastore[time_idx,pos_idx,chan_idx,:]).read().result()),
                    scale=[pixel_size_um,pixel_size_um],
                    translate=[stage_positions[pos_idx,1],stage_positions[pos_idx,2]],
                    name = "p"+str(pos_idx).zfill(3)+"_c"+str(chan_idx),
                    blending="additive",
                    colormap=colormaps[chan_idx],
                    contrast_limits = [50,2000]
                )
                
                channel_layers[chan_idx].append(layer)
                
    for chan_idx in range(datastore.shape[2]):
        link_layers(channel_layers[chan_idx],("contrast_limits","gamma"))
            
    napari.run()
    

if __name__ == "__main__":
    root_path = Path(r"G:\20250305_bulbc_brain_control\full_run_006.zarr")
    display(root_path)