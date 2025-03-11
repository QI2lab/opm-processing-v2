"""
Display OPM data

This file is meant for quick deskew and display of qi2lab OPM results. No deconvolution and it assumes everything fits in memory.
"""

from pathlib import Path
import tensorstore as ts
import napari
from cmap import Colormap

def display(root_path: Path):
    """Display deskewed OPM data.
    
    This code assumes data is already deskewed and on disk. Pass in the raw data
    folder for now to ensure metadata is read properly.
    
    Parameters
    ----------
    root_path: Path
        Path to OPM pymmcoregui zarr file.
    """
    
    max_z_output_path = root_path.parents[0] / Path(str(root_path.stem)+"_max_z_fused.zarr")

    # open datastore on disk
    spec = {
        "driver" : "zarr3",
        "kvstore" : {
            "driver" : "file",
            "path" : str(max_z_output_path)
        }
    }
    datastore = ts.open(spec).result()
 
    viewer = napari.Viewer()
    colormaps = [
        Colormap("chrisluts:bop_purple").to_napari(),
        Colormap("chrisluts:bop_blue").to_napari(),
        Colormap("chrisluts:bop_orange").to_napari(),
    ]
    viewer = napari.Viewer()
    for time_idx in range(datastore.shape[0]):
        for pos_idx in range(datastore.shape[1]):
            for chan_idx in range(datastore.shape[2]):
                viewer.add_image(
                    datastore[time_idx,pos_idx,chan_idx,:],
                    name="ch"+str(chan_idx).zfill(2),
                    scale=[.115,.115],
                    blending="additive",
                    colormap=colormaps[chan_idx],
                )
    napari.run()
    

if __name__ == "__main__":
    root_path = Path(r"G:\20250305_bulbc_brain_control\full_run_006.zarr")
    display(root_path)