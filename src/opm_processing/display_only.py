"""
Deskew and display OPM data

This file is meant for quick deskew and display of qi2lab OPM results. No deconvolution and it assumes everything fits in memory.
"""

from pathlib import Path
import tensorstore as ts
import napari
import json
from opm_processing.dataio.metadata import find_key, extract_stage_positions
import numpy as np

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
    
    # Read metadata
    zattrs_path = root_path / Path(".zattrs")
    with open(zattrs_path, "r") as f:
        zattrs = json.load(f)

    pixel_size_um = float(find_key(zattrs,"pixel_size_um"))
    stage_positions = extract_stage_positions(zattrs)
    output_path = root_path.parents[0] / Path(str(root_path.stem)+"_deskewed.zarr")

    if stage_y_flipped:
        stage_y_max = np.max(stage_positions[:,1])
        for pos_idx, _ in enumerate(stage_positions):
            stage_positions[pos_idx,1] = stage_y_max - stage_positions[pos_idx,1]
    
    # open datastore on disk
    spec = {
            "driver" : "zarr3",
            "kvstore" : {
                "driver" : "file",
                "path" : str(output_path)
            }
        }
    datastore = ts.open(spec).result()
    
    # populate napari viewer using lazy loading
    viewer = napari.Viewer()
    for time_idx in range(datastore.shape[0]):
        for pos_idx in range(datastore.shape[1]):
            viewer.add_image(
                datastore[:,pos_idx,:,:],
                scale=[2*pixel_size_um,pixel_size_um,pixel_size_um],
                translate=stage_positions[pos_idx],
                name = "t"+str(time_idx).zfill(2)+"_p"+str(pos_idx).zfill(3),
                blending="additive"
            )
            
    napari.run()
    

if __name__ == "__main__":
    root_path = Path(r"G:\20250226_merfish_test\merfish_test.zarr")
    display(root_path)