from pathlib import Path
import tensorstore as ts
import json
import numpy as np
import dask.array as da
# from multiview_stitcher import spatial_image_utils as si_utils
# from multiview_stitcher import (
#     fusion,
#     io,
#     msi_utils,
#     vis_utils,
#     ngff_utils,
#     param_utils,
#     registration,
# )

def register_and_fuse(root_path: Path):
    """Register and fuse deskewed OPM data.
    
    This code assumes data is already deskewed and on disk. Pass in the raw data
    folder for now to ensure metadata is read properly.
    
    Parameters
    ----------
    root_path: Path
        Path to OPM pymmcoregui zarr file.
    """
    deskewed_data_path = root_path.parents[0] / Path(str(root_path.stem)+"_deskewed.zarr")
    spec = {
        "driver" : "zarr3",
        "kvstore" : {
            "driver" : "file",
            "path" : str(deskewed_data_path)
        }
    }
    datastore = ts.open(spec).result()
    
    
    # Access the kvstore directly
    attrs_store = datastore.kvstore

    # Read attributes (as ReadResult)
    attrs_result = attrs_store.read(".zattrs").result()

    if attrs_result.value is not None:
        attrs_json = attrs_result.value.decode('utf-8')
        attributes = json.loads(attrs_json)
        print(attributes)
    else:
        print("No attributes found.")
        
    if __name__ == "__main__":
        root_path = Path(r"G:\20250325_OB_stage\full_run_004.zarr")
        register_and_fuse(root_path)