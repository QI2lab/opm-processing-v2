"""
Fuse qi2lab OPM data

This file registers and fuses deskewed qi2lab OPM data.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from pathlib import Path
import tensorstore as ts
import json
from opm_processing.dataio.zarr_handlers import TensorStoreWrapper
import dask.array as da
import dask.diagnostics as diag
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import (
    fusion,
    msi_utils,
    ngff_utils,
    registration,
)
import typer
from tqdm import trange

app = typer.Typer()
app.pretty_exceptions_enable = False

@app.command()
def register_and_fuse(root_path: Path):
    """Register and fuse deskewed OPM data.
    
    This code assumes data is already deskewed and on disk.
    
    Usage: `fuse "/path/to/qi2lab_acquisition.zarr"
    
    Output will be in `/path/to/qi2lab_acquisition_fused_deskewed.ome.zarr`
    
    Parameters
    ----------
    root_path: Path
        Path to OPM pymmcoregui zarr file.
    """
    
    # Open zarr using tensorstore
    deskewed_data_path = root_path.parents[0] / Path(str(root_path.stem)+"_deskewed.zarr")
    spec = {
        "driver" : "zarr3",
        "kvstore" : {
            "driver" : "file",
            "path" : str(deskewed_data_path)
        }
    }
    datastore = ts.open(spec).result()
    
    # Read metadata
    attrs_json = deskewed_data_path / Path("zarr.json")
    with open(attrs_json, 'r') as file:
        metadata = json.load(file)
    
    # Extract deskewed image scale
    deskewed_voxel_size_um = metadata['attributes']['deskewed_voxel_size_um']
    scale = {
        "z" : deskewed_voxel_size_um[0],
        "y" : deskewed_voxel_size_um[1],
        "x" : deskewed_voxel_size_um[2]
    }
    
    # Create list of multiscale spatial images over all tiles
    print('Lazy loading deskewed data and stage coordinates...')
    msims = []
    for time_idx in trange(datastore.shape[0],desc="t",leave=False):
        for pos_idx in trange(datastore.shape[1],desc="p",leave=False):
            stage_pos_zyx_um = metadata['attributes']['per_index_metadata'][str(time_idx)][str(pos_idx)]['0']['stage_position']
            translation = {
                "z" : stage_pos_zyx_um[0],
                "y" : stage_pos_zyx_um[1],
                "x" : stage_pos_zyx_um[2]
            }
            
            # Convert tensorstore to dask.array using custom wrapper 
            da_array = da.squeeze(
                da.from_array(
                    TensorStoreWrapper(datastore[time_idx,pos_idx,:]),
                    chunks=datastore.chunk_layout.read_chunk.shape[2:],
                )
            )
            
            # create spatial image
            sim = si_utils.get_sim_from_array(
                da_array,
                dims=["c", "z", "y", "x"],
                scale=scale,
                translation=translation,
                transform_key="stage",
            )
            
            # convert spatial image to multiscale spatial image
            msim = msi_utils.get_msim_from_sim(sim)
            msims.append(msim)
    
    # Calculation global registration using the first channel and downsampling
    print('Calculating registrations...')
    with diag.ProgressBar():
        _ = registration.register(
            msims,
            registration_binning={'z': 3, 'y': 5, 'x': 5},
            reg_channel_index=0,
            transform_key="stage",
            new_transform_key='affine_registered',
            pre_registration_pruning_method="alternating_pattern",
            post_registration_do_quality_filter=True,
            post_registration_quality_threshold=0.5,
            groupwise_resolution_kwargs={
                'transform': 'translation',
            }
        )
        
    # Create dask task map for fusion given output chunksize
    print('Constructing task map for tile fusion...')
    with diag.ProgressBar():
        fused = fusion.fuse(
            [msi_utils.get_sim_from_msim(msim) for msim in msims],
            transform_key="affine_registered",
            #output_chunksize=256
        )
        
    # Save fusion to disk
    output_filename = root_path.parents[0] / Path(str(root_path.stem)+"_fused_deskewed.ome.zarr")
    print(f"Saving fused data to {output_filename}...")
    with diag.ProgressBar():
        fused = ngff_utils.write_sim_to_ome_zarr(
            fused, output_filename, overwrite=True
        )

        
# entry for point for CLI        
def main():
    app()

if __name__ == "__main__":
    main()