"""
Deskew qi2lab OPM data.

This file deskews and creates maximum projections of raw qi2lab OPM data.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from pathlib import Path
import tensorstore as ts
from opm_processing.imageprocessing.opmpsf import generate_skewed_psf
# from opm_processing.imageprocessing.rlgc import chunked_rlgc
from opm_processing.imageprocessing.maxtilefusion import TileFusion
from opm_processing.dataio.metadata import extract_channels, find_key, extract_stage_positions, update_global_metadata, update_per_index_metadata
from opm_processing.dataio.zarr_handlers import create_via_tensorstore, write_via_tensorstore
import json
import numpy as np
from tqdm import tqdm
import typer

from tifffile import TiffWriter, imread
import napari
from napari.experimental import link_layers

# TODO
from opm_processing.imageprocessing.flatfield import estimate_illuminations


def run_estimate_illuminations(datastore, camera_offset, camera_conversion, conn):
    """Helper function to run estimate_illuminations in a subprocess.
    
    This is necessary because jaxlib does not release GPU memory until the
    process exists. So we need to isolate it so that the GPU can be used for
    other processing tasks.
    
    Parameters
    ----------
    datastore: TensorStore
        TensorStore object containing the data.
    camera_offset: float
        Camera offset value.
    camera_conversion: float
        Camera conversion value.
    conn: Pipe
        Pipe connection to send the result back to the main process.
    """
    from opm_processing.imageprocessing.flatfield import estimate_illuminations

    try:
        flatfields = estimate_illuminations(datastore, camera_offset, camera_conversion)
        conn.send(flatfields)
    except Exception as e:
        conn.send(e)
    finally:
        conn.close()
    
def call_estimate_illuminations(datastore, camera_offset, camera_conversion):
    """Helper function to call estimate_illuminations in a subprocess.
    
    This is necessary because jaxlib does not release GPU memory until the
    process exists. So we need to isolate it so that the GPU can be used for
    other processing tasks.
    
    Parameters
    ----------
    datastore: TensorStore
        TensorStore object containing the data.
    camera_offset: float
        Camera offset value.
    camera_conversion: float
        Camera conversion value.
    
    Returns
    -------
    flatfields: np.ndarray
        Estimated illuminations.
    """
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(
        target=run_estimate_illuminations,
        args=(datastore, camera_offset, camera_conversion, child_conn)
    )
    p.start()
    result = parent_conn.recv()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError("Subprocess failed")

    if isinstance(result, Exception):
        raise result

    return result

app = typer.Typer()
app.pretty_exceptions_enable = False

@app.command()
def process():
    root_path = Path(
        r'G:\20250521_test_autophagy\20250521_174119_test_timelapse\test_timelapse.zarr',
        # r"G:\20250523_autophagy\20250523_202020_projection_timelapse\projection_timelapse.zarr"
        )
    deconvolve=False
    flatfield_correction = True
    create_fused_projection = True
    write_fused_projection_tiff = True
    pos_range = None
    time_range = [1,5]
    app = typer.Typer()
    app.pretty_exceptions_enable = False

    # open raw datastore
    spec = {
        "driver" : "zarr",
        "kvstore" : {
            "driver" : "file",
            "path" : str(root_path)
        }
    }
    datastore = ts.open(spec).result()[:, :, :, None, :, : ]
    print(
        f'\nDatastore properties:',
        f'\n  shape:{datastore.shape}'
    )

    # Read metadata
    zattrs_path = root_path / Path(".zattrs")
    with open(zattrs_path, "r") as f:
        zattrs = json.load(f)

    pixel_size_um = float(find_key(zattrs,"pixel_size_um"))
    opm_tilt_deg = float(find_key(zattrs,"angle_deg"))    
    camera_offset = float(find_key(zattrs,"offset"))
    camera_conversion = float(find_key(zattrs,"e_to_ADU"))

    channels = extract_channels(zattrs)
    stage_positions = extract_stage_positions(zattrs)

    # TO DO: start writing these in metadata!
    stage_x_flipped = False
    stage_y_flipped = True
    stage_z_flipped = True

    # flip x positions w.r.t. camera <-> stage orientation
    # TO DO: this axis is probably affected by the scan_flip flag, need to think
    #        about that.
    if stage_x_flipped:
        stage_x_max = np.max(stage_positions[:,2])
        for pos_idx, _ in enumerate(stage_positions):
            stage_positions[pos_idx,2] = stage_x_max - stage_positions[pos_idx,2]

    # flip y positions w.r.t. camera <-> stage orientation
    if stage_y_flipped:
        stage_y_max = np.max(stage_positions[:,1])
        for pos_idx, _ in enumerate(stage_positions):
            stage_positions[pos_idx,1] = stage_y_max - stage_positions[pos_idx,1]

    # flip z positions w.r.t. camera <-> stage orientation
    if stage_z_flipped:
        stage_z_max = np.max(stage_positions[:,0])
        for pos_idx, _ in enumerate(stage_positions):
            stage_positions[pos_idx,0] = stage_z_max - stage_positions[pos_idx,0]

    if time_range is not None:
        time_shape = time_range[1]
    else:
        time_shape = datastore.shape[0]
        
    if pos_range is not None:
        pos_shape = pos_range[1]
    else:
        pos_shape = datastore.shape[1]
        
    datastore_shape = [
        time_shape,
        pos_shape,
        datastore.shape[2],
        1,
        datastore.shape[-2],
        datastore.shape[-1]
    ]
    print(
        f'  New datastore shape: {datastore_shape}'
        )
    # create tensorstore object for writing. This is NOT compatible with OME-NGFF!
    output_path = root_path.parents[0] / Path(str(root_path.stem)+"_processed.zarr")
    ts_store = create_via_tensorstore(output_path,datastore_shape)
        
    #  Run flat field correction on raw datastore
    if flatfield_correction:
        flatfield_path = root_path.parents[0] / Path(str(root_path.stem)+"_flatfield.ome.tif")
        if flatfield_path.exists():
            flatfields = imread(flatfield_path).astype(np.float32)
        else:
            flatfields = estimate_illuminations(datastore, camera_offset, camera_conversion)
            
            # flatfields = call_estimate_illuminations(datastore, camera_offset, camera_conversion)
            with TiffWriter(flatfield_path, bigtiff=True) as tif:
                metadata={
                    'axes': "CYX",
                    'SignificantBits': 32,
                    'PhysicalSizeX': pixel_size_um,
                    'PhysicalSizeXUnit': 'µm',
                    'PhysicalSizeY': pixel_size_um,
                    'PhysicalSizeYUnit': 'µm',
                }
                options = dict(
                    photometric='minisblack',
                    resolutionunit='CENTIMETER',
                )
                tif.write(
                    flatfields,
                    resolution=(
                        1e4 / pixel_size_um,
                        1e4 / pixel_size_um
                    ),
                    **options,
                    metadata=metadata
                )
    else:
        flatfields = np.ones((datastore.shape[2],datastore.shape[-2],datastore.shape[-1]),dtype=np.float32)

    ts_writes = []
    
    if time_range is not None:
        time_iterator = tqdm(range(time_range[0],time_range[1]),desc="t")
    else:
        time_iterator = tqdm(range(datastore.shape[0]),desc="t")
        
    if pos_range is not None:
        pos_iterator = tqdm(range(pos_range[0],pos_range[1]),desc="p",leave=False)
    else:
        pos_iterator = tqdm(range(datastore.shape[1]),desc="p",leave=False)

    for t_idx in time_iterator:
        for pos_idx in pos_iterator:
            for chan_idx in tqdm(range(datastore.shape[2]),desc="c",leave=False):
                camera_corrected_data = (
                    ((np.squeeze(datastore[t_idx,pos_idx,chan_idx,:].read().result()).astype(np.float32)-camera_offset)*camera_conversion)/flatfields[chan_idx,:].astype(np.float32)
                ).clip(0,2**16-1).astype(np.uint16)                    
                                # create future objects for async data writing
                
                update_per_index_metadata(
                    ts_store = ts_store, 
                    metadata = {
                        'stage_position': stage_positions[pos_idx], 
                        'channel': channels[chan_idx]
                        }, 
                    index_location = (t_idx, pos_idx, chan_idx, 0)
                )

                # add current camera corrected data to write out
                ts_writes.append(
                    write_via_tensorstore(
                        ts_store = ts_store,
                        data = np.expand_dims(camera_corrected_data, axis=0),
                        data_location = [t_idx,pos_idx,chan_idx]
                    )
                )
                
    # wait for writes to finish
    for ts_write in ts_writes:
        ts_write.result()
        
    update_global_metadata(
        ts_store = ts_store,
        global_metadata= {
                "raw_pixel_size_um" : pixel_size_um,
                "opm_tilt_deg" : opm_tilt_deg,
                "camera_corrected" : True,
                "camera_offset" : camera_offset,
                "camera_e_to_ADU" : camera_conversion,
                "voxel_size_um" : [pixel_size_um, pixel_size_um],
                "stage_x_flipped": stage_x_flipped,
                "stage_y_flipped": stage_y_flipped,
                "stage_z_flipped": stage_z_flipped,
                "flatfield_corrected": flatfield_correction
            }
    )
    if create_fused_projection:       
        print("\nFusing max projection using stage positions...")
        fused_output_path = root_path.parents[0] / Path(str(root_path.stem)+"_fused.zarr")
        
        if pos_range is not None:
            tile_positions = stage_positions[pos_range[0]:pos_range[1],1:]
            
        else:
            tile_positions = stage_positions[:,1:]
        
        
        temp_fusion = TileFusion(
                ts_dataset = ts_store[0,:,:,:,:,:],
                tile_positions = tile_positions,
                output_path=fused_output_path,
                pixel_size=np.asarray((pixel_size_um,pixel_size_um),dtype=np.float32),
            )
        fused_shape = temp_fusion.fused_shape
        ts_fused_store = create_via_tensorstore(
            fused_output_path, 
            data_shape=fused_shape
        )
        
        ts_fused_writes = []
        for t_idx in time_iterator:
        
            for chan_idx in channels:
                # Fuse a single time point
                tile_fusion = TileFusion(
                    ts_dataset = ts_store[t_idx,chan_idx,:,:,:,:],
                    tile_positions = tile_positions,
                    output_path=fused_output_path,
                    pixel_size=np.asarray((pixel_size_um,pixel_size_um),dtype=np.float32),
                )
                tile_fusion.run()
                # Add this fused timepoint the the ts to be written
                ts_fused_writes.append(
                    write_via_tensorstore(
                        ts_store = ts_fused_store,
                        data = tile_fusion.fused_ts[0,0,0].read().result().astype(np.float32),
                        data_location = [t_idx,0,chan_idx]
                    )
                )
            for ts_f_write in ts_fused_writes:
                ts_f_write.result()
    # if write_fused_projection_tiff:
    #     tiff_dir_path = output_path.parent / Path("fused_projection_tiff_output")
    #     tiff_dir_path.mkdir(exist_ok=True)
    #     max_spec = {
    #         "driver" : "zarr3",
    #         "kvstore" : {
    #             "driver" : "file",
    #             "path" : str(fused_output_path)
    #         }
    #     }
    #     max_proj_datastore = ts.open(max_spec).result()
    #     for t_idx in tqdm(range(max_proj_datastore.shape[0]),desc="t"):
    #         max_projection = np.squeeze(np.asarray(max_proj_datastore[t_idx,0,chan_idx,:].read().result()))
            
    #         filename = Path(f"fused_z_max_projection_t{t_idx}.ome.tiff")
    #         filename_path = tiff_dir_path /  Path(filename)
    #         if len(max_projection.shape) == 2:
    #             axes = "YX"
    #         else:
    #             axes = "CYX"
            
    #         with TiffWriter(filename_path, bigtiff=True) as tif:
    #             metadata={
    #                 'axes': axes,
    #                 'SignificantBits': 16,
    #                 'PhysicalSizeX': pixel_size_um,
    #                 'PhysicalSizeXUnit': 'µm',
    #                 'PhysicalSizeY': pixel_size_um,
    #                 'PhysicalSizeYUnit': 'µm',
    #             }
    #             options = dict(
    #                 compression='zlib',
    #                 compressionargs={'level': 8},
    #                 predictor=True,
    #                 photometric='minisblack',
    #                 resolutionunit='CENTIMETER',
    #             )
    #             tif.write(
    #                 max_projection,
    #                 resolution=(
    #                     1e4 / pixel_size_um,
    #                     1e4 / pixel_size_um
    #                 ),
    #                 **options,
    #                 metadata=metadata
    #             )
            
# entry for point for CLI        
def main():
    app()

if __name__ == "__main__":
    main()
    
    