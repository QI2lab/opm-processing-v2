"""
Command line tool for postprocessing qi2lab OPM data.

This script assumes data is generated by opm-v2 GUI and the resulting data is saved in zarr3 format.

By default, a max projection datastore is created for quick viewing.

History:
---------
- **2025/03**: Updated for new qi2lab OPM processing pipeline.
"""

from pathlib import Path
import napari.utils
import tensorstore as ts
import napari
from napari.experimental import link_layers
from cmap import Colormap
from opm_processing.imageprocessing.opmtools import deskew, downsample_axis, deskew_shape_estimator
from opm_processing.imageprocessing.utils import flatfield_correction, optimize_stage_positions
from opm_processing.dataio.metadata import extract_channels, find_key, extract_stage_positions, update_global_metadata, update_per_index_metadata
from opm_processing.dataio.zarr_handlers import create_via_tensorstore, write_via_tensorstore
import json
import numpy as np
from tqdm import tqdm
from basicpy import BaSiC
import typer

app = typer.Typer()
app.pretty_exceptions_enable = False

@app.command()
def postprocess(
    root_path: Path,
    max_projection: bool = True,
    flatfield_after_deskew: bool = True,
    optimize_stage_pos: bool = True,
    display_max_projection: bool = True,
    display_full_volume: bool = False,
    z_downsample_level: int = 2
):
    """Postprocess qi2lab OPM dataset.
    
    This code assumes data is generated by opm-v2 GUI and the resulting data is 
    saved using OPMMirrorHandler. All revelant metadata is read from imaging files, 
    including stage transformation, camera parameters, and channels. 
    
    Parameters
    ----------
    root_path: Path
        Path to OPM pymmcoregui zarr file.
    max_projection: bool, default = True
        Create a maximum projection datastore.
    display_max_projection: bool, default = False
        Display maximum projection in napari.
    display_full_volume: bool, default = False
        Display full volume in napari.
    flatfield_after_deskew: bool, default = True
        Estimate and apply flatfield correction after deskew. Uses max projection data if available.
    z_downsample_level: int, default = 2
        Amount to downsample deskewed data in z.
    """
    
    # open raw datastore
    spec = {
        "driver" : "zarr",
        "kvstore" : {
            "driver" : "file",
            "path" : str(root_path)
        }
    }
    datastore = ts.open(spec).result()

    # Read metadata
    zattrs_path = root_path / Path(".zattrs")
    with open(zattrs_path, "r") as f:
        zattrs = json.load(f)

    image_mirror_step_um = float(find_key(zattrs,"image_mirror_step_um"))
    pixel_size_um = float(find_key(zattrs,"pixel_size_um"))
    opm_tilt_deg = float(find_key(zattrs,"angle_deg"))
    
    camera_offset = float(find_key(zattrs,"offset"))
    camera_conversion = float(find_key(zattrs,"e_to_ADU"))
    
    channels = extract_channels(zattrs)
    stage_positions = extract_stage_positions(zattrs)
    stage_y_flipped = True
    stage_z_flipped = True

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
    
    # estimate shape of one deskewed volume
    deskewed_shape = deskew_shape_estimator(
        [datastore.shape[-3],datastore.shape[-2],datastore.shape[-1]],
        theta=opm_tilt_deg,
        distance=image_mirror_step_um,
        pixel_size=pixel_size_um
    )

    datastore_shape = [
        datastore.shape[0],
        datastore.shape[1],
        datastore.shape[2],
        deskewed_shape[0]//z_downsample_level,
        deskewed_shape[1],
        deskewed_shape[2]
    ]
    # create array to hold one deskewed volume 
    deskewed = np.zeros(
        (deskewed_shape[0]//z_downsample_level,deskewed_shape[1],deskewed_shape[2]),
        dtype=np.uint16
    )
    
    # create tensorstore object for writing. This is NOT compatible with OME-NGFF!
    output_path = root_path.parents[0] / Path(str(root_path.stem)+"_deskewed.zarr")
    ts_store = create_via_tensorstore(output_path,datastore_shape)

    if max_projection:
        max_z_datastore_shape = [
            datastore.shape[0],
            datastore.shape[1],
            datastore.shape[2],
            1,
            deskewed_shape[1],
            deskewed_shape[2]
        ]

        # create array to hold one maximum projection deskewed volume 
        max_z_deskewed = np.zeros(
            (1,deskewed_shape[1],deskewed_shape[2]),
            dtype=np.uint16
        )

        # create tensorstore object for writing. This is NOT compatible with OME-NGFF!
        max_z_output_path = root_path.parents[0] / Path(str(root_path.stem)+"_max_z_deskewed.zarr")
        max_z_ts_store = create_via_tensorstore(max_z_output_path,max_z_datastore_shape)
    
    # loop over all components and stream to zarr using tensorstore
    ts_writes = []
    if max_projection:
        ts_max_writes = []
    
    for t_idx in tqdm(range(datastore.shape[0]),desc="t"):
        for pos_idx in tqdm(range(datastore.shape[1]),desc="p",leave=False):
            for chan_idx in tqdm(range(datastore.shape[2]),desc="c",leave=False):
                camera_corrected_data = ((np.squeeze(datastore[t_idx,pos_idx,chan_idx,:].read().result()).astype(np.float32)-camera_offset)*camera_conversion).clip(0,2**16-1).astype(np.uint16)
                deskewed = downsample_axis(
                    deskew(
                        camera_corrected_data,
                        theta = opm_tilt_deg,
                        distance = image_mirror_step_um,
                        pixel_size = pixel_size_um,
                    ),
                    level = z_downsample_level,
                    axis = 0
                )

                update_per_index_metadata(
                    ts_store = ts_store, 
                    metadata = {"stage_position": stage_positions[pos_idx], 'channel': channels[chan_idx]}, 
                    index_location = (t_idx,pos_idx,chan_idx)
                )
                
                if max_projection:
                    max_z_deskewed = np.max(deskewed,axis=0,keepdims=True)
                    update_per_index_metadata(
                        ts_store = max_z_ts_store, 
                        metadata = {"stage_position": stage_positions[pos_idx], 'channel': channels[chan_idx]}, 
                        index_location = (t_idx,pos_idx,chan_idx)
                    )
                    # create future objects for async data writing
                    ts_max_writes.append(
                        write_via_tensorstore(
                            ts_store = max_z_ts_store,
                            data = max_z_deskewed,
                            data_location = [t_idx,pos_idx,chan_idx]
                        )
                    )

                # create future objects for async data writing
                ts_writes.append(
                    write_via_tensorstore(
                        ts_store = ts_store,
                        data = deskewed,
                        data_location = [t_idx,pos_idx,chan_idx]
                    )
                )
                
    # wait for writes to finish
    for ts_write in tqdm(ts_writes,desc='writes'):
        ts_write.result()

    if max_projection:
        for ts_max_write in tqdm(ts_max_writes,desc='max writes'):
            ts_max_write.result()

    update_global_metadata(
        ts_store = ts_store,
        global_metadata= {
                "image_mirror_step_um" : image_mirror_step_um,
                "raw_pixel_size_um" : pixel_size_um,
                "opm_tilt_deg" : opm_tilt_deg,
                "camera_corrected" : True,
                "camera_offset" : camera_offset,
                "camera_e_to_ADU" : camera_conversion,
                "deskewed_voxel_size_um" : [z_downsample_level*pixel_size_um, pixel_size_um, pixel_size_um],
                "stage_y_flipped": stage_y_flipped,
                "stage_z_flipped": stage_z_flipped,
            }
    )

    if max_projection:
        update_global_metadata(
            ts_store = max_z_ts_store,
            global_metadata= {
                "image_mirror_step_um" : image_mirror_step_um,
                "raw_pixel_size_um" : pixel_size_um,
                "opm_tilt_deg" : opm_tilt_deg,
                "camera_corrected" : True,
                "camera_offset" : camera_offset,
                "camera_e_to_ADU" : camera_conversion,
                "deskewed_voxel_size_um" : [1, pixel_size_um, pixel_size_um],
                "stage_y_flipped": stage_y_flipped,
                "stage_z_flipped": stage_z_flipped,
                "flatfield_corrected": flatfield_after_deskew
            }
        )
        
    del deskewed, ts_write, ts_store
    if max_projection:
        del max_z_deskewed, ts_max_write, max_z_ts_store

    if flatfield_after_deskew and max_projection:
        # open deskewed datastore
        spec = {
            "driver" : "zarr3",
            "kvstore" : {
                "driver" : "file",
                "path" : str(max_z_output_path)
            }
        }
        max_z_datastore = ts.open(spec).result()

        flatfields = np.zeros((datastore.shape[2],datastore.shape[-2],datastore.shape[-1]),dtype=np.float32)
        sample_indices = np.random.choice(datastore.shape[1], size=50, replace=False)
        for chan_idx in range(datastore.shape[2]):
            temp_images = np.squeeze(datastore[:,:,sample_indices,:].read().result())
            basic = BaSiC(get_darkfield=False, get_flatfield=True)
            basic.autotune(temp_images, early_stop=True, n_iter=100)
            flatfields[chan_idx,:] = basic.flatfield

        ts_writes = []
        for pos_idx in tqdm(range(datastore.shape[1]),desc="pos"):
            for chan_idx in tqdm(range(datastore.shape[2]),desc="chan"):
                corrected_data = flatfield_correction(datastore[:, pos_idx, chan_idx, :].read().result(), flatfields[chan_idx, :])
                future = max_z_datastore[ :, pos_idx, chan_idx, : ].write(corrected_data)
                ts_writes.append(future)

        for ts_write in ts_writes:
            ts_write.result()
        
        flatfields_6d = np.expand_dims(flatfields, axis=(0, 1, 3))
        flatfields_6d_shape = [
            1,
            1,
            flatfields.shape[0],
            1,
            flatfields.shape[1],
            flatfields.shape[2]
        ]

        flatfields_output_path = root_path.parents[0] / Path(str(root_path.stem)+"_flatfields.zarr")
        flatfields_ts_store = create_via_tensorstore(flatfields_output_path,flatfields_6d_shape)
        flatfields_ts_store[0,0,:,0,:].write(flatfields).result()

        del flatfields, flatfields_6d, flatfields_ts_store, ts_writes
        
    if optimize_stage_pos:
        optimize_stage_positions(max_z_datastore)
        
        if display_max_projection:
            return

    if display_max_projection and max_projection:
        # open deskewed maximum z datastore
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
                    if optimize_stage_pos:
                        stage_position = stage_positions[pos_idx] # + optimized_stage_pos[pos_idx]
                    layer = viewer.add_image(
                        np.squeeze(datastore[time_idx,pos_idx,chan_idx,:].read().result()),
                        scale=[pixel_size_um,pixel_size_um],
                        translate=stage_position,
                        name = "p"+str(pos_idx).zfill(3)+"_c"+str(chan_idx),
                        blending="additive",
                        colormap=colormaps[chan_idx],
                        contrast_limits = [50,2000]
                    )
                    
                    channel_layers[chan_idx].append(layer)
                    
        for chan_idx in range(datastore.shape[2]):
            link_layers(channel_layers[chan_idx],("contrast_limits","gamma"))
            
        napari.run()

    elif display_full_volume:
        # open deskewed datastore
        spec = {
            "driver" : "zarr3",
            "kvstore" : {
                "driver" : "file",
                "path" : str(output_path)
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
                        datastore[time_idx,pos_idx,chan_idx,:],
                        scale=[z_downsample_level*pixel_size_um,pixel_size_um,pixel_size_um],
                        translate=stage_positions[pos_idx],
                        name = "p"+str(pos_idx).zfill(3)+"_c"+str(chan_idx),
                        blending="additive",
                        colormap=colormaps[chan_idx],
                        contrast_limits = [50,2000]
                    )
                    
                    channel_layers[chan_idx].append(layer)
                    
        for chan_idx in range(datastore.shape[2]):
            link_layers(channel_layers[chan_idx],("contrast_limits","gamma"))
            
        napari.run()

# entry for point for CLI        
def main():
    app()

if __name__ == "__main__":
    main()