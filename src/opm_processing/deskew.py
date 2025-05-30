"""
Deskew qi2lab OPM data.

This file deskews and creates maximum projections of raw qi2lab OPM data.
"""

import multiprocessing as mp
mp.set_start_method('forkserver', force=True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)

from pathlib import Path
import tensorstore as ts
from opm_processing.imageprocessing.opmpsf import generate_skewed_psf
from opm_processing.imageprocessing.rlgc import chunked_rlgc
from opm_processing.imageprocessing.opmtools import orthogonal_deskew, deskew_shape_estimator
from opm_processing.imageprocessing.maxtilefusion import TileFusion
from opm_processing.dataio.metadata import extract_channels, find_key, extract_stage_positions, update_global_metadata, update_per_index_metadata
from opm_processing.dataio.zarr_handlers import create_via_tensorstore, write_via_tensorstore
import json
import numpy as np
from tqdm import tqdm
import typer
from tifffile import TiffWriter, imread

app = typer.Typer()
app.pretty_exceptions_enable = False

@app.command()
def deskew(
    root_path: Path,
    deconvolve: bool = False,
    max_projection: bool = True,
    flatfield_correction: bool = False,
    create_fused_max_projection: bool = True,
    write_fused_max_projection_tiff: bool = False,
    z_downsample_level: int = 2,
    time_range: tuple[int,int] = None,
    pos_range: tuple[int,int] = None,    
):
    """Postprocess qi2lab OPM dataset.
    
    This code assumes data is generated by opm-v2 GUI and the resulting data is \
    saved using OPMMirrorHandler. All revelant metadata is read from imaging \
    files, including stage transformation, camera parameters, and channels.
    
    Usage: `deskew "/path/to/qi2lab_acquisition.zarr"`
    
    See docstring for the various options available.
    
    Outputs are in:
    - Deskewed 3D individual deskewed tiles: \
        `"/path/to/qi2lab_acquisition_deskewed.zarr"`
    - Maximum Z projected individual deskewed tiles: \
        `"/path/to/qi2lab_acquisition_max_z_deskewed.zarr"`
    - Maximum Z projection fused deskewed tiles: \
        `"/path/to/qi2lab_acquisition_max_z_fused.zarr"`
    
    Parameters
    ----------
    root_path: Path
        Path to OPM pymmcoregui zarr file.
    deconvolve: bool, default = False
        Deconvolve the data using RLGC.
    max_projection: bool, default = True
        Create a maximum projection datastore.
    flatfield_correction: bool, default = True
        Estimate and apply flatfield correction on raw data.
    create_fused_max_projection: bool, default = True
        Create stage position fused max Z projection.
    write_fused_max_projection_tiff: bool, default = False
        Write fused maxZ  projection to OME-TIFF file.
    z_downsample_level: int, default = 2
        Amount to downsample deskewed data in z.
    time_range: list[int,int], default = None
        Range of timepoints to reconstruct.
    pos_range: list[int,int], default = None
        Range of stage positions to reconstruct.     
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

    opm_mode = str(find_key(zattrs, "mode"))
    if "mirror" in opm_mode:
        scan_axis_step_um = float(find_key(zattrs,"image_mirror_step_um"))
        excess_scan_positions = 0
    elif "stage" in opm_mode:
        scan_axis_step_um = float(find_key(zattrs,"scan_axis_step_um"))
        excess_scan_positions = int(find_key(zattrs,"excess_scan_positions"))
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
    
    # estimate shape of one deskewed volume
    deskewed_shape, pad_y, pad_x = deskew_shape_estimator(
        [datastore.shape[-3]-excess_scan_positions,datastore.shape[-2],datastore.shape[-1]],
        theta=opm_tilt_deg,
        distance=scan_axis_step_um,
        pixel_size=pixel_size_um
    )

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
    if not(deconvolve):
        output_path = root_path.parents[0] / Path(str(root_path.stem)+"_deskewed.zarr")
    else:
        output_path = root_path.parents[0] / Path(str(root_path.stem)+"_decon_deskewed.zarr")
    ts_store = create_via_tensorstore(output_path,datastore_shape)
    
    if max_projection:
        max_z_datastore_shape = [
            time_shape,
            pos_shape,
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
        if not(deconvolve):
            max_z_output_path = root_path.parents[0] / Path(str(root_path.stem)+"_max_z_deskewed.zarr")
        else:
            max_z_output_path = root_path.parents[0] / Path(str(root_path.stem)+"_max_z_decon_deskewed.zarr")
        max_z_ts_store = create_via_tensorstore(max_z_output_path,max_z_datastore_shape)
        
    
    if flatfield_correction:
        
        flatfield_path = root_path.parents[0] / Path(str(root_path.stem)+"_flatfield.ome.tif")
        if flatfield_path.exists():
            flatfields = imread(flatfield_path).astype(np.float32)
        else:
            flatfields = call_estimate_illuminations(datastore, camera_offset, camera_conversion)
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
        
    # loop over all components and stream to zarr using tensorstore
    ts_writes = []
    if max_projection:
        ts_max_writes = []
        
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
                camera_corrected_data = (((np.squeeze(datastore[t_idx,pos_idx,chan_idx,:].read().result()).astype(np.float32)-camera_offset)*camera_conversion)/flatfields[chan_idx,:].astype(np.float32)).clip(0,2**16-1).astype(np.uint16)
                if "stage" in opm_mode:
                    flip_scan = True
                else:
                    flip_scan = False
                
                if flip_scan:
                    camera_corrected_data = np.flip(camera_corrected_data,axis=0)
                    
                if deconvolve:
                    if pos_idx == 0 and chan_idx == 0:
                        psfs = []
                        for psf_idx in range(datastore.shape[2]):
                            psf = generate_skewed_psf(
                                em_wvl=float(int(str(channels[psf_idx]).rstrip("nm")) / 1000),
                                pixel_size_um=pixel_size_um,
                                scan_axis_step_um=scan_axis_step_um,
                                pz=0.0,
                                plot=False
                            )
                            psfs.append(psf)

                    # This code is for debugging RLGC deconvolution
                    # ------------------------------------
                    # decon_temp = chunked_rlgc(
                    #     camera_corrected_data, 
                    #     psf,
                    #     scan_chunk_size=384,
                    #     scan_overlap_size=64
                    # )
                    
                    # import napari
                    # viewer = napari.Viewer()
                    # viewer.add_image(decon_temp)
                    # viewer.add_image(camera_corrected_data)
                    # napari.run()
                    # ------------------------------------
                    if camera_corrected_data.shape[1]==256:
                        chunk_size = 384
                    elif camera_corrected_data.shape[1]==512:
                        chunk_size = 196
                    deconvolved_data = chunked_rlgc(
                        camera_corrected_data[excess_scan_positions:,:,:],
                        np.asarray(psfs[chan_idx]),
                        scan_chunk_size=chunk_size,
                        scan_overlap_size=32
                    )
                    
                    deskewed = orthogonal_deskew(
                        deconvolved_data,
                        theta = opm_tilt_deg,
                        distance = scan_axis_step_um,
                        pixel_size = pixel_size_um
                    )
                else:        
                    deskewed = orthogonal_deskew(
                        camera_corrected_data[excess_scan_positions:,:,:],
                        theta = opm_tilt_deg,
                        distance = scan_axis_step_um,
                        pixel_size = pixel_size_um
                    )

              
                update_per_index_metadata(
                    ts_store = ts_store, 
                    metadata = {"stage_position": stage_positions[pos_idx], 'channel': channels[chan_idx]}, 
                    index_location = (t_idx, pos_idx, chan_idx)
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
    for ts_write in ts_writes:
        ts_write.result()

    if max_projection:
        for ts_max_write in ts_max_writes:
            ts_max_write.result()

    if "mirror" in opm_mode:
        update_global_metadata(
            ts_store = ts_store,
            global_metadata= {
                    "scan_axis_step_um" : scan_axis_step_um,
                    "raw_pixel_size_um" : pixel_size_um,
                    "opm_tilt_deg" : opm_tilt_deg,
                    "camera_corrected" : True,
                    "camera_offset" : camera_offset,
                    "camera_e_to_ADU" : camera_conversion,
                    "deskewed_voxel_size_um" : [z_downsample_level*pixel_size_um, pixel_size_um, pixel_size_um],
                    "stage_x_flipped": stage_x_flipped,
                    "stage_y_flipped": stage_y_flipped,
                    "stage_z_flipped": stage_z_flipped,
                    "flatfield_corrected": flatfield_correction
                }
        )
    elif "stage" in opm_mode:
        update_global_metadata(
            ts_store = ts_store,
            global_metadata= {
                    "scan_axis_step_um" : scan_axis_step_um,
                    "raw_pixel_size_um" : pixel_size_um,
                    "opm_tilt_deg" : opm_tilt_deg,
                    "camera_corrected" : True,
                    "camera_offset" : camera_offset,
                    "camera_e_to_ADU" : camera_conversion,
                    "deskewed_voxel_size_um" : [z_downsample_level*pixel_size_um, pixel_size_um, pixel_size_um],
                    "stage_x_flipped": stage_x_flipped,
                    "stage_y_flipped": stage_y_flipped,
                    "stage_z_flipped": stage_z_flipped,
                    "flatfield_corrected": flatfield_correction
                }
        )
    if max_projection:
        if "mirror" in opm_mode:
            update_global_metadata(
                ts_store = max_z_ts_store,
                global_metadata= {
                    "scan_axis_step_um" : scan_axis_step_um,
                    "raw_pixel_size_um" : pixel_size_um,
                    "opm_tilt_deg" : opm_tilt_deg,
                    "camera_corrected" : True,
                    "camera_offset" : camera_offset,
                    "camera_e_to_ADU" : camera_conversion,
                    "deskewed_voxel_size_um" : [pixel_size_um, pixel_size_um],
                    "stage_x_flipped": stage_x_flipped,
                    "stage_y_flipped": stage_y_flipped,
                    "stage_z_flipped": stage_z_flipped,
                    "flatfield_corrected": flatfield_correction
                }
            )
        elif "stage" in opm_mode:
            update_global_metadata(
                ts_store = max_z_ts_store,
                global_metadata= {
                    "scan_axis_step_um" : scan_axis_step_um,
                    "raw_pixel_size_um" : pixel_size_um,
                    "opm_tilt_deg" : opm_tilt_deg,
                    "camera_corrected" : True,
                    "camera_offset" : camera_offset,
                    "camera_e_to_ADU" : camera_conversion,
                    "deskewed_voxel_size_um" : [pixel_size_um, pixel_size_um],
                    "stage_x_flipped": stage_x_flipped,
                    "stage_y_flipped": stage_y_flipped,
                    "stage_z_flipped": stage_z_flipped,
                    "flatfield_corrected": flatfield_correction
                }
            )
            
    del deskewed, ts_write, ts_store
    if max_projection:
        del max_z_deskewed, ts_max_write
        
    if create_fused_max_projection:
        max_z_output_path = root_path.parents[0] / Path(str(root_path.stem)+"_max_z_deskewed.zarr")
        # open datastore on disk
        spec = {
            "driver" : "zarr3",
            "kvstore" : {
                "driver" : "file",
                "path" : str(max_z_output_path)
            }
        }
        max_z_ts_store = ts.open(spec).result()
        
        print("\nFusing max projection using stage positions...")
        fused_output_path = root_path.parents[0] / Path(str(root_path.stem)+"_max_z_fused.zarr")
        
        if pos_range is not None:
            tile_positions = stage_positions[pos_range[0]:pos_range[1],1:]
            
        else:
            tile_positions = stage_positions[:,1:]
        
        tile_fusion = TileFusion(
            ts_dataset = max_z_ts_store,
            tile_positions = tile_positions,
            output_path=fused_output_path,
            pixel_size=np.asarray((pixel_size_um,pixel_size_um),dtype=np.float32),
        )
        tile_fusion.run()
        
        if write_fused_max_projection_tiff:
            tiff_dir_path = max_z_output_path.parent / Path("fused_max_projection_tiff_output")
            tiff_dir_path.mkdir(exist_ok=True)
            max_spec = {
                "driver" : "zarr3",
                "kvstore" : {
                    "driver" : "file",
                    "path" : str(fused_output_path)
                }
            }
            max_proj_datastore = ts.open(max_spec).result()
            for t_idx in tqdm(range(max_proj_datastore.shape[0]),desc="t"):
                max_projection = np.squeeze(np.asarray(max_proj_datastore[t_idx,0,chan_idx,:].read().result()))
                
                filename = Path(f"fused_z_max_projection_t{t_idx}.ome.tiff")
                filename_path = tiff_dir_path /  Path(filename)
                if len(max_projection.shape) == 2:
                    axes = "YX"
                else:
                    axes = "CYX"
                
                with TiffWriter(filename_path, bigtiff=True) as tif:
                    metadata={
                        'axes': axes,
                        'SignificantBits': 16,
                        'PhysicalSizeX': pixel_size_um,
                        'PhysicalSizeXUnit': 'µm',
                        'PhysicalSizeY': pixel_size_um,
                        'PhysicalSizeYUnit': 'µm',
                    }
                    options = dict(
                        compression='zlib',
                        compressionargs={'level': 8},
                        predictor=True,
                        photometric='minisblack',
                        resolutionunit='CENTIMETER',
                    )
                    tif.write(
                        max_projection,
                        resolution=(
                            1e4 / pixel_size_um,
                            1e4 / pixel_size_um
                        ),
                        **options,
                        metadata=metadata
                    )

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

# entry for point for CLI        
def main():
    app()

if __name__ == "__main__":
    main()