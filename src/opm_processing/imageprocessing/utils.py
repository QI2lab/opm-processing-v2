"""
Image processing functions for qi2lab 3D MERFISH.

This module includes various utilities for image processing, such as
downsampling, padding, and chunked GPU-based deconvolution.

History:
---------
- **2025/03**: Updated for new qi2lab OPM processing pipeline.
- **2024/12**: Refactored repo structure.
- **2024/07**: Added numba-accelerated downsampling, padding helper functions,
               and chunked GPU deconvolution.
"""

import numpy as np
import gc
import dask.array as da
import dask.diagnostics
from pathlib import Path
try:
    from multiview_stitcher import spatial_image_utils as si_utils
    from multiview_stitcher import msi_utils, registration, fusion, ngff_utils
except:
    pass
from numpy.typing import NDArray
from basicpy import BaSiC
import builtins

# GPU
CUPY_AVIALABLE = True
try:
    import cupy as cp  # type: ignore
except ImportError:
    xp = np
    CUPY_AVIALABLE = False
    from scipy import ndimage  # type: ignore
else:
    xp = cp
    from cupyx.scipy import ndimage  # type: ignore


def replace_hot_pixels(
    noise_map: NDArray, 
    data: NDArray, 
    threshold: float = 375.0
) -> NDArray:
    """Replace hot pixels with median values surrounding them.

    Parameters
    ----------
    noise_map: NDArray
        darkfield image collected at long exposure time to get hot pixels
    data: NDArray
        ND data [broadcast_dim,z,y,x]

    Returns
    -------
    data: NDArray
        hotpixel corrected data
    """

    data = xp.asarray(data, dtype=xp.float32)
    noise_map = xp.asarray(noise_map, dtype=xp.float32)

    # threshold darkfield_image to generate bad pixel matrix
    hot_pixels = xp.squeeze(xp.asarray(noise_map))
    hot_pixels[hot_pixels <= threshold] = 0
    hot_pixels[hot_pixels > threshold] = 1
    hot_pixels = hot_pixels.astype(xp.float32)
    inverted_hot_pixels = xp.ones_like(hot_pixels) - hot_pixels.copy()

    data = xp.asarray(data, dtype=xp.float32)
    for z_idx in range(data.shape[0]):
        median = ndimage.median_filter(data[z_idx, :, :], size=3)
        data[z_idx, :] = inverted_hot_pixels * data[z_idx, :] + hot_pixels * median

    data[data < 0] = 0

    if CUPY_AVIALABLE:
        data = xp.asnumpy(data).astype(np.uint16)
        gc.collect()
        cp.clear_memo()
        cp._default_memory_pool.free_all_blocks()
    else:
        data = data.astype(np.uint16)

    return data

def downsample_image_yx(image: NDArray, level: int = 2) -> NDArray:
    """2D plane downsampling.

    Parameters
    ----------
    image: NDArray
        3D image to be downsampled
    level: int
        isotropic downsampling level

    Returns
    -------
    downsampled_image: NDArray
        downsampled 3D image
    """

    downsampled_image = downsample_axis(
        downsample_axis(image, level, 1),
        level, 2
    )

    return downsampled_image

def downsample_image_isotropic(image: NDArray, level: int = 2) -> NDArray:
    """3D isotropic downsampling.

    Parameters
    ----------
    image: NDArray
        3D image to be downsampled
    level: int
        isotropic downsampling level

    Returns
    -------
    downsampled_image: NDArray
        downsampled 3D image
    """

    downsampled_image = downsample_axis(
        downsample_axis(
            downsample_axis(image, level, 0), 
            level, 1), 
        level, 2
    )

    return downsampled_image

def downsample_axis(image: NDArray, level: int = 2, axis: int = 0) -> NDArray:
    """Downsampling along a specified axis.
    
    Parameters
    ----------
    image: NDArray
        image to downsample.
    level: int
        integer amount to downsample.
    axis: int
        axis to apply to.
    
    Returns
    -------
    downsampled_image: NDArray
        downsampled image.    
    """
    
    # Compute new shape
    original_shape = image.shape
    new_size = original_shape[axis] // level
    new_shape = list(original_shape)
    new_shape[axis] = new_size

    # Trim excess values to ensure reshape works properly
    trim_size = new_size * level
    slicing = [slice(0, trim_size) if i == axis else slice(None) for i in range(image.ndim)]
    image_trimmed = image[tuple(slicing)]

    # Reshape and compute mean along the downsampled axis
    reshaped_shape = list(image_trimmed.shape)
    reshaped_shape[axis] = new_size
    reshaped_shape.insert(axis + 1, level)

    image_reshaped = image_trimmed.reshape(reshaped_shape)
    downsampled_image = image_reshaped.mean(axis=axis + 1)

    return downsampled_image.astype(image.dtype)

def create_fused_max_z_projection(
    ts_store, 
    ome_output_path: Path, 
    voxel_size_zyx_um: list[NDArray],
    stage_positions_zyx_um: list[NDArray],
    reg_axis: int = 0
):
    """Optimize stage positions and create fused maximum z projection.
    
    Parameters
    ----------
    ts_store: Tensorstore
        datastore
    ome_output_path: Path
        output path for max z projection fused ome.zarr
    voxel_size_zyx_um: NDArray
        voxel size of deskewed data in microns
    stage_positions_zyx_um: list[NDArray]
        stage position list of deskewed data in microns
    reg_axis : int, default = 0
        axis to use for registration
    """
    
    max_projection_data = np.squeeze(ts_store.read().result())
    
    msims = []
    scale = {"y": voxel_size_zyx_um[1], "x": voxel_size_zyx_um[2]}
    for pos_idx in range(max_projection_data.shape[0]):
        tile_grid_positions = {
            "y": np.round(stage_positions_zyx_um[pos_idx,1], 2),
            "x": np.round(stage_positions_zyx_um[pos_idx,2], 2),
        }
        
        sim = si_utils.get_sim_from_array(
            max_projection_data[pos_idx,:],
            dims=["c"] + list(scale.keys()),
            scale=scale,
            translation=tile_grid_positions,
            transform_key="stage_metadata"
        )
        
        msim = msi_utils.get_msim_from_sim(sim)
        msims.append(msim)
    
    print("Registering views...")
    with dask.diagnostics.ProgressBar():
        _ = registration.register(
            msims,
            registration_binning={'y': 3, 'x': 3},
            reg_channel_index=reg_axis,
            transform_key="stage_metadata",
            new_transform_key='affine_registered',
            pre_registration_pruning_method="keep_axis_aligned",
            post_registration_do_quality_filter=True
        )
    
    print("Building fusion graph...")
    fused = fusion.fuse(
        [msi_utils.get_sim_from_msim(msim) for msim in msims],
        transform_key='affine_registered',
        output_chunksize=256,
    )
    
    print(f'Fusing views and saving output to {str(ome_output_path)}...')
    with dask.diagnostics.ProgressBar():
        fused = ngff_utils.write_sim_to_ome_zarr(
            fused, ome_output_path, overwrite=True
        )
         
def no_op(*args, **kwargs):
    """Function to monkey patch print to suppress output.
    
    Parameters
    ----------
    args: Any
        positional arguments
    kwargs: Any
        keyword arguments
    """
    
    pass

def estimate_illuminations(datastore, camera_offset,camera_conversion):
    flatfields = np.zeros((datastore.shape[2],datastore.shape[-2],datastore.shape[-1]),dtype=np.float32)
    n_image_batches = 50
    if datastore.shape[-3] > 5000:
        n_rand_images = 5000
    else:
        n_rand_images = datastore.shape[-3]
        n_rand_images -= n_rand_images % n_image_batches
    n_images_to_max = n_rand_images // n_image_batches
    
    if datastore.shape[2] > 5:
        flatfield_pos_iterator = list(np.random.choice(datastore.shape[2], size=5, replace=False))
    else:
        flatfield_pos_iterator = range(datastore.shape[2])
    
    for pos_idx in flatfield_pos_iterator:
        for chan_idx in range(datastore.shape[2]):
            sample_indices = list(np.random.choice(datastore.shape[-3], size=n_rand_images, replace=False))
            temp_images = ((np.squeeze(datastore[0,0,chan_idx,sample_indices,:].read().result()).astype(np.float32)-camera_offset)*camera_conversion).clip(0,2**16-1).astype(np.uint16)
            temp_images = temp_images.reshape(n_image_batches, n_images_to_max, temp_images.shape[-2], temp_images.shape[-1])
            temp_images = np.squeeze(np.mean(temp_images,axis=1))
            original_print = builtins.print
            builtins.print= no_op
            basic = BaSiC(
                get_darkfield=False,
                darkfield=np.zeros((temp_images.shape[-2]//4,temp_images.shape[-1]//4),dtype=np.float64),
                flatfield=np.zeros((temp_images.shape[-2]//4,temp_images.shape[-1]//4),dtype=np.float64)
            )
            basic.autotune(temp_images)
            basic.fit(temp_images)
            builtins.print = original_print
            if pos_idx == 0:
                flatfields[chan_idx,:] = (np.squeeze(basic.flatfield) / np.max(np.squeeze(basic.flatfield),axis=(0,1))).astype(np.float32)
            else:
                new_flatfield = (np.squeeze(basic.flatfield) / np.max(np.squeeze(basic.flatfield),axis=(0,1))).astype(np.float32)
                flatfields[chan_idx,:] = (flatfields[chan_idx,:] + new_flatfield)/2
    else:
        flatfields = np.ones((datastore.shape[2],datastore.shape[-2],datastore.shape[-1]),dtype=np.float32)
        
    return flatfields