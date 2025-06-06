from basicpy import BaSiC
import builtins
import jax
import numpy as np
import gc

jax.config.update("jax_enable_compilation_cache", False)


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
    # flatfields shape: c, y, x
    flatfields = np.zeros((datastore.shape[2],datastore.shape[-2],datastore.shape[-1]),dtype=np.float32)
    
    # Define the number of z-images for flat field calculation
    if datastore.shape[-3] > 5000:
        n_image_batches = 25
        n_rand_images = 5000
    elif datastore.shape[-3]==1:
        n_image_batches = 1 
        n_rand_images = 1
    else:
        n_image_batches = 25
        n_rand_images = datastore.shape[-3]
        n_rand_images -= n_rand_images % n_image_batches
    n_images_to_max = n_rand_images // n_image_batches
        
    # Define the number of position samples for flat field calculation
    n_pos_samples = 15
    if datastore.shape[1] > n_pos_samples+5:
        flatfield_pos_iterator = list(np.random.choice(range(datastore.shape[1]//2-(n_pos_samples+5)//2,datastore.shape[1]//2+(n_pos_samples+5)//2), size=n_pos_samples, replace=False))
    else:
        n_pos_samples = datastore.shape[1]
        flatfield_pos_iterator = range(datastore.shape[1])
    
    for chan_idx in range(datastore.shape[2]):
        images = []
        for pos_idx in flatfield_pos_iterator:
            sample_indices = list(np.random.choice(datastore.shape[-3], size=n_rand_images, replace=False))
            try:
                temp_images = ((np.squeeze(datastore[0,pos_idx,chan_idx,sample_indices,:].read().result()).astype(np.float32)-camera_offset)*camera_conversion).clip(0,2**16-1).astype(np.uint16)
            except Exception:
                temp_images = (np.squeeze(np.asarray(datastore[0,pos_idx,chan_idx,sample_indices,:],dtype=np.float32)-camera_offset)*camera_conversion).clip(0,2**16-1).astype(np.uint16)
            temp_images = temp_images.reshape(n_image_batches, n_images_to_max, temp_images.shape[-2], temp_images.shape[-1])
            temp_images = np.squeeze(np.mean(temp_images,axis=1))
            images.append(temp_images)
        images = np.asarray(images,dtype=np.float32)
        images = images.reshape(n_pos_samples*n_image_batches,images.shape[-2],images.shape[-1])
        original_print = builtins.print
        builtins.print= no_op
        basic = BaSiC(
            get_darkfield=False,
            darkfield=np.zeros((temp_images.shape[-2]//4,temp_images.shape[-1]//4),dtype=np.float64),
            flatfield=np.zeros((temp_images.shape[-2]//4,temp_images.shape[-1]//4),dtype=np.float64)
        )
        basic.autotune(images)
        basic.fit(images)
        builtins.print = original_print
        flatfields[chan_idx,:] = np.squeeze(basic.flatfield).astype(np.float32)
        
        del basic, images, temp_images

        gc.collect()
        jax.clear_caches()
        gc.collect()

    return flatfields
