import acquire_zarr as aqz
from pathlib import Path
import tensorstore as ts
import numpy as np

def create_via_acquire(output_path,dimension_shape):
    settings = aqz.StreamSettings(
        store_path=str(output_path),
        data_type=aqz.DataType.UINT16,
        version=aqz.ZarrVersion.V3,
        multiscale=False
    )
    
    settings.compression = aqz.CompressionSettings(
        compressor=aqz.Compressor.BLOSC1,
        codec=aqz.CompressionCodec.BLOSC_LZ4,
        level=4,
        shuffle=2,
    )

    settings.dimensions.extend([
        aqz.Dimension(
            name="t",
            type=aqz.DimensionType.TIME,
            array_size_px=dimension_shape[0],
            chunk_size_px=1,
            shard_size_chunks=1
        ),
        aqz.Dimension(
            name="p",
            type=aqz.DimensionType.SPACE,
            array_size_px=dimension_shape[1],
            chunk_size_px=1,
            shard_size_chunks=1
        ),
        aqz.Dimension(
            name="c",
            type=aqz.DimensionType.CHANNEL,
            array_size_px=dimension_shape[2],
            chunk_size_px=1,
            shard_size_chunks=1
        ),
        aqz.Dimension(
            name="z",
            type=aqz.DimensionType.SPACE,
            array_size_px=dimension_shape[3],
            chunk_size_px=256,
            shard_size_chunks=1
        ),
        aqz.Dimension(
            name="y",
            type=aqz.DimensionType.SPACE,
            array_size_px=dimension_shape[4],
            chunk_size_px=dimension_shape[4]//4,
            shard_size_chunks=2
        ),
        aqz.Dimension(
            name="x",
            type=aqz.DimensionType.SPACE,
            array_size_px=dimension_shape[5],
            chunk_size_px=dimension_shape[5]//4,
            shard_size_chunks=2
        )
    ])
    stream = aqz.ZarrStream(settings)

    return stream

def write_via_acquire(stream,data):
    
    stream.append(data)
    
# ------------------------------------------------------------------------

def create_multiscale_dict(
    axes: dict, 
    base_scale: float, 
    top_translation: list, 
    channels: dict, 
    num_levels: int
) -> dict:
    """
    Build a multiscale dictionary where each successive level's scale is twice the previous one,
    and the translation for the y–x axes is adjusted to account for down-sampling.

    Parameters
    ----------
    axes : dict
        A dictionary defining the axes metadata. Example:
        {
            "t": {"type": "time"},
            "c": {"type": "channel"},
            "y": {"type": "space", "unit": "micrometer"},
            "x": {"type": "space", "unit": "micrometer"}
        }
    base_scale : float
        The scale factor for the top-level (level 0) image for the y and x axes.
    top_translation : list
        A two-element list representing the top-level (level 0) translation for y and x axes.
    channels : dict or list
        Channel definitions for the OMERO metadata. If provided as a dictionary, its values are used as the channel list.
    num_levels : int
        The total number of levels to generate. For each level i, the scale is computed as:
            base_scale * 2**i,
        and the translation for y and x is adjusted by an offset of:
            (2**i - 1) * (base_scale / 2).

    Returns
    -------
    dict
        A dictionary with the following structure:
        {
            "multiscales": [
                {
                    "axes": [
                        { "name": "t", ... },
                        { "name": "c", ... },
                        { "name": "y", ... },
                        { "name": "x", ... }
                    ],
                    "datasets": [
                        {
                            "coordinateTransformations": [
                                {
                                    "scale": [1.0, 1.0, <scale>, <scale>],
                                    "type": "scale"
                                },
                                {
                                    "translation": [0, 0, <adjusted_y>, <adjusted_x>],
                                    "type": "translation"
                                }
                            ],
                            "path": "<level>"
                        },
                        ...
                    ],
                    "name": "/",
                    "version": "0.4"
                }
            ],
            "omero": {
                "channels": [ ... ]
            }
        }
    """
    # Convert axes dictionary to a list of axis definitions.
    axes_list = []
    for axis_name, axis_info in axes.items():
        axis_entry = {"name": axis_name}
        axis_entry.update(axis_info)
        axes_list.append(axis_entry)
    
    datasets = []
    # Generate datasets for each level.
    for level in range(num_levels):
        # Compute the scale factor for the current level (for y and x axes).
        level_scale = base_scale * (2 ** level)
        # Compute the translation offset: (2**level - 1) * (base_scale / 2)
        offset = (2 ** level - 1) * (base_scale / 2)
        # Adjust the top-level translation for y and x.
        adjusted_translation_y = top_translation[0] + offset
        adjusted_translation_x = top_translation[1] + offset
        
        # Construct the full scale and translation lists, assuming fixed values for t and c.
        scale_list = [1.0, 1.0, level_scale, level_scale]
        translation_list = [0, 0, adjusted_translation_y, adjusted_translation_x]
        
        dataset = {
            "path": str(level),
            "coordinateTransformations": [
                {
                    "scale": scale_list,
                    "type": "scale"
                },
                {
                    "translation": translation_list,
                    "type": "translation"
                }
            ]
        }
        datasets.append(dataset)
    
    # Convert channels to a list if provided as a dictionary.
    if isinstance(channels, dict):
        channel_list = list(channels.values())
    else:
        channel_list = channels

    return {
        "multiscales": [
            {
                "axes": axes_list,
                "datasets": datasets,
                "name": "/",
                "version": "0.4"
            }
        ],
        "omero": {
            "channels": channel_list
        }
    }

def create_via_tensorstore(output_path: Path | str, data_shape: list[int]):
    """Create a TensorStore Zarr v3 driver with a 6D array.

    The configuration specifies:
      - A 6D array of shape `data_shape`.
      - Chunk sizes computed as [1, 1, 1, 1, data_shape[4] // 4, data_shape[5] // 4].
      - Compression using Blosc with zstd, compression level 5, and bitshuffle.
      - Shards defined as [1, 1, 1, data_shape[2], data_shape[3], data_shape[4]].

    Parameters
    ----------
    output_path : str or Path
        store location on disk
    data_shape : tuple or list of int
        A 6-element tuple or list representing the size of the 6D array (e.g., (time, pos, channel, z, y, x)).

    Returns
    -------
    ts_store
        tensorstore object
    """
    
    # Define chunk shape
    chunk_shape = [1, 1, 1, 1, data_shape[4]//4, data_shape[5]//4]

    # Define shard shape (each shard contains all of `y, x`)
    shard_shape = [1, 1, 1, data_shape[3], data_shape[4], data_shape[5]]  

    config = {
        "driver": "zarr3",
        "kvstore": {
            "driver": "file",
            "path": str(output_path)
        },
        "metadata": {
            "shape": data_shape,
            "chunk_grid": {
                "name": "regular",
                "configuration": {
                    "chunk_shape": shard_shape
                }
            },
            "chunk_key_encoding": {"name": "default"},
            "codecs": [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": chunk_shape,
                        "codecs": [
                            {"name": "bytes", "configuration": {"endian": "little"}},
                            {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 5, "shuffle": "bitshuffle"}}
                        ],
                        "index_codecs": [
                            {"name": "bytes", "configuration": {"endian": "little"}},
                            {"name": "crc32c"}
                        ],
                        "index_location": "end"
                    }
                }
            ],
            "data_type": "uint16"
        }
    }

    ts_store = ts.open(config, create=True, open=True).result()
    
    return ts_store

def write_via_tensorstore(ts_store, data, data_location):
    """Write a 3D data block to a tensorstore at a specific location in the first three dimensions.

    The tensorstore is assumed to be a 6D array (for example, with dimensions 
    [time, pos, channel, z, y, x]), where the first three dimensions (indices 0, 1, 2) are specified 
    by data_location and the data block corresponds to the last three dimensions (indices 3, 4, 5).

    Parameters
    ----------
    ts_store : TensorStore
        An open TensorStore object.
    data : numpy.ndarray
        A 3D NumPy array whose shape matches the size of the tensorstore in its last three dimensions.
    data_location : list of int
        A list of three integers specifying the indices for the first three dimensions of the tensorstore 
        at which to write the data block.

    Returns
    -------
    Future
        A future representing the asynchronous write operation.
    """
    
    # Construct the indexing tuple by combining the fixed indices from data_location
    # with full slices for the last three dimensions.
    index = tuple(data_location) + (slice(None), slice(None), slice(None))
    
    # Initiate the write operation (this returns a Future).
    future = ts_store[index].write(data)
    return future