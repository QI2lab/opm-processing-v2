"""
Metadata tools for qi2lab OPM tensorstore datastores. The instrument produces a zarr2 store
with full micro-manager metadata in the .zattrs. The processed data produces a zarr3 store
with minimal metadata, sufficient for downstream processing.

History:
---------
- **2025/03**: Updated for new qi2lab OPM processing pipeline.
"""

import numpy as np
import json

def extract_stage_positions(data):
    """Extract stage positions from qi2lab tensostore.
    
    Extracts x_pos, y_pos, and z_pos from the 'Stage' metadata where 'p' values are unique.
    Returns a sorted array of positions in ascending order of 'p'.
    
    Parameters
    ----------
    data : dict
        metadata from .zattrs
        
    Returns
    -------
    stage_positions: np.ndarray
        stage positions in ascending order of position idx ("p")
    """
    
    positions = []
    p_values = []

    for entry in data["frame_metadatas"]:
        try:
            p = entry["mda_event"]["index"]["p"]
            stage_data = entry["mda_event"]["metadata"]["Stage"]

            # Ensure valid stage data
            x_pos = stage_data["x_pos"]
            y_pos = stage_data["y_pos"]
            z_pos = stage_data["z_pos"]

            positions.append([z_pos, y_pos, x_pos])
            p_values.append(p)
        except KeyError:
            continue  # Skip entries missing expected keys

    # Convert to numpy arrays
    positions = np.array(positions)
    p_values = np.array(p_values)

    # Get unique p values and their first occurrence indices
    _, unique_indices = np.unique(p_values, return_index=True)

    # Extract corresponding positions sorted by unique p values
    sorted_positions = positions[unique_indices]

    return sorted_positions

def extract_channels(data: dict, channels=None) -> list:
    """Extract channel names from qi2lab tensorstore metadata.

    Parameters
    ----------
    data : dict
        Metadata from .zattrs.
    channels : set, optional

    Returns
    -------
    channels : list
        Set of channel names.
    """

    if channels is None:
        channels = set()
        
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "current_channel":
                channels.add(value)
            else:
                extract_channels(value, channels)
    elif isinstance(data, list):
        for item in data:
            extract_channels(item, channels)
    return list(channels)


def find_key(data: dict, target_key: str) -> dict | list | None:
    """Recursively find the first occurrence of target_key in a nested structure.
    
    Parameters
    ----------
    data : dict
        Nested dictionary or list to search.
    target_key : str
        Key to search for.
        
    Returns
    -------
    key_data :
        dict | list | None
    """

    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                return value
            found = find_key(value, target_key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = find_key(item, target_key)
            if found is not None:
                return found
    return None

def update_global_metadata(ts_store, global_metadata):
    """Update global metadata inside a Zarr v3 array/group.
    
    Parameters
    ----------
    ts_store : tensorstore.TensorStore
        TensorStore object to update.
    global_metadata : dict
        Global metadata dictionary.
    """
    
    # Read existing metadata
    read_result = ts_store.kvstore.read("zarr.json").result()
    if read_result.state == 'missing':
        existing_metadata = {}
    else:
        existing_metadata = json.loads(read_result.value.decode("utf-8"))

    # Update global metadata
    global_metadata = convert_metadata(global_metadata)
    existing_metadata.setdefault("attributes", {}).update(global_metadata)

    # Write updated metadata back to zarr.json
    ts_store.kvstore.write("zarr.json", json.dumps(existing_metadata).encode("utf-8")).result()

def update_per_index_metadata(ts_store, metadata, index_location):
    """Update per-index metadata inside a Zarr v3 array/group.
    
    Parameters
    ----------
    ts_store : tensorstore.TensorStore
        TensorStore object to update.
    metadata : dict
        Metadata dictionary for this specific (t_idx, pos_idx, chan_idx).
    index_location : tuple
        Tuple of (t_idx, pos_idx, chan_idx).
    """

    # Read existing metadata
    read_result = ts_store.kvstore.read("zarr.json").result()
    if read_result.state == 'missing':
        existing_metadata = {}
    else:
        existing_metadata = json.loads(read_result.value.decode("utf-8"))

    # Ensure per-index metadata structure inside attributes
    attributes = existing_metadata.setdefault("attributes", {})
    per_index_metadata = attributes.setdefault("per_index_metadata", {})

    t_dict = per_index_metadata.setdefault(str(index_location[0]), {})
    pos_dict = t_dict.setdefault(str(index_location[1]), {})
    metadata = convert_metadata(metadata)
    pos_dict[str(index_location[2])] = metadata

    # Write updated metadata back to zarr.json
    ts_store.kvstore.write("zarr.json", json.dumps(existing_metadata).encode("utf-8")).result()

def convert_metadata(obj):
    """Ensure all metadata entries can be serialized.
    
    Parameters
    ----------
    obj: dict
        Metadata dict that may or may not serialize.
        
    Returns
    -------
    obj: dict
        Metadata dict that can be serialized.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_metadata(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_metadata(v) for v in obj]
    return obj