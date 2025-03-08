import numpy as np
import tensorstore as ts

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

def extract_channels(data: dict) -> set:
    """Extract channel names from qi2lab tensorstore metadata.

    Parameters
    ----------
    data : dict
        Metadata from .zattrs.
    channels : set, optional

    Returns
    -------
    channels : set
        Set of channel names.
    """

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
    return channels


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
    """Update global metadata in the root Zarr metadata.
    
    Parameters
    ----------
    ts_store : tensorstore.TensorStore
        TensorStore object to update.
    global_metadata : dict
        New global metadata to add or update.
    """

    spec = ts_store.spec()
    spec.update({"metadata": global_metadata})
    ts_store = ts.open(spec).result()  # Apply the update

def update_per_index_metadata(ts_store, metadata, index_location):
    """Update metadata for a specific (t_idx, pos_idx, chan_idx).
    
    Parameters
    ----------
    ts_store : tensorstore.TensorStore
        TensorStore object to update.
    metadata : dict
        New metadata to add or update.
    index_location : tuple
        Tuple of (t_idx, pos_idx, chan_idx) to update.
    """
    
    spec = ts_store.spec().to_json()
    existing_metadata = spec.get("metadata", {})

    per_index_metadata = existing_metadata.setdefault("per_index_metadata", {})
    t_dict = per_index_metadata.setdefault(str(index_location[0]), {})
    pos_dict = t_dict.setdefault(str(index_location[1]), {})
    pos_dict[str(index_location[2])] = metadata
    
    spec.update({"metadata": existing_metadata})
    ts_store = ts.open(spec).result()  # Apply the update