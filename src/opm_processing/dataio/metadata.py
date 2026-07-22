"""Metadata tools for qi2lab OPM tensorstore datastores.

The instrument produces a zarr2 store
with full micro-manager metadata in the .zattrs. The processed data produces a zarr3 store
with minimal metadata, sufficient for downstream processing.

History:
---------
- **2025/03**: Updated for new qi2lab OPM processing pipeline.
"""

import numpy as np


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
    t_values = []
    for entry in data.get("frame_metadatas", []):
        try:
            t_values.append(entry["mda_event"]["index"]["t"])
        except (KeyError, TypeError):
            continue
    if len(t_values) == 0:
        # no valid t found
        return np.empty((0, 3), dtype=float)
    first_t = min(t_values)

    positions = []
    p_values = []
    first_t = 0
    for entry in data["frame_metadatas"]:
        try:
            idx = entry["mda_event"]["index"]
            if idx["t"] != first_t:
                # skip any frames not at t=first_t
                continue
            p = idx["p"]
            stage_data = entry["mda_event"]["metadata"]["Stage"]

            # Ensure valid stage data
            x_pos = stage_data["x_pos"]
            y_pos = stage_data["y_pos"]
            z_pos = stage_data["z_pos"]

            positions.append([z_pos, x_pos, y_pos])
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


def extract_channels(data: dict) -> list:
    """Extract channel names from qi2lab tensorstore metadata in order.

    Parameters
    ----------
    data : dict
        Metadata from .zattrs.

    Returns
    -------
    channels : list
        Ordered list of unique channel names.
    """
    seen = set()
    channels = []

    frame_metadatas = data.get("frame_metadatas", [])

    for frame in frame_metadatas:
        mda_event = frame.get("mda_event", {})
        metadata = mda_event.get("metadata", {})
        daq = metadata.get("DAQ", {})
        channel = daq.get("current_channel")

        if channel and channel not in seen:
            seen.add(channel)
            channels.append(channel)

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
