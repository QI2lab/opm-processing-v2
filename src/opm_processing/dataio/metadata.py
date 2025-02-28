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