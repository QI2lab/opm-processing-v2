"""NOTE: This currently does not work with our TensorStore zarr stores."""

import json
import os
from pathlib import Path
import numpy as np
from math import isclose, copysign
import copy


def extract_tile_indices_and_signed_overlap(
    zarr_json_path: Path,
) -> tuple[dict[tuple[int, int], tuple[int, int, int]], tuple[float, float, float]]:
    """
    Assign (z, y, x) tile indices using stage_position in ZYX order,
    ensuring tile (0, 0) → (0, 0, 0), and compute signed overlaps
    using the smallest nonzero step per axis with preserved direction.

    Parameters
    ----------
    zarr_json_path : Path
        Path to the Zarr v3 `zarr.json`.

    Returns
    -------
    tuple[
        dict[(tile, channel), (z_idx, y_idx, x_idx)],
        tuple[float, float, float]
    ]
        Logical index mapping, and signed overlap (z, y, x) in microns.
    """

    with open(zarr_json_path, "r") as f:
        meta = json.load(f)

    metadata = meta["attributes"]["per_index_metadata"]["0"]
    shape = meta["shape"]
    z_size, y_size, x_size = shape[3], shape[4], shape[5]
    vz, vy, vx = meta["attributes"]["deskewed_voxel_size_um"]

    # Collect stage positions
    stage_list = []
    for tile_str, ch_dict in metadata.items():
        tile = int(tile_str)
        for ch_str, ch_meta in ch_dict.items():
            ch = int(ch_str)
            zyx = tuple(round(v, 5) for v in ch_meta["stage_position"])
            stage_list.append(((tile, ch), zyx))

    stage_list.sort()

    index_map: dict[tuple[int, int], tuple[int, int, int]] = {}
    origin = stage_list[0][1]
    z_vals, y_vals, x_vals = [origin[0]], [origin[1]], [origin[2]]
    steps = {"z": None, "y": None, "x": None}

    for key, zyx in stage_list:
        z, y, x = zyx

        # Find matching index or append new
        def get_or_add(val_list: list[float], val: float) -> int:
            for i, known in enumerate(val_list):
                if isclose(val, known, abs_tol=1e-4):
                    return i
            val_list.append(val)
            return len(val_list) - 1

        z_idx = get_or_add(z_vals, z)
        y_idx = get_or_add(y_vals, y)
        x_idx = get_or_add(x_vals, x)

        index_map[key] = (z_idx, y_idx, x_idx)

        # Record first step per axis
        if steps["z"] is None and len(z_vals) > 1:
            steps["z"] = z_vals[1] - z_vals[0]
        if steps["y"] is None and len(y_vals) > 1:
            steps["y"] = y_vals[1] - y_vals[0]
        if steps["x"] is None and len(x_vals) > 1:
            steps["x"] = x_vals[1] - x_vals[0]

    # Compute signed overlaps
    overlap_z = z_size * vz - abs(steps["z"]) if steps["z"] else 0.0
    overlap_y = y_size * vy - abs(steps["y"]) if steps["y"] else 0.0
    overlap_x = x_size * vx - abs(steps["x"]) if steps["x"] else 0.0

    if steps["z"]: overlap_z = copysign(overlap_z, steps["z"])
    if steps["y"]: overlap_y = copysign(overlap_y, steps["y"])
    if steps["x"]: overlap_x = copysign(overlap_x, steps["x"])

    return index_map, (np.round(overlap_z,2), np.round(overlap_y,2), np.round(overlap_x,2))

def create_correct_zarray(
    dtype: str,
    z_size: int,
    y_size: int,
    x_size: int,
    codecs: list,
    fill_value,
) -> dict:
    """
    Construct a valid .zarray metadata dict for a 3D symlinked Zarr v3 array
    using the original sharding configuration but rewriting the chunk shape.
    """
    zyx_shape = [z_size, y_size, x_size]
    zarray = {
        "zarr_format": 3,
        "shape": zyx_shape,
        "data_type": dtype,
        "chunk_grid": {
            "name": "regular",
            "configuration": {
                "chunk_shape": zyx_shape
            },
        },
        "chunk_key_encoding": {"name": "default"},
        "fill_value": fill_value,
        "node_type": "array",
    }

    new_codecs = copy.deepcopy(codecs)
    if new_codecs and new_codecs[0]["name"] == "sharding_indexed":
        new_codecs[0]["configuration"]["chunk_shape"] = zyx_shape
    zarray["codecs"] = new_codecs

    return zarray

def generate_symlinks(
    store_root: Path,
    output_root: Path,
    timestamp: str = "0000000msec_0000000000msecAbs",
) -> None:
    """
    Create standalone 3D Zarr folders compatible with PyPetaKit5D from a Zarr v3 store,
    using logical tile indices inferred from stage position.

    Parameters
    ----------
    store_root : Path
        Path to the Zarr store root (must contain `c/zarr.json` and chunk hierarchy).
    output_root : Path
        Output directory for the new `.zarr` folders.
    timestamp : str
        Constant timestamp string for folder names.
    """
    
    zarr_json_path = store_root / "zarr.json"

    index_map, overlaps_zyx_um = extract_tile_indices_and_signed_overlap(zarr_json_path)

    with open(zarr_json_path) as f:
        meta = json.load(f)

    shape = meta["shape"]  # [T, Tile, Channel, Z, Y, X]
    dtype = meta["data_type"]
    codecs = meta["codecs"]
    fill_value = meta.get("fill_value", 0)
    attr = meta["attributes"]["per_index_metadata"]

    t_len = shape[0]
    tile_len = shape[1]
    channel_len = shape[2]
    z_size, y_size, x_size = shape[3], shape[4], shape[5]

    for t in range(t_len):
        t_meta = attr[str(t)]

        for tile in range(tile_len):
            tile_meta = t_meta.get(str(tile), {})

            for ch in range(channel_len):
                ch_meta = tile_meta.get(str(ch))
                if not ch_meta:
                    continue

                channel_name = ch_meta.get("channel", f"ch{ch}")
                zyx = index_map.get((tile, ch))
                if zyx is None:
                    continue
                z_idx, y_idx, x_idx = zyx

                folder_name = (
                    f"Scan_Iter_{tile:04d}_0000_CamA_ch{ch}_CAM1_stack000_"
                    f"{channel_name}_{timestamp}_{x_idx:03d}x_{y_idx:03d}y_{z_idx:03d}z_{t:04d}t.zarr"
                )
                out_path = output_root / folder_name
                out_path.mkdir(parents=True, exist_ok=True)

                # Create .zarray for 3D slice
                zarray = create_correct_zarray(
                    dtype=dtype,
                    z_size=z_size,
                    y_size=y_size,
                    x_size=x_size,
                    codecs=codecs,
                    fill_value=fill_value,
                )

                with open(out_path / ".zarray", "w") as f:
                    json.dump(zarray, f)

                # Correct flat shard layout: symlink single directory "0" with index + data_*
                shard_source = store_root / "c" / str(t) / str(tile) / str(ch)
                shard_target = out_path / "0"

                if shard_target.exists() or shard_target.is_symlink():
                    shard_target.unlink()
                os.symlink(shard_source.resolve(), shard_target)
    
    return overlaps_zyx_um

def cleanup_symlinks(output_root: Path) -> None:
    """
    Remove `.zarray` files and symlinked chunk folders created for PyPetaKit5D export.

    This function:
    - Deletes `.zarray` files inside each `.zarr` folder
    - Unlinks any integer-named subfolders (e.g., `0/`, `1/`) that are symbolic links
    - Removes the `.zarr` folder itself if it becomes empty

    Parameters
    ----------
    output_root : Path
        Path to the directory containing the `*.zarr` folders created by `generate_pypetakit5d_folders`.

    Returns
    -------
    None
    """
    for zarr_dir in output_root.glob("*.zarr"):
        if not zarr_dir.is_dir():
            continue

        # Delete .zarray file if it exists
        zarray_path = zarr_dir / ".zarray"
        if zarray_path.exists():
            zarray_path.unlink()

        # Delete all symlinked chunk folders like 0/, 1/, ...
        for sub in zarr_dir.iterdir():
            if sub.is_symlink() and sub.name.isdigit():
                sub.unlink()

        # Optionally remove the folder if it's now empty
        try:
            zarr_dir.rmdir()
        except OSError:
            # Directory not empty (e.g., stray files remain)
            pass
