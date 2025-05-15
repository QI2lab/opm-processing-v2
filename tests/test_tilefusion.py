from opm_processing.imageprocessing.tilefusion import TileFusion
from pathlib import Path
import json
import tensorstore as ts
from tqdm import trange
import numpy as np

root_path = Path("/mnt/data2/qi2lab/20250513_human_OB/whole_OB_slice_polya.zarr/")

deskewed_data_path = root_path.parents[0] / Path(str(root_path.stem)+"_decon_deskewed.zarr")
if not(deskewed_data_path.exists()):
    deskewed_data_path = root_path.parents[0] / Path(str(root_path.stem)+"_deskewed.zarr")
    if not(deskewed_data_path.exists()):
        raise "Deskew data first."

spec = {
    "driver" : "zarr3",
    "kvstore" : {
        "driver" : "file",
        "path" : str(deskewed_data_path)
    }
}
datastore = ts.open(spec).result()

attrs_json = deskewed_data_path / Path("zarr.json")
with open(attrs_json, 'r') as file:
    metadata = json.load(file)
    
tile_positions = []
for time_idx in trange(datastore.shape[0],desc="t",leave=False):
    for pos_idx in trange(datastore.shape[1],desc="p",leave=False):
        tile_positions.append(metadata['attributes']['per_index_metadata'][str(time_idx)][str(pos_idx)]['0']['stage_position'])

output_path = root_path.parents[0] / Path(str(root_path.stem)+"_fused_deskewed.zarr")

tile_fuser = TileFusion(
    deskewed_data_path = deskewed_data_path,
    tile_positions = tile_positions,
    output_path = output_path,
    pixel_size = metadata['attributes']['deskewed_voxel_size_um'],
    blend_pixels= [10,400,400],
    debug=False
)
tile_fuser.refine_tile_positions_with_cross_correlation(
    downsample_factors=[3,5,5]
)
tile_fuser.optimize_shifts()

for idx, off in enumerate(tile_fuser.global_offsets):
    tile_fuser.tile_positions[idx] += off * np.array(
        tile_fuser.pixel_size
    )
tile_fuser._compute_fused_image_space()
tile_fuser._pad_to_tile_multiple()
tile_fuser._create_fused_tensorstore()
tile_fuser._fuse_tiles()