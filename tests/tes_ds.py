import tensorstore as ts
from pathlib import Path
import math
from tqdm import trange

def create_fused_tensorstore(
    output_path: str | Path,
    padded_shape,
    chunk_shape,
    z_slices_per_shard: int = 4
) -> ts.TensorStore:
    """
    Create the output TensorStore with sharded Z-slices and chunked Y/X.
    """
    full_shape = [1, 1, 1, *padded_shape]
    shard_chunk = [1, 1, 1, z_slices_per_shard, padded_shape[1], padded_shape[2]]
    codec_chunk = [1, 1, 1, z_slices_per_shard, chunk_shape[0], chunk_shape[1]]
    config = {
        "context": {
            "file_io_concurrency": {"limit": 4},
            "data_copy_concurrency": {"limit": 4},
            "file_io_memmap": True,
            "file_io_sync": False,
        },
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(output_path)},
        "metadata": {
            "shape": full_shape,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shard_chunk}},
            "chunk_key_encoding": {"name": "default"},
            "codecs": [{
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": codec_chunk,
                    "codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 5, "shuffle": "bitshuffle"}}
                    ],
                    "index_codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "crc32c"}
                    ],
                    "index_location": "end",
                }
            }],
            "data_type": "uint16",
            "dimension_names": "tpczyx"
        }
    }
    return ts.open(config, create=True, open=True).result()

def downsample_shards(factor=2, z_slices_per_shard=4):
    root = Path("/mnt/data2/qi2lab/20250513_human_OB/whole_OB_slice_polya.zarr/")
    inp_path = root.parents[0] / f"{root.stem}_fused_deskewed.zarr"
    out_path = root.parents[0] / f"{root.stem}_fused_deskewed.ome.zarr/1/"

    # Open input store
    inp = ts.open({
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(inp_path)}
    }).result()

    # Determine shapes
    _, _, _, Z, Y, X = inp.spec().shape
    new_z, new_y, new_x = Z // factor, Y // factor, X // factor

    # Compute Y/X chunk sizes via gcd
    g = math.gcd(new_y, new_x)
    div = next((d for d in range(2, int(math.isqrt(g)) + 1) if g % d == 0), g if g > 1 else 1)
    chunk_y, chunk_x = new_y // div, new_x // div

    # Create output store
    out = create_fused_tensorstore(
        output_path=out_path,
        padded_shape=(new_z, new_y, new_x),
        chunk_shape=(chunk_y, chunk_x),
        z_slices_per_shard=z_slices_per_shard
    )

    # Back-calculate and stream each shard
    for z0 in trange(0, new_z, z_slices_per_shard, desc="downsample"):
        block_z = min(z_slices_per_shard, new_z - z0)
        in_z0 = z0 * factor
        in_z1 = in_z0 + block_z * factor

        for y0 in range(0, new_y, chunk_y):
            block_y = min(chunk_y, new_y - y0)
            in_y0 = y0 * factor
            in_y1 = in_y0 + block_y * factor

            for x0 in range(0, new_x, chunk_x):
                block_x = min(chunk_x, new_x - x0)
                in_x0 = x0 * factor
                in_x1 = in_x0 + block_x * factor

                # Load exactly the needed input region
                slab = inp[
                    :, :, :,
                    in_z0:in_z1,
                    in_y0:in_y1,
                    in_x0:in_x1
                ].read().result()

                # Downsample by striding
                down = slab[
                    ..., ::factor, ::factor, ::factor
                ]

                # Write to output shard
                out[
                    :, :, :,
                    z0:z0 + block_z,
                    y0:y0 + block_y,
                    x0:x0 + block_x
                ].write(down).result()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn")
    downsample_shards(factor=2, z_slices_per_shard=4)

