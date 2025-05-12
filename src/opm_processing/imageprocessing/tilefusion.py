import numpy as np
import tensorstore as ts
from pathlib import Path
from tqdm import tqdm
from skimage.registration import phase_cross_correlation

class TileFusion3D:
    """
    A class for 3D (ZYX) fusion of overlapping image tiles in a TensorStore dataset.

    Parameters
    ----------
    ts_dataset : str or Path
        Path to input TensorStore (shape: T, N, C, Z, Y, X).
    tile_positions : list of tuple of float
        Tile origins in physical units [(z, y, x), ...].
    output_path : str or Path
        Path for fused output TensorStore.
    pixel_size : tuple of float
        Voxel size (z, y, x) in physical units.
    """
    def __init__(
        self,
        ts_dataset: str | Path,
        tile_positions: list[tuple[float, float, float]],
        output_path: str | Path,
        pixel_size: tuple[float, float, float]
    ):
        # Open input lazily
        self.ts_dataset = ts.open(str(ts_dataset), open=True).result()
        # Tile origins in physical (z, y, x)
        self.tile_positions = np.array(tile_positions, dtype=float)
        self.output_path = Path(output_path)
        self.pixel_size = pixel_size  # (z, y, x)

        # Unpack dims: T, N, C, Z, Y, X
        self.time_dim, self.position_dim, self.channels, self.z_dim, H, W = self.ts_dataset.shape
        self.tile_shape = (self.z_dim, H, W)

        # 2D blending mask (Y,X)
        self.weight_mask = self._generate_blending_weights()

        # Placeholders for fused volume
        self.offset = None           # (min_z, min_y, min_x)
        self.unpadded_shape = None   # (Z, Y, X)
        self.padded_shape = None     # (Z_pad, Y_pad, X_pad)
        self.pad = (0, 0, 0)         # (pad_z, pad_y, pad_x)
        self.fused_ts = None

    def _compute_fused_image_space(self):
        # Compute physical extents across z, y, x
        min_z = np.min(self.tile_positions[:, 0])
        min_y = np.min(self.tile_positions[:, 1])
        min_x = np.min(self.tile_positions[:, 2])
        max_z = np.max(self.tile_positions[:, 0]) + self.tile_shape[0] * self.pixel_size[0]
        max_y = np.max(self.tile_positions[:, 1]) + self.tile_shape[1] * self.pixel_size[1]
        max_x = np.max(self.tile_positions[:, 2]) + self.tile_shape[2] * self.pixel_size[2]

        # Convert to voxel counts
        size_z = int(np.ceil((max_z - min_z) / self.pixel_size[0]))
        size_y = int(np.ceil((max_y - min_y) / self.pixel_size[1]))
        size_x = int(np.ceil((max_x - min_x) / self.pixel_size[2]))

        return (size_z, size_y, size_x), (min_z, min_y, min_x)

    def _pad_to_tile_multiple(self, shape: tuple[int, int, int]):
        tz, th, tw = self.tile_shape
        size_z, size_y, size_x = shape
        pad_z = (-size_z) % tz
        pad_y = (-size_y) % th
        pad_x = (-size_x) % tw
        return (size_z + pad_z, size_y + pad_y, size_x + pad_x), (pad_z, pad_y, pad_x)

    def _generate_blending_weights(self, blend_pixels: tuple[int, int] = (200, 400)) -> np.ndarray:
        # 2D cosine feather in Y and X
        H, W = self.tile_shape[1:]
        by, bx = min(blend_pixels[0], H//2), min(blend_pixels[1], W//2)
        wy = np.ones(H, dtype=np.float32)
        wx = np.ones(W, dtype=np.float32)
        zy = np.linspace(0, np.pi, by)
        zx = np.linspace(0, np.pi, bx)
        fy = 0.5 * (1 - np.cos(zy))
        fx = 0.5 * (1 - np.cos(zx))
        wy[:by], wy[-by:] = fy, fy[::-1]
        wx[:bx], wx[-bx:] = fx, fx[::-1]
        return np.outer(wy, wx)

    def _create_fused_tensorstore(self):
        # Chunk/shard: full tile blocks in ZYX
        tz, th, tw = self.tile_shape
        chunk = [1, 1, self.channels, tz, th // 4, tw // 4]
        shard = [1, 1, self.channels, tz, th, tw]
        config = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(self.output_path)},
            "metadata": {
                "shape": [1, 1, self.channels, *self.padded_shape],
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shard}},
                "chunk_key_encoding": {"name": "default"},
                "codecs": [{
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": chunk,
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
                }],
                "data_type": "uint16"
            }
        }
        return ts.open(config, create=True, open=True).result()

    def _fuse_tiles(self):
        wm_full = self.weight_mask  # (Y, X)
        for idx, (z_phys, y_phys, x_phys) in enumerate(tqdm(self.tile_positions, desc="Fusing")):
            # Compute start indices in Z, Y, X
            z0 = int((z_phys - self.offset[0]) / self.pixel_size[0])
            y0 = int((y_phys - self.offset[1]) / self.pixel_size[1])
            x0 = int((x_phys - self.offset[2]) / self.pixel_size[2])
            z1 = z0 + self.tile_shape[0]
            y1 = y0 + self.tile_shape[1]
            x1 = x0 + self.tile_shape[2]

            # Read tile block (C, Z, Y, X)
            tile = self.ts_dataset[0, idx, :, :, :, :].read().result().astype(np.float32)

            # Read existing fused block
            block = self.fused_ts[0, 0, :, z0:z1, y0:y1, x0:x1].read().result().astype(np.float32)

            # Determine overlap lengths
            _, z_len, y_len, x_len = block.shape
            if z_len == 0 or y_len == 0 or x_len == 0:
                # no overlap, skip
                continue

            # Crop tile to actual overlap
            tile_crop = tile[:, :z_len, :y_len, :x_len]

            # Crop weight mask if needed
            if (y_len, x_len) != wm_full.shape:
                wm = wm_full[:y_len, :x_len]
            else:
                wm = wm_full
            wmb = wm[np.newaxis, np.newaxis, :, :]

            # Blend
            weighted = tile_crop * wmb
            block += weighted
            weight_sum = np.ones_like(block)
            weight_sum += wmb
            weight_sum[weight_sum == 0] = 1
            fused = block / weight_sum

            # Write back
            out = np.clip(fused, 0, 65535).astype(np.uint16)
            self.fused_ts[0, 0, :, z0:z0+z_len, y0:y0+y_len, x0:x0+x_len].write(out).result()

    def refine_tile_positions_with_cross_correlation(self, upsample_factor: int = 10) -> dict:
        """
        Refine tile origins by phase cross-correlation on overlapping patches.
        Adjust in Z, Y, X if necessary.
        """
        shifts = {}
        N = len(self.tile_positions)
        for i in range(N):
            for j in range(i + 1, N):
                dz_phys = self.tile_positions[j,0] - self.tile_positions[i,0]
                dy_phys = self.tile_positions[j,1] - self.tile_positions[i,1]
                dx_phys = self.tile_positions[j,2] - self.tile_positions[i,2]
                dz = int(round(dz_phys / self.pixel_size[0]))
                dy = int(round(dy_phys / self.pixel_size[1]))
                dx = int(round(dx_phys / self.pixel_size[2]))

                # Calculate overlap boxes
                z1_i, z2_i = max(0, dz), min(self.tile_shape[0], dz + self.tile_shape[0])
                y1_i, y2_i = max(0, dy), min(self.tile_shape[1], dy + self.tile_shape[1])
                x1_i, x2_i = max(0, dx), min(self.tile_shape[2], dx + self.tile_shape[2])
                z1_j, z2_j = max(0, -dz), min(self.tile_shape[0], self.tile_shape[0] - dz)
                y1_j, y2_j = max(0, -dy), min(self.tile_shape[1], self.tile_shape[1] - dy)
                x1_j, x2_j = max(0, -dx), min(self.tile_shape[2], self.tile_shape[2] - dx)
                if z2_i <= z1_i or y2_i <= y1_i or x2_i <= x1_i:
                    continue

                patch_i = self.ts_dataset[0,i,0,z1_i:z2_i, y1_i:y2_i, x1_i:x2_i].read().result().astype(np.float32)
                patch_j = self.ts_dataset[0,j,0,z1_j:z2_j, y1_j:y2_j, x1_j:x2_j].read().result().astype(np.float32)

                shift, error, _ = phase_cross_correlation(patch_i, patch_j, upsample_factor=upsample_factor)
                dz_shift, dy_shift, dx_shift = shift

                self.tile_positions[j,0] += dz_shift * self.pixel_size[0]
                self.tile_positions[j,1] += dy_shift * self.pixel_size[1]
                self.tile_positions[j,2] += dx_shift * self.pixel_size[2]

                shifts[(i, j)] = (dz_shift, dy_shift, dx_shift)
        return shifts

    def run(self):
        """Full pipeline: refine, compute, pad, create store, fuse."""
        shifts = self.refine_tile_positions_with_cross_correlation()
        print("Shifts:", shifts)

        self.unpadded_shape, self.offset = self._compute_fused_image_space()
        self.padded_shape, self.pad = self._pad_to_tile_multiple(self.unpadded_shape)
        print(f"Unpadded shape: {self.unpadded_shape}, pad: {self.pad}, padded: {self.padded_shape}")

        self.fused_ts = self._create_fused_tensorstore()
        self._fuse_tiles()
