import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)

import numpy as np
import tensorstore as ts
from pathlib import Path
from tqdm import trange, tqdm
import cupy as cp
from concurrent.futures import ThreadPoolExecutor
from cucim.skimage.measure import block_reduce
from cucim.skimage.exposure import match_histograms
from cucim.skimage.registration import phase_cross_correlation
from cupyx.scipy.ndimage import shift as cp_shift
from opm_processing.imageprocessing.ssim_cuda import structural_similarity_cupy_sep_shared as ssim_cuda
import json
# import numba

# ----------------------------------
#  RawModule fusion kernel
# ----------------------------------
_fuse_kernel_code = r'''
extern "C" __global__
void fuse3d(
    const unsigned short* __restrict__ tile,
    const unsigned short* __restrict__ blk,
    const float*         __restrict__ w,
    unsigned short*      __restrict__ out,
    int C, int Z, int Y, int X
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C*Z*Y*X;
    if (idx >= total) return;
    int tmp = idx;
    int x = tmp % X; tmp /= X;
    int y = tmp % Y; tmp /= Y;
    int z = tmp % Z; tmp /= Z;
    int widx = z*Y*X + y*X + x;
    float weight = w[widx];
    float t = tile[idx];
    float b = blk[idx];
    float v = t*weight + b*(1.0f - weight);
    out[idx] = (unsigned short)min(max(v + 0.5f, 0.0f), 65535.0f);
}
'''
_module = cp.RawModule(code=_fuse_kernel_code, options=("-std=c++11",))
_fuse3d = _module.get_function('fuse3d')


class TileFusion:
    """
    3D (ZYX) fusion of overlapping image tiles with robust blending.

    Parameters
    ----------
    ts_dataset : str or Path
        Path to input TensorStore (shape: T, N, C, Z, Y, X).
    tile_positions : list of tuple of float
        Origins in physical units [(z, y, x), ...].
    output_path : str or Path
        Path for fused output TensorStore.
    pixel_size : tuple of float
        Voxel size (z, y, x) in physical units.
    blend_pixels : tuple of int, optional
        Feather width in voxels for (z, y, x). Default is (50, 200, 200).
    """

    def __init__(
        self,
        deskewed_data_path: str | Path,
        tile_positions: list[tuple[float, float, float]],
        output_path: str | Path,
        pixel_size: tuple[float, float, float],
        blend_pixels: tuple[int, int, int] = (50, 200, 200),
        debug = False
    ):
        # Open input dataset
        spec = {
            "driver" : "zarr3",
            "kvstore" : {
                "driver" : "file",
                "path" : str(deskewed_data_path)
            }
        }
        self.ts_dataset = ts.open(spec).result()
        self.tile_positions = np.array(
            tile_positions, dtype=float
        )
        self.output_path = Path(output_path)
        self.pixel_size = pixel_size

        # Dataset dims: T, N, C, Z, Y, X
        (
            self.time_dim,
            self.position_dim,
            self.channels,
            self.z_dim,
            height,
            width,
        ) = self.ts_dataset.shape

        self.tile_shape = (self.z_dim, height, width)
        self.Y = height
        self.X = width

        # 3D weight mask
        self.weight_3d = self._generate_3d_blending_weights(blend_pixels)
        # upload once to GPU
        self.weight_3d_gpu = cp.asarray(self.weight_3d, dtype=cp.float32)
                # no persistent GPU buffers; allocate per-slab in _process_slab

        # Fusion volume attributes
        self.offset = None
        self.unpadded_shape = None
        self.chunk_shape = (1,1,1,1,512,512)
        self.chunk_y = self.chunk_shape[-2]
        self.chunk_x = self.chunk_shape[-1]
        self._buf_tile = cp.empty((self.channels, self.z_dim, self.chunk_y, self.chunk_x), cp.uint16)
        self._buf_blk  = cp.empty_like(self._buf_tile)
        self._buf_out  = cp.empty_like(self._buf_tile)
        # thread pool for TensorStore I/O (interior writes and slab reads/writes)
        self.io_executor = ThreadPoolExecutor(max_workers=2)
        # CUDA streams to overlap host→device, compute, and device→host
        self.streams = [cp.cuda.Stream() for _ in range(4)]
        self._next_stream = 0
        
        self.padded_shape = None
        self.pad = (0, 0, 0)
        self.fused_ts = None

        # Registration metrics
        self.pairwise_metrics = {}
        self.global_offsets = None
        
        self.blend_pixels = blend_pixels
        self.debug = debug

    def _generate_3d_blending_weights(
        self,
        blend_pixels: tuple[int, int, int]
    ) -> np.ndarray:
        """
        Generate 3D cosine-feathered blending weights.

        Parameters
        ----------
        blend_pixels : tuple of int
            Feather width (bz, by, bx) in voxels.

        Returns
        -------
        weight : ndarray
            A (Z, Y, X) weight array.
        """
        bz, by, bx = blend_pixels
        tz, th, tw = self.tile_shape

        # Clamp feather widths
        bz = min(bz, tz // 2)
        by = min(by, th // 2)
        bx = min(bx, tw // 2)

        # Create 1D cosine profiles
        z_profile = np.ones(tz, dtype=np.float32)
        y_profile = np.ones(th, dtype=np.float32)
        x_profile = np.ones(tw, dtype=np.float32)

        z_line = np.linspace(0, np.pi, bz)
        y_line = np.linspace(0, np.pi, by)
        x_line = np.linspace(0, np.pi, bx)

        z_feather = 0.5 * (1 - np.cos(z_line))
        y_feather = 0.5 * (1 - np.cos(y_line))
        x_feather = 0.5 * (1 - np.cos(x_line))

        z_profile[:bz] = z_feather
        z_profile[-bz:] = z_feather[::-1]

        y_profile[:by] = y_feather
        y_profile[-by:] = y_feather[::-1]

        x_profile[:bx] = x_feather
        x_profile[-bx:] = x_feather[::-1]

        # Build 3D mask
        weight = (
            z_profile[:, None, None]
            * y_profile[None, :, None]
            * x_profile[None, None, :]
        )

        return weight

    def _compute_fused_image_space(self) -> None:
        """
        Compute and store fused volume shape and physical offset.
        """
        pos = self.tile_positions
        min_z, min_y, min_x = np.min(pos, axis=0)

        max_z = (
            np.max(pos[:, 0])
            + self.tile_shape[0] * self.pixel_size[0]
        )
        max_y = (
            np.max(pos[:, 1])
            + self.tile_shape[1] * self.pixel_size[1]
        )
        max_x = (
            np.max(pos[:, 2])
            + self.tile_shape[2] * self.pixel_size[2]
        )

        size_z = int(
            np.ceil((max_z - min_z) / self.pixel_size[0])
        )
        size_y = int(
            np.ceil((max_y - min_y) / self.pixel_size[1])
        )
        size_x = int(
            np.ceil((max_x - min_x) / self.pixel_size[2])
        )

        self.unpadded_shape = (size_z, size_y, size_x)
        self.offset = (min_z, min_y, min_x)

    def _pad_to_chunk_multiple(self) -> None:
        """
        Compute and store padded_shape and pad for chunk multiples.
        """
        _, _, _, tz, th, tw = self.chunk_shape
        sz, sy, sx = self.unpadded_shape

        pad_z = (-sz) % tz
        pad_y = (-sy) % th
        pad_x = (-sx) % tw

        self.padded_shape = (sz + pad_z, sy + pad_y, sx + pad_x)
        self.pad = (pad_z, pad_y, pad_x)

    def _create_fused_tensorstore(self) -> None:
        """
        Create the output TensorStore using padded_shape.
        """
        # full volume shape in T,P,C,Z,Y,X:
        full_shape = [1, 1, self.channels, *self.padded_shape]
        # codec will break each Y×X into 512×512 tiles (per t,p,c,z):
        codec_chunk = [1, 1, 1, 1, self.chunk_shape[-2], self.chunk_shape[-1]]
        # shard groups the *entire* Y×X plane for each t,p,c,z:
        shard_chunk = [1, 1, 1, 1, self.padded_shape[1], self.padded_shape[2]]

        config = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(self.output_path)},
            "metadata": {
                "shape": full_shape,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": shard_chunk},
                },
                "chunk_key_encoding": {"name": "default"},
                "codecs": [
                    {
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": codec_chunk,
                            "codecs": [
                                {
                                    "name": "bytes",
                                    "configuration": {"endian": "little"},
                                },
                                {
                                    "name": "blosc",
                                    "configuration": {
                                        "cname": "zstd",
                                        "clevel": 5,
                                        "shuffle": "bitshuffle",
                                    },
                                },
                            ],
                            "index_codecs": [
                                {
                                    "name": "bytes",
                                    "configuration": {"endian": "little"},
                                },
                                {"name": "crc32c"},
                            ],
                            "index_location": "end",
                        },
                    }
                ],
                "data_type": "uint16",
            },
        }

        self.fused_ts = ts.open(
            config, create=True, open=True
        ).result()
        
    def _write_interior(self, interior, z0, y0, x0, bz, tz, by, th, bx, tw):
        z_slice = slice(z0 + bz, z0 + tz - bz)
        y_slice = slice(y0 + by, y0 + th - by)
        x_slice = slice(x0 + bx, x0 + tw - bx)
        self.fused_ts[0,0,:, z_slice, y_slice, x_slice].write(interior).result()

    def _write_chunk(self, data, z0, y0, x0, zs, ze, ys0, ys1, xs0, xs1):
        z_slice = slice(z0 + zs, z0 + ze)
        y_slice = slice(y0 + ys0, y0 + ys1)
        x_slice = slice(x0 + xs0, x0 + xs1)
        arr = data.reshape((self.channels, ze - zs, ys1 - ys0, xs1 - xs0))
        self.fused_ts[0,0,:, z_slice, y_slice, x_slice].write(arr).result()

    def _process_slab(self, idx, z0, y0, x0, tile, slab_spec):
        zs, ze, ys, ye, xs, xe = slab_spec
        Zs = ze - zs
        # subdivide boundary slab into chunk_y × chunk_x blocks
        for ys0 in range(ys, ye, self.chunk_y):
            ys1 = min(ye, ys0 + self.chunk_y)
            for xs0 in range(xs, xe, self.chunk_x):
                xs1 = min(xe, xs0 + self.chunk_x)
                # read fused block
                blk_future = self.io_executor.submit(
                    self.fused_ts[0,0,:, slice(z0+zs, z0+ze),
                                           slice(y0+ys0, y0+ys1),
                                           slice(x0+xs0, x0+xs1)].read().result
                )
                blk_arr = blk_future.result().astype(np.uint16)
                # copy into GPU buffers
                self._buf_tile[:, zs:ze, : (ys1-ys0), : (xs1-xs0)] = cp.asarray(
                    tile[:, zs:ze, ys0:ys1, xs0:xs1], cp.uint16)
                self._buf_blk[:, zs:ze, : (ys1-ys0), : (xs1-xs0)] = cp.asarray(
                    blk_arr.reshape((self.channels, Zs, ys1-ys0, xs1-xs0)))
                # launch kernel on stream
                stream = self.streams[self._next_stream]
                self._next_stream = (self._next_stream + 1) % len(self.streams)
                with stream:
                    total = self.channels * Zs * (ys1-ys0) * (xs1-xs0)
                    threads = 256
                    blocks = (total + threads - 1) // threads
                    wflat = self.weight_3d_gpu[zs:ze, ys0:ys1, xs0:xs1].ravel()
                    _fuse3d((blocks,), (threads,), (
                        self._buf_tile.ravel(),
                        self._buf_blk.ravel(),
                        wflat,
                        self._buf_out.ravel(),
                        self.channels, Zs, ys1-ys0, xs1-xs0
                    ))
                    dev2host = self._buf_out.get(stream=stream)
                # write fused chunk
                self.io_executor.submit(
                    self._write_chunk,
                    dev2host, z0, y0, x0, zs, ze, ys0, ys1, xs0, xs1
                )

    def _fuse_tiles(self):
        tz, th, tw = self.tile_shape
        bz, by, bx = self.blend_pixels
        slab_specs = [
            (0, bz, 0, th, 0, tw),
            (tz-bz, tz, 0, th, 0, tw),
            (bz, tz-bz, 0, by, 0, tw),
            (bz, tz-bz, th-by, th, 0, tw),
            (bz, tz-bz, by, th-by, 0, bx),
            (bz, tz-bz, by, th-by, tw-bx, tw),
        ]
        for idx, (z_phys, y_phys, x_phys) in enumerate(
            tqdm(self.tile_positions, desc="fuse-tile", leave=False)
        ):
            z0 = int((z_phys - self.offset[0]) / self.pixel_size[0])
            y0 = int((y_phys - self.offset[1]) / self.pixel_size[1])
            x0 = int((x_phys - self.offset[2]) / self.pixel_size[2])
            tile = (
                self.ts_dataset[0, idx, :, :, :, :]
                .read().result()
                .astype(np.uint16)
            )
            # Schedule interior write
            self.io_executor.submit(
                self._write_interior,
                tile[:, bz:tz-bz, by:th-by, bx:tw-bx],
                z0, y0, x0, bz, tz, by, th, bx, tw
            )
            # GPU-boundary slabs
            for spec in slab_specs:
                self._process_slab(idx, z0, y0, x0, tile, spec)
            del tile
        self.io_executor.shutdown(wait=True)

    def save_pairwise_metrics(self, filepath: str | Path) -> None:
        """
        Save self.pairwise_metrics (results of refine_tile_positions) to a JSON file.
        """
        path = Path(filepath)
        out = {f"{i},{j}": list(v) for (i, j), v in self.pairwise_metrics.items()}
        with open(path, 'w') as f:
            json.dump(out, f)

    def load_pairwise_metrics(self, filepath: str | Path) -> None:
        """
        Load pairwise_metrics from a JSON file previously saved.
        """
        path = Path(filepath)
        with open(path, 'r') as f:
            data = json.load(f)
        self.pairwise_metrics = {}
        for key, v in data.items():
            i, j = map(int, key.split(','))
            self.pairwise_metrics[(i, j)] = tuple(v)

    @staticmethod
    def register_and_score(
        g1: cp.ndarray,
        g2: cp.ndarray,
        win_size: int = 7,
        debug = False
    ) -> tuple[tuple[float, float, float], float] | tuple[None, None]:
        """
        Given two overlapping GPU patches (ZYX, uint16), histogram‐match g2→g1,
        compute subpixel shift via phase‐corr, apply it, then return
        ((dz,dy,dx), mean_ssim).  If registration fails, returns (None, None).
        """

        # 1) histogram‐match moving→fixed
        try:
            g2m = match_histograms(g2, g1)
        except Exception as e:
            print("histogram", e)
            return None, None

        # 2) subpixel shift via phase_cross_correlation
        try:
            shift, _, _ = phase_cross_correlation(
                g1, g2m, disambiguate=True, normalization="phase"
            )
            if debug:
                print(shift)
        except Exception as e:
            print("phase_cc",e)
            return None, None

        # 3) apply shift (ZYX order) with bilinear interp
        try:
            g2s = cp_shift(g2m, shift=shift, order=1)
        except Exception as e:
            print("shift",e)
            return None, None

        # 4) compute SSIM
        try:
            ssim_val = ssim_cuda(
                g1, g2s,
                win_size=win_size
            )
        except Exception as e:
            print("ssim",e)
            return None, None

        return list(shift), float(ssim_val)
    
    def refine_tile_positions_with_cross_correlation(
            self,
            downsample_factors: tuple[int, int, int] = (3, 5, 5),
            ssim_window: int = 7,
            ch_idx: int = 0,
            threshold: float = 0.0,
        ) -> None:
        """
        For each tile‐pair (i,j) at each timepoint:
        1) extract overlapping subvolumes,
        2) downsample → GPU uint16,
        3) call register_and_score → (dz,dy,dx,SSIM),
        4) scale shift back to original voxels and store in self.pairwise_metrics.
        """
        self.pairwise_metrics.clear()
        n_pos = self.position_dim
        executor = ThreadPoolExecutor(max_workers=2)
        
        def bounds_1d(offset: int, length: int) -> tuple[int, int]:
            lo = max(0, offset)
            hi = min(length, offset + length)
            return lo, hi

        for t in range(self.time_dim):
            base = t * n_pos

            for i_pos in trange(n_pos,desc="register",leave=False):
                i = base + i_pos

                for j_pos in range(i_pos + 1, n_pos):
                    j = base + j_pos

                    # 1) integer‐voxel offset from tile i → j
                    phys_diff = self.tile_positions[j] - self.tile_positions[i]
                    vox_off = np.round(phys_diff / np.array(self.pixel_size)).astype(int)
                    dz, dy, dx = vox_off

                    # 2) compute matching 1D bounds
                    D, H, W = self.tile_shape
                    b_i = [
                        bounds_1d(dz, D),
                        bounds_1d(dy, H),
                        bounds_1d(dx, W),
                    ]
                    b_j = [
                        bounds_1d(-dz, D),
                        bounds_1d(-dy, H),
                        bounds_1d(-dx, W),
                    ]

                    # skip if no overlap
                    if any(hi <= lo for lo, hi in b_i):
                        continue

                    # 3) async read both patches
                    def read_patch(idx, bnds):
                        z0, z1 = bnds[0]
                        y0, y1 = bnds[1]
                        x0, x1 = bnds[2]
                        arr = (self.ts_dataset[0, idx, ch_idx,
                                            z0:z1, y0:y1, x0:x1]
                            .read().result()
                            .astype(np.float32))
                        return arr

                    fut_i = executor.submit(read_patch, i, b_i)
                    fut_j = executor.submit(read_patch, j, b_j)
                    patch_i = fut_i.result()
                    patch_j = fut_j.result()

                    # 4) downsample on CPU then upload to GPU as uint16
                    g1 = block_reduce(
                        cp.asarray(patch_i,dtype=cp.float32), 
                        tuple(downsample_factors), 
                        cp.mean
                    )
                    g2 = block_reduce(
                        cp.asarray(patch_j,dtype=cp.float32), 
                        tuple(downsample_factors), 
                        cp.mean
                    )

                    # # 5) effective pixel size after downsampling
                    # pixel_size_ds = tuple(
                    #     self.pixel_size[d] * downsample_factors[d]
                    #     for d in range(3)
                    # ) 

                    # 6) register + SSIM
                    shift_ds, ssim_val = self.register_and_score(
                        g1, 
                        g2,
                        win_size=ssim_window,
                        debug=self.debug
                    )
                    if shift_ds is None or ssim_val < threshold and not(threshold==0.0):
                        continue

                    # 7) scale shift back to original‐voxel units
                    dz_s = int(shift_ds[0] * downsample_factors[0])
                    dy_s = int(shift_ds[1] * downsample_factors[1])
                    dx_s = int(shift_ds[2] * downsample_factors[2])

                    # 8) store metrics
                    self.pairwise_metrics[(i, j)] = (dz_s, dy_s, dx_s, ssim_val)
                    if self.debug:
                        print(f"Pair {(i,j)} → shift {(dz_s,dy_s,dx_s)}, SSIM {ssim_val:.4f}")
        executor.shutdown(wait=True)

    def optimize_shifts(self):
        """
        Compute and store global optimized offsets with a smoothness prior,
        now weighting each pairwise constraint by its SSIM quality.

        Populates `self.global_offsets` as an (N, 3) array of (dz, dy, dx).
        """
        metrics = self.pairwise_metrics
        N = len(self.tile_positions)

        # Build neighbor graph from pairwise overlaps
        neighbors = {i: set() for i in range(N)}
        for (i, j) in metrics:
            neighbors[i].add(j)
            neighbors[j].add(i)

        # Prepare array for optimized shifts
        d_opt = np.zeros((N, 3), dtype=np.float64)
        smoothness_weight = 0.1

        # Solve one weighted least‐squares problem per axis
        for axis in range(3):
            rows = []
            vals = []

            # 1) Weighted pairwise shift constraints
            for (i, j), v in metrics.items():
                shift_axis = v[axis]
                ssim = v[3]
                # weight the equation (x_j - x_i = shift) by SSIM:
                w = np.sqrt(ssim)
                row = np.zeros(N, dtype=np.float64)
                row[j] =  w
                row[i] = -w
                rows.append(row)
                vals.append(w * shift_axis)

            # 2) Smoothness prior: unweighted penalty for neighbor consistency
            for i in range(N):
                neigh = neighbors[i]
                deg = len(neigh)
                if deg == 0:
                    continue
                row = np.zeros(N, dtype=np.float64)
                row[i] = smoothness_weight * deg
                for j in neigh:
                    row[j] = -smoothness_weight
                rows.append(row)
                vals.append(0.0)

            # 3) Zero‐mean gauge to fix global translation
            row = np.ones(N, dtype=np.float64)
            rows.append(row)
            vals.append(0.0)

            # Stack into A x = b and solve
            A = np.vstack(rows)
            b = np.array(vals, dtype=np.float64)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            d_opt[:, axis] = sol

        self.global_offsets = d_opt

    def run(
        self,
        downsample_factors: tuple[int, int, int] = (1, 1, 1),
        upsample_factor: int = 10,
        threshold: float = 0.0,
    ) -> None:
        """
        Execute full fusion pipeline.

        Parameters
        ----------
        downsample_factors : tuple of int
            Factors for block_reduce before registration.
        upsample_factor : int
            Upsampling factor for registration.
        threshold : float
            Minimum NCC to accept shifts.
        """
        self.refine_tile_positions_with_cross_correlation(
            downsample_factors,
            upsample_factor,
            threshold,
        )
        self.optimize_shifts()
        for idx, off in enumerate(self.global_offsets):
            self.tile_positions[idx] += off * np.array(
                self.pixel_size
            )
        self._compute_fused_image_space()
        self._pad_to_tile_multiple()
        self._create_fused_tensorstore()
        self._fuse_tiles()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn")
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
    metric_path = root_path.parents[0] / Path(str(root_path.stem)+"_metrics.json")
    try:
        tile_fuser.load_pairwise_metrics(metric_path)
    except Exception:
        tile_fuser.refine_tile_positions_with_cross_correlation(
            downsample_factors=[3,5,5]
        )
        tile_fuser.save_pairwise_metrics(metric_path)
    tile_fuser.optimize_shifts()

    for idx, off in enumerate(tile_fuser.global_offsets):
        tile_fuser.tile_positions[idx] += off * np.array(
            tile_fuser.pixel_size
        )
    tile_fuser._compute_fused_image_space()
    tile_fuser._pad_to_chunk_multiple()
    tile_fuser._create_fused_tensorstore()
    tile_fuser._fuse_tiles()