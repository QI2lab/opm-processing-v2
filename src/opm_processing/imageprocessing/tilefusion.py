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
    unsigned short uv = (unsigned short)min(max(v + 0.5f, 0.0f), 65535.0f);
    out[idx] = uv;
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

        # 3D weight mask
        self.weight_3d = self._generate_3d_blending_weights(blend_pixels)
        # upload once to GPU
        self.weight_3d_gpu = cp.asarray(self.weight_3d, dtype=cp.float32)
                # no persistent GPU buffers; allocate per-slab in _process_slab

        # Fusion volume attributes
        self.offset = None
        self.unpadded_shape = None
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

    def _pad_to_tile_multiple(self) -> None:
        """
        Compute and store padded_shape and pad for tile multiples.
        """
        tz, th, tw = self.tile_shape
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
        tz, th, tw = self.tile_shape
        chunk = [1, 1, self.channels, tz, th // 4, tw // 4]
        shard = [1, 1, self.channels, tz, th, tw]

        config = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(self.output_path)},
            "metadata": {
                "shape": [1, 1, self.channels, *self.padded_shape],
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": shard},
                },
                "chunk_key_encoding": {"name": "default"},
                "codecs": [
                    {
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": chunk,
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

    # @staticmethod
    # @numba.njit(parallel=True, cache=True)
    # def _blend_arrays(tc, block, wm):
    #     """
    #     JIT-blended fusion of two volumes with weights wm in parallel.

    #     Parameters
    #     ----------
    #     tc : ndarray
    #         Tile crop array of shape (C, Z, Y, X).
    #     block : ndarray
    #         Existing fused block, same shape.
    #     wm : ndarray
    #         Weight mask array of shape (Z, Y, X).

    #     Returns
    #     -------
    #     fused : ndarray
    #         Fused result array of shape (C, Z, Y, X).
    #     """
    #     C, Z, Y, X = block.shape
    #     fused = np.empty((C, Z, Y, X), dtype=np.float32)
    #     # Parallel loops over channels and z
    #     for c in numba.prange(C):
    #         for z in range(Z):
    #             for y in range(Y):
    #                 for x in range(X):
    #                     wv = wm[z, y, x]
    #                     t = tc[c, z, y, x]
    #                     b = block[c, z, y, x]
    #                     num = t * wv + b * (1.0 - wv)
    #                     den = wv + (1.0 - wv)
    #                     fused[c, z, y, x] = num / den if den != 0.0 else 0.0
    #     return fused

    def _process_slab(self, idx, z0, y0, x0, tile, slab_spec):
        """
        Read existing fused slab, fuse with tile slab on GPU, write back.
        slab_spec = (zs, ze, ys, ye, xs, xe)
        """
        zs, ze, ys, ye, xs, xe = slab_spec
        # slab origin slices
        zslice = slice(z0 + zs, z0 + ze)
        yslice = slice(y0 + ys, y0 + ye)
        xslice = slice(x0 + xs, x0 + xe)
        # read existing fused block for this slab
        blk = self.fused_ts[0,0,:, zslice, yslice, xslice]  
        blk_arr = blk.read().result().astype(np.uint16)
        # dimensions of slab
        C = self.channels
        Z = ze - zs
        Y = ye - ys
        X = xe - xs
        total = C * Z * Y * X
        # allocate GPU buffers sized exactly for this slab
        tile_buf = cp.empty(total, dtype=cp.uint16)
        blk_buf = cp.empty(total, dtype=cp.uint16)
        out_buf = cp.empty(total, dtype=cp.uint16)
        # copy tile slab and block into buffers
        tile_sub = tile[:, zs:ze, ys:ye, xs:xe].ravel()
        tile_buf[:] = cp.asarray(tile_sub, dtype=cp.uint16)
        blk_buf[:] = cp.asarray(blk_arr.ravel(), dtype=cp.uint16)
        # launch fusion kernel
        threads = 256
        blocks = (total + threads - 1) // threads
        weight_flat = self.weight_3d_gpu[zs:ze, ys:ye, xs:xe].ravel()
        _fuse3d((blocks,), (threads,), (tile_buf, blk_buf, weight_flat, out_buf, C, Z, Y, X))
        # retrieve and write fused slab
        fused = out_buf.get().reshape((C, Z, Y, X))
        return self.fused_ts[0,0,:, zslice, yslice, xslice].write(fused)

    def _fuse_tiles(self) -> None:
        """
        Fuse all tiles by:
        - doing the interior region copy immediately,
        - submitting only the slab‐fusion tasks for one tile,
        - waiting for those to finish (and drop the tile),
        - then moving on to the next tile.
        This keeps peak RAM bounded to ~one‐tile’s worth.
        """
        tz, th, tw = self.tile_shape
        bz, by, bx = self.blend_pixels
        zi0, zi1 = bz, tz - bz
        yi0, yi1 = by, th - by
        xi0, xi1 = bx, tw - bx

        slab_specs = [
            (0, zi0, 0, th, 0, tw),
            (zi1, tz, 0, th, 0, tw),
            (zi0, zi1, 0, yi0, 0, tw),
            (zi0, zi1, yi1, th, 0, tw),
            (zi0, zi1, yi0, yi1, 0, xi0),
            (zi0, zi1, yi0, yi1, xi1, tw),
        ]

        executor = ThreadPoolExecutor(max_workers=2)

        for idx, (z_phys, y_phys, x_phys) in enumerate(
            tqdm(self.tile_positions, desc="fuse-tile", leave=False)
        ):
            # compute write origin
            z0 = int((z_phys - self.offset[0]) / self.pixel_size[0])
            y0 = int((y_phys - self.offset[1]) / self.pixel_size[1])
            x0 = int((x_phys - self.offset[2]) / self.pixel_size[2])

            # 1) Read the full tile once
            tile = self.ts_dataset[0, idx, :, :, :, :].read().result().astype(np.uint16)

            # 2) Interior region: copy immediately (no GPU)
            interior = tile[:, zi0:zi1, yi0:yi1, xi0:xi1]
            wf_int = self.fused_ts[
                0,0,:, z0+zi0:z0+zi1,
                    y0+yi0:y0+yi1,
                    x0+xi0:x0+xi1
            ].write(interior)
            wf_int.result()  # wait here for sequential write

            # 3) Slabs: submit GPU‐blend jobs
            slab_futures = []
            for spec in slab_specs:
                slab_futures.append(
                    executor.submit(
                        self._process_slab,
                        idx, z0, y0, x0, tile, spec
                    )
                )

            # 4) Drain this tile’s slab jobs before moving on
            for fut in slab_futures:
                write_future = fut.result()    # returns the TS-write future
                write_future.result()          # wait for that write to finish

            # drop `tile` and its buffers here
            del tile

        # clean up executor
        executor.shutdown(wait=True)


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