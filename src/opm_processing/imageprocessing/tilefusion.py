import numpy as np
import tensorstore as ts
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from numba import njit, prange
import cupy as cp
from cucim.skimage.measure import block_reduce
from cucim.skimage.exposure import match_histograms
from cucim.skimage.registration import phase_cross_correlation
from cupyx.scipy.ndimage import shift as cp_shift
from opm_processing.imageprocessing.ssim_cuda import structural_similarity_cupy_sep_shared as ssim_cuda
import json
from tqdm import trange, tqdm
import gc

@njit(parallel=True)
def _accumulate_tile_shard(
    fused,     # float32[C, dz, Yp, Xp]
    weight,    # float32[C, dz, Yp, Xp]
    sub,       # float32[C, sub_dz, Y, X]
    w3d,       # float32[sub_dz, Y, X]
    z_off, y_off, x_off
):
    """
    Parallel over sub_dz * Y slices instead of just channels.
    """
    C, dz, Yp, Xp = fused.shape
    _, sub_dz, Y, X = sub.shape
    total = sub_dz * Y

    # each iteration handles one (dz_i, y) across all channels
    for idx in prange(total):
        dz_i = idx // Y
        y    = idx %  Y
        gz   = z_off + dz_i
        gy   = y_off + y

        w_line = w3d[dz_i, y]        # shape: (X,)
        for c in range(C):
            sub_line = sub[c, dz_i, y]  # shape: (X,)
            base_f = fused[c, gz, gy]
            base_w = weight[c, gz, gy]
            for x in range(X):
                gx = x_off + x
                w  = w_line[x]
                base_f[gx] += sub_line[x] * w
                base_w[gx] += w


@njit(parallel=True)
def _normalize_shard(fused, weight):
    """
    Parallel over every (z, y, x) in the shard.
    """
    C, dz, Yp, Xp = fused.shape
    total = C * dz * Yp

    # flatten (c, z, y) into one index, leave x inner
    for idx in prange(total):
        c    = idx // (dz * Yp)
        rem  = idx %  (dz * Yp)
        z    = rem // Yp
        y    = rem %  Yp

        base_f = fused[c, z, y]
        base_w = weight[c, z, y]
        for x in range(Xp):
            w = base_w[x]
            if w > 0.0:
                base_f[x] = base_f[x] / w
            else:
                base_f[x] = 0.0

@njit(parallel=True)
def _blend_numba(sub_i, sub_j,
                 wz_i, wy_i, wx_i,
                 wz_j, wy_j, wx_j,
                 out_f):
    """
    Numba‐accelerated 3D feather blend of two sub‐volumes.

    Parameters
    ----------
    sub_i, sub_j : ndarray, shape (dz, dy, dx)
        Float32 input sub‐volumes.
    wz_i, wy_i, wx_i : ndarray
        1D weight profiles for sub_i along z, y, x.
    wz_j, wy_j, wx_j : ndarray
        1D weight profiles for sub_j along z, y, x.
    out_f : ndarray, shape (dz, dy, dx)
        Preallocated output buffer.

    Returns
    -------
    out_f : ndarray, shape (dz, dy, dx)
        Blended result.
    """
    dz, dy, dx = sub_i.shape

    for z in prange(dz):
        wi_z = wz_i[z]
        wj_z = wz_j[z]
        for y in range(dy):
            wi_zy = wi_z * wy_i[y]
            wj_zy = wj_z * wy_j[y]
            for x in range(dx):
                wi = wi_zy * wx_i[x]
                wj = wj_zy * wx_j[x]
                tot = wi + wj
                if tot > 1e-6:
                    out_f[z, y, x] = (
                        wi * sub_i[z, y, x]
                        + wj * sub_j[z, y, x]
                    ) / tot
                else:
                    out_f[z, y, x] = sub_i[z, y, x]
    return out_f


class TileFusion:
    """
    3D tile fusion with CPU-parallel, Numba-accelerated feather blending.

    Integrates existing registration and global‐optimization. Writes
    fused volume to Zarr v3 via TensorStore, using the exact schema 
    and chunking you provided.
    """

    def __init__(self,
                 deskewed_data_path: str | Path,
                 tile_positions: list[tuple[float, float, float]],
                 pixel_size: tuple[float, float, float],
                 blend_pixels: tuple[int, int, int] = (20, 400, 400),
                 max_workers: int = 8,
                 debug: bool = False):
        """
        Initialize parameters and open input TensorStore.

        Parameters
        ----------
        deskewed_data_path : str or Path
            Path to input Zarr v3 store (shape T,N,C,Z,Y,X).
        tile_positions : list of tuple of float
            Physical origins [(z,y,x), ...] for each tile.
        pixel_size : tuple of float
            Voxel size in (z,y,x) physical units.
        blend_pixels : tuple of int, optional
            Feather widths (bz,by,bx). Default (50,200,200).
        debug : bool, optional
            If True, print debug info. Default False.
        """
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        spec_in = {
            "context": {
                # memory-map the file for direct page-cache I/O
                "file_io_memmap": True,
                # don’t fsync every write (let the OS batch it)
                "file_io_sync": False,
                # allow up to max_workers concurrent file reads/writes
                "file_io_concurrency": {"limit": max_workers},
                "data_copy_concurrency": {"limit": max_workers},
            },
            "driver": "zarr3",
            "kvstore": {
                "driver": "file",
                "path": str(deskewed_data_path)
            }
        }

        self.ts_dataset = ts.open(
            spec_in,
            create=False,
            open=True
        ).result()

        # Dataset dims: T, N, C, Z, Y, X
        (self.time_dim, self.position_dim, self.channels,
         self.z_dim, self.Y, self.X) = self.ts_dataset.shape
        self.tile_shape = (self.z_dim, self.Y, self.X)

        self.tile_positions = np.array(tile_positions, dtype=float)
        self.pixel_size = pixel_size
        self.blend_pixels = blend_pixels
        self.debug = debug

        # Precompute 1D feather profiles
        bz, by, bx = blend_pixels
        self.z_profile = self._make_1d_profile(self.z_dim, bz)
        self.y_profile = self._make_1d_profile(self.Y,   by)
        self.x_profile = self._make_1d_profile(self.X,   bx)

        # Fixed chunk shape per your spec:
        self.chunk_shape = (1, 1, 1, 1, 512, 512)
        self.chunk_y = self.chunk_shape[-2]
        self.chunk_x = self.chunk_shape[-1]

        # Placeholders to be set later
        self.offset = None
        self.unpadded_shape = None
        self.padded_shape = None
        self.pad = (0, 0, 0)
        self.fused_ts = None
        self.max_workers = max_workers


    @staticmethod
    def _make_1d_profile(length: int,
                        blend: int) -> np.ndarray:
        """
        Create a 1D linear ramp profile over `blend` voxels at each end.

        Parameters
        ----------
        length : int
            Number of voxels in the axis.
        blend : int
            Feather width in voxels.

        Returns
        -------
        prof : ndarray, shape (length,)
            Values between 0 and 1.
        """
        prof = np.ones(length, dtype=np.float32)
        if blend > 0:
            ramp = np.linspace(0, 1, blend, endpoint=False,
                               dtype=np.float32)
            prof[:blend]  = ramp
            prof[-blend:] = ramp[::-1]
        return prof
    
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
            threshold: float = 0.75,
            max_link_shift: float | tuple[float, float, float] | None = [20.,400.,400.]
        ) -> None:
        """
        For each tile‐pair (i,j) at each timepoint:
        1) extract overlapping subvolumes,
        2) downsample → GPU uint16,
        3) call register_and_score → (dz,dy,dx,SSIM),
        4) scale shift back to original voxels and store in self.pairwise_metrics.
        """
        if max_link_shift is not None:
            if isinstance(max_link_shift, (int, float)):
                max_link_shift = (max_link_shift,)*3
            else:
                max_link_shift = tuple(max_link_shift)
        
        try:
            self.pairwise_metrics.clear()
        except Exception:
            self.pairwise_metrics = {}
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
                    ssim_val = np.round(ssim_val,3)
                    if shift_ds is None or ssim_val < threshold and not(threshold==0.0):
                        continue

                    # 7) scale shift back to original‐voxel units
                    dz_s = int(shift_ds[0] * downsample_factors[0])
                    dy_s = int(shift_ds[1] * downsample_factors[1])
                    dx_s = int(shift_ds[2] * downsample_factors[2])
                    
                    if max_link_shift is not None and (
                        abs(dz_s) > max_link_shift[0]
                        or abs(dy_s) > max_link_shift[1]
                        or abs(dx_s) > max_link_shift[2]
                        ):
                            if self.debug:
                                print(f"Dropping link {(i,j)} shift={(dz_s,dy_s,dx_s)} > {max_link_shift}")
                            continue
                        
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


    def _compute_fused_image_space(self) -> None:
        """
        Compute the unpadded fused volume shape and physical offset.
        """
        pos = self.tile_positions
        min_z, min_y, min_x = np.min(pos, axis=0)

        max_z = (np.max(pos[:, 0])
                 + self.z_dim * self.pixel_size[0])
        max_y = (np.max(pos[:, 1])
                 + self.Y    * self.pixel_size[1])
        max_x = (np.max(pos[:, 2])
                 + self.X    * self.pixel_size[2])

        size_z = int(np.ceil((max_z - min_z)
                             / self.pixel_size[0]))
        size_y = int(np.ceil((max_y - min_y)
                             / self.pixel_size[1]))
        size_x = int(np.ceil((max_x - min_x)
                             / self.pixel_size[2]))

        self.unpadded_shape = (size_z, size_y, size_x)
        self.offset = (min_z, min_y, min_x)


    def _pad_to_tile_multiple(self) -> None:
        """
        Pad unpadded_shape up to multiples of one tile (z_dim,Y,X).
        """
        tz, ty, tx = self.z_dim, self.Y, self.X
        sz, sy, sx = self.unpadded_shape

        pz = (-sz) % tz
        py = (-sy) % ty
        px = (-sx) % tx

        self.pad = (pz, py, px)
        self.padded_shape = (sz + pz, sy + py, sx + px)
        
    def _pad_to_chunk_multiple(self) -> None:
        """
        Pad unpadded_shape up to exact multiples of the codec chunk (self.chunk_y, self.chunk_x).
        """
        tz, th, tw = self.chunk_shape[-3:]  # (1, 512, 512) effectively
        sz, sy, sx = self.unpadded_shape

        pad_z = (-sz) % tz
        pad_y = (-sy) % th
        pad_x = (-sx) % tw

        self.pad = (pad_z, pad_y, pad_x)
        self.padded_shape = (sz + pad_z, sy + pad_y, sx + pad_x)


    def _create_fused_tensorstore(self,
                                  output_path: str | Path,
                                  z_slices_per_shard: int = 4) -> None:
        """
        Create the output TensorStore with larger shard-depth
        (grouping multiple Z-slices per shard) to improve HDD throughput.

        Parameters
        ----------
        output_path : str or Path
            Path for the fused Zarr v3 store.
        z_slices_per_shard : int, optional
            Number of Z slices to pack into each shard file. Default 4.
        """
        self.output_path = Path(output_path)

        # Full volume shape in T,P,C,Z,Y,X
        full_shape = [1, 1, self.channels, *self.padded_shape]

        # Shard will now group z_slices_per_shard Z-planes at once:
        shard_chunk = [
            1, 1, 1, z_slices_per_shard,
            self.padded_shape[1],
            self.padded_shape[2],
        ]
        self.shard_chunk = shard_chunk

        # Inside each shard, we still break into 512×512 blocks:
        codec_chunk = [
            1, 1, 1, z_slices_per_shard,
            self.chunk_y,
            self.chunk_x,
        ]

        config = {
            # bump up IO concurrency so TensorStore can pipeline reads/writes
            "context": {
                "file_io_concurrency": {"limit": self.max_workers},
                "data_copy_concurrency": {"limit": self.max_workers},
                "file_io_memmap": True,
                "file_io_sync": False,
            },
            "driver": "zarr3",
            "kvstore": {
                "driver": "file",
                "path": str(self.output_path)
            },
            "metadata": {
                "shape": full_shape,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": shard_chunk}
                },
                "chunk_key_encoding": {"name": "default"},
                "codecs": [
                    {
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": codec_chunk,
                            "codecs": [
                                {"name": "bytes",
                                 "configuration": {"endian": "little"}},
                                {"name": "blosc",
                                 "configuration": {
                                     "cname": "zstd",
                                     "clevel": 5,
                                     "shuffle": "bitshuffle"
                                 }}
                            ],
                            "index_codecs": [
                                {"name": "bytes",
                                 "configuration": {"endian": "little"}},
                                {"name": "crc32c"}
                            ],
                            "index_location": "end",
                        }
                    }
                ],
                "data_type": "uint16"
            }
        }

        self.fused_ts = ts.open(
            config, create=True, open=True
        ).result()

    def _find_overlaps(self, offsets):
        """
        Identify all pairwise overlapping regions in voxel space.

        Parameters
        ----------
        offsets : list of tuple of int
            Voxel offsets [(z0,y0,x0), ...] per tile.

        Returns
        -------
        overlaps : list of (i, j, region)
            region = (z0,z1,y0,y1,x0,x1)
        """
        overlaps = []
        n = len(offsets)

        for i in range(n):
            z0_i, y0_i, x0_i = offsets[i]
            for j in range(i + 1, n):
                z0_j, y0_j, x0_j = offsets[j]

                z1_i = z0_i + self.z_dim
                y1_i = y0_i + self.Y
                x1_i = x0_i + self.X

                z1_j = z0_j + self.z_dim
                y1_j = y0_j + self.Y
                x1_j = x0_j + self.X

                z0 = max(z0_i, z0_j)
                z1 = min(z1_i, z1_j)
                y0 = max(y0_i, y0_j)
                y1 = min(y1_i, y1_j)
                x0 = max(x0_i, x0_j)
                x1 = min(x1_i, x1_j)

                if z1 > z0 and y1 > y0 and x1 > x0:
                    overlaps.append((i, j, (z0, z1, y0, y1,
                                            x0, x1)))
        return overlaps


    def _blend_region(self,
                      i: int,
                      j: int,
                      region: tuple[int, int, int, int, int, int],
                      offsets: list[tuple[int, int, int]]) -> None:
        """
        Feather-blend one overlapping region between tiles i and j.

        Parameters
        ----------
        i, j : int
            Indices of the two tiles to blend.
        region : tuple of int
            (z0, z1, y0, y1, x0, x1) in global voxel coordinates.
        offsets : list of tuple of int
            Voxel offsets for each tile: [(z0, y0, x0), ...].
        """
        z0, z1, y0, y1, x0, x1 = region
        oz_i, oy_i, ox_i = offsets[i]
        oz_j, oy_j, ox_j = offsets[j]

        # Read sub-volumes from the two tiles (shape: C × dz × dy × dx)
        sub_i = self.ts_dataset[
            0, i, slice(None),
            slice(z0 - oz_i, z1 - oz_i),
            slice(y0 - oy_i, y1 - oy_i),
            slice(x0 - ox_i, x1 - ox_i)
        ].read().result().astype(np.float32)

        sub_j = self.ts_dataset[
            0, j, slice(None),
            slice(z0 - oz_j, z1 - oz_j),
            slice(y0 - oy_j, y1 - oy_j),
            slice(x0 - ox_j, x1 - ox_j)
        ].read().result().astype(np.float32)

        C, dz, dy, dx = sub_i.shape
        fused = np.empty((C, dz, dy, dx), dtype=np.float32)

        # Slice 1D feather profiles to local overlap extents
        zi_i = slice(z0 - oz_i, z1 - oz_i)
        yi_i = slice(y0 - oy_i, y1 - oy_i)
        xi_i = slice(x0 - ox_i, x1 - ox_i)

        zi_j = slice(z0 - oz_j, z1 - oz_j)
        yi_j = slice(y0 - oy_j, y1 - oy_j)
        xi_j = slice(x0 - ox_j, x1 - ox_j)

        wz_i = self.z_profile[zi_i]
        wy_i = self.y_profile[yi_i]
        wx_i = self.x_profile[xi_i]

        wz_j = self.z_profile[zi_j]
        wy_j = self.y_profile[yi_j]
        wx_j = self.x_profile[xi_j]

        # Blend each channel independently
        for c in range(C):
            buff = np.empty((dz, dy, dx), dtype=np.float32)
            fused[c] = _blend_numba(
                sub_i[c], sub_j[c],
                wz_i, wy_i, wx_i,
                wz_j, wy_j, wx_j,
                buff
            )

        # Write fused result back into the output store
        self.fused_ts[
            0, 0, slice(None),
            slice(z0, z1),
            slice(y0, y1),
            slice(x0, x1)
        ].write(fused.astype(np.uint16)).result()
        
        del sub_i, sub_j, fused


    def _copy_nonoverlap(self,
                         idx: int,
                         offsets: list[tuple[int, int, int]],
                         overlaps: list[tuple[int, int, tuple[int, int, int, int, int, int]]]
                         ) -> None:
        """
        Copy non-overlapping slabs of tile `idx` directly to the fused store.

        Parameters
        ----------
        idx : int
            Index of the tile whose exclusive regions to copy.
        offsets : list of tuple of int
            Voxel offsets for each tile.
        overlaps : list of (i, j, region)
            Overlap regions computed by _find_overlaps.
        """
        oz, oy, ox = offsets[idx]
        tz, ty, tx = self.z_dim, self.Y, self.X

        # Start with the full tile bounding box
        regions = [(oz, oz + tz, oy, oy + ty, ox, ox + tx)]

        # Subtract each overlap involving this tile
        for (i, j, (z0, z1, y0, y1, x0, x1)) in overlaps:
            if idx not in (i, j):
                continue
            new_regs = []
            for (rz0, rz1, ry0, ry1, rx0, rx1) in regions:
                if (x1 <= rx0 or x0 >= rx1 or
                    y1 <= ry0 or y0 >= ry1 or
                    z1 <= rz0 or z0 >= rz1):
                    # No intersection
                    new_regs.append((rz0, rz1, ry0, ry1, rx0, rx1))
                else:
                    # Carve out up to six sub-regions
                    if z0 > rz0:
                        new_regs.append((rz0, z0, ry0, ry1, rx0, rx1))
                    if z1 < rz1:
                        new_regs.append((z1, rz1, ry0, ry1, rx0, rx1))
                    if y0 > ry0:
                        new_regs.append((
                            max(rz0, z0),
                            min(rz1, z1),
                            ry0, y0,
                            rx0, rx1
                        ))
                    if y1 < ry1:
                        new_regs.append((
                            max(rz0, z0),
                            min(rz1, z1),
                            y1, ry1,
                            rx0, rx1
                        ))
                    if x0 > rx0:
                        new_regs.append((
                            max(rz0, z0),
                            min(rz1, z1),
                            max(ry0, y0),
                            min(ry1, y1),
                            rx0, x0
                        ))
                    if x1 < rx1:
                        new_regs.append((
                            max(rz0, z0),
                            min(rz1, z1),
                            max(ry0, y0),
                            min(ry1, y1),
                            x1, rx1
                        ))
            regions = new_regs

        # Copy each remaining exclusive region
        for (z0, z1, y0, y1, x0, x1) in regions:
            if z1 <= z0 or y1 <= y0 or x1 <= x0:
                continue
            block = self.ts_dataset[
                0, idx, slice(None),
                slice(z0 - oz, z1 - oz),
                slice(y0 - oy, y1 - oy),
                slice(x0 - ox, x1 - ox)
            ].read().result().astype(np.uint16)

            self.fused_ts[
                0, 0, slice(None),
                slice(z0, z1),
                slice(y0, y1),
                slice(x0, x1)
            ].write(block).result()
            
            del block

    def _fuse_tiles_cpu(self):
        """
        Perform CPU‐parallel fusion with bounded concurrency:
        1) blend overlaps,
        2) copy non‐overlaps,
        but never more than self.max_workers tasks allocate sub‐volumes at once.
        """
        # 1) compute voxel offsets
        offsets = []
        for z_phys, y_phys, x_phys in self.tile_positions:
            z0 = int((z_phys - self.offset[0]) / self.pixel_size[0])
            y0 = int((y_phys - self.offset[1]) / self.pixel_size[1])
            x0 = int((x_phys - self.offset[2]) / self.pixel_size[2])
            offsets.append((z0, y0, x0))

        # 2) find all overlap regions
        overlaps = self._find_overlaps(offsets)

        # 3) blend overlaps with bounded concurrency & progress bar
        blend_tasks = [(i, j, region, offsets) for i, j, region in overlaps]
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = set()
            pbar = tqdm(total=len(blend_tasks), desc="Blending Overlaps", unit="reg")
            it = iter(blend_tasks)
            # Prime the first batch
            for _ in range(min(self.max_workers, len(blend_tasks))):
                i, j, region, offs = next(it)
                futures.add(ex.submit(self._blend_region, i, j, region, offs))
            # As each finishes, schedule a new one
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for f in done:
                    f.result()
                    futures.remove(f)
                    pbar.update()
                    try:
                        i, j, region, offs = next(it)
                    except StopIteration:
                        continue
                    futures.add(ex.submit(self._blend_region, i, j, region, offs))
            pbar.close()

        # 4) copy non‐overlaps with bounded concurrency & progress bar
        copy_tasks = list(range(len(offsets)))
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = set()
            pbar = tqdm(total=len(copy_tasks), desc="Copying Tiles", unit="tile")
            it = iter(copy_tasks)
            for _ in range(min(self.max_workers, len(copy_tasks))):
                idx = next(it)
                futures.add(ex.submit(self._copy_nonoverlap, idx, offsets, overlaps))
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for f in done:
                    f.result()
                    futures.remove(f)
                    pbar.update()
                    try:
                        idx = next(it)
                    except StopIteration:
                        continue
                    futures.add(ex.submit(self._copy_nonoverlap, idx, offsets, overlaps))
            pbar.close()

    def _fuse_by_shard(self):
        """
        Shard‐centric fusion with Numba‐accelerated kernels, non-blocking writes.

        For each output shard:
        1. Read all overlapping tiles’ sub‐volumes.
        2. Accumulate weighted sums into one buffer.
        3. Normalize by weights.
        4. Issue a non-blocking write; collect its future.
        After looping all shards, wait on all futures.
        """
        # Compute integer voxel offsets for each tile
        offsets = [
            (
                int((z - self.offset[0]) / self.pixel_size[0]),
                int((y - self.offset[1]) / self.pixel_size[1]),
                int((x - self.offset[2]) / self.pixel_size[2])
            )
            for z, y, x in self.tile_positions
        ]

        z_step = self.shard_chunk[3]
        padded_Y = self.padded_shape[1]
        padded_X = self.padded_shape[2]
        nz = (self.padded_shape[0] + z_step - 1) // z_step

        write_futures = []

        for shard_idx in tqdm(range(nz), desc="Fusing Shards"):
            z0 = shard_idx * z_step
            z1 = min(z0 + z_step, self.padded_shape[0])
            dz = z1 - z0

            # Allocate buffers
            fused_block = np.zeros((self.channels, dz, padded_Y, padded_X),
                                dtype=np.float32)
            weight_sum  = np.zeros_like(fused_block)

            # Accumulate per‐tile
            for t_idx, (oz, oy, ox) in enumerate(offsets):
                tz0 = max(z0, oz)
                tz1 = min(z1, oz + self.z_dim)
                if tz1 <= tz0:
                    continue

                local_z0 = tz0 - oz
                local_z1 = tz1 - oz

                sub = self.ts_dataset[
                    0, t_idx, slice(None),
                    slice(local_z0, local_z1),
                    slice(0, self.Y),
                    slice(0, self.X)
                ].read().result().astype(np.float32)

                wz = self.z_profile[local_z0:local_z1]
                wy = self.y_profile
                wx = self.x_profile
                w3d = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]

                z_off = tz0 - z0
                y_off = oy
                x_off = ox

                _accumulate_tile_shard(
                    fused_block, weight_sum,
                    sub, w3d,
                    z_off, y_off, x_off
                )

            # Normalize
            _normalize_shard(fused_block, weight_sum)
            fused_block = fused_block.astype(np.uint16)

            # Issue non-blocking write and collect future
            fut = self.fused_ts[
                0, 0, slice(None),
                slice(z0, z1),
                slice(0, padded_Y),
                slice(0, padded_X)
            ].write(fused_block)
            write_futures.append(fut)

        # Wait for all writes to finish
        for fut in tqdm(write_futures, desc="Writing Shards", unit="shard"):
            fut.result()


    def run_cpu_fusion(self,
                       downsample_factors=(3, 5, 5),
                       ssim_window: int = 7,
                       threshold: float = 0.0,
                       output_path: str | Path = None):
        """
        Execute the full CPU fusion pipeline.

        Parameters
        ----------
        downsample_factors : tuple of int, optional
            Block‐reduce factors for registration.
        ssim_window : int, optional
            Window size for SSIM in registration.
        threshold : float, optional
            SSIM threshold to accept shifts.
        output_path : str or Path, optional
            Path for fused output Zarr. If None, uses previously set path.
        """
        # 1) registration + global optimization
        self.refine_tile_positions_with_cross_correlation(
            downsample_factors,
            ssim_window,
            0,  # channel index
            threshold
        )
        self.optimize_shifts()

        # 2) compute image space & pad
        self._compute_fused_image_space()
        self._pad_to_tile_multiple()

        # 3) create the fused store
        if output_path is not None:
            self.output_path = output_path
        self._create_fused_tensorstore(self.output_path)

        # 4) fuse tiles on CPU
        self._fuse_tiles_cpu()

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
        pixel_size = metadata['attributes']['deskewed_voxel_size_um'],
        blend_pixels= [20,300,300],
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
    tile_fuser.refine_tile_positions_with_cross_correlation(
            downsample_factors=[3,5,5]
        )
    tile_fuser.save_pairwise_metrics(metric_path)
    tile_fuser.optimize_shifts()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
        
    tile_fuser._compute_fused_image_space()
    tile_fuser._pad_to_chunk_multiple()
    tile_fuser._create_fused_tensorstore(output_path = output_path)
    tile_fuser._fuse_by_shard()