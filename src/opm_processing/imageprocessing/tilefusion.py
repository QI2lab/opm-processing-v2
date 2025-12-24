"""
2D/3D tile fusion for qi2lab OPM data.

This module implements a class with GPU, Numba, and CuPy‐accelerated kernels 
for tile registration and fusion of TPCZYX qi2lab-OPM stacks.
 
The final fused volume is written to a ome-ngff v0.5 datastore using tensorstore.
"""

import gc
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Sequence, Tuple, Union, Any, Dict, Optional

import numpy as np
import tensorstore as ts
from numba import njit, prange
from tqdm import trange

try:
    import cupy as cp  # type: ignore
    from cupyx.scipy.ndimage import shift as cp_shift  # type: ignore
    from cucim.skimage.exposure import match_histograms  # type: ignore
    from cucim.skimage.measure import block_reduce  # type: ignore
    from cucim.skimage.registration import phase_cross_correlation  # type: ignore
    from opm_processing.imageprocessing.ssim_cuda import (  # type: ignore
        structural_similarity_cupy_sep_shared as ssim_cuda,
    )

    xp = cp
    USING_GPU = True
except Exception:
    cp = None  # type: ignore
    cp_shift = None  # type: ignore
    from skimage.exposure import match_histograms  # type: ignore
    from skimage.measure import block_reduce  # type: ignore
    from skimage.registration import phase_cross_correlation  # type: ignore
    from scipy.ndimage import shift as _shift_cpu  # type: ignore
    from skimage.metrics import structural_similarity as _ssim_cpu  # type: ignore

    xp = np
    USING_GPU = False


def _shift_array(arr: Any, shift_vec: Any) -> Any:
    """Shift array using GPU if available, else CPU fallback."""
    if USING_GPU and cp_shift is not None:
        return cp_shift(arr, shift=shift_vec, order=1, prefilter=False)
    return _shift_cpu(arr, shift=shift_vec, order=1, prefilter=False)


def _ssim(arr1: Any, arr2: Any, win_size: int) -> float:
    """SSIM wrapper that routes to GPU kernel or CPU skimage."""
    if USING_GPU and 'ssim_cuda' in globals():
        return float(ssim_cuda(arr1, arr2, win_size=win_size))
    arr1_np = np.asarray(arr1)
    arr2_np = np.asarray(arr2)
    data_range = float(arr1_np.max() - arr1_np.min())
    if data_range == 0:
        data_range = 1.0
    return float(_ssim_cpu(arr1_np, arr2_np, win_size=win_size, data_range=data_range))


@njit(parallel=True)
def _accumulate_tile_shard(
    fused: np.ndarray,
    weight: np.ndarray,
    sub: np.ndarray,
    w3d: np.ndarray,
    z_off: int,
    y_off: int,
    x_off: int,
) -> None:
    """
    Weighted accumulation of a sub-volume into the fused buffer.

    Parameters
    ----------
    fused : float32[C, dz, Y, X]
        Accumulation buffer.
    weight : float32[C, dz, Y, X]
        Weight accumulation buffer.
    sub : float32[C, sub_dz, Y, X]
        Sub-volume to blend.
    w3d : float32[sub_dz, Y, X]
        Weight profile volume.
    z_off, y_off, x_off : int
        Offsets of sub-volume in the fused volume.
    """
    C, dz, Yp, Xp = fused.shape
    _, sub_dz, Y, X = sub.shape
    total = sub_dz * Y

    for idx in prange(total):
        dz_i = idx // Y
        y_i = idx % Y
        gz = z_off + dz_i
        gy = y_off + y_i
        w_line = w3d[dz_i, y_i]
        for c in range(C):
            sub_line = sub[c, dz_i, y_i]
            base_f = fused[c, gz, gy]
            base_w = weight[c, gz, gy]
            for x_i in range(X):
                gx = x_off + x_i
                w_val = w_line[x_i]
                base_f[gx] += sub_line[x_i] * w_val
                base_w[gx] += w_val


@njit(parallel=True)
def _normalize_shard(fused: np.ndarray, weight: np.ndarray) -> None:
    """
    Normalize the fused buffer by its weight buffer, in-place.

    Parameters
    ----------
    fused : float32[C, dz, Y, X]
        Accumulation buffer to normalize.
    weight : float32[C, dz, Y, X]
        Corresponding weights.
    """
    C, dz, Yp, Xp = fused.shape
    total = C * dz * Yp

    for idx in prange(total):
        c = idx // (dz * Yp)
        rem = idx % (dz * Yp)
        z_i = rem // Yp
        y_i = rem % Yp
        base_f = fused[c, z_i, y_i]
        base_w = weight[c, z_i, y_i]
        for x_i in range(Xp):
            w_val = base_w[x_i]
            base_f[x_i] = base_f[x_i] / w_val if w_val > 0 else 0.0


@njit(parallel=True)
def _blend_numba(
    sub_i: np.ndarray,
    sub_j: np.ndarray,
    wz_i: np.ndarray,
    wy_i: np.ndarray,
    wx_i: np.ndarray,
    wz_j: np.ndarray,
    wy_j: np.ndarray,
    wx_j: np.ndarray,
    out_f: np.ndarray,
) -> np.ndarray:
    """
    Feather-blend two overlapping sub-volumes.

    Parameters
    ----------
    sub_i, sub_j : (dz, dy, dx) float32
        Input sub-volumes.
    wz_i, wy_i, wx_i : 1D float32
        Weight profiles for sub_i.
    wz_j, wy_j, wx_j : 1D float32
        Weight profiles for sub_j.
    out_f : (dz, dy, dx) float32
        Pre-allocated output buffer.

    Returns
    -------
    out_f : (dz, dy, dx) float32
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
                    out_f[z, y, x] = (wi * sub_i[z, y, x] + wj * sub_j[z, y, x]) / tot
                else:
                    out_f[z, y, x] = sub_i[z, y, x]
    return out_f


class TileFusion:
    """
    GPU-accelerated tile registration and fusion for 3D ZYX stacks.

    Parameters
    ----------
    root_path : str or Path
        Path to the base Zarr store for fusion.
    blend_pixels : tuple of int
        Feather widths (bz, by, bx).
    downsample_factors : tuple of int
        Block-reduce factors for registration.
    ssim_window : int
        Window size for SSIM.
    threshold : float
        SSIM acceptance threshold.
    multiscale_factors : sequence of int
        Downsampling factors for multiscale.
    resolution_multiples : sequence of int or tuple
        Spatial resolution multiples per axis.
        max_workers : int
        Maximum parallel I/O workers.
    debug : bool
        If True, prints debug info.
    metrics_filename : str
        Filename for storing registration metrics.
    channel_to_use : int
        Channel index for registration.
    multiscale_downsample : str
        Either "stride" (default) or "block_mean" to control multiscale reduction.
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        blend_pixels: Tuple[int, int, int] = (20, 600, 400),
        downsample_factors: Tuple[int, int, int] = (3, 5, 5),
        ssim_window: int = 15,
        threshold: float = 0.7,
        multiscale_factors: Sequence[int] = (2, 4, 8, 16, 32),
        resolution_multiples: Sequence[Union[int, Sequence[int]]] = (
            (1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8), (16,16,16), (32,32,32)
        ),
        max_workers: int = 8,
        debug: bool = False,
        metrics_filename: str = "metrics.json",
        channel_to_use: int = 0,
        multiscale_downsample: str = "stride",
    ):

        self.root = Path(root_path)
        base = self.root.parents[0]
        stem = self.root.stem

        desk = base / f"{stem}_decon_deskewed.zarr"
        if not desk.exists():
            desk = base / f"{stem}_deskewed.zarr"
            if not desk.exists():
                raise FileNotFoundError("Deskewed data store not found.")
        self.deskewed = desk

        with open(self.deskewed / "zarr.json", "r") as f:
            meta = json.load(f)
        ds = ts.open({
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(self.deskewed)},
        }).result()

        self._tile_positions = [
            tuple(meta["attributes"]["per_index_metadata"][str(t)][str(p)]["0"]["stage_position"])
            for t in range(ds.shape[0])
            for p in range(ds.shape[1])
        ]
        self._pixel_size = tuple(meta["attributes"]["deskewed_voxel_size_um"])

        self.downsample_factors = tuple(downsample_factors)
        self.ssim_window = int(ssim_window)
        self.threshold = float(threshold)
        self.multiscale_factors = tuple(multiscale_factors)
        self.resolution_multiples = [
            r if hasattr(r, "__len__") else (r, r, r)
            for r in resolution_multiples
        ]
        self._max_workers = int(max_workers)
        self._debug = bool(debug)
        self.metrics_filename = metrics_filename
        self._blend_pixels = tuple(blend_pixels)
        self.channel_to_use = channel_to_use
        if multiscale_downsample not in ("stride", "block_mean"):
            raise ValueError('multiscale_downsample must be "stride" or "block_mean".')
        self.multiscale_downsample = multiscale_downsample

        spec = {
            "context": {
                "file_io_concurrency": {"limit": self._max_workers},
                "data_copy_concurrency": {"limit": self._max_workers},
            },
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(self.deskewed)},
        }
        ts_full = ts.open(spec, create=False, open=True).result()
        self.ts = ts_full

        shape = self.ts.shape
        if len(shape) == 6:
            (
                self.time_dim,
                self.position_dim,
                self.channels,
                self.z_dim,
                self.Y,
                self.X,
            ) = shape
        elif len(shape) == 5:
            self.time_dim, self.position_dim, self.channels, self.Y, self.X = shape
            self.z_dim = 1
        else:
            raise ValueError(f"Unsupported data rank {len(shape)}; expected 5 or 6.")

        self._is_2d = self.z_dim == 1

        self._update_profiles()
        self.chunk_shape = (1, 1, 1, 1024, 1024)
        self.chunk_y, self.chunk_x = self.chunk_shape[-2:]

        self.pairwise_metrics: Dict[Tuple[int,int], Tuple[int,int,int,float]] = {}
        self.global_offsets: Optional[np.ndarray] = None
        self.offset: Optional[Tuple[float,float,float]] = None
        self.unpadded_shape: Optional[Tuple[int,int,int]] = None
        self.padded_shape: Optional[Tuple[int,int,int]] = None
        self.pad = (0, 0, 0)
        self.fused_ts = None
    
    @property
    def tile_positions(self) -> List[Tuple[float, float, float]]:
        """
        Stage positions for each tile (z, y, x).
        """
        return self._tile_positions

    @tile_positions.setter
    def tile_positions(self, positions: Sequence[Tuple[float, float, float]]):
        if any(len(p) != 3 for p in positions):
            raise ValueError("Each position must be a 3-tuple.")
        self._tile_positions = [tuple(p) for p in positions]

    @property
    def pixel_size(self) -> Tuple[float, float, float]:
        """
        Voxel size in (z, y, x).
        """
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, size: Tuple[float, float, float]):
        if len(size) != 3:
            raise ValueError("pixel_size must be a 3-tuple.")
        self._pixel_size = tuple(float(x) for x in size)

    @property
    def blend_pixels(self) -> Tuple[int, int, int]:
        """
        Feather widths in (bz, by, bx).
        """
        return self._blend_pixels

    @blend_pixels.setter
    def blend_pixels(self, bp: Tuple[int, int, int]):
        if len(bp) != 3:
            raise ValueError("blend_pixels must be a 3-tuple.")
        self._blend_pixels = tuple(bp)
        self._update_profiles()

    @property
    def max_workers(self) -> int:
        """
        Maximum concurrent I/O workers.
        """
        return self._max_workers

    @max_workers.setter
    def max_workers(self, mw: int):
        if mw < 1:
            raise ValueError("max_workers must be >= 1.")
        self._max_workers = int(mw)

    @property
    def debug(self) -> bool:
        """
        Debug flag for verbose logging.
        """
        return self._debug

    @debug.setter
    def debug(self, flag: bool):
        self._debug = bool(flag)

    def _update_profiles(self) -> None:
        """
        Recompute 1D feather profiles from blend_pixels.
        """
        bz, by, bx = self._blend_pixels
        self.z_profile = self._make_1d_profile(self.z_dim, bz)
        self.y_profile = self._make_1d_profile(self.Y, by)
        self.x_profile = self._make_1d_profile(self.X, bx)

    @staticmethod
    def _make_1d_profile(length: int, blend: int) -> np.ndarray:
        """
        Create a linear ramp profile over `blend` voxels at each end.

        Parameters
        ----------
        length : int
            Number of voxels.
        blend : int
            Ramp width.

        Returns
        -------
        prof : (length,) float32
            Linear profile.
        """
        blend = min(blend, length)
        prof = np.ones(length, dtype=np.float32)
        if blend > 0:
            ramp = np.linspace(0, 1, blend, endpoint=False, dtype=np.float32)
            prof[:blend] = ramp
            prof[-blend:] = ramp[::-1]
        return prof

    def _read_tile_volume(
        self,
        tile_idx: int,
        ch_sel: Union[int, slice],
        z_slice: slice,
        y_slice: slice,
        x_slice: slice,
    ) -> np.ndarray:
        """
        Read a tile subvolume, padding 2D data with a singleton z axis.
        """
        if self._is_2d:
            arr = self.ts[0, tile_idx, ch_sel, y_slice, x_slice].read().result()
            if arr.ndim == 2:
                arr = arr[None, None, :, :]
            elif arr.ndim == 3:
                arr = arr[:, None, :, :]
        else:
            arr = self.ts[0, tile_idx, ch_sel, z_slice, y_slice, x_slice].read().result()
        return arr.astype(np.float32)

    @staticmethod
    def register_and_score(
        g1: Any,
        g2: Any,
        win_size: int,
        debug: bool = True,
    ) -> Union[Tuple[Tuple[float, float, float], float], Tuple[None, None]]:
        """
        Histogram-match g2→g1, compute subpixel shift, and SSIM.

        Parameters
        ----------
        g1, g2 : array-like
            Fixed and moving patches (ZYX).
        win_size : int
            SSIM window.
        debug : bool
            If True, print intermediate info.

        Returns
        -------
        shift : (dz, dy, dx)
            Subpixel shift.
        ssim_val : float
            SSIM score.
        """
        arr1 = xp.asarray(g1, dtype=xp.float32)
        arr2 = xp.asarray(g2, dtype=xp.float32)
        # squeeze leading singleton channel if present
        while arr1.ndim > 2 and arr1.shape[0] == 1:
            arr1 = arr1[0]
            arr2 = arr2[0]

        arr2 = match_histograms(arr2, arr1)
        shift, _, _ = phase_cross_correlation(
            arr1,
            arr2,
            disambiguate=True,
            normalization="phase",
            upsample_factor=10,
            overlap_ratio=0.5,
        )
        if arr1.ndim == 2 and len(shift) == 2:
            shift_apply = xp.asarray(shift, dtype=xp.float32)
            shift_ret = xp.asarray([0.0, shift[0], shift[1]], dtype=xp.float32)
        else:
            shift_apply = xp.asarray(shift, dtype=xp.float32)
            shift_ret = shift_apply
        g2s = _shift_array(arr2, shift_vec=shift_apply)
        ssim_val = _ssim(arr1, g2s, win_size=win_size)
        out_shift = cp.asnumpy(shift_ret) if USING_GPU else np.asarray(shift_ret)
        return tuple(float(s) for s in out_shift), float(ssim_val)
    
    def refine_tile_positions_with_cross_correlation(
        self,
        downsample_factors: Tuple[int, int, int] = None,
        ssim_window: int = None,
        ch_idx: int = 0,
        threshold: float = None
    ) -> None:
        """
        Detect and score overlaps between each tile pair via cross-correlation.

        Parameters
        ----------
        downsample_factors : tuple of int, optional
            Block-reduce factors for registration.
        ssim_window : int, optional
            Window size for SSIM.
        ch_idx : int, optional
            Channel to use.
        threshold : float, optional
            SSIM threshold to accept a link.
        """
        df = downsample_factors or self.downsample_factors
        if len(df) == 3:
            df = (1, *df[1:]) if self.z_dim == 1 else tuple(df)
        elif len(df) == 4:
            df = tuple(df)
        else:
            raise ValueError("downsample_factors must have length 3 or 4")
        # Ensure block size aligns with channel-first arrays (C, Z, Y, X)
        reduce_block = None
        sw = ssim_window       or self.ssim_window
        th = threshold         or self.threshold
        self.pairwise_metrics.clear()
        n_pos = self.position_dim
        executor = ThreadPoolExecutor(max_workers=2)

        def bounds_1d(off, length):
            return max(0, off), min(length, off + length)

        for t in range(self.time_dim):
            base = t * n_pos
            for i_pos in trange(n_pos, desc="register", leave=True):
                i = base + i_pos
                for j_pos in range(i_pos + 1, n_pos):
                    j = base + j_pos
                    phys = (np.array(self._tile_positions[j]) -
                            np.array(self._tile_positions[i]))
                    vox_off = np.round(phys / np.array(self._pixel_size)).astype(int)
                    dz, dy, dx = vox_off
                    bounds_i = [bounds_1d(dz, self.z_dim),
                                bounds_1d(dy, self.Y),
                                bounds_1d(dx, self.X)]
                    bounds_j = [bounds_1d(-dz, self.z_dim),
                                bounds_1d(-dy, self.Y),
                                bounds_1d(-dx, self.X)]

                    if any(hi <= lo for lo, hi in bounds_i):
                        continue

                    def read_patch(idx, bnds):
                        z0, z1 = bnds[0]
                        y0, y1 = bnds[1]
                        x0, x1 = bnds[2]
                        return self._read_tile_volume(
                            idx,
                            ch_idx,
                            slice(z0, z1),
                            slice(y0, y1),
                            slice(x0, x1),
                        )

                    patch_i = executor.submit(read_patch, i, bounds_i).result()
                    patch_j = executor.submit(read_patch, j, bounds_j).result()

                    arr_i = xp.asarray(patch_i)
                    arr_j = xp.asarray(patch_j)
                    if reduce_block is None:
                        # prepend 1 for channel axis if needed
                        reduce_block = (
                            (1,) + tuple(df)
                            if arr_i.ndim == len(df) + 1
                            else tuple(df)
                        )
                    g1 = block_reduce(arr_i, reduce_block, xp.mean)
                    g2 = block_reduce(arr_j, reduce_block, xp.mean)
                    shift_ds, ssim_val = self.register_and_score(
                        g1, g2, win_size=sw, debug=self._debug
                    )
                    if shift_ds is None:
                        continue
                    score = float(max(ssim_val, 1e-6))
                    if th != 0.0 and score < th:
                        continue

                    dz_s, dy_s, dx_s = [int(np.round(shift_ds[k] * df[k])) for k in range(3)]
                    max_shift = (20, 50, 100)  # adjust to your expected maximum neighbor‐tile displacement

                    if abs(dz_s) > max_shift[0] or abs(dy_s) > max_shift[1] or abs(dx_s) > max_shift[2]:
                        if self._debug:
                            print(f"Dropping link {(i,j)} shift={(dz_s,dy_s,dx_s)} — exceeds max {max_shift}")
                        continue

                    self.pairwise_metrics[(i, j)] = (
                        dz_s, dy_s, dx_s, round(score, 3)
                    )

        executor.shutdown(wait=True)

    @staticmethod
    def _solve_global(
        links: List[Dict[str, Any]],
        n_tiles: int,
        fixed_indices: List[int]
    ) -> np.ndarray:
        """
        Solve a linear least-squares for all 3 axes at once,
        given weighted pairwise links and fixed tile indices.
        """
        shifts = np.zeros((n_tiles, 3), dtype=np.float64)
        for axis in range(3):
            m = len(links) + len(fixed_indices)
            A = np.zeros((m, n_tiles), dtype=np.float64)
            b = np.zeros(m, dtype=np.float64)
            row = 0
            for link in links:
                i, j = link['i'], link['j']
                t, w = link['t'][axis], link['w']
                A[row, j] = w
                A[row, i] = -w
                b[row] = w * t
                row += 1
            for idx in fixed_indices:
                A[row, idx] = 1.0
                b[row] = 0.0
                row += 1
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            shifts[:, axis] = sol
        return shifts

    def _two_round_opt(
        self,
        links: List[Dict[str, Any]],
        n_tiles: int,
        fixed_indices: List[int],
        rel_thresh: float,
        abs_thresh: float,
        iterative: bool
    ) -> np.ndarray:
        """
        Perform two-round (or iterative two-round) robust optimization:
        1. Solve on all links.
        2. Remove any link whose residual > max(abs_thresh, rel_thresh * median(residuals)).
        3. Re-solve on the remaining links.
        If iterative=True, repeat step 2 + 3 until no more links are removed.
        """
        shifts = self._solve_global(links, n_tiles, fixed_indices)

        def compute_res(ls: List[Dict[str, Any]], sh: np.ndarray) -> np.ndarray:
            return np.array([
                np.linalg.norm(sh[l['j']] - sh[l['i']] - l['t'])
                for l in ls
            ])

        work = links.copy()
        res = compute_res(work, shifts)
        cutoff = max(abs_thresh, rel_thresh * np.median(res))
        outliers = set(np.where(res > cutoff)[0])

        if iterative:
            while outliers:
                for k in sorted(outliers, reverse=True):
                    work.pop(k)
                shifts = self._solve_global(work, n_tiles, fixed_indices)
                res = compute_res(work, shifts)
                cutoff = max(abs_thresh, rel_thresh * np.median(res))
                outliers = set(np.where(res > cutoff)[0])
        else:
            for k in sorted(outliers, reverse=True):
                work.pop(k)
            shifts = self._solve_global(work, n_tiles, fixed_indices)

        return shifts

    def optimize_shifts(
        self,
        method: str = 'ONE_ROUND',
        rel_thresh: float = 0.3,
        abs_thresh: float = 5.0,
        iterative: bool = False
    ) -> None:
        """
        Globally optimize tile shifts using either:
          - ONE_ROUND: single least-squares solve, or
          - TWO_ROUND_SIMPLE: remove outliers once then re-solve, or
          - TWO_ROUND_ITERATIVE: remove outliers repeatedly until none remain.

        Parameters
        ----------
        method : {'ONE_ROUND', 'TWO_ROUND_SIMPLE', 'TWO_ROUND_ITERATIVE'}
        rel_thresh : float
            Relative threshold (fraction of median residual) for link removal.
        abs_thresh : float
            Absolute threshold for link removal.
        iterative : bool
            If True, repeat outlier removal until convergence.
        """
        links: List[Dict[str, Any]] = []
        for (i, j), v in self.pairwise_metrics.items():
            links.append({
                'i': i,
                'j': j,
                't': np.array(v[:3], dtype=np.float64),
                'w': np.sqrt(v[3])
            })
        if not links:
            self.global_offsets = np.zeros((self.position_dim, 3), dtype=np.float64)
            return

        n = len(self._tile_positions)
        fixed = [0]  # by default, fix tile index 0 at zero shift

        if method == 'ONE_ROUND':
            d_opt = self._solve_global(links, n, fixed)
        elif method.startswith('TWO_ROUND'):
            d_opt = self._two_round_opt(
                links,
                n,
                fixed,
                rel_thresh,
                abs_thresh,
                method.endswith('ITERATIVE')
            )
        else:
            raise ValueError(f"Unknown method {method}")

        self.global_offsets = d_opt
    
    def save_pairwise_metrics(self, filepath: Union[str, Path]) -> None:
        """
        Save pairwise_metrics to a JSON file.

        Parameters
        ----------
        filepath : str or Path
            Output JSON path.
        """
        path = Path(filepath)
        out = {f"{i},{j}": list(v)
               for (i, j), v in self.pairwise_metrics.items()}
        with open(path, "w") as f:
            json.dump(out, f)

    def load_pairwise_metrics(self, filepath: Union[str, Path]) -> None:
        """
        Load pairwise_metrics from a JSON file.

        Parameters
        ----------
        filepath : str or Path
            Input JSON path.
        """
        path = Path(filepath)
        with open(path, "r") as f:
            data = json.load(f)
        self.pairwise_metrics = {
            tuple(map(int, k.split(","))): tuple(v)
            for k, v in data.items()
        }

    def _compute_fused_image_space(self) -> None:
        """
        Compute fused volume physical shape and offset based on tile positions.
        """
        pos = np.array(self._tile_positions)
        min_z, min_y, min_x = pos.min(axis=0)
        max_z = pos[:, 0].max() + self.z_dim * self._pixel_size[0]
        max_y = pos[:, 1].max() + self.Y    * self._pixel_size[1]
        max_x = pos[:, 2].max() + self.X    * self._pixel_size[2]

        sz = int(np.ceil((max_z - min_z) / self._pixel_size[0]))
        sy = int(np.ceil((max_y - min_y) / self._pixel_size[1]))
        sx = int(np.ceil((max_x - min_x) / self._pixel_size[2]))

        self.unpadded_shape = (sz, sy, sx)
        self.offset = (min_z, min_y, min_x)
        self.center = (
            (max_x - min_x) / 2,
            (max_y - min_y) / 2,
            (max_z - min_z) / 2
        )

    def _pad_to_chunk_multiple(self) -> None:
        """
        Pad unpadded_shape to exact multiples of tile shape (z_dim, Y, X).
        """
        tz, ty, tx = self.z_dim, self.Y, self.X
        sz, sy, sx = self.unpadded_shape

        pz = (-sz) % tz
        py = (-sy) % ty
        px = (-sx) % tx

        self.pad = (pz, py, px)
        self.padded_shape = (sz + pz, sy + py, sx + px)

    def _create_fused_tensorstore(
        self,
        output_path: Union[str, Path],
        z_slices_per_shard: int = 4
    ) -> None:
        """
        Create the output Zarr v3 store for the fused volume.

        Parameters
        ----------
        output_path : str or Path
            Path to create fused store.
        z_slices_per_shard : int
            Z-depth per shard.
        """
        out = Path(output_path)
        full_shape = [1, self.channels, *self.padded_shape]
        shard_chunk = [1, 1, z_slices_per_shard,
                       self.chunk_y * 2, self.chunk_x * 2]
        codec_chunk = [1, 1, 1, self.chunk_y, self.chunk_x]
        self.shard_chunk = shard_chunk

        config = {
            "context": {
                "file_io_concurrency": {"limit": self.max_workers},
                "data_copy_concurrency": {"limit": self.max_workers},
            },
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(out)},
            "metadata": {
                "shape": full_shape,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": shard_chunk}
                },
                "chunk_key_encoding": {"name": "default"},
                "codecs": [{
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
                        "index_location": "end"
                    }
                }],
                "data_type": "uint16",
                "dimension_names": ["t","c","z","y","x"]
            }
        }

        self.fused_ts = ts.open(config, create=True, open=True).result()

    def _find_overlaps(
        self,
        offsets: List[Tuple[int, int, int]]
    ) -> List[Tuple[int, int, Tuple[int, int, int, int, int, int]]]:
        """
        Identify overlapping regions between all tile pairs.

        Parameters
        ----------
        offsets : list of (z0, y0, x0)
            Tile voxel offsets.

        Returns
        -------
        overlaps : list of (i, j, region)
            region = (z0, z1, y0, y1, x0, x1)
        """
        overlaps: List[Tuple[int,int,Tuple[int,int,int,int,int,int]]] = []
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
                    overlaps.append((i, j, (z0, z1, y0, y1, x0, x1)))
        return overlaps

    def _blend_region(
        self,
        i: int,
        j: int,
        region: Tuple[int, int, int, int, int, int],
        offsets: List[Tuple[int, int, int]]
    ) -> None:
        """
        Feather-blend one overlapping region between tiles i and j.

        Parameters
        ----------
        i, j : int
            Tile indices.
        region : (z0, z1, y0, y1, x0, x1)
            Global voxel bounds of overlap.
        offsets : list of (z0, y0, x0)
            Tile voxel offsets.
        """
        z0, z1, y0, y1, x0, x1 = region
        oz_i, oy_i, ox_i = offsets[i]
        oz_j, oy_j, ox_j = offsets[j]

        sub_i = self._read_tile_volume(
            i,
            slice(None),
            slice(z0 - oz_i, z1 - oz_i),
            slice(y0 - oy_i, y1 - oy_i),
            slice(x0 - ox_i, x1 - ox_i),
        )
        sub_j = self._read_tile_volume(
            j,
            slice(None),
            slice(z0 - oz_j, z1 - oz_j),
            slice(y0 - oy_j, y1 - oy_j),
            slice(x0 - ox_j, x1 - ox_j),
        )

        C, dz, dy, dx = sub_i.shape
        fused = np.empty((C, dz, dy, dx), dtype=np.float32)

        zi_i = slice(z0 - oz_i, z1 - oz_i)
        yi_i = slice(y0 - oy_i, y1 - oy_i)
        xi_i = slice(x0 - ox_i, x1 - ox_i)
        zi_j = slice(z0 - oz_j, z1 - oz_j)
        yi_j = slice(y0 - oy_j, y1 - oy_j)
        xi_j = slice(x0 - ox_j, x1 - ox_j)

        wz_i, wy_i, wx_i = (
            self.z_profile[zi_i],
            self.y_profile[yi_i],
            self.x_profile[xi_i]
        )
        wz_j, wy_j, wx_j = (
            self.z_profile[zi_j],
            self.y_profile[yi_j],
            self.x_profile[xi_j]
        )

        for c in range(C):
            buf = np.empty((dz, dy, dx), dtype=np.float32)
            fused[c] = _blend_numba(
                sub_i[c], sub_j[c],
                wz_i, wy_i, wx_i,
                wz_j, wy_j, wx_j,
                buf
            )

        self.fused_ts[
            0, slice(None),
            slice(z0, z1),
            slice(y0, y1),
            slice(x0, x1)
        ].write(fused.astype(np.uint16)).result()

    def _copy_nonoverlap(
        self,
        idx: int,
        offsets: List[Tuple[int, int, int]],
        overlaps: List[Tuple[int, int, Tuple[int, int, int, int, int, int]]]
    ) -> None:
        """
        Copy non-overlapping slabs of tile `idx` directly to fused store.

        Parameters
        ----------
        idx : int
            Tile index.
        offsets : list of (z0, y0, x0)
            Tile voxel offsets.
        overlaps : list of (i, j, region)
            Overlap regions.
        """
        oz, oy, ox = offsets[idx]
        tz, ty, tx = self.z_dim, self.Y, self.X
        regions = [(oz, oz + tz, oy, oy + ty, ox, ox + tx)]

        for (i, j, (z0, z1, y0, y1, x0, x1)) in overlaps:
            if idx not in (i, j):
                continue
            new_regs = []
            for (rz0, rz1, ry0, ry1, rx0, rx1) in regions:
                if (x1 <= rx0 or x0 >= rx1 or
                    y1 <= ry0 or y0 >= ry1 or
                    z1 <= rz0 or z0 >= rz1):
                    new_regs.append((rz0, rz1, ry0, ry1, rx0, rx1))
                else:
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

        for (z0, z1, y0, y1, x0, x1) in regions:
            if z1 <= z0 or y1 <= y0 or x1 <= x0:
                continue
            block = self._read_tile_volume(
                idx,
                slice(None),
                slice(z0 - oz, z1 - oz),
                slice(y0 - oy, y1 - oy),
                slice(x0 - ox, x1 - ox),
            ).astype(np.uint16)
            self.fused_ts[
                0, slice(None),
                slice(z0, z1),
                slice(y0, y1),
                slice(x0, x1)
            ].write(block).result()

    def _fuse_by_shard(self) -> None:
        """
        Shard-centric fusion, channel by channel to cap memory.
        """
        offsets = [
            (
                int((z - self.offset[0]) / self._pixel_size[0]),
                int((y - self.offset[1]) / self._pixel_size[1]),
                int((x - self.offset[2]) / self._pixel_size[2])
            )
            for (z, y, x) in self._tile_positions
        ]
        z_step = self.shard_chunk[2]
        pad_Y, pad_X = self.padded_shape[1], self.padded_shape[2]
        nz = (self.padded_shape[0] + z_step - 1) // z_step

        futures = []

        for shard_idx in trange(nz, desc="scale0", leave=True):
            z0 = shard_idx * z_step
            z1 = min(z0 + z_step, self.padded_shape[0])
            dz = z1 - z0

            # process each channel independently
            for c in range(self.channels):
                # 1-channel accum buffers
                fused_block = np.zeros((1, dz, pad_Y, pad_X), dtype=np.float32)
                weight_sum  = np.zeros_like(fused_block)

                # accumulate every tile into this channel
                for t_idx, (oz, oy, ox) in enumerate(offsets):
                    tz0 = max(z0, oz)
                    tz1 = min(z1, oz + self.z_dim)
                    if tz1 <= tz0:
                        continue

                    local_z0, local_z1 = tz0 - oz, tz1 - oz
                    # read only channel c
                    sub = self._read_tile_volume(
                        t_idx,
                        slice(c, c + 1),
                        slice(local_z0, local_z1),
                        slice(0, self.Y),
                        slice(0, self.X),
                    )

                    wz = self.z_profile[local_z0:local_z1]
                    wy = self.y_profile
                    wx = self.x_profile
                    w3d = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]

                    z_off = tz0 - z0
                    _accumulate_tile_shard(
                        fused_block, weight_sum,
                        sub, w3d,
                        z_off, oy, ox
                    )

                # normalize and schedule write for this channel
                _normalize_shard(fused_block, weight_sum)
                fut = self.fused_ts[
                    0, slice(c, c+1),
                    slice(z0, z1),
                    slice(0, pad_Y),
                    slice(0, pad_X)
                ].write(fused_block.astype(np.uint16))
                futures.append(fut)

                # immediate cleanup
                del fused_block, weight_sum
                gc.collect()
                if USING_GPU and cp is not None:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()

        # wait for all writes to finish
        for fut in futures:
            fut.result()

    def _create_multiscales(
        self,
        omezarr_path: Path,
        factors: Sequence[int] = (2, 4, 8),
        z_slices_per_shard: int = 4
    ) -> None:
        """
        Build NGFF multiscales by downsampling Z/Y/X iteratively.

        Parameters
        ----------
        omezarr_path : Path
            Root of the NGFF group.
        factors : sequence of int
            Downsampling factors per scale.
        z_slices_per_shard : int
            Z-depth grouping for shards.
        """
        inp = None
        for idx, factor in enumerate(factors):
            out_path = omezarr_path / f"scale{idx+1}" / "image"
            if inp is not None:
                del inp
            prev = omezarr_path / f"scale{idx}" / "image"
            inp = ts.open({
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": str(prev)}
            }).result()

            factor_to_use = (factors[idx] // factors[idx-1]
                             if idx > 0 else factors[0])
            _, _, Z, Y, X = inp.shape
            z_factor = factor_to_use if not self._is_2d else 1
            new_z = max(1, Z // z_factor)
            new_y, new_x = Y // factor_to_use, X // factor_to_use
            shard_z = min(z_slices_per_shard, new_z)

            # choose chunk_y, chunk_x
            chunk_y = (1024 if new_y >= 2048 else
                       new_y // 4 if new_y >= 4 else 1)
            chunk_x = (1024 if new_x >= 2048 else
                       new_x // 4 if new_x >= 4 else 1)

            self.padded_shape = (new_z, new_y, new_x)
            self.chunk_y, self.chunk_x = chunk_y, chunk_x

            self._create_fused_tensorstore(
                output_path=out_path,
                z_slices_per_shard=shard_z
            )

            for z0 in trange(0, new_z, shard_z, desc=f'scale{idx+1}', leave=True):
                bz = min(shard_z, new_z - z0)
                in_z0 = z0 * z_factor
                in_z1 = min(Z, (z0 + bz) * z_factor)
                for y0 in range(0, new_y, chunk_y):
                    by = min(chunk_y, new_y - y0)
                    in_y0 = y0 * factor_to_use
                    in_y1 = min(Y, (y0 + by) * factor_to_use)
                    for x0 in range(0, new_x, chunk_x):
                        bx = min(chunk_x, new_x - x0)
                        in_x0 = x0 * factor_to_use
                        in_x1 = min(X, (x0 + bx) * factor_to_use)

                        slab = inp[:, :, in_z0:in_z1, in_y0:in_y1, in_x0:in_x1].read().result()
                        if self.multiscale_downsample == "stride":
                            down = slab[..., ::z_factor, ::factor_to_use, ::factor_to_use]
                        else:
                            arr = xp.asarray(slab)
                            block = (1, 1, z_factor, factor_to_use, factor_to_use)
                            down_arr = block_reduce(arr, block_size=block, func=xp.mean)
                            down = cp.asnumpy(down_arr) if USING_GPU and cp is not None else np.asarray(down_arr)
                        down = down.astype(slab.dtype, copy=False)
                        self.fused_ts[
                            :, :,
                            z0:z0 + bz,
                            y0:y0 + by,
                            x0:x0 + bx
                        ].write(down).result()

            ngff = {
                "attributes": {"_ARRAY_DIMENSIONS": ["t", "c", "z", "y", "x"]},
                "zarr_format": 3,
                "consolidated_metadata": "null",
                "node_type": "group",
            }
            with open(omezarr_path / f"scale{idx+1}" / "zarr.json", "w") as f:
                json.dump(ngff, f, indent=2)

    def _generate_ngff_zarr3_json(
        self,
        omezarr_path: Path,
        resolution_multiples: Sequence[Union[int, Sequence[int]]],
        dataset_name: str = "image",
        version: str = "0.5"
    ) -> None:
        """
        Write OME-NGFF v0.5 multiscales JSON for Zarr3.

        Parameters
        ----------
        omezarr_path : Path
            Root path of the NGFF group.
        resolution_multiples : sequence
            Resolution factors per scale.
        dataset_name : str
            Name of the dataset node.
        version : str
            NGFF version.
        """
        axes = [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ]
        norm_res = [
            tuple(r) if hasattr(r, "__len__") else (r, r, r)
            for r in resolution_multiples
        ]
        base_scale = [1.0, 1.0] + [float(s) for s in self._pixel_size]
        trans = [0.0, 0.0] + list(self.center)

        datasets = []
        prev_sp = base_scale[2:]
        for lvl, factors in enumerate(norm_res):
            spatial = [base_scale[i + 2] * factors[i] for i in range(3)]
            scale = [1.0, 1.0] + spatial
            if lvl == 0:
                translation = trans
            else:
                translation = [
                    0.0,
                    0.0,
                    datasets[-1]["coordinateTransformations"][1]["translation"][2] + 0.5 * prev_sp[0],
                    datasets[-1]["coordinateTransformations"][1]["translation"][3] + 0.5 * prev_sp[1],
                    datasets[-1]["coordinateTransformations"][1]["translation"][4] + 0.5 * prev_sp[2],
                ]
            datasets.append({
                "path": f"scale{lvl}/{dataset_name}",
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale},
                    {"type": "translation", "translation": translation},
                ],
            })
            prev_sp = spatial

        mult = {
            "axes": axes,
            "datasets": datasets,
            "name": dataset_name,
            "@type": "ngff:Image",
        }
        metadata = {
            "attributes": {"ome": {"version": version, "multiscales": [mult]}},
            "zarr_format": 3,
            "node_type": "group",
        }
        with open(omezarr_path / "zarr.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def run(self) -> None:
        """
        Execute the full tile fusion pipeline end-to-end.
        """
        base = self.root.parents[0]
        metrics_path = base / self.metrics_filename

        try:
            self.load_pairwise_metrics(metrics_path)
            self.optimize_shifts()
        except FileNotFoundError:
            self.refine_tile_positions_with_cross_correlation(
                downsample_factors=self.downsample_factors,
                ch_idx=self.channel_to_use,
                threshold=self.threshold)
            self.save_pairwise_metrics(metrics_path)

        self.optimize_shifts(
            method="TWO_ROUND_ITERATIVE",
            rel_thresh=.5,
            abs_thresh=1.5,
            iterative=True
        )
        gc.collect()
        if USING_GPU and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        self._tile_positions = [
            tuple(np.array(pos) + off * np.array(self._pixel_size))
            for pos, off in zip(self._tile_positions, self.global_offsets)
        ]

        # Fusion
        self._compute_fused_image_space()
        self._pad_to_chunk_multiple()
        omezarr = base / f"{self.root.stem}_fused_deskewed.ome.zarr"
        scale0 = omezarr / "scale0" / "image"
        self._create_fused_tensorstore(output_path=scale0)
        self._fuse_by_shard()

        # Write NGFF JSON for scale0
        Path(omezarr / "scale0").mkdir(parents=True, exist_ok=True)
        ngff = {
            "attributes": {"_ARRAY_DIMENSIONS": ["t", "c", "z", "y", "x"]},
            "zarr_format": 3,
            "consolidated_metadata": "null",
            "node_type": "group",
        }
        with open(omezarr / "scale0" / "zarr.json", "w") as f:
            json.dump(ngff, f, indent=2)

        # Build coarser scales
        self._create_multiscales(omezarr, factors=self.multiscale_factors)
        self._generate_ngff_zarr3_json(
            omezarr, resolution_multiples=self.resolution_multiples
        )

if __name__ == "__main__":
    fusion = TileFusion("/mnt/data2/qi2lab/20250513_human_OB/whole_OB_slice_polya.zarr")
    fusion.run()
