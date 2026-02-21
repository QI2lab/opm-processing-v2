"""
2D/3D tile fusion for qi2lab OPM data.

This module implements a class with GPU, Numba, and CuPy‐accelerated kernels
for tile registration and fusion of TPCZYX qi2lab-OPM stacks.

The final fused volume is written to a ome-ngff v0.5 datastore using tensorstore.
"""

import gc
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cupy as cp
import numpy as np
import SimpleITK as sitk
import tensorstore as ts
from cucim.skimage.exposure import match_histograms
from cucim.skimage.measure import block_reduce
from cucim.skimage.registration import phase_cross_correlation
from cupyx.scipy.ndimage import shift as cp_shift
from numba import njit, prange
from tqdm import trange

from opm_processing.imageprocessing.ssim_cuda import (
    structural_similarity_cupy_sep_shared as ssim_cuda,
)


@njit(parallel=True)
def _accumulate_tile_shard_1d(
    fused: np.ndarray,
    weight: np.ndarray,
    sub: np.ndarray,
    wz: np.ndarray,
    wy: np.ndarray,
    wx: np.ndarray,
    z_off: int,
    y_off: int,
    x_off: int,
) -> None:
    """
    Accumulate sub-volume into fused/weight using separable 1D profiles.

    fused, weight: float32[1, dz, pad_Y, pad_X]
    sub:          float32[1, sub_dz, Y, X]
    wz:           float32[sub_dz]
    wy:           float32[Y]
    wx:           float32[X]
    offsets are in fused buffer coordinates.
    """
    _, dz, Yp, Xp = fused.shape
    _, sub_dz, Y, X = sub.shape
    total = sub_dz * Y

    for idx in prange(total):
        dz_i = idx // Y
        y_i = idx % Y

        gz = z_off + dz_i
        gy = y_off + y_i

        base_f = fused[0, gz, gy]
        base_w = weight[0, gz, gy]
        sub_line = sub[0, dz_i, y_i]

        w_zy = wz[dz_i] * wy[y_i]
        for x_i in range(X):
            gx = x_off + x_i
            w_val = w_zy * wx[x_i]
            base_f[gx] += sub_line[x_i] * w_val
            base_w[gx] += w_val

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
            (1, 1, 1),
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 8),
            (16, 16, 16),
            (32, 32, 32),
        ),
        max_workers: int = 8,
        debug: bool = False,
        metrics_filename: str = "metrics.json",
        channel_to_use: int = 0,
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
        ds = ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": str(self.deskewed)},
            }
        ).result()

        self._tile_positions = [
            tuple(
                meta["attributes"]["per_index_metadata"][str(t)][str(p)]["0"][
                    "stage_position"
                ]
            )
            for t in range(ds.shape[0])
            for p in range(ds.shape[1])
        ]
        self._pixel_size = tuple(meta["attributes"]["deskewed_voxel_size_um"])

        self.downsample_factors = tuple(downsample_factors)
        self.ssim_window = int(ssim_window)
        self.threshold = float(threshold)
        self.multiscale_factors = tuple(multiscale_factors)
        self.resolution_multiples = [
            r if hasattr(r, "__len__") else (r, r, r) for r in resolution_multiples
        ]
        self._max_workers = int(max_workers)
        self._debug = bool(debug)
        self.metrics_filename = metrics_filename
        self._blend_pixels = tuple(blend_pixels)
        self.channel_to_use = channel_to_use

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

        (
            self.time_dim,
            self.position_dim,
            self.channels,
            self.z_dim,
            self.Y,
            self.X,
        ) = self.ts.shape

        self._update_profiles()
        self.chunk_shape = (1, 1, 1, 1024, 1024)
        self.chunk_y, self.chunk_x = self.chunk_shape[-2:]

        self.pairwise_metrics: Dict[Tuple[int, int], Tuple[int, int, int, float]] = {}
        self.global_offsets: Optional[np.ndarray] = None
        self.offset: Optional[Tuple[float, float, float]] = None
        self.unpadded_shape: Optional[Tuple[int, int, int]] = None
        self.padded_shape: Optional[Tuple[int, int, int]] = None
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
        prof = np.ones(length, dtype=np.float32)
        if blend > 0:
            ramp = np.linspace(0, 1, blend, endpoint=False, dtype=np.float32)
            prof[:blend] = ramp
            prof[-blend:] = ramp[::-1]
        return prof

    # --------------------------------------------------------------------- #
    #  Candidate-pair generation (new)
    # --------------------------------------------------------------------- #

    def _tile_centers_vox_for_time(self, t: int) -> np.ndarray:
        """
        Return tile centers in voxel units for a given timepoint.

        Notes
        -----
        This pipeline uses stage positions as tile centers (see `_compute_fused_image_space`).
        """
        n_pos = self.position_dim
        base = t * n_pos
        pos_phys = np.asarray(self._tile_positions[base: base + n_pos], dtype=np.float64)
        pix = np.asarray(self._pixel_size, dtype=np.float64)
        return np.rint(pos_phys / pix).astype(np.int64)  # (n_pos, 3) in Z/Y/X voxels

    def _build_candidate_pairs_hashgrid(
        self,
        centers_zyx_vox: np.ndarray,
        tile_shape_zyx: tuple[int, int, int],
        margin_zyx: tuple[int, int, int] = (0, 0, 0),
    ) -> list[tuple[int, int]]:
        """
        Build candidate overlapping tile pairs using a 3D hash grid, with a strict
        overlap-necessary condition using center distances.

        Parameters
        ----------
        centers_zyx_vox
            (n_tiles, 3) tile centers in voxel units (Z/Y/X).
        tile_shape_zyx
            (tz, ty, tx) tile size in voxels.
        margin_zyx
            Extra margin (mz, my, mx) in voxels to avoid dropping near-boundary pairs.

        Returns
        -------
        pairs
            List of (i, j) with i < j, candidates likely to overlap in Z/Y/X.
        """
        tz, ty, tx = tile_shape_zyx
        mz, my, mx = margin_zyx

        # Necessary condition for overlap with center-based extents:
        # |d_axis| < tile_axis + margin_axis
        # (matches your bounds logic, which uses full tile length as the overlap window)
        thr_z = max(1, tz + mz)
        thr_y = max(1, ty + my)
        thr_x = max(1, tx + mx)

        # Hash grid cell sizes set to the thresholds so any overlapping pair
        # must lie in same or adjacent cells.
        bz = thr_z
        by = thr_y
        bx = thr_x

        bins: dict[tuple[int, int, int], list[int]] = {}
        for idx, (cz, cy, cx) in enumerate(centers_zyx_vox):
            key = (int(cz // bz), int(cy // by), int(cx // bx))
            bins.setdefault(key, []).append(idx)

        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        for i, (cz, cy, cx) in enumerate(centers_zyx_vox):
            kz, ky, kx = int(cz // bz), int(cy // by), int(cx // bx)
            for dzb in (-1, 0, 1):
                for dyb in (-1, 0, 1):
                    for dxb in (-1, 0, 1):
                        neigh = bins.get((kz + dzb, ky + dyb, kx + dxb))
                        if not neigh:
                            continue
                        for j in neigh:
                            if j <= i:
                                continue
                            key = (i, j)
                            if key in seen:
                                continue

                            dz = int(centers_zyx_vox[j, 0] - cz)
                            dy = int(centers_zyx_vox[j, 1] - cy)
                            dx = int(centers_zyx_vox[j, 2] - cx)

                            if abs(dz) < thr_z and abs(dy) < thr_y and abs(dx) < thr_x:
                                pairs.append((i, j))
                                seen.add(key)

        return pairs


    # --------------------------------------------------------------------- #
    #  Registration functions (existing + modified refine method)
    # --------------------------------------------------------------------- #

    @staticmethod
    def register_with_sitk(
        fixed: np.ndarray,
        moving: np.ndarray,
        voxel_size: tuple[float, float, float],
        init_offset: tuple[float, float, float],
        debug: bool = False,
    ) -> tuple[tuple[float, float, float], np.ndarray]:
        """
        Register `moving` → `fixed` with SimpleITK Translation + Mattes MI,
        initializing from a provided physical offset, and correctly handling
        a CompositeTransform by extracting the optimized translation.

        Parameters
        ----------
        fixed : np.ndarray
            Fixed image block, shape (Z, Y, X).
        moving : np.ndarray
            Moving image block, shape (Z, Y, X).
        voxel_size : tuple[float, float, float]
            Voxel size in (z, y, x).
        init_offset : tuple[float, float, float]
            Initial translation (x, y, z) in physical units.
        debug : bool, optional
            If True, print iteration metrics, by default False.

        Returns
        -------
        shift_vox : tuple[float, float, float]
            The translation (dz, dy, dx) in full-resolution voxels.
        aligned : np.ndarray
            The moving image resampled onto the fixed image grid.
        """
        fixed_img = sitk.GetImageFromArray(fixed)
        moving_img = sitk.GetImageFromArray(moving)
        spacing = (voxel_size[2], voxel_size[1], voxel_size[0])
        fixed_img.SetSpacing(spacing)
        moving_img.SetSpacing(spacing)

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsANTSNeighborhoodCorrelation(5)
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-4, numberOfIterations=100
        )
        reg.SetInterpolator(sitk.sitkLinear)

        init_tx = sitk.TranslationTransform(3)
        init_tx.SetOffset(init_offset)
        reg.SetInitialTransform(init_tx, inPlace=False)

        if debug:
            reg.AddCommand(
                sitk.sitkIterationEvent,
                lambda: print(
                    f"[SITK] Iter {reg.GetOptimizerIteration():3d} "
                    f"Metric={reg.GetMetricValue():.4f}"
                ),
            )

        try:
            final_tx = reg.Execute(fixed_img, moving_img)
        except RuntimeError as e:
            msg = str(e)
            if "All samples map outside moving image buffer" in msg:
                if debug:
                    print("[SITK] Metric failed—using init_offset only.")
                final_tx = init_tx
            else:
                raise

        if isinstance(final_tx, sitk.CompositeTransform):
            idx = 1 if final_tx.GetNumberOfTransforms() > 1 else 0
            opt_tx = final_tx.GetNthTransform(idx)
            try:
                disp_phys = opt_tx.GetOffset()
            except AttributeError:
                disp_phys = tuple(opt_tx.GetParameters())
        else:
            disp_phys = final_tx.GetOffset()

        dz = disp_phys[2] / voxel_size[0]
        dy = disp_phys[1] / voxel_size[1]
        dx = disp_phys[0] / voxel_size[2]
        shift_vox = (dz, dy, dx)

        resampled = sitk.Resample(
            moving_img,
            fixed_img,
            final_tx,
            sitk.sitkLinear,
            0.0,
            moving_img.GetPixelID(),
        )
        aligned = sitk.GetArrayFromImage(resampled).astype(np.float32)

        return shift_vox, aligned

    @staticmethod
    def register_and_score(
        g1_full: cp.ndarray,
        g2_full: cp.ndarray,
        downsample_factors: tuple[int, int, int],
        win_size: int,
        debug: bool = True,
    ) -> Union[Tuple[Tuple[float, float, float], float], Tuple[None, None]]:
        """
        Downsample on GPU, histogram-match g2→g1, compute subpixel shift, and SSIM.

        Parameters
        ----------
        g1_full, g2_full : cp.ndarray
            Full-resolution fixed and moving patches (ZYX) on GPU.
        downsample_factors : tuple[int, int, int]
            Block-reduce factors (dz, dy, dx).
        win_size : int
            SSIM window.
        debug : bool
            If True, print intermediate info.

        Returns
        -------
        shift : (dz, dy, dx)
            Subpixel shift in *downsampled grid units* (ZYX).
            Caller should scale by downsample_factors to get full-res voxels.
        ssim_val : float
            SSIM score.
        """
        try:
            # Downsample on GPU (mean)
            g1 = block_reduce(g1_full, downsample_factors, cp.mean)
            g2 = block_reduce(g2_full, downsample_factors, cp.mean)

            # Match histograms (GPU)
            g2m = match_histograms(g2, g1)

            # Register z-max projections first (Y/X shift)
            g2_zmax = cp.max(g2m, axis=0)
            g1_zmax = cp.max(g1, axis=0)

            shift_zmax, _, _ = phase_cross_correlation(
                g1_zmax,
                g2_zmax,
                disambiguate=True,
                normalization="phase",
                upsample_factor=10,
                overlap_ratio=1.0,
            )
            if debug:
                print(f"shift_zmax: {shift_zmax}")

            shift_zmax_expanded = np.array([0, shift_zmax[0], shift_zmax[1]], dtype=np.float32)
            if debug:
                print(f"shift_zmax_expanded: {shift_zmax_expanded}")

            g2s_zmax = cp_shift(
                g2m,
                shift=shift_zmax_expanded,
                order=1,
                prefilter=False,
            )

            # Now register full 3D (downsampled)
            shift, _, _ = phase_cross_correlation(
                g1,
                g2s_zmax,
                disambiguate=True,
                normalization="phase",
                upsample_factor=10,
                overlap_ratio=1.0,
            )

            final_shift = np.array(shift_zmax_expanded + shift, dtype=np.float32)
            if debug:
                print(f"final_shift: {final_shift}")

            # Apply final shift and compute SSIM
            g2s = cp_shift(g2m, shift=final_shift, order=1, prefilter=False)
            ssim_val = ssim_cuda(g1, g2s, win_size=win_size)

            return tuple(float(s) for s in final_shift), float(ssim_val)
        except Exception:
            return None, None

    def _build_pruned_candidate_pairs_for_time(
        self,
        t: int,
        downsample_factors: tuple[int, int, int],
        max_geom_neighbors_per_tile: int = 3,
        min_overlap_vox: tuple[int, int, int] | None = None,
        min_overlap_ratio: tuple[float, float, float] = (0.02, 0.05, 0.05),
        margin_vox: tuple[int, int, int] | None = None,
    ) -> tuple[list[tuple[int, int, int, int, int]], dict[str, int]]:
        """
        Build a pruned list of candidate pairs for a given timepoint using geometry only.

        Returns
        -------
        pruned : list of (i_pos, j_pos, dz0, dy0, dx0)
            Indices are *within* the timepoint (0..position_dim-1). Offsets are in voxels (ZYX).
        stats : dict
            Diagnostics counters for debug printing.
        """
        n_pos = self.position_dim
        base = t * n_pos

        pos_phys = np.asarray(self._tile_positions[base: base + n_pos], dtype=np.float64)
        pix = np.asarray(self._pixel_size, dtype=np.float64)
        pos_vox = np.rint(pos_phys / pix).astype(np.int64)  # (n_pos, 3) Z/Y/X vox

        df = downsample_factors
        if margin_vox is None:
            margin_vox = (max(2, df[0]), max(2, df[1]), max(2, df[2]))

        if min_overlap_vox is None:
            # Conservative defaults; tighten/loosen as needed for your geometry.
            min_overlap_vox = (max(4 * df[0], 16), max(8 * df[1], 256), max(8 * df[2], 256))

        mz, my, mx = margin_vox
        tz, ty, tx = self.z_dim, self.Y, self.X

        # Hash-grid candidates (necessary condition for overlap)
        candidates = self._build_candidate_pairs_hashgrid(
            pos_vox,
            tile_shape_zyx=(tz, ty, tx),
            margin_zyx=(mz, my, mx),
        )
        raw_candidates = len(candidates)

        def overlap_len_from_offset(off: int, length: int) -> int:
            a = abs(off)
            return (length - a) if a < length else 0

        min_ov_z, min_ov_y, min_ov_x = min_overlap_vox
        min_rz, min_ry, min_rx = min_overlap_ratio

        geom_ok = 0
        geom_dropped = 0

        # Per-tile edge lists for kNN-style pruning
        per_tile: dict[int, list[tuple[tuple[float, float], int, int, int, int, int]]] = {}
        # value: list of (key, i_pos, j_pos, dz0, dy0, dx0)
        # key sorts by: (larger overlap volume first, then smaller normalized distance)

        for (i_pos, j_pos) in candidates:
            dz0 = int(pos_vox[j_pos, 0] - pos_vox[i_pos, 0])
            dy0 = int(pos_vox[j_pos, 1] - pos_vox[i_pos, 1])
            dx0 = int(pos_vox[j_pos, 2] - pos_vox[i_pos, 2])

            ov_z = overlap_len_from_offset(dz0, tz)
            ov_y = overlap_len_from_offset(dy0, ty)
            ov_x = overlap_len_from_offset(dx0, tx)

            if ov_z <= 0 or ov_y <= 0 or ov_x <= 0:
                geom_dropped += 1
                continue

            if (
                ov_z < min_ov_z
                or ov_y < min_ov_y
                or ov_x < min_ov_x
                or (ov_z / tz) < min_rz
                or (ov_y / ty) < min_ry
                or (ov_x / tx) < min_rx
            ):
                geom_dropped += 1
                continue

            geom_ok += 1

            ov_vol = float(ov_z) * float(ov_y) * float(ov_x)
            # normalized L1 distance
            dist = (
                abs(dz0) / max(1, tz)
                + abs(dy0) / max(1, ty)
                + abs(dx0) / max(1, tx)
            )

            # Sort key: larger overlap volume first, then smaller distance.
            # Use negative volume so "smaller key" is better.
            key = (-ov_vol, dist)

            per_tile.setdefault(i_pos, []).append((key, i_pos, j_pos, dz0, dy0, dx0))
            per_tile.setdefault(j_pos, []).append((key, i_pos, j_pos, dz0, dy0, dx0))

        keep_pairs: set[tuple[int, int]] = set()
        for edges in per_tile.values():
            edges.sort(key=lambda e: e[0])
            for e in edges[:max_geom_neighbors_per_tile]:
                _, ii, jj, *_ = e
                a, b = (ii, jj) if ii < jj else (jj, ii)
                keep_pairs.add((a, b))

        pruned: list[tuple[int, int, int, int, int]] = []
        for (i_pos, j_pos) in keep_pairs:
            dz0 = int(pos_vox[j_pos, 0] - pos_vox[i_pos, 0])
            dy0 = int(pos_vox[j_pos, 1] - pos_vox[i_pos, 1])
            dx0 = int(pos_vox[j_pos, 2] - pos_vox[i_pos, 2])
            pruned.append((i_pos, j_pos, dz0, dy0, dx0))

        stats = {
            "raw_candidates": raw_candidates,
            "geom_ok": geom_ok,
            "geom_dropped": geom_dropped,
            "pruned_candidates": len(pruned),
        }
        return pruned, stats


    def refine_tile_positions_with_cross_correlation(
        self,
        downsample_factors: tuple[int, int, int] = None,
        ssim_window: int = None,
        ch_idx: int = 0,
        threshold: float = None,
        use_sitk_refinement: bool = False,
    ) -> None:
        """
        Full-res overlap reads + GPU downsampling/registration via register_and_score.

        This version:
          - Builds a pruned candidate list per timepoint using geometry only (no I/O).
          - Reads only the overlap region at full resolution for each candidate pair.
          - Calls register_and_score (which performs GPU block_reduce mean downsampling).
          - Optionally performs SITK refinement after applying coarse shift to moving patch.

        Debug behavior:
          - Prints pair diagnostics (raw_candidates, geom_ok, pruned_candidates, etc.) per timepoint.
        """
        import time

        df = downsample_factors or self.downsample_factors
        sw = ssim_window or self.ssim_window
        th = threshold or self.threshold

        self.pairwise_metrics.clear()
        n_pos = self.position_dim

        # Controls pair count (tune these if you still see too many pairs)
        max_geom_neighbors_per_tile = 3

        executor = ThreadPoolExecutor(max_workers=self._max_workers)

        def bounds_1d(off: int, length: int) -> tuple[int, int]:
            lo = 0 if off < 0 else off
            hi = length if off > 0 else length + off
            lo = max(0, lo)
            hi = min(length, hi)
            return lo, hi

        def read_patch_fullres(idx: int, bnds) -> np.ndarray:
            z0, z1 = bnds[0]
            y0, y1 = bnds[1]
            x0, x1 = bnds[2]
            return (
                self.ts[0, idx, ch_idx, z0:z1, y0:y1, x0:x1]
                .read()
                .result()
                .astype(np.float32)
            )

        for t in range(self.time_dim):
            base = t * n_pos
            t0 = time.perf_counter()

            pruned, stats = self._build_pruned_candidate_pairs_for_time(
                t=t,
                downsample_factors=df,
                max_geom_neighbors_per_tile=max_geom_neighbors_per_tile,
            )

            bounds_ok = 0
            coarse_ok = 0
            refined = 0
            kept = 0

            t_read = 0.0
            t_reg = 0.0
            t_sitk = 0.0

            for k in trange(len(pruned), desc="register", leave=True):
                i_pos, j_pos, dz0, dy0, dx0 = pruned[k]
                i = base + i_pos
                j = base + j_pos

                b_i = [
                    bounds_1d(dz0, self.z_dim),
                    bounds_1d(dy0, self.Y),
                    bounds_1d(dx0, self.X),
                ]
                b_j = [
                    bounds_1d(-dz0, self.z_dim),
                    bounds_1d(-dy0, self.Y),
                    bounds_1d(-dx0, self.X),
                ]
                if any(hi <= lo for lo, hi in b_i):
                    continue
                bounds_ok += 1

                # Full-res read of overlap region (CPU)
                tr0 = time.perf_counter()
                f_i = executor.submit(read_patch_fullres, i, b_i)
                f_j = executor.submit(read_patch_fullres, j, b_j)
                patch_i = f_i.result()
                patch_j = f_j.result()
                tr1 = time.perf_counter()
                t_read += (tr1 - tr0)

                # GPU coarse registration (downsampling happens inside register_and_score)
                tg0 = time.perf_counter()
                g1_full = cp.asarray(patch_i)
                g2_full = cp.asarray(patch_j)

                shift_ds, ssim_val = self.register_and_score(
                    g1_full,
                    g2_full,
                    downsample_factors=df,
                    win_size=sw,
                    debug=self._debug,
                )
                tg1 = time.perf_counter()
                t_reg += (tg1 - tg0)

                if shift_ds is None or (ssim_val < th and th != 0.0):
                    continue
                coarse_ok += 1

                coarse_full = [shift_ds[a] * df[a] for a in range(3)]

                if use_sitk_refinement:
                    refined += 1
                    ts0 = time.perf_counter()

                    pj_shifted_gpu = cp_shift(
                        g2_full,
                        shift=tuple(coarse_full),
                        order=1,
                        prefilter=False,
                    )
                    patch_j_shifted = cp.asnumpy(pj_shifted_gpu)

                    sitk_shift, _ = self.register_with_sitk(
                        patch_i,
                        patch_j_shifted,
                        voxel_size=self._pixel_size,
                        init_offset=(0.0, 0.0, 0.0),
                        debug=self._debug,
                    )
                    total_shift = [coarse_full[a] + sitk_shift[a] for a in range(3)]

                    ts1 = time.perf_counter()
                    t_sitk += (ts1 - ts0)
                else:
                    total_shift = coarse_full

                dz_s, dy_s, dx_s = (
                    int(round(total_shift[0])),
                    int(round(total_shift[1])),
                    int(round(total_shift[2])),
                )

                max_shift = (50, 200, 200)
                if (
                    abs(dz_s) > max_shift[0]
                    or abs(dy_s) > max_shift[1]
                    or abs(dx_s) > max_shift[2]
                ):
                    if self._debug:
                        print(
                            f"Dropping link {(i, j)} shift={(dz_s, dy_s, dx_s)} "
                            f"exceeds {max_shift}"
                        )
                    continue

                self.pairwise_metrics[(i, j)] = (dz_s, dy_s, dx_s, round(ssim_val, 3))
                kept += 1

            t1 = time.perf_counter()

            if self._debug:
                print(
                    "[register diagnostics] "
                    f"t={t} raw_candidates={stats['raw_candidates']} geom_ok={stats['geom_ok']} "
                    f"geom_dropped={stats['geom_dropped']} pruned_candidates={stats['pruned_candidates']} "
                    f"bounds_ok={bounds_ok} coarse_ok={coarse_ok} refined={refined} kept={kept} "
                    f"t_read={t_read:.3f}s t_reg={t_reg:.3f}s t_sitk={t_sitk:.3f}s "
                    f"t_total={(t1 - t0):.3f}s"
                )

        executor.shutdown(wait=True)


    @staticmethod
    def _solve_global(
        links: List[Dict[str, Any]], n_tiles: int, fixed_indices: List[int]
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
                i, j = link["i"], link["j"]
                t, w = link["t"][axis], link["w"]
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
        iterative: bool,
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
            return np.array([np.linalg.norm(sh[l["j"]] - sh[l["i"]] - l["t"]) for l in ls])

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

    def _prune_pairwise_metrics_xy_z(
        self,
        max_xy_links_per_tile: int = 4,
        max_z_links_per_tile: int = 2,
        require_xy_link: bool = True,
    ) -> None:
        """
        Prune self.pairwise_metrics so that each tile has:
          - <= max_xy_links_per_tile links classified as XY-neighbors
          - <= max_z_links_per_tile links classified as Z-neighbors

        Also enforces that every tile has at least one XY link (unless require_xy_link=False).

        Classification is based on the *expected* center-to-center separation (from stage positions)
        normalized by tile dimensions:
          - Z-link if |dz|/z_dim is the dominant normalized separation and |dz| > 0
          - otherwise XY-link
        """
        if not self.pairwise_metrics:
            raise ValueError("pairwise_metrics is empty; cannot optimize shifts without links.")

        n_tiles = len(self._tile_positions)

        # Tile centers in voxels (ZYX), consistent with other parts of this module
        pos_phys = np.asarray(self._tile_positions, dtype=np.float64)
        pix = np.asarray(self._pixel_size, dtype=np.float64)
        centers_vox = np.rint(pos_phys / pix).astype(np.int64)  # (n_tiles, 3)

        tz, ty, tx = int(self.z_dim), int(self.Y), int(self.X)

        def classify_link(i: int, j: int) -> str:
            dv = centers_vox[j] - centers_vox[i]  # Z/Y/X
            nz = abs(dv[0]) / max(1, tz)
            ny = abs(dv[1]) / max(1, ty)
            nx = abs(dv[2]) / max(1, tx)

            # Z-neighbor only if Z separation is dominant and nonzero
            if abs(dv[0]) > 0 and nz >= ny and nz >= nx:
                return "z"
            return "xy"

        # Build sortable edge list (greedy selection by SSIM)
        edges: list[tuple[float, int, int, int, int, int, str]] = []
        for (i, j), v in self.pairwise_metrics.items():
            dz_s, dy_s, dx_s, ssim_val = int(v[0]), int(v[1]), int(v[2]), float(v[3])
            cls = classify_link(i, j)
            edges.append((ssim_val, i, j, dz_s, dy_s, dx_s, cls))

        # Sort by SSIM descending, keep best links first
        edges.sort(key=lambda e: e[0], reverse=True)

        xy_deg = np.zeros(n_tiles, dtype=np.int32)
        z_deg = np.zeros(n_tiles, dtype=np.int32)

        pruned: dict[tuple[int, int], tuple[int, int, int, float]] = {}

        for ssim_val, i, j, dz_s, dy_s, dx_s, cls in edges:
            if cls == "xy":
                if xy_deg[i] >= max_xy_links_per_tile or xy_deg[j] >= max_xy_links_per_tile:
                    continue
                pruned[(i, j)] = (dz_s, dy_s, dx_s, round(ssim_val, 3))
                xy_deg[i] += 1
                xy_deg[j] += 1
            else:  # cls == "z"
                if z_deg[i] >= max_z_links_per_tile or z_deg[j] >= max_z_links_per_tile:
                    continue
                pruned[(i, j)] = (dz_s, dy_s, dx_s, round(ssim_val, 3))
                z_deg[i] += 1
                z_deg[j] += 1

        # Validate: every tile must have >=1 XY link (as requested)
        if require_xy_link:
            bad = np.where(xy_deg == 0)[0].tolist()
            if bad:
                # Provide helpful context for debugging
                bad_pos = [tuple(map(float, self._tile_positions[k])) for k in bad[:20]]
                more = "" if len(bad) <= 20 else f" (showing first 20 of {len(bad)})"
                raise ValueError(
                    "Global optimization aborted: one or more tiles have zero XY links after pruning. "
                    f"Tiles: {bad}{more}. "
                    f"Example stage positions (z,y,x) for failing tiles: {bad_pos}{more}. "
                    "This indicates missing overlaps or overly strict link acceptance upstream."
                )

        if self._debug:
            print(
                "[link pruning] "
                f"edges_in={len(edges)} edges_kept={len(pruned)} "
                f"max_xy={max_xy_links_per_tile} max_z={max_z_links_per_tile} "
                f"xy_deg[min/med/max]={int(xy_deg.min())}/{int(np.median(xy_deg))}/{int(xy_deg.max())} "
                f"z_deg[min/med/max]={int(z_deg.min())}/{int(np.median(z_deg))}/{int(z_deg.max())}"
            )

        # Apply pruning in-place
        self.pairwise_metrics = pruned


    def optimize_shifts(
        self,
        method: str = "ONE_ROUND",
        rel_thresh: float = 0.3,
        abs_thresh: float = 5.0,
        iterative: bool = False,
    ) -> None:
        """
        Globally optimize tile shifts using either:
          - ONE_ROUND
          - TWO_ROUND_SIMPLE
          - TWO_ROUND_ITERATIVE

        Additional constraint (enforced before solving):
          - Per tile: <= 4 XY links and <= 2 Z links
          - Allow 0 Z links and as few as 2 XY links
          - Require at least 1 XY link per tile; otherwise raise an error
        """
        # Enforce final graph constraints before building the system
        self._prune_pairwise_metrics_xy_z(
            max_xy_links_per_tile=4,
            max_z_links_per_tile=2,
            require_xy_link=True,
        )

        links: List[Dict[str, Any]] = []
        for (i, j), v in self.pairwise_metrics.items():
            links.append(
                {
                    "i": i,
                    "j": j,
                    "t": np.array(v[:3], dtype=np.float64),
                    "w": np.sqrt(float(v[3])),
                }
            )

        n = len(self._tile_positions)
        fixed = [n - 1]  # keep your existing behavior

        if method == "ONE_ROUND":
            d_opt = self._solve_global(links, n, fixed)
        elif method.startswith("TWO_ROUND"):
            d_opt = self._two_round_opt(
                links,
                n,
                fixed,
                rel_thresh,
                abs_thresh,
                method.endswith("ITERATIVE"),
            )
        else:
            raise ValueError(f"Unknown method {method}")

        self.global_offsets = d_opt


    def save_pairwise_metrics(self, filepath: Union[str, Path]) -> None:
        """
        Save pairwise_metrics to a JSON file.
        """
        path = Path(filepath)
        out = {f"{i},{j}": list(v) for (i, j), v in self.pairwise_metrics.items()}
        with open(path, "w") as f:
            json.dump(out, f)

    def load_pairwise_metrics(self, filepath: Union[str, Path]) -> None:
        """
        Load pairwise_metrics from a JSON file.
        """
        path = Path(filepath)
        with open(path, "r") as f:
            data = json.load(f)
        self.pairwise_metrics = {tuple(map(int, k.split(","))): tuple(v) for k, v in data.items()}

    def _compute_fused_image_space(self) -> None:
        """
        Compute fused volume extents assuming stage positions correspond to the *tile origin*
        (physical location of voxel (0,0,0) in each tile) in (z, y, x).

        This avoids under-sizing the fused bounding box that occurs if positions are treated
        as tile centers.
        """
        pos = np.asarray(self._tile_positions, dtype=np.float64)  # (n_tiles, 3) in physical units (z,y,x)

        tile_ext = np.array(
            [
                self.z_dim * self._pixel_size[0],
                self.Y * self._pixel_size[1],
                self.X * self._pixel_size[2],
            ],
            dtype=np.float64,
        )

        min_zyx = pos.min(axis=0)
        max_zyx = (pos + tile_ext).max(axis=0)

        sz = int(np.ceil((max_zyx[0] - min_zyx[0]) / self._pixel_size[0]))
        sy = int(np.ceil((max_zyx[1] - min_zyx[1]) / self._pixel_size[1]))
        sx = int(np.ceil((max_zyx[2] - min_zyx[2]) / self._pixel_size[2]))

        self.unpadded_shape = (sz, sy, sx)
        self.offset = (float(min_zyx[0]), float(min_zyx[1]), float(min_zyx[2]))

        # Used later for NGFF translations; define as bbox center in physical units
        self.center = (
            float((max_zyx[2] - min_zyx[2]) / 2.0),
            float((max_zyx[1] - min_zyx[1]) / 2.0),
            float((max_zyx[0] - min_zyx[0]) / 2.0),
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
        self, output_path: Union[str, Path], z_slices_per_shard: int = 4
    ) -> None:
        """
        Create the output Zarr v3 store for the fused volume.
        """
        out = Path(output_path)
        full_shape = [1, self.channels, *self.padded_shape]
        shard_chunk = [1, 1, z_slices_per_shard, self.chunk_y * 2, self.chunk_x * 2]
        codec_chunk = [1, 1, 1, self.chunk_y, self.chunk_x]
        self.shard_chunk = shard_chunk

        config = {
            "context": {
                "file_io_concurrency": {"limit": self.max_workers},
                "data_copy_concurrency": {"limit": self.max_workers},
                # "file_io_memmap": True,
                # "file_io_sync": False,
            },
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(out)},
            "metadata": {
                "shape": full_shape,
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shard_chunk}},
                "chunk_key_encoding": {"name": "default"},
                "codecs": [
                    {
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": codec_chunk,
                            "codecs": [
                                {"name": "bytes", "configuration": {"endian": "little"}},
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
                                {"name": "bytes", "configuration": {"endian": "little"}},
                                {"name": "crc32c"},
                            ],
                            "index_location": "end",
                        },
                    }
                ],
                "data_type": "uint16",
                "dimension_names": ["t", "c", "z", "y", "x"],
            },
        }

        self.fused_ts = ts.open(config, create=True, open=True).result()

    def _find_overlaps(
        self, offsets: List[Tuple[int, int, int]]
    ) -> List[Tuple[int, int, Tuple[int, int, int, int, int, int]]]:
        """
        Identify overlapping regions between all tile pairs.
        """
        overlaps: List[Tuple[int, int, Tuple[int, int, int, int, int, int]]] = []
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
        offsets: List[Tuple[int, int, int]],
    ) -> None:
        """
        Feather-blend one overlapping region between tiles i and j.
        """
        z0, z1, y0, y1, x0, x1 = region
        oz_i, oy_i, ox_i = offsets[i]
        oz_j, oy_j, ox_j = offsets[j]

        sub_i = (
            self.ts[
                0,
                i,
                slice(None),
                slice(z0 - oz_i, z1 - oz_i),
                slice(y0 - oy_i, y1 - oy_i),
                slice(x0 - ox_i, x1 - ox_i),
            ]
            .read()
            .result()
            .astype(np.float32)
        )
        sub_j = (
            self.ts[
                0,
                j,
                slice(None),
                slice(z0 - oz_j, z1 - oz_j),
                slice(y0 - oy_j, y1 - oy_j),
                slice(x0 - ox_j, x1 - ox_j),
            ]
            .read()
            .result()
            .astype(np.float32)
        )

        C, dz, dy, dx = sub_i.shape
        fused = np.empty((C, dz, dy, dx), dtype=np.float32)

        zi_i = slice(z0 - oz_i, z1 - oz_i)
        yi_i = slice(y0 - oy_i, y1 - oy_i)
        xi_i = slice(x0 - ox_i, x1 - ox_i)
        zi_j = slice(z0 - oz_j, z1 - oz_j)
        yi_j = slice(y0 - oy_j, y1 - oy_j)
        xi_j = slice(x0 - ox_j, x1 - ox_j)

        wz_i, wy_i, wx_i = (self.z_profile[zi_i], self.y_profile[yi_i], self.x_profile[xi_i])
        wz_j, wy_j, wx_j = (self.z_profile[zi_j], self.y_profile[yi_j], self.x_profile[xi_j])

        for c in range(C):
            buf = np.empty((dz, dy, dx), dtype=np.float32)
            fused[c] = _blend_numba(
                sub_i[c],
                sub_j[c],
                wz_i,
                wy_i,
                wx_i,
                wz_j,
                wy_j,
                wx_j,
                buf,
            )

        self.fused_ts[
            0,
            slice(None),
            slice(z0, z1),
            slice(y0, y1),
            slice(x0, x1),
        ].write(fused.astype(np.uint16)).result()

    def _copy_nonoverlap(
        self,
        idx: int,
        offsets: List[Tuple[int, int, int]],
        overlaps: List[Tuple[int, int, Tuple[int, int, int, int, int, int]]],
    ) -> None:
        """
        Copy non-overlapping slabs of tile `idx` directly to fused store.
        """
        oz, oy, ox = offsets[idx]
        tz, ty, tx = self.z_dim, self.Y, self.X
        regions = [(oz, oz + tz, oy, oy + ty, ox, ox + tx)]

        for (i, j, (z0, z1, y0, y1, x0, x1)) in overlaps:
            if idx not in (i, j):
                continue
            new_regs = []
            for (rz0, rz1, ry0, ry1, rx0, rx1) in regions:
                if (x1 <= rx0 or x0 >= rx1 or y1 <= ry0 or y0 >= ry1 or z1 <= rz0 or z0 >= rz1):
                    new_regs.append((rz0, rz1, ry0, ry1, rx0, rx1))
                else:
                    if z0 > rz0:
                        new_regs.append((rz0, z0, ry0, ry1, rx0, rx1))
                    if z1 < rz1:
                        new_regs.append((z1, rz1, ry0, ry1, rx0, rx1))
                    if y0 > ry0:
                        new_regs.append((max(rz0, z0), min(rz1, z1), ry0, y0, rx0, rx1))
                    if y1 < ry1:
                        new_regs.append((max(rz0, z0), min(rz1, z1), y1, ry1, rx0, rx1))
                    if x0 > rx0:
                        new_regs.append(
                            (max(rz0, z0), min(rz1, z1), max(ry0, y0), min(ry1, y1), rx0, x0)
                        )
                    if x1 < rx1:
                        new_regs.append(
                            (max(rz0, z0), min(rz1, z1), max(ry0, y0), min(ry1, y1), x1, rx1)
                        )
            regions = new_regs

        for (z0, z1, y0, y1, x0, x1) in regions:
            if z1 <= z0 or y1 <= y0 or x1 <= x0:
                continue
            block = (
                self.ts[
                    0,
                    idx,
                    slice(None),
                    slice(z0 - oz, z1 - oz),
                    slice(y0 - oy, y1 - oy),
                    slice(x0 - ox, x1 - ox),
                ]
                .read()
                .result()
                .astype(np.uint16)
            )
            self.fused_ts[
                0,
                slice(None),
                slice(z0, z1),
                slice(y0, y1),
                slice(x0, x1),
            ].write(block).result()

    def _fuse_by_shard(self) -> None:
        """
        Hybrid shard fusion to reduce peak memory without killing throughput.

        Keeps big accum buffers per (z_shard, channel) for throughput, but:
        - avoids allocating dense w3d weights (major peak reduction),
        - limits outstanding writes (prevents buffer retention spikes).

        Tuning knobs:
        - max_pending_writes: 0 = lowest memory, less overlap;
                            1–2 = overlap write/compute with bounded memory.
        """
        offsets = [
            (
                int((z - self.offset[0]) / self._pixel_size[0]),
                int((y - self.offset[1]) / self._pixel_size[1]),
                int((x - self.offset[2]) / self._pixel_size[2]),
            )
            for (z, y, x) in self._tile_positions
        ]

        if self._debug:
            for t_idx, (oz, oy, ox) in enumerate(offsets):
                if oy + self.Y > self.padded_shape[1] or ox + self.X > self.padded_shape[2] or oz + self.z_dim > self.padded_shape[0]:
                    raise ValueError(
                        "Tile does not fit in fused volume. "
                        f"t_idx={t_idx} off_zyx={(oz,oy,ox)} tile_zyx={(self.z_dim,self.Y,self.X)} "
                        f"padded_shape={self.padded_shape} offset_phys={self.offset} tile_pos_phys={self._tile_positions[t_idx]}"
                    )

        z_step = int(self.shard_chunk[2])
        pad_Y, pad_X = int(self.padded_shape[1]), int(self.padded_shape[2])
        nz = (int(self.padded_shape[0]) + z_step - 1) // z_step

        # Bounded write queue to cap peak RAM held by in-flight writes.
        # 0 = immediate .result() (lowest peak), 1 = overlap one write with next compute (often good).
        max_pending_writes = 1
        pending = deque()

        # Profiles reused across all tiles
        wy = self.y_profile.astype(np.float32)
        wx = self.x_profile.astype(np.float32)

        if self._debug:
            bytes_per_buf = 1 * z_step * pad_Y * pad_X * 4
            print(
                "[fusion] hybrid mode: "
                f"z_step={z_step} pad_Y={pad_Y} pad_X={pad_X} "
                f"accum_buffers≈{2 * bytes_per_buf / 1e9:.2f} GB per channel/shard "
                f"max_pending_writes={max_pending_writes} (adds up to that many extra buffers in-flight)"
            )

        for shard_idx in trange(nz, desc="scale0", leave=True):
            z0 = shard_idx * z_step
            z1 = min(z0 + z_step, int(self.padded_shape[0]))
            dz = z1 - z0

            for c in range(self.channels):
                fused_block = np.zeros((1, dz, pad_Y, pad_X), dtype=np.float32)
                weight_sum = np.zeros_like(fused_block)

                # Accumulate every tile into this (z_shard, channel)
                for t_idx, (oz, oy, ox) in enumerate(offsets):
                    tz0 = max(z0, oz)
                    tz1 = min(z1, oz + self.z_dim)
                    if tz1 <= tz0:
                        continue

                    local_z0 = tz0 - oz
                    local_z1 = tz1 - oz

                    sub = (
                        self.ts[
                            0,
                            t_idx,
                            slice(c, c + 1),
                            slice(local_z0, local_z1),
                            slice(0, self.Y),
                            slice(0, self.X),
                        ]
                        .read()
                        .result()
                        .astype(np.float32)
                    )

                    wz = self.z_profile[local_z0:local_z1].astype(np.float32)
                    z_off = int(tz0 - z0)

                    if self._debug:
                        assert fused_block.dtype == np.float32
                        assert weight_sum.dtype == np.float32
                        assert sub.dtype == np.float32
                        assert fused_block.shape[1] >= (z_off + sub.shape[1])
                        assert fused_block.shape[2] >= (oy + sub.shape[2]), (
                            "Y overflow: "
                            f"pad_Y={fused_block.shape[2]} oy={oy} sub_Y={sub.shape[2]} "
                            f"(oy+sub_Y)={oy + sub.shape[2]} "
                            f"offset_y_phys={self.offset[1]} pix_y={self._pixel_size[1]} "
                            f"tile_y_phys={self._tile_positions[t_idx][1]}"
                        )
                        assert fused_block.shape[3] >= (ox + sub.shape[3])
                        assert z_off >= 0 and oy >= 0 and ox >= 0
                    
                    _accumulate_tile_shard_1d(
                        fused_block,
                        weight_sum,
                        sub,
                        wz,
                        wy,
                        wx,
                        z_off,
                        int(oy),
                        int(ox),
                    )

                _normalize_shard(fused_block, weight_sum)

                # Schedule write, but bound the number of in-flight writes.
                fut = self.fused_ts[
                    0,
                    slice(c, c + 1),
                    slice(z0, z1),
                    slice(0, pad_Y),
                    slice(0, pad_X),
                ].write(fused_block.astype(np.uint16))

                pending.append((fut, fused_block, weight_sum))

                # If too many in-flight writes, wait for the oldest and release its buffers.
                while len(pending) > max_pending_writes:
                    old_fut, old_f, old_w = pending.popleft()
                    old_fut.result()
                    del old_f, old_w
                    gc.collect()

                # Important: do NOT free CuPy pools here; fusion is CPU/Numba-heavy and
                # frequent pool flushing hurts throughput. Only flush at coarse granularity.
                del fused_block, weight_sum

            # End of shard: drain remaining writes for this shard
            while pending:
                old_fut, old_f, old_w = pending.popleft()
                old_fut.result()
                del old_f, old_w
            gc.collect()

            # Optional: occasional GPU pool trimming (rare)
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()


    def _create_multiscales(
        self,
        omezarr_path: Path,
        factors: Sequence[int] = (2, 4, 8),
        z_slices_per_shard: int = 4,
    ) -> None:
        """
        Build NGFF multiscales by downsampling Z/Y/X iteratively.
        """
        inp = None
        for idx, factor in enumerate(factors):
            out_path = omezarr_path / f"scale{idx + 1}" / "image"
            if inp is not None:
                del inp
            prev = omezarr_path / f"scale{idx}" / "image"
            inp = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(prev)}}).result()

            factor_to_use = (factors[idx] // factors[idx - 1] if idx > 0 else factors[0])
            _, _, Z, Y, X = inp.shape
            new_z, new_y, new_x = Z // factor_to_use, Y // factor_to_use, X // factor_to_use
            shard_z = min(z_slices_per_shard, new_z)

            chunk_y = 1024 if new_y >= 2048 else new_y // 4 if new_y >= 4 else 1
            chunk_x = 1024 if new_x >= 2048 else new_x // 4 if new_x >= 4 else 1

            self.padded_shape = (new_z, new_y, new_x)
            self.chunk_y, self.chunk_x = chunk_y, chunk_x

            self._create_fused_tensorstore(output_path=out_path, z_slices_per_shard=shard_z)

            for z0 in trange(0, new_z, shard_z, desc=f"scale{idx + 1}", leave=True):
                bz = min(shard_z, new_z - z0)
                in_z0, in_z1 = z0 * factor_to_use, (z0 + bz) * factor_to_use
                for y0 in range(0, new_y, chunk_y):
                    by = min(chunk_y, new_y - y0)
                    in_y0, in_y1 = y0 * factor_to_use, (y0 + by) * factor_to_use
                    for x0 in range(0, new_x, chunk_x):
                        bx = min(chunk_x, new_x - x0)
                        in_x0, in_x1 = x0 * factor_to_use, (x0 + bx) * factor_to_use

                        slab = inp[:, :, in_z0:in_z1, in_y0:in_y1, in_x0:in_x1].read().result()
                        down = slab[..., ::factor_to_use, ::factor_to_use, ::factor_to_use]
                        self.fused_ts[:, :, z0 : z0 + bz, y0 : y0 + by, x0 : x0 + bx].write(down).result()

            ngff = {
                "attributes": {"_ARRAY_DIMENSIONS": ["t", "c", "z", "y", "x"]},
                "zarr_format": 3,
                "consolidated_metadata": "null",
                "node_type": "group",
            }
            with open(omezarr_path / f"scale{idx + 1}" / "zarr.json", "w") as f:
                json.dump(ngff, f, indent=2)

    def _generate_ngff_zarr3_json(
        self,
        omezarr_path: Path,
        resolution_multiples: Sequence[Union[int, Sequence[int]]],
        dataset_name: str = "image",
        version: str = "0.5",
        translation_mode: str = "origin",
    ) -> None:
        """
        Write OME-NGFF v0.5 multiscales JSON for Zarr3.

        Parameters
        ----------
        omezarr_path : Path
            Root path of the NGFF group.
        resolution_multiples : sequence
            Spatial resolution factors per scale (e.g. (1,1,1), (2,2,2), (4,4,4), ...).
        dataset_name : str
            Name of the dataset node.
        version : str
            NGFF version.
        translation_mode : {'origin', 'centered'}
            - 'origin'  : Keep translation identical for all scales (voxel (0,0,0) maps to the same
                        physical origin at every scale). This is correct for decimation
                        downsampling (your current `::factor_to_use`).
            - 'centered': Shift translation by half of the previous scale voxel size per relative
                        downsample step. This is appropriate for block-averaging semantics where
                        the coarse voxel represents the center of a block of finer voxels.
        """
        if translation_mode not in ("origin", "centered"):
            raise ValueError("translation_mode must be 'origin' or 'centered'.")

        axes = [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]

        norm_res = [
            tuple(r) if hasattr(r, "__len__") else (r, r, r)
            for r in resolution_multiples
        ]

        # Base voxel size in microns for (z,y,x)
        base_spatial = [float(s) for s in self._pixel_size]  # (z,y,x)

        # Physical origin of fused dataset at voxel (0,0,0), in microns (z,y,x)
        base_translation_spatial = [float(self.offset[0]), float(self.offset[1]), float(self.offset[2])]

        datasets: list[dict[str, Any]] = []

        # Track prior scale spatial size for optional 'centered' shifts
        prev_spatial = base_spatial
        prev_factors = (1.0, 1.0, 1.0)
        translation_spatial = base_translation_spatial.copy()

        for lvl, factors in enumerate(norm_res):
            fz, fy, fx = (float(factors[0]), float(factors[1]), float(factors[2]))
            spatial = [base_spatial[0] * fz, base_spatial[1] * fy, base_spatial[2] * fx]
            scale = [1.0, 1.0] + spatial

            if lvl == 0:
                translation_spatial = base_translation_spatial.copy()
            else:
                if translation_mode == "origin":
                    translation_spatial = base_translation_spatial.copy()
                else:
                    # 'centered': shift by half a previous-voxel per relative step, per axis.
                    # relative factor is (current_factor / previous_factor) per axis
                    rel = []
                    for a, (cur, prev) in enumerate(zip((fz, fy, fx), prev_factors)):
                        r = cur / prev if prev > 0 else 1.0
                        # Robustly snap to nearest integer when close
                        r_int = float(int(round(r)))
                        rel.append(r_int if abs(r - r_int) < 1e-6 else 1.0)

                    translation_spatial = base_translation_spatial.copy()
                    # Apply cumulative centered offset across levels:
                    # for each level, add 0.5*(rel-1)*prev_spatial to the previous translation.
                    # This matches block-center conventions for block-averaging.
                    # We reconstruct cumulative translation by iterating levels, so keep a running value.
                    # First, rebuild running translation up to this level:
                    translation_spatial = base_translation_spatial.copy()
                    running_prev_spatial = base_spatial
                    running_prev_factors = (1.0, 1.0, 1.0)
                    for k in range(1, lvl + 1):
                        kf = norm_res[k]
                        pkf = norm_res[k - 1]
                        rel_k = []
                        for cur_k, prev_k in zip(kf, pkf):
                            r = float(cur_k) / float(prev_k) if float(prev_k) > 0 else 1.0
                            r_int = float(int(round(r)))
                            rel_k.append(r_int if abs(r - r_int) < 1e-6 else 1.0)

                        translation_spatial[0] += 0.5 * (rel_k[0] - 1.0) * running_prev_spatial[0]
                        translation_spatial[1] += 0.5 * (rel_k[1] - 1.0) * running_prev_spatial[1]
                        translation_spatial[2] += 0.5 * (rel_k[2] - 1.0) * running_prev_spatial[2]

                        running_prev_spatial = [
                            base_spatial[0] * float(kf[0]),
                            base_spatial[1] * float(kf[1]),
                            base_spatial[2] * float(kf[2]),
                        ]
                        running_prev_factors = (float(kf[0]), float(kf[1]), float(kf[2]))

            datasets.append(
                {
                    "path": f"scale{lvl}/{dataset_name}",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": scale},
                        {
                            "type": "translation",
                            "translation": [0.0, 0.0] + [
                                float(translation_spatial[0]),
                                float(translation_spatial[1]),
                                float(translation_spatial[2]),
                            ],
                        },
                    ],
                }
            )

            prev_spatial = spatial
            prev_factors = (fz, fy, fx)

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
                downsample_factors=(3, 5, 5),
                ch_idx=self.channel_to_use,
                threshold=0.6,
                use_sitk_refinement=False,
            )
            self.save_pairwise_metrics(metrics_path)

        self.optimize_shifts(method="TWO_ROUND_ITERATIVE", rel_thresh=1.5, abs_thresh=3.5, iterative=True)
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        self._tile_positions = [
            tuple(np.array(pos) + off * np.array(self._pixel_size))
            for pos, off in zip(self._tile_positions, self.global_offsets)
        ]

        self._compute_fused_image_space()
        self._pad_to_chunk_multiple()
        omezarr = base / f"{self.root.stem}_fused_deskewed.ome.zarr"
        scale0 = omezarr / "scale0" / "image"
        self._create_fused_tensorstore(output_path=scale0)
        self._fuse_by_shard()

        ngff = {
            "attributes": {"_ARRAY_DIMENSIONS": ["t", "c", "z", "y", "x"]},
            "zarr_format": 3,
            "consolidated_metadata": "null",
            "node_type": "group",
        }
        with open(omezarr / "scale0" / "zarr.json", "w") as f:
            json.dump(ngff, f, indent=2)

        self._create_multiscales(omezarr, factors=self.multiscale_factors)
        self._generate_ngff_zarr3_json(omezarr, resolution_multiples=self.resolution_multiples)


if __name__ == "__main__":
    fusion = TileFusion("/mnt/data2/qi2lab/20250513_human_OB/whole_OB_slice_polya.zarr")
    fusion.run()
