"""
2D/3D tile fusion for qi2lab OPM data.

This module implements GPU-optional (CuPy/cuCIM) and CPU (NumPy/SciPy/skimage)
registration plus Numba-accelerated feather-weighted fusion of TPCZYX stacks.

Pipeline summary
----------------
1) For each timepoint independently, register overlapping tile pairs using
   phase cross-correlation + SSIM scoring.
2) For each timepoint independently, solve a robust global least-squares system
   (two-round iterative outlier rejection) with tile 0 anchored to zero offset.
3) Build a *global* fused coordinate space spanning all timepoints.
4) Fuse each timepoint into the shared global space using weighted accumulation.
5) Build NGFF multiscales and write OME-NGFF v0.5 JSON for Zarr v3 stores.

Notes
-----
- The output store layout is (t, c, z, y, x).
- If CuPy/cuCIM are available, registration downsampling and shift operations
  use GPU; fusion is performed on CPU using Numba kernels.
"""

import gc
import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import tensorstore as ts
from numba import njit, prange
from tqdm import trange, tqdm


# -----------------------------------------------------------------------------
# Optional GPU stack (safe globals)
# -----------------------------------------------------------------------------
USING_GPU = False

cp: Any | None = None
cp_shift: Any | None = None
ssim_cuda: Any | None = None

match_histograms: Any
block_reduce: Any
phase_cross_correlation: Any

_shift_cpu: Any | None = None
_ssim_cpu: Any | None = None

xp: Any = np

try:
    import cupy as _cp  # type: ignore
    from cucim.skimage.exposure import match_histograms as _mh  # type: ignore
    from cucim.skimage.measure import block_reduce as _br  # type: ignore
    from cucim.skimage.registration import phase_cross_correlation as _pcc  # type: ignore
    from cupyx.scipy.ndimage import shift as _cp_shift  # type: ignore
    from opm_processing.imageprocessing.ssim_cuda import (  # type: ignore
        structural_similarity_cupy_sep_shared as _ssim_cuda,
    )

    cp = _cp
    xp = _cp
    match_histograms = _mh
    block_reduce = _br
    phase_cross_correlation = _pcc
    cp_shift = _cp_shift
    ssim_cuda = _ssim_cuda
    USING_GPU = True
except Exception as exc:  # noqa: BLE001
    # GPU stack unavailable; fall back to CPU.
    from scipy.ndimage import shift as _cpu_shift  # type: ignore
    from skimage.exposure import match_histograms as _mh  # type: ignore
    from skimage.measure import block_reduce as _br  # type: ignore
    from skimage.metrics import structural_similarity as _cpu_ssim  # type: ignore
    from skimage.registration import phase_cross_correlation as _pcc  # type: ignore

    match_histograms = _mh
    block_reduce = _br
    phase_cross_correlation = _pcc
    _shift_cpu = _cpu_shift
    _ssim_cpu = _cpu_ssim
    USING_GPU = False

def _shift_array(arr: Any, shift_vec: Any) -> Any:
    """
    Shift an array using GPU if available, else CPU fallback.

    Parameters
    ----------
    arr : array-like
        Input array (2D or 3D) in the same backend namespace as `shift_vec`.
    shift_vec : array-like
        Shift vector in pixels/voxels. For 2D arrays, shape is (2,); for 3D
        arrays, shape is (3,). Uses (y, x) for 2D and (z, y, x) for 3D.

    Returns
    -------
    shifted : array-like
        Shifted array in the same backend namespace as the input.
    """
    if USING_GPU and cp_shift is not None:
        return cp_shift(arr, shift=shift_vec, order=1, prefilter=False)
    if _shift_cpu is None:
        raise RuntimeError("CPU shift backend is unavailable.")
    return _shift_cpu(arr, shift=shift_vec, order=1, prefilter=False)


def _ssim(arr1: Any, arr2: Any, win_size: int) -> float:
    """
    Compute SSIM, routing to GPU kernel if available, else CPU skimage.

    Parameters
    ----------
    arr1 : array-like
        Reference image/volume.
    arr2 : array-like
        Comparison image/volume (same shape as `arr1`).
    win_size : int
        SSIM window size.

    Returns
    -------
    score : float
        Structural similarity index in [-1, 1] (typically [0, 1] for images).
    """
    if USING_GPU and ssim_cuda is not None:
        return float(ssim_cuda(arr1, arr2, win_size=win_size))

    if _ssim_cpu is None:
        raise RuntimeError("CPU SSIM backend is unavailable.")

    arr1_np = np.asarray(arr1)
    arr2_np = np.asarray(arr2)

    data_range = float(np.ptp(arr1_np))
    if data_range == 0.0:
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
    Accumulate a weighted sub-volume into fused and weight buffers (in-place).

    Parameters
    ----------
    fused : numpy.ndarray
        Float32 accumulation buffer of shape (C, dz, Y, X) for the current shard.
    weight : numpy.ndarray
        Float32 weight accumulation buffer of shape (C, dz, Y, X).
    sub : numpy.ndarray
        Float32 tile slab of shape (C, sub_dz, Y_tile, X_tile) to blend.
    w3d : numpy.ndarray
        Float32 per-voxel weights of shape (sub_dz, Y_tile, X_tile).
    z_off : int
        Offset of `sub` z=0 within `fused` shard coordinates.
    y_off : int
        Offset of `sub` y=0 within fused global Y coordinates.
    x_off : int
        Offset of `sub` x=0 within fused global X coordinates.

    Returns
    -------
    None
        Operates in-place on `fused` and `weight`.
    """
    c_dim, _, _, _ = fused.shape
    _, sub_dz, y_sub, x_sub = sub.shape
    total = sub_dz * y_sub

    for idx in prange(total):
        dz_i = idx // y_sub
        y_i = idx % y_sub
        gz = z_off + dz_i
        gy = y_off + y_i
        w_line = w3d[dz_i, y_i]
        for c in range(c_dim):
            sub_line = sub[c, dz_i, y_i]
            base_f = fused[c, gz, gy]
            base_w = weight[c, gz, gy]
            for x_i in range(x_sub):
                gx = x_off + x_i
                w_val = w_line[x_i]
                base_f[gx] += sub_line[x_i] * w_val
                base_w[gx] += w_val


@njit(parallel=True)
def _normalize_shard(fused: np.ndarray, weight: np.ndarray) -> None:
    """
    Normalize fused by weight in-place.

    Parameters
    ----------
    fused : numpy.ndarray
        Float32 accumulation buffer of shape (C, dz, Y, X).
    weight : numpy.ndarray
        Float32 weight buffer of shape (C, dz, Y, X).

    Returns
    -------
    None
        Operates in-place on `fused`.
    """
    c_dim, dz, y_dim, x_dim = fused.shape
    total = c_dim * dz * y_dim

    for idx in prange(total):
        c = idx // (dz * y_dim)
        rem = idx % (dz * y_dim)
        z_i = rem // y_dim
        y_i = rem % y_dim
        base_f = fused[c, z_i, y_i]
        base_w = weight[c, z_i, y_i]
        for x_i in range(x_dim):
            w_val = base_w[x_i]
            base_f[x_i] = base_f[x_i] / w_val if w_val > 0 else 0.0


class TileFusion:
    """
    Register and fuse multi-tile OPM acquisitions into a global OME-NGFF Zarr v3 store.

    This implementation:
    - registers tiles independently per timepoint,
    - performs robust global shift optimization per timepoint (tile 0 anchored),
    - builds a global fused coordinate system spanning all timepoints,
    - fuses each timepoint into that common coordinate system,
    - writes a Zarr v3 OME-NGFF (t, c, z, y, x) store including multiscales.

    Parameters
    ----------
    root_path : str or pathlib.Path
        Path used to infer the processed datastore location. The code searches for:
        `{stem}_decon_deskewed.zarr`, `{stem}_deskewed.zarr`,
        `{stem}_decon_projection.zarr`, `{stem}_projection.zarr`.
    blend_pixels : tuple[int, int, int], default=(20, 600, 400)
        Feather ramp widths (bz, by, bx) used to build 1D weight profiles.
    downsample_factors : tuple[int, int, int], default=(3, 5, 5)
        Block-reduce factors (z, y, x) for registration patches.
    ssim_window : int, default=15
        SSIM window size for registration scoring.
    threshold : float, default=0.7
        Minimum SSIM score for accepting a pairwise link. Use 0.0 to accept all.
    multiscale_factors : sequence[int], default=(2, 4, 8, 16, 32)
        Downsampling factors for creating multiscale pyramid levels.
    resolution_multiples : sequence[int | sequence[int]], default=((1,1,1), ..., (32,32,32))
        Spatial scale multipliers recorded into NGFF metadata for each pyramid level.
    max_workers : int, default=8
        TensorStore I/O concurrency limit.
    debug : bool, default=False
        If True, emits debug logs.
    metrics_filename : str, default="stitching_metrics.json"
        File used to save/load pairwise registration links.
    channel_to_use : int, default=0
        Channel index used for registration.
    multiscale_downsample : {"stride", "block_mean"}, default="stride"
        Method for multiscale downsampling.
    """

    def __init__(
        self,
        root_path: str | Path,
        blend_pixels: tuple[int, int, int] = (20, 600, 400),
        downsample_factors: tuple[int, int, int] = (3, 5, 5),
        ssim_window: int = 15,
        threshold: float = 0.7,
        multiscale_factors: Sequence[int] = (2, 4, 8, 16, 32),
        resolution_multiples: Sequence[int | Sequence[int]] = (
            (1, 1, 1),
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 8),
            (16, 16, 16),
            (32, 32, 32),
        ),
        max_workers: int = 8,
        debug: bool = False,
        metrics_filename: str = "stitching_metrics.json",
        channel_to_use: int = 0,
        multiscale_downsample: str = "stride",
    ) -> None:
        self.root = Path(root_path)
        base = self.root.parents[0]
        stem = self.root.stem

        data_path = base / f"{stem}_decon_deskewed.zarr"
        if not data_path.exists():
            data_path = base / f"{stem}_deskewed.zarr"
            if not data_path.exists():
                data_path = base / f"{stem}_decon_projection.zarr"
                if not data_path.exists():
                    data_path = base / f"{stem}_projection.zarr"
                    if not data_path.exists():
                        raise FileNotFoundError("Processed data store not found.")
        self.data = data_path

        with open(self.data / "zarr.json", "r") as f:
            meta = json.load(f)

        ds = ts.open(
            {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(self.data)}}
        ).result()

        self._tile_positions: list[tuple[float, float, float]] = [
            tuple(
                meta["attributes"]["per_index_metadata"][str(t)][str(p)]["0"][
                    "stage_position"
                ]
            )
            for t in range(int(ds.shape[0]))
            for p in range(int(ds.shape[1]))
        ]
        self._pixel_size: tuple[float, float, float] = tuple(
            float(x) for x in meta["attributes"]["deskewed_voxel_size_um"]
        )

        self.downsample_factors = tuple(int(x) for x in downsample_factors)
        self.ssim_window = int(ssim_window)
        self.threshold = float(threshold)
        self.multiscale_factors = tuple(int(x) for x in multiscale_factors)
        self.resolution_multiples: list[tuple[int, int, int]] = [
            tuple(r) if hasattr(r, "__len__") else (int(r), int(r), int(r))
            for r in resolution_multiples
        ]
        self._max_workers = int(max_workers)
        self._debug = bool(debug)
        self.metrics_filename = str(metrics_filename)
        self._blend_pixels = tuple(int(x) for x in blend_pixels)
        self.channel_to_use = int(channel_to_use)

        if multiscale_downsample not in ("stride", "block_mean"):
            raise ValueError('multiscale_downsample must be "stride" or "block_mean".')
        self.multiscale_downsample = multiscale_downsample

        spec = {
            "context": {
                "file_io_concurrency": {"limit": self._max_workers},
                "data_copy_concurrency": {"limit": self._max_workers},
            },
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(self.data)},
        }
        self.ts = ts.open(spec, create=False, open=True).result()

        shape = self.ts.shape
        if len(shape) == 6:
            (
                self.time_dim,
                self.position_dim,
                self.channels,
                self.z_dim,
                self.y_dim,
                self.x_dim,
            ) = shape
            self._is_2d = False
        elif len(shape) == 5:
            self.time_dim, self.position_dim, self.channels, self.y_dim, self.x_dim = shape
            self.z_dim = 1
            self._is_2d = True
        else:
            raise ValueError(f"Unsupported data rank {len(shape)}; expected 5 or 6.")

        self._update_profiles()

        self.chunk_y = 1024
        self.chunk_x = 1024

        self.pairwise_metrics: dict[tuple[int, int], tuple[int, int, int, float]] = {}
        self.global_offsets: np.ndarray | None = None

        self.offset_um: tuple[float, float, float] | None = None
        self.unpadded_shape: tuple[int, int, int] | None = None
        self.padded_shape: tuple[int, int, int] | None = None

        self.fused_ts: ts.TensorStore | None = None
        self.shard_chunk: list[int] | None = None

    @property
    def debug(self) -> bool:
        """
        Get the debug flag.

        Returns
        -------
        debug : bool
            True if debug logging is enabled.
        """
        return self._debug

    @debug.setter
    def debug(self, flag: bool) -> None:
        """
        Set the debug flag.

        Parameters
        ----------
        flag : bool
            True to enable debug logging.

        Returns
        -------
        None
        """
        self._debug = bool(flag)

    def _update_profiles(self) -> None:
        """
        Recompute 1D feather profiles from blend_pixels and current data shape.

        Returns
        -------
        None
            Populates `z_profile`, `y_profile`, and `x_profile` attributes.
        """
        bz, by, bx = self._blend_pixels
        self.z_profile = self._make_1d_profile(int(self.z_dim), int(bz))
        self.y_profile = self._make_1d_profile(int(self.y_dim), int(by))
        self.x_profile = self._make_1d_profile(int(self.x_dim), int(bx))

    @staticmethod
    def _make_1d_profile(length: int, blend: int) -> np.ndarray:
        """
        Create a 1D feather profile with linear ramps at both ends.

        Parameters
        ----------
        length : int
            Axis length in voxels.
        blend : int
            Ramp width (voxels) at each end.

        Returns
        -------
        prof : numpy.ndarray
            Float32 array of shape (length,). Values are in [0, 1] with ramped
            edges; for very small `length` (<= 2) or `blend` <= 0, returns ones.

        Raises
        ------
        ValueError
            If `length` <= 0.
        """
        length = int(length)
        blend = int(blend)

        if length <= 0:
            raise ValueError(f"length must be > 0, got {length}")

        if length <= 2 or blend <= 0:
            return np.ones(length, dtype=np.float32)

        blend = min(blend, length)
        prof = np.ones(length, dtype=np.float32)

        if blend >= length:
            x = np.linspace(0.0, 1.0, length, dtype=np.float32)
            tent = 1.0 - np.abs(2.0 * x - 1.0)
            tent[0] = 0.0
            tent[-1] = 0.0
            tent[1:-1] = np.maximum(tent[1:-1], np.float32(1e-6))
            return tent.astype(np.float32, copy=False)

        ramp = np.linspace(0.0, 1.0, blend, endpoint=False, dtype=np.float32)
        prof[:blend] = ramp
        prof[-blend:] = ramp[::-1]

        if float(prof.max()) <= 0.0:
            prof[:] = 1.0
        return prof

    def _read_tile_volume(
        self,
        tile_idx: int,
        ch_sel: int | slice,
        z_slice: slice,
        y_slice: slice,
        x_slice: slice,
    ) -> np.ndarray:
        """
        Read a tile subvolume using a global flattened tile index.

        For time series:
            tile_idx = t_idx * position_dim + pos_idx

        Parameters
        ----------
        tile_idx : int
            Flattened tile index.
        ch_sel : int or slice
            Channel selection. If int, returns channel-first with singleton
            channel removed by TensorStore read. If slice, returns explicit C axis.
        z_slice : slice
            Z slice for 3D data. Ignored for 2D inputs except for normalization.
        y_slice : slice
            Y slice.
        x_slice : slice
            X slice.

        Returns
        -------
        arr : numpy.ndarray
            Float32 array in channel-first form:
            - 3D: (Z, Y, X) for int channel, or (C, Z, Y, X) for slice channels.
            - 2D: normalized to (C, 1, Y, X).

        Raises
        ------
        ValueError
            If `tile_idx` is negative or read rank is unexpected.
        IndexError
            If `tile_idx` maps to a timepoint >= time_dim.
        """
        n_pos = int(self.position_dim)
        if tile_idx < 0:
            raise ValueError(f"tile_idx must be >= 0, got {tile_idx}")

        t_idx = tile_idx // n_pos
        pos_idx = tile_idx % n_pos

        if t_idx >= int(self.time_dim):
            raise IndexError(
                f"tile_idx={tile_idx} maps to t_idx={t_idx}, but time_dim={self.time_dim}"
            )

        if self._is_2d:
            arr = self.ts[t_idx, pos_idx, ch_sel, y_slice, x_slice].read().result()
            if arr.ndim == 2:
                arr = arr[None, None, :, :]
            elif arr.ndim == 3:
                arr = arr[:, None, :, :]
            else:
                raise ValueError(f"Unexpected 2D tile ndim={arr.ndim}, shape={arr.shape}")
            return arr.astype(np.float32, copy=False)

        arr = self.ts[t_idx, pos_idx, ch_sel, z_slice, y_slice, x_slice].read().result()
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def register_and_score(g1: Any, g2: Any, win_size: int) -> tuple[tuple[float, float, float], float]:
        """
        Register `g2` to `g1` and compute an SSIM score.

        Steps:
        1) histogram-match g2 -> g1
        2) phase cross-correlation to estimate subpixel shift
        3) shift g2 by that estimate
        4) compute SSIM between g1 and shifted g2

        Parameters
        ----------
        g1 : array-like
            Fixed patch (2D YX or 3D ZYX).
        g2 : array-like
            Moving patch (same shape as g1).
        win_size : int
            SSIM window size.

        Returns
        -------
        shift : tuple[float, float, float]
            Estimated shift in (dz, dy, dx). For 2D patches, dz is 0.
        score : float
            SSIM score after shifting g2.

        Raises
        ------
        RuntimeError
            If a required backend function is unavailable.
        """
        arr1 = xp.asarray(g1, dtype=xp.float32)
        arr2 = xp.asarray(g2, dtype=xp.float32)

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
        score = _ssim(arr1, g2s, win_size=win_size)

        if USING_GPU and cp is not None:
            out_shift = cp.asnumpy(shift_ret)
        else:
            out_shift = np.asarray(shift_ret)

        return tuple(float(s) for s in out_shift), float(score)

    def refine_tile_positions_with_cross_correlation(
        self,
        downsample_factors: tuple[int, int, int] | None = None,
        ssim_window: int | None = None,
        ch_idx: int = 0,
        threshold: float | None = None,
    ) -> None:
        """
        Register and score all overlapping tile pairs independently per timepoint.

        Parameters
        ----------
        downsample_factors : tuple[int, int, int] or None, optional
            Per-axis block-reduce factors (z, y, x) for patch downsampling.
            If None, uses `self.downsample_factors`.
        ssim_window : int or None, optional
            SSIM window size. If None, uses `self.ssim_window`.
        ch_idx : int, default=0
            Channel index used to extract patches.
        threshold : float or None, optional
            SSIM acceptance threshold. If None, uses `self.threshold`.

        Returns
        -------
        None
            Populates `self.pairwise_metrics` with keys (i, j) in global
            flattened tile index space and values (dz, dy, dx, score).

        Notes
        -----
        - Links never cross timepoints; i and j always belong to the same t.
        - For 2D data, dz is forced to 0 and overlap gating ignores z.
        """
        df_in = self.downsample_factors if downsample_factors is None else downsample_factors
        sw = self.ssim_window if ssim_window is None else int(ssim_window)
        th = self.threshold if threshold is None else float(threshold)

        self.pairwise_metrics.clear()
        n_pos = int(self.position_dim)

        def bounds_1d(off: int, length: int) -> tuple[int, int]:
            return max(0, off), min(length, off + length)

        def effective_df_for_patch(patch: Any, df_zyx: tuple[int, int, int]) -> tuple[int, int, int]:
            arr = np.asarray(patch)
            if arr.ndim == 4:
                zyx = (arr.shape[1], arr.shape[2], arr.shape[3])
            elif arr.ndim == 3:
                zyx = (arr.shape[0], arr.shape[1], arr.shape[2])
            else:
                raise ValueError(f"Unexpected patch ndim={arr.ndim}, shape={arr.shape}")

            ez = max(1, min(int(df_zyx[0]), int(zyx[0])))
            ey = max(1, min(int(df_zyx[1]), int(zyx[1])))
            ex = max(1, min(int(df_zyx[2]), int(zyx[2])))
            return (ez, ey, ex)

        df_zyx_base = (int(df_in[0]), int(df_in[1]), int(df_in[2]))
        if int(self.z_dim) == 1:
            df_zyx_base = (1, df_zyx_base[1], df_zyx_base[2])

        with ThreadPoolExecutor(max_workers=2) as executor:
            for t in range(int(self.time_dim)):
                base = t * n_pos

                for i_pos in trange(
                    n_pos,
                    desc=f"register t={t + 1}/{int(self.time_dim)}",
                    leave=False,
                ):
                    i = base + i_pos

                    for j_pos in range(i_pos + 1, n_pos):
                        j = base + j_pos

                        phys = np.array(self._tile_positions[j]) - np.array(self._tile_positions[i])
                        vox_off = np.round(phys / np.array(self._pixel_size)).astype(int)

                        dz = int(vox_off[0])
                        dy = int(vox_off[1])
                        dx = int(vox_off[2])

                        if self._is_2d:
                            dz = 0

                        bounds_i = [
                            bounds_1d(dz, int(self.z_dim)),
                            bounds_1d(dy, int(self.y_dim)),
                            bounds_1d(dx, int(self.x_dim)),
                        ]
                        bounds_j = [
                            bounds_1d(-dz, int(self.z_dim)),
                            bounds_1d(-dy, int(self.y_dim)),
                            bounds_1d(-dx, int(self.x_dim)),
                        ]

                        if any(hi <= lo for (lo, hi) in bounds_i):
                            continue

                        def read_patch(gidx: int, bnds: list[tuple[int, int]]) -> np.ndarray:
                            z0, z1 = bnds[0]
                            y0, y1 = bnds[1]
                            x0, x1 = bnds[2]
                            return self._read_tile_volume(
                                gidx,
                                ch_idx,
                                slice(z0, z1),
                                slice(y0, y1),
                                slice(x0, x1),
                            )

                        patch_i = executor.submit(read_patch, i, bounds_i).result()
                        patch_j = executor.submit(read_patch, j, bounds_j).result()

                        df_zyx_eff = effective_df_for_patch(patch_i, df_zyx_base)

                        arr_i = xp.asarray(patch_i)
                        arr_j = xp.asarray(patch_j)

                        if arr_i.ndim == 4:
                            reduce_block = (1, *df_zyx_eff)
                        elif arr_i.ndim == 3:
                            reduce_block = df_zyx_eff
                        else:
                            raise ValueError(f"Unexpected patch ndim={arr_i.ndim}, shape={arr_i.shape}")

                        g1 = block_reduce(arr_i, block_size=reduce_block, func=xp.mean)
                        g2 = block_reduce(arr_j, block_size=reduce_block, func=xp.mean)

                        shift_ds, score = self.register_and_score(g1, g2, win_size=sw)
                        score = float(max(score, 1e-6))

                        if th != 0.0 and score < th:
                            continue

                        dz_s = int(np.round(shift_ds[0] * df_zyx_eff[0]))
                        dy_s = int(np.round(shift_ds[1] * df_zyx_eff[1]))
                        dx_s = int(np.round(shift_ds[2] * df_zyx_eff[2]))

                        max_shift = (20, 50, 100)
                        if (
                            abs(dz_s) > max_shift[0]
                            or abs(dy_s) > max_shift[1]
                            or abs(dx_s) > max_shift[2]
                        ):
                            if self._debug:
                                print(
                                    "Dropping link (%d, %d) shift=%s exceeds max=%s",
                                    i,
                                    j,
                                    (dz_s, dy_s, dx_s),
                                    max_shift,
                                )
                            continue

                        self.pairwise_metrics[(i, j)] = (dz_s, dy_s, dx_s, round(score, 3))

    @staticmethod
    def _solve_global(links: list[dict[str, Any]], n_tiles: int, fixed_indices: list[int]) -> np.ndarray:
        """
        Solve dense least-squares shifts per-axis with fixed tile constraints.

        Parameters
        ----------
        links : list[dict[str, Any]]
            List of link dicts, each with:
            - i : int, source tile index (local within timepoint)
            - j : int, destination tile index (local within timepoint)
            - t : numpy.ndarray, shape (3,), measured shift (dz, dy, dx)
            - w : float, weight
        n_tiles : int
            Number of tiles in the local timepoint graph.
        fixed_indices : list[int]
            Indices constrained to zero shift (anchors).

        Returns
        -------
        shifts : numpy.ndarray
            Array of shape (n_tiles, 3) containing optimized shifts (dz, dy, dx).
        """
        shifts = np.zeros((n_tiles, 3), dtype=np.float64)
        for axis in range(3):
            m = len(links) + len(fixed_indices)
            a = np.zeros((m, n_tiles), dtype=np.float64)
            b = np.zeros(m, dtype=np.float64)

            row = 0
            for link in links:
                i = int(link["i"])
                j = int(link["j"])
                t = float(link["t"][axis])
                w = float(link["w"])
                a[row, j] = w
                a[row, i] = -w
                b[row] = w * t
                row += 1

            for idx in fixed_indices:
                a[row, idx] = 1.0
                b[row] = 0.0
                row += 1

            sol, *_ = np.linalg.lstsq(a, b, rcond=None)
            shifts[:, axis] = sol

        return shifts

    def _two_round_opt(
        self,
        links: list[dict[str, Any]],
        n_tiles: int,
        fixed_indices: list[int],
        rel_thresh: float,
        abs_thresh: float,
        iterative: bool,
    ) -> np.ndarray:
        """
        Perform robust two-round (optionally iterative) optimization.

        Parameters
        ----------
        links : list[dict[str, Any]]
            Link list as in `_solve_global`.
        n_tiles : int
            Number of tiles in local graph.
        fixed_indices : list[int]
            Anchor indices constrained to zero.
        rel_thresh : float
            Relative threshold multiplier against median residual.
        abs_thresh : float
            Absolute residual threshold (voxels).
        iterative : bool
            If True, repeats outlier rejection until convergence.

        Returns
        -------
        shifts : numpy.ndarray
            Array of shape (n_tiles, 3) containing optimized shifts.
        """
        shifts = self._solve_global(links, n_tiles, fixed_indices)

        def residuals(ls: list[dict[str, Any]], sh: np.ndarray) -> np.ndarray:
            return np.array(
                [np.linalg.norm(sh[l["j"]] - sh[l["i"]] - l["t"]) for l in ls],
                dtype=np.float64,
            )

        work = links.copy()
        res = residuals(work, shifts)
        cutoff = max(abs_thresh, rel_thresh * float(np.median(res)))
        outliers = set(np.where(res > cutoff)[0])

        if iterative:
            while outliers:
                for k in sorted(outliers, reverse=True):
                    work.pop(k)
                shifts = self._solve_global(work, n_tiles, fixed_indices)
                res = residuals(work, shifts)
                cutoff = max(abs_thresh, rel_thresh * float(np.median(res)))
                outliers = set(np.where(res > cutoff)[0])
        else:
            for k in sorted(outliers, reverse=True):
                work.pop(k)
            shifts = self._solve_global(work, n_tiles, fixed_indices)

        return shifts

    def optimize_shifts(
        self,
        method: str = "TWO_ROUND_ITERATIVE",
        rel_thresh: float = 0.5,
        abs_thresh: float = 1.5,
    ) -> None:
        """
        Optimize global shifts independently per timepoint, anchoring tile 0.

        Parameters
        ----------
        method : str, default="TWO_ROUND_ITERATIVE"
            Optimization method. Supported values:
            - "ONE_ROUND"
            - "TWO_ROUND"
            - "TWO_ROUND_ITERATIVE"
        rel_thresh : float, default=0.5
            Relative outlier cutoff (multiplier of median residual).
        abs_thresh : float, default=1.5
            Absolute outlier cutoff (voxels).

        Returns
        -------
        None
            Populates `self.global_offsets` with shape (time_dim * position_dim, 3).

        Notes
        -----
        - If `self.pairwise_metrics` is empty, global_offsets is set to all zeros.
        - Tile index 0 is anchored at each timepoint (local index 0).
        """
        n_pos = int(self.position_dim)
        n_t = int(self.time_dim)
        n_tiles_total = len(self._tile_positions)

        self.global_offsets = np.zeros((n_tiles_total, 3), dtype=np.float64)
        if not self.pairwise_metrics:
            return

        links_all: list[dict[str, Any]] = [
            {
                "i": int(i),
                "j": int(j),
                "t": np.array(v[:3], dtype=np.float64),
                "w": float(np.sqrt(v[3])),
            }
            for (i, j), v in self.pairwise_metrics.items()
        ]

        is_iterative = method.endswith("ITERATIVE")
        if method == "ONE_ROUND":
            mode = "ONE_ROUND"
        elif method.startswith("TWO_ROUND"):
            mode = "TWO_ROUND"
        else:
            raise ValueError(f"Unknown method {method!r}")

        for t in range(n_t):
            base = t * n_pos
            lo = base
            hi = base + n_pos

            local_links: list[dict[str, Any]] = []
            for link in links_all:
                i = int(link["i"])
                j = int(link["j"])
                if lo <= i < hi and lo <= j < hi:
                    local_links.append(
                        {"i": i - base, "j": j - base, "t": link["t"], "w": link["w"]}
                    )

            if not local_links:
                continue

            fixed = [0]
            if mode == "ONE_ROUND":
                d_opt = self._solve_global(local_links, n_pos, fixed)
            else:
                d_opt = self._two_round_opt(
                    local_links,
                    n_pos,
                    fixed,
                    rel_thresh=rel_thresh,
                    abs_thresh=abs_thresh,
                    iterative=is_iterative,
                )

            self.global_offsets[base:hi, :] = d_opt

    def save_pairwise_metrics(self, filepath: str | Path) -> None:
        """
        Save pairwise link metrics to JSON.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Output JSON path.

        Returns
        -------
        None
        """
        path = Path(filepath)
        out = {f"{i},{j}": list(v) for (i, j), v in self.pairwise_metrics.items()}
        with open(path, "w") as f:
            json.dump(out, f)

    def load_pairwise_metrics(self, filepath: str | Path) -> None:
        """
        Load pairwise link metrics from JSON.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Input JSON path.

        Returns
        -------
        None
            Populates `self.pairwise_metrics`.

        Raises
        ------
        FileNotFoundError
            If the JSON file does not exist.
        """
        path = Path(filepath)
        with open(path, "r") as f:
            data = json.load(f)
        self.pairwise_metrics = {
            tuple(map(int, k.split(","))): tuple(v) for k, v in data.items()
        }

    def _compute_fused_image_space(self) -> None:
        """
        Compute a global fused space spanning all timepoints.

        Returns
        -------
        None
            Sets:
            - `self.unpadded_shape` : (Z, Y, X) voxels
            - `self.offset_um` : (z, y, x) physical origin (microns)
        """
        pos = np.asarray(self._tile_positions, dtype=np.float64)
        min_z, min_y, min_x = pos.min(axis=0)

        dz_um, dy_um, dx_um = self._pixel_size

        max_z = float(pos[:, 0].max() + float(self.z_dim) * dz_um)
        max_y = float(pos[:, 1].max() + float(self.y_dim) * dy_um)
        max_x = float(pos[:, 2].max() + float(self.x_dim) * dx_um)

        sz = int(np.ceil((max_z - min_z) / dz_um))
        sy = int(np.ceil((max_y - min_y) / dy_um))
        sx = int(np.ceil((max_x - min_x) / dx_um))

        self.unpadded_shape = (sz, sy, sx)
        self.offset_um = (float(min_z), float(min_y), float(min_x))

    def _pad_to_chunk_multiple(self) -> None:
        """
        Pad the fused shape to multiples of tile shape.

        Returns
        -------
        None
            Sets `self.padded_shape` as (Z, Y, X).
        """
        if self.unpadded_shape is None:
            raise RuntimeError("unpadded_shape not computed.")

        tz, ty, tx = int(self.z_dim), int(self.y_dim), int(self.x_dim)
        sz, sy, sx = self.unpadded_shape

        pz = (-sz) % tz
        py = (-sy) % ty
        px = (-sx) % tx

        self.padded_shape = (sz + pz, sy + py, sx + px)

    def _create_fused_tensorstore(
        self,
        output_path: str | Path,
        z_slices_per_shard: int = 4,
    ) -> tuple[ts.TensorStore, list[int]]:
        """
        Create the output Zarr v3 store for fused data.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Output dataset path (the `.../scale0/image` node).
        z_slices_per_shard : int, default=4
            Number of z slices per sharding chunk.

        Returns
        -------
        handle : tensorstore.TensorStore
            Open TensorStore handle to the created dataset.
        shard_chunk : list[int]
            Chunk shape used for the regular chunk grid (t, c, z, y, x).

        Raises
        ------
        RuntimeError
            If `self.padded_shape` is not computed.
        """
        if self.padded_shape is None:
            raise RuntimeError("padded_shape not computed.")

        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        t_dim = int(self.time_dim)
        c_dim = int(self.channels)
        z_dim, y_dim, x_dim = (
            int(self.padded_shape[0]),
            int(self.padded_shape[1]),
            int(self.padded_shape[2]),
        )
        full_shape = [t_dim, c_dim, z_dim, y_dim, x_dim]

        shard_chunk = [1, 1, int(z_slices_per_shard), int(self.chunk_y) * 2, int(self.chunk_x) * 2]
        codec_chunk = [1, 1, 1, int(self.chunk_y), int(self.chunk_x)]

        config = {
            "context": {
                "file_io_concurrency": {"limit": int(self._max_workers)},
                "data_copy_concurrency": {"limit": int(self._max_workers)},
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
                                    "configuration": {"cname": "zstd", "clevel": 5, "shuffle": "bitshuffle"},
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

        handle = ts.open(config, create=True, open=True).result()
        return handle, shard_chunk

    def _fuse_by_shard(self) -> None:
        """
        Fuse all timepoints into the global fused store using shard-centric writes.

        Returns
        -------
        None
            Writes into `self.fused_ts`.

        Raises
        ------
        RuntimeError
            If fusion prerequisites are not initialized.
        """
        if self.fused_ts is None or self.shard_chunk is None or self.offset_um is None or self.padded_shape is None:
            raise RuntimeError("Fusion not initialized: compute fused space and create output store first.")

        n_pos = int(self.position_dim)
        n_t = int(self.time_dim)

        z_step = int(self.shard_chunk[2])
        pad_z, pad_y, pad_x = (int(self.padded_shape[0]), int(self.padded_shape[1]), int(self.padded_shape[2]))
        nz = (pad_z + z_step - 1) // z_step

        dz_um, dy_um, dx_um = self._pixel_size
        off_z_um, off_y_um, off_x_um = self.offset_um

        futures: list[Any] = []

        for t in trange(n_t, desc=f"scale0", leave=True):
            base = t * n_pos

            offsets_t: list[tuple[int, int, int]] = []
            for p in range(n_pos):
                z_um, y_um, x_um = self._tile_positions[base + p]
                oz = int(np.round((z_um - off_z_um) / dz_um))
                oy = int(np.round((y_um - off_y_um) / dy_um))
                ox = int(np.round((x_um - off_x_um) / dx_um))
                offsets_t.append((oz, oy, ox))

            for shard_idx in range(nz):
                z0 = shard_idx * z_step
                z1 = min(z0 + z_step, pad_z)
                dz = z1 - z0

                for c in range(int(self.channels)):
                    fused_block = np.zeros((1, dz, pad_y, pad_x), dtype=np.float32)
                    weight_sum = np.zeros_like(fused_block)

                    for p, (oz, oy, ox) in enumerate(offsets_t):
                        tz0 = max(z0, oz)
                        tz1 = min(z1, oz + int(self.z_dim))
                        if tz1 <= tz0:
                            continue

                        local_z0 = tz0 - oz
                        local_z1 = tz1 - oz
                        tile_gidx = base + p

                        sub = self._read_tile_volume(
                            tile_gidx,
                            slice(c, c + 1),
                            slice(local_z0, local_z1),
                            slice(0, int(self.y_dim)),
                            slice(0, int(self.x_dim)),
                        )

                        wz = self.z_profile[local_z0:local_z1]
                        wy = self.y_profile
                        wx = self.x_profile
                        w3d = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]

                        z_off = tz0 - z0
                        _accumulate_tile_shard(
                            fused_block,
                            weight_sum,
                            sub,
                            w3d.astype(np.float32, copy=False),
                            int(z_off),
                            int(oy),
                            int(ox),
                        )

                    _normalize_shard(fused_block, weight_sum)

                    fut = self.fused_ts[
                        t,
                        slice(c, c + 1),
                        slice(z0, z1),
                        slice(0, pad_y),
                        slice(0, pad_x),
                    ].write(fused_block.astype(np.uint16))
                    futures.append(fut)

                    del fused_block, weight_sum
                    gc.collect()
                    if USING_GPU and cp is not None:
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()

        for fut in futures:
            fut.result()

    def _create_multiscales(
        self,
        omezarr_path: Path,
        factors: Sequence[int] = (2, 4, 8),
        z_slices_per_shard: int = 4,
    ) -> None:
        """
        Build NGFF multiscales by downsampling Z/Y/X iteratively.

        Parameters
        ----------
        omezarr_path : pathlib.Path
            Root NGFF group path.
        factors : sequence[int], default=(2, 4, 8)
            Pyramid factors. Each level is downsampled relative to scale0.
        z_slices_per_shard : int, default=4
            Z slices per sharding chunk for each scale.

        Returns
        -------
        None
            Writes additional scales under `omezarr_path/scale{n}/image`.

        Notes
        -----
        - Preserves (t, c) axes unchanged.
        - Does not modify the scale0 TensorStore handle.
        - Progress display:
            * Outer bar (scales) persists.
            * Inner bar (timepoints) is per-scale and disappears when that scale finishes.
        """
        pad0 = self.padded_shape
        cy0, cx0 = int(self.chunk_y), int(self.chunk_x)
        shard0 = self.shard_chunk
        fused0 = self.fused_ts

        inp: ts.TensorStore | None = None
        try:
            # Outer bar: scales (stays)
            for idx, factor in enumerate(factors):
                out_path = omezarr_path / f"scale{idx + 1}" / "image"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                prev = omezarr_path / f"scale{idx}" / "image"
                inp = ts.open(
                    {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(prev)}}
                ).result()

                factor_to_use = (
                    int(factor) if idx == 0 else int(factor) // int(factors[idx - 1])
                )
                if factor_to_use < 1:
                    raise ValueError(f"Invalid pyramid factors: {factors}")

                z_factor = factor_to_use if not self._is_2d else 1

                t_dim = int(inp.shape[0])
                c_dim = int(inp.shape[1])
                z_in = int(inp.shape[2])
                y_in = int(inp.shape[3])
                x_in = int(inp.shape[4])

                new_z = max(1, (z_in + z_factor - 1) // z_factor)
                new_y = max(1, (y_in + factor_to_use - 1) // factor_to_use)
                new_x = max(1, (x_in + factor_to_use - 1) // factor_to_use)
                shard_z = min(int(z_slices_per_shard), int(new_z))

                chunk_y = 1024 if new_y >= 2048 else max(1, new_y // 4)
                chunk_x = 1024 if new_x >= 2048 else max(1, new_x // 4)

                # Scale-local geometry (only for writing this scale)
                self.padded_shape = (int(new_z), int(new_y), int(new_x))
                self.chunk_y, self.chunk_x = int(chunk_y), int(chunk_x)

                out_ts, _ = self._create_fused_tensorstore(
                    output_path=out_path, z_slices_per_shard=shard_z
                )

                # Inner bar: timepoints (disappears)
                # Writes timepoints one-by-one so t_idx can be arbitrarily large.
                c_block = min(4, c_dim)
                for t in trange(t_dim,desc=f"scale{idx+1}", leave=True):
                    t0 = int(t)
                    t1 = t0 + 1

                    for c0 in range(0, c_dim, c_block):
                        c1 = min(c_dim, c0 + c_block)

                        for z0 in range(0, new_z, shard_z):
                            bz = min(shard_z, new_z - z0)
                            in_z0 = z0 * z_factor
                            in_z1 = min(z_in, (z0 + bz) * z_factor)

                            for y0 in range(0, new_y, chunk_y):
                                by = min(chunk_y, new_y - y0)
                                in_y0 = y0 * factor_to_use
                                in_y1 = min(y_in, (y0 + by) * factor_to_use)

                                for x0 in range(0, new_x, chunk_x):
                                    bx = min(chunk_x, new_x - x0)
                                    in_x0 = x0 * factor_to_use
                                    in_x1 = min(x_in, (x0 + bx) * factor_to_use)

                                    slab = inp[
                                        t0:t1,
                                        c0:c1,
                                        in_z0:in_z1,
                                        in_y0:in_y1,
                                        in_x0:in_x1,
                                    ].read().result()

                                    if self.multiscale_downsample == "stride":
                                        down = slab[
                                            ...,
                                            ::z_factor,
                                            ::factor_to_use,
                                            ::factor_to_use,
                                        ]
                                    else:
                                        arr = xp.asarray(slab)
                                        block = (1, 1, z_factor, factor_to_use, factor_to_use)
                                        down_arr = block_reduce(arr, block_size=block, func=xp.mean)
                                        down = (
                                            cp.asnumpy(down_arr)
                                            if USING_GPU and cp is not None
                                            else np.asarray(down_arr)
                                        )

                                    down = down.astype(slab.dtype, copy=False)

                                    out_ts[
                                        t0:t1,
                                        c0:c1,
                                        z0 : z0 + bz,
                                        y0 : y0 + by,
                                        x0 : x0 + bx,
                                    ].write(down).result()

                # zarr.json for this scale group
                ngff = {
                    "attributes": {"_ARRAY_DIMENSIONS": ["t", "c", "z", "y", "x"]},
                    "zarr_format": 3,
                    "consolidated_metadata": "null",
                    "node_type": "group",
                }
                with open(omezarr_path / f"scale{idx + 1}" / "zarr.json", "w") as f:
                    json.dump(ngff, f, indent=2)

        finally:
            # Restore scale0 state
            self.padded_shape = pad0
            self.chunk_y, self.chunk_x = cy0, cx0
            self.shard_chunk = shard0
            self.fused_ts = fused0

    def _generate_ngff_zarr3_json(
        self,
        omezarr_path: Path,
        resolution_multiples: Sequence[int | Sequence[int]],
        dataset_name: str = "image",
        version: str = "0.5",
    ) -> None:
        """
        Write OME-NGFF v0.5 metadata (zarr.json) for a Zarr v3 multiscale group.

        Parameters
        ----------
        omezarr_path : pathlib.Path
            Root NGFF group path.
        resolution_multiples : sequence[int | sequence[int]]
            Spatial resolution multipliers per pyramid level.
        dataset_name : str, default="image"
            Dataset node name within each scale group.
        version : str, default="0.5"
            NGFF version string.

        Returns
        -------
        None
            Writes `omezarr_path/zarr.json`.

        Raises
        ------
        RuntimeError
            If `offset_um` is not computed.
        """
        if self.offset_um is None:
            raise RuntimeError("offset_um not computed.")

        axes = [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ]

        norm_res: list[tuple[int, int, int]] = [
            tuple(r) if hasattr(r, "__len__") else (int(r), int(r), int(r))
            for r in resolution_multiples
        ]

        base_scale = [1.0, 1.0] + [float(s) for s in self._pixel_size]
        off_z, off_y, off_x = self.offset_um
        trans0 = [0.0, 0.0, float(off_z), float(off_y), float(off_x)]

        datasets: list[dict[str, Any]] = []
        prev_sp = base_scale[2:]

        for lvl, factors in enumerate(norm_res):
            spatial = [base_scale[i + 2] * float(factors[i]) for i in range(3)]
            scale = [1.0, 1.0] + spatial

            if lvl == 0:
                translation = trans0
            else:
                prev_translation = datasets[-1]["coordinateTransformations"][1]["translation"]
                translation = [
                    0.0,
                    0.0,
                    float(prev_translation[2]) + 0.5 * float(prev_sp[0]),
                    float(prev_translation[3]) + 0.5 * float(prev_sp[1]),
                    float(prev_translation[4]) + 0.5 * float(prev_sp[2]),
                ]

            datasets.append(
                {
                    "path": f"scale{lvl}/{dataset_name}",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": scale},
                        {"type": "translation", "translation": translation},
                    ],
                }
            )
            prev_sp = spatial

        mult = {"axes": axes, "datasets": datasets, "name": dataset_name, "@type": "ngff:Image"}
        metadata = {
            "attributes": {"ome": {"version": version, "multiscales": [mult]}},
            "zarr_format": 3,
            "node_type": "group",
        }

        with open(omezarr_path / "zarr.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def run(self) -> None:
        """
        Execute the full registration + fusion pipeline.

        Returns
        -------
        None
            Writes the fused NGFF store to:
            `{base}/{stem}_fused_deskewed.ome.zarr`

        Raises
        ------
        RuntimeError
            If required intermediate computations fail.
        """
        base = self.root.parents[0]
        metrics_path = base / self.metrics_filename

        try:
            self.load_pairwise_metrics(metrics_path)
        except FileNotFoundError:
            self.refine_tile_positions_with_cross_correlation(
                downsample_factors=self.downsample_factors,
                ch_idx=self.channel_to_use,
                threshold=self.threshold,
            )
            self.save_pairwise_metrics(metrics_path)

        self.optimize_shifts(method="TWO_ROUND_ITERATIVE", rel_thresh=0.5, abs_thresh=1.5)

        gc.collect()
        if USING_GPU and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        if self.global_offsets is None:
            raise RuntimeError("global_offsets was not computed.")

        self._tile_positions = [
            tuple(np.array(pos) + off * np.array(self._pixel_size))
            for pos, off in zip(self._tile_positions, self.global_offsets)
        ]

        self._compute_fused_image_space()
        self._pad_to_chunk_multiple()

        omezarr = base / f"{self.root.stem}_fused.ome.zarr"
        scale0 = omezarr / "scale0" / "image"

        self.fused_ts, self.shard_chunk = self._create_fused_tensorstore(output_path=scale0)
        self._fuse_by_shard()

        (omezarr / "scale0").mkdir(parents=True, exist_ok=True)
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