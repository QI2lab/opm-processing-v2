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
5) Build and stream NGFF multiscales through yaozarrs.

Notes
-----
- The output store layout is (t, c, z, y, x).
- If CuPy/cuCIM are available, registration downsampling and shift operations
  use GPU; fusion is performed on CPU using Numba kernels.
"""

import gc
import json
import math
from collections import deque
from collections.abc import Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import tensorstore as ts
from numba import njit
from tqdm import tqdm
from yaozarrs import v05
from yaozarrs.write.v05 import prepare_image

from opm_processing.cuda import preload_cuda_libraries
from opm_processing.dataio.acquisition import (
    acquisition_stem,
    inspect_acquisition,
    resolve_acquisition_path,
)
from opm_processing.dataio.ngff import (
    round_spatial_values,
    round_tczyx_transform,
)
from opm_processing.dataio.position_collection import open_position_collection
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)
from opm_processing.dataio.roi import PhysicalRoi
from opm_processing.imageprocessing.coordinates import (
    stage_z_level_indices,
    stage_positions_to_image_coordinates,
)
from opm_processing.imageprocessing.opmtools import orthogonal_deskew


# -----------------------------------------------------------------------------
# Optional GPU stack (safe globals)
# -----------------------------------------------------------------------------
USING_GPU = False
GPU_IMPORT_ERROR: Exception | None = None

cp: Any | None = None
ssim_cuda: Any | None = None

match_histograms: Any
block_reduce: Any
phase_cross_correlation: Any
sobel_filter: Any

_ssim_cpu: Any | None = None

xp: Any = np

preload_cuda_libraries()

try:
    import cupy as _cp  # type: ignore
    from cucim.skimage.exposure import match_histograms as _mh  # type: ignore
    from cucim.skimage.measure import block_reduce as _br  # type: ignore
    from cucim.skimage.metrics import structural_similarity as _ssim_gpu  # type: ignore
    from cucim.skimage.registration import phase_cross_correlation as _pcc  # type: ignore
    from cupyx.scipy.ndimage import sobel as _sobel  # type: ignore

    cp = _cp
    xp = _cp
    match_histograms = _mh
    block_reduce = _br
    phase_cross_correlation = _pcc
    sobel_filter = _sobel
    ssim_cuda = _ssim_gpu
    USING_GPU = True
except Exception as error:  # noqa: BLE001
    # GPU stack unavailable; fall back to CPU.
    GPU_IMPORT_ERROR = error
    from skimage.exposure import match_histograms as _mh  # type: ignore
    from skimage.measure import block_reduce as _br  # type: ignore
    from skimage.metrics import structural_similarity as _cpu_ssim  # type: ignore
    from skimage.registration import phase_cross_correlation as _pcc  # type: ignore
    from scipy.ndimage import sobel as _sobel  # type: ignore

    match_histograms = _mh
    block_reduce = _br
    phase_cross_correlation = _pcc
    sobel_filter = _sobel
    _ssim_cpu = _cpu_ssim
    USING_GPU = False


def _default_fusion_workers() -> int:
    """Use up to eight physical cores to preserve deep, aligned I/O blocks.

    Returns
    -------
    int
        Concurrent fusion worker count bounded by available resources.
    """
    physical = psutil.cpu_count(logical=False)
    if physical is not None:
        return max(1, min(8, int(physical)))
    logical = psutil.cpu_count(logical=True)
    return max(1, min(8, int(logical or 2) // 2))


def fusion_backend_status(
    max_workers: int | None = None,
) -> dict[str, str | bool | int | None]:
    """Return observable registration and fusion backend information.

    Parameters
    ----------
    max_workers : int | None
        Concurrent fusion workers, or None for the available-memory default.

    Returns
    -------
    dict
        Active registration and fusion backends, worker count, and any CUDA import error.
    """
    workers = _default_fusion_workers() if max_workers is None else int(max_workers)
    return {
        "gpu_registration": USING_GPU,
        "registration_backend": "cupy/cucim" if USING_GPU else "numpy/scikit-image",
        "fusion_backend": "threaded-blocks/numba-cpu",
        "fusion_workers": workers,
        "gpu_error": None if GPU_IMPORT_ERROR is None else repr(GPU_IMPORT_ERROR),
    }


def require_gpu_backend() -> None:
    """Raise with the captured initialization error when CUDA is unavailable."""
    if USING_GPU:
        return
    detail = "unknown initialization error"
    if GPU_IMPORT_ERROR is not None:
        detail = f"{type(GPU_IMPORT_ERROR).__name__}: {GPU_IMPORT_ERROR}"
    raise RuntimeError(
        "GPU registration was requested, but the CuPy/cuCIM backend could not "
        f"initialize ({detail}). Install/run the project's gpu extra; on Windows, "
        "also install the local pure-Python cuCIM package described in README.md."
    )


_PROCESSED_SUFFIXES = (
    "_deskewed.ome.zarr",
    "_projection.ome.zarr",
    "_2d.ome.zarr",
    "_decon_deskewed.ome.zarr",
    "_decon_projection.ome.zarr",
    "_decon_2d.ome.zarr",
)


def _processed_store_identity(path: Path) -> tuple[str, str] | None:
    """Return the acquisition stem and processed suffix for a full tile store.

    Parameters
    ----------
    path : Path
        Candidate processed tile collection directory.

    Returns
    -------
    tuple[str, str] | None
        Acquisition name and supported processed suffix, or None for other stores.
    """
    for suffix in sorted(_PROCESSED_SUFFIXES, key=len, reverse=True):
        if path.name.endswith(suffix):
            stem = path.name[: -len(suffix)]
            if stem.endswith("_max_z"):
                return None
            return stem, suffix
    return None


def resolve_fusion_input(
    path: str | Path,
) -> tuple[Path, Path, str, Path | None]:
    """Locate fusion-ready processed data from a source or output directory.

    Returns the output directory, processed collection, acquisition stem, and
    source acquisition path when the latter can be resolved locally.

    Parameters
    ----------
    path : str | Path
        Acquisition, processed tile store, or directory containing processed outputs.

    Returns
    -------
    tuple[Path, Path, str, Path | None]
        Output directory, tile collection, acquisition name, and locally resolved raw source.
    """
    candidate = Path(path).expanduser().resolve()

    direct_identity = _processed_store_identity(candidate)
    search_dir = candidate.parent if direct_identity is not None else candidate
    if direct_identity is not None:
        matches = [(candidate, *direct_identity)]
    else:
        matches = []
        for item in search_dir.iterdir():
            if not item.is_dir():
                continue
            identity = _processed_store_identity(item)
            if identity is not None:
                matches.append((item, *identity))

    if matches:
        stems = {stem for _, stem, _ in matches}
        if len(stems) != 1:
            found = ", ".join(sorted(stems))
            raise ValueError(
                f"Expected processed data for one acquisition in {search_dir}, "
                f"found: {found}"
            )
        data_path, stem, _ = min(
            matches,
            key=lambda match: _PROCESSED_SUFFIXES.index(match[2]),
        )
        return search_dir, data_path, stem, None

    source_path = resolve_acquisition_path(candidate)
    output_dir = source_path.parent
    stem = acquisition_stem(source_path)
    checked = []
    for suffix in _PROCESSED_SUFFIXES:
        processed_path = output_dir / f"{stem}{suffix}"
        checked.append(processed_path)
        if processed_path.is_dir():
            return output_dir, processed_path, stem, source_path
    raise FileNotFoundError(
        "Processed data store not found. Checked: "
        + ", ".join(str(item) for item in checked)
    )


def _ssim(arr1: Any, arr2: Any, win_size: int) -> float:
    """Compute SSIM, routing to GPU kernel if available, else CPU skimage.

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
        data_range = float(xp.ptp(arr1))
        if data_range == 0.0:
            data_range = 1.0
        return float(
            ssim_cuda(
                arr1,
                arr2,
                win_size=win_size,
                data_range=data_range,
            )
        )

    if _ssim_cpu is None:
        raise RuntimeError("CPU SSIM backend is unavailable.")

    arr1_np = np.asarray(arr1)
    arr2_np = np.asarray(arr2)

    data_range = float(np.ptp(arr1_np))
    if data_range == 0.0:
        data_range = 1.0

    return float(_ssim_cpu(arr1_np, arr2_np, win_size=win_size, data_range=data_range))


def _aligned_registration_views(
    fixed: Any,
    moving: Any,
    shift: Sequence[float],
) -> tuple[Any, Any]:
    """Return corresponding valid views after an integer moving-image shift.

    Parameters
    ----------
    fixed : Any
        Reference overlap patch used for pairwise registration.
    moving : Any
        Neighbor overlap patch to align with the reference.
    shift : Sequence[float]
        Integer translation applied to the moving patch, in array-axis order.

    Returns
    -------
    tuple
        Matching fixed and shifted-moving views clipped to their overlapping samples.
    """
    fixed_slices: list[slice] = []
    moving_slices: list[slice] = []
    for length, offset_float in zip(fixed.shape, shift):
        offset = int(np.rint(float(offset_float)))
        if abs(offset) >= int(length):
            raise ValueError("registration shift leaves no overlapping samples")
        if offset >= 0:
            fixed_slices.append(slice(offset, int(length)))
            moving_slices.append(slice(0, int(length) - offset))
        else:
            fixed_slices.append(slice(0, int(length) + offset))
            moving_slices.append(slice(-offset, int(length)))

    return fixed[tuple(fixed_slices)], moving[tuple(moving_slices)]


def _bounded_phase_correlation_peak(
    fixed: Any,
    moving: Any,
    max_shift: Sequence[float],
) -> Any:
    """Return the strongest phase-correlation peak inside physical shift limits.

    Parameters
    ----------
    fixed : Any
        Reference overlap patch used for pairwise registration.
    moving : Any
        Neighbor overlap patch to align with the reference.
    max_shift : Sequence[float]
        Maximum absolute registration correction along each sampled axis.

    Returns
    -------
    np.ndarray
        Best integer phase-correlation correction within the per-axis search limits.
    """
    limits = xp.asarray(max_shift, dtype=xp.float32)

    frequency_product = xp.fft.fftn(fixed) * xp.fft.fftn(moving).conj()
    magnitude = xp.abs(frequency_product)
    epsilon = xp.finfo(magnitude.dtype).eps
    frequency_product /= xp.maximum(magnitude, epsilon)
    cross_correlation = xp.abs(xp.fft.ifftn(frequency_product))

    valid_peak = xp.ones(cross_correlation.shape, dtype=xp.bool_)
    signed_coordinates: list[Any] = []
    for axis, (length, limit) in enumerate(zip(cross_correlation.shape, limits)):
        coordinates = xp.arange(length)
        signed = xp.where(
            coordinates > length // 2,
            coordinates - length,
            coordinates,
        )
        signed_coordinates.append(signed)
        reshape = [1] * cross_correlation.ndim
        reshape[axis] = length
        valid_peak &= xp.abs(signed.reshape(reshape)) <= limit

    peak_index = xp.unravel_index(
        xp.argmax(xp.where(valid_peak, cross_correlation, -xp.inf)),
        cross_correlation.shape,
    )
    return xp.asarray(
        [signed_coordinates[axis][index] for axis, index in enumerate(peak_index)],
        dtype=xp.float32,
    )


class _RegistrationDeviceBuffers:
    """Grow-only reusable CuPy staging buffers for a registration pair."""

    def __init__(self, cupy_module: Any) -> None:
        """Create reusable GPU registration transfer buffers.

        Parameters
        ----------
        cupy_module : Any
            Initialized CuPy module used for allocation and asynchronous host transfers.
        """
        self._cupy = cupy_module
        self._buffers: list[Any | None] = [None, None]
        self._dtypes: list[np.dtype[Any] | None] = [None, None]

    def _stage_one(self, index: int, host: np.ndarray) -> Any:
        """Copy an overlap patch into one reusable GPU staging buffer.

        Parameters
        ----------
        index : int
            Tile or registration-link index used by this calculation.
        host : np.ndarray
            Host overlap patch copied into the staging buffer.

        Returns
        -------
        Any
            Device overlap array populated by the host transfer.
        """
        contiguous = np.ascontiguousarray(host)
        dtype = contiguous.dtype
        required = int(contiguous.size)
        buffer = self._buffers[index]
        if (
            buffer is None
            or self._dtypes[index] != dtype
            or int(buffer.size) < required
        ):
            buffer = self._cupy.empty(required, dtype=dtype)
            self._buffers[index] = buffer
            self._dtypes[index] = dtype
        view = buffer[:required].reshape(contiguous.shape)
        view.set(contiguous)
        return view

    def stage(self, fixed: np.ndarray, moving: np.ndarray) -> tuple[Any, Any]:
        """Copy a host pair into reusable device allocations.

        Parameters
        ----------
        fixed : np.ndarray
            Reference overlap patch used for pairwise registration.
        moving : np.ndarray
            Neighbor overlap patch to align with the reference.

        Returns
        -------
        tuple
            Reusable device arrays populated with the fixed and moving overlap patches.
        """
        return self._stage_one(0, fixed), self._stage_one(1, moving)


@njit(nogil=True)
def _accumulate_tile_block(
    fused: np.ndarray,
    weight: np.ndarray,
    source: np.ndarray,
    channel_present: np.ndarray,
    valid_zy: np.ndarray,
    z_weights: np.ndarray,
    y_weights: np.ndarray,
    x_weights: np.ndarray,
    z_off: int,
    y_off: int,
    x_off: int,
    source_z: int = 0,
    source_y: int = 0,
    source_x: int = 0,
    channel_gains: np.ndarray | None = None,
) -> None:
    """Accumulate supported channel rows without copying source sub-volumes.

    Parameters
    ----------
    fused : numpy.ndarray
        Float32 accumulation buffer of shape (C, dz, Y, X) for the current block.
    weight : numpy.ndarray
        Float32 weight accumulation buffer of shape (C, dz, Y, X), or
        (1, dz, Y, X) when every contributing tile contains all channels.
    source : numpy.ndarray
        Source CZYX block. The feather-profile lengths select the crop extent;
        source offsets locate that crop without making a strided sub-volume.
    channel_present : numpy.ndarray
        Boolean vector identifying channels that contain signal in this tile.
    valid_zy : numpy.ndarray
        Metadata-derived deskew support for the slab, with shape
        (sub_dz, Y_tile). Unsupported rows contribute neither signal nor weight;
        pixel intensity is never used to infer support.
    z_weights, y_weights, x_weights : numpy.ndarray
        Separable float32 feather profiles for the tile sub-volume.
    z_off : int
        Z origin of the weighted crop within the accumulation buffers.
    y_off : int
        Y origin of the weighted crop within the accumulation buffers.
    x_off : int
        X origin of the weighted crop within the accumulation buffers.
    source_z, source_y, source_x : int
        Crop origin inside the source block, defaulting to its first voxel.
    channel_gains : numpy.ndarray | None
        Float32 intensity correction per channel for this acquisition depth,
        or None for unit gains. Correction precedes feathering without allocating
        a scaled source block.

    Returns
    -------
    None
        Operates in-place on `fused` and `weight`.

    Notes
    -----
    Uncorrected uint16 sources retain Numba's double-precision multiply/add
    intermediates before storing into float32 accumulation. Casting them first
    can move normalized pixels across an integer truncation boundary.
    """
    c_dim, _, _, _ = fused.shape
    sub_dz, y_sub, x_sub = len(z_weights), len(y_weights), len(x_weights)
    total = sub_dz * y_sub

    for c in range(c_dim):
        if not channel_present[c]:
            continue
        weight_channel = 0 if weight.shape[0] == 1 else c
        gain = np.float32(1) if channel_gains is None else channel_gains[c]
        for idx in range(total):
            dz_i = idx // y_sub
            y_i = idx % y_sub
            if not valid_zy[dz_i, y_i]:
                continue
            gz = z_off + dz_i
            gy = y_off + y_i
            zy_weight = z_weights[dz_i] * y_weights[y_i]
            for x_i in range(x_sub):
                gx = x_off + x_i
                w_val = zy_weight * x_weights[x_i]
                if weight.shape[0] != 1 or c == 0:
                    weight[weight_channel, gz, gy, gx] += w_val
                if channel_gains is None:
                    fused[c, gz, gy, gx] += (
                        source[c, source_z + dz_i, source_y + y_i, source_x + x_i]
                        * w_val
                    )
                else:
                    value = (
                        np.float32(
                            source[c, source_z + dz_i, source_y + y_i, source_x + x_i]
                        )
                        * gain
                    )
                    fused[c, gz, gy, gx] += value * w_val


@njit(nogil=True, error_model="numpy")
def _normalize_block(
    fused: np.ndarray,
    weight: np.ndarray,
    output: np.ndarray,
    z_off: int = 0,
    y_off: int = 0,
    x_off: int = 0,
) -> None:
    """Normalize and cast channel rows directly into the output block.

    Parameters
    ----------
    fused : numpy.ndarray
        Float32 accumulation buffer of shape (C, dz, Y, X).
    weight : numpy.ndarray
        Float32 weight buffer of shape (C, dz, Y, X), or a single shared
        channel when all contributing tiles contain every channel.
    output : numpy.ndarray
        Destination CZYX block with float32 or uint16 dtype. Uint16 values clip
        to [0, 65535] and truncate; float32 values retain their range.
        Unsupported pixels become zero.
    z_off, y_off, x_off : int
        Origin of the normalized region in the destination block.

    Returns
    -------
    None
        Writes into `output`, which may alias `fused` for in-place normalization.
    """
    c_dim, dz, y_dim, x_dim = fused.shape
    total = dz * y_dim

    for c in range(c_dim):
        weight_channel = 0 if weight.shape[0] == 1 else c
        for idx in range(total):
            z_i = idx // y_dim
            y_i = idx % y_dim
            for x_i in range(x_dim):
                w_val = weight[weight_channel, z_i, y_i, x_i]
                value = fused[c, z_i, y_i, x_i] / w_val if w_val > 0 else np.float32(0)
                if output.itemsize == 2:
                    value = min(max(value, np.float32(0)), np.float32(65535))
                output[c, z_off + z_i, y_off + y_i, x_off + x_i] = value


def _partition_fusion_block(
    bounds: tuple[int, int, int, int, int, int],
    contributors: Sequence[tuple[int, tuple[int, int, int]]],
    tile_shapes: Sequence[tuple[int, int, int]] | tuple[int, int, int],
) -> list[
    tuple[
        tuple[int, int, int, int, int, int],
        list[tuple[int, tuple[int, int, int]]],
    ]
]:
    """Partition a block wherever its set of contributing tiles changes.

    This preserves the block-oriented memory bound while exposing exclusive tile
    interiors for direct copying. Adjacent X regions with identical contributors
    are merged to avoid unnecessary TensorStore reads and writes.

    Parameters
    ----------
    bounds : tuple[int, int, int, int, int, int]
        Half-open spatial block bounds in ZYX order.
    contributors : Sequence[tuple[int, tuple[int, int, int]]]
        Tile indices and integer canvas ZYX origins intersecting the output block.
    tile_shapes : Sequence[tuple[int, int, int]] | tuple[int, int, int]
        Per-tile spatial dimensions in ZYX order.

    Returns
    -------
    list
        Spatial sub-blocks paired with the tiles contributing to each sub-block.
    """
    z0, z1, y0, y1, x0, x1 = bounds
    common_shape = (
        tuple(int(value) for value in tile_shapes)
        if len(tile_shapes) == 3
        and all(isinstance(value, (int, np.integer)) for value in tile_shapes)
        else None
    )

    def shape_for(tile_index: int) -> tuple[int, int, int]:
        """Read the source spatial dimensions for one contributing tile.

        Parameters
        ----------
        tile_index : int
            Index into the selected processed tile collection.

        Returns
        -------
        tuple[int, int, int]
            Tile dimensions in ZYX order.
        """
        if common_shape is not None:
            return common_shape
        return tuple(int(value) for value in tile_shapes[tile_index])

    cuts = [{z0, z1}, {y0, y1}, {x0, x1}]
    for tile_index, (oz, oy, ox) in contributors:
        tile_z, tile_y, tile_x = shape_for(tile_index)
        for axis_cuts, lo, hi, tile_lo, tile_length in (
            (cuts[0], z0, z1, oz, tile_z),
            (cuts[1], y0, y1, oy, tile_y),
            (cuts[2], x0, x1, ox, tile_x),
        ):
            axis_cuts.add(max(lo, tile_lo))
            axis_cuts.add(min(hi, tile_lo + tile_length))

    z_cuts, y_cuts, x_cuts = (sorted(axis) for axis in cuts)
    regions: list[
        tuple[
            tuple[int, int, int, int, int, int],
            list[tuple[int, tuple[int, int, int]]],
        ]
    ] = []
    for rz0, rz1 in zip(z_cuts, z_cuts[1:]):
        for ry0, ry1 in zip(y_cuts, y_cuts[1:]):
            for rx0, rx1 in zip(x_cuts, x_cuts[1:]):
                region_contributors = [
                    (p, offset)
                    for p, offset in contributors
                    if offset[0] <= rz0
                    and offset[0] + shape_for(p)[0] >= rz1
                    and offset[1] <= ry0
                    and offset[1] + shape_for(p)[1] >= ry1
                    and offset[2] <= rx0
                    and offset[2] + shape_for(p)[2] >= rx1
                ]
                if not region_contributors:
                    continue
                region = (rz0, rz1, ry0, ry1, rx0, rx1)
                if (
                    regions
                    and regions[-1][1] == region_contributors
                    and regions[-1][0][:4] == region[:4]
                    and regions[-1][0][5] == region[4]
                ):
                    previous, previous_contributors = regions[-1]
                    regions[-1] = (
                        (*previous[:5], region[5]),
                        previous_contributors,
                    )
                else:
                    regions.append((region, region_contributors))
    return regions


def _registration_tile_shapes(
    tile_shape_zyx: Sequence[int] | Sequence[Sequence[int]],
    tile_count: int,
) -> np.ndarray:
    """Normalize one common shape or one ZYX shape per registration tile.

    Parameters
    ----------
    tile_shape_zyx : Sequence[int] | Sequence[Sequence[int]]
        Shared or per-tile spatial dimensions in ZYX order.
    tile_count : int
        Number of tiles represented by the shape array.

    Returns
    -------
    np.ndarray
        ZYX dimensions broadcast to one spatial shape per tile.
    """
    shapes = np.asarray(tile_shape_zyx, dtype=np.float64)
    if shapes.shape == (3,):
        shapes = np.broadcast_to(shapes, (tile_count, 3))
    return shapes


def _select_nearest_registration_pairs(
    positions_zyx: Sequence[Sequence[float]],
    tile_shape_zyx: Sequence[int] | Sequence[Sequence[int]],
    pixel_size_zyx: Sequence[float],
    *,
    is_2d: bool = False,
) -> list[tuple[int, int]]:
    """Select a connected nearest-overlap graph for pairwise registration.

    All geometric overlaps are found first. Within each connected component,
    the distance cutoff is the shortest one that still connects that component.
    Every overlap at or below that cutoff is retained, preserving useful cycles
    between face-adjacent tiles while dropping diagonal and farther redundant
    overlaps.

    Parameters
    ----------
    positions_zyx
        Image-placement tile positions in physical ZYX coordinates.
    tile_shape_zyx
        Common ZYX shape or one ZYX shape per tile, in voxels.
    pixel_size_zyx
        ZYX voxel spacing in physical units.
    is_2d
        Ignore physical Z displacement for projection data.

    Returns
    -------
    list[tuple[int, int]]
        Local position-index pairs to register.
    """
    positions = np.asarray(positions_zyx, dtype=np.float64)
    pixel_size = np.asarray(pixel_size_zyx, dtype=np.float64)
    tile_shapes = _registration_tile_shapes(tile_shape_zyx, len(positions))

    physical_shapes = tile_shapes * pixel_size
    edges: list[tuple[float, int, int]] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            delta = positions[j] - positions[i]
            overlap = np.logical_and(
                positions[i] < positions[j] + physical_shapes[j],
                positions[j] < positions[i] + physical_shapes[i],
            )
            if is_2d:
                overlap[0] = True
                delta = delta.copy()
                delta[0] = 0.0
            if np.all(overlap):
                pair_scale = np.maximum(physical_shapes[i], physical_shapes[j])
                distance = float(np.linalg.norm(delta / pair_scale))
                edges.append((distance, i, j))

    if not edges:
        return []

    parent = list(range(len(positions)))

    def find(index: int) -> int:
        """Find a tile component representative with path compression.

        Parameters
        ----------
        index : int
            Tile or registration-link index used by this calculation.

        Returns
        -------
        int
            Representative tile index of the connected component.
        """
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> bool:
        """Merge the connected components containing two tiles.

        Parameters
        ----------
        left : int
            First tile index whose connected component is merged.
        right : int
            Second tile index whose connected component is merged.

        Returns
        -------
        bool
            Updates the component parents in place.
        """
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return False
        parent[right_root] = left_root
        return True

    for _, i, j in edges:
        union(i, j)
    component_by_node = [find(index) for index in range(len(positions))]

    edges_by_component: dict[int, list[tuple[float, int, int]]] = {}
    nodes_by_component: dict[int, set[int]] = {}
    for edge in edges:
        _, i, j = edge
        component = component_by_node[i]
        edges_by_component.setdefault(component, []).append(edge)
        nodes_by_component.setdefault(component, set()).update((i, j))

    selected: list[tuple[int, int]] = []
    for component, component_edges in edges_by_component.items():
        component_edges.sort()
        component_nodes = nodes_by_component[component]
        local_parent = {node: node for node in component_nodes}

        def local_find(index: int) -> int:
            """Find a tile representative within the current overlap component.

            Parameters
            ----------
            index : int
                Tile or registration-link index used by this calculation.

            Returns
            -------
            int
                Representative tile index within the overlap component.
            """
            while local_parent[index] != index:
                local_parent[index] = local_parent[local_parent[index]]
                index = local_parent[index]
            return index

        merges_needed = len(component_nodes) - 1
        merges = 0
        cutoff = component_edges[-1][0]
        for distance, i, j in component_edges:
            i_root = local_find(i)
            j_root = local_find(j)
            if i_root != j_root:
                local_parent[j_root] = i_root
                merges += 1
            if merges == merges_needed:
                cutoff = distance
                break

        tolerance = max(1e-12, cutoff * 1e-6)
        selected.extend(
            (i, j)
            for distance, i, j in component_edges
            if distance <= cutoff + tolerance
        )

    selected.sort()
    return selected


def _fit_depth_gains(
    level_count: int, measurements: Sequence[tuple[int, int, float]]
) -> np.ndarray:
    """Fit multiplicative depth gains from registered overlap measurements.

    Parameters
    ----------
    level_count : int
        Number of acquisition depth levels, including unmeasured levels.
    measurements : sequence of tuple[int, int, float]
        Fixed depth, moving depth, and log intensity ratio for each patch.
        Repeated measurements are reduced to their median for each depth pair.

    Returns
    -------
    numpy.ndarray
        One gain per depth. The first depth in each connected component has
        unit gain; unmeasured depths also retain unit gain.
    """
    grouped = {}
    for left, right, log_ratio in measurements:
        if left > right:
            left, right, log_ratio = right, left, -log_ratio
        grouped.setdefault((left, right), []).append(log_ratio)
    links = [
        {
            "i": left,
            "j": right,
            "t": np.asarray((np.median(values), 0, 0)),
            "w": math.sqrt(len(values)),
        }
        for (left, right), values in grouped.items()
    ]
    adjacency = [set() for _ in range(level_count)]
    for left, right in grouped:
        adjacency[left].add(right)
        adjacency[right].add(left)
    anchors = []
    visited = set()
    for level in range(level_count):
        if level in visited:
            continue
        anchors.append(level)
        frontier = [level]
        visited.add(level)
        while frontier:
            unseen = adjacency[frontier.pop()] - visited
            visited.update(unseen)
            frontier.extend(unseen)
    shifts = TileFusion._solve_global(links, level_count, anchors)
    return np.exp(shifts[:, 0])


def _registration_sampling(
    overlap_shape: Sequence[int],
    downsample_factors: Sequence[int],
    ssim_window: int,
    is_2d: bool,
) -> tuple[int, ...]:
    """Choose downsampling that retains enough samples for overlap scoring.

    Parameters
    ----------
    overlap_shape : sequence of int
        Overlap dimensions in ZYX order.
    downsample_factors : sequence of int
        Requested downsampling along ZYX.
    ssim_window : int
        Requested structural similarity window width.
    is_2d : bool
        Whether scoring excludes the singleton Z axis.

    Returns
    -------
    tuple of int
        Effective ZYX downsampling factors for this overlap.
    """
    active_shape = overlap_shape[1:] if is_2d else overlap_shape
    window = min(ssim_window, min(active_shape))
    window -= 1 - window % 2
    return tuple(
        max(1, min(factor, length // max(3, window)))
        for factor, length in zip(downsample_factors, overlap_shape)
    )


def _infer_registration_limits(
    positions_zyx: Sequence[Sequence[float]],
    tile_shapes_zyx: Sequence[Sequence[int]],
    pixel_size_zyx: Sequence[float],
    downsample_factors: Sequence[int],
    angle_deg: float | None,
    *,
    is_2d: bool = False,
) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Infer search envelopes from physical overlaps and OPM scan geometry.

    Stage-placement uncertainty is budgeted at ten percent of the smallest
    face overlap on each tiled axis, with two registration sampling bins as
    a floor. Untiled axes use ten percent of their tile extent. Depth pairs
    additionally allow the oblique-plane footprint displacement dz*cot(theta)
    along image Y. This is a search allowance, not a shear of tile origins.
    Each pair retains at least half its overlap along every active axis.

    Parameters
    ----------
    positions_zyx : sequence of sequence of float
        Nominal tile origins in physical ZYX coordinates.
    tile_shapes_zyx : sequence of sequence of int
        Tile dimensions in ZYX pixels.
    pixel_size_zyx : sequence of float
        Physical voxel spacing in the same units as the origins.
    downsample_factors : sequence of int
        Requested ZYX registration downsampling factors.
    angle_deg : float or None
        Oblique scan angle in degrees, or None when unavailable.
    is_2d : bool, default = False
        Whether tiles represent planar images.

    Returns
    -------
    dict
        Overlapping tile pairs mapped to maximum ZYX corrections in pixels.
    """
    positions = np.asarray(positions_zyx, dtype=np.float64)
    if len(positions) == 0:
        return {}
    shapes = _registration_tile_shapes(tile_shapes_zyx, len(positions))
    spacing = np.asarray(pixel_size_zyx, dtype=np.float64)
    sampling = np.asarray(downsample_factors, dtype=np.int64)
    if sampling.shape != (3,) or np.any(sampling < 1):
        raise ValueError("downsample_factors must contain three positive values")
    pairs = _select_registration_pairs(positions, shapes, spacing, is_2d=is_2d)
    nearest_pairs = set(
        _select_nearest_registration_pairs(
            positions,
            shapes,
            spacing,
            is_2d=is_2d,
        )
    )
    overlaps = {}
    face_overlaps: list[list[float]] = [[], [], []]
    for left, right in pairs:
        delta = np.abs(positions[right] - positions[left]) / spacing
        overlap = (
            np.minimum(
                positions[left] + shapes[left] * spacing,
                positions[right] + shapes[right] * spacing,
            )
            - np.maximum(positions[left], positions[right])
        ) / spacing
        if is_2d:
            delta[0] = 0
            overlap[0] = 1
        overlaps[left, right] = overlap
        # The dominant normalized separation identifies the tile face. Do
        # not let a small diagonal intersection set another axis's budget.
        axis = int(np.argmax(delta / np.minimum(shapes[left], shapes[right])))
        if (left, right) in nearest_pairs and delta[axis] > 0:
            face_overlaps[axis].append(float(overlap[axis]))
    extent = np.min(shapes, axis=0)
    base = np.maximum(
        2 * sampling,
        np.ceil(
            0.1
            * np.asarray(
                [
                    min(values) if values else extent[axis]
                    for axis, values in enumerate(face_overlaps)
                ]
            )
        ),
    )
    cotangent = (
        1.0 / math.tan(math.radians(angle_deg))
        if angle_deg is not None and 0 < angle_deg < 90 and not is_2d
        else 0.0
    )
    limits = {}
    for pair, overlap in overlaps.items():
        left, right = pair
        budget = base.copy()
        budget[1] += (
            abs(positions[right, 0] - positions[left, 0]) * cotangent / spacing[1]
        )
        budget = np.minimum(np.ceil(budget), np.floor(overlap / 2))
        if is_2d:
            budget[0] = 0
        limits[pair] = tuple(int(value) for value in budget)
    return limits


def _select_registration_pairs(
    positions_zyx: Sequence[Sequence[float]],
    tile_shape_zyx: Sequence[int] | Sequence[Sequence[int]],
    pixel_size_zyx: Sequence[float],
    *,
    is_2d: bool = False,
) -> list[tuple[int, int]]:
    """Return every pair whose physical tile bounding boxes overlap.

    Parameters
    ----------
    positions_zyx : Sequence[Sequence[float]]
        Physical tile origins in ZYX order, in micrometers.
    tile_shape_zyx : Sequence[int] | Sequence[Sequence[int]]
        Shared or per-tile spatial dimensions in ZYX order.
    pixel_size_zyx : Sequence[float]
        Physical voxel spacing in ZYX order, in micrometers.
    is_2d : bool
        Use projection-plane geometry rather than volumetric overlap.

    Returns
    -------
    list[tuple[int, int]]
        Neighbor tile pairs selected from stage geometry and physical overlap.
    """
    positions = np.asarray(positions_zyx, dtype=np.float64)
    pixel_size = np.asarray(pixel_size_zyx, dtype=np.float64)
    tile_shapes = _registration_tile_shapes(tile_shape_zyx, len(positions))

    physical_shapes = tile_shapes * pixel_size
    selected = []
    for left in range(len(positions)):
        for right in range(left + 1, len(positions)):
            overlap = np.logical_and(
                positions[left] < positions[right] + physical_shapes[right],
                positions[right] < positions[left] + physical_shapes[left],
            )
            if is_2d:
                overlap[0] = True
            if np.all(overlap):
                selected.append((left, right))
    return selected


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
        Source acquisition, processed collection, or processing output
        directory. The code discovers deskewed or projection data there;
        deconvolved products are fallbacks when a non-deconvolved product is
        unavailable.
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
    max_workers : int or None, default=None
        Maximum number of fusion blocks rendered concurrently. By default,
        uses up to eight physical CPU cores so blocks can retain enough Z depth
        for efficient source-chunk reads.
    debug : bool, default=False
        If True, emits debug logs.
    channel_to_use : int, default=0
        Channel index used for registration.
    reverse_stage_y : bool, default=True
        Reverse stage-derived Y for tile placement without flipping tile pixels.
    reverse_stage_z : bool or None, default=None
        Reverse physical stage Z into laboratory Z for tile placement. ``None``
        applies the transform unless processed coordinates were already flipped.
    multiscale_downsample : {"stride", "block_mean"}, default="stride"
        Method for multiscale downsampling.
    fusion_ram_fraction : float, default=0.4
        Fraction of currently available host RAM available to fusion buffers.
    max_in_flight_writes : int, default=2
        Maximum number of TensorStore writes retained before applying backpressure.
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
        max_workers: int | None = None,
        debug: bool = False,
        channel_to_use: int = 0,
        multiscale_downsample: str = "stride",
        fusion_ram_fraction: float = 0.4,
        max_in_flight_writes: int = 2,
        chunk_shape_yx: tuple[int, int] = (1024, 1024),
        optimization_rel_threshold: float = 0.5,
        optimization_abs_threshold: float = 1.5,
        max_registration_shift_zyx: tuple[int, int, int] | None = None,
        reverse_stage_y: bool = True,
        reverse_stage_z: bool | None = None,
        roi_selection: PhysicalRoi | None = None,
        normalize_depth_intensity: bool = True,
    ) -> None:
        """Initialize registration and fusion for a processed acquisition.

        Parameters
        ----------
        root_path
            Source acquisition, processed collection, or processing output
            directory used to discover processed data.
        blend_pixels
            Feathering width in ZYX voxels.
        downsample_factors
            Registration downsampling factors in ZYX order.
        ssim_window
            Structural-similarity window width.
        threshold
            Minimum accepted pairwise registration score.
        multiscale_factors
            Downsampling factors for pyramid levels.
        resolution_multiples
            NGFF spatial scale multiples for pyramid levels.
        max_workers
            Maximum number of fusion blocks rendered concurrently.
        debug
            Whether to emit diagnostic messages.
        channel_to_use
            Channel index used for registration.
        multiscale_downsample
            Pyramid downsampling method.
        fusion_ram_fraction
            Fraction of available host RAM allocated to fusion buffers.
        max_in_flight_writes
            Maximum number of pending TensorStore writes.
        chunk_shape_yx
            Spatial output chunk shape.
        optimization_rel_threshold
            Relative residual threshold for registration outliers.
        optimization_abs_threshold
            Absolute residual threshold in effective registration sampling bins.
        max_registration_shift_zyx
            Maximum accepted registration correction in ZYX voxels. None
            infers a separate limit for each pair from stage spacing, tile
            overlap, voxel size, scan angle, and registration sampling.
        reverse_stage_y
            Whether to reverse stage-derived image-Y placement. Tile pixel
            arrays are not modified.
        reverse_stage_z
            Whether to convert physical stage Z into the opposite laboratory-Z
            placement coordinate. ``None`` detects whether processing already
            transformed the stored positions. Tile pixels and scan ordering are
            never modified.
        roi_selection
            Optional physical YX crop. Tile selection ignores Z so every stage-Z
            level intersecting the rectangle is retained.
        normalize_depth_intensity
            Match depth-layer intensities using image-registered overlaps at
            the same stage XY. All XY tiles at one depth share the same gain.

        Returns
        -------
        None
            No value is returned.
        """
        (
            self.output_dir,
            self.data,
            self.acquisition_name,
            source_path,
        ) = resolve_fusion_input(root_path)

        if max_workers is None:
            max_workers = _default_fusion_workers()
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._max_workers = int(max_workers)
        self._source_cache_bytes = min(
            4 * 1024**3,
            max(1, int(psutil.virtual_memory().available * 0.1)),
        )
        self._source_context = ts.Context(
            {
                "cache_pool": {
                    "total_bytes_limit": self._source_cache_bytes,
                },
                "data_copy_concurrency": {
                    "limit": min(8, self._max_workers),
                },
            }
        )
        collection = open_position_collection(self.data)
        self.processing_state = ProcessingState.read(
            processing_state_path(self.output_dir, self.acquisition_name)
        )
        self.processing_state.run(self.data)
        state_source_path = Path(self.processing_state.document["source"]["path"])
        if source_path is not None and source_path != state_source_path:
            raise ValueError(
                "Processing state source does not match the fusion acquisition"
            )
        self.root = state_source_path
        self.acquisition = inspect_acquisition(self.root)
        source_open_futures = [
            ts.open(
                array.spec(retain_context=False),
                context=self._source_context,
                recheck_cached_data=False,
            )
            for array in collection.arrays
        ]
        self.position_arrays = tuple(future.result() for future in source_open_futures)
        self.output_dtype = np.dtype(self.position_arrays[0].dtype.numpy_dtype)
        input_chunks = self.position_arrays[0].chunk_layout.read_chunk.shape
        self._input_chunk_zyx = tuple(int(value) for value in input_chunks[-3:])
        roi_series = self.processing_state.roi_series(self.data)
        self._variable_roi_tiles = bool(roi_series)
        self.reverse_stage_y = bool(reverse_stage_y)
        if reverse_stage_z is None:
            reverse_stage_z = not self.acquisition.stage_axis_flips_xyz[2]
        self.reverse_stage_z = bool(reverse_stage_z)
        self._pixel_size = tuple(float(value) for value in collection.voxel_size_um)
        self.roi_selection = roi_selection
        if self._variable_roi_tiles:
            records = roi_series
            time_count = max(int(record["time_index"]) for record in records) + 1
            self._tile_positions = []
            self._tile_shapes = []
            self._tile_time_indices = []
            self._tile_source_position_indices = []
            self._roi_tile_records = tuple(records)
            tiles_by_time: list[list[int]] = [[] for _ in range(time_count)]
            for tile_index, (record, array, origin) in enumerate(
                zip(
                    records,
                    self.position_arrays,
                    collection.spatial_origins_zyx_um,
                )
            ):
                time_index = int(record["time_index"])
                position_index = int(record["position_index"])
                initial_origin = tuple(float(value) for value in origin)
                array_shape = tuple(int(value) for value in array.shape)
                self._tile_positions.append(initial_origin)
                self._tile_shapes.append(array_shape[-3:])
                self._tile_time_indices.append(time_index)
                self._tile_source_position_indices.append(position_index)
                tiles_by_time[time_index].append(tile_index)
            self._tiles_by_time = tuple(tuple(group) for group in tiles_by_time)
            self.time_dim = time_count
            self.position_dim = max(len(group) for group in tiles_by_time)
            self.channels = int(self.position_arrays[0].shape[1])
            self.z_dim = max(shape[0] for shape in self._tile_shapes)
            self.y_dim = max(shape[1] for shape in self._tile_shapes)
            self.x_dim = max(shape[2] for shape in self._tile_shapes)
            self._source_position_indices = tuple(
                dict.fromkeys(self._tile_source_position_indices)
            )
            # These origins already include the optimized full-fusion placement
            # plus the scan/X read and post-deskew crop offsets. They seed a new
            # registration against the processed (normally deconvolved) ROI tiles.
            self._reuse_registered_roi_placements = False
        else:
            collection_shape = collection.shape
            stage_positions = [
                position
                for _ in range(collection_shape[0])
                for position in collection.stage_positions_zxy
            ]
            self._tile_positions = [
                tuple(position)
                for position in stage_positions_to_image_coordinates(
                    stage_positions,
                    reverse_y=self.reverse_stage_y,
                    reverse_z=self.reverse_stage_z,
                    opm_angle_deg=float(self.acquisition.angle_deg),
                )
            ]
            selected_positions: tuple[int, ...] = tuple(range(collection_shape[1]))
            if roi_selection is not None:
                selected_positions = roi_selection.position_indices
                if not selected_positions:
                    raise ValueError(
                        "The ROI does not intersect any processed positions"
                    )
                self.position_arrays = tuple(
                    self.position_arrays[index] for index in selected_positions
                )
                self._tile_positions = [
                    roi_selection.registered_tile_origin_zyx_um(time_index, index)
                    for time_index in range(int(collection_shape[0]))
                    for index in selected_positions
                ]
                self._reuse_registered_roi_placements = True
            else:
                self._reuse_registered_roi_placements = False
            self._source_position_indices = selected_positions
            (
                self.time_dim,
                _original_position_dim,
                self.channels,
                self.z_dim,
                self.y_dim,
                self.x_dim,
            ) = collection_shape
            self.position_dim = len(self._source_position_indices)
            common_shape = (
                int(self.z_dim),
                int(self.y_dim),
                int(self.x_dim),
            )
            self._tile_shapes = [common_shape] * len(self._tile_positions)
            self._tiles_by_time = tuple(
                tuple(
                    range(
                        time_index * int(self.position_dim),
                        (time_index + 1) * int(self.position_dim),
                    )
                )
                for time_index in range(int(self.time_dim))
            )
            self._tile_time_indices = [
                time_index
                for time_index in range(int(self.time_dim))
                for _ in range(int(self.position_dim))
            ]
            self._tile_source_position_indices = [
                position_index
                for _ in range(int(self.time_dim))
                for position_index in self._source_position_indices
            ]

        self.downsample_factors = tuple(int(x) for x in downsample_factors)
        self.ssim_window = int(ssim_window)
        self.threshold = float(threshold)
        self.multiscale_factors = tuple(int(x) for x in multiscale_factors)
        if any(value < 1 for value in self.multiscale_factors):
            raise ValueError("multiscale_factors must contain positive integers")
        self.resolution_multiples: list[tuple[int, int, int]] = [
            tuple(r) if hasattr(r, "__len__") else (int(r), int(r), int(r))
            for r in resolution_multiples
        ]
        self._debug = bool(debug)
        self._blend_pixels = tuple(int(x) for x in blend_pixels)
        self.channel_to_use = int(channel_to_use)
        self.normalize_depth_intensity = bool(normalize_depth_intensity)
        self._depth_intensity_gains = np.ones(
            (len(self._tile_positions), int(self.channels)), dtype=np.float32
        )
        self._depth_intensity_report = None

        if multiscale_downsample not in ("stride", "block_mean"):
            raise ValueError('multiscale_downsample must be "stride" or "block_mean".')
        self.multiscale_downsample = multiscale_downsample
        if not 0.0 < fusion_ram_fraction <= 1.0:
            raise ValueError("fusion_ram_fraction must satisfy 0 < value <= 1")
        if max_in_flight_writes < 1:
            raise ValueError("max_in_flight_writes must be at least 1")
        self.fusion_ram_fraction = float(fusion_ram_fraction)
        self.max_in_flight_writes = int(max_in_flight_writes)
        if len(chunk_shape_yx) != 2 or any(value < 1 for value in chunk_shape_yx):
            raise ValueError("chunk_shape_yx must contain two positive values")
        if optimization_rel_threshold < 0 or optimization_abs_threshold < 0:
            raise ValueError("optimization thresholds must be nonnegative")
        self.chunk_y, self.chunk_x = (int(value) for value in chunk_shape_yx)
        self.optimization_rel_threshold = float(optimization_rel_threshold)
        self.optimization_abs_threshold = float(optimization_abs_threshold)
        self._automatic_registration_limits = max_registration_shift_zyx is None
        self._pair_registration_limits = {}
        self._is_2d = all(shape[0] == 1 for shape in self._tile_shapes)
        if self._automatic_registration_limits:
            for tile_indices in self._tiles_by_time:
                local_limits = _infer_registration_limits(
                    [self._tile_positions[index] for index in tile_indices],
                    [self._tile_shapes[index] for index in tile_indices],
                    self._pixel_size,
                    self.downsample_factors,
                    self.acquisition.angle_deg,
                    is_2d=self._is_2d,
                )
                self._pair_registration_limits.update(
                    {
                        (tile_indices[left], tile_indices[right]): limits
                        for (left, right), limits in local_limits.items()
                    }
                )
            max_registration_shift_zyx = tuple(
                max(
                    (
                        limits[axis]
                        for limits in self._pair_registration_limits.values()
                    ),
                    default=0,
                )
                for axis in range(3)
            )
            print(
                "Automatic registration limits: per-pair geometry; "
                f"maximum ZYX corrections {max_registration_shift_zyx} pixels."
            )
        if len(max_registration_shift_zyx) != 3 or any(
            value < 0 for value in max_registration_shift_zyx
        ):
            raise ValueError(
                "max_registration_shift_zyx must contain three nonnegative values"
            )
        self.max_registration_shift_zyx = tuple(
            int(value) for value in max_registration_shift_zyx
        )
        if self._variable_roi_tiles:
            self._tile_support_zy = self._build_variable_roi_support_masks()
        else:
            self._deskew_support_zy = self._build_deskew_support_mask()
            self._tile_support_zy = [self._deskew_support_zy] * len(self._tile_shapes)
        if not 0 <= self.channel_to_use < int(self.channels):
            raise ValueError(
                "channel_to_use must be between 0 and "
                f"{int(self.channels) - 1}; got {self.channel_to_use}"
            )

        self._update_profiles()
        self._tile_profiles = [
            (
                self._make_1d_profile(shape[0], self._blend_pixels[0]),
                self._make_1d_profile(shape[1], self._blend_pixels[1]),
                self._make_1d_profile(shape[2], self._blend_pixels[2]),
            )
            for shape in self._tile_shapes
        ]
        self._tile_channel_nonzero: dict[tuple[int, int], bool] = {}
        self._load_zero_channel_metadata()

        self.pairwise_metrics: dict[tuple[int, int], tuple[int, int, int, float]] = {}
        self.global_offsets: np.ndarray | None = None

        self.offset_um: tuple[float, float, float] | None = None
        self.unpadded_shape: tuple[int, int, int] | None = None
        self.padded_shape: tuple[int, int, int] | None = None

        self.fused_ts: ts.TensorStore | None = None
        self.write_block_shape: list[int] | None = None

    @property
    def debug(self) -> bool:
        """Get the debug flag.

        Returns
        -------
        debug : bool
            True if debug logging is enabled.
        """
        return self._debug

    @debug.setter
    def debug(self, flag: bool) -> None:
        """Set the debug flag.

        Parameters
        ----------
        flag : bool
            True to enable debug logging.

        Returns
        -------
        None
        """
        self._debug = bool(flag)

    def _build_deskew_support_mask(self) -> np.ndarray:
        """Generate exact deskew support from metadata without reading pixels.

        Returns
        -------
        np.ndarray
            Boolean ZY support mask shared by the full-size tiles.
        """
        if self._is_2d:
            # Projection processing does not deskew or pad its image plane.
            return np.ones((1, int(self.y_dim)), dtype=bool)
        raw_sizes = self.acquisition.index_sizes
        raw_scan_count = int(raw_sizes["z"])
        raw_camera_y = int(raw_sizes["y"])
        scan_count = (
            raw_scan_count
            - int(
                self.acquisition.excess_scan_start_positions
                or self.acquisition.excess_scan_positions
            )
            - int(self.acquisition.excess_scan_end_positions)
        )

        angle_deg = float(self.acquisition.angle_deg)
        scan_step_um = float(self.acquisition.scan_axis_step_um)
        raw_pixel_size_um = float(self.acquisition.pixel_size_um)
        reconstruction = self.processing_state.run(self.data).get("reconstruction")
        if reconstruction is not None:
            scan_count, raw_camera_y, _ = map(int, reconstruction["shape_syx"])
            scan_step_um = float(reconstruction["scan_axis_step_um"])
        z_downsample = int(round(float(self._pixel_size[0]) / raw_pixel_size_um))

        support = (
            orthogonal_deskew(
                np.full(
                    (scan_count, raw_camera_y, 1),
                    1024.0,
                    dtype=np.float32,
                ),
                theta=angle_deg,
                distance=scan_step_um,
                pixel_size=raw_pixel_size_um,
                divisible_by=1,
                downsample_factor=z_downsample,
            )[..., 0]
            != 0
        )
        padded_support = np.zeros(
            (int(self.z_dim), int(self.y_dim)),
            dtype=bool,
        )
        padded_support[:, : support.shape[1]] = support
        return padded_support

    def _build_variable_roi_support_masks(self) -> list[np.ndarray]:
        """Build the cropped deskew-validity mask for every variable ROI tile.

        Returns
        -------
        list[np.ndarray]
            Boolean ZY masks in cropped tile series order.
        """
        angle_deg = float(self.acquisition.angle_deg)
        scan_step_um = float(self.acquisition.scan_axis_step_um)
        raw_pixel_size_um = float(self.acquisition.pixel_size_um)
        z_downsample = int(round(float(self._pixel_size[0]) / raw_pixel_size_um))

        masks: list[np.ndarray] = []
        for record in self._roi_tile_records:
            scan_count, camera_y, _camera_x = map(int, record["skewed_shape_syx"])
            crop_y0, crop_y1, _crop_x0, _crop_x1 = map(int, record["deskew_crop_yx"])
            support = (
                orthogonal_deskew(
                    np.full(
                        (scan_count, camera_y, 1),
                        1024.0,
                        dtype=np.float32,
                    ),
                    theta=angle_deg,
                    distance=scan_step_um,
                    pixel_size=raw_pixel_size_um,
                    divisible_by=4,
                    downsample_factor=z_downsample,
                )[..., 0]
                != 0
            )
            support = support[:, crop_y0:crop_y1]
            masks.append(np.ascontiguousarray(support))
        return masks

    def _update_profiles(self) -> None:
        """Recompute 1D feather profiles from blend_pixels and current data shape.

        Returns
        -------
        None
            Populates `z_profile`, `y_profile`, and `x_profile` attributes.
        """
        bz, by, bx = self._blend_pixels
        self.z_profile = self._make_1d_profile(int(self.z_dim), int(bz))
        self.y_profile = self._make_1d_profile(int(self.y_dim), int(by))
        self.x_profile = self._make_1d_profile(int(self.x_dim), int(bx))

    def _load_zero_channel_metadata(self) -> None:
        """Load completed and zero-channel decisions from processing state."""
        completed = self.processing_state.completed_tiles(self.data)
        expected = set(zip(self._tile_time_indices, self._tile_source_position_indices))
        missing = expected - completed
        if missing:
            raise ValueError(
                "Fusion input is incomplete; processing state is missing "
                f"{len(missing)} tile checkpoints"
            )
        zero_keys = self.processing_state.zero_channels(self.data)
        for tile_index, (time_index, position_index) in enumerate(
            zip(self._tile_time_indices, self._tile_source_position_indices)
        ):
            for channel_index in range(int(self.channels)):
                key = (int(time_index), int(position_index), channel_index)
                self._tile_channel_nonzero[(tile_index, channel_index)] = (
                    key not in zero_keys
                )
        zero_count = len(zero_keys)
        if zero_count:
            print(
                f"Excluding {zero_count} zero tile/channel volumes from fusion weights."
            )

    def _channel_has_signal(self, tile_index: int, channel_index: int) -> bool:
        """Return processing's durable tile/channel signal decision.

        Parameters
        ----------
        tile_index : int
            Index into the selected processed tile collection.
        channel_index : int
            Acquisition channel index.

        Returns
        -------
        bool
            True unless this tile/channel was checkpointed as empty during processing.
        """
        return self._tile_channel_nonzero[(tile_index, channel_index)]

    @staticmethod
    def _make_1d_profile(length: int, blend: int) -> np.ndarray:
        """Create a 1D feather profile with linear ramps at both ends.

        Parameters
        ----------
        length : int
            Axis length in voxels.
        blend : int
            Ramp width (voxels) at each end.

        Returns
        -------
        prof : numpy.ndarray
            Float32 array of shape (length,). Values are in (0, 1] with
            pixel-centered ramped edges; for very small `length` (<= 2) or
            `blend` <= 0, returns ones.
        """
        length = int(length)
        blend = int(blend)

        if length <= 2 or blend <= 0:
            return np.ones(length, dtype=np.float32)

        blend = min(blend, length)
        prof = np.ones(length, dtype=np.float32)

        if blend >= length:
            indices = np.arange(length, dtype=np.float32)
            tent = np.minimum(indices + 1.0, length - indices)
            tent /= tent.max()
            return tent.astype(np.float32, copy=False)

        ramp = np.arange(1, blend + 1, dtype=np.float32) / np.float32(blend)
        prof[:blend] = ramp
        prof[-blend:] = ramp[::-1]
        return prof

    def _read_tile_volume(
        self,
        tile_idx: int,
        ch_sel: int | slice,
        z_slice: slice,
        y_slice: slice,
        x_slice: slice,
        dtype: Any = np.float32,
    ) -> np.ndarray:
        """Read a tile subvolume using a global flattened tile index.

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
        dtype : numpy dtype or None, default=numpy.float32
            Requested output dtype. ``None`` preserves the stored dtype.

        Returns
        -------
        arr : numpy.ndarray
            Float32 array in channel-first form:
            - 3D: (Z, Y, X) for int channel, or (C, Z, Y, X) for slice channels.
            - 2D: normalized to (C, 1, Y, X).
        """
        if getattr(self, "_variable_roi_tiles", False):
            t_idx = 0
            pos_idx = tile_idx
        else:
            n_pos = int(self.position_dim)
            t_idx = tile_idx // n_pos
            pos_idx = tile_idx % n_pos

        if self._is_2d:
            arr = (
                self.position_arrays[pos_idx][t_idx, ch_sel, 0, y_slice, x_slice]
                .read()
                .result()
            )
            arr = arr.reshape((-1, 1, *arr.shape[-2:]))
            return arr if dtype is None else arr.astype(dtype, copy=False)

        arr = (
            self.position_arrays[pos_idx][t_idx, ch_sel, z_slice, y_slice, x_slice]
            .read()
            .result()
        )
        return arr if dtype is None else arr.astype(dtype, copy=False)

    def _read_registration_patch(
        self,
        tile_idx: int,
        channel_index: int,
        bounds_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    ) -> np.ndarray:
        """Read one overlap through TensorStore's shared native chunk cache.

        Parameters
        ----------
        tile_idx : int
            Index into the selected processed tile collection.
        channel_index : int
            Acquisition channel index.
        bounds_zyx : tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
            Half-open tile-local registration bounds in ZYX pixels.

        Returns
        -------
        np.ndarray
            Tile-local ZYX overlap patch read for the selected registration channel.
        """
        return self._read_tile_volume(
            tile_idx,
            channel_index,
            slice(*bounds_zyx[0]),
            slice(*bounds_zyx[1]),
            slice(*bounds_zyx[2]),
            dtype=None,
        )

    @staticmethod
    def register_and_score(
        g1: Any,
        g2: Any,
        win_size: int,
        max_shift: Sequence[float] | None = None,
        allow_bounded_peak: bool = True,
        registration_axis: int | None = None,
    ) -> tuple[tuple[float, float, float], float]:
        """Register `g2` to `g1` and compute an SSIM score.

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
            Maximum SSIM window size. Thin aligned overlaps use the largest
            odd window that fits, with at least three voxels per axis.
        max_shift : sequence[float] or None
            Maximum permitted shift in downsampled patch coordinates. The
            upstream fully disambiguated result is used when it lies inside
            these limits; otherwise the strongest bounded phase peak is used.
        allow_bounded_peak : bool, default=True
            Whether an out-of-range upstream result may be replaced by the
            strongest in-range periodic peak. Thin-Z overlaps disable this
            because their bounded alternatives are underdetermined.
        registration_axis : int or None, default=None
            Axis normal to the physical tile boundary. When provided, an
            edge-based phase-correlation candidate is scored alongside the
            intensity candidate to suppress periodic crop-boundary bias.

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
        disambiguate = True
        if max_shift is not None:
            limits = xp.asarray(max_shift, dtype=xp.float32)
            disambiguate = not all(
                2.0 * float(limit) < float(length)
                for limit, length in zip(max_shift, arr1.shape)
            )

        def correlate(fixed: Any, moving: Any) -> Any:
            """Calculate phase correlation with the selected CPU or CUDA backend.

            Parameters
            ----------
            fixed : Any
                Reference overlap patch used for pairwise registration.
            moving : Any
                Neighbor overlap patch to align with the reference.

            Returns
            -------
            Any
                Measured phase-correlation shift in registration pixels.
            """
            shift, _, _ = phase_cross_correlation(
                fixed,
                moving,
                disambiguate=disambiguate,
                normalization="phase",
                upsample_factor=10,
                overlap_ratio=0.5,
            )
            result = xp.asarray(shift, dtype=xp.float32)
            if (
                not disambiguate
                and max_shift is not None
                and bool(xp.any(xp.abs(result) >= limits * 0.5))
            ):
                shift, _, _ = phase_cross_correlation(
                    fixed,
                    moving,
                    disambiguate=True,
                    normalization="phase",
                    upsample_factor=10,
                    overlap_ratio=0.5,
                )
                result = xp.asarray(shift, dtype=xp.float32)
            return result

        def score_shift(candidate: Any) -> float:
            """Evaluate overlap similarity for a candidate registration correction.

            Parameters
            ----------
            candidate : Any
                Candidate ZYX registration correction in pixels.

            Returns
            -------
            float
                Structural similarity of the translated overlapping patches.
            """
            try:
                fixed_view, moving_view = _aligned_registration_views(
                    arr1,
                    arr2,
                    candidate,
                )
                # Stage-depth overlaps can be thinner than the configured
                # window even without downsampling. Score the valid aligned
                # support with the largest odd window that fits it.
                effective_window = min(win_size, min(fixed_view.shape))
                effective_window -= 1 - effective_window % 2
                if effective_window < 3:
                    return -math.inf
                return _ssim(fixed_view, moving_view, win_size=effective_window)
            except ValueError:
                return -math.inf

        shift_apply = correlate(arr1, arr2)
        if (
            max_shift is not None
            and allow_bounded_peak
            and bool(xp.any(xp.abs(shift_apply) > limits))
        ):
            shift_apply = _bounded_phase_correlation_peak(
                arr1,
                arr2,
                max_shift,
            )

        score = score_shift(shift_apply)
        if registration_axis is not None:
            axis = int(registration_axis)
            if not 0 <= axis < arr1.ndim:
                raise ValueError("registration_axis is outside the patch rank")
            edge_shift = correlate(
                sobel_filter(arr1, axis=axis),
                sobel_filter(arr2, axis=axis),
            )
            edge_is_allowed = max_shift is None or not bool(
                xp.any(xp.abs(edge_shift) > limits)
            )
            if edge_is_allowed and not bool(
                xp.all(xp.rint(edge_shift) == xp.rint(shift_apply))
            ):
                edge_score = score_shift(edge_shift)
                if edge_score > score:
                    shift_apply = edge_shift
                    score = edge_score

        if arr1.ndim == 2 and len(shift_apply) == 2:
            shift_ret = xp.concatenate(
                (
                    xp.zeros(1, dtype=xp.float32),
                    shift_apply.astype(xp.float32, copy=False),
                )
            )
        else:
            shift_ret = shift_apply

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
        """Register and score all overlapping tile pairs independently per timepoint.

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
        df_in = (
            self.downsample_factors
            if downsample_factors is None
            else downsample_factors
        )
        sw = self.ssim_window if ssim_window is None else int(ssim_window)
        th = self.threshold if threshold is None else float(threshold)

        self.pairwise_metrics.clear()

        def overlap_bounds_1d(
            offset: int,
            fixed_length: int,
            moving_length: int,
        ) -> tuple[tuple[int, int], tuple[int, int]]:
            """Return equal-length overlap bounds for two shifted intervals.

            Parameters
            ----------
            offset
                Moving origin relative to the fixed origin, in voxels.
            fixed_length
                Fixed tile length.
            moving_length
                Moving tile length.

            Returns
            -------
            tuple of tuple
                Fixed bounds followed by moving bounds.
            """
            fixed = (
                max(0, offset),
                min(fixed_length, offset + moving_length),
            )
            moving = (
                max(0, -offset),
                min(moving_length, fixed_length - offset),
            )
            return fixed, moving

        df_zyx_base = (int(df_in[0]), int(df_in[1]), int(df_in[2]))
        if int(self.z_dim) == 1:
            df_zyx_base = (1, df_zyx_base[1], df_zyx_base[2])
        available_host_bytes = int(psutil.virtual_memory().available)
        read_workers = min(4, max(1, self._max_workers))
        device_buffers = (
            _RegistrationDeviceBuffers(cp) if USING_GPU and cp is not None else None
        )
        n_pos = int(self.position_dim)
        tile_groups = getattr(
            self,
            "_tiles_by_time",
            tuple(
                tuple(range(t * n_pos, (t + 1) * n_pos))
                for t in range(int(self.time_dim))
            ),
        )
        all_tile_shapes = getattr(
            self,
            "_tile_shapes",
            [(int(self.z_dim), int(self.y_dim), int(self.x_dim))]
            * len(self._tile_positions),
        )

        with ThreadPoolExecutor(max_workers=read_workers) as executor:
            for t, tile_indices in enumerate(tile_groups):
                positions = [self._tile_positions[index] for index in tile_indices]
                tile_shapes = [all_tile_shapes[index] for index in tile_indices]
                nearest_registration_pairs = set(
                    _select_nearest_registration_pairs(
                        positions,
                        tile_shapes,
                        self._pixel_size,
                        is_2d=self._is_2d,
                    )
                )
                registration_pairs = _select_registration_pairs(
                    positions,
                    tile_shapes,
                    self._pixel_size,
                    is_2d=self._is_2d,
                )
                candidates: dict[
                    tuple[int, int],
                    tuple[int, int, int, float],
                ] = {}
                candidate_scores: dict[tuple[int, int], float] = {}
                rejected_shifts: list[tuple[int, int, int]] = []
                pair_specs = []
                input_itemsize = int(
                    np.dtype(self.position_arrays[0].dtype.numpy_dtype).itemsize
                )
                for i_pos, j_pos in registration_pairs:
                    i = tile_indices[i_pos]
                    j = tile_indices[j_pos]
                    if not self._channel_has_signal(
                        i, ch_idx
                    ) or not self._channel_has_signal(j, ch_idx):
                        continue

                    phys = np.array(self._tile_positions[j]) - np.array(
                        self._tile_positions[i]
                    )
                    vox_off = np.round(phys / np.array(self._pixel_size)).astype(int)

                    dz = int(vox_off[0])
                    dy = int(vox_off[1])
                    dx = int(vox_off[2])

                    if self._is_2d:
                        dz = 0

                    shape_i = all_tile_shapes[i]
                    shape_j = all_tile_shapes[j]
                    paired_bounds = [
                        overlap_bounds_1d(offset, int(length_i), int(length_j))
                        for offset, length_i, length_j in zip(
                            (dz, dy, dx),
                            shape_i,
                            shape_j,
                        )
                    ]
                    bounds_i = [bounds[0] for bounds in paired_bounds]
                    bounds_j = [bounds[1] for bounds in paired_bounds]
                    if any(hi <= lo for lo, hi in bounds_i):
                        continue
                    overlap_shape = tuple(hi - lo for lo, hi in bounds_i)
                    active_shape = overlap_shape[1:] if self._is_2d else overlap_shape
                    if min(active_shape) < 3:
                        continue
                    df_zyx_eff = list(
                        _registration_sampling(
                            overlap_shape, df_zyx_base, sw, self._is_2d
                        )
                    )
                    thin_z_overlap = (
                        not self._is_2d and overlap_shape[0] // df_zyx_base[0] < sw
                    )
                    if (
                        thin_z_overlap
                        and (i_pos, j_pos) not in nearest_registration_pairs
                    ):
                        continue
                    if thin_z_overlap and sw > 1:
                        maximum_z_factor = max(
                            1,
                            overlap_shape[0] // sw,
                        )
                        df_zyx_eff[0] = min(
                            df_zyx_eff[0],
                            maximum_z_factor,
                        )
                    separation = np.abs(vox_off).astype(np.float64) / np.maximum(
                        np.asarray(shape_i, dtype=np.float64),
                        np.asarray(shape_j, dtype=np.float64),
                    )
                    if self._is_2d:
                        separation[0] = -math.inf
                    boundary_axis = int(np.argmax(separation))
                    registration_axis = (
                        None
                        if thin_z_overlap
                        else boundary_axis - 1
                        if self._is_2d
                        else boundary_axis
                    )
                    df_zyx_eff_tuple = tuple(df_zyx_eff)
                    pair_specs.append(
                        (
                            i_pos,
                            j_pos,
                            i,
                            j,
                            tuple(bounds_i),
                            tuple(bounds_j),
                            df_zyx_eff_tuple,
                            thin_z_overlap,
                            registration_axis,
                            (2 * math.prod(overlap_shape) * input_itemsize),
                        )
                    )

                read_ahead_budget = min(
                    4 * 1024**3,
                    max(1, int(available_host_bytes * 0.1)),
                )
                read_ahead_pairs = 2 if read_workers > 1 else 1
                pending_reads: deque[
                    tuple[
                        tuple[Any, ...],
                        Future[np.ndarray],
                        Future[np.ndarray],
                    ]
                ] = deque()
                spec_iterator = iter(pair_specs)
                deferred_spec: tuple[Any, ...] | None = None
                pending_read_bytes = 0

                def fill_read_ahead() -> None:
                    """Schedule overlap reads until the bounded prefetch queue is full."""
                    nonlocal deferred_spec, pending_read_bytes
                    while len(pending_reads) < read_ahead_pairs:
                        if deferred_spec is not None:
                            spec = deferred_spec
                            deferred_spec = None
                        else:
                            try:
                                spec = next(spec_iterator)
                            except StopIteration:
                                return
                        _, _, i, j, bounds_i, bounds_j, _, _, _, pair_bytes = spec
                        if (
                            pending_reads
                            and pending_read_bytes + pair_bytes > read_ahead_budget
                        ):
                            deferred_spec = spec
                            return
                        pending_reads.append(
                            (
                                spec,
                                executor.submit(
                                    self._read_registration_patch,
                                    i,
                                    ch_idx,
                                    bounds_i,
                                ),
                                executor.submit(
                                    self._read_registration_patch,
                                    j,
                                    ch_idx,
                                    bounds_j,
                                ),
                            )
                        )
                        pending_read_bytes += int(pair_bytes)

                fill_read_ahead()
                progress = tqdm(
                    total=len(pair_specs),
                    desc=f"register pairs t={t + 1}/{int(self.time_dim)}",
                    leave=False,
                    unit="pair",
                )
                while pending_reads:
                    (
                        spec,
                        future_i,
                        future_j,
                    ) = pending_reads.popleft()
                    (
                        i_pos,
                        j_pos,
                        i,
                        j,
                        _,
                        _,
                        df_zyx_eff,
                        thin_z_overlap,
                        registration_axis,
                        pair_bytes,
                    ) = spec
                    patch_i = future_i.result()
                    patch_j = future_j.result()

                    if self._is_2d:
                        patch_i = np.asarray(patch_i)[0, 0]
                        patch_j = np.asarray(patch_j)[0, 0]
                        reduce_block = df_zyx_eff[1:]
                    else:
                        reduce_block = df_zyx_eff
                    if device_buffers is not None:
                        staged_i, staged_j = device_buffers.stage(
                            np.asarray(patch_i),
                            np.asarray(patch_j),
                        )
                    else:
                        staged_i = xp.asarray(patch_i)
                        staged_j = xp.asarray(patch_j)
                    g1 = block_reduce(
                        staged_i,
                        block_size=reduce_block,
                        func=xp.mean,
                    )
                    g2 = block_reduce(
                        staged_j,
                        block_size=reduce_block,
                        func=xp.mean,
                    )

                    max_shift = getattr(self, "_pair_registration_limits", {}).get(
                        (i, j), self.max_registration_shift_zyx
                    )
                    if self._is_2d:
                        max_shift_ds = (
                            max_shift[1] / df_zyx_eff[1],
                            max_shift[2] / df_zyx_eff[2],
                        )
                    else:
                        max_shift_ds = tuple(
                            limit / factor
                            for limit, factor in zip(max_shift, df_zyx_eff)
                        )
                    shift_ds, score = self.register_and_score(
                        g1,
                        g2,
                        win_size=sw,
                        max_shift=max_shift_ds,
                        allow_bounded_peak=not thin_z_overlap,
                        registration_axis=registration_axis,
                    )
                    progress.update()
                    shift_ds_array = np.asarray(shift_ds, dtype=np.float64)
                    if not math.isfinite(score) or not np.all(
                        np.isfinite(shift_ds_array)
                    ):
                        pending_read_bytes -= int(pair_bytes)
                        fill_read_ahead()
                        continue
                    coarse_shift = np.rint(
                        shift_ds_array * np.asarray(df_zyx_eff, dtype=np.float64)
                    ).astype(np.int64)

                    score = float(max(score, 1e-6))
                    dz_s, dy_s, dx_s = (int(value) for value in coarse_shift)

                    if (
                        abs(dz_s) > max_shift[0]
                        or abs(dy_s) > max_shift[1]
                        or abs(dx_s) > max_shift[2]
                    ):
                        rejected_shifts.append((dz_s, dy_s, dx_s))
                        if self._debug:
                            print(
                                "Dropping link (%d, %d) shift=%s exceeds max=%s",
                                i,
                                j,
                                (dz_s, dy_s, dx_s),
                                max_shift,
                            )
                        pending_read_bytes -= int(pair_bytes)
                        fill_read_ahead()
                        continue

                    candidates[(i_pos, j_pos)] = (
                        dz_s,
                        dy_s,
                        dx_s,
                        round(score, 3),
                    )
                    candidate_scores[(i_pos, j_pos)] = score
                    pending_read_bytes -= int(pair_bytes)
                    fill_read_ahead()
                progress.close()
                if rejected_shifts:
                    largest = tuple(
                        int(value) for value in np.max(np.abs(rejected_shifts), axis=0)
                    )
                    print(
                        f"Registration warning: {len(rejected_shifts)} pairs at "
                        f"time index {t} exceeded the ZYX shift limits "
                        f"(maximum envelope {self.max_registration_shift_zyx} pixels); largest "
                        f"measured absolute corrections were {largest}. "
                        "Check --max-registration-shift-zyx if tiles remain misaligned."
                    )
                if self._debug:
                    print(
                        "Registration source cache: TensorStore native LRU, "
                        f"{self._source_cache_bytes / 1024**3:.2f} GiB limit"
                    )

                selected = {
                    pair
                    for pair in candidates
                    if th == 0.0 or candidate_scores[pair] >= th
                }

                self.pairwise_metrics.update(
                    {
                        (tile_indices[left], tile_indices[right]): candidates[
                            (left, right)
                        ]
                        for left, right in sorted(selected)
                    }
                )

    @staticmethod
    def _solve_global(
        links: list[dict[str, Any]],
        n_tiles: int,
        fixed_indices: list[int],
    ) -> np.ndarray:
        """Solve dense least-squares shifts per-axis with fixed tile constraints.

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
                w = float(link["w"]) / float(link.get("sampling", (1, 1, 1))[axis])
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
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Reject inconsistent cycle edges without disconnecting registered tiles.

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
            Absolute residual threshold in effective registration sampling bins.
        iterative : bool
            If True, repeats outlier rejection until convergence.

        Returns
        -------
        tuple
            Optimized shifts and the registration links retained by the solve.
        """
        shifts = self._solve_global(links, n_tiles, fixed_indices)

        def residuals(ls: list[dict[str, Any]], sh: np.ndarray) -> np.ndarray:
            """Calculate Euclidean residuals for registration links.

            Parameters
            ----------
            ls : list[dict[str, Any]]
                Pairwise links containing tile indices, translations, and sampling.
            sh : np.ndarray
                Current per-tile ZYX corrections in pixels.

            Returns
            -------
            np.ndarray
                Per-link residual lengths in registration sampling units.
            """
            return np.array(
                [
                    np.linalg.norm(
                        (sh[link["j"]] - sh[link["i"]] - link["t"])
                        / np.asarray(link.get("sampling", (1, 1, 1)))
                    )
                    for link in ls
                ],
                dtype=np.float64,
            )

        work = links.copy()
        res = residuals(work, shifts)
        if len(res) == 0:
            return shifts, work
        cutoff = max(abs_thresh, rel_thresh * float(np.median(res)))
        outliers = set(np.where(res > cutoff)[0])

        def can_remove(index: int) -> bool:
            # A bridge is the sole measured placement between its two groups.
            # Reject only redundant cycle edges, keeping every group anchored.
            """Check whether rejecting one registration link preserves connectivity.

            Parameters
            ----------
            index : int
                Tile or registration-link index used by this calculation.

            Returns
            -------
            bool
                True when removing the link preserves its tile component.
            """
            adjacency = [set() for _ in range(n_tiles)]
            for edge_index, link in enumerate(work):
                if edge_index != index:
                    adjacency[link["i"]].add(link["j"])
                    adjacency[link["j"]].add(link["i"])
            target = work[index]["j"]
            visited = {work[index]["i"]}
            frontier = list(visited)
            while frontier:
                node = frontier.pop()
                if node == target:
                    return True
                unseen = adjacency[node] - visited
                visited.update(unseen)
                frontier.extend(unseen)
            return False

        while outliers:
            removable = next(
                (
                    index
                    for index in sorted(
                        outliers, key=lambda index: res[index], reverse=True
                    )
                    if can_remove(index)
                ),
                None,
            )
            if removable is None:
                break
            work.pop(removable)
            # Refit after every rejection. Bulk removal can discard all
            # parallel depth links whose residuals came from the same cycle.
            shifts = self._solve_global(work, n_tiles, fixed_indices)
            if iterative:
                res = residuals(work, shifts)
                cutoff = max(abs_thresh, rel_thresh * float(np.median(res)))
                outliers = set(np.where(res > cutoff)[0])
            else:
                res = np.delete(res, removable)
                outliers = {
                    index - (index > removable)
                    for index in outliers
                    if index != removable
                }

        return shifts, work

    def optimize_shifts(
        self,
        method: str = "TWO_ROUND_ITERATIVE",
        rel_thresh: float = 0.5,
        abs_thresh: float = 1.5,
    ) -> None:
        """Optimize shifts per connected image-link component and timepoint.

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
            Absolute outlier cutoff in effective registration sampling bins.

        Returns
        -------
        None
            Populates ``self.global_offsets`` with one ZYX correction per tile.

        Notes
        -----
        - Every tile must have at least one accepted image-registration link.
        - Each connected component is anchored at one zero correction, preserving
          the stage-derived placement between components.
        """
        n_tiles_total = len(self._tile_positions)

        self.global_offsets = np.zeros((n_tiles_total, 3), dtype=np.float64)
        self.optimized_pairwise_metrics = {}
        if not self.pairwise_metrics:
            return

        def pair_sampling(i, j):
            """Read effective registration sampling for a tile pair.

            Parameters
            ----------
            i
                First tile index in the registration pair.
            j
                Second tile index in the registration pair.

            Returns
            -------
            array or scalar
                Effective per-axis sampling of the registration pair.
            """
            if not hasattr(self, "_tile_shapes"):
                return (1, 1, 1)
            offset = np.rint(
                (np.asarray(self._tile_positions[j]) - self._tile_positions[i])
                / self._pixel_size
            ).astype(int)
            overlap = tuple(
                min(length_i, delta + length_j) - max(0, delta)
                for delta, length_i, length_j in zip(
                    offset, self._tile_shapes[i], self._tile_shapes[j]
                )
            )
            return _registration_sampling(
                overlap, self.downsample_factors, self.ssim_window, self._is_2d
            )

        links_all: list[dict[str, Any]] = [
            {
                "i": int(i),
                "j": int(j),
                "t": np.array(v[:3], dtype=np.float64),
                "w": float(np.sqrt(v[3])),
                "sampling": pair_sampling(i, j),
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

        for tile_indices in self._tiles_by_time:
            n_pos = len(tile_indices)
            global_to_local = {
                tile_index: local_index
                for local_index, tile_index in enumerate(tile_indices)
            }

            local_links: list[dict[str, Any]] = []
            for link in links_all:
                i = int(link["i"])
                j = int(link["j"])
                if i in global_to_local and j in global_to_local:
                    local_links.append(
                        {
                            "i": global_to_local[i],
                            "j": global_to_local[j],
                            "t": link["t"],
                            "w": link["w"],
                            "sampling": link["sampling"],
                        }
                    )

            if not local_links:
                continue

            adjacency = [set() for _ in range(n_pos)]
            for link in local_links:
                left = int(link["i"])
                right = int(link["j"])
                adjacency[left].add(right)
                adjacency[right].add(left)
            fixed = []
            visited: set[int] = set()
            for start in range(n_pos):
                if start in visited:
                    continue
                fixed.append(start)
                visited.add(start)
                frontier = [start]
                while frontier:
                    current = frontier.pop()
                    unseen = adjacency[current] - visited
                    visited.update(unseen)
                    frontier.extend(unseen)
            if mode == "ONE_ROUND":
                d_opt = self._solve_global(local_links, n_pos, fixed)
                retained_links = local_links
            else:
                d_opt, retained_links = self._two_round_opt(
                    local_links,
                    n_pos,
                    fixed,
                    rel_thresh=rel_thresh,
                    abs_thresh=abs_thresh,
                    iterative=is_iterative,
                )

            self.global_offsets[list(tile_indices), :] = d_opt
            for link in retained_links:
                pair = (tile_indices[link["i"]], tile_indices[link["j"]])
                self.optimized_pairwise_metrics[pair] = self.pairwise_metrics[pair]

    def estimate_depth_intensity_gains(self) -> None:
        """Estimate channel gains shared by all XY tiles at each depth.

        Sample registered overlaps only between distinct depths acquired at
        the same XY stage position. Fit robust log intensity ratios separately
        for each channel and time point, anchoring the first connected depth.
        Store gains and provenance for fusion without modifying source tiles.
        Planar acquisitions and acquisitions with one depth retain unit gains.

        Returns
        -------
        None
            Gains and provenance are stored on this instance.
        """
        raw_positions = np.asarray(self.acquisition.stage_positions_zxy)
        source_levels = stage_z_level_indices(raw_positions)
        levels = source_levels[np.asarray(self._tile_source_position_indices)]
        level_count = int(source_levels.max()) + 1
        if self._is_2d or level_count < 2:
            return
        observations = {}
        depth_links = [
            (i, j, value)
            for (i, j), value in getattr(self, "optimized_pairwise_metrics", {}).items()
            if levels[i] != levels[j]
            and np.allclose(
                raw_positions[self._tile_source_position_indices[i], 1:],
                raw_positions[self._tile_source_position_indices[j], 1:],
                atol=0.001,
                rtol=0,
            )
            and not np.isclose(
                raw_positions[self._tile_source_position_indices[i], 0],
                raw_positions[self._tile_source_position_indices[j], 0],
                atol=0.001,
                rtol=0,
            )
        ]
        for i, j, value in tqdm(
            depth_links, desc="match depth intensities", unit="pair"
        ):
            # Pairwise image alignment identifies corresponding specimen pixels.
            # Use it before applying the optimized global placements.
            offset = np.rint(
                (np.asarray(self._tile_positions[j]) - self._tile_positions[i])
                / self._pixel_size
            ).astype(int) + np.asarray(value[:3], dtype=int)
            bounds = [
                (max(0, delta), min(length_i, delta + length_j))
                for delta, length_i, length_j in zip(
                    offset, self._tile_shapes[i], self._tile_shapes[j]
                )
            ]
            if any(hi - lo < 4 for lo, hi in bounds):
                continue
            for fraction in (0.2, 0.5, 0.8):
                patch_bounds = []
                for axis, (lo, hi) in enumerate(bounds):
                    length = min(hi - lo, 32 if axis == 0 else 256)
                    if axis > 0:
                        length -= length % 4
                    start = lo + int(round(fraction * (hi - lo - length)))
                    patch_bounds.append((start, start + length))
                moving_bounds = [
                    (lo - delta, hi - delta)
                    for (lo, hi), delta in zip(patch_bounds, offset)
                ]
                fixed_slices = tuple(slice(lo, hi) for lo, hi in patch_bounds)
                moving_slices = tuple(slice(lo, hi) for lo, hi in moving_bounds)
                fixed = self._read_tile_volume(i, slice(None), *fixed_slices)
                moving = self._read_tile_volume(j, slice(None), *moving_slices)
                support = (
                    self._tile_support_zy[i][fixed_slices[:2]]
                    & self._tile_support_zy[j][moving_slices[:2]]
                )
                channels, depth, height, width = fixed.shape
                fixed = fixed.reshape(
                    channels, depth, height // 4, 4, width // 4, 4
                ).mean(axis=(3, 5))
                moving = moving.reshape(
                    channels, depth, height // 4, 4, width // 4, 4
                ).mean(axis=(3, 5))
                support = support.reshape(depth, height // 4, 4).all(axis=2)[:, :, None]
                for channel in range(int(self.channels)):
                    if not self._channel_has_signal(
                        i, channel
                    ) or not self._channel_has_signal(j, channel):
                        continue
                    a, b = fixed[channel], moving[channel]
                    valid = (
                        support & np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
                    )
                    if np.count_nonzero(valid) < 64:
                        continue
                    signal = a + b
                    valid &= signal >= np.median(signal[valid])
                    log_ratio = float(np.median(np.log(a[valid]) - np.log(b[valid])))
                    if math.isfinite(log_ratio):
                        key = (int(self._tile_time_indices[i]), channel)
                        observations.setdefault(key, []).append(
                            (int(levels[i]), int(levels[j]), log_ratio)
                        )
        records = []
        for (time, channel), measurements in sorted(observations.items()):
            gains = _fit_depth_gains(level_count, measurements)
            for tile, (tile_time, level) in enumerate(
                zip(self._tile_time_indices, levels)
            ):
                if tile_time == time:
                    self._depth_intensity_gains[tile, channel] = gains[level]
            records.append(
                {
                    "time_index": time,
                    "channel_index": channel,
                    "depth_gains": gains.tolist(),
                    "overlap_samples": len(measurements),
                }
            )
            print(
                f"Depth intensity gains t={time}, c={channel}: {np.round(gains, 3).tolist()}"
            )
        self._depth_intensity_report = {
            "enabled": True,
            "method": "same-XY-registered-overlaps-robust-depth-log-gains-v1",
            "anchor": "first depth in each connected depth component",
            "tile_depth_indices": levels.tolist(),
            "channels": records,
        }

    def _scale_depth_source(
        self, tile: int, source: np.ndarray, channel: int | None = None
    ) -> np.ndarray:
        """Apply the depth gains to a fusion source block.

        Parameters
        ----------
        tile : int
            Source tile index.
        source : numpy.ndarray
            Source intensities in CZYX order, or ZYX for one selected channel.
        channel : int | None
            Selected channel index for a ZYX source, or None for all channels.

        Returns
        -------
        numpy.ndarray
            Float32 scaled intensities, or the original block for unit gains.
            The input array is never modified.
        """
        gains = getattr(self, "_depth_intensity_gains", None)
        if gains is None:
            return source
        gain = gains[tile] if channel is None else gains[tile, channel]
        if np.all(gain == 1):
            return source
        if channel is None:
            gain = gain[:, None, None, None]
        return source.astype(np.float32, copy=False) * gain

    def _cast_fusion_values(self, source: np.ndarray) -> np.ndarray:
        """Convert fusion intensities to the configured output dtype.

        Parameters
        ----------
        source : numpy.ndarray
            Intensities to write to a fused output block.

        Returns
        -------
        numpy.ndarray
            Float32 intensities or uint16 intensities clipped to its range.
            Returns the original array when its dtype already matches the output.
            The input array is never modified.
        """
        dtype = getattr(self, "output_dtype", np.dtype(np.uint16))
        if source.dtype == dtype:
            return source
        if dtype == np.dtype(np.float32):
            return source.astype(np.float32, copy=False)
        return np.clip(source, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)

    def report_registration_connectivity(self) -> None:
        """Report overlapping tiles left in separate registration components.

        Inspect the links retained by global optimization independently for
        each time point and warn when overlapping tiles remain disconnected.

        Returns
        -------
        None
            Connectivity diagnostics are printed without changing placements.
        """
        for time_index, tile_indices in enumerate(self._tiles_by_time):
            parent = {index: index for index in tile_indices}

            def find(index: int) -> int:
                """Find a tile component representative with path compression.

                Parameters
                ----------
                index : int
                    Tile or registration-link index used by this calculation.

                Returns
                -------
                int
                    Representative tile index of the connected component.
                """
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            for left, right in getattr(
                self, "optimized_pairwise_metrics", self.pairwise_metrics
            ):
                if left in parent and right in parent:
                    parent[find(right)] = find(left)
            overlapping = _select_registration_pairs(
                [self._tile_positions[index] for index in tile_indices],
                [self._tile_shapes[index] for index in tile_indices],
                self._pixel_size,
                is_2d=self._is_2d,
            )
            disconnected = [
                (tile_indices[left], tile_indices[right])
                for left, right in overlapping
                if find(tile_indices[left]) != find(tile_indices[right])
            ]
            if disconnected:
                components = len(
                    {find(index) for pair in disconnected for index in pair}
                )
                print(
                    f"Registration warning: overlapping tiles at time index "
                    f"{time_index} remain in {components} disconnected components. "
                    "Fusion will retain stage placement between those components; "
                    "they are not mutually registered. Check the registration "
                    "channel, threshold, and --max-registration-shift-zyx."
                )

    def save_pairwise_metrics(self) -> None:
        """Persist pairwise links in the acquisition processing state."""
        self.processing_state.save_registration(
            self.data,
            configuration=self._registration_cache_settings(),
            pairwise_metrics={
                f"{i},{j}": list(values)
                for (i, j), values in self.pairwise_metrics.items()
            },
        )

    def load_pairwise_metrics(self) -> None:
        """Load compatible pairwise links from processing state."""
        data = self.processing_state.registration(
            self.data,
            configuration=self._registration_cache_settings(),
        )
        raw_metrics = data["pairwise_metrics"]

        metrics = {}
        invalid_count = self._count_invalid_pairwise_metrics(raw_metrics)
        for key, value in raw_metrics.items():
            try:
                indices = tuple(map(int, key.split(",")))
                values = tuple(value)
                valid = (
                    len(indices) == 2
                    and len(values) == 4
                    and all(math.isfinite(float(item)) for item in values)
                    and float(values[3]) > 0.0
                )
            except (TypeError, ValueError):
                valid = False

            if valid:
                metrics[indices] = values

        if invalid_count:
            raise ValueError(
                f"Processing state contains {invalid_count} invalid registration metrics"
            )
        self.pairwise_metrics = metrics

    def _registration_cache_settings(self) -> dict[str, Any]:
        """Return every setting that can change pairwise registration.

        Returns
        -------
        dict
            Output-affecting registration settings fingerprinted in the processing journal.
        """
        return {
            "channel_index": self.channel_to_use,
            "stage_y_reversed": self.reverse_stage_y,
            "stage_z_reversed_to_lab": self.reverse_stage_z,
            "stage_z_to_image_y": "none-orthogonal-stage-placement-v3",
            "pair_selection": "all-thick-overlaps-nearest-thin-z-overlaps-v2",
            "acceptance": "threshold-only-no-forced-bridges-v2",
            "overlap_roi": "full-physical-overlap-main-v1",
            "downsample": ("upstream-block-reduce-with-adaptive-zyx-ssim-support-v2"),
            "phase_registration": (
                "thin-z-full-reject-out-of-range-thick-bounded-peak-v7"
            ),
            "score": ("upstream-cucim-ssim-adaptive-aligned-window-boundary-edge-v4"),
            "downsample_factors_zyx": list(self.downsample_factors),
            "ssim_window": self.ssim_window,
            "max_registration_shift_zyx": list(self.max_registration_shift_zyx),
            "shift_limits_policy": (
                "stage-overlap-oblique-footprint-per-pair-v1"
                if getattr(self, "_automatic_registration_limits", False)
                else "explicit"
            ),
            "pair_shift_limits": {
                f"{left},{right}": list(limits)
                for (left, right), limits in getattr(
                    self, "_pair_registration_limits", {}
                ).items()
            },
            "threshold": self.threshold,
            "roi_bounds_yx_um": (
                None
                if getattr(self, "roi_selection", None) is None
                else list(self.roi_selection.bounds_yx_um)
            ),
            "roi_position_indices": list(getattr(self, "_source_position_indices", ())),
            "zero_registration_tiles": [
                tile_index
                for tile_index in range(len(self._tile_positions))
                if not self._channel_has_signal(
                    tile_index,
                    self.channel_to_use,
                )
            ],
        }

    @staticmethod
    def _count_invalid_pairwise_metrics(data: object) -> int:
        """Count malformed or nonfinite pairwise registration records.

        Parameters
        ----------
        data : object
            Saved pairwise registration metrics being checked for finite translations.

        Returns
        -------
        int
            Number of saved links containing invalid or nonfinite registration measurements.
        """
        if not isinstance(data, dict):
            return 1

        invalid_count = 0
        for key, value in data.items():
            try:
                indices = tuple(map(int, key.split(",")))
                values = tuple(value)
                valid = (
                    len(indices) == 2
                    and len(values) == 4
                    and all(math.isfinite(float(item)) for item in values)
                    and float(values[3]) > 0.0
                )
            except (AttributeError, TypeError, ValueError):
                valid = False
            if not valid:
                invalid_count += 1
        return invalid_count

    def _compute_fused_image_space(self) -> None:
        """Compute a global fused space spanning all timepoints.

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

        max_z = max(
            float(position[0]) + int(shape[0]) * dz_um
            for position, shape in zip(self._tile_positions, self._tile_shapes)
        )
        max_y = max(
            float(position[1]) + int(shape[1]) * dy_um
            for position, shape in zip(self._tile_positions, self._tile_shapes)
        )
        max_x = max(
            float(position[2]) + int(shape[2]) * dx_um
            for position, shape in zip(self._tile_positions, self._tile_shapes)
        )
        if getattr(self, "roi_selection", None) is not None:
            roi_y0, roi_y1, roi_x0, roi_x1 = self.roi_selection.bounds_yx_um
            min_y, max_y = float(roi_y0), float(roi_y1)
            min_x, max_x = float(roi_x0), float(roi_x1)

        sz = int(np.ceil((max_z - min_z) / dz_um))
        sy = int(np.ceil((max_y - min_y) / dy_um))
        sx = int(np.ceil((max_x - min_x) / dx_um))

        self.unpadded_shape = (sz, sy, sx)
        self.offset_um = (float(min_z), float(min_y), float(min_x))

    def _pad_to_multiscale_multiple(self) -> None:
        """Minimally pad the fused shape for integer multiscale levels.

        Returns
        -------
        None
            Sets `self.padded_shape` as (Z, Y, X).
        """
        sz, sy, sx = self.unpadded_shape
        alignment = math.lcm(1, *(int(value) for value in self.multiscale_factors))
        z_alignment = 1 if self._is_2d else alignment

        pz = (-sz) % z_alignment
        py = (-sy) % alignment
        px = (-sx) % alignment

        self.padded_shape = (sz + pz, sy + py, sx + px)

    def _prepare_fused_image(
        self,
        output_path: str | Path,
        z_slices_per_write: int | None = None,
    ) -> tuple[ts.TensorStore, list[int]]:
        """Create the fused multiscale Image through yaozarrs.

        Parameters
        ----------
        output_path : str | Path
            Destination fused OME-Zarr image directory.
        z_slices_per_write : int or None
            Z depth rendered per scale-0 block. By default, this matches the
            input Zarr chunk depth so source chunks are not decompressed again
            for successive shallow slabs.

        Returns
        -------
        tuple[ts.TensorStore, list[int]]
            Level-zero output handle and YX pyramid reduction factors.
        """
        factors = (1, *self.multiscale_factors)
        dz, dy, dx = round_spatial_values(self._pixel_size)
        offset_z, offset_y, offset_x = round_spatial_values(self.offset_um)
        axes = [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
        datasets = []
        specs = []
        z0, y0, x0 = (int(value) for value in self.padded_shape)
        for level, factor in enumerate(factors):
            z_factor = 1 if self._is_2d else int(factor)
            center_shift = 0.5 if self.multiscale_downsample == "block_mean" else 0.0
            translation = round_tczyx_transform(
                (
                    0,
                    0,
                    offset_z + center_shift * (z_factor - 1) * dz,
                    offset_y + center_shift * (factor - 1) * dy,
                    offset_x + center_shift * (factor - 1) * dx,
                )
            )
            datasets.append(
                {
                    "path": str(level),
                    "coordinateTransformations": [
                        {
                            "scale": round_tczyx_transform(
                                (
                                    1,
                                    1,
                                    dz * z_factor,
                                    dy * factor,
                                    dx * factor,
                                )
                            )
                        },
                        {"translation": translation},
                    ],
                }
            )
            specs.append(
                (
                    (
                        int(self.time_dim),
                        int(self.channels),
                        max(1, (z0 + z_factor - 1) // z_factor),
                        max(1, (y0 + factor - 1) // factor),
                        max(1, (x0 + factor - 1) // factor),
                    ),
                    getattr(self, "output_dtype", np.dtype(np.uint16)),
                )
            )

        image = v05.Image(
            multiscales=[v05.Multiscale(name="fused", axes=axes, datasets=datasets)]
        )
        # Keep storage chunks shallow enough for interactive plane viewing.
        # Rendering uses a separate, much deeper RAM block chosen below.
        output_chunk_z = 1 if self._is_2d else min(4, z0)
        self._output_chunk_z = output_chunk_z
        codec_chunks = (
            1,
            1,
            output_chunk_z,
            int(self.chunk_y),
            int(self.chunk_x),
        )
        _, self._multiscale_arrays = prepare_image(
            output_path,
            image,
            specs,
            extra_attributes={},
            chunks=codec_chunks,
            writer="tensorstore",
            overwrite=True,
        )
        if z_slices_per_write is None:
            z_slices_per_write = self._fusion_z_step(z0)
        write_block_shape = [
            1,
            1,
            min(int(z_slices_per_write), z0),
            int(self.chunk_y) * 2,
            int(self.chunk_x) * 2,
        ]
        return self._multiscale_arrays["0"], write_block_shape

    def _fusion_z_step(self, z_size: int) -> int:
        """Choose an input-aware Z depth that still permits aligned Y/X blocks.

        Parameters
        ----------
        z_size : int
            Output Z extent available to the fusion slab.

        Returns
        -------
        int
            Z slab size fitting the memory budget and requested output extent.
        """
        output_chunk_z = max(1, int(self._output_chunk_z))
        target_z = min(
            int(z_size),
            max(int(self._input_chunk_zyx[0]), output_chunk_z),
        )
        available = int(psutil.virtual_memory().available)
        budget = max(1, int(available * self.fusion_ram_fraction))
        concurrent_buffers = max(
            1, int(self._max_workers) + int(self.max_in_flight_writes)
        )
        bytes_per_voxel = max(16, 12 * int(self.channels) + 4)
        bytes_per_z_plane = int(self.chunk_y) * int(self.chunk_x) * bytes_per_voxel
        max_z = max(
            output_chunk_z,
            budget // concurrent_buffers // max(1, bytes_per_z_plane),
        )
        aligned_z = max(
            output_chunk_z,
            (min(target_z, max_z) // output_chunk_z) * output_chunk_z,
        )
        return min(int(z_size), aligned_z)

    def _fusion_block_shape(
        self,
        z_depth: int,
        y_size: int,
        x_size: int,
    ) -> tuple[int, int]:
        """Choose a host-RAM-bounded Y/X accumulator shape for one Z slab.

        Parameters
        ----------
        z_depth : int
            Number of Z planes held in each accumulator.
        y_size : int
            Full output extent along Y, in pixels.
        x_size : int
            Full output extent along X, in pixels.

        Returns
        -------
        tuple[int, int]
            YX block dimensions that fit the per-worker memory budget.
        """
        available = int(psutil.virtual_memory().available)
        budget = max(1, int(available * self.fusion_ram_fraction))
        concurrent_buffers = max(
            1, int(self._max_workers) + int(self.max_in_flight_writes)
        )
        per_block_budget = max(1, budget // concurrent_buffers)

        # A mixed block holds uint16 source/output arrays plus float32 overlap
        # accumulators. Budget per concurrently rendered block, including channels.
        bytes_per_voxel = max(16, 12 * int(self.channels) + 4)
        max_xy_pixels = max(
            1,
            per_block_budget // (max(1, z_depth) * bytes_per_voxel),
        )
        aspect = float(y_size) / float(max(1, x_size))
        block_y = max(1, int(np.sqrt(max_xy_pixels * aspect)))
        block_y = min(y_size, block_y, int(self.chunk_y) * 8)
        block_x = min(
            x_size,
            max(1, max_xy_pixels // block_y),
            int(self.chunk_x) * 8,
        )

        block_y = max(min(y_size, int(self.chunk_y)), block_y)
        block_x = max(min(x_size, int(self.chunk_x)), block_x)
        if block_y >= self.chunk_y:
            block_y = max(self.chunk_y, (block_y // self.chunk_y) * self.chunk_y)
        if block_x >= self.chunk_x:
            block_x = max(self.chunk_x, (block_x // self.chunk_x) * self.chunk_x)
        return int(block_y), int(block_x)

    def _render_fusion_block(
        self,
        t: int,
        bounds: tuple[int, int, int, int, int, int],
        contributors: Sequence[tuple[int, tuple[int, int, int]]],
    ) -> tuple[Any, np.ndarray, int, int]:
        """Render one independent output block without writing it.

        Each contributing tile is read at most once. Exclusive regions are copied
        directly into the output. Overlaps use float32 feather accumulation and
        normalize without crop, clipping, or cast buffers. Float32 output reuses
        the accumulation buffer; uint16 normalization writes directly into output.

        Parameters
        ----------
        t : int
            Output timepoint index.
        bounds : tuple[int, int, int, int, int, int]
            Half-open spatial block bounds in ZYX order.
        contributors : Sequence[tuple[int, tuple[int, int, int]]]
            Tile indices and integer canvas ZYX origins intersecting the output block.

        Returns
        -------
        tuple
            Destination selection, rendered pixels, direct-copy region count, and blended region count.
        """
        z0, z1, y0, y1, x0, x1 = bounds
        common_shape = (int(self.z_dim), int(self.y_dim), int(self.x_dim))
        tile_shapes = getattr(
            self,
            "_tile_shapes",
            [common_shape] * len(self._tile_positions),
        )
        regions = _partition_fusion_block(bounds, contributors, tile_shapes)

        if len(regions) == 1 and len(regions[0][1]) == 1:
            region, region_contributors = regions[0]
            rz0, rz1, ry0, ry1, rx0, rx1 = region
            p, (oz, oy, ox) = region_contributors[0]
            direct = self._read_tile_volume(
                p,
                slice(None),
                slice(rz0 - oz, rz1 - oz),
                slice(ry0 - oy, ry1 - oy),
                slice(rx0 - ox, rx1 - ox),
                dtype=getattr(self, "output_dtype", np.dtype(np.uint16)),
            )
            direct = self._cast_fusion_values(self._scale_depth_source(p, direct))
            selection = (
                t,
                slice(None),
                slice(rz0, rz1),
                slice(ry0, ry1),
                slice(rx0, rx1),
            )
            return selection, direct, 1, 0

        source_blocks: dict[int, tuple[tuple[int, int, int], np.ndarray]] = {}
        for p, (oz, oy, ox) in contributors:
            tile_shape = tile_shapes[p]
            tz0, tz1 = max(z0, oz), min(z1, oz + tile_shape[0])
            ty0, ty1 = max(y0, oy), min(y1, oy + tile_shape[1])
            tx0, tx1 = max(x0, ox), min(x1, ox + tile_shape[2])
            source = self._read_tile_volume(
                p,
                slice(None),
                slice(tz0 - oz, tz1 - oz),
                slice(ty0 - oy, ty1 - oy),
                slice(tx0 - ox, tx1 - ox),
                dtype=None,
            )
            source_blocks[p] = ((tz0, ty0, tx0), source)
        depth_gains = getattr(self, "_depth_intensity_gains", None)

        output = np.zeros(
            (int(self.channels), z1 - z0, y1 - y0, x1 - x0),
            dtype=getattr(self, "output_dtype", np.dtype(np.uint16)),
        )
        direct_regions = 0
        blended_regions = 0
        for region, region_contributors in regions:
            rz0, rz1, ry0, ry1, rx0, rx1 = region
            output_selection = (
                slice(None),
                slice(rz0 - z0, rz1 - z0),
                slice(ry0 - y0, ry1 - y0),
                slice(rx0 - x0, rx1 - x0),
            )
            if len(region_contributors) == 1:
                p, _ = region_contributors[0]
                (sz0, sy0, sx0), source = source_blocks[p]
                output[output_selection] = self._cast_fusion_values(
                    self._scale_depth_source(
                        p,
                        source[
                            :,
                            rz0 - sz0 : rz1 - sz0,
                            ry0 - sy0 : ry1 - sy0,
                            rx0 - sx0 : rx1 - sx0,
                        ],
                    )
                )
                direct_regions += 1
                continue

            fused = np.zeros(
                (
                    int(self.channels),
                    rz1 - rz0,
                    ry1 - ry0,
                    rx1 - rx0,
                ),
                dtype=np.float32,
            )
            channel_flags = [
                np.asarray(
                    [
                        self._channel_has_signal(p, channel_index)
                        for channel_index in range(int(self.channels))
                    ],
                    dtype=np.bool_,
                )
                for p, _offset in region_contributors
            ]
            shared_weights = all(np.all(present) for present in channel_flags)
            weight = np.zeros(
                (1 if shared_weights else int(self.channels), *fused.shape[1:]),
                dtype=np.float32,
            )
            for (p, (oz, oy, ox)), present in zip(
                region_contributors, channel_flags, strict=True
            ):
                (sz0, sy0, sx0), source = source_blocks[p]
                local_z = slice(rz0 - oz, rz1 - oz)
                local_y = slice(ry0 - oy, ry1 - oy)
                local_x = slice(rx0 - ox, rx1 - ox)
                tile_supports = getattr(self, "_tile_support_zy", None)
                tile_support = (
                    self._deskew_support_zy
                    if tile_supports is None
                    else tile_supports[p]
                )
                tile_profiles = getattr(self, "_tile_profiles", None)
                if tile_profiles is None:
                    z_profile = self.z_profile
                    y_profile = self.y_profile
                    x_profile = self.x_profile
                else:
                    z_profile, y_profile, x_profile = tile_profiles[p]
                _accumulate_tile_block(
                    fused,
                    weight,
                    source,
                    present,
                    tile_support[local_z, local_y],
                    z_profile[local_z],
                    y_profile[local_y],
                    x_profile[local_x],
                    0,
                    0,
                    0,
                    rz0 - sz0,
                    ry0 - sy0,
                    rx0 - sx0,
                    (
                        None
                        if depth_gains is None or np.all(depth_gains[p] == 1)
                        else depth_gains[p]
                    ),
                )

            region_output = output[output_selection]
            if output.dtype == np.dtype(np.float32):
                _normalize_block(fused, weight, fused)
                region_output[:] = fused
            else:
                _normalize_block(fused, weight, output, rz0 - z0, ry0 - y0, rx0 - x0)
            for channel_index in range(int(self.channels)):
                active_tiles = [
                    tile_index
                    for (tile_index, _offset), present in zip(
                        region_contributors, channel_flags, strict=True
                    )
                    if present[channel_index]
                ]
                if not active_tiles:
                    region_output[channel_index] = 0
                    continue
                if len(active_tiles) != 1:
                    continue
                tile_index = active_tiles[0]
                (sz0, sy0, sx0), source = source_blocks[tile_index]
                region_output[channel_index] = self._cast_fusion_values(
                    self._scale_depth_source(
                        tile_index,
                        source[
                            channel_index,
                            rz0 - sz0 : rz1 - sz0,
                            ry0 - sy0 : ry1 - sy0,
                            rx0 - sx0 : rx1 - sx0,
                        ],
                        channel_index,
                    )
                )
            blended_regions += 1

        selection = (
            t,
            slice(None),
            slice(z0, z1),
            slice(y0, y1),
            slice(x0, x1),
        )
        return selection, output, direct_regions, blended_regions

    def _fuse_by_blocks(self) -> None:
        """Fuse all timepoints into the global fused store using bounded block writes.

        Returns
        -------
        None
            Writes into `self.fused_ts`.

        Raises
        ------
        RuntimeError
            If fusion prerequisites are not initialized.
        """
        if (
            self.fused_ts is None
            or self.write_block_shape is None
            or self.offset_um is None
            or self.padded_shape is None
        ):
            raise RuntimeError(
                "Fusion not initialized: compute fused space and create output store first."
            )

        n_t = int(self.time_dim)
        n_pos = int(self.position_dim)
        tiles_by_time = getattr(
            self,
            "_tiles_by_time",
            tuple(tuple(range(t * n_pos, (t + 1) * n_pos)) for t in range(n_t)),
        )
        common_shape = (int(self.z_dim), int(self.y_dim), int(self.x_dim))
        tile_shapes = getattr(
            self,
            "_tile_shapes",
            [common_shape] * len(self._tile_positions),
        )

        z_step = int(self.write_block_shape[2])
        pad_z, pad_y, pad_x = (
            int(self.padded_shape[0]),
            int(self.padded_shape[1]),
            int(self.padded_shape[2]),
        )

        dz_um, dy_um, dx_um = self._pixel_size
        off_z_um, off_y_um, off_x_um = self.offset_um

        pending_writes: deque[Any] = deque()
        direct_regions = 0
        blended_regions = 0
        empty_blocks = 0

        def queue_write(selection: Any, value: np.ndarray) -> None:
            """Issue one bounded asynchronous output write.

            Parameters
            ----------
            selection : Any
                Destination TensorStore region for an asynchronous write.
            value : np.ndarray
                Rendered output block retained until its asynchronous write completes.
            """
            pending_writes.append(self.fused_ts[selection].write(value))
            if len(pending_writes) >= self.max_in_flight_writes:
                pending_writes.popleft().result()

        block_layout: list[tuple[int, int, int, int]] = []
        spatial_blocks = 0
        for z0 in range(0, pad_z, z_step):
            z1 = min(z0 + z_step, pad_z)
            block_y, block_x = self._fusion_block_shape(z1 - z0, pad_y, pad_x)
            block_layout.append((z0, z1, block_y, block_x))
            spatial_blocks += math.ceil(pad_y / block_y) * math.ceil(pad_x / block_x)
        chunk_total = n_t * int(self.channels) * spatial_blocks

        scale_bar = tqdm(
            total=n_t,
            desc="scale0",
            leave=True,
            unit="timepoint",
        )
        chunk_bar = tqdm(
            total=chunk_total,
            desc="scale0 chunks",
            leave=False,
            unit="chunk",
        )
        render_workers = max(1, int(self._max_workers))
        pending_renders: set[Future[Any]] = set()

        def collect_rendered(done: set[Future[Any]]) -> None:
            """Queue completed block writes and update observable statistics.

            Parameters
            ----------
            done : set[Future[Any]]
                Completed futures whose results are collected and checked.
            """
            nonlocal direct_regions, blended_regions
            for future in done:
                selection, value, direct_count, blended_count = future.result()
                queue_write(selection, value)
                direct_regions += int(direct_count)
                blended_regions += int(blended_count)
                chunk_bar.update(int(self.channels))

        with ThreadPoolExecutor(max_workers=render_workers) as render_executor:
            for t in range(n_t):
                offsets_t: list[tuple[int, tuple[int, int, int]]] = []
                for tile_index in tiles_by_time[t]:
                    z_um, y_um, x_um = self._tile_positions[tile_index]
                    oz = int(np.round((z_um - off_z_um) / dz_um))
                    oy = int(np.round((y_um - off_y_um) / dy_um))
                    ox = int(np.round((x_um - off_x_um) / dx_um))
                    offsets_t.append((tile_index, (oz, oy, ox)))

                for z0, z1, block_y, block_x in block_layout:
                    for y0 in range(0, pad_y, block_y):
                        y1 = min(y0 + block_y, pad_y)
                        for x0 in range(0, pad_x, block_x):
                            x1 = min(x0 + block_x, pad_x)
                            overlapping = [
                                (tile_index, offset)
                                for tile_index, offset in offsets_t
                                if any(
                                    self._channel_has_signal(
                                        tile_index,
                                        channel_index,
                                    )
                                    for channel_index in range(int(self.channels))
                                )
                                and offset[0] + tile_shapes[tile_index][0] > z0
                                and offset[0] < z1
                                and offset[1] + tile_shapes[tile_index][1] > y0
                                and offset[1] < y1
                                and offset[2] + tile_shapes[tile_index][2] > x0
                                and offset[2] < x1
                            ]
                            if not overlapping:
                                empty_blocks += 1
                                chunk_bar.update(int(self.channels))
                                continue
                            pending_renders.add(
                                render_executor.submit(
                                    self._render_fusion_block,
                                    t,
                                    (z0, z1, y0, y1, x0, x1),
                                    overlapping,
                                )
                            )
                            if len(pending_renders) >= render_workers:
                                done, pending_renders = wait(
                                    pending_renders,
                                    return_when=FIRST_COMPLETED,
                                )
                                collect_rendered(done)

                while pending_renders:
                    done, pending_renders = wait(
                        pending_renders,
                        return_when=FIRST_COMPLETED,
                    )
                    collect_rendered(done)
                while pending_writes:
                    pending_writes.popleft().result()
                scale_bar.update()

        while pending_writes:
            pending_writes.popleft().result()

        chunk_bar.close()
        scale_bar.close()
        self._fusion_stats = {
            "direct_regions": direct_regions,
            "blended_regions": blended_regions,
            "empty_blocks": empty_blocks,
        }
        if self._debug:
            print(
                "[fusion] regions: "
                f"direct={direct_regions}, blended={blended_regions}, "
                f"empty={empty_blocks}, render_workers={render_workers}"
            )

    def _write_multiscales(self) -> None:
        """Downsample successive scales in parallel, storage-aligned blocks.

        Returns
        -------
        None
            No value is returned.
        """
        previous_factor = 1
        for level, factor in enumerate(self.multiscale_factors, start=1):
            factor = int(factor)
            if factor > previous_factor and factor % previous_factor == 0:
                inp = self._multiscale_arrays[str(level - 1)]
                relative_factor = factor // previous_factor
            else:
                inp = self._multiscale_arrays["0"]
                relative_factor = factor

            out = self._multiscale_arrays[str(level)]
            z_factor = 1 if self._is_2d else relative_factor
            out_z, out_y, out_x = (int(value) for value in out.shape[-3:])
            block_z = min(out_z, max(1, int(self.write_block_shape[2])))
            block_y = min(out_y, max(1, int(self.chunk_y)))
            block_x = min(out_x, max(1, int(self.chunk_x)))

            chunk_total = (
                int(self.time_dim)
                * int(self.channels)
                * math.ceil(out_z / block_z)
                * math.ceil(out_y / block_y)
                * math.ceil(out_x / block_x)
            )
            scale_bar = tqdm(
                total=int(self.time_dim),
                desc=f"scale{level}",
                leave=True,
                unit="timepoint",
            )
            chunk_bar = tqdm(
                total=chunk_total,
                desc=f"scale{level} chunks",
                leave=False,
                unit="chunk",
            )

            def write_block(
                t: int,
                c: int,
                z0: int,
                z1: int,
                y0: int,
                y1: int,
                x0: int,
                x1: int,
            ) -> None:
                """Read, downsample, and write one output-aligned block.

                Parameters
                ----------
                t : int
                    Output timepoint index.
                c : int
                    Output channel index.
                z0 : int
                    Inclusive Z bound of the output block.
                z1 : int
                    Exclusive Z bound of the output block.
                y0 : int
                    Inclusive Y bound of the output block.
                y1 : int
                    Exclusive Y bound of the output block.
                x0 : int
                    Inclusive X bound of the output block.
                x1 : int
                    Exclusive X bound of the output block.
                """
                slab = (
                    inp[
                        t,
                        c,
                        slice(
                            z0 * z_factor,
                            min(int(inp.shape[2]), z1 * z_factor),
                        ),
                        slice(
                            y0 * relative_factor,
                            min(int(inp.shape[3]), y1 * relative_factor),
                        ),
                        slice(
                            x0 * relative_factor,
                            min(int(inp.shape[4]), x1 * relative_factor),
                        ),
                    ]
                    .read()
                    .result()
                )
                if self.multiscale_downsample == "stride":
                    down = slab[
                        ::z_factor,
                        ::relative_factor,
                        ::relative_factor,
                    ]
                else:
                    arr = xp.asarray(slab)
                    down_arr = block_reduce(
                        arr,
                        block_size=(
                            z_factor,
                            relative_factor,
                            relative_factor,
                        ),
                        func=xp.mean,
                    )
                    down = (
                        cp.asnumpy(down_arr)
                        if USING_GPU and cp is not None
                        else np.asarray(down_arr)
                    )
                out[
                    t,
                    c,
                    slice(z0, z1),
                    slice(y0, y1),
                    slice(x0, x1),
                ].write(
                    down[: z1 - z0, : y1 - y0, : x1 - x0].astype(
                        getattr(self, "output_dtype", np.dtype(np.uint16)),
                        copy=False,
                    )
                ).result()

            workers = max(1, int(self._max_workers))
            pending: set[Future[Any]] = set()

            def collect_completed(done: set[Future[Any]]) -> None:
                """Propagate worker failures and advance the transient bar.

                Parameters
                ----------
                done : set[Future[Any]]
                    Completed futures whose results are collected and checked.
                """
                for future in done:
                    future.result()
                    chunk_bar.update()

            with ThreadPoolExecutor(max_workers=workers) as executor:
                for t in range(int(self.time_dim)):
                    for c in range(int(self.channels)):
                        for z0 in range(0, out_z, block_z):
                            z1 = min(z0 + block_z, out_z)
                            for y0 in range(0, out_y, block_y):
                                y1 = min(y0 + block_y, out_y)
                                for x0 in range(0, out_x, block_x):
                                    x1 = min(x0 + block_x, out_x)
                                    pending.add(
                                        executor.submit(
                                            write_block,
                                            t,
                                            c,
                                            z0,
                                            z1,
                                            y0,
                                            y1,
                                            x0,
                                            x1,
                                        )
                                    )
                                    if len(pending) >= workers:
                                        done, pending = wait(
                                            pending,
                                            return_when=FIRST_COMPLETED,
                                        )
                                        collect_completed(done)
                    while pending:
                        done, pending = wait(
                            pending,
                            return_when=FIRST_COMPLETED,
                        )
                        collect_completed(done)
                    scale_bar.update()

            chunk_bar.close()
            scale_bar.close()
            previous_factor = factor

    def run(self) -> None:
        """Execute the full registration + fusion pipeline.

        Returns
        -------
        None
            Writes the fused NGFF store to:
            `{processing_output}/{stem}_fused.ome.zarr`

        Raises
        ------
        RuntimeError
            If required intermediate computations fail.
        """
        reuse_registered = getattr(
            self,
            "_reuse_registered_roi_placements",
            False,
        )
        if reuse_registered:
            print("Reusing registered full-fusion tile placements for ROI fusion.")
            self.global_offsets = np.zeros(
                (len(self._tile_positions), 3),
                dtype=np.float64,
            )
        else:
            recomputed_metrics = False
            try:
                self.load_pairwise_metrics()
            except ValueError:
                self.refine_tile_positions_with_cross_correlation(
                    downsample_factors=self.downsample_factors,
                    ch_idx=self.channel_to_use,
                    threshold=self.threshold,
                )
                recomputed_metrics = True

            if recomputed_metrics:
                self.save_pairwise_metrics()

            self.optimize_shifts(
                method="TWO_ROUND_ITERATIVE",
                rel_thresh=self.optimization_rel_threshold,
                abs_thresh=self.optimization_abs_threshold,
            )
            self.report_registration_connectivity()

        if self.normalize_depth_intensity:
            self.estimate_depth_intensity_gains()

        gc.collect()
        if USING_GPU and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        if not reuse_registered:
            self._tile_positions = [
                tuple(np.array(pos) + off * np.array(self._pixel_size))
                for pos, off in zip(self._tile_positions, self.global_offsets)
            ]

        self._compute_fused_image_space()
        self._pad_to_multiscale_multiple()

        omezarr = self.output_dir / f"{self.acquisition_name}_fused.ome.zarr"
        self.fused_ts, self.write_block_shape = self._prepare_fused_image(omezarr)
        self._fuse_by_blocks()
        self._write_multiscales()
        report_path = (
            self.output_dir / f"{self.acquisition_name}_depth_intensity_gains.json"
        )
        report_path.write_text(
            json.dumps(self._depth_intensity_report or {"enabled": False}, indent=2)
        )
        if not reuse_registered:
            self.processing_state.complete_registration(
                self.data,
                fused_path=omezarr,
                tiles=(
                    {
                        "time_index": int(self._tile_time_indices[tile_index]),
                        "position_index": int(
                            self._tile_source_position_indices[tile_index]
                        ),
                        "origin_zyx_um": list(round_spatial_values(tile_position)),
                    }
                    for tile_index, tile_position in enumerate(self._tile_positions)
                ),
            )
