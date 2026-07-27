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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import tensorstore as ts
from numba import njit, prange
from tqdm import tqdm
from yaozarrs import v05
from yaozarrs.write.v05 import prepare_image

from opm_processing.cuda import preload_cuda_libraries
from opm_processing.dataio.acquisition import (
    acquisition_stem,
    resolve_acquisition_path,
)
from opm_processing.dataio.position_collection import open_position_collection
from opm_processing.imageprocessing.coordinates import (
    stage_positions_to_image_coordinates,
)


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

_ssim_cpu: Any | None = None

xp: Any = np

preload_cuda_libraries()

try:
    import cupy as _cp  # type: ignore
    from cucim.skimage.exposure import match_histograms as _mh  # type: ignore
    from cucim.skimage.measure import block_reduce as _br  # type: ignore
    from cucim.skimage.registration import phase_cross_correlation as _pcc  # type: ignore
    from opm_processing.imageprocessing.ssim_cuda import (  # type: ignore
        structural_similarity_cupy_sep_shared as _ssim_cuda,
    )

    cp = _cp
    xp = _cp
    match_histograms = _mh
    block_reduce = _br
    phase_cross_correlation = _pcc
    ssim_cuda = _ssim_cuda
    USING_GPU = True
except Exception as error:  # noqa: BLE001
    # GPU stack unavailable; fall back to CPU.
    GPU_IMPORT_ERROR = error
    from skimage.exposure import match_histograms as _mh  # type: ignore
    from skimage.measure import block_reduce as _br  # type: ignore
    from skimage.metrics import structural_similarity as _cpu_ssim  # type: ignore
    from skimage.registration import phase_cross_correlation as _pcc  # type: ignore

    match_histograms = _mh
    block_reduce = _br
    phase_cross_correlation = _pcc
    _ssim_cpu = _cpu_ssim
    USING_GPU = False


def _default_fusion_workers() -> int:
    """Use up to eight physical cores to preserve deep, aligned I/O blocks."""
    physical = psutil.cpu_count(logical=False)
    if physical is not None:
        return max(1, min(8, int(physical)))
    logical = psutil.cpu_count(logical=True)
    return max(1, min(8, int(logical or 2) // 2))


def fusion_backend_status(
    max_workers: int | None = None,
) -> dict[str, str | bool | int | None]:
    """Return observable registration and fusion backend information."""
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
        return float(ssim_cuda(arr1, arr2, win_size=win_size))

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
    """Return corresponding valid views after applying an integer moving shift.

    Registration uses subpixel phase correlation, but SSIM is only an acceptance
    score. Scoring the nearest integer-aligned overlap avoids resampling the
    entire volume and excludes the artificial zero padding introduced by an
    image-shift operation.
    """
    if fixed.shape != moving.shape:
        raise ValueError("registration patches must have matching shapes")
    if fixed.ndim != len(shift):
        raise ValueError("registration shift rank must match the patch rank")

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


def _bounded_disambiguate_shift(
    fixed: np.ndarray,
    moving: np.ndarray,
    periodic_shift: Sequence[float],
    max_shift: Sequence[float] | None,
) -> np.ndarray:
    """Resolve Fourier wraparound among physically permitted shift candidates.

    Phase correlation reports each axis modulo its array length. When the
    permitted correction reaches beyond half an array dimension, compare the
    few equivalent candidates directly over their valid real-space overlaps.
    This supplies the disambiguation needed by small overlaps without shifting
    and rescoring every full-volume wraparound quadrant.
    """
    shift = np.asarray(periodic_shift, dtype=np.float32)
    if shift.shape != (fixed.ndim,):
        raise ValueError("periodic shift rank must match the patch rank")
    if max_shift is None:
        return shift

    limits = np.asarray(max_shift, dtype=np.float32)
    if limits.shape != shift.shape:
        raise ValueError("maximum shift rank must match the patch rank")

    axis_candidates: list[list[float]] = []
    for value, length, limit in zip(shift, fixed.shape, limits):
        candidates = {
            float(value) + cycle * int(length)
            for cycle in (-1, 0, 1)
            if abs(float(value) + cycle * int(length)) <= float(limit) + 0.5
        }
        if not candidates:
            candidates = {float(value)}
        axis_candidates.append(sorted(candidates, key=lambda item: (abs(item), item)))

    if math.prod(len(candidates) for candidates in axis_candidates) == 1:
        return shift

    best_shift = shift
    best_score = -math.inf
    for candidate_values in product(*axis_candidates):
        try:
            fixed_view, moving_view = _aligned_registration_views(
                fixed,
                moving,
                candidate_values,
            )
        except ValueError:
            continue

        sample_count = int(math.prod(fixed_view.shape))
        if sample_count < 2:
            continue
        fixed_sum = np.sum(fixed_view, dtype=np.float64)
        moving_sum = np.sum(moving_view, dtype=np.float64)
        cross_sum = np.sum(fixed_view * moving_view, dtype=np.float64)
        fixed_sq_sum = np.sum(fixed_view * fixed_view, dtype=np.float64)
        moving_sq_sum = np.sum(moving_view * moving_view, dtype=np.float64)
        covariance = cross_sum - fixed_sum * moving_sum / sample_count
        fixed_energy = fixed_sq_sum - fixed_sum * fixed_sum / sample_count
        moving_energy = moving_sq_sum - moving_sum * moving_sum / sample_count
        denominator = math.sqrt(max(fixed_energy * moving_energy, 0.0))
        score = covariance / denominator if denominator > 0.0 else -math.inf
        if score > best_score:
            best_score = score
            best_shift = np.asarray(candidate_values, dtype=np.float32)

    return best_shift


@njit(nogil=True)
def _accumulate_tile_block(
    fused: np.ndarray,
    weight: np.ndarray,
    sub: np.ndarray,
    z_weights: np.ndarray,
    y_weights: np.ndarray,
    x_weights: np.ndarray,
    z_off: int,
    y_off: int,
    x_off: int,
) -> None:
    """Accumulate a weighted sub-volume into fused and weight buffers (in-place).

    Parameters
    ----------
    fused : numpy.ndarray
        Float32 accumulation buffer of shape (C, dz, Y, X) for the current block.
    weight : numpy.ndarray
        Float32 weight accumulation buffer of shape (dz, Y, X).
    sub : numpy.ndarray
        Tile slab of shape (C, sub_dz, Y_tile, X_tile) to blend.
    z_weights, y_weights, x_weights : numpy.ndarray
        Separable float32 feather profiles for the tile sub-volume.
    z_off : int
        Offset of `sub` z=0 within `fused` block coordinates.
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

    for idx in range(total):
        dz_i = idx // y_sub
        y_i = idx % y_sub
        gz = z_off + dz_i
        gy = y_off + y_i
        base_w = weight[gz, gy]
        zy_weight = z_weights[dz_i] * y_weights[y_i]
        for x_i in range(x_sub):
            gx = x_off + x_i
            w_val = zy_weight * x_weights[x_i]
            base_w[gx] += w_val
        for c in range(c_dim):
            sub_line = sub[c, dz_i, y_i]
            base_f = fused[c, gz, gy]
            for x_i in range(x_sub):
                gx = x_off + x_i
                w_val = zy_weight * x_weights[x_i]
                base_f[gx] += sub_line[x_i] * w_val


@njit(nogil=True)
def _normalize_block(fused: np.ndarray, weight: np.ndarray) -> None:
    """Normalize fused by weight in-place.

    Parameters
    ----------
    fused : numpy.ndarray
        Float32 accumulation buffer of shape (C, dz, Y, X).
    weight : numpy.ndarray
        Float32 weight buffer of shape (dz, Y, X).

    Returns
    -------
    None
        Operates in-place on `fused`.
    """
    c_dim, dz, y_dim, x_dim = fused.shape
    total = dz * y_dim

    for idx in range(total):
        z_i = idx // y_dim
        y_i = idx % y_dim
        base_w = weight[z_i, y_i]
        for c in range(c_dim):
            base_f = fused[c, z_i, y_i]
            for x_i in range(x_dim):
                w_val = base_w[x_i]
                base_f[x_i] = base_f[x_i] / w_val if w_val > 0 else 0.0


def _partition_fusion_block(
    bounds: tuple[int, int, int, int, int, int],
    contributors: Sequence[tuple[int, tuple[int, int, int]]],
    tile_shape: tuple[int, int, int],
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
    """
    z0, z1, y0, y1, x0, x1 = bounds
    tile_z, tile_y, tile_x = tile_shape
    cuts = [{z0, z1}, {y0, y1}, {x0, x1}]
    for _, (oz, oy, ox) in contributors:
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
                    and offset[0] + tile_z >= rz1
                    and offset[1] <= ry0
                    and offset[1] + tile_y >= ry1
                    and offset[2] <= rx0
                    and offset[2] + tile_x >= rx1
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


def _select_registration_pairs(
    positions_zyx: Sequence[Sequence[float]],
    tile_shape_zyx: Sequence[int],
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
        Per-tile ZYX shape in voxels.
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
    tile_shape = np.asarray(tile_shape_zyx, dtype=np.float64)
    pixel_size = np.asarray(pixel_size_zyx, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_zyx must have shape (positions, 3)")
    if tile_shape.shape != (3,) or pixel_size.shape != (3,):
        raise ValueError("tile_shape_zyx and pixel_size_zyx must have length 3")

    physical_shape = tile_shape * pixel_size
    edges: list[tuple[float, int, int]] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            delta = positions[j] - positions[i]
            if is_2d:
                delta = delta.copy()
                delta[0] = 0.0
            if np.all(np.abs(delta) < physical_shape):
                distance = float(np.linalg.norm(delta / physical_shape))
                edges.append((distance, i, j))

    if not edges:
        return []

    parent = list(range(len(positions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> bool:
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


def _crop_registration_overlap(
    bounds_i: Sequence[tuple[int, int]],
    bounds_j: Sequence[tuple[int, int]],
    downsample_factors_zyx: Sequence[int],
    max_downsampled_shape_zyx: Sequence[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Center-crop corresponding overlap bounds to a bounded registration ROI.

    Both tiles are trimmed identically on every axis, preserving their physical
    correspondence while avoiding full-face reads that are immediately reduced
    for registration.

    Parameters
    ----------
    bounds_i, bounds_j
        Corresponding ZYX overlap bounds for the two tiles.
    downsample_factors_zyx
        Registration reduction factors in ZYX order.
    max_downsampled_shape_zyx
        Maximum reduced registration ROI shape in ZYX order.

    Returns
    -------
    tuple[list[tuple[int, int]], list[tuple[int, int]]]
        Cropped bounds for the two tiles.
    """
    if not (
        len(bounds_i)
        == len(bounds_j)
        == len(downsample_factors_zyx)
        == len(max_downsampled_shape_zyx)
        == 3
    ):
        raise ValueError("registration overlap arguments must all have length 3")

    cropped_i: list[tuple[int, int]] = []
    cropped_j: list[tuple[int, int]] = []
    for first, second, factor, target in zip(
        bounds_i,
        bounds_j,
        downsample_factors_zyx,
        max_downsampled_shape_zyx,
    ):
        first_length = int(first[1]) - int(first[0])
        second_length = int(second[1]) - int(second[0])
        if first_length != second_length or first_length <= 0:
            raise ValueError("paired registration overlap bounds must match")

        maximum_length = max(1, int(factor)) * max(1, int(target))
        retained_length = min(first_length, maximum_length)
        trim_before = (first_length - retained_length) // 2
        trim_after = first_length - retained_length - trim_before
        cropped_i.append((int(first[0]) + trim_before, int(first[1]) - trim_after))
        cropped_j.append((int(second[0]) + trim_before, int(second[1]) - trim_after))

    return cropped_i, cropped_j


def _deskew_geometry_crop_y(
    deskewed_z: int,
    angle_deg: float,
    voxel_size_zyx: Sequence[float],
    stored_y: int,
) -> int:
    """Return the rectangular Y crop implied by the oblique camera geometry."""
    if deskewed_z < 2 or stored_y < 1:
        raise ValueError("deskewed Z and stored Y dimensions must be positive")
    if len(voxel_size_zyx) != 3 or any(
        float(value) <= 0.0 for value in voxel_size_zyx
    ):
        raise ValueError("deskewed voxel size must contain three positive values")
    angle_rad = math.radians(float(angle_deg))
    if not 0.0 < abs(angle_rad) < math.pi / 2:
        raise ValueError("OPM angle must be strictly between 0 and 90 degrees")
    z_extent_in_y_pixels = (
        deskewed_z * float(voxel_size_zyx[0]) / float(voxel_size_zyx[1])
    )
    crop_y = int(math.ceil(z_extent_in_y_pixels / abs(math.tan(angle_rad))))
    if 2 * crop_y >= stored_y:
        raise ValueError(
            "deskew geometry crop would remove the entire stored Y dimension: "
            f"{crop_y} pixels per side from {stored_y}"
        )
    return crop_y


def _processing_step_applied(
    attributes: dict[str, Any],
    step_name: str,
) -> bool:
    """Return whether a named recorded processing step was already applied."""
    provenance = attributes.get("opm_processing")
    if not isinstance(provenance, dict):
        return False
    steps = provenance.get("steps")
    if not isinstance(steps, list):
        return False
    return any(
        isinstance(step, dict)
        and step.get("name") == step_name
        and bool(step.get("applied"))
        for step in steps
    )


@njit(parallel=True, nogil=True, cache=True)
def _mean_downsample_3d_cpu(
    source: np.ndarray,
    z_factor: int,
    y_factor: int,
    x_factor: int,
) -> np.ndarray:
    """Mean-downsample one ZYX array with a parallel bounded reduction."""
    z_size, y_size, x_size = source.shape
    out_z = (z_size + z_factor - 1) // z_factor
    out_y = (y_size + y_factor - 1) // y_factor
    out_x = (x_size + x_factor - 1) // x_factor
    output = np.empty((out_z, out_y, out_x), dtype=np.float32)
    output_size = out_z * out_y * out_x

    for output_index in prange(output_size):
        out_z_index = output_index // (out_y * out_x)
        remainder = output_index - out_z_index * out_y * out_x
        out_y_index = remainder // out_x
        out_x_index = remainder - out_y_index * out_x
        z_start = out_z_index * z_factor
        y_start = out_y_index * y_factor
        x_start = out_x_index * x_factor

        total = np.float32(0.0)
        count = 0
        for z in range(z_start, min(z_start + z_factor, z_size)):
            for y in range(y_start, min(y_start + y_factor, y_size)):
                for x in range(x_start, min(x_start + x_factor, x_size)):
                    total += np.float32(source[z, y, x])
                    count += 1
        output[out_z_index, out_y_index, out_x_index] = total / count

    return output


def _mean_downsample_registration(
    array: np.ndarray,
    factors: Sequence[int],
) -> np.ndarray:
    """Mean-downsample a 2D/3D registration patch without generic block reduction."""
    source_array = np.asarray(array)
    if source_array.ndim not in (2, 3):
        raise ValueError("registration patches must be 2D or 3D")
    if len(factors) != source_array.ndim or any(int(value) < 1 for value in factors):
        raise ValueError("downsampling factors must match the patch rank")

    was_2d = source_array.ndim == 2
    if was_2d:
        source_array = source_array[None, ...]
        factors_zyx = (1, int(factors[0]), int(factors[1]))
    else:
        factors_zyx = tuple(int(value) for value in factors)

    z_factor, y_factor, x_factor = factors_zyx
    reduced = _mean_downsample_3d_cpu(
        source_array,
        z_factor,
        y_factor,
        x_factor,
    )
    return reduced[0] if was_2d else reduced


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
        `{stem}_deskewed.ome.zarr`, `{stem}_projection.ome.zarr`, followed by
        their legacy `.zarr` names. Deconvolved products are used as fallbacks
        only when no corresponding non-deconvolved product exists.
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
    metrics_filename : str, default="stitching_metrics.json"
        File used to save/load pairwise registration links.
    channel_to_use : int, default=0
        Channel index used for registration.
    reverse_stage_y : bool, default=True
        Reverse stage-derived Y for tile placement without flipping tile pixels.
    crop_deskewed_trapezoid : bool, default=False
        Crop the geometry-derived trapezoidal Y ends from deskewed volumes before
        registration and fusion.
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
        metrics_filename: str = "stitching_metrics.json",
        channel_to_use: int = 0,
        multiscale_downsample: str = "stride",
        fusion_ram_fraction: float = 0.4,
        max_in_flight_writes: int = 2,
        chunk_shape_yx: tuple[int, int] = (1024, 1024),
        optimization_rel_threshold: float = 0.5,
        optimization_abs_threshold: float = 1.5,
        max_registration_shift_zyx: tuple[int, int, int] = (20, 50, 100),
        reverse_stage_y: bool = True,
        crop_deskewed_trapezoid: bool = False,
    ) -> None:
        """Initialize registration and fusion for a processed acquisition.

        Parameters
        ----------
        root_path
            Original acquisition path used to discover processed collections.
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
        metrics_filename
            Pairwise-registration metrics filename.
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
            Absolute residual threshold for registration outliers.
        max_registration_shift_zyx
            Maximum accepted registration correction in ZYX voxels.
        reverse_stage_y
            Whether to reverse stage-derived image-Y placement. Tile pixel
            arrays are not modified.
        crop_deskewed_trapezoid
            Whether to crop both Y ends using the OPM angle, deskewed Z extent,
            and voxel sizes before registration and fusion.

        Returns
        -------
        None
            No value is returned.
        """
        self.root = resolve_acquisition_path(root_path)
        base = self.root.parents[0]
        stem = acquisition_stem(self.root)

        candidates = (
            base / f"{stem}_deskewed.ome.zarr",
            base / f"{stem}_projection.ome.zarr",
            base / f"{stem}_deskewed.zarr",
            base / f"{stem}_projection.zarr",
            base / f"{stem}_decon_deskewed.ome.zarr",
            base / f"{stem}_decon_projection.ome.zarr",
            base / f"{stem}_decon_deskewed.zarr",
            base / f"{stem}_decon_projection.zarr",
        )
        data_path = next((path for path in candidates if path.exists()), None)
        if data_path is None:
            checked = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(
                f"Processed data store not found. Checked: {checked}"
            )
        self.data = data_path

        collection = open_position_collection(self.data)
        self._input_attributes = dict(collection.attributes)
        self.position_arrays = collection.arrays
        try:
            input_chunks = self.position_arrays[0].chunk_layout.read_chunk.shape
            self._input_chunk_zyx = tuple(int(value) for value in input_chunks[-3:])
        except (AttributeError, TypeError):
            self._input_chunk_zyx = tuple(int(value) for value in collection.shape[-3:])
        stage_positions = [
            position
            for _ in range(collection.shape[0])
            for position in collection.attributes["stage_positions"]
        ]
        self.reverse_stage_y = bool(reverse_stage_y)
        self._tile_positions = [
            tuple(position)
            for position in stage_positions_to_image_coordinates(
                stage_positions,
                reverse_y=self.reverse_stage_y,
            )
        ]
        self._pixel_size: tuple[float, float, float] = tuple(
            float(x) for x in collection.attributes["deskewed_voxel_size_um"]
        )

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
        if max_workers is None:
            max_workers = _default_fusion_workers()
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._max_workers = int(max_workers)
        self._debug = bool(debug)
        self.metrics_filename = str(metrics_filename)
        self._blend_pixels = tuple(int(x) for x in blend_pixels)
        self.channel_to_use = int(channel_to_use)

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
        if len(max_registration_shift_zyx) != 3 or any(
            value < 0 for value in max_registration_shift_zyx
        ):
            raise ValueError(
                "max_registration_shift_zyx must contain three nonnegative values"
            )
        self.max_registration_shift_zyx = tuple(
            int(value) for value in max_registration_shift_zyx
        )
        self.crop_deskewed_trapezoid = bool(crop_deskewed_trapezoid)

        (
            self.time_dim,
            self.position_dim,
            self.channels,
            self.z_dim,
            self.y_dim,
            self.x_dim,
        ) = collection.shape
        self._is_2d = self.z_dim == 1
        self._source_y_dim = int(self.y_dim)
        self._source_y_start = 0
        self.deskew_geometry_crop_y = 0
        if self.crop_deskewed_trapezoid:
            self._configure_deskew_geometry_crop()
        if not 0 <= self.channel_to_use < int(self.channels):
            raise ValueError(
                "channel_to_use must be between 0 and "
                f"{int(self.channels) - 1}; got {self.channel_to_use}"
            )

        self._update_profiles()

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

        Parameters
        ----------
        None
            This callable has no parameters.

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

    def _configure_deskew_geometry_crop(self) -> None:
        """Convert stored tiles to their rectangular geometry-cropped Y extent."""
        if _processing_step_applied(
            self._input_attributes,
            "crop_after_deskew",
        ):
            return

        if self._is_2d:
            raise ValueError(
                "--crop-ends requires a volumetric deskewed input"
            )
        angle_deg = self._input_attributes.get("opm_tilt_deg")
        if angle_deg is None:
            raise ValueError(
                "Cannot crop deskew geometry because acquisition metadata lacks "
                "the OPM angle"
            )

        crop_y = _deskew_geometry_crop_y(
            int(self.z_dim),
            float(angle_deg),
            self._pixel_size,
            int(self._source_y_dim),
        )
        self._source_y_start = crop_y
        self.deskew_geometry_crop_y = crop_y
        self.y_dim = int(self._source_y_dim) - 2 * crop_y

        leading_crop_um = crop_y * float(self._pixel_size[1])
        self._tile_positions = [
            (float(z), float(y) + leading_crop_um, float(x))
            for z, y, x in self._tile_positions
        ]

    def _source_y_slice(self, cropped_slice: slice) -> slice:
        """Map an effective cropped-tile Y slice to the stored source tile."""
        start, stop, step = cropped_slice.indices(int(self.y_dim))
        if step != 1:
            raise ValueError("fusion Y slices must use a unit positive step")
        return slice(
            start + int(self._source_y_start),
            stop + int(self._source_y_start),
        )

    def _update_profiles(self) -> None:
        """Recompute 1D feather profiles from blend_pixels and current data shape.

        Parameters
        ----------
        None
            This callable has no parameters.

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

        y_slice = self._source_y_slice(y_slice)
        if self._is_2d:
            arr = (
                self.position_arrays[pos_idx][t_idx, ch_sel, 0, y_slice, x_slice]
                .read()
                .result()
            )
            if arr.ndim == 2:
                arr = arr[None, None, :, :]
            elif arr.ndim == 3:
                arr = arr[:, None, :, :]
            else:
                raise ValueError(
                    f"Unexpected 2D tile ndim={arr.ndim}, shape={arr.shape}"
                )
            return arr if dtype is None else arr.astype(dtype, copy=False)

        arr = (
            self.position_arrays[pos_idx][t_idx, ch_sel, z_slice, y_slice, x_slice]
            .read()
            .result()
        )
        return arr if dtype is None else arr.astype(dtype, copy=False)

    @staticmethod
    def register_and_score(
        g1: Any,
        g2: Any,
        win_size: int,
        max_shift: Sequence[float] | None = None,
    ) -> tuple[tuple[float, float, float], float]:
        """Register `g2` to `g1` and compute an SSIM score.

        Steps:
        1) histogram-match g2 -> g1
        2) phase cross-correlation to estimate subpixel shift
        3) extract corresponding valid views at the nearest integer shift
        4) compute SSIM over that aligned overlap

        Parameters
        ----------
        g1 : array-like
            Fixed patch (2D YX or 3D ZYX).
        g2 : array-like
            Moving patch (same shape as g1).
        win_size : int
            SSIM window size.
        max_shift : sequence[float] or None
            Maximum permitted shift in downsampled patch coordinates. When an
            axis can wrap within this range, equivalent Fourier shifts are
            disambiguated by direct overlap correlation.

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

        data_range_1 = float(xp.max(arr1) - xp.min(arr1))
        data_range_2 = float(xp.max(arr2) - xp.min(arr2))
        if (
            not math.isfinite(data_range_1)
            or not math.isfinite(data_range_2)
            or data_range_1 <= 0.0
            or data_range_2 <= 0.0
        ):
            return (0.0, 0.0, 0.0), float("nan")

        arr2 = match_histograms(arr2, arr1)

        shift, _, _ = phase_cross_correlation(
            arr1,
            arr2,
            # Resolve only physically permitted wraparound candidates below.
            # cuCIM's generic path shifts the full volume and correlates every
            # wraparound quadrant, including candidates outside our bounds.
            disambiguate=False,
            normalization="phase",
            upsample_factor=10,
            overlap_ratio=0.5,
        )

        if USING_GPU and cp is not None:
            shift_array = cp.asnumpy(cp.asarray(shift, dtype=cp.float32))
        else:
            shift_array = np.asarray(shift, dtype=np.float32)
        shift_array = _bounded_disambiguate_shift(
            np.asarray(g1, dtype=np.float32),
            np.asarray(g2, dtype=np.float32),
            shift_array,
            max_shift,
        )

        arr1_score, arr2_score = _aligned_registration_views(
            arr1,
            arr2,
            shift_array,
        )
        # The custom CUDA SSIM kernels consume dense ZYX buffers. Cropped views
        # generally retain parent strides, so materialize them before scoring.
        arr1_score = xp.ascontiguousarray(arr1_score)
        arr2_score = xp.ascontiguousarray(arr2_score)
        score = _ssim(arr1_score, arr2_score, win_size=win_size)

        if arr1.ndim == 2 and len(shift_array) == 2:
            out_shift = np.asarray(
                [0.0, shift_array[0], shift_array[1]],
                dtype=np.float32,
            )
        else:
            out_shift = shift_array

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
        n_pos = int(self.position_dim)
        registration_pairs = _select_registration_pairs(
            self._tile_positions[:n_pos],
            (self.z_dim, self.y_dim, self.x_dim),
            self._pixel_size,
            is_2d=self._is_2d,
        )

        def bounds_1d(off: int, length: int) -> tuple[int, int]:
            """Clip a shifted interval to one array dimension.

            Parameters
            ----------
            off : int
                Value supplied for ``off``.
            length : int
                Value supplied for ``length``.

            Returns
            -------
            tuple[int, int]
                Result produced by the callable.
            """
            return max(0, off), min(length, off + length)

        df_zyx_base = (int(df_in[0]), int(df_in[1]), int(df_in[2]))
        if int(self.z_dim) == 1:
            df_zyx_base = (1, df_zyx_base[1], df_zyx_base[2])
        max_downsampled_shape_zyx = (
            min(64, int(self.z_dim)),
            min(512, int(self.y_dim)),
            min(512, int(self.x_dim)),
        )

        with ThreadPoolExecutor(max_workers=min(2, self._max_workers)) as executor:
            for t in range(int(self.time_dim)):
                base = t * n_pos

                for i_pos, j_pos in tqdm(
                    registration_pairs,
                    desc=f"register pairs t={t + 1}/{int(self.time_dim)}",
                    leave=False,
                    unit="pair",
                ):
                    i = base + i_pos
                    j = base + j_pos

                    phys = np.array(self._tile_positions[j]) - np.array(
                        self._tile_positions[i]
                    )
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
                    bounds_i, bounds_j = _crop_registration_overlap(
                        bounds_i,
                        bounds_j,
                        df_zyx_base,
                        max_downsampled_shape_zyx,
                    )
                    overlap_shape = tuple(hi - lo for lo, hi in bounds_i)
                    df_zyx_eff = tuple(
                        max(1, min(factor, length))
                        for factor, length in zip(df_zyx_base, overlap_shape)
                    )

                    def read_patch(
                        gidx: int, bnds: list[tuple[int, int]]
                    ) -> np.ndarray:
                        """Read one bounded tile patch for pairwise registration.

                        Parameters
                        ----------
                        gidx : int
                            Value supplied for ``gidx``.
                        bnds : list[tuple[int, int]]
                            Value supplied for ``bnds``.

                        Returns
                        -------
                        np.ndarray
                            Result produced by the callable.
                        """
                        z0, z1 = bnds[0]
                        y0, y1 = bnds[1]
                        x0, x1 = bnds[2]
                        return self._read_tile_volume(
                            gidx,
                            ch_idx,
                            slice(z0, z1),
                            slice(y0, y1),
                            slice(x0, x1),
                            dtype=None,
                        )

                    future_i = executor.submit(read_patch, i, bounds_i)
                    future_j = executor.submit(read_patch, j, bounds_j)
                    patch_i = future_i.result()
                    patch_j = future_j.result()

                    if self._is_2d:
                        patch_i = np.asarray(patch_i)[0, 0]
                        patch_j = np.asarray(patch_j)[0, 0]
                        reduction_factors = df_zyx_eff[1:]
                    else:
                        reduction_factors = df_zyx_eff
                    g1 = _mean_downsample_registration(
                        patch_i,
                        reduction_factors,
                    )
                    g2 = _mean_downsample_registration(
                        patch_j,
                        reduction_factors,
                    )

                    max_shift = self.max_registration_shift_zyx
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
                    )
                    if not math.isfinite(score) or not all(
                        math.isfinite(value) for value in shift_ds
                    ):
                        continue
                    score = float(max(score, 1e-6))

                    if th != 0.0 and score < th:
                        continue

                    dz_s = int(np.round(shift_ds[0] * df_zyx_eff[0]))
                    dy_s = int(np.round(shift_ds[1] * df_zyx_eff[1]))
                    dx_s = int(np.round(shift_ds[2] * df_zyx_eff[2]))

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

                    self.pairwise_metrics[(i, j)] = (
                        dz_s,
                        dy_s,
                        dx_s,
                        round(score, 3),
                    )

    @staticmethod
    def _solve_global(
        links: list[dict[str, Any]], n_tiles: int, fixed_indices: list[int]
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
        """Perform robust two-round (optionally iterative) optimization.

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
            """Calculate Euclidean residuals for registration links.

            Parameters
            ----------
            ls : list[dict[str, Any]]
                Value supplied for ``ls``.
            sh : np.ndarray
                Value supplied for ``sh``.

            Returns
            -------
            np.ndarray
                Result produced by the callable.
            """
            return np.array(
                [
                    np.linalg.norm(sh[link["j"]] - sh[link["i"]] - link["t"])
                    for link in ls
                ],
                dtype=np.float64,
            )

        work = links.copy()
        res = residuals(work, shifts)
        if len(res) == 0:
            return shifts
        cutoff = max(abs_thresh, rel_thresh * float(np.median(res)))
        outliers = set(np.where(res > cutoff)[0])

        if iterative:
            while outliers:
                for k in sorted(outliers, reverse=True):
                    work.pop(k)
                if not work:
                    shifts = self._solve_global(work, n_tiles, fixed_indices)
                    break
                shifts = self._solve_global(work, n_tiles, fixed_indices)
                res = residuals(work, shifts)
                if len(res) == 0:
                    break
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
        """Optimize global shifts independently per timepoint, anchoring tile 0.

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
        """Save pairwise link metrics to JSON.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Output JSON path.

        Returns
        -------
        None
        """
        path = Path(filepath)
        out = {
            "schema_version": 10,
            "registration": {
                "channel_index": self.channel_to_use,
                "stage_y_reversed": self.reverse_stage_y,
                "pair_selection": "connected-nearest-overlap-v1",
                "overlap_roi": "centered-64x512x512-v1",
                "downsample": "bounded-numba-mean-v1",
                "phase_disambiguation": "bounded-direct-overlap-v1",
                "crop_deskewed_trapezoid": bool(
                    getattr(self, "crop_deskewed_trapezoid", False)
                ),
                "deskew_geometry_crop_y": int(
                    getattr(self, "deskew_geometry_crop_y", 0)
                ),
            },
            "pairwise_metrics": {
                f"{i},{j}": list(v) for (i, j), v in self.pairwise_metrics.items()
            },
        }
        with open(path, "w") as f:
            json.dump(out, f)

    def load_pairwise_metrics(self, filepath: str | Path) -> None:
        """Load pairwise link metrics from JSON.

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

        if not isinstance(data, dict):
            raise ValueError(f"{path} does not contain a registration metrics object")

        if "pairwise_metrics" not in data:
            legacy_metrics = data
            invalid_count = self._count_invalid_pairwise_metrics(legacy_metrics)
            if invalid_count:
                raise ValueError(
                    f"{path} contains {invalid_count} invalid registration metrics"
                )
            raise ValueError(
                f"{path} uses an unscoped legacy registration cache; "
                "registration must be recomputed"
            )

        expected_registration = {
            "channel_index": self.channel_to_use,
            "stage_y_reversed": self.reverse_stage_y,
            "pair_selection": "connected-nearest-overlap-v1",
            "overlap_roi": "centered-64x512x512-v1",
            "downsample": "bounded-numba-mean-v1",
            "phase_disambiguation": "bounded-direct-overlap-v1",
            "crop_deskewed_trapezoid": bool(
                getattr(self, "crop_deskewed_trapezoid", False)
            ),
            "deskew_geometry_crop_y": int(
                getattr(self, "deskew_geometry_crop_y", 0)
            ),
        }
        if data.get("schema_version") != 10:
            raise ValueError(f"{path} has an unsupported registration cache schema")
        if data.get("registration") != expected_registration:
            raise ValueError(
                f"{path} was created for different registration settings; "
                "registration must be recomputed"
            )

        raw_metrics = data["pairwise_metrics"]
        if not isinstance(raw_metrics, dict):
            raise ValueError(f"{path} contains invalid registration metrics")

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
                f"{path} contains {invalid_count} invalid registration metrics"
            )
        self.pairwise_metrics = metrics

    @staticmethod
    def _count_invalid_pairwise_metrics(data: object) -> int:
        """Count malformed or nonfinite pairwise registration records."""
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

        Parameters
        ----------
        None
            This callable has no parameters.

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

    def _pad_to_multiscale_multiple(self) -> None:
        """Minimally pad the fused shape for integer multiscale levels.

        Parameters
        ----------
        None
            This callable has no parameters.

        Returns
        -------
        None
            Sets `self.padded_shape` as (Z, Y, X).
        """
        if self.unpadded_shape is None:
            raise RuntimeError("unpadded_shape not computed.")

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
            Value supplied for ``output path``.
        z_slices_per_write : int or None
            Z depth rendered per scale-0 block. By default, this matches the
            input Zarr chunk depth so source chunks are not decompressed again
            for successive shallow slabs.

        Returns
        -------
        tuple[ts.TensorStore, list[int]]
            Result produced by the callable.
        """
        if self.padded_shape is None:
            raise RuntimeError("padded_shape not computed.")
        if self.offset_um is None:
            raise RuntimeError("offset_um not computed.")

        factors = (1, *self.multiscale_factors)
        dz, dy, dx = self._pixel_size
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
            translation = [
                0,
                0,
                float(self.offset_um[0] + 0.5 * (z_factor - 1) * dz),
                float(self.offset_um[1] + 0.5 * (factor - 1) * dy),
                float(self.offset_um[2] + 0.5 * (factor - 1) * dx),
            ]
            datasets.append(
                {
                    "path": str(level),
                    "coordinateTransformations": [
                        {"scale": [1, 1, dz * z_factor, dy * factor, dx * factor]},
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
                    np.uint16,
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
        extra_attributes = None
        if bool(getattr(self, "crop_deskewed_trapezoid", False)):
            extra_attributes = {
                "opm_fusion": {
                    "deskew_geometry_crop": {
                        "applied": bool(self.deskew_geometry_crop_y),
                        "requested": True,
                        "axis": "y",
                        "pixels_per_side": int(self.deskew_geometry_crop_y),
                    }
                }
            }
        _, self._multiscale_arrays = prepare_image(
            output_path,
            image,
            specs,
            extra_attributes=extra_attributes,
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
        """Choose an input-aware Z depth that still permits aligned Y/X blocks."""
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
            Value supplied for ``z depth``.
        y_size : int
            Value supplied for ``y size``.
        x_size : int
            Value supplied for ``x size``.

        Returns
        -------
        tuple[int, int]
            Result produced by the callable.
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
        base: int,
        bounds: tuple[int, int, int, int, int, int],
        contributors: Sequence[tuple[int, tuple[int, int, int]]],
    ) -> tuple[Any, np.ndarray, int, int]:
        """Render one independent output block without writing it.

        Each contributing tile is read at most once. Exclusive regions are copied
        into the integer output, while only regions with multiple contributors use
        float32 feather accumulation.
        """
        z0, z1, y0, y1, x0, x1 = bounds
        tile_shape = (int(self.z_dim), int(self.y_dim), int(self.x_dim))
        regions = _partition_fusion_block(bounds, contributors, tile_shape)

        if len(regions) == 1 and len(regions[0][1]) == 1:
            region, region_contributors = regions[0]
            rz0, rz1, ry0, ry1, rx0, rx1 = region
            p, (oz, oy, ox) = region_contributors[0]
            direct = self._read_tile_volume(
                base + p,
                slice(None),
                slice(rz0 - oz, rz1 - oz),
                slice(ry0 - oy, ry1 - oy),
                slice(rx0 - ox, rx1 - ox),
                dtype=np.uint16,
            )
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
            tz0, tz1 = max(z0, oz), min(z1, oz + tile_shape[0])
            ty0, ty1 = max(y0, oy), min(y1, oy + tile_shape[1])
            tx0, tx1 = max(x0, ox), min(x1, ox + tile_shape[2])
            source = self._read_tile_volume(
                base + p,
                slice(None),
                slice(tz0 - oz, tz1 - oz),
                slice(ty0 - oy, ty1 - oy),
                slice(tx0 - ox, tx1 - ox),
                dtype=None,
            )
            source_blocks[p] = ((tz0, ty0, tx0), source)

        output = np.zeros(
            (int(self.channels), z1 - z0, y1 - y0, x1 - x0),
            dtype=np.uint16,
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
                output[output_selection] = source[
                    :,
                    rz0 - sz0 : rz1 - sz0,
                    ry0 - sy0 : ry1 - sy0,
                    rx0 - sx0 : rx1 - sx0,
                ]
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
            weight = np.zeros(fused.shape[1:], dtype=np.float32)
            for p, (oz, oy, ox) in region_contributors:
                (sz0, sy0, sx0), source = source_blocks[p]
                sub = np.ascontiguousarray(
                    source[
                        :,
                        rz0 - sz0 : rz1 - sz0,
                        ry0 - sy0 : ry1 - sy0,
                        rx0 - sx0 : rx1 - sx0,
                    ]
                )
                local_z = slice(rz0 - oz, rz1 - oz)
                local_y = slice(ry0 - oy, ry1 - oy)
                local_x = slice(rx0 - ox, rx1 - ox)
                _accumulate_tile_block(
                    fused,
                    weight,
                    sub,
                    self.z_profile[local_z],
                    self.y_profile[local_y],
                    self.x_profile[local_x],
                    0,
                    0,
                    0,
                )

            _normalize_block(fused, weight)
            output[output_selection] = fused.astype(np.uint16)
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

        Parameters
        ----------
        None
            This callable has no parameters.

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

        n_pos = int(self.position_dim)
        n_t = int(self.time_dim)

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
            """Issue one bounded asynchronous output write."""
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
            """Queue completed block writes and update observable statistics."""
            nonlocal direct_regions, blended_regions
            for future in done:
                selection, value, direct_count, blended_count = future.result()
                queue_write(selection, value)
                direct_regions += int(direct_count)
                blended_regions += int(blended_count)
                chunk_bar.update(int(self.channels))

        with ThreadPoolExecutor(max_workers=render_workers) as render_executor:
            for t in range(n_t):
                base = t * n_pos

                offsets_t: list[tuple[int, int, int]] = []
                for p in range(n_pos):
                    z_um, y_um, x_um = self._tile_positions[base + p]
                    oz = int(np.round((z_um - off_z_um) / dz_um))
                    oy = int(np.round((y_um - off_y_um) / dy_um))
                    ox = int(np.round((x_um - off_x_um) / dx_um))
                    offsets_t.append((oz, oy, ox))

                for z0, z1, block_y, block_x in block_layout:
                    for y0 in range(0, pad_y, block_y):
                        y1 = min(y0 + block_y, pad_y)
                        for x0 in range(0, pad_x, block_x):
                            x1 = min(x0 + block_x, pad_x)
                            overlapping = [
                                (p, offset)
                                for p, offset in enumerate(offsets_t)
                                if offset[0] + int(self.z_dim) > z0
                                and offset[0] < z1
                                and offset[1] + int(self.y_dim) > y0
                                and offset[1] < y1
                                and offset[2] + int(self.x_dim) > x0
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
                                    base,
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

        Parameters
        ----------
        None
            This callable has no parameters.

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
                """Read, downsample, and write one output-aligned block."""
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
                        np.uint16,
                        copy=False,
                    )
                ).result()

            workers = max(1, int(self._max_workers))
            pending: set[Future[Any]] = set()

            def collect_completed(done: set[Future[Any]]) -> None:
                """Propagate worker failures and advance the transient bar."""
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

        Parameters
        ----------
        None
            This callable has no parameters.

        Returns
        -------
        None
            Writes the fused NGFF store to:
            `{base}/{stem}_fused.ome.zarr`

        Raises
        ------
        RuntimeError
            If required intermediate computations fail.
        """
        base = self.root.parents[0]
        metrics_path = base / self.metrics_filename

        try:
            self.load_pairwise_metrics(metrics_path)
        except (FileNotFoundError, ValueError):
            self.refine_tile_positions_with_cross_correlation(
                downsample_factors=self.downsample_factors,
                ch_idx=self.channel_to_use,
                threshold=self.threshold,
            )
            self.save_pairwise_metrics(metrics_path)

        self.optimize_shifts(
            method="TWO_ROUND_ITERATIVE",
            rel_thresh=self.optimization_rel_threshold,
            abs_thresh=self.optimization_abs_threshold,
        )

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
        self._pad_to_multiscale_multiple()

        omezarr = base / f"{acquisition_stem(self.root)}_fused.ome.zarr"
        self.fused_ts, self.write_block_shape = self._prepare_fused_image(omezarr)
        self._fuse_by_blocks()
        self._write_multiscales()
