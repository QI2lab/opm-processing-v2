"""Calibrated tissue previews and representative saved AO correction locations.

Pupils identify corrections at the centers of their assigned source-tile
groups. They do not mark calibrated AO measurement sites. Source assignments use
recorded mappings or saved physical stage plans; registered image placements
come from the repaired processing metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from yaozarrs import open_group

from opm_processing.dataio.acquisition import inspect_acquisition
from opm_processing.dataio.ao_results import ao_tile_assignments, read_ao_stage_plan
from opm_processing.dataio.ngff import SPATIAL_DECIMALS
from opm_processing.dataio.processing_state import (
    ProcessingState,
    image_processing_state_path,
)
from opm_processing.dataio.roi import projection_tile_footprints
from opm_processing.export_rois import resolve_roi_source
from opm_processing.imageprocessing.coordinates import stage_z_level_indices

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class TissueGeometry:
    """A calibrated preview and correction symbols in physical image coordinates.

    ``extent_yx_um`` contains the selected preview's pixel-edge bounds, ordered
    y0, y1, x0, x1. Its origin/scale identify pixel centers. ``shape_yx``,
    ``origin_yx_um``, and ``scale_yx_um`` describe level zero. Center dictionaries
    contain physical YX coordinates keyed by logical depth and AO result index.
    """

    image: np.ndarray
    shape_yx: tuple[int, int]
    origin_yx_um: tuple[float, float]
    scale_yx_um: tuple[float, float]
    preview_origin_yx_um: tuple[float, float]
    preview_scale_yx_um: tuple[float, float]
    extent_yx_um: tuple[float, float, float, float]
    centers_by_depth_index: dict[int, dict[int, tuple[float, float]]]
    depth_offsets: dict[int, float | None]
    provenance: dict[str, Any]
    planned_centers_by_depth_index: dict[int, dict[int, tuple[float, float]]] = field(
        default_factory=dict
    )
    planned_grid_by_depth_index: dict[int, dict[str, list[float]]] = field(
        default_factory=dict
    )
    images_by_depth: dict[int, np.ndarray] = field(default_factory=dict)
    display_limits_by_depth: dict[int, tuple[float, float]] = field(
        default_factory=dict
    )
    layer_projection_rois_by_depth: dict[int, dict[str, Any]] = field(
        default_factory=dict
    )


def _pixel_edge_extent(
    shape_yx: Sequence[int],
    origin_yx_um: Sequence[float],
    scale_yx_um: Sequence[float],
) -> tuple[float, float, float, float]:
    """Convert calibrated first-pixel centers to physical YX edge bounds."""
    origin, scale = np.asarray(origin_yx_um), np.asarray(scale_yx_um)
    lower = origin - 0.5 * scale
    upper = origin + (np.asarray(shape_yx) - 0.5) * scale
    return float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1])


def _compose_transform(
    dataset: dict[str, Any], multiscale: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Compose ordered dataset and enclosing NGFF diagonal affine transforms."""
    scale, origin = np.ones(5, dtype=float), np.zeros(5, dtype=float)
    for transform in (
        *dataset.get("coordinateTransformations", []),
        *multiscale.get("coordinateTransformations", []),
    ):
        kind = transform["type"]
        if kind == "identity":
            continue
        if kind not in ("scale", "translation"):
            raise ValueError(f"Unsupported tissue coordinate transformation: {kind}")
        values = np.asarray(transform[kind], dtype=float)
        if values.shape != (5,) or not np.isfinite(values).all():
            raise ValueError("Tissue transforms must contain five finite TCZYX values")
        if kind == "scale":
            scale *= values
            origin *= values
        else:
            origin += values
    if not np.isfinite(scale).all() or np.any(scale[-3:] <= 0):
        raise ValueError("Tissue preview requires positive calibrated spatial sampling")
    return scale, origin


def _saved_layer(
    root: Path, selected: Sequence[Any], timepoint: int, depth: int
) -> Path:
    """Locate the exact saved stage plans associated with one AO result layer."""
    layers = set()
    for region in selected:
        result = getattr(region, "result", None)
        path = getattr(result, "zarr_path", None)
        if path is not None:
            candidate = Path(path).expanduser().resolve().parent.parent
            if (candidate / "ao_stage_positions.json").is_file():
                layers.add(candidate)
    if not layers:
        directory = root.parent if root.name.endswith(".zarr") else root
        if (directory / "ao_stage_positions.json").is_file():
            layers.add(directory)
        else:
            candidates = (
                *directory.glob(f"time_{timepoint}_z_{depth}_ao_grid_results"),
                *directory.glob(
                    f"*_ao_grid_results/time_{timepoint}_z_{depth}_ao_grid_results"
                ),
            )
            layers.update(path.resolve() for path in candidates if path.is_dir())
    if len(layers) != 1:
        raise ValueError(f"Expected one saved AO stage plan at depth {depth}")
    layer = layers.pop()
    if not (layer / "exp_stage_positions.json").is_file():
        raise ValueError(f"Missing experiment stage plan in {layer}")
    return layer


def _intended_geometry(
    ao_xy: np.ndarray,
    indices: Sequence[int],
    grid_shape_yx: Sequence[int] | None,
    extent_yx_um: tuple[float, float, float, float],
) -> tuple[dict[int, tuple[float, float]], dict[str, list[float]]]:
    """Center saved corrections in equal intended cells across the full canvas.

    Saved stage coordinate ranks retain correction identity and orientation.
    Their physical spacing and absolute coordinates do not determine this
    illustration of intended coverage.
    """
    if grid_shape_yx is None:
        raise ValueError("Intended AO grid requires recorded requested grid dimensions")
    rows, columns = grid_shape_yx
    unique_x, unique_y = np.unique(ao_xy[:, 0]), np.unique(ao_xy[:, 1])
    if (
        (len(unique_x), len(unique_y)) != (rows, columns)
        or len(ao_xy) != rows * columns
        or len(np.unique(ao_xy, axis=0)) != len(ao_xy)
        or sorted(indices) != list(range(rows * columns))
    ):
        raise ValueError(
            "Saved AO corrections must match the recorded requested Cartesian grid"
        )
    y0, y1, x0, x1 = extent_yx_um
    height, width = y1 - y0, x1 - x0
    centers = {}
    for index in indices:
        scan_rank = int(np.searchsorted(unique_x, ao_xy[index, 0]))
        tile_rank = int(np.searchsorted(unique_y, ao_xy[index, 1]))
        y = y1 - (scan_rank + 0.5) * height / rows
        x = x0 + (tile_rank + 0.5) * width / columns
        centers[index] = (round(y, 3), round(x, 3))
    grid = {
        "horizontal_y_um": [
            round(float(value), 3) for value in np.linspace(y0, y1, rows + 1)[1:-1]
        ],
        "vertical_x_um": [
            round(float(value), 3) for value in np.linspace(x0, x1, columns + 1)[1:-1]
        ],
    }
    return centers, grid


def read_tissue_geometry(
    root: Path,
    projection_path: Path | None,
    timepoint: int,
    channel: int,
    regions: Sequence[Any],
    *,
    planned_grid: bool = False,
    full_resolution: bool = False,
    load_image: bool = True,
) -> TissueGeometry:
    """Read calibrated tissue and representative historical correction centers.

    Parameters
    ----------
    root : pathlib.Path
        Acquisition directory, store, or saved AO results directory.
    projection_path : pathlib.Path or None
        Explicit registered fused maximum-Z image, or automatic discovery.
    timepoint, channel : int
        Tissue image indices. The AO plans must belong to the same timepoint.
    regions : sequence
        Saved AO results with ``depth`` and ``index`` attributes. A native
        result's ``zarr_path`` identifies its stage-plan directory when present.
    planned_grid : bool
        Also compute equal intended cells across the full calibrated canvas,
        using the recorded requested dimensions and saved correction ordering.
    full_resolution : bool
        Read the finest stored image level at native sampling. Otherwise select
        a calibrated pyramid level whose largest dimension is at most 1800.
    load_image : bool
        Load the shared maximum projection. False reads only its calibration
        when separate local layer projections will supply the pixels.

    Returns
    -------
    TissueGeometry
        Raw float32 preview pixels, exact level-specific calibration,
        representative assigned-tile centers, and saved-plan provenance.

    Raises
    ------
    ValueError
        Tissue calibration, saved plans, or historical tile correspondence
        cannot be established unambiguously.
    """
    root = Path(root).expanduser().resolve()
    if not regions:
        raise ValueError("A tissue overlay requires at least one saved AO result")
    discovery = root
    if projection_path is None and root.name.endswith("_ao_grid_results"):
        discovery = root.parent
    preview = resolve_roi_source(projection_path or discovery)
    suffix = "_max_z_fused.ome.zarr"
    if not preview.name.endswith(suffix):
        raise ValueError("AO tissue overlays require a fused maximum-Z OME-Zarr image")
    group = open_group(preview)
    image_metadata = group.attrs.get("ome", {})
    multiscales = image_metadata.get("multiscales", [])
    if len(multiscales) != 1:
        raise ValueError("Expected one calibrated tissue multiscale image")
    multiscale = multiscales[0]
    axes = multiscale["axes"]
    if [axis["name"] for axis in axes] != list("tczyx") or any(
        axis.get("unit") != "micrometer" for axis in axes[-3:]
    ):
        raise ValueError("Tissue preview requires TCZYX spatial axes in micrometers")
    levels = []
    for dataset in multiscale["datasets"]:
        array = group[dataset["path"]].to_tensorstore()
        shape = tuple(int(value) for value in array.shape)
        if len(shape) != 5 or shape[2] != 1:
            raise ValueError(
                "Tissue preview must have a singleton maximum-projection Z"
            )
        scale, origin = _compose_transform(dataset, multiscale)
        levels.append((dataset, array, shape, scale, origin))
    if not levels:
        raise ValueError("Tissue image has no pyramid datasets")
    base = min(levels, key=lambda level: float(np.prod(level[3][-2:])))
    suitable = [level for level in levels if max(level[2][-2:]) <= 1800]
    if full_resolution:
        selected = base
    elif suitable:
        selected = min(suitable, key=lambda level: float(np.prod(level[3][-2:])))
    else:
        selected = max(levels, key=lambda level: float(np.prod(level[3][-2:])))
    dataset, array, shape, scale, origin = selected
    if not 0 <= timepoint < shape[0] or not 0 <= channel < shape[1]:
        raise ValueError("Requested tissue timepoint or channel is outside the image")
    image = (
        np.asarray(array[timepoint, channel, 0].read().result(), dtype=np.float32)
        if load_image
        else np.empty((0, 0), dtype=np.float32)
    )
    extent = tuple(
        round(value, 3)
        for value in _pixel_edge_extent(shape[-2:], origin[-2:], scale[-2:])
    )

    state_path = image_processing_state_path(preview, preview.name.removesuffix(suffix))
    state = ProcessingState.read(state_path)
    acquisition = inspect_acquisition(state.source_path())
    ao_configuration = (
        open_group(acquisition.path)
        .attrs.get("opm_v2", {})
        .get("configuration", {})
        .get("acq_config", {})
        .get("AO", {})
    )
    requested_grid_shape = None
    requested_grid_source = None
    dimensions = [
        ao_configuration.get("num_scan_positions"),
        ao_configuration.get("num_tile_positions"),
    ]
    if all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in dimensions
    ):
        requested_grid_shape = dimensions
        requested_grid_source = {
            "path": str(acquisition.path / "zarr.json"),
            "rows_y_key": "attributes.opm_v2.configuration.acq_config.AO.num_scan_positions",
            "columns_x_key": "attributes.opm_v2.configuration.acq_config.AO.num_tile_positions",
        }
    stage = np.asarray(acquisition.stage_positions_zxy, dtype=float)
    depths = stage_z_level_indices(stage)
    footprints = projection_tile_footprints(preview)
    centers: dict[int, dict[int, tuple[float, float]]] = {}
    planned_centers: dict[int, dict[int, tuple[float, float]]] = {}
    planned_grids: dict[int, dict[str, list[float]]] = {}
    planned_supported = requested_grid_shape is not None
    if planned_grid and not planned_supported:
        raise ValueError("Intended AO grid requires recorded requested grid dimensions")
    depth_offsets: dict[int, float | None] = {}
    plan_records = []
    depth_records = {}
    for depth in sorted({int(region.depth) for region in regions}):
        selected_regions = [region for region in regions if int(region.depth) == depth]
        indices = [int(region.index) for region in selected_regions]
        if len(set(indices)) != len(indices):
            raise ValueError(f"Duplicate saved AO indices at depth {depth}")
        layer = _saved_layer(root, selected_regions, timepoint, depth)
        ao_plan, ao_provenance = read_ao_stage_plan(
            layer / "ao_stage_positions.json", depth
        )
        experiment, experiment_provenance = read_ao_stage_plan(
            layer / "exp_stage_positions.json", depth
        )
        plan_records.extend((ao_provenance, experiment_provenance))
        assigned, geometry_source = ao_tile_assignments(
            layer, acquisition.stage_positions_zxy, ao_plan, experiment
        )
        if geometry_source is not None:
            plan_records.append(geometry_source)
        if min(indices) < 0 or max(indices) >= len(ao_plan):
            raise ValueError(
                f"Saved AO indices do not identify the stage plan at depth {depth}"
            )
        ao_xy = np.asarray([[point["x"], point["y"]] for point in ao_plan], dtype=float)
        planned_reference = {}
        if planned_grid:
            planned_centers[depth], planned_grids[depth] = _intended_geometry(
                ao_xy, indices, requested_grid_shape, extent
            )
            planned_reference = {
                "planned_centers_yx_um": planned_centers[depth],
                "planned_grid_boundaries": planned_grids[depth],
            }
        depth_tiles = [
            tile
            for tile in footprints
            if int(tile["time_index"]) == timepoint
            and depths[int(tile["position_index"])] == depth
        ]
        if not depth_tiles:
            raise ValueError(f"No tissue source tiles identify AO depth {depth}")
        groups: dict[int, list[dict[str, Any]]] = {index: [] for index in indices}
        assignment_records = []
        for tile in depth_tiles:
            position = int(tile["position_index"])
            if position not in assigned:
                raise ValueError(
                    f"Source tile {position} does not match one saved stage plan at depth {depth}"
                )
            region_index = assigned[position]
            assignment_records.append({**tile, "region_index": region_index})
            if region_index in groups:
                groups[region_index].append(tile)
        centers[depth] = {}
        counts = {}
        for index, tiles in groups.items():
            counts[index] = len(tiles)
            if not tiles:
                continue
            tile_centers = np.asarray(
                [
                    ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2)
                    for bounds in (tile["bounds_yx_um"] for tile in tiles)
                ]
            )
            center = tile_centers.mean(axis=0)
            centers[depth][index] = tuple(round(float(value), 3) for value in center)
        offsets = {
            float(point["sample_depth_um"])
            for point in experiment
            if "sample_depth_um" in point
        }
        if len(offsets) > 1 or any(not np.isfinite(offset) for offset in offsets):
            raise ValueError(
                f"Saved stage plan has inconsistent sample depths at depth {depth}"
            )
        depth_offsets[depth] = offsets.pop() if offsets else None
        depth_records[depth] = {
            **planned_reference,
            "assignment": "recorded ao_grid_geometry.json tile_ao_indices; entire tile"
            if geometry_source is not None
            else "inferred from saved acquisition plans; nearest physical stage XY",
            "assigned_tile_counts": counts,
            "unused_region_indices": [
                index for index, count in counts.items() if not count
            ],
            "assignments": assignment_records,
        }

    return TissueGeometry(
        image=image,
        shape_yx=tuple(int(value) for value in base[2][-2:]),
        origin_yx_um=tuple(round(float(value), 3) for value in base[4][-2:]),
        scale_yx_um=tuple(float(value) for value in base[3][-2:]),
        preview_origin_yx_um=tuple(round(float(value), 3) for value in origin[-2:]),
        preview_scale_yx_um=tuple(float(value) for value in scale[-2:]),
        extent_yx_um=extent,
        centers_by_depth_index=centers,
        depth_offsets=depth_offsets,
        planned_centers_by_depth_index=planned_centers,
        planned_grid_by_depth_index=planned_grids,
        provenance={
            "projection": str(preview),
            "source_acquisition": str(acquisition.path),
            "processing_state": str(state_path),
            "requested_ao_grid_shape_yx": requested_grid_shape,
            "requested_ao_grid_shape_source": requested_grid_source,
            "coordinate_version": group.attrs.get("opm_processing_coordinates", {}).get(
                "version"
            ),
            "timepoint": int(timepoint),
            "channel": int(channel),
            "full_resolution": bool(full_resolution),
            "native_shape_yx": list(base[2][-2:]),
            "native_scale_yx_um": [float(value) for value in base[3][-2:]],
            "native_origin_yx_um": [round(float(value), 3) for value in base[4][-2:]],
            "preview_dataset": dataset["path"],
            "preview_shape_yx": list(shape[-2:]),
            "shared_projection_loaded": bool(load_image),
            "preview_scale_tczyx": scale.tolist(),
            "preview_origin_tczyx_um": [
                *origin[:2].tolist(),
                *(round(float(value), 3) for value in origin[-3:]),
            ],
            "preview_extent_yx_um": list(extent),
            "placement": "mean centers of registered source tiles assigned to each AO correction",
            "pupil_location": "representative correction identity; not an AO measurement landmark",
            "assignment_rule": "recorded tile assignments when available; otherwise historical nearest physical stage XY",
            "assignment_verification": "source positions matched to saved plans at recorded coordinate precision; no per-frame mirror telemetry",
            "planned_placement": "intended equal cells across the full acquisition canvas; pupils centered in their intended cells, not recorded sampling sites",
            "planned_grid_definition": "equal physical subdivisions of the calibrated tissue canvas using the recorded requested grid dimensions",
            "planned_index_orientation": "saved physical scan X ranks ascend bottom to top; saved physical stage Y ranks ascend left to right",
            "planned_placement_supported": planned_supported,
            "planned_grid_requested": bool(planned_grid),
            "planned_placement_limitation": (
                None
                if planned_supported
                else "Intended display requires recorded requested grid dimensions"
            ),
            "stage_plan_coordinate_decimals": SPATIAL_DECIMALS,
            "stage_plans": plan_records,
            "depths": depth_records,
        },
    )
