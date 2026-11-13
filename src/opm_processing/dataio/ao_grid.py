"""Registered acquisition-polygon Grid-AO geometry shared with SJS plotting."""

from __future__ import annotations

import copy
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from opm_processing.dataio.acquisition import inspect_acquisition
from opm_processing.dataio.ao_results import (
    AOOptimizationResult,
    read_ao_layers,
    read_ao_stage_plan,
)
from opm_processing.dataio.position_collection import open_position_collection
from opm_processing.dataio.roi import (
    _level_zero_grid,
    _registered_projection_context,
    projection_tile_footprints,
)
from opm_processing.imageprocessing.coordinates import (
    STAGE_COORDINATE_VERSION,
    stage_positions_to_image_coordinates,
    stage_z_level_indices,
)

if TYPE_CHECKING:
    from pathlib import Path


def read_ao_grid(source: Path, timepoint: int, ao_root: Path | None = None) -> dict:
    """Read the approved intended grid in the registered preview's world frame.

    Parameters
    ----------
    source : pathlib.Path
        Registered full-acquisition maximum-Z preview.
    timepoint : int
        Displayed time. Use the latest saved AO plan at its first depth.
    ao_root : pathlib.Path or None
        Explicit AO results root, otherwise use normal acquisition discovery.

    Returns
    -------
    dict
        Equal-area polygon cells, clipped grid segments and intended center stars.
    """
    from yaozarrs import open_group

    state, processed, _ = _registered_projection_context(source)
    acquisition = inspect_acquisition(state.source_path())
    layers = read_ao_layers(acquisition, source, ao_root)
    available = [key for key in layers if key[0] <= timepoint]
    if not available:
        raise ValueError("The Grid-AO overlay requires a saved AO plan")
    depth = min(key[1] for key in available)
    saved_time, _ = max(key for key in available if key[1] == depth)
    layer = layers[saved_time, depth][0]
    ao, _ = read_ao_stage_plan(layer / "ao_stage_positions.json", depth)
    experiment, _ = read_ao_stage_plan(layer / "exp_stage_positions.json", depth)
    if acquisition.mode != "stage" or any(acquisition.stage_axis_flips_xyz):
        raise ValueError(
            "The approved AO polygon overlay requires an unflipped stage scan"
        )
    if any("lab_scan_um" not in point for point in experiment):
        raise ValueError("The AO experiment plan must identify lab_scan_um")
    offsets = np.array([point["x"] - point["lab_scan_um"] for point in experiment])
    if np.ptp(offsets) > 0.001:
        raise ValueError("The AO layer has a spatially varying stage-to-scan offset")
    scan_offset = float(np.median(offsets))
    collection = open_position_collection(processed)
    coordinates = collection.attributes["opm_processing_coordinates"]
    if coordinates["version"] != STAGE_COORDINATE_VERSION:
        raise ValueError("The AO overlay requires current stage-endpoint coordinates")
    placement = np.asarray(coordinates["placement_stage_positions_zxy_um"])
    nominal = stage_positions_to_image_coordinates(placement) + np.asarray(
        collection.spatial_origins_zyx_um
    )
    depths = stage_z_level_indices(acquisition.stage_positions_zxy)
    footprints = [
        tile
        for tile in projection_tile_footprints(source)
        if tile["time_index"] == timepoint and depths[tile["position_index"]] == depth
    ]
    if not footprints:
        raise ValueError("No source tiles at the displayed time and AO depth")
    config = open_group(acquisition.path).attrs["opm_v2"]["configuration"]
    shape = [
        int(config["acq_config"]["AO"][name])
        for name in ("num_scan_positions", "num_tile_positions")
    ]
    targets = np.asarray([[-(point["x"] - scan_offset), point["y"]] for point in ao])
    ranks = [np.unique(targets[:, axis]) for axis in (0, 1)]
    if list(map(len, ranks)) != shape or len(ao) != np.prod(shape):
        raise ValueError("AO targets do not match the recorded Cartesian grid")
    stars = []
    for index, point in enumerate(ao):
        recorded = AOOptimizationResult(
            layer / f"grid_pos_{index}" / "ao_results.zarr", saved_time, index
        ).metadata["stage_position"]
        stars.append(
            {
                "index": index,
                "row_index": int(np.searchsorted(ranks[0], targets[index, 0])),
                "column_index": int(np.searchsorted(ranks[1], targets[index, 1])),
                "planned_stage_xyz_um": _um([point[axis] for axis in "xyz"]),
                "stage_readback_xyz_um": _um([recorded[axis] for axis in "xyz"]),
            }
        )
    origin, spacing = _level_zero_grid(source)
    grid = {
        "timepoint": timepoint,
        "saved_ao_timepoint": saved_time,
        "depth_index": depth,
        "grid_shape_yx": shape,
        "projection": {"native_origin_yx_um": origin, "native_scale_yx_um": spacing},
        "acquisition_polygon": _acquisition_polygon(
            config.get("resolved_spatial_plan", {}).get("regions", []),
            footprints,
            nominal,
            scan_offset,
        ),
        "stars": stars,
        "provenance": {"ao_layer": str(layer)},
    }
    if grid["acquisition_polygon"] is None:
        raise ValueError("The Grid-AO overlay requires the saved acquisition polygon")
    return polygon_cell_centers(equal_area_polygon_grid(grid))


def _um(value: Any) -> Any:
    """Serialize micrometer positions at the recorded 0.001 µm precision."""
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Coordinate values must be finite")
    return np.round(array, 3).tolist()


def _acquisition_polygon(
    regions: list[dict[str, Any]],
    footprints: list[dict[str, Any]],
    nominal_origins: np.ndarray,
    scan_offset: float,
) -> dict[str, Any] | None:
    """Map one saved stage-XY acquisition ROI through local tile translations.

    The saved region is the acquisition selection boundary. It is independent
    of tissue intensity and does not replace the region with a tile union or
    convex hull. Edge samples retain the original concave vertex ordering.
    """
    polygons = [
        (index, region)
        for index, region in enumerate(regions)
        if region.get("polygon", {}).get("vertices")
    ]
    if not polygons:
        return None
    if len(polygons) != 1:
        raise ValueError("AO polygon overlay requires one saved acquisition region")
    region_index, region = polygons[0]
    original = np.asarray(region["polygon"]["vertices"], dtype=np.float64)
    if original.ndim != 2 or original.shape[1] != 2 or not np.isfinite(original).all():
        raise ValueError("Saved acquisition polygon requires finite stage XY vertices")
    vertices = original[np.r_[True, np.any(np.diff(original, axis=0), axis=1)]]
    if len(vertices) > 1 and np.array_equal(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    if len(vertices) < 3:
        raise ValueError("Saved acquisition polygon requires three distinct vertices")
    tiles = []
    for tile in footprints:
        index = int(tile["position_index"])
        delta = np.asarray(tile["origin_zyx_um"], dtype=float) - nominal_origins[index]
        bounds = np.asarray(tile["bounds_yx_um"], dtype=float) - delta[[1, 1, 2, 2]]
        tiles.append((index, delta[1:], bounds[[0, 2]], bounds[[1, 3]]))
    mapped = []
    for edge, (start, stop) in enumerate(
        zip(vertices, np.roll(vertices, -1, axis=0), strict=True)
    ):
        steps = max(1, int(np.ceil(np.linalg.norm(stop - start) / 100.0)))
        for part in range(steps):
            stage_xy = start + (stop - start) * part / steps
            nominal = np.asarray([-(stage_xy[0] - scan_offset), stage_xy[1]])
            candidates = []
            for index, delta, low, high in tiles:
                distance_box = float(
                    np.linalg.norm(
                        np.maximum(np.maximum(low - nominal, nominal - high), 0)
                    )
                )
                distance_center = float(np.linalg.norm(nominal - (low + high) / 2))
                candidates.append((distance_box, distance_center, index, delta))
            candidates.sort(key=lambda item: item[:3])
            distance_box, _, index, delta = candidates[0]
            mapped.append(
                {
                    "source_edge_index": edge,
                    "source_edge_fraction": part / steps,
                    "stage_xy_um": _um(stage_xy),
                    "nominal_world_yx_um": _um(nominal),
                    "chosen_position_index": index,
                    "distance_outside_nominal_tile_um": _um(distance_box),
                    "registration_translation_yx_um": _um(delta),
                    "world_yx_um": _um(nominal + delta),
                }
            )
    return {
        "coordinate_role": "registered_saved_acquisition_selection_polygon",
        "coordinate_units": "micrometer",
        "region_name": region.get("name"),
        "source_metadata_key": (
            "attributes.opm_v2.configuration.resolved_spatial_plan."
            f"regions[{region_index}].polygon.vertices"
        ),
        "source_stage_vertices_xy_um": _um(original),
        "distinct_stage_vertices_xy_um": _um(vertices),
        "convex_hull_requested": bool(region["polygon"].get("convex_hull", False)),
        "edge_maximum_spacing_um": 100.0,
        "world_yx_um": [point["world_yx_um"] for point in mapped],
        "mapped_edge_samples": mapped,
        "maximum_nearest_tile_extrapolation_um": max(
            point["distance_outside_nominal_tile_um"] for point in mapped
        ),
        "registration_choice": (
            "Containing same-time/depth nominal footprint with nearest center; "
            "if outside retained deskew footprints, nearest box then nearest center; "
            "ties resolved by position index"
        ),
        "caveat": (
            "Recorded acquisition selection boundary, not a tissue segmentation or "
            "union of acquired tile rectangles. Local tile translations do not define "
            "a single global affine. Edge sampling follows this explicit local mapping."
        ),
    }


def _polygons(geometry: Any) -> list[Any]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if not hasattr(geometry, "geoms"):
        return []
    return [part for child in geometry.geoms for part in _polygons(child)]


def _lines(geometry: Any) -> list[Any]:
    if geometry.is_empty:
        return []
    if geometry.geom_type in ("LineString", "LinearRing"):
        return [geometry]
    if not hasattr(geometry, "geoms"):
        return []
    return [part for child in geometry.geoms for part in _lines(child)]


def _area_cuts(polygon: Any, axis: int, count: int) -> list[float]:
    """Find quantiles of polygon area along one Cartesian display axis."""
    xmin, ymin, xmax, ymax = polygon.bounds
    lower, upper = (xmin, xmax) if axis == 0 else (ymin, ymax)
    edges = [lower]
    for rank in range(1, count):
        left, right = lower, upper
        target = polygon.area * rank / count
        for _ in range(64):
            middle = (left + right) / 2
            clip = (
                box(xmin - 1, ymin - 1, middle, ymax + 1)
                if axis == 0
                else box(xmin - 1, ymin - 1, xmax + 1, middle)
            )
            if polygon.intersection(clip).area < target:
                left = middle
            else:
                right = middle
        # These are derived grid positions, serialized at 0.001 micrometer.
        edges.append(round((left + right) / 2, 3))
    return [*edges, upper]


def equal_area_polygon_grid(grid_data: dict[str, Any]) -> dict[str, Any]:
    """Use the saved polygon, keeping every recorded AO star in place.

    Rows are defined in the final clockwise display orientation. First cut
    three equal-area horizontal bands, then divide each band into three
    equal-area columns. No intensity mask or bounding rectangle determines
    cell area. The white grid follows the polygon and its clipped dividers.
    """
    result = copy.deepcopy(grid_data)
    footprint = result["acquisition_polygon"]
    source_yx = np.asarray(footprint["world_yx_um"], dtype=float)
    if source_yx.ndim != 2 or source_yx.shape[1] != 2 or len(source_yx) < 3:
        raise ValueError(
            "A saved acquisition polygon requires at least three YX vertices"
        )
    if result["grid_shape_yx"] != [3, 3] or len(result["stars"]) != 9:
        raise ValueError("This polygon layout requires nine saved AO positions")
    if not np.isfinite(source_yx).all():
        raise ValueError("Acquisition vertices must be finite")
    # source YX -> clockwise display Cartesian XY = (-sourceY, sourceX).
    display_xy = np.column_stack((-source_yx[:, 0], source_yx[:, 1]))
    polygon = Polygon(display_xy)
    if not polygon.is_valid or polygon.area <= 0 or len(polygon.interiors):
        raise ValueError(
            "The acquisition footprint must be a valid positive-area polygon without holes"
        )
    xmin, _ymin, xmax, _ymax = polygon.bounds
    y_edges = _area_cuts(polygon, 1, 3)
    cells, boundaries, row_records = [], [polygon.boundary], []
    for y in y_edges[1:-1]:
        boundaries.append(
            polygon.intersection(LineString([(xmin - 1, y), (xmax + 1, y)]))
        )
    for row, (ya, yb) in enumerate(pairwise(y_edges)):
        band = polygon.intersection(box(xmin - 1, ya, xmax + 1, yb))
        x_edges = _area_cuts(band, 0, 3)
        row_records.append(
            {
                "display_row": row,
                "display_y_edges_um": [ya, yb],
                "display_x_edges_um": x_edges,
            }
        )
        for x in x_edges[1:-1]:
            boundaries.append(band.intersection(LineString([(x, ya - 1), (x, yb + 1)])))
        for column, (xa, xb) in enumerate(pairwise(x_edges)):
            cell = band.intersection(box(xa, ya, xb, yb))
            pieces = _polygons(cell)
            if len(pieces) != 1:
                raise ValueError(
                    "An equal-area cell is disconnected; choose a different partition orientation"
                )
            coords = np.asarray(pieces[0].exterior.coords)
            yx = np.column_stack((-coords[:, 0], coords[:, 1]))
            cells.append(
                {
                    "display_row": row,
                    "display_column": column,
                    "area_um2": cell.area,
                    "fraction_of_acquisition": cell.area / polygon.area,
                    "polygon_yx_um": np.round(yx, 3).tolist(),
                }
            )
    segments = []
    for line in _lines(unary_union(boundaries)):
        coords = np.asarray(line.coords)
        yx = np.column_stack((-coords[:, 0], coords[:, 1]))
        for first, second in pairwise(yx):
            segment = np.round([first, second], 3)
            if not np.array_equal(segment[0], segment[1]):
                segments.append(segment.tolist())
    target = polygon.area / 9
    result["grid_mode"] = "equal_area_acquisition_polygon"
    result["grid"] = {
        "bounds_yx_um": np.round(
            [
                source_yx[:, 0].min(),
                source_yx[:, 0].max(),
                source_yx[:, 1].min(),
                source_yx[:, 1].max(),
            ],
            3,
        ).tolist(),
        "horizontal_y_um": [],
        "vertical_x_um": [],
        "segments_yx_um": segments,
        "polygon_yx_um": np.round(source_yx, 3).tolist(),
        "cells": cells,
        "display_rows": row_records,
        "definition": "Three equal-area bands, each split into three equal-area cells, clipped to the saved acquisition polygon; bands follow final clockwise display orientation",
        "coverage_reference": "Recorded acquisition selection polygon mapped through source-tile registration",
        "coverage_fraction_of_acquisition": 1.0,
        "total_area_um2": polygon.area,
        "target_cell_area_um2": target,
        "maximum_relative_area_error": max(
            abs(cell["area_um2"] - target) / target for cell in cells
        ),
        "star_coordinate_role": "recorded_AO_stage_reference",
    }
    result.setdefault("provenance", {})["grid_layout"] = result["grid"]["definition"]
    result["provenance"]["grid_is_illustrative_partition"] = True
    return result


def polygon_cell_centers(grid_data: dict[str, Any]) -> dict[str, Any]:
    """Place intended markers at each clipped cell's area centroid.

    Saved correction identities follow their original logical grid rank,
    rather than containment of the acquired reference in the new partition.
    Actual acquisition references remain separate, unchanged provenance.
    """
    result = copy.deepcopy(grid_data)
    if result.get("grid_mode") != "equal_area_acquisition_polygon":
        raise ValueError("Intended polygon centers require the equal-area polygon grid")
    rows, columns = result["grid_shape_yx"]
    cells = {
        (cell["display_row"], cell["display_column"]): cell
        for cell in result["grid"]["cells"]
    }
    if len(cells) != rows * columns:
        raise ValueError("Every intended polygon cell must have a distinct rank")
    markers = []
    for star in result["stars"]:
        rank = (star["column_index"], rows - 1 - star["row_index"])
        cell = cells[rank]
        polygon = Polygon(np.asarray(cell["polygon_yx_um"], dtype=float)[:, ::-1])
        center = polygon.centroid
        if not polygon.contains(center):
            raise ValueError("An intended area centroid lies outside its polygon cell")
        yx = np.round([center.y, center.x], 3).tolist()
        actual = copy.deepcopy(star.get("saved_actual_sample", star))
        if actual.get("coordinate_role", "").startswith("intended"):
            raise ValueError(
                "Intended center metadata must retain its actual saved sample"
            )
        markers.append(
            {
                "index": star["index"],
                "row_index": star["row_index"],
                "column_index": star["column_index"],
                "display_row": rank[0],
                "display_column": rank[1],
                "coordinate_role": "intended_polygon_cell_centroid",
                "world_yx_um": yx,
                "intended_cell_center_yx_um": yx,
                "saved_actual_sample": actual,
            }
        )
    result["stars"] = markers
    result["star_position_mode"] = "cell-centers"
    result["grid"]["star_coordinate_role"] = "intended_polygon_cell_centroid"
    result.setdefault("provenance", {})["star_coordinate_role"] = (
        "Intended area centroid of each equal-area polygon cell; an illustrative layout, not a recorded measurement location"
    )
    result["provenance"]["star_location_caveat"] = (
        "Recorded acquisition references remain unchanged under saved_actual_sample; displayed stars show intended polygon-cell centers"
    )
    return result
