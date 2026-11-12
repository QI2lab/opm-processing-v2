"""Read saved opm-v2 AO coefficient states through native Zarr storage.

Both the historical Zarr v2 groups and current Zarr v3 groups written by
``opm_v2.utils.sensorless_ao.save_optimization_results`` store modal states in
``starting_coeffs`` and ``optimal_coeffs``. This reader does not estimate mirror
shape, reconstruct pupils, or alter coefficient units.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from yaozarrs import open_group

from opm_processing.dataio.acquisition import acquisition_stem
from opm_processing.dataio.ngff import round_spatial_values

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opm_processing.dataio.acquisition import AcquisitionMetadata
    from opm_processing.dataio.roi import PhysicalRoi


@dataclass
class AOOptimizationResult:
    """One saved optimization's native coefficient states and array metadata."""

    zarr_path: Path
    time_index: int
    position_index: int

    def __post_init__(self) -> None:
        """Open the actual generic Zarr group without materializing images."""
        self.zarr_path = Path(self.zarr_path).expanduser().absolute()
        self._group = open_group(self.zarr_path)

    @cached_property
    def metadata(self) -> dict[str, Any]:
        """Return recorded optimization settings without inferring missing modes."""
        return dict(self._group.attrs)

    def _array(self, name: str) -> np.ndarray:
        """Read one named saved array with the installed TensorStore backend."""
        return np.asarray(self._group[name].to_tensorstore().read().result())

    @cached_property
    def starting_coefficients(self) -> np.ndarray:
        """Return the finite saved starting vector in its original OPD units."""
        values = np.asarray(self._array("starting_coeffs"), dtype=float)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError("Saved AO starting coefficients must be a finite vector")
        return values

    @cached_property
    def optimal_coefficients(self) -> np.ndarray:
        """Return saved coefficient vectors after iterations or modal updates."""
        values = np.asarray(self._array("optimal_coeffs"), dtype=float)
        if (
            values.ndim != 2
            or values.shape[1] != self.starting_coefficients.size
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "Saved AO coefficient history must use the starting modal basis"
            )
        return values

    @cached_property
    def final_coefficients(self) -> np.ndarray:
        """Return the last completed iteration, matching the saved producer state."""
        modes = self.metadata.get("modes_to_optimize", [])
        iterations = int(self.metadata.get("num_iterations", 1))
        if iterations < 1 or not isinstance(modes, (list, tuple)):
            raise ValueError("Saved AO optimization dimensions are invalid")
        expected = iterations * len(modes)
        completed = self.optimal_coefficients.shape[0]
        if completed != expected and (not modes or completed != iterations):
            raise ValueError(
                "Saved AO history must contain every iteration or every modal update"
            )
        if not completed:
            return self.starting_coefficients.copy()
        return self.optimal_coefficients[-1].copy()


def read_ao_stage_plan(
    path: Path, depth: int
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Validate a depth's saved position plan and retain its exact fingerprint."""
    content = path.read_bytes()
    positions = json.loads(content)
    if not isinstance(positions, list) or not positions:
        raise ValueError(f"Expected a nonempty saved stage plan: {path}")
    xyz = np.asarray(
        [[point[axis] for axis in "xyz"] for point in positions], dtype=float
    )
    if not np.isfinite(xyz).all():
        raise ValueError(f"Saved stage positions must be finite: {path}")
    if any(int(point.get("depth_index", depth)) != depth for point in positions):
        raise ValueError(f"Saved depth indices disagree with AO layer: {path}")
    return positions, {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def ao_tile_assignments(
    layer: Path,
    stage_positions_zxy: Sequence[Sequence[float]],
    ao: Sequence[dict[str, Any]],
    experiment: Sequence[dict[str, Any]],
) -> tuple[dict[int, int], dict[str, str] | None]:
    """Resolve source-position IDs to the corrections used during acquisition.

    Recorded assignments take precedence and must agree with source coordinates.
    Older plans reconstruct the historical nearest physical XY assignment.
    Source identities use Z/laboratory-scan/Y coordinates rounded to three
    decimals in micrometers. Return the mapping and recorded-geometry provenance,
    or None for provenance when assignments were inferred from the stage plans.
    """
    geometry_path = layer / "ao_grid_geometry.json"
    position_ids = None
    provenance = None
    if geometry_path.is_file():
        content = geometry_path.read_bytes()
        geometry = json.loads(content)
        position_ids = np.asarray(geometry.get("position_indices", []))
        indices = np.asarray(geometry.get("tile_ao_indices", []))
        if (
            geometry.get("schema_version") != 1
            or geometry.get("units") != "um"
            or geometry.get("correction_scope") != "entire_tile"
            or position_ids.shape != (len(experiment),)
            or indices.shape != position_ids.shape
            or not np.issubdtype(position_ids.dtype, np.integer)
            or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(position_ids)) != len(position_ids)
            or np.any(position_ids < 0)
            or np.any(position_ids >= len(stage_positions_zxy))
            or np.any(indices < 0)
            or np.any(indices >= len(ao))
        ):
            raise ValueError(f"Invalid recorded AO mapping: {geometry_path}")
        provenance = {
            "path": str(geometry_path),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    else:
        xy = np.asarray([[point["x"], point["y"]] for point in ao])
        positions = np.asarray([[point["x"], point["y"]] for point in experiment])
        indices = np.argmin(
            np.sum((positions[:, None] - xy[None]) ** 2, axis=2), axis=1
        )
    assignments = {}
    for row, (point, index) in enumerate(zip(experiment, indices, strict=True)):
        stage = round_spatial_values(
            (point["z"], point.get("lab_scan_um", point["x"]), point["y"])
        )
        identity = stage
        if position_ids is not None:
            identity = int(position_ids[row])
            if round_spatial_values(stage_positions_zxy[identity]) != stage:
                raise ValueError(
                    f"Recorded AO source position {identity} disagrees with raw stage coordinates: {geometry_path}"
                )
        if identity in assignments:
            raise ValueError(f"Duplicate experiment stage position in {layer}: {stage}")
        assignments[identity] = int(index)
    if position_ids is None:
        assignments = {
            position: assignments[stage]
            for position, stage in enumerate(
                map(round_spatial_values, stage_positions_zxy)
            )
            if stage in assignments
        }
    return assignments, provenance


def read_ao_layers(
    acquisition: AcquisitionMetadata,
    preview_path: Path,
    ao_root: Path | None = None,
) -> dict:
    """Read saved AO plans and source-tile assignments once for an ROI batch.

    Parameters
    ----------
    acquisition : AcquisitionMetadata
        Original raw acquisition with physical stage positions.
    preview_path : pathlib.Path
        Preview used to locate AO results when they are separate from raw data.
    ao_root : pathlib.Path or None
        Saved AO grid root or layer. Otherwise search beside raw data and preview.

    Returns
    -------
    dict
        Time/depth keys mapped to layer paths, assignments and provenance.
        Empty when no AO results exist.
    """
    if ao_root is None:
        stem = acquisition_stem(acquisition.path)
        roots = []
        for directory in dict.fromkeys((acquisition.path.parent, preview_path.parent)):
            roots = [
                directory / name
                for name in (f"{stem}.ome_ao_grid_results", f"{stem}_ao_grid_results")
                if (directory / name).is_dir()
            ]
            if roots:
                break
        if not roots:
            return {}
        if len(roots) != 1:
            raise ValueError("Multiple saved AO grids found; select --ao-root")
        root = roots[0]
    else:
        root = Path(ao_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"AO result directory does not exist: {root}")
    layers = {}
    candidates = (
        [root] if (root / "ao_stage_positions.json").is_file() else root.iterdir()
    )
    for layer in candidates:
        match = re.fullmatch(
            r"time_(\d+)(?:_z_(\d+))?_ao_grid(?:_results)?", layer.name
        )
        if match is None and layer != root:
            continue
        key = (int(match[1]), int(match[2] or 0)) if match else (0, 0)
        if key in layers:
            raise ValueError(f"Duplicate saved AO time/depth {key}")
        plans, provenance = [], []
        for name in ("ao_stage_positions.json", "exp_stage_positions.json"):
            path = layer / name
            plan, source = read_ao_stage_plan(path, key[1])
            plans.append(plan)
            provenance.append(source)
        ao, experiment = plans
        assignments, geometry_source = ao_tile_assignments(
            layer, acquisition.stage_positions_zxy, ao, experiment
        )
        if geometry_source is not None:
            provenance.append(geometry_source)
        layers[key] = (layer, assignments, provenance, geometry_source is not None)
    if not layers:
        raise ValueError(f"No saved AO grid layers found in {root}")
    return layers


def select_roi_ao(
    acquisition: AcquisitionMetadata,
    roi: PhysicalRoi,
    ao_root: Path | None = None,
    *,
    ao_layers: dict | None = None,
) -> list[dict[str, Any]]:
    """Select the largest saved correction footprint at each ROI time and depth.

    Parameters
    ----------
    acquisition : AcquisitionMetadata
        Original raw acquisition with physical stage positions.
    roi : PhysicalRoi
        Selected bounds and source-tile footprints in registered coordinates.
    ao_root : pathlib.Path or None
        Saved AO grid root or layer. Otherwise search beside raw data and preview.
    ao_layers : dict or None
        Plans returned by read_ao_layers and shared by the current ROI batch.

    Returns
    -------
    list of dict
        Selected result paths and saved-plan provenance. Empty when no AO exists.
        Recorded assignments take precedence; older grids use nearest stage XY.
        Areas are unions of intersected tile footprints without repeated overlap.
    """
    from shapely import box, union_all

    if not roi.tile_footprints:
        raise ValueError("AO selection requires saved source-tile footprints")
    layers = (
        read_ao_layers(acquisition, roi.source_path, ao_root)
        if ao_layers is None
        else ao_layers
    )
    if not layers:
        return []
    y0, y1, x0, x1 = roi.bounds_yx_um
    boundary = box(x0, y0, x1, y1)
    footprints = {}
    for tile in roi.tile_footprints:
        position, time = int(tile["position_index"]), int(tile["time_index"])
        if roi.position_indices is not None and position not in roi.position_indices:
            continue
        a, b, c, d = tile["bounds_yx_um"]
        intersection = boundary.intersection(box(c, a, d, b))
        if intersection.area:
            footprints[time, position] = intersection
    records, assigned_tiles = [], set()
    for time in sorted({key[0] for key in footprints}):
        for depth in sorted({key[1] for key in layers}):
            available = [key for key in layers if key[1] == depth and key[0] <= time]
            if not available:
                continue
            saved_time, _ = max(available)
            layer, assignments, provenance, recorded = layers[saved_time, depth]
            groups = defaultdict(list)
            members = defaultdict(list)
            for (tile_time, position), intersection in footprints.items():
                if tile_time != time:
                    continue
                if position not in assignments:
                    continue
                if (time, position) in assigned_tiles:
                    raise ValueError(
                        f"ROI source tile {(time, position)} matches multiple saved AO depth plans"
                    )
                assigned_tiles.add((time, position))
                index = assignments[position]
                groups[index].append(intersection)
                members[index].append(position)
            if not groups:
                continue
            areas = {
                index: float(union_all(polygons).area)
                for index, polygons in groups.items()
            }
            selected = max(sorted(areas), key=areas.get)
            store = layer / f"grid_pos_{selected}" / "ao_results.zarr"
            if not store.is_dir():
                raise ValueError(f"Selected AO correction is missing: {store}")
            records.append(
                {
                    "time_index": time,
                    "depth_index": depth,
                    "ao_time_index": saved_time,
                    "region_index": selected,
                    "result_path": str(store),
                    "area_um2": areas[selected],
                    "candidate_areas_um2": {
                        str(index): area for index, area in sorted(areas.items())
                    },
                    "position_indices": members[selected],
                    "assignment": "recorded ao_grid_geometry.json tile_ao_indices; entire tile"
                    if recorded
                    else "inferred from saved acquisition plans; nearest physical stage XY",
                    "selection": "largest union area within requested ROI; lowest saved index breaks ties",
                    "plans": provenance,
                }
            )
    missing = sorted(set(footprints) - assigned_tiles)
    if missing:
        raise ValueError(
            f"Saved AO plans do not identify ROI source tiles {missing}; select the matching --ao-root"
        )
    return records


def ao_wavelength_nm(
    result: AOOptimizationResult,
    configuration: dict[str, Any],
    override: float | None = None,
) -> float:
    """Resolve phase wavelength from recorded AO channel settings or an override.

    Parameters
    ----------
    result : AOOptimizationResult
        Selected saved optimization with its illumination state.
    configuration : dict
        Raw acquisition's recorded opm-v2 configuration.
    override : float or None
        Explicit display wavelength in nanometers, taking precedence when supplied.

    Returns
    -------
    float
        Positive wavelength in nanometers; missing or ambiguous metadata is rejected.
    """
    if override is not None:
        values = [float(override)]
    else:
        states = result.metadata.get("channel_states", [])
        identifiers = configuration.get("OPM", {}).get("channel_ids", [])
        ao = configuration.get("acq_config", {}).get("AO", {})
        if states and len(states) == len(identifiers):
            names = [
                name
                for name, enabled in zip(identifiers, states, strict=True)
                if enabled
            ]
        else:
            names = [ao.get("active_channel_id", ao.get("channel_id", ""))]
        values = []
        for name in names:
            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*nm\s*", str(name))
            if match:
                values.append(float(match[1]))
        if len(names) != 1:
            values = []
    if len(values) != 1 or not np.isfinite(values[0]) or values[0] <= 0:
        raise ValueError(
            "AO phase needs one recorded illumination wavelength; supply --ao-wavelength-nm"
        )
    return values[0]
