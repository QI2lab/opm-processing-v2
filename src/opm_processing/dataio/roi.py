"""Physical-coordinate rectangular ROI interchange for napari and processing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from yaozarrs import open_group, v05

from opm_processing.dataio.ngff import round_spatial, round_spatial_values
from opm_processing.dataio.position_collection import open_position_collection
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)


ROI_SCHEMA = "opm-processing-roi-v1"


def _registered_projection_context(
    path: Path,
) -> tuple[ProcessingState, Path, dict[str, Any]]:
    """Return the state, processed tiles, and registration for a max-Z image."""
    suffix = "_max_z_fused.ome.zarr"
    if not path.name.endswith(suffix):
        raise ValueError(
            "ROI export requires the registered maximum-Z projection created by fuse"
        )
    state = ProcessingState.read(
        processing_state_path(path.parent, path.name[: -len(suffix)])
    )
    processed_path = state.registered_output_for_max_projection(path)
    registration = state.registration(processed_path)
    tiles = registration.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise ValueError("Registration state lacks final per-tile origins")
    return state, processed_path, registration


def _registered_tile_footprints(path: Path) -> tuple[dict[str, Any], ...]:
    """Derive registered YX footprints from state plus standard OME metadata."""
    state, processed_path, registration = _registered_projection_context(path)
    if state.roi_series(processed_path):
        raise ValueError("ROI selection requires a full-acquisition fusion")
    collection = open_position_collection(processed_path)
    pixel_y, pixel_x = collection.voxel_size_um[-2:]
    footprints = []
    for tile in registration["tiles"]:
        try:
            position_index = int(tile["position_index"])
            time_index = int(tile["time_index"])
            origin = tuple(float(value) for value in tile["origin_zyx_um"])
            tile_shape = collection.arrays[position_index].shape
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ValueError("Registration state contains an invalid tile") from error
        footprints.append(
            {
                "time_index": time_index,
                "position_index": position_index,
                "origin_zyx_um": list(round_spatial_values(origin)),
                "bounds_yx_um": list(
                    round_spatial_values(
                        (
                            origin[1],
                            origin[1] + int(tile_shape[-2]) * pixel_y,
                            origin[2],
                            origin[2] + int(tile_shape[-1]) * pixel_x,
                        )
                    )
                ),
            }
        )
    return tuple(footprints)


def validate_registered_max_projection(path: str | Path) -> Path:
    """Require an ROI canvas projected from registered full-volume fusion."""
    source = Path(path).expanduser().resolve()
    root = open_group(source)
    metadata = root.ome_metadata()
    if not isinstance(metadata, v05.Image):
        raise ValueError("ROI export requires a single fused OME-Zarr Image canvas")
    _registered_tile_footprints(source)
    return source


@dataclass(frozen=True)
class PhysicalRoi:
    """A grid-aligned physical YX rectangle that always retains all Z values."""

    bounds_yx_um: tuple[float, float, float, float]
    source_path: Path
    grid_origin_yx_um: tuple[float, float]
    pixel_size_yx_um: tuple[float, float]
    position_indices: tuple[int, ...] | None = None
    tile_footprints: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Validate and normalize the immutable ROI contract."""
        y0, y1, x0, x1 = (float(value) for value in self.bounds_yx_um)
        if not all(math.isfinite(value) for value in (y0, y1, x0, x1)):
            raise ValueError("ROI bounds must be finite")
        if y0 >= y1 or x0 >= x1:
            raise ValueError("ROI bounds must satisfy min < max on Y and X")
        if any(float(value) <= 0 for value in self.pixel_size_yx_um):
            raise ValueError("ROI pixel sizes must be positive")
        if self.position_indices is not None and any(
            int(value) < 0 for value in self.position_indices
        ):
            raise ValueError("ROI position indices must be nonnegative")

    @classmethod
    def read(cls, path: str | Path) -> PhysicalRoi:
        """Read and validate an ROI JSON file."""
        roi_path = Path(path).expanduser().resolve()
        document = json.loads(roi_path.read_text(encoding="utf-8"))
        if document.get("schema") != ROI_SCHEMA:
            raise ValueError(f"Unsupported ROI schema in {roi_path}")
        if document.get("all_z") is not True:
            raise ValueError("ROI must explicitly retain all Z values")
        bounds = document.get("bounds_yx_um")
        grid = document.get("source_grid")
        if not isinstance(bounds, dict) or not isinstance(grid, dict):
            raise ValueError("ROI JSON lacks bounds or source-grid metadata")
        positions = document.get("position_indices")
        return cls(
            bounds_yx_um=(
                float(bounds["y"][0]),
                float(bounds["y"][1]),
                float(bounds["x"][0]),
                float(bounds["x"][1]),
            ),
            source_path=Path(document["source_path"]).expanduser().resolve(),
            grid_origin_yx_um=tuple(float(value) for value in grid["origin_yx_um"]),
            pixel_size_yx_um=tuple(float(value) for value in grid["pixel_size_yx_um"]),
            position_indices=(
                None
                if positions is None
                else tuple(sorted({int(value) for value in positions}))
            ),
            tile_footprints=tuple(document.get("tile_footprints", ())),
        )

    def write(self, path: str | Path) -> Path:
        """Write the stable ROI JSON interchange format."""
        roi_path = Path(path).expanduser().resolve()
        roi_path.parent.mkdir(parents=True, exist_ok=True)
        y0, y1, x0, x1 = self.bounds_yx_um
        document = {
            "schema": ROI_SCHEMA,
            "source_path": str(self.source_path),
            "coordinate_system": "ome-ngff-physical",
            "unit": "micrometer",
            "axes": ["y", "x"],
            "bounds_yx_um": {"y": [y0, y1], "x": [x0, x1]},
            "source_grid": {
                "origin_yx_um": list(self.grid_origin_yx_um),
                "pixel_size_yx_um": list(self.pixel_size_yx_um),
            },
            "all_z": True,
            "position_indices": (
                None if self.position_indices is None else list(self.position_indices)
            ),
            "tile_footprints": list(self.tile_footprints),
        }
        roi_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return roi_path

    def tile_origin_yx_um(
        self,
        time_index: int,
        position_index: int,
        fallback: tuple[float, float],
    ) -> tuple[float, float]:
        """Return the registered tile origin when available, otherwise stage placement."""
        matches = [
            item
            for item in self.tile_footprints
            if int(item.get("position_index", -1)) == int(position_index)
            and int(item.get("time_index", time_index)) == int(time_index)
        ]
        if not matches:
            matches = [
                item
                for item in self.tile_footprints
                if int(item.get("position_index", -1)) == int(position_index)
            ]
        if not matches:
            return fallback
        bounds = matches[0].get("bounds_yx_um")
        if not isinstance(bounds, list) or len(bounds) != 4:
            return fallback
        return float(bounds[0]), float(bounds[2])

    def registered_tile_origin_zyx_um(
        self,
        time_index: int,
        position_index: int,
    ) -> tuple[float, float, float] | None:
        """Return the original registered full-fusion placement for one tile."""
        matches = [
            item
            for item in self.tile_footprints
            if int(item.get("position_index", -1)) == int(position_index)
            and int(item.get("time_index", time_index)) == int(time_index)
        ]
        if not matches:
            return None
        origin = matches[0].get("origin_zyx_um")
        if not isinstance(origin, list) or len(origin) != 3:
            return None
        return tuple(float(value) for value in origin)


@dataclass(frozen=True)
class SkewedRoiBounds:
    """A bounded skewed scan/X read; camera Y remains deliberately complete."""

    scan_start: int
    scan_stop: int
    x_start: int
    x_stop: int


def world_roi_to_skewed_bounds(
    roi_bounds_yx_um: tuple[float, float, float, float],
    tile_origin_yx_um: tuple[float, float],
    skewed_shape_syx: tuple[int, int, int],
    *,
    pixel_size_um: float,
    scan_step_um: float,
    angle_deg: float,
    crop_y_pixels: int = 0,
    halo_scan: int = 0,
    halo_x: int = 0,
) -> SkewedRoiBounds | None:
    """Map a lab-world YX rectangle into a conservative skewed scan/X box.

    Camera Y is never cropped because it parameterizes lab Z. The scan bounds
    enclose the diagonal preimage of the requested lab-Y interval over every
    camera-Y row.
    """
    scan_count, camera_y, camera_x = (int(value) for value in skewed_shape_syx)
    roi_y0, roi_y1, roi_x0, roi_x1 = roi_bounds_yx_um
    origin_y, origin_x = tile_origin_yx_um
    local_y0 = (roi_y0 - origin_y) / pixel_size_um + crop_y_pixels
    local_y1 = (roi_y1 - origin_y) / pixel_size_um + crop_y_pixels
    local_x0 = (roi_x0 - origin_x) / pixel_size_um
    local_x1 = (roi_x1 - origin_x) / pixel_size_um

    theta = math.radians(float(angle_deg))
    camera_y_projection = max(0, camera_y - 1) * math.cos(theta)
    scan_step_pixels = float(scan_step_um) / float(pixel_size_um)
    scan_start = math.floor((local_y0 - camera_y_projection) / scan_step_pixels) - int(
        halo_scan
    )
    scan_stop = math.ceil(local_y1 / scan_step_pixels) + 1 + int(halo_scan)
    x_start = math.floor(local_x0) - int(halo_x)
    x_stop = math.ceil(local_x1) + int(halo_x)

    scan_start = max(0, scan_start)
    scan_stop = min(scan_count, scan_stop)
    x_start = max(0, x_start)
    x_stop = min(camera_x, x_stop)
    if scan_start >= scan_stop or x_start >= x_stop:
        return None
    return SkewedRoiBounds(scan_start, scan_stop, x_start, x_stop)


def _level_zero_grid(path: Path) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the level-zero physical YX origin and spacing from an NGFF Image."""
    root = open_group(path)
    metadata = root.ome_metadata()
    if not isinstance(metadata, v05.Image):
        raise ValueError("ROI export requires a single fused OME-Zarr Image canvas")
    dataset = metadata.multiscales[0].datasets[0]
    scale = dataset.scale_transform.scale
    translation = (
        [0.0] * len(scale)
        if dataset.translation_transform is None
        else dataset.translation_transform.translation
    )
    return (
        round_spatial_values(translation[-2:]),
        round_spatial_values(scale[-2:]),
    )


def align_bounds_to_grid(
    bounds_yx_um: tuple[float, float, float, float],
    origin_yx_um: tuple[float, float],
    pixel_size_yx_um: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Expand physical bounds to the enclosing level-zero pixel grid."""
    y0, y1, x0, x1 = (float(value) for value in bounds_yx_um)
    oy, ox = (float(value) for value in origin_yx_um)
    sy, sx = (float(value) for value in pixel_size_yx_um)

    def snap_grid_index(value: float) -> float:
        """Remove floating-point noise around an exact integer grid edge."""
        nearest = round(value)
        return float(nearest) if math.isclose(value, nearest, abs_tol=1e-7) else value

    iy0 = math.floor(snap_grid_index((y0 - oy) / sy))
    iy1 = math.ceil(snap_grid_index((y1 - oy) / sy))
    ix0 = math.floor(snap_grid_index((x0 - ox) / sx))
    ix1 = math.ceil(snap_grid_index((x1 - ox) / sx))
    return (
        round_spatial(oy + iy0 * sy),
        round_spatial(oy + iy1 * sy),
        round_spatial(ox + ix0 * sx),
        round_spatial(ox + ix1 * sx),
    )


def intersecting_position_indices(
    bounds_yx_um: tuple[float, float, float, float],
    footprints: list[dict[str, Any]],
) -> tuple[int, ...]:
    """Return source positions whose physical YX footprints overlap an ROI."""
    roi_y0, roi_y1, roi_x0, roi_x1 = bounds_yx_um
    return tuple(
        sorted(
            {
                int(footprint["position_index"])
                for footprint in intersecting_tile_footprints(
                    bounds_yx_um,
                    footprints,
                )
            }
        )
    )


def intersecting_tile_footprints(
    bounds_yx_um: tuple[float, float, float, float],
    footprints: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return complete per-timepoint tile records intersecting an ROI."""
    roi_y0, roi_y1, roi_x0, roi_x1 = bounds_yx_um
    selected = []
    for footprint in footprints:
        bounds = footprint.get("bounds_yx_um")
        if not isinstance(bounds, list) or len(bounds) != 4:
            continue
        tile_y0, tile_y1, tile_x0, tile_x1 = (float(value) for value in bounds)
        if (
            tile_y1 > roi_y0
            and tile_y0 < roi_y1
            and tile_x1 > roi_x0
            and tile_x0 < roi_x1
        ):
            selected.append(dict(footprint))
    return tuple(selected)


def roi_from_world_rectangle(
    source_path: str | Path,
    vertices_world: np.ndarray,
) -> PhysicalRoi:
    """Create a physical ROI from napari rectangle vertices in world coordinates."""
    source = validate_registered_max_projection(source_path)
    vertices = np.asarray(vertices_world, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] < 2 or vertices.shape[1] < 2:
        raise ValueError("ROI rectangle vertices must have shape (n, ndim>=2)")
    y_values = vertices[:, -2]
    x_values = vertices[:, -1]
    origin, spacing = _level_zero_grid(source)
    aligned_bounds = align_bounds_to_grid(
        (
            float(y_values.min()),
            float(y_values.max()),
            float(x_values.min()),
            float(x_values.max()),
        ),
        origin,
        spacing,
    )
    footprints = list(_registered_tile_footprints(source))
    selected_footprints = intersecting_tile_footprints(aligned_bounds, footprints)
    selected = tuple(
        sorted({int(item["position_index"]) for item in selected_footprints})
    )
    return PhysicalRoi(
        bounds_yx_um=aligned_bounds,
        source_path=source,
        grid_origin_yx_um=origin,
        pixel_size_yx_um=spacing,
        position_indices=selected or None,
        tile_footprints=selected_footprints,
    )


def roi_from_image_pixel_rectangle(
    source_path: str | Path,
    vertices_image_data: np.ndarray,
) -> PhysicalRoi:
    """Create a physical ROI from level-zero napari image-data coordinates.

    A Shapes layer has its own transform, so its world coordinates cannot be
    assumed to equal the OME-NGFF physical coordinates of an Image layer. The
    caller must first map the shape vertices through the reference Image
    layer's ``world_to_data`` transform. This function then applies the
    authoritative level-zero NGFF scale and translation.
    """
    source = Path(source_path).expanduser().resolve()
    vertices = np.asarray(vertices_image_data, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] < 2 or vertices.shape[1] < 2:
        raise ValueError("ROI image vertices must have shape (n, ndim>=2)")
    origin, spacing = _level_zero_grid(source)
    physical_vertices = np.empty((vertices.shape[0], 2), dtype=np.float64)
    physical_vertices[:, 0] = origin[0] + vertices[:, -2] * spacing[0]
    physical_vertices[:, 1] = origin[1] + vertices[:, -1] * spacing[1]
    return roi_from_world_rectangle(source, physical_vertices)
