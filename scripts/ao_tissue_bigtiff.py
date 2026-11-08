"""Render calibrated AO overlays as bounded-memory, native tiled BigTIFFs."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tifffile
from matplotlib import colormaps, rcParams
from matplotlib.colors import Normalize

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence


TILE_SHAPE = (1024, 1024)
TIFF_WORKERS = 4
TIFF_BUFFER_BYTES = 16 * 1024 * 1024
_HISTOGRAM_BYTES = 8 * 1024 * 1024


def native_display_limits(image: Any) -> tuple[float, float]:
    """Calculate exact linear 1/99.8 percentiles from a streamed integer histogram.

    Only strips are materialized. Zero-valued pixels outside a layer ROI count
    exactly as they do in the existing full-array percentile calculation.
    """
    shape = tuple(int(value) for value in image.shape)
    dtype = np.dtype(image.dtype)
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("Native tissue contrast requires a nonempty YX image")
    if dtype.kind != "u" or dtype.itemsize not in (1, 2):
        raise ValueError("Native tissue contrast requires uint8 or uint16 pixels")
    bins = 1 << (8 * dtype.itemsize)
    histogram = np.zeros(bins, dtype=np.uint64)
    rows = max(1, min(1024, _HISTOGRAM_BYTES // (shape[1] * dtype.itemsize)))
    for y0 in range(0, shape[0], rows):
        strip = np.asarray(image[y0 : min(y0 + rows, shape[0]), :])
        histogram += np.bincount(strip.reshape(-1), minlength=bins).astype(np.uint64)
    cumulative = histogram.cumsum(dtype=np.uint64)
    count = int(cumulative[-1])
    limits = []
    for percentile in (1.0, 99.8):
        rank = (count - 1) * (percentile / 100.0)
        lower, upper = math.floor(rank), math.ceil(rank)
        a = int(np.searchsorted(cumulative, lower, side="right"))
        b = int(np.searchsorted(cumulative, upper, side="right"))
        limits.append(float(a + (b - a) * (rank - lower)))
    return limits[0], limits[1]


class NativeTissueRaster:
    """Compose independent native RGB tiles using globally fixed geometry.

    Input pupils retain SJS's lower-origin orientation. Background image rows
    increase downward. All geometric masks use global pixel coordinates, so
    arbitrary render partitions produce the same pixels without seams.
    Two-dimensional backgrounds use calibrated grayscale limits. RGB uint8
    backgrounds are already display-mapped and pass through without scaling.
    """

    def __init__(
        self,
        geometry: Any,
        image: Any,
        regions: Sequence[Any],
        *,
        depth: int,
        timepoint: int = 0,
        pupil_diameter_um: float,
        scale_bar_um: float = 200.0,
        display_limits: tuple[float, float] | None = None,
        centers_yx_um: Mapping[int, tuple[float, float]] | None = None,
        grid_boundaries: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self.geometry, self.image = geometry, image
        self.depth, self.timepoint = int(depth), int(timepoint)
        image_shape = tuple(int(value) for value in image.shape)
        self._rgb_background = len(image_shape) == 3 and image_shape[-1] == 3
        if self._rgb_background and np.dtype(image.dtype) != np.dtype("uint8"):
            raise ValueError("Native RGB tissue backgrounds require uint8 pixels")
        self.shape_yx = image_shape[:2]
        if (
            len(image_shape) not in (2, 3)
            or (len(image_shape) == 3 and not self._rgb_background)
            or len(self.shape_yx) != 2
            or min(self.shape_yx) < 1
        ):
            raise ValueError("Native tissue raster requires a nonempty YX or RGB image")
        if tuple(geometry.shape_yx) != self.shape_yx:
            raise ValueError("Native tissue raster must match the calibrated canvas")
        self.y0, self.y1, self.x0, self.x1 = map(float, geometry.extent_yx_um)
        self.height_um, self.width_um = self.y1 - self.y0, self.x1 - self.x0
        if min(self.height_um, self.width_um) <= 0:
            raise ValueError("Native tissue extent must have positive dimensions")
        if not np.isfinite(pupil_diameter_um) or pupil_diameter_um <= 0:
            raise ValueError("Pupil display diameter must be positive and finite")
        if not np.isfinite(scale_bar_um) or not 0 < scale_bar_um <= 0.8 * self.width_um:
            raise ValueError("Scale bar must be positive and fit inside the canvas")
        self.pupil_diameter_um = float(pupil_diameter_um)
        self.scale_bar_um = float(scale_bar_um)
        self.dy, self.dx = (
            self.height_um / self.shape_yx[0],
            self.width_um / self.shape_yx[1],
        )
        if self._rgb_background:
            self.display_limits = (0.0, 255.0)
        else:
            self.display_limits = (
                native_display_limits(image)
                if display_limits is None
                else tuple(map(float, display_limits))
            )
        if (
            len(self.display_limits) != 2
            or not np.isfinite(self.display_limits).all()
            or self.display_limits[1] < self.display_limits[0]
        ):
            raise ValueError("Tissue display limits must be finite and increasing")
        self._denominator = max(
            self.display_limits[1] - self.display_limits[0], np.finfo(float).eps
        )
        self._gray = colormaps["gray"](np.linspace(0, 1, 256), bytes=True)[:, :3]
        self._phase = colormaps[rcParams["image.cmap"]]
        self._phase_norm = Normalize(-2 * np.pi, 2 * np.pi)
        self._bar_lut = self._phase(np.linspace(0, 1, 256), bytes=True)[:, :3]
        self._line_scale = self.shape_yx[1] / (8 * 160 * 0.96)
        points_to_pixels = 160 / 72
        self.grid_width = 0.75 * self._line_scale * points_to_pixels
        self.grid_alpha = 0.3
        self.bar_white_width = min(
            3 * self._line_scale * points_to_pixels, (self.scale_bar_um / self.dx) / 25
        )
        self.bar_black_width = self.bar_white_width * 5 / 3
        if centers_yx_um is None:
            centers_yx_um = geometry.planned_centers_by_depth_index.get(self.depth, {})
        if grid_boundaries is None:
            grid_boundaries = geometry.planned_grid_by_depth_index.get(self.depth)
        if not centers_yx_um or grid_boundaries is None:
            raise ValueError(
                "Native overlays require intended AO cell centers and boundaries"
            )
        self.grid = {
            name: [float(value) for value in grid_boundaries.get(name, [])]
            for name in ("horizontal_y_um", "vertical_x_um")
        }
        self._pupils = []
        radius = self.pupil_diameter_um / 2
        for region in regions:
            index = int(region.index)
            if index not in centers_yx_um:
                raise ValueError(f"Missing intended center for AO correction {index}")
            y, x = map(float, centers_yx_um[index])
            if not (self.y0 <= y <= self.y1 and self.x0 <= x <= self.x1):
                raise ValueError("AO correction center lies outside the tissue canvas")
            pupil = np.ma.masked_invalid(np.ma.asarray(region.pupil, dtype=float))
            if pupil.ndim != 2 or min(pupil.shape) < 1:
                raise ValueError(
                    "AO pupil must be a nonempty two-dimensional phase map"
                )
            yy = (np.arange(pupil.shape[0]) + 0.5) / pupil.shape[0] * 2 - 1
            xx = (np.arange(pupil.shape[1]) + 0.5) / pupil.shape[1] * 2 - 1
            pupil = np.ma.masked_where(yy[:, None] ** 2 + xx[None, :] ** 2 > 1, pupil)
            rgba = self._phase(self._phase_norm(pupil), bytes=True)
            left = x + radius + 0.08 * self.pupil_diameter_um
            bar_width = 0.06 * self.pupil_diameter_um
            if left + bar_width > self.x1:
                left = x - radius - 0.08 * self.pupil_diameter_um - bar_width
            bounds = self._pixel_bounds(y - radius, y + radius, x - radius, x + radius)
            bar = self._pixel_bounds(y - radius, y + radius, left, left + bar_width)
            outline = min(
                0.8 * self._line_scale * points_to_pixels, (bar[3] - bar[2]) * 0.2
            )
            self._pupils.append(
                {
                    "index": index,
                    "center_yx_um": [round(y, 3), round(x, 3)],
                    "rgba": rgba,
                    "bounds": bounds,
                    "colorbar": bar,
                    "outline": outline,
                    "colorbar_bounds_yx_um": [
                        round(y - radius, 3),
                        round(y + radius, 3),
                        round(left, 3),
                        round(left + bar_width, 3),
                    ],
                }
            )
        if not self._pupils:
            raise ValueError("Native tissue overlay has no selected AO pupils")
        right = self.x1 - 0.06 * self.width_um
        bar_y = self.y1 - 0.07 * self.height_um
        self._scale_x0 = (right - self.scale_bar_um - self.x0) / self.dx
        self._scale_x1 = (right - self.x0) / self.dx
        self._scale_y = (bar_y - self.y0) / self.dy
        self.scale_bar = {
            "length_um": self.scale_bar_um,
            "label_visible": False,
            "endpoints_yx_um": [
                [round(bar_y, 3), round(right - self.scale_bar_um, 3)],
                [round(bar_y, 3), round(right, 3)],
            ],
            "white_stroke_pixels": self.bar_white_width,
            "black_stroke_pixels": self.bar_black_width,
        }

    def _pixel_bounds(
        self, y0: float, y1: float, x0: float, x1: float
    ) -> tuple[float, float, float, float]:
        return (
            (y0 - self.y0) / self.dy,
            (y1 - self.y0) / self.dy,
            (x0 - self.x0) / self.dx,
            (x1 - self.x0) / self.dx,
        )

    @staticmethod
    def _coverage(
        bounds: tuple[float, float, float, float], y0: int, y1: int, x0: int, x1: int
    ) -> np.ndarray:
        top, bottom, left, right = bounds
        yy, xx = np.arange(y0, y1), np.arange(x0, x1)
        cy = np.clip(np.minimum(yy + 1, bottom) - np.maximum(yy, top), 0, 1)
        cx = np.clip(np.minimum(xx + 1, right) - np.maximum(xx, left), 0, 1)
        return (cy[:, None] * cx[None, :]).astype(np.float32)

    @staticmethod
    def _blend(target: np.ndarray, color: Any, alpha: np.ndarray) -> None:
        weight = np.asarray(alpha, dtype=np.float32)[..., None]
        target[:] = np.rint(target * (1 - weight) + np.asarray(color) * weight).astype(
            np.uint8
        )

    @staticmethod
    def _pupil_patch(
        pupil: dict[str, Any], y0: int, y1: int, x0: int, x1: int
    ) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        """Sample a pupil on global native pixel centers, retaining lower origin."""
        top, bottom, left, right = pupil["bounds"]
        ya, yb = max(y0, math.ceil(top - 0.5)), min(y1, math.ceil(bottom - 0.5))
        xa, xb = max(x0, math.ceil(left - 0.5)), min(x1, math.ceil(right - 0.5))
        if ya >= yb or xa >= xb:
            return None
        source = pupil["rgba"]
        if source.dtype != np.uint8 or source.ndim != 3 or source.shape[-1] != 4:
            raise ValueError("Pupil sampling requires uint8 RGBA phase patches")
        rows = np.clip(
            ((np.arange(ya, yb) + 0.5 - top) / (bottom - top) * source.shape[0]).astype(
                int
            ),
            0,
            source.shape[0] - 1,
        )
        columns = np.clip(
            (
                (np.arange(xa, xb) + 0.5 - left) / (right - left) * source.shape[1]
            ).astype(int),
            0,
            source.shape[1] - 1,
        )
        patch = source[source.shape[0] - 1 - rows[:, None], columns[None, :]]
        return (ya, yb, xa, xb), patch

    def render_tile(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Render an arbitrary half-open native pixel rectangle as RGB uint8."""
        if not (0 <= y0 < y1 <= self.shape_yx[0] and 0 <= x0 < x1 <= self.shape_yx[1]):
            raise ValueError("Native render bounds lie outside the tissue canvas")
        if self._rgb_background:
            result = np.array(self.image[y0:y1, x0:x1, :], dtype=np.uint8, copy=True)
        else:
            values = np.asarray(self.image[y0:y1, x0:x1], dtype=np.float64)
            normalized = np.clip(
                (values - self.display_limits[0]) / self._denominator, 0, 1
            )
            indices = np.minimum((normalized * 256).astype(np.uint16), 255)
            result = self._gray[indices].copy()
        for y in self.grid["horizontal_y_um"]:
            center = (y - self.y0) / self.dy
            coverage = np.clip(
                self.grid_width / 2 + 0.5 - np.abs(np.arange(y0, y1) + 0.5 - center),
                0,
                1,
            )
            if coverage.any():
                self._blend(
                    result,
                    255,
                    np.broadcast_to(coverage[:, None], result.shape[:2])
                    * self.grid_alpha,
                )
        for x in self.grid["vertical_x_um"]:
            center = (x - self.x0) / self.dx
            coverage = np.clip(
                self.grid_width / 2 + 0.5 - np.abs(np.arange(x0, x1) + 0.5 - center),
                0,
                1,
            )
            if coverage.any():
                self._blend(
                    result,
                    255,
                    np.broadcast_to(coverage[None, :], result.shape[:2])
                    * self.grid_alpha,
                )
        for pupil in self._pupils:
            sampled = self._pupil_patch(pupil, y0, y1, x0, x1)
            if sampled is not None:
                (ya, yb, xa, xb), patch = sampled
                self._blend(
                    result[ya - y0 : yb - y0, xa - x0 : xb - x0],
                    patch[..., :3],
                    patch[..., 3] / 255,
                )
            bt, bb, bl, br = pupil["colorbar"]
            half = pupil["outline"] / 2
            if bb + half <= y0 or bt - half >= y1 or br + half <= x0 or bl - half >= x1:
                continue
            coverage = self._coverage((bt, bb, bl, br), y0, y1, x0, x1)
            rows = np.minimum(
                (
                    np.clip(1 - (np.arange(y0, y1) + 0.5 - bt) / (bb - bt), 0, 1) * 256
                ).astype(int),
                255,
            )
            self._blend(result, self._bar_lut[rows][:, None, :], coverage)
            outer = self._coverage(
                (bt - half, bb + half, bl - half, br + half), y0, y1, x0, x1
            )
            inner = self._coverage(
                (bt + half, bb - half, bl + half, br - half), y0, y1, x0, x1
            )
            self._blend(result, 0, np.clip(outer - inner, 0, 1))
        for width, color in ((self.bar_black_width, 0), (self.bar_white_width, 255)):
            coverage = self._coverage(
                (
                    self._scale_y - width / 2,
                    self._scale_y + width / 2,
                    self._scale_x0,
                    self._scale_x1,
                ),
                y0,
                y1,
                x0,
                x1,
            )
            if coverage.any():
                self._blend(result, color, coverage)
        return result

    def iter_tiles(self) -> Iterator[np.ndarray]:
        """Yield tiles in TIFF row-major order with bounded render memory."""
        h, w = self.shape_yx
        total = math.ceil(h / TILE_SHAPE[0]) * math.ceil(w / TILE_SHAPE[1])
        interval = max(16, min(128, total // 20))
        completed = 0
        for y0 in range(0, h, TILE_SHAPE[0]):
            for x0 in range(0, w, TILE_SHAPE[1]):
                yield self.render_tile(
                    y0, min(y0 + TILE_SHAPE[0], h), x0, min(x0 + TILE_SHAPE[1], w)
                )
                completed += 1
                if completed == total or completed % interval == 0:
                    print(
                        f"Native BigTIFF depth {self.depth}: {completed}/{total} tiles",
                        flush=True,
                    )

    def metadata(self) -> dict[str, Any]:
        """Return display provenance without retaining any raster patches."""
        return {
            "timepoint": self.timepoint,
            "depth_index": self.depth,
            "pixel_shape_yx": list(self.shape_yx),
            "native_scale_yx_um": [
                round(float(value), 3) for value in self.geometry.scale_yx_um
            ],
            "native_origin_yx_um": [
                round(float(value), 3)
                for value in getattr(
                    self.geometry,
                    "origin_yx_um",
                    (self.y0 + self.dy / 2, self.x0 + self.dx / 2),
                )
            ],
            "extent_yx_um": [
                round(value, 3) for value in (self.y0, self.y1, self.x0, self.x1)
            ],
            "tissue_display_limits": list(self.display_limits),
            "tissue_display_percentiles": [1, 99.8],
            "pupil_display_diameter_um": round(self.pupil_diameter_um, 3),
            "pupil_phase_limit_rad": float(2 * np.pi),
            "pupil_normalization": "linear",
            "pupil_orientation": "source row zero at visible bottom",
            "pupil_sampling": "nearest source phase sample on global native pixel centers",
            "pupil_colormap": self._phase.name,
            "grid_line_alpha": self.grid_alpha,
            "planned_grid": self.grid,
            "colorbars": "one beside each pupil; no ticks or text",
            "scale_bar": self.scale_bar,
            "regions": [
                {
                    key: p[key]
                    for key in ("index", "center_yx_um", "colorbar_bounds_yx_um")
                }
                for p in self._pupils
            ],
            "projection_roi": getattr(
                self.geometry, "layer_projection_rois_by_depth", {}
            ).get(self.depth),
            "source_mip_dtype": str(np.dtype(self.image.dtype)),
            "source_mip_sha256": getattr(
                self.geometry, "layer_projection_rois_by_depth", {}
            )
            .get(self.depth, {})
            .get("mip_sha256"),
            "source_channel_mip_sha256": getattr(
                self.geometry, "layer_projection_rois_by_depth", {}
            )
            .get(self.depth, {})
            .get("mip_sha256_by_channel"),
            "background": "RGB composite of local acquisition-layer Z projections"
            if self._rgb_background
            else "local acquisition-layer Z projection",
            "background_rgb_passthrough": self._rgb_background,
            "background_composite": getattr(self.geometry, "provenance", {}).get(
                "tissue_composite"
            ),
        }


def write_tissue_bigtiffs(
    regions: Sequence[Any],
    geometry: Any,
    output: Path,
    *,
    timepoint: int,
    pupil_diameter_um: float,
    scale_bar_um: float = 200,
    depth_workers: int = 1,
) -> dict[str, Any]:
    """Write independent native RGB BigTIFFs with at most three depth workers.

    Each worker owns its TIFF writer and bounded compression queue. The
    default is sequential; returned pages always follow sorted depth order.
    """
    if (
        not isinstance(depth_workers, (int, np.integer))
        or isinstance(depth_workers, bool)
        or not 1 <= depth_workers <= 3
    ):
        raise ValueError("Native TIFF depth workers must be an integer from 1 to 3")
    depths = sorted({int(region.depth) for region in regions})
    if not depths:
        raise ValueError("No AO depths selected for native tissue overlays")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    layers = getattr(geometry, "images_by_depth", {})
    for depth in depths:
        if depth not in layers:
            raise ValueError(f"Missing native local Z projection for depth {depth}")

    def write_depth(depth: int) -> dict[str, Any]:
        """Own all raster state and the output file for one acquisition depth."""
        raster = NativeTissueRaster(
            geometry,
            layers[depth],
            [r for r in regions if int(r.depth) == depth],
            depth=depth,
            timepoint=timepoint,
            pupil_diameter_um=pupil_diameter_um,
            scale_bar_um=scale_bar_um,
            display_limits=getattr(geometry, "display_limits_by_depth", {}).get(depth),
        )
        h, w = raster.shape_yx
        path = output / f"ao_tissue_overlay_t{timepoint:03d}_z{depth:03d}.tif"
        resolution = []
        for spacing in geometry.scale_yx_um[::-1]:
            value = Fraction(10000) / Fraction(str(float(spacing)))
            resolution.append((value.numerator, value.denominator))
        metadata = raster.metadata()
        with tifffile.TiffWriter(path, bigtiff=True) as writer:
            writer.write(
                raster.iter_tiles(),
                shape=(h, w, 3),
                dtype=np.uint8,
                photometric="rgb",
                tile=TILE_SHAPE,
                compression="deflate",
                compressionargs={"level": 4},
                predictor=True,
                maxworkers=TIFF_WORKERS,
                buffersize=TIFF_BUFFER_BYTES,
                resolution=tuple(resolution),
                resolutionunit="CENTIMETER",
                software="opm-processing-v2 AO tissue overlay",
                metadata={
                    "axes": "YXS",
                    "PhysicalSizeX": float(geometry.scale_yx_um[1]),
                    "PhysicalSizeY": float(geometry.scale_yx_um[0]),
                    "PhysicalSizeUnit": "micrometer",
                    "OriginYXUm": metadata["native_origin_yx_um"],
                    "ExtentYXUm": metadata["extent_yx_um"],
                    "AOOverlay": {
                        key: metadata[key]
                        for key in (
                            "timepoint",
                            "depth_index",
                            "pupil_display_diameter_um",
                            "tissue_display_limits",
                            "pixel_shape_yx",
                            "source_mip_dtype",
                            "source_mip_sha256",
                            "source_channel_mip_sha256",
                            "background",
                            "background_rgb_passthrough",
                            "background_composite",
                        )
                    },
                },
            )
        print(f"Saved native tissue overlay at depth {depth}: {path}", flush=True)
        return {"tif": str(path), "native_bigtiff": str(path), **metadata}

    workers = min(int(depth_workers), len(depths))
    if workers == 1:
        pages = [write_depth(depth) for depth in depths]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pages = list(executor.map(write_depth, depths))
    return {
        "format": "lossless tiled RGB BigTIFF",
        "full_resolution": True,
        "native_pixel_shape_yx": list(geometry.shape_yx),
        "tile_shape_yx": list(TILE_SHAPE),
        "compression": "deflate level 4 with horizontal predictor",
        "compression_workers": TIFF_WORKERS,
        "depth_workers": workers,
        "compression_buffer_bytes": TIFF_BUFFER_BYTES,
        "pupil_display_diameter_um": round(float(pupil_diameter_um), 3),
        "grid_line_alpha": 0.3,
        "pages": pages,
    }
