"""Replace opaque pupil phase pixels while retaining existing tissue TIFF tiles."""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imagecodecs
import numpy as np
import tifffile

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from scripts.ao_tissue_bigtiff import NativeTissueRaster


def _patches(
    raster: NativeTissueRaster, y0: int, y1: int, x0: int, x1: int
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Sample new phase RGBA on the same global native pixel centers."""
    patches = []
    for pupil in raster._pupils:
        top, bottom, left, right = pupil["bounds"]
        ya, yb = max(y0, math.ceil(top - 0.5)), min(y1, math.ceil(bottom - 0.5))
        xa, xb = max(x0, math.ceil(left - 0.5)), min(x1, math.ceil(right - 0.5))
        if ya >= yb or xa >= xb:
            continue
        source = pupil["rgba"]
        if source.dtype != np.uint8 or source.ndim != 3 or source.shape[-1] != 4:
            raise ValueError("Pupil replacement requires uint8 RGBA phase patches")
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
        alpha = patch[..., 3]
        if np.any((alpha != 0) & (alpha != 255)):
            raise ValueError(
                "Phase replacement supports only opaque or transparent pupil pixels"
            )
        if np.any(alpha):
            patches.append(((ya, yb, xa, xb), patch))
    return patches


def rephase_tissue_bigtiff(
    source: Path,
    destination: Path,
    raster: NativeTissueRaster,
    *,
    phase_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy compressed tissue tiles and replace only opaque new pupil samples.

    Source and destination must be different files. The caller supplies the
    original pupil placement and unchanged masks/color limits in ``raster``;
    its background array is never read. Existing grids, local colorbars,
    scale bar, image pixels outside pupil masks, and spatial metadata remain
    unchanged. No source image or metadata file is written.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Rephasing must write a separate TIFF, never its source")
    if destination.exists():
        raise FileExistsError(f"Rephased TIFF already exists: {destination}")
    provenance = json.loads(json.dumps(dict(phase_provenance), allow_nan=False))
    if not provenance:
        raise ValueError("Rephased pupils require an explicit phase reference")
    raster_metadata = raster.metadata()
    counts = {"copied_compressed_tiles": 0, "reencoded_tiles": 0}
    with tifffile.TiffFile(source) as original:
        if not original.is_bigtiff or len(original.pages) != 1:
            raise ValueError("Pupil replacement requires a single-page native BigTIFF")
        page = original.pages[0]
        if (
            not page.is_tiled
            or page.shape != (*raster.shape_yx, 3)
            or page.dtype != np.dtype("uint8")
            or int(page.photometric) != 2
            or int(page.planarconfig) != 1
            or int(page.samplesperpixel) != 3
            or int(page.compression) not in (8, 32946)
            or int(page.predictor) != 2
        ):
            raise ValueError(
                "Pupil replacement requires tiled contiguous RGB uint8 Deflate with horizontal prediction"
            )
        header = json.loads(page.description)
        if (
            header["OriginYXUm"] != raster_metadata["native_origin_yx_um"]
            or header["ExtentYXUm"] != raster_metadata["extent_yx_um"]
            or header["AOOverlay"]["pupil_display_diameter_um"]
            != raster_metadata["pupil_display_diameter_um"]
        ):
            raise ValueError(
                "Replacement pupils must retain the original physical canvas and display diameter"
            )
        header["AOOverlay"]["phase_provenance"] = provenance
        header["AOOverlay"]["pupil_phase_limit_rad"] = raster_metadata[
            "pupil_phase_limit_rad"
        ]
        header["AOOverlay"]["pupil_normalization"] = raster_metadata[
            "pupil_normalization"
        ]
        ny, nx = (
            math.ceil(page.shape[0] / page.tilelength),
            math.ceil(page.shape[1] / page.tilewidth),
        )
        total = ny * nx
        if len(page.dataoffsets) != total or len(page.databytecounts) != total:
            raise ValueError("Source TIFF tile index is incomplete")
        file_size = source.stat().st_size
        for offset, count in zip(page.dataoffsets, page.databytecounts, strict=True):
            if (
                int(offset) <= 0
                or int(count) <= 0
                or int(offset) + int(count) > file_size
            ):
                raise ValueError(
                    "Source TIFF has an empty or truncated compressed tile"
                )
        x_resolution = page.tags["XResolution"].value
        y_resolution = page.tags["YResolution"].value
        resolution_unit = page.tags["ResolutionUnit"].value
        tile_shape = (page.tilelength, page.tilewidth)

        def pieces() -> Iterator[bytes]:
            """Yield already compressed segments in the original TIFF tile order."""
            with source.open("rb") as stream:
                for index in range(total):
                    y0, x0 = (index // nx) * tile_shape[0], (index % nx) * tile_shape[1]
                    y1, x1 = (
                        min(y0 + tile_shape[0], page.shape[0]),
                        min(x0 + tile_shape[1], page.shape[1]),
                    )
                    replacements = _patches(raster, y0, y1, x0, x1)
                    stream.seek(page.dataoffsets[index])
                    compressed = stream.read(page.databytecounts[index])
                    if len(compressed) != page.databytecounts[index]:
                        raise ValueError(
                            "Source TIFF changed or became truncated during rephasing"
                        )
                    if replacements:
                        decoded, location, _shape = page.decode(compressed, index)
                        if (
                            decoded is None
                            or decoded.shape != (1, *tile_shape, 3)
                            or tuple(location[2:4]) != (y0, x0)
                        ):
                            raise ValueError(
                                "Source TIFF tile has an unexpected decoded layout"
                            )
                        tile = np.array(
                            decoded[0], dtype=np.uint8, copy=True, order="C"
                        )
                        for (ya, yb, xa, xb), patch in replacements:
                            target = tile[ya - y0 : yb - y0, xa - x0 : xb - x0]
                            opaque = patch[..., 3] == 255
                            target[opaque] = patch[..., :3][opaque]
                        predicted = imagecodecs.delta_encode(tile, axis=-2)
                        compressed = zlib.compress(
                            memoryview(predicted).cast("B"), level=4
                        )
                        counts["reencoded_tiles"] += 1
                    else:
                        counts["copied_compressed_tiles"] += 1
                    yield compressed
                    if (index + 1) % 256 == 0:
                        print(
                            f"Rephased native depth {raster.depth}: {index + 1}/{total} tiles",
                            flush=True,
                        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tifffile.TiffWriter(
            destination, mode="x", bigtiff=True, byteorder=original.byteorder
        ) as writer:
            writer.write(
                pieces(),
                shape=page.shape,
                dtype=np.uint8,
                photometric="rgb",
                tile=tile_shape,
                compression=int(page.compression),
                predictor=2,
                resolution=(x_resolution, y_resolution),
                resolutionunit=resolution_unit,
                description=json.dumps(header, ensure_ascii=True, allow_nan=False),
                metadata=None,
                software="opm-processing-v2 AO pupil rephase",
                maxworkers=1,
                buffersize=4 * 1024 * 1024,
            )
    print(f"Saved rephased native depth {raster.depth}: {destination}", flush=True)
    return {
        "source_tif": str(source),
        "tif": str(destination),
        "native_bigtiff": str(destination),
        "depth_index": raster.depth,
        "timepoint": raster.timepoint,
        "pixel_shape_yx": list(raster.shape_yx),
        "phase_provenance": provenance,
        "pupil_phase_limit_rad": raster_metadata["pupil_phase_limit_rad"],
        "pupil_normalization": raster_metadata["pupil_normalization"],
        "regions": raster_metadata["regions"],
        **counts,
    }
