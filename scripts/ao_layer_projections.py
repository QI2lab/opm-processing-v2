"""Calculate local layer maxima from acquisition-defined three-dimensional ROIs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from yaozarrs import open_group

from opm_processing.dataio.acquisition import inspect_acquisition
from opm_processing.dataio.position_collection import open_position_collection
from opm_processing.dataio.processing_state import ProcessingState, resolve_store_path
from opm_processing.imageprocessing.coordinates import stage_z_level_indices
from opm_processing.imageprocessing.opmtools import deskew_shape_estimator

if __package__:
    from scripts.ao_tissue_geometry import _compose_transform
else:
    from ao_tissue_geometry import _compose_transform

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


XY_BLOCK_SHAPE = (1024, 1024)
READ_WORKERS = 8
CACHE_CHECKPOINT_BLOCKS = 8
CACHE_SCHEMA = "opm-ao-layer-projection-cache-v1"
CACHE_LOCK_NAME = ".layer_projection.lock"


def _prepare_projection(
    geometry: Any, timepoint: int, channel: int, depths: Sequence[int]
) -> tuple[Any, dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    """Define physical camera slabs and map their ROIs into the existing fusion."""
    if geometry.provenance.get("full_resolution") is not True:
        raise ValueError(
            "Local layer projections require full-resolution tissue geometry"
        )
    selected_depths = sorted({int(depth) for depth in depths})
    if not selected_depths or selected_depths[0] < 0:
        raise ValueError("Select at least one nonnegative acquisition depth")
    preview = Path(geometry.provenance["projection"])
    state = ProcessingState.read(Path(geometry.provenance["processing_state"]))
    processed = state.registered_output_for_max_projection(preview)
    registration = state.registration(processed)
    if not registration.get("fused_path") or not registration.get("tiles"):
        raise ValueError(
            "Local layer projections require a completed full-volume registration"
        )
    if state.roi_series(processed):
        raise ValueError(
            "Local layer projection requires the full-acquisition tile collection"
        )
    fused = resolve_store_path(registration["fused_path"], state.path.parent)
    group = open_group(fused)
    scales = group.attrs.get("ome", {}).get("multiscales", [])
    if len(scales) != 1:
        raise ValueError("Expected one calibrated fused-volume multiscale")
    multiscale = scales[0]
    axes = multiscale["axes"]
    if [axis["name"] for axis in axes] != list("tczyx") or any(
        axis.get("unit") != "micrometer" for axis in axes[-3:]
    ):
        raise ValueError(
            "Local projections require fused TCZYX coordinates in micrometers"
        )
    datasets = multiscale["datasets"]
    if not datasets:
        raise ValueError("Fused volume has no image datasets")
    dataset = min(
        datasets,
        key=lambda item: float(np.prod(_compose_transform(item, multiscale)[0][-3:])),
    )
    scale, origin = _compose_transform(dataset, multiscale)
    native = group[dataset["path"]].to_tensorstore()
    shape = tuple(int(value) for value in native.shape)
    if len(shape) != 5 or shape[2] <= 1:
        raise ValueError("Local projections require a three-dimensional fused volume")
    if not 0 <= timepoint < shape[0] or not 0 <= channel < shape[1]:
        raise ValueError(
            "Requested layer projection timepoint or channel is outside the volume"
        )
    if (
        tuple(geometry.shape_yx) != shape[-2:]
        or not np.allclose(geometry.scale_yx_um, scale[-2:], rtol=0, atol=1e-9)
        or not np.allclose(
            geometry.origin_yx_um, origin[-2:], rtol=0, atol=0.0005 + 1e-9
        )
    ):
        raise ValueError(
            "Fused volume and tissue canvas have incompatible native XY calibration"
        )

    acquisition = inspect_acquisition(state.source_path())
    if "stage" not in acquisition.mode.lower() or any(acquisition.stage_axis_flips_xyz):
        raise ValueError(
            "Acquisition-defined layer projections require unflipped stage geometry"
        )
    collection = open_position_collection(processed)
    voxel = np.asarray(collection.voxel_size_um, dtype=float)
    if not np.allclose(voxel, scale[-3:], rtol=0, atol=1e-9):
        raise ValueError(
            "Processed camera slabs and fused volume disagree on native sampling"
        )
    pixel = float(acquisition.pixel_size_um)
    angle = float(acquisition.angle_deg)
    camera_y = int(acquisition.index_sizes["y"])
    averaging = round(float(voxel[0]) / pixel)
    if averaging < 1 or abs(round(averaging * pixel, 3) - voxel[0]) > 1e-9:
        raise ValueError(
            "Processed Z sampling is incompatible with camera-pixel averaging"
        )
    scan_count = (
        acquisition.scan_position_count
        - (acquisition.excess_scan_start_positions or acquisition.excess_scan_positions)
        - acquisition.excess_scan_end_positions
    )
    estimated, _, _, _ = deskew_shape_estimator(
        (scan_count, camera_y, int(acquisition.index_sizes["x"])),
        theta=angle,
        distance=float(acquisition.scan_axis_step_um),
        pixel_size=pixel,
        crop_after_deskew=False,
        downsample_factor=averaging,
    )
    slab_planes = int(estimated[0])
    first_bin_center = round(0.5 * (averaging - 1) * pixel, 3)
    camera_axial_span = camera_y * pixel * math.sin(math.radians(angle))
    stage = np.asarray(acquisition.stage_positions_zxy, dtype=float)
    levels = stage_z_level_indices(stage)
    local_origins = np.asarray(collection.spatial_origins_zyx_um, dtype=float)
    if len(collection.arrays) != len(stage) or local_origins.shape != stage.shape:
        raise ValueError(
            "Source slabs do not correspond to all recorded acquisition positions"
        )
    rois: dict[int, list[dict[str, Any]]] = {depth: [] for depth in selected_depths}
    seen = set()
    for tile in registration["tiles"]:
        if int(tile["time_index"]) != timepoint:
            continue
        position = int(tile["position_index"])
        if not 0 <= position < len(stage) or position in seen:
            raise ValueError(
                "Registration has invalid or duplicate acquisition source positions"
            )
        seen.add(position)
        depth = int(levels[position])
        if depth not in rois:
            continue
        source_shape = tuple(
            int(value) for value in collection.arrays[position].shape[-3:]
        )
        if source_shape[0] != slab_planes:
            raise ValueError(
                "Source slab Z size differs from acquisition ROI height and illumination angle"
            )
        if abs(local_origins[position, 0] - first_bin_center) > 0.0005 + 1e-9:
            raise ValueError(
                "Source slab has an unsupported axial crop or detector reference"
            )
        nominal = (
            np.array((-stage[position, 0], -stage[position, 1], stage[position, 2]))
            + local_origins[position]
        )
        # Define the axial source ROI from acquisition camera geometry first.
        # Registration supplies only its known coordinate mapping into pixels
        # already present in the fused image.
        nominal[0] = -stage[position, 0] + first_bin_center
        registered = np.asarray(tile["origin_zyx_um"], dtype=float)
        translation = registered - nominal
        mapped = nominal + translation
        start = np.rint((mapped - origin[-3:]) / voxel).astype(np.int64)
        stop = start + np.asarray(source_shape)
        clipped_start = np.maximum(start, 0)
        clipped_stop = np.minimum(stop, shape[-3:])
        if np.any(clipped_start >= clipped_stop):
            continue
        nominal_edges = (
            nominal[0] - 0.5 * voxel[0],
            nominal[0] + (slab_planes - 0.5) * voxel[0],
        )
        rois[depth].append(
            {
                "position_index": position,
                "time_index": timepoint,
                "raw_stage_z_um": round(float(stage[position, 0]), 3),
                "nominal_acquisition_origin_zyx_um": [
                    round(float(value), 3) for value in nominal
                ],
                "nominal_axial_voxel_edges_um": [
                    round(float(value), 3) for value in nominal_edges
                ],
                "registration_translation_zyx_um": [
                    round(float(value), 3) for value in translation
                ],
                "registered_voxel_box_zyx": [
                    [int(a), int(b)]
                    for a, b in zip(clipped_start, clipped_stop, strict=True)
                ],
                "unclipped_registered_voxel_box_zyx": [
                    [int(a), int(b)] for a, b in zip(start, stop, strict=True)
                ],
            }
        )
    provenance = {}
    for depth, boxes in rois.items():
        if not boxes:
            raise ValueError(
                f"No acquisition-defined layer ROIs remain at depth {depth}"
            )
        z0 = min(box["registered_voxel_box_zyx"][0][0] for box in boxes)
        z1 = max(box["registered_voxel_box_zyx"][0][1] for box in boxes)
        provenance[depth] = {
            "fused_source": str(fused),
            "dataset": dataset["path"],
            "processed_source": str(processed),
            "raw_acquisition": str(acquisition.path),
            "timepoint": int(timepoint),
            "channel": int(channel),
            "depth_index": depth,
            "native_shape_zyx": list(shape[-3:]),
            "native_scale_zyx_um": voxel.tolist(),
            "native_origin_zyx_um": origin[-3:].tolist(),
            "camera_roi_height_pixels": camera_y,
            "camera_pixel_size_um": pixel,
            "illumination_angle_deg": angle,
            "camera_roi_axial_span_um": round(camera_axial_span, 3),
            "source_slab_z_planes": slab_planes,
            "z_averaging_factor": averaging,
            "first_z_bin_center_um": first_bin_center,
            "aggregate_z_start_stop": [z0, z1],
            "aggregate_z_voxel_edges_um": [
                round(float(origin[2] + (index - 0.5) * voxel[0]), 3)
                for index in (z0, z1)
            ],
            "roi_definition": "union of acquisition camera slabs at recorded stage Z, mapped through each source tile's known registration translation",
            "projection": "local maximum only within each layer's mapped 3D ROI at each XY position",
            "tile_rois": boxes,
        }
    return native, rois, provenance


def _source_metadata_fingerprints(
    geometry: Any, provenance: dict[int, dict[str, Any]]
) -> list[dict[str, str]]:
    """Fingerprint source calibration and ROI metadata, without hashing pixels."""
    first = next(iter(provenance.values()))
    fused, processed, raw = (
        Path(first[key])
        for key in ("fused_source", "processed_source", "raw_acquisition")
    )
    preview = Path(geometry.provenance["projection"])
    paths = {
        fused / "zarr.json",
        fused / str(first["dataset"]) / "zarr.json",
        processed / "zarr.json",
        raw / "zarr.json",
        preview / "zarr.json",
        preview / str(geometry.provenance.get("preview_dataset", "0")) / "zarr.json",
        Path(geometry.provenance["processing_state"]),
    }
    xml = processed / "OME" / "METADATA.ome.xml"
    if xml.is_file():
        paths.add(xml)
    positions = {
        int(roi["position_index"])
        for record in provenance.values()
        for roi in record["tile_rois"]
    }
    for position in positions:
        metadata = processed / str(position) / "zarr.json"
        paths.add(metadata)
        document = json.loads(metadata.read_bytes())
        dataset = document["attributes"]["ome"]["multiscales"][0]["datasets"][0]["path"]
        paths.add(processed / str(position) / dataset / "zarr.json")
    records = []
    for path in sorted(paths):
        if not path.is_file():
            raise ValueError(f"Projection cache source metadata is missing: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append({"path": str(path.resolve()), "sha256": digest})
    return records


def _cache_json(value: Any) -> bytes:
    """Serialize finite cache metadata deterministically."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


@contextmanager
def _projection_cache_lock(directory: Path) -> Iterator[None]:
    """Hold an exclusive OS lock that is released automatically on process exit."""
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CACHE_LOCK_NAME
    if path.resolve() != path:
        raise ValueError(
            "Projection cache lock must not redirect through a symbolic link"
        )
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            def lock() -> None:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

            def unlock() -> None:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def lock() -> None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def unlock() -> None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

        try:
            lock()
        except OSError as error:
            raise ValueError("Projection cache already has an active writer") from error
        try:
            yield
        finally:
            unlock()


def _write_cache_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Durably replace the completion bitmap after its image writes are flushed."""
    descriptor, temporary = tempfile.mkstemp(
        prefix=".layer-cache-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_cache_json(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _flush_cache_arrays(images: dict[int, np.ndarray], paths: dict[int, Path]) -> None:
    """Make native output pixels durable before publishing completed blocks."""
    for depth, image in images.items():
        image.flush()
        with paths[depth].open("r+b") as stream:
            os.fsync(stream.fileno())


def _open_projection_cache(
    directory: Path,
    context: dict[str, Any],
    shape_yx: tuple[int, int],
    dtype: np.dtype,
    depths: Sequence[int],
    block_count: int,
) -> tuple[dict[int, np.ndarray], np.ndarray, dict[str, Any]]:
    """Open strictly matching output memmaps, or initialize them in row strips."""
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "layer_projection_cache.json"
    fingerprint = hashlib.sha256(_cache_json(context)).hexdigest()
    expected_images = {
        str(depth): {
            "path": f"layer_{depth:03d}.npy",
            "shape_yx": list(shape_yx),
            "dtype": dtype.str,
        }
        for depth in depths
    }
    if path.exists():
        manifest = json.loads(path.read_bytes())
        if (
            manifest.get("schema") != CACHE_SCHEMA
            or manifest.get("fingerprint") != fingerprint
            or manifest.get("context") != context
        ):
            raise ValueError(
                "Projection cache fingerprint does not match source metadata or ROI geometry"
            )
        if manifest.get("images") != expected_images or manifest.get("phase") not in (
            "initializing",
            "active",
            "complete",
        ):
            raise ValueError("Projection cache image layout is incompatible")
        completed = manifest.get("completed_xy_blocks", [])
        if len(completed) != block_count or any(
            type(value) is not bool for value in completed
        ):
            raise ValueError("Projection cache completion bitmap is invalid")
        if manifest["phase"] == "initializing" and any(completed):
            raise ValueError("Projection cache initialization bitmap is inconsistent")
        if manifest["phase"] == "complete" and not all(completed):
            raise ValueError(
                "Projection cache is marked complete with unfinished blocks"
            )
    else:
        if any(entry.name != CACHE_LOCK_NAME for entry in directory.iterdir()):
            raise ValueError(
                "Projection cache directory is not empty and has no valid manifest"
            )
        manifest = {
            "schema": CACHE_SCHEMA,
            "fingerprint": fingerprint,
            "context": context,
            "images": expected_images,
            "phase": "initializing",
            "completed_xy_blocks": [False] * block_count,
            "volume_read_bytes": 0,
        }
        _write_cache_manifest(path, manifest)
    paths = {depth: directory / expected_images[str(depth)]["path"] for depth in depths}
    images = {}
    initialize = manifest["phase"] == "initializing"
    for depth, image_path in paths.items():
        if image_path.resolve() != image_path:
            raise ValueError(
                "Projection cache images must not redirect through symbolic links"
            )
        if initialize:
            image = np.lib.format.open_memmap(
                image_path, mode="w+", dtype=dtype, shape=shape_yx
            )
            rows = max(1, min(256, (32 * 1024**2) // (shape_yx[1] * dtype.itemsize)))
            for row in range(0, shape_yx[0], rows):
                image[row : row + rows].fill(0)
                if row % (8 * rows) == 0:
                    image.flush()
        else:
            try:
                image = np.lib.format.open_memmap(image_path, mode="r")
                if (
                    image.shape != shape_yx
                    or image.dtype != dtype
                    or not image.flags.c_contiguous
                    or image_path.stat().st_size != image.offset + image.nbytes
                ):
                    raise ValueError(
                        "Projection cache image shape, dtype, or size is invalid"
                    )
                image = np.lib.format.open_memmap(image_path, mode="r+")
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"Projection cache image is invalid: {image_path}"
                ) from error
        images[depth] = image
    if initialize:
        _flush_cache_arrays(images, paths)
        manifest["phase"] = "active"
        _write_cache_manifest(path, manifest)
    return (
        images,
        np.asarray(manifest["completed_xy_blocks"], dtype=bool),
        {
            "path": path,
            "manifest": manifest,
            "image_paths": paths,
        },
    )


def _image_sha(image: np.ndarray) -> str:
    """Hash contiguous row strips without copying a native image into memory."""
    digest = hashlib.sha256()
    for row in range(0, image.shape[0], 256):
        digest.update(memoryview(image[row : row + 256]).cast("B"))
    return digest.hexdigest()


def _checkpoint_projection_cache(
    images: dict[int, np.ndarray],
    completed_blocks: np.ndarray,
    cache: dict[str, Any],
    volume_read_bytes: int,
) -> None:
    """Publish completed projection blocks only after their pixels are durable."""
    _flush_cache_arrays(images, cache["image_paths"])
    cache["manifest"]["completed_xy_blocks"] = completed_blocks.tolist()
    cache["manifest"]["volume_read_bytes"] = volume_read_bytes
    _write_cache_manifest(cache["path"], cache["manifest"])


def _finalize_projection_cache(
    images: dict[int, np.ndarray], cache: dict[str, Any] | None
) -> dict[str, str]:
    """Hash completed native projections and publish their cache completion."""
    hashes = {str(depth): _image_sha(image) for depth, image in images.items()}
    if cache is not None:
        expected_hashes = cache["manifest"].get("mip_sha256_by_depth")
        if cache["manifest"]["phase"] == "complete" and expected_hashes != hashes:
            raise ValueError("Projection cache image checksum mismatch")
        cache["manifest"]["mip_sha256_by_depth"] = hashes
        cache["manifest"]["phase"] = "complete"
        _write_cache_manifest(cache["path"], cache["manifest"])
    return hashes


def _read_layer_projections_unlocked(
    geometry: Any,
    *,
    timepoint: int,
    channel: int,
    depths: Sequence[int],
    cache_dir: Path | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    """Project acquisition layer ROIs without materializing the fused 3D volume.

    Parameters
    ----------
    geometry : TissueGeometry
        Full-resolution calibrated tissue canvas and its processing provenance.
    timepoint, channel : int
        Fused image indices to read.
    depths : sequence of int
        Logical acquisition depth layers to project separately.
    cache_dir : pathlib.Path or None
        Native memory-mapped output cache. Matching completed XY blocks resume
        after interruption; source image pixels are never written.

    Returns
    -------
    tuple of dict
        Native two-dimensional maxima and physical ROI provenance per depth.
        Pixels outside a layer's source ROI union remain zero.
    """
    native, rois, provenance = _prepare_projection(geometry, timepoint, channel, depths)
    dtype = np.dtype(getattr(native.dtype, "numpy_dtype", native.dtype))
    if dtype.kind not in "uif":
        raise ValueError("Layer maxima require a real numeric fused image")
    shape_yx = tuple(int(value) for value in native.shape[-2:])
    blocks = [
        (
            y0,
            min(y0 + XY_BLOCK_SHAPE[0], shape_yx[0]),
            x0,
            min(x0 + XY_BLOCK_SHAPE[1], shape_yx[1]),
        )
        for y0 in range(0, shape_yx[0], XY_BLOCK_SHAPE[0])
        for x0 in range(0, shape_yx[1], XY_BLOCK_SHAPE[1])
    ]
    cache = None
    if cache_dir is None:
        images = {depth: np.zeros(shape_yx, dtype=dtype) for depth in rois}
        completed_blocks = np.zeros(len(blocks), dtype=bool)
    else:
        print("Checking local projection cache metadata...", flush=True)
        context = json.loads(
            _cache_json(
                {
                    "shape_yx": list(shape_yx),
                    "dtype": dtype.str,
                    "xy_block_shape": list(XY_BLOCK_SHAPE),
                    "depths": list(rois),
                    "timepoint": int(timepoint),
                    "channel": int(channel),
                    "canvas_origin_yx_um": list(geometry.origin_yx_um),
                    "canvas_scale_yx_um": list(geometry.scale_yx_um),
                    "layer_rois": provenance,
                    "source_metadata": _source_metadata_fingerprints(
                        geometry, provenance
                    ),
                }
            )
        )
        images, completed_blocks, cache = _open_projection_cache(
            Path(cache_dir), context, shape_yx, dtype, list(rois), len(blocks)
        )

    def project_block(bounds: tuple[int, int, int, int]) -> int:
        """Read one XY block once and apply only its intersecting layer ROIs."""
        y0, y1, x0, x1 = bounds
        hits = []
        for depth, boxes in rois.items():
            for roi in boxes:
                (za, zb), (ya, yb), (xa, xb) = roi["registered_voxel_box_zyx"]
                if ya < y1 and yb > y0 and xa < x1 and xb > x0:
                    hits.append(
                        (
                            depth,
                            za,
                            zb,
                            max(ya, y0),
                            min(yb, y1),
                            max(xa, x0),
                            min(xb, x1),
                        )
                    )
        if not hits:
            return 0
        z0 = min(hit[1] for hit in hits)
        z1 = max(hit[2] for hit in hits)
        volume = np.asarray(
            native[timepoint, channel, z0:z1, y0:y1, x0:x1].read().result()
        )
        covered = {
            depth: np.zeros((y1 - y0, x1 - x0), dtype=bool)
            for depth in {hit[0] for hit in hits}
        }
        for depth, za, zb, ya, yb, xa, xb in hits:
            local_max = volume[
                za - z0 : zb - z0, ya - y0 : yb - y0, xa - x0 : xb - x0
            ].max(axis=0)
            target = images[depth][ya:yb, xa:xb]
            mask = covered[depth][ya - y0 : yb - y0, xa - x0 : xb - x0]
            np.maximum(target, local_max, out=target, where=mask)
            np.copyto(target, local_max, where=~mask)
            mask[:] = True
        return int(volume.nbytes)

    total = len(blocks)
    resumed = completed = int(completed_blocks.sum())
    bytes_read = int(cache["manifest"].get("volume_read_bytes", 0)) if cache else 0
    interval = max(1, min(64, total // 20))
    pending_blocks = iter(
        (index, bounds)
        for index, bounds in enumerate(blocks)
        if not completed_blocks[index]
    )
    since_checkpoint = 0

    def checkpoint() -> None:
        """Publish only completed block writes after their native cache is durable."""
        if cache is not None:
            _checkpoint_projection_cache(images, completed_blocks, cache, bytes_read)

    print(
        f"Local layer projections: {completed}/{total} XY blocks ({resumed} resumed)",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=READ_WORKERS) as executor:
        pending = {}
        for _ in range(READ_WORKERS):
            item = next(pending_blocks, None)
            if item is not None:
                index, bounds = item
                pending[executor.submit(project_block, bounds)] = index
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                bytes_read += future.result()
                completed_blocks[pending.pop(future)] = True
                completed += 1
                since_checkpoint += 1
                if since_checkpoint >= CACHE_CHECKPOINT_BLOCKS:
                    checkpoint()
                    since_checkpoint = 0
                if completed == total or completed % interval == 0:
                    print(
                        f"Local layer projections: {completed}/{total} XY blocks",
                        flush=True,
                    )
                item = next(pending_blocks, None)
                if item is not None:
                    index, bounds = item
                    pending[executor.submit(project_block, bounds)] = index
    checkpoint()
    if (
        cache is not None
        and _source_metadata_fingerprints(geometry, provenance)
        != cache["manifest"]["context"]["source_metadata"]
    ):
        raise ValueError(
            "Projection source metadata changed during calculation; cache cannot be reused"
        )
    hashes = _finalize_projection_cache(images, cache)
    for depth, record in provenance.items():
        record["xy_block_shape"] = list(XY_BLOCK_SHAPE)
        record["read_workers"] = READ_WORKERS
        record["shared_volume_read_bytes"] = bytes_read
        record["mip_sha256"] = hashes[str(depth)]
        record["resumed_xy_blocks"] = resumed
        if cache is not None:
            record["projection_cache"] = str(cache["path"])
            record["projection_cache_fingerprint"] = cache["manifest"]["fingerprint"]
    return images, provenance


def read_layer_projections(
    geometry: Any,
    *,
    timepoint: int,
    channel: int,
    depths: Sequence[int],
    cache_dir: Path | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    """Calculate native layer maxima with an optional exclusive resumable cache.

    Parameters
    ----------
    geometry : TissueGeometry
        Calibrated native canvas and acquisition/processing provenance.
    timepoint, channel : int
        Fused-volume indices.
    depths : sequence of int
        Logical acquisition layers to project independently.
    cache_dir : pathlib.Path or None
        Output memmap cache directory. One writer holds an OS lock for the full
        call; matching completed blocks resume after interruption.

    Returns
    -------
    tuple of dict
        Native layer images and their physical ROI/cache provenance.
    """
    if cache_dir is None:
        return _read_layer_projections_unlocked(
            geometry, timepoint=timepoint, channel=channel, depths=depths
        )
    with _projection_cache_lock(Path(cache_dir)):
        return _read_layer_projections_unlocked(
            geometry,
            timepoint=timepoint,
            channel=channel,
            depths=depths,
            cache_dir=Path(cache_dir),
        )
