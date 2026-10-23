"""Metadata-only reports for raw oblique acquisitions and laboratory images."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ome_types import from_xml
from yaozarrs import open_group, v05

from opm_processing.dataio.acquisition import (
    AcquisitionMetadata,
    _event_parts,
    _initial_frame_metadata,
    acquisition_stem,
    inspect_acquisition,
)
from opm_processing.dataio.processing_state import ProcessingState


def hardware_timing(
    frames: list[dict], channel_count: int, scan_positions: int
) -> dict[str, Any]:
    """Calculate hardware-triggered timing from actual camera frame exposures.

    Parameters
    ----------
    frames : list of dict
        Camera frame records in either supported acquisition metadata format.
    channel_count : int
        Number of stored camera channels, including combined-laser images once.
    scan_positions : int
        Acquired scan positions, including planes subsequently discarded.

    Returns
    -------
    dict
        Channel-ordered exposures and volume time in milliseconds. Unknown
        exposures remain None; software arrival times and requested camera
        settings do not determine hardware timing.
    """
    exposures: list[float | None] = [None] * channel_count
    for frame in frames:
        if frame.get("exposure_time") is not None:
            exposure, factor = frame["exposure_time"], 1000
        elif frame.get("exposure_ms") is not None:
            exposure, factor = frame["exposure_ms"], 1
        else:
            continue
        try:
            value = round(float(exposure) * factor, 2)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid camera frame exposure") from error
        if isinstance(exposure, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("Camera frame exposures must be finite and positive")
        index, _ = _event_parts(frame)
        channel = int(index.get("c", 0))
        if not 0 <= channel < channel_count:
            raise ValueError(
                "Camera frame channel index is outside the stored channels"
            )
        if exposures[channel] is not None and exposures[channel] != value:
            raise ValueError("Camera frame exposures vary between scans or positions")
        exposures[channel] = value
    volume = (
        round(scan_positions * sum(exposures), 6)
        if exposures and all(value is not None for value in exposures)
        else None
    )
    return {"frame_exposure_ms": exposures, "volume_time_ms": volume}


def raw_report(acquisition: AcquisitionMetadata) -> dict[str, Any]:
    """Describe scan and tilted-camera coordinates without relabeling stored data.

    Parameters
    ----------
    acquisition : AcquisitionMetadata
        Metadata inspected from an existing raw acquisition.

    Returns
    -------
    dict
        Raw dimensions in TPCS Y-prime X order, camera calibration and hardware
        timing. Y-prime is distance along the camera plane, not laboratory Y.
    """
    report = acquisition.to_dict()
    names = {"z": "s", "y": "y'"}
    axes = [names.get(axis, axis) for axis in acquisition.axes]
    report["axes"] = axes
    report["index_sizes"] = dict(zip(axes, acquisition.shape, strict=True))
    report["acquisition_order"] = [
        names.get(axis, axis) for axis in acquisition.acquisition_order
    ]
    report["coordinate_frame"] = "oblique"
    report["dimension_names"] = {
        "s": "scan",
        "y'": "tilted camera plane",
        "x": "camera X",
    }
    pitch = acquisition.pixel_size_um
    report["sampling_um"] = {
        "s": acquisition.scan_axis_step_um,
        "y'": pitch,
        "x": pitch,
    }
    report["camera_field_of_view_um"] = {
        names.get(axis, axis): size * pitch if pitch is not None else None
        for axis, size in zip(acquisition.axes, acquisition.shape, strict=True)
        if axis in ("y", "x")
    }
    report["y_prime_sampling_projected_y_um"] = (
        pitch * math.cos(math.radians(acquisition.angle_deg))
        if pitch is not None and acquisition.angle_deg is not None
        else None
    )
    count = len(acquisition.channels) * (
        2 if acquisition.scan_position_count > 1 else 1
    )
    frames = [
        frame
        for array_path in acquisition.array_paths
        for frame in _initial_frame_metadata(
            acquisition.path / Path(array_path).parent / "zarr.json", count
        )
    ]
    timing = hardware_timing(
        frames, len(acquisition.channels), acquisition.scan_position_count
    )
    report["timing"] = timing
    for channel, exposure in zip(
        report["channels"], timing["frame_exposure_ms"], strict=True
    ):
        channel["exposure_ms"] = exposure
    return report


def processed_report(path: Path) -> dict[str, Any]:
    """Read each processed image's laboratory sampling and physical dimensions.

    Parameters
    ----------
    path : pathlib.Path
        Processed root Image or Bio-Formats2Raw position collection.

    Returns
    -------
    dict
        Per-series dimensions, channels, ZYX spacing, origins and physical
        volume dimensions. Dimensions cover N voxels, each with its recorded
        spacing; they are not the center-to-center span of N minus one voxels.
    """
    root = open_group(path)
    layout = root.ome_metadata()
    series = (
        root["OME"].ome_metadata().series if isinstance(layout, v05.Bf2Raw) else ("",)
    )
    companion = path / "OME" / "METADATA.ome.xml"
    ome = (
        from_xml(companion.read_text(encoding="utf-8")) if companion.is_file() else None
    )
    images = []
    units = {"meter": 1e6, "millimeter": 1e3, "micrometer": 1, "nanometer": 1e-3}
    for series_index, name in enumerate(series):
        group = root[str(name)] if name else root
        metadata = group.ome_metadata()
        if not isinstance(metadata, v05.Image):
            raise ValueError(f"Expected an OME-Zarr image in {path}")
        multiscale = metadata.multiscales[0]
        dataset = multiscale.datasets[0]
        shape = group[dataset.path].metadata.shape
        axes = [axis.name for axis in multiscale.axes]
        scale, origin = [1.0] * len(axes), [0.0] * len(axes)
        for transform in (
            *dataset.coordinateTransformations,
            *(multiscale.coordinateTransformations or ()),
        ):
            if transform.type == "scale":
                scale = [
                    value * factor
                    for value, factor in zip(scale, transform.scale, strict=True)
                ]
                origin = [
                    value * factor
                    for value, factor in zip(origin, transform.scale, strict=True)
                ]
            else:
                origin = [
                    value + offset
                    for value, offset in zip(origin, transform.translation, strict=True)
                ]
        spacing_um, origin_um, dimensions_um = {}, {}, {}
        for axis, size, step, offset in zip(
            multiscale.axes, shape, scale, origin, strict=True
        ):
            if axis.name not in "zyx":
                continue
            factor = units.get(axis.unit)
            spacing_um[axis.name] = None if factor is None else step * factor
            origin_um[axis.name] = None if factor is None else offset * factor
            dimensions_um[axis.name] = None if factor is None else size * step * factor
        images.append(
            {
                "series": str(name),
                "axes": axes,
                "shape": list(shape),
                "index_sizes": dict(zip(axes, shape, strict=True)),
                "sampling_um": spacing_um,
                "origin_um": origin_um,
                "volume_dimensions_um": dimensions_um,
                "channels": [channel.label for channel in metadata.omero.channels]
                if metadata.omero
                else [
                    channel.name for channel in ome.images[series_index].pixels.channels
                ]
                if ome is not None
                else [],
                "pyramid_levels": len(multiscale.datasets),
            }
        )
    return {"path": str(path), "coordinate_frame": "laboratory", "images": images}


def inspect_dataset(path: str | Path) -> dict[str, Any]:
    """Inspect available raw and processed stores without opening pixel arrays.

    Parameters
    ----------
    path : str or pathlib.Path
        Raw store, processed store or directory containing acquisition products.

    Returns
    -------
    dict
        Raw acquisition report when available and associated processed images.
        Processing journals associate custom output names and source paths.
    """
    path = Path(path).expanduser().resolve()
    direct = (path / "zarr.json").is_file() or (path / ".zattrs").is_file()
    directory = path.parent if direct else path
    stores = sorted(item for item in directory.glob("*.zarr") if item.is_dir())
    if direct and path not in stores:
        stores.append(path)
    roots = {store: open_group(store) for store in stores}
    raw_paths = [store for store, root in roots.items() if "opm_v2" in root.attrs]
    if direct and path in raw_paths:
        raw_paths = [path]
    elif direct:
        raw_paths = [
            store
            for store in raw_paths
            if path.name.startswith(acquisition_stem(store) + "_")
        ]
    if len(raw_paths) > 1:
        raise ValueError("Multiple raw acquisitions found; specify one raw store")
    associated = set()
    for sidecar in directory.glob("*.processing.json"):
        state = ProcessingState.read(sidecar)
        source = Path(state.document["source"]["path"])
        if not source.is_absolute():
            source = sidecar.parent / source
        source = source.resolve()
        products = {sidecar.parent / name for name in state.document["outputs"]}
        products.update(
            sidecar.parent / record[key]
            for record in state.document["registration"].values()
            for key in ("fused_path", "max_projection_path")
            if record.get(key)
        )
        if source in raw_paths or (direct and path in products):
            associated.update(products)
            if source.is_dir() and not raw_paths:
                raw_paths = [source]
    raw = raw_report(inspect_acquisition(raw_paths[0])) if raw_paths else None
    if raw is not None:
        prefix = acquisition_stem(raw["path"]) + "_"
        associated.update(
            store
            for store in stores
            if store.name.startswith(prefix) and "opm_v2" not in roots[store].attrs
        )
    elif not direct:
        associated.update(store for store in stores if store not in raw_paths)
    if direct and path not in raw_paths:
        associated = {path}
    processed = []
    for store in sorted(associated):
        if store.is_dir():
            product = processed_report(store)
            product["timing"] = raw["timing"] if raw is not None else None
            processed.append(product)
    if raw is None and not processed:
        raise ValueError(f"No raw or processed OME-Zarr data found in {path}")
    return {"raw": raw, "processed": processed}
