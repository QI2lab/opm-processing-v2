"""Inspect and open current OME-Zarr OPM acquisitions with yaozarrs.

``inspect_acquisition`` traverses only yaozarrs metadata objects. Pixel arrays
are not opened until ``open_acquisition_datastore`` is called explicitly.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaozarrs
from yaozarrs import ZarrGroup, v05


@dataclass(frozen=True)
class ChannelMetadata:
    """Acquisition settings for one stored channel."""

    index: int
    name: str
    wavelength_nm: float | None
    exposure_ms: float | None
    laser_power: float | None


@dataclass(frozen=True)
class AcquisitionMetadata:
    """Normalized, metadata-only description of an OPM acquisition."""

    path: Path
    storage_format: str
    mode: str
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    array_paths: tuple[str, ...]
    acquisition_order: tuple[str, ...]
    channels: tuple[ChannelMetadata, ...]
    stage_positions_zxy: tuple[tuple[float, float, float], ...]
    scan_start_positions_xyz: tuple[tuple[float, float, float], ...]
    scan_end_positions_xyz: tuple[tuple[float, float, float], ...]
    scan_axis: str | None
    scan_axis_step_um: float | None
    pixel_size_um: float | None
    angle_deg: float | None
    camera_offset: float | None
    camera_conversion: float | None
    excess_scan_positions: int
    excess_scan_start_positions: int
    excess_scan_end_positions: int
    orientations: tuple[tuple[str, str], ...]
    sidecar_paths: tuple[str, ...]

    @property
    def index_sizes(self) -> dict[str, int]:
        """Return logical axis sizes keyed by lower-case axis name."""
        return dict(zip(self.axes, self.shape))

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return channel names in storage order."""
        return tuple(channel.name for channel in self.channels)

    @property
    def tile_count(self) -> int:
        """Return the number of independently positioned image series."""
        return self.index_sizes.get("p", 1)

    @property
    def scan_position_count(self) -> int:
        """Return the number of scan-axis samples per tile."""
        return self.index_sizes.get("z", 1)

    @property
    def scan_span_um(self) -> float | None:
        """Return the center-to-center span of the scan-axis samples."""
        if self.scan_axis_step_um is None:
            return None
        return (self.scan_position_count - 1) * self.scan_axis_step_um

    @property
    def orientation_map(self) -> dict[str, str]:
        """Return acquisition orientation settings keyed by metadata name."""
        return dict(self.orientations)

    @property
    def stage_axis_flips_xyz(self) -> tuple[bool, bool, bool]:
        """Derive stage-coordinate flips from recorded camera orientation."""
        orientations = {
            key: value.strip().lower() for key, value in self.orientations
        }

        def is_flipped(key: str) -> bool:
            return orientations.get(key, "normal") in {
                "negative",
                "flipped",
                "reverse",
                "reversed",
            }

        xy_flipped = is_flipped("camera_XYstage_orientation")
        return (
            xy_flipped,
            xy_flipped,
            is_flipped("camera_Zstage_orientation"),
        )

    @property
    def scan_axis_reversed(self) -> bool:
        """Return whether stored scan samples run toward decreasing stage values."""
        if (
            self.scan_axis is not None
            and self.scan_axis in "xyz"
            and self.scan_start_positions_xyz
        ):
            axis = "xyz".index(self.scan_axis)
            deltas = [
                end[axis] - start[axis]
                for start, end in zip(
                    self.scan_start_positions_xyz,
                    self.scan_end_positions_xyz,
                )
                if end[axis] != start[axis]
            ]
            if deltas:
                return sum(deltas) < 0
        orientation_key = (
            "camera_mirror_orientation"
            if "mirror" in self.mode
            else "camera_XYstage_orientation"
        )
        return self.orientation_map.get(orientation_key, "normal").lower() in {
            "negative",
            "flipped",
            "reverse",
            "reversed",
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest."""
        result = asdict(self)
        result["path"] = str(self.path)
        result["index_sizes"] = self.index_sizes
        result["channel_names"] = list(self.channel_names)
        result["tile_count"] = self.tile_count
        result["scan_position_count"] = self.scan_position_count
        result["scan_span_um"] = self.scan_span_um
        result["channel_count"] = len(self.channels)
        result["stage_axis_flips_xyz"] = self.stage_axis_flips_xyz
        result["scan_axis_reversed"] = self.scan_axis_reversed
        return result


def acquisition_stem(path: str | Path) -> str:
    """Return a stable acquisition name for ``.zarr`` and ``.ome.zarr`` paths."""
    name = Path(path).name
    for suffix in (".ome.zarr", ".zarr"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def resolve_acquisition_path(path: str | Path) -> Path:
    """Resolve either an acquisition store or its containing directory."""
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir():
        raise ValueError(f"Acquisition path is not a directory: {candidate}")
    if (candidate / "zarr.json").is_file() or (candidate / ".zattrs").is_file():
        return candidate
    stores = sorted(
        item
        for item in candidate.iterdir()
        if item.is_dir()
        and item.name.endswith(".zarr")
        and ((item / "zarr.json").is_file() or (item / ".zattrs").is_file())
    )
    if len(stores) == 1:
        return stores[0]

    # Processing adds more Zarr stores beside the source acquisition.  Current
    # OPM-v2 acquisitions are uniquely marked by acquisition-only metadata at
    # the root, so use yaozarrs to disambiguate without opening pixel arrays.
    opm_v2_stores = []
    for store in stores:
        try:
            root = yaozarrs.open_group(store)
        except (OSError, TypeError, ValueError):
            continue
        opm_v2 = root.attrs.get("opm_v2")
        if isinstance(opm_v2, dict) and {
            "acquisition_order",
            "configuration",
            "index_sizes",
        }.issubset(opm_v2):
            opm_v2_stores.append(store)

    if len(opm_v2_stores) == 1:
        return opm_v2_stores[0]

    if len(opm_v2_stores) > 1:
        matches = ", ".join(str(store) for store in opm_v2_stores)
        raise ValueError(
            f"Expected one OPM-v2 acquisition Zarr store in {candidate}, "
            f"found {len(opm_v2_stores)}: {matches}"
        )

    if len(stores) != 1:
        raise ValueError(
            f"Expected exactly one acquisition Zarr store in {candidate}, "
            f"found {len(stores)}"
        )
    return stores[0]


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any, default: int = 0) -> int:
    number = _number(value)
    return default if number is None else int(number)


def _wavelength(name: str) -> float | None:
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*nm\b", name, re.IGNORECASE)
    return float(match.group(1)) if match else None


def _event_parts(frame: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return normalized event index and event metadata for both schemas."""
    if "event_index" in frame:
        return frame.get("event_index", {}), frame.get("event_metadata", {})
    event = frame.get("mda_event", {})
    return event.get("index", {}), event.get("metadata", {})


def _first_by_channel(frames: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for frame in frames:
        index, _ = _event_parts(frame)
        channel = _integer(index.get("c"), -1)
        if channel >= 0 and channel not in selected:
            selected[channel] = frame
    return selected


def _channel_metadata(
    names: list[str],
    frames: list[dict[str, Any]],
    configured_powers: list[Any] | None = None,
    configured_exposures: list[Any] | None = None,
) -> tuple[ChannelMetadata, ...]:
    by_channel = _first_by_channel(frames)
    channels: list[ChannelMetadata] = []
    for channel_index, name in enumerate(names):
        frame = by_channel.get(channel_index, {})
        _, metadata = _event_parts(frame)
        daq = metadata.get("DAQ", {})
        camera = metadata.get("Camera", {})
        exposure = _number(camera.get("exposure_ms"))
        if exposure is None:
            exposure_seconds = _number(frame.get("exposure_time"))
            if exposure_seconds is not None:
                exposure = exposure_seconds * 1000.0
        if exposure is None and configured_exposures:
            if channel_index < len(configured_exposures):
                exposure = _number(configured_exposures[channel_index])
        powers = configured_powers or daq.get("laser_powers", [])
        power = _number(powers[channel_index]) if channel_index < len(powers) else None
        channels.append(
            ChannelMetadata(
                index=channel_index,
                name=name,
                wavelength_nm=_wavelength(name),
                exposure_ms=exposure,
                laser_power=power,
            )
        )
    return tuple(channels)


def _positions_from_frame_sets(
    frame_sets: list[list[dict[str, Any]]],
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
    str | None,
    float | None,
]:
    positions_zxy: list[tuple[float, float, float]] = []
    starts_xyz: list[tuple[float, float, float]] = []
    ends_xyz: list[tuple[float, float, float]] = []
    scan_steps: list[tuple[str, float]] = []
    for frames in frame_sets:
        candidates: list[tuple[int, int, int, tuple[float, float, float]]] = []
        for frame in frames:
            index, metadata = _event_parts(frame)
            stage = metadata.get("Stage", {})
            xyz = tuple(_number(stage.get(f"{axis}_pos")) for axis in "xyz")
            if any(value is None for value in xyz):
                continue
            candidates.append(
                (
                    _integer(index.get("t"), 0),
                    _integer(index.get("z"), 0),
                    _integer(index.get("c"), 0),
                    xyz,  # type: ignore[arg-type]
                )
            )
        if not candidates:
            continue
        # Frames are supplied per series; use one channel from the first timepoint.
        first_time = min(item[0] for item in candidates)
        first_channel = min(item[2] for item in candidates if item[0] == first_time)
        trajectory = sorted(
            {
                z: xyz
                for time, z, channel, xyz in candidates
                if time == first_time and channel == first_channel
            }.items()
        )
        if not trajectory:
            continue
        start = trajectory[0][1]
        end = trajectory[-1][1]
        starts_xyz.append(start)
        ends_xyz.append(end)
        positions_zxy.append((start[2], start[0], start[1]))
        if len(trajectory) > 1:
            delta = [trajectory[1][1][i] - start[i] for i in range(3)]
            axis_index = max(range(3), key=lambda i: abs(delta[i]))
            if abs(delta[axis_index]) > 0:
                scan_steps.append(("xyz"[axis_index], abs(delta[axis_index])))
    scan_axis = scan_steps[0][0] if scan_steps else None
    scan_step = scan_steps[0][1] if scan_steps else None
    return (
        tuple(positions_zxy),
        tuple(starts_xyz),
        tuple(ends_xyz),
        scan_axis,
        scan_step,
    )


def _sidecars(path: Path) -> tuple[str, ...]:
    return tuple(
        str(item.resolve())
        for item in sorted(path.parent.glob("*.json"))
        if item.resolve() != (path / "zarr.json").resolve()
    )


def _inspect_ome_zarr(path: Path, root: ZarrGroup) -> AcquisitionMetadata:
    """Normalize an OME-Zarr collection using yaozarrs metadata models."""
    attributes = root.attrs
    opm = attributes.get("opm_v2")
    if not isinstance(opm, dict):
        raise ValueError(f"OME-Zarr store lacks opm_v2 acquisition metadata: {path}")
    layout = root.ome_metadata()
    if not isinstance(layout, v05.Bf2Raw):
        raise ValueError(f"Expected an OME-Zarr v0.5 Bio-Formats2Raw layout: {path}")
    ome_group = root["OME"]
    if not isinstance(ome_group, ZarrGroup):
        raise ValueError(f"OME metadata node is not a group: {path}")
    series_metadata = ome_group.ome_metadata()
    if not isinstance(series_metadata, v05.Series):
        raise ValueError(f"OME group lacks typed series metadata: {path}")
    series = series_metadata.series

    frame_sets: list[list[dict[str, Any]]] = []
    array_shapes: list[tuple[int, ...]] = []
    dimension_names: tuple[str, ...] = ()
    channel_names: list[str] = []
    pixel_size_um: float | None = None
    for position_index, series_name in enumerate(series):
        image_group = root[str(series_name)]
        if not isinstance(image_group, ZarrGroup):
            raise ValueError(f"Series {series_name} is not an image group")
        frame_sets.append(
            list(image_group.attrs.get("ome_writers", {}).get("frame_metadata", []))
        )
        image_metadata = image_group.ome_metadata()
        if not isinstance(image_metadata, v05.Image):
            raise ValueError(f"Series {series_name} lacks typed OME image metadata")
        if len(image_metadata.multiscales) != 1:
            raise ValueError(f"Series {series_name} must contain one multiscale image")
        multiscale = image_metadata.multiscales[0]
        if len(multiscale.datasets) < 1:
            raise ValueError(f"Series {series_name} has no image datasets")
        dataset_path = multiscale.datasets[0].path
        array = image_group[dataset_path]
        shape = tuple(int(value) for value in (array.metadata.shape or ()))
        if not shape:
            raise ValueError(f"Series {series_name} has no array shape")
        array_shapes.append(shape)
        names = tuple(
            str(value).lower()
            for value in (getattr(array.metadata, "dimension_names", None) or [])
        )
        if names:
            dimension_names = names
        if position_index == 0:
            if image_metadata.omero is not None:
                channel_names = [
                    str(channel.label or f"channel-{index}")
                    for index, channel in enumerate(image_metadata.omero.channels)
                ]
            axes_model = [axis.name for axis in multiscale.axes]
            scale = multiscale.datasets[0].scale_transform.scale
            if "x" in axes_model:
                pixel_size_um = _number(scale[axes_model.index("x")])

    if any(shape != array_shapes[0] for shape in array_shapes[1:]):
        raise ValueError("All OPM position series must have the same array shape")
    if not dimension_names:
        dimension_names = tuple("tczyx"[-len(array_shapes[0]) :])
    axes = (dimension_names[0], "p", *dimension_names[1:])
    shape = (array_shapes[0][0], len(series), *array_shapes[0][1:])
    sizes = opm.get("index_sizes", {})
    for axis, expected in sizes.items():
        if axis in axes and int(expected) != shape[axes.index(axis)]:
            raise ValueError(
                f"opm_v2 index size {axis}={expected} disagrees with array shape {shape}"
            )

    frames = [frame for frame_set in frame_sets for frame in frame_set]
    first_frame = frames[0] if frames else {}
    _, first_metadata = _event_parts(first_frame)
    daq = first_metadata.get("DAQ", {})
    camera = first_metadata.get("Camera", {})
    opm_frame = first_metadata.get("OPM", {})
    config = opm.get("configuration", {}).get("acq_config", {})
    daq_config = config.get("DAQ", {})
    if not channel_names:
        found: dict[int, str] = {}
        for frame in frames:
            index, metadata = _event_parts(frame)
            found.setdefault(
                _integer(index.get("c"), 0),
                str(metadata.get("DAQ", {}).get("current_channel", "")),
            )
        channel_names = [found.get(index) or f"channel-{index}" for index in range(shape[2])]
    channel_states = list(daq_config.get("channel_states", []))

    def enabled_values(key: str) -> list[Any]:
        values = list(daq_config.get(key, []))
        if len(channel_states) == len(values):
            values = [value for value, enabled in zip(values, channel_states) if enabled]
        return values

    configured_powers = enabled_values("channel_powers")
    configured_exposures = enabled_values("channel_exposures_ms")
    channels = _channel_metadata(
        channel_names, frames, configured_powers, configured_exposures
    )
    positions, starts, ends, scan_axis, measured_step = _positions_from_frame_sets(frame_sets)
    configured_step = _number(daq.get("scan_axis_step_um"))
    if configured_step is None:
        configured_step = _number(daq_config.get("scan_axis_step_um"))
    scan_step = abs(configured_step) if configured_step is not None else measured_step
    if measured_step is not None and configured_step is not None:
        tolerance = max(1e-6, abs(configured_step) * 1e-3)
        if abs(measured_step - abs(configured_step)) > tolerance:
            raise ValueError(
                f"Measured scan step {measured_step} um disagrees with metadata "
                f"step {configured_step} um"
            )

    if pixel_size_um is None:
        pixel_size_um = _number(first_frame.get("pixel_size_um"))
    orientations = tuple(
        (key, str(value))
        for key, value in opm_frame.items()
        if key.endswith("_orientation")
    )
    mode = str(daq.get("mode") or config.get("opm_mode") or "unknown")
    return AcquisitionMetadata(
        path=path,
        storage_format="opm-v2-ome-zarr-v3",
        mode=mode,
        axes=axes,
        shape=shape,
        array_paths=tuple(
            f"{name}/{root[str(name)].ome_metadata().multiscales[0].datasets[0].path}"
            for name in series
        ),
        acquisition_order=tuple(str(axis) for axis in opm.get("acquisition_order", [])),
        channels=channels,
        stage_positions_zxy=positions,
        scan_start_positions_xyz=starts,
        scan_end_positions_xyz=ends,
        scan_axis=scan_axis,
        scan_axis_step_um=scan_step,
        pixel_size_um=pixel_size_um,
        angle_deg=_number(opm_frame.get("angle_deg")),
        camera_offset=_number(camera.get("offset")),
        camera_conversion=_number(camera.get("e_to_ADU")),
        excess_scan_positions=_integer(opm_frame.get("excess_scan_positions")),
        excess_scan_start_positions=_integer(opm_frame.get("excess_scan_start_positions")),
        excess_scan_end_positions=_integer(opm_frame.get("excess_scan_end_positions")),
        orientations=orientations,
        sidecar_paths=_sidecars(path),
    )


def inspect_acquisition(
    path: str | Path, *, root: ZarrGroup | None = None
) -> AcquisitionMetadata:
    """Parse an OME-Zarr OPM manifest without opening or reading pixel data."""
    store = resolve_acquisition_path(path)
    if root is None:
        try:
            root = yaozarrs.open_group(store)
        except ValueError as error:
            raise ValueError(
                "The shared acquisition inspector requires a group-based OME-Zarr "
                f"store; {store} is a legacy root-array acquisition"
            ) from error
    return _inspect_ome_zarr(store, root)


def open_acquisition_datastore(
    acquisition: AcquisitionMetadata | str | Path,
):
    """Open a logical TPCZYX TensorStore after metadata inspection."""
    import tensorstore as ts

    metadata = (
        acquisition
        if isinstance(acquisition, AcquisitionMetadata)
        else inspect_acquisition(acquisition)
    )
    root = yaozarrs.open_group(metadata.path)
    series_metadata = root["OME"].ome_metadata()
    arrays = [
        root[series_name][array_path.split("/", 1)[1]].to_tensorstore()
        for series_name, array_path in zip(
            series_metadata.series, metadata.array_paths, strict=True
        )
    ]
    if not arrays:
        raise ValueError(f"Acquisition has no arrays: {metadata.path}")
    return ts.stack(arrays, axis=1)


def main() -> None:
    """Print a metadata-only acquisition manifest as JSON."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("path", type=Path, help="Zarr store or containing directory")
    arguments = parser.parse_args()
    print(json.dumps(inspect_acquisition(arguments.path).to_dict(), indent=2))


if __name__ == "__main__":
    main()
