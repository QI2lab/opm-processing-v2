"""Inspect and open current filesystem-backed OME-Zarr OPM acquisitions.

Large stage scans embed complete frame histories in every position group.  The
inspector reads only the initial records needed to recover each stage
trajectory; pixel arrays are not opened until requested explicitly.
"""

from __future__ import annotations

import json
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
        """Return logical axis sizes keyed by lower-case axis name.

        Returns
        -------
        dict[str, int]
            Logical acquired dimensions keyed by lowercase axis name.
        """
        return dict(zip(self.axes, self.shape))

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return channel names in storage order.

        Returns
        -------
        tuple[str, ...]
            Channel names in stored channel order.
        """
        return tuple(channel.name for channel in self.channels)

    @property
    def tile_count(self) -> int:
        """Return the number of independently positioned image series.

        Returns
        -------
        int
            Number of independently positioned acquisition series.
        """
        return self.index_sizes.get("p", 1)

    @property
    def scan_position_count(self) -> int:
        """Return the number of scan-axis samples per tile.

        Returns
        -------
        int
            Scan-plane count per tile, or one for a planar acquisition.
        """
        return self.index_sizes.get("z", 1)

    @property
    def is_2d(self) -> bool:
        """Return whether each acquired T/P/C item has one spatial image plane.

        Time, position, and channel are iteration axes. An absent Z index and
        an explicit singleton Z index both describe two-dimensional acquired
        data; two or more Z samples describe a three-dimensional acquisition.

        Returns
        -------
        bool
            True for an absent or singleton scan-plane dimension.
        """
        return self.index_sizes.get("z", 1) == 1

    @property
    def scan_span_um(self) -> float | None:
        """Return the center-to-center span of the scan-axis samples.

        Returns
        -------
        float | None
            Center-to-center scan span in micrometers, or None without recorded spacing.
        """
        if self.scan_axis_step_um is None:
            return None
        return (self.scan_position_count - 1) * self.scan_axis_step_um

    @property
    def orientation_map(self) -> dict[str, str]:
        """Return acquisition orientation settings keyed by metadata name.

        Returns
        -------
        dict[str, str]
            Recorded acquisition orientation settings by name.
        """
        return dict(self.orientations)

    @property
    def stage_axis_flips_xyz(self) -> tuple[bool, bool, bool]:
        """Derive stage-coordinate flips from recorded camera orientation.

        Returns
        -------
        tuple[bool, bool, bool]
            Stage-placement sign reversals for X, Y, and Z.
        """
        orientations = {key: value.strip().lower() for key, value in self.orientations}

        def is_flipped(key: str) -> bool:
            """Interpret the recorded camera/stage orientation sign.

            Parameters
            ----------
            key : str
                Camera/stage orientation setting whose sign is interpreted.

            Returns
            -------
            bool
                True for a negative or reversed recorded orientation.
            """
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
        """Return whether stored scan samples must be reversed before deskew.

        Returns
        -------
        bool
            Whether scan samples must be reversed before production deskewing.
        """
        # During a stage scan the sample moves opposite the stage trajectory in
        # the stationary OPM imaging plane. The deskew convention follows sample
        # coordinates, so stage-scan samples must always be reversed regardless
        # of whether the recorded stage positions increase or decrease.
        if "stage" in self.mode.lower():
            return True

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
        """Return a JSON-serializable manifest.

        Returns
        -------
        dict[str, Any]
            JSON-compatible acquisition description with derived dimensions and geometry.
        """
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
    """Return a stable acquisition name for ``.zarr`` and ``.ome.zarr`` paths.

    Parameters
    ----------
    path : str | Path
        Acquisition path with a .zarr or .ome.zarr suffix.

    Returns
    -------
    str
        Acquisition name with its .ome.zarr or .zarr suffix removed.
    """
    name = Path(path).name
    for suffix in (".ome.zarr", ".zarr"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def resolve_acquisition_path(path: str | Path) -> Path:
    """Resolve either an acquisition store or its containing directory.

    Parameters
    ----------
    path : str | Path
        Acquisition store or directory containing one raw acquisition.

    Returns
    -------
    Path
        Absolute path to the single selected raw acquisition store.
    """
    candidate = Path(path).expanduser().resolve()
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
        root = yaozarrs.open_group(store)
        if "opm_v2" in root.attrs:
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
    """Convert an optional acquisition numeric value to a float.

    Parameters
    ----------
    value : Any
        Scalar or structured metadata value to convert.

    Returns
    -------
    float | None
        Float value, or None when an optional field is absent or nonnumeric.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any, default: int = 0) -> int:
    """Read an optional acquisition index with its default.

    Parameters
    ----------
    value : Any
        Scalar or structured metadata value to convert.
    default : int
        Integer returned when the metadata field is absent.

    Returns
    -------
    int
        Integer index or the supplied default.
    """
    number = _number(value)
    return default if number is None else int(number)


def _wavelength(name: str) -> float | None:
    """Infer the emission wavelength from a named acquisition channel.

    Parameters
    ----------
    name : str
        Stored channel name used to infer an emission wavelength in nanometers.

    Returns
    -------
    float | None
        Emission wavelength in nanometers, or None when absent from the name.
    """
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*nm\b", name, re.IGNORECASE)
    return float(match.group(1)) if match else None


def _event_parts(frame: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return normalized event index and event metadata for both schemas.

    Parameters
    ----------
    frame : dict[str, Any]
        Camera frame record containing event indices and instrument metadata.

    Returns
    -------
    tuple[dict, dict]
        Event index and instrument metadata dictionaries from the frame record.
    """
    if "event_index" in frame:
        return frame.get("event_index", {}), frame.get("event_metadata", {})
    event = frame.get("mda_event", {})
    return event.get("index", {}), event.get("metadata", {})


def _initial_frame_metadata(path: Path, count: int) -> list[dict[str, Any]]:
    """Decode at most ``count`` leading frame records without loading the journal.

    Parameters
    ----------
    path : Path
        Position-group zarr.json containing the frame history.
    count : int
        Maximum number of leading frame records to decode.

    Returns
    -------
    list[dict]
        Leading frame records available in the position journal, up to count.
    """
    if count < 1:
        return []
    marker = '"frame_metadata"'
    decoder = json.JSONDecoder()
    buffer = ""
    cursor: int | None = None
    frames: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        while len(frames) < count:
            if cursor is None:
                marker_index = buffer.find(marker)
                if marker_index >= 0:
                    array_start = buffer.find("[", marker_index + len(marker))
                    if array_start >= 0:
                        cursor = array_start + 1
            if cursor is not None:
                while len(frames) < count:
                    while cursor < len(buffer) and buffer[cursor] in " \t\r\n,":
                        cursor += 1
                    if cursor < len(buffer) and buffer[cursor] == "]":
                        return frames
                    try:
                        value, end = decoder.raw_decode(buffer, cursor)
                    except json.JSONDecodeError:
                        break
                    frames.append(value)
                    cursor = end
            chunk = stream.read(64 * 1024)
            if not chunk:
                if cursor is None:
                    return []
                if frames:
                    return frames
                raise ValueError(f"Invalid frame_metadata array: {path}")
            buffer += chunk
    return frames


def _array_layout(path: Path) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Read the small Zarr-v3 array document without opening its parent group.

    Parameters
    ----------
    path : Path
        Zarr array directory whose zarr.json supplies shape and dimension names.

    Returns
    -------
    tuple[tuple[int, ...], tuple[str, ...]]
        Stored array shape and lowercase dimension names.
    """
    metadata_path = Path(path) / "zarr.json"
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    shape = tuple(int(value) for value in document["shape"])
    names = tuple(str(value).lower() for value in document["dimension_names"])
    return shape, names


def _first_by_channel(frames: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Select the first initial frame recorded for each channel.

    Parameters
    ----------
    frames : list[dict[str, Any]]
        Leading acquisition frame records used to select one record per channel.

    Returns
    -------
    dict[int, dict[str, Any]]
        Channel indices mapped to their first acquisition frame.
    """
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
    """Combine stored channel names with recorded exposure and laser settings.

    Parameters
    ----------
    names : list[str]
        Ordered channel names from the OME image metadata.
    frames : list[dict[str, Any]]
        Numerically ordered projection TIFF paths for one position/channel.
    configured_powers : list[Any] | None
        Per-channel laser powers from acquisition configuration, when present.
    configured_exposures : list[Any] | None
        Per-channel exposure times in milliseconds from acquisition configuration.

    Returns
    -------
    tuple[ChannelMetadata, ...]
        Ordered channel records with wavelengths, exposures, and laser powers.
    """
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
    *,
    scan_position_count: int | None = None,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
    str | None,
    float | None,
]:
    """Recover per-position scan endpoints and stage motion from initial frames.

    Parameters
    ----------
    frame_sets : list[list[dict[str, Any]]]
        Initial frame records for each acquisition position.
    scan_position_count : int | None
        Total acquired scan planes used to extrapolate the stage trajectory.

    Returns
    -------
    tuple[tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...], str | None, float | None]
        Stage origins, scan endpoints, inferred scan axis, and step size.
    """
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
        if scan_position_count is not None and len(trajectory) > 1:
            first_step = tuple(
                trajectory[1][1][axis] - start[axis] for axis in range(3)
            )
            end = tuple(
                start[axis] + first_step[axis] * (scan_position_count - 1)
                for axis in range(3)
            )
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
    """List JSON sidecars beside the acquisition store.

    Parameters
    ----------
    path : Path
        Acquisition store whose neighboring JSON files are listed.

    Returns
    -------
    tuple[str, ...]
        Absolute paths of neighboring JSON sidecars.
    """
    return tuple(
        str(item.resolve())
        for item in sorted(path.parent.glob("*.json"))
        if item.resolve() != (path / "zarr.json").resolve()
    )


def _inspect_ome_zarr(path: Path, root: ZarrGroup) -> AcquisitionMetadata:
    """Normalize the current OPM OME-Zarr layout without parsing every journal.

    Parameters
    ----------
    path : Path
        Raw OME-Zarr acquisition directory.
    root : ZarrGroup
        Opened OME-Zarr root group used to avoid rereading root metadata.

    Returns
    -------
    AcquisitionMetadata
        Acquisition description assembled from OME metadata and initial frame records.
    """
    attributes = root.attrs
    opm = attributes["opm_v2"]
    layout = root.ome_metadata()
    if isinstance(layout, v05.Bf2Raw):
        ome_group = root["OME"]
        series_metadata = ome_group.ome_metadata()
        series_names = tuple(str(series_name) for series_name in series_metadata.series)
        first_image_group = root[series_names[0]]
    else:
        # ome-writers stores one-position acquisitions directly as an Image
        # group. Multi-position acquisitions use the Bio-Formats2Raw layout.
        series_names = ("",)
        first_image_group = root

    image_metadata = first_image_group.ome_metadata()
    multiscale = image_metadata.multiscales[0]
    dataset_path = str(multiscale.datasets[0].path)

    array_shapes: list[tuple[int, ...]] = []
    array_paths: list[str] = []
    dimension_names: tuple[str, ...] = ()
    for series_name in series_names:
        relative_path = f"{series_name}/{dataset_path}" if series_name else dataset_path
        current_shape, current_names = _array_layout(path / relative_path)
        array_paths.append(relative_path)
        array_shapes.append(current_shape)
        if current_names:
            dimension_names = current_names

    if not dimension_names:
        dimension_names = tuple("tczyx"[-len(array_shapes[0]) :])
    axes = (dimension_names[0], "p", *dimension_names[1:])
    shape = (array_shapes[0][0], len(series_names), *array_shapes[0][1:])

    channel_names: list[str] = []
    if image_metadata.omero is not None:
        channel_names = [
            str(channel.label or f"channel-{index}")
            for index, channel in enumerate(image_metadata.omero.channels)
        ]
    pixel_size_um: float | None = None
    axes_model = [axis.name for axis in multiscale.axes]
    scale = multiscale.datasets[0].scale_transform.scale
    if "x" in axes_model:
        pixel_size_um = _number(scale[axes_model.index("x")])

    channel_count = shape[axes.index("c")]
    scan_position_count = shape[axes.index("z")] if "z" in axes else 1
    initial_frame_count = channel_count * (2 if scan_position_count > 1 else 1)
    if len(series_names) == 1 and not series_names[0]:
        all_frames = list(
            first_image_group.attrs.get("ome_writers", {}).get("frame_metadata", [])
        )
        frame_sets = [all_frames[:initial_frame_count]]
    else:
        frame_sets = [
            _initial_frame_metadata(
                path / series_name / "zarr.json", initial_frame_count
            )
            for series_name in series_names
        ]

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
        channel_names = [
            found.get(index) or f"channel-{index}" for index in range(shape[2])
        ]
    channel_states = list(daq_config.get("channel_states", []))

    def enabled_values(key: str) -> list[Any]:
        """Select acquisition channel settings for the enabled channels.

        Parameters
        ----------
        key : str
            Acquisition configuration or orientation field to read.

        Returns
        -------
        list[Any]
            Configured values corresponding to the enabled acquisition channels.
        """
        values = list(daq_config.get(key, []))
        if len(channel_states) == len(values):
            values = [
                value for value, enabled in zip(values, channel_states) if enabled
            ]
        return values

    configured_powers = enabled_values("channel_powers")
    configured_exposures = enabled_values("channel_exposures_ms")
    channels = _channel_metadata(
        channel_names, frames, configured_powers, configured_exposures
    )
    positions, starts, ends, scan_axis, measured_step = _positions_from_frame_sets(
        frame_sets,
        scan_position_count=scan_position_count,
    )
    configured_step = _number(daq.get("scan_axis_step_um"))
    if configured_step is None:
        configured_step = _number(daq.get("image_mirror_step_um"))
    if configured_step is None:
        configured_step = _number(daq_config.get("scan_axis_step_um"))
    if configured_step is None:
        configured_step = _number(daq_config.get("image_mirror_step_um"))
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
        array_paths=tuple(array_paths),
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
        excess_scan_start_positions=_integer(
            opm_frame.get("excess_scan_start_positions")
        ),
        excess_scan_end_positions=_integer(opm_frame.get("excess_scan_end_positions")),
        orientations=orientations,
        sidecar_paths=_sidecars(path),
    )


def inspect_acquisition(
    path: str | Path, *, root: ZarrGroup | None = None
) -> AcquisitionMetadata:
    """Parse an OME-Zarr OPM manifest without opening or reading pixel data.

    Parameters
    ----------
    path : str | Path
        Acquisition store or containing directory to inspect without reading pixels.
    root : ZarrGroup | None
        Opened OME-Zarr root group used to avoid rereading root metadata.

    Returns
    -------
    AcquisitionMetadata
        Normalized acquisition dimensions, channels, calibration, and scan geometry.
    """
    store = resolve_acquisition_path(path)
    if root is None:
        root = yaozarrs.open_group(store)
    return _inspect_ome_zarr(store, root)


def open_acquisition_datastore(
    acquisition: AcquisitionMetadata | str | Path,
):
    """Open a logical TPCZYX TensorStore after metadata inspection.

    Parameters
    ----------
    acquisition : AcquisitionMetadata | str | Path
        Inspected acquisition dimensions, stage geometry, and camera calibration.

    Returns
    -------
    TensorStore
        Virtual TPCZYX view stacking the per-position TCZYX acquisition arrays.
    """
    import tensorstore as ts

    metadata = (
        acquisition
        if isinstance(acquisition, AcquisitionMetadata)
        else inspect_acquisition(acquisition)
    )
    arrays = []
    for array_path in metadata.array_paths:
        arrays.append(
            ts.open(
                {
                    "driver": "zarr3",
                    "kvstore": {
                        "driver": "file",
                        "path": str(metadata.path / array_path),
                    },
                }
            ).result()
        )
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
