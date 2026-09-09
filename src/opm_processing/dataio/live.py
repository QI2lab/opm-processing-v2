"""Live-acquisition sidecars and raw OME-Zarr tile readiness."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterator

from opm_processing.dataio.acquisition import (
    AcquisitionMetadata,
    ChannelMetadata,
    acquisition_stem,
)

LIVE_MANIFEST_SCHEMA = "opm_v2.live_acquisition"
LIVE_MANIFEST_VERSION = "1.0"
LIVE_POLL_INTERVAL_SECONDS = 30.0
_TERMINAL_EVENTS = {"completed", "canceled", "errored"}


@dataclass(frozen=True)
class LiveSidecars:
    """Paths shared by acquisition control and live processing."""

    manifest: Path
    log: Path


@dataclass(frozen=True)
class LiveManifest:
    """Immutable acquisition plan required by live processing."""

    path: Path
    acquisition_id: str
    data_path: Path
    mode: str
    index_sizes: dict[str, int]
    acquisition_order: tuple[str, ...]
    channels: tuple[ChannelMetadata, ...]
    stage_positions_zxy: tuple[tuple[float, float, float], ...]
    scan_axis: str | None
    scan_axis_step_um: float
    pixel_size_um: float
    angle_deg: float
    camera_offset: float
    camera_conversion: float
    excess_scan_positions: int
    excess_scan_start_positions: int
    excess_scan_end_positions: int
    orientations: tuple[tuple[str, str], ...]

    @classmethod
    def read(cls, path: str | Path) -> "LiveManifest":
        """Read and validate one live-acquisition manifest."""
        manifest_path = Path(path).expanduser().resolve()
        with manifest_path.open(encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, dict):
            raise ValueError(f"Live manifest must contain a JSON object: {path}")
        if document.get("schema") != LIVE_MANIFEST_SCHEMA:
            raise ValueError(
                f"Unsupported live manifest schema: {document.get('schema')!r}"
            )
        if str(document.get("schema_version")) != LIVE_MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported live manifest version: {document.get('schema_version')!r}"
            )

        acquisition_id = str(document.get("acquisition_id", "")).strip()
        if not acquisition_id:
            raise ValueError("Live manifest lacks acquisition_id")
        raw_data_path = document.get("data_path")
        if not isinstance(raw_data_path, str) or not raw_data_path.strip():
            raise ValueError("Live manifest lacks data_path")
        data_path = Path(raw_data_path).expanduser()
        if not data_path.is_absolute():
            data_path = manifest_path.parent / data_path
        data_path = data_path.resolve()

        sizes_document = document.get("index_sizes")
        if not isinstance(sizes_document, dict):
            raise ValueError("Live manifest lacks index_sizes")
        index_sizes = {
            axis: int(sizes_document.get(axis, 1)) for axis in ("t", "p", "c", "z")
        }
        if any(size < 1 for size in index_sizes.values()):
            raise ValueError("Live manifest index sizes must all be positive")

        acquisition_order = tuple(
            str(axis) for axis in document.get("acquisition_order", ())
        )
        if set(acquisition_order) != set(index_sizes):
            raise ValueError(
                "Live manifest acquisition_order must contain t, p, c, and z once"
            )

        channel_documents = document.get("channels")
        if (
            not isinstance(channel_documents, list)
            or len(channel_documents) != index_sizes["c"]
        ):
            raise ValueError("Live manifest must contain one channel entry per channel")
        channels = []
        for index, channel in enumerate(channel_documents):
            if not isinstance(channel, dict):
                raise ValueError("Live manifest channel entries must be objects")
            name = str(channel.get("name", "")).strip()
            if not name:
                raise ValueError(f"Live manifest channel {index} lacks a name")
            channels.append(
                ChannelMetadata(
                    index=index,
                    name=name,
                    wavelength_nm=_optional_float(channel.get("wavelength_nm")),
                    exposure_ms=_optional_float(channel.get("exposure_ms")),
                    laser_power=_optional_float(channel.get("laser_power")),
                )
            )

        position_documents = document.get("stage_positions_zxy")
        if (
            not isinstance(position_documents, list)
            or len(position_documents) != index_sizes["p"]
        ):
            raise ValueError(
                "Live manifest must contain one ZXY stage position per position"
            )
        stage_positions = tuple(
            _coordinate_triplet(position, "stage position")
            for position in position_documents
        )

        orientations_document = document.get("orientations", {})
        if not isinstance(orientations_document, dict):
            raise ValueError("Live manifest orientations must be an object")

        return cls(
            path=manifest_path,
            acquisition_id=acquisition_id,
            data_path=data_path,
            mode=str(document.get("mode", "")).strip(),
            index_sizes=index_sizes,
            acquisition_order=acquisition_order,
            channels=tuple(channels),
            stage_positions_zxy=stage_positions,
            scan_axis=_optional_string(document.get("scan_axis")),
            scan_axis_step_um=_required_float(document, "scan_axis_step_um"),
            pixel_size_um=_required_float(document, "pixel_size_um"),
            angle_deg=_required_float(document, "angle_deg"),
            camera_offset=_required_float(document, "camera_offset"),
            camera_conversion=_required_float(document, "camera_e_to_adu"),
            excess_scan_positions=int(document.get("excess_scan_positions", 0)),
            excess_scan_start_positions=int(
                document.get("excess_scan_start_positions", 0)
            ),
            excess_scan_end_positions=int(document.get("excess_scan_end_positions", 0)),
            orientations=tuple(
                (str(key), str(value)) for key, value in orientations_document.items()
            ),
        )

    def apply(self, acquisition: AcquisitionMetadata) -> AcquisitionMetadata:
        """Overlay upfront manifest values onto inspected OME-Zarr metadata."""
        if acquisition.path.resolve() != self.data_path:
            raise ValueError(
                "Live manifest data_path does not match the requested acquisition: "
                f"{self.data_path} != {acquisition.path.resolve()}"
            )
        actual_sizes = acquisition.index_sizes
        for axis, expected in self.index_sizes.items():
            if actual_sizes.get(axis, 1) != expected:
                raise ValueError(
                    f"Live manifest {axis}={expected} disagrees with OME-Zarr "
                    f"{axis}={actual_sizes.get(axis, 1)}"
                )
        return replace(
            acquisition,
            mode=self.mode,
            acquisition_order=self.acquisition_order,
            channels=self.channels,
            stage_positions_zxy=self.stage_positions_zxy,
            scan_axis=self.scan_axis,
            scan_axis_step_um=self.scan_axis_step_um,
            pixel_size_um=self.pixel_size_um,
            angle_deg=self.angle_deg,
            camera_offset=self.camera_offset,
            camera_conversion=self.camera_conversion,
            excess_scan_positions=self.excess_scan_positions,
            excess_scan_start_positions=self.excess_scan_start_positions,
            excess_scan_end_positions=self.excess_scan_end_positions,
            orientations=self.orientations,
            sidecar_paths=tuple(
                dict.fromkeys((*acquisition.sidecar_paths, str(self.path)))
            ),
        )


def resolve_live_sidecars(acquisition_path: str | Path) -> LiveSidecars:
    """Return the single manifest and log associated with an acquisition."""
    path = Path(acquisition_path).expanduser().resolve()
    stem = acquisition_stem(path)
    return LiveSidecars(
        manifest=path.parent / f"{stem}.manifest.json",
        log=path.parent / f"{stem}.log.jsonl",
    )


def read_lifecycle_event(path: str | Path, acquisition_id: str) -> str | None:
    """Return the most recent recognized acquisition lifecycle event.

    A final unterminated line is ignored because it may be observed while the
    controller is appending it.
    """
    log_path = Path(path)
    if not log_path.is_file():
        return None
    contents = log_path.read_text(encoding="utf-8")
    lines = contents.splitlines(keepends=True)
    latest: str | None = None
    for index, line in enumerate(lines):
        if index == len(lines) - 1 and not line.endswith(("\n", "\r")):
            continue
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL record in {log_path}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Lifecycle records must be JSON objects: {log_path}")
        record_id = record.get("acquisition_id")
        if record_id is not None and str(record_id) != acquisition_id:
            raise ValueError(
                f"Lifecycle log acquisition_id {record_id!r} does not match "
                f"manifest acquisition_id {acquisition_id!r}"
            )
        event = str(record.get("event", "")).casefold()
        if event in {"started", *_TERMINAL_EVENTS}:
            latest = event
    return latest


class ZarrTileReadiness:
    """Determine complete T/P tiles from raw Zarr v3 chunk keys."""

    def __init__(self, acquisition: AcquisitionMetadata) -> None:
        if acquisition.axes != ("t", "p", "c", "z", "y", "x"):
            raise ValueError(
                f"Live deskewing requires logical TPCZYX input; got {acquisition.axes}"
            )
        self.acquisition = acquisition
        self._arrays = tuple(
            _LiveArrayChunks.read(acquisition.path / relative_path)
            for relative_path in acquisition.array_paths
        )
        if len(self._arrays) != acquisition.index_sizes["p"]:
            raise ValueError("OME-Zarr array count does not match position count")

    def tile_is_ready(self, time_index: int, position_index: int) -> bool:
        """Return whether every raw chunk for one T/P tile exists."""
        if not 0 <= time_index < self.acquisition.index_sizes["t"]:
            raise IndexError(f"Time index is out of bounds: {time_index}")
        if not 0 <= position_index < self.acquisition.index_sizes["p"]:
            raise IndexError(f"Position index is out of bounds: {position_index}")
        return self._arrays[position_index].tile_is_ready(time_index)

    def ready_tiles(self) -> set[tuple[int, int]]:
        """Return every tile whose complete raw chunk set is present."""
        return {
            (time_index, position_index)
            for time_index in range(self.acquisition.index_sizes["t"])
            for position_index in range(self.acquisition.index_sizes["p"])
            if self.tile_is_ready(time_index, position_index)
        }


@dataclass(frozen=True)
class _LiveArrayChunks:
    path: Path
    shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    encoding: str
    separator: str

    @classmethod
    def read(
        cls,
        path: Path,
        *,
        require_frame_chunks: bool = True,
    ) -> "_LiveArrayChunks":
        metadata_path = path / "zarr.json"
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("node_type") != "array" or metadata.get("zarr_format") != 3:
            raise ValueError(f"Live processing requires a Zarr v3 array: {path}")
        shape = tuple(int(value) for value in metadata.get("shape", ()))
        names = tuple(
            str(value).lower() for value in metadata.get("dimension_names", ())
        )
        if shape == () or names != ("t", "c", "z", "y", "x"):
            raise ValueError(
                "Live processing requires per-position TCZYX arrays; "
                f"got dimensions {names} at {path}"
            )
        chunk_shape = tuple(
            int(value)
            for value in metadata.get("chunk_grid", {})
            .get("configuration", {})
            .get("chunk_shape", ())
        )
        if len(chunk_shape) != len(shape):
            raise ValueError(f"Invalid chunk grid metadata: {path}")
        if require_frame_chunks and chunk_shape[:3] != (1, 1, 1):
            raise ValueError(
                "Live processing requires one-frame chunks along T, C, and Z; "
                f"got {chunk_shape} at {path}"
            )
        if any(
            codec.get("name") == "sharding_indexed"
            for codec in metadata.get("codecs", ())
            if isinstance(codec, dict)
        ):
            raise ValueError(f"Live processing does not support sharded arrays: {path}")
        encoding_document = metadata.get("chunk_key_encoding", {})
        encoding = str(encoding_document.get("name", "default"))
        if encoding not in {"default", "v2"}:
            raise ValueError(f"Unsupported chunk-key encoding {encoding!r}: {path}")
        default_separator = "/" if encoding == "default" else "."
        separator = str(
            encoding_document.get("configuration", {}).get(
                "separator", default_separator
            )
        )
        if separator not in {"/", "."}:
            raise ValueError(f"Unsupported chunk-key separator {separator!r}: {path}")
        return cls(path, shape, chunk_shape, encoding, separator)

    def tile_is_ready(self, time_index: int) -> bool:
        chunk_counts = tuple(
            math.ceil(size / chunk)
            for size, chunk in zip(self.shape, self.chunk_shape, strict=True)
        )
        coordinate_ranges = (
            (time_index,),
            range(chunk_counts[1]),
            range(chunk_counts[2]),
            range(chunk_counts[3]),
            range(chunk_counts[4]),
        )
        return all(
            self._chunk_path(coordinates).is_file()
            for coordinates in product(*coordinate_ranges)
        )

    def _chunk_path(self, coordinates: tuple[int, ...]) -> Path:
        components = [str(value) for value in coordinates]
        if self.encoding == "default":
            components.insert(0, "c")
        if self.separator == "/":
            return self.path.joinpath(*components)
        return self.path / self.separator.join(components)


def iter_live_tiles(
    readiness: ZarrTileReadiness,
    manifest: LiveManifest,
    log_path: str | Path,
    *,
    poll_interval: float = LIVE_POLL_INTERVAL_SECONDS,
    sleeper: Callable[[float], None] = time.sleep,
    completed_tiles: set[tuple[int, int]] | None = None,
) -> Iterator[tuple[int, int]]:
    """Yield newly complete, unprocessed tiles until acquisition termination."""
    expected = {
        (time_index, position_index)
        for time_index in range(manifest.index_sizes["t"])
        for position_index in range(manifest.index_sizes["p"])
    }
    yielded = set(completed_tiles or ())
    unexpected = yielded - expected
    if unexpected:
        raise ValueError(
            f"Completed live tiles are outside the manifest plan: {sorted(unexpected)}"
        )
    while True:
        available = sorted(readiness.ready_tiles() - yielded)
        for tile in available:
            yielded.add(tile)
            yield tile

        event = read_lifecycle_event(log_path, manifest.acquisition_id)
        if event == "completed" and yielded == expected:
            return
        if event in {"canceled", "errored"}:
            raise RuntimeError(
                f"Live acquisition {event} after {len(yielded)} of "
                f"{len(expected)} complete tiles"
            )
        print(
            "Live processing is caught up; waiting "
            f"{poll_interval:g} seconds for another complete tile..."
        )
        sleeper(poll_interval)


def _required_float(document: dict[str, Any], key: str) -> float:
    value = _optional_float(document.get(key))
    if value is None:
        raise ValueError(f"Live manifest lacks numeric {key}")
    return value


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _coordinate_triplet(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Live manifest {label} must contain three values")
    return tuple(float(item) for item in value)  # type: ignore[return-value]
