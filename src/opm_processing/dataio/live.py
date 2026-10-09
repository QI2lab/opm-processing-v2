"""Live-acquisition sidecars and raw OME-Zarr tile readiness."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

from opm_processing.dataio.acquisition import (
    AcquisitionMetadata,
    ChannelMetadata,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

LIVE_MANIFEST_SCHEMA = "opm_v2.live_acquisition"
LIVE_MANIFEST_VERSION = "1.0"
LIVE_POLL_INTERVAL_SECONDS = 30.0
_TERMINAL_EVENTS = {"completed", "canceled", "errored"}


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
    def read(cls, path: str | Path) -> LiveManifest:
        """Read the live controller's acquisition plan for the supported schema.

        Parameters
        ----------
        path : str | Path
            Live manifest JSON beside the acquisition store.

        Returns
        -------
        LiveManifest
            Planned dimensions, channels, calibration, and stage positions.
        """
        manifest_path = Path(path).expanduser().resolve()
        with manifest_path.open(encoding="utf-8") as stream:
            document = json.load(stream)
        if document.get("schema") != LIVE_MANIFEST_SCHEMA:
            raise ValueError(
                f"Unsupported live manifest schema: {document.get('schema')!r}"
            )
        if str(document.get("schema_version")) != LIVE_MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported live manifest version: {document.get('schema_version')!r}"
            )

        acquisition_id = document["acquisition_id"]
        raw_data_path = document["data_path"]
        data_path = Path(raw_data_path).expanduser()
        if not data_path.is_absolute():
            data_path = manifest_path.parent / data_path
        data_path = data_path.resolve()

        sizes_document = document["index_sizes"]
        index_sizes = {
            axis: int(sizes_document.get(axis, 1)) for axis in ("t", "p", "c", "z")
        }

        acquisition_order = tuple(str(axis) for axis in document["acquisition_order"])

        channel_documents = document["channels"]
        channels = []
        for index, channel in enumerate(channel_documents):
            name = channel["name"]
            channels.append(
                ChannelMetadata(
                    index=index,
                    name=name,
                    wavelength_nm=channel.get("wavelength_nm"),
                    exposure_ms=channel.get("exposure_ms"),
                    laser_power=channel.get("laser_power"),
                )
            )

        position_documents = document["stage_positions_zxy"]
        stage_positions = tuple(
            tuple(float(value) for value in position) for position in position_documents
        )

        orientations_document = document.get("orientations", {})

        return cls(
            path=manifest_path,
            acquisition_id=acquisition_id,
            data_path=data_path,
            mode=str(document.get("mode", "")).strip(),
            index_sizes=index_sizes,
            acquisition_order=acquisition_order,
            channels=tuple(channels),
            stage_positions_zxy=stage_positions,
            scan_axis=document.get("scan_axis"),
            scan_axis_step_um=float(document["scan_axis_step_um"]),
            pixel_size_um=float(document["pixel_size_um"]),
            angle_deg=float(document["angle_deg"]),
            camera_offset=float(document["camera_offset"]),
            camera_conversion=float(document["camera_e_to_adu"]),
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
        """Overlay upfront manifest values onto inspected OME-Zarr metadata.

        Parameters
        ----------
        acquisition : AcquisitionMetadata
            Inspected acquisition dimensions, stage geometry, and camera calibration.

        Returns
        -------
        AcquisitionMetadata
            Acquisition description with the live manifest plan and calibration applied.
        """
        if acquisition.path.resolve() != self.data_path:
            raise ValueError(
                "Live manifest data_path does not match the requested acquisition: "
                f"{self.data_path} != {acquisition.path.resolve()}"
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


def read_lifecycle_event(path: str | Path, acquisition_id: str) -> str | None:
    """Return the most recent recognized acquisition lifecycle event.

    A final unterminated line is ignored because it may be observed while the
    controller is appending it.

    Parameters
    ----------
    path : str | Path
        Controller lifecycle JSONL file beside the acquisition.
    acquisition_id : str
        Manifest identity that must match lifecycle log records.

    Returns
    -------
    str | None
        Last complete recognized lifecycle event, or None before any event is logged.
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
        record = json.loads(stripped)
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
        """Read per-position chunk layouts from the acquisition's array paths.

        Parameters
        ----------
        acquisition : AcquisitionMetadata
            Inspected filesystem-backed TCZYX acquisition series.
        """
        self.acquisition = acquisition
        self._arrays = tuple(
            LiveArrayChunks.read(acquisition.path / relative_path)
            for relative_path in acquisition.array_paths
        )

    def tile_is_ready(self, time_index: int, position_index: int) -> bool:
        """Return whether every raw chunk for one T/P tile exists.

        Parameters
        ----------
        time_index : int
            Acquisition timepoint index.
        position_index : int
            Acquisition position index.

        Returns
        -------
        bool
            True when every required raw chunk file for the selected tile exists.
        """
        if not 0 <= time_index < self.acquisition.index_sizes["t"]:
            raise IndexError(f"Time index is out of bounds: {time_index}")
        if not 0 <= position_index < self.acquisition.index_sizes["p"]:
            raise IndexError(f"Position index is out of bounds: {position_index}")
        return self._arrays[position_index].tile_is_ready(time_index)

    def ready_tiles(self) -> set[tuple[int, int]]:
        """Return every tile whose complete raw chunk set is present.

        Returns
        -------
        set[tuple[int, int]]
            Timepoint/position pairs whose complete raw chunk sets are present.
        """
        return {
            (time_index, position_index)
            for time_index in range(self.acquisition.index_sizes["t"])
            for position_index in range(self.acquisition.index_sizes["p"])
            if self.tile_is_ready(time_index, position_index)
        }


@dataclass(frozen=True)
class LiveArrayChunks:
    """Array dimensions and chunk-key layout used to detect completed tiles."""

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
    ) -> LiveArrayChunks:
        """Read a position array's dimensions and raw chunk-key layout.

        Parameters
        ----------
        path : Path
            Position array directory containing zarr.json.
        require_frame_chunks : bool
            Require separate T/C/Z frame chunks for live raw-data reads.

        Returns
        -------
        LiveArrayChunks
            Layout used to enumerate every chunk needed for a complete tile.
        """
        metadata_path = path / "zarr.json"
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        shape = tuple(int(value) for value in metadata["shape"])
        chunk_shape = tuple(
            int(value)
            for value in metadata["chunk_grid"]["configuration"]["chunk_shape"]
        )
        if require_frame_chunks and chunk_shape[:3] != (1, 1, 1):
            raise ValueError(
                "Live processing requires one-frame chunks along T, C, and Z; "
                f"got {chunk_shape} at {path}"
            )
        if any(
            codec.get("name") == "sharding_indexed"
            for codec in metadata.get("codecs", ())
        ):
            raise ValueError(f"Live processing does not support sharded arrays: {path}")
        encoding_document = metadata.get("chunk_key_encoding", {})
        encoding = str(encoding_document.get("name", "default"))
        default_separator = "/" if encoding == "default" else "."
        separator = str(
            encoding_document.get("configuration", {}).get(
                "separator", default_separator
            )
        )
        return cls(path, shape, chunk_shape, encoding, separator)

    def tile_is_ready(self, time_index: int) -> bool:
        """Return whether all channel, plane, and detector chunks are present.

        Parameters
        ----------
        time_index : int
            Timepoint whose chunk keys are checked.

        Returns
        -------
        bool
            True once every required chunk file exists.
        """
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
        """Encode one TCZYX chunk coordinate as its on-disk path.

        Parameters
        ----------
        coordinates : tuple[int, ...]
            Chunk-grid indices in TCZYX order.

        Returns
        -------
        Path
            File path for the array's Zarr chunk-key encoding.
        """
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
    """Yield newly complete, unprocessed tiles until acquisition termination.

    Parameters
    ----------
    readiness : ZarrTileReadiness
        Chunk-readiness reader for the acquisition position arrays.
    manifest : LiveManifest
        Upfront live acquisition plan defining the expected timepoints and positions.
    log_path : str | Path
        Controller lifecycle JSONL file beside the acquisition store.
    poll_interval : float
        Seconds to wait between checks while acquisition is still running.
    sleeper : Callable[[float], None]
        Wait function, replaceable by a test clock.
    completed_tiles : set[tuple[int, int]] | None
        Durable timepoint/position checkpoints already processed.

    Yields
    ------
    tuple[int, int]
        Newly complete timepoint/position pair not present in the durable checkpoints.
    """
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
