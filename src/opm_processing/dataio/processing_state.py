"""Durable processing state stored beside, never inside, image stores.

The state document is intentionally strict.  It describes only outputs made by
the current processing contract; older journals and custom Zarr attributes are
not imported.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


PROCESSING_STATE_SCHEMA = "opm-processing-state"
PROCESSING_STATE_VERSION = 1


def processing_state_path(output_root: Path, acquisition_stem: str) -> Path:
    """Return the single processing-state path for an acquisition output root."""
    root = Path(output_root).expanduser().resolve()
    return root / f"{acquisition_stem}.processing.json"


def _json_value(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return [_json_value(item) for item in sorted(value)]
    if hasattr(value, "item"):
        return value.item()
    return value


def _tile_records(values: Iterable[tuple[int, ...]]) -> list[list[int]]:
    """Normalize index tuples for deterministic persistence."""
    return [list(map(int, value)) for value in sorted(set(values))]


def _configuration_fingerprint(configuration: dict[str, Any]) -> str:
    """Hash only output-affecting settings for exact resume validation."""
    encoded = json.dumps(
        _json_value(configuration),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class ProcessingState:
    """Mutable, atomically persisted processing state for one acquisition."""

    path: Path
    document: dict[str, Any]

    @classmethod
    def create(cls, path: Path, source_path: Path) -> "ProcessingState":
        """Create a new empty state document and persist it immediately."""
        state_path = Path(path).expanduser().resolve()
        source = Path(source_path).expanduser().resolve()
        state = cls(
            path=state_path,
            document={
                "schema": PROCESSING_STATE_SCHEMA,
                "schema_version": PROCESSING_STATE_VERSION,
                "source": {"path": str(source)},
                "outputs": {},
                "registration": {},
            },
        )
        state.save()
        return state

    @classmethod
    def read(cls, path: Path) -> "ProcessingState":
        """Read and strictly validate a current processing-state document."""
        state_path = Path(path).expanduser().resolve()
        if not state_path.is_file():
            raise FileNotFoundError(f"Processing state does not exist: {state_path}")
        try:
            document = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid processing-state JSON: {state_path}") from error
        if not isinstance(document, dict):
            raise ValueError(f"Processing state must be a JSON object: {state_path}")
        if document.get("schema") != PROCESSING_STATE_SCHEMA:
            raise ValueError(f"Unsupported processing-state schema: {state_path}")
        if document.get("schema_version") != PROCESSING_STATE_VERSION:
            raise ValueError(f"Unsupported processing-state version: {state_path}")
        if not isinstance(document.get("source"), dict):
            raise ValueError(f"Processing state lacks source identity: {state_path}")
        for section in ("outputs", "registration"):
            if not isinstance(document.get(section), dict):
                raise ValueError(
                    f"Processing-state section {section!r} must be an object: "
                    f"{state_path}"
                )
        return cls(path=state_path, document=document)

    @classmethod
    def open(
        cls, path: Path, source_path: Path, *, overwrite: bool
    ) -> "ProcessingState":
        """Create or reopen state while enforcing its acquisition identity."""
        state_path = Path(path).expanduser().resolve()
        source = Path(source_path).expanduser().resolve()
        if not state_path.exists():
            return cls.create(state_path, source)
        state = cls.read(state_path)
        recorded_source = Path(str(state.document["source"].get("path", ""))).resolve()
        if recorded_source != source:
            if overwrite:
                return cls.create(state_path, source)
            raise ValueError(
                "Processing state belongs to a different acquisition: "
                f"{recorded_source} != {source}"
            )
        return state

    def _run_key(self, output_path: Path) -> str:
        """Return a root-relative POSIX key for one output artifact."""
        output = Path(output_path).expanduser().resolve()
        try:
            return output.relative_to(self.path.parent).as_posix()
        except ValueError as error:
            raise ValueError(
                f"Processing output must be inside {self.path.parent}: {output}"
            ) from error

    def initialize_run(
        self,
        output_path: Path,
        *,
        configuration: dict[str, Any],
        roi_series: Iterable[dict[str, Any]] = (),
        overwrite: bool,
    ) -> None:
        """Create a run or validate that a resumable run is exactly compatible."""
        key = self._run_key(output_path)
        expected = {
            "path": key,
            "configuration_sha256": _configuration_fingerprint(configuration),
        }
        series = [_json_value(item) for item in roi_series]
        outputs = self.document["outputs"]
        existing = outputs.get(key)
        if existing is not None and not overwrite:
            comparable = {name: existing.get(name) for name in expected}
            if comparable != expected or existing.get("roi_series", []) != series:
                raise ValueError(
                    f"Existing processing run is incompatible with {output_path}"
                )
            return
        record = {
            **expected,
            "completed_tiles": [],
            "zero_channels": [],
        }
        if series:
            record["roi_series"] = series
        outputs[key] = record
        self.document["registration"].pop(key, None)
        self.save()

    def run(self, output_path: Path) -> dict[str, Any]:
        """Return a required run record."""
        key = self._run_key(output_path)
        try:
            run = self.document["outputs"][key]
        except KeyError as error:
            raise ValueError(
                f"Processing state has no run for {output_path}"
            ) from error
        if not isinstance(run, dict):
            raise ValueError(f"Invalid processing run for {output_path}")
        return run

    def completed_tiles(self, output_path: Path) -> set[tuple[int, int]]:
        """Return durable completed time/position pairs."""
        return {
            tuple(map(int, item))
            for item in self.run(output_path).get("completed_tiles", ())
        }

    def zero_channels(self, output_path: Path) -> set[tuple[int, int, int]]:
        """Return durable empty time/position/channel decisions."""
        return {
            tuple(map(int, item))
            for item in self.run(output_path).get("zero_channels", ())
        }

    def completed_channels(self, output_path: Path) -> set[tuple[int, int, int]]:
        """Return durable T/P/C checkpoints; older runs have only tile records."""
        return {
            tuple(map(int, item))
            for item in self.run(output_path).get("completed_channels", ())
        }

    def complete_channel(
        self,
        output_path: Path,
        time_index: int,
        position_index: int,
        channel_index: int,
        *,
        is_zero: bool = False,
    ) -> None:
        """Atomically checkpoint a channel after its output write completes."""
        run = self.run(output_path)
        key = (int(time_index), int(position_index), int(channel_index))
        completed = self.completed_channels(output_path)
        completed.add(key)
        zero = self.zero_channels(output_path)
        if is_zero:
            zero.add(key)
        else:
            zero.discard(key)
        run["completed_channels"] = _tile_records(completed)
        run["zero_channels"] = _tile_records(zero)
        self.save()

    def roi_series(self, output_path: Path) -> tuple[dict[str, Any], ...]:
        """Return the required source and crop mapping for variable ROI series."""
        records = self.run(output_path).get("roi_series")
        if records is None:
            return ()
        if not isinstance(records, list) or any(
            not isinstance(item, dict) for item in records
        ):
            raise ValueError(f"Invalid ROI series mapping for {output_path}")
        return tuple(records)

    def complete_tile(
        self,
        output_path: Path,
        time_index: int,
        position_index: int,
        *,
        zero_channels: Iterable[int] = (),
    ) -> None:
        """Atomically record one tile after all of its output writes complete."""
        run = self.run(output_path)
        completed = {tuple(map(int, item)) for item in run.get("completed_tiles", ())}
        completed.add((int(time_index), int(position_index)))
        zero = {tuple(map(int, item)) for item in run.get("zero_channels", ())}
        zero.update(
            (int(time_index), int(position_index), int(channel_index))
            for channel_index in zero_channels
        )
        run["completed_tiles"] = _tile_records(completed)
        run["zero_channels"] = _tile_records(zero)
        self.save()

    def save_registration(
        self,
        output_path: Path,
        *,
        configuration: dict[str, Any],
        pairwise_metrics: dict[str, Any],
    ) -> None:
        """Persist pairwise registration links for one processed output."""
        key = self._run_key(output_path)
        self.run(output_path)
        self.document["registration"][key] = {
            "configuration_sha256": _configuration_fingerprint(configuration),
            "pairwise_metrics": _json_value(pairwise_metrics),
        }
        self.save()

    def registration(
        self,
        output_path: Path,
        *,
        configuration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return registration state, optionally validating its settings."""
        key = self._run_key(output_path)
        record = self.document["registration"].get(key)
        if not isinstance(record, dict):
            raise ValueError(f"Processing state has no registration for {output_path}")
        if configuration is not None and record.get(
            "configuration_sha256"
        ) != _configuration_fingerprint(configuration):
            raise ValueError("Saved registration settings do not match this run")
        return record

    def complete_registration(
        self,
        output_path: Path,
        *,
        fused_path: Path,
        tiles: Iterable[dict[str, Any]],
    ) -> None:
        """Record final registered tile origins and their fused artifact."""
        record = self.registration(output_path)
        record["fused_path"] = self._run_key(fused_path)
        record["tiles"] = [_json_value(tile) for tile in tiles]
        self.save()

    def set_registered_max_projection(
        self,
        output_path: Path,
        *,
        max_projection_path: Path,
    ) -> None:
        """Record the max projection derived from a registered fused artifact."""
        record = self.registration(output_path)
        record["max_projection_path"] = self._run_key(max_projection_path)
        self.save()

    def registered_output_for_fused(self, fused_path: Path) -> Path:
        """Return the processed output associated with a registered fusion."""
        fused_key = self._run_key(fused_path)
        matches = [
            key
            for key, record in self.document["registration"].items()
            if isinstance(record, dict) and record.get("fused_path") == fused_key
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one completed registration for {fused_path}, "
                f"found {len(matches)}"
            )
        return self.path.parent / matches[0]

    def registered_output_for_max_projection(self, path: Path) -> Path:
        """Return the processed output associated with a registered max-Z image."""
        projection_key = self._run_key(path)
        matches = [
            key
            for key, record in self.document["registration"].items()
            if isinstance(record, dict)
            and record.get("max_projection_path") == projection_key
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one registered maximum projection for {path}, "
                f"found {len(matches)}"
            )
        return self.path.parent / matches[0]

    def save(self) -> None:
        """Durably replace the state document with a canonical JSON snapshot."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    _json_value(self.document),
                    stream,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            for attempt in range(5):
                try:
                    temporary.replace(self.path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.02 * (attempt + 1))
        finally:
            temporary.unlink(missing_ok=True)
