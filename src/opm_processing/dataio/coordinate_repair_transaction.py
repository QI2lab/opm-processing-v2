"""Apply reviewed metadata repairs across explicitly named acquisition roots."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opm_processing.dataio import coordinate_repair as base

MULTI_REPAIR_SCHEMA = "opm-stage-coordinate-repair-v2"


def _roots(plan: dict[str, Any]) -> dict[str, Path]:
    """Validate canonical, distinct roots before resolving any plan record."""
    declared = plan.get("roots")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("Repair plan lacks explicit acquisition roots")
    roots = {}
    for identifier, value in declared.items():
        if not isinstance(identifier, str) or not isinstance(value, str):
            raise ValueError("Invalid repair root identity")
        path = Path(value)
        if not path.is_absolute() or path.resolve() != path or not path.is_dir():
            raise ValueError("Repair root is not an existing canonical directory")
        if any(
            path.is_relative_to(other) or other.is_relative_to(path)
            for other in roots.values()
        ):
            raise ValueError(
                "Repair roots must be distinct, non-overlapping directories"
            )
        roots[identifier] = path
    if roots.get("root_0") != Path(plan.get("root", "")):
        raise ValueError("Primary repair root does not match its declared identity")
    return roots


def _path(roots: dict[str, Path], record: dict[str, Any]) -> Path:
    """Keep a record within the specific root named by the reviewed plan."""
    identifier = record.get("root_id")
    if identifier not in roots:
        raise ValueError(f"Unknown repair root: {identifier}")
    relative = record.get("path")
    if not isinstance(relative, str):
        raise ValueError("Repair path must be a relative string")
    return base._inside(roots[identifier], relative)


def _dependencies(plan: dict[str, Any], roots: dict[str, Path]) -> None:
    """Reject changed geometry or source metadata before replacing any target."""
    dependencies = plan.get("read_dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("Repair plan lacks validated input dependencies")
    for dependency in dependencies:
        with _path(roots, dependency).open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        if fingerprint != dependency["sha256"]:
            raise ValueError(
                "Repair input dependency changed; no files changed. Generate a fresh plan."
            )


def apply_split_stage_coordinate_repair(plan: dict[str, Any]) -> dict[str, Any]:
    """Back up both volumes, then atomically replace only approved metadata.

    A failure restores already replaced files across every root. All backups
    and their root-specific manifests are durable and verified before the
    first metadata replacement.
    """
    if plan.get("schema") != MULTI_REPAIR_SCHEMA or plan.get(
        "plan_sha256"
    ) != base._plan_digest(plan):
        raise ValueError("Repair plan fingerprint is invalid")
    roots = _roots(plan)
    records = plan.get("files")
    if not isinstance(records, list):
        raise ValueError("Repair plan lacks metadata records")
    if not records:
        return {"status": "already_repaired", "changed_files": 0}
    paths = []
    originals = []
    for record in records:
        path = _path(roots, record)
        if not base._metadata_allowed(record["path"], plan["stem"]):
            raise ValueError("Repair plan includes a non-metadata path")
        after = record["after_text"].encode("utf-8")
        if base._sha(after) != record["after_sha256"]:
            raise ValueError("Repair postimage fingerprint is invalid")
        if path in paths:
            raise ValueError("Repair contains a duplicate metadata target")
        original = path.read_bytes()
        if path.name == "zarr.json" and any(
            json.loads(value).get("node_type") != "group" for value in (original, after)
        ):
            raise ValueError("Repair cannot replace array metadata")
        paths.append(path)
        originals.append(original)
    fingerprints = [base._sha(value) for value in originals]
    if all(
        value == record["after_sha256"]
        for value, record in zip(fingerprints, records, strict=True)
    ):
        return {"status": "already_repaired", "changed_files": 0}
    if any(
        value != record["before_sha256"]
        for value, record in zip(fingerprints, records, strict=True)
    ):
        raise ValueError(
            "Repair preimage mismatch; no files changed. Generate a fresh plan."
        )
    _dependencies(plan, roots)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    relative_backup = f".coordinate-repair-backups/{stamp}_{plan['plan_sha256'][:12]}"
    backups = {}
    for identifier in sorted({record["root_id"] for record in records}):
        backup = base._inside(roots[identifier], relative_backup)
        backup.mkdir(parents=True, exist_ok=False)
        backups[identifier] = backup
    for record, original in zip(records, originals, strict=True):
        target = base._inside(backups[record["root_id"]], record["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        if base._sha(target.read_bytes()) != record["before_sha256"]:
            raise RuntimeError("Repair backup verification failed; no metadata changed")
    for identifier, backup in backups.items():
        manifest = {
            "schema": MULTI_REPAIR_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "root_id": identifier,
            "root": str(roots[identifier]),
            "roots": {key: str(value) for key, value in roots.items()},
            "files": [
                {
                    key: record[key]
                    for key in ("root_id", "path", "before_sha256", "after_sha256")
                }
                for record in records
                if record["root_id"] == identifier
            ],
        }
        content = base._json_bytes(manifest)
        path = backup / "manifest.json"
        with path.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if path.read_bytes() != content:
            raise RuntimeError("Repair backup manifest verification failed")

    # Backups may take time on another volume. Check every input once more
    # before making the first change to either acquisition directory.
    _dependencies(plan, roots)
    if any(
        base._sha(path.read_bytes()) != record["before_sha256"]
        for path, record in zip(paths, records, strict=True)
    ):
        raise ValueError("Metadata changed during backup; no metadata changed")
    completed = []
    try:
        for index, (path, record) in enumerate(zip(paths, records, strict=True)):
            if base._sha(path.read_bytes()) != record["before_sha256"]:
                raise ValueError("Metadata changed while applying the reviewed plan")
            completed.append(index)
            base._atomic_write(path, record["after_text"].encode("utf-8"))
        if any(
            base._sha(path.read_bytes()) != record["after_sha256"]
            for path, record in zip(paths, records, strict=True)
        ):
            raise RuntimeError("Repair postimage verification failed")
    except BaseException as error:
        conflicts = []
        for index in reversed(completed):
            try:
                current = base._sha(paths[index].read_bytes())
                if current == records[index]["after_sha256"]:
                    base._atomic_write(paths[index], originals[index])
                elif current != records[index]["before_sha256"]:
                    conflicts.append(str(paths[index]))
            except BaseException as recovery_error:
                conflicts.append(f"{paths[index]}: {recovery_error!r}")
        if conflicts:
            raise RuntimeError(
                f"Concurrent metadata changes prevented complete rollback; backups: {backups}; conflicts: {conflicts}"
            ) from error
        raise
    return {
        "status": "applied",
        "changed_files": len(records),
        "backup_directories": {key: str(value) for key, value in backups.items()},
        "plan_sha256": plan["plan_sha256"],
    }
