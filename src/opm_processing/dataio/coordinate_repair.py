"""Plan and apply a metadata-only repair of uniform legacy stage coordinates.

The repair is limited to positively scanned, uncropped, uniform geometry.
Explicit additional roots support acquisitions whose raw, processed, and fused
stores span volumes. It never opens an image array or writes array metadata,
chunks, TIFFs, or raw acquisition files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from ome_types import from_xml, to_xml
from ome_types.model import UnitsLength

from opm_processing.dataio.acquisition import (
    acquisition_stem,
    inspect_acquisition,
    resolve_acquisition_path,
)
from opm_processing.dataio.ngff import round_spatial_values
from opm_processing.imageprocessing.coordinates import (
    STAGE_COORDINATE_VERSION,
    deskewed_stage_scan_offsets,
)
from opm_processing.imageprocessing.opmtools import deskew_shape_estimator

REPAIR_SCHEMA = "opm-stage-coordinate-repair-v1"
_COLLECTION_SUFFIXES = (
    "_deskewed.ome.zarr",
    "_decon_deskewed.ome.zarr",
    "_max_z_deskewed.ome.zarr",
    "_max_z_decon_deskewed.ome.zarr",
)


def _sha(data: bytes) -> str:
    """Return a byte-exact content fingerprint."""
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    """Encode metadata deterministically without changing numerical data."""
    return (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def _plan_digest(plan: dict[str, Any]) -> str:
    """Fingerprint the complete reviewed plan, including every postimage."""
    value = {key: item for key, item in plan.items() if key != "plan_sha256"}
    return _sha(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    )


def _inside(root: Path, relative: str) -> Path:
    """Resolve an untrusted relative path without allowing workspace escape."""
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError(f"Unsafe repair path: {relative}")
    target = (root / part).resolve()
    if not target.is_relative_to(root) or target != root / part:
        raise ValueError(f"Repair path escapes its acquisition root: {relative}")
    return target


def _metadata_allowed(relative: str, stem: str) -> bool:
    """Permit only the exact journal and image-group metadata files."""
    parts = Path(relative).parts
    if parts == (f"{stem}.processing.json",):
        return True
    collections = {f"{stem}{suffix}" for suffix in _COLLECTION_SUFFIXES}
    images = {f"{stem}_fused.ome.zarr", f"{stem}_max_z_fused.ome.zarr"}
    if not parts:
        return False
    if parts[0] in images:
        return parts[1:] == ("zarr.json",)
    if parts[0] not in collections:
        return False
    return (
        parts[1:] == ("zarr.json",)
        or parts[1:] == ("OME", "METADATA.ome.xml")
        or (len(parts) == 3 and parts[1].isdigit() and parts[2] == "zarr.json")
    )


def _read_group(path: Path) -> dict[str, Any]:
    """Read group metadata and refuse image-array metadata."""
    value = json.loads(path.read_bytes())
    if value.get("node_type") != "group":
        raise ValueError(f"Expected group metadata, not an array: {path}")
    return value


def _datasets(group: dict[str, Any]) -> list[dict[str, Any]]:
    """Require an explicit micrometer TCZYX coordinate grid."""
    multiscales = group["attributes"]["ome"]["multiscales"]
    if len(multiscales) != 1:
        raise ValueError("Repair requires one multiscale image per group")
    axes = multiscales[0]["axes"]
    if [axis["name"] for axis in axes] != list("tczyx") or any(
        axis.get("unit") != "micrometer" for axis in axes[-3:]
    ):
        raise ValueError("Repair requires TCZYX image coordinates in micrometers")
    return multiscales[0]["datasets"]


def _transform(dataset: dict[str, Any], kind: str) -> list[float]:
    """Require one explicit scale or translation transform."""
    matches = [
        t[kind] for t in dataset["coordinateTransformations"] if t["type"] == kind
    ]
    if len(matches) != 1 or len(matches[0]) != 5:
        raise ValueError(f"Expected one explicit TCZYX {kind}")
    return matches[0]


def _shift_datasets(group: dict[str, Any], delta: np.ndarray) -> None:
    """Translate every level while retaining its existing level-specific origin."""
    for dataset in _datasets(group):
        translation = _transform(dataset, "translation")
        translation[-3:] = round_spatial_values(np.asarray(translation[-3:]) + delta)


def _physical_plan(root: Path, source: Path, positions: np.ndarray) -> dict[str, Any]:
    """Attach authoritative commanded physical positions when saved plans match."""
    directory = root / f"{source.name.removesuffix('.zarr')}_ao_grid_results"
    records: dict[tuple[float, ...], tuple[float, ...]] = {}
    files = sorted(
        directory.glob("time_*_z_*_ao_grid_results/exp_stage_positions.json")
    )
    for path in files:
        for point in json.loads(path.read_bytes()):
            if "lab_scan_um" not in point:
                continue
            key = round_spatial_values((point["z"], point["lab_scan_um"], point["y"]))
            physical = round_spatial_values((point["z"], point["x"], point["y"]))
            if key in records and records[key] != physical:
                raise ValueError(
                    "Saved physical stage plans have conflicting references"
                )
            records[key] = physical
    if not records:
        return {}
    keys = [round_spatial_values(position) for position in positions]
    if any(key not in records for key in keys):
        raise ValueError(
            "Saved physical plans do not identify every recorded source position"
        )
    return {
        "planned_physical_stage_positions_zxy_um": [list(records[key]) for key in keys],
        "physical_stage_plan_source": [str(path.relative_to(root)) for path in files],
        "physical_stage_plan_kind": "saved-commanded-position-plan; not per-frame readback",
    }


def plan_stage_coordinate_repair(root: str | Path) -> dict[str, Any]:
    """Build a reviewable metadata-only repair plan without writing any file.

    Parameters
    ----------
    root : str or pathlib.Path
        Acquisition directory containing the raw and four processed collections.

    Returns
    -------
    dict
        Validated whitelist, exact pre/postimage hashes, postimages, and plan hash.
        An already repaired acquisition returns an empty change list.
    """
    directory = Path(root).expanduser().resolve()
    source = resolve_acquisition_path(directory)
    source_dependencies = [
        path
        for path in (
            source / "zarr.json",
            source / "OME" / "METADATA.ome.xml",
            *source.glob("*/zarr.json"),
            *source.glob("*/*/zarr.json"),
        )
        if path.is_file()
    ]
    dependency_hashes = {
        str(path.relative_to(directory)): _sha(path.read_bytes())
        for path in source_dependencies
    }
    acquisition = inspect_acquisition(source)
    stem = acquisition_stem(source)
    if source.parent != directory:
        raise ValueError(
            "Raw source must be directly inside the selected acquisition root"
        )
    if acquisition.mode != "stage" or acquisition.is_2d:
        raise ValueError("Repair requires a three-dimensional stage acquisition")
    if not acquisition.scan_axis_reversed or any(acquisition.stage_axis_flips_xyz):
        raise ValueError("Repair supports positive, unflipped stage trajectories only")
    if any(
        (
            acquisition.excess_scan_positions,
            acquisition.excess_scan_start_positions,
            acquisition.excess_scan_end_positions,
        )
    ):
        raise ValueError("Repair has not been validated for excess-frame cropping")
    positions = np.asarray(acquisition.stage_positions_zxy, dtype=float)
    scan_offsets = deskewed_stage_scan_offsets(acquisition)
    if len(positions) == 0 or not np.allclose(
        scan_offsets, scan_offsets[0], rtol=0, atol=1e-9
    ):
        raise ValueError("Per-tile coordinate shifts are not one common translation")
    stores = [directory / f"{stem}{suffix}" for suffix in _COLLECTION_SUFFIXES]
    fused_stores = [
        directory / f"{stem}_fused.ome.zarr",
        directory / f"{stem}_max_z_fused.ome.zarr",
    ]
    groups = [_read_group(path / "zarr.json") for path in (*stores, *fused_stores)]
    versions = [
        group.get("attributes", {}).get("opm_processing_coordinates", {}).get("version")
        for group in groups
    ]
    base_plan: dict[str, Any] = {
        "schema": REPAIR_SCHEMA,
        "root": str(directory),
        "stem": stem,
        "source": str(source.relative_to(directory)),
        "coordinate_version": STAGE_COORDINATE_VERSION,
        "files": [],
        "stale_exports": [
            str(path.relative_to(directory))
            for path in (
                *directory.glob("*_fused.ome.tif"),
                directory / "ao_overview",
                directory / "roi_exports",
            )
            if path.exists()
        ],
    }
    if all(version == STAGE_COORDINATE_VERSION for version in versions):
        base_plan["status"] = "already_repaired"
        base_plan["plan_sha256"] = _plan_digest(base_plan)
        return base_plan
    if any(version is not None for version in versions):
        raise ValueError(
            "Mixed or unsupported coordinate versions require manual review"
        )

    changes: dict[str, dict[str, str]] = {}

    def add(path: Path, after: bytes) -> None:
        relative = str(path.relative_to(directory))
        if not _metadata_allowed(relative, stem):
            raise ValueError(
                f"Repair attempted an unapproved metadata path: {relative}"
            )
        before = path.read_bytes()
        if before != after:
            changes[relative] = {
                "path": relative,
                "before_sha256": _sha(before),
                "after_sha256": _sha(after),
                "after_text": after.decode("utf-8"),
            }

    first = _datasets(_read_group(stores[0] / "0" / "zarr.json"))[0]
    spacing = np.asarray(_transform(first, "scale")[-3:])
    pixel = float(acquisition.pixel_size_um)
    averaging = round(float(spacing[0]) / pixel)
    if averaging < 1 or not np.allclose(
        spacing, (pixel * averaging, pixel, pixel), rtol=0, atol=1e-9
    ):
        raise ValueError(
            "Processed spacing is inconsistent with the recorded camera calibration"
        )
    expected_shape, _, _, _ = deskew_shape_estimator(
        acquisition.shape[-3:],
        theta=float(acquisition.angle_deg),
        distance=float(acquisition.scan_axis_step_um),
        pixel_size=pixel,
        crop_after_deskew=False,
        downsample_factor=averaging,
    )
    delta = np.asarray(
        round_spatial_values(scan_offsets[0] + (0.5 * (averaging - 1) * pixel, 0, 0))
    )
    coordinate_metadata = {
        "version": STAGE_COORDINATE_VERSION,
        "stage_axes": ["z", "x", "y"],
        "stage_positions_zxy_um": [list(round_spatial_values(p)) for p in positions],
        "placement_stage_positions_zxy_um": [
            list(round_spatial_values(p)) for p in positions
        ],
        "stage_axis_flips_xyz": list(acquisition.stage_axis_flips_xyz),
        "image_axes": ["z", "y", "x"],
        "image_origin_reference": "recorded-stage-plus-local-ngff-translation",
        "camera_reference": "first-stored-detector-pixel",
        "stage_position_reference": "recorded-acquisition-frame-coordinates",
        **_physical_plan(directory, source, positions),
    }
    for store, group in zip(stores, groups[:4], strict=True):
        projection = "_max_z_" in store.name
        expected_origin = np.asarray(
            (0.5 * (int(expected_shape[0]) - 1) * spacing[0] if projection else 0, 0, 0)
        )
        xml_path = store / "OME" / "METADATA.ome.xml"
        ome = from_xml(xml_path.read_text("utf-8"))
        if len(ome.images) != len(positions):
            raise ValueError(
                "Processed image count does not match raw source positions"
            )
        for index, (image, position) in enumerate(
            zip(ome.images, positions, strict=True)
        ):
            label = image.stage_label
            if label is None or not np.allclose(
                (label.z, label.y, label.x), position, rtol=0, atol=0.0005 + 1e-9
            ):
                raise ValueError(
                    "Legacy stage-label reference is not the expected recorded acquisition frame"
                )
            label.z, label.x, label.y = round_spatial_values(position)
            label.x_unit = label.y_unit = label.z_unit = UnitsLength.MICROMETER
            group_path = store / str(index) / "zarr.json"
            series = _read_group(group_path)
            for level, dataset in enumerate(_datasets(series)):
                array = json.loads(
                    (store / str(index) / dataset["path"] / "zarr.json").read_bytes()
                )
                shape = array["shape"]
                if array.get("node_type") != "array" or len(shape) != 5:
                    raise ValueError("Unexpected processed array metadata")
                factor_y = _transform(dataset, "scale")[-2] / pixel
                factor_x = _transform(dataset, "scale")[-1] / pixel
                if (
                    abs(factor_x - factor_y) > 1e-9
                    or factor_y < 1
                    or factor_y != round(factor_y)
                ):
                    raise ValueError(
                        "Repair requires a stride-sampled isotropic YX pyramid"
                    )
                factor = round(factor_y)
                target = (
                    1 if projection else int(expected_shape[0]),
                    (int(expected_shape[1]) + factor - 1) // factor,
                    (int(expected_shape[2]) + factor - 1) // factor,
                )
                if tuple(shape[-3:]) != target or not np.allclose(
                    _transform(dataset, "translation")[-3:],
                    expected_origin,
                    rtol=0,
                    atol=0.0005 + 1e-9,
                ):
                    raise ValueError(
                        "Processed geometry is cropped or uses an unrecognized local origin"
                    )
                if level == 0 and not np.allclose(
                    _transform(dataset, "scale")[-3:], spacing, rtol=0, atol=1e-9
                ):
                    raise ValueError(
                        "Processed collections disagree on spatial calibration"
                    )
            _shift_datasets(series, delta)
            add(group_path, _json_bytes(series))
        add(xml_path, to_xml(ome).encode())
        attributes = group.setdefault("attributes", {})
        attributes["opm_processing_coordinates"] = coordinate_metadata
        attributes["opm_processing_stage_positions"] = {
            "axes": ["z", "x", "y"],
            "unit": "micrometer",
            "positions": coordinate_metadata["stage_positions_zxy_um"],
        }
        add(store / "zarr.json", _json_bytes(group))

    journal_path = directory / f"{stem}.processing.json"
    journal = json.loads(journal_path.read_bytes())
    registration = journal.get("registration", {})
    expected_keys = {f"{stem}_deskewed.ome.zarr", f"{stem}_decon_deskewed.ome.zarr"}
    if set(registration) != expected_keys:
        raise ValueError(
            "Registration journal has an unexpected set of tile collections"
        )
    associated = [
        record
        for record in registration.values()
        if record.get("fused_path") == fused_stores[0].name
    ]
    if (
        len(associated) != 1
        or associated[0].get("max_projection_path") != fused_stores[1].name
    ):
        raise ValueError("Fused image ownership is not unambiguous")
    tile_min = np.min(
        [tile["origin_zyx_um"] for tile in associated[0]["tiles"]], axis=0
    )
    base_origin = _transform(_datasets(groups[4])[0], "translation")[-3:]
    if not np.allclose(tile_min, base_origin, rtol=0, atol=0.0005 + 1e-9):
        raise ValueError("Fused origin does not match the registered tile space")
    for record in registration.values():
        tiles = record.get("tiles", [])
        if len(tiles) != len(positions) * acquisition.index_sizes["t"]:
            raise ValueError("Registration does not retain every source tile")
        for tile in tiles:
            tile["origin_zyx_um"] = list(
                round_spatial_values(np.asarray(tile["origin_zyx_um"]) + delta)
            )
        record["coordinate_repair"] = {
            "version": STAGE_COORDINATE_VERSION,
            "delta_zyx_um": delta.tolist(),
            "preserved_legacy_configuration_hash": True,
        }
    journal["coordinate_repair"] = {
        "version": STAGE_COORDINATE_VERSION,
        "delta_zyx_um": delta.tolist(),
        "scope": "uniform metadata translation; pixels and pairwise correlations unchanged",
        "derived_exports_stale": base_plan["stale_exports"],
    }
    add(journal_path, _json_bytes(journal))
    for store, group in zip(fused_stores, groups[4:], strict=True):
        _shift_datasets(group, delta)
        group["attributes"]["opm_processing_coordinates"] = {
            "version": STAGE_COORDINATE_VERSION,
            "metadata_repair_delta_zyx_um": delta.tolist(),
        }
        add(store / "zarr.json", _json_bytes(group))
    base_plan.update(
        status="ready",
        delta_zyx_um=delta.tolist(),
        assumptions=[
            "positive unflipped stage scans with a common scan span",
            "uncropped processed stride pyramids",
            "one uniform translation preserves image pixels and registration deltas",
            "recorded frame X can be depth-normalized; planned physical coordinates are separate",
            "AO hardware correction applications are not reconstructed as measured telemetry",
        ],
        files=[changes[key] for key in sorted(changes)],
    )
    dependencies = [
        *[path for store in stores for path in store.glob("*/*/zarr.json")],
        *[path for store in fused_stores for path in store.glob("*/zarr.json")],
        *(directory / f"{source.name.removesuffix('.zarr')}_ao_grid_results").glob(
            "time_*_z_*_ao_grid_results/exp_stage_positions.json"
        ),
    ]
    for path in dependencies:
        if path.is_file():
            dependency_hashes[str(path.relative_to(directory))] = _sha(
                path.read_bytes()
            )
    base_plan["read_dependencies"] = [
        {"path": relative, "sha256": fingerprint}
        for relative, fingerprint in sorted(dependency_hashes.items())
    ]
    for record in base_plan["read_dependencies"]:
        if _sha(_inside(directory, record["path"]).read_bytes()) != record["sha256"]:
            raise ValueError(
                "Repair inputs changed during planning; retry after acquisition is idle"
            )
    base_plan["plan_sha256"] = _plan_digest(base_plan)
    return base_plan
