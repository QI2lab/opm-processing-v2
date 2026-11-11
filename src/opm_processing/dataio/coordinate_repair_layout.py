"""Plan uniform stage-coordinate repairs across explicitly supplied directories."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ome_types import from_xml, to_xml
from ome_types.model import UnitsLength

from opm_processing.dataio import coordinate_repair as base
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

if TYPE_CHECKING:
    from collections.abc import Sequence


SPLIT_REPAIR_SCHEMA = "opm-stage-coordinate-repair-v2"


def _file_sha(path: Path) -> str:
    """Hash metadata histories without retaining entire frame histories in RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _uniform_scan_offset(offsets: np.ndarray) -> np.ndarray:
    """Require one scan correction at the declared micrometer precision."""
    values = np.asarray(offsets, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not len(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Source positions do not share one uniform scan-origin correction"
        )
    quantized = np.asarray([round_spatial_values(value) for value in values])
    if not np.array_equal(quantized, np.broadcast_to(quantized[0], quantized.shape)):
        raise ValueError(
            "Source positions do not share one uniform scan-origin correction"
        )
    return quantized[0].copy()


def _resolve_recorded(value: str, directory: Path, roots: dict[str, Path]) -> Path:
    """Honor an existing absolute association before considering portable copies."""
    recorded = Path(value).expanduser()
    if recorded.is_absolute() and recorded.exists():
        resolved = recorded.resolve()
    elif not recorded.is_absolute() and (directory / recorded).exists():
        resolved = (directory / recorded).resolve()
    else:
        candidates = {
            (root / recorded.name).resolve()
            for root in roots.values()
            if (root / recorded.name).exists()
        }
        if len(candidates) != 1:
            raise ValueError(f"Missing or ambiguous recorded artifact: {value}")
        resolved = candidates.pop()
    if not any(resolved.is_relative_to(root) for root in roots.values()):
        raise ValueError(
            f"Recorded artifact lies outside the supplied repair roots: {value}"
        )
    return resolved


def plan_split_stage_coordinate_repair(
    root: str | Path, additional_roots: Sequence[str | Path]
) -> dict[str, Any]:
    """Build a metadata-only repair plan for one acquisition split across roots.

    Parameters
    ----------
    root : str or pathlib.Path
        Primary directory containing the raw acquisition and saved physical plans.
    additional_roots : sequence of str or pathlib.Path
        Explicit additional directories containing processed data or its journal.

    Returns
    -------
    dict
        Version-two plan with root-qualified artifact paths, exact preimages and
        postimages, input dependencies, excluded artifacts, and a plan checksum.
        Planning reads metadata only and never writes a file.
    """
    directories = list(
        dict.fromkeys(
            Path(path).expanduser().resolve() for path in (root, *additional_roots)
        )
    )
    if any(not directory.is_dir() for directory in directories):
        raise ValueError(
            "Every explicitly supplied repair root must be an existing directory"
        )
    if any(
        a.is_relative_to(b) or b.is_relative_to(a)
        for i, a in enumerate(directories)
        for b in directories[i + 1 :]
    ):
        raise ValueError("Repair roots must not overlap")
    roots = {f"root_{index}": directory for index, directory in enumerate(directories)}
    primary = directories[0]
    source = resolve_acquisition_path(primary)
    if source.parent != primary:
        raise ValueError(
            "Raw acquisition must be directly inside the primary repair root"
        )
    stem = acquisition_stem(source)

    def identity(path: Path) -> dict[str, str]:
        for root_id, directory in roots.items():
            if path.is_relative_to(directory):
                relative = str(path.relative_to(directory))
                if base._inside(directory, relative) != path:
                    break
                return {"root_id": root_id, "path": relative}
        raise ValueError(
            f"Repair artifact is outside the declared roots or redirected: {path}"
        )

    stores: dict[str, Path] = {}
    for suffix in base._COLLECTION_SUFFIXES:
        matches = [
            directory / f"{stem}{suffix}"
            for directory in directories
            if (directory / f"{stem}{suffix}").is_dir()
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Processed collection has ambiguous copies: {stem}{suffix}"
            )
        if matches:
            stores[suffix] = matches[0]
    volumes = {
        suffix: store for suffix, store in stores.items() if "_max_z_" not in suffix
    }
    if not volumes:
        raise ValueError(
            "No full processed tile collection was found in the supplied roots"
        )
    journals = [
        directory / f"{stem}.processing.json"
        for directory in directories
        if (directory / f"{stem}.processing.json").is_file()
    ]
    if len(journals) != 1:
        raise ValueError("Expected one processing journal across the supplied roots")
    journal_path = journals[0]
    initial_contents = {journal_path: journal_path.read_bytes()}
    journal = json.loads(initial_contents[journal_path])
    if (
        _resolve_recorded(journal["source"]["path"], journal_path.parent, roots)
        != source
    ):
        raise ValueError(
            "Processing journal does not identify the selected raw acquisition"
        )
    registrations = journal.get("registration", {})
    owners = []
    for key, record in registrations.items():
        processed = _resolve_recorded(key, journal_path.parent, roots)
        if processed not in volumes.values():
            raise ValueError(
                "Journal registration refers to an unavailable processed collection"
            )
        if record.get("fused_path") or record.get("max_projection_path"):
            if not record.get("fused_path") or not record.get("max_projection_path"):
                raise ValueError(
                    "Current fused image pair has incomplete journal ownership"
                )
            owners.append((key, processed, record))
    if len(owners) != 1:
        raise ValueError(
            "Expected one currently owned fused image and maximum projection"
        )
    owner_key, owner, owning_record = owners[0]
    fused = _resolve_recorded(owning_record["fused_path"], journal_path.parent, roots)
    maximum = _resolve_recorded(
        owning_record["max_projection_path"], journal_path.parent, roots
    )
    if (
        fused.name != f"{stem}_fused.ome.zarr"
        or maximum.name != f"{stem}_max_z_fused.ome.zarr"
    ):
        raise ValueError("Current fused artifacts have unexpected acquisition names")
    active = [*stores.values(), fused, maximum]
    groups = {}
    for store in active:
        path = store / "zarr.json"
        initial_contents[path] = path.read_bytes()
        group = json.loads(initial_contents[path])
        if group.get("node_type") != "group":
            raise ValueError(f"Expected image group metadata: {path}")
        groups[store] = group
    skipped = [
        {
            **identity(candidate),
            "reason": "unowned duplicate; current journal associates a different fused artifact",
        }
        for directory in directories
        for name, owned in ((fused.name, fused), (maximum.name, maximum))
        if (candidate := directory / name).is_dir() and candidate != owned
    ]
    stale = [
        {
            **identity(path),
            "reason": "derived export retains the earlier coordinate frame",
        }
        for directory in directories
        for path in (
            *directory.glob("*_fused.ome.tif"),
            *directory.glob("*_fused.ome.tiff"),
            *directory.glob("ao_overview*"),
            directory / "roi_exports",
        )
        if path.exists()
    ]
    plan: dict[str, Any] = {
        "schema": SPLIT_REPAIR_SCHEMA,
        "root": str(primary),
        "roots": {key: str(path) for key, path in roots.items()},
        "stem": stem,
        "source": identity(source),
        "coordinate_version": STAGE_COORDINATE_VERSION,
        "artifact_identities": {
            "raw_source": identity(source),
            "processed_collections": [identity(store) for store in stores.values()],
            "processing_journal": identity(journal_path),
            "owning_processed_collection": identity(owner),
            "fused_image": identity(fused),
            "max_projection": identity(maximum),
        },
        "skipped_artifacts": skipped,
        "stale_exports": stale,
        "files": [],
        "read_dependencies": [],
    }
    versions = [
        group.get("attributes", {}).get("opm_processing_coordinates", {}).get("version")
        for group in groups.values()
    ]
    if all(version == STAGE_COORDINATE_VERSION for version in versions):
        if (
            journal.get("coordinate_repair", {}).get("version")
            != STAGE_COORDINATE_VERSION
        ):
            raise ValueError(
                "Repaired image versions disagree with the processing journal"
            )
        plan["status"] = "already_repaired"
        plan["plan_sha256"] = base._plan_digest(plan)
        return plan
    if any(version is not None for version in versions):
        raise ValueError(
            "Mixed or unsupported coordinate versions require manual review"
        )

    dependencies: dict[tuple[str, str], dict[str, str]] = {}

    def remember(path: Path, fingerprint: str | None = None) -> None:
        record = identity(path)
        key = (record["root_id"], record["path"])
        digest = _file_sha(path) if fingerprint is None else fingerprint
        if key in dependencies and dependencies[key]["sha256"] != digest:
            raise ValueError("Repair metadata changed during planning")
        dependencies[key] = {**record, "sha256": digest}

    def read_json(path: Path) -> dict[str, Any]:
        content = path.read_bytes()
        remember(path, base._sha(content))
        return json.loads(content)

    def read_group(path: Path) -> dict[str, Any]:
        value = read_json(path)
        if value.get("node_type") != "group":
            raise ValueError(f"Expected group metadata, not an image array: {path}")
        return value

    def read_array(path: Path) -> dict[str, Any]:
        value = read_json(path)
        shape = value.get("shape", ())
        if (
            value.get("node_type") != "array"
            or len(shape) != 5
            or any(
                not isinstance(size, int) or isinstance(size, bool) or size <= 0
                for size in shape
            )
        ):
            raise ValueError(f"Expected nonempty TCZYX array metadata: {path}")
        return value

    changes: dict[tuple[str, str], dict[str, str]] = {}

    def add(path: Path, content: bytes) -> None:
        record = identity(path)
        if not base._metadata_allowed(record["path"], stem):
            raise ValueError("Split repair attempted a non-metadata write")
        before = path.read_bytes()
        remember(path, base._sha(before))
        if content != before:
            changes[(record["root_id"], record["path"])] = {
                **record,
                "before_sha256": base._sha(before),
                "after_sha256": base._sha(content),
                "after_text": content.decode("utf-8"),
            }

    for path, content in initial_contents.items():
        remember(path, base._sha(content))
    source_dependencies = {
        source / "zarr.json",
        source / "OME" / "METADATA.ome.xml",
        *source.glob("*/zarr.json"),
        *source.glob("*/*/zarr.json"),
        primary / f"{stem}.manifest.json",
    }
    for path in sorted(source_dependencies):
        if path.is_file():
            remember(path)
    ao_root = primary / f"{source.name.removesuffix('.zarr')}_ao_grid_results"
    for name in ("exp_stage_positions.json", "ao_stage_positions.json"):
        for path in sorted(ao_root.glob(f"time_*_z_*_ao_grid_results/{name}")):
            remember(path)
    acquisition = inspect_acquisition(source)
    if (
        acquisition.mode != "stage"
        or acquisition.is_2d
        or not acquisition.scan_axis_reversed
        or any(acquisition.stage_axis_flips_xyz)
    ):
        raise ValueError(
            "Split repair requires positive unflipped three-dimensional stage scans"
        )
    if any(
        (
            acquisition.excess_scan_positions,
            acquisition.excess_scan_start_positions,
            acquisition.excess_scan_end_positions,
        )
    ):
        raise ValueError(
            "Split repair has not been validated for excess-frame cropping"
        )
    positions = np.asarray(acquisition.stage_positions_zxy, dtype=float)
    offsets = deskewed_stage_scan_offsets(acquisition)
    if not len(positions):
        raise ValueError(
            "Source positions do not share one uniform scan-origin correction"
        )
    uniform_offset = _uniform_scan_offset(offsets)
    groups = {store: read_group(store / "zarr.json") for store in active}
    journal = read_json(journal_path)
    registrations = journal["registration"]
    owning_record = registrations[owner_key]
    first_store = next(iter(volumes.values()))
    first = base._datasets(read_group(first_store / "0" / "zarr.json"))[0]
    first_scale = np.asarray(base._transform(first, "scale"), dtype=float)
    spacing = first_scale[-3:]
    pixel = float(acquisition.pixel_size_um)
    averaging = round(float(spacing[0]) / pixel)
    if averaging < 1 or not np.allclose(
        spacing, (pixel * averaging, pixel, pixel), rtol=0, atol=1e-9
    ):
        raise ValueError(
            "Processed sampling differs from the recorded camera calibration"
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
        round_spatial_values(
            np.add(uniform_offset, (0.5 * (averaging - 1) * pixel, 0, 0))
        )
    )
    physical = base._physical_plan(primary, source, positions)
    if physical:
        physical["physical_stage_plan_source"] = [
            str((primary / value).resolve())
            for value in physical["physical_stage_plan_source"]
        ]
    coordinate_metadata = {
        "version": STAGE_COORDINATE_VERSION,
        "stage_axes": ["z", "x", "y"],
        "stage_positions_zxy_um": [
            list(round_spatial_values(position)) for position in positions
        ],
        "placement_stage_positions_zxy_um": [
            list(round_spatial_values(position)) for position in positions
        ],
        "stage_axis_flips_xyz": list(acquisition.stage_axis_flips_xyz),
        "image_axes": ["z", "y", "x"],
        "image_origin_reference": "recorded-stage-plus-local-ngff-translation",
        "camera_reference": "first-stored-detector-pixel",
        "stage_position_reference": "recorded-acquisition-frame-coordinates",
        **physical,
    }
    t, c = acquisition.index_sizes["t"], acquisition.index_sizes["c"]
    for suffix, store in stores.items():
        projection = "_max_z_" in suffix
        expected_origin = (
            0.5 * (int(expected_shape[0]) - 1) * spacing[0] if projection else 0,
            0,
            0,
        )
        xml_path = store / "OME" / "METADATA.ome.xml"
        xml = xml_path.read_bytes()
        remember(xml_path, base._sha(xml))
        ome = from_xml(xml.decode("utf-8"))
        if len(ome.images) != len(positions):
            raise ValueError(
                "Processed image count differs from the recorded source positions"
            )
        for index, (image, position) in enumerate(
            zip(ome.images, positions, strict=True)
        ):
            label = image.stage_label
            if (
                label is None
                or any(
                    unit != UnitsLength.MICROMETER
                    for unit in (label.z_unit, label.x_unit, label.y_unit)
                )
                or not np.allclose(
                    (label.z, label.y, label.x), position, rtol=0, atol=0.0005 + 1e-9
                )
            ):
                raise ValueError(
                    "Processed stage labels do not retain the expected legacy source axes and units"
                )
            if (image.pixels.size_t, image.pixels.size_c) != (t, c):
                raise ValueError(
                    "Processed OME time or channel counts disagree with acquisition"
                )
            if (image.pixels.size_z, image.pixels.size_y, image.pixels.size_x) != (
                1 if projection else int(expected_shape[0]),
                int(expected_shape[1]),
                int(expected_shape[2]),
            ):
                raise ValueError(
                    "Processed OME image shape differs from acquisition geometry"
                )
            label.z, label.x, label.y = round_spatial_values(position)
            label.x_unit = label.y_unit = label.z_unit = UnitsLength.MICROMETER
            group_path = store / str(index) / "zarr.json"
            series = read_group(group_path)
            for dataset in base._datasets(series):
                scale = np.asarray(base._transform(dataset, "scale"))
                factor = scale[-1] / pixel
                if (
                    factor < 1
                    or abs(factor - round(factor)) > 1e-9
                    or abs(scale[-2] / pixel - factor) > 1e-9
                    or not np.allclose(scale[:3], first_scale[:3], rtol=0, atol=1e-9)
                ):
                    raise ValueError(
                        "Processed pyramid sampling is incompatible with the native camera grid"
                    )
                factor = round(factor)
                shape = tuple(
                    read_array(store / str(index) / dataset["path"] / "zarr.json")[
                        "shape"
                    ]
                )
                expected = (
                    t,
                    c,
                    1 if projection else int(expected_shape[0]),
                    (int(expected_shape[1]) + factor - 1) // factor,
                    (int(expected_shape[2]) + factor - 1) // factor,
                )
                if shape != expected or not np.allclose(
                    base._transform(dataset, "translation")[-3:],
                    expected_origin,
                    rtol=0,
                    atol=0.0005 + 1e-9,
                ):
                    raise ValueError(
                        "Processed image is cropped or has an unsupported local coordinate origin"
                    )
            base._shift_datasets(series, delta)
            add(group_path, base._json_bytes(series))
        add(xml_path, to_xml(ome).encode())
        attributes = groups[store].setdefault("attributes", {})
        attributes["opm_processing_coordinates"] = coordinate_metadata
        attributes["opm_processing_stage_positions"] = {
            "axes": ["z", "x", "y"],
            "unit": "micrometer",
            "positions": coordinate_metadata["stage_positions_zxy_um"],
        }
        add(store / "zarr.json", base._json_bytes(groups[store]))

    expected_tiles = {
        (time, position) for time in range(t) for position in range(len(positions))
    }
    for record in registrations.values():
        tiles = record.get("tiles", [])
        if (
            len(tiles) != len(expected_tiles)
            or {
                (int(tile["time_index"]), int(tile["position_index"])) for tile in tiles
            }
            != expected_tiles
        ):
            raise ValueError(
                "Registration does not retain every source time and position"
            )
    native_tiles = np.asarray(
        [tile["origin_zyx_um"] for tile in owning_record["tiles"]], dtype=float
    )
    if not np.isfinite(native_tiles).all():
        raise ValueError("Registered tile coordinates must be finite")
    full_datasets = base._datasets(groups[fused])
    max_datasets = base._datasets(groups[maximum])
    if [item["path"] for item in full_datasets] != [
        item["path"] for item in max_datasets
    ]:
        raise ValueError("Owned fused volume and projection disagree on pyramid levels")
    base_origin = np.asarray(base._transform(full_datasets[0], "translation")[-3:])
    if not np.allclose(
        native_tiles.min(axis=0), base_origin, rtol=0, atol=0.0005 + 1e-9
    ):
        raise ValueError(
            "Owned fused origin differs from the journal's current registered tile space"
        )
    full_shape = None
    factors = []
    for full_dataset, max_dataset in zip(full_datasets, max_datasets, strict=True):
        fs, ms = (
            np.asarray(base._transform(item, "scale"))
            for item in (full_dataset, max_dataset)
        )
        factor = fs[-1] / spacing[-1]
        if (
            factor < 1
            or abs(factor - round(factor)) > 1e-9
            or not np.allclose(fs[:2], first_scale[:2], rtol=0, atol=1e-9)
            or not np.allclose(fs[-3:], spacing * factor, rtol=0, atol=1e-9)
            or not np.allclose(fs, ms, rtol=0, atol=1e-9)
        ):
            raise ValueError(
                "Owned fused pyramid has incompatible physical calibration"
            )
        factor = round(factor)
        factors.append(factor)
        full_array = read_array(fused / full_dataset["path"] / "zarr.json")
        max_array = read_array(maximum / max_dataset["path"] / "zarr.json")
        shape, max_shape = tuple(full_array["shape"]), tuple(max_array["shape"])
        if full_shape is None:
            if factor != 1:
                raise ValueError("Owned fused pyramid must start at native sampling")
            full_shape = shape
        expected = (
            *full_shape[:2],
            *((int(value) + factor - 1) // factor for value in full_shape[-3:]),
        )
        ft, mt = (
            np.asarray(base._transform(item, "translation"))
            for item in (full_dataset, max_dataset)
        )
        native_projection_center = (
            base_origin[0] + 0.5 * (full_shape[2] - 1) * spacing[0]
        )
        if (
            shape != expected
            or shape[:2] != (t, c)
            or max_shape != (t, c, 1, *shape[-2:])
            or not np.isfinite(ft).all()
            or not np.isfinite(mt).all()
            or not np.allclose(ft[-2:], mt[-2:], rtol=0, atol=0.0005 + 1e-9)
            or (factor == 1 and abs(mt[2] - native_projection_center) > 0.0005 + 1e-9)
        ):
            raise ValueError(
                "Owned fused volume and maximum projection have incompatible geometry"
            )
    extent = (
        np.max(native_tiles + np.asarray(expected_shape) * spacing, axis=0)
        - native_tiles.min(axis=0)
    ) / spacing
    nearest = np.rint(extent)
    extent = np.where(abs(extent - nearest) <= 1e-7, nearest, extent)
    alignment = math.lcm(1, *factors)
    padded = tuple(
        int((math.ceil(float(value)) + alignment - 1) // alignment * alignment)
        for value in extent
    )
    if tuple(full_shape[-3:]) != padded:
        raise ValueError(
            "Owned fused shape does not enclose the journal's current registered tiles"
        )

    for record in registrations.values():
        for tile in record["tiles"]:
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
        "derived_exports_stale": stale,
    }
    add(journal_path, base._json_bytes(journal))
    for store in (fused, maximum):
        base._shift_datasets(groups[store], delta)
        groups[store]["attributes"]["opm_processing_coordinates"] = {
            "version": STAGE_COORDINATE_VERSION,
            "metadata_repair_delta_zyx_um": delta.tolist(),
        }
        add(store / "zarr.json", base._json_bytes(groups[store]))
    plan.update(
        status="ready",
        delta_zyx_um=delta.tolist(),
        assumptions=[
            "positive unflipped stage scans with one common retained scan span",
            "uncropped processed stride pyramids at recorded camera calibration",
            "current fused artifacts are identified by explicit journal ownership",
            "unowned same-named duplicates are excluded from metadata writes",
            "one common translation preserves pixels and pairwise correlation measurements",
            "planned physical stage coordinates remain distinct from normalized frame coordinates",
        ],
        files=[changes[key] for key in sorted(changes)],
        read_dependencies=[dependencies[key] for key in sorted(dependencies)],
    )
    for dependency in plan["read_dependencies"]:
        path = base._inside(roots[dependency["root_id"]], dependency["path"])
        if _file_sha(path) != dependency["sha256"]:
            raise ValueError(
                "Repair input changed during planning; retry after acquisition is idle"
            )
    plan["plan_sha256"] = base._plan_digest(plan)
    return plan
