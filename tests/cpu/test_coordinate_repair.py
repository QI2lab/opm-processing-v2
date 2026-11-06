"""Repair approved metadata while proving all raw and image files stay intact."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ome_types import from_xml, to_xml

from opm_processing.dataio import coordinate_repair
from opm_processing.dataio.coordinate_repair import (
    apply_stage_coordinate_repair,
    plan_stage_coordinate_repair,
)
from opm_processing.dataio.position_collection import create_position_collection
from opm_processing.imageprocessing.coordinates import STAGE_COORDINATE_VERSION

pytestmark = pytest.mark.unit


def _write_json(path, value):
    """Persist a visible independent metadata fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), "utf-8")


def _snapshot(root):
    """Fingerprint every existing file without opening image arrays."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def legacy_repair_case(acquisition_factory, tmp_path):
    """Build two positively scanned legacy tiles and explicit fused pyramids."""
    dataset = acquisition_factory(
        np.zeros((1, 2, 1, 6, 4, 5), np.uint16),
        name="repair_case",
        mode="stage",
        stage_positions_zxy=((3, 100, 20), (3, 150, 20)),
        scan_axis_step_um=2,
        pixel_size_um=1,
        camera_offset=0,
        camera_conversion=1,
    )
    stem = "repair_case"
    suffixes = (
        "_deskewed.ome.zarr",
        "_decon_deskewed.ome.zarr",
        "_max_z_deskewed.ome.zarr",
        "_max_z_decon_deskewed.ome.zarr",
    )
    for suffix in suffixes:
        path = tmp_path / f"{stem}{suffix}"
        collection = create_position_collection(
            path,
            (1, 2, 1, 1, 16, 8),
            (2, 1, 1),
            stage_positions=((3, 100, 20), (3, 150, 20)),
            spatial_offset_um=(0, 0, 0),
            multiscale_factors_yx=(2,) if "_max_z_" in suffix else (),
        )
        # Stored image bytes exist and must survive repair byte-for-byte.
        for array in collection.arrays:
            array.write(np.ones(array.shape, np.uint16)).result()
        xml_path = path / "OME" / "METADATA.ome.xml"
        ome = from_xml(xml_path.read_text("utf-8"))
        for image in ome.images:
            image.stage_label.x, image.stage_label.y = (
                image.stage_label.y,
                image.stage_label.x,
            )
        xml_path.write_text(to_xml(ome), "utf-8")
        group_path = path / "zarr.json"
        group = json.loads(group_path.read_text())
        group["attributes"].pop("opm_processing_stage_positions", None)
        _write_json(group_path, group)
    axes = [
        {"name": name, "type": "space", "unit": "micrometer"}
        if name in "zyx"
        else {"name": name, "type": "time" if name == "t" else "channel"}
        for name in "tczyx"
    ]
    for projection in (False, True):
        name = f"{stem}_{'max_z_' if projection else ''}fused.ome.zarr"
        path = tmp_path / name
        datasets = []
        for level, factor in enumerate((1, 2)):
            origin_z = -2 if projection and level == 0 else -3
            datasets.append(
                {
                    "path": str(level),
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1, 1, 2 * factor, factor, factor]},
                        {
                            "type": "translation",
                            "translation": [0, 0, origin_z, -150, 20],
                        },
                    ],
                }
            )
            _write_json(
                path / str(level) / "zarr.json",
                {
                    "node_type": "array",
                    "shape": [1, 1, 1 if projection else 2, 66 // factor, 8 // factor],
                },
            )
            chunk = path / str(level) / "c" / "0"
            chunk.parent.mkdir(parents=True, exist_ok=True)
            chunk.write_bytes(b"an opaque compressed pixel chunk")
        _write_json(
            path / "zarr.json",
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {
                        "version": "0.5",
                        "multiscales": [{"axes": axes, "datasets": datasets}],
                    }
                },
            },
        )
    tiles = [
        {"time_index": 0, "position_index": i, "origin_zyx_um": [-3, -x, 20]}
        for i, x in enumerate((100, 150))
    ]
    registration = {}
    for suffix in suffixes[:2]:
        registration[f"{stem}{suffix}"] = {
            "configuration_sha256": "old-registration-fingerprint",
            "pairwise_metrics": {"0,1": {"shift": [0, 0, 0], "ssim": 0.9}},
            "tiles": copy.deepcopy(tiles),
        }
    registration[f"{stem}_decon_deskewed.ome.zarr"].update(
        fused_path=f"{stem}_fused.ome.zarr",
        max_projection_path=f"{stem}_max_z_fused.ome.zarr",
    )
    _write_json(
        tmp_path / f"{stem}.processing.json",
        {
            "schema": "fixture",
            "source": {"path": str(dataset.path)},
            "outputs": {},
            "registration": registration,
        },
    )
    return tmp_path


def test_metadata_repair_keeps_every_pixel_and_raw_file_unchanged(legacy_repair_case):
    """Only whitelisted metadata changes; every level retains its relative origin."""
    root = legacy_repair_case
    before = _snapshot(root)
    plan = plan_stage_coordinate_repair(root)
    assert plan["delta_zyx_um"] == [0.5, -10, 0]
    paths = {record["path"] for record in plan["files"]}
    assert len(paths) == 19
    assert all(not path.startswith("repair_case.ome.zarr") for path in paths)
    assert all(not path.endswith("0/0/zarr.json") for path in paths)
    result = apply_stage_coordinate_repair(plan)
    assert result["status"] == "applied"
    after = _snapshot(root)
    for path, fingerprint in before.items():
        if path not in paths:
            assert after[path] == fingerprint, path
    assert {path for path in before if before[path] != after[path]} == paths
    group = json.loads((root / "repair_case_fused.ome.zarr" / "zarr.json").read_text())
    for dataset in group["attributes"]["ome"]["multiscales"][0]["datasets"]:
        assert dataset["coordinateTransformations"][1]["translation"][-3:] == [
            -2.5,
            -160,
            20,
        ]
    max_group = json.loads(
        (root / "repair_case_max_z_fused.ome.zarr" / "zarr.json").read_text()
    )
    assert [
        d["coordinateTransformations"][1]["translation"][2]
        for d in max_group["attributes"]["ome"]["multiscales"][0]["datasets"]
    ] == [-1.5, -2.5]
    xml = from_xml(
        (root / "repair_case_deskewed.ome.zarr" / "OME" / "METADATA.ome.xml").read_text(
            "utf-8"
        )
    )
    assert (
        xml.images[0].stage_label.x,
        xml.images[0].stage_label.y,
        xml.images[0].stage_label.z,
    ) == (100, 20, 3)
    journal = json.loads((root / "repair_case.processing.json").read_text())
    for record in journal["registration"].values():
        assert record["configuration_sha256"] == "old-registration-fingerprint"
        assert record["pairwise_metrics"] == {"0,1": {"shift": [0, 0, 0], "ssim": 0.9}}
        assert record["tiles"][0]["origin_zyx_um"] == [-2.5, -110, 20]
    assert (root / ".coordinate-repair-backups").is_dir()
    assert apply_stage_coordinate_repair(plan)["status"] == "already_repaired"
    assert plan_stage_coordinate_repair(root)["status"] == "already_repaired"


def test_preimage_mismatch_never_starts_a_repair(legacy_repair_case):
    """A stale reviewed plan cannot overwrite later metadata changes."""
    root = legacy_repair_case
    plan = plan_stage_coordinate_repair(root)
    path = root / plan["files"][-1]["path"]
    path.write_bytes(path.read_bytes() + b"\n")
    before = _snapshot(root)
    with pytest.raises(ValueError, match="preimage mismatch"):
        apply_stage_coordinate_repair(plan)
    assert _snapshot(root) == before
    assert not (root / ".coordinate-repair-backups").exists()


@pytest.mark.parametrize("fail_after_replace", [False, True])
def test_interrupted_repair_rolls_back_approved_metadata(
    legacy_repair_case, monkeypatch, fail_after_replace
):
    """A failed replacement restores all earlier metadata from exact preimages."""
    root = legacy_repair_case
    plan = plan_stage_coordinate_repair(root)
    before = _snapshot(root)
    original = coordinate_repair._atomic_write
    calls = 0

    def fail_once(path, value):
        nonlocal calls
        calls += 1
        if calls == 3:
            if fail_after_replace:
                original(path, value)
            raise OSError("simulated interrupted metadata replacement")
        original(path, value)

    monkeypatch.setattr(coordinate_repair, "_atomic_write", fail_once)
    with pytest.raises(OSError, match="simulated interrupted"):
        apply_stage_coordinate_repair(plan)
    after = _snapshot(root)
    assert all(after[path] == value for path, value in before.items())
    assert list((root / ".coordinate-repair-backups").glob("*/manifest.json"))


def test_mixed_coordinate_versions_are_rejected(legacy_repair_case):
    """Never apply a common shift twice to one of several image artifacts."""
    root = legacy_repair_case
    path = root / "repair_case_fused.ome.zarr" / "zarr.json"
    group = json.loads(path.read_text())
    group["attributes"]["opm_processing_coordinates"] = {
        "version": STAGE_COORDINATE_VERSION
    }
    _write_json(path, group)
    before = _snapshot(root)
    with pytest.raises(ValueError, match="Mixed or unsupported"):
        plan_stage_coordinate_repair(root)
    assert _snapshot(root) == before


def test_cropped_or_unequal_geometry_is_rejected(legacy_repair_case):
    """Metadata-only repair must not conceal a tile-dependent placement change."""
    root = legacy_repair_case
    path = root / "repair_case_deskewed.ome.zarr" / "0" / "0" / "zarr.json"
    array = json.loads(path.read_text())
    array["shape"][-2] -= 1
    _write_json(path, array)
    before = _snapshot(root)
    with pytest.raises(ValueError, match="cropped or uses"):
        plan_stage_coordinate_repair(root)
    assert _snapshot(root) == before


def test_tampered_plan_or_array_metadata_write_is_rejected(legacy_repair_case):
    """Even a rehashed plan cannot expand the metadata whitelist into image arrays."""
    root = legacy_repair_case
    plan = plan_stage_coordinate_repair(root)
    bad = copy.deepcopy(plan)
    bad["files"][0]["path"] = "repair_case_deskewed.ome.zarr/0/0/zarr.json"
    before = _snapshot(root)
    with pytest.raises(ValueError, match="fingerprint"):
        apply_stage_coordinate_repair(bad)
    bad["plan_sha256"] = coordinate_repair._plan_digest(bad)
    with pytest.raises(ValueError, match="non-metadata path"):
        apply_stage_coordinate_repair(bad)
    assert _snapshot(root) == before


def test_changed_read_dependency_does_not_touch_any_metadata(legacy_repair_case):
    """Geometry or source changes invalidate a reviewed plan before any backup."""
    root = legacy_repair_case
    plan = plan_stage_coordinate_repair(root)
    path = root / "repair_case_deskewed.ome.zarr" / "0" / "0" / "zarr.json"
    path.write_bytes(path.read_bytes() + b"\n")
    before = _snapshot(root)
    with pytest.raises(ValueError, match="input dependency changed"):
        apply_stage_coordinate_repair(plan)
    assert _snapshot(root) == before
    assert not (root / ".coordinate-repair-backups").exists()


@pytest.fixture
def split_repair_case(legacy_repair_case):
    """Separate raw/current fusion from processed tiles and their journal."""
    root = legacy_repair_case
    primary, additional = root / "volume_f", root / "volume_g"
    primary.mkdir()
    additional.mkdir()
    stem = "repair_case"
    journal = json.loads((root / f"{stem}.processing.json").read_text())
    destinations = {
        f"{stem}.ome.zarr": primary,
        f"{stem}_fused.ome.zarr": primary,
        f"{stem}_max_z_fused.ome.zarr": primary,
        f"{stem}_deskewed.ome.zarr": additional,
        f"{stem}_max_z_deskewed.ome.zarr": additional,
    }
    for name, destination in destinations.items():
        source, target = root / name, destination / name
        assert source.resolve().is_relative_to(root.resolve())
        assert target.resolve().is_relative_to(root.resolve())
        source.rename(target)
    # Full fusion pads native Z to two planes for its factor-two pyramid.
    # The coarse image therefore has one plane, as the actual producer writes.
    coarse_path = primary / f"{stem}_fused.ome.zarr" / "1" / "zarr.json"
    coarse = json.loads(coarse_path.read_text())
    coarse["shape"][2] = 1
    _write_json(coarse_path, coarse)
    owner = copy.deepcopy(journal["registration"][f"{stem}_deskewed.ome.zarr"])
    owner["fused_path"] = str(primary / f"{stem}_fused.ome.zarr")
    owner["max_projection_path"] = str(primary / f"{stem}_max_z_fused.ome.zarr")
    owner_key = str(additional / f"{stem}_deskewed.ome.zarr")
    journal["source"]["path"] = str(primary / f"{stem}.ome.zarr")
    journal["registration"] = {owner_key: owner}
    journal_path = additional / f"{stem}.processing.json"
    _write_json(journal_path, journal)
    (root / f"{stem}.processing.json").unlink()

    # This older same-named G output has no journal owner and different geometry.
    orphan = additional / f"{stem}_fused.ome.zarr"
    current = primary / f"{stem}_fused.ome.zarr"
    group = json.loads((current / "zarr.json").read_text())
    for dataset in group["attributes"]["ome"]["multiscales"][0]["datasets"]:
        dataset["coordinateTransformations"][1]["translation"][3] += 500
        level = dataset["path"]
        array = json.loads((current / level / "zarr.json").read_text())
        array["shape"][-2] += 7
        _write_json(orphan / level / "zarr.json", array)
        chunk = orphan / level / "c" / "0"
        chunk.parent.mkdir(parents=True, exist_ok=True)
        chunk.write_bytes(b"unowned historical image pixels")
    _write_json(orphan / "zarr.json", group)
    return SimpleNamespace(
        root=root,
        primary=primary,
        additional=additional,
        orphan=orphan,
        journal_path=journal_path,
        owner_key=owner_key,
        journal=journal,
    )


def _split_record_path(plan, record):
    """Resolve a reviewed multi-root fixture record without assuming one root."""
    return Path(plan["roots"][record["root_id"]]) / record["path"]


def test_split_repair_changes_only_owned_available_artifacts(split_repair_case):
    """Repair split outputs once while preserving absolute paths and orphan pixels."""
    case = split_repair_case
    before = _snapshot(case.root)
    plan = plan_stage_coordinate_repair(
        case.primary, additional_roots=(case.additional,)
    )
    assert plan["schema"] == "opm-stage-coordinate-repair-v2"
    assert plan["roots"] == {
        "root_0": str(case.primary),
        "root_1": str(case.additional),
    }
    assert plan["source"] == {"root_id": "root_0", "path": "repair_case.ome.zarr"}
    assert plan["delta_zyx_um"] == [0.5, -10, 0]
    approved = {
        str(_split_record_path(plan, record).relative_to(case.root))
        for record in plan["files"]
    }
    assert all(
        not _split_record_path(plan, record).is_relative_to(
            case.primary / "repair_case.ome.zarr"
        )
        for record in plan["files"]
    )
    assert all(
        not _split_record_path(plan, record).is_relative_to(case.orphan)
        for record in plan["files"]
    )
    assert all("decon" not in record["path"] for record in plan["files"])
    assert "repair_case_fused.ome.zarr" in json.dumps(plan["skipped_artifacts"])
    result = apply_stage_coordinate_repair(plan)
    assert result["status"] == "applied"
    assert set(result["backup_directories"]) == {"root_0", "root_1"}
    after = _snapshot(case.root)
    assert {path for path in before if before[path] != after[path]} == approved
    for path, fingerprint in before.items():
        if path not in approved:
            assert after[path] == fingerprint, path
    journal = json.loads(case.journal_path.read_text())
    assert journal["source"]["path"] == case.journal["source"]["path"]
    assert set(journal["registration"]) == {case.owner_key}
    owner = journal["registration"][case.owner_key]
    original = case.journal["registration"][case.owner_key]
    for key in (
        "fused_path",
        "max_projection_path",
        "configuration_sha256",
        "pairwise_metrics",
    ):
        assert owner[key] == original[key]
    assert owner["tiles"][0]["origin_zyx_um"] == [-2.5, -110, 20]
    for name in ("repair_case_fused.ome.zarr", "repair_case_max_z_fused.ome.zarr"):
        group = json.loads((case.primary / name / "zarr.json").read_text())
        assert (
            group["attributes"]["opm_processing_coordinates"]["version"]
            == STAGE_COORDINATE_VERSION
        )
        assert (
            group["attributes"]["ome"]["multiscales"][0]["datasets"][0][
                "coordinateTransformations"
            ][1]["translation"][3]
            == -160
        )
    assert not (case.additional / "repair_case_decon_deskewed.ome.zarr").exists()
    assert apply_stage_coordinate_repair(plan)["status"] == "already_repaired"
    assert (
        plan_stage_coordinate_repair(case.primary, additional_roots=(case.additional,))[
            "status"
        ]
        == "already_repaired"
    )


def test_changed_external_dependency_refuses_the_entire_split_repair(split_repair_case):
    """Raw metadata on the other root is checked before any backup or write."""
    case = split_repair_case
    plan = plan_stage_coordinate_repair(
        case.primary, additional_roots=(case.additional,)
    )
    dependency = next(
        record
        for record in plan["read_dependencies"]
        if record["root_id"] == "root_0"
        and record["path"].startswith("repair_case.ome.zarr")
    )
    path = _split_record_path(plan, dependency)
    path.write_bytes(path.read_bytes() + b"\n")
    before = _snapshot(case.root)
    with pytest.raises(ValueError, match="dependency changed"):
        apply_stage_coordinate_repair(plan)
    assert _snapshot(case.root) == before
    assert not (case.primary / ".coordinate-repair-backups").exists()
    assert not (case.additional / ".coordinate-repair-backups").exists()


def test_split_repair_backs_up_both_roots_before_writing_and_rolls_back(
    split_repair_case, monkeypatch
):
    """A failure after replacing metadata on both roots restores both preimages."""
    case = split_repair_case
    plan = plan_stage_coordinate_repair(
        case.primary, additional_roots=(case.additional,)
    )
    before = _snapshot(case.root)
    approved = {
        _split_record_path(plan, record).resolve(): record for record in plan["files"]
    }
    original = coordinate_repair._atomic_write
    touched_roots = set()
    failed = False

    def fail_after_crossing_roots(path, content):
        nonlocal failed
        resolved = Path(path).resolve()
        if resolved not in approved or failed:
            return original(path, content)
        for root_id, root_path in plan["roots"].items():
            manifests = list(
                (Path(root_path) / ".coordinate-repair-backups").glob("*/manifest.json")
            )
            assert len(manifests) == 1
            manifest = json.loads(manifests[0].read_text())
            assert manifest["root_id"] == root_id
            for record in plan["files"]:
                if record["root_id"] == root_id:
                    backup = manifests[0].parent / record["path"]
                    assert (
                        hashlib.sha256(backup.read_bytes()).hexdigest()
                        == record["before_sha256"]
                    )
        touched_roots.add(approved[resolved]["root_id"])
        original(path, content)
        if len(touched_roots) == 2:
            failed = True
            raise OSError("simulated replacement failure across both roots")

    monkeypatch.setattr(coordinate_repair, "_atomic_write", fail_after_crossing_roots)
    with pytest.raises(OSError, match="across both roots"):
        apply_stage_coordinate_repair(plan)
    assert touched_roots == {"root_0", "root_1"}
    after = _snapshot(case.root)
    assert all(after[path] == fingerprint for path, fingerprint in before.items())


@pytest.mark.parametrize("tampering", ("unknown-root", "escape", "array-metadata"))
def test_rehashed_split_plan_cannot_expand_its_write_whitelist(
    split_repair_case, tampering
):
    """A recomputed digest does not authorize another root or image-array writes."""
    case = split_repair_case
    plan = plan_stage_coordinate_repair(
        case.primary, additional_roots=(case.additional,)
    )
    if tampering == "unknown-root":
        plan["files"][0]["root_id"] = "unreviewed-root"
    elif tampering == "escape":
        plan["files"][0]["path"] = "../outside/zarr.json"
    else:
        plan["files"][0].update(
            root_id="root_1", path="repair_case_deskewed.ome.zarr/0/0/zarr.json"
        )
    plan["plan_sha256"] = coordinate_repair._plan_digest(plan)
    before = _snapshot(case.root)
    with pytest.raises(ValueError, match=r"root|Unsafe|non-metadata|escape"):
        apply_stage_coordinate_repair(plan)
    assert _snapshot(case.root) == before


def test_failed_restoration_does_not_skip_the_other_root(
    split_repair_case, monkeypatch
):
    """Continue rollback after one restore fails and report both recovery locations."""
    case = split_repair_case
    plan = plan_stage_coordinate_repair(
        case.primary, additional_roots=(case.additional,)
    )
    before = _snapshot(case.root)
    approved = {
        _split_record_path(plan, record).resolve(): record for record in plan["files"]
    }
    originals = {path: path.read_bytes() for path in approved}
    original_write = coordinate_repair._atomic_write
    touched_roots, restored_paths = set(), set()
    blocked_restore = None

    def fail_apply_and_one_restore(path, content):
        nonlocal blocked_restore
        path = Path(path).resolve()
        if path not in approved:
            return original_write(path, content)
        if blocked_restore is not None:
            if path == blocked_restore and content == originals[path]:
                raise OSError("simulated blocked restoration")
            original_write(path, content)
            if content == originals[path]:
                restored_paths.add(path)
            return None
        original_write(path, content)
        touched_roots.add(approved[path]["root_id"])
        if len(touched_roots) == 2:
            blocked_restore = path
            raise OSError("simulated apply failure after writing both roots")
        return None

    monkeypatch.setattr(coordinate_repair, "_atomic_write", fail_apply_and_one_restore)
    with pytest.raises(RuntimeError, match="complete rollback") as error:
        apply_stage_coordinate_repair(plan)
    assert blocked_restore is not None
    other_roots = set(plan["roots"]) - {approved[blocked_restore]["root_id"]}
    assert other_roots <= {approved[path]["root_id"] for path in restored_paths}
    after = _snapshot(case.root)
    blocked_relative = str(blocked_restore.relative_to(case.root))
    assert all(
        after[path] == fingerprint
        for path, fingerprint in before.items()
        if path != blocked_relative
    )
    assert (
        hashlib.sha256(blocked_restore.read_bytes()).hexdigest()
        == approved[blocked_restore]["after_sha256"]
    )
    assert "simulated blocked restoration" in str(error.value)
    normalized_error = str(error.value).replace("\\\\", "/").replace("\\", "/")
    backup_directories = {}
    for root_id, root_path in plan["roots"].items():
        manifest = next(
            (Path(root_path) / ".coordinate-repair-backups").glob("*/manifest.json")
        )
        assert manifest.parent.as_posix() in normalized_error
        backup_directories[root_id] = manifest.parent
    blocked_record = approved[blocked_restore]
    backup = backup_directories[blocked_record["root_id"]] / blocked_record["path"]
    assert backup.read_bytes() == originals[blocked_restore]
