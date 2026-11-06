"""Repair approved metadata while proving all raw and image files stay intact."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from ome_types import from_xml

from opm_processing.dataio import coordinate_repair
from opm_processing.dataio.coordinate_repair import (
    apply_stage_coordinate_repair,
    plan_stage_coordinate_repair,
)
from opm_processing.imageprocessing.coordinates import STAGE_COORDINATE_VERSION
from tests.fixtures.coordinates import write_metadata_json

pytestmark = pytest.mark.unit


def test_metadata_repair_keeps_every_pixel_and_raw_file_unchanged(
    legacy_repair_case, stored_file_bytes
):
    """Only whitelisted metadata changes; every level retains its relative origin."""
    root = legacy_repair_case
    before = stored_file_bytes(root)
    plan = plan_stage_coordinate_repair(root)
    assert plan["delta_zyx_um"] == [0.5, -10, 0]
    paths = {Path(record["path"]) for record in plan["files"]}
    assert len(paths) == 19
    assert all(not path.parts[0] == "repair_case.ome.zarr" for path in paths)
    assert all(not path.as_posix().endswith("0/0/zarr.json") for path in paths)
    journal_before = json.loads((root / "repair_case.processing.json").read_text())
    result = apply_stage_coordinate_repair(plan)
    assert result["status"] == "applied"
    after = stored_file_bytes(root)
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
    for key, record in journal["registration"].items():
        assert (
            record["configuration_sha256"]
            == journal_before["registration"][key]["configuration_sha256"]
        )
        assert record["pairwise_metrics"] == {"0,1": {"shift": [0, 0, 0], "ssim": 0.9}}
        assert record["tiles"][0]["origin_zyx_um"] == [-2.5, -110, 20]
    assert (root / ".coordinate-repair-backups").is_dir()
    assert apply_stage_coordinate_repair(plan)["status"] == "already_repaired"
    assert plan_stage_coordinate_repair(root)["status"] == "already_repaired"


def test_preimage_mismatch_never_starts_a_repair(legacy_repair_case, stored_file_bytes):
    """A stale reviewed plan cannot overwrite later metadata changes."""
    root = legacy_repair_case
    plan = plan_stage_coordinate_repair(root)
    path = root / plan["files"][-1]["path"]
    path.write_bytes(path.read_bytes() + b"\n")
    before = stored_file_bytes(root)
    with pytest.raises(ValueError, match="preimage mismatch"):
        apply_stage_coordinate_repair(plan)
    assert stored_file_bytes(root) == before
    assert not (root / ".coordinate-repair-backups").exists()


@pytest.mark.parametrize("fail_after_replace", [False, True])
def test_interrupted_repair_rolls_back_approved_metadata(
    legacy_repair_case, monkeypatch, fail_after_replace, stored_file_bytes
):
    """A failed replacement restores all earlier metadata from exact preimages."""
    root = legacy_repair_case
    plan = plan_stage_coordinate_repair(root)
    before = stored_file_bytes(root)
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
    after = stored_file_bytes(root)
    assert all(after[path] == value for path, value in before.items())
    assert list((root / ".coordinate-repair-backups").glob("*/manifest.json"))


def test_mixed_coordinate_versions_are_rejected(legacy_repair_case, stored_file_bytes):
    """Never apply a common shift twice to one of several image artifacts."""
    root = legacy_repair_case
    path = root / "repair_case_fused.ome.zarr" / "zarr.json"
    group = json.loads(path.read_text())
    group["attributes"]["opm_processing_coordinates"] = {
        "version": STAGE_COORDINATE_VERSION
    }
    write_metadata_json(path, group)
    before = stored_file_bytes(root)
    with pytest.raises(ValueError, match="Mixed or unsupported"):
        plan_stage_coordinate_repair(root)
    assert stored_file_bytes(root) == before


def test_cropped_or_unequal_geometry_is_rejected(legacy_repair_case, stored_file_bytes):
    """Metadata-only repair must not conceal a tile-dependent placement change."""
    root = legacy_repair_case
    path = root / "repair_case_deskewed.ome.zarr" / "0" / "0" / "zarr.json"
    array = json.loads(path.read_text())
    array["shape"][-2] -= 1
    write_metadata_json(path, array)
    before = stored_file_bytes(root)
    with pytest.raises(ValueError, match="cropped or uses"):
        plan_stage_coordinate_repair(root)
    assert stored_file_bytes(root) == before


def test_tampered_plan_or_array_metadata_write_is_rejected(
    legacy_repair_case, stored_file_bytes
):
    """Even a rehashed plan cannot expand the metadata whitelist into image arrays."""
    root = legacy_repair_case
    plan = plan_stage_coordinate_repair(root)
    bad = copy.deepcopy(plan)
    bad["files"][0]["path"] = "repair_case_deskewed.ome.zarr/0/0/zarr.json"
    before = stored_file_bytes(root)
    with pytest.raises(ValueError, match="fingerprint"):
        apply_stage_coordinate_repair(bad)
    bad["plan_sha256"] = coordinate_repair._plan_digest(bad)
    with pytest.raises(ValueError, match="non-metadata path"):
        apply_stage_coordinate_repair(bad)
    assert stored_file_bytes(root) == before


def test_changed_read_dependency_does_not_touch_any_metadata(
    legacy_repair_case, stored_file_bytes
):
    """Geometry or source changes invalidate a reviewed plan before any backup."""
    root = legacy_repair_case
    plan = plan_stage_coordinate_repair(root)
    path = root / "repair_case_deskewed.ome.zarr" / "0" / "0" / "zarr.json"
    path.write_bytes(path.read_bytes() + b"\n")
    before = stored_file_bytes(root)
    with pytest.raises(ValueError, match="input dependency changed"):
        apply_stage_coordinate_repair(plan)
    assert stored_file_bytes(root) == before
    assert not (root / ".coordinate-repair-backups").exists()


def test_split_repair_changes_only_owned_available_artifacts(
    split_repair_case, stored_file_bytes
):
    """Repair split outputs once while preserving absolute paths and orphan pixels."""
    case = split_repair_case
    before = stored_file_bytes(case.root)
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
        (case.roots[record["root_id"]] / record["path"]).relative_to(case.root)
        for record in plan["files"]
    }
    assert all(
        not (case.roots[record["root_id"]] / record["path"]).is_relative_to(
            case.primary / "repair_case.ome.zarr"
        )
        for record in plan["files"]
    )
    assert all(
        not (case.roots[record["root_id"]] / record["path"]).is_relative_to(case.orphan)
        for record in plan["files"]
    )
    assert all("decon" not in record["path"] for record in plan["files"])
    assert "repair_case_fused.ome.zarr" in json.dumps(plan["skipped_artifacts"])
    result = apply_stage_coordinate_repair(plan)
    assert result["status"] == "applied"
    assert set(result["backup_directories"]) == {"root_0", "root_1"}
    after = stored_file_bytes(case.root)
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


def test_changed_external_dependency_refuses_the_entire_split_repair(
    split_repair_case, stored_file_bytes
):
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
    path = case.roots[dependency["root_id"]] / dependency["path"]
    path.write_bytes(path.read_bytes() + b"\n")
    before = stored_file_bytes(case.root)
    with pytest.raises(ValueError, match="dependency changed"):
        apply_stage_coordinate_repair(plan)
    assert stored_file_bytes(case.root) == before
    assert not (case.primary / ".coordinate-repair-backups").exists()
    assert not (case.additional / ".coordinate-repair-backups").exists()


def test_split_repair_backs_up_both_roots_before_writing_and_rolls_back(
    split_repair_case, monkeypatch, stored_file_bytes
):
    """A failure after replacing metadata on both roots restores both preimages."""
    case = split_repair_case
    plan = plan_stage_coordinate_repair(
        case.primary, additional_roots=(case.additional,)
    )
    before = stored_file_bytes(case.root)
    approved = {
        (case.roots[record["root_id"]] / record["path"]).resolve(): record
        for record in plan["files"]
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
    after = stored_file_bytes(case.root)
    assert all(after[path] == fingerprint for path, fingerprint in before.items())


@pytest.mark.parametrize("tampering", ("unknown-root", "escape", "array-metadata"))
def test_rehashed_split_plan_cannot_expand_its_write_whitelist(
    split_repair_case, tampering, stored_file_bytes
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
    before = stored_file_bytes(case.root)
    with pytest.raises(ValueError, match=r"root|Unsafe|non-metadata|escape"):
        apply_stage_coordinate_repair(plan)
    assert stored_file_bytes(case.root) == before


def test_failed_restoration_does_not_skip_the_other_root(
    split_repair_case, monkeypatch, stored_file_bytes
):
    """Continue rollback after one restore fails and report both recovery locations."""
    case = split_repair_case
    plan = plan_stage_coordinate_repair(
        case.primary, additional_roots=(case.additional,)
    )
    before = stored_file_bytes(case.root)
    approved = {
        (case.roots[record["root_id"]] / record["path"]).resolve(): record
        for record in plan["files"]
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
    after = stored_file_bytes(case.root)
    blocked_relative = blocked_restore.relative_to(case.root)
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
