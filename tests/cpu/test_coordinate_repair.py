"""Repair approved metadata while proving all raw and image files stay intact."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
from ome_types import from_xml, to_xml

from opm_processing.dataio.coordinate_repair import (
    plan_stage_coordinate_repair,
)
from opm_processing.dataio.position_collection import create_position_collection

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
    """Planning lists only approved metadata and leaves every original byte unchanged."""
    root = legacy_repair_case
    before = _snapshot(root)
    plan = plan_stage_coordinate_repair(root)
    assert plan["delta_zyx_um"] == [0.5, -10, 0]
    paths = {record["path"] for record in plan["files"]}
    assert len(paths) == 19
    assert all(not path.startswith("repair_case.ome.zarr") for path in paths)
    assert all(not path.endswith("0/0/zarr.json") for path in paths)
    assert _snapshot(root) == before
