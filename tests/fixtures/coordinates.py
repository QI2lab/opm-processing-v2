"""Real image stores and prescribed trajectories for coordinate metadata units."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ome_types import from_xml, to_xml

from opm_processing.dataio.position_collection import create_position_collection
from opm_processing.dataio.processing_state import ProcessingState
from opm_processing.imageprocessing.coordinates import STAGE_COORDINATE_VERSION


def write_metadata_json(path, value):
    """Persist an explicitly prescribed metadata mutation for an error case."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), "utf-8")


@pytest.fixture
def stage_trajectory_metadata(acquisition_metadata):
    """Provide prescribed per-plane trajectories for isolated coordinate units."""

    def create(*, steps=(0.4,), leading=0, trailing=0):
        """Construct exact per-plane stage trajectories without accessing pixels."""
        starts = tuple(
            (102.123456789 + i * 100, -9.123456789, 70.0) for i in range(len(steps))
        )
        ends = tuple(
            (x + step * 10, y, z) for (x, y, z), step in zip(starts, steps, strict=True)
        )
        return replace(
            acquisition_metadata,
            path=Path("stage.ome.zarr"),
            storage_format="test",
            mode="stage",
            axes=("t", "p", "c", "z", "y", "x"),
            shape=(1, len(steps), 1, 11, 4, 4),
            array_paths=(),
            acquisition_order=("t", "p", "z", "c"),
            channels=(),
            stage_positions_zxy=tuple((z, x, y) for x, y, z in starts),
            scan_start_positions_xyz=starts,
            scan_end_positions_xyz=ends,
            scan_axis="x",
            scan_axis_step_um=0.4,
            pixel_size_um=0.115,
            angle_deg=30,
            camera_offset=100,
            camera_conversion=0.24,
            excess_scan_positions=0,
            excess_scan_start_positions=leading,
            excess_scan_end_positions=trailing,
            orientations=(),
            sidecar_paths=(),
        )

    return create


@pytest.fixture
def stage_coordinate_case(acquisition_factory, tmp_path):
    """Create two physical scan trajectories with a known ten-micrometer span."""
    positions = ((3.0, 100.0, 20.0), (3.0, 150.0, 20.0))
    dataset = acquisition_factory(
        np.zeros((1, 2, 1, 6, 4, 5), np.uint16),
        mode="stage",
        pixel_size_um=1.0,
        scan_axis_step_um=2.0,
        stage_positions_zxy=positions,
        scan_start_positions_xyz=((100.0, 20.0, 3.0), (150.0, 20.0, 3.0)),
        scan_end_positions_xyz=((110.0, 20.0, 3.0), (160.0, 20.0, 3.0)),
    )
    return SimpleNamespace(
        dataset=dataset,
        directory=tmp_path,
        positions=positions,
        marker={
            "version": STAGE_COORDINATE_VERSION,
            "placement_stage_positions_zxy_um": positions,
        },
    )


@pytest.fixture
def legacy_repair_case(acquisition_factory, fused_store_factory, tmp_path):
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
        write_metadata_json(group_path, group)
    fused = fused_store_factory(
        np.ones((1, 1, 2, 66, 8), np.uint16),
        name=f"{stem}_fused",
        spacing=(2, 1, 1),
        origin=(-3, -150, 20),
        factors=(1, 2),
    )
    maximum = fused_store_factory(
        np.ones((1, 1, 1, 66, 8), np.uint16),
        name=f"{stem}_max_z_fused",
        spacing=(2, 1, 1),
        level_origins=((-2, -150, 20), (-3, -150, 20)),
        factors=(1, 2),
    )
    tiles = [
        {"time_index": 0, "position_index": i, "origin_zyx_um": [-3, -x, 20]}
        for i, x in enumerate((100, 150))
    ]
    state = ProcessingState.create(tmp_path / f"{stem}.processing.json", dataset.path)
    for suffix in suffixes[:2]:
        processed = tmp_path / f"{stem}{suffix}"
        state.initialize_run(processed, configuration={}, overwrite=True)
        for position in range(2):
            state.complete_channel(processed, 0, position, 0)
            state.complete_tile(processed, 0, position)
        state.save_registration(
            processed,
            configuration={},
            pairwise_metrics={"0,1": {"shift": [0, 0, 0], "ssim": 0.9}},
        )
        # Preserve both measured placements while transferring fused ownership
        # to the deconvolved source through the actual state transition.
        state.complete_registration(processed, fused_path=fused, tiles=tiles)
    state.set_registered_max_projection(processed, max_projection_path=maximum)
    return tmp_path


@pytest.fixture
def split_repair_case(legacy_repair_case, fused_store_factory):
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
    owner = copy.deepcopy(journal["registration"][f"{stem}_deskewed.ome.zarr"])
    owner["fused_path"] = str(primary / f"{stem}_fused.ome.zarr")
    owner["max_projection_path"] = str(primary / f"{stem}_max_z_fused.ome.zarr")
    owner_key = str(additional / f"{stem}_deskewed.ome.zarr")
    journal["source"]["path"] = str(primary / f"{stem}.ome.zarr")
    journal["registration"] = {owner_key: owner}
    journal_path = additional / f"{stem}.processing.json"
    write_metadata_json(journal_path, journal)
    (root / f"{stem}.processing.json").unlink()

    # This older same-named G output has no journal owner and different geometry.
    orphan = additional / f"{stem}_fused.ome.zarr"
    historical = fused_store_factory(
        np.full((1, 1, 2, 73, 8), 7, np.uint16),
        name="historical_fused",
        spacing=(2, 1, 1),
        origin=(-3, 350, 20),
        factors=(1, 2),
    )
    historical.rename(orphan)
    return SimpleNamespace(
        root=root,
        roots={"root_0": primary, "root_1": additional},
        primary=primary,
        additional=additional,
        orphan=orphan,
        journal_path=journal_path,
        owner_key=owner_key,
        journal=journal,
    )
