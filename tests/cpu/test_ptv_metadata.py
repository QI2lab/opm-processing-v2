"""Self-contained processed PTV metadata through genuine producer writers."""

import copy
import json

import pytest
from yaozarrs import open_group

from opm_processing.dataio.position_collection import create_position_collection
from opm_processing.dataio.ptv import publish_ptv_metadata, read_processed_snapshot


def manifest(image_paths=("0",)):
    """Return explicit synthetic processing provenance, without implied calibration."""
    return {
        "schema_version": 1,
        "dataset_id": "synthetic-processed",
        "producer_version": "0.3.0",
        "product_kind": "deconvolved_deskewed",
        "processing_history": [{"operation": "synthetic_processed_fixture"}],
        "registration": {"applied": False, "coordinate_frame": "stored_processed"},
        "calibration": {"gain_offset_applied": False, "uncertainty_validated": False},
        "images": [
            {
                "path": name,
                "source_indices": [0, 1, 3],
                "volume_status_by_channel": [["valid", "invalid", "incomplete"]],
            }
            for name in image_paths
        ],
    }


def collection(tmp_path):
    """Allocate a tiny actual processed writer output; no numerical processing runs."""
    path = tmp_path / "processed.ome.zarr"
    create_position_collection(
        path,
        (3, 1, 1, 7, 12, 14),
        (0.23, 0.115, 0.15),
        spatial_offset_um=(-2.1, 3.2, -4.3),
        time_interval_s=0.085,
        multiscale_factors_yx=(2,),
        channels=["beads"],
        attributes={
            "opm_processing": {
                "timing": {
                    "schema_version": 1,
                    "n_acquired_planes": 17,
                    "dt_plane_s": 0.005,
                    "dt_volume_s": 0.085,
                    "source": "synthetic_trigger",
                }
            }
        },
    )
    return path


@pytest.mark.integration
def test_actual_writer_roundtrip_retains_geometry_ids_status_and_hardware(tmp_path):
    path = collection(tmp_path)
    publish_ptv_metadata(path, manifest())
    snapshot = read_processed_snapshot(path)
    metadata = snapshot.metadata
    assert snapshot.array.shape == (3, 1, 7, 12, 14)
    assert metadata["shape_zyx"] == (7, 12, 14)
    assert metadata["spacing_zyx_um"] == (0.23, 0.115, 0.15)
    assert metadata["origin_zyx_um"] == (-2.1, 3.2, -4.3)
    assert metadata["timing"]["n_acquired_planes"] == 17
    assert metadata["source_indices"] == [0, 1, 3]
    assert metadata["volume_status"] == ["valid", "invalid", "incomplete"]
    assert metadata["processing_history"] == manifest()["processing_history"]
    assert metadata["stored_ngff"]["channel_name"] == "beads"
    assert "opm_processing" in open_group(path).attrs


@pytest.mark.integration
def test_highest_resolution_resolved_without_literal_array_path(tmp_path):
    path = collection(tmp_path)
    metadata_path = path / "0/zarr.json"
    values = json.loads(metadata_path.read_text(encoding="utf-8"))
    values["attributes"]["ome"]["multiscales"][0]["datasets"][0]["path"] = (
        "native-resolution"
    )
    (path / "0/0").rename(path / "0/native-resolution")
    metadata_path.write_text(json.dumps(values), encoding="utf-8")
    publish_ptv_metadata(path, manifest())
    assert read_processed_snapshot(path).metadata["array_path"] == "0/native-resolution"


@pytest.mark.integration
@pytest.mark.parametrize(
    "problem",
    [
        "missing_unit",
        "missing_translation",
        "unitless_time",
        "wrong_cadence",
        "wrong_status_count",
        "projection_kind",
        "raw_kind",
    ],
)
def test_metadata_traps_are_rejected_before_reader_defaults(tmp_path, problem):
    path = collection(tmp_path)
    contract = manifest()
    metadata_path = path / "0/zarr.json"
    values = json.loads(metadata_path.read_text(encoding="utf-8"))
    scale = values["attributes"]["ome"]["multiscales"][0]
    if problem == "missing_unit":
        scale["axes"][2].pop("unit")
    elif problem == "missing_translation":
        for dataset in scale["datasets"]:
            dataset["coordinateTransformations"] = [
                t
                for t in dataset["coordinateTransformations"]
                if t["type"] != "translation"
            ]
    elif problem == "unitless_time":
        scale["axes"][0].pop("unit")
    elif problem == "wrong_cadence":
        for dataset in scale["datasets"]:
            dataset["coordinateTransformations"][0]["scale"][0] = 1
    elif problem == "wrong_status_count":
        contract["images"][0]["volume_status_by_channel"][0].pop()
    elif problem == "projection_kind":
        contract["product_kind"] = "max_z_deskewed"
    else:
        contract["product_kind"] = "raw_scan"
    metadata_path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError):
        publish_ptv_metadata(path, contract)
    assert "ptv" not in open_group(path).attrs["opm_processing"]


@pytest.mark.integration
def test_physical_unit_conversion_and_self_contained_fused_layout(
    tmp_path, fusion_operator
):
    from opm_processing.imageprocessing.tilefusion import TileFusion

    path = tmp_path / "fused.ome.zarr"
    operator = fusion_operator(
        TileFusion,
        time_interval_s=0.085,
        multiscale_factors=(2,),
        multiscale_downsample="stride",
        _pixel_size=(0.23, 0.115, 0.15),
        offset_um=(-2.1, 3.2, -4.3),
        padded_shape=(7, 12, 14),
        _is_2d=False,
        time_dim=3,
        channels=1,
        chunk_y=4,
        chunk_x=4,
    )
    operator._prepare_fused_image(path, z_slices_per_write=2)
    document_path = path / "zarr.json"
    values = json.loads(document_path.read_text(encoding="utf-8"))
    timing = open_group(collection(tmp_path)).attrs["opm_processing"]["timing"]
    values["attributes"]["opm_processing"] = {"timing": timing}
    scale = values["attributes"]["ome"]["multiscales"][0]
    scale["axes"][0]["unit"] = "millisecond"
    for axis in scale["axes"][2:]:
        axis["unit"] = "nanometer"
    for dataset in scale["datasets"]:
        dataset["coordinateTransformations"][0]["scale"] = [
            v * 1000 if i != 1 else v
            for i, v in enumerate(dataset["coordinateTransformations"][0]["scale"])
        ]
        dataset["coordinateTransformations"][1]["translation"][2:] = [
            v * 1000 for v in dataset["coordinateTransformations"][1]["translation"][2:]
        ]
    document_path.write_text(json.dumps(values), encoding="utf-8")
    contract = copy.deepcopy(manifest(("",)))
    contract["product_kind"] = "fused_volume"
    publish_ptv_metadata(path, contract)
    result = read_processed_snapshot(path)
    assert result.metadata["array_path"] == "0"
    assert result.metadata["spacing_zyx_um"] == pytest.approx((0.23, 0.115, 0.15))
    assert result.metadata["origin_zyx_um"] == pytest.approx((-2.1, 3.2, -4.3))
    assert result.metadata["stored_ngff"]["time_unit"] == "millisecond"


@pytest.mark.integration
def test_durable_journal_export_distinguishes_unwritten_zeroed_and_valid(tmp_path):
    from opm_processing.dataio.processing_state import ProcessingState
    from opm_processing.dataio.ptv import mark_ptv_tile, publish_processing_state_ptv

    path = collection(tmp_path)
    state = ProcessingState.create(
        tmp_path / "processing.json", tmp_path / "raw.ome.zarr"
    )
    configuration = {
        "kind": "deskewed",
        "calibration": {
            "gain_offset_applied": True,
            "camera_offset_adu": 100,
            "camera_conversion": 0.5,
        },
    }
    state.initialize_run(path, configuration=configuration, overwrite=True)
    publish_processing_state_ptv(path, state)
    initial = read_processed_snapshot(path)
    assert initial.metadata["volume_status"] == ["incomplete"] * 3
    state.complete_tile(path, 0, 0)
    mark_ptv_tile(path, 0, 0)
    state.complete_tile(path, 1, 0, zero_channels=(0,))
    mark_ptv_tile(path, 1, 0, zero_channels=(0,))
    snapshot = read_processed_snapshot(path)
    assert snapshot.metadata["volume_status"] == ["valid", "invalid", "incomplete"]
    assert snapshot.metadata["processing_history"][0]["configuration"] == configuration
    assert snapshot.metadata["calibration"] == configuration["calibration"]
    # Re-export from durable state after restart preserves dataset identity/status.
    publish_processing_state_ptv(path, ProcessingState.read(state.path))
    assert read_processed_snapshot(path).metadata == snapshot.metadata


@pytest.mark.integration
@pytest.mark.parametrize("contradict", [False, True])
def test_explicit_timestamps_use_hardware_cadence_with_source_gaps(
    tmp_path, contradict
):
    path = collection(tmp_path)
    target = path / "zarr.json"
    document = json.loads(target.read_text(encoding="utf-8"))
    document["attributes"]["opm_processing"]["timing"]["clock_precision_s"] = 1e-6
    target.write_text(json.dumps(document), encoding="utf-8")
    contract = manifest()
    contract["images"][0]["stored_timestamps_s"] = [
        4.0,
        4.085,
        4.085 if contradict else 4.255,
    ]
    if contradict:
        with pytest.raises(ValueError, match="timestamp"):
            publish_ptv_metadata(path, contract)
    else:
        publish_ptv_metadata(path, contract)
        assert read_processed_snapshot(path).metadata["stored_timestamps_s"] == [
            4.0,
            4.085,
            4.255,
        ]
