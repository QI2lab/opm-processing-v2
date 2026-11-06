"""Project physical camera slabs while excluding peaks outside each local ROI."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scripts import ao_layer_projections
from scripts.ao_layer_projections import read_layer_projections

pytestmark = pytest.mark.unit


@pytest.fixture
def layer_case(monkeypatch, tmp_path):
    """Define two sloped registered layers with clipped source camera slabs."""
    volume = np.zeros((1, 1, 8, 5, 7), dtype=np.uint16)
    volume[0, 0, 2, 0, 0] = 10
    volume[0, 0, 3, 0, 0] = 20
    volume[0, 0, 0, 0, 0] = 30
    volume[0, 0, 5, 0, 0] = 999  # Inside global Z union, outside this XY slab.
    volume[0, 0, 5, 3, 5] = 40
    volume[0, 0, 3, 3, 5] = 25
    volume[0, 0, :, 4, 0] = 1000  # Outside every tile's XY footprint.
    calls = []
    fused = tmp_path / "fused.ome.zarr"
    processed = tmp_path / "processed.ome.zarr"
    raw = tmp_path / "raw.ome.zarr"
    preview = tmp_path / "max_z_fused.ome.zarr"
    state_path = tmp_path / "state.json"

    class Array:
        shape = volume.shape
        dtype = np.dtype(np.uint16)

        def to_tensorstore(self):
            return self

        def __getitem__(self, selection):
            calls.append(selection)
            return SimpleNamespace(
                read=lambda: SimpleNamespace(result=lambda: volume[selection].copy())
            )

    class Group(dict):
        def __init__(self):
            super().__init__({"0": Array()})
            self.attrs = {
                "ome": {
                    "multiscales": [
                        {
                            "axes": [
                                {
                                    "name": axis,
                                    **({"unit": "micrometer"} if axis in "zyx" else {}),
                                }
                                for axis in "tczyx"
                            ],
                            "datasets": [
                                {
                                    "path": "0",
                                    "coordinateTransformations": [
                                        {"type": "scale", "scale": [1, 1, 1, 1, 1]},
                                        {
                                            "type": "translation",
                                            "translation": [0, 0, -20, 0, 0],
                                        },
                                    ],
                                }
                            ],
                        }
                    ]
                }
            }

    tiles = [
        {"time_index": 0, "position_index": i, "origin_zyx_um": origin}
        for i, origin in enumerate(
            ((-18, 0, 0), (-16, 3, 5), (-21, -1, -1), (-17, 3, 5))
        )
    ]
    record = {"fused_path": str(fused), "tiles": tiles}
    state = SimpleNamespace(
        path=state_path,
        registered_output_for_max_projection=lambda _: processed,
        registration=lambda _: record,
        roi_series=lambda _: (),
        source_path=lambda: raw,
    )
    acquisition = SimpleNamespace(
        path=raw,
        mode="stage",
        stage_axis_flips_xyz=(False, False, False),
        pixel_size_um=1,
        angle_deg=30,
        index_sizes={"y": 4, "x": 3},
        scan_position_count=5,
        scan_axis_step_um=1,
        excess_scan_positions=0,
        excess_scan_start_positions=0,
        excess_scan_end_positions=0,
        stage_positions_zxy=((10, 100, 0), (10, 100, 4), (12, 100, 0), (12, 100, 4)),
    )
    collection = SimpleNamespace(
        voxel_size_um=(1, 1, 1),
        arrays=[SimpleNamespace(shape=(1, 1, 2, 3, 3)) for _ in range(4)],
        spatial_origins_zyx_um=((0, -4, 0),) * 4,
    )
    geometry = SimpleNamespace(
        shape_yx=(5, 7),
        scale_yx_um=(1, 1),
        origin_yx_um=(0, 0),
        provenance={
            "full_resolution": True,
            "projection": str(preview),
            "processing_state": str(state_path),
        },
    )
    monkeypatch.setattr(ao_layer_projections.ProcessingState, "read", lambda _: state)
    monkeypatch.setattr(ao_layer_projections, "resolve_store_path", lambda *_: fused)
    monkeypatch.setattr(ao_layer_projections, "open_group", lambda _: Group())
    monkeypatch.setattr(
        ao_layer_projections, "inspect_acquisition", lambda _: acquisition
    )
    monkeypatch.setattr(
        ao_layer_projections, "open_position_collection", lambda _: collection
    )
    monkeypatch.setattr(ao_layer_projections, "XY_BLOCK_SHAPE", (2, 3))
    metadata_files = {
        fused / "zarr.json": {"attributes": Group().attrs},
        fused / "0" / "zarr.json": {"shape": list(volume.shape), "data_type": "uint16"},
        preview / "zarr.json": {"projection": "maximum"},
        preview / "0" / "zarr.json": {"shape": [1, 1, 1, 5, 7]},
        processed / "zarr.json": {"collection": "source camera slabs"},
        raw / "zarr.json": {"camera_height": 4, "pixel_size_um": 1, "angle_deg": 30},
        state_path: {"registration": record},
    }
    for position in range(4):
        metadata_files[processed / str(position) / "zarr.json"] = {
            "attributes": {"ome": {"multiscales": [{"datasets": [{"path": "0"}]}]}}
        }
        metadata_files[processed / str(position) / "0" / "zarr.json"] = {
            "shape": [1, 1, 2, 3, 3]
        }
    for path, document in metadata_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document), "utf-8")
    return SimpleNamespace(
        geometry=geometry,
        volume=volume,
        calls=calls,
        collection=collection,
        record=record,
        array_type=Array,
        fused=fused,
        cache_dir=tmp_path / "projection-cache",
    )


def test_layer_mips_use_physical_slabs_mapped_into_local_registered_rois(layer_case):
    """A distant axial peak cannot contaminate a thinner local camera slab."""
    case = layer_case
    original = case.volume.copy()
    images, provenance = read_layer_projections(
        case.geometry, timepoint=0, channel=0, depths=[0, 1]
    )
    assert images[0].shape == images[1].shape == (5, 7)
    assert images[0].dtype == np.dtype(np.uint16)
    assert images[0][0, 0] == 20  # Excludes the bright Z=5 voxel at this XY.
    assert images[1][0, 0] == 30
    assert images[0][3, 5] == 40
    assert images[1][3, 5] == 25
    assert images[0][4, 0] == images[1][4, 0] == 0
    assert provenance[0]["camera_roi_axial_span_um"] == 2
    assert provenance[0]["source_slab_z_planes"] == 2
    assert provenance[0]["aggregate_z_start_stop"] == [2, 6]
    first = provenance[0]["tile_rois"][0]
    assert first["raw_stage_z_um"] == 10
    assert first["nominal_acquisition_origin_zyx_um"][0] == -10
    assert first["registration_translation_zyx_um"][0] == -8
    assert first["registered_voxel_box_zyx"] == [[2, 4], [0, 3], [0, 3]]
    clipped = provenance[1]["tile_rois"][0]
    assert clipped["unclipped_registered_voxel_box_zyx"] == [[-1, 1], [-1, 2], [-1, 2]]
    assert clipped["registered_voxel_box_zyx"] == [[0, 1], [0, 2], [0, 2]]
    assert provenance[0]["mip_sha256"] != provenance[1]["mip_sha256"]
    # Every occupied XY block is read once, sharing that read across layers.
    xy_reads = [
        (item[3].start, item[3].stop, item[4].start, item[4].stop)
        for item in case.calls
    ]
    assert xy_reads and len(xy_reads) == len(set(xy_reads))
    assert all(item[2].start >= 0 and item[2].stop <= 8 for item in case.calls)
    assert all(y1 <= 5 and x1 <= 7 for _, y1, _, x1 in xy_reads)
    np.testing.assert_array_equal(case.volume, original)


def test_layer_projection_rejects_non_native_canvas_before_reading(layer_case):
    """A thumbnail cannot silently determine native ROI pixel locations."""
    case = layer_case
    case.geometry.provenance["full_resolution"] = False
    with pytest.raises(ValueError, match="full-resolution tissue"):
        read_layer_projections(case.geometry, timepoint=0, channel=0, depths=[0])
    assert not case.calls


def test_layer_projection_rejects_incompatible_xy_calibration(layer_case):
    """Native source samples must align exactly with the rendered canvas."""
    case = layer_case
    case.geometry.origin_yx_um = (1, 0)
    with pytest.raises(ValueError, match="incompatible native XY"):
        read_layer_projections(case.geometry, timepoint=0, channel=0, depths=[0])
    assert not case.calls


def test_layer_projection_validates_camera_physics_against_processed_slab(layer_case):
    """A wrong source Z shape cannot redefine the acquisition layer thickness."""
    case = layer_case
    case.collection.arrays[0].shape = (1, 1, 3, 3, 3)
    with pytest.raises(ValueError, match="acquisition ROI height"):
        read_layer_projections(case.geometry, timepoint=0, channel=0, depths=[0])
    assert not case.calls


def test_layer_projection_requires_owning_fused_registration(layer_case):
    """The maximum projection alone cannot supply a layer-specific background."""
    case = layer_case
    case.record.pop("fused_path")
    with pytest.raises(ValueError, match="completed full-volume registration"):
        read_layer_projections(case.geometry, timepoint=0, channel=0, depths=[0])
    assert not case.calls


def test_projection_memmap_cache_matches_native_mips_and_skips_completed_reads(
    layer_case,
):
    """Completed native cache pixels are reused exactly without rereading volume data."""
    case = layer_case
    expected, _ = read_layer_projections(
        case.geometry, timepoint=0, channel=0, depths=[0, 1]
    )
    images, records = read_layer_projections(
        case.geometry, timepoint=0, channel=0, depths=[0, 1], cache_dir=case.cache_dir
    )
    assert all(isinstance(image, np.memmap) for image in images.values())
    for depth in expected:
        np.testing.assert_array_equal(images[depth], expected[depth])
    calls = len(case.calls)
    resumed, resumed_records = read_layer_projections(
        case.geometry, timepoint=0, channel=0, depths=[0, 1], cache_dir=case.cache_dir
    )
    assert len(case.calls) == calls
    for depth in expected:
        np.testing.assert_array_equal(resumed[depth], expected[depth])
        assert records[depth]["mip_sha256"] == resumed_records[depth]["mip_sha256"]
        assert resumed_records[depth]["resumed_xy_blocks"] == 9
    manifest = json.loads((case.cache_dir / "layer_projection_cache.json").read_text())
    assert manifest["phase"] == "complete"
    assert all(manifest["completed_xy_blocks"])


def test_projection_cache_resumes_after_interruption_with_identical_mips(
    layer_case, monkeypatch
):
    """Only flushed blocks survive interruption; unfinished pixels are recomputed."""
    case = layer_case
    monkeypatch.setattr(ao_layer_projections, "READ_WORKERS", 1)
    monkeypatch.setattr(ao_layer_projections, "CACHE_CHECKPOINT_BLOCKS", 1)
    original = case.array_type.__getitem__

    def interrupt_after_one_read(array, selection):
        if case.calls:
            raise OSError("simulated source interruption")
        return original(array, selection)

    monkeypatch.setattr(case.array_type, "__getitem__", interrupt_after_one_read)
    with pytest.raises(OSError, match="simulated source interruption"):
        read_layer_projections(
            case.geometry,
            timepoint=0,
            channel=0,
            depths=[0, 1],
            cache_dir=case.cache_dir,
        )
    manifest = json.loads((case.cache_dir / "layer_projection_cache.json").read_text())
    assert any(manifest["completed_xy_blocks"])
    assert not all(manifest["completed_xy_blocks"])
    monkeypatch.setattr(case.array_type, "__getitem__", original)
    case.calls.clear()
    resumed, records = read_layer_projections(
        case.geometry, timepoint=0, channel=0, depths=[0, 1], cache_dir=case.cache_dir
    )
    first_xy = (0, 2, 0, 3)
    resumed_reads = [
        (item[3].start, item[3].stop, item[4].start, item[4].stop)
        for item in case.calls
    ]
    assert first_xy not in resumed_reads
    assert records[0]["resumed_xy_blocks"] > 0
    expected, _ = read_layer_projections(
        case.geometry, timepoint=0, channel=0, depths=[0, 1]
    )
    for depth in expected:
        np.testing.assert_array_equal(resumed[depth], expected[depth])


def test_projection_cache_rejects_changed_source_metadata_without_rereading_pixels(
    layer_case,
):
    """A different native array description invalidates all saved completion bits."""
    case = layer_case
    read_layer_projections(
        case.geometry, timepoint=0, channel=0, depths=[0, 1], cache_dir=case.cache_dir
    )
    metadata = case.fused / "0" / "zarr.json"
    metadata.write_bytes(metadata.read_bytes() + b"\n")
    case.calls.clear()
    with pytest.raises(ValueError, match="cache fingerprint"):
        read_layer_projections(
            case.geometry,
            timepoint=0,
            channel=0,
            depths=[0, 1],
            cache_dir=case.cache_dir,
        )
    assert not case.calls


def test_projection_cache_rejects_truncated_native_array(layer_case):
    """Correct metadata cannot make a truncated output array safe to reuse."""
    case = layer_case
    read_layer_projections(
        case.geometry, timepoint=0, channel=0, depths=[0, 1], cache_dir=case.cache_dir
    )
    image_path = case.cache_dir / "layer_000.npy"
    with image_path.open("r+b") as stream:
        stream.truncate(image_path.stat().st_size - 2)
    case.calls.clear()
    with pytest.raises(ValueError, match="cache image"):
        read_layer_projections(
            case.geometry,
            timepoint=0,
            channel=0,
            depths=[0, 1],
            cache_dir=case.cache_dir,
        )
    assert not case.calls


def test_projection_cache_allows_only_one_writer_and_releases_its_os_lock(tmp_path):
    """A second invocation cannot truncate arrays owned by an active writer."""
    with ao_layer_projections._projection_cache_lock(tmp_path):
        with pytest.raises(ValueError, match="already has an active writer"):
            with ao_layer_projections._projection_cache_lock(tmp_path):
                pytest.fail("A second writer acquired the active projection cache")
    with ao_layer_projections._projection_cache_lock(tmp_path):
        assert (tmp_path / ao_layer_projections.CACHE_LOCK_NAME).is_file()
