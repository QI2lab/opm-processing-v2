"""Project physical camera slabs while excluding peaks outside each local ROI."""

import json
from dataclasses import replace

import numpy as np
import pytest
import zarr
from scripts import ao_layer_projections
from scripts.ao_layer_projections import read_layer_projections
from yaozarrs import open_group

pytestmark = pytest.mark.unit


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
    for depth in (0, 1):
        np.testing.assert_array_equal(images[depth], case.expected[depth])
    native = open_group(case.fused)["0"].to_tensorstore().read().result()
    np.testing.assert_array_equal(native, original)
    np.testing.assert_array_equal(case.volume, original)


def test_layer_projection_rejects_non_native_canvas_before_reading(layer_case):
    """A thumbnail cannot silently determine native ROI pixel locations."""
    case = layer_case
    case.geometry.provenance["full_resolution"] = False
    with pytest.raises(ValueError, match="full-resolution tissue"):
        read_layer_projections(case.geometry, timepoint=0, channel=0, depths=[0])


def test_layer_projection_rejects_incompatible_xy_calibration(layer_case):
    """Native source samples must align exactly with the rendered canvas."""
    case = layer_case
    case.geometry = replace(case.geometry, origin_yx_um=(1, 0))
    with pytest.raises(ValueError, match="incompatible native XY"):
        read_layer_projections(case.geometry, timepoint=0, channel=0, depths=[0])


def test_layer_projection_validates_camera_physics_against_processed_slab(layer_case):
    """A wrong source Z shape cannot redefine the acquisition layer thickness."""
    case = layer_case
    zarr.open_array(str(case.collection.path / "0/0"), mode="r+").resize(
        (1, 1, 3, 3, 3)
    )
    with pytest.raises(ValueError, match="acquisition ROI height"):
        read_layer_projections(case.geometry, timepoint=0, channel=0, depths=[0])


def test_layer_projection_requires_owning_fused_registration(layer_case):
    """The maximum projection alone cannot supply a layer-specific background."""
    case = layer_case
    case.record.pop("fused_path")
    case.state.save()
    with pytest.raises(ValueError, match="completed full-volume registration"):
        read_layer_projections(case.geometry, timepoint=0, channel=0, depths=[0])


def test_projection_memmap_cache_matches_native_mips_and_skips_completed_reads(
    layer_case,
):
    """Completed native cache pixels are reused exactly without rereading volume data."""
    case = layer_case
    expected = case.expected
    images, records = read_layer_projections(
        case.geometry, timepoint=0, channel=0, depths=[0, 1], cache_dir=case.cache_dir
    )
    assert all(isinstance(image, np.memmap) for image in images.values())
    for depth in expected:
        np.testing.assert_array_equal(images[depth], expected[depth])
    resumed, resumed_records = read_layer_projections(
        case.geometry, timepoint=0, channel=0, depths=[0, 1], cache_dir=case.cache_dir
    )
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
    original = ao_layer_projections._write_cache_manifest

    def interrupt_checkpoint(path, manifest):
        if sum(manifest.get("completed_xy_blocks", [])) > 1:
            raise OSError("simulated source interruption")
        result = original(path, manifest)
        return result

    monkeypatch.setattr(
        ao_layer_projections, "_write_cache_manifest", interrupt_checkpoint
    )
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
    monkeypatch.setattr(ao_layer_projections, "_write_cache_manifest", original)
    resumed, records = read_layer_projections(
        case.geometry, timepoint=0, channel=0, depths=[0, 1], cache_dir=case.cache_dir
    )
    assert records[0]["resumed_xy_blocks"] > 0
    expected = case.expected
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
    with pytest.raises(ValueError, match="cache fingerprint"):
        read_layer_projections(
            case.geometry,
            timepoint=0,
            channel=0,
            depths=[0, 1],
            cache_dir=case.cache_dir,
        )


def test_projection_cache_rejects_truncated_native_array(layer_case):
    """Correct metadata cannot make a truncated output array safe to reuse."""
    case = layer_case
    read_layer_projections(
        case.geometry, timepoint=0, channel=0, depths=[0, 1], cache_dir=case.cache_dir
    )
    image_path = case.cache_dir / "layer_000.npy"
    with image_path.open("r+b") as stream:
        stream.truncate(image_path.stat().st_size - 2)
    with pytest.raises(ValueError, match="cache image"):
        read_layer_projections(
            case.geometry,
            timepoint=0,
            channel=0,
            depths=[0, 1],
            cache_dir=case.cache_dir,
        )


def test_projection_cache_allows_only_one_writer_and_releases_its_os_lock(tmp_path):
    """A second invocation cannot truncate arrays owned by an active writer."""
    with ao_layer_projections._projection_cache_lock(tmp_path):
        with pytest.raises(ValueError, match="already has an active writer"):
            with ao_layer_projections._projection_cache_lock(tmp_path):
                pytest.fail("A second writer acquired the active projection cache")
    with ao_layer_projections._projection_cache_lock(tmp_path):
        assert (tmp_path / ao_layer_projections.CACHE_LOCK_NAME).is_file()
