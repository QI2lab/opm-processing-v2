"""Registered commands process the same simulated ground-truth acquisition."""

import json
from copy import deepcopy

import numpy as np
import pytest
import tifffile
from ome_types import from_xml

from opm_processing.dataio.acquisition import inspect_acquisition
from opm_processing.dataio.position_collection import (
    open_image_array,
    open_position_collection,
)
from opm_processing.dataio.processing_state import ProcessingState
from opm_processing.dataio.roi import projection_tile_footprints
from opm_processing.export_ome_tiff import load_export_metadata
from opm_processing.imageprocessing.tilefusion import TileFusion
from opm_processing.process import confirm_partial_output_resume

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("deconvolve", [False])
@pytest.mark.parametrize("pipeline_case", [("stage", False, 0.4, None)], indirect=True)
def test_fuse_output_dir_keeps_registration_with_processed(
    pipeline_case, invoke_cli, tmp_path, monkeypatch
):
    """Render elsewhere from cached registration, preserving pixels and provenance."""
    case = pipeline_case
    source = case.processed.parent
    destination = tmp_path.parent / f"{tmp_path.name}-rendered" / "nested"
    state_path = source / "pipeline.processing.json"
    before = ProcessingState.read(state_path).registration(case.processed)
    expected = open_image_array(case.output).read().result()
    measurements = []
    measure = TileFusion.register_and_score

    def observe_measurement(*args, **kwargs):
        result = measure(*args, **kwargs)
        measurements.append(result)
        return result

    monkeypatch.setattr(
        TileFusion, "register_and_score", staticmethod(observe_measurement)
    )
    invoke_cli(
        "fuse",
        case.processed,
        output_dir=destination,
        blend_pixels=(0, 0, 0),
        downsample_factors=(1, 1, 1),
        registration_channel=case.dataset.metadata.shape[2] - 1,
        ssim_window=3,
        registration_threshold=0.1,
        max_registration_shift_zyx=(2, 2, 2),
        chunk_shape_yx=(16, 16),
        max_workers=1,
        normalize_depth_intensity=False,
    )
    assert not measurements
    fused = destination / case.output.name
    maximum = destination / "pipeline_max_z_fused.ome.zarr"
    np.testing.assert_array_equal(open_image_array(fused).read().result(), expected)
    np.testing.assert_array_equal(
        open_image_array(maximum).read().result(), expected.max(axis=2, keepdims=True)
    )
    assert not (destination / state_path.name).exists()
    assert (source / "pipeline_depth_intensity_gains.json").is_file()
    assert not (destination / "pipeline_depth_intensity_gains.json").exists()
    state = ProcessingState.read(state_path)
    after = state.registration(case.processed)
    assert after["configuration_sha256"] == before["configuration_sha256"]
    assert after["pairwise_metrics"] == before["pairwise_metrics"]
    assert after["tiles"] == before["tiles"]
    assert state.registered_output_for_fused(fused) == case.processed
    assert state.registered_output_for_max_projection(maximum) == case.processed
    footprints = projection_tile_footprints(maximum)
    assert (
        len(footprints)
        == case.dataset.metadata.shape[0] * case.dataset.metadata.shape[1]
    )
    _, provenance = load_export_metadata(fused)
    assert json.loads(provenance["Acquisition"])["path"] == str(case.dataset.path)
    inspection = json.loads(invoke_cli("inspect-opm", fused).output)
    assert inspection["raw"]["path"] == str(case.dataset.path)
    inspection = json.loads(invoke_cli("inspect-opm", destination).output)
    assert inspection["raw"]["path"] == str(case.dataset.path)
    invoke_cli(
        "fuse",
        case.processed,
        output_dir=destination,
        regenerate_max_z=True,
        max_workers=1,
    )
    np.testing.assert_array_equal(
        open_image_array(maximum).read().result(), expected.max(axis=2, keepdims=True)
    )


@pytest.mark.parametrize("deconvolve", [False])
@pytest.mark.parametrize(
    "pipeline_case",
    [
        ("mirror", False, 0.4, None),
        ("stage", False, 1.2, None),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "cli_export_options", ["all-gray", "selected-color"], indirect=True
)
def test_inspection_and_exports_process_simulated_acquisition(
    pipeline_case,
    cli_export_options,
    invoke_cli,
    tmp_path,
    relocate_acquisition,
    monkeypatch,
):
    """Parse real commands and reopen pixels, physical coordinates and timestamps."""
    case = pipeline_case
    options = cli_export_options
    metadata = case.dataset.metadata
    source = relocate_acquisition(
        case.dataset.path.parent, copy=not options["depth_color"]
    )
    assert case.dataset.path.exists() == (not options["depth_color"])
    dataset_path = source / case.dataset.path.name
    processed = source / case.processed.name
    state_path = source / "pipeline.processing.json"
    state = ProcessingState.read(state_path)
    previous_run = deepcopy(state.run(processed))
    invoke_cli("process", dataset_path, resume=True, **case.options)
    state = ProcessingState.read(state_path)
    assert state.run(processed) == previous_run
    assert state.source_path() == dataset_path
    measurements = []
    measure = TileFusion.register_and_score

    def observe_measurement(*args, **kwargs):
        """Retain actual measurements if a compatible fusion cache is bypassed."""
        result = measure(*args, **kwargs)
        measurements.append(result)
        return result

    monkeypatch.setattr(
        TileFusion, "register_and_score", staticmethod(observe_measurement)
    )
    invoke_cli(
        "fuse",
        processed if options["depth_color"] else source,
        blend_pixels=(0, 0, 0),
        downsample_factors=(1, 1, 1),
        registration_channel=metadata.shape[2] - 1,
        ssim_window=3,
        registration_threshold=0.1,
        max_registration_shift_zyx=(2, 2, 2),
        chunk_shape_yx=(16, 16),
        max_workers=1,
        normalize_depth_intensity=False,
    )
    invoke_cli(
        "fuse",
        dataset_path if options["depth_color"] else source,
        regenerate_max_z=True,
        max_workers=1,
    )
    assert not measurements
    fused = open_image_array(source / "pipeline_fused.ome.zarr").read().result()
    maximum = open_image_array(source / "pipeline_max_z_fused.ome.zarr").read().result()
    np.testing.assert_array_equal(maximum, fused.max(axis=2, keepdims=True))
    collection = open_position_collection(processed)
    assert not confirm_partial_output_resume(
        inspect_acquisition(dataset_path),
        output_dir=source,
        deconvolve=False,
        max_projection=True,
        resume=False,
    )
    report = json.loads(
        invoke_cli(
            "inspect-opm", source if options["depth_color"] else dataset_path
        ).output
    )
    manifest = report["raw"]
    assert manifest["shape"] == list(case.dataset.raw_data.shape)
    assert manifest["scan_axis_step_um"] == metadata.scan_axis_step_um
    assert manifest["camera_offset"] == metadata.camera_offset
    assert manifest["camera_conversion"] == metadata.camera_conversion

    destination = tmp_path / "ome-tiff"
    invoke_cli(
        "export-ome-tiff",
        source / case.output.name,
        output=destination,
        workers=options["workers"],
    )
    for suffix in ("fused", "max_z_fused"):
        image = source / f"pipeline_{suffix}.ome.zarr"
        expected = open_image_array(image).read().result()
        with tifffile.TiffFile(destination / f"pipeline_{suffix}.ome.tif") as tif:
            np.testing.assert_array_equal(
                tif.asarray().reshape(expected.shape), expected
            )
            ome = from_xml(tif.ome_metadata, validate=True)
            pixels = ome.images[0].pixels
            annotations = ome.structured_annotations.map_annotations[-1].value
            assert json.loads(annotations["Acquisition"])["path"] == str(dataset_path)
            assert (pixels.size_t, pixels.size_c, pixels.size_z) == expected.shape[:3]
            assert pixels.time_increment == pytest.approx(
                case.collection.time_interval_s
            )

    invoke_cli(
        "export-projections",
        source,
        acquisition=dataset_path if not options["depth_color"] else None,
        **options,
    )
    files = sorted((source / "projection_frames").rglob("*.tiff"))
    times = range(*(options["timepoints"] or (0, metadata.shape[0])))
    assert len(files) == len(times) * metadata.shape[1] * metadata.shape[2]
    for path in files:
        with tifffile.TiffFile(path) as tif:
            saved = tif.shaped_metadata[0]
            position, channel = saved["position"], saved["channel"]
            reference = collection.arrays[position][0, channel].read().result()
            np.testing.assert_allclose(
                saved["contrast_limits"], np.percentile(reference, (0.001, 99.999))
            )
            assert saved["timepoint"] in times
            assert saved["elapsed_ms"] == pytest.approx(
                saved["timepoint"] * metadata.volume_interval_s * 1000
            )
            assert saved["source_voxel_size_zyx_um"] == list(
                case.collection.voxel_size_um
            )
            assert tif.pages[0].photometric.name == (
                "RGB" if options["depth_color"] else "MINISBLACK"
            )
            assert np.any(tif.asarray() > 0)
