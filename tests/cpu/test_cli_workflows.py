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
)
from opm_processing.dataio.processing_state import ProcessingState
from opm_processing.dataio.roi import projection_tile_footprints
from opm_processing.export_ome_tiff import load_export_metadata
from opm_processing.imageprocessing.tilefusion import TileFusion
from opm_processing.process import confirm_partial_output_resume

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("deconvolve", [False])
@pytest.mark.parametrize(
    "pipeline_case",
    [
        ("mirror", False, 0.4, None),
        ("stage", False, 1.2, None),
    ],
    indirect=True,
)
@pytest.mark.parametrize("copied", [False, True], ids=["moved", "copied"])
def test_inspection_and_exports_process_simulated_acquisition(
    pipeline_case,
    copied,
    invoke_cli,
    tmp_path,
    relocate_acquisition,
    monkeypatch,
):
    """Parse real commands and reopen pixels, physical coordinates and timestamps."""
    case = pipeline_case
    metadata = case.dataset.metadata
    expected_fused = open_image_array(case.output).read().result()
    source = relocate_acquisition(case.dataset.path.parent, copy=copied)
    assert case.dataset.path.exists() == copied
    dataset_path = source / case.dataset.path.name
    processed = source / case.processed.name
    state_path = source / "pipeline.processing.json"
    state = ProcessingState.read(state_path)
    previous_run = deepcopy(state.run(processed))
    previous_registration = deepcopy(state.registration(processed))
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
    output_dir = tmp_path / "rendered/nested" if copied else source
    invoke_cli(
        "fuse",
        source if copied else processed,
        output_dir=output_dir if copied else None,
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
        source if copied else dataset_path,
        output_dir=output_dir if copied else None,
        regenerate_max_z=True,
        max_workers=1,
    )
    assert not measurements
    fused_path = output_dir / "pipeline_fused.ome.zarr"
    maximum_path = output_dir / "pipeline_max_z_fused.ome.zarr"
    fused = open_image_array(fused_path).read().result()
    maximum = open_image_array(maximum_path).read().result()
    assert fused.dtype == expected_fused.dtype
    np.testing.assert_array_equal(fused, expected_fused)
    np.testing.assert_array_equal(maximum, fused.max(axis=2, keepdims=True))
    state = ProcessingState.read(state_path)
    registration = state.registration(processed)
    for key in ("configuration_sha256", "pairwise_metrics", "tiles"):
        assert registration[key] == previous_registration[key]
    assert state.registered_output_for_fused(fused_path) == processed
    assert state.registered_output_for_max_projection(maximum_path) == processed
    assert (
        len(projection_tile_footprints(maximum_path))
        == metadata.shape[0] * metadata.shape[1]
    )
    if copied:
        assert not (output_dir / state_path.name).exists()
        assert (source / "pipeline_depth_intensity_gains.json").is_file()
        assert not (output_dir / "pipeline_depth_intensity_gains.json").exists()
        _, provenance = load_export_metadata(fused_path)
        assert json.loads(provenance["Acquisition"])["path"] == str(dataset_path)
        for path in (fused_path, output_dir):
            report = json.loads(invoke_cli("inspect-opm", path).output)
            assert report["raw"]["path"] == str(dataset_path)
    assert not confirm_partial_output_resume(
        inspect_acquisition(dataset_path),
        output_dir=source,
        deconvolve=False,
        max_projection=True,
        resume=False,
    )
    report = json.loads(
        invoke_cli("inspect-opm", dataset_path if copied else source).output
    )
    manifest = report["raw"]
    assert manifest["shape"] == list(case.dataset.raw_data.shape)
    assert manifest["scan_axis_step_um"] == metadata.scan_axis_step_um
    assert manifest["camera_offset"] == metadata.camera_offset
    assert manifest["camera_conversion"] == metadata.camera_conversion

    destination = tmp_path / "ome-tiff"
    invoke_cli(
        "export-ome-tiff",
        fused_path,
        output=destination,
        workers=1 if copied else 2,
    )
    for suffix in ("fused", "max_z_fused"):
        image = output_dir / f"pipeline_{suffix}.ome.zarr"
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
