"""Processing-component checks with controlled opm-v2 acquisition fixtures."""

import numpy as np
import pytest

from opm_processing.dataio.position_collection import (
    open_position_collection,
)
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)
from opm_processing.imageprocessing.camera import camera_correct
from opm_processing.process import format_processed_output
from tests.simulation.camera import digitize
from tests.simulation.metrics import camera_flux_accuracy, camera_noise_accuracy

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("save_float32", [False, True], ids=["uint16", "float32"])
@pytest.mark.parametrize(
    "separate_output", [False, True], ids=["source-dir", "output-dir"]
)
@pytest.mark.parametrize(
    "opm_v2_projection_zarr",
    [("projection", 1.0), ("stage", 1.0)],
    indirect=True,
    ids=["projection", "stage"],
)
def test_projection_output_preserves_calibration_geometry_and_provenance(
    opm_v2_projection_zarr, save_float32, separate_output, invoke_cli
):
    """Output dtype and destination preserve calibrated pixels and physical metadata."""
    fixture = opm_v2_projection_zarr
    parent = fixture.path.parent
    output_dir = parent / "nested/processed" if separate_output else parent
    invoke_cli(
        "process",
        fixture.path,
        save_float32=save_float32,
        output=output_dir if separate_output else None,
    )
    label = "2d" if fixture.metadata.mode == "stage" else "projection"
    output_path = output_dir / f"{fixture.path.stem}_{label}.ome.zarr"
    collection = open_position_collection(output_path)
    assert collection.shape == (2, 1, 2, 1, 16, 18)
    assert collection.channel_names == fixture.metadata.channel_names
    assert collection.voxel_size_um == (1.0, 0.115, 0.115)
    np.testing.assert_allclose(
        collection.stage_positions_zxy, fixture.metadata.stage_positions_zxy
    )
    state = ProcessingState.read(processing_state_path(output_dir, fixture.path.stem))
    assert state.document["source"]["path"] == str(fixture.path.resolve())
    expected = format_processed_output(
        camera_correct(
            fixture.raw_data[:, 0, :, 0],
            fixture.metadata.camera_offset,
            fixture.metadata.camera_conversion,
        ),
        save_float32,
    )
    actual = collection.arrays[0].read().result()
    assert actual.dtype == (np.float32 if save_float32 else np.uint16)
    np.testing.assert_array_equal(actual[:, :, 0], expected)
    truth = fixture.expectation[:, 0]
    aperture = np.ones(truth.shape, bool)
    accuracy = camera_noise_accuracy(actual, truth, fixture.microscope.camera, aperture)
    assert accuracy.accepted, accuracy
    assert camera_flux_accuracy(actual, truth, fixture.microscope.camera, aperture)
    assert not (parent / f"{fixture.path.stem}_stagefused.ome.zarr").exists()
    assert "opm_v2" not in collection.attributes
    if fixture.metadata.mode == "stage":
        assert not (output_dir / f"{fixture.path.stem}_projection.ome.zarr").exists()
    if separate_output:
        assert not (parent / output_path.name).exists()


@pytest.mark.parametrize("resume", [False, True])
def test_projection_cli_resume_preserves_completed_pixels(
    opm_v2_projection_zarr, resume, invoke_cli
):
    """Exercise the CLI flag through changed raw pixels and durable outputs."""
    fixture = opm_v2_projection_zarr
    invoke_cli("process", fixture.path)
    output_path = fixture.path.parent / f"{fixture.path.stem}_projection.ome.zarr"
    expected = format_processed_output(
        camera_correct(
            fixture.raw_data[:, 0, :, 0],
            fixture.metadata.camera_offset,
            fixture.metadata.camera_conversion,
        ),
        False,
    )[:, :, None]
    np.testing.assert_array_equal(
        open_position_collection(output_path).arrays[0].read().result(),
        expected,
    )
    from opm_processing.dataio.acquisition import open_acquisition_datastore

    changed_truth = 2 * fixture.expectation
    changed_raw = digitize(changed_truth, fixture.microscope.camera, seed=73)
    assert changed_raw.max() < np.iinfo(np.uint16).max
    open_acquisition_datastore(fixture.path).write(changed_raw).result()
    invoke_cli("process", fixture.path, resume=resume)
    after = open_position_collection(output_path).arrays[0].read().result()
    if not resume:
        expected = format_processed_output(
            camera_correct(
                changed_raw[:, 0, :, 0],
                fixture.metadata.camera_offset,
                fixture.metadata.camera_conversion,
            ),
            False,
        )[:, :, None]
    np.testing.assert_array_equal(after, expected)
    truth = (fixture.expectation if resume else changed_truth)[:, 0]
    aperture = np.ones(truth.shape, bool)
    accuracy = camera_noise_accuracy(after, truth, fixture.microscope.camera, aperture)
    assert accuracy.accepted, accuracy
    assert camera_flux_accuracy(after, truth, fixture.microscope.camera, aperture)
