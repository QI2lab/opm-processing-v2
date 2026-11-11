"""Processing-component checks with controlled opm-v2 acquisition fixtures."""

import numpy as np
import pytest

from opm_processing.dataio.position_collection import (
    open_image_array,
    open_position_collection,
)
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)
from opm_processing.imageprocessing.camera import camera_correct
from opm_processing.process import format_processed_output
from tests.simulation.camera import digitize

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "opm_v2_projection_zarr,save_float32,separate_output",
    [
        pytest.param(
            ("projection", 1.0), False, False, id="projection-source-dir-uint16"
        ),
        pytest.param(
            ("projection", 1.0), True, True, id="projection-output-dir-float32"
        ),
        pytest.param(("stage", 1.0), False, True, id="stage-output-dir-uint16"),
        pytest.param(("stage", 1.0), True, False, id="stage-source-dir-float32"),
        pytest.param(
            ("projection", 1.0, 3),
            False,
            False,
            id="selected-positions-source-dir-uint16",
        ),
        pytest.param(
            ("projection", 1.0, 3),
            True,
            True,
            id="selected-positions-output-dir-float32",
        ),
    ],
    indirect=["opm_v2_projection_zarr"],
)
def test_projection_output_preserves_calibration_geometry_and_provenance(
    opm_v2_projection_zarr, save_float32, separate_output, invoke_cli
):
    """Output dtype and destination preserve calibrated pixels and physical metadata."""
    fixture = opm_v2_projection_zarr
    parent = fixture.path.parent
    output_dir = parent / "nested/processed" if separate_output else parent
    selected = (1, 3) if fixture.metadata.tile_count > 1 else None
    positions = range(*(selected or (0, fixture.metadata.tile_count)))
    invoke_cli(
        "process",
        fixture.path,
        save_float32=save_float32,
        output=output_dir if separate_output else None,
        pos_range=selected,
    )
    label = "2d" if fixture.metadata.mode == "stage" else "projection"
    output_path = output_dir / f"{fixture.path.stem}_{label}.ome.zarr"
    collection = open_position_collection(output_path)
    assert collection.shape == fixture.raw_data.shape
    assert collection.channel_names == fixture.metadata.channel_names
    assert collection.voxel_size_um == (1.0, 0.115, 0.115)
    np.testing.assert_allclose(
        collection.stage_positions_zxy, fixture.metadata.stage_positions_zxy
    )
    state = ProcessingState.read(processing_state_path(output_dir, fixture.path.stem))
    assert state.document["source"]["path"] == str(fixture.path.resolve())
    assert state.completed_tiles(output_path) == {
        (t, p) for t in range(2) for p in positions
    }
    assert state.run(output_path)["configuration"]["selection"]["position"] == (
        None if selected is None else list(selected)
    )
    for position, array in enumerate(collection.arrays):
        actual = array.read().result()
        assert actual.dtype == (np.float32 if save_float32 else np.uint16)
        if position not in positions:
            assert not np.any(actual)
            continue
        expected = format_processed_output(
            camera_correct(
                fixture.raw_data[:, position, :, 0],
                fixture.metadata.camera_offset,
                fixture.metadata.camera_conversion,
            ),
            save_float32,
        )
        np.testing.assert_array_equal(actual[:, :, 0], expected)
    fused_path = output_dir / f"{fixture.path.stem}_stagefused.ome.zarr"
    if selected is None:
        assert not fused_path.exists()
    else:
        from yaozarrs import open_group

        height, width = fixture.raw_data.shape[-2:]
        stride = width + 2
        actual = open_image_array(fused_path).read().result()
        assert actual.shape[:4] == (2, 2, 1, height)
        # The production canvas pads the physical footprint to eight pixels.
        assert actual.shape[-1] % 8 == 0
        assert 0 <= actual.shape[-1] - (width + stride) < 8
        expected = np.zeros_like(actual)
        for position, start in zip(positions, (0, stride), strict=True):
            expected[..., start : start + width] = format_processed_output(
                camera_correct(
                    fixture.raw_data[:, position],
                    fixture.metadata.camera_offset,
                    fixture.metadata.camera_conversion,
                ),
                save_float32,
            )
        eps = np.finfo(np.float32).eps
        np.testing.assert_allclose(
            actual, expected, rtol=2 * eps / (1 - 2 * eps), atol=0
        )
        provenance = open_group(fused_path).attrs["opm_processing_projection"]
        assert provenance["source_collection"] == output_path.name
        assert provenance["source_position_indices"] == list(positions)
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
