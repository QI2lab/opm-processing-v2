"""Processing-component checks with controlled opm-v2 acquisition fixtures."""

import numpy as np
import pytest
from typer.testing import CliRunner

from opm_processing.dataio.position_collection import (
    open_position_collection,
)
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)
from opm_processing.process import app as process_app
from opm_processing.process import process

pytestmark = pytest.mark.integration


def test_singleton_z_stage_mode_writes_2d_name_without_stage_fusion(
    tmp_path,
    opm_v2_acquisition_factory,
    simulated_camera,
) -> None:
    """Keep dimensionality and acquisition mode distinct in output names."""
    raw = np.stack(
        [
            simulated_camera((1, 4, 5), geometry="cartesian", exposure=1 + t / 4).raw
            for t in range(2)
        ]
    )[:, None, None]
    dataset = opm_v2_acquisition_factory(raw, name="single_stage", mode="stage")
    path = dataset.path
    process(
        root_path=path,
        deconvolve=False,
        flatfield_correction=False,
        write_fused_max_projection_tiff=False,
    )

    output_path = tmp_path / "single_stage_2d.ome.zarr"
    assert output_path.is_dir()
    assert not (tmp_path / "single_stage_projection.ome.zarr").exists()
    assert not (tmp_path / "single_stage_stagefused.ome.zarr").exists()
    output = open_position_collection(output_path)
    np.testing.assert_array_equal(
        output.arrays[0].read().result(),
        ((raw[:, 0].astype(np.float32) - 100) * 0.25).astype(np.uint16),
    )
    assert "opm_v2" not in output.attributes


@pytest.mark.parametrize("save_float32", [False, True], ids=["uint16", "float32"])
@pytest.mark.parametrize(
    "separate_output", [False, True], ids=["source-dir", "output-dir"]
)
def test_projection_output_preserves_calibration_geometry_and_provenance(
    opm_v2_projection_zarr, save_float32, separate_output
):
    """Output dtype and destination preserve calibrated pixels and physical metadata."""
    fixture = opm_v2_projection_zarr
    parent = fixture.path.parent
    output_dir = parent / "nested/processed" if separate_output else parent
    args = [str(fixture.path)]
    if save_float32:
        args.append("--save-float32")
    if separate_output:
        args.extend(["--output", str(output_dir)])
    result = CliRunner().invoke(process_app, args)
    assert result.exit_code == 0, result.output
    output_path = output_dir / f"{fixture.path.stem}_projection.ome.zarr"
    collection = open_position_collection(output_path)
    assert collection.shape == (2, 1, 2, 1, 16, 18)
    assert collection.channel_names == fixture.channel_names
    assert collection.voxel_size_um == (1.0, 0.115, 0.115)
    np.testing.assert_allclose(
        collection.stage_positions_zxy, fixture.stage_positions_zxy
    )
    state = ProcessingState.read(processing_state_path(output_dir, fixture.path.stem))
    assert state.document["source"]["path"] == str(fixture.path.resolve())
    expected = np.clip(
        (fixture.raw_data[:, 0].astype(np.float32) - fixture.camera_offset)
        * fixture.camera_conversion,
        0,
        np.iinfo(np.uint16).max,
    )
    if not save_float32:
        expected = expected.astype(np.uint16)
    actual = collection.arrays[0].read().result()
    assert actual.dtype == (np.float32 if save_float32 else np.uint16)
    np.testing.assert_array_equal(actual[:, :, 0], expected)
    assert not (parent / f"{fixture.path.stem}_stagefused.ome.zarr").exists()
    if separate_output:
        assert not (parent / output_path.name).exists()


@pytest.mark.parametrize("resume", [False, True])
def test_projection_cli_resume_preserves_completed_pixels(
    opm_v2_projection_zarr, resume
):
    """Exercise the CLI flag through changed raw pixels and durable outputs."""
    fixture = opm_v2_projection_zarr
    args = [str(fixture.path)]
    first = CliRunner().invoke(process_app, args)
    assert first.exit_code == 0, first.output
    output_path = fixture.path.parent / f"{fixture.path.stem}_projection.ome.zarr"
    expected = np.maximum(
        (fixture.raw_data[:, 0].astype(np.float32) - fixture.camera_offset)
        * fixture.camera_conversion,
        0,
    ).astype(np.uint16)[:, :, None]
    np.testing.assert_array_equal(
        open_position_collection(output_path).arrays[0].read().result(),
        expected,
    )
    from opm_processing.dataio.acquisition import open_acquisition_datastore

    open_acquisition_datastore(fixture.path).write(np.uint16(900)).result()
    if resume:
        args.append("--resume")
    second = CliRunner().invoke(process_app, args)
    assert second.exit_code == 0, second.output
    after = open_position_collection(output_path).arrays[0].read().result()
    if not resume:
        expected.fill(
            np.uint16((900 - fixture.camera_offset) * fixture.camera_conversion)
        )
    np.testing.assert_array_equal(after, expected)
