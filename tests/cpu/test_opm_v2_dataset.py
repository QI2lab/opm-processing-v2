"""Integration coverage for acquisitions produced by QI2lab/opm-v2."""

import numpy as np
import pytest
from typer.testing import CliRunner
from yaozarrs import open_group

from opm_processing.dataio.position_collection import (
    open_position_collection,
)
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)
from opm_processing.fuse import app as fuse_app
from opm_processing.process import app as process_app
from opm_processing.process import process


@pytest.mark.integration
def test_singleton_z_stage_mode_writes_2d_name_without_stage_fusion(
    tmp_path,
    acquisition_factory,
) -> None:
    """Keep dimensionality and acquisition mode distinct in output names."""
    raw = np.arange(40, dtype=np.uint16).reshape(2, 1, 1, 1, 4, 5) + 200
    dataset = acquisition_factory(raw, name="single_stage", mode="stage")
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
    assert not any(key.startswith("opm_") for key in output.attributes)


@pytest.mark.integration
def test_process_runs_end_to_end_on_opm_v2_projection_zarr(
    opm_v2_projection_zarr,
):
    """Verify projection acquisitions process and fuse end to end."""
    fixture = opm_v2_projection_zarr

    process(
        root_path=fixture.path,
        deconvolve=False,
        flatfield_correction=False,
        write_fused_max_projection_tiff=False,
    )

    collection_path = fixture.path.parent / f"{fixture.path.stem}_projection.ome.zarr"
    collection = open_position_collection(collection_path)
    assert collection.shape == (2, 1, 2, 1, 16, 18)
    assert collection.channel_names == fixture.channel_names
    np.testing.assert_allclose(
        collection.stage_positions_zxy, fixture.stage_positions_zxy
    )

    processed = collection.arrays[0].read().result()
    expected = np.clip(
        (fixture.raw_data[:, 0].astype(np.float32) - fixture.camera_offset)
        * fixture.camera_conversion,
        0,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    np.testing.assert_array_equal(processed[:, :, 0], expected)

    assert not (
        fixture.path.parent / f"{fixture.path.stem}_stagefused.ome.zarr"
    ).exists()


@pytest.mark.integration
def test_float32_output_is_preserved_for_single_position_projection(
    opm_v2_projection_zarr,
) -> None:
    """Keep calibrated fractions without creating redundant stage fusion."""
    fixture = opm_v2_projection_zarr

    process(
        root_path=fixture.path,
        deconvolve=False,
        save_float32=True,
        flatfield_correction=False,
        write_fused_max_projection_tiff=False,
    )

    collection_path = fixture.path.parent / f"{fixture.path.stem}_projection.ome.zarr"
    processed = open_position_collection(collection_path).arrays[0].read().result()

    assert processed.dtype == np.float32
    expected = np.maximum(
        (fixture.raw_data[:, 0].astype(np.float32) - fixture.camera_offset)
        * fixture.camera_conversion,
        0,
    )
    np.testing.assert_array_equal(processed[:, :, 0], expected)
    assert not (
        fixture.path.parent / f"{fixture.path.stem}_stagefused.ome.zarr"
    ).exists()


@pytest.mark.integration
def test_output_directory_is_created_and_can_be_passed_to_fuse(
    opm_v2_projection_zarr,
) -> None:
    """Keep self-contained processing and fusion artifacts outside the source."""
    fixture = opm_v2_projection_zarr
    output_dir = fixture.path.parent / "nested" / "processed"

    process_result = CliRunner().invoke(
        process_app,
        [str(fixture.path), "--output", str(output_dir)],
    )
    assert process_result.exit_code == 0, process_result.output
    assert output_dir.is_dir()

    processed_path = output_dir / f"{fixture.path.stem}_projection.ome.zarr"
    assert processed_path.is_dir()
    assert not (
        fixture.path.parent / f"{fixture.path.stem}_projection.ome.zarr"
    ).exists()

    collection = open_position_collection(processed_path)
    assert collection.channel_names == fixture.channel_names
    np.testing.assert_allclose(
        collection.stage_positions_zxy,
        fixture.stage_positions_zxy,
    )
    assert collection.voxel_size_um == (1.0, 0.115, 0.115)
    state_path = processing_state_path(output_dir, fixture.path.stem)
    state = ProcessingState.read(state_path)
    assert state.document["source"]["path"] == str(fixture.path.resolve())

    fuse_result = CliRunner().invoke(
        fuse_app,
        [str(output_dir), "--max-workers", "1"],
    )
    assert fuse_result.exit_code == 0, fuse_result.output
    assert (output_dir / f"{fixture.path.stem}_fused.ome.zarr").is_dir()
    max_z_path = output_dir / f"{fixture.path.stem}_max_z_fused.ome.zarr"
    assert max_z_path.is_dir()
    assert not any(key.startswith("opm_") for key in open_group(max_z_path).attrs)
    state = ProcessingState.read(state_path)
    registration = state.registration(processed_path)
    assert registration["fused_path"] == f"{fixture.path.stem}_fused.ome.zarr"
    assert registration["max_projection_path"] == max_z_path.name
    assert registration["tiles"]
    assert not (output_dir / "stitching_metrics.json").exists()
    processed_pixels = collection.arrays[0].read().result()
    expected = np.maximum(
        (fixture.raw_data[:, 0].astype(np.float32) - fixture.camera_offset)
        * fixture.camera_conversion,
        0,
    ).astype(np.uint16)
    np.testing.assert_array_equal(processed_pixels[:, :, 0], expected)
    fused = (
        open_group(output_dir / f"{fixture.path.stem}_fused.ome.zarr")["0"]
        .to_tensorstore()
        .read()
        .result()
    )
    projected = open_group(max_z_path)["0"].to_tensorstore().read().result()
    # Single-position projection fusion must copy every source pixel.
    np.testing.assert_array_equal(fused[..., :16, :18], processed_pixels)
    np.testing.assert_array_equal(projected, fused.max(axis=2, keepdims=True))


@pytest.mark.integration
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
    before = open_position_collection(output_path).arrays[0].read().result()
    raw = open_position_collection(fixture.path)
    for array in raw.arrays:
        array.write(np.uint16(900)).result()
    if resume:
        args.append("--resume")
    second = CliRunner().invoke(process_app, args)
    assert second.exit_code == 0, second.output
    after = open_position_collection(output_path).arrays[0].read().result()
    if resume:
        np.testing.assert_array_equal(after, before)
    else:
        expected = np.uint16((900 - fixture.camera_offset) * fixture.camera_conversion)
        np.testing.assert_array_equal(after, np.full_like(after, expected))
