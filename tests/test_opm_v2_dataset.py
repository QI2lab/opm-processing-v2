"""Integration coverage for acquisitions produced by QI2lab/opm-v2."""

import numpy as np
from typer.testing import CliRunner

from opm_processing.dataio.position_collection import (
    open_image_array,
    open_position_collection,
)
from opm_processing.fuse import app as fuse_app
from opm_processing.process import app as process_app
from opm_processing.process import process


def test_process_runs_end_to_end_on_opm_v2_projection_zarr(
    opm_v2_projection_zarr,
):
    """Verify projection acquisitions process and fuse end to end.

    Parameters
    ----------
    opm_v2_projection_zarr : object
        Value supplied for ``opm v2 projection zarr``.

    Returns
    -------
    None
        No value is returned.
    """
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
    assert collection.attributes["channels"] == list(fixture.channel_names)
    np.testing.assert_allclose(
        collection.attributes["stage_positions"], fixture.stage_positions_zxy
    )

    processed = collection.arrays[0].read().result()
    expected = np.clip(
        (fixture.raw_data[:, 0].astype(np.float32) - fixture.camera_offset)
        * fixture.camera_conversion,
        0,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    np.testing.assert_array_equal(processed[:, :, 0], expected)

    fused_path = fixture.path.parent / f"{fixture.path.stem}_stagefused.ome.zarr"
    fused = open_image_array(fused_path).read().result()
    assert fused.shape == (2, 2, 1, 16, 24)
    np.testing.assert_array_equal(fused[..., :16, :18], processed)


def test_float32_output_is_preserved_through_projection_fusion(
    opm_v2_projection_zarr,
) -> None:
    """Keep calibrated fractions in both per-position and fused products."""
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
    fused_path = fixture.path.parent / f"{fixture.path.stem}_stagefused.ome.zarr"
    fused = open_image_array(fused_path).read().result()

    assert processed.dtype == np.float32
    assert fused.dtype == np.float32
    assert np.any(processed != np.floor(processed))
    np.testing.assert_allclose(fused[..., :16, :18], processed, rtol=1e-6)


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
    provenance = collection.attributes["opm_processing"]
    assert provenance["source"]["path"] == str(fixture.path.resolve())
    assert collection.attributes["stage_positions"]
    assert collection.attributes["deskewed_voxel_size_um"]

    fuse_result = CliRunner().invoke(
        fuse_app,
        [str(output_dir), "--max-workers", "1"],
    )
    assert fuse_result.exit_code == 0, fuse_result.output
    assert (output_dir / f"{fixture.path.stem}_fused.ome.zarr").is_dir()
    assert (output_dir / "stitching_metrics.json").is_file()
