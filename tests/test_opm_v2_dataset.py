"""Integration coverage for acquisitions produced by QI2lab/opm-v2."""

import importlib

import numpy as np
from typer.testing import CliRunner
from yaozarrs import open_group

from opm_processing.dataio.acquisition import AcquisitionMetadata, ChannelMetadata
from opm_processing.dataio.position_collection import (
    create_position_collection,
    open_position_collection,
)
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)
from opm_processing.fuse import app as fuse_app
from opm_processing.process import app as process_app
from opm_processing.process import _planar_output_labels, process


def test_planar_output_names_distinguish_mode_from_dimensionality() -> None:
    """Reserve the projection label for acquisitions recorded in projection mode."""
    assert _planar_output_labels("projection", False) == (
        "projection",
        "projection",
    )
    assert _planar_output_labels("projection", True) == (
        "decon_projection",
        "deconvolved_projection",
    )
    assert _planar_output_labels("mirror", False) == ("2d", "2d")
    assert _planar_output_labels("stage", True) == (
        "decon_2d",
        "deconvolved_2d",
    )


def test_singleton_z_stage_mode_writes_2d_name_without_stage_fusion(
    tmp_path,
    monkeypatch,
) -> None:
    """Keep dimensionality and acquisition mode distinct in output names."""
    process_module = importlib.import_module("opm_processing.process")
    path = tmp_path / "single_stage.ome.zarr"
    collection = create_position_collection(
        path,
        (2, 1, 1, 1, 4, 5),
        (1.0, 0.115, 0.115),
        stage_positions=((30.0, 100.0, 200.0),),
        channels=("488nm",),
    )
    raw = np.arange(2 * 1 * 1 * 4 * 5, dtype=np.uint16).reshape(2, 1, 1, 4, 5) + 200
    collection.arrays[0].write(raw).result()
    metadata = AcquisitionMetadata(
        path=path,
        storage_format="opm-v2-ome-zarr-v3",
        mode="stage",
        axes=("t", "p", "c", "z", "y", "x"),
        shape=(2, 1, 1, 1, 4, 5),
        array_paths=("0/0",),
        acquisition_order=("t", "p", "z", "c"),
        channels=(ChannelMetadata(0, "488nm", 488.0, 2.0, 10.0),),
        stage_positions_zxy=((30.0, 100.0, 200.0),),
        scan_start_positions_xyz=(),
        scan_end_positions_xyz=(),
        scan_axis="x",
        scan_axis_step_um=0.4,
        pixel_size_um=0.115,
        angle_deg=30.0,
        camera_offset=100.0,
        camera_conversion=0.25,
        excess_scan_positions=0,
        excess_scan_start_positions=0,
        excess_scan_end_positions=0,
        orientations=(),
        sidecar_paths=(),
    )
    monkeypatch.setattr(process_module, "inspect_acquisition", lambda _path: metadata)

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
    assert not any(key.startswith("opm_") for key in output.attributes)


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


def test_2d_deconvolution_loops_over_time_position_and_channel(
    opm_v2_projection_zarr,
    monkeypatch,
) -> None:
    """Route each T/P/C image independently through central-plane 2D RLGC."""
    fixture = opm_v2_projection_zarr
    process_module = importlib.import_module("opm_processing.process")
    rlgc_module = importlib.import_module("opm_processing.imageprocessing.rlgc")
    generated_psfs = []
    calls = []

    def fake_generate_proj_psf(*, em_wvl, pixel_size_um):
        del em_wvl, pixel_size_um
        psf = np.stack(
            (
                np.full((5, 5), 1.0, dtype=np.float32),
                np.full((5, 5), 2.0, dtype=np.float32),
                np.full((5, 5), 4.0, dtype=np.float32),
            )
        )
        generated_psfs.append(psf)
        return psf

    def fake_rlgc_2d(*, image, skewed_psf, **kwargs):
        del kwargs
        calls.append((np.asarray(image).shape, skewed_psf))
        return np.asarray(image, dtype=np.float32)

    monkeypatch.setattr(process_module, "generate_proj_psf", fake_generate_proj_psf)
    monkeypatch.setattr(rlgc_module, "rlgc_2d", fake_rlgc_2d)

    process(
        root_path=fixture.path,
        deconvolve=True,
        flatfield_correction=False,
        write_fused_max_projection_tiff=False,
    )

    expected_calls = (
        fixture.raw_data.shape[0]
        * fixture.raw_data.shape[1]
        * fixture.raw_data.shape[2]
    )
    assert len(calls) == expected_calls
    assert len(generated_psfs) == fixture.raw_data.shape[2]
    assert all(shape == fixture.raw_data.shape[-2:] for shape, _psf in calls)
    assert all(psf.shape == (3, 5, 5) for _shape, psf in calls)

    output = open_position_collection(
        fixture.path.parent / f"{fixture.path.stem}_decon_projection.ome.zarr"
    )
    assert not any(key.startswith("opm_") for key in output.attributes)


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
    assert np.any(processed != np.floor(processed))
    assert not (
        fixture.path.parent / f"{fixture.path.stem}_stagefused.ome.zarr"
    ).exists()


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
