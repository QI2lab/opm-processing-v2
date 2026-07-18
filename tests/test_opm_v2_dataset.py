"""Integration coverage for acquisitions produced by QI2lab/opm-v2."""

import json

import numpy as np

from opm_processing.dataio.position_collection import (
    open_image_array,
    open_position_collection,
)
from opm_processing.imageprocessing.opmtools import deskew_shape_estimator
from opm_processing.process import process


def test_opm_v2_fixture_matches_handler_storage_schema(opm_v2_projection_zarr):
    """Verify the synthetic fixture matches the opm-v2 storage schema.

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
    zarray = json.loads((fixture.path / ".zarray").read_text())
    zattrs = json.loads((fixture.path / ".zattrs").read_text())

    assert zarray["zarr_format"] == 2
    assert zarray["shape"] == list(fixture.raw_data.shape)
    assert zarray["chunks"] == [1, 1, 1, 16, 18]
    assert set(zattrs) == {"frame_metadatas"}
    first_frame = zattrs["frame_metadatas"][0]
    assert first_frame["pixel_size_um"] == fixture.pixel_size_um
    assert first_frame["mda_event"]["index"] == {"t": 0, "p": 0, "c": 0}
    assert first_frame["mda_event"]["metadata"]["DAQ"]["mode"] == "projection"


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
        stage_x_flipped=False,
        stage_y_flipped=False,
        stage_z_flipped=False,
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


def test_process_runs_normal_skewed_opm_v2_acquisition(opm_v2_skewed_zarr):
    """Verify mirror- and stage-scanned skewed acquisitions process correctly.

    Parameters
    ----------
    opm_v2_skewed_zarr : object
        Value supplied for ``opm v2 skewed zarr``.

    Returns
    -------
    None
        No value is returned.
    """
    fixture = opm_v2_skewed_zarr

    process(
        root_path=fixture.path,
        deconvolve=False,
        max_projection=True,
        flatfield_correction=False,
        create_fused_max_projection=False,
        z_downsample_level=1,
        write_fused_max_projection_tiff=False,
        stage_x_flipped=False,
        stage_y_flipped=False,
        stage_z_flipped=False,
    )

    input_shape = (
        fixture.raw_data.shape[-3] - fixture.excess_scan_positions,
        fixture.raw_data.shape[-2],
        fixture.raw_data.shape[-1],
    )
    expected_zyx, _, _, _ = deskew_shape_estimator(
        input_shape,
        theta=30.0,
        distance=fixture.scan_axis_step_um,
        pixel_size=fixture.pixel_size_um,
        crop_after_deskew=False,
    )
    deskewed_path = fixture.path.parent / f"{fixture.path.stem}_deskewed.ome.zarr"
    deskewed_collection = open_position_collection(deskewed_path)
    assert deskewed_collection.shape == (1, 1, 1, *expected_zyx)
    assert deskewed_collection.attributes["channels"] == list(fixture.channel_names)
    assert deskewed_collection.attributes["scan_axis_step_um"] == (
        fixture.scan_axis_step_um
    )
    np.testing.assert_allclose(
        deskewed_collection.attributes["stage_positions"],
        fixture.stage_positions_zxy,
    )

    deskewed = deskewed_collection.arrays[0].read().result()
    assert deskewed.dtype == np.uint16
    assert np.count_nonzero(deskewed) > 0
    assert deskewed.max() <= np.iinfo(np.uint16).max

    max_path = fixture.path.parent / f"{fixture.path.stem}_max_z_deskewed.ome.zarr"
    max_collection = open_position_collection(max_path)
    max_projection = max_collection.arrays[0].read().result()
    assert max_collection.shape == (1, 1, 1, 1, expected_zyx[1], expected_zyx[2])
    np.testing.assert_array_equal(max_projection, deskewed.max(axis=2, keepdims=True))
