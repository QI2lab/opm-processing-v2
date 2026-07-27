"""Integration coverage for acquisitions produced by QI2lab/opm-v2."""

import numpy as np

from opm_processing.dataio.position_collection import (
    open_image_array,
    open_position_collection,
)
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
