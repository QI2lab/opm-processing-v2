"""Recover known planar emitters through disk-to-disk CUDA deconvolution."""

import numpy as np
import pytest

from opm_processing.dataio.position_collection import (
    open_position_collection,
)
from opm_processing.process import process


@pytest.mark.integration
def test_planar_deconvolution_recovers_saved_emitters_in_every_series(
    cupy_gpu, planar_camera_acquisition, tmp_path
):
    """Recover photon flux and emitter positions through real multi-series processing."""
    case = planar_camera_acquisition
    process(
        case.dataset.path,
        deconvolve=True,
        decon_psf_paths=[case.psf_path, case.psf_path],
        flatfield_correction=False,
        save_float32=True,
        create_fused_max_projection=False,
        write_fused_max_projection_tiff=False,
    )
    output = open_position_collection(
        tmp_path / "planar_points_decon_projection.ome.zarr"
    )
    for position, array in enumerate(output.arrays):
        saved = array.read().result()
        expected = case.truth[:, position]
        measured = case.photons[:, position]
        for time in range(2):
            for channel in range(2):
                recovered = saved[time, channel]
                truth = expected[time, channel]
                observed = measured[time, channel]
                assert np.mean((recovered - truth) ** 2) < np.mean(
                    (observed - truth) ** 2
                )
                np.testing.assert_allclose(recovered.sum(), observed.sum(), rtol=0.1)
                for row, column in ((12, 13), (28, 30)):
                    region = recovered[0, row - 2 : row + 3, column - 2 : column + 3]
                    assert np.unravel_index(np.argmax(region), region.shape) == (2, 2)
