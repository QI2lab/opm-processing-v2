"""Stage placement across acquisition metadata, deskew and fusion."""

import numpy as np
import pytest

from opm_processing.dataio.position_collection import (
    open_image_array,
    open_position_collection,
)
from opm_processing.imageprocessing.tilefusion import TileFusion
from opm_processing.process import process

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("angle", (30.0, 45.0))
@pytest.mark.parametrize("mode", ("stage", "mirror"))
def test_physical_stage_z_translation_preserves_registered_xy(
    processing_options, tmp_path, stage_placement_acquisition
):
    """Sample a fixed bead through tilted planes without using placement code.

    For stage scanning the optical plane is stationary and specimen coordinates
    equal laboratory coordinates minus the instantaneous stage displacement.
    For mirror scanning the plane translates while the stage remains fixed.
    Both experiments move the stage only in Z between volumes.
    """
    path = stage_placement_acquisition.path
    pixel_um = stage_placement_acquisition.pixel_um
    process(
        root_path=path,
        deconvolve=False,
        **processing_options,
    )
    collection = open_position_collection(tmp_path / "physical_stage_deskewed.ome.zarr")
    fusion = TileFusion(
        root_path=path,
        downsample_factors=(1, 1, 1),
        ssim_window=3,
        threshold=0.1,
        max_registration_shift_zyx=(2, 2, 2),
        multiscale_factors=(),
        resolution_multiples=((1, 1, 1),),
        max_workers=1,
    )
    origins = np.asarray(fusion._tile_positions)
    peaks = []
    for tile in collection.arrays:
        volume = tile[0, 0].read().result()
        peaks.append(np.unravel_index(np.argmax(volume), volume.shape))
    # Absolute scan-start conventions add one common Y origin in stage mode.
    # Relative physical feature positions must agree without registration.
    placed_peaks = np.asarray(peaks) * pixel_um + origins
    np.testing.assert_allclose(
        (placed_peaks - placed_peaks[0]) / pixel_um, 0.0, atol=1 + 1e-8
    )
    np.testing.assert_allclose(np.asarray(peaks)[:, 1:] - (64, 24), 0, atol=1)
    fusion.run()
    np.testing.assert_allclose(fusion.global_offsets, 0.0, atol=1.0)
    assert len(fusion.pairwise_metrics) >= 2
    written = open_image_array(tmp_path / "physical_stage_fused.ome.zarr")
    fused = written[0, 0].read().result()
    peak = np.asarray(np.unravel_index(np.argmax(fused), fused.shape))
    physical_peak = peak * pixel_um + fusion.offset_um
    np.testing.assert_allclose(
        (physical_peak - placed_peaks[0]) / pixel_um, 0.0, atol=2.0
    )
