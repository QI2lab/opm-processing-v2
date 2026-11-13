"""Stage placement across acquisition metadata, deskew and fusion."""

import json

import numpy as np
import pytest
from yaozarrs import open_group

from opm_processing.dataio.position_collection import (
    open_image_array,
    open_position_collection,
)
from opm_processing.imageprocessing.tilefusion import TileFusion
from opm_processing.process import process

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("cupy_gpu")]


@pytest.mark.parametrize("mode,crop", (("mirror", False), ("stage", True)))
def test_physical_stage_z_translation_preserves_registered_xy(
    processing_options, tmp_path, stage_placement_acquisition, crop
):
    """Observe a fixed fiducial amid structured lines without using placement code.

    For stage scanning the optical plane is stationary and specimen coordinates
    equal laboratory coordinates minus the instantaneous stage displacement.
    For mirror scanning the plane translates while the stage remains fixed.
    Both experiments move the stage only in Z between volumes.
    """
    case = stage_placement_acquisition
    path = case.path
    pixel_um = case.metadata.pixel_size_um
    microscope = case.ground_truth.microscope
    angle = microscope.angle_deg
    process(
        root_path=path,
        deconvolve=False,
        save_float32=False,
        crop_after_deskew=crop,
        **processing_options,
    )
    collection = open_position_collection(tmp_path / "physical_stage_deskewed.ome.zarr")
    fusion = TileFusion(
        root_path=path,
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
    # The independent forward model fixes the bead in physical coordinates.
    # A common scan-length error must not disappear in a relative comparison.
    expected_peak_um = case.bead_zyx_um
    placed_peaks = np.asarray(peaks) * pixel_um + origins
    np.testing.assert_allclose(
        (placed_peaks - expected_peak_um) / pixel_um, 0.0, atol=1 + 1e-8
    )
    # Independent noise can select opposite PSF lobes within the per-view
    # absolute bound; their argmax difference is not a placement measurement.
    crop_y = (
        int(np.ceil(case.raw_data.shape[-2] * np.cos(np.deg2rad(angle)))) if crop else 0
    )
    expected_yx = expected_peak_um[1:] / pixel_um - (crop_y, 0)
    np.testing.assert_allclose(np.asarray(peaks)[:, 1:] - expected_yx, 0, atol=1)
    fusion.run()
    np.testing.assert_allclose(fusion.global_offsets, 0.0, atol=1.0)
    assert len(fusion.pairwise_metrics) >= 2
    report = json.loads((tmp_path / "physical_stage_depth_histograms.json").read_text())
    assert report == fusion._depth_intensity_report
    assert report["enabled"] and report["tile_depth_indices"] == [0, 1, 2]
    (channel,) = report["channels"]
    assert (channel["time_index"], channel["channel_index"]) == (0, 0)
    assert channel["reference_position_indices"] == [0]
    assert channel["reference_voxels"] > 0
    assert [layer["depth_index"] for layer in channel["layers"]] == [1, 2]
    assert all(layer["mapping_knots"] > 0 for layer in channel["layers"])
    written = open_image_array(tmp_path / "physical_stage_fused.ome.zarr")
    fused = written[0, 0].read().result()
    peak = np.asarray(np.unravel_index(np.argmax(fused), fused.shape))
    metadata = (
        open_group(tmp_path / "physical_stage_fused.ome.zarr")
        .ome_metadata()
        .multiscales[0]
        .datasets[0]
    )
    np.testing.assert_array_equal(
        metadata.scale_transform.scale[-3:], collection.voxel_size_um
    )
    physical_peak = (
        peak * np.asarray(metadata.scale_transform.scale[-3:])
        + metadata.translation_transform.translation[-3:]
    )
    np.testing.assert_allclose(
        (physical_peak - expected_peak_um) / pixel_um, 0.0, atol=2.0
    )
