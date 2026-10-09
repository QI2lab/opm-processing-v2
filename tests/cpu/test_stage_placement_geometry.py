"""Independent optical forward model for stage-Z tile placement."""

import numpy as np
import pytest

from opm_processing.dataio.position_collection import (
    open_image_array,
    open_position_collection,
)
from opm_processing.imageprocessing.tilefusion import TileFusion
from opm_processing.process import process
from tests.fixtures.acquisition import opm_v2_frame_metadata, write_opm_v2_zarr


@pytest.mark.integration
@pytest.mark.parametrize("angle", (30.0, 45.0))
@pytest.mark.parametrize("mode", ("stage", "mirror"))
def test_physical_stage_z_translation_preserves_registered_xy(
    processing_options, tmp_path, angle, mode
):
    """Sample a fixed bead through tilted planes without using placement code.

    For stage scanning the optical plane is stationary and specimen coordinates
    equal laboratory coordinates minus the instantaneous stage displacement.
    For mirror scanning the plane translates while the stage remains fixed.
    Both experiments move the stage only in Z between volumes.
    """
    pixel_um = 0.115
    step_um = 0.23
    scans, camera_height, camera_width = 40, 64, 48
    camera_u, camera_v = np.indices((camera_height, camera_width), dtype=float)
    theta = np.deg2rad(angle)
    plane_z = camera_u * pixel_um * np.sin(theta)
    plane_y = camera_u * pixel_um * np.cos(theta)
    plane_x = camera_v * pixel_um
    bead_zyx_um = np.asarray((20, 64, 24)) * pixel_um
    sigma_um = 1.5 * pixel_um
    stage_z_values_um = -np.asarray((0, 6, 12)) * pixel_um
    raw = np.zeros((1, 3, 1, scans, camera_height, camera_width), dtype=np.uint16)
    frames = []
    for position, stage_z_um in enumerate(stage_z_values_um):
        for scan in range(scans):
            if mode == "stage":
                # Forward acquisition: the stationary plane sees the specimen
                # move from negative scan-axis displacement back to zero.
                stage_scan_um = (scan - (scans - 1)) * step_um
                plane_scan_um = 0.0
            else:
                stage_scan_um = 0.0
                plane_scan_um = scan * step_um
            specimen_z = plane_z - stage_z_um
            specimen_y = plane_y + plane_scan_um - stage_scan_um
            specimen_x = plane_x
            radius_squared = (
                (specimen_z - bead_zyx_um[0]) ** 2
                + (specimen_y - bead_zyx_um[1]) ** 2
                + (specimen_x - bead_zyx_um[2]) ** 2
            )
            raw[0, position, 0, scan] = np.rint(
                20_000 * np.exp(-radius_squared / (2 * sigma_um**2))
            ).astype(np.uint16)
            frames.append(
                opm_v2_frame_metadata(
                    index={"t": 0, "p": position, "c": 0, "z": scan},
                    daq_metadata={
                        "mode": mode,
                        "channel_states": [True],
                        "exposure_channels_ms": [10.0],
                        "laser_powers": [10.0],
                        "interleaved": True,
                        "blanking": True,
                        "current_channel": "488nm",
                        "scan_axis_step_um": step_um,
                        "image_mirror_position": float(scan),
                        "image_mirror_step_um": step_um,
                        "image_mirror_range_um": scans * step_um,
                    },
                    opm_metadata={
                        "angle_deg": angle,
                        "camera_Zstage_orientation": "normal",
                        "camera_XYstage_orientation": "normal",
                        "camera_mirror_orientation": "normal",
                    },
                    stage_metadata={
                        "x_pos": stage_scan_um,
                        "y_pos": 0.0,
                        "z_pos": stage_z_um,
                        "excess_image": False,
                    },
                    camera_shape_yx=(camera_height, camera_width),
                    pixel_size_um=pixel_um,
                    camera_offset=0.0,
                    camera_conversion=1.0,
                    runner_time_ms=float(position * scans + scan),
                    hardware_triggered=True,
                )
            )
    path = tmp_path / "physical_stage.zarr"
    write_opm_v2_zarr(
        path=path,
        raw_data=raw,
        labels=("t", "p", "c", "z", "y", "x"),
        chunks=(1, 1, 1, 1, camera_height, camera_width),
        frame_metadatas=frames,
    )
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
