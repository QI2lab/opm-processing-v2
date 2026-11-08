"""Stage placement across acquisition metadata, deskew and fusion."""

import numpy as np
import pytest
from scipy.ndimage import maximum_filter
from yaozarrs import open_group

from opm_processing.dataio.position_collection import (
    open_image_array,
    open_position_collection,
)
from opm_processing.imageprocessing.opmpsf import get_skewed_coords
from opm_processing.imageprocessing.tilefusion import TileFusion
from opm_processing.process import process
from tests.simulation.camera_likelihood import (
    camera_sampling_accuracy,
    camera_sampling_envelope,
    sampling_stencil_footprint,
    sampling_stencil_radii,
)
from tests.simulation.metrics import camera_object_apertures, camera_volume_samples

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("mode", ("stage", "mirror"))
@pytest.mark.parametrize("crop", (False, True))
def test_physical_stage_z_translation_preserves_registered_xy(
    processing_options, tmp_path, stage_placement_acquisition, crop, pipeline_geometry
):
    """Sample a fixed bead through tilted planes without using placement code.

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
        save_float32=True,
        crop_after_deskew=crop,
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
    # The independent forward model fixes the bead in physical coordinates.
    # A common scan-length error must not disappear in a relative comparison.
    expected_peak_um = np.asarray((20, 64, 24)) * pixel_um
    placed_peaks = np.asarray(peaks) * pixel_um + origins
    np.testing.assert_allclose(
        (placed_peaks - expected_peak_um) / pixel_um, 0.0, atol=1 + 1e-8
    )
    np.testing.assert_allclose(
        (placed_peaks - placed_peaks[0]) / pixel_um, 0.0, atol=1 + 1e-8
    )
    crop_y = int(np.ceil(64 * np.cos(np.deg2rad(angle)))) if crop else 0
    np.testing.assert_allclose(np.asarray(peaks)[:, 1:] - (64 - crop_y, 24), 0, atol=1)
    fusion.run()
    np.testing.assert_allclose(fusion.global_offsets, 0.0, atol=1.0)
    assert len(fusion.pairwise_metrics) >= 2
    written = open_image_array(tmp_path / "physical_stage_fused.ome.zarr")
    fused = written[0, 0].read().result()
    peak = np.asarray(np.unravel_index(np.argmax(fused), fused.shape))
    physical_peak = peak * pixel_um + fusion.offset_um
    np.testing.assert_allclose(
        (physical_peak - expected_peak_um) / pixel_um, 0.0, atol=2.0
    )
    from opm_processing.imageprocessing.opmpsf import camera_grid_center

    means, raw = case.means[0, :, 0], case.raw_data[0, :, 0]
    if case.metadata.scan_axis_reversed:
        means, raw = means[:, ::-1], raw[:, ::-1]
    shape = raw.shape[-3:]
    geometry = pipeline_geometry(shape, microscope)
    native_support = camera_volume_samples(
        geometry.support.astype(float),
        (0, 0, 0),
        (pixel_um,) * 3,
        (0, 0, 0),
        shape,
        microscope,
    )
    raw_origins = np.zeros((len(means), 3))
    raw_origins[:, 0] = -case.stage_z_values_um
    camera_center = camera_grid_center(
        shape, pixel_um, microscope.scan_step_um, microscope.angle_deg
    )
    dataset = open_group(tmp_path / "physical_stage_fused.ome.zarr").ome_metadata()
    dataset = dataset.multiscales[0].datasets[0]
    gain = 2 * pixel_um / microscope.scan_step_um
    theta = np.deg2rad(angle)
    native_x, native_y, native_z = get_skewed_coords(
        shape, pixel_um, microscope.scan_step_um, theta
    )
    for fused_output in (False, True):
        spacing = np.asarray(
            dataset.scale_transform.scale[-3:]
            if fused_output
            else collection.voxel_size_um
        )
        for position, mean in enumerate(means):
            image = (
                fused
                if fused_output
                else collection.arrays[position][0, 0].read().result()
            )
            origin = (
                dataset.translation_transform.translation[-3:]
                if fused_output
                else origins[position]
            )
            offsets = None
            pitch = spacing * (2 if fused_output else 1)
            radii = np.asarray(sampling_stencil_radii(microscope, pitch))
            footprint = sampling_stencil_footprint(
                microscope, pitch, origin_y_error_um=0.0005
            )
            if fused_output:
                delta = raw_origins - raw_origins[position]
                offsets = np.column_stack(
                    (
                        (delta[:, 1] - delta[:, 0] / np.tan(theta))
                        / microscope.scan_step_um,
                        delta[:, 0] / (pixel_um * np.sin(theta)),
                        delta[:, 2] / pixel_um,
                    )
                )
                # Nearest-bound sampling may choose either side of a fractional
                # native coordinate. Retain that extra original-camera sample.
                guard = np.any(
                    ~np.isclose(offsets, np.rint(offsets), rtol=0, atol=1e-8), axis=0
                ).astype(int)
                radii += guard
                footprint = maximum_filter(
                    np.pad(footprint, tuple((g, g) for g in guard)),
                    size=2 * guard + 1,
                    mode="constant",
                    cval=False,
                )
            envelope = camera_sampling_envelope(
                means if fused_output else mean,
                raw if fused_output else raw[position],
                microscope.camera,
                radii=radii,
                footprint=footprint,
                views=len(means) if fused_output else 1,
                view_offsets_syx=offsets,
            )
            sample = (origin, spacing, raw_origins[position], shape, microscope)
            support = camera_volume_samples(np.ones(image.shape), *sample)
            support = (
                np.minimum(support, native_support) >= 1 - 32 * np.finfo(float).eps
            )
            # Keep every contributing sample inside the original finite
            # specimen's PSF-eroded domain, before examining recorded pixels.
            stencil_x, stencil_y, stencil_z = get_skewed_coords(
                footprint.shape, pixel_um, microscope.scan_step_um, theta
            )
            stencil_center = camera_grid_center(
                footprint.shape, pixel_um, microscope.scan_step_um, angle
            )
            reach = tuple(
                np.abs(np.broadcast_to(axis - center, footprint.shape)[footprint]).max()
                for axis, center in zip(
                    (stencil_z, stencil_y, stencil_x), stencil_center, strict=True
                )
            )
            for coordinate, stage_origin, bead, axis, halo, radius in zip(
                (native_z, native_y, native_x),
                raw_origins[position],
                expected_peak_um,
                case.ground_truth.axes,
                microscope.psf_half_extent_zyx_um,
                reach,
                strict=True,
            ):
                physical = coordinate + stage_origin - bead
                support &= (physical >= axis[0] + halo + radius) & (
                    physical <= axis[-1] - halo - radius
                )
            prediction = camera_volume_samples(image / gain, *sample)
            apertures = camera_object_apertures(
                case.ground_truth,
                microscope,
                shape,
                camera_offset_zyx_um=camera_center
                + raw_origins[position]
                - expected_peak_um,
            )
            assert all(np.any(aperture & support) for aperture in apertures[:2])
            for aperture in apertures:
                selected = aperture & support
                if np.any(selected):
                    accuracy = camera_sampling_accuracy(
                        prediction, mean, envelope, selected
                    )
                    assert accuracy.accepted, accuracy
