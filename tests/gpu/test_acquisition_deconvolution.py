"""Deconvolve unmodified, single-position opm-v2 writer output."""

import numpy as np
import pytest
from scipy.special import kl_div
from yaozarrs import open_group

from opm_processing.dataio.acquisition import (
    ChannelMetadata,
    inspect_acquisition,
    open_acquisition_datastore,
)
from opm_processing.dataio.position_collection import open_position_collection
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)


@pytest.mark.integration
@pytest.mark.parametrize("kind", ("line-pairs",))
def test_as_acquired_combined_channel_deconvolution(
    cupy_gpu,
    simulated_camera,
    opm_v2_acquisition_factory,
    invoke_cli,
    monkeypatch,
    small_cpu_kernels,
    production_forward_predictions,
    simulation_parameters,
    kind,
):
    """Improve true camera likelihood without injecting PSF calibration into raw data."""
    from opm_processing.imageprocessing import rlgc
    from opm_processing.imageprocessing.opmtools import orthogonal_deskew

    observed = simulated_camera(
        (
            simulation_parameters["whole_scan_planes"],
            *simulation_parameters["camera_shape_yx"],
        ),
        kind=kind,
        step=1.2,
        budget=(3000, 4000),
    )
    microscope = observed.microscope
    dataset = opm_v2_acquisition_factory(
        observed.raw[None, None, None],
        name="as_acquired",
        channels=(ChannelMetadata(0, "561 + 637", None, 10.0, 10.0),),
        scan_axis_step_um=1.2,
        camera_offset=microscope.camera.offset_adu,
        camera_conversion=microscope.camera.electrons_per_adu,
        camera_read_noise_e=microscope.camera.read_noise_e,
        stage_positions_zxy=((0.0, 0.0, 0.0),),
    )
    original_attributes = dict(open_group(dataset.path).attrs)
    assert "opm_processing" not in original_attributes
    assert "multiscales" in original_attributes["ome"]
    metadata = inspect_acquisition(dataset.path)
    assert "psf_parameters" not in (metadata.deconvolution or {})
    np.testing.assert_array_equal(
        open_acquisition_datastore(metadata).read().result(), dataset.raw_data
    )
    reconstructed = []
    reconstruct = rlgc.rlgc

    def observe(*args, **kwargs):
        """Retain the real reconstruction to score against simulated camera means."""
        result = reconstruct(*args, **kwargs)
        reconstructed.append(result)
        return result

    monkeypatch.setattr(rlgc, "rlgc", observe)
    invoke_cli(
        "process",
        dataset.path,
        deconvolve=True,
        max_projection=False,
        save_float32=True,
        z_downsample_level=1,
        decon_crop_scan=12,
    )
    assert dict(open_group(dataset.path).attrs) == original_attributes
    (recovered,) = reconstructed
    assert recovered.shape == (
        (observed.raw.shape[0] - 1) * 3 + 1,
        *observed.raw.shape[1:],
    )
    measured = production_forward_predictions[-1]
    # Sparse progress is measured against the actual production initializer,
    # with both predictions evaluated on the acquired camera planes.
    assert (
        kl_div(observed.expectation, measured["prediction"]).sum()
        < kl_div(observed.expectation, measured["baseline_prediction"]).sum()
    )
    output = dataset.path.parent / "as_acquired_decon_deskewed.ome.zarr"
    collection = open_position_collection(output)
    saved = collection.arrays[0].read().result()
    assert np.isfinite(saved).all() and np.all(saved >= 0)
    np.testing.assert_array_equal(
        saved[0, 0],
        orthogonal_deskew(
            recovered,
            theta=microscope.angle_deg,
            distance=0.4,
            pixel_size=microscope.pixel_size_um,
            downsample_factor=1,
        ),
    )
    np.testing.assert_allclose(
        collection.voxel_size_um, microscope.pixel_size_um, rtol=0, atol=0.0005
    )
    np.testing.assert_allclose(
        collection.spatial_origins_zyx_um[0], (0, 0, 0), rtol=0, atol=0.0005
    )
    state = ProcessingState.read(
        processing_state_path(dataset.path.parent, "as_acquired")
    )
    settings = state.run(output)["configuration"]["deconvolution"]
    assert settings["solver_revision"] == rlgc.RLGC_SOLVER_REVISION
    assert settings["emission_wavelengths_nm"] == [637.0]
    assert settings["psf_parameters"]["lens"]["NA"] == 1.35
    assert settings["scan_upsample_factor"] == 3
    assert settings["psf_sha256"] is None
