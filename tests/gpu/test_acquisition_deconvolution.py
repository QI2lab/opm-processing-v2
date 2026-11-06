"""Deconvolve unmodified, single-position opm-v2 writer output."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.interpolate import interp1d
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
from tests.simulation.metrics import (
    camera_noise_accuracy,
    camera_object_apertures,
    camera_prediction,
)


@pytest.mark.integration
@pytest.mark.parametrize(
    "diagnostics", [False, True], ids=["progress-default", "diagnostics"]
)
def test_as_acquired_combined_channel_deconvolution(
    cupy_gpu,
    simulated_camera,
    opm_v2_acquisition_factory,
    invoke_cli,
    monkeypatch,
    small_cpu_kernels,
    diagnostics,
    assert_reconstruction_storage,
):
    """Improve true camera likelihood without injecting PSF calibration into raw data."""
    from opm_processing.imageprocessing import rlgc
    from opm_processing.imageprocessing.camera import camera_correct
    from opm_processing.imageprocessing.opmtools import orthogonal_deskew

    observed = simulated_camera((17, 32, 36), step=1.2, budget=(3000, 4000))
    microscope = observed.microscope
    dataset = opm_v2_acquisition_factory(
        observed.raw[None, None, None],
        name="as_acquired",
        channels=(ChannelMetadata(0, "561 + 637", None, 10.0, 10.0),),
        scan_axis_step_um=1.2,
        camera_offset=microscope.camera.offset_adu,
        camera_conversion=microscope.camera.electrons_per_adu,
        stage_positions_zxy=((0.0, 0.0, 0.0),),
    )
    original_attributes = dict(open_group(dataset.path).attrs)
    assert "opm_processing" not in original_attributes
    assert "multiscales" in original_attributes["ome"]
    metadata = inspect_acquisition(dataset.path)
    assert not metadata.deconvolution
    np.testing.assert_array_equal(
        open_acquisition_datastore(metadata).read().result(), dataset.raw_data
    )
    reconstructed = []
    reconstruct = rlgc.rlgc

    def observe(*args, **kwargs):
        """Retain the real reconstruction to score against simulated camera means."""
        result = reconstruct(*args, **kwargs)
        reconstructed.append(
            (result.copy(), np.asarray(args[1]).copy(), kwargs["boundary"])
        )
        return result

    monkeypatch.setattr(rlgc, "rlgc", observe)
    result = invoke_cli(
        "process",
        dataset.path,
        deconvolve=True,
        max_projection=False,
        save_float32=True,
        z_downsample_level=1,
        decon_crop_scan=12,
        **({"decon_verbose": 1} if diagnostics else {}),
    )
    assert "decon:" in result.output
    assert ("RLGC acquired_shape=" in result.output) == diagnostics
    assert ("iteration=" in result.output) == diagnostics
    assert ("Automatic RLGC scan refinement:" in result.output) == diagnostics
    assert dict(open_group(dataset.path).attrs) == original_attributes
    ((recovered, psf, boundary),) = reconstructed
    assert recovered.shape == (49, 32, 36)
    baseline = camera_correct(
        observed.raw, microscope.camera.offset_adu, microscope.camera.electrons_per_adu
    )
    baseline = interp1d(np.arange(17) * 3, baseline, axis=0)(np.arange(49))
    psf /= psf.sum()
    boundary = "constant" if boundary == "zero" else boundary
    scores = [
        kl_div(
            observed.expectation,
            camera_prediction(density, psf, boundary=boundary)[::3],
        ).sum()
        for density in (recovered, baseline)
    ]
    assert scores[0] < scores[1]
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
    refined = replace(microscope, scan_step_um=0.4)
    apertures = camera_object_apertures(
        observed.ground_truth, microscope, observed.expectation.shape
    )
    # The inverse must predict independently simulated optical photons.
    prediction = camera_prediction(recovered, psf, boundary=boundary)[::3]
    for aperture in apertures:
        accuracy = camera_noise_accuracy(
            prediction, observed.expectation, microscope.camera, aperture
        )
        assert accuracy.accepted, accuracy
    # This separate numerical certificate never widens the optical target.
    acquired_apertures = []
    for aperture in apertures[:2]:
        mask = np.zeros(recovered.shape, bool)
        mask[::3] = aperture
        acquired_apertures.append(mask)
    assert_reconstruction_storage(
        saved[0, 0],
        (recovered,),
        collection.spatial_origins_zyx_um[0],
        collection.voxel_size_um,
        (0.0, 0.0, 0.0),
        refined,
        acquired_apertures,
    )
    state = ProcessingState.read(
        processing_state_path(dataset.path.parent, "as_acquired")
    )
    settings = state.run(output)["configuration"]["deconvolution"]
    assert settings["emission_wavelengths_nm"] == [637.0]
    assert settings["psf_parameters"]["lens"]["NA"] == 1.35
    assert settings["scan_upsample_factor"] == 3
    assert settings["psf_sha256"] is None
