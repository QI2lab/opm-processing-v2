"""Deconvolve unmodified, single-position opm-v2 writer output."""

import importlib
from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest
from scipy.special import kl_div
from yaozarrs import open_group

from opm_processing.dataio.acquisition import (
    ChannelMetadata,
    inspect_acquisition,
    open_acquisition_datastore,
)
from opm_processing.dataio.position_collection import (
    open_position_collection,
    update_group_attributes,
)
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)


@pytest.mark.integration
def test_as_acquired_combined_channel_deconvolution(
    cupy_gpu,
    simulated_camera,
    opm_v2_acquisition_factory,
    invoke_cli,
    monkeypatch,
    production_forward_predictions,
    simulation_parameters,
):
    """Improve true camera likelihood without injecting PSF calibration into raw data."""
    from opm_processing.imageprocessing import rlgc
    from opm_processing.imageprocessing.opmtools import deskew_shape_estimator

    processing = importlib.import_module("opm_processing.process")

    observed = simulated_camera(
        (
            simulation_parameters["whole_scan_planes"],
            *simulation_parameters["camera_shape_yx"],
        ),
        kind="line-pairs",
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
    original = {
        path.relative_to(dataset.path): sha256(path.read_bytes()).digest()
        for path in dataset.path.rglob("*")
        if path.is_file()
    }
    assert "opm_processing" not in original_attributes
    assert "multiscales" in original_attributes["ome"]
    metadata = inspect_acquisition(dataset.path)
    assert "psf_parameters" not in (metadata.deconvolution or {})
    np.testing.assert_array_equal(
        open_acquisition_datastore(metadata).read().result(), dataset.raw_data
    )
    deskewed = []
    deskew = processing.orthogonal_deskew

    def observe(*args, **kwargs):
        """Observe actual refined input and output bytes before persistence."""
        assert args[0].shape == (
            (observed.raw.shape[0] - 1) * 3 + 1,
            *observed.raw.shape[1:],
        )
        assert kwargs["distance"] == pytest.approx(0.4)
        result = deskew(*args, **kwargs)
        assert result.dtype == np.float32 and result.flags.c_contiguous
        deskewed.append((result.shape, sha256(memoryview(result)).hexdigest()))
        return result

    monkeypatch.setattr(processing, "orthogonal_deskew", observe)
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
    ((shape, digest),) = deskewed
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
    expected_shape, *_ = deskew_shape_estimator(
        ((observed.raw.shape[0] - 1) * 3 + 1, *observed.raw.shape[1:]),
        microscope.angle_deg,
        0.4,
        microscope.pixel_size_um,
        crop_after_deskew=False,
        downsample_factor=1,
    )
    assert saved.dtype == np.float32 and saved.shape[-3:] == shape == tuple(
        expected_shape
    )
    assert np.isfinite(saved).all() and np.all(saved >= 0)
    assert sha256(memoryview(saved[0, 0])).hexdigest() == digest
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
    assert settings["read_noise"] == microscope.camera.read_noise_e
    assert settings["psf_sha256"] is None
    options = {
        "deconvolve": True,
        "max_projection": False,
        "save_float32": True,
        "z_downsample_level": 1,
        "decon_crop_scan": 12,
        "resume": True,
    }
    before = len(production_forward_predictions)
    invoke_cli("process", dataset.path, **options)
    assert len(deskewed) == 1
    assert len(production_forward_predictions) == before
    np.testing.assert_array_equal(collection.arrays[0].read().result(), saved)
    assert {
        path.relative_to(dataset.path): sha256(path.read_bytes()).digest()
        for path in dataset.path.rglob("*")
        if path.is_file()
    } == original
    parameters = deepcopy(settings["psf_parameters"])
    parameters["support_fwhms"] *= 1.2
    update_group_attributes(
        dataset.path,
        {"opm_processing": {"deconvolution": {"psf_parameters": parameters}}},
    )
    with pytest.raises(ValueError, match=r"incompatible|configuration"):
        invoke_cli("process", dataset.path, **options)
    assert len(deskewed) == 1
    assert len(production_forward_predictions) == before
    np.testing.assert_array_equal(collection.arrays[0].read().result(), saved)
