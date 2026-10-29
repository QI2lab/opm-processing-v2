"""Resolve production PSFs directly from unmodified opm-v2 writer output."""

import numpy as np
import pytest
from yaozarrs import open_group

from opm_processing.dataio.acquisition import (
    ChannelMetadata,
    inspect_acquisition,
    open_acquisition_datastore,
)
from opm_processing.dataio.deconvolution import (
    emission_wavelengths_from_metadata,
    load_deconvolution_psfs,
    scan_upsample_from_metadata,
)


@pytest.mark.unit
def test_actual_acquisition_without_processing_metadata_generates_psf(
    simulated_camera, opm_v2_acquisition_factory
):
    """Raw writer output supplies geometry while processing supplies default optics."""
    observed = simulated_camera((17, 32, 36), step=1.2)
    microscope = observed.microscope
    dataset = opm_v2_acquisition_factory(
        observed.raw[None, None, None],
        channels=(ChannelMetadata(0, "561 + 637", None, 10.0, 10.0),),
        scan_axis_step_um=1.2,
        camera_offset=microscope.camera.offset_adu,
        camera_conversion=microscope.camera.electrons_per_adu,
    )
    attributes = dict(open_group(dataset.path).attrs)
    assert "opm_processing" not in attributes
    assert attributes["opm_v2"]["storage_backend"] == "tensorstore"
    metadata = inspect_acquisition(dataset.path)
    assert not metadata.deconvolution
    assert metadata.channel_names == ("561 + 637",)
    assert scan_upsample_from_metadata(metadata) == 3
    assert emission_wavelengths_from_metadata(metadata) == (0.637,)
    np.testing.assert_array_equal(
        open_acquisition_datastore(metadata).read().result(), dataset.raw_data
    )
    kernels, parameters = load_deconvolution_psfs(metadata, reconstruction_step_um=0.4)
    assert parameters["lens"]["NA"] == 1.35
    (kernel,) = kernels
    assert kernel.ndim == 3 and all(size % 2 for size in kernel.shape)
    assert np.all(np.isfinite(kernel) & (kernel >= 0)) and kernel.sum() > 0
    np.testing.assert_allclose(
        kernel,
        kernel[..., ::-1],
        rtol=1e-6,
        atol=8 * np.finfo(np.float32).eps * kernel.max(),
    )
    assert dict(open_group(dataset.path).attrs) == attributes
