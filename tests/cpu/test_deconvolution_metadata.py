"""Resolve acquired calibration and processing defaults in physical units."""

from dataclasses import replace

import numpy as np
import pytest
from yaozarrs import open_group

from opm_processing.dataio.acquisition import inspect_acquisition
from opm_processing.dataio.deconvolution import (
    detector_gain_from_metadata,
    emission_wavelengths_from_metadata,
    load_deconvolution_psfs,
    psf_parameters_from_metadata,
    read_noise_from_metadata,
    scan_upsample_from_metadata,
)
from opm_processing.dataio.position_collection import (
    open_position_collection,
    update_group_attributes,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "step,factor",
    [(0.2, 1), (0.4, 1), (0.8, 2), (1.2, 3), (2.4, 6), (1.200000000000001, 3)],
)
def test_scan_refinement_uses_recorded_spacing(acquisition_metadata, step, factor):
    """Keep fine acquisition sampling and refine coarse steps to 0.4 micrometers."""
    acquisition = replace(acquisition_metadata, scan_axis_step_um=step)
    assert scan_upsample_from_metadata(acquisition) == factor
    assert step / factor == pytest.approx(min(step, 0.4))


@pytest.mark.parametrize("step", [None, 0, -1, np.nan, np.inf, 0.6])
def test_scan_refinement_rejects_unsupported_spacing(acquisition_metadata, step):
    """Require valid spacing that the integer acquired-plane operator can represent."""
    with pytest.raises(ValueError, match=r"scan.*step|scan_axis_step_um"):
        scan_upsample_from_metadata(
            replace(acquisition_metadata, scan_axis_step_um=step)
        )


def test_planar_data_does_not_require_scan_spacing(acquisition_metadata):
    """A single acquired plane needs no scan-grid refinement."""
    shape = (*acquisition_metadata.shape[:3], 1, *acquisition_metadata.shape[-2:])
    assert (
        scan_upsample_from_metadata(
            replace(acquisition_metadata, shape=shape, scan_axis_step_um=None)
        )
        == 1
    )


@pytest.mark.parametrize("suffix", [".npy", ".ome.tif"])
def test_measured_psfs_preserve_pixels_without_theoretical_calibration(
    tmp_path, acquisition_metadata, suffix
):
    """Load measured optical samples unchanged using NumPy and tifffile readers."""
    from tifffile import imwrite

    kernel = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    kernel /= kernel.sum()
    path = tmp_path / f"measured_psf{suffix}"
    if suffix == ".npy":
        np.save(path, kernel)
    else:
        imwrite(path, kernel, photometric="minisblack", metadata={"axes": "ZYX"})
    kernels, parameters = load_deconvolution_psfs(acquisition_metadata, [path])
    np.testing.assert_array_equal(kernels[0], kernel)
    assert parameters is None


@pytest.mark.parametrize("count", [0, 2])
def test_measured_psf_count_must_match_recorded_channels(
    tmp_path, acquisition_metadata, count
):
    """Reject missing or extra channel kernels before opening their files."""
    with pytest.raises(ValueError, match="one PSF per channel"):
        load_deconvolution_psfs(
            acquisition_metadata,
            [tmp_path / f"channel_{index}.npy" for index in range(count)],
        )


@pytest.mark.parametrize(
    "location, calibration, expected",
    [
        ("root", None, 0),
        ("root", {"rms": 0, "units": "electrons"}, 0),
        ("root", {"rms": 1.6, "units": "electrons"}, 1.6),
        ("root", {"rms": 6.4, "units": "ADU"}, 1.6),
        ("root", {"rms": 0.7, "units": "calibrated-detector-counts"}, 0.7),
        ("read_noise_e", 1.6, 1.6),
        ("read_noise_adu", 6.4, 1.6),
    ],
)
def test_noise_is_extracted_and_calibrated_once(
    deconvolution_acquisition, location, calibration, expected
):
    """Read RMS noise from stored calibration and convert its units once."""
    case = deconvolution_acquisition
    root = open_group(case.path)
    if location == "root":
        if calibration is not None:
            update_group_attributes(
                case.path,
                {"opm_processing": {"deconvolution": {"read_noise": calibration}}},
            )
    else:
        writer = root["0"].attrs["ome_writers"]
        writer["frame_metadata"][0]["mda_event"]["metadata"]["Camera"][location] = (
            calibration
        )
        update_group_attributes(case.path / "0", {"ome_writers": writer})
    acquisition = inspect_acquisition(case.path)
    assert read_noise_from_metadata(acquisition) == pytest.approx(expected)
    assert acquisition.pixel_size_um == case.metadata.pixel_size_um
    assert acquisition.scan_axis_step_um == case.metadata.scan_axis_step_um


@pytest.mark.parametrize(
    "calibration",
    [
        {"rms": -1, "units": "electrons"},
        {"rms": np.nan, "units": "electrons"},
        {"rms": np.inf, "units": "electrons"},
        {"rms": 1e30, "units": "electrons"},
        {"rms": 1.6, "units": "unknown"},
    ],
)
def test_invalid_recorded_noise_is_not_silently_replaced(
    acquisition_metadata, calibration
):
    """Reject invalid RMS noise instead of selecting a different noise model."""
    acquisition = replace(
        acquisition_metadata, deconvolution={"read_noise": calibration}
    )
    with pytest.raises(ValueError, match=r"[Rr]ead.noise"):
        read_noise_from_metadata(acquisition)


def test_as_acquired_data_uses_processing_optics(deconvolution_acquisition):
    """Raw stores need no processing attributes to select the silicone objective."""
    acquisition = inspect_acquisition(deconvolution_acquisition.path)
    assert not acquisition.deconvolution
    parameters = psf_parameters_from_metadata(acquisition)
    assert parameters["lens"]["NA"] == 1.35
    assert parameters["lens"]["ni"] == 1.4
    assert parameters["skewed_depth_um"] == 0.0
    assert emission_wavelengths_from_metadata(acquisition) == (0.488,)
    parameters["lens"]["NA"] = 0.5
    assert psf_parameters_from_metadata(acquisition)["lens"]["NA"] == 1.35


@pytest.mark.parametrize(
    "labels, expected",
    [
        (["488nm", "561nm", "637nm"], (0.488, 0.561, 0.637)),
        (["561 + 637"], (0.637,)),
        (["637nm + 561 NM"], (0.637,)),
        (["488", "561 + 637"], (0.488, 0.637)),
    ],
)
def test_channel_wavelength_approximation_preserves_storage_order(
    acquisition_metadata, labels, expected
):
    """Combined excitation remains one camera channel with its longest wavelength."""
    template = acquisition_metadata.channels[0]
    acquisition = replace(
        acquisition_metadata,
        channels=tuple(
            replace(template, index=index, name=label, wavelength_nm=None)
            for index, label in enumerate(labels)
        ),
    )
    assert emission_wavelengths_from_metadata(acquisition) == expected


def test_unresolved_channel_requires_wavelength_calibration(acquisition_metadata):
    """Name an unresolvable channel rather than guessing its spectrum."""
    channel = replace(
        acquisition_metadata.channels[0], name="beads", wavelength_nm=None
    )
    acquisition = replace(acquisition_metadata, channels=(channel,))
    with pytest.raises(ValueError, match=r"beads.*emission_wavelengths_nm"):
        emission_wavelengths_from_metadata(acquisition)
    channel = replace(channel, wavelength_nm=615)
    assert emission_wavelengths_from_metadata(
        replace(acquisition, channels=(channel,))
    ) == (0.615,)


@pytest.mark.parametrize(
    "values", [None, 615, [], [0], [-615], [np.nan], [np.inf], [561, 637], ["unknown"]]
)
def test_invalid_explicit_wavelengths_do_not_use_defaults(acquisition_metadata, values):
    """Explicit calibration takes priority, including validation of invalid data."""
    acquisition = replace(
        acquisition_metadata, deconvolution={"emission_wavelengths_nm": values}
    )
    with pytest.raises(ValueError, match="wavelength"):
        emission_wavelengths_from_metadata(acquisition)


@pytest.mark.parametrize("projection", [False, True])
def test_as_acquired_combined_channel_psf_has_lateral_symmetry(
    deconvolution_acquisition, projection
):
    """A centered emitter has lateral reflection symmetry with default optics."""
    acquisition = inspect_acquisition(deconvolution_acquisition.path)
    channel = replace(acquisition.channels[0], name="561 + 637", wavelength_nm=None)
    acquisition = replace(acquisition, channels=(channel,), scan_axis_step_um=1.2)
    kernels, parameters = load_deconvolution_psfs(
        acquisition, projection=projection, reconstruction_step_um=0.4
    )
    assert parameters["lens"]["NA"] == 1.35
    assert len(kernels) == 1
    kernel = kernels[0]
    assert kernel.ndim == 3
    assert (kernel.shape[0] == 1) == projection
    assert all(size % 2 == 1 for size in kernel.shape)
    assert np.all(np.isfinite(kernel) & (kernel >= 0))
    assert kernel.sum() > 0
    np.testing.assert_allclose(
        kernel,
        kernel[..., ::-1],
        rtol=1e-6,
        atol=8 * np.finfo(np.float32).eps * kernel.max(),
    )


@pytest.mark.parametrize("profile", [None, [0.7, 0.8, 0.9, 1, 1.1]])
def test_optical_calibration_round_trips_without_instrument_defaults(
    deconvolution_acquisition, simulation_parameters, profile
):
    """Preserve recorded optics, emission wavelengths and optional detector gain."""
    case = deconvolution_acquisition
    parameters = {"lens": dict(simulation_parameters["microscope"]["psf_parameters"])}
    settings = {"psf_parameters": parameters, "emission_wavelengths_nm": [615]}
    if profile is not None:
        settings["detector_gain_x"] = profile
    update_group_attributes(case.path, {"opm_processing": {"deconvolution": settings}})
    acquisition = inspect_acquisition(case.path)
    assert psf_parameters_from_metadata(acquisition) == parameters
    assert emission_wavelengths_from_metadata(acquisition) == (0.615,)
    gain = detector_gain_from_metadata(acquisition, case.raw_data.shape[-1])
    if profile is None:
        assert gain is None
    else:
        np.testing.assert_array_equal(gain, np.asarray(profile, np.float32))


@pytest.mark.parametrize("profile", [[1, 1], [1, 1, 0, 1, 1], [1, 1, np.nan, 1, 1]])
def test_invalid_detector_calibration_is_rejected(acquisition_metadata, profile):
    """Reject incomplete, nonpositive or nonfinite detector gain profiles."""
    acquisition = replace(
        acquisition_metadata, deconvolution={"detector_gain_x": profile}
    )
    with pytest.raises(ValueError, match="detector_gain_x"):
        detector_gain_from_metadata(acquisition, 5)


@pytest.mark.parametrize(
    "field",
    [
        "pixel_size_um",
        "angle_deg",
        "camera_offset",
        "camera_conversion",
        "scan_axis_step_um",
    ],
)
def test_missing_required_calibration_is_an_explicit_error(acquisition_metadata, field):
    """Name the missing physical calibration in the processing error."""
    acquisition = replace(acquisition_metadata, **{field: None})
    with pytest.raises(ValueError, match=field):
        acquisition.validate_processing_calibration()


@pytest.mark.parametrize("unit, multiplier", [("micrometer", 1), ("nanometer", 1000)])
def test_camera_pitch_uses_recorded_ngff_units(
    deconvolution_acquisition, unit, multiplier
):
    """Convert the recorded NGFF camera pitch into micrometers."""
    case = deconvolution_acquisition
    image = open_group(case.path)["0"]
    ome = image.attrs["ome"]
    multiscale = ome["multiscales"][0]
    scales = multiscale["datasets"][0]["coordinateTransformations"][0]["scale"]
    for index, axis in enumerate(multiscale["axes"]):
        if axis["name"] in ("x", "y"):
            axis["unit"] = unit
            scales[index] = case.metadata.pixel_size_um * multiplier
    update_group_attributes(case.path / "0", {"ome": ome})
    assert inspect_acquisition(case.path).pixel_size_um == pytest.approx(
        case.metadata.pixel_size_um
    )


@pytest.mark.parametrize("unit, multiplier", [("micrometer", 1), ("nanometer", 1000)])
def test_collection_reads_ngff_spacing_and_origin_in_physical_units(
    deconvolution_acquisition, unit, multiplier
):
    """NGFF calibration is authoritative when the XML companion is stale."""
    path = deconvolution_acquisition.path
    group = open_group(path)["0"]
    metadata = group.ome_metadata().model_dump(mode="json", exclude_none=True)
    multiscale = metadata["multiscales"][0]
    transforms = multiscale["datasets"][0]["coordinateTransformations"]
    spacing, origin = (0.31, 0.12, 0.12), (-2.3, 4.5, 6.7)
    for axis in multiscale["axes"][-3:]:
        axis["unit"] = unit
    transforms[0]["scale"][-3:] = [value * multiplier for value in spacing]
    transforms[1]["translation"][-3:] = [value * multiplier for value in origin]
    update_group_attributes(path / "0", {"ome": metadata})
    collection = open_position_collection(path)
    np.testing.assert_allclose(collection.voxel_size_um, spacing, atol=1e-14)
    np.testing.assert_allclose(collection.spatial_origins_zyx_um, [origin], atol=1e-14)
