"""Illumination function units and simulated processing CLI integration."""

import importlib
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts
from scipy.special import kl_div
from typer.testing import CliRunner

from opm_processing.dataio.live import read_lifecycle_event
from opm_processing.dataio.position_collection import open_position_collection
from opm_processing.imageprocessing.flatfield import estimate_illuminations
from opm_processing.process import app


@pytest.mark.integration
@pytest.mark.parametrize("output_flatfield", (False, True))
def test_processing_uses_existing_flatfield_with_separate_output(
    projection_illumination_case,
    tmp_path: Path,
    output_flatfield: bool,
) -> None:
    """Verify source reuse and output precedence through saved pixel values."""
    from opm_processing.dataio.acquisition import acquisition_stem
    from opm_processing.dataio.position_collection import open_position_collection
    from opm_processing.process import process, write_flatfield

    case = projection_illumination_case
    acquisition = case.dataset
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    filename = f"{acquisition_stem(acquisition.path)}_flatfield.ome.tif"
    field_shape = (1, acquisition.raw_data.shape[2], *acquisition.raw_data.shape[-2:])
    source_field = np.full(field_shape, 2.0, dtype=np.float32)
    selected_field = source_field
    write_flatfield(
        acquisition.path.parent / filename, source_field, acquisition.pixel_size_um
    )
    if output_flatfield:
        selected_field = np.full(field_shape, 1.25, dtype=np.float32)
        write_flatfield(
            output_dir / filename, selected_field, acquisition.pixel_size_um
        )

    process(
        root_path=acquisition.path,
        output=output_dir,
        flatfield_correction=True,
        deconvolve=False,
        save_float32=True,
        write_fused_max_projection_tiff=False,
    )
    saved = (
        open_position_collection(
            output_dir / f"{acquisition_stem(acquisition.path)}_projection.ome.zarr"
        )
        .arrays[0]
        .read()
        .result()
    )
    calibrated = (
        acquisition.raw_data[:, 0].astype(np.float32) - acquisition.camera_offset
    ) * acquisition.camera_conversion
    expected = (calibrated / selected_field[0])[:, :, np.newaxis]
    np.testing.assert_allclose(saved, expected, rtol=1e-6)
    # The original object and optical model predict the calibrated photons;
    # neither raw counts nor a second processing pass supplies this truth.
    mean = case.means / selected_field[0]
    observations = saved[:, :, 0]
    true_deviance = (2 * kl_div(observations, mean)).sum()
    for incorrect_brightness in (0.5, 1.5):
        assert (
            true_deviance
            < (2 * kl_div(observations, mean * incorrect_brightness)).sum()
        )


@pytest.mark.unit
@pytest.mark.parametrize("channels", [1, 3])
def test_shared_flatfield_round_trips_and_is_reused_at_multiple_depths(
    tmp_path: Path, channels
):
    """The saved maximum field is reused unchanged for every stage depth."""
    from opm_processing.process import (
        load_or_estimate_flatfield,
        read_current_flatfield,
        write_flatfield,
    )

    path = tmp_path / "flatfield.ome.tif"
    flatfields = np.linspace(0.5, 1.5, channels * 8 * 12, dtype=np.float32).reshape(
        1, channels, 8, 12
    )
    write_flatfield(path, flatfields, 0.115)
    actual = read_current_flatfield(path, flatfields.shape)
    assert actual is not None
    np.testing.assert_array_equal(actual, flatfields)
    actual = load_or_estimate_flatfield(
        path,
        ts.array(np.zeros((1, 2, channels, 4, 8, 12), dtype=np.uint16)),
        100,
        0.25,
        0.115,
        np.asarray(((10, 20, 30), (0, 20, 30))),
        detector_gain_x=None,
        signal_threshold=None,
        minimum_signal_fraction=0.01,
        signal_mask=None,
    )
    np.testing.assert_array_equal(actual, flatfields)


@pytest.mark.unit
def test_old_per_depth_flatfield_is_not_reused(tmp_path):
    """Replace obsolete depth-specific fields with the shared-depth estimator."""
    from tifffile import imwrite

    from opm_processing.process import read_current_flatfield

    path = tmp_path / "old-flatfield.ome.tif"
    imwrite(
        path,
        np.ones((2, 1, 8, 12), dtype=np.float32),
        photometric="minisblack",
        metadata={"axes": "ZCYX"},
        software="opm-processing/basicpy-stage-depth-camera-contract-v10",
    )
    assert read_current_flatfield(path, (1, 1, 8, 12)) is None


@pytest.mark.integration
@pytest.mark.parametrize("scan_planes", [1, 4])
def test_processing_applies_the_same_field_at_every_depth(
    tmp_path,
    constant_illumination_acquisition,
    scan_planes,
    assert_uniform_photon_accuracy,
):
    """Calibrate every channel and depth against analytically known photons."""
    from opm_processing.process import process, write_flatfield

    case = constant_illumination_acquisition
    acquisition = case.dataset
    write_flatfield(
        tmp_path / "sample_flatfield.ome.tif",
        case.fields,
        acquisition.metadata.pixel_size_um,
    )
    process(
        root_path=acquisition.path,
        output=tmp_path / "corrected",
        flatfield_correction=True,
        deconvolve=False,
        save_float32=True,
        z_downsample_level=1,
        max_projection=False,
        create_fused_max_projection=False,
        write_fused_max_projection_tiff=False,
    )
    suffix = "projection" if scan_planes == 1 else "deskewed"
    corrected = open_position_collection(
        tmp_path / "corrected" / f"sample_{suffix}.ome.zarr"
    )
    for position, array in enumerate(corrected.arrays):
        assert_uniform_photon_accuracy(
            array.read().result(), case.scene, position=position
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "illumination_observations", ("maximum", "masked", "tiles"), indirect=True
)
def test_real_estimator_predicts_known_detector_illumination(illumination_observations):
    """Estimate identifiable fields within a predefined absolute camera accuracy target."""
    from tests.simulation.camera_likelihood import camera_sampling_accuracy

    case = illumination_observations
    actual = estimate_illuminations(
        case.datastore,
        case.offset,
        case.conversion,
        case.coordinates,
        signal_mask=case.mask,
    )[0]
    assert actual.shape == case.expected.shape
    assert np.all(np.isfinite(actual) & (actual > 0))
    for expected, estimated, envelope in zip(
        case.expected, actual, case.envelopes, strict=True
    ):
        if np.all(expected == 1):
            np.testing.assert_array_equal(estimated, 1)
            continue
        if case.mask is not None:
            # An excluded depth must not floor known dim detector regions to unity.
            assert np.any(estimated[expected < 1] < 1)
        truth = case.source * expected
        prediction = case.source * estimated
        # This is a camera-space accuracy requirement for a nonlinear estimator,
        # rather than a claim that BaSiC itself performs convex interpolation.
        aperture = np.ones(expected.shape, bool)
        measured = camera_sampling_accuracy(prediction, truth, envelope, aperture)
        assert measured.accepted, measured
        uncorrected = camera_sampling_accuracy(
            np.full_like(truth, case.source), truth, envelope, aperture
        )
        assert not uncorrected.accepted, uncorrected
        assert (2 * kl_div(truth, prediction)).sum() < (
            2 * kl_div(truth, case.source)
        ).sum()
        if case.detector_x is not None:
            stripe = np.s_[:, case.detector_x - 3 : case.detector_x + 4]
            assert (2 * kl_div(truth[stripe], prediction[stripe])).sum() < (
                2 * kl_div(truth[stripe], case.source)
            ).sum()


@pytest.mark.unit
@pytest.mark.parametrize("level", (0, -1, np.nan))
def test_invalid_cached_fields_are_not_reused(tmp_path, level):
    """Reject invalid fields at write time and refresh invalid existing caches."""
    from tifffile import imwrite

    from opm_processing.process import (
        FLATFIELD_SOFTWARE,
        read_current_flatfield,
        write_flatfield,
    )

    path = tmp_path / "invalid.ome.tif"
    field = np.full((1, 1, 5, 7), level, np.float32)
    with pytest.raises(ValueError, match="finite and positive"):
        write_flatfield(path, field, 0.115)
    assert not path.exists()
    imwrite(
        path,
        field,
        photometric="minisblack",
        metadata={"axes": "ZCYX"},
        software=FLATFIELD_SOFTWARE,
    )
    assert read_current_flatfield(path, field.shape) is None


@pytest.mark.integration
def test_live_wait_defers_processing_and_selects_flatfield(
    live_wait_case, monkeypatch, assert_roi_point_accuracy, deskew_uniform_expectation
):
    """Wait for both completion and chunks, then reuse or estimate illumination."""
    processing = importlib.import_module("opm_processing.process")

    case = live_wait_case
    estimates = []
    estimate_in_worker = processing.call_estimate_illuminations

    def estimate_after_completion(store, offset, conversion, positions, gain, mask):
        """Estimate illumination only after the recorded acquisition completes."""
        assert case.kind == "estimate"
        assert (
            read_lifecycle_event(case.dataset.log, case.dataset.manifest.acquisition_id)
            == "completed"
        )
        np.testing.assert_array_equal(store.read().result(), case.raw)
        estimates.append(True)
        return estimate_in_worker(
            store,
            offset,
            conversion,
            positions,
            gain,
            mask,
        )

    monkeypatch.setattr(
        processing, "call_estimate_illuminations", estimate_after_completion
    )
    args = [
        str(case.dataset.path.parent),
        "--live-wait",
        "--save-float32",
        "--output",
        str(case.destination),
        "--no-max-projection",
        "--no-create-fused-max-projection",
        "--z-downsample-level",
        "1",
    ]
    if case.kind != "pending-chunks":
        args.append("--flatfield-correction")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.exception
    assert (
        case.sleeps
        == [30.0] * {"estimate": 2, "pending-chunks": 1, "cached": 0}[case.kind]
    )
    assert estimates == ([True] if case.kind == "estimate" else [])
    if case.kind == "pending-chunks":
        assert not case.field_path.exists()
    else:
        field = processing.read_current_flatfield(
            case.field_path, (1, 1, *case.raw.shape[-2:])
        )
        assert field is not None
        assert np.all(np.isfinite(field) & (field > 0))
        if case.kind == "cached":
            np.testing.assert_array_equal(field, 2)
        assert (
            "starting offline processing"
            if case.kind == "estimate"
            else "Using existing flatfield"
        ) in result.output
    output = open_position_collection(case.destination / "sample_deskewed.ome.zarr")
    data = output.arrays[0].read().result()
    span = (case.raw.shape[-3] - 1) * case.scene.microscope.scan_step_um
    np.testing.assert_allclose(
        output.spatial_origins_zyx_um[0], (0, -span, 0), atol=0.0005, rtol=0
    )
    np.testing.assert_allclose(
        output.voxel_size_um, (0.115, 0.115, 0.115), atol=0, rtol=0
    )
    if case.calibration_photons is not None:
        expected_calibration = deskew_uniform_expectation(
            case.raw.shape[-3:], case.calibration_photons, dtype=np.float32
        )
        camera = case.scene.microscope.camera
        sigma = np.sqrt(
            case.calibration_photons
            + camera.read_noise_e**2
            + camera.electrons_per_adu**2 / 12
        )
        gain = (
            2 * case.scene.microscope.pixel_size_um / case.scene.microscope.scan_step_um
        )
        # Positive interpolation cannot exceed the variance of one input
        # camera sample. Use the declared dose and six shot/read-noise SDs.
        np.testing.assert_allclose(
            data[0, 0], expected_calibration, rtol=0, atol=6 * gain * sigma
        )
        np.testing.assert_array_equal(data[0, 0][expected_calibration == 0], 0)
    assert_roi_point_accuracy(
        data[case.sample_timepoint : case.sample_timepoint + 1],
        case.scene,
        output.spatial_origins_zyx_um[0],
        output.voxel_size_um,
        raw_origin=(0, -span, 0),
        illumination=field[0, 0] if case.kind != "pending-chunks" else 1.0,
    )
