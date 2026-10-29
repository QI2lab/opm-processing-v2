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
    opm_v2_projection_zarr,
    tmp_path: Path,
    output_flatfield: bool,
) -> None:
    """Verify source reuse and output precedence through saved pixel values."""
    from opm_processing.dataio.acquisition import acquisition_stem
    from opm_processing.dataio.position_collection import open_position_collection
    from opm_processing.process import process, write_flatfield

    acquisition = opm_v2_projection_zarr
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


@pytest.mark.unit
@pytest.mark.parametrize("scan_planes", [1, 4])
def test_processing_applies_the_same_field_at_every_depth(
    tmp_path, constant_illumination_acquisition, scan_planes
):
    """Calibrate every channel and depth against analytically known photons."""
    from opm_processing.imageprocessing.opmtools import orthogonal_deskew
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
    suffix = "2d" if scan_planes == 1 else "deskewed"
    corrected = open_position_collection(
        tmp_path / "corrected" / f"sample_{suffix}.ome.zarr"
    )
    for position, array in enumerate(corrected.arrays):
        for time, channel in np.ndindex(2, 2):
            value = (
                case.photons[time, position, channel] / case.fields[0, channel, 0, 0]
            )
            expected = (
                np.full((1, 8, 12), value, np.float32)
                if scan_planes == 1
                else orthogonal_deskew(
                    np.full((scan_planes, 8, 12), value, dtype=np.float32),
                    distance=acquisition.metadata.scan_axis_step_um,
                    pixel_size=acquisition.metadata.pixel_size_um,
                    theta=acquisition.metadata.angle_deg,
                    downsample_factor=1,
                )
            )
            np.testing.assert_allclose(
                array[time, channel].read().result(),
                expected,
                rtol=4 * np.finfo(np.float32).eps,
                atol=0,
            )


@pytest.mark.unit
@pytest.mark.parametrize(
    "illumination_observations", ("maximum", "masked", "tiles"), indirect=True
)
def test_real_estimator_predicts_known_detector_illumination(illumination_observations):
    """Known illumination predicts calibration photons better than unit sensitivity."""
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
    for expected, estimated in zip(case.expected, actual, strict=True):
        if np.all(expected == 1):
            np.testing.assert_array_equal(estimated, 1)
            continue
        if case.mask is not None:
            # An excluded depth must not floor known dim detector regions to unity.
            assert np.any(estimated[expected < 1] < 1)
        truth = case.source * expected
        prediction = case.source * estimated
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


@pytest.mark.unit
def test_live_wait_defers_processing_and_selects_flatfield(live_wait_case, monkeypatch):
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
    actual = (
        open_position_collection(case.destination / "sample_deskewed.ome.zarr")
        .arrays[0][0, 0]
        .read()
        .result()
    )
    expected = case.expected
    if case.kind == "estimate":
        from opm_processing.imageprocessing.opmtools import orthogonal_deskew

        expected = orthogonal_deskew(
            np.maximum(case.raw[0, 0, 0, ::-1].astype(np.float32) - 100, 0)
            * 0.24
            / field[0, 0],
            theta=30,
            distance=0.4,
            pixel_size=0.115,
            downsample_factor=1,
        )
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=4 * np.finfo(np.float32).eps,
        atol=1e-5 if case.kind == "estimate" else 0,
    )
