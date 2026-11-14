"""GPU illumination fitting and acquisition-to-estimator integration."""

import importlib

import numpy as np
import pytest
from scipy.special import kl_div

from opm_processing.dataio.live import read_lifecycle_event
from opm_processing.dataio.position_collection import open_position_collection
from opm_processing.imageprocessing.flatfield import estimate_illuminations

pytestmark = pytest.mark.usefixtures("torch_gpu")


@pytest.mark.unit
@pytest.mark.parametrize(
    "topology,decision,storage",
    [
        pytest.param("single", None, False, id="array-single-all"),
        pytest.param("single", -1, True, id="store-single-unchecked-excluded"),
        pytest.param("tiles", None, True, id="store-tiles-all"),
        pytest.param("tiles", -1, False, id="array-tiles-unchecked-excluded"),
        pytest.param("depths", None, False, id="array-depths-all"),
        pytest.param("depths", -1, True, id="store-depths-unchecked-excluded"),
    ],
)
def test_fit_samples_preserve_calibration_and_depth_groups(
    illumination_fit_case, topology, decision
):
    """Keep time-zero camera units and every selected depth in one fit per channel."""
    case = illumination_fit_case
    mask = None
    selected = np.arange(len(case.levels))
    if decision is not None:
        mask = np.ones(case.raw.shape[:3], np.int8)
        if len(selected) > 1:
            mask[0, 1] = decision
            selected = selected[selected != 1]
        else:
            mask[1] = decision  # Later times are not used for estimation.
    actual = estimate_illuminations(
        case.datastore,
        100,
        0.5,
        case.coordinates,
        detector_gain_x=case.gain,
        signal_mask=mask,
    )
    expected_levels = [
        np.median(case.levels[selected, channel], axis=1)
        if len(selected) > 1
        else case.levels[selected[0], channel]
        for channel in range(2)
    ]
    for images, levels in zip(case.calls, expected_levels, strict=True):
        np.testing.assert_allclose(
            images,
            np.broadcast_to(levels[:, None, None], images.shape),
            rtol=4 * np.finfo(np.float32).eps,
            atol=0,
        )
    assert actual.shape == ((2, 5, 7) if case.coordinates is None else (1, 2, 5, 7))
    np.testing.assert_allclose(actual, 1, rtol=4 * np.finfo(np.float32).eps, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "illumination_observations", ("depths", "masked", "tiles"), indirect=True
)
def test_real_estimator_predicts_known_detector_illumination(illumination_observations):
    """Estimate identifiable fields within a predefined absolute camera accuracy target."""
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
        assert np.all(
            np.isfinite(prediction)
            & (prediction >= envelope.lower)
            & (prediction <= envelope.upper)
        )
        assert np.any((case.source < envelope.lower) | (case.source > envelope.upper))
        assert (2 * kl_div(truth, prediction)).sum() < (
            2 * kl_div(truth, case.source)
        ).sum()
        if case.detector_x is not None:
            stripe = np.s_[:, case.detector_x - 3 : case.detector_x + 4]
            assert (2 * kl_div(truth[stripe], prediction[stripe])).sum() < (
                2 * kl_div(truth[stripe], case.source)
            ).sum()


@pytest.mark.integration
def test_live_wait_estimates_illumination_after_acquisition_completes(
    live_wait_case,
    monkeypatch,
    assert_roi_point_accuracy,
    invoke_cli,
):
    """Wait for completed acquisition and chunks before fitting illumination."""
    processing = importlib.import_module("opm_processing.process")
    from opm_processing.imageprocessing.opmtools import deskew_coverage

    case = live_wait_case
    estimates = []
    estimate_in_worker = processing.call_estimate_illuminations

    def estimate_after_completion(store, offset, conversion, positions, gain, mask):
        """Estimate illumination only after the recorded acquisition completes."""
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
    invoke_cli(
        "process",
        case.dataset.path.parent,
        live_wait=True,
        save_float32=True,
        output=case.destination,
        max_projection=False,
        create_fused_max_projection=False,
        z_downsample_level=1,
        flatfield_correction=True,
    )
    assert case.sleeps == [30.0, 30.0]
    assert estimates == [True]
    field = processing.read_current_flatfield(
        case.field_path, (1, 1, *case.raw.shape[-2:])
    )
    assert field is not None
    assert np.all(np.isfinite(field) & (field > 0))
    output = open_position_collection(case.destination / "sample_deskewed.ome.zarr")
    data = output.arrays[0].read().result()
    span = (case.raw.shape[-3] - 1) * case.scene.microscope.scan_step_um
    np.testing.assert_allclose(
        output.spatial_origins_zyx_um[0], (0, -span, 0), atol=0.0005, rtol=0
    )
    np.testing.assert_allclose(
        output.voxel_size_um, (0.115, 0.115, 0.115), atol=0, rtol=0
    )
    microscope = case.scene.microscope
    coverage = deskew_coverage(
        case.raw.shape[-3:],
        data.shape[-3:],
        microscope.angle_deg,
        microscope.scan_step_um,
        microscope.pixel_size_um,
    )
    support = (coverage[..., None] > 0) & (
        np.arange(data.shape[-1]) < case.raw.shape[-1]
    )
    np.testing.assert_array_equal(data[0, 0][~support], 0)
    assert_roi_point_accuracy(
        data[case.sample_timepoint : case.sample_timepoint + 1],
        case.scene,
        output.spatial_origins_zyx_um[0],
        output.voxel_size_um,
        raw_origin=(0, -span, 0),
    )
