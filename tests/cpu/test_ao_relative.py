"""Keep one common modal reference throughout depth-dependent AO figures."""

import copy
from functools import partial

import numpy as np
import pytest
from scripts import ao_wavekit_phase
from scripts.ao_wavekit_phase import phase_from_modal_coeff as sdk_phase
from scripts.export_ao_relative import relative_regions

pytestmark = pytest.mark.unit

_phase = partial(sdk_phase, samples=512, norm=True, wavelength=0.488, out="radians")


def test_every_depth_subtracts_original_common_start_and_preserves_sources(
    ao_modal_chain, ao_analytical_phase
):
    """Later inherited starts must never replace the original shared baseline."""
    manifest, saved, reference = ao_modal_chain()
    original_manifest = copy.deepcopy(manifest)
    original_pupils = [region.pupil.copy() for region in saved]
    regions, provenance, records = relative_regions(
        manifest,
        list(reversed(saved)),
        ao_wavekit_phase,
    )
    assert len(regions) == len(records) == 6
    assert [(region.depth, region.index) for region in regions] == list(
        np.ndindex(3, 2)
    )
    originals = {(region.depth, region.index): region for region in saved}
    inputs = {
        (record["depth_index"], record["region_index"]): record
        for record in manifest["regions"]
    }
    for region, record in zip(regions, records, strict=True):
        coefficients = np.asarray(record["plotted_coefficients"])
        source = inputs[region.depth, region.index]
        expected = np.asarray(source["final_coefficients"]) - reference
        layer_delta = (
            np.asarray(source["final_coefficients"]) - source["starting_coefficients"]
        )
        np.testing.assert_array_equal(coefficients, expected)
        np.testing.assert_array_equal(record["plotted_coefficients"], expected)
        np.testing.assert_allclose(record["cumulative_coefficients"], expected)
        np.testing.assert_array_equal(record["layer_delta_coefficients"], layer_delta)
        analytical = ao_analytical_phase(expected)
        np.testing.assert_array_equal(region.pupil.mask, analytical.mask)
        np.testing.assert_allclose(
            region.pupil.compressed(), analytical.compressed(), rtol=0, atol=2e-12
        )
        assert region.result is originals[region.depth, region.index].result
        assert record["relative_rms_opd_nm"] == pytest.approx(
            1000 * np.linalg.norm(expected)
        )
        if region.depth:
            assert not np.allclose(coefficients, layer_delta)
            preceding = inputs[region.depth - 1, region.index]
            np.testing.assert_array_equal(
                source["starting_coefficients"], preceding["final_coefficients"]
            )
    assert provenance["kind"] == "common_acquisition_start"
    np.testing.assert_array_equal(provenance["reference_coefficients"], reference)
    assert provenance["reference_depth_index"] == 0
    assert provenance["reference_region_indices"] == [0, 1]
    assert provenance["quantity"].startswith("cumulative requested modal correction")
    assert manifest == original_manifest
    for before, after in zip(original_pupils, saved, strict=True):
        np.testing.assert_array_equal(before.data, after.pupil.data)
        np.testing.assert_array_equal(before.mask, after.pupil.mask)


def test_shrinking_full_command_can_require_growing_cumulative_correction(
    ao_modal_sequence_factory,
):
    """Warm starts carry full states; their signed changes add once per layer."""
    reference = np.array([0.3, 0.0, 0.0])
    full_commands = np.array([[0.25, 0, 0], [0.2, 0, 0], [0.1, 0, 0]])
    starts = np.vstack((reference, full_commands[:-1]))
    case = ao_modal_sequence_factory(
        reference, {(depth, 0): final for depth, final in enumerate(full_commands)}
    )
    manifest, saved = case.manifest, case.saved

    _, _, records = relative_regions(manifest, saved, ao_wavekit_phase)
    cumulative = np.cumsum(full_commands - starts, axis=0)
    np.testing.assert_allclose(cumulative[:, 0], [-0.05, -0.1, -0.2])
    np.testing.assert_allclose(
        [r["cumulative_coefficients"] for r in records], cumulative
    )
    np.testing.assert_allclose([r["plotted_coefficients"] for r in records], cumulative)
    np.testing.assert_allclose(
        [r["layer_delta_coefficients"] for r in records], full_commands - starts
    )
    np.testing.assert_allclose(
        [r["relative_rms_opd_nm"] for r in records], [50, 100, 200]
    )
    assert np.all(np.diff(np.linalg.norm(full_commands, axis=1)) < 0)
    assert np.all(np.diff(np.linalg.norm(cumulative, axis=1)) > 0)


def test_signed_layer_changes_cancel_in_the_cumulative_command(ao_modal_chain):
    """Adding absolute mode amplitudes or layer RMS values would overcount."""
    manifest, saved, _ = ao_modal_chain()
    commands = ([0.4, -0.2, 0.1], [0.2, -0.2, 0.1], [0.35, -0.2, 0.1])
    prior = [0.3, -0.2, 0.1]
    for depth, final in enumerate(commands):
        record = manifest["regions"][depth]
        record["starting_coefficients"] = list(prior)
        record["final_coefficients"] = list(final)
        saved[depth].pupil = _phase(final)
        prior = final

    _, _, records = relative_regions(manifest, saved, ao_wavekit_phase)
    selected = [r for r in records if r["region_index"] == 0]
    np.testing.assert_allclose(
        [r["cumulative_coefficients"][0] for r in selected], [0.1, -0.1, 0.05]
    )
    assert selected[-1]["relative_rms_opd_nm"] == pytest.approx(50)
    assert sum(
        np.linalg.norm(r["layer_delta_coefficients"]) for r in selected
    ) == pytest.approx(0.45)


def test_full_command_preserves_nonzero_initial_spherical_correction(
    ao_modal_sequence_factory,
):
    """Full mode retains the initial radial terms rather than subtracting them."""
    baseline = np.zeros(32)
    baseline[7], baseline[14] = 0.15, -0.04
    finals = []
    for depth in range(3):
        final = baseline.copy()
        final[7] += 0.01 * (depth + 1)
        final[5] = 0.02 * (depth + 1)
        finals.append(final)
    case = ao_modal_sequence_factory(
        baseline, {(depth, 0): final for depth, final in enumerate(finals)}
    )
    manifest, saved = case.manifest, case.saved
    regions, provenance, plotted = relative_regions(
        manifest,
        saved,
        ao_wavekit_phase,
        include_starting=True,
    )
    assert plotted[0]["plotted_coefficients"][7] == pytest.approx(0.16)
    assert plotted[0]["plotted_coefficients"][14] == pytest.approx(-0.04)
    assert plotted[0]["cumulative_coefficients"][7] == pytest.approx(0.01)
    assert plotted[0]["cumulative_coefficients"][14] == 0
    for region, record, final in zip(regions, plotted, finals, strict=True):
        np.testing.assert_array_equal(record["plotted_coefficients"], final)
        np.testing.assert_allclose(record["cumulative_coefficients"], final - baseline)
        assert record["plotted_rms_opd_nm"] == pytest.approx(
            1000 * np.linalg.norm(final)
        )
        assert record["relative_rms_opd_nm"] == pytest.approx(
            1000 * np.linalg.norm(final - baseline)
        )
        np.testing.assert_array_equal(region.pupil.data, saved[region.depth].pupil.data)
        np.testing.assert_array_equal(region.pupil.mask, saved[region.depth].pupil.mask)
    assert provenance["kind"] == "configured_mirror_reference"
    np.testing.assert_array_equal(provenance["reference_coefficients"], np.zeros(32))
    np.testing.assert_array_equal(
        provenance["original_starting_coefficients"], baseline
    )


def test_full_command_is_starting_state_plus_signed_layer_changes_once(ao_modal_chain):
    """Including the starting state changes the plot, not the accumulated deltas."""
    manifest, saved, baseline = ao_modal_chain()
    reader = ao_wavekit_phase
    _, _, relative = relative_regions(manifest, saved, reader)
    _, _, full = relative_regions(manifest, saved, reader, include_starting=True)
    manual = {index: baseline.copy() for index in range(2)}
    for record, relative_record in zip(full, relative, strict=True):
        index = record["region_index"]
        manual[index] += (
            np.asarray(record["final_coefficients"]) - record["starting_coefficients"]
        )
        np.testing.assert_allclose(record["plotted_coefficients"], manual[index])
        np.testing.assert_array_equal(
            record["plotted_coefficients"], record["final_coefficients"]
        )
        np.testing.assert_array_equal(
            record["layer_delta_coefficients"],
            relative_record["layer_delta_coefficients"],
        )
        np.testing.assert_array_equal(
            record["cumulative_coefficients"],
            relative_record["cumulative_coefficients"],
        )
        assert record["relative_rms_opd_nm"] == relative_record["relative_rms_opd_nm"]
        assert record["plotted_rms_opd_nm"] == pytest.approx(
            1000 * np.linalg.norm(manual[index])
        )


def test_broken_per_position_warm_start_chain_is_rejected(ao_modal_chain):
    """A prior command from another grid site is not a valid inherited state."""
    manifest, saved, _ = ao_modal_chain()
    manifest["regions"][1]["starting_coefficients"] = list(
        manifest["regions"][3]["final_coefficients"]
    )
    with pytest.raises(ValueError, match=r"(?i)(inherit|previous|chain|warm)"):
        relative_regions(manifest, saved, ao_wavekit_phase)


def test_noncommon_original_start_is_rejected(ao_modal_chain):
    """A distinct initial vector at one grid position cannot be silently reused."""
    manifest, saved, _ = ao_modal_chain()
    manifest["regions"][3]["starting_coefficients"][0] += 0.001
    with pytest.raises(ValueError, match="single common"):
        relative_regions(manifest, saved, ao_wavekit_phase)


@pytest.mark.parametrize("bad_coefficients", ([0.1, 0.2], [0.1, np.nan, 0.2]))
def test_invalid_final_modal_basis_is_rejected(bad_coefficients, ao_modal_chain):
    """Every reconstructed final command must use the original finite basis."""
    manifest, saved, _ = ao_modal_chain()
    manifest["regions"][0]["final_coefficients"] = bad_coefficients
    with pytest.raises(ValueError, match="common modal basis"):
        relative_regions(manifest, saved, ao_wavekit_phase)


def test_reconstruction_cannot_change_original_pupil_aperture(ao_modal_chain):
    """A changed reader mask would invalidate the tissue-background reuse."""
    manifest, saved, _ = ao_modal_chain()

    saved[0].pupil = np.ma.array(saved[0].pupil.data, mask=False)

    with pytest.raises(ValueError, match="aperture mask"):
        relative_regions(manifest, saved, ao_wavekit_phase)
