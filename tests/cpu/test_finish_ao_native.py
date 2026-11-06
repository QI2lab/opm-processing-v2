"""Continue native exports from completed caches without changing viewing copies."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scripts import finish_ao_native
from scripts.ao_wavekit_phase import BASIS_VERSION, phase_from_modal_coeff

pytestmark = pytest.mark.unit


def test_finish_cached_native_export_preserves_inputs_and_publishes_atomically(
    completed_viewing_export, monkeypatch, stored_file_bytes
):
    """Reuse read-only scalar caches and replace only the final native manifest field."""
    case = completed_viewing_export
    before = stored_file_bytes(case.output)
    manifest_path = case.output / "ao_publication.json"
    original_manifest = manifest_path.read_bytes()
    geometry = finish_ao_native.load_cached_geometry(case.output, case.manifest)
    assert geometry.shape_yx == (40, 60)
    np.testing.assert_array_equal(
        geometry.images_by_depth[0][0:1, 0:1], [[[255, 0, 255]]]
    )
    original_replace = finish_ao_native.os.replace
    replacements = []

    def observe_publication(source, destination, *args, **kwargs):
        if Path(destination) == manifest_path:
            assert manifest_path.read_bytes() == original_manifest
            pending = json.loads(Path(source).read_text("utf-8"))
            assert len(pending["native_tissue_overlays"]["pages"]) == 3
            assert all(
                Path(page["tif"]).is_file()
                for page in pending["native_tissue_overlays"]["pages"]
            )
            replacements.append(Path(source))
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(finish_ao_native.os, "replace", observe_publication)
    native = finish_ao_native.finish_native(case.output, depth_workers=2)
    assert native["depth_workers"] == 2
    assert [page["depth_index"] for page in native["pages"]] == [0, 1, 2]
    after = stored_file_bytes(case.output)
    assert all(
        after[path] == digest
        for path, digest in before.items()
        if path.as_posix() != "ao_publication.json"
    )
    assert len(replacements) == 1
    assert not replacements[0].exists()
    persisted = json.loads(manifest_path.read_text("utf-8"))
    assert persisted["native_tissue_overlays"] == native
    expected = copy.deepcopy(case.manifest)
    expected["native_tissue_overlays"] = native
    assert persisted == expected
    for page in native["pages"]:
        depth = page["depth_index"]
        expected_hashes = {str(c): case.hashes[c, depth] for c in (0, 1, 3)}
        assert page["source_channel_mip_sha256"] == expected_hashes
        assert page["source_mip_sha256"] is None
        with tifffile.TiffFile(page["tif"]) as image:
            assert image.is_bigtiff
            rgb = image.pages[0].asarray()
            assert rgb.shape == (40, 60, 3)
            metadata = json.loads(image.pages[0].description)["AOOverlay"]
            assert metadata["source_channel_mip_sha256"] == expected_hashes
            for channel_index, channel in enumerate((0, 1, 3)):
                column = ((channel_index + depth) % 3) * 20 + 10
                np.testing.assert_array_equal(rgb[3, column], case.colors[channel])


@pytest.mark.parametrize("mismatch", ("fingerprint", "mip-sha"))
def test_mismatched_preview_cache_provenance_prevents_native_writes(
    completed_viewing_export, stored_file_bytes, mismatch
):
    """A cache differing from its completed previews cannot create native outputs."""
    case = completed_viewing_export
    path = case.cache_paths[1]
    cache = json.loads(path.read_text("utf-8"))
    if mismatch == "fingerprint":
        cache["fingerprint"] = "f" * 64
    else:
        cache["mip_sha256_by_depth"]["0"] = "f" * 64
    path.write_text(json.dumps(cache))
    before = stored_file_bytes(case.output)

    with pytest.raises(ValueError, match="provenance differs"):
        finish_ao_native.finish_native(case.output, depth_workers=2)
    assert stored_file_bytes(case.output) == before
    assert not (case.output / "native").exists()


@pytest.mark.parametrize("state", ("starting", "final", "change"))
def test_saved_manifest_basis_preserves_legacy_or_reconstructs_sdk_phase(
    saved_phase_inputs, state
):
    """Legacy previews retain their old phase; flagged previews use the corrected basis."""
    case = saved_phase_inputs
    case.manifest["coefficient_state"] = state
    before = copy.deepcopy(case.manifest)
    legacy = finish_ao_native.load_saved_regions(case.manifest)[0]
    flagged = copy.deepcopy(case.manifest)
    flagged["pupil_reconstruction"] = {"basis_version": BASIS_VERSION}
    corrected = finish_ao_native.load_saved_regions(flagged)[0]
    selected = {
        "starting": case.starting,
        "final": case.final,
        "change": case.final - case.starting,
    }[state]
    expected = phase_from_modal_coeff(
        selected, samples=512, norm=True, wavelength=0.488, out="radians"
    )
    legacy_coefficients = selected.copy()
    legacy_coefficients[5:7] = legacy_coefficients[5:7][::-1]
    expected_legacy = phase_from_modal_coeff(
        legacy_coefficients, samples=512, norm=True, wavelength=0.488, out="radians"
    )
    assert legacy.pupil.shape == corrected.pupil.shape == (512, 512)
    np.testing.assert_array_equal(legacy.pupil.mask, corrected.pupil.mask)
    np.testing.assert_array_equal(corrected.pupil.data, expected.data)
    np.testing.assert_array_equal(legacy.pupil.data, expected_legacy.data)
    assert not np.array_equal(legacy.pupil.compressed(), corrected.pupil.compressed())
    assert (corrected.depth, corrected.index) == (1, 2)
    assert case.manifest == before
    np.testing.assert_array_equal(
        case.starting, before["regions"][0]["starting_coefficients"]
    )
    np.testing.assert_array_equal(
        case.final, before["regions"][0]["final_coefficients"]
    )


@pytest.mark.parametrize(
    "mismatch", ("unknown-basis", "source-coefficients", "reader-sha")
)
def test_versioned_phase_loading_rejects_unknown_basis_or_changed_inputs(
    saved_phase_inputs, mismatch
):
    """A basis flag must not bypass coefficient and approved-reader integrity checks."""
    case = saved_phase_inputs
    manifest = copy.deepcopy(case.manifest)
    manifest["pupil_reconstruction"] = {"basis_version": BASIS_VERSION}
    if mismatch == "unknown-basis":
        manifest["pupil_reconstruction"]["basis_version"] = "unrecognized-v2"
        message = "Unknown saved pupil reconstruction basis"
    elif mismatch == "source-coefficients":
        manifest["regions"][0]["final_coefficients"][5] += 0.01
        message = "Saved AO coefficients changed"
    else:
        manifest["reader_sha256"] = "f" * 64
        message = "approved pupil reader has changed"
    with pytest.raises(ValueError, match=message):
        finish_ao_native.load_saved_regions(manifest)
