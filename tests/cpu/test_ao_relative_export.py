"""Preserve native tissue while exporting physically defined modal changes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image
from scripts import export_ao_relative
from scripts.ao_saved_viewing_background import read_saved_viewing_background
from scripts.ao_tissue_bigtiff import NativeTissueRaster
from scripts.ao_wavekit_phase import phase_from_modal_coeff
from scripts.export_ao_overview import Region

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "include_starting", [False, True], ids=["change", "full-state"]
)
def test_relative_export_reuses_original_backgrounds_and_one_common_start(
    ao_relative_export_case,
    ao_analytical_phase,
    include_starting,
):
    """Actual saved states preserve tissue RGB and telescope signed modal updates."""
    fixture = ao_relative_export_case
    case = fixture.case
    source, output = case.output, case.output.parent / "relative"
    baseline, finals, backgrounds = (
        fixture.baseline,
        fixture.finals,
        fixture.backgrounds,
    )
    original_hashes = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*")
        if p.is_file()
    }
    result = export_ao_relative.export_relative(
        source, output, workers=2, include_starting=include_starting
    )
    assert json.loads((output / "ao_publication.json").read_text()) == json.loads(
        json.dumps(result)
    )
    assert result["coefficient_state"] == (
        "final" if include_starting else "relative_to_acquisition_start"
    )
    reference = result["pupil_reference"]
    assert reference["reference_depth_index"] == 0
    reconstruction = (
        result["pupil_reconstruction"]
        if include_starting
        else reference["reconstruction"]
    )
    assert reconstruction["basis_version"] == export_ao_relative.BASIS_VERSION
    assert reconstruction["signed_nm_modes"] == [
        list(pair) for pair in export_ao_relative.WAVEKIT_MODE_NM
    ]
    np.testing.assert_array_equal(
        reference["reference_coefficients"],
        np.zeros_like(baseline) if include_starting else baseline,
    )
    if include_starting:
        np.testing.assert_array_equal(
            reference["original_starting_coefficients"], baseline
        )
        assert reference["starting_correction_included"] is True
    assert reference["kind"] == (
        "configured_mirror_reference"
        if include_starting
        else "common_acquisition_start"
    )
    assert len(result["grids"]) == 2
    for depth, record in enumerate(result["regions"]):
        change = finals[depth] - baseline
        expected = finals[depth] if include_starting else change
        np.testing.assert_array_equal(record["plotted_coefficients"], expected)
        np.testing.assert_allclose(
            record["cumulative_coefficients"], change, rtol=0, atol=1e-12
        )
        start = baseline if depth == 0 else finals[depth - 1]
        np.testing.assert_array_equal(
            record["layer_delta_coefficients"], finals[depth] - start
        )
        assert (
            record["starting_coefficients"]
            == case.manifest["regions"][depth]["starting_coefficients"]
        )
        assert (
            record["final_coefficients"]
            == case.manifest["regions"][depth]["final_coefficients"]
        )
        assert record["plotted_rms_opd_nm"] == pytest.approx(
            np.linalg.norm(expected) * 1000
        )
        assert record["relative_rms_opd_nm"] == pytest.approx(
            np.linalg.norm(change) * 1000
        )
        analytical = ao_analytical_phase(expected)
        assert record["plotted_phase_rms_rad"] == pytest.approx(
            np.std(analytical.compressed()), abs=1e-12
        )
        if include_starting:
            assert "relative_phase_rms_rad" not in record
            assert record["plotted_coefficients"][7] == finals[depth][7]
            assert record["plotted_coefficients"][7] != pytest.approx(change[7])
        if depth:
            assert not np.allclose(expected, finals[depth] - finals[0])
    rows, columns = np.indices((64, 96))
    opaque_pupil = (rows + 0.5 - 28) ** 2 + (columns + 0.5 - 40) ** 2 <= 8**2
    outside_viewing_pupil = np.ones((64, 96), bool)
    outside_viewing_pupil[18:38, 30:50] = False
    for depth in range(2):
        old_page, new_page = (
            case.native["pages"][depth],
            result["native_tissue_overlays"]["pages"][depth],
        )
        old_pixels = tifffile.imread(old_page["tif"])
        with tifffile.TiffFile(new_page["tif"]) as image:
            assert image.is_bigtiff and image.pages[0].predictor == 2
            new_pixels = image.pages[0].asarray()
            header = json.loads(image.pages[0].description)["AOOverlay"]
        if include_starting:
            np.testing.assert_array_equal(new_pixels, old_pixels)
            delta = phase_from_modal_coeff(
                finals[depth] - baseline,
                samples=512,
                norm=True,
                wavelength=0.488,
                out="radians",
            )
            raster = NativeTissueRaster(
                case.geometry,
                backgrounds[depth],
                [Region(depth, 0, None, delta)],
                depth=depth,
                pupil_diameter_um=16,
                scale_bar_um=5,
            )
            delta_pixels = raster.render_tile(0, 64, 0, 96)
            assert np.any(new_pixels[opaque_pupil] != delta_pixels[opaque_pupil])
        else:
            assert np.any(new_pixels[opaque_pupil] != old_pixels[opaque_pupil])
        np.testing.assert_array_equal(
            new_pixels[~opaque_pupil], old_pixels[~opaque_pupil]
        )
        assert header["phase_provenance"] == reference
        assert (
            new_page["source_channel_mip_sha256"]
            == old_page["source_channel_mip_sha256"]
        )
        old_preview, new_preview = (
            case.previews["pages"][depth],
            result["tissue_overlay"]["pages"][depth],
        )
        recovered, _ = read_saved_viewing_background(Path(new_preview["svg"]), (64, 96))
        np.testing.assert_array_equal(recovered, backgrounds[depth])
        with (
            Image.open(old_preview["png"]) as old_png,
            Image.open(new_preview["png"]) as new_png,
        ):
            old_rgb, new_rgb = (
                np.asarray(old_png.convert("RGB")),
                np.asarray(new_png.convert("RGB")),
            )
        if include_starting:
            np.testing.assert_array_equal(old_rgb, new_rgb)
        else:
            assert np.any(old_rgb != new_rgb)
        np.testing.assert_array_equal(
            old_rgb[outside_viewing_pupil], new_rgb[outside_viewing_pupil]
        )
    assert {
        p: hashlib.sha256(p.read_bytes()).hexdigest() for p in original_hashes
    } == original_hashes


def test_cumulative_updates_follow_each_positions_exact_previous_layer(
    ao_two_position_warm_start,
):
    """Signed per-layer vectors telescope from a nonzero common starting state."""
    baseline, final, manifest, saved, reader = ao_two_position_warm_start
    original = json.loads(json.dumps(manifest))
    _regions, reference, records = export_ao_relative.relative_regions(
        manifest, saved, reader
    )
    running = {0: np.zeros(2), 1: np.zeros(2)}
    for record in records:
        depth, index = record["depth_index"], record["region_index"]
        start = baseline if depth == 0 else final[depth - 1, index]
        delta = final[depth, index] - start
        running[index] += delta
        np.testing.assert_array_equal(record["layer_delta_coefficients"], delta)
        np.testing.assert_allclose(
            record["cumulative_coefficients"], running[index], atol=1e-12
        )
        np.testing.assert_array_equal(
            record["plotted_coefficients"], final[depth, index] - baseline
        )
        assert record["source"] == original["regions"][depth * 2 + index]["source"]
    assert manifest == original
    np.testing.assert_array_equal(reference["reference_coefficients"], baseline)
    assert not np.isclose(
        np.linalg.norm(final[2, 0] - baseline),
        sum(
            np.linalg.norm(final[z, 0] - (baseline if z == 0 else final[z - 1, 0]))
            for z in range(3)
        ),
    )


def test_cumulative_export_rejects_a_broken_position_warmstart_chain(
    ao_two_position_warm_start,
):
    """A mislabeled or reset later-layer seed cannot be silently accumulated."""
    _baseline, _final, manifest, saved, reader = ao_two_position_warm_start
    manifest["regions"][3]["starting_coefficients"][0] += 0.001
    with pytest.raises(ValueError, match=r"(?i)warm|starting|previous"):
        export_ao_relative.relative_regions(manifest, saved, reader)
