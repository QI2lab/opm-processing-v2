"""Export relative pupils without changing completed tissue display samples."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from PIL import Image
from scripts import ao_tissue_bigtiff, export_ao_relative
from scripts.ao_saved_viewing_background import read_saved_viewing_background
from scripts.ao_tissue_bigtiff import NativeTissueRaster, write_tissue_bigtiffs
from scripts.ao_tissue_overlay import write_tissue_overlay
from scripts.export_ao_overview import Region

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "include_starting", [False, True], ids=["change", "full-state"]
)
def test_relative_export_reuses_original_backgrounds_and_one_common_start(
    tmp_path, monkeypatch, include_starting
):
    """Inherited later starts do not alter the common reference or tissue RGB."""
    source, output = tmp_path / "original", tmp_path / "relative"
    source.mkdir()
    reference = source / "publication_reference.py"
    reference.write_text("# Publication styling reference\n", encoding="utf-8")
    coordinates = (np.arange(8) + 0.5) / 4 - 1
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    mask = yy**2 + xx**2 > 1
    mask[3:5, 3:5] = True

    def phase(coefficients):
        a, b = coefficients[:2]
        radius_squared = xx**2 + yy**2
        spherical = coefficients[7] * (6 * radius_squared**2 - 6 * radius_squared + 1)
        return np.ma.array(2 * np.pi * (a * xx + b * yy + spherical), mask=mask.copy())

    baseline = np.array([0.2, -0.1, 0, 0, 0, 0, 0, 0.12])
    finals = (
        np.array([0.5, 0.4, 0, 0, 0, 0, 0, 0.18]),
        np.array([0.9, 0.7, 0, 0, 0, 0, 0, 0.25]),
    )
    records = [
        {
            "depth_index": depth,
            "region_index": 0,
            "starting_coefficients": (baseline if depth == 0 else finals[0]).tolist(),
            "final_coefficients": final.tolist(),
        }
        for depth, final in enumerate(finals)
    ]
    saved = [
        Region(depth, 0, object(), phase(final)) for depth, final in enumerate(finals)
    ]
    rows, columns = np.indices((64, 96))
    backgrounds = {
        depth: np.stack(
            (
                (3 * columns + rows + depth * 31) % 256,
                (columns + 5 * rows + depth * 47) % 256,
                (7 * columns + 2 * rows + depth * 19) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        for depth in range(2)
    }
    geometry = SimpleNamespace(
        image=backgrounds[0],
        images_by_depth=backgrounds,
        shape_yx=(64, 96),
        scale_yx_um=(1, 1),
        origin_yx_um=(0.5, 0.5),
        extent_yx_um=(0, 64, 0, 96),
        depth_offsets={0: 0, 1: 11.76},
        planned_centers_by_depth_index={depth: {0: (28, 40)} for depth in range(2)},
        planned_grid_by_depth_index={
            depth: {"horizontal_y_um": [20, 44], "vertical_x_um": [32, 64]}
            for depth in range(2)
        },
        layer_projection_rois_by_depth={
            depth: {
                "mip_sha256_by_channel": {
                    str(channel): hashlib.sha256(
                        background[:, :, channel_index].tobytes()
                    ).hexdigest()
                    for channel_index, channel in enumerate((0, 1, 3))
                }
            }
            for depth, background in backgrounds.items()
        },
        provenance={"tissue_composite": {"channels": [0, 1, 3]}},
    )
    monkeypatch.setattr(ao_tissue_bigtiff, "TILE_SHAPE", (16, 16))
    previews = write_tissue_overlay(
        saved,
        geometry,
        source / "previews",
        timepoint=0,
        pupil_diameter_um=16,
        scale_bar_um=5,
        planned_grid=True,
        output_shape_yx=(64, 96),
    )
    native = write_tissue_bigtiffs(
        saved,
        geometry,
        source / "native",
        timepoint=0,
        pupil_diameter_um=16,
        scale_bar_um=5,
    )
    original = {
        "timepoint": 0,
        "reader": str(reference),
        "coefficient_state": "final",
        "wavelength_nm": 488,
        "regions": records,
        "grids": [{"reference": str(reference)}],
        "tissue_overlay": previews,
        "native_tissue_overlays": native,
    }
    manifest_path = source / "ao_publication.json"
    manifest_path.write_text(json.dumps(original), encoding="utf-8")
    original_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    calls = []

    def approved_phase(coefficients, **kwargs):
        calls.append((coefficients.copy(), kwargs))
        return phase(coefficients)

    monkeypatch.setattr(export_ao_relative, "phase_from_modal_coeff", approved_phase)
    monkeypatch.setattr(
        export_ao_relative, "load_saved_regions", lambda manifest: saved
    )
    monkeypatch.setattr(
        export_ao_relative, "load_cached_geometry", lambda root, manifest: geometry
    )
    result = export_ao_relative.export_relative(
        source, output, workers=2, include_starting=include_starting
    )

    assert json.loads(
        (output / "ao_publication.json").read_text("utf-8")
    ) == json.loads(json.dumps(result))
    assert result["coefficient_state"] == (
        "final" if include_starting else "relative_to_acquisition_start"
    )
    assert result["pupil_reference"]["reference_depth_index"] == 0
    reconstruction = (
        result["pupil_reconstruction"]
        if include_starting
        else result["pupil_reference"]["reconstruction"]
    )
    assert reconstruction["basis_version"] == export_ao_relative.BASIS_VERSION
    assert reconstruction["signed_nm_modes"] == [
        list(pair) for pair in export_ao_relative.WAVEKIT_MODE_NM
    ]
    np.testing.assert_array_equal(
        result["pupil_reference"]["reference_coefficients"],
        np.zeros_like(baseline) if include_starting else baseline,
    )
    if include_starting:
        np.testing.assert_array_equal(
            result["pupil_reference"]["original_starting_coefficients"], baseline
        )
        assert result["pupil_reference"]["starting_correction_included"] is True
    assert result["pupil_reference"]["kind"] == (
        "configured_mirror_reference"
        if include_starting
        else "common_acquisition_start"
    )
    assert len(calls) == len(result["grids"]) == 2
    for depth, ((coefficients, kwargs), record) in enumerate(
        zip(calls, result["regions"], strict=True)
    ):
        change = finals[depth] - baseline
        expected = finals[depth] if include_starting else change
        np.testing.assert_array_equal(coefficients, expected)
        np.testing.assert_array_equal(record["plotted_coefficients"], expected)
        np.testing.assert_allclose(
            record["cumulative_coefficients"], change, rtol=0, atol=1e-12
        )
        start = baseline if depth == 0 else finals[depth - 1]
        np.testing.assert_array_equal(
            record["layer_delta_coefficients"], finals[depth] - start
        )
        assert (
            record["starting_coefficients"] == records[depth]["starting_coefficients"]
        )
        assert record["final_coefficients"] == records[depth]["final_coefficients"]
        assert record["plotted_rms_opd_nm"] == pytest.approx(
            np.linalg.norm(expected) * 1000
        )
        assert record["relative_rms_opd_nm"] == pytest.approx(
            np.linalg.norm(change) * 1000
        )
        if include_starting:
            assert "relative_phase_rms_rad" not in record
            assert record["plotted_coefficients"][7] == finals[depth][7]
            assert record["plotted_coefficients"][7] != pytest.approx(change[7])
        np.testing.assert_array_equal(kwargs.pop("mode_indices"), np.arange(1, 9))
        assert kwargs == {
            "samples": 512,
            "norm": True,
            "wavelength": 0.488,
            "out": "radians",
        }
        if depth:
            assert not np.allclose(coefficients, finals[depth] - finals[0])

    opaque_pupil = np.zeros((64, 96), dtype=bool)
    opaque_pupil[20:36, 32:48] = np.repeat(
        np.repeat((~mask)[::-1], 2, axis=0), 2, axis=1
    )
    # Viewing copies can interpolate pupil samples within their display square.
    outside_viewing_pupil = np.ones((64, 96), dtype=bool)
    outside_viewing_pupil[18:38, 30:50] = False
    for depth in range(2):
        old_page, new_page = (
            native["pages"][depth],
            result["native_tissue_overlays"]["pages"][depth],
        )
        old_pixels = tifffile.imread(old_page["tif"])
        with tifffile.TiffFile(new_page["tif"]) as image:
            assert image.is_bigtiff and image.pages[0].predictor == 2
            new_pixels = image.pages[0].asarray()
            header = json.loads(image.pages[0].description)["AOOverlay"]
        if include_starting:
            np.testing.assert_array_equal(new_pixels, old_pixels)
            delta_raster = NativeTissueRaster(
                geometry,
                backgrounds[depth],
                [Region(depth, 0, object(), phase(finals[depth] - baseline))],
                depth=depth,
                pupil_diameter_um=16,
                scale_bar_um=5,
            )
            delta_pixels = delta_raster.render_tile(0, 64, 0, 96)
            assert np.any(new_pixels[opaque_pupil] != delta_pixels[opaque_pupil])
        else:
            assert np.any(new_pixels[opaque_pupil] != old_pixels[opaque_pupil])
        np.testing.assert_array_equal(
            new_pixels[~opaque_pupil], old_pixels[~opaque_pupil]
        )
        assert header["phase_provenance"] == result["pupil_reference"]
        assert (
            new_page["source_channel_mip_sha256"]
            == old_page["source_channel_mip_sha256"]
        )

        old_preview = previews["pages"][depth]
        new_preview = result["tissue_overlay"]["pages"][depth]
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
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in original_hashes
    } == original_hashes


def _warmstarted_case():
    """Use vector updates whose norms cannot be added as scalar depth errors."""
    baseline = np.array([0.2, -0.1])
    final = {
        (0, 0): np.array([0.5, 0.4]),
        (1, 0): np.array([0.9, 0.7]),
        (2, 0): np.array([0.6, 1.2]),
        (0, 1): np.array([0.1, 0.3]),
        (1, 1): np.array([-0.2, 0.1]),
        (2, 1): np.array([-0.5, 0.6]),
    }
    records, saved = [], []
    for (depth, index), vector in sorted(final.items()):
        records.append(
            {
                "depth_index": depth,
                "region_index": index,
                "source": f"saved_z{depth}_r{index}.zarr",
                "starting_coefficients": (
                    baseline if depth == 0 else final[depth - 1, index]
                ).tolist(),
                "final_coefficients": vector.tolist(),
            }
        )
        saved.append(Region(depth, index, object(), np.ma.zeros((8, 8))))
    manifest = {"coefficient_state": "final", "wavelength_nm": 488, "regions": records}
    reader = SimpleNamespace(
        phase_from_modal_coeff=lambda coefficients, **kwargs: np.ma.zeros((8, 8))
    )
    return baseline, final, manifest, saved, reader


def test_cumulative_updates_follow_each_positions_exact_previous_layer():
    """Signed per-layer vectors telescope from a nonzero common starting state."""
    baseline, final, manifest, saved, reader = _warmstarted_case()
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


def test_cumulative_export_rejects_a_broken_position_warmstart_chain():
    """A mislabeled or reset later-layer seed cannot be silently accumulated."""
    _baseline, _final, manifest, saved, reader = _warmstarted_case()
    manifest["regions"][3]["starting_coefficients"][0] += 0.001
    with pytest.raises(ValueError, match=r"(?i)warm|starting|previous"):
        export_ao_relative.relative_regions(manifest, saved, reader)
