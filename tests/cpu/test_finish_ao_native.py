"""Continue native exports from completed caches without changing viewing copies."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from PIL import Image
from scripts import finish_ao_native
from scripts.ao_wavekit_phase import BASIS_VERSION, phase_from_modal_coeff

pytestmark = pytest.mark.unit


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _snapshot(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def completed_viewing_export(tmp_path):
    """Make nine completed scalar caches and three existing preview file pairs."""
    output = tmp_path / "export"
    output.mkdir()
    depths, channels = (0, 1, 2), (0, 1, 3)
    colors = {0: [255, 0, 255], 1: [0, 255, 255], 3: [255, 165, 0]}
    limits, cache_paths, fingerprints, hashes = {}, {}, {}, {}
    for channel_index, channel in enumerate(channels):
        directory = output / (
            "projection_cache"
            if channel == 1
            else f"projection_cache_channel_{channel}"
        )
        directory.mkdir()
        context = {
            "channel": channel,
            "timepoint": 0,
            "depths": list(depths),
            "shape_yx": [40, 60],
            "canvas_origin_yx_um": [0.5, 0.5],
            "canvas_scale_yx_um": [1, 1],
        }
        fingerprint = hashlib.sha256(
            (json.dumps(context, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
        cache = {
            "schema": "opm-ao-layer-projection-cache-v1",
            "fingerprint": fingerprint,
            "context": context,
            "phase": "complete",
            "completed_xy_blocks": [True],
            "images": {},
            "mip_sha256_by_depth": {},
        }
        limits[str(channel)] = {}
        for depth in depths:
            pixels = np.zeros((40, 60), np.uint16)
            column = ((channel_index + depth) % 3) * 20
            amplitude = (channel_index + 1) * (depth + 1) * 100
            pixels[:, column : column + 20] = amplitude
            filename = f"layer_{depth:03d}.npy"
            np.save(directory / filename, pixels, allow_pickle=False)
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            hashes[channel, depth] = digest
            limits[str(channel)][str(depth)] = [0, amplitude]
            cache["images"][str(depth)] = {
                "path": filename,
                "shape_yx": [40, 60],
                "dtype": pixels.dtype.str,
            }
            cache["mip_sha256_by_depth"][str(depth)] = digest
        cache_paths[channel] = directory / "layer_projection_cache.json"
        fingerprints[channel] = fingerprint
        _write_json(cache_paths[channel], cache)

    previews = output / "previews"
    previews.mkdir()
    pages = []
    for depth in depths:
        png, svg = previews / f"layer_{depth}.png", previews / f"layer_{depth}.svg"
        Image.new("RGB", (6, 4), (depth * 50, 20, 40)).save(png)
        svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="6" height="4"/>',
            encoding="utf-8",
        )
        pages.append(
            {
                "depth_index": depth,
                "png": str(png),
                "svg": str(svg),
                "scale_bar": {"length_um": 5},
                "planned_grid": {"horizontal_y_um": [], "vertical_x_um": []},
                "regions": [{"region_index": 0, "center_yx_um": [20, 30]}],
                "projection_roi": {
                    "depth_index": depth,
                    "aggregate_z_start_stop": [depth, depth + 2],
                    "projection_cache_by_channel": {
                        str(c): str(cache_paths[c]) for c in channels
                    },
                    "projection_cache_fingerprint_by_channel": {
                        str(c): fingerprints[c] for c in channels
                    },
                    "mip_sha256_by_channel": {
                        str(c): hashes[c, depth] for c in channels
                    },
                },
            }
        )
    manifest = {
        "timepoint": 0,
        "coefficient_state": "final",
        "wavelength_nm": 488,
        "native_tissue_overlays": {"status": "rendering"},
        "regions": [
            {
                "depth_index": d,
                "region_index": 0,
                "starting_coefficients": [0] * 32,
                "final_coefficients": [0] * 32,
            }
            for d in depths
        ],
        "tissue_overlay": {
            "extent_yx_um": [0, 40, 0, 60],
            "pupil_display_diameter_um": 8,
            "geometry": {
                "tissue_composite": {
                    "channels": list(channels),
                    "colors_rgb8": {str(c): color for c, color in colors.items()},
                    "display_limits_by_channel_depth": limits,
                }
            },
            "pages": pages,
        },
    }
    _write_json(output / "ao_publication.json", manifest)
    regions = [
        SimpleNamespace(depth=d, index=0, pupil=np.ma.zeros((8, 8))) for d in depths
    ]
    return SimpleNamespace(
        output=output,
        manifest=manifest,
        cache_paths=cache_paths,
        hashes=hashes,
        colors=colors,
        regions=regions,
    )


def test_finish_cached_native_export_preserves_inputs_and_publishes_atomically(
    completed_viewing_export, monkeypatch
):
    """Reuse read-only scalar caches and replace only the final native manifest field."""
    case = completed_viewing_export
    before = _snapshot(case.output)
    manifest_path = case.output / "ao_publication.json"
    original_manifest = manifest_path.read_bytes()
    geometry = finish_ao_native.load_cached_geometry(case.output, case.manifest)
    assert geometry.shape_yx == (40, 60)
    np.testing.assert_array_equal(
        geometry.images_by_depth[0][0:1, 0:1], [[[255, 0, 255]]]
    )
    monkeypatch.setattr(
        finish_ao_native, "load_saved_regions", lambda manifest: case.regions
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
    after = _snapshot(case.output)
    assert all(
        after[path] == digest
        for path, digest in before.items()
        if path != "ao_publication.json"
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
    completed_viewing_export, monkeypatch, mismatch
):
    """A cache differing from its completed previews cannot create native outputs."""
    case = completed_viewing_export
    path = case.cache_paths[1]
    cache = json.loads(path.read_text("utf-8"))
    if mismatch == "fingerprint":
        cache["fingerprint"] = "f" * 64
    else:
        cache["mip_sha256_by_depth"]["0"] = "f" * 64
    _write_json(path, cache)
    before = _snapshot(case.output)

    def unexpected_pupil_load(manifest):
        raise AssertionError(
            "Cache validation must precede pupil loading and TIFF writes"
        )

    monkeypatch.setattr(finish_ao_native, "load_saved_regions", unexpected_pupil_load)
    with pytest.raises(ValueError, match="provenance differs"):
        finish_ao_native.finish_native(case.output, depth_workers=2)
    assert _snapshot(case.output) == before
    assert not (case.output / "native").exists()


@pytest.fixture
def saved_phase_inputs(tmp_path, monkeypatch):
    """Supply saved asymmetric modes through a versioned, immutable reader source."""
    path = tmp_path / "approved_reader.py"
    path.write_text("# Saved acquisition reader\n", encoding="utf-8")
    starting = np.zeros(32)
    starting[5:7] = (0.04, -0.02)
    final = starting.copy()
    final[5:7] += (0.09, -0.08)
    calls = []

    def legacy_pupil(*, coefficient_state, **options):
        calls.append((coefficient_state, options.copy()))
        selected = {
            "starting": starting,
            "final": final,
            "change": final - starting,
        }[coefficient_state].copy()
        selected[5:7] = selected[5:7][::-1]
        return phase_from_modal_coeff(selected, **options)

    result = SimpleNamespace(
        starting_coefficients=starting,
        final_coefficients=final,
        get_pupil=legacy_pupil,
    )
    reader = SimpleNamespace(AOOptimizationResult=lambda *args: result)
    monkeypatch.setattr(finish_ao_native, "load_reference", lambda source: reader)
    manifest = {
        "reader": str(path),
        "reader_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "timepoint": 0,
        "wavelength_nm": 488,
        "coefficient_state": "final",
        "regions": [
            {
                "source": str(tmp_path / "ao_results.zarr"),
                "depth_index": 1,
                "region_index": 2,
                "starting_coefficients": starting.tolist(),
                "final_coefficients": final.tolist(),
            }
        ],
    }
    return SimpleNamespace(
        manifest=manifest, starting=starting, final=final, calls=calls, result=result
    )


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
    assert case.calls == [
        (state, {"samples": 512, "norm": True, "wavelength": 0.488, "out": "radians"})
    ]
    assert corrected.result is legacy.result is case.result
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
    assert not case.calls
