"""Change native pupil phase while retaining every existing tissue display pixel."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from scripts import ao_tissue_bigtiff
from scripts.ao_tissue_bigtiff import NativeTissueRaster, write_tissue_bigtiffs
from scripts.ao_tissue_rephase import rephase_tissue_bigtiff

pytestmark = pytest.mark.unit


class _UnreadBackground:
    """Rephasing must use existing TIFF pixels rather than reloading backgrounds."""

    shape = (35, 53, 3)
    dtype = np.dtype(np.uint8)

    def __array__(self, dtype=None, copy=None):
        raise AssertionError("Rephasing must not materialize a background")

    def __getitem__(self, selection):
        raise AssertionError("Rephasing must not read background pixels")


@pytest.fixture
def phase_case(tmp_path, monkeypatch):
    """Make predicted RGB tiles with edge pupils, masked holes and existing lines."""
    yy, xx = np.indices((35, 53))
    background = np.stack(
        (
            (7 * xx + 3 * yy) % 256,
            (5 * xx + 11 * yy + 17) % 256,
            (13 * xx + 2 * yy + 41) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[3:5, 3:5] = True
    new_phase = np.repeat(
        np.repeat([[-np.pi, -np.pi / 2], [np.pi / 2, np.pi]], 4, axis=0), 4, axis=1
    )
    old_regions = [
        SimpleNamespace(
            depth=0,
            index=index,
            pupil=np.ma.array(np.full((8, 8), -3 * np.pi / 4), mask=mask.copy()),
        )
        for index in range(3)
    ]
    new_regions = [
        SimpleNamespace(
            depth=0, index=index, pupil=np.ma.array(new_phase.copy(), mask=mask.copy())
        )
        for index in range(3)
    ]
    geometry = SimpleNamespace(
        shape_yx=(35, 53),
        scale_yx_um=(1, 0.5),
        origin_yx_um=(0.5, 0.25),
        extent_yx_um=(0, 35, 0, 26.5),
        images_by_depth={0: background},
        planned_centers_by_depth_index={
            0: {0: (10.5, 9.25), 1: (32.5, 2.75), 2: (2.5, 24.75)}
        },
        planned_grid_by_depth_index={
            0: {"horizontal_y_um": [10.5, 25.5], "vertical_x_um": [9.25, 20.25]}
        },
        layer_projection_rois_by_depth={
            0: {"mip_sha256": hashlib.sha256(background.tobytes()).hexdigest()}
        },
    )
    monkeypatch.setattr(ao_tissue_bigtiff, "TILE_SHAPE", (16, 16))
    original = write_tissue_bigtiffs(
        old_regions,
        geometry,
        tmp_path / "original",
        timepoint=0,
        pupil_diameter_um=12,
        scale_bar_um=5,
    )
    raster = NativeTissueRaster(
        geometry,
        _UnreadBackground(),
        new_regions,
        depth=0,
        pupil_diameter_um=12,
        scale_bar_um=5,
    )
    expected = NativeTissueRaster(
        geometry, background, new_regions, depth=0, pupil_diameter_um=12, scale_bar_um=5
    ).render_tile(0, 35, 0, 53)
    provenance = {
        "coefficient_state": "final_minus_common_starting",
        "reference": {"timepoint": 0, "depth_index": 0, "region_index": 0},
        "reference_starting_coefficients": np.linspace(
            0.0123456, 0.1234567, 32
        ).tolist(),
    }
    return SimpleNamespace(
        source=original["pages"][0]["tif"],
        destination=tmp_path / "relative" / "native.tif",
        raster=raster,
        expected=expected,
        provenance=provenance,
        geometry=geometry,
        regions=new_regions,
        phase=new_phase,
        mask=mask,
    )


def test_rephase_changes_only_pupil_pixels_and_reuses_compressed_tissue_tiles(
    phase_case,
):
    """Predictor-encoded edge tiles decode correctly and masked grids remain exact."""
    case = phase_case

    source = Path(case.source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    with tifffile.TiffFile(source) as original:
        original_pixels = original.pages[0].asarray()
        original_metadata = json.loads(original.pages[0].description)
    record = rephase_tissue_bigtiff(
        source, case.destination, case.raster, phase_provenance=case.provenance
    )
    changed = np.any(case.expected != original_pixels, axis=-1)
    assert changed[32:, :18].any() and changed[:9, 48:].any()
    assert not changed[10, 18]  # Transparent pupil hole over a grid intersection.
    with (
        tifffile.TiffFile(source) as original,
        tifffile.TiffFile(case.destination) as target,
    ):
        page, old_page = target.pages[0], original.pages[0]
        assert target.is_bigtiff and page.is_tiled
        assert page.predictor == 2
        assert page.compression == old_page.compression
        actual = page.asarray()
        np.testing.assert_array_equal(actual, case.expected)
        np.testing.assert_array_equal(actual[~changed], original_pixels[~changed])
        metadata = json.loads(page.description)
        assert metadata["AOOverlay"]["phase_provenance"] == case.provenance
        for key in (
            "PhysicalSizeX",
            "PhysicalSizeY",
            "PhysicalSizeUnit",
            "OriginYXUm",
            "ExtentYXUm",
        ):
            assert metadata[key] == original_metadata[key]
        assert page.tags["XResolution"].value == old_page.tags["XResolution"].value
        assert page.tags["YResolution"].value == old_page.tags["YResolution"].value
        copied = reencoded = 0
        with source.open("rb") as old_stream, case.destination.open("rb") as new_stream:
            for index, (y0, x0) in enumerate(
                (y, x) for y in range(0, 35, 16) for x in range(0, 53, 16)
            ):
                if changed[y0 : y0 + 16, x0 : x0 + 16].any():
                    reencoded += 1
                    continue
                copied += 1
                old_stream.seek(old_page.dataoffsets[index])
                new_stream.seek(page.dataoffsets[index])
                assert new_stream.read(page.databytecounts[index]) == old_stream.read(
                    old_page.databytecounts[index]
                )
        assert copied > 0 and reencoded > 0
        assert record["copied_compressed_tiles"] == copied
        assert record["reencoded_tiles"] == reencoded
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert record["phase_provenance"] == case.provenance
    for region in case.regions:
        np.testing.assert_array_equal(region.pupil.data, case.phase)
        np.testing.assert_array_equal(region.pupil.mask, case.mask)


@pytest.mark.parametrize("invalid", ("source-destination", "physical-canvas"))
def test_rephase_refuses_inplace_or_changed_physical_geometry(phase_case, invalid):
    """A replacement cannot overwrite its source or reinterpret native coordinates."""
    case = phase_case

    source = Path(case.source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = source if invalid == "source-destination" else case.destination
    raster = case.raster
    if invalid == "physical-canvas":
        geometry = SimpleNamespace(**vars(case.geometry))
        geometry.extent_yx_um = (0, 35, 0, 27)
        raster = NativeTissueRaster(
            geometry,
            _UnreadBackground(),
            case.regions,
            depth=0,
            pupil_diameter_um=12,
            scale_bar_um=5,
        )
    with pytest.raises(ValueError, match=r"separate TIFF|physical canvas"):
        rephase_tissue_bigtiff(
            source, destination, raster, phase_provenance=case.provenance
        )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert not case.destination.exists()
