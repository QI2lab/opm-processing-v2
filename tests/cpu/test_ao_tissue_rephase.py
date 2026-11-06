"""Change native pupil phase while retaining every existing tissue display pixel."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scripts.ao_tissue_bigtiff import NativeTissueRaster
from scripts.ao_tissue_rephase import rephase_tissue_bigtiff

pytestmark = pytest.mark.unit


def test_rephase_changes_only_pupil_pixels_and_reuses_compressed_tissue_tiles(
    phase_case,
    ao_memmap_reads,
):
    """Predictor-encoded edge tiles decode correctly and masked grids remain exact."""
    case = phase_case

    source = Path(case.source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    with tifffile.TiffFile(source) as original:
        original_pixels = original.pages[0].asarray()
        original_metadata = json.loads(original.pages[0].description)
    ao_memmap_reads.clear()
    record = rephase_tissue_bigtiff(
        source, case.destination, case.raster, phase_provenance=case.provenance
    )
    assert not ao_memmap_reads
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
        geometry = replace(case.geometry, extent_yx_um=(0, 35, 0, 27))
        raster = NativeTissueRaster(
            geometry,
            case.background,
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
