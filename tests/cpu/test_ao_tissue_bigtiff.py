"""Stream native AO tissue pixels without changing phase or physical geometry."""

import json

import matplotlib
import numpy as np
import pytest
import tifffile
from matplotlib.colors import Normalize
from scripts.ao_tissue_bigtiff import (
    NativeTissueRaster,
    native_display_limits,
    write_tissue_bigtiffs,
)

pytestmark = pytest.mark.unit


def test_streamed_uint16_percentiles_match_interpolated_numpy_quantiles(
    native_percentile_case, ao_memmap_reads
):
    """Histogram rank interpolation agrees with exact percentiles across strips."""
    case = native_percentile_case
    pixels = case.pixels
    source = case.source
    limits = native_display_limits(source)
    np.testing.assert_allclose(
        limits, np.percentile(pixels, (1, 99.8)), rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(limits, (2.6, 65513.6), rtol=0, atol=1e-9)
    assert [shape for _, shape in ao_memmap_reads] == [(1, 7)] * 3


@pytest.mark.parametrize(("block_height", "block_width"), [(7, 11), (13, 17)])
def test_arbitrary_native_tiles_stitch_without_alpha_or_pupil_seams(
    native_quadrant_raster_case, ao_memmap_reads, block_height, block_width
):
    """Odd render boundaries preserve grid blending and asymmetric SJS colors."""
    case = native_quadrant_raster_case
    pixels = case.pixels
    phase = case.phase
    mask = case.mask
    pupil = case.pupil
    regions = case.regions
    geometry = case.geometry
    reference = NativeTissueRaster(
        geometry,
        pixels,
        regions,
        depth=0,
        pupil_diameter_um=8,
        scale_bar_um=10,
        display_limits=(0, 200),
    ).render_tile(0, 79, 0, 1001)
    source = case.source
    raster = NativeTissueRaster(
        geometry,
        source,
        regions,
        depth=0,
        pupil_diameter_um=8,
        scale_bar_um=10,
        display_limits=(0, 200),
    )
    stitched = np.empty((79, 1001, 3), np.uint8)
    for y0 in range(0, 79, block_height):
        for x0 in range(0, 1001, block_width):
            y1, x1 = min(y0 + block_height, 79), min(x0 + block_width, 1001)
            stitched[y0:y1, x0:x1] = raster.render_tile(y0, y1, x0, x1)
    np.testing.assert_array_equal(stitched, reference)
    assert all(
        shape[0] <= block_height and shape[1] <= block_width
        for _, shape in ao_memmap_reads
    )
    assert ao_memmap_reads
    endpoints = np.asarray(raster.metadata()["scale_bar"]["endpoints_yx_um"])
    assert endpoints[1, 1] - endpoints[0, 1] == pytest.approx(10)

    np.testing.assert_array_equal(reference[1, 1], (128, 128, 128))
    # Full-coverage white grid lines blend once; their crossing blends twice.
    np.testing.assert_array_equal(reference[15, 1], (166, 166, 166))
    np.testing.assert_array_equal(reference[1, 17], (166, 166, 166))
    np.testing.assert_array_equal(reference[15, 17], (193, 193, 193))
    color = matplotlib.colormaps[matplotlib.rcParams["image.cmap"]]
    normalize = Normalize(-2 * np.pi, 2 * np.pi)
    for row, column, value in (
        (37, 326, -np.pi),
        (37, 334, -np.pi / 2),
        (33, 326, np.pi / 2),
        (33, 334, np.pi),
    ):
        np.testing.assert_array_equal(
            reference[row, column], color(normalize(value), bytes=True)[:3]
        )
    np.testing.assert_array_equal(pupil.data, phase)
    np.testing.assert_array_equal(pupil.mask, mask)


def test_native_tiled_bigtiff_round_trip_preserves_rgb_and_calibration(
    tmp_path, native_gray_tiff_case, ao_memmap_reads
):
    """Lossless edge tiles decode exactly and retain anisotropic pixel calibration."""
    case = native_gray_tiff_case
    pixels = case.pixels
    mip_hash = case.mip_hash
    geometry = case.geometry
    regions = case.regions
    expected = NativeTissueRaster(
        geometry,
        pixels,
        regions,
        depth=2,
        timepoint=1,
        pupil_diameter_um=4,
        scale_bar_um=3,
    ).render_tile(0, 35, 0, 53)
    record = write_tissue_bigtiffs(
        regions,
        geometry,
        tmp_path,
        timepoint=1,
        pupil_diameter_um=4,
        scale_bar_um=3,
    )
    assert record["native_pixel_shape_yx"] == [35, 53]
    assert len(record["pages"]) == 1
    with tifffile.TiffFile(record["pages"][0]["tif"]) as image:
        assert image.is_bigtiff
        page = image.pages[0]
        assert page.is_tiled
        assert (page.tilelength, page.tilewidth) == (16, 16)
        assert page.photometric == tifffile.PHOTOMETRIC.RGB
        assert page.shape == (35, 53, 3)
        np.testing.assert_array_equal(page.asarray(), expected)
        assert page.tags["ResolutionUnit"].value == tifffile.RESUNIT.CENTIMETER
        x_resolution, y_resolution = (
            page.tags["XResolution"].value,
            page.tags["YResolution"].value,
        )
        assert x_resolution[0] / x_resolution[1] == 40000
        assert y_resolution[0] / y_resolution[1] == 20000
        metadata = json.loads(page.description)
        assert metadata["PhysicalSizeX"] == 0.25
        assert metadata["PhysicalSizeY"] == 0.5
        assert metadata["PhysicalSizeUnit"] == "micrometer"
        assert metadata["AOOverlay"]["source_mip_sha256"] == mip_hash
        assert metadata["AOOverlay"]["depth_index"] == 2
        assert metadata["AOOverlay"]["timepoint"] == 1
    assert ao_memmap_reads
    assert all(shape[0] <= 16 and shape[1] <= 53 for _, shape in ao_memmap_reads)


@pytest.mark.parametrize(("block_height", "block_width"), [(7, 11), (9, 13)])
def test_rgb_native_tiles_preserve_precomposed_colors_and_transparent_pupil_holes(
    native_rgb_raster_case, ao_memmap_reads, block_height, block_width
):
    """Precomposed RGB values survive bounded rendering and arbitrary tile seams."""
    case = native_rgb_raster_case
    pixels = case.pixels
    original = case.original
    mask = case.mask
    pupil = case.pupil
    regions = case.regions
    geometry = case.geometry
    expected = NativeTissueRaster(
        geometry, pixels, regions, depth=0, pupil_diameter_um=4, scale_bar_um=3
    ).render_tile(0, 41, 0, 67)
    source = case.source
    raster = NativeTissueRaster(
        geometry, source, regions, depth=0, pupil_diameter_um=4, scale_bar_um=3
    )
    stitched = np.empty_like(pixels)
    for y0 in range(0, 41, block_height):
        for x0 in range(0, 67, block_width):
            y1, x1 = min(y0 + block_height, 41), min(x0 + block_width, 67)
            stitched[y0:y1, x0:x1] = raster.render_tile(y0, y1, x0, x1)
    np.testing.assert_array_equal(stitched, expected)
    assert all(
        shape[0] <= block_height and shape[1] <= block_width
        for _, shape in ao_memmap_reads
    )
    assert ao_memmap_reads
    for row, column in ((0, 0), (4, 9), (40, 0), (20, 33)):
        # The final point lies in the pupil's explicit transparent central hole.
        np.testing.assert_array_equal(expected[row, column], pixels[row, column])
    np.testing.assert_array_equal(pixels, original)
    np.testing.assert_array_equal(pupil.mask, mask)


def test_rgb_native_bigtiff_preserves_composite_channel_colors(
    tmp_path, native_rgb_tiff_case, ao_memmap_reads
):
    """Stream magenta, cyan and orange without applying another contrast mapping."""
    case = native_rgb_tiff_case
    pixels = case.pixels
    channel_hashes = case.channel_hashes
    composite = case.composite
    geometry = case.geometry
    regions = case.regions
    expected = NativeTissueRaster(
        geometry, pixels, regions, depth=2, pupil_diameter_um=4, scale_bar_um=3
    ).render_tile(0, 35, 0, 53)

    contrasts = case.contrasts
    record = write_tissue_bigtiffs(
        regions, geometry, tmp_path, timepoint=0, pupil_diameter_um=4, scale_bar_um=3
    )
    with tifffile.TiffFile(record["pages"][0]["tif"]) as image:
        assert image.is_bigtiff
        page = image.pages[0]
        assert page.is_tiled and page.shape == (35, 53, 3)
        assert page.photometric == tifffile.PHOTOMETRIC.RGB
        actual = page.asarray()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[2, 2], (255, 0, 255))
        np.testing.assert_array_equal(actual[2, 25], (0, 255, 255))
        np.testing.assert_array_equal(actual[2, 45], (255, 128, 0))
        metadata = json.loads(page.description)["AOOverlay"]
        assert metadata["background_rgb_passthrough"] is True
        assert metadata["background_composite"] == composite
        assert metadata["source_channel_mip_sha256"] == channel_hashes
        assert metadata["source_mip_sha256"] is None
    assert not contrasts
    assert ao_memmap_reads
    assert all(shape[0] <= 16 and shape[1] <= 16 for _, shape in ao_memmap_reads)


def test_parallel_depth_exports_keep_native_pixels_order_and_source_identity(
    tmp_path, native_parallel_case, ao_memmap_reads
):
    """Independent depth files retain the correct RGB source and stable metadata order."""
    case = native_parallel_case
    pixels = case.pixels
    sources = case.sources
    hashes = case.hashes
    geometry = case.geometry
    regions = case.regions
    record = write_tissue_bigtiffs(
        regions,
        geometry,
        tmp_path,
        timepoint=4,
        pupil_diameter_um=8,
        scale_bar_um=5,
        depth_workers=2,
    )
    assert record["depth_workers"] == 2
    assert [page["depth_index"] for page in record["pages"]] == [0, 2]
    for page_record in record["pages"]:
        depth = page_record["depth_index"]
        expected = NativeTissueRaster(
            geometry,
            pixels[depth],
            [region for region in regions if region.depth == depth],
            depth=depth,
            timepoint=4,
            pupil_diameter_um=8,
            scale_bar_um=5,
        ).render_tile(0, 43, 0, 61)
        with tifffile.TiffFile(page_record["tif"]) as image:
            actual = image.pages[0].asarray()
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual[2, 2], pixels[depth][2, 2])
            metadata = json.loads(image.pages[0].description)["AOOverlay"]
            assert metadata["depth_index"] == depth
            assert metadata["timepoint"] == 4
            assert metadata["source_mip_sha256"] == hashes[depth]
        assert page_record["source_mip_sha256"] == hashes[depth]
        assert any(name == sources[depth].filename for name, _ in ao_memmap_reads)
    assert all(shape[0] <= 16 and shape[1] <= 16 for _, shape in ao_memmap_reads)
