"""Stream native AO tissue pixels without changing phase or physical geometry."""

import hashlib
import json
from threading import Event

import matplotlib
import numpy as np
import pytest
import tifffile
from matplotlib.colors import Normalize
from scripts import ao_tissue_bigtiff
from scripts.ao_tissue_bigtiff import (
    NativeTissueRaster,
    native_display_limits,
    write_tissue_bigtiffs,
)

pytestmark = pytest.mark.unit


def test_streamed_uint16_percentiles_match_interpolated_numpy_quantiles(
    monkeypatch, ao_memmap_source, ao_memmap_reads
):
    """Histogram rank interpolation agrees with exact percentiles across strips."""
    pixels = np.asarray(
        [
            0,
            13,
            17,
            40,
            66,
            100,
            223,
            456,
            789,
            1234,
            2345,
            4567,
            6789,
            9000,
            12345,
            23456,
            45678,
            56789,
            60000,
            65000,
            65535,
        ],
        dtype=np.uint16,
    ).reshape(3, 7)[:, ::-1]
    source = ao_memmap_source(pixels)
    monkeypatch.setattr(ao_tissue_bigtiff, "_HISTOGRAM_BYTES", 14)
    limits = native_display_limits(source)
    np.testing.assert_allclose(
        limits, np.percentile(pixels, (1, 99.8)), rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(limits, (2.6, 65513.6), rtol=0, atol=1e-9)
    assert [shape for _, shape in ao_memmap_reads] == [(1, 7)] * 3


def test_arbitrary_native_tiles_stitch_without_alpha_or_pupil_seams(
    ao_memmap_source, ao_memmap_reads, ao_geometry_factory, ao_region_factory
):
    """Odd render boundaries preserve grid blending and asymmetric SJS colors."""
    pixels = np.full((79, 1001), 100, np.uint16)
    phase = np.repeat(
        np.repeat([[-np.pi, -np.pi / 2], [np.pi / 2, np.pi]], 4, axis=0), 4, axis=1
    )
    coordinate = np.linspace(-1, 1, 8)
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    mask = xx * xx + yy * yy > 1
    pupil = np.ma.array(phase.copy(), mask=mask.copy())
    regions = [ao_region_factory(depth=0, index=0, pupil=pupil)]
    geometry = ao_geometry_factory(
        pixels,
        shape_yx=pixels.shape,
        scale_yx_um=(0.5, 0.25),
        extent_yx_um=(-5, 34.5, 20, 270.25),
        planned_centers_by_depth_index={0: {0: (12.75, 102.625)}},
        planned_grid_by_depth_index={
            0: {"horizontal_y_um": [2.75], "vertical_x_um": [24.375]}
        },
    )
    reference = NativeTissueRaster(
        geometry,
        pixels,
        regions,
        depth=0,
        pupil_diameter_um=8,
        scale_bar_um=10,
        display_limits=(0, 200),
    ).render_tile(0, 79, 0, 1001)
    for block_height, block_width in ((7, 11), (13, 17)):
        source = ao_memmap_source(pixels)
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
        assert all(shape[0] <= 13 and shape[1] <= 17 for _, shape in ao_memmap_reads)
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
    tmp_path,
    monkeypatch,
    ao_memmap_source,
    ao_memmap_reads,
    ao_geometry_factory,
    ao_region_factory,
):
    """Lossless edge tiles decode exactly and retain anisotropic pixel calibration."""
    pixels = np.arange(35 * 53, dtype=np.uint16).reshape(35, 53)
    source = ao_memmap_source(pixels)
    mip_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
    geometry = ao_geometry_factory(
        pixels,
        images_by_depth={2: source},
        shape_yx=pixels.shape,
        scale_yx_um=(0.5, 0.25),
        extent_yx_um=(-5, 12.5, 20, 33.25),
        planned_centers_by_depth_index={2: {0: (3, 26)}},
        planned_grid_by_depth_index={
            2: {"horizontal_y_um": [2.75], "vertical_x_um": [24.375]}
        },
        layer_projection_rois_by_depth={2: {"mip_sha256": mip_hash}},
    )
    regions = [ao_region_factory(depth=2, index=0, pupil=np.ma.zeros((8, 8)))]
    monkeypatch.setattr(ao_tissue_bigtiff, "TILE_SHAPE", (16, 16))
    monkeypatch.setattr(ao_tissue_bigtiff, "_HISTOGRAM_BYTES", 212)
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


def test_rgb_native_tiles_preserve_precomposed_colors_and_transparent_pupil_holes(
    ao_memmap_source, ao_memmap_reads, ao_geometry_factory, ao_region_factory
):
    """Precomposed RGB values survive bounded rendering and arbitrary tile seams."""
    yy, xx = np.indices((41, 67))
    pixels = np.stack(
        (
            (7 * xx + 3 * yy) % 256,
            (5 * xx + 11 * yy + 17) % 256,
            (13 * xx + 2 * yy + 41) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    original = pixels.copy()
    mask = np.zeros((8, 8), dtype=bool)
    mask[3:5, 3:5] = True
    pupil = np.ma.array(np.zeros((8, 8)), mask=mask)
    regions = [ao_region_factory(depth=0, index=0, pupil=pupil)]
    geometry = ao_geometry_factory(
        pixels,
        shape_yx=(41, 67),
        scale_yx_um=(0.5, 0.25),
        extent_yx_um=(0, 20.5, 0, 16.75),
        planned_centers_by_depth_index={0: {0: (10.25, 8.375)}},
        planned_grid_by_depth_index={
            0: {"horizontal_y_um": [6.25], "vertical_x_um": [3.125]}
        },
    )
    expected = NativeTissueRaster(
        geometry, pixels, regions, depth=0, pupil_diameter_um=4, scale_bar_um=3
    ).render_tile(0, 41, 0, 67)
    for block_height, block_width in ((7, 11), (9, 13)):
        source = ao_memmap_source(pixels)
        raster = NativeTissueRaster(
            geometry, source, regions, depth=0, pupil_diameter_um=4, scale_bar_um=3
        )
        stitched = np.empty_like(pixels)
        for y0 in range(0, 41, block_height):
            for x0 in range(0, 67, block_width):
                y1, x1 = min(y0 + block_height, 41), min(x0 + block_width, 67)
                stitched[y0:y1, x0:x1] = raster.render_tile(y0, y1, x0, x1)
        np.testing.assert_array_equal(stitched, expected)
        assert all(shape[0] <= 9 and shape[1] <= 13 for _, shape in ao_memmap_reads)
        assert ao_memmap_reads
    for row, column in ((0, 0), (4, 9), (40, 0), (20, 33)):
        # The final point lies in the pupil's explicit transparent central hole.
        np.testing.assert_array_equal(expected[row, column], pixels[row, column])
    np.testing.assert_array_equal(pixels, original)
    np.testing.assert_array_equal(pupil.mask, mask)


def test_rgb_native_bigtiff_preserves_composite_channel_colors(
    tmp_path,
    monkeypatch,
    ao_memmap_source,
    ao_memmap_reads,
    ao_geometry_factory,
    ao_region_factory,
):
    """Stream magenta, cyan and orange without applying another contrast mapping."""
    pixels = np.empty((35, 53, 3), np.uint8)
    pixels[:, :17] = (255, 0, 255)
    pixels[:, 17:35] = (0, 255, 255)
    pixels[:, 35:] = (255, 128, 0)
    source = ao_memmap_source(pixels)
    channel_hashes = {"0": "0" * 64, "1": "1" * 64, "3": "3" * 64}
    composite = {"channels": [0, 1, 3], "colors": ["magenta", "cyan", "orange"]}
    geometry = ao_geometry_factory(
        pixels,
        shape_yx=(35, 53),
        scale_yx_um=(0.5, 0.25),
        extent_yx_um=(-5, 12.5, 20, 33.25),
        images_by_depth={2: source},
        display_limits_by_depth={2: (100, 101)},
        planned_centers_by_depth_index={2: {0: (3, 26)}},
        planned_grid_by_depth_index={
            2: {"horizontal_y_um": [2.75], "vertical_x_um": [24.375]}
        },
        layer_projection_rois_by_depth={2: {"mip_sha256_by_channel": channel_hashes}},
        provenance={"tissue_composite": composite},
    )
    regions = [ao_region_factory(depth=2, index=0, pupil=np.ma.zeros((8, 8)))]
    expected = NativeTissueRaster(
        geometry, pixels, regions, depth=2, pupil_diameter_um=4, scale_bar_um=3
    ).render_tile(0, 35, 0, 53)

    contrasts = []
    real_limits = ao_tissue_bigtiff.native_display_limits

    def observe_contrast(image):
        """Execute actual contrast calculation if the RGB path incorrectly requests it."""
        contrasts.append(image)
        return real_limits(image)

    monkeypatch.setattr(ao_tissue_bigtiff, "native_display_limits", observe_contrast)
    monkeypatch.setattr(ao_tissue_bigtiff, "TILE_SHAPE", (16, 16))
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
    tmp_path,
    monkeypatch,
    ao_memmap_source,
    ao_memmap_reads,
    ao_geometry_factory,
    ao_region_factory,
):
    """Independent depth files retain the correct RGB source and stable metadata order."""
    pixels = {depth: np.empty((43, 61, 3), np.uint8) for depth in (2, 0)}
    pixels[0][:] = (210, 20, 70)
    pixels[2][:] = (0, 150, 230)
    sources = {depth: ao_memmap_source(image) for depth, image in pixels.items()}
    hashes = {
        depth: hashlib.sha256(image.tobytes()).hexdigest()
        for depth, image in pixels.items()
    }
    geometry = ao_geometry_factory(
        pixels[0],
        shape_yx=(43, 61),
        scale_yx_um=(1, 0.5),
        extent_yx_um=(0, 43, 0, 30.5),
        images_by_depth=sources,
        planned_centers_by_depth_index={depth: {0: (21.5, 15.25)} for depth in pixels},
        planned_grid_by_depth_index={
            depth: {"horizontal_y_um": [10, 30], "vertical_x_um": [8]}
            for depth in pixels
        },
        layer_projection_rois_by_depth={
            depth: {"mip_sha256": value} for depth, value in hashes.items()
        },
    )
    regions = [
        ao_region_factory(depth=depth, index=0, pupil=np.ma.zeros((8, 8)))
        for depth in (2, 0)
    ]
    second_depth_started = Event()
    original_tiles = NativeTissueRaster.iter_tiles

    def overlapping_depth_tiles(raster):
        if raster.depth == 0:
            assert second_depth_started.wait(timeout=5), "Depth exports did not overlap"
        else:
            second_depth_started.set()
        yield from original_tiles(raster)

    monkeypatch.setattr(NativeTissueRaster, "iter_tiles", overlapping_depth_tiles)
    monkeypatch.setattr(ao_tissue_bigtiff, "TILE_SHAPE", (16, 16))
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
