"""Real native cache arrays and shared physical AO rendering inputs."""

import hashlib
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from scripts import ao_tissue_bigtiff
from scripts.ao_tissue_bigtiff import NativeTissueRaster, write_tissue_bigtiffs


@pytest.fixture
def ao_memmap_source(tmp_path):
    """Persist native tissue caches using the NumPy format used by production.

    Returns
    -------
    callable
        Write a real memory-mapped cached image and reopen it read-only.
    """
    counter = iter(range(10000))

    def create(pixels):
        """Persist calibrated display input without a custom array adapter."""
        path = tmp_path / f"native-cache-{next(counter)}.npy"
        np.save(path, pixels)
        return np.load(path, mmap_mode="r")

    return create


@pytest.fixture
def ao_memmap_reads(monkeypatch):
    """Observe actual NumPy memmap materialization without replacing its reads."""
    reads = []

    def observe_operation(operation):
        """Wrap one conversion without changing its native array implementation."""

        def observe(value, *args, **kwargs):
            """Record each real cached slice and execute its actual conversion."""
            if isinstance(value, np.memmap):
                reads.append((value.filename, value.shape))
            return operation(value, *args, **kwargs)

        return observe

    for name in ("asarray", "array"):
        monkeypatch.setattr(np, name, observe_operation(getattr(np, name)))
    return reads


@pytest.fixture
def phase_case(
    tmp_path,
    monkeypatch,
    ao_memmap_source,
    ao_geometry_factory,
    ao_region_factory,
    ao_rgb_field,
    ao_quadrant_pupil,
):
    """Make predicted RGB tiles with edge pupils, masked holes and existing lines."""
    background = ao_memmap_source(ao_rgb_field((35, 53)))
    mask = np.zeros((8, 8), dtype=bool)
    mask[3:5, 3:5] = True
    new_phase = ao_quadrant_pupil.data.copy()
    old_regions = [
        ao_region_factory(
            depth=0,
            index=index,
            pupil=np.ma.array(np.full((8, 8), -3 * np.pi / 4), mask=mask.copy()),
        )
        for index in range(3)
    ]
    new_regions = [
        ao_region_factory(
            depth=0, index=index, pupil=np.ma.array(new_phase.copy(), mask=mask.copy())
        )
        for index in range(3)
    ]
    geometry = ao_geometry_factory(
        background,
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
        background,
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
        background=background,
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


@pytest.fixture
def native_percentile_case(monkeypatch, ao_memmap_source):
    """Share the prescribed native image, pupils and calibrated rendering setup."""
    pixels = np.asarray(
        [
            [0, 13, 17, 40, 66, 100, 223],
            [456, 789, 1234, 2345, 4567, 6789, 9000],
            [12345, 23456, 45678, 56789, 60000, 65000, 65535],
        ],
        dtype=np.uint16,
    )[:, ::-1]
    source = ao_memmap_source(pixels)
    monkeypatch.setattr(ao_tissue_bigtiff, "_HISTOGRAM_BYTES", 14)
    return SimpleNamespace(
        pixels=pixels,
        source=source,
    )


@pytest.fixture
def native_quadrant_raster_case(
    ao_memmap_source, ao_geometry_factory, ao_region_factory, ao_quadrant_pupil
):
    """Share the prescribed native image, pupils and calibrated rendering setup."""
    pixels = np.full((79, 1001), 100, np.uint16)
    pupil = ao_quadrant_pupil.copy()
    phase, mask = pupil.data.copy(), pupil.mask.copy()
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
    source = ao_memmap_source(pixels)
    return SimpleNamespace(
        pixels=pixels,
        phase=phase,
        mask=mask,
        pupil=pupil,
        regions=regions,
        geometry=geometry,
        source=source,
    )


@pytest.fixture
def native_gray_tiff_case(
    monkeypatch, ao_memmap_source, ao_geometry_factory, ao_region_factory
):
    """Share the prescribed native image, pupils and calibrated rendering setup."""
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
    return SimpleNamespace(
        pixels=pixels,
        mip_hash=mip_hash,
        geometry=geometry,
        regions=regions,
    )


@pytest.fixture
def native_rgb_raster_case(
    ao_memmap_source, ao_geometry_factory, ao_region_factory, ao_rgb_field
):
    """Share the prescribed native image, pupils and calibrated rendering setup."""
    pixels = ao_rgb_field((41, 67))
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
    source = ao_memmap_source(pixels)
    return SimpleNamespace(
        pixels=pixels,
        original=original,
        mask=mask,
        pupil=pupil,
        regions=regions,
        geometry=geometry,
        source=source,
    )


@pytest.fixture
def native_rgb_tiff_case(monkeypatch, ao_memmap_source, native_gray_tiff_case):
    """Share the prescribed native image, pupils and calibrated rendering setup."""
    pixels = np.empty((35, 53, 3), np.uint8)
    pixels[:, :17] = (255, 0, 255)
    pixels[:, 17:35] = (0, 255, 255)
    pixels[:, 35:] = (255, 128, 0)
    source = ao_memmap_source(pixels)
    channel_hashes = {"0": "0" * 64, "1": "1" * 64, "3": "3" * 64}
    composite = {"channels": [0, 1, 3], "colors": ["magenta", "cyan", "orange"]}
    geometry = replace(
        native_gray_tiff_case.geometry,
        image=pixels,
        images_by_depth={2: source},
        display_limits_by_depth={2: (100, 101)},
        layer_projection_rois_by_depth={2: {"mip_sha256_by_channel": channel_hashes}},
        provenance={"tissue_composite": composite},
    )
    contrasts = []
    real_limits = ao_tissue_bigtiff.native_display_limits

    def observe_contrast(image):
        """Execute actual contrast calculation if the RGB path incorrectly requests it."""
        contrasts.append(image)
        return real_limits(image)

    monkeypatch.setattr(ao_tissue_bigtiff, "native_display_limits", observe_contrast)
    return SimpleNamespace(
        contrasts=contrasts,
        pixels=pixels,
        channel_hashes=channel_hashes,
        composite=composite,
        geometry=geometry,
        regions=native_gray_tiff_case.regions,
    )


@pytest.fixture
def native_parallel_case(
    monkeypatch, ao_memmap_source, ao_geometry_factory, ao_region_factory
):
    """Share the prescribed native image, pupils and calibrated rendering setup."""
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
    return SimpleNamespace(
        pixels=pixels,
        sources=sources,
        hashes=hashes,
        geometry=geometry,
        regions=regions,
    )
