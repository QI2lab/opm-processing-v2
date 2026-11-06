"""Real native cache arrays and shared physical AO rendering inputs."""

import hashlib
from dataclasses import replace
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
        output = np.lib.format.open_memmap(
            path, mode="w+", shape=pixels.shape, dtype=pixels.dtype
        )
        output[...] = pixels
        output.flush()
        del output
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
    tmp_path, monkeypatch, ao_memmap_source, ao_geometry_factory, ao_region_factory
):
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
    background = ao_memmap_source(background)
    geometry = replace(geometry, image=background, images_by_depth={0: background})
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
