"""Validate NGFF tissue calibration and saved whole-tile AO assignments."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scripts import ao_tissue_geometry
from scripts.ao_tissue_geometry import (
    _compose_transform,
    _intended_geometry,
    read_tissue_geometry,
)

pytestmark = pytest.mark.unit


def test_projection_transform_composes_dataset_then_parent_in_order():
    """A parent scale also scales the dataset translation before adding its own."""
    scale, origin = _compose_transform(
        {
            "coordinateTransformations": [
                {"type": "scale", "scale": [1, 1, 1, 0.4, 0.8]},
                {"type": "translation", "translation": [0, 0, 0, 0.15, 0.3]},
            ]
        },
        {
            "coordinateTransformations": [
                {"type": "scale", "scale": [1, 1, 1, 2, 3]},
                {"type": "translation", "translation": [0, 0, 0, 20, 30]},
            ]
        },
    )
    np.testing.assert_allclose(scale, [1, 1, 1, 0.8, 2.4], rtol=0, atol=1e-12)
    np.testing.assert_allclose(origin, [0, 0, 0, 20.3, 30.9], rtol=0, atol=1e-12)


@pytest.mark.parametrize("full_resolution", [False, True])
def test_tissue_centers_follow_saved_assignments_and_selected_level_calibration(
    tmp_path, monkeypatch, full_resolution
):
    """Depth compensation and thumbnail transforms preserve physical tile centers."""
    preview = tmp_path / "specimen_max_z_fused.ome.zarr"
    datasets = []
    for index, (dy, dx, oy, ox) in enumerate(((0.1, 0.2, 0, 0), (0.4, 0.8, 0.15, 0.3))):
        datasets.append(
            {
                "path": str(index),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1, 1, 1, dy, dx]},
                    {"type": "translation", "translation": [0, 0, 0, oy, ox]},
                ],
            }
        )
    axes = [
        {"name": name, **({"unit": "micrometer"} if name in "zyx" else {})}
        for name in "tczyx"
    ]
    multiscale = {
        "axes": axes,
        "datasets": datasets,
        "coordinateTransformations": [
            {"type": "scale", "scale": [1, 1, 1, 2, 3]},
            {"type": "translation", "translation": [0, 0, 0, 20, 30]},
        ],
    }
    pixels = np.arange(600 * 800, dtype=np.float32).reshape(600, 800)
    # A zero-stride native array exercises the real large-shape selection
    # without allocating a full-resolution tissue image in the test.
    native_pixels = (
        np.broadcast_to(np.float32(123), (2400, 3200)) if full_resolution else None
    )

    class Array:
        def __init__(self, shape, data=None):
            self.shape = shape
            self.data = data

        def to_tensorstore(self):
            return self

        def __getitem__(self, selection):
            assert selection == (0, 0, 0)
            assert self.data is not None, "The full-resolution tissue must not be read"
            return SimpleNamespace(
                read=lambda: SimpleNamespace(result=lambda: self.data)
            )

    class Group(dict):
        def __init__(self, *args):
            super().__init__(*args)
            self.attrs = {"ome": {"multiscales": [multiscale]}}
            self.attrs["opm_v2"] = {
                "configuration": {
                    "acq_config": {
                        "AO": {"num_scan_positions": 3, "num_tile_positions": 3}
                    }
                }
            }

    group = Group(
        {
            "0": Array((1, 1, 1, 2400, 3200), native_pixels),
            "1": Array((1, 1, 1, 600, 800), pixels),
        }
    )
    stage = []
    tiles = []
    regions = []
    for depth in range(2):
        layer = tmp_path / f"time_0_z_{depth}_ao_grid_results"
        layer.mkdir()
        correction = 7 * depth
        ao = [
            {"x": x + correction, "y": y, "z": 100 + 100 * depth}
            for y in (15, 30, 45)
            for x in (35, 60, 85)
        ]
        experiment = []
        for scan, x in enumerate((10, 110)):
            for lateral, y in enumerate(range(0, 70, 10)):
                position = len(stage)
                z = 100 + 100 * depth
                stage.append((z, x, y))
                experiment.append(
                    {
                        "z": z,
                        "x": x + correction,
                        "lab_scan_um": x,
                        "y": y,
                        "sample_depth_um": 11.76 * depth,
                    }
                )
                center_y = 100 + 50 * scan + 1.234 * depth
                center_x = 200 + 10 * lateral
                tiles.append(
                    {
                        "time_index": 0,
                        "position_index": position,
                        "bounds_yx_um": [
                            center_y - 20,
                            center_y + 20,
                            center_x - 5,
                            center_x + 5,
                        ],
                    }
                )
        (layer / "ao_stage_positions.json").write_text(json.dumps(ao), "utf-8")
        (layer / "exp_stage_positions.json").write_text(json.dumps(experiment), "utf-8")
        regions.extend(SimpleNamespace(depth=depth, index=index) for index in range(9))
    state = SimpleNamespace(source_path=lambda: tmp_path / "raw.ome.zarr")
    monkeypatch.setattr(ao_tissue_geometry, "resolve_roi_source", lambda _: preview)
    monkeypatch.setattr(ao_tissue_geometry, "open_group", lambda _: group)
    monkeypatch.setattr(
        ao_tissue_geometry, "image_processing_state_path", lambda *_: Path("state.json")
    )
    monkeypatch.setattr(ao_tissue_geometry.ProcessingState, "read", lambda _: state)
    monkeypatch.setattr(
        ao_tissue_geometry,
        "inspect_acquisition",
        lambda _: SimpleNamespace(
            stage_positions_zxy=stage,
            stage_axis_flips_xyz=(False, False, False),
            path=tmp_path / "raw.ome.zarr",
        ),
    )
    monkeypatch.setattr(
        ao_tissue_geometry, "projection_tile_footprints", lambda _: tiles
    )
    geometry = read_tissue_geometry(
        tmp_path,
        None,
        0,
        0,
        regions,
        planned_grid=True,
        full_resolution=full_resolution,
    )
    if full_resolution:
        assert geometry.image.shape == (2400, 3200)
        assert geometry.image[0, 0] == geometry.image[-1, -1] == 123
        assert np.shares_memory(geometry.image, native_pixels)
    else:
        np.testing.assert_array_equal(geometry.image, pixels)
    assert geometry.shape_yx == (2400, 3200)
    assert geometry.origin_yx_um == (20, 30)
    assert geometry.preview_origin_yx_um == (
        (20, 30) if full_resolution else (20.3, 30.9)
    )
    np.testing.assert_allclose(
        geometry.preview_scale_yx_um,
        (0.2, 0.6) if full_resolution else (0.8, 2.4),
        rtol=0,
        atol=1e-12,
    )
    assert geometry.provenance["full_resolution"] is full_resolution
    assert geometry.provenance["native_shape_yx"] == [2400, 3200]
    assert geometry.provenance["preview_dataset"] == ("0" if full_resolution else "1")
    np.testing.assert_allclose(
        geometry.provenance["native_scale_yx_um"], (0.2, 0.6), rtol=0, atol=1e-12
    )
    assert geometry.extent_yx_um == (19.9, 499.9, 29.7, 1949.7)
    assert geometry.depth_offsets == {0: 0, 1: 11.76}
    for depth in range(2):
        centers = geometry.centers_by_depth_index[depth]
        assert set(centers) == {0, 2, 3, 5, 6, 8}
        assert centers[0] == (round(100 + 1.234 * depth, 3), 210)
        assert centers[8] == (round(150 + 1.234 * depth, 3), 250)
        assert geometry.provenance["depths"][depth]["assigned_tile_counts"] == {
            0: 3,
            1: 0,
            2: 3,
            3: 1,
            4: 0,
            5: 1,
            6: 3,
            7: 0,
            8: 3,
        }
        planned = geometry.planned_centers_by_depth_index[depth]
        assert set(planned) == set(range(9))
        assert planned[0] == (419.9, 349.7)
        assert planned[2] == (99.9, 349.7)
        assert planned[4] == (259.9, 989.7)
        assert planned[6] == (419.9, 1629.7)
        assert planned[8] == (99.9, 1629.7)
        assert geometry.planned_grid_by_depth_index[depth] == {
            "horizontal_y_um": [179.9, 339.9],
            "vertical_x_um": [669.7, 1309.7],
        }


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        (None, "recorded requested grid dimensions"),
        ((2, 3), "match the recorded requested"),
    ],
)
def test_intended_geometry_requires_recorded_dimensions_and_matching_corrections(
    shape, message
):
    """Never infer a requested grid or invent a correction for an empty cell."""
    with pytest.raises(ValueError, match=message):
        _intended_geometry(
            np.asarray([(x, y) for y in (15, 30, 45) for x in (35, 60, 85)]),
            list(range(9)),
            shape,
            (0, 90, 0, 90),
        )
