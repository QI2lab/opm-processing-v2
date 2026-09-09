"""Physical ROI interchange and world-to-skew coordinate coverage."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from yaozarrs import v05
from yaozarrs.write.v05 import prepare_image

from opm_processing.dataio.roi import (
    PhysicalRoi,
    roi_from_image_pixel_rectangle,
    roi_from_world_rectangle,
    validate_registered_max_projection,
    world_roi_to_skewed_bounds,
)
from opm_processing.dataio.position_collection import (
    create_position_collection,
    create_variable_position_collection,
    open_position_collection,
)
from opm_processing.dataio.processing_state import ProcessingState
from opm_processing.imageprocessing.tilefusion import TileFusion
from opm_processing import display as display_module
from opm_processing import process_roi as process_roi_module


def _record_registration(
    tmp_path: Path,
    processed_path: Path,
    tiles: list[dict[str, object]],
    *,
    roi_series: tuple[dict[str, object], ...] = (),
) -> tuple[Path, ProcessingState]:
    """Create the exact durable state required by fusion and ROI selection."""
    stem = processed_path.name.split("_", 1)[0]
    source = tmp_path / f"{stem}.ome.zarr"
    source.mkdir(exist_ok=True)
    state = ProcessingState.create(tmp_path / f"{stem}.processing.json", source)
    state.initialize_run(
        processed_path,
        configuration={},
        roi_series=roi_series,
        overwrite=True,
    )
    for time_index, position_index in sorted(
        {(int(tile["time_index"]), int(tile["position_index"])) for tile in tiles}
    ):
        state.complete_tile(processed_path, time_index, position_index)
    state.save_registration(
        processed_path,
        configuration={},
        pairwise_metrics={},
    )
    fused = tmp_path / f"{stem}_fused.ome.zarr"
    maximum = tmp_path / f"{stem}_max_z_fused.ome.zarr"
    state.complete_registration(processed_path, fused_path=fused, tiles=tiles)
    state.set_registered_max_projection(
        processed_path,
        max_projection_path=maximum,
    )
    return maximum, state


def test_napari_world_rectangle_round_trips_grid_and_all_z(tmp_path: Path) -> None:
    """Save a pyramid-independent ROI plus every overlapping stage-Z level."""
    source = tmp_path / "sample_max_z_fused.ome.zarr"
    image = v05.Image(
        multiscales=[
            v05.Multiscale(
                name="preview",
                axes=[
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=[1, 1, 1, 0.5, 0.25]),
                            v05.TranslationTransformation(
                                translation=[0, 0, 3, 10, 20]
                            ),
                        ],
                    )
                ],
            )
        ]
    )
    _, arrays = prepare_image(
        source,
        image,
        ((1, 1, 1, 8, 8), np.uint16),
        extra_attributes={},
        writer="tensorstore",
        overwrite=True,
    )
    arrays["0"].write(np.zeros((1, 1, 1, 8, 8), dtype=np.uint16)).result()
    tiles = [
        {"time_index": 0, "position_index": 2, "origin_zyx_um": [0, 10, 20]},
        {"time_index": 0, "position_index": 5, "origin_zyx_um": [4, 10, 20]},
        {"time_index": 0, "position_index": 9, "origin_zyx_um": [0, 30, 40]},
    ]
    processed = create_position_collection(
        tmp_path / "sample_projection.ome.zarr",
        (1, 10, 1, 1, 6, 8),
        (1.0, 0.5, 0.25),
        stage_positions=((0, 0, 0),) * 10,
    )
    del processed
    _record_registration(tmp_path, tmp_path / "sample_projection.ome.zarr", tiles)

    roi = roi_from_world_rectangle(
        source,
        np.asarray(
            [
                [0.0, 0.0, 0.0, 10.6, 20.3],
                [0.0, 0.0, 0.0, 12.1, 21.1],
            ]
        ),
    )
    assert roi.bounds_yx_um == (10.5, 12.5, 20.25, 21.25)
    assert roi.position_indices == (2, 5)
    assert len(roi.tile_footprints) == 2

    path = roi.write(tmp_path / "selection.json")
    reopened = PhysicalRoi.read(path)
    assert reopened == roi


def test_napari_image_pixels_are_converted_to_ngff_physical_coordinates(
    tmp_path: Path,
) -> None:
    """Apply NGFF translation after mapping the Shapes layer to Image data."""
    source = tmp_path / "sample_max_z_fused.ome.zarr"
    image = v05.Image(
        multiscales=[
            v05.Multiscale(
                name="preview",
                axes=[
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=[1, 1, 1, 0.115, 0.115]),
                            v05.TranslationTransformation(
                                translation=[0, 0, 0, -8192.189, 213.58]
                            ),
                        ],
                    )
                ],
            )
        ]
    )
    _, arrays = prepare_image(
        source,
        image,
        ((1, 1, 1, 5000, 5000), np.uint16),
        extra_attributes={},
        writer="tensorstore",
        overwrite=True,
    )
    arrays["0"][0, 0, 0, 0, 0].write(0).result()
    processed_path = tmp_path / "sample_projection.ome.zarr"
    create_position_collection(
        processed_path,
        (1, 1, 1, 1, 5000, 5000),
        (1.0, 0.115, 0.115),
        stage_positions=((0, 0, 0),),
    )
    _record_registration(
        tmp_path,
        processed_path,
        [
            {
                "time_index": 0,
                "position_index": 0,
                "origin_zyx_um": [0.0, -8192.189, 213.58],
            }
        ],
    )

    roi = roi_from_image_pixel_rectangle(
        source,
        np.asarray([[0, 0, 1808.0, 3554.0], [0, 0, 2148.0, 4262.0]]),
    )

    assert roi.bounds_yx_um == (-7984.269, -7945.169, 622.29, 703.71)


def test_roi_rejects_stage_only_fused_max_projection(tmp_path: Path) -> None:
    """A process-time stage mosaic is not a registered ROI canvas."""
    source = tmp_path / "stage_only_max.ome.zarr"
    image = v05.Image(
        multiscales=[
            v05.Multiscale(
                name="preview",
                axes=[
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=[1, 1, 1, 1, 1]),
                            v05.TranslationTransformation(translation=[0, 0, 0, 0, 0]),
                        ],
                    )
                ],
            )
        ]
    )
    prepare_image(
        source,
        image,
        ((1, 1, 1, 8, 8), np.uint16),
        extra_attributes={},
        writer="tensorstore",
        overwrite=True,
    )

    with np.testing.assert_raises_regex(ValueError, "registered maximum-Z"):
        validate_registered_max_projection(source)


def test_display_resolves_store_stem_from_acquisition_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Directory timestamps must not replace the acquisition store stem."""
    acquisition_directory = tmp_path / "20260814_122538_area_A0"
    acquisition_directory.mkdir()
    acquisition_path = acquisition_directory / "area_A0.ome.zarr"
    acquisition_path.mkdir()
    fused_path = acquisition_directory / "area_A0_max_z_fused.ome.zarr"
    fused_path.mkdir()
    monkeypatch.setattr(
        display_module,
        "resolve_acquisition_path",
        lambda path: (
            acquisition_path if Path(path) == acquisition_directory else Path(path)
        ),
    )

    assert (
        display_module._resolve_data_path(acquisition_directory, "fused-max-z")
        == fused_path
    )


def test_display_uses_standard_ome_collection_channels_and_stage_positions(
    tmp_path: Path,
) -> None:
    """Configure plugin layers without removed custom root attributes."""
    path = tmp_path / "sample_deskewed.ome.zarr"
    create_position_collection(
        path,
        (2, 2, 2, 1, 4, 4),
        (1.0, 0.5, 0.5),
        stage_positions=((1.0, 10.0, 20.0), (2.0, 30.0, 40.0)),
        channels=("488nm", "561nm"),
    )
    layers = [
        SimpleNamespace(
            multiscale=False,
            data=np.zeros((2, 1, 4, 4), dtype=np.uint16),
            visible=True,
            translate=None,
        )
        for _ in range(4)
    ]

    display_module._configure_collection_layers(
        path,
        layers,
        pos_range=(1, 2),
        time_range=(1, 2),
    )

    assert [layer.visible for layer in layers] == [False, False, True, True]
    assert [layer.data.shape for layer in layers] == [(1, 1, 4, 4)] * 4
    assert [layer.translate for layer in layers] == [
        (0.0, 1.0, 10.0, 20.0),
        (0.0, 1.0, 10.0, 20.0),
        (0.0, 2.0, 30.0, 40.0),
        (0.0, 2.0, 30.0, 40.0),
    ]


def test_display_configures_single_fused_image_without_collection_open(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A fused NGFF Image is not a Bio-Formats2Raw position collection."""
    path = tmp_path / "sample_max_z_fused.ome.zarr"
    image = v05.Image(
        multiscales=[
            v05.Multiscale(
                name="fused-preview",
                axes=[
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=[1, 1, 1, 1, 1])
                        ],
                    )
                ],
            )
        ]
    )
    prepare_image(
        path,
        image,
        ((2, 1, 1, 8, 8), np.uint16),
        extra_attributes={},
        writer="tensorstore",
        overwrite=True,
    )
    layer = SimpleNamespace(
        multiscale=False,
        data=np.zeros((2, 1, 8, 8), dtype=np.uint16),
    )

    def fail_collection_open(_path: Path) -> None:
        raise AssertionError("fused image must not use the collection loader")

    monkeypatch.setattr(
        display_module,
        "open_position_collection",
        fail_collection_open,
    )

    collection = display_module._configure_collection_layers(
        path,
        [layer],
        pos_range=None,
        time_range=(1, 2),
    )

    assert collection is None
    assert layer.data.shape == (1, 1, 8, 8)


def test_world_roi_maps_to_skewed_scan_x_and_keeps_all_camera_y() -> None:
    """Enclose the diagonal lab-Y preimage without defining a camera-Y crop."""
    bounds = world_roi_to_skewed_bounds(
        (103.0, 107.0, 202.0, 204.0),
        (100.0, 200.0),
        (10, 5, 12),
        pixel_size_um=1.0,
        scan_step_um=2.0,
        angle_deg=30.0,
        halo_scan=1,
        halo_x=2,
    )
    assert bounds is not None
    assert bounds.scan_start == 0
    assert bounds.scan_stop == 6
    assert bounds.x_start == 0
    assert bounds.x_stop == 6

    # Every skewed sample mapping into the requested lab-Y range is enclosed,
    # for every camera-Y row (which spans lab Z).
    for scan in range(10):
        for camera_y in range(5):
            lab_y = scan * 2.0 + camera_y * np.cos(np.deg2rad(30.0))
            if 3.0 <= lab_y < 7.0:
                assert bounds.scan_start <= scan < bounds.scan_stop


def test_registered_tile_origin_is_selected_per_timepoint() -> None:
    """Use registered placement for each timepoint when mapping back to a tile."""
    roi = PhysicalRoi(
        bounds_yx_um=(0.0, 2.0, 0.0, 2.0),
        source_path=Path("preview.ome.zarr"),
        grid_origin_yx_um=(0.0, 0.0),
        pixel_size_yx_um=(1.0, 1.0),
        position_indices=(4,),
        tile_footprints=(
            {
                "time_index": 0,
                "position_index": 4,
                "bounds_yx_um": [10.0, 20.0, 30.0, 40.0],
            },
            {
                "time_index": 1,
                "position_index": 4,
                "bounds_yx_um": [11.0, 21.0, 32.0, 42.0],
            },
        ),
    )
    assert roi.tile_origin_yx_um(0, 4, (99.0, 99.0)) == (10.0, 30.0)
    assert roi.tile_origin_yx_um(1, 4, (99.0, 99.0)) == (11.0, 32.0)


def test_roi_fusion_keeps_multiple_stage_z_levels_and_crops_only_yx(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Filter arbitrary source positions while preserving their complete Z span."""
    processed_path = tmp_path / "sample_projection.ome.zarr"
    collection = create_position_collection(
        processed_path,
        (1, 3, 1, 1, 4, 4),
        (1.0, 1.0, 1.0),
        stage_positions=((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (0.0, 100.0, 100.0)),
    )
    for array in collection.arrays:
        array.write(np.ones(array.shape, dtype=np.uint16)).result()
    tiles = [
        {"time_index": 0, "position_index": 0, "origin_zyx_um": [0, 0, 0]},
        {"time_index": 0, "position_index": 1, "origin_zyx_um": [5, 0, 0]},
        {"time_index": 0, "position_index": 2, "origin_zyx_um": [0, 100, 100]},
    ]
    _record_registration(tmp_path, processed_path, tiles)
    monkeypatch.setattr(
        "opm_processing.imageprocessing.tilefusion.inspect_acquisition",
        lambda _path: SimpleNamespace(
            stage_axis_flips_xyz=(False, False, True),
            angle_deg=45.0,
        ),
    )
    roi = PhysicalRoi(
        bounds_yx_um=(-2.0, 3.0, -1.0, 3.0),
        source_path=processed_path,
        grid_origin_yx_um=(0.0, 0.0),
        pixel_size_yx_um=(1.0, 1.0),
        position_indices=(0, 1),
        tile_footprints=(
            {
                "time_index": 0,
                "position_index": 0,
                "origin_zyx_um": [0.0, 0.0, 0.0],
                "bounds_yx_um": [0.0, 4.0, 0.0, 4.0],
            },
            {
                "time_index": 0,
                "position_index": 1,
                "origin_zyx_um": [5.0, 0.0, 0.0],
                "bounds_yx_um": [0.0, 4.0, 0.0, 4.0],
            },
        ),
    )

    fusion = TileFusion(
        processed_path,
        roi_selection=roi,
        max_workers=1,
        reverse_stage_y=False,
        reverse_stage_z=False,
    )
    assert fusion._source_position_indices == (0, 1)
    assert fusion._reuse_registered_roi_placements is True
    assert fusion.position_dim == 2
    assert len(fusion.position_arrays) == 2
    assert len(fusion._tile_positions) == 2
    assert {position[0] for position in fusion._tile_positions} == {0.0, 5.0}

    fusion._compute_fused_image_space()
    assert fusion.offset_um == (0.0, -2.0, -1.0)
    assert fusion.unpadded_shape == (6, 5, 4)


def test_variable_roi_tiles_are_stored_cropped_and_reopened(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """ROI series retain their independent crop shapes and physical origins."""
    processed_path = tmp_path / "sample_decon_deskewed.ome.zarr"
    shapes = ((1, 1, 2, 4, 4), (1, 1, 2, 5, 3))
    records = [
        {
            "time_index": 0,
            "position_index": 7,
            "origin_zyx_um": [0.0, 10.0, 20.0],
            "shape_tczyx": list(shapes[0]),
            "skewed_shape_syx": [2, 4, 4],
            "deskew_crop_yx": [0, 4, 0, 4],
        },
        {
            "time_index": 0,
            "position_index": 8,
            "origin_zyx_um": [0.0, 12.0, 22.0],
            "shape_tczyx": list(shapes[1]),
            "skewed_shape_syx": [2, 4, 3],
            "deskew_crop_yx": [2, 7, 0, 3],
        },
    ]
    collection = create_variable_position_collection(
        processed_path,
        shapes,
        (1.0, 1.0, 1.0),
        spatial_origins_zyx_um=(
            (0.0, 10.0, 20.0),
            (0.0, 12.0, 22.0),
        ),
    )
    collection.arrays[0].write(np.full(shapes[0], 10, dtype=np.uint16)).result()
    collection.arrays[1].write(np.full(shapes[1], 30, dtype=np.uint16)).result()

    reopened = open_position_collection(processed_path)
    assert [tuple(array.shape) for array in reopened.arrays] == list(shapes)
    assert not any(key.startswith("opm_") for key in reopened.attributes)

    roi_series = tuple(
        {
            key: record[key]
            for key in (
                "time_index",
                "position_index",
                "skewed_shape_syx",
                "deskew_crop_yx",
            )
        }
        for record in records
    )
    _record_registration(
        tmp_path,
        processed_path,
        records,
        roi_series=roi_series,
    )
    monkeypatch.setattr(
        "opm_processing.imageprocessing.tilefusion.inspect_acquisition",
        lambda _path: SimpleNamespace(
            stage_axis_flips_xyz=(False, False, False),
            angle_deg=30.0,
            scan_axis_step_um=1.0,
            pixel_size_um=1.0,
        ),
    )

    roi = PhysicalRoi(
        bounds_yx_um=(10.0, 17.0, 20.0, 25.0),
        source_path=tmp_path / "sample_max_z_fused.ome.zarr",
        grid_origin_yx_um=(10.0, 20.0),
        pixel_size_yx_um=(1.0, 1.0),
        position_indices=(7, 8),
    )
    fusion = TileFusion(
        processed_path,
        roi_selection=roi,
        max_workers=1,
        multiscale_factors=(2,),
        chunk_shape_yx=(4, 4),
    )
    assert fusion._variable_roi_tiles is True
    assert fusion._reuse_registered_roi_placements is False
    assert fusion._tile_shapes == [(2, 4, 4), (2, 5, 3)]
    assert fusion._tiles_by_time == ((0, 1),)
    fusion._compute_fused_image_space()
    assert fusion.offset_um == (0.0, 10.0, 20.0)
    assert fusion.unpadded_shape == (2, 7, 5)

    registration_calls = []

    def register_processed_roi(_fixed, _moving, **_kwargs):
        registration_calls.append(True)
        return (0.0, 0.0, 0.0), 1.0

    fusion.register_and_score = register_processed_roi
    fusion.run()
    assert registration_calls
    fused = np.asarray(fusion.fused_ts.read().result())
    assert fused.shape == (1, 1, 2, 8, 6)
    assert int(fused.max()) == 30


def test_process_roi_defaults_to_deconvolution_and_hands_positions_to_fusion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The command orchestrates deconvolution and one exact fusion selection."""
    acquisition_path = tmp_path / "sample.ome.zarr"
    acquisition_path.mkdir()
    roi_path = tmp_path / "selection.json"
    PhysicalRoi(
        bounds_yx_um=(1.0, 3.0, 2.0, 4.0),
        source_path=tmp_path / "sample_max_z_fused.ome.zarr",
        grid_origin_yx_um=(0.0, 0.0),
        pixel_size_yx_um=(1.0, 1.0),
        position_indices=(7, 3),
    ).write(roi_path)
    acquisition = SimpleNamespace(
        path=acquisition_path,
        is_2d=False,
        stage_axis_flips_xyz=(False, True, True),
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        process_roi_module, "inspect_acquisition", lambda _path: acquisition
    )
    monkeypatch.setattr(
        process_roi_module,
        "validate_registered_max_projection",
        lambda _path: None,
    )

    def fake_process_skewed(**kwargs) -> None:
        calls["process"] = kwargs

    monkeypatch.setattr(process_roi_module, "process_skewed", fake_process_skewed)
    monkeypatch.setattr(
        process_roi_module.ProcessingState,
        "read",
        lambda _path: SimpleNamespace(
            roi_series=lambda _output: (
                {"time_index": 0, "position_index": 3},
                {"time_index": 0, "position_index": 7},
            )
        ),
    )

    class FakeFusion:
        def __init__(self, **kwargs) -> None:
            calls["fusion"] = kwargs

        def run(self) -> None:
            calls["fusion_ran"] = True

    monkeypatch.setattr(process_roi_module, "TileFusion", FakeFusion)
    monkeypatch.setattr(
        process_roi_module,
        "regenerate_fused_max_projection",
        lambda source, destination: calls.update(max_z=(source, destination)),
    )

    process_roi_module.process_roi(acquisition_path, roi_path)

    process_call = calls["process"]
    assert isinstance(process_call, dict)
    assert process_call["deconvolve"] is True
    assert process_call["max_projection"] is False
    assert process_call["create_fused_max_projection"] is False
    assert process_call["resume"] is False
    fusion_call = calls["fusion"]
    assert isinstance(fusion_call, dict)
    assert fusion_call["roi_selection"].position_indices == (3, 7)
    assert calls["fusion_ran"] is True
    assert calls["max_z"] == (
        tmp_path / "sample_roi" / "sample_fused.ome.zarr",
        tmp_path / "sample_roi" / "sample_max_z_fused.ome.zarr",
    )


def test_process_roi_uses_display_default_json_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Omitting the JSON argument resolves the filename produced by display."""
    acquisition_path = tmp_path / "sample.ome.zarr"
    acquisition_path.mkdir()
    acquisition = SimpleNamespace(
        path=acquisition_path,
        is_2d=False,
        stage_axis_flips_xyz=(False, True, True),
    )
    expected_roi_path = tmp_path / "sample_roi.json"
    seen: dict[str, Path] = {}

    monkeypatch.setattr(
        process_roi_module, "inspect_acquisition", lambda _path: acquisition
    )

    def fake_read(path: Path) -> PhysicalRoi:
        seen["path"] = Path(path)
        raise RuntimeError("stop after resolving the default")

    monkeypatch.setattr(process_roi_module.PhysicalRoi, "read", fake_read)
    with np.testing.assert_raises_regex(RuntimeError, "stop after resolving"):
        process_roi_module.process_roi(acquisition_path)

    assert seen["path"] == expected_roi_path
