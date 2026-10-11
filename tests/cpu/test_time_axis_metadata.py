"""Physical time-axis regression coverage through the actual producer writer."""

from dataclasses import replace

import pytest
from yaozarrs import open_group

from opm_processing.dataio.acquisition import inspect_acquisition
from opm_processing.dataio.position_collection import (
    create_position_collection,
    create_variable_position_collection,
    open_position_collection,
)


@pytest.mark.integration
@pytest.mark.parametrize("cropped", [False, True])
def test_written_time_axis_interval_and_units(tmp_path, cropped):
    path = tmp_path / "timed.ome.zarr"
    if cropped:
        create_variable_position_collection(
            path,
            [(3, 1, 4, 8, 8)],
            (0.23, 0.115, 0.115),
            spatial_origins_zyx_um=[(0, 0, 0)],
            time_interval_s=0.085,
        )
    else:
        create_position_collection(
            path,
            (3, 1, 1, 128, 8, 8),
            (0.23, 0.115, 0.115),
            multiscale_factors_yx=(2,),
            time_interval_s=0.085,
        )
    image = open_group(path)["0"].ome_metadata()
    assert image.multiscales[0].axes[0].unit == "second"
    assert all(
        dataset.scale_transform.scale[0] == 0.085
        for dataset in image.multiscales[0].datasets
    )
    assert open_position_collection(path).time_interval_s == 0.085


@pytest.mark.unit
def test_hardware_cadence_uses_acquired_count(current_single_position_mirror_timelapse):
    source = inspect_acquisition(current_single_position_mirror_timelapse)
    acquired = replace(source, acquired_plane_count=17, plane_period_s=0.005)
    assert acquired.volume_interval_s == pytest.approx(0.085)
    assert acquired.scan_position_count != 17
    assert (
        replace(
            source, acquired_plane_count=None, plane_period_s=None
        ).volume_interval_s
        is None
    )


@pytest.mark.integration
@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_time_scale_rejected(tmp_path, interval):
    with pytest.raises(ValueError, match="time_interval_s"):
        create_position_collection(
            tmp_path / "invalid.ome.zarr",
            (2, 1, 1, 4, 8, 8),
            (0.23, 0.115, 0.115),
            time_interval_s=interval,
        )


@pytest.mark.integration
def test_unknown_time_does_not_become_one_second(tmp_path):
    path = tmp_path / "unknown.ome.zarr"
    create_position_collection(path, (2, 1, 1, 4, 8, 8), (0.23, 0.115, 0.115))
    assert open_group(path)["0"].ome_metadata().multiscales[0].axes[0].unit is None
    assert open_position_collection(path).time_interval_s is None


@pytest.mark.integration
def test_processing_propagates_hardware_cadence(
    current_single_position_mirror_timelapse, tmp_path
):
    from opm_processing.process import process

    raw = current_single_position_mirror_timelapse
    import json

    metadata_path = raw / "zarr.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["attributes"]["opm_v2"]["timing"] = {
        "n_acquired_planes": 4,
        "dt_plane_s": 0.005,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    process(
        root_path=raw,
        max_projection=True,
        create_fused_max_projection=False,
        z_downsample_level=1,
    )
    for name in (
        "single_mirror_deskewed.ome.zarr",
        "single_mirror_max_z_deskewed.ome.zarr",
    ):
        result = open_position_collection(raw.parent / name)
        assert result.time_interval_s == pytest.approx(0.020)


@pytest.mark.integration
@pytest.mark.parametrize("projection", [False, True])
def test_fused_writer_preserves_time_axis(tmp_path, fusion_operator, projection):
    import numpy as np

    from opm_processing.imageprocessing.maxtilefusion import MaxTileFusion

    path = tmp_path / "fused.ome.zarr"
    if projection:
        operator = fusion_operator(
            MaxTileFusion,
            time_interval_s=0.085,
            time_range=None,
            time_dim=3,
            fused_shape=(8, 8),
            channels=1,
            z_pixel_size=0.23,
            pixel_size=(0.115, 0.115),
            offset=(0, 0),
            spatial_offset_z_um=0,
            output_dtype=np.dtype("uint16"),
            output_path=path,
            chunk_size=8,
        )
        operator.create_fused_image()
    else:
        operator = fusion_operator(
            time_interval_s=0.085,
            multiscale_factors=(2,),
            multiscale_downsample="stride",
            _pixel_size=(0.23, 0.115, 0.115),
            offset_um=(0, 0, 0),
            padded_shape=(4, 8, 8),
            _is_2d=False,
            time_dim=3,
            channels=1,
            chunk_y=4,
            chunk_x=4,
        )
        operator._prepare_fused_image(path, z_slices_per_write=2)
    image = open_group(path).ome_metadata()
    assert image.multiscales[0].axes[0].unit == "second"
    assert all(
        d.scale_transform.scale[0] == 0.085 for d in image.multiscales[0].datasets
    )
