"""Acquisition metadata units and native storage/processing integration."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from yaozarrs import open_group, v05

from opm_processing.dataio.acquisition import (
    inspect_acquisition,
    open_acquisition_datastore,
)
from opm_processing.dataio.position_collection import (
    create_position_collection,
    open_position_collection,
    update_group_attributes,
)
from opm_processing.process import process

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        ({"known_values": np.arange(3)}, {"known_values": [0, 1, 2]}),
        (
            {"gain": np.float32(0.25), "indices": (1, 2), "selected": {2, 1}},
            {"gain": 0.25, "indices": [1, 2], "selected": [1, 2]},
        ),
        (float("nan"), None),
    ],
)
def test_native_attribute_updates_preserve_known_pixels_and_ngff(
    acquisition_factory, attributes, expected
):
    """Update attributes without changing analytical pixel values or NGFF geometry."""
    values = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(1, 1, 1, 3, 4, 5)
    dataset = acquisition_factory(values)
    original = dict(open_group(dataset.path).attrs)
    if expected is not None:
        update_group_attributes(dataset.path, {"specimen": attributes})
        assert open_group(dataset.path).attrs["specimen"] == expected
    else:
        with pytest.raises(ValueError, match="JSON"):
            update_group_attributes(dataset.path, {"specimen": attributes})
    root = open_group(dataset.path)
    for key, value in original.items():
        assert root.attrs[key] == value
    np.testing.assert_array_equal(
        open_acquisition_datastore(dataset.path).read().result(), values
    )


@pytest.mark.unit
def test_scan_metadata_retains_all_physical_scan_planes(
    scan_camera_acquisition,
):
    """Retain every physical scan plane after metadata-directed crop and reversal."""
    fixture = scan_camera_acquisition
    acquisition = inspect_acquisition(fixture.path)
    start = acquisition.excess_scan_start_positions or acquisition.excess_scan_positions
    end = acquisition.excess_scan_end_positions
    assert fixture.raw_data.shape[-3] - start - end == 16
    raw = open_acquisition_datastore(acquisition)[0, 0, 0].read().result()
    ordered = raw[::-1] if acquisition.scan_axis_reversed else raw
    useful = ordered[start : ordered.shape[0] - end]
    assert useful.shape == (16, 32, 32)
    np.testing.assert_array_equal(useful, fixture.expected)


@pytest.mark.unit
@pytest.mark.parametrize("downsample", ["stride", "block_mean"])
def test_position_collection_multiscales_round_spatial_metadata(
    tmp_path: Path,
    downsample,
) -> None:
    """Round NGFF spacing/origins and preserve block-center coordinates."""
    path = tmp_path / "rounded_projection.ome.zarr"
    collection = create_position_collection(
        path,
        (1, 1, 1, 1, 17, 21),
        (0.34567, 0.11549, 0.11651),
        stage_positions=((1.23456, 2.34567, 3.45678),),
        spatial_offset_um=(4.56789, 5.67891, 6.78912),
        multiscale_factors_yx=(2, 4),
        multiscale_downsample=downsample,
    )

    assert collection.multiscale_factors_yx == (1, 2, 4)
    reduction = np.ceil if downsample == "stride" else np.floor
    assert [tuple(level[0].shape) for level in collection.multiscale_arrays] == [
        (1, 1, 1, int(reduction(17 / f)), int(reduction(21 / f))) for f in (1, 2, 4)
    ]
    reopened = open_group(path)
    metadata = reopened["0"].ome_metadata()
    assert isinstance(metadata, v05.Image)
    datasets = metadata.multiscales[0].datasets
    expected_scales = (
        [1.0, 1.0, 0.346, 0.115, 0.117],
        [1.0, 1.0, 0.346, 0.23, 0.234],
        [1.0, 1.0, 0.346, 0.46, 0.468],
    )
    expected_translations = (
        [0.0, 0.0, 4.568, 5.679, 6.789],
        [0.0, 0.0, 4.568, 5.679, 6.789],
        [0.0, 0.0, 4.568, 5.679, 6.789],
    )
    for dataset, scale, translation in zip(
        datasets, expected_scales, expected_translations, strict=False
    ):
        assert dataset.scale_transform.scale == scale
        assert dataset.translation_transform is not None
        expected_origin = np.asarray(translation)
        if downsample == "block_mean":
            factor = scale[-1] / expected_scales[0][-1]
            expected_origin[-2:] += (
                0.5 * (factor - 1) * np.array(expected_scales[0][-2:])
            )
        np.testing.assert_allclose(
            dataset.translation_transform.translation,
            expected_origin,
            rtol=0,
            atol=0.0005 + 1e-12,
        )

    assert "stage_positions" not in reopened.attrs
    assert "opm_multiscale_downsample" not in reopened.attrs
    reopened_collection = open_position_collection(path)
    assert reopened_collection.voxel_size_um == (0.346, 0.115, 0.117)
    assert reopened_collection.stage_positions_zxy == ((1.235, 2.346, 3.457),)
    assert reopened_collection.spatial_origins_zyx_um == ((4.568, 5.679, 6.789),)
    opened_collection = open_position_collection(path)
    assert opened_collection.multiscale_factors_yx == (1, 2, 4)
    assert opened_collection.multiscale_downsample == downsample


@pytest.mark.unit
def test_current_stage_metadata_is_discovered_without_array_open(
    current_opm_v2_stage_scan: Path,
) -> None:
    """Recover all acquisition metadata written into a synthetic collection."""
    metadata = inspect_acquisition(current_opm_v2_stage_scan.parent)

    assert metadata.storage_format == "opm-v2-ome-zarr-v3"
    assert metadata.mode == "stage"
    assert metadata.axes == ("t", "p", "c", "z", "y", "x")
    assert metadata.shape == (1, 2, 2, 3, 4, 5)
    assert metadata.tile_count == 2
    assert metadata.scan_position_count == 3
    assert metadata.channel_names == ("488nm", "561nm")
    assert [channel.wavelength_nm for channel in metadata.channels] == [488.0, 561.0]
    assert [channel.exposure_ms for channel in metadata.channels] == [10.0, 15.0]
    assert [channel.laser_power for channel in metadata.channels] == [12.0, 18.0]
    assert metadata.stage_positions_zxy == (
        (30.0, 100.0, 200.0),
        (30.0, 100.0, 220.0),
    )
    assert metadata.scan_start_positions_xyz == (
        (100.0, 200.0, 30.0),
        (100.0, 220.0, 30.0),
    )
    np.testing.assert_allclose(
        metadata.scan_end_positions_xyz,
        ((100.8, 200.0, 30.0), (100.8, 220.0, 30.0)),
    )
    assert metadata.scan_axis == "x"
    assert metadata.scan_axis_step_um == pytest.approx(0.4)
    assert metadata.pixel_size_um == pytest.approx(0.115)
    assert metadata.angle_deg == pytest.approx(30.0)
    assert metadata.camera_offset == pytest.approx(100.0)
    assert metadata.camera_conversion == pytest.approx(0.24)
    assert metadata.stage_axis_flips_xyz == (False, False, True)
    assert metadata.scan_axis_reversed is True


@pytest.mark.unit
def test_current_stage_collection_opens_as_virtual_tpczyx(
    current_opm_v2_stage_scan: Path,
) -> None:
    """Recover exact pixels from every virtualized position and channel."""
    metadata = inspect_acquisition(current_opm_v2_stage_scan)
    datastore = open_acquisition_datastore(metadata)

    assert metadata.is_2d is False
    assert (
        replace(
            metadata,
            axes=("t", "p", "c", "y", "x"),
            shape=(1, 2, 2, 4, 5),
        ).is_2d
        is True
    )
    assert replace(metadata, shape=(1, 2, 2, 1, 4, 5)).is_2d is True
    assert datastore.rank == 6
    assert tuple(datastore.shape) == metadata.shape
    assert tuple(datastore.domain.labels) == ("t", "", "c", "z", "y", "x")

    zz, yy, xx = np.mgrid[:3, :4, :5]
    for position in range(2):
        for channel in range(2):
            expected = 1000 * position + 100 * channel + 10 * zz + 2 * yy + xx
            actual = datastore[0, position, channel].read().result()
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.unit
def test_single_position_root_image_opens_as_virtual_tpczyx(
    current_single_position_mirror_timelapse: Path,
) -> None:
    """Normalize ome-writers' one-position Image layout with a P=1 axis."""
    path = current_single_position_mirror_timelapse
    metadata = inspect_acquisition(path)
    datastore = open_acquisition_datastore(metadata)

    assert metadata.storage_format == "opm-v2-ome-zarr-v3"
    assert metadata.mode == "mirror"
    assert metadata.axes == ("t", "p", "c", "z", "y", "x")
    assert metadata.shape == (3, 1, 1, 4, 5, 6)
    assert metadata.array_paths == ("0",)
    assert metadata.scan_axis_step_um == pytest.approx(0.4)
    assert metadata.stage_positions_zxy == ((30.0, 100.0, 200.0),)
    assert tuple(datastore.shape) == metadata.shape
    expected = (
        np.arange(3 * 1 * 4 * 5 * 6, dtype=np.uint16).reshape(3, 1, 4, 5, 6) + 200
    )
    np.testing.assert_array_equal(datastore[:, 0].read().result(), expected)


@pytest.mark.integration
@pytest.mark.parametrize("save_float32", [False, True])
def test_single_position_mirror_timelapse_preserves_calibrated_fluorescence(
    calibrated_mirror_timelapse, processing_options, save_float32
):
    """Reopen all timepoints and compare with the analytical 30-degree Jacobian."""
    run = calibrated_mirror_timelapse
    process(root_path=run.path, save_float32=save_float32, **processing_options)
    output = open_position_collection(
        run.path.parent / "single_mirror_deskewed.ome.zarr"
    )
    assert output.shape[:3] == (3, 1, 1)
    dtype = np.float32 if save_float32 else np.uint16
    for time, expected in enumerate(run.expected):
        actual = output.arrays[0][time, 0].read().result()
        assert actual.dtype == dtype
        np.testing.assert_allclose(actual, expected.astype(dtype), rtol=2e-6, atol=1e-5)
    assert "opm_v2" not in output.attributes
