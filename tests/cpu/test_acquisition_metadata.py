"""Metadata-only coverage for the current opm-v2 OME-Zarr writer layout."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tifffile import imread
from yaozarrs import open_group, v05

from opm_processing.dataio.acquisition import (
    inspect_acquisition,
    open_acquisition_datastore,
)
from opm_processing.dataio.convert_timelapse_data import convert_timelapse
from opm_processing.dataio.position_collection import (
    create_position_collection,
    open_position_collection,
)
from opm_processing.imageprocessing.opmtools import orthogonal_deskew
from opm_processing.process import process

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
def test_position_collection_multiscales_round_spatial_metadata(
    tmp_path: Path,
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
    )

    assert collection.multiscale_factors_yx == (1, 2, 4)
    assert [tuple(level[0].shape) for level in collection.multiscale_arrays] == [
        (1, 1, 1, 17, 21),
        (1, 1, 1, 9, 11),
        (1, 1, 1, 5, 6),
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
        assert dataset.translation_transform.translation == translation

    assert "stage_positions" not in reopened.attrs
    assert "opm_multiscale_downsample" not in reopened.attrs
    reopened_collection = open_position_collection(path)
    assert reopened_collection.voxel_size_um == (0.346, 0.115, 0.117)
    assert reopened_collection.stage_positions_zxy == ((1.235, 2.346, 3.457),)
    assert reopened_collection.spatial_origins_zyx_um == ((4.568, 5.679, 6.789),)
    opened_collection = open_position_collection(path)
    assert opened_collection.multiscale_factors_yx == (1, 2, 4)


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
def test_single_position_mirror_timelapse_processes_every_timepoint(
    current_single_position_mirror_timelapse: Path,
) -> None:
    """Deskew every timepoint from a one-position root Image acquisition."""
    path = current_single_position_mirror_timelapse
    process(
        root_path=path,
        max_projection=False,
        z_downsample_level=1,
    )

    output = open_position_collection(path.parent / "single_mirror_deskewed.ome.zarr")
    assert output.shape[:3] == (3, 1, 1)
    for time_index in range(3):
        raw = (
            np.arange(3 * 4 * 5 * 6, dtype=np.uint16).reshape(3, 4, 5, 6)[time_index]
            + 200
        )
        expected = orthogonal_deskew(
            (raw.astype(np.float32) - 100) * np.float32(0.25),
            distance=0.4,
            pixel_size=0.115,
            downsample_factor=1,
        ).astype(np.uint16)
        np.testing.assert_array_equal(
            output.arrays[0][time_index, 0].read().result(), expected
        )
    assert not (path.parent / "single_mirror_max_z_deskewed.ome.zarr").exists()
    assert not (path.parent / "single_mirror_max_z_fused.ome.zarr").exists()


@pytest.mark.integration
def test_single_position_mirror_timelapse_can_save_float32(
    current_single_position_mirror_timelapse: Path,
) -> None:
    """Preserve fractional calibrated values for the reported acquisition layout."""
    path = current_single_position_mirror_timelapse
    process(
        root_path=path,
        save_float32=True,
        max_projection=False,
        create_fused_max_projection=False,
        z_downsample_level=1,
    )

    output = open_position_collection(path.parent / "single_mirror_deskewed.ome.zarr")
    values = np.asarray(output.arrays[0][0, 0].read().result())
    assert values.dtype == np.float32
    raw = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6) + 200
    expected = orthogonal_deskew(
        (raw.astype(np.float32) - 100) * np.float32(0.25),
        distance=0.4,
        pixel_size=0.115,
        downsample_factor=1,
    )
    np.testing.assert_array_equal(values, expected)
    assert not any(key.startswith("opm_") for key in output.attributes)


@pytest.mark.integration
def test_timelapse_converter_accepts_current_collection(
    current_opm_v2_stage_scan: Path,
) -> None:
    """Converted TIFF pixels must exactly match the requested source data."""
    output_dir = current_opm_v2_stage_scan.parent / "converted"
    written = convert_timelapse(
        current_opm_v2_stage_scan.parent,
        output_dir=output_dir,
        time_range=(0, 1),
        stage_range=(0, 1),
        scan_range=(0, 1),
        create_tiff=True,
    )

    assert written == [output_dir / "pos_0_scan_0.tiff"]
    converted = imread(written[0])
    yy, xx = np.mgrid[:4, :5]
    expected = np.stack((2 * yy + xx, 100 + 2 * yy + xx))[np.newaxis, ...]
    np.testing.assert_array_equal(converted, expected)
