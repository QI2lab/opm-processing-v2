"""Preserve stage identities and three-decimal micrometer image coordinates."""

import numpy as np
import pytest
from ome_types import from_xml
from yaozarrs import open_group

from opm_processing.dataio.position_collection import (
    create_position_collection,
    create_variable_position_collection,
    open_position_collection,
    update_group_attributes,
)

pytestmark = pytest.mark.unit


def test_stage_landmarks_preserve_physical_axes_at_micrometer_precision(tmp_path):
    """OME XYZ labels and the ZXY API retain stage identity to one nanometer."""
    stage_zxy_um = (
        (2409.360123, 4622.896123, -91.320987),
        (2444.640123, 5265.176543, 1011.990123),
    )
    expected_stage = ((2409.36, 4622.896, -91.321), (2444.64, 5265.177, 1011.99))
    path = tmp_path / "recorded_stage.ome.zarr"
    written = create_position_collection(
        path,
        (1, 2, 1, 4, 8, 12),
        (0.230246, 0.11549, 0.117123),
        stage_positions=stage_zxy_um,
    )
    reopened = open_position_collection(path)
    np.testing.assert_array_equal(written.stage_positions_zxy, expected_stage)
    np.testing.assert_array_equal(reopened.stage_positions_zxy, expected_stage)
    assert reopened.voxel_size_um == (0.23, 0.115, 0.117)
    ome = from_xml((path / "OME" / "METADATA.ome.xml").read_text("utf-8"))
    for image, (stage_z, stage_x, stage_y) in zip(
        ome.images, expected_stage, strict=True
    ):
        assert image.stage_label is not None
        assert image.stage_label.x == stage_x
        assert image.stage_label.y == stage_y
        assert image.stage_label.z == stage_z
        assert image.pixels.physical_size_x == 0.117
        assert image.pixels.physical_size_y == 0.115
        assert image.pixels.physical_size_z == 0.23


@pytest.mark.parametrize("sampling", ("stride", "block_mean"))
def test_each_position_retains_its_own_local_image_origin(tmp_path, sampling):
    """Per-position image coordinates use three decimals in micrometers."""
    origins = ((0.05751, -697.200123, 0.012345), (0.17251, -642.480123, -0.098765))
    expected_origins = ((0.058, -697.2, 0.012), (0.173, -642.48, -0.099))
    path = tmp_path / "local_origins.ome.zarr"
    written = create_position_collection(
        path,
        (1, 2, 1, 4, 8, 12),
        (0.230246, 0.120123, 0.120123),
        spatial_offsets_zyx_um=origins,
        multiscale_factors_yx=(2, 4),
        multiscale_downsample=sampling,
    )
    reopened = open_position_collection(path)
    np.testing.assert_array_equal(written.spatial_origins_zyx_um, expected_origins)
    np.testing.assert_array_equal(reopened.spatial_origins_zyx_um, expected_origins)
    assert reopened.multiscale_downsample == sampling
    assert reopened.voxel_size_um == (0.23, 0.12, 0.12)
    root = open_group(path)
    for position, origin in enumerate(expected_origins):
        metadata = root[str(position)].ome_metadata()
        for factor, dataset in zip(
            (1, 2, 4), metadata.multiscales[0].datasets, strict=True
        ):
            shift = (factor - 1) * 0.12 / 2 if sampling == "block_mean" else 0
            expected_origin = np.add(origin, (0, shift, shift))
            translation = dataset.translation_transform.translation[-3:]
            np.testing.assert_allclose(translation, expected_origin, rtol=0, atol=1e-12)
            for value in (*translation, *dataset.scale_transform.scale[-3:]):
                assert value == round(value, 3)


def test_common_image_metadata_has_three_micrometer_decimals(tmp_path):
    """Local origins, calibrations and stage coordinates share micrometer precision."""
    path = tmp_path / "spatial_precision.ome.zarr"
    create_position_collection(
        path,
        (1, 1, 1, 1, 8, 12),
        (0.230246, 0.11549, 0.117123),
        stage_positions=((1.234567, 123.456789, -12.345678),),
        spatial_offset_um=(0.05751, -697.200123, 4622.896123),
    )
    root = open_group(path)
    dataset = root["0"].ome_metadata().multiscales[0].datasets[0]
    values = [
        *dataset.translation_transform.translation[-3:],
        *dataset.scale_transform.scale[-3:],
        *root.attrs["opm_processing_stage_positions"]["positions"][0],
    ]
    for value in values:
        assert value == round(value, 3)


def test_variable_roi_stage_identity_is_separate_from_image_origin(tmp_path):
    """Cropped ROI origins must not replace the original physical stage labels."""
    path = tmp_path / "roi_stage.ome.zarr"
    create_variable_position_collection(
        path,
        ((1, 1, 3, 8, 12), (1, 1, 2, 4, 6)),
        (0.230246, 0.11549, 0.117123),
        spatial_origins_zyx_um=(
            (0.05751, -697.200123, 0.012345),
            (0.17251, -642.480123, -0.098765),
        ),
        stage_positions=(
            (2409.360123, 4622.896123, -91.320987),
            (2444.640123, 5265.176543, 1011.990123),
        ),
    )
    reopened = open_position_collection(path)
    expected_stage = ((2409.36, 4622.896, -91.321), (2444.64, 5265.177, 1011.99))
    np.testing.assert_array_equal(reopened.stage_positions_zxy, expected_stage)
    np.testing.assert_array_equal(
        reopened.spatial_origins_zyx_um,
        ((0.058, -697.2, 0.012), (0.173, -642.48, -0.099)),
    )
    ome = from_xml((path / "OME" / "METADATA.ome.xml").read_text("utf-8"))
    for image, (z, stage_x, stage_y) in zip(ome.images, expected_stage, strict=True):
        assert image.stage_label is not None
        assert (image.stage_label.z, image.stage_label.x, image.stage_label.y) == (
            z,
            stage_x,
            stage_y,
        )


def test_millimeter_stage_coordinates_convert_before_micrometer_rounding(tmp_path):
    """Retain millimeter precision through conversion to three-decimal micrometers."""
    path = tmp_path / "millimeter_stage.ome.zarr"
    create_position_collection(path, (1, 1, 1, 1, 4, 6), (0.23, 0.115, 0.115))
    update_group_attributes(
        path,
        {
            "opm_processing_stage_positions": {
                "axes": ["z", "x", "y"],
                "unit": "millimeter",
                "positions": [[0.001234567, 0.123456789, -0.012345678]],
            }
        },
    )
    assert open_position_collection(path).stage_positions_zxy == (
        (1.235, 123.457, -12.346),
    )
