"""Preserve stage identities and three-decimal micrometer image coordinates."""

import numpy as np
import pytest
from ome_types import from_xml
from yaozarrs import open_group

from opm_processing.dataio.ngff import pyramid_yx_sampling, tczyx_image_metadata
from opm_processing.dataio.position_collection import (
    create_position_collection,
    create_variable_position_collection,
    open_position_collection,
    update_group_attributes,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "scales,origins,message",
    [
        ([(1, 2, 3)], [(0, 0, 0)], "equal integer YX scale factors"),
        ([(1, 1.5, 1.5)], [(0, 0, 0)], "equal integer YX scale factors"),
        ([(1, 2, 2)], [(0, 0.25, 0.25)], "origins do not match"),
        (
            [(1, 2, 2), (1, 4, 4)],
            [(0, 0, 0), (0, 1.5, 1.5)],
            "levels disagree on YX sampling",
        ),
    ],
)
def test_pyramid_reader_rejects_inconsistent_physical_grids(scales, origins, message):
    """Reject unsupported sampling rather than misplacing coarser pixels."""
    multiscale = tczyx_image_metadata(
        [(1, 1, 1), *scales], [(0, 0, 0), *origins]
    ).multiscales[0]
    with pytest.raises(ValueError, match=message):
        pyramid_yx_sampling(multiscale)


@pytest.mark.parametrize(
    "variable,sampling,spacing",
    [
        (False, "stride", (0.230246, 0.120123, 0.120123)),
        (False, "block_mean", (0.230246, 0.120123, 0.120123)),
        (True, "stride", (0.230246, 0.11549, 0.117123)),
    ],
)
def test_position_metadata_keeps_stage_identity_and_local_origins(
    tmp_path, variable, sampling, spacing
):
    """Round physical metadata while preserving stage axes, crop origins and pyramids."""
    stages = (
        (2409.360123, 4622.896123, -91.320987),
        (2444.640123, 5265.176543, 1011.990123),
    )
    expected_stages = ((2409.36, 4622.896, -91.321), (2444.64, 5265.177, 1011.99))
    origins = ((0.05751, -697.200123, 0.012345), (0.17251, -642.480123, -0.098765))
    expected_origins = ((0.058, -697.2, 0.012), (0.173, -642.48, -0.099))
    expected_spacing = tuple(round(value, 3) for value in spacing)
    path = tmp_path / "position_metadata.ome.zarr"
    if variable:
        written = create_variable_position_collection(
            path,
            ((1, 1, 3, 8, 12), (1, 1, 2, 4, 6)),
            spacing,
            spatial_origins_zyx_um=origins,
            stage_positions=stages,
        )
    else:
        written = create_position_collection(
            path,
            (1, 2, 1, 4, 8, 12),
            spacing,
            spatial_offsets_zyx_um=origins,
            stage_positions=stages,
            multiscale_factors_yx=(2, 4),
            multiscale_downsample=sampling,
        )
    reopened = open_position_collection(path)
    for collection in (written, reopened):
        np.testing.assert_array_equal(collection.stage_positions_zxy, expected_stages)
        np.testing.assert_array_equal(
            collection.spatial_origins_zyx_um, expected_origins
        )
        assert collection.voxel_size_um == expected_spacing
    assert reopened.multiscale_downsample == sampling
    root = open_group(path)
    for position, origin in enumerate(expected_origins):
        datasets = root[str(position)].ome_metadata().multiscales[0].datasets
        assert len(datasets) == (1 if variable else 3)
        for factor, dataset in zip((1, 2, 4), datasets, strict=False):
            shift = (factor - 1) * 0.12 / 2 if sampling == "block_mean" else 0
            translation = dataset.translation_transform.translation[-3:]
            np.testing.assert_allclose(
                translation, np.add(origin, (0, shift, shift)), rtol=0, atol=1e-12
            )
            for value in (*translation, *dataset.scale_transform.scale[-3:]):
                assert value == round(value, 3)
    ome = from_xml((path / "OME/METADATA.ome.xml").read_text("utf-8"))
    for image, (z, x, y) in zip(ome.images, expected_stages, strict=True):
        assert image.stage_label is not None
        assert (image.stage_label.z, image.stage_label.x, image.stage_label.y) == (
            z,
            x,
            y,
        )
        assert (
            image.pixels.physical_size_z,
            image.pixels.physical_size_y,
            image.pixels.physical_size_x,
        ) == expected_spacing


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
