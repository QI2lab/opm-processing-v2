"""DCT quality selection and block fusion through the production overlap path."""

import numpy as np
import pytest

from opm_processing.dataio.roi import PhysicalRoi
from opm_processing.imageprocessing.tilefusion import _normalize_block


@pytest.mark.unit
def test_dct_fusion_selects_structure_independently_per_channel(cupy_gpu):
    """Low-frequency structure beats a flat view, including partial DCT cells."""
    from opm_processing.imageprocessing.content_fusion import content_aware_fusion

    _, y, x = np.indices((35, 37, 39))
    sharp = (
        1000
        + 200 * np.cos(np.pi * (x + 0.5) / 16)
        + 100 * np.cos(np.pi * (y + 0.5) / 16)
    ).astype(np.float32)
    samples = np.stack(
        (np.stack((sharp, sharp * 0 + 1000)), np.stack((sharp * 0 + 1000, sharp)))
    )
    geometric = np.ones_like(samples)
    geometric[..., 0, :, :] = 0
    numerator, weight = content_aware_fusion(samples, geometric)
    _normalize_block(numerator, weight, numerator)
    np.testing.assert_allclose(
        numerator[:, 1:], np.broadcast_to(sharp[1:], numerator[:, 1:].shape), rtol=2e-6
    )
    np.testing.assert_array_equal(numerator[:, 0], 0)


@pytest.mark.unit
def test_flat_content_retains_geometric_weights_and_valid_dark_pixels(cupy_gpu):
    """Equal scores feather normally; zero coverage contributes no photons."""
    from opm_processing.imageprocessing.content_fusion import content_aware_fusion

    samples = np.zeros((2, 1, 5, 7, 9), np.float32)
    samples[1] = 256
    geometric = np.ones_like(samples)
    geometric[1] = 3
    geometric[..., 0, 0, 0] = 0
    numerator, weight = content_aware_fusion(samples, geometric)
    np.testing.assert_array_equal(numerator[..., 1:], 768)
    np.testing.assert_array_equal(weight[..., 1:], 4)
    _normalize_block(numerator, weight, numerator)
    assert numerator[0, 0, 0, 0] == 0
    np.testing.assert_array_equal(numerator[..., 1:], 192)


@pytest.mark.integration
@pytest.mark.parametrize(
    "roi_depths", [False, True], ids=["footprints", "cropped-depths"]
)
def test_content_fusion_preserves_direct_channels_and_block_boundaries(
    block_fusion, roi_depths
):
    """Stored sharp/flat overlaps agree across output layouts after depth mapping."""
    _, y, x = np.indices((35, 37, 91))
    truth = (
        1000
        + 200 * np.cos(np.pi * (x + 0.5) / 16)
        + 100 * np.cos(np.pi * (y + 0.5) / 16)
    ).astype(np.uint16)
    first = np.stack((np.full((35, 37, 71), 1000, np.uint16), truth[..., :71]))
    second = np.stack((truth[..., 20:] * 2, np.zeros((35, 37, 71), np.uint16)))
    fusion = block_fusion(
        [first, second], [(0, 0, 0), (0, 0, 20)], truth.shape, content_aware=True
    )
    fusion._depth_histogram_maps = {(1, 0): np.arange(65536, dtype=np.float32) / 2}
    if roi_depths:
        fusion.roi_selection = PhysicalRoi((0, 37, 0, 91), fusion.data, (0, 0), (1, 1))
        fusion._tile_depth_indices = np.array([0, 1])
    fusion._fuse_by_blocks()
    expected = fusion.fused_ts.read().result()
    np.testing.assert_allclose(
        expected[0, 0, ..., 20:], truth[..., 20:], atol=1, rtol=0
    )
    np.testing.assert_array_equal(expected[0, 1, ..., :71], first[1])
    np.testing.assert_array_equal(expected[0, 1, ..., 71:], 0)
    fusion.write_block_shape[2] = 9
    fusion.chunk_y, fusion.chunk_x = 3, 5
    fusion._fuse_by_blocks()
    np.testing.assert_array_equal(fusion.fused_ts.read().result(), expected)


@pytest.mark.integration
def test_content_fusion_recovers_supported_mosaic(fusion_mosaic, block_fusion):
    """Recover shared object truth across wedges, absent channels and timepoints."""
    case = fusion_mosaic
    fusion = block_fusion(
        case.tiles,
        case.origins,
        case.shape,
        support=case.supports,
        present=case.present,
        profiles=case.profiles,
        time_indices=(0, 0, 0, 1, 1, 1),
        output_dtype=np.float32,
        chunk_shape_yx=(3, 5),
        block_shape=[1, 1, 9, 24, 40],
        content_aware=True,
    )
    fusion._fuse_by_blocks()
    np.testing.assert_allclose(
        fusion.fused_ts.read().result(), case.expected, rtol=2e-6, atol=1e-5
    )
