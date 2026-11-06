"""Validate native additive channel colors without materializing whole images."""

import numpy as np
import pytest
import tensorstore as ts
from scripts.ao_tissue_composite import CompositeLayerProxy, make_composite_layers

pytestmark = pytest.mark.unit

COLORS = {0: (255, 0, 255), 1: (0, 255, 255), 3: (255, 165, 0)}


def test_additive_colors_use_acquisition_channel_ids_and_clip_saturation():
    """Channel insertion order cannot swap magenta, cyan, and orange signals."""
    images = {
        3: {0: ts.array(np.asarray([[0, 0, 10, 10]], dtype=np.uint16))},
        1: {0: ts.array(np.asarray([[0, 10, 0, 10]], dtype=np.uint16))},
        0: {0: ts.array(np.asarray([[10, 0, 0, 10]], dtype=np.uint16))},
    }
    layers = make_composite_layers(
        images, COLORS, {channel: {0: (0, 10)} for channel in images}
    )
    proxy = layers[0]
    assert proxy.shape == (1, 4, 3)
    assert proxy.dtype == np.dtype(np.uint8)
    np.testing.assert_array_equal(
        proxy[:, :],
        [[[255, 0, 255], [0, 255, 255], [255, 165, 0], [255, 255, 255]]],
    )
    np.testing.assert_array_equal(proxy[:, :, 1], [[0, 255, 165, 255]])
    np.testing.assert_array_equal(proxy[:, :, ...], proxy[:, :])


def test_each_layer_uses_its_channel_limits_with_linear_rounding():
    """Half-strength fluorescence stays linear, including orange's green component."""
    image = ts.array(np.asarray([[5, 10, 15, 20, 25]], dtype=np.uint16))
    layers = make_composite_layers(
        {3: {0: image, 1: image}},
        {3: COLORS[3]},
        {3: {0: (10, 20), 1: (5, 25)}},
    )
    np.testing.assert_array_equal(
        layers[0][:, :],
        [[[0, 0, 0], [0, 0, 0], [128, 82, 0], [255, 165, 0], [255, 165, 0]]],
    )
    np.testing.assert_array_equal(layers[1][:, 2:3], [[[128, 82, 0]]])
    constant = CompositeLayerProxy(
        {0: ts.array(np.full((2, 2), 7, np.uint16))}, {0: COLORS[0]}, {0: (7, 7)}
    )
    np.testing.assert_array_equal(constant[:, :], np.zeros((2, 2, 3), np.uint8))


def test_composite_reads_only_bounded_native_or_strided_xy_slices():
    """Huge source images support small views while refusing full materialization."""
    source = ts.array(np.full((10000, 10000), 5, dtype=np.uint16))
    proxy = CompositeLayerProxy({1: source}, {1: COLORS[1]}, {1: (0, 10)})
    with pytest.raises(TypeError, match="bounded XY slices"):
        np.asarray(proxy)
    with pytest.raises(ValueError, match="bounded XY tiles"):
        _ = proxy[:, :]
    tile = proxy[10:12, 20:23]
    assert tile.shape == (2, 3, 3)
    np.testing.assert_array_equal(tile[0, 0], [0, 128, 128])
    preview = proxy[::100, ::200]
    assert preview.shape == (100, 50, 3)


@pytest.mark.parametrize("limits", [(-1, -2), (0, float("nan")), (0, float("inf"))])
def test_invalid_composite_limits_are_rejected(limits):
    """Unknown or reversed display spans cannot silently recolor fluorescence."""
    with pytest.raises(ValueError, match="display limits"):
        CompositeLayerProxy(
            {0: ts.array(np.zeros((2, 2), np.uint16))}, {0: COLORS[0]}, {0: limits}
        )


def test_composite_requires_matching_channels_depths_and_native_shapes():
    """Missing inputs cannot be silently replaced with another channel or depth."""
    source = ts.array(np.zeros((2, 2), np.uint16))
    with pytest.raises(ValueError, match="same channels"):
        make_composite_layers({0: {0: source}}, {1: COLORS[1]}, {0: {0: (0, 1)}})
    with pytest.raises(ValueError, match="same acquisition depths"):
        make_composite_layers(
            {0: {0: source}, 1: {1: source}},
            {0: COLORS[0], 1: COLORS[1]},
            {0: {0: (0, 1)}, 1: {1: (0, 1)}},
        )
    with pytest.raises(ValueError, match="native XY shape"):
        CompositeLayerProxy(
            {0: source, 1: source[:, :1]},
            {0: COLORS[0], 1: COLORS[1]},
            {0: (0, 1), 1: (0, 1)},
        )
