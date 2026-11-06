"""Viewing copies average normalized display pixels with bounded native reads."""

import numpy as np
import pytest
from scripts.ao_tissue_preview import make_layer_previews

pytestmark = pytest.mark.unit


class _SlicedImage:
    """Allow bounded source slices while refusing whole-image conversion."""

    def __init__(self, pixels):
        self.pixels = pixels
        self.shape, self.dtype = pixels.shape, pixels.dtype
        self.requests = []

    def __array__(self, dtype=None, copy=None):
        raise AssertionError("Viewing copies must not convert the entire source")

    def __getitem__(self, selection):
        block = self.pixels[selection]
        assert block.shape[0] <= 1024 and block.shape[1] <= 1024
        self.requests.append(block.shape)
        return block


def test_area_mean_uses_each_layers_display_colors_and_real_edge_counts():
    """Normalize before averaging, without treating partial edge bins as zeros."""
    pixels = np.zeros((5, 7), np.uint16)
    pixels[0, 0] = 1000
    pixels[:4, 4:] = 200
    pixels[4, :4] = 300
    pixels[4, 4:] = (0, 200, 1000)
    first = _SlicedImage(pixels)
    second = _SlicedImage(np.full((5, 7), 1000, np.uint16))
    previews, factor = make_layer_previews(
        {2: first, 5: second}, {2: (100, 300), 5: (1000, 3000)}, max_edge=3
    )
    assert factor == 4
    # Display values are 0, 128 or 255. The bottom/right bins contain 1x4,
    # 4x3 and 1x3 native samples rather than a padded 4x4 square.
    expected = np.asarray([[16, 128], [255, 128]], dtype=np.uint8)
    np.testing.assert_array_equal(
        previews[2], np.repeat(expected[..., None], 3, axis=-1)
    )
    np.testing.assert_array_equal(previews[5], np.zeros((2, 2, 3), np.uint8))
    assert first.requests == second.requests == [(5, 7)]


def test_preview_reads_bounded_blocks_across_odd_native_edges():
    """A checkerboard averages identically across source-block and image edges."""
    rows, columns = np.ogrid[:1031, :1043]
    pixels = (((rows + columns) % 2) * 1000).astype(np.uint16)
    source = _SlicedImage(pixels)
    previews, factor = make_layer_previews({0: source}, {0: (0, 1000)}, max_edge=600)
    assert factor == 2
    assert previews[0].shape == (516, 522, 3)
    expected = np.full((516, 522, 3), 128, np.uint8)
    expected[-1, -1] = 0  # The final unpaired native sample is black.
    np.testing.assert_array_equal(previews[0], expected)
    assert source.requests == [(1024, 1024), (1024, 19), (7, 1024), (7, 19)]


def test_rgb_preview_preserves_channel_components_and_partial_edges():
    """Average precomposed colors independently without grayscale remapping."""
    rows, columns = np.ogrid[:1031, :1043]
    colors = np.asarray(((10, 30, 90), (250, 130, 170)), dtype=np.uint8)
    pixels = colors[(rows + columns) % 2]
    source = _SlicedImage(pixels)
    previews, factor = make_layer_previews({3: source}, {}, max_edge=600)
    assert factor == 2
    expected = np.empty((516, 522, 3), np.uint8)
    expected[:] = (130, 80, 130)
    expected[-1, -1] = (10, 30, 90)
    np.testing.assert_array_equal(previews[3], expected)
    assert source.requests == [(1024, 1024, 3), (1024, 19, 3), (7, 1024, 3), (7, 19, 3)]
