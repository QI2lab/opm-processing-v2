"""CPU checks of calibrated grayscale and RGB conversion to video pixels."""

import numpy as np
import pytest

from opm_processing.encode_projections import (
    image_nv12,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("minimum_shape", [(2, 2), (49, 145)])
def test_nv12_preserves_gray_calibration_and_padding(minimum_shape):
    """Black, midpoint and white map to video-range luma with neutral chroma."""
    frame = np.array([[0, 128, 255]], dtype=np.uint8)
    buffer, width, height = image_nv12(frame, minimum_shape=minimum_shape)
    planes = buffer.reshape(height * 3 // 2, width)
    np.testing.assert_array_equal(planes[0, :3], [16, 126, 235])
    assert np.all(planes[1:height] == 16)
    assert np.all(planes[height:] == 128)
    assert height >= minimum_shape[0] and width >= minimum_shape[1]
    assert height % 2 == width % 2 == 0


def test_rgb_nv12_known_bt709_primaries():
    # Published BT.709 limited-range primary values, each on a full chroma block.
    """Compare color conversion with published BT.709 primary calibration values."""
    frame = np.repeat(
        np.repeat(
            np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8),
            2,
            axis=0,
        ),
        2,
        axis=1,
    )
    buffer, width, height = image_nv12(frame)
    planes = buffer.reshape(height * 3 // 2, width)
    np.testing.assert_array_equal(planes[0, :6:2], [63, 173, 32])
    np.testing.assert_array_equal(planes[height, :6], [102, 240, 42, 26, 240, 118])
