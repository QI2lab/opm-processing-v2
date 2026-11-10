"""Automatically chunked RLGC from physical acquisition through fused datastore."""

import pytest

from tests.pipeline import (
    test_extended_mosaic_acquisition_to_fusion as test_extended_mosaic_acquisition_to_fusion,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("cupy_gpu"),
    pytest.mark.parametrize(
        ("mosaic_acquisition", "deconvolve"),
        [
            pytest.param(("stage", 0.4, 0, "whole"), False, id="stage-native-whole"),
            pytest.param(("stage", 1.2, 0, "tiled"), True, id="stage-sparse3-tiled"),
        ],
        indirect=["mosaic_acquisition"],
    ),
]
