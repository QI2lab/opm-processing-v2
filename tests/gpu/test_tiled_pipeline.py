"""Automatic memory-limited RLGC compared with full processing of identical mosaics."""

import pytest

from tests.pipeline import (
    test_extended_mosaic_full_and_tiled_agree as test_extended_mosaic_full_and_tiled_agree,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("cupy_gpu"),
    pytest.mark.parametrize("deconvolve", [True]),
    pytest.mark.parametrize(
        "mosaic_acquisition",
        [
            pytest.param((mode, step), id=f"{mode}-{step}um")
            for mode, step in [("stage", 0.4), ("mirror", 1.2)]
        ],
        indirect=True,
    ),
]
