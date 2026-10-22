"""Extended mosaics with identical full and tiled orthogonal processing."""

import pytest

from tests.pipeline import (
    test_extended_mosaic_full_and_tiled_agree as test_extended_mosaic_full_and_tiled_agree,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.parametrize("deconvolve", [False]),
    pytest.mark.parametrize(
        "mosaic_acquisition",
        [
            pytest.param((mode, step), id=f"{mode}-{step}um")
            for mode, step in [
                (mode, step) for mode in ("stage", "mirror") for step in (0.4, 1.2)
            ]
        ],
        indirect=True,
    ),
]
