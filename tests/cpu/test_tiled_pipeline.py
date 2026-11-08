"""Extended mosaics with identical full and tiled orthogonal processing."""

import pytest

from tests.pipeline import (
    test_extended_mosaic_acquisition_to_fusion as test_extended_mosaic_acquisition_to_fusion,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.parametrize("deconvolve", [False]),
    pytest.mark.parametrize(
        "mosaic_acquisition",
        [
            pytest.param((mode, step, 0, layout), id=f"{mode}-{step}um-{layout}")
            for mode in ("stage", "mirror")
            for step in (0.4, 1.2)
            for layout in ("whole", "tiled")
        ],
        indirect=True,
    ),
]
