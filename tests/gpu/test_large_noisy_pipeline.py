"""Writer-backed tiled RLGC and deskew of full noisy extended specimens."""

import pytest

from tests.pipeline import (
    test_large_noisy_scene_acquisition_to_fusion as test_large_noisy_scene_acquisition_to_fusion,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("cupy_gpu"),
    pytest.mark.parametrize("deconvolve", [True]),
    pytest.mark.parametrize(
        ("noisy_extended_scene", "large_noisy_acquisition"),
        [
            pytest.param(
                (kind, "uniform", 1, layout, True),
                (mode, factor),
                id=f"{kind.split('-')[0]}-{layout}{50 if layout == 'whole' else 200}-{mode}-{'native' if factor == 1 else 'sparse3'}",
            )
            for kind, layout, mode, factor in (
                ("line-pairs", "tiled", "mirror", 1),
                ("sphere", "whole", "stage", 3),
            )
        ],
        indirect=True,
    ),
]
