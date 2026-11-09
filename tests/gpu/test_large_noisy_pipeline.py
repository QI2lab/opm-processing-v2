"""Writer-backed tiled RLGC and deskew of full noisy extended specimens."""

import pytest

from tests.pipeline import (
    test_large_noisy_scene_acquisition_to_fusion as test_large_noisy_scene_acquisition_to_fusion,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("cupy_gpu"),
    pytest.mark.parametrize("deconvolve", [False, True]),
    pytest.mark.parametrize(
        "noisy_extended_scene",
        [
            (kind, "uniform", 1, layout, True)
            for kind in ("line-pairs", "sphere")
            for layout in ("whole", "tiled")
        ],
        indirect=True,
        ids=("line-whole50", "line-tiled200", "sphere-whole50", "sphere-tiled200"),
    ),
    pytest.mark.parametrize(
        "large_noisy_acquisition",
        [(mode, factor) for mode in ("stage", "mirror") for factor in (1, 3)],
        indirect=True,
        ids=("stage-native", "stage-sparse3", "mirror-native", "mirror-sparse3"),
    ),
]
