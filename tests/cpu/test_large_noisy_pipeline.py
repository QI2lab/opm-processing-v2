"""Writer-backed tiled deskew of full noisy line-pair and sphere acquisitions."""

import pytest

from tests.pipeline import (
    test_large_noisy_scene_acquisition_to_fusion as test_large_noisy_scene_acquisition_to_fusion,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.parametrize("deconvolve", [False]),
    pytest.mark.parametrize(
        "noisy_extended_scene",
        [(kind, "uniform", 1) for kind in ("line-pairs", "sphere")],
        indirect=True,
        ids=("line-pairs", "sphere"),
    ),
    pytest.mark.parametrize(
        "large_noisy_acquisition",
        [("stage", 1), ("mirror", 3)],
        indirect=True,
        ids=("stage-native", "mirror-sparse3"),
    ),
]
