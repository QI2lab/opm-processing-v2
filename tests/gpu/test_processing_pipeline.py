"""Real deskew/registration pipelines, with and without full or tiled CUDA RLGC."""

import pytest

from tests.pipeline import (
    test_qi2lab_acquisition_to_fused_ngff_v05 as test_qi2lab_acquisition_to_fused_ngff_v05,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("cupy_gpu"),
    pytest.mark.parametrize(
        ("pipeline_case", "deconvolve"),
        [
            pytest.param(
                ("stage", True, 0.4, None),
                True,
                id="True-stage-0.4um-estimated-taper",
            ),
        ],
        indirect=["pipeline_case"],
    ),
]
