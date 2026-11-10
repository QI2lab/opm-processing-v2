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
                ("stage", True, 1.2, None),
                False,
                id="False-stage-1.2um-estimated-taper",
            ),
            pytest.param(
                ("stage", False, 0.4, None),
                True,
                id="True-stage-0.4um-uniform",
            ),
            pytest.param(
                ("mirror", True, 1.2, 39),
                True,
                id="True-mirror-1.2um-estimated-taper",
            ),
        ],
        indirect=["pipeline_case"],
    ),
]
