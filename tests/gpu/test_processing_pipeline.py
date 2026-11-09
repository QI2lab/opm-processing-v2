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
                (mode, illumination, step, 39 if deconvolve and illumination else None),
                deconvolve,
                id=f"{deconvolve}-{mode}-{step}um-{'estimated-taper' if illumination else 'uniform'}",
            )
            for deconvolve in (False, True)
            for mode in ("stage", "mirror")
            for step in (0.4, 1.2)
            for illumination in (False, True)
        ],
        indirect=["pipeline_case"],
    ),
]
