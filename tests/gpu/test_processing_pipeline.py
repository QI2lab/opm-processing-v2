"""Run the shared full pipeline with real full-volume and tiled CUDA RLGC."""

import pytest

from tests.pipeline import (
    test_qi2lab_acquisition_to_fused_ngff_v05 as test_qi2lab_acquisition_to_fused_ngff_v05,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("cupy_gpu"),
    pytest.mark.parametrize("deconvolve", [True]),
    pytest.mark.parametrize(
        "pipeline_case",
        [
            (mode, illumination, step, crop)
            for mode in ("stage", "mirror")
            for step in (0.4, 1.2)
            for illumination, crop in ((False, None), (True, 4))
        ],
        indirect=True,
        ids=[
            f"{mode}-{step}um-{method}"
            for mode in ("stage", "mirror")
            for step in (0.4, 1.2)
            for method in (
                "full-generated-psf",
                "tiled-generated-psf-estimated-taper",
            )
        ],
    ),
]
