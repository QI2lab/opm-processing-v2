"""CPU parameterization of the shared qi2lab-to-NGFF integration scenario."""

import pytest

from tests.pipeline import (
    test_qi2lab_acquisition_to_fused_ngff_v05 as test_qi2lab_acquisition_to_fused_ngff_v05,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.parametrize("deconvolve", [False]),
    pytest.mark.parametrize(
        "pipeline_case",
        [
            (mode, illumination, step, None, False)
            for mode in ("stage", "mirror")
            for illumination in (False, True)
            for step in (0.4, 1.2)
        ],
        indirect=True,
        ids=[
            f"{mode}-{profile}-{step}um"
            for mode in ("stage", "mirror")
            for profile in ("uniform", "estimated-taper")
            for step in (0.4, 1.2)
        ],
    ),
]
