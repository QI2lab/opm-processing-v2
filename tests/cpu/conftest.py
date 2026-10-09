"""Select the real NumPy/SciPy registration backend for every CPU test."""

import numpy as np
import pytest
from scipy import ndimage
from skimage.exposure import match_histograms
from skimage.measure import block_reduce
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

from opm_processing.imageprocessing import tilefusion


@pytest.fixture(autouse=True)
def cpu_registration_backend(monkeypatch):
    """Run CPU tests on the production CPU operators even on CUDA workstations.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Temporarily select NumPy and the functions used by the CPU fallback.

    Returns
    -------
    None
        CPU registration and fusion execute normally; image calculations and
        outputs are not mocked. The previous backend is restored after the test.
    """
    backend = {
        "USING_GPU": False,
        "cp": None,
        "xp": np,
        "ssim_cuda": None,
        "_ssim_cpu": structural_similarity,
        "match_histograms": match_histograms,
        "block_reduce": block_reduce,
        "phase_cross_correlation": phase_cross_correlation,
        "sobel_filter": ndimage.sobel,
        "shift_filter": ndimage.shift,
        "minimum_filter": ndimage.minimum_filter,
    }
    for name, operator in backend.items():
        monkeypatch.setattr(tilefusion, name, operator)
