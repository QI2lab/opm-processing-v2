"""qi2lab OPM post-processing package v2."""

import os

from opm_processing.cuda import suppress_spurious_cupy_cuda_path_warning


os.environ.setdefault("NVRTC_OPTIONS", "--std=c++17")
os.environ.setdefault("CCCL_IGNORE_DEPRECATED_CPP_DIALECT", "1")
suppress_spurious_cupy_cuda_path_warning()

__version__ = "0.6.0"
__author__ = "Douglas Shepherd"
__email__ = "douglas.shepherd@asu.edu"
