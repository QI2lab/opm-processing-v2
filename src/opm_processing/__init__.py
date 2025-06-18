"""qi2lab OPM post-processing package v2"""
import os
os.environ.setdefault("NVRTC_OPTIONS", "--std=c++17")
os.environ.setdefault("CCCL_IGNORE_DEPRECATED_CPP_DIALECT", "1")

__version__ = '0.6.0'
__author__ = "Douglas Shepherd"
__email__ = "douglas.shepherd@asu.edu"