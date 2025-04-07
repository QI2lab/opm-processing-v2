# opm-processing-v2

[![License](https://img.shields.io/pypi/l/opm-processing-v2.svg?color=green)](https://github.com/qi2lab/opm-processing-v2/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/opm-processing-v2.svg?color=green)](https://pypi.org/project/opm-processing-v2)
[![Python Version](https://img.shields.io/pypi/pyversions/opm-processing-v2.svg?color=green)](https://python.org)
[![CI](https://github.com/qi2lab/opm-processing-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/qi2lab/opm-processing-v2/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/qi2lab/opm-processing-v2/branch/main/graph/badge.svg)](https://codecov.io/gh/qi2lab/opm-processing-v2)

## Overview 

This package is the 2nd generation of the Quantitative Imaging and Inference (qi2lab) oblique plane microscopy (OPM) processing software. Currently, it assumes that data is generated using the [2nd generation qi2lab OPM control code](https://github.com/QI2lab/opm-v2). The core algorithms can be used for any microscope that acquires data at a skewed angle, including diSPIM, LLSM, or OPM. Please open an issue if you would like help adapting the code to work with your microscope, we are happy to assist.

The goal is provide highly performant data I/O using Tensorstore combined deskewing, downsampling, and maximum Z projection operations using Numba.

Additionally uses [BaSiCPy](https://github.com/peng-lab/BaSiCPy) to estimate illumination profiles and [multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher) to register and fuse tiled data into ome-zarr v0.4 format.

Raw data deconvolution is planned, but will require GPU-accleration.

## Installation

Create a python 3.12 environment,
```bash
conda create -n opmprocessing python=3.12
```

activate the environment,
```bash
conda activate opmprocessing
```

If on Linux, we can use an Nvidia GPU to accelerate flatfield calculation
```bash
conda install -c conda-forge -c nvidia -c rapidsai cupy=13.4 cucim=25.02 pycudadecon "cuda-version>=12.0,<=12.8" cudnn cutensor nccl
```

and install the repository
```bash
pip install "opm-processing-v2 @ git+https://github.com/QI2lab/opm-processing-v2"
```

## Usage

Activate the conda environment.

To deskew raw data,
```bash
deskew "/path/to/qi2lab_acquisition.zarr"
```

The defaults parameters generate three zarr3 compliant datastores:
1. Full 3D data (`/path/to/qi2lab_acquisition_deskewed.zarr`) with dimensions `tpczyx`.
2. Maximum Z projections (`/path/to/qi2lab_acquisition_max_z_deskewed.zarr`) with dimensions `tpcyx`. 
3. Stage-position fused maximum z projections (`/path/to/qi2lab_acquisition_maxz.zarr`) with dimensions `tcyx`.

All three datastores are camera offset and gain corrected. The fused datastore uses the provided stage positions, without optimization.

To display deskewed data, 
```bash
display "/path/to/qi2lab_acquisition.zarr" --to_display full
```

There are three `to_display` options that correspond to the three datastores described above,
1. full
2. max-z
3. fused-max-z

To register and fused desekwed data,
```bash
fuse "/path/to/qi2lab_acquisition.zarr"
```

The registered and fused data will be in `/path/to/qi2lab_acquisition_fused_deskewed.ome.zarr`