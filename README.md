# opm-processing-v2

[![License](https://img.shields.io/pypi/l/opm-processing-v2.svg?color=green)](https://github.com/qi2lab/opm-processing-v2/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/opm-processing-v2.svg?color=green)](https://pypi.org/project/opm-processing-v2)
[![Python Version](https://img.shields.io/pypi/pyversions/opm-processing-v2.svg?color=green)](https://python.org)
[![CI](https://github.com/qi2lab/opm-processing-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/qi2lab/opm-processing-v2/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/qi2lab/opm-processing-v2/branch/main/graph/badge.svg)](https://codecov.io/gh/qi2lab/opm-processing-v2)

## Overview 

qi2lab OPM post-processing package v2. Data reading and writing using Tensorstore. Performs Numba-accelerated deskewing, downsampling, flatfield correction, and maximum Z projection operations. Additionally uses the BaSiCPy library to estimate illumination profiles. 

Deconvolution is planned, but will require GPU-accleration. Currently, this library uses highly parallelized CPU algorithms for all calculations.

## Installation

Create a python 3.11 environment,
```bash
conda create -n opmprocessing python=3.11
```

activate the environment,
```bash
conda activate opmprocessing
```

and install the repository
```bash
pip install "opm-processing-v2 @ git+https://github.com/QI2lab/opm-processing-v2"
```

## Usage

Activate the conda environment. From the top-level directory of the repository,
```bash
postprocess "/path/to/qi2labacquition.zarr"
```

The defaults parameters generate two zarr3 compliant datastores, one for the full 3D data (`/path/to/qi2labacquition_deskewed.zarr`) with dimensions `tpczyx` and one with maximum Z projections (`/path/to/qi2labacquition_max_z_projection.zarr`) with dimensions `tpcyx`. Both datastores are camera offset and gain corrected, but only the maximum Z projection datastore has post-hoc flatfield correction estimated from the deskewed maximum Z projected data. The flatfield estimation for each channel is saved in `/path/to/qi2labacquition_flatfields.zarr`.