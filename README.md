# opm-processing-v2

<!-- [![License](https://img.shields.io/pypi/l/opm-processing-v2.svg?color=green)](https://github.com/QI2lab/opm-processing-v2/blob/2b85d72afad0bbd6e2c52c1b733a5b5ac211a9ab/LICENSE)
# [![PyPI](https://img.shields.io/pypi/v/opm-processing-v2.svg?color=green)](https://pypi.org/project/opm-processing-v2)
# [![Python Version](https://img.shields.io/pypi/pyversions/opm-processing-v2.svg?color=green)](https://python.org)
[![CI](https://github.com/qi2lab/opm-processing-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/qi2lab/opm-processing-v2/actions/workflows/ci.yml)
# [![codecov](https://codecov.io/gh/qi2lab/opm-processing-v2/branch/main/graph/badge.svg)](https://codecov.io/gh/qi2lab/opm-processing-v2) -->

## Overview 

This package is the 2nd generation of the Arizona State University Quantitative Imaging and Inference Lab (qi2lab) oblique plane microscopy (OPM) processing software. Currently, it assumes that data is generated using (1) our [2nd generation OPM control code](https://github.com/QI2lab/opm-v2) or (2) the [ASI single-objective light sheet](https://www.asiimaging.com/products/light-sheet-microscopy/single-objective-light-sheet/) Micromanager plugin. The ASI instrument support is experimental and will continue to evolve as we get more data examples from "in the wild" instruments. 

The core algorithms can be used for any microscope that acquires data at a skewed angle, including diSPIM, LLSM, or OPM. Please open an issue if you would like help adapting the code to work with your microscope, we are happy to assist.

The goal is provide highly performant data I/O via [Tensorstore](https://google.github.io/tensorstore/) and image processing (illumination correction, deconvolution, deskewing, downsampling, maximum Z projection, and 3D stitching+fusion) via [Numba](https://numba.pydata.org/), [CuPy](https://cupy.dev/), and [cuCIM](https://github.com/rapidsai/cucim?tab=readme-ov-file).

We rely on [BaSiCPy](https://github.com/peng-lab/BaSiCPy) to post-hoc estimate illumination profiles and a modified version of [gradient consensus Richardson-Lucy deconvolution](https://zenodo.org/records/10278919) to perform 3D deconvolution.

## Installation

Create a python 3.12 environment,
```bash
conda create -n opmprocessing python=3.12
```

activate the environment,
```bash
conda activate opmprocessing
```

install the repository and register the local cuda code. On Windows, this will also install the `skimage` portion of `cucim`.
```bash
pip install "opm-processing-v2 @ git+https://github.com/QI2lab/opm-processing-v2"
setup-cuda
conda deactivate opmprocessing
```

## Usage

Activate the conda environment,
```bash
conda activate opmprocessing
```

To deskew raw data,
```bash
process "/path/to/qi2lab_acquisition.zarr"
```

If you get an error, make sure you ran `setup-cuda`!

The defaults parameters generate different outputs depending if it acquisition is of oblique or projection data.

For oblique data, there are three zarr3 compliant datastores:
1. Full 3D data (`/path/to/qi2lab_acquisition_deskewed.zarr`) with dimensions `tpczyx`.
2. Maximum Z projections (`/path/to/qi2lab_acquisition_max_z_deskewed.zarr`) with dimensions `tpcyx`. 
3. Stage-position fused maximum z projections (`/path/to/qi2lab_acquisition_maxz.zarr`) with dimensions `tcyx`.

For projection data, there are two zarr3 compliant datastores:
1. Full 2D projection data (`/path/to/qi2lab_acquisition_deconvolved.zarr`) with dimensions `tpczyx`.
2. Stage-position fused 2D projection data (`/path/to/qi2lab_acquisition_fused.zarr`) with dimensions `tcyx`.

All datastores are camera offset and gain corrected. The fused datastore uses the provided stage positions, without optimization.

To display deskewed data, 
```bash
display "/path/to/qi2lab_acquisition.zarr" --to_display full
```

There are three `to_display` options that correspond to the three datastores described above,
1. full
2. max-z
3. fused-max-z
4. fused-full

To register and fuse optionally deconvolved and desekwed data into an ome-ngff v0.5 datastore,
```bash
fuse "/path/to/qi2lab_acquisition.zarr"
```

The registered, optionally deconvolved, and fused data will be in `/path/to/qi2lab_acquisition_fused_deskewed.ome.zarr`
