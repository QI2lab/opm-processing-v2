# opm-processing-v2

<!-- [![License](https://img.shields.io/pypi/l/opm-processing-v2.svg?color=green)](https://github.com/QI2lab/opm-processing-v2/blob/2b85d72afad0bbd6e2c52c1b733a5b5ac211a9ab/LICENSE)
# [![PyPI](https://img.shields.io/pypi/v/opm-processing-v2.svg?color=green)](https://pypi.org/project/opm-processing-v2)
# [![Python Version](https://img.shields.io/pypi/pyversions/opm-processing-v2.svg?color=green)](https://python.org)
[![CI](https://github.com/qi2lab/opm-processing-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/qi2lab/opm-processing-v2/actions/workflows/ci.yml)
# [![codecov](https://codecov.io/gh/qi2lab/opm-processing-v2/branch/main/graph/badge.svg)](https://codecov.io/gh/qi2lab/opm-processing-v2) -->

## Overview 

This package is the 2nd generation of the Arizona State University Quantitative Imaging and Inference Lab (qi2lab) oblique plane microscopy (OPM) processing software. Currently, it assumes that data is generated using (1) our [2nd generation OPM control code](https://github.com/QI2lab/opm-v2) or (2) the [ASI single-objective light sheet](https://www.asiimaging.com/products/light-sheet-microscopy/single-objective-light-sheet/) Micromanager plugin. The ASI instrument support is experimental and will continue to evolve as we get more data examples from "in the wild" instruments. 

The core algorithms can be used for any microscope that acquires data at a skewed angle, including diSPIM, LLSM, or OPM. Please open an issue if you would like help adapting the code to work with your microscope, we are happy to assist.

The pipeline reads legacy acquisition stores with [TensorStore](https://google.github.io/tensorstore/) and writes OME-Zarr v0.5 images with [yaozarrs](https://github.com/dpshepherd/yaozarrs). Multi-position outputs use yaozarrs' Bio-Formats2Raw collection layout, with one `TCZYX` image series per position. Image processing (illumination correction, deconvolution, deskewing, downsampling, maximum Z projection, and 3D stitching and fusion) uses [Numba](https://numba.pydata.org/), [CuPy](https://cupy.dev/), and [cuCIM](https://github.com/rapidsai/cucim?tab=readme-ov-file).

We rely on [BaSiCPy](https://github.com/peng-lab/BaSiCPy) to post-hoc estimate illumination profiles and a modified version of [gradient consensus Richardson-Lucy deconvolution](https://zenodo.org/records/10278919) to perform 3D deconvolution.

## Installation

This project uses one `uv` environment for processing, visualization, and
development. The optional GPU environment installs the CUDA 12.9 runtime and
toolkit wheels through the project dependencies. GPU execution also requires a
compatible NVIDIA device and driver.

Install `uv` if needed,
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, `uv` can also be installed from an existing Python environment,
```powershell
python -m pip install uv
```

Clone the repository and enter it,
```bash
git clone https://github.com/QI2lab/opm-processing-v2
cd opm-processing-v2
```

Create and sync a CPU environment,
```bash
uv sync
```

On a machine with a compatible NVIDIA GPU and driver, include the GPU extra,
```bash
uv sync --extra gpu
```

For development tools, include the `dev` group. Add `--extra gpu` on a GPU
workstation,
```bash
uv sync --group dev
```

On Windows, cuCIM is not available as a standard wheel. After syncing the GPU
extra, install cuCIM from source in an administrator terminal,

1. Enable symbolic links for Git: `git config --global --add core.symlinks true`.
2. Install cuCIM into the UV environment:

```bash
uv pip install -e "git+https://github.com/rapidsai/cucim.git@v25.04.00#egg=cucim-cu12&subdirectory=python/cucim"
```

## Usage

To deskew raw data,
```bash
uv run process "/path/to/qi2lab_acquisition.zarr"
```

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
uv run display "/path/to/qi2lab_acquisition.zarr" --to_display full
```

There are three `to_display` options that correspond to the three datastores described above,
1. full
2. max-z
3. fused-max-z

To register and fuse optionally deconvolved and desekwed data into an ome-ngff v0.5 datastore,
```bash
uv run fuse "/path/to/qi2lab_acquisition.zarr"
```

The registered, optionally deconvolved, and fused data will be in `/path/to/qi2lab_acquisition_fused_deskewed.ome.zarr`. This data can be viewed by dragging and dropping the folder into napari and selecting the `napari-ome-zarr` plugin for viewing.
