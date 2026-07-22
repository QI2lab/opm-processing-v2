# opm-processing-v2

<!-- [![License](https://img.shields.io/pypi/l/opm-processing-v2.svg?color=green)](https://github.com/QI2lab/opm-processing-v2/blob/2b85d72afad0bbd6e2c52c1b733a5b5ac211a9ab/LICENSE)
# [![PyPI](https://img.shields.io/pypi/v/opm-processing-v2.svg?color=green)](https://pypi.org/project/opm-processing-v2)
# [![Python Version](https://img.shields.io/pypi/pyversions/opm-processing-v2.svg?color=green)](https://python.org)
[![CI](https://github.com/qi2lab/opm-processing-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/qi2lab/opm-processing-v2/actions/workflows/ci.yml)
# [![codecov](https://codecov.io/gh/qi2lab/opm-processing-v2/branch/main/graph/badge.svg)](https://codecov.io/gh/qi2lab/opm-processing-v2) -->

## Overview 

This package is the 2nd generation of the Arizona State University Quantitative Imaging and Inference Lab (qi2lab) oblique plane microscopy (OPM) processing software. Currently, it assumes that data is generated using (1) our [2nd generation OPM control code](https://github.com/QI2lab/opm-v2) or (2) the [ASI single-objective light sheet](https://www.asiimaging.com/products/light-sheet-microscopy/single-objective-light-sheet/) Micromanager plugin. The ASI instrument support is experimental and will continue to evolve as we get more data examples from "in the wild" instruments. 

The core algorithms can be used for any microscope that acquires data at a skewed angle, including diSPIM, LLSM, or OPM. Please open an issue if you would like help adapting the code to work with your microscope, we are happy to assist.

The pipeline uses [yaozarrs](https://github.com/imaging-formats/yaozarrs/) for current OPM-v2 [OME-Zarr v0.5](https://ngff.openmicroscopy.org/0.5/) metadata and collection traversal, then exposes per-position image series through [TensorStore](https://google.github.io/tensorstore/) for processing. Legacy Zarr v2 root-array acquisitions remain supported. Multi-position outputs use the Bio-Formats2Raw collection layout, with one `TCZYX` OME-Zarr image series per position and a validated `OME/METADATA.ome.xml` companion. Fused outputs are single `TCZYX` OME-Zarr images. All arrays use regular chunks without sharding. Image processing (illumination correction, deconvolution, deskewing, downsampling, maximum Z projection, and 3D stitching and fusion) uses [Numba](https://numba.pydata.org/), [CuPy](https://cupy.dev/), and [cuCIM](https://github.com/rapidsai/cucim?tab=readme-ov-file).

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

Run the substantive CUDA correctness tests with GPU availability required,
```bash
OPM_REQUIRE_GPU=1 uv run --extra gpu --group dev pytest -m gpu -v
```

Without ``OPM_REQUIRE_GPU=1`` the same tests skip cleanly on CPU-only machines.
The required mode is intended for GPU workstations and GPU CI so a missing
driver, CuPy, or cuCIM cannot produce a silently green test run.

For development tools, include the `dev` group. Add `--extra gpu` on a GPU
workstation,
```bash
uv sync --group dev
```

On native Windows, cuCIM is not available as a standard wheel. This project
uses only the pure-Python `cucim.skimage` package, which can run with the
Windows CuPy wheel installed by the GPU extra. The compiled `cucim.clara`
image-I/O package is not available with this installation.

After syncing the GPU extra, enable Windows Developer Mode or open an
administrator PowerShell terminal so Git can create symbolic links. Configure
Git before cloning cuCIM, then install its Python package from the local
checkout. Current versions of `uv` require a local directory for editable
installs, and the RAPIDS CUDA packaging logic must be disabled for this
platform:

```powershell
git config --global --replace-all core.symlinks true
git clone -c core.symlinks=true `
    --branch v26.06.00 `
    --depth 1 `
    https://github.com/rapidsai/cucim.git `
    .\src\cucim

$env:RAPIDS_DISABLE_CUDA = "true"
uv pip install --no-deps -e .\src\cucim\python\cucim
Remove-Item Env:\RAPIDS_DISABLE_CUDA
```

Verify that CuPy and the cuCIM image-processing package import successfully:

```powershell
uv run --no-sync python -c "import cupy; import cucim; from cucim.skimage.registration import phase_cross_correlation; print(cucim.__version__, 'cuCIM skimage OK')"
```

On Windows, keep `--no-sync` on subsequent GPU commands. An exact `uv run`
sync does not know about the manually installed local cuCIM package and removes
it; omitting the GPU extra also removes CuPy. Re-run the sync and local cuCIM
installation steps whenever project dependencies change.

The CUDA 12 PyPI component wheels intentionally use separate package roots for
the runtime and NVRTC libraries, so there is no valid single `CUDA_PATH` to set.
CuPy documents this directly in its
[`_get_cuda_path` implementation](https://github.com/cupy/cupy/blob/v14.1.1/cupy/_environment.py#L142-L191).
On Windows this package suppresses CuPy's corresponding path warning only when
both component-wheel layouts are present; CUDA availability and execution are
still checked normally.

## Usage

Inspect an acquisition before processing it:

```bash
uv run inspect-opm "/path/to/acquisition_directory"
```

The path may be the `.ome.zarr` store itself or a directory containing exactly
one acquisition store. This command uses yaozarrs to read only Zarr and OME
metadata; it does not open TensorStore or read image chunks. Its JSON manifest
reports the scan mode, logical `TPCZYX` shape, tile and scan-position counts,
channel names, wavelengths, exposure times and powers, scan-axis spacing,
pixel size, camera calibration, tile start/end positions, OPM orientations,
and acquisition sidecars.

Process a raw acquisition with:

```bash
uv run process "/path/to/qi2lab_acquisition.zarr"
```

Flatfield estimation is controlled only by `--flatfield-correction`. When it is
enabled, the pipeline constructs `BaSiC()` with the installed BaSiCPy defaults;
BaSiCPy automatically selects CUDA when PyTorch reports it as available.

Current opm-v2 acquisitions are OME-Zarr v0.5 Bio-Formats2Raw collections:
each tile is stored as a `TCZYX` image series. The processing and timelapse
conversion commands expose those series as one virtual `TPCZYX` TensorStore,
without copying or reshaping the source data. Legacy root-array Zarr v2
acquisitions remain supported by the compatibility path.

The source acquisition remains unchanged. Processing creates OME-Zarr v0.5 outputs next to it. Per-position outputs are Bio-Formats2Raw collections; the position is represented by a separate `TCZYX` image series rather than a `P` axis inside an array. Maximum projections retain a singleton `Z` axis.

For an oblique acquisition with stem `qi2lab_acquisition`, processing can create:

| Output | Contents |
| --- | --- |
| `qi2lab_acquisition_deskewed.ome.zarr` | Deskewed per-position `TCZYX` collection |
| `qi2lab_acquisition_decon_deskewed.ome.zarr` | Deconvolved and deskewed per-position `TCZYX` collection |
| `qi2lab_acquisition_max_z_deskewed.ome.zarr` | Per-position maximum-Z projections |
| `qi2lab_acquisition_max_z_decon_deskewed.ome.zarr` | Deconvolved per-position maximum-Z projections |
| `qi2lab_acquisition_max_z_fused.ome.zarr` | Stage-position fused maximum-Z image |

The `decon` filenames are selected when `--deconvolve` is enabled. Maximum-Z
outputs are controlled by `--max-projection` and
`--create-fused-max-projection`.

For a projection acquisition, processing can create:

| Output | Contents |
| --- | --- |
| `qi2lab_acquisition_projection.ome.zarr` | Per-position `TCZYX` collection with singleton `Z` |
| `qi2lab_acquisition_decon_projection.ome.zarr` | Deconvolved per-position collection with singleton `Z` |
| `qi2lab_acquisition_stagefused.ome.zarr` | Stage-position fused projection image |

Per-position collections include axis, physical scale, channel, processing, and
stage-position metadata. Their image and stage metadata is also represented in
`OME/METADATA.ome.xml`. A namespaced `opm_processing` root attribute is written
through `yaozarrs.Bf2RawBuilder.extra_attributes`; it records the source,
software versions, timestamp, output kind, input selection, and ordered
raw-to-processed operations with resolved parameters. Camera calibration, scan
spacing/direction, excess-frame cropping, wavelengths, and stage orientation are
read from acquisition metadata rather than supplied as CLI options. Fused images
retain their `TCZYX` axes and physical scales in OME-Zarr metadata.

### Display

Open processed data through the `napari-ome-zarr` reader with:

```bash
uv run display "/path/to/qi2lab_acquisition.zarr" --to-display full
```

The command passes the OME-Zarr path directly to `napari-ome-zarr`; it does not load the complete array before opening napari. Use one of these views:

| `--to-display` | Output selected |
| --- | --- |
| `full` | Deskewed per-position collection |
| `max-z` | Per-position maximum-Z collection (default) |
| `fused-max-z` | Stage-position fused maximum-Z image |
| `fused-full` | Registered and fused multiscale image |

For collection views, `--time-range START STOP` and `--pos-range START STOP`
limit the visible range. Position layers retain their stage translations, and
channel display settings are linked across positions.

### Registration And Fusion

Register and fuse processed tiles into a multiscale OME-Zarr v0.5 image with:

```bash
uv run fuse "/path/to/qi2lab_acquisition.zarr"
```

The path may be the acquisition `.ome.zarr` store itself or its containing
directory, even when that directory already contains processed Zarr outputs.

Fusion always reports its selected registration and scale-0 fusion backends.
To require CUDA registration and fail on an incomplete GPU environment on
Linux, run:

```bash
uv run --extra gpu fuse "/path/to/qi2lab_acquisition.zarr" --require-gpu
```

On Windows, after the GPU sync and local cuCIM installation above, use:

```powershell
uv run --no-sync fuse "C:\path\to\qi2lab_acquisition.zarr" --require-gpu
```

CUDA accelerates registration. Scale-0 fusion directly copies single-tile
regions and reserves feather blending for actual overlaps. Independent output
blocks render concurrently through GIL-releasing Numba CPU kernels, each source
tile is read once per block, and each completed block is written once. Fused
shapes are padded only to the smallest boundary that supports all configured
multiscale factors, avoiding tile-sized padding.

The command discovers the corresponding deskewed or projection collection and writes `/path/to/qi2lab_acquisition_fused.ome.zarr`. Registration is optimized per timepoint, and fusion writes a single chunked `TCZYX` image with an OME-Zarr multiscale pyramid. Open it with `--to-display fused-full`, or open the `.ome.zarr` directory directly in napari with the `napari-ome-zarr` reader.
