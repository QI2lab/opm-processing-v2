# opm-processing-v2

Post-processing for current qi2lab opm-v2 OME-Zarr acquisitions.

## Install

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

For deconvolution and registration on a CUDA workstation:

```bash
uv sync --extra gpu
```

Native Windows also requires the pure-Python `cucim.skimage` package from a
local cuCIM checkout because RAPIDS does not publish its standard wheel there.

## Commands

Inspect an acquisition:

```bash
uv run inspect-opm "/path/to/acquisition"
```

Process it (uint16 output by default):

```bash
uv run process "/path/to/acquisition"
uv run process "/path/to/acquisition" --deconvolve --flatfield-correction
uv run process "/path/to/acquisition" --save-float32
```

Use `--skip-empty-below VALUE` to zero empty channel tiles before illumination
correction, deconvolution, and deskew. Use `--resume` to continue from completed
tiles; without it, the selected output is overwritten.

Process during acquisition by supplying a precomputed `CYX` illumination TIFF.
The acquisition argument must be its containing directory:

```bash
uv run process "/path/to/acquisition-directory" --live "/path/to/illumination.ome.tif"
```

Register, fuse, and create the registered multiscale max-Z image:

```bash
uv run fuse "/path/to/acquisition-or-output-directory"
```

Draw and save a rectangular ROI from that registered max-Z image, then process
and fuse the selected raw-data region:

```bash
uv run display "/path/to/acquisition"
uv run process-ROI "/path/to/acquisition"
```

Both commands default to `<acquisition-stem>_roi.json`. Processing state is kept
in one `<acquisition-stem>.processing.json` beside the outputs; image stores
contain only OME/NGFF image metadata.

See every option and default with:

```bash
uv run process --help
uv run fuse --help
uv run display --help
uv run process-ROI --help
```

## Tests

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

Require rather than skip CUDA tests with:

```bash
OPM_REQUIRE_GPU=1 uv run --extra gpu --group dev pytest -m gpu
```
