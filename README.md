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

`process-ROI` resumes by default. Rerun the same command after an interruption:
completed tiles and channels are skipped, and any channel whose write was not
checkpointed is processed again. Existing runs with tile-only checkpoints retain
their completed tiles and repeat the unfinished tile. Resume requires the same
processing settings and ROI tile mapping; use `--no-resume` to overwrite the ROI
output and start again.

Both commands default to `<acquisition-stem>_roi.json`. Processing state is kept
in one `<acquisition-stem>.processing.json` beside the outputs; image stores
contain only OME/NGFF image metadata.

When processed outputs are stored separately from the raw acquisition, pass
their directory to `process-ROI`. It reads the raw source path from the single
`<acquisition-stem>.processing.json` there, loads the ROI JSON from that directory,
and writes to its `<acquisition-stem>_roi` subdirectory. The raw acquisition must
still be accessible. You can also specify the raw data, ROI JSON, and destination
explicitly:

```bash
uv run process-ROI "/path/to/processed-outputs"
uv run process-ROI "/path/to/raw.ome.zarr" "/path/to/selection_roi.json" --output "/path/to/roi-output"
```

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
