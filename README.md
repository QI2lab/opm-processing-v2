# opm-processing-v2

Calibration, deskewing, deconvolution, registration, and fusion for qi2lab
opm-v2 OME-Zarr acquisitions.

## Documentation

Read the [documentation](https://qi2lab.github.io/opm-processing-v2/), starting
with [installation](https://qi2lab.github.io/opm-processing-v2/getting-started/installation/)
and the [first workflow](https://qi2lab.github.io/opm-processing-v2/getting-started/quickstart/).

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).
Windows and Linux installations include CUDA dependencies by default; Windows
also requires the [cuCIM setup](docs/getting-started/installation.md#cucim-on-windows).

```bash
uv sync
uv run process "/path/to/acquisition" --deconvolve
uv run fuse "/path/to/acquisition"
uv run export-ome-tiff "/path/to/acquisition"
```

## Commands

Each command's reference describes its arguments, options, and an example.
Append `--help` to any command for its installed options and defaults.

| Command | Purpose |
| --- | --- |
| [inspect-opm](docs/reference/cli/inspect-opm.md) | Inspect acquisition metadata. |
| [process](docs/reference/cli/process.md) | Calibrate, deconvolve, and deskew. |
| [fuse](docs/reference/cli/fuse.md) | Register and blend processed tiles. |
| [display](docs/reference/cli/display.md) | View outputs and select an ROI. |
| [process-ROI](docs/reference/cli/process-roi.md) | Process and fuse selected raw regions. |
| [export-ome-tiff](docs/reference/cli/export-ome-tiff.md) | Export fused images as compressed OME-BigTIFF. |
| [export-projections](docs/reference/cli/export-projections.md) | Export annotated projections and movies. |
| [encode-projections](docs/reference/cli/encode-projections.md) | Encode existing projection TIFFs as movies. |
| [convert-timelapse](docs/reference/cli/convert-timelapse.md) | Export raw scan-plane timelapses. |

## Development

See [contributing](docs/development/contributing.md),
[running tests](docs/development/testing.md), and
[simulation examples](docs/examples/index.md).

Preview the documentation in a separate environment:

```bash
uv venv .venv-docs --python 3.12
uv --no-config pip install --python .venv-docs --group docs
uv run --python .venv-docs --no-project mkdocs serve
```

BSD-3-Clause licensed; see [LICENSE](LICENSE).
