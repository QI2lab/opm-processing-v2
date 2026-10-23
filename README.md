# opm-processing-v2

Calibration, deskewing, deconvolution, registration, and fusion for qi2lab
opm-v2 OME-Zarr acquisitions.

## Documentation

Read the [documentation](https://qi2lab.github.io/opm-processing-v2/), starting
with [installation](https://qi2lab.github.io/opm-processing-v2/getting-started/installation/)
and the [first workflow](https://qi2lab.github.io/opm-processing-v2/getting-started/quickstart/).

To select and reconstruct raw regions interactively, follow the
[ROI export guide](docs/guides/roi.md).

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).
Windows and Linux installations include CUDA dependencies by default; Windows
also requires the [cuCIM setup](docs/getting-started/installation.md#cucim-on-windows).
The base distribution provides the yaozarrs/TensorStore IO SDK. Processing,
visualization and their existing PyTorch dependency are in the `processing`
extra; the default checkout workflow still selects them. IO consumers install
the base distribution without extras. In a checkout, `uv sync --no-default-groups`
selects only IO dependencies.

```bash
uv sync
uv run process "/path/to/acquisition" --deconvolve
uv run fuse "/path/to/acquisition"
uv run export-ome-tiff "/path/to/acquisition"
```

Deconvolution reads camera and optical calibration from OME-Zarr. See the
[required metadata and solver parameters](docs/methods/deconvolution.md).
Missing read-noise calibration selects the no-read-noise model.

## Commands

Each command's reference describes its arguments, options, and an example.
Append `--help` to any command for its installed options and defaults.

| Command | Purpose |
| --- | --- |
| [inspect-opm](docs/reference/cli/inspect-opm.md) | Inspect acquisition metadata. |
| [process](docs/reference/cli/process.md) | Calibrate, deconvolve, and deskew. |
| [fuse](docs/reference/cli/fuse.md) | Register and blend processed tiles. |
| [export-ome-tiff](docs/reference/cli/export-ome-tiff.md) | Export fused images as compressed OME-BigTIFF. |
| [export-rois](docs/reference/cli/export-rois.md) | Select multiple ROIs and reconstruct raw regions into OME-Zarr and optional TIFF. |
| [export-projections](docs/reference/cli/export-projections.md) | Export annotated projections and movies. |
| [encode-projections](docs/reference/cli/encode-projections.md) | Encode existing projection TIFFs as movies. |

## Development

See [contributing](docs/development/contributing.md),
[running tests](docs/development/testing.md), and
[deconvolution usage](docs/examples/index.md).

Preview the documentation in a separate environment:

```bash
uv venv .venv-docs --python 3.12
uv --no-config pip install --python .venv-docs --group docs
uv run --python .venv-docs --no-project mkdocs serve
```

BSD-3-Clause licensed; see [LICENSE](LICENSE).
