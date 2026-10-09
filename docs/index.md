# opm-processing-v2

Process qi2lab opm-v2 OME-Zarr acquisitions from camera counts to registered
volumes. The workflow provides camera calibration, illumination correction,
RLGC deconvolution, deskewing, tile registration, fusion, and image export.

## Start here

1. [Install](getting-started/installation.md) Python dependencies and GPU support.
2. Follow the [first workflow](getting-started/quickstart.md) on an acquisition.
3. Use the [command reference](reference/cli/index.md) for arguments and options.

## Find the right documentation

| Task | Page |
| --- | --- |
| Process a dataset or resume a run | [Processing guide](guides/processing.md) |
| Process while acquisition is running | [Live acquisition](guides/live.md) |
| Register and combine stage tiles | [Registration and fusion](guides/fusion.md) |
| Select a smaller raw-data region | [ROI guide](guides/roi.md) |
| Save OME-TIFFs, projections, or movies | [Export guide](guides/exports.md) |
| Open outputs in napari or Fiji | [Viewing and troubleshooting](guides/viewing.md) |
| Understand axes, output files, and metadata | [Data reference](reference/data.md) |
| Understand reconstruction calculations | [Processing methods](methods/deskew.md) |
| Run a simulated acquisition | [Simulation examples](examples/index.md) |
| Contribute changes or run tests | [Development](development/contributing.md) |

The Python reference is collected from source docstrings. Development pages
describe contributing, running tests, and building documentation.
