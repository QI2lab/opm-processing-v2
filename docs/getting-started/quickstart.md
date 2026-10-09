# First workflow

Start with an acquisition directory containing its raw OME-Zarr store. Commands
also accept the store directly where described in their reference. Run from
the repository checkout after completing [installation](installation.md).

## Inspect and process

Check the recorded dimensions, channels, camera calibration, and scan geometry:

```bash
uv run inspect-opm "/path/to/acquisition"
```

Calibrate and deskew the camera data, applying deconvolution first:

```bash
uv run process "/path/to/acquisition" --deconvolve
```

Deconvolution requires a working CUDA GPU. Omit `--deconvolve` to calibrate and
deskew without deconvolution. Processed images default to uint16. See the
[processing guide](../guides/processing.md) for output selection and resume.

## Register and fuse

```bash
uv run fuse "/path/to/acquisition" --registration-channel 0
```

The channel index is zero-based. Choose a channel with signal in neighboring
tile overlaps. Directory selection prefers deconvolved, deskewed data and prints
the selected input. The result includes a registered fused volume and maximum-Z
projection. See [fusion](../guides/fusion.md) for registration and depth gains.

## View or export

View the registered projection without creating an ROI:

```bash
uv run display "/path/to/acquisition" --no-roi
```

Export the highest-resolution fused volume and maximum-Z projection:

```bash
uv run export-ome-tiff "/path/to/acquisition"
```

Open the OME-BigTIFF files through Fiji's Bio-Formats importer. See
[viewing](../guides/viewing.md) and [exports](../guides/exports.md) for larger
datasets and presentation images.
