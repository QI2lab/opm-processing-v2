# export-rois

Select multiple XY rectangles on a deskewed maximum-Z preview, then reconstruct
raw regions and export each as OME-Zarr. Full Z, all channels, and all timepoints
are retained. Multiple positions are locally registered and fused. With
deconvolution enabled, scan-grid refinement is inferred from the recorded scan
step toward 0.4 micrometers, using the same rule as `process`.

| Argument | Description |
| --- | --- |
| `ROOT_PATH` | Raw acquisition store, processed directory, or explicit maximum-Z preview. |

Preview lookup prefers `*_max_z_fused.ome.zarr`, then
`*_max_z_decon_deskewed.ome.zarr`, then `*_max_z_deskewed.ome.zarr`.
Unfused previews must contain exactly one position. Multiple matches require
an explicit preview path. A processing journal linking to accessible raw data
is required. Missing previews report a `uv run process` command to create one.

| Option | Description |
| --- | --- |
| `--output PATH` | Destination; default `roi_exports` beside the preview. |
| `--deconvolve / --no-deconvolve` | Enable RLGC before deskewing; default disabled. |
| `--tiff` | Also export compressed OME-BigTIFF; retain final OME-Zarr. |
| `--flatfield-correction` | Reuse or estimate illumination correction. |
| `--save-float32` | Preserve fractional intensities; default uint16. |
| `--z-downsample-level INTEGER` | Output Z binning factor; default 1. |
| `--registration-channel INTEGER` | Zero-based local registration channel; default 0. |
| `--decon-psf-paths PATH` | Channel-ordered measured PSFs; repeat per channel. Otherwise use recorded optical calibration. |
| `--decon-crop-scan INTEGER` | Retained reconstruction scan planes per chunk; default automatic. |
| `--decon-gpu-id INTEGER` | CUDA device; default 0. |
| `--decon-verbose INTEGER` | Reconstruction verbosity; default 1. |
| `--resume / --no-resume` | Default: reuse saved selections and compatible checkpoints. Disable to redraw and restart. |
| `--workers INTEGER` | TIFF compression workers; default 4. |

```bash
uv run export-rois "/path/to/processed" --deconvolve --tiff
```

See the [ROI guide](../../guides/roi.md) for selection, output layout, metadata,
and resume behavior.
