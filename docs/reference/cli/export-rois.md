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

After moving or copying an acquisition directory, pass its current location.
Raw stores and preview source collections beside that input take precedence over
saved absolute paths. Saved selections and compatible ROI checkpoints resume
when the acquisition and preview retain their names and relative layout.

| Option | Description |
| --- | --- |
| `--output PATH` | Destination; default `roi_exports` beside the preview. With resume, a new destination can reuse selections from that default directory without copying processed data. |
| `--deconvolve / --no-deconvolve` | Crop raw tiles with PSF and registration margins, then deconvolve and deskew; default disabled. |
| `--tiff` | Also export compressed OME-BigTIFF; retain final OME-Zarr. |
| `--figures` | Save JPEGli `roi_overview.jpg` at quality 100 using napari's display settings, with unnumbered outlines and a scale bar. Images are isotropically reduced to at most 65,500 pixels per side. Both AO pupil types are exported automatically. |
| `--ao-grid` | Add white dashed acquisition-polygon boundaries and nine equal-area Grid-AO cells, with filled stars at intended cell centers, to the full overview. Implies `--figures`; uses the shared SJS geometry. |
| `--ao-root PATH` | Saved AO grid directory; otherwise discover beside the raw acquisition or preview. |
| `--ao-wavelength-nm FLOAT` | Explicit phase wavelength in nanometers when recorded AO illumination is missing or ambiguous. |
| `--flatfield-correction` | Reuse or estimate illumination correction. |
| `--content-aware` | Use GPU DCT-entropy overlap fusion of processed ROI tiles; default disabled. |
| `--save-float32` | Preserve fractional intensities; default uint16. |
| `--z-downsample-level INTEGER` | Output Z binning factor; default 1. |
| `--registration-channel INTEGER` | Required zero-based local registration channel. Use 0 for single-channel data. |
| `--decon-psf-paths PATH` | Channel-ordered measured PSFs; repeat per channel. Otherwise use recorded optical calibration or processing defaults. |
| `--decon-crop-scan INTEGER` | Retained reconstruction scan planes per chunk; default automatic. |
| `--decon-gpu-id INTEGER` | CUDA device; default 0. |
| `--decon-verbose INTEGER` | Opt-in reconstruction diagnostics: default 0 (off); 1 reports chunk geometry and iterations. |
| `--resume / --no-resume` | Default: reuse saved selections and compatible checkpoints. Disable to redraw and restart. |
| `--workers INTEGER` | TIFF compression workers; default 4. |

```bash
uv run export-rois "/path/to/processed" --registration-channel 1 --deconvolve --tiff
uv run export-rois "/path/to/processed" --registration-channel 1 --figures
uv run export-rois "/path/to/processed" --registration-channel 1 --ao-grid --resume
```

See the [ROI guide](../../guides/roi.md) for selection, output layout, metadata,
and resume behavior.
