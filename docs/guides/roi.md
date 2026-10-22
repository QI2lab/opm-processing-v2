# Interactive ROI processing and export

`export-rois` opens an existing deskewed maximum-Z preview in napari. Draw
multiple XY rectangles and close napari to reconstruct the corresponding raw
regions. Each ROI retains full Z, every channel, and every timepoint.

## Prepare a preview

Create the preview first; the ROI command does not create one automatically:

```bash
uv run process "/path/to/acquisition" --no-deconvolve
```

For multiple positions, processing creates a stage-placed maximum-Z mosaic.
If you have already run `fuse`, its registered maximum-Z projection is used
instead. A single-position timelapse can use its per-position maximum-Z store;
napari retains the time slider.

Keep the raw acquisition, preview source collection, and processing journal accessible. The preview,
recorded camera calibration, scan geometry, and tile placements determine
which raw data correspond to each rectangle.

## Select and process

```bash
uv run export-rois "/path/to/processed"
```

You can also pass the raw acquisition store or the exact preview path:

```bash
uv run export-rois "/path/to/sample_max_z_fused.ome.zarr" --deconvolve --registration-channel 1
uv run export-rois "/path/to/sample_max_z_deskewed.ome.zarr" --deconvolve --tiff
```

1. Inspect channels and, for timelapses, move the time slider.
2. In the `export ROIs` Shapes layer, draw any number of axis-aligned rectangles.
   Move, resize, or delete rectangles using napari's Shapes tools.
3. Close napari. Selections are saved before processing starts.

Selections refer to the full-resolution grid regardless of the displayed
pyramid level. Every rectangle applies to all timepoints; the displayed
channel and timepoint do not restrict the output. Closing without rectangles
creates no exports.

Raw data are camera-calibrated, optionally deconvolved with RLGC, and
orthogonally deskewed. Multi-position ROIs retain overlap for a fresh local
registration and fusion, including when the preview was already registered.
The final image is cropped to the requested physical XY bounds. Single-position
ROIs are assembled across time without measuring pairwise registration links.
`--flatfield-correction` enables the same illumination handling as `process`.
Deconvolution is disabled unless `--deconvolve` is supplied.

## Outputs and resume

The default destination is `roi_exports` beside the preview. Use `--output`
to choose another directory. Each `roi_001`, `roi_002`, ... subdirectory contains
its saved ROI, processing checkpoints, processed tiles, final
`<stem>_fused.ome.zarr`, and maximum-Z projection. The final OME-Zarr uses NGFF
v0.5 and retains global spatial coordinates and recorded temporal calibration.
The default Z binning factor is one; `--z-downsample-level` changes it explicitly.

`--tiff` additionally writes `<stem>_roi_001.ome.tiff` in each ROI directory.
These are lossless zlib-compressed OME-BigTIFF files with global origins,
per-plane positions, channel metadata, and processing provenance. OME-Zarr
outputs are retained.

Rerun the same command to reuse saved selections and compatible completed
checkpoints without reopening napari. Local registration checkpoints belong to
the ROI run; preview registration is never used as the final ROI solution.
Use `--no-resume` to draw new rectangles and restart processing. An incompatible
configuration is rejected during resume.

See the [command reference](../reference/cli/export-rois.md) for options.
