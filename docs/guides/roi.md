# Selecting and processing an ROI

## Export ROIs from an existing fused volume

```bash
uv run export-rois "/path/to/processed"
```

This opens the full fused multiscale volume in napari with thick-slice
maximum-intensity projection enabled. The initial Z slab covers the entire
volume; right-click the Z slider to adjust its thickness and move the slider
to inspect another depth. Timepoints remain separate. Draw multiple
axis-aligned YX rectangles in the `export ROIs` layer, then close napari.
The ROI outlines remain visible when changing the Z slab or timepoint.
Each rectangle exports all Z planes, channels, and timepoints from level zero,
regardless of the displayed slab or pyramid level while drawing. Selections round
outward to pixel bounds and clip to the image extent. Overlapping ROIs export
independently; closing without any rectangles creates no exports.

Outputs are lossless zlib-compressed OME-BigTIFF files at
`roi_exports/<stem>_roi_001.ome.tiff`, numbered in shape order. Use `--output`
to choose a destination, `--overwrite` to replace existing files, and
`--workers` to configure compression workers. Acquisition metadata and
processing provenance follow `export-ome-tiff`; stage and per-plane positions
retain the crop's global coordinates, with level-zero crop bounds recorded
in an OME annotation. Draw in the default YX view; rotated rectangles are
rejected.

First create a registered maximum-Z projection with [fuse](fusion.md). ROI
selection uses its calibrated world coordinates.

## Draw a rectangle

```bash
uv run display "/path/to/processed"
```

Draw one rectangle in the napari `processing ROI` layer and close the viewer
to save it. The default destination is `<stem>_roi.json` beside the data.
`--roi-output PATH` selects another destination. ROI mode requires
`fused-max-z`; use `--no-roi` for other display modes.

## Process the selected raw region

```bash
uv run process-ROI "/path/to/processed"
```

The command resolves the raw acquisition through the processing journal,
reads the saved ROI, processes its raw-data regions, and fuses the cropped
tiles. Keep the source acquisition accessible. Deconvolution and resume are
enabled by default; output goes to `<stem>_roi` in the input directory.

Supply the raw acquisition, ROI, and destination explicitly when needed:

```bash
uv run process-ROI "/path/to/raw.ome.zarr" "/path/to/selection_roi.json" --output "/path/to/roi-output"
```

## Resume or restart

Rerun the same command after interruption. Compatible completed tile/channel
checkpoints are reused; unfinished writes are processed again. Use
`--no-resume` to overwrite the ROI output. `--crop-after-deskew` is inapplicable
because this workflow already writes the selected physical crop.

To restrict fusion of existing processed tiles without reprocessing the raw
data, use `fuse --roi PATH`. See the
[display](../reference/cli/display.md) and
[process-ROI](../reference/cli/process-roi.md) references.
