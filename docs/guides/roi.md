# Exporting and processing ROIs

## Export ROIs from a fused volume or single-position timelapse

Use `export-rois` to draw several XY rectangles on an existing volume and
save each crop as a compressed OME-TIFF. Each export includes the full Z
depth, every channel, and every timepoint at the highest resolution.

### Choose the input

Run from the repository checkout after installing the processing dependencies
with `uv sync`. Pass the directory containing your processed data:

```bash
uv run export-rois "/path/to/processed"
```

Directory lookup uses the first available kind below and skips maximum-Z
projection stores. If more than one store matches that kind, pass the desired
store directly.

| Priority | Store | Supported input |
| --- | --- | --- |
| 1 | `<stem>_fused.ome.zarr` | Full fused volume, including multiscale data. |
| 2 | `<stem>_decon_deskewed.ome.zarr` | Deconvolved, deskewed collection with exactly one position. |
| 3 | `<stem>_deskewed.ome.zarr` | Deskewed collection with exactly one position. |

For a single-position timelapse, you can open its store directly without
running fusion:

```bash
uv run export-rois "/path/to/sample_decon_deskewed.ome.zarr"
```

For a specific fused dataset:

```bash
uv run export-rois "/path/to/sample_fused.ome.zarr"
```

Direct input also accepts a standalone TCZYX OME-Zarr image with another
filename. Multi-position unfused collections require [fusion](fusion.md)
before ROI export.

### Draw and export

1. Napari opens in the XY view with a thick-slice maximum-intensity projection
   covering all Z planes. Right-click the Z slider to adjust the slab thickness;
   move the slider to inspect another depth. For timelapses, use the time slider
   to inspect another timepoint. Channel visibility and contrast help inspection.
2. Keep the `export ROIs` layer selected. Its rectangle drawing tool is active
   initially. Click and drag to draw each axis-aligned XY rectangle. Zoom and pan
   as needed; the image can display a reduced pyramid level while you draw.
3. Review your rectangles. Use the Shapes selection tool to resize or move them
   and delete unwanted rectangles. Return to the rectangle tool to add more.
   Outlines remain visible as you change the Z slab or timepoint. Keep the XY
   view and rectangles aligned to its axes; rotated rectangles are rejected.
4. Close the napari window to start export. Leave the terminal running until it
   reports a `Created` path for each ROI. Progress shows timepoints, channels,
   and Z planes as the TIFFs are written.

The displayed slab, channel visibility, contrast, and current timepoint do not
restrict the exports. Every rectangle crops XY only and includes all Z planes,
channels, and timepoints from level zero. Bounds round outward to pixels and
clip to the image extent. Overlapping ROIs export independently. Closing without
any rectangles creates no exports.

### Output and metadata

Outputs are lossless zlib-compressed OME-BigTIFF files at
`roi_exports/<stem>_roi_001.ome.tiff`, numbered in shape order. The default
`roi_exports` directory is beside the selected OME-Zarr store, including when
you pass the store directly. For example, two rectangles from
`sample_decon_deskewed.ome.zarr` produce `sample_roi_001.ome.tiff` and
`sample_roi_002.ome.tiff`.

Choose a destination and compression worker count explicitly:

```bash
uv run export-rois "/path/to/sample_decon_deskewed.ome.zarr" --output "/path/to/crops" --workers 4
```

Existing export files are protected. To replace them, rerun the command with
`--overwrite`, draw your rectangles again, and close napari. Selections are
held for the current session; this command saves the TIFF crops.

Acquisition metadata and processing provenance follow `export-ome-tiff`;
stage and per-plane positions
retain the crop's global coordinates, with level-zero crop bounds recorded
in an OME annotation. For single-position collections with a local zero NGFF
origin, the OME stage label supplies the global origin. Calibrated time
intervals and per-plane elapsed times are retained. See the
[export-rois reference](../reference/cli/export-rois.md) for all options.

## Draw a rectangle

First create a registered maximum-Z projection with [fuse](fusion.md). ROI
selection uses its calibrated world coordinates.

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
