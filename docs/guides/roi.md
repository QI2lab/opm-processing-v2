# Selecting and processing an ROI

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
