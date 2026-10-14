# export-rois

Open a volume in napari, draw multiple XY rectangles, then close the viewer
to export each rectangle as a lossless zlib-compressed OME-BigTIFF. Exports
retain every Z plane, channel, and timepoint at level zero, with calibrated
positions and available acquisition metadata.

| Argument | Description |
| --- | --- |
| `ROOT_PATH` | Processed output directory or an OME-Zarr store directly. Supports full fused images and single-position collections. |

Directory lookup prefers `*_fused.ome.zarr`, then
`*_decon_deskewed.ome.zarr`, then `*_deskewed.ome.zarr`, excluding maximum-Z
projections. Multiple matches require an explicit store path. Unfused
collections must contain exactly one position.

| Option | Description |
| --- | --- |
| `--output PATH` | Destination directory; default `roi_exports` beside the selected store. |
| `--workers INTEGER` | Compression workers; default 4, minimum 1. |
| `--overwrite` | Replace existing exports; disabled by default. |
| `--help` | Show installed arguments, options, and defaults. |

```bash
uv run export-rois "/path/to/processed"
uv run export-rois "/path/to/sample_decon_deskewed.ome.zarr" --output "/path/to/crops"
```

Napari starts with a full-Z thick-slice maximum projection. Right-click the
Z slider to adjust thickness, inspect timepoints, and draw axis-aligned
rectangles in `export ROIs`. Display settings affect inspection; exports
always include the full depth, all channels, and all timepoints.

Outputs are `<stem>_roi_001.ome.tiff`, `<stem>_roi_002.ome.tiff`, and so on,
in shape order. Closing without rectangles writes nothing. Rerunning requires
drawing new selections; existing files require `--overwrite` to replace.

See the [ROI export guide](../../guides/roi.md#export-rois-from-a-fused-volume-or-single-position-timelapse)
for the drawing workflow and metadata details.
