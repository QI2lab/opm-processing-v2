# Command reference

Run commands with `uv run` from the repository checkout. Each page gives the
command's purpose, arguments, options, and an example. Append `--help` to any
command for its installed options and defaults.

Indices are zero-based. Range arguments use an inclusive start and exclusive
stop. Supply tuple options as separate values, such as
`--max-registration-shift-zyx 20 250 100`.

| Command | Purpose |
| --- | --- |
| [inspect-opm](inspect-opm.md) | Print acquisition metadata. |
| [process](process.md) | Calibrate, deconvolve, and deskew an acquisition. |
| [fuse](fuse.md) | Register and blend processed tiles. |
| [display](display.md) | View processed images and select an ROI. |
| [process-ROI](process-roi.md) | Process and fuse a selected raw-data region. |
| [export-ome-tiff](export-ome-tiff.md) | Write fused images as compressed OME-BigTIFF. |
| [export-projections](export-projections.md) | Write annotated projections and optional movies. |
| [encode-projections](encode-projections.md) | Encode existing projection TIFF sequences. |
| [convert-timelapse](convert-timelapse.md) | Export raw scan-plane timelapses. |
