# Command reference

Run commands with `uv run` from the repository checkout. Each page gives the
command's purpose, arguments, options, and an example. Append `--help` to any
command for its installed options and defaults.

Indices are zero-based. Range arguments use an inclusive start and exclusive
stop. Supply tuple options as separate values, such as
`--max-registration-shift-zyx 20 250 100`.

| Command | Purpose |
| --- | --- |
| [inspect-opm](inspect-opm.md) | Report raw/processed dimensions, physical sizes and camera timing. |
| [process](process.md) | Calibrate, deconvolve, and deskew an acquisition. |
| [fuse](fuse.md) | Register and blend processed tiles. |
| [export-ome-tiff](export-ome-tiff.md) | Write fused images as compressed OME-BigTIFF. |
| [export-rois](export-rois.md) | Select multiple ROIs and reconstruct raw regions into OME-Zarr and optional TIFF. |
| [export-projections](export-projections.md) | Write annotated projections and optional movies. |
| [encode-projections](encode-projections.md) | Encode existing projection TIFF sequences. |
