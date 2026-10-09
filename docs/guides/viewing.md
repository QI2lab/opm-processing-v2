# Viewing and troubleshooting

## napari

```bash
uv run display "/path/to/processed" --to-display fused-full --no-roi
```

Choose `full`, `max-z`, `fused-full`, or `fused-max-z`. Full and per-tile
maximum-Z modes display processed position collections. Fused modes display
the registered output. `--time-range` and `--pos-range` select visible ranges.
ROI drawing is enabled by default and requires the registered fused maximum-Z
image; see [ROI selection](roi.md).

## Fiji

Open exported OME-BigTIFF through **Plugins > Bio-Formats > Bio-Formats Importer**.
Use a virtual stack for datasets that cannot fit in memory. Confirm channel
names, channel count, and physical spacing in the imported image.

The exporter stores UTF-8 OME-XML. Earlier files written with numeric unit
references can have their metadata repaired using `export-ome-tiff
--metadata-only`; this updates both existing fused exports without changing pixels.

## Large-image browsing

Current exports are tiled at 512 x 512 and contain one resolution. Tiling enables
region reads, but the standard Bio-Formats virtual stack requests whole planes
on a cache miss. Scrolling therefore still requires substantial decoding for
large planes. [Bio-Formats implementation](https://github.com/ome/bioformats/blob/develop/components/bio-formats-plugins/src/loci/plugins/util/BFVirtualStack.java)

The [tifffile examples](https://github.com/cgohlke/tifffile#examples) demonstrate
smaller tiles and pyramidal output. Those examples do not establish an optimal
tile size for Fiji. Any layout change needs measurements with the actual viewer,
compression, and dataset. The exporter currently has no pyramid option.

For multiresolution OME-TIFF, Fiji's
[BigDataViewer Playground Bio-Formats importer](https://bigdataviewer-playground-documentation.readthedocs.io/en/latest/opening_images/opening_images.html)
provides on-demand viewing and an optional automatic pyramid. This is a separate
viewer workflow from a standard ImageJ hyperstack.

## Common problems

| Symptom | Check |
| --- | --- |
| Unexpected fusion source | Read the selected path printed by `fuse`; pass the store explicitly. |
| Disconnected registration groups | Check overlap signal, channel selection, and correction limits. |
| Visible depth seams | Review registered placement, depth gains, and geometric support. |
| Resume rejects the run | Use the original settings and matching journal, or explicitly restart. |
| ROI cannot open | Run `fuse` to create the registered maximum-Z image first. |
| CUDA registration unavailable | Check GPU installation; `--require-gpu` reports failure instead of falling back. |
| Movie encoding fails | Check NVENC support and the installed NVIDIA driver. |
