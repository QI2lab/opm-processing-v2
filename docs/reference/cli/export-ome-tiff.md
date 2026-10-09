# export-ome-tiff

Export the highest-resolution fused volume and fused maximum-Z OME-Zarr to
OME-BigTIFF with channel and spatial metadata. Uses zlib level 8 and prediction.
Accepts the acquisition/output directory or full fused store; writes beside them.

| Option | Description |
| --- | --- |
| `--output PATH` | Set the destination directory. |
| `--workers VALUE` | Set compression workers; default 4. |
| `--overwrite` | Replace existing TIFF exports. |
| `--metadata-only` | Repair existing exports' OME XML encoding without rewriting pixels. |

```bash
uv run export-ome-tiff "/path/to/acquisition" --overwrite
```

Append `--help` to list all options and defaults.
