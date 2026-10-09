# fuse

Register and blend processed tiles, saving fused OME-Zarr and maximum-Z images.
Accepts an acquisition/output directory or processed store; directory selection
prefers deconvolved, deskewed data.

| Option | Description |
| --- | --- |
| `--registration-channel VALUE` | Select the zero-based registration channel; default 0. |
| `--max-registration-shift-zyx Z Y X` | Override correction limits inferred from stage positions and scan geometry. |
| `--registration-threshold VALUE` | Minimum overlap similarity; default 0.7. |
| `--blend-pixels Z Y X` | Set edge blending widths; default 20 600 400. |
| `--downsample-factors Z Y X` | Set registration sampling factors; default 3 5 5. |
| `--ssim-window VALUE` | Set the overlap similarity window; default 15. |
| `--no-normalize-depth-intensity` | Disable brightness matching across depths; XY brightness differences are retained. |
| `--roi PATH` | Restrict fusion to a saved ROI. |
| `--regenerate-max-z` | Recreate maximum-Z images from existing fused data. |
| `--require-gpu` | Require CUDA registration. |
| `--chunk-shape-yx Y X` | Set fusion block dimensions; default 1024 1024. |
| `--fusion-ram-fraction VALUE` | Set available RAM fraction for fusion; default 0.4. |
| `--max-workers VALUE` | Set CPU fusion workers; default up to eight physical cores. |
| `--max-in-flight-writes VALUE` | Limit queued block writes; default 2. |
| `--optimization-rel-threshold VALUE` | Set the registration outlier multiplier; default 0.5. |
| `--optimization-abs-threshold VALUE` | Set the minimum registration residual cutoff; default 1.5. |

```bash
uv run fuse "/path/to/acquisition" --registration-channel 2 --max-registration-shift-zyx 20 250 100
```

Append `--help` to list all options and defaults.
