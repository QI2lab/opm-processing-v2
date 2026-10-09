# process

Calibrate and deskew an acquisition, optionally deconvolving it. Saves uint16
output beside the acquisition by default.

| Option | Description |
| --- | --- |
| `--deconvolve` | Apply RLGC deconvolution before deskewing. |
| `--flatfield-correction` | Correct uneven illumination. |
| `--save-float32` | Preserve fractional intensities in float32 output. |
| `--skip-empty-below VALUE` | Zero channel tiles with insufficient signal above this intensity. |
| `--skip-empty-min-signal-fraction VALUE` | Required fraction above the cutoff; default 0.01. |
| `--resume` | Continue compatible completed tiles; otherwise overwrite output. |
| `--output PATH` | Set the output directory. |
| `--live PATH` | Process during acquisition using a supplied CYX illumination TIFF; pass the acquisition directory. |
| `--time-range START STOP` | Select timepoints. |
| `--pos-range START STOP` | Select positions. |
| `--z-downsample-level VALUE` | Deskewed Z reduction factor; default 2. |
| `--crop-after-deskew` | Crop Y to the region supported across all Z planes. |
| `--no-max-projection` | Disable per-tile and stage-placed maximum-Z images. |
| `--no-create-fused-max-projection` | Disable the stage-placed maximum-Z mosaic. |
| `--write-fused-max-projection-tiff` | Also save that mosaic as OME-TIFF. |
| `--decon-crop-scan VALUE` | Set scan planes per deconvolution chunk; default automatic. |
| `--decon-gpu-id VALUE` | Select the CUDA device; default 0. |
| `--decon-verbose VALUE` | Set solver verbosity; 0 suppresses iteration reports. |
| `--decon-fallback-step-scan VALUE` | Set chunk reduction after GPU allocation failure. |
| `--decon-psf-paths PATH` | Supply channel-ordered PSFs; repeat for each channel. |
| `--decon-scan-upsample VALUE` | Experimental 3D scan upsampling; requires deconvolution and full-volume GPU memory. |
| `--eager-mode` | Enable eager stopping for planar deconvolution. |

```bash
uv run process "/path/to/acquisition" --deconvolve --flatfield-correction
```

Append `--help` to list all options and defaults.
