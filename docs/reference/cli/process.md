# process

Calibrate and deskew an acquisition, optionally deconvolving it. Saves uint16
output beside the acquisition by default.

With `--deconvolve`, 3D scan refinement is inferred from recorded spacing toward
0.4 micrometers. For example, a 1.2 micrometer acquisition uses factor 3. Already
finer acquisitions retain their spacing; coarser steps must be integer multiples
of 0.4 micrometers.
Measured PSFs must be sampled at the inferred reconstruction spacing.

| Option | Description |
| --- | --- |
| `--deconvolve` | Apply RLGC deconvolution before deskewing. |
| `--flatfield-correction` | Include illumination in the deconvolution measurement model, or divide calibrated data by it when deconvolution is disabled. |
| `--save-float32` | Preserve fractional intensities in float32 output. |
| `--skip-empty-below VALUE` | Zero channel tiles with insufficient signal above this intensity. |
| `--skip-empty-min-signal-fraction VALUE` | Required fraction above the cutoff; default 0.01. |
| `--resume` | Continue compatible completed tiles. If omitted, partial outputs offer Resume / Overwrite / Cancel (default: Cancel); completed outputs are overwritten. |
| `--output PATH` | Set the output directory. |
| `--live PATH` | Process during acquisition using a supplied CYX illumination TIFF; pass the acquisition directory. |
| `--live-wait` | Wait for successful live completion and all raw tiles, then process offline. Combine with `--flatfield-correction` to estimate missing illumination. Mutually exclusive with `--live`; pass the acquisition directory. |
| `--time-range START STOP` | Select timepoints. |
| `--pos-range START STOP` | Select positions. |
| `--z-downsample-level VALUE` | Deskewed Z reduction factor; default 2. |
| `--crop-after-deskew` | Crop Y to the region supported across all Z planes. |
| `--no-max-projection` | Disable per-tile and stage-placed maximum-Z images. |
| `--no-create-fused-max-projection` | Disable the stage-placed maximum-Z mosaic. |
| `--write-fused-max-projection-tiff` | Also save that mosaic as OME-TIFF. |
| `--decon-crop-scan VALUE` | Set scan planes per deconvolution chunk; default automatic. |
| `--decon-gpu-id VALUE` | Select the CUDA device; default 0. |
| `--decon-verbose VALUE` | Opt-in solver diagnostics: default 0 (off); 1 reports chunk geometry and iterations. |
| `--decon-fallback-step-scan VALUE` | Set chunk reduction after GPU allocation failure. |
| `--decon-psf-paths PATH` | Supply channel-ordered PSFs; repeat for each channel. |
| `--eager-mode` | Legacy planar KL-stopping option; has no effect with MATLAB-default GC stopping. |

```bash
uv run process "/path/to/acquisition" --deconvolve --flatfield-correction
```

Start before acquisition finishes, without a precomputed illumination file:

```bash
uv run process "/path/to/acquisition-directory" --live-wait --flatfield-correction --deconvolve
```

See the [live acquisition guide](../../guides/live.md) for required controller
sidecars and completion behavior.

Append `--help` to list all options and defaults.

The photon-GC solver revision is recorded with the processing configuration.
Resume rejects incompatible solver configurations; start a fresh output or
intentionally overwrite instead.
Processing without deconvolution retains its existing calibration and flatfield
behavior.
