# process-ROI

Process the raw region selected by `display`, then fuse its tiles. Accepts a raw
acquisition or processed-output directory, followed by an optional ROI JSON path.
Defaults to `<stem>_roi.json`, deconvolution enabled, and `<stem>_roi` output.

| Option | Description |
| --- | --- |
| `--no-deconvolve` | Disable deconvolution. |
| `--no-resume` | Overwrite ROI output; completed tiles/channels are resumed by default. |
| `--output PATH` | Set the ROI output directory. |
| `--registration-channel VALUE` | Select the zero-based registration channel; default 0. |
| `--crop-after-deskew` | Unsupported for ROI processing, which already writes a physical crop. |

Also accepts the [process](process.md) options `--flatfield-correction`, `--save-float32`,
`--z-downsample-level`, `--decon-crop-scan`, `--decon-gpu-id`, `--decon-verbose`,
and `--decon-psf-paths`.

```bash
uv run process-ROI "/path/to/acquisition" "/path/to/selection_roi.json" --output "/path/to/roi-output"
```

Append `--help` to list all options and defaults.
