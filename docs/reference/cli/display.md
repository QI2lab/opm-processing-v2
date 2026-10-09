# display

Open processed data in napari and save a rectangular ROI. Defaults to the
registered fused maximum-Z image and `<stem>_roi.json` beside the data.

| Option | Description |
| --- | --- |
| `--to-display VALUE` | Select `full`, `max-z`, `fused-full`, or `fused-max-z`. |
| `--no-roi` | View without creating an ROI; required for other display modes. |
| `--roi-output PATH` | Set the ROI JSON destination. |
| `--time-range START STOP` | Select visible timepoints. |
| `--pos-range START STOP` | Select visible positions. |

```bash
uv run display "/path/to/acquisition" --roi-output "/path/to/selection_roi.json"
```

Append `--help` to list all options and defaults.
