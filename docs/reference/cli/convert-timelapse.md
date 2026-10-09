# convert-timelapse

Export raw timelapses at selected positions and scan planes as OME-TIFF, RAW/YAML,
or calibrated time-mean TIFFs. Accepts an acquisition store or its containing
directory. Output defaults to `converted_files` beside the store.

| Option | Description |
| --- | --- |
| `--output-dir PATH` | Set the destination directory. |
| `--time-range START STOP` | Select timepoints. |
| `--stage-range START STOP` | Select positions. |
| `--scan-range START STOP` | Select scan planes. |
| `--fov-x-range START STOP` | Select camera columns. |
| `--create-raw` | Write RAW data with YAML shape sidecars. |
| `--create-time-projection` | Write calibrated means over time per channel. |
| `--no-create-tiff` | Disable raw timelapse OME-TIFF output. |
| `--camera-offset VALUE` | Override electronic background in ADU. |
| `--camera-conversion VALUE` | Override calibrated intensity per ADU. |

```bash
uv run convert-timelapse "/path/to/acquisition" --stage-range 0 1 --scan-range 0 1 --create-time-projection
```

Append `--help` to list all options and defaults.
