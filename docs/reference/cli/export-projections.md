# export-projections

Export annotated XY/XZ/YZ projection TIFFs per timepoint, channel, and position.
Accepts the directory containing one full deskewed store; the deconvolved version
is preferred when both versions of that acquisition exist. Saves
8-bit display images under `projection_frames` beside the store. Timestamps require
the processed NGFF time scale or recorded hardware volume timing.

After moving or copying an acquisition directory, pass its current location.
The raw store beside the processed data takes precedence over the original path
in the processing journal. `--acquisition` overrides this selection; the recorded
path remains a fallback when raw data is stored separately.

| Option | Description |
| --- | --- |
| `--output PATH` | Set the output root. |
| `--acquisition PATH` | Supply raw acquisition metadata for timestamps. |
| `--scale-bar-um VALUE` | Set scale bar length in micrometers; default automatic. |
| `--timepoints START STOP` | Select timepoints; contrast stays fixed to timepoint 0. |
| `--depth-color` | Color brightest-voxel depth and add depth legends. |
| `--depth-colormap NAME` | Select the depth color map; default `turbo`. |
| `--workers VALUE` | Set concurrent timepoint workers; default 2. |
| `--video` | Also create an H.264 MP4 per channel/position; requires NVIDIA NVENC. |
| `--fps VALUE` | Override playback rate; default acquired volume rate. |
| `--bitrate VALUE` | Set video bits per second; default 8000000. |
| `--gpu VALUE` | Select the NVENC GPU; default 0. |

```bash
uv run export-projections "/path/to/acquisition" --timepoints 100 200 --depth-color --video
```

Append `--help` to list all options and defaults.
