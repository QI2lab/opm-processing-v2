# encode-projections

Encode existing projection TIFF sequences as MP4s. Accepts the acquisition/output
directory used by `export-projections`; requires NVIDIA NVENC.

| Option | Description |
| --- | --- |
| `--fps VALUE` | Override playback rate; default saved acquisition rate. |
| `--bitrate VALUE` | Set video bits per second; default 8000000. |
| `--gpu VALUE` | Select the NVENC GPU; default 0. |
| `--codec NAME` | Select `h264` (default) or `hevc`. |
| `--rate-control NAME` | Select `cbr` (default) or `vbr`. |

```bash
uv run encode-projections "/path/to/acquisition" --bitrate 8000000
```

Append `--help` to list all options and defaults.
