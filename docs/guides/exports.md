# Exporting images and movies

## Fused OME-BigTIFF

```bash
uv run export-ome-tiff "/path/to/processed"
```

This reads level zero of `<stem>_fused.ome.zarr` and
`<stem>_max_z_fused.ome.zarr`, writing corresponding `.ome.tif` files beside
them. All timepoints, channels, pixel values, and the source dtype are retained.
Both exports use BigTIFF, 512 x 512 tiles, zlib compression level 8, and
`predictor=True`. Tiles are streamed without loading the full volume.
Progress reports timepoints, channels, and Z planes per channel separately.

OME-XML records dimensions, voxel spacing, registered positions, and available
channel names, colors, wavelengths, exposures, timing, acquisition date, and
instrument metadata. Acquisition settings, source XML, NGFF metadata, and the
processing journal are embedded as annotations when available. Keep the source
acquisition and journal accessible to retain their metadata.

`--metadata-only` repairs the XML encoding of existing exports without rewriting
or recompressing pixels. See [export-ome-tiff](../reference/cli/export-ome-tiff.md).

## Annotated projection TIFFs

```bash
uv run export-projections "/path/to/acquisition-root" --depth-color
```

Supply the directory containing one full deconvolved, deskewed store. Each
timepoint, channel, and position produces a full-resolution 8-bit display TIFF:
XY at upper left, XZ below, and YZ to the right. Panels share a physical display
scale and include labels, a scale bar, and an acquisition timestamp.

Contrast remains fixed to timepoint 0 for each position/channel. Its 0.001st and
99.999th intensity percentiles define the limits, with a min/max fallback when
they coincide. Timestamps use acquired scan-plane count times summed channel
exposures, excluding moves and other acquisition overhead. The raw acquisition
is located through the journal or supplied with `--acquisition`.

Depth coloring uses the brightest voxel's depth: Z for XY, Y for XZ, and X for
YZ. Exact ties select the first voxel. The `turbo` color map is the default;
`--depth-colormap` changes it. Depth legends stay fixed across time.

TIFFs are losslessly compressed display renderings, including their annotations.
Use the OME-Zarr or fused OME-BigTIFF for numerical image analysis. Output is
under `projection_frames/<dataset>/p000/c000/`, with a `depth_color` subdirectory
for colored images. Re-exporting replaces matching TIFFs.

## Movies

```bash
uv run export-projections "/path/to/acquisition-root" --timepoints 100 200 --video
```

The zero-based stop is exclusive. This exports and encodes timepoints 100 through
199 using their original acquisition timestamps. Movie encoding follows TIFF
export; existing TIFFs outside the selected range remain on disk.

Encode an existing dataset's TIFF sequences instead:

```bash
uv run encode-projections "/path/to/acquisition-root" --bitrate 8000000
```

Movies use NVIDIA NVENC, defaulting to H.264 and 8 Mbps constant bitrate.
Playback defaults to the recorded acquired volume rate, including fractional
rates. `--fps` overrides playback without changing burned-in timestamps.
Each TIFF supplies one frame in timepoint order. Existing-frame encoding rejects
missing or duplicate indices and skips legacy `downsample_*` directories.

Hardware encoding requires an NVENC-capable NVIDIA GPU and compatible driver.
MP4s are lossy presentation copies. See
[export-projections](../reference/cli/export-projections.md) and
[encode-projections](../reference/cli/encode-projections.md) for options.

## Raw timelapses

[convert-timelapse](../reference/cli/convert-timelapse.md) exports the time series
at selected raw positions and scan planes. Raw timelapse TIFFs retain camera
counts; optional time-mean projections apply camera calibration. This command
does not export a registered fused volume.
