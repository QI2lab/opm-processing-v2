# opm-processing-v2

Post-processing for current qi2lab opm-v2 OME-Zarr acquisitions.

## Install

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

On Windows and Linux, this includes the default `gpu` dependency group, which
selects the CUDA extra for deconvolution and registration. Plain `uv sync` and
`uv run` keep CuPy and its NVIDIA runtime dependencies installed; you do not
need to repeat `--extra gpu`. The explicit extra remains supported.

To deliberately omit the default GPU group (for a non-CUDA environment):

```bash
uv sync --no-group gpu
```

Native Windows also requires the pure-Python `cucim.skimage` package from a
local cuCIM checkout because RAPIDS does not publish its standard wheel there.

### cuCIM on Windows

Install cuCIM from the tagged source checkout using an editable installation:

1. In the Start menu, right-click **Miniforge Prompt** and select
   **Run as Administrator** so Git can create symbolic links.
2. Enable symbolic links globally for Git:

   ```bat
   git config --global --add core.symlinks true
   ```

3. Change to this project's directory and activate the environment created by
   `uv sync`:

   ```bat
   cd /d C:\Users\qi2lab\Documents\github\opm-processing-v2
   .venv\Scripts\activate.bat
   ```

4. Install cuCIM:

   ```bat
   pip install -e "git+https://github.com/rapidsai/cucim.git@v26.08.00#egg=cucim-cu12&subdirectory=python/cucim"
   ```

## Commands

Inspect an acquisition:

```bash
uv run inspect-opm "/path/to/acquisition"
```

Process it (uint16 output by default):

```bash
uv run process "/path/to/acquisition"
uv run process "/path/to/acquisition" --deconvolve --flatfield-correction
uv run process "/path/to/acquisition" --save-float32
```

Use `--skip-empty-below VALUE` to zero empty channel tiles before illumination
correction, deconvolution, and deskew. Use `--resume` to continue from completed
tiles; without it, the selected output is overwritten.

Process during acquisition by supplying a precomputed `CYX` illumination TIFF.
The acquisition argument must be its containing directory:

```bash
uv run process "/path/to/acquisition-directory" --live "/path/to/illumination.ome.tif"
```

Register, fuse, and create the registered multiscale max-Z image:

```bash
uv run fuse "/path/to/acquisition-or-output-directory"
```

Registration search limits are inferred separately for each overlapping tile
pair from stage spacing, processed tile dimensions, voxel spacing, scan angle,
and registration downsampling. Depth overlaps allow the oblique-plane footprint
displacement along Y; corrections must retain at least half the pair's overlap.
Downsampling and the SSIM window adapt to the available overlap. Links still
need to pass `--registration-threshold` (default 0.7). Use
`--max-registration-shift-zyx Z Y X` to override the automatic limits, and
`--registration-channel` to select the channel with suitable tissue signal.
Global optimization weights each measured axis by its effective registration
sampling and evaluates residuals in those sampling units. It rejects inconsistent
cycle edges one at a time, refitting after each rejection, while retaining the
links that anchor connected tile groups. Connectivity warnings use the links
retained by optimization.
Depth-layer brightness is normalized from registered overlaps at the same
stage XY. Each depth shares one gain per timepoint and channel across all its
XY tiles; brightness differences between XY fields are retained. The first
depth anchors the intensity scale. Gains are recorded in
`<stem>_depth_intensity_gains.json`. Use `--no-normalize-depth-intensity` to
preserve the deskewed tiles' intensity scales. Raw and deskewed arrays are
never modified by fusion normalization.

Enable or disable depth normalization explicitly when running fusion:

```bash
uv run fuse "/path/to/acquisition" --normalize-depth-intensity
uv run fuse "/path/to/acquisition" --no-normalize-depth-intensity
```

Draw and save a rectangular ROI from that registered max-Z image, then process
and fuse the selected raw-data region:

```bash
uv run display "/path/to/acquisition"
uv run process-ROI "/path/to/acquisition"
```

`process-ROI` resumes by default. Rerun the same command after an interruption:
completed tiles and channels are skipped, and any channel whose write was not
checkpointed is processed again. Existing runs with tile-only checkpoints retain
their completed tiles and repeat the unfinished tile. Resume requires the same
processing settings and ROI tile mapping; use `--no-resume` to overwrite the ROI
output and start again.

Both commands default to `<acquisition-stem>_roi.json`. Processing state is kept
in one `<acquisition-stem>.processing.json` beside the outputs; image stores
contain only OME/NGFF image metadata.

When processed outputs are stored separately from the raw acquisition, pass
their directory to `process-ROI`. It reads the raw source path from the single
`<acquisition-stem>.processing.json` there, loads the ROI JSON from that directory,
and writes to its `<acquisition-stem>_roi` subdirectory. The raw acquisition must
still be accessible. You can also specify the raw data, ROI JSON, and destination
explicitly:

```bash
uv run process-ROI "/path/to/processed-outputs"
uv run process-ROI "/path/to/raw.ome.zarr" "/path/to/selection_roi.json" --output "/path/to/roi-output"
```

See every option and default with:

```bash
uv run process --help
uv run fuse --help
uv run display --help
uv run process-ROI --help
```

## Export timepoint projection images

Both commands take the acquisition root directory only. They automatically select
the single full `*_decon_deskewed.ome.zarr` store immediately inside that root,
excluding max-projection stores. Missing or ambiguous stores produce an error.
Export one annotated TIFF per timepoint, channel, and position:

```bash
uv run export-projections "/path/to/acquisition-root"
uv run export-projections "/path/to/acquisition-root" --scale-bar-um 5
```

The canvas places XY at upper left, XZ below it, and YZ to its right (Y vertical).
All projections share one physical display scale derived from the deskewed voxel
sizes. The empty lower-right corner contains a scale bar and a `MM:SS:mmm`
timestamp starting at zero, with `min:s:ms` units immediately below it. Timing uses the **original acquired** scan-plane
count times the sum of channel exposures, excluding overhead and position moves.
For equal exposures this is planes Ã— exposure Ã— channel count. The acquisition
is located through the processing sidecar; use `--acquisition /path/to/raw.ome.zarr`
if that link is unavailable.

Each position/channel uses the first volume's 0.001st and 99.999th intensity percentiles
for the entire sequence. Coincident percentiles fall back to that volume's
minimum and maximum. TIFFs are rendered 8-bit grayscale display images, with
timing, contrast limits, and physical scale also saved in their metadata.
Outputs default to
`projection_frames/<dataset>/p000/c000/<dataset>_t0000.tiff` beside the input store;
`--output` changes the output root. Re-running replaces matching TIFFs.

Render only a range of timepoints with zero-based indices and an exclusive stop:

```powershell
uv run export-projections "/path/to/acquisition-root" --timepoints 100 200 --video
```

This writes frames 100?199 with their original filenames and acquisition timestamps.
Contrast remains fixed to dataset timepoint 0, which is read even when outside the
selected range. With `--video`, only these frames are encoded, into a movie named
`<dataset>_t0100-t0199.mp4`; existing TIFFs outside the range are left alone.
Omit `--timepoints` to export all frames. Downsampling options are not supported.

### Depth-colored projections

Full-resolution TIFFs use lossless Deflate compression. XY, XZ, and YZ labels
are rendered in the upper-left corner of their respective panels; depth legend
tick labels are rounded to whole micrometers. These annotations are stored in
the TIFF pixels before movie encoding.

Export uses two concurrent timepoint workers by default, without Dask. Set
`--workers 4` to read, render, and write up to four timepoints concurrently, or
`--workers 1` for serial export. Each worker holds a volume plus rendering
buffers, so memory use grows with the worker count; disk bandwidth can limit
the benefit of additional workers. First-frame contrast limits remain fixed,
and movie frames are ordered by timepoint regardless of completion order.
Movie encoding starts after the TIFF sequence finishes.

```powershell
uv run export-projections "/path/to/acquisition-root" --depth-color --video
```

`--depth-color` colors the depth of the brightest voxel along each projection
ray: Z for XY, Y for XZ, and X for YZ. Brightness uses the same fixed first-frame
contrast limits as grayscale export. This independent Python implementation
follows the slice LUT and intensity modulation concept in
[ZstackDepthColorCode](https://github.com/UU-cellbiology/ZstackDepthColorCode).
It selects the intensity maximum before coloring; it does not combine RGB
maxima from different depths. Exact ties select the first voxel.

Each canvas includes three labeled color lookup bars in micrometers, spanning
local voxel centers from zero to `(axis length - 1) * voxel spacing`. These
ranges stay fixed across time. `--depth-colormap turbo` selects the default LUT;
other names supported by `cmap` may be used. The bars show full-brightness colors;
dim signal has correspondingly darker colors. Physical display resampling can blend colors from neighboring pixels.

RGB TIFFs and movies are saved under `p000/c000/depth_color/`. TIFF metadata
records the depth ranges and LUT. Full-resolution TIFFs contain all legends and
annotations; movie encoding uses those stored pixels.
Omitting `--depth-color` keeps grayscale export.

### Encode projection movies

Create TIFFs and one MP4 per channel/position together:

```bash
uv run export-projections "/path/to/acquisition-root" --video
```

Or encode the selected dataset's existing frames in
`<root>/projection_frames/<dataset>/`, searching its channel/position subdirectories:

```bash
uv run encode-projections "/path/to/acquisition-root" --bitrate 8000000
```

TIFF export writes only full-resolution frames. Timestamps, time units, scale
bars, panel labels, and depth legends are rendered into the TIFFs with 20-pixel
text. TIFFs use lossless Deflate compression.

The encoder reads those TIFFs directly and writes `<dataset>.mp4` at 8 Mbps by
default. There is no downsampled export or companion movie. Existing legacy
`downsample_*` frame folders are skipped. Playback defaults
to the acquisition volume rate stored in the TIFF metadata (1000 divided by
`volume_interval_ms`), retaining fractional rates for real-time playback.
`--fps` explicitly overrides that rate; burned-in acquisition timestamps stay unchanged. Each
TIFF becomes exactly one video frame, sorted by numeric timepoint index. Missing
or duplicate indices are rejected by the existing-frame command.

Encoding follows NVIDIA's
[PyNvVideoCodec workflow](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/using_pynvvideocodec_apis.html#video-encoding),
using hardware H.264, P7/high-quality tuning, 8 Mbps constant bitrate, and 8-bit 4:2:0
video in a fast-start MP4 (`avc1`). FFmpeg only muxes the encoded packets; it does
not re-encode them. Grayscale is explicitly converted to limited-range luma
with neutral chroma and BT.709 signaling. Dimensions are padded on the right
and bottom to even sizes (at least 128 pixels), without stretching or cropping
the projections or scale bar.

`--bitrate` sets the target bitrate in bits per second (default 8000000), and
`--gpu` selects the NVIDIA GPU. The existing-frame encoder also accepts
`--codec h264|hevc` and `--rate-control cbr|vbr` for compression comparisons.
HEVC has narrower playback compatibility; VBR does not guarantee smaller files
at the same average bitrate. An NVENC-capable GPU and compatible NVIDIA driver
are required. PyNvVideoCodec is installed by `uv sync` on Windows/Linux.
MP4s are lossy presentation copies; keep TIFFs and OME-Zarr data for analysis.
Re-running replaces matching MP4s only after encoding and muxing succeed.

## Simulation and figure scripts

Non-core simulations, experiments, and figure generators live in `scripts/`.
Run them as modules from the repository root so their package imports resolve:

```bash
uv run python -m scripts.opm_simulation --output diagnostics/meridian_sphere
uv run --with matplotlib python -m scripts.plot_opm_simulation diagnostics/meridian_sphere/scan_0.2um
```

Each script's opening docstring describes its inputs, calculations, and outputs.
The [sphere simulation workflow](docs/meridian_sphere_simulation.md) and
[publication figure workflow](docs/opm_publication_figure.md) show the full commands.
These repository scripts are separate from the installed processing commands.

## Tests

Tests are either unit tests of isolated numerical or state behavior, or
integration tests using simulated objects with real disk inputs and verified
disk outputs. Smoke tests, API/argument-forwarding tests, and benchmark-only
tests are excluded. GPU is a hardware requirement marker, not a test category.

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

Require rather than skip CUDA tests with:

```bash
OPM_REQUIRE_GPU=1 uv run --extra gpu --group dev pytest -m gpu
```
