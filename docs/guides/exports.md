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

## Reconstructed ROI exports

```bash
uv run export-rois "/path/to/processed" --registration-channel 1 --deconvolve --tiff
```

Draw multiple XY rectangles on an existing deskewed maximum-Z preview in
napari. Closing the viewer reconstructs their raw regions, retaining full Z,
all channels, and all timepoints. Multi-position ROIs are locally registered
and fused. `--deconvolve` enables RLGC; `--tiff` adds compressed OME-BigTIFF
alongside the final OME-Zarr. Outputs go to numbered subdirectories under
`roi_exports` and retain global coordinates.
Follow the [ROI export guide](roi.md)
or see the [command reference](../reference/cli/export-rois.md).

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

## AO publication pupil grids

```bash
uv run python scripts/export_ao_overview.py "/path/to/acquisition"
```

The exporter reads saved `grid_pos_*/ao_results.zarr` stores directly and writes
one PNG and SVG per depth in `ao_publication`. The figures follow
`sjs_opm_scripts/analysis/ao_grid_phase_plots.py`: two columns, a 12 by 5-inch
canvas per row, 0.2 subplot spacing, normalized circular pupils, linear plus/minus
2-pi colors, and per-panel colorbars with 30-point text and pi ticks.

Saved coefficients are read through the portable local
`scripts/ao_saved_results.py`; an external SJS checkout is optional.
Pupil reconstruction uses the local WaveKit modal convention and RMS
normalization, checked against analytical circular modes. The default state is
final phase at 488 nm.
`--coefficient-state`, `--wavelength-nm`, `--timepoint`, and `--depth-index`
select the saved data. `--ao-reader` and `--plot-reference` accept explicit reader
and formatting reference paths, including the original SJS files.
`--output` selects the destination; `--overwrite` replaces generated
outputs there.

Use `--coefficients` to additionally save starting, final and signed-change
coefficient bars with the local reader. An explicit external reader uses its own
plotting and save method. `ao_publication.json` records
source/reference provenance and the original starting and final coefficients.

To also place the pupils over the fused tissue projection:

```bash
uv run python scripts/export_ao_overview.py "/path/to/acquisition" --tissue-overlay
```

Each Z level produces its own tissue PNG and SVG with SJS pupil colors and a
small, unlabeled colorbar beside each correction. Region labels are omitted.
Titles and all other text are also omitted. Each image includes an unlabeled
spatial scale bar calibrated from the projection's NGFF
transforms. `--scale-bar-um` sets its length; otherwise
the exporter selects a suitable length automatically. `--channel` selects the
tissue channel and `--projection` supplies an explicit fused maximum-Z store.
The pupil symbols default to a 112.5 µm display diameter, set by
`--pupil-diameter-um`.

Use `--planned-grid` to divide the full calibrated acquisition area into the
requested number of equal AO cells, placing each saved correction in the middle
of its intended cell. The grid dimensions come from the recorded AO configuration;
the original correction order is preserved. This illustrates intended coverage,
independent of the recorded sampling positions and acquired-tile assignments.
Grid lines use alpha 0.3 so underlying tissue remains visible.

Add `--full-resolution` to read the native projection level and export a PNG at
the source image's pixel dimensions, with one source sample per output pixel.
The SVG retains the native tissue image. Without this flag, the exporter uses
a pyramid preview and a figure-sized PNG. Grid placement, pupil size, and the
scale bar use the calibration of the selected image level.

Use `--local-z-projection` for a separate tissue background at each AO depth.
This reads the native fused 3D volume and computes each maximum inside that
layer's acquisition-defined camera ROIs. Detector ROI height, pixel size, OPM
angle, and recorded stage Z define each slab; its registration transform maps
the slab into the existing fused image. The projection follows each tile's
local XYZ bounds and preserves the common full-resolution XY canvas.

Local projection implies `--full-resolution`. The manifest records the camera
geometry, mapped voxel ROIs, projection hashes, and per-layer display limits.
Overlapping acquired slabs retain the signal already blended in the fused
volume. Saved XY ROI selections with `all_z` do not define separate depth slabs.

Overlay pupils identify representative centers of source-tile groups inferred
from the saved historical AO assignment plans. They do not mark calibrated AO
measurement sites or outline correction boundaries. This tissue composition
extends the standalone SJS pupil figure. Standalone grids retain every saved
correction. The tissue PNG/SVG and placement/scale-bar provenance are saved
alongside them.

## Repair legacy stage coordinates without rewriting pixels

For the uniform, positive, uncropped stage-scan geometry supported by the repair
validator, prepare a metadata-only plan:

```bash
uv run python scripts/repair_stage_coordinates.py "/path/to/acquisition" --plan repair-plan.json
uv run python scripts/repair_stage_coordinates.py --apply repair-plan.json
```

For data split across acquisition and processing/output directories, name both
roots explicitly:

```bash
uv run python scripts/repair_stage_coordinates.py "/path/to/acquisition" --additional-root "/path/to/processing" --plan repair-plan.json
```

The split-root plan follows the journal's existing artifact links and labels
every metadata change and dependency with its root. Unreferenced older fused
stores are reported separately and left untouched. Before any replacement, it
creates verified metadata backups and manifests on every volume being changed.
An interrupted repair restores already replaced metadata across those roots.

The plan lists every JSON or OME-XML change and its original and replacement
hashes. Applying it backs up the original metadata before changing local image
translations, OME stage labels, registered tile origins, and fused transforms.
Raw acquisition files, array metadata, image chunks, and TIFF exports are outside
the write whitelist. Changed inputs invalidate the plan; an applied plan is
idempotent. Existing TIFF and ROI exports remain historical and are reported
separately. Create new ROI exports after repair.
