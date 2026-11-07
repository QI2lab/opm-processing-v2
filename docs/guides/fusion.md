# Registration and fusion

Run fusion on completed processed tiles:

```bash
uv run fuse "/path/to/processed" --registration-channel 2
```

## Select the input

Directory selection prefers `*_decon_deskewed.ome.zarr` when available and
prints the selected store. Supply the processed store itself to select it
explicitly. Fusion writes a registered multiscale volume and its maximum-Z
projection beside the processed data.

To render the fused volume and maximum-Z OME-Zarrs to another folder or drive:

```powershell
uv run fuse "G:\data\acquisition" --registration-channel 0 --output-dir "H:\fused\acquisition"
```

The destination is created if needed. The processing journal, cached pairwise
registration, final tile positions, and depth-gain report stay beside the
deskewed input. Choosing another destination reuses compatible registration.
The external image stores contain a link to that journal so projection
regeneration, ROI selection, and TIFF export can still find their provenance.
Keep the linked input/journal accessible. To regenerate just the maximum-Z
image, use the same input and `--output-dir` with `--regenerate-max-z`.

## Register neighboring tiles

Choose a channel with shared tissue signal in tile overlaps. Stage positions,
tile shapes, spacing, and scan geometry determine initial placement.
Automatic search limits read the acquisition's recorded `Positions` overlap
settings. Stage scans use `scan_axis_overlap_um` when available; the
fractional `scan_axis_overlap`, `tile_axis_overlap`, and `z_axis_overlap`
settings are converted using raw camera and scan geometry. The full recorded
overlap supplies the initial allowance, bounded by actual common support.
Missing settings use measured stage-geometry overlaps, without assuming an
overlap percentage. Registration downsampling adapts to each overlap.
The similarity threshold defaults to 0.7.

Empty overlaps and one-sided constant backgrounds are rejected before histogram
matching, which could otherwise turn a blank patch into a bright constant image.
An optional directional edge image with zero energy is skipped; a valid
intensity-based registration is retained.
Periodic equivalents of each measured phase peak are checked against physical
shift limits and scored on common image support before an excessive shift is
rejected. This includes thin depth overlaps: resolving the same peak's period
does not require selecting a different correlation peak. The period is the
actual sampled patch shape, including block-reduction padding. A small
equivalent shift still has to pass similarity and axial-feature checks.
If real-space disambiguation is degenerate, its unresolved periodic shift is
discarded. Registration can use an independently resolved intensity or edge
candidate; the pair is rejected when neither provides a valid measurement.
Other backend warnings remain visible.

Depth-face registration retains extra Z context beyond the nominal overlap.
After measuring the lateral displacement, it searches the contextual samples
for Z alignment using correlation of axial intensity differences on common
geometric support, then checks the original-intensity match with SSIM.
Shared XY background can produce high SSIM at many Z shifts; it does not
establish an axial registration. Initial automatic depth limits use the recorded
depth overlap and require at least three actually matched planes, rather
than being capped by the nominal overlap width.
Explicit correction limits still apply; measured peaks beyond them are
rejected. Shift-limit warnings count only candidates meeting the SSIM threshold;
low-similarity out-of-range candidates are reported separately and do not
suggest expanding the search. The warning's largest ZYX values are maxima
on each axis across the listed candidates and can come from different pairs.
A weak
lateral match in the nominal overlap is checked again using the wider context.
Changes to this method invalidate saved pairwise registration so
the next fusion run recomputes it.

If an out-of-range peak meets the similarity threshold, automatic registration
increases only that pair's offending axis limits to the absolute measured
correction plus a small margin: five percent, with a minimum of two pixels or
one effective registration sampling step. For example, a 15-pixel Z correction
measured at full Z sampling expands a 13-pixel limit to 17; a 284-pixel Y
correction at five-pixel sampling expands a 198-pixel limit to 299.
Unaffected pairs and axes keep their existing limits. Every retry **remeasures
only pairs whose limits increased**, then refits the global solution and
outlier rejection across all timepoints using the complete set of raw pairwise
measurements. Unchanged measurements are retained, including links previously
rejected by global optimization, so the updated graph can reconsider them.
Offsets are applied only after the final pass. A peak
measured on the old search boundary is provisional and is measured again in
the wider search. The default is two retries. SSIM thresholds, geometric
support checks, and global outlier rejection still apply. Low-similarity
rejected peaks alone do not trigger expansion. Disconnected groups without
out-of-range peaks do not trigger it.

Use `--registration-retries 0` to disable automatic retries, or choose another
nonnegative retry count. Final measurements and effective limits are cached;
changing the retry count or initial limits invalidates the cache. Existing
registration caches from before this retry policy are recomputed once.
Changing shared registration settings, such as the channel or downsampling,
starts a fresh full pairwise registration pass.

Override the automatic correction limits when the acquisition needs a specific
search envelope, using processed-image pixels in ZYX order:

```bash
uv run fuse "/path/to/processed" --registration-channel 2 --max-registration-shift-zyx 20 250 100
```

An explicit `--max-registration-shift-zyx` is a hard limit and is never expanded
automatically, regardless of the retry count.

Inspect connectivity warnings before accepting a mosaic. They describe links
retained after global optimization. Unconnected groups lack a measured relative
alignment. Review input choice, overlap signal, and registration limits when
a dataset expected to be connected produces disconnected groups.

## Match brightness across depths

Depth normalization is enabled by default. Registered overlaps at the same
stage XY supply one gain per depth, timepoint, and channel, shared across that
depth's XY tiles. Differences between XY fields are retained. Fitted gains are
saved in `<stem>_depth_intensity_gains.json`; source tiles remain unchanged.

Use `--no-normalize-depth-intensity` to omit these gains. The
[method page](../methods/fusion.md) explains gain fitting and geometric coverage.

## Recreate the registered projection

```bash
uv run fuse "/path/to/processed" --regenerate-max-z
```

This projects every resolution of the existing registered fused image without
registering and blending the tiles again. See the
[fuse reference](../reference/cli/fuse.md) for all options.
