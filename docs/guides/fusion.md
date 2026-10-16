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

## Register neighboring tiles

Choose a channel with shared tissue signal in tile overlaps. Stage positions,
tile shapes, spacing, and scan geometry determine initial placement and
automatic search limits. Registration downsampling adapts to each overlap.
The similarity threshold defaults to 0.7.

Depth-face registration retains extra Z context beyond the nominal overlap.
After measuring the lateral displacement, it searches the contextual samples
for Z alignment using correlation on common geometric support, then checks
the match with SSIM. Automatic depth corrections permit up to twenty percent
of the tile depth and require at least three actually matched planes, rather
than being capped by the nominal overlap width.
Explicit correction limits still apply; measured peaks beyond them are
rejected, with tile indices and corrections reported in the warning. A weak
lateral match in the nominal overlap is checked again using the wider context.
Changes to this method invalidate saved pairwise registration so
the next fusion run recomputes it.

Override the automatic correction limits when the acquisition needs a specific
search envelope, using processed-image pixels in ZYX order:

```bash
uv run fuse "/path/to/processed" --registration-channel 2 --max-registration-shift-zyx 20 250 100
```

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
