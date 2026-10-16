# Processing during acquisition

Live processing accepts the acquisition's containing directory and a
precomputed illumination OME-TIFF with CYX axes:

```bash
uv run process "/path/to/acquisition-directory" --live "/path/to/illumination.ome.tif" --deconvolve
```

## Wait for completion before processing

When no illumination file is available, use `--live-wait` with
`--flatfield-correction`:

```bash
uv run process "/path/to/acquisition-directory" --live-wait --flatfield-correction --deconvolve
```

The command waits for a successful `completed` lifecycle event and every
expected raw timepoint/position tile. No illumination estimation or image
processing starts while acquisition is running. After completion, it refreshes
the metadata and runs normal offline processing, estimating the shared
per-channel flatfield from the acquired data. An existing current flatfield
is reused through the usual lookup beside the outputs or source acquisition.

`--live-wait` takes no filename and cannot be combined with `--live PATH`.
Pass the acquisition's containing directory. Without `--flatfield-correction`,
it waits and then processes without illumination correction. Offline options
such as `--time-range`, `--pos-range`, `--output`, and `--resume` remain
available. A canceled or errored experiment stops the command before processing
begins. Restart the command after an interruption to wait again; use `--resume`
to continue compatible processing checkpoints once acquisition has completed.

## Required acquisition files

The acquisition controller writes one `*.manifest.json`, the OME-Zarr hierarchy,
and the corresponding `*.log.jsonl` lifecycle log in the directory. The manifest
identifies the store, acquisition identity, expected dimensions, and calibration.
Both live modes wait for the manifest and required metadata, then check raw
chunk keys for complete timepoint/position tiles. Polling occurs every 30
seconds. Single-position root images and multi-position collections are
recognized.

Live processing supports mirror and stage scans with the supported unsharded
Zarr v3 chunk layout. The supplied illumination is used directly; it is never
estimated during streaming. `--live PATH` cannot be combined with
`--flatfield-correction`, time ranges, or position ranges.

## Completion and interruption

With `--live PATH`, complete tiles are processed and checkpointed as they
become available. A completed lifecycle event ends processing once all
expected tiles are complete.
Canceled or errored acquisitions terminate with the available completion count.
Existing compatible live checkpoints are reused on restart.

Register the finished output with [fuse](fusion.md). See the
[command reference](../reference/cli/process.md) for processing options and the
[data reference](../reference/data.md) for checkpoints.
