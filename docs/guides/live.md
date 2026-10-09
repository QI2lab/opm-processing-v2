# Processing during acquisition

Live processing accepts the acquisition's containing directory and a
precomputed illumination OME-TIFF with CYX axes:

```bash
uv run process "/path/to/acquisition-directory" --live "/path/to/illumination.ome.tif" --deconvolve
```

## Required acquisition files

The acquisition controller writes one `*.manifest.json`, the OME-Zarr hierarchy,
and the corresponding `*.log.jsonl` lifecycle log in the directory. The manifest
identifies the store, acquisition identity, expected dimensions, and calibration.
Processing waits for the manifest and required metadata, then checks raw chunk
keys for complete timepoint/position tiles.

Live processing supports mirror and stage scans with the supported unsharded
Zarr v3 chunk layout. The supplied illumination is used directly; it is never
estimated during the run. `--live` cannot be combined with
`--flatfield-correction`, time ranges, or position ranges.

## Completion and interruption

Complete tiles are processed and checkpointed as they become available. A
completed lifecycle event ends processing once all expected tiles are complete.
Canceled or errored acquisitions terminate with the available completion count.
Existing compatible live checkpoints are reused on restart.

Register the finished output with [fuse](fusion.md). See the
[command reference](../reference/cli/process.md) for processing options and the
[data reference](../reference/data.md) for checkpoints.
