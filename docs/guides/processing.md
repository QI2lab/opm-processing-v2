# Processing an acquisition

`process` reads recorded camera calibration, channel settings, stage positions,
and scan geometry from the raw acquisition. Mirror and stage scans are deskewed;
native planar acquisitions follow the planar processing path.

## Choose corrections and output

```bash
uv run process "/path/to/acquisition" --deconvolve --flatfield-correction --output "/path/to/processed"
```

Camera calibration runs before optional illumination correction and
deconvolution. `--flatfield-correction` estimates or reuses illumination fields.
`--deconvolve` uses GPU RLGC; channel-ordered PSFs can be supplied with repeated
`--decon-psf-paths` options. Otherwise theoretical PSFs are generated.

The default saved pixels are clipped uint16 values. `--save-float32` retains
fractional calibrated intensities. `--z-downsample-level` controls the integer
reduction of deskewed Z, defaulting to 2. Calibration and interpolation
conventions are described in [methods](../methods/deskew.md).

## Physical time axis

When the raw acquisition records `opm_v2.timing.n_acquired_planes` and
`opm_v2.timing.dt_plane_s`, processing writes their product as the OME-NGFF T
scale with unit `second`. The count includes acquired planes discarded during
processing; deskewed Z size and camera exposure do not determine the cadence.
The same scale is preserved in projections, mosaics, and every pyramid level.
Position collections also record the OME-XML time increment in seconds.

For legacy acquisitions without a hardware-period field, supply the confirmed
period explicitly:

```bash
uv run --locked process "/path/to/raw.ome.zarr" --plane-period-ms 5 --output "/path/to/processed"
```

For 17 acquired planes at 5.000 ms per plane, the output T scale is `0.085`
seconds per volume. The processed store records the acquired count, hardware
period, volume interval, and confirmation source under `opm_processing.timing`.
A supplied period must agree with any recorded hardware period. Unknown timing
remains a unitless index axis. Resume rejects a conflicting existing time scale.

## Select or skip data

`--time-range START STOP` and `--pos-range START STOP` select zero-based,
stop-exclusive ranges. With `--skip-empty-below VALUE`, a channel tile is zeroed
when too little calibrated signal reaches the cutoff. The minimum signal
fraction defaults to 0.01 and can be set with
`--skip-empty-min-signal-fraction`. This check precedes illumination correction.

## Resume a run

```bash
uv run process "/path/to/acquisition" --deconvolve --output "/path/to/processed" --resume
```

Resume uses the processing journal beside the outputs and requires compatible
settings. Durably completed tiles are reused. Without `--resume`, the selected
output run is overwritten. Keep the journal with its image stores; see the
[data reference](../reference/data.md).

## Produce a registered mosaic

The maximum-Z mosaic created during processing uses nominal stage placement.
Run [fuse](fusion.md) to register tiles and create the registered fused volume
and projection. When output is stored separately, pass that output directory
to subsequent commands.

The [process reference](../reference/cli/process.md) lists options. Experimental
scan upsampling and its limitations are described in the
[deconvolution methods](../methods/deconvolution.md#sub-sampled-acquisitions).
