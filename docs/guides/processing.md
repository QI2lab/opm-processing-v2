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
For each channel, the correction is the pixelwise maximum of the estimated
stage-Z fields and is shared across all depths and timepoints. This prevents
sample-free camera regions at one depth from imposing a separate correction
on that depth. Depths excluded by signal selection do not contribute; channels
without selected signal use a unity field. Saved flatfields contain one field
per channel. Cached fields from the older per-depth estimator are re-estimated.

To launch processing before acquisition finishes without a precomputed
illumination file, add `--live-wait` alongside `--flatfield-correction` and
pass the acquisition directory. The command waits for completion before
estimating illumination or processing images. See the
[live acquisition guide](live.md#wait-for-completion-before-processing).

`--deconvolve` uses GPU RLGC; channel-ordered PSFs can be supplied with repeated
`--decon-psf-paths` options. Otherwise theoretical PSFs are generated.
With deconvolution, camera-calibrated detector estimates retain their precision
during splitting, and flatfield and detector-gain responses enter the solver's
measurement model. Without deconvolution, correction still divides the
calibrated floating-point image by these responses before deskewing.

The default saved pixels are clipped uint16 values. `--save-float32` retains
fractional calibrated intensities. `--z-downsample-level` controls the integer
reduction of deskewed Z, defaulting to 2. Calibration and interpolation
conventions are described in [methods](../methods/deskew.md).

## Physical time axis

Processing reads the actual hardware camera exposure from per-frame
`exposure_time` (seconds), or legacy `exposure_ms` (milliseconds), and rounds
each exposure to two decimal places in milliseconds. For this acquisition
system, the rounded exposure defines the hardware frame period. The volume
interval is the original acquired scan-plane count multiplied by the sum of
these periods across stored camera channels. Simultaneous wavelengths stored
in one image count as one camera channel. Discarded acquired planes are included;
deskewed Z size does not affect timing. Software timestamps (`delta_t`,
`runner_time_ms`, and `ElapsedTime-ms`) and file-writing delays are ignored.
The resulting OME-NGFF T scale has unit `second`.
The same scale is preserved in projections, mosaics, and every pyramid level.
Position collections also record the OME-XML time increment in seconds.

For legacy acquisitions without recorded hardware exposures, supply the confirmed
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
settings. Durably completed tiles are reused. If you omit `--resume` and a
partial output is found, `process` offers **Resume / Overwrite / Cancel** before
starting processing or changing the outputs. Enter `resume` (or `r`) to keep
completed tiles, `overwrite` (or `o`) to restart the selected output run, or
`cancel` (or `c`) to stop. Cancel is the default: pressing Enter or closing the
input stream stops the command. Choices are case-insensitive. Pass `--resume`
explicitly for unattended restarts.

The perpendicular-distance orthogonal interpolation update changes the saved
pixels. Runs produced with the earlier equal-plane weights must be regenerated;
their interpolation settings are incompatible with `--resume`. Use the existing
overwrite choice to restart a partial run with the new interpolation.

The check uses the previous run's selected timepoints and positions, its
completion journal, and the presence of position metadata. It also prompts
when an existing output has no readable completion journal. Confirming resume
still requires compatible settings and readable image metadata; it does not
repair damaged output. Completed output runs retain the existing overwrite
behavior when `--resume` is omitted. Keep the journal with its image stores;
see the [data reference](../reference/data.md).

## Produce a registered mosaic

The maximum-Z mosaic created during processing uses nominal stage placement.
Run [fuse](fusion.md) to register tiles and create the registered fused volume
and projection. When output is stored separately, pass that output directory
to subsequent commands.

The [process reference](../reference/cli/process.md) lists options. Experimental
scan upsampling and its limitations are described in the
[deconvolution methods](../methods/deconvolution.md#sparse-scans).
