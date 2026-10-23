# inspect-opm

Report available raw and processed OME-Zarr metadata as JSON, without reading
image pixels. Accepts a raw store, a processed store, or an acquisition directory.

```bash
uv run inspect-opm "/path/to/acquisition"
uv run inspect-opm "/path/to/acquisition/sample_deskewed.ome.zarr"
```

The report contains `raw` and `processed` sections. Acquisition directories and
raw paths include associated processing products when present. Processing
journals also identify products with custom names and connect a processed store
to its raw source. A processed store remains inspectable when the raw source is
unavailable; unknown calibration or timing is reported as `null`.

Raw dimensions use `s` for **scan positions**, `y'` for distance along the tilted
camera plane, and `x` for camera columns. `sampling_um` records scan spacing and
the calibrated camera pixel pitch. `camera_field_of_view_um` gives the physical
camera-plane dimensions. `y_prime_sampling_projected_y_um` is the camera pitch
multiplied by cos(tilt angle). No raw laboratory-Z spacing is reported.

`timing.frame_exposure_ms` lists actual camera exposures in stored channel order,
rounded to two decimal places in milliseconds. Hardware-triggered volume time is
`sum(frame_exposure_ms) × acquired scan positions`, including positions later
discarded during processing. Software arrival timestamps, requested exposures,
and processed Z depth do not determine volume time. Combined laser channels
stored in one camera image count as one exposure.

Processed images use laboratory `z`, `y`, and `x`. Each position/series reports
its level-zero dimensions, voxel spacing, origin, channels and pyramid levels.
`volume_dimensions_um` gives **Z (µm) × Y (µm) × X (µm)** using voxel count times
recorded spacing, including any stored padding. These are the dimensions of one
volume, independent of its channel and timepoint counts. Dataset and shared
multiscale transforms are composed before reporting physical dimensions.

Append `--help` to list all options and defaults.
