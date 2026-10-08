# Fusion optimization

Fusion now uses fewer intermediate arrays while preserving the saved pixels,
feather profiles, geometric support masks, missing-channel handling, depth gains,
chunk layouts, worker counts, and bounded asynchronous writes.

## Full-volume rendering

The accumulation kernel traverses contiguous X rows within each channel. It
receives crop coordinates into the original source block rather than copying
each overlap into a separate contiguous array. Channels without signal and rows
outside the deskew support contribute neither intensities nor weights.

When every contributing tile contains every channel, all channels have the same
feather denominator. A single weight buffer replaces the separate channel
buffers. Missing-channel regions retain independent denominators. For three
complete channels this reduces weight-buffer storage and writes by two thirds.

Depth correction multiplies the source value by its channel gain during
accumulation, before feather weighting. It avoids allocating a corrected copy
of the whole source block. Exclusive regions apply the same gains when copied.
The multiplication order and intermediate precision match the previous
implementation. Uncorrected uint16 pixels retain double-precision weighted
multiply/add intermediates stored into float32 accumulation. Corrected sources
retain float32 gain multiplication before feathering.

Normalization writes clipped, truncated uint16 values directly into the output.
Float32 output normalizes the accumulation buffer in place before copying its
region into the output. Neither path allocates a separate clipping or cast
buffer for an overlap. Existing parallelism operates between blocks; the kernels
release the GIL without introducing another thread pool.

## Maximum-projection fusion

Projection fusion weights each owned float32 tile crop in place and reuses the
accumulation array for division, clipping, and rounding. Globally zero channels
remain excluded from the denominator. Projection integer output still rounds
to nearest, while full-volume integer output still truncates.

A shared TensorStore source cache retains decoded chunks between the zero-channel
scan and spatial fusion reads. It uses the full-volume fusion cache policy:
10% of available host RAM, capped at 4 GiB. The inputs are completed processed
images and remain unchanged during fusion. The cache belongs to that fusion
instance; it adds no global state or processing option.

## Measurements, 2026-10-07

The reference was the working tree immediately before this optimization, after
the cleanup and restoration of the uint16 direct-copy path. One-off comparisons
loaded snapshots of the original kernels, rendering method, and projection
constructor/fusion method. Compilation, warmup, store creation, registration,
and pyramid generation were excluded. The following are median seconds from
nine alternating trials on this Windows workstation.

Rendering used three-channel uint16 sources with nonuniform feather profiles
and 50% X overlap. The dimensions below describe each source tile. Float32
labels refer to the output format.

| Rendering case | Before (s) | After (s) | Speedup |
| --- | ---: | ---: | ---: |
| 8 x 128 x 128, uint16 output | 0.00121 | 0.00103 | 1.17x |
| 8 x 128 x 128, float32 output | 0.00148 | 0.00157 | 0.94x |
| 32 x 512 x 512, uint16 output | 0.11032 | 0.07625 | 1.45x |
| 32 x 512 x 512, float32 output | 0.13120 | 0.12102 | 1.08x |
| Depth-normalized, uint16 output | 0.20322 | 0.10790 | 1.88x |
| Depth-normalized, float32 output | 0.18600 | 0.11525 | 1.61x |

Small float32 blocks are approximately 0.09 ms slower; no speedup is claimed
for that case. Depth-normalized rendering used two 32 x 512 x 512 tiles
overlapping by 16 Z planes. The forward model used a Gaussian fluorescent object,
optical paths of 0 and 16 micrometers, and known Beer–Lambert attenuation
coefficients `log(2, 3, 4) / 16` per micrometer. Camera pixels were quantized to
uint16. Both implementations produced identical pixels and stayed within the
correction-gain-adjusted half-ADU bound, plus integer truncation where applicable.

Maximum-projection comparisons used three-channel 1024 x 1024 tiles with
512-pixel X overlap and 512 x 512 fusion chunks. In-memory timings include the
zero-channel scan, fusion, and output assembly. Disk timings use a simulated
Gaussian photon object with Poisson counts, real OME-Zarr input/output stores,
and eleven alternating trials after warmup.

| Projection fusion | Before (s) | After (s) | Speedup |
| --- | ---: | ---: | ---: |
| In memory, uint16 | 0.05349 | 0.03586 | 1.49x |
| In memory, float32 | 0.04140 | 0.02743 | 1.51x |
| Disk to disk, uint16 | 0.15345 | 0.12174 | 1.26x |
| Disk to disk, float32 | 0.25589 | 0.21628 | 1.18x |

All rendering comparisons matched the original arrays exactly. Reopened
projection outputs also matched each other exactly and agreed with the simulated
photon object within float32 arithmetic precision. Full-volume disk comparisons
verified saved pixels for XY overlap, depth overlap, and separated fields, but
their write timings varied substantially between trials. They do not establish
a reliable full-volume disk-throughput improvement. The rendering gains above
measure the CPU work separately from that storage variability.

Numerical tests cover strided crops, explicit source/destination offsets,
supported dark pixels, missing channels, shared denominators, known attenuation,
float32 range preservation, uint16 saturation, and truncation at integer
boundaries with mixed-precision accumulation. Existing disk-to-disk tests
cover fusion, depth normalization, ROI processing, and multiscale storage.
Performance comparisons remain outside the unit/integration test suite.

Final validation: **237 tests passed** with `OPM_REQUIRE_GPU=1`, strict markers,
and the pytest cache disabled. Ruff lint and formatting checks passed across
`src/opm_processing`, `scripts`, and `tests`; `git diff --check` also passed.
Temporary benchmark snapshots and generated stores were removed.
