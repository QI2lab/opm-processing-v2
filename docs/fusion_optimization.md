# Fusion optimization

The 2026-10-07 optimization uses fewer intermediate arrays while preserving the saved pixels,
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

## Input selection, Z coverage and registration support, 2026-10-08

The reported directory command selected `balmer_tissue_stage_deskewed.ome.zarr`
even though the deconvolved volume existed. Its processing journal confirmed
that source. Directory selection now prefers deconvolved data within each
processed kind, with deskewed volumes ahead of planar images. The CLI prints
the selected store. An explicit processed-store path retains its selection.

Deskew averages laboratory Z samples into the saved Z bin. At the interpolation
boundary, some samples are invalid and contribute zero. A two-sample bin with
one valid sample therefore contains half of the complete-bin response. Treating
every nonzero support row as complete gave that dim bin full fusion weight.
A simulated disk-to-disk case reproduced a 10 percent dark seam despite exact
alignment and exactly recovered Beer–Lambert depth gains.

Fusion now derives the fraction of valid samples from the existing constant-input
deskew calculation and known full response `2 * pixel_size / scan_step`. Round
the resulting valid-sample count before dividing by the Z averaging factor, so
complete bins retain exactly unit coverage. This uses scan geometry rather than
image brightness. Supported zero-valued specimen pixels retain their weights.

For an overlap, source values already include their coverage. Accumulate their
original feathered signal and multiply the feather denominator by coverage.
Exclusive regions normalize nonzero coverage directly. Registration patches and
depth-gain measurements use the same geometric normalization; the registration
fingerprint invalidates measurements made with the previous convention.
Depth gains remain one value per depth/channel/time, shared across XY fields.
Source stores and deskew arithmetic remain unchanged.

Registration also treated the rectangular intersection of tile bounds as valid
specimen throughout. Adjacent Y fields have opposing deskew wedges: padding in
one patch can coincide with real signal in the other. This lowered SSIM despite
correct measured shifts. On channel 2, four links from tiles 45–48 to tiles
52–55 scored 0.50–0.67 when padding was included, versus 0.92–0.96 when only
fully supported SSIM windows were scored. Fractional phase-correlation shifts
were also rounded before scoring, losing precision on deconvolved features.

Use the metadata-derived deskew support for each patch, reduced with a minimum
over the same sampling blocks as its image. Exclude incomplete blocks, including
the padded final X block. Align both the image and its support at the measured
fractional shift, trim wholly unsupported edge planes, and score windows wholly
inside the common support. Integer shifts retain their array-view fast path.
The channel, correction limits, phase-correlation search and 0.7 acceptance
threshold remain the same. No link is accepted solely to connect components.
The registration fingerprint invalidates previously scored measurements.

Replacing a fused artifact with a different processed source also transfers its
journal association and invalidates the earlier maximum projection. Pairwise
measurements for the previous source remain available. This prevents the shared
output filenames from being associated with both plain and deconvolved inputs.

The regression writes a known fluorescent line and background to a simulated
camera acquisition, applies known attenuation `exp(-log(2) * depth / 24 um)`,
and runs real processing and fusion reads/writes. Simulated processed inputs
contain either the line or its Gaussian optical blur. Exact registered stage
placements isolate fusion from feature-registration uncertainty. Reopened fused
and maximum-projection pixels recover the deconvolved object through the Z
overlap to float32 precision. Numerical cases also cover half-filled bins with
strided crops, missing channels and both output dtypes.

Nine warmed, alternating CPU trials on two fully covered three-channel
32 × 512 × 512 uint16 sources compare the corrected accumulation/normalization
with the optimized implementation immediately before this fix. Saved pixels are
identical. Median time is 0.08963 versus 0.08980 seconds without depth gains,
and 0.07786 versus 0.07685 seconds with gains. These differences (−0.2 and
+1.3 percent) do not indicate loss of the earlier fusion optimization. Allocations
and resets are excluded; no new disk-throughput claim is made.

The actual deconvolved acquisition now has one connected component containing
all 56 tiles on channel 2 with ZYX limits 20/250/100 and SSIM threshold 0.7.
Registration accepted 119 measured links; global optimization retained 118,
including all four adjacent-row links connecting the previously isolated group.
Their retained SSIM scores are 0.917, 0.934, 0.939 and 0.954.

Numerical regressions independently check fractional translation of an analytic
intensity ramp, SSIM exclusion of opposing wedge padding, supported dark pixels,
and rejection of unrelated signal. CUDA registration checks use integer shifts
and exact periodic fractional shifts. The ROI regression also runs real
disk-to-disk reconstruction, registration, fusion and maximum projection on a
known three-channel line object, without mocked processing boundaries.

The read-only acquisition check, fusion previews and test reports are saved in ignored
`diagnostics/fusion_20261008/`. The acquisition's existing fused outputs have
not been overwritten by these diagnostics.

Validation: 203 non-GPU-marked tests and both CUDA registration cases passed.
Ruff lint and formatting checks passed for all 72 source, script and test files;
`git diff --check` passed. Temporary scripts, snapshots and generated test stores
were removed after verification.


### Deconvolution pipeline validation

The revised deconvolution recovers finer features in the simulated tiled
workflow. Tiny overlaps now use full Z registration sampling in that physics
integration, and fused resolution is compared with its deconvolved source.
Some thin depth overlaps contain common observed voxels but no complete 3D SSIM
window. When an aligned depth overlap is no thicker than the configured window
plus two planes, scoring may compare XY means over the same common valid
planes. This preserves geometric support and uses the existing threshold and
search bounds. Unit checks retain a known match and reject independent signal;
the disk-to-disk workflows check connected graphs, absolute coordinates,
object correlation and retained resolution. Fusion execution is unchanged.
