# Registration and blending

## Pairwise and global placement

Stage positions define nominal physical tile placement. Overlapping tile bounds,
processed shapes, voxel spacing, scan angle, and effective registration sampling
define initial per-pair search envelopes. Acquisition overlap comes from
`opm_v2.configuration.acq_config.Positions`: `scan_axis_overlap_um` takes
precedence for stage scans; otherwise `scan_axis_overlap` supplies the scan
fraction. `tile_axis_overlap` and `z_axis_overlap` supply the camera-X and
depth fractions. Fractions are converted using raw acquisition dimensions and
calibration, before processing crops or downsampling. The recorded overlap
sets the search allowance directly, without an additional percentage of it.
Missing settings fall back to measured nearest-face overlaps, or tile extent
on untiled axes. Two registration sampling bins provide a minimum allowance.
These envelopes account for oblique depth
displacement and retain at least half the nominal overlap outside the depth
search. Depth alignment uses wider context and requires at least three
actually matched planes.
Depth candidates are selected using normalized correlation of adjacent-Z
intensity differences, with support required on both adjacent planes. This
removes Z-invariant XY background from the depth evidence. Original-intensity
SSIM remains a separate acceptance check; high SSIM alone cannot resolve depth.

Registration estimates image shifts and scores corresponding supported samples.
Phase translations are periodic in the sampled patch dimensions. All
equivalents inside the physical correction limits are scored on real-space
common support, including for thin depth overlaps. This avoids trusting an
upstream disambiguation that selects a tiny strip at the opposite crop edge.
The period includes any padding from block reduction. These equivalents
remain subject to the same SSIM and contextual depth-evidence gates.
For supported overlaps it compares phase and directional-edge proposals with a
masked, nonperiodic normalized-correlation peak searched inside the physical
shift limits. The normalization uses only common acquisition support, so padded
crop edges do not provide registration evidence. Nominal stage placement is
also scored when a measurement is available. Every proposal is scored at the
full-resolution integer correction that fusion actually retains, including
when the registration images are downsampled; scoring a fractional correction
and rounding it afterward could otherwise select a different placement.
Downsampling and the similarity window adapt to overlap size. Thin depth
overlaps can use their common XY projection when a full 3D scoring window does
not fit. Accepted links enter a global placement fit.

Out-of-range peaks meeting the similarity threshold trigger a bounded automatic
retry. Only the rejected pair's offending axis limits increase, to the absolute
measured correction plus five percent, with a minimum margin of two pixels or
one effective registration sampling bin. Unaffected limits stay unchanged.
Each retry remeasures only pairs with expanded limits and repeats the global
fit and outlier rejection across all timepoints from nominal stage placement.
Unchanged raw measurements are reused, including links previously rejected by
the global fit. Shared registration setting changes invalidate the cache and
start a fresh full pairwise pass. The default permits two retries;
explicit user-supplied limits stay fixed. Similarity and actual support checks
continue to gate every measurement. Registration caches record initial settings
and final effective limits so a later run can reuse the completed search.
The proposal and scoring revisions are part of the cache fingerprint, so
registrations measured with the preceding algorithm are recomputed.

Global optimization evaluates residuals in effective registration sampling units.
Inconsistent redundant links are removed one at a time with a refit after each
removal. Links that anchor connected tile groups are retained. Connectivity
warnings describe the optimized graph; separate groups retain stage-derived
placement between them.

## Depth intensity normalization

Full acquisitions and ROIs identify depth layers from repeated visits to recorded
physical stage XY locations. Registered Z offsets do not create additional
depth layers. Matching runs separately for each timepoint and channel, after
flatfield correction, optional deconvolution, and deskewing.

For full acquisitions, the complete supported processed stack at each later
depth is histogram-matched to the first acquired depth at the same physical XY
footprint. For ROIs, contributing tiles are combined by acquisition depth inside
the selected XY bounds. Each combined cropped depth layer is histogram-matched
to the first acquired layer of that ROI. An unexpectedly missing first depth
raises an error rather than substituting a later depth.

Samples are normalized for interpolation coverage before matching. Geometric
deskew padding is excluded, while valid zero-intensity observations remain in
the histogram. Combined layers retain the processed output dtype: uint16 inputs
use the upstream integer histogram path and a lookup table of at most 65,536
entries; float32 inputs retain floating-point precision.
The upstream `match_histograms` implementation supplies the
intensity mapping. cuCIM is preferred when its estimated workspace fits GPU
memory; scikit-image handles volumes that do not fit or encounter a GPU allocation
failure.

The resulting mapping applies consistently to every fusion block, with coverage
restored for weighted accumulation. Source arrays remain unchanged. Matching
provenance stays beside the processed tiles in
`<stem>_depth_histograms.json`.

## Feathering and coverage

Tile edges receive separable Z, Y, and X feather weights. Each block accumulates
weighted intensity and a corresponding denominator before normalization.
Deskew support comes from metadata and interpolation geometry, including
partial Z-bin coverage. A dark pixel within valid support still contributes
weight; pixel intensity is not used to infer geometric coverage.

The numerator already contains the deskew coverage factor, so fusion applies
coverage to the denominator. This avoids fading partly supported depth bins.
Missing channels are excluded from that channel's accumulation. Camera-X padding
has zero feather weight.

With `--content-aware`, overlap samples are coverage-normalized before scoring.
The GPU computes DCT entropy in 32-pixel cells, clamped to the overlap extent,
using the [pinned multiview-stitcher implementation](https://github.com/multiview-stitcher/multiview-stitcher/blob/708fd8f96576e4e2934f1bc5d3e51e374c97de9f/src/multiview_stitcher/weights.py#L85):
L2-normalized coefficients inside the L1 frequency cutoff of half the smallest
cell dimension. Scores are shifted by their per-cell minimum across contributors,
normalized, and linearly interpolated from cell centers. Constant cells have
zero entropy, avoiding amplification of FFT roundoff. Content weights multiply
geometric feather/coverage weights; zero totals fall back to geometric weights.
The DCT lattice is anchored to the physical overlap, with neighboring-cell
context for interpolation, so output-block boundaries do not reset the weights.
ROIs still combine each depth before applying its histogram map, then content-blend
the corrected layers. Source reads and direct-channel copies share the existing
accumulator; the GPU calculation batches contributors and channels.

## Execution

Registration requires CuPy/cuCIM. The scikit-image fallback applies only to
depth histogram matching.
Fusion uses threaded blocks and Numba CPU kernels with bounded memory and queued
writes. Content-aware fusion instead uses one GPU render worker, with block
sizing that accounts for GPU working memory and DCT context. Depth and coverage
corrections retain source precision before the final output cast, including
directly copied regions. Full-volume integer output clips
and truncates; maximum-projection
fusion retains its rounding convention.

Use the [fusion guide](../guides/fusion.md) for practical settings.
