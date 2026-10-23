# Registration and blending

## Pairwise and global placement

Stage positions define nominal physical tile placement. Overlapping tile bounds,
processed shapes, voxel spacing, scan angle, and effective registration sampling
define initial per-pair search envelopes. These account for oblique depth
displacement and retain at least half the nominal overlap outside the depth
search. Depth alignment uses wider context and requires at least three
actually matched planes.
Depth candidates are selected using normalized correlation of adjacent-Z
intensity differences, with support required on both adjacent planes. This
removes Z-invariant XY background from the depth evidence. Original-intensity
SSIM remains a separate acceptance check; high SSIM alone cannot resolve depth.

Registration estimates image shifts and scores corresponding supported samples.
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

Only registered pairs at the same stage XY and different depths provide gain
measurements. Supported positive overlap samples are corrected for geometric
coverage, spatially averaged, and filtered for shared signal. Robust log ratios
constrain multiplicative gains independently for each timepoint and channel.
The first depth in each connected gain component is anchored to one; unmeasured
depths retain unit gains. Every XY tile at a depth shares that depth's gain.

The fitter estimates gains from observations and does not require a tissue
attenuation coefficient.

Gains apply during fusion and are recorded beside the fused volume. Source
arrays remain unchanged. XY-field brightness differences are preserved.

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

## Execution

Registration uses CuPy/cuCIM when available and otherwise NumPy/scikit-image.
Fusion uses threaded blocks and Numba CPU kernels with bounded memory and queued
writes. Depth and coverage corrections retain source precision before the final
output cast, including directly copied regions. Full-volume integer output clips
and truncates; maximum-projection
fusion retains its rounding convention.

Use the [fusion guide](../guides/fusion.md) for practical settings.
