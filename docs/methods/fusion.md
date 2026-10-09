# Registration and blending

## Pairwise and global placement

Stage positions define nominal physical tile placement. Overlapping tile bounds,
processed shapes, voxel spacing, scan angle, and effective registration sampling
define per-pair search envelopes. The automatic limits retain at least half
the overlap along active axes and account for oblique depth displacement.

Registration estimates image shifts and scores corresponding supported samples.
Downsampling and the similarity window adapt to overlap size. Thin depth
overlaps can use their common XY projection when a full 3D scoring window does
not fit. Accepted links enter a global placement fit.

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

The production fitter estimates gains from observations. It does not require a
known tissue attenuation coefficient. Physical tests use a known Beer-Lambert
forward model, `I(d) = I(0) * exp(-mu * d)`, so expected correction gains follow
independently from `exp(mu * (d - d_anchor))`. The known coefficient belongs to
the simulated ground truth; it is not a value inferred from the test result.

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
Missing channels are excluded from that channel's accumulation.

## Execution

Registration uses CuPy/cuCIM when available and otherwise NumPy/scikit-image.
Fusion uses threaded blocks and Numba CPU kernels with bounded memory and queued
writes. Full-volume integer output clips and truncates; maximum-projection
fusion retains its rounding convention.

Use the [fusion guide](../guides/fusion.md) for practical settings.
