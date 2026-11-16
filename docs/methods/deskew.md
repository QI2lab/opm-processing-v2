# Calibration and deskew

## Camera and illumination

Camera calibration converts ADU to nonnegative float32 intensity:

```text
intensity = max((camera_counts - camera_offset) * camera_conversion, 0)
```

A recorded detector-X response can additionally correct camera sensitivity.
Cropping retains the original detector-column offset so that this correction
uses the matching calibration columns. Optional illumination correction divides
the calibrated data by a CYX illumination field. Empty-channel decisions occur
after camera calibration and before illumination division.

Illumination fitting samples tiles separately at each stage-Z level. The final
field for each channel is the pixelwise maximum over those depth estimates,
shared by every stage position and timepoint. Each fitted depth field has unit
spatial mean; the maximum retains its resulting scale. Empty depth/channel groups are
excluded from the maximum; a channel with no selected signal uses unity. Fitting
uses distributed time-zero samples with recorded camera calibration. Saved fields
must be finite and positive; incompatible or invalid cached fields are estimated
again.

These calculations are separate from deconvolution and interpolation. See
[camera_correct](../reference/python/imageprocessing.md#opm_processing.imageprocessing.camera.camera_correct)
for parameter units and detector-gain behavior.

## Instrument and laboratory coordinates

The raw stack is ordered scan, camera Y, camera X. For a lateral scan, plane
position `s`, detector coordinates `u, v`, and oblique angle `theta` map as:

```text
X = u
Y = s + v * cos(theta)
Z = v * sin(theta)
```

The orthogonal interpolator projects each laboratory output point onto adjacent
camera planes and uses four source samples. Each perpendicular foot is
interpolated between two detector rows. If the output point has perpendicular
distances `d0` and `d1` from the two planes, those two interpolated values receive
weights `d1/(d0+d1)` and `d0/(d0+d1)`. The weights sum to one and reproduce a
linear laboratory intensity field. X remains the detector-column axis.
Output shape, padding, origin, and scan orientation are determined from the
acquisition geometry. See the [geometry diagram](../orthogonal_interpolation_diagram.md)
for the interpolation construction.

Stage scan origins refer to the first retained camera plane after scan ordering
and excess-plane removal. A positive stage-X trajectory is reversed, so its
deskewed image origin includes the displacement to the retained scan endpoint;
using the acquisition start would shift every absolute image coordinate by one
scan length. Negative trajectories retain their acquired order. Mixed scan
directions in one collection are rejected. Deskew cropping adds its physical Y
offset, and Z averaging places the first output sample at the bin center.

Recorded stage coordinates retain their acquisition reference in the processed
metadata and physical XYZ OME StageLabels. Spatial metadata is stored to three
decimal places. Image-placement
coordinates and local NGFF translations are separate. The
`opm_processing_coordinates` attributes identify the coordinate version and the
recorded and orientation-adjusted stage references. The detector reference is
the first stored camera pixel; this does not invent an optical-center
calibration. Acquisition records that use depth-normalized scan coordinates
retain that reference rather than being relabeled as physical stage readbacks.

New fusion can correct the missing endpoint origin of existing positive stage
deskewed tiles without changing their pixels. Registration caches use the new
coordinate version. Existing fused images and their ROI coordinates must be
regenerated together; older stage previews are rejected for ROI selection.
Legacy negative-direction stage tiles require reprocessing because their
stored pixel order also differs.

## Sampling and intensity

Before Z averaging, laboratory spacing is the camera pixel size along each
spatial axis. `--z-downsample-level N` averages N adjacent laboratory Z planes,
giving ZYX spacing `(N * pixel_size, pixel_size, pixel_size)`.

The normalized interpolation is multiplied by the photon-volume Jacobian. At
the supported 30-degree angle this is `2 * pixel_size / scan_step`, because an
input voxel has volume `scan_step * pixel_size^2 * sin(30 degrees)` and an output
voxel has volume `pixel_size^3`. Perpendicular-distance weighting replaces the
former equal-plane average while preserving this gain for constant fluorescence.
Unsupported regions and padding remain zero. A partly covered Z bin retains the
full averaging divisor; its geometric coverage is accounted for during fusion,
even when averaging leaves only one stored Z plane. Cropping retains its physical
Y offset, and output padding is applied after the crop.

The default output casts clipped values to uint16. `--save-float32` preserves
fractional intensity. Changing output dtype does not change the geometric model.

The interpolation revision is recorded in the processing configuration. Stores
made with earlier interpolation weights or truncated final Z bins require
reprocessing; `--resume` rejects incompatible configurations.

## Execution

CPU kernels parallelize independent detector or output rows. Coordinates retain
float64 precision, with float32 interpolation weights and intensities.

The array API requires explicit `theta`, `distance` and `pixel_size`; acquisition
processing extracts them from the OME-Zarr. Use `deskew_shape_estimator` with
`crop_after_deskew=False` and the same `downsample_factor` to allocate output for
`orthogonal_deskew`, including the final partial Z bin.

An optional writable float32 `out` array must have the estimated output shape;
NumPy views and memory-mapped arrays are supported. Supplied buffers are fully
written, including zero support and padding.

Acquisition processing deskews the full channel in RAM when it fits. When output
exceeds available RAM, Ryomen divides only the acquired scan axis, retaining full
camera YX planes. Each output slab uses global coordinates and is written once.
Tiled RLGC keeps its shared KL stop and passes finalized scan cores directly to a
single deskew/write worker while the GPU continues the next core. Only the source
planes needed across deskew boundaries are retained; RLGC halos are discarded.
Both paths use the same interpolation kernel and final partial Z bin.
