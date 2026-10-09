# Calibration and deskew

## Camera and illumination

Camera calibration converts ADU to nonnegative float32 intensity:

```text
intensity = max((camera_counts - camera_offset) * camera_conversion, 0)
```

The stage-scan path can additionally divide by the measured detector-X response.
Cropping retains the original detector-column offset so that this correction
uses the matching calibration columns. Optional illumination correction divides
the calibrated data by a CYX illumination field. Empty-channel decisions occur
after camera calibration and before illumination division.

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
camera planes and uses four source samples. X remains the detector-column axis.
Output shape, padding, origin, and scan orientation are determined from the
acquisition geometry. See the [geometry diagram](../orthogonal_interpolation_diagram.md)
for the interpolation construction.

## Sampling and intensity

Before Z averaging, laboratory spacing is the camera pixel size along each
spatial axis. `--z-downsample-level N` averages N adjacent laboratory Z planes,
giving ZYX spacing `(N * pixel_size, pixel_size, pixel_size)`.

Interpolation includes the existing camera-pixel/scan-step intensity factor,
`pixel_size / scan_step`. This convention must be included when comparing
absolute intensities or integrated fluorescence with a simulated object.
Unsupported regions and padding remain zero. A partly covered Z bin retains the
full averaging divisor; its geometric coverage is accounted for during fusion.

The default output casts clipped values to uint16. `--save-float32` preserves
fractional intensity. Changing output dtype does not change the geometric model.

## Execution

CPU kernels parallelize independent detector or output rows. Coordinates retain
float64 precision, with float32 interpolation weights and intensities.
