# Fine-grid Cartesian PSF sampling experiment

`psf_sampling_experiment.py` is separate from production PSF generation and
orthogonal deskew. Neither production file is modified, and the CLI does not
select this experiment.

The model is the existing 100x silicone vectorial Cartesian PSF, transformed
into instrument coordinates by skewing and slicing. This experiment evaluates
that sampling explicitly in physical coordinates. It does not replace the
optical model or infer a defect from the orientation of a deskewed point.

## Order of operations

1. Generate the Cartesian optical PSF at a finer isotropic spacing.
2. Define a ground-truth fluorescent object on that fine Cartesian grid.
3. Convolve there, before either skewing or discarding scan planes.
4. Interpolate the blurred field in all three dimensions at raw pixel centers:
   `X = column * pixel`,
   `Y = scan * step + row * pixel * cos(angle)`,
   `Z = row * pixel * sin(angle)`.

Indices are centered, with an explicit optional XYZ center in microns. Evaluating
these coordinates directly avoids an extra intermediate resampling of a skewed
array. `sample_skewed` preserves field-value units; it does not renormalize image
data. For a convolution kernel, normalize the sampled PSF to unit sum afterward.
Raw detector-area integration, noise and motion are separate from this sampling
check and are not included. The optical library's fine-pixel integration remains
part of evaluating the Cartesian field.

For the undersampled reconstruction, sample the kernel at the reconstruction
scan step (for example 0.4 um), and generate measured data at the acquired step
(0.8 um). Camera-row and camera-column pitch remain 0.115 um in both cases.

## Numerical checks

The unit tests use a physical affine fluorescence-density field, which has
exact interpolated values, to check coordinate mapping at 30 and 38 degrees,
including a shifted origin and an even-sized camera dimension.

Integration tests generate a finite-size Gaussian fluorophore distribution
blurred by the vectorial optical PSF. The fluorophore distribution is not a
Gaussian approximation to the PSF. The result is sampled at 0.2, 0.4 and 0.8 um
scan spacing with 115 nm camera pixels.

Fine-grid spacings are 57.5, 28.75 and 14.375 nm. Physical support boundaries
are identical across all three grids; changing boundaries would confound
interpolation error with cropped PSF tails. Against the finest reference:

| Cartesian spacing | Sampled PSF relative L2 error | Blurred-object relative L2 error |
| --- | ---: | ---: |
| 57.5 nm | 2.27–2.30% | 2.69% |
| 28.75 nm | 0.356–0.362% | 0.808–0.810% |

These are finite-support convergence checks, not a measurement of accuracy
against a real bead acquisition. A further test verifies that 0.8 um data
sample exactly the same physical planes as every fourth plane of a centered
0.2 um scan. All ten unit/integration cases pass.

## Comparison with the current generator

The current generator creates a finer XY/Z optical grid, then interpolates XY
while assigning optical slice `ii` directly to camera row `ii`. The experiment
instead interpolates Z explicitly as well. Any comparison must hold optical
support, centering, normalization, and the existing empirical taper fixed before
attributing differences to that indexing. The experimental reference omits
that taper so it cannot by itself establish that the production PSF is wrong.
No conclusion about the live-data tilt follows from these tests yet.
