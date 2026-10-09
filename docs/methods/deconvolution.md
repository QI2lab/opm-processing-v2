# PSFs and deconvolution

## Optical sampling

Production PSF generation models the silicone-objective optics, then samples
the PSF in the same oblique coordinates as the acquisition. The emission
wavelength, camera pixel size, scan step, and angle determine the sampled kernel.
Supplied PSFs must match the reconstruction grid and channel order. Planar
processing uses the central plane of a 3D PSF.

The solver normalizes the PSF to unit sum by default. Native OPM deconvolution
uses per-axis PSF halos, symmetric image padding, and FFT-friendly sizes with
small prime factors. Padded values participate in the loop; padding is removed
from the returned image. See the [PSF and solver reference](../reference/python/imageprocessing.md).

## RLGC update

RLGC repeatedly splits the measured counts into complementary halves. Whole
counts use a 50:50 binomial draw. Fractional calibrated counts use a weighted-count
approximation that assigns the remainder to either half with equal probability.
Their sum is always the measured image.

Each half's observation/prediction ratio is backprojected. The update multiplier
is the mean of the two backprojected ratios. The product of their deviations
from unity is convolved with the forward/adjoint autocorrelation operator to
form the consensus map. The consensus sign controls which multiplicative
updates are applied. The core update and sign rule follow the designated local
`expansion-processing` reference.

Initialization backprojects the measured image. Native stopping compares the
current and previous predictions against the same freshly drawn halves using
normalized observation-to-prediction KLD. In safe mode, worsening either half
restores the previous reconstruction. Other stopping criteria are the updated
pixel fraction, maximum relative change, and the iteration limit.

KLD is a stopping statistic. Physical acceptance checks specimen recovery,
fluorescence, localization, and two-point profiles against known ground truth.
See [test requirements](../development/testing.md).

The recorded audit baseline is the local
`expansion-processing/src/expansion_processing/rlgc.py` at revision
`098b8155f27bfbebf84ebaefc0a1114a64f3c287`, with SHA-256
`96d3f2ffef3b6f52cf46fcc2c9fe2ad70e29dddc8f60ed61066dfb96dd70a5ec`.
The core update is the reference requirement. Current initialization, fractional
sampling, stopping, and OPM padding adaptations are described above; the complete
loop is not claimed to be identical to that reference.

## Sub-sampled acquisitions

Experimental `--decon-scan-upsample N` reconstructs a finer scan grid using
convolution followed by selection of the acquired planes. The adjoint scatters
observations into that grid before backprojection; missing planes are not
zero-valued measurements. This operator has its own sensitivity correction and
finite-volume boundary condition.

The experimental path requires a full GPU volume and currently supports neither
scan chunking nor ROI processing. Its validation scope and boundary differences
are described in [undersampled reconstruction](../undersampled_rl_experiment.md).
Fully sampled and sub-sampled data must be assessed against the same physical
object and resolution measurements.

## Experiments

The [fine-grid PSF experiment](../psf_sampling_experiment.md) is separate from
production PSF generation. Lookup-table count splitting and PSF edge tapering
remain experiments in the audit scripts; the default solver uses its existing
sampler and optical PSF. Distribution checks alone cannot establish equivalent
object recovery or two-point resolution.
