# PSFs and deconvolution

RLGC combines Richardson-Lucy updates with gradient consensus to limit noise
amplification. Reconstruction and consensus follow Equations (3)-(5) in
[Liu and Stallinga](https://doi.org/10.1364/BOE.597710). Binomial splitting and
prediction-to-half KL stopping follow
[Manton and collaborators' Colab notebook](https://colab.research.google.com/drive/1mfVNSCaYHz1g56g92xBkIoa8190XNJpJ).
The source header in `imageprocessing/rlgc.py` describes the optical and GPU adaptations.

## Process an OME-Zarr

```bash
uv run process "/path/to/acquisition.ome.zarr" --deconvolve --save-float32
```

Processing extracts axes, camera pixel pitch, scan step, oblique angle, channels,
camera offset and intensity conversion from the acquisition. It calibrates raw
camera values, deconvolves in the oblique acquisition grid, then deskews and
rotates the result. `--save-float32` preserves fractional output values.
For 3D deconvolution, the recorded scan spacing automatically determines integer
refinement toward 0.4 micrometers: 0.8 micrometers uses factor 2 and 1.2 uses
factor 3. Acquisitions already at 0.4 micrometers or finer retain their recorded
spacing. Coarser steps must be integer multiples of 0.4 micrometers for the
acquired-plane solver.
Without deconvolution, processing uses the acquired scan spacing.
The [usage example](../examples/index.md) runs the same pipeline from Python.

As-acquired OME-Zarr stores do not need processing-specific PSF attributes.
Processing defaults to the qi2lab 100x, NA 1.35 silicone objective, with immersion
index 1.4, specimen index 1.38, coverslip index 1.515 and thickness 170 micrometers,
and design working distance 300 micrometers. The default source depth is zero
for oblique scans and 15 micrometers for planar kernels. Camera pitch, scan step
and angle still come from acquisition metadata.

Without recorded emission calibration, processing uses the channel wavelength
as a PSF approximation. Combined labels such as `561 + 637` represent one stored
camera channel and use the longest wavelength, 637 nm in this example. This
approximation does not resolve separate emission spectra within a combined image.
The resolved optical settings and wavelengths are recorded in the processing
configuration. The acquisition store is left unchanged.

Optional overrides belong under the root's `opm_processing.deconvolution` object:

| Field | Expected value |
| --- | --- |
| `read_noise` | Optional object with `rms` and `units`. Units are `ADU`, `electrons`, or `calibrated-detector-counts`. Missing calibration selects zero read noise. |
| `emission_wavelengths_nm` | One positive emission wavelength per channel, in storage order; overrides the channel wavelength approximation. |
| `psf_parameters` | Complete optical and numerical PSF settings described below; overrides the processing defaults. |
| `detector_gain_x` | Optional measured positive response, one value per detector column. Applied only when recorded, independently of image width or acquisition mode. |

Frame `Camera.read_noise_e` or `Camera.read_noise_adu` is also read when the root
has no read-noise calibration. Noise specified in ADU is multiplied by the recorded
camera intensity conversion. Offset subtraction does not change noise RMS.
Noise already in calibrated units is used directly, without applying the gain again.
No camera model or scan mode is inferred when calibration is missing.

`psf_parameters.lens` contains `NA`, `ni0`, `ni`, `ns`, `ng0`, `ng`, `tg0`, `tg`,
and `ti0`: numerical aperture, design/actual immersion and specimen refractive
indices, design/actual coverslip refractive indices and thicknesses, and design
working distance. Lengths are in
micrometers. Supply `skewed_depth_um` and `projection_depth_um` as source depths;
`taper_sigma_px_3d` and `taper_sigma_px_2d` as Gaussian taper widths in pixels;
`support_fwhms`, `depth_width_scale`, and `grid_oversampling` as dimensionless
numerical controls; `depth_scale_um` in micrometers; and `projection_nz`,
`projection_nxy`, `projection_dz_um` as the planar PSF grid dimensions and axial
spacing. These values must describe the acquisition's optical calibration.

The generated 3D PSF integrates over the tilted camera pixels using the recorded
camera pitch. `grid_oversampling` is a positive integer controlling optical-grid
resolution and midpoint integration samples along each pixel dimension.

To use measured kernels instead, repeat `--decon-psf-paths` once per channel.
Files may be NumPy `.npy` or TIFF, in camera YX or scan-YX order, sampled on the
reconstruction grid. The library uses the OME-Zarr's camera and geometry metadata
for either PSF source.

## Call the array solver

```python
from opm_processing.dataio.deconvolution import read_noise_from_metadata
from opm_processing.imageprocessing.rlgc import rlgc

result = rlgc(
    image=calibrated_data,
    psf=sampled_psf,
    read_noise=read_noise_from_metadata(acquisition),
)
```

Here `acquisition` is returned by `inspect_acquisition`; `calibrated_data` and
`sampled_psf` are arrays prepared as follows:

| Parameter | Expected value and default |
| --- | --- |
| `image` | Nonnegative calibrated floating-point YX or scan-YX data, after subtracting the recorded camera offset and multiplying by its recorded conversion. Exclude T, position and C axes; process each volume separately. |
| `psf` | Nonnegative finite YX or scan-YX kernel, centered and sampled on the reconstruction grid. Normalized to unit sum by default. A planar image uses the central plane of a 3D kernel. |
| `read_noise` | RMS in the same units as `image`; default `0` selects no read noise. |
| `detector_response` | Optional positive YX illumination/sensitivity response, broadcast over scan planes. Pass the image before dividing by this response; default `None`. |
| `gpu_id` | CUDA device index; default `0`. |
| `crop_scan` | Retained scan-chunk length in reconstruction planes; default `None` attempts the full scan and reduces it only after an actual allocation failure. Successful sizes are reused across a tile's channels. KL-enabled tiling parks chunks in RAM and continues them to one shared iteration budget. |
| `fallback_step_scan` | Tile reduction after allocation failure; default reduces the length by one quarter. Shared-KL runs allow fallback only before any completed update. |
| `rng_seed` | Seed for fresh half-image splits; default `42`. |
| `verbose` | Progress bars show chunk visits. Diagnostics are off by default (`0`); set `1` for chunk geometry, iterations and stopping reasons. |
| `limit`, `max_delta` | Updated-fraction and relative-change thresholds for RLGC with KL disabled; both default `0.001`. Ordinary RL uses only `max_delta`. |
| `use_kl` | Select the iteration budget using the reference two-half KL divergence; default `True`. Tiled RLGC shares the largest first-local-KL accepted iteration. Fraction/change thresholds do not preempt KL. |
| `scan_upsample_factor` | Integer scan-grid refinement; default `1`. For sparse acquisition use `boundary="zero"`. |
| `boundary` | Default `"reflect"` for native sampling; `"zero"` for the acquired-plane sparse operator. |
| `pad_yx` | Symmetric detector-axis padding for native reconstruction; default `True`. Zero-boundary reconstruction always pads Y/X. |
| `max_iterations` | Optional safety cap; default `None` for RLGC. Ordinary RL (`gradient_consensus=False`) requires a finite cap or positive `max_delta`. |
| `storage` | `"gpu"` (default) or `"host"` for the active chunk's reconstruction arrays; FFTs use CUDA in either case. KL-enabled tiling also stores continuation checkpoints in system RAM. |
| `callback` | Optional function receiving provisional iterates, including lookahead updates that may be discarded. Native arrays include padding. Per-chunk iteration numbers continue across visits. |

The return value is a NumPy `float32` array with the input rank. With scan
refinement N, M acquired planes produce `(M-1)*N+1` output planes; YX dimensions
are retained. No camera calibration, deskew or file writing occurs inside `rlgc`.
See the [Python reference](../reference/python/imageprocessing.md) for all controls.

## Photon gradient consensus

Processing subtracts the recorded camera offset, applies its conversion, clips
negative calibrated estimates to zero, and retains floating-point precision.
RLGC initializes a constant estimate at the mean calibrated intensity for reflected,
unit-response convolution. Otherwise it uses `sum(J)/sum(A1)` to initialize in object
units, including detector response, PSF normalization and boundary/sampling geometry. Normalized KL cannot
identify a global intensity-scale error, so its rollback state must already use
the recorded sensitivity calibration.

Each iteration splits integer photons with exact binomial thinning at every
count level. For fractional calibrated counts `S=n+f`, draw `Binomial(n, 0.5)`
and independently assign the remainder `f` to either half with equal probability.
The second half is obtained by subtraction, preserving the measured sum and
nonnegative halves. Both halves have expected value `S/2`; their conditional
variance is `(n+f**2)/4`. This fractional extension removes the notebook's
systematic assignment of the remainder to the second half. It remains an
approximation for camera data, without claiming independent Poisson halves.

For recorded read-noise RMS `sigma`, split `J + sigma**2`. The shifted-count
approximation follows the paper's cited read-noise methods: the full-data ratio
is `(J + sigma**2)/max(Ax + sigma**2, 1e-12)`, and each half-exposure ratio is
`2*shifted_half/max(Ax + sigma**2, 1e-12)`. These shifts cancel the half-variance
subtraction algebraically, avoiding clipped intermediate counts in the updates.
Processing reads the RMS calibration from acquisition metadata; absent noise
calibration selects zero. This is an approximate Poisson-Gaussian model.

Equation (4) forms the additive half-image changes `delta_i = x*(u_i-1)` and
smooths their product with the optical interaction kernel. The spatially varying
`x**2` weighting is inside that smoothing. Equation (5) applies the shared
full-data RL update only where consensus is strictly positive; zero and negative
consensus retain the preceding voxel value.

KL divergence terminates nonempty RLGC reconstruction by default. As in Colab,
normalize the acquired camera prediction `Ax` and each fresh half after adding
a `1e-4` pseudocount. Stop when both prediction-to-half KL scores increase from
the preceding iteration, restoring the preceding reconstruction. This compares
measurement-space arrays for both native and refined scan grids. With read
noise, the KL statistic uses nonnegative variance-corrected halves; the full
and half RL updates retain the shifted counts described above.
This defines each chunk's first accepted KL stopping iteration; tiled execution
uses those iterations to select the shared budget described below.

Accepted-update fraction and maximum-relative-change thresholds do not terminate
a KL-enabled run. They remain available for RLGC with `use_kl=False`, with both
defaults `0.001`. Relative change is the largest absolute update divided by
`max(max(reconstruction), 1e-12)`. An explicitly supplied `max_iterations` remains
a safety cap. An all-zero calibrated chunk returns zero without iterations.

Both RL and RLGC use the same full-data RL multiplier.
`gradient_consensus=False` removes the consensus gate and KL termination for
controlled ordinary-RL comparisons; it does not select a different RL update.
Ordinary RL uses its explicit iteration cap or positive `max_delta`.

## PSFs, response and execution

PSFs must match channel order and reconstruction sampling. Theoretical PSFs use
resolved optical settings and wavelengths, and recorded oblique geometry.
Planar processing uses the central plane of a supplied 3D PSF. Kernels are
normalized to unit sum. Supplying measured `--decon-psf-paths` bypasses theoretical
generation. Explicit optical calibration takes precedence over defaults. An
unresolvable channel wavelength requires emission calibration or a measured PSF.

Flatfield and measured detector-X response enter the forward model
`A(x)=F*convolution(x)` and its sensitivity-normalized adjoint. They are not
divided into noisy measurements before deconvolution or applied again afterward.
This detector-response model approximates excitation flatfields, which physically
act before emission blur. Without deconvolution, calibration and flatfield division
retain their usual order.

The single `rlgc` solver supports planar, native and undersampled reconstruction.
Native observations use symmetric PSF halos and a free padded reconstruction,
with the acquired core returned afterward. Zero-boundary reconstruction always
includes linear-convolution Y/X padding, even with `pad_yx=False`, preventing
detector-edge wraparound. Both paths use CuPy's cuFFT-friendly dimensions and
reuse FFT plans and device allocations. Ryomen schedules scan chunks with a
discarded halo of one full PSF scan support on either side. With KL enabled, chunks
share the largest first-local-KL accepted iteration, subject to `max_iterations`.
This is distinct from a pooled whole-volume KL
divergence. Haloed reconstruction checkpoints, owned rollback cores and the actual
random-generator state are parked in system RAM. Each chunk is visited at most
twice, continuing only missing updates without replaying completed iterations.
Acquisition processing consumes finalized cores in scan order, deskewing and
writing each core while the GPU continues the next one. It does not assemble a
second full reconstruction. The array API still returns the complete volume
unless `on_reconstructed_chunk` is supplied.
GPU working memory remains bounded by the active chunk; RAM checkpoint storage
grows with the number and extent of overlapping chunks. Initialization and random
halves remain per chunk, so tiled and whole-volume reconstructions need not match.

Before allocating checkpoints, the solver checks available RAM for their total
storage, output and workspace. Insufficient RAM raises an error. Device allocations
are limited to available physical GPU memory, respecting tighter caller limits.
In shared-KL runs, startup allocation failure can reduce the scan
chunk, but an allocation failure after a completed update is reported without
restarting the reconstruction. FFT plans and device allocations are reused between chunks and
released after the volume. NumPy and CuPy random generators need not produce
identical random halves. Processing records the solver revision in its configuration;
outputs from earlier revisions are incompatible with `--resume`.

## Sparse scans

With `scan_upsample_factor=N`, the array solver reconstructs a finer scan grid while
predicting only acquired planes. Its adjoint scatters observations onto that
grid; missing planes are not zero measurements. An acquisition with M planes
produces `(M-1)*N+1` reconstruction planes, preserving endpoints.

The PSF and subsequent deskew use the finer spacing. This path uses the same
photon consensus and KL divergence termination, including sequential scan
chunking. Both `process` and `export-rois` infer this factor from acquisition
metadata. ROI raw reads remain
on the acquired grid; deconvolution and subsequent deskew use the refined grid.
Both OPM paths use the same read-noise variance shift and acquired-camera KL
comparison.
See the [solver reference](../reference/python/imageprocessing.md) for Python parameters.
