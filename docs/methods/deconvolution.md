# PSFs and deconvolution

RLGC combines Richardson–Lucy updates with a gradient-consensus gate to limit
noise amplification. It follows [Manton and collaborators' implementation](https://github.com/jdmanton/rlgc)
and the MATLAB algorithm accompanying
[Liu and Stallinga's work](https://doi.org/10.1364/BOE.597710), including its
`gc_deconv.m` read-noise correction.
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

OME-NGFF does not specify the optical calibration needed to generate this PSF.
Record the following acquisition attributes under the root's
`opm_processing.deconvolution` object:

| Field | Expected value |
| --- | --- |
| `read_noise` | Optional object with `rms` and `units`. Units are `ADU`, `electrons`, or `calibrated-detector-counts`. Missing calibration selects zero read noise. |
| `emission_wavelengths_nm` | One positive emission wavelength per channel, in storage order; required for theoretical PSFs. |
| `psf_parameters` | Optical and numerical PSF settings described below; required unless measured PSFs are supplied. |
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
| `crop_scan` | FFT tile length in reconstruction scan planes; default `None` estimates it from free GPU memory. Tiling shares one reconstruction and stopping history. |
| `fallback_step_scan` | Tile reduction after allocation failure; default reduces the length by one quarter. |
| `rng_seed` | Seed for fresh half-image splits; default `42`. |
| `limit`, `max_delta` | Reference updated-fraction and relative-change thresholds; both default `0.001`. |
| `use_kl` | Reference two-half KL stopping; default `True`. |
| `scan_upsample_factor` | Integer scan-grid refinement; default `1`. For sparse acquisition use `boundary="zero"`. |
| `boundary` | Default `"reflect"` for native sampling; `"zero"` for the acquired-plane sparse operator. |
| `max_iterations` | Optional safety cap; default `None` for RLGC. Ordinary RL (`gradient_consensus=False`) requires a finite cap. |
| `storage` | `"gpu"` or `"host"` for reconstruction arrays; default selected from tile size. FFTs use CUDA in either case. |
| `callback` | Optional function receiving the iteration index and completed reconstruction. |

The return value is a NumPy `float32` array with the input rank. With scan
refinement N, M acquired planes produce `(M-1)*N+1` output planes; YX dimensions
are retained. No camera calibration, deskew or file writing occurs inside `rlgc`.
See the [Python reference](../reference/python/imageprocessing.md) for all controls.

## Photon gradient consensus

Processing applies the recorded camera offset and conversion before deconvolution,
clipping negative calibrated estimates to zero and retaining floating-point precision.
RLGC initializes a constant estimate at the mean calibrated intensity. Each iteration
uses `do_binosplit.m`'s rounded counts: binomial sampling below 150 and a Gaussian
approximation at 150 or higher, with out-of-range Gaussian draws replaced by
half the rounded count. A uniform random fraction of the signed rounding
remainder is added to the first half. The second half is formed by subtraction.
For read-noise RMS σ in calibrated input units, split `J + σ²` instead of `J`,
subtract `σ²/2` from each half, then clip independently at zero, following
`gc_deconv.m`. Processing reads the camera RMS from the OME-Zarr calibration.
If read noise is absent, it uses zero and selects the no-read-noise model.

The consensus map blurs `(u1-1)*(u2-1)`, where `u1` and `u2` are the two half-image
multiplicative updates. It has no intensity weighting. Negative consensus rejects
an update and zero permits it. The full RL update is computed directly from the
measured image, since independently clipped halves need not sum to that image.
The full ratio is `(J + σ²)/max(Ax + σ², 1e-12)`; each half ratio is
`(half + σ²)/(0.5*max(Ax + σ², 1e-12))`, exactly as in `gc_deconv.m`.

KL stopping is enabled by default. Following `gc_deconv.m`, normalize the predicted
acquired pixels and each fresh half after adding its `1e-4` pseudocount. Stop when
both prediction-to-half KL scores exceed their preceding-iteration values,
keeping the current estimate as MATLAB does. This measurement-space convention
also applies to sparse scans; object voxels are not compared directly with
measured pixels. This selects MATLAB's `use_KL=1` branch. Its read-noise correction
is a shifted-count approximation. CuPy and MATLAB use different random generators,
so identical seed numbers do not imply identical random halves.

The existing accepted-update fraction threshold of 0.001 and maximum absolute
change divided by `max(max(reconstruction), 1e-12)` threshold of 0.001 remain active.
Ordinary RL has no built-in stopping rule for noisy data.

## PSFs, response and execution

PSFs must match channel order and reconstruction sampling. Theoretical PSFs use
recorded optical calibration, emission wavelengths and oblique geometry.
Excitation labels such as `488nm` are not substituted for emission wavelengths.
Planar processing uses the central plane of a supplied 3D PSF. Kernels are
normalized to unit sum. Supplying measured `--decon-psf-paths` bypasses theoretical
generation. Missing required PSF calibration produces an error; the library does
not choose an objective, refractive index, wavelength or specimen depth.

Flatfield and measured detector-X response enter the forward model
`A(x)=F*convolution(x)` and its sensitivity-normalized adjoint. They are not
divided into noisy measurements before deconvolution or applied again afterward.
This detector-response model approximates excitation flatfields, which physically
act before emission blur. Without deconvolution, calibration and flatfield division
retain their usual order.

The single `rlgc` solver supports planar, native and undersampled reconstruction.
Native reconstruction uses symmetric PSF halos, efficient FFT dimensions and
cached workspaces. Scan chunks use overlap-save convolution within each global
iteration: initialization, random halves, consensus and stopping are shared
across the volume. Chunk boundaries are not independently reconstructed.
Allocation fallback can retain global state in host RAM while evaluating only
haloed FFT tiles on the GPU. Host RAM must accommodate the reconstruction state.
NumPy and CuPy random generators need not produce identical random halves.
Processing records the solver configuration; incompatible outputs cannot be
mixed by `--resume`.

## Sparse scans

With `scan_upsample_factor=N`, the array solver reconstructs a finer scan grid while
predicting only acquired planes. Its adjoint scatters observations onto that
grid; missing planes are not zero measurements. An acquisition with M planes
produces `(M-1)*N+1` reconstruction planes, preserving endpoints.

The PSF and subsequent deskew use the finer spacing. This path uses the same
photon consensus and stopping rules, including the same FFT tiling. Both `process`
and `export-rois` infer this factor from acquisition metadata. ROI raw reads remain
on the acquired grid; deconvolution and subsequent deskew use the refined grid.
Both OPM paths follow `gc_deconv.m` for the noise shift. The MATLAB
`gc_deconv_up.m` file instead shifts the split by σ rather than σ²; that
inconsistent split formula is not used here.
See the [solver reference](../reference/python/imageprocessing.md) for Python parameters.
