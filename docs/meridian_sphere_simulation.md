# Meridian-sphere OPM simulation

This is a separate, stationary-object forward simulation. Production orthogonal
deskew and PSF generation are unchanged. The forward check below does not use
deconvolution; the subsequent GC comparison uses the saved noisy camera data.

## Object and optics

The ground truth is a hollow sphere with **10 um meridian-centerline diameter**
and **1 um diameter fluorescent tubes**. Its outer diameter is therefore 11 um.
Twelve equally spaced pole-to-pole meridians form six full great circles through
the Z axis. Tubes join at the poles; overlapping fluorescence is a union, so
the poles are not artificially brighter because multiple tubes overlap. The
interior and gaps between meridians contain no fluorophores.

The object is defined first on an isotropic **50 nm Cartesian grid**, finer than
the 115 nm camera pitch and its 57.5 nm projected Z spacing at 30 degrees.
The optical PSF is the vectorial 100x silicone model, with configurable NA and
wavelength (defaults NA 1.35, 637 nm; immersion RI 1.4, sample RI 1.38). Optical
support is +/-2.4 um in Z and +/-1.2 um in XY. Both simulation routes use this
same finite-support PSF. There is no empirical taper.

## Independent forward routes

**OPM route:** interpolate the fine Cartesian object and PSF into instrument
coordinates. The fine instrument grid has 0.1 um scan spacing and 57.5 nm
camera-row/column spacing. Convolve the object with the normalized skewed PSF
on that fine grid. Only then select the requested acquired scan planes and
sample/average camera pixels. The transformations are

```
X = camera_column * pixel_pitch
Y = scan_index * scan_step + camera_row * pixel_pitch * cos(angle)
Z = camera_row * pixel_pitch * sin(angle)
```

All grids have explicit centered coordinates in microns. Optical halos and
the complete object are retained before convolution, which uses zero exterior
fluorescence rather than FFT wraparound.

**Cartesian microscope route:** independently convolve the ground truth and
Cartesian optical PSF on the fine Cartesian grid, then digitally sample it.
For the comparison, evaluate it at the physical coordinates of the deskewed
output. No registration, fitted gain, or alignment optimization is applied.

There is also a direct check *before deskew*: sample the Cartesian convolution
on the raw OPM grid and compare it with the independently computed skewed
convolution. This distinguishes forward-model sampling error from later
deskew interpolation differences.

Production deskew applies a constant-field gain of `2*pixel/scan_step`. Since
the simulator's clean arrays represent fluorescence density, multiply its
output by the reciprocal of that known factor for the comparison. This does
not change production deskew or fit image brightness. Deskew Z downsampling is
disabled for these comparisons; output voxels are 115 nm isotropic.

## Camera sampling and noise

Start with `--camera-samples 1`, which samples camera-pixel centers without
noise. After geometry validation, `--camera-samples 3` averages a 3x3 set of
uniform subpixel centers within each detector pixel. This approximates binning
the finer irradiance field over the camera pixel area. The OPM detector plane
is tilted; the Cartesian microscope detector averages in laboratory XY. Their
pixel-area responses need not be identical.

Convert the averaged intensity to expected detected photoelectrons using
`--peak-electrons`, add `--background-electrons`, draw Poisson counts, and then
add Gaussian read noise with `--read-noise` in electrons RMS. Noise is applied
once per camera pixel after averaging, not once per interpolation subpixel.
Both microscopes use the same exposure gain and independent noise draws. The
gain is based on the peak of the fine Cartesian optical image, not fitted to
the sampled OPM image. Background-subtracted density is clipped at zero before
deskew, introducing a small positive background bias; original electron-valued
raw data are also saved. ADC quantization, saturation, motion, and rolling
shutter timing are not included.

## Reproduce

Run from the repository root:

```powershell
uv run python -m opm_processing.imageprocessing.opm_simulation --output diagnostics/meridian_sphere --scan-steps 0.2 0.8
```

Then add camera integration and noise:

```powershell
uv run python -m opm_processing.imageprocessing.opm_simulation --output diagnostics/meridian_sphere_camera --scan-steps 0.2 0.8 --camera-samples 3 --peak-electrons 500 --background-electrons 2 --read-noise 1.5
```

Geometry and optics can be changed with `--diameter`, `--tube-diameter`,
`--meridians`, `--angle`, `--pixel-size`, `--na`, `--wavelength`,
`--fine-spacing`, and `--fine-scan-step`. Wavelength and all spatial inputs are
in microns. `--seed` controls camera noise. Finer grids cost more memory/time.

Each run exports fine ground truth and both PSFs. Each acquisition directory
contains raw skewed data, deskewed data, the Cartesian microscope reference,
the noise-free reference, and sampled truth. Cartesian comparison volumes
are OME-TIFFs with physical voxel sizes, as are the fine Cartesian truth and PSF.
Raw TIFF Z denotes scan index, not
Cartesian Z. `simulation.json` records the actual axes, physical origins,
parameters, and quantitative comparisons. `comparison.png` shows maximum
projections, `sections.png` shows the central planes through the hollow sphere,
and `raw_instrument.png` shows the oblique data before deskewing.

Render the saved volumes without rerunning optics:

```powershell
uv run --with matplotlib python -m opm_processing.imageprocessing.plot_opm_simulation diagnostics/meridian_sphere/scan_0.2um
```

## Results and checks

For the noiseless default object, the independent forward routes agree to
1.96% relative L2 error and 0.999913 correlation. After unchanged orthogonal
deskew:

| Acquired scan step | Error vs Cartesian microscope | Correlation | Total signal ratio |
| --- | ---: | ---: | ---: |
| 0.2 um | 2.81% | 0.99944 | 1.00037 |
| 0.8 um | 8.84% | 0.99222 | 1.00028 |

With 3x3 pixel integration, 500 peak electrons, 2 background electrons and
1.5-electron RMS read noise, errors against the noise-free Cartesian reference
are 4.88% and 9.66%, respectively. Comparisons against an independently noisy
Cartesian image have larger errors (8.47% and 11.87%).

Errors/correlations use voxels above 1% of the reference peak, excluding the
large empty background; total-signal ratios use the complete compared volume.
Projection figures use shared intensity scales and include the projection of
the absolute volumetric error, rather than differences between independent
maximum projections.

The 23 unit/integration cases cover exact tube geometry and empty interior,
pixel-area quadrature, Poisson/read-noise moments, the two convolution routes,
physical sphere centering, deskew/reference agreement at two scan steps, and
the existing fine-grid PSF sampling-convergence checks. Arrays stay in memory
during tests. These validate this stationary forward model; they do not explain
motion or the residual tilt in the live acquisition.

## GC after the complete camera simulation

The required order is **fine object -> optical convolution -> oblique camera
sampling and pixel integration -> photon/read noise -> GC -> orthogonal
deskew**. Noiseless geometry-check data are rejected by the GC comparison.
Earlier exploratory GC runs on those noiseless data were withdrawn and are
not evidence for changing production code.

The comparison reads the existing `raw_electrons.tif`, subtracts the recorded
background, clips negative calibrated counts as in the simulation's camera
calibration, and runs the existing GC solver. It does not regenerate the image
or draw another noise realization. An SHA-256 of the input electron-data file
is saved with the results. Conversion back to density uses the original camera
gain, not a fitted normalization. For older exports without that gain recorded,
its exact definition is recalculated from the saved fine truth and PSF and
checked against the saved calibrated camera image. New exports record it.

The deconvolution PSF is sampled from the same fine Cartesian optical PSF at
the reconstruction spacing and includes camera-pixel integration. Native GC
uses the acquired scan step. Optional missing-plane GC uses the finer requested
step, preserves the original endpoints, and passes that step to unchanged
orthogonal deskew. The Cartesian microscope reference is also deconvolved with
its corresponding pixel-integrated PSF.

```powershell
uv run python -m opm_processing.imageprocessing.deconvolve_opm_simulation diagnostics/meridian_sphere_camera/scan_0.2um
uv run python -m opm_processing.imageprocessing.deconvolve_opm_simulation diagnostics/meridian_sphere_camera/scan_0.8um
uv run python -m opm_processing.imageprocessing.deconvolve_opm_simulation diagnostics/meridian_sphere_camera/scan_0.8um --scan-upsample 2
```

Outputs are in `decon_native` or `decon_upsample2` subdirectories and can be
rendered with the same plotting module. Unlike the optical-image comparisons
above, the following relative L2 errors include **all voxels** and compare with
the **unblurred ground truth**, including false fluorescence outside the tubes:

| Acquisition / reconstruction | Before deconvolution | After deconvolution |
| --- | ---: | ---: |
| OPM 0.2 / 0.2 um | 72.20% | 38.68% |
| OPM 0.8 / 0.8 um | 73.44% | 42.70% |
| OPM 0.8 / 0.4 um | 73.44% | 39.31% |
| OPM 1.2 / 0.4 um | 74.78% | 38.54% |
| Cartesian microscope | 72.60% | 39.35% |

The two 0.8 um runs use exactly the same input electron data. These are noisy,
finite-resolution reconstructions, not exact recovery of the binary tube
boundaries. No production deconvolution, PSF-generation or deskew changes were
needed for these comparisons.

The 1.2 um acquisition uses the same optics, 3x3 camera integration, 500 peak
electrons, 2 background electrons, 1.5-electron read noise and seed 42. Its 37
measured planes reconstruct to 109 planes at 0.4 um using factor 3. Reproduce it:

```powershell
uv run python -m opm_processing.imageprocessing.opm_simulation --output diagnostics/meridian_sphere_camera_1p2 --scan-steps 1.2 --camera-samples 3 --peak-electrons 500 --background-electrons 2 --read-noise 1.5 --seed 42
uv run python -m opm_processing.imageprocessing.deconvolve_opm_simulation diagnostics/meridian_sphere_camera_1p2/scan_1.2um --scan-upsample 3
```

Against the deconvolved Cartesian reference, this reconstruction has 14.15%
relative error and 0.98699 correlation on the reference mask, compared with
11.07% and 0.99126 for the 0.8 / 0.4 um reconstruction. Its slightly lower error
against unblurred truth in this single noise realization does not establish an
advantage for coarser acquisition.
