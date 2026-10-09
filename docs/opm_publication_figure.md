# Oblique sampling and reconstruction of a fluorescent meridian sphere

## Figure caption

**Oblique-plane acquisition and reconstruction at three scan spacings.**
**a**, Maximum-intensity projections of the Cartesian ground truth, the
unblurred object sampled in instrument coordinates, and simulated camera
volumes acquired at scan spacings of 0.4, 0.8, and 1.2 µm. Ground truth comprises
12 fluorescent meridian arcs with 1 µm tube diameter on a hollow sphere of
10 µm centerline diameter (11 µm outer diameter), defined on a 50 nm Cartesian
grid. The object and a vectorial optical PSF (NA 1.35, wavelength 637 nm) were
sampled into the 30° oblique instrument frame and convolved before camera
sampling. Fine instrument sampling was 0.1 µm along the scan axis and 57.5 nm
along both camera axes. Detector pixels of 115 nm were integrated using 3 × 3
subpixel quadrature. Detection included Poisson photon noise with a calibration
of 500 electrons at the peak of the fine Cartesian blurred object, 2 electrons
of background per pixel, and Gaussian read noise of 1.5 electrons RMS.
Camera data are shown after subtraction of the known background, clipping
negative values to zero, and conversion to the original fluorescence-density
units. The 0.8 and 1.2 µm acquisitions are displayed on a common 0.4 µm scan
grid, retaining the measured planes and inserting exact zeros in every
unmeasured plane. These zeros are display placeholders, not observations used
by deconvolution. The 0.4, 0.8 and 1.2 µm acquisitions contain 109, 55 and 37
measured planes, respectively, across the same 43.2 µm scan-center span.
**b**, Maximum-intensity projections of the independently simulated and
GC-deconvolved Cartesian microscope reference and the three OPM reconstructions.
The 0.4 µm acquisition uses native GC; the 0.8 and 1.2 µm acquisitions use the
experimental missing-plane GC implementation with factors 2 and 3,
respectively. All OPM reconstructions have 0.4 µm scan sampling before unchanged
orthogonal deskew. Final comparison volumes have isotropic 115 nm voxels.
All panels use the same linear fluorescence-density display range of 0–1;
values above 1 are displayed as white and retained unchanged in source data.
There is no individual image normalization or display interpolation. Coordinates and aspect ratios
are physical; fields of view differ between instrument and Cartesian panels.

Rows show XY, XZ and YZ maximum-intensity projections for Cartesian volumes.
Primed axes denote the corresponding raw-array views, with X′ = camera column,
Y′ = camera row, and Z′ = scan position. In physical Cartesian coordinates,
X = X′, Y = Z′ + Y′ cos(30°), and Z = Y′ sin(30°). Thus primed views are not
Cartesian projections of the raw data. Missing planes produce black gaps in
X′Z′ and Y′Z′; maximum projection along Z′ removes these gaps in X′Y′.

## Reproducibility and interpretation

The simulation uses the same per-pixel photon calibration at all scan steps;
it does not hold total photons per volume constant. Seed 42 is used for the
camera model and seed 43 for the Cartesian reference. Differently shaped
acquisitions have different noise realizations despite sharing the seed.
The reference volumes and their physical axes were checked to agree across
all three reconstruction exports. Each raw electron-data hash was checked
against the input recorded by its reconstruction.

The binary ground-truth tubes are intentionally blurred and sampled by the
microscope; deconvolution does not exactly recover their hard boundaries.
This figure represents a stationary, matched-PSF simulation and does not
establish the cause of the tilt in live experimental data.

Render the figure from the saved acquisitions:

```powershell
uv run --no-sync --with matplotlib python -m scripts.publication_opm_simulation --acquisitions diagnostics/meridian_sphere_camera_0p4/scan_0.4um diagnostics/meridian_sphere_camera/scan_0.8um diagnostics/meridian_sphere_camera_1p2/scan_1.2um --output diagnostics/publication_opm_sampling
```

The output contains the combined figure and separate forward/reconstruction
panels as PDF, SVG, 600-dpi PNG and LZW-compressed TIFF. PDF/SVG preserve vector
labels with embedded raster data. `projection_source_data.npz` contains every
displayed full projection and its physical axes; plots crop only the view,
not the volume used in the projection. `manifest.json` records source paths,
input hashes, physical grids, sampling masks and reconstruction metrics.
`camera_*_zero_filled.tif` stores the exact display volumes in scan/row/column
order, as specified by the manifest, rather than Cartesian Z/Y/X.

## Verification of camera-plane geometry

`plane_geometry.png` (also PDF/SVG) displays the same saved camera data twice:
in row/scan coordinates and with raw pixel cells placed at their Cartesian YZ
coordinates. No image intensities are interpolated and production deskew is
not involved in this diagnostic. Each frame at scan position s obeys
Z = tan(30°) (Y - s), so missing planes appear as diagonal gaps in the Cartesian
view. Horizontal gaps in the instrument view describe the same missing data.
The actual sampling function was also checked using the analytic Cartesian
field Y - Z cot(30°), which must equal s at every pixel in a camera frame.
Maximum errors were below 1e-15 µm for all three scan steps; results are saved
in `plane_geometry_check.json`. This checks the coordinate mapping, not every
optical assumption in the simulation.

```powershell
uv run --no-sync --with matplotlib python -m scripts.plot_opm_plane_geometry diagnostics/publication_opm_sampling
```

`fixed_objective_frames.png` (also PDF/SVG) makes a different check: the
Cartesian object is blurred first, its laboratory Y coordinates are shifted,
and it is sampled at a stationary oblique detector plane. Detector points are
held fixed for all five displayed frames. This route is compared with the
original convolution in the fine instrument frame, with camera integration
applied to both. Relative full-frame L2 differences are 1.74%, 1.96%, 2.13%,
1.96%, and 1.74% at scan positions -8, -4, 0, 4, and 8 µm. These calculations
share the same spatially invariant optical PSF assumption; they check two
discretizations of translation and sampling, not independent optical models.
The figure shows a sphere-outline schematic with a fixed blue detector plane,
the predicted individual frames, and the corresponding saved noisy frames.
Both image rows share display limits 0–0.5, with higher values displayed white.
No camera-frame panel is a projection through the stack.

```powershell
uv run --no-sync --with matplotlib python -m scripts.plot_fixed_objective_simulation diagnostics/meridian_sphere_camera_0p4/scan_0.4um --output diagnostics/publication_opm_sampling
```
