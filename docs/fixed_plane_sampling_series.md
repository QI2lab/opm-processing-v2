# Fixed-plane acquisition and reconstruction at four scan spacings

The figures are saved in `diagnostics/fixed_plane_sampling_series` as PDF, SVG
and 600-dpi PNG, with frame arrays in NPZ files and acquisition/reconstruction
provenance in `manifest.json`.

## Acquisition figures

`acquisition_0.4um`, `acquisition_0.8um`, `acquisition_1.2um`, and
`acquisition_1.6um` each show seven acquired frames across the sphere's passage
through a stationary oblique plane. Columns select the nearest acquired frame
to scan positions -10.4, -8, -4, 0, 4, 8, and 10.4 µm, with actual object displacements
explicitly labeled. Thus these widely separated snapshots are not consecutive
camera frames, and the selected positions vary slightly with acquisition step.
The camera display spans -8 to +8 µm in columns and -16 to +16 µm in rows;
the small margin outside the saved detector support is displayed black.

The top row illustrates the sphere's lateral translation and the fixed blue
detection plane in laboratory YZ coordinates. The middle row independently
predicts each frame by translating the Cartesian blurred object past that
plane and integrating camera pixels. The bottom row shows the existing noisy
camera acquisition. A scan position s corresponds to object displacement -s
in the fixed-objective view. Camera images use a shared linear display range
of 0–0.5 fluorescence-density units; larger values display white.

`consecutive_acquisitions` compares all four acquisitions at the same nine
consecutive 0.4 µm positions (-1.6 through +1.6 µm). Missing observations are
shown as exact zeros, without text overlays on the images. These zeros are display
placeholders and are not treated as measured zeros by deconvolution.

## Reconstruction figures

`reconstruction_0.4_to_0.4um`, `reconstruction_0.8_to_0.4um`,
`reconstruction_1.2_to_0.4um`, and `reconstruction_1.6_to_0.4um` use the same
seven physical scan positions (-10.4, -8, -4, 0, 4, 8, and 10.4 µm) for all acquisitions.
The four rows are:

1. Sphere displacement relative to the stationary detector plane.
2. The GC-deconvolved Cartesian microscope reference, sampled onto that plane.
3. OPM GC reconstruction in instrument coordinates, labeled according to
   whether a camera frame was acquired at that scan position.
4. The final orthogonally deskewed OPM reconstruction, re-sliced onto the same
   oblique plane to permit a direct frame comparison.

Rows 2 and 4 are trilinear oblique slices of Cartesian volumes, not raw camera
measurements or Cartesian projections. No extra optical convolution, camera
integration, or noise is added to these reconstructed object estimates. The
final re-slicing is a visualization operation; it does not change the saved
reconstruction. All reconstruction images share a 0–1 linear display range;
larger values display white. Native GC is used for 0.4 µm, and missing-plane
GC uses factors 2, 3, and 4 for the other acquisitions. Every reconstruction
has a 0.4 µm scan step before unchanged orthogonal deskew.

`consecutive_reconstructions` shows the corresponding instrument-coordinate
GC estimates at the same nine positions as `consecutive_acquisitions`.
"Measured" identifies a location with an acquired camera frame, while
"inferred" identifies a location without one; all displayed values are
deconvolved estimates, including those at measured locations.

## Simulation conditions and limits

The stationary ground truth has 12 meridian arcs on a 10 µm centerline-diameter
sphere, 1 µm tube diameter and 50 nm Cartesian sampling. The model uses a
30° oblique plane, NA 1.35, wavelength 637 nm, 115 nm camera pixels and 3 × 3
pixel-area quadrature. Camera noise is Poisson plus 1.5-electron RMS read
noise, with 2 background electrons per pixel and a calibration of 500 electrons
at the fine Cartesian blurred-object peak. Photon calibration per pixel is
fixed across acquisition spacings; total photons per volume are not fixed.
Camera seed 42 and Cartesian-reference seed 43 are reused, but differing
array shapes give different realizations.

The first three acquisitions cover scan centers from -21.6 to +21.6 µm;
1.6 µm acquisition rounds the padded bounds outward to -22.4 to +22.4 µm.
This preserves a centered object and complete optical support, yielding
29 measured and 113 reconstructed planes for 1.6 µm. The other acquisitions
have 109, 55 and 37 measured planes and 109 reconstructed planes. The shifted
output-grid phase means the 1.6 µm Cartesian reference is sampled on its own
recorded Cartesian axes; frame comparisons use physical coordinates for
every dataset, without registration or a fitted intensity normalization.

Input electron-data hashes are checked against the hashes recorded by each
reconstruction. All reconstructed arrays must be finite, selected planes must
land on the recorded 0.4 µm reconstruction grid, and physical settings must
agree across cases. The simulations share a spatially invariant optical PSF
assumption and do not establish the cause of artifacts in moving live data.
No production PSF, GC or orthogonal-deskew code is changed for these figures.

## Acquisition-to-reconstruction Cartesian maximum projections

`acquisition_reconstruction_mips` is a separate figure with one row per
acquisition spacing and paired acquisition/reconstruction columns for XY,
XZ and YZ. These are true Cartesian maximum-intensity projections through a
common object-containing box, with voxel centers restricted to ±7.8 µm in
each Cartesian direction, shown within ±8 µm axes. They are not individual
frames or projections over raw array indices.

Each acquisition is first placed on a 0.4 µm scan grid, with exact zeros in
the unmeasured planes. For display only, its raw cells are assigned to 50 nm
Cartesian grid points by nearest neighbor using the inverse physical map
row = Z/sin(theta), scan = Y - Z*cot(theta), column = X. This coordinate
conversion does not interpolate fluorescence into missing planes, estimate
missing data, or call production deskew. The fine Cartesian display grid
does not imply finer acquired resolution. Maximum projections are then taken
along physical X, Y or Z. Because the missing planes are oblique and parallel
to X, their gaps remain diagonal in YZ; projection along Y or Z can cover
gaps with signal from other planes.

The reconstruction panels use the saved final orthogonally deskewed GC
volumes at their recorded Cartesian sampling, cropped to the same physical
box. All panels use a common linear display range 0–1; values above 1 appear
white and remain unchanged in exported source data. The acquisition panels
retain optical blur and noise; reconstruction panels show deconvolved object
estimates. The input hashes, measured/zero-plane counts, physical display
grids and projection arrays are saved alongside the figure.

```powershell
uv run --no-sync --with matplotlib python -m opm_processing.imageprocessing.plot_acquisition_reconstruction_projections --acquisitions diagnostics/meridian_sphere_camera_0p4/scan_0.4um diagnostics/meridian_sphere_camera/scan_0.8um diagnostics/meridian_sphere_camera_1p2/scan_1.2um diagnostics/meridian_sphere_camera_1p6/scan_1.6um --output diagnostics/fixed_plane_sampling_series
```

## Reproduce the figures

```powershell
uv run --no-sync --with matplotlib python -m opm_processing.imageprocessing.publication_fixed_plane_series --acquisitions diagnostics/meridian_sphere_camera_0p4/scan_0.4um diagnostics/meridian_sphere_camera/scan_0.8um diagnostics/meridian_sphere_camera_1p2/scan_1.2um diagnostics/meridian_sphere_camera_1p6/scan_1.6um --output diagnostics/fixed_plane_sampling_series
```
