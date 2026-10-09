# Undersampled RL experiment

The experimental function is in `src/opm_processing/imageprocessing/rlgc_undersampled.py`.
It reuses the existing GPU convolution, PSF padding, count splitting, gradient
consensus and KLD helpers. This experiment processes a single volume in GPU memory.

## CLI

```powershell
uv run process "F:\path\to\acquisition" --deconvolve --decon-scan-upsample 4 --output "F:\path\to\experimental-output"
```

`--decon-scan-upsample N` explicitly selects the experimental solver. For a
0.8 um acquisition, `N=4` reconstructs at 0.2 um. The same finer spacing is used
for PSF generation, output allocation and deskewing. Choose an integer factor
for the desired spacing; this is not enabled automatically above 0.4 um.
`--deconvolve` without the new flag continues to use the existing solver.

The flag requires `--deconvolve` and a 3D mirror or stage acquisition. The first
version runs each channel/tile volume entirely on the GPU: scan chunking options
(`--decon-crop-scan`, `--decon-fallback-step-scan`) and ROI processing are not
supported. `--decon-psf-paths` accepts PSFs already sampled at the finer step.
Use `--save-float32` to preserve fractional output intensities.

Use a separate `--output` directory to retain a conventional reconstruction for
comparison. Output names follow the existing deconvolution convention. `--resume`
checks the solver/factor/geometry fingerprint. Reconstruction geometry is stored
in the processing sidecar, leaving acquisition metadata unchanged; `fuse` uses
that geometry for tile support masks. For example:

```powershell
uv run fuse "F:\path\to\experimental-output"
```

## Python

```python
from opm_processing.imageprocessing.opmpsf import generate_skewed_psf
from opm_processing.imageprocessing.rlgc_undersampled import rlgc_undersampled

acquired_step_um = 0.8
factor = 4
fine_step_um = acquired_step_um / factor
psf = generate_skewed_psf(
    em_wvl=0.610,
    pixel_size_um=0.115,
    scan_axis_step_um=fine_step_um,
    theta_deg=30,
)
reconstruction = rlgc_undersampled(
    measured_volume,
    psf,
    scan_upsample_factor=factor,
    gradient_consensus=True,
    max_iterations=100,
)
```

Input and output axes are **scan, camera Y, camera X**. The output scan step
is `fine_step_um`; it has `(N - 1) * factor + 1` planes. The first and last
acquired plane positions are preserved. Any subsequent deskew must use this
fine step. The PSF generator is unchanged; its scan-step argument selects the
reconstruction sampling. A PSF generated at the coarse acquisition spacing
cannot describe convolution on the finer grid.

For controlled ordinary-RL comparisons, set `gradient_consensus=False` and
`max_delta=0`, and use the same `max_iterations` for both acquisitions. GC mode
uses measured-plane sensitivity and observation-to-prediction KLD rollback
against the same fresh halves. The previous prediction is reused without an
extra forward FFT. All fine voxels enter convergence; defaults are limit 0.001,
max_delta 0.001 and at most 100 updates, matching the native convergence defaults.
Both solvers retain the RLGC multiplicative update and consensus gate. Their
sampling operators and boundary conditions differ.

The forward model is `H = S C`: convolution at fine sampling followed by
selection of every `factor`th scan plane. Its transpose scatters measurements
into zeros and convolves with the flipped PSF. The multiplicative update is
`u *= H.T(data / H(u)) / H.T(ones)`. The denominator compensates for the missing
observations. Missing planes are never treated as zero-valued measurements.

This experiment assumes zero fluorescence outside the field of view, giving
an exact finite-volume forward/adjoint pair. Production RLGC uses a reflected
boundary constraint, so edge behavior is deliberately different. No background,
read-noise, illumination or scattering model has been added. Counts must be
nonnegative. Integer photon counts retain the existing binomial split model;
fractional corrected counts use the existing approximate split.

## Physical validation

Run on a CUDA GPU from PowerShell:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:OPM_REQUIRE_GPU='1'
uv run python -m pytest tests/test_rlgc_undersampled.py -q -s
```

The tests construct 380 nm diameter fluorescent spheres and a 220 nm diameter
finite filament in laboratory coordinates, integrating each skew-grid voxel
with 27 subvoxel samples. They use the existing vectorial PSF for a 1.35 NA
silicone objective, 610 nm emission, immersion RI 1.40, sample RI 1.38, 115 nm
camera pixels and a 30 degree OPM angle, including the production generator's
existing lateral apodization. For test runtime only, centered PSF
tails containing at most 0.15% of its energy are omitted and the PSF is
renormalized; there is no Gaussian substitute.

Independent CPU linear convolution simulates the full 0.2 um acquisition.
Selecting every third or fourth plane simulates 0.6 or 0.8 um acquisition at
the same exposure per plane. This is the physical blur-then-sample operation:
convolving a decimated specimen would incorrectly discard fluorophores between
planes. Separate cases use noise-free photon expectations and Poisson photon
counts. The full and sparse noisy data share acquired-plane realizations to
isolate the effect of omitting planes; the sparse acquisition has fewer total
photons, without compensating gain.

The main full/sparse accuracy test disables gradient consensus and runs 60
ordinary RL updates. Its results establish sampling performance for that RL
operator; they do not establish native reference RLGC acceptance.
Both acquisitions are reconstructed on the same fine grid and compared with
the known specimen. A third baseline performs ordinary RL on the coarse grid
using the same optical PSF sampled at that spacing, then interpolates its result.
Assertions cover absolute reconstruction error without
fitted gain or translation, missing-plane error, reblurred predictions at both
measured and withheld planes, total fluorescence, and bead centroids in microns.
Each case prints its physical error metrics with pytest's `-s` option.
Numerical unit tests operate on arrays. Integration tests write simulated
camera data and acquisition metadata to temporary stores, process those real
files, and reopen the saved output for physical comparisons.
Further tests exercise GC with photon noise, check the optical forward model
and adjoint against independent CPU convolution (including a displaced optical
origin), and compare several RL iterations with a CPU reference at factors
1, 3 and 4.

The combined-channel CLI integration test simulates fluorescence at 637 nm,
Poisson detection and camera calibration for a 0.8 um acquisition reconstructed
at 0.4 um. It runs the real CLI, optical model, GPU reconstruction and deskewing
for mirror and stage scans, with real image reads and writes. Assertions compare against
the known specimen, withheld-plane photon expectations and bead coordinates
in microns. The integration test observes the real solver and uses a bounded
optical PSF to keep runtime manageable; neither image storage nor reconstruction
is replaced. Tests do not use API dispatch as an acceptance criterion.

These matched-PSF simulations validate the specified model and quantify its
sampling loss. They do not demonstrate recovery of arbitrary missing spatial
frequencies or robustness to PSF mismatch. Scan chunking remains outside this
experiment.

## Interpretation of simulation results

The physical tests compare reconstruction against known synthetic specimens
under a matched, shift-invariant optical model. Passing those tests does not
establish that this model explains the live acquisition or any residual tilt.

The previously reported benchmark table used an experimental axial PSF spacing
change and has been withdrawn with that change. Production orthogonal deskew
and PSF generation remain unchanged. See the [tilt investigation audit](deskew_psf_alignment.md)
for the metadata checks and limitations of the earlier point-source simulation.
