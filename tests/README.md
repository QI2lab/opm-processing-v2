# Test audit — 2026-09-11

The suite contains 153 collected cases: 39 unit tests and 114 integration tests.
CUDA execution is an additional requirement for 37 cases. Collection rejects
tests without exactly one `unit` or `integration` marker. These labels describe
the test boundary; the assertions below establish its value.

## What changed

- Removed seven standalone flag-introspection, dispatch-call, naming-table, and
  ROI orchestration tests. Their behavior is covered by processing actual
  fixtures, reopening outputs, and exercising CLI resume/overwrite decisions.
- Replaced the mocked 2D RLGC wrapper test with real GPU reconstruction of known
  emitters, using distracting noncentral PSF planes and both accepted image ranks.
- Constructed the FFT impulse response independently of production PSF padding.
  Both KLD implementations now use an independent normalized-probability oracle.
- Replaced exact RNG-sequence compatibility with binomial probabilities. The
  fractional split checks conservation, both means, and variance; deterministic
  halving and asymmetric treatment of fractions fail.
- Added absolute photon-flux and unscaled error checks to point reconstruction.
  Fitting an arbitrary intensity scale can no longer conceal doubled output.
- Replaced the permissive dim-emitter assertion with a paired-background
  experiment that requires concentration of signal at the known emitter and
  bounds total flux error. Whole-image and background error checks also guard
  against noise that paired subtraction can conceal. Its camera simulation now
  separates electron shot noise, read noise, and ADU quantization, as described
  below; the earlier model did not represent the acquisition camera.
- Added split-label symmetry checks in both stopping modes. Exchanging the two
  halves on alternate iterations must leave reconstruction unchanged; this
  exposed comparisons of KLD scores against different random data splits.
- Replaced nonzero/shape/file-existence-only output checks with calibrated pixels,
  cropped arrays, fused overlaps, projections, and reopened checkpoints.
- Removed progress-bar call assertions and fixed hash snapshots. Resume acceptance
  and rejection now exercise persisted configuration fingerprints behaviorally.
- Removed output-dependent masking from reconstruction correlation. Observable
  regions come from the physical acquisition geometry, and zero reconstructed
  values inside those regions remain in the score. Chunked deskew comparison
  no longer fits an arbitrary intensity scale.

## Retained coverage

Counts include parameterized cases.

| File | Unit | Integration | Numerical or state oracle |
| --- | ---: | ---: | --- |
| `test_acquisition_metadata.py` | 0 | 7 | Explicit metadata, coordinates, every timepoint's calibrated pixels, TIFF contents |
| `test_flatfield.py` | 8 | 6 | Detector calibration, clipping, occupancy decisions, independent illumination fields |
| `test_gpu_processing.py` | 9 | 19 | Direct convolution, relative entropy, SSIM, binomial statistics, known emitter and registration truth, split-label symmetry, three camera readout modes |
| `test_live_processing.py` | 5 | 13 | Chunk readiness, lifecycle transitions, exact saved channels/projections, interrupted processing |
| `test_opm_v2_dataset.py` | 0 | 7 | Projection calibration, distinct T/C data, solver-boundary result persistence, CLI resume and fusion |
| `test_opm_v2_tiled_reconstruction.py` | 0 | 8 | Forward-projected objects, geometry-selected correlation, shell widths, registration and fused overlaps |
| `test_opmtools.py` | 3 | 2 | Fractional photon density at three scan steps, chunk agreement, reconstructed shell widths |
| `test_process_roi_input.py` | 0 | 12 | Real cropped outputs and checkpoints, source resolution, rejection before output creation |
| `test_processing_state.py` | 0 | 8 | Durable JSON state, configuration compatibility, sibling isolation, failed atomic replacement |
| `test_roi.py` | 2 | 11 | Physical coordinates, selected layer pixels, ROI extents, exact variable-tile fusion |
| `test_roi_resume.py` | 0 | 14 | Interrupted writes/checkpoints, retained versus recomputed pixels, complete ROI fusion/projection |
| `test_tilefusion.py` | 12 | 7 | Independent weighted overlap arrays, zero contributors, physical placement, reopened pyramids |

Mocks remain where they isolate a boundary: selected GPU solves, known
registration displacements, BaSiC fitting, GUI layers, acquisition metadata,
polling, and injected write failures. Tests with a substituted solver verify
processing/storage integration; they do not establish deconvolution quality.
Real GPU reconstruction and real BaSiC recovery have separate integration tests.
Preservation tests for empty or smooth images intentionally check invariants;
the reconstruction tests supply the complementary requirement to improve signal.

Illumination recovery permits 5% relative field error through resampling and
residual calibration. Point reconstruction bounds total photon error at 10%.
The dim-emitter experiment permits 20% differential flux error and requires a
strict increase of signal in the emitter core. These are behavioral acceptance
bounds, not saved arrays generated by the implementation.

## Reconstruction investigation and validation

The initial audit exposed three dim-emitter failures and a combined Y/X/Z tiled
correlation failure: **141 passed, 4 failed**, recorded in
`diagnostics/audit-final.log`. The hosted-CI command was executed locally with
coverage enabled: **116 passed, 29 GPU cases deselected**, recorded in
`diagnostics/audit-ci.log`. The revised GitHub Actions matrix has not been run
remotely.

The accepted split-KLD correction evaluates the previous and current estimates
against the same newly drawn halves. Previously, merely swapping split labels
on alternate iterations changed the result by up to 179 photons per pixel in
the safe-mode fixture. Both modes now have numerical label-symmetry checks.
Keeping the original normalized-backprojection initialization, this correction
reduces bright-point unscaled MSE from 779.47 to 324.55 (input MSE 1114.67).

The proposed observed-image initialization was **rejected and reverted**. It
passed the previous 147 tests by concentrating the dim emitter, but paired
background subtraction concealed retained noise. Adding whole-image MSE and
background-only MSE assertions detects this failure for all three seeds.
The expected object is the known emitter plus constant background 1.5; the
reference is not the sampled noisy background and no intensity scale is fitted.

Controlled checks held the PSF, data, split RNG, stopping criteria, and update
factor fixed. They compared four mathematically defined starts, with a fifth
run using the original initialization and original stopping as the control:

| Initial estimate (corrected stopping) | Dim whole-image MSE, three seeds | Core gain, three seeds | Differential flux ratio, three seeds |
| --- | --- | --- | --- |
| Normalized backprojection, retained | 0.362 / 0.359 / 0.363 | 0.577 / 0.753 / 0.538 | 1.020 / 1.268 / 1.023 |
| Reference constant image mean | 0.358 / 0.357 / 0.360 | 0.989 / 0.743 / 0.328 | 1.772 / 1.979 / 1.104 |
| Observed image | 0.698 / 0.716 / 0.706 | 1.000 / 1.000 / 1.000 | 0.996 / 1.000 / 1.001 |
| One ordinary RL step from observed image, rejected | 0.703 / 0.722 / 0.713 | 1.237 / 1.215 / 1.210 | 0.996 / 0.994 / 0.991 |

Seeds are 7, 31, and 83. Input MSE is 0.698 / 0.715 / 0.706. No candidate
satisfies both strict emitter concentration and whole-image error improvement,
along with the existing 20% differential flux bound. The normalized adjoint
is one full-data RL step from a positive constant for a normalized PSF. The
constant mean is the initialization in the
[reference implementation](https://github.com/jdmanton/rlgc/blob/master/rlgc.py#L92).
No smoothing radius, mixture weight, extra iteration count, stopping threshold,
or factor of one half was tuned to pass these checks.

Reproducible controlled-check code and measurements are in
`diagnostics/check_rlgc_initial_conditions.py` and
`diagnostics/rlgc_initial_conditions/metrics.json`. The test run demonstrating
that all three noisy-start cases fail the new MSE assertion is
`diagnostics/initial-noisy-error-tests.log`. Full-suite verification after the
rollback finished with **143 passed, 4 failed** in 128.18 seconds, recorded in
`diagnostics/initial-checked-final.log`. The three dim-emitter cases pass the new
whole-image and background error assertions but still fail concentration. The
combined Y/X/Z tiled case still has minimum correlation 0.518 versus 0.55.
Both stopping-symmetry cases pass. No acceptance bound was relaxed or failure
hidden with `skip` or `xfail`; the low-photon recovery problem remains open.

### Combined tiled reconstruction: stopping-default correction

The combined tiled failure was isolated before registration and fusion. Full
volume and scan-chunk reconstruction gave nearly the same failing score, so
neither chunk seams nor stage registration accounted for the loss. Comparing
the working tree against HEAD revealed that `chunked_rlgc`'s default
`max_delta` had been increased from 0.01 to 0.1, inconsistent with its committed
value, its docstring, and the base solver. Those runs stopped after only four
or five iterations.

Restoring the committed **0.01** default is a one-line production change. With
the same PSF, seeds, initial estimate, halo, deskew implementation, and 0.55
acceptance threshold, the controlled comparison gives:

| Tile | Correlation at 0.1 | Correlation at 0.01 | Skewed-space MSE at 0.1 | Skewed-space MSE at 0.01 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.569 | 0.645 | 919,976 | 610,902 |
| 1 | 0.518 | 0.611 | 1,177,652 | 808,258 |
| 2 | 0.566 | 0.643 | 1,252,807 | 826,525 |
| 3 | 0.534 | 0.623 | 1,314,814 | 891,803 |

MSE is against the known ideal camera-grid object, before deskewing, without
fitting a gain or excluding zero reconstructed values. The result supports the
restored stopping criterion independently of the lab-coordinate correlation.
More iterations are expected with this restored default. The separate
deskew-weighting experiment remains diagnostic only; no deskew source or
deskew acceptance tests were changed for this fix.

Controlled measurements are in
`diagnostics/combined_tiled_probe/stopping_defaults.json`; the eight complete
tiled pipeline checks all passed in 50.39 seconds, logged in
`diagnostics/combined-default-fix-tests.log`. All remaining tests were then run
without repeating that file: **136 passed, 3 failed** in 86.33 seconds, logged
in `diagnostics/combined-fix-other-tests.log`. Across the complete 147-case
suite, **144 pass and only the three existing dim-emitter cases fail**. Ruff,
formatting, and `git diff --check` also pass. The combined tiled failure is
resolved without changing deskew, initialization, or the acceptance criteria.

### Dim emitters: camera-model correction

The remaining three failures used `0.24 * Poisson(1.5 / 0.24)` as background,
but unit-electron Poisson noise for the added emitter. This gives background
variance 0.36 at mean 1.5 and omits read noise. The 0.24 conversion is electrons
per ADU, not the size of an independently arriving photoelectron.

The [Hamamatsu technical note, page 15](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/sys/SCAS0138E_C14440-20UP_tec.pdf)
specifies 0.24 electrons/ADU, 100 ADU offset, and read-noise RMS of 0.7, 1.0,
and 1.6 electrons for ultra-quiet, standard, and fast scan, respectively. The
saved acquisition summary reports C15440-20UP, conversion 0.2400, offset
100.0000, `ScanMode=3`, and a 4.9 microsecond line interval, consistent with
fast scan. These are manufacturer specifications and saved settings, not a
measurement of the individual camera's per-pixel noise.

The corrected test generates Poisson electrons, adds Gaussian read noise,
digitizes to uint16 ADU, and applies the same offset/gain/nonnegative clipping
as the processing pipeline. It covers all three documented readout modes and
the original seeds 7, 31, and 83. The 100-electron emitter and PSF are unchanged.
Concentration is measured relative to the added signal actually present in
the calibrated image; total flux is still required to stay within 20% of the
underlying injected electron count. Whole-image and background-only MSE must
improve against the known expected object, without an intensity-scale fit.

For fast scan, measured core gains are **1.229 / 1.764 / 1.149**, reconstructed
total-electron ratios **0.834 / 0.887 / 0.875**, and whole-image MSE
**1.317 / 1.312 / 1.508**, versus **3.164 / 3.200 / 3.201** for the calibrated
observations. A negative control substituted the discarded observed-image RL
initialization in memory: all three fast-mode cases failed the MSE assertion
(3.856 / 4.485 / 4.581), recorded in
`diagnostics/dim-camera-negative-control.log`.

**This is a correction to the test's camera model, not a new solver fix.**
The original quieter-than-Poisson stress input still fails concentration; its
measurements remain in `diagnostics/dim_consensus/metrics.json`. Lowering
`limit`, changing KLD direction, or modifying consensus padding did not satisfy
its joint concentration/flux requirements and were not applied. Production
initialization, deskew, splitting, and `max_delta=0.01` are unchanged in this
step. The weighted fractional split remains an approximation for analog
camera measurements, rather than an exact independent Poisson split.

After this test correction, the complete suite finished with **153 passed**
in 141.86 seconds (`diagnostics/dim-camera-full-tests.log`), including all 37
CUDA cases. No additional production solver change was made. This passing
result applies to the revised camera model and does not resolve the quieter
synthetic stress case or establish satisfactory real-image noise behavior.

### Real-data limits

Cached calibrated volumes 152?155 and 223 were compared at `crop_scan=13`,
`limit=0.1`, `max_delta=0.1`, seed 42. The rejected observed-image initialization
made the sampled dim feature at time 223 stronger and narrower, but increased
adjacent-X background difference standard deviation from 0.126 to 0.375
(deskew only: 0.310). It reduced reblurred-output MSE by 3.4?4.1% across the five
volumes, but those noisy observations are not object ground truth.

Artifacts in `diagnostics/rlgc_failure_fix/`, including the time-223 OME-TIFF and
comparison image, document that **rejected initialization**, not the currently
retained solver. No acquisition or existing processed store was overwritten.
Existing ROI checkpoints do not detect solver source changes; use a fresh
output or `--no-resume` when rerunning an existing ROI.

A fresh accepted-solver time-223 OME-TIFF and comparison are exported under
`diagnostics/dim_consensus/accepted_t223*`. This run takes 63 iterations and
visibly amplifies background grain at the retained `max_delta=0.01`; its
"accepted" filename identifies the current code, not image-quality approval.
Known emitters were also injected
into the cached real background at expected totals of 30, 60, 120, and 300
electrons. Some individual weak-emitter realizations still lose core signal;
these measurements do not establish universal recovery of real dim features.
See `diagnostics/dim_consensus/real_injected_emitters.json`. The camera-based
synthetic test is a simplified model, without per-pixel read-noise maps.

A shifted-Poisson read-noise approximation was tried only in a diagnostic
module, adding the published read-noise variance to observations and model
predictions. At fast-mode variance 2.56, synthetic MSE improved to
0.766 / 0.760 / 0.829, but measured core gains were 0.995 / 1.736 / 0.704.
It therefore fails the joint recovery requirement and was not adopted.
Preserving signed calibrated measurements before that correction also failed
the joint core/flux criteria. Neither experiment changed camera calibration,
initialization, splitting, deskew, or stopping defaults in production.
These results are recorded in `diagnostics/dim_consensus/read_noise_offset_metrics.json`
and `diagnostics/dim_consensus/signed_noise_offset_metrics.json`.

Eight reversible, in-memory fault probes were all detected: identity RLGC,
doubled RLGC gain, shifted PSF centering, deterministic half splitting, biased
fractional splitting, zero KLD, identity 2D RLGC, and integer conversion without
clipping. Local probe results are in `diagnostics/audit_mutations.json`.
This is targeted fault injection, not a claim of exhaustive mutation coverage.

### Controlled initialization trials

The follow-up trial changes only the initial estimate in an in-memory solver
copy. `max_delta=0.01`, `limit=0.1`, safe mode, PSF, calibration, splitting,
padding, deskew, and seed assignments stay fixed. The four estimates are the
current normalized backprojection, a uniform measured-mean image, the measured
image with a 1e-6 floor to avoid multiplicative zero locking, and the current
backprojection followed by one additional ordinary full-data RL step.

The diagnostic applies the existing camera test's object-MSE, background-MSE,
core-concentration, and total-flux criteria to all nine noise-mode/seed cases.
It also retains the three legacy sub-Poisson stress cases without relaxing
their criteria. These are controlled numerical experiments, not a rerun of
the complete pytest suite.

| Initialization | Camera cases satisfying all criteria | Fast-mode mean object MSE | Time-223 background difference SD |
| --- | ---: | ---: | ---: |
| Current backprojection | 9/9 | 1.379 | 0.646 |
| Uniform measured mean | 8/9 | 1.317 | 0.653 |
| Measured image | 0/9 | 4.397 | 0.771 |
| Backprojection plus one RL step | 7/9 | 1.380 | 0.631 |

The background statistic is adjacent-X difference standard deviation in the
same deskewed ROI, Z 20:40, Y 100:150, X 600:900. Deskew-only time 223 is 0.310.
It measures local roughness, not object reconstruction error. All four starts
fail the legacy stress cases and give a negative paired core response for the
same weakest 30-electron injection realization on the real background.
Reconstructed images themselves remain finite and nonnegative; the negative
quantity is the signal-added reconstruction minus the background reconstruction.
The extra RL step's small noise reduction does not establish a dim-emitter fix.
No initialization change was adopted.

Scripts, a solver-source snapshot, per-iteration logs, numerical measurements,
and time-223 fixed-scale comparisons are in
`diagnostics/initialization_trials/`, with the drivers at
`diagnostics/trial_initializations.py` and
`diagnostics/summarize_initialization_trials.py`. Real-image trials include
volumes 152-155 and 223, with local OME-TIFF exports for each initialization.

## Running the suite

Use Python 3.12 and the project's development dependencies. CUDA-marked tests
must run on a configured NVIDIA workstation; `OPM_REQUIRE_GPU=1` turns missing
GPU support into failure. Hosted CI runs the remaining tests on Linux and
Windows. Its dependency installation uses the actual `dev` dependency group;
the GPU extra supplies imports used by substituted GPU boundaries even on
hosts without CUDA hardware. Hosted CI does not validate GPU correctness.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:OPM_REQUIRE_GPU = '1'
$env:CUPY_CACHE_DIR = Join-Path (Get-Location) 'diagnostics/cupy-cache'
.\.venv\Scripts\python.exe -m pytest --strict-markers -q -p no:cacheprovider
```

Use `-m unit`, `-m integration`, or `-m 'not gpu'` to select a boundary.
On this workstation, the default pytest temporary directory has an ACL problem;
the audit uses fresh `--basetemp=diagnostics/<run-name>` directories. Pytest
clears an existing explicit base directory, so choose a dedicated test path.
Validation reprocessed cached copies of five volumes, without overwriting the
G: acquisition. These tests cannot certify every real low-photon acquisition
or an unavailable GPU.
