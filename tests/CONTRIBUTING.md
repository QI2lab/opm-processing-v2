# Test conventions

Only unit and integration tests are permitted. Classify every test with exactly
one of those markers and put it in `tests/cpu` or `tests/gpu`; collection supplies
the hardware marker. Do not add smoke, API, regression, snapshot, benchmark or
argument-forwarding tests. Cover core numerical processing, scientific accuracy
and the acquisition/storage lifecycle. Figure generation, display/video rendering,
coordinate-repair utilities, legacy adapters and PTV are outside this suite.

Use shared fixtures in `tests/fixtures` for data, acquisition settings, optical
models, storage and cleanup. Reuse those fixtures and parameterize settings rather
than adding parallel workflow builders. Keep mutable data function-scoped and
cache expensive immutable optics and fixed camera realizations, keyed by their
complete physical settings.
Call production functions directly; do not add small pass-through helpers,
single-use wrappers or generalized test frameworks. Keep tests and NumPy-style
docstrings concise. Do not import helpers from test modules or `conftest.py`.
Keep workflow tests focused on their interaction: durable pixels, coordinates,
selection, resume and storage. Put detailed numerical acceptance in the shared
core processing cases. Do not replay the processing pipeline or rebuild
camera-space envelopes in every workflow to check the same operation again.

Unit tests use known analytical objects, independently blurred scenes or explicitly
prescribed metadata and decision states. Use actual library operators and installed
packages, not copies of processing algorithms. Analytical geometry, camera
probabilities and physical truth belong in the shared simulation/measurement code.
Check those independent references against analytical results or installed
operators before relying on them for accuracy assertions.

Every integration must execute the complete chain:

1. Construct a known fluorescent object in physical coordinates.
2. Simulate optics, excitation before emission blur, camera-pixel integration,
   Poisson photoelectrons, calibrated read noise and unsigned ADC digitization.
3. Write camera frames through the installed opm-v2 `OpmDataHandler` lifecycle
   and its real ome-writers TensorStore backend, using
   `opm_v2_acquisition_factory`.
4. Read the written metadata and run actual processing functions or registered
   commands with their real parsers. Deconvolution must generate its production
   optical kernels inside processing; other paths do not generate unused inverse
   kernels merely for test setup.
5. Reopen final storage and assert pixels, coordinates, completion and accuracy
   against the original simulated truth.

Hand-built acquisition lookalikes, injected processing calibration attributes,
cached solver outputs and measured-PSF options cannot replace integration stages.
Measured-PSF loading, explicit optical-parameter calibration, metadata-only
operations, GUI selection and format serialization belong in units. Missing
producer dependencies are errors. Calibrated-input workflows may read detector
fields saved with the production writer from the simulated detector response;
calibration-estimation integrations must estimate fields from independently
simulated calibration frames.

Use actual yaozarrs/TensorStore writers and readers. Native I/O needs pytest
temporary directories; reuse filesystem mocks only where Python can intercept
the operations. Zarr-python is allowed for attribute edits and napari viewing,
not pixel processing/storage. Do not invent arrays, futures, codecs, optimizers
or package results. Observers execute the real operation. Mock only necessary
filesystem/UI interaction and inject faults only in error-handling tests.

Scientific assertions must not fit reconstruction gain or shifts, select regions
from reconstructed intensities, substitute production output for truth, or relax
accuracy to match a failure. Use independent camera realizations, fixed physical
source/background apertures and camera-space recovery/noise measurements.
Simulator-truth likelihood evaluates accuracy and never selects production stopping.
RLGC retains its held-out acquired-prediction KL stopping rule. Resolution uses
directional PSF first minima (Rayleigh); tolerances follow precision, sampling or stated error
bounds. Do not use raw camera-count variance as an acceptance distribution for
nonlinear, correlated reconstruction estimates. Geometry and photon-volume
checks use the supported 30-degree convention.

Core processing integrations check raw inputs, generated PSFs, haloed sequential
scan chunks, deskew, registration, fusion and reopened outputs. Check bounded chunk
state and partial last tiles where those behaviors are under test. Sources must
cross multiple real tile borders with foreground on both sides. Comparing two
processing layouts alone is not scientific accuracy. Scan chunks have their own
initialization and random halves; default tiled RLGC continues them to the latest
first-local-KL accepted iteration without replay. Test real allocation fallback
with CUDA memory-pool limits, not fabricated out-of-memory exceptions.

`tests/gpu/conftest.py` scopes the early hardware gate and module cleanup to CUDA tests.
RLGC and CuPy/cuCIM pair-registration tests belong in `tests/gpu`; CPU workflows
cover deskew, geometry, cached placement and CPU fusion. RLGC quality evaluations
use the shared noisy line-pair and sphere scenes, with native and sparse scan sampling in whole-volume
and tiled layouts.

Large-scene units and writer-backed integrations share the same physical camera
simulation. Require foreground across multiple observed tile borders and verify
actual Ryomen execution, shifted final deskew crops and callback continuation
without replay. Check dark-noise variance in known optical vacuum, excluding
spatially varying fluorescent background. Noisy deskew samples use bounds from
the original local camera samples. Reconstruction checks use independent optical truth,
KL recovery and measured noise behavior, without unvalidated six-sigma photon
totals or chi-square noise-floor gates.

Physical raw-camera fixtures use `(scan, tilt row, orthogonal column)` order:
native 0.4-µm scans have `(50, 256, 1900)` samples for whole-volume cases and
`(200, 256, 1900)` for tiled cases. Coarser scans select the same physical
exposures with fewer planes; preserve the original sensor coordinates and stage
acquisition direction instead of re-centering the smaller arrays. Analytical
operator and probability references may use the small arrays their equations
require. Physical accuracy apertures need PSF margins in the actual observed
camera or ROI, with no exception to accommodate an undersized fixture.

Extend existing acquisition workflows for interruption/resume, moves and copies,
separate raw/output directories, cached registration, numerical projections and
ROI/TIFF export. Preserve metadata and journals; assert current paths, pixels,
physical units and completion. Keep filesystem and execution assertions local to
the workflow that exercises them instead of installing global observers.

Install the real producer without microscope-control dependencies:

```bash
uv sync --group dev
uv pip install --no-deps "opm-v2 @ git+https://github.com/QI2lab/opm-v2@ae059d962835520bf2c1b37edbb335bf5e617b23"
```

Update producer and writer pins together. Hosted CI runs the explicit CPU core
and native-writer selection in `.github/workflows/ci.yml`. BaSiC fitting, RLGC
and pair registration are GPU-only; cached-field correction and metadata checks
remain on CPU. The manual Science validation workflow provides `cpu-science`,
`gpu-science`, `full-cpu`, and `full-gpu` profiles.
Supply the JSON label or labels of an existing runner with qualified RAM and
scratch capacity; standard hosted capacity has not been validated for science
profiles. GPU profiles require the qualified Python/CUDA/cuCIM environment on
the runner's PATH, including test/producer dependencies and the Windows cuCIM
setup where applicable. They install the current checkout editable without
replacing CUDA libraries. CUDA tests skip when CuPy or a CUDA device is unavailable;
BaSiC fitting also requires PyTorch CUDA and skips before expensive fixture setup
when it is unavailable. Numerical and allocation errors remain failures.
Science profiles serialize to
avoid sharing a GPU concurrently.
Both workflows retain JUnit and coverage XML as run artifacts, including failures.
Relevant CUDA changes require a real GPU validation result before merge.
Report CPU/CUDA results separately and honor session restrictions on reruns.
Do not hide numerical failures with skip/xfail. Runtime/memory measurements and local
investigation reports belong in ignored `diagnostics/` or `.tmp/`, outside
pytest and published documentation.
