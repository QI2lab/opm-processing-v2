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
cache only expensive immutable optics, keyed by their complete physical settings.
Call production functions directly; do not add small pass-through helpers,
single-use wrappers or generalized test frameworks. Keep tests and NumPy-style
docstrings concise. Do not import helpers from test modules or `conftest.py`.

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
source/background apertures and absolute photon/camera likelihood. Likelihood is
an accuracy measure, not a held-out stopping rule. Resolution uses directional
PSF first minima (Rayleigh); tolerances follow precision, sampling or stated error
bounds. Geometry and photon-volume checks use the supported 30-degree convention.

Core processing integrations check raw inputs, generated PSFs, haloed sequential
scan chunks, deskew, registration, fusion and reopened outputs. Check bounded chunk
state and partial last tiles where those behaviors are under test. Sources must
cross multiple real tile borders with foreground on both sides. Comparing two
processing layouts alone is not scientific accuracy. Independent chunks have
their own initialization and stopping histories. Test real allocation fallback
with CUDA memory-pool limits, not fabricated out-of-memory exceptions.

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

Update producer and writer pins together. Hosted CI runs CPU tests only. Relevant
CUDA changes also need targeted GPU units/integrations on hardware with
`OPM_REQUIRE_GPU=1`; only unavailable hardware may skip. Report CPU/CUDA results
separately and honor any session restriction on reruns. Do not hide failures with
skip/xfail. Runtime/memory measurements and local investigation reports belong in
ignored `diagnostics/` or `.tmp/`, outside pytest and published documentation.
