# Test conventions

Only unit and integration tests are permitted. Do not add regression checks,
snapshot tests, or saved outputs from previous runs as expected results. This
rule applies to test behavior as well as names and markers.

Classify every case as either unit or integration. Unit tests isolate numerical
operators, decisions, serialization, file formats or GUI coordinate handling.
Integration tests cover the complete simulated acquisition to final datastore.
Put CPU cases in `tests/cpu` and CUDA cases in `tests/gpu`; collection
supplies the hardware marker. Do not add smoke, benchmark or argument-forwarding
cases under another label.

Use shared fixtures in `tests/fixtures` for setup, input file structures,
acquisition metadata, optical objects, calibration and real package-backed datasets.
Keep mutable data function-scoped; cache expensive immutable optical models at
session scope. Parameterize independent settings rather than copying
tests. Do not import helpers from another test module or `conftest.py`.
Integration tests must consume shared forward-model ground truth, then process
the simulated observations through production code. CLI integrations invoke the
registered applications and their actual parsers; numerical processing, storage,
and encoders must execute. Collection requires a shared simulation fixture.
Reuse an existing builder before adding a fixture. Avoid tiny wrappers that only
rename a call. Use NumPy-style docstrings; Ruff checks test functions and fixtures
as well as library code.

Every integration must execute this chain:

1. Construct a known fluorescent object in physical coordinates.
2. Simulate microscope optics, illumination, camera-pixel integration, photon
   statistics and detector digitization with the shared forward model.
3. Deliver those camera frames through the installed
   `opm_v2.handlers.opm_data_handler.OpmDataHandler` lifecycle and its real
   ome-writers TensorStore backend, using `opm_v2_acquisition_factory`.
4. Read the actual written acquisition metadata, generate production PSFs, and
   execute the actual processing functions or registered CLI. CUDA deconvolution
   integrations must generate and consume their kernels inside processing.
5. Reopen the final datastore and check pixels, coordinates, physical accuracy
   and durable completion against the simulated truth and specified expectations.

CPU integrations generate the production optical kernels from the written
metadata and exercise calibration, deskew, storage and any requested downstream
operations. CUDA integrations additionally exercise RLGC with internally generated
PSFs. Expensive immutable optical kernels may be cached only by their complete
resolved physical settings. Kernels supplied to CUDA processing are never replaced
with cached test outputs or PSF files.

Hand-built lookalike metadata, direct writes with the processed-output writer,
injected `opm_processing.deconvolution` attributes and measured PSF options cannot
replace stages of an integration. Explicit optical calibration and measured-PSF
loading belong in isolated unit checks. Include single-position acquisitions,
combined channels and multi-position collections. A check stopping at metadata,
PSF generation, GUI selection or format serialization is a unit, not a complete
integration. Keep its assertions; do not count it as full-pipeline coverage.

Collection requires the actual producer and physical simulation fixtures.
`integration_pipeline_records` observes successful execution of the real
processing functions and verifies that every producer source has complete,
readable final outputs for all selected tiles, and that deconvolution used
generated optical parameters.

Command integrations must also cover the acquisition's filesystem lifecycle.
Move or copy real writer outputs before resuming processing, reusing fusion
registration, regenerating projections, exporting ROIs/TIFFs, inspecting data,
and encoding movies. Include copies whose original remains accessible, separate
raw/output directories, and nested ROI outputs. Preserve the source metadata and
journals during relocation; reopen the final data at its current location and
check pixels, physical metadata, and completion. Use the shared relocation
fixture so the complete-pipeline guard tracks actual producer files through
moves and copies. Extend these acquisition workflows instead of adding isolated
tests that reproduce a past exception.

The dev group supplies the writer's Python runtime dependencies, pinned to the
versions used by opm-v2. Install the actual producer package without its unrelated
microscope-control dependencies before running these integrations (CI does this
too):

```bash
uv sync --group dev
uv pip install --no-deps "opm-v2 @ git+https://github.com/QI2lab/opm-v2@ae059d962835520bf2c1b37edbb335bf5e617b23"
```

Update the producer and writer pins together when changing the acquisition
contract. Missing producer dependencies are errors, not reasons to skip tests.

Use library entry points and installed packages. Persist OME-Zarr fixtures with
yaozarrs and its TensorStore writer; use real TensorStore arrays for in-memory
numerical inputs. Native storage uses pytest temporary directories, since Python
filesystem mocks cannot intercept TensorStore's native file I/O. Do not replace
package APIs with custom arrays, futures, codecs, storage adapters or optimizers.
Read image/group metadata and open on-disk arrays through yaozarrs. TensorStore
provides native block writes. Zarr-python is permitted for attribute edits and
through napari-ome-zarr for viewing; pixel processing and storage must use
yaozarrs or TensorStore. Reuse the production metadata updater when constructing
opm-v2-shaped acquisition fixtures; tests must not require an experimental
dataset or an external data drive.
Observers must execute the real operation. Fault injection belongs only in
explicit error-handling tests; every successful numerical path must execute the
real implementation. Integration tests must retain production storage and worker
processes. Assert pixels, coordinates or state transitions against independent
expectations. Keep performance measurements outside pytest.

Simulation and filesystem mocking are the only test-specific implementations.
Every numerical test must use a known analytical object or a known blurred
ground-truth scene. Decision and metadata units use explicitly prescribed state,
dimensions and physical units. Do not fabricate package results, futures or
warnings; exercise actual installed-package behavior. A tiled/untiled comparison
alone cannot establish accuracy. Verify numerical operators against analytical
truth or installed independent operators, and reconstruction against the known
camera means. Scenes must cross multiple actual tile borders, including partial
last tiles, with foreground on both sides of internal borders.
Use production algorithms and installed packages for processing and numerical
assertions; do not maintain alternative deskew, PSF or reconstruction algorithms.
The single camera forward model belongs in `tests/simulation`. Simulation and
deconvolution share the production optical PSF and pixel integration. Render fine
objects, multiply excitation before emission blur, and digitize Poisson
photoelectrons plus calibrated read noise into unsigned ADC codes. Use declared
exposures and seeded noise, with no fitted reconstruction gain.

Reconstruction and registration accuracy use absolute photon/camera likelihood,
with independent camera realizations and fixed physical source/background
regions. Likelihood is an accuracy metric, not a held-out stopping rule.
Resolution uses direction-specific PSF first minima (Rayleigh). Numerical
operator tolerances must follow precision, sampling or a stated error bound.
Geometry and photon-volume checks use the supported 30-degree OPM convention.
Do not weaken physics assertions, substitute production output for truth, or
hide an accuracy failure with skip/xfail. Only unavailable hardware may skip.

Standard processing integrations must enforce the execution contract as well as
image accuracy: raw camera data, production PSFs, sequential haloed scan-chunk
deconvolution, deskewing, registration, fusion, and reopened final storage. Observe
the real numerical calls to check that state is bounded by chunk size and PSF
support, including scan refinement and partial last chunks. Unit checks may use
an actual CUDA memory-pool limit to exercise allocation fallback; do not fabricate
out-of-memory errors or replace numerical work. Independent chunks have separate
initialization and stopping histories: judge each result against known optical
truth, not saved outputs or another processing layout. Memory timing and peak-RSS
measurements remain standalone diagnostics outside pytest.

The hosted CI job runs the CPU suite only. CUDA processing changes also require
the relevant GPU unit and integration suites on CUDA hardware with
`OPM_REQUIRE_GPU=1`, so a missing runtime is
an error rather than a skipped validation. Report CPU and CUDA results separately;
a green CPU job does not establish deconvolution correctness.

Local investigation notes and generated reports belong in ignored `.tmp/` or
`diagnostics/`, not published documentation. Keep commands and general contributor
instructions concise; report the validation performed with the change itself.
