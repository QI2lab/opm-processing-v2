# Test conventions

Only unit and integration tests are permitted. Do not add regression checks,
snapshot tests, or saved outputs from previous runs as expected results. This
rule applies to test behavior as well as names and markers.

Classify every case as either unit or integration. Unit tests isolate numerical
operators or decisions; integration tests check interacting processing and storage
components. Put CPU cases in `tests/cpu` and CUDA cases in `tests/gpu`; collection
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

Local investigation notes and generated reports belong in ignored `.tmp/` or
`diagnostics/`, not published documentation. Keep commands and general contributor
instructions concise; report the validation performed with the change itself.
