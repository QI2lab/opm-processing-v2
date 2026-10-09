# Test review rules

Every test must have exactly one `unit` or `integration` marker. Put CPU tests
in `tests/cpu/` and CUDA/NVENC tests in `tests/gpu/`; collection adds the matching
hardware marker. Hardware requirements do not replace either category.

- Unit tests check an isolated numerical operator, geometry calculation, or
  state transition against independent expected values. Storage-component
  tests may use temporary files but remain unit tests.
- Unit tests use the shared TensorStore dataset mocks, including sliced reads,
  writes, dtype, chunks and completed futures. Real TensorStore datasets remain
  required for integration tests. Do not mock processing algorithms or replace
  pixel assertions with mock call counts.
- Integration tests define a simulated object with known intensity or physical
  geometry, write its input data to disk, execute the workflow with real image
  reads and writes, reopen its output, and compare the saved data with truth.
  File existence, mock calls, and successful return codes are insufficient.
- Simulated brightness differences between acquisition depths must use
  Beer–Lambert attenuation with an explicit optical path and known attenuation
  coefficients in the forward model. Derive expected correction gains from
  those coefficients independently of the fitted result.
- Smoke tests, API/argument-forwarding tests, regression tests, and benchmark-only tests are not
  permitted. A CLI may launch a disk-to-disk integration workflow, but the
  assertions must verify resulting data rather than its interface.
- Keep acquisition formats, camera calibration, optical objects, storage and
  workflow settings in the corresponding `tests/fixtures/` modules. Reuse
  factory fixtures for differing shapes, channels and scan modes. Do not import
  helpers from `conftest.py` or another test module.
- Keep independent forward models and measurements in `tests/reference/`.
  Expected values must describe known physics or state transitions, rather than
  snapshots of production output or historical bugs.
- Integration tests run real metadata inspection, reconstruction, registration,
  fusion and storage for the workflow being tested. Supply simulated PSFs on
  disk instead of replacing PSF generation or solvers. Fault injection may
  interrupt genuine writes or checkpoints to test recovery; it must not replace
  successful image calculations.
- Use function scope for mutable datasets and journals. Cache expensive optical
  fixtures at session scope when their inputs remain immutable. Seed each
  simulated noise realization explicitly, and parameterize independent factors
  separately. Small analytical examples and acceptance bounds belong beside
  their assertions; fixtures own repeated setup and configuration.

Collection checks enforce the categories and require temporary disk storage
for integration tests. Review must also verify simulated ground truth and the
complete read/process/write path; fixtures alone cannot prove those properties.

Use the deconvolution reference implementation:
the local `expansion-processing/src/expansion_processing/rlgc.py`, pinned in
[the processing methods](https://github.com/qi2lab/opm-processing-v2/blob/perf/numba-deskew-corrections/docs/methods/deconvolution.md).
Keep the RLGC multiplicative update and gradient-consensus rule numerically
consistent with that reference, accounting for documented OPM adaptations.
Changes must pass the same physical validation for fully sampled
and sub-sampled acquisitions; seeded pixel identity is required only for changes
intended to preserve arithmetic and random draws.
Use identical prescribed splits to check arithmetic and stopping independently;
also check recovery of the known ground object, absolute fluorescence, physical
localization and two-point profiles at known separations. Measure peak positions,
separation and valley depth in laboratory units. Establish resolution tolerances
from the pinned reference and stated measurement criterion.
Account explicitly for known camera and deskew intensity conventions and voxel
volumes; fitting a gain to the reconstructed result is not physical validation.
Normalized KLD is a stopping statistic, not an absolute Poisson likelihood.
Do not weaken tolerances or replace the optical truth to obtain a passing
optimization. Report reference/physics differences separately. Lookup sampling
and PSF tapering require distribution, object-recovery and two-point validation
before production acceptance. Measure runtime outside pytest; performance
scripts belong in scripts/ and generated reports in ignored diagnostics/.
