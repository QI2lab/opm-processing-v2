# Test review rules

Every test must have exactly one `unit` or `integration` marker. `gpu` records
a hardware requirement and does not replace either category.

- Unit tests check an isolated numerical operator, geometry calculation, or
  state transition against independent expected values. Storage-component
  tests may use temporary files but remain unit tests.
- Integration tests define a simulated object with known intensity or physical
  geometry, write its input data to disk, execute the workflow with real image
  reads and writes, reopen its output, and compare the saved data with truth.
  File existence, mock calls, and successful return codes are insufficient.
- Simulated brightness differences between acquisition depths must use
  Beer–Lambert attenuation with an explicit optical path and known attenuation
  coefficients in the forward model. Derive expected correction gains from
  those coefficients independently of the fitted result.
- Smoke tests, API/argument-forwarding tests, and benchmark-only tests are not
  permitted. A CLI may launch a disk-to-disk integration workflow, but the
  assertions must verify resulting data rather than its interface.

Collection checks enforce the categories and require temporary disk storage
for integration tests. Review must also verify simulated ground truth and the
complete read/process/write path; fixtures alone cannot prove those properties.

Use the user-designated deconvolution reference as the measured baseline:
the local `expansion-processing/src/expansion_processing/rlgc.py`, pinned in
[the processing methods](https://github.com/qi2lab/opm-processing-v2/blob/perf/numba-deskew-corrections/docs/methods/deconvolution.md). The former OPM loop
is not an interchangeable reference; it differed in initialization, boundaries,
fractional splitting, and stopping. Define the permitted OPM adaptations before
accepting an optimization as reference-equivalent.
The current user authorization permits changes to PSF generation, stopping,
initialization, sampling and execution strategy. Keep the RLGC multiplicative
update and gradient-consensus rule recognizable and numerically audited against
the reference. Changes must pass the same physical validation for fully sampled
and sub-sampled acquisitions; seeded pixel identity is required only for changes
intended to preserve arithmetic and random draws.
Use identical prescribed splits to check arithmetic and stopping independently;
also check recovery of the known ground object, absolute fluorescence, physical
localization and two-point profiles at known separations. Measure peak positions,
separation and valley depth in laboratory units. Establish resolution tolerances
from the pinned reference and stated measurement criterion; do not invent a
prediction-fit cutoff and use its failure to claim RLGC does not recover objects.
Account explicitly for known camera and deskew intensity conventions and voxel
volumes; fitting a gain to the reconstructed result is not physical validation.
Normalized KLD is a stopping statistic, not an absolute Poisson likelihood.
Do not weaken tolerances or replace the optical truth to obtain a passing
optimization. Report reference/physics differences separately. Lookup sampling
and PSF tapering require distribution, object-recovery and two-point validation
before production acceptance. Measure runtime outside pytest; performance
scripts belong in scripts/ and generated reports in ignored diagnostics/.
