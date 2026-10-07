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
