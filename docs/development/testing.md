# Running tests

Run the unit and integration suites from the repository root using an installed development environment:

```powershell
uv run --no-sync python -m pytest tests/cpu --strict-markers -q
uv run --no-sync python -m pytest tests/gpu --strict-markers -q
```

CPU tests run in GitHub Actions. GPU tests require CUDA; inspect the pytest summary for unavailable hardware.
The existing GPU environment does not need to be synchronized to run tests.

Shared fixtures provide controlled metadata, file structures and pixel arrays.
Tests use production algorithms and installed packages. Synthetic acquisitions
share production optics through the forward model in `tests/simulation/`.
See [test contribution rules](https://github.com/qi2lab/opm-processing-v2/blob/perf/numba-deskew-corrections/tests/CONTRIBUTING.md)
for adding or changing cases.

Run repository checks before submitting:

```powershell
uv run --no-sync prek run --all-files
```
