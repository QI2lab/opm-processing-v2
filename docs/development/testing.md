# Running tests

Run the unit and integration suites from the repository root using an installed development environment:

```powershell
uv run --no-sync python -m pytest tests/cpu --strict-markers -q
$env:OPM_REQUIRE_GPU = "1"
uv run --no-sync python -m pytest tests/gpu --strict-markers -q
```

CPU tests run in GitHub Actions. GPU validation requires CUDA and
`OPM_REQUIRE_GPU=1`, so missing hardware fails the requested validation.
The existing GPU environment does not need to be synchronized to run tests.

Shared fixtures provide controlled metadata, file structures and pixel arrays.
Tests use production algorithms and installed packages. Synthetic acquisitions
share production optics through the forward model in `tests/simulation/`.
See [test contribution rules](https://github.com/qi2lab/opm-processing-v2/blob/main/tests/CONTRIBUTING.md)
for adding or changing cases.

Run repository checks before submitting:

```powershell
uv run --no-sync prek run --all-files
```
