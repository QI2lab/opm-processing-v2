# Running tests

Run the unit and integration suites from the repository root using an installed development environment:

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"
uv run --no-sync python -m pytest tests/cpu --strict-markers -q
$env:CUDA_VISIBLE_DEVICES = "0"
uv run --no-sync python -m pytest tests/gpu --strict-markers -q
```

GitHub Actions runs the selected CPU numerical core, three projection-writer
routes and resume/restart checks; Linux also runs one real illumination fit.
The manual Science validation workflow provides the larger science selections
and full CPU/GPU suites on a qualified runner. GPU tests skip when CuPy or CUDA
hardware is unavailable; numerical and allocation failures remain errors.
The existing GPU environment does not need to be synchronized to run tests.

The full suites retain all 72 noisy RLGC numerical cases, eight noisy deskew
cases and four stage-placement cases. Writer integrations use 16 large-scene
cases, eight mosaics, three calibration/reconstruction pipelines and two
relocation/export workflows. These cover both scan modes with representative
combinations while the numerical cases retain the full scientific matrix.

Shared fixtures provide controlled metadata, file structures and pixel arrays.
Tests use production algorithms and installed packages. Synthetic acquisitions
share production optics through the forward model in `tests/simulation/`.
See [test contribution rules](https://github.com/qi2lab/opm-processing-v2/blob/main/tests/CONTRIBUTING.md)
for adding or changing cases.

Run repository checks before submitting:

```powershell
uv run --no-sync prek run --all-files
```
