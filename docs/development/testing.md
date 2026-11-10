# Running tests

Run the unit and integration suites from the repository root using an installed development environment:

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"
uv run --no-sync python -m pytest tests/cpu --strict-markers -q
$env:CUDA_VISIBLE_DEVICES = "0"
uv run --no-sync python -m pytest tests/gpu --strict-markers -q
```

GitHub Actions runs the selected CPU numerical core, three projection-writer
routes and resume/restart checks. BaSiC fitting tests run only in the GPU suite;
cached-field correction and metadata checks remain on CPU.
The manual Science validation workflow provides the larger science selections
and full CPU/GPU suites on a qualified runner. GPU tests skip when CuPy or CUDA
hardware is unavailable; numerical and allocation failures remain errors.
BaSiC fitting additionally requires PyTorch CUDA; missing hardware skips these
tests before expensive fixture setup.
The existing GPU environment does not need to be synchronized to run tests.

Numerical sample sweeps live in focused units. The 24 noisy RLGC cases cover
line pairs and spheres at three photon budgets, native and coarse scan spacing,
and whole 50-plane or tiled 200-plane acquisitions. Illumination profiles and
the two tiled core sizes are balanced across those cases. Eight noisy deskew
cases retain whole/tiled layouts, specimen types and scan spacings.

Five main writer-backed numerical integrations cover two large specimens, two
mosaics and one calibrated multi-timepoint, multi-channel reconstruction. They
exercise fixed and automatically planned chunks, whole and tiled RLGC, both scan
modes, fitted illumination, registration and fusion. Separate focused tests
cover stage placement and ROI export without repeating the numerical sample sweeps.
Storage and checkpoint checks use CPU processing; GPU command checks exercise
illumination, deconvolution or fusion, including registration. The manual GPU
science selection includes the BaSiC fitting tests, all 24
noisy RLGC units and the five main numerical integrations; the full suite also
runs the focused workflow checks.

Shared fixtures provide controlled metadata, file structures and pixel arrays.
Tests use production algorithms and installed packages. Synthetic acquisitions
share production optics through the forward model in `tests/simulation/`.
See [test contribution rules](https://github.com/qi2lab/opm-processing-v2/blob/main/tests/CONTRIBUTING.md)
for adding or changing cases.

Run repository checks before submitting:

```powershell
uv run --no-sync prek run --all-files
```
