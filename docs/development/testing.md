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

Each behavior has one primary test family; integrations check the interaction
of those components against the same original physical truth.

| Behavior | Primary owner |
| --- | --- |
| Reconstruction recovery and noise | GPU `test_rlgc_physics.py::test_tiled_gc_recovers_noisy_extended_objects_without_rl_noise` (24 cases) |
| RL updates, consensus, KL and continuation | Focused GPU RLGC units; scalar FFT/KL operators in `test_gpu_processing.py` |
| Count splitting | GPU `test_photon_splitting.py` |
| Detector response and allocation fallback | GPU `test_rlgc_detector_response.py` units |
| Deskew interpolation and boundaries | CPU `test_opmtools.py`, including eight noisy physical cases |
| Illumination estimation and sample selection | GPU `test_illumination.py` |
| Pair correlation, aliases and retries | GPU registration units; CPU registration units own geometry and graph solving |
| Metadata, time, ROI selection and checkpoint decisions | Focused CPU units; corresponding workflows own persistence and recovery |

Five main writer-backed integrations exercise representative processing routes:

| Route | Time × position × channel | Reconstruction |
| --- | --- | --- |
| Native stage with fitted illumination | 2 × 1 × 2 | Whole |
| Native mirror with lines and a known placement error | 1 × 2 × 1 | Explicit scan chunks |
| Sparse stage with a sphere | 1 × 1 × 1 | Whole |
| Native stage adjacent mosaic | 1 × 2 × 1 | Disabled |
| Sparse stage adjacent mosaic | 1 × 2 × 1 | Automatically planned chunks |

The calibrated route owns time/channel reconstruction routing. The native tiled
pair owns placement-error recovery and full-volume coincident blending. The
adjacent mosaics own overlap and block-boundary behavior. Focused stage placement,
combined-channel and cropped-reconstruction cases retain their distinct geometry
and optical metadata contracts.
Storage and checkpoint checks use CPU processing; GPU command checks exercise
illumination, deconvolution or fusion, including registration. The manual GPU
science selection includes the BaSiC fitting tests, all 24
noisy RLGC units and the five main numerical integrations; the full suite also
runs the focused workflow checks.

Shared fixtures provide controlled metadata, file structures and pixel arrays.
Tests use production algorithms and installed packages. Synthetic acquisitions
share production optics through the forward model in `tests/simulation/`.
Bounded session caches retain only immutable optical expectations and fixed camera
realizations. Writable acquisitions, outputs, checkpoints and observers remain
function-scoped. Small-array thread limits apply only to analytical unit fixtures.
See [test contribution rules](https://github.com/qi2lab/opm-processing-v2/blob/main/tests/CONTRIBUTING.md)
for adding or changing cases.

Run repository checks before submitting:

```powershell
uv run --no-sync prek run --all-files
```
