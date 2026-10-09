# Running and adding tests

Run these commands from the repository root. They use PowerShell syntax.

## Set up and run

Install the development dependencies and run tests that do not require CUDA:

```powershell
uv sync --group dev --no-default-groups
uv run --no-sync python -m pytest tests/cpu --strict-markers -q
```

For a CUDA workstation, follow the [GPU installation instructions](../getting-started/installation.md),
install the GPU dependencies, and require working CUDA during testing:

```powershell
uv sync --group dev --extra gpu
$env:OPM_REQUIRE_GPU = "1"
uv run --no-sync python -m pytest tests/gpu --strict-markers -q
Remove-Item Env:OPM_REQUIRE_GPU
```

Without `OPM_REQUIRE_GPU=1`, unavailable CUDA causes GPU tests to skip. Check
the summary for skipped tests before treating a run as GPU validation.

## Run a smaller selection

```powershell
uv run --no-sync python -m pytest -q -m unit
uv run --no-sync python -m pytest -q -m integration
uv run --no-sync python -m pytest -q tests/cpu/test_depth_intensity_normalization.py
```

The CPU directory is the GitHub Actions suite and always uses the real CPU
registration backend. The GPU directory requires CUDA or NVENC. Running
`python -m pytest` collects both; `-m cpu` and `-m gpu` select the same groups.
Add `-x` to stop at the first failure, or `-vv` to see each test case as it runs.

## Add a test

Put tests in `test_*.py` files and choose `@pytest.mark.unit` for an isolated
calculation or state transition, or `@pytest.mark.integration` for a simulated
object processed from disk input to verified disk output. Use `tmp_path` for
generated files so tests do not depend on local acquisitions.

Use the shared factories in `tests/fixtures/`; `conftest.py` loads them through
`pytest_plugins` and enforces test categories. Keep setup with the logic it models:

| Fixture module | Responsibility |
| --- | --- |
| `acquisition` | Upstream scan layouts, camera metadata and processing options |
| `camera`, `illumination` | Detector transfer and known sensitivity fields |
| `optics`, `tiling` | Physical objects, PSFs, placement errors and acceptance settings |
| `tensorstore`, `storage`, `export` | Unit dataset mocks, saved NGFF images and OME provenance |
| `live`, `roi`, `state` | Acquisition lifecycle, physical regions and durable checkpoints |
| `fusion`, `hardware` | Isolated numerical operators and execution requirements |

Independent optical models and measurements belong in `tests/reference/`.
Reuse a factory and override the settings that define the case. Keep expected
physical values and tolerances visible in the test. Integration tests must
reopen saved pixels; simulated return values and call counts are insufficient.
Unit tests use the shared TensorStore doubles with functional pixel reads and
writes. Integration tests retain real TensorStore datasets on disk.

For a concrete integration example, see
[the depth-normalization test](https://github.com/qi2lab/opm-processing-v2/blob/perf/numba-deskew-corrections/tests/cpu/test_depth_intensity_normalization.py). It:

1. Simulates an object with known geometry and brightness.
2. Writes its camera pixels and acquisition metadata to disk.
3. Runs processing and fusion using those files.
4. Reopens the outputs and compares the saved data with expected values.

Choose tolerances from sampling, camera quantization, or another stated error
bound. Run the affected tests and `uv run --no-sync prek run --all-files` before
submitting a change. The review rules below also apply to numerical optimizations.

--8<-- "tests/CONTRIBUTING.md"
