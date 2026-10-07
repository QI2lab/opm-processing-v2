# Running and adding tests

Run these commands from the repository root. They use PowerShell syntax.

## Set up and run

Install the development dependencies and run tests that do not require CUDA:

```powershell
uv sync --group dev --no-default-groups
uv run --no-sync python -m pytest --strict-markers -q -m "not gpu"
```

For a CUDA workstation, follow the [GPU installation instructions](../README.md),
install the GPU dependencies, and require working CUDA during testing:

```powershell
uv sync --group dev --extra gpu
$env:OPM_REQUIRE_GPU = "1"
uv run --no-sync python -m pytest --strict-markers -q
Remove-Item Env:OPM_REQUIRE_GPU
```

Without `OPM_REQUIRE_GPU=1`, unavailable CUDA causes GPU tests to skip. Check
the summary for skipped tests before treating a run as GPU validation.

## Run a smaller selection

```powershell
uv run --no-sync python -m pytest -q -m unit
uv run --no-sync python -m pytest -q -m integration
uv run --no-sync python -m pytest -q tests/test_depth_intensity_normalization.py
```

Add `-m gpu` to select tests requiring GPU hardware. Add `-x` to stop at the
first failure, or `-vv` to see each test case as it runs.

## Add a test

Put tests in `test_*.py` files and choose `@pytest.mark.unit` for an isolated
calculation or state transition, or `@pytest.mark.integration` for a simulated
object processed from disk input to verified disk output. Use `tmp_path` for
generated files so tests do not depend on local acquisitions.

For a concrete integration example, see
[the depth-normalization test](test_depth_intensity_normalization.py). It:

1. Simulates an object with known geometry and brightness.
2. Writes its camera pixels and acquisition metadata to disk.
3. Runs processing and fusion using those files.
4. Reopens the outputs and compares the saved data with expected values.

Choose tolerances from sampling, camera quantization, or another stated error
bound. Run the affected tests and `uv run --no-sync ruff check tests` before
submitting a change. Detailed review rules are in [CONTRIBUTING.md](CONTRIBUTING.md).
