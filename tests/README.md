# Tests

Install pytest into the project environment, then run correctness tests:

```powershell
uv pip install --python .venv/Scripts/python.exe pytest
uv run --no-sync python -m pytest --strict-markers -q
```

Use `-m unit`, `-m integration`, or `-m "not gpu"` to select tests.
Set `OPM_REQUIRE_GPU=1` to require working CUDA for GPU tests.

Performance benchmarks are optional:

```powershell
uv run --no-sync python -m pytest tests/benchmarks --run-benchmarks -s
```

Benchmarks compare warmed-up medians against `main` at 4 and 16 threads.
Use `-k small`, `-k large`, or `-k chunked`; chunked requires about 90 GiB free RAM.
See [workloads and timing results](../docs/deskew_optimization.md).
