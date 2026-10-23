# Running tests

```powershell
uv run --no-sync python -m pytest tests/cpu --strict-markers -q
uv run --no-sync python -m pytest tests/gpu --strict-markers -q
```

Use the existing development environment. CUDA is required for GPU cases.
See [contribution rules](CONTRIBUTING.md) and the
[testing guide](../docs/development/testing.md).

Simulation setup and the single forward model live in `tests/simulation`.
Simulation and deconvolution share production PSF generation and pixel integration.
Shared configurations live in `tests/fixtures/configs`.
To render the saved synthetic comparison outputs without reconstructing:

```powershell
uv run --no-sync python -m tests.deconvolution_figures --config tests/fixtures/configs/deconvolution_figures.json --output diagnostics/rlgc_final --render-only
```

Omit `--render-only` to generate a new comparison series in a new output directory.
