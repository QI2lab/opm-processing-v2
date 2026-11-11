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
